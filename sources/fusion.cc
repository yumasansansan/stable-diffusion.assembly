// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

#include "fusion.h"

#include <cmath>
#include <cstdint>

#include "kernels.h"
#include "scalar_math.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

namespace {

bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const auto * a0 = static_cast<const char *>(a->data);
    const auto * b0 = static_cast<const char *>(b->data);
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

bool same_layout(const ggml_tensor * a, const ggml_tensor * b) {
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (a->ne[k] != b->ne[k] || a->nb[k] != b->nb[k]) {
            return false;
        }
    }
    return a->data == b->data && a->type == b->type;
}

void silu_run(float * out, std::int64_t n) {
    for (std::int64_t k = 0; k < n; ++k) {
        out[k] = silu(out[k]);
    }
}

void binary_run(float * out, std::int64_t n, const float * operand, bool broadcast, Epilogue_step::Kind kind) {
    if (broadcast) {
        const float s = *operand;
        if (kind == Epilogue_step::Kind::add) {
            for (std::int64_t k = 0; k < n; ++k) {
                out[k] = out[k] + s;
            }
        }
        else {
            for (std::int64_t k = 0; k < n; ++k) {
                out[k] = out[k] * s;
            }
        }
        return;
    }
    if (kind == Epilogue_step::Kind::add) {
        for (std::int64_t k = 0; k < n; ++k) {
            out[k] = out[k] + operand[k];
        }
    }
    else {
        for (std::int64_t k = 0; k < n; ++k) {
            out[k] = out[k] * operand[k];
        }
    }
}

}  // namespace

void Epilogue::apply(float * out, std::int64_t n, std::int64_t i0, std::int64_t i1, std::int64_t i2, std::int64_t i3) const {
    for (int s = 0; s < count; ++s) {
        const Epilogue_step & step = steps[s];
        if (step.kind == Epilogue_step::Kind::silu) {
            silu_run(out, n);
            continue;
        }
        const char * p = step.data + i0 * step.nb[0] + i1 * step.nb[1] + i2 * step.nb[2] + i3 * step.nb[3];
        binary_run(out, n, reinterpret_cast<const float *>(p), step.nb[0] == 0, step.kind);
    }
}

void Epilogue::apply_positions(float * out, std::int64_t n, std::int64_t position, std::int64_t i2, std::int64_t i3) const {
    for (int s = 0; s < count; ++s) {
        const Epilogue_step & step = steps[s];
        if (step.kind == Epilogue_step::Kind::silu) {
            silu_run(out, n);
            continue;
        }
        // Broadcast along the rows of the image, or laid out as the output:
        // the position then steps by the element (match_chain() checks it).
        const char * p = step.data + position * step.nb[0] + i2 * step.nb[2] + i3 * step.nb[3];
        binary_run(out, n, reinterpret_cast<const float *>(p), step.nb[0] == 0, step.kind);
    }
}

namespace {

// Whether node `from` or any after it reads one of `tensors`, or a view of one.
bool read_after(ggml_cgraph * g, int from, const ggml_tensor * const * tensors, int count);

}  // namespace

Epilogue match_chain(const ggml_cgraph * graph, int index, bool silu, bool positions, bool redirect) {
    Epilogue e;
    auto * g = const_cast<ggml_cgraph *>(graph);
    const int n_nodes = ggml_graph_n_nodes(g);
    const ggml_tensor * head = ggml_graph_node(g, index);
    e.last_node = index;
    e.ne0 = head->ne[0];
    if (head->type != GGML_TYPE_F32 || head->nb[0] != sizeof(float)) {
        return e;
    }
    int j = index + 1;
    for (; j < n_nodes && e.count < Epilogue::kMaxSteps; ++j) {
        const ggml_tensor * node = ggml_graph_node(g, j);
        if (ggml_op_is_empty(node->op)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !same_layout(node, head)) {
            break;
        }
        Epilogue_step step;
        if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU && silu &&
            node->src[0] != nullptr && node->src[0]->data == head->data) {
            step.kind = Epilogue_step::Kind::silu;
        }
        else if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
            const ggml_tensor * a = node->src[0];
            const ggml_tensor * b = node->src[1];
            if (a == nullptr || b == nullptr) {
                break;
            }
            // The chain goes on through src0, which ggml's in-place operations
            // take as the tensor they change, the other operand added or
            // multiplied: a + b and a * b, in that order, as ggml has them.
            if (a->data != head->data || !ggml_are_same_shape(a, head) || overlaps(b, head) ||
                b->type != GGML_TYPE_F32 || b->nb[0] != sizeof(float)) {
                break;
            }
            bool fits = true;
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                if (b->ne[k] != head->ne[k] && b->ne[k] != 1) {
                    fits = false;
                }
                step.nb[k] = b->ne[k] == 1 ? 0 : static_cast<std::int64_t>(b->nb[k]);
            }
            if (positions) {
                // The positions of an image run on across its rows: the operand
                // is one number for them all, or lies in rows as the output.
                const bool broadcast = step.nb[0] == 0 && step.nb[1] == 0;
                const bool rows = step.nb[0] == static_cast<std::int64_t>(sizeof(float)) &&
                                  step.nb[1] == head->ne[0] * static_cast<std::int64_t>(sizeof(float));
                fits = fits && (broadcast || rows);
            }
            if (!fits) {
                break;
            }
            step.kind = node->op == GGML_OP_ADD ? Epilogue_step::Kind::add : Epilogue_step::Kind::mul;
            step.data = static_cast<const char *>(b->data);
            e.operands[e.count] = b;
        }
        else {
            break;
        }
        e.steps[e.count++] = step;
        e.last_node = j;
    }
    // The residual connection: an addition, not in place, of the chain's
    // tensor and another of its shape and layout, into a third of the same.
    // The views between are allowed; any other node ends the chain where it is.
    for (; redirect && j < n_nodes && ggml_op_is_empty(ggml_graph_node(g, j)->op); ++j) {
    }
    if (!redirect || j >= n_nodes || e.count >= Epilogue::kMaxSteps) {
        return e;
    }
    ggml_tensor * node = ggml_graph_node(g, j);
    const auto alike = [&](const ggml_tensor * t) {
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            return false;
        }
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            if (t->ne[k] != head->ne[k] || t->nb[k] != head->nb[k]) {
                return false;
            }
        }
        return true;
    };
    if (node->op != GGML_OP_ADD || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !alike(node) ||
        node->data == head->data) {
        return e;
    }
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    const ggml_tensor * other = a != nullptr && a->data == head->data && alike(a) ? b
                                : b != nullptr && b->data == head->data && alike(b) ? a
                                                                                     : nullptr;
    if (!alike(other) || overlaps(other, head)) {
        return e;
    }
    // Nothing after reads the operation's tensor, or the chain's.
    const ggml_tensor * unwritten[Epilogue::kMaxSteps + 8];
    int count = 0;
    for (int i = index; i <= e.last_node && count < Epilogue::kMaxSteps + 8; ++i) {
        const ggml_tensor * t = ggml_graph_node(g, i);
        if (t->data != head->data) {
            continue;
        }
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return e;
        }
        unwritten[count++] = t;
    }
    if (read_after(g, j + 1, unwritten, count)) {
        return e;
    }
    if (other->data == node->data) {
        // The sum in the memory of the other operand: begun from it, which
        // only additions before may be reordered with.
        for (int k = 0; k < e.count; ++k) {
            if (e.steps[k].kind != Epilogue_step::Kind::add) {
                return e;
            }
        }
        e.from_redirect = true;
        e.last_node = j;
        e.redirect = node;
        return e;
    }
    Epilogue_step step;
    step.kind = Epilogue_step::Kind::add;
    step.data = static_cast<const char *>(other->data);
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        step.nb[k] = static_cast<std::int64_t>(other->nb[k]);
    }
    e.operands[e.count] = other;
    e.steps[e.count++] = step;
    e.last_node = j;
    e.redirect = node;
    return e;
}

bool redirect_apart(const ggml_tensor * node, const Epilogue & chain) {
    if (chain.redirect == nullptr) {
        return true;
    }
    const ggml_tensor * out = chain.redirect;
    for (const ggml_tensor * s : node->src) {
        if (s != nullptr && s->data != nullptr && overlaps(out, s)) {
            return false;
        }
    }
    for (int k = 0; k < chain.count; ++k) {
        if (chain.operands[k] != nullptr && overlaps(out, chain.operands[k])) {
            return false;
        }
    }
    return true;
}

void gated_gelu_run(float * out, const float * x, const float * gate, std::int64_t n) {
    // The kernels written by hand take the numbers in groups of sixty-four
    // (AVX-512) or sixteen (AVX2); the C++ loop what is left, and all of them
    // where those are not taken.
    std::int64_t k = 0;
#if SDASM_X86_64
    if (avx512_kernels()) {
        k = n / 64 * 64;
        sdasm_gated_gelu_avx512(out, x, gate, k);
    }
    else if (avx2_kernels()) {
        k = n / 16 * 16;
        sdasm_gated_gelu_avx2(out, x, gate, k);
    }
#endif
    for (; k < n; ++k) {
        out[k] = x[k] * gelu(gate[k]);
    }
}

namespace {

bool read_after(ggml_cgraph * g, int from, const ggml_tensor * const * tensors, int count) {
    const int n_nodes = ggml_graph_n_nodes(g);
    for (int i = from; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(g, i);
        for (const ggml_tensor * s : node->src) {
            for (const ggml_tensor * t = s; t != nullptr; t = t->view_src) {
                for (int k = 0; k < count; ++k) {
                    if (t == tensors[k]) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// A view of `whole` that is its columns [begin, begin + width), every row of
// it as the whole has them.
bool columns_of(const ggml_tensor * view, const ggml_tensor * whole, std::int64_t begin, std::int64_t width) {
    if (view->type != GGML_TYPE_F32 || view->ne[0] != width || view->nb[0] != sizeof(float) ||
        static_cast<const char *>(view->data) != static_cast<const char *>(whole->data) + begin * static_cast<std::int64_t>(sizeof(float))) {
        return false;
    }
    for (int k = 1; k < GGML_MAX_DIMS; ++k) {
        if (view->ne[k] != whole->ne[k] || view->nb[k] != whole->nb[k]) {
            return false;
        }
    }
    return true;
}

}  // namespace

Gated_gelu match_gated_gelu(const ggml_cgraph * graph, int index, const Epilogue & chain) {
    Gated_gelu r;
    auto * g = const_cast<ggml_cgraph *>(graph);
    const int n_nodes = ggml_graph_n_nodes(g);
    const int j = chain.last_node + 1;
    if (j + 5 > n_nodes) {
        return r;
    }
    ggml_tensor * head = ggml_graph_node(g, index);
    ggml_tensor * v0 = ggml_graph_node(g, j);
    ggml_tensor * v1 = ggml_graph_node(g, j + 1);
    ggml_tensor * cont = ggml_graph_node(g, j + 2);
    ggml_tensor * gelu = ggml_graph_node(g, j + 3);
    ggml_tensor * mul = ggml_graph_node(g, j + 4);
    if (head->type != GGML_TYPE_F32 || !ggml_is_contiguous(head) || head->ne[0] % 2 != 0 || v0->op != GGML_OP_VIEW ||
        v1->op != GGML_OP_VIEW || cont->op != GGML_OP_CONT || gelu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(gelu) != GGML_UNARY_OP_GELU || mul->op != GGML_OP_MUL) {
        return r;
    }
    ggml_tensor * gate = cont->src[0];
    ggml_tensor * x = gate == v0 ? v1 : gate == v1 ? v0 : nullptr;
    const std::int64_t half = head->ne[0] / 2;
    if (x == nullptr || !columns_of(x, head, 0, half) || !columns_of(gate, head, half, half) ||
        gelu->src[0] != cont || gelu->data != cont->data || mul->src[0] != x || mul->src[1] != gelu ||
        mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mul, x)) {
        return r;
    }
    // Nothing after reads what is not written: the product, the chain's nodes,
    // the halves, the gate and its GELU; and none is an output of the graph.
    const ggml_tensor * unwritten[Epilogue::kMaxSteps + 6];
    int count = 0;
    for (int i = index; i <= chain.last_node; ++i) {
        unwritten[count++] = ggml_graph_node(g, i);
    }
    for (const ggml_tensor * t : {static_cast<const ggml_tensor *>(v0), static_cast<const ggml_tensor *>(v1),
                                  static_cast<const ggml_tensor *>(cont), static_cast<const ggml_tensor *>(gelu)}) {
        unwritten[count++] = t;
    }
    for (int k = 0; k < count; ++k) {
        if ((unwritten[k]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return r;
        }
    }
    if (read_after(g, j + 5, unwritten, count)) {
        return r;
    }
    r.out = mul;
    r.last_node = j + 4;
    return r;
}

}  // namespace sdasm
