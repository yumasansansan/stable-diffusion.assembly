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
#include <span>

#include "kernels.h"
#include "scalar_math.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

namespace {

bool tensors_overlap(const ggml_tensor * const a, const ggml_tensor * const b) {
    return overlaps(a->data, ggml_nbytes(a), b->data, ggml_nbytes(b));
}

bool same_layout(const ggml_tensor * const a, const ggml_tensor * const b) {
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (extent(a, k) != extent(b, k) || stride_bytes(a, k) != stride_bytes(b, k)) {
            return false;
        }
    }
    return a->data == b->data && a->type == b->type;
}

void silu_run(Elements<float> out) {
    for (float & x : out) {
        x = silu(x);
    }
}

// out = out + operand, or out * operand, element by element; or by the one
// number operand[0], when it is broadcast.
void binary_run(Elements<float> out, Elements<const float> operand, bool broadcast, Epilogue_step::Kind kind) {
    const std::int64_t n = out.size();
    if (broadcast) {
        const float s = operand[0];
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
    const Elements<const float> o = operand.first(n);
    if (kind == Epilogue_step::Kind::add) {
        for (std::int64_t k = 0; k < n; ++k) {
            out[k] = out[k] + o[k];
        }
    }
    else {
        for (std::int64_t k = 0; k < n; ++k) {
            out[k] = out[k] * o[k];
        }
    }
}

}  // namespace

void Epilogue::apply(Elements<float> out, std::int64_t i0, std::int64_t i1, std::int64_t i2, std::int64_t i3) const {
    for (const Epilogue_step & step : std::span(steps).first(static_cast<std::size_t>(count))) {
        if (step.kind == Epilogue_step::Kind::silu) {
            silu_run(out);
            continue;
        }
        const std::int64_t at = i0 * step.nb[0] + i1 * step.nb[1] + i2 * step.nb[2] + i3 * step.nb[3];
        binary_run(out, step.operand.from(at), step.nb[0] == 0, step.kind);
    }
}

void Epilogue::apply_positions(Elements<float> out, std::int64_t position, std::int64_t i2, std::int64_t i3) const {
    for (const Epilogue_step & step : std::span(steps).first(static_cast<std::size_t>(count))) {
        if (step.kind == Epilogue_step::Kind::silu) {
            silu_run(out);
            continue;
        }
        // Broadcast along the rows of the image, or laid out as the output:
        // the position then steps by the element (match_chain() checks it).
        const std::int64_t at = position * step.nb[0] + i2 * step.nb[2] + i3 * step.nb[3];
        binary_run(out, step.operand.from(at), step.nb[0] == 0, step.kind);
    }
}

Graph_nodes graph_nodes(ggml_tensor * const * const nodes, int n_nodes) {
    // The array of the graph's nodes, which ggml hands over with their number.
#pragma clang unsafe_buffer_usage begin
    return {nodes, static_cast<std::size_t>(n_nodes)};
#pragma clang unsafe_buffer_usage end
}

namespace {

// Whether node `from` or any after it reads one of `tensors`, or a view of one.
bool read_after(Graph_nodes nodes, int from, std::span<const ggml_tensor * const> tensors);

// The tensors of a chain that nothing after is to read.
constexpr std::size_t kMaxUnwritten = Epilogue::kMaxSteps + 8;
struct Unwritten {
    std::array<const ggml_tensor *, kMaxUnwritten> tensors{};
    std::size_t count = 0;

    void add(const ggml_tensor * const t) {
        SDASM_REQUIRE(count < tensors.size());
        tensors[count] = t;
        ++count;
    }
    std::span<const ggml_tensor * const> all() const { return std::span(tensors).first(count); }
};

void add_step(Epilogue & e, const Epilogue_step & step, const ggml_tensor * const operand) {
    const auto i = static_cast<std::size_t>(e.count);
    e.steps[i] = step;
    e.operands[i] = operand;
    ++e.count;
}

}  // namespace

Epilogue match_chain(Graph_nodes nodes, int index, bool silu, bool positions, bool redirect) {
    Epilogue e;
    const auto n_nodes = static_cast<int>(nodes.size());
    const ggml_tensor * const head = node_at(nodes, index);
    e.last_node = index;
    e.ne0 = head->ne[0];
    if (!holds<float>(head) || head->nb[0] != sizeof(float)) {
        return e;
    }
    int j = index + 1;
    for (; j < n_nodes && e.count < Epilogue::kMaxSteps; ++j) {
        const ggml_tensor * const node = node_at(nodes, j);
        if (ggml_op_is_empty(node->op)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !same_layout(node, head)) {
            break;
        }
        Epilogue_step step;
        const ggml_tensor * operand = nullptr;
        if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU && silu &&
            node->src[0] != nullptr && node->src[0]->data == head->data) {
            step.kind = Epilogue_step::Kind::silu;
        }
        else if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
            const ggml_tensor * const a = node->src[0];
            const ggml_tensor * const b = node->src[1];
            if (a == nullptr || b == nullptr) {
                break;
            }
            // The chain goes on through src0, which ggml's in-place operations
            // take as the tensor they change, the other operand added or
            // multiplied: a + b and a * b, in that order, as ggml has them.
            if (a->data != head->data || !ggml_are_same_shape(a, head) || tensors_overlap(b, head) || !holds<float>(b) ||
                b->nb[0] != sizeof(float)) {
                break;
            }
            bool fits = true;
            for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                if (extent(b, k) != extent(head, k) && extent(b, k) != 1) {
                    fits = false;
                }
                step.nb[static_cast<std::size_t>(k)] = extent(b, k) == 1 ? 0 : stride_of<float>(b, k);
            }
            if (positions) {
                // The positions of an image run on across its rows: the operand
                // is one number for them all, or lies in rows as the output.
                const bool broadcast = step.nb[0] == 0 && step.nb[1] == 0;
                const bool rows = step.nb[0] == 1 && step.nb[1] == head->ne[0];
                fits = fits && (broadcast || rows);
            }
            if (!fits) {
                break;
            }
            step.kind = node->op == GGML_OP_ADD ? Epilogue_step::Kind::add : Epilogue_step::Kind::mul;
            step.operand = elements_of<const float>(b);
            operand = b;
        }
        else {
            break;
        }
        add_step(e, step, operand);
        e.last_node = j;
    }
    // The residual connection: an addition, not in place, of the chain's
    // tensor and another of its shape and layout, into a third of the same.
    // The views between are allowed; any other node ends the chain where it is.
    for (; redirect && j < n_nodes && ggml_op_is_empty(node_at(nodes, j)->op); ++j) {
    }
    if (!redirect || j >= n_nodes || e.count >= Epilogue::kMaxSteps) {
        return e;
    }
    ggml_tensor * const node = node_at(nodes, j);
    const auto alike = [&](const ggml_tensor * const t) {
        if (t == nullptr || !holds<float>(t)) {
            return false;
        }
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            if (extent(t, k) != extent(head, k) || stride_bytes(t, k) != stride_bytes(head, k)) {
                return false;
            }
        }
        return true;
    };
    if (node->op != GGML_OP_ADD || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !alike(node) ||
        node->data == head->data) {
        return e;
    }
    const ggml_tensor * const a = node->src[0];
    const ggml_tensor * const b = node->src[1];
    const ggml_tensor * const other = a != nullptr && a->data == head->data && alike(a) ? b
                                : b != nullptr && b->data == head->data && alike(b) ? a
                                                                                     : nullptr;
    if (!alike(other) || tensors_overlap(other, head)) {
        return e;
    }
    // Nothing after reads the operation's tensor, or the chain's.
    Unwritten unwritten;
    for (int i = index; i <= e.last_node && unwritten.count < kMaxUnwritten; ++i) {
        const ggml_tensor * const t = node_at(nodes, i);
        if (t->data != head->data) {
            continue;
        }
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return e;
        }
        unwritten.add(t);
    }
    if (read_after(nodes, j + 1, unwritten.all())) {
        return e;
    }
    if (other->data == node->data) {
        // The sum in the memory of the other operand: begun from it, which
        // only additions before may be reordered with.
        for (const Epilogue_step & step : std::span(e.steps).first(static_cast<std::size_t>(e.count))) {
            if (step.kind != Epilogue_step::Kind::add) {
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
    step.operand = elements_of<const float>(other);
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        step.nb[static_cast<std::size_t>(k)] = stride_of<float>(other, k);
    }
    add_step(e, step, other);
    e.last_node = j;
    e.redirect = node;
    return e;
}

bool redirect_apart(const ggml_tensor * const node, const Epilogue & chain) {
    if (chain.redirect == nullptr) {
        return true;
    }
    const ggml_tensor * const out = chain.redirect;
    for (const ggml_tensor * const s : node->src) {
        if (s != nullptr && s->data != nullptr && tensors_overlap(out, s)) {
            return false;
        }
    }
    for (const ggml_tensor * const operand : std::span(chain.operands).first(static_cast<std::size_t>(chain.count))) {
        if (operand != nullptr && tensors_overlap(out, operand)) {
            return false;
        }
    }
    return true;
}

void gated_gelu_run(Elements<float> out, Elements<const float> x, Elements<const float> gate) {
    const std::int64_t n = out.size();
    SDASM_REQUIRE(x.size() >= n && gate.size() >= n);
    // The kernels written by hand take the numbers in groups of sixty-four
    // (AVX-512) or sixteen (AVX2); the C++ loop what is left, and all of them
    // where those are not taken.
    std::int64_t k = 0;
#if SDASM_X86_64
    if (avx512_kernels()) {
        k = n / 64 * 64;
        sdasm_gated_gelu_avx512(out.data(), x.data(), gate.data(), k);
    }
    else if (avx2_kernels()) {
        k = n / 16 * 16;
        sdasm_gated_gelu_avx2(out.data(), x.data(), gate.data(), k);
    }
#endif
    for (; k < n; ++k) {
        out[k] = x[k] * gelu(gate[k]);
    }
}

namespace {

bool read_after(Graph_nodes nodes, int from, std::span<const ggml_tensor * const> tensors) {
    for (const ggml_tensor * const node : nodes.subspan(static_cast<std::size_t>(from))) {
        for (const ggml_tensor * const s : node->src) {
            for (const ggml_tensor * t = s; t != nullptr; t = t->view_src) {
                for (const ggml_tensor * const u : tensors) {
                    if (t == u) {
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
bool columns_of(const ggml_tensor * const view, const ggml_tensor * const whole, std::int64_t begin,
                std::int64_t width) {
    if (!holds<float>(view) || !holds<float>(whole) || view->ne[0] != width || view->nb[0] != sizeof(float) ||
        begin > whole->ne[0] || elements_of<const float>(view).data() != elements_of<const float>(whole).from(begin).data()) {
        return false;
    }
    for (int k = 1; k < GGML_MAX_DIMS; ++k) {
        if (extent(view, k) != extent(whole, k) || stride_bytes(view, k) != stride_bytes(whole, k)) {
            return false;
        }
    }
    return true;
}

}  // namespace

Gated_gelu match_gated_gelu(Graph_nodes nodes, int index, const Epilogue & chain) {
    Gated_gelu r;
    const auto n_nodes = static_cast<int>(nodes.size());
    const int j = chain.last_node + 1;
    if (j + 5 > n_nodes) {
        return r;
    }
    const ggml_tensor * const head = node_at(nodes, index);
    const ggml_tensor * const v0 = node_at(nodes, j);
    const ggml_tensor * const v1 = node_at(nodes, j + 1);
    const ggml_tensor * const cont = node_at(nodes, j + 2);
    const ggml_tensor * const gelu = node_at(nodes, j + 3);
    ggml_tensor * const mul = node_at(nodes, j + 4);
    if (head->type != GGML_TYPE_F32 || !ggml_is_contiguous(head) || head->ne[0] % 2 != 0 || v0->op != GGML_OP_VIEW ||
        v1->op != GGML_OP_VIEW || cont->op != GGML_OP_CONT || gelu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(gelu) != GGML_UNARY_OP_GELU || mul->op != GGML_OP_MUL) {
        return r;
    }
    const ggml_tensor * const gate = cont->src[0];
    const ggml_tensor * const x = gate == v0 ? v1 : gate == v1 ? v0 : nullptr;
    const std::int64_t half = head->ne[0] / 2;
    if (x == nullptr || !columns_of(x, head, 0, half) || !columns_of(gate, head, half, half) ||
        gelu->src[0] != cont || gelu->data != cont->data || mul->src[0] != x || mul->src[1] != gelu ||
        mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mul, x)) {
        return r;
    }
    // Nothing after reads what is not written: the product, the chain's nodes,
    // the halves, the gate and its GELU; and none is an output of the graph.
    Unwritten unwritten;
    for (int i = index; i <= chain.last_node; ++i) {
        unwritten.add(node_at(nodes, i));
    }
    for (const ggml_tensor * const t : {static_cast<const ggml_tensor *>(v0), static_cast<const ggml_tensor *>(v1),
                                  static_cast<const ggml_tensor *>(cont), static_cast<const ggml_tensor *>(gelu)}) {
        unwritten.add(t);
    }
    for (const ggml_tensor * const t : unwritten.all()) {
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return r;
        }
    }
    if (read_after(nodes, j + 5, unwritten.all())) {
        return r;
    }
    r.out = mul;
    r.last_node = j + 4;
    return r;
}

}  // namespace sdasm
