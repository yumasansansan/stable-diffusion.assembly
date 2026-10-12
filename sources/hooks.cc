// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The functions ggml's CPU backend calls (sdasm.h), which hand the nodes this
// project takes to the operations of ops.h. Whether a node is taken depends on
// the node alone, so that every thread, and the planning of the graph before
// them, answers the same for it; the chain after a node (fusion.h) depends on
// the graph alone, and so is the same for every thread too.

#include "sdasm.h"

#include "fusion.h"
#include "kernels.h"
#include "ops.h"
#include "store.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace {

enum class Op { none, conv2d, winograd, mul_mat, attention, group_norm, norm };

Op taken(const ggml_tensor * const node) {
    if (sdasm::kernels() == sdasm::Kernels::upstream) {
        return Op::none;
    }
    // ggml's operations are many, and this project takes a few of them: the
    // rest go by.
    const ggml_op op = node->op;
    if (op == GGML_OP_CONV_2D) {
        if (sdasm::winograd_supported(node)) {
            return Op::winograd;
        }
        return sdasm::conv2d_supported(node) ? Op::conv2d : Op::none;
    }
    if (op == GGML_OP_MUL_MAT) {
        return sdasm::mul_mat_supported(node) ? Op::mul_mat : Op::none;
    }
    if (op == GGML_OP_FLASH_ATTN_EXT) {
        return sdasm::attention_supported(node) ? Op::attention : Op::none;
    }
    if (op == GGML_OP_GROUP_NORM) {
        return sdasm::group_norm_supported(node) ? Op::group_norm : Op::none;
    }
    if (op == GGML_OP_NORM) {
        return sdasm::norm_supported(node) ? Op::norm : Op::none;
    }
    return Op::none;
}

// Computes the node, and the chain after it when there is one; says how many
// nodes of the graph that was, or 0 when the node is not taken.
int compute(const ggml_compute_params * const params, const ggml_tensor * const node, const sdasm::Epilogue & chain,
            int index, const sdasm::Gated_gelu & gated = {}) {
    if (gated.out != nullptr) {
        sdasm::mul_mat_compute_gated(params, node, chain, gated.out);
        return gated.last_node - index + 1;
    }
    const int count = chain.empty() ? 1 : chain.last_node - index + 1;
    switch (taken(node)) {
    case Op::conv2d:
        sdasm::conv2d_compute(params, node, chain);
        return count;
    case Op::winograd:
        sdasm::winograd_compute(params, node, chain);
        return count;
    case Op::mul_mat:
        sdasm::mul_mat_compute(params, node, chain);
        return count;
    case Op::attention:
        sdasm::attention_compute(params, node);
        return 1;
    case Op::group_norm:
        sdasm::group_norm_compute(params, node, chain);
        return count;
    case Op::norm:
        sdasm::norm_compute(params, node, chain);
        return count;
    case Op::none:
        break;
    }
    return 0;
}

}  // namespace

extern "C" bool sdasm_compute_forward(const ggml_compute_params * const params, const ggml_tensor * const node) {
    if (params->use_ref) {
        return false;
    }
    return compute(params, node, sdasm::Epilogue{}, 0) > 0;
}

extern "C" int sdasm_compute_nodes(const ggml_compute_params * const params, ggml_tensor * const * const nodes,
                                   int n_nodes, int index) {
    if (params->use_ref || nodes == nullptr || n_nodes <= 0 || index < 0 || index >= n_nodes) {
        return 0;
    }
    const sdasm::Graph_nodes graph = sdasm::graph_nodes(nodes, n_nodes);
    const ggml_tensor * const node = sdasm::node_at(graph, index);
    // What the node writes was packed from nothing that is kept, or is not to
    // be kept any more (store.h).
    if (params->ith == 0 && node->data != nullptr) {
        sdasm::forget(node->data, ggml_nbytes(node));
    }
    sdasm::Epilogue chain;
    sdasm::Gated_gelu gated;
    // What a fused node writes in place of the node's own tensor, too.
    const auto forget_written = [&]() {
        for (const ggml_tensor * const t : {static_cast<const ggml_tensor *>(chain.redirect),
                                      static_cast<const ggml_tensor *>(gated.out)}) {
            if (params->ith == 0 && t != nullptr && t->data != nullptr) {
                sdasm::forget(t->data, ggml_nbytes(t));
            }
        }
    };
    if (sdasm::tuning().fuse != 0) {
        switch (taken(node)) {
        case Op::conv2d:
        case Op::winograd:
            chain = sdasm::match_chain(graph, index, false, true, sdasm::tuning().fuse_residual != 0);
            if (!sdasm::redirect_apart(node, chain)) {
                chain = sdasm::match_chain(graph, index, false, true);
            }
            break;
        case Op::mul_mat:
            chain = sdasm::match_chain(graph, index, false, false, sdasm::tuning().fuse_residual != 0);
            if (!sdasm::redirect_apart(node, chain)) {
                chain = sdasm::match_chain(graph, index, false, false);
            }
            if (chain.redirect == nullptr) {
                gated = sdasm::match_gated_gelu(graph, index, chain);
                if (gated.out != nullptr && !sdasm::mul_mat_gated_supported(node, chain, gated.out)) {
                    gated = {};
                }
            }
            break;
        case Op::group_norm:
        case Op::norm:
            chain = sdasm::match_chain(graph, index, true, false);
            break;
        case Op::attention:
        case Op::none:
            break;
        }
    }
    forget_written();
    if (!sdasm_profile_enabled()) {
        return compute(params, node, chain, index, gated);
    }
    const std::uint64_t start = sdasm_profile_now();
    const int count = compute(params, node, chain, index, gated);
    if (count > 0) {
        sdasm::profile_busy(n_nodes, node, count - 1, params->nth,
                            static_cast<double>(sdasm_profile_now() - start) * 1e-9);
    }
    return count;
}

extern "C" int sdasm_token_downsampling(std::int64_t width, std::int64_t height) {
    const sdasm::Tuning & t = sdasm::tuning();
    if (t.todo < 2 || width <= 0 || height <= 0 || width * height < t.todo_tokens) {
        return 1;
    }
    return t.todo;
}

extern "C" bool sdasm_work_size(const ggml_tensor * const node, int n_threads, size_t * const size) {
    switch (taken(node)) {
    case Op::conv2d:
        *size = sdasm::conv2d_work_size(node, n_threads);
        return true;
    case Op::winograd:
        *size = sdasm::winograd_work_size(node, n_threads);
        return true;
    case Op::mul_mat:
        *size = sdasm::mul_mat_work_size(node, n_threads);
        return true;
    case Op::attention:
        *size = sdasm::attention_work_size(node, n_threads);
        return true;
    case Op::group_norm:
    case Op::norm:
        *size = 0;
        return true;
    case Op::none:
        break;
    }
    return false;
}
