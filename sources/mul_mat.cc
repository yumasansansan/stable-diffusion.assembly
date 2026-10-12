// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// GGML_OP_MUL_MAT of a matrix of weights W, src0, the same for every matrix of
// src1, by the matrix product (gemm.h): dst[m][n] = sum over k of
// src1[m][k] W[n][k], for every row m of src1 in every one of its matrices,
// which, for the linear layers of a model, are the tokens. ggml rounds src1 to
// FP16 when W is FP16, and sums the products of the rounded numbers; here
// src1 stays in single precision.
//
// In the product, A is src1, packed by each thread for the block it works on,
// and B is W transposed, packed once by all threads together, widened to
// single precision when it is FP16. Each row of C is a row of dst.

#include <algorithm>
#include <cstdint>

#include "fusion.h"
#include "gemm.h"
#include "kernels.h"
#include "memory.h"
#include "ops.h"
#include "phases.h"
#include "quant.h"
#include "store.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

namespace {

struct Mul_mat {
    const ggml_tensor * weight;
    ggml_type wtype;
    Elements<const float> w32;          // [K, N], rows wnb1 apart, when the weights are of single precision
    Elements<const std::uint16_t> w16;  // or FP16
    std::int64_t wnb1;
    Elements<const float> x;  // [K, M], rows xnb1 apart, the matrices one after another
    std::int64_t xnb1;
    Elements<float> y;  // [N, M]
    std::int64_t ynb1;
    std::int64_t m, n, k;
    std::int64_t ne1, ne2;  // the rows and matrices of the output, for the chain
};

// The strides are in elements. Of a node that mul_mat_supported() takes.
Mul_mat geometry(const ggml_tensor * const node) {
    const ggml_tensor * const w = node->src[0];
    const ggml_tensor * const x = node->src[1];
    Mul_mat g{};
    g.weight = w;
    g.wtype = w->type;
    if (w->type == GGML_TYPE_F16) {
        g.w16 = elements_of<const std::uint16_t>(w);
        g.wnb1 = stride_of<std::uint16_t>(w, 1);
    }
    else {
        g.w32 = elements_of<const float>(w);
        g.wnb1 = stride_of<float>(w, 1);
    }
    g.x = elements_of<const float>(x);
    g.xnb1 = stride_of<float>(x, 1);
    g.y = elements_of<float>(node);
    g.ynb1 = stride_of<float>(node, 1);
    g.k = w->ne[0];
    g.n = w->ne[1];
    g.m = x->ne[1] * x->ne[2] * x->ne[3];
    g.ne1 = node->ne[1];
    g.ne2 = node->ne[2];
    return g;
}

// Rows [r0, r0 + rows) of the weights over [k0, k1) into panels of p (gemm.h).
void pack_weight_rows(Elements<float> dst, const Mul_mat & g, std::int64_t r0, std::int64_t rows, std::int64_t k0,
                      std::int64_t k1, int p) {
    if (g.wtype == GGML_TYPE_F16) {
        pack_rows_f16(dst, g.w16.from(r0 * g.wnb1), g.wnb1, rows, k0, k1, p);
    }
    else {
        pack_rows_f32(dst, g.w32.from(r0 * g.wnb1), g.wnb1, rows, k0, k1, p);
    }
}

// Row m of the output from column j on, `count` of it.
Elements<float> output_row(const Mul_mat & g, std::int64_t m, std::int64_t j, std::int64_t count) {
    return g.y.part(m * g.ynb1 + j, count);
}

// Whether a product is quantized (quant.h): of 256 rows or more. The products
// of fewer -- the text encoder's, of the prompt's 77 tokens, the projections
// of the prompt to cross-attention's keys and values, the time embedding's --
// are a few hundredths of the work, and steer the whole picture: they stay in
// single precision, as cross-attention does.
constexpr std::int64_t kQuantizedRows = 256;

bool quantized_product(const Mul_mat & g) {
    return quantized(Quantized_op::linear) && g.m >= kQuantizedRows;
}

struct Plan {
    // The rows of src1 packed once, by all threads together, and each thread
    // packing the weights of its block of columns: for few rows, when the
    // weights are the large operand. Otherwise the weights are packed once and
    // each thread packs the rows of its block.
    bool shared_rows;
    std::int64_t mc, nc, kc;
    std::int64_t mblocks, nblocks;
};

std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

std::int64_t round_up(std::int64_t a, std::int64_t b) {
    return ceil_div(a, b) * b;
}

Plan plan(const Mul_mat & g, int n_threads) {
    const Tuning & t = tuning();
    Plan pl{};
    pl.kc = std::min<std::int64_t>(g.k, t.mul_mat_kc);
    pl.shared_rows = g.m <= t.mul_mat_shared_rows && g.n >= 4 * kNr;
    if (pl.shared_rows) {
        const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
        pl.nc = std::clamp<std::int64_t>(g.n / wanted / kNr * kNr, kNr, 16 * kNr);
        pl.nblocks = ceil_div(g.n, pl.nc);
        pl.mc = round_up(g.m, kMr);
        pl.mblocks = 1;
        return pl;
    }
    pl.mc = std::min<std::int64_t>(round_up(g.m, kMr), round_up(t.mul_mat_mc, kMr));
    pl.mblocks = ceil_div(g.m, pl.mc);
    pl.nblocks = 1;
    pl.nc = round_up(g.n, kNr);
    // Enough blocks for every thread to take several.
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    for (std::int64_t split = 2; pl.mblocks * pl.nblocks < wanted && pl.nc > 4 * kNr; ++split) {
        pl.nc = std::min(pl.nc, round_up(ceil_div(g.n, split), kNr));
        pl.nblocks = ceil_div(g.n, pl.nc);
    }
    return pl;
}

// The weights, transposed, in panels of kNr columns over the whole of K,
// [panel][k][kNr]: the panels of B, which the store keeps (store.h).
std::int64_t weights_count(const Mul_mat & g) {
    return round_up(g.n, kNr) * g.k;
}

// The work buffer of the products of single precision: with shared rows, the
// rows of src1 packed over the whole of K, and each thread's panels of
// weights; otherwise the panels of the weights, and each thread's rows.
struct Single_layout {
    Work_layout layout;
    Shared<float> shared;
    Own<float> mine;
};

Single_layout single_layout(const Mul_mat & g, const Plan & pl) {
    Single_layout l;
    if (pl.shared_rows) {
        l.shared = l.layout.shared<float>(round_up(g.m, kMr) * g.k);
        l.mine = l.layout.own<float>(pl.nc * pl.kc);
    }
    else {
        l.shared = l.layout.shared<float>(weights_count(g));
        l.mine = l.layout.own<float>(pl.mc * pl.kc);
    }
    return l;
}

}  // namespace

bool mul_mat_supported(const ggml_tensor * const node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * const w = node->src[0];
    const ggml_tensor * const x = node->src[1];
    // The weights are one matrix, and the rows of the other operand and of the
    // result lie one after another, matrix after matrix. Products too small to
    // pay for packing the weights are left to ggml.
    const bool weights = (w->type == GGML_TYPE_F16 && holds<std::uint16_t>(w)) || (w->type == GGML_TYPE_F32 && holds<float>(w));
    return weights && w->ne[2] == 1 && w->ne[3] == 1 && w->nb[0] == ggml_type_size(w->type) && holds<float>(x) &&
           ggml_is_contiguous(x) && holds<float>(node) && ggml_is_contiguous(node) && w->extra == nullptr &&
           x->ne[1] * x->ne[2] * x->ne[3] >= 16 && w->ne[0] >= 16;
}

namespace {

std::size_t gated_bytes(const Mul_mat & g, const Plan & pl, int n_threads);
bool amx_product(const Mul_mat & g);
std::size_t amx_work_size(const Mul_mat & g, int n_threads);
std::size_t quantized_work_size(const Mul_mat & g, int n_threads);

}  // namespace

std::size_t mul_mat_work_size(const ggml_tensor * const node, int n_threads) {
    const Mul_mat g = geometry(node);
    if (quantized_product(g)) {
        return quantized_work_size(g, n_threads);
    }
    const Plan pl = plan(g, n_threads);
    // Enough for the gated GELU after it too (fusion.h), which needs more of
    // each thread; and for the products on AMX's tiles, when they go there.
    const std::size_t bytes = std::max(single_layout(g, pl).layout.bytes(n_threads), gated_bytes(g, pl, n_threads));
    return amx_product(g) ? std::max(bytes, amx_work_size(g, n_threads)) : bytes;
}

namespace {

constexpr std::int64_t kPanelsPerItem = 2;

std::int64_t weight_items(const Mul_mat & g) {
    return ceil_div(ceil_div(g.n, kNr), kPanelsPerItem);
}

void pack_weights(Elements<float> weights, const Mul_mat & g, std::int64_t item) {
    const std::int64_t r0 = item * kPanelsPerItem * kNr;
    const std::int64_t count = std::min<std::int64_t>(kPanelsPerItem * kNr, g.n - r0);
    pack_weight_rows(weights.from(r0 * g.k), g, r0, count, 0, g.k, kNr);
}

void apply_chain(const Epilogue & chain, const Mul_mat & g, std::int64_t ii, std::int64_t count, std::int64_t jj,
                 std::int64_t cols) {
    for (std::int64_t r = 0; r < count; ++r) {
        const std::int64_t m = ii + r;
        chain.apply(output_row(g, m, jj, cols), jj, m % g.ne1, (m / g.ne1) % g.ne2, m / (g.ne1 * g.ne2));
    }
}

void compute_shared_rows(const ggml_compute_params * const params, const Mul_mat & g, const Plan & pl,
                         const Epilogue & chain) {
    const Single_layout l = single_layout(g, pl);
    Work_buffer buffer(params, l.layout);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    // The panels of the weights the store keeps, packed with the rows when
    // they are not yet; without them, each block packs its own as it goes.
    const std::shared_ptr<Packed> kept = packed(g.weight, Layout::linear_columns, weights_count(g));
    const bool fill = kept != nullptr && !kept->ready();

    Work_queue queue(params, buffer);
    const Elements<float> rows = buffer[l.shared];
    const Elements<float> weights = buffer[l.mine];
    // The rows of src1, a panel of kMr rows to an item, over the whole of K;
    // then the weights, when the store is to keep them.
    const std::int64_t row_items = ceil_div(g.m, kMr);
    const std::int64_t pack_items = row_items + (fill ? weight_items(g) : 0);
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        if (item >= row_items) {
            const Phase_scope scope(Phase::pack_weights);
            pack_weights(kept->singles(), g, item - row_items);
            continue;
        }
        const Phase_scope scope(Phase::pack_inputs);
        const std::int64_t r0 = item * kMr;
        pack_rows_f32(rows.from(r0 * g.k), g.x.from(r0 * g.xnb1), g.xnb1, std::min<std::int64_t>(kMr, g.m - r0), 0, g.k, kMr);
    }
    queue.barrier();
    if (fill && params->ith == 0) {
        publish(kept);
    }

    const std::vector<Block> blocks = guided_blocks(1, g.n, kNr, pl.nc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    for (; item < pack_items + units; item = queue.next()) {
        const Block & block = blocks[static_cast<std::size_t>(item - pack_items)];
        const std::int64_t j0 = block.start;
        const std::int64_t j1 = j0 + block.size;
        for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, g.k - k0);
            if (kept == nullptr) {
                const Phase_scope scope(Phase::pack_weights);
                pack_weight_rows(weights, g, j0, j1 - j0, k0, k0 + kc, kNr);
            }
            const bool last = k0 + kc == g.k;
            const Phase_scope scope(Phase::product);
            const std::int64_t b_next = (kept != nullptr ? g.k : kc) * kNr;
            for (std::int64_t jj = j0; jj < j1; jj += step) {
                const Elements<const float> b =
                    kept != nullptr ? kept->singles().from(jj * g.k + k0 * kNr) : weights.from((jj - j0) * kc);
                const std::int64_t cols = std::min(step, j1 - jj);
                for (std::int64_t ii = 0; ii < g.m; ii += kMr) {
                    const std::int64_t count = std::min<std::int64_t>(kMr, g.m - ii);
                    tiles(uk, wide, kc, rows.from(ii * g.k + k0 * kMr), b, b_next, g.y.from(ii * g.ynb1 + jj), g.ynb1,
                          k0 > 0 || chain.from_redirect, count, cols);
                    if (last && !chain.empty()) {
                        apply_chain(chain, g, ii, count, jj, cols);
                    }
                }
            }
        }
    }
}

void compute_shared_weights(const ggml_compute_params * const params, const Mul_mat & g, const Plan & pl,
                            const Epilogue & chain) {
    const Single_layout l = single_layout(g, pl);
    Work_buffer buffer(params, l.layout);
    // The panels of the weights: the store's, packed here the first time, or
    // the work buffer's, packed here every time.
    const std::shared_ptr<Packed> kept = packed(g.weight, Layout::linear_columns, weights_count(g));
    const bool fill = kept == nullptr || !kept->ready();
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const std::int64_t b_next = g.k * kNr;

    Work_queue queue(params, buffer);
    const Elements<float> weights = kept != nullptr ? kept->singles() : buffer[l.shared];
    const Elements<float> rows = buffer[l.mine];
    const std::int64_t pack_items = fill ? weight_items(g) : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            pack_weights(weights, g, item);
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }

    const std::vector<Block> blocks = guided_blocks(1, g.m, kMr, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.nblocks;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const std::int64_t jb = u % pl.nblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.nblocks)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        const std::int64_t j0 = jb * pl.nc;
        const std::int64_t j1 = std::min(g.n, j0 + pl.nc);
        for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, g.k - k0);
            {
                const Phase_scope scope(Phase::pack_inputs);
                pack_rows_f32(rows, g.x.from(i0 * g.xnb1), g.xnb1, i1 - i0, k0, k0 + kc, kMr);
            }
            const Phase_scope scope(Phase::product);
            for (std::int64_t jj = j0; jj < j1; jj += step) {
                const Elements<const float> b = weights.from(jj * g.k + k0 * kNr);
                const std::int64_t cols = std::min(step, j1 - jj);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    const std::int64_t count = std::min<std::int64_t>(kMr, i1 - ii);
                    tiles(uk, wide, kc, rows.from(((ii - i0) / kMr) * kc * kMr), b, b_next, g.y.from(ii * g.ynb1 + jj),
                          g.ynb1, k0 > 0 || chain.from_redirect, count, cols);
                    if (k0 + kc == g.k && !chain.empty()) {
                        apply_chain(chain, g, ii, count, jj, cols);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- quantized

// The products of numbers quantized to integers (quant.h): A is src1, its rows
// quantized by all threads together, a panel of kMr rows over the whole of K
// to an item; B the weights, quantized likewise a panel of kNr columns to an
// item, or kept by the store. Every tile is then a call of the micro-kernel
// over the whole of K: the panels are a quarter of the bytes of those of
// single precision, and a panel of B, over K = 1280 in groups of 32, takes
// 25 kilobytes of the first level of the cache, where it stays while the
// panels of A of the block go by.
std::uint64_t quantized_variant(const Quant & q) {
    return static_cast<std::uint64_t>(q.quads) | static_cast<std::uint64_t>(q.weight_max) << 16 |
           static_cast<std::uint64_t>(q.weight_groups ? 1 : 0) << 32;
}

// The rows of a panel of A, of a block of them and the rows the blocks of rows
// are multiples of: kMr for the micro-kernels; on AMX-INT8, a tile's sixteen,
// in blocks of 32.
struct Quantized_shape {
    Qkernel_amx amx;
    int p;
    std::int64_t unit;
};

Quantized_shape quantized_shape(const Quant & q) {
    const Qkernel_amx amx = qkernel_amx_a_unsigned(q);
    return amx != nullptr ? Quantized_shape{amx, static_cast<int>(kAmxTile), kAmxBlock} : Quantized_shape{nullptr, kMr, kMr};
}

// The panels of A, and of B, the weights: on the tiles, A's rounded up to a
// pair, a block of 32 rows; B's one more than the columns take, of zeros, so
// that a block of 32 columns of the tiles has its second panel wherever it
// begins, at a column a multiple of 16.
std::int64_t quantized_row_panels(const Mul_mat & g, const Quantized_shape & sh) {
    return sh.amx != nullptr ? round_up(ceil_div(g.m, sh.p), 2) : ceil_div(g.m, sh.p);
}
std::int64_t quantized_weight_panels(const Mul_mat & g) {
    return ceil_div(g.n, kNr) + 1;
}

std::int64_t quantized_weights_count(const Mul_mat & g, const Quant & q) {
    return quantized_weight_panels(g) * qpanel_bytes(g.k, kNr, q);
}

constexpr std::int64_t kQuantizedGatedColumns = 64;

// The quantized rows and weights, shared, and each thread's tiles of a block
// of rows by a chunk of the columns of both halves, for the gated GELU after
// the product, and on the tiles the block it copies a part of.
struct Quantized_layout {
    Work_layout layout;
    Shared<std::uint8_t> rows, weights;
    Own<float> gated, spare;
};

std::int64_t quantized_mc(const Mul_mat & g, const Quantized_shape & sh) {
    return std::min<std::int64_t>(round_up(g.m, sh.unit), round_up(tuning().mul_mat_mc, sh.unit));
}

Quantized_layout quantized_layout(const Mul_mat & g, const Quant & q, const Quantized_shape & sh) {
    Quantized_layout l;
    l.rows = l.layout.shared<std::uint8_t>(quantized_row_panels(g, sh) * qpanel_bytes(g.k, sh.p, q));
    l.weights = l.layout.shared<std::uint8_t>(quantized_weights_count(g, q));
    l.gated = l.layout.own<float>(quantized_mc(g, sh) * 2 * kQuantizedGatedColumns);
    l.spare = l.layout.own<float>(sh.amx != nullptr ? kAmxBlock * kAmxBlock : 0);
    return l;
}

std::size_t quantized_work_size(const Mul_mat & g, int n_threads) {
    const Quant q = quant();
    return quantized_layout(g, q, quantized_shape(q)).layout.bytes(n_threads);
}

// The quantized rows of src1 and the weights, packed by the queue's first
// items, from `item`, a panel of A's rows or of kNr columns to an item, and
// the threads met after them: the panels, and the number of those items. A
// panel past the last row or column is zeros.
struct Quantized_operands {
    Elements<const std::uint8_t> rows;
    Elements<const std::uint8_t> weights;
    std::int64_t items;
};

Quantized_operands quantize_operands(const ggml_compute_params * const params, const Mul_mat & g, const Quant & q,
                                     const Quantized_shape & sh, const std::shared_ptr<Packed> & kept,
                                     Elements<std::uint8_t> rows, Elements<std::uint8_t> work_weights, Work_queue & queue,
                                     int & item) {
    const std::int64_t a_panel = qpanel_bytes(g.k, sh.p, q);
    const std::int64_t b_panel = qpanel_bytes(g.k, kNr, q);
    const Elements<std::uint8_t> weights = kept != nullptr ? kept->octets() : work_weights;
    const bool fill = kept == nullptr || !kept->ready();
    const std::int64_t row_items = quantized_row_panels(g, sh);
    const std::int64_t pack_items = row_items + (fill ? quantized_weight_panels(g) : 0);
    for (; item < pack_items; item = queue.next()) {
        if (item < row_items) {
            const Phase_scope scope(Phase::pack_inputs);
            const std::int64_t r0 = item * sh.p;
            const Elements<std::uint8_t> dst = rows.part(item * a_panel, a_panel);
            if (r0 >= g.m) {
                fill_with<std::uint8_t>(dst, 0);
                continue;
            }
            quantize_inputs(dst, g.x.from(r0 * g.xnb1), g.xnb1, std::min<std::int64_t>(sh.p, g.m - r0), g.k, sh.p,
                            Qlayout::rows, q);
            continue;
        }
        const Phase_scope scope(Phase::pack_weights);
        const std::int64_t panel = item - row_items;
        const std::int64_t c0 = panel * kNr;
        const Elements<std::uint8_t> dst = weights.part(panel * b_panel, b_panel);
        if (c0 >= g.n) {
            fill_with<std::uint8_t>(dst, 0);
            continue;
        }
        if (g.wtype == GGML_TYPE_F16) {
            quantize_weights_f16(dst, g.w16.from(c0 * g.wnb1), g.wnb1, std::min<std::int64_t>(kNr, g.n - c0), g.k, kNr,
                                 Qlayout::columns, q);
        }
        else {
            quantize_weights_f32(dst, g.w32.from(c0 * g.wnb1), g.wnb1, std::min<std::int64_t>(kNr, g.n - c0), g.k, kNr,
                                 Qlayout::columns, q);
        }
    }
    queue.barrier();
    if (fill && kept != nullptr && params->ith == 0) {
        publish(kept);
    }
    return {rows, weights, pack_items};
}

void compute_quantized(const ggml_compute_params * const params, const Mul_mat & g, const Epilogue & chain) {
    const Quant q = quant();
    const Quantized_shape sh = quantized_shape(q);
    const std::int64_t groups = qgroups(g.k, q);
    const std::int64_t a_panel = qpanel_bytes(g.k, sh.p, q);
    const std::int64_t b_panel = qpanel_bytes(g.k, kNr, q);
    const Quantized_layout l = quantized_layout(g, q, sh);
    Work_buffer buffer(params, l.layout);
    const std::shared_ptr<Packed> kept =
        packed(g.weight, Layout::quantized_linear, quantized_weights_count(g, q), quantized_variant(q));
    const Qkernel kernel = qkernel_a_unsigned();
    const Qkernel_wide wide = qkernel_wide_a_unsigned();
    // The columns a step takes: a block of the tiles; or two panels with the
    // wide kernel, one without.
    const std::int64_t step = sh.amx != nullptr ? kAmxBlock : wide != nullptr ? 2 * kNr : kNr;

    Work_queue queue(params, buffer);
    const Elements<float> spare = buffer[l.spare];
    int item = queue.next();
    const Quantized_operands operands =
        quantize_operands(params, g, q, sh, kept, buffer[l.rows], buffer[l.weights], queue, item);
    const std::int64_t pack_items = operands.items;

    // Blocks of rows by blocks of columns, enough of them for every thread to
    // take several.
    const std::int64_t mc = quantized_mc(g, sh);
    const std::int64_t mblocks = ceil_div(g.m, mc);
    std::int64_t nc = round_up(g.n, step);
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(params->nth);
    for (std::int64_t split = 2; mblocks * ceil_div(g.n, nc) < wanted && nc > 4 * kNr; ++split) {
        nc = std::min(nc, round_up(ceil_div(g.n, split), step));
    }
    const std::int64_t nblocks = ceil_div(g.n, nc);
    const std::vector<Block> blocks = guided_blocks(1, g.m, sh.unit, mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * nblocks;
    bool configured = false;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const Block & block = blocks[static_cast<std::size_t>(u / nblocks)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        const std::int64_t j0 = (u % nblocks) * nc;
        const std::int64_t j1 = std::min(g.n, j0 + nc);
        const Phase_scope scope(Phase::product);
        if (sh.amx != nullptr && !configured) {
            qamx_begin();
            configured = true;
        }
        for (std::int64_t jj = j0; jj < j1; jj += step) {
            const Elements<const std::uint8_t> b = operands.weights.from((jj / kNr) * b_panel);
            const std::int64_t cols = std::min(step, j1 - jj);
            for (std::int64_t ii = i0; ii < i1; ii += sh.unit) {
                const std::int64_t count = std::min(sh.unit, i1 - ii);
                const Elements<const std::uint8_t> a = operands.rows.from((ii / sh.p) * a_panel);
                const Elements<float> c = g.y.from(ii * g.ynb1 + jj);
                if (sh.amx != nullptr) {
                    qblock_amx(sh.amx, groups, q.quads, a, a_panel, b, b_panel, c, g.ynb1, chain.from_redirect, count,
                               cols, spare);
                }
                else {
                    qtiles(kernel, wide, groups, q.quads, a, b, b_panel, c, g.ynb1, chain.from_redirect, count, cols);
                }
                if (!chain.empty()) {
                    apply_chain(chain, g, ii, count, jj, cols);
                }
            }
        }
    }
    if (configured) {
        qamx_end();
    }
}

void gated_epilogue(const Mul_mat & g, const Epilogue & chain, const ggml_tensor * const out, Elements<float> buffer,
                    std::int64_t i0, std::int64_t i1, std::int64_t c0, std::int64_t width);

// The product before the gated GELU, quantized: as compute_quantized(), a
// block of rows by a chunk of 64 columns of each half at a time, into the
// thread's tiles, which gated_epilogue() takes as compute_gated()'s.
void compute_quantized_gated(const ggml_compute_params * const params, const Mul_mat & g, const Epilogue & chain,
                             const ggml_tensor * const out) {
    const Quant q = quant();
    const Quantized_shape sh = quantized_shape(q);
    const std::int64_t groups = qgroups(g.k, q);
    const std::int64_t a_panel = qpanel_bytes(g.k, sh.p, q);
    const std::int64_t b_panel = qpanel_bytes(g.k, kNr, q);
    const std::int64_t half = g.n / 2;
    const Quantized_layout l = quantized_layout(g, q, sh);
    Work_buffer work(params, l.layout);
    const std::shared_ptr<Packed> kept =
        packed(g.weight, Layout::quantized_linear, quantized_weights_count(g, q), quantized_variant(q));
    const Qkernel kernel = qkernel_a_unsigned();
    const Qkernel_wide wide = qkernel_wide_a_unsigned();
    const std::int64_t step = sh.amx != nullptr ? kAmxBlock : wide != nullptr ? 2 * kNr : kNr;
    const std::int64_t ldb = 2 * kQuantizedGatedColumns;

    Work_queue queue(params, work);
    const Elements<float> buffer = work[l.gated];
    int item = queue.next();
    const Quantized_operands operands =
        quantize_operands(params, g, q, sh, kept, work[l.rows], work[l.weights], queue, item);
    const std::int64_t pack_items = operands.items;

    const std::int64_t mc = quantized_mc(g, sh);
    const std::int64_t chunks = ceil_div(half, kQuantizedGatedColumns);
    const std::vector<Block> blocks = guided_blocks(1, g.m, sh.unit, mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * chunks;
    bool configured = false;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const Block & block = blocks[static_cast<std::size_t>(u / chunks)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        const std::int64_t c0 = (u % chunks) * kQuantizedGatedColumns;
        const std::int64_t width = std::min(kQuantizedGatedColumns, half - c0);
        {
            const Phase_scope scope(Phase::product);
            if (sh.amx != nullptr && !configured) {
                qamx_begin();
                configured = true;
            }
            for (std::int64_t side = 0; side < 2; ++side) {
                for (std::int64_t jj = 0; jj < width; jj += step) {
                    const Elements<const std::uint8_t> b = operands.weights.from(((side * half + c0 + jj) / kNr) * b_panel);
                    for (std::int64_t ii = i0; ii < i1; ii += sh.unit) {
                        const Elements<const std::uint8_t> a = operands.rows.from((ii / sh.p) * a_panel);
                        const Elements<float> c = buffer.from((ii - i0) * ldb + side * kQuantizedGatedColumns + jj);
                        if (sh.amx != nullptr) {
                            // A whole block into the buffer, which has room
                            // for it: the rows past the block's and the
                            // columns past the chunk's go unread.
                            qblock_amx(sh.amx, groups, q.quads, a, a_panel, b, b_panel, c, ldb, false, kAmxBlock, kAmxBlock,
                                       Elements<float>());
                        }
                        else {
                            qtiles(kernel, wide, groups, q.quads, a, b, b_panel, c, ldb, false,
                                   std::min<std::int64_t>(kMr, i1 - ii), std::min(step, width - jj));
                        }
                    }
                }
            }
        }
        const Phase_scope scope(Phase::transform_out);
        gated_epilogue(g, chain, out, buffer, i0, i1, c0, width);
    }
    if (configured) {
        qamx_end();
    }
}

// ---------------------------------------------------------------- on AMX

// The products of FP16 weights on AMX's tiles, of numbers split in parts
// (gemm.h, x86_64/amx.S and amx_fp16.S, which give the precision): A is src1,
// in tiles of sixteen rows by 32 of k; B is W transposed, in tiles of sixteen
// pairs of k by sixteen columns, in panels of sixteen columns over the whole of
// K, packed by all threads together, or kept by the store. The kernel computes
// a block of 32 rows by 32 columns of C, two tiles by two, straight into dst; K
// is rounded up to 32 with zeros, as are the rows and columns of the last
// blocks, and a block that reaches past dst is computed into the thread's own
// and copied.
//
// The work is shared out as on the vector units, by blocks of rows and of
// columns, each thread packing the rows of its block a stretch of K at a time:
// the stretch of B the block's columns take, 32 by 256 by four bytes, stays in
// the first level of the cache while the rows go by.
bool amx_product(const Mul_mat & g) {
    return split_form(Split_use::linear).kernel != nullptr && g.wtype == GGML_TYPE_F16;
}

// The elements of a panel of sixteen columns of B over the whole of K, and of
// all of B: the panels of the columns rounded up to 32. A block of k of a tile
// of A or B is a_bytes or b_bytes, of elements of two bytes.
std::int64_t amx_panel(const Mul_mat & g, const Split_form & f) {
    return round_up(g.k, kAmxBlock) / kAmxBlock * (f.b_bytes / 2);
}

std::int64_t amx_weights_count(const Mul_mat & g, const Split_form & f) {
    return round_up(g.n, kAmxBlock) / kAmxTile * amx_panel(g, f);
}

struct Amx_plan {
    std::int64_t kc;  // a multiple of 32
    std::int64_t mc;  // a multiple of 32
    std::int64_t nc;  // a multiple of 32
    std::int64_t nblocks;
};

Amx_plan amx_plan(const Mul_mat & g, int n_threads) {
    const Tuning & t = tuning();
    Amx_plan pl{};
    pl.kc = std::min(round_up(g.k, kAmxBlock), round_up(std::max(t.amx_kc, 1), kAmxBlock));
    pl.mc = std::min(round_up(g.m, kAmxBlock), round_up(std::max(t.amx_mc, 1), kAmxBlock));
    const std::int64_t mblocks = ceil_div(g.m, pl.mc);
    pl.nc = round_up(g.n, kAmxBlock);
    pl.nblocks = 1;
    // Enough blocks for every thread to take several.
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    for (std::int64_t split = 2; mblocks * pl.nblocks < wanted && pl.nc > 2 * kAmxBlock; ++split) {
        pl.nc = std::min(pl.nc, round_up(ceil_div(g.n, split), kAmxBlock));
        pl.nblocks = ceil_div(g.n, pl.nc);
    }
    return pl;
}

constexpr std::int64_t kGatedColumns = 64;

// The work buffer of the products on AMX: the tiles of B, shared, and each
// thread's tiles of its rows and the block it copies a part of -- and, for the
// gated GELU after the product, its tiles of the products of both halves.
struct Amx_layout {
    Work_layout layout;
    Shared<std::uint16_t> weights;
    Own<std::uint16_t> rows;
    Own<float> spare;
    Own<float> buffer;  // the gated GELU's
};

Amx_layout amx_layout(const Mul_mat & g, const Split_form & f, const Amx_plan & pl, bool gated) {
    Amx_layout l;
    l.weights = l.layout.shared<std::uint16_t>(amx_weights_count(g, f));
    // The plain product's rows are a stretch of K at a time; the gated one's
    // over the whole of K, for blocks of the same rows.
    const std::int64_t mc = gated ? std::min(round_up(g.m, kAmxBlock), round_up(std::max(tuning().amx_mc, 1), kAmxBlock))
                                  : pl.mc;
    const std::int64_t kc = gated ? round_up(g.k, kAmxBlock) : pl.kc;
    l.rows = l.layout.own<std::uint16_t>(mc / kAmxTile * (kc / kAmxBlock) * (f.a_bytes / 2));
    l.spare = l.layout.own<float>(kAmxBlock * kAmxBlock);
    l.buffer = l.layout.own<float>(gated ? mc * 2 * kGatedColumns : 0);
    return l;
}

// The work buffer of either product on AMX, the plain and the gated.
std::size_t amx_work_size(const Mul_mat & g, int n_threads) {
    const Split_form f = split_form(Split_use::linear);
    const Amx_plan pl = amx_plan(g, n_threads);
    return std::max(amx_layout(g, f, pl, false).layout.bytes(n_threads), amx_layout(g, f, pl, true).layout.bytes(n_threads));
}

// The tiles of B: the store's, or the work buffer's, packed by all threads
// together, two panels of sixteen columns to an item of the work queue, when
// they are not ready; a panel past the last column is zeros.
struct Amx_weights {
    std::shared_ptr<Packed> kept;
    Elements<std::uint16_t> tiles;
    std::int64_t items;
};

Amx_weights amx_weights(const Mul_mat & g, const std::shared_ptr<Packed> & kept, Elements<std::uint16_t> work) {
    Amx_weights w;
    w.kept = kept;
    w.tiles = kept != nullptr ? kept->halves() : work;
    w.items = kept == nullptr || !kept->ready() ? round_up(g.n, kAmxBlock) / kAmxBlock : 0;
    return w;
}

std::shared_ptr<Packed> amx_kept(const Mul_mat & g, const Split_form & f) {
    return packed(g.weight, f.fp16 ? Layout::amx_linear_fp16 : Layout::amx_linear, amx_weights_count(g, f));
}

// The items of the packing of B, as the queue hands them out from `item`, then
// the wait for every thread; the first item after.
int amx_pack_weights(const ggml_compute_params * const params, const Mul_mat & g, const Split_form & f,
                     const Amx_weights & w, Work_queue & queue, int item) {
    if (w.items == 0) {
        return item;
    }
    const std::int64_t panel = amx_panel(g, f);
    for (; item < w.items; item = queue.next()) {
        const Phase_scope scope(Phase::pack_weights);
        for (std::int64_t p = 0; p < 2; ++p) {
            const std::int64_t c0 = item * kAmxBlock + p * kAmxTile;
            const Elements<std::uint16_t> dst = w.tiles.part(c0 / kAmxTile * panel, panel);
            if (c0 >= g.n) {
                fill_with<std::uint16_t>(dst, 0);
                continue;
            }
            const std::int64_t cols = std::min(kAmxTile, g.n - c0);
            const Elements<const std::uint16_t> src = g.w16.from(c0 * g.wnb1);
            SDASM_CHECK(src.size() >= (cols - 1) * g.wnb1 + g.k);
            f.pack_weights(dst.data(), src.data(), bytes_of<std::uint16_t>(g.wnb1), cols, g.k);
        }
    }
    queue.barrier();
    if (params->ith == 0) {
        publish(w.kept);
    }
    return item;
}

// The rows [i0, i1) of src1 over the stretch [k0, k0 + kc) of K, kc a multiple
// of 32, into the thread's tiles, a tile of sixteen rows `tile` elements after
// the last, rounded up to a pair with zeros.
void amx_pack_rows_block(const Mul_mat & g, const Split_form & f, Elements<std::uint16_t> rows, std::int64_t i0,
                         std::int64_t i1, std::int64_t k0, std::int64_t kc) {
    const Phase_scope scope(Phase::pack_inputs);
    const std::int64_t tile = kc / kAmxBlock * (f.a_bytes / 2);
    for (std::int64_t t = 0; t < round_up(i1 - i0, kAmxBlock) / kAmxTile; ++t) {
        const std::int64_t r0 = i0 + t * kAmxTile;
        const Elements<std::uint16_t> dst = rows.part(t * tile, tile);
        if (r0 >= i1) {
            fill_with<std::uint16_t>(dst, 0);
            continue;
        }
        const std::int64_t count = std::min(kAmxTile, i1 - r0);
        const std::int64_t k = std::min(kc, g.k - k0);
        const Elements<const float> src = g.x.from(r0 * g.xnb1 + k0);
        SDASM_CHECK(src.size() >= (count - 1) * g.xnb1 + k);
        f.pack_rows(dst.data(), src.data(), bytes_of<float>(g.xnb1), count, k);
    }
}

// The products of the packed rows [i0, i1) with the columns [j0, j1) of B over
// the stretch at k0 of kc: rows, the rows' tiles of the stretch, a tile of
// sixteen rows `tile` elements after the last; the block of C at rows ii and
// columns jj written at out(ii, jj), rows ldc elements apart; then done(ii,
// rows, jj, cols).
template <class Out, class Done>
void amx_products(const Split_form & f, const Mul_mat & g, Elements<const std::uint16_t> tiles,
                  Elements<const std::uint16_t> rows, std::int64_t tile, Elements<float> spare, std::int64_t i0,
                  std::int64_t i1, std::int64_t j0, std::int64_t j1, std::int64_t k0, std::int64_t kc, std::int64_t ldc,
                  bool accumulate, Out out, Done done) {
    const Phase_scope scope(Phase::product);
    const std::int64_t panel = amx_panel(g, f);
    for (std::int64_t jj = j0; jj < j1; jj += kAmxBlock) {
        const Elements<const std::uint16_t> b = tiles.from(jj / kAmxTile * panel + k0 / kAmxBlock * (f.b_bytes / 2));
        const std::int64_t cols = std::min(kAmxBlock, j1 - jj);
        for (std::int64_t ii = i0; ii < i1; ii += kAmxBlock) {
            const Elements<const std::uint16_t> a = rows.from((ii - i0) / kAmxTile * tile);
            const std::int64_t count = std::min(kAmxBlock, i1 - ii);
            split_block(f, kc / kAmxBlock, a, tile, b, panel, out(ii, jj), ldc, accumulate, count, cols, spare);
            done(ii, count, jj, cols);
        }
    }
}

void compute_amx(const ggml_compute_params * const params, const Mul_mat & g, const Epilogue & chain) {
    const Split_form f = split_form(Split_use::linear);
    const Amx_plan pl = amx_plan(g, params->nth);
    const Amx_layout l = amx_layout(g, f, pl, false);
    Work_buffer buffer(params, l.layout);
    const std::shared_ptr<Packed> kept = amx_kept(g, f);

    Work_queue queue(params, buffer);
    const Amx_weights w = amx_weights(g, kept, buffer[l.weights]);
    const Elements<std::uint16_t> rows = buffer[l.rows];
    const Elements<float> spare = buffer[l.spare];
    int item = amx_pack_weights(params, g, f, w, queue, queue.next());
    const std::vector<Block> blocks = guided_blocks(1, g.m, kAmxBlock, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.nblocks;
    bool configured = false;
    for (; item < w.items + units; item = queue.next()) {
        if (!configured) {
            split_begin(f);
            configured = true;
        }
        const std::int64_t u = item - w.items;
        const std::int64_t jb = u % pl.nblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.nblocks)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        const std::int64_t j0 = jb * pl.nc;
        const std::int64_t j1 = std::min(g.n, j0 + pl.nc);
        for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, round_up(g.k, kAmxBlock) - k0);
            amx_pack_rows_block(g, f, rows, i0, i1, k0, kc);
            const bool last = k0 + pl.kc >= g.k;
            amx_products(
                f, g, w.tiles, rows, kc / kAmxBlock * (f.a_bytes / 2), spare, i0, i1, j0, j1, k0, kc, g.ynb1,
                k0 > 0 || chain.from_redirect, [&](std::int64_t ii, std::int64_t jj) { return g.y.from(ii * g.ynb1 + jj); },
                [&](std::int64_t ii, std::int64_t count, std::int64_t jj, std::int64_t cols) {
                    if (last && !chain.empty()) {
                        apply_chain(chain, g, ii, count, jj, cols);
                    }
                });
        }
    }
    if (configured) {
        split_end(f);
    }
}

// The gated GELU after the product (fusion.h): out[m][j] = x GELU(gate), with x
// = P[m][j] and gate = P[m][half + j], P the product with its chain. A block of
// output is the columns [c0, c0 + kGatedColumns) of the half for a block of
// rows: the product of its x and of its gate, a tile of each at a time, goes to
// the thread's own buffer, [row][x columns, gate columns], where the chain and
// the gated GELU take it into out. The product is never written whole.
struct Gated_layout {
    Work_layout layout;
    Shared<float> shared;  // the rows of src1, with shared rows; otherwise the weights
    Own<float> packed;     // the weights of both halves of the columns of a block; or its rows over the whole of K
    Own<float> buffer;
};

Gated_layout gated_layout(const Mul_mat & g, const Plan & pl) {
    Gated_layout l;
    l.shared = l.layout.shared<float>(pl.shared_rows ? round_up(g.m, kMr) * g.k : weights_count(g));
    l.packed = l.layout.own<float>(pl.shared_rows ? 2 * kGatedColumns * pl.kc : pl.mc * g.k);
    l.buffer = l.layout.own<float>((pl.shared_rows ? round_up(g.m, kMr) : pl.mc) * 2 * kGatedColumns);
    return l;
}

std::size_t gated_bytes(const Mul_mat & g, const Plan & pl, int n_threads) {
    return gated_layout(g, pl).layout.bytes(n_threads);
}

void gated_epilogue(const Mul_mat & g, const Epilogue & chain, const ggml_tensor * const out, Elements<float> buffer,
                    std::int64_t i0, std::int64_t i1, std::int64_t c0, std::int64_t width) {
    static_assert(kGatedColumns == kQuantizedGatedColumns, "the tiles of both halves lie alike");
    const std::int64_t half = g.n / 2;
    const Elements<float> result = elements_of<float>(out);
    const std::int64_t onb1 = stride_of<float>(out, 1);
    for (std::int64_t m = i0; m < i1; ++m) {
        const Elements<float> x = buffer.part((m - i0) * 2 * kGatedColumns, width);
        const Elements<float> gate = buffer.part((m - i0) * 2 * kGatedColumns + kGatedColumns, width);
        if (!chain.empty()) {
            const std::int64_t r1 = m % g.ne1;
            const std::int64_t r2 = (m / g.ne1) % g.ne2;
            const std::int64_t r3 = m / (g.ne1 * g.ne2);
            chain.apply(x, c0, r1, r2, r3);
            chain.apply(gate, half + c0, r1, r2, r3);
        }
        gated_gelu_run(result.part(m * onb1 + c0, width), x, gate);
    }
}

void compute_gated(const ggml_compute_params * const params, const Mul_mat & g, const Plan & pl, const Epilogue & chain,
                   const ggml_tensor * const out) {
    const Gated_layout l = gated_layout(g, pl);
    Work_buffer work(params, l.layout);
    const std::int64_t half = g.n / 2;
    const std::int64_t ldb = 2 * kGatedColumns;
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const std::int64_t chunks = ceil_div(half, kGatedColumns);
    // The weights, packed once by all threads (or kept by the store), when the
    // rows are not shared.
    const std::shared_ptr<Packed> kept =
        pl.shared_rows ? nullptr : packed(g.weight, Layout::linear_columns, weights_count(g));

    Work_queue queue(params, work);
    const Elements<float> packed_mine = work[l.packed];
    const Elements<float> buffer = work[l.buffer];

    // The tiles of a block of rows [i0, i1) and columns [c0, c0 + width) of
    // the half, for a stretch [k0, k0 + kc) of K: a(ii) the panel of rows, b(jj)
    // the panel of columns jj of the product, the next b_next elements after it.
    const auto products = [&](std::int64_t i0, std::int64_t i1, std::int64_t c0, std::int64_t width, std::int64_t k0,
                              std::int64_t kc, std::int64_t b_next, const auto & a, const auto & b) {
        const Phase_scope scope(Phase::product);
        for (std::int64_t side = 0; side < 2; ++side) {
            for (std::int64_t jj = 0; jj < width; jj += step) {
                const Elements<const float> pb = b(side * half + c0 + jj, side, jj);
                const std::int64_t cols = std::min(step, width - jj);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    const Elements<float> c = buffer.from((ii - i0) * ldb + side * kGatedColumns + jj);
                    tiles(uk, wide, kc, a(ii), pb, b_next, c, ldb, k0 > 0, std::min<std::int64_t>(kMr, i1 - ii), cols);
                }
            }
        }
    };

    int item = queue.next();
    if (pl.shared_rows) {
        // The rows of src1, packed once by all threads, a panel of kMr rows to
        // an item, over the whole of K; then the blocks of columns over every
        // row, each packing its weights.
        const Elements<float> rows = work[l.shared];
        const std::int64_t row_items = ceil_div(g.m, kMr);
        for (; item < row_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_inputs);
            const std::int64_t r0 = item * kMr;
            pack_rows_f32(rows.from(r0 * g.k), g.x.from(r0 * g.xnb1), g.xnb1, std::min<std::int64_t>(kMr, g.m - r0), 0,
                          g.k, kMr);
        }
        queue.barrier();
        for (; item < row_items + chunks; item = queue.next()) {
            const std::int64_t c0 = (item - row_items) * kGatedColumns;
            const std::int64_t width = std::min(kGatedColumns, half - c0);
            for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, g.k - k0);
                {
                    const Phase_scope scope(Phase::pack_weights);
                    for (std::int64_t side = 0; side < 2; ++side) {
                        pack_weight_rows(packed_mine.from(side * kGatedColumns * kc), g, side * half + c0, width, k0,
                                         k0 + kc, kNr);
                    }
                }
                products(0, g.m, c0, width, k0, kc, kc * kNr,
                         [&](std::int64_t ii) { return Elements<const float>(rows.from(ii * g.k + k0 * kMr)); },
                         [&](std::int64_t, std::int64_t side, std::int64_t jj) {
                             return Elements<const float>(packed_mine.from(side * kGatedColumns * kc + (jj / kNr) * kc * kNr));
                         });
            }
            const Phase_scope scope(Phase::transform_out);
            gated_epilogue(g, chain, out, buffer, 0, g.m, c0, width);
        }
        return;
    }

    // The weights, packed once by all threads (or kept by the store); then the
    // blocks of rows, each packing its rows over the whole of K and going
    // through the blocks of columns of the half.
    const Elements<float> weights = kept != nullptr ? kept->singles() : work[l.shared];
    const bool fill = kept == nullptr || !kept->ready();
    const std::int64_t pack_items = fill ? weight_items(g) : 0;
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            pack_weights(weights, g, item);
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }
    const std::vector<Block> blocks = guided_blocks(1, g.m, kMr, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    for (; item < pack_items + units; item = queue.next()) {
        const Block & block = blocks[static_cast<std::size_t>(item - pack_items)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        // The rows of the block, a panel [K][kMr] of each kMr, stretch after
        // stretch of K: the stretch [k0, k0 + kc) of a panel at k0 * (i1 - i0
        // rounded up) + panel * kc * kMr.
        const std::int64_t rows = round_up(i1 - i0, kMr);
        {
            const Phase_scope scope(Phase::pack_inputs);
            for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, g.k - k0);
                pack_rows_f32(packed_mine.from(k0 * rows), g.x.from(i0 * g.xnb1), g.xnb1, i1 - i0, k0, k0 + kc, kMr);
            }
        }
        for (std::int64_t c0 = 0; c0 < half; c0 += kGatedColumns) {
            const std::int64_t width = std::min(kGatedColumns, half - c0);
            for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, g.k - k0);
                products(i0, i1, c0, width, k0, kc, g.k * kNr,
                         [&](std::int64_t ii) {
                             return Elements<const float>(packed_mine.from(k0 * rows + ((ii - i0) / kMr) * kc * kMr));
                         },
                         [&](std::int64_t j, std::int64_t, std::int64_t) {
                             return Elements<const float>(weights.from(j * g.k + k0 * kNr));
                         });
            }
            const Phase_scope scope(Phase::transform_out);
            gated_epilogue(g, chain, out, buffer, i0, i1, c0, width);
        }
    }
}

// The gated GELU's products on AMX's tiles: B's tiles packed once by all threads,
// or kept; then the blocks of rows, each packing its rows over the whole of K
// and going through the chunks of columns of the half as compute_gated() does,
// the products of x and of the gate into the thread's buffer. The columns of
// the gate begin a panel of B: half is a multiple of sixteen.
bool gated_on_amx(const Mul_mat & g) {
    return amx_product(g) && (g.n / 2) % kAmxTile == 0;
}

void compute_gated_amx(const ggml_compute_params * const params, const Mul_mat & g, const Epilogue & chain,
                       const ggml_tensor * const out) {
    const Split_form f = split_form(Split_use::linear);
    const Amx_plan pl = amx_plan(g, params->nth);
    const Amx_layout l = amx_layout(g, f, pl, true);
    Work_buffer work(params, l.layout);
    const std::shared_ptr<Packed> kept = amx_kept(g, f);
    const std::int64_t half = g.n / 2;
    const std::int64_t ldb = 2 * kGatedColumns;
    const std::int64_t kp = round_up(g.k, kAmxBlock);
    const std::int64_t kc_max = round_up(std::max(tuning().amx_kc, 1), kAmxBlock);
    const std::int64_t mc = std::min(round_up(g.m, kAmxBlock), round_up(std::max(tuning().amx_mc, 1), kAmxBlock));

    Work_queue queue(params, work);
    const Amx_weights w = amx_weights(g, kept, work[l.weights]);
    const Elements<std::uint16_t> rows = work[l.rows];
    const Elements<float> spare = work[l.spare];
    const Elements<float> buffer = work[l.buffer];
    int item = amx_pack_weights(params, g, f, w, queue, queue.next());
    const std::vector<Block> blocks = guided_blocks(1, g.m, kAmxBlock, mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    bool configured = false;
    for (; item < w.items + units; item = queue.next()) {
        if (!configured) {
            split_begin(f);
            configured = true;
        }
        const Block & block = blocks[static_cast<std::size_t>(item - w.items)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        amx_pack_rows_block(g, f, rows, i0, i1, 0, kp);
        for (std::int64_t c0 = 0; c0 < half; c0 += kGatedColumns) {
            const std::int64_t width = std::min(kGatedColumns, half - c0);
            for (std::int64_t k0 = 0; k0 < g.k; k0 += kc_max) {
                const std::int64_t kc = std::min(kc_max, kp - k0);
                for (std::int64_t side = 0; side < 2; ++side) {
                    const std::int64_t j0 = side * half + c0;
                    amx_products(
                        f, g, w.tiles, rows.from(k0 / kAmxBlock * (f.a_bytes / 2)), kp / kAmxBlock * (f.a_bytes / 2), spare,
                        i0, i1, j0, j0 + width, k0, kc, ldb, k0 > 0,
                        [&](std::int64_t ii, std::int64_t jj) {
                            return buffer.from((ii - i0) * ldb + side * kGatedColumns + (jj - j0));
                        },
                        [](std::int64_t, std::int64_t, std::int64_t, std::int64_t) {});
                }
            }
            const Phase_scope scope(Phase::transform_out);
            gated_epilogue(g, chain, out, buffer, i0, i1, c0, width);
        }
    }
    if (configured) {
        split_end(f);
    }
}

// Whether the memory of tensor t and [data, data + size) are apart.
bool apart(const ggml_tensor * const t, const void * const data, std::size_t size);

}  // namespace

bool mul_mat_gated_supported(const ggml_tensor * const node, const Epilogue & chain, const ggml_tensor * const out) {
    // Quantized, each half's columns begin a panel.
    if (quantized_product(geometry(node)) && node->ne[0] / 2 % kNr != 0) {
        return false;
    }
    // What the result is written to is not what the product reads: the
    // allocator may have given it the memory of the layer's input, which is
    // free once the layer has read it, as it would be without the fusion.
    const ggml_tensor * const w = node->src[0];
    const ggml_tensor * const x = node->src[1];
    if (!apart(out, w->data, ggml_nbytes(w)) || !apart(out, x->data, ggml_nbytes(x)) ||
        !apart(out, node->data, ggml_nbytes(node))) {
        return false;
    }
    for (int s = 0; s < chain.count; ++s) {
        // An operand of the chain: at most the size of the product.
        const ggml_tensor * const operand = chain.operands[static_cast<std::size_t>(s)];
        if (operand != nullptr && !apart(out, operand->data, ggml_nbytes(operand))) {
            return false;
        }
    }
    return holds<float>(out) && out->nb[0] == sizeof(float) && out->ne[0] == node->ne[0] / 2;
}

namespace {

bool apart(const ggml_tensor * const t, const void * const data, std::size_t size) {
    return !overlaps(t->data, ggml_nbytes(t), data, size);
}

}  // namespace

void mul_mat_compute_gated(const ggml_compute_params * const params, const ggml_tensor * const node,
                           const Epilogue & chain, const ggml_tensor * const out) {
    const Mul_mat g = geometry(node);
    if (quantized_product(g)) {
        compute_quantized_gated(params, g, chain, out);
        return;
    }
    if (gated_on_amx(g)) {
        compute_gated_amx(params, g, chain, out);
        return;
    }
    compute_gated(params, g, plan(g, params->nth), chain, out);
}

void mul_mat_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain) {
    Mul_mat g = geometry(node);
    if (chain.redirect != nullptr) {
        // The residual connection's tensor, of the same layout (fusion.h).
        g.y = elements_of<float>(chain.redirect);
    }
    if (quantized_product(g)) {
        compute_quantized(params, g, chain);
        return;
    }
    if (amx_product(g)) {
        compute_amx(params, g, chain);
        return;
    }
    const Plan pl = plan(g, params->nth);
    if (pl.shared_rows) {
        compute_shared_rows(params, g, pl, chain);
    }
    else {
        compute_shared_weights(params, g, pl, chain);
    }
}

}  // namespace sdasm
