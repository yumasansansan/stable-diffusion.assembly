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
#include <cstring>

#include "fusion.h"
#include "gemm.h"
#include "kernels.h"
#include "ops.h"
#include "phases.h"
#include "store.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

namespace {

struct Mul_mat {
    const ggml_tensor * weight;
    const char * w;  // [K, N], rows wnb1 apart
    ggml_type wtype;
    std::int64_t wnb1;
    const char * x;  // [K, M], rows xnb1 apart, the matrices one after another
    std::int64_t xnb1;
    char * y;  // [N, M]
    std::int64_t ynb1;
    std::int64_t m, n, k;
    std::int64_t ne1, ne2;  // the rows and matrices of the output, for the chain
};

Mul_mat geometry(const ggml_tensor * node) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    Mul_mat g{};
    g.weight = w;
    g.w = static_cast<const char *>(w->data);
    g.wtype = w->type;
    g.wnb1 = static_cast<std::int64_t>(w->nb[1]);
    g.x = static_cast<const char *>(x->data);
    g.xnb1 = static_cast<std::int64_t>(x->nb[1]);
    g.y = static_cast<char *>(node->data);
    g.ynb1 = static_cast<std::int64_t>(node->nb[1]);
    g.k = w->ne[0];
    g.n = w->ne[1];
    g.m = x->ne[1] * x->ne[2] * x->ne[3];
    g.ne1 = node->ne[1];
    g.ne2 = node->ne[2];
    return g;
}

struct Plan {
    // The rows of src1 packed once, by all threads together, and each thread
    // packing the weights of its block of columns: for few rows, when the
    // weights are the large operand. Otherwise the weights are packed once and
    // each thread packs the rows of its block.
    bool shared_rows;
    std::int64_t mc, nc, kc;
    std::int64_t mblocks, nblocks;
    std::size_t shared_bytes, thread_bytes;
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
        pl.shared_bytes = align_up(static_cast<std::size_t>(round_up(g.m, kMr) * g.k) * sizeof(float));
        pl.thread_bytes = align_up(static_cast<std::size_t>(pl.nc * pl.kc) * sizeof(float));
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
    pl.shared_bytes = align_up(static_cast<std::size_t>(round_up(g.n, kNr) * g.k) * sizeof(float));
    pl.thread_bytes = align_up(static_cast<std::size_t>(pl.mc * pl.kc) * sizeof(float));
    return pl;
}

}  // namespace

bool mul_mat_supported(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    // The weights are one matrix, and the rows of the other operand and of the
    // result lie one after another, matrix after matrix. Products too small to
    // pay for packing the weights are left to ggml.
    return (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32) && w->ne[2] == 1 && w->ne[3] == 1 &&
           w->nb[0] == ggml_type_size(w->type) && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) &&
           node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) &&
           w->extra == nullptr && x->ne[1] * x->ne[2] * x->ne[3] >= 16 && w->ne[0] >= 16;
}

namespace {

std::size_t gated_thread_bytes(const Mul_mat & g, const Plan & pl);

}  // namespace

namespace {

bool amx_product(const Mul_mat & g);
std::size_t amx_work_size(const Mul_mat & g, int n_threads);

}  // namespace

std::size_t mul_mat_work_size(const ggml_tensor * node, int n_threads) {
    const Mul_mat g = geometry(node);
    const Plan pl = plan(g, n_threads);
    // Enough for the gated GELU after it too (fusion.h), which needs more of
    // each thread; and for the products on AMX's tiles, when they go there.
    const std::size_t thread_bytes = std::max(pl.thread_bytes, gated_thread_bytes(g, pl));
    const std::size_t bytes = kAlign + pl.shared_bytes + thread_bytes * static_cast<std::size_t>(n_threads);
    return amx_product(g) ? std::max(bytes, amx_work_size(g, n_threads)) : bytes;
}

namespace {

// The weights, transposed, in panels of kNr columns over the whole of K,
// [panel][k][kNr]: the panels of B, which the store keeps (store.h).
std::size_t weights_bytes(const Mul_mat & g) {
    return static_cast<std::size_t>(round_up(g.n, kNr) * g.k) * sizeof(float);
}

constexpr std::int64_t kPanelsPerItem = 2;

std::int64_t weight_items(const Mul_mat & g) {
    return ceil_div(ceil_div(g.n, kNr), kPanelsPerItem);
}

void pack_weights(float * weights, const Mul_mat & g, std::int64_t item) {
    const std::int64_t r0 = item * kPanelsPerItem * kNr;
    const std::int64_t count = std::min<std::int64_t>(kPanelsPerItem * kNr, g.n - r0);
    float * dst = weights + r0 * g.k;
    if (g.wtype == GGML_TYPE_F16) {
        pack_rows_f16(dst, reinterpret_cast<const std::uint16_t *>(g.w + r0 * g.wnb1), g.wnb1, count, 0, g.k, kNr);
    }
    else {
        pack_rows_f32(dst, reinterpret_cast<const float *>(g.w + r0 * g.wnb1), g.wnb1, count, 0, g.k, kNr);
    }
}

void apply_chain(const Epilogue & chain, const Mul_mat & g, float * out, std::int64_t ii, std::int64_t count, std::int64_t jj,
                 std::int64_t cols) {
    for (std::int64_t r = 0; r < count; ++r) {
        const std::int64_t m = ii + r;
        chain.apply(reinterpret_cast<float *>(reinterpret_cast<char *>(out) + r * g.ynb1), cols, jj, m % g.ne1,
                    (m / g.ne1) % g.ne2, m / (g.ne1 * g.ne2));
    }
}

void compute_shared_rows(const ggml_compute_params * params, const Mul_mat & g, const Plan & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    float * rows = reinterpret_cast<float *>(base);
    float * weights = reinterpret_cast<float *>(base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith));
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    // The panels of the weights the store keeps, packed with the rows when
    // they are not yet; without them, each block packs its own as it goes.
    const std::shared_ptr<Packed> kept = packed(g.weight, Layout::linear_columns, weights_bytes(g));
    const bool fill = kept != nullptr && !kept->ready();

    Work_queue queue(params);
    // The rows of src1, a panel of kMr rows to an item, over the whole of K;
    // then the weights, when the store is to keep them.
    const std::int64_t row_items = ceil_div(g.m, kMr);
    const std::int64_t pack_items = row_items + (fill ? weight_items(g) : 0);
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        if (item >= row_items) {
            const Phase_scope scope(Phase::pack_weights);
            pack_weights(kept->data, g, item - row_items);
            continue;
        }
        const Phase_scope scope(Phase::pack_inputs);
        const std::int64_t r0 = item * kMr;
        pack_rows_f32(rows + r0 * g.k, reinterpret_cast<const float *>(g.x + r0 * g.xnb1), g.xnb1,
                      std::min<std::int64_t>(kMr, g.m - r0), 0, g.k, kMr);
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
                if (g.wtype == GGML_TYPE_F16) {
                    pack_rows_f16(weights, reinterpret_cast<const std::uint16_t *>(g.w + j0 * g.wnb1), g.wnb1, j1 - j0, k0,
                                  k0 + kc, kNr);
                }
                else {
                    pack_rows_f32(weights, reinterpret_cast<const float *>(g.w + j0 * g.wnb1), g.wnb1, j1 - j0, k0,
                                  k0 + kc, kNr);
                }
            }
            const bool last = k0 + kc == g.k;
            const Phase_scope scope(Phase::product);
            const std::int64_t b_next = (kept != nullptr ? g.k : kc) * kNr * static_cast<std::int64_t>(sizeof(float));
            for (std::int64_t jj = j0; jj < j1; jj += step) {
                const float * b = kept != nullptr ? kept->data + jj * g.k + k0 * kNr : weights + (jj - j0) * kc;
                const std::int64_t cols = std::min(step, j1 - jj);
                for (std::int64_t ii = 0; ii < g.m; ii += kMr) {
                    const float * a = rows + ii * g.k + k0 * kMr;
                    float * out = reinterpret_cast<float *>(g.y + ii * g.ynb1) + jj;
                    const std::int64_t count = std::min<std::int64_t>(kMr, g.m - ii);
                    tiles(uk, wide, kc, a, b, b_next, out, g.ynb1, k0 > 0 || chain.from_redirect, count, cols);
                    if (last && !chain.empty()) {
                        apply_chain(chain, g, out, ii, count, jj, cols);
                    }
                }
            }
        }
    }
}

void compute_shared_weights(const ggml_compute_params * params, const Mul_mat & g, const Plan & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    // The panels of the weights: the store's, packed here the first time, or
    // the work buffer's, packed here every time.
    const std::shared_ptr<Packed> kept = packed(g.weight, Layout::linear_columns, weights_bytes(g));
    float * weights = kept != nullptr ? kept->data : reinterpret_cast<float *>(base);
    const bool fill = kept == nullptr || !kept->ready();
    float * rows = reinterpret_cast<float *>(base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith));
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const std::int64_t b_next = g.k * kNr * static_cast<std::int64_t>(sizeof(float));

    Work_queue queue(params);
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
                pack_rows_f32(rows, reinterpret_cast<const float *>(g.x + i0 * g.xnb1), g.xnb1, i1 - i0, k0, k0 + kc, kMr);
            }
            const Phase_scope scope(Phase::product);
            for (std::int64_t jj = j0; jj < j1; jj += step) {
                const float * b = weights + jj * g.k + k0 * kNr;
                const std::int64_t cols = std::min(step, j1 - jj);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    const float * a = rows + ((ii - i0) / kMr) * kc * kMr;
                    float * out = reinterpret_cast<float *>(g.y + ii * g.ynb1) + jj;
                    const std::int64_t count = std::min<std::int64_t>(kMr, i1 - ii);
                    tiles(uk, wide, kc, a, b, b_next, out, g.ynb1, k0 > 0 || chain.from_redirect, count, cols);
                    if (k0 + kc == g.k && !chain.empty()) {
                        apply_chain(chain, g, out, ii, count, jj, cols);
                    }
                }
            }
        }
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

// The bytes of a panel of sixteen columns of B over the whole of K, and of all
// of B: the panels of the columns rounded up to 32.
std::int64_t amx_panel_bytes(const Mul_mat & g, const Split_form & f) {
    return round_up(g.k, kAmxBlock) / kAmxBlock * f.b_bytes;
}

std::size_t amx_weights_bytes(const Mul_mat & g, const Split_form & f) {
    return static_cast<std::size_t>(round_up(g.n, kAmxBlock) / kAmxTile * amx_panel_bytes(g, f));
}

struct Amx_plan {
    std::int64_t kc;  // a multiple of 32
    std::int64_t mc;  // a multiple of 32
    std::int64_t nc;  // a multiple of 32
    std::int64_t nblocks;
    std::size_t rows_bytes;
    std::size_t thread_bytes;
};

Amx_plan amx_plan(const Mul_mat & g, const Split_form & f, int n_threads) {
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
    pl.rows_bytes = align_up(static_cast<std::size_t>(pl.mc / kAmxTile * (pl.kc / kAmxBlock) * f.a_bytes));
    pl.thread_bytes = pl.rows_bytes + align_up(kAmxBlock * kAmxBlock * sizeof(float));
    return pl;
}

struct Gated_amx_plan;
std::size_t gated_amx_thread_bytes(const Mul_mat & g, const Split_form & f);

// The work buffer of either product on AMX, the plain and the gated.
std::size_t amx_work_size(const Mul_mat & g, int n_threads) {
    const Split_form f = split_form(Split_use::linear);
    const Amx_plan pl = amx_plan(g, f, n_threads);
    const std::size_t thread_bytes = std::max(pl.thread_bytes, gated_amx_thread_bytes(g, f));
    return kAlign + align_up(amx_weights_bytes(g, f)) + thread_bytes * static_cast<std::size_t>(n_threads);
}

// The tiles of B: the store's, or the work buffer's, packed by all threads
// together, two panels of sixteen columns to an item of the work queue, when
// they are not ready; a panel past the last column is zeros.
struct Amx_weights {
    std::shared_ptr<Packed> kept;
    std::uint16_t * tiles;
    std::int64_t items;
};

Amx_weights amx_weights(const Mul_mat & g, const Split_form & f, char * base) {
    Amx_weights w;
    w.kept = packed(g.weight, f.fp16 ? Layout::amx_linear_fp16 : Layout::amx_linear, amx_weights_bytes(g, f));
    w.tiles = reinterpret_cast<std::uint16_t *>(w.kept != nullptr ? reinterpret_cast<char *>(w.kept->data) : base);
    w.items = w.kept == nullptr || !w.kept->ready() ? round_up(g.n, kAmxBlock) / kAmxBlock : 0;
    return w;
}

std::uint16_t * at_bytes(std::uint16_t * p, std::int64_t bytes) {
    return p + bytes / 2;
}

// The items of the packing of B, as the queue hands them out from `item`, then
// the wait for every thread; the first item after.
int amx_pack_weights(const ggml_compute_params * params, const Mul_mat & g, const Split_form & f, const Amx_weights & w,
                     Work_queue & queue, int item) {
    if (w.items == 0) {
        return item;
    }
    const std::int64_t panel_bytes = amx_panel_bytes(g, f);
    for (; item < w.items; item = queue.next()) {
        const Phase_scope scope(Phase::pack_weights);
        for (std::int64_t p = 0; p < 2; ++p) {
            const std::int64_t c0 = item * kAmxBlock + p * kAmxTile;
            std::uint16_t * dst = at_bytes(w.tiles, c0 / kAmxTile * panel_bytes);
            if (c0 >= g.n) {
                std::memset(dst, 0, static_cast<std::size_t>(panel_bytes));
                continue;
            }
            f.pack_weights(dst, reinterpret_cast<const std::uint16_t *>(g.w + c0 * g.wnb1), g.wnb1,
                           std::min(kAmxTile, g.n - c0), g.k);
        }
    }
    queue.barrier();
    if (params->ith == 0) {
        publish(w.kept);
    }
    return item;
}

// The rows [i0, i1) of src1 over the stretch [k0, k0 + kc) of K, kc a multiple
// of 32, into the thread's tiles, a tile of sixteen rows tile_bytes after the
// last, rounded up to a pair with zeros.
void amx_pack_rows_block(const Mul_mat & g, const Split_form & f, std::uint16_t * rows, std::int64_t i0, std::int64_t i1,
                         std::int64_t k0, std::int64_t kc) {
    const Phase_scope scope(Phase::pack_inputs);
    const std::int64_t tile_bytes = kc / kAmxBlock * f.a_bytes;
    for (std::int64_t t = 0; t < round_up(i1 - i0, kAmxBlock) / kAmxTile; ++t) {
        const std::int64_t r0 = i0 + t * kAmxTile;
        std::uint16_t * dst = at_bytes(rows, t * tile_bytes);
        if (r0 >= i1) {
            std::memset(dst, 0, static_cast<std::size_t>(tile_bytes));
            continue;
        }
        f.pack_rows(dst, reinterpret_cast<const float *>(g.x + r0 * g.xnb1) + k0, g.xnb1, std::min(kAmxTile, i1 - r0),
                    std::min(kc, g.k - k0));
    }
}

// The products of the packed rows [i0, i1) with the columns [j0, j1) of B over
// the stretch at k0 of kc: rows, the rows' tiles of the stretch, a tile of
// sixteen rows tile_bytes after the last; the block of C at rows ii and
// columns jj written at out(ii, jj), rows ldc bytes apart; then done(ii, rows,
// jj, cols).
template <class Out, class Done>
void amx_products(const Split_form & f, const Mul_mat & g, const std::uint16_t * tiles, const std::uint16_t * rows,
                  std::int64_t tile_bytes, float * spare, std::int64_t i0, std::int64_t i1, std::int64_t j0, std::int64_t j1,
                  std::int64_t k0, std::int64_t kc, std::int64_t ldc, bool accumulate, Out out, Done done) {
    const Phase_scope scope(Phase::product);
    const std::int64_t panel_bytes = amx_panel_bytes(g, f);
    for (std::int64_t jj = j0; jj < j1; jj += kAmxBlock) {
        const std::uint16_t * b = tiles + (jj / kAmxTile * panel_bytes + k0 / kAmxBlock * f.b_bytes) / 2;
        const std::int64_t cols = std::min(kAmxBlock, j1 - jj);
        for (std::int64_t ii = i0; ii < i1; ii += kAmxBlock) {
            const std::uint16_t * a = rows + (ii - i0) / kAmxTile * tile_bytes / 2;
            const std::int64_t count = std::min(kAmxBlock, i1 - ii);
            split_block(f, kc / kAmxBlock, a, tile_bytes, b, panel_bytes, out(ii, jj), ldc, accumulate, count, cols, spare);
            done(ii, count, jj, cols);
        }
    }
}

void compute_amx(const ggml_compute_params * params, const Mul_mat & g, const Epilogue & chain) {
    const Split_form f = split_form(Split_use::linear);
    const Amx_plan pl = amx_plan(g, f, params->nth);
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    const Amx_weights w = amx_weights(g, f, base);
    char * mine = base + align_up(amx_weights_bytes(g, f)) + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    auto * rows = reinterpret_cast<std::uint16_t *>(mine);
    auto * spare = reinterpret_cast<float *>(mine + pl.rows_bytes);

    Work_queue queue(params);
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
                f, g, w.tiles, rows, kc / kAmxBlock * f.a_bytes, spare, i0, i1, j0, j1, k0, kc, g.ynb1,
                k0 > 0 || chain.from_redirect,
                [&](std::int64_t ii, std::int64_t jj) { return reinterpret_cast<float *>(g.y + ii * g.ynb1) + jj; },
                [&](std::int64_t ii, std::int64_t count, std::int64_t jj, std::int64_t cols) {
                    if (last && !chain.empty()) {
                        apply_chain(chain, g, reinterpret_cast<float *>(g.y + ii * g.ynb1) + jj, ii, count, jj, cols);
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
constexpr std::int64_t kGatedColumns = 64;

std::size_t gated_thread_bytes(const Mul_mat & g, const Plan & pl) {
    const std::int64_t rows = pl.shared_rows ? round_up(g.m, kMr) : pl.mc;
    // The packed weights of both halves of the columns of a block, with shared
    // rows; the rows of the block, packed over the whole of K, otherwise.
    const std::size_t packed = pl.shared_rows ? static_cast<std::size_t>(2 * kGatedColumns * pl.kc)
                                              : static_cast<std::size_t>(pl.mc * g.k);
    const std::size_t buffer = static_cast<std::size_t>(rows * 2 * kGatedColumns);
    return align_up(packed * sizeof(float)) + align_up(buffer * sizeof(float));
}

void gated_epilogue(const Mul_mat & g, const Epilogue & chain, const ggml_tensor * out, float * buffer, std::int64_t i0,
                    std::int64_t i1, std::int64_t c0, std::int64_t width) {
    const std::int64_t half = g.n / 2;
    const auto onb1 = static_cast<std::int64_t>(out->nb[1]);
    for (std::int64_t m = i0; m < i1; ++m) {
        float * x = buffer + (m - i0) * 2 * kGatedColumns;
        float * gate = x + kGatedColumns;
        if (!chain.empty()) {
            const std::int64_t r1 = m % g.ne1;
            const std::int64_t r2 = (m / g.ne1) % g.ne2;
            const std::int64_t r3 = m / (g.ne1 * g.ne2);
            chain.apply(x, width, c0, r1, r2, r3);
            chain.apply(gate, width, half + c0, r1, r2, r3);
        }
        gated_gelu_run(reinterpret_cast<float *>(static_cast<char *>(out->data) + m * onb1) + c0, x, gate, width);
    }
}

void compute_gated(const ggml_compute_params * params, const Mul_mat & g, const Plan & pl, const Epilogue & chain,
                   const ggml_tensor * out) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    const std::int64_t half = g.n / 2;
    const std::size_t thread_bytes = std::max(pl.thread_bytes, gated_thread_bytes(g, pl));
    char * mine = base + pl.shared_bytes + thread_bytes * static_cast<std::size_t>(params->ith);
    const std::int64_t rows_max = pl.shared_rows ? round_up(g.m, kMr) : pl.mc;
    const std::size_t packed_bytes = align_up((pl.shared_rows ? static_cast<std::size_t>(2 * kGatedColumns * pl.kc)
                                                              : static_cast<std::size_t>(pl.mc * g.k)) * sizeof(float));
    auto * packed_mine = reinterpret_cast<float *>(mine);
    auto * buffer = reinterpret_cast<float *>(mine + packed_bytes);
    const std::int64_t ldb = 2 * kGatedColumns * static_cast<std::int64_t>(sizeof(float));
    (void) rows_max;
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const std::int64_t chunks = ceil_div(half, kGatedColumns);

    // The tiles of a block of rows [i0, i1) and columns [c0, c0 + width) of
    // the half, for a stretch [k0, k0 + kc) of K: a(ii) the panel of rows, b(jj)
    // the panel of columns jj of the product, the next b_next bytes after it.
    const auto products = [&](std::int64_t i0, std::int64_t i1, std::int64_t c0, std::int64_t width, std::int64_t k0,
                              std::int64_t kc, std::int64_t b_next, const auto & a, const auto & b) {
        const Phase_scope scope(Phase::product);
        for (int side = 0; side < 2; ++side) {
            for (std::int64_t jj = 0; jj < width; jj += step) {
                const float * pb = b(side * half + c0 + jj, side, jj);
                const std::int64_t cols = std::min(step, width - jj);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    float * c = buffer + (ii - i0) * 2 * kGatedColumns + side * kGatedColumns + jj;
                    tiles(uk, wide, kc, a(ii), pb, b_next, c, ldb, k0 > 0, std::min<std::int64_t>(kMr, i1 - ii), cols);
                }
            }
        }
    };

    Work_queue queue(params);
    int item = queue.next();
    if (pl.shared_rows) {
        // The rows of src1, packed once by all threads, a panel of kMr rows to
        // an item, over the whole of K; then the blocks of columns over every
        // row, each packing its weights.
        float * rows = reinterpret_cast<float *>(base);
        const std::int64_t row_items = ceil_div(g.m, kMr);
        for (; item < row_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_inputs);
            const std::int64_t r0 = item * kMr;
            pack_rows_f32(rows + r0 * g.k, reinterpret_cast<const float *>(g.x + r0 * g.xnb1), g.xnb1,
                          std::min<std::int64_t>(kMr, g.m - r0), 0, g.k, kMr);
        }
        queue.barrier();
        for (; item < row_items + chunks; item = queue.next()) {
            const std::int64_t c0 = (item - row_items) * kGatedColumns;
            const std::int64_t width = std::min(kGatedColumns, half - c0);
            for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, g.k - k0);
                {
                    const Phase_scope scope(Phase::pack_weights);
                    for (int side = 0; side < 2; ++side) {
                        const std::int64_t j = side * half + c0;
                        float * dst = packed_mine + side * kGatedColumns * kc;
                        if (g.wtype == GGML_TYPE_F16) {
                            pack_rows_f16(dst, reinterpret_cast<const std::uint16_t *>(g.w + j * g.wnb1), g.wnb1, width, k0,
                                          k0 + kc, kNr);
                        }
                        else {
                            pack_rows_f32(dst, reinterpret_cast<const float *>(g.w + j * g.wnb1), g.wnb1, width, k0, k0 + kc,
                                          kNr);
                        }
                    }
                }
                products(0, g.m, c0, width, k0, kc, kc * kNr * static_cast<std::int64_t>(sizeof(float)),
                         [&](std::int64_t ii) { return rows + ii * g.k + k0 * kMr; },
                         [&](std::int64_t, int side, std::int64_t jj) {
                             return packed_mine + side * kGatedColumns * kc + (jj / kNr) * kc * kNr;
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
    const std::shared_ptr<Packed> kept = packed(g.weight, Layout::linear_columns, weights_bytes(g));
    float * weights = kept != nullptr ? kept->data : reinterpret_cast<float *>(base);
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
                pack_rows_f32(packed_mine + k0 * rows, reinterpret_cast<const float *>(g.x + i0 * g.xnb1), g.xnb1, i1 - i0,
                              k0, k0 + kc, kMr);
            }
        }
        for (std::int64_t c0 = 0; c0 < half; c0 += kGatedColumns) {
            const std::int64_t width = std::min(kGatedColumns, half - c0);
            for (std::int64_t k0 = 0; k0 < g.k; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, g.k - k0);
                products(i0, i1, c0, width, k0, kc, g.k * kNr * static_cast<std::int64_t>(sizeof(float)),
                         [&](std::int64_t ii) { return packed_mine + k0 * rows + ((ii - i0) / kMr) * kc * kMr; },
                         [&](std::int64_t j, int, std::int64_t) { return weights + j * g.k + k0 * kNr; });
            }
            const Phase_scope scope(Phase::transform_out);
            gated_epilogue(g, chain, out, buffer, i0, i1, c0, width);
        }
    }
}
}  // namespace

namespace {

// The gated GELU's products on AMX's tiles: B's tiles packed once by all threads,
// or kept; then the blocks of rows, each packing its rows over the whole of K
// and going through the chunks of columns of the half as compute_gated() does,
// the products of x and of the gate into the thread's buffer. The columns of
// the gate begin a panel of B: half is a multiple of sixteen.
bool gated_on_amx(const Mul_mat & g) {
    return amx_product(g) && (g.n / 2) % kAmxTile == 0;
}

struct Gated_amx_plan {
    std::int64_t mc;
    std::size_t rows_bytes, spare_bytes, buffer_bytes, thread_bytes;
};

Gated_amx_plan gated_amx_plan(const Mul_mat & g, const Split_form & f) {
    Gated_amx_plan pl{};
    pl.mc = std::min(round_up(g.m, kAmxBlock), round_up(std::max(tuning().amx_mc, 1), kAmxBlock));
    pl.rows_bytes = align_up(static_cast<std::size_t>(pl.mc / kAmxTile * (round_up(g.k, kAmxBlock) / kAmxBlock) * f.a_bytes));
    pl.spare_bytes = align_up(kAmxBlock * kAmxBlock * sizeof(float));
    pl.buffer_bytes = align_up(static_cast<std::size_t>(pl.mc * 2 * kGatedColumns) * sizeof(float));
    pl.thread_bytes = pl.rows_bytes + pl.spare_bytes + pl.buffer_bytes;
    return pl;
}

std::size_t gated_amx_thread_bytes(const Mul_mat & g, const Split_form & f) {
    return gated_amx_plan(g, f).thread_bytes;
}

void compute_gated_amx(const ggml_compute_params * params, const Mul_mat & g, const Epilogue & chain, const ggml_tensor * out) {
    const Split_form f = split_form(Split_use::linear);
    const Gated_amx_plan pl = gated_amx_plan(g, f);
    const std::int64_t half = g.n / 2;
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    const Amx_weights w = amx_weights(g, f, base);
    char * mine = base + align_up(amx_weights_bytes(g, f)) + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    auto * rows = reinterpret_cast<std::uint16_t *>(mine);
    auto * spare = reinterpret_cast<float *>(mine + pl.rows_bytes);
    auto * buffer = reinterpret_cast<float *>(mine + pl.rows_bytes + pl.spare_bytes);
    const std::int64_t ldb = 2 * kGatedColumns * static_cast<std::int64_t>(sizeof(float));
    const std::int64_t kp = round_up(g.k, kAmxBlock);
    const std::int64_t kc_max = round_up(std::max(tuning().amx_kc, 1), kAmxBlock);

    Work_queue queue(params);
    int item = amx_pack_weights(params, g, f, w, queue, queue.next());
    const std::vector<Block> blocks = guided_blocks(1, g.m, kAmxBlock, pl.mc, params->nth);
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
                        f, g, w.tiles, rows + k0 / kAmxBlock * f.a_bytes / 2, kp / kAmxBlock * f.a_bytes, spare, i0, i1,
                        j0, j0 + width, k0, kc, ldb, k0 > 0,
                        [&](std::int64_t ii, std::int64_t jj) {
                            return buffer + (ii - i0) * 2 * kGatedColumns + side * kGatedColumns + (jj - j0);
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

}  // namespace

bool mul_mat_gated_supported(const ggml_tensor * node, const Epilogue & chain, const ggml_tensor * out) {
    // What the result is written to is not what the product reads: the
    // allocator may have given it the memory of the layer's input, which is
    // free once the layer has read it, as it would be without the fusion.
    const auto apart = [](const ggml_tensor * a, const void * data, std::size_t size) {
        const auto * a0 = static_cast<const char *>(a->data);
        const auto * b0 = static_cast<const char *>(data);
        return a0 + ggml_nbytes(a) <= b0 || b0 + size <= a0;
    };
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    if (!apart(out, w->data, ggml_nbytes(w)) || !apart(out, x->data, ggml_nbytes(x)) ||
        !apart(out, node->data, ggml_nbytes(node))) {
        return false;
    }
    for (int s = 0; s < chain.count; ++s) {
        // An operand of the chain: at most the size of the product.
        if (chain.steps[s].data != nullptr && !apart(out, chain.steps[s].data, ggml_nbytes(node))) {
            return false;
        }
    }
    return out->nb[0] == sizeof(float) && out->ne[0] == node->ne[0] / 2;
}

void mul_mat_compute_gated(const ggml_compute_params * params, ggml_tensor * node, const Epilogue & chain,
                           const ggml_tensor * out) {
    const Mul_mat g = geometry(node);
    if (gated_on_amx(g)) {
        compute_gated_amx(params, g, chain, out);
        return;
    }
    compute_gated(params, g, plan(g, params->nth), chain, out);
}

void mul_mat_compute(const ggml_compute_params * params, ggml_tensor * node, const Epilogue & chain) {
    Mul_mat g = geometry(node);
    if (chain.redirect != nullptr) {
        // The residual connection's tensor, of the same layout (fusion.h).
        g.y = static_cast<char *>(chain.redirect->data);
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
