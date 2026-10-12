// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The direct convolution, GGML_OP_CONV_2D, as a matrix product of the weights
// and the columns of the image (gemm.h): for each image, the output of OC
// channels by OH * OW positions is the product of the weights, OC rows of
// K = IC * KH * KW, and the columns, K rows of OH * OW, whose row k = (ic, kh,
// kw) is the input of channel ic shifted by kh and kw to every output position.
// ggml computes the same through im2col, which writes the columns out whole --
// nine copies of the image for a kernel of three by three, half a gigabyte for
// a layer of the VAE's decoder at 512 by 512 -- rounded to FP16 when the weights
// are, and multiplies them by the weights after. Here the columns are never
// written out: each thread packs the stretch of them that its block of the
// product needs, from the image, into a panel the size of the cache, and in
// single precision, so that a product of FP16 weights and the image is that of
// the weights and the image as it is, where ggml's is that of the weights and
// the image rounded to FP16.
//
// The weights are packed once, by all threads together, into panels of kMr
// output channels over the whole of K, widened to single precision when they
// are FP16. The output is written as ggml lays it out, [OW, OH, OC, N], each
// row of C being a channel of an image; the bias the graph adds after is not
// this operation's.
//
// On AMX, the products of FP16 weights go to the tiles (gemm.h): A is the
// weights, in tiles of sixteen output channels by 32 of K, packed once by all
// threads together; B the columns, in tiles of pairs of K by sixteen
// positions, made from the panels pack_columns() writes in single precision
// (sdasm_amx_pack_panel()), both split in two parts of bfloat16. The blocks
// are those of the weights shared, for every image.

#include <algorithm>
#include <array>
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

// The strides are in elements.
struct Conv {
    const ggml_tensor * kernel;
    Elements<const float> x;  // [IW, IH, IC, N]
    std::int64_t iw, ih, ic, n;
    std::int64_t xnb1, xnb2, xnb3;
    ggml_type wtype;
    Elements<const float> w32;          // [KW, KH, IC, OC], of single precision
    Elements<const std::uint16_t> w16;  // or FP16
    std::int64_t kw, kh, oc;
    std::int64_t wnb3;
    Elements<float> y;  // [OW, OH, OC, N]
    std::int64_t ow, oh;
    std::int64_t ynb2, ynb3;
    int s0, s1, p0, p1, d0, d1;
    std::int64_t k;  // ic * kh * kw
    std::int64_t p;  // ow * oh
};

// Of a node that conv2d_supported() takes.
Conv geometry(const ggml_tensor * const node) {
    const ggml_tensor * const kernel = node->src[0];
    const ggml_tensor * const input = node->src[1];
    Conv c{};
    c.kernel = kernel;
    c.x = elements_of<const float>(input);
    c.iw = input->ne[0];
    c.ih = input->ne[1];
    c.ic = input->ne[2];
    c.n = input->ne[3];
    c.xnb1 = stride_of<float>(input, 1);
    c.xnb2 = stride_of<float>(input, 2);
    c.xnb3 = stride_of<float>(input, 3);
    c.wtype = kernel->type;
    if (kernel->type == GGML_TYPE_F16) {
        c.w16 = elements_of<const std::uint16_t>(kernel);
        c.wnb3 = stride_of<std::uint16_t>(kernel, 3);
    }
    else {
        c.w32 = elements_of<const float>(kernel);
        c.wnb3 = stride_of<float>(kernel, 3);
    }
    c.kw = kernel->ne[0];
    c.kh = kernel->ne[1];
    c.oc = kernel->ne[3];
    c.y = elements_of<float>(node);
    c.ow = node->ne[0];
    c.oh = node->ne[1];
    c.ynb2 = stride_of<float>(node, 2);
    c.ynb3 = stride_of<float>(node, 3);
    c.s0 = node->op_params[0];
    c.s1 = node->op_params[1];
    c.p0 = node->op_params[2];
    c.p1 = node->op_params[3];
    c.d0 = node->op_params[4];
    c.d1 = node->op_params[5];
    c.k = c.ic * c.kh * c.kw;
    c.p = c.ow * c.oh;
    return c;
}

// The weights of output channels [r0, r0 + rows) over [k0, k1) of K into
// panels of p (gemm.h).
void pack_weight_rows(Elements<float> dst, const Conv & c, std::int64_t r0, std::int64_t rows, std::int64_t k0,
                      std::int64_t k1, int p) {
    if (c.wtype == GGML_TYPE_F16) {
        pack_rows_f16(dst, c.w16.from(r0 * c.wnb3), c.wnb3, rows, k0, k1, p);
    }
    else {
        pack_rows_f32(dst, c.w32.from(r0 * c.wnb3), c.wnb3, rows, k0, k1, p);
    }
}

// The output of channel `channel` of image n from position `position` on, and
// the chain after the convolution over `rows` channels from there, `cols`
// positions of each.
void apply_chain_positions(const Epilogue & chain, const Conv & c, std::int64_t n, std::int64_t channel, std::int64_t rows,
                           std::int64_t position, std::int64_t cols) {
    for (std::int64_t r = 0; r < rows; ++r) {
        chain.apply_positions(c.y.part(n * c.ynb3 + (channel + r) * c.ynb2 + position, cols), position, channel + r, n);
    }
}

struct Plan {
    // The columns of every image packed once, by all threads together, and
    // each thread packing the weights of its block of channels: for the small
    // images of a model's deepest layers, whose weights are the large operand.
    // Otherwise the weights are packed once and each thread packs the columns
    // of its block of positions.
    bool shared_columns;
    std::int64_t chunk;  // with shared columns: the stretch of K packed at a time
    std::int64_t mc, nc, kc;
    std::int64_t mblocks, nblocks;
};

std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

std::int64_t round_up(std::int64_t a, std::int64_t b) {
    return ceil_div(a, b) * b;
}

Plan plan(const Conv & c, int n_threads) {
    const Tuning & t = tuning();
    Plan pl{};
    pl.kc = std::min<std::int64_t>(c.k, t.conv_kc);
    pl.shared_columns = c.n * c.p <= t.conv_shared_columns && c.p % kNr == 0;
    if (pl.shared_columns) {
        // Blocks of channels enough for every thread to take several.
        const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
        pl.mc = std::clamp<std::int64_t>(c.oc / wanted / kMr * kMr, kMr, round_up(t.conv_mc, kMr));
        pl.mblocks = ceil_div(c.oc, pl.mc);
        pl.nc = c.n * c.p;
        pl.nblocks = 1;
        // The columns of a stretch of K at a time, as many as fit a part of the
        // last-level cache that every core reads them from: the columns of the
        // whole of K can be twenty times its size, and each block of channels
        // reads them all.
        const std::int64_t fit = static_cast<std::int64_t>(t.conv_columns_bytes) / (c.n * c.p * 4);
        pl.chunk = std::clamp<std::int64_t>(fit / pl.kc * pl.kc, pl.kc, round_up(c.k, pl.kc));
        return pl;
    }
    // As many output channels in a block as fit the cache beside the columns,
    // in blocks of about the same size.
    pl.mblocks = ceil_div(c.oc, t.conv_mc);
    pl.mc = round_up(ceil_div(c.oc, pl.mblocks), kMr);
    pl.nc = round_up(t.conv_nc, kNr);
    pl.nblocks = ceil_div(c.p, pl.nc);
    // Enough blocks for every thread to take several, so that the last ones
    // taken do not leave the others waiting long.
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    while (c.n * pl.mblocks * pl.nblocks < wanted && pl.nc > 32) {
        pl.nc /= 2;
        pl.nblocks = ceil_div(c.p, pl.nc);
    }
    for (std::int64_t split = pl.mblocks + 1; c.n * pl.mblocks * pl.nblocks < wanted && pl.mc > 8 * kMr; ++split) {
        pl.mc = std::min(pl.mc, round_up(ceil_div(c.oc, split), kMr));
        pl.mblocks = ceil_div(c.oc, pl.mc);
    }
    return pl;
}

// The weights in panels of kMr output channels over the whole of K,
// [panel][k][kMr]: the panels of A, which the store keeps (store.h).
std::int64_t weights_count(const Conv & c) {
    return round_up(c.oc, kMr) * c.k;
}

// The work buffer of the products of single precision: with shared columns,
// the columns of a chunk of K, and each thread's panels of weights; otherwise
// the panels of the weights, and each thread's columns.
struct Single_layout {
    Work_layout layout;
    Shared<float> shared;
    Own<float> mine;
};

Single_layout single_layout(const Conv & c, const Plan & pl) {
    Single_layout l;
    if (pl.shared_columns) {
        l.shared = l.layout.shared<float>(c.n * c.p * pl.chunk);
        l.mine = l.layout.own<float>(pl.mc * pl.kc);
    }
    else {
        l.shared = l.layout.shared<float>(weights_count(c));
        l.mine = l.layout.own<float>(pl.nc * pl.kc);
    }
    return l;
}

// Packs the columns of the image for positions [j0, j1) of image n and the
// stretch [k0, k1) of K, in panels of kNr positions: [panel][k1 - k0][kNr]. The
// positions past j1 are zeros.
void pack_columns(Elements<float> dst, const Conv & c, std::int64_t n, std::int64_t j0, std::int64_t j1, std::int64_t k0,
                  std::int64_t k1) {
    const std::int64_t kc = k1 - k0;
    const Elements<const float> image = c.x.from(n * c.xnb3);
    const std::int64_t taps = c.kh * c.kw;
    const bool pointwise = c.kh == 1 && c.kw == 1 && c.s0 == 1 && c.s1 == 1 && c.p0 == 0 && c.p1 == 0 &&
                           c.xnb1 == c.iw && c.ow == c.iw;
    for (std::int64_t jj = j0; jj < j1; jj += kNr) {
        const Elements<float> panel = dst.part((jj - j0) / kNr * kc * kNr, kc * kNr);
        const std::int64_t valid = std::min<std::int64_t>(kNr, j1 - jj);
        if (pointwise) {
            // A kernel of one by one: the columns are the image itself, its
            // positions one after another across the rows.
            for (std::int64_t k = k0; k < k1; ++k) {
                const Elements<const float> src = image.part(k * c.xnb2 + jj, valid);
                const Elements<float> d = panel.part((k - k0) * kNr, kNr);
                for (std::int64_t l = 0; l < kNr; ++l) {
                    d[l] = l < valid ? src[l] : 0.0f;
                }
            }
            continue;
        }
        const std::int64_t oh = jj / c.ow;
        const std::int64_t ow0 = jj % c.ow;
        std::int64_t ic = k0 / taps;
        std::int64_t tap = k0 % taps;
        std::int64_t kh = tap / c.kw;
        std::int64_t kw = tap % c.kw;
        if (c.s0 == 1 && valid == kNr && ow0 + kNr <= c.ow) {
            // Sixteen positions of one row of the output, at a stride of one:
            // each row of the columns is sixteen numbers in a row of the image,
            // or fewer and zeros at its edges.
            for (std::int64_t k = k0; k < k1; ++k) {
                const Elements<float> d = panel.part((k - k0) * kNr, kNr);
                const std::int64_t y = oh * c.s1 + kh * c.d1 - c.p1;
                const std::int64_t xs = ow0 + kw * c.d0 - c.p0;
                if (y < 0 || y >= c.ih) {
                    fill_with(d, 0.0f);
                }
                else if (xs >= 0 && xs + kNr <= c.iw) {
                    copy(d, image.part(ic * c.xnb2 + y * c.xnb1 + xs, kNr));
                }
                else {
                    const Elements<const float> row = image.part(ic * c.xnb2 + y * c.xnb1, c.iw);
                    for (std::int64_t l = 0; l < kNr; ++l) {
                        const std::int64_t x = xs + l;
                        d[l] = x >= 0 && x < c.iw ? row[x] : 0.0f;
                    }
                }
                if (++kw == c.kw) {
                    kw = 0;
                    if (++kh == c.kh) {
                        kh = 0;
                        ++ic;
                    }
                }
            }
            continue;
        }
        // Any other stride, or positions that run over the end of a row of the
        // output: one position at a time.
        std::array<std::int64_t, kNr> pos_y{};
        std::array<std::int64_t, kNr> pos_x{};
        for (std::size_t l = 0; l < kNr; ++l) {
            const std::int64_t j = jj + static_cast<std::int64_t>(l);
            pos_y[l] = j < j1 ? (j / c.ow) * c.s1 - c.p1 : -(INT64_C(1) << 40);
            pos_x[l] = j < j1 ? (j % c.ow) * c.s0 - c.p0 : -(INT64_C(1) << 40);
        }
        for (std::int64_t k = k0; k < k1; ++k) {
            const Elements<float> d = panel.part((k - k0) * kNr, kNr);
            const Elements<const float> plane = image.from(ic * c.xnb2);
            for (std::size_t l = 0; l < kNr; ++l) {
                const std::int64_t y = pos_y[l] + kh * c.d1;
                const std::int64_t x = pos_x[l] + kw * c.d0;
                d[static_cast<std::int64_t>(l)] = (y >= 0 && y < c.ih && x >= 0 && x < c.iw) ? plane[y * c.xnb1 + x] : 0.0f;
            }
            if (++kw == c.kw) {
                kw = 0;
                if (++kh == c.kh) {
                    kh = 0;
                    ++ic;
                }
            }
        }
    }
}

}  // namespace

bool conv2d_supported(const ggml_tensor * const node) {
    if (node->op != GGML_OP_CONV_2D) {
        return false;
    }
    const ggml_tensor * const kernel = node->src[0];
    const ggml_tensor * const input = node->src[1];
    const bool weights = (kernel->type == GGML_TYPE_F16 && holds<std::uint16_t>(kernel)) ||
                         (kernel->type == GGML_TYPE_F32 && holds<float>(kernel));
    return weights && ggml_is_contiguous(kernel) && holds<float>(input) && input->nb[0] == sizeof(float) &&
           holds<float>(node) && ggml_is_contiguous(node) && kernel->ne[2] == input->ne[2] && node->op_params[0] > 0 &&
           node->op_params[1] > 0 && node->op_params[4] > 0 && node->op_params[5] > 0;
}

namespace {

bool on_amx(const Conv & c);
std::size_t amx_work_size(const Conv & c, int n_threads);

// ---------------------------------------------------------------- quantized

// The products in integers (quant.h, Tuning::bits): the weights, signed, each
// output channel's in groups of K, quantized once and kept by the store; the
// columns of a block of positions packed in single precision as above, a
// stretch of K at a time, and quantized, unsigned with a zero point, each
// position's in groups of the stretch. The blocks are those of the weights
// shared, whatever the size of the images.
struct Qplan {
    Quant q;
    std::int64_t kc;  // a multiple of the group
    std::int64_t mc, nc, mblocks, nblocks;
    std::int64_t a_panel, b_panel;
};

Qplan qplan(const Conv & c, int n_threads) {
    const Tuning & t = tuning();
    Qplan pl{};
    pl.q = quant();
    const std::int64_t group = 4 * pl.q.quads;
    pl.kc = std::max(group, std::min<std::int64_t>(round_up(c.k, group), t.conv_kc) / group * group);
    pl.mblocks = ceil_div(c.oc, t.conv_mc);
    pl.mc = round_up(ceil_div(c.oc, pl.mblocks), kMr);
    pl.nc = round_up(t.conv_nc, kNr);
    pl.nblocks = ceil_div(c.p, pl.nc);
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    while (c.n * pl.mblocks * pl.nblocks < wanted && pl.nc > 32) {
        pl.nc /= 2;
        pl.nblocks = ceil_div(c.p, pl.nc);
    }
    for (std::int64_t split = pl.mblocks + 1; c.n * pl.mblocks * pl.nblocks < wanted && pl.mc > 8 * kMr; ++split) {
        pl.mc = std::min(pl.mc, round_up(ceil_div(c.oc, split), kMr));
        pl.mblocks = ceil_div(c.oc, pl.mc);
    }
    pl.a_panel = qpanel_bytes(c.k, kMr, pl.q);
    pl.b_panel = qpanel_bytes(pl.kc, kNr, pl.q);
    return pl;
}

std::int64_t quantized_weights_count(const Conv & c, const Qplan & pl) {
    return ceil_div(c.oc, kMr) * pl.a_panel;
}

// The quantized weights, shared; each thread's columns of a block, in single
// precision and quantized.
struct Quantized_layout {
    Work_layout layout;
    Shared<std::uint8_t> weights;
    Own<float> columns;
    Own<std::uint8_t> quantized;
};

Quantized_layout quantized_layout(const Conv & c, const Qplan & pl) {
    Quantized_layout l;
    l.weights = l.layout.shared<std::uint8_t>(quantized_weights_count(c, pl));
    l.columns = l.layout.own<float>(round_up(pl.nc, kNr) * pl.kc);
    l.quantized = l.layout.own<std::uint8_t>(ceil_div(pl.nc, kNr) * pl.b_panel);
    return l;
}

std::uint64_t qvariant(const Quant & q) {
    return static_cast<std::uint64_t>(q.quads) | static_cast<std::uint64_t>(q.weight_max) << 16 |
           static_cast<std::uint64_t>(q.weight_groups ? 1 : 0) << 32;
}

void compute_quantized(const ggml_compute_params * const params, const Conv & c, const Qplan & pl,
                       const Epilogue & chain) {
    const Quant & q = pl.q;
    const std::int64_t group = 4 * q.quads;
    const Quantized_layout l = quantized_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const std::shared_ptr<Packed> kept =
        packed(c.kernel, Layout::quantized_conv, quantized_weights_count(c, pl), qvariant(q));
    const bool fill = kept == nullptr || !kept->ready();
    const Qkernel kernel = qkernel_b_unsigned();
    const Qkernel_wide wide = qkernel_wide_b_unsigned();
    const std::int64_t step = wide != nullptr ? 2 * kNr : kNr;
    const std::int64_t a_group = qgroup_bytes(q.quads, kMr);

    Work_queue queue(params, buffer);
    const Elements<std::uint8_t> weights = kept != nullptr ? kept->octets() : buffer[l.weights];
    const Elements<float> columns = buffer[l.columns];
    const Elements<std::uint8_t> qcolumns = buffer[l.quantized];
    // The weights, a panel of kMr output channels to an item.
    const std::int64_t pack_items = fill ? ceil_div(c.oc, kMr) : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            const std::int64_t r0 = item * kMr;
            const std::int64_t rows = std::min<std::int64_t>(kMr, c.oc - r0);
            const Elements<std::uint8_t> dst = weights.part(item * pl.a_panel, pl.a_panel);
            if (c.wtype == GGML_TYPE_F16) {
                quantize_weights_f16(dst, c.w16.from(r0 * c.wnb3), c.wnb3, rows, c.k, kMr, Qlayout::rows, q);
            }
            else {
                quantize_weights_f32(dst, c.w32.from(r0 * c.wnb3), c.wnb3, rows, c.k, kMr, Qlayout::rows, q);
            }
        }
        queue.barrier();
        if (kept != nullptr && params->ith == 0) {
            publish(kept);
        }
    }

    const std::vector<Block> blocks = guided_blocks(c.n, c.p, kNr, pl.nc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.mblocks;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const std::int64_t ib = u % pl.mblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.mblocks)];
        const std::int64_t n = block.segment;
        const std::int64_t i0 = ib * pl.mc;
        const std::int64_t i1 = std::min(c.oc, i0 + pl.mc);
        const std::int64_t j0 = block.start;
        const std::int64_t j1 = j0 + block.size;
        const std::int64_t panels = ceil_div(j1 - j0, kNr);
        for (std::int64_t k0 = 0; k0 < c.k; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.k - k0);
            {
                const Phase_scope scope(Phase::pack_inputs);
                pack_columns(columns, c, n, j0, j1, k0, k0 + kc);
                for (std::int64_t p = 0; p < panels; ++p) {
                    quantize_panel_inputs(qcolumns.part(p * pl.b_panel, pl.b_panel), columns.part(p * kc * kNr, kc * kNr),
                                          kc, kNr, Qlayout::columns, q);
                }
            }
            const bool last = k0 + kc == c.k;
            const std::int64_t groups = ceil_div(kc, group);
            const Phase_scope scope(Phase::product);
            for (std::int64_t jj = j0; jj < j1; jj += step) {
                const Elements<const std::uint8_t> b = qcolumns.from(((jj - j0) / kNr) * pl.b_panel);
                const std::int64_t cols = std::min(step, j1 - jj);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    const Elements<const std::uint8_t> a = weights.from((ii / kMr) * pl.a_panel + k0 / group * a_group);
                    const std::int64_t rows = std::min<std::int64_t>(kMr, i1 - ii);
                    qtiles(kernel, wide, groups, q.quads, a, b, pl.b_panel, c.y.from(n * c.ynb3 + ii * c.ynb2 + jj), c.ynb2,
                           k0 > 0 || chain.from_redirect, rows, cols);
                    if (last && !chain.empty()) {
                        apply_chain_positions(chain, c, n, ii, rows, jj, cols);
                    }
                }
            }
        }
    }
}

}  // namespace

std::size_t conv2d_work_size(const ggml_tensor * const node, int n_threads) {
    const Conv c = geometry(node);
    if (quantized(Quantized_op::convolution)) {
        return quantized_layout(c, qplan(c, n_threads)).layout.bytes(n_threads);
    }
    if (on_amx(c)) {
        return amx_work_size(c, n_threads);
    }
    return single_layout(c, plan(c, n_threads)).layout.bytes(n_threads);
}

namespace {

constexpr std::int64_t kPanelsPerItem = 4;

std::int64_t weight_items(const Conv & c) {
    return ceil_div(ceil_div(c.oc, kMr), kPanelsPerItem);
}

void pack_weights(Elements<float> weights, const Conv & c, std::int64_t item) {
    const std::int64_t r0 = item * kPanelsPerItem * kMr;
    const std::int64_t rows = std::min<std::int64_t>(kPanelsPerItem * kMr, c.oc - r0);
    pack_weight_rows(weights.from(r0 * c.k), c, r0, rows, 0, c.k, kMr);
}

void compute_shared_columns(const ggml_compute_params * const params, const Conv & c, const Plan & pl,
                            const Epilogue & chain) {
    const Single_layout l = single_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t panels = c.n * c.p / kNr;
    const std::vector<Block> blocks = guided_blocks(1, c.oc, kMr, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    // The panels of the weights the store keeps, packed first when they are not
    // yet; without them, each block packs its own as it goes.
    const std::shared_ptr<Packed> kept = packed(c.kernel, Layout::conv_rows, weights_count(c));
    const bool fill = kept != nullptr && !kept->ready();

    // A stretch of K at a time: its columns, packed by all threads together,
    // [panel][k][kNr], a panel of kNr positions over kStretch of the stretch
    // to an item; then the blocks of channels, each adding the product of its
    // weights and those columns to its outputs. The threads meet after each.
    constexpr std::int64_t kStretch = 512;
    Work_queue queue(params, buffer);
    const Elements<float> columns = buffer[l.shared];
    const Elements<float> weights = buffer[l.mine];
    int item = queue.next();
    std::int64_t end = 0;
    if (fill) {
        end = weight_items(c);
        for (; item < end; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            pack_weights(kept->singles(), c, item);
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }
    for (std::int64_t c0 = 0; c0 < c.k; c0 += pl.chunk) {
        const std::int64_t c1 = std::min(c.k, c0 + pl.chunk);
        const std::int64_t width = c1 - c0;
        const std::int64_t stretches = ceil_div(width, kStretch);
        end += panels * stretches;
        for (; item < end; item = queue.next()) {
            const std::int64_t i = item - (end - panels * stretches);
            const std::int64_t panel = i / stretches;
            const std::int64_t k0 = c0 + (i % stretches) * kStretch;
            const std::int64_t k1 = std::min(c1, k0 + kStretch);
            const std::int64_t j = panel * kNr;
            const Phase_scope scope(Phase::pack_inputs);
            pack_columns(columns.from((panel * width + (k0 - c0)) * kNr), c, j / c.p, j % c.p, j % c.p + kNr, k0, k1);
        }
        queue.barrier();
        end += units;
        for (; item < end; item = queue.next()) {
            const Block & block = blocks[static_cast<std::size_t>(item - (end - units))];
            const std::int64_t i0 = block.start;
            const std::int64_t i1 = i0 + block.size;
            for (std::int64_t k0 = c0; k0 < c1; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c1 - k0);
                if (kept == nullptr) {
                    const Phase_scope scope(Phase::pack_weights);
                    pack_weight_rows(weights, c, i0, i1 - i0, k0, k0 + kc, kMr);
                }
                const bool last = k0 + kc == c.k;
                const Phase_scope scope(Phase::product);
                for (std::int64_t panel = 0; panel < panels;) {
                    const std::int64_t n = panel * kNr / c.p;
                    const std::int64_t position = panel * kNr % c.p;
                    // Two panels at once when the wide micro-kernel takes them,
                    // and both are of this image.
                    const std::int64_t cols = wide != nullptr && panel + 1 < panels && position + 2 * kNr <= c.p ? 2 * kNr : kNr;
                    const Elements<const float> b = columns.from((panel * width + (k0 - c0)) * kNr);
                    for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                        const Elements<const float> a =
                            kept != nullptr ? kept->singles().from(ii * c.k + k0 * kMr) : weights.from((ii - i0) * kc);
                        const std::int64_t rows = std::min<std::int64_t>(kMr, i1 - ii);
                        tiles(uk, wide, kc, a, b, width * kNr, c.y.from(n * c.ynb3 + ii * c.ynb2 + position), c.ynb2,
                              k0 > 0 || chain.from_redirect, rows, cols);
                        if (last && !chain.empty()) {
                            apply_chain_positions(chain, c, n, ii, rows, position, cols);
                        }
                    }
                    panel += cols / kNr;
                }
            }
        }
        // The next stretch's columns go where these are.
        queue.barrier();
    }
}

void compute_shared_weights(const ggml_compute_params * const params, const Conv & c, const Plan & pl,
                            const Epilogue & chain) {
    const Single_layout l = single_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    // The panels of the weights: the store's, packed here the first time, or
    // the work buffer's, packed here every time.
    const std::shared_ptr<Packed> kept = packed(c.kernel, Layout::conv_rows, weights_count(c));
    const bool fill = kept == nullptr || !kept->ready();
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const bool rows_outer = tuning().conv_rows_outer != 0;

    Work_queue queue(params, buffer);
    const Elements<float> weights = kept != nullptr ? kept->singles() : buffer[l.shared];
    const Elements<float> columns = buffer[l.mine];
    const std::int64_t pack_items = fill ? weight_items(c) : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            pack_weights(weights, c, item);
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }

    const std::vector<Block> blocks = guided_blocks(c.n, c.p, kNr, pl.nc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.mblocks;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const std::int64_t ib = u % pl.mblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.mblocks)];
        const std::int64_t n = block.segment;
        const std::int64_t i0 = ib * pl.mc;
        const std::int64_t i1 = std::min(c.oc, i0 + pl.mc);
        const std::int64_t j0 = block.start;
        const std::int64_t j1 = j0 + block.size;
        for (std::int64_t k0 = 0; k0 < c.k; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.k - k0);
            {
                const Phase_scope scope(Phase::pack_inputs);
                pack_columns(columns, c, n, j0, j1, k0, k0 + kc);
            }
            const bool last = k0 + kc == c.k;
            const Phase_scope scope(Phase::product);
            const auto one = [&](std::int64_t ii, std::int64_t jj) {
                const Elements<const float> a = weights.from(ii * c.k + k0 * kMr);
                const Elements<const float> b = columns.from(((jj - j0) / kNr) * kc * kNr);
                const std::int64_t rows = std::min<std::int64_t>(kMr, i1 - ii);
                const std::int64_t cols = std::min(step, j1 - jj);
                tiles(uk, wide, kc, a, b, kc * kNr, c.y.from(n * c.ynb3 + ii * c.ynb2 + jj), c.ynb2,
                      k0 > 0 || chain.from_redirect, rows, cols);
                if (last && !chain.empty()) {
                    apply_chain_positions(chain, c, n, ii, rows, jj, cols);
                }
            };
            if (rows_outer) {
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    for (std::int64_t jj = j0; jj < j1; jj += step) {
                        one(ii, jj);
                    }
                }
            }
            else {
                for (std::int64_t jj = j0; jj < j1; jj += step) {
                    for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                        one(ii, jj);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- on AMX

bool on_amx(const Conv & c) {
    return split_form(Split_use::convolution).kernel != nullptr && c.wtype == GGML_TYPE_F16;
}

// The elements of the tiles of A of sixteen output channels over the whole of
// K, rounded up to 32, and of all of them, the channels rounded up to 32. A
// block of k of a tile is a_bytes, of elements of two bytes.
std::int64_t amx_row(const Conv & c, const Split_form & f) {
    return round_up(c.k, kAmxBlock) / kAmxBlock * (f.a_bytes / 2);
}

std::int64_t amx_weights_count(const Conv & c, const Split_form & f) {
    return round_up(c.oc, kAmxBlock) / kAmxTile * amx_row(c, f);
}

struct Amx_plan {
    std::int64_t mc, nc, kc, mblocks;
};

Amx_plan amx_plan(const Conv & c, int n_threads) {
    const Tuning & t = tuning();
    Amx_plan pl{};
    pl.kc = std::min(round_up(c.k, kAmxBlock), round_up(std::max(t.amx_kc, 1), kAmxBlock));
    pl.mblocks = ceil_div(c.oc, std::max(t.conv_mc, 1));
    pl.mc = round_up(ceil_div(c.oc, pl.mblocks), kAmxBlock);
    pl.mblocks = ceil_div(c.oc, pl.mc);
    pl.nc = round_up(t.conv_nc, kAmxBlock);
    // Enough blocks for every thread to take several.
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    while (c.n * pl.mblocks * ceil_div(c.p, pl.nc) < wanted && pl.nc > kAmxBlock) {
        pl.nc = round_up(pl.nc / 2, kAmxBlock);
    }
    return pl;
}

// The tiles of A, shared; each thread's columns of a block in single
// precision, their tiles of B, and the block it copies a part of.
struct Amx_layout {
    Work_layout layout;
    Shared<std::uint16_t> weights;
    Own<float> columns;
    Own<std::uint16_t> tiles;
    Own<float> spare;
};

Amx_layout amx_layout(const Conv & c, const Split_form & f, const Amx_plan & pl) {
    Amx_layout l;
    l.weights = l.layout.shared<std::uint16_t>(amx_weights_count(c, f));
    l.columns = l.layout.own<float>(pl.nc * pl.kc);
    l.tiles = l.layout.own<std::uint16_t>(pl.nc / kAmxTile * (pl.kc / kAmxBlock) * (f.b_bytes / 2));
    l.spare = l.layout.own<float>(kAmxBlock * kAmxBlock);
    return l;
}

std::size_t amx_work_size(const Conv & c, int n_threads) {
    const Split_form f = split_form(Split_use::convolution);
    return amx_layout(c, f, amx_plan(c, n_threads)).layout.bytes(n_threads);
}

void compute_amx(const ggml_compute_params * const params, const Conv & c, const Epilogue & chain) {
    const Split_form f = split_form(Split_use::convolution);
    const Amx_plan pl = amx_plan(c, params->nth);
    const Amx_layout l = amx_layout(c, f, pl);
    Work_buffer buffer(params, l.layout);
    const std::shared_ptr<Packed> kept =
        packed(c.kernel, f.fp16 ? Layout::amx_conv_fp16 : Layout::amx_conv, amx_weights_count(c, f));
    const bool fill = kept == nullptr || !kept->ready();
    const std::int64_t row = amx_row(c, f);

    Work_queue queue(params, buffer);
    const Elements<std::uint16_t> weights = kept != nullptr ? kept->halves() : buffer[l.weights];
    const Elements<float> columns = buffer[l.columns];
    const Elements<std::uint16_t> tiles = buffer[l.tiles];
    const Elements<float> spare = buffer[l.spare];
    // The weights, a tile of sixteen output channels to an item; those past
    // the last are zeros.
    const std::int64_t pack_items = fill ? round_up(c.oc, kAmxBlock) / kAmxTile : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            const Elements<std::uint16_t> dst = weights.part(item * row, row);
            const std::int64_t r0 = item * kAmxTile;
            if (r0 >= c.oc) {
                fill_with(dst, std::uint16_t{0});
                continue;
            }
            const std::int64_t rows = std::min(kAmxTile, c.oc - r0);
            const Elements<const std::uint16_t> src = c.w16.from(r0 * c.wnb3);
            SDASM_CHECK(src.size() >= (rows - 1) * c.wnb3 + c.k);
            f.pack_weight_rows(dst.data(), src.data(), bytes_of<std::uint16_t>(c.wnb3), rows, c.k);
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }

    const std::vector<Block> blocks = guided_blocks(c.n, c.p, kAmxBlock, pl.nc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.mblocks;
    bool configured = false;
    for (; item < pack_items + units; item = queue.next()) {
        if (!configured) {
            split_begin(f);
            configured = true;
        }
        const std::int64_t u = item - pack_items;
        const std::int64_t ib = u % pl.mblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.mblocks)];
        const std::int64_t n = block.segment;
        const std::int64_t i0 = ib * pl.mc;
        const std::int64_t i1 = std::min(c.oc, i0 + pl.mc);
        const std::int64_t j0 = block.start;
        const std::int64_t j1 = j0 + block.size;
        const std::int64_t panels = ceil_div(j1 - j0, kNr);
        for (std::int64_t k0 = 0; k0 < c.k; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.k - k0);
            const std::int64_t kb = ceil_div(kc, kAmxBlock);
            const std::int64_t panel = kb * (f.b_bytes / 2);
            {
                // The columns in single precision, then their tiles of B; one
                // more of zeros past an odd number of panels, for the last pair.
                const Phase_scope scope(Phase::pack_inputs);
                pack_columns(columns, c, n, j0, j1, k0, k0 + kc);
                for (std::int64_t p = 0; p < panels; ++p) {
                    const Elements<std::uint16_t> dst = tiles.part(p * panel, panel);
                    const Elements<const float> src = columns.part(p * kc * kNr, kc * kNr);
                    f.pack_panel(dst.data(), src.data(), kc);
                }
                if (panels % 2 != 0) {
                    fill_with(tiles.part(panels * panel, panel), std::uint16_t{0});
                }
            }
            const bool last = k0 + kc == c.k;
            const Phase_scope scope(Phase::product);
            for (std::int64_t jj = j0; jj < j1; jj += kAmxBlock) {
                const Elements<const std::uint16_t> b = tiles.from((jj - j0) / kAmxTile * panel);
                const std::int64_t cols = std::min(kAmxBlock, j1 - jj);
                for (std::int64_t ii = i0; ii < i1; ii += kAmxBlock) {
                    const Elements<const std::uint16_t> a = weights.from(ii / kAmxTile * row + k0 / kAmxBlock * (f.a_bytes / 2));
                    const std::int64_t rows = std::min(kAmxBlock, i1 - ii);
                    split_block(f, kb, a, row, b, panel, c.y.from(n * c.ynb3 + ii * c.ynb2 + jj), c.ynb2,
                                k0 > 0 || chain.from_redirect, rows, cols, spare);
                    if (last && !chain.empty()) {
                        apply_chain_positions(chain, c, n, ii, rows, jj, cols);
                    }
                }
            }
        }
    }
    if (configured) {
        split_end(f);
    }
}

}  // namespace

void conv2d_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain) {
    Conv c = geometry(node);
    if (chain.redirect != nullptr) {
        // The residual connection's tensor, of the same layout (fusion.h).
        c.y = elements_of<float>(chain.redirect);
    }
    if (quantized(Quantized_op::convolution)) {
        compute_quantized(params, c, qplan(c, params->nth), chain);
        return;
    }
    if (on_amx(c)) {
        compute_amx(params, c, chain);
        return;
    }
    const Plan pl = plan(c, params->nth);
    if (pl.shared_columns) {
        compute_shared_columns(params, c, pl, chain);
    }
    else {
        compute_shared_weights(params, c, pl, chain);
    }
}

}  // namespace sdasm
