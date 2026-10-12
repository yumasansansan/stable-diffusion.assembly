// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The convolution of a kernel of three by three, at a stride of one, by
// Winograd's minimal filtering F(2x2, 3x3) (Lavin and Gray, "Fast Algorithms
// for Convolutional Neural Networks", 2016): each tile of two by two outputs is
// computed from the four by four inputs around it as
//
//     Y = A^T [ sum over ic of (G g G^T) (.) (B^T d B) ] A
//
// where g is the kernel of a channel, d the tile of its input and (.) the
// product element by element, with
//
//     B^T = | 1  0 -1  0 |    G = |  1    0    0  |    A^T = | 1  1  1  0 |
//           | 0  1  1  0 |        | 1/2  1/2  1/2 |          | 0  1 -1 -1 |
//           | 0 -1  1  0 |        | 1/2 -1/2  1/2 |
//           | 0  1  0 -1 |        |  0    0    1  |
//
// The sum over the channels is, for each of the sixteen elements xi of a tile, a
// matrix product (gemm.h) of the transformed weights, OC by IC, and the
// transformed inputs, IC by tiles: sixteen products of 16 * OC * IC * tiles
// multiplications in all, where the direct convolution makes 36 * OC * IC *
// tiles, 2.25 times as many. The transforms add, subtract and halve, and are
// computed in single precision, as the products are; what they round adds to
// the rounding of the products a little, measured against the exact result by
// tests/ops_test.cc beside the direct convolution's and ggml's own (which rounds
// the image to FP16 before the products, and is far the coarsest of the three).
//
// For images of many tiles, the weights are transformed once, by all threads
// together, into the panels of A of each of the sixteen products. The inputs
// are transformed by the thread that packs them, element xi of a tile at a
// time, straight into the panels of B, sixteen tiles of a row of tiles to a
// panel, as the columns of the direct convolution are packed from the image:
// no transformed image is written out. A thread's block is a stretch of tiles
// of an image and a block of output channels, whose sixteen products it keeps,
// [xi][channel][tile], until it transforms them into the outputs.
//
// The small images of a model's deepest layers -- 16 by 16 and 8 by 8 in SD
// 1.5's UNet, with 1280 channels and more -- have few tiles and many weights:
// transformed, the weights of such a layer are a hundred megabytes, which
// would go out to memory and back for products of a hundred or so tiles. There
// the inputs are transformed instead, once, by all threads together, every tile
// of every image; and a block is a block of output channels over all of them,
// whose thread transforms the weights of its channels itself, a stretch of the
// input channels at a time, into panels that stay in its cache: each weight is
// transformed once, and none is written out.
//
// For FP16 kernels and images of 16 by 16 outputs and more, F(4x4, 3x3) is
// taken instead where the products are of single precision (the section at
// the end): 36 elements to a tile of sixteen outputs, 1.78 times fewer
// multiplications, with transforms of their own written by hand
// (x86_64/winograd4_*.S), and a path of its own for images of few tiles, which
// at those sizes are the UNet's 32 by 32 and 16 by 16 of 640 channels and
// more. At 8 by 8, its transform of the weights costs more than its products
// save, and F(2x2, 3x3) is kept. Its outputs
// round about seven times as much as those of F(2x2, 3x3), and remain 40 to 100
// times closer to the exact ones than ggml's, which round the image to FP16.
//
// On AMX, the sixteen products of FP16 kernels go to the tiles (gemm.h), in
// the same blocks: A is the transformed weights of an xi, in tiles of sixteen
// output channels by 32 input channels, and B its transformed inputs, in tiles
// of pairs of input channels by sixteen tiles, made from the panels that the
// transforms write in single precision (sdasm_amx_pack_panel()); both split
// in two parts of bfloat16, x1 + x2, and the products x1 y1 + x1 y2 + x2 y1
// summed in single precision, into the same [xi][channel][tile] the transform
// of the outputs reads. ggml's form rounds the image to FP16 before its
// products; these are within 3 2^-16 of the products of the transformed
// numbers, which the transform of the outputs adds by eights.

#include <algorithm>
#include <array>
#include <bit>
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

constexpr int kXi = 16;  // the elements of a tile of four by four

// The strides are in elements.
struct Conv {
    const ggml_tensor * kernel;
    Elements<const float> x;  // [IW, IH, IC, N]
    std::int64_t iw, ih, ic, n;
    std::int64_t xnb1, xnb2, xnb3;
    ggml_type wtype;
    Elements<const float> w32;          // [3, 3, IC, OC], of single precision
    Elements<const std::uint16_t> w16;  // or FP16
    std::int64_t oc;
    std::int64_t wnb3;
    Elements<float> y;  // [OW, OH, OC, N]
    std::int64_t ow, oh;
    std::int64_t ynb1, ynb2, ynb3;
    int p0, p1;
    std::int64_t tw, th;  // tiles across and down
};

// Of a node that winograd_supported() takes.
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
    c.oc = kernel->ne[3];
    c.y = elements_of<float>(node);
    c.ow = node->ne[0];
    c.oh = node->ne[1];
    c.ynb1 = stride_of<float>(node, 1);
    c.ynb2 = stride_of<float>(node, 2);
    c.ynb3 = stride_of<float>(node, 3);
    c.p0 = node->op_params[2];
    c.p1 = node->op_params[3];
    c.tw = (c.ow + 1) / 2;
    c.th = (c.oh + 1) / 2;
    return c;
}

std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

std::int64_t round_up(std::int64_t a, std::int64_t b) {
    return ceil_div(a, b) * b;
}

// The first element of channel ic of image n, of the input, and of channel r
// of image n, of the output.
std::int64_t input_plane(const Conv & c, std::int64_t n, std::int64_t ic) {
    return n * c.xnb3 + ic * c.xnb2;
}
std::int64_t output_plane(const Conv & c, std::int64_t n, std::int64_t r) {
    return n * c.ynb3 + r * c.ynb2;
}

// The weights of output channels [r0, r0 + rows) over [k0, k1) of the input
// channels' nine numbers each into panels of p (gemm.h).
void pack_weight_rows(Elements<float> dst, const Conv & c, std::int64_t r0, std::int64_t rows, std::int64_t k0,
                      std::int64_t k1, int p) {
    if (c.wtype == GGML_TYPE_F16) {
        pack_rows_f16(dst, c.w16.from(r0 * c.wnb3), c.wnb3, rows, k0, k1, p);
    }
    else {
        pack_rows_f32(dst, c.w32.from(r0 * c.wnb3), c.wnb3, rows, k0, k1, p);
    }
}

// The large-image path keeps the sixteen tiles of a panel of B in the order
// its packing finds them in, which is not quite their own: within each eight,
// tiles 0, 1, 4, 5 and then 2, 3, 6, 7 (pack_inputs()). The lane of tile l of
// a block, l counted from the block's first, is shuffled(l); the order is its
// own inverse.
constexpr std::array<std::int64_t, 8> kShuffle = {0, 1, 4, 5, 2, 3, 6, 7};

std::int64_t shuffled(std::int64_t l) {
    return (l & ~std::int64_t{7}) | kShuffle[static_cast<std::size_t>(l & 7)];
}

// Small images: every tile of every image in one block of output channels.
bool small(const Conv & c) {
    return c.n * c.tw * c.th <= tuning().winograd_small_tiles;
}

struct Plan {
    bool small;
    std::int64_t mc, kc;
    std::int64_t tiles;  // tiles in a block, a multiple of kNr
    std::int64_t mblocks, tblocks;
    std::int64_t panels;  // small images: the panels of kNr tiles of all images
};

// The widened weights of kWidened input channels of a panel of output
// channels, [channel][3 * 3][kMr], and the floats a full load past them reads.
constexpr std::int64_t kWidened = 8;
constexpr std::int64_t kWidenedFloats = kWidened * 9 * kMr + 8;

// The sixteen xi of a tile go to sixteen blocks of floats, one for each, which
// the transforms write, or read, all at once; blocks a multiple of 4 KiB apart
// would put the sixteen in the same set of the first-level cache, which holds
// eight or twelve. Each block is kSkew floats, a line, further on than the last.
constexpr std::int64_t kSkew = 16;

std::int64_t skewed(std::int64_t floats) {
    return round_up(floats, kSkew) + kSkew;
}

// The floats between the transformed weights of one xi and the next, in a
// block of mc channels and kc input channels.
std::int64_t xi_stride(std::int64_t mc, std::int64_t kc) {
    return skewed(mc * kc);
}

// Small images: the floats between the transformed inputs of one xi and the
// next, and between the products of one xi and the next, in a block of mc
// channels.
std::int64_t inputs_stride(std::int64_t panels, std::int64_t ic) {
    return skewed(panels * kNr * ic);
}

std::int64_t products_stride(std::int64_t mc, std::int64_t nt) {
    return skewed(mc * nt);
}

Plan plan_small(const Conv & c) {
    const Tuning & t = tuning();
    Plan pl{};
    pl.small = true;
    pl.panels = ceil_div(c.n * c.tw * c.th, kNr);
    pl.kc = std::min<std::int64_t>(c.ic, std::max(kWidened, static_cast<std::int64_t>(t.winograd_small_kc)));
    pl.mc = round_up(std::min<std::int64_t>(c.oc, std::max(kMr, t.winograd_small_mc)), kMr);
    return pl;
}

Plan plan(const Conv & c, int n_threads) {
    if (small(c)) {
        return plan_small(c);
    }
    const Tuning & t = tuning();
    Plan pl{};
    pl.kc = std::min<std::int64_t>(c.ic, t.winograd_kc);
    pl.mblocks = ceil_div(c.oc, t.winograd_mc);
    pl.mc = round_up(ceil_div(c.oc, pl.mblocks), kMr);
    pl.mblocks = ceil_div(c.oc, pl.mc);
    // Tiles in rows of tiles: a block is whole rows when a row fills panels,
    // and runs on across rows otherwise.
    pl.tiles = round_up(std::min<std::int64_t>(t.winograd_tiles, c.tw * c.th), kNr);
    pl.tblocks = ceil_div(c.tw * c.th, pl.tiles);
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    while (c.n * pl.mblocks * pl.tblocks < wanted && pl.tiles > kNr) {
        pl.tiles = round_up(pl.tiles / 2, kNr);
        pl.tblocks = ceil_div(c.tw * c.th, pl.tiles);
    }
    return pl;
}

// The large images' transformed weights of every xi, [xi][panel][ic][kMr].
std::int64_t weights_count(const Conv & c) {
    return kXi * xi_stride(round_up(c.oc, kMr), c.ic);
}

// A thread's panels of the inputs of the four xi of a row of B^T, for a block
// of kc input channels and `tiles` tiles, each `stride` floats from the last;
// and the floats of them, which the widened weights fit in too.
std::int64_t panels_stride(std::int64_t kc, std::int64_t tiles) {
    return skewed(kc * tiles);
}
std::int64_t panels_count(std::int64_t kc, std::int64_t tiles) {
    return std::max(4 * panels_stride(kc, tiles), round_up(kWidenedFloats, kSkew));
}

// The work buffer of F(2x2, 3x3) in single precision. Large images: the
// transformed weights, shared, and each thread's panels and products. Small
// images: the transformed inputs of every tile, shared, and each thread's
// widened weights, transformed weights, a channel of an image with its padding
// and products.
struct Single_layout {
    Work_layout layout;
    Shared<float> shared;
    Own<float> panels;  // large: the inputs' panels; small: the widened weights
    Own<float> weights, scratch;  // small images
    Own<float> products;
};

Single_layout single_layout(const Conv & c, const Plan & pl) {
    Single_layout l;
    if (pl.small) {
        l.shared = l.layout.shared<float>(kXi * inputs_stride(pl.panels, c.ic));
        l.panels = l.layout.own<float>(kWidenedFloats);
        l.weights = l.layout.own<float>(kXi * xi_stride(pl.mc, pl.kc));
        l.scratch = l.layout.own<float>((2 * c.th + 2) * (2 * c.tw + 2));
        l.products = l.layout.own<float>(kXi * products_stride(pl.mc, pl.panels * kNr));
        return l;
    }
    l.shared = l.layout.shared<float>(weights_count(c));
    l.panels = l.layout.own<float>(panels_count(pl.kc, pl.tiles));
    l.products = l.layout.own<float>(kXi * products_stride(pl.mc, pl.tiles));
    return l;
}

// The four by four inputs of tile t of image n, channel ic, rows and columns,
// zeros outside the image: of the channel's plane, its first element at plane.
inline float input_at(const Conv & c, Elements<const float> plane, std::int64_t y, std::int64_t x) {
    return (y >= 0 && y < c.ih && x >= 0 && x < c.iw) ? plane[y * c.xnb1 + x] : 0.0f;
}

// Row i of B^T applied to four numbers.
inline float bt(int i, float d0, float d1, float d2, float d3) {
    switch (i) {
    case 0:
        return d0 - d2;
    case 1:
        return d1 + d2;
    case 2:
        return d2 - d1;
    default:
        return d1 - d3;
    }
}

// A row of zeros, which the kernels written by hand read in place of a row of
// the image above or below it: as far as they read a row.
alignas(64) constexpr std::array<float, 2 * kNr + 16> kZeros{};

// Packs the elements xi = (r, s), s = 0..3, of the transformed inputs of tiles
// [t0, t1) of image n and channels [k0, k1) into panels of kNr tiles,
// [panel][k1 - k0][kNr], tile l of the block in lane shuffled(l), zeros past
// t1: those of element (r, s) `stride` floats after those of (r, s - 1). The
// four take the same two rows of each tile, which row r of B^T adds or
// subtracts, and are packed together so that those rows are read and taken
// apart once for the four.
void pack_inputs(Elements<float> dst, std::int64_t stride, const Conv & c, std::int64_t n, std::int64_t t0, std::int64_t t1,
                 std::int64_t k0, std::int64_t k1, int r, bool shuffle) {
    const std::int64_t kc = k1 - k0;
    // The rows of the tile that row r of B^T takes, with their signs.
    constexpr std::array<std::array<int, 2>, 4> kRows = {{{0, 2}, {1, 2}, {1, 2}, {1, 3}}};
    constexpr std::array<std::array<float, 2>, 4> kSigns = {{{1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}}};
    const auto rows_of = [&](std::size_t q) { return kRows[static_cast<std::size_t>(r)][q]; };
    const std::int64_t span = 3 * stride + ceil_div(t1 - t0, kNr) * kc * kNr;
    SDASM_REQUIRE(dst.size() >= span);
    for (std::int64_t tt = t0; tt < t1; tt += kNr) {
        const Elements<float> out = dst.from((tt - t0) / kNr * kc * kNr);
        const std::int64_t valid = std::min<std::int64_t>(kNr, t1 - tt);
        const std::int64_t ty = tt / c.tw;
        const std::int64_t tx0 = tt % c.tw;
        const std::int64_t y0 = 2 * ty - c.p1;
        const std::int64_t x0 = 2 * tx0 - c.p0;
#if SDASM_X86_64
        // Rows A and B of the tile, those of row r of B^T, for the kernels
        // written by hand: the row's first number, of channel k0, the bytes from
        // it to the tiles' first input, and the bytes to the next channel's;
        // a row outside the image, a row of zeros.
        const auto row = [&](int q, const float *& p, std::int64_t & offset, std::int64_t & step) {
            const std::int64_t y = y0 + rows_of(static_cast<std::size_t>(q));
            if (y < 0 || y >= c.ih) {
                p = kZeros.data();
                offset = 0;
                step = 0;
                return;
            }
            const Elements<const float> first = c.x.from(input_plane(c, n, k0) + y * c.xnb1);
            SDASM_CHECK(first.size() >= (kc - 1) * c.xnb2 + c.iw);
            p = first.data();
            offset = bytes_of<float>(x0);
            step = bytes_of<float>(c.xnb2);
        };
        if (avx512_kernels() && valid == kNr && tx0 + kNr <= c.tw) {
            // Sixteen tiles of a row by the kernel for AVX-512, in their order:
            // the rows as below, and the lanes of its four loads of sixteen
            // that are inside the image.
            constexpr std::array<int, 4> kOffsets = {0, 16, 2, 18};
            Winograd_inputs job{};
            for (std::size_t m = 0; m < 4; ++m) {
                std::uint16_t lanes = 0;
                for (int lane = 0; lane < 16; ++lane) {
                    const std::int64_t x = x0 + kOffsets[m] + lane;
                    lanes = static_cast<std::uint16_t>(lanes | ((x >= 0 && x < c.iw ? 1u : 0u) << lane));
                }
                job.lanes[m] = lanes;
            }
            const int qa = r == 2 ? 1 : 0;
            job.dst = out.data();
            job.stride = bytes_of<float>(stride);
            row(qa, job.a, job.a_offset, job.a_step);
            row(1 - qa, job.b, job.b_offset, job.b_step);
            job.channels = kc;
            job.subtract = r == 1 ? 0 : 1;
            sdasm_winograd_inputs_avx512(&job);
            continue;
        }
        if (shuffle && valid == kNr && tx0 + kNr <= c.tw) {
            // Sixteen tiles of a row, by the kernel written by hand, rows A
            // and B of the tile being those of row r of B^T, A plus or minus B
            // -- the one of rows 1 and 2 of r = 2 that is subtracted taken as A
            // minus B with the two exchanged. A row outside the image is a row
            // of zeros, and the inputs past the left and the right edges are
            // masked out of the loads.
            constexpr std::array<int, 8> kOffsets = {0, 8, 2, 10, 16, 24, 18, 26};
            alignas(32) std::array<std::int32_t, 8 * 8> masks{};
            const bool inside = x0 >= 0 && x0 + 2 * kNr + 2 <= c.iw;
            for (std::size_t m = 0; m < 8; ++m) {
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const std::int64_t x = x0 + kOffsets[m] + static_cast<std::int64_t>(lane);
                    masks[m * 8 + lane] = x >= 0 && x < c.iw ? -1 : 0;
                }
            }
            const int qa = r == 2 ? 1 : 0;
            Winograd_inputs job{};
            job.dst = out.data();
            job.stride = bytes_of<float>(stride);
            row(qa, job.a, job.a_offset, job.a_step);
            row(1 - qa, job.b, job.b_offset, job.b_step);
            job.channels = kc;
            // A row of zeros is read where a row of the image would be, and as
            // far: kZeros is long enough for it, masks or none.
            job.masks = inside ? nullptr : masks.data();
            job.subtract = r == 1 ? 0 : 1;
            sdasm_winograd_inputs_avx2(&job);
            continue;
        }
#endif
        if (!shuffle && valid == kNr && tx0 + kNr <= c.tw && x0 >= 0 && x0 + 2 * kNr + 2 <= c.iw) {
            // Sixteen tiles of a row, inside the image across: d[j] of tile l is
            // the input at 2 l + j along the two rows that row r of B^T takes,
            // or zeros for a row outside the image.
            for (std::int64_t k = k0; k < k1; ++k) {
                const Elements<const float> plane = c.x.from(input_plane(c, n, k));
                std::array<std::array<float, kNr>, 4> sum{};
                for (std::size_t q = 0; q < 2; ++q) {
                    const std::int64_t y = y0 + rows_of(q);
                    if (y < 0 || y >= c.ih) {
                        continue;
                    }
                    const Elements<const float> p = plane.part(y * c.xnb1 + x0, 2 * kNr + 2);
                    const float sign = kSigns[static_cast<std::size_t>(r)][q];
                    for (std::size_t l = 0; l < kNr; ++l) {
                        const auto x = static_cast<std::int64_t>(2 * l);
                        const float d0 = p[x];
                        const float d1 = p[x + 1];
                        const float d2 = p[x + 2];
                        const float d3 = p[x + 3];
                        sum[0][l] += sign * (d0 - d2);
                        sum[1][l] += sign * (d1 + d2);
                        sum[2][l] += sign * (d2 - d1);
                        sum[3][l] += sign * (d1 - d3);
                    }
                }
                for (std::size_t s4 = 0; s4 < 4; ++s4) {
                    const Elements<float> d = out.part(static_cast<std::int64_t>(s4) * stride + (k - k0) * kNr, kNr);
                    for (std::size_t l = 0; l < kNr; ++l) {
                        d[static_cast<std::int64_t>(l)] = sum[s4][l];
                    }
                }
            }
            continue;
        }
        // Tiles at the edges of the image, or running over the end of a row of
        // tiles: one at a time.
        for (std::int64_t k = k0; k < k1; ++k) {
            const Elements<const float> plane = c.x.from(input_plane(c, n, k));
            for (std::int64_t l = 0; l < kNr; ++l) {
                std::array<float, 4> o{};
                if (l < valid) {
                    const std::int64_t t = tt + l;
                    const std::int64_t yy = 2 * (t / c.tw) - c.p1;
                    const std::int64_t xx = 2 * (t % c.tw) - c.p0;
                    std::array<std::array<float, 4>, 4> d{};
                    for (std::size_t i = 0; i < 4; ++i) {
                        for (std::size_t j = 0; j < 4; ++j) {
                            d[i][j] = input_at(c, plane, yy + static_cast<std::int64_t>(i), xx + static_cast<std::int64_t>(j));
                        }
                    }
                    for (int s4 = 0; s4 < 4; ++s4) {
                        std::array<float, 4> rw{};
                        for (std::size_t i = 0; i < 4; ++i) {
                            rw[i] = bt(s4, d[i][0], d[i][1], d[i][2], d[i][3]);
                        }
                        o[static_cast<std::size_t>(s4)] = bt(r, rw[0], rw[1], rw[2], rw[3]);
                    }
                }
                const std::int64_t lane = shuffle ? shuffled(l) : l;
                for (std::size_t s4 = 0; s4 < 4; ++s4) {
                    out[static_cast<std::int64_t>(s4) * stride + (k - k0) * kNr + lane] = o[s4];
                }
            }
        }
    }
}

// Y = A^T M A for tiles [t0, t1) of image n and output channels [r0, r1), from
// their sixteen products, [xi][channel][tile]: those of xi `stride` floats
// apart, of channel r (r - r0) * nt floats after products, of tile t at t - t0
// -- or at shuffled(t - t0), with `shuffle`; added to the outputs there when
// the chain begins its sum from them (fusion.h). Then the chain after the
// convolution, over the rows of the outputs written.
void transform_outputs(const Conv & c, Elements<const float> products, std::int64_t stride, std::int64_t nt, std::int64_t n,
                       std::int64_t t0, std::int64_t t1, std::int64_t r0, std::int64_t r1, const Epilogue & chain,
                       bool shuffle) {
    const Phase_scope scope(Phase::transform_out);
    const bool add = chain.from_redirect;
    for (std::int64_t r = r0; r < r1; ++r) {
        const Elements<const float> m = products.from((r - r0) * nt);
        const std::int64_t plane = output_plane(c, n, r);
        // The output row y from column x on.
        const auto output = [&](std::int64_t y, std::int64_t x) { return c.y.from(plane + y * c.ynb1 + x); };
        for (std::int64_t t = t0; t < t1;) {
            const std::int64_t ty = t / c.tw;
            const std::int64_t tx = t % c.tw;
            // The tiles of this row of tiles from t on, whole in the output.
            const std::int64_t whole = std::min({t1 - t, c.tw - tx, (c.ow - 2 * tx) / 2});
#if SDASM_X86_64
            if (!shuffle && whole >= 16 && 2 * ty + 2 <= c.oh && avx512_kernels()) {
                // Sixteen tiles at a time, in their order, by the kernel for
                // AVX-512.
                const std::int64_t groups = whole / 16;
                SDASM_CHECK(m.size() >= (kXi - 1) * stride + (t - t0) + 16 * groups);
                sdasm_winograd_outputs_avx512(m.from(t - t0).data(), bytes_of<float>(stride),
                                              output(2 * ty, 2 * tx).first(32 * groups).data(),
                                              output(2 * ty + 1, 2 * tx).first(32 * groups).data(), groups, add ? 1 : 0);
                t += 16 * groups;
                continue;
            }
            if (shuffle && whole >= 8 && (t - t0) % 8 == 0 && 2 * ty + 2 <= c.oh) {
                // Eight tiles at a time, in the shuffled lanes the packing left
                // them in, by the kernel written by hand.
                const std::int64_t groups = whole / 8;
                SDASM_CHECK(m.size() >= (kXi - 1) * stride + (t - t0) + 8 * groups);
                sdasm_winograd_outputs_avx2(m.from(t - t0).data(), bytes_of<float>(stride),
                                            output(2 * ty, 2 * tx).first(16 * groups).data(),
                                            output(2 * ty + 1, 2 * tx).first(16 * groups).data(), groups, add ? 1 : 0);
                t += 8 * groups;
                continue;
            }
#endif
            if (!shuffle && whole > 0 && 2 * ty + 2 <= c.oh) {
                const Elements<float> row0 = output(2 * ty, 2 * tx).first(2 * whole);
                const Elements<float> row1 = output(2 * ty + 1, 2 * tx).first(2 * whole);
                const Elements<const float> v = m.from(t - t0);
                for (std::int64_t l = 0; l < whole; ++l) {
                    // A^T M A: rows 0 and 1 of A^T M, then their columns.
                    std::array<float, 4> a0{};
                    std::array<float, 4> a1{};
                    for (std::size_t j = 0; j < 4; ++j) {
                        const auto jj = static_cast<std::int64_t>(j);
                        const float v0 = v[(0 * 4 + jj) * stride + l];
                        const float v1 = v[(1 * 4 + jj) * stride + l];
                        const float v2 = v[(2 * 4 + jj) * stride + l];
                        const float v3 = v[(3 * 4 + jj) * stride + l];
                        a0[j] = v0 + v1 + v2;
                        a1[j] = v1 - v2 - v3;
                    }
                    const float y00 = a0[0] + a0[1] + a0[2];
                    const float y01 = a0[1] - a0[2] - a0[3];
                    const float y10 = a1[0] + a1[1] + a1[2];
                    const float y11 = a1[1] - a1[2] - a1[3];
                    row0[2 * l] = add ? row0[2 * l] + y00 : y00;
                    row0[2 * l + 1] = add ? row0[2 * l + 1] + y01 : y01;
                    row1[2 * l] = add ? row1[2 * l] + y10 : y10;
                    row1[2 * l + 1] = add ? row1[2 * l + 1] + y11 : y11;
                }
                t += whole;
                continue;
            }
            // A tile at an edge of the output, or in shuffled lanes: on its own.
            const std::int64_t l = t - t0;
            const std::int64_t lane = shuffle ? shuffled(l) : l;
            std::array<float, kXi> v{};
            for (std::size_t xi = 0; xi < kXi; ++xi) {
                v[xi] = m[static_cast<std::int64_t>(xi) * stride + lane];
            }
            std::array<float, 4> a0{};
            std::array<float, 4> a1{};
            for (std::size_t j = 0; j < 4; ++j) {
                a0[j] = v[0 * 4 + j] + v[1 * 4 + j] + v[2 * 4 + j];
                a1[j] = v[1 * 4 + j] - v[2 * 4 + j] - v[3 * 4 + j];
            }
            const auto put = [&](float & p, float x) { p = add ? p + x : x; };
            const std::int64_t yy = 2 * ty;
            const std::int64_t xx = 2 * tx;
            const Elements<float> row0 = output(yy, 0);
            put(row0[xx], a0[0] + a0[1] + a0[2]);
            if (xx + 1 < c.ow) {
                put(row0[xx + 1], a0[1] - a0[2] - a0[3]);
            }
            if (yy + 1 < c.oh) {
                const Elements<float> row1 = output(yy + 1, 0);
                put(row1[xx], a1[0] + a1[1] + a1[2]);
                if (xx + 1 < c.ow) {
                    put(row1[xx + 1], a1[1] - a1[2] - a1[3]);
                }
            }
            ++t;
        }
        // The chain after the convolution, over the rows of outputs the tiles
        // wrote.
        if (chain.count > 0) {
            for (std::int64_t t = t0; t < t1;) {
                const std::int64_t ty = t / c.tw;
                const std::int64_t txa = t % c.tw;
                const std::int64_t txb = std::min(c.tw, txa + (t1 - t));
                const std::int64_t x0 = 2 * txa;
                const std::int64_t x1 = std::min(c.ow, 2 * txb);
                for (std::int64_t yy = 2 * ty; yy < std::min(c.oh, 2 * ty + 2); ++yy) {
                    chain.apply_positions(output(yy, x0).first(x1 - x0), yy * c.ow + x0, r, n);
                }
                t += txb - txa;
            }
        }
    }
}

// Small images: the transformed inputs of channels [k0, k1) of image n, every
// tile of it, B^T d B as bt() computes it, element xi of the tile that is g-th
// of all images at v[xi * inputs_stride() + ((g / kNr) * ic + k) * kNr + g % kNr]
// -- the panels of B of each xi, [panel][ic][kNr]. The lanes past the last tile of
// the last image are zeros. `scratch` holds a channel of the image with the
// zeros around it that the tiles reach.
void transform_inputs_small(Elements<float> v, const Conv & c, std::int64_t panels, std::int64_t n, std::int64_t k0,
                            std::int64_t k1, Elements<float> scratch) {
    const std::int64_t sw = 2 * c.tw + 2;
    const std::int64_t sh = 2 * c.th + 2;
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t last = c.n * tiles;
    const std::int64_t vstride = inputs_stride(panels, c.ic);
    const auto at = [&](std::int64_t xi, std::int64_t g, std::int64_t k) -> float & {
        return v[xi * vstride + ((g / kNr) * c.ic + k) * kNr + g % kNr];
    };
    for (std::int64_t k = k0; k < k1; ++k) {
        const Elements<const float> plane = c.x.from(input_plane(c, n, k));
        for (std::int64_t y = 0; y < sh; ++y) {
            const Elements<float> row = scratch.part(y * sw, sw);
            const std::int64_t iy = y - c.p1;
            for (std::int64_t x = 0; x < sw; ++x) {
                row[x] = input_at(c, plane, iy, x - c.p0);
            }
        }
        for (std::int64_t t = 0; t < tiles; ++t) {
            const Elements<const float> d = scratch.from(2 * (t / c.tw) * sw + 2 * (t % c.tw));
            const std::int64_t g = n * tiles + t;
            for (int s = 0; s < 4; ++s) {
                std::array<float, 4> row{};
                for (std::size_t i = 0; i < 4; ++i) {
                    const Elements<const float> di = d.from(static_cast<std::int64_t>(i) * sw);
                    row[i] = bt(s, di[0], di[1], di[2], di[3]);
                }
                for (int r = 0; r < 4; ++r) {
                    at(r * 4 + s, g, k) = bt(r, row[0], row[1], row[2], row[3]);
                }
            }
        }
        if (n == c.n - 1) {
            for (std::int64_t g = last; g < panels * kNr; ++g) {
                for (std::int64_t xi = 0; xi < kXi; ++xi) {
                    at(xi, g, k) = 0.0f;
                }
            }
        }
    }
}

// G g G^T of the kernels of output channels [r0, r0 + rows), rows <= kMr,
// and input channels [k0, k1), into a panel of A of each xi, [k1 - k0][kMr]:
// that of xi `stride` floats after that of xi - 1. Each kWidened input channels
// are widened first into `widened`, [channel][3 * 3][kMr], and their kernels
// transformed kMr at a time, a channel to a lane: G g first, a column of g at a
// time, then (G g) G^T, a row at a time, each sum of three taken left to right
// and halved after.
void transform_weights(Elements<float> dst, std::int64_t stride, const Conv & c, std::int64_t r0, std::int64_t rows,
                       std::int64_t k0, std::int64_t k1, Elements<float> widened) {
    SDASM_REQUIRE(dst.size() >= (kXi - 1) * stride + (k1 - k0) * kMr && widened.size() >= kWidenedFloats);
    for (std::int64_t s0 = k0; s0 < k1; s0 += kWidened) {
        const std::int64_t s1 = std::min(k1, s0 + kWidened);
        pack_weight_rows(widened, c, r0, rows, s0 * 9, s1 * 9, kMr);
#if SDASM_X86_64
        if (avx2_kernels()) {
            // The kernel writes a panel's channels eight floats at a time, two
            // past the six of the last: into the next panel's, or the skew
            // after the last.
            sdasm_winograd_weights_avx2(dst.from((s0 - k0) * kMr).data(), bytes_of<float>(stride), widened.data(), s1 - s0);
            continue;
        }
#endif
        for (std::int64_t k = s0; k < s1; ++k) {
            // The nine numbers of the kernels of the panel, kMr side by side.
            const Elements<const float> g = widened.part((k - s0) * 9 * kMr, 9 * kMr);
            std::array<std::array<std::array<float, kMr>, 3>, 4> t{};  // G g
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t lane = 0; lane < kMr; ++lane) {
                    const auto jj = static_cast<std::int64_t>(j);
                    const auto ll = static_cast<std::int64_t>(lane);
                    const float g0 = g[(0 * 3 + jj) * kMr + ll];
                    const float g1 = g[(1 * 3 + jj) * kMr + ll];
                    const float g2 = g[(2 * 3 + jj) * kMr + ll];
                    t[0][j][lane] = g0;
                    t[1][j][lane] = 0.5f * (g0 + g1 + g2);
                    t[2][j][lane] = 0.5f * (g0 - g1 + g2);
                    t[3][j][lane] = g2;
                }
            }
            const Elements<float> out = dst.from((k - k0) * kMr);
            for (std::size_t i = 0; i < 4; ++i) {
                const auto ii = static_cast<std::int64_t>(i);
                for (std::size_t lane = 0; lane < kMr; ++lane) {
                    const auto ll = static_cast<std::int64_t>(lane);
                    const float a = t[i][0][lane];
                    const float b = t[i][1][lane];
                    const float d = t[i][2][lane];
                    out[(ii * 4 + 0) * stride + ll] = a;
                    out[(ii * 4 + 1) * stride + ll] = 0.5f * (a + b + d);
                    out[(ii * 4 + 2) * stride + ll] = 0.5f * (a - b + d);
                    out[(ii * 4 + 3) * stride + ll] = d;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- on AMX

bool on_amx(const Conv & c) {
    return split_form(Split_use::winograd).kernel != nullptr && c.wtype == GGML_TYPE_F16;
}

// Elements of two bytes in a block of k of a tile of AMX-BF16's form.
constexpr std::int64_t kTileHalves = kAmxTileBytes / 2;

// The elements of the tiles of A of sixteen output channels over all input
// channels, and of all of an xi, the channels rounded up to 32.
std::int64_t amx_row_tile(const Conv & c) {
    return round_up(c.ic, kAmxBlock) / kAmxBlock * kTileHalves;
}

std::int64_t amx_xi(const Conv & c, std::int64_t rows) {
    return round_up(rows, kAmxBlock) / kAmxTile * amx_row_tile(c);
}

// The scratch of the transform of the weights into tiles: the kernels of
// sixteen output channels by 32 input channels, widened, [32 * 3 * 3][16], and
// their transforms, [xi][32][16].
constexpr std::int64_t kAmxWidened = kAmxBlock * 9 * kAmxTile;
constexpr std::int64_t kAmxRowsFloats = kAmxWidened + kXi * kAmxBlock * kAmxTile;

// The tiles of A of output channels [r0, r0 + 16) over input channels [k0, k1),
// k0 a multiple of 32: those of xi `xi` elements after those of xi - 1, those of
// the block of k at k0 at (k0 - kbase) / 32 * kTileHalves after. The kernels of
// a block of 32 input channels widened side by side (pack_rows_f16()), then
// transformed, sixteen output channels at a time, and split into the tiles
// (x86_64/amx_winograd.S); the output channels past the last, and the input
// channels, transform to zeros.
void amx_weights_tile(Elements<std::uint16_t> dst, std::int64_t xi, std::int64_t kbase, const Conv & c, std::int64_t r0,
                      std::int64_t k0, std::int64_t k1, Elements<float> scratch) {
    const Elements<float> widened = scratch.first(kAmxWidened);
    const Elements<float> transformed = scratch.part(kAmxWidened, kAmxRowsFloats - kAmxWidened);
    const std::int64_t rows = std::clamp<std::int64_t>(c.oc - r0, 0, kAmxTile);
    for (std::int64_t kb = k0; kb < k1; kb += kAmxBlock) {
        const std::int64_t kc = std::min(kAmxBlock, c.ic - kb);
        const Elements<std::uint16_t> out = dst.from((kb - kbase) / kAmxBlock * kTileHalves);
        SDASM_CHECK(out.size() >= (kXi - 1) * xi + kTileHalves);
        if (rows == 0 || kc <= 0) {
            for (std::int64_t i = 0; i < kXi; ++i) {
                fill_with(out.part(i * xi, kTileHalves), std::uint16_t{0});
            }
            continue;
        }
        pack_rows_f16(widened, c.w16.from(r0 * c.wnb3), c.wnb3, rows, kb * 9, (kb + kc) * 9, kAmxTile);
        sdasm_amx_winograd_weights(out.data(), bytes_of<std::uint16_t>(xi), widened.data(), kc, transformed.data());
    }
}

struct Amx_plan {
    std::int64_t mc, kc, tiles, mblocks, tblocks;
    std::int64_t stride;  // the floats between the panels of the inputs of two xi
    std::int64_t panels;  // small images: the panels of sixteen tiles of all images, rounded up to two
};

Amx_plan amx_plan(const Conv & c, int n_threads) {
    const Tuning & t = tuning();
    Amx_plan pl{};
    if (small(c)) {
        pl.panels = round_up(ceil_div(c.n * c.tw * c.th, kNr), 2);
        pl.kc = std::min(round_up(c.ic, kAmxBlock), round_up(std::max(t.winograd_small_kc, 1), kAmxBlock));
        pl.mc = std::min(round_up(c.oc, kAmxBlock), round_up(std::max(t.winograd_small_mc, 1), kAmxBlock));
        return pl;
    }
    pl.kc = std::min(round_up(c.ic, kAmxBlock), round_up(std::max(t.winograd_kc, 1), kAmxBlock));
    pl.mblocks = ceil_div(c.oc, t.winograd_mc);
    pl.mc = round_up(ceil_div(c.oc, pl.mblocks), kAmxBlock);
    pl.mblocks = ceil_div(c.oc, pl.mc);
    pl.tiles = round_up(std::min<std::int64_t>(t.winograd_tiles, c.tw * c.th), kNr);
    pl.tblocks = ceil_div(c.tw * c.th, pl.tiles);
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    while (c.n * pl.mblocks * pl.tblocks < wanted && pl.tiles > kNr) {
        pl.tiles = round_up(pl.tiles / 2, kNr);
        pl.tblocks = ceil_div(c.tw * c.th, pl.tiles);
    }
    pl.stride = skewed(pl.kc * pl.tiles);
    return pl;
}

// The work buffer on AMX. Large images: the tiles of A of every xi, shared;
// each thread's inputs of four xi in single precision, the tiles of B of an
// xi, its products -- or G g G^T of sixteen channels by 32, before the products
// begin -- and the block it copies a part of. Small images: the inputs of every
// xi in single precision and their tiles of B, shared; each thread's tiles of A
// of every xi, G g G^T, a channel of an image with its padding, its products
// and the block it copies a part of.
struct Amx_layout {
    Work_layout layout;
    Shared<std::uint16_t> weights;  // large images
    Shared<float> inputs;           // small images
    Shared<std::uint16_t> inputs_amx;
    Own<float> panels;
    Own<std::uint16_t> columns;
    Own<float> products;
    Own<float> rows;     // small images
    Own<float> scratch;  // small images
    Own<float> spare;
};

Amx_layout amx_layout(const Conv & c, const Amx_plan & pl) {
    Amx_layout l;
    if (small(c)) {
        l.inputs = l.layout.shared<float>(kXi * inputs_stride(pl.panels, c.ic));
        l.inputs_amx = l.layout.shared<std::uint16_t>(kXi * pl.panels * amx_row_tile(c));
        l.columns = l.layout.own<std::uint16_t>(kXi * pl.mc / kAmxTile * (pl.kc / kAmxBlock) * kTileHalves);
        l.rows = l.layout.own<float>(kAmxRowsFloats);
        l.scratch = l.layout.own<float>((2 * c.th + 2) * (2 * c.tw + 2));
        l.products = l.layout.own<float>(kXi * products_stride(pl.mc, pl.panels * kNr));
        l.spare = l.layout.own<float>(kAmxBlock * kAmxBlock);
        return l;
    }
    l.weights = l.layout.shared<std::uint16_t>(kXi * amx_xi(c, c.oc));
    l.panels = l.layout.own<float>(4 * pl.stride);
    l.columns = l.layout.own<std::uint16_t>(round_up(pl.tiles, kAmxBlock) / kAmxTile * (pl.kc / kAmxBlock) * kTileHalves);
    l.products = l.layout.own<float>(std::max(kXi * products_stride(pl.mc, pl.tiles), kAmxRowsFloats));
    l.spare = l.layout.own<float>(kAmxBlock * kAmxBlock);
    return l;
}

// The products of the tiles of A of an xi, rows [i0, i1) of them from `a`
// (row_tile elements between the tiles of sixteen rows), with the tiles of B
// of its columns [0, nt) from `b` (column elements between the tiles of
// sixteen columns), over `blocks` blocks of k, into [row][column] at `out`,
// rows nt floats apart; added to it when accumulate says so.
void amx_xi_products(const Split_form & f, Elements<const std::uint16_t> a, std::int64_t row_tile,
                     Elements<const std::uint16_t> b, std::int64_t column, std::int64_t blocks, std::int64_t i0,
                     std::int64_t i1, std::int64_t nt, Elements<float> out, bool accumulate, Elements<float> spare) {
    for (std::int64_t jj = 0; jj < nt; jj += kAmxBlock) {
        for (std::int64_t ii = i0; ii < i1; ii += kAmxBlock) {
            split_block(f, blocks, a.from((ii - i0) / kAmxTile * row_tile), row_tile, b.from(jj / kAmxTile * column), column,
                        out.from((ii - i0) * nt + jj), nt, accumulate, std::min(kAmxBlock, i1 - ii),
                        std::min(kAmxBlock, nt - jj), spare);
        }
    }
}

void compute_large_amx(const ggml_compute_params * const params, const Conv & c, const Amx_plan & pl,
                       const Epilogue & chain) {
    const Amx_layout l = amx_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const std::int64_t xi_tiles = amx_xi(c, c.oc);
    const std::int64_t row_tile = amx_row_tile(c);
    const std::shared_ptr<Packed> kept = packed(c.kernel, Layout::amx_winograd, kXi * xi_tiles);
    const bool fill = kept == nullptr || !kept->ready();
    const Split_form f = split_form(Split_use::winograd);

    Work_queue queue(params, buffer);
    const Elements<std::uint16_t> weights = kept != nullptr ? kept->halves() : buffer[l.weights];
    const Elements<float> panels = buffer[l.panels];
    const Elements<std::uint16_t> columns = buffer[l.columns];
    const Elements<float> products = buffer[l.products];
    const Elements<float> rows = products;  // G g G^T, before the products begin
    const Elements<float> spare = buffer[l.spare];
    // The weights, a tile of sixteen output channels to an item, over every
    // input channel: those past the last are zeros.
    const std::int64_t pack_items = fill ? round_up(c.oc, kAmxBlock) / kAmxTile : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            amx_weights_tile(weights.from(item * row_tile), xi_tiles, 0, c, item * kAmxTile, 0, round_up(c.ic, kAmxBlock),
                             rows);
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }

    const std::int64_t tiles = c.tw * c.th;
    const std::vector<Block> blocks = guided_blocks(c.n, tiles, kNr, pl.tiles, params->nth);
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
        const std::int64_t t0 = block.start;
        const std::int64_t t1 = t0 + block.size;
        const std::int64_t nt = round_up(t1 - t0, kNr);
        const std::int64_t pstride = products_stride(pl.mc, nt);
        for (int r = 0; r < 4; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                const std::int64_t kb = ceil_div(kc, kAmxBlock);
                const std::int64_t column = kb * kTileHalves;
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs(panels, pl.stride, c, n, t0, t1, k0, k0 + kc, r, false);
                }
                for (std::int64_t s = 0; s < 4; ++s) {
                    const std::int64_t xi = 4 * r + s;
                    {
                        // The tiles of B of this xi, from its panels; one more of
                        // zeros past an odd number of them, for the last pair.
                        const Phase_scope scope(Phase::pack_inputs);
                        for (std::int64_t p = 0; p < nt / kNr; ++p) {
                            const Elements<std::uint16_t> dst = columns.part(p * column, column);
                            const Elements<const float> src = panels.part(s * pl.stride + p * kc * kNr, kc * kNr);
                            sdasm_amx_pack_panel(dst.data(), src.data(), kc);
                        }
                        if ((nt / kNr) % 2 != 0) {
                            fill_with(columns.part((nt / kNr) * column, column), std::uint16_t{0});
                        }
                    }
                    const Phase_scope scope(Phase::product);
                    amx_xi_products(f, weights.from(xi * xi_tiles + i0 / kAmxTile * row_tile + k0 / kAmxBlock * kTileHalves),
                                    row_tile, columns, column, kb, i0, i1, nt, products.from(xi * pstride), k0 > 0, spare);
                }
            }
        }
        transform_outputs(c, products, pstride, nt, n, t0, t1, i0, i1, chain, false);
    }
    if (configured) {
        split_end(f);
    }
}

void compute_small_amx(const ggml_compute_params * const params, const Conv & c, const Amx_plan & pl,
                       const Epilogue & chain) {
    const Amx_layout l = amx_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const Split_form f = split_form(Split_use::winograd);
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t nt = pl.panels * kNr;
    const std::int64_t panel = amx_row_tile(c);  // a panel of sixteen tiles over every input channel

    // The inputs in single precision, by all threads together, sixteen channels
    // of an image to an item; then their tiles of B, an xi and a pair of panels
    // to an item.
    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params, buffer);
    const Elements<float> inputs = buffer[l.inputs];
    const Elements<std::uint16_t> inputs_amx = buffer[l.inputs_amx];
    const Elements<std::uint16_t> weights = buffer[l.columns];
    const Elements<float> rows = buffer[l.rows];
    const Elements<float> scratch = buffer[l.scratch];
    const Elements<float> products = buffer[l.products];
    const Elements<float> spare = buffer[l.spare];
    const std::int64_t groups = ceil_div(c.ic, kInputChannels);
    const std::int64_t input_items = c.n * groups;
    int item = queue.next();
    for (; item < input_items; item = queue.next()) {
        const std::int64_t n = item / groups;
        const std::int64_t k0 = (item % groups) * kInputChannels;
        const Phase_scope scope(Phase::pack_inputs);
        transform_inputs_small(inputs, c, pl.panels, n, k0, std::min(c.ic, k0 + kInputChannels), scratch);
    }
    queue.barrier();
    const std::int64_t tile_items = kXi * pl.panels / 2;
    for (; item < input_items + tile_items; item = queue.next()) {
        const std::int64_t xi = (item - input_items) / (pl.panels / 2);
        const std::int64_t pair = (item - input_items) % (pl.panels / 2);
        const Phase_scope scope(Phase::pack_inputs);
        for (std::int64_t p = 2 * pair; p < 2 * pair + 2; ++p) {
            const Elements<std::uint16_t> dst = inputs_amx.part((xi * pl.panels + p) * panel, panel);
            const Elements<const float> src = inputs.part(xi * inputs_stride(pl.panels, c.ic) + p * c.ic * kNr, c.ic * kNr);
            sdasm_amx_pack_panel(dst.data(), src.data(), c.ic);
        }
    }
    queue.barrier();

    // The blocks of output channels, each over every tile, transforming the
    // weights of its channels a stretch of the input channels at a time.
    const std::vector<Block> blocks = guided_blocks(1, c.oc, kAmxBlock, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    const std::int64_t pstride = products_stride(pl.mc, nt);
    bool configured = false;
    for (; item < input_items + tile_items + units; item = queue.next()) {
        if (!configured) {
            split_begin(f);
            configured = true;
        }
        const Block & block = blocks[static_cast<std::size_t>(item - input_items - tile_items)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, round_up(c.ic, kAmxBlock) - k0);
            const std::int64_t kb = kc / kAmxBlock;
            const std::int64_t row_tile = kb * kTileHalves;
            const std::int64_t xi_tiles = round_up(i1 - i0, kAmxBlock) / kAmxTile * row_tile;
            {
                const Phase_scope scope(Phase::pack_weights);
                for (std::int64_t r0 = i0; r0 < i0 + round_up(i1 - i0, kAmxBlock); r0 += kAmxTile) {
                    amx_weights_tile(weights.from((r0 - i0) / kAmxTile * row_tile), xi_tiles, k0, c, r0, k0, k0 + kc, rows);
                }
            }
            const Phase_scope scope(Phase::product);
            for (std::int64_t xi = 0; xi < kXi; ++xi) {
                amx_xi_products(f, weights.from(xi * xi_tiles), row_tile,
                                inputs_amx.from(xi * pl.panels * panel + k0 / kAmxBlock * kTileHalves), panel, kb, i0, i1,
                                nt, products.from(xi * pstride), k0 > 0, spare);
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs(c, products.from(n * tiles), pstride, nt, n, 0, tiles, i0, i1, chain, false);
        }
    }
    if (configured) {
        split_end(f);
    }
}

// The weights of the next stretch of input channels [next0, next1) of output
// channels [r0, r1), fetched into the cache while the products of this one are
// computed: their nine numbers of each input channel, a line of the cache at a
// time. A hint to the processor, which reads nothing it would not.
void prefetch_weights(const Conv & c, std::int64_t r0, std::int64_t r1, std::int64_t next0, std::int64_t next1) {
    if (next0 >= next1) {
        return;
    }
    const auto lines = [&](auto weights, std::int64_t per_line) {
        for (std::int64_t r = r0; r < r1; ++r) {
            const auto row = weights.part(r * c.wnb3 + next0 * 9, (next1 - next0) * 9);
            for (std::int64_t e = 0; e < row.size(); e += per_line) {
                __builtin_prefetch(row.from(e).data(), 0, 2);
            }
            __builtin_prefetch(row.from(row.size() - 1).data(), 0, 2);
        }
    };
    if (c.wtype == GGML_TYPE_F16) {
        lines(c.w16, 32);
    }
    else {
        lines(c.w32, 16);
    }
}

void compute_small(const ggml_compute_params * const params, const Conv & c, const Plan & pl, const Epilogue & chain) {
    const Single_layout l = single_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t nt = pl.panels * kNr;
    const std::int64_t vstride = inputs_stride(pl.panels, c.ic);
    const std::int64_t pstride = products_stride(pl.mc, nt);

    // The inputs, by all threads together, sixteen channels of an image to an
    // item.
    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params, buffer);
    const Elements<float> inputs = buffer[l.shared];
    const Elements<float> widened = buffer[l.panels];
    const Elements<float> weights = buffer[l.weights];
    const Elements<float> scratch = buffer[l.scratch];
    const Elements<float> products = buffer[l.products];
    const std::int64_t groups = ceil_div(c.ic, kInputChannels);
    const std::int64_t input_items = c.n * groups;
    int item = queue.next();
    for (; item < input_items; item = queue.next()) {
        const std::int64_t n = item / groups;
        const std::int64_t k0 = (item % groups) * kInputChannels;
        const Phase_scope scope(Phase::pack_inputs);
        transform_inputs_small(inputs, c, pl.panels, n, k0, std::min(c.ic, k0 + kInputChannels), scratch);
    }
    queue.barrier();

    // The blocks of output channels, each over every tile.
    const std::vector<Block> blocks = guided_blocks(1, c.oc, kMr, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    for (; item < input_items + units; item = queue.next()) {
        const Block & block = blocks[static_cast<std::size_t>(item - input_items)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.ic - k0);
            const std::int64_t stride = xi_stride(pl.mc, kc);
            {
                // Panel after panel, so that what a panel's last stores write
                // past it is written over by the next one's.
                const Phase_scope scope(Phase::pack_weights);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    transform_weights(weights.from((ii - i0) * kc), stride, c, ii, std::min<std::int64_t>(kMr, i1 - ii), k0,
                                      k0 + kc, widened);
                }
            }
            const Phase_scope scope(Phase::product);
            // The weights of the next stretch of input channels, fetched into
            // the cache while the products of this one are computed, a
            // sixteenth of the rows at each xi: read when they are needed, they
            // are short stretches of many rows, which the processor fetches no
            // faster than it reads them.
            const std::int64_t next0 = k0 + kc;
            const std::int64_t next1 = tuning().prefetch != 0 ? std::min(c.ic, next0 + pl.kc) : next0;
            const std::int64_t rows_per_xi = ceil_div(i1 - i0, kXi);
            for (std::int64_t xi = 0; xi < kXi; ++xi) {
                prefetch_weights(c, i0 + xi * rows_per_xi, std::min(i1, i0 + (xi + 1) * rows_per_xi), next0, next1);
                const Elements<const float> u = weights.from(xi * stride);
                const Elements<float> product = products.from(xi * pstride);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    for (std::int64_t p = 0; p < pl.panels; p += panel_step(wide) / kNr) {
                        sdasm::tiles(uk, wide, kc, u.from((ii - i0) * kc), inputs.from(xi * vstride + (p * c.ic + k0) * kNr),
                                     c.ic * kNr, product.from((ii - i0) * nt + p * kNr), nt, k0 > 0,
                                     std::min<std::int64_t>(kMr, i1 - ii), std::min(panel_step(wide), nt - p * kNr));
                    }
                }
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs(c, products.from(n * tiles), pstride, nt, n, 0, tiles, i0, i1, chain, false);
        }
    }
}

// ---------------------------------------------------------------- F(4x4, 3x3)

// Winograd's F(4x4, 3x3) for the images of many tiles, on the vector units
// (Tuning::winograd = 2): tiles of four by four outputs from six by six inputs,
// 36 elements xi to the sixteen outputs of a tile, 2.25 multiplications an
// output, where F(2x2, 3x3) makes 4 and the direct convolution 9: 1.78 times
// fewer than F(2x2, 3x3). Its points are 0, 1, -1, 2, -1/2 and infinity,
// which of the sets measured round least:
//
//     A^T = | 1  1  1  1    1    0 |    G = |   1      0      0   |
//           | 0  1 -1  2  -1/2   0 |        | -1/3   -1/3   -1/3  |
//           | 0  1  1  4   1/4   0 |        |  1/3   -1/3    1/3  |
//           | 0  1 -1  8  -1/8   1 |        |  1/15   2/15   4/15 |
//                                           | -16/15  8/15  -4/15 |
//     B^T = | 1  3/2   -2   -3/2   1   0 |  |   0      0      1   |
//           | 0  -1   -5/2  -1/2   1   0 |
//           | 0   1    1/2  -5/2   1   0 |
//           | 0 -1/2   -1    1/2   1   0 |
//           | 0   2    -1    -2    1   0 |
//           | 0   1    3/2   -2  -3/2  1 |
//
// The transforms are larger, and round more: measured on random images, an
// output is within 7 10^-7 of the magnitudes summed at worst, where F(2x2,
// 3x3)'s are within 1 10^-7 and ggml's, of the image rounded to FP16, 2 to 4
// 10^-5.

constexpr int kXi4 = 36;
using Row6 = std::array<float, 6>;
constexpr std::array<Row6, 6> kBT4 = {{{1.0f, 1.5f, -2.0f, -1.5f, 1.0f, 0.0f},
                                       {0.0f, -1.0f, -2.5f, -0.5f, 1.0f, 0.0f},
                                       {0.0f, 1.0f, 0.5f, -2.5f, 1.0f, 0.0f},
                                       {0.0f, -0.5f, -1.0f, 0.5f, 1.0f, 0.0f},
                                       {0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f},
                                       {0.0f, 1.0f, 1.5f, -2.0f, -1.5f, 1.0f}}};
constexpr std::array<Row6, 4> kAT4 = {{{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
                                       {0.0f, 1.0f, -1.0f, 2.0f, -0.5f, 0.0f},
                                       {0.0f, 1.0f, 1.0f, 4.0f, 0.25f, 0.0f},
                                       {0.0f, 1.0f, -1.0f, 8.0f, -0.125f, 1.0f}}};

float bt4(std::int64_t i, std::int64_t j) {
    return kBT4[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}
float at4(std::int64_t i, std::int64_t j) {
    return kAT4[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}

// For FP16 weights alone, whose products ggml computes of the image rounded to
// FP16: those of FP32 weights it computes in single precision, closer than
// F(4x4, 3x3) can.
// The kernels written by hand for AVX2 leave the eight tiles of each eight of a
// panel in the order 0, 2, 4, 6, 1, 3, 5, 7 (x86_64/winograd4_avx2.S): tile l
// of a block in lane shuffled4(l).
constexpr std::array<std::int64_t, 8> kShuffle4 = {0, 4, 1, 5, 2, 6, 3, 7};

std::int64_t shuffled4(std::int64_t l) {
    return (l & ~std::int64_t{7}) | kShuffle4[static_cast<std::size_t>(l & 7)];
}

bool four(const Conv & c) {
    return tuning().winograd == 2 && c.wtype == GGML_TYPE_F16 && !on_amx(c) && c.ow * c.oh >= tuning().winograd4_positions;
}

struct Plan4 {
    std::int64_t tw, th;  // tiles of four by four across and down
    std::int64_t mc, kc, tiles, mblocks, tblocks;
    std::int64_t stride;  // the floats between the panels of the inputs of two xi
};

Plan4 plan4(const Conv & c, int n_threads) {
    const Tuning & t = tuning();
    Plan4 pl{};
    pl.tw = (c.ow + 3) / 4;
    pl.th = (c.oh + 3) / 4;
    pl.kc = std::min<std::int64_t>(c.ic, t.winograd_kc);
    pl.mblocks = ceil_div(c.oc, t.winograd_mc);
    pl.mc = round_up(ceil_div(c.oc, pl.mblocks), kMr);
    pl.mblocks = ceil_div(c.oc, pl.mc);
    pl.tiles = round_up(std::min<std::int64_t>(t.winograd_tiles, pl.tw * pl.th), kNr);
    pl.tblocks = ceil_div(pl.tw * pl.th, pl.tiles);
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(n_threads);
    while (c.n * pl.mblocks * pl.tblocks < wanted && pl.tiles > kNr) {
        pl.tiles = round_up(pl.tiles / 2, kNr);
        pl.tblocks = ceil_div(pl.tw * pl.th, pl.tiles);
    }
    pl.stride = skewed(pl.kc * pl.tiles);
    return pl;
}

// The transformed weights of every xi, shared; each thread's panels of the
// inputs of six xi -- where its widened weights go first -- and its products.
struct Layout4 {
    Work_layout layout;
    Shared<float> weights;
    Own<float> panels;
    Own<float> products;
};

Layout4 layout4(const Conv & c, const Plan4 & pl) {
    Layout4 l;
    l.weights = l.layout.shared<float>(kXi4 * xi_stride(round_up(c.oc, kMr), c.ic));
    l.panels = l.layout.own<float>(std::max(6 * pl.stride, round_up(kWidenedFloats, kSkew)));
    l.products = l.layout.own<float>(kXi4 * products_stride(pl.mc, pl.tiles));
    return l;
}

// G g G^T of the kernels of output channels [r0, r0 + rows), rows <= kMr, and
// input channels [k0, k1), into a panel of A of each xi, [k1 - k0][kMr]: that
// of xi `stride` floats after that of xi - 1. The kernels of each eight input
// channels are widened first into `widened`, [channel][3 * 3][kMr]
// (pack_rows_f16()), and transformed kMr at a time, a lane to each output
// channel, in single precision: G g a column at a time, then (G g) G^T a row
// at a time.
using Row3 = std::array<float, 3>;
constexpr std::array<Row3, 6> kG4f = {{{1.0f, 0.0f, 0.0f},
                                       {-1.0f / 3, -1.0f / 3, -1.0f / 3},
                                       {1.0f / 3, -1.0f / 3, 1.0f / 3},
                                       {1.0f / 15, 2.0f / 15, 4.0f / 15},
                                       {-16.0f / 15, 8.0f / 15, -4.0f / 15},
                                       {0.0f, 0.0f, 1.0f}}};

void transform_weights4(Elements<float> dst, std::int64_t stride, const Conv & c, std::int64_t r0, std::int64_t rows,
                        std::int64_t k0, std::int64_t k1, Elements<float> widened) {
    SDASM_REQUIRE(dst.size() >= (kXi4 - 1) * stride + (k1 - k0) * kMr && widened.size() >= kWidenedFloats);
    for (std::int64_t s0 = k0; s0 < k1; s0 += kWidened) {
        const std::int64_t s1 = std::min(k1, s0 + kWidened);
        pack_weight_rows(widened, c, r0, rows, s0 * 9, s1 * 9, kMr);
#if SDASM_X86_64
        if (avx2_kernels()) {
            sdasm_winograd4_weights_avx2(dst.from((s0 - k0) * kMr).data(), bytes_of<float>(stride), widened.data(), s1 - s0);
            continue;
        }
#endif
        for (std::int64_t k = s0; k < s1; ++k) {
            const Elements<const float> g = widened.part((k - s0) * 9 * kMr, 9 * kMr);
            std::array<std::array<std::array<float, kMr>, 3>, 6> t{};  // G g
            for (std::size_t i = 0; i < 6; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    for (std::size_t lane = 0; lane < kMr; ++lane) {
                        const auto jj = static_cast<std::int64_t>(j);
                        const auto ll = static_cast<std::int64_t>(lane);
                        t[i][j][lane] = kG4f[i][0] * g[(0 * 3 + jj) * kMr + ll] + kG4f[i][1] * g[(1 * 3 + jj) * kMr + ll] +
                                        kG4f[i][2] * g[(2 * 3 + jj) * kMr + ll];
                    }
                }
            }
            const Elements<float> out = dst.from((k - k0) * kMr);
            for (std::size_t i = 0; i < 6; ++i) {
                for (std::size_t j = 0; j < 6; ++j) {
                    for (std::size_t lane = 0; lane < kMr; ++lane) {
                        out[static_cast<std::int64_t>(i * 6 + j) * stride + static_cast<std::int64_t>(lane)] =
                            t[i][0][lane] * kG4f[j][0] + t[i][1][lane] * kG4f[j][1] + t[i][2][lane] * kG4f[j][2];
                    }
                }
            }
        }
    }
}

#if SDASM_X86_64
// The rows of image n, channel k0, that row r of B^T takes of the tiles whose
// first input is at row y0, column x0, for the kernels written by hand: each
// row's first number and its coefficient; the bytes from each to the tiles'
// first input; the bytes to the next channel's rows. The rows outside the
// image, and those of coefficient 0, are left out.
void rows4(Winograd4_inputs & job, const Conv & c, std::int64_t n, std::int64_t k0, std::int64_t kc, std::int64_t y0,
           std::int64_t x0, std::int64_t r) {
    job.step = bytes_of<float>(c.xnb2);
    job.channels = kc;
    job.offset = bytes_of<float>(x0);
    job.rows = 0;
    for (std::int64_t i = 0; i < 6; ++i) {
        const std::int64_t y = y0 + i;
        if (bt4(r, i) == 0.0f || y < 0 || y >= c.ih) {
            continue;
        }
        const Elements<const float> first = c.x.from(input_plane(c, n, k0) + y * c.xnb1);
        SDASM_CHECK(first.size() >= (kc - 1) * c.xnb2 + c.iw);
        const auto at = static_cast<std::size_t>(job.rows);
        job.row[at] = first.data();
        job.coefficient[at] = bt4(r, i);
        ++job.rows;
    }
}
#endif

// The elements xi = (r, s), s = 0..5, of the transformed inputs of tiles [t0,
// t1) of image n and channels [k0, k1), into panels of kNr tiles, [panel][k1 -
// k0][kNr], zeros past t1: those of (r, s) `stride` floats after those of
// (r, s - 1). Row r of B^T first takes its combination of the six rows of each
// tile, then each s its combination of the six numbers of that.
void pack_inputs4(Elements<float> dst, std::int64_t stride, const Conv & c, const Plan4 & pl, std::int64_t n,
                  std::int64_t t0, std::int64_t t1, std::int64_t k0, std::int64_t k1, int r, bool shuffle) {
    const std::int64_t kc = k1 - k0;
    SDASM_REQUIRE(dst.size() >= 5 * stride + ceil_div(t1 - t0, kNr) * kc * kNr);
    for (std::int64_t tt = t0; tt < t1; tt += kNr) {
        const Elements<float> out = dst.from((tt - t0) / kNr * kc * kNr);
        const std::int64_t valid = std::min<std::int64_t>(kNr, t1 - tt);
        const std::int64_t row_of_tiles = tt / pl.tw;
        const std::int64_t first = tt % pl.tw;
        const std::int64_t y0 = 4 * row_of_tiles - c.p1;
        const std::int64_t x0 = 4 * first - c.p0;
#if SDASM_X86_64
        if (valid == kNr && first + kNr <= pl.tw && avx512_kernels()) {
            // Sixteen tiles of a row by the kernel for AVX-512: the rows that
            // row r of B^T takes, inside the image, and the lanes of its loads.
            Winograd4_inputs job{};
            job.dst = out.data();
            job.stride = bytes_of<float>(stride);
            rows4(job, c, n, k0, kc, y0, x0, r);
            for (std::size_t m = 0; m < 5; ++m) {
                std::uint16_t lanes = 0;
                for (int lane = 0; lane < 16; ++lane) {
                    const std::int64_t x = x0 + 16 * static_cast<std::int64_t>(m) + lane;
                    const bool inside = x >= 0 && x < c.iw && 16 * static_cast<std::int64_t>(m) + lane < 4 * kNr + 2;
                    lanes = static_cast<std::uint16_t>(lanes | ((inside ? 1u : 0u) << lane));
                }
                job.lanes[m] = lanes;
            }
            sdasm_winograd4_inputs_avx512(&job);
            continue;
        }
        if (valid == kNr && first + kNr <= pl.tw && shuffle) {
            // Sixteen tiles of a row by the kernel for AVX2, eight at a time,
            // with masks for the loads that reach past the image.
            for (std::int64_t half = 0; half < 2; ++half) {
                const std::int64_t xh = x0 + 32 * half;
                Winograd4_inputs job{};
                job.dst = out.from(8 * half).data();
                job.stride = bytes_of<float>(stride);
                rows4(job, c, n, k0, kc, y0, xh, r);
                alignas(32) std::array<std::int32_t, 5 * 8> masks{};
                const bool inside = xh >= 0 && xh + 34 <= c.iw;
                for (std::size_t m = 0; m < 5; ++m) {
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        const std::int64_t at = static_cast<std::int64_t>(8 * m + lane);
                        const std::int64_t x = xh + at;
                        masks[8 * m + lane] = x >= 0 && x < c.iw && at < 34 ? -1 : 0;
                    }
                }
                job.masks = inside ? nullptr : masks.data();
                sdasm_winograd4_inputs_avx2(&job);
            }
            continue;
        }
#endif
        if (valid == kNr && first + kNr <= pl.tw) {
            // Sixteen tiles of a row: the combination of row r of B^T of the
            // rows of the tiles, 66 numbers along the row, zeros outside the
            // image; then d[j] of tile l, its number 4 l + j, by each s.
            constexpr std::int64_t kSpan = 4 * kNr + 2;
            const std::int64_t xa = std::max<std::int64_t>(0, -x0);
            const std::int64_t xb = std::min<std::int64_t>(kSpan, c.iw - x0);
            for (std::int64_t k = k0; k < k1; ++k) {
                const Elements<const float> plane = c.x.from(input_plane(c, n, k));
                std::array<float, kSpan> row{};
                for (std::int64_t i = 0; i < 6; ++i) {
                    const float b = bt4(r, i);
                    const std::int64_t y = y0 + i;
                    if (b == 0.0f || y < 0 || y >= c.ih) {
                        continue;
                    }
                    for (std::int64_t x = xa; x < xb; ++x) {
                        row[static_cast<std::size_t>(x)] += b * plane[y * c.xnb1 + x0 + x];
                    }
                }
                std::array<std::array<float, kNr>, 6> u{};
                for (std::size_t l = 0; l < kNr; ++l) {
                    for (std::size_t j = 0; j < 6; ++j) {
                        u[j][l] = row[4 * l + j];
                    }
                }
                for (std::int64_t s4 = 0; s4 < 6; ++s4) {
                    const Elements<float> o = out.part(s4 * stride + (k - k0) * kNr, kNr);
                    for (std::size_t l = 0; l < kNr; ++l) {
                        float v = 0.0f;
                        for (std::size_t j = 0; j < 6; ++j) {
                            v += bt4(s4, static_cast<std::int64_t>(j)) * u[j][l];
                        }
                        o[static_cast<std::int64_t>(l)] = v;
                    }
                }
            }
            continue;
        }
        for (std::int64_t k = k0; k < k1; ++k) {
            const Elements<const float> plane = c.x.from(input_plane(c, n, k));
            for (std::int64_t l = 0; l < kNr; ++l) {
                std::array<float, 6> o{};
                if (l < valid) {
                    const std::int64_t t = tt + l;
                    const std::int64_t ty0 = 4 * (t / pl.tw) - c.p1;
                    const std::int64_t tx0 = 4 * (t % pl.tw) - c.p0;
                    std::array<float, 6> u{};
                    for (std::int64_t i = 0; i < 6; ++i) {
                        const float b = bt4(r, i);
                        if (b == 0.0f) {
                            continue;
                        }
                        for (std::size_t j = 0; j < 6; ++j) {
                            u[j] += b * input_at(c, plane, ty0 + i, tx0 + static_cast<std::int64_t>(j));
                        }
                    }
                    for (std::size_t s4 = 0; s4 < 6; ++s4) {
                        float v = 0.0f;
                        for (std::size_t j = 0; j < 6; ++j) {
                            v += kBT4[s4][j] * u[j];
                        }
                        o[s4] = v;
                    }
                }
                const std::int64_t lane = shuffle ? shuffled4(l) : l;
                for (std::size_t s4 = 0; s4 < 6; ++s4) {
                    out[static_cast<std::int64_t>(s4) * stride + (k - k0) * kNr + lane] = o[s4];
                }
            }
        }
    }
}

// Y = A^T M A for tiles [t0, t1) of image n and output channels [r0, r1), from
// their 36 products, [xi][channel][tile], those of xi `stride` floats apart;
// added to the outputs when the chain begins its sum from them. Then the chain.
void transform_outputs4(const Conv & c, const Plan4 & pl, Elements<const float> products, std::int64_t stride,
                        std::int64_t nt, std::int64_t n, std::int64_t t0, std::int64_t t1, std::int64_t r0,
                        std::int64_t r1, const Epilogue & chain, bool shuffle) {
    const Phase_scope scope(Phase::transform_out);
    const bool add = chain.from_redirect;
    for (std::int64_t r = r0; r < r1; ++r) {
        const Elements<const float> m = products.from((r - r0) * nt);
        const std::int64_t plane = output_plane(c, n, r);
        const auto output = [&](std::int64_t y, std::int64_t x) { return c.y.from(plane + y * c.ynb1 + x); };
        for (std::int64_t t = t0; t < t1;) {
#if SDASM_X86_64
            const std::int64_t ty = t / pl.tw;
            const std::int64_t tx = t % pl.tw;
            // The tiles of this row of tiles from t on, whole in the output.
            const std::int64_t whole = std::min({t1 - t, pl.tw - tx, (c.ow - 4 * tx) / 4});
            if (!shuffle && whole >= 16 && 4 * ty + 4 <= c.oh && avx512_kernels()) {
                // Sixteen tiles at a time by the kernel for AVX-512.
                const std::int64_t groups = whole / 16;
                SDASM_CHECK(m.size() >= (kXi4 - 1) * stride + (t - t0) + 16 * groups);
                const Elements<float> rows = output(4 * ty, 4 * tx);
                SDASM_CHECK(rows.size() >= 3 * c.ynb1 + 64 * groups);
                sdasm_winograd4_outputs_avx512(m.from(t - t0).data(), bytes_of<float>(stride), rows.data(),
                                               bytes_of<float>(c.ynb1), groups, add ? 1 : 0);
                t += 16 * groups;
                continue;
            }
            if (shuffle && whole >= 8 && (t - t0) % 8 == 0 && 4 * ty + 4 <= c.oh) {
                // Eight tiles at a time, in the shuffled lanes the packing left
                // them in, by the kernel for AVX2.
                const std::int64_t groups = whole / 8;
                SDASM_CHECK(m.size() >= (kXi4 - 1) * stride + (t - t0) + 8 * groups);
                const Elements<float> rows = output(4 * ty, 4 * tx);
                SDASM_CHECK(rows.size() >= 3 * c.ynb1 + 32 * groups);
                sdasm_winograd4_outputs_avx2(m.from(t - t0).data(), bytes_of<float>(stride), rows.data(),
                                             bytes_of<float>(c.ynb1), groups, add ? 1 : 0);
                t += 8 * groups;
                continue;
            }
#endif
            const std::int64_t l = shuffle ? shuffled4(t - t0) : t - t0;
            std::array<Row6, 4> a{};
            for (std::size_t i = 0; i < 4; ++i) {
                for (std::size_t j = 0; j < 6; ++j) {
                    float sum = 0.0f;
                    for (std::size_t x = 0; x < 6; ++x) {
                        sum += kAT4[i][x] * m[static_cast<std::int64_t>(x * 6 + j) * stride + l];
                    }
                    a[i][j] = sum;
                }
            }
            const std::int64_t yy = 4 * (t / pl.tw);
            const std::int64_t xx = 4 * (t % pl.tw);
            for (std::int64_t i = 0; i < 4 && yy + i < c.oh; ++i) {
                const Elements<float> row = output(yy + i, 0);
                for (std::int64_t jj = 0; jj < 4 && xx + jj < c.ow; ++jj) {
                    float y = 0.0f;
                    for (std::int64_t j = 0; j < 6; ++j) {
                        y += a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * at4(jj, j);
                    }
                    row[xx + jj] = add ? row[xx + jj] + y : y;
                }
            }
            ++t;
        }
        if (chain.count > 0) {
            for (std::int64_t t = t0; t < t1;) {
                const std::int64_t ty = t / pl.tw;
                const std::int64_t txa = t % pl.tw;
                const std::int64_t txb = std::min(pl.tw, txa + (t1 - t));
                const std::int64_t x0 = 4 * txa;
                const std::int64_t x1 = std::min(c.ow, 4 * txb);
                for (std::int64_t yy = 4 * ty; yy < std::min(c.oh, 4 * ty + 4); ++yy) {
                    chain.apply_positions(output(yy, x0).first(x1 - x0), yy * c.ow + x0, r, n);
                }
                t += txb - txa;
            }
        }
    }
}

void compute_large4(const ggml_compute_params * const params, const Conv & c, const Plan4 & pl,
                    const Epilogue & chain) {
    const Layout4 l = layout4(c, pl);
    Work_buffer buffer(params, l.layout);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const std::int64_t per_xi = xi_stride(round_up(c.oc, kMr), c.ic);
    // The kernels written by hand for AVX2 shuffle the tiles of each eight;
    // those for AVX-512, like the C++ ones, keep them in their order.
    const bool shuffle = avx2_kernels() && !avx512_kernels();

    Work_queue queue(params, buffer);
    const Elements<float> weights = buffer[l.weights];
    const Elements<float> panels = buffer[l.panels];
    const Elements<float> products = buffer[l.products];
    // The weights, a panel of kMr output channels to an item.
    const std::int64_t pack_items = ceil_div(c.oc, kMr);
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const Phase_scope scope(Phase::pack_weights);
        transform_weights4(weights.from(item * c.ic * kMr), per_xi, c, item * kMr, std::min<std::int64_t>(kMr, c.oc - item * kMr),
                           0, c.ic, panels);
    }
    queue.barrier();

    const std::int64_t tiles = pl.tw * pl.th;
    const std::vector<Block> blocks = guided_blocks(c.n, tiles, kNr, pl.tiles, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.mblocks;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const std::int64_t ib = u % pl.mblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.mblocks)];
        const std::int64_t n = block.segment;
        const std::int64_t i0 = ib * pl.mc;
        const std::int64_t i1 = std::min(c.oc, i0 + pl.mc);
        const std::int64_t t0 = block.start;
        const std::int64_t t1 = t0 + block.size;
        const std::int64_t nt = round_up(t1 - t0, kNr);
        const std::int64_t pstride = products_stride(pl.mc, nt);
        for (int r = 0; r < 6; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs4(panels, pl.stride, c, pl, n, t0, t1, k0, k0 + kc, r, shuffle);
                }
                const Phase_scope scope(Phase::product);
                for (std::int64_t s4 = 0; s4 < 6; ++s4) {
                    const std::int64_t xi = 6 * r + s4;
                    const Elements<float> product = products.from(xi * pstride);
                    const Elements<const float> u_xi = weights.from(xi * per_xi);
                    for (std::int64_t jj = 0; jj < nt; jj += step) {
                        const Elements<const float> b = panels.from(s4 * pl.stride + (jj / kNr) * kc * kNr);
                        for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                            sdasm::tiles(uk, wide, kc, u_xi.from((ii / kMr * c.ic + k0) * kMr), b, kc * kNr,
                                         product.from((ii - i0) * nt + jj), nt, k0 > 0, std::min<std::int64_t>(kMr, i1 - ii),
                                         std::min(step, nt - jj));
                        }
                    }
                }
            }
        }
        transform_outputs4(c, pl, products, pstride, nt, n, t0, t1, i0, i1, chain, shuffle);
    }
}

// F(4x4, 3x3) of images of few tiles, as compute_small() takes F(2x2, 3x3)'s:
// the inputs of every tile of every image transformed once, by all threads
// together, and each block of output channels transforming its weights a
// stretch of the input channels at a time into panels that stay in its cache.
// Transformed, the weights of F(4x4, 3x3) are eight times the bytes of FP16
// ones, which written out and read back would take longer than the products.
bool small4(const Conv & c) {
    return c.n * ((c.ow + 3) / 4) * ((c.oh + 3) / 4) <= tuning().winograd_small_tiles;
}

struct Plan4_small {
    std::int64_t tw, th, panels, mc, kc;
};

Plan4_small plan4_small(const Conv & c) {
    const Tuning & t = tuning();
    Plan4_small pl{};
    pl.tw = (c.ow + 3) / 4;
    pl.th = (c.oh + 3) / 4;
    pl.panels = ceil_div(c.n * pl.tw * pl.th, kNr);
    pl.kc = std::min<std::int64_t>(c.ic, std::max(kWidened, static_cast<std::int64_t>(t.winograd_small_kc)));
    pl.mc = round_up(std::min<std::int64_t>(c.oc, std::max(kMr, t.winograd_small_mc)), kMr);
    return pl;
}

// The transformed inputs of every tile, shared; each thread's widened weights,
// transformed weights, a channel of an image with its padding and products.
struct Layout4_small {
    Work_layout layout;
    Shared<float> inputs;
    Own<float> widened, weights, scratch, products;
};

Layout4_small layout4_small(const Conv & c, const Plan4_small & pl) {
    Layout4_small l;
    l.inputs = l.layout.shared<float>(kXi4 * inputs_stride(pl.panels, c.ic));
    l.widened = l.layout.own<float>(kWidenedFloats);
    l.weights = l.layout.own<float>(kXi4 * xi_stride(pl.mc, pl.kc));
    l.scratch = l.layout.own<float>((4 * pl.th + 2) * (4 * pl.tw + 2));
    l.products = l.layout.own<float>(kXi4 * products_stride(pl.mc, pl.panels * kNr));
    return l;
}

// The transformed inputs of channels [k0, k1) of image n, every tile, element
// xi of the tile that is g-th of all images at v[xi * inputs_stride() + ((g /
// kNr) * ic + k) * kNr + g % kNr]; the lanes past the last tile of the last
// image, zeros. `scratch` holds a channel of the image with the zeros around it.
void transform_inputs_small4(Elements<float> v, const Conv & c, const Plan4_small & pl, std::int64_t n, std::int64_t k0,
                             std::int64_t k1, Elements<float> scratch, bool shuffle) {
    const std::int64_t sw = 4 * pl.tw + 2;
    const std::int64_t sh = 4 * pl.th + 2;
    const std::int64_t tiles = pl.tw * pl.th;
    const std::int64_t last = c.n * tiles;
    const std::int64_t vstride = inputs_stride(pl.panels, c.ic);
    const auto at = [&](std::int64_t xi, std::int64_t g, std::int64_t k) -> float & {
        return v[xi * vstride + ((g / kNr) * c.ic + k) * kNr + g % kNr];
    };
    const auto zeros_past_last = [&](std::int64_t k) {
        if (n == c.n - 1) {
            for (std::int64_t g = last; g < pl.panels * kNr; ++g) {
                for (std::int64_t xi = 0; xi < kXi4; ++xi) {
                    at(xi, g, k) = 0.0f;
                }
            }
        }
    };
#if SDASM_X86_64
    if (shuffle && pl.tw % 8 == 0) {
        // Every row of tiles in eights, each eight in a lane of eight of its
        // panel, by the kernel for AVX2, a row of B^T at a time over the
        // channels.
        for (std::int64_t t = 0; t < tiles; t += 8) {
            const std::int64_t g = n * tiles + t;
            const std::int64_t y0 = 4 * (t / pl.tw) - c.p1;
            const std::int64_t x0 = 4 * (t % pl.tw) - c.p0;
            alignas(32) std::array<std::int32_t, 5 * 8> masks{};
            const bool inside = x0 >= 0 && x0 + 34 <= c.iw;
            for (std::size_t m = 0; m < 5; ++m) {
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const auto o = static_cast<std::int64_t>(8 * m + lane);
                    const std::int64_t x = x0 + o;
                    masks[8 * m + lane] = x >= 0 && x < c.iw && o < 34 ? -1 : 0;
                }
            }
            for (std::int64_t r = 0; r < 6; ++r) {
                Winograd4_inputs job{};
                const Elements<float> dst = v.from(6 * r * vstride + ((g / kNr) * c.ic + k0) * kNr + g % kNr);
                SDASM_CHECK(dst.size() >= 5 * vstride + (k1 - k0) * kNr);
                job.dst = dst.data();
                job.stride = bytes_of<float>(vstride);
                rows4(job, c, n, k0, k1 - k0, y0, x0, r);
                job.masks = inside ? nullptr : masks.data();
                sdasm_winograd4_inputs_avx2(&job);
            }
        }
        for (std::int64_t k = k0; k < k1; ++k) {
            zeros_past_last(k);
        }
        return;
    }
#else
    (void) shuffle;
#endif
    for (std::int64_t k = k0; k < k1; ++k) {
        const Elements<const float> plane = c.x.from(input_plane(c, n, k));
        for (std::int64_t y = 0; y < sh; ++y) {
            const Elements<float> row = scratch.part(y * sw, sw);
            const std::int64_t iy = y - c.p1;
            for (std::int64_t x = 0; x < sw; ++x) {
                row[x] = input_at(c, plane, iy, x - c.p0);
            }
        }
        for (std::int64_t t = 0; t < tiles; ++t) {
            const Elements<const float> d = scratch.from(4 * (t / pl.tw) * sw + 4 * (t % pl.tw));
            const std::int64_t g = n * tiles + t;
            for (std::int64_t r = 0; r < 6; ++r) {
                std::array<float, 6> u{};
                for (std::int64_t i = 0; i < 6; ++i) {
                    const float b = bt4(r, i);
                    for (std::size_t j = 0; j < 6; ++j) {
                        u[j] += b * d[i * sw + static_cast<std::int64_t>(j)];
                    }
                }
                for (std::int64_t s4 = 0; s4 < 6; ++s4) {
                    float x = 0.0f;
                    for (std::size_t j = 0; j < 6; ++j) {
                        x += bt4(s4, static_cast<std::int64_t>(j)) * u[j];
                    }
                    at(r * 6 + s4, g, k) = x;
                }
            }
        }
        zeros_past_last(k);
    }
}

void compute_small4(const ggml_compute_params * const params, const Conv & c, const Plan4_small & pl,
                    const Epilogue & chain) {
    const Layout4_small l = layout4_small(c, pl);
    Work_buffer buffer(params, l.layout);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t tiles = pl.tw * pl.th;
    const std::int64_t nt = pl.panels * kNr;
    const std::int64_t vstride = inputs_stride(pl.panels, c.ic);
    const std::int64_t pstride = products_stride(pl.mc, nt);
    const Plan4 layout = {pl.tw, pl.th, 0, 0, 0, 0, 0, 0};
    // The kernels for AVX2 take rows of tiles of eights, in their shuffled
    // order, on any processor with AVX2: at 32 by 32, rows are of eight tiles.
    const bool shuffle = avx2_kernels() && pl.tw % 8 == 0;

    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params, buffer);
    const Elements<float> inputs = buffer[l.inputs];
    const Elements<float> widened = buffer[l.widened];
    const Elements<float> weights = buffer[l.weights];
    const Elements<float> scratch = buffer[l.scratch];
    const Elements<float> products = buffer[l.products];
    const std::int64_t groups = ceil_div(c.ic, kInputChannels);
    const std::int64_t input_items = c.n * groups;
    int item = queue.next();
    for (; item < input_items; item = queue.next()) {
        const std::int64_t n = item / groups;
        const std::int64_t k0 = (item % groups) * kInputChannels;
        const Phase_scope scope(Phase::pack_inputs);
        transform_inputs_small4(inputs, c, pl, n, k0, std::min(c.ic, k0 + kInputChannels), scratch, shuffle);
    }
    queue.barrier();

    const std::vector<Block> blocks = guided_blocks(1, c.oc, kMr, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    for (; item < input_items + units; item = queue.next()) {
        const Block & block = blocks[static_cast<std::size_t>(item - input_items)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.ic - k0);
            const std::int64_t stride = xi_stride(pl.mc, kc);
            {
                const Phase_scope scope(Phase::pack_weights);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    transform_weights4(weights.from((ii - i0) * kc), stride, c, ii, std::min<std::int64_t>(kMr, i1 - ii), k0,
                                       k0 + kc, widened);
                }
            }
            const Phase_scope scope(Phase::product);
            // The weights of the next stretch of input channels fetched into
            // the cache while the products of this one are computed, as
            // compute_small() fetches them: a 36th of the rows at each xi.
            const std::int64_t next0 = k0 + kc;
            const std::int64_t next1 = tuning().prefetch != 0 ? std::min(c.ic, next0 + pl.kc) : next0;
            const std::int64_t rows_per_xi = ceil_div(i1 - i0, kXi4);
            for (std::int64_t xi = 0; xi < kXi4; ++xi) {
                prefetch_weights(c, i0 + xi * rows_per_xi, std::min(i1, i0 + (xi + 1) * rows_per_xi), next0, next1);
                const Elements<const float> u = weights.from(xi * stride);
                const Elements<float> product = products.from(xi * pstride);
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    for (std::int64_t p = 0; p < pl.panels; p += panel_step(wide) / kNr) {
                        sdasm::tiles(uk, wide, kc, u.from((ii - i0) * kc), inputs.from(xi * vstride + (p * c.ic + k0) * kNr),
                                     c.ic * kNr, product.from((ii - i0) * nt + p * kNr), nt, k0 > 0,
                                     std::min<std::int64_t>(kMr, i1 - ii), std::min(panel_step(wide), nt - p * kNr));
                    }
                }
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs4(c, layout, products.from(n * tiles), pstride, nt, n, 0, tiles, i0, i1, chain, shuffle);
        }
    }
}

// ---------------------------------------------------------------- quantized

// F(2x2, 3x3) with its products in integers (quant.h, Tuning::bits): the
// transforms as above, in single precision, and each xi's product of the
// transformed weights and inputs on the quantized micro-kernel -- the weights
// signed, each output channel's in groups of the input channels, the inputs
// unsigned with a zero point, each tile's likewise, a scale for each xi of
// each: the sixteen elements of a transformed tile are of ranges that one
// scale would waste the bits of, as Andri et al. (2022) found of the taps of
// F(4x4, 3x3). Measured on SD 1.5, its outputs are within 7 times 10^-3 of
// the exact ones in the L2 norm, F(4x4, 3x3)'s in integers 1.7 times 10^-2,
// whose transforms make more of the rounding of the products.
struct Qplan {
    Quant q;
    std::int64_t kc;       // a stretch of the input channels, a multiple of the group
    std::int64_t opanels;  // panels of kMr output channels
    std::int64_t u_panel;  // a quantized panel of transformed weights over every input channel
    std::int64_t v_panel;  // a quantized panel of kNr tiles' transformed inputs over a stretch (large)
                           // or over every input channel (small)
    std::int64_t fstride;  // large images: the floats between the panels of the inputs of two xi
    Plan base;             // the blocks of the products, as in single precision
};

std::uint64_t qvariant(const Quant & q) {
    return static_cast<std::uint64_t>(q.quads) | static_cast<std::uint64_t>(q.weight_max) << 16 |
           static_cast<std::uint64_t>(q.weight_groups ? 1 : 0) << 32;
}

Qplan qplan(const Conv & c, int n_threads) {
    Qplan pl{};
    pl.q = quant();
    pl.base = plan(c, n_threads);
    const std::int64_t group = 4 * pl.q.quads;
    pl.kc = std::max(group, pl.base.kc / group * group);
    pl.opanels = ceil_div(c.oc, kMr);
    pl.u_panel = qpanel_bytes(c.ic, kMr, pl.q);
    pl.v_panel = pl.base.small ? qpanel_bytes(c.ic, kNr, pl.q) : qpanel_bytes(pl.kc, kNr, pl.q);
    pl.fstride = panels_stride(pl.base.kc, pl.base.tiles);
    return pl;
}

// The quantized weights of every xi and output channel, by all threads
// together, or the store's; each thread transforms a panel of them a stretch of
// input channels at a time. Small images: the inputs of every tile, in single
// precision and quantized, by all threads. Large: each thread's inputs of a
// block of tiles, in single precision and quantized.
struct Quantized_layout {
    Work_layout layout;
    Shared<std::uint8_t> weights;
    Shared<float> inputs;            // small images
    Shared<std::uint8_t> qinputs;    // small images
    Own<float> widened, floats;
    Own<std::uint8_t> quantized;     // large images
    Own<float> scratch;              // small images
    Own<float> products;
};

Quantized_layout quantized_layout(const Conv & c, const Qplan & pl) {
    Quantized_layout l;
    const Plan & bp = pl.base;
    l.weights = l.layout.shared<std::uint8_t>(kXi * pl.opanels * pl.u_panel);
    const std::int64_t transformed = kXi * xi_stride(kMr, pl.kc);
    l.widened = l.layout.own<float>(kWidenedFloats);
    if (bp.small) {
        l.inputs = l.layout.shared<float>(kXi * inputs_stride(bp.panels, c.ic));
        l.qinputs = l.layout.shared<std::uint8_t>(kXi * bp.panels * pl.v_panel);
        l.floats = l.layout.own<float>(transformed);
        l.scratch = l.layout.own<float>((2 * c.th + 2) * (2 * c.tw + 2));
        l.products = l.layout.own<float>(kXi * products_stride(bp.mc, bp.panels * kNr));
        return l;
    }
    l.floats = l.layout.own<float>(std::max(panels_count(bp.kc, bp.tiles), transformed));
    l.quantized = l.layout.own<std::uint8_t>(4 * (bp.tiles / kNr) * pl.v_panel);
    l.products = l.layout.own<float>(kXi * products_stride(bp.mc, bp.tiles));
    return l;
}

// The quantized transformed weights, the store's or the work buffer's, packed
// by the queue's first items, from `item`, when they are not ready -- a panel
// of kMr output channels to an item -- and the threads met after them: the
// weights, and the number of those items, which the items after count from.
struct Quantized_weights {
    Elements<const std::uint8_t> data;
    std::int64_t items;
};

Quantized_weights quantized_weights(const ggml_compute_params * const params, const Conv & c, const Qplan & pl,
                                    const std::shared_ptr<Packed> & kept, Elements<std::uint8_t> work,
                                    Elements<float> floats, Elements<float> widened, Work_queue & queue, int & item) {
    const Elements<std::uint8_t> weights = kept != nullptr ? kept->octets() : work;
    if (kept != nullptr && kept->ready()) {
        return {weights, 0};
    }
    const std::int64_t group = 4 * pl.q.quads;
    const std::int64_t a_group = qgroup_bytes(pl.q.quads, kMr);
    const std::int64_t stride = xi_stride(kMr, pl.kc);
    for (; item < pl.opanels; item = queue.next()) {
        const Phase_scope scope(Phase::pack_weights);
        for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.ic - k0);
            transform_weights(floats, stride, c, item * kMr, std::min<std::int64_t>(kMr, c.oc - item * kMr), k0, k0 + kc,
                              widened);
            for (std::int64_t xi = 0; xi < kXi; ++xi) {
                quantize_panel_weights(weights.from((xi * pl.opanels + item) * pl.u_panel + k0 / group * a_group),
                                       floats.part(xi * stride, kc * kMr), kc, kMr, Qlayout::rows, pl.q);
            }
        }
    }
    queue.barrier();
    if (kept != nullptr && params->ith == 0) {
        publish(kept);
    }
    return {weights, pl.opanels};
}

std::shared_ptr<Packed> quantized_kept(const Conv & c, const Qplan & pl) {
    return packed(c.kernel, Layout::quantized_winograd, kXi * pl.opanels * pl.u_panel, qvariant(pl.q));
}

void compute_large_quantized(const ggml_compute_params * const params, const Conv & c, const Qplan & pl,
                             const Epilogue & chain) {
    const Quant & q = pl.q;
    const std::int64_t group = 4 * q.quads;
    const Quantized_layout l = quantized_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const std::shared_ptr<Packed> kept = quantized_kept(c, pl);
    const Qkernel kernel = qkernel_b_unsigned();
    const Qkernel_wide wide = qkernel_wide_b_unsigned();
    const std::int64_t step = wide != nullptr ? 2 * kNr : kNr;
    const bool shuffle = avx2_kernels() && !avx512_kernels();
    const std::int64_t a_group = qgroup_bytes(q.quads, kMr);

    Work_queue queue(params, buffer);
    const Elements<float> widened = buffer[l.widened];
    const Elements<float> floats = buffer[l.floats];
    const Elements<std::uint8_t> inputs = buffer[l.quantized];
    const Elements<float> products = buffer[l.products];
    int item = queue.next();
    const Quantized_weights packed_weights =
        quantized_weights(params, c, pl, kept, buffer[l.weights], floats, widened, queue, item);
    const Elements<const std::uint8_t> weights = packed_weights.data;
    const std::int64_t pack_items = packed_weights.items;

    const Plan & bp = pl.base;
    const std::int64_t tiles = c.tw * c.th;
    const std::vector<Block> blocks = guided_blocks(c.n, tiles, kNr, bp.tiles, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * bp.mblocks;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const std::int64_t ib = u % bp.mblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / bp.mblocks)];
        const std::int64_t n = block.segment;
        const std::int64_t i0 = ib * bp.mc;
        const std::int64_t i1 = std::min(c.oc, i0 + bp.mc);
        const std::int64_t t0 = block.start;
        const std::int64_t t1 = t0 + block.size;
        const std::int64_t nt = round_up(t1 - t0, kNr);
        const std::int64_t ntp = nt / kNr;
        const std::int64_t pstride = products_stride(bp.mc, nt);
        for (int r = 0; r < 4; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs(floats, pl.fstride, c, n, t0, t1, k0, k0 + kc, r, shuffle);
                    for (std::int64_t s = 0; s < 4; ++s) {
                        for (std::int64_t p = 0; p < ntp; ++p) {
                            quantize_panel_inputs(inputs.part((s * ntp + p) * pl.v_panel, pl.v_panel),
                                                  floats.part(s * pl.fstride + p * kc * kNr, kc * kNr), kc, kNr,
                                                  Qlayout::columns, q);
                        }
                    }
                }
                const Phase_scope scope(Phase::product);
                const std::int64_t groups = ceil_div(kc, group);
                for (std::int64_t s = 0; s < 4; ++s) {
                    const std::int64_t xi = 4 * r + s;
                    const Elements<float> product = products.from(xi * pstride);
                    for (std::int64_t jj = 0; jj < nt; jj += step) {
                        const Elements<const std::uint8_t> b = inputs.from((s * ntp + jj / kNr) * pl.v_panel);
                        for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                            const Elements<const std::uint8_t> a =
                                weights.from((xi * pl.opanels + ii / kMr) * pl.u_panel + k0 / group * a_group);
                            qtiles(kernel, wide, groups, q.quads, a, b, pl.v_panel, product.from((ii - i0) * nt + jj), nt,
                                   k0 > 0, std::min<std::int64_t>(kMr, i1 - ii), std::min(step, nt - jj));
                        }
                    }
                }
            }
        }
        transform_outputs(c, products, pstride, nt, n, t0, t1, i0, i1, chain, shuffle);
    }
}

void compute_small_quantized(const ggml_compute_params * const params, const Conv & c, const Qplan & pl,
                             const Epilogue & chain) {
    const Quant & q = pl.q;
    const Plan & bp = pl.base;
    const std::int64_t group = 4 * q.quads;
    const Quantized_layout l = quantized_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    const std::shared_ptr<Packed> kept = quantized_kept(c, pl);
    const Qkernel kernel = qkernel_b_unsigned();
    const Qkernel_wide wide = qkernel_wide_b_unsigned();
    const std::int64_t step = wide != nullptr ? 2 * kNr : kNr;
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t nt = bp.panels * kNr;
    const std::int64_t b_group = qgroup_bytes(q.quads, kNr);
    const std::int64_t pstride = products_stride(bp.mc, nt);

    // The weights, when they are not ready; the inputs, by all threads
    // together, sixteen channels of an image to an item; then quantized, a
    // panel of an xi to an item.
    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params, buffer);
    const Elements<float> inputs = buffer[l.inputs];
    const Elements<std::uint8_t> qinputs = buffer[l.qinputs];
    const Elements<float> widened = buffer[l.widened];
    const Elements<float> floats = buffer[l.floats];
    const Elements<float> scratch = buffer[l.scratch];
    const Elements<float> products = buffer[l.products];
    int item = queue.next();
    const Quantized_weights packed_weights =
        quantized_weights(params, c, pl, kept, buffer[l.weights], floats, widened, queue, item);
    const Elements<const std::uint8_t> weights = packed_weights.data;
    const std::int64_t channel_groups = ceil_div(c.ic, kInputChannels);
    const std::int64_t input_items = packed_weights.items + c.n * channel_groups;
    for (; item < input_items; item = queue.next()) {
        const std::int64_t i = item - packed_weights.items;
        const std::int64_t n = i / channel_groups;
        const std::int64_t k0 = (i % channel_groups) * kInputChannels;
        const Phase_scope scope(Phase::pack_inputs);
        transform_inputs_small(inputs, c, bp.panels, n, k0, std::min(c.ic, k0 + kInputChannels), scratch);
    }
    queue.barrier();
    const std::int64_t quantize_items = input_items + kXi * bp.panels;
    for (; item < quantize_items; item = queue.next()) {
        const std::int64_t i = item - input_items;
        const std::int64_t xi = i / bp.panels;
        const std::int64_t p = i % bp.panels;
        const Phase_scope scope(Phase::pack_inputs);
        quantize_panel_inputs(qinputs.part(i * pl.v_panel, pl.v_panel),
                              inputs.part(xi * inputs_stride(bp.panels, c.ic) + p * c.ic * kNr, c.ic * kNr), c.ic, kNr,
                              Qlayout::columns, q);
    }
    queue.barrier();

    const std::vector<Block> blocks = guided_blocks(1, c.oc, kMr, bp.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    const std::int64_t a_group = qgroup_bytes(q.quads, kMr);
    for (; item < quantize_items + units; item = queue.next()) {
        const Block & block = blocks[static_cast<std::size_t>(item - quantize_items)];
        const std::int64_t i0 = block.start;
        const std::int64_t i1 = i0 + block.size;
        const std::int64_t panels = ceil_div(i1 - i0, kMr);
        for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
            const std::int64_t kc = std::min(pl.kc, c.ic - k0);
            const Phase_scope scope(Phase::product);
            const std::int64_t groups = ceil_div(kc, group);
            for (std::int64_t xi = 0; xi < kXi; ++xi) {
                const Elements<float> product = products.from(xi * pstride);
                for (std::int64_t ip = 0; ip < panels; ++ip) {
                    const Elements<const std::uint8_t> a =
                        weights.from((xi * pl.opanels + i0 / kMr + ip) * pl.u_panel + k0 / group * a_group);
                    for (std::int64_t jj = 0; jj < nt; jj += step) {
                        const Elements<const std::uint8_t> b =
                            qinputs.from((xi * bp.panels + jj / kNr) * pl.v_panel + k0 / group * b_group);
                        qtiles(kernel, wide, groups, q.quads, a, b, pl.v_panel, product.from(ip * kMr * nt + jj), nt, k0 > 0,
                               std::min<std::int64_t>(kMr, i1 - i0 - ip * kMr), std::min(step, nt - jj));
                    }
                }
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs(c, products.from(n * tiles), pstride, nt, n, 0, tiles, i0, i1, chain, false);
        }
    }
}

}  // namespace

bool winograd_supported(const ggml_tensor * const node) {
    if (node->op != GGML_OP_CONV_2D || tuning().winograd == 0) {
        return false;
    }
    const ggml_tensor * const kernel = node->src[0];
    const ggml_tensor * const input = node->src[1];
    const std::span<const std::int32_t> p(node->op_params);
    const bool weights = (kernel->type == GGML_TYPE_F16 && holds<std::uint16_t>(kernel)) ||
                         (kernel->type == GGML_TYPE_F32 && holds<float>(kernel));
    if (!(kernel->ne[0] == 3 && kernel->ne[1] == 3 && p[0] == 1 && p[1] == 1 && p[4] == 1 && p[5] == 1 && p[2] >= 0 &&
          p[2] <= 2 && p[3] >= 0 && p[3] <= 2 && weights && ggml_is_contiguous(kernel) && holds<float>(input) &&
          input->nb[0] == sizeof(float) && holds<float>(node) && ggml_is_contiguous(node) &&
          kernel->ne[2] == input->ne[2])) {
        return false;
    }
    return node->ne[0] * node->ne[1] >= tuning().winograd_positions || small(geometry(node));
}

std::size_t winograd_work_size(const ggml_tensor * const node, int n_threads) {
    const Conv c = geometry(node);
    if (quantized(Quantized_op::convolution)) {
        return quantized_layout(c, qplan(c, n_threads)).layout.bytes(n_threads);
    }
    if (four(c)) {
        if (small4(c)) {
            return layout4_small(c, plan4_small(c)).layout.bytes(n_threads);
        }
        return layout4(c, plan4(c, n_threads)).layout.bytes(n_threads);
    }
    if (on_amx(c)) {
        return amx_layout(c, amx_plan(c, n_threads)).layout.bytes(n_threads);
    }
    return single_layout(c, plan(c, n_threads)).layout.bytes(n_threads);
}

namespace {

void compute_large(const ggml_compute_params * const params, const Conv & c, const Plan & pl, const Epilogue & chain) {
    const Single_layout l = single_layout(c, pl);
    Work_buffer buffer(params, l.layout);
    // The transformed weights: the store's, transformed here the first time,
    // or the work buffer's, transformed here every time.
    const std::shared_ptr<Packed> kept = packed(c.kernel, Layout::winograd, weights_count(c));
    const bool fill = kept == nullptr || !kept->ready();
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const std::int64_t per_xi = xi_stride(round_up(c.oc, kMr), c.ic);
    // The kernels written by hand for AVX2 pack the tiles of each eight in
    // shuffled lanes, and transform the outputs from them; those for AVX-512,
    // like the C++ ones, keep the tiles in their order.
    const bool shuffle = avx2_kernels() && !avx512_kernels();

    Work_queue queue(params, buffer);
    const Elements<float> weights = kept != nullptr ? kept->singles() : buffer[l.shared];
    const Elements<float> panels = buffer[l.panels];
    const Elements<float> products = buffer[l.products];
    // The weights, two panels of kMr output channels to an item, widened in
    // the thread's panels of inputs, which it has not begun to fill.
    constexpr std::int64_t kPanels = 2;
    const std::int64_t pack_items = fill ? ceil_div(ceil_div(c.oc, kMr), kPanels) : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            for (std::int64_t panel = item * kPanels; panel < std::min((item + 1) * kPanels, ceil_div(c.oc, kMr)); ++panel) {
                transform_weights(weights.from(panel * c.ic * kMr), per_xi, c, panel * kMr,
                                  std::min<std::int64_t>(kMr, c.oc - panel * kMr), 0, c.ic, panels);
            }
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }

    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t stride = panels_stride(pl.kc, pl.tiles);
    const std::vector<Block> blocks = guided_blocks(c.n, tiles, kNr, pl.tiles, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size()) * pl.mblocks;
    for (; item < pack_items + units; item = queue.next()) {
        const std::int64_t u = item - pack_items;
        const std::int64_t ib = u % pl.mblocks;
        const Block & block = blocks[static_cast<std::size_t>(u / pl.mblocks)];
        const std::int64_t n = block.segment;
        const std::int64_t i0 = ib * pl.mc;
        const std::int64_t i1 = std::min(c.oc, i0 + pl.mc);
        const std::int64_t t0 = block.start;
        const std::int64_t t1 = t0 + block.size;
        const std::int64_t nt = round_up(t1 - t0, kNr);
        const std::int64_t pstride = products_stride(pl.mc, nt);
        // The sixteen products, [xi][channel][tile]: the four of a row r of
        // B^T a stretch of input channels at a time, from the panels of their
        // inputs, packed together.
        for (int r = 0; r < 4; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs(panels, stride, c, n, t0, t1, k0, k0 + kc, r, shuffle);
                }
                const Phase_scope scope(Phase::product);
                for (std::int64_t s = 0; s < 4; ++s) {
                    const std::int64_t xi = 4 * r + s;
                    const Elements<float> product = products.from(xi * pstride);
                    const Elements<const float> u_xi = weights.from(xi * per_xi);
                    for (std::int64_t jj = 0; jj < nt; jj += step) {
                        const Elements<const float> b = panels.from(s * stride + (jj / kNr) * kc * kNr);
                        for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                            sdasm::tiles(uk, wide, kc, u_xi.from((ii / kMr * c.ic + k0) * kMr), b, kc * kNr,
                                         product.from((ii - i0) * nt + jj), nt, k0 > 0, std::min<std::int64_t>(kMr, i1 - ii),
                                         std::min(step, nt - jj));
                        }
                    }
                }
            }
        }
        transform_outputs(c, products, pstride, nt, n, t0, t1, i0, i1, chain, shuffle);
    }
}

}  // namespace

void winograd_compute(const ggml_compute_params * const params, const ggml_tensor * const node,
                      const Epilogue & chain) {
    Conv c = geometry(node);
    if (chain.redirect != nullptr) {
        // The residual connection's tensor, of the same layout (fusion.h).
        c.y = elements_of<float>(chain.redirect);
    }
    if (quantized(Quantized_op::convolution)) {
        const Qplan pl = qplan(c, params->nth);
        if (pl.base.small) {
            compute_small_quantized(params, c, pl, chain);
        }
        else {
            compute_large_quantized(params, c, pl, chain);
        }
        return;
    }
    if (four(c)) {
        if (small4(c)) {
            compute_small4(params, c, plan4_small(c), chain);
        }
        else {
            compute_large4(params, c, plan4(c, params->nth), chain);
        }
        return;
    }
    if (on_amx(c)) {
        const Amx_plan pl = amx_plan(c, params->nth);
        if (small(c)) {
            compute_small_amx(params, c, pl, chain);
        }
        else {
            compute_large_amx(params, c, pl, chain);
        }
        return;
    }
    const Plan pl = plan(c, params->nth);
    if (pl.small) {
        compute_small(params, c, pl, chain);
        return;
    }
    compute_large(params, c, pl, chain);
}

}  // namespace sdasm
