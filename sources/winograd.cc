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
#include <bit>
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

constexpr int kXi = 16;  // the elements of a tile of four by four

struct Conv {
    const ggml_tensor * kernel;
    const char * x;  // [IW, IH, IC, N]
    std::int64_t iw, ih, ic, n;
    std::int64_t xnb1, xnb2, xnb3;
    const char * w;  // [3, 3, IC, OC]
    ggml_type wtype;
    std::int64_t oc;
    std::int64_t wnb2, wnb3;
    char * y;  // [OW, OH, OC, N]
    std::int64_t ow, oh;
    std::int64_t ynb1, ynb2, ynb3;
    int p0, p1;
    std::int64_t tw, th;  // tiles across and down
};

Conv geometry(const ggml_tensor * node) {
    const ggml_tensor * kernel = node->src[0];
    const ggml_tensor * input = node->src[1];
    Conv c{};
    c.kernel = kernel;
    c.x = static_cast<const char *>(input->data);
    c.iw = input->ne[0];
    c.ih = input->ne[1];
    c.ic = input->ne[2];
    c.n = input->ne[3];
    c.xnb1 = static_cast<std::int64_t>(input->nb[1]);
    c.xnb2 = static_cast<std::int64_t>(input->nb[2]);
    c.xnb3 = static_cast<std::int64_t>(input->nb[3]);
    c.w = static_cast<const char *>(kernel->data);
    c.wtype = kernel->type;
    c.oc = kernel->ne[3];
    c.wnb2 = static_cast<std::int64_t>(kernel->nb[2]);
    c.wnb3 = static_cast<std::int64_t>(kernel->nb[3]);
    c.y = static_cast<char *>(node->data);
    c.ow = node->ne[0];
    c.oh = node->ne[1];
    c.ynb1 = static_cast<std::int64_t>(node->nb[1]);
    c.ynb2 = static_cast<std::int64_t>(node->nb[2]);
    c.ynb3 = static_cast<std::int64_t>(node->nb[3]);
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

// The large-image path keeps the sixteen tiles of a panel of B in the order
// its packing finds them in, which is not quite their own: within each eight,
// tiles 0, 1, 4, 5 and then 2, 3, 6, 7 (pack_inputs()). The lane of tile l of
// a block, l counted from the block's first, is shuffled(l); the order is its
// own inverse.
constexpr std::int64_t kShuffle[8] = {0, 1, 4, 5, 2, 3, 6, 7};

std::int64_t shuffled(std::int64_t l) {
    return (l & ~std::int64_t{7}) | kShuffle[l & 7];
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
    std::size_t shared_bytes;
    std::size_t panel_bytes;    // a thread's packed inputs; small images: its widened weights
    std::size_t weight_bytes;   // small images: a thread's transformed weights
    std::size_t scratch_bytes;  // small images: a thread's channel of an image, padded
    std::size_t product_bytes;  // a thread's sixteen products
    std::size_t thread_bytes;
};

// The widened weights of kWidened input channels of a panel of output
// channels, [channel][3 * 3][kMr], and the floats a full load past them reads.
constexpr std::int64_t kWidened = 8;

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
    pl.shared_bytes = align_up(static_cast<std::size_t>(kXi * inputs_stride(pl.panels, c.ic)) * sizeof(float));
    pl.panel_bytes = align_up(static_cast<std::size_t>(kWidened * 9 * kMr + 8) * sizeof(float));
    pl.weight_bytes = align_up(static_cast<std::size_t>(kXi * xi_stride(pl.mc, pl.kc)) * sizeof(float));
    pl.scratch_bytes = align_up(static_cast<std::size_t>((2 * c.th + 2) * (2 * c.tw + 2)) * sizeof(float));
    pl.product_bytes = align_up(static_cast<std::size_t>(kXi * products_stride(pl.mc, pl.panels * kNr)) * sizeof(float));
    pl.thread_bytes = pl.panel_bytes + pl.weight_bytes + pl.scratch_bytes + pl.product_bytes;
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
    pl.shared_bytes = align_up(static_cast<std::size_t>(kXi * xi_stride(round_up(c.oc, kMr), c.ic)) * sizeof(float));
    pl.panel_bytes = std::max(4 * static_cast<std::size_t>(skewed(pl.kc * pl.tiles)) * sizeof(float),
                              align_up(static_cast<std::size_t>(kWidened * 9 * kMr + 8) * sizeof(float)));
    pl.product_bytes = align_up(static_cast<std::size_t>(kXi * products_stride(pl.mc, pl.tiles)) * sizeof(float));
    pl.thread_bytes = pl.panel_bytes + pl.product_bytes;
    return pl;
}

// The four by four inputs of tile t of image n, channel ic, rows and columns,
// zeros outside the image.
inline float input_at(const Conv & c, const char * plane, std::int64_t y, std::int64_t x) {
    return (y >= 0 && y < c.ih && x >= 0 && x < c.iw) ? reinterpret_cast<const float *>(plane + y * c.xnb1)[x] : 0.0f;
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

// Packs the elements xi = (r, s), s = 0..3, of the transformed inputs of tiles
// [t0, t1) of image n and channels [k0, k1) into panels of kNr tiles,
// [panel][k1 - k0][kNr], tile l of the block in lane shuffled(l), zeros past
// t1: those of element (r, s) at dst + s * stride. The four take the same two
// rows of each tile, which row r of B^T adds or subtracts, and are packed
// together so that those rows are read and taken apart once for the four.
void pack_inputs(float * dst, std::int64_t stride, const Conv & c, std::int64_t n, std::int64_t t0, std::int64_t t1,
                 std::int64_t k0, std::int64_t k1, int r, bool shuffle) {
    const std::int64_t kc = k1 - k0;
    const char * image = c.x + n * c.xnb3;
    // The rows of the tile that row r of B^T takes, with their signs.
    constexpr int kRows[4][2] = {{0, 2}, {1, 2}, {1, 2}, {1, 3}};
    constexpr float kSigns[4][2] = {{1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}};
    for (std::int64_t tt = t0; tt < t1; tt += kNr, dst += kc * kNr) {
        const std::int64_t valid = std::min<std::int64_t>(kNr, t1 - tt);
        const std::int64_t ty = tt / c.tw;
        const std::int64_t tx0 = tt % c.tw;
        const std::int64_t y0 = 2 * ty - c.p1;
        const std::int64_t x0 = 2 * tx0 - c.p0;
#if SDASM_X86_64
        if (avx512_kernels() && valid == kNr && tx0 + kNr <= c.tw) {
            // Sixteen tiles of a row by the kernel for AVX-512, in their order:
            // the rows as below, and the lanes of its four loads of sixteen
            // that are inside the image.
            alignas(64) static constexpr float kZeros[2 * kNr + 16] = {};
            constexpr int kOffsets[4] = {0, 16, 2, 18};
            Winograd_inputs job{};
            for (int m = 0; m < 4; ++m) {
                std::uint16_t lanes = 0;
                for (int lane = 0; lane < 16; ++lane) {
                    const std::int64_t x = x0 + kOffsets[m] + lane;
                    lanes = static_cast<std::uint16_t>(lanes | ((x >= 0 && x < c.iw ? 1u : 0u) << lane));
                }
                job.lanes[m] = lanes;
            }
            const int qa = r == 2 ? 1 : 0;
            const auto row = [&](int q, const char *& p, std::int64_t & step) {
                const std::int64_t y = y0 + kRows[r][q];
                if (y < 0 || y >= c.ih) {
                    p = reinterpret_cast<const char *>(kZeros);
                    step = 0;
                    return;
                }
                p = reinterpret_cast<const char *>(reinterpret_cast<std::uintptr_t>(image + k0 * c.xnb2 + y * c.xnb1) +
                                                   static_cast<std::uintptr_t>(x0 * static_cast<std::int64_t>(sizeof(float))));
                step = c.xnb2;
            };
            job.dst = dst;
            job.stride = stride * static_cast<std::int64_t>(sizeof(float));
            row(qa, job.a, job.a_step);
            row(1 - qa, job.b, job.b_step);
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
            alignas(32) static constexpr float kZeros[2 * kNr + 8] = {};
            constexpr int kOffsets[8] = {0, 8, 2, 10, 16, 24, 18, 26};
            alignas(32) std::int32_t masks[8][8];
            const bool inside = x0 >= 0 && x0 + 2 * kNr + 2 <= c.iw;
            for (int m = 0; m < 8; ++m) {
                for (int lane = 0; lane < 8; ++lane) {
                    const std::int64_t x = x0 + kOffsets[m] + lane;
                    masks[m][lane] = x >= 0 && x < c.iw ? -1 : 0;
                }
            }
            const int qa = r == 2 ? 1 : 0;
            const auto row = [&](int q, const char *& p, std::int64_t & step) {
                const std::int64_t y = y0 + kRows[r][q];
                if (y < 0 || y >= c.ih) {
                    p = reinterpret_cast<const char *>(kZeros);
                    step = 0;
                    return;
                }
                // The address of the first input, which may lie before the row:
                // it is formed but never read.
                p = reinterpret_cast<const char *>(reinterpret_cast<std::uintptr_t>(image + k0 * c.xnb2 + y * c.xnb1) +
                                                   static_cast<std::uintptr_t>(x0 * static_cast<std::int64_t>(sizeof(float))));
                step = c.xnb2;
            };
            Winograd_inputs job{};
            job.dst = dst;
            job.stride = stride * static_cast<std::int64_t>(sizeof(float));
            row(qa, job.a, job.a_step);
            row(1 - qa, job.b, job.b_step);
            job.channels = kc;
            // A row of zeros is read where a row of the image would be, and as
            // far: kZeros is long enough for it, masks or none.
            job.masks = inside ? nullptr : &masks[0][0];
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
                const char * plane = image + k * c.xnb2;
                float sum[4][kNr] = {};
                for (int q = 0; q < 2; ++q) {
                    const std::int64_t y = y0 + kRows[r][q];
                    if (y < 0 || y >= c.ih) {
                        continue;
                    }
                    const float * p = reinterpret_cast<const float *>(plane + y * c.xnb1) + x0;
                    const float sign = kSigns[r][q];
                    for (int l = 0; l < kNr; ++l) {
                        const float d0 = p[2 * l];
                        const float d1 = p[2 * l + 1];
                        const float d2 = p[2 * l + 2];
                        const float d3 = p[2 * l + 3];
                        sum[0][l] += sign * (d0 - d2);
                        sum[1][l] += sign * (d1 + d2);
                        sum[2][l] += sign * (d2 - d1);
                        sum[3][l] += sign * (d1 - d3);
                    }
                }
                for (int s4 = 0; s4 < 4; ++s4) {
                    for (int l = 0; l < kNr; ++l) {
                        dst[s4 * stride + (k - k0) * kNr + l] = sum[s4][l];
                    }
                }
            }
            continue;
        }
        // Tiles at the edges of the image, or running over the end of a row of
        // tiles: one at a time.
        for (std::int64_t k = k0; k < k1; ++k) {
            const char * plane = image + k * c.xnb2;
            for (std::int64_t l = 0; l < kNr; ++l) {
                float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (l < valid) {
                    const std::int64_t t = tt + l;
                    const std::int64_t yy = 2 * (t / c.tw) - c.p1;
                    const std::int64_t xx = 2 * (t % c.tw) - c.p0;
                    float d[4][4];
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            d[i][j] = input_at(c, plane, yy + i, xx + j);
                        }
                    }
                    for (int s4 = 0; s4 < 4; ++s4) {
                        float row[4];
                        for (int i = 0; i < 4; ++i) {
                            row[i] = bt(s4, d[i][0], d[i][1], d[i][2], d[i][3]);
                        }
                        out[s4] = bt(r, row[0], row[1], row[2], row[3]);
                    }
                }
                const std::int64_t lane = shuffle ? shuffled(l) : l;
                for (int s4 = 0; s4 < 4; ++s4) {
                    dst[s4 * stride + (k - k0) * kNr + lane] = out[s4];
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
void transform_outputs(const Conv & c, const float * products, std::size_t stride, std::int64_t nt, std::int64_t n,
                       std::int64_t t0, std::int64_t t1, std::int64_t r0, std::int64_t r1, const Epilogue & chain,
                       bool shuffle) {
    const Phase_scope scope(Phase::transform_out);
    const bool add = chain.from_redirect;
    char * image = c.y + n * c.ynb3;
    for (std::int64_t r = r0; r < r1; ++r) {
        const float * m = products + (r - r0) * nt;
        char * plane = image + r * c.ynb2;
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
                sdasm_winograd_outputs_avx512(m + (t - t0), static_cast<std::int64_t>(stride * sizeof(float)),
                                              reinterpret_cast<float *>(plane + 2 * ty * c.ynb1) + 2 * tx,
                                              reinterpret_cast<float *>(plane + (2 * ty + 1) * c.ynb1) + 2 * tx, groups,
                                              add ? 1 : 0);
                t += 16 * groups;
                continue;
            }
            if (shuffle && whole >= 8 && (t - t0) % 8 == 0 && 2 * ty + 2 <= c.oh) {
                // Eight tiles at a time, in the shuffled lanes the packing left
                // them in, by the kernel written by hand.
                const std::int64_t groups = whole / 8;
                sdasm_winograd_outputs_avx2(m + (t - t0), static_cast<std::int64_t>(stride * sizeof(float)),
                                            reinterpret_cast<float *>(plane + 2 * ty * c.ynb1) + 2 * tx,
                                            reinterpret_cast<float *>(plane + (2 * ty + 1) * c.ynb1) + 2 * tx, groups,
                                            add ? 1 : 0);
                t += 8 * groups;
                continue;
            }
#endif
            if (!shuffle && whole > 0 && 2 * ty + 2 <= c.oh) {
                float * row0 = reinterpret_cast<float *>(plane + 2 * ty * c.ynb1) + 2 * tx;
                float * row1 = reinterpret_cast<float *>(plane + (2 * ty + 1) * c.ynb1) + 2 * tx;
                const float * v = m + (t - t0);
                for (std::int64_t l = 0; l < whole; ++l) {
                    // A^T M A: rows 0 and 1 of A^T M, then their columns.
                    float a0[4];
                    float a1[4];
                    for (int j = 0; j < 4; ++j) {
                        const float v0 = v[static_cast<std::size_t>(0 * 4 + j) * stride + static_cast<std::size_t>(l)];
                        const float v1 = v[static_cast<std::size_t>(1 * 4 + j) * stride + static_cast<std::size_t>(l)];
                        const float v2 = v[static_cast<std::size_t>(2 * 4 + j) * stride + static_cast<std::size_t>(l)];
                        const float v3 = v[static_cast<std::size_t>(3 * 4 + j) * stride + static_cast<std::size_t>(l)];
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
            const auto lane = static_cast<std::size_t>(shuffle ? shuffled(l) : l);
            float v[kXi];
            for (int xi = 0; xi < kXi; ++xi) {
                v[xi] = m[static_cast<std::size_t>(xi) * stride + lane];
            }
            float a0[4];
            float a1[4];
            for (int j = 0; j < 4; ++j) {
                a0[j] = v[0 * 4 + j] + v[1 * 4 + j] + v[2 * 4 + j];
                a1[j] = v[1 * 4 + j] - v[2 * 4 + j] - v[3 * 4 + j];
            }
            const auto put = [&](float * p, float x) { *p = add ? *p + x : x; };
            const std::int64_t yy = 2 * ty;
            const std::int64_t xx = 2 * tx;
            float * row0 = reinterpret_cast<float *>(plane + yy * c.ynb1);
            put(row0 + xx, a0[0] + a0[1] + a0[2]);
            if (xx + 1 < c.ow) {
                put(row0 + xx + 1, a0[1] - a0[2] - a0[3]);
            }
            if (yy + 1 < c.oh) {
                float * row1 = reinterpret_cast<float *>(plane + (yy + 1) * c.ynb1);
                put(row1 + xx, a1[0] + a1[1] + a1[2]);
                if (xx + 1 < c.ow) {
                    put(row1 + xx + 1, a1[1] - a1[2] - a1[3]);
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
                    float * row = reinterpret_cast<float *>(plane + yy * c.ynb1) + x0;
                    chain.apply_positions(row, x1 - x0, yy * c.ow + x0, r, n);
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
void transform_inputs_small(float * v, const Conv & c, std::int64_t panels, std::int64_t n, std::int64_t k0,
                            std::int64_t k1, float * scratch) {
    const std::int64_t sw = 2 * c.tw + 2;
    const std::int64_t sh = 2 * c.th + 2;
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t last = c.n * tiles;
    const auto at = [&](std::int64_t xi, std::int64_t g, std::int64_t k) {
        return v + xi * inputs_stride(panels, c.ic) + ((g / kNr) * c.ic + k) * kNr + g % kNr;
    };
    for (std::int64_t k = k0; k < k1; ++k) {
        const char * plane = c.x + n * c.xnb3 + k * c.xnb2;
        for (std::int64_t y = 0; y < sh; ++y) {
            float * row = scratch + y * sw;
            const std::int64_t iy = y - c.p1;
            for (std::int64_t x = 0; x < sw; ++x) {
                const std::int64_t ix = x - c.p0;
                row[x] = iy >= 0 && iy < c.ih && ix >= 0 && ix < c.iw
                             ? reinterpret_cast<const float *>(plane + iy * c.xnb1)[ix]
                             : 0.0f;
            }
        }
        for (std::int64_t t = 0; t < tiles; ++t) {
            const float * d = scratch + 2 * (t / c.tw) * sw + 2 * (t % c.tw);
            const std::int64_t g = n * tiles + t;
            for (int s = 0; s < 4; ++s) {
                float row[4];
                for (int i = 0; i < 4; ++i) {
                    const float * di = d + i * sw;
                    row[i] = bt(s, di[0], di[1], di[2], di[3]);
                }
                for (int r = 0; r < 4; ++r) {
                    *at(r * 4 + s, g, k) = bt(r, row[0], row[1], row[2], row[3]);
                }
            }
        }
        if (n == c.n - 1) {
            for (std::int64_t g = last; g < panels * kNr; ++g) {
                for (std::int64_t xi = 0; xi < kXi; ++xi) {
                    *at(xi, g, k) = 0.0f;
                }
            }
        }
    }
}

// G g G^T of the kernels of output channels [r0, r0 + rows), rows <= kMr,
// and input channels [k0, k1), into a panel of A of each xi, [k1 - k0][kMr]:
// that of xi at dst + xi * stride. Each kWidened input channels are widened
// first into `widened`, [channel][3 * 3][kMr], and their kernels transformed
// kMr at a time, a channel to a lane: G g first, a column of g at a time, then
// (G g) G^T, a row at a time, each sum of three taken left to right and halved
// after.
void transform_weights(float * dst, std::int64_t stride, const Conv & c, std::int64_t r0, std::int64_t rows,
                       std::int64_t k0, std::int64_t k1, float * widened) {
    for (std::int64_t s0 = k0; s0 < k1; s0 += kWidened) {
        const std::int64_t s1 = std::min(k1, s0 + kWidened);
        if (c.wtype == GGML_TYPE_F16) {
            pack_rows_f16(widened, reinterpret_cast<const std::uint16_t *>(c.w + r0 * c.wnb3), c.wnb3, rows, s0 * 9,
                          s1 * 9, kMr);
        }
        else {
            pack_rows_f32(widened, reinterpret_cast<const float *>(c.w + r0 * c.wnb3), c.wnb3, rows, s0 * 9, s1 * 9,
                          kMr);
        }
#if SDASM_X86_64
        if (avx2_kernels()) {
            sdasm_winograd_weights_avx2(dst + (s0 - k0) * kMr, stride * static_cast<std::int64_t>(sizeof(float)), widened,
                                        s1 - s0);
            continue;
        }
#endif
        for (std::int64_t k = s0; k < s1; ++k) {
            // The nine numbers of the kernels of the panel, kMr side by side.
            const float * g = widened + (k - s0) * 9 * kMr;
            float t[4][3][kMr];  // G g
            for (int j = 0; j < 3; ++j) {
                for (int lane = 0; lane < kMr; ++lane) {
                    const float g0 = g[(0 * 3 + j) * kMr + lane];
                    const float g1 = g[(1 * 3 + j) * kMr + lane];
                    const float g2 = g[(2 * 3 + j) * kMr + lane];
                    t[0][j][lane] = g0;
                    t[1][j][lane] = 0.5f * (g0 + g1 + g2);
                    t[2][j][lane] = 0.5f * (g0 - g1 + g2);
                    t[3][j][lane] = g2;
                }
            }
            float * out = dst + (k - k0) * kMr;
            for (int i = 0; i < 4; ++i) {
                for (int lane = 0; lane < kMr; ++lane) {
                    const float a = t[i][0][lane];
                    const float b = t[i][1][lane];
                    const float d = t[i][2][lane];
                    out[(i * 4 + 0) * stride + lane] = a;
                    out[(i * 4 + 1) * stride + lane] = 0.5f * (a + b + d);
                    out[(i * 4 + 2) * stride + lane] = 0.5f * (a - b + d);
                    out[(i * 4 + 3) * stride + lane] = d;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- on AMX

bool on_amx(const Conv & c) {
    return split_form(Split_use::winograd).kernel != nullptr && c.wtype == GGML_TYPE_F16;
}

// The bytes of the tiles of A of sixteen output channels over all input
// channels, and of all of an xi, the channels rounded up to 32.
std::int64_t amx_row_tile_bytes(const Conv & c) {
    return round_up(c.ic, kAmxBlock) / kAmxBlock * kAmxTileBytes;
}

std::int64_t amx_xi_bytes(const Conv & c, std::int64_t rows) {
    return round_up(rows, kAmxBlock) / kAmxTile * amx_row_tile_bytes(c);
}

// The scratch of the transform of the weights into tiles: the kernels of
// sixteen output channels by 32 input channels, widened, [32 * 3 * 3][16], and
// their transforms, [xi][32][16].
constexpr std::int64_t kAmxRowsFloats = kAmxBlock * 9 * kAmxTile + kXi * kAmxBlock * kAmxTile;

// The tiles of A of output channels [r0, r0 + 16) over input channels [k0, k1),
// k0 a multiple of 32: those of xi at dst + xi * xi_bytes, those of the block
// of k at k0 at (k0 - kbase) / 32 * kAmxTileBytes after. The kernels of a block
// of 32 input channels widened side by side (pack_rows_f16()), then
// transformed, sixteen output channels at a time, and split into the tiles
// (x86_64/amx_winograd.S); the output channels past the last, and the input
// channels, transform to zeros.
void amx_weights_tile(std::uint16_t * dst, std::int64_t xi_bytes, std::int64_t kbase, const Conv & c, std::int64_t r0,
                      std::int64_t k0, std::int64_t k1, float * scratch) {
    float * widened = scratch;
    float * transformed = scratch + kAmxBlock * 9 * kAmxTile;
    const std::int64_t rows = std::clamp<std::int64_t>(c.oc - r0, 0, kAmxTile);
    for (std::int64_t kb = k0; kb < k1; kb += kAmxBlock) {
        const std::int64_t kc = std::min(kAmxBlock, c.ic - kb);
        std::uint16_t * out = dst + (kb - kbase) / kAmxBlock * kAmxTileBytes / 2;
        if (rows == 0 || kc <= 0) {
            for (int xi = 0; xi < kXi; ++xi) {
                std::memset(out + xi * xi_bytes / 2, 0, kAmxTileBytes);
            }
            continue;
        }
        pack_rows_f16(widened, reinterpret_cast<const std::uint16_t *>(c.w + r0 * c.wnb3), c.wnb3, rows, kb * 9,
                      (kb + kc) * 9, kAmxTile);
        sdasm_amx_winograd_weights(out, xi_bytes, widened, kc, transformed);
    }
}

struct Amx_plan {
    std::int64_t mc, kc, tiles, mblocks, tblocks;
    std::int64_t stride;           // the floats between the panels of the inputs of two xi
    std::size_t shared_bytes;      // the tiles of A of every xi; small images: the inputs too
    std::size_t panel_bytes;       // the inputs of four xi in single precision
    std::size_t column_bytes;      // the tiles of B of an xi; small images: the tiles of A of every xi
    std::size_t product_bytes;
    std::size_t rows_bytes;        // G g G^T of sixteen channels by 32
    std::size_t scratch_bytes;     // small images: a channel of an image, padded
    std::size_t thread_bytes;
    std::int64_t panels;           // small images: the panels of sixteen tiles of all images, rounded up to two
    std::size_t inputs_bytes;      // small images: the inputs of every xi in single precision
};

Amx_plan amx_plan(const Conv & c, int n_threads) {
    const Tuning & t = tuning();
    Amx_plan pl{};
    pl.rows_bytes = align_up(static_cast<std::size_t>(kAmxRowsFloats) * sizeof(float));
    const std::size_t spare_bytes = align_up(kAmxBlock * kAmxBlock * sizeof(float));
    if (small(c)) {
        pl.panels = round_up(ceil_div(c.n * c.tw * c.th, kNr), 2);
        pl.kc = std::min(round_up(c.ic, kAmxBlock), round_up(std::max(t.winograd_small_kc, 1), kAmxBlock));
        pl.mc = std::min(round_up(c.oc, kAmxBlock), round_up(std::max(t.winograd_small_mc, 1), kAmxBlock));
        pl.inputs_bytes = align_up(static_cast<std::size_t>(kXi * inputs_stride(pl.panels, c.ic)) * sizeof(float));
        pl.shared_bytes = pl.inputs_bytes +
                          align_up(static_cast<std::size_t>(kXi * pl.panels * amx_row_tile_bytes(c)));
        pl.column_bytes = align_up(static_cast<std::size_t>(kXi * pl.mc / kAmxTile * (pl.kc / kAmxBlock) * kAmxTileBytes));
        pl.scratch_bytes = align_up(static_cast<std::size_t>((2 * c.th + 2) * (2 * c.tw + 2)) * sizeof(float));
        pl.product_bytes =
            align_up(static_cast<std::size_t>(kXi * products_stride(pl.mc, pl.panels * kNr)) * sizeof(float));
        pl.thread_bytes = pl.column_bytes + pl.rows_bytes + pl.scratch_bytes + pl.product_bytes + spare_bytes;
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
    pl.shared_bytes = align_up(static_cast<std::size_t>(kXi * amx_xi_bytes(c, c.oc)));
    pl.stride = skewed(pl.kc * pl.tiles);
    pl.panel_bytes = align_up(4 * static_cast<std::size_t>(pl.stride) * sizeof(float));
    pl.column_bytes = align_up(static_cast<std::size_t>(round_up(pl.tiles, kAmxBlock) / kAmxTile * (pl.kc / kAmxBlock) *
                                                        kAmxTileBytes));
    pl.product_bytes = align_up(static_cast<std::size_t>(kXi * products_stride(pl.mc, pl.tiles)) * sizeof(float));
    pl.thread_bytes = pl.panel_bytes + pl.column_bytes + std::max(pl.product_bytes, pl.rows_bytes) + spare_bytes;
    return pl;
}

// The products of the tiles of A of an xi, rows [i0, i1) of them from `a`
// (row_bytes between the tiles of sixteen rows), with the tiles of B of its
// columns [0, nt) from `b` (column_bytes between the tiles of sixteen
// columns), over `blocks` blocks of k, into [row][column] at `out`, rows nt
// floats apart; added to it when accumulate says so.
void amx_xi_products(const Split_form & f, const std::uint16_t * a, std::int64_t row_bytes, const std::uint16_t * b, std::int64_t column_bytes,
                     std::int64_t blocks, std::int64_t i0, std::int64_t i1, std::int64_t nt, float * out, bool accumulate,
                     float * spare) {
    const std::int64_t ldc = nt * static_cast<std::int64_t>(sizeof(float));
    for (std::int64_t jj = 0; jj < nt; jj += kAmxBlock) {
        for (std::int64_t ii = i0; ii < i1; ii += kAmxBlock) {
            split_block(f, blocks, a + (ii - i0) / kAmxTile * row_bytes / 2, row_bytes,
                        b + jj / kAmxTile * column_bytes / 2, column_bytes, out + (ii - i0) * nt + jj, ldc, accumulate,
                        std::min(kAmxBlock, i1 - ii), std::min(kAmxBlock, nt - jj), spare);
        }
    }
}

void compute_large_amx(const ggml_compute_params * params, const Conv & c, const Amx_plan & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    const std::int64_t xi_bytes = amx_xi_bytes(c, c.oc);
    const std::int64_t row_bytes = amx_row_tile_bytes(c);
    const std::shared_ptr<Packed> kept = packed(c.kernel, Layout::amx_winograd, static_cast<std::size_t>(kXi * xi_bytes));
    auto * weights = reinterpret_cast<std::uint16_t *>(kept != nullptr ? reinterpret_cast<char *>(kept->data) : base);
    const bool fill = kept == nullptr || !kept->ready();
    char * mine = base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    auto * panels = reinterpret_cast<float *>(mine);
    auto * columns = reinterpret_cast<std::uint16_t *>(mine + pl.panel_bytes);
    auto * products = reinterpret_cast<float *>(mine + pl.panel_bytes + pl.column_bytes);
    auto * rows = products;  // G g G^T, before the products begin
    auto * spare = reinterpret_cast<float *>(mine + pl.panel_bytes + pl.column_bytes + std::max(pl.product_bytes, pl.rows_bytes));
    const Split_form f = split_form(Split_use::winograd);

    Work_queue queue(params);
    // The weights, a tile of sixteen output channels to an item, over every
    // input channel: those past the last are zeros.
    const std::int64_t pack_items = fill ? round_up(c.oc, kAmxBlock) / kAmxTile : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            amx_weights_tile(weights + item * row_bytes / 2, xi_bytes, 0, c, item * kAmxTile, 0, round_up(c.ic, kAmxBlock),
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
        const auto pstride = static_cast<std::size_t>(products_stride(pl.mc, nt));
        for (int r = 0; r < 4; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                const std::int64_t kb = ceil_div(kc, kAmxBlock);
                const std::int64_t column_bytes = kb * kAmxTileBytes;
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs(panels, pl.stride, c, n, t0, t1, k0, k0 + kc, r, false);
                }
                for (int s = 0; s < 4; ++s) {
                    const int xi = 4 * r + s;
                    {
                        // The tiles of B of this xi, from its panels; one more of
                        // zeros past an odd number of them, for the last pair.
                        const Phase_scope scope(Phase::pack_inputs);
                        for (std::int64_t p = 0; p < nt / kNr; ++p) {
                            sdasm_amx_pack_panel(columns + p * column_bytes / 2, panels + s * pl.stride + p * kc * kNr, kc);
                        }
                        if ((nt / kNr) % 2 != 0) {
                            std::memset(columns + (nt / kNr) * column_bytes / 2, 0, static_cast<std::size_t>(column_bytes));
                        }
                    }
                    const Phase_scope scope(Phase::product);
                    amx_xi_products(f, weights + (xi * xi_bytes + i0 / kAmxTile * row_bytes + k0 / kAmxBlock * kAmxTileBytes) / 2,
                                    row_bytes, columns, column_bytes, kb, i0, i1, nt, products + static_cast<std::size_t>(xi) * pstride, k0 > 0,
                                    spare);
                }
            }
        }
        transform_outputs(c, products, pstride, nt, n, t0, t1, i0, i1, chain, false);
    }
    if (configured) {
        split_end(f);
    }
}

void compute_small_amx(const ggml_compute_params * params, const Conv & c, const Amx_plan & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    auto * inputs = reinterpret_cast<float *>(base);
    auto * inputs_amx = reinterpret_cast<std::uint16_t *>(base + pl.inputs_bytes);
    char * mine = base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    auto * weights = reinterpret_cast<std::uint16_t *>(mine);
    auto * rows = reinterpret_cast<float *>(mine + pl.column_bytes);
    auto * scratch = reinterpret_cast<float *>(mine + pl.column_bytes + pl.rows_bytes);
    auto * products = reinterpret_cast<float *>(mine + pl.column_bytes + pl.rows_bytes + pl.scratch_bytes);
    auto * spare = reinterpret_cast<float *>(mine + pl.column_bytes + pl.rows_bytes + pl.scratch_bytes + pl.product_bytes);
    const Split_form f = split_form(Split_use::winograd);
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t nt = pl.panels * kNr;
    const std::int64_t panel_bytes = amx_row_tile_bytes(c);  // a panel of sixteen tiles over every input channel

    // The inputs in single precision, by all threads together, sixteen channels
    // of an image to an item; then their tiles of B, an xi and a pair of panels
    // to an item.
    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params);
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
            sdasm_amx_pack_panel(inputs_amx + (xi * pl.panels + p) * panel_bytes / 2,
                                 inputs + xi * inputs_stride(pl.panels, c.ic) + p * c.ic * kNr, c.ic);
        }
    }
    queue.barrier();

    // The blocks of output channels, each over every tile, transforming the
    // weights of its channels a stretch of the input channels at a time.
    const std::vector<Block> blocks = guided_blocks(1, c.oc, kAmxBlock, pl.mc, params->nth);
    const auto units = static_cast<std::int64_t>(blocks.size());
    const auto pstride = static_cast<std::size_t>(products_stride(pl.mc, nt));
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
            const std::int64_t row_bytes = kb * kAmxTileBytes;
            const std::int64_t xi_bytes = round_up(i1 - i0, kAmxBlock) / kAmxTile * row_bytes;
            {
                const Phase_scope scope(Phase::pack_weights);
                for (std::int64_t r0 = i0; r0 < i0 + round_up(i1 - i0, kAmxBlock); r0 += kAmxTile) {
                    amx_weights_tile(weights + (r0 - i0) / kAmxTile * row_bytes / 2, xi_bytes, k0, c, r0, k0, k0 + kc, rows);
                }
            }
            const Phase_scope scope(Phase::product);
            for (int xi = 0; xi < kXi; ++xi) {
                amx_xi_products(f, weights + xi * xi_bytes / 2, row_bytes,
                                inputs_amx + (xi * pl.panels * panel_bytes + k0 / kAmxBlock * kAmxTileBytes) / 2, panel_bytes,
                                kb, i0, i1, nt, products + static_cast<std::size_t>(xi) * pstride, k0 > 0, spare);
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs(c, products + n * tiles, pstride, nt, n, 0, tiles, i0, i1, chain, false);
        }
    }
    if (configured) {
        split_end(f);
    }
}

void compute_small(const ggml_compute_params * params, const Conv & c, const Plan & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    float * inputs = reinterpret_cast<float *>(base);
    char * mine = base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    float * widened = reinterpret_cast<float *>(mine);
    float * weights = reinterpret_cast<float *>(mine + pl.panel_bytes);
    float * scratch = reinterpret_cast<float *>(mine + pl.panel_bytes + pl.weight_bytes);
    float * products = reinterpret_cast<float *>(mine + pl.panel_bytes + pl.weight_bytes + pl.scratch_bytes);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t tiles = c.tw * c.th;
    const std::int64_t nt = pl.panels * kNr;

    // The inputs, by all threads together, sixteen channels of an image to an
    // item.
    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params);
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
                    transform_weights(weights + (ii - i0) * kc, stride, c, ii, std::min<std::int64_t>(kMr, i1 - ii), k0, k0 + kc,
                                      widened);
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
            const std::int64_t width = 9 * static_cast<std::int64_t>(ggml_type_size(c.wtype));
            const std::int64_t rows_per_xi = ceil_div(i1 - i0, kXi);
            for (int xi = 0; xi < kXi; ++xi) {
                for (std::int64_t r = i0 + xi * rows_per_xi; next0 < next1 && r < std::min(i1, i0 + (xi + 1) * rows_per_xi); ++r) {
                    const auto line = [&](std::int64_t channel) {
                        return reinterpret_cast<std::uintptr_t>(c.w + r * c.wnb3 + channel * width) / 64 * 64;
                    };
                    for (std::uintptr_t q = line(next0); q < line(next1 - 1) + 64; q += 64) {
                        __builtin_prefetch(reinterpret_cast<const char *>(q), 0, 2);
                    }
                }
                const float * u = weights + static_cast<std::size_t>(xi) * static_cast<std::size_t>(stride);
                float * product = products + static_cast<std::size_t>(xi) * static_cast<std::size_t>(products_stride(pl.mc, nt));
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    const float * a = u + (ii - i0) * kc;
                    for (std::int64_t p = 0; p < pl.panels; p += panel_step(wide) / kNr) {
                        const float * b = inputs + xi * inputs_stride(pl.panels, c.ic) + (p * c.ic + k0) * kNr;
                        float * out = product + (ii - i0) * nt + p * kNr;
                        sdasm::tiles(uk, wide, kc, a, b, c.ic * kNr * static_cast<std::int64_t>(sizeof(float)), out,
                                     nt * static_cast<std::int64_t>(sizeof(float)), k0 > 0,
                                     std::min<std::int64_t>(kMr, i1 - ii), std::min(panel_step(wide), nt - p * kNr));
                    }
                }
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs(c, products + n * tiles, static_cast<std::size_t>(products_stride(pl.mc, nt)), nt, n, 0,
                              tiles, i0, i1, chain, false);
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
constexpr float kBT4[6][6] = {{1.0f, 1.5f, -2.0f, -1.5f, 1.0f, 0.0f}, {0.0f, -1.0f, -2.5f, -0.5f, 1.0f, 0.0f},
                              {0.0f, 1.0f, 0.5f, -2.5f, 1.0f, 0.0f},  {0.0f, -0.5f, -1.0f, 0.5f, 1.0f, 0.0f},
                              {0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.5f, -2.0f, -1.5f, 1.0f}};
constexpr float kAT4[4][6] = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
                              {0.0f, 1.0f, -1.0f, 2.0f, -0.5f, 0.0f},
                              {0.0f, 1.0f, 1.0f, 4.0f, 0.25f, 0.0f},
                              {0.0f, 1.0f, -1.0f, 8.0f, -0.125f, 1.0f}};

// For FP16 weights alone, whose products ggml computes of the image rounded to
// FP16: those of FP32 weights it computes in single precision, closer than
// F(4x4, 3x3) can.
// The kernels written by hand for AVX2 leave the eight tiles of each eight of a
// panel in the order 0, 2, 4, 6, 1, 3, 5, 7 (x86_64/winograd4_avx2.S): tile l
// of a block in lane shuffled4(l).
constexpr std::int64_t kShuffle4[8] = {0, 4, 1, 5, 2, 6, 3, 7};

std::int64_t shuffled4(std::int64_t l) {
    return (l & ~std::int64_t{7}) | kShuffle4[l & 7];
}

bool four(const Conv & c) {
    return tuning().winograd == 2 && c.wtype == GGML_TYPE_F16 && !on_amx(c) && c.ow * c.oh >= tuning().winograd4_positions;
}

struct Plan4 {
    std::int64_t tw, th;  // tiles of four by four across and down
    std::int64_t mc, kc, tiles, mblocks, tblocks;
    std::int64_t stride;  // the floats between the panels of the inputs of two xi
    std::size_t shared_bytes, panel_bytes, product_bytes, thread_bytes;
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
    pl.shared_bytes = align_up(static_cast<std::size_t>(kXi4 * xi_stride(round_up(c.oc, kMr), c.ic)) * sizeof(float));
    pl.stride = skewed(pl.kc * pl.tiles);
    pl.panel_bytes = std::max(6 * static_cast<std::size_t>(pl.stride) * sizeof(float),
                              align_up(static_cast<std::size_t>(kMr * 9) * sizeof(float)));
    pl.product_bytes = align_up(static_cast<std::size_t>(kXi4 * products_stride(pl.mc, pl.tiles)) * sizeof(float));
    pl.thread_bytes = pl.panel_bytes + pl.product_bytes;
    return pl;
}

// G g G^T of the kernels of output channels [r0, r0 + rows), rows <= kMr, and
// input channels [k0, k1), into a panel of A of each xi, [k1 - k0][kMr]: that
// of xi at dst + xi * stride. The kernels of each eight input channels are widened
// first into `widened`, [channel][3 * 3][kMr] (pack_rows_f16()), and
// transformed kMr at a time, a lane to each output channel, in single
// precision: G g a column at a time, then (G g) G^T a row at a time.
constexpr float kG4f[6][3] = {{1.0f, 0.0f, 0.0f},
                              {-1.0f / 3, -1.0f / 3, -1.0f / 3},
                              {1.0f / 3, -1.0f / 3, 1.0f / 3},
                              {1.0f / 15, 2.0f / 15, 4.0f / 15},
                              {-16.0f / 15, 8.0f / 15, -4.0f / 15},
                              {0.0f, 0.0f, 1.0f}};

void transform_weights4(float * dst, std::int64_t stride, const Conv & c, std::int64_t r0, std::int64_t rows,
                        std::int64_t k0, std::int64_t k1, float * widened) {
    for (std::int64_t s0 = k0; s0 < k1; s0 += kWidened) {
        const std::int64_t s1 = std::min(k1, s0 + kWidened);
        if (c.wtype == GGML_TYPE_F16) {
            pack_rows_f16(widened, reinterpret_cast<const std::uint16_t *>(c.w + r0 * c.wnb3), c.wnb3, rows, s0 * 9,
                          s1 * 9, kMr);
        }
        else {
            pack_rows_f32(widened, reinterpret_cast<const float *>(c.w + r0 * c.wnb3), c.wnb3, rows, s0 * 9, s1 * 9, kMr);
        }
#if SDASM_X86_64
        if (avx2_kernels()) {
            sdasm_winograd4_weights_avx2(dst + (s0 - k0) * kMr, stride * static_cast<std::int64_t>(sizeof(float)), widened,
                                         s1 - s0);
            continue;
        }
#endif
        for (std::int64_t k = s0; k < s1; ++k) {
            const float * g = widened + (k - s0) * 9 * kMr;
            float t[6][3][kMr];  // G g
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int lane = 0; lane < kMr; ++lane) {
                        t[i][j][lane] = kG4f[i][0] * g[(0 * 3 + j) * kMr + lane] + kG4f[i][1] * g[(1 * 3 + j) * kMr + lane] +
                                        kG4f[i][2] * g[(2 * 3 + j) * kMr + lane];
                    }
                }
            }
            float * out = dst + (k - k0) * kMr;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    for (int lane = 0; lane < kMr; ++lane) {
                        out[(i * 6 + j) * stride + lane] =
                            t[i][0][lane] * kG4f[j][0] + t[i][1][lane] * kG4f[j][1] + t[i][2][lane] * kG4f[j][2];
                    }
                }
            }
        }
    }
}

// The elements xi = (r, s), s = 0..5, of the transformed inputs of tiles [t0,
// t1) of image n and channels [k0, k1), into panels of kNr tiles, [panel][k1 -
// k0][kNr], zeros past t1: those of (r, s) at dst + s * stride. Row r of B^T
// first takes its combination of the six rows of each tile, then each s its
// combination of the six numbers of that.
void pack_inputs4(float * dst, std::int64_t stride, const Conv & c, const Plan4 & pl, std::int64_t n, std::int64_t t0,
                  std::int64_t t1, std::int64_t k0, std::int64_t k1, int r, bool shuffle) {
    const std::int64_t kc = k1 - k0;
    const char * image = c.x + n * c.xnb3;
    for (std::int64_t tt = t0; tt < t1; tt += kNr, dst += kc * kNr) {
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
            job.dst = dst;
            job.stride = stride * static_cast<std::int64_t>(sizeof(float));
            job.step = c.xnb2;
            job.channels = kc;
            for (int i = 0; i < 6; ++i) {
                const std::int64_t y = y0 + i;
                if (kBT4[r][i] == 0.0f || y < 0 || y >= c.ih) {
                    continue;
                }
                // The address of the first input, which may lie before the
                // row: it is formed but never read.
                job.row[job.rows] = reinterpret_cast<const char *>(
                    reinterpret_cast<std::uintptr_t>(image + k0 * c.xnb2 + y * c.xnb1) +
                    static_cast<std::uintptr_t>(x0 * static_cast<std::int64_t>(sizeof(float))));
                job.coefficient[job.rows] = kBT4[r][i];
                ++job.rows;
            }
            for (int m = 0; m < 5; ++m) {
                std::uint16_t lanes = 0;
                for (int lane = 0; lane < 16; ++lane) {
                    const std::int64_t x = x0 + 16 * m + lane;
                    lanes = static_cast<std::uint16_t>(lanes | ((x >= 0 && x < c.iw && 16 * m + lane < 4 * kNr + 2 ? 1u : 0u) << lane));
                }
                job.lanes[m] = lanes;
            }
            sdasm_winograd4_inputs_avx512(&job);
            continue;
        }
        if (valid == kNr && first + kNr <= pl.tw && shuffle) {
            // Sixteen tiles of a row by the kernel for AVX2, eight at a time,
            // with masks for the loads that reach past the image.
            for (int half = 0; half < 2; ++half) {
                const std::int64_t xh = x0 + 32 * half;
                Winograd4_inputs job{};
                job.dst = dst + 8 * half;
                job.stride = stride * static_cast<std::int64_t>(sizeof(float));
                job.step = c.xnb2;
                job.channels = kc;
                for (int i = 0; i < 6; ++i) {
                    const std::int64_t y = y0 + i;
                    if (kBT4[r][i] == 0.0f || y < 0 || y >= c.ih) {
                        continue;
                    }
                    job.row[job.rows] = reinterpret_cast<const char *>(
                        reinterpret_cast<std::uintptr_t>(image + k0 * c.xnb2 + y * c.xnb1) +
                        static_cast<std::uintptr_t>(xh * static_cast<std::int64_t>(sizeof(float))));
                    job.coefficient[job.rows] = kBT4[r][i];
                    ++job.rows;
                }
                alignas(32) std::int32_t masks[5][8];
                const bool inside = xh >= 0 && xh + 34 <= c.iw;
                for (int m = 0; m < 5; ++m) {
                    for (int lane = 0; lane < 8; ++lane) {
                        const std::int64_t x = xh + 8 * m + lane;
                        masks[m][lane] = x >= 0 && x < c.iw && 8 * m + lane < 34 ? -1 : 0;
                    }
                }
                job.masks = inside ? nullptr : &masks[0][0];
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
                const char * plane = image + k * c.xnb2;
                float row[kSpan] = {};
                for (int i = 0; i < 6; ++i) {
                    const float b = kBT4[r][i];
                    const std::int64_t y = y0 + i;
                    if (b == 0.0f || y < 0 || y >= c.ih) {
                        continue;
                    }
                    const float * p = reinterpret_cast<const float *>(plane + y * c.xnb1);
                    for (std::int64_t x = xa; x < xb; ++x) {
                        row[x] += b * p[x0 + x];
                    }
                }
                float u[6][kNr];
                for (int l = 0; l < kNr; ++l) {
                    for (int j = 0; j < 6; ++j) {
                        u[j][l] = row[4 * l + j];
                    }
                }
                for (int s4 = 0; s4 < 6; ++s4) {
                    float * out = dst + s4 * stride + (k - k0) * kNr;
                    for (int l = 0; l < kNr; ++l) {
                        float v = 0.0f;
                        for (int j = 0; j < 6; ++j) {
                            v += kBT4[s4][j] * u[j][l];
                        }
                        out[l] = v;
                    }
                }
            }
            continue;
        }
        for (std::int64_t k = k0; k < k1; ++k) {
            const char * plane = image + k * c.xnb2;
            for (std::int64_t l = 0; l < kNr; ++l) {
                float out[6] = {};
                if (l < valid) {
                    const std::int64_t t = tt + l;
                    const std::int64_t ty0 = 4 * (t / pl.tw) - c.p1;
                    const std::int64_t tx0 = 4 * (t % pl.tw) - c.p0;
                    float u[6] = {};
                    for (int i = 0; i < 6; ++i) {
                        const float b = kBT4[r][i];
                        if (b == 0.0f) {
                            continue;
                        }
                        for (int j = 0; j < 6; ++j) {
                            u[j] += b * input_at(c, plane, ty0 + i, tx0 + j);
                        }
                    }
                    for (int s4 = 0; s4 < 6; ++s4) {
                        float v = 0.0f;
                        for (int j = 0; j < 6; ++j) {
                            v += kBT4[s4][j] * u[j];
                        }
                        out[s4] = v;
                    }
                }
                const std::int64_t lane = shuffle ? shuffled4(l) : l;
                for (int s4 = 0; s4 < 6; ++s4) {
                    dst[s4 * stride + (k - k0) * kNr + lane] = out[s4];
                }
            }
        }
    }
}

// Y = A^T M A for tiles [t0, t1) of image n and output channels [r0, r1), from
// their 36 products, [xi][channel][tile], those of xi `stride` floats apart;
// added to the outputs when the chain begins its sum from them. Then the chain.
void transform_outputs4(const Conv & c, const Plan4 & pl, const float * products, std::size_t stride, std::int64_t nt,
                        std::int64_t n, std::int64_t t0, std::int64_t t1, std::int64_t r0, std::int64_t r1,
                        const Epilogue & chain, bool shuffle) {
    const Phase_scope scope(Phase::transform_out);
    const bool add = chain.from_redirect;
    char * image = c.y + n * c.ynb3;
    for (std::int64_t r = r0; r < r1; ++r) {
        const float * m = products + (r - r0) * nt;
        char * plane = image + r * c.ynb2;
        for (std::int64_t t = t0; t < t1;) {
#if SDASM_X86_64
            const std::int64_t ty = t / pl.tw;
            const std::int64_t tx = t % pl.tw;
            // The tiles of this row of tiles from t on, whole in the output.
            const std::int64_t whole = std::min({t1 - t, pl.tw - tx, (c.ow - 4 * tx) / 4});
            if (!shuffle && whole >= 16 && 4 * ty + 4 <= c.oh && avx512_kernels()) {
                // Sixteen tiles at a time by the kernel for AVX-512.
                const std::int64_t groups = whole / 16;
                sdasm_winograd4_outputs_avx512(m + (t - t0), static_cast<std::int64_t>(stride * sizeof(float)),
                                               reinterpret_cast<float *>(plane + 4 * ty * c.ynb1) + 4 * tx, c.ynb1, groups,
                                               add ? 1 : 0);
                t += 16 * groups;
                continue;
            }
            if (shuffle && whole >= 8 && (t - t0) % 8 == 0 && 4 * ty + 4 <= c.oh) {
                // Eight tiles at a time, in the shuffled lanes the packing left
                // them in, by the kernel for AVX2.
                const std::int64_t groups = whole / 8;
                sdasm_winograd4_outputs_avx2(m + (t - t0), static_cast<std::int64_t>(stride * sizeof(float)),
                                             reinterpret_cast<float *>(plane + 4 * ty * c.ynb1) + 4 * tx, c.ynb1, groups,
                                             add ? 1 : 0);
                t += 8 * groups;
                continue;
            }
#endif
            const auto l = static_cast<std::size_t>(shuffle ? shuffled4(t - t0) : t - t0);
            float a[4][6];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 6; ++j) {
                    float sum = 0.0f;
                    for (int x = 0; x < 6; ++x) {
                        sum += kAT4[i][x] * m[static_cast<std::size_t>(x * 6 + j) * stride + l];
                    }
                    a[i][j] = sum;
                }
            }
            const std::int64_t yy = 4 * (t / pl.tw);
            const std::int64_t xx = 4 * (t % pl.tw);
            for (int i = 0; i < 4 && yy + i < c.oh; ++i) {
                float * row = reinterpret_cast<float *>(plane + (yy + i) * c.ynb1);
                for (int jj = 0; jj < 4 && xx + jj < c.ow; ++jj) {
                    float y = 0.0f;
                    for (int j = 0; j < 6; ++j) {
                        y += a[i][j] * kAT4[jj][j];
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
                    float * row = reinterpret_cast<float *>(plane + yy * c.ynb1) + x0;
                    chain.apply_positions(row, x1 - x0, yy * c.ow + x0, r, n);
                }
                t += txb - txa;
            }
        }
    }
}

void compute_large4(const ggml_compute_params * params, const Conv & c, const Plan4 & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    float * weights = reinterpret_cast<float *>(base);
    char * mine = base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    float * panels = reinterpret_cast<float *>(mine);
    float * products = reinterpret_cast<float *>(mine + pl.panel_bytes);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const auto per_xi = static_cast<std::size_t>(xi_stride(round_up(c.oc, kMr), c.ic));
    // The kernels written by hand for AVX2 shuffle the tiles of each eight;
    // those for AVX-512, like the C++ ones, keep them in their order.
    const bool shuffle = avx2_kernels() && !avx512_kernels();

    Work_queue queue(params);
    // The weights, a panel of kMr output channels to an item.
    const std::int64_t pack_items = ceil_div(c.oc, kMr);
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const Phase_scope scope(Phase::pack_weights);
        transform_weights4(weights + item * c.ic * kMr, static_cast<std::int64_t>(per_xi), c, item * kMr,
                           std::min<std::int64_t>(kMr, c.oc - item * kMr), 0, c.ic, panels);
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
        const std::int64_t ldc = nt * static_cast<std::int64_t>(sizeof(float));
        const auto pstride = static_cast<std::size_t>(products_stride(pl.mc, nt));
        for (int r = 0; r < 6; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs4(panels, pl.stride, c, pl, n, t0, t1, k0, k0 + kc, r, shuffle);
                }
                const Phase_scope scope(Phase::product);
                for (int s4 = 0; s4 < 6; ++s4) {
                    const int xi = 6 * r + s4;
                    float * product = products + static_cast<std::size_t>(xi) * pstride;
                    const float * u_xi = weights + static_cast<std::size_t>(xi) * per_xi;
                    for (std::int64_t jj = 0; jj < nt; jj += step) {
                        const float * b = panels + s4 * pl.stride + (jj / kNr) * kc * kNr;
                        for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                            const float * a = u_xi + (ii / kMr * c.ic + k0) * kMr;
                            float * out = product + (ii - i0) * nt + jj;
                            sdasm::tiles(uk, wide, kc, a, b, kc * kNr * static_cast<std::int64_t>(sizeof(float)), out, ldc,
                                         k0 > 0, std::min<std::int64_t>(kMr, i1 - ii), std::min(step, nt - jj));
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
    std::size_t shared_bytes, widened_bytes, weight_bytes, scratch_bytes, product_bytes, thread_bytes;
};

Plan4_small plan4_small(const Conv & c) {
    const Tuning & t = tuning();
    Plan4_small pl{};
    pl.tw = (c.ow + 3) / 4;
    pl.th = (c.oh + 3) / 4;
    pl.panels = ceil_div(c.n * pl.tw * pl.th, kNr);
    pl.kc = std::min<std::int64_t>(c.ic, std::max(kWidened, static_cast<std::int64_t>(t.winograd_small_kc)));
    pl.mc = round_up(std::min<std::int64_t>(c.oc, std::max(kMr, t.winograd_small_mc)), kMr);
    pl.shared_bytes = align_up(static_cast<std::size_t>(kXi4 * inputs_stride(pl.panels, c.ic)) * sizeof(float));
    pl.widened_bytes = align_up(static_cast<std::size_t>(kWidened * 9 * kMr + 8) * sizeof(float));
    pl.weight_bytes = align_up(static_cast<std::size_t>(kXi4 * xi_stride(pl.mc, pl.kc)) * sizeof(float));
    pl.scratch_bytes = align_up(static_cast<std::size_t>((4 * pl.th + 2) * (4 * pl.tw + 2)) * sizeof(float));
    pl.product_bytes = align_up(static_cast<std::size_t>(kXi4 * products_stride(pl.mc, pl.panels * kNr)) * sizeof(float));
    pl.thread_bytes = pl.widened_bytes + pl.weight_bytes + pl.scratch_bytes + pl.product_bytes;
    return pl;
}

// The transformed inputs of channels [k0, k1) of image n, every tile, element
// xi of the tile that is g-th of all images at v[xi * inputs_stride() + ((g /
// kNr) * ic + k) * kNr + g % kNr]; the lanes past the last tile of the last
// image, zeros. `scratch` holds a channel of the image with the zeros around it.
void transform_inputs_small4(float * v, const Conv & c, const Plan4_small & pl, std::int64_t n, std::int64_t k0,
                             std::int64_t k1, float * scratch, bool shuffle) {
    const std::int64_t sw = 4 * pl.tw + 2;
    const std::int64_t sh = 4 * pl.th + 2;
    const std::int64_t tiles = pl.tw * pl.th;
    const std::int64_t last = c.n * tiles;
    const std::int64_t vstride = inputs_stride(pl.panels, c.ic);
#if SDASM_X86_64
    if (shuffle && pl.tw % 8 == 0) {
        // Every row of tiles in eights, each eight in a lane of eight of its
        // panel, by the kernel for AVX2, a row of B^T at a time over the
        // channels.
        const char * image = c.x + n * c.xnb3;
        for (std::int64_t t = 0; t < tiles; t += 8) {
            const std::int64_t g = n * tiles + t;
            const std::int64_t y0 = 4 * (t / pl.tw) - c.p1;
            const std::int64_t x0 = 4 * (t % pl.tw) - c.p0;
            alignas(32) std::int32_t masks[5][8];
            const bool inside = x0 >= 0 && x0 + 34 <= c.iw;
            for (int m = 0; m < 5; ++m) {
                for (int lane = 0; lane < 8; ++lane) {
                    const std::int64_t x = x0 + 8 * m + lane;
                    masks[m][lane] = x >= 0 && x < c.iw && 8 * m + lane < 34 ? -1 : 0;
                }
            }
            for (int r = 0; r < 6; ++r) {
                Winograd4_inputs job{};
                job.dst = v + 6 * r * vstride + ((g / kNr) * c.ic + k0) * kNr + g % kNr;
                job.stride = vstride * static_cast<std::int64_t>(sizeof(float));
                job.step = c.xnb2;
                job.channels = k1 - k0;
                for (int i = 0; i < 6; ++i) {
                    const std::int64_t y = y0 + i;
                    if (kBT4[r][i] == 0.0f || y < 0 || y >= c.ih) {
                        continue;
                    }
                    job.row[job.rows] = reinterpret_cast<const char *>(
                        reinterpret_cast<std::uintptr_t>(image + k0 * c.xnb2 + y * c.xnb1) +
                        static_cast<std::uintptr_t>(x0 * static_cast<std::int64_t>(sizeof(float))));
                    job.coefficient[job.rows] = kBT4[r][i];
                    ++job.rows;
                }
                job.masks = inside ? nullptr : &masks[0][0];
                sdasm_winograd4_inputs_avx2(&job);
            }
        }
        if (n == c.n - 1) {
            for (std::int64_t k = k0; k < k1; ++k) {
                for (std::int64_t g = last; g < pl.panels * kNr; ++g) {
                    for (int xi = 0; xi < kXi4; ++xi) {
                        v[xi * vstride + ((g / kNr) * c.ic + k) * kNr + g % kNr] = 0.0f;
                    }
                }
            }
        }
        return;
    }
#else
    (void) shuffle;
#endif
    for (std::int64_t k = k0; k < k1; ++k) {
        const char * plane = c.x + n * c.xnb3 + k * c.xnb2;
        for (std::int64_t y = 0; y < sh; ++y) {
            float * row = scratch + y * sw;
            const std::int64_t iy = y - c.p1;
            for (std::int64_t x = 0; x < sw; ++x) {
                const std::int64_t ix = x - c.p0;
                row[x] = iy >= 0 && iy < c.ih && ix >= 0 && ix < c.iw ? reinterpret_cast<const float *>(plane + iy * c.xnb1)[ix]
                                                                      : 0.0f;
            }
        }
        for (std::int64_t t = 0; t < tiles; ++t) {
            const float * d = scratch + 4 * (t / pl.tw) * sw + 4 * (t % pl.tw);
            const std::int64_t g = n * tiles + t;
            float * out = v + ((g / kNr) * c.ic + k) * kNr + g % kNr;
            for (int r = 0; r < 6; ++r) {
                float u[6] = {};
                for (int i = 0; i < 6; ++i) {
                    const float b = kBT4[r][i];
                    for (int j = 0; j < 6; ++j) {
                        u[j] += b * d[i * sw + j];
                    }
                }
                for (int s4 = 0; s4 < 6; ++s4) {
                    float x = 0.0f;
                    for (int j = 0; j < 6; ++j) {
                        x += kBT4[s4][j] * u[j];
                    }
                    out[(r * 6 + s4) * vstride] = x;
                }
            }
        }
        if (n == c.n - 1) {
            for (std::int64_t g = last; g < pl.panels * kNr; ++g) {
                for (int xi = 0; xi < kXi4; ++xi) {
                    v[xi * vstride + ((g / kNr) * c.ic + k) * kNr + g % kNr] = 0.0f;
                }
            }
        }
    }
}

void compute_small4(const ggml_compute_params * params, const Conv & c, const Plan4_small & pl, const Epilogue & chain) {
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    float * inputs = reinterpret_cast<float *>(base);
    char * mine = base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    float * widened = reinterpret_cast<float *>(mine);
    float * weights = reinterpret_cast<float *>(mine + pl.widened_bytes);
    float * scratch = reinterpret_cast<float *>(mine + pl.widened_bytes + pl.weight_bytes);
    float * products = reinterpret_cast<float *>(mine + pl.widened_bytes + pl.weight_bytes + pl.scratch_bytes);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t tiles = pl.tw * pl.th;
    const std::int64_t nt = pl.panels * kNr;
    const std::int64_t vstride = inputs_stride(pl.panels, c.ic);
    const auto pstride = static_cast<std::size_t>(products_stride(pl.mc, nt));
    const Plan4 layout = {pl.tw, pl.th, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // The kernels for AVX2 take rows of tiles of eights, in their shuffled
    // order, on any processor with AVX2: at 32 by 32, rows are of eight tiles.
    const bool shuffle = avx2_kernels() && pl.tw % 8 == 0;

    constexpr std::int64_t kInputChannels = 16;
    Work_queue queue(params);
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
                    transform_weights4(weights + (ii - i0) * kc, stride, c, ii, std::min<std::int64_t>(kMr, i1 - ii), k0, k0 + kc,
                                       widened);
                }
            }
            const Phase_scope scope(Phase::product);
            // The weights of the next stretch of input channels fetched into
            // the cache while the products of this one are computed, as
            // compute_small() fetches them: a 36th of the rows at each xi.
            const std::int64_t next0 = k0 + kc;
            const std::int64_t next1 = tuning().prefetch != 0 ? std::min(c.ic, next0 + pl.kc) : next0;
            const std::int64_t width = 9 * static_cast<std::int64_t>(ggml_type_size(c.wtype));
            const std::int64_t rows_per_xi = ceil_div(i1 - i0, kXi4);
            for (int xi = 0; xi < kXi4; ++xi) {
                for (std::int64_t r = i0 + xi * rows_per_xi; next0 < next1 && r < std::min(i1, i0 + (xi + 1) * rows_per_xi); ++r) {
                    const auto line = [&](std::int64_t channel) {
                        return reinterpret_cast<std::uintptr_t>(c.w + r * c.wnb3 + channel * width) / 64 * 64;
                    };
                    for (std::uintptr_t q = line(next0); q < line(next1 - 1) + 64; q += 64) {
                        __builtin_prefetch(reinterpret_cast<const char *>(q), 0, 2);
                    }
                }
                const float * u = weights + static_cast<std::size_t>(xi) * static_cast<std::size_t>(stride);
                float * product = products + static_cast<std::size_t>(xi) * pstride;
                for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                    const float * a = u + (ii - i0) * kc;
                    for (std::int64_t p = 0; p < pl.panels; p += panel_step(wide) / kNr) {
                        const float * b = inputs + xi * vstride + (p * c.ic + k0) * kNr;
                        float * out = product + (ii - i0) * nt + p * kNr;
                        sdasm::tiles(uk, wide, kc, a, b, c.ic * kNr * static_cast<std::int64_t>(sizeof(float)), out,
                                     nt * static_cast<std::int64_t>(sizeof(float)), k0 > 0,
                                     std::min<std::int64_t>(kMr, i1 - ii), std::min(panel_step(wide), nt - p * kNr));
                    }
                }
            }
        }
        for (std::int64_t n = 0; n < c.n; ++n) {
            transform_outputs4(c, layout, products + n * tiles, pstride, nt, n, 0, tiles, i0, i1, chain, shuffle);
        }
    }
}

}  // namespace

bool winograd_supported(const ggml_tensor * node) {
    if (node->op != GGML_OP_CONV_2D || tuning().winograd == 0) {
        return false;
    }
    const ggml_tensor * kernel = node->src[0];
    const ggml_tensor * input = node->src[1];
    const int * p = node->op_params;
    return kernel->ne[0] == 3 && kernel->ne[1] == 3 && p[0] == 1 && p[1] == 1 && p[4] == 1 && p[5] == 1 &&
           p[2] >= 0 && p[2] <= 2 && p[3] >= 0 && p[3] <= 2 &&
           (kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32) && ggml_is_contiguous(kernel) &&
           input->type == GGML_TYPE_F32 && input->nb[0] == sizeof(float) && node->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(node) && kernel->ne[2] == input->ne[2] &&
           (node->ne[0] * node->ne[1] >= tuning().winograd_positions || small(geometry(node)));
}

std::size_t winograd_work_size(const ggml_tensor * node, int n_threads) {
    const Conv c = geometry(node);
    if (four(c)) {
        if (small4(c)) {
            const Plan4_small pl = plan4_small(c);
            return kAlign + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(n_threads);
        }
        const Plan4 pl = plan4(c, n_threads);
        return kAlign + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(n_threads);
    }
    if (on_amx(c)) {
        const Amx_plan pl = amx_plan(c, n_threads);
        return kAlign + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(n_threads);
    }
    const Plan pl = plan(c, n_threads);
    return kAlign + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(n_threads);
}

void winograd_compute(const ggml_compute_params * params, ggml_tensor * node, const Epilogue & chain) {
    Conv c = geometry(node);
    if (chain.redirect != nullptr) {
        // The residual connection's tensor, of the same layout (fusion.h).
        c.y = static_cast<char *>(chain.redirect->data);
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
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    // The transformed weights: the store's, transformed here the first time,
    // or the work buffer's, transformed here every time.
    const std::shared_ptr<Packed> kept = packed(c.kernel, Layout::winograd, pl.shared_bytes);
    float * weights = kept != nullptr ? kept->data : reinterpret_cast<float *>(base);
    const bool fill = kept == nullptr || !kept->ready();
    char * mine = base + pl.shared_bytes + pl.thread_bytes * static_cast<std::size_t>(params->ith);
    float * panels = reinterpret_cast<float *>(mine);
    float * products = reinterpret_cast<float *>(mine + pl.panel_bytes);
    const Microkernel uk = microkernel();
    const Microkernel_wide wide = microkernel_wide();
    const std::int64_t step = panel_step(wide);
    const auto per_xi = static_cast<std::size_t>(xi_stride(round_up(c.oc, kMr), c.ic));
    // The kernels written by hand for AVX2 pack the tiles of each eight in
    // shuffled lanes, and transform the outputs from them; those for AVX-512,
    // like the C++ ones, keep the tiles in their order.
    const bool shuffle = avx2_kernels() && !avx512_kernels();

    Work_queue queue(params);
    // The weights, two panels of kMr output channels to an item, widened in
    // the thread's panels of inputs, which it has not begun to fill.
    constexpr std::int64_t kPanels = 2;
    const std::int64_t pack_items = fill ? ceil_div(ceil_div(c.oc, kMr), kPanels) : 0;
    int item = queue.next();
    if (fill) {
        for (; item < pack_items; item = queue.next()) {
            const Phase_scope scope(Phase::pack_weights);
            for (std::int64_t panel = item * kPanels; panel < std::min((item + 1) * kPanels, ceil_div(c.oc, kMr)); ++panel) {
                transform_weights(weights + panel * c.ic * kMr, static_cast<std::int64_t>(per_xi), c, panel * kMr,
                                  std::min<std::int64_t>(kMr, c.oc - panel * kMr), 0, c.ic, panels);
            }
        }
        queue.barrier();
        if (params->ith == 0) {
            publish(kept);
        }
    }

    const std::int64_t tiles = c.tw * c.th;
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
        const std::int64_t ldc = nt * static_cast<std::int64_t>(sizeof(float));
        // The sixteen products, [xi][channel][tile]: the four of a row r of
        // B^T a stretch of input channels at a time, from the panels of their
        // inputs, packed together.
        const std::int64_t stride = static_cast<std::int64_t>(pl.panel_bytes / 4 / sizeof(float)) / kSkew * kSkew;
        for (int r = 0; r < 4; ++r) {
            for (std::int64_t k0 = 0; k0 < c.ic; k0 += pl.kc) {
                const std::int64_t kc = std::min(pl.kc, c.ic - k0);
                {
                    const Phase_scope scope(Phase::pack_inputs);
                    pack_inputs(panels, stride, c, n, t0, t1, k0, k0 + kc, r, shuffle);
                }
                const Phase_scope scope(Phase::product);
                for (int s = 0; s < 4; ++s) {
                    const int xi = 4 * r + s;
                    float * product = products + static_cast<std::size_t>(xi) * static_cast<std::size_t>(products_stride(pl.mc, nt));
                    const float * u_xi = weights + static_cast<std::size_t>(xi) * per_xi;
                    for (std::int64_t jj = 0; jj < nt; jj += step) {
                        const float * b = panels + s * stride + (jj / kNr) * kc * kNr;
                        for (std::int64_t ii = i0; ii < i1; ii += kMr) {
                            const float * a = u_xi + (ii / kMr * c.ic + k0) * kMr;
                            float * out = product + (ii - i0) * nt + jj;
                            sdasm::tiles(uk, wide, kc, a, b, kc * kNr * static_cast<std::int64_t>(sizeof(float)), out, ldc,
                                         k0 > 0, std::min<std::int64_t>(kMr, i1 - ii), std::min(step, nt - jj));
                        }
                    }
                }
            }
        }
        transform_outputs(c, products, static_cast<std::size_t>(products_stride(pl.mc, nt)), nt, n, t0, t1, i0, i1, chain,
                          shuffle);
    }
}

}  // namespace sdasm
