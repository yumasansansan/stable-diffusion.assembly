// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The products of numbers quantized to integers, for the operations whose
// precision the run gives up for speed (Tuning::bits): every number of a
// product's two operands rounded to an integer of few bits times a scale, the
// products of the integers summed exactly in 32 bits, as VNNI's vpdpbusd and
// AMX's tdpbusd sum them -- four products of a byte by a byte to a lane -- and
// the sum scaled back to single precision.
//
// One operand is a weight of the model, or what stands for one (attention's
// keys and values); the other the input the weight is applied to. The weights
// are signed and symmetric: w = s q, |q| <= 2^(b - 1) - 1. The inputs are
// unsigned with a zero point, the range of each group of them from its least
// to its greatest, 0 within it: x = s (q - z), 0 <= q <= 2^b - 1 -- the
// activations after SiLU and GELU lie mostly on one side of 0, and the
// transformed inputs of Winograd's convolution have ranges that a symmetric
// scale wastes half of. Of SD 1.5 quantized so, the image after twenty steps
// was 5 dB of PSNR closer to the exact one than with symmetric scales, of the
// linear layers and of Winograd's convolutions alike.
//
// A scale holds for a group of k -- `group` numbers of the sum, a row's or a
// column's -- so that one large input does not cost every other input of its
// row its precision: measured on SD 1.5's activations, the outputs of its
// linear layers are 1.6 times as far from the exact ones with a scale for the
// whole of a row, a token, as with groups of 64, and 1.2 times as near with
// groups of 32, whose products take a sixth longer. The micro-kernel sums the
// products of a group in integers and adds them to its tile in single
// precision, scaled:
//
//     c[i][j] += sa[i] sb[j] sum(qa qb) - oa[i] ob[j],
//
// sa and sb the scales of row i and column j of the group, and of the two
// offsets one the input's s z, the other the weight's s sum(q) over the group:
// s_x s_w sum((q_x - z) q_w) = s_x s_w sum(q_x q_w) - (s_x z)(s_w sum(q_w)).
//
// A panel holds kMr rows of A or kNr columns of B, a group at a time: the
// group's bytes, then its P scales and P offsets, floats. A's bytes lie row by
// row, [kMr][4 g] -- a quad of a row is what vpdpbusd broadcasts, and the
// quantization of a row of the inputs stores its group in one piece -- and B's
// quad by quad, [g][kNr][4], the four bytes of each column of a quad side by
// side, a vector of vpdpbusd.

#ifndef SDASM_QUANT_H
#define SDASM_QUANT_H

#include <cstddef>
#include <cstdint>

#include "gemm.h"

namespace sdasm {

// What the tuning asks of the quantized products.
struct Quant {
    std::int64_t quads;    // the quads of k of a group: group / 4
    int input_levels;      // the inputs' greatest integer, 2^input_bits - 1
    int weight_max;        // the weights' greatest magnitude, 2^(bits - 1) - 1
    bool weight_groups;    // the weights' scale for each group, or for a whole row of k
};

// Whether the products of a kind of operation are quantized (Tuning::bits and
// Tuning::quantize), and how.
enum class Quantized_op { linear = 1, convolution = 2, attention = 4 };
bool quantized(Quantized_op op);
Quant quant();

// Whether the kernels in effect have quantized products faster than those of
// single precision: the C++ ones, the reference, always; those written by
// hand, where the level in effect has VNNI (AVX-VNNI, AVX512-VNNI), or, on
// AVX2 without it, where the integers are narrow enough for vpmaddubsw's sums
// of two products in a word -- seven bits of the weights with eight of the
// inputs, or eight with seven -- which with vpmaddwd make twice the products
// of the FMAs where the processor has two multipliers of vector integers, as
// many where it has one (x86_64/quant_maddubs.S). Otherwise bits asks for
// nothing.
bool quantized_kernels();

// The bytes of a group of a panel of P rows or columns, and of a panel over k.
constexpr std::int64_t qgroup_bytes(std::int64_t quads, int p) {
    return quads * p * 4 + 2 * p * static_cast<std::int64_t>(sizeof(float));
}
inline std::int64_t qgroups(std::int64_t k, const Quant & q) {
    return (k + 4 * q.quads - 1) / (4 * q.quads);
}
inline std::int64_t qpanel_bytes(std::int64_t k, int p, const Quant & q) {
    return qgroups(k, q) * qgroup_bytes(q.quads, p);
}

// The micro-kernel: the tile of kMr rows by kNr columns of C, rows ldc bytes
// apart, = (accumulate ? c : 0) + the sum over `groups` groups, one or more, of
// `quads` quads of k of the panels a and b, as above. In the form a_unsigned, A
// holds the inputs and B the weights (the linear layers); in b_unsigned, the
// other way about (the convolutions, attention).
using Qkernel = void (*)(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b, float * c,
                         std::int64_t ldc, std::int64_t accumulate);

Qkernel qkernel_a_unsigned();
Qkernel qkernel_b_unsigned();

// A tile of kMr rows by two panels of B, 2 * kNr columns, the second b_next
// bytes after the first, for the processors whose loads hold back the tile of
// one panel (x86_64/quant_avx512.S); none where there is none.
using Qkernel_wide = void (*)(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                              std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);

Qkernel_wide qkernel_wide_a_unsigned();
Qkernel_wide qkernel_wide_b_unsigned();

// A block of 32 rows by 32 columns of C on AMX-INT8's tiles, rows ldc bytes
// apart, = (accumulate ? c : 0) + the sum over `groups` groups, one or more, of
// `quads` quads of k of the panels of A, the inputs, of sixteen rows -- those
// of rows 16 to 31 a_next bytes after those of rows 0 to 15 -- and of B, the
// weights, of sixteen columns, those of columns 16 to 31 b_next bytes after
// (x86_64/amx_int8.S). For groups of a multiple of 64 numbers of k, where the
// tiles are the level in effect and the processor has AMX-INT8; none otherwise.
// qamx_begin() configures the thread's tiles for it, and qamx_end() gives them
// back.
using Qkernel_amx = void (*)(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, std::int64_t a_next,
                             const std::uint8_t * b, std::int64_t b_next, float * c, std::int64_t ldc,
                             std::int64_t accumulate);

Qkernel_amx qkernel_amx_a_unsigned(const Quant & q);
void qamx_begin();
void qamx_end();

// The block of rows by cols of C at c, rows ldc elements apart, by the kernel
// on the tiles: straight into c when it is whole, 32 by 32, and through
// `spare`, 32 by 32, of which the part is copied, when it reaches past C.
void qblock_amx(Qkernel_amx kernel, std::int64_t groups, std::int64_t quads, Elements<const std::uint8_t> a,
                std::int64_t a_next, Elements<const std::uint8_t> b, std::int64_t b_next, Elements<float> c,
                std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols, Elements<float> spare);

void qgemm_6x16_au_cxx(std::int64_t groups, std::int64_t quads, const std::uint8_t * const a,
                       const std::uint8_t * const b, float * const c, std::int64_t ldc, std::int64_t accumulate);
void qgemm_6x16_bu_cxx(std::int64_t groups, std::int64_t quads, const std::uint8_t * const a,
                       const std::uint8_t * const b, float * const c, std::int64_t ldc, std::int64_t accumulate);

// A tile's part, rows by cols of it, through a whole tile of the kernel's own.
// The panels are of `groups` groups; c's rows lie ldc elements apart.
void qtile_part(Qkernel kernel, std::int64_t groups, std::int64_t quads, Elements<const std::uint8_t> a,
                Elements<const std::uint8_t> b, Elements<float> c, std::int64_t ldc, bool accumulate, std::int64_t rows,
                std::int64_t cols);

inline void qtile(Qkernel kernel, std::int64_t groups, std::int64_t quads, Elements<const std::uint8_t> a,
                  Elements<const std::uint8_t> b, Elements<float> c, std::int64_t ldc, bool accumulate, std::int64_t rows,
                  std::int64_t cols) {
    if (rows == kMr && cols == kNr) {
        SDASM_CHECK(a.size() >= groups * qgroup_bytes(quads, kMr) && b.size() >= groups * qgroup_bytes(quads, kNr) &&
                    c.size() >= (kMr - 1) * ldc + kNr);
        kernel(groups, quads, a.data(), b.data(), c.data(), bytes_of<float>(ldc), accumulate ? 1 : 0);
        return;
    }
    qtile_part(kernel, groups, quads, a, b, c, ldc, accumulate, rows, cols);
}

// The tiles of a panel of A and of the panels of B at b, cols columns of them,
// at most 2 * kNr with the wide kernel and kNr without, the second panel b_next
// bytes after the first: by the wide kernel when the two panels are whole, a
// panel at a time otherwise.
inline void qtiles(Qkernel kernel, Qkernel_wide wide, std::int64_t groups, std::int64_t quads,
                   Elements<const std::uint8_t> a, Elements<const std::uint8_t> b, std::int64_t b_next, Elements<float> c,
                   std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols) {
    if (wide != nullptr && rows == kMr && cols == 2 * kNr) {
        SDASM_CHECK(a.size() >= groups * qgroup_bytes(quads, kMr) &&
                    b.size() >= b_next + groups * qgroup_bytes(quads, kNr) && c.size() >= (kMr - 1) * ldc + 2 * kNr);
        wide(groups, quads, a.data(), b.data(), b_next, c.data(), bytes_of<float>(ldc), accumulate ? 1 : 0);
        return;
    }
    qtile(kernel, groups, quads, a, b, c, ldc, accumulate, rows, cols < kNr ? cols : kNr);
    if (cols > kNr) {
        qtile(kernel, groups, quads, a, b.from(b_next), c.from(kNr), ldc, accumulate, rows, cols - kNr);
    }
}

// The layout of a quantized panel: A's, row by row in each group, or B's,
// quad by quad.
enum class Qlayout { rows, columns };

// Packs `rows` rows of k inputs, src, rows `stride` elements apart, into a
// panel of p rows or columns -- kMr rows of A, kNr columns of B, or the sixteen
// rows of a tile of AMX -- unsigned with a zero point; those past them, and k
// past its end, zeros.
void quantize_inputs(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t stride, std::int64_t rows,
                     std::int64_t k, int p, Qlayout layout, const Quant & q);

// Packs `rows` rows of k weights, FP16 or single precision, into a panel of p
// rows or columns, signed and symmetric.
void quantize_weights_f16(Elements<std::uint8_t> dst, Elements<const std::uint16_t> src, std::int64_t stride,
                          std::int64_t rows, std::int64_t k, int p, Qlayout layout, const Quant & q);
void quantize_weights_f32(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t stride, std::int64_t rows,
                          std::int64_t k, int p, Qlayout layout, const Quant & q);

// Quantizes a panel of single precision as the kernels of single precision
// lay it out, [k][p] -- a stretch of k of p rows of A or columns of B, k a
// multiple of the group or the panel's end -- into a quantized panel of p:
// the inputs unsigned with a zero point, the weights signed, each lane's scale
// for each group of k.
void quantize_panel_inputs(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t k, int p, Qlayout layout,
                           const Quant & q);
void quantize_panel_weights(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t k, int p, Qlayout layout,
                            const Quant & q);

// The scales and the offsets of a quantized panel's group, P floats each, which
// lie among its bytes, after the group's integers (the order above): read and
// written as the bytes they are.
void read_floats(Elements<const std::uint8_t> bytes, Elements<float> values);
void write_floats(Elements<std::uint8_t> bytes, Elements<const float> values);

}  // namespace sdasm

#endif
