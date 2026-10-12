// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The matrix product that the convolutions and the matrix products of the CPU
// backend are computed by: C = A B, of single-precision numbers, every product
// summed in single precision.
//
// A is m by k and B is k by n, and the product is computed a tile of
// kMr rows by kNr columns at a time by the micro-kernel, which keeps the tile in
// registers through a stretch of k: kMr numbers of A, one of each row, broadcast
// across a vector, against kNr numbers of B, one row of B in two vectors. The
// micro-kernel reads both from packed panels, which lay out a stretch of k of
// kMr rows of A, or of kNr columns of B, in the order it reads them, so that it
// reads memory in one stream for each: a panel of A is [k][kMr], one of B
// [k][kNr]. The operand that is a weight of the model is packed once per
// operation, by all threads together, over the whole of k; the other is packed
// by each thread for the block it works on, a stretch of k at a time, from the
// layout it has in the graph.
//
// The work is shared out by blocks of C, which the threads take one after
// another as each finishes the last; a processor whose cores are not all alike
// (the i7-1360P here has four that each do 2.7 times the work of each of the
// other eight) keeps them all busy that way, where a split fixed in advance
// leaves the fast cores waiting for the slow.

#ifndef SDASM_GEMM_H
#define SDASM_GEMM_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "memory.h"

struct ggml_compute_params;

namespace sdasm {

inline constexpr int kMr = 6;
inline constexpr int kNr = 16;

// c[0..kMr)[0..kNr) = (accumulate ? c : 0) + a b, where a is a panel [kc][kMr]
// and b a panel [kc][kNr], 32-byte aligned, and the rows of c lie ldc bytes
// apart. kc is at least 1.
using Microkernel = void (*)(std::int64_t kc, const float * a, const float * b, float * c,
                             std::int64_t ldc, std::int64_t accumulate);

// The micro-kernel that the products are computed with (cpu.cc).
Microkernel microkernel();

// A micro-kernel of a tile of kMr rows by two panels of B, 2 * kNr columns,
// the second panel b_next bytes after the first: for processors whose loads,
// not their multiplications, hold back the tile of one panel
// (x86_64/gemm_avx512.S). None where there is none, or the tuning says not
// to take it (Tuning::wide), which takes it by default on every processor with
// AVX-512. On GitHub's runners (the workflow Measure), against the tile of one
// panel, it made the linear layers and the convolutions of Ice Lake-SP 18 to
// 60% faster and those of Zen 5 8 to 21%, and those of Zen 4, whose units take
// the 512 bits in two halves, no faster and no slower (-2 to +3%).
using Microkernel_wide = void (*)(std::int64_t kc, const float * a, const float * b, std::int64_t b_next, float * c,
                                  std::int64_t ldc, std::int64_t accumulate);

Microkernel_wide microkernel_wide();

// The columns of B a step of the loops over its panels takes: two panels with
// the wide micro-kernel, one without.
inline std::int64_t panel_step(Microkernel_wide wide) {
    return wide != nullptr ? 2 * 16 : 16;
}

// The micro-kernel of attention's scores (attention.cc): the tile of kMr keys
// by kNr queries that the micro-kernel computes, added to c when accumulate
// says so, scaled by `scale` and written as the micro-kernel writes a tile;
// and max[0..kNr), each query's largest score so far, raised to the largest of
// the six keys' -- the scale and the maximum that ggml's softmax begins with,
// taken while the tile is in registers.
using Scores_kernel = void (*)(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                               float * max, float scale, std::int64_t accumulate);

Scores_kernel scores_kernel();

// The same of a tile of kMr keys by two panels of queries, 2 * kNr, the second
// b_next bytes after the first, its two halves written to two tiles of the
// scores c_next bytes apart, each rows ldc bytes apart; with bit 0 of flags
// added to them, with bit 1 scaled and max[0..2 kNr) raised, without it a
// partial sum (x86_64/gemm_avx512.S). Where the wide micro-kernel is taken,
// whose loads it shares; none otherwise.
using Scores_kernel_wide = void (*)(std::int64_t kc, const float * a, const float * b, std::int64_t b_next, float * c,
                                    std::int64_t c_next, std::int64_t ldc, float * max, float scale, std::int64_t flags);

Scores_kernel_wide scores_kernel_wide();

// The exponentials of attention's softmax (attention.cc): the scaled scores of
// n keys, n a multiple of eight, for kNr queries, [key][kNr], in place,
// exp(score - max[query]), and their sums added to total[query] in double
// precision.
using Exponentials = void (*)(float * s, std::int64_t n, const float * max, double * total);

Exponentials exponentials_kernel();

// What a product needs of the threads' work buffer: the shared packed weights,
// then a block of each operand for each thread.
struct Gemm_plan {
    std::int64_t mc;  // rows of C in a block, a multiple of kMr
    std::int64_t nc;  // columns of C in a block, a multiple of kNr
    std::int64_t kc;  // the stretch of k of a packed block
    std::size_t shared_bytes;
    std::size_t thread_bytes;
};

inline constexpr std::size_t kAlign = kWorkAlign;

constexpr std::size_t align_up(std::size_t n, std::size_t a = kAlign) {
    return (n + a - 1) / a * a;
}

// The bytes of n elements of T, as the kernels written by hand take strides.
template <class T>
constexpr std::int64_t bytes_of(std::int64_t n) {
    return n * static_cast<std::int64_t>(sizeof(T));
}

// Packs rows [0, rows) of a matrix, src, whose rows of k elements lie `stride`
// elements apart, into panels of P rows, [panel][k1 - k0][P], with the rows
// past the end of the matrix as zeros: P is kMr or kNr. The elements are float
// or FP16 (the bits of an IEEE half), widened to float.
void pack_rows_f32(Elements<float> dst, Elements<const float> src, std::int64_t stride, std::int64_t rows,
                   std::int64_t k0, std::int64_t k1, int p);
void pack_rows_f16(Elements<float> dst, Elements<const std::uint16_t> src, std::int64_t stride, std::int64_t rows,
                   std::int64_t k0, std::int64_t k1, int p);

// The part of a tile that rows and cols say: the micro-kernel's whole tile
// into a buffer of its own, and the part copied out.
void tile_part(Microkernel uk, std::int64_t kc, Elements<const float> a, Elements<const float> b, Elements<float> c,
               std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols);

// Computes one tile of C from two panels, a of kc kMr numbers and b of kc kNr,
// into c, rows ldc elements apart: a kMr by kNr tile when rows and cols say it
// is whole, and the part of one that they say otherwise.
inline void tile(Microkernel uk, std::int64_t kc, Elements<const float> a, Elements<const float> b, Elements<float> c,
                 std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols) {
    if (rows == kMr && cols == kNr) {
        SDASM_CHECK(a.size() >= kc * kMr && b.size() >= kc * kNr && c.size() >= (kMr - 1) * ldc + kNr);
        uk(kc, a.data(), b.data(), c.data(), bytes_of<float>(ldc), accumulate ? 1 : 0);
        return;
    }
    tile_part(uk, kc, a, b, c, ldc, accumulate, rows, cols);
}

// The tiles of a panel of A and of the panels of B at b, cols columns of them,
// at most panel_step(wide): the second panel b_next elements after the first.
// By the wide micro-kernel when there is one and the two panels are whole, and
// a panel at a time by tile() otherwise.
inline void tiles(Microkernel uk, Microkernel_wide wide, std::int64_t kc, Elements<const float> a, Elements<const float> b,
                  std::int64_t b_next, Elements<float> c, std::int64_t ldc, bool accumulate, std::int64_t rows,
                  std::int64_t cols) {
    if (wide != nullptr && rows == kMr && cols == 2 * kNr) {
        SDASM_CHECK(a.size() >= kc * kMr && b.size() >= b_next + kc * kNr && c.size() >= (kMr - 1) * ldc + 2 * kNr);
        wide(kc, a.data(), b.data(), bytes_of<float>(b_next), c.data(), bytes_of<float>(ldc), accumulate ? 1 : 0);
        return;
    }
    tile(uk, kc, a, b, c, ldc, accumulate, rows, cols < kNr ? cols : kNr);
    if (cols > kNr) {
        tile(uk, kc, a, b.from(b_next), c.from(kNr), ldc, accumulate, rows, cols - kNr);
    }
}

// The products of numbers split in parts, for the operations whose weights are
// FP16 (x86_64/amx.S and amx_fp16.S, which give their precision): A and B in
// tiles -- A's of sixteen rows by 32 of k, B's of sixteen pairs of k by sixteen
// columns -- and C in single precision, a block of 32 by 32 at a time. The
// forms:
//
//  - AMX-BF16's: every number of either operand in two parts of bfloat16,
//    three products for each one; kAmxTileBytes a block of k of a tile;
//  - AMX-FP16's, where the processor has it (Granite Rapids and after;
//    Tuning::amx_fp16): the FP16 weights as they are, 1024 bytes a block, and
//    the other operand in two parts of FP16, two products for each one;
//  - AVX512-BF16's vdpbf16ps on the vector units, AMX-BF16's layout, at the
//    level avx512 when Tuning::bf16 asks for it, for its measurement.
//
// Winograd's products, of two operands of single precision, take two parts of
// bfloat16 in any form.
inline constexpr std::int64_t kAmxTile = 16;
inline constexpr std::int64_t kAmxBlock = 32;
inline constexpr std::int64_t kAmxTileBytes = 2048;

using Split_kernel = void (*)(std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next, const std::uint16_t * b,
                              std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);

// What the products are of: the linear layers' (the weights B), the direct
// convolutions' (the weights A), Winograd's (both of single precision).
enum class Split_use { linear, convolution, winograd };

struct Split_form {
    Split_kernel kernel = nullptr;  // none: the products are of single precision
    bool tiles = false;             // on AMX's tiles, configured before and released after
    bool fp16 = false;              // AMX-FP16's form
    std::int64_t a_bytes = kAmxTileBytes;  // a block of k of a tile of A, every part
    std::int64_t b_bytes = kAmxTileBytes;
    // A from rows of single precision (the linear layers' src1)
    void (*pack_rows)(std::uint16_t *, const float *, std::int64_t, std::int64_t, std::int64_t) = nullptr;
    // B from rows of FP16 weights, a row for each column (the linear layers')
    void (*pack_weights)(std::uint16_t *, const std::uint16_t *, std::int64_t, std::int64_t, std::int64_t) = nullptr;
    // A from rows of FP16 weights (the convolutions')
    void (*pack_weight_rows)(std::uint16_t *, const std::uint16_t *, std::int64_t, std::int64_t, std::int64_t) = nullptr;
    // B from a panel [k][16] of single precision (the convolutions' columns)
    void (*pack_panel)(std::uint16_t *, const float *, std::int64_t) = nullptr;
};

Split_form split_form(Split_use use);

// The thread's tiles configured for the form, when it is on them, before its
// first block; and given back after its last.
void split_begin(const Split_form & form);
void split_end(const Split_form & form);

// A block of 32 by 32 of C over `blocks` blocks of k, from the tiles of A at a
// (those of its rows 16 to 31 a_next elements after) and of B at b (its
// columns 16 to 31 b_next elements after): straight into c, rows ldc elements
// apart, when the block is whole, rows by cols, and through `spare`, 32 by 32,
// when it reaches past C; added to C when accumulate says so.
void split_block(const Split_form & form, std::int64_t blocks, Elements<const std::uint16_t> a, std::int64_t a_next,
                 Elements<const std::uint16_t> b, std::int64_t b_next, Elements<float> c, std::int64_t ldc,
                 bool accumulate, std::int64_t rows, std::int64_t cols, Elements<float> spare);

// A stretch of one of several ranges of the same length -- the positions of
// one image of a batch, say -- that a block of work covers.
struct Block {
    std::int64_t segment;
    std::int64_t start;
    std::int64_t size;
};

// Splits `segments` ranges of `length` each into blocks of sizes that are
// multiples of `align` (but for the last of a range), at most `largest`, and
// guided (Polychronopoulos and Kuck, 1987): once what is left of all ranges is
// less than enough for every thread to take two blocks of the largest size,
// each block is half of what is left shared among the threads, so that the
// blocks taken last are short and the threads finish close together, whichever
// cores they run on. The blocks are in the order the work items take them,
// range after range.
std::vector<Block> guided_blocks(std::int64_t segments, std::int64_t length, std::int64_t align, std::int64_t largest,
                                 int threads);

// The work items of an operation, numbered from 0, which the threads take one
// at a time as each finishes the last. Every thread constructs the queue on
// entry to the operation, which holds them all until the count is ready; the
// first item each takes is its own thread number, and the items after are
// handed out by a counter that ggml's thread pool keeps for its own kernels.
// An operation of two phases numbers the items of both in one sequence and has
// the threads meet at barrier() between them: an item a thread takes past the
// end of the first phase is the first it does in the second. The threads first
// meet in the constructor, and the arrays of the operation's work buffer, which
// each thread began the lifetimes of before, are theirs to reach after it
// (memory.h).
class Work_queue {
public:
    explicit Work_queue(const ggml_compute_params * const params [[clang::lifetimebound]]);
    Work_queue(const ggml_compute_params * const params [[clang::lifetimebound]], Work_buffer & buffer);
    int next();
    void barrier() const;

private:
    // The first meeting, where the first thread sets the counter.
    void meet() const;

    const ggml_compute_params * params_;
    bool first_ = true;
};

}  // namespace sdasm

#endif
