// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// Attention, GGML_OP_FLASH_ATTN_EXT without a mask, computed as
// stable-diffusion.cpp computes it when it does not ask for flash attention:
// the scores of a head are the products of its queries and keys, scaled; their
// softmax over the keys, with the sum of the exponentials in double precision,
// gives each query its weights; the output is the sum of the values
// by those weights. stable-diffusion.cpp does each of these as an operation of
// its own, through a matrix of scores of every query against every key, a
// gibibyte for the heads of SD 1.5's first layer at 512 by 512, written to
// memory and read back by each operation after. Here the scores of a block of
// queries are all that is ever there, in the cache of the thread that works on
// them, from their product to the weights; and the softmax is ggml's, two
// passes over the scores rather than the online softmax of flash attention,
// whose rescaling as the maximum moves rounds what this does not: the scaled
// scores and their maximum are ggml's, their exponentials (exponential()) a
// little closer to the exact ones than ggml's, and their sum is in double
// precision as ggml's is. The division by that sum comes at the end, once for
// each output rather than once for each weight: the sum of the values by the
// exponentials, divided by their sum, is the sum of the values by the weights,
// with a rounding fewer in each term.
//
// The products are matrix products (gemm.h) with the scores transposed, keys by
// queries: the keys of a head are packed once, by all threads, into panels of
// A, kMr keys over the dimensions of the head, and its values into panels of
// kMr dimensions over the keys; a block of queries is packed by the thread that
// takes it into panels of B, kNr queries over the dimensions. The micro-kernel
// of the scores (scores_kernel()) then writes the scores of kMr keys for kNr
// queries, scaled, and keeps the largest of each query; they are a stretch of a
// panel of B of the second product, kNr queries over the keys, a query to a
// lane. That product goes a stretch of keys at a time, the exponentials of the
// stretch taken just before the values are summed by them, while they are in
// the nearest cache. Its output is the transposed output of the block,
// dimensions by queries, which is divided by the sums and written out a query
// at a time. Where the wide micro-kernel is taken (gemm.h: AVX-512's tile of
// 32 columns, whose loads keep up with its multiplications), both products
// take two panels of queries at a time, the scores' kernel writing its two
// halves to the two panels (scores_kernel_wide()).
//
// The sums are kept short, as ggml's dot product keeps them -- it sums 32 at a
// time and then the 32 sums -- so that they round no more than its own do: a
// score is summed over at most attention_sum_dims dimensions in the registers,
// and the sums of the stretches added in turn; an output over at most
// attention_sum_keys keys in the registers, those sums added over a stretch of
// the second product, and the stretches' sums added to the output. Measured by
// tests/ops_test.cc against the exact result, the outputs are then as close to
// it as those of stable-diffusion.cpp's graph of operations in ggml's kernels,
// closer for SD 1.5's heads of 40 dimensions; summed in the registers over a
// stretch of 256 keys, as at first, they were up to six times further for
// heads of 160. It costs no time that the profile of SD 1.5 can tell.
//
// On AMX's tiles (x86_64/amx_single.S), the products are in single precision
// too, each number split in three parts of bfloat16 and each product taken as
// six; and the queries are the rows: the scores of a block of queries,
// queries by keys, are Q K^T -- A the queries' tiles, B the keys' -- and its
// output, queries by dimensions, is P V -- A the tiles of the exponentials, B
// the values'. The keys' and the values' tiles are packed once for each head,
// by all threads together; a block's queries and its exponentials by the
// thread that takes it. A tile sums 32 products, two chains of sixteen, before
// it adds them to its sums, as ggml's dot product sums 32 at a time.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "gemm.h"
#include "kernels.h"
#include "memory.h"
#include "ops.h"
#include "phases.h"
#include "quant.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"
#include "scalar_math.h"

namespace sdasm {

namespace {

// The strides are in elements of each tensor's own type.
struct Attention {
    Elements<const float> q;  // [d, lq, h, b]
    std::int64_t qnb1, qnb2, qnb3;
    ggml_type ktype;
    Elements<const float> k32;          // [d, lk, hk, b], of single precision
    Elements<const std::uint16_t> k16;  // or FP16
    std::int64_t knb1, knb2, knb3;
    ggml_type vtype;
    Elements<const float> v32;  // [dv, lk, hk, b]
    Elements<const std::uint16_t> v16;
    std::int64_t vnb1, vnb2, vnb3;
    Elements<float> out;  // [dv, h, lq, b]
    std::int64_t onb1, onb2, onb3;
    std::int64_t d, dv, lq, lk, h, hk, b;
    float scale;
    std::int64_t queries;    // queries in a block, a multiple of kNr
    std::int64_t key_block;  // keys in a stretch of the second product, a multiple of eight
    std::int64_t sum_keys;   // the most keys the micro-kernel sums an output over at once
    std::int64_t sum_dims;   // the most dimensions it sums a score over at once
};

// Of a node that attention_supported() takes.
Attention geometry(const ggml_tensor * const node) {
    const ggml_tensor * const q = node->src[0];
    const ggml_tensor * const k = node->src[1];
    const ggml_tensor * const v = node->src[2];
    Attention a{};
    a.q = elements_of<const float>(q);
    a.qnb1 = stride_of<float>(q, 1);
    a.qnb2 = stride_of<float>(q, 2);
    a.qnb3 = stride_of<float>(q, 3);
    a.ktype = k->type;
    if (k->type == GGML_TYPE_F16) {
        a.k16 = elements_of<const std::uint16_t>(k);
        a.knb1 = stride_of<std::uint16_t>(k, 1);
        a.knb2 = stride_of<std::uint16_t>(k, 2);
        a.knb3 = stride_of<std::uint16_t>(k, 3);
    }
    else {
        a.k32 = elements_of<const float>(k);
        a.knb1 = stride_of<float>(k, 1);
        a.knb2 = stride_of<float>(k, 2);
        a.knb3 = stride_of<float>(k, 3);
    }
    a.vtype = v->type;
    if (v->type == GGML_TYPE_F16) {
        a.v16 = elements_of<const std::uint16_t>(v);
        a.vnb1 = stride_of<std::uint16_t>(v, 1);
        a.vnb2 = stride_of<std::uint16_t>(v, 2);
        a.vnb3 = stride_of<std::uint16_t>(v, 3);
    }
    else {
        a.v32 = elements_of<const float>(v);
        a.vnb1 = stride_of<float>(v, 1);
        a.vnb2 = stride_of<float>(v, 2);
        a.vnb3 = stride_of<float>(v, 3);
    }
    a.out = elements_of<float>(node);
    a.onb1 = stride_of<float>(node, 1);
    a.onb2 = stride_of<float>(node, 2);
    a.onb3 = stride_of<float>(node, 3);
    a.d = q->ne[0];
    a.dv = v->ne[0];
    a.lq = q->ne[1];
    a.lk = k->ne[1];
    a.h = q->ne[2];
    a.hk = k->ne[2];
    a.b = q->ne[3];
    a.scale = std::bit_cast<float>(node->op_params[0]);
    a.queries = std::max<std::int64_t>(kNr, tuning().attention_queries / kNr * kNr);
    a.key_block = std::max(8, tuning().attention_keys / 8 * 8);
    a.sum_keys = std::max(1, tuning().attention_sum_keys);
    a.sum_dims = std::max(1, tuning().attention_sum_dims);
    return a;
}

std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

std::int64_t round_up(std::int64_t a, std::int64_t b) {
    return ceil_div(a, b) * b;
}

// The first element of the keys, and of the values, of head hk of image b.
std::int64_t keys_at(const Attention & a, std::int64_t b, std::int64_t hk) {
    return b * a.knb3 + hk * a.knb2;
}
std::int64_t values_at(const Attention & a, std::int64_t b, std::int64_t hk) {
    return b * a.vnb3 + hk * a.vnb2;
}

// The number of a value, widened.
float value(const Attention & a, std::int64_t at) {
    return a.vtype == GGML_TYPE_F16 ? static_cast<float>(std::bit_cast<_Float16>(a.v16[at])) : a.v32[at];
}

// Keys [r0, r0 + rows) of head hk of image b over the head's dimensions into
// panels of p (gemm.h).
void pack_keys(Elements<float> dst, const Attention & a, std::int64_t b, std::int64_t hk, std::int64_t r0, std::int64_t rows,
               int p) {
    const std::int64_t at = keys_at(a, b, hk) + r0 * a.knb1;
    if (a.ktype == GGML_TYPE_F16) {
        pack_rows_f16(dst, a.k16.from(at), a.knb1, rows, 0, a.d, p);
    }
    else {
        pack_rows_f32(dst, a.k32.from(at), a.knb1, rows, 0, a.d, p);
    }
}

// Row i of the output of head h of image b: query q of it, dv numbers.
Elements<float> output_row(const Attention & a, std::int64_t b, std::int64_t h, std::int64_t q) {
    return a.out.part(b * a.onb3 + h * a.onb1 + q * a.onb2, a.dv);
}

// The work buffer of attention in single precision: the packed keys and
// values of every head, shared; and each thread's packed queries, its scores,
// its transposed output and the sum of a stretch of keys, the largest score of
// each of its queries and the sum of the exponentials of each, in double
// precision.
struct Layout {
    std::int64_t keys_floats;    // the packed keys of one head
    std::int64_t values_floats;  // the packed values of one head
    Work_layout layout;
    Shared<float> heads;
    Own<float> queries, scores, output, stretch_sum, maxima;
    Own<double> sums;
};

Layout layout(const Attention & a) {
    Layout l;
    l.keys_floats = round_up(a.lk, kMr) * a.d;
    l.values_floats = round_up(a.dv, kMr) * a.lk;
    l.heads = l.layout.shared<float>((l.keys_floats + l.values_floats) * a.hk * a.b);
    l.queries = l.layout.own<float>(a.queries * a.d);
    l.scores = l.layout.own<float>(a.queries * round_up(a.lk, kMr));
    l.output = l.layout.own<float>(round_up(a.dv, kMr) * a.queries);
    l.stretch_sum = l.layout.own<float>(round_up(a.dv, kMr) * a.queries);
    l.maxima = l.layout.own<float>(a.queries);
    l.sums = l.layout.own<double>(a.queries);
    return l;
}

// Packs the values [0, lk) of one head, src from its first, into panels of kMr
// dimensions: [panel][lk][kMr], zeros past the last dimension.
template <class T, class Convert>
void pack_values(Elements<float> dst, Elements<const T> src, std::int64_t stride, std::int64_t lk, std::int64_t dv,
                 Convert convert) {
    for (std::int64_t d0 = 0; d0 < dv; d0 += kMr) {
        const Elements<float> panel = dst.part(d0 / kMr * lk * kMr, lk * kMr);
        const std::int64_t width = std::min<std::int64_t>(kMr, dv - d0);
        for (std::int64_t j = 0; j < lk; ++j) {
            const Elements<const T> row = src.part(j * stride + d0, width);
            const Elements<float> out = panel.part(j * kMr, kMr);
            std::int64_t i = 0;
            for (; i < width; ++i) {
                out[i] = convert(row[i]);
            }
            for (; i < kMr; ++i) {
                out[i] = 0.0f;
            }
        }
    }
}

}  // namespace

// The exponentials of n keys, n a multiple of eight, in place, and their sums
// added to total: the C++ form of sdasm_exponentials_avx2 (x86_64/softmax_avx2.S),
// a loop over the sixteen queries of a key for the compiler to vectorize, each
// query's sum in double precision.
void exponentials_cxx(float * const s_first, std::int64_t n, const float * const max_first,
                      double * const total_first) {
    SDASM_REQUIRE(n >= 0);
    // The kernel's operands, handed over as pointers: n keys of kNr queries.
#pragma clang unsafe_buffer_usage begin
    const Elements<float> s = elements_at(s_first, n * kNr);
    const Elements<const float> max = elements_at(max_first, kNr);
    const Elements<double> total = elements_at(total_first, kNr);
#pragma clang unsafe_buffer_usage end
    for (std::int64_t j = 0; j < n; ++j) {
        const Elements<float> row = s.part(j * kNr, kNr);
        for (std::int64_t l = 0; l < kNr; ++l) {
            const float e = exponential(row[l] - max[l]);
            row[l] = e;
            total[l] += static_cast<double>(e);
        }
    }
}

namespace {

// The exponentials of the scaled scores of keys [j0, j1) of kNr queries, a
// query to a lane, in a panel laid out [key][kNr], in place, a stretch at a
// time, and their sums added to total, in double precision: eight keys at a
// time from j0, a multiple of eight, by the kernel of the form taken; and at
// the end of the keys, those past the last eight by expf, each added to total.
void exponentials(Elements<float> s, std::int64_t j0, std::int64_t j1, Elements<const float> max, Elements<double> total) {
    const std::int64_t whole = (j1 - j0) / 8 * 8;
    SDASM_CHECK(s.size() >= j1 * kNr && max.size() >= kNr && total.size() >= kNr);
    if (whole > 0) {
        exponentials_kernel()(s.from(j0 * kNr).data(), whole, max.data(), total.data());
    }
    for (std::int64_t j = j0 + whole; j < j1; ++j) {
        const Elements<float> row = s.part(j * kNr, kNr);
        for (std::int64_t l = 0; l < kNr; ++l) {
            const float e = std::exp(row[l] - max[l]);
            row[l] = e;
            total[l] += static_cast<double>(e);
        }
    }
}

// ---------------------------------------------------------------- on AMX

constexpr std::int64_t kTileRows = 16;
constexpr std::int64_t kBlock = 32;           // the rows and columns of a block of the kernel, and its k
constexpr std::int64_t kBlockHalves = 1536;   // a block of k of a tile's rows or columns, three parts of two bytes

// On the tiles, or, with products of 16 bits (Tuning::bits), on AVX512-BF16's
// vector units where there are no tiles: the same tiles in memory, the first
// parts of each multiplied alone (x86_64/half.S).
bool half_products() {
    return tuning().bits == 16;
}

bool on_vectors(const Attention &) {
    return half_products() && !amx_kernels() && avx512_kernels() && cpu().avx512_bf16;
}

// In single precision, attention on the tiles takes six products for each one,
// and runs at half the rate of AVX-512's FMAs on Emerald Rapids (measured on
// GitHub's runners): it is taken when Tuning::amx_attention asks. With
// products of 16 bits, one for each, it is taken where the tiles are.
bool on_amx(const Attention & a) {
    return (amx_kernels() && (tuning().amx_attention != 0 || half_products())) || on_vectors(a);
}

using Single_kernel = void (*)(std::int64_t, const std::uint16_t *, std::int64_t, const std::uint16_t *, std::int64_t,
                               float *, std::int64_t, std::int64_t);

Single_kernel single_kernel(const Attention & a) {
#if SDASM_X86_64
    if (on_vectors(a)) {
        return sdasm_bf16_single_1x1_avx512;
    }
    return half_products() ? sdasm_amx_single_1x1 : sdasm_amx_single_2x2;
#else
    (void) a;
    return nullptr;
#endif
}

// The work buffer on AMX's tiles, in elements of two bytes or of their own
// type: the tiles of every head's keys and values, shared; and each thread's
// tiles of its queries, its scores, the tiles of its exponentials, its output
// and the sum of a stretch of keys, the largest score of each of its queries
// and the sum of the exponentials of each.
struct Amx_layout {
    std::int64_t queries;  // a block's, a multiple of 32
    std::int64_t stretch;  // the keys of a stretch of the second product, a multiple of 32
    std::int64_t dp, kp, vp;  // d, lk and dv rounded up to 32
    std::int64_t key_tile;    // a tile of sixteen keys over the whole of d
    std::int64_t value_tile;  // a tile of sixteen dimensions over all the keys
    std::int64_t head;        // a head's keys and values
    Work_layout layout;
    Shared<std::uint16_t> heads;
    Own<std::uint16_t> queries_tiles;
    Own<float> scores;
    Own<std::uint16_t> weights;
    Own<float> output, stretch_sum, maxima;
    Own<double> sums;
};

Amx_layout amx_layout(const Attention & a) {
    Amx_layout l;
    l.queries = std::max<std::int64_t>(kBlock, tuning().amx_queries / kBlock * kBlock);
    l.stretch = std::max<std::int64_t>(kBlock, a.key_block / kBlock * kBlock);
    l.dp = round_up(a.d, kBlock);
    l.kp = round_up(a.lk, kBlock);
    l.vp = round_up(a.dv, kBlock);
    l.key_tile = l.dp / kBlock * kBlockHalves;
    l.value_tile = l.kp / kBlock * kBlockHalves;
    // Each head's tiles a line of the cache from the last.
    l.head = round_up(l.kp / kTileRows * l.key_tile + l.vp / kTileRows * l.value_tile, 32);
    l.heads = l.layout.shared<std::uint16_t>(l.head * a.hk * a.b);
    l.queries_tiles = l.layout.own<std::uint16_t>(l.queries / kTileRows * l.key_tile);
    l.scores = l.layout.own<float>(l.queries * l.kp);
    l.weights = l.layout.own<std::uint16_t>(l.queries / kTileRows * (l.stretch / kBlock) * kBlockHalves);
    l.output = l.layout.own<float>(l.queries * l.vp);
    l.stretch_sum = l.layout.own<float>(l.queries * l.vp);
    l.maxima = l.layout.own<float>(l.queries);
    l.sums = l.layout.own<double>(l.queries);
    return l;
}

void compute_amx(const ggml_compute_params * const params, const Attention & a) {
    const Amx_layout l = amx_layout(a);
    Work_buffer buffer(params, l.layout);
    const std::int64_t values_offset = l.kp / kTileRows * l.key_tile;

    Work_queue queue(params, buffer);
    const Elements<std::uint16_t> heads = buffer[l.heads];
    const Elements<std::uint16_t> queries = buffer[l.queries_tiles];
    const Elements<float> scores = buffer[l.scores];
    const Elements<std::uint16_t> weights = buffer[l.weights];
    const Elements<float> output = buffer[l.output];
    const Elements<float> stretch_sum = buffer[l.stretch_sum];
    const Elements<float> maxima = buffer[l.maxima];
    const Elements<double> sums = buffer[l.sums];
    const auto head_of = [&](std::int64_t b, std::int64_t hk) { return heads.part((b * a.hk + hk) * l.head, l.head); };
    // The keys and the values of every head, an item each: a tile of the keys
    // for each sixteen keys, over d; one of the values for each sixteen
    // dimensions, over the keys; those past the last, zeros.
    const std::int64_t pack_items = 2 * a.b * a.hk;
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const std::int64_t head = item / 2;
        const std::int64_t b = head / a.hk;
        const std::int64_t hk = head % a.hk;
        const Elements<std::uint16_t> tiles = head_of(b, hk);
        const Phase_scope scope(Phase::pack_weights);
        if (item % 2 == 0) {
            const std::int64_t src = keys_at(a, b, hk);
            for (std::int64_t j = 0; j < l.kp; j += kTileRows) {
                const Elements<std::uint16_t> dst = tiles.part(j / kTileRows * l.key_tile, l.key_tile);
                if (j >= a.lk) {
                    fill_with(dst, std::uint16_t{0});
                    continue;
                }
                const std::int64_t cols = std::min(kTileRows, a.lk - j);
                if (a.ktype == GGML_TYPE_F16) {
                    const Elements<const std::uint16_t> keys = a.k16.from(src + j * a.knb1);
                    SDASM_CHECK(keys.size() >= (cols - 1) * a.knb1 + a.d);
                    sdasm_amx_pack_columns3_f16(dst.data(), keys.data(), bytes_of<std::uint16_t>(a.knb1), cols, a.d);
                }
                else {
                    const Elements<const float> keys = a.k32.from(src + j * a.knb1);
                    SDASM_CHECK(keys.size() >= (cols - 1) * a.knb1 + a.d);
                    sdasm_amx_pack_columns3_f32(dst.data(), keys.data(), bytes_of<float>(a.knb1), cols, a.d);
                }
            }
        }
        else {
            const std::int64_t src = values_at(a, b, hk);
            for (std::int64_t c = 0; c < l.vp; c += kTileRows) {
                const Elements<std::uint16_t> dst = tiles.part(values_offset + c / kTileRows * l.value_tile, l.value_tile);
                if (c >= a.dv) {
                    fill_with(dst, std::uint16_t{0});
                    continue;
                }
                const std::int64_t cols = std::min(kTileRows, a.dv - c);
                if (a.vtype == GGML_TYPE_F16) {
                    const Elements<const std::uint16_t> column = a.v16.from(src + c);
                    SDASM_CHECK(column.size() >= (a.lk - 1) * a.vnb1 + cols);
                    sdasm_amx_pack_panel3_f16(dst.data(), column.data(), bytes_of<std::uint16_t>(a.vnb1), cols, a.lk);
                }
                else {
                    const Elements<const float> column = a.v32.from(src + c);
                    SDASM_CHECK(column.size() >= (a.lk - 1) * a.vnb1 + cols);
                    sdasm_amx_pack_panel3_f32(dst.data(), column.data(), bytes_of<float>(a.vnb1), cols, a.lk);
                }
            }
        }
    }
    queue.barrier();

    const std::int64_t blocks = ceil_div(a.lq, l.queries);
    const std::int64_t units = a.b * a.h * blocks;
    const Single_kernel kernel = single_kernel(a);
    const bool tiles = !on_vectors(a);
    const bool half = half_products();
    bool configured = false;
    // The kernel's products of tiles a of `blocks` blocks of k and b, rows
    // a_next and columns b_next elements apart, into c, rows ldc apart.
    const auto product = [&](std::int64_t count, Elements<const std::uint16_t> ta, std::int64_t a_next,
                             Elements<const std::uint16_t> tb, std::int64_t b_next, Elements<float> c, std::int64_t ldc,
                             bool accumulate) {
        SDASM_CHECK(ta.size() >= a_next + count * kBlockHalves && tb.size() >= b_next + count * kBlockHalves &&
                    c.size() >= (kBlock - 1) * ldc + kBlock);
        kernel(count, ta.data(), bytes_of<std::uint16_t>(a_next), tb.data(), bytes_of<std::uint16_t>(b_next), c.data(),
               bytes_of<float>(ldc), accumulate ? 1 : 0);
    };
    for (; item < pack_items + units; item = queue.next()) {
        if (tiles && !configured) {
            sdasm_amx_configure();
            configured = true;
        }
        std::int64_t u = item - pack_items;
        const std::int64_t block = u % blocks;
        u /= blocks;
        const std::int64_t h = u % a.h;
        const std::int64_t b = u / a.h;
        const std::int64_t hk = h / (a.h / a.hk);
        const std::int64_t q0 = block * l.queries;
        const std::int64_t nq = std::min(l.queries, a.lq - q0);
        const std::int64_t rows = round_up(nq, kBlock);
        const Elements<const std::uint16_t> keys = head_of(b, hk);
        const Elements<const std::uint16_t> values = keys.from(values_offset);
        {
            // The queries' tiles, sixteen to a tile; past nq, zeros.
            const Phase_scope scope(Phase::pack_inputs);
            const std::int64_t src = b * a.qnb3 + h * a.qnb2 + q0 * a.qnb1;
            for (std::int64_t r = 0; r < rows; r += kTileRows) {
                const std::int64_t count = std::clamp<std::int64_t>(nq - r, 0, kTileRows);
                const Elements<std::uint16_t> dst = queries.part(r / kTileRows * l.key_tile, l.key_tile);
                const Elements<const float> from = count > 0 ? a.q.from(src + r * a.qnb1) : a.q;
                SDASM_CHECK(count == 0 || from.size() >= (count - 1) * a.qnb1 + a.d);
                sdasm_amx_pack_rows3(dst.data(), from.data(), bytes_of<float>(a.qnb1), count, a.d);
            }
        }
        {
            // The scores, queries by keys, every key of the block's queries; then
            // their scale and the largest of each query.
            const Phase_scope scope(Phase::product);
            for (std::int64_t jj = 0; jj < l.kp; jj += kBlock) {
                for (std::int64_t ii = 0; ii < rows; ii += kBlock) {
                    product(l.dp / kBlock, queries.from(ii / kTileRows * l.key_tile), l.key_tile,
                            keys.from(jj / kTileRows * l.key_tile), l.key_tile, scores.from(ii * l.kp + jj), l.kp, false);
                }
            }
        }
        {
            const Phase_scope scope(Phase::softmax);
            for (std::int64_t i = 0; i < nq; ++i) {
                const Elements<float> row = scores.part(i * l.kp, a.lk);
                float top = -INFINITY;
                for (float & x : row) {
                    x *= a.scale;
                    top = std::max(top, x);
                }
                maxima[i] = top;
                sums[i] = 0;
            }
        }
        // The exponentials and the sum of the values by them, a stretch of keys
        // at a time: the output, queries by dimensions.
        for (std::int64_t j0 = 0; j0 < a.lk; j0 += l.stretch) {
            const std::int64_t j1 = std::min(a.lk, j0 + l.stretch);
            const std::int64_t kb = ceil_div(j1 - j0, kBlock);
            {
                const Phase_scope scope(Phase::softmax);
                for (std::int64_t i = 0; i < nq; ++i) {
                    const Elements<float> row = scores.part(i * l.kp, a.lk);
                    const std::int64_t whole = (j1 - j0) / 64 * 64;
                    if (whole > 0) {
                        sdasm_exponentials_row_avx512(row.part(j0, whole).data(), whole, maxima[i], sums.part(i, 1).data());
                    }
                    for (std::int64_t j = j0 + whole; j < j1; ++j) {
                        row[j] = std::exp(row[j] - maxima[i]);
                        sums[i] += static_cast<double>(row[j]);
                    }
                }
            }
            {
                // Their tiles, as A: the keys past j1 in the last block are
                // zeros, as are the queries past nq.
                const Phase_scope scope(Phase::pack_inputs);
                for (std::int64_t r = 0; r < rows; r += kTileRows) {
                    const std::int64_t count = std::clamp<std::int64_t>(nq - r, 0, kTileRows);
                    const Elements<std::uint16_t> dst = weights.part(r / kTileRows * kb * kBlockHalves, kb * kBlockHalves);
                    const Elements<const float> from = count > 0 ? Elements<const float>(scores.from(r * l.kp + j0)) : scores;
                    SDASM_CHECK(count == 0 || from.size() >= (count - 1) * l.kp + (j1 - j0));
                    sdasm_amx_pack_rows3(dst.data(), from.data(), bytes_of<float>(l.kp), count, j1 - j0);
                }
            }
            // The stretch's sum, begun from zeros and added to the output
            // after: the kernel adds its small products first only to sums of
            // its own. Products of 16 bits add to the output itself.
            const Phase_scope scope(Phase::product);
            const Elements<float> into = j0 == 0 || half ? output : stretch_sum;
            for (std::int64_t dd = 0; dd < l.vp; dd += kBlock) {
                for (std::int64_t ii = 0; ii < rows; ii += kBlock) {
                    product(kb, weights.from(ii / kTileRows * kb * kBlockHalves), kb * kBlockHalves,
                            values.from(dd / kTileRows * l.value_tile + j0 / kBlock * kBlockHalves), l.value_tile,
                            into.from(ii * l.vp + dd), l.vp, half && j0 > 0);
                }
            }
            if (j0 > 0 && !half) {
                for (std::int64_t i = 0; i < rows * l.vp; ++i) {
                    output[i] += stretch_sum[i];
                }
            }
        }
        // Each query's output divided by the sum of its exponentials: by the
        // reciprocal of the sum, rounded to single precision.
        const Phase_scope scope(Phase::transform_out);
        for (std::int64_t i = 0; i < nq; ++i) {
            const auto reciprocal = static_cast<float>(1.0 / sums[i]);
            const Elements<float> row = output_row(a, b, h, q0 + i);
            const Elements<const float> o = output.part(i * l.vp, a.dv);
            for (std::int64_t c = 0; c < a.dv; ++c) {
                row[c] = o[c] * reciprocal;
            }
        }
    }
    if (configured) {
        sdasm_amx_release();
    }
}

// ---------------------------------------------------------------- quantized

// Self-attention's products in integers (quant.h, Tuning::bits): the keys,
// signed, each over the whole of the head's dimensions, and the queries,
// unsigned with a zero point, likewise -- their product, a score, is then
// one group, scaled once, the scale of the softmax folded into the queries'
// scales; the values, signed, each dimension's in groups of keys
// (Tuning::group), and the exponentials, unsigned, each query's in groups of
// keys too: an exponential of 8 bits of one scale for all the keys rounds the
// small ones away, 10% and more of the output lost where they are many, and
// the group's own scale keeps them. Measured on SD 1.5, the outputs of its
// self-attention are then within 1 to 4 times 10^-3 of the exact ones, in the
// L2 norm. Its cross-attention, over 77 tokens of the prompt, a fiftieth of
// the work, stays in single precision: over so few keys, a few large weights
// and many small ones, it is ten times further.
constexpr std::int64_t kQuantizedKeys = 256;  // the fewest keys quantized

bool quantized_attention(const Attention & a) {
    return quantized(Quantized_op::attention) && a.lk >= kQuantizedKeys;
}

// The quantization of the keys and queries, one group over the dimensions,
// and of the values and exponentials, groups of keys.
Quant over_dims(const Attention & a) {
    Quant q = quant();
    q.quads = ceil_div(a.d, 4);
    q.weight_groups = true;
    return q;
}

Quant over_keys() {
    Quant q = quant();
    q.weight_groups = true;
    return q;
}

// The work buffer of attention in integers: every head's quantized keys and
// values, shared; and each thread's panel of single precision, for the
// packing, its queries in single precision and quantized, its scores, its
// quantized exponentials, its output, the largest score of each of its queries
// and the sum of the exponentials of each.
struct Quantized_layout {
    std::int64_t key_block;  // the keys of a stretch, a multiple of 32
    std::int64_t lk_groups;  // lk rounded up to 32
    std::int64_t keys;       // a head's quantized keys: panels of kMr keys
    std::int64_t head;       // and its values: panels of kMr dimensions over every key
    Work_layout layout;
    Shared<std::uint8_t> heads;
    Own<float> floats, queries;
    Own<std::uint8_t> qqueries;
    Own<float> scores;
    Own<std::uint8_t> weights;
    Own<float> output, maxima;
    Own<double> sums;
};

Quantized_layout quantized_layout(const Attention & a) {
    Quantized_layout l;
    const Quant qd = over_dims(a);
    const Quant qk = over_keys();
    const std::int64_t group = 4 * qk.quads;
    l.key_block = std::max<std::int64_t>(group, a.key_block / group * group);
    l.lk_groups = round_up(a.lk, group);
    // Each head's keys, and its values, a line of the cache from the last.
    l.keys = round_up(ceil_div(a.lk, kMr) * qpanel_bytes(a.d, kMr, qd), 64);
    l.head = l.keys + round_up(ceil_div(a.dv, kMr) * qpanel_bytes(l.lk_groups, kMr, qk), 64);
    l.heads = l.layout.shared<std::uint8_t>(l.head * a.hk * a.b);
    l.floats = l.layout.own<float>(std::max(a.d, l.lk_groups) * kMr);
    l.queries = l.layout.own<float>(a.queries * a.d);
    l.qqueries = l.layout.own<std::uint8_t>(a.queries / kNr * qpanel_bytes(a.d, kNr, qd));
    l.scores = l.layout.own<float>(a.queries * round_up(a.lk, kMr));
    l.weights = l.layout.own<std::uint8_t>(a.queries / kNr * qpanel_bytes(l.key_block, kNr, qk));
    l.output = l.layout.own<float>(round_up(a.dv, kMr) * a.queries);
    l.maxima = l.layout.own<float>(a.queries);
    l.sums = l.layout.own<double>(a.queries);
    return l;
}

void compute_quantized(const ggml_compute_params * const params, const Attention & a) {
    const Quantized_layout l = quantized_layout(a);
    Work_buffer buffer(params, l.layout);
    const Quant qd = over_dims(a);
    const Quant qk = over_keys();
    const Qkernel kernel = qkernel_b_unsigned();
    const std::int64_t key_panel = qpanel_bytes(a.d, kMr, qd);
    const std::int64_t value_panel = qpanel_bytes(l.lk_groups, kMr, qk);
    const std::int64_t query_panel = qpanel_bytes(a.d, kNr, qd);
    const std::int64_t weight_panel = qpanel_bytes(l.key_block, kNr, qk);

    Work_queue queue(params, buffer);
    const Elements<std::uint8_t> heads = buffer[l.heads];
    const Elements<float> floats = buffer[l.floats];
    const Elements<float> queries = buffer[l.queries];
    const Elements<std::uint8_t> qqueries = buffer[l.qqueries];
    const Elements<float> scores = buffer[l.scores];
    const Elements<std::uint8_t> weights = buffer[l.weights];
    const Elements<float> output = buffer[l.output];
    const Elements<float> maxima = buffer[l.maxima];
    const Elements<double> sums = buffer[l.sums];
    const auto head_of = [&](std::int64_t b, std::int64_t hk) { return heads.part((b * a.hk + hk) * l.head, l.head); };
    // The keys and the values of every head: a panel of kMr keys, or of kMr
    // dimensions, to an item, packed in single precision and quantized.
    const std::int64_t key_panels = ceil_div(a.lk, kMr);
    const std::int64_t value_panels = ceil_div(a.dv, kMr);
    const std::int64_t per_head = key_panels + value_panels;
    const std::int64_t pack_items = a.b * a.hk * per_head;
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const std::int64_t head = item / per_head;
        const std::int64_t panel = item % per_head;
        const std::int64_t b = head / a.hk;
        const std::int64_t hk = head % a.hk;
        const Elements<std::uint8_t> keys = head_of(b, hk);
        const Phase_scope scope(Phase::pack_weights);
        if (panel < key_panels) {
            const std::int64_t r0 = panel * kMr;
            pack_keys(floats, a, b, hk, r0, std::min<std::int64_t>(kMr, a.lk - r0), kMr);
            quantize_panel_weights(keys.part(panel * key_panel, key_panel), floats, a.d, kMr, Qlayout::rows, qd);
        }
        else {
            const std::int64_t d0 = (panel - key_panels) * kMr;
            const std::int64_t src = values_at(a, b, hk);
            const std::int64_t width = std::min<std::int64_t>(kMr, a.dv - d0);
            // The panel of kMr dimensions over the keys, [lk][kMr], zeros past
            // the last dimension and the last key.
            for (std::int64_t j = 0; j < l.lk_groups; ++j) {
                for (std::int64_t i = 0; i < kMr; ++i) {
                    floats[j * kMr + i] = j < a.lk && i < width ? value(a, src + j * a.vnb1 + d0 + i) : 0.0f;
                }
            }
            quantize_panel_weights(keys.part(l.keys + (panel - key_panels) * value_panel, value_panel), floats, l.lk_groups,
                                   kMr, Qlayout::rows, qk);
        }
    }
    queue.barrier();

    const std::int64_t blocks = ceil_div(a.lq, a.queries);
    const std::int64_t units = a.b * a.h * blocks;
    const std::int64_t lk_padded = key_panels * kMr;
    for (; item < pack_items + units; item = queue.next()) {
        std::int64_t u = item - pack_items;
        const std::int64_t block = u % blocks;
        u /= blocks;
        const std::int64_t h = u % a.h;
        const std::int64_t b = u / a.h;
        const std::int64_t hk = h / (a.h / a.hk);
        const std::int64_t q0 = block * a.queries;
        const std::int64_t nq = std::min(a.queries, a.lq - q0);
        const std::int64_t qpanels = ceil_div(nq, kNr);
        const Elements<const std::uint8_t> keys = head_of(b, hk);
        const Elements<const std::uint8_t> values = keys.from(l.keys);

        {
            // The queries, a panel of kNr, the scale of the softmax folded into
            // their scales and offsets.
            const Phase_scope scope(Phase::pack_inputs);
            pack_rows_f32(queries, a.q.from(b * a.qnb3 + h * a.qnb2 + q0 * a.qnb1), a.qnb1, nq, 0, a.d, kNr);
            for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                const Elements<std::uint8_t> panel = qqueries.part(qp * query_panel, query_panel);
                quantize_panel_inputs(panel, queries.part(qp * a.d * kNr, a.d * kNr), a.d, kNr, Qlayout::columns, qd);
                std::array<float, 2 * kNr> meta{};
                const Elements<std::uint8_t> where = panel.part(qd.quads * kNr * 4, 2 * kNr * 4);
                read_floats(where, meta);
                for (float & m : meta) {
                    m *= a.scale;
                }
                write_floats(where, meta);
            }
        }
        {
            // The scores, keys by queries, scaled; and each query's greatest.
            const Phase_scope scope(Phase::product);
            for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                const Elements<float> panel = scores.part(qp * lk_padded * kNr, lk_padded * kNr);
                // Each tile's greatest taken while it is in the nearest cache.
                std::array<float, kNr> top{};
                fill_with(Elements<float>(top), -INFINITY);
                for (std::int64_t kp = 0; kp < key_panels; ++kp) {
                    const std::int64_t rows = std::min<std::int64_t>(kMr, a.lk - kp * kMr);
                    const Elements<float> tile = panel.part(kp * kMr * kNr, kMr * kNr);
                    qtile(kernel, 1, qd.quads, keys.part(kp * key_panel, key_panel),
                          qqueries.part(qp * query_panel, query_panel), tile, kNr, false, rows, kNr);
                    for (std::int64_t j = 0; j < rows; ++j) {
                        for (std::size_t lane = 0; lane < kNr; ++lane) {
                            top[lane] = std::max(top[lane], tile[j * kNr + static_cast<std::int64_t>(lane)]);
                        }
                    }
                }
                for (std::size_t lane = 0; lane < kNr; ++lane) {
                    maxima[qp * kNr + static_cast<std::int64_t>(lane)] = top[lane];
                    sums[qp * kNr + static_cast<std::int64_t>(lane)] = 0;
                }
            }
        }
        // The exponentials of a stretch of keys, quantized, and the sum of the
        // values by them.
        for (std::int64_t j0 = 0; j0 < a.lk; j0 += l.key_block) {
            const std::int64_t j1 = std::min(a.lk, j0 + l.key_block);
            {
                const Phase_scope scope(Phase::softmax);
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    const Elements<float> panel = scores.part(qp * lk_padded * kNr, lk_padded * kNr);
                    exponentials(panel, j0, j1, maxima.part(qp * kNr, kNr), sums.part(qp * kNr, kNr));
                    quantize_panel_inputs(weights.part(qp * weight_panel, weight_panel), panel.part(j0 * kNr, (j1 - j0) * kNr),
                                          j1 - j0, kNr, Qlayout::columns, qk);
                }
            }
            const Phase_scope scope(Phase::product);
            const std::int64_t groups = ceil_div(j1 - j0, 4 * qk.quads);
            for (std::int64_t dp = 0; dp < value_panels; ++dp) {
                const Elements<const std::uint8_t> v =
                    values.from(dp * value_panel + (j0 / (4 * qk.quads)) * qgroup_bytes(qk.quads, kMr));
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    qtile(kernel, groups, qk.quads, v, weights.from(qp * weight_panel), output.from(dp * kMr * a.queries + qp * kNr),
                          a.queries, j0 > 0, kMr, kNr);
                }
            }
        }
        const Phase_scope scope(Phase::transform_out);
        for (std::int64_t i = 0; i < nq; ++i) {
            const auto reciprocal = static_cast<float>(1.0 / sums[i]);
            const Elements<float> row = output_row(a, b, h, q0 + i);
            for (std::int64_t c = 0; c < a.dv; ++c) {
                row[c] = output[c * a.queries + i] * reciprocal;
            }
        }
    }
}

}  // namespace

bool attention_supported(const ggml_tensor * const node) {
    if (node->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * const q = node->src[0];
    const ggml_tensor * const k = node->src[1];
    const ggml_tensor * const v = node->src[2];
    const float max_bias = std::bit_cast<float>(node->op_params[1]);
    const float softcap = std::bit_cast<float>(node->op_params[2]);
    const auto kv = [](const ggml_tensor * const t) {
        return (t->type == GGML_TYPE_F32 && holds<float>(t)) || (t->type == GGML_TYPE_F16 && holds<std::uint16_t>(t));
    };
    return node->src[3] == nullptr && node->src[4] == nullptr && max_bias == 0.0f && softcap == 0.0f &&
           holds<float>(q) && kv(k) && kv(v) && q->nb[0] == sizeof(float) && k->nb[0] == ggml_type_size(k->type) &&
           v->nb[0] == ggml_type_size(v->type) && holds<float>(node) && ggml_is_contiguous(node) && q->ne[0] == k->ne[0] &&
           k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && k->ne[2] > 0 && q->ne[2] % k->ne[2] == 0 &&
           q->ne[3] == k->ne[3] && q->ne[3] == v->ne[3] && k->ne[1] >= 1;
}

std::size_t attention_work_size(const ggml_tensor * const node, int n_threads) {
    const Attention a = geometry(node);
    if (quantized_attention(a)) {
        return quantized_layout(a).layout.bytes(n_threads);
    }
    if (on_amx(a)) {
        return amx_layout(a).layout.bytes(n_threads);
    }
    return layout(a).layout.bytes(n_threads);
}

void attention_compute(const ggml_compute_params * const params, const ggml_tensor * const node) {
    const Attention a = geometry(node);
    if (quantized_attention(a)) {
        compute_quantized(params, a);
        return;
    }
    if (on_amx(a)) {
        compute_amx(params, a);
        return;
    }
    const Layout l = layout(a);
    Work_buffer buffer(params, l.layout);
    const Microkernel uk = microkernel();
    const Scores_kernel sk = scores_kernel();
    // Where the wide micro-kernel is taken (gemm.h), two panels of queries at
    // a time, in both products: the scores and the sum of the values.
    const Microkernel_wide wide = microkernel_wide();
    const Scores_kernel_wide skw = scores_kernel_wide();

    Work_queue queue(params, buffer);
    const Elements<float> heads = buffer[l.heads];
    const Elements<float> queries = buffer[l.queries];
    const Elements<float> scores = buffer[l.scores];
    const Elements<float> output = buffer[l.output];
    const Elements<float> stretch_sum = buffer[l.stretch_sum];
    const Elements<float> maxima = buffer[l.maxima];
    const Elements<double> sums = buffer[l.sums];
    const std::int64_t head_floats = l.keys_floats + l.values_floats;
    const auto keys_of = [&](std::int64_t b, std::int64_t hk) { return heads.part((b * a.hk + hk) * head_floats, head_floats); };
    // The keys and the values of every head, an item each.
    const std::int64_t pack_items = 2 * a.b * a.hk;
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const std::int64_t head = item / 2;
        const std::int64_t b = head / a.hk;
        const std::int64_t hk = head % a.hk;
        const Elements<float> keys = keys_of(b, hk);
        const Phase_scope scope(Phase::pack_weights);
        if (item % 2 == 0) {
            pack_keys(keys, a, b, hk, 0, a.lk, kMr);
        }
        else {
            const std::int64_t src = values_at(a, b, hk);
            const Elements<float> values = keys.from(l.keys_floats);
            if (a.vtype == GGML_TYPE_F16) {
                pack_values<std::uint16_t>(values, a.v16.from(src), a.vnb1, a.lk, a.dv,
                                           [](std::uint16_t x) { return static_cast<float>(std::bit_cast<_Float16>(x)); });
            }
            else {
                pack_values<float>(values, a.v32.from(src), a.vnb1, a.lk, a.dv, [](float x) { return x; });
            }
        }
    }
    queue.barrier();

    const std::int64_t blocks = ceil_div(a.lq, a.queries);
    const std::int64_t units = a.b * a.h * blocks;
    const std::int64_t panels_keys = ceil_div(a.lk, kMr);
    const std::int64_t panels_dims = ceil_div(a.dv, kMr);
    const std::int64_t lk_padded = panels_keys * kMr;
    for (; item < pack_items + units; item = queue.next()) {
        std::int64_t u = item - pack_items;
        const std::int64_t block = u % blocks;
        u /= blocks;
        const std::int64_t h = u % a.h;
        const std::int64_t b = u / a.h;
        const std::int64_t hk = h / (a.h / a.hk);
        const std::int64_t q0 = block * a.queries;
        const std::int64_t nq = std::min(a.queries, a.lq - q0);
        const std::int64_t qpanels = ceil_div(nq, kNr);
        const Elements<const float> keys = keys_of(b, hk);
        const Elements<const float> values = keys.from(l.keys_floats);

        {
            const Phase_scope scope(Phase::pack_inputs);
            pack_rows_f32(queries, a.q.from(b * a.qnb3 + h * a.qnb2 + q0 * a.qnb1), a.qnb1, nq, 0, a.d, kNr);
        }
        // The scores, keys by queries, scaled: a panel [lk][kNr] for each kNr
        // queries; and the largest score of each query.
        for (std::int64_t i = 0; i < qpanels * kNr; ++i) {
            maxima[i] = -INFINITY;
            sums[i] = 0;
        }
        {
            const Phase_scope scope(Phase::product);
            // A score is a sum over the dimensions of the head: in stretches of
            // no more than a.sum_dims, each summed in the registers and then
            // added to the sum of those before, so that no sum of the
            // micro-kernel runs long enough to round more than ggml's dot
            // product, which sums 32 at a time.
            const std::int64_t stretches = ceil_div(a.d, a.sum_dims);
            const std::int64_t stretch = ceil_div(a.d, stretches);
            const auto product = [&](Elements<const float> ka, Elements<const float> qb, Elements<float> s, bool scaled,
                                     Elements<float> top) {
                for (std::int64_t e0 = 0; e0 < a.d; e0 += stretch) {
                    const std::int64_t e = std::min(stretch, a.d - e0);
                    const Elements<const float> pa = ka.part(e0 * kMr, e * kMr);
                    const Elements<const float> pb = qb.part(e0 * kNr, e * kNr);
                    SDASM_CHECK(s.size() >= kMr * kNr);
                    if (scaled && e0 + e == a.d) {
                        SDASM_CHECK(top.size() >= kNr);
                        sk(e, pa.data(), pb.data(), s.data(), bytes_of<float>(kNr), top.data(), a.scale, e0 > 0 ? 1 : 0);
                    }
                    else {
                        uk(e, pa.data(), pb.data(), s.data(), bytes_of<float>(kNr), e0 > 0 ? 1 : 0);
                    }
                }
            };
            // Two panels of queries, their tiles of the scores lk_padded kNr
            // apart, a stretch of the dimensions at a time as above.
            const auto product_wide = [&](Elements<const float> ka, Elements<const float> qb, Elements<float> s,
                                          Elements<float> top) {
                const std::int64_t q_next = a.d * kNr;
                const std::int64_t s_next = lk_padded * kNr;
                SDASM_CHECK(s.size() >= s_next + kMr * kNr && top.size() >= 2 * kNr);
                for (std::int64_t e0 = 0; e0 < a.d; e0 += stretch) {
                    const std::int64_t e = std::min(stretch, a.d - e0);
                    const Elements<const float> pa = ka.part(e0 * kMr, e * kMr);
                    const Elements<const float> pb = qb.part(e0 * kNr, q_next + e * kNr);
                    const std::int64_t flags = (e0 > 0 ? 1 : 0) | (e0 + e == a.d ? 2 : 0);
                    skw(e, pa.data(), pb.data(), bytes_of<float>(q_next), s.data(), bytes_of<float>(s_next),
                        bytes_of<float>(kNr), top.data(), a.scale, flags);
                }
            };
            const std::int64_t pairs = skw != nullptr ? qpanels / 2 : 0;
            const std::int64_t whole = a.lk / kMr;
            for (std::int64_t kp = 0; kp < whole; ++kp) {
                const Elements<const float> ka = keys.part(kp * kMr * a.d, kMr * a.d);
                for (std::int64_t qp = 0; qp < 2 * pairs; qp += 2) {
                    product_wide(ka, queries.from(qp * a.d * kNr), scores.from(qp * lk_padded * kNr + kp * kMr * kNr),
                                 maxima.part(qp * kNr, 2 * kNr));
                }
                for (std::int64_t qp = 2 * pairs; qp < qpanels; ++qp) {
                    const Elements<float> s = scores.part(qp * lk_padded * kNr + kp * kMr * kNr, kMr * kNr);
                    product(ka, queries.part(qp * a.d * kNr, a.d * kNr), s, true, maxima.part(qp * kNr, kNr));
                }
            }
            // The keys past the last kMr: their panel by the micro-kernel, and
            // then their scale and their maximum, which the zeros past them in
            // the panel have no part in.
            if (whole < panels_keys) {
                const Elements<const float> ka = keys.part(whole * kMr * a.d, kMr * a.d);
                const std::int64_t rows = a.lk - whole * kMr;
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    const Elements<float> s = scores.part(qp * lk_padded * kNr + whole * kMr * kNr, kMr * kNr);
                    product(ka, queries.part(qp * a.d * kNr, a.d * kNr), s, false, Elements<float>());
                    for (std::int64_t i = 0; i < rows; ++i) {
                        for (std::int64_t lane = 0; lane < kNr; ++lane) {
                            const float x = s[i * kNr + lane] * a.scale;
                            s[i * kNr + lane] = x;
                            maxima[qp * kNr + lane] = std::max(maxima[qp * kNr + lane], x);
                        }
                    }
                }
            }
        }
        // The exponentials and the sum of the values by them, a stretch of keys
        // at a time: the output, dimensions by queries.
        for (std::int64_t j0 = 0; j0 < a.lk; j0 += a.key_block) {
            const std::int64_t j1 = std::min(a.lk, j0 + a.key_block);
            {
                const Phase_scope scope(Phase::softmax);
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    exponentials(scores.part(qp * lk_padded * kNr, lk_padded * kNr), j0, j1, maxima.part(qp * kNr, kNr),
                                 sums.part(qp * kNr, kNr));
                }
            }
            // The sum over the keys in three steps: a.sum_keys keys at a time in
            // the registers, those sums added together over the stretch, and
            // the stretch's sum added to the output: no sum runs long enough
            // to round more than ggml's dot product, which sums 32 at a time
            // and then the 32 sums.
            const Phase_scope scope(Phase::product);
            const Elements<float> into = j0 == 0 ? output : stretch_sum;
            const std::int64_t step = panel_step(wide) / kNr;
            for (std::int64_t dp = 0; dp < panels_dims; ++dp) {
                for (std::int64_t qp = 0; qp < qpanels; qp += step) {
                    const std::int64_t cols = std::min(step, qpanels - qp) * kNr;
                    const Elements<float> o = into.from(dp * kMr * a.queries + qp * kNr);
                    for (std::int64_t k0 = j0; k0 < j1; k0 += a.sum_keys) {
                        const std::int64_t count = std::min(a.sum_keys, j1 - k0);
                        const Elements<const float> va = values.part(dp * a.lk * kMr + k0 * kMr, count * kMr);
                        const Elements<const float> pb = scores.from(qp * lk_padded * kNr + k0 * kNr);
                        tiles(uk, wide, count, va, pb, lk_padded * kNr, o, a.queries, k0 > j0, kMr, cols);
                    }
                }
            }
            if (j0 > 0) {
                const std::int64_t n = panels_dims * kMr * a.queries;
                for (std::int64_t i = 0; i < n; ++i) {
                    output[i] += stretch_sum[i];
                }
            }
        }
        // Each query's output divided by the sum of its exponentials: by the
        // reciprocal of the sum, rounded to single precision, as ggml's softmax
        // multiplies the exponentials by it.
        const Phase_scope scope(Phase::transform_out);
        for (std::int64_t i = 0; i < nq; ++i) {
            const auto reciprocal = static_cast<float>(1.0 / sums[i]);
            const Elements<float> row = output_row(a, b, h, q0 + i);
            for (std::int64_t c = 0; c < a.dv; ++c) {
                row[c] = output[c * a.queries + i] * reciprocal;
            }
        }
    }
}

}  // namespace sdasm
