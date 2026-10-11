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
// at a time.
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
#include <cmath>
#include <cstdint>
#include <cstring>

#include "gemm.h"
#include "kernels.h"
#include "ops.h"
#include "phases.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"
#include "scalar_math.h"

namespace sdasm {

namespace {


struct Attention {
    const char * q;  // [d, lq, h, b]
    std::int64_t qnb1, qnb2, qnb3;
    const char * k;  // [d, lk, hk, b]
    std::int64_t knb1, knb2, knb3;
    ggml_type ktype;
    const char * v;  // [dv, lk, hk, b]
    std::int64_t vnb1, vnb2, vnb3;
    ggml_type vtype;
    char * out;  // [dv, h, lq, b]
    std::int64_t onb1, onb2, onb3;
    std::int64_t d, dv, lq, lk, h, hk, b;
    float scale;
    std::int64_t queries;    // queries in a block, a multiple of kNr
    std::int64_t key_block;  // keys in a stretch of the second product, a multiple of eight
    std::int64_t sum_keys;   // the most keys the micro-kernel sums an output over at once
    std::int64_t sum_dims;   // the most dimensions it sums a score over at once
};

Attention geometry(const ggml_tensor * node) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    Attention a{};
    a.q = static_cast<const char *>(q->data);
    a.qnb1 = static_cast<std::int64_t>(q->nb[1]);
    a.qnb2 = static_cast<std::int64_t>(q->nb[2]);
    a.qnb3 = static_cast<std::int64_t>(q->nb[3]);
    a.k = static_cast<const char *>(k->data);
    a.knb1 = static_cast<std::int64_t>(k->nb[1]);
    a.knb2 = static_cast<std::int64_t>(k->nb[2]);
    a.knb3 = static_cast<std::int64_t>(k->nb[3]);
    a.ktype = k->type;
    a.v = static_cast<const char *>(v->data);
    a.vnb1 = static_cast<std::int64_t>(v->nb[1]);
    a.vnb2 = static_cast<std::int64_t>(v->nb[2]);
    a.vnb3 = static_cast<std::int64_t>(v->nb[3]);
    a.vtype = v->type;
    a.out = static_cast<char *>(node->data);
    a.onb1 = static_cast<std::int64_t>(node->nb[1]);
    a.onb2 = static_cast<std::int64_t>(node->nb[2]);
    a.onb3 = static_cast<std::int64_t>(node->nb[3]);
    a.d = q->ne[0];
    a.dv = v->ne[0];
    a.lq = q->ne[1];
    a.lk = k->ne[1];
    a.h = q->ne[2];
    a.hk = k->ne[2];
    a.b = q->ne[3];
    std::memcpy(&a.scale, &node->op_params[0], sizeof a.scale);
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

struct Layout {
    std::size_t keys_floats;    // the packed keys of one head
    std::size_t values_floats;  // the packed values of one head
    std::size_t shared_bytes;
    std::size_t queries_bytes;  // a thread's packed queries
    std::size_t scores_bytes;   // a thread's scores
    std::size_t output_bytes;   // a thread's transposed output, and its sum over a stretch of keys
    std::size_t maxima_bytes;   // the largest score of each of a thread's queries
    std::size_t sums_bytes;     // the sum of the exponentials of each, in double precision
    std::size_t thread_bytes;
};

Layout layout(const Attention & a) {
    Layout l{};
    l.keys_floats = static_cast<std::size_t>(round_up(a.lk, kMr) * a.d);
    l.values_floats = static_cast<std::size_t>(round_up(a.dv, kMr) * a.lk);
    l.shared_bytes = align_up((l.keys_floats + l.values_floats) * sizeof(float) * static_cast<std::size_t>(a.hk * a.b));
    l.queries_bytes = align_up(static_cast<std::size_t>(a.queries * a.d) * sizeof(float));
    l.scores_bytes = align_up(static_cast<std::size_t>(a.queries * round_up(a.lk, kMr)) * sizeof(float));
    l.output_bytes = align_up(static_cast<std::size_t>(round_up(a.dv, kMr) * a.queries) * sizeof(float));
    l.maxima_bytes = align_up(static_cast<std::size_t>(a.queries) * sizeof(float));
    l.sums_bytes = align_up(static_cast<std::size_t>(a.queries) * sizeof(double));
    l.thread_bytes = l.queries_bytes + l.scores_bytes + 2 * l.output_bytes + l.maxima_bytes + l.sums_bytes;
    return l;
}

// Packs the values [0, lk) of one head into panels of kMr dimensions:
// [panel][lk][kMr], zeros past the last dimension.
template <class T, class Convert>
void pack_values(float * dst, const char * src, std::int64_t stride, std::int64_t lk, std::int64_t dv, Convert convert) {
    for (std::int64_t d0 = 0; d0 < dv; d0 += kMr, dst += lk * kMr) {
        const std::int64_t width = std::min<std::int64_t>(kMr, dv - d0);
        for (std::int64_t j = 0; j < lk; ++j) {
            const T * row = reinterpret_cast<const T *>(src + j * stride) + d0;
            float * out = dst + j * kMr;
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
void exponentials_cxx(float * s, std::int64_t n, const float * max, double * total) {
    for (std::int64_t j = 0; j < n; ++j) {
        float * row = s + j * kNr;
        for (int l = 0; l < kNr; ++l) {
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
void exponentials(float * s, std::int64_t j0, std::int64_t j1, const float * max, double * total) {
    const std::int64_t whole = (j1 - j0) / 8 * 8;
    if (whole > 0) {
        exponentials_kernel()(s + j0 * kNr, whole, max, total);
    }
    for (std::int64_t j = j0 + whole; j < j1; ++j) {
        float * row = s + j * kNr;
        for (int l = 0; l < kNr; ++l) {
            const float e = std::exp(row[l] - max[l]);
            row[l] = e;
            total[l] += static_cast<double>(e);
        }
    }
}

}  // namespace

namespace {

// ---------------------------------------------------------------- on AMX

constexpr std::int64_t kTileRows = 16;
constexpr std::int64_t kBlock = 32;           // the rows and columns of a block of the kernel, and its k
constexpr std::int64_t kBlockBytes = 3072;    // a block of k of a tile's rows or columns, three parts

bool on_amx(const Attention &) {
    return amx_kernels() && tuning().amx_attention != 0;
}

struct Amx_layout {
    std::int64_t queries;    // a block's, a multiple of 32
    std::int64_t stretch;    // the keys of a stretch of the second product, a multiple of 32
    std::int64_t dp, kp, vp;  // d, lk and dv rounded up to 32
    std::int64_t key_tile_bytes;    // a tile of sixteen keys over the whole of d
    std::int64_t value_tile_bytes;  // a tile of sixteen dimensions over all the keys
    std::size_t head_bytes;         // a head's keys and values
    std::size_t shared_bytes;
    std::size_t queries_bytes, scores_bytes, weights_bytes, output_bytes, maxima_bytes, sums_bytes;
    std::size_t thread_bytes;
};

Amx_layout amx_layout(const Attention & a) {
    Amx_layout l{};
    l.queries = std::max<std::int64_t>(kBlock, tuning().amx_queries / kBlock * kBlock);
    l.stretch = std::max<std::int64_t>(kBlock, a.key_block / kBlock * kBlock);
    l.dp = round_up(a.d, kBlock);
    l.kp = round_up(a.lk, kBlock);
    l.vp = round_up(a.dv, kBlock);
    l.key_tile_bytes = l.dp / kBlock * kBlockBytes;
    l.value_tile_bytes = l.kp / kBlock * kBlockBytes;
    l.head_bytes = align_up(static_cast<std::size_t>(l.kp / kTileRows * l.key_tile_bytes + l.vp / kTileRows * l.value_tile_bytes));
    l.shared_bytes = l.head_bytes * static_cast<std::size_t>(a.hk * a.b);
    l.queries_bytes = align_up(static_cast<std::size_t>(l.queries / kTileRows * l.key_tile_bytes));
    l.scores_bytes = align_up(static_cast<std::size_t>(l.queries * l.kp) * sizeof(float));
    l.weights_bytes = align_up(static_cast<std::size_t>(l.queries / kTileRows * (l.stretch / kBlock) * kBlockBytes));
    l.output_bytes = align_up(static_cast<std::size_t>(l.queries * l.vp) * sizeof(float));
    l.maxima_bytes = align_up(static_cast<std::size_t>(l.queries) * sizeof(float));
    l.sums_bytes = align_up(static_cast<std::size_t>(l.queries) * sizeof(double));
    l.thread_bytes = l.queries_bytes + l.scores_bytes + l.weights_bytes + 2 * l.output_bytes + l.maxima_bytes + l.sums_bytes;
    return l;
}

void compute_amx(const ggml_compute_params * params, const Attention & a) {
    const Amx_layout l = amx_layout(a);
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    char * mine = base + l.shared_bytes + l.thread_bytes * static_cast<std::size_t>(params->ith);
    auto * queries = reinterpret_cast<std::uint16_t *>(mine);
    mine += l.queries_bytes;
    auto * scores = reinterpret_cast<float *>(mine);
    mine += l.scores_bytes;
    auto * weights = reinterpret_cast<std::uint16_t *>(mine);
    mine += l.weights_bytes;
    auto * output = reinterpret_cast<float *>(mine);
    mine += l.output_bytes;
    auto * stretch_sum = reinterpret_cast<float *>(mine);
    mine += l.output_bytes;
    auto * maxima = reinterpret_cast<float *>(mine);
    mine += l.maxima_bytes;
    auto * sums = reinterpret_cast<double *>(mine);
    const auto head_of = [&](std::int64_t b, std::int64_t hk) {
        return reinterpret_cast<std::uint16_t *>(base + l.head_bytes * static_cast<std::size_t>(b * a.hk + hk));
    };
    const auto at = [](std::uint16_t * p, std::int64_t bytes) { return p + bytes / 2; };
    const std::int64_t values_offset = l.kp / kTileRows * l.key_tile_bytes;

    Work_queue queue(params);
    // The keys and the values of every head, an item each: a tile of the keys
    // for each sixteen keys, over d; one of the values for each sixteen
    // dimensions, over the keys; those past the last, zeros.
    const std::int64_t pack_items = 2 * a.b * a.hk;
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const std::int64_t head = item / 2;
        const std::int64_t b = head / a.hk;
        const std::int64_t hk = head % a.hk;
        std::uint16_t * tiles = head_of(b, hk);
        const Phase_scope scope(Phase::pack_weights);
        if (item % 2 == 0) {
            const char * src = a.k + b * a.knb3 + hk * a.knb2;
            for (std::int64_t j = 0; j < l.kp; j += kTileRows) {
                std::uint16_t * dst = at(tiles, j / kTileRows * l.key_tile_bytes);
                if (j >= a.lk) {
                    std::memset(dst, 0, static_cast<std::size_t>(l.key_tile_bytes));
                    continue;
                }
                const std::int64_t cols = std::min(kTileRows, a.lk - j);
                if (a.ktype == GGML_TYPE_F16) {
                    sdasm_amx_pack_columns3_f16(dst, reinterpret_cast<const std::uint16_t *>(src + j * a.knb1), a.knb1, cols, a.d);
                }
                else {
                    sdasm_amx_pack_columns3_f32(dst, reinterpret_cast<const float *>(src + j * a.knb1), a.knb1, cols, a.d);
                }
            }
        }
        else {
            const char * src = a.v + b * a.vnb3 + hk * a.vnb2;
            const std::size_t element = a.vtype == GGML_TYPE_F16 ? 2 : 4;
            for (std::int64_t c = 0; c < l.vp; c += kTileRows) {
                std::uint16_t * dst = at(tiles, values_offset + c / kTileRows * l.value_tile_bytes);
                if (c >= a.dv) {
                    std::memset(dst, 0, static_cast<std::size_t>(l.value_tile_bytes));
                    continue;
                }
                const std::int64_t cols = std::min(kTileRows, a.dv - c);
                const char * column = src + static_cast<std::size_t>(c) * element;
                if (a.vtype == GGML_TYPE_F16) {
                    sdasm_amx_pack_panel3_f16(dst, reinterpret_cast<const std::uint16_t *>(column), a.vnb1, cols, a.lk);
                }
                else {
                    sdasm_amx_pack_panel3_f32(dst, reinterpret_cast<const float *>(column), a.vnb1, cols, a.lk);
                }
            }
        }
    }
    queue.barrier();

    const std::int64_t blocks = ceil_div(a.lq, l.queries);
    const std::int64_t units = a.b * a.h * blocks;
    const std::int64_t ld_scores = l.kp * static_cast<std::int64_t>(sizeof(float));
    const std::int64_t ld_output = l.vp * static_cast<std::int64_t>(sizeof(float));
    bool configured = false;
    for (; item < pack_items + units; item = queue.next()) {
        if (!configured) {
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
        std::uint16_t * keys = head_of(b, hk);
        std::uint16_t * values = at(keys, values_offset);
        {
            // The queries' tiles, sixteen to a tile; past nq, zeros.
            const Phase_scope scope(Phase::pack_inputs);
            const char * src = a.q + b * a.qnb3 + h * a.qnb2 + q0 * a.qnb1;
            for (std::int64_t r = 0; r < rows; r += kTileRows) {
                sdasm_amx_pack_rows3(at(queries, r / kTileRows * l.key_tile_bytes),
                                     reinterpret_cast<const float *>(src + r * a.qnb1), a.qnb1,
                                     std::clamp<std::int64_t>(nq - r, 0, kTileRows), a.d);
            }
        }
        {
            // The scores, queries by keys, every key of the block's queries; then
            // their scale and the largest of each query.
            const Phase_scope scope(Phase::product);
            for (std::int64_t jj = 0; jj < l.kp; jj += kBlock) {
                for (std::int64_t ii = 0; ii < rows; ii += kBlock) {
                    sdasm_amx_single_2x2(l.dp / kBlock, at(queries, ii / kTileRows * l.key_tile_bytes), l.key_tile_bytes,
                                         at(keys, jj / kTileRows * l.key_tile_bytes), l.key_tile_bytes,
                                         scores + ii * l.kp + jj, ld_scores, 0);
                }
            }
        }
        {
            const Phase_scope scope(Phase::softmax);
            for (std::int64_t i = 0; i < nq; ++i) {
                float * row = scores + i * l.kp;
                float top = -INFINITY;
                for (std::int64_t j = 0; j < a.lk; ++j) {
                    row[j] *= a.scale;
                    top = std::max(top, row[j]);
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
                    float * row = scores + i * l.kp;
                    const std::int64_t whole = (j1 - j0) / 64 * 64;
                    if (whole > 0) {
                        sdasm_exponentials_row_avx512(row + j0, whole, maxima[i], sums + i);
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
                    sdasm_amx_pack_rows3(at(weights, r / kTileRows * kb * kBlockBytes),
                                         scores + r * l.kp + j0, ld_scores, std::clamp<std::int64_t>(nq - r, 0, kTileRows),
                                         j1 - j0);
                }
            }
            // The stretch's sum, begun from zeros and added to the output
            // after: the kernel adds its small products first only to sums of
            // its own.
            const Phase_scope scope(Phase::product);
            float * into = j0 == 0 ? output : stretch_sum;
            for (std::int64_t dd = 0; dd < l.vp; dd += kBlock) {
                for (std::int64_t ii = 0; ii < rows; ii += kBlock) {
                    sdasm_amx_single_2x2(kb, at(weights, ii / kTileRows * kb * kBlockBytes), kb * kBlockBytes,
                                         at(values, dd / kTileRows * l.value_tile_bytes + j0 / kBlock * kBlockBytes),
                                         l.value_tile_bytes, into + ii * l.vp + dd, ld_output, 0);
                }
            }
            if (j0 > 0) {
                for (std::int64_t i = 0; i < rows * l.vp; ++i) {
                    output[i] += stretch_sum[i];
                }
            }
        }
        // Each query's output divided by the sum of its exponentials: by the
        // reciprocal of the sum, rounded to single precision.
        const Phase_scope scope(Phase::transform_out);
        char * dst = a.out + b * a.onb3 + h * a.onb1;
        for (std::int64_t i = 0; i < nq; ++i) {
            const auto reciprocal = static_cast<float>(1.0 / sums[i]);
            float * row = reinterpret_cast<float *>(dst + (q0 + i) * a.onb2);
            const float * o = output + i * l.vp;
            for (std::int64_t c = 0; c < a.dv; ++c) {
                row[c] = o[c] * reciprocal;
            }
        }
    }
    if (configured) {
        sdasm_amx_release();
    }
}

}  // namespace

bool attention_supported(const ggml_tensor * node) {
    if (node->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    float max_bias = 0;
    float softcap = 0;
    std::memcpy(&max_bias, &node->op_params[1], sizeof max_bias);
    std::memcpy(&softcap, &node->op_params[2], sizeof softcap);
    const auto kv_type = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16; };
    return node->src[3] == nullptr && node->src[4] == nullptr && max_bias == 0.0f && softcap == 0.0f &&
           q->type == GGML_TYPE_F32 && kv_type(k) && kv_type(v) && q->nb[0] == sizeof(float) &&
           k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
           node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && q->ne[0] == k->ne[0] &&
           k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && q->ne[2] % k->ne[2] == 0 && q->ne[3] == k->ne[3] &&
           q->ne[3] == v->ne[3] && k->ne[1] >= 1;
}

std::size_t attention_work_size(const ggml_tensor * node, int n_threads) {
    const Attention a = geometry(node);
    if (on_amx(a)) {
        const Amx_layout l = amx_layout(a);
        return kAlign + l.shared_bytes + l.thread_bytes * static_cast<std::size_t>(n_threads);
    }
    const Layout l = layout(a);
    return kAlign + l.shared_bytes + l.thread_bytes * static_cast<std::size_t>(n_threads);
}

void attention_compute(const ggml_compute_params * params, ggml_tensor * node) {
    const Attention a = geometry(node);
    if (on_amx(a)) {
        compute_amx(params, a);
        return;
    }
    const Layout l = layout(a);
    char * base = reinterpret_cast<char *>(align_up(reinterpret_cast<std::uintptr_t>(params->wdata)));
    float * shared = reinterpret_cast<float *>(base);
    char * mine = base + l.shared_bytes + l.thread_bytes * static_cast<std::size_t>(params->ith);
    float * queries = reinterpret_cast<float *>(mine);
    float * scores = reinterpret_cast<float *>(mine + l.queries_bytes);
    float * output = reinterpret_cast<float *>(mine + l.queries_bytes + l.scores_bytes);
    float * stretch_sum = reinterpret_cast<float *>(mine + l.queries_bytes + l.scores_bytes + l.output_bytes);
    float * maxima = reinterpret_cast<float *>(mine + l.queries_bytes + l.scores_bytes + 2 * l.output_bytes);
    auto * sums = reinterpret_cast<double *>(mine + l.queries_bytes + l.scores_bytes + 2 * l.output_bytes + l.maxima_bytes);
    const Microkernel uk = microkernel();
    const Scores_kernel sk = scores_kernel();
    const auto keys_of = [&](std::int64_t b, std::int64_t hk) {
        return shared + static_cast<std::size_t>(b * a.hk + hk) * (l.keys_floats + l.values_floats);
    };

    Work_queue queue(params);
    // The keys and the values of every head, an item each.
    const std::int64_t pack_items = 2 * a.b * a.hk;
    int item = queue.next();
    for (; item < pack_items; item = queue.next()) {
        const std::int64_t head = item / 2;
        const std::int64_t b = head / a.hk;
        const std::int64_t hk = head % a.hk;
        float * keys = keys_of(b, hk);
        const Phase_scope scope(Phase::pack_weights);
        if (item % 2 == 0) {
            const char * src = a.k + b * a.knb3 + hk * a.knb2;
            if (a.ktype == GGML_TYPE_F16) {
                pack_rows_f16(keys, reinterpret_cast<const std::uint16_t *>(src), a.knb1, a.lk, 0, a.d, kMr);
            }
            else {
                pack_rows_f32(keys, reinterpret_cast<const float *>(src), a.knb1, a.lk, 0, a.d, kMr);
            }
        }
        else {
            const char * src = a.v + b * a.vnb3 + hk * a.vnb2;
            float * values = keys + l.keys_floats;
            if (a.vtype == GGML_TYPE_F16) {
                pack_values<std::uint16_t>(values, src, a.vnb1, a.lk, a.dv, [](std::uint16_t x) { return _cvtsh_ss(x); });
            }
            else {
                pack_values<float>(values, src, a.vnb1, a.lk, a.dv, [](float x) { return x; });
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
        const float * keys = keys_of(b, hk);
        const float * values = keys + l.keys_floats;

        {
            const Phase_scope scope(Phase::pack_inputs);
            pack_rows_f32(queries, reinterpret_cast<const float *>(a.q + b * a.qnb3 + h * a.qnb2 + q0 * a.qnb1), a.qnb1,
                          nq, 0, a.d, kNr);
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
            const auto product = [&](const float * ka, const float * qb, float * s, bool scaled, float * top) {
                for (std::int64_t e0 = 0; e0 < a.d; e0 += stretch) {
                    const std::int64_t e = std::min(stretch, a.d - e0);
                    if (scaled && e0 + e == a.d) {
                        sk(e, ka + e0 * kMr, qb + e0 * kNr, s, kNr * sizeof(float), top, a.scale, e0 > 0 ? 1 : 0);
                    }
                    else {
                        uk(e, ka + e0 * kMr, qb + e0 * kNr, s, kNr * sizeof(float), e0 > 0 ? 1 : 0);
                    }
                }
            };
            const std::int64_t whole = a.lk / kMr;
            for (std::int64_t kp = 0; kp < whole; ++kp) {
                const float * ka = keys + kp * kMr * a.d;
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    float * s = scores + qp * lk_padded * kNr + kp * kMr * kNr;
                    product(ka, queries + qp * a.d * kNr, s, true, maxima + qp * kNr);
                }
            }
            // The keys past the last kMr: their panel by the micro-kernel, and
            // then their scale and their maximum, which the zeros past them in
            // the panel have no part in.
            if (whole < panels_keys) {
                const float * ka = keys + whole * kMr * a.d;
                const std::int64_t rows = a.lk - whole * kMr;
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    float * s = scores + qp * lk_padded * kNr + whole * kMr * kNr;
                    product(ka, queries + qp * a.d * kNr, s, false, nullptr);
                    for (std::int64_t i = 0; i < rows; ++i) {
                        float * p = s + i * kNr;
                        for (int lane = 0; lane < kNr; ++lane) {
                            const float x = p[lane] * a.scale;
                            p[lane] = x;
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
                    exponentials(scores + qp * lk_padded * kNr, j0, j1, maxima + qp * kNr, sums + qp * kNr);
                }
            }
            // The sum over the keys in three steps: a.sum_keys keys at a time in
            // the registers, those sums added together over the stretch, and
            // the stretch's sum added to the output: no sum runs long enough
            // to round more than ggml's dot product, which sums 32 at a time
            // and then the 32 sums.
            const Phase_scope scope(Phase::product);
            float * into = j0 == 0 ? output : stretch_sum;
            const std::int64_t ldc = a.queries * static_cast<std::int64_t>(sizeof(float));
            for (std::int64_t dp = 0; dp < panels_dims; ++dp) {
                for (std::int64_t qp = 0; qp < qpanels; ++qp) {
                    float * o = into + dp * kMr * a.queries + qp * kNr;
                    for (std::int64_t k0 = j0; k0 < j1; k0 += a.sum_keys) {
                        const float * va = values + dp * a.lk * kMr + k0 * kMr;
                        const float * pb = scores + qp * lk_padded * kNr + k0 * kNr;
                        uk(std::min(a.sum_keys, j1 - k0), va, pb, o, ldc, k0 > j0 ? 1 : 0);
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
        char * dst = a.out + b * a.onb3 + h * a.onb1;
        for (std::int64_t i = 0; i < nq; ++i) {
            const auto reciprocal = static_cast<float>(1.0 / sums[i]);
            float * row = reinterpret_cast<float *>(dst + (q0 + i) * a.onb2);
            for (std::int64_t c = 0; c < a.dv; ++c) {
                row[c] = output[c * a.queries + i] * reciprocal;
            }
        }
    }
}

}  // namespace sdasm
