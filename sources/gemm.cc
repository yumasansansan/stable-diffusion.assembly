// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

#include "gemm.h"
#include "kernels.h"
#include "phases.h"

#include <algorithm>
#include <bit>
#include <cstring>

#include "ggml-cpu-impl.h"

namespace sdasm {

namespace {

// Rows [r0, r0 + rows) of a matrix into panels of p rows, [panel][k1 - k0][p]:
// for each k, the p numbers of a panel side by side, zeros for the rows past
// the end. Each panel is read a row at a time, along k, and written a column at
// a time; the compiler vectorizes the reading. With the kernels written by hand
// for AVX2, a whole panel of six or sixteen rows is transposed by them (x86_64/
// pack_avx2.S), eight numbers of each row at a time, and only what is left
// past the last eight of k here.
template <class T, class Widen, class Hand>
void pack(float * dst, const T * src, std::int64_t stride, std::int64_t rows, std::int64_t k0, std::int64_t k1, int p,
          Widen widen, Hand hand) {
    const std::int64_t kc = k1 - k0;
    const char * base = reinterpret_cast<const char *>(src);
    for (std::int64_t r0 = 0; r0 < rows; r0 += p, dst += kc * p) {
        const std::int64_t valid = std::min<std::int64_t>(p, rows - r0);
        std::int64_t done = 0;
        if (valid == p) {
            done = hand(dst, reinterpret_cast<const T *>(base + r0 * stride) + k0, kc);
        }
        for (std::int64_t i = 0; i < valid; ++i) {
            const T * row = reinterpret_cast<const T *>(base + (r0 + i) * stride) + k0;
            for (std::int64_t k = done; k < kc; ++k) {
                dst[k * p + i] = widen(row[k]);
            }
        }
        for (std::int64_t i = valid; i < p; ++i) {
            for (std::int64_t k = done; k < kc; ++k) {
                dst[k * p + i] = 0.0f;
            }
        }
    }
}

}  // namespace

void pack_rows_f32(float * dst, const float * src, std::int64_t stride, std::int64_t rows, std::int64_t k0,
                   std::int64_t k1, int p) {
    pack(dst, src, stride, rows, k0, k1, p, [](float x) { return x; },
         [&](float * d, const float * s, std::int64_t kc) -> std::int64_t {
#if SDASM_X86_64
             if (avx2_kernels() && (p == 16 || p == 6)) {
                 const std::int64_t whole = kc / 8 * 8;
                 (p == 16 ? sdasm_pack16_f32_avx2 : sdasm_pack6_f32_avx2)(d, s, stride, whole);
                 return whole;
             }
#endif
             (void) d;
             (void) s;
             (void) kc;
             return 0;
         });
}

void pack_rows_f16(float * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t rows,
                   std::int64_t k0, std::int64_t k1, int p) {
    pack(dst, src, stride, rows, k0, k1, p, [](std::uint16_t h) { return static_cast<float>(std::bit_cast<_Float16>(h)); },
         [&](float * d, const std::uint16_t * s, std::int64_t kc) -> std::int64_t {
#if SDASM_X86_64
             if (avx2_kernels() && (p == 16 || p == 6)) {
                 const std::int64_t whole = kc / 8 * 8;
                 (p == 16 ? sdasm_pack16_f16_avx2 : sdasm_pack6_f16_avx2)(d, s, stride, whole);
                 return whole;
             }
#endif
             (void) d;
             (void) s;
             (void) kc;
             return 0;
         });
}

void tile_part(Microkernel uk, std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
               bool accumulate, std::int64_t rows, std::int64_t cols) {
    alignas(32) float t[kMr * kNr];
    uk(kc, a, b, t, kNr * sizeof(float), 0);
    for (std::int64_t i = 0; i < rows; ++i) {
        float * ci = reinterpret_cast<float *>(reinterpret_cast<char *>(c) + i * ldc);
        for (std::int64_t j = 0; j < cols; ++j) {
            ci[j] = accumulate ? ci[j] + t[i * kNr + j] : t[i * kNr + j];
        }
    }
}

void split_block(const Split_form & form, std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next,
                 const std::uint16_t * b, std::int64_t b_next, float * c, std::int64_t ldc, bool accumulate,
                 std::int64_t rows, std::int64_t cols, float * spare) {
    const Split_kernel kernel = form.kernel;
    if (rows == kAmxBlock && cols == kAmxBlock) {
        kernel(blocks, a, a_next, b, b_next, c, ldc, accumulate ? 1 : 0);
        return;
    }
    const auto row = [&](std::int64_t i) { return reinterpret_cast<float *>(reinterpret_cast<char *>(c) + i * ldc); };
    if (accumulate) {
        for (std::int64_t i = 0; i < kAmxBlock; ++i) {
            for (std::int64_t j = 0; j < kAmxBlock; ++j) {
                spare[i * kAmxBlock + j] = i < rows && j < cols ? row(i)[j] : 0.0f;
            }
        }
    }
    kernel(blocks, a, a_next, b, b_next, spare, kAmxBlock * sizeof(float), accumulate ? 1 : 0);
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            row(i)[j] = spare[i * kAmxBlock + j];
        }
    }
}

std::vector<Block> guided_blocks(std::int64_t segments, std::int64_t length, std::int64_t align, std::int64_t largest,
                                 int threads) {
    std::vector<Block> blocks;
    std::int64_t left = segments * length;
    const std::int64_t share = 2 * static_cast<std::int64_t>(threads);
    for (std::int64_t s = 0; s < segments; ++s) {
        for (std::int64_t start = 0; start < length;) {
            std::int64_t size = tuning().guided != 0 ? std::max(align, (left / share) / align * align) : largest;
            size = std::min({size, largest, length - start});
            blocks.push_back({s, start, size});
            start += size;
            left -= size;
        }
    }
    return blocks;
}

Work_queue::Work_queue(const ggml_compute_params * params) : params_(params) {
    if (params->ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, params->nth);
    }
    ggml_barrier(params->threadpool);
}

int Work_queue::next() {
    if (first_) {
        first_ = false;
        return params_->ith;
    }
    return ggml_threadpool_chunk_add(params_->threadpool, 1);
}

void Work_queue::barrier() const {
    const Phase_scope scope(Phase::wait);
    ggml_barrier(params_->threadpool);
}

}  // namespace sdasm
