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
#include <array>
#include <bit>

#include "ggml-cpu-impl.h"

namespace sdasm {

namespace {

// Rows [0, rows) of a matrix into panels of p rows, [panel][k1 - k0][p]: for
// each k, the p numbers of a panel side by side, zeros for the rows past the
// end. Each panel is read a row at a time, along k, and written a column at a
// time; the compiler vectorizes the reading. With the kernels written by hand
// for AVX2, a whole panel of six or sixteen rows is transposed by them (x86_64/
// pack_avx2.S), eight numbers of each row at a time, and only what is left
// past the last eight of k here.
template <class T, class Widen, class Hand>
void pack(Elements<float> dst, Elements<const T> src, std::int64_t stride, std::int64_t rows, std::int64_t k0,
          std::int64_t k1, int p, Widen widen, Hand hand) {
    const std::int64_t kc = k1 - k0;
    for (std::int64_t r0 = 0; r0 < rows; r0 += p) {
        const Elements<float> panel = dst.part(r0 / p * kc * p, kc * p);
        const std::int64_t valid = std::min<std::int64_t>(p, rows - r0);
        std::int64_t done = 0;
        if (valid == p) {
            done = hand(panel, src.from(r0 * stride + k0), kc);
        }
        for (std::int64_t i = 0; i < valid; ++i) {
            const Elements<const T> row = src.part((r0 + i) * stride + k0, kc);
            for (std::int64_t k = done; k < kc; ++k) {
                panel[k * p + i] = widen(row[k]);
            }
        }
        for (std::int64_t i = valid; i < p; ++i) {
            for (std::int64_t k = done; k < kc; ++k) {
                panel[k * p + i] = 0.0f;
            }
        }
    }
}

}  // namespace

void pack_rows_f32(Elements<float> dst, Elements<const float> src, std::int64_t stride, std::int64_t rows, std::int64_t k0,
                   std::int64_t k1, int p) {
    pack(dst, src, stride, rows, k0, k1, p, [](float x) { return x; },
         [&](Elements<float> d, Elements<const float> s, std::int64_t kc) -> std::int64_t {
#if SDASM_X86_64
             if (avx2_kernels() && (p == 16 || p == 6)) {
                 // Eight numbers of each of the p rows at a time.
                 const std::int64_t whole = kc / 8 * 8;
                 SDASM_CHECK(d.size() >= whole * p && (whole == 0 || s.size() >= (p - 1) * stride + whole));
                 (p == 16 ? sdasm_pack16_f32_avx2 : sdasm_pack6_f32_avx2)(d.data(), s.data(), bytes_of<float>(stride),
                                                                          whole);
                 return whole;
             }
#endif
             (void) d;
             (void) s;
             (void) kc;
             return 0;
         });
}

void pack_rows_f16(Elements<float> dst, Elements<const std::uint16_t> src, std::int64_t stride, std::int64_t rows,
                   std::int64_t k0, std::int64_t k1, int p) {
    pack(dst, src, stride, rows, k0, k1, p, [](std::uint16_t h) { return static_cast<float>(std::bit_cast<_Float16>(h)); },
         [&](Elements<float> d, Elements<const std::uint16_t> s, std::int64_t kc) -> std::int64_t {
#if SDASM_X86_64
             if (avx2_kernels() && (p == 16 || p == 6)) {
                 const std::int64_t whole = kc / 8 * 8;
                 SDASM_CHECK(d.size() >= whole * p && (whole == 0 || s.size() >= (p - 1) * stride + whole));
                 (p == 16 ? sdasm_pack16_f16_avx2 : sdasm_pack6_f16_avx2)(d.data(), s.data(),
                                                                          bytes_of<std::uint16_t>(stride), whole);
                 return whole;
             }
#endif
             (void) d;
             (void) s;
             (void) kc;
             return 0;
         });
}

void tile_part(Microkernel uk, std::int64_t kc, Elements<const float> a, Elements<const float> b, Elements<float> c,
               std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols) {
    std::array<float, kMr * kNr> t{};
    SDASM_CHECK(a.size() >= kc * kMr && b.size() >= kc * kNr);
    uk(kc, a.data(), b.data(), t.data(), bytes_of<float>(kNr), 0);
    const Elements<const float> tile = t;
    for (std::int64_t i = 0; i < rows; ++i) {
        const Elements<float> ci = c.part(i * ldc, cols);
        for (std::int64_t j = 0; j < cols; ++j) {
            ci[j] = accumulate ? ci[j] + tile[i * kNr + j] : tile[i * kNr + j];
        }
    }
}

void split_block(const Split_form & form, std::int64_t blocks, Elements<const std::uint16_t> a, std::int64_t a_next,
                 Elements<const std::uint16_t> b, std::int64_t b_next, Elements<float> c, std::int64_t ldc, bool accumulate,
                 std::int64_t rows, std::int64_t cols, Elements<float> spare) {
    const Split_kernel kernel = form.kernel;
    // Each of the two tiles of A and of B: a block of k of a tile is a_bytes,
    // or b_bytes, of elements of two bytes.
    SDASM_CHECK(a.size() >= a_next + blocks * form.a_bytes / 2 && b.size() >= b_next + blocks * form.b_bytes / 2);
    if (rows == kAmxBlock && cols == kAmxBlock) {
        SDASM_CHECK(c.size() >= (kAmxBlock - 1) * ldc + kAmxBlock);
        kernel(blocks, a.data(), bytes_of<std::uint16_t>(a_next), b.data(), bytes_of<std::uint16_t>(b_next), c.data(),
               bytes_of<float>(ldc), accumulate ? 1 : 0);
        return;
    }
    if (accumulate) {
        for (std::int64_t i = 0; i < kAmxBlock; ++i) {
            for (std::int64_t j = 0; j < kAmxBlock; ++j) {
                spare[i * kAmxBlock + j] = i < rows && j < cols ? c[i * ldc + j] : 0.0f;
            }
        }
    }
    SDASM_CHECK(spare.size() >= kAmxBlock * kAmxBlock);
    kernel(blocks, a.data(), bytes_of<std::uint16_t>(a_next), b.data(), bytes_of<std::uint16_t>(b_next), spare.data(),
           bytes_of<float>(kAmxBlock), accumulate ? 1 : 0);
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            c[i * ldc + j] = spare[i * kAmxBlock + j];
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

Work_queue::Work_queue(const ggml_compute_params * const params) : params_(params) {
    meet();
}

Work_queue::Work_queue(const ggml_compute_params * const params, Work_buffer & buffer) : params_(params) {
    meet();
    buffer.met();
}

void Work_queue::meet() const {
    if (params_->ith == 0) {
        ggml_threadpool_chunk_set(params_->threadpool, params_->nth);
    }
    ggml_barrier(params_->threadpool);
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
