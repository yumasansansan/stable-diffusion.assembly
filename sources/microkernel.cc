// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The micro-kernels of the matrix product in plain C++: the tile of kMr rows by
// kNr columns kept in an array the compiler keeps in registers, summed over k
// by the product of a number of A and a row of B -- a loop of kNr over
// contiguous floats, which it vectorizes for the target, a register or two of
// them for each row. What the compiler makes of it is the yardstick the
// kernels written by hand (x86_64/gemm_avx2.S) are measured against, and the
// kernels the products are computed with where those are not (cpu.cc).

#include <algorithm>
#include <array>
#include <cstdint>

#include "gemm.h"
#include "kernels.h"

namespace sdasm {

namespace {

// The operands of a kernel of kc, as the kernels written by hand take them:
// panels of kMr and kNr numbers over kc, and C, kMr rows ldc bytes apart.
struct Operands {
    Elements<const float> a, b;
    Elements<float> c;
    std::int64_t ldc;  // elements
};

Operands operands(std::int64_t kc, const float * const a, const float * const b, float * const c, std::int64_t ldc) {
    SDASM_REQUIRE(kc >= 0 && ldc % bytes_of<float>(1) == 0 && ldc >= bytes_of<float>(kNr));
    const std::int64_t row = ldc / bytes_of<float>(1);
    // The kernels' operands, handed over as pointers.
#pragma clang unsafe_buffer_usage begin
    return {elements_at(a, kc * kMr), elements_at(b, kc * kNr), elements_at(c, (kMr - 1) * row + kNr), row};
#pragma clang unsafe_buffer_usage end
}

}  // namespace

// Each kernel sums its tile in an array of its own, which the compiler can
// keep in registers only when the array is the function's and its loops over
// rows and columns have bounds it knows: the sum is written out in each.

void gemm_6x16_cxx(std::int64_t kc, const float * const a, const float * const b, float * const c, std::int64_t ldc,
                   std::int64_t accumulate) {
    const Operands o = operands(kc, a, b, c, ldc);
    std::array<std::array<float, kNr>, kMr> t{};
    for (std::int64_t k = 0; k < kc; ++k) {
        for (std::size_t i = 0; i < kMr; ++i) {
            for (std::size_t j = 0; j < kNr; ++j) {
                t[i][j] += o.a[k * kMr + static_cast<std::int64_t>(i)] * o.b[k * kNr + static_cast<std::int64_t>(j)];
            }
        }
    }
    for (std::size_t i = 0; i < kMr; ++i) {
        const Elements<float> r = o.c.part(static_cast<std::int64_t>(i) * o.ldc, kNr);
        for (std::size_t j = 0; j < kNr; ++j) {
            const auto jj = static_cast<std::int64_t>(j);
            r[jj] = accumulate != 0 ? r[jj] + t[i][j] : t[i][j];
        }
    }
}

void scores_6x16_cxx(std::int64_t kc, const float * const a, const float * const b, float * const c, std::int64_t ldc,
                     float * const max, float scale, std::int64_t accumulate) {
    const Operands o = operands(kc, a, b, c, ldc);
    std::array<std::array<float, kNr>, kMr> t{};
    for (std::int64_t k = 0; k < kc; ++k) {
        for (std::size_t i = 0; i < kMr; ++i) {
            for (std::size_t j = 0; j < kNr; ++j) {
                t[i][j] += o.a[k * kMr + static_cast<std::int64_t>(i)] * o.b[k * kNr + static_cast<std::int64_t>(j)];
            }
        }
    }
    // The largest so far in an array of the function's own, which no store to
    // c can be taken to change.
#pragma clang unsafe_buffer_usage begin
    const Elements<float> largest = elements_at(max, kNr);
#pragma clang unsafe_buffer_usage end
    std::array<float, kNr> m{};
    for (std::size_t j = 0; j < kNr; ++j) {
        m[j] = largest[static_cast<std::int64_t>(j)];
    }
    for (std::size_t i = 0; i < kMr; ++i) {
        const Elements<float> r = o.c.part(static_cast<std::int64_t>(i) * o.ldc, kNr);
        for (std::size_t j = 0; j < kNr; ++j) {
            const auto jj = static_cast<std::int64_t>(j);
            const float s = (accumulate != 0 ? r[jj] + t[i][j] : t[i][j]) * scale;
            r[jj] = s;
            m[j] = std::max(m[j], s);
        }
    }
    for (std::size_t j = 0; j < kNr; ++j) {
        largest[static_cast<std::int64_t>(j)] = m[j];
    }
}

}  // namespace sdasm
