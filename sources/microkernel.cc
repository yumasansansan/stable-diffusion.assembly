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
#include <cstdint>

#include "gemm.h"
#include "kernels.h"

namespace sdasm {

namespace {

float * row(float * c, std::int64_t ldc, int i) {
    return reinterpret_cast<float *>(reinterpret_cast<char *>(c) + i * ldc);
}

}  // namespace

// Each kernel sums its tile in an array of its own, which the compiler can
// keep in registers only when the array is the function's and its loops over
// rows and columns have bounds it knows: the sum is written out in each.

void gemm_6x16_cxx(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                   std::int64_t accumulate) {
    float t[kMr][kNr] = {};
    for (std::int64_t k = 0; k < kc; ++k) {
        for (int i = 0; i < kMr; ++i) {
            for (int j = 0; j < kNr; ++j) {
                t[i][j] += a[k * kMr + i] * b[k * kNr + j];
            }
        }
    }
    for (int i = 0; i < kMr; ++i) {
        float * r = row(c, ldc, i);
        for (int j = 0; j < kNr; ++j) {
            r[j] = accumulate != 0 ? r[j] + t[i][j] : t[i][j];
        }
    }
}

void scores_6x16_cxx(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc, float * max,
                     float scale, std::int64_t accumulate) {
    float t[kMr][kNr] = {};
    for (std::int64_t k = 0; k < kc; ++k) {
        for (int i = 0; i < kMr; ++i) {
            for (int j = 0; j < kNr; ++j) {
                t[i][j] += a[k * kMr + i] * b[k * kNr + j];
            }
        }
    }
    // The largest so far in an array of the function's own, which no store to
    // c can be taken to change.
    float m[kNr];
    for (int j = 0; j < kNr; ++j) {
        m[j] = max[j];
    }
    for (int i = 0; i < kMr; ++i) {
        float * r = row(c, ldc, i);
        for (int j = 0; j < kNr; ++j) {
            const float s = (accumulate != 0 ? r[j] + t[i][j] : t[i][j]) * scale;
            r[j] = s;
            m[j] = std::max(m[j], s);
        }
    }
    for (int j = 0; j < kNr; ++j) {
        max[j] = m[j];
    }
}

}  // namespace sdasm
