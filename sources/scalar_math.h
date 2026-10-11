// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The functions of single precision that the C++ kernels apply element by
// element, written for the compiler to vectorize: no branches, no calls into
// the C library, nothing but arithmetic, std::fma and the bits of a float,
// which a vectorizer turns into vector instructions of whatever kind the target
// has (AVX2, AVX-512, NEON) and, inlined into a loop over elements, does
// eight or sixteen or four at a time. The kernels written by hand (x86_64/) have
// their own, tuned to AVX2.

#ifndef SDASM_SCALAR_MATH_H
#define SDASM_SCALAR_MATH_H

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace sdasm {

// exp(x) for x in [-87.3, 88.7], and its limits beyond: x is clamped to that
// range, below which exp(x) is under the least normal number (and taken as it)
// and above which it is the infinity it rounds to. x = n ln(2) + r, n = x / ln(2)
// rounded to the nearest -- the sum with 1.5 * 2^23 rounds it -- and r reduced
// in two steps of a high and a low part of ln(2), |r| <= ln(2) / 2; exp(r) by
// the minimax polynomial of Cephes' expf, and 2^n built in the exponent of a
// float. Within two units in the last place over the range.
inline float exponential(float x) {
    x = std::min(std::max(x, -87.3f), 88.73f);
    const float magic = 0x1.8p23f;
    const float z = std::fma(x, 1.44269504088896341f, magic);
    const float n = z - magic;
    float r = std::fma(n, -0.693359375f, x);
    r = std::fma(n, 2.12194440e-4f, r);
    float p = 1.9875691500e-4f;
    p = std::fma(p, r, 1.3981999507e-3f);
    p = std::fma(p, r, 8.3334519073e-3f);
    p = std::fma(p, r, 4.1665795894e-2f);
    p = std::fma(p, r, 1.6666665459e-1f);
    p = std::fma(p, r, 5.0000001201e-1f);
    p = std::fma(p * r, r, r) + 1.0f;
    // 2^n: n + 127 in the exponent's place; n = 128 makes the infinity.
    const auto e = static_cast<std::uint32_t>(std::bit_cast<std::int32_t>(z) - 0x4b400000 + 127) << 23;
    return p * std::bit_cast<float>(e);
}

// SiLU, x / (1 + exp(-x)).
inline float silu(float x) {
    return x / (1.0f + exponential(-x));
}

// GELU by its definition with tanh, 0.5 x (1 + tanh(y)) with
// y = sqrt(2 / pi) (x + 0.044715 x^3), as x / (1 + exp(-2 y)).
inline float gelu(float x) {
    const float y = x * std::fma(0.044715f * x, x, 1.0f);
    return x / (1.0f + exponential(-1.5957691216057308f * y));
}

}  // namespace sdasm

#endif
