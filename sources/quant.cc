// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The quantized products of quant.h in plain C++: the micro-kernels, a loop
// over the four bytes of a quad of each row and column for the compiler to
// vectorize, and the packing of inputs and weights, rounding as the vector
// units round, to the nearest and to even on a tie.

#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kernels.h"

#include "ggml.h"

namespace sdasm {

bool quantized(Quantized_op op) {
    const Tuning & t = tuning();
    return t.bits >= 2 && t.bits <= 8 && (t.quantize & static_cast<int>(op)) != 0;
}

Quant quant() {
    const Tuning & t = tuning();
    Quant q{};
    q.quads = std::clamp(t.group / 4, 1, 256);
    q.input_levels = (1 << std::clamp(t.input_bits, 2, 8)) - 1;
    q.weight_max = (1 << (std::clamp(t.bits, 2, 8) - 1)) - 1;
    q.weight_groups = t.weight_group != 0;
    return q;
}

namespace {

float * row_of(float * c, std::int64_t ldc, int i) {
    return reinterpret_cast<float *>(reinterpret_cast<char *>(c) + i * ldc);
}

// The kernel's tile in an array of its own, the group's sums in another; the
// bytes of A read as unsigned when AUnsigned, those of B as signed, and the
// other way about otherwise.
template <bool AUnsigned>
void qgemm(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b, float * c, std::int64_t ldc,
           std::int64_t accumulate) {
    float t[kMr][kNr];
    for (int i = 0; i < kMr; ++i) {
        const float * r = row_of(c, ldc, i);
        for (int j = 0; j < kNr; ++j) {
            t[i][j] = accumulate != 0 ? r[j] : 0.0f;
        }
    }
    const std::int64_t a_bytes = qgroup_bytes(quads, kMr);
    const std::int64_t b_bytes = qgroup_bytes(quads, kNr);
    for (std::int64_t group = 0; group < groups; ++group, a += a_bytes, b += b_bytes) {
        std::int32_t sum[kMr][kNr] = {};
        for (std::int64_t q = 0; q < quads; ++q) {
            const std::uint8_t * qa = a + q * 4;
            const std::uint8_t * qb = b + q * kNr * 4;
            for (int i = 0; i < kMr; ++i) {
                for (int j = 0; j < kNr; ++j) {
                    for (int u = 0; u < 4; ++u) {
                        const std::uint8_t ai = qa[i * quads * 4 + u];
                        const std::int32_t x = AUnsigned ? static_cast<std::int32_t>(ai)
                                                         : static_cast<std::int32_t>(static_cast<std::int8_t>(ai));
                        const std::int32_t y = AUnsigned ? static_cast<std::int32_t>(static_cast<std::int8_t>(qb[j * 4 + u]))
                                                         : static_cast<std::int32_t>(qb[j * 4 + u]);
                        sum[i][j] += x * y;
                    }
                }
            }
        }
        float sa[kMr], oa[kMr], sb[kNr], ob[kNr];
        std::memcpy(sa, a + quads * kMr * 4, sizeof sa);
        std::memcpy(oa, a + quads * kMr * 4 + sizeof sa, sizeof oa);
        std::memcpy(sb, b + quads * kNr * 4, sizeof sb);
        std::memcpy(ob, b + quads * kNr * 4 + sizeof sb, sizeof ob);
        for (int i = 0; i < kMr; ++i) {
            for (int j = 0; j < kNr; ++j) {
                t[i][j] = std::fma(sa[i] * sb[j], static_cast<float>(sum[i][j]), t[i][j]);
                t[i][j] = std::fma(-oa[i], ob[j], t[i][j]);
            }
        }
    }
    for (int i = 0; i < kMr; ++i) {
        float * r = row_of(c, ldc, i);
        for (int j = 0; j < kNr; ++j) {
            r[j] = t[i][j];
        }
    }
}

// The scale and the offset of a group of inputs and their bytes: those of
// the group's least and greatest numbers, 0 between them, levels steps apart.
void quantize_input_group(std::uint8_t * dst, std::int64_t step, const float * x, std::int64_t n, int levels, float & scale,
                          float & offset) {
    float lo = 0.0f, hi = 0.0f;
    for (std::int64_t e = 0; e < n; ++e) {
        lo = std::min(lo, x[e]);
        hi = std::max(hi, x[e]);
    }
    const float s = (hi - lo) / static_cast<float>(levels);
    const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
    const float zero = std::nearbyint(-lo * inverse);
    for (std::int64_t e = 0; e < n; ++e) {
        const float q = std::clamp(std::nearbyint(x[e] * inverse) + zero, 0.0f, static_cast<float>(levels));
        dst[e / 4 * step + e % 4] = static_cast<std::uint8_t>(q);
    }
    scale = s;
    offset = s * zero;
}

// The weights of a group, at the scale given, and their sum.
std::int32_t quantize_weight_group(std::uint8_t * dst, std::int64_t step, const float * w, std::int64_t n, float inverse,
                                   int max) {
    std::int32_t sum = 0;
    for (std::int64_t e = 0; e < n; ++e) {
        const auto q = static_cast<std::int32_t>(std::clamp(std::nearbyint(w[e] * inverse), static_cast<float>(-max),
                                                            static_cast<float>(max)));
        dst[e / 4 * step + e % 4] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        sum += q;
    }
    return sum;
}

template <class Load>
void quantize_weights(std::uint8_t * dst, std::int64_t rows, std::int64_t k, int p, const Quant & q, Load load) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    std::memset(dst, 0, static_cast<std::size_t>(groups * bytes));
    float w[1024];
    for (std::int64_t r = 0; r < rows; ++r) {
        // The scale of the whole row, unless each group has its own.
        float row_max = 0.0f;
        if (!q.weight_groups) {
            for (std::int64_t e = 0; e < k; ++e) {
                row_max = std::max(row_max, std::fabs(load(r, e)));
            }
        }
        for (std::int64_t gi = 0; gi < groups; ++gi) {
            const std::int64_t n = std::min(group, k - gi * group);
            float m = row_max;
            for (std::int64_t e = 0; e < group; ++e) {
                w[e] = e < n ? load(r, gi * group + e) : 0.0f;
                if (q.weight_groups) {
                    m = std::max(m, std::fabs(w[e]));
                }
            }
            const float s = m / static_cast<float>(q.weight_max);
            const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
            std::uint8_t * g = dst + gi * bytes;
            const std::int32_t sum = p == kMr ? quantize_weight_group(g + r * group, 4, w, group, inverse, q.weight_max)
                                              : quantize_weight_group(g + r * 4, p * 4, w, group, inverse, q.weight_max);
            const float offset = s * static_cast<float>(sum);
            std::memcpy(g + q.quads * p * 4 + r * static_cast<std::int64_t>(sizeof(float)), &s, sizeof s);
            std::memcpy(g + q.quads * p * 4 + (p + r) * static_cast<std::int64_t>(sizeof(float)), &offset, sizeof offset);
        }
    }
}

}  // namespace

void qgemm_6x16_au_cxx(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b, float * c,
                       std::int64_t ldc, std::int64_t accumulate) {
    qgemm<true>(groups, quads, a, b, c, ldc, accumulate);
}

void qgemm_6x16_bu_cxx(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b, float * c,
                       std::int64_t ldc, std::int64_t accumulate) {
    qgemm<false>(groups, quads, a, b, c, ldc, accumulate);
}

void qtile_part(Qkernel kernel, std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                float * c, std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols) {
    alignas(64) float tile[kMr * kNr];
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            tile[i * kNr + j] = accumulate ? row_of(c, ldc, static_cast<int>(i))[j] : 0.0f;
        }
    }
    kernel(groups, quads, a, b, tile, kNr * static_cast<std::int64_t>(sizeof(float)), accumulate ? 1 : 0);
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            row_of(c, ldc, static_cast<int>(i))[j] = tile[i * kNr + j];
        }
    }
}

namespace {

// A row's group of n inputs, n at most `group`, into its bytes, contiguous,
// the group's scale and offset returned: the least and greatest a lane of
// sixteen at a time, so that the compiler makes vector minima and maxima of
// them with no reduction to order, then those of the lanes.
void quantize_input_row(std::uint8_t * dst, const float * x, std::int64_t n, std::int64_t group, int levels, float & scale,
                        float & offset) {
    float lo[kNr] = {}, hi[kNr] = {};
    std::int64_t e = 0;
    for (; e + kNr <= n; e += kNr) {
        for (int l = 0; l < kNr; ++l) {
            lo[l] = std::min(lo[l], x[e + l]);
            hi[l] = std::max(hi[l], x[e + l]);
        }
    }
    for (; e < n; ++e) {
        lo[0] = std::min(lo[0], x[e]);
        hi[0] = std::max(hi[0], x[e]);
    }
    float least = 0.0f, greatest = 0.0f;
    for (int l = 0; l < kNr; ++l) {
        least = std::min(least, lo[l]);
        greatest = std::max(greatest, hi[l]);
    }
    const float s = (greatest - least) / static_cast<float>(levels);
    const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
    const float zero = std::nearbyint(-least * inverse);
    const auto top = static_cast<float>(levels);
    for (e = 0; e < n; ++e) {
        dst[e] = static_cast<std::uint8_t>(std::clamp(std::nearbyint(x[e] * inverse) + zero, 0.0f, top));
    }
    // The rest of the group stands for zeros: the zero point.
    for (; e < group; ++e) {
        dst[e] = static_cast<std::uint8_t>(zero);
    }
    scale = s;
    offset = s * zero;
}

}  // namespace

void quantize_inputs(std::uint8_t * dst, const float * src, std::int64_t stride, std::int64_t rows, std::int64_t k, int p,
                     const Quant & q) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    if (p == kMr) {
        // A's rows, each group's bytes in one piece; the rows past `rows`,
        // zeros of no scale.
        for (std::int64_t gi = 0; gi < groups; ++gi) {
            std::uint8_t * g = dst + gi * bytes;
            float scale[kMr] = {}, offset[kMr] = {};
            for (std::int64_t r = 0; r < kMr; ++r) {
                if (r >= rows) {
                    std::memset(g + r * group, 0, static_cast<std::size_t>(group));
                    continue;
                }
                const float * line = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src) + r * stride);
                quantize_input_row(g + r * group, line + gi * group, std::min(group, k - gi * group), group, q.input_levels,
                                   scale[r], offset[r]);
            }
            std::memcpy(g + q.quads * kMr * 4, scale, sizeof scale);
            std::memcpy(g + q.quads * kMr * 4 + sizeof scale, offset, sizeof offset);
        }
        return;
    }
    std::memset(dst, 0, static_cast<std::size_t>(groups * bytes));
    float x[1024];
    for (std::int64_t r = 0; r < rows; ++r) {
        const float * line = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src) + r * stride);
        for (std::int64_t gi = 0; gi < groups; ++gi) {
            const std::int64_t n = std::min(group, k - gi * group);
            for (std::int64_t e = 0; e < group; ++e) {
                x[e] = e < n ? line[gi * group + e] : 0.0f;
            }
            std::uint8_t * g = dst + gi * bytes;
            float scale = 0.0f, offset = 0.0f;
            // A's rows lie one after another, B's columns quad by quad.
            if (p == kMr) {
                quantize_input_group(g + r * group, 4, x, group, q.input_levels, scale, offset);
            }
            else {
                quantize_input_group(g + r * 4, p * 4, x, group, q.input_levels, scale, offset);
            }
            std::memcpy(g + q.quads * p * 4 + r * static_cast<std::int64_t>(sizeof(float)), &scale, sizeof scale);
            std::memcpy(g + q.quads * p * 4 + (p + r) * static_cast<std::int64_t>(sizeof(float)), &offset, sizeof offset);
        }
    }
}

void quantize_weights_f16(std::uint8_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t rows,
                          std::int64_t k, int p, const Quant & q) {
    quantize_weights(dst, rows, k, p, q, [&](std::int64_t r, std::int64_t e) {
        const auto * line = reinterpret_cast<const ggml_fp16_t *>(reinterpret_cast<const char *>(src) + r * stride);
        return ggml_fp16_to_fp32(line[e]);
    });
}

void quantize_weights_f32(std::uint8_t * dst, const float * src, std::int64_t stride, std::int64_t rows, std::int64_t k,
                          int p, const Quant & q) {
    quantize_weights(dst, rows, k, p, q, [&](std::int64_t r, std::int64_t e) {
        return reinterpret_cast<const float *>(reinterpret_cast<const char *>(src) + r * stride)[e];
    });
}

// The kernels written by hand where the processor has VNNI at the level in
// effect: AVX512-VNNI's at avx512 and above, AVX-VNNI's at avx2.
void quantize_panel_inputs(std::uint8_t * dst, const float * src, std::int64_t k, int p, const Quant & q) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    const auto levels = static_cast<float>(q.input_levels);
    std::int64_t gi = 0;
#if SDASM_X86_64
    // The whole groups of a panel of B by the kernel written by hand.
    if (p == kNr && avx2_kernels()) {
        gi = k / group;
        sdasm_quantize16_avx2(dst, src, gi, q.quads, levels);
    }
#endif
    for (; gi < groups; ++gi) {
        const std::int64_t n = std::min(group, k - gi * group);
        const float * x = src + gi * group * p;
        std::uint8_t * g = dst + gi * bytes;
        // Each lane's least and greatest, 0 between them: the zero point.
        float lo[kNr], hi[kNr], inverse[kNr], zero[kNr];
        for (int l = 0; l < p; ++l) {
            lo[l] = 0.0f;
            hi[l] = 0.0f;
        }
        for (std::int64_t e = 0; e < n; ++e) {
            for (int l = 0; l < p; ++l) {
                lo[l] = std::min(lo[l], x[e * p + l]);
                hi[l] = std::max(hi[l], x[e * p + l]);
            }
        }
        float scale[kNr], offset[kNr];
        for (int l = 0; l < p; ++l) {
            scale[l] = (hi[l] - lo[l]) / levels;
            inverse[l] = scale[l] > 0.0f ? 1.0f / scale[l] : 0.0f;
            zero[l] = std::nearbyint(-lo[l] * inverse[l]);
            offset[l] = scale[l] * zero[l];
        }
        for (std::int64_t e = 0; e < group; ++e) {
            for (int l = 0; l < p; ++l) {
                const float v = e < n ? x[e * p + l] : 0.0f;
                const float qv = std::clamp(std::nearbyint(v * inverse[l]) + zero[l], 0.0f, levels);
                g[p == kMr ? l * group + e : (e / 4 * p + l) * 4 + e % 4] = static_cast<std::uint8_t>(qv);
            }
        }
        std::memcpy(g + q.quads * p * 4, scale, static_cast<std::size_t>(p) * sizeof(float));
        std::memcpy(g + q.quads * p * 4 + p * static_cast<std::int64_t>(sizeof(float)), offset,
                    static_cast<std::size_t>(p) * sizeof(float));
    }
}

void quantize_panel_weights(std::uint8_t * dst, const float * src, std::int64_t k, int p, const Quant & q) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    const auto max = static_cast<float>(q.weight_max);
    // The scale of each lane over the whole of k, unless each group has its own.
    float whole[kNr] = {};
    for (std::int64_t e = 0; !q.weight_groups && e < k; ++e) {
        for (int l = 0; l < p; ++l) {
            whole[l] = std::max(whole[l], std::fabs(src[e * p + l]));
        }
    }
    for (std::int64_t gi = 0; gi < groups; ++gi) {
        const std::int64_t n = std::min(group, k - gi * group);
        const float * w = src + gi * group * p;
        std::uint8_t * g = dst + gi * bytes;
        float m[kNr], scale[kNr], inverse[kNr];
        std::int32_t sum[kNr] = {};
        for (int l = 0; l < p; ++l) {
            m[l] = whole[l];
        }
        for (std::int64_t e = 0; q.weight_groups && e < n; ++e) {
            for (int l = 0; l < p; ++l) {
                m[l] = std::max(m[l], std::fabs(w[e * p + l]));
            }
        }
        for (int l = 0; l < p; ++l) {
            scale[l] = m[l] / max;
            inverse[l] = scale[l] > 0.0f ? 1.0f / scale[l] : 0.0f;
        }
        for (std::int64_t e = 0; e < group; ++e) {
            for (int l = 0; l < p; ++l) {
                const float v = e < n ? w[e * p + l] : 0.0f;
                const auto qv = static_cast<std::int32_t>(std::clamp(std::nearbyint(v * inverse[l]), -max, max));
                g[p == kMr ? l * group + e : (e / 4 * p + l) * 4 + e % 4] = static_cast<std::uint8_t>(static_cast<std::int8_t>(qv));
                sum[l] += qv;
            }
        }
        float offset[kNr];
        for (int l = 0; l < p; ++l) {
            offset[l] = scale[l] * static_cast<float>(sum[l]);
        }
        std::memcpy(g + q.quads * p * 4, scale, static_cast<std::size_t>(p) * sizeof(float));
        std::memcpy(g + q.quads * p * 4 + p * static_cast<std::int64_t>(sizeof(float)), offset,
                    static_cast<std::size_t>(p) * sizeof(float));
    }
}

Qkernel qkernel_a_unsigned() {
#if SDASM_X86_64
    if (avx512_kernels() && cpu().avx512_vnni) {
        return sdasm_qgemm_6x16_au_avx512;
    }
    if (avx2_kernels() && cpu().avx_vnni) {
        return sdasm_qgemm_6x16_au_avxvnni;
    }
#endif
    return qgemm_6x16_au_cxx;
}

Qkernel qkernel_b_unsigned() {
#if SDASM_X86_64
    if (avx512_kernels() && cpu().avx512_vnni) {
        return sdasm_qgemm_6x16_bu_avx512;
    }
    if (avx2_kernels() && cpu().avx_vnni) {
        return sdasm_qgemm_6x16_bu_avxvnni;
    }
#endif
    return qgemm_6x16_bu_cxx;
}

Qkernel_wide qkernel_wide_a_unsigned() {
#if SDASM_X86_64
    if (avx512_kernels() && cpu().avx512_vnni) {
        return sdasm_qgemm_6x32_au_avx512;
    }
#endif
    return nullptr;
}

Qkernel_wide qkernel_wide_b_unsigned() {
#if SDASM_X86_64
    if (avx512_kernels() && cpu().avx512_vnni) {
        return sdasm_qgemm_6x32_bu_avx512;
    }
#endif
    return nullptr;
}

}  // namespace sdasm
