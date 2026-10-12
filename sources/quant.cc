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
#include <array>
#include <bit>
#include <cmath>

#include "kernels.h"

#include "ggml.h"

namespace sdasm {

namespace {

// The most rows of a panel: a tile of AMX's.
constexpr int kAmxRows = 16;

#if SDASM_X86_64
// Whether two products of an unsigned byte of the inputs' levels by a signed
// one of the weights' greatest magnitude sum to no more than a word holds:
// what vpmaddubsw's sums take without saturating (x86_64/quant_maddubs.S).
bool narrow(const Quant & q) {
    return 2 * q.input_levels * q.weight_max <= 32767;
}
#endif

}  // namespace

bool quantized_kernels() {
    if (kernels() != Kernels::assembly) {
        return kernels() == Kernels::cxx;
    }
#if SDASM_X86_64
    return (avx512_kernels() && cpu().avx512_vnni) || (avx2_kernels() && (cpu().avx_vnni || narrow(quant())));
#else
    return true;
#endif
}

bool quantized(Quantized_op op) {
    const Tuning & t = tuning();
    return t.bits >= 2 && t.bits <= 8 && (t.quantize & static_cast<int>(op)) != 0 && quantized_kernels();
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

void read_floats(Elements<const std::uint8_t> bytes, Elements<float> values) {
    for (std::int64_t i = 0; i < values.size(); ++i) {
        std::array<std::uint8_t, sizeof(float)> b{};
        for (std::size_t u = 0; u < b.size(); ++u) {
            b[u] = bytes[i * 4 + static_cast<std::int64_t>(u)];
        }
        values[i] = std::bit_cast<float>(b);
    }
}

void write_floats(Elements<std::uint8_t> bytes, Elements<const float> values) {
    for (std::int64_t i = 0; i < values.size(); ++i) {
        const auto b = std::bit_cast<std::array<std::uint8_t, sizeof(float)>>(values[i]);
        for (std::size_t u = 0; u < b.size(); ++u) {
            bytes[i * 4 + static_cast<std::int64_t>(u)] = b[u];
        }
    }
}

namespace {

// The operands of a quantized kernel, as the kernels written by hand take
// them: panels of `groups` groups of kMr rows and kNr columns, and C, kMr rows
// ldc bytes apart.
struct Qoperands {
    Elements<const std::uint8_t> a, b;
    Elements<float> c;
    std::int64_t ldc;  // elements
};

Qoperands qoperands(std::int64_t groups, std::int64_t quads, const std::uint8_t * const a, const std::uint8_t * const b,
                    float * const c, std::int64_t ldc) {
    SDASM_REQUIRE(groups >= 0 && quads >= 1 && ldc % bytes_of<float>(1) == 0 && ldc >= bytes_of<float>(kNr));
    const std::int64_t row = ldc / bytes_of<float>(1);
    // The kernels' operands, handed over as pointers.
#pragma clang unsafe_buffer_usage begin
    return {elements_at(a, groups * qgroup_bytes(quads, kMr)), elements_at(b, groups * qgroup_bytes(quads, kNr)),
            elements_at(c, (kMr - 1) * row + kNr), row};
#pragma clang unsafe_buffer_usage end
}

// The kernel's tile in an array of its own, the group's sums in another; the
// bytes of A read as unsigned when AUnsigned, those of B as signed, and the
// other way about otherwise.
template <bool AUnsigned>
void qgemm(std::int64_t groups, std::int64_t quads, const std::uint8_t * const a_first,
           const std::uint8_t * const b_first, float * const c_first, std::int64_t ldc_bytes, std::int64_t accumulate) {
    const Qoperands o = qoperands(groups, quads, a_first, b_first, c_first, ldc_bytes);
    std::array<std::array<float, kNr>, kMr> t{};
    for (std::size_t i = 0; i < kMr; ++i) {
        const Elements<const float> r = o.c.part(static_cast<std::int64_t>(i) * o.ldc, kNr);
        for (std::size_t j = 0; j < kNr; ++j) {
            t[i][j] = accumulate != 0 ? r[static_cast<std::int64_t>(j)] : 0.0f;
        }
    }
    const std::int64_t a_bytes = qgroup_bytes(quads, kMr);
    const std::int64_t b_bytes = qgroup_bytes(quads, kNr);
    for (std::int64_t group = 0; group < groups; ++group) {
        const Elements<const std::uint8_t> a = o.a.part(group * a_bytes, a_bytes);
        const Elements<const std::uint8_t> b = o.b.part(group * b_bytes, b_bytes);
        std::array<std::array<std::int32_t, kNr>, kMr> sum{};
        for (std::int64_t q = 0; q < quads; ++q) {
            for (std::size_t i = 0; i < kMr; ++i) {
                for (std::size_t j = 0; j < kNr; ++j) {
                    for (std::int64_t u = 0; u < 4; ++u) {
                        const std::uint8_t ai = a[static_cast<std::int64_t>(i) * quads * 4 + q * 4 + u];
                        const std::uint8_t bj = b[q * kNr * 4 + static_cast<std::int64_t>(j) * 4 + u];
                        const std::int32_t x = AUnsigned ? static_cast<std::int32_t>(ai)
                                                         : static_cast<std::int32_t>(static_cast<std::int8_t>(ai));
                        const std::int32_t y = AUnsigned ? static_cast<std::int32_t>(static_cast<std::int8_t>(bj))
                                                         : static_cast<std::int32_t>(bj);
                        sum[i][j] += x * y;
                    }
                }
            }
        }
        std::array<float, kMr> sa{}, oa{};
        std::array<float, kNr> sb{}, ob{};
        read_floats(a.part(quads * kMr * 4, kMr * 4), sa);
        read_floats(a.part(quads * kMr * 4 + kMr * 4, kMr * 4), oa);
        read_floats(b.part(quads * kNr * 4, kNr * 4), sb);
        read_floats(b.part(quads * kNr * 4 + kNr * 4, kNr * 4), ob);
        for (std::size_t i = 0; i < kMr; ++i) {
            for (std::size_t j = 0; j < kNr; ++j) {
                t[i][j] = std::fma(sa[i] * sb[j], static_cast<float>(sum[i][j]), t[i][j]);
                t[i][j] = std::fma(-oa[i], ob[j], t[i][j]);
            }
        }
    }
    for (std::size_t i = 0; i < kMr; ++i) {
        const Elements<float> r = o.c.part(static_cast<std::int64_t>(i) * o.ldc, kNr);
        for (std::size_t j = 0; j < kNr; ++j) {
            r[static_cast<std::int64_t>(j)] = t[i][j];
        }
    }
}

// The scale and the offset of a group of inputs and their bytes: those of
// the group's least and greatest numbers, 0 between them, levels steps apart;
// number e at dst[e / 4 * step + e % 4].
void quantize_input_group(Elements<std::uint8_t> dst, std::int64_t step, Elements<const float> x, int levels, float & scale,
                          float & offset) {
    float lo = 0.0f, hi = 0.0f;
    for (const float v : x) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float s = (hi - lo) / static_cast<float>(levels);
    const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
    const float zero = std::nearbyint(-lo * inverse);
    for (std::int64_t e = 0; e < x.size(); ++e) {
        const float q = std::clamp(std::nearbyint(x[e] * inverse) + zero, 0.0f, static_cast<float>(levels));
        dst[e / 4 * step + e % 4] = static_cast<std::uint8_t>(q);
    }
    scale = s;
    offset = s * zero;
}

// The weights of a group, at the scale given, and their sum; weight e at
// dst[e / 4 * step + e % 4].
std::int32_t quantize_weight_group(Elements<std::uint8_t> dst, std::int64_t step, Elements<const float> w, float inverse,
                                   int max) {
    std::int32_t sum = 0;
    for (std::int64_t e = 0; e < w.size(); ++e) {
        const auto q = static_cast<std::int32_t>(std::clamp(std::nearbyint(w[e] * inverse), static_cast<float>(-max),
                                                            static_cast<float>(max)));
        dst[e / 4 * step + e % 4] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        sum += q;
    }
    return sum;
}

// The most numbers of a group: 4 quads.clamp's greatest (quant()).
constexpr std::int64_t kMaxGroup = 1024;

template <class Load>
void quantize_weights(Elements<std::uint8_t> dst, std::int64_t rows, std::int64_t k, int p, Qlayout layout, const Quant & q,
                      Load load) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    SDASM_REQUIRE(group <= kMaxGroup);
    fill_with<std::uint8_t>(dst.first(groups * bytes), 0);
    std::array<float, kMaxGroup> wbuf{};
    const Elements<float> w = Elements<float>(wbuf).first(group);
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
            const Elements<std::uint8_t> g = dst.part(gi * bytes, bytes);
            const std::int32_t sum = layout == Qlayout::rows
                                         ? quantize_weight_group(g.from(r * group), 4, w, inverse, q.weight_max)
                                         : quantize_weight_group(g.from(r * 4), p * 4, w, inverse, q.weight_max);
            const std::array<float, 1> scale{s};
            const std::array<float, 1> offset{s * static_cast<float>(sum)};
            write_floats(g.part(q.quads * p * 4 + r * 4, 4), scale);
            write_floats(g.part(q.quads * p * 4 + (p + r) * 4, 4), offset);
        }
    }
}

}  // namespace

void qgemm_6x16_au_cxx(std::int64_t groups, std::int64_t quads, const std::uint8_t * const a,
                       const std::uint8_t * const b, float * const c, std::int64_t ldc, std::int64_t accumulate) {
    qgemm<true>(groups, quads, a, b, c, ldc, accumulate);
}

void qgemm_6x16_bu_cxx(std::int64_t groups, std::int64_t quads, const std::uint8_t * const a,
                       const std::uint8_t * const b, float * const c, std::int64_t ldc, std::int64_t accumulate) {
    qgemm<false>(groups, quads, a, b, c, ldc, accumulate);
}

void qtile_part(Qkernel kernel, std::int64_t groups, std::int64_t quads, Elements<const std::uint8_t> a,
                Elements<const std::uint8_t> b, Elements<float> c, std::int64_t ldc, bool accumulate, std::int64_t rows,
                std::int64_t cols) {
    alignas(64) std::array<float, kMr * kNr> t{};
    const Elements<float> tile = t;
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            tile[i * kNr + j] = accumulate ? c[i * ldc + j] : 0.0f;
        }
    }
    SDASM_CHECK(a.size() >= groups * qgroup_bytes(quads, kMr) && b.size() >= groups * qgroup_bytes(quads, kNr));
    kernel(groups, quads, a.data(), b.data(), tile.data(), bytes_of<float>(kNr), accumulate ? 1 : 0);
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            c[i * ldc + j] = tile[i * kNr + j];
        }
    }
}

namespace {

// A row's group of inputs, x, of at most `group`, into its bytes, contiguous,
// dst; the group's scale and offset returned: the least and greatest a lane of
// sixteen at a time, so that the compiler makes vector minima and maxima of
// them with no reduction to order, then those of the lanes.
void quantize_input_row(Elements<std::uint8_t> dst, Elements<const float> x, std::int64_t group, int levels, float & scale,
                        float & offset) {
    const std::int64_t n = x.size();
    std::array<float, kNr> lo{}, hi{};
    std::int64_t e = 0;
    for (; e + kNr <= n; e += kNr) {
        for (std::size_t l = 0; l < kNr; ++l) {
            lo[l] = std::min(lo[l], x[e + static_cast<std::int64_t>(l)]);
            hi[l] = std::max(hi[l], x[e + static_cast<std::int64_t>(l)]);
        }
    }
    for (; e < n; ++e) {
        lo[0] = std::min(lo[0], x[e]);
        hi[0] = std::max(hi[0], x[e]);
    }
    float least = 0.0f, greatest = 0.0f;
    for (std::size_t l = 0; l < kNr; ++l) {
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

void quantize_inputs(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t stride, std::int64_t rows,
                     std::int64_t k, int p, Qlayout layout, const Quant & q) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    SDASM_REQUIRE(p <= kAmxRows && group <= kMaxGroup);
    if (layout == Qlayout::rows) {
        // A's rows, each group's bytes in one piece; the rows past `rows`,
        // zeros of no scale.
        for (std::int64_t gi = 0; gi < groups; ++gi) {
            const Elements<std::uint8_t> g = dst.part(gi * bytes, bytes);
            std::array<float, kAmxRows> scale{}, offset{};
            for (std::int64_t r = 0; r < p; ++r) {
                const auto ri = static_cast<std::size_t>(r);
                if (r >= rows) {
                    fill_with<std::uint8_t>(g.part(r * group, group), 0);
                    continue;
                }
                const std::int64_t n = std::min(group, k - gi * group);
                quantize_input_row(g.part(r * group, group), src.part(r * stride + gi * group, n), group, q.input_levels,
                                   scale[ri], offset[ri]);
            }
            write_floats(g.part(q.quads * p * 4, p * 4), Elements<const float>(scale).first(p));
            write_floats(g.part(q.quads * p * 4 + p * 4, p * 4), Elements<const float>(offset).first(p));
        }
        return;
    }
    fill_with<std::uint8_t>(dst.first(groups * bytes), 0);
    std::array<float, kMaxGroup> xbuf{};
    const Elements<float> x = Elements<float>(xbuf).first(group);
    for (std::int64_t r = 0; r < rows; ++r) {
        const Elements<const float> line = src.part(r * stride, k);
        for (std::int64_t gi = 0; gi < groups; ++gi) {
            const std::int64_t n = std::min(group, k - gi * group);
            for (std::int64_t e = 0; e < group; ++e) {
                x[e] = e < n ? line[gi * group + e] : 0.0f;
            }
            const Elements<std::uint8_t> g = dst.part(gi * bytes, bytes);
            std::array<float, 1> scale{}, offset{};
            // B's columns quad by quad.
            quantize_input_group(g.from(r * 4), p * 4, x, q.input_levels, scale[0], offset[0]);
            write_floats(g.part(q.quads * p * 4 + r * 4, 4), scale);
            write_floats(g.part(q.quads * p * 4 + (p + r) * 4, 4), offset);
        }
    }
}

void quantize_weights_f16(Elements<std::uint8_t> dst, Elements<const std::uint16_t> src, std::int64_t stride,
                          std::int64_t rows, std::int64_t k, int p, Qlayout layout, const Quant & q) {
    quantize_weights(dst, rows, k, p, layout, q, [&](std::int64_t r, std::int64_t e) {
        return static_cast<float>(std::bit_cast<_Float16>(src[r * stride + e]));
    });
}

void quantize_weights_f32(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t stride, std::int64_t rows,
                          std::int64_t k, int p, Qlayout layout, const Quant & q) {
    quantize_weights(dst, rows, k, p, layout, q, [&](std::int64_t r, std::int64_t e) { return src[r * stride + e]; });
}

// The kernels written by hand where the processor has VNNI at the level in
// effect: AVX512-VNNI's at avx512 and above, AVX-VNNI's at avx2.
void quantize_panel_inputs(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t k, int p, Qlayout layout,
                           const Quant & q) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    const auto levels = static_cast<float>(q.input_levels);
    SDASM_REQUIRE(p <= kNr);
    std::int64_t gi = 0;
#if SDASM_X86_64
    // The whole groups of a panel of B by the kernel written by hand.
    if (layout == Qlayout::columns && p == kNr && avx2_kernels()) {
        gi = k / group;
        SDASM_CHECK(dst.size() >= gi * bytes && src.size() >= gi * group * p);
        sdasm_quantize16_avx2(dst.data(), src.data(), gi, q.quads, levels);
    }
#endif
    for (; gi < groups; ++gi) {
        const std::int64_t n = std::min(group, k - gi * group);
        const Elements<const float> x = src.part(gi * group * p, n * p);
        const Elements<std::uint8_t> g = dst.part(gi * bytes, bytes);
        // Each lane's least and greatest, 0 between them: the zero point.
        std::array<float, kNr> lo{}, hi{}, inverse{}, zero{}, scale{}, offset{};
        for (std::int64_t e = 0; e < n; ++e) {
            for (std::int64_t l = 0; l < p; ++l) {
                const auto li = static_cast<std::size_t>(l);
                lo[li] = std::min(lo[li], x[e * p + l]);
                hi[li] = std::max(hi[li], x[e * p + l]);
            }
        }
        for (std::size_t l = 0; l < static_cast<std::size_t>(p); ++l) {
            scale[l] = (hi[l] - lo[l]) / levels;
            inverse[l] = scale[l] > 0.0f ? 1.0f / scale[l] : 0.0f;
            zero[l] = std::nearbyint(-lo[l] * inverse[l]);
            offset[l] = scale[l] * zero[l];
        }
        for (std::int64_t e = 0; e < group; ++e) {
            for (std::int64_t l = 0; l < p; ++l) {
                const auto li = static_cast<std::size_t>(l);
                const float v = e < n ? x[e * p + l] : 0.0f;
                const float qv = std::clamp(std::nearbyint(v * inverse[li]) + zero[li], 0.0f, levels);
                g[layout == Qlayout::rows ? l * group + e : (e / 4 * p + l) * 4 + e % 4] = static_cast<std::uint8_t>(qv);
            }
        }
        write_floats(g.part(q.quads * p * 4, p * 4), Elements<const float>(scale).first(p));
        write_floats(g.part(q.quads * p * 4 + p * 4, p * 4), Elements<const float>(offset).first(p));
    }
}

void quantize_panel_weights(Elements<std::uint8_t> dst, Elements<const float> src, std::int64_t k, int p, Qlayout layout,
                            const Quant & q) {
    const std::int64_t group = 4 * q.quads;
    const std::int64_t groups = qgroups(k, q);
    const std::int64_t bytes = qgroup_bytes(q.quads, p);
    const auto max = static_cast<float>(q.weight_max);
    SDASM_REQUIRE(p <= kNr);
    // The scale of each lane over the whole of k, unless each group has its own.
    std::array<float, kNr> whole{};
    for (std::int64_t e = 0; !q.weight_groups && e < k; ++e) {
        for (std::int64_t l = 0; l < p; ++l) {
            const auto li = static_cast<std::size_t>(l);
            whole[li] = std::max(whole[li], std::fabs(src[e * p + l]));
        }
    }
    for (std::int64_t gi = 0; gi < groups; ++gi) {
        const std::int64_t n = std::min(group, k - gi * group);
        const Elements<const float> w = src.part(gi * group * p, n * p);
        const Elements<std::uint8_t> g = dst.part(gi * bytes, bytes);
        std::array<float, kNr> m = whole, scale{}, inverse{}, offset{};
        std::array<std::int32_t, kNr> sum{};
        for (std::int64_t e = 0; q.weight_groups && e < n; ++e) {
            for (std::int64_t l = 0; l < p; ++l) {
                const auto li = static_cast<std::size_t>(l);
                m[li] = std::max(m[li], std::fabs(w[e * p + l]));
            }
        }
        for (std::size_t l = 0; l < static_cast<std::size_t>(p); ++l) {
            scale[l] = m[l] / max;
            inverse[l] = scale[l] > 0.0f ? 1.0f / scale[l] : 0.0f;
        }
        for (std::int64_t e = 0; e < group; ++e) {
            for (std::int64_t l = 0; l < p; ++l) {
                const auto li = static_cast<std::size_t>(l);
                const float v = e < n ? w[e * p + l] : 0.0f;
                const auto qv = static_cast<std::int32_t>(std::clamp(std::nearbyint(v * inverse[li]), -max, max));
                g[layout == Qlayout::rows ? l * group + e : (e / 4 * p + l) * 4 + e % 4] =
                    static_cast<std::uint8_t>(static_cast<std::int8_t>(qv));
                sum[li] += qv;
            }
        }
        for (std::size_t l = 0; l < static_cast<std::size_t>(p); ++l) {
            offset[l] = scale[l] * static_cast<float>(sum[l]);
        }
        write_floats(g.part(q.quads * p * 4, p * 4), Elements<const float>(scale).first(p));
        write_floats(g.part(q.quads * p * 4 + p * 4, p * 4), Elements<const float>(offset).first(p));
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
    if (avx2_kernels() && narrow(quant())) {
        return sdasm_qgemm_6x16_au_avx2;
    }
#endif
    return qgemm_6x16_au_cxx;
}

Qkernel_amx qkernel_amx_a_unsigned(const Quant & q) {
#if SDASM_X86_64
    if (amx_kernels() && cpu().amx_int8 && q.quads % 16 == 0) {
        return sdasm_amx_qgemm_32x32_au;
    }
#else
    (void) q;
#endif
    return nullptr;
}

void qamx_begin() {
#if SDASM_X86_64
    sdasm_amx_configure();
#endif
}

void qamx_end() {
#if SDASM_X86_64
    sdasm_amx_release();
#endif
}

void qblock_amx(Qkernel_amx kernel, std::int64_t groups, std::int64_t quads, Elements<const std::uint8_t> a,
                std::int64_t a_next, Elements<const std::uint8_t> b, std::int64_t b_next, Elements<float> c,
                std::int64_t ldc, bool accumulate, std::int64_t rows, std::int64_t cols, Elements<float> spare) {
    constexpr std::int64_t kBlock = 32;
    SDASM_CHECK(a.size() >= a_next + groups * qgroup_bytes(quads, 16) &&
                b.size() >= b_next + groups * qgroup_bytes(quads, kNr));
    if (rows == kBlock && cols == kBlock) {
        SDASM_CHECK(c.size() >= (kBlock - 1) * ldc + kBlock);
        kernel(groups, quads, a.data(), a_next, b.data(), b_next, c.data(), bytes_of<float>(ldc), accumulate ? 1 : 0);
        return;
    }
    SDASM_CHECK(spare.size() >= kBlock * kBlock && c.size() >= (rows - 1) * ldc + cols);
    if (accumulate) {
        for (std::int64_t i = 0; i < kBlock; ++i) {
            for (std::int64_t j = 0; j < kBlock; ++j) {
                spare[i * kBlock + j] = i < rows && j < cols ? c[i * ldc + j] : 0.0f;
            }
        }
    }
    kernel(groups, quads, a.data(), a_next, b.data(), b_next, spare.data(), bytes_of<float>(kBlock), accumulate ? 1 : 0);
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            c[i * ldc + j] = spare[i * kBlock + j];
        }
    }
}

Qkernel qkernel_b_unsigned() {
#if SDASM_X86_64
    if (avx512_kernels() && cpu().avx512_vnni) {
        return sdasm_qgemm_6x16_bu_avx512;
    }
    if (avx2_kernels() && cpu().avx_vnni) {
        return sdasm_qgemm_6x16_bu_avxvnni;
    }
    if (avx2_kernels() && narrow(quant())) {
        return sdasm_qgemm_6x16_bu_avx2;
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
