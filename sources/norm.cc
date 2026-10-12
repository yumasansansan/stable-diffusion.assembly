// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// Group normalisation and layer normalisation (GGML_OP_GROUP_NORM,
// GGML_OP_NORM), with the chain after them (fusion.h): the scale and the shift
// of the normalisation, and the SiLU of a ResBlock. ggml computes each of these
// as an operation of its own, every one reading the tensor and writing it again;
// here the tensor is read three times, for its mean, for its variance and for
// the output, and written once, with the chain applied to each row as it is
// written. The sums are in double precision, as ggml's are, kept in kLanes
// partial sums that the compiler holds in vector registers and adds together
// at the end of a row.
//
// ggml shares a group normalisation out among threads by groups, one fixed
// share each; here a group of an image is an item that the threads take as each
// finishes the last.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

#include "fusion.h"
#include "gemm.h"
#include "memory.h"
#include "ops.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

namespace {

constexpr std::size_t kLanes = 8;

// The sum of a row in double precision.
double row_sum(Elements<const float> x) {
    const std::int64_t n = x.size();
    std::array<double, kLanes> part{};
    std::int64_t i = 0;
    for (; i + static_cast<std::int64_t>(kLanes) <= n; i += static_cast<std::int64_t>(kLanes)) {
        for (std::size_t l = 0; l < kLanes; ++l) {
            part[l] += static_cast<double>(x[i + static_cast<std::int64_t>(l)]);
        }
    }
    double s = 0.0;
    for (; i < n; ++i) {
        s += static_cast<double>(x[i]);
    }
    for (const double p : part) {
        s += p;
    }
    return s;
}

// The sum of the squares of (x - mean) of a row, each square in single
// precision and the sum in double.
double row_squares(Elements<const float> x, float mean) {
    const std::int64_t n = x.size();
    std::array<double, kLanes> part{};
    std::int64_t i = 0;
    for (; i + static_cast<std::int64_t>(kLanes) <= n; i += static_cast<std::int64_t>(kLanes)) {
        for (std::size_t l = 0; l < kLanes; ++l) {
            const float v = x[i + static_cast<std::int64_t>(l)] - mean;
            part[l] += static_cast<double>(v * v);
        }
    }
    double s = 0.0;
    for (; i < n; ++i) {
        const float v = x[i] - mean;
        s += static_cast<double>(v * v);
    }
    for (const double p : part) {
        s += p;
    }
    return s;
}

void normalize(Elements<float> y, Elements<const float> x, float mean, float scale) {
    for (std::int64_t i = 0; i < y.size(); ++i) {
        y[i] = (x[i] - mean) * scale;
    }
}

}  // namespace

bool group_norm_supported(const ggml_tensor * const node) {
    const ggml_tensor * const x = node->src[0];
    return node->op == GGML_OP_GROUP_NORM && holds<float>(x) && holds<float>(node) && ggml_is_contiguous(x) &&
           ggml_is_contiguous(node) && ggml_are_same_shape(x, node) && node->op_params[0] > 0;
}

void group_norm_compute(const ggml_compute_params * const params, const ggml_tensor * const node,
                        const Epilogue & chain) {
    const ggml_tensor * const src = node->src[0];
    const std::int64_t ne0 = src->ne[0];
    const std::int64_t ne1 = src->ne[1];
    const std::int64_t ne2 = src->ne[2];
    const std::int64_t ne3 = src->ne[3];
    const float eps = std::bit_cast<float>(node->op_params[1]);
    const int groups = node->op_params[0];
    const std::int64_t per_group = (ne2 + groups - 1) / groups;
    const Elements<const float> in = elements_of<const float>(src);
    const Elements<float> out = elements_of<float>(node);
    const auto at = [](const ggml_tensor * const t, std::int64_t c, std::int64_t r, std::int64_t i3) {
        return r * stride_of<float>(t, 1) + c * stride_of<float>(t, 2) + i3 * stride_of<float>(t, 3);
    };

    Work_queue queue(params);
    const std::int64_t items = static_cast<std::int64_t>(groups) * ne3;
    for (std::int64_t item = queue.next(); item < items; item = queue.next()) {
        const std::int64_t i3 = item / groups;
        const std::int64_t g = item % groups;
        const std::int64_t c0 = g * per_group;
        const std::int64_t c1 = std::min(ne2, c0 + per_group);
        if (c0 >= c1) {
            continue;
        }
        // The channels of a group are contiguous: its rows one after another.
        const std::int64_t count = ne0 * ne1 * (c1 - c0);
        const Elements<const float> group = in.part(at(src, c0, 0, i3), count);
        const float mean = static_cast<float>(row_sum(group) / static_cast<double>(count));
        const float variance = static_cast<float>(row_squares(group, mean) / static_cast<double>(count));
        const float scale = 1.0f / std::sqrt(variance + eps);
        for (std::int64_t c = c0; c < c1; ++c) {
            for (std::int64_t r = 0; r < ne1; ++r) {
                const Elements<float> y = out.part(at(node, c, r, i3), ne0);
                normalize(y, in.part(at(src, c, r, i3), ne0), mean, scale);
                chain.apply(y, 0, r, c, i3);
            }
        }
    }
}

bool norm_supported(const ggml_tensor * const node) {
    const ggml_tensor * const x = node->src[0];
    return node->op == GGML_OP_NORM && holds<float>(x) && holds<float>(node) && x->nb[0] == sizeof(float) &&
           node->nb[0] == sizeof(float) && ggml_are_same_shape(x, node);
}

void norm_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain) {
    const ggml_tensor * const src = node->src[0];
    const std::int64_t ne0 = src->ne[0];
    const std::int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
    const float eps = std::bit_cast<float>(node->op_params[0]);
    const Elements<const float> in = elements_of<const float>(src);
    const Elements<float> out = elements_of<float>(node);
    const auto at = [](const ggml_tensor * const t, std::int64_t i1, std::int64_t i2, std::int64_t i3) {
        return i1 * stride_of<float>(t, 1) + i2 * stride_of<float>(t, 2) + i3 * stride_of<float>(t, 3);
    };

    Work_queue queue(params);
    constexpr std::int64_t kRows = 16;
    const std::int64_t items = (rows + kRows - 1) / kRows;
    for (std::int64_t item = queue.next(); item < items; item = queue.next()) {
        for (std::int64_t r = item * kRows; r < std::min(rows, (item + 1) * kRows); ++r) {
            const std::int64_t i1 = r % src->ne[1];
            const std::int64_t i2 = (r / src->ne[1]) % src->ne[2];
            const std::int64_t i3 = r / (src->ne[1] * src->ne[2]);
            const Elements<const float> x = in.part(at(src, i1, i2, i3), ne0);
            const Elements<float> y = out.part(at(node, i1, i2, i3), ne0);
            const float mean = static_cast<float>(row_sum(x) / static_cast<double>(ne0));
            const float variance = static_cast<float>(row_squares(x, mean) / static_cast<double>(ne0));
            normalize(y, x, mean, 1.0f / std::sqrt(variance + eps));
            chain.apply(y, 0, i1, i2, i3);
        }
    }
}

}  // namespace sdasm
