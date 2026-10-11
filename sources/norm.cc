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
#include <cmath>
#include <cstdint>
#include <cstring>

#include "fusion.h"
#include "gemm.h"
#include "ops.h"

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

namespace {

std::int64_t nb(const ggml_tensor * t, int k) {
    return static_cast<std::int64_t>(t->nb[k]);
}

constexpr int kLanes = 8;

// The sum of a row in double precision.
double row_sum(const float * x, std::int64_t n) {
    double part[kLanes] = {};
    std::int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (int l = 0; l < kLanes; ++l) {
            part[l] += static_cast<double>(x[i + l]);
        }
    }
    double s = 0.0;
    for (; i < n; ++i) {
        s += static_cast<double>(x[i]);
    }
    for (double p : part) {
        s += p;
    }
    return s;
}

// The sum of the squares of (x - mean) of a row, each square in single
// precision and the sum in double.
double row_squares(const float * x, std::int64_t n, float mean) {
    double part[kLanes] = {};
    std::int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (int l = 0; l < kLanes; ++l) {
            const float v = x[i + l] - mean;
            part[l] += static_cast<double>(v * v);
        }
    }
    double s = 0.0;
    for (; i < n; ++i) {
        const float v = x[i] - mean;
        s += static_cast<double>(v * v);
    }
    for (double p : part) {
        s += p;
    }
    return s;
}

void normalize(float * y, const float * x, std::int64_t n, float mean, float scale) {
    for (std::int64_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * scale;
    }
}

}  // namespace

bool group_norm_supported(const ggml_tensor * node) {
    const ggml_tensor * x = node->src[0];
    return node->op == GGML_OP_GROUP_NORM && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(x) && ggml_is_contiguous(node) && ggml_are_same_shape(x, node);
}

void group_norm_compute(const ggml_compute_params * params, ggml_tensor * node, const Epilogue & chain) {
    const ggml_tensor * src = node->src[0];
    const std::int64_t ne0 = src->ne[0];
    const std::int64_t ne1 = src->ne[1];
    const std::int64_t ne2 = src->ne[2];
    const std::int64_t ne3 = src->ne[3];
    float eps;
    std::memcpy(&eps, node->op_params + 1, sizeof eps);
    const int groups = node->op_params[0];
    const std::int64_t per_group = (ne2 + groups - 1) / groups;

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
        const auto row_in = [&](std::int64_t c, std::int64_t r) {
            return reinterpret_cast<const float *>(static_cast<const char *>(src->data) + r * nb(src, 1) + c * nb(src, 2) +
                                                   i3 * nb(src, 3));
        };
        const auto row_out = [&](std::int64_t c, std::int64_t r) {
            return reinterpret_cast<float *>(static_cast<char *>(node->data) + r * nb(node, 1) + c * nb(node, 2) +
                                             i3 * nb(node, 3));
        };
        // The channels of a group are contiguous: its rows one after another.
        const std::int64_t count = ne0 * ne1 * (c1 - c0);
        const float * first = row_in(c0, 0);
        const float mean = static_cast<float>(row_sum(first, count) / static_cast<double>(count));
        const float variance = static_cast<float>(row_squares(first, count, mean) / static_cast<double>(count));
        const float scale = 1.0f / std::sqrt(variance + eps);
        for (std::int64_t c = c0; c < c1; ++c) {
            for (std::int64_t r = 0; r < ne1; ++r) {
                float * y = row_out(c, r);
                normalize(y, row_in(c, r), ne0, mean, scale);
                chain.apply(y, ne0, 0, r, c, i3);
            }
        }
    }
}

bool norm_supported(const ggml_tensor * node) {
    const ggml_tensor * x = node->src[0];
    return node->op == GGML_OP_NORM && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
           x->nb[0] == sizeof(float) && node->nb[0] == sizeof(float) && ggml_are_same_shape(x, node);
}

void norm_compute(const ggml_compute_params * params, ggml_tensor * node, const Epilogue & chain) {
    const ggml_tensor * src = node->src[0];
    const std::int64_t ne0 = src->ne[0];
    const std::int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
    float eps;
    std::memcpy(&eps, node->op_params, sizeof eps);

    Work_queue queue(params);
    constexpr std::int64_t kRows = 16;
    const std::int64_t items = (rows + kRows - 1) / kRows;
    for (std::int64_t item = queue.next(); item < items; item = queue.next()) {
        for (std::int64_t r = item * kRows; r < std::min(rows, (item + 1) * kRows); ++r) {
            const std::int64_t i1 = r % src->ne[1];
            const std::int64_t i2 = (r / src->ne[1]) % src->ne[2];
            const std::int64_t i3 = r / (src->ne[1] * src->ne[2]);
            const auto * x = reinterpret_cast<const float *>(static_cast<const char *>(src->data) + i1 * nb(src, 1) +
                                                             i2 * nb(src, 2) + i3 * nb(src, 3));
            auto * y = reinterpret_cast<float *>(static_cast<char *>(node->data) + i1 * nb(node, 1) + i2 * nb(node, 2) +
                                                 i3 * nb(node, 3));
            const float mean = static_cast<float>(row_sum(x, ne0) / static_cast<double>(ne0));
            const float variance = static_cast<float>(row_squares(x, ne0, mean) / static_cast<double>(ne0));
            normalize(y, x, ne0, mean, 1.0f / std::sqrt(variance + eps));
            chain.apply(y, ne0, 0, i1, i2, i3);
        }
    }
}

}  // namespace sdasm
