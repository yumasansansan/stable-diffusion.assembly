// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

#include "memory.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <print>

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace sdasm {

void fail(const char * const what, const char * const file, int line) {
    std::print(stderr, "stable-diffusion.assembly: {}:{}: {} does not hold\n", file, line, what);
    std::abort();
}

std::vector<std::string_view> program_arguments(int argc, char ** const argv) {
    std::vector<std::string_view> arguments;
    if (argc <= 0 || argv == nullptr) {
        return arguments;
    }
    // argv, argc C strings, as the C library hands them to main().
#pragma clang unsafe_buffer_usage begin
    const std::span<char * const> all(argv, static_cast<std::size_t>(argc));
#pragma clang unsafe_buffer_usage end
    for (const char * const argument : all) {
        arguments.emplace_back(argument != nullptr ? argument : "");
    }
    return arguments;
}

template <>
bool type_is<float>(const ggml_tensor * const t) {
    return t->type == GGML_TYPE_F32;
}

template <>
bool type_is<std::uint16_t>(const ggml_tensor * const t) {
    return t->type == GGML_TYPE_F16;
}

bool aligned(void * const p, std::size_t alignment) {
    // std::align moves p to the first address of the alignment at or after it,
    // which is p itself when p is aligned.
    void * q = p;
    std::size_t space = alignment;
    return std::align(alignment, 1, q, space) == p;
}

bool overlaps(const void * const a, std::size_t a_size, const void * const b, std::size_t b_size) {
    if (a == nullptr || b == nullptr || a_size == 0 || b_size == 0) {
        return false;
    }
    // The bytes of each, as ggml hands them over: an address and a size.
#pragma clang unsafe_buffer_usage begin
    const std::span<const std::byte> x(static_cast<const std::byte *>(a), a_size);
    const std::span<const std::byte> y(static_cast<const std::byte *>(b), b_size);
#pragma clang unsafe_buffer_usage end
    const std::less<const std::byte *> less;
    return less(x.data(), std::to_address(y.end())) && less(y.data(), std::to_address(x.end()));
}

std::size_t data_bytes(const ggml_tensor * const t) {
    return ggml_nbytes(t);
}

std::int64_t stride_bytes(const ggml_tensor * const t, int dimension) {
    const std::span<const std::size_t, 4> nb(t->nb);
    return static_cast<std::int64_t>(nb[static_cast<std::size_t>(dimension)]);
}

std::int64_t extent(const ggml_tensor * const t, int dimension) {
    const std::span<const std::int64_t, 4> ne(t->ne);
    return ne[static_cast<std::size_t>(dimension)];
}

void * data_address(const ggml_tensor * const t) {
    return t->data;
}

namespace {

std::size_t align_bytes(std::size_t n) {
    return (n + kWorkAlign - 1) / kWorkAlign * kWorkAlign;
}

}  // namespace

int Work_layout::add(bool shared, std::int64_t count, std::size_t size, Create create) {
    SDASM_REQUIRE(count_ < kMaxArrays && count >= 0 &&
                  static_cast<std::size_t>(count) <= std::numeric_limits<std::size_t>::max() / 2 / size);
    const std::size_t bytes = static_cast<std::size_t>(count) * size;
    std::size_t & total = shared ? shared_bytes_ : thread_bytes_;
    arrays_[static_cast<std::size_t>(count_)] = Array{shared, total, static_cast<std::size_t>(count), bytes, create};
    total += align_bytes(bytes);
    return count_++;
}

std::size_t Work_layout::bytes(int n_threads) const {
    if (count_ == 0) {
        return 0;
    }
    return shared_bytes_ + thread_bytes_ * static_cast<std::size_t>(n_threads) + kWorkAlign;
}

Work_buffer::Work_buffer(const ggml_compute_params * const params, const Work_layout & layout) : count_(layout.count_) {
    if (count_ == 0) {
        return;
    }
    const std::size_t thread_bytes = layout.thread_bytes_;
    const std::size_t need = layout.shared_bytes_ + thread_bytes * static_cast<std::size_t>(params->nth);
    if (need == 0) {
        // Arrays of no elements, which need no memory.
        for (int i = 0; i < count_; ++i) {
            creates_[static_cast<std::size_t>(i)] = layout.arrays_[static_cast<std::size_t>(i)].create;
        }
        return;
    }
    // The first aligned byte of the buffer, with the bytes of every array
    // after it: what work_size() asked for, at most kWorkAlign bytes in.
    void * first = params->wdata;
    std::size_t space = params->wsize;
    SDASM_REQUIRE(first != nullptr && std::align(kWorkAlign, need, first, space) != nullptr);
    // The buffer's bytes, ggml's array of them.
#pragma clang unsafe_buffer_usage begin
    const Elements<std::byte> bytes = elements_at(static_cast<std::byte *>(first), static_cast<std::int64_t>(need));
#pragma clang unsafe_buffer_usage end
    const std::size_t mine = layout.shared_bytes_ + thread_bytes * static_cast<std::size_t>(params->ith);
    for (int i = 0; i < count_; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const Work_layout::Array & a = layout.arrays_[k];
        const std::size_t offset = a.shared ? a.offset : mine + a.offset;
        const Elements<std::byte> where = bytes.part(static_cast<std::int64_t>(offset), static_cast<std::int64_t>(a.bytes));
        arrays_[k] = a.create(where.data(), a.count);
        creates_[k] = a.create;
    }
}

}  // namespace sdasm
