// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

#include "store.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <print>
#include <string_view>
#include <mutex>
#include <string>
#include <tuple>

#include "kernels.h"
#include "sdasm.h"

#include "ggml.h"

namespace sdasm {

namespace {

constexpr int kFilling = 0;
constexpr int kReady = 1;

// A packed form's key: where, and of what, it was packed. The address is only
// compared, as std::less orders addresses, never read.
struct Key {
    const void * begin;
    std::string name;
    int type;
    std::array<std::int64_t, 4> ne;
    std::array<std::size_t, 4> nb;
    Layout layout;
    std::uint64_t variant;

    bool operator<(const Key & other) const {
        if (begin != other.begin) {
            return std::less<const void *>{}(begin, other.begin);
        }
        return std::tie(name, type, ne, nb, layout, variant) <
               std::tie(other.name, other.type, other.ne, other.nb, other.layout, other.variant);
    }
};

struct Entry {
    std::shared_ptr<Packed> packed;  // none when the budget had no room
    std::size_t size;                // the bytes it was packed from
};

// The budget of the quantized layouts when the environment gives none.
constexpr std::size_t kQuantizedBudget = std::size_t{8} << 30;

// SDASM_STORE_MB, mebibytes, when it is a number of them that a size holds; the
// default otherwise.
std::size_t budget_from_environment(bool quantized) {
    const std::size_t otherwise = quantized ? kQuantizedBudget : 0;
    const char * const set = std::getenv("SDASM_STORE_MB");
    if (set == nullptr) {
        return otherwise;
    }
    const std::string_view value = set;
    std::size_t mebibytes = 0;
    const auto [end, error] = std::from_chars(value.begin(), value.end(), mebibytes);
    if (error != std::errc{} || end != value.end() || mebibytes > std::numeric_limits<std::size_t>::max() >> 20) {
        std::print(stderr, "stable-diffusion.assembly: SDASM_STORE_MB={} is not a number of mebibytes; taking {}\n", value,
                   otherwise >> 20);
        return otherwise;
    }
    return mebibytes << 20;
}

bool quantized_layout(Layout layout) {
    return layout == Layout::quantized_linear || layout == Layout::quantized_winograd || layout == Layout::quantized_conv;
}

struct State {
    std::mutex mutex;
    std::map<Key, Entry> entries;
    std::atomic<std::size_t> count{0};
    std::size_t used = 0;
    std::size_t budget = budget_from_environment(false);
    // The quantized layouts', apart.
    std::size_t quantized_used = 0;
    std::size_t quantized_budget = budget_from_environment(true);
};

// The store is never destroyed: a thread may still be computing when the
// program ends, and what it holds is the program's memory, which goes with it.
State & state() {
    [[clang::no_destroy]] static State s;
    return s;
}

}  // namespace

Element element_of(Layout layout) {
    switch (layout) {
    case Layout::conv_rows:
    case Layout::winograd:
    case Layout::linear_columns:
        return Element::single;
    case Layout::amx_linear:
    case Layout::amx_winograd:
    case Layout::amx_conv:
    case Layout::amx_linear_fp16:
    case Layout::amx_conv_fp16:
        return Element::half;
    case Layout::quantized_linear:
    case Layout::quantized_winograd:
    case Layout::quantized_conv:
        return Element::byte;
    }
    return Element::byte;
}

namespace {

std::size_t element_bytes(Element e) {
    switch (e) {
    case Element::single:
        return sizeof(float);
    case Element::half:
        return sizeof(std::uint16_t);
    case Element::byte:
        return sizeof(std::uint8_t);
    }
    return 1;
}

}  // namespace

Packed::Packed(Element element, std::int64_t count)
    : element_(element), count_(count), bytes_(static_cast<std::size_t>(count) * element_bytes(element)) {
    SDASM_REQUIRE(count >= 0);
    memory_ = ::operator new(std::max<std::size_t>(bytes_, 1), std::align_val_t{kWorkAlign}, std::nothrow);
    if (memory_ == nullptr) {
        return;
    }
    // The array of the layout's elements begins its lifetime in the memory,
    // which it fills.
    const auto n = static_cast<std::size_t>(count);
    switch (element_) {
    case Element::single:
        array_ = std::start_lifetime_as_array<float>(memory_, n);
        break;
    case Element::half:
        array_ = std::start_lifetime_as_array<std::uint16_t>(memory_, n);
        break;
    case Element::byte:
        array_ = std::start_lifetime_as_array<std::uint8_t>(memory_, n);
        break;
    }
}

Packed::~Packed() {
    if (memory_ != nullptr) {
        ::operator delete(memory_, std::align_val_t{kWorkAlign});
    }
}

// The array began its lifetime as its elements' type, and its address was
// kept as void *: each is taken back as what it was.
Elements<float> Packed::singles() const {
    SDASM_REQUIRE(element_ == Element::single && memory_ != nullptr);
#pragma clang unsafe_buffer_usage begin
    return elements_at(static_cast<float *>(array_), count_);
#pragma clang unsafe_buffer_usage end
}

Elements<std::uint16_t> Packed::halves() const {
    SDASM_REQUIRE(element_ == Element::half && memory_ != nullptr);
#pragma clang unsafe_buffer_usage begin
    return elements_at(static_cast<std::uint16_t *>(array_), count_);
#pragma clang unsafe_buffer_usage end
}

Elements<std::uint8_t> Packed::octets() const {
    SDASM_REQUIRE(element_ == Element::byte && memory_ != nullptr);
#pragma clang unsafe_buffer_usage begin
    return elements_at(static_cast<std::uint8_t *>(array_), count_);
#pragma clang unsafe_buffer_usage end
}

bool Packed::ready() const {
    return state.load(std::memory_order_acquire) == kReady;
}

std::shared_ptr<Packed> packed(const ggml_tensor * const tensor, Layout layout, std::int64_t count,
                               std::uint64_t variant) {
    Key key{tensor->data, tensor->name, tensor->type, {}, {}, layout, variant};
    for (int k = 0; k < 4; ++k) {
        key.ne[static_cast<std::size_t>(k)] = extent(tensor, k);
        key.nb[static_cast<std::size_t>(k)] = static_cast<std::size_t>(stride_bytes(tensor, k));
    }
    State & s = state();
    std::lock_guard<std::mutex> const lock(s.mutex);
    const auto it = s.entries.find(key);
    if (it != s.entries.end()) {
        return it->second.packed;
    }
    std::shared_ptr<Packed> p;
    const Element element = element_of(layout);
    const std::size_t bytes = static_cast<std::size_t>(count) * element_bytes(element);
    std::size_t & used = quantized_layout(layout) ? s.quantized_used : s.used;
    const std::size_t budget = quantized_layout(layout) ? s.quantized_budget : s.budget;
    if (count >= 0 && used + bytes <= budget) {
        p = std::make_shared<Packed>(element, count);
        if (p->allocated()) {
            used += bytes;
            p->state.store(kFilling, std::memory_order_relaxed);
        }
        else {
            p = nullptr;
        }
    }
    s.entries.emplace(key, Entry{p, ggml_nbytes(tensor)});
    s.count.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void publish(const std::shared_ptr<Packed> & p) {
    if (p != nullptr) {
        p->state.store(kReady, std::memory_order_release);
    }
}

void forget(const void * const begin, std::size_t size) {
    State & s = state();
    if (s.count.load(std::memory_order_relaxed) == 0 || size == 0 || tuning().store_forget == 0) {
        return;
    }
    std::lock_guard<std::mutex> const lock(s.mutex);
    // Every entry whose memory shares a byte with [begin, begin + size).
    for (auto it = s.entries.begin(); it != s.entries.end();) {
        if (overlaps(it->first.begin, it->second.size, begin, size)) {
            if (it->second.packed != nullptr) {
                (quantized_layout(it->first.layout) ? s.quantized_used : s.used) -= it->second.packed->bytes();
            }
            it = s.entries.erase(it);
            s.count.fetch_sub(1, std::memory_order_relaxed);
        }
        else {
            ++it;
        }
    }
}

}  // namespace sdasm

extern "C" void sdasm_forget(const void * const begin, size_t size) {
    sdasm::forget(begin, size);
}
