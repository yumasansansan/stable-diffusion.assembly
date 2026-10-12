// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).

#include "store.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

struct Key {
    const char * begin;
    std::string name;
    int type;
    std::int64_t ne[4];
    std::size_t nb[4];
    Layout layout;
    std::uint64_t variant;

    bool operator<(const Key & other) const {
        return std::tie(begin, name, type, ne[0], ne[1], ne[2], ne[3], nb[0], nb[1], nb[2], nb[3], layout, variant) <
               std::tie(other.begin, other.name, other.type, other.ne[0], other.ne[1], other.ne[2], other.ne[3], other.nb[0],
                        other.nb[1], other.nb[2], other.nb[3], other.layout, other.variant);
    }
};

struct Entry {
    std::shared_ptr<Packed> packed;  // none when the budget had no room
    std::size_t size;                // the bytes it was packed from
};

// The budget of the quantized layouts when the environment gives none.
constexpr std::size_t kQuantizedBudget = std::size_t{8} << 30;

std::size_t budget_from_environment(bool quantized) {
    if (const char * value = std::getenv("SDASM_STORE_MB")) {
        return static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) << 20;
    }
    return quantized ? kQuantizedBudget : 0;
}

bool quantized_layout(Layout layout) {
    return layout == Layout::quantized_linear || layout == Layout::quantized_winograd || layout == Layout::quantized_conv;
}

struct State {
    std::mutex mutex;
    std::map<Key, Entry> entries;
    std::atomic<std::size_t> count{0};
    std::size_t used = 0;
    std::size_t largest = 0;  // the most bytes any entry was packed from
    std::size_t budget = budget_from_environment(false);
    // The quantized layouts', apart.
    std::size_t quantized_used = 0;
    std::size_t quantized_budget = budget_from_environment(true);
};

State & state() {
    static State s;
    return s;
}

}  // namespace

bool Packed::ready() const {
    return state.load(std::memory_order_acquire) == kReady;
}

std::shared_ptr<Packed> packed(const ggml_tensor * tensor, Layout layout, std::size_t bytes, std::uint64_t variant) {
    Key key{static_cast<const char *>(tensor->data), tensor->name, tensor->type, {}, {}, layout, variant};
    for (int k = 0; k < 4; ++k) {
        key.ne[k] = tensor->ne[k];
        key.nb[k] = tensor->nb[k];
    }
    State & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.entries.find(key);
    if (it != s.entries.end()) {
        return it->second.packed;
    }
    std::shared_ptr<Packed> p;
    std::size_t & used = quantized_layout(layout) ? s.quantized_used : s.used;
    const std::size_t budget = quantized_layout(layout) ? s.quantized_budget : s.budget;
    if (used + bytes <= budget) {
        void * memory = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
        if (memory != nullptr) {
            used += bytes;
            p = std::shared_ptr<Packed>(new Packed, [](Packed * q) {
                std::free(q->data);
                delete q;
            });
            p->data = static_cast<float *>(memory);
            p->bytes = bytes;
            p->state.store(kFilling, std::memory_order_relaxed);
        }
    }
    s.entries.emplace(key, Entry{p, ggml_nbytes(tensor)});
    s.largest = std::max(s.largest, ggml_nbytes(tensor));
    s.count.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void publish(const std::shared_ptr<Packed> & p) {
    if (p != nullptr) {
        p->state.store(kReady, std::memory_order_release);
    }
}

void forget(const void * begin, std::size_t size) {
    State & s = state();
    if (s.count.load(std::memory_order_relaxed) == 0 || size == 0 || tuning().store_forget == 0) {
        return;
    }
    const char * b = static_cast<const char *>(begin);
    const char * e = b + size;
    std::lock_guard<std::mutex> lock(s.mutex);
    // The entries are in the order of the addresses they were packed from: those
    // that can reach [b, e) begin no further than the largest of them before b.
    Key probe{};
    probe.begin = reinterpret_cast<std::uintptr_t>(b) > s.largest ? b - s.largest : nullptr;
    for (auto it = s.entries.lower_bound(probe); it != s.entries.end() && it->first.begin < e;) {
        const char * eb = it->first.begin;
        if (b < eb + it->second.size) {
            if (it->second.packed != nullptr) {
                (quantized_layout(it->first.layout) ? s.quantized_used : s.used) -= it->second.packed->bytes;
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

extern "C" void sdasm_forget(const void * begin, size_t size) {
    sdasm::forget(begin, size);
}
