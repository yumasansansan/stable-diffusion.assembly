// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// Where the time of a graph goes, operation by operation: with SDASM_PROFILE
// set in the environment, ggml's CPU backend (patches/ggml/0002-...) times every
// node on its first thread, from the moment the node starts to the moment the
// last thread is done with it, which is the barrier after it, and this sums the
// times by operation, by the types and shapes of its operands and by graph, and
// writes them to the standard error when the program ends. A node's time is the
// wall-clock time the graph spends on it, the threads that waited for the
// slowest included; the floating-point operations of the products are counted
// too, for the rate they run at.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <map>
#include <mutex>
#include <print>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fusion.h"
#include "kernels.h"
#include "phases.h"
#include "sdasm.h"

#include "ggml.h"

namespace {

struct Entry {
    double seconds = 0;
    double flops = 0;
    std::uint64_t count = 0;
    double busy = 0;  // the time threads spent in the kernels of this project, summed
    int threads = 0;
    std::array<double, static_cast<std::size_t>(sdasm::Phase::count)> phases{};  // nanoseconds, summed over threads
};

thread_local sdasm::Phase_times phase_cycles{};

std::string shape(const ggml_tensor * const t) {
    if (t == nullptr) {
        return "-";
    }
    return std::format("{}[{},{},{},{}]{}", ggml_type_name(t->type), sdasm::extent(t, 0), sdasm::extent(t, 1),
                       sdasm::extent(t, 2), sdasm::extent(t, 3), ggml_is_contiguous(t) ? "" : "nc");
}

// The floating-point operations of the products; of the other operations,
// none counted.
double flops_of(const ggml_tensor * const t) {
    if (t->op == GGML_OP_MUL_MAT) {
        const ggml_tensor * const a = t->src[0];
        const ggml_tensor * const b = t->src[1];
        return 2.0 * static_cast<double>(a->ne[0] * a->ne[1] * b->ne[1] * b->ne[2] * b->ne[3]);
    }
    if (t->op == GGML_OP_CONV_2D) {
        const ggml_tensor * const k = t->src[0];
        return 2.0 * static_cast<double>(k->ne[0] * k->ne[1] * k->ne[2] * k->ne[3] * t->ne[0] * t->ne[1] * t->ne[3]);
    }
    if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * const q = t->src[0];
        const ggml_tensor * const k = t->src[1];
        const ggml_tensor * const v = t->src[2];
        return 2.0 * static_cast<double>(q->ne[1] * k->ne[1] * q->ne[2] * q->ne[3] * (q->ne[0] + v->ne[0]));
    }
    return 0;
}

void write_profile();

class Profile {
public:
    Profile() {
        const char * const value = std::getenv("SDASM_PROFILE");
        enabled_ = value != nullptr && *value != '\0' && *value != '0';
        if (enabled_ && std::atexit(write_profile) != 0) {
            std::print(stderr, "stable-diffusion.assembly: SDASM_PROFILE is set, and the profile cannot be written at the end\n");
        }
    }
    Profile(const Profile &) = delete;
    Profile & operator=(const Profile &) = delete;

    bool enabled() const { return enabled_; }

    void node(int graph_nodes, const ggml_tensor * const node, int fused, double seconds) {
        const std::string op = op_of(graph_nodes, node, fused);
        const std::string key = shape_of(graph_nodes, node, fused);
        const double f = flops_of(node);
        std::lock_guard<std::mutex> const lock(mutex_);
        add(by_shape_[key], seconds, f);
        add(by_op_[op], seconds, f);
    }

    // A thread's time in the kernels of this project for a node, which against
    // the node's time on every thread says how long they waited for each other.
    void busy(int graph_nodes, const ggml_tensor * const node, int fused, int threads, double seconds) {
        sdasm::Phase_times cycles{};
        sdasm::phases_take(cycles);
        const std::string op = op_of(graph_nodes, node, fused);
        const std::string key = shape_of(graph_nodes, node, fused);
        std::lock_guard<std::mutex> const lock(mutex_);
        for (Entry * const e : {&by_op_[op], &by_shape_[key]}) {
            e->busy += seconds;
            e->threads = threads;
            for (std::size_t i = 0; i < cycles.size(); ++i) {
                e->phases[i] += static_cast<double>(cycles[i]);
            }
        }
    }

    void graph(int graph_nodes, double seconds) {
        std::lock_guard<std::mutex> const lock(mutex_);
        add(by_graph_[graph_nodes], seconds, 0);
    }

private:
    static std::string op_of(int graph_nodes, const ggml_tensor * const node, int fused) {
        std::string op = std::to_string(graph_nodes) + "|" + ggml_op_desc(node);
        if (fused > 0) {
            op += "+" + std::to_string(fused);
        }
        return op;
    }

    static std::string shape_of(int graph_nodes, const ggml_tensor * const node, int fused) {
        std::string key = op_of(graph_nodes, node, fused) + " " + shape(node) + " <- " + shape(node->src[0]) + " , " +
                          shape(node->src[1]);
        if (node->src[2] != nullptr) {
            key += " , " + shape(node->src[2]);
        }
        return key;
    }

    // The busy time of an entry's threads by phase, the rest of it untimed.
    void write_phases(const Entry & e, int indent) const {
        const double busy_cycles = e.busy * 1e9;
        double timed = 0;
        for (const double c : e.phases) {
            timed += c;
        }
        if (timed > 0 && busy_cycles > 0) {
            std::print(stderr, "  {}", std::string(static_cast<std::size_t>(indent), ' '));
            for (std::size_t i = 0; i < e.phases.size(); ++i) {
                if (e.phases[i] > 0) {
                    std::print(stderr, " {} {:.1f}%", sdasm::phase_name(static_cast<sdasm::Phase>(i)),
                               100.0 * e.phases[i] / busy_cycles);
                }
            }
            std::print(stderr, " untimed {:.1f}%\n", 100.0 * (busy_cycles - timed) / busy_cycles);
        }
    }

    static void add(Entry & e, double seconds, double flops) {
        e.seconds += seconds;
        e.flops += flops;
        e.count += 1;
    }

    static std::vector<std::pair<std::string, Entry>> of_graph(const std::map<std::string, Entry> & all,
                                                               const std::string & prefix) {
        std::vector<std::pair<std::string, Entry>> list;
        for (const auto & [key, entry] : all) {
            if (key.starts_with(prefix)) {
                list.emplace_back(key.substr(prefix.size()), entry);
            }
        }
        std::sort(list.begin(), list.end(), [](const auto & a, const auto & b) { return a.second.seconds > b.second.seconds; });
        return list;
    }

public:
    void write() {
        std::lock_guard<std::mutex> const lock(mutex_);
        double total = 0;
        for (const auto & [nodes, entry] : by_graph_) {
            total += entry.seconds;
        }
        std::print(stderr, "\n==== SDASM_PROFILE: {:.3f} s computing graphs ====\n", total);
        for (const auto & [nodes, entry] : by_graph_) {
            std::print(stderr, "\n-- graph of {} nodes: {:.3f} s in {} runs --\n", nodes, entry.seconds, entry.count);
            const std::string prefix = std::to_string(nodes) + "|";
            for (const auto & [op, e] : of_graph(by_op_, prefix)) {
                std::print(stderr, "  {:<24} {:9.3f} s {:5.1f}% {:8} calls", op, e.seconds, 100.0 * e.seconds / entry.seconds,
                           e.count);
                if (e.flops > 0) {
                    std::print(stderr, "  {:7.1f} GFLOP/s", e.flops / e.seconds * 1e-9);
                }
                if (e.threads > 0) {
                    std::print(stderr, "  threads busy {:5.1f}%", 100.0 * e.busy / (e.seconds * e.threads));
                }
                std::print(stderr, "\n");
                write_phases(e, 24);
            }
            std::print(stderr, "  by operands:\n");
            const auto shapes = of_graph(by_shape_, prefix);
            for (std::size_t i = 0; i < shapes.size() && i < 40; ++i) {
                const auto & [what, e] = shapes[i];
                std::print(stderr, "  {:8.3f} s {:5.1f}% {:6} x  {}", e.seconds, 100.0 * e.seconds / entry.seconds, e.count,
                           what);
                if (e.flops > 0) {
                    std::print(stderr, "  {:.1f} GFLOP/s", e.flops / e.seconds * 1e-9);
                }
                std::print(stderr, "\n");
                write_phases(e, 30);
            }
        }
    }

private:
    bool enabled_ = false;
    std::mutex mutex_;
    std::map<std::string, Entry> by_shape_;
    std::map<std::string, Entry> by_op_;
    std::map<int, Entry> by_graph_;
};

// The profile is never destroyed -- the threads of a pool may still be about
// when the program ends -- and what it has summed is written out when the
// program ends, by write_profile(), which its constructor has std::atexit()
// call.
Profile & profile() {
    [[clang::no_destroy]] static Profile p;
    return p;
}

void write_profile() {
    profile().write();
}

}  // namespace

extern "C" {

bool sdasm_profile_enabled(void) {
    return profile().enabled();
}

std::uint64_t sdasm_profile_now(void) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

void sdasm_profile_node(int n_nodes, const ggml_tensor * const node, int fused, std::uint64_t nanoseconds) {
    profile().node(n_nodes, node, fused, static_cast<double>(nanoseconds) * 1e-9);
}

}  // extern "C"

namespace sdasm {

const char * phase_name(Phase p) {
    switch (p) {
    case Phase::pack_weights:
        return "weights";
    case Phase::pack_inputs:
        return "inputs";
    case Phase::product:
        return "products";
    case Phase::softmax:
        return "softmax";
    case Phase::transform_out:
        return "outputs";
    case Phase::epilogue:
        return "chain";
    case Phase::wait:
        return "waiting";
    case Phase::count:
        break;
    }
    return "?";
}

bool phases_enabled() {
    static const bool enabled = profile().enabled();
    return enabled;
}

void phase_add(Phase p, std::uint64_t cycles) {
    phase_cycles.at(static_cast<std::size_t>(p)) += cycles;
}

void phases_take(Phase_times & cycles) {
    cycles = phase_cycles;
    phase_cycles = {};
}

void profile_busy(int n_nodes, const ggml_tensor * const node, int fused, int threads, double seconds) {
    profile().busy(n_nodes, node, fused, threads, seconds);
}

}  // namespace sdasm

extern "C" {

void sdasm_profile_graph(ggml_tensor * const * const nodes, int n_nodes, std::uint64_t nanoseconds) {
    if (nodes == nullptr || n_nodes <= 0) {
        return;
    }
    const sdasm::Graph_nodes graph = sdasm::graph_nodes(nodes, n_nodes);
    const int n = n_nodes;
    // SDASM_PROFILE=graph writes out the nodes of each graph the first time a
    // graph of its size is computed, with the nodes each takes its operands
    // from, for the fusions to be read off.
    [[clang::no_destroy]] static std::mutex mutex;
    [[clang::no_destroy]] static std::map<int, bool> written;
    const char * const value = std::getenv("SDASM_PROFILE");
    if (value != nullptr && std::string(value) == "graph") {
        std::lock_guard<std::mutex> const lock(mutex);
        if (!written[n]) {
            written[n] = true;
            std::map<const ggml_tensor *, int> index;
            for (int i = 0; i < n; ++i) {
                index[sdasm::node_at(graph, i)] = i;
            }
            std::print(stderr, "== graph of {} nodes\n", n);
            for (int i = 0; i < n; ++i) {
                const ggml_tensor * const t = sdasm::node_at(graph, i);
                std::print(stderr, "{:5} {:<16} {}", i, ggml_op_desc(t), shape(t));
                for (const ggml_tensor * const source : std::span(t->src).first(4)) {
                    if (source == nullptr) {
                        break;
                    }
                    const auto it = index.find(source);
                    if (it != index.end()) {
                        std::print(stderr, " <{}>", it->second);
                    }
                    else {
                        std::print(stderr, " {}", shape(source));
                    }
                }
                std::print(stderr, "{}\n", t->view_src != nullptr ? " (view)" : "");
            }
        }
    }
    profile().graph(n, static_cast<double>(nanoseconds) * 1e-9);
}

}  // extern "C"
