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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
    double phases[static_cast<int>(sdasm::Phase::count)] = {};  // nanoseconds, summed over threads
};

thread_local std::uint64_t phase_cycles[static_cast<int>(sdasm::Phase::count)] = {};

std::string shape(const ggml_tensor * t) {
    if (t == nullptr) {
        return "-";
    }
    char buffer[128];
    std::snprintf(buffer, sizeof buffer, "%s[%lld,%lld,%lld,%lld]%s", ggml_type_name(t->type),
                  static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
                  static_cast<long long>(t->ne[2]), static_cast<long long>(t->ne[3]),
                  ggml_is_contiguous(t) ? "" : "nc");
    return buffer;
}

double flops_of(const ggml_tensor * t) {
    switch (t->op) {
    case GGML_OP_MUL_MAT: {
        const ggml_tensor * a = t->src[0];
        const ggml_tensor * b = t->src[1];
        return 2.0 * static_cast<double>(a->ne[0] * a->ne[1] * b->ne[1] * b->ne[2] * b->ne[3]);
    }
    case GGML_OP_CONV_2D: {
        const ggml_tensor * k = t->src[0];
        return 2.0 * static_cast<double>(k->ne[0] * k->ne[1] * k->ne[2] * k->ne[3] * t->ne[0] * t->ne[1] * t->ne[3]);
    }
    case GGML_OP_FLASH_ATTN_EXT: {
        const ggml_tensor * q = t->src[0];
        const ggml_tensor * k = t->src[1];
        const ggml_tensor * v = t->src[2];
        return 2.0 * static_cast<double>(q->ne[1] * k->ne[1] * q->ne[2] * q->ne[3] * (q->ne[0] + v->ne[0]));
    }
    default:
        return 0;
    }
}

class Profile {
public:
    Profile() {
        const char * value = std::getenv("SDASM_PROFILE");
        enabled_ = value != nullptr && *value != '\0' && *value != '0';
    }
    Profile(const Profile &) = delete;
    Profile & operator=(const Profile &) = delete;
    ~Profile() {
        if (enabled_) {
            write();
        }
    }

    bool enabled() const { return enabled_; }

    void node(int graph_nodes, const ggml_tensor * node, int fused, double seconds) {
        const std::string op = op_of(graph_nodes, node, fused);
        const std::string key = shape_of(graph_nodes, node, fused);
        const double f = flops_of(node);
        std::lock_guard<std::mutex> lock(mutex_);
        add(by_shape_[key], seconds, f);
        add(by_op_[op], seconds, f);
    }

    // A thread's time in the kernels of this project for a node, which against
    // the node's time on every thread says how long they waited for each other.
    void busy(int graph_nodes, const ggml_tensor * node, int fused, int threads, double seconds) {
        std::uint64_t cycles[static_cast<int>(sdasm::Phase::count)];
        sdasm::phases_take(cycles);
        const std::string op = op_of(graph_nodes, node, fused);
        const std::string key = shape_of(graph_nodes, node, fused);
        std::lock_guard<std::mutex> lock(mutex_);
        for (Entry * e : {&by_op_[op], &by_shape_[key]}) {
            e->busy += seconds;
            e->threads = threads;
            for (int i = 0; i < static_cast<int>(sdasm::Phase::count); ++i) {
                e->phases[i] += static_cast<double>(cycles[i]);
            }
        }
    }

    void graph(int graph_nodes, double seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        add(by_graph_[graph_nodes], seconds, 0);
    }

private:
    static std::string op_of(int graph_nodes, const ggml_tensor * node, int fused) {
        std::string op = std::to_string(graph_nodes) + "|" + ggml_op_desc(node);
        if (fused > 0) {
            op += "+" + std::to_string(fused);
        }
        return op;
    }

    static std::string shape_of(int graph_nodes, const ggml_tensor * node, int fused) {
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
        for (double c : e.phases) {
            timed += c;
        }
        if (timed > 0 && busy_cycles > 0) {
            std::fprintf(stderr, "  %*s", indent, "");
            for (int i = 0; i < static_cast<int>(sdasm::Phase::count); ++i) {
                if (e.phases[i] > 0) {
                    std::fprintf(stderr, " %s %.1f%%", sdasm::phase_name(static_cast<sdasm::Phase>(i)),
                                 100.0 * e.phases[i] / busy_cycles);
                }
            }
            std::fprintf(stderr, " untimed %.1f%%\n", 100.0 * (busy_cycles - timed) / busy_cycles);
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

    void write() {
        std::lock_guard<std::mutex> lock(mutex_);
        double total = 0;
        for (const auto & [nodes, entry] : by_graph_) {
            total += entry.seconds;
        }
        std::fprintf(stderr, "\n==== SDASM_PROFILE: %.3f s computing graphs ====\n", total);
        for (const auto & [nodes, entry] : by_graph_) {
            std::fprintf(stderr, "\n-- graph of %d nodes: %.3f s in %llu runs --\n", nodes, entry.seconds,
                         static_cast<unsigned long long>(entry.count));
            const std::string prefix = std::to_string(nodes) + "|";
            for (const auto & [op, e] : of_graph(by_op_, prefix)) {
                std::fprintf(stderr, "  %-24s %9.3f s %5.1f%% %8llu calls", op.c_str(), e.seconds,
                             100.0 * e.seconds / entry.seconds, static_cast<unsigned long long>(e.count));
                if (e.flops > 0) {
                    std::fprintf(stderr, "  %7.1f GFLOP/s", e.flops / e.seconds * 1e-9);
                }
                if (e.threads > 0) {
                    std::fprintf(stderr, "  threads busy %5.1f%%", 100.0 * e.busy / (e.seconds * e.threads));
                }
                std::fprintf(stderr, "\n");
                write_phases(e, 24);
            }
            std::fprintf(stderr, "  by operands:\n");
            const auto shapes = of_graph(by_shape_, prefix);
            for (std::size_t i = 0; i < shapes.size() && i < 40; ++i) {
                const auto & [what, e] = shapes[i];
                std::fprintf(stderr, "  %8.3f s %5.1f%% %6llu x  %s", e.seconds, 100.0 * e.seconds / entry.seconds,
                             static_cast<unsigned long long>(e.count), what.c_str());
                if (e.flops > 0) {
                    std::fprintf(stderr, "  %.1f GFLOP/s", e.flops / e.seconds * 1e-9);
                }
                std::fprintf(stderr, "\n");
                write_phases(e, 30);
            }
        }
    }

    bool enabled_ = false;
    std::mutex mutex_;
    std::map<std::string, Entry> by_shape_;
    std::map<std::string, Entry> by_op_;
    std::map<int, Entry> by_graph_;
};

Profile & profile() {
    static Profile p;
    return p;
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

void sdasm_profile_node(const ggml_cgraph * graph, const ggml_tensor * node, int fused, std::uint64_t nanoseconds) {
    profile().node(ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)), node, fused,
                   static_cast<double>(nanoseconds) * 1e-9);
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
    phase_cycles[static_cast<int>(p)] += cycles;
}

void phases_take(std::uint64_t (&cycles)[static_cast<int>(Phase::count)]) {
    for (int i = 0; i < static_cast<int>(Phase::count); ++i) {
        cycles[i] = phase_cycles[i];
        phase_cycles[i] = 0;
    }
}

void profile_busy(const ggml_cgraph * graph, const ggml_tensor * node, int fused, int threads, double seconds) {
    profile().busy(ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)), node, fused, threads, seconds);
}

}  // namespace sdasm

extern "C" {

void sdasm_profile_graph(const ggml_cgraph * graph, std::uint64_t nanoseconds) {
    auto * g = const_cast<ggml_cgraph *>(graph);
    const int n = ggml_graph_n_nodes(g);
    // SDASM_PROFILE=graph writes out the nodes of each graph the first time a
    // graph of its size is computed, with the nodes each takes its operands
    // from, for the fusions to be read off.
    static std::mutex mutex;
    static std::map<int, bool> written;
    const char * value = std::getenv("SDASM_PROFILE");
    if (value != nullptr && std::string(value) == "graph") {
        std::lock_guard<std::mutex> lock(mutex);
        if (!written[n]) {
            written[n] = true;
            std::map<const ggml_tensor *, int> index;
            for (int i = 0; i < n; ++i) {
                index[ggml_graph_node(g, i)] = i;
            }
            std::fprintf(stderr, "== graph of %d nodes\n", n);
            for (int i = 0; i < n; ++i) {
                const ggml_tensor * t = ggml_graph_node(g, i);
                std::fprintf(stderr, "%5d %-16s %s", i, ggml_op_desc(t), shape(t).c_str());
                for (int s = 0; s < 4 && t->src[s] != nullptr; ++s) {
                    const auto it = index.find(t->src[s]);
                    if (it != index.end()) {
                        std::fprintf(stderr, " <%d>", it->second);
                    }
                    else {
                        std::fprintf(stderr, " %s", shape(t->src[s]).c_str());
                    }
                }
                std::fprintf(stderr, "%s\n", t->view_src != nullptr ? " (view)" : "");
            }
        }
    }
    profile().graph(n, static_cast<double>(nanoseconds) * 1e-9);
}

}  // extern "C"
