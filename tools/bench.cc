// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     sdasm_bench [--threads N] [--seconds S]
//                 [--forms assembly,avx2,avx512,amx,cxx,upstream]
//                 [--variant label:[isa=<level>,]name=value,...]...
//                 [--only <substring>] [--tsv <file>]
//     sdasm_bench --info
//     sdasm_bench --peak [--threads N] [--seconds S] [--tsv <file>]
//
// Times the operations of SD 1.5's UNet and VAE at 512 by 512 that the kernels
// take, at their shapes in the model, through ggml's CPU backend, in each form
// of the kernels: each operation is computed again and again for the seconds
// given, after one computation that is not counted, and its rate is the
// floating-point operations of a computation over the fastest time of one. The
// forms of an operation are timed one after another, so that they meet the
// processor in about the same state.
//
// The forms are those of SDASM_KERNELS -- assembly, cxx and upstream -- and the
// kernels written by hand held to a level of the instruction set, by its name
// (avx2, avx512, amx), which is left out on a processor without it. --tsv
// appends each rate to a file as well, a line of tab-separated fields for each
// operation and form: the operation, the form, the level its kernels ran at
// (ggml for ggml's), the threads, GFLOP/s and the fastest time in seconds.
// --info prints what the processor has of the extensions the kernels are
// written for, and the levels. --peak times the most FMAs the processor makes
// on the threads given, at each width it has (peak_x86_64.S): the rate an
// operation's is a share of, written as the operation "fma peak", in the form
// avx2 or avx512.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kernels.h"

#include "ggml-cpu.h"
#include "ggml.h"

namespace {

void fill(ggml_tensor * t) {
    static std::mt19937 g(1);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const std::int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        auto * p = static_cast<float *>(t->data);
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = d(g);
        }
    }
    else {
        auto * p = static_cast<ggml_fp16_t *>(t->data);
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = ggml_fp32_to_fp16(d(g));
        }
    }
}

struct Case {
    std::string name;
    double flops;
    std::function<ggml_tensor *(ggml_context *)> build;
};

Case conv(int w, int h, int ic, int oc, int k, int n) {
    char name[96];
    std::snprintf(name, sizeof name, "conv2d %dx%dx%d -> %d, %dx%d, %d images", w, h, ic, oc, k, k, n);
    const double flops = 2.0 * w * h * ic * oc * k * k * n;
    return {name, flops, [=](ggml_context * ctx) {
                ggml_tensor * kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k, k, ic, oc);
                ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, ic, n);
                fill(kernel);
                fill(x);
                return ggml_conv_2d_direct(ctx, kernel, x, 1, 1, k / 2, k / 2, 1, 1);
            }};
}

Case linear(int in, int out, int rows, int batch) {
    char name[96];
    std::snprintf(name, sizeof name, "linear %d -> %d, %d rows x %d", in, out, rows, batch);
    const double flops = 2.0 * in * out * rows * batch;
    return {name, flops, [=](ggml_context * ctx) {
                ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, in, out);
                ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, rows, batch);
                fill(w);
                fill(x);
                return ggml_mul_mat(ctx, w, x);
            }};
}

Case attention(int d, int heads, int lq, int lk, int batch) {
    char name[96];
    std::snprintf(name, sizeof name, "attention %d x %d heads, %d queries, %d keys, %d batches", d, heads, lq, lk, batch);
    const double flops = 4.0 * d * heads * lq * static_cast<double>(lk) * batch;
    return {name, flops, [=](ggml_context * ctx) {
                ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d * heads, lq, batch);
                ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d * heads, lk, batch);
                ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d * heads, lk, batch);
                fill(q);
                fill(k);
                fill(v);
                ggml_tensor * qv = ggml_permute(ctx, ggml_reshape_4d(ctx, q, d, heads, lq, batch), 0, 2, 1, 3);
                ggml_tensor * kv = ggml_permute(ctx, ggml_reshape_4d(ctx, k, d, heads, lk, batch), 0, 2, 1, 3);
                ggml_tensor * vv = ggml_permute(ctx, ggml_reshape_4d(ctx, v, d, heads, lk, batch), 0, 2, 1, 3);
                ggml_tensor * out = ggml_flash_attn_ext(ctx, qv, kv, vv, nullptr, 1.0f / std::sqrt(static_cast<float>(d)), 0, 0);
                ggml_prec_set_acc(out, GGML_PREC_F32);
                return out;
            }};
}

std::vector<Case> cases() {
    return {
        // The UNet, for two images: the conditional and the unconditional one.
        conv(64, 64, 320, 320, 3, 2),
        conv(32, 32, 640, 640, 3, 2),
        conv(32, 32, 1280, 1280, 3, 2),
        conv(16, 16, 1280, 1280, 3, 2),
        conv(8, 8, 1280, 1280, 3, 2),
        conv(16, 16, 2560, 1280, 3, 2),
        conv(64, 64, 320, 320, 1, 2),
        linear(320, 320, 4096, 2),
        linear(640, 640, 1024, 2),
        linear(1280, 1280, 256, 2),
        linear(320, 2560, 4096, 2),
        linear(1280, 10240, 256, 2),
        linear(1280, 320, 4096, 2),
        linear(768, 320, 77, 2),
        attention(40, 8, 4096, 4096, 2),
        attention(80, 8, 1024, 1024, 2),
        attention(40, 8, 4096, 77, 2),
        // The VAE's decoder.
        conv(64, 64, 512, 512, 3, 1),
        conv(128, 128, 512, 512, 3, 1),
        conv(256, 256, 256, 256, 3, 1),
        conv(256, 256, 512, 512, 3, 1),
        conv(512, 512, 128, 128, 3, 1),
        conv(512, 512, 256, 256, 3, 1),
        attention(512, 1, 4096, 4096, 1),
    };
}

struct Config {
    std::string label;
    sdasm::Kernels form;
    sdasm::Isa level;  // the cap of the levels of the kernels written by hand
    sdasm::Tuning tuning;
};

int info() {
    const sdasm::Cpu & c = sdasm::cpu();
    std::printf("avx2 %d avx512 %d avx512_bf16 %d avx512_fp16 %d avx_vnni %d amx_bf16 %d amx_fp16 %d\n", c.avx2, c.avx512,
                c.avx512_bf16, c.avx512_fp16, c.avx_vnni, c.amx_bf16, c.amx_fp16);
    std::printf("levels:");
    for (const sdasm::Isa level : {sdasm::Isa::avx2, sdasm::Isa::avx512, sdasm::Isa::amx}) {
        if (sdasm::isa_available() >= level) {
            std::printf(" %s", sdasm::isa_name(level));
        }
    }
    std::printf("\nkernels in effect: %s\n", sdasm::kernels_isa());
    return 0;
}

#if SDASM_X86_64
extern "C" {
void sdasm_peak_avx2(std::int64_t n, const float * seed);
void sdasm_peak_avx512(std::int64_t n, const float * seed);
}
#endif

double now();

// The FMAs of `threads` threads at once, each making `kernel`'s n times over
// and again for the seconds given; the rate of the fastest round.
double peak(void (*kernel)(std::int64_t, const float *), double flops_per_n, int threads, double seconds) {
    alignas(64) static float seed[64];
    std::mt19937 g(7);
    std::uniform_real_distribution<float> d(0.5f, 1.5f);
    for (float & x : seed) {
        x = d(g);
    }
    constexpr std::int64_t kRound = 1 << 22;
    double best = 0;
    const double start = now();
    while (now() - start < seconds) {
        std::vector<std::thread> pool;
        const double t0 = now();
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&] { kernel(kRound, seed); });
        }
        for (auto & t : pool) {
            t.join();
        }
        best = std::max(best, flops_per_n * static_cast<double>(kRound) * threads / (now() - t0));
    }
    return best;
}

int peaks(int threads, double seconds, std::FILE * table) {
#if SDASM_X86_64
    std::printf("%d threads, %.1f s at each width; GFLOP/s of the fastest round\n", threads, seconds);
    const auto one = [&](const char * name, void (*kernel)(std::int64_t, const float *), double flops) {
        const double rate = peak(kernel, flops, threads, seconds) * 1e-9;
        std::printf("fma peak %-8s %10.1f\n", name, rate);
        if (table != nullptr) {
            std::fprintf(table, "fma peak\t%s\t%s\t%d\t%.2f\t0\n", name, name, threads, rate);
        }
    };
    if (sdasm::cpu().avx2) {
        one("avx2", sdasm_peak_avx2, 12.0 * 8 * 2);
    }
    if (sdasm::cpu().avx512) {
        one("avx512", sdasm_peak_avx512, 24.0 * 16 * 2);
    }
#else
    (void) threads, (void) seconds, (void) table;
#endif
    return 0;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char ** argv) {
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    double seconds = 2.0;
    std::string only;
    std::string tsv;
    bool peak_only = false;
    std::vector<Config> configs;
    // A form by its name: one of SDASM_KERNELS, or a level of the kernels
    // written by hand, which is false on a processor without that level.
    const auto add_form = [&](const std::string & f) {
        for (auto k : {sdasm::Kernels::assembly, sdasm::Kernels::cxx, sdasm::Kernels::upstream}) {
            if (f == sdasm::kernels_name(k)) {
                configs.push_back({f, k, sdasm::Isa::amx, sdasm::tuning()});
                return true;
            }
        }
        sdasm::Isa level{};
        if (sdasm::isa_named(f.c_str(), level) && level != sdasm::Isa::cxx) {
            if (sdasm::isa_available() < level) {
                std::fprintf(stderr, "%s: this processor has no %s; leaving it out\n", argv[0], f.c_str());
                return true;
            }
            configs.push_back({f, sdasm::Kernels::assembly, level, sdasm::tuning()});
            return true;
        }
        return false;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        }
        else if (a == "--seconds" && i + 1 < argc) {
            seconds = std::atof(argv[++i]);
        }
        else if (a == "--only" && i + 1 < argc) {
            only = argv[++i];
        }
        else if (a == "--tsv" && i + 1 < argc) {
            tsv = argv[++i];
        }
        else if (a == "--info") {
            return info();
        }
        else if (a == "--peak") {
            peak_only = true;
        }
        else if (a == "--forms" && i + 1 < argc) {
            const std::string list = argv[++i];
            for (std::size_t start = 0; start <= list.size();) {
                const std::size_t comma = std::min(list.find(',', start), list.size());
                const std::string f = list.substr(start, comma - start);
                if (!f.empty() && !add_form(f)) {
                    std::fprintf(stderr, "%s: no form is called %s\n", argv[0], f.c_str());
                    return 2;
                }
                start = comma + 1;
            }
        }
        else if (a == "--variant" && i + 1 < argc) {
            // label:name=value,... -- the kernels written by hand, tuned so;
            // isa=<level> among the names holds them to that level, and leaves
            // the variant out on a processor without it.
            const std::string v = argv[++i];
            const std::size_t colon = v.find(':');
            Config c{v.substr(0, colon), sdasm::Kernels::assembly, sdasm::Isa::amx, sdasm::tuning()};
            std::string names = colon != std::string::npos ? v.substr(colon + 1) : std::string();
            bool level_given = false;
            if (const std::size_t at = names.find("isa="); at != std::string::npos) {
                level_given = true;
                const std::size_t end = std::min(names.find(',', at), names.size());
                const std::string level = names.substr(at + 4, end - at - 4);
                if (!sdasm::isa_named(level.c_str(), c.level) || c.level == sdasm::Isa::cxx) {
                    std::fprintf(stderr, "%s: no level is called %s\n", argv[0], level.c_str());
                    return 2;
                }
                names.erase(at, end < names.size() ? end - at + 1 : end - at);
                if (!names.empty() && names.back() == ',') {
                    names.pop_back();
                }
            }
            if (!names.empty() && !sdasm::parse_tuning(names.c_str(), c.tuning)) {
                std::fprintf(stderr, "%s: a name it does not know in %s\n", argv[0], v.c_str());
                return 2;
            }
            if (level_given && sdasm::isa_available() < c.level) {
                std::fprintf(stderr, "%s: this processor has no %s; leaving %s out\n", argv[0],
                             sdasm::isa_name(c.level), c.label.c_str());
                continue;
            }
            if (c.tuning.bf16 != 0 && !sdasm::cpu().avx512_bf16) {
                std::fprintf(stderr, "%s: this processor has no AVX512-BF16; leaving %s out\n", argv[0], c.label.c_str());
                continue;
            }
            configs.push_back(c);
        }
        else {
            std::fprintf(stderr,
                         "usage: %s [--threads N] [--seconds S] [--forms assembly,avx2,avx512,amx,cxx,upstream]\n"
                         "          [--variant label:name=value,...]... [--only text] [--tsv file]\n"
                         "       %s --info\n"
                         "       %s --peak [--threads N] [--seconds S] [--tsv file]\n",
                         argv[0], argv[0], argv[0]);
            return 2;
        }
    }
    if (configs.empty()) {
        for (const char * f : {"assembly", "cxx", "upstream"}) {
            add_form(f);
        }
    }
    std::FILE * table = nullptr;
    if (!tsv.empty()) {
        table = std::fopen(tsv.c_str(), "a");
        if (table == nullptr) {
            std::fprintf(stderr, "%s: cannot open %s\n", argv[0], tsv.c_str());
            return 1;
        }
    }
    if (peak_only) {
        const int status = peaks(threads, seconds, table);
        if (table != nullptr) {
            std::fclose(table);
        }
        return status;
    }
    std::printf("%d threads, %.1f s an operation and configuration, taking turns; GFLOP/s of the fastest computation\n",
                threads, seconds);
    std::printf("%-56s", "");
    for (const auto & c : configs) {
        std::printf(" %10s", c.label.c_str());
    }
    std::printf("\n");
    std::vector<std::uint8_t> work;
    for (const Case & c : cases()) {
        if (!only.empty() && c.name.find(only) == std::string::npos) {
            continue;
        }
        ggml_init_params params{std::size_t{1} << 30, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * out = c.build(ctx);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        // The configurations take turns, a computation each, so that a processor
        // that slows as it warms slows them alike.
        const auto compute = [&](const Config & config) {
            sdasm::set_kernels(config.form);
            sdasm::set_isa_cap(config.level);
            sdasm::set_tuning(config.tuning);
            ggml_cplan plan = ggml_graph_plan(graph, threads, nullptr);
            if (work.size() < plan.work_size) {
                work.resize(plan.work_size);
            }
            plan.work_data = work.data();
            const double t0 = now();
            ggml_graph_compute(graph, &plan);
            return now() - t0;
        };
        std::vector<double> best(configs.size(), HUGE_VAL);
        for (const auto & config : configs) {
            compute(config);
        }
        const double start = now();
        while (now() - start < seconds * static_cast<double>(configs.size())) {
            for (std::size_t i = 0; i < configs.size(); ++i) {
                best[i] = std::min(best[i], compute(configs[i]));
            }
        }
        std::printf("%-56s", c.name.c_str());
        for (double b : best) {
            std::printf(" %10.1f", c.flops / b * 1e-9);
        }
        std::printf("\n");
        std::fflush(stdout);
        if (table != nullptr) {
            for (std::size_t i = 0; i < configs.size(); ++i) {
                sdasm::set_kernels(configs[i].form);
                sdasm::set_isa_cap(configs[i].level);
                const bool upstream = configs[i].form == sdasm::Kernels::upstream;
                std::fprintf(table, "%s\t%s\t%s\t%d\t%.2f\t%.6f\n", c.name.c_str(), configs[i].label.c_str(),
                             upstream ? "ggml" : sdasm::kernels_isa(), threads, c.flops / best[i] * 1e-9, best[i]);
            }
            std::fflush(table);
        }
        ggml_free(ctx);
    }
    if (table != nullptr) {
        std::fclose(table);
    }
    return 0;
}
