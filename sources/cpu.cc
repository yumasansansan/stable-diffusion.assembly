// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// Which form of the kernels runs. The kernels written by hand are taken unless
// the environment says otherwise -- SDASM_KERNELS=cxx takes the C++ kernels
// throughout, for a measurement of the one form against the other on the same
// build, and SDASM_KERNELS=upstream takes none of this project's -- and of them,
// for each kernel, the one for the richest level of the instruction set that the
// processor has (isa()), down to the C++ one where none is there; SDASM_ISA
// holds them to a level below, for a measurement of one level against another.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gemm.h"
#include "kernels.h"

namespace sdasm {

namespace {

Cpu detect() {
    Cpu c;
#if SDASM_X86_64
    __builtin_cpu_init();
    c.intel = __builtin_cpu_is("intel");
    c.avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    c.avx512 = c.avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
               __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
    c.avx512_bf16 = c.avx512 && __builtin_cpu_supports("avx512bf16");
    c.avx512_fp16 = c.avx512 && __builtin_cpu_supports("avx512fp16");
    c.avx_vnni = c.avx2 && __builtin_cpu_supports("avxvnni");
    c.avx512_vnni = c.avx512 && __builtin_cpu_supports("avx512vnni");
    // AMX's tiles are a state of the processor that Linux gives a process only
    // when it asks (sdasm_amx_request(), x86_64/amx.S), and that a system which
    // does not know them never gives: what cpuid says is not enough.
    c.amx_bf16 = c.avx512_bf16 && __builtin_cpu_supports("amx-tile") && __builtin_cpu_supports("amx-bf16") &&
                 sdasm_amx_request() == 0;
    c.amx_fp16 = c.amx_bf16 && __builtin_cpu_supports("amx-fp16");
#endif
    return c;
}

Kernels kernels_from_environment() {
    const char * value = std::getenv("SDASM_KERNELS");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "assembly") == 0) {
        return Kernels::assembly;
    }
    if (std::strcmp(value, "cxx") == 0) {
        return Kernels::cxx;
    }
    if (std::strcmp(value, "upstream") == 0) {
        return Kernels::upstream;
    }
    std::fprintf(stderr,
                 "stable-diffusion.assembly: SDASM_KERNELS=%s is none of assembly, cxx and upstream; "
                 "taking assembly\n",
                 value);
    return Kernels::assembly;
}

}  // namespace

namespace {

// The form chosen, set from the environment when it is first asked for. The
// tests set it themselves, between operations, to compute the same one in each
// form (set_kernels()).
std::atomic<int> chosen_kernels{-1};

}  // namespace

Kernels kernels() {
    int k = chosen_kernels.load(std::memory_order_relaxed);
    if (k < 0) {
        int unset = -1;
        k = static_cast<int>(kernels_from_environment());
        if (!chosen_kernels.compare_exchange_strong(unset, k, std::memory_order_relaxed)) {
            k = unset;
        }
    }
    return static_cast<Kernels>(k);
}

void set_kernels(Kernels k) {
    chosen_kernels.store(static_cast<int>(k), std::memory_order_relaxed);
}

const char * kernels_name(Kernels k) {
    switch (k) {
    case Kernels::assembly:
        return "assembly";
    case Kernels::cxx:
        return "cxx";
    case Kernels::upstream:
        return "upstream";
    }
    return "?";
}

namespace {

struct Tuning_field {
    const char * name;
    int Tuning::*field;
};

constexpr Tuning_field kTuningFields[] = {
    {"conv_kc", &Tuning::conv_kc},
    {"conv_mc", &Tuning::conv_mc},
    {"conv_nc", &Tuning::conv_nc},
    {"conv_rows_outer", &Tuning::conv_rows_outer},
    {"conv_shared_columns", &Tuning::conv_shared_columns},
    {"conv_columns_bytes", &Tuning::conv_columns_bytes},
    {"mul_mat_kc", &Tuning::mul_mat_kc},
    {"mul_mat_mc", &Tuning::mul_mat_mc},
    {"mul_mat_shared_rows", &Tuning::mul_mat_shared_rows},
    {"amx_kc", &Tuning::amx_kc},
    {"amx_mc", &Tuning::amx_mc},
    {"amx_attention", &Tuning::amx_attention},
    {"amx_queries", &Tuning::amx_queries},
    {"amx_fp16", &Tuning::amx_fp16},
    {"attention_queries", &Tuning::attention_queries},
    {"attention_keys", &Tuning::attention_keys},
    {"attention_sum_keys", &Tuning::attention_sum_keys},
    {"attention_sum_dims", &Tuning::attention_sum_dims},
    {"winograd", &Tuning::winograd},
    {"winograd_positions", &Tuning::winograd_positions},
    {"winograd4_positions", &Tuning::winograd4_positions},
    {"winograd_kc", &Tuning::winograd_kc},
    {"winograd_mc", &Tuning::winograd_mc},
    {"winograd_tiles", &Tuning::winograd_tiles},
    {"winograd_small_tiles", &Tuning::winograd_small_tiles},
    {"winograd_small_mc", &Tuning::winograd_small_mc},
    {"winograd_small_kc", &Tuning::winograd_small_kc},
    {"prefetch", &Tuning::prefetch},
    {"wide", &Tuning::wide},
    {"bf16", &Tuning::bf16},
    {"fuse", &Tuning::fuse},
    {"fuse_residual", &Tuning::fuse_residual},
    {"guided", &Tuning::guided},
    {"store_forget", &Tuning::store_forget},
    {"bits", &Tuning::bits},
    {"input_bits", &Tuning::input_bits},
    {"group", &Tuning::group},
    {"weight_group", &Tuning::weight_group},
    {"quantize", &Tuning::quantize},
};

Tuning tuning_from_environment() {
    Tuning t;
    const char * value = std::getenv("SDASM_TUNE");
    if (value != nullptr && !parse_tuning(value, t)) {
        std::fprintf(stderr, "stable-diffusion.assembly: SDASM_TUNE=%s has a name it does not know\n", value);
    }
    return t;
}

Tuning & current_tuning() {
    static Tuning t = tuning_from_environment();
    return t;
}

}  // namespace

bool parse_tuning(const char * text, Tuning & t) {
    bool ok = true;
    std::string all = text;
    for (std::size_t start = 0; start < all.size();) {
        std::size_t comma = all.find(',', start);
        if (comma == std::string::npos) {
            comma = all.size();
        }
        const std::string item = all.substr(start, comma - start);
        start = comma + 1;
        const std::size_t equals = item.find('=');
        if (equals == std::string::npos) {
            ok = false;
            continue;
        }
        const std::string name = item.substr(0, equals);
        bool known = false;
        for (const auto & f : kTuningFields) {
            if (name == f.name) {
                t.*f.field = std::atoi(item.c_str() + equals + 1);
                known = true;
            }
        }
        ok = ok && known;
    }
    return ok;
}

const Tuning & tuning() {
    return current_tuning();
}

void set_tuning(const Tuning & t) {
    current_tuning() = t;
}

const Cpu & cpu() {
    static const Cpu c = detect();
    return c;
}

namespace {

constexpr const char * kIsaNames[] = {"cxx", "avx2", "avx512", "amx"};

// The cap of the levels, set from the environment when it is first asked for:
// none, unless SDASM_ISA names one.
std::atomic<int> isa_cap{-1};

int cap_from_environment() {
    const char * value = std::getenv("SDASM_ISA");
    Isa level = Isa::amx;
    if (value != nullptr && *value != '\0' && !isa_named(value, level)) {
        std::fprintf(stderr, "stable-diffusion.assembly: SDASM_ISA=%s is none of cxx, avx2, avx512 and amx; taking amx\n",
                     value);
    }
    return static_cast<int>(level);
}

}  // namespace

Isa isa_available() {
    const Cpu & c = cpu();
    return c.amx_bf16 ? Isa::amx : c.avx512 ? Isa::avx512 : c.avx2 ? Isa::avx2 : Isa::cxx;
}

Isa isa() {
    if (kernels() != Kernels::assembly) {
        return Isa::cxx;
    }
    int cap = isa_cap.load(std::memory_order_relaxed);
    if (cap < 0) {
        int unset = -1;
        cap = cap_from_environment();
        if (!isa_cap.compare_exchange_strong(unset, cap, std::memory_order_relaxed)) {
            cap = unset;
        }
    }
    return static_cast<Isa>(std::min(cap, static_cast<int>(isa_available())));
}

void set_isa_cap(Isa cap) {
    isa_cap.store(static_cast<int>(cap), std::memory_order_relaxed);
}

const char * isa_name(Isa level) {
    return kIsaNames[static_cast<int>(level)];
}

bool isa_named(const char * name, Isa & level) {
    for (int i = 0; i < 4; ++i) {
        if (std::strcmp(name, kIsaNames[i]) == 0) {
            level = static_cast<Isa>(i);
            return true;
        }
    }
    return false;
}

const char * kernels_isa() {
    return isa_name(isa());
}

bool avx2_kernels() {
    return SDASM_X86_64 != 0 && isa() >= Isa::avx2;
}

bool avx512_kernels() {
    return SDASM_X86_64 != 0 && isa() >= Isa::avx512;
}

bool amx_kernels() {
    return SDASM_X86_64 != 0 && isa() >= Isa::amx;
}

Microkernel microkernel() {
#if SDASM_X86_64
    if (avx512_kernels()) {
        return sdasm_gemm_6x16_avx512;
    }
    if (avx2_kernels()) {
        return sdasm_gemm_6x16_avx2;
    }
#endif
    return gemm_6x16_cxx;
}

Split_form split_form(Split_use use) {
    Split_form f;
#if SDASM_X86_64
    const bool tiles = amx_kernels();
    const bool vector = !tiles && avx512_kernels() && cpu().avx512_bf16 && tuning().bf16 != 0;
    if (!tiles && !vector) {
        return f;
    }
    f.tiles = tiles;
    if (tiles && use != Split_use::winograd && cpu().amx_fp16 && tuning().amx_fp16 != 0) {
        f.fp16 = true;
        if (use == Split_use::linear) {
            f.kernel = sdasm_amx_fp16_2x2_a2;
            f.b_bytes = kAmxTileBytes / 2;
        }
        else {
            f.kernel = sdasm_amx_fp16_2x2_b2;
            f.a_bytes = kAmxTileBytes / 2;
        }
        f.pack_rows = sdasm_amx_pack_rows_fp16;
        f.pack_weights = sdasm_amx_pack_weights_fp16;
        f.pack_weight_rows = sdasm_amx_pack_rows_f16_one;
        f.pack_panel = sdasm_amx_pack_panel_fp16;
        return f;
    }
    f.kernel = tiles ? sdasm_amx_bf16_2x2 : sdasm_bf16_2x2_avx512;
    f.pack_rows = sdasm_amx_pack_rows;
    f.pack_weights = sdasm_amx_pack_weights;
    f.pack_weight_rows = sdasm_amx_pack_rows_f16;
    f.pack_panel = sdasm_amx_pack_panel;
#else
    (void) use;
#endif
    return f;
}

void split_begin(const Split_form & form) {
#if SDASM_X86_64
    if (form.tiles) {
        sdasm_amx_configure();
    }
#else
    (void) form;
#endif
}

void split_end(const Split_form & form) {
#if SDASM_X86_64
    if (form.tiles) {
        sdasm_amx_release();
    }
#else
    (void) form;
#endif
}

Microkernel_wide microkernel_wide() {
#if SDASM_X86_64
    const int wide = tuning().wide;
    if (avx512_kernels() && (wide > 0 || (wide < 0 && cpu().intel))) {
        return sdasm_gemm_6x32_avx512;
    }
#endif
    return nullptr;
}

Scores_kernel scores_kernel() {
#if SDASM_X86_64
    if (avx512_kernels()) {
        return sdasm_scores_6x16_avx512;
    }
    if (avx2_kernels()) {
        return sdasm_scores_6x16_avx2;
    }
#endif
    return scores_6x16_cxx;
}

Exponentials exponentials_kernel() {
#if SDASM_X86_64
    if (avx512_kernels()) {
        return sdasm_exponentials_avx512;
    }
    if (avx2_kernels()) {
        return sdasm_exponentials_avx2;
    }
#endif
    return exponentials_cxx;
}

}  // namespace sdasm
