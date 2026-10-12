// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     sdasm_ops_test <operation> [<threads>]
//
// Computes the operations this project takes through ggml's CPU backend, as
// stable-diffusion.cpp has them computed, in each form of the kernels -- the
// ones written by hand, the C++ ones and ggml's own -- and computes each again
// in double precision, from the same numbers, to measure how far each form is
// from the exact result. The measure of an output is its error divided by the
// sum of the magnitudes of the products it adds up, which is what single
// precision can be asked to answer for: a sum of k products in single precision
// is within about k units of 2^-24 of that, whatever cancels in it.
//
// A test passes when the kernels written by hand and the C++ ones are each as
// close to the exact result as single precision allows -- they need not give
// the same numbers: each sums in the order that suits it; ggml's own form is
// measured beside them, and what it gives is shown, not judged: it rounds the
// image to FP16 before the products when the weights are FP16, as
// stable-diffusion.cpp has always computed them.

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

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

struct Result {
    std::vector<float> values;
    double worst = 0;  // the largest error, relative to the magnitudes summed
};

std::mt19937 & generator() {
    static std::mt19937 g(20261011);
    return g;
}

void fill(ggml_tensor * t, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    const std::int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        auto * p = static_cast<float *>(t->data);
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = d(generator());
        }
    }
    else {
        auto * p = static_cast<ggml_fp16_t *>(t->data);
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = ggml_fp32_to_fp16(d(generator()));
        }
    }
}

double at(const ggml_tensor * t, std::int64_t i0, std::int64_t i1, std::int64_t i2, std::int64_t i3) {
    const char * p = static_cast<const char *>(t->data) + i0 * static_cast<std::int64_t>(t->nb[0]) +
                     i1 * static_cast<std::int64_t>(t->nb[1]) + i2 * static_cast<std::int64_t>(t->nb[2]) +
                     i3 * static_cast<std::int64_t>(t->nb[3]);
    if (t->type == GGML_TYPE_F32) {
        float v;
        std::memcpy(&v, p, sizeof v);
        return static_cast<double>(v);
    }
    ggml_fp16_t h;
    std::memcpy(&h, p, sizeof h);
    return static_cast<double>(ggml_fp16_to_fp32(h));
}

// A form of the kernels a test computes in: those written by hand, held to a
// level of the instruction set, the C++ ones, or ggml's own.
struct Form {
    const char * name;
    sdasm::Kernels kernels;
    sdasm::Isa level;
};

// The kernels written by hand at each level the processor has -- the levels
// below its richest are tested on it too, as a processor that has only them
// would run them -- then the C++ ones, then ggml's, when `upstream` says so.
std::vector<Form> forms(bool upstream) {
    std::vector<Form> f;
    for (const sdasm::Isa level : {sdasm::Isa::avx2, sdasm::Isa::avx512, sdasm::Isa::amx}) {
        if (sdasm::isa_available() >= level) {
            f.push_back({sdasm::isa_name(level), sdasm::Kernels::assembly, level});
        }
    }
    f.push_back({"cxx", sdasm::Kernels::cxx, sdasm::Isa::cxx});
    if (upstream) {
        f.push_back({"upstream", sdasm::Kernels::upstream, sdasm::Isa::cxx});
    }
    return f;
}

void use(const Form & form) {
    sdasm::set_kernels(form.kernels);
    sdasm::set_isa_cap(form.level);
}

// Back to the kernels written by hand at the richest level the processor has.
void use_default() {
    sdasm::set_kernels(sdasm::Kernels::assembly);
    sdasm::set_isa_cap(sdasm::Isa::amx);
}

// What a sum of k products in single precision can be asked to be within, of
// the sum of their magnitudes: about k units of 2^-24 at the very worst, and
// the products of a sum here number in the thousands.
constexpr double kSingle = 1e-5;

// The same for the products on AMX's tiles (x86_64/amx.S), which multiply
// numbers of bfloat16: x = x1 + x2 and w = w1 + w2, each part of 8 bits, and
// x w taken as x1 w1 + x1 w2 + x2 w1. Each part is rounded to the nearest, and
// x1 + x2 is within 2^-16 of x, as w1 + w2 is of w (exactly w for FP16), and
// x2 w2, left out, within 2^-16 of x w: a product within 2^-15, summed in
// single precision. ggml's form rounds the image to FP16 when the weights are
// FP16 -- 2^-11, sixteen times as far.
constexpr double kAmxProducts = 0x1p-15 + kSingle;

// Computes the graph in each form of the kernels, and measures each output
// against exact(i), which gives the exact value and the sum of magnitudes.
// Each form but ggml's is to be within kSingle, and the kernels for AMX within
// amx_bound: kSingle where they compute in single precision, kAmxProducts where
// the products go to the tiles.
template <class Exact>
bool run(const char * name, ggml_context * ctx, ggml_tensor * out, int threads, Exact exact,
         double amx_bound = kSingle) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const std::int64_t n = ggml_nelements(out);
    const std::vector<Form> all = forms(true);
    std::vector<Result> results(all.size());
    for (std::size_t f = 0; f < all.size(); ++f) {
        use(all[f]);
        std::memset(out->data, 0xff, ggml_nbytes(out));
        if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) {
            std::printf("FAIL %s: the graph was not computed\n", name);
            use_default();
            return false;
        }
        results[f].values.assign(static_cast<const float *>(out->data), static_cast<const float *>(out->data) + n);
    }
    use_default();
    for (std::int64_t i = 0; i < n; ++i) {
        double magnitude = 0;
        const double value = exact(i, magnitude);
        const double scale = magnitude > 0 ? magnitude : 1;
        for (auto & r : results) {
            const double e = std::fabs(static_cast<double>(r.values[static_cast<std::size_t>(i)]) - value) / scale;
            r.worst = std::isfinite(e) ? std::fmax(r.worst, e) : HUGE_VAL;
        }
    }
    bool ok = true;
    std::string errors;
    for (std::size_t f = 0; f < all.size(); ++f) {
        if (all[f].kernels != sdasm::Kernels::upstream) {
            ok = ok && results[f].worst <= (all[f].level == sdasm::Isa::amx ? amx_bound : kSingle);
        }
        char text[64];
        std::snprintf(text, sizeof text, "%s%s %.3g", f == 0 ? "" : ", ", all[f].name, results[f].worst);
        errors += text;
    }
    std::printf("%s %s: error relative to the magnitudes: %s\n", ok ? "ok  " : "FAIL", name, errors.c_str());
    return ok;
}

// The quantized forms (quant.h, Tuning::bits): each within kQuantized of the
// exact result, of the magnitudes summed, and the kernels written by hand
// within kSingle of the C++ ones, which sum the same integers. Inputs of 8
// bits in [-1, 1] are within 1/255 of themselves, weights of 8 bits in
// [-0.25, 0.25] within 1/1016: a product within about 1% of the magnitudes in
// the worst case, the sum of a few hundred of them, with errors of either
// sign, nearer 0.1%.
constexpr double kQuantized = 1e-2;

// Attention's forms differ in the last bits of their exponentials, as in single
// precision, and an exponential that lands next to the boundary between two
// steps of 8 bits rounds to one in one form and to the other in another: a
// step of a group of 32 keys, of the sum of thousands.
constexpr double kQuantizedApart = 1e-3;

template <class Exact>
bool run_quantized(const char * name, ggml_context * ctx, ggml_tensor * out, int threads, Exact exact,
                   double apart_bound = kSingle) {
    const sdasm::Tuning saved = sdasm::tuning();
    sdasm::Tuning t = saved;
    t.bits = 8;
    sdasm::set_tuning(t);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const std::int64_t n = ggml_nelements(out);
    const std::vector<Form> all = forms(false);
    std::vector<Result> results(all.size());
    bool computed = true;
    for (std::size_t f = 0; f < all.size() && computed; ++f) {
        use(all[f]);
        std::memset(out->data, 0xff, ggml_nbytes(out));
        computed = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
        results[f].values.assign(static_cast<const float *>(out->data), static_cast<const float *>(out->data) + n);
    }
    use_default();
    sdasm::set_tuning(saved);
    if (!computed) {
        std::printf("FAIL %s: the graph was not computed\n", name);
        return false;
    }
    const Result & cxx = results.back();
    std::vector<double> apart(all.size(), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        double magnitude = 0;
        const double value = exact(i, magnitude);
        const double scale = magnitude > 0 ? magnitude : 1;
        const auto at_i = static_cast<std::size_t>(i);
        for (std::size_t f = 0; f < all.size(); ++f) {
            const double v = static_cast<double>(results[f].values[at_i]);
            const double e = std::fabs(v - value) / scale;
            const double d = std::fabs(v - static_cast<double>(cxx.values[at_i])) / scale;
            results[f].worst = std::isfinite(e) ? std::fmax(results[f].worst, e) : HUGE_VAL;
            apart[f] = std::isfinite(d) ? std::fmax(apart[f], d) : HUGE_VAL;
        }
    }
    bool ok = true;
    std::string errors;
    for (std::size_t f = 0; f < all.size(); ++f) {
        ok = ok && results[f].worst <= kQuantized && apart[f] <= apart_bound;
        char text[96];
        std::snprintf(text, sizeof text, "%s%s %.3g (%.2g from cxx)", f == 0 ? "" : ", ", all[f].name, results[f].worst, apart[f]);
        errors += text;
    }
    std::printf("%s %s, quantized: error relative to the magnitudes: %s\n", ok ? "ok  " : "FAIL", name, errors.c_str());
    return ok;
}

struct Conv_case {
    int w, h, ic, oc, kw, kh, s0, s1, p0, p1, d0, d1, n;
    ggml_type type;
};

bool conv2d(const Conv_case & c, int threads, bool quantized = false) {
    const std::size_t bytes = std::size_t{512} << 20;
    ggml_init_params params{bytes, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, c.type, c.kw, c.kh, c.ic, c.oc);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.w, c.h, c.ic, c.n);
    fill(w, 0.25f);
    fill(x, 1.0f);
    ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1);
    char name[160];
    std::snprintf(name, sizeof name, "conv2d %s %dx%dx%d -> %d, kernel %dx%d, stride %d,%d, padding %d,%d, dilation %d,%d, %d images",
                  ggml_type_name(c.type), c.w, c.h, c.ic, c.oc, c.kw, c.kh, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1, c.n);
    const std::int64_t ow = y->ne[0];
    const std::int64_t oh = y->ne[1];
    const auto exact = [&](std::int64_t i, double & magnitude) {
        const std::int64_t ox = i % ow;
        const std::int64_t oy = (i / ow) % oh;
        const std::int64_t oc = (i / (ow * oh)) % c.oc;
        const std::int64_t n = i / (ow * oh * c.oc);
        double sum = 0;
        magnitude = 0;
        for (std::int64_t ic = 0; ic < c.ic; ++ic) {
            for (std::int64_t ky = 0; ky < c.kh; ++ky) {
                for (std::int64_t kx = 0; kx < c.kw; ++kx) {
                    const std::int64_t iy = oy * c.s1 + ky * c.d1 - c.p1;
                    const std::int64_t ix = ox * c.s0 + kx * c.d0 - c.p0;
                    if (iy < 0 || iy >= c.h || ix < 0 || ix >= c.w) {
                        continue;
                    }
                    const double p = at(w, kx, ky, ic, oc) * at(x, ix, iy, ic, n);
                    sum += p;
                    magnitude += std::fabs(p);
                }
            }
        }
        return sum;
    };
    const bool ok = quantized ? run_quantized(name, ctx, y, threads, exact) : run(name, ctx, y, threads, exact, kAmxProducts);
    ggml_free(ctx);
    return ok;
}

struct Mul_mat_case {
    int k, n, m, batch;
    ggml_type type;
};

bool mul_mat(const Mul_mat_case & c, int threads, bool quantized = false) {
    const std::size_t bytes = std::size_t{512} << 20;
    ggml_init_params params{bytes, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, c.type, c.k, c.n);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, c.m, c.batch);
    fill(w, 0.25f);
    fill(x, 1.0f);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    char name[128];
    std::snprintf(name, sizeof name, "mul_mat %s %d -> %d, %d rows x %d", ggml_type_name(c.type), c.k, c.n, c.m, c.batch);
    const auto exact = [&](std::int64_t i, double & magnitude) {
        const std::int64_t j = i % c.n;
        const std::int64_t r = (i / c.n) % c.m;
        const std::int64_t b = i / (std::int64_t{c.n} * c.m);
        double sum = 0;
        magnitude = 0;
        for (std::int64_t k = 0; k < c.k; ++k) {
            const double p = at(w, k, j, 0, 0) * at(x, k, r, b, 0);
            sum += p;
            magnitude += std::fabs(p);
        }
        return sum;
    };
    const bool ok = quantized ? run_quantized(name, ctx, y, threads, exact) : run(name, ctx, y, threads, exact, kAmxProducts);
    ggml_free(ctx);
    return ok;
}

struct Attention_case {
    int d, lq, lk, h, hk, b;
    ggml_type kv_type;
};

// The output of attention, [d * h, lq, b] as stable-diffusion.cpp has it, by
// the graph it builds without flash attention: the scores, their scale, their
// softmax and the product with the values.
ggml_tensor * attention_graph(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, const Attention_case & c,
                              float scale) {
    ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, c.d, c.h, c.lq, c.b), 0, 2, 1, 3));
    qh = ggml_reshape_3d(ctx, qh, c.d, c.lq, c.h * c.b);
    ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, c.d, c.hk, c.lk, c.b), 0, 2, 1, 3));
    kh = ggml_reshape_3d(ctx, kh, c.d, c.lk, c.hk * c.b);
    ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, c.d, c.hk, c.lk, c.b), 1, 2, 0, 3));
    vh = ggml_reshape_3d(ctx, vh, c.lk, c.d, c.hk * c.b);
    ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
    kq = ggml_scale_inplace(ctx, kq, scale);
    kq = ggml_soft_max_inplace(ctx, kq);
    ggml_tensor * kqv = ggml_mul_mat(ctx, vh, kq);
    kqv = ggml_reshape_4d(ctx, kqv, c.d, c.lq, c.h, c.b);
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, kqv, c.d * c.h, c.lq, c.b);
}

bool attention(const Attention_case & c, int threads, bool quantized = false) {
    const std::size_t bytes = std::size_t{1} << 30;
    ggml_init_params params{bytes, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.d * c.h, c.lq, c.b);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, c.kv_type, c.d * c.hk, c.lk, c.b);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, c.kv_type, c.d * c.hk, c.lk, c.b);
    fill(q, 1.0f);
    fill(k, 1.0f);
    fill(v, 1.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.d));
    ggml_tensor * qv = ggml_permute(ctx, ggml_reshape_4d(ctx, q, c.d, c.h, c.lq, c.b), 0, 2, 1, 3);
    ggml_tensor * kv = ggml_permute(ctx, ggml_reshape_4d(ctx, k, c.d, c.hk, c.lk, c.b), 0, 2, 1, 3);
    ggml_tensor * vv = ggml_permute(ctx, ggml_reshape_4d(ctx, v, c.d, c.hk, c.lk, c.b), 0, 2, 1, 3);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, qv, kv, vv, nullptr, scale, 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    // The exact output: the weights of each query, and the magnitude of the sum
    // of the values by them.
    const std::int64_t group = c.h / c.hk;
    std::vector<double> exact(static_cast<std::size_t>(ggml_nelements(out)));
    std::vector<double> magnitudes(exact.size());
    std::vector<double> w(static_cast<std::size_t>(c.lk));
    for (std::int64_t b = 0; b < c.b; ++b) {
        for (std::int64_t h = 0; h < c.h; ++h) {
            const std::int64_t hk = h / group;
            for (std::int64_t i = 0; i < c.lq; ++i) {
                double max = -HUGE_VAL;
                for (std::int64_t j = 0; j < c.lk; ++j) {
                    double sum = 0;
                    for (std::int64_t e = 0; e < c.d; ++e) {
                        sum += at(q, h * c.d + e, i, b, 0) * at(k, hk * c.d + e, j, b, 0);
                    }
                    w[static_cast<std::size_t>(j)] = sum * static_cast<double>(scale);
                    max = std::fmax(max, w[static_cast<std::size_t>(j)]);
                }
                double total = 0;
                for (auto & x : w) {
                    x = std::exp(x - max);
                    total += x;
                }
                for (std::int64_t e = 0; e < c.d; ++e) {
                    double sum = 0;
                    double magnitude = 0;
                    for (std::int64_t j = 0; j < c.lk; ++j) {
                        const double p = w[static_cast<std::size_t>(j)] / total * at(v, hk * c.d + e, j, b, 0);
                        sum += p;
                        magnitude += std::fabs(p);
                    }
                    const auto index = static_cast<std::size_t>(e + c.d * (h + c.h * (i + c.lq * b)));
                    exact[index] = sum;
                    magnitudes[index] = magnitude;
                }
            }
        }
    }
    char name[160];
    std::snprintf(name, sizeof name, "attention q %s, k and v %s: %d dimensions, %d queries, %d keys, %d heads (%d of keys), %d batches",
                  ggml_type_name(GGML_TYPE_F32), ggml_type_name(c.kv_type), c.d, c.lq, c.lk, c.h, c.hk, c.b);
    const auto exact_at = [&](std::int64_t i, double & magnitude) {
        magnitude = magnitudes[static_cast<std::size_t>(i)];
        return exact[static_cast<std::size_t>(i)];
    };
    if (quantized) {
        const bool ok = run_quantized(name, ctx, out, threads, exact_at, kQuantizedApart);
        ggml_free(ctx);
        return ok;
    }
    bool ok = run(name, ctx, out, threads, exact_at);

    // The graph stable-diffusion.cpp builds without flash attention, in ggml's
    // own kernels, which is what the operation stands in for.
    if (c.hk == c.h && c.kv_type == GGML_TYPE_F32) {
        ggml_tensor * reference = attention_graph(ctx, q, k, v, c, scale);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, reference);
        sdasm::set_kernels(sdasm::Kernels::upstream);
        const bool computed = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
        use_default();
        double worst = 0;
        for (std::int64_t e = 0; computed && e < ggml_nelements(reference); ++e) {
            const auto i = static_cast<std::size_t>(e);
            const double value = static_cast<double>(static_cast<const float *>(reference->data)[i]);
            worst = std::fmax(worst, std::fabs(value - exact[i]) / magnitudes[i]);
        }
        std::printf("     the graph without flash attention, in ggml's kernels: %.3g\n", worst);
        ok = ok && computed;
    }
    ggml_free(ctx);
    return ok;
}

// A chain of operations after an operation the kernels take, computed in each
// form of the kernels with the chain fused into the operation and apart from it:
// the two need only be within `close` of each other, relative to the largest
// output, for the fused operation may sum in an order of its own.
bool fusion(const char * name, int threads, const std::function<ggml_tensor *(ggml_context *)> & build,
            double close = 1e-6) {
    const std::size_t bytes = std::size_t{512} << 20;
    ggml_init_params params{bytes, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * out = build(ctx);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    // The chain changes the output of the head in place, which the graph
    // computes again from the inputs each time.
    const auto compute = [&](const Form & form, int fuse) {
        use(form);
        sdasm::Tuning t = sdasm::tuning();
        t.fuse = fuse;
        sdasm::set_tuning(t);
        ggml_graph_compute_with_ctx(ctx, graph, threads);
        const auto * p = static_cast<const float *>(out->data);
        return std::vector<float>(p, p + ggml_nelements(out));
    };
    bool ok = true;
    double worst = 0;
    for (const Form & form : forms(false)) {
        const std::vector<float> fused = compute(form, 1);
        const std::vector<float> apart = compute(form, 0);
        double largest = 0;
        double difference = 0;
        for (std::size_t i = 0; i < fused.size(); ++i) {
            largest = std::fmax(largest, std::fabs(static_cast<double>(apart[i])));
            difference = std::fmax(difference, std::fabs(static_cast<double>(fused[i]) - static_cast<double>(apart[i])));
        }
        const double relative = largest > 0 ? difference / largest : difference;
        ok = ok && std::isfinite(relative) && relative <= close;
        worst = std::fmax(worst, relative);
    }
    use_default();
    sdasm::Tuning t = sdasm::tuning();
    t.fuse = 1;
    sdasm::set_tuning(t);
    std::printf("%s fusion %s: fused and apart, in each form, within %.3g of the largest output\n", ok ? "ok  " : "FAIL",
                name, worst);
    ggml_free(ctx);
    return ok;
}

// The weights the kernels keep packed between graphs (store.h) are forgotten
// when they change: an operation computed, its weights set anew through ggml's
// backend, and the operation computed again gives what a fresh computation with
// the new weights gives, and not what the old weights gave.
bool store(const char * name, int threads,
           const std::function<ggml_tensor *(ggml_context *, ggml_tensor ** weights)> & build) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    const auto set = [](ggml_tensor * t, std::uint32_t seed) {
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        const auto n = static_cast<std::size_t>(ggml_nelements(t));
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> v(n);
            for (auto & x : v) {
                x = d(g);
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
        else {
            std::vector<ggml_fp16_t> v(n);
            for (auto & x : v) {
                x = ggml_fp32_to_fp16(d(g));
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    };
    // The inputs from seed 100, the weights from `first`, computed, and with
    // `second` given, the weights set from it and computed again.
    const auto run = [&](std::uint32_t first, std::uint32_t second) {
        ggml_init_params params{std::size_t{64} << 20, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * weights = nullptr;
        ggml_tensor * out = build(ctx, &weights);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op == GGML_OP_NONE && t != weights) {
                set(t, 100);
            }
        }
        set(weights, first);
        ggml_backend_graph_compute(backend, graph);
        if (second != 0) {
            set(weights, second);
            ggml_backend_graph_compute(backend, graph);
        }
        std::vector<float> result(static_cast<std::size_t>(ggml_nelements(out)));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return result;
    };
    // In each form, which may keep weights packed in a layout of its own.
    bool ok = true;
    for (const Form & form : forms(false)) {
        use(form);
        const std::vector<float> old_weights = run(1, 0);
        const std::vector<float> changed = run(1, 2);
        const std::vector<float> fresh = run(2, 0);
        // Recomputed with the weights set anew, the output is the fresh one's, and
        // not the old weights' (which it differs from in its first digits).
        double largest = 0;
        double from_fresh = 0;
        double from_old = 0;
        for (std::size_t i = 0; i < changed.size(); ++i) {
            largest = std::fmax(largest, std::fabs(static_cast<double>(fresh[i])));
            from_fresh = std::fmax(from_fresh, std::fabs(static_cast<double>(changed[i]) - static_cast<double>(fresh[i])));
            from_old = std::fmax(from_old, std::fabs(static_cast<double>(changed[i]) - static_cast<double>(old_weights[i])));
        }
        const bool same = from_fresh <= 1e-6 * largest;
        const bool moved = from_old > 1e-3 * largest;
        ok = ok && same && moved;
        std::printf("%s store %s, %s: weights set anew are %s%s\n", same && moved ? "ok  " : "FAIL", name, form.name,
                    moved ? "used" : "NOT USED (the old packing was kept)", same ? "" : ", and differ from a fresh computation");
    }
    use_default();
    ggml_backend_free(backend);
    return ok;
}


// stable-diffusion.cpp's GEGLU after a linear layer: the layer with its bias,
// split into halves x and gate along dimension 0, the gate made contiguous,
// its GELU taken in place, and x multiplied by it. The kernels compute it all
// at once (fusion.h), with GELU in single precision where ggml reads it from a
// table of FP16; the error of each form is measured against the exact
// x GELU(gate), relative to the magnitudes it is computed from: those of the
// products summed into x and the gate, by how much the result moves with each.
bool gated_gelu(int k, int half, int m, int batch, int threads, bool quantized = false) {
    const std::size_t bytes = std::size_t{512} << 20;
    ggml_init_params params{bytes, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, 2 * half);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, batch);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * half);
    fill(w, 0.25f);
    fill(x, 1.0f);
    fill(b, 1.0f);
    ggml_tensor * y = ggml_add_inplace(ctx, ggml_mul_mat(ctx, w, x), b);
    ggml_tensor * xs = ggml_view_4d(ctx, y, half, m, batch, 1, y->nb[1], y->nb[2], y->nb[3], 0);
    ggml_tensor * gate = ggml_view_4d(ctx, y, half, m, batch, 1, y->nb[1], y->nb[2], y->nb[3], static_cast<std::size_t>(half) * y->nb[0]);
    gate = ggml_gelu_inplace(ctx, ggml_cont(ctx, gate));
    ggml_tensor * out = ggml_mul(ctx, xs, gate);

    std::vector<double> exact(static_cast<std::size_t>(ggml_nelements(out)));
    std::vector<double> magnitudes(exact.size());
    const auto gelu = [](double v) { return 0.5 * v * (1.0 + std::tanh(0.79788456080286535587989211986876 * (v + 0.044715 * v * v * v))); };
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(m) * batch; ++r) {
        for (std::int64_t j = 0; j < half; ++j) {
            double sum[2] = {at(b, j, 0, 0, 0), at(b, half + j, 0, 0, 0)};
            double magnitude[2] = {std::fabs(sum[0]), std::fabs(sum[1])};
            for (int side = 0; side < 2; ++side) {
                for (std::int64_t e = 0; e < k; ++e) {
                    const double p = at(w, e, side * half + j, 0, 0) * at(x, e, r % m, r / m, 0);
                    sum[side] += p;
                    magnitude[side] += std::fabs(p);
                }
            }
            // d/dv of GELU(v) lies within [-0.13, 1.13].
            const auto i = static_cast<std::size_t>(r * half + j);
            exact[i] = sum[0] * gelu(sum[1]);
            magnitudes[i] = magnitude[0] * std::fabs(gelu(sum[1])) + std::fabs(sum[0]) * 1.13 * magnitude[1];
        }
    }
    char name[128];
    std::snprintf(name, sizeof name, "gated GELU after linear %d -> 2 x %d, %d rows x %d", k, half, m, batch);
    const auto exact_at = [&](std::int64_t i, double & magnitude) {
        magnitude = magnitudes[static_cast<std::size_t>(i)];
        return exact[static_cast<std::size_t>(i)];
    };
    const bool ok = quantized ? run_quantized(name, ctx, out, threads, exact_at) : run(name, ctx, out, threads, exact_at, kAmxProducts);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s conv2d|mul_mat|attention|gated|fusion|store|quantized [threads]\n", argv[0]);
        return 2;
    }
    const std::string what = argv[1];
    const int threads = argc > 2 ? std::atoi(argv[2]) : static_cast<int>(std::thread::hardware_concurrency());
    std::printf("%d threads\n", threads);
    bool ok = true;
    if (what == "conv2d") {
        const Conv_case cases[] = {
            // The convolutions of the UNet of SD 1.5 at 512 by 512, smaller.
            {64, 64, 64, 48, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {16, 16, 96, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {8, 8, 96, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {32, 32, 64, 40, 1, 1, 1, 1, 0, 0, 1, 1, 2, GGML_TYPE_F16},
            {33, 33, 24, 30, 3, 3, 2, 2, 1, 1, 1, 1, 1, GGML_TYPE_F16},
            // Edges everywhere: sizes of no multiple of a panel, FP32 weights.
            {17, 13, 5, 7, 3, 3, 1, 1, 1, 1, 1, 1, 3, GGML_TYPE_F32},
            {40, 9, 7, 13, 5, 3, 1, 2, 2, 0, 2, 1, 1, GGML_TYPE_F32},
            {100, 3, 20, 25, 3, 3, 1, 1, 0, 1, 1, 1, 1, GGML_TYPE_F16},
            {512, 4, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, GGML_TYPE_F16},
            // Large enough for Winograd's F(2x2, 3x3): odd sizes, rows of tiles
            // that do not fill panels, no padding, FP32 weights.
            {65, 33, 24, 20, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {64, 20, 13, 11, 3, 3, 1, 1, 0, 0, 1, 1, 1, GGML_TYPE_F16},
            {96, 16, 32, 17, 3, 3, 1, 1, 1, 1, 1, 1, 1, GGML_TYPE_F32},
            {128, 128, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, GGML_TYPE_F16},
        };
        for (const auto & c : cases) {
            ok = conv2d(c, threads) && ok;
        }
    }
    else if (what == "mul_mat") {
        const Mul_mat_case cases[] = {
            {320, 320, 512, 2, GGML_TYPE_F16},
            {768, 1280, 77, 2, GGML_TYPE_F16},
            {1280, 320, 64, 1, GGML_TYPE_F16},
            {77, 100, 23, 1, GGML_TYPE_F32},
            {300, 17, 31, 3, GGML_TYPE_F16},
            {40, 2560, 96, 1, GGML_TYPE_F32},
        };
        for (const auto & c : cases) {
            ok = mul_mat(c, threads) && ok;
        }
    }
    else if (what == "quantized") {
        // Winograd's convolutions of small and large images, and the direct
        // ones, of the cases above.
        const Conv_case conv_cases[] = {
            {64, 64, 64, 48, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {16, 16, 96, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {8, 8, 96, 64, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {32, 32, 64, 40, 1, 1, 1, 1, 0, 0, 1, 1, 2, GGML_TYPE_F16},
            {33, 33, 24, 30, 3, 3, 2, 2, 1, 1, 1, 1, 1, GGML_TYPE_F16},
            {17, 13, 5, 7, 3, 3, 1, 1, 1, 1, 1, 1, 3, GGML_TYPE_F32},
            {65, 33, 24, 20, 3, 3, 1, 1, 1, 1, 1, 1, 2, GGML_TYPE_F16},
            {64, 20, 13, 11, 3, 3, 1, 1, 0, 0, 1, 1, 1, GGML_TYPE_F16},
            {96, 16, 32, 17, 3, 3, 1, 1, 1, 1, 1, 1, 1, GGML_TYPE_F32},
        };
        for (const auto & c : conv_cases) {
            ok = conv2d(c, threads, true) && ok;
        }
        // The linear layers of 256 rows and more, which are quantized, sizes of
        // no multiple of a panel or a group among them; and of fewer, which
        // are not.
        const Mul_mat_case cases[] = {
            {320, 320, 512, 2, GGML_TYPE_F16},
            {768, 1280, 300, 1, GGML_TYPE_F16},
            {1280, 320, 256, 1, GGML_TYPE_F16},
            {77, 100, 300, 1, GGML_TYPE_F32},
            {300, 17, 131, 3, GGML_TYPE_F16},
            {40, 2560, 296, 1, GGML_TYPE_F32},
            {768, 320, 77, 2, GGML_TYPE_F16},
        };
        for (const auto & c : cases) {
            ok = mul_mat(c, threads, true) && ok;
        }
        // The gated GELU after the product, fused; and of halves of no multiple
        // of a panel, apart.
        ok = gated_gelu(320, 1280, 200, 2, threads, true) && ok;
        ok = gated_gelu(96, 2000, 300, 1, threads, true) && ok;
        ok = gated_gelu(40, 100, 300, 1, threads, true) && ok;
        // Self-attention, over enough keys to be quantized, and cross-attention,
        // over too few, in single precision.
        const Attention_case attention_cases[] = {
            {40, 300, 4096, 2, 2, 2, GGML_TYPE_F32},
            {80, 100, 1024, 2, 2, 1, GGML_TYPE_F16},
            {160, 64, 256, 2, 2, 2, GGML_TYPE_F32},
            {40, 200, 77, 2, 2, 2, GGML_TYPE_F32},
            {13, 37, 290, 4, 2, 3, GGML_TYPE_F32},
        };
        for (const auto & c : attention_cases) {
            ok = attention(c, threads, true) && ok;
        }
    }
    else if (what == "attention") {
        const Attention_case cases[] = {
            // SD 1.5's self-attention at 64 by 64, fewer queries and heads.
            {40, 300, 4096, 2, 2, 2, GGML_TYPE_F32},
            // Its cross-attention: 77 tokens.
            {40, 200, 77, 2, 2, 2, GGML_TYPE_F32},
            {80, 100, 1024, 2, 2, 1, GGML_TYPE_F32},
            {160, 64, 256, 2, 2, 2, GGML_TYPE_F32},
            // The VAE's: one head of 512.
            {512, 40, 300, 1, 1, 1, GGML_TYPE_F32},
            // Edges: dimensions and keys of no multiple of a panel, heads sharing
            // keys, FP16 keys and values.
            {13, 37, 29, 4, 2, 3, GGML_TYPE_F32},
            {64, 50, 100, 2, 1, 1, GGML_TYPE_F16},
        };
        for (const auto & c : cases) {
            ok = attention(c, threads) && ok;
        }
    }
    else if (what == "gated") {
        // SD 1.5's at 64 by 64 and 16 by 16, fewer rows; and sizes of no
        // multiple of a panel.
        ok = gated_gelu(320, 1280, 200, 2, threads) && ok;
        ok = gated_gelu(1280, 5120, 64, 2, threads) && ok;
        ok = gated_gelu(40, 100, 37, 1, threads) && ok;
        ok = gated_gelu(96, 2000, 1500, 1, threads) && ok;
    }
    else if (what == "fusion") {
        // A convolution, its bias in place, and the input added in place after.
        ok = fusion("conv2d, bias", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 24, 40);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 40, 36, 24, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 40);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, 1, 1, 1, 1);
                 return ggml_add_inplace(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 40, 1));
             }) && ok;
        ok = fusion("conv2d 1x1, bias, scale per image", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 16, 20);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 9, 16, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 20);
                 ggml_tensor * e = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 20, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(e, 1.0f);
                 ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, 0, 0, 1, 1);
                 y = ggml_add_inplace(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 20, 1));
                 return ggml_mul_inplace(ctx, y, e);
             }) && ok;
        ok = fusion("conv2d by Winograd, bias, residual", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 16, 24);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 66, 40, 16, 1);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
                 ggml_tensor * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 66, 40, 24, 1);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(r, 1.0f);
                 ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, 1, 1, 1, 1);
                 y = ggml_add_inplace(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 24, 1));
                 return ggml_add_inplace(ctx, y, r);
             }) && ok;
        // Residual connections: the sum of the operation's output and another
        // tensor written to a tensor of its own, the operation's output on the
        // left of the addition or on the right.
        ok = fusion("conv2d, bias, residual not in place", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 24, 40);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 40, 36, 24, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 40);
                 ggml_tensor * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 40, 36, 40, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(r, 1.0f);
                 ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, 1, 1, 1, 1);
                 y = ggml_add_inplace(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 40, 1));
                 return ggml_add(ctx, y, r);
             }) && ok;
        ok = fusion("conv2d of small images by Winograd, residual on the left", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 48);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 64, 2);
                 ggml_tensor * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 48, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(r, 1.0f);
                 return ggml_add(ctx, r, ggml_conv_2d_direct(ctx, w, x, 1, 1, 1, 1, 1, 1));
             }) && ok;
        ok = fusion("linear, bias, residual not in place", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 96, 70);
                 ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 96, 50, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 70);
                 ggml_tensor * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 70, 50, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(r, 1.0f);
                 return ggml_add(ctx, ggml_add_inplace(ctx, ggml_mul_mat(ctx, w, x), b), r);
             }) && ok;
        // Residual connections whose sum is written in the memory of the
        // residual, as the allocator often has it: begun from the residual. The
        // residual is a copy, made anew by each computation of the graph.
        ok = fusion("conv2d, bias, residual in its own memory", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 96, 40);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 40, 36, 96, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 40);
                 ggml_tensor * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 40, 36, 40, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(r, 1.0f);
                 ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, 0, 0, 1, 1);
                 y = ggml_add_inplace(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 40, 1));
                 return ggml_add_inplace(ctx, ggml_dup(ctx, r), y);
             }, 1e-6) && ok;
        ok = fusion("conv2d by Winograd, residual in its own memory", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 16, 24);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 66, 40, 16, 1);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
                 ggml_tensor * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 66, 40, 24, 1);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(r, 1.0f);
                 ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, 1, 1, 1, 1);
                 y = ggml_add_inplace(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 24, 1));
                 return ggml_add_inplace(ctx, ggml_dup(ctx, r), y);
             }, 1e-6) && ok;
        ok = fusion("conv2d of small images by Winograd, residual in its own memory", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 48);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 15, 9, 64, 2);
                 ggml_tensor * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 15, 9, 48, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(r, 1.0f);
                 return ggml_add_inplace(ctx, ggml_dup(ctx, r), ggml_conv_2d_direct(ctx, w, x, 1, 1, 1, 1, 1, 1));
             }, 1e-6) && ok;
        ok = fusion("linear, bias, residual in its own memory", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 300, 70);
                 ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 300, 1100, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 70);
                 ggml_tensor * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 70, 1100, 2);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 fill(r, 1.0f);
                 return ggml_add_inplace(ctx, ggml_dup(ctx, r), ggml_add_inplace(ctx, ggml_mul_mat(ctx, w, x), b));
             }, 1e-6) && ok;
        ok = fusion("linear, bias", threads, [](ggml_context * ctx) {
                 ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 96, 70);
                 ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 96, 50, 2);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 70);
                 fill(w, 0.25f);
                 fill(x, 1.0f);
                 fill(b, 1.0f);
                 return ggml_add_inplace(ctx, ggml_mul_mat(ctx, w, x), b);
             }) && ok;
        ok = fusion("group norm, scale, shift, SiLU", threads, [](ggml_context * ctx) {
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 36, 20, 64, 2);
                 ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 64, 1);
                 ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 64, 1);
                 fill(x, 3.0f);
                 fill(g, 1.0f);
                 fill(b, 1.0f);
                 ggml_tensor * y = ggml_group_norm(ctx, x, 32, 1e-6f);
                 y = ggml_mul_inplace(ctx, y, g);
                 y = ggml_add_inplace(ctx, y, b);
                 return ggml_silu_inplace(ctx, y);
             }) && ok;
        ok = fusion("group norm of rows of no multiple of eight, scale, shift", threads, [](ggml_context * ctx) {
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 13, 7, 30, 1);
                 ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 30, 1);
                 ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 30, 1);
                 fill(x, 3.0f);
                 fill(g, 1.0f);
                 fill(b, 1.0f);
                 ggml_tensor * y = ggml_group_norm(ctx, x, 10, 1e-5f);
                 y = ggml_mul_inplace(ctx, y, g);
                 y = ggml_add_inplace(ctx, y, b);
                 return ggml_silu_inplace(ctx, y);
             }) && ok;
        ok = fusion("layer norm, scale, shift", threads, [](ggml_context * ctx) {
                 ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 320, 77, 2);
                 ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 320);
                 ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 320);
                 fill(x, 3.0f);
                 fill(g, 1.0f);
                 fill(b, 1.0f);
                 ggml_tensor * y = ggml_norm(ctx, x, 1e-5f);
                 y = ggml_mul_inplace(ctx, y, g);
                 return ggml_add_inplace(ctx, y, b);
             }) && ok;
    }
    else if (what == "store") {
        // The store keeps nothing unless the environment gives it a budget.
        setenv("SDASM_STORE_MB", "1024", 0);
        ok = store("conv2d", threads, [](ggml_context * ctx, ggml_tensor ** weights) {
                 *weights = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 16, 24);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 20, 12, 16, 2);
                 return ggml_conv_2d_direct(ctx, *weights, x, 1, 1, 1, 1, 1, 1);
             }) && ok;
        ok = store("conv2d by Winograd", threads, [](ggml_context * ctx, ggml_tensor ** weights) {
                 *weights = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 16, 24);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 32, 16, 1);
                 return ggml_conv_2d_direct(ctx, *weights, x, 1, 1, 1, 1, 1, 1);
             }) && ok;
        ok = store("conv2d of shared columns", threads, [](ggml_context * ctx, ggml_tensor ** weights) {
                 *weights = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 40, 64);
                 ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 40, 2);
                 return ggml_conv_2d_direct(ctx, *weights, x, 1, 1, 1, 1, 1, 1);
             }) && ok;
        ok = store("linear", threads, [](ggml_context * ctx, ggml_tensor ** weights) {
                 *weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 80);
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 2000);
                 return ggml_mul_mat(ctx, *weights, x);
             }) && ok;
        ok = store("linear of shared rows", threads, [](ggml_context * ctx, ggml_tensor ** weights) {
                 *weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 80);
                 ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 50);
                 return ggml_mul_mat(ctx, *weights, x);
             }) && ok;
    }
    else {
        std::fprintf(stderr, "unknown operation '%s'\n", what.c_str());
        return 2;
    }
    return ok ? 0 : 1;
}
