// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// Every kernel in each of its forms: the one in plain C++, which the compiler
// vectorizes for whatever the build targets, and those written by hand
// (x86_64/), one for each extension of the instruction set they are written
// for, which cpu.cc chooses among by what the processor has.

#ifndef SDASM_KERNELS_H
#define SDASM_KERNELS_H

#include <cstddef>
#include <cstdint>

struct ggml_cgraph;
struct ggml_tensor;

namespace sdasm {

// The forms of the kernels a build can run, as SDASM_KERNELS names them in the
// environment: assembly, the default, takes the kernels written by hand where
// there are any; cxx takes the C++ ones throughout; upstream takes none of this
// project's, and leaves every operation to ggml.
enum class Kernels { assembly, cxx, upstream };

Kernels kernels();
const char * kernels_name(Kernels k);

// What the processor running the program offers of the extensions the kernels
// written by hand are written for, read once from cpuid; all false on a
// processor other than x86-64, or in a build without those kernels.
struct Cpu {
    bool intel = false;        // the vendor, for the choices the kernels make by the cores
    bool avx2 = false;         // with FMA and F16C, as x86-64-v3 has them
    bool avx512 = false;       // AVX-512 F, VL, BW and DQ, as AVX10.1 has them
    bool avx512_bf16 = false;
    bool avx512_fp16 = false;
    bool avx_vnni = false;
    bool avx512_vnni = false;
    bool amx_bf16 = false;     // AMX-TILE and AMX-BF16, with AVX512-BF16 and the system's leave to use the tiles
    bool amx_fp16 = false;
};

const Cpu & cpu();

// The levels of the kernels written by hand, each with all the one before it
// has: avx2, the least this project builds for; avx512; and amx, whose matrix
// products are computed on AMX's tiles (x86_64/amx.S) and all else by the
// kernels for AVX-512. cxx is none of them: the C++ kernels throughout.
enum class Isa { cxx, avx2, avx512, amx };

// The level in effect: the richest the processor has (isa_available()), held to
// none when the form of the kernels in effect is not assembly, and to the cap,
// which SDASM_ISA=avx2|avx512|amx in the environment sets, and set_isa_cap()
// for the tests and the tools, which measure each level on the same processor.
Isa isa();
Isa isa_available();
void set_isa_cap(Isa cap);
const char * isa_name(Isa level);
// The level of a name isa_name() gives, if it is one.
bool isa_named(const char * name, Isa & level);

// The name of the level in effect, for the record of a measurement.
const char * kernels_isa();

// Whether the kernels in effect are those written by hand for AVX2, FMA and
// F16C or a level above: the form is assembly and the processor has the three.
bool avx2_kernels();
// Whether they are those for AVX-512 F, VL, BW and DQ, or AMX.
bool avx512_kernels();
// Whether the matrix products go to AMX's tiles.
bool amx_kernels();

// A thread's time in this project's kernels for a node of a graph, for the
// profile (profile.cc).
void profile_busy(const ggml_cgraph * graph, const ggml_tensor * node, int fused, int threads, double seconds);
// For the tests, which compute the same operation in each form in turn: the
// form every operation after takes, whatever the environment said.
void set_kernels(Kernels k);

// The sizes of the blocks the operations work in, and the order of their
// loops: the values the operations are measured best with on the i7-1360P, which
// the environment may change for an experiment (SDASM_TUNE=name=value,...), as
// the developer tools do between operations (set_tuning()).
struct Tuning {
    int conv_kc = 256;             // the stretch of K of a block of the convolution
    int conv_mc = 480;             // the most output channels of a block
    int conv_nc = 128;             // the positions of a block, at the most
    int conv_rows_outer = 0;       // panels of weights outermost: 1, of columns: 0
    int conv_shared_columns = 1024;// the most positions of all images whose columns are shared
    int conv_columns_bytes = 4 << 20; // the most bytes of shared columns packed at a time
    int mul_mat_kc = 256;
    int mul_mat_mc = 96;           // the rows of a block of the linear layers
    int mul_mat_shared_rows = 1024;// the most rows of src1 that are shared
    int amx_kc = 256;              // on AMX's tiles: the stretch of K of a block, a multiple of 32
    int amx_mc = 128;              // and the rows of a block, a multiple of 32
    int amx_attention = 1;         // attention on the tiles, in single precision (amx_single.S): 1, not: 0
    int amx_queries = 64;          // the queries of its blocks, a multiple of 32
    int amx_fp16 = 1;              // AMX-FP16's products of FP16 weights, where the processor has it (gemm.h): 1, not: 0
    int attention_queries = 32;    // the queries of a block of attention
    int attention_keys = 256;      // the keys of a stretch of its second product
    int attention_sum_keys = 32;   // the most keys an output is summed over in the registers
    int attention_sum_dims = 40;   // the most dimensions a score is summed over in the registers
    int winograd = 2;              // 3x3 convolutions by F(2x2, 3x3) and F(4x4, 3x3) for large images: 2; F(2x2) alone: 1; directly: 0
    int winograd_positions = 1024; // the least outputs of an image it is used for
    int winograd4_positions = 256; // with winograd = 2, the least outputs of an image F(4x4, 3x3) is used for
    int winograd_kc = 256;
    int winograd_mc = 512;
    int winograd_tiles = 64;
    int winograd_small_tiles = 256;// the most tiles of all images computed as small images
    int winograd_small_mc = 48;    // the output channels of a block of small images
    int winograd_small_kc = 128;
    int prefetch = 1;              // the weights of the next stretch fetched ahead: 1, not: 0
    int wide = -1;                 // AVX-512's tile of 32 columns (gemm.h): 1, of 16: 0; -1: on Intel's processors
    int bf16 = 0;                  // at avx512, the products of FP16 weights by AVX512-BF16 (gemm.h): 1, for measurement
    int fuse = 1;                  // the chains after operations with them: 1, apart: 0
    int fuse_residual = 1;         // and the residual connections that end them: 1, apart: 0
    int guided = 1;                // blocks shrinking toward the end of an operation: 1, all alike: 0
    int store_forget = 1;          // the store forgets what changes: 1; 0 only for its canary test
    int bits = 0;                  // the products': 0 single precision; 2 to 8 integers, the weights of that many bits (quant.h)
    int input_bits = 8;            // and the inputs', 2 to 8, unsigned with a zero point
    int group = 64;                // the numbers of k sharing a scale, a multiple of 4
    int weight_group = 1;          // the weights' scale for each group: 1; for a whole row of k: 0
    int quantize = 7;              // with bits, the operations quantized: 1 the linear layers, 2 the convolutions, 4 attention
};

const Tuning & tuning();
void set_tuning(const Tuning & t);
// Parses name=value,... into t, and says which name it did not know, if any.
bool parse_tuning(const char * text, Tuning & t);

void gemm_6x16_cxx(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                   std::int64_t accumulate);
void scores_6x16_cxx(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc, float * max,
                     float scale, std::int64_t accumulate);
void exponentials_cxx(float * s, std::int64_t n, const float * max, double * total);

// A job of the transform of Winograd's inputs written by hand
// (x86_64/winograd_avx2.S, which reads its fields at their offsets): sixteen
// tiles of a row of tiles, channel after channel.
struct Winograd_inputs {
    float * dst;               // the panel of s = 0; that of s at dst + s * stride bytes
    std::int64_t stride;
    const char * a;            // row A of the tiles at their first input, the first channel's
    std::int64_t a_step;       // bytes to the next channel's: a plane, or 0 for a row of zeros
    const char * b;            // row B, likewise
    std::int64_t b_step;
    std::int64_t channels;
    const std::int32_t * masks;  // AVX2: the lanes of the eight loads inside the image, or none: all are
    std::int64_t subtract;     // A minus B, or A plus B
    std::uint16_t lanes[4];    // AVX-512: the lanes of the loads at 0, 16, 2 and 18 inside the image
};

// A job of the transform of F(4x4, 3x3)'s inputs written by hand
// (x86_64/winograd4_*.S, which read its fields at their offsets): the six
// elements xi of a row r of B^T of sixteen tiles (AVX-512) or eight (AVX2) of a
// row of tiles, channel after channel.
struct Winograd4_inputs {
    float * dst;               // the panel of s = 0; that of s at dst + s * stride bytes
    std::int64_t stride;
    std::int64_t step;         // bytes to the next channel's rows: a plane
    std::int64_t channels;
    std::int64_t rows;         // the rows of the tiles combined, at most six
    const char * row[6];       // each one's number at the tiles' first input, of the first channel
    float coefficient[6];      // B^T[r][i] of each
    std::uint16_t lanes[5];    // AVX-512: the lanes of the loads at 0, 16, 32, 48 and 64 inside the image
    const std::int32_t * masks;  // AVX2: [5][8], the lanes of the loads at 0, 8, 16, 24 and 32; or none: all are
};

static_assert(offsetof(Winograd4_inputs, rows) == 32 && offsetof(Winograd4_inputs, row) == 40 &&
              offsetof(Winograd4_inputs, coefficient) == 88 && offsetof(Winograd4_inputs, lanes) == 112 &&
              offsetof(Winograd4_inputs, masks) == 128);

}  // namespace sdasm

// The kernels written by hand, for x86-64 with AVX2, FMA and F16C (x86_64/).
extern "C" {

void sdasm_gemm_6x16_avx2(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                          std::int64_t accumulate);
void sdasm_scores_6x16_avx2(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                            float * max, float scale, std::int64_t accumulate);
void sdasm_exponentials_avx2(float * s, std::int64_t n, const float * max, double * total);
void sdasm_pack16_f32_avx2(float * dst, const float * src, std::int64_t stride, std::int64_t kc);
void sdasm_pack16_f16_avx2(float * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t kc);
void sdasm_pack6_f32_avx2(float * dst, const float * src, std::int64_t stride, std::int64_t kc);
void sdasm_pack6_f16_avx2(float * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t kc);
void sdasm_winograd_inputs_avx2(const sdasm::Winograd_inputs * job);
void sdasm_winograd_outputs_avx2(const float * m, std::int64_t stride, float * row0, float * row1, std::int64_t groups,
                                 std::int64_t add);
void sdasm_winograd_weights_avx2(float * dst, std::int64_t stride, const float * g, std::int64_t channels);
void sdasm_gated_gelu_avx2(float * out, const float * x, const float * gate, std::int64_t n);
void sdasm_winograd4_inputs_avx2(const sdasm::Winograd4_inputs * job);
void sdasm_winograd4_weights_avx2(float * dst, std::int64_t stride, const float * g, std::int64_t channels);
void sdasm_winograd4_outputs_avx2(const float * m, std::int64_t stride, float * row, std::int64_t ldc, std::int64_t groups,
                                  std::int64_t add);

// The micro-kernels of the quantized products for AVX-VNNI (x86_64/quant_avx2.S).
void sdasm_qgemm_6x16_au_avxvnni(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                                 float * c, std::int64_t ldc, std::int64_t accumulate);
void sdasm_qgemm_6x16_bu_avxvnni(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                                 float * c, std::int64_t ldc, std::int64_t accumulate);

// The quantization of a panel of sixteen columns (x86_64/quantize_avx2.S).
void sdasm_quantize16_avx2(std::uint8_t * dst, const float * src, std::int64_t groups, std::int64_t quads, float levels);

// And for AVX512-VNNI (x86_64/quant_avx512.S).
void sdasm_qgemm_6x16_au_avx512(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                                float * c, std::int64_t ldc, std::int64_t accumulate);
void sdasm_qgemm_6x16_bu_avx512(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                                float * c, std::int64_t ldc, std::int64_t accumulate);
void sdasm_qgemm_6x32_au_avx512(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                                std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);
void sdasm_qgemm_6x32_bu_avx512(std::int64_t groups, std::int64_t quads, const std::uint8_t * a, const std::uint8_t * b,
                                std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);

// The kernels written by hand for AVX-512 F, VL, BW and DQ (x86_64/*_avx512.S).
void sdasm_gemm_6x16_avx512(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                            std::int64_t accumulate);
void sdasm_gemm_6x32_avx512(std::int64_t kc, const float * a, const float * b, std::int64_t b_next, float * c,
                            std::int64_t ldc, std::int64_t accumulate);
void sdasm_scores_6x16_avx512(std::int64_t kc, const float * a, const float * b, float * c, std::int64_t ldc,
                              float * max, float scale, std::int64_t accumulate);
void sdasm_exponentials_avx512(float * s, std::int64_t n, const float * max, double * total);
void sdasm_winograd_inputs_avx512(const sdasm::Winograd_inputs * job);
void sdasm_winograd_outputs_avx512(const float * m, std::int64_t stride, float * row0, float * row1, std::int64_t groups,
                                   std::int64_t add);
void sdasm_gated_gelu_avx512(float * out, const float * x, const float * gate, std::int64_t n);
void sdasm_winograd4_inputs_avx512(const sdasm::Winograd4_inputs * job);
void sdasm_winograd4_outputs_avx512(const float * m, std::int64_t stride, float * row, std::int64_t ldc,
                                    std::int64_t groups, std::int64_t add);

// AMX-TILE and AMX-BF16 with AVX512-BF16 (x86_64/amx.S): Linux's leave for
// the process to use the tiles, 0 when given; the tiles' configuration and
// their release; the packing of A and B into tiles of bfloat16 in two parts,
// and the product of a block of 32 by 32 of C.
std::int64_t sdasm_amx_request();
void sdasm_amx_configure();
void sdasm_amx_release();
void sdasm_amx_pack_rows(std::uint16_t * dst, const float * src, std::int64_t stride, std::int64_t rows, std::int64_t k);
void sdasm_amx_pack_rows_f16(std::uint16_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t rows,
                             std::int64_t k);
void sdasm_amx_pack_weights(std::uint16_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t cols,
                            std::int64_t k);
void sdasm_amx_pack_panel(std::uint16_t * dst, const float * src, std::int64_t k);
// AVX512-BF16 (x86_64/bf16_avx512.S): the same block as sdasm_amx_bf16_2x2, on
// the vector units.
void sdasm_bf16_2x2_avx512(std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next, const std::uint16_t * b,
                           std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);
// AMX-FP16 (x86_64/amx_fp16.S): the FP16 weights as they are, the other
// operand in two parts of FP16.
void sdasm_amx_pack_rows_fp16(std::uint16_t * dst, const float * src, std::int64_t stride, std::int64_t rows,
                              std::int64_t k);
void sdasm_amx_pack_weights_fp16(std::uint16_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t cols,
                                 std::int64_t k);
void sdasm_amx_pack_rows_f16_one(std::uint16_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t rows,
                                 std::int64_t k);
void sdasm_amx_pack_panel_fp16(std::uint16_t * dst, const float * src, std::int64_t k);
void sdasm_amx_fp16_2x2_a2(std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next, const std::uint16_t * b,
                           std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);
void sdasm_amx_fp16_2x2_b2(std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next, const std::uint16_t * b,
                           std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);
// Single precision on the tiles, for attention (x86_64/amx_single.S): each
// number in three parts of bfloat16, six products for each one.
void sdasm_amx_pack_rows3(std::uint16_t * dst, const float * src, std::int64_t stride, std::int64_t rows, std::int64_t k);
void sdasm_amx_pack_columns3_f32(std::uint16_t * dst, const float * src, std::int64_t stride, std::int64_t cols,
                                 std::int64_t k);
void sdasm_amx_pack_columns3_f16(std::uint16_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t cols,
                                 std::int64_t k);
void sdasm_amx_pack_panel3_f32(std::uint16_t * dst, const float * src, std::int64_t stride, std::int64_t cols,
                               std::int64_t k);
void sdasm_amx_pack_panel3_f16(std::uint16_t * dst, const std::uint16_t * src, std::int64_t stride, std::int64_t cols,
                               std::int64_t k);
void sdasm_amx_single_2x2(std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next, const std::uint16_t * b,
                          std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);
void sdasm_exponentials_row_avx512(float * s, std::int64_t n, float max, double * total);
// Winograd's transformed weights into the tiles of A (x86_64/amx_winograd.S).
void sdasm_amx_winograd_weights(std::uint16_t * dst, std::int64_t xi_bytes, const float * widened, std::int64_t kc,
                                float * scratch);
void sdasm_amx_bf16_2x2(std::int64_t blocks, const std::uint16_t * a, std::int64_t a_next, const std::uint16_t * b,
                        std::int64_t b_next, float * c, std::int64_t ldc, std::int64_t accumulate);

}

#endif
