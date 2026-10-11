// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_amx_winograd_weights(uint16_t *dst, int64_t xi_bytes,
//                                     const float *widened, int64_t kc,
//                                     float *scratch);
//
// Winograd's transformed weights, G g G^T, as the tiles of A of the products
// on AMX's tiles (winograd.cc): sixteen output channels by a block of 32 input
// channels, each of the sixteen elements xi of a tile into a tile of x1 and
// one of x2 at dst + xi * xi_bytes, the parts of bfloat16 of amx.S. The
// kernels come widened to single precision and side by side, sixteen output
// channels to a register, [k][3 * 3][16] (pack_rows_f16() of gemm.h); input
// channels past kc are zeros.
//
// The transform of an input channel takes the sixteen output channels at once,
// G g a column at a time and then (G g) G^T a row at a time, each sum of three
// taken left to right and halved after, as the C++ of winograd.cc takes it;
// each xi of the 32 input channels goes to `scratch` (32 kilobytes, 64-byte
// aligned), [xi][k][16]. Then each xi's 32 registers, k by output channel, are
// transposed sixteen by sixteen into the rows of its tile, output channel by
// k, split in two parts and stored. The C++ that transforms a kernel at a time
// with scalar arithmetic takes some 25 cycles a kernel, more than its products
// on the tiles take for the small images of the UNet's deepest layers.
//
// AVX-512 F, BW and VL, and AVX512-BF16; assembled by Clang, in Intel's syntax,
// for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 2
.Lhalf: .float 0.5

        .text

// The transpose of the sixteen rows (amx.S has the same) of sixteen dwords in zmm0..zmm15, into the
// same registers, zmm16..zmm31 its temporaries: the dwords of pairs of rows
// interleaved, then their qwords, then their 128-bit lanes, twice.
.macro TRANSPOSE16_W
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpckldq zmm24, zmm8, zmm9
        vpunpckhdq zmm25, zmm8, zmm9
        vpunpckldq zmm26, zmm10, zmm11
        vpunpckhdq zmm27, zmm10, zmm11
        vpunpckldq zmm28, zmm12, zmm13
        vpunpckhdq zmm29, zmm12, zmm13
        vpunpckldq zmm30, zmm14, zmm15
        vpunpckhdq zmm31, zmm14, zmm15
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vpunpcklqdq zmm8, zmm24, zmm26
        vpunpckhqdq zmm9, zmm24, zmm26
        vpunpcklqdq zmm10, zmm25, zmm27
        vpunpckhqdq zmm11, zmm25, zmm27
        vpunpcklqdq zmm12, zmm28, zmm30
        vpunpckhqdq zmm13, zmm28, zmm30
        vpunpcklqdq zmm14, zmm29, zmm31
        vpunpckhqdq zmm15, zmm29, zmm31
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xdd
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm21, zmm1, zmm5, 0xdd
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm22, zmm2, zmm6, 0xdd
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm23, zmm3, zmm7, 0xdd
        vshufi32x4 zmm24, zmm8, zmm12, 0x88
        vshufi32x4 zmm28, zmm8, zmm12, 0xdd
        vshufi32x4 zmm25, zmm9, zmm13, 0x88
        vshufi32x4 zmm29, zmm9, zmm13, 0xdd
        vshufi32x4 zmm26, zmm10, zmm14, 0x88
        vshufi32x4 zmm30, zmm10, zmm14, 0xdd
        vshufi32x4 zmm27, zmm11, zmm15, 0x88
        vshufi32x4 zmm31, zmm11, zmm15, 0xdd
        vshufi32x4 zmm0, zmm16, zmm24, 0x88
        vshufi32x4 zmm8, zmm16, zmm24, 0xdd
        vshufi32x4 zmm1, zmm17, zmm25, 0x88
        vshufi32x4 zmm9, zmm17, zmm25, 0xdd
        vshufi32x4 zmm2, zmm18, zmm26, 0x88
        vshufi32x4 zmm10, zmm18, zmm26, 0xdd
        vshufi32x4 zmm3, zmm19, zmm27, 0x88
        vshufi32x4 zmm11, zmm19, zmm27, 0xdd
        vshufi32x4 zmm4, zmm20, zmm28, 0x88
        vshufi32x4 zmm12, zmm20, zmm28, 0xdd
        vshufi32x4 zmm5, zmm21, zmm29, 0x88
        vshufi32x4 zmm13, zmm21, zmm29, 0xdd
        vshufi32x4 zmm6, zmm22, zmm30, 0x88
        vshufi32x4 zmm14, zmm22, zmm30, 0xdd
        vshufi32x4 zmm7, zmm23, zmm31, 0x88
        vshufi32x4 zmm15, zmm23, zmm31, 0xdd
.endm


        .globl sdasm_amx_winograd_weights
        .type sdasm_amx_winograd_weights, @function
        .p2align 5
// rdi dst, rsi xi_bytes, rdx widened, rcx kc, r8 scratch
sdasm_amx_winograd_weights:
        vbroadcastss zmm31, [rip + .Lhalf]
        vpxord zmm30, zmm30, zmm30
        mov r10, r8
        xor r11, r11                            // k
1:
        // Input channel k: its xi to scratch + 2048 xi + 64 k.
        cmp r11, rcx
        jae 2f
        vmovups zmm0, [rdx + 64 * 0]
        vmovups zmm1, [rdx + 64 * 1]
        vmovups zmm2, [rdx + 64 * 2]
        vmovups zmm3, [rdx + 64 * 3]
        vmovups zmm4, [rdx + 64 * 4]
        vmovups zmm5, [rdx + 64 * 5]
        vmovups zmm6, [rdx + 64 * 6]
        vmovups zmm7, [rdx + 64 * 7]
        vmovups zmm8, [rdx + 64 * 8]
        vaddps zmm9, zmm0, zmm3           // g0 + g1
        vsubps zmm12, zmm0, zmm3          // g0 - g1
        vaddps zmm9, zmm9, zmm6
        vaddps zmm12, zmm12, zmm6
        vmulps zmm9, zmm9, zmm31
        vmulps zmm12, zmm12, zmm31
        vaddps zmm10, zmm1, zmm4           // g0 + g1
        vsubps zmm13, zmm1, zmm4          // g0 - g1
        vaddps zmm10, zmm10, zmm7
        vaddps zmm13, zmm13, zmm7
        vmulps zmm10, zmm10, zmm31
        vmulps zmm13, zmm13, zmm31
        vaddps zmm11, zmm2, zmm5           // g0 + g1
        vsubps zmm14, zmm2, zmm5          // g0 - g1
        vaddps zmm11, zmm11, zmm8
        vaddps zmm14, zmm14, zmm8
        vmulps zmm11, zmm11, zmm31
        vmulps zmm14, zmm14, zmm31
        vmovaps [r8 + 2048 * 0 ], zmm0
        vaddps zmm15, zmm0, zmm1
        vsubps zmm16, zmm0, zmm1
        vaddps zmm15, zmm15, zmm2
        vaddps zmm16, zmm16, zmm2
        vmulps zmm15, zmm15, zmm31
        vmulps zmm16, zmm16, zmm31
        vmovaps [r8 + 2048 * 1], zmm15
        vmovaps [r8 + 2048 * 2], zmm16
        vmovaps [r8 + 2048 * 3], zmm2
        vmovaps [r8 + 2048 * 4 ], zmm9
        vaddps zmm15, zmm9, zmm10
        vsubps zmm16, zmm9, zmm10
        vaddps zmm15, zmm15, zmm11
        vaddps zmm16, zmm16, zmm11
        vmulps zmm15, zmm15, zmm31
        vmulps zmm16, zmm16, zmm31
        vmovaps [r8 + 2048 * 5], zmm15
        vmovaps [r8 + 2048 * 6], zmm16
        vmovaps [r8 + 2048 * 7], zmm11
        vmovaps [r8 + 2048 * 8 ], zmm12
        vaddps zmm15, zmm12, zmm13
        vsubps zmm16, zmm12, zmm13
        vaddps zmm15, zmm15, zmm14
        vaddps zmm16, zmm16, zmm14
        vmulps zmm15, zmm15, zmm31
        vmulps zmm16, zmm16, zmm31
        vmovaps [r8 + 2048 * 9], zmm15
        vmovaps [r8 + 2048 * 10], zmm16
        vmovaps [r8 + 2048 * 11], zmm14
        vmovaps [r8 + 2048 * 12 ], zmm6
        vaddps zmm15, zmm6, zmm7
        vsubps zmm16, zmm6, zmm7
        vaddps zmm15, zmm15, zmm8
        vaddps zmm16, zmm16, zmm8
        vmulps zmm15, zmm15, zmm31
        vmulps zmm16, zmm16, zmm31
        vmovaps [r8 + 2048 * 13], zmm15
        vmovaps [r8 + 2048 * 14], zmm16
        vmovaps [r8 + 2048 * 15], zmm8
        add rdx, 576
        jmp 3f
2:
        .irp xi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovaps [r8 + 2048 * \xi], zmm30
        .endr
3:
        add r8, 64
        inc r11
        cmp r11, 32
        jb 1b

        // Each xi: its k 0 to 15, transposed, back to its scratch; its k 16 to
        // 31, transposed; then each output channel's 32 k split and stored.
        mov r8, r10
        mov rax, rdi
        xor r11, r11                            // xi
4:
        mov r9, r8
        .irp k, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovaps zmm\k, [r9 + 64 * \k]
        .endr
        TRANSPOSE16_W
        .irp k, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovaps [r9 + 64 * \k], zmm\k
        .endr
        .irp k, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovaps zmm\k, [r9 + 1024 + 64 * \k]
        .endr
        TRANSPOSE16_W
        vmovaps zmm16, [r9 + 64 * 0]
        vcvtne2ps2bf16 zmm17, zmm0, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm0, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 0], zmm17
        vmovups [rax + 64 * 0 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 1]
        vcvtne2ps2bf16 zmm17, zmm1, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm1, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 1], zmm17
        vmovups [rax + 64 * 1 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 2]
        vcvtne2ps2bf16 zmm17, zmm2, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm2, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 2], zmm17
        vmovups [rax + 64 * 2 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 3]
        vcvtne2ps2bf16 zmm17, zmm3, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm3, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 3], zmm17
        vmovups [rax + 64 * 3 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 4]
        vcvtne2ps2bf16 zmm17, zmm4, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm4, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 4], zmm17
        vmovups [rax + 64 * 4 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 5]
        vcvtne2ps2bf16 zmm17, zmm5, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm5, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 5], zmm17
        vmovups [rax + 64 * 5 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 6]
        vcvtne2ps2bf16 zmm17, zmm6, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm6, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 6], zmm17
        vmovups [rax + 64 * 6 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 7]
        vcvtne2ps2bf16 zmm17, zmm7, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm7, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 7], zmm17
        vmovups [rax + 64 * 7 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 8]
        vcvtne2ps2bf16 zmm17, zmm8, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm8, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 8], zmm17
        vmovups [rax + 64 * 8 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 9]
        vcvtne2ps2bf16 zmm17, zmm9, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm9, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 9], zmm17
        vmovups [rax + 64 * 9 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 10]
        vcvtne2ps2bf16 zmm17, zmm10, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm10, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 10], zmm17
        vmovups [rax + 64 * 10 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 11]
        vcvtne2ps2bf16 zmm17, zmm11, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm11, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 11], zmm17
        vmovups [rax + 64 * 11 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 12]
        vcvtne2ps2bf16 zmm17, zmm12, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm12, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 12], zmm17
        vmovups [rax + 64 * 12 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 13]
        vcvtne2ps2bf16 zmm17, zmm13, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm13, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 13], zmm17
        vmovups [rax + 64 * 13 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 14]
        vcvtne2ps2bf16 zmm17, zmm14, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm14, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 14], zmm17
        vmovups [rax + 64 * 14 + 1024], zmm21
        vmovaps zmm16, [r9 + 64 * 15]
        vcvtne2ps2bf16 zmm17, zmm15, zmm16
        vpmovzxwd zmm18, ymm17
        vpslld zmm18, zmm18, 16
        vextracti64x4 ymm19, zmm17, 1
        vpmovzxwd zmm19, ymm19
        vpslld zmm19, zmm19, 16
        vsubps zmm16, zmm16, zmm18
        vsubps zmm20, zmm15, zmm19
        vcvtne2ps2bf16 zmm21, zmm20, zmm16
        vmovups [rax + 64 * 15], zmm17
        vmovups [rax + 64 * 15 + 1024], zmm21
        add r8, 2048
        add rax, rsi
        inc r11
        cmp r11, 16
        jb 4b
        vzeroupper
        ret
        .size sdasm_amx_winograd_weights, .-sdasm_amx_winograd_weights

        .section .note.GNU-stack, "", @progbits
