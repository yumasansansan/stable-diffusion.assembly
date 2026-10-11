// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The products of FP16 weights on AMX-FP16's tiles (Granite Rapids and after),
// as amx.S computes them on AMX-BF16's, for the linear layers and the direct
// convolutions: AMX-TILE, AMX-FP16, AVX-512 F, BW and VL.
//
// tdpfp16ps multiplies numbers of FP16, whose 11 bits hold an FP16 weight
// exactly, and sums the products in single precision, each product exact; it
// takes denormal numbers of FP16 as they are. The weights go to the tiles as
// they are, one part; the other operand, of single precision, in two parts of
// FP16, x = x1 + x2, x1 rounded to the nearest -- what ggml rounds it to before
// its products -- and x2 what x - x1 leaves, rounded: two products for each
// one, w x1 + w x2, where AMX-BF16's take three. x1 + x2 is within 2^-22 of x
// over FP16's range, and never further from x than x1, ggml's own rounding;
// past the range, |x| over 65504, x1 is infinite, as ggml's FP16 is.
//
//     void sdasm_amx_pack_rows_fp16(uint16_t *dst, const float *src,
//                                   int64_t stride, int64_t rows, int64_t k);
//
// The tiles of A of sixteen rows of single precision, as sdasm_amx_pack_rows
// packs them, in parts of FP16: x1, then x2, 2048 bytes a block of 32 of k.
//
//     void sdasm_amx_pack_weights_fp16(uint16_t *dst, const uint16_t *src,
//                                      int64_t stride, int64_t cols, int64_t k);
//
// The tiles of B from sixteen rows of FP16 weights, as sdasm_amx_pack_weights
// packs them, the weights as they are: a row for each pair of k, the pairs of
// the sixteen columns side by side, 1024 bytes a block.
//
//     void sdasm_amx_pack_rows_f16_one(uint16_t *dst, const uint16_t *src,
//                                      int64_t stride, int64_t rows, int64_t k);
//
// The tiles of A of sixteen rows of FP16 weights, as they are: 1024 bytes a
// block.
//
//     void sdasm_amx_pack_panel_fp16(uint16_t *dst, const float *src, int64_t k);
//
// The tiles of B from a panel [k][16] of single precision, as
// sdasm_amx_pack_panel packs it, in parts of FP16: 2048 bytes a block.
//
//     void sdasm_amx_fp16_2x2_a2(int64_t blocks, const uint16_t *a,
//                                int64_t a_next, const uint16_t *b,
//                                int64_t b_next, float *c, int64_t ldc,
//                                int64_t accumulate);
//     void sdasm_amx_fp16_2x2_b2(...);
//
// A block of 32 by 32 of C, as sdasm_amx_bf16_2x2: A in two parts and B the
// weights (a2), or A the weights and B in two parts (b2); six loads to eight
// products a block of k.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 6
.Lzeros16:
        .zero 128
.Linterleave16: // the words of two rows of sixteen, one of each in turn
        .short 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
        .short 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31

        .text

// k1 and k2: the numbers of a block of 32 inside k, of the r8 left.
.macro BLOCK_MASKS16
        mov eax, -1
        mov r11, 32
        cmp r8, r11
        cmovb r11, r8
        bzhi eax, eax, r11d
        kmovd k1, eax
        kshiftrd k2, k1, 16
.endm

        .globl sdasm_amx_pack_rows_fp16
        .type sdasm_amx_pack_rows_fp16, @function
        .p2align 5
// rdi dst, rsi src, rdx stride, rcx rows, r8 k
sdasm_amx_pack_rows_fp16:
        vpxord zmm8, zmm8, zmm8
        test r8, r8
        jle 9f
1:
        BLOCK_MASKS16
        mov r9, rsi
        xor r10, r10
2:
        mov rax, r10
        shl rax, 6
        cmp r10, rcx
        jae 4f
        vmovups zmm0{k1}{z}, [r9]
        vmovups zmm1{k2}{z}, [r9 + 64]
        vcvtps2ph ymm4, zmm0, 0             // x1, rounded to the nearest
        vcvtps2ph ymm5, zmm1, 0
        vcvtph2ps zmm6, ymm4
        vcvtph2ps zmm7, ymm5
        vinserti64x4 zmm2, zmm4, ymm5, 1
        vsubps zmm6, zmm0, zmm6          // x - x1, exactly
        vsubps zmm7, zmm1, zmm7
        vcvtps2ph ymm6, zmm6, 0             // x2
        vcvtps2ph ymm7, zmm7, 0
        vinserti64x4 zmm3, zmm6, ymm7, 1
        vmovups [rdi + rax], zmm2
        vmovups [rdi + rax + 1024], zmm3
        jmp 5f
4:
        vmovups [rdi + rax], zmm8
        vmovups [rdi + rax + 1024], zmm8
5:
        add r9, rdx
        inc r10
        cmp r10, 16
        jb 2b
        add rdi, 2048
        add rsi, 128
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_rows_fp16, .-sdasm_amx_pack_rows_fp16

        .globl sdasm_amx_pack_rows_f16_one
        .type sdasm_amx_pack_rows_f16_one, @function
        .p2align 5
// rdi dst, rsi src, rdx stride, rcx rows, r8 k
sdasm_amx_pack_rows_f16_one:
        vpxord zmm8, zmm8, zmm8
        test r8, r8
        jle 9f
1:
        BLOCK_MASKS16
        kunpckwd k1, k2, k1                     // the 32 words of the block
        mov r9, rsi
        xor r10, r10
2:
        mov rax, r10
        shl rax, 6
        cmp r10, rcx
        jae 4f
        vmovdqu16 zmm0{k1}{z}, [r9]
        vmovups [rdi + rax], zmm0
        jmp 5f
4:
        vmovups [rdi + rax], zmm8
5:
        add r9, rdx
        inc r10
        cmp r10, 16
        jb 2b
        add rdi, 1024
        add rsi, 64
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_rows_f16_one, .-sdasm_amx_pack_rows_f16_one

// The transpose of the sixteen rows (amx.S has the same) of sixteen dwords in zmm0..zmm15, into the
// same registers, zmm16..zmm31 its temporaries: the dwords of pairs of rows
// interleaved, then their qwords, then their 128-bit lanes, twice.
.macro TRANSPOSE16_H
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

        .globl sdasm_amx_pack_weights_fp16
        .type sdasm_amx_pack_weights_fp16, @function
        .p2align 5
// rdi dst, rsi src, rdx stride, rcx cols, r8 k
sdasm_amx_pack_weights_fp16:
        test r8, r8
        jle 9f
        lea r10, [rip + .Lzeros16]
1:
        BLOCK_MASKS16
        kunpckwd k1, k2, k1
        mov r9, rsi
        .irp c, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        mov r11, r9
        cmp rcx, \c
        cmovbe r11, r10
        vmovdqu16 zmm\c{k1}{z}, [r11]
        add r9, rdx
        .endr
        TRANSPOSE16_H
        .irp p, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovups [rdi + 64 * \p], zmm\p
        .endr
        add rdi, 1024
        add rsi, 64
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_weights_fp16, .-sdasm_amx_pack_weights_fp16

        .globl sdasm_amx_pack_panel_fp16
        .type sdasm_amx_pack_panel_fp16, @function
        .p2align 5
// rdi dst, rsi src, rdx k
sdasm_amx_pack_panel_fp16:
        test rdx, rdx
        jle 9f
        vmovdqu16 zmm31, [rip + .Linterleave16]
        lea r10, [rip + .Lzeros16]
        xor r8, r8                              // the first k of the block
1:
        xor r9, r9                              // the pair, 0 to 15
2:
        lea rax, [r8 + 2 * r9]
        mov r11, rax
        shl r11, 6
        add r11, rsi
        cmp rax, rdx
        cmovae r11, r10
        vmovups zmm0, [r11]
        inc rax
        mov r11, rax
        shl r11, 6
        add r11, rsi
        cmp rax, rdx
        cmovae r11, r10
        vmovups zmm1, [r11]
        vcvtps2ph ymm4, zmm0, 0             // x1, rounded to the nearest
        vcvtps2ph ymm5, zmm1, 0
        vcvtph2ps zmm6, ymm4
        vcvtph2ps zmm7, ymm5
        vinserti64x4 zmm2, zmm4, ymm5, 1
        vsubps zmm6, zmm0, zmm6          // x - x1, exactly
        vsubps zmm7, zmm1, zmm7
        vcvtps2ph ymm6, zmm6, 0             // x2
        vcvtps2ph ymm7, zmm7, 0
        vinserti64x4 zmm3, zmm6, ymm7, 1
        vpermw zmm2, zmm31, zmm2
        vpermw zmm3, zmm31, zmm3
        mov rax, r9
        shl rax, 6
        vmovups [rdi + rax], zmm2
        vmovups [rdi + rax + 1024], zmm3
        inc r9
        cmp r9, 16
        jb 2b
        add rdi, 2048
        add r8, 32
        cmp r8, rdx
        jb 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_panel_fp16, .-sdasm_amx_pack_panel_fp16

        .globl sdasm_amx_fp16_2x2_a2
        .type sdasm_amx_fp16_2x2_a2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_fp16_2x2_a2:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9
        add rdx, rsi
        add r8, rcx
        mov eax, 64
        cmp qword ptr [rsp + 16], 0
        je 1f
        tileloadd tmm0, [r9 + r10 * 1]
        tileloadd tmm1, [r9 + r10 * 1 + 64]
        tileloadd tmm2, [r11 + r10 * 1]
        tileloadd tmm3, [r11 + r10 * 1 + 64]
        jmp 2f
1:
        tilezero tmm0
        tilezero tmm1
        tilezero tmm2
        tilezero tmm3
2:
        test rdi, rdi
        jle 8f
        .p2align 4
3:
        tileloadd tmm4, [rsi + rax * 1]         // x1
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]         // w
        tileloadd tmm7, [r8 + rax * 1]
        tdpfp16ps tmm0, tmm4, tmm6
        tdpfp16ps tmm1, tmm4, tmm7
        tdpfp16ps tmm2, tmm5, tmm6
        tdpfp16ps tmm3, tmm5, tmm7
        tileloadd tmm4, [rsi + rax * 1 + 1024]  // x2
        tileloadd tmm5, [rdx + rax * 1 + 1024]
        tdpfp16ps tmm0, tmm4, tmm6
        tdpfp16ps tmm1, tmm4, tmm7
        tdpfp16ps tmm2, tmm5, tmm6
        tdpfp16ps tmm3, tmm5, tmm7
        add rsi, 2048
        add rdx, 2048
        add rcx, 1024
        add r8, 1024
        dec rdi
        jnz 3b
8:
        tilestored [r9 + r10 * 1], tmm0
        tilestored [r9 + r10 * 1 + 64], tmm1
        tilestored [r11 + r10 * 1], tmm2
        tilestored [r11 + r10 * 1 + 64], tmm3
        ret
        .size sdasm_amx_fp16_2x2_a2, .-sdasm_amx_fp16_2x2_a2

        .globl sdasm_amx_fp16_2x2_b2
        .type sdasm_amx_fp16_2x2_b2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_fp16_2x2_b2:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9
        add rdx, rsi
        add r8, rcx
        mov eax, 64
        cmp qword ptr [rsp + 16], 0
        je 1f
        tileloadd tmm0, [r9 + r10 * 1]
        tileloadd tmm1, [r9 + r10 * 1 + 64]
        tileloadd tmm2, [r11 + r10 * 1]
        tileloadd tmm3, [r11 + r10 * 1 + 64]
        jmp 2f
1:
        tilezero tmm0
        tilezero tmm1
        tilezero tmm2
        tilezero tmm3
2:
        test rdi, rdi
        jle 8f
        .p2align 4
3:
        tileloadd tmm4, [rsi + rax * 1]         // w
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]         // x1
        tileloadd tmm7, [r8 + rax * 1]
        tdpfp16ps tmm0, tmm4, tmm6
        tdpfp16ps tmm1, tmm4, tmm7
        tdpfp16ps tmm2, tmm5, tmm6
        tdpfp16ps tmm3, tmm5, tmm7
        tileloadd tmm6, [rcx + rax * 1 + 1024]  // x2
        tileloadd tmm7, [r8 + rax * 1 + 1024]
        tdpfp16ps tmm0, tmm4, tmm6
        tdpfp16ps tmm1, tmm4, tmm7
        tdpfp16ps tmm2, tmm5, tmm6
        tdpfp16ps tmm3, tmm5, tmm7
        add rsi, 1024
        add rdx, 1024
        add rcx, 2048
        add r8, 2048
        dec rdi
        jnz 3b
8:
        tilestored [r9 + r10 * 1], tmm0
        tilestored [r9 + r10 * 1 + 64], tmm1
        tilestored [r11 + r10 * 1], tmm2
        tilestored [r11 + r10 * 1 + 64], tmm3
        ret
        .size sdasm_amx_fp16_2x2_b2, .-sdasm_amx_fp16_2x2_b2

        .section .note.GNU-stack, "", @progbits
