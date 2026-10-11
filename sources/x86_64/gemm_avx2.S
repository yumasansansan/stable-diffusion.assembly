// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_gemm_6x16_avx2(int64_t kc, const float *a, const float *b,
//                               float *c, int64_t ldc, int64_t accumulate);
//
// The micro-kernel of the matrix product (gemm.h) for AVX2 and FMA: a tile of
// six rows by sixteen columns of C, which stays in twelve registers through the
// kc steps of k, from a panel of A laid out [kc][6] and one of B laid out
// [kc][16], 32-byte aligned. Each step loads the row of B in two vectors and
// broadcasts each number of A across one, for twelve multiplications that add
// into the tile. ldc is the distance between the rows of C in bytes; the tile
// replaces them, or is added to them when accumulate is not zero.
//
// It does what the C++ micro-kernel (microkernel.cc) does, and in the same
// order, which the compiler, given that loop, makes of it too: the instructions
// are the same, but for the unrolling of the loop over k, four steps at a time
// here.
//
//     void sdasm_scores_6x16_avx2(int64_t kc, const float *a, const float *b,
//                                 float *c, int64_t ldc, float *max,
//                                 float scale, int64_t accumulate);
//
// The micro-kernel of attention's scores (attention.cc): the same tile, of six
// keys by sixteen queries, added to c when accumulate is not zero (c holding
// the sums of the stretches of k before), multiplied by scale in the registers
// it was summed in and written to c; then the largest of its six rows, lane by
// lane, raised into the sixteen numbers at max, each query's largest score so
// far. It is what scores_6x16_cxx() in microkernel.cc does, in the same
// order.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// One step of k: the row of B at b + 64*\i and the numbers of A at a + 24*\i.
.macro STEP i
        vmovaps ymm12, [rdx + 64*\i]
        vmovaps ymm13, [rdx + 64*\i + 32]
        vbroadcastss ymm14, [rsi + 24*\i]
        vfmadd231ps ymm0, ymm14, ymm12
        vfmadd231ps ymm1, ymm14, ymm13
        vbroadcastss ymm15, [rsi + 24*\i + 4]
        vfmadd231ps ymm2, ymm15, ymm12
        vfmadd231ps ymm3, ymm15, ymm13
        vbroadcastss ymm14, [rsi + 24*\i + 8]
        vfmadd231ps ymm4, ymm14, ymm12
        vfmadd231ps ymm5, ymm14, ymm13
        vbroadcastss ymm15, [rsi + 24*\i + 12]
        vfmadd231ps ymm6, ymm15, ymm12
        vfmadd231ps ymm7, ymm15, ymm13
        vbroadcastss ymm14, [rsi + 24*\i + 16]
        vfmadd231ps ymm8, ymm14, ymm12
        vfmadd231ps ymm9, ymm14, ymm13
        vbroadcastss ymm15, [rsi + 24*\i + 20]
        vfmadd231ps ymm10, ymm15, ymm12
        vfmadd231ps ymm11, ymm15, ymm13
.endm

        .globl sdasm_gemm_6x16_avx2
        .type sdasm_gemm_6x16_avx2, @function
        .p2align 5
sdasm_gemm_6x16_avx2:
        vxorps xmm0, xmm0, xmm0
        vxorps xmm1, xmm1, xmm1
        vxorps xmm2, xmm2, xmm2
        vxorps xmm3, xmm3, xmm3
        vxorps xmm4, xmm4, xmm4
        vxorps xmm5, xmm5, xmm5
        vxorps xmm6, xmm6, xmm6
        vxorps xmm7, xmm7, xmm7
        vxorps xmm8, xmm8, xmm8
        vxorps xmm9, xmm9, xmm9
        vxorps xmm10, xmm10, xmm10
        vxorps xmm11, xmm11, xmm11
        // rcx, rax and r10 address the rows of C two by two, r8 the second of
        // each pair.
        lea rax, [rcx + 2*r8]
        lea r10, [rax + 2*r8]
        prefetcht0 [rcx]
        prefetcht0 [rcx + r8]
        prefetcht0 [rax]
        prefetcht0 [rax + r8]
        prefetcht0 [r10]
        prefetcht0 [r10 + r8]

        mov r11, rdi
        shr r11, 2
        jz 2f
        .p2align 5
1:
        STEP 0
        STEP 1
        STEP 2
        STEP 3
        add rsi, 96
        add rdx, 256
        dec r11
        jnz 1b
2:
        and edi, 3
        jz 4f
3:
        STEP 0
        add rsi, 24
        add rdx, 64
        dec edi
        jnz 3b
4:
        test r9, r9
        jz 5f
        vaddps ymm0, ymm0, [rcx]
        vaddps ymm1, ymm1, [rcx + 32]
        vaddps ymm2, ymm2, [rcx + r8]
        vaddps ymm3, ymm3, [rcx + r8 + 32]
        vaddps ymm4, ymm4, [rax]
        vaddps ymm5, ymm5, [rax + 32]
        vaddps ymm6, ymm6, [rax + r8]
        vaddps ymm7, ymm7, [rax + r8 + 32]
        vaddps ymm8, ymm8, [r10]
        vaddps ymm9, ymm9, [r10 + 32]
        vaddps ymm10, ymm10, [r10 + r8]
        vaddps ymm11, ymm11, [r10 + r8 + 32]
5:
        vmovups [rcx], ymm0
        vmovups [rcx + 32], ymm1
        vmovups [rcx + r8], ymm2
        vmovups [rcx + r8 + 32], ymm3
        vmovups [rax], ymm4
        vmovups [rax + 32], ymm5
        vmovups [rax + r8], ymm6
        vmovups [rax + r8 + 32], ymm7
        vmovups [r10], ymm8
        vmovups [r10 + 32], ymm9
        vmovups [r10 + r8], ymm10
        vmovups [r10 + r8 + 32], ymm11
        vzeroupper
        ret
        .size sdasm_gemm_6x16_avx2, .-sdasm_gemm_6x16_avx2

        .globl sdasm_scores_6x16_avx2
        .type sdasm_scores_6x16_avx2, @function
        .p2align 5
sdasm_scores_6x16_avx2:
        // The scale waits below the stack pointer, in the red zone of a
        // function that calls none, while the tile takes every register.
        vmovss [rsp - 8], xmm0
        vxorps xmm0, xmm0, xmm0
        vxorps xmm1, xmm1, xmm1
        vxorps xmm2, xmm2, xmm2
        vxorps xmm3, xmm3, xmm3
        vxorps xmm4, xmm4, xmm4
        vxorps xmm5, xmm5, xmm5
        vxorps xmm6, xmm6, xmm6
        vxorps xmm7, xmm7, xmm7
        vxorps xmm8, xmm8, xmm8
        vxorps xmm9, xmm9, xmm9
        vxorps xmm10, xmm10, xmm10
        vxorps xmm11, xmm11, xmm11
        lea rax, [rcx + 2*r8]
        lea r10, [rax + 2*r8]

        mov r11, rdi
        shr r11, 2
        jz 2f
        .p2align 5
1:
        STEP 0
        STEP 1
        STEP 2
        STEP 3
        add rsi, 96
        add rdx, 256
        dec r11
        jnz 1b
2:
        and edi, 3
        jz 4f
3:
        STEP 0
        add rsi, 24
        add rdx, 64
        dec edi
        jnz 3b
4:
        // accumulate, the seventh argument, on the stack above the return
        // address.
        cmp qword ptr [rsp + 8], 0
        je 5f
        vaddps ymm0, ymm0, [rcx]
        vaddps ymm1, ymm1, [rcx + 32]
        vaddps ymm2, ymm2, [rcx + r8]
        vaddps ymm3, ymm3, [rcx + r8 + 32]
        vaddps ymm4, ymm4, [rax]
        vaddps ymm5, ymm5, [rax + 32]
        vaddps ymm6, ymm6, [rax + r8]
        vaddps ymm7, ymm7, [rax + r8 + 32]
        vaddps ymm8, ymm8, [r10]
        vaddps ymm9, ymm9, [r10 + 32]
        vaddps ymm10, ymm10, [r10 + r8]
        vaddps ymm11, ymm11, [r10 + r8 + 32]
5:
        vbroadcastss ymm12, [rsp - 8]
        vmulps ymm0, ymm0, ymm12
        vmulps ymm1, ymm1, ymm12
        vmulps ymm2, ymm2, ymm12
        vmulps ymm3, ymm3, ymm12
        vmulps ymm4, ymm4, ymm12
        vmulps ymm5, ymm5, ymm12
        vmulps ymm6, ymm6, ymm12
        vmulps ymm7, ymm7, ymm12
        vmulps ymm8, ymm8, ymm12
        vmulps ymm9, ymm9, ymm12
        vmulps ymm10, ymm10, ymm12
        vmulps ymm11, ymm11, ymm12
        vmovups [rcx], ymm0
        vmovups [rcx + 32], ymm1
        vmovups [rcx + r8], ymm2
        vmovups [rcx + r8 + 32], ymm3
        vmovups [rax], ymm4
        vmovups [rax + 32], ymm5
        vmovups [rax + r8], ymm6
        vmovups [rax + r8 + 32], ymm7
        vmovups [r10], ymm8
        vmovups [r10 + 32], ymm9
        vmovups [r10 + r8], ymm10
        vmovups [r10 + r8 + 32], ymm11
        // The largest of rows 0 and 1, 2 and 3, 4 and 5, then of those, then of
        // that and the largest so far: the left eight queries, then the right.
        vmaxps ymm0, ymm0, ymm2
        vmaxps ymm4, ymm4, ymm6
        vmaxps ymm8, ymm8, ymm10
        vmaxps ymm0, ymm0, ymm4
        vmaxps ymm0, ymm0, ymm8
        vmaxps ymm0, ymm0, [r9]
        vmovups [r9], ymm0
        vmaxps ymm1, ymm1, ymm3
        vmaxps ymm5, ymm5, ymm7
        vmaxps ymm9, ymm9, ymm11
        vmaxps ymm1, ymm1, ymm5
        vmaxps ymm1, ymm1, ymm9
        vmaxps ymm1, ymm1, [r9 + 32]
        vmovups [r9 + 32], ymm1
        vzeroupper
        ret
        .size sdasm_scores_6x16_avx2, .-sdasm_scores_6x16_avx2

        .section .note.GNU-stack, "", @progbits
