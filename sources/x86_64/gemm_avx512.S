// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_gemm_6x16_avx512(int64_t kc, const float *a, const float *b,
//                                 float *c, int64_t ldc, int64_t accumulate);
//     void sdasm_scores_6x16_avx512(int64_t kc, const float *a,
//                                   const float *b, float *c, int64_t ldc,
//                                   float *max, float scale,
//                                   int64_t accumulate);
//
// The micro-kernels of gemm_avx2.S for AVX-512 (F, with VL, BW and DQ, as
// AVX10.1 has them): the same tile of six rows by sixteen columns, from the
// same panels, a row of the tile in one register of sixteen. Each step of k
// loads the row of B in one register, and multiplies it by each of the six
// numbers of A broadcast from memory by the multiplication itself ({1to16}).
// Six registers summing one after another would wait on the latency of the
// multiplication; the even steps of k are summed in six registers and the odd
// in six others, which are added together at the end -- twelve sums at once,
// in the order of the steps within each.
//
// The scores' kernel then multiplies the tile by the scale, held in a
// register of its own (AVX-512 has thirty-two), writes it, and raises the
// sixteen largest scores at max to the largest of the six rows.
//
//     void sdasm_gemm_6x32_avx512(int64_t kc, const float *a, const float *b,
//                                 int64_t b_next, float *c, int64_t ldc,
//                                 int64_t accumulate);
//
// A tile of six rows by 32 columns, from a panel of A and two of B, the second
// b_next bytes after the first (gemm.h, Microkernel_wide). Each step of k loads
// a row of each panel of B, and each number of A broadcast into a register by
// vbroadcastss, a load alone, which two multiplications take: eight loads to
// twelve FMAs, where the tile of sixteen columns makes seven to six. On the
// cores of Intel that make two FMAs of sixteen a cycle and two loads, those
// seven take three and a half cycles to the FMAs' three, and the eight here
// four to their six. Twelve sums, each summed in the order of the steps.
//
//     void sdasm_scores_6x32_avx512(int64_t kc, const float *a,
//                                   const float *b, int64_t b_next, float *c,
//                                   int64_t c_next, int64_t ldc, float *max,
//                                   float scale, int64_t flags);
//
// The tile of six rows by 32 columns of attention's scores, from a panel of
// the keys and two of the queries, the second b_next bytes after the first,
// as sdasm_gemm_6x32_avx512 computes it: its columns 0 to 15 written at c and
// 16 to 31 at c + c_next bytes, each six rows ldc bytes apart -- the two
// panels of the scores of two panels of queries, which lie apart. With bit 0
// of flags, the tile is added to them; with bit 1, it is multiplied by the
// scale and the 32 largest scores at max raised to the largest of the six
// rows, as sdasm_scores_6x16_avx512 does; without, it is a partial sum.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// One step of k into the sums of set \s (0: zmm0..zmm5, 1: zmm6..zmm11): the
// row of B at rdx + 64 * \i into zmm\b, the six numbers of A at rsi + 24 * \i.
.macro STEP i, b, s0, s1, s2, s3, s4, s5
        vmovups zmm\b, [rdx + 64 * (\i)]
        vfmadd231ps zmm\s0, zmm\b, dword ptr [rsi + 24 * (\i) + 0]{1to16}
        vfmadd231ps zmm\s1, zmm\b, dword ptr [rsi + 24 * (\i) + 4]{1to16}
        vfmadd231ps zmm\s2, zmm\b, dword ptr [rsi + 24 * (\i) + 8]{1to16}
        vfmadd231ps zmm\s3, zmm\b, dword ptr [rsi + 24 * (\i) + 12]{1to16}
        vfmadd231ps zmm\s4, zmm\b, dword ptr [rsi + 24 * (\i) + 16]{1to16}
        vfmadd231ps zmm\s5, zmm\b, dword ptr [rsi + 24 * (\i) + 20]{1to16}
.endm
.macro EVEN i
        STEP \i, 16, 0, 1, 2, 3, 4, 5
.endm
.macro ODD i
        STEP \i, 17, 6, 7, 8, 9, 10, 11
.endm

// The product of the panels into zmm0..zmm5: rdi kc, rsi a, rdx b.
.macro PRODUCT
        vpxord zmm0, zmm0, zmm0
        vpxord zmm1, zmm1, zmm1
        vpxord zmm2, zmm2, zmm2
        vpxord zmm3, zmm3, zmm3
        vpxord zmm4, zmm4, zmm4
        vpxord zmm5, zmm5, zmm5
        vpxord zmm6, zmm6, zmm6
        vpxord zmm7, zmm7, zmm7
        vpxord zmm8, zmm8, zmm8
        vpxord zmm9, zmm9, zmm9
        vpxord zmm10, zmm10, zmm10
        vpxord zmm11, zmm11, zmm11
        mov r11, rdi
        shr r11, 2
        jz 2f
        .p2align 5
1:
        EVEN 0
        ODD 1
        EVEN 2
        ODD 3
        add rsi, 96
        add rdx, 256
        dec r11
        jnz 1b
2:
        and edi, 3
        jz 4f
3:
        EVEN 0
        add rsi, 24
        add rdx, 64
        dec edi
        jz 4f
        ODD 0
        add rsi, 24
        add rdx, 64
        dec edi
        jnz 3b
4:
        vaddps zmm0, zmm0, zmm6
        vaddps zmm1, zmm1, zmm7
        vaddps zmm2, zmm2, zmm8
        vaddps zmm3, zmm3, zmm9
        vaddps zmm4, zmm4, zmm10
        vaddps zmm5, zmm5, zmm11
        // The rows of C: rcx, rcx + r8, rcx + 2 r8, rax, rcx + 4 r8, r10.
        lea rax, [rcx + 2*r8]
        add rax, r8
        lea r10, [rax + 2*r8]
.endm

// C added to the tile, a row to a register.
.macro ADD_C
        vaddps zmm0, zmm0, [rcx]
        vaddps zmm1, zmm1, [rcx + r8]
        vaddps zmm2, zmm2, [rcx + 2*r8]
        vaddps zmm3, zmm3, [rax]
        vaddps zmm4, zmm4, [rcx + 4*r8]
        vaddps zmm5, zmm5, [r10]
.endm

.macro STORE_C
        vmovups [rcx], zmm0
        vmovups [rcx + r8], zmm1
        vmovups [rcx + 2*r8], zmm2
        vmovups [rax], zmm3
        vmovups [rcx + 4*r8], zmm4
        vmovups [r10], zmm5
.endm

        .globl sdasm_gemm_6x16_avx512
        .type sdasm_gemm_6x16_avx512, @function
        .p2align 5
sdasm_gemm_6x16_avx512:
        PRODUCT
        test r9, r9
        jz 5f
        ADD_C
5:
        STORE_C
        vzeroupper
        ret
        .size sdasm_gemm_6x16_avx512, .-sdasm_gemm_6x16_avx512

        .globl sdasm_scores_6x16_avx512
        .type sdasm_scores_6x16_avx512, @function
        .p2align 5
// r9 max, xmm0 scale, [rsp + 8] accumulate
sdasm_scores_6x16_avx512:
        vbroadcastss zmm31, xmm0
        PRODUCT
        cmp qword ptr [rsp + 8], 0
        je 5f
        ADD_C
5:
        vmulps zmm0, zmm0, zmm31
        vmulps zmm1, zmm1, zmm31
        vmulps zmm2, zmm2, zmm31
        vmulps zmm3, zmm3, zmm31
        vmulps zmm4, zmm4, zmm31
        vmulps zmm5, zmm5, zmm31
        STORE_C
        vmaxps zmm6, zmm0, zmm1
        vmaxps zmm7, zmm2, zmm3
        vmaxps zmm8, zmm4, zmm5
        vmaxps zmm6, zmm6, zmm7
        vmaxps zmm6, zmm6, zmm8
        vmaxps zmm6, zmm6, [r9]
        vmovups [r9], zmm6
        vzeroupper
        ret
        .size sdasm_scores_6x16_avx512, .-sdasm_scores_6x16_avx512

// ---------------------------------------------------------------- 6 by 32

// One step of k of the tile of six rows by 32 columns: the rows of the two
// panels of B at rdx + 64 * \i and rdx + rcx + 64 * \i into zmm12 and zmm13,
// the six numbers of A at rsi + 24 * \i broadcast into zmm14..zmm19; columns 0
// to 15 of row r summed in zmm r, 16 to 31 in zmm 6 + r.
.macro WIDE_STEP i
        vmovups zmm12, [rdx + 64 * (\i)]
        vmovups zmm13, [rdx + rcx + 64 * (\i)]
        vbroadcastss zmm14, dword ptr [rsi + 24 * (\i) + 0]
        vbroadcastss zmm15, dword ptr [rsi + 24 * (\i) + 4]
        vbroadcastss zmm16, dword ptr [rsi + 24 * (\i) + 8]
        vbroadcastss zmm17, dword ptr [rsi + 24 * (\i) + 12]
        vbroadcastss zmm18, dword ptr [rsi + 24 * (\i) + 16]
        vbroadcastss zmm19, dword ptr [rsi + 24 * (\i) + 20]
        vfmadd231ps zmm0, zmm12, zmm14
        vfmadd231ps zmm6, zmm13, zmm14
        vfmadd231ps zmm1, zmm12, zmm15
        vfmadd231ps zmm7, zmm13, zmm15
        vfmadd231ps zmm2, zmm12, zmm16
        vfmadd231ps zmm8, zmm13, zmm16
        vfmadd231ps zmm3, zmm12, zmm17
        vfmadd231ps zmm9, zmm13, zmm17
        vfmadd231ps zmm4, zmm12, zmm18
        vfmadd231ps zmm10, zmm13, zmm18
        vfmadd231ps zmm5, zmm12, zmm19
        vfmadd231ps zmm11, zmm13, zmm19
.endm

        .globl sdasm_gemm_6x32_avx512
        .type sdasm_gemm_6x32_avx512, @function
        .p2align 5
// rdi kc, rsi a, rdx b, rcx b_next, r8 c, r9 ldc, [rsp + 8] accumulate
sdasm_gemm_6x32_avx512:
        vpxord zmm0, zmm0, zmm0
        vpxord zmm1, zmm1, zmm1
        vpxord zmm2, zmm2, zmm2
        vpxord zmm3, zmm3, zmm3
        vpxord zmm4, zmm4, zmm4
        vpxord zmm5, zmm5, zmm5
        vpxord zmm6, zmm6, zmm6
        vpxord zmm7, zmm7, zmm7
        vpxord zmm8, zmm8, zmm8
        vpxord zmm9, zmm9, zmm9
        vpxord zmm10, zmm10, zmm10
        vpxord zmm11, zmm11, zmm11
        mov r11, rdi
        shr r11, 2
        jz 2f
        .p2align 5
1:
        WIDE_STEP 0
        WIDE_STEP 1
        WIDE_STEP 2
        WIDE_STEP 3
        add rsi, 96
        add rdx, 256
        dec r11
        jnz 1b
2:
        and edi, 3
        jz 4f
3:
        WIDE_STEP 0
        add rsi, 24
        add rdx, 64
        dec edi
        jnz 3b
4:
        // The rows of C: r8, r8 + r9, r8 + 2 r9, r10, r8 + 4 r9, r11.
        lea r10, [r8 + 2*r9]
        add r10, r9
        lea r11, [r10 + 2*r9]
        cmp qword ptr [rsp + 8], 0
        je 5f
        vaddps zmm0, zmm0, [r8]
        vaddps zmm6, zmm6, [r8 + 64]
        vaddps zmm1, zmm1, [r8 + r9]
        vaddps zmm7, zmm7, [r8 + r9 + 64]
        vaddps zmm2, zmm2, [r8 + 2*r9]
        vaddps zmm8, zmm8, [r8 + 2*r9 + 64]
        vaddps zmm3, zmm3, [r10]
        vaddps zmm9, zmm9, [r10 + 64]
        vaddps zmm4, zmm4, [r8 + 4*r9]
        vaddps zmm10, zmm10, [r8 + 4*r9 + 64]
        vaddps zmm5, zmm5, [r11]
        vaddps zmm11, zmm11, [r11 + 64]
5:
        vmovups [r8], zmm0
        vmovups [r8 + 64], zmm6
        vmovups [r8 + r9], zmm1
        vmovups [r8 + r9 + 64], zmm7
        vmovups [r8 + 2*r9], zmm2
        vmovups [r8 + 2*r9 + 64], zmm8
        vmovups [r10], zmm3
        vmovups [r10 + 64], zmm9
        vmovups [r8 + 4*r9], zmm4
        vmovups [r8 + 4*r9 + 64], zmm10
        vmovups [r11], zmm5
        vmovups [r11 + 64], zmm11
        vzeroupper
        ret
        .size sdasm_gemm_6x32_avx512, .-sdasm_gemm_6x32_avx512

        .globl sdasm_scores_6x32_avx512
        .type sdasm_scores_6x32_avx512, @function
        .p2align 5
// rdi kc, rsi a, rdx b, rcx b_next, r8 c, r9 c_next, [rsp + 8] ldc,
// [rsp + 16] max, xmm0 scale, [rsp + 24] flags
sdasm_scores_6x32_avx512:
        vbroadcastss zmm31, xmm0
        vpxord zmm0, zmm0, zmm0
        vpxord zmm1, zmm1, zmm1
        vpxord zmm2, zmm2, zmm2
        vpxord zmm3, zmm3, zmm3
        vpxord zmm4, zmm4, zmm4
        vpxord zmm5, zmm5, zmm5
        vpxord zmm6, zmm6, zmm6
        vpxord zmm7, zmm7, zmm7
        vpxord zmm8, zmm8, zmm8
        vpxord zmm9, zmm9, zmm9
        vpxord zmm10, zmm10, zmm10
        vpxord zmm11, zmm11, zmm11
        mov r11, rdi
        shr r11, 2
        jz 2f
        .p2align 5
1:
        WIDE_STEP 0
        WIDE_STEP 1
        WIDE_STEP 2
        WIDE_STEP 3
        add rsi, 96
        add rdx, 256
        dec r11
        jnz 1b
2:
        and edi, 3
        jz 4f
3:
        WIDE_STEP 0
        add rsi, 24
        add rdx, 64
        dec edi
        jnz 3b
4:
        // The rows of the two tiles: r8 and r9, then ldc, 2 ldc, 3 ldc (r10),
        // 4 ldc and 5 ldc (r11) after each.
        mov rax, [rsp + 8]
        lea r10, [rax + 2*rax]
        lea r11, [rax + 4*rax]
        add r9, r8
        test byte ptr [rsp + 24], 1
        jz 5f
        vaddps zmm0, zmm0, [r8]
        vaddps zmm1, zmm1, [r8 + rax]
        vaddps zmm2, zmm2, [r8 + 2*rax]
        vaddps zmm3, zmm3, [r8 + r10]
        vaddps zmm4, zmm4, [r8 + 4*rax]
        vaddps zmm5, zmm5, [r8 + r11]
        vaddps zmm6, zmm6, [r9]
        vaddps zmm7, zmm7, [r9 + rax]
        vaddps zmm8, zmm8, [r9 + 2*rax]
        vaddps zmm9, zmm9, [r9 + r10]
        vaddps zmm10, zmm10, [r9 + 4*rax]
        vaddps zmm11, zmm11, [r9 + r11]
5:
        test byte ptr [rsp + 24], 2
        jz 6f
        vmulps zmm0, zmm0, zmm31
        vmulps zmm1, zmm1, zmm31
        vmulps zmm2, zmm2, zmm31
        vmulps zmm3, zmm3, zmm31
        vmulps zmm4, zmm4, zmm31
        vmulps zmm5, zmm5, zmm31
        vmulps zmm6, zmm6, zmm31
        vmulps zmm7, zmm7, zmm31
        vmulps zmm8, zmm8, zmm31
        vmulps zmm9, zmm9, zmm31
        vmulps zmm10, zmm10, zmm31
        vmulps zmm11, zmm11, zmm31
        mov rdx, [rsp + 16]
        vmaxps zmm12, zmm0, zmm1
        vmaxps zmm13, zmm2, zmm3
        vmaxps zmm14, zmm4, zmm5
        vmaxps zmm12, zmm12, zmm13
        vmaxps zmm12, zmm12, zmm14
        vmaxps zmm12, zmm12, [rdx]
        vmovups [rdx], zmm12
        vmaxps zmm12, zmm6, zmm7
        vmaxps zmm13, zmm8, zmm9
        vmaxps zmm14, zmm10, zmm11
        vmaxps zmm12, zmm12, zmm13
        vmaxps zmm12, zmm12, zmm14
        vmaxps zmm12, zmm12, [rdx + 64]
        vmovups [rdx + 64], zmm12
6:
        vmovups [r8], zmm0
        vmovups [r8 + rax], zmm1
        vmovups [r8 + 2*rax], zmm2
        vmovups [r8 + r10], zmm3
        vmovups [r8 + 4*rax], zmm4
        vmovups [r8 + r11], zmm5
        vmovups [r9], zmm6
        vmovups [r9 + rax], zmm7
        vmovups [r9 + 2*rax], zmm8
        vmovups [r9 + r10], zmm9
        vmovups [r9 + 4*rax], zmm10
        vmovups [r9 + r11], zmm11
        vzeroupper
        ret
        .size sdasm_scores_6x32_avx512, .-sdasm_scores_6x32_avx512

        .section .note.GNU-stack, "", @progbits
