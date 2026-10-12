// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_qgemm_6x16_au_avx512(int64_t groups, int64_t quads,
//                                     const uint8_t *a, const uint8_t *b,
//                                     float *c, int64_t ldc,
//                                     int64_t accumulate);
//     void sdasm_qgemm_6x16_bu_avx512(...);
//
// The micro-kernels of quant_avx2.S for AVX512-VNNI, a row of the tile in one
// register of sixteen lanes: a quad of k loads the four bytes of B's sixteen
// columns in one register and adds the products of each row's four bytes of A
// to the row. Six sums one after another would wait on vpdpbusd's latency of
// five cycles; the even quads are summed in six registers and the odd ones in
// six others, added together at the end of each group, as gemm_avx512.S sums
// the steps of k of single precision. Where B is the unsigned operand (bu), A's
// bytes are broadcast by vpdpbusd itself ({1to16}); where A is (au), by
// vpbroadcastd into a register, which vpdpbusd's unsigned operand has to be.
//
//     void sdasm_qgemm_6x32_au_avx512(int64_t groups, int64_t quads,
//                                     const uint8_t *a, const uint8_t *b,
//                                     int64_t b_next, float *c, int64_t ldc,
//                                     int64_t accumulate);
//     void sdasm_qgemm_6x32_bu_avx512(...);
//
// A tile of six rows by 32 columns, from a panel of A and two of B, the second
// b_next bytes after the first (quant.h, Qkernel_wide): eight loads to twelve
// vpdpbusd, where the tile of sixteen makes seven to six -- what the tile of 32
// of single precision does for the loads of Intel's cores (gemm_avx512.S).
//
// Each scales its sums into C after each group as quant_avx2.S does, in the
// same order, the scales' and the offsets' products taking the row's scale and
// offset broadcast from memory by the multiplications themselves.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// ---------------------------------------------------------------- 6 by 16

// One quad of k into the six sums zmm\r0..zmm\r5: B's sixteen columns at
// rcx + 64*\i into zmm\b, A's six rows at rdx + 24*\i.
.macro QUAD16 i, b, au, r0, r1, r2, r3, r4, r5
        vmovdqu32 zmm\b, [rcx + 64*\i]
        ROW16 \i, \b, \au, 0, \r0
        ROW16 \i, \b, \au, 1, \r1
        ROW16 \i, \b, \au, 2, \r2
        ROW16 \i, \b, \au, 3, \r3
        ROW16 \i, \b, \au, 4, \r4
        ROW16 \i, \b, \au, 5, \r5
.endm

// Row \row of A's quad \i by B's quad in zmm\b, into zmm\d: A's rows 0 to 2
// at rdx, rdx + 4*rsi and rdx + 8*rsi, 3 to 5 the same from rax.
.macro ROW16 i, b, au, row, d
.if \row == 0
        AROW \i, \au, \b, \d, [rdx + 4*\i]
.elseif \row == 1
        AROW \i, \au, \b, \d, [rdx + 4*rsi + 4*\i]
.elseif \row == 2
        AROW \i, \au, \b, \d, [rdx + 8*rsi + 4*\i]
.elseif \row == 3
        AROW \i, \au, \b, \d, [rax + 4*\i]
.elseif \row == 4
        AROW \i, \au, \b, \d, [rax + 4*rsi + 4*\i]
.else
        AROW \i, \au, \b, \d, [rax + 8*rsi + 4*\i]
.endif
.endm

.macro AROW i, au, b, d, at:vararg
.if \au
        vpbroadcastd zmm14, \at
        vpdpbusd zmm\d, zmm14, zmm\b
.else
        vpdpbusd zmm\d, zmm\b, dword ptr \at{1to16}
.endif
.endm

// Row \r of the tile, its sums in zmm\r, scaled and added to C's row at \row:
// rdx holds the rows' scales and offsets, zmm16 and zmm17 the columns'.
.macro SCALE16 r, row
        vmulps zmm15, zmm16, dword ptr [rdx + 4*\r]{1to16}
        vcvtdq2ps zmm\r, zmm\r
        vfmadd213ps zmm\r, zmm15, [\row]
        vfnmadd231ps zmm\r, zmm17, dword ptr [rdx + 24 + 4*\r]{1to16}
        vmovups [\row], zmm\r
.endm

.macro QGEMM16 name, au
        .globl \name
        .type \name, @function
        .p2align 5
\name:
        // The rows of C: r8, r8 + r9, r8 + 2 r9, r10, r10 + r9, r10 + 2 r9.
        lea r10, [r9 + 2*r9]
        add r10, r8
        cmp qword ptr [rsp + 8], 0
        jne 1f
        vpxord zmm0, zmm0, zmm0
        vmovups [r8], zmm0
        vmovups [r8 + r9], zmm0
        vmovups [r8 + 2*r9], zmm0
        vmovups [r10], zmm0
        vmovups [r10 + r9], zmm0
        vmovups [r10 + 2*r9], zmm0
1:
        prefetcht0 [r8]
        prefetcht0 [r8 + r9]
        prefetcht0 [r8 + 2*r9]
        prefetcht0 [r10]
        prefetcht0 [r10 + r9]
        prefetcht0 [r10 + 2*r9]
        .p2align 4
2:
        // A group: the even quads into zmm0..zmm5, the odd into zmm6..zmm11;
        // rax addresses A's rows 3 to 5.
        lea rax, [rsi + 2*rsi]
        lea rax, [rdx + 4*rax]
.irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        vpxord zmm\r, zmm\r, zmm\r
.endr
        mov r11, rsi
        shr r11, 1
        jz 4f
        .p2align 4
3:
        QUAD16 0, 12, \au, 0, 1, 2, 3, 4, 5
        QUAD16 1, 13, \au, 6, 7, 8, 9, 10, 11
        add rdx, 8
        add rax, 8
        add rcx, 128
        dec r11
        jnz 3b
4:
        test esi, 1
        jz 5f
        QUAD16 0, 12, \au, 0, 1, 2, 3, 4, 5
        add rdx, 4
        add rcx, 64
5:
        // rdx to the group's scales and offsets, 20*rsi bytes on.
        lea r11, [rsi + 4*rsi]
        lea rdx, [rdx + 4*r11]
        vpaddd zmm0, zmm0, zmm6
        vpaddd zmm1, zmm1, zmm7
        vpaddd zmm2, zmm2, zmm8
        vpaddd zmm3, zmm3, zmm9
        vpaddd zmm4, zmm4, zmm10
        vpaddd zmm5, zmm5, zmm11
        vmovups zmm16, [rcx]
        vmovups zmm17, [rcx + 64]
        SCALE16 0, r8
        SCALE16 1, r8 + r9
        SCALE16 2, r8 + 2*r9
        SCALE16 3, r10
        SCALE16 4, r10 + r9
        SCALE16 5, r10 + 2*r9
        add rdx, 48
        add rcx, 128
        dec rdi
        jnz 2b
        vzeroupper
        ret
        .size \name, . - \name
.endm

        QGEMM16 sdasm_qgemm_6x16_au_avx512, 1
        QGEMM16 sdasm_qgemm_6x16_bu_avx512, 0

// ---------------------------------------------------------------- 6 by 32

// One quad of k of the tile of 32 columns: the two panels' sixteen columns at
// rcx + 64*\i and rcx + r8 + 64*\i, A's six rows at rdx + 24*\i; the first
// panel's sums in zmm0..zmm5, the second's in zmm6..zmm11.
.macro QUAD32 i, au
        vmovdqu32 zmm12, [rcx + 64*\i]
        vmovdqu32 zmm13, [rcx + r8 + 64*\i]
        ROW32 \i, \au, 0, 0, 6
        ROW32 \i, \au, 1, 1, 7
        ROW32 \i, \au, 2, 2, 8
        ROW32 \i, \au, 3, 3, 9
        ROW32 \i, \au, 4, 4, 10
        ROW32 \i, \au, 5, 5, 11
.endm

// Row \row of A's quad \i by both panels' quads, into zmm\d and zmm\e.
.macro ROW32 i, au, row, d, e
.if \row == 0
        AROW32 \au, \d, \e, [rdx + 4*\i]
.elseif \row == 1
        AROW32 \au, \d, \e, [rdx + 4*rsi + 4*\i]
.elseif \row == 2
        AROW32 \au, \d, \e, [rdx + 8*rsi + 4*\i]
.elseif \row == 3
        AROW32 \au, \d, \e, [rax + 4*\i]
.elseif \row == 4
        AROW32 \au, \d, \e, [rax + 4*rsi + 4*\i]
.else
        AROW32 \au, \d, \e, [rax + 8*rsi + 4*\i]
.endif
.endm

.macro AROW32 au, d, e, at:vararg
.if \au
        vpbroadcastd zmm14, \at
        vpdpbusd zmm\d, zmm14, zmm12
        vpdpbusd zmm\e, zmm14, zmm13
.else
        vpdpbusd zmm\d, zmm12, dword ptr \at{1to16}
        vpdpbusd zmm\e, zmm13, dword ptr \at{1to16}
.endif
.endm

.macro QGEMM32 name, au
        .globl \name
        .type \name, @function
        .p2align 5
\name:
        // The rows of C: r9, r9 + r10, r9 + 2 r10, r11, r11 + r10, r11 + 2 r10.
        mov r10, [rsp + 8]
        lea r11, [r10 + 2*r10]
        add r11, r9
        cmp qword ptr [rsp + 16], 0
        jne 1f
        vpxord zmm0, zmm0, zmm0
        vmovups [r9], zmm0
        vmovups [r9 + 64], zmm0
        vmovups [r9 + r10], zmm0
        vmovups [r9 + r10 + 64], zmm0
        vmovups [r9 + 2*r10], zmm0
        vmovups [r9 + 2*r10 + 64], zmm0
        vmovups [r11], zmm0
        vmovups [r11 + 64], zmm0
        vmovups [r11 + r10], zmm0
        vmovups [r11 + r10 + 64], zmm0
        vmovups [r11 + 2*r10], zmm0
        vmovups [r11 + 2*r10 + 64], zmm0
1:
        .p2align 4
2:
.irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        vpxord zmm\r, zmm\r, zmm\r
.endr
        lea rax, [rsi + 2*rsi]
        lea rax, [rdx + 4*rax]
        mov r11, rsi
        .p2align 4
3:
        QUAD32 0, \au
        add rdx, 4
        add rax, 4
        add rcx, 64
        dec r11
        jnz 3b
        // rdx to the group's scales and offsets, 20*rsi bytes on; r11 to C's
        // rows 3 to 5.
        lea rax, [rsi + 4*rsi]
        lea rdx, [rdx + 4*rax]
        lea r11, [r10 + 2*r10]
        add r11, r9
        // The columns' scales and offsets of the two panels.
        vmovups zmm16, [rcx]
        vmovups zmm17, [rcx + 64]
        vmovups zmm18, [rcx + r8]
        vmovups zmm19, [rcx + r8 + 64]
        SCALE32 0, 6, r9
        SCALE32 1, 7, r9 + r10
        SCALE32 2, 8, r9 + 2*r10
        SCALE32 3, 9, r11
        SCALE32 4, 10, r11 + r10
        SCALE32 5, 11, r11 + 2*r10
        add rdx, 48
        add rcx, 128
        dec rdi
        jnz 2b
        vzeroupper
        ret
        .size \name, . - \name
.endm

// Row \r of the tile of 32 columns, its sums in zmm\r and zmm\h, scaled and
// added to C's row at \row.
.macro SCALE32 r, h, row
        vmulps zmm14, zmm16, dword ptr [rdx + 4*\r]{1to16}
        vmulps zmm15, zmm18, dword ptr [rdx + 4*\r]{1to16}
        vcvtdq2ps zmm\r, zmm\r
        vcvtdq2ps zmm\h, zmm\h
        vfmadd213ps zmm\r, zmm14, [\row]
        vfmadd213ps zmm\h, zmm15, [\row + 64]
        vfnmadd231ps zmm\r, zmm17, dword ptr [rdx + 24 + 4*\r]{1to16}
        vfnmadd231ps zmm\h, zmm19, dword ptr [rdx + 24 + 4*\r]{1to16}
        vmovups [\row], zmm\r
        vmovups [\row + 64], zmm\h
.endm

        QGEMM32 sdasm_qgemm_6x32_au_avx512, 1
        QGEMM32 sdasm_qgemm_6x32_bu_avx512, 0

        .section .note.GNU-stack, "", @progbits
