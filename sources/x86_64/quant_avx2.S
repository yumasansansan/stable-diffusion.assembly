// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_qgemm_6x16_au_avxvnni(int64_t groups, int64_t quads,
//                                      const uint8_t *a, const uint8_t *b,
//                                      float *c, int64_t ldc,
//                                      int64_t accumulate);
//     void sdasm_qgemm_6x16_bu_avxvnni(...);
//
// The micro-kernels of the quantized products (quant.h) for AVX-VNNI, the
// VEX form of vpdpbusd that Alder Lake's cores and those after have without
// AVX-512: each adds to a lane of 32 bits the four products of the bytes of
// a quad of k, one operand's bytes unsigned and the other's signed. The tile
// of six rows by sixteen columns sums a group in twelve registers of integers,
// as the micro-kernel of single precision (gemm_avx2.S) sums a stretch of k in
// twelve of floats: a quad loads the sixteen columns' four bytes of B in two
// vectors and broadcasts each row's four bytes of A -- A's rows lie one after
// another within a group, B's columns quad by quad (quant.h) -- for twelve vpdpbusd --
// 384 products where twelve FMAs make 96. Measured on the i7-1360P, a P-core
// issues vpdpbusd as often as it issues FMAs, four times the products, and an
// E-core half as often, twice the products.
//
// After each group the sums are scaled and added to C, a row at a time:
// c += sa sb sum - oa ob, sa sb in two registers the products of the row's
// scale and the columns', the offsets' product subtracted by an FMA, as
// qgemm_6x16_au_cxx() in quant.cc does in the same order. The two forms differ
// only in which operand is unsigned: au's A, the linear layers' inputs, bu's
// B, the convolutions' and attention's.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// One quad of k: the four bytes of B's sixteen columns at rcx + 64*\i, of A's
// six rows at rdx + 4*\i, rdx + rsi*4 + 4*\i and rdx + rsi*8 + 4*\i, and the
// same at rax, rsi the quads of a group and A's rows 4*rsi bytes apart; \au
// says A is the unsigned operand.
.macro QUAD i, au
        vmovdqu ymm12, [rcx + 64*\i]
        vmovdqu ymm13, [rcx + 64*\i + 32]
        vpbroadcastd ymm14, [rdx + 4*\i]
        vpbroadcastd ymm15, [rdx + 4*rsi + 4*\i]
.if \au
        {vex} vpdpbusd ymm0, ymm14, ymm12
        {vex} vpdpbusd ymm1, ymm14, ymm13
        vpbroadcastd ymm14, [rdx + 8*rsi + 4*\i]
        {vex} vpdpbusd ymm2, ymm15, ymm12
        {vex} vpdpbusd ymm3, ymm15, ymm13
        vpbroadcastd ymm15, [rax + 4*\i]
        {vex} vpdpbusd ymm4, ymm14, ymm12
        {vex} vpdpbusd ymm5, ymm14, ymm13
        vpbroadcastd ymm14, [rax + 4*rsi + 4*\i]
        {vex} vpdpbusd ymm6, ymm15, ymm12
        {vex} vpdpbusd ymm7, ymm15, ymm13
        vpbroadcastd ymm15, [rax + 8*rsi + 4*\i]
        {vex} vpdpbusd ymm8, ymm14, ymm12
        {vex} vpdpbusd ymm9, ymm14, ymm13
        {vex} vpdpbusd ymm10, ymm15, ymm12
        {vex} vpdpbusd ymm11, ymm15, ymm13
.else
        {vex} vpdpbusd ymm0, ymm12, ymm14
        {vex} vpdpbusd ymm1, ymm13, ymm14
        vpbroadcastd ymm14, [rdx + 8*rsi + 4*\i]
        {vex} vpdpbusd ymm2, ymm12, ymm15
        {vex} vpdpbusd ymm3, ymm13, ymm15
        vpbroadcastd ymm15, [rax + 4*\i]
        {vex} vpdpbusd ymm4, ymm12, ymm14
        {vex} vpdpbusd ymm5, ymm13, ymm14
        vpbroadcastd ymm14, [rax + 4*rsi + 4*\i]
        {vex} vpdpbusd ymm6, ymm12, ymm15
        {vex} vpdpbusd ymm7, ymm13, ymm15
        vpbroadcastd ymm15, [rax + 8*rsi + 4*\i]
        {vex} vpdpbusd ymm8, ymm12, ymm14
        {vex} vpdpbusd ymm9, ymm13, ymm14
        {vex} vpdpbusd ymm10, ymm12, ymm15
        {vex} vpdpbusd ymm11, ymm13, ymm15
.endif
.endm

// Row \r of the tile, its sums in ymm\lo and ymm\hi, scaled and added to C's
// row at \row: rdx holds the scales of A's rows and, 24 bytes after, their
// offsets; rcx the scales of B's columns and, 64 bytes after, theirs.
.macro SCALE r, lo, hi, row
        vbroadcastss ymm14, [rdx + 4*\r]
        vmulps ymm12, ymm14, [rcx]
        vmulps ymm13, ymm14, [rcx + 32]
        vcvtdq2ps ymm\lo, ymm\lo
        vcvtdq2ps ymm\hi, ymm\hi
        vfmadd213ps ymm\lo, ymm12, [\row]
        vfmadd213ps ymm\hi, ymm13, [\row + 32]
        vbroadcastss ymm15, [rdx + 24 + 4*\r]
        vfnmadd231ps ymm\lo, ymm15, [rcx + 64]
        vfnmadd231ps ymm\hi, ymm15, [rcx + 96]
        vmovups [\row], ymm\lo
        vmovups [\row + 32], ymm\hi
.endm

.macro QGEMM name, au
        .globl \name
        .type \name, @function
        .p2align 5
\name:
        // r8, r8 + r9 and r8 + 2*r9 address the first three rows of C, r10
        // and the same after it the last three.
        lea r10, [r9 + 2*r9]
        add r10, r8
        vxorps xmm0, xmm0, xmm0
        // A tile not to be added to starts from zeros.
        cmp qword ptr [rsp + 8], 0
        jne 1f
        vmovups [r8], ymm0
        vmovups [r8 + 32], ymm0
        vmovups [r8 + r9], ymm0
        vmovups [r8 + r9 + 32], ymm0
        vmovups [r8 + 2*r9], ymm0
        vmovups [r8 + 2*r9 + 32], ymm0
        vmovups [r10], ymm0
        vmovups [r10 + 32], ymm0
        vmovups [r10 + r9], ymm0
        vmovups [r10 + r9 + 32], ymm0
        vmovups [r10 + 2*r9], ymm0
        vmovups [r10 + 2*r9 + 32], ymm0
1:
        prefetcht0 [r8]
        prefetcht0 [r8 + r9]
        prefetcht0 [r8 + 2*r9]
        prefetcht0 [r10]
        prefetcht0 [r10 + r9]
        prefetcht0 [r10 + 2*r9]
        .p2align 4
2:
        // A group: its sums from zero, two quads at a time, then the last;
        // rax addresses A's rows 3 to 5, 12*rsi bytes after rdx.
        lea rax, [rsi + 2*rsi]
        lea rax, [rdx + 4*rax]
        vpxor xmm0, xmm0, xmm0
        vpxor xmm1, xmm1, xmm1
        vpxor xmm2, xmm2, xmm2
        vpxor xmm3, xmm3, xmm3
        vpxor xmm4, xmm4, xmm4
        vpxor xmm5, xmm5, xmm5
        vpxor xmm6, xmm6, xmm6
        vpxor xmm7, xmm7, xmm7
        vpxor xmm8, xmm8, xmm8
        vpxor xmm9, xmm9, xmm9
        vpxor xmm10, xmm10, xmm10
        vpxor xmm11, xmm11, xmm11
        mov r11, rsi
        shr r11, 1
        jz 4f
        .p2align 4
3:
        QUAD 0, \au
        QUAD 1, \au
        add rdx, 8
        add rax, 8
        add rcx, 128
        dec r11
        jnz 3b
4:
        test esi, 1
        jz 5f
        QUAD 0, \au
        add rdx, 4
        add rcx, 64
5:
        // rdx to the group's scales and offsets, after the six rows: 4*rsi
        // bytes into the first row, 20*rsi before its end.
        lea r11, [rsi + 4*rsi]
        lea rdx, [rdx + 4*r11]
        SCALE 0, 0, 1, r8
        SCALE 1, 2, 3, r8 + r9
        SCALE 2, 4, 5, r8 + 2*r9
        SCALE 3, 6, 7, r10
        SCALE 4, 8, 9, r10 + r9
        SCALE 5, 10, 11, r10 + 2*r9
        add rdx, 48
        add rcx, 128
        dec rdi
        jnz 2b
        vzeroupper
        ret
        .size \name, . - \name
.endm

        QGEMM sdasm_qgemm_6x16_au_avxvnni, 1
        QGEMM sdasm_qgemm_6x16_bu_avxvnni, 0

        .section .note.GNU-stack, "", @progbits
