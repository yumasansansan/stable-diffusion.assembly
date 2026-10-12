// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_qgemm_6x16_au_avx2(int64_t groups, int64_t quads,
//                                   const uint8_t *a, const uint8_t *b,
//                                   float *c, int64_t ldc, int64_t accumulate);
//     void sdasm_qgemm_6x16_bu_avx2(...);
//
// The micro-kernels of quant_avx2.S for AVX2 without VNNI -- Haswell to Coffee
// Lake, Zen to Zen 3 -- for products whose integers are narrow enough: the
// unsigned operand's of input_bits and the signed operand's of bits such that
// two of their products sum to no more than a word holds, 2 (2^input_bits - 1)
// (2^(bits - 1) - 1) <= 32767 -- seven bits of the weights with eight of the
// inputs, or eight of the weights with seven. vpmaddubsw multiplies each
// unsigned byte by its signed byte and sums the pairs into words, which
// saturate past 32767 and so never do here; vpmaddwd by ones sums the pairs of
// words into the 32-bit lanes vpdpbusd would have summed the quad into; vpaddd
// adds them to the sums. The sums are the integers vpdpbusd makes, exactly,
// and the outputs those of AVX-VNNI's kernels and the C++ ones to the bit.
//
// Three instructions for each of AVX-VNNI's one, of which the two
// multiplications take the vector units' multipliers: 32 products of bytes
// for each pair, where an FMA makes eight of single precision -- twice the
// products a cycle on a core that issues two multiplications of integers a
// cycle, as two FMAs (Skylake's cores and those after, Zen 3), and as many on
// one that issues one (Haswell, Broadwell, Zen 2), which gains nothing. On a
// P-core of the i7-1360P the kernel makes 1.55 times the products of
// gemm_avx2.S's, an E-core's as many. A quad loads B's sixteen columns' four
// bytes in two vectors, as quant_avx2.S does, and each row's four bytes of A
// broadcast, and the twelve sums take twelve registers, the products one, the
// broadcast one and B two: the ones that vpmaddwd multiplies by are read from
// memory.
//
// The scaling after each group is quant_avx2.S's, in the same order.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 5
.Lones:
        .rept 16
        .short 1
        .endr

        .text

// The products of row quad \at (A's four bytes) and B's sixteen columns in
// ymm12 and ymm13 into the sums ymm\lo and ymm\hi; \au says A is the unsigned
// operand.
.macro ROW au, lo, hi, at:vararg
        vpbroadcastd ymm14, \at
.if \au
        vpmaddubsw ymm15, ymm14, ymm12
        vpmaddwd ymm15, ymm15, [rip + .Lones]
        vpaddd ymm\lo, ymm\lo, ymm15
        vpmaddubsw ymm15, ymm14, ymm13
.else
        vpmaddubsw ymm15, ymm12, ymm14
        vpmaddwd ymm15, ymm15, [rip + .Lones]
        vpaddd ymm\lo, ymm\lo, ymm15
        vpmaddubsw ymm15, ymm13, ymm14
.endif
        vpmaddwd ymm15, ymm15, [rip + .Lones]
        vpaddd ymm\hi, ymm\hi, ymm15
.endm

// One quad of k: the four bytes of B's sixteen columns at rcx + 64*\i, of A's
// six rows at rdx + 4*\i, rdx + rsi*4 + 4*\i and rdx + rsi*8 + 4*\i, and the
// same at rax, rsi the quads of a group and A's rows 4*rsi bytes apart.
.macro QUAD i, au
        vmovdqu ymm12, [rcx + 64*\i]
        vmovdqu ymm13, [rcx + 64*\i + 32]
        ROW \au, 0, 1, [rdx + 4*\i]
        ROW \au, 2, 3, [rdx + 4*rsi + 4*\i]
        ROW \au, 4, 5, [rdx + 8*rsi + 4*\i]
        ROW \au, 6, 7, [rax + 4*\i]
        ROW \au, 8, 9, [rax + 4*rsi + 4*\i]
        ROW \au, 10, 11, [rax + 8*rsi + 4*\i]
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
// rdi groups, rsi quads, rdx a, rcx b, r8 c, r9 ldc, [rsp + 8] accumulate
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

        QGEMM sdasm_qgemm_6x16_au_avx2, 1
        QGEMM sdasm_qgemm_6x16_bu_avx2, 0

        .section .note.GNU-stack, "", @progbits
