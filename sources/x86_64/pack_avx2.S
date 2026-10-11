// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_pack16_f32_avx2(float *dst, const float *src, int64_t stride,
//                                int64_t kc);
//     void sdasm_pack16_f16_avx2(float *dst, const uint16_t *src,
//                                int64_t stride, int64_t kc);
//     void sdasm_pack6_f32_avx2(float *dst, const float *src, int64_t stride,
//                               int64_t kc);
//     void sdasm_pack6_f16_avx2(float *dst, const uint16_t *src,
//                               int64_t stride, int64_t kc);
//
// A panel of the matrix product (gemm.h) packed from rows of a matrix: sixteen
// rows, or six, each of kc numbers, kc a multiple of eight, the rows stride
// bytes apart, into [kc][16] or [kc][6] at dst -- the numbers of the rows side
// by side for each k. The FP16 forms widen the numbers to single precision on
// the way, by F16C.
//
// The packing is a transposition, eight rows by eight numbers at a time: the
// eight vectors of the rows' numbers, unpacked two by two, shuffled four by
// four and their halves exchanged, are the eight vectors of the numbers of each
// k. A panel of sixteen is two such columns of eight rows; a panel of six is one
// of eight with the last two rows zeros, of which the first six lanes of each
// vector are stored. The C++ form (gemm.cc) is a loop over the rows and the
// numbers, which the compiler turns into scattered stores. Measured on the
// i7-1360P, the rows in the first-level cache: on a P-core, a panel of sixteen
// is packed 3 to 3.8 times as fast as by the C++ form, one of six 2.4 times;
// on an E-core, 8 and 3 times.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// ymm0..ymm7 = the transposition of ymm0..ymm7, ymm8..ymm15 overwritten.
.macro TRANSPOSE8
        vunpcklps ymm8, ymm0, ymm1
        vunpckhps ymm9, ymm0, ymm1
        vunpcklps ymm10, ymm2, ymm3
        vunpckhps ymm11, ymm2, ymm3
        vunpcklps ymm12, ymm4, ymm5
        vunpckhps ymm13, ymm4, ymm5
        vunpcklps ymm14, ymm6, ymm7
        vunpckhps ymm15, ymm6, ymm7
        vshufps ymm0, ymm8, ymm10, 0x44
        vshufps ymm1, ymm8, ymm10, 0xee
        vshufps ymm2, ymm9, ymm11, 0x44
        vshufps ymm3, ymm9, ymm11, 0xee
        vshufps ymm4, ymm12, ymm14, 0x44
        vshufps ymm5, ymm12, ymm14, 0xee
        vshufps ymm6, ymm13, ymm15, 0x44
        vshufps ymm7, ymm13, ymm15, 0xee
        vperm2f128 ymm8, ymm0, ymm4, 0x20
        vperm2f128 ymm9, ymm1, ymm5, 0x20
        vperm2f128 ymm10, ymm2, ymm6, 0x20
        vperm2f128 ymm11, ymm3, ymm7, 0x20
        vperm2f128 ymm12, ymm0, ymm4, 0x31
        vperm2f128 ymm13, ymm1, ymm5, 0x31
        vperm2f128 ymm14, ymm2, ymm6, 0x31
        vperm2f128 ymm15, ymm3, ymm7, 0x31
.endm

// Loads of eight numbers of a row into ymm\v: single precision as they are, or
// FP16 widened.
.macro LOAD_F32 v, address
        vmovups ymm\v, \address
.endm
.macro LOAD_F16 v, address
        vcvtph2ps ymm\v, \address
.endm

// The eight rows from rax, rax + rcx, ..., rax + 7 rcx (r9 = 3 rcx, r10 =
// 5 rcx, r11 = 7 rcx), each at byte offset \offset, into ymm0..ymm7.
.macro LOAD8 load, offset
        \load 0, [rax + \offset]
        \load 1, [rax + rcx + \offset]
        \load 2, [rax + 2*rcx + \offset]
        \load 3, [rax + r9 + \offset]
        \load 4, [rax + 4*rcx + \offset]
        \load 5, [rax + r10 + \offset]
        \load 6, [rax + 2*r9 + \offset]
        \load 7, [rax + r11 + \offset]
.endm

// The transposed vectors (ymm8..ymm15, k = 0..7) stored eight floats apart in
// a panel of \p floats a k, at [rdi + \column * 4].
.macro STORE8 p, column
        vmovups [rdi + 4*(0 * (\p) + (\column))], ymm8
        vmovups [rdi + 4*(1 * (\p) + (\column))], ymm9
        vmovups [rdi + 4*(2 * (\p) + (\column))], ymm10
        vmovups [rdi + 4*(3 * (\p) + (\column))], ymm11
        vmovups [rdi + 4*(4 * (\p) + (\column))], ymm12
        vmovups [rdi + 4*(5 * (\p) + (\column))], ymm13
        vmovups [rdi + 4*(6 * (\p) + (\column))], ymm14
        vmovups [rdi + 4*(7 * (\p) + (\column))], ymm15
.endm

// The first six lanes of each transposed vector, for a panel of six.
.macro STORE6_ONE v, k
        vmovups [rdi + 24 * (\k)], xmm\v
        vextractf128 xmm0, ymm\v, 1
        vmovlps [rdi + 24 * (\k) + 16], xmm0
.endm
.macro STORE6
        STORE6_ONE 8, 0
        STORE6_ONE 9, 1
        STORE6_ONE 10, 2
        STORE6_ONE 11, 3
        STORE6_ONE 12, 4
        STORE6_ONE 13, 5
        STORE6_ONE 14, 6
        STORE6_ONE 15, 7
.endm

// rdi dst, rsi src, rdx stride, rcx kc; the size of a number, \size.
.macro PACK16 name, load, size
        .globl \name
        .type \name, @function
        .p2align 5
\name:
        shr rcx, 3                      // blocks of eight numbers
        jz 9f
        mov r8, rcx
        mov rcx, rdx                    // the stride, for the addressing
        lea r9, [rcx + 2*rcx]
        lea r10, [rcx + 4*rcx]
        lea r11, [r9 + 4*rcx]
1:
        mov rax, rsi
        LOAD8 \load, 0
        TRANSPOSE8
        STORE8 16, 0
        lea rax, [rsi + 8*rcx]          // rows 8..15
        LOAD8 \load, 0
        TRANSPOSE8
        STORE8 16, 8
        add rsi, 8 * \size
        add rdi, 4 * 8 * 16
        dec r8
        jnz 1b
9:
        vzeroupper
        ret
        .size \name, .-\name
.endm

.macro PACK6 name, load, size
        .globl \name
        .type \name, @function
        .p2align 5
\name:
        shr rcx, 3
        jz 9f
        mov r8, rcx
        mov rcx, rdx
        lea r9, [rcx + 2*rcx]
        lea r10, [rcx + 4*rcx]
1:
        mov rax, rsi
        \load 0, [rax]
        \load 1, [rax + rcx]
        \load 2, [rax + 2*rcx]
        \load 3, [rax + r9]
        \load 4, [rax + 4*rcx]
        \load 5, [rax + r10]
        vxorps xmm6, xmm6, xmm6
        vxorps xmm7, xmm7, xmm7
        TRANSPOSE8
        STORE6
        add rsi, 8 * \size
        add rdi, 4 * 8 * 6
        dec r8
        jnz 1b
9:
        vzeroupper
        ret
        .size \name, .-\name
.endm

        PACK16 sdasm_pack16_f32_avx2, LOAD_F32, 4
        PACK16 sdasm_pack16_f16_avx2, LOAD_F16, 2
        PACK6 sdasm_pack6_f32_avx2, LOAD_F32, 4
        PACK6 sdasm_pack6_f16_avx2, LOAD_F16, 2

        .section .note.GNU-stack, "", @progbits
