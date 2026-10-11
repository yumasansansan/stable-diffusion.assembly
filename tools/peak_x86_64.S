// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_peak_avx2(int64_t n, const float *seed);
//     void sdasm_peak_avx512(int64_t n, const float *seed);
//
// The most multiplications and additions a core makes, for sdasm_bench --peak
// (bench.cc), against which the rate of an operation is a share: n times,
// twelve FMAs of eight lanes for AVX2, twenty-four of sixteen for AVX-512, each
// adding to a sum of its own -- more sums than the latency of an FMA times the
// FMAs a cycle -- the product of two registers of numbers from `seed`, 64
// floats unlike each other. The sums grow, and their bits change, as those of
// a product's sums do: a core multiplying zeros, or numbers that do not
// change, draws less power and may run faster than it ever does on numbers.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

        .globl sdasm_peak_avx2
        .type sdasm_peak_avx2, @function
        .p2align 5
// rdi n, rsi seed
sdasm_peak_avx2:
        vmovups ymm12, [rsi]
        vmovups ymm13, [rsi + 32]
        vmovups ymm14, [rsi + 64]
        vmovups ymm15, [rsi + 96]
        vmovaps ymm0, ymm13
        vmovaps ymm1, ymm14
        vmovaps ymm2, ymm15
        vmovaps ymm3, ymm13
        vmovaps ymm4, ymm14
        vmovaps ymm5, ymm15
        vmovaps ymm6, ymm13
        vmovaps ymm7, ymm14
        vmovaps ymm8, ymm15
        vmovaps ymm9, ymm13
        vmovaps ymm10, ymm14
        vmovaps ymm11, ymm15
        test rdi, rdi
        jle 9f
        .p2align 5
1:
        vfmadd231ps ymm0, ymm12, ymm13
        vfmadd231ps ymm1, ymm12, ymm14
        vfmadd231ps ymm2, ymm12, ymm15
        vfmadd231ps ymm3, ymm12, ymm13
        vfmadd231ps ymm4, ymm12, ymm14
        vfmadd231ps ymm5, ymm12, ymm15
        vfmadd231ps ymm6, ymm12, ymm13
        vfmadd231ps ymm7, ymm12, ymm14
        vfmadd231ps ymm8, ymm12, ymm15
        vfmadd231ps ymm9, ymm12, ymm13
        vfmadd231ps ymm10, ymm12, ymm14
        vfmadd231ps ymm11, ymm12, ymm15
        dec rdi
        jnz 1b
9:
        vzeroupper
        ret
        .size sdasm_peak_avx2, .-sdasm_peak_avx2

        .globl sdasm_peak_avx512
        .type sdasm_peak_avx512, @function
        .p2align 5
// rdi n, rsi seed
sdasm_peak_avx512:
        vmovups zmm24, [rsi]
        vmovups zmm25, [rsi + 64]
        vmovups zmm26, [rsi + 128]
        vmovups zmm27, [rsi + 192]
        vpermilps zmm28, zmm25, 0x1b
        vpermilps zmm29, zmm26, 0x1b
        vpermilps zmm30, zmm27, 0x1b
        vmovaps zmm0, zmm25
        vmovaps zmm1, zmm26
        vmovaps zmm2, zmm27
        vmovaps zmm3, zmm28
        vmovaps zmm4, zmm29
        vmovaps zmm5, zmm30
        vmovaps zmm6, zmm25
        vmovaps zmm7, zmm26
        vmovaps zmm8, zmm27
        vmovaps zmm9, zmm28
        vmovaps zmm10, zmm29
        vmovaps zmm11, zmm30
        vmovaps zmm12, zmm25
        vmovaps zmm13, zmm26
        vmovaps zmm14, zmm27
        vmovaps zmm15, zmm28
        vmovaps zmm16, zmm29
        vmovaps zmm17, zmm30
        vmovaps zmm18, zmm25
        vmovaps zmm19, zmm26
        vmovaps zmm20, zmm27
        vmovaps zmm21, zmm28
        vmovaps zmm22, zmm29
        vmovaps zmm23, zmm30
        test rdi, rdi
        jle 9f
        .p2align 5
1:
        vfmadd231ps zmm0, zmm24, zmm25
        vfmadd231ps zmm1, zmm24, zmm26
        vfmadd231ps zmm2, zmm24, zmm27
        vfmadd231ps zmm3, zmm24, zmm28
        vfmadd231ps zmm4, zmm24, zmm29
        vfmadd231ps zmm5, zmm24, zmm30
        vfmadd231ps zmm6, zmm24, zmm25
        vfmadd231ps zmm7, zmm24, zmm26
        vfmadd231ps zmm8, zmm24, zmm27
        vfmadd231ps zmm9, zmm24, zmm28
        vfmadd231ps zmm10, zmm24, zmm29
        vfmadd231ps zmm11, zmm24, zmm30
        vfmadd231ps zmm12, zmm24, zmm25
        vfmadd231ps zmm13, zmm24, zmm26
        vfmadd231ps zmm14, zmm24, zmm27
        vfmadd231ps zmm15, zmm24, zmm28
        vfmadd231ps zmm16, zmm24, zmm29
        vfmadd231ps zmm17, zmm24, zmm30
        vfmadd231ps zmm18, zmm24, zmm25
        vfmadd231ps zmm19, zmm24, zmm26
        vfmadd231ps zmm20, zmm24, zmm27
        vfmadd231ps zmm21, zmm24, zmm28
        vfmadd231ps zmm22, zmm24, zmm29
        vfmadd231ps zmm23, zmm24, zmm30
        dec rdi
        jnz 1b
9:
        vzeroupper
        ret
        .size sdasm_peak_avx512, .-sdasm_peak_avx512

        .section .note.GNU-stack, "", @progbits
