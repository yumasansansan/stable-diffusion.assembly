// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_bf16_2x2_avx512(int64_t blocks, const uint16_t *a,
//                                int64_t a_next, const uint16_t *b,
//                                int64_t b_next, float *c, int64_t ldc,
//                                int64_t accumulate);
//
// What sdasm_amx_bf16_2x2 computes (amx.S), from the same tiles, on the vector
// units of AVX512-BF16 rather than on AMX's tiles: a block of 32 by 32 of C,
// each product x1 w1 + x1 w2 + x2 w1 of numbers split in two parts of
// bfloat16. vdpbf16ps adds the products of a pair of k, sixteen columns of B
// against the pair of a row of A broadcast from memory ({1to16}), to each sum
// in single precision, rounding to the nearest after each.
//
// This is a form for measurement (Tuning::bf16): the instruction makes 32
// products to the FMA's sixteen, but the parts make three products of each one,
// half again as many instructions as the FMAs of single precision, which
// llvm-mca's models of Cooper Lake, Sapphire Rapids, Zen 4 and Zen 5 run no
// faster than those FMAs; the processors themselves are measured on GitHub's
// runners (.github/workflows/measure.yml). AMX's tiles make sixteen times the
// products of the FMAs.
// The block is computed eight rows at a time, by the sixteen sums of eight rows
// by 32 columns, two registers to a row: per pair of k, four rows of B, and
// six instructions for each row of A.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

        .globl sdasm_bf16_2x2_avx512
        .type sdasm_bf16_2x2_avx512, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_bf16_2x2_avx512:
        push rbx
        push rbp
        push r12
        push r13
        push r14
        mov r11, [rsp + 48]                     // ldc
        mov r12, [rsp + 56]                     // accumulate
        mov r13, r9                             // the rows of C of this eight
        mov r14, rdx                            // a_next
        mov rbx, rsi                            // A's first tile of rows 0 to 15
        mov rbp, rcx                            // B's tile of columns 0 to 15
        xor esi, esi                            // the eight, 0 to 3
1:
        // r10: the first row of A of this eight, in the tiles of the first block.
        mov r10, rsi
        and r10, 1
        shl r10, 9                              // rows 8 to 15 of a tile: 512 bytes on
        add r10, rbx
        cmp rsi, 2
        jb 2f
        add r10, r14                            // rows 16 to 31: the next tiles
2:
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
        vpxord zmm12, zmm12, zmm12
        vpxord zmm13, zmm13, zmm13
        vpxord zmm14, zmm14, zmm14
        vpxord zmm15, zmm15, zmm15
        mov rdx, rbp
        lea r9, [rbp + r8]
        mov rcx, rdi
        test rcx, rcx
        jle 5f
3:
        vmovups zmm16, [rdx + 64 * 0]
        vmovups zmm17, [r9 + 64 * 0]
        vmovups zmm18, [rdx + 64 * 0 + 1024]
        vmovups zmm19, [r9 + 64 * 0 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 0 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 0 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 1]
        vmovups zmm17, [r9 + 64 * 1]
        vmovups zmm18, [rdx + 64 * 1 + 1024]
        vmovups zmm19, [r9 + 64 * 1 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 1 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 1 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 2]
        vmovups zmm17, [r9 + 64 * 2]
        vmovups zmm18, [rdx + 64 * 2 + 1024]
        vmovups zmm19, [r9 + 64 * 2 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 2 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 2 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 3]
        vmovups zmm17, [r9 + 64 * 3]
        vmovups zmm18, [rdx + 64 * 3 + 1024]
        vmovups zmm19, [r9 + 64 * 3 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 3 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 3 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 4]
        vmovups zmm17, [r9 + 64 * 4]
        vmovups zmm18, [rdx + 64 * 4 + 1024]
        vmovups zmm19, [r9 + 64 * 4 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 4 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 4 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 5]
        vmovups zmm17, [r9 + 64 * 5]
        vmovups zmm18, [rdx + 64 * 5 + 1024]
        vmovups zmm19, [r9 + 64 * 5 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 5 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 5 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 6]
        vmovups zmm17, [r9 + 64 * 6]
        vmovups zmm18, [rdx + 64 * 6 + 1024]
        vmovups zmm19, [r9 + 64 * 6 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 6 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 6 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 7]
        vmovups zmm17, [r9 + 64 * 7]
        vmovups zmm18, [rdx + 64 * 7 + 1024]
        vmovups zmm19, [r9 + 64 * 7 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 7 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 7 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 8]
        vmovups zmm17, [r9 + 64 * 8]
        vmovups zmm18, [rdx + 64 * 8 + 1024]
        vmovups zmm19, [r9 + 64 * 8 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 8 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 8 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 9]
        vmovups zmm17, [r9 + 64 * 9]
        vmovups zmm18, [rdx + 64 * 9 + 1024]
        vmovups zmm19, [r9 + 64 * 9 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 9 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 9 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 10]
        vmovups zmm17, [r9 + 64 * 10]
        vmovups zmm18, [rdx + 64 * 10 + 1024]
        vmovups zmm19, [r9 + 64 * 10 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 10 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 10 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 11]
        vmovups zmm17, [r9 + 64 * 11]
        vmovups zmm18, [rdx + 64 * 11 + 1024]
        vmovups zmm19, [r9 + 64 * 11 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 11 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 11 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 12]
        vmovups zmm17, [r9 + 64 * 12]
        vmovups zmm18, [rdx + 64 * 12 + 1024]
        vmovups zmm19, [r9 + 64 * 12 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 12 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 12 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 13]
        vmovups zmm17, [r9 + 64 * 13]
        vmovups zmm18, [rdx + 64 * 13 + 1024]
        vmovups zmm19, [r9 + 64 * 13 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 13 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 13 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 14]
        vmovups zmm17, [r9 + 64 * 14]
        vmovups zmm18, [rdx + 64 * 14 + 1024]
        vmovups zmm19, [r9 + 64 * 14 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 14 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 14 + 1024]{1to16}
        vmovups zmm16, [rdx + 64 * 15]
        vmovups zmm17, [r9 + 64 * 15]
        vmovups zmm18, [rdx + 64 * 15 + 1024]
        vmovups zmm19, [r9 + 64 * 15 + 1024]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm0, zmm18, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm1, zmm19, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm2, zmm18, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm3, zmm19, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm4, zmm18, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm5, zmm19, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm6, zmm18, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm7, zmm19, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm8, zmm18, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm9, zmm19, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm10, zmm18, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm11, zmm19, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm12, zmm18, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm13, zmm19, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpbf16ps zmm14, zmm18, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpbf16ps zmm15, zmm19, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 15 + 1024]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 15 + 1024]{1to16}
        add r10, 2048
        add rdx, 2048
        add r9, 2048
        dec rcx
        jnz 3b
5:
        mov rax, r13
        test r12, r12
        jz 6f
        vaddps zmm0, zmm0, [rax]
        vaddps zmm1, zmm1, [rax + 64]
        add rax, r11
        vaddps zmm2, zmm2, [rax]
        vaddps zmm3, zmm3, [rax + 64]
        add rax, r11
        vaddps zmm4, zmm4, [rax]
        vaddps zmm5, zmm5, [rax + 64]
        add rax, r11
        vaddps zmm6, zmm6, [rax]
        vaddps zmm7, zmm7, [rax + 64]
        add rax, r11
        vaddps zmm8, zmm8, [rax]
        vaddps zmm9, zmm9, [rax + 64]
        add rax, r11
        vaddps zmm10, zmm10, [rax]
        vaddps zmm11, zmm11, [rax + 64]
        add rax, r11
        vaddps zmm12, zmm12, [rax]
        vaddps zmm13, zmm13, [rax + 64]
        add rax, r11
        vaddps zmm14, zmm14, [rax]
        vaddps zmm15, zmm15, [rax + 64]
        add rax, r11
        mov rax, r13
6:
        vmovups [rax], zmm0
        vmovups [rax + 64], zmm1
        add rax, r11
        vmovups [rax], zmm2
        vmovups [rax + 64], zmm3
        add rax, r11
        vmovups [rax], zmm4
        vmovups [rax + 64], zmm5
        add rax, r11
        vmovups [rax], zmm6
        vmovups [rax + 64], zmm7
        add rax, r11
        vmovups [rax], zmm8
        vmovups [rax + 64], zmm9
        add rax, r11
        vmovups [rax], zmm10
        vmovups [rax + 64], zmm11
        add rax, r11
        vmovups [rax], zmm12
        vmovups [rax + 64], zmm13
        add rax, r11
        vmovups [rax], zmm14
        vmovups [rax + 64], zmm15
        add rax, r11
        mov r13, rax
        inc rsi
        cmp rsi, 4
        jb 1b
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        vzeroupper
        ret
        .size sdasm_bf16_2x2_avx512, .-sdasm_bf16_2x2_avx512

        .section .note.GNU-stack, "", @progbits
