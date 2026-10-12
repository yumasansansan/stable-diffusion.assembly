// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The products of numbers of 16 bits, one product for each one, for the run
// that gives up precision for speed (Tuning::bits = 16, gemm.h): the kernels
// of amx.S, amx_fp16.S, amx_single.S and bf16_avx512.S that multiply the first
// parts alone of the tiles those files' packing writes -- x1, each number
// rounded to the nearest bfloat16 or FP16 -- and leave the others out.
//
//     void sdasm_amx_bf16_1x1(int64_t blocks, const uint16_t *a, int64_t a_next,
//                             const uint16_t *b, int64_t b_next, float *c,
//                             int64_t ldc, int64_t accumulate);
//
// A block of 32 by 32 of C from the tiles of amx.S, x1 w1 of bfloat16: four
// loads to four products a block of k, where sdasm_amx_bf16_2x2 makes ten to
// twelve. A product is within 2^-8 of x w.
//
//     void sdasm_amx_fp16_1x1_a2(...);
//     void sdasm_amx_fp16_1x1_b2(...);
//
// The same from the tiles of amx_fp16.S, the FP16 weights as they are and the
// other operand's x1 of FP16, A's (a2) or B's (b2): w x1, each product exact,
// which is ggml's own product, the image rounded to FP16 -- as precise as
// stable-diffusion.cpp, and a product for each one.
//
//     void sdasm_amx_single_1x1(...);
//
// The same from the tiles of amx_single.S, of three parts, 3072 bytes a block:
// attention's x1 y1 of bfloat16.
//
//     void sdasm_bf16_1x1_avx512(...);
//     void sdasm_bf16_single_1x1_avx512(...);
//
// What sdasm_amx_bf16_1x1 and sdasm_amx_single_1x1 compute, on the vector
// units of AVX512-BF16, as bf16_avx512.S computes the products of two parts:
// vdpbf16ps adds the products of a pair of k, sixteen columns of B against the
// pair of a row of A broadcast from memory ({1to16}), to sums of single
// precision. The block goes eight rows at a time, the sixteen sums of eight
// rows by 32 columns, two registers a row; a pair of k loads two rows of B for
// sixteen vdpbf16ps -- 512 products, where sixteen FMAs of single precision
// make 256.
//
//     void sdasm_avx10_fp16_1x1_a2(...);
//     void sdasm_avx10_fp16_1x1_b2(...);
//
// What sdasm_amx_fp16_1x1_a2 and _b2 compute, on the vector units of AVX10.2,
// from the same tiles: vdpphps adds the products of a pair of k of FP16,
// sixteen columns of B against the pair of a row of A broadcast from memory,
// to sums of single precision, each product exact -- eight rows at a time by
// 32 columns, as sdasm_bf16_1x1_avx512 goes.
//
//     void sdasm_fp16_1x1_a2_avx512(...);
//     void sdasm_fp16_1x1_b2_avx512(...);
//
// The same on AVX512-FP16, which has no such sum: vfmadd231ph multiplies and
// adds in FP16, the pair of a row of A broadcast to the 32 lanes and each lane
// of B one number of a column's pair, four rows at a time by 32 columns. A
// lane's sum of FP16 goes into single precision every two blocks of k, after
// 32 products, before its rounding grows, and the two lanes of each column's
// pair are added at the end.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// ---------------------------------------------------------------- on the tiles

        .globl sdasm_amx_bf16_1x1
        .type sdasm_amx_bf16_1x1, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_bf16_1x1:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9                             // rows 16 to 31 of C
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
        tileloadd tmm4, [rsi + rax * 1]
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        add rsi, 2048
        add rdx, 2048
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
        .size sdasm_amx_bf16_1x1, .-sdasm_amx_bf16_1x1

        .globl sdasm_amx_fp16_1x1_a2
        .type sdasm_amx_fp16_1x1_a2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_fp16_1x1_a2:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9                             // rows 16 to 31 of C
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
        tileloadd tmm4, [rsi + rax * 1]
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
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
        .size sdasm_amx_fp16_1x1_a2, .-sdasm_amx_fp16_1x1_a2

        .globl sdasm_amx_fp16_1x1_b2
        .type sdasm_amx_fp16_1x1_b2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_fp16_1x1_b2:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9                             // rows 16 to 31 of C
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
        tileloadd tmm4, [rsi + rax * 1]
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
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
        .size sdasm_amx_fp16_1x1_b2, .-sdasm_amx_fp16_1x1_b2

        .globl sdasm_amx_single_1x1
        .type sdasm_amx_single_1x1, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_single_1x1:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9                             // rows 16 to 31 of C
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
        tileloadd tmm4, [rsi + rax * 1]
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        add rsi, 3072
        add rdx, 3072
        add rcx, 3072
        add r8, 3072
        dec rdi
        jnz 3b
8:
        tilestored [r9 + r10 * 1], tmm0
        tilestored [r9 + r10 * 1 + 64], tmm1
        tilestored [r11 + r10 * 1], tmm2
        tilestored [r11 + r10 * 1 + 64], tmm3
        ret
        .size sdasm_amx_single_1x1, .-sdasm_amx_single_1x1

// ---------------------------------------------------------------- on the vector units

        .globl sdasm_bf16_1x1_avx512
        .type sdasm_bf16_1x1_avx512, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_bf16_1x1_avx512:
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
        .p2align 4
3:
        vmovups zmm16, [rdx + 64 * 0]
        vmovups zmm17, [r9 + 64 * 0]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vmovups zmm16, [rdx + 64 * 1]
        vmovups zmm17, [r9 + 64 * 1]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vmovups zmm16, [rdx + 64 * 2]
        vmovups zmm17, [r9 + 64 * 2]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vmovups zmm16, [rdx + 64 * 3]
        vmovups zmm17, [r9 + 64 * 3]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vmovups zmm16, [rdx + 64 * 4]
        vmovups zmm17, [r9 + 64 * 4]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vmovups zmm16, [rdx + 64 * 5]
        vmovups zmm17, [r9 + 64 * 5]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vmovups zmm16, [rdx + 64 * 6]
        vmovups zmm17, [r9 + 64 * 6]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vmovups zmm16, [rdx + 64 * 7]
        vmovups zmm17, [r9 + 64 * 7]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vmovups zmm16, [rdx + 64 * 8]
        vmovups zmm17, [r9 + 64 * 8]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vmovups zmm16, [rdx + 64 * 9]
        vmovups zmm17, [r9 + 64 * 9]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vmovups zmm16, [rdx + 64 * 10]
        vmovups zmm17, [r9 + 64 * 10]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vmovups zmm16, [rdx + 64 * 11]
        vmovups zmm17, [r9 + 64 * 11]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vmovups zmm16, [rdx + 64 * 12]
        vmovups zmm17, [r9 + 64 * 12]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vmovups zmm16, [rdx + 64 * 13]
        vmovups zmm17, [r9 + 64 * 13]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vmovups zmm16, [rdx + 64 * 14]
        vmovups zmm17, [r9 + 64 * 14]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vmovups zmm16, [rdx + 64 * 15]
        vmovups zmm17, [r9 + 64 * 15]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
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
        .size sdasm_bf16_1x1_avx512, .-sdasm_bf16_1x1_avx512

        .globl sdasm_bf16_single_1x1_avx512
        .type sdasm_bf16_single_1x1_avx512, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_bf16_single_1x1_avx512:
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
        .p2align 4
3:
        vmovups zmm16, [rdx + 64 * 0]
        vmovups zmm17, [r9 + 64 * 0]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vmovups zmm16, [rdx + 64 * 1]
        vmovups zmm17, [r9 + 64 * 1]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vmovups zmm16, [rdx + 64 * 2]
        vmovups zmm17, [r9 + 64 * 2]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vmovups zmm16, [rdx + 64 * 3]
        vmovups zmm17, [r9 + 64 * 3]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vmovups zmm16, [rdx + 64 * 4]
        vmovups zmm17, [r9 + 64 * 4]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vmovups zmm16, [rdx + 64 * 5]
        vmovups zmm17, [r9 + 64 * 5]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vmovups zmm16, [rdx + 64 * 6]
        vmovups zmm17, [r9 + 64 * 6]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vmovups zmm16, [rdx + 64 * 7]
        vmovups zmm17, [r9 + 64 * 7]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vmovups zmm16, [rdx + 64 * 8]
        vmovups zmm17, [r9 + 64 * 8]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vmovups zmm16, [rdx + 64 * 9]
        vmovups zmm17, [r9 + 64 * 9]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vmovups zmm16, [rdx + 64 * 10]
        vmovups zmm17, [r9 + 64 * 10]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vmovups zmm16, [rdx + 64 * 11]
        vmovups zmm17, [r9 + 64 * 11]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vmovups zmm16, [rdx + 64 * 12]
        vmovups zmm17, [r9 + 64 * 12]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vmovups zmm16, [rdx + 64 * 13]
        vmovups zmm17, [r9 + 64 * 13]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vmovups zmm16, [rdx + 64 * 14]
        vmovups zmm17, [r9 + 64 * 14]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vmovups zmm16, [rdx + 64 * 15]
        vmovups zmm17, [r9 + 64 * 15]
        vdpbf16ps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpbf16ps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpbf16ps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpbf16ps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpbf16ps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpbf16ps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpbf16ps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpbf16ps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpbf16ps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        add r10, 3072
        add rdx, 3072
        add r9, 3072
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
        .size sdasm_bf16_single_1x1_avx512, .-sdasm_bf16_single_1x1_avx512

// ---------------------------------------------------------------- AVX10.2

        .globl sdasm_avx10_fp16_1x1_a2
        .type sdasm_avx10_fp16_1x1_a2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_avx10_fp16_1x1_a2:
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
        .p2align 4
3:
        vmovups zmm16, [rdx + 64 * 0]
        vmovups zmm17, [r9 + 64 * 0]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vmovups zmm16, [rdx + 64 * 1]
        vmovups zmm17, [r9 + 64 * 1]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vmovups zmm16, [rdx + 64 * 2]
        vmovups zmm17, [r9 + 64 * 2]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vmovups zmm16, [rdx + 64 * 3]
        vmovups zmm17, [r9 + 64 * 3]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vmovups zmm16, [rdx + 64 * 4]
        vmovups zmm17, [r9 + 64 * 4]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vmovups zmm16, [rdx + 64 * 5]
        vmovups zmm17, [r9 + 64 * 5]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vmovups zmm16, [rdx + 64 * 6]
        vmovups zmm17, [r9 + 64 * 6]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vmovups zmm16, [rdx + 64 * 7]
        vmovups zmm17, [r9 + 64 * 7]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vmovups zmm16, [rdx + 64 * 8]
        vmovups zmm17, [r9 + 64 * 8]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vmovups zmm16, [rdx + 64 * 9]
        vmovups zmm17, [r9 + 64 * 9]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vmovups zmm16, [rdx + 64 * 10]
        vmovups zmm17, [r9 + 64 * 10]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vmovups zmm16, [rdx + 64 * 11]
        vmovups zmm17, [r9 + 64 * 11]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vmovups zmm16, [rdx + 64 * 12]
        vmovups zmm17, [r9 + 64 * 12]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vmovups zmm16, [rdx + 64 * 13]
        vmovups zmm17, [r9 + 64 * 13]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vmovups zmm16, [rdx + 64 * 14]
        vmovups zmm17, [r9 + 64 * 14]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vmovups zmm16, [rdx + 64 * 15]
        vmovups zmm17, [r9 + 64 * 15]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        add r10, 2048
        add rdx, 1024
        add r9, 1024
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
        .size sdasm_avx10_fp16_1x1_a2, .-sdasm_avx10_fp16_1x1_a2

        .globl sdasm_avx10_fp16_1x1_b2
        .type sdasm_avx10_fp16_1x1_b2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_avx10_fp16_1x1_b2:
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
        .p2align 4
3:
        vmovups zmm16, [rdx + 64 * 0]
        vmovups zmm17, [r9 + 64 * 0]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 0]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 0]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 0]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 0]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 0]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 0]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 0]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 0]{1to16}
        vmovups zmm16, [rdx + 64 * 1]
        vmovups zmm17, [r9 + 64 * 1]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 1]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 1]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 1]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 1]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 1]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 1]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 1]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 1]{1to16}
        vmovups zmm16, [rdx + 64 * 2]
        vmovups zmm17, [r9 + 64 * 2]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 2]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 2]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 2]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 2]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 2]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 2]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 2]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 2]{1to16}
        vmovups zmm16, [rdx + 64 * 3]
        vmovups zmm17, [r9 + 64 * 3]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 3]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 3]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 3]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 3]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 3]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 3]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 3]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 3]{1to16}
        vmovups zmm16, [rdx + 64 * 4]
        vmovups zmm17, [r9 + 64 * 4]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 4]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 4]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 4]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 4]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 4]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 4]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 4]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 4]{1to16}
        vmovups zmm16, [rdx + 64 * 5]
        vmovups zmm17, [r9 + 64 * 5]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 5]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 5]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 5]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 5]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 5]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 5]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 5]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 5]{1to16}
        vmovups zmm16, [rdx + 64 * 6]
        vmovups zmm17, [r9 + 64 * 6]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 6]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 6]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 6]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 6]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 6]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 6]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 6]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 6]{1to16}
        vmovups zmm16, [rdx + 64 * 7]
        vmovups zmm17, [r9 + 64 * 7]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 7]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 7]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 7]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 7]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 7]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 7]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 7]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 7]{1to16}
        vmovups zmm16, [rdx + 64 * 8]
        vmovups zmm17, [r9 + 64 * 8]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 8]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 8]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 8]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 8]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 8]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 8]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 8]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 8]{1to16}
        vmovups zmm16, [rdx + 64 * 9]
        vmovups zmm17, [r9 + 64 * 9]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 9]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 9]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 9]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 9]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 9]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 9]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 9]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 9]{1to16}
        vmovups zmm16, [rdx + 64 * 10]
        vmovups zmm17, [r9 + 64 * 10]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 10]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 10]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 10]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 10]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 10]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 10]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 10]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 10]{1to16}
        vmovups zmm16, [rdx + 64 * 11]
        vmovups zmm17, [r9 + 64 * 11]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 11]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 11]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 11]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 11]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 11]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 11]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 11]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 11]{1to16}
        vmovups zmm16, [rdx + 64 * 12]
        vmovups zmm17, [r9 + 64 * 12]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 12]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 12]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 12]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 12]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 12]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 12]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 12]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 12]{1to16}
        vmovups zmm16, [rdx + 64 * 13]
        vmovups zmm17, [r9 + 64 * 13]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 13]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 13]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 13]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 13]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 13]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 13]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 13]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 13]{1to16}
        vmovups zmm16, [rdx + 64 * 14]
        vmovups zmm17, [r9 + 64 * 14]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 14]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 14]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 14]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 14]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 14]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 14]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 14]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 14]{1to16}
        vmovups zmm16, [rdx + 64 * 15]
        vmovups zmm17, [r9 + 64 * 15]
        vdpphps zmm0, zmm16, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpphps zmm1, zmm17, dword ptr [r10 + 64 * 0 + 4 * 15]{1to16}
        vdpphps zmm2, zmm16, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpphps zmm3, zmm17, dword ptr [r10 + 64 * 1 + 4 * 15]{1to16}
        vdpphps zmm4, zmm16, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpphps zmm5, zmm17, dword ptr [r10 + 64 * 2 + 4 * 15]{1to16}
        vdpphps zmm6, zmm16, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpphps zmm7, zmm17, dword ptr [r10 + 64 * 3 + 4 * 15]{1to16}
        vdpphps zmm8, zmm16, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpphps zmm9, zmm17, dword ptr [r10 + 64 * 4 + 4 * 15]{1to16}
        vdpphps zmm10, zmm16, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpphps zmm11, zmm17, dword ptr [r10 + 64 * 5 + 4 * 15]{1to16}
        vdpphps zmm12, zmm16, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpphps zmm13, zmm17, dword ptr [r10 + 64 * 6 + 4 * 15]{1to16}
        vdpphps zmm14, zmm16, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        vdpphps zmm15, zmm17, dword ptr [r10 + 64 * 7 + 4 * 15]{1to16}
        add r10, 1024
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
        .size sdasm_avx10_fp16_1x1_b2, .-sdasm_avx10_fp16_1x1_b2

// ---------------------------------------------------------------- AVX512-FP16

        .section .rodata
        .p2align 6
// The even and the odd lanes of two registers of eight pairs, by vpermt2ps.
.Leven:
        .long 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
.Lodd:
        .long 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        .text

// The FP16 sums of the four rows, zmm0..7, added to those of single precision,
// zmm8..23 -- each FP16 sum's lanes 0 to 15 to one register and 16 to 31 to the
// next -- and begun again from zeros.
.macro DRAIN
        vcvtph2psx zmm24, ymm0
        vextracti32x8 ymm25, zmm0, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm8, zmm8, zmm24
        vaddps zmm9, zmm9, zmm25
        vpxord zmm0, zmm0, zmm0
        vcvtph2psx zmm24, ymm1
        vextracti32x8 ymm25, zmm1, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm10, zmm10, zmm24
        vaddps zmm11, zmm11, zmm25
        vpxord zmm1, zmm1, zmm1
        vcvtph2psx zmm24, ymm2
        vextracti32x8 ymm25, zmm2, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm12, zmm12, zmm24
        vaddps zmm13, zmm13, zmm25
        vpxord zmm2, zmm2, zmm2
        vcvtph2psx zmm24, ymm3
        vextracti32x8 ymm25, zmm3, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm14, zmm14, zmm24
        vaddps zmm15, zmm15, zmm25
        vpxord zmm3, zmm3, zmm3
        vcvtph2psx zmm24, ymm4
        vextracti32x8 ymm25, zmm4, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm16, zmm16, zmm24
        vaddps zmm17, zmm17, zmm25
        vpxord zmm4, zmm4, zmm4
        vcvtph2psx zmm24, ymm5
        vextracti32x8 ymm25, zmm5, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm18, zmm18, zmm24
        vaddps zmm19, zmm19, zmm25
        vpxord zmm5, zmm5, zmm5
        vcvtph2psx zmm24, ymm6
        vextracti32x8 ymm25, zmm6, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm20, zmm20, zmm24
        vaddps zmm21, zmm21, zmm25
        vpxord zmm6, zmm6, zmm6
        vcvtph2psx zmm24, ymm7
        vextracti32x8 ymm25, zmm7, 1
        vcvtph2psx zmm25, ymm25
        vaddps zmm22, zmm22, zmm24
        vaddps zmm23, zmm23, zmm25
        vpxord zmm7, zmm7, zmm7
.endm

        .globl sdasm_fp16_1x1_a2_avx512
        .type sdasm_fp16_1x1_a2_avx512, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_fp16_1x1_a2_avx512:
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        mov r11, [rsp + 56]                     // ldc
        mov r12, [rsp + 64]                     // accumulate
        mov r13, r9                             // the rows of C of this four
        mov r14, rdx                            // a_next
        mov rbx, rsi                            // A's first tile of rows 0 to 15
        mov rbp, rcx                            // B's tile of columns 0 to 15
        vmovdqu32 zmm30, [rip + .Leven]
        vmovdqu32 zmm31, [rip + .Lodd]
        xor esi, esi                            // the four, 0 to 7
1:
        // r10: the first row of A of this four, in the tiles of the first block.
        mov r10, rsi
        and r10, 3
        shl r10, 8                              // four rows of 64 bytes
        add r10, rbx
        cmp rsi, 4
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
        vpxord zmm16, zmm16, zmm16
        vpxord zmm17, zmm17, zmm17
        vpxord zmm18, zmm18, zmm18
        vpxord zmm19, zmm19, zmm19
        vpxord zmm20, zmm20, zmm20
        vpxord zmm21, zmm21, zmm21
        vpxord zmm22, zmm22, zmm22
        vpxord zmm23, zmm23, zmm23
        mov rdx, rbp
        lea r9, [rbp + r8]
        mov rcx, rdi
        xor r15d, r15d                          // blocks since the last move to single precision
        test rcx, rcx
        jle 5f
        .p2align 4
3:
        vmovdqu16 zmm24, [rdx + 64 * 0]
        vmovdqu16 zmm25, [r9 + 64 * 0]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 0]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 0]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 0]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 0]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 1]
        vmovdqu16 zmm25, [r9 + 64 * 1]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 1]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 1]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 1]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 1]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 2]
        vmovdqu16 zmm25, [r9 + 64 * 2]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 2]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 2]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 2]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 2]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 3]
        vmovdqu16 zmm25, [r9 + 64 * 3]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 3]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 3]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 3]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 3]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 4]
        vmovdqu16 zmm25, [r9 + 64 * 4]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 4]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 4]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 4]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 4]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 5]
        vmovdqu16 zmm25, [r9 + 64 * 5]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 5]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 5]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 5]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 5]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 6]
        vmovdqu16 zmm25, [r9 + 64 * 6]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 6]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 6]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 6]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 6]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 7]
        vmovdqu16 zmm25, [r9 + 64 * 7]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 7]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 7]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 7]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 7]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 8]
        vmovdqu16 zmm25, [r9 + 64 * 8]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 8]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 8]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 8]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 8]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 9]
        vmovdqu16 zmm25, [r9 + 64 * 9]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 9]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 9]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 9]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 9]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 10]
        vmovdqu16 zmm25, [r9 + 64 * 10]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 10]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 10]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 10]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 10]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 11]
        vmovdqu16 zmm25, [r9 + 64 * 11]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 11]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 11]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 11]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 11]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 12]
        vmovdqu16 zmm25, [r9 + 64 * 12]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 12]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 12]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 12]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 12]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 13]
        vmovdqu16 zmm25, [r9 + 64 * 13]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 13]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 13]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 13]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 13]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 14]
        vmovdqu16 zmm25, [r9 + 64 * 14]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 14]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 14]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 14]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 14]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 15]
        vmovdqu16 zmm25, [r9 + 64 * 15]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 15]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 15]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 15]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 15]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        add r10, 2048
        add rdx, 1024
        add r9, 1024
        inc r15
        cmp r15, 2
        jb 4f
        DRAIN
        xor r15d, r15d
4:
        dec rcx
        jnz 3b
        DRAIN
5:
        // Each column's even and odd lanes added, and the four rows to C.
        mov rax, r13
        vmovaps zmm24, zmm8
        vpermt2ps zmm24, zmm30, zmm9
        vpermt2ps zmm8, zmm31, zmm9
        vaddps zmm8, zmm24, zmm8
        vmovaps zmm24, zmm10
        vpermt2ps zmm24, zmm30, zmm11
        vpermt2ps zmm10, zmm31, zmm11
        vaddps zmm10, zmm24, zmm10
        test r12, r12
        jz 6f
        vaddps zmm8, zmm8, [rax]
        vaddps zmm10, zmm10, [rax + 64]
6:
        vmovups [rax], zmm8
        vmovups [rax + 64], zmm10
        add rax, r11
        vmovaps zmm24, zmm12
        vpermt2ps zmm24, zmm30, zmm13
        vpermt2ps zmm12, zmm31, zmm13
        vaddps zmm12, zmm24, zmm12
        vmovaps zmm24, zmm14
        vpermt2ps zmm24, zmm30, zmm15
        vpermt2ps zmm14, zmm31, zmm15
        vaddps zmm14, zmm24, zmm14
        test r12, r12
        jz 6f
        vaddps zmm12, zmm12, [rax]
        vaddps zmm14, zmm14, [rax + 64]
6:
        vmovups [rax], zmm12
        vmovups [rax + 64], zmm14
        add rax, r11
        vmovaps zmm24, zmm16
        vpermt2ps zmm24, zmm30, zmm17
        vpermt2ps zmm16, zmm31, zmm17
        vaddps zmm16, zmm24, zmm16
        vmovaps zmm24, zmm18
        vpermt2ps zmm24, zmm30, zmm19
        vpermt2ps zmm18, zmm31, zmm19
        vaddps zmm18, zmm24, zmm18
        test r12, r12
        jz 6f
        vaddps zmm16, zmm16, [rax]
        vaddps zmm18, zmm18, [rax + 64]
6:
        vmovups [rax], zmm16
        vmovups [rax + 64], zmm18
        add rax, r11
        vmovaps zmm24, zmm20
        vpermt2ps zmm24, zmm30, zmm21
        vpermt2ps zmm20, zmm31, zmm21
        vaddps zmm20, zmm24, zmm20
        vmovaps zmm24, zmm22
        vpermt2ps zmm24, zmm30, zmm23
        vpermt2ps zmm22, zmm31, zmm23
        vaddps zmm22, zmm24, zmm22
        test r12, r12
        jz 6f
        vaddps zmm20, zmm20, [rax]
        vaddps zmm22, zmm22, [rax + 64]
6:
        vmovups [rax], zmm20
        vmovups [rax + 64], zmm22
        add rax, r11
        mov r13, rax
        inc rsi
        cmp rsi, 8
        jb 1b
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        vzeroupper
        ret
        .size sdasm_fp16_1x1_a2_avx512, .-sdasm_fp16_1x1_a2_avx512

        .globl sdasm_fp16_1x1_b2_avx512
        .type sdasm_fp16_1x1_b2_avx512, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_fp16_1x1_b2_avx512:
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        mov r11, [rsp + 56]                     // ldc
        mov r12, [rsp + 64]                     // accumulate
        mov r13, r9                             // the rows of C of this four
        mov r14, rdx                            // a_next
        mov rbx, rsi                            // A's first tile of rows 0 to 15
        mov rbp, rcx                            // B's tile of columns 0 to 15
        vmovdqu32 zmm30, [rip + .Leven]
        vmovdqu32 zmm31, [rip + .Lodd]
        xor esi, esi                            // the four, 0 to 7
1:
        // r10: the first row of A of this four, in the tiles of the first block.
        mov r10, rsi
        and r10, 3
        shl r10, 8                              // four rows of 64 bytes
        add r10, rbx
        cmp rsi, 4
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
        vpxord zmm16, zmm16, zmm16
        vpxord zmm17, zmm17, zmm17
        vpxord zmm18, zmm18, zmm18
        vpxord zmm19, zmm19, zmm19
        vpxord zmm20, zmm20, zmm20
        vpxord zmm21, zmm21, zmm21
        vpxord zmm22, zmm22, zmm22
        vpxord zmm23, zmm23, zmm23
        mov rdx, rbp
        lea r9, [rbp + r8]
        mov rcx, rdi
        xor r15d, r15d                          // blocks since the last move to single precision
        test rcx, rcx
        jle 5f
        .p2align 4
3:
        vmovdqu16 zmm24, [rdx + 64 * 0]
        vmovdqu16 zmm25, [r9 + 64 * 0]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 0]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 0]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 0]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 0]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 1]
        vmovdqu16 zmm25, [r9 + 64 * 1]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 1]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 1]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 1]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 1]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 2]
        vmovdqu16 zmm25, [r9 + 64 * 2]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 2]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 2]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 2]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 2]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 3]
        vmovdqu16 zmm25, [r9 + 64 * 3]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 3]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 3]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 3]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 3]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 4]
        vmovdqu16 zmm25, [r9 + 64 * 4]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 4]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 4]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 4]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 4]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 5]
        vmovdqu16 zmm25, [r9 + 64 * 5]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 5]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 5]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 5]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 5]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 6]
        vmovdqu16 zmm25, [r9 + 64 * 6]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 6]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 6]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 6]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 6]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 7]
        vmovdqu16 zmm25, [r9 + 64 * 7]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 7]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 7]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 7]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 7]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 8]
        vmovdqu16 zmm25, [r9 + 64 * 8]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 8]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 8]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 8]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 8]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 9]
        vmovdqu16 zmm25, [r9 + 64 * 9]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 9]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 9]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 9]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 9]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 10]
        vmovdqu16 zmm25, [r9 + 64 * 10]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 10]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 10]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 10]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 10]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 11]
        vmovdqu16 zmm25, [r9 + 64 * 11]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 11]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 11]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 11]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 11]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 12]
        vmovdqu16 zmm25, [r9 + 64 * 12]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 12]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 12]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 12]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 12]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 13]
        vmovdqu16 zmm25, [r9 + 64 * 13]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 13]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 13]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 13]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 13]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 14]
        vmovdqu16 zmm25, [r9 + 64 * 14]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 14]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 14]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 14]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 14]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        vmovdqu16 zmm24, [rdx + 64 * 15]
        vmovdqu16 zmm25, [r9 + 64 * 15]
        vpbroadcastd zmm26, [r10 + 64 * 0 + 4 * 15]
        vpbroadcastd zmm27, [r10 + 64 * 1 + 4 * 15]
        vpbroadcastd zmm28, [r10 + 64 * 2 + 4 * 15]
        vpbroadcastd zmm29, [r10 + 64 * 3 + 4 * 15]
        vfmadd231ph zmm0, zmm24, zmm26
        vfmadd231ph zmm1, zmm25, zmm26
        vfmadd231ph zmm2, zmm24, zmm27
        vfmadd231ph zmm3, zmm25, zmm27
        vfmadd231ph zmm4, zmm24, zmm28
        vfmadd231ph zmm5, zmm25, zmm28
        vfmadd231ph zmm6, zmm24, zmm29
        vfmadd231ph zmm7, zmm25, zmm29
        add r10, 1024
        add rdx, 2048
        add r9, 2048
        inc r15
        cmp r15, 2
        jb 4f
        DRAIN
        xor r15d, r15d
4:
        dec rcx
        jnz 3b
        DRAIN
5:
        // Each column's even and odd lanes added, and the four rows to C.
        mov rax, r13
        vmovaps zmm24, zmm8
        vpermt2ps zmm24, zmm30, zmm9
        vpermt2ps zmm8, zmm31, zmm9
        vaddps zmm8, zmm24, zmm8
        vmovaps zmm24, zmm10
        vpermt2ps zmm24, zmm30, zmm11
        vpermt2ps zmm10, zmm31, zmm11
        vaddps zmm10, zmm24, zmm10
        test r12, r12
        jz 6f
        vaddps zmm8, zmm8, [rax]
        vaddps zmm10, zmm10, [rax + 64]
6:
        vmovups [rax], zmm8
        vmovups [rax + 64], zmm10
        add rax, r11
        vmovaps zmm24, zmm12
        vpermt2ps zmm24, zmm30, zmm13
        vpermt2ps zmm12, zmm31, zmm13
        vaddps zmm12, zmm24, zmm12
        vmovaps zmm24, zmm14
        vpermt2ps zmm24, zmm30, zmm15
        vpermt2ps zmm14, zmm31, zmm15
        vaddps zmm14, zmm24, zmm14
        test r12, r12
        jz 6f
        vaddps zmm12, zmm12, [rax]
        vaddps zmm14, zmm14, [rax + 64]
6:
        vmovups [rax], zmm12
        vmovups [rax + 64], zmm14
        add rax, r11
        vmovaps zmm24, zmm16
        vpermt2ps zmm24, zmm30, zmm17
        vpermt2ps zmm16, zmm31, zmm17
        vaddps zmm16, zmm24, zmm16
        vmovaps zmm24, zmm18
        vpermt2ps zmm24, zmm30, zmm19
        vpermt2ps zmm18, zmm31, zmm19
        vaddps zmm18, zmm24, zmm18
        test r12, r12
        jz 6f
        vaddps zmm16, zmm16, [rax]
        vaddps zmm18, zmm18, [rax + 64]
6:
        vmovups [rax], zmm16
        vmovups [rax + 64], zmm18
        add rax, r11
        vmovaps zmm24, zmm20
        vpermt2ps zmm24, zmm30, zmm21
        vpermt2ps zmm20, zmm31, zmm21
        vaddps zmm20, zmm24, zmm20
        vmovaps zmm24, zmm22
        vpermt2ps zmm24, zmm30, zmm23
        vpermt2ps zmm22, zmm31, zmm23
        vaddps zmm22, zmm24, zmm22
        test r12, r12
        jz 6f
        vaddps zmm20, zmm20, [rax]
        vaddps zmm22, zmm22, [rax + 64]
6:
        vmovups [rax], zmm20
        vmovups [rax + 64], zmm22
        add rax, r11
        mov r13, rax
        inc rsi
        cmp rsi, 8
        jb 1b
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        vzeroupper
        ret
        .size sdasm_fp16_1x1_b2_avx512, .-sdasm_fp16_1x1_b2_avx512

        .section .note.GNU-stack, "", @progbits
