// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The quantized products (quant.h) on the tiles of AMX-INT8, with AVX-512 F to
// scale them, in the configuration of amx.S -- eight tiles of sixteen rows of
// 64 bytes:
//
//     void sdasm_amx_qgemm_32x32_au(int64_t groups, int64_t quads,
//                                   const uint8_t *a, int64_t a_next,
//                                   const uint8_t *b, int64_t b_next, float *c,
//                                   int64_t ldc, int64_t accumulate);
//
// C = (accumulate ? C : 0) + A B for a block of 32 rows by 32 columns of C,
// rows ldc bytes apart, over `groups` groups of `quads` quads of k, quads a
// multiple of sixteen -- a group of 64 numbers of k, or of a multiple of 64. A
// holds the inputs, unsigned with a zero point, as panels of sixteen rows
// (Qlayout::rows), those of rows 16 to 31 a_next bytes after those of rows 0 to
// 15; B the weights, signed, as panels of sixteen columns (Qlayout::columns),
// those of columns 16 to 31 b_next bytes after. A group of a panel of A is a
// tile's sixteen rows, 4 quads bytes apart, each 64 numbers of k a tile; one of
// B is sixteen columns by the quads, each sixteen quads a tile as tdpbusd takes
// it -- for each quad, the four bytes of each column side by side -- the
// layout of the panels of AVX512-VNNI's kernels. After each group's integers,
// its sixteen scales and sixteen offsets, floats.
//
// Each group's products are summed exactly in 32 bits on the tiles -- C's four
// tiles in tmm0 to tmm3, by tdpbusd of A's two tiles, tmm4 and tmm5, and B's
// two, tmm6 and tmm7 -- stored into one of two buffers on the stack, and
// scaled into C with AVX-512 as quant_avx512.S scales them, in the same order:
//
//     c[i][j] += (sa[i] sb[j]) sum - oa[i] ob[j].
//
// The scaling of a group is about twice the time of its products when the
// group is 64 numbers of k, as long when it is 128: so the scaling of each
// group is written after the products of the next, which do not wait on it,
// and the processor takes the two together, the tiles' products and the
// vector units' scaling, the sums of the next group going to the other buffer.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix
        .text

// The locals, above the two buffers of four tiles of 32-bit sums (8192 bytes).
.set LDC, 8192                  // C's stride in bytes
.set META, 8200                 // the bytes of a group's integers: 64 quads
.set PREV_A, 8208               // the previous group's scales of A's rows 0 to 15
.set PREV_B, 8216               // and of B's columns 0 to 15
.set PREV_BUFFER, 8224          // and the offset of the buffer of its sums
.set BUFFER, 8232               // the offset of the buffer of this group's
.set PREVIOUS, 8240             // whether there is a previous group to scale
.set FRAME, 8256

// Row \r of a tile of A: its sums in the two tiles of 16 columns at rsi +
// \left and rsi + \right, scaled into C's row at r11 and on to the next row.
// rbx holds the rows' scales and offsets, zmm16 to zmm19 the columns'.
.macro ROWQ r, left, right
        vcvtdq2ps zmm0, [rsi + \left + 64*\r]
        vcvtdq2ps zmm1, [rsi + \right + 64*\r]
        vmulps zmm2, zmm16, dword ptr [rbx + 4*\r]{1to16}
        vmulps zmm3, zmm17, dword ptr [rbx + 4*\r]{1to16}
        vfmadd213ps zmm0, zmm2, [r11]
        vfmadd213ps zmm1, zmm3, [r11 + 64]
        vfnmadd231ps zmm0, zmm18, dword ptr [rbx + 64 + 4*\r]{1to16}
        vfnmadd231ps zmm1, zmm19, dword ptr [rbx + 64 + 4*\r]{1to16}
        vmovups [r11], zmm0
        vmovups [r11 + 64], zmm1
        add r11, [rsp + LDC]
.endm

// The previous group's sums scaled into C, all 32 rows.
.macro SCALE_PREVIOUS
        mov rsi, [rsp + PREV_BUFFER]
        add rsi, rsp
        mov rax, [rsp + PREV_B]
        vmovups zmm16, [rax]                    // B's scales and offsets, columns 0 to 15
        vmovups zmm18, [rax + 64]
        add rax, r9
        vmovups zmm17, [rax]                    // and 16 to 31
        vmovups zmm19, [rax + 64]
        mov rbx, [rsp + PREV_A]
        mov r11, r10
        .irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ROWQ \r, 0, 1024
        .endr
        add rbx, rcx                            // A's rows 16 to 31
        .irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ROWQ \r, 2048, 3072
        .endr
.endm

        .globl sdasm_amx_qgemm_32x32_au
        .type sdasm_amx_qgemm_32x32_au, @function
        .p2align 5
// rdi groups, rsi quads, rdx a, rcx a_next, r8 b, r9 b_next, [rsp + 8] c,
// [rsp + 16] ldc, [rsp + 24] accumulate
sdasm_amx_qgemm_32x32_au:
        push rbp
        mov rbp, rsp
        push rbx
        push r12
        push r13
        push r14
        push r15
        sub rsp, FRAME + 64
        and rsp, -64
        mov r10, [rbp + 16]                     // c
        mov rax, [rbp + 24]
        mov [rsp + LDC], rax
        // C from zeros, unless it is added to.
        cmp qword ptr [rbp + 32], 0
        jne 1f
        vpxord zmm0, zmm0, zmm0
        mov r11, r10
        mov eax, 32
2:
        vmovups [r11], zmm0
        vmovups [r11 + 64], zmm0
        add r11, [rsp + LDC]
        dec eax
        jnz 2b
1:
        lea r12, [4*rsi]                        // A's rows of a group: 4 quads bytes apart
        mov rax, rsi
        shl rax, 6
        mov [rsp + META], rax                   // a group's integers: 64 quads bytes
        lea r13, [rax + 128]                    // a group, its scales and offsets after
        mov r14, rsi
        shr r14, 4                              // the tiles of k of a group
        mov r15d, 64                            // a tile of B's rows, and of the sums
        mov qword ptr [rsp + BUFFER], 0
        mov qword ptr [rsp + PREVIOUS], 0
3:
        tilezero tmm0
        tilezero tmm1
        tilezero tmm2
        tilezero tmm3
        mov rax, rdx                            // A's rows 0 to 15, this tile of k
        mov rsi, r8                             // B's columns 0 to 15
        mov r11, r14
4:
        lea rbx, [rax + rcx]                    // A's rows 16 to 31
        tileloadd tmm4, [rax + r12 * 1]
        tileloadd tmm5, [rbx + r12 * 1]
        lea rbx, [rsi + r9]                     // B's columns 16 to 31
        tileloadd tmm6, [rsi + r15 * 1]
        tileloadd tmm7, [rbx + r15 * 1]
        tdpbusd tmm0, tmm4, tmm6
        tdpbusd tmm1, tmm4, tmm7
        tdpbusd tmm2, tmm5, tmm6
        tdpbusd tmm3, tmm5, tmm7
        add rax, 64
        add rsi, 1024
        dec r11
        jnz 4b
        // The previous group's scaling, while these products run.
        cmp qword ptr [rsp + PREVIOUS], 0
        je 5f
        SCALE_PREVIOUS
5:
        // This group's sums out of the tiles, into its buffer.
        mov rax, [rsp + BUFFER]
        mov [rsp + PREV_BUFFER], rax
        add rax, rsp
        tilestored [rax + r15 * 1], tmm0
        tilestored [rax + r15 * 1 + 1024], tmm1
        tilestored [rax + r15 * 1 + 2048], tmm2
        tilestored [rax + r15 * 1 + 3072], tmm3
        xor qword ptr [rsp + BUFFER], 4096
        mov rax, [rsp + META]
        lea rbx, [rdx + rax]                    // this group's scales and offsets
        mov [rsp + PREV_A], rbx
        lea rax, [r8 + rax]
        mov [rsp + PREV_B], rax
        mov qword ptr [rsp + PREVIOUS], 1
        add rdx, r13
        add r8, r13
        dec rdi
        jnz 3b
        SCALE_PREVIOUS
        lea rsp, [rbp - 40]
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp
        vzeroupper
        ret
        .size sdasm_amx_qgemm_32x32_au, .-sdasm_amx_qgemm_32x32_au

        .section .note.GNU-stack, "", @progbits
