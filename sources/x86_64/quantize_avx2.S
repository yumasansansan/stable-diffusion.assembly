// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_quantize16_avx2(uint8_t *dst, const float *src,
//                                int64_t groups, int64_t quads,
//                                float levels);
//
// The quantization of a panel of sixteen columns of B (quant.h) for AVX2:
// `groups` groups of 4*quads rows of src, [k][16] in single precision -- the
// layout of the panels of the kernels of single precision -- into the bytes of
// each group, [quads][16][4], then its sixteen scales and sixteen offsets.
// Each column of a group is quantized as an input, unsigned with a zero point,
// to levels + 1 levels: its least and greatest, 0 between them, in two
// registers of eight a side, a vertical minimum and maximum a row; the scale
// (greatest - least) / levels, its reciprocal (0 for a scale of 0), the zero
// point round(-least / scale); each number round(x / scale) + zero, held to
// [0, levels], a quad of rows at a time, converted to integers and packed --
// vpackusdw and vpackuswb lay the four rows' bytes of four columns side by side
// in each half of a register, and vpshufb turns each half from rows by columns
// to columns by rows, the four bytes of a column together, as vpdpbusd reads
// them. It is quantize_panel_inputs() of quant.cc, in the same order, to the
// bit.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 5
// Within each half of a register, the sixteen bytes of four rows of four
// columns, row after row, to column after column.
.Lcolumns:
        .byte 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
        .byte 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
.Lone:
        .long 0x3f800000

        .text

// A quad of rows at rsi, round(x / scale) + zero held to [0, levels] in ymm\a
// (columns 0..7) and ymm\b (columns 8..15) of row \r: ymm8/ymm9 the
// reciprocals, ymm10/ymm11 the zero points, ymm12 levels, ymm13 zeros.
.macro LEVEL r, a, b
        vmulps ymm\a, ymm8, [rsi + 64*\r]
        vmulps ymm\b, ymm9, [rsi + 64*\r + 32]
        vroundps ymm\a, ymm\a, 12
        vroundps ymm\b, ymm\b, 12
        vaddps ymm\a, ymm\a, ymm10
        vaddps ymm\b, ymm\b, ymm11
        vmaxps ymm\a, ymm\a, ymm13
        vmaxps ymm\b, ymm\b, ymm13
        vminps ymm\a, ymm\a, ymm12
        vminps ymm\b, ymm\b, ymm12
        vcvttps2dq ymm\a, ymm\a
        vcvttps2dq ymm\b, ymm\b
.endm

        .globl sdasm_quantize16_avx2
        .type sdasm_quantize16_avx2, @function
        .p2align 5
sdasm_quantize16_avx2:
        vbroadcastss ymm12, xmm0
        vxorps xmm13, xmm13, xmm13
        vmovdqa ymm15, [rip + .Lcolumns]
        test rdx, rdx
        jz 9f
        .p2align 4
1:
        // The group's least and greatest of each column, from 0.
        vxorps xmm0, xmm0, xmm0
        vxorps xmm1, xmm1, xmm1
        vxorps xmm2, xmm2, xmm2
        vxorps xmm3, xmm3, xmm3
        lea r8, [4*rcx]
        mov rax, rsi
2:
        vminps ymm0, ymm0, [rax]
        vminps ymm1, ymm1, [rax + 32]
        vmaxps ymm2, ymm2, [rax]
        vmaxps ymm3, ymm3, [rax + 32]
        add rax, 64
        dec r8
        jnz 2b
        // The scales (ymm6, ymm7), their reciprocals (ymm8, ymm9) and the
        // zero points (ymm10, ymm11).
        vsubps ymm6, ymm2, ymm0
        vsubps ymm7, ymm3, ymm1
        vdivps ymm6, ymm6, ymm12
        vdivps ymm7, ymm7, ymm12
        vbroadcastss ymm14, [rip + .Lone]
        vdivps ymm8, ymm14, ymm6
        vdivps ymm9, ymm14, ymm7
        vcmpps ymm4, ymm6, ymm13, 14
        vcmpps ymm5, ymm7, ymm13, 14
        vandps ymm8, ymm8, ymm4
        vandps ymm9, ymm9, ymm5
        vxorps xmm4, xmm4, xmm4
        vsubps ymm0, ymm4, ymm0
        vsubps ymm1, ymm4, ymm1
        vmulps ymm10, ymm0, ymm8
        vmulps ymm11, ymm1, ymm9
        vroundps ymm10, ymm10, 12
        vroundps ymm11, ymm11, 12
        // The bytes, a quad of rows at a time.
        mov r8, rcx
3:
        LEVEL 0, 0, 1
        LEVEL 1, 2, 3
        vpackusdw ymm0, ymm0, ymm2
        vpackusdw ymm1, ymm1, ymm3
        LEVEL 2, 2, 3
        LEVEL 3, 4, 5
        vpackusdw ymm2, ymm2, ymm4
        vpackusdw ymm3, ymm3, ymm5
        vpackuswb ymm0, ymm0, ymm2
        vpackuswb ymm1, ymm1, ymm3
        vpshufb ymm0, ymm0, ymm15
        vpshufb ymm1, ymm1, ymm15
        vmovdqu [rdi], ymm0
        vmovdqu [rdi + 32], ymm1
        add rsi, 256
        add rdi, 64
        dec r8
        jnz 3b
        // The scales and the offsets, scale times zero point.
        vmovups [rdi], ymm6
        vmovups [rdi + 32], ymm7
        vmulps ymm6, ymm6, ymm10
        vmulps ymm7, ymm7, ymm11
        vmovups [rdi + 64], ymm6
        vmovups [rdi + 96], ymm7
        add rdi, 128
        dec rdx
        jnz 1b
9:
        vzeroupper
        ret
        .size sdasm_quantize16_avx2, . - sdasm_quantize16_avx2

        .section .note.GNU-stack, "", @progbits
