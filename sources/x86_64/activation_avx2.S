// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_gated_gelu_avx2(float *out, const float *x, const float *gate,
//                                int64_t n);
//
// The gated GELU of stable-diffusion.cpp's GEGLU (fusion.h): out = x
// GELU(gate) for n numbers, n a multiple of sixteen, GELU by its definition
// with tanh, gate / (1 + exp(-2 y)), y = sqrt(2 / pi) (gate + 0.044715
// gate^3), as gelu() in scalar_math.h has it. The exponential is that of the
// softmax (softmax_avx2.S), its argument clamped to [-87, 88]: below, it is
// under 1.2e-38, and beside the 1 it is added to nothing; above, the GELU of
// a gate under -9.8, which the exponential makes -0 or a few units of 1e-38,
// as against its exact value of less than 1e-21. The division is the
// processor's, rounded to the nearest.
//
// Eight numbers to a register, two registers at a time, their instructions
// interleaved for the latency of a chain of some twenty operations, the
// division's longest.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 5
.Ltable:        // 2^(k/8), k = 0..7
        .float 0x1p+0, 0x1.172b84p+0, 0x1.306fep+0, 0x1.4bfdaep+0
        .float 0x1.6a09e6p+0, 0x1.8ace54p+0, 0x1.ae89fap+0, 0x1.d5818ep+0
        .p2align 2
.Lmagic: .float 0x1.8p23
.Lscale: .float 11.541560327111707      // 8 / ln(2)
.Lhigh: .float 0x1.62e43p-4             // ln(2) / 8, its high part
.Llow:  .float -0x1.05c610p-32          // and what it leaves
.Llower: .float -87.0
.Lupper: .float 88.0
.Lc4:   .float 0x1.555556p-5            // 1/24
.Lc3:   .float 0x1.555556p-3            // 1/6
.Lc2:   .float 0.5
.Lone:  .float 1.0
.Lcubic: .float 0.044715
.Ltwice: .float -1.5957691216057308     // -2 sqrt(2 / pi)
.Lexponent: .long 0xff800000

        .text

// Two vectors of gates at [rdx + \o] and [rdx + \o + 32] into the GELU,
// multiplied by the x at [rsi + ...], stored at [rdi + ...]. Registers: chain
// A ymm0..ymm4, chain B ymm5..ymm9; ymm15 8 / ln(2), ymm14 1.5 2^23, ymm13 the
// table, ymm12 1, ymm11 0.044715, ymm10 -2 sqrt(2 / pi).
.macro TWO o
        vmovups ymm0, [rdx + \o]                // g
        vmovups ymm5, [rdx + \o + 32]
        vmulps ymm1, ymm0, ymm0                 // g^2
        vmulps ymm6, ymm5, ymm5
        vfmadd213ps ymm1, ymm11, ymm12          // 0.044715 g^2 + 1
        vfmadd213ps ymm6, ymm11, ymm12
        vmulps ymm1, ymm1, ymm0                 // y = g (1 + 0.044715 g^2)
        vmulps ymm6, ymm6, ymm5
        vmulps ymm1, ymm1, ymm10                // the exponential's argument, -2 y
        vmulps ymm6, ymm6, ymm10
        vmaxps ymm1, ymm1, [rip + .Lclamps]
        vmaxps ymm6, ymm6, [rip + .Lclamps]
        vminps ymm1, ymm1, [rip + .Lclamps + 32]
        vminps ymm6, ymm6, [rip + .Lclamps + 32]
        vmovaps ymm2, ymm15
        vmovaps ymm7, ymm15
        vfmadd213ps ymm2, ymm1, ymm14           // z
        vfmadd213ps ymm7, ymm6, ymm14
        vsubps ymm3, ymm2, ymm14                // n
        vsubps ymm8, ymm7, ymm14
        vfnmadd231ps ymm1, ymm3, [rip + .Lconstants + 0]        // r = x - n ln(2)/8
        vfnmadd231ps ymm6, ymm8, [rip + .Lconstants + 0]
        vfnmadd231ps ymm1, ymm3, [rip + .Lconstants + 32]
        vfnmadd231ps ymm6, ymm8, [rip + .Lconstants + 32]
        vpermps ymm3, ymm2, ymm13               // 2^((n mod 8)/8)
        vpermps ymm8, ymm7, ymm13
        vpslld ymm2, ymm2, 20
        vpslld ymm7, ymm7, 20
        vpand ymm2, ymm2, [rip + .Lconstants + 64]
        vpand ymm7, ymm7, [rip + .Lconstants + 64]
        vpaddd ymm3, ymm3, ymm2                 // times 2^(n div 8)
        vpaddd ymm8, ymm8, ymm7
        vmovaps ymm4, [rip + .Lconstants + 96]  // 1/24
        vmovaps ymm9, [rip + .Lconstants + 96]
        vfmadd213ps ymm4, ymm1, [rip + .Lconstants + 128]       // + 1/6
        vfmadd213ps ymm9, ymm6, [rip + .Lconstants + 128]
        vfmadd213ps ymm4, ymm1, [rip + .Lconstants + 160]       // + 1/2
        vfmadd213ps ymm9, ymm6, [rip + .Lconstants + 160]
        vfmadd213ps ymm4, ymm1, ymm12
        vfmadd213ps ymm9, ymm6, ymm12
        vfmadd213ps ymm4, ymm1, ymm12           // exp(r)
        vfmadd213ps ymm9, ymm6, ymm12
        vfmadd213ps ymm4, ymm3, ymm12           // 1 + exp(-2 y)
        vfmadd213ps ymm9, ymm8, ymm12
        vdivps ymm0, ymm0, ymm4                 // GELU(g)
        vdivps ymm5, ymm5, ymm9
        vmulps ymm0, ymm0, [rsi + \o]           // x GELU(g)
        vmulps ymm5, ymm5, [rsi + \o + 32]
        vmovups [rdi + \o], ymm0
        vmovups [rdi + \o + 32], ymm5
.endm

        .globl sdasm_gated_gelu_avx2
        .type sdasm_gated_gelu_avx2, @function
        .p2align 5
// rdi out, rsi x, rdx gate, rcx n
sdasm_gated_gelu_avx2:
        shr rcx, 4
        jz 9f
        vbroadcastss ymm15, [rip + .Lscale]
        vbroadcastss ymm14, [rip + .Lmagic]
        vmovaps ymm13, [rip + .Ltable]
        vbroadcastss ymm12, [rip + .Lone]
        vbroadcastss ymm11, [rip + .Lcubic]
        vbroadcastss ymm10, [rip + .Ltwice]
1:
        TWO 0
        add rdi, 64
        add rsi, 64
        add rdx, 64
        dec rcx
        jnz 1b
9:
        vzeroupper
        ret
        .size sdasm_gated_gelu_avx2, .-sdasm_gated_gelu_avx2

        .section .rodata
        .p2align 5
.Lclamps:
        .rept 8
        .float -87.0
        .endr
        .rept 8
        .float 88.0
        .endr
.Lconstants:
        .rept 8
        .float 0x1.62e43p-4
        .endr
        .rept 8
        .float -0x1.05c610p-32
        .endr
        .rept 8
        .long 0xff800000
        .endr
        .rept 8
        .float 0x1.555556p-5
        .endr
        .rept 8
        .float 0x1.555556p-3
        .endr
        .rept 8
        .float 0.5
        .endr

        .section .note.GNU-stack, "", @progbits
