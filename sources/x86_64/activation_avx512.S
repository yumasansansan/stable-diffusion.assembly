// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_gated_gelu_avx512(float *out, const float *x,
//                                  const float *gate, int64_t n);
//
// The gated GELU of activation_avx2.S for AVX-512: n a multiple of
// sixty-four, sixteen numbers to a register, four registers at a time; the
// exponential that of softmax_avx512.S, of a table of sixteen and a
// polynomial of degree three, its argument clamped to [-87, 88].
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 6
.Ltable:        // 2^(j/16), j = 0..15
        .float 0x1p+0, 0x1.0b5586p+0, 0x1.172b84p+0, 0x1.2387a6p+0
        .float 0x1.306fep+0, 0x1.3dea64p+0, 0x1.4bfdaep+0, 0x1.5ab07ep+0
        .float 0x1.6a09e6p+0, 0x1.7a1148p+0, 0x1.8ace54p+0, 0x1.9c4918p+0
        .float 0x1.ae89fap+0, 0x1.c199bep+0, 0x1.d5818ep+0, 0x1.ea4afap+0
        .p2align 2
.Lmagic: .float 0x1.8p23
.Lscale: .float 0x1.715476p+4           // 16 / ln(2)
.Lhigh: .float 0x1.62e43p-5
.Llow:  .float -0x1.05c61p-33
.Llower: .float -87.0
.Lupper: .float 88.0
.Lc3:   .float 0x1.555556p-3
.Lc2:   .float 0.5
.Lone:  .float 1.0
.Lcubic: .float 0.044715
.Ltwice: .float -1.5957691216057308
.Lexponent: .long 0xff800000

        .text

        .globl sdasm_gated_gelu_avx512
        .type sdasm_gated_gelu_avx512, @function
        .p2align 5
// rdi out, rsi x, rdx gate, rcx n
sdasm_gated_gelu_avx512:
        shr rcx, 6
        jz 9f
        vmovups zmm31, [rip + .Ltable]
        vbroadcastss zmm30, [rip + .Lmagic]
        vbroadcastss zmm29, [rip + .Lscale]
        vbroadcastss zmm28, [rip + .Lhigh]
        vbroadcastss zmm27, [rip + .Llow]
        vbroadcastss zmm26, [rip + .Llower]
        vbroadcastss zmm25, [rip + .Lupper]
        vbroadcastss zmm24, [rip + .Lc3]
        vbroadcastss zmm23, [rip + .Lc2]
        vbroadcastss zmm22, [rip + .Lone]
        vbroadcastss zmm21, [rip + .Lcubic]
        vbroadcastss zmm20, [rip + .Ltwice]
        vpbroadcastd zmm19, [rip + .Lexponent]
1:
        // Four registers of sixteen, register c in zmm c (g), zmm 4 + c (the
        // exponential's argument, then r), zmm 8 + c (z), zmm 12 + c (n, then
        // the table's factor) and zmm 16 + c (exp(r), then 1 + exp(-2 y)).
        vmovups zmm0, [rdx + 64 * 0]            // g
        vmovups zmm1, [rdx + 64 * 1]
        vmovups zmm2, [rdx + 64 * 2]
        vmovups zmm3, [rdx + 64 * 3]
        vmulps zmm4, zmm0, zmm0                 // g^2
        vmulps zmm5, zmm1, zmm1
        vmulps zmm6, zmm2, zmm2
        vmulps zmm7, zmm3, zmm3
        vfmadd213ps zmm4, zmm21, zmm22          // 0.044715 g^2 + 1
        vfmadd213ps zmm5, zmm21, zmm22
        vfmadd213ps zmm6, zmm21, zmm22
        vfmadd213ps zmm7, zmm21, zmm22
        vmulps zmm4, zmm4, zmm0                 // y
        vmulps zmm5, zmm5, zmm1
        vmulps zmm6, zmm6, zmm2
        vmulps zmm7, zmm7, zmm3
        vmulps zmm4, zmm4, zmm20                // -2 y
        vmulps zmm5, zmm5, zmm20
        vmulps zmm6, zmm6, zmm20
        vmulps zmm7, zmm7, zmm20
        vmaxps zmm4, zmm4, zmm26
        vmaxps zmm5, zmm5, zmm26
        vmaxps zmm6, zmm6, zmm26
        vmaxps zmm7, zmm7, zmm26
        vminps zmm4, zmm4, zmm25
        vminps zmm5, zmm5, zmm25
        vminps zmm6, zmm6, zmm25
        vminps zmm7, zmm7, zmm25
        vmovaps zmm8, zmm29
        vmovaps zmm9, zmm29
        vmovaps zmm10, zmm29
        vmovaps zmm11, zmm29
        vfmadd213ps zmm8, zmm4, zmm30           // z
        vfmadd213ps zmm9, zmm5, zmm30
        vfmadd213ps zmm10, zmm6, zmm30
        vfmadd213ps zmm11, zmm7, zmm30
        vsubps zmm12, zmm8, zmm30               // n
        vsubps zmm13, zmm9, zmm30
        vsubps zmm14, zmm10, zmm30
        vsubps zmm15, zmm11, zmm30
        vfnmadd231ps zmm4, zmm12, zmm28         // r
        vfnmadd231ps zmm5, zmm13, zmm28
        vfnmadd231ps zmm6, zmm14, zmm28
        vfnmadd231ps zmm7, zmm15, zmm28
        vfnmadd231ps zmm4, zmm12, zmm27
        vfnmadd231ps zmm5, zmm13, zmm27
        vfnmadd231ps zmm6, zmm14, zmm27
        vfnmadd231ps zmm7, zmm15, zmm27
        vpermps zmm12, zmm8, zmm31              // 2^((n mod 16)/16)
        vpermps zmm13, zmm9, zmm31
        vpermps zmm14, zmm10, zmm31
        vpermps zmm15, zmm11, zmm31
        vpslld zmm8, zmm8, 19
        vpslld zmm9, zmm9, 19
        vpslld zmm10, zmm10, 19
        vpslld zmm11, zmm11, 19
        vpandd zmm8, zmm8, zmm19
        vpandd zmm9, zmm9, zmm19
        vpandd zmm10, zmm10, zmm19
        vpandd zmm11, zmm11, zmm19
        vpaddd zmm12, zmm12, zmm8               // times 2^(n div 16)
        vpaddd zmm13, zmm13, zmm9
        vpaddd zmm14, zmm14, zmm10
        vpaddd zmm15, zmm15, zmm11
        vmovaps zmm16, zmm24
        vmovaps zmm17, zmm24
        vmovaps zmm18, zmm24
        vmovaps zmm19, zmm24
        vfmadd213ps zmm16, zmm4, zmm23          // r/6 + 1/2
        vfmadd213ps zmm17, zmm5, zmm23
        vfmadd213ps zmm18, zmm6, zmm23
        vfmadd213ps zmm19, zmm7, zmm23
        vfmadd213ps zmm16, zmm4, zmm22
        vfmadd213ps zmm17, zmm5, zmm22
        vfmadd213ps zmm18, zmm6, zmm22
        vfmadd213ps zmm19, zmm7, zmm22
        vfmadd213ps zmm16, zmm4, zmm22          // exp(r)
        vfmadd213ps zmm17, zmm5, zmm22
        vfmadd213ps zmm18, zmm6, zmm22
        vfmadd213ps zmm19, zmm7, zmm22
        vfmadd213ps zmm16, zmm12, zmm22         // 1 + exp(-2 y)
        vfmadd213ps zmm17, zmm13, zmm22
        vfmadd213ps zmm18, zmm14, zmm22
        vfmadd213ps zmm19, zmm15, zmm22
        vdivps zmm0, zmm0, zmm16                // GELU(g)
        vdivps zmm1, zmm1, zmm17
        vdivps zmm2, zmm2, zmm18
        vdivps zmm3, zmm3, zmm19
        vmulps zmm0, zmm0, [rsi + 64 * 0]       // x GELU(g)
        vmulps zmm1, zmm1, [rsi + 64 * 1]
        vmulps zmm2, zmm2, [rsi + 64 * 2]
        vmulps zmm3, zmm3, [rsi + 64 * 3]
        vmovups [rdi + 64 * 0], zmm0
        vmovups [rdi + 64 * 1], zmm1
        vmovups [rdi + 64 * 2], zmm2
        vmovups [rdi + 64 * 3], zmm3
        add rdi, 256
        add rsi, 256
        add rdx, 256
        dec rcx
        jnz 1b
9:
        vzeroupper
        ret
        .size sdasm_gated_gelu_avx512, .-sdasm_gated_gelu_avx512

        .section .note.GNU-stack, "", @progbits
