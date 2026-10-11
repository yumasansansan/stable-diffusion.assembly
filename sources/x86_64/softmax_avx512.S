// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_exponentials_avx512(float *s, int64_t n, const float *max,
//                                    double *total);
//
// The exponentials of attention's softmax (softmax_avx2.S) for AVX-512: a key
// of sixteen queries is one register, exp(score - max) of all sixteen at once,
// and the sums of each eight keys, the even keys and the odd apart and then
// together, added to total in double precision.
//
// With sixteen lanes to a permutation, the table of the exponential has sixteen
// entries: x, no less than -87, is split as x = (n + f) ln(2) / 16, n = x 16 /
// ln(2) rounded to the nearest by the sum with 1.5 * 2^23, r = x - n ln(2) / 16
// in two steps of a high and a low part, |r| <= ln(2) / 32; then exp(x) =
// 2^(n div 16) 2^((n mod 16) / 16) exp(r), the middle factor from the table by
// vpermps, the first added to its exponent, and exp(r) by its Taylor
// polynomial of degree three, whose error is under r^4 / 24 < 1e-8 -- an
// operation fewer than the eight-entry table of AVX2 takes. Four keys are
// computed at once, their instructions interleaved, for the latency of each
// chain of fifteen operations.
//
//     void sdasm_exponentials_row_avx512(float *s, int64_t n, float max,
//                                        double *total);
//
// The same exponentials of a row of n scores of one query, n a multiple of
// 64, as attention on AMX's tiles lays them out (attention.cc): exp(score -
// max) in place, four registers of sixteen keys at a time, their sum in single
// precision, then in double precision, added to *total.
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
.Lhigh: .float 0x1.62e43p-5             // ln(2) / 16, its high part
.Llow:  .float -0x1.05c61p-33           // and what it leaves
.Lclamp: .float -87.0
.Lc3:   .float 0x1.555556p-3            // 1/6
.Lc2:   .float 0.5
.Lone:  .float 1.0
.Lexponent: .long 0xff800000

        .text

// Constants: zmm31 the table, zmm30 1.5 2^23, zmm29 16 / ln(2), zmm28 and
// zmm27 the high and the low part of ln(2) / 16, zmm26 -87, zmm25 1/6, zmm24
// 1/2, zmm23 1, zmm22 the bits of the exponent; zmm21 the maxima of the
// sixteen queries.
//
// Four keys at once, \k0 to \k0 + 3 (in rows of 64 bytes from rax), the steps
// of their four chains interleaved, key c in zmm c (x, then r), zmm 4 + c (z),
// zmm 8 + c (the table's factor) and zmm 12 + c (exp(r), then the
// exponential); their exponentials added, the even keys' to zmm\sum0, the odd
// keys' to zmm\sum1.
.macro FOUR k0, sum0, sum1
        vmovups zmm0, [rax + 64 * (\k0 + 0)]
        vmovups zmm1, [rax + 64 * (\k0 + 1)]
        vmovups zmm2, [rax + 64 * (\k0 + 2)]
        vmovups zmm3, [rax + 64 * (\k0 + 3)]
        vsubps zmm0, zmm0, zmm21
        vsubps zmm1, zmm1, zmm21
        vsubps zmm2, zmm2, zmm21
        vsubps zmm3, zmm3, zmm21
        vmaxps zmm0, zmm0, zmm26
        vmaxps zmm1, zmm1, zmm26
        vmaxps zmm2, zmm2, zmm26
        vmaxps zmm3, zmm3, zmm26
        vmovaps zmm4, zmm29
        vmovaps zmm5, zmm29
        vmovaps zmm6, zmm29
        vmovaps zmm7, zmm29
        vfmadd213ps zmm4, zmm0, zmm30           // z = x 16/ln(2) + 1.5 2^23
        vfmadd213ps zmm5, zmm1, zmm30
        vfmadd213ps zmm6, zmm2, zmm30
        vfmadd213ps zmm7, zmm3, zmm30
        vsubps zmm8, zmm4, zmm30                // n
        vsubps zmm9, zmm5, zmm30
        vsubps zmm10, zmm6, zmm30
        vsubps zmm11, zmm7, zmm30
        vfnmadd231ps zmm0, zmm8, zmm28          // r = x - n ln(2)/16
        vfnmadd231ps zmm1, zmm9, zmm28
        vfnmadd231ps zmm2, zmm10, zmm28
        vfnmadd231ps zmm3, zmm11, zmm28
        vfnmadd231ps zmm0, zmm8, zmm27
        vfnmadd231ps zmm1, zmm9, zmm27
        vfnmadd231ps zmm2, zmm10, zmm27
        vfnmadd231ps zmm3, zmm11, zmm27
        vpermps zmm8, zmm4, zmm31               // 2^((n mod 16)/16)
        vpermps zmm9, zmm5, zmm31
        vpermps zmm10, zmm6, zmm31
        vpermps zmm11, zmm7, zmm31
        vpslld zmm4, zmm4, 19                   // n div 16, to the exponent
        vpslld zmm5, zmm5, 19
        vpslld zmm6, zmm6, 19
        vpslld zmm7, zmm7, 19
        vpandd zmm4, zmm4, zmm22
        vpandd zmm5, zmm5, zmm22
        vpandd zmm6, zmm6, zmm22
        vpandd zmm7, zmm7, zmm22
        vpaddd zmm8, zmm8, zmm4
        vpaddd zmm9, zmm9, zmm5
        vpaddd zmm10, zmm10, zmm6
        vpaddd zmm11, zmm11, zmm7
        vmovaps zmm12, zmm25
        vmovaps zmm13, zmm25
        vmovaps zmm14, zmm25
        vmovaps zmm15, zmm25
        vfmadd213ps zmm12, zmm0, zmm24          // r/6 + 1/2
        vfmadd213ps zmm13, zmm1, zmm24
        vfmadd213ps zmm14, zmm2, zmm24
        vfmadd213ps zmm15, zmm3, zmm24
        vfmadd213ps zmm12, zmm0, zmm23          // (...) r + 1
        vfmadd213ps zmm13, zmm1, zmm23
        vfmadd213ps zmm14, zmm2, zmm23
        vfmadd213ps zmm15, zmm3, zmm23
        vfmadd213ps zmm12, zmm0, zmm23          // exp(r)
        vfmadd213ps zmm13, zmm1, zmm23
        vfmadd213ps zmm14, zmm2, zmm23
        vfmadd213ps zmm15, zmm3, zmm23
        vmulps zmm12, zmm12, zmm8
        vmulps zmm13, zmm13, zmm9
        vmulps zmm14, zmm14, zmm10
        vmulps zmm15, zmm15, zmm11
        vmovups [rax + 64 * (\k0 + 0)], zmm12
        vmovups [rax + 64 * (\k0 + 1)], zmm13
        vmovups [rax + 64 * (\k0 + 2)], zmm14
        vmovups [rax + 64 * (\k0 + 3)], zmm15
        vaddps zmm\sum0, zmm\sum0, zmm12
        vaddps zmm\sum1, zmm\sum1, zmm13
        vaddps zmm\sum0, zmm\sum0, zmm14
        vaddps zmm\sum1, zmm\sum1, zmm15
.endm

        .globl sdasm_exponentials_avx512
        .type sdasm_exponentials_avx512, @function
        .p2align 5
// rdi s, rsi n, rdx max, rcx total
sdasm_exponentials_avx512:
        vmovups zmm31, [rip + .Ltable]
        vbroadcastss zmm30, [rip + .Lmagic]
        vbroadcastss zmm29, [rip + .Lscale]
        vbroadcastss zmm28, [rip + .Lhigh]
        vbroadcastss zmm27, [rip + .Llow]
        vbroadcastss zmm26, [rip + .Lclamp]
        vbroadcastss zmm25, [rip + .Lc3]
        vbroadcastss zmm24, [rip + .Lc2]
        vbroadcastss zmm23, [rip + .Lone]
        vpbroadcastd zmm22, [rip + .Lexponent]
        vmovups zmm21, [rdx]
        mov rax, rdi
        mov r9, rsi
        shr r9, 3
        jz 9f
1:
        // Eight keys: the even ones' sums in zmm16, the odd ones' in zmm17.
        vpxord zmm16, zmm16, zmm16
        vpxord zmm17, zmm17, zmm17
        FOUR 0, 16, 17
        FOUR 4, 16, 17
        vaddps zmm16, zmm16, zmm17
        vcvtps2pd zmm18, ymm16
        vextractf32x8 ymm16, zmm16, 1
        vcvtps2pd zmm19, ymm16
        vaddpd zmm18, zmm18, [rcx]
        vaddpd zmm19, zmm19, [rcx + 64]
        vmovupd [rcx], zmm18
        vmovupd [rcx + 64], zmm19
        add rax, 512
        dec r9
        jnz 1b
9:
        vzeroupper
        ret
        .size sdasm_exponentials_avx512, .-sdasm_exponentials_avx512

        .globl sdasm_exponentials_row_avx512
        .type sdasm_exponentials_row_avx512, @function
        .p2align 5
// rdi s, rsi n, xmm0 max, rdx total
sdasm_exponentials_row_avx512:
        vbroadcastss zmm21, xmm0
        vmovups zmm31, [rip + .Ltable]
        vbroadcastss zmm30, [rip + .Lmagic]
        vbroadcastss zmm29, [rip + .Lscale]
        vbroadcastss zmm28, [rip + .Lhigh]
        vbroadcastss zmm27, [rip + .Llow]
        vbroadcastss zmm26, [rip + .Lclamp]
        vbroadcastss zmm25, [rip + .Lc3]
        vbroadcastss zmm24, [rip + .Lc2]
        vbroadcastss zmm23, [rip + .Lone]
        vpbroadcastd zmm22, [rip + .Lexponent]
        vxorpd zmm18, zmm18, zmm18              // the sums in double precision
        vxorpd zmm19, zmm19, zmm19
        mov rax, rdi
        mov r9, rsi
        shr r9, 6
        jz 2f
1:
        vpxord zmm16, zmm16, zmm16
        vpxord zmm17, zmm17, zmm17
        FOUR 0, 16, 17
        vaddps zmm16, zmm16, zmm17
        vcvtps2pd zmm20, ymm16
        vaddpd zmm18, zmm18, zmm20
        vextractf32x8 ymm16, zmm16, 1
        vcvtps2pd zmm20, ymm16
        vaddpd zmm19, zmm19, zmm20
        add rax, 256
        dec r9
        jnz 1b
2:
        vaddpd zmm18, zmm18, zmm19
        vextractf64x4 ymm19, zmm18, 1
        vaddpd ymm18, ymm18, ymm19
        vextractf64x2 xmm19, ymm18, 1
        vaddpd xmm18, xmm18, xmm19
        vunpckhpd xmm19, xmm18, xmm18
        vaddsd xmm18, xmm18, xmm19
        vaddsd xmm18, xmm18, qword ptr [rdx]
        vmovsd qword ptr [rdx], xmm18
        vzeroupper
        ret
        .size sdasm_exponentials_row_avx512, .-sdasm_exponentials_row_avx512

        .section .note.GNU-stack, "", @progbits
