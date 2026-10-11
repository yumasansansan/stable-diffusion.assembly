// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
//     void sdasm_exponentials_avx2(float *s, int64_t n, const float *max,
//                                  double *total);
//
// The exponentials of attention's softmax (attention.cc): s is a panel of the
// scaled scores of n keys, n a multiple of eight, for sixteen queries, laid out
// [key][16] and 32-byte aligned; each score becomes exp(score - max[q]), q its
// query, and total[q] (a double) gains the sum of the query's exponentials,
// eight keys at a time summed in single precision, the even keys and the odd
// apart and then together.
//
// The exponential: x, no less than -87 (below which the exponential is under
// 1.2e-38, nothing beside the largest score's 1), is split as
// x = (n + f) ln(2) / 8 with n an integer, by n = x 8 / ln(2) rounded to the
// nearest (the sum with 1.5 * 2^23 rounds it, and leaves it in the low bits of
// the sum's representation) and r = x - n ln(2) / 8 in two steps of a high and
// a low part of ln(2) / 8; then exp(x) = 2^(n div 8) 2^((n mod 8) / 8) exp(r),
// the middle factor from a table of eight that the low three bits of n pick
// with vpermps, the first added to its exponent, and exp(r), |r| <= ln(2) / 16,
// by its Taylor polynomial of degree four. Over [-87, 0] it is within 1.6 units
// in the last place of the exact exponential, against 2.0 for ggml_v_expf.
//
// The C++ form (attention.cc, exponential() of scalar_math.h) reduces x by
// ln(2) alone and takes Cephes' polynomial of degree six, which the compiler
// vectorizes: four more steps to each chain of operations that each wait for
// the last. A P-core, with its large out-of-order window, runs many chains at
// once, but an E-core (Gracemont) holds few enough vector operations waiting
// that it runs barely two, and waits on the latency of the chain. Here the
// table of eight shortens the chain to some fifteen, and the instructions of
// two chains, two keys, are interleaved, which keeps an E-core's units busy.
// Measured on the i7-1360P, a block of keys in the first-level cache: 0.43
// ticks of the time-stamp counter an exponential on a P-core, three times as
// fast as the C++ form's 1.31; 1.8 on an E-core, 2.3 times as fast as its 4.1.

        .intel_syntax noprefix

        .section .rodata
        .p2align 5
.Ltable:        // 2^(k/8), k = 0..7
        .float 0x1p+0, 0x1.172b84p+0, 0x1.306fep+0, 0x1.4bfdaep+0
        .float 0x1.6a09e6p+0, 0x1.8ace54p+0, 0x1.ae89fap+0, 0x1.d5818ep+0
.Lc4:   .rept 8
        .float 0x1.555556p-5    // 1/24
        .endr
.Lc3:   .rept 8
        .float 0x1.555556p-3    // 1/6
        .endr
.Lc2:   .rept 8
        .float 0.5
        .endr
.Lclamp:
        .rept 8
        .float -87.0
        .endr
.Lhigh: .rept 8
        .float 0x1.62e43p-4     // ln(2)/8, its high part
        .endr
.Llow:  .rept 8
        .float -0x1.05c610p-32  // and what it leaves
        .endr
.Lexponent:
        .rept 8
        .long 0xff800000
        .endr
        .p2align 2
.Lmagic:
        .float 0x1.8p23
.Lscale:
        .float 11.541560327111707       // 8 / ln(2)
.Lone:  .float 1.0

        .text

// One key of a chain: the score at [rax + \offset] in ymm\x, z in ymm\z, the
// table's factor in ymm\t, exp(r) in ymm\p; written back, and added to the sum
// in ymm\sum. ymm15 holds the half's maxima, ymm14 1.5 * 2^23, ymm13
// 8 / ln(2), ymm12 the table, ymm11 1, ymm10 1/2.
//
// The two chains' steps are interleaved by the macro below, a step of one and
// then the same step of the other.
.macro TWO a_x, a_z, a_t, a_p, a_off, a_sum, b_x, b_z, b_t, b_p, b_off, b_sum
        vmovaps ymm\a_x, [rax + \a_off]
        vmovaps ymm\b_x, [rax + \b_off]
        vsubps ymm\a_x, ymm\a_x, ymm15
        vsubps ymm\b_x, ymm\b_x, ymm15
        vmaxps ymm\a_x, ymm\a_x, [rip + .Lclamp]
        vmaxps ymm\b_x, ymm\b_x, [rip + .Lclamp]
        vmovaps ymm\a_z, ymm13
        vmovaps ymm\b_z, ymm13
        vfmadd213ps ymm\a_z, ymm\a_x, ymm14     // z = x 8/ln(2) + 1.5 2^23
        vfmadd213ps ymm\b_z, ymm\b_x, ymm14
        vsubps ymm\a_t, ymm\a_z, ymm14          // n
        vsubps ymm\b_t, ymm\b_z, ymm14
        vfnmadd231ps ymm\a_x, ymm\a_t, [rip + .Lhigh]   // r = x - n ln(2)/8
        vfnmadd231ps ymm\b_x, ymm\b_t, [rip + .Lhigh]
        vfnmadd231ps ymm\a_x, ymm\a_t, [rip + .Llow]
        vfnmadd231ps ymm\b_x, ymm\b_t, [rip + .Llow]
        vpermps ymm\a_t, ymm\a_z, ymm12         // 2^((n mod 8)/8)
        vpermps ymm\b_t, ymm\b_z, ymm12
        vpslld ymm\a_z, ymm\a_z, 20             // n div 8, to the exponent
        vpslld ymm\b_z, ymm\b_z, 20
        vpand ymm\a_z, ymm\a_z, [rip + .Lexponent]
        vpand ymm\b_z, ymm\b_z, [rip + .Lexponent]
        vpaddd ymm\a_t, ymm\a_t, ymm\a_z
        vpaddd ymm\b_t, ymm\b_t, ymm\b_z
        vmovaps ymm\a_p, [rip + .Lc4]
        vmovaps ymm\b_p, [rip + .Lc4]
        vfmadd213ps ymm\a_p, ymm\a_x, [rip + .Lc3]
        vfmadd213ps ymm\b_p, ymm\b_x, [rip + .Lc3]
        vfmadd213ps ymm\a_p, ymm\a_x, ymm10
        vfmadd213ps ymm\b_p, ymm\b_x, ymm10
        vfmadd213ps ymm\a_p, ymm\a_x, ymm11
        vfmadd213ps ymm\b_p, ymm\b_x, ymm11
        vfmadd213ps ymm\a_p, ymm\a_x, ymm11     // exp(r)
        vfmadd213ps ymm\b_p, ymm\b_x, ymm11
        vmulps ymm\a_p, ymm\a_p, ymm\a_t
        vmulps ymm\b_p, ymm\b_p, ymm\b_t
        vmovaps [rax + \a_off], ymm\a_p
        vmovaps [rax + \b_off], ymm\b_p
        vaddps ymm\a_sum, ymm\a_sum, ymm\a_p
        vaddps ymm\b_sum, ymm\b_sum, ymm\b_p
.endm

        .globl sdasm_exponentials_avx2
        .type sdasm_exponentials_avx2, @function
        .p2align 5
sdasm_exponentials_avx2:
        vbroadcastss ymm14, [rip + .Lmagic]
        vbroadcastss ymm13, [rip + .Lscale]
        vmovaps ymm12, [rip + .Ltable]
        vbroadcastss ymm11, [rip + .Lone]
        vmovaps ymm10, [rip + .Lc2]
        // The left eight queries over all the keys, then the right eight: r8
        // the first query of the half.
        xor r8d, r8d
1:
        vmovups ymm15, [rdx + 4*r8]
        lea rax, [rdi + 4*r8]
        mov r9, rsi
        shr r9, 3
        jz 3f
2:
        // Eight keys: the even ones' sum in ymm8, the odd ones' in ymm9.
        vxorps xmm8, xmm8, xmm8
        vxorps xmm9, xmm9, xmm9
        TWO 0, 1, 2, 3, 0, 8,    4, 5, 6, 7, 64, 9
        TWO 0, 1, 2, 3, 128, 8,  4, 5, 6, 7, 192, 9
        TWO 0, 1, 2, 3, 256, 8,  4, 5, 6, 7, 320, 9
        TWO 0, 1, 2, 3, 384, 8,  4, 5, 6, 7, 448, 9
        vaddps ymm8, ymm8, ymm9
        vcvtps2pd ymm0, xmm8
        vextractf128 xmm8, ymm8, 1
        vcvtps2pd ymm1, xmm8
        vaddpd ymm0, ymm0, [rcx + 8*r8]
        vaddpd ymm1, ymm1, [rcx + 8*r8 + 32]
        vmovupd [rcx + 8*r8], ymm0
        vmovupd [rcx + 8*r8 + 32], ymm1
        add rax, 512
        dec r9
        jnz 2b
3:
        add r8, 8
        cmp r8, 16
        jne 1b
        vzeroupper
        ret
        .size sdasm_exponentials_avx2, .-sdasm_exponentials_avx2

        .section .note.GNU-stack, "", @progbits
