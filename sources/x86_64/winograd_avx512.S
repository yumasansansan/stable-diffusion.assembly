// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The transforms of Winograd's F(2x2, 3x3) (winograd.cc) for AVX-512 F, VL,
// BW and DQ, sixteen tiles to a register, in their own order:
//
//     void sdasm_winograd_inputs_avx512(const Winograd_inputs *job);
//
// What sdasm_winograd_inputs_avx2 computes (winograd_avx2.S): of two rows A
// and B of the input, d[j] = row[2l + j] for tile l, the columns of B^T d that
// each s takes, and A plus or minus B. The thirty-four numbers of a row are
// four loads of sixteen, at 0, 16, 2 and 18, each masked to the image by the
// lanes the job gives (k-registers); the even and the odd numbers of two of
// them are picked out by vpermi2ps, which takes from both at once and leaves
// the sixteen tiles in their order.
//
//     void sdasm_winograd_outputs_avx512(const float *m, int64_t stride,
//                                        float *row0, float *row1,
//                                        int64_t groups, int64_t add);
//
// Y = A^T M A for groups of sixteen tiles of a row of tiles, one output
// channel, M of a group's tiles at m + xi * stride bytes; the two outputs
// across of each tile interleaved by vpermt2ps into thirty-two of each row
// of the output, written or added.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 6
.Leven: .long 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
.Lodd:  .long 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
.Llow:  .long 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
.Lhigh: .long 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
        .p2align 2
.Lsign: .long 0x80000000

        .text

// ---------------------------------------------------------------- inputs

.set J_DST, 0
.set J_STRIDE, 8
.set J_A, 16
.set J_A_STEP, 24
.set J_B, 32
.set J_B_STEP, 40
.set J_CHANNELS, 48
.set J_SUBTRACT, 64
.set J_LANES, 72                // four 16-bit masks: the loads at 0, 16, 2, 18

// The four columns of B^T d of the sixteen tiles of the row at \row, into
// zmm\c0..zmm\c3: loads into zmm\t0..zmm\t3, masked by k1..k4.
.macro COLUMNS row, c0, c1, c2, c3, t0, t1, t2, t3
        vmovups zmm\t0{k1}{z}, [\row]
        vmovups zmm\t1{k2}{z}, [\row + 64]
        vmovups zmm\t2{k3}{z}, [\row + 8]
        vmovups zmm\t3{k4}{z}, [\row + 72]
        vmovaps zmm\c0, zmm30
        vpermi2ps zmm\c0, zmm\t0, zmm\t1        // e: d0
        vmovaps zmm\c1, zmm31
        vpermi2ps zmm\c1, zmm\t0, zmm\t1        // o: d1
        vmovaps zmm\t0, zmm30
        vpermi2ps zmm\t0, zmm\t2, zmm\t3        // e2: d2
        vmovaps zmm\t1, zmm31
        vpermi2ps zmm\t1, zmm\t2, zmm\t3        // o2: d3
        vsubps zmm\c2, zmm\t0, zmm\c1           // d2 - d1
        vsubps zmm\c3, zmm\c1, zmm\t1           // d1 - d3
        vaddps zmm\c1, zmm\c1, zmm\t0           // d1 + d2
        vsubps zmm\c0, zmm\c0, zmm\t0           // d0 - d2
.endm

        .globl sdasm_winograd_inputs_avx512
        .type sdasm_winograd_inputs_avx512, @function
        .p2align 5
sdasm_winograd_inputs_avx512:
        mov rax, [rdi + J_DST]
        mov rdx, [rdi + J_STRIDE]
        lea r10, [rdx + 2*rdx]
        mov rsi, [rdi + J_A]
        mov rcx, [rdi + J_B]
        mov r8, [rdi + J_CHANNELS]
        kmovw k1, [rdi + J_LANES + 0]
        kmovw k2, [rdi + J_LANES + 2]
        kmovw k3, [rdi + J_LANES + 4]
        kmovw k4, [rdi + J_LANES + 6]
        vmovups zmm30, [rip + .Leven]
        vmovups zmm31, [rip + .Lodd]
        vpxord zmm29, zmm29, zmm29
        cmp qword ptr [rdi + J_SUBTRACT], 0
        je 1f
        vpbroadcastd zmm29, [rip + .Lsign]
1:
        test r8, r8
        jz 9f
2:
        COLUMNS rsi, 0, 1, 2, 3, 4, 5, 6, 7
        COLUMNS rcx, 8, 9, 10, 11, 4, 5, 6, 7
        vpxord zmm8, zmm8, zmm29
        vpxord zmm9, zmm9, zmm29
        vpxord zmm10, zmm10, zmm29
        vpxord zmm11, zmm11, zmm29
        vaddps zmm0, zmm0, zmm8
        vaddps zmm1, zmm1, zmm9
        vaddps zmm2, zmm2, zmm10
        vaddps zmm3, zmm3, zmm11
        vmovups [rax], zmm0
        vmovups [rax + rdx], zmm1
        vmovups [rax + 2*rdx], zmm2
        vmovups [rax + r10], zmm3
        add rax, 64
        add rsi, [rdi + J_A_STEP]
        add rcx, [rdi + J_B_STEP]
        dec r8
        jnz 2b
9:
        vzeroupper
        ret
        .size sdasm_winograd_inputs_avx512, .-sdasm_winograd_inputs_avx512

// ---------------------------------------------------------------- outputs

// a0[j] into zmm\a0 and a1[j] into zmm\a1 from the four rows of M, at rdi,
// r11, rax and r9, column j at \offset.
.macro COLUMN a0, a1, offset
        vmovups zmm8, [rdi \offset]
        vmovups zmm9, [r11 \offset]
        vmovups zmm10, [rax \offset]
        vmovups zmm11, [r9 \offset]
        vaddps zmm\a0, zmm8, zmm9
        vaddps zmm\a0, zmm\a0, zmm10
        vsubps zmm12, zmm9, zmm10
        vsubps zmm\a1, zmm12, zmm11
.endm

.macro OUTPUTS add
1:
        COLUMN 0, 4, ""
        COLUMN 1, 5, "+ rsi"
        COLUMN 2, 6, "+ 2*rsi"
        COLUMN 3, 7, "+ r10"
        vaddps zmm8, zmm0, zmm1                 // y00
        vaddps zmm8, zmm8, zmm2
        vsubps zmm9, zmm1, zmm2                 // y01
        vsubps zmm9, zmm9, zmm3
        vaddps zmm10, zmm4, zmm5                // y10
        vaddps zmm10, zmm10, zmm6
        vsubps zmm11, zmm5, zmm6                // y11
        vsubps zmm11, zmm11, zmm7
        vmovaps zmm0, zmm8
        vpermt2ps zmm0, zmm30, zmm9             // tiles 0..7 of row 0: x 0..15
        vpermt2ps zmm8, zmm31, zmm9             // tiles 8..15: x 16..31
        vmovaps zmm2, zmm10
        vpermt2ps zmm2, zmm30, zmm11
        vpermt2ps zmm10, zmm31, zmm11
.if \add
        vaddps zmm0, zmm0, [rdx]
        vaddps zmm8, zmm8, [rdx + 64]
        vaddps zmm2, zmm2, [rcx]
        vaddps zmm10, zmm10, [rcx + 64]
.endif
        vmovups [rdx], zmm0
        vmovups [rdx + 64], zmm8
        vmovups [rcx], zmm2
        vmovups [rcx + 64], zmm10
        add rdi, 64
        add r11, 64
        add rax, 64
        add r9, 64
        add rdx, 128
        add rcx, 128
        dec r8
        jnz 1b
.endm

        .globl sdasm_winograd_outputs_avx512
        .type sdasm_winograd_outputs_avx512, @function
        .p2align 5
// rdi m, rsi stride, rdx row0, rcx row1, r8 groups, r9 add
sdasm_winograd_outputs_avx512:
        test r8, r8
        jz 9f
        vmovups zmm30, [rip + .Llow]
        vmovups zmm31, [rip + .Lhigh]
        mov r10, r9
        lea r11, [rdi + 4*rsi]
        lea rax, [rdi + 8*rsi]
        lea r9, [r11 + 8*rsi]
        test r10, r10
        lea r10, [rsi + 2*rsi]
        jnz 5f
        OUTPUTS 0
        jmp 9f
5:
        OUTPUTS 1
9:
        vzeroupper
        ret
        .size sdasm_winograd_outputs_avx512, .-sdasm_winograd_outputs_avx512

        .section .note.GNU-stack, "", @progbits
