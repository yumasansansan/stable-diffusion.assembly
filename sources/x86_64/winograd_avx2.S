// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The transforms of Winograd's F(2x2, 3x3) (winograd.cc) for AVX2:
//
//     void sdasm_winograd_inputs_avx2(const Winograd_inputs *job);
//
// Four elements xi = (r, s), s = 0..3, of the transformed inputs of sixteen
// tiles of a row of tiles, channel after channel: for each, two rows of the
// input, A and B (zeros when outside the image), give d[j] = row[2l + j] for
// tile l, j = 0..3; the column of B^T d that each s takes of each row --
// d0 - d2, d1 + d2, d2 - d1, d1 - d3 -- and of the two rows A plus or minus B,
// as row r of B^T has it. The numbers 2l + j are taken from two vectors of
// eight in each half of them, the even ones and the odd, by one vshufps each,
// which leaves the tiles of each eight in the order 0, 1, 4, 5, 2, 3, 6, 7 --
// the order the packing keeps, and the transform of the outputs undoes.
// Inputs past the edges of the image are masked out of the loads (vmaskmovps).
//
//     void sdasm_winograd_outputs_avx2(const float *m, int64_t stride,
//                                      float *row0, float *row1,
//                                      int64_t groups, int64_t add);
//
// Y = A^T M A for groups of eight tiles of a row of tiles, one output channel:
// M of a group's tiles at m + xi * stride bytes, eight floats, in the order the
// packing leaves them; the two outputs across of each tile, of both rows,
// interleaved into sixteen of each row of the output -- which the interleaving
// within each half of the vectors puts in order, the tiles being in that one --
// and written to row0 and row1, or added to what is there when add is not zero.
//
//     void sdasm_winograd_weights_avx2(float *dst, int64_t stride,
//                                      const float *g, int64_t channels);
//
// G g G^T of the kernels of a panel of six output channels, channel after
// channel: g the kernels widened, [channel][3 * 3][6], dst the panel of each
// xi, xi at dst + xi * stride bytes, [channel][6]. Each channel's six lanes
// are stored as eight, the last two the next channel's, which it writes over;
// the last channel's as six.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 2
.Lhalf: .float 0.5
.Lsign: .long 0x80000000

        .text

// ---------------------------------------------------------------- inputs

// The job (winograd.cc, struct Winograd_inputs), at rdi.
.set J_DST, 0
.set J_STRIDE, 8
.set J_A, 16
.set J_A_STEP, 24
.set J_B, 32
.set J_B_STEP, 40
.set J_CHANNELS, 48
.set J_MASKS, 56
.set J_SUBTRACT, 64
.set J_A_OFFSET, 80               // bytes from row A to the tiles' first input
.set J_B_OFFSET, 88

// Eight numbers of a row at [\row + 4 * \offset] into ymm\v, masked by the
// mask of the load, the \index-th of eight at [r11], or not masked.
// (The arguments are expressions, put in parentheses where they are used.)
.macro LOADM v, row, offset, index
        vmovdqu ymm15, [r11 + 32 * (\index)]
        vmaskmovps ymm\v, ymm15, [\row + 4 * (\offset)]
.endm
.macro LOADU v, row, offset, index
        vmovups ymm\v, [\row + 4 * (\offset)]
.endm

// The four columns of B^T d of eight tiles of one row, half \h of sixteen,
// into ymm\c0..ymm\c3 (four registers in a row, from \c0), with ymm\t0..ymm\t3
// for the loads.
.macro COLUMNS load, row, h, c0, c1, c2, c3, t0, t1, t2, t3
        \load \t0, \row, 16 * \h + 0, 4 * \h + 0
        \load \t1, \row, 16 * \h + 8, 4 * \h + 1
        \load \t2, \row, 16 * \h + 2, 4 * \h + 2
        \load \t3, \row, 16 * \h + 10, 4 * \h + 3
        vshufps ymm\c0, ymm\t0, ymm\t1, 0x88    // e: d0
        vshufps ymm\c1, ymm\t0, ymm\t1, 0xdd    // o: d1
        vshufps ymm\t0, ymm\t2, ymm\t3, 0x88    // e2: d2
        vshufps ymm\t1, ymm\t2, ymm\t3, 0xdd    // o2: d3
        vsubps ymm\c2, ymm\t0, ymm\c1           // d2 - d1
        vsubps ymm\c3, ymm\c1, ymm\t1           // d1 - d3
        vaddps ymm\c1, ymm\c1, ymm\t0           // d1 + d2
        vsubps ymm\c0, ymm\c0, ymm\t0           // d0 - d2
.endm

// One channel: both halves of both rows, A plus B with B's sign flipped by
// ymm14 (the sign bit, or nothing), stored at rax + s * rdx + 32 * h.
.macro CHANNEL load
.irp h, 0, 1
        COLUMNS \load, rsi, \h, 0, 1, 2, 3, 4, 5, 6, 7
        COLUMNS \load, rcx, \h, 8, 9, 10, 11, 4, 5, 6, 7
        vxorps ymm8, ymm8, ymm14
        vxorps ymm9, ymm9, ymm14
        vxorps ymm10, ymm10, ymm14
        vxorps ymm11, ymm11, ymm14
        vaddps ymm0, ymm0, ymm8
        vaddps ymm1, ymm1, ymm9
        vaddps ymm2, ymm2, ymm10
        vaddps ymm3, ymm3, ymm11
        vmovups [rax + 32 * (\h)], ymm0
        vmovups [rax + rdx + 32 * (\h)], ymm1
        vmovups [rax + 2*rdx + 32 * (\h)], ymm2
        vmovups [rax + r10 + 32 * (\h)], ymm3
.endr
.endm

        .globl sdasm_winograd_inputs_avx2
        .type sdasm_winograd_inputs_avx2, @function
        .p2align 5
sdasm_winograd_inputs_avx2:
        mov rax, [rdi + J_DST]
        mov rdx, [rdi + J_STRIDE]
        lea r10, [rdx + 2*rdx]
        // The rows at the tiles' first inputs, which may lie before the
        // rows: the addresses are formed here, and nothing outside the
        // image is read (the masks).
        mov rsi, [rdi + J_A]
        add rsi, [rdi + J_A_OFFSET]
        mov rcx, [rdi + J_B]
        add rcx, [rdi + J_B_OFFSET]
        mov r8, [rdi + J_CHANNELS]
        mov r11, [rdi + J_MASKS]
        vxorps xmm14, xmm14, xmm14
        cmp qword ptr [rdi + J_SUBTRACT], 0
        je 1f
        vbroadcastss ymm14, [rip + .Lsign]
1:
        test r8, r8
        jz 9f
        test r11, r11
        jnz 5f
2:
        CHANNEL LOADU
        add rax, 64
        add rsi, [rdi + J_A_STEP]
        add rcx, [rdi + J_B_STEP]
        dec r8
        jnz 2b
        jmp 9f
5:
        CHANNEL LOADM
        add rax, 64
        add rsi, [rdi + J_A_STEP]
        add rcx, [rdi + J_B_STEP]
        dec r8
        jnz 5b
9:
        vzeroupper
        ret
        .size sdasm_winograd_inputs_avx2, .-sdasm_winograd_inputs_avx2

// ---------------------------------------------------------------- outputs

// a0[j] = (M0j + M1j) + M2j into ymm\a0 and a1[j] = (M1j - M2j) - M3j into
// ymm\a1, M_ij the products of xi = 4 i + j: rows i = 0..3 of M at rdi, r11,
// rax and r9, column j at \offset from each.
.macro COLUMN a0, a1, offset
        vmovups ymm8, [rdi \offset]
        vmovups ymm9, [r11 \offset]
        vmovups ymm10, [rax \offset]
        vmovups ymm11, [r9 \offset]
        vaddps ymm\a0, ymm8, ymm9
        vaddps ymm\a0, ymm\a0, ymm10
        vsubps ymm12, ymm9, ymm10
        vsubps ymm\a1, ymm12, ymm11
.endm

// Groups of eight tiles, the outputs stored, or added to the rows' (\add).
.macro OUTPUTS add
1:
        COLUMN 0, 4, ""
        COLUMN 1, 5, "+ rsi"
        COLUMN 2, 6, "+ 2*rsi"
        COLUMN 3, 7, "+ r10"
        vaddps ymm8, ymm0, ymm1                 // y00 = (a0[0] + a0[1]) + a0[2]
        vaddps ymm8, ymm8, ymm2
        vsubps ymm9, ymm1, ymm2                 // y01 = (a0[1] - a0[2]) - a0[3]
        vsubps ymm9, ymm9, ymm3
        vaddps ymm10, ymm4, ymm5                // y10 = (a1[0] + a1[1]) + a1[2]
        vaddps ymm10, ymm10, ymm6
        vsubps ymm11, ymm5, ymm6                // y11 = (a1[1] - a1[2]) - a1[3]
        vsubps ymm11, ymm11, ymm7
        vunpcklps ymm0, ymm8, ymm9              // tiles 0..3 of row 0: x 0..7
        vunpckhps ymm1, ymm8, ymm9              // tiles 4..7: x 8..15
        vunpcklps ymm2, ymm10, ymm11
        vunpckhps ymm3, ymm10, ymm11
.if \add
        vaddps ymm0, ymm0, [rdx]
        vaddps ymm1, ymm1, [rdx + 32]
        vaddps ymm2, ymm2, [rcx]
        vaddps ymm3, ymm3, [rcx + 32]
.endif
        vmovups [rdx], ymm0
        vmovups [rdx + 32], ymm1
        vmovups [rcx], ymm2
        vmovups [rcx + 32], ymm3
        add rdi, 32
        add r11, 32
        add rax, 32
        add r9, 32
        add rdx, 64
        add rcx, 64
        dec r8
        jnz 1b
.endm

        .globl sdasm_winograd_outputs_avx2
        .type sdasm_winograd_outputs_avx2, @function
        .p2align 5
// rdi m, rsi stride, rdx row0, rcx row1, r8 groups, r9 add
sdasm_winograd_outputs_avx2:
        test r8, r8
        jz 9f
        mov r10, r9                             // add, until r9 is a row of M
        lea r11, [rdi + 4*rsi]                  // row 1 of M: xi 4..7
        lea rax, [rdi + 8*rsi]                  // row 2: xi 8..11
        lea r9, [r11 + 8*rsi]                   // row 3: xi 12..15
        test r10, r10
        lea r10, [rsi + 2*rsi]                  // 3 stride, column 3
        jnz 5f
        OUTPUTS 0
        jmp 9f
5:
        OUTPUTS 1
9:
        vzeroupper
        ret
        .size sdasm_winograd_outputs_avx2, .-sdasm_winograd_outputs_avx2

// ---------------------------------------------------------------- weights

// G g of column j of the kernels: g0, g1, g2 the column's rows, at
// [rdx + 4 * 6 * (3 * row + j)]; t0j = g0 into ymm\t0, t1j = 0.5 ((g0 + g1)
// + g2) into ymm\t1, t2j = 0.5 ((g0 - g1) + g2) into ymm\t2, t3j = g2 into
// ymm\t3. ymm15 holds 0.5.
.macro GG j, t0, t1, t2, t3
        vmovups ymm\t0, [rdx + 4 * 6 * (0 + (\j))]
        vmovups ymm14, [rdx + 4 * 6 * (3 + (\j))]
        vmovups ymm\t3, [rdx + 4 * 6 * (6 + (\j))]
        vaddps ymm\t1, ymm\t0, ymm14
        vaddps ymm\t1, ymm\t1, ymm\t3
        vmulps ymm\t1, ymm\t1, ymm15
        vsubps ymm\t2, ymm\t0, ymm14
        vaddps ymm\t2, ymm\t2, ymm\t3
        vmulps ymm\t2, ymm\t2, ymm15
.endm

// The four xi of row i of (G g) G^T from a = ti0, b = ti1, d = ti2: a,
// 0.5 ((a + b) + d), 0.5 ((a - b) + d), d; stored at rax + (4 i + s) * rsi,
// by \store (eight floats, or six).
.macro ROW i, a, b, d, store
        \store \a, 4 * \i + 0
        vaddps ymm14, ymm\a, ymm\b
        vaddps ymm14, ymm14, ymm\d
        vmulps ymm14, ymm14, ymm15
        \store 14, 4 * \i + 1
        vsubps ymm14, ymm\a, ymm\b
        vaddps ymm14, ymm14, ymm\d
        vmulps ymm14, ymm14, ymm15
        \store 14, 4 * \i + 2
        \store \d, 4 * \i + 3
.endm

// ymm\v to the panel of xi: rax + xi * rsi, xi * rsi from r8 = xi * rsi
// computed afresh (imul) -- six of the eight lanes, or all eight.
.macro STORE8 v, xi
        imul r8, rsi, (\xi)
        vmovups [rax + r8], ymm\v
.endm
.macro STORE6 v, xi
        imul r8, rsi, (\xi)
        vmovups [rax + r8], xmm\v
        vextractf128 xmm13, ymm\v, 1
        vmovlps [rax + r8 + 16], xmm13
.endm

// One channel: t into ymm0..ymm11 (t[i][j] in ymm(4 j + i)), then the sixteen
// xi.
.macro WEIGHTS store
        GG 0, 0, 1, 2, 3
        GG 1, 4, 5, 6, 7
        GG 2, 8, 9, 10, 11
        ROW 0, 0, 4, 8, \store
        ROW 1, 1, 5, 9, \store
        ROW 2, 2, 6, 10, \store
        ROW 3, 3, 7, 11, \store
.endm

        .globl sdasm_winograd_weights_avx2
        .type sdasm_winograd_weights_avx2, @function
        .p2align 5
// rdi dst, rsi stride, rdx g, rcx channels
sdasm_winograd_weights_avx2:
        test rcx, rcx
        jz 9f
        vbroadcastss ymm15, [rip + .Lhalf]
        mov rax, rdi
        dec rcx
        jz 5f
1:
        WEIGHTS STORE8
        add rdx, 4 * 9 * 6
        add rax, 4 * 6
        dec rcx
        jnz 1b
5:
        WEIGHTS STORE6
9:
        vzeroupper
        ret
        .size sdasm_winograd_weights_avx2, .-sdasm_winograd_weights_avx2

        .section .note.GNU-stack, "", @progbits
