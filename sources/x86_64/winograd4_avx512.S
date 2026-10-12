// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The transforms of Winograd's F(4x4, 3x3) (winograd.cc) for AVX-512 F, VL,
// BW and DQ, sixteen tiles of a row of tiles to a register, in their order:
//
//     void sdasm_winograd4_inputs_avx512(const Winograd4_inputs *job);
//
// The six elements xi = (r, s) of the transformed inputs of the sixteen tiles,
// channel after channel: the rows the job names, those of the tiles that row r
// of B^T takes, combined by their coefficients over the 66 numbers the tiles
// span -- five loads of sixteen each, masked to the image by the lanes the job
// gives -- into R; then d[j] of tile l is R[4 l + j], picked out of two pairs
// of registers by vpermi2ps and joined by vinsertf32x8, those of j = 4 and 5 a
// tile on from those of 0 and 1 by valignd; and each s its combination of
// them, into the panel of s.
//
//     void sdasm_winograd4_outputs_avx512(const float *m, int64_t stride,
//                                         float *row, int64_t ldc,
//                                         int64_t groups, int64_t add);
//
// Y = A^T M A for groups of sixteen tiles of a row of tiles, one output
// channel, M of a group's tiles at m + xi * stride bytes, the four rows of the
// output ldc bytes apart from row; each row's four numbers of each tile, sixteen
// registers' lanes, transposed four by four within each 128-bit lane and then
// across them into 64 consecutive outputs, written or added.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 6
.Lgather:       // the numbers 4 l + j of two registers of sixteen, j = 0..3, in lanes 0..7
        .long 0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28
        .long 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29
        .long 2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18, 22, 26, 30
        .long 3, 7, 11, 15, 19, 23, 27, 31, 3, 7, 11, 15, 19, 23, 27, 31
        .p2align 2
.Lc1p5:   .float 1.5
.Lcm1p5:  .float -1.5
.Lcm2:    .float -2.0
.Lc2:     .float 2.0
.Lcm2p5:  .float -2.5
.Lc0p5:   .float 0.5
.Lcm0p5:  .float -0.5
.Lc4:     .float 4.0
.Lc6:     .float 6.0
.Lc0p25:  .float 0.25
.Lcm0p125: .float -0.125

        .text

.set J4_DST, 0
.set J4_STRIDE, 8
.set J4_STEP, 16
.set J4_CHANNELS, 24
.set J4_ROWS, 32
.set J4_ROW, 40
.set J4_COEFFICIENT, 88
.set J4_LANES, 112
.set J4_MASKS, 128
.set J4_OFFSET, 136               // bytes from each row to the tiles' first input

        .globl sdasm_winograd4_inputs_avx512
        .type sdasm_winograd4_inputs_avx512, @function
        .p2align 5
// rdi job
sdasm_winograd4_inputs_avx512:
        push rbx
        mov rax, [rdi + J4_DST]
        mov rdx, [rdi + J4_STRIDE]
        lea rbx, [rdx + 2*rdx]                  // 3 stride
        mov r8, [rdi + J4_CHANNELS]
        mov r9, [rdi + J4_ROWS]
        mov r10, [rdi + J4_STEP]
        kmovw k1, [rdi + J4_LANES + 0]
        kmovw k2, [rdi + J4_LANES + 2]
        kmovw k3, [rdi + J4_LANES + 4]
        kmovw k4, [rdi + J4_LANES + 6]
        kmovw k5, [rdi + J4_LANES + 8]
        vmovdqu32 zmm20, [rip + .Lgather]
        vmovdqu32 zmm21, [rip + .Lgather + 64]
        vmovdqu32 zmm22, [rip + .Lgather + 128]
        vmovdqu32 zmm23, [rip + .Lgather + 192]
        // The channel's offset in bytes, from each row's first input of the
        // tiles, which may lie before the row: the addresses are formed
        // here, and nothing outside the image is read (the masks).
        mov rcx, [rdi + J4_OFFSET]
        test r8, r8
        jz 9f
1:
        // The combination of the rows: 66 numbers in zmm0..zmm4.
        vpxord zmm0, zmm0, zmm0
        vpxord zmm1, zmm1, zmm1
        vpxord zmm2, zmm2, zmm2
        vpxord zmm3, zmm3, zmm3
        vpxord zmm4, zmm4, zmm4
        xor r11, r11
2:
        cmp r11, r9
        jae 3f
        mov rsi, [rdi + J4_ROW + 8*r11]
        add rsi, rcx
        vbroadcastss zmm6, [rdi + J4_COEFFICIENT + 4*r11]
        vmovups zmm5{k1}{z}, [rsi]
        vfmadd231ps zmm0, zmm5, zmm6
        vmovups zmm5{k2}{z}, [rsi + 64]
        vfmadd231ps zmm1, zmm5, zmm6
        vmovups zmm5{k3}{z}, [rsi + 128]
        vfmadd231ps zmm2, zmm5, zmm6
        vmovups zmm5{k4}{z}, [rsi + 192]
        vfmadd231ps zmm3, zmm5, zmm6
        vmovups zmm5{k5}{z}, [rsi + 256]
        vfmadd231ps zmm4, zmm5, zmm6
        inc r11
        jmp 2b
3:
        // u[j] = R[4 l + j] of the sixteen tiles l: the first eight from
        // zmm0 and zmm1, the second from zmm2 and zmm3; u[4] and u[5] are
        // u[0] and u[1] a tile on, with R[64] and R[65] last.
        vmovdqa32 zmm8, zmm20
        vpermi2ps zmm8, zmm0, zmm1
        vmovdqa32 zmm24, zmm20
        vpermi2ps zmm24, zmm2, zmm3
        vinsertf32x8 zmm8, zmm8, ymm24, 1
        vmovdqa32 zmm9, zmm21
        vpermi2ps zmm9, zmm0, zmm1
        vmovdqa32 zmm24, zmm21
        vpermi2ps zmm24, zmm2, zmm3
        vinsertf32x8 zmm9, zmm9, ymm24, 1
        vmovdqa32 zmm10, zmm22
        vpermi2ps zmm10, zmm0, zmm1
        vmovdqa32 zmm24, zmm22
        vpermi2ps zmm24, zmm2, zmm3
        vinsertf32x8 zmm10, zmm10, ymm24, 1
        vmovdqa32 zmm11, zmm23
        vpermi2ps zmm11, zmm0, zmm1
        vmovdqa32 zmm24, zmm23
        vpermi2ps zmm24, zmm2, zmm3
        vinsertf32x8 zmm11, zmm11, ymm24, 1
        valignd zmm12, zmm4, zmm8, 1
        valignd zmm24, zmm4, zmm4, 1
        valignd zmm13, zmm24, zmm9, 1
        vaddps zmm14, zmm8, zmm12
        vfmadd231ps zmm14, zmm9, dword ptr [rip + .Lc1p5]{1to16}
        vfmadd231ps zmm14, zmm10, dword ptr [rip + .Lcm2]{1to16}
        vfmadd231ps zmm14, zmm11, dword ptr [rip + .Lcm1p5]{1to16}
        vsubps zmm15, zmm12, zmm9
        vfmadd231ps zmm15, zmm10, dword ptr [rip + .Lcm2p5]{1to16}
        vfmadd231ps zmm15, zmm11, dword ptr [rip + .Lcm0p5]{1to16}
        vaddps zmm16, zmm9, zmm12
        vfmadd231ps zmm16, zmm10, dword ptr [rip + .Lc0p5]{1to16}
        vfmadd231ps zmm16, zmm11, dword ptr [rip + .Lcm2p5]{1to16}
        vsubps zmm17, zmm12, zmm10
        vfmadd231ps zmm17, zmm9, dword ptr [rip + .Lcm0p5]{1to16}
        vfmadd231ps zmm17, zmm11, dword ptr [rip + .Lc0p5]{1to16}
        vsubps zmm18, zmm12, zmm10
        vfmadd231ps zmm18, zmm9, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm18, zmm11, dword ptr [rip + .Lcm2]{1to16}
        vaddps zmm19, zmm9, zmm13
        vfmadd231ps zmm19, zmm10, dword ptr [rip + .Lc1p5]{1to16}
        vfmadd231ps zmm19, zmm11, dword ptr [rip + .Lcm2]{1to16}
        vfmadd231ps zmm19, zmm12, dword ptr [rip + .Lcm1p5]{1to16}
        vmovups [rax], zmm14
        vmovups [rax + rdx], zmm15
        vmovups [rax + 2*rdx], zmm16
        vmovups [rax + rbx], zmm17
        vmovups [rax + 4*rdx], zmm18
        lea r11, [rax + 4*rdx]
        vmovups [r11 + rdx], zmm19
        add rax, 64
        add rcx, r10
        dec r8
        jnz 1b
9:
        pop rbx
        vzeroupper
        ret
        .size sdasm_winograd4_inputs_avx512, .-sdasm_winograd4_inputs_avx512

        .globl sdasm_winograd4_outputs_avx512
        .type sdasm_winograd4_outputs_avx512, @function
        .p2align 5
// rdi m, rsi stride, rdx row, rcx ldc, r8 groups, r9 add
sdasm_winograd4_outputs_avx512:
        push rbx
        push r12
        test r8, r8
        jz 9f
        mov rbx, rcx                            // ldc
        mov r12, rdx                            // row 0 of the group
        mov rax, rsi
        imul rax, rax, 6                       // the stride of x, six xi
        lea r10, [rax + 2*rax]                 // 3
        lea r11, [rax + 4*rax]                 // 5
1:
        mov rcx, rdi
        // Column 0: a[i][0] = (A^T M)[i][0] into zmm0..zmm3.
        vmovups zmm24, [rcx]
        vmovups zmm25, [rcx + rax]
        vmovups zmm26, [rcx + 2*rax]
        vmovups zmm27, [rcx + r10]
        vmovups zmm28, [rcx + 4*rax]
        vmovups zmm29, [rcx + r11]
        vaddps zmm30, zmm25, zmm26
        vsubps zmm31, zmm25, zmm26
        vaddps zmm0, zmm30, zmm27
        vaddps zmm0, zmm0, zmm28
        vaddps zmm0, zmm0, zmm24
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm30, zmm27, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm1, zmm31
        vfmadd231ps zmm1, zmm28, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm2, zmm30
        vfmadd231ps zmm2, zmm28, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm31, zmm28, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm3, zmm31, zmm29
        add rcx, rsi
        // Column 1: a[i][1] = (A^T M)[i][1] into zmm4..zmm7.
        vmovups zmm24, [rcx]
        vmovups zmm25, [rcx + rax]
        vmovups zmm26, [rcx + 2*rax]
        vmovups zmm27, [rcx + r10]
        vmovups zmm28, [rcx + 4*rax]
        vmovups zmm29, [rcx + r11]
        vaddps zmm30, zmm25, zmm26
        vsubps zmm31, zmm25, zmm26
        vaddps zmm4, zmm30, zmm27
        vaddps zmm4, zmm4, zmm28
        vaddps zmm4, zmm4, zmm24
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm30, zmm27, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm5, zmm31
        vfmadd231ps zmm5, zmm28, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm6, zmm30
        vfmadd231ps zmm6, zmm28, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm31, zmm28, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm7, zmm31, zmm29
        add rcx, rsi
        // Column 2: a[i][2] = (A^T M)[i][2] into zmm8..zmm11.
        vmovups zmm24, [rcx]
        vmovups zmm25, [rcx + rax]
        vmovups zmm26, [rcx + 2*rax]
        vmovups zmm27, [rcx + r10]
        vmovups zmm28, [rcx + 4*rax]
        vmovups zmm29, [rcx + r11]
        vaddps zmm30, zmm25, zmm26
        vsubps zmm31, zmm25, zmm26
        vaddps zmm8, zmm30, zmm27
        vaddps zmm8, zmm8, zmm28
        vaddps zmm8, zmm8, zmm24
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm30, zmm27, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm9, zmm31
        vfmadd231ps zmm9, zmm28, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm10, zmm30
        vfmadd231ps zmm10, zmm28, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm31, zmm28, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm11, zmm31, zmm29
        add rcx, rsi
        // Column 3: a[i][3] = (A^T M)[i][3] into zmm12..zmm15.
        vmovups zmm24, [rcx]
        vmovups zmm25, [rcx + rax]
        vmovups zmm26, [rcx + 2*rax]
        vmovups zmm27, [rcx + r10]
        vmovups zmm28, [rcx + 4*rax]
        vmovups zmm29, [rcx + r11]
        vaddps zmm30, zmm25, zmm26
        vsubps zmm31, zmm25, zmm26
        vaddps zmm12, zmm30, zmm27
        vaddps zmm12, zmm12, zmm28
        vaddps zmm12, zmm12, zmm24
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm30, zmm27, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm13, zmm31
        vfmadd231ps zmm13, zmm28, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm14, zmm30
        vfmadd231ps zmm14, zmm28, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm31, zmm28, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm15, zmm31, zmm29
        add rcx, rsi
        // Column 4: a[i][4] = (A^T M)[i][4] into zmm16..zmm19.
        vmovups zmm24, [rcx]
        vmovups zmm25, [rcx + rax]
        vmovups zmm26, [rcx + 2*rax]
        vmovups zmm27, [rcx + r10]
        vmovups zmm28, [rcx + 4*rax]
        vmovups zmm29, [rcx + r11]
        vaddps zmm30, zmm25, zmm26
        vsubps zmm31, zmm25, zmm26
        vaddps zmm16, zmm30, zmm27
        vaddps zmm16, zmm16, zmm28
        vaddps zmm16, zmm16, zmm24
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm30, zmm27, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm17, zmm31
        vfmadd231ps zmm17, zmm28, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm18, zmm30
        vfmadd231ps zmm18, zmm28, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm31, zmm28, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm19, zmm31, zmm29
        add rcx, rsi
        // Column 5: a[i][5] = (A^T M)[i][5] into zmm20..zmm23.
        vmovups zmm24, [rcx]
        vmovups zmm25, [rcx + rax]
        vmovups zmm26, [rcx + 2*rax]
        vmovups zmm27, [rcx + r10]
        vmovups zmm28, [rcx + 4*rax]
        vmovups zmm29, [rcx + r11]
        vaddps zmm30, zmm25, zmm26
        vsubps zmm31, zmm25, zmm26
        vaddps zmm20, zmm30, zmm27
        vaddps zmm20, zmm20, zmm28
        vaddps zmm20, zmm20, zmm24
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm30, zmm27, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm21, zmm31
        vfmadd231ps zmm21, zmm28, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm22, zmm30
        vfmadd231ps zmm22, zmm28, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm31, zmm27, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm31, zmm28, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm23, zmm31, zmm29
        add rcx, rsi
        mov rdx, r12
        // Row 0 of the outputs: y[0][jj], then sixteen tiles' four numbers in order.
        vaddps zmm28, zmm4, zmm8
        vsubps zmm29, zmm4, zmm8
        vaddps zmm24, zmm28, zmm12
        vaddps zmm24, zmm24, zmm16
        vaddps zmm24, zmm24, zmm0
        vfmadd231ps zmm29, zmm12, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm28, zmm12, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm25, zmm29
        vfmadd231ps zmm25, zmm16, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm26, zmm28
        vfmadd231ps zmm26, zmm16, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm29, zmm12, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm29, zmm16, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm27, zmm29, zmm20
        vunpcklps zmm28, zmm24, zmm25
        vunpckhps zmm29, zmm24, zmm25
        vunpcklps zmm30, zmm26, zmm27
        vunpckhps zmm31, zmm26, zmm27
        vunpcklpd zmm24, zmm28, zmm30
        vunpckhpd zmm25, zmm28, zmm30
        vunpcklpd zmm26, zmm29, zmm31
        vunpckhpd zmm27, zmm29, zmm31
        vshuff32x4 zmm28, zmm24, zmm25, 0x44
        vshuff32x4 zmm29, zmm26, zmm27, 0x44
        vshuff32x4 zmm30, zmm24, zmm25, 0xee
        vshuff32x4 zmm31, zmm26, zmm27, 0xee
        vshuff32x4 zmm24, zmm28, zmm29, 0x88
        vshuff32x4 zmm25, zmm28, zmm29, 0xdd
        vshuff32x4 zmm26, zmm30, zmm31, 0x88
        vshuff32x4 zmm27, zmm30, zmm31, 0xdd
        test r9, r9
        jz 30f
        vaddps zmm24, zmm24, [rdx]
        vaddps zmm25, zmm25, [rdx + 64]
        vaddps zmm26, zmm26, [rdx + 128]
        vaddps zmm27, zmm27, [rdx + 192]
30:
        vmovups [rdx], zmm24
        vmovups [rdx + 64], zmm25
        vmovups [rdx + 128], zmm26
        vmovups [rdx + 192], zmm27
        add rdx, rbx                           // the next row of outputs
        // Row 1 of the outputs: y[1][jj], then sixteen tiles' four numbers in order.
        vaddps zmm28, zmm5, zmm9
        vsubps zmm29, zmm5, zmm9
        vaddps zmm24, zmm28, zmm13
        vaddps zmm24, zmm24, zmm17
        vaddps zmm24, zmm24, zmm1
        vfmadd231ps zmm29, zmm13, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm28, zmm13, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm25, zmm29
        vfmadd231ps zmm25, zmm17, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm26, zmm28
        vfmadd231ps zmm26, zmm17, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm29, zmm13, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm29, zmm17, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm27, zmm29, zmm21
        vunpcklps zmm28, zmm24, zmm25
        vunpckhps zmm29, zmm24, zmm25
        vunpcklps zmm30, zmm26, zmm27
        vunpckhps zmm31, zmm26, zmm27
        vunpcklpd zmm24, zmm28, zmm30
        vunpckhpd zmm25, zmm28, zmm30
        vunpcklpd zmm26, zmm29, zmm31
        vunpckhpd zmm27, zmm29, zmm31
        vshuff32x4 zmm28, zmm24, zmm25, 0x44
        vshuff32x4 zmm29, zmm26, zmm27, 0x44
        vshuff32x4 zmm30, zmm24, zmm25, 0xee
        vshuff32x4 zmm31, zmm26, zmm27, 0xee
        vshuff32x4 zmm24, zmm28, zmm29, 0x88
        vshuff32x4 zmm25, zmm28, zmm29, 0xdd
        vshuff32x4 zmm26, zmm30, zmm31, 0x88
        vshuff32x4 zmm27, zmm30, zmm31, 0xdd
        test r9, r9
        jz 31f
        vaddps zmm24, zmm24, [rdx]
        vaddps zmm25, zmm25, [rdx + 64]
        vaddps zmm26, zmm26, [rdx + 128]
        vaddps zmm27, zmm27, [rdx + 192]
31:
        vmovups [rdx], zmm24
        vmovups [rdx + 64], zmm25
        vmovups [rdx + 128], zmm26
        vmovups [rdx + 192], zmm27
        add rdx, rbx                           // the next row of outputs
        // Row 2 of the outputs: y[2][jj], then sixteen tiles' four numbers in order.
        vaddps zmm28, zmm6, zmm10
        vsubps zmm29, zmm6, zmm10
        vaddps zmm24, zmm28, zmm14
        vaddps zmm24, zmm24, zmm18
        vaddps zmm24, zmm24, zmm2
        vfmadd231ps zmm29, zmm14, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm28, zmm14, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm25, zmm29
        vfmadd231ps zmm25, zmm18, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm26, zmm28
        vfmadd231ps zmm26, zmm18, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm29, zmm14, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm29, zmm18, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm27, zmm29, zmm22
        vunpcklps zmm28, zmm24, zmm25
        vunpckhps zmm29, zmm24, zmm25
        vunpcklps zmm30, zmm26, zmm27
        vunpckhps zmm31, zmm26, zmm27
        vunpcklpd zmm24, zmm28, zmm30
        vunpckhpd zmm25, zmm28, zmm30
        vunpcklpd zmm26, zmm29, zmm31
        vunpckhpd zmm27, zmm29, zmm31
        vshuff32x4 zmm28, zmm24, zmm25, 0x44
        vshuff32x4 zmm29, zmm26, zmm27, 0x44
        vshuff32x4 zmm30, zmm24, zmm25, 0xee
        vshuff32x4 zmm31, zmm26, zmm27, 0xee
        vshuff32x4 zmm24, zmm28, zmm29, 0x88
        vshuff32x4 zmm25, zmm28, zmm29, 0xdd
        vshuff32x4 zmm26, zmm30, zmm31, 0x88
        vshuff32x4 zmm27, zmm30, zmm31, 0xdd
        test r9, r9
        jz 32f
        vaddps zmm24, zmm24, [rdx]
        vaddps zmm25, zmm25, [rdx + 64]
        vaddps zmm26, zmm26, [rdx + 128]
        vaddps zmm27, zmm27, [rdx + 192]
32:
        vmovups [rdx], zmm24
        vmovups [rdx + 64], zmm25
        vmovups [rdx + 128], zmm26
        vmovups [rdx + 192], zmm27
        add rdx, rbx                           // the next row of outputs
        // Row 3 of the outputs: y[3][jj], then sixteen tiles' four numbers in order.
        vaddps zmm28, zmm7, zmm11
        vsubps zmm29, zmm7, zmm11
        vaddps zmm24, zmm28, zmm15
        vaddps zmm24, zmm24, zmm19
        vaddps zmm24, zmm24, zmm3
        vfmadd231ps zmm29, zmm15, dword ptr [rip + .Lc2]{1to16}
        vfmadd231ps zmm28, zmm15, dword ptr [rip + .Lc4]{1to16}
        vmovaps zmm25, zmm29
        vfmadd231ps zmm25, zmm19, dword ptr [rip + .Lcm0p5]{1to16}
        vmovaps zmm26, zmm28
        vfmadd231ps zmm26, zmm19, dword ptr [rip + .Lc0p25]{1to16}
        vfmadd231ps zmm29, zmm15, dword ptr [rip + .Lc6]{1to16}
        vfmadd231ps zmm29, zmm19, dword ptr [rip + .Lcm0p125]{1to16}
        vaddps zmm27, zmm29, zmm23
        vunpcklps zmm28, zmm24, zmm25
        vunpckhps zmm29, zmm24, zmm25
        vunpcklps zmm30, zmm26, zmm27
        vunpckhps zmm31, zmm26, zmm27
        vunpcklpd zmm24, zmm28, zmm30
        vunpckhpd zmm25, zmm28, zmm30
        vunpcklpd zmm26, zmm29, zmm31
        vunpckhpd zmm27, zmm29, zmm31
        vshuff32x4 zmm28, zmm24, zmm25, 0x44
        vshuff32x4 zmm29, zmm26, zmm27, 0x44
        vshuff32x4 zmm30, zmm24, zmm25, 0xee
        vshuff32x4 zmm31, zmm26, zmm27, 0xee
        vshuff32x4 zmm24, zmm28, zmm29, 0x88
        vshuff32x4 zmm25, zmm28, zmm29, 0xdd
        vshuff32x4 zmm26, zmm30, zmm31, 0x88
        vshuff32x4 zmm27, zmm30, zmm31, 0xdd
        test r9, r9
        jz 33f
        vaddps zmm24, zmm24, [rdx]
        vaddps zmm25, zmm25, [rdx + 64]
        vaddps zmm26, zmm26, [rdx + 128]
        vaddps zmm27, zmm27, [rdx + 192]
33:
        vmovups [rdx], zmm24
        vmovups [rdx + 64], zmm25
        vmovups [rdx + 128], zmm26
        vmovups [rdx + 192], zmm27
        add rdx, rbx                           // the next row of outputs
        add rdi, 64
        add r12, 256
        dec r8
        jnz 1b
9:
        pop r12
        pop rbx
        vzeroupper
        ret
        .size sdasm_winograd4_outputs_avx512, .-sdasm_winograd4_outputs_avx512

        .section .note.GNU-stack, "", @progbits
