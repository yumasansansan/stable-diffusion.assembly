// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The transforms of Winograd's F(4x4, 3x3) (winograd.cc) for AVX2 and FMA,
// eight tiles of a row of tiles to a register, in the order 0, 2, 4, 6, 1, 3,
// 5, 7 within each eight:
//
//     void sdasm_winograd4_inputs_avx2(const Winograd4_inputs *job);
//
// What sdasm_winograd4_inputs_avx512 computes, for eight tiles: the rows the job
// names, combined by their coefficients over the 34 numbers the tiles span,
// five loads of eight (masked to the image by vmaskmovps when the job gives
// masks), into R. Tile l's four numbers 4 l..4 l + 3 lie in a 128-bit lane of
// R, two tiles to a register, and a transpose of four by four within the lanes
// gives each j a register of the eight tiles, in the order above; the lanes a
// tile on, joined by vperm2f128, give j = 4 and 5. The panel of each s takes
// the eight in that order, as the products do; their outputs come back to it.
//
//     void sdasm_winograd4_weights_avx2(float *dst, int64_t stride,
//                                       const float *g, int64_t channels);
//
// G g G^T of the kernels of a panel of six output channels, channel after
// channel, as sdasm_winograd_weights_avx2 transforms F(2x2, 3x3)'s
// (winograd_avx2.S): g widened, [channel][3 * 3][6]; xi of channel k into the
// panel at dst + xi * stride bytes + 24 k. Each of the six rows of G g, a
// column at a time from the kernel's three rows, then its six xi from it --
// each combination of G taken as (g0 + g1 + g2) (-1/3), (g0 - g1 + g2) / 3,
// (g0 + 2 g1 + 4 g2) / 15 and (8 g1 - 16 g0 - 4 g2) / 15.
//
//     void sdasm_winograd4_outputs_avx2(const float *m, int64_t stride,
//                                       float *row, int64_t ldc, int64_t groups,
//                                       int64_t add);
//
// Y = A^T M A for groups of eight tiles of a row of tiles in that order, one
// output channel: A^T M a column at a time into 768 bytes of the stack, then
// each row's four numbers of the eight tiles, transposed four by four within
// the 128-bit lanes, which puts the tiles back in their order: 32 consecutive
// outputs of the row, written or added.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
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
.set J4_MASKS, 128
.set J4_OFFSET, 136               // bytes from each row to the tiles' first input

// One row's five loads of eight, by \how: plain or masked, into the sums.
.macro ROW how
.ifc \how, plain
        vmovups ymm5, [rsi]
        vfmadd231ps ymm0, ymm5, ymm6
        vmovups ymm5, [rsi + 32]
        vfmadd231ps ymm1, ymm5, ymm6
        vmovups ymm5, [rsi + 64]
        vfmadd231ps ymm2, ymm5, ymm6
        vmovups ymm5, [rsi + 96]
        vfmadd231ps ymm3, ymm5, ymm6
        vmaskmovps ymm5, ymm11, [rsi + 128]
        vfmadd231ps ymm4, ymm5, ymm6
.else
        vmaskmovps ymm5, ymm7, [rsi]
        vfmadd231ps ymm0, ymm5, ymm6
        vmaskmovps ymm5, ymm8, [rsi + 32]
        vfmadd231ps ymm1, ymm5, ymm6
        vmaskmovps ymm5, ymm9, [rsi + 64]
        vfmadd231ps ymm2, ymm5, ymm6
        vmaskmovps ymm5, ymm10, [rsi + 96]
        vfmadd231ps ymm3, ymm5, ymm6
        vmaskmovps ymm5, ymm11, [rsi + 128]
        vfmadd231ps ymm4, ymm5, ymm6
.endif
.endm

        .globl sdasm_winograd4_inputs_avx2
        .type sdasm_winograd4_inputs_avx2, @function
        .p2align 5
// rdi job
sdasm_winograd4_inputs_avx2:
        push rbx
        push r12
        push r13
        push r14
        mov rax, [rdi + J4_DST]
        mov rdx, [rdi + J4_STRIDE]
        lea rbx, [rdx + 2*rdx]                  // 3 stride
        lea r13, [rdx + 4*rdx]                  // 5 stride
        mov r8, [rdi + J4_CHANNELS]
        mov r9, [rdi + J4_ROWS]
        mov r10, [rdi + J4_STEP]
        lea r14, [4*r10]                        // four channels on
        mov r12, [rdi + J4_MASKS]
        test r12, r12
        jz 1f
        vmovdqu ymm7, [r12]
        vmovdqu ymm8, [r12 + 32]
        vmovdqu ymm9, [r12 + 64]
        vmovdqu ymm10, [r12 + 96]
1:
        // The last load takes two numbers alone, 32 and 33, inside the image
        // or not as the masks say.
        lea r11, [rip + .Ltwo]
        test r12, r12
        jz 2f
        lea r11, [r12 + 128]
2:
        vmovdqu ymm11, [r11]
        // The channel's offset in bytes, from each row's first input of the
        // tiles, which may lie before the row: the addresses are formed
        // here, and nothing outside the image is read (the masks).
        mov rcx, [rdi + J4_OFFSET]
        test r8, r8
        jz 9f
3:
        vxorps ymm0, ymm0, ymm0
        vxorps ymm1, ymm1, ymm1
        vxorps ymm2, ymm2, ymm2
        vxorps ymm3, ymm3, ymm3
        vxorps ymm4, ymm4, ymm4
        xor r11, r11
4:
        cmp r11, r9
        jae 6f
        mov rsi, [rdi + J4_ROW + 8*r11]
        add rsi, rcx
        // The row four channels on, fetched into the cache ahead: the channels
        // lie a plane of the image apart, further than the processor follows
        // a stream of loads. Its 34 numbers reach over four lines of the cache
        // at most, and a fetch every 45 bytes touches each. A fetch is a hint,
        // which reads nothing into a register and faults on nothing, past the
        // last channel too. Measured on the i7-1360P, F(4x4) of 64 by 64 by
        // 320 channels: the transform of the inputs took 21% less time against
        // the products on one thread, and 16% less on sixteen.
        prefetcht0 [rsi + r14]
        prefetcht0 [rsi + r14 + 45]
        prefetcht0 [rsi + r14 + 90]
        prefetcht0 [rsi + r14 + 135]
        vbroadcastss ymm6, [rdi + J4_COEFFICIENT + 4*r11]
        test r12, r12
        jnz 5f
        ROW plain
        inc r11
        jmp 4b
5:
        ROW masked
        inc r11
        jmp 4b
6:
        // u[j] of the eight tiles in the order 0, 2, 4, 6, 1, 3, 5, 7: the
        // four numbers of each tile in each 128-bit lane, transposed four by
        // four; u[4] and u[5] likewise from the lanes a tile on.
        vperm2f128 ymm12, ymm0, ymm1, 0x21
        vperm2f128 ymm13, ymm1, ymm2, 0x21
        vperm2f128 ymm14, ymm2, ymm3, 0x21
        vperm2f128 ymm15, ymm3, ymm4, 0x21
        vunpcklps ymm12, ymm12, ymm13
        vunpcklps ymm14, ymm14, ymm15
        vunpcklpd ymm4, ymm12, ymm14            // u[4]
        vunpckhpd ymm5, ymm12, ymm14            // u[5]
        vunpcklps ymm12, ymm0, ymm1
        vunpckhps ymm13, ymm0, ymm1
        vunpcklps ymm14, ymm2, ymm3
        vunpckhps ymm15, ymm2, ymm3
        vunpcklpd ymm0, ymm12, ymm14            // u[0]
        vunpckhpd ymm1, ymm12, ymm14            // u[1]
        vunpcklpd ymm2, ymm13, ymm15            // u[2]
        vunpckhpd ymm3, ymm13, ymm15            // u[3]
        vaddps ymm14, ymm0, ymm4
        vbroadcastss ymm15, [rip + .Lc1p5]
        vfmadd231ps ymm14, ymm1, ymm15
        vbroadcastss ymm15, [rip + .Lcm2]
        vfmadd231ps ymm14, ymm2, ymm15
        vbroadcastss ymm15, [rip + .Lcm1p5]
        vfmadd231ps ymm14, ymm3, ymm15
        vmovups [rax], ymm14
        vsubps ymm14, ymm4, ymm1
        vbroadcastss ymm15, [rip + .Lcm2p5]
        vfmadd231ps ymm14, ymm2, ymm15
        vbroadcastss ymm15, [rip + .Lcm0p5]
        vfmadd231ps ymm14, ymm3, ymm15
        vmovups [rax + rdx], ymm14
        vaddps ymm14, ymm1, ymm4
        vbroadcastss ymm15, [rip + .Lc0p5]
        vfmadd231ps ymm14, ymm2, ymm15
        vbroadcastss ymm15, [rip + .Lcm2p5]
        vfmadd231ps ymm14, ymm3, ymm15
        vmovups [rax + 2*rdx], ymm14
        vsubps ymm14, ymm4, ymm2
        vbroadcastss ymm15, [rip + .Lcm0p5]
        vfmadd231ps ymm14, ymm1, ymm15
        vbroadcastss ymm15, [rip + .Lc0p5]
        vfmadd231ps ymm14, ymm3, ymm15
        vmovups [rax + rbx], ymm14
        vsubps ymm14, ymm4, ymm2
        vbroadcastss ymm15, [rip + .Lc2]
        vfmadd231ps ymm14, ymm1, ymm15
        vbroadcastss ymm15, [rip + .Lcm2]
        vfmadd231ps ymm14, ymm3, ymm15
        vmovups [rax + 4*rdx], ymm14
        vaddps ymm14, ymm1, ymm5
        vbroadcastss ymm15, [rip + .Lc1p5]
        vfmadd231ps ymm14, ymm2, ymm15
        vbroadcastss ymm15, [rip + .Lcm2]
        vfmadd231ps ymm14, ymm3, ymm15
        vbroadcastss ymm15, [rip + .Lcm1p5]
        vfmadd231ps ymm14, ymm4, ymm15
        vmovups [rax + r13], ymm14
        add rax, 64
        add rcx, r10
        dec r8
        jnz 3b
9:
        pop r14
        pop r13
        pop r12
        pop rbx
        vzeroupper
        ret
        .size sdasm_winograd4_inputs_avx2, .-sdasm_winograd4_inputs_avx2

        .section .rodata
        .p2align 5
.Ltwo:  // the lanes of the last load when the masks are none: numbers 32 and 33
        .long -1, -1, 0, 0, 0, 0, 0, 0

        .text

        .globl sdasm_winograd4_outputs_avx2
        .type sdasm_winograd4_outputs_avx2, @function
        .p2align 5
// rdi m, rsi stride, rdx row, rcx ldc, r8 groups, r9 add
sdasm_winograd4_outputs_avx2:
        push rbx
        push r12
        push rbp
        mov rbp, rsp
        sub rsp, 768
        and rsp, -32
        test r8, r8
        jz 9f
        mov rbx, rcx
        mov r12, rdx
        mov rax, rsi
        imul rax, rax, 6
        lea r10, [rax + 2*rax]
        lea r11, [rax + 4*rax]
1:
        mov rcx, rdi
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + rax]
        vmovups ymm2, [rcx + 2*rax]
        vmovups ymm3, [rcx + r10]
        vmovups ymm4, [rcx + 4*rax]
        vmovups ymm5, [rcx + r11]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vmovaps [rsp + 0], ymm6
        vmovaps [rsp + 32], ymm7
        vmovaps [rsp + 64], ymm8
        vmovaps [rsp + 96], ymm9
        add rcx, rsi
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + rax]
        vmovups ymm2, [rcx + 2*rax]
        vmovups ymm3, [rcx + r10]
        vmovups ymm4, [rcx + 4*rax]
        vmovups ymm5, [rcx + r11]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vmovaps [rsp + 128], ymm6
        vmovaps [rsp + 160], ymm7
        vmovaps [rsp + 192], ymm8
        vmovaps [rsp + 224], ymm9
        add rcx, rsi
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + rax]
        vmovups ymm2, [rcx + 2*rax]
        vmovups ymm3, [rcx + r10]
        vmovups ymm4, [rcx + 4*rax]
        vmovups ymm5, [rcx + r11]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vmovaps [rsp + 256], ymm6
        vmovaps [rsp + 288], ymm7
        vmovaps [rsp + 320], ymm8
        vmovaps [rsp + 352], ymm9
        add rcx, rsi
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + rax]
        vmovups ymm2, [rcx + 2*rax]
        vmovups ymm3, [rcx + r10]
        vmovups ymm4, [rcx + 4*rax]
        vmovups ymm5, [rcx + r11]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vmovaps [rsp + 384], ymm6
        vmovaps [rsp + 416], ymm7
        vmovaps [rsp + 448], ymm8
        vmovaps [rsp + 480], ymm9
        add rcx, rsi
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + rax]
        vmovups ymm2, [rcx + 2*rax]
        vmovups ymm3, [rcx + r10]
        vmovups ymm4, [rcx + 4*rax]
        vmovups ymm5, [rcx + r11]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vmovaps [rsp + 512], ymm6
        vmovaps [rsp + 544], ymm7
        vmovaps [rsp + 576], ymm8
        vmovaps [rsp + 608], ymm9
        add rcx, rsi
        vmovups ymm0, [rcx]
        vmovups ymm1, [rcx + rax]
        vmovups ymm2, [rcx + 2*rax]
        vmovups ymm3, [rcx + r10]
        vmovups ymm4, [rcx + 4*rax]
        vmovups ymm5, [rcx + r11]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vmovaps [rsp + 640], ymm6
        vmovaps [rsp + 672], ymm7
        vmovaps [rsp + 704], ymm8
        vmovaps [rsp + 736], ymm9
        add rcx, rsi
        mov rdx, r12
        vmovaps ymm0, [rsp + 0]
        vmovaps ymm1, [rsp + 128]
        vmovaps ymm2, [rsp + 256]
        vmovaps ymm3, [rsp + 384]
        vmovaps ymm4, [rsp + 512]
        vmovaps ymm5, [rsp + 640]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vunpcklps ymm10, ymm6, ymm7
        vunpckhps ymm11, ymm6, ymm7
        vunpcklps ymm12, ymm8, ymm9
        vunpckhps ymm13, ymm8, ymm9
        vunpcklpd ymm6, ymm10, ymm12            // tiles 0 and 1
        vunpckhpd ymm7, ymm10, ymm12            // 2 and 3
        vunpcklpd ymm8, ymm11, ymm13            // 4 and 5
        vunpckhpd ymm9, ymm11, ymm13            // 6 and 7
        test r9, r9
        jz 30f
        vaddps ymm6, ymm6, [rdx]
        vaddps ymm7, ymm7, [rdx + 32]
        vaddps ymm8, ymm8, [rdx + 64]
        vaddps ymm9, ymm9, [rdx + 96]
30:
        vmovups [rdx], ymm6
        vmovups [rdx + 32], ymm7
        vmovups [rdx + 64], ymm8
        vmovups [rdx + 96], ymm9
        add rdx, rbx
        vmovaps ymm0, [rsp + 32]
        vmovaps ymm1, [rsp + 160]
        vmovaps ymm2, [rsp + 288]
        vmovaps ymm3, [rsp + 416]
        vmovaps ymm4, [rsp + 544]
        vmovaps ymm5, [rsp + 672]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vunpcklps ymm10, ymm6, ymm7
        vunpckhps ymm11, ymm6, ymm7
        vunpcklps ymm12, ymm8, ymm9
        vunpckhps ymm13, ymm8, ymm9
        vunpcklpd ymm6, ymm10, ymm12            // tiles 0 and 1
        vunpckhpd ymm7, ymm10, ymm12            // 2 and 3
        vunpcklpd ymm8, ymm11, ymm13            // 4 and 5
        vunpckhpd ymm9, ymm11, ymm13            // 6 and 7
        test r9, r9
        jz 31f
        vaddps ymm6, ymm6, [rdx]
        vaddps ymm7, ymm7, [rdx + 32]
        vaddps ymm8, ymm8, [rdx + 64]
        vaddps ymm9, ymm9, [rdx + 96]
31:
        vmovups [rdx], ymm6
        vmovups [rdx + 32], ymm7
        vmovups [rdx + 64], ymm8
        vmovups [rdx + 96], ymm9
        add rdx, rbx
        vmovaps ymm0, [rsp + 64]
        vmovaps ymm1, [rsp + 192]
        vmovaps ymm2, [rsp + 320]
        vmovaps ymm3, [rsp + 448]
        vmovaps ymm4, [rsp + 576]
        vmovaps ymm5, [rsp + 704]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vunpcklps ymm10, ymm6, ymm7
        vunpckhps ymm11, ymm6, ymm7
        vunpcklps ymm12, ymm8, ymm9
        vunpckhps ymm13, ymm8, ymm9
        vunpcklpd ymm6, ymm10, ymm12            // tiles 0 and 1
        vunpckhpd ymm7, ymm10, ymm12            // 2 and 3
        vunpcklpd ymm8, ymm11, ymm13            // 4 and 5
        vunpckhpd ymm9, ymm11, ymm13            // 6 and 7
        test r9, r9
        jz 32f
        vaddps ymm6, ymm6, [rdx]
        vaddps ymm7, ymm7, [rdx + 32]
        vaddps ymm8, ymm8, [rdx + 64]
        vaddps ymm9, ymm9, [rdx + 96]
32:
        vmovups [rdx], ymm6
        vmovups [rdx + 32], ymm7
        vmovups [rdx + 64], ymm8
        vmovups [rdx + 96], ymm9
        add rdx, rbx
        vmovaps ymm0, [rsp + 96]
        vmovaps ymm1, [rsp + 224]
        vmovaps ymm2, [rsp + 352]
        vmovaps ymm3, [rsp + 480]
        vmovaps ymm4, [rsp + 608]
        vmovaps ymm5, [rsp + 736]
        vaddps ymm10, ymm1, ymm2
        vsubps ymm11, ymm1, ymm2
        vaddps ymm6, ymm10, ymm3
        vaddps ymm6, ymm6, ymm4
        vaddps ymm6, ymm6, ymm0
        vbroadcastss ymm12, [rip + .Lc2]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lc4]
        vfmadd231ps ymm10, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p5]
        vmovaps ymm7, ymm11
        vfmadd231ps ymm7, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc0p25]
        vmovaps ymm8, ymm10
        vfmadd231ps ymm8, ymm4, ymm12
        vbroadcastss ymm12, [rip + .Lc6]
        vfmadd231ps ymm11, ymm3, ymm12
        vbroadcastss ymm12, [rip + .Lcm0p125]
        vfmadd231ps ymm11, ymm4, ymm12
        vaddps ymm9, ymm11, ymm5
        vunpcklps ymm10, ymm6, ymm7
        vunpckhps ymm11, ymm6, ymm7
        vunpcklps ymm12, ymm8, ymm9
        vunpckhps ymm13, ymm8, ymm9
        vunpcklpd ymm6, ymm10, ymm12            // tiles 0 and 1
        vunpckhpd ymm7, ymm10, ymm12            // 2 and 3
        vunpcklpd ymm8, ymm11, ymm13            // 4 and 5
        vunpckhpd ymm9, ymm11, ymm13            // 6 and 7
        test r9, r9
        jz 33f
        vaddps ymm6, ymm6, [rdx]
        vaddps ymm7, ymm7, [rdx + 32]
        vaddps ymm8, ymm8, [rdx + 64]
        vaddps ymm9, ymm9, [rdx + 96]
33:
        vmovups [rdx], ymm6
        vmovups [rdx + 32], ymm7
        vmovups [rdx + 64], ymm8
        vmovups [rdx + 96], ymm9
        add rdx, rbx
        add rdi, 32
        add r12, 128
        dec r8
        jnz 1b
9:
        mov rsp, rbp
        pop rbp
        pop r12
        pop rbx
        vzeroupper
        ret
        .size sdasm_winograd4_outputs_avx2, .-sdasm_winograd4_outputs_avx2

// ---------------------------------------------------------------- weights

        .section .rodata
        .p2align 2
.Lcm1o3:  .float -0.33333334
.Lc1o3:   .float 0.33333334
.Lc1o15:  .float 0.06666667
.Lc8:     .float 8.0
.Lcm16:   .float -16.0
.Lcm4:    .float -4.0

        .text

// ymm\v to the panel of xi: rax + xi * rsi -- six of its eight lanes, or all
// eight, which the next channel's write over.
.macro STORE8_4 v, xi
        imul r8, rsi, (\xi)
        vmovups [rax + r8], ymm\v
.endm
.macro STORE6_4 v, xi
        imul r8, rsi, (\xi)
        vmovups [rax + r8], xmm\v
        vextractf128 xmm13, ymm\v, 1
        vmovlps [rax + r8 + 16], xmm13
.endm

.macro WEIGHTS4 store
        vmovups ymm0, [rdx + 0]
        vmovups ymm1, [rdx + 24]
        vmovups ymm2, [rdx + 48]
        vmovups ymm3, [rdx + 72]
        vmovups ymm4, [rdx + 96]
        vmovups ymm5, [rdx + 120]
        vmovups ymm6, [rdx + 144]
        vmovups ymm7, [rdx + 168]
        vmovups ymm8, [rdx + 192]
        // Row 0 of G g, then its six xi.
        vmovaps ymm9, ymm0
        vmovaps ymm10, ymm1
        vmovaps ymm11, ymm2
        vmovaps ymm12, ymm9
        \store 12, 0
        vaddps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 1
        vsubps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 2
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm12, ymm9
        vfmadd231ps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 3
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm12, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 4
        vmovaps ymm12, ymm11
        \store 12, 5
        // Row 1 of G g, then its six xi.
        vaddps ymm9, ymm0, ymm3
        vaddps ymm9, ymm9, ymm6
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm9, ymm9, ymm13
        vaddps ymm10, ymm1, ymm4
        vaddps ymm10, ymm10, ymm7
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm10, ymm10, ymm13
        vaddps ymm11, ymm2, ymm5
        vaddps ymm11, ymm11, ymm8
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm11, ymm11, ymm13
        vmovaps ymm12, ymm9
        \store 12, 6
        vaddps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 7
        vsubps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 8
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm12, ymm9
        vfmadd231ps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 9
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm12, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 10
        vmovaps ymm12, ymm11
        \store 12, 11
        // Row 2 of G g, then its six xi.
        vsubps ymm9, ymm0, ymm3
        vaddps ymm9, ymm9, ymm6
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm9, ymm9, ymm13
        vsubps ymm10, ymm1, ymm4
        vaddps ymm10, ymm10, ymm7
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm10, ymm10, ymm13
        vsubps ymm11, ymm2, ymm5
        vaddps ymm11, ymm11, ymm8
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm11, ymm11, ymm13
        vmovaps ymm12, ymm9
        \store 12, 12
        vaddps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 13
        vsubps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 14
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm12, ymm9
        vfmadd231ps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 15
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm12, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 16
        vmovaps ymm12, ymm11
        \store 12, 17
        // Row 3 of G g, then its six xi.
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm9, ymm0
        vfmadd231ps ymm9, ymm3, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm9, ymm6, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm9, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm10, ymm1
        vfmadd231ps ymm10, ymm4, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm10, ymm7, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm10, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm11, ymm2
        vfmadd231ps ymm11, ymm5, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm11, ymm8, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm11, ymm11, ymm13
        vmovaps ymm12, ymm9
        \store 12, 18
        vaddps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 19
        vsubps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 20
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm12, ymm9
        vfmadd231ps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 21
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm12, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 22
        vmovaps ymm12, ymm11
        \store 12, 23
        // Row 4 of G g, then its six xi.
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm9, ymm3, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm9, ymm0, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm9, ymm6, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm9, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm10, ymm4, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm10, ymm1, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm10, ymm7, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm10, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm11, ymm5, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm11, ymm2, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm11, ymm8, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm11, ymm11, ymm13
        vmovaps ymm12, ymm9
        \store 12, 24
        vaddps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 25
        vsubps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 26
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm12, ymm9
        vfmadd231ps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 27
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm12, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 28
        vmovaps ymm12, ymm11
        \store 12, 29
        // Row 5 of G g, then its six xi.
        vmovaps ymm9, ymm6
        vmovaps ymm10, ymm7
        vmovaps ymm11, ymm8
        vmovaps ymm12, ymm9
        \store 12, 30
        vaddps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lcm1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 31
        vsubps ymm12, ymm9, ymm10
        vaddps ymm12, ymm12, ymm11
        vbroadcastss ymm13, [rip + .Lc1o3]
        vmulps ymm12, ymm12, ymm13
        \store 12, 32
        vbroadcastss ymm13, [rip + .Lc2]
        vmovaps ymm12, ymm9
        vfmadd231ps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lc4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 33
        vbroadcastss ymm13, [rip + .Lc8]
        vmulps ymm12, ymm10, ymm13
        vbroadcastss ymm13, [rip + .Lcm16]
        vfmadd231ps ymm12, ymm9, ymm13
        vbroadcastss ymm13, [rip + .Lcm4]
        vfmadd231ps ymm12, ymm11, ymm13
        vbroadcastss ymm13, [rip + .Lc1o15]
        vmulps ymm12, ymm12, ymm13
        \store 12, 34
        vmovaps ymm12, ymm11
        \store 12, 35
.endm

        .globl sdasm_winograd4_weights_avx2
        .type sdasm_winograd4_weights_avx2, @function
        .p2align 5
// rdi dst, rsi stride, rdx g, rcx channels
sdasm_winograd4_weights_avx2:
        test rcx, rcx
        jz 9f
        mov rax, rdi
        dec rcx
        jz 5f
1:
        WEIGHTS4 STORE8_4
        add rdx, 4 * 9 * 6
        add rax, 4 * 6
        dec rcx
        jnz 1b
5:
        WEIGHTS4 STORE6_4
9:
        vzeroupper
        ret
        .size sdasm_winograd4_weights_avx2, .-sdasm_winograd4_weights_avx2

        .section .note.GNU-stack, "", @progbits
