// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The matrix products on the tiles of AMX (AMX-TILE and AMX-BF16, with AVX-512
// F, BW, VL and AVX512-BF16 for the packing), for the products whose weights
// are FP16:
//
//     int64_t sdasm_amx_request(void);
//
// The tiles hold eight kilobytes of a thread's state, which Linux saves only
// for a process that has asked for them, by arch_prctl(ARCH_REQ_XCOMP_PERM,
// XFEATURE_XTILEDATA); until then an instruction on the tiles faults. The
// request is the system call itself, which returns 0 when the leave is given
// and a negative error number when it is not -- by a system that does not know
// the tiles, or that runs on a processor without them. On a system other than
// Linux, it returns -1: the tiles are not used.
//
//     void sdasm_amx_configure(void);
//     void sdasm_amx_release(void);
//
// The configuration every kernel here takes -- eight tiles of sixteen rows of
// 64 bytes -- loaded into the thread's tiles before them, and the tiles given
// back after, which the processor and the system then need not keep.
//
// **The precision.** A tile multiplies numbers of bfloat16, whose 8 bits of
// significand are too few for the products: ggml's form of these products
// rounds the image to FP16, 11 bits, and the kernels here are not to do worse.
// So each number is split in two parts of bfloat16, each rounded to the
// nearest: x = x1 + x2, where x1 is x rounded and x2 what it leaves, rounded,
// so that x1 + x2 is within 2^-16 of x; and the weights likewise, w = w1 + w2,
// which is exactly w, an FP16 number of 11 bits. The product is x1 w1 + x1 w2 +
// x2 w1, three products on the tiles, which leave out x2 w2, within 2^-16 of
// x w: the product is within 2^-15 of x w, sixteen times as close as ggml's
// (2^-11). The tiles sum in single precision, rounding to the nearest; they
// flush denormal numbers, below 2^-126, to zero.
//
//     void sdasm_amx_pack_rows(uint16_t *dst, const float *src, int64_t stride,
//                              int64_t rows, int64_t k);
//
// The tiles of A, of sixteen rows of src, `stride` bytes apart -- rows past
// `rows` (1 to 16) are zeros -- over k numbers, rounded up to blocks of 32 with
// zeros: for each block, the tile of x1, sixteen rows of 32 numbers of bfloat16,
// then that of x2, 2048 bytes a block. Each part is rounded by AVX512-BF16's
// vcvtne2ps2bf16, and x - x1 is exact in single precision.
//
//     void sdasm_amx_pack_rows_f16(uint16_t *dst, const uint16_t *src,
//                                  int64_t stride, int64_t rows, int64_t k);
//
// The same of rows of FP16 numbers, widened by vcvtph2ps: the weights of the
// convolutions, A there; x1 + x2 is then exactly x.
//
//     void sdasm_amx_pack_weights(uint16_t *dst, const uint16_t *src,
//                                 int64_t stride, int64_t cols, int64_t k);
//
// The tiles of B from sixteen rows of FP16 weights, a row of k numbers for each
// column of B, `stride` bytes apart; columns past `cols` (1 to 16) are zeros,
// and k is rounded up to blocks of 32. A tile of B has a row for each pair of
// k, 2p and 2p + 1, which holds the pair of each of the sixteen columns side by
// side, as the products take them: for each block, that of w1, then that of
// w2, 2048 bytes a block. The rows of the weights, widened to single precision
// by vcvtph2ps and split, give a register of sixteen pairs for each column,
// and the sixteen registers are transposed, as dwords, into the rows of the
// tile.
//
//     void sdasm_amx_pack_panel(uint16_t *dst, const float *src, int64_t k);
//
// The tiles of B from a panel of sixteen columns in single precision, [k][16],
// as the convolutions pack them (gemm.h): k rounded up to 32 with zeros, and
// for each block, the tile of the first parts, then that of the second. The
// two rows of a pair of k are split, and their parts interleaved, a word of one
// and a word of the other, by vpermw.
//
//     void sdasm_amx_bf16_2x2(int64_t blocks, const uint16_t *a, int64_t a_next,
//                             const uint16_t *b, int64_t b_next, float *c,
//                             int64_t ldc, int64_t accumulate);
//
// C = (accumulate ? C : 0) + A B for a block of 32 rows by 32 columns of C,
// rows ldc bytes apart, of `blocks` blocks of k: A from the tiles
// sdasm_amx_pack_rows packs, those of rows 16 to 31 a_next bytes after those
// of rows 0 to 15; B from those sdasm_amx_pack_weights packs, those of columns
// 16 to 31 b_next bytes after. The four tiles of C stay in tiles 0 to 3; A's
// two in 4 and 5, B's two in 6 and 7. A block of k is twelve products, each
// tile of C taking x1 w1, x1 w2 and x2 w1; the tiles of w1 are loaded twice,
// ten loads in all.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 6
.Lconfig:
        // Palette 1; then, after 14 bytes kept 0, the bytes of a row of each
        // tile, and the rows of each: tiles 0 to 7 of 16 rows of 64 bytes.
        .byte 1, 0
        .zero 14
        .short 64, 64, 64, 64, 64, 64, 64, 64
        .short 0, 0, 0, 0, 0, 0, 0, 0
        .byte 16, 16, 16, 16, 16, 16, 16, 16
        .byte 0, 0, 0, 0, 0, 0, 0, 0
.Lzeros:
        .zero 64
.Linterleave:   // the words of two rows of sixteen, one of each in turn
        .short 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
        .short 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31

        .text

        .globl sdasm_amx_request
        .type sdasm_amx_request, @function
        .p2align 4
sdasm_amx_request:
#if defined(__linux__)
        mov eax, 158                    // SYS_arch_prctl
        mov edi, 0x1023                 // ARCH_REQ_XCOMP_PERM
        mov esi, 18                     // XFEATURE_XTILEDATA
        syscall
#else
        mov rax, -1
#endif
        ret
        .size sdasm_amx_request, .-sdasm_amx_request

        .globl sdasm_amx_configure
        .type sdasm_amx_configure, @function
        .p2align 4
sdasm_amx_configure:
        ldtilecfg [rip + .Lconfig]
        ret
        .size sdasm_amx_configure, .-sdasm_amx_configure

        .globl sdasm_amx_release
        .type sdasm_amx_release, @function
        .p2align 4
sdasm_amx_release:
        tilerelease
        ret
        .size sdasm_amx_release, .-sdasm_amx_release

// ---------------------------------------------------------------- packing

// k1 and k2: the numbers of a block of 32 inside k, of the r8 left: the first
// sixteen and the second.
.macro BLOCK_MASKS
        mov eax, -1
        mov r11, 32
        cmp r8, r11
        cmovb r11, r8
        bzhi eax, eax, r11d
        kmovd k1, eax
        kshiftrd k2, k1, 16
.endm

// The rows of a block of k into tiles of x1 and x2, \type f32 or f16: rdi
// dst, rsi src, rdx stride, rcx rows, r8 k.
.macro PACK_ROWS type
        vpxord zmm6, zmm6, zmm6
        test r8, r8
        jle 9f
1:
        BLOCK_MASKS
        mov r9, rsi
        xor r10, r10
2:
        cmp r10, rcx
        jae 4f
.ifc \type, f32
        vmovups zmm0{k1}{z}, [r9]
        vmovups zmm1{k2}{z}, [r9 + 64]
.else
        vmovdqu16 ymm0{k1}{z}, [r9]
        vmovdqu16 ymm1{k2}{z}, [r9 + 32]
        vcvtph2ps zmm0, ymm0
        vcvtph2ps zmm1, ymm1
.endif
        vcvtne2ps2bf16 zmm2, zmm1, zmm0         // x1
        vpmovzxwd zmm3, ymm2
        vpslld zmm3, zmm3, 16
        vextracti64x4 ymm4, zmm2, 1
        vpmovzxwd zmm4, ymm4
        vpslld zmm4, zmm4, 16
        vsubps zmm0, zmm0, zmm3                 // x - x1, exactly
        vsubps zmm1, zmm1, zmm4
        vcvtne2ps2bf16 zmm5, zmm1, zmm0         // x2
        mov rax, r10
        shl rax, 6
        vmovups [rdi + rax], zmm2
        vmovups [rdi + rax + 1024], zmm5
        add r9, rdx
        inc r10
        cmp r10, 16
        jb 2b
        jmp 5f
4:
        // The rows past the last: zeros.
        mov rax, r10
        shl rax, 6
        vmovups [rdi + rax], zmm6
        vmovups [rdi + rax + 1024], zmm6
        inc r10
        cmp r10, 16
        jb 4b
5:
        add rdi, 2048
.ifc \type, f32
        add rsi, 128
.else
        add rsi, 64
.endif
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
.endm

        .globl sdasm_amx_pack_rows
        .type sdasm_amx_pack_rows, @function
        .p2align 5
sdasm_amx_pack_rows:
        PACK_ROWS f32
        .size sdasm_amx_pack_rows, .-sdasm_amx_pack_rows

        .globl sdasm_amx_pack_rows_f16
        .type sdasm_amx_pack_rows_f16, @function
        .p2align 5
sdasm_amx_pack_rows_f16:
        PACK_ROWS f16
        .size sdasm_amx_pack_rows_f16, .-sdasm_amx_pack_rows_f16

// The transpose of the sixteen rows of sixteen dwords in zmm0..zmm15, into the
// same registers, zmm16..zmm31 its temporaries: the dwords of pairs of rows
// interleaved, then their qwords, then their 128-bit lanes, twice.
.macro TRANSPOSE16
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpckldq zmm24, zmm8, zmm9
        vpunpckhdq zmm25, zmm8, zmm9
        vpunpckldq zmm26, zmm10, zmm11
        vpunpckhdq zmm27, zmm10, zmm11
        vpunpckldq zmm28, zmm12, zmm13
        vpunpckhdq zmm29, zmm12, zmm13
        vpunpckldq zmm30, zmm14, zmm15
        vpunpckhdq zmm31, zmm14, zmm15
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vpunpcklqdq zmm8, zmm24, zmm26
        vpunpckhqdq zmm9, zmm24, zmm26
        vpunpcklqdq zmm10, zmm25, zmm27
        vpunpckhqdq zmm11, zmm25, zmm27
        vpunpcklqdq zmm12, zmm28, zmm30
        vpunpckhqdq zmm13, zmm28, zmm30
        vpunpcklqdq zmm14, zmm29, zmm31
        vpunpckhqdq zmm15, zmm29, zmm31
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xdd
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm21, zmm1, zmm5, 0xdd
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm22, zmm2, zmm6, 0xdd
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm23, zmm3, zmm7, 0xdd
        vshufi32x4 zmm24, zmm8, zmm12, 0x88
        vshufi32x4 zmm28, zmm8, zmm12, 0xdd
        vshufi32x4 zmm25, zmm9, zmm13, 0x88
        vshufi32x4 zmm29, zmm9, zmm13, 0xdd
        vshufi32x4 zmm26, zmm10, zmm14, 0x88
        vshufi32x4 zmm30, zmm10, zmm14, 0xdd
        vshufi32x4 zmm27, zmm11, zmm15, 0x88
        vshufi32x4 zmm31, zmm11, zmm15, 0xdd
        vshufi32x4 zmm0, zmm16, zmm24, 0x88
        vshufi32x4 zmm8, zmm16, zmm24, 0xdd
        vshufi32x4 zmm1, zmm17, zmm25, 0x88
        vshufi32x4 zmm9, zmm17, zmm25, 0xdd
        vshufi32x4 zmm2, zmm18, zmm26, 0x88
        vshufi32x4 zmm10, zmm18, zmm26, 0xdd
        vshufi32x4 zmm3, zmm19, zmm27, 0x88
        vshufi32x4 zmm11, zmm19, zmm27, 0xdd
        vshufi32x4 zmm4, zmm20, zmm28, 0x88
        vshufi32x4 zmm12, zmm20, zmm28, 0xdd
        vshufi32x4 zmm5, zmm21, zmm29, 0x88
        vshufi32x4 zmm13, zmm21, zmm29, 0xdd
        vshufi32x4 zmm6, zmm22, zmm30, 0x88
        vshufi32x4 zmm14, zmm22, zmm30, 0xdd
        vshufi32x4 zmm7, zmm23, zmm31, 0x88
        vshufi32x4 zmm15, zmm23, zmm31, 0xdd
.endm

// Column \c of the block of k into zmm\c, as sixteen pairs: from r9, the row of
// the column while \c < cols (rcx), the zeros after; rdx to the next row.
.macro WIDEN c
        mov r11, r9
        cmp rcx, \c
        cmovbe r11, r10
        vmovdqu16 ymm16{k1}{z}, [r11]
        vmovdqu16 ymm17{k2}{z}, [r11 + 32]
        vcvtph2ps zmm16, ymm16
        vcvtph2ps zmm17, ymm17
        vcvtne2ps2bf16 zmm18, zmm17, zmm16      // w1
        add r9, rdx
.endm
.macro HI c
        WIDEN \c
        vmovdqa64 zmm\c, zmm18
.endm
.macro LO c
        WIDEN \c
        vpmovzxwd zmm19, ymm18
        vpslld zmm19, zmm19, 16
        vextracti64x4 ymm18, zmm18, 1
        vpmovzxwd zmm18, ymm18
        vpslld zmm18, zmm18, 16
        vsubps zmm16, zmm16, zmm19              // w - w1, exactly
        vsubps zmm17, zmm17, zmm18
        vcvtne2ps2bf16 zmm\c, zmm17, zmm16      // w2
.endm
// zmm0..zmm15, rows of the tile, to rdi + \offset.
.macro STORE16 offset
        .irp p, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovups [rdi + \offset + 64 * \p], zmm\p
        .endr
.endm

        .globl sdasm_amx_pack_weights
        .type sdasm_amx_pack_weights, @function
        .p2align 5
// rdi dst, rsi src, rdx stride, rcx cols, r8 k
sdasm_amx_pack_weights:
        test r8, r8
        jle 9f
        lea r10, [rip + .Lzeros]
1:
        // k2 and k1 are masks of sixteen numbers of FP16 here, a word each.
        BLOCK_MASKS
        mov r9, rsi
        HI 0
        HI 1
        HI 2
        HI 3
        HI 4
        HI 5
        HI 6
        HI 7
        HI 8
        HI 9
        HI 10
        HI 11
        HI 12
        HI 13
        HI 14
        HI 15
        TRANSPOSE16
        STORE16 0
        mov r9, rsi
        LO 0
        LO 1
        LO 2
        LO 3
        LO 4
        LO 5
        LO 6
        LO 7
        LO 8
        LO 9
        LO 10
        LO 11
        LO 12
        LO 13
        LO 14
        LO 15
        TRANSPOSE16
        STORE16 1024
        add rdi, 2048
        add rsi, 64
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_weights, .-sdasm_amx_pack_weights

        .globl sdasm_amx_pack_panel
        .type sdasm_amx_pack_panel, @function
        .p2align 5
// rdi dst, rsi src, rdx k
sdasm_amx_pack_panel:
        test rdx, rdx
        jle 9f
        vmovdqu16 zmm31, [rip + .Linterleave]
        lea r10, [rip + .Lzeros]
        xor r8, r8                              // the first k of the block
1:
        xor r9, r9                              // the pair, 0 to 15
2:
        // Rows 2p and 2p + 1 of the block, or zeros past k.
        lea rax, [r8 + 2 * r9]
        mov r11, rax
        shl r11, 6
        add r11, rsi
        cmp rax, rdx
        cmovae r11, r10
        vmovups zmm0, [r11]
        inc rax
        mov r11, rax
        shl r11, 6
        add r11, rsi
        cmp rax, rdx
        cmovae r11, r10
        vmovups zmm1, [r11]
        vcvtne2ps2bf16 zmm2, zmm1, zmm0         // the first parts of the two rows
        vpmovzxwd zmm3, ymm2
        vpslld zmm3, zmm3, 16
        vextracti64x4 ymm4, zmm2, 1
        vpmovzxwd zmm4, ymm4
        vpslld zmm4, zmm4, 16
        vsubps zmm0, zmm0, zmm3                 // what they leave, exactly
        vsubps zmm1, zmm1, zmm4
        vcvtne2ps2bf16 zmm5, zmm1, zmm0         // the second parts
        vpermw zmm2, zmm31, zmm2
        vpermw zmm5, zmm31, zmm5
        mov rax, r9
        shl rax, 6
        vmovups [rdi + rax], zmm2
        vmovups [rdi + rax + 1024], zmm5
        inc r9
        cmp r9, 16
        jb 2b
        add rdi, 2048
        add r8, 32
        cmp r8, rdx
        jb 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_panel, .-sdasm_amx_pack_panel

// ---------------------------------------------------------------- products

        .globl sdasm_amx_bf16_2x2
        .type sdasm_amx_bf16_2x2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_bf16_2x2:
        mov r10, [rsp + 8]
        mov r11, r10
        shl r11, 4
        add r11, r9                             // rows 16 to 31 of C
        add rdx, rsi                            // A's tiles of rows 16 to 31
        add r8, rcx                             // B's tiles of columns 16 to 31
        mov eax, 64                             // the stride of a packed tile
        cmp qword ptr [rsp + 16], 0
        je 1f
        tileloadd tmm0, [r9 + r10 * 1]
        tileloadd tmm1, [r9 + r10 * 1 + 64]
        tileloadd tmm2, [r11 + r10 * 1]
        tileloadd tmm3, [r11 + r10 * 1 + 64]
        jmp 2f
1:
        tilezero tmm0
        tilezero tmm1
        tilezero tmm2
        tilezero tmm3
2:
        test rdi, rdi
        jle 8f
        .p2align 4
3:
        tileloadd tmm4, [rsi + rax * 1]         // x1
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]         // w1
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        tileloadd tmm6, [rcx + rax * 1 + 1024]  // w2
        tileloadd tmm7, [r8 + rax * 1 + 1024]
        tdpbf16ps tmm0, tmm4, tmm6
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        tileloadd tmm4, [rsi + rax * 1 + 1024]  // x2
        tileloadd tmm5, [rdx + rax * 1 + 1024]
        tileloadd tmm6, [rcx + rax * 1]         // w1 again
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        add rsi, 2048
        add rdx, 2048
        add rcx, 2048
        add r8, 2048
        dec rdi
        jnz 3b
8:
        tilestored [r9 + r10 * 1], tmm0
        tilestored [r9 + r10 * 1 + 64], tmm1
        tilestored [r11 + r10 * 1], tmm2
        tilestored [r11 + r10 * 1 + 64], tmm3
        ret
        .size sdasm_amx_bf16_2x2, .-sdasm_amx_bf16_2x2

        .section .note.GNU-stack, "", @progbits
