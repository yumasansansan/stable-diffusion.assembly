// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// Products of single precision on AMX's tiles, for attention (attention.cc),
// which ggml computes in single precision and the kernels are not to compute
// less precisely: AMX-TILE and AMX-BF16, with AVX512-BF16, F, BW and VL.
//
// A number of single precision has 24 bits of significand, and three parts of
// bfloat16, of 8 each, hold it exactly: x1 is x rounded to the nearest, x2 what
// x - x1 leaves, rounded, and x3 what is left after that, rounded -- each
// difference exact in single precision, the last part exact in bfloat16 --
// unless x3 falls below 2^-126, where the tiles take a part as zero: below
// some 10^-33. The product x y is taken as x1 y1 + x1 y2 + x2 y1 + x1 y3 +
// x2 y2 + x3 y1, six products on the tiles, which leave out x2 y3, x3 y2 and
// x3 y3, within 2^-23 of x y between them: about what a product of single
// precision is rounded by, 2^-24. The tiles sum in single precision, and the
// order of the sums matters as much: the five small products of every block
// of k are summed first, while the sums are small and round little, and x1 y1
// after, into sums begun from zeros. So summed, the outputs of attention are
// as close to the exact ones as ggml's in single precision, or closer
// (tests/ops_test.cc); x1 y1 summed with the others, block by block, they were
// up to four times further.
//
//     void sdasm_amx_pack_rows3(uint16_t *dst, const float *src,
//                               int64_t stride, int64_t rows, int64_t k);
//
// The tiles of A of sixteen rows of src, `stride` bytes apart -- rows past
// `rows` are zeros -- over k numbers, rounded up to blocks of 32 with zeros:
// for each block, the tiles of x1, x2 and x3, 3072 bytes a block.
//
//     void sdasm_amx_pack_columns3_f32(uint16_t *dst, const float *src,
//                                      int64_t stride, int64_t cols, int64_t k);
//     void sdasm_amx_pack_columns3_f16(uint16_t *dst, const uint16_t *src,
//                                      int64_t stride, int64_t cols, int64_t k);
//
// The tiles of B from sixteen rows of k numbers, one for each column of B, as
// sdasm_amx_pack_weights packs them (amx.S): a row for each pair of k, the
// pairs of the sixteen columns side by side; each of the three parts a tile,
// 3072 bytes a block of k.
//
//     void sdasm_amx_pack_panel3_f32(uint16_t *dst, const float *src,
//                                    int64_t stride, int64_t cols, int64_t k);
//     void sdasm_amx_pack_panel3_f16(uint16_t *dst, const uint16_t *src,
//                                    int64_t stride, int64_t cols, int64_t k);
//
// The tiles of B from k rows, `stride` bytes apart, of sixteen columns each --
// columns past `cols` are zeros -- as sdasm_amx_pack_panel packs a panel: the
// two rows of a pair interleaved, word by word, by vpermw.
//
//     void sdasm_amx_single_2x2(int64_t blocks, const uint16_t *a,
//                               int64_t a_next, const uint16_t *b,
//                               int64_t b_next, float *c, int64_t ldc,
//                               int64_t accumulate);
//
// sdasm_amx_bf16_2x2 (amx.S) of the three parts: a block of 32 by 32 of C,
// tiles 0 to 3; A's two tiles in 4 and 5, B's in 6 and 7. The small products
// over every block of k, sixteen loads to twenty products a block, then x1 y1
// over them again, four to four. Begun from zeros, or from C when accumulate
// says so -- whose sums the small products then round as much as the large.
//
// Assembled by Clang, in Intel's syntax, for the System V calling convention.

        .intel_syntax noprefix

        .section .rodata
        .p2align 6
.Lzeros3:
        .zero 128
.Linterleave3:  // the words of two rows of sixteen, one of each in turn
        .short 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
        .short 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31

        .text

// k1 and k2: the numbers of a block of 32 inside k, of the r8 left: the first
// sixteen and the second.
.macro BLOCK_MASKS3
        mov eax, -1
        mov r11, 32
        cmp r8, r11
        cmovb r11, r8
        bzhi eax, eax, r11d
        kmovd k1, eax
        kshiftrd k2, k1, 16
.endm

        .globl sdasm_amx_pack_rows3
        .type sdasm_amx_pack_rows3, @function
        .p2align 5
// rdi dst, rsi src, rdx stride, rcx rows, r8 k
sdasm_amx_pack_rows3:
        vpxord zmm8, zmm8, zmm8
        test r8, r8
        jle 9f
1:
        BLOCK_MASKS3
        mov r9, rsi
        xor r10, r10
2:
        mov rax, r10
        shl rax, 6
        cmp r10, rcx
        jae 4f
        vmovups zmm0{k1}{z}, [r9]
        vmovups zmm1{k2}{z}, [r9 + 64]
        vcvtne2ps2bf16 zmm2, zmm1, zmm0
        vpmovzxwd zmm5, ymm2
        vpslld zmm5, zmm5, 16
        vextracti64x4 ymm6, zmm2, 1
        vpmovzxwd zmm6, ymm6
        vpslld zmm6, zmm6, 16
        vsubps zmm0, zmm0, zmm5
        vsubps zmm1, zmm1, zmm6
        vcvtne2ps2bf16 zmm3, zmm1, zmm0
        vpmovzxwd zmm7, ymm3
        vpslld zmm7, zmm7, 16
        vextracti64x4 ymm9, zmm3, 1
        vpmovzxwd zmm9, ymm9
        vpslld zmm9, zmm9, 16
        vsubps zmm0, zmm0, zmm7
        vsubps zmm1, zmm1, zmm9
        vcvtne2ps2bf16 zmm4, zmm1, zmm0
        vmovups [rdi + rax], zmm2
        vmovups [rdi + rax + 1024], zmm3
        vmovups [rdi + rax + 2048], zmm4
        jmp 5f
4:
        vmovups [rdi + rax], zmm8
        vmovups [rdi + rax + 1024], zmm8
        vmovups [rdi + rax + 2048], zmm8
5:
        add r9, rdx
        inc r10
        cmp r10, 16
        jb 2b
        add rdi, 3072
        add rsi, 128
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
        .size sdasm_amx_pack_rows3, .-sdasm_amx_pack_rows3

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

// Column \c of the block of k widened into zmm16 and zmm17: from r9, the row of
// the column while \c < cols (rcx), the zeros after; rdx to the next row.
.macro WIDEN3 type, c
        mov r11, r9
        cmp rcx, \c
        cmovbe r11, r10
.ifc \type, f32
        vmovups zmm16{k1}{z}, [r11]
        vmovups zmm17{k2}{z}, [r11 + 64]
.else
        vmovdqu16 ymm16{k1}{z}, [r11]
        vmovdqu16 ymm17{k2}{z}, [r11 + 32]
        vcvtph2ps zmm16, ymm16
        vcvtph2ps zmm17, ymm17
.endif
        add r9, rdx
.endm
// Part \part (1, 2 or 3) of column \c into zmm\c, as sixteen pairs.
.macro PART3 type, part, c
        WIDEN3 \type, \c
.if \part == 1
        vcvtne2ps2bf16 zmm\c, zmm17, zmm16
.else
        vcvtne2ps2bf16 zmm18, zmm17, zmm16
        vpmovzxwd zmm21, ymm18
        vpslld zmm21, zmm21, 16
        vextracti64x4 ymm22, zmm18, 1
        vpmovzxwd zmm22, ymm22
        vpslld zmm22, zmm22, 16
        vsubps zmm16, zmm16, zmm21
        vsubps zmm17, zmm17, zmm22
        vcvtne2ps2bf16 zmm19, zmm17, zmm16
        vpmovzxwd zmm23, ymm19
        vpslld zmm23, zmm23, 16
        vextracti64x4 ymm24, zmm19, 1
        vpmovzxwd zmm24, ymm24
        vpslld zmm24, zmm24, 16
        vsubps zmm16, zmm16, zmm23
        vsubps zmm17, zmm17, zmm24
        vcvtne2ps2bf16 zmm20, zmm17, zmm16
.if \part == 2
        vmovdqa64 zmm\c, zmm19
.else
        vmovdqa64 zmm\c, zmm20
.endif
.endif
.endm
.macro STORE16_3 offset
        .irp p, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vmovups [rdi + \offset + 64 * \p], zmm\p
        .endr
.endm
// rdi dst, rsi src, rdx stride, rcx cols, r8 k; \bytes the bytes of a block
// of k of a row of the source.
.macro PACK_COLUMNS3 type, bytes
        test r8, r8
        jle 9f
        lea r10, [rip + .Lzeros3]
1:
        BLOCK_MASKS3
        .irp part, 1, 2, 3
        mov r9, rsi
        .irp c, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        PART3 \type, \part, \c
        .endr
        TRANSPOSE16
        STORE16_3 "(1024 * (\part - 1))"
        .endr
        add rdi, 3072
        add rsi, \bytes
        sub r8, 32
        jg 1b
9:
        vzeroupper
        ret
.endm

        .globl sdasm_amx_pack_columns3_f32
        .type sdasm_amx_pack_columns3_f32, @function
        .p2align 5
sdasm_amx_pack_columns3_f32:
        PACK_COLUMNS3 f32, 128
        .size sdasm_amx_pack_columns3_f32, .-sdasm_amx_pack_columns3_f32

        .globl sdasm_amx_pack_columns3_f16
        .type sdasm_amx_pack_columns3_f16, @function
        .p2align 5
sdasm_amx_pack_columns3_f16:
        PACK_COLUMNS3 f16, 64
        .size sdasm_amx_pack_columns3_f16, .-sdasm_amx_pack_columns3_f16

// rdi dst, rsi src, rdx stride, rcx cols, r8 k: rows 2p and 2p + 1 of each
// block, sixteen columns of each masked to cols by k3, interleaved in each part.
.macro PACK_PANEL3 type
        test r8, r8
        jle 9f
        vmovdqu16 zmm31, [rip + .Linterleave3]
        lea r10, [rip + .Lzeros3]
        mov eax, -1
        bzhi eax, eax, ecx
        kmovd k3, eax
        xor r9, r9                              // the first k of the block
1:
        xor rcx, rcx                            // the pair, 0 to 15
2:
        lea rax, [r9 + 2 * rcx]
        mov r11, rax
        imul r11, rdx
        add r11, rsi
        cmp rax, r8
        cmovae r11, r10
.ifc \type, f32
        vmovups zmm0{k3}{z}, [r11]
.else
        vmovdqu16 ymm0{k3}{z}, [r11]
        vcvtph2ps zmm0, ymm0
.endif
        inc rax
        mov r11, rax
        imul r11, rdx
        add r11, rsi
        cmp rax, r8
        cmovae r11, r10
.ifc \type, f32
        vmovups zmm1{k3}{z}, [r11]
.else
        vmovdqu16 ymm1{k3}{z}, [r11]
        vcvtph2ps zmm1, ymm1
.endif
        vcvtne2ps2bf16 zmm2, zmm1, zmm0
        vpmovzxwd zmm5, ymm2
        vpslld zmm5, zmm5, 16
        vextracti64x4 ymm6, zmm2, 1
        vpmovzxwd zmm6, ymm6
        vpslld zmm6, zmm6, 16
        vsubps zmm0, zmm0, zmm5
        vsubps zmm1, zmm1, zmm6
        vcvtne2ps2bf16 zmm3, zmm1, zmm0
        vpmovzxwd zmm7, ymm3
        vpslld zmm7, zmm7, 16
        vextracti64x4 ymm9, zmm3, 1
        vpmovzxwd zmm9, ymm9
        vpslld zmm9, zmm9, 16
        vsubps zmm0, zmm0, zmm7
        vsubps zmm1, zmm1, zmm9
        vcvtne2ps2bf16 zmm4, zmm1, zmm0
        vpermw zmm2, zmm31, zmm2
        vpermw zmm3, zmm31, zmm3
        vpermw zmm4, zmm31, zmm4
        mov rax, rcx
        shl rax, 6
        vmovups [rdi + rax], zmm2
        vmovups [rdi + rax + 1024], zmm3
        vmovups [rdi + rax + 2048], zmm4
        inc rcx
        cmp rcx, 16
        jb 2b
        add rdi, 3072
        add r9, 32
        cmp r9, r8
        jb 1b
9:
        vzeroupper
        ret
.endm

        .globl sdasm_amx_pack_panel3_f32
        .type sdasm_amx_pack_panel3_f32, @function
        .p2align 5
sdasm_amx_pack_panel3_f32:
        PACK_PANEL3 f32
        .size sdasm_amx_pack_panel3_f32, .-sdasm_amx_pack_panel3_f32

        .globl sdasm_amx_pack_panel3_f16
        .type sdasm_amx_pack_panel3_f16, @function
        .p2align 5
sdasm_amx_pack_panel3_f16:
        PACK_PANEL3 f16
        .size sdasm_amx_pack_panel3_f16, .-sdasm_amx_pack_panel3_f16

        .globl sdasm_amx_single_2x2
        .type sdasm_amx_single_2x2, @function
        .p2align 5
// rdi blocks, rsi a, rdx a_next, rcx b, r8 b_next, r9 c, [rsp + 8] ldc,
// [rsp + 16] accumulate
sdasm_amx_single_2x2:
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
        // The small products first, over every block of k, while the sums are
        // small, so that each rounds as little as it can; x1 y1 after.
        push rsi
        push rdx
        push rcx
        push r8
        push rdi
        .p2align 4
3:
        tileloadd tmm4, [rsi + rax * 1]
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1 + 1024]
        tileloadd tmm7, [r8 + rax * 1 + 1024]
        tdpbf16ps tmm0, tmm4, tmm6              // x1 y2
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        tileloadd tmm6, [rcx + rax * 1 + 2048]
        tileloadd tmm7, [r8 + rax * 1 + 2048]
        tdpbf16ps tmm0, tmm4, tmm6              // x1 y3
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        tileloadd tmm4, [rsi + rax * 1 + 1024]
        tileloadd tmm5, [rdx + rax * 1 + 1024]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6              // x2 y1
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        tileloadd tmm6, [rcx + rax * 1 + 1024]
        tileloadd tmm7, [r8 + rax * 1 + 1024]
        tdpbf16ps tmm0, tmm4, tmm6              // x2 y2
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        tileloadd tmm4, [rsi + rax * 1 + 2048]
        tileloadd tmm5, [rdx + rax * 1 + 2048]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6              // x3 y1
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        add rsi, 3072
        add rdx, 3072
        add rcx, 3072
        add r8, 3072
        dec rdi
        jnz 3b
        pop rdi
        pop r8
        pop rcx
        pop rdx
        pop rsi
        .p2align 4
4:
        tileloadd tmm4, [rsi + rax * 1]
        tileloadd tmm5, [rdx + rax * 1]
        tileloadd tmm6, [rcx + rax * 1]
        tileloadd tmm7, [r8 + rax * 1]
        tdpbf16ps tmm0, tmm4, tmm6              // x1 y1
        tdpbf16ps tmm1, tmm4, tmm7
        tdpbf16ps tmm2, tmm5, tmm6
        tdpbf16ps tmm3, tmm5, tmm7
        add rsi, 3072
        add rdx, 3072
        add rcx, 3072
        add r8, 3072
        dec rdi
        jnz 4b
8:
        tilestored [r9 + r10 * 1], tmm0
        tilestored [r9 + r10 * 1 + 64], tmm1
        tilestored [r11 + r10 * 1], tmm2
        tilestored [r11 + r10 * 1 + 64], tmm3
        ret
        .size sdasm_amx_single_2x2, .-sdasm_amx_single_2x2

        .section .note.GNU-stack, "", @progbits
