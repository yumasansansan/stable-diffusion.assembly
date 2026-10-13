<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

This file is part of stable-diffusion.assembly. The SPDX lines name its
copyright holder and its license, in the machine-readable form of the REUSE
specification: the GNU General Public License, version 3 or any later version
(LICENSES/GPL-3.0-or-later.txt).
-->

# stable-diffusion.assembly
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) on the CPU, with the operations that take its time computed by kernels of its own: in plain C++ that the compiler vectorizes for any processor, and by hand in assembly for each level of x86-64's instruction set, chosen at run time.

English | [日本語](README.ja.md)

## What it is

stable-diffusion.cpp computes its models through [ggml](https://github.com/ggml-org/ggml)'s CPU backend. This project leaves stable-diffusion.cpp and ggml as they are but for a few small patches (`patches/`) through which ggml's CPU backend offers its heaviest operations to the kernels of `sources/`, which compute them instead:

- **Matrix products** (`MUL_MAT`, the linear layers), **convolutions** (`CONV_2D`) and **attention** (`FLASH_ATTN_EXT`), by a matrix product of packed panels that a micro-kernel computes a tile at a time, its work shared out among the threads in blocks that shrink toward the end, so that the fast and the slow cores of a hybrid processor finish together.
- The convolutions of three by three by **Winograd's F(4x4, 3x3)** on images of 16 by 16 and more, four times fewer multiplications, and **F(2x2, 3x3)** on the 8 by 8 images of the deepest layers, 2.25 times fewer, each with a path of its own for images of few tiles; the others **directly**, without the copies of the image that im2col writes.
- **Attention** of a block of queries kept in the cache from the scores to the output, the softmax's exponentials taken a stretch of keys at a time just before the values are summed by them.
- The operations after a product -- the bias, the scale, GroupNorm and LayerNorm with their scale and shift, SiLU, the gated GELU of GEGLU, the residual connection -- **fused** into it, computed while its output is in the cache.

Each kernel has three forms: `cxx`, plain C++ with no intrinsics, which the compiler vectorizes for AVX2, AVX-512 or NEON alike; `assembly`, written by hand for each level of the instruction set (`sources/x86_64/`), the default; and `upstream`, none of this project's, ggml's own. The form, and the level, are chosen at run time.

## How fast

SD 1.5 at 512 by 512, 20 steps, on an Intel Core i7-1360P (4 Golden Cove and 8 Gracemont cores, AVX2, at its power limit), the sampler and the VAE's decoder:

| | Sampling (20 steps) | VAE decoder | |
| --- | ---: | ---: | --- |
| stable-diffusion.cpp, Release, `-march=native`, its best number of threads (8) | 387.45 s | 32.93 s | |
| stable-diffusion.assembly, 12 threads | 70.8 s | 3.16 s | 5.5 and 10.4 times as fast |

The UNet's time goes to the convolutions (about 37%), the linear layers (37%) and attention (22%). The linear layers reach about 70% of the most FMAs the processor makes at its power limit, and attention 60%; the convolutions, through Winograd's transforms, 1.2 to 2.5 times it, counted in the operations of the direct convolution. The profile (`SDASM_PROFILE=1`) breaks each operation down by shape and phase.

SDXL (Hassaku XL, at 832 by 1216, CFG 6, Euler A) on the same processor, with other work running beside it -- the steps of two runs differ by up to 8% -- a step past the first two, the mean of two runs that took turns, and the VAE's decoder:

| | A step | VAE decoder | |
| --- | ---: | ---: | --- |
| stable-diffusion.cpp, as above (8 threads, two steps) | 131 s | 136 s | |
| stable-diffusion.assembly, 12 threads | 39.3 s | 15.1 s | 3.3 and 9.0 times as fast |
| `bits=8` (below) | 16.2 s | 11.6 s | 8.1 and 11.7 times |

The linear layers of SDXL's transformers, ten deep on the grid of 26 by 38, take two thirds of its UNet's time (66%; attention 17%, the convolutions 16%), and ggml computes them better than its convolutions. `bits=8`'s image is the same picture, 24.3 dB of PSNR off the exact one (SSIM 0.90). Each takes 9 to 12 GB of memory at its most, the model's file mapped.

Processors other than this one are measured on GitHub's runners (see [Measuring on GitHub's runners](#measuring-on-githubs-runners)).

## Precision

The kernels never compute less precisely than stable-diffusion.cpp does. ggml rounds the image to FP16 before its products whenever the weights are FP16 -- the linear layers and the convolutions of SD 1.5 -- and sums the products of the rounded numbers; the kernels for AVX2 and AVX-512 keep the image in single precision, and their outputs are one or two orders of magnitude closer to the exact ones. The tests (`tests/ops_test.cc`) measure every form and every level against the same computed in double precision, the error divided by the sum of the magnitudes of the products summed:

| Operation | AVX2 / AVX-512 / C++ | AMX | ggml (upstream) |
| --- | ---: | ---: | ---: |
| Linear layers (FP16 weights) | 0.4 to 2.4 × 10⁻⁷ | 0.5 to 1.8 × 10⁻⁶ (AMX-BF16), 3.5 to 7.0 × 10⁻⁸ (AMX-FP16) | 2.9 to 19 × 10⁻⁵ |
| Convolutions, F(2x2, 3x3) and direct (FP16 weights) | 0.7 to 2.1 × 10⁻⁷ | 1.2 to 6.8 × 10⁻⁶; direct on AMX-FP16, 0.6 to 0.8 × 10⁻⁷ | 3.9 to 13 × 10⁻⁵ |
| Convolutions, F(4x4, 3x3) (FP16 weights, 16 by 16 and more) | 5 to 11 × 10⁻⁷ | (F(2x2, 3x3) on AMX) | 4.1 to 13 × 10⁻⁵ |
| Attention (single precision) | 1.4 to 19 × 10⁻⁸, as close as ggml's graph, or closer | 1.3 to 7.1 × 10⁻⁸ | 2.0 to 16 × 10⁻⁸ (its graph) |

**On AMX's tiles**, which multiply bfloat16 (8 bits of significand, against FP16's 11), each number is split in two parts, x = x1 + x2 and w = w1 + w2, each rounded to the nearest, and a product taken as x1 w1 + x1 w2 + x2 w1: within 2⁻¹⁵ of the exact one, sixteen times as close as ggml's rounding to FP16 (2⁻¹¹). Where the processor has AMX-FP16 (Granite Rapids), the FP16 weights go to the tiles as they are and the other operand in two parts of FP16, x1 being ggml's own rounding and x2 what it leaves: two products for each one, w x1 + w x2, each exact, within 2⁻²² of the exact product and never further from it than ggml's -- as close as single precision, and a third fewer products. Winograd's products, of two operands of single precision, keep the two parts of bfloat16. Attention, which ggml computes in single precision, is computed in single precision on the tiles too, when `amx_attention=1` asks for it: three parts of bfloat16 hold a number of single precision exactly, and a product is taken as six products, x1 y1 + x1 y2 + x2 y1 + x1 y3 + x2 y2 + x3 y1, the five small ones summed first, while the sums are small, and x1 y1 after; so summed, the outputs are as close to the exact ones as ggml's or closer, where summed block by block they were up to four times further. Six products for each one and the heads of 40 and 80 dimensions of SD 1.5, padded to 64 and 96, made it 1.4 to 2.1 times slower than AVX-512's on Emerald Rapids, though 7% faster for the VAE's one head of 512, and by default attention stays on the vector units. The GELU of GEGLU is computed in single precision, where ggml reads it from a table of FP16.

## Less precision, more speed

### Products quantized to integers

With `SDASM_TUNE=bits=8` the products are computed in integers instead -- stable-diffusion.cpp's image no longer, but an image of the same model, in less time. Every number of a product's two operands is rounded to an integer of a few bits times a scale, the products of the integers are summed exactly in 32 bits, as VNNI's `vpdpbusd` sums them, four products of a byte by a byte to a lane, and the sum is scaled back:

- the **weights** are signed, each output channel's scale shared by a group of K (`group`, 64 numbers by default), quantized once and kept from one step to the next;
- the **inputs** are unsigned with a zero point, each row's -- a token, a position of the image, a tile of Winograd's transform -- in groups of K too: the activations after SiLU and GELU lie mostly on one side of 0, and a symmetric scale wastes a bit of them;
- the convolutions of three by three by **F(2x2, 3x3)**, each element of a transformed tile its own scales (*tap-wise*, as [Andri et al., 2022](https://arxiv.org/abs/2209.12982) quantize F(4x4, 3x3)): F(4x4, 3x3) in integers makes 2.3 times more of the rounding than F(2x2), and the direct convolution 1.3 times less, with 2.25 times the products; the others directly;
- **self-attention**'s scores of queries and keys, each over the head's dimensions, and the sum of the values by the exponentials, both in groups of keys, the exponentials each query's: of one scale for all the keys, an exponential of 8 bits rounds the small ones away. **Cross-attention**, over the prompt's 77 tokens, stays in single precision: a fiftieth of the work, and ten times the rounding;
- the linear layers with the gated GELU of GEGLU after them, fused; those of fewer than 256 rows -- the text encoder's, the projections of the prompt to cross-attention's keys and values, the time embedding's -- stay in single precision, a few hundredths of the work that steer the whole picture.

The micro-kernels are those of single precision in integers: 6 by 16 for AVX-VNNI (`x86_64/quant_avx2.S`, the VEX `vpdpbusd` of Alder Lake's cores and those after), 6 by 16 and 6 by 32 for AVX512-VNNI (`quant_avx512.S`), and the C++ form, whose integers the compiler vectorizes; the quantization of a panel of sixteen columns by AVX2 (`quantize_avx2.S`). On AMX-INT8's tiles (Sapphire Rapids and after), the linear layers' products, in blocks of 32 by 32 (`amx_int8.S`): a group's products summed by `tdpbusd`, sixteen rows by sixteen columns by 64 numbers of k to an instruction, and scaled by AVX-512 while the tiles sum the next group's, for groups of a multiple of 64 numbers of k. The scaling takes about twice the time of the products with groups of 64, as long with groups of 128, so that the tiles' integers are about as fast as `bits=16`'s products of bfloat16 with groups of 64 and twice as fast with groups of 128 or more -- what the workflow `Measure` is to tell (the variants `int8`, `int8-g128`, `int8-g256` and `int8-avx512`). All of them give the same integers, and the same outputs to the bit, but for attention's exponentials (`tests/ops_test.cc quantized`).

On AVX2 without VNNI -- Haswell to Coffee Lake, Zen to Zen 3 -- the kernel of 6 by 16 multiplies by `vpmaddubsw` instead (`quant_maddubs.S`): 32 unsigned bytes by signed ones, each two products summed into a word, which `vpmaddwd` by ones sums in pairs into the lanes of 32 bits that `vpdpbusd` sums four products into, and `vpaddd` adds to the sums. The integers are the same as long as two products cannot pass a word's 32767: with weights of 7 bits and inputs of 8 (`bits=7`), or weights of 8 and inputs of 7 (`input_bits=7`); with both of 8, those processors stay in single precision. Its two multiplications for VNNI's one make twice the products of the FMAs where the processor has two multipliers of vector integers -- Skylake's cores and those after, Zen 3 -- and as many where it has one: Haswell, Broadwell and Zen 2 gain nothing. On the i7-1360P with AVX-VNNI taken for absent, a P-core's kernel makes 1.55 times the products of its FMAs (AVX-VNNI's 3.0 times) and an E-core's as many (1.8 times); on one P-core the linear layers are 1.45 to 1.7 times as fast as in single precision and self-attention 1.3 times, but the convolutions of three by three only 1.0 to 1.25 times, and the VAE's largest 0.9 times, for the integers' F(2x2, 3x3) takes 1.8 times the products of single precision's F(4x4, 3x3). SD 1.5 samples 1.18 times as fast with `bits=7`, and as fast with `quantize=5`, which keeps the convolutions in single precision, the VAE as fast as without, and the images closer to the exact ones: of the eight prompts above, 26.0 dB of PSNR off them in the median (17.7 to 35.9) and an SSIM of 0.90, against 23.2 dB and 0.84 with `bits=7` alone, 23.2 dB and 0.83 with `input_bits=7`, and `bits=8`'s 25.4 dB and 0.89. On GitHub's Zen 3 (EPYC 7763, 36 runners of the workflow `Measure`, the variants `int7-avx2` and `int8a7-avx2`) the same: the linear layers 1.55 to 1.74 times as fast as in single precision, self-attention 1.33 to 1.39 times and the VAE's 1.75 times, the convolutions of three by three 0.92 to 1.17 times on the UNet's grids of 16 by 16 and more, and 0.87 to 1.02 times the VAE's.

On the i7-1360P, a P-core makes four times the products of its FMAs with `vpdpbusd`, an E-core twice as many. SD 1.5 at 512 by 512, 20 steps:

| | Sampling (20 steps) | VAE decoder |
| --- | ---: | ---: |
| single precision (the default) | 71.3 s | 3.18 s |
| `bits=8` | 38.7 s | 2.31 s |

Measured on SD 1.5's own activations, each operation's output is within 5 × 10⁻³ of the exact one in the L2 norm for the linear layers, 6 to 7 × 10⁻³ for the convolutions and 1 to 4 × 10⁻³ for self-attention. The images are of the same quality, but the sampler -- Euler A by default, which adds noise at each step -- carries the difference through its twenty steps: of eight prompts, seven give the same picture, its details moved, and one another picture; their PSNR off the exact images is 25.4 dB in the median (17.9 to 34.2), their SSIM 0.89 (0.77 to 0.98), and with `group=32` 26.5 dB and 0.89. stable-diffusion.cpp's own models quantized to 8 bits (`--type q8_0`, W8A8 in groups of 32 on the linear layers alone) move them too: 34 dB off on the first prompt, where `bits=8` is 34 dB off, and 23 dB on the one that `bits=8` takes to another picture.

| Tuning | |
| --- | --- |
| `bits=8\|7\|6\|4` | the weights' bits; the inputs keep `input_bits` (8). 7, or `input_bits=7`, for AVX2 without VNNI |
| `input_bits=2..8` | the inputs' |
| `group=32\|64\|128` | the numbers of K that share a scale: 32 closer to the exact image, 128 no faster than 64 here |
| `weight_group=0` | the weights' scale for the whole of K rather than each group |
| `quantize=1..7` | the operations quantized: 1 the linear layers, 2 the convolutions, 4 attention, added (7, all, by default); 5 for AVX2 without VNNI |

Of the 4-bit weights, as of 6, the image suffers more than the time gains: the integers are bytes either way, and the products the same. Without VNNI, 7 bits on one side are what lets the products be of integers at all.

### Products of 16 bits

With `SDASM_TUNE=bits=16`, on a processor that has them, the products are of numbers of 16 bits, one product for each one: where the exact path splits each number in two or three parts on AMX's tiles, this takes the first parts alone (`x86_64/half.S`).

- On AMX's tiles, bfloat16, x1 w1, each product within 2⁻⁸ of the exact one; with AMX-FP16 (Granite Rapids), the FP16 weights as they are and the other operand rounded to FP16, w x1, each product exact -- ggml's own product, as precise as stable-diffusion.cpp; attention's x1 y1 of bfloat16.
- On the vector units of a processor with AVX-512 and no AMX: AVX10.2's `vdpphps`, pairs of FP16 summed in single precision, AMX-FP16's products; without AVX10.2, AVX512-FP16's `vfmadd231ph` (Intel's cores since Sapphire Rapids), the FP16 weights' products exact, summed in FP16, each sum moved into single precision after 32 products, so that the other operand and the sums of 32 products are held to FP16's range, 65504, as on AMX-FP16's tiles; without either, AVX512-BF16's `vdpbf16ps` (Zen 4 and 5). Intel's cores split `vdpbf16ps` into four µops, two cycles apart: half the products of their FMAs, where `vfmadd231ph` makes twice them, with half the error of bfloat16's. `half_format=1` takes bfloat16 and `half_format=2` AVX512-FP16 wherever they are.
- Winograd's products, of two operands of single precision, are of bfloat16 on the tiles and stay in single precision on the vector units: of 16 bits, F(2x2, 3x3) would take 2.25 times the products of single precision's F(4x4, 3x3). Attention's are of bfloat16 on the tiles, and on the vector units where bfloat16 is the format; elsewhere they stay in single precision.

Elsewhere -- AVX2 alone, as on the i7-1360P -- the products stay in single precision. `tests/ops_test.cc half` holds each form to 8 × 10⁻³ of the magnitudes summed, under SDE as Cooper Lake, Sapphire Rapids, Granite Rapids, Nova Lake (AVX10.2) and Diamond Rapids; how fast they are, the workflow `Measure` times on GitHub's runners (the variants `half`, `half-avx512`, `half-bf16` and `half-fp16`). The linear layers on AVX512-BF16's vector units are 1.5 to 1.9 times as fast as in single precision on Zen 4 and 1.0 to 1.6 times on Zen 5, but 0.55 to 0.65 times on Emerald Rapids and Granite Rapids, whose vector units take AVX512-FP16's instead, 1.1 to 1.7 times, and whose default is the tiles: 1.3 to 1.8 and 1.1 to 1.3 times as fast as the exact path on them. Winograd's of bfloat16 on the vector units were 0.47 to 0.68 times as fast on the grids of 32 by 32 of Zen 4 and 5, 0.26 to 0.66 times on those Intel's, and attention's 0.29 to 0.65 times on those Intel's.

### Attention's keys and values from a coarser grid

`SDASM_TUNE=todo=2` takes the keys and values of attention of an image's tokens with themselves from every second row and column of the grid of tokens -- the token nearest each point of a grid of a quarter of the tokens -- and the queries from every token: ToDo's downsampling of the keys and values by their nearest neighbours ([Smith, Saxena and Saha, 2024](https://arxiv.org/abs/2402.13573)). `todo=3` takes every third row and column; `todo_tokens` (2048) is the least tokens of a grid it is taken for -- in SD 1.5 at 512 by 512, the grid of 64 by 64 and not that of 32 by 32. stable-diffusion.cpp's blocks of attention ask for the factor (`patches/stable-diffusion.cpp/0004-...`) and take the coarse tokens before their projections to keys and values, which project a quarter of them.

SD 1.5 at 512 by 512 on the i7-1360P, sixteen threads, the steps past the processor's first boost:

| | A step | |
| --- | ---: | ---: |
| single precision | 4.31 s | |
| `todo=2` | 3.46 s | 1.25 times as fast |
| `bits=8` | 2.29 s | 1.89 times |
| `bits=8,todo=2` | 1.91 s | 2.26 times |

Of the eight prompts of the quantized products, at 20 steps, the images of `todo=2` are 20.5 dB of PSNR off the exact ones in the median (16.8 to 27.8), their SSIM 0.74 (0.64 to 0.91): the same pictures for most, their details moved more than by `bits=8`, and the lighthouse's composition changed. `todo=3` is 18.6 dB off. The paper measured its method at 1536 by 1536 and more, where neighbouring tokens are nearer alike than at 512. On SDXL at 832 by 1216, whose first grid, of 104 by 152, has no attention, ToDo takes the grid of 52 by 76 alone, and a step is 1.06 times as fast (1.16 and 0.98 in two rounds; with `bits=8`, 1.02), its image 20.5 dB off the exact one (SSIM 0.83): that grid's self-attention is about half of SDXL's attention, which is 17% of its UNet's time.

## The levels of the instruction set

| Level | Extensions | What the kernels written by hand use it for |
| --- | --- | --- |
| `avx2` | AVX2, FMA, F16C (x86-64-v3) | the micro-kernels (6 by 16), those of integers by AVX-VNNI where there is, or else by `vpmaddubsw`, the softmax's exponentials, the packing, the transforms of F(2x2, 3x3) and F(4x4, 3x3), the gated GELU |
| `avx512` | AVX-512 F, VL, BW, DQ (AVX10.1) | the same, sixteen lanes wide; masked loads at the edges of an image; a micro-kernel of 6 by 32, whose loads keep up with two FMAs of sixteen a cycle: on GitHub's runners, the linear layers and the convolutions 18 to 60% faster than with 6 by 16 on Ice Lake-SP, 8 to 21% on Zen 5, and on Zen 4, whose units take 512 bits in two halves, as fast; attention's two products take two panels of queries at a time with it |
| `amx` | AMX-TILE, AMX-BF16, AVX512-BF16; AMX-FP16 and AMX-INT8 where there are | the products of the linear layers and of the convolutions (Winograd's included) on the tiles, in bfloat16 split in two parts, or with AMX-FP16 the weights as they are and the other operand in two parts of FP16; with `amx_attention=1`, attention's in three parts of bfloat16; with `bits=8`, the linear layers' integers by AMX-INT8; Winograd's transform of the weights for them by AVX-512; all else as `avx512` |

The level in effect is the richest the processor has; `SDASM_ISA=avx2|avx512|amx` holds it to a level below, and the tests compute every operation at every level the processor has. A build for x86-64-v3 (`SDASM_ARCH=avx2`, the one that is shipped) runs every level; the C++ is then compiled for AVX2.

Of AVX-512's other extensions, none speeds up these operations at that precision. VNNI (and AVX-VNNI) multiplies integers, which the numbers have to be quantized to: the quantized products above. AVX512-BF16's `vdpbf16ps` makes 32 products to an FMA's 16, but the two parts make three products of each one, half again as many instructions as the FMAs of single precision, which llvm-mca's models of Cooper Lake, Sapphire Rapids, Zen 4 and Zen 5 run no faster; it is kept as a variant to measure on real processors (`SDASM_TUNE=bf16=1`, on the tiles' packing). AVX512-FP16 sums in FP16 and would round every partial sum to 11 bits. IFMA, VBMI, VBMI2, BITALG, VPOPCNTDQ, VP2INTERSECT, GFNI, VAES and VPCLMULQDQ have nothing to do with single precision. AMX's tiles are another matter: they make sixteen times the products of AVX-512's FMAs, and with three products for each one, five times as many.

## Building

On Linux, x86-64 (elsewhere the C++ kernels alone, for now), with Clang, LLD and the LLVM tools of version 23 from [apt.llvm.org](https://apt.llvm.org), CMake 3.29 or newer and Ninja. `ci/setup.sh` installs them on Ubuntu as the CI does.

```bash
git clone --recursive https://github.com/yumasansansan/stable-diffusion.assembly.git
```

```bash
cmake --preset release
```

```bash
cmake --build --preset release
```

The presets (`cmake --list-presets`): `release` (-O3, ThinLTO, for x86-64-v3 by default), `debug`, `relwithdebinfo`, `sanitize` and `tsan` (address and undefined, thread and undefined), and `emulated`, the tests for x86-64-v3 for Intel SDE, every index checked (`SDASM_CHECKS`, as in `debug` and the sanitizers' presets). `-DSDASM_ARCH=native` builds the C++ for the machine that builds it.

The program is stable-diffusion.cpp's, `build/release/bin/sd-cli`, with all its options:

```bash
build/release/bin/sd-cli -m v1-5-pruned-emaonly.safetensors -p "a lovely cat" --steps 20 -t 12
```

### Environment

| Variable | |
| --- | --- |
| `SDASM_KERNELS=assembly\|cxx\|upstream` | the form of the kernels (assembly by default) |
| `SDASM_ISA=avx2\|avx512\|amx` | the richest level the kernels written by hand take |
| `SDASM_TUNE=name=value,...` | the sizes of the blocks and the choices of the kernels (`Tuning` in `sources/kernels.h`): `bits=8` the quantized products, `bits=16` the products of 16 bits, `todo=2` attention's keys and values from a coarser grid, `wide=0` the micro-kernel of 6 by 16 on AVX-512 rather than 6 by 32, `bf16=1` the products of AVX512-BF16, `amx_fp16=0` AMX-BF16's form on a processor with AMX-FP16, `amx_attention=1` attention on the tiles rather than the vector units |
| `SDASM_PROFILE=1` | each operation's time, by shape and by phase, at the end |
| `SDASM_STORE_MB=n` | keep up to n MiB of packed weights from one step to the next; the quantized ones are kept unless it is given (up to 8 GiB) |

## Testing

The tests and the benchmark are built with `-DSDASM_BUILD_TESTS=ON -DSDASM_BUILD_TOOLS=ON` (`ci/build.sh release avx2` configures so, checks the toolchain and builds):

```bash
cmake --preset release -DSDASM_BUILD_TESTS=ON -DSDASM_BUILD_TOOLS=ON
```

```bash
cmake --build --preset release
```

```bash
ctest --preset release
```

`sdasm_ops_test <operation> [threads]` computes each operation in every form and at every level against double precision, and prints how close each came; `ci/test.sh` runs them all. `ci/emulated.sh <sde64>` runs the tests of the `emulated` preset under [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) as Haswell, Skylake-SP, Cooper Lake, Ice Lake-SP, Sapphire Rapids, Arrow Lake, Granite Rapids, Nova Lake and Diamond Rapids, AMX's tiles and AVX10.2 included, and the tunings that are not the default on the models whose kernels they change: what no runner of GitHub's may have, checked to read no more than it is handed.

`sdasm_bench` times the operations of SD 1.5 at their shapes in every form (`--forms assembly,avx2,avx512,amx,cxx,upstream`) and variant (`--variant label:isa=avx512,wide=0`), and `--peak` the most FMAs the processor makes at each width.

## The code

The sources are C++26, built with Clang and warned of everything Clang warns of (`-Weverything`), but for what is not about the code -- that it would not compile as an older C or C++, where a structure has padding, and `-Wswitch-default`, which `-Wcovered-switch-default` contradicts -- and every warning stops the build (`SDASM_WERROR`, on in every preset).

Memory is reached through spans (`sources/memory.h`), never through a pointer cast to another type, made from an integer or made into one: the data of ggml's tensors, behind `void *`, is taken as the elements their type says they are, once, after checking the type, the alignment and the strides, which an operation checks before it takes a node; the work buffer ggml hands an operation is laid out into arrays of the types it needs, which begin their lifetimes as those types (`std::start_lifetime_as_array`) before its threads first meet; the store's buffers are allocated as what they hold. With `-Wunsafe-buffer-usage` there is no pointer arithmetic, no index into an array of C's and no call of the C library that reads or writes through a pointer and a length in the sources; the few places that are handed a pointer by another's code -- ggml's arrays, `argv`, the C++ kernels, which take the pointers the kernels written by hand take -- say so, each in a `#pragma clang unsafe_buffer_usage` block of its own. Every index is checked against the bounds of its span in the Debug build and in the sanitizers' (`_GLIBCXX_ASSERTIONS`); a Release build checks what it is handed once, where it takes it, and computes as fast as before.

The sanitizers' builds (`sanitize`, `tsan`) run every test, ggml's own kernels included, with nothing turned off or suppressed: AddressSanitizer with its leak checker, UndefinedBehaviorSanitizer with all its checks -- the patches have ggml call the functions of its tables of types through pointers of their own types, which `-fsanitize=function` checks -- and ThreadSanitizer. ggml's threads run on ggml's own pool rather than on OpenMP: one dependency fewer, as fast, and a pool whose synchronisation the sanitizers see.

## Measuring on GitHub's runners

The workflow `Measure` (`.github/workflows/measure.yml`), started by hand, builds once and tests and times the same build on as many runners at once as it is given -- sixty by default -- which are virtual machines on whatever processors GitHub has: AMD's Zen 3, 4 and 5, Intel's Ice Lake, Emerald Rapids and Granite Rapids have been seen, with AVX-512 and AMX among them. Each runner records its processor; the summary of the run groups the runners by processor and gives, for each, the median rate of every operation in every form and variant, how many times one level's another's, and the share of the FMA peak each reaches.

## License

GPL-3.0-or-later (`LICENSES/GPL-3.0-or-later.txt`). The patches name the holders of the code they change, which is under the MIT License; the submodules carry the licensing of their own projects. Every file names its copyright holder and license in the form of the [REUSE](https://reuse.software) specification, which `reuse lint` checks in the CI.
