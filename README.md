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

Processors other than this one are measured on GitHub's runners (see [Measuring on GitHub's runners](#measuring-on-githubs-runners)).

## Precision

The kernels never compute less precisely than stable-diffusion.cpp does. ggml rounds the image to FP16 before its products whenever the weights are FP16 -- the linear layers and the convolutions of SD 1.5 -- and sums the products of the rounded numbers; the kernels for AVX2 and AVX-512 keep the image in single precision, and their outputs are one or two orders of magnitude closer to the exact ones. The tests (`tests/ops_test.cc`) measure every form and every level against the same computed in double precision, the error divided by the sum of the magnitudes of the products summed:

| Operation | AVX2 / AVX-512 / C++ | AMX | ggml (upstream) |
| --- | ---: | ---: | ---: |
| Linear layers (FP16 weights) | 0.4 to 2.4 × 10⁻⁷ | 0.5 to 1.8 × 10⁻⁶ (AMX-BF16), 3.5 to 7.0 × 10⁻⁸ (AMX-FP16) | 2.9 to 19 × 10⁻⁵ |
| Convolutions, F(2x2, 3x3) and direct (FP16 weights) | 0.7 to 2.1 × 10⁻⁷ | 1.2 to 6.8 × 10⁻⁶; direct on AMX-FP16, 0.6 to 0.8 × 10⁻⁷ | 3.9 to 13 × 10⁻⁵ |
| Convolutions, F(4x4, 3x3) (FP16 weights, 16 by 16 and more) | 5 to 11 × 10⁻⁷ | (F(2x2, 3x3) on AMX) | 4.1 to 13 × 10⁻⁵ |
| Attention (single precision) | 1.4 to 19 × 10⁻⁸, as close as ggml's graph, or closer | 1.3 to 7.1 × 10⁻⁸ | 2.0 to 16 × 10⁻⁸ (its graph) |

**On AMX's tiles**, which multiply bfloat16 (8 bits of significand, against FP16's 11), each number is split in two parts, x = x1 + x2 and w = w1 + w2, each rounded to the nearest, and a product taken as x1 w1 + x1 w2 + x2 w1: within 2⁻¹⁵ of the exact one, sixteen times as close as ggml's rounding to FP16 (2⁻¹¹). Where the processor has AMX-FP16 (Granite Rapids), the FP16 weights go to the tiles as they are and the other operand in two parts of FP16, x1 being ggml's own rounding and x2 what it leaves: two products for each one, w x1 + w x2, each exact, within 2⁻²² of the exact product and never further from it than ggml's -- as close as single precision, and a third fewer products. Winograd's products, of two operands of single precision, keep the two parts of bfloat16. Attention, which ggml computes in single precision, is computed in single precision on the tiles too: three parts of bfloat16 hold a number of single precision exactly, and a product is taken as six products, x1 y1 + x1 y2 + x2 y1 + x1 y3 + x2 y2 + x3 y1, the five small ones summed first, while the sums are small, and x1 y1 after; so summed, the outputs are as close to the exact ones as ggml's or closer, where summed block by block they were up to four times further. The GELU of GEGLU is computed in single precision, where ggml reads it from a table of FP16.

## The levels of the instruction set

| Level | Extensions | What the kernels written by hand use it for |
| --- | --- | --- |
| `avx2` | AVX2, FMA, F16C (x86-64-v3) | the micro-kernels (6 by 16), the softmax's exponentials, the packing, the transforms of F(2x2, 3x3) and F(4x4, 3x3), the gated GELU |
| `avx512` | AVX-512 F, VL, BW, DQ (AVX10.1) | the same, sixteen lanes wide; masked loads at the edges of an image; on Intel's processors a micro-kernel of 6 by 32, whose loads keep up with two FMAs of sixteen a cycle |
| `amx` | AMX-TILE, AMX-BF16, AVX512-BF16; AMX-FP16 where there is | the products of the linear layers and of the convolutions (Winograd's included) on the tiles, in bfloat16 split in two parts, or with AMX-FP16 the weights as they are and the other operand in two parts of FP16; attention's in three parts of bfloat16; Winograd's transform of the weights for them by AVX-512; all else as `avx512` |

The level in effect is the richest the processor has; `SDASM_ISA=avx2|avx512|amx` holds it to a level below, and the tests compute every operation at every level the processor has. A build for x86-64-v3 (`SDASM_ARCH=avx2`, the one that is shipped) runs every level; the C++ is then compiled for AVX2.

Of AVX-512's other extensions, none speeds up these operations at that precision. VNNI (and AVX-VNNI) multiplies integers, which the numbers would have to be quantized to. AVX512-BF16's `vdpbf16ps` makes 32 products to an FMA's 16, but the two parts make three products of each one, half again as many instructions as the FMAs of single precision, which llvm-mca's models of Cooper Lake, Sapphire Rapids, Zen 4 and Zen 5 run no faster; it is kept as a variant to measure on real processors (`SDASM_TUNE=bf16=1`, on the tiles' packing). AVX512-FP16 sums in FP16 and would round every partial sum to 11 bits. IFMA, VBMI, VBMI2, BITALG, VPOPCNTDQ, VP2INTERSECT, GFNI, VAES and VPCLMULQDQ have nothing to do with single precision. AMX's tiles are another matter: they make sixteen times the products of AVX-512's FMAs, and with three products for each one, five times as many.

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

The presets (`cmake --list-presets`): `release` (-O3, ThinLTO, for x86-64-v3 by default), `debug`, `relwithdebinfo`, `sanitize` and `tsan` (address and undefined, thread and undefined), and `emulated`, the tests for x86-64-v3 for Intel SDE. `-DSDASM_ARCH=native` builds the C++ for the machine that builds it.

The program is stable-diffusion.cpp's, `build/release/bin/sd-cli`, with all its options:

```bash
build/release/bin/sd-cli -m v1-5-pruned-emaonly.safetensors -p "a lovely cat" --steps 20 -t 12
```

### Environment

| Variable | |
| --- | --- |
| `SDASM_KERNELS=assembly\|cxx\|upstream` | the form of the kernels (assembly by default) |
| `SDASM_ISA=avx2\|avx512\|amx` | the richest level the kernels written by hand take |
| `SDASM_TUNE=name=value,...` | the sizes of the blocks and the choices of the kernels (`Tuning` in `sources/kernels.h`): `wide=0\|1` the micro-kernel of 6 by 32 on AVX-512, `bf16=1` the products of AVX512-BF16, `amx_fp16=0` AMX-BF16's form on a processor with AMX-FP16, `amx_attention=0` attention on the vector units |
| `SDASM_PROFILE=1` | each operation's time, by shape and by phase, at the end |
| `SDASM_STORE_MB=n` | keep up to n MiB of packed weights from one step to the next |

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

`sdasm_ops_test <operation> [threads]` computes each operation in every form and at every level against double precision, and prints how close each came; `ci/test.sh` runs them all. `ci/emulated.sh <sde64>` runs the tests of the `emulated` preset under [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) as Haswell, Skylake-SP, Ice Lake-SP, Sapphire Rapids, Arrow Lake and Granite Rapids, AMX's tiles included.

`sdasm_bench` times the operations of SD 1.5 at their shapes in every form (`--forms assembly,avx2,avx512,amx,cxx,upstream`) and variant (`--variant label:isa=avx512,wide=0`), and `--peak` the most FMAs the processor makes at each width.

## Measuring on GitHub's runners

The workflow `Measure` (`.github/workflows/measure.yml`), started by hand, builds once and tests and times the same build on as many runners at once as it is given -- sixty by default -- which are virtual machines on whatever processors GitHub has: AMD's Zen 3, 4 and 5, Intel's Ice Lake, Emerald Rapids and Granite Rapids have been seen, with AVX-512 and AMX among them. Each runner records its processor; the summary of the run groups the runners by processor and gives, for each, the median rate of every operation in every form and variant, how many times one level's another's, and the share of the FMA peak each reaches.

## License

GPL-3.0-or-later (`LICENSES/GPL-3.0-or-later.txt`). The patches name the holders of the code they change, which is under the MIT License; the submodules carry the licensing of their own projects. Every file names its copyright holder and license in the form of the [REUSE](https://reuse.software) specification, which `reuse lint` checks in the CI.
