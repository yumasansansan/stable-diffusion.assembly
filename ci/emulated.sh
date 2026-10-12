#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/emulated.sh <sde64> [model...]
#
# Runs the tests of the emulated preset (cmake --preset emulated; ninja -C
# build/emulated sdasm_ops_test) under Intel's Software Development Emulator,
# as processors the machine running it may not be: the kernels written by hand
# for each extension of the instruction set are chosen by what the processor
# reports (sources/cpu.cc), and SDE reports what the processor it emulates
# has, and stops at any instruction that processor would not have run. The
# preset compiles the C++ for x86-64-v3, which every model here has, so that
# whatever stops is the kernels' own; and checks every index (SDASM_CHECKS), so
# that what a kernel is handed is checked to hold all it reads, where no
# sanitizer runs: the kernels of extensions a machine lacks run only here.
#
# The models, by default: Haswell (AVX2 alone), Skylake-SP (AVX-512 F, CD, VL,
# BW, DQ), Cooper Lake (and AVX512-BF16, no AMX), Ice Lake-SP (and VNNI, VBMI,
# VBMI2, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES, VPCLMULQDQ), Sapphire Rapids (and
# BF16, FP16, AMX-BF16), Arrow Lake (AVX2 with AVX-VNNI and AVX-NE-CONVERT, no
# AVX-512), Granite Rapids (AVX10.1, AMX-FP16), Nova Lake (AVX10.2, no AMX) and
# Diamond Rapids (AVX10.2 and AMX): every form of the products of 16 bits among
# them. Each operation is tested at every level of the kernels
# the model has (tests/ops_test.cc), and the convolutions of three by three
# by F(2x2, 3x3) alone as well (f2x2), and directly, without Winograd's
# transform (direct); and the products quantized to integers (quantized),
# AVX512-VNNI's and AVX-VNNI's kernels and, on Haswell, vpmaddubsw's of seven
# bits against the C++ ones, and those of 16 bits (half). Then the tunings that
# are not the default, on the models whose kernels they change: AVX-512's
# micro-kernel of 6 by 16 (wide=0), AMX's attention (amx_attention=1),
# AVX512-BF16's products of two parts (bf16=1), and the tiles' bfloat16 where
# AMX-FP16 would be taken (amx_fp16=0). A run is failed by a test that fails,
# by an instruction the model does not have, and by a check that does not hold,
# which stops the program.
set -euo pipefail

sde=$1
shift
models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then
  models=(hsw skx cpx icx spr arl gnr nvl dmr)
fi
test=build/emulated/tests/sdasm_ops_test
# Each run: its name, the tuning it takes (- for none), the operation of
# tests/ops_test.cc, and the models it is for (all, or some, comma-separated).
runs=(
  "conv2d - conv2d all"
  "f2x2 winograd=1 conv2d all"
  "direct winograd=0 conv2d all"
  "mul_mat - mul_mat all"
  "attention - attention all"
  "gated - gated all"
  "fusion - fusion all"
  "store - store all"
  "quantized - quantized all"
  "half - half all"
  "narrow wide=0 mul_mat skx,gnr"
  "narrow wide=0 conv2d skx,gnr"
  "narrow wide=0 attention skx,gnr"
  "narrow wide=0 gated skx,gnr"
  "amx-attn amx_attention=1 attention spr,gnr,dmr"
  "bf16 bf16=1 mul_mat cpx,spr,gnr"
  "bf16 bf16=1 conv2d cpx,spr,gnr"
  "bf16 bf16=1 gated cpx"
  "amx-bf16 amx_fp16=0 mul_mat gnr,dmr"
  "amx-bf16 amx_fp16=0 conv2d gnr,dmr"
  "amx-bf16 amx_fp16=0 half gnr,dmr"
)
status=0
for model in "${models[@]}"; do
  for run in "${runs[@]}"; do
    read -r name tune op only <<<"$run"
    if [[ $only != all && ",$only," != *",$model,"* ]]; then
      continue
    fi
    [[ $tune == - ]] && tune=
    code=0
    log=$(SDASM_TUNE=$tune "$sde" "-$model" -- "$test" "$op" 4 2>&1) || code=$?
    failed=$(grep -cE '^FAIL|SDE-ERROR|does not hold' <<<"$log" || true)
    passed=$(grep -c '^ok' <<<"$log" || true)
    label=$name
    [[ $name != "$op" ]] && label="$name $op"
    printf '%-4s %-18s %3d ok %3d failed%s\n' "$model" "$label" "$passed" "$failed" "$([[ $code -ne 0 ]] && echo ", exit $code")"
    if [[ $failed -ne 0 || $code -ne 0 ]]; then
      grep -E '^FAIL|SDE-ERROR|does not hold' <<<"$log" | head -5
      status=1
    fi
  done
done
exit "$status"
