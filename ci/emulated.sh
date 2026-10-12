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
# whatever stops is the kernels' own.
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
# bits against the C++ ones, and those of 16 bits (half).
set -euo pipefail

sde=$1
shift
models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then
  models=(hsw skx cpx icx spr arl gnr nvl dmr)
fi
test=build/emulated/tests/sdasm_ops_test
status=0
for model in "${models[@]}"; do
  for op in conv2d f2x2 direct mul_mat attention gated fusion store quantized half; do
    if [[ $op == direct ]]; then
      log=$(SDASM_TUNE=winograd=0 "$sde" "-$model" -- "$test" conv2d 4 2>&1) || status=1
    elif [[ $op == f2x2 ]]; then
      log=$(SDASM_TUNE=winograd=1 "$sde" "-$model" -- "$test" conv2d 4 2>&1) || status=1
    else
      log=$("$sde" "-$model" -- "$test" "$op" 4 2>&1) || status=1
    fi
    failed=$(grep -cE '^FAIL|SDE-ERROR' <<<"$log" || true)
    passed=$(grep -c '^ok' <<<"$log" || true)
    printf '%-4s %-10s %3d ok %3d failed\n' "$model" "$op" "$passed" "$failed"
    if [[ $failed -ne 0 ]]; then
      grep -E '^FAIL|SDE-ERROR' <<<"$log" | head -5
      status=1
    fi
  done
done
exit "$status"
