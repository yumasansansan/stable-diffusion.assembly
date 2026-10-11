#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/measure.sh <tools> <out> <forms> <seconds> [<only> [<variant>...]]
#
# Measures the kernels on the processor of this machine, with the programs a
# Release build made (<tools>: sdasm_bench, sdasm_ops_test and the OpenMP runtime
# they were linked with, as .github/workflows/measure.yml packs them), and
# writes what it found to <out>:
#
#   runner.txt  the processor -- its name, its extensions, its caches, its
#               threads -- and the levels of the kernels it has
#               (sdasm_bench --info);
#   tests.tsv   each operation the tests compute, ok or failed, on this
#               processor, in every form and at every level it has
#               (tests/ops_test.cc); tests.log, all they printed;
#   bench.tsv   the most FMAs the processor makes at each width
#               (sdasm_bench --peak), then the rate of each operation in each
#               form (sdasm_bench --tsv, given <forms>, <seconds> and <only>)
#               and each variant (sdasm_bench --variant: label:isa=<level>,
#               name=value,..., the kernels at that level tuned so); bench.log,
#               the tables.
#
# The tests run before the timing, and a failure fails the script after the
# timing, so that what was measured is kept either way.
set -euo pipefail

tools=$1
out=$2
forms=$3
seconds=$4
only=${5:-}
variants=("${@:6}")

mkdir -p "$out"
export LD_LIBRARY_PATH="$tools${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
threads=$(nproc)

{
  echo "model: $(grep -m 1 '^model name' /proc/cpuinfo | cut -d : -f 2- | sed 's/^ *//')"
  echo "threads: $threads"
  # The extensions the kernels are written for, or could be, as Linux lists
  # them: not all of /proc/cpuinfo's flags.
  flags=" $(grep -m 1 '^flags' /proc/cpuinfo | cut -d : -f 2) "
  listed=
  for f in avx2 fma f16c avx_vnni avx512f avx512vl avx512bw avx512dq avx512cd avx512_bf16 avx512_fp16 \
           avx512_vnni avx512vbmi avx512_vbmi2 avx512ifma avx512_vpopcntdq avx512_bitalg avx512_vp2intersect \
           gfni vaes vpclmulqdq amx_tile amx_bf16 amx_int8 amx_fp16 amx_complex avx10; do
    if [[ $flags == *" $f "* ]]; then
      listed+=" $f"
    fi
  done
  echo "flags:$listed"
  LC_ALL=C lscpu | grep -E '^(L1d|L2|L3)' | sed -E 's/ +/ /g; s/^/cache /'
  echo "kernel: $(uname -r)"
  "$tools/sdasm_bench" --info
} | tee "$out/runner.txt"

echo "== the tests, on $threads threads"
status=0
: > "$out/tests.tsv"
: > "$out/tests.log"
for op in conv2d mul_mat attention gated fusion store; do
  if "$tools/sdasm_ops_test" "$op" "$threads" >> "$out/tests.log" 2>&1; then
    printf '%s\tok\n' "$op" >> "$out/tests.tsv"
  else
    printf '%s\tfailed\n' "$op" >> "$out/tests.tsv"
    status=1
  fi
done
grep -E '^(ok|FAIL)' "$out/tests.log" || true
cat "$out/tests.tsv"

echo "== the most FMAs, and the operations, timed"
"$tools/sdasm_bench" --peak --threads "$threads" --seconds "$seconds" --tsv "$out/bench.tsv" 2>&1 | tee "$out/bench.log"
args=(--threads "$threads" --seconds "$seconds" --forms "$forms" --tsv "$out/bench.tsv")
if [ -n "$only" ]; then
  args+=(--only "$only")
fi
for v in "${variants[@]}"; do
  args+=(--variant "$v")
done
"$tools/sdasm_bench" "${args[@]}" 2>&1 | tee -a "$out/bench.log"

exit "$status"
