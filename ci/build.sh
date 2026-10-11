#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt). It comes from ADLplug-Next, by the same
# author, under the same license.
#
#   ci/build.sh <preset> <avx2|native> [<cmake option>...]
#
# Configures a CMake preset, with the tests and the developer tools, for the
# given instruction set (SDASM_ARCH: avx2, x86-64-v3, for a build that is
# shipped, and native, the machine's own, for one that is run where it is made)
# and any further options.
#
# The configuration checks the toolchain (cmake/LLVMToolchain.cmake), and this
# script shows what it checked, in the log and, in GitHub Actions, in the
# summary of the job. Before building, it checks the -march flags of the compile
# commands, and ThinLTO in those of Release builds, whose every compile and link
# option it lists (ci/flags.py). Then it builds, with every command shown in
# full, and shows the newest version that sd-cli asks of the libraries of the
# system.
set -euo pipefail

preset=$1
arch=$2
shift 2

args=(--preset "$preset" -DSDASM_BUILD_TOOLS=ON -DSDASM_BUILD_TESTS=ON)
case $arch in
  avx2 | native) args+=("-DSDASM_ARCH=$arch") ;;
  *) echo "error: unknown instruction set '$arch'" >&2; exit 2 ;;
esac
args+=("$@")

cmake "${args[@]}"

# Every tool that the configuration checked, with the version it reported. A
# missing record means that the check did not run, and the build does not go on.
toolchain=build/$preset/llvm-toolchain.txt
if [ ! -s "$toolchain" ]; then
  echo "error: the configuration left no record of its toolchain check ($toolchain)" >&2
  exit 1
fi
echo "== LLVM toolchain, as the configuration checked it"
if [ -n "${SDASM_LLVM_MAJOR:-}" ]; then
  echo "   required major version (SDASM_LLVM_MAJOR): $SDASM_LLVM_MAJOR"
fi
while IFS=$'\t' read -r part version program; do
  printf '   %-8s  %-26s %s\n' "$version" "$part" "$program"
done < "$toolchain"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### LLVM toolchain of $preset ($arch)"
    echo
    if [ -n "${SDASM_LLVM_MAJOR:-}" ]; then
      echo "Required major version (\`SDASM_LLVM_MAJOR\`): $SDASM_LLVM_MAJOR"
      echo
    fi
    echo "| Version | Part | Program |"
    echo "| --- | --- | --- |"
    while IFS=$'\t' read -r part version program; do
      echo "| $version | $part | \`$program\` |"
    done < "$toolchain"
    echo
  } >> "$GITHUB_STEP_SUMMARY"
fi

echo "== -march flags in the compile commands"
grep -o -E -- '-march=[a-z0-9.-]+' "build/$preset/compile_commands.json" | sort | uniq -c || echo "(none)"

# Release builds are made with ThinLTO throughout, and a compile command
# without it would mean code left out of link-time optimisation with nothing to
# say so. An assembly file (sources/x86_64/*.S) is machine code as it is
# written, which link-time optimisation has nothing to do with: CMake gives it
# no -flto.
case $preset in
  release)
    commands=$(grep '"command"' "build/$preset/compile_commands.json" | grep -v -E '\.(S|s|asm)\.o ' || true)
    total=$(grep -c . <<< "$commands" || true)
    without=$(grep -v -c -- '-flto=thin' <<< "$commands" || true)
    echo "== ThinLTO: $((total - without)) of $total compile commands"
    if [ "$total" -eq 0 ] || [ "$without" -ne 0 ]; then
      echo "error: $preset has compile commands without -flto=thin:" >&2
      grep -v -- '-flto=thin' <<< "$commands" | head -n 5 >&2 || true
      exit 1
    fi
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::group::== options of the compile and link commands (ci/flags.py)"
    else
      echo "== options of the compile and link commands (ci/flags.py)"
    fi
    python3 ci/flags.py "build/$preset"
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::endgroup::"
    fi
    ;;
esac

# Ninja shows every command in full, rather than its short description.
VERBOSE=1 cmake --build --preset "$preset"

cli=build/$preset/bin/sd-cli
echo "== $cli"
ls -l "$cli"
ldd "$cli"
# The C library and the C++ library are the system's, and sd-cli asks for a
# version of each by name: the newest version it asks for is the oldest system
# that can run it.
echo "== the newest version sd-cli asks of the system's libraries"
for library in GLIBC GLIBCXX CXXABI; do
  asked=$(llvm-objdump -T "$cli" | grep -o -E "${library}_[0-9.]+" | sort -u -V | tail -1)
  printf '   %s\n' "${asked:-(no $library)}"
done
