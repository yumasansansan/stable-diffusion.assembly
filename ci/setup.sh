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
# Sets up a GitHub Actions runner of Ubuntu to build and test
# stable-diffusion.assembly: Clang, LLD and the LLVM tools of one major version,
# from apt.llvm.org, whose signing key is checked by its fingerprint, with the
# OpenMP runtime that ggml's threads run on, clang-tidy of the same version, and
# the C++ library's headers of the GCC its runtime comes from. The tools go first
# on PATH for the later steps.
#
# To move to another version of LLVM, change llvm_major below; the
# configuration of every build checks that its toolchain is of that version
# (cmake/LLVMToolchain.cmake).
set -euo pipefail

llvm_major=23

# The packages of apt.llvm.org, signed with this key.
apt_key_fingerprint=6084F3CF814B57C1CF12EFD515CF4D18AF4F7421

if [ "${RUNNER_OS:-}" != Linux ] || [ ! -f /etc/os-release ]; then
  echo "error: stable-diffusion.assembly builds on Linux for now, and this runner is '${RUNNER_OS:-}'" >&2
  exit 1
fi

key=$RUNNER_TEMP/apt.llvm.org.asc
keyring=/usr/share/keyrings/apt.llvm.org.gpg
curl --fail --location --silent --show-error --retry 3 --retry-all-errors --output "$key" \
  https://apt.llvm.org/llvm-snapshot.gpg.key
fingerprint=$(gpg --show-keys --with-colons "$key" | awk -F : '$1 == "fpr" && !found { print $10; found = 1 }')
if [ "$fingerprint" != "$apt_key_fingerprint" ]; then
  echo "error: the apt.llvm.org key has fingerprint $fingerprint, expected $apt_key_fingerprint" >&2
  exit 1
fi
gpg --dearmor < "$key" | sudo tee "$keyring" > /dev/null

codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
echo "deb [signed-by=$keyring] https://apt.llvm.org/$codename/ llvm-toolchain-$codename-$llvm_major main" |
  sudo tee /etc/apt/sources.list.d/apt.llvm.org.list > /dev/null

# **The C++ library's headers, of the GCC its runtime comes from.** Clang
# compiles against the headers of the newest GCC it finds them for, and an image
# may carry the headers of an older GCC than the one its libstdc++6 was built
# from: a program then sees only what the older headers declare, while it runs on
# the newer library. The development package of the runtime's own GCC closes the
# gap, whichever GCC that is.
runtime=$(dpkg-query --show --showformat '${source:Package}' libstdc++6)
case "$runtime" in
  gcc-[0-9]*) ;;
  *) echo "error: libstdc++6 was built from '$runtime', not from a gcc-<N> source package" >&2; exit 1 ;;
esac

sudo apt-get update -qq
# libomp is the OpenMP runtime of this LLVM, which ggml's thread pool runs on;
# clang-tidy is of the same version as the compiler, so that it parses what the
# compiler compiles.
sudo apt-get install -y -qq --no-install-recommends \
  "clang-$llvm_major" "lld-$llvm_major" "llvm-$llvm_major" "libomp-$llvm_major-dev" \
  "clang-tidy-$llvm_major" "clang-tools-$llvm_major" \
  "libstdc++-${runtime#gcc-}-dev"
llvm_bin=/usr/lib/llvm-$llvm_major/bin

version=$("$llvm_bin/clang" --version)
version=${version%%$'\n'*}
echo "$version"
case "$version" in
  *"clang version $llvm_major."*) ;;
  *) echo "error: expected Clang $llvm_major" >&2; exit 1 ;;
esac
cmake --version
ninja --version
echo "== the processor of this runner"
grep -m 1 '^model name' /proc/cpuinfo || true
nproc

echo "$llvm_bin" >> "$GITHUB_PATH"
echo "SDASM_LLVM_MAJOR=$llvm_major" >> "$GITHUB_ENV"
