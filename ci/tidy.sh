#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/tidy.sh <build directory>
#
# Checks this project's sources with clang-tidy, as .clang-tidy says, by the
# compile commands of a configured build (CMAKE_EXPORT_COMPILE_COMMANDS, on in
# every preset): every finding is an error. clang-tidy is of the version of the
# compiler (ci/setup.sh), which compiled the commands.
set -euo pipefail

build=$1
if [ ! -f "$build/compile_commands.json" ]; then
  echo "error: $build has no compile_commands.json; configure it first" >&2
  exit 2
fi

compiler=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$build/CMakeCache.txt")
major=$("$compiler" --version | sed -n 's/.*clang version \([0-9]*\)\..*/\1/p' | head -1)
tidy=$(command -v "clang-tidy-$major" || command -v clang-tidy)
echo "== $("$tidy" --version | grep -i version | head -1)"

status=0
for file in sources/*.cc tests/*.cc tools/*.cc; do
  echo "== $file"
  "$tidy" --quiet -p "$build" "$file" || status=1
done
exit "$status"
