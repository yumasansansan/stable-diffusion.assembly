#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
#   ci/test.sh <preset>
#
# Runs the tests of a preset that ci/build.sh built: each operation the kernels
# take, computed in every form of the kernels through ggml's CPU backend and
# measured against the same in double precision (tests/ops_test.cc), on as many
# threads as the runner has and on one, three and seven. The precision each form
# reached is printed by the tests whose output CTest shows, and so they are run
# once more with all of it, for the log.
set -euo pipefail

preset=$1
status=0

ctest --preset "$preset" || status=$?

echo "== the precision of each form of the kernels, on every thread of this runner"
for op in conv2d mul_mat attention; do
  "build/$preset/tests/sdasm_ops_test" "$op" || status=1
done

exit "$status"
