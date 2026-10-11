# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
# The developer tools: sdasm_bench times the operations the kernels take at the
# shapes of SD 1.5, in each form of the kernels (bench.cc), and on x86-64 the
# most FMAs the processor makes, for the share of it each reaches
# (peak_x86_64.S).

add_executable(sdasm_bench bench.cc)
if(SDASM_X86_64)
  target_sources(sdasm_bench PRIVATE peak_x86_64.S)
  target_compile_definitions(sdasm_bench PRIVATE SDASM_X86_64=1)
else()
  target_compile_definitions(sdasm_bench PRIVATE SDASM_X86_64=0)
endif()
sdasm_own_target(sdasm_bench)
target_link_libraries(sdasm_bench PRIVATE ggml sdasm)
target_include_directories(sdasm_bench SYSTEM PRIVATE "${SDASM_GGML_DIR}/include")
target_compile_definitions(sdasm_bench PRIVATE ${SDASM_SD_DEFINITIONS})
