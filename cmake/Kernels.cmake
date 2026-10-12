# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
# The kernels (sources/): a library of their own, compiled with this project's
# warning flags and language standards, and linked into ggml's CPU backend,
# which declares the two functions it calls them by and leaves their code to
# this build (patches/ggml/0001-...). The kernels written by hand, the .S files
# of sources/x86_64/, go through Clang, the build's assembler, with the C
# preprocessor; they are machine code as written, which link-time optimisation
# has nothing to do with, and are built on x86-64 alone.
#
# They reach into ggml's CPU backend for its thread pool (its barrier and the
# counter its kernels share work out by) and for the layout of its tensors, so
# they are compiled with its private headers on the include path, as system
# headers: ggml's code is not held to this project's warnings.

set(SDASM_GGML_DIR "${PROJECT_SOURCE_DIR}/thirdparty/stable-diffusion.cpp/ggml")

add_library(sdasm STATIC
  "${PROJECT_SOURCE_DIR}/sources/attention.cc"
  "${PROJECT_SOURCE_DIR}/sources/conv2d.cc"
  "${PROJECT_SOURCE_DIR}/sources/cpu.cc"
  "${PROJECT_SOURCE_DIR}/sources/fusion.cc"
  "${PROJECT_SOURCE_DIR}/sources/gemm.cc"
  "${PROJECT_SOURCE_DIR}/sources/hooks.cc"
  "${PROJECT_SOURCE_DIR}/sources/microkernel.cc"
  "${PROJECT_SOURCE_DIR}/sources/mul_mat.cc"
  "${PROJECT_SOURCE_DIR}/sources/norm.cc"
  "${PROJECT_SOURCE_DIR}/sources/profile.cc"
  "${PROJECT_SOURCE_DIR}/sources/quant.cc"
  "${PROJECT_SOURCE_DIR}/sources/store.cc"
  "${PROJECT_SOURCE_DIR}/sources/winograd.cc")
if(SDASM_X86_64)
  target_sources(sdasm PRIVATE
    "${PROJECT_SOURCE_DIR}/sources/x86_64/activation_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/activation_avx512.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/amx.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/amx_fp16.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/amx_single.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/amx_winograd.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/bf16_avx512.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/gemm_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/gemm_avx512.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/pack_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/quant_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/quant_avx512.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/quantize_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/softmax_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/softmax_avx512.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/winograd_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/winograd4_avx2.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/winograd_avx512.S"
    "${PROJECT_SOURCE_DIR}/sources/x86_64/winograd4_avx512.S")
  target_compile_definitions(sdasm PRIVATE SDASM_X86_64=1)
else()
  target_compile_definitions(sdasm PRIVATE SDASM_X86_64=0)
endif()
sdasm_own_target(sdasm)
target_include_directories(sdasm PUBLIC "${PROJECT_SOURCE_DIR}/sources")
target_include_directories(sdasm SYSTEM PRIVATE
  "${SDASM_GGML_DIR}/include"
  "${SDASM_GGML_DIR}/src"
  "${SDASM_GGML_DIR}/src/ggml-cpu")
# The definitions stable-diffusion.cpp gives its directory, ggml's included,
# change the layout of ggml's tensors -- GGML_MAX_NAME, the length of a name,
# sets where everything after it lies -- so the kernels are compiled with them
# too, and see a tensor as ggml does.
get_directory_property(SDASM_SD_DEFINITIONS
  DIRECTORY "${PROJECT_SOURCE_DIR}/thirdparty/stable-diffusion.cpp" COMPILE_DEFINITIONS)
target_compile_definitions(sdasm PRIVATE ${SDASM_SD_DEFINITIONS})
set_property(TARGET sdasm PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(ggml-cpu PRIVATE sdasm)
target_compile_definitions(stable-diffusion PRIVATE SD_USE_SDASM)
