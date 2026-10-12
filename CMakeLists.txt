# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
# stable-diffusion.assembly is stable-diffusion.cpp (thirdparty/, a submodule,
# with the ggml it carries) with the heavy operations of ggml's CPU backend
# replaced by kernels of its own (sources/): matrix products, convolutions and
# attention, written in C++ and, where that is measured to be faster, by hand
# in assembly. The patches of patches/ lead the two projects to those kernels;
# what the build makes is stable-diffusion.cpp's library and its command-line
# program, sd-cli, with them in.

cmake_minimum_required(VERSION 3.29)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# The version comes from the git history (cmake/Version.cmake).
include(Version)
project(stable-diffusion.assembly VERSION "${SDASM_VERSION}" LANGUAGES C CXX ASM)
set(PROJECT_HOMEPAGE "https://github.com/yumasansansan/stable-diffusion.assembly")

# --------------------------------------------------------------------------
# The presets are how this project is configured
#
# CMakePresets.json chooses the compilers, the linker, the generator and the
# settings that stable-diffusion.assembly is built and tested with. A
# configuration put together by hand differs from them without saying so, which
# is how a build ends up with flags the project never chose; the checks below
# would catch a missing compiler or linker, but not the rest.
set(SDASM_PRESET "" CACHE STRING
  "The preset this build was configured with; CMakePresets.json sets it")
if(NOT SDASM_PRESET)
  message(FATAL_ERROR
    "Configure through a preset: cmake --preset release, and cmake --list-presets names them "
    "all. The presets choose the compilers, the linker, the generator and the settings this "
    "project is built and tested with, which a configuration by hand differs from without saying "
    "so. A build that has to differ can inherit a preset of its own in CMakeUserPresets.json, or "
    "say -DSDASM_PRESET=<a name for it> and own the settings it is then left with.")
endif()
message(STATUS "Preset: ${SDASM_PRESET}")

# The warning policy, the language standards, the assembly and LTO below are
# written for Clang; GCC and MSVC are not supported compilers.
if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang" OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "stable-diffusion.assembly is built with Clang, not ${CMAKE_CXX_COMPILER_ID}. Configure "
    "with CMAKE_C_COMPILER=clang and CMAKE_CXX_COMPILER=clang++, as the presets do.")
endif()

# LLD links everything: CMake gives CMAKE_LINKER_TYPE to every target and to
# try_compile().
if(NOT CMAKE_LINKER_TYPE STREQUAL "LLD")
  message(FATAL_ERROR
    "stable-diffusion.assembly is linked with LLD, but CMAKE_LINKER_TYPE is "
    "'${CMAKE_LINKER_TYPE}'. Configure with -DCMAKE_LINKER_TYPE=LLD, as the presets do.")
endif()

# The kernels written by hand are for x86-64; on any other processor the C++
# kernels, which the compiler vectorizes for it, are all there is.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(SDASM_X86_64 ON)
else()
  set(SDASM_X86_64 OFF)
  message(STATUS "stable-diffusion.assembly: ${CMAKE_SYSTEM_PROCESSOR}, the C++ kernels alone")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "stable-diffusion.assembly builds for Linux for now: the kernels written by hand follow "
    "the System V calling convention alone, and this is ${CMAKE_SYSTEM_NAME}.")
endif()

# The toolchain has to be LLVM's, of one version (cmake/LLVMToolchain.cmake).
include(LLVMToolchain)

# --------------------------------------------------------------------------
# Options

option(SDASM_WERROR "Treat warnings in stable-diffusion.assembly's own sources as errors" OFF)
option(SDASM_BUILD_TESTS "Build the tests and register them with CTest" OFF)
option(SDASM_BUILD_TOOLS "Build the developer tools (tools/: the benchmark of the kernels)" OFF)
if(SDASM_X86_64)
  set(SDASM_ARCH_DEFAULT avx2)
else()
  set(SDASM_ARCH_DEFAULT native)
endif()
set(SDASM_ARCH "${SDASM_ARCH_DEFAULT}" CACHE STRING
  "Instruction set: avx2 (x86-64-v3, the default on x86-64) or native (the machine's own)")
set_property(CACHE SDASM_ARCH PROPERTY STRINGS avx2 native)
set(SDASM_SANITIZERS "" CACHE STRING "Sanitizers to build with, comma-separated (address, undefined, thread)")

# Applied per target rather than globally: these variables are inherited by
# add_subdirectory(), and stable-diffusion.cpp and ggml build their own sources
# as C11 and C++17.
function(sdasm_set_language_standard target)
  set_target_properties("${target}" PROPERTIES
    C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    CXX_STANDARD 26 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
endfunction()

# No file of this project is a C++ module, and none imports one. Left alone,
# CMake scans every C++26 source for modules: work for a build to do and nothing
# for it to find.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# --------------------------------------------------------------------------
# Link-time optimisation
#
# Release builds are made with link-time optimisation, for every target: it is
# set before stable-diffusion.cpp is added, so it and ggml get it too, and the
# kernels can be inlined into the code that calls them across library
# boundaries. With Clang, CMake's IPO support means ThinLTO (-flto=thin). With
# ThinLTO, LLD runs the optimisation across modules and the code generation when
# it links, at levels of its own, which are 2 unless it is told otherwise; they
# are given to it as 3. An assembly file is machine code as it is written, which
# link-time optimisation has nothing to do with: CMake gives it no -flto.

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
add_link_options("$<$<CONFIG:Release>:LINKER:--lto-O3,--lto-CGO3>")

# --------------------------------------------------------------------------
# Instruction set
#
# Builds come in two kinds: avx2 (x86-64-v3: AVX2, BMI1 and BMI2, FMA, F16C,
# LZCNT and MOVBE -- Haswell, Zen and later), which is the least this project
# builds for and the default, and native (whatever the machine building it has,
# for a build that stays on it, which has to be x86-64-v3 at least). Like LTO,
# this is set before stable-diffusion.cpp is added, so that it and ggml are
# compiled for it as well; ggml's own choice of instructions (GGML_NATIVE and
# the GGML_<extension> options) is set to the same below.
#
# The C++ kernels are compiled for the instruction set chosen, which the compiler
# vectorizes them for. The kernels written by hand need no more than avx2 has;
# those for the extensions beyond it are chosen, kernel by kernel, by what the
# processor running them has (sources/cpu.cc).
#
# On a processor other than x86-64 there is only native, the machine's own.

if(NOT SDASM_ARCH MATCHES "^(avx2|native)$")
  message(FATAL_ERROR "SDASM_ARCH=${SDASM_ARCH} is not an instruction set of this project: avx2 or native.")
endif()
if(SDASM_ARCH STREQUAL "avx2" AND NOT SDASM_X86_64)
  message(FATAL_ERROR "SDASM_ARCH=avx2 is an instruction set of x86-64, and this is ${CMAKE_SYSTEM_PROCESSOR}.")
endif()
if(SDASM_ARCH STREQUAL "avx2")
  set(SDASM_ARCH_FLAGS -march=x86-64-v3)
elseif(NOT SDASM_X86_64)
  set(SDASM_ARCH_FLAGS -march=native)
else()
  set(SDASM_ARCH_FLAGS -march=native)
  set(SDASM_NATIVE_PROBE "${CMAKE_BINARY_DIR}/CMakeFiles/SDASM_native_probe.c")
  file(WRITE "${SDASM_NATIVE_PROBE}" "int SDASM_native_probe;\n")
  execute_process(
    COMMAND "${CMAKE_C_COMPILER}" -march=native -dM -E "${SDASM_NATIVE_PROBE}"
    OUTPUT_VARIABLE SDASM_NATIVE_MACROS
    ERROR_VARIABLE SDASM_NATIVE_ERRORS
    RESULT_VARIABLE SDASM_NATIVE_STATUS)
  if(NOT SDASM_NATIVE_STATUS EQUAL 0)
    message(FATAL_ERROR "${CMAKE_C_COMPILER} could not say what -march=native is on this machine:\n${SDASM_NATIVE_ERRORS}")
  endif()
  # x86-64-v3 at the least, as avx2 is: a machine without all of it is one this
  # project does not build for.
  foreach(SDASM_FEATURE IN ITEMS AVX2 FMA BMI BMI2 F16C LZCNT MOVBE)
    if(NOT SDASM_NATIVE_MACROS MATCHES "#define __${SDASM_FEATURE}__ ")
      message(FATAL_ERROR
        "SDASM_ARCH=native: this machine has no ${SDASM_FEATURE}, and x86-64-v3 is the least "
        "stable-diffusion.assembly builds for.")
    endif()
  endforeach()
endif()
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${SDASM_ARCH_FLAGS}>")
message(STATUS "Instruction set: ${SDASM_ARCH} (${SDASM_ARCH_FLAGS})")

# --------------------------------------------------------------------------
# Optimisation flags
#
# Release builds are optimised as far as the meaning of the code allows. Besides
# -O3 and LTO at level 3, every target, stable-diffusion.cpp and ggml included,
# is compiled with these:
# - -fno-math-errno: the math functions are not expected to set errno, so that
#   the compiler can put instructions in their place and vectorise loops that
#   call them. What they compute stays IEEE's.
# - -fomit-frame-pointer: the frame pointer register is left to the code.
# - -fno-plt: calls into shared libraries go through the GOT, without the PLT's
#   stubs.
# - -fvisibility-inlines-hidden-static-local-var: the static variables of hidden
#   inline member functions are hidden too, and reached without the GOT.
# - -mtls-dialect=gnu2: thread-local variables of a shared object are reached
#   through TLS descriptors, without a call to __tls_get_addr. The link gives it
#   to the LTO backend as well.
# LLD is given optimisations of its own: -O3 (branches to branches go straight
# to their targets, and strings share their tails), --icf=safe (identical
# functions whose addresses are not taken are folded) and --Bsymbolic-functions
# (a shared object calls its own functions directly). Nothing here trades the
# precision of floating-point results: no -ffast-math or any of its parts, and
# contraction stays as the compiler has it.

add_compile_options(
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-fno-math-errno>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-fomit-frame-pointer>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-fno-plt>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-fvisibility-inlines-hidden-static-local-var>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-mtls-dialect=gnu2>")
add_link_options(
  "$<$<CONFIG:Release>:LINKER:-O3,--icf=safe,--Bsymbolic-functions>"
  "$<$<CONFIG:Release>:-mtls-dialect=gnu2>")

# --------------------------------------------------------------------------
# Sanitizers
#
# SDASM_SANITIZERS names the sanitizers of a build, comma-separated (address,
# undefined, thread), and gives them to every target: this project's own code,
# stable-diffusion.cpp and ggml. Undefined behaviour stops the program
# (-fno-sanitize-recover=all) instead of being reported and passed over, and the
# frame pointer stays, so that the reports name the callers. The kernels written
# by hand are not instrumented -- the sanitizers instrument what Clang compiles,
# and an assembly file is not compiled -- but every buffer they read and write is
# one that instrumented code allocated and handed them.
#
# thread finds what the others cannot see at all: two threads that reach the
# same memory with nothing to order the one against the other. ggml's CPU
# backend runs every operation on all its threads at once, and the kernels share
# out the work among them, so these are mistakes this project can make wherever
# they share anything. It cannot be built together with address, and has a
# preset of its own.
#
# Sanitizers are not for Release builds, whose optimisations and link-time
# optimisation would stand in the way: the sanitize and tsan presets configure
# RelWithDebInfo.
if(SDASM_SANITIZERS)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
      "SDASM_SANITIZERS is not for Release builds, which are optimised with LTO. Configure "
      "RelWithDebInfo, as the sanitize and tsan presets do.")
  endif()
  string(REPLACE "," ";" SDASM_SANITIZER_LIST "${SDASM_SANITIZERS}")
  foreach(sanitizer IN LISTS SDASM_SANITIZER_LIST)
    if(NOT sanitizer MATCHES "^(address|undefined|thread)$")
      message(FATAL_ERROR
        "SDASM_SANITIZERS names '${sanitizer}', which is not one of address, undefined, thread.")
    endif()
  endforeach()
  if("address" IN_LIST SDASM_SANITIZER_LIST AND "thread" IN_LIST SDASM_SANITIZER_LIST)
    message(FATAL_ERROR
      "SDASM_SANITIZERS names both address and thread, which cannot be built together.")
  endif()
  message(STATUS "Sanitizers: ${SDASM_SANITIZERS}")
  # The C++ library's checks, too: of every index into its containers and its
  # spans -- this project's own (memory.h) among them -- of every precondition
  # it can check cheaply.
  add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_ASSERTIONS>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=${SDASM_SANITIZERS}>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-sanitize-recover=all>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>")
  add_link_options("-fsanitize=${SDASM_SANITIZERS}")
  set(SDASM_SANITIZER_TEST_ENVIRONMENT "UBSAN_OPTIONS=print_stacktrace=1")
  if("thread" IN_LIST SDASM_SANITIZER_LIST)
    list(APPEND SDASM_SANITIZER_TEST_ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1")
  endif()
endif()

# --------------------------------------------------------------------------
# Warnings, for this project's own sources
#
# Every warning Clang has (-Weverything), new ones as Clang adds them, but for
# those that are not about the code at all:
#
#  - that it would not compile as an older C or C++ -- the standards are C23
#    and C++26 (-Wc++98-compat and the -Wpre-c++NN-compat, -Wpre-cNN-compat);
#  - where a structure has padding (-Wpadded), which says where the compiler
#    inserts it, not that anything is wrong;
#  - -Wswitch-default, which wants a default label in every switch: a switch
#    over an enumeration of this project's names every value and has none, so
#    that -Wswitch tells of a value added and not handled, and
#    -Wcovered-switch-default, which stays, holds it to that.
#
# -Wunsafe-buffer-usage among the rest: no pointer arithmetic, no index into an
# array of C's, no call of a function of the C library that writes or reads
# through a pointer and a length, in the sources. Memory is reached through
# spans (memory.h), and the few places that are handed a pointer by another's
# code -- ggml's, the C library's, the C++ kernels', which take the pointers of
# the kernels written by hand -- say so, each in a block of its own
# (#pragma clang unsafe_buffer_usage).
#
# stable-diffusion.cpp and ggml keep their own warnings; the headers of theirs
# that this project includes are system headers to it.
set(SDASM_WARNING_FLAGS -Weverything
  -Wno-c++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -Wno-pre-c++17-compat-pedantic
  -Wno-pre-c++20-compat-pedantic -Wno-pre-c++23-compat-pedantic -Wno-pre-c++26-compat-pedantic
  -Wno-pre-c11-compat -Wno-pre-c23-compat -Wno-pre-c2y-compat
  -Wno-padded -Wno-switch-default)
if(SDASM_WERROR)
  list(APPEND SDASM_WARNING_FLAGS -Werror)
else()
  list(APPEND SDASM_WARNING_FLAGS -Werror=return-type)
endif()

# Gives a target of this project its language standards and warning flags.
function(sdasm_own_target target)
  sdasm_set_language_standard("${target}")
  target_compile_options("${target}" PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:${SDASM_WARNING_FLAGS}>")
endfunction()

# --------------------------------------------------------------------------
# stable-diffusion.cpp and ggml
#
# The patches go in first (cmake/Patches.cmake). stable-diffusion.cpp is built
# with its command-line program and without the web frontend of its server,
# which needs pnpm, for the CPU alone; ggml's choice of instructions follows
# SDASM_ARCH. ggml's own fast paths for matrix products of other libraries
# (llamafile's sgemm, BLAS) stay off, as stable-diffusion.cpp has them: the
# kernels of this project take the products in their place.
#
# ggml's threads run on ggml's own pool, not on OpenMP, which it would take by
# default where the compiler has it: one library fewer to depend on, for
# nothing -- an image of SD 1.5 samples and decodes as fast on either (on the
# i7-1360P, 31.09 s against 30.88 s for eight steps and 3.37 s against 3.48 s
# for the decoder, the mean of six runs each, the same image) -- and a pool
# whose synchronisation the sanitizers see, as they cannot see the OpenMP
# runtime's, which they do not instrument.
include(Patches)

set(SD_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(SD_SERVER_BUILD_FRONTEND OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
if(SDASM_ARCH STREQUAL "native")
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(SDASM_GGML_EXTENSION IN ITEMS SSE42 AVX AVX2 BMI2 FMA F16C)
    set(GGML_${SDASM_GGML_EXTENSION} ON CACHE BOOL "" FORCE)
  endforeach()
  foreach(SDASM_GGML_EXTENSION IN ITEMS AVX_VNNI AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16
                                        AMX_TILE AMX_INT8 AMX_BF16)
    set(GGML_${SDASM_GGML_EXTENSION} OFF CACHE BOOL "" FORCE)
  endforeach()
endif()

add_subdirectory(thirdparty/stable-diffusion.cpp SYSTEM)

# --------------------------------------------------------------------------
# The kernels (cmake/Kernels.cmake)
include(Kernels)

if(SDASM_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(SDASM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
