# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
# The tests. ops.* compute each operation the kernels take, through ggml's CPU
# backend, in every form of the kernels, and measure each against the same
# computed in double precision (ops_test.cc); ops.fusion checks the chains
# computed with an operation against the same computed apart, and ops.store the
# weights kept between graphs against weights set anew. Each runs on as many
# threads as the machine has, and on one, three and seven, so that the work
# shared out among threads is checked to give the same as the work done by one,
# whatever the number.

add_executable(sdasm_ops_test ops_test.cc)
sdasm_own_target(sdasm_ops_test)
target_link_libraries(sdasm_ops_test PRIVATE ggml sdasm)
target_include_directories(sdasm_ops_test SYSTEM PRIVATE
  "${SDASM_GGML_DIR}/include")
target_compile_definitions(sdasm_ops_test PRIVATE ${SDASM_SD_DEFINITIONS})

foreach(SDASM_TEST_OP IN ITEMS conv2d mul_mat attention gated fusion store)
  add_test(NAME ops.${SDASM_TEST_OP} COMMAND sdasm_ops_test ${SDASM_TEST_OP})
  # The operations with the weights kept between graphs as well (store.h): the
  # first computation of each fills the store, and those after read from it.
  add_test(NAME ops.${SDASM_TEST_OP}.store COMMAND sdasm_ops_test ${SDASM_TEST_OP})
  set_property(TEST ops.${SDASM_TEST_OP}.store PROPERTY ENVIRONMENT "SDASM_STORE_MB=4096")
  foreach(SDASM_TEST_THREADS IN ITEMS 1 3 7)
    add_test(NAME ops.${SDASM_TEST_OP}.threads_${SDASM_TEST_THREADS}
      COMMAND sdasm_ops_test ${SDASM_TEST_OP} ${SDASM_TEST_THREADS})
  endforeach()
endforeach()

# A test that cannot fail says nothing: the store's, with the store told never to
# forget, has to find the old weights still packed and fail.
add_test(NAME ops.store.canary COMMAND sdasm_ops_test store)
set_tests_properties(ops.store.canary PROPERTIES ENVIRONMENT "SDASM_TUNE=store_forget=0" WILL_FAIL TRUE)

# The direct convolution of the shapes Winograd's transforms take as well, and
# F(2x2, 3x3) of those F(4x4, 3x3) takes.
add_test(NAME ops.conv2d.direct COMMAND sdasm_ops_test conv2d)
set_property(TEST ops.conv2d.direct PROPERTY ENVIRONMENT "SDASM_TUNE=winograd=0")
add_test(NAME ops.conv2d.f2x2 COMMAND sdasm_ops_test conv2d)
set_property(TEST ops.conv2d.f2x2 PROPERTY ENVIRONMENT "SDASM_TUNE=winograd=1")

get_property(SDASM_TESTS DIRECTORY PROPERTY TESTS)
if(SDASM_SANITIZER_TEST_ENVIRONMENT)
  set_property(TEST ${SDASM_TESTS} APPEND PROPERTY ENVIRONMENT ${SDASM_SANITIZER_TEST_ENVIRONMENT})
endif()
