# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
# The version of stable-diffusion.assembly, from the git history: 0.1.N, where
# N counts the commits of main, until a first release. The version to show adds
# the time of the commit, in UTC, and its hash, 0.1.N+YYYYMMDD.HHMM.git<hash>,
# so that a commit gives the same version however often it is built. Without git
# or the history, as in an archive of the sources or a shallow clone, the
# version is 0.1.0+unknown.
#
# Sets SDASM_VERSION, for project(), and SDASM_VERSION_DISPLAY. Included before
# project().

set(SDASM_VERSION "0.1.0")
set(SDASM_VERSION_DISPLAY "0.1.0+unknown")

find_program(SDASM_GIT git DOC "git, which gives the version and applies the patches")

# Runs git in the source tree; the output is empty if git fails.
function(sdasm_git out)
  execute_process(COMMAND "${SDASM_GIT}" -C "${CMAKE_CURRENT_SOURCE_DIR}" ${ARGN}
    OUTPUT_VARIABLE output RESULT_VARIABLE result
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT result EQUAL 0)
    set(output "")
  endif()
  set(${out} "${output}" PARENT_SCOPE)
endfunction()

if(SDASM_GIT)
  sdasm_git(SDASM_GIT_COUNT rev-list --first-parent --count HEAD)
  sdasm_git(SDASM_GIT_TIME log -1 --format=%ct HEAD)
  sdasm_git(SDASM_GIT_HASH rev-parse --short=7 HEAD)

  if(SDASM_GIT_COUNT MATCHES "^[0-9]+$" AND SDASM_GIT_TIME MATCHES "^[0-9]+$" AND
     SDASM_GIT_HASH MATCHES "^[0-9a-f]+$")
    # string(TIMESTAMP) formats the time in SOURCE_DATE_EPOCH, when there is one,
    # instead of the current time.
    if(DEFINED ENV{SOURCE_DATE_EPOCH})
      set(SDASM_SAVED_EPOCH "$ENV{SOURCE_DATE_EPOCH}")
    endif()
    set(ENV{SOURCE_DATE_EPOCH} "${SDASM_GIT_TIME}")
    string(TIMESTAMP SDASM_GIT_WHEN "%Y%m%d.%H%M" UTC)
    if(DEFINED SDASM_SAVED_EPOCH)
      set(ENV{SOURCE_DATE_EPOCH} "${SDASM_SAVED_EPOCH}")
    else()
      unset(ENV{SOURCE_DATE_EPOCH})
    endif()
    set(SDASM_VERSION "0.1.${SDASM_GIT_COUNT}")
    set(SDASM_VERSION_DISPLAY "${SDASM_VERSION}+${SDASM_GIT_WHEN}.git${SDASM_GIT_HASH}")
  endif()

  # A commit, or checking out another one, configures the build again, so that
  # the version follows. The log of HEAD changes with both.
  foreach(SDASM_GIT_FILE IN ITEMS HEAD logs/HEAD)
    sdasm_git(SDASM_GIT_PATH rev-parse --git-path "${SDASM_GIT_FILE}")
    if(SDASM_GIT_PATH)
      get_filename_component(SDASM_GIT_PATH "${SDASM_GIT_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
      if(EXISTS "${SDASM_GIT_PATH}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SDASM_GIT_PATH}")
      endif()
    endif()
  endforeach()
endif()

message(STATUS "stable-diffusion.assembly ${SDASM_VERSION_DISPLAY}")
