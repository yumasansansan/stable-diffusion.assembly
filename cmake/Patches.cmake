# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of stable-diffusion.assembly. The SPDX lines name its
# copyright holder and its license, in the machine-readable form of the REUSE
# specification: the GNU General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt). It comes from ADLplug-Next, by the same
# author, under the same license.
#
# The patches of patches/, applied to stable-diffusion.cpp and to the ggml it
# carries as a submodule of its own when CMake configures, so that every build
# has them. They are how this project's kernels (sources/) reach the code that
# runs them: ggml's CPU backend hands an operation to them before it computes
# it itself, and stable-diffusion.cpp builds its graphs out of the operations
# those kernels take. They stay for as long as the kernels are this project's.
#
# Configuring twice is not an error: a patch that is already in place is left
# alone. Editing one makes CMake configure again.

find_program(SDASM_GIT git DOC "git, which applies the patches to the submodules")
if(NOT SDASM_GIT)
  message(FATAL_ERROR
    "git is needed to patch the submodules, and was not found. "
    "It is needed for the submodules themselves as well.")
endif()

# Applies the series of patches, in the order given, to the working tree of
# <directory>. Later patches of a series may change what earlier ones wrote, so
# a series is applied, and taken off, one patch at a time, in order; and what is
# in the tree is not guessed from the patches but recorded: a copy of the series
# applied is kept in the submodule's own git directory (sdasm-patches/), which no
# checkout touches. A configuration finds the series there and does nothing when
# it is the series of patches/; when the series has changed, it takes the one
# recorded off, last patch first, and applies the new one. A tree without
# changes of its own (its submodules are not its own) is taken to have no
# patch in it, whatever was recorded.
function(sdasm_patch_series directory)
  set(tree "${PROJECT_SOURCE_DIR}/${directory}")
  if(NOT EXISTS "${tree}/.git")
    message(FATAL_ERROR
      "${directory} is not a checked-out submodule; the submodules are part of "
      "the sources: git submodule update --init --recursive")
  endif()

  # A submodule of a repository that another user checked out is a working tree
  # that git refuses to touch unless it is told the ownership is expected; that
  # is what a CI container running as root hands it. The setting lasts for each
  # command.
  set(git "${SDASM_GIT}" -c "safe.directory=*")

  execute_process(COMMAND ${git} rev-parse --absolute-git-dir
    WORKING_DIRECTORY "${tree}" OUTPUT_VARIABLE git_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "git does not find the repository of ${directory}.")
  endif()
  set(record "${git_dir}/sdasm-patches")

  # The series wanted, and the one recorded.
  set(wanted "")
  foreach(patch IN LISTS ARGN)
    set(file "${PROJECT_SOURCE_DIR}/${patch}")
    if(NOT EXISTS "${file}")
      message(FATAL_ERROR "The patch ${patch} is missing.")
    endif()
    set_property(DIRECTORY "${PROJECT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${file}")
    file(SHA256 "${file}" hash)
    get_filename_component(name "${patch}" NAME)
    list(APPEND wanted "${name}:${hash}")
  endforeach()
  set(recorded "")
  if(EXISTS "${record}/series")
    file(STRINGS "${record}/series" recorded)
  endif()

  execute_process(COMMAND ${git} diff --quiet --ignore-submodules HEAD
    WORKING_DIRECTORY "${tree}" RESULT_VARIABLE changed)
  if(changed EQUAL 0)
    set(recorded "")
  elseif(recorded STREQUAL wanted)
    foreach(patch IN LISTS ARGN)
      message(STATUS "Patch, already applied: ${patch}")
    endforeach()
    return()
  elseif(NOT recorded)
    message(FATAL_ERROR
      "${directory} has changes, and no record of the patches applied to it: they are not "
      "this project's, or were made before it kept one. git -C ${directory} status shows what "
      "is there; git -C ${directory} checkout -- . throws local changes away, and the next "
      "configuration applies the patches.")
  endif()

  # Off with the series recorded, last patch first.
  list(REVERSE recorded)
  foreach(entry IN LISTS recorded)
    string(REGEX REPLACE ":[0-9a-f]+$" "" name "${entry}")
    execute_process(COMMAND ${git} apply --reverse "${record}/${name}"
      WORKING_DIRECTORY "${tree}" RESULT_VARIABLE result ERROR_VARIABLE complaint)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR
        "The patch ${name}, recorded as applied to ${directory}, does not come off it:\n"
        "${complaint}\nThe tree has changes besides the patches. git -C ${directory} status "
        "shows what is there; git -C ${directory} checkout -- . throws local changes away.")
    endif()
    message(STATUS "Patch, taken off to apply its series again: ${name}")
  endforeach()
  file(REMOVE_RECURSE "${record}")

  # On with the series of patches/, recorded as it goes.
  file(MAKE_DIRECTORY "${record}")
  set(applied "")
  foreach(patch IN LISTS ARGN)
    set(file "${PROJECT_SOURCE_DIR}/${patch}")
    execute_process(COMMAND ${git} apply "${file}"
      WORKING_DIRECTORY "${tree}" RESULT_VARIABLE result ERROR_VARIABLE complaint)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR
        "${patch} does not apply to ${directory}:\n${complaint}\n"
        "The submodule may have moved on and carry the change already, or the patches before "
        "it may not have left what it changes.")
    endif()
    get_filename_component(name "${patch}" NAME)
    file(COPY_FILE "${file}" "${record}/${name}")
    file(SHA256 "${file}" hash)
    list(APPEND applied "${name}:${hash}")
    list(JOIN applied "\n" lines)
    file(WRITE "${record}/series" "${lines}\n")
    message(STATUS "Patch: ${patch}")
  endforeach()
endfunction()

file(GLOB SDASM_GGML_PATCHES RELATIVE "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/patches/ggml/*.patch")
file(GLOB SDASM_SD_PATCHES RELATIVE "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/patches/stable-diffusion.cpp/*.patch")
list(SORT SDASM_GGML_PATCHES)
list(SORT SDASM_SD_PATCHES)
sdasm_patch_series("thirdparty/stable-diffusion.cpp/ggml" ${SDASM_GGML_PATCHES})
sdasm_patch_series("thirdparty/stable-diffusion.cpp" ${SDASM_SD_PATCHES})
# A patch added to or taken from the directories configures again as well.
set_property(DIRECTORY "${PROJECT_SOURCE_DIR}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/patches/ggml" "${PROJECT_SOURCE_DIR}/patches/stable-diffusion.cpp")
