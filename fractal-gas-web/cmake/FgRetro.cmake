# Sega Genesis backend (Genesis Plus GX libretro core from the stable-retro
# submodule) — native builds only. Self-contained: defines the core .so build,
# the fg_retro_env static library and the fg_retro_tests executable.
#
# The core is built with its own Makefile.libretro from a copy of the source
# tree placed in the build directory, so the submodule stays pristine (no .o
# files inside third_party/). No core source is modified.

if(EMSCRIPTEN)
  return()
endif()

set(FG_RETRO_SUBMODULE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-retro)
set(FG_RETRO_CORE_SRC ${FG_RETRO_SUBMODULE}/cores/genesis)
set(FG_RETRO_DATA
    ${FG_RETRO_SUBMODULE}/stable_retro/data/stable/Airstriker-Genesis-v0)

if(NOT EXISTS ${FG_RETRO_CORE_SRC}/Makefile.libretro
   OR NOT EXISTS ${FG_RETRO_DATA}/rom.md)
  message(STATUS "fg_retro: stable-retro submodule not present, skipping")
  return()
endif()

find_package(ZLIB REQUIRED)  # Level1.state is a gzipped savestate

# ---------------------------------------------------------------------------
# Genesis Plus GX core shared library, built once via Makefile.libretro.
# Sources are copied to the build tree at configure time (only when absent,
# so incremental make stays incremental).
# ---------------------------------------------------------------------------
set(FG_RETRO_CORE_BUILD ${CMAKE_BINARY_DIR}/gpgx-core)
set(FG_RETRO_CORE_SO ${FG_RETRO_CORE_BUILD}/genesis_plus_gx_libretro.so)

if(NOT EXISTS ${FG_RETRO_CORE_BUILD}/Makefile.libretro)
  file(COPY ${FG_RETRO_CORE_SRC}/ DESTINATION ${FG_RETRO_CORE_BUILD})
endif()

include(ProcessorCount)
ProcessorCount(FG_RETRO_NPROC)
if(FG_RETRO_NPROC EQUAL 0)
  set(FG_RETRO_NPROC 4)
endif()

add_custom_command(
  OUTPUT ${FG_RETRO_CORE_SO}
  COMMAND make -f Makefile.libretro platform=unix "GIT_VERSION= fg"
          -j${FG_RETRO_NPROC}
  WORKING_DIRECTORY ${FG_RETRO_CORE_BUILD}
  DEPENDS ${FG_RETRO_CORE_SRC}/Makefile.libretro
  COMMENT "Building Genesis Plus GX libretro core (${FG_RETRO_CORE_SO})"
  VERBATIM)
add_custom_target(fg_retro_core_so DEPENDS ${FG_RETRO_CORE_SO})

# ---------------------------------------------------------------------------
# fg_retro_env: RetroCore (per-instance dlopen'd core copy) + RetroEnv
# (BatchEnv implementation with the Airstriker reward).
# ---------------------------------------------------------------------------
add_library(fg_retro_env STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/retro_core.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/retro_env.cpp
)
# libretro.h is vendored by stable-retro (header-only use of the submodule).
target_include_directories(fg_retro_env PUBLIC ${FG_RETRO_SUBMODULE}/src)
target_link_libraries(fg_retro_env PUBLIC fractal_gas_core ZLIB::ZLIB
                      ${CMAKE_DL_LIBS})
add_dependencies(fg_retro_env fg_retro_core_so)

# Default asset locations, overridable at runtime via FG_RETRO_CORE /
# FG_RETRO_ROM / FG_RETRO_STATE environment variables (see the tests).
set(FG_RETRO_DEFINES
  FG_RETRO_CORE_SO_DEFAULT="${FG_RETRO_CORE_SO}"
  FG_RETRO_ROM_DEFAULT="${FG_RETRO_DATA}/rom.md"
  FG_RETRO_STATE_DEFAULT="${FG_RETRO_DATA}/Level1.state"
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(fg_retro_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_retro_env.cpp
)
target_include_directories(fg_retro_tests PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_definitions(fg_retro_tests PRIVATE ${FG_RETRO_DEFINES})
target_link_libraries(fg_retro_tests PRIVATE fg_retro_env)

enable_testing()
add_test(NAME fg_retro_tests COMMAND fg_retro_tests)
