# Atari backend (ALE / Arcade Learning Environment). Self-contained: builds
# ALE's core from the third_party/ale submodule via add_subdirectory (Python
# bindings, SDL, vector and wasm interfaces all disabled — the core needs
# only C++17, zlib and threads), then adds:
#   fg_atari_env    static lib: src/atari_env.cpp (AtariEnv : BatchEnv)
#   fg_atari_tests  test runner: tests/test_main.cpp + tests/test_atari_env.cpp
# Included from the top-level CMakeLists via include(... OPTIONAL); everything
# is skipped for Emscripten builds or when the submodule is not checked out.

if(EMSCRIPTEN)
  return()
endif()

set(FG_ALE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ale)
if(NOT EXISTS ${FG_ALE_DIR}/src/ale/ale_interface.hpp)
  message(STATUS "FgAtari: third_party/ale submodule not checked out - skipping Atari backend")
  return()
endif()

find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  message(STATUS "FgAtari: zlib not found - skipping Atari backend")
  return()
endif()

# ALE build options (cache-forced so the submodule's defaults don't win).
set(BUILD_CPP_LIB ON CACHE BOOL "ALE C++ interface" FORCE)
set(BUILD_PYTHON_LIB OFF CACHE BOOL "ALE Python bindings" FORCE)
set(SDL_SUPPORT OFF CACHE BOOL "ALE SDL display/sound" FORCE)
set(BUILD_VECTOR_LIB OFF CACHE BOOL "ALE vector interface" FORCE)
set(BUILD_VECTOR_XLA_LIB OFF CACHE BOOL "ALE vector XLA support" FORCE)
set(BUILD_WASM_LIB OFF CACHE BOOL "ALE wasm interface" FORCE)

# EXCLUDE_FROM_ALL: only the targets fg_atari_env actually links (ale-lib and
# the ale object lib) get built. The submodule is never modified.
add_subdirectory(${FG_ALE_DIR} ${CMAKE_BINARY_DIR}/ale-build EXCLUDE_FROM_ALL)

add_library(fg_atari_env STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/atari_env.cpp)
# ALE's include dirs are directory-scoped inside the submodule, so consumers
# must add them explicitly: the source tree (headers are included as
# "ale/..."), plus the binary dir where the configured version.hpp lands.
target_include_directories(fg_atari_env PRIVATE
  ${FG_ALE_DIR}/src
  ${CMAKE_BINARY_DIR}/ale-build/src/ale)
target_link_libraries(fg_atari_env
  PUBLIC fractal_gas_core
  PRIVATE ale-lib)

add_executable(fg_atari_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_atari_env.cpp)
target_include_directories(fg_atari_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(fg_atari_tests PRIVATE fg_atari_env)

add_test(NAME fg_atari_tests COMMAND fg_atari_tests)
