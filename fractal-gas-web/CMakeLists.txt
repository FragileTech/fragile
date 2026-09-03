cmake_minimum_required(VERSION 3.16)
project(fractal_gas_web C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# nes-py C++ NES emulator core (git submodule). Built from source directly
# (its own SCons build is ignored). lib_nes_env.cpp / emulator_pool.cpp are
# the Python-ctypes glue and are excluded.
# ---------------------------------------------------------------------------
set(NES_PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nes-py/nes_py/nes)
file(GLOB NES_CORE_SOURCES
  ${NES_PY_DIR}/src/*.cpp
  ${NES_PY_DIR}/src/mappers/*.cpp
)
list(FILTER NES_CORE_SOURCES EXCLUDE REGEX "lib_nes_env|emulator_pool")

add_library(nes_core STATIC ${NES_CORE_SOURCES})
target_include_directories(nes_core PUBLIC ${NES_PY_DIR}/include)

# ---------------------------------------------------------------------------
# Fractal gas core (the C++ port of the algorithm)
# ---------------------------------------------------------------------------
add_library(fractal_gas_core STATIC
  src/tensor_ops.cpp
  src/walker_state.cpp
  src/cloning.cpp
  src/mario_reward.cpp
  src/nes_env.cpp
  src/thread_pool.cpp
  src/fractal_gas.cpp
  src/fractal_tree.cpp
  src/visit_grid.cpp
)
target_include_directories(fractal_gas_core PUBLIC src)
target_link_libraries(fractal_gas_core PUBLIC nes_core)

find_package(Threads REQUIRED)
target_link_libraries(fractal_gas_core PUBLIC Threads::Threads)

# Optional console backends (each file owns its whole target setup; absent
# files are skipped). See cmake/FgAtari.cmake (ALE) and cmake/FgRetro.cmake
# (stable-retro / Genesis).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FgAtari.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FgRetro.cmake OPTIONAL)

if(EMSCRIPTEN)
  # -------------------------------------------------------------------------
  # WebAssembly demo. pthreads need SharedArrayBuffer, which needs COOP/COEP
  # headers (see serve.py). Memory growth is off in the main module (fragile
  # with pthreads), so preallocate generously.
  # -------------------------------------------------------------------------
  # -fexceptions: without it any C++ throw aborts the whole wasm runtime.
  # -msimd128: wasm SIMD (supported by every current browser) — mainly for
  # the vectorized L2 distance kernel in tensor_ops.cpp.
  set(WASM_FLAGS "-pthread" "-fexceptions" "-msimd128")
  target_compile_options(nes_core PRIVATE ${WASM_FLAGS})
  target_compile_options(fractal_gas_core PRIVATE ${WASM_FLAGS})

  # ---- Atari (ALE) — instance-based, keeps pthread parallelism in wasm ----
  # Requires the emscripten zlib port: run `embuilder build zlib` once.
  set(ZLIB_LIBRARY ${EMSCRIPTEN_SYSROOT}/lib/wasm32-emscripten/libz.a CACHE FILEPATH "" FORCE)
  set(ZLIB_INCLUDE_DIR ${EMSCRIPTEN_SYSROOT}/include CACHE PATH "" FORCE)
  set(BUILD_CPP_LIB ON CACHE BOOL "" FORCE)
  set(BUILD_PYTHON_LIB OFF CACHE BOOL "" FORCE)
  set(SDL_SUPPORT OFF CACHE BOOL "" FORCE)
  set(BUILD_VECTOR_LIB OFF CACHE BOOL "" FORCE)
  set(BUILD_VECTOR_XLA_LIB OFF CACHE BOOL "" FORCE)
  set(BUILD_WASM_LIB OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/ale ${CMAKE_BINARY_DIR}/ale-build EXCLUDE_FROM_ALL)
  # ALE links into the pthread-enabled main module; the Genesis shim below
  # is single-threaded, so -pthread is scoped to ALE's targets rather than
  # set globally.
  foreach(ale_target ale ale-lib)
    if(TARGET ${ale_target})
      target_compile_options(${ale_target} PRIVATE -pthread)
    endif()
  endforeach()

  # ---- Genesis (Genesis Plus GX) — one core-worker shim module per      ----
  # ---- Web Worker (see src/retro_farm_env.cpp / src/retro_shim.cpp).   ----
  # libretro cores are global-state singletons and emscripten's dynamic
  # linking faults under pthreads, so instead each plain Web Worker
  # instantiates its OWN copy of a statically linked shim module
  # (web/retro_shim.{js,wasm}) — separate memories, true isolation, true
  # parallelism. The shim is single-threaded, so the core builds WITHOUT
  # -pthread.
  set(GPGX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-retro/cores/genesis)
  set(GPGX_BUILD ${CMAKE_BINARY_DIR}/gpgx-wasm)
  set(GPGX_LIB ${GPGX_BUILD}/libgpgx.a)
  add_custom_command(
    OUTPUT ${GPGX_LIB} ${GPGX_BUILD}/fg_file_stream.o ${GPGX_BUILD}/fg_file_stream_transforms.o
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${GPGX_BUILD}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${GPGX_SRC} ${GPGX_BUILD}
    # The core hardcodes -fcommon (unsupported by wasm objects); patch the
    # BUILD COPY only — the submodule stays pristine.
    COMMAND sed -i "s/^CFLAGS := -fcommon/CFLAGS := -fno-common/" ${GPGX_BUILD}/Makefile.libretro
    # Two globals are TENTATIVELY DEFINED in headers (config in osd.h,
    # zbank_memory_map in membnk.h — CRLF file, so no $ anchors). ELF merges
    # such commons, but wasm-ld would give each object its own copy, so
    # config_default()'s settings never reach the core and the game freezes.
    # Patch the build copy to proper extern declarations + one definition.
    COMMAND sed -i "84s/^struct$/struct gpgx_config_s/" ${GPGX_BUILD}/libretro/osd.h
    COMMAND sed -i "s/^} config;/}; extern struct gpgx_config_s config;/" ${GPGX_BUILD}/libretro/osd.h
    COMMAND sed -i "s/^} zbank_memory_map\\[256\\];/}; extern struct _zbank_memory_map zbank_memory_map[256];/" ${GPGX_BUILD}/core/membnk.h
    COMMAND sh -c "echo 'struct gpgx_config_s config;' >> ${GPGX_BUILD}/libretro/libretro.c"
    COMMAND sh -c "echo 'struct _zbank_memory_map zbank_memory_map[256];' >> ${GPGX_BUILD}/core/membnk.c"
    # scrc32.c duplicates zlib's crc32 (the shim links the zlib port);
    # drop it from the build copy's source list.
    COMMAND sed -i "/scrc32.c/d" ${GPGX_BUILD}/libretro/Makefile.common
    # NOTE on savestates: GPGX's state.c memcpys whole structs that contain
    # pointers (Z80_Regs, YM2612), so savestate byte layout is host-ABI
    # specific — stable-retro's x86-64-written .state files CANNOT be
    # restored by the wasm32 build. The shim therefore boots the game from
    # power-on instead (retro_boot_to_gameplay); walker blobs are unaffected
    # (written and read by the same build).
    COMMAND make -C ${GPGX_BUILD} -f Makefile.libretro platform=emscripten
            STATIC_LINKING=1 CC=${CMAKE_C_COMPILER} AR=${CMAKE_AR} -j
    # The makefile names the archive *.bc, which emcc treats as LLVM
    # bitcode — rename to .a.
    COMMAND ${CMAKE_COMMAND} -E rename
      ${GPGX_BUILD}/genesis_plus_gx_libretro_emscripten.bc ${GPGX_LIB}
    # With STATIC_LINKING=1 the makefile omits these libretro-common
    # helpers, expecting the frontend to provide them.
    COMMAND ${CMAKE_C_COMPILER} -O2 -fno-common
      -I${GPGX_BUILD}/libretro/libretro-common/include
      -c ${GPGX_BUILD}/libretro/libretro-common/streams/file_stream.c
      -o ${GPGX_BUILD}/fg_file_stream.o
    COMMAND ${CMAKE_C_COMPILER} -O2 -fno-common
      -I${GPGX_BUILD}/libretro/libretro-common/include
      -c ${GPGX_BUILD}/libretro/libretro-common/streams/file_stream_transforms.c
      -o ${GPGX_BUILD}/fg_file_stream_transforms.o
    COMMENT "Building Genesis Plus GX static core for the wasm shim"
    VERBATIM
  )
  add_custom_target(gpgx_core_wasm DEPENDS ${GPGX_LIB})

  # The core-worker shim module: one statically linked GPGX + the shared
  # game logic, instantiated once per Web Worker. Single-threaded; memory
  # growth allowed (no pthreads here).
  add_executable(retro_shim src/retro_shim.cpp src/retro_core.cpp)
  add_dependencies(retro_shim gpgx_core_wasm)
  target_compile_options(retro_shim PRIVATE -fexceptions)
  target_compile_definitions(retro_shim PRIVATE FG_RETRO_STATIC)
  target_include_directories(retro_shim PRIVATE src third_party/stable-retro/src)
  target_link_libraries(retro_shim PRIVATE ${GPGX_LIB}
    ${GPGX_BUILD}/fg_file_stream.o ${GPGX_BUILD}/fg_file_stream_transforms.o
    ${ZLIB_LIBRARY})
  set_target_properties(retro_shim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web
    LINK_FLAGS "-fexceptions -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createRetroShim -sENVIRONMENT=worker -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sFORCE_FILESYSTEM=1 -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPF32,UTF8ToString -sEXPORTED_FUNCTIONS=_malloc,_free -O3"
  )

  add_executable(fractal_gas
    src/wasm_bindings.cpp
    src/atari_env.cpp
    src/retro_farm_env.cpp
  )
  add_dependencies(fractal_gas retro_shim)
  target_compile_options(fractal_gas PRIVATE ${WASM_FLAGS})
  target_include_directories(fractal_gas PRIVATE
    third_party/ale/src
    ${CMAKE_BINARY_DIR}/ale-build/src/ale
    third_party/stable-retro/src  # libretro.h (via retro_core.hpp)
  )
  target_link_libraries(fractal_gas PRIVATE fractal_gas_core ale-lib)

  # Ship the freely-distributable Airstriker ROM next to the page.
  add_custom_command(TARGET fractal_gas POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-retro/stable_retro/data/stable/Airstriker-Genesis-v0/rom.md
      ${CMAKE_CURRENT_SOURCE_DIR}/web/airstriker.rom
  )
  set_target_properties(fractal_gas PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web
    LINK_FLAGS "-pthread -fexceptions -sPTHREAD_POOL_SIZE=9 -sINITIAL_MEMORY=536870912 -sALLOW_MEMORY_GROWTH=0 -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createFractalGasModule -sENVIRONMENT=worker -sEXPORTED_RUNTIME_METHODS=HEAPU8 -sEXPORTED_FUNCTIONS=_malloc,_free --bind -O3"
  )
else()
  # -------------------------------------------------------------------------
  # Native CLI + tests
  # -------------------------------------------------------------------------
  add_executable(fg_cli native/main.cpp)
  target_link_libraries(fg_cli PRIVATE fractal_gas_core)

  add_executable(fg_tests
    tests/test_main.cpp
    tests/test_tensor_ops.cpp
    tests/test_cloning.cpp
    tests/test_fractal_gas.cpp
    tests/test_nes_env.cpp
    tests/test_montezuma_logic.cpp
    tests/test_visit_grid.cpp
    tests/test_fractal_tree.cpp
  )
  target_include_directories(fg_tests PRIVATE tests)
  target_link_libraries(fg_tests PRIVATE fractal_gas_core)

  enable_testing()
  add_test(NAME fg_tests COMMAND fg_tests)
endif()
