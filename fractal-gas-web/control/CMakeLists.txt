set(CONTROL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
find_package(Threads REQUIRED)
add_library(fg_control_core STATIC
  ${CONTROL_ROOT}/src/control/agent_types.cpp
  ${CONTROL_ROOT}/src/control/actuators.cpp
  ${CONTROL_ROOT}/src/control/extensions.cpp
  ${CONTROL_ROOT}/src/control/scene.cpp
  ${CONTROL_ROOT}/src/control/state.cpp
  ${CONTROL_ROOT}/src/control/physics.cpp
  ${CONTROL_ROOT}/src/control/wave.cpp
)
option(FG_CONTROL_PROFILE "Instrument collision work in benchmark builds" OFF)
if(FG_CONTROL_PROFILE)
  target_compile_definitions(fg_control_core PUBLIC FG_CONTROL_PROFILE)
endif()
set_target_properties(fg_control_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fg_control_core PUBLIC ${CONTROL_ROOT}/src)
target_link_libraries(fg_control_core PUBLIC Threads::Threads fg_fractal_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fg_control_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()
if(EMSCRIPTEN)
  target_compile_options(fg_control_core PRIVATE -fexceptions -msimd128)
  if(FG_CONTROL_THREADS)
    target_compile_options(fg_control_core PUBLIC -pthread)
    target_link_options(fg_control_core PUBLIC -pthread)
  endif()
  add_executable(fg_control ${CONTROL_ROOT}/src/control/c_api.cpp)
  target_link_libraries(fg_control PRIVATE fg_control_core)
  target_compile_options(fg_control PRIVATE -fexceptions)
  target_link_options(fg_control PRIVATE --no-entry -fexceptions -O3
    "-sMODULARIZE=1" "-sEXPORT_ES6=1" "-sEXPORT_NAME=createControlModule"
    "-sENVIRONMENT=worker,node" "-sALLOW_MEMORY_GROWTH=1" "-sINITIAL_MEMORY=33554432"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAPF32,HEAPF64,HEAP32,UTF8ToString,stringToUTF8,lengthBytesUTF8"
    "-sEXPORTED_FUNCTIONS=_malloc,_free")
  set_target_properties(fg_control PROPERTIES SUFFIX ".mjs" RUNTIME_OUTPUT_DIRECTORY ${CONTROL_ROOT}/web/lab/engine)
  if(FG_CONTROL_THREADS)
    set_target_properties(fg_control PROPERTIES OUTPUT_NAME "control-threaded")
    # The caller is one slot. Prewarm only the requested additional workers.
    target_link_options(fg_control PRIVATE
      "-sPTHREAD_POOL_SIZE=Math.max(0,Math.min(64,Module['controlThreads']||4)-1)"
      "-sPTHREAD_POOL_SIZE_STRICT=2")
  else()
    set_target_properties(fg_control PROPERTIES OUTPUT_NAME "control")
  endif()
else()
  add_library(fg_control SHARED ${CONTROL_ROOT}/src/control/c_api.cpp)
  target_link_libraries(fg_control PRIVATE fg_control_core)
  enable_testing()
  add_executable(fg_control_tests ${CONTROL_ROOT}/tests/test_main.cpp ${CONTROL_ROOT}/tests/test_control.cpp ${CONTROL_ROOT}/tests/test_control_collisions.cpp ${CONTROL_ROOT}/tests/test_control_scheduling.cpp)
  target_include_directories(fg_control_tests PRIVATE ${CONTROL_ROOT}/tests)
  target_link_libraries(fg_control_tests PRIVATE fg_control_core)
  add_test(NAME fg_control_tests COMMAND fg_control_tests)
  add_executable(fg_control_allocations ${CONTROL_ROOT}/tests/test_control_allocations.cpp)
  target_link_libraries(fg_control_allocations PRIVATE fg_control_core)
  add_test(NAME fg_control_allocations COMMAND fg_control_allocations)
  if(FG_CONTROL_PROFILE)
    add_executable(fg_control_collision_benchmark collision_benchmark.cpp)
    target_link_libraries(fg_control_collision_benchmark PRIVATE fg_control_core)
  endif()
  add_executable(fg_control_benchmark benchmark.cpp)
  target_link_libraries(fg_control_benchmark PRIVATE fg_control_core)
endif()
