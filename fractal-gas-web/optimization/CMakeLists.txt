set(OPT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_library(fg_optimization_core STATIC
  ${OPT_ROOT}/src/optimization/benchmark.cpp
  ${OPT_ROOT}/src/optimization/gas.cpp
  ${OPT_ROOT}/src/optimization/engine.cpp)
target_link_libraries(fg_optimization_core PUBLIC fg_swarm_core)
set_target_properties(fg_optimization_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fg_optimization_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()
if(EMSCRIPTEN)
  target_compile_options(fg_optimization_core PRIVATE -fexceptions -msimd128)
  add_executable(fg_optimization ${OPT_ROOT}/src/optimization/c_api.cpp)
  target_compile_options(fg_optimization PRIVATE -fexceptions)
  target_link_options(fg_optimization PRIVATE --no-entry -fexceptions -O3
    "-sMODULARIZE=1" "-sEXPORT_ES6=1" "-sEXPORT_NAME=createOptimizationModule"
    "-sENVIRONMENT=worker,node" "-sALLOW_MEMORY_GROWTH=1" "-sINITIAL_MEMORY=33554432"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAPF64,UTF8ToString,stringToUTF8,lengthBytesUTF8"
    "-sEXPORTED_FUNCTIONS=_malloc,_free")
  set_target_properties(fg_optimization PROPERTIES OUTPUT_NAME optimization SUFFIX .mjs
    RUNTIME_OUTPUT_DIRECTORY ${OPT_ROOT}/web/optimization/engine)
else()
  add_library(fg_optimization SHARED ${OPT_ROOT}/src/optimization/c_api.cpp)
  add_executable(fg_optimization_tests ${OPT_ROOT}/tests/test_main.cpp
    ${OPT_ROOT}/tests/optimization/test_optimization.cpp)
  target_include_directories(fg_optimization_tests PRIVATE ${OPT_ROOT}/tests)
  target_link_libraries(fg_optimization_tests PRIVATE fg_optimization_core)
  add_test(NAME fg_optimization_tests COMMAND fg_optimization_tests)
  # Existing draw-for-draw Wave/Graph regressions without console emulators.
  add_executable(fg_swarm_tests ${OPT_ROOT}/tests/test_main.cpp
    ${OPT_ROOT}/tests/test_tensor_ops.cpp ${OPT_ROOT}/tests/test_cloning.cpp
    ${OPT_ROOT}/tests/test_fractal_gas.cpp ${OPT_ROOT}/tests/test_fractal_tree.cpp)
  target_include_directories(fg_swarm_tests PRIVATE ${OPT_ROOT}/tests)
  target_link_libraries(fg_swarm_tests PRIVATE fg_swarm_core)
  add_test(NAME fg_swarm_tests COMMAND fg_swarm_tests)
endif()
target_link_libraries(fg_optimization PRIVATE fg_optimization_core)
