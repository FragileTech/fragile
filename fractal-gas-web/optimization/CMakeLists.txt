set(OPT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_library(fg_coco STATIC ${OPT_ROOT}/third_party/coco/coco.c)
target_include_directories(fg_coco PUBLIC ${OPT_ROOT}/third_party/coco)
set_target_properties(fg_coco PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)
if(NOT MSVC)
  target_compile_options(fg_coco PRIVATE -ffp-contract=off)
  target_link_libraries(fg_coco PUBLIC m)
endif()
# Pinned libcmaes, built with the same Eigen ABI as the optimization core.
set(CMA_ROOT ${OPT_ROOT}/third_party/libcmaes)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cma-generated/libcmaes)
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cma-generated/libcmaes/cmaes_export.h CONTENT "#define CMAES_EXPORT\n")
configure_file(${CMA_ROOT}/include/libcmaes/libcmaes_config.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cma-generated/libcmaes/libcmaes_config.h)
add_library(fg_libcmaes STATIC
  ${CMA_ROOT}/src/acovarianceupdate.cc ${CMA_ROOT}/src/cmaparameters.cc
  ${CMA_ROOT}/src/cmastopcriteria.cc ${CMA_ROOT}/src/covarianceupdate.cc
  ${CMA_ROOT}/src/esostrategy.cc ${CMA_ROOT}/src/pwq_bound_strategy.cc
  ${CMA_ROOT}/src/vdcmaupdate.cc ${CMA_ROOT}/src/bipopcmastrategy.cc
  ${CMA_ROOT}/src/cmasolutions.cc ${CMA_ROOT}/src/cmastrategy.cc
  ${CMA_ROOT}/src/ipopcmastrategy.cc)
target_include_directories(fg_libcmaes SYSTEM PUBLIC ${CMA_ROOT}/include
  ${CMAKE_CURRENT_BINARY_DIR}/cma-generated ${OPT_ROOT}/third_party/eigen)
target_compile_definitions(fg_libcmaes PUBLIC EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE EIGEN_DONT_VECTORIZE)
set_target_properties(fg_libcmaes PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fg_libcmaes PRIVATE -ffp-contract=off)
endif()
if(EMSCRIPTEN)
  target_compile_options(fg_libcmaes PRIVATE -fexceptions)
endif()
add_library(fg_optimization_core STATIC
  ${OPT_ROOT}/src/optimization/benchmark.cpp
  ${OPT_ROOT}/src/optimization/coco_benchmark.cpp
  ${OPT_ROOT}/src/optimization/perturbation.cpp
  ${OPT_ROOT}/src/optimization/environment.cpp
  ${OPT_ROOT}/src/optimization/gas.cpp
  ${OPT_ROOT}/src/optimization/gas2017.cpp
  ${OPT_ROOT}/src/optimization/engine.cpp
  ${OPT_ROOT}/src/optimization/cma.cpp)
target_link_libraries(fg_optimization_core PUBLIC fg_swarm_core fg_coco fg_libcmaes)
target_include_directories(fg_optimization_core SYSTEM PRIVATE
  ${OPT_ROOT}/third_party/lbfgspp/include ${OPT_ROOT}/third_party/eigen)
target_compile_definitions(fg_optimization_core PRIVATE EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE EIGEN_DONT_VECTORIZE)
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
    # COCO rotation generation nests 48 KiB and 16 KiB scratch arrays.
    "-sSTACK_SIZE=1048576" "-sSTACK_OVERFLOW_CHECK=2"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAPF64,UTF8ToString,stringToUTF8,lengthBytesUTF8"
    "-sEXPORTED_FUNCTIONS=_malloc,_free")
  set_target_properties(fg_optimization PROPERTIES OUTPUT_NAME optimization SUFFIX .mjs
    RUNTIME_OUTPUT_DIRECTORY ${OPT_ROOT}/web/optimization/engine)
else()
  add_library(fg_optimization SHARED ${OPT_ROOT}/src/optimization/c_api.cpp)
  add_executable(fg_optimization_tests ${OPT_ROOT}/tests/test_main.cpp
    ${OPT_ROOT}/tests/optimization/test_optimization.cpp
    ${OPT_ROOT}/tests/optimization/test_coco.cpp
    ${OPT_ROOT}/tests/optimization/test_cma.cpp
    ${OPT_ROOT}/tests/optimization/test_gas2017.cpp)
  target_compile_definitions(fg_optimization_tests PRIVATE
    COCO_FIXTURES="${OPT_ROOT}/tests/optimization/coco-fixtures.json"
    CMA_FIXTURES="${OPT_ROOT}/tests/optimization/cma-precision-fixtures.json")
  target_include_directories(fg_optimization_tests PRIVATE ${OPT_ROOT}/tests)
  target_link_libraries(fg_optimization_tests PRIVATE fg_optimization_core)
  add_test(NAME fg_optimization_tests COMMAND fg_optimization_tests)
  # Existing draw-for-draw Wave/Graph regressions without console emulators.
  add_executable(fg_swarm_tests ${OPT_ROOT}/tests/test_main.cpp
    ${OPT_ROOT}/tests/test_tensor_ops.cpp ${OPT_ROOT}/tests/test_cloning.cpp
    ${OPT_ROOT}/tests/test_fractal_gas.cpp ${OPT_ROOT}/tests/test_fractal_tree.cpp
    ${OPT_ROOT}/tests/test_arcade_planner.cpp)
  target_include_directories(fg_swarm_tests PRIVATE ${OPT_ROOT}/tests)
  target_link_libraries(fg_swarm_tests PRIVATE fg_swarm_core)
  add_test(NAME fg_swarm_tests COMMAND fg_swarm_tests)
endif()
target_link_libraries(fg_optimization PRIVATE fg_optimization_core)
