# Shared emulator-independent numerical and swarm libraries.
find_package(Threads REQUIRED)
add_library(fg_fractal_core STATIC
  src/fractal/tensor_ops.cpp src/fractal/cloning.cpp src/thread_pool.cpp
  src/fractal/exploration_tree.cpp src/fractal/visit_grid.cpp)
add_library(fg_numeric_core ALIAS fg_fractal_core)
target_include_directories(fg_fractal_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fg_fractal_core PUBLIC Threads::Threads)
add_library(fg_swarm_core STATIC
  src/walker_state.cpp src/fractal_gas.cpp src/fractal_tree.cpp
  src/arcade_planner.cpp)
target_link_libraries(fg_swarm_core PUBLIC fg_numeric_core)
foreach(target fg_fractal_core fg_swarm_core)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
  endif()
  if(EMSCRIPTEN)
    target_compile_options(${target} PRIVATE -fexceptions -msimd128)
    if(FG_CONTROL_THREADS OR (NOT FG_CONTROL_ONLY AND NOT FG_OPTIMIZATION_ONLY))
      target_compile_options(${target} PUBLIC -pthread)
      target_link_options(${target} PUBLIC -pthread)
    endif()
  endif()
endforeach()
