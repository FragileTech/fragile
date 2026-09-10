set(LLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
if(EMSCRIPTEN)
  add_executable(fg_llm ${LLM_ROOT}/src/llm/c_api.cpp)
  target_compile_options(fg_llm PRIVATE -fexceptions)
  target_link_libraries(fg_llm PRIVATE fg_swarm_core)
  target_link_options(fg_llm PRIVATE --no-entry -fexceptions -O3
    "-sASYNCIFY=1" "-sASYNCIFY_STACK_SIZE=131072" "-sMODULARIZE=1" "-sEXPORT_ES6=1"
    "-sEXPORT_NAME=createLlmModule" "-sENVIRONMENT=worker,node" "-sALLOW_MEMORY_GROWTH=1"
    "-sEXPORTED_RUNTIME_METHODS=ccall,UTF8ToString,stringToUTF8,lengthBytesUTF8"
    "-sEXPORTED_FUNCTIONS=_malloc,_free")
  set_target_properties(fg_llm PROPERTIES OUTPUT_NAME llm SUFFIX .mjs
    RUNTIME_OUTPUT_DIRECTORY ${LLM_ROOT}/web/llm/engine)
else()
  add_library(fg_llm SHARED ${LLM_ROOT}/src/llm/c_api.cpp)
  target_link_libraries(fg_llm PRIVATE fg_swarm_core)
  add_executable(fg_llm_tests ${LLM_ROOT}/tests/test_main.cpp
    ${LLM_ROOT}/tests/llm/test_llm.cpp ${LLM_ROOT}/tests/test_tensor_ops.cpp
    ${LLM_ROOT}/tests/test_cloning.cpp ${LLM_ROOT}/tests/test_fractal_gas.cpp
    ${LLM_ROOT}/tests/test_fractal_tree.cpp ${LLM_ROOT}/tests/test_arcade_planner.cpp
    ${LLM_ROOT}/tests/test_fractal_core.cpp)
  target_include_directories(fg_llm_tests PRIVATE ${LLM_ROOT}/tests)
  target_link_libraries(fg_llm_tests PRIVATE fg_swarm_core)
  add_test(NAME fg_llm_tests COMMAND fg_llm_tests)
endif()
