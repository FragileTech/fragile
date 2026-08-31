# CMake generated Testfile for 
# Source directory: /home/guillem/fragile/fractal-gas-web
# Build directory: /home/guillem/fragile/fractal-gas-web/build-atari
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(fg_atari_tests "/home/guillem/fragile/fractal-gas-web/build-atari/fg_atari_tests")
set_tests_properties(fg_atari_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/guillem/fragile/fractal-gas-web/cmake/FgAtari.cmake;55;add_test;/home/guillem/fragile/fractal-gas-web/cmake/FgAtari.cmake;0;;/home/guillem/fragile/fractal-gas-web/CMakeLists.txt;46;include;/home/guillem/fragile/fractal-gas-web/CMakeLists.txt;0;")
add_test(fg_tests "/home/guillem/fragile/fractal-gas-web/build-atari/fg_tests")
set_tests_properties(fg_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/guillem/fragile/fractal-gas-web/CMakeLists.txt;86;add_test;/home/guillem/fragile/fractal-gas-web/CMakeLists.txt;0;")
subdirs("ale-build")
