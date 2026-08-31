#include <cstdio>

#include "test_framework.hpp"

int main() {
  int failed_tests = 0;
  for (const auto& test : fgtest::registry()) {
    fgtest::current_test() = test.name.c_str();
    const int before = fgtest::failure_count();
    test.fn();
    const bool ok = fgtest::failure_count() == before;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", test.name.c_str());
    if (!ok) ++failed_tests;
  }
  std::printf("\n%zu tests, %d failed\n", fgtest::registry().size(),
              failed_tests);
  return failed_tests == 0 ? 0 : 1;
}
