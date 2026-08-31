// Minimal dependency-free test harness.
#ifndef FRACTAL_GAS_TEST_FRAMEWORK_HPP
#define FRACTAL_GAS_TEST_FRAMEWORK_HPP

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace fgtest {

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline const char*& current_test() {
  static const char* name = "";
  return name;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

#define TEST_CASE(name)                                                     \
  static void test_fn_##name();                                             \
  static ::fgtest::Registrar registrar_##name(#name, test_fn_##name);       \
  static void test_fn_##name()

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::printf("  FAIL %s:%d [%s]: %s\n", __FILE__, __LINE__,            \
                  ::fgtest::current_test(), #cond);                         \
      ++::fgtest::failure_count();                                          \
    }                                                                       \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                                              \
  do {                                                                      \
    const double va = static_cast<double>(a);                               \
    const double vb = static_cast<double>(b);                               \
    const double scale = std::max(1.0, std::max(std::fabs(va), std::fabs(vb))); \
    if (!(std::fabs(va - vb) <= (tol) * scale)) {                           \
      std::printf("  FAIL %s:%d [%s]: %s (%g) !~ %s (%g)\n", __FILE__,      \
                  __LINE__, ::fgtest::current_test(), #a, va, #b, vb);      \
      ++::fgtest::failure_count();                                          \
    }                                                                       \
  } while (0)

}  // namespace fgtest

#endif  // FRACTAL_GAS_TEST_FRAMEWORK_HPP
