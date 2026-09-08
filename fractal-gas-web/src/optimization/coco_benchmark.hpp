#pragma once
#include <memory>

#include "optimization/benchmark.hpp"

namespace fg::optimization {
// Thin ownership/evaluation adapter over the pinned upstream C implementation.
class CocoBenchmark {
 public:
  CocoBenchmark(int function, int dimension, int instance);
  ~CocoBenchmark();
  double evaluate(const double* x, bool simulation = false) const;
  double minimum() const;
  const std::string& problem_id() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
void append_coco_catalog(Json& benchmarks);
}  // namespace fg::optimization
