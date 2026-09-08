#pragma once
#include <memory>
#include <string>
#include <vector>

#include "control/json.hpp"
#include "rng.hpp"

namespace fg::optimization {
using control::Json;
using control::JsonReader;
constexpr double pi = 3.14159265358979323846;
// Explicit integer-to-uniform mapping avoids libc++/libstdc++ distribution
// differences between WebAssembly and native builds.
class OptimizationRng final : public Rng {
 public:
  explicit OptimizationRng(uint64_t seed) : generator_(seed) {}
  float uniform01() override {
    return float(generator_() >> 40) * (1.0f / 16777216.0f);
  }
  int64_t randint(int64_t low, int64_t high) override {
    if (high <= low)
      throw std::invalid_argument("Empty random integer interval");
    const uint64_t range = uint64_t(high - low),
                   threshold = (uint64_t(0) - range) % range;
    uint64_t draw;
    do {
      draw = generator_();
    } while (draw < threshold);
    return low + int64_t(draw % range);
  }

 private:
  std::mt19937_64 generator_;
};

std::string stringify(const Json& value);
Json number(double value);
Json array(const std::vector<double>& values);
int integer(const Json& value, int fallback, int low, int high,
            const char* name);
double bounded(const Json& value, double fallback, double low, double high,
               const char* name);
// Box-Muller without cached draws: replay fixtures can inject every uniform.
double normal(Rng& rng);
const std::string& catalog_json();

class CocoBenchmark;
class Benchmark {
 public:
  explicit Benchmark(const Json& config);
  ~Benchmark();
  std::string id;
  int d;
  double low, high;
  bool stochastic = false;
  bool coco = false;
  Json config;
  double evaluate(const float* x, Rng* rng = nullptr) const;
  double evaluate_optimization(const float* x, Rng* rng = nullptr) const;
  void gradient(const float* x, float* out, bool optimization = false) const;
  mutable uint64_t evaluations = 0;
  mutable double best_observed = INFINITY;
  // Upper bound: nonsmooth classic functions only use queries at a cusp.
  uint64_t gradient_evaluations() const {
    return coco || id == "eggholder" || id == "holder_table" ? 2 * uint64_t(d)
                                                             : 0;
  }
  void initial(float* x, Rng& rng) const;
  bool valid(const float* x) const;
  void wrap(float* x) const;

 private:
  double alpha = .1, lambda = .13, radius = 1, tilt = 0, stddev = 1;
  int components = 3;
  std::vector<double> centers, stds, weights;
  std::unique_ptr<CocoBenchmark> coco_problem;
  double value(const std::vector<double>& x) const;
  void observe(const double* x, double y) const;
};
}  // namespace fg::optimization
