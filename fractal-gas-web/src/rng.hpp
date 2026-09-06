// Random number generation for the fractal gas port.
//
// All algorithm-level randomness flows through the Rng interface so tests can
// substitute recorded draws and reproduce a Python reference run. Bitwise
// compatibility with torch's Philox RNG is out of scope; equivalence with the
// Python implementation is at the distribution/semantics level (see README).
#ifndef FRACTAL_GAS_RNG_HPP
#define FRACTAL_GAS_RNG_HPP

#include <cstdint>
#include <random>
#include <sstream>
#include <locale>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fg {

class Rng {
 public:
  virtual ~Rng() = default;

  /// Uniform sample in [0, 1). One draw per call (like one torch.rand element).
  virtual float uniform01() = 0;

  /// Uniform integer in [lo, hi) — torch.randint convention (hi exclusive).
  virtual int64_t randint(int64_t lo, int64_t hi) = 0;

  /// Random permutation of 0..n-1 (torch.randperm equivalent), built with
  /// Fisher-Yates from randint draws.
  std::vector<int32_t> permutation(int32_t n) {
    std::vector<int32_t> p(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) p[static_cast<size_t>(i)] = i;
    for (int32_t i = n - 1; i > 0; --i) {
      const auto j = static_cast<int32_t>(randint(0, i + 1));
      std::swap(p[static_cast<size_t>(i)], p[static_cast<size_t>(j)]);
    }
    return p;
  }
};

/// Default production RNG: one mt19937_64 stream per FractalGas instance
/// (the Python implementation uses one global torch RNG stream for
/// companions, clone draws, actions and dt — a single stream reproduces the
/// same coupling structure).
class Mt19937Rng final : public Rng {
 public:
  explicit Mt19937Rng(uint64_t seed) : gen_(seed) {}

  std::string checkpoint() const {
    std::ostringstream out;out.imbue(std::locale::classic());out<<gen_;return out.str();
  }
  void restore(const std::string& state) {
    std::istringstream in(state);in.imbue(std::locale::classic());
    std::mt19937_64 next;if(!(in>>next))throw std::invalid_argument("Invalid planner RNG state");
    in>>std::ws;if(!in.eof())throw std::invalid_argument("Trailing planner RNG data");gen_=next;
  }
  float uniform01() override {
    return static_cast<float>(std::generate_canonical<double, 53>(gen_));
  }

  int64_t randint(int64_t lo, int64_t hi) override {
    std::uniform_int_distribution<int64_t> dist(lo, hi - 1);
    return dist(gen_);
  }

 private:
  std::mt19937_64 gen_;
};

}  // namespace fg

#endif  // FRACTAL_GAS_RNG_HPP
