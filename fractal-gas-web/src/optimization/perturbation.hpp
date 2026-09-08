#pragma once
#include <functional>
#include <memory>

#include "optimization/benchmark.hpp"

namespace fg::optimization {
// A strategy draws one vector. The environment adds it to positions; BAOAB
// scales it by the existing thermal-noise coefficient during its O stage.
// Position and benchmark access permit future state-dependent proposals.
class Perturbation {
 public:
  virtual ~Perturbation() = default;
  virtual void sample(const float* position, float* delta, int dimensions,
                      Rng& rng) const = 0;
};
using PerturbationFactory =
    std::function<std::unique_ptr<Perturbation>(const Benchmark&, const Json&)>;
void register_perturbation(const std::string& id, const std::string& name,
                           PerturbationFactory factory, const Json& parameters);
std::unique_ptr<Perturbation> make_perturbation(const Benchmark&, const Json&);
Json perturbation_catalog();
}  // namespace fg::optimization
