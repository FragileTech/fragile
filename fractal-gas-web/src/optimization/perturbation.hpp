#pragma once
#include <functional>
#include <memory>

#include "optimization/benchmark.hpp"

namespace fg::optimization {
struct PerturbationContext {
  double normalized_objective;
};
// A strategy draws one vector. The environment adds it to positions; BAOAB
// scales it by the existing thermal-noise coefficient during its O stage.
// Position and benchmark access permit future state-dependent proposals.
struct PerturbationTransition {
  std::vector<float> origin;
  std::vector<double> displacement;
  int draws = 0;
  double improvement = 0;
  double scale = 1;  // Accepted retry scale, independent of covariance shape.
};
class Perturbation {
 public:
  virtual ~Perturbation() = default;
  // Observation may collect data, but only update() changes sampling geometry.
  virtual void observe(const PerturbationTransition&) {}
  virtual void update() {}
  virtual void reset() {}
  virtual void sample(const float* position, float* delta, int dimensions,
                      Rng& rng) const = 0;
  // The first position is fixed at the action origin; the second is current.
  virtual void sample_action(const float*, const float* position, float* delta,
                             int dimensions, Rng& rng) const {
    sample(position, delta, dimensions, rng);
  }
  virtual void sample_with_context(const float* position, float* delta,
                                   int dimensions, Rng& rng,
                                   const PerturbationContext*) const {
    sample(position, delta, dimensions, rng);
  }
};
using PerturbationFactory =
    std::function<std::unique_ptr<Perturbation>(const Benchmark&, const Json&)>;
void register_perturbation(const std::string& id, const std::string& name,
                           PerturbationFactory factory, const Json& parameters,
                           const Json& algorithms = Json{});
std::unique_ptr<Perturbation> make_perturbation(const Benchmark&, const Json&);
Json perturbation_catalog();
}  // namespace fg::optimization
