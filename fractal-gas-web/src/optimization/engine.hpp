#pragma once
#include <functional>
#include <memory>

#include "fractal/euclidean.hpp"
#include "optimization/benchmark.hpp"
#include "optimization/perturbation.hpp"
#include "swarm_algorithm.hpp"

namespace fg::optimization {
struct Settings {
  explicit Settings(const Json& input);
  Json json;
  std::string algorithm, companion, clone_companion, objective, perturbation;
  int walkers, max_walkers, seed, dt_min, dt_max, clone_every, substeps, elites;
  int horizon, max_horizon;
  int gas_local_evaluations;
  bool gas_tabu, gas_local_search;
  uint64_t max_evaluations;
  double proposal, gamma, beta, delta_t, epsilon, clone_epsilon, lambda_alg, reward_coef,
      distance_coef, eta, sigma_min, amplitude, epsilon_dist, rho, p_max, epsilon_clone, sigma_x,
      restitution;
  bool periodic, potential_force, cloning, kinetic, consensus_prefix;
  double score(double value) const { return objective == "minimize" ? -value : value; }
  bool better(double a, double b) const { return score(a) > score(b); }
  double worst() const { return objective == "minimize" ? INFINITY : -INFINITY; }
  bool cma() const { return algorithm == "cmaes_active" || algorithm == "cmaes_bipop"; }
  bool planning() const { return algorithm == "fmc" || algorithm == "wave_jump"; }
};
using Population = fractal::EuclideanPopulation;
// Pure operators share the Python reference's semantics and accept injected
// RNG.
std::vector<int32_t> select_companions(const Population&, const Settings&, const Benchmark&,
                                       const std::string&, double, Rng&);
std::vector<float> fitness(const Population&, const Settings&, const Benchmark&,
                           const std::vector<int32_t>&);
void clone_population(Population&, const Settings&, const std::vector<int32_t>&,
                      const std::vector<uint8_t>&, Rng&);
void baoab(Population&, const Settings&, const Benchmark&, Rng&,
           const Perturbation* noise = nullptr);
class Algorithm {
 public:
  virtual ~Algorithm() = default;
  virtual const double* precise_positions() const { return nullptr; }
  virtual bool finished() const { return false; }
  virtual Json metadata() const { return JsonReader(std::string("{}")).read(); }
  virtual Json resolved_config() const { return JsonReader(std::string("{}")).read(); }
  virtual uint64_t next_population_size() const { return population().n; }
  virtual void step() = 0;
  virtual const Population& population() const = 0;
  virtual uint64_t evaluations() const = 0;
  // Conservative admission check; never split or alter an optimizer's step.
  virtual uint64_t next_evaluations_upper_bound() const = 0;
  // Expose the direction-adjusted score on a common baseline, including any
  // cumulative reward held by the underlying algorithm.
  virtual double objective_score(int i) const = 0;
};
using Factory = std::function<std::unique_ptr<Algorithm>(Benchmark&, const Settings&)>;
void register_algorithm(const std::string& id, const std::string& name, bool velocity,
                        Factory factory, const Json& parameters = Json{});
std::string discovery_json();
class Session {
 public:
  explicit Session(const Json& config);
  Benchmark benchmark;
  Settings settings;
  std::unique_ptr<Algorithm> algorithm;
  uint64_t iteration = 0;
  double best = INFINITY;
  std::vector<double> snapshot;
  std::string config_json;
  std::string status_json() const;
  void step();
  void capture();
};
}  // namespace fg::optimization
