#pragma once
#include <functional>
#include <memory>

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
  double proposal, gamma, beta, delta_t, epsilon, clone_epsilon, lambda_alg,
      reward_coef, distance_coef, eta, sigma_min, amplitude, epsilon_dist, rho,
      p_max, epsilon_clone, sigma_x, restitution;
  bool periodic, potential_force, cloning, kinetic, consensus_prefix;
  double score(double value) const {
    return objective == "minimize" ? -value : value;
  }
  bool better(double a, double b) const { return score(a) > score(b); }
  double worst() const {
    return objective == "minimize" ? INFINITY : -INFINITY;
  }
  bool planning() const {
    return algorithm == "fmc" || algorithm == "wave_jump";
  }
};
struct Population {
  int n = 0, d = 0;
  bool has_velocity = false;
  std::vector<float> x, v, fitness;
  std::vector<double> objective;
  std::vector<uint8_t> alive, cloned, leaf;
  std::vector<int32_t> companions, clone_companions, parent;
  void resize(int count, int dimensions);
};
// Pure operators share the Python reference's semantics and accept injected
// RNG.
std::vector<int32_t> select_companions(const Population&, const Settings&,
                                       const Benchmark&, const std::string&,
                                       double, Rng&);
std::vector<float> fitness(const Population&, const Settings&, const Benchmark&,
                           const std::vector<int32_t>&);
void clone_population(Population&, const Settings&, const std::vector<int32_t>&,
                      const std::vector<uint8_t>&, Rng&);
void baoab(Population&, const Settings&, const Benchmark&, Rng&,
           const Perturbation* noise = nullptr);
class Algorithm {
 public:
  virtual ~Algorithm() = default;
  virtual void step() = 0;
  virtual const Population& population() const = 0;
  virtual uint64_t evaluations() const = 0;
  // Expose the direction-adjusted score on a common baseline, including any
  // cumulative reward held by the underlying algorithm.
  virtual double objective_score(int i) const = 0;
};
using Factory =
    std::function<std::unique_ptr<Algorithm>(Benchmark&, const Settings&)>;
void register_algorithm(const std::string& id, const std::string& name,
                        bool velocity, Factory factory,
                        const Json& parameters = Json{});
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
  void step();
  void capture();
};
}  // namespace fg::optimization
