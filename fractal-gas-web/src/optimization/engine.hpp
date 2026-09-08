#pragma once
#include <functional>
#include <memory>

#include "optimization/benchmark.hpp"
#include "swarm_algorithm.hpp"

namespace fg::optimization {
struct Settings {
  explicit Settings(const Json& input);
  Json json;
  std::string algorithm, companion, clone_companion;
  int walkers, max_walkers, seed, dt_min, dt_max, clone_every, substeps, elites;
  double proposal, gamma, beta, delta_t, epsilon, clone_epsilon, lambda_alg,
      reward_coef, distance_coef, eta, sigma_min, amplitude, epsilon_dist, rho,
      p_max, epsilon_clone, sigma_x, restitution;
  bool periodic, potential_force, cloning, kinetic;
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
void baoab(Population&, const Settings&, const Benchmark&, Rng&);
class Algorithm {
 public:
  virtual ~Algorithm() = default;
  virtual void step() = 0;
  virtual const Population& population() const = 0;
  virtual uint64_t evaluations() const = 0;
  virtual double objective_score(int i) const {
    return -population().objective.at(i);
  }
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
