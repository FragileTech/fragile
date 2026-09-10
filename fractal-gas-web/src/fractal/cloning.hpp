// Translation of FractalCloningOperator
// (src/fragile/fractalai/videogames/cloning.py).
//
// The math is split into pure layers (given companions / given uniforms) so
// tests can replay companion indices and uniform draws recorded from the
// Python implementation, plus rng-driven wrappers used in production. The
// rng-facing sample methods are virtual so a full FractalGas step can be
// replayed by overriding them.
#ifndef FRACTAL_GAS_CLONING_HPP
#define FRACTAL_GAS_CLONING_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include "fractal/rng.hpp"
#include "fractal/diagnostics.hpp"
#include "fractal/tensor_ops.hpp"

namespace fg {

class ThreadPool;

class FractalCloningOperator {
 public:
  mutable CloneDiagnostics diagnostics;
  DistanceMetric distance_metric = DistanceMetric::L2;
  float dist_coef = 1.0f;
  float reward_coef = 1.0f;
  bool use_cumulative_reward = false;
  float eps = 1e-8f;
  /// Borrowed (not owned): the env's worker pool, used to parallelize the
  /// distance computation in fitness_with_companions. nullptr -> serial.
  ThreadPool* pool = nullptr;

  virtual ~FractalCloningOperator() = default;

  // -- pure layers (deterministic given the random inputs) --------------------

  /// Fitness math with companions given:
  ///   distances = l2(obs, obs[companions])
  ///   signal    = cumulative if use_cumulative_reward else step rewards
  ///   vr        = asym(distances)^dist_coef * asym(signal)^reward_coef
  std::vector<float> fitness_with_companions(const std::vector<float>& observations, int32_t n,
                                             int32_t obs_dim,
                                             const std::vector<float>& cumulative_rewards,
                                             const std::vector<float>& step_rewards,
                                             const std::vector<int32_t>& companions) const;

  /// Clone probabilities with companions given:
  ///   p_i = (vr[comp_i] - vr_i) / (vr_i > eps ? vr_i : eps)
  std::vector<float> clone_probs_with_companions(const std::vector<float>& virtual_rewards,
                                                 const std::vector<int32_t>& companions) const;

  /// Decision with uniforms given: will_clone = (p > u) | dead.
  std::vector<uint8_t> decide_with_uniforms(const std::vector<float>& clone_probs,
                                            const std::vector<float>& uniforms,
                                            const std::vector<uint8_t>& alive) const;

  // -- rng-facing sampling (virtual for replay tests) -------------------------

  virtual std::vector<int32_t> sample_companions(const std::vector<uint8_t>& alive,
                                                 Rng& rng) const;
  virtual std::vector<float> sample_uniforms(int32_t n, Rng& rng) const;

  virtual void sample_companions_into(const std::vector<uint8_t>& alive, Rng& rng,
                                      std::vector<int32_t>& out) const;
  virtual void sample_uniforms_into(int32_t n, Rng& rng, std::vector<float>& out) const;
  void fitness_with_companions_into(const std::vector<float>& observations, int32_t n,
                                    int32_t obs_dim, const std::vector<float>& cumulative_rewards,
                                    const std::vector<float>& step_rewards,
                                    const std::vector<int32_t>& companions,
                                    std::vector<float>& out) const;
  void calculate_fitness_into(const std::vector<float>& observations, int32_t n, int32_t obs_dim,
                              const std::vector<float>& cumulative_rewards,
                              const std::vector<float>& step_rewards,
                              const std::vector<uint8_t>& alive, Rng& rng, std::vector<float>& out,
                              std::vector<int32_t>& companions) const;
  void decide_cloning_into(const std::vector<float>& fitness, const std::vector<uint8_t>& alive,
                           Rng& rng, std::vector<int32_t>& companions,
                           std::vector<uint8_t>& mask) const;
  // -- production API matching the Python methods -----------------------------

  /// calculate_fitness(): one companion draw, then the fitness math.
  /// Returns (virtual_rewards, companions).
  std::pair<std::vector<float>, std::vector<int32_t>> calculate_fitness(
      const std::vector<float>& observations, int32_t n, int32_t obs_dim,
      const std::vector<float>& cumulative_rewards, const std::vector<float>& step_rewards,
      const std::vector<uint8_t>& alive, Rng& rng) const;

  /// decide_cloning(): a SECOND independent companion draw, clone probs,
  /// stochastic decision, dead walkers always clone.
  /// Returns (companions, will_clone).
  std::pair<std::vector<int32_t>, std::vector<uint8_t>> decide_cloning(
      const std::vector<float>& virtual_rewards, const std::vector<uint8_t>& alive,
      Rng& rng) const;

 private:
  mutable CompanionScratch companion_scratch_;
  mutable std::vector<float> distances_, normalized_rewards_, uniforms_;
};

}  // namespace fg

#endif  // FRACTAL_GAS_CLONING_HPP
