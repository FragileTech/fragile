#include "cloning.hpp"

#include <cmath>

#include "tensor_ops.hpp"

namespace fg {

std::vector<float> FractalCloningOperator::fitness_with_companions(
    const std::vector<float>& observations, int32_t n, int32_t obs_dim,
    const std::vector<float>& cumulative_rewards,
    const std::vector<float>& step_rewards,
    const std::vector<int32_t>& companions) const {
  const std::vector<float> distances =
      l2_norm_companions(observations, companions, n, obs_dim, pool);
  const std::vector<float>& reward_signal =
      use_cumulative_reward ? cumulative_rewards : step_rewards;

  const std::vector<float> distance_norm = asymmetric_rescale(distances);
  const std::vector<float> reward_norm = asymmetric_rescale(reward_signal);

  std::vector<float> virtual_rewards(static_cast<size_t>(n));
  for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
    virtual_rewards[i] = std::pow(distance_norm[i], dist_coef) *
                         std::pow(reward_norm[i], reward_coef);
  }
  return virtual_rewards;
}

std::vector<float> FractalCloningOperator::clone_probs_with_companions(
    const std::vector<float>& virtual_rewards,
    const std::vector<int32_t>& companions) const {
  const size_t n = virtual_rewards.size();
  std::vector<float> probs(n);
  for (size_t i = 0; i < n; ++i) {
    const float vr = virtual_rewards[i];
    const float vr_comp = virtual_rewards[static_cast<size_t>(companions[i])];
    const float denom = vr > eps ? vr : eps;
    probs[i] = (vr_comp - vr) / denom;
  }
  return probs;
}

std::vector<uint8_t> FractalCloningOperator::decide_with_uniforms(
    const std::vector<float>& clone_probs, const std::vector<float>& uniforms,
    const std::vector<uint8_t>& alive) const {
  const size_t n = clone_probs.size();
  std::vector<uint8_t> will_clone(n);
  for (size_t i = 0; i < n; ++i) {
    const bool clones = clone_probs[i] > uniforms[i];
    will_clone[i] = (clones || !alive[i]) ? 1 : 0;  // dead walkers always clone
  }
  return will_clone;
}

std::vector<int32_t> FractalCloningOperator::sample_companions(
    const std::vector<uint8_t>& alive, Rng& rng) const {
  return random_alive_compas(alive, rng);
}

std::vector<float> FractalCloningOperator::sample_uniforms(int32_t n,
                                                           Rng& rng) const {
  std::vector<float> u(static_cast<size_t>(n));
  for (auto& v : u) v = rng.uniform01();
  return u;
}

std::pair<std::vector<float>, std::vector<int32_t>>
FractalCloningOperator::calculate_fitness(
    const std::vector<float>& observations, int32_t n, int32_t obs_dim,
    const std::vector<float>& cumulative_rewards,
    const std::vector<float>& step_rewards, const std::vector<uint8_t>& alive,
    Rng& rng) const {
  std::vector<int32_t> companions = sample_companions(alive, rng);
  std::vector<float> virtual_rewards = fitness_with_companions(
      observations, n, obs_dim, cumulative_rewards, step_rewards, companions);
  return {std::move(virtual_rewards), std::move(companions)};
}

std::pair<std::vector<int32_t>, std::vector<uint8_t>>
FractalCloningOperator::decide_cloning(
    const std::vector<float>& virtual_rewards,
    const std::vector<uint8_t>& alive, Rng& rng) const {
  const auto n = static_cast<int32_t>(virtual_rewards.size());
  std::vector<int32_t> companions = sample_companions(alive, rng);
  const std::vector<float> probs =
      clone_probs_with_companions(virtual_rewards, companions);
  const std::vector<float> uniforms = sample_uniforms(n, rng);
  std::vector<uint8_t> will_clone = decide_with_uniforms(probs, uniforms, alive);
  return {std::move(companions), std::move(will_clone)};
}

}  // namespace fg
