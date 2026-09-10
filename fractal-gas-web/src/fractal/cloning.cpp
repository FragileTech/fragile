#include "fractal/cloning.hpp"

#include <cmath>

#include "fractal/tensor_ops.hpp"

namespace fg {

std::vector<float> FractalCloningOperator::fitness_with_companions(
    const std::vector<float>& observations, int32_t n, int32_t obs_dim,
    const std::vector<float>& cumulative_rewards, const std::vector<float>& step_rewards,
    const std::vector<int32_t>& companions) const {
  std::vector<float> out;
  fitness_with_companions_into(observations, n, obs_dim, cumulative_rewards, step_rewards,
                               companions, out);
  return out;
}
void FractalCloningOperator::fitness_with_companions_into(
    const std::vector<float>& observations, int32_t n, int32_t obs_dim,
    const std::vector<float>& cumulative_rewards, const std::vector<float>& step_rewards,
    const std::vector<int32_t>& companions, std::vector<float>& out) const {
  companion_distances_into(observations, companions, n, obs_dim, pool, distances_, distance_metric);
  if (diagnostics.enabled) {
    diagnostics.decisions.assign(n, {});
    for (int i = 0; i < n; ++i) {
      auto& d = diagnostics.decisions[i];
      d.slot = i; d.distance = distances_[i]; d.distance_companion = companions[i];
    }
  }
  asymmetric_rescale_into(distances_, distances_);
  asymmetric_rescale_into(use_cumulative_reward ? cumulative_rewards : step_rewards,
                          normalized_rewards_);
  if (diagnostics.enabled)
    for (int i = 0; i < n; ++i) {
      diagnostics.decisions[i].distance_norm = distances_[i];
      diagnostics.decisions[i].reward_norm = normalized_rewards_[i];
    }
  out.resize(n);
  for (int i = 0; i < n; ++i)
    out[i] = std::pow(distances_[i], dist_coef) * std::pow(normalized_rewards_[i], reward_coef);
}

std::vector<float> FractalCloningOperator::clone_probs_with_companions(
    const std::vector<float>& virtual_rewards, const std::vector<int32_t>& companions) const {
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

std::vector<int32_t> FractalCloningOperator::sample_companions(const std::vector<uint8_t>& alive,
                                                               Rng& rng) const {
  std::vector<int32_t> out;
  sample_companions_into(alive, rng, out);
  return out;
}

std::vector<float> FractalCloningOperator::sample_uniforms(int32_t n, Rng& rng) const {
  std::vector<float> out;
  sample_uniforms_into(n, rng, out);
  return out;
}

void FractalCloningOperator::sample_companions_into(const std::vector<uint8_t>& alive, Rng& rng,
                                                    std::vector<int32_t>& out) const {
  random_alive_compas_into(alive, rng, out, companion_scratch_);
}
void FractalCloningOperator::sample_uniforms_into(int32_t n, Rng& rng,
                                                  std::vector<float>& out) const {
  out.resize(n);
  for (auto& v : out) v = rng.uniform01();
}
void FractalCloningOperator::calculate_fitness_into(const std::vector<float>& observations,
                                                    int32_t n, int32_t obs_dim,
                                                    const std::vector<float>& cumulative_rewards,
                                                    const std::vector<float>& step_rewards,
                                                    const std::vector<uint8_t>& alive, Rng& rng,
                                                    std::vector<float>& out,
                                                    std::vector<int32_t>& companions) const {
  sample_companions_into(alive, rng, companions);
  fitness_with_companions_into(observations, n, obs_dim, cumulative_rewards, step_rewards,
                               companions, out);
}
void FractalCloningOperator::decide_cloning_into(const std::vector<float>& fitness,
                                                 const std::vector<uint8_t>& alive, Rng& rng,
                                                 std::vector<int32_t>& companions,
                                                 std::vector<uint8_t>& mask) const {
  sample_companions_into(alive, rng, companions);
  sample_uniforms_into(int32_t(alive.size()), rng, uniforms_);
  mask.resize(alive.size());
  for (size_t i = 0; i < alive.size(); ++i) {
    float p = (fitness[companions[i]] - fitness[i]) / (fitness[i] > eps ? fitness[i] : eps);
    mask[i] = p > uniforms_[i] || !alive[i];
    if (diagnostics.enabled && i < diagnostics.decisions.size()) {
      auto& d = diagnostics.decisions[i];
      d.fitness = fitness[i]; d.donor_fitness = fitness[companions[i]];
      d.clone_donor = companions[i]; d.clone_score = p; d.draw = uniforms_[i];
      d.alive = alive[i]; d.wanted = mask[i]; d.cloned = mask[i];
    }
  }
}
std::pair<std::vector<float>, std::vector<int32_t>> FractalCloningOperator::calculate_fitness(
    const std::vector<float>& observations, int32_t n, int32_t obs_dim,
    const std::vector<float>& cumulative_rewards, const std::vector<float>& step_rewards,
    const std::vector<uint8_t>& alive, Rng& rng) const {
  std::vector<int32_t> companions;
  std::vector<float> fitness;
  calculate_fitness_into(observations, n, obs_dim, cumulative_rewards, step_rewards, alive, rng,
                         fitness, companions);
  return {std::move(fitness), std::move(companions)};
}

std::pair<std::vector<int32_t>, std::vector<uint8_t>> FractalCloningOperator::decide_cloning(
    const std::vector<float>& virtual_rewards, const std::vector<uint8_t>& alive, Rng& rng) const {
  std::vector<int32_t> companions;
  std::vector<uint8_t> mask;
  decide_cloning_into(virtual_rewards, alive, rng, companions, mask);
  return {std::move(companions), std::move(mask)};
}
}  // namespace fg
