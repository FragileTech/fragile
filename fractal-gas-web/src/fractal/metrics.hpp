#pragma once
#include <cstdint>
namespace fg {
struct StepInfo {
  int32_t iteration = 0;
  int32_t num_cloned = 0;
  int32_t num_revived = 0;
  int32_t alive_count = 0;
  float mean_reward = 0.0f;
  float max_reward = 0.0f;
  float min_reward = 0.0f;
  float mean_virtual_reward = 0.0f;
  float max_virtual_reward = 0.0f;
  float min_virtual_reward = 0.0f;
  float mean_dt = 0.0f;
  int32_t min_dt = 0;
  int32_t max_dt = 0;
  int32_t best_walker_idx = 0;
  // Population bookkeeping. Wave: all three equal N. Graph: the live tree
  // size after the step, its leaf count (the mask that drove the step) and
  // the number of walkers that actually ran the environment.
  int32_t n_walkers = 0;
  int32_t n_leaves = 0;
  int32_t num_stepped = 0;
};
}  // namespace fg
