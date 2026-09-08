#include "fractal_gas.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "tensor_ops.hpp"

namespace fg {

FractalGas::FractalGas(BatchEnv& env, FractalGasParams params, std::unique_ptr<Rng> rng,
                       std::unique_ptr<FractalCloningOperator> clone_op,
                       std::unique_ptr<RandomActionOperator> kinetic_op)
    : env_(env),
      params_(params),
      clone_op_(clone_op ? std::move(clone_op) : std::make_unique<FractalCloningOperator>()),
      kinetic_op_(kinetic_op ? std::move(kinetic_op) : std::make_unique<RandomActionOperator>()),
      visits_(params.agg_block_size, params.erase_coef),
      backend_(env, visits_),
      action_policy_{env, *kinetic_op_},
      core_(backend_, action_policy_),
      state_(core_.current),
      exploration_tree_(core_.tree) {
  count_visits_ = params_.count_visits && env_.has_visit_key();
  owns_rng_ = !rng;
  rng_ = rng ? std::move(rng) : std::make_unique<Mt19937Rng>(params_.seed);
  clone_op_->dist_coef = params_.dist_coef;
  clone_op_->reward_coef = params_.reward_coef;
  clone_op_->use_cumulative_reward = params_.use_cumulative_reward;
  clone_op_->pool = env.worker_pool();

  kinetic_op_->dt_min = params_.dt_min;
  kinetic_op_->dt_max = params_.dt_max;
}

void FractalGas::reset() {
  std::vector<char> init_state;
  std::vector<float> init_obs;
  env_.reset(init_state, init_obs);

  if (owns_rng_) rng_ = std::make_unique<Mt19937Rng>(params_.seed);
  visits_.reset();
  total_steps_ = total_clones_ = total_frames_ = 0;
  iteration_count_ = 0;
  start_from(init_state, init_obs);
}

void FractalGas::start_from(const std::vector<char>& init_state,
                            const std::vector<float>& init_obs, const WalkerInfo* info) {
  const int32_t n = params_.N;
  const auto d = static_cast<int32_t>(init_obs.size());

  core_.reset(n, d, 1, env_.has_walker_info());
  state_.states.assign(n, init_state);
  for (int i = 0; i < n; ++i)
    std::copy(init_obs.begin(), init_obs.end(), state_.observations.begin() + size_t(i) * d);
  if (info) std::fill(state_.infos.begin(), state_.infos.end(), *info);
  visits_.set_erase_coef(params_.erase_coef);
  visits_.set_block_size(params_.agg_block_size);

  best_frame_.clear();
  core_.begin_history(params_.recording, params_.record_observations ? size_t(d) : 0,
                      init_obs.data(), std::vector<uint8_t>(init_state.begin(), init_state.end()));
}

StepInfo FractalGas::step() {
  const int32_t n = state_.N;
  backend_.count_visits = count_visits_;
  backend_.visit_reward = params_.visit_reward;
  backend_.visit_coef = params_.visit_coef;
  backend_.record_observations = params_.record_observations;
  const auto& m = core_.step(params_.n_elite, *clone_op_, *rng_);
  total_steps_ += n;
  total_clones_ += m.cloned;
  total_frames_ += m.frames;
  ++iteration_count_;
  StepInfo info;
  info.iteration = iteration_count_;
  info.num_cloned = m.cloned;
  info.num_revived = m.revived;
  info.alive_count = m.alive;
  info.mean_reward = m.mean_reward;
  info.max_reward = m.max_reward;
  info.min_reward = m.min_reward;
  info.mean_virtual_reward = m.mean_fitness;
  info.max_virtual_reward = m.max_fitness;
  info.min_virtual_reward = m.min_fitness;
  info.mean_dt = m.mean_dt;
  info.min_dt = m.min_dt;
  info.max_dt = m.max_dt;

  // 9. Record the best walker's frame. Display-only choice: when the env
  // provides a showcase score (e.g. Mario's (world, level, x)), use it so
  // the demo follows the furthest run in the highest level; otherwise
  // argmax cumulative reward (first max), as in the Python version.
  int32_t best_idx = 0;
  if (env_.has_display_score()) {
    float best_score = env_.display_score(0);
    for (int32_t i = 1; i < n; ++i) {
      const float score = env_.display_score(i);
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
  } else {
    for (int32_t i = 1; i < n; ++i) {
      if (state_.rewards[static_cast<size_t>(i)] > state_.rewards[static_cast<size_t>(best_idx)]) {
        best_idx = i;
      }
    }
  }
  info.best_walker_idx = best_idx;
  info.n_walkers = n;
  info.n_leaves = n;
  info.num_stepped = n;
  if (params_.record_frames) {
    env_.render_frame(state_.states[static_cast<size_t>(best_idx)], best_frame_);
  }

  return info;
}

std::vector<StepInfo> FractalGas::run(int32_t max_iterations, bool stop_when_all_dead) {
  reset();
  std::vector<StepInfo> history;
  history.reserve(static_cast<size_t>(max_iterations));
  for (int32_t it = 0; it < max_iterations; ++it) {
    history.push_back(step());
    if (stop_when_all_dead && state_.alive_count() == 0) break;
  }
  return history;
}

std::pair<int32_t, float> FractalGas::get_best_walker() const {
  int32_t best_idx = 0;
  for (int32_t i = 1; i < state_.N; ++i) {
    if (state_.rewards[static_cast<size_t>(i)] > state_.rewards[static_cast<size_t>(best_idx)]) {
      best_idx = i;
    }
  }
  return {best_idx, state_.rewards[static_cast<size_t>(best_idx)]};
}

}  // namespace fg
