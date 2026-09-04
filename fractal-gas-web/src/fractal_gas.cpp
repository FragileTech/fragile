#include "fractal_gas.hpp"

#include "tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace fg {

namespace {

/// torch.topk(k, sorted=True) over concatenated rewards: indices of the k
/// largest values in descending order, ties resolved by lower index.
std::vector<int32_t> topk_indices(const std::vector<float>& values, int32_t k) {
  std::vector<int32_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);
  const auto kk = static_cast<size_t>(
      std::min<int64_t>(k, static_cast<int64_t>(values.size())));
  std::partial_sort(idx.begin(), idx.begin() + static_cast<long>(kk), idx.end(),
                    [&values](int32_t a, int32_t b) {
                      const float va = values[static_cast<size_t>(a)];
                      const float vb = values[static_cast<size_t>(b)];
                      if (va != vb) return va > vb;
                      return a < b;
                    });
  idx.resize(kk);
  return idx;
}

}  // namespace

FractalGas::FractalGas(BatchEnv& env, FractalGasParams params,
                       std::unique_ptr<Rng> rng,
                       std::unique_ptr<FractalCloningOperator> clone_op,
                       std::unique_ptr<RandomActionOperator> kinetic_op)
    : env_(env),
      params_(params),
      visits_(params.agg_block_size, params.erase_coef) {
  count_visits_ = params_.count_visits && env_.has_visit_key();
  rng_ = rng ? std::move(rng) : std::make_unique<Mt19937Rng>(params_.seed);
  if (clone_op) {
    clone_op_ = std::move(clone_op);
  } else {
    clone_op_ = std::make_unique<FractalCloningOperator>();
  }
  clone_op_->dist_coef = params_.dist_coef;
  clone_op_->reward_coef = params_.reward_coef;
  clone_op_->use_cumulative_reward = params_.use_cumulative_reward;
  clone_op_->pool = env.worker_pool();

  if (kinetic_op) {
    kinetic_op_ = std::move(kinetic_op);
  } else {
    kinetic_op_ = std::make_unique<RandomActionOperator>();
  }
  kinetic_op_->dt_min = params_.dt_min;
  kinetic_op_->dt_max = params_.dt_max;
}

void FractalGas::reset() {
  std::vector<char> init_state;
  std::vector<float> init_obs;
  env_.reset(init_state, init_obs);

  const int32_t n = params_.N;
  const auto d = static_cast<int32_t>(init_obs.size());

  state_ = WalkerState{};
  state_.N = n;
  state_.obs_dim = d;
  state_.states.assign(static_cast<size_t>(n), init_state);
  state_.observations.resize(static_cast<size_t>(n) * static_cast<size_t>(d));
  for (int32_t i = 0; i < n; ++i) {
    std::copy(init_obs.begin(), init_obs.end(),
              state_.observations.begin() +
                  static_cast<size_t>(i) * static_cast<size_t>(d));
  }
  state_.rewards.assign(static_cast<size_t>(n), 0.0f);
  state_.step_rewards.assign(static_cast<size_t>(n), 0.0f);
  state_.dones.assign(static_cast<size_t>(n), 0);
  state_.truncated.assign(static_cast<size_t>(n), 0);
  state_.actions.assign(static_cast<size_t>(n), 0);
  state_.dt.assign(static_cast<size_t>(n), 1);
  state_.has_virtual_rewards = false;
  state_.virtual_rewards.clear();
  state_.has_infos = false;
  state_.infos.clear();
  visits_.reset();
  visits_.set_erase_coef(params_.erase_coef);
  visits_.set_block_size(params_.agg_block_size);

  total_steps_ = 0;
  total_clones_ = 0;
  total_frames_ = 0;
  iteration_count_ = 0;
  has_elite_ = false;
  elite_walkers_ = WalkerState{};
  best_frame_.clear();
}

StepInfo FractalGas::step() {
  const int32_t n = state_.N;

  // 0. Inject elites into the first n_elite positions.
  if (params_.n_elite > 0 && has_elite_) {
    state_.inject(elite_walkers_, std::min(params_.n_elite, elite_walkers_.N));
  }

  const std::vector<uint8_t> alive = state_.alive_mask();

  // 1. Calculate fitness (virtual rewards) on the pre-clone state.
  auto fitness_result = clone_op_->calculate_fitness(
      state_.observations, n, state_.obs_dim, state_.rewards,
      state_.step_rewards, alive, *rng_);
  std::vector<float>& virtual_rewards = fitness_result.first;
  // Optional visit-count term (see FractalGasParams): relativize(-visits)
  // over all walkers, multiplied into the fitness like the tree does.
  if (count_visits_ && params_.visit_reward && state_.has_infos) {
    std::vector<VisitKey> keys(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      const WalkerInfo& wi = state_.infos[static_cast<size_t>(i)];
      keys[static_cast<size_t>(i)] = VisitKey{wi.visit_plane, wi.visit_x, wi.visit_y};
    }
    std::vector<float> sums;
    visits_.block_sums(keys, sums);
    for (float& s : sums) s = -s;
    const std::vector<float> other = asymmetric_rescale(sums);
    for (int32_t i = 0; i < n; ++i) {
      const double o = static_cast<double>(other[static_cast<size_t>(i)]);
      virtual_rewards[static_cast<size_t>(i)] *= static_cast<float>(
          params_.visit_coef == 1.0f ? o : std::pow(o, static_cast<double>(params_.visit_coef)));
    }
  }

  // 2. Decide cloning (second independent companion draw).
  auto decision = clone_op_->decide_cloning(virtual_rewards, alive, *rng_);
  const std::vector<int32_t>& clone_companions = decision.first;
  const std::vector<uint8_t>& will_clone = decision.second;

  // 3. Clone walker data (gather from the pre-clone arrays).
  WalkerState state_after_clone = state_.clone(clone_companions, will_clone);
  state_after_clone.virtual_rewards = virtual_rewards;
  state_after_clone.has_virtual_rewards = true;

  int32_t num_cloned = 0;
  for (const uint8_t w : will_clone) num_cloned += w;
  total_clones_ += num_cloned;

  // 4. Kinetic operator: fresh random actions + dt, env batch step.
  std::vector<std::vector<char>> new_states;
  std::vector<float> observations;
  std::vector<float> step_rewards;
  std::vector<uint8_t> dones;
  std::vector<uint8_t> truncated;
  kinetic_op_->apply(env_, state_after_clone.states, nullptr, *rng_, new_states,
                     observations, step_rewards, dones, truncated);

  // 4b. All-dead revive: when no walker survived the step, un-done the
  // recoverable ("soft") deaths — e.g. an Atari life loss with lives left —
  // so the swarm continues from the post-life-loss states. With any walker
  // still alive, soft-dead walkers stay dead and clone away as usual, so
  // losing a life is a death signal and reviving is the last resort.
  int32_t num_revived = 0;
  if (env_.has_recoverable_dones()) {
    bool any_alive = false;
    for (int32_t i = 0; i < n && !any_alive; ++i) {
      any_alive = !dones[static_cast<size_t>(i)] &&
                  !truncated[static_cast<size_t>(i)];
    }
    if (!any_alive) {
      for (int32_t i = 0; i < n; ++i) {
        const auto ui = static_cast<size_t>(i);
        if (dones[ui] && !truncated[ui] && env_.done_is_recoverable(i)) {
          dones[ui] = 0;
          ++num_revived;
        }
      }
    }
  }

  // 5-7. Build the new WalkerState with updated cumulative rewards.
  WalkerState new_state;
  new_state.N = n;
  new_state.obs_dim = state_.obs_dim;
  new_state.states = std::move(new_states);
  new_state.observations = std::move(observations);
  new_state.rewards.resize(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    new_state.rewards[ui] = state_after_clone.rewards[ui] + step_rewards[ui];
  }
  new_state.step_rewards = std::move(step_rewards);
  new_state.dones = std::move(dones);
  new_state.truncated = std::move(truncated);
  new_state.actions = kinetic_op_->last_actions;
  new_state.dt = kinetic_op_->last_dt;
  new_state.virtual_rewards = virtual_rewards;
  new_state.has_virtual_rewards = true;
  // Per-walker info from the env (batch index == walker index here), kept
  // in the state so it follows the walker through cloning and elites.
  if (env_.has_walker_info()) {
    new_state.infos.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) new_state.infos[static_cast<size_t>(i)] = env_.walker_info(i);
    new_state.has_infos = true;
    if (count_visits_) {
      std::vector<VisitKey> keys(static_cast<size_t>(n));
      for (int32_t i = 0; i < n; ++i) {
        const WalkerInfo& wi = new_state.infos[static_cast<size_t>(i)];
        keys[static_cast<size_t>(i)] = VisitKey{wi.visit_plane, wi.visit_x, wi.visit_y};
      }
      visits_.update(keys);
    }
  }

  state_ = std::move(new_state);

  total_steps_ += n;
  for (int32_t i = 0; i < n; ++i) {
    const int32_t f = env_.frames_stepped(i);
    total_frames_ += f >= 0 ? f : state_.dt[static_cast<size_t>(i)];
  }
  ++iteration_count_;

  // Collect info.
  StepInfo info;
  info.iteration = iteration_count_;
  info.num_cloned = num_cloned;
  info.num_revived = num_revived;
  info.alive_count = state_.alive_count();

  double reward_sum = 0.0;
  float reward_max = -std::numeric_limits<float>::infinity();
  float reward_min = std::numeric_limits<float>::infinity();
  for (const float r : state_.rewards) {
    reward_sum += r;
    reward_max = std::max(reward_max, r);
    reward_min = std::min(reward_min, r);
  }
  info.mean_reward = static_cast<float>(reward_sum / n);
  info.max_reward = reward_max;
  info.min_reward = reward_min;

  double vr_sum = 0.0;
  float vr_max = -std::numeric_limits<float>::infinity();
  float vr_min = std::numeric_limits<float>::infinity();
  for (const float v : virtual_rewards) {
    vr_sum += v;
    vr_max = std::max(vr_max, v);
    vr_min = std::min(vr_min, v);
  }
  info.mean_virtual_reward = static_cast<float>(vr_sum / n);
  info.max_virtual_reward = vr_max;
  info.min_virtual_reward = vr_min;

  int64_t dt_sum = 0;
  int32_t dt_min = std::numeric_limits<int32_t>::max();
  int32_t dt_max = std::numeric_limits<int32_t>::min();
  for (const int32_t d : state_.dt) {
    dt_sum += d;
    dt_min = std::min(dt_min, d);
    dt_max = std::max(dt_max, d);
  }
  info.mean_dt = static_cast<float>(static_cast<double>(dt_sum) / n);
  info.min_dt = dt_min;
  info.max_dt = dt_max;

  // 8. Update the elite buffer; fold its max into max_reward.
  if (params_.n_elite > 0) {
    update_elites();
    float elite_max = -std::numeric_limits<float>::infinity();
    for (const float r : elite_walkers_.rewards) elite_max = std::max(elite_max, r);
    info.max_reward = std::max(elite_max, info.max_reward);
  }

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
      if (state_.rewards[static_cast<size_t>(i)] >
          state_.rewards[static_cast<size_t>(best_idx)]) {
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

void FractalGas::update_elites() {
  const int32_t k = params_.n_elite;

  if (!has_elite_) {
    // First call: take the top k from the current population.
    const std::vector<int32_t> top = topk_indices(state_.rewards, k);
    elite_walkers_ = WalkerState::extract(state_, top);
    has_elite_ = true;
    return;
  }

  // Concatenate elite and current rewards, pick the global top k.
  std::vector<float> all_rewards = elite_walkers_.rewards;
  all_rewards.insert(all_rewards.end(), state_.rewards.begin(),
                     state_.rewards.end());
  const std::vector<int32_t> top = topk_indices(all_rewards, k);

  const int32_t n_elite_current = elite_walkers_.N;
  std::vector<int32_t> elite_idx;
  std::vector<int32_t> state_idx;
  for (const int32_t t : top) {
    if (t < n_elite_current) {
      elite_idx.push_back(t);
    } else {
      state_idx.push_back(t - n_elite_current);
    }
  }

  if (state_idx.empty()) {
    elite_walkers_ = WalkerState::extract(elite_walkers_, elite_idx);
  } else if (elite_idx.empty()) {
    elite_walkers_ = WalkerState::extract(state_, state_idx);
  } else {
    elite_walkers_ =
        WalkerState::concat(WalkerState::extract(elite_walkers_, elite_idx),
                            WalkerState::extract(state_, state_idx));
  }
}

std::vector<StepInfo> FractalGas::run(int32_t max_iterations,
                                      bool stop_when_all_dead) {
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
    if (state_.rewards[static_cast<size_t>(i)] >
        state_.rewards[static_cast<size_t>(best_idx)]) {
      best_idx = i;
    }
  }
  return {best_idx, state_.rewards[static_cast<size_t>(best_idx)]};
}

}  // namespace fg
