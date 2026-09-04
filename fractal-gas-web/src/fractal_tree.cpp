#include "fractal_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensor_ops.hpp"
#include "thread_pool.hpp"

namespace fg {

// ---- sampler ---------------------------------------------------------------

std::vector<int32_t> FractalTreeSampler::sample_companions(
    const std::vector<uint8_t>& alive, Rng& rng) const {
  return random_alive_compas(alive, rng);
}

std::vector<float> FractalTreeSampler::sample_uniforms(int32_t n,
                                                       Rng& rng) const {
  std::vector<float> u(static_cast<size_t>(n));
  for (auto& v : u) v = rng.uniform01();
  return u;
}

std::vector<int32_t> FractalTreeSampler::sample_actions(int32_t k,
                                                        int32_t n_actions,
                                                        Rng& rng) const {
  std::vector<int32_t> actions(static_cast<size_t>(k));
  for (auto& a : actions) a = static_cast<int32_t>(rng.randint(0, n_actions));
  return actions;
}

std::vector<int32_t> FractalTreeSampler::sample_dt(int32_t k, int32_t dt_min,
                                                   int32_t dt_max,
                                                   Rng& rng) const {
  // np.random.randint(dt_min, dt_max + 1): inclusive upper bound here.
  std::vector<int32_t> dt(static_cast<size_t>(k));
  for (auto& d : dt) d = static_cast<int32_t>(rng.randint(dt_min, dt_max + 1));
  return dt;
}

// ---- state -----------------------------------------------------------------

std::vector<uint8_t> TreeState::alive_mask() const {
  std::vector<uint8_t> alive(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    alive[static_cast<size_t>(i)] = oobs[static_cast<size_t>(i)] ? 0 : 1;
  }
  return alive;
}

int32_t TreeState::alive_count() const {
  int32_t count = 0;
  for (int32_t i = 0; i < n; ++i) {
    if (!oobs[static_cast<size_t>(i)]) ++count;
  }
  return count;
}

// ---- tree ------------------------------------------------------------------

FractalTree::FractalTree(BatchEnv& env, FractalTreeParams params,
                         std::unique_ptr<Rng> rng,
                         std::unique_ptr<FractalTreeSampler> sampler)
    : env_(env),
      params_(params),
      visits_(params.agg_block_size, params.erase_coef) {
  if (params_.start_walkers < 1) params_.start_walkers = 1;
  if (params_.min_leafs < 1) params_.min_leafs = 1;
  if (params_.max_walkers < params_.start_walkers) {
    params_.max_walkers = params_.start_walkers;
  }
  rng_ = rng ? std::move(rng) : std::make_unique<Mt19937Rng>(params_.seed);
  sampler_ = sampler ? std::move(sampler)
                     : std::make_unique<FractalTreeSampler>();
  count_visits_ = params_.count_visits && env_.has_visit_key();
}

void FractalTree::grow(int32_t count) {
  // Fresh slots hold the reset values of the preallocated reference buffers:
  // parent 0 (the root), oobs True, zero rewards, zero observation, no
  // state, leaf.
  const auto d = static_cast<size_t>(state_.obs_dim);
  for (int32_t c = 0; c < count; ++c) {
    state_.states.emplace_back();
    state_.observations.insert(state_.observations.end(), d, 0.0f);
    state_.rewards.push_back(0.0f);
    state_.cum_rewards.push_back(0.0f);
    state_.oobs.push_back(1);
    state_.parent.push_back(0);
    state_.is_leaf.push_back(1);
    state_.actions.push_back(0);
    state_.dt.push_back(0);
    state_.virtual_rewards.push_back(0.0f);
    state_.other_rewards.push_back(1.0f);
    state_.distances.push_back(0.0f);
    state_.clone_probs.push_back(0.0f);
    state_.distance_ix.push_back(0);
    state_.clone_ix.push_back(0);
    state_.wants_clone.push_back(0);
    state_.is_cloned.push_back(0);
    state_.will_clone.push_back(0);
    state_.info.emplace_back();
  }
  state_.n += count;
}

void FractalTree::copy_info_from_env(const std::vector<int32_t>& batch_walkers) {
  if (!env_.has_walker_info()) return;
  for (size_t j = 0; j < batch_walkers.size(); ++j) {
    state_.info[static_cast<size_t>(batch_walkers[j])] =
        env_.walker_info(static_cast<int32_t>(j));
  }
}

void FractalTree::collect_visit_keys(const std::vector<int32_t>& walkers,
                                     std::vector<VisitKey>& keys) const {
  keys.resize(walkers.size());
  for (size_t j = 0; j < walkers.size(); ++j) {
    const WalkerInfo& wi = state_.info[static_cast<size_t>(walkers[j])];
    keys[j] = VisitKey{wi.visit_plane, wi.visit_x, wi.visit_y};
  }
}

void FractalTree::reset() {
  state_ = TreeState{};
  total_steps_ = 0;
  total_clones_ = 0;
  iteration_ = 0;
  best_frame_.clear();
  visits_.reset();
  visits_.set_erase_coef(params_.erase_coef);
  visits_.set_block_size(params_.agg_block_size);

  const int32_t n = std::min(params_.start_walkers, params_.max_walkers);
  // reset(): start_action = sample_actions(start_walkers) BEFORE the env
  // reset (core.py:814-817).
  const std::vector<int32_t> actions =
      sampler_->sample_actions(n, env_.n_actions(), *rng_);

  std::vector<char> init_state;
  std::vector<float> init_obs;
  env_.reset(init_state, init_obs);
  state_.obs_dim = static_cast<int32_t>(init_obs.size());
  grow(n);
  for (int32_t i = 0; i < n; ++i) {
    state_.states[static_cast<size_t>(i)] = init_state;
    state_.actions[static_cast<size_t>(i)] = actions[static_cast<size_t>(i)];
  }

  // step_env(): dt sampled, then every walker steps once from the reset
  // state.
  const std::vector<int32_t> dt =
      sampler_->sample_dt(n, params_.dt_min, params_.dt_max, *rng_);
  std::vector<std::vector<char>> batch_states(state_.states.begin(),
                                              state_.states.begin() + n);
  std::vector<std::vector<char>> new_states(static_cast<size_t>(n));
  std::vector<float> observations(static_cast<size_t>(n) *
                                  static_cast<size_t>(state_.obs_dim));
  std::vector<float> rewards(static_cast<size_t>(n), 0.0f);
  std::vector<uint8_t> dones(static_cast<size_t>(n), 0);
  std::vector<uint8_t> truncated(static_cast<size_t>(n), 0);
  env_.step_batch(batch_states, actions, dt, new_states, observations, rewards,
                  dones, truncated);

  // The root (walker 0) keeps the reset STATE and stays oobs (reset_tensors
  // sets oobs = ones and only rows 1.. are overwritten), but takes the
  // stepped observation like every other row (core.py:826-834).
  const auto d = static_cast<size_t>(state_.obs_dim);
  std::copy(observations.begin(), observations.end(),
            state_.observations.begin());
  for (int32_t i = 1; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    state_.rewards[ui] = rewards[ui];
    state_.cum_rewards[ui] = rewards[ui];
    state_.oobs[ui] = dones[ui] ? 1 : 0;
    state_.states[ui] = std::move(new_states[ui]);
  }
  for (int32_t i = 0; i < n; ++i) {
    state_.dt[static_cast<size_t>(i)] = dt[static_cast<size_t>(i)];
  }
  (void)d;

  std::vector<int32_t> all(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) all[static_cast<size_t>(i)] = i;
  copy_info_from_env(all);
  if (count_visits_) {
    // MontezumaTree.reset(): update_visits(observ) over every walker.
    std::vector<VisitKey> keys;
    collect_visit_keys(all, keys);
    visits_.update(keys);
  }
}

int32_t FractalTree::best_index() const {
  // torch.argmax: first maximum.
  int32_t best = 0;
  for (int32_t i = 1; i < state_.n; ++i) {
    if (state_.cum_rewards[static_cast<size_t>(i)] >
        state_.cum_rewards[static_cast<size_t>(best)]) {
      best = i;
    }
  }
  return best;
}

std::pair<int32_t, float> FractalTree::get_best_walker() const {
  if (state_.n == 0) return {0, 0.0f};
  const int32_t best = best_index();
  return {best, state_.cum_rewards[static_cast<size_t>(best)]};
}

StepInfo FractalTree::step() {
  const int32_t n = state_.n;
  const auto d = static_cast<size_t>(state_.obs_dim);
  const std::vector<uint8_t> alive = state_.alive_mask();

  // ---- 1. calculate_virtual_reward (uses the PREVIOUS leaf mask) ----------
  const std::vector<int32_t> compas1 = sampler_->sample_companions(alive, *rng_);
  const std::vector<float> distances = l2_norm_companions(
      state_.observations, compas1, n, state_.obs_dim, env_.worker_pool());
  const std::vector<float> distance_norm = asymmetric_rescale(distances);
  const auto reward_stats = mean_std_masked(state_.cum_rewards, state_.is_leaf);
  const std::vector<float> rewards_norm = relativize_with_stats(
      state_.cum_rewards, reward_stats.first, reward_stats.second);

  std::vector<float> other(static_cast<size_t>(n), 1.0f);
  if (count_visits_) {
    // MontezumaTree.calculate_other_reward(): minus the 5x5 block visit
    // count of every walker's cell, relativized with leaf-only statistics.
    std::vector<int32_t> all(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) all[static_cast<size_t>(i)] = i;
    std::vector<VisitKey> keys;
    collect_visit_keys(all, keys);
    std::vector<float> sums;
    visits_.block_sums(keys, sums);
    std::vector<float> visits_val(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      visits_val[static_cast<size_t>(i)] = -sums[static_cast<size_t>(i)];
    }
    const auto stats = mean_std_masked(visits_val, state_.is_leaf);
    other = relativize_with_stats(visits_val, stats.first, stats.second);
  }

  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    const double vr =
        std::pow(static_cast<double>(distance_norm[ui]),
                 static_cast<double>(params_.dist_coef)) *
        std::pow(static_cast<double>(rewards_norm[ui]),
                 static_cast<double>(params_.reward_coef)) *
        static_cast<double>(other[ui]);
    state_.virtual_rewards[ui] = static_cast<float>(vr);
    state_.distances[ui] = distances[ui];
    state_.distance_ix[ui] = compas1[ui];
    state_.other_rewards[ui] = other[ui];
  }

  // ---- 2. is_leaf = get_is_leaf(parent) ------------------------------------
  std::fill(state_.is_leaf.begin(), state_.is_leaf.end(), 1);
  for (int32_t i = 0; i < n; ++i) {
    state_.is_leaf[static_cast<size_t>(state_.parent[static_cast<size_t>(i)])] = 0;
  }

  // ---- 3. calculate_clone --------------------------------------------------
  const std::vector<int32_t> compas2 = sampler_->sample_companions(alive, *rng_);
  const float eps = params_.eps;
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    const float vr = state_.virtual_rewards[ui];
    const float vr_c = state_.virtual_rewards[static_cast<size_t>(compas2[ui])];
    state_.clone_probs[ui] = (vr_c - vr) / (vr > eps ? vr : eps);
    state_.clone_ix[ui] = compas2[ui];
  }
  const std::vector<float> uniforms = sampler_->sample_uniforms(n, *rng_);
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    state_.wants_clone[ui] = state_.clone_probs[ui] > uniforms[ui] ? 1 : 0;
  }

  // ---- 4. is_cloned, dead forcing, will_clone ------------------------------
  std::fill(state_.is_cloned.begin(), state_.is_cloned.end(), 0);
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    if (state_.wants_clone[ui]) {
      state_.is_cloned[static_cast<size_t>(compas2[ui])] = 1;
    }
  }
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    if (state_.oobs[ui]) state_.wants_clone[ui] = 1;
    state_.will_clone[ui] =
        (state_.wants_clone[ui] && state_.is_leaf[ui] && !state_.is_cloned[ui])
            ? 1
            : 0;
  }
  int32_t leaves = 0;
  for (int32_t i = 0; i < n; ++i) leaves += state_.is_leaf[static_cast<size_t>(i)];
  bool any = false;
  for (int32_t i = 0; i < n && !any; ++i) any = state_.will_clone[static_cast<size_t>(i)] != 0;
  if (!any) {
    ++iteration_;
    return collect_info(0, leaves);
  }

  // ---- 5. clone_data --------------------------------------------------------
  state_.will_clone[static_cast<size_t>(best_index())] = 0;
  std::vector<int32_t> cloning;
  for (int32_t i = 0; i < n; ++i) {
    if (state_.will_clone[static_cast<size_t>(i)]) cloning.push_back(i);
  }
  // clone_tensor: x[will_clone] = x[compas][will_clone] — the right-hand
  // side is read from the pre-clone arrays (gather), so copy the sources
  // first.
  std::vector<std::vector<char>> src_states(cloning.size());
  std::vector<float> src_obs(cloning.size() * d);
  std::vector<float> src_reward(cloning.size());
  std::vector<float> src_cum(cloning.size());
  std::vector<WalkerInfo> src_info(cloning.size());
  for (size_t j = 0; j < cloning.size(); ++j) {
    const auto c = static_cast<size_t>(compas2[static_cast<size_t>(cloning[j])]);
    src_states[j] = state_.states[c];
    std::copy(state_.observations.begin() + static_cast<long>(c * d),
              state_.observations.begin() + static_cast<long>((c + 1) * d),
              src_obs.begin() + static_cast<long>(j * d));
    src_reward[j] = state_.rewards[c];
    src_cum[j] = state_.cum_rewards[c];
    src_info[j] = state_.info[c];
  }
  for (size_t j = 0; j < cloning.size(); ++j) {
    const auto ui = static_cast<size_t>(cloning[j]);
    state_.states[ui] = std::move(src_states[j]);
    std::copy(src_obs.begin() + static_cast<long>(j * d),
              src_obs.begin() + static_cast<long>((j + 1) * d),
              state_.observations.begin() + static_cast<long>(ui * d));
    state_.rewards[ui] = src_reward[j];
    state_.cum_rewards[ui] = src_cum[j];
    state_.info[ui] = src_info[j];
    // The line that turns the wave into a tree.
    state_.parent[ui] = compas2[ui];
  }

  // Deviation: a walker that gathered an EMPTY state (every walker dead and
  // companions fell back to arange) cannot step; drop it from the batch.
  std::vector<int32_t> stepping;
  stepping.reserve(cloning.size());
  for (const int32_t i : cloning) {
    if (state_.states[static_cast<size_t>(i)].empty()) {
      state_.will_clone[static_cast<size_t>(i)] = 0;
    } else {
      stepping.push_back(i);
    }
  }
  const auto k = static_cast<int32_t>(stepping.size());
  total_clones_ += k;

  // ---- 6. step_walkers ------------------------------------------------------
  if (k > 0) {
    const std::vector<int32_t> actions =
        sampler_->sample_actions(k, env_.n_actions(), *rng_);
    const std::vector<int32_t> dt =
        sampler_->sample_dt(k, params_.dt_min, params_.dt_max, *rng_);
    std::vector<std::vector<char>> batch_states(static_cast<size_t>(k));
    for (int32_t j = 0; j < k; ++j) {
      batch_states[static_cast<size_t>(j)] =
          state_.states[static_cast<size_t>(stepping[static_cast<size_t>(j)])];
    }
    std::vector<std::vector<char>> new_states(static_cast<size_t>(k));
    std::vector<float> observations(static_cast<size_t>(k) * d);
    std::vector<float> rewards(static_cast<size_t>(k), 0.0f);
    std::vector<uint8_t> dones(static_cast<size_t>(k), 0);
    std::vector<uint8_t> truncated(static_cast<size_t>(k), 0);
    env_.step_batch(batch_states, actions, dt, new_states, observations,
                    rewards, dones, truncated);
    for (int32_t j = 0; j < k; ++j) {
      const auto uj = static_cast<size_t>(j);
      const auto ui = static_cast<size_t>(stepping[uj]);
      std::copy(observations.begin() + static_cast<long>(uj * d),
                observations.begin() + static_cast<long>((uj + 1) * d),
                state_.observations.begin() + static_cast<long>(ui * d));
      state_.rewards[ui] = rewards[uj];
      state_.cum_rewards[ui] += rewards[uj];
      state_.oobs[ui] = dones[uj] ? 1 : 0;
      state_.states[ui] = std::move(new_states[uj]);
      state_.actions[ui] = actions[uj];
      state_.dt[ui] = dt[uj];
    }
    copy_info_from_env(stepping);
    if (count_visits_) {
      std::vector<VisitKey> keys;
      collect_visit_keys(stepping, keys);
      visits_.update(keys);
    }
  }

  // ---- 7. growth --------------------------------------------------------------
  const int32_t missing = params_.min_leafs - leaves;
  if (missing > 0) {
    const int32_t room = params_.max_walkers - state_.n;
    if (room > 0) grow(std::min(missing, room));
  }

  // ---- 8. bookkeeping ---------------------------------------------------------
  total_steps_ += k;
  ++iteration_;
  return collect_info(k, leaves);
}

StepInfo FractalTree::collect_info(int32_t k, int32_t leaves) {
  const int32_t n = state_.n;
  StepInfo info;
  info.iteration = iteration_;
  info.num_cloned = k;
  info.num_stepped = k;
  info.num_revived = 0;
  info.alive_count = state_.alive_count();
  info.n_walkers = n;
  info.n_leaves = leaves;

  double reward_sum = 0.0;
  float reward_max = -std::numeric_limits<float>::infinity();
  float reward_min = std::numeric_limits<float>::infinity();
  double vr_sum = 0.0;
  float vr_max = -std::numeric_limits<float>::infinity();
  float vr_min = std::numeric_limits<float>::infinity();
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    const float r = state_.cum_rewards[ui];
    reward_sum += r;
    reward_max = std::max(reward_max, r);
    reward_min = std::min(reward_min, r);
    const float v = state_.virtual_rewards[ui];
    vr_sum += v;
    vr_max = std::max(vr_max, v);
    vr_min = std::min(vr_min, v);
  }
  if (n > 0) {
    info.mean_reward = static_cast<float>(reward_sum / n);
    info.max_reward = reward_max;
    info.min_reward = reward_min;
    info.mean_virtual_reward = static_cast<float>(vr_sum / n);
    info.max_virtual_reward = vr_max;
    info.min_virtual_reward = vr_min;
  }

  int64_t dt_sum = 0;
  int32_t dt_min = std::numeric_limits<int32_t>::max();
  int32_t dt_max = std::numeric_limits<int32_t>::min();
  int32_t dt_count = 0;
  for (int32_t i = 0; i < n; ++i) {
    const auto ui = static_cast<size_t>(i);
    if (!state_.will_clone[ui]) continue;
    dt_sum += state_.dt[ui];
    dt_min = std::min(dt_min, state_.dt[ui]);
    dt_max = std::max(dt_max, state_.dt[ui]);
    ++dt_count;
  }
  if (dt_count > 0) {
    info.mean_dt = static_cast<float>(static_cast<double>(dt_sum) / dt_count);
    info.min_dt = dt_min;
    info.max_dt = dt_max;
  }

  // Showcase: the env's display score when it has one (read from the stored
  // per-walker info, never from the env's batch cache), else the best
  // cumulative reward. Walkers without a state cannot be rendered.
  int32_t best = best_index();
  if (env_.has_display_score() && env_.has_walker_info()) {
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_scored = -1;
    for (int32_t i = 0; i < n; ++i) {
      const auto ui = static_cast<size_t>(i);
      if (state_.states[ui].empty()) continue;
      if (state_.info[ui].score > best_score) {
        best_score = state_.info[ui].score;
        best_scored = i;
      }
    }
    if (best_scored >= 0) best = best_scored;
  }
  info.best_walker_idx = best;
  if (params_.record_frames && !state_.states[static_cast<size_t>(best)].empty()) {
    env_.render_frame(state_.states[static_cast<size_t>(best)], best_frame_);
  }
  return info;
}

}  // namespace fg
