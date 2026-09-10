#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "fractal/metrics.hpp"
#include "fractal/diagnostics.hpp"
#include "fractal/population.hpp"
#include "fractal/tensor_ops.hpp"
#include "fractal/visit_grid.hpp"
namespace fg::fractal {
struct GraphConfig {
  int32_t start_walkers = 15;  // demo: start_walkers = min_leafs = 15
  int32_t min_leafs = 15;
  int32_t max_walkers = 100000;  // hard cap on the population
  DistanceMetric distance_metric = DistanceMetric::L2;
  float dist_coef = 1.0f;
  float reward_coef = 1.0f;
  int32_t dt_min = 1;  // INCLUSIVE range like the wave; the reference's
  int32_t dt_max = 4;  // UniformDtSampler(1, 5) draws {1, 2, 3, 4}
  float eps = 1e-8f;
  bool count_visits = true;  // effective only when env.has_visit_key()
  bool visit_reward = true;  // ablation: multiply the visit term into vr
  float visit_coef = 1.0f;   // exponent on the visit term (reference: 1)
  float erase_coef = 0.05f;
  int32_t agg_block_size = 5;
  bool record_frames = false;
  uint64_t seed = 0;
};

template <class Storage, class Info, class Action>
struct GraphPopulation {
  using action_type = Action;
  int32_t action_dim = 1;
  int32_t n = 0;
  int32_t obs_dim = 0;
  Storage states;                   // empty = never stepped
  std::vector<float> observations;  // [n * obs_dim]
  std::vector<float> rewards;       // last step reward
  std::vector<float> cum_rewards;
  std::vector<uint8_t> oobs;  // dead flag (env done)
  std::vector<int32_t> parent;
  std::vector<uint8_t> is_leaf;  // mask of the last phase 2
  std::vector<Action> actions;
  std::vector<int32_t> dt;
  std::vector<float> virtual_rewards;
  std::vector<float> other_rewards;
  std::vector<float> distances;
  std::vector<float> clone_probs;
  std::vector<int32_t> distance_ix;  // compas1
  std::vector<int32_t> clone_ix;     // compas2
  std::vector<uint8_t> wants_clone;
  std::vector<uint8_t> is_cloned;
  std::vector<uint8_t> will_clone;
  std::vector<Info> info;  // per walker, gathered on clone

  std::vector<uint8_t> alive_mask() const {
    std::vector<uint8_t> out(n);
    for (int i = 0; i < n; ++i) out[i] = !oobs[i];
    return out;
  }
  int32_t alive_count() const {
    int count = 0;
    for (int i = 0; i < n; ++i) count += !oobs[i];
    return count;
  }
};
template <class Backend, class Sampler>
class Graph {
 public:
  using State =
      GraphPopulation<typename Backend::Storage, typename Backend::Info, typename Backend::Action>;
  Backend& backend_;
  Sampler& sampler_;
  Rng& rng_;
  GraphConfig& params_;
  State state_;
  CloneDiagnostics diagnostics;
  VisitGrid visits_;
  bool count_visits_;
  int64_t total_steps_ = 0, total_clones_ = 0, total_frames_ = 0;
  int32_t iteration_ = 0;
  Graph(Backend& b, Sampler& sampler, Rng& rng, GraphConfig& config)
      : backend_(b),
        sampler_(sampler),
        rng_(rng),
        params_(config),
        visits_(config.agg_block_size, config.erase_coef),
        count_visits_(config.count_visits && b.has_visit_key()) {}
  void grow(int32_t count) {
    // Fresh slots hold the reset values of the preallocated reference buffers:
    // parent 0 (the root), oobs True, zero rewards, zero observation, no
    // state, leaf.
    const auto d = static_cast<size_t>(state_.obs_dim);
    for (int32_t c = 0; c < count; ++c) {
      state_.observations.insert(state_.observations.end(), d, 0.0f);
      state_.rewards.push_back(0.0f);
      state_.cum_rewards.push_back(0.0f);
      state_.oobs.push_back(1);
      state_.parent.push_back(0);
      state_.is_leaf.push_back(1);
      state_.actions.insert(state_.actions.end(), state_.action_dim, 0);
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
    backend_.grow(state_.states, state_.n);
  }

  void collect_visit_keys(const std::vector<int32_t>& walkers, std::vector<VisitKey>& keys) const {
    keys.resize(walkers.size());
    for (size_t j = 0; j < walkers.size(); ++j) {
      keys[j] = backend_.visit_key(state_.info[static_cast<size_t>(walkers[j])]);
    }
  }

  void reset() {
    state_ = {};
    diagnostics.decisions.clear();
    total_steps_ = total_clones_ = total_frames_ = 0;
    iteration_ = 0;
    visits_.reset();
    visits_.set_erase_coef(params_.erase_coef);
    visits_.set_block_size(params_.agg_block_size);
    int n = std::min(params_.start_walkers, params_.max_walkers);
    auto actions = sampler_.sample_actions(n, backend_.n_actions(), rng_);
    auto root = backend_.reset_root();
    state_.obs_dim = backend_.observation_dim();
    state_.action_dim = backend_.action_dim();
    grow(n);
    backend_.broadcast(root, state_.states, n);
    auto dt = sampler_.sample_dt(n, params_.dt_min, params_.dt_max, rng_);
    prepare_batch(n);
    backend_.transition(state_, Selection{size_t(n), {}, {}}, actions, dt, batch_);
    state_.observations = batch_.observations;
    state_.actions = std::move(actions);
    for (int i = 0; i < n; ++i) {
      if (i) {
        state_.rewards[i] = state_.cum_rewards[i] = batch_.step_rewards[i];
        state_.oobs[i] = batch_.dones[i] || batch_.truncated[i];
        backend_.commit(batch_.states, i, state_.states, i);
      }
      state_.dt[i] = dt[i];
      total_frames_ += batch_.actual_dt[i];
      if (batch_.has_infos) state_.info[i] = batch_.infos[i];
    }
    if (count_visits_) {
      std::vector<int32_t> all(n);
      std::iota(all.begin(), all.end(), 0);
      auto& keys = keys_;
      collect_visit_keys(all, keys);
      visits_.update(keys);
    }
  }
  template <class B>
  static auto eligible(const B& b, const State& s, int i, int)
      -> decltype(b.best_candidate(s.states, size_t(i))) {
    return b.best_candidate(s.states, size_t(i));
  }
  template <class B>
  static bool eligible(const B&, const State&, int, long) { return true; }
  int32_t best_index() const {
    // torch.argmax: first maximum.
    int32_t best = -1;
    for (int32_t i = 0; i < state_.n; ++i) {
      if (eligible(backend_, state_, i, 0) &&
          (best < 0 || state_.cum_rewards[i] > state_.cum_rewards[best])) best = i;
    }
    return best;
  }

  std::pair<int32_t, float> get_best_walker() const {
    if (state_.n == 0) return {0, 0.0f};
    const int32_t best = best_index();
    return {best, best < 0 ? 0.f : state_.cum_rewards[static_cast<size_t>(best)]};
  }

  StepInfo step() {
    const int32_t n = state_.n;
    const auto d = static_cast<size_t>(state_.obs_dim);
    auto& alive = alive_;
    alive.resize(n);
    for (int i = 0; i < n; ++i) alive[i] = !state_.oobs[i];

    // ---- 1. calculate_virtual_reward (uses the PREVIOUS leaf mask) ----------
    auto& compas1 = compas1_;
    sampler_.sample_companions_into(alive, rng_, compas1);
    auto& distances = distances_;
    companion_distances_into(state_.observations, compas1, n, state_.obs_dim, backend_.pool(),
                             distances, params_.distance_metric);
    auto& distance_norm = distance_norm_;
    asymmetric_rescale_into(distances, distance_norm);
    const auto reward_stats = mean_std_masked(state_.cum_rewards, state_.is_leaf);
    auto& rewards_norm = rewards_norm_;
    relativize_with_stats_into(state_.cum_rewards, reward_stats.first, reward_stats.second,
                               rewards_norm);

    auto& other = other_;
    other.assign(n, 1.f);
    if (count_visits_ && params_.visit_reward) {
      // MontezumaTree.calculate_other_reward(): minus the 5x5 block visit
      // count of every walker's cell, relativized with leaf-only statistics.
      auto& all = all_;
      all.resize(n);
      for (int32_t i = 0; i < n; ++i) all[static_cast<size_t>(i)] = i;
      auto& keys = keys_;
      collect_visit_keys(all, keys);
      auto& sums = sums_;
      visits_.block_sums(keys, sums);
      auto& visits_val = visits_val_;
      visits_val.resize(n);
      for (int32_t i = 0; i < n; ++i) {
        visits_val[static_cast<size_t>(i)] = -sums[static_cast<size_t>(i)];
      }
      const auto stats = mean_std_masked(visits_val, state_.is_leaf);
      relativize_with_stats_into(visits_val, stats.first, stats.second, other);
      // vr = distance^dist_coef * reward^reward_coef * other^visit_coef; the
      // reference multiplies `other` directly (exponent 1, kept exact).
      if (params_.visit_coef != 1.0f) {
        for (float& o : other) {
          o = static_cast<float>(
              std::pow(static_cast<double>(o), static_cast<double>(params_.visit_coef)));
        }
      }
    }

    for (int32_t i = 0; i < n; ++i) {
      const auto ui = static_cast<size_t>(i);
      const double vr = std::pow(static_cast<double>(distance_norm[ui]),
                                 static_cast<double>(params_.dist_coef)) *
                        std::pow(static_cast<double>(rewards_norm[ui]),
                                 static_cast<double>(params_.reward_coef)) *
                        static_cast<double>(other[ui]);
      state_.virtual_rewards[ui] = static_cast<float>(vr);
      state_.distances[ui] = distances[ui];
      state_.distance_ix[ui] = compas1[ui];
      state_.other_rewards[ui] = other[ui];
    }

    if (diagnostics.enabled) {
      diagnostics.decisions.assign(n, {});
      for (int i = 0; i < n; ++i) {
        auto& v = diagnostics.decisions[i];
        v.slot = i; v.distance_companion = compas1[i];
        v.distance = distances[i]; v.distance_norm = distance_norm[i];
        v.reward_norm = rewards_norm[i]; v.other = other[i];
        v.fitness = state_.virtual_rewards[i]; v.alive = alive[i];
        v.normalization_leaf = state_.is_leaf[i];
      }
    }

    // ---- 2. is_leaf = get_is_leaf(parent) ------------------------------------
    std::fill(state_.is_leaf.begin(), state_.is_leaf.end(), 1);
    for (int32_t i = 0; i < n; ++i) {
      state_.is_leaf[static_cast<size_t>(state_.parent[static_cast<size_t>(i)])] = 0;
    }

    // ---- 3. calculate_clone --------------------------------------------------
    auto& compas2 = compas2_;
    sampler_.sample_companions_into(alive, rng_, compas2);
    const float eps = params_.eps;
    for (int32_t i = 0; i < n; ++i) {
      const auto ui = static_cast<size_t>(i);
      const float vr = state_.virtual_rewards[ui];
      const float vr_c = state_.virtual_rewards[static_cast<size_t>(compas2[ui])];
      state_.clone_probs[ui] = (vr_c - vr) / (vr > eps ? vr : eps);
      state_.clone_ix[ui] = compas2[ui];
    }
    auto& uniforms = uniforms_;
    sampler_.sample_uniforms_into(n, rng_, uniforms);
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
          (state_.wants_clone[ui] && state_.is_leaf[ui] && !state_.is_cloned[ui]) ? 1 : 0;
    }
    if (diagnostics.enabled)
      for (int i = 0; i < n; ++i) {
        auto& v = diagnostics.decisions[i];
        v.clone_donor = compas2[i]; v.donor_fitness = state_.virtual_rewards[compas2[i]];
        v.clone_score = state_.clone_probs[i]; v.draw = uniforms[i];
        v.leaf = state_.is_leaf[i]; v.donor_protected = state_.is_cloned[i];
        v.wanted = state_.wants_clone[i]; v.cloned = state_.will_clone[i];
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
    const int best = best_index();
    if (best >= 0) {
      state_.will_clone[static_cast<size_t>(best)] = 0;
      if (diagnostics.enabled) {
        diagnostics.decisions[best].best_protected = true;
        diagnostics.decisions[best].cloned = false;
      }
    }
    auto& cloning = cloning_;
    cloning.clear();
    for (int32_t i = 0; i < n; ++i) {
      if (state_.will_clone[static_cast<size_t>(i)]) cloning.push_back(i);
    }
    stepping_.clear();
    sources_.clear();
    prior_rewards_.clear();
    for (int i : cloning) {
      int donor = compas2[i];
      state_.parent[i] = donor;
      if (!backend_.valid_slot(state_.states, donor)) {
        state_.will_clone[i] = 0;
        if (diagnostics.enabled) {
          diagnostics.decisions[i].invalid_donor = true;
          diagnostics.decisions[i].cloned = false;
        }
        continue;
      }
      stepping_.push_back(i);
      sources_.push_back(donor);
      prior_rewards_.push_back(state_.cum_rewards[donor]);
    }
    const int k = int(stepping_.size());
    total_clones_ += k;
    if (k) {
      auto& actions = actions_;
      auto& dt = dt_;
      sampler_.sample_actions_into(k, backend_.n_actions(), rng_, actions);
      sampler_.sample_dt_into(k, params_.dt_min, params_.dt_max, rng_, dt);
      prepare_batch(k);
      backend_.transition(state_, Selection{size_t(k), {sources_.data(), size_t(k)}, {}}, actions,
                          dt, batch_);
      for (int j = 0; j < k; ++j) {
        int i = stepping_[j];
        std::copy_n(batch_.observations.data() + size_t(j) * d, d,
                    state_.observations.data() + size_t(i) * d);
        state_.rewards[i] = batch_.step_rewards[j];
        state_.cum_rewards[i] = prior_rewards_[j] + batch_.step_rewards[j];
        state_.oobs[i] = batch_.dones[j] || batch_.truncated[j];
        backend_.commit(batch_.states, j, state_.states, i);
        for (int c = 0; c < state_.action_dim; ++c)
          state_.actions[size_t(i) * state_.action_dim + c] =
              actions[size_t(j) * state_.action_dim + c];
        state_.dt[i] = dt[j];
        total_frames_ += batch_.actual_dt[j];
        if (batch_.has_infos) state_.info[i] = batch_.infos[j];
      }
      if (count_visits_) {
        auto& keys = keys_;
        collect_visit_keys(stepping_, keys);
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

  StepInfo collect_info(int32_t k, int32_t leaves) {
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

    info.best_walker_idx = best_index();
    return info;
  }

 private:
  typename Backend::Batch batch_;
  std::vector<int32_t> stepping_, sources_;
  std::vector<float> prior_rewards_, distances_, distance_norm_, rewards_norm_, other_, sums_,
      visits_val_, uniforms_;
  std::vector<int32_t> compas1_, compas2_, all_, cloning_, dt_;
  std::vector<typename Backend::Action> actions_;
  std::vector<uint8_t> alive_;
  std::vector<VisitKey> keys_;
  void prepare_batch(int n) {
    batch_.has_infos = backend_.has_infos();
    batch_.resize_metadata(n, state_.obs_dim, state_.action_dim);
    backend_.resize(batch_, n);
  }
};
}  // namespace fg::fractal
