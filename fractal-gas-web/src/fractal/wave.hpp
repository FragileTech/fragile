#pragma once
#include <cmath>
#include <limits>
#include <numeric>

#include "fractal/cloning.hpp"
#include "fractal/exploration_tree.hpp"
#include "fractal/population.hpp"

namespace fg::fractal {
struct WaveMetrics {
  uint32_t iteration = 0, cloned = 0, revived = 0, alive = 0, pruned = 0;
  int64_t frames = 0;
  float mean_reward = 0, max_reward = 0, min_reward = 0;
  float mean_fitness = 0, max_fitness = 0, min_fitness = 0, mean_dt = 0;
  int32_t min_dt = 0, max_dt = 0;
};

// One lifecycle for both packed and opaque states. Backend hooks perform only
// storage, transition, observation, and optional environment reward/trace work.
template <class State, class Backend, class ActionPolicy>
class Wave {
 public:
  State current, next, elite, elite_next;
  ExplorationTree tree;
  WaveMetrics metrics;
  bool has_elite = false;
  std::vector<int32_t> sources;
  explicit Wave(Backend& backend, ActionPolicy& policy) : backend_(backend), policy_(policy) {}
  const std::vector<int32_t>& fitness_companions() const { return fitness_companions_; }
  const std::vector<int32_t>& clone_companions() const { return companions_; }
  const std::vector<uint8_t>& clone_mask() const { return mask_; }
  // Fixed working storage, excluding recording growth and backend kernel scratch.
  template <class StorageBytes>
  size_t working_bytes(StorageBytes bytes) const {
    size_t total = 0;
    for (const auto* s : {&current, &next, &elite, &elite_next})
      total += bytes(s->states) + s->metadata_bytes();
    return total + capacity_bytes(sources) + capacity_bytes(alive_) + capacity_bytes(mask_) +
           capacity_bytes(companions_) + capacity_bytes(fitness_companions_) +
           capacity_bytes(fitness_) + capacity_bytes(ranked_) + capacity_bytes(pins_) +
           capacity_bytes(pose_) + capacity_bytes(record_action_);
  }
  void prepare(State& s, int n, int obs, int act, bool infos) {
    s.has_infos = infos;
    s.resize_metadata(n, obs, act);
    backend_.resize(s, n);
  }
  void reset(int n, int obs, int act, bool infos) {
    prepare(current, n, obs, act, infos);
    prepare(next, n, obs, act, infos);
    for (auto* s : {&current, &next}) {
      std::fill(s->rewards.begin(), s->rewards.end(), 0);
      std::fill(s->step_rewards.begin(), s->step_rewards.end(), 0);
      std::fill(s->virtual_rewards.begin(), s->virtual_rewards.end(), 0);
      std::fill(s->actions.begin(), s->actions.end(), 0);
      std::fill(s->root_actions.begin(), s->root_actions.end(), 0);
      std::fill(s->dones.begin(), s->dones.end(), 0);
      std::fill(s->truncated.begin(), s->truncated.end(), 0);
      std::fill(s->recoverable.begin(), s->recoverable.end(), 0);
      std::fill(s->lineage.begin(), s->lineage.end(), 0);
      std::fill(s->dt.begin(), s->dt.end(), 1);
      std::fill(s->actual_dt.begin(), s->actual_dt.end(), 0);
      s->has_virtual_rewards = false;
    }
    metrics = {};
    metrics.alive = n;
    has_elite = false;
    sources.resize(n);
    std::iota(sources.begin(), sources.end(), 0);
    companions_ = fitness_companions_ = sources;
    mask_.assign(n, 0);
    alive_.resize(n);
  }
  void begin_history(RecordingMode mode, size_t pose_dim, const float* root_pose,
                     std::vector<uint8_t> snapshot) {
    tree.reset(mode, current.action_dim, pose_dim);
    pose_.resize(pose_dim);
    record_action_.resize(current.action_dim);
    tree.root_snapshot = mode == RecordingMode::Off ? std::vector<uint8_t>{} : std::move(snapshot);
    uint32_t root = tree.append(0, 0, record_action_.data(), root_pose, 0, 0, 0, 0);
    std::fill(current.lineage.begin(), current.lineage.end(), root);
  }
  void copy_row(const State& from, size_t i, State& to, size_t j) {
    backend_.copy(from, i, to, j);
    copy_metadata(from, i, to, j);
  }
  const WaveMetrics& step(int elites, FractalCloningOperator& cloning, Rng& rng) {
    const int n = current.N, ad = current.action_dim;
    elites = std::clamp(elites, 0, n);
    if (!elites) has_elite = false;
    if (has_elite)
      for (int i = 0; i < std::min(elites, elite.N); ++i) copy_row(elite, i, current, i);
    backend_.observe(current);
    for (int i = 0; i < n; ++i) alive_[i] = current.alive(i);
    cloning.calculate_fitness_into(current.observations, n, current.obs_dim, current.rewards,
                                   current.step_rewards, alive_, rng, fitness_,
                                   fitness_companions_);
    backend_.fitness_bonus(current, fitness_);
    cloning.decide_cloning_into(fitness_, alive_, rng, companions_, mask_);
    // Elite slots are protected from replacement. They remain valid donor
    // sources for other walkers, but their own clone flags are always clear.
    if (has_elite) {
      const int injected = std::min(elites, elite.N);
      for (int i = 0; i < injected; ++i) mask_[static_cast<size_t>(i)] = 0;
    }
    uint32_t iteration = metrics.iteration + 1;
    metrics = {};
    metrics.iteration = iteration;
    for (int i = 0; i < n; ++i) {
      sources[i] = mask_[i] ? companions_[i] : i;
      metrics.cloned += mask_[i];
    }
    policy_.sample(current, sources, iteration == 1, next.actions, next.dt, rng);
    backend_.transition(current, Selection{size_t(n), {sources.data(), size_t(n)}, {}},
                        next.actions, next.dt, next);
    bool any_alive = false;
    for (int i = 0; i < n; ++i) any_alive |= next.alive(i);
    if (!any_alive)
      for (int i = 0; i < n; ++i)
        if (next.dones[i] && !next.truncated[i] && next.recoverable[i]) {
          next.dones[i] = 0;
          ++metrics.revived;
        }
    tree.reserve(n);
    double rewards = 0, fitness_sum = 0, durations = 0;
    metrics.max_reward = metrics.max_fitness = -std::numeric_limits<float>::infinity();
    metrics.min_reward = metrics.min_fitness = std::numeric_limits<float>::infinity();
    metrics.min_dt = std::numeric_limits<int32_t>::max();
    for (int i = 0; i < n; ++i) {
      const int donor = sources[i];
      next.rewards[i] = current.rewards[donor] + next.step_rewards[i];
      std::copy_n((iteration == 1 ? next.actions.data() + size_t(i) * ad
                                  : current.root_actions.data() + size_t(donor) * ad),
                  ad, next.root_actions.data() + size_t(i) * ad);
      next.virtual_rewards[i] = fitness_[i];
      metrics.alive += next.alive(i);
      metrics.frames += next.actual_dt[i];
      rewards += next.rewards[i];
      fitness_sum += fitness_[i];
      durations += next.dt[i];
      metrics.max_reward = std::max(metrics.max_reward, next.rewards[i]);
      metrics.min_reward = std::min(metrics.min_reward, next.rewards[i]);
      metrics.max_fitness = std::max(metrics.max_fitness, fitness_[i]);
      metrics.min_fitness = std::min(metrics.min_fitness, fitness_[i]);
      metrics.min_dt = std::min(metrics.min_dt, next.dt[i]);
      metrics.max_dt = std::max(metrics.max_dt, next.dt[i]);
      if (tree.mode != RecordingMode::Off) {
        backend_.pose(next, i, pose_.data());
        for (int k = 0; k < ad; ++k) record_action_[k] = float(next.actions[size_t(i) * ad + k]);
        next.lineage[i] = tree.append(current.lineage[donor], next.actual_dt[i],
                                      record_action_.data(), pose_.data(), next.rewards[i],
                                      next.step_rewards[i], fitness_[i], backend_.flags(next, i));
      }
    }
    metrics.mean_reward = float(rewards / n);
    metrics.mean_fitness = float(fitness_sum / n);
    metrics.mean_dt = float(durations / n);
    next.has_virtual_rewards = true;
    std::swap(current, next);
    backend_.after_transition(current);
    update_elites(elites);
    if (has_elite)
      for (float r : elite.rewards) metrics.max_reward = std::max(metrics.max_reward, r);
    if (tree.mode == RecordingMode::Pruned) {
      pins_.assign(current.lineage.begin(), current.lineage.end());
      if (has_elite) pins_.insert(pins_.end(), elite.lineage.begin(), elite.lineage.end());
      metrics.pruned = uint32_t(tree.prune(pins_));
    }
    return metrics;
  }
  template <class SaveStorage>
  void save(CheckpointWriter& out, SaveStorage storage) const {
    out.scalar(metrics);
    out.scalar(uint32_t(has_elite));
    out.vector(sources);
    out.vector(fitness_companions_);
    out.vector(companions_);
    out.vector(mask_);
    auto state = [&](const State& s) {
      storage(s.states);
      out.vector(s.observations);
      out.vector(s.rewards);
      out.vector(s.step_rewards);
      out.vector(s.virtual_rewards);
      out.vector(s.dones);
      out.vector(s.truncated);
      out.vector(s.recoverable);
      out.vector(s.actions);
      out.vector(s.root_actions);
      out.vector(s.dt);
      out.vector(s.actual_dt);
      out.vector(s.lineage);
      out.vector(s.infos);
      out.scalar(uint32_t(s.has_virtual_rewards));
    };
    state(current);
    if (has_elite) state(elite);
    tree.save_checkpoint(out);
  }
  template <class LoadStorage>
  void load(CheckpointReader& in, int elites, LoadStorage storage) {
    metrics = in.scalar<WaveMetrics>();
    auto has = in.scalar<uint32_t>();
    if (has > 1 || metrics.alive > uint32_t(current.N) || (has && !elites))
      throw std::invalid_argument("Invalid Wave checkpoint");
    has_elite = has;
    auto indices = [&](std::vector<int32_t>& out) {
      out = in.template vector<int32_t>();
      if (out.size() != size_t(current.N))
        throw std::invalid_argument("Invalid checkpoint donor shape");
      for (int i : out)
        if (i < 0 || i >= current.N) throw std::invalid_argument("Invalid checkpoint donor index");
    };
    indices(sources);
    indices(fitness_companions_);
    indices(companions_);
    mask_ = in.template vector<uint8_t>();
    if (mask_.size() != size_t(current.N))
      throw std::invalid_argument("Invalid checkpoint clone mask");
    for (auto m : mask_)
      if (m > 1) throw std::invalid_argument("Invalid checkpoint clone mask");
    auto values = [&](auto& target) {
      using T = typename std::decay_t<decltype(target)>::value_type;
      auto value = in.template vector<T>();
      if (value.size() != target.size())
        throw std::invalid_argument("Checkpoint population shape mismatch");
      if constexpr (std::is_floating_point_v<T>)
        for (auto x : value)
          if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite population checkpoint");
      target = std::move(value);
    };
    auto state = [&](State& s) {
      storage(s.states);
      values(s.observations);
      values(s.rewards);
      values(s.step_rewards);
      values(s.virtual_rewards);
      values(s.dones);
      values(s.truncated);
      values(s.recoverable);
      values(s.actions);
      values(s.root_actions);
      values(s.dt);
      values(s.actual_dt);
      values(s.lineage);
      values(s.infos);
      auto v = in.scalar<uint32_t>();
      if (v > 1) throw std::invalid_argument("Invalid fitness checkpoint flag");
      s.has_virtual_rewards = v;
      for (int i = 0; i < s.N; ++i)
        if (s.dones[i] > 1 || s.truncated[i] > 1 || s.recoverable[i] > 1 || s.dt[i] < 1 ||
            s.actual_dt[i] < 0 || s.actual_dt[i] > s.dt[i])
          throw std::invalid_argument("Invalid population transition checkpoint");
    };
    state(current);
    if (has_elite) {
      prepare(elite, elites, current.obs_dim, current.action_dim, current.has_infos);
      state(elite);
    }
    tree.load_checkpoint(in);
    pose_.resize(tree.pose_dim());
    record_action_.resize(tree.action_dim());
    if (tree.mode != RecordingMode::Off) {
      for (auto id : current.lineage) tree.node(id);
      if (has_elite)
        for (auto id : elite.lineage) tree.node(id);
    }
  }

 private:
  Backend& backend_;
  ActionPolicy& policy_;
  std::vector<uint8_t> alive_, mask_;
  std::vector<int32_t> companions_, fitness_companions_;
  std::vector<float> fitness_;
  std::vector<int32_t> ranked_;
  std::vector<uint32_t> pins_;
  std::vector<float> pose_, record_action_;
  void update_elites(int count) {
    if (!count) return;
    const int old = has_elite ? elite.N : 0;
    ranked_.clear();
    ranked_.reserve(current.N + old);
    for (int i = 0; i < old; ++i)
      if (elite.alive(i)) ranked_.push_back(i);
    for (int i = 0; i < current.N; ++i)
      if (current.alive(i)) ranked_.push_back(old + i);
    if (ranked_.empty()) {
      has_elite = false;
      return;
    }
    const int selected = std::min(count, static_cast<int>(ranked_.size()));
    prepare(elite_next, count, current.obs_dim, current.action_dim, current.has_infos);
    elite_next.has_virtual_rewards = current.has_virtual_rewards;
    auto reward = [&](int i) { return i < old ? elite.rewards[i] : current.rewards[i - old]; };
    std::partial_sort(ranked_.begin(), ranked_.begin() + selected, ranked_.end(), [&](int a, int b) {
      return reward(a) == reward(b) ? a < b : reward(a) > reward(b);
    });
    for (int i = 0; i < count; ++i) {
      // Keep the elite bank at the requested size when fewer alive candidates
      // exist by repeating the best alive candidate; no dead state is stored.
      int src = ranked_[static_cast<size_t>(i % selected)];
      copy_row(src < old ? elite : current, src < old ? src : src - old, elite_next, i);
    }
    std::swap(elite, elite_next);
    has_elite = true;
  }
};
}  // namespace fg::fractal
