#pragma once
#include "kinetic.hpp"
#include "tensor_ops.hpp"
#include "visit_grid.hpp"
#include "walker_state.hpp"

namespace fg {
// Adapter for existing emulator and seed-action benchmark batch interfaces.
class SnapshotBackend {
 public:
  BatchEnv& env;
  VisitGrid& visits;
  bool count_visits = false, visit_reward = false, record_observations = true;
  float visit_coef = 1;
  SnapshotBackend(BatchEnv& e, VisitGrid& v) : env(e), visits(v) {}
  void resize(WalkerState& s, int n) { s.states.resize(n); }
  void copy(const WalkerState& src, size_t i, WalkerState& dst, size_t j) {
    dst.states[j] = src.states[i];
  }
  void observe(WalkerState&) {}
  void fitness_bonus(const WalkerState& s, std::vector<float>& fitness) {
    if (!count_visits || !visit_reward || !s.has_infos) return;
    keys_.resize(s.N);
    for (int i = 0; i < s.N; ++i) {
      const auto& w = s.infos[i];
      keys_[i] = {w.visit_plane, w.visit_x, w.visit_y};
    }
    visits.block_sums(keys_, sums_);
    for (auto& v : sums_) v = -v;
    asymmetric_rescale_into(sums_, sums_);
    const auto& other = sums_;
    for (int i = 0; i < s.N; ++i)
      fitness[i] *= float(visit_coef == 1 ? double(other[i])
                                          : std::pow(double(other[i]), double(visit_coef)));
  }
  template <class Input>
  void transition(const Input& src, fractal::Selection rows, const std::vector<int32_t>& actions,
                  const std::vector<int32_t>& dt, WalkerState& dst) {
    if (actions.size() != rows.count || dt.size() != rows.count ||
        (rows.sources.data && rows.sources.size != rows.count) ||
        (rows.destinations.data && rows.destinations.size != rows.count))
      throw std::invalid_argument("Invalid snapshot transition shape");
    for (size_t i = 0; i < rows.count; ++i)
      if (rows.sources.index(i) >= src.states.size() ||
          rows.destinations.index(i) >= size_t(dst.N))
        throw std::invalid_argument("Invalid snapshot transition index");
    if (rows.destinations.data) {
      batch_.has_infos = env.has_walker_info();
      batch_.resize_metadata(rows.count, dst.obs_dim, 1);
      resize(batch_, rows.count);
      transition(src, fractal::Selection{rows.count, rows.sources, {}}, actions, dt, batch_);
      for (size_t i = 0; i < rows.count; ++i) {
        size_t j = rows.destinations.index(i);
        std::swap(dst.states[j], batch_.states[i]);
        fractal::copy_transition(batch_, i, dst, j);
      }
      return;
    }
    staged_.resize(rows.count);
    for (size_t i = 0; i < rows.count; ++i) staged_[i] = src.states[rows.sources.index(i)];
    // Wave output is already a dense batch. Graph uses the same indexed
    // transition through its compact staging batch before scattering leaves.
    env.step_batch(staged_, actions, dt, dst.states, dst.observations, dst.step_rewards, dst.dones,
                   dst.truncated);
    dst.has_infos = env.has_walker_info();
    if (dst.has_infos) dst.infos.resize(rows.count);
    for (size_t i = 0; i < rows.count; ++i) {
      if (dst.has_infos) dst.infos[i] = env.walker_info(i);
      dst.recoverable[i] = env.has_recoverable_dones() && env.done_is_recoverable(i);
      int frames = env.frames_stepped(i);
      dst.actual_dt[i] = frames < 0 ? dt[i] : frames;
    }
  }
  void after_transition(const WalkerState& s) {
    if (!count_visits || !s.has_infos) return;
    keys_.resize(s.N);
    for (int i = 0; i < s.N; ++i) {
      const auto& w = s.infos[i];
      keys_[i] = {w.visit_plane, w.visit_x, w.visit_y};
    }
    visits.update(keys_);
  }
  void pose(const WalkerState& s, size_t i, float* out) {
    if (record_observations) std::copy_n(s.observations.data() + i * s.obs_dim, s.obs_dim, out);
  }
  uint32_t flags(const WalkerState& s, size_t i) { return s.alive(i) ? 0u : 1u; }

 private:
  std::vector<std::vector<char>> staged_;
  std::vector<VisitKey> keys_;
  std::vector<float> sums_;
  WalkerState batch_;
};
struct DiscreteActions {
  BatchEnv& env;
  RandomActionOperator& sampler;
  void sample(const WalkerState& s, const std::vector<int32_t>&, bool,
              std::vector<int32_t>& actions, std::vector<int32_t>& dt, Rng& rng) {
    sampler.sample_actions_into(s.N, env.n_actions(), rng, actions);
    sampler.sample_dt_into(s.N, rng, dt);
    sampler.last_actions = actions;
    sampler.last_dt = dt;
  }
};
}  // namespace fg
