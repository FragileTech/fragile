#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace fg::fractal {
template <class T>
size_t capacity_bytes(const std::vector<T>& values) {
  return values.capacity() * sizeof(T);
}
// Non-owning C++17 batch view. Empty index views mean identity mapping.
template <class T>
struct View {
  T* data = nullptr;
  size_t size = 0;
  T& operator[](size_t i) const { return data[i]; }
  size_t index(size_t i) const { return data ? size_t(data[i]) : i; }
};
struct Selection {
  size_t count;
  View<const int32_t> sources, destinations;
};

// Environment storage is a type parameter, never a per-walker base class.
template <class Storage, class Info, class Action>
struct Population {
  using action_type = Action;
  Storage states;
  int32_t N = 0, obs_dim = 0, action_dim = 1;
  std::vector<float> observations, rewards, step_rewards, virtual_rewards;
  std::vector<uint8_t> dones, truncated, recoverable;
  std::vector<Action> actions, root_actions;
  std::vector<int32_t> dt, actual_dt;
  std::vector<Info> infos;
  std::vector<uint32_t> lineage;
  bool has_virtual_rewards = false, has_infos = false;
  size_t metadata_bytes() const {
    return capacity_bytes(observations) + capacity_bytes(rewards) + capacity_bytes(step_rewards) +
           capacity_bytes(virtual_rewards) + capacity_bytes(dones) + capacity_bytes(truncated) +
           capacity_bytes(recoverable) + capacity_bytes(actions) + capacity_bytes(root_actions) +
           capacity_bytes(dt) + capacity_bytes(actual_dt) + capacity_bytes(infos) +
           capacity_bytes(lineage);
  }
  bool alive(size_t i) const { return !(dones[i] || truncated[i]); }
  int32_t alive_count() const {
    int32_t count = 0;
    for (int32_t i = 0; i < N; ++i) count += alive(i);
    return count;
  }
  std::vector<uint8_t> alive_mask() const {
    std::vector<uint8_t> out(N);
    for (int32_t i = 0; i < N; ++i) out[i] = alive(i);
    return out;
  }
  void resize_metadata(int32_t n, int32_t obs, int32_t act) {
    N = n;
    obs_dim = obs;
    action_dim = act;
    observations.resize(size_t(n) * obs);
    rewards.resize(n);
    step_rewards.resize(n);
    virtual_rewards.resize(n);
    dones.resize(n);
    truncated.resize(n);
    recoverable.resize(n);
    actions.resize(size_t(n) * act);
    root_actions.resize(size_t(n) * act);
    dt.resize(n);
    actual_dt.resize(n);
    lineage.resize(n);
    if (has_infos) infos.resize(n);
  }
};

template <class State>
void copy_metadata(const State& from, size_t i, State& to, size_t j) {
  to.rewards[j] = from.rewards[i];
  to.step_rewards[j] = from.step_rewards[i];
  to.dones[j] = from.dones[i];
  to.truncated[j] = from.truncated[i];
  to.recoverable[j] = from.recoverable[i];
  to.dt[j] = from.dt[i];
  to.actual_dt[j] = from.actual_dt[i];
  to.virtual_rewards[j] = from.virtual_rewards[i];
  to.lineage[j] = from.lineage[i];
  std::copy_n(from.observations.data() + i * from.obs_dim, from.obs_dim,
              to.observations.data() + j * to.obs_dim);
  std::copy_n(from.actions.data() + i * from.action_dim, from.action_dim,
              to.actions.data() + j * to.action_dim);
  std::copy_n(from.root_actions.data() + i * from.action_dim, from.action_dim,
              to.root_actions.data() + j * to.action_dim);
  if (from.has_infos) to.infos[j] = from.infos[i];
}
// Scatter only transition outputs; accumulated rewards and ancestry belong to
// the algorithm and are deliberately not interpreted by a state backend.
template <class State>
void copy_transition(const State& from, size_t i, State& to, size_t j) {
  std::copy_n(from.observations.data() + i * from.obs_dim, from.obs_dim,
              to.observations.data() + j * to.obs_dim);
  to.step_rewards[j] = from.step_rewards[i];
  to.dones[j] = from.dones[i];
  to.truncated[j] = from.truncated[i];
  to.recoverable[j] = from.recoverable[i];
  to.actual_dt[j] = from.actual_dt[i];
  if (from.has_infos) to.infos[j] = from.infos[i];
}
}  // namespace fg::fractal
