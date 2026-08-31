#include "walker_state.hpp"

#include "tensor_ops.hpp"

namespace fg {

std::vector<uint8_t> WalkerState::alive_mask() const {
  std::vector<uint8_t> mask(static_cast<size_t>(N));
  for (int32_t i = 0; i < N; ++i) mask[static_cast<size_t>(i)] = alive(i) ? 1 : 0;
  return mask;
}

int32_t WalkerState::alive_count() const {
  int32_t count = 0;
  for (int32_t i = 0; i < N; ++i) count += alive(i) ? 1 : 0;
  return count;
}

WalkerState WalkerState::clone(const std::vector<int32_t>& companions,
                               const std::vector<uint8_t>& will_clone) const {
  WalkerState out;
  out.N = N;
  out.obs_dim = obs_dim;

  out.states = states;
  for (size_t i = 0; i < will_clone.size(); ++i) {
    if (will_clone[i]) out.states[i] = states[static_cast<size_t>(companions[i])];
  }

  out.observations = gather_clone_rows(observations, companions, will_clone, obs_dim);
  out.rewards = gather_clone(rewards, companions, will_clone);
  out.step_rewards = gather_clone(step_rewards, companions, will_clone);
  out.dones = gather_clone(dones, companions, will_clone);
  out.truncated = gather_clone(truncated, companions, will_clone);
  out.actions = gather_clone(actions, companions, will_clone);
  out.dt = gather_clone(dt, companions, will_clone);

  out.has_virtual_rewards = has_virtual_rewards;
  if (has_virtual_rewards) {
    out.virtual_rewards = gather_clone(virtual_rewards, companions, will_clone);
  }
  return out;
}

void WalkerState::inject(const WalkerState& source, int32_t count) {
  const auto d = static_cast<size_t>(obs_dim);
  for (int32_t i = 0; i < count; ++i) {
    const auto ui = static_cast<size_t>(i);
    states[ui] = source.states[ui];
    for (size_t k = 0; k < d; ++k) {
      observations[ui * d + k] = source.observations[ui * d + k];
    }
    rewards[ui] = source.rewards[ui];
    step_rewards[ui] = source.step_rewards[ui];
    dones[ui] = source.dones[ui];
    truncated[ui] = source.truncated[ui];
    actions[ui] = source.actions[ui];
    dt[ui] = source.dt[ui];
    if (has_virtual_rewards && source.has_virtual_rewards) {
      virtual_rewards[ui] = source.virtual_rewards[ui];
    }
  }
}

WalkerState WalkerState::extract(const WalkerState& s,
                                 const std::vector<int32_t>& indices) {
  WalkerState out;
  out.N = static_cast<int32_t>(indices.size());
  out.obs_dim = s.obs_dim;
  const auto d = static_cast<size_t>(s.obs_dim);

  out.states.reserve(indices.size());
  out.observations.resize(indices.size() * d);
  out.rewards.resize(indices.size());
  out.step_rewards.resize(indices.size());
  out.dones.resize(indices.size());
  out.truncated.resize(indices.size());
  out.actions.resize(indices.size());
  out.dt.resize(indices.size());
  out.has_virtual_rewards = s.has_virtual_rewards;
  if (s.has_virtual_rewards) out.virtual_rewards.resize(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    const auto j = static_cast<size_t>(indices[i]);
    out.states.push_back(s.states[j]);
    for (size_t k = 0; k < d; ++k) out.observations[i * d + k] = s.observations[j * d + k];
    out.rewards[i] = s.rewards[j];
    out.step_rewards[i] = s.step_rewards[j];
    out.dones[i] = s.dones[j];
    out.truncated[i] = s.truncated[j];
    out.actions[i] = s.actions[j];
    out.dt[i] = s.dt[j];
    if (s.has_virtual_rewards) out.virtual_rewards[i] = s.virtual_rewards[j];
  }
  return out;
}

WalkerState WalkerState::concat(const WalkerState& a, const WalkerState& b) {
  WalkerState out;
  out.N = a.N + b.N;
  out.obs_dim = a.obs_dim;

  auto cat = [](auto& dst, const auto& x, const auto& y) {
    dst = x;
    dst.insert(dst.end(), y.begin(), y.end());
  };
  cat(out.states, a.states, b.states);
  cat(out.observations, a.observations, b.observations);
  cat(out.rewards, a.rewards, b.rewards);
  cat(out.step_rewards, a.step_rewards, b.step_rewards);
  cat(out.dones, a.dones, b.dones);
  cat(out.truncated, a.truncated, b.truncated);
  cat(out.actions, a.actions, b.actions);
  cat(out.dt, a.dt, b.dt);
  // Python guard: concat vr only when both sides have it.
  out.has_virtual_rewards = a.has_virtual_rewards && b.has_virtual_rewards;
  if (out.has_virtual_rewards) {
    cat(out.virtual_rewards, a.virtual_rewards, b.virtual_rewards);
  }
  return out;
}

}  // namespace fg
