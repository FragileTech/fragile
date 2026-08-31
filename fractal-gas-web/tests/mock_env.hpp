// Deterministic mock environment used by the full-step tests and by the
// Python fixture generator (tests/fixtures/generate_fixtures.py implements
// the IDENTICAL dynamics). Keep both in sync.
//
// State: 3 floats [p, q, r]. Observation = the state. Dynamics for one
// step with action a and frame-skip dt:
//   p' = p + (a + 1) * dt
//   q' = q + 0.5 * dt
//   r' = r + 1
//   reward = 0.1f * (a - 1) * dt + 0.01f * p'
//   done   = p' > 20
// The reward is state-dependent so walker rewards are (near) unique — the
// Python elite buffer relies on torch.topk whose tie order is arbitrary, so
// fixtures must avoid reward ties to be replayable.
#ifndef FRACTAL_GAS_MOCK_ENV_HPP
#define FRACTAL_GAS_MOCK_ENV_HPP

#include <cstring>

#include "env.hpp"

namespace fg {

class MockEnv final : public BatchEnv {
 public:
  static constexpr int32_t kObsDim = 3;

  int32_t n_actions() const override { return 4; }
  int32_t obs_dim() const override { return kObsDim; }

  void reset(std::vector<char>& state, std::vector<float>& obs) override {
    const float init[kObsDim] = {0.0f, 0.0f, 0.0f};
    state.resize(sizeof(init));
    std::memcpy(state.data(), init, sizeof(init));
    obs.assign(init, init + kObsDim);
  }

  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions,
                  const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& new_states,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override {
    for (size_t i = 0; i < states.size(); ++i) {
      float s[kObsDim];
      std::memcpy(s, states[i].data(), sizeof(s));
      const float a = static_cast<float>(actions[i]);
      const float d = static_cast<float>(dt[i]);
      s[0] += (a + 1.0f) * d;
      s[1] += 0.5f * d;
      s[2] += 1.0f;
      new_states[i].resize(sizeof(s));
      std::memcpy(new_states[i].data(), s, sizeof(s));
      for (int32_t k = 0; k < kObsDim; ++k) {
        observations[i * kObsDim + static_cast<size_t>(k)] = s[k];
      }
      rewards[i] = 0.1f * (a - 1.0f) * d + 0.01f * s[0];
      dones[i] = s[0] > 20.0f ? 1 : 0;
      truncated[i] = 0;
    }
  }

  void render_frame(const std::vector<char>&, std::vector<uint8_t>& rgba) override {
    rgba.clear();
  }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }
};

}  // namespace fg

#endif  // FRACTAL_GAS_MOCK_ENV_HPP
