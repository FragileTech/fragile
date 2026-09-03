// Deterministic mock environment with a visit-count key, used by the tree
// tests and by tests/fixtures/generate_tree_fixtures.py (VisitMockEnv there
// implements the IDENTICAL dynamics). Keep both in sync.
//
// State = observation = 3 floats holding integers [x, y, room] on the
// Montezuma-sized grid (160 x 160 cells, 24 rooms). One step with action a
// and frame-skip dt:
//   x'    = (x + (a + 1) * dt) mod 160
//   y'    = (y + dt) mod 160
//   room' = (room + (a == 3)) mod 24
//   reward = 0.1f * (a - 1) * dt + 0.01f * x'
//   done   = x' > 150
// walker_info() exposes (room, x, y) as the visit key.
#ifndef FRACTAL_GAS_VISIT_MOCK_ENV_HPP
#define FRACTAL_GAS_VISIT_MOCK_ENV_HPP

#include <cstring>

#include "env.hpp"

namespace fg {

class VisitMockEnv final : public BatchEnv {
 public:
  static constexpr int32_t kObsDim = 3;
  static constexpr int32_t kW = 160, kH = 160, kRooms = 24;
  static constexpr int32_t kDoneX = 150;

  int32_t n_actions() const override { return 4; }
  int32_t obs_dim() const override { return kObsDim; }

  void reset(std::vector<char>& state, std::vector<float>& obs) override {
    const float init[kObsDim] = {80.0f, 80.0f, 1.0f};
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
    info_.resize(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
      float s[kObsDim];
      std::memcpy(s, states[i].data(), sizeof(s));
      const int32_t a = actions[i];
      const int32_t d = dt[i];
      const int32_t nx = (static_cast<int32_t>(s[0]) + (a + 1) * d) % kW;
      const int32_t ny = (static_cast<int32_t>(s[1]) + d) % kH;
      const int32_t nroom = (static_cast<int32_t>(s[2]) + (a == 3 ? 1 : 0)) % kRooms;
      s[0] = static_cast<float>(nx);
      s[1] = static_cast<float>(ny);
      s[2] = static_cast<float>(nroom);
      new_states[i].resize(sizeof(s));
      std::memcpy(new_states[i].data(), s, sizeof(s));
      for (int32_t k = 0; k < kObsDim; ++k) {
        observations[i * kObsDim + static_cast<size_t>(k)] = s[k];
      }
      rewards[i] = 0.1f * (static_cast<float>(a) - 1.0f) * static_cast<float>(d) +
                   0.01f * static_cast<float>(nx);
      dones[i] = nx > kDoneX ? 1 : 0;
      truncated[i] = 0;
      WalkerInfo& wi = info_[i];
      wi = WalkerInfo{};
      wi.x = nx;
      wi.y = ny;
      wi.world = nroom;
      wi.has_visit_key = true;
      wi.visit_plane = nroom;
      wi.visit_x = nx;
      wi.visit_y = ny;
    }
  }

  bool has_walker_info() const override { return true; }
  const WalkerInfo& walker_info(int32_t batch_index) const override {
    static const WalkerInfo kEmpty{};
    const auto ui = static_cast<size_t>(batch_index);
    return ui < info_.size() ? info_[ui] : kEmpty;
  }
  bool has_visit_key() const override { return true; }

  void render_frame(const std::vector<char>&, std::vector<uint8_t>& rgba) override {
    rgba.clear();
  }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }

 private:
  std::vector<WalkerInfo> info_;
};

}  // namespace fg

#endif  // FRACTAL_GAS_VISIT_MOCK_ENV_HPP
