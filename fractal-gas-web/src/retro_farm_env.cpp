#ifdef __EMSCRIPTEN__

#include "retro_farm_env.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/threading.h>

#include <cstring>
#include <stdexcept>

namespace fg {

namespace {

int32_t atomic_load_i32(const int32_t* p) {
  return __atomic_load_n(p, __ATOMIC_SEQ_CST);
}
void atomic_store_i32(int32_t* p, int32_t v) {
  __atomic_store_n(p, v, __ATOMIC_SEQ_CST);
}

}  // namespace

RetroFarmEnv::RetroFarmEnv(uintptr_t regions_ptr, int n_workers,
                           size_t blob_len, int32_t obs_mode, RetroGame game)
    : game_(game),
      obs_mode_(obs_mode),
      n_workers_(n_workers < 1 ? 1 : (n_workers > 8 ? 8 : n_workers)),
      regions_(reinterpret_cast<uint8_t*>(regions_ptr)),
      blob_len_(blob_len) {
  if (!regions_ || blob_len_ == 0 || blob_len_ > kBlobCap) {
    throw std::runtime_error("RetroFarmEnv: invalid pre-spawned farm");
  }
  // Workers already reported ready (worker.js awaited them); sanity-check.
  for (int slot = 0; slot < n_workers_; ++slot) {
    if (atomic_load_i32(header(slot) + kStatus) != 1) {
      throw std::runtime_error("RetroFarmEnv: core worker not ready");
    }
  }
  set_reward_weights(reward_weights_);
}

void RetroFarmEnv::set_reward_weights(const SonicRewardWeights& w) {
  static_assert(sizeof(SonicRewardWeights) ==
                    kSonicRewardWeightCount * sizeof(float),
                "SonicRewardWeights must be a plain float array");
  reward_weights_ = w;
  for (int slot = 0; slot < n_workers_; ++slot) {
    int32_t* h = header(slot);
    // Slots are idle between step_batch calls, so plain stores suffice; the
    // worker reads the words under its own Atomics.load of the ctrl word.
    std::memcpy(h + kWeights, &w, sizeof(w));
    atomic_store_i32(h + kWeightCount, kSonicRewardWeightCount);
  }
}

void RetroFarmEnv::post_command(int slot, int32_t cmd) {
  int32_t* h = header(slot);
  atomic_store_i32(h + kCtrl, cmd);
  // Wake the worker's Atomics.wait on the ctrl word.
  emscripten_futex_wake(h + kCtrl, 1);
}

void RetroFarmEnv::wait_idle(int slot, const char* what) {
  int32_t* h = header(slot);
  while (atomic_load_i32(h + kCtrl) > 0) emscripten_thread_sleep(0.2);
  if (atomic_load_i32(h + kCtrl) < 0) {
    atomic_store_i32(h + kCtrl, 0);
    throw std::runtime_error(std::string("RetroFarmEnv: worker failed in ") +
                             what);
  }
}

void RetroFarmEnv::reset(std::vector<char>& state, std::vector<float>& obs) {
  if (has_initial_) {
    state = initial_state_;
    obs = initial_obs_;
    return;
  }
  // Boot on worker 0 (a one-time ~10s of emulated frames).
  post_command(0, kCmdBoot);
  wait_idle(0, "boot");

  state.assign(blob_area(0), blob_area(0) + blob_len_);
  const size_t d = static_cast<size_t>(obs_dim());
  obs.resize(d);
  std::memcpy(obs.data(), obs_area(0), d * sizeof(float));

  initial_state_.assign(state.begin(), state.end());
  initial_obs_ = obs;
  has_initial_ = true;
}

void RetroFarmEnv::step_batch(const std::vector<std::vector<char>>& states,
                              const std::vector<int32_t>& actions,
                              const std::vector<int32_t>& dt,
                              std::vector<std::vector<char>>& new_states,
                              std::vector<float>& observations,
                              std::vector<float>& rewards,
                              std::vector<uint8_t>& dones,
                              std::vector<uint8_t>& truncated) {
  const auto n = static_cast<int32_t>(states.size());
  const size_t d = static_cast<size_t>(obs_dim());
  display_cache_.resize(static_cast<size_t>(n));
  pos_cache_.resize(static_cast<size_t>(n) * 6);
  tiles_.resize(static_cast<size_t>(n) * kTileBytes);
  float* obs_base = observations.data();

  // Dispatch in waves: one walker per worker at a time. Walker i is always
  // handled by worker i % n_workers (deterministic mapping).
  for (int32_t wave_start = 0; wave_start < n; wave_start += n_workers_) {
    const int32_t wave_end =
        wave_start + n_workers_ < n ? wave_start + n_workers_ : n;
    for (int32_t i = wave_start; i < wave_end; ++i) {
      const int slot = i - wave_start;
      const auto ui = static_cast<size_t>(i);
      int32_t* h = header(slot);
      std::memcpy(blob_area(slot), states[ui].data(), blob_len_);
      atomic_store_i32(h + kAction, actions[ui]);
      atomic_store_i32(h + kDt, dt[ui]);
      post_command(slot, kCmdStep);
    }
    for (int32_t i = wave_start; i < wave_end; ++i) {
      const int slot = i - wave_start;
      const auto ui = static_cast<size_t>(i);
      wait_idle(slot, "step");
      int32_t* h = header(slot);
      new_states[ui].assign(blob_area(slot), blob_area(slot) + blob_len_);
      std::memcpy(obs_base + ui * d, obs_area(slot), d * sizeof(float));
      float reward_bits;
      std::memcpy(&reward_bits, h + kReward, sizeof(float));
      rewards[ui] = reward_bits;
      dones[ui] = atomic_load_i32(h + kDone) ? 1 : 0;
      float display_bits;
      std::memcpy(&display_bits, h + kDisplay, sizeof(float));
      display_cache_[ui] = display_bits;
      for (int k = 0; k < 6; ++k) {
        pos_cache_[ui * 6 + k] = atomic_load_i32(h + kX + k);
      }
      std::memcpy(tiles_.data() + ui * kTileBytes, tile_area(slot),
                  kTileBytes);
      truncated[ui] = 0;
    }
  }
}

float RetroFarmEnv::display_score(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui] : 0.0f;
}

void RetroFarmEnv::render_frame(const std::vector<char>& state,
                                std::vector<uint8_t>& rgba) {
  std::memcpy(blob_area(0), state.data(), blob_len_);
  post_command(0, kCmdRender);
  wait_idle(0, "render");
  rgba.assign(rgba_area(0),
              rgba_area(0) + static_cast<size_t>(kRetroFrameWidth) *
                                 static_cast<size_t>(kRetroFrameHeight) * 4);
}

}  // namespace fg

#endif  // __EMSCRIPTEN__
