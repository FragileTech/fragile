// RetroFarmEnv — wasm-only Genesis batch environment that runs N
// INDEPENDENT instantiations of the statically-linked GPGX shim module
// (src/retro_shim.cpp), one per plain Web Worker, and dispatches per-walker
// step jobs to them through the main module's shared memory + Atomics.
//
// Why: libretro cores are global-state singletons, and emscripten's dynamic
// linking (dlopen'd side-module copies) faults under pthreads. Separate
// module instantiations in separate workers give true isolation and true
// parallelism with no fragile machinery. Each job is a pure function of
// (blob, action, dt): the worker copies the blob out of shared memory,
// steps its core, and copies the results back.
//
// Protocol per worker slot (a region in this module's heap, which is a
// SharedArrayBuffer): a ctrl word the C++ side sets to a command code
// (STEP/BOOT/RENDER) and wakes the worker's Atomics.wait; the worker
// processes and stores 0 (or a negative error) back. The C++ side waits by
// polling (no cross-runtime futex assumptions).
#ifndef FRACTAL_GAS_RETRO_FARM_ENV_HPP
#define FRACTAL_GAS_RETRO_FARM_ENV_HPP
#ifdef __EMSCRIPTEN__

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "env.hpp"
#include "retro_game_logic.hpp"

namespace fg {

class RetroFarmEnv final : public BatchEnv {
 public:
  /// Adopts a farm that worker.js has ALREADY spawned and awaited (nested
  /// workers can only be spawned while the JS event loop is alive, which it
  /// is not once C++ blocks): regions_ptr = base of n_workers job regions
  /// in this module's heap, blob_len = serialize size + carry as reported
  /// by the workers. JS owns the workers and the region memory.
  RetroFarmEnv(uintptr_t regions_ptr, int n_workers, size_t blob_len,
               int32_t obs_mode, RetroGame game);
  ~RetroFarmEnv() override = default;

  int32_t n_actions() const override { return kRetroNumActions; }
  int32_t obs_dim() const override { return retro_obs_dim(game_, obs_mode_); }

  void reset(std::vector<char>& state, std::vector<float>& obs) override;

  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions,
                  const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& new_states,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override;

  bool has_display_score() const override { return true; }
  float display_score(int32_t walker_index) const override;

  /// Fog-of-war swarm map data, cached per walker from the last step_batch.
  static constexpr int32_t kTileW = 40, kTileH = 28;  // frame / 8, RGB
  static constexpr size_t kTileBytes = kTileW * kTileH * 3;
  int32_t walker_x(int32_t i) const { return pos_cache_[i * 6 + 0]; }
  int32_t walker_y(int32_t i) const { return pos_cache_[i * 6 + 1]; }
  int32_t walker_zone(int32_t i) const { return pos_cache_[i * 6 + 2]; }
  int32_t walker_act(int32_t i) const { return pos_cache_[i * 6 + 3]; }
  int32_t walker_cam_x(int32_t i) const { return pos_cache_[i * 6 + 4]; }
  int32_t walker_cam_y(int32_t i) const { return pos_cache_[i * 6 + 5]; }
  int32_t walker_count() const {
    return static_cast<int32_t>(pos_cache_.size() / 6);
  }
  const uint8_t* walker_tiles() const { return tiles_.data(); }
  size_t walker_tiles_size() const { return tiles_.size(); }

  /// Generic per-walker info (visit key: plane = zone*16 + act, cell =
  /// level x/y; Coords mode only).
  bool has_walker_info() const override { return true; }
  const WalkerInfo& walker_info(int32_t batch_index) const override;
  bool has_visit_key() const override { return obs_mode_ == 3; }
  int32_t frames_stepped(int32_t batch_index) const override {
    const auto ui = static_cast<size_t>(batch_index);
    return ui < frames_cache_.size() ? frames_cache_[ui] : -1;
  }

  void render_frame(const std::vector<char>& state,
                    std::vector<uint8_t>& rgba) override;
  int32_t frame_width() const override { return kRetroFrameWidth; }
  int32_t frame_height() const override { return kRetroFrameHeight; }

  int workers() const { return n_workers_; }

  /// Live-tunable Sonic reward term weights: written into every slot's
  /// header words; each core worker picks them up on its next STEP job.
  void set_reward_weights(const SonicRewardWeights& w);
  const SonicRewardWeights& reward_weights() const { return reward_weights_; }

 private:
  // Region layout (bytes from region base). Header words are int32/float32.
  static constexpr size_t kHeaderBytes = 128;
  static constexpr size_t kBlobCap = 0x200000;   // 2MB >= STATE_SIZE + carry
  static constexpr size_t kObsCap = 0x100000;    // 1MB >= RGB obs floats
  static constexpr size_t kRgbaCap = 0x80000;    // 512KB >= 320*224*4
  static constexpr size_t kTileCap = 0x1000;     // 4KB >= 40*28*3 fog tile
  static constexpr size_t kRegionSize =
      kHeaderBytes + kBlobCap + kObsCap + kRgbaCap + kTileCap;

  // Header word indices (int32).
  enum : int { kCtrl = 0, kStatus = 1, kAction = 2, kDt = 3, kDone = 4,
               kBlobLen = 5, kReward = 6, kDisplay = 7,
               // Fog-of-war swarm map: player position, level, camera.
               kX = 8, kY = 9, kZone = 10, kAct = 11, kCamX = 12,
               kCamY = 13,
               // Frames the core really emulated in the last STEP job.
               kFrames = 14,
               // Reward term weights: count, then kSonicRewardWeightCount
               // float32 words (SonicRewardWeights field order).
               kWeightCount = 16, kWeights = 17 };
  // Ctrl commands (worker stores 0 back when done, negative on error).
  enum : int32_t { kCmdStep = 1, kCmdBoot = 2, kCmdRender = 3 };

  uint8_t* region(int slot) const {
    return regions_ + static_cast<size_t>(slot) * kRegionSize;
  }
  int32_t* header(int slot) const {
    return reinterpret_cast<int32_t*>(region(slot));
  }
  uint8_t* blob_area(int slot) const { return region(slot) + kHeaderBytes; }
  uint8_t* obs_area(int slot) const {
    return region(slot) + kHeaderBytes + kBlobCap;
  }
  uint8_t* rgba_area(int slot) const {
    return region(slot) + kHeaderBytes + kBlobCap + kObsCap;
  }
  uint8_t* tile_area(int slot) const {
    return region(slot) + kHeaderBytes + kBlobCap + kObsCap + kRgbaCap;
  }

  void post_command(int slot, int32_t cmd);
  void wait_idle(int slot, const char* what);

  RetroGame game_;
  int32_t obs_mode_;
  int n_workers_;
  uint8_t* regions_ = nullptr;
  size_t blob_len_ = 0;  // serialize size + carry, reported by the workers
  SonicRewardWeights reward_weights_;

  std::vector<float> display_cache_;
  std::vector<int32_t> pos_cache_;  // 6 ints per walker (see accessors)
  std::vector<WalkerInfo> info_cache_;
  std::vector<int32_t> frames_cache_;
  std::vector<uint8_t> tiles_;      // kTileBytes per walker

  std::vector<char> initial_state_;
  std::vector<float> initial_obs_;
  bool has_initial_ = false;
};

}  // namespace fg

#endif  // __EMSCRIPTEN__
#endif  // FRACTAL_GAS_RETRO_FARM_ENV_HPP
