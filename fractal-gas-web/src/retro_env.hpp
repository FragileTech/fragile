// RetroEnv — Sega Genesis batch environment backed by the Genesis Plus GX
// libretro core (vendored by the stable-retro submodule), with stable-retro's
// Airstriker-Genesis integration data hardcoded for reward/termination.
//
// One RetroCore (a private dlopen'd copy of the core .so — see
// retro_core.hpp) per thread-pool slot; walker states travel as
// retro_serialize blobs with a reward-carry tail (last score), so cloning a
// walker is a byte copy and any core instance can resume any walker.
#ifndef FRACTAL_GAS_RETRO_ENV_HPP
#define FRACTAL_GAS_RETRO_ENV_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "env.hpp"
#include "nes_env.hpp"  // reuses the fg::ObsMode enum (kRam/kRgb/kGray/kCoords)
#include "retro_core.hpp"
#include "retro_game_logic.hpp"  // RetroGame, action masks, dims, per-game math
#include "thread_pool.hpp"

namespace fg {

class RetroEnv final : public BatchEnv {
 public:
  /// core_so_path: the built genesis_plus_gx_libretro.so (each instance
  /// dlopens its own temp copy; ignored in FG_RETRO_STATIC builds).
  /// rom_path: the game ROM.
  /// state_path: gzipped stable-retro savestate to start from. Pass "" to
  /// instead BOOT the game from power-on and auto-navigate to gameplay
  /// (tapping START until the game is running) — required in wasm, where
  /// stable-retro's x86-64-written savestates cannot be restored (GPGX
  /// serializes whole structs containing pointers, so the byte layout is
  /// host-ABI-dependent; walker blobs are unaffected since they are written
  /// and read by the same build).
  /// Throws std::runtime_error when the inputs cannot be used.
  RetroEnv(std::string core_so_path, std::string rom_path,
           std::string state_path, int n_threads,
           ObsMode obs_mode = ObsMode::kRam,
           RetroGame game = RetroGame::kAirstriker);
  ~RetroEnv() override;

  int32_t n_actions() const override;
  int32_t obs_dim() const override;
  ObsMode obs_mode() const { return obs_mode_; }

  void reset(std::vector<char>& state, std::vector<float>& obs) override;

  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions,
                  const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& new_states,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override;

  /// Showcase ranking: the game score (read from work RAM during step_batch,
  /// independent of the observation mode).
  bool has_display_score() const override { return true; }
  float display_score(int32_t walker_index) const override;

  void render_frame(const std::vector<char>& state,
                    std::vector<uint8_t>& rgba) override;
  int32_t frame_width() const override { return kRetroFrameWidth; }
  int32_t frame_height() const override { return kRetroFrameHeight; }

  int threads() const { return pool_->size(); }
  ThreadPool* worker_pool() override { return pool_.get(); }


 private:
  // The per-game math (reward carry, frame stepping, boot, obs) lives in
  // retro_game_logic.hpp, shared with the wasm core-worker shim.
  void step_one(int slot, const std::vector<char>& blob, int32_t action,
                int32_t dt, std::vector<char>& new_blob, float* obs_row,
                float& reward, uint8_t& done, float& display);
  std::vector<char> dump_with_carry(RetroCore& core, const RetroCarry& carry);
  static RetroCarry read_carry(const std::vector<char>& blob);
  void load_blob(RetroCore& core, const std::vector<char>& blob);

  std::string core_so_path_;
  std::string rom_path_;
  std::string state_path_;
  ObsMode obs_mode_;
  RetroGame game_;
  // ORDER MATTERS: the cores are dlopen'd BEFORE the pool threads are
  // spawned. Under emscripten, a thread blocked in a condvar never returns
  // to the JS event loop and therefore never synchronizes libraries loaded
  // after it started — calling side-module code from such a thread faults.
  // Threads created after the dlopens start with the libraries loaded.
  std::vector<std::unique_ptr<RetroCore>> cores_;  // one per pool slot
  std::unique_ptr<ThreadPool> pool_;
  std::vector<float> display_cache_;               // per walker

  size_t serialize_size_ = 0;  // constant for a loaded ROM (STATE_SIZE)

  // First reset() restores Level1.state and caches the resulting blob;
  // later resets return the cache.
  std::vector<char> initial_state_;
  std::vector<float> initial_obs_;
  bool has_initial_ = false;
};

}  // namespace fg

#endif  // FRACTAL_GAS_RETRO_ENV_HPP
