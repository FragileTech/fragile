// NesMarioEnv — the plangym-equivalent batch environment backed by the
// nes-py C++ emulator core. One NES::Emulator instance per thread-pool slot;
// walker states travel as dump_state() blobs with the Mario reward carry
// appended, so cloning a walker is a byte copy and any emulator instance can
// resume any walker via load_state().
#ifndef FRACTAL_GAS_NES_ENV_HPP
#define FRACTAL_GAS_NES_ENV_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "env.hpp"
#include "mario_reward.hpp"
#include "thread_pool.hpp"

namespace NES {
class Emulator;
}

namespace fg {

/// gym-super-mario-bros COMPLEX_MOVEMENT action list, encoded with nes-py's
/// controller bit map (A=0x01, B=0x02, select=0x04, start=0x08, up=0x10,
/// down=0x20, left=0x40, right=0x80):
///   NOOP, right, right+A, right+B, right+A+B, A,
///   left, left+A, left+B, left+A+B, down, up
extern const uint8_t kMarioActionMasks[12];

/// What the env exposes as the walker observation (the vector the fractal
/// gas measures L2 distances on). The reward always reads RAM directly and
/// is unaffected by the mode.
enum class ObsMode : int32_t {
  kRam = 0,     // the 2KB CPU RAM, one float per byte (2048 dims)
  kRgb = 1,     // the 256x240 screen as RGB floats (184320 dims)
  kGray = 2,    // the 256x240 screen as luminance floats (61440 dims)
  kCoords = 3,  // fast game-state tuple, see kCoordsDim below
};

/// Coords layout: [x, y_pixel, y_viewport, world, stage, time, h_velocity,
/// v_velocity, sub_area, power_up]. Raw values (no per-dimension
/// normalization, matching the algorithm's raw-observation convention).
constexpr int32_t kCoordsDim = 10;

class NesMarioEnv final : public BatchEnv {
 public:
  /// Throws std::runtime_error on an unreadable/invalid/unsupported ROM.
  /// start_world (1-8) / start_stage (1-4) select the SMB level the run
  /// begins in (gym-super-mario-bros style RAM write during boot).
  NesMarioEnv(std::string rom_path, int n_threads, ObsMode obs_mode = ObsMode::kRam,
              int32_t start_world = 1, int32_t start_stage = 1);
  ~NesMarioEnv() override;

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

  /// Showcase ranking: highest (world, stage) first, then x-position within
  /// the level. Computed from RAM during step_batch (independent of the
  /// observation mode) and cached per walker.
  bool has_display_score() const override { return true; }
  float display_score(int32_t walker_index) const override;

  /// World/stage of a walker from the last step_batch (1-indexed for display).
  int32_t walker_world(int32_t walker_index) const;
  int32_t walker_stage(int32_t walker_index) const;

  /// Level x-position (world pixels) / on-screen y-pixel of a walker from
  /// the last step_batch — feeds the level-map swarm overlay in the web UI.
  int32_t walker_x(int32_t walker_index) const;
  int32_t walker_y(int32_t walker_index) const;

  void render_frame(const std::vector<char>& state,
                    std::vector<uint8_t>& rgba) override;
  int32_t frame_width() const override;
  int32_t frame_height() const override;

  int threads() const { return pool_.size(); }
  ThreadPool* worker_pool() override { return &pool_; }

 private:
  struct DisplayInfo {
    float score = 0.0f;
    uint8_t world = 0;
    uint8_t stage = 0;
    int32_t x = 0;   // level x-position in world pixels
    uint8_t y = 0;   // player y-pixel on screen (ram[0x3B8])
  };

  void step_one(int slot, const std::vector<char>& blob, int32_t action,
                int32_t dt, std::vector<char>& new_blob, float* obs_row,
                float& reward, uint8_t& done, DisplayInfo& display);
  void fill_obs(NES::Emulator& emu, float* obs_row) const;
  std::vector<char> dump_with_carry(NES::Emulator& emu, const MarioCarry& carry);
  static MarioCarry read_carry(const std::vector<char>& blob);

  std::string rom_path_;
  ObsMode obs_mode_;
  int32_t start_world_;
  int32_t start_stage_;
  ThreadPool pool_;
  std::vector<std::unique_ptr<NES::Emulator>> emulators_;  // one per pool slot
  std::vector<DisplayInfo> display_cache_;                 // per walker

  // First reset() boots the game and caches the initial state; later resets
  // return the cache (nes-py's _backup()/_restore() pattern). Re-booting on
  // a used emulator is a warm boot with stale RAM and is not reliable.
  std::vector<char> initial_state_;
  std::vector<float> initial_obs_;
  bool has_initial_ = false;
};

}  // namespace fg

#endif  // FRACTAL_GAS_NES_ENV_HPP
