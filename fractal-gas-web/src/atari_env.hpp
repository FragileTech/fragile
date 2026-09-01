// AtariEnv — the plangym-equivalent batch environment backed by the Arcade
// Learning Environment (ALE / Stella). One ale::ALEInterface per thread-pool
// slot, all loading the same ROM; walker states travel as the byte string
// from ALEState::serialize() (obtained via cloneSystemState(), i.e. the full
// system state INCLUDING the pseudo-random number generator) with a 4-byte
// float carry appended (the cumulative episode score, used only for the
// display ranking). Any emulator instance can resume any walker by
// reconstructing an ALEState from the string and calling
// restoreSystemState(), so blobs are portable across instances.
#ifndef FRACTAL_GAS_ATARI_ENV_HPP
#define FRACTAL_GAS_ATARI_ENV_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "env.hpp"
#include "thread_pool.hpp"

namespace ale {
class ALEInterface;
}

namespace fg {

/// What the env exposes as the walker observation (the vector the fractal
/// gas measures L2 distances on). Rewards always come from ALE's per-game
/// score detection and are unaffected by the mode.
///
/// kCoords: Atari has no generic game-state tuple (RAM layout is per-game),
/// so kCoords falls back to kRam — the 128 RAM bytes ARE the compact state.
enum class AtariObsMode : int32_t {
  kRam = 0,     // the 128 bytes of Atari 2600 RAM, one float per byte
  kRgb = 1,     // the screen as RGB floats (width*height*3 dims)
  kGray = 2,    // the screen as grayscale floats (width*height dims)
  kCoords = 3,  // alias for kRam (see above)
};

class AtariEnv final : public BatchEnv {
 public:
  /// Throws std::runtime_error on an unreadable ROM file. NOTE: ALE itself
  /// calls std::exit(1) on a ROM whose MD5 does not match any supported
  /// game and whose filename cannot be mapped to one — use snake_case names
  /// of supported games (e.g. "breakout.bin").
  AtariEnv(std::string rom_path, int n_threads,
           AtariObsMode obs_mode = AtariObsMode::kRam);
  ~AtariEnv() override;

  int32_t n_actions() const override;
  int32_t obs_dim() const override;
  AtariObsMode obs_mode() const { return obs_mode_; }

  void reset(std::vector<char>& state, std::vector<float>& obs) override;

  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions,
                  const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& new_states,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override;

  /// Showcase ranking: the cumulative episode score (sum of ALE rewards
  /// since reset), carried inside each walker blob and cached per walker
  /// during step_batch.
  bool has_display_score() const override { return true; }
  float display_score(int32_t walker_index) const override;

  /// A life loss with lives remaining is a recoverable death: the walker's
  /// done fires (so the swarm avoids losing lives) but FractalGas may revive
  /// it when every walker is dead. Games without a life counter (ALE
  /// lives() == 0) never soft-die.
  bool has_recoverable_dones() const override { return true; }
  bool done_is_recoverable(int32_t walker_index) const override;

  void render_frame(const std::vector<char>& state,
                    std::vector<uint8_t>& rgba) override;
  int32_t frame_width() const override { return screen_w_; }
  int32_t frame_height() const override { return screen_h_; }

  int threads() const { return pool_.size(); }
  ThreadPool* worker_pool() override { return &pool_; }

 private:
  void step_one(int slot, const std::vector<char>& blob, int32_t action,
                int32_t dt, std::vector<char>& new_blob, float* obs_row,
                float& reward, uint8_t& done, uint8_t& trunc, float& display,
                uint8_t& recoverable);
  void fill_obs(int slot, float* obs_row);
  std::vector<char> blob_from(ale::ALEInterface& a, float episode_return);
  void restore_blob(ale::ALEInterface& a, const std::vector<char>& blob);
  static float read_carry(const std::vector<char>& blob);

  std::string rom_path_;
  AtariObsMode obs_mode_;
  ThreadPool pool_;
  std::vector<std::unique_ptr<ale::ALEInterface>> emulators_;  // one per slot
  std::vector<int32_t> action_set_;  // minimal action set (ale::Action values)
  int32_t screen_w_ = 0;
  int32_t screen_h_ = 0;
  std::vector<std::vector<unsigned char>> scratch_;  // per-slot pixel buffer
  std::vector<float> display_cache_;                 // per walker
  std::vector<uint8_t> recoverable_cache_;           // per walker

  // First reset() caches the post-reset_game() state; later resets return
  // the cache (same pattern as NesMarioEnv).
  std::vector<char> initial_state_;
  std::vector<float> initial_obs_;
  bool has_initial_ = false;
};

}  // namespace fg

#endif  // FRACTAL_GAS_ATARI_ENV_HPP
