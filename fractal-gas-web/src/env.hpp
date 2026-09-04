// Batch environment interface — the C++ equivalent of the plangym surface the
// Python FractalGas relies on (reset(return_state=True) + step_batch).
// Environment states are opaque per-walker byte blobs.
#ifndef FRACTAL_GAS_ENV_HPP
#define FRACTAL_GAS_ENV_HPP

#include <cstdint>
#include <vector>

namespace fg {

class ThreadPool;

/// Per-walker game information an env can expose after step_batch, indexed
/// by BATCH position (like display_score). Consumed by the algorithms for
/// the showcase ranking, the map overlays in the web UI and the tree's
/// visit-count reward. Fields a game does not have stay 0.
struct WalkerInfo {
  float score = 0.0f;      // showcase ranking (== display_score)
  int32_t x = 0;           // map position: Mario level x / Montezuma in-room px / Sonic level x
  int32_t y = 0;           // Mario screen y / Montezuma in-room py / Sonic level y
  int32_t world = 0;       // Mario world (1-based) / Montezuma room / Sonic zone
  int32_t stage = 0;       // Mario stage (1-based) / Montezuma level / Sonic act
  int32_t cam_x = 0;       // Sonic camera (fog-of-war tiles)
  int32_t cam_y = 0;
  int32_t lives = 0;       // Montezuma extras
  int32_t inventory = 0;
  // Visit-count key for the tree algorithm (valid iff has_visit_key): a
  // plane id (level / room) and an integer cell position inside it.
  bool has_visit_key = false;
  int32_t visit_plane = 0;
  int32_t visit_x = 0;
  int32_t visit_y = 0;
};

class BatchEnv {
 public:
  virtual ~BatchEnv() = default;

  virtual int32_t n_actions() const = 0;
  virtual int32_t obs_dim() const = 0;

  /// The env's worker pool, when it owns one, so the algorithm can reuse
  /// its threads between step_batch calls (they idle during the fitness
  /// phase). nullptr means "compute serially".
  virtual ThreadPool* worker_pool() { return nullptr; }

  /// Reset the environment; writes the initial state blob and observation
  /// (obs_dim floats). Equivalent of env.reset(return_state=True).
  virtual void reset(std::vector<char>& state, std::vector<float>& obs) = 0;

  /// Step every walker: apply actions[i] for dt[i] env frames starting from
  /// states[i]. Outputs are pre-sized by the caller:
  ///   new_states [N], observations [N*obs_dim], rewards/dones/truncated [N].
  /// Rewards are summed over the dt frames (plangym frame-skip semantics,
  /// stopping early when the episode ends).
  virtual void step_batch(const std::vector<std::vector<char>>& states,
                          const std::vector<int32_t>& actions,
                          const std::vector<int32_t>& dt,
                          std::vector<std::vector<char>>& new_states,
                          std::vector<float>& observations,
                          std::vector<float>& rewards,
                          std::vector<uint8_t>& dones,
                          std::vector<uint8_t>& truncated) = 0;

  /// Optional score used ONLY to pick the walker shown in the demo (frame
  /// recording) — it does not enter the algorithm. When has_display_score()
  /// is false the caller falls back to cumulative reward. Indexed by walker;
  /// the env caches whatever it needs during step_batch (this keeps the
  /// showcase independent of the observation mode).
  virtual bool has_display_score() const { return false; }
  virtual float display_score(int32_t /*walker_index*/) const { return 0.0f; }

  /// Optional: marks a walker's done as a recoverable "soft death" (e.g. an
  /// Atari life loss with the game still playable). Cached per walker during
  /// step_batch. FractalGas revives soft-dead walkers only when the whole
  /// swarm is dead; hard game-overs stay dead.
  virtual bool has_recoverable_dones() const { return false; }
  virtual bool done_is_recoverable(int32_t /*walker_index*/) const {
    return false;
  }

  /// Optional per-walker info of batch position i of the LAST step_batch
  /// (see WalkerInfo). Algorithms that step only a subset of their walkers
  /// copy it per walker right after the batch.
  virtual bool has_walker_info() const { return false; }
  virtual const WalkerInfo& walker_info(int32_t /*batch_index*/) const {
    static const WalkerInfo kEmpty{};
    return kEmpty;
  }
  /// True when walker_info() fills the visit-count key in the env's CURRENT
  /// observation mode (the demo counts visits only on Coords tuples).
  virtual bool has_visit_key() const { return false; }

  /// Frames actually emulated for batch position i of the LAST step_batch
  /// (envs stop a step early on death / game over), or -1 when the env does
  /// not track it and the requested dt should be assumed.
  virtual int32_t frames_stepped(int32_t /*batch_index*/) const { return -1; }

  /// Render an RGBA frame (frame_width*frame_height*4 bytes) for a state
  /// blob. May leave `rgba` empty when the env has no visual output.
  virtual void render_frame(const std::vector<char>& state,
                            std::vector<uint8_t>& rgba) = 0;
  virtual int32_t frame_width() const = 0;
  virtual int32_t frame_height() const = 0;
};

}  // namespace fg

#endif  // FRACTAL_GAS_ENV_HPP
