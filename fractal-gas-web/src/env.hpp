// Batch environment interface — the C++ equivalent of the plangym surface the
// Python FractalGas relies on (reset(return_state=True) + step_batch).
// Environment states are opaque per-walker byte blobs.
#ifndef FRACTAL_GAS_ENV_HPP
#define FRACTAL_GAS_ENV_HPP

#include <cstdint>
#include <vector>

namespace fg {

class ThreadPool;

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

  /// Render an RGBA frame (frame_width*frame_height*4 bytes) for a state
  /// blob. May leave `rgba` empty when the env has no visual output.
  virtual void render_frame(const std::vector<char>& state,
                            std::vector<uint8_t>& rgba) = 0;
  virtual int32_t frame_width() const = 0;
  virtual int32_t frame_height() const = 0;
};

}  // namespace fg

#endif  // FRACTAL_GAS_ENV_HPP
