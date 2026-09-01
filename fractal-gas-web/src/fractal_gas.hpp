// Translation of the FractalGas base class
// (src/fragile/fractalai/fractal_gas.py): reset / step / run, elite buffer,
// per-iteration metrics, best-walker frame recording.
#ifndef FRACTAL_GAS_FRACTAL_GAS_HPP
#define FRACTAL_GAS_FRACTAL_GAS_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "cloning.hpp"
#include "env.hpp"
#include "kinetic.hpp"
#include "rng.hpp"
#include "walker_state.hpp"

namespace fg {

struct FractalGasParams {
  int32_t N = 32;
  float dist_coef = 1.0f;
  float reward_coef = 1.0f;
  bool use_cumulative_reward = false;
  int32_t dt_min = 1;
  int32_t dt_max = 4;
  int32_t n_elite = 0;
  bool record_frames = false;
  uint64_t seed = 0;
};

/// The info dict emitted by FractalGas.step() (tensor-valued diagnostic
/// entries from the Python version are omitted).
struct StepInfo {
  int32_t iteration = 0;
  int32_t num_cloned = 0;
  int32_t num_revived = 0;
  int32_t alive_count = 0;
  float mean_reward = 0.0f;
  float max_reward = 0.0f;
  float min_reward = 0.0f;
  float mean_virtual_reward = 0.0f;
  float max_virtual_reward = 0.0f;
  float min_virtual_reward = 0.0f;
  float mean_dt = 0.0f;
  int32_t min_dt = 0;
  int32_t max_dt = 0;
  int32_t best_walker_idx = 0;
};

class FractalGas {
 public:
  /// The operators are owned; passing subclasses (with overridden sampling)
  /// enables replay tests. Defaults reproduce the Python configuration.
  FractalGas(BatchEnv& env, FractalGasParams params,
             std::unique_ptr<Rng> rng = nullptr,
             std::unique_ptr<FractalCloningOperator> clone_op = nullptr,
             std::unique_ptr<RandomActionOperator> kinetic_op = nullptr);

  const FractalGasParams& params() const { return params_; }
  const WalkerState& state() const { return state_; }

  // Live-tunable parameters (used by the web demo's sidebar).
  void set_dist_coef(float v) {
    params_.dist_coef = v;
    clone_op_->dist_coef = v;
  }
  void set_reward_coef(float v) {
    params_.reward_coef = v;
    clone_op_->reward_coef = v;
  }
  void set_use_cumulative_reward(bool v) {
    params_.use_cumulative_reward = v;
    clone_op_->use_cumulative_reward = v;
  }
  void set_dt_range(int32_t lo, int32_t hi) {
    params_.dt_min = lo;
    params_.dt_max = hi;
    kinetic_op_->dt_min = lo;
    kinetic_op_->dt_max = hi;
  }
  void set_n_elite(int32_t k) {
    params_.n_elite = k;
    if (k <= 0) {
      has_elite_ = false;
      elite_walkers_ = WalkerState{};
    }
  }

  /// FractalGas.reset(): env reset, replicate the initial state N times,
  /// zero all walker arrays, clear metrics and the elite buffer.
  void reset();

  /// One iteration of the algorithm, preserving the Python phase order.
  StepInfo step();

  std::vector<StepInfo> run(int32_t max_iterations,
                            bool stop_when_all_dead = false);

  /// (index, cumulative reward) of the best walker.
  std::pair<int32_t, float> get_best_walker() const;

  /// RGBA frame of the best walker from the last step (record_frames only).
  const std::vector<uint8_t>& best_frame() const { return best_frame_; }

  int64_t total_steps() const { return total_steps_; }
  int64_t total_clones() const { return total_clones_; }
  int32_t iteration_count() const { return iteration_count_; }

 private:
  void update_elites();

  BatchEnv& env_;
  FractalGasParams params_;
  std::unique_ptr<Rng> rng_;
  std::unique_ptr<FractalCloningOperator> clone_op_;
  std::unique_ptr<RandomActionOperator> kinetic_op_;

  WalkerState state_;
  WalkerState elite_walkers_;
  bool has_elite_ = false;

  std::vector<uint8_t> best_frame_;

  int64_t total_steps_ = 0;
  int64_t total_clones_ = 0;
  int32_t iteration_count_ = 0;
};

}  // namespace fg

#endif  // FRACTAL_GAS_FRACTAL_GAS_HPP
