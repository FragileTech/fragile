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
#include "swarm_algorithm.hpp"
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

class FractalGas final : public SwarmAlgorithm {
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
  void set_dist_coef(float v) override {
    params_.dist_coef = v;
    clone_op_->dist_coef = v;
  }
  void set_reward_coef(float v) override {
    params_.reward_coef = v;
    clone_op_->reward_coef = v;
  }
  void set_use_cumulative_reward(bool v) override {
    params_.use_cumulative_reward = v;
    clone_op_->use_cumulative_reward = v;
  }
  void set_dt_range(int32_t lo, int32_t hi) override {
    params_.dt_min = lo;
    params_.dt_max = hi;
    kinetic_op_->dt_min = lo;
    kinetic_op_->dt_max = hi;
  }
  void set_n_elite(int32_t k) override {
    params_.n_elite = k;
    if (k <= 0) {
      has_elite_ = false;
      elite_walkers_ = WalkerState{};
    }
  }

  /// FractalGas.reset(): env reset, replicate the initial state N times,
  /// zero all walker arrays, clear metrics and the elite buffer.
  void reset() override;

  /// One iteration of the algorithm, preserving the Python phase order.
  StepInfo step() override;

  std::vector<StepInfo> run(int32_t max_iterations,
                            bool stop_when_all_dead = false);

  /// (index, cumulative reward) of the best walker.
  std::pair<int32_t, float> get_best_walker() const override;

  /// RGBA frame of the best walker from the last step (record_frames only).
  const std::vector<uint8_t>& best_frame() const override { return best_frame_; }

  int64_t total_steps() const override { return total_steps_; }
  int64_t total_clones() const override { return total_clones_; }
  int32_t iteration_count() const override { return iteration_count_; }

  // SwarmAlgorithm population access. The env's batch cache is valid by
  // walker index here because every walker steps each iteration.
  int32_t n_walkers() const override { return state_.N; }
  const std::vector<char>& walker_state(int32_t i) const override {
    return state_.states[static_cast<size_t>(i)];
  }
  bool walker_alive(int32_t i) const override { return state_.alive(i); }
  float walker_cum_reward(int32_t i) const override {
    return state_.rewards[static_cast<size_t>(i)];
  }
  bool has_walker_info() const override { return env_.has_walker_info(); }
  const WalkerInfo& walker_info(int32_t i) const override {
    return env_.walker_info(i);
  }

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
