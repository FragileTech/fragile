// Common surface of the two swarm algorithms the demo can run:
//   - "Wave"  = FractalGas  (src/fractal_gas.hpp): fixed population, every
//               walker steps each iteration, elite buffer.
//   - "Graph" = FractalTree (src/fractal_tree.hpp): growing tree of states,
//               only cloning leaves step, parents recorded.
// The bindings, the native CLI and the UI only talk to this interface. The
// two implementations are independent of each other (they share nothing but
// the pure tensor helpers, exactly like the two Python references).
#ifndef FRACTAL_GAS_SWARM_ALGORITHM_HPP
#define FRACTAL_GAS_SWARM_ALGORITHM_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include "env.hpp"
#include "visit_grid.hpp"

namespace fg {

/// The info dict emitted by one iteration of either algorithm (tensor-valued
/// diagnostic entries from the Python versions are omitted).
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
  // Population bookkeeping. Wave: all three equal N. Graph: the live tree
  // size after the step, its leaf count (the mask that drove the step) and
  // the number of walkers that actually ran the environment.
  int32_t n_walkers = 0;
  int32_t n_leaves = 0;
  int32_t num_stepped = 0;
};

class SwarmAlgorithm {
 public:
  virtual ~SwarmAlgorithm() = default;

  virtual void reset() = 0;
  virtual StepInfo step() = 0;

  /// Live population size; walker indices below are in [0, n_walkers()).
  virtual int32_t n_walkers() const = 0;
  /// Env state blob of a walker (may be EMPTY for a tree slot that has not
  /// been stepped yet).
  virtual const std::vector<char>& walker_state(int32_t i) const = 0;
  virtual bool walker_alive(int32_t i) const = 0;
  virtual float walker_cum_reward(int32_t i) const = 0;
  /// Tree structure: parent index (self for the root and for the wave) and
  /// whether the walker is a leaf.
  virtual int32_t walker_parent(int32_t i) const { return i; }
  virtual bool walker_is_leaf(int32_t /*i*/) const { return true; }
  /// Per-WALKER game info (valid by walker index, unlike the env's
  /// batch-indexed cache).
  virtual bool has_walker_info() const = 0;
  virtual const WalkerInfo& walker_info(int32_t i) const = 0;

  /// (index, cumulative reward) of the best walker.
  virtual std::pair<int32_t, float> get_best_walker() const = 0;
  /// RGBA frame of the showcased walker from the last step (when frames are
  /// recorded).
  virtual const std::vector<uint8_t>& best_frame() const = 0;

  /// Visit-count grid (per-pixel occupancy), when the algorithm counts
  /// visits in the current setup (Coords mode on a game with a map).
  virtual bool counting_visits() const { return false; }
  virtual const VisitGrid* visit_grid() const { return nullptr; }

  virtual int64_t total_steps() const = 0;
  virtual int64_t total_clones() const = 0;
  virtual int32_t iteration_count() const = 0;

  // Live-tunable parameters. Each algorithm ignores the ones it lacks.
  virtual void set_dist_coef(float v) = 0;
  virtual void set_reward_coef(float v) = 0;
  virtual void set_dt_range(int32_t lo, int32_t hi) = 0;
  virtual void set_use_cumulative_reward(bool /*v*/) {}
  virtual void set_n_elite(int32_t /*k*/) {}
  virtual void set_erase_coef(float /*v*/) {}
  /// Graph: pooling window (pixels per side) of the visit-count reward.
  virtual void set_agg_block_size(int32_t /*b*/) {}
  /// Graph: whether the visit-count term multiplies into the virtual reward
  /// (ablation switch; counting itself continues).
  virtual void set_visit_reward(bool /*on*/) {}
  /// Exponent of the visit-count term (1 = the reference's plain product).
  virtual void set_visit_coef(float /*c*/) {}
};

}  // namespace fg

#endif  // FRACTAL_GAS_SWARM_ALGORITHM_HPP
