// FractalTree ("Graph") — faithful translation of the tree variant of the
// fractal gas used by the old Montezuma demo: FractalTree / MontezumaTree at
// commit cb9f3296 (src/fragile/core.py, src/fragile/videogames.py,
// src/fragile/fractalai.py, src/fragile/actions.py).
//
// Unlike the wave (every walker steps each iteration), the tree keeps every
// visited state as a node: each iteration the LEAVES that decide to clone
// copy a companion's state, record it as their parent and step the env from
// there; the walkers they cloned from (and every interior node) stay frozen.
// The population grows so that at least min_leafs leaves exist, up to
// max_walkers. This file is independent of FractalGas: it shares only the
// pure tensor helpers (random_alive_compas, l2, relativize) that both Python
// references also share.
//
// Per iteration (core.py::step_tree, phase order and RNG draw order):
//   D1 compas1 = random_alive_compas(oobs, observ)
//      distance_norm = relativize(l2(obs, obs[compas1]))
//      rewards_norm  = relativize(cum_reward, mean/std over is_leaf)   (leaf
//                      mask of the PREVIOUS iteration; all ones after reset)
//      other         = visit-count reward (Coords mode) or 1
//      vr = distance_norm^dist_coef * rewards_norm^reward_coef * other
//   P2 is_leaf = ones; is_leaf[parent] = False  (root: parent[0] = 0)
//   D2 compas2 = random_alive_compas(oobs, vr)
//      clone_probs = (vr[compas2] - vr) / where(vr > eps, vr, eps)
//   D3 wants_clone = clone_probs > rand(n)
//      is_cloned[compas2[wants_clone]] = True; wants_clone[oobs] = True
//      will_clone = wants_clone & is_leaf & ~is_cloned; none -> iteration++
//   P5 will_clone[argmax cum_reward] = False; gather obs/reward/cum_reward/
//      state/info from compas2 for will_clone; parent[will_clone] = compas2
//   D4 actions(k)  D5 dt(k)   (only the k = will_clone walkers step)
//      observ/reward/state <- env; cum_reward += reward; oobs = done
//      update_visits(stepped walkers)
//   P7 new = min_leafs - is_leaf.sum(); grow (fresh: parent 0, oobs True,
//      cum_reward 0, zero obs, no state), capped at max_walkers
//   P8 total_steps += k; iteration++
// reset(): actions(start) BEFORE env.reset, dt(start), one batch step from
// the reset state; walker 0 keeps the reset STATE (root, oobs True) but takes
// the stepped observation like the reference.
//
// Documented deviations from the reference: total_steps counts the walkers
// that really stepped (the reference over-counts by the fresh slots),
// max_walkers is a hard cap on growth (the reference preallocates and would
// index out of range), and a walker that would step from an EMPTY state
// (only reachable when every walker is dead, where the reference crashes on
// a None state) is dropped from the batch.
#ifndef FRACTAL_GAS_FRACTAL_TREE_HPP
#define FRACTAL_GAS_FRACTAL_TREE_HPP

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "backends/snapshot_graph.hpp"
#include "env.hpp"
#include "fractal/graph.hpp"
#include "rng.hpp"
#include "swarm_algorithm.hpp"
#include "visit_grid.hpp"

namespace fg {

using FractalTreeParams = fractal::GraphConfig;

/// The tree's RNG-facing choices. Virtual so tests can replay recorded
/// draws from the Python reference (same pattern as the wave operators).
class FractalTreeSampler {
 public:
  virtual void sample_companions_into(const std::vector<uint8_t>& alive, Rng& rng,
                                      std::vector<int32_t>& out) const;
  virtual void sample_uniforms_into(int32_t n, Rng& rng, std::vector<float>& out) const;
  virtual void sample_actions_into(int32_t n, int32_t count, Rng& rng,
                                   std::vector<int32_t>& out) const;
  virtual void sample_dt_into(int32_t n, int32_t low, int32_t high, Rng& rng,
                              std::vector<int32_t>& out) const;

  virtual ~FractalTreeSampler() = default;
  /// random_alive_compas(oobs, ...) — alive-only companions (with
  /// replacement when some walkers are dead), then a random permutation.
  virtual std::vector<int32_t> sample_companions(const std::vector<uint8_t>& alive,
                                                 Rng& rng) const;
  /// torch.rand(n): the clone-decision thresholds.
  virtual std::vector<float> sample_uniforms(int32_t n, Rng& rng) const;
  /// RandomPolicy: k uniform discrete actions.
  virtual std::vector<int32_t> sample_actions(int32_t k, int32_t n_actions, Rng& rng) const;
  /// UniformDtSampler: k frame skips uniform in [dt_min, dt_max] inclusive.
  virtual std::vector<int32_t> sample_dt(int32_t k, int32_t dt_min, int32_t dt_max,
                                         Rng& rng) const;

 private:
  mutable CompanionScratch companion_scratch_;
};

/// Structure-of-arrays tree state; every array is sized n (the live
/// population). Fresh slots appended by growth hold the reset values.
using TreeState = fractal::GraphPopulation<SnapshotGraphBackend::Storage, WalkerInfo, int32_t>;

class FractalTree final : public SwarmAlgorithm {
 public:
  FractalTree(BatchEnv& env, FractalTreeParams params, std::unique_ptr<Rng> rng = nullptr,
              std::unique_ptr<FractalTreeSampler> sampler = nullptr);

  const FractalTreeParams& params() const { return params_; }
  const TreeState& state() const { return state_; }
  const VisitGrid& visits() const { return visits_; }
  bool counting_visits() const override { return count_visits_; }
  const VisitGrid* visit_grid() const override { return &visits_; }
  bool visit_reward_on() const { return params_.visit_reward; }

  void reset() override;
  StepInfo step() override;

  int32_t n_walkers() const override { return state_.n; }
  const std::vector<char>& walker_state(int32_t i) const override {
    return state_.states[static_cast<size_t>(i)];
  }
  bool walker_alive(int32_t i) const override { return !state_.oobs[static_cast<size_t>(i)]; }
  float walker_cum_reward(int32_t i) const override {
    return state_.cum_rewards[static_cast<size_t>(i)];
  }
  int32_t walker_parent(int32_t i) const override { return state_.parent[static_cast<size_t>(i)]; }
  bool walker_is_leaf(int32_t i) const override {
    return state_.is_leaf[static_cast<size_t>(i)] != 0;
  }
  bool has_walker_info() const override { return env_.has_walker_info(); }
  const WalkerInfo& walker_info(int32_t i) const override {
    return state_.info[static_cast<size_t>(i)];
  }

  std::pair<int32_t, float> get_best_walker() const override;
  const std::vector<uint8_t>& best_frame() const override { return best_frame_; }
  int64_t total_steps() const override { return total_steps_; }
  int64_t total_clones() const override { return total_clones_; }
  int64_t total_frames() const override { return total_frames_; }
  int32_t iteration_count() const override { return iteration_; }

  void set_dist_coef(float v) override { params_.dist_coef = v; }
  void set_reward_coef(float v) override { params_.reward_coef = v; }
  void set_dt_range(int32_t lo, int32_t hi) override {
    params_.dt_min = lo;
    params_.dt_max = hi;
  }
  void set_erase_coef(float v) override {
    params_.erase_coef = v;
    visits_.set_erase_coef(v);
  }
  /// Live: the per-pixel counters are kept, only the reward's pooling
  /// window changes from the next iteration on.
  void set_agg_block_size(int32_t b) override {
    params_.agg_block_size = b < 1 ? 1 : b;
    visits_.set_block_size(params_.agg_block_size);
  }
  /// Live ablation switch: with the term off `other = 1` from the next
  /// iteration on, but the grid keeps counting so the heatmap stays
  /// available and switching back on has the full history. (The reference's
  /// count_visits=False disabled both; here counting is tied to the env.)
  void set_visit_reward(bool on) override { params_.visit_reward = on; }
  void set_visit_coef(float c) override { params_.visit_coef = c; }

 private:
  int32_t best_index() const;  // first argmax of cum_rewards
  BatchEnv& env_;
  FractalTreeParams params_;
  std::unique_ptr<Rng> rng_;
  std::unique_ptr<FractalTreeSampler> sampler_;
  VisitGrid adapter_visits_;
  SnapshotGraphBackend backend_;
  fractal::Graph<SnapshotGraphBackend, FractalTreeSampler> core_;
  bool& count_visits_;
  TreeState& state_;
  VisitGrid& visits_;
  std::vector<uint8_t> best_frame_;
  int64_t &total_steps_, &total_clones_, &total_frames_;
  int32_t& iteration_;
};
}  // namespace fg
#endif
