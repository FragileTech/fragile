#pragma once
#include "backends/control.hpp"
#include "cloning.hpp"
#include "control/checkpoint_io.hpp"
#include "control/physics.hpp"
#include "exploration_tree.hpp"
#include "fractal/planner.hpp"
#include "fractal/wave.hpp"

namespace fg::control {
struct WaveConfig {
  uint32_t walkers = 128, horizon = 16, frames = 6, elites = 0;
  DistanceMetric distance_metric = DistanceMetric::L2;
  float distance_coef = 1, reward_coef = 1, noise = .2f;
  bool cumulative = true, inertial = true;
  RecordingMode recording = RecordingMode::Full;
  int planning_algorithm = 2, max_horizon = 0;
  bool consensus_prefix = true;
};
struct WaveStats {
  uint32_t iterations = 0, alive = 0, cloned = 0, collisions = 0, frames = 0, pruned = 0;
  float mean_reward = 0, max_reward = 0, dead_ratio = 0, clone_ratio = 0, mean_fitness = 0;
};
class PackedWave {
 private:
  mutable ControlBackend backend_;
  ContinuousActions action_policy_;
  mutable fractal::Wave<PackedPopulation, ControlBackend, ContinuousActions> core_;

 public:
  Physics& physics;
  WaveConfig config;
  ExplorationTree& tree;
  WaveStats stats;
  StateBatch &current, &next, &elite;
  std::vector<float>&actions, &root_actions, &rewards, &step_rewards, &observations;
  std::vector<uint32_t>& node_ids;
  PackedWave(Physics& physics, WaveConfig config, uint64_t seed);
  void reset(const StateBatch& source, size_t row = 0);
  void reseed(uint64_t seed) { rng_ = std::make_unique<Mt19937Rng>(seed); }
  void step();
  std::vector<float> select_action() const;
  const PackedPopulation& population() const {
    backend_.observe(core_.current);
    return core_.current;
  }
  std::vector<float> neutral_action() const;
  uint32_t best_leaf() const;
  uint32_t common_ancestor() const;
  StateBatch replay(uint32_t node_id);
  void save_checkpoint(CheckpointWriter& out) const;
  void load_checkpoint(CheckpointReader& in);
  size_t working_bytes() const {
    return root_.bytes() + core_.working_bytes([](const StateBatch& b) { return b.bytes(); });
  }

 private:
  std::unique_ptr<Rng> rng_;
  FractalCloningOperator cloning_;
  StateBatch root_;
};
// An incremental controller: each advance() commits one complete Wave
// iteration. Hosts can enforce deadlines/cancel between iterations without
// partial resampling.
class FmcPlanner {
 public:
  PackedWave wave;
  fractal::Planner<float> search;
  std::vector<float> selected;
  bool ready = false;
  FmcPlanner(Physics& physics, WaveConfig config, uint64_t seed) : wave(physics, config, seed) {
    configure();
  }
  void configure() {
    fractal::PlannerSettings s;
    s.algorithm = wave.config.planning_algorithm;
    s.horizon = wave.config.horizon;
    s.max_horizon = wave.config.max_horizon;
    s.consensus_prefix = wave.config.consensus_prefix;
    s.selection = fractal::RootSelection::PopulationMean;
    s.commit_frames = wave.config.frames;
    search.begin(s);
    search.result.action_dim = wave.physics.scene->channels.size();
  }
  void begin(const StateBatch& state, size_t row = 0) {
    wave.reset(state, row);
    configure();
    ready = false;
    selected.clear();
  }
  bool advance() {
    ready =
        search.advance([&] { wave.step(); }, wave.population(), wave.tree, wave.neutral_action());
    if (ready) select();
    return ready;
  }
  void finish() {
    search.depth = wave.stats.iterations;
    search.finish(wave.population(), wave.tree, wave.neutral_action());
    ready = true;
    select();
  }
  void wave_step() {
    wave.step();
    search.depth = wave.stats.iterations;
    search.result = {};
    search.result.action_dim = wave.physics.scene->channels.size();
    ready = false;
    selected.clear();
  }
  void select() {
    if (!search.result.actions.empty())
      selected.assign(search.result.actions.begin(),
                      search.result.actions.begin() + wave.physics.scene->channels.size());
  }
};
}  // namespace fg::control
