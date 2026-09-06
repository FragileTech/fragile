#pragma once
#include "cloning.hpp"
#include "control/physics.hpp"
#include "exploration_tree.hpp"

namespace fg::control {
struct WaveConfig {
  uint32_t walkers = 128, horizon = 16, frames = 6, elites = 0;
  float distance_coef = 1, reward_coef = 1, noise = .2f;
  bool cumulative = true, inertial = true;
  RecordingMode recording = RecordingMode::Full;
};
struct WaveStats {
  uint32_t iterations = 0, alive = 0, cloned = 0, collisions = 0, frames = 0,
           pruned = 0;
  float mean_reward = 0, max_reward = 0, dead_ratio = 0, clone_ratio = 0,
        mean_fitness = 0;
};
class PackedWave {
 public:
  Physics& physics;
  WaveConfig config;
  ExplorationTree tree;
  WaveStats stats;
  StateBatch current, next, elite;
  std::vector<float> actions, root_actions, rewards, step_rewards, observations;
  std::vector<uint32_t> node_ids;
  PackedWave(Physics& physics, WaveConfig config, uint64_t seed);
  void reset(const StateBatch& source, size_t row = 0);
  void reseed(uint64_t seed) { rng_ = std::make_unique<Mt19937Rng>(seed); }
  void step();
  std::vector<float> select_action() const;
  StateBatch replay(uint32_t node_id);
  void save_checkpoint(CheckpointWriter& out) const;
  void load_checkpoint(CheckpointReader& in);

 private:
  std::unique_ptr<Rng> rng_;
  FractalCloningOperator cloning_;
  std::vector<uint8_t> alive_;
  std::vector<int32_t> sources_, durations_;
  std::vector<float> next_roots_, next_rewards_, next_actions_, pose_,
      elite_rewards_, elite_actions_, elite_roots_;
  std::vector<uint32_t> next_nodes_, elite_nodes_;
  std::vector<StepResult> results_;
  std::vector<int32_t> ranked_;
  bool has_elite_ = false;
  StateBatch root_;
  void update_elites();
};
// An incremental controller: each advance() commits one complete Wave
// iteration. Hosts can enforce deadlines/cancel between iterations without
// partial resampling.
class FmcPlanner {
 public:
  PackedWave wave;
  std::vector<float> selected;
  bool ready = false;
  FmcPlanner(Physics& physics, WaveConfig config, uint64_t seed)
      : wave(physics, config, seed) {}
  void begin(const StateBatch& state, size_t row = 0) {
    wave.reset(state, row);
    ready = false;
    selected.clear();
  }
  bool advance() {
    if (!ready) {
      wave.step();
      ready =
          wave.stats.iterations >= wave.config.horizon || wave.stats.alive == 0;
    }
    if (ready) selected = wave.select_action();
    return ready;
  }
  void finish() {
    selected = wave.select_action();
    ready = true;
  }
};
}  // namespace fg::control
