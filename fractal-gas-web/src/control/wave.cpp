#include "control/wave.hpp"

#include <numeric>

namespace fg::control {
PackedWave::PackedWave(Physics& p, WaveConfig c, uint64_t seed)
    : backend_{p},
      action_policy_{*p.scene},
      core_(backend_, action_policy_),
      physics(p),
      config(c),
      tree(core_.tree),
      current(core_.current.states),
      next(core_.next.states),
      elite(core_.elite.states),
      actions(core_.current.actions),
      root_actions(core_.current.root_actions),
      rewards(core_.current.rewards),
      step_rewards(core_.current.step_rewards),
      observations(core_.current.observations),
      node_ids(core_.current.lineage),
      rng_(std::make_unique<Mt19937Rng>(seed)),
      root_(1, *p.scene) {
  if (!c.walkers || c.walkers > 8192 || !c.horizon || c.horizon > 4096 || c.elites > c.walkers ||
      c.frames < 1 || c.frames > 4096 || p.scene->controlled.empty())
    throw std::invalid_argument("Invalid Wave configuration");
  core_.reset(c.walkers, p.observation_dim(), p.scene->channels.size(), true);
  cloning_.pool = &p.pool;
}
void PackedWave::reset(const StateBatch& source, size_t row) {
  if (source.fingerprint != current.fingerprint || row >= source.count)
    throw std::invalid_argument("Planner root mismatch");
  std::memcpy(root_.row(0), source.row(row), root_.layout.words * 4);
  core_.reset(config.walkers, physics.observation_dim(), physics.scene->channels.size(), true);
  for (size_t i = 0; i < current.count; ++i) {
    std::memcpy(current.row(i), root_.row(0), current.layout.words * 4);
    physics.observe(current.row(i), observations.data() + i * physics.observation_dim());
  }
  stats = {};
  stats.alive = config.walkers;
  std::vector<float> pose(physics.scene->controlled.size() * 2);
  backend_.pose(core_.current, 0, pose.data());
  std::vector<uint8_t> snapshot;
  if (config.recording != RecordingMode::Off) {
    snapshot.resize(root_.serialized_size());
    root_.serialize(snapshot.data(), snapshot.size());
  }
  core_.begin_history(config.recording, pose.size(), pose.data(), std::move(snapshot));
}
void PackedWave::step() {
  action_policy_.inertial = config.inertial;
  action_policy_.noise = config.noise;
  action_policy_.frames = config.frames;
  cloning_.dist_coef = config.distance_coef;
  cloning_.reward_coef = config.reward_coef;
  cloning_.use_cumulative_reward = config.cumulative;
  const auto& m = core_.step(config.elites, cloning_, *rng_);
  stats.iterations = m.iteration;
  stats.alive = m.alive;
  stats.cloned = m.cloned;
  stats.frames = m.frames;
  stats.pruned = m.pruned;
  stats.mean_reward = m.mean_reward;
  stats.max_reward = *std::max_element(rewards.begin(), rewards.end());
  stats.mean_fitness = m.mean_fitness;
  stats.dead_ratio = 1 - float(m.alive) / config.walkers;
  stats.clone_ratio = float(m.cloned) / config.walkers;
  stats.collisions = 0;
  for (const auto& r : core_.current.infos) stats.collisions += r.collisions;
}
uint32_t PackedWave::common_ancestor() const {
  if (tree.mode == RecordingMode::Off || !stats.iterations)
    throw std::logic_error("Common ancestor requires a recorded search");
  return fractal::common_ancestor(population(), tree);
}
uint32_t PackedWave::best_leaf() const {
  if (tree.mode == RecordingMode::Off || !stats.iterations)
    throw std::logic_error("Best leaf requires a recorded search");
  return node_ids[fractal::best_walker(population())];
}
std::vector<float> PackedWave::neutral_action() const {
  std::vector<float> action(physics.scene->channels.size());
  for (size_t k = 0; k < action.size(); ++k)
    action[k] = std::clamp(0.f, physics.scene->channels[k].low, physics.scene->channels[k].high);
  return action;
}
std::vector<float> PackedWave::select_action() const {
  fractal::PlannerSettings s;
  s.selection = fractal::RootSelection::PopulationMean;
  s.horizon = config.horizon;
  s.commit_frames = config.frames;
  return fractal::select_plan(population(), tree, s, stats.iterations, neutral_action(), true)
      .actions;
}
StateBatch PackedWave::replay(uint32_t id) {
  StateBatch result(1, *physics.scene);
  std::memcpy(result.row(0), root_.row(0), root_.layout.words * 4);
  auto path = tree.branch(id);
  StepResult step;
  for (uint32_t node : path)
    if (tree.node(node).parent)
      physics.step_world(result.row(0), tree.action(node), int(tree.node(node).frames), step);
  return result;
}
void PackedWave::save_checkpoint(CheckpointWriter& out) const {
  out.string(static_cast<const Mt19937Rng&>(*rng_).checkpoint());
  out.scalar(stats);
  auto storage = [&](const StateBatch& b) {
    std::vector<uint8_t> bytes(b.serialized_size());
    b.serialize(bytes.data(), bytes.size());
    out.vector(bytes);
  };
  storage(root_);
  core_.save(out, storage);
}
void PackedWave::load_checkpoint(CheckpointReader& in) {
  static_cast<Mt19937Rng&>(*rng_).restore(in.string());
  stats = in.scalar<WaveStats>();
  auto storage = [&](StateBatch& b) {
    auto bytes = in.vector<uint8_t>(512 * 1024 * 1024);
    b.deserialize(bytes.data(), bytes.size());
  };
  storage(root_);
  core_.load(in, config.elites, storage);
  if (stats.iterations != core_.metrics.iteration || stats.alive != core_.metrics.alive ||
      tree.mode != config.recording || tree.action_dim() != physics.scene->channels.size() ||
      tree.pose_dim() != physics.scene->controlled.size() * 2)
    throw std::invalid_argument("Checkpoint tree layout mismatch");
}
}  // namespace fg::control
