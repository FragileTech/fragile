#include "control/wave.hpp"

#include <numeric>

namespace fg::control {
PackedWave::PackedWave(Physics& p, WaveConfig c, uint64_t seed)
    : physics(p),
      config(c),
      current(c.walkers, *p.scene),
      next(c.walkers, *p.scene),
      elite(c.elites, *p.scene),
      rng_(std::make_unique<Mt19937Rng>(seed)),
      root_(1, *p.scene) {
  if (!c.walkers || c.walkers > 8192 || !c.horizon || c.horizon > 4096 ||
      c.elites > c.walkers || c.frames < 1 || c.frames > 4096 ||
      p.scene->controlled.empty())
    throw std::invalid_argument("Invalid Wave configuration");
  const size_t n = c.walkers, d = p.scene->channels.size();
  actions.resize(n * d);
  root_actions.resize(n * d);
  next_roots_.resize(n * d);
  next_actions_.resize(n * d);
  rewards.resize(n);
  step_rewards.resize(n);
  next_rewards_.resize(n);
  alive_.resize(n);
  sources_.resize(n);
  durations_.assign(n, c.frames);
  observations.resize(n * p.observation_dim());
  node_ids.resize(n);
  next_nodes_.resize(n);
  results_.resize(n);
  pose_.resize(p.scene->controlled.size() * 2);
  elite_rewards_.resize(c.elites);
  elite_actions_.resize(c.elites * d);
  elite_roots_.resize(c.elites * d);
  elite_nodes_.resize(c.elites);
  ranked_.resize(n + c.elites);
  cloning_.pool = &p.pool;
  cloning_.dist_coef = c.distance_coef;
  cloning_.reward_coef = c.reward_coef;
  cloning_.use_cumulative_reward = c.cumulative;
}
void PackedWave::reset(const StateBatch& source, size_t row) {
  if (source.fingerprint != current.fingerprint || row >= source.count)
    throw std::invalid_argument("Planner root mismatch");
  std::memcpy(root_.row(0), source.row(row), root_.layout.words * 4);
  for (size_t i = 0; i < current.count; ++i)
    std::memcpy(current.row(i), root_.row(0), current.layout.words * 4);
  std::fill(rewards.begin(), rewards.end(), 0);
  std::fill(step_rewards.begin(), step_rewards.end(), 0);
  std::fill(actions.begin(), actions.end(), 0);
  std::fill(root_actions.begin(), root_actions.end(), 0);
  stats = {};
  stats.alive = config.walkers;
  has_elite_ = false;
  tree.reset(config.recording, physics.scene->channels.size(), pose_.size());
  for (size_t c = 0; c < physics.scene->controlled.size(); ++c) {
    Vec2 p =
        position(root_.row(0), current.layout, physics.scene->controlled[c]);
    pose_[2 * c] = p.x;
    pose_[2 * c + 1] = p.y;
  }
  uint32_t root = tree.append(0, 0, actions.data(), pose_.data(), 0, 0, 0, 0);
  std::fill(node_ids.begin(), node_ids.end(), root);
  if (tree.mode != RecordingMode::Off) {
    tree.root_snapshot.resize(root_.serialized_size());
    root_.serialize(tree.root_snapshot.data(), tree.root_snapshot.size());
  }
}
void PackedWave::step() {
  const size_t n = config.walkers, d = physics.scene->channels.size(),
               obs_dim = physics.observation_dim();
  tree.reserve(n);
  // Elite state AND lineage travel together. The packed bank has no per-walker
  // allocations.
  if (has_elite_)
    for (size_t i = 0; i < config.elites; ++i) {
      std::memcpy(current.row(i), elite.row(i), current.layout.words * 4);
      rewards[i] = elite_rewards_[i];
      node_ids[i] = elite_nodes_[i];
      std::copy_n(elite_actions_.data() + i * d, d, actions.data() + i * d);
      std::copy_n(elite_roots_.data() + i * d, d, root_actions.data() + i * d);
    }
  physics.pool.parallel_for(int32_t(n), [&](int32_t i, int) {
    alive_[i] = !word(current.row(i), 7);
    physics.observe(current.row(i), observations.data() + i * obs_dim);
  });
  // These are the existing C++ Wave operators, including both independent
  // companion draws, rescaling, and forced cloning of dead walkers.
  auto fitness =
      cloning_.calculate_fitness(observations, int32_t(n), int32_t(obs_dim),
                                 rewards, step_rewards, alive_, *rng_);
  auto decision = cloning_.decide_cloning(fitness.first, alive_, *rng_);
  stats.cloned = 0;
  for (size_t i = 0; i < n; ++i) {
    sources_[i] = decision.second[i] ? decision.first[i] : int32_t(i);
    stats.cloned += decision.second[i];
  }
  // Sampling is deliberately on the coordinator, independent of scheduling.
  for (size_t i = 0; i < n; ++i)
    for (size_t k = 0; k < d; ++k) {
      float a;
      if (config.inertial && stats.iterations > 0) {
        float u = std::max(1e-7f, rng_->uniform01()), v = rng_->uniform01();
        a = actions[size_t(sources_[i]) * d + k] +
            config.noise * std::sqrt(-2 * std::log(u)) * std::cos(2 * pi * v);
      } else
        a = physics.scene->channels[k].low +
            rng_->uniform01() * (physics.scene->channels[k].high -
                                 physics.scene->channels[k].low);
      next_actions_[i * d + k] = std::clamp(a, physics.scene->channels[k].low,
                                            physics.scene->channels[k].high);
      next_roots_[i * d + k] = stats.iterations
                                   ? root_actions[size_t(sources_[i]) * d + k]
                                   : next_actions_[i * d + k];
    }
  physics.step(current, sources_.data(), next_actions_.data(),
               durations_.data(), next, results_.data());
  stats.alive = stats.collisions = stats.frames = 0;
  stats.mean_reward = stats.mean_fitness = 0;
  stats.max_reward = -1e30f;
  for (size_t i = 0; i < n; ++i) {
    next_rewards_[i] = rewards[sources_[i]] + results_[i].reward;
    stats.alive += !results_[i].dead;
    stats.collisions += results_[i].collisions;
    stats.frames += results_[i].frames;
    stats.mean_reward += next_rewards_[i] / float(n);
    stats.max_reward = std::max(stats.max_reward, next_rewards_[i]);
    stats.mean_fitness += fitness.first[i] / float(n);
    for (size_t c = 0; c < physics.scene->controlled.size(); ++c) {
      Vec2 p = position(next.row(i), next.layout, physics.scene->controlled[c]);
      pose_[2 * c] = p.x;
      pose_[2 * c + 1] = p.y;
    }
    uint32_t flags = results_[i].dead ? 1u : 0u;
    for (size_t t = 0; t < next.layout.tethers; ++t)
      if (word(next.row(i), next.layout.joints + 2 * t)) flags |= 2;
    next_nodes_[i] = tree.append(node_ids[sources_[i]], results_[i].frames,
                                 next_actions_.data() + i * d, pose_.data(),
                                 next_rewards_[i], results_[i].reward,
                                 fitness.first[i], flags);
    step_rewards[i] = results_[i].reward;
  }
  std::swap(current, next);
  rewards.swap(next_rewards_);
  actions.swap(next_actions_);
  root_actions.swap(next_roots_);
  node_ids.swap(next_nodes_);
  ++stats.iterations;
  stats.dead_ratio = 1 - float(stats.alive) / float(n);
  stats.clone_ratio = float(stats.cloned) / float(n);
  update_elites();
  if (tree.mode == RecordingMode::Pruned) {
    std::vector<uint32_t> pins = node_ids;
    pins.insert(pins.end(), elite_nodes_.begin(), elite_nodes_.end());
    stats.pruned = uint32_t(tree.prune(pins));
  }
}
void PackedWave::update_elites() {
  if (!config.elites) return;
  const size_t n = config.walkers, d = physics.scene->channels.size(),
               old = has_elite_ ? config.elites : 0;
  ranked_.resize(n + old);
  std::iota(ranked_.begin(), ranked_.end(), 0);
  auto reward = [&](int i) {
    return size_t(i) < old ? elite_rewards_[i] : rewards[i - old];
  };
  std::stable_sort(ranked_.begin(), ranked_.end(),
                   [&](int a, int b) { return reward(a) > reward(b); });
  // Stage old elite rows in the now-unused output bank; avoid aliasing when
  // selecting an old elite more than once or permuting their order.
  for (size_t i = 0; i < old; ++i)
    std::memcpy(next.row(i), elite.row(i), current.layout.words * 4);
  auto old_rewards = elite_rewards_, old_actions = elite_actions_,
       old_roots = elite_roots_;
  auto old_nodes = elite_nodes_;
  for (size_t i = 0; i < config.elites; ++i) {
    int ix = ranked_[i];
    bool previous = size_t(ix) < old;
    size_t row = previous ? ix : ix - old;
    std::memcpy(elite.row(i), previous ? next.row(row) : current.row(row),
                current.layout.words * 4);
    elite_rewards_[i] = previous ? old_rewards[row] : rewards[row];
    elite_nodes_[i] = previous ? old_nodes[row] : node_ids[row];
    std::copy_n((previous ? old_actions.data() : actions.data()) + row * d, d,
                elite_actions_.data() + i * d);
    std::copy_n((previous ? old_roots.data() : root_actions.data()) + row * d,
                d, elite_roots_.data() + i * d);
  }
  has_elite_ = true;
}
uint32_t PackedWave::common_ancestor() const {
  if (tree.mode == RecordingMode::Off || stats.iterations == 0)
    throw std::logic_error("Common ancestor requires a recorded search");
  uint32_t common = 0;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    if (word(current.row(i), 7)) continue;
    uint32_t node = node_ids[i];
    if (!common) { common = node; continue; }
    while (common != node) {
      if (tree.node(common).depth >= tree.node(node).depth)
        common = tree.node(common).parent;
      else
        node = tree.node(node).parent;
    }
  }
  return common;
}

uint32_t PackedWave::best_leaf() const {
  if (tree.mode == RecordingMode::Off || stats.iterations == 0)
    throw std::logic_error("Best leaf requires a recorded search");
  size_t best = 0;
  for (size_t i = 1; i < rewards.size(); ++i) {
    const bool alive = !word(current.row(i), 7);
    const bool best_alive = !word(current.row(best), 7);
    if ((alive && !best_alive) ||
        (alive == best_alive && rewards[i] > rewards[best]))
      best = i;
  }
  return node_ids[best];
}

std::vector<float> PackedWave::select_action() const {
  std::vector<float> action(physics.scene->channels.size(), 0);
  if (!stats.alive || !stats.iterations) {
    for (size_t k = 0; k < action.size(); ++k)
      action[k] = std::clamp(0.f, physics.scene->channels[k].low,
                             physics.scene->channels[k].high);
    return action;
  }
  // Match PlanningFractalGas: mean over the final population, with cloning
  // providing its implicit weighting. No unlabelled reward weighting.
  for (size_t i = 0; i < config.walkers; ++i)
    for (size_t k = 0; k < action.size(); ++k)
      action[k] += root_actions[i * action.size() + k] / float(config.walkers);
  return action;
}
StateBatch PackedWave::replay(uint32_t id) {
  StateBatch result(1, *physics.scene);
  std::memcpy(result.row(0), root_.row(0), root_.layout.words * 4);
  auto path = tree.branch(id);
  StepResult step;
  for (uint32_t node : path)
    if (tree.node(node).parent)
      physics.step_world(result.row(0), tree.action(node),
                         int(tree.node(node).frames), step);
  return result;
}
void PackedWave::save_checkpoint(CheckpointWriter& out) const {
  auto state = [&](const StateBatch& b) {
    std::vector<uint8_t> bytes(b.serialized_size());
    b.serialize(bytes.data(), bytes.size());
    out.vector(bytes);
  };
  out.string(static_cast<const Mt19937Rng&>(*rng_).checkpoint());
  out.scalar(stats);
  out.scalar(uint32_t(has_elite_));
  state(root_);
  state(current);
  state(elite);
  out.vector(actions);
  out.vector(root_actions);
  out.vector(rewards);
  out.vector(step_rewards);
  out.vector(node_ids);
  out.vector(elite_rewards_);
  out.vector(elite_actions_);
  out.vector(elite_roots_);
  out.vector(elite_nodes_);
  tree.save_checkpoint(out);
}
void PackedWave::load_checkpoint(CheckpointReader& in) {
  auto state = [&](StateBatch& b) {
    auto bytes = in.vector<uint8_t>(512 * 1024 * 1024);
    b.deserialize(bytes.data(), bytes.size());
  };
  static_cast<Mt19937Rng&>(*rng_).restore(in.string());
  stats = in.scalar<WaveStats>();
  const auto has = in.scalar<uint32_t>();
  if (has > 1 || stats.alive > config.walkers)
    throw std::invalid_argument("Invalid checkpoint Wave stats");
  has_elite_ = has;
  state(root_);
  state(current);
  state(elite);
  auto values = [&](auto& target) {
    using T = typename std::decay_t<decltype(target)>::value_type;
    auto v = in.vector<T>();
    if (v.size() != target.size())
      throw std::invalid_argument("Checkpoint planner shape mismatch");
    if constexpr (std::is_same_v<T, float>)
      for (float x : v)
        if (!std::isfinite(x))
          throw std::invalid_argument("Nonfinite checkpoint planner data");
    target = std::move(v);
  };
  values(actions);
  values(root_actions);
  values(rewards);
  values(step_rewards);
  values(node_ids);
  values(elite_rewards_);
  values(elite_actions_);
  values(elite_roots_);
  values(elite_nodes_);
  tree.load_checkpoint(in);
  if (tree.action_dim() != physics.scene->channels.size() ||
      tree.pose_dim() != pose_.size())
    throw std::invalid_argument("Checkpoint tree layout mismatch");
  if (tree.mode != RecordingMode::Off) {
    for (auto id : node_ids) tree.node(id);
    if (has_elite_)
      for (auto id : elite_nodes_) tree.node(id);
  }
}
}  // namespace fg::control
