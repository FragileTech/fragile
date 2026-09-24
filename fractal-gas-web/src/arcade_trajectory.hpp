// A paused-run view: Graph snapshots stay in their existing storage. Wave
// ancestry is a compact action path replayed from its saved root on demand.
#pragma once
#include <algorithm>
#include <unordered_map>
#include "arcade_memory.hpp"
#include "arcade_planner.hpp"
#include "fractal_tree.hpp"

namespace fg {
class ArcadeTrajectory {
 public:
  std::vector<const std::vector<char>*> snapshots;
  std::vector<ArcadeAction> actions;
  std::vector<char> root, cursor;
  size_t position = 0;
  int walker = -1;

  void clear() { *this = ArcadeTrajectory{}; }
  size_t size() const { return snapshots.empty() ? (root.empty() ? 0 : actions.size() + 1) : snapshots.size(); }
  void select(SwarmAlgorithm& algo, ArcadePlanner* planner, int slot) {
    clear();
    walker = slot < 0 ? algo.get_best_walker().first : slot;
    if (walker < 0 || walker >= algo.n_walkers() || algo.walker_state(walker).empty())
      throw std::invalid_argument("This walker has no trajectory yet. Select another walker.");
    if (auto* graph = dynamic_cast<FractalTree*>(&algo)) {
      struct Entry { uint64_t parent; const std::vector<char>* state; };
      std::unordered_map<uint64_t, Entry> nodes;
      const auto& s = graph->state();
      arcade_memory::require((size_t(s.n) + graph->frozen_nodes().size()) * 96ULL);
      for (int i = 0; i < s.n; ++i)
        if (s.node_ids[i] != UINT64_MAX && !s.states[i].empty())
          nodes.emplace(s.node_ids[i], Entry{s.parent_ids[i], &s.states[i]});
      for (const auto& n : graph->frozen_nodes()) nodes.emplace(n.id, Entry{n.parent_id, &n.state});
      uint64_t id = s.node_ids[walker];
      for (size_t hops = 0; hops <= nodes.size(); ++hops) {
        auto it = nodes.find(id);
        if (it == nodes.end()) throw std::runtime_error("Missing trajectory ancestor");
        snapshots.push_back(it->second.state);
        if (id == 0) {
          std::reverse(snapshots.begin(), snapshots.end());
          return;
        }
        id = it->second.parent;
      }
      throw std::runtime_error("Cyclic trajectory ancestry");
    }
    auto& gas = dynamic_cast<FractalGas&>(algo);
    const auto& tree = gas.exploration_tree();
    arcade_memory::require(tree.root_snapshot.size() * 4ULL + tree.size() * 16ULL +
        (planner ? planner->played_actions().size() * sizeof(ArcadeAction) : 0));
    root.assign(tree.root_snapshot.begin(), tree.root_snapshot.end());
    if (planner) {
      root = planner->initial_state();
      const auto& played = planner->played_actions();
      actions.assign(played.begin(), played.begin() + planner->search_prefix_size());
    }
    for (uint32_t id : tree.branch(gas.state().lineage[walker])) {
      const auto& node = tree.node(id);
      if (node.parent) actions.push_back({int32_t(tree.action(id)[0]), int32_t(node.frames)});
    }
    cursor = root;
  }
  // Limit reconstruction per request so a long seek can be cancelled. No
  // population states, RNG, or recorded ancestry are changed by playback.
  bool seek(BatchEnv& env, size_t target, std::vector<uint8_t>& rgba) {
    if (target >= size()) throw std::out_of_range("Trajectory position out of range");
    if (!snapshots.empty()) {
      env.render_frame(*snapshots[target], rgba);
      position = target;
      return true;
    }
    if (target < position) { cursor = root; position = 0; }
    for (int budget = 0; position < target && budget < 32; ++budget) {
      const auto& a = actions[position];
      std::vector<std::vector<char>> out(1);
      std::vector<float> obs(env.obs_dim()), rewards(1);
      std::vector<uint8_t> done(1), truncated(1);
      if (a.frames > 0) {
        env.step_batch({cursor}, {a.action}, {a.frames}, out, obs, rewards, done, truncated);
        cursor = std::move(out[0]);
      }
      ++position;
    }
    if (position != target) return false;
    env.render_frame(cursor, rgba);
    return true;
  }
};
}  // namespace fg
