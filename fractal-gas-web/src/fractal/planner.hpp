#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>

#include "fractal/exploration_tree.hpp"

namespace fg::fractal {
enum class RootSelection { DiscreteVote, PopulationMean };
struct PlannerSettings {
  int algorithm = 2, horizon = 32, max_horizon = 0, commit_frames = 1;
  bool consensus_prefix = true;
  RootSelection selection = RootSelection::DiscreteVote;
  int maximum() const { return max_horizon ? max_horizon : std::min(4096, 2 * horizon); }
  void validate() const {
    if (algorithm != 2 && algorithm != 3) throw std::invalid_argument("Unknown Fractal planner");
    if (horizon < 1 || horizon > 4096)
      throw std::invalid_argument("Search horizon must be between 1 and 4096");
    if (algorithm == 3 && consensus_prefix && (maximum() < horizon || maximum() > 4096))
      throw std::invalid_argument(
          "Maximum search horizon must be at least the normal horizon and at most 4096");
  }
};
template <class Action>
struct Plan {
  bool ready = false;
  int action_dim = 1;
  uint32_t selected_leaf = 0;
  std::vector<Action> actions;
  std::vector<int32_t> frames;
  std::string mode;
};
inline std::vector<uint32_t> executable_branch(const ExplorationTree& tree, uint32_t leaf) {
  auto path = tree.branch(leaf);
  path.erase(std::remove_if(path.begin(), path.end(),
                            [&](uint32_t id) { return tree.node(id).frames == 0; }),
             path.end());
  return path;
}
template <class State>
int best_walker(const State& s) {
  if (!s.N) throw std::logic_error("Planner has no walkers");
  int best = 0;
  for (int i = 1; i < s.N; ++i)
    if ((s.alive(i) && !s.alive(best)) ||
        (s.alive(i) == s.alive(best) && s.rewards[i] > s.rewards[best]))
      best = i;
  return best;
}
template <class State>
uint32_t common_ancestor(const State& s, const ExplorationTree& tree) {
  uint32_t common = 0;
  for (int i = 0; i < s.N; ++i) {
    if (!s.alive(i)) continue;
    uint32_t node = s.lineage[i];
    if (!common) {
      common = node;
      continue;
    }
    while (common != node) {
      if (tree.node(common).depth >= tree.node(node).depth)
        common = tree.node(common).parent;
      else
        node = tree.node(node).parent;
    }
  }
  return common;
}
template <class State>
Plan<typename State::action_type> select_plan(
    const State& s, const ExplorationTree& tree, const PlannerSettings& settings, int depth,
    const std::vector<typename State::action_type>& neutral = {}, bool force = false) {
  using Action = typename State::action_type;
  settings.validate();
  Plan<Action> out;
  out.action_dim = s.action_dim;
  bool alive = s.alive_count() > 0;
  if (!force && alive && depth < settings.horizon) return out;
  if (settings.algorithm == 2 && settings.selection == RootSelection::PopulationMean) {
    out.actions.assign(s.action_dim, 0);
    out.frames = {settings.commit_frames};
    out.ready = true;
    if (!alive || !depth) {
      out.actions = neutral;
      out.mode = "neutral action";
      return out;
    }
    for (int i = 0; i < s.N; ++i)
      for (int k = 0; k < s.action_dim; ++k)
        out.actions[k] += s.root_actions[size_t(i) * s.action_dim + k] / float(s.N);
    out.mode = "population mean";
    return out;
  }
  if (s.lineage.size() != size_t(s.N) || tree.mode == RecordingMode::Off)
    throw std::runtime_error("Planner requires recorded action ancestry");
  int best = best_walker(s);
  auto better = [&](int a, int b) {
    return s.rewards[a] > s.rewards[b] || (s.rewards[a] == s.rewards[b] && a < b);
  };
  auto path = executable_branch(tree, s.lineage[best]);
  if (!alive) {
    if (path.size() > 1) path.resize(1);
    out.mode = "all-dead fallback";
  } else if (settings.algorithm == 2) {
    std::map<int32_t, std::vector<int>> votes;
    for (int i = 0; i < s.N; ++i)
      if (s.alive(i)) {
        auto branch = executable_branch(tree, s.lineage[i]);
        if (!branch.empty()) votes[int32_t(*tree.action(branch.front()))].push_back(i);
      }
    if (!votes.empty()) {
      auto winner = votes.begin();
      for (auto i = votes.begin(); i != votes.end(); ++i)
        if (i->second.size() > winner->second.size()) winner = i;
      best = winner->second.front();
      for (int i : winner->second)
        if (better(i, best)) best = i;
      path = executable_branch(tree, s.lineage[best]);
      path.resize(1);
    }
    out.mode = "first-action vote";
  } else if (settings.consensus_prefix) {
    uint32_t shared = common_ancestor(s, tree);
    auto prefix = shared ? executable_branch(tree, shared) : std::vector<uint32_t>{};
    if (!prefix.empty()) {
      path = std::move(prefix);
      out.mode = "shared prefix";
    } else {
      if (!force && depth < settings.maximum()) return out;
      if (path.size() > 1) path.resize(1);
      out.mode = "horizon fallback";
    }
  } else
    out.mode = "full path";
  if (path.empty())
    throw std::runtime_error(
        "Planner found no executable trajectory. Reset or change the search settings.");
  out.ready = true;
  out.selected_leaf = s.lineage[best];
  for (auto id : path) {
    for (int k = 0; k < s.action_dim; ++k) out.actions.push_back(Action(tree.action(id)[k]));
    out.frames.push_back(tree.node(id).frames);
  }
  return out;
}
// Hosts retain their own clocks and execute the returned edges. Every advance
// completes an entire search iteration before checking the shared policy.
template <class Action>
class Planner {
 public:
  PlannerSettings settings;
  Plan<Action> result;
  int depth = 0;
  void begin(PlannerSettings s) {
    s.validate();
    settings = s;
    result = {};
    depth = 0;
  }
  template <class Step, class State>
  bool advance(Step step, const State& state, const ExplorationTree& tree,
               const std::vector<Action>& neutral = {}) {
    if (!result.ready) {
      step();
      ++depth;
      result = select_plan(state, tree, settings, depth, neutral);
    }
    return result.ready;
  }
  template <class State>
  void finish(const State& state, const ExplorationTree& tree,
              const std::vector<Action>& neutral = {}) {
    if (!result.ready) result = select_plan(state, tree, settings, depth, neutral, true);
  }
  void save(CheckpointWriter& out) const {
    out.scalar(depth);
    out.scalar(uint32_t(result.ready));
    out.scalar(result.action_dim);
    out.scalar(result.selected_leaf);
    out.vector(result.actions);
    out.vector(result.frames);
    out.string(result.mode);
  }
  void load(CheckpointReader& in, int action_dim) {
    depth = in.scalar<int>();
    auto ready = in.scalar<uint32_t>();
    result.action_dim = in.scalar<int>();
    result.selected_leaf = in.scalar<uint32_t>();
    result.actions = in.template vector<Action>();
    result.frames = in.template vector<int32_t>();
    result.mode = in.string();
    if (depth < 0 || depth == std::numeric_limits<int>::max() || ready > 1 ||
        result.action_dim != action_dim ||
        result.actions.size() != result.frames.size() * size_t(action_dim))
      throw std::invalid_argument("Invalid planner checkpoint");
    if (ready && result.frames.empty()) throw std::invalid_argument("Missing checkpoint plan");
    for (auto a : result.actions)
      if (!std::isfinite(double(a))) throw std::invalid_argument("Invalid checkpoint action");
    for (auto f : result.frames)
      if (f < 1 || f > 4096) throw std::invalid_argument("Invalid checkpoint duration");
    result.ready = ready;
  }
};
}  // namespace fg::fractal
