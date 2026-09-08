#include "arcade_planner.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace fg {
int ArcadePlannerSettings::maximum() const {
  return max_horizon == 0 ? std::min(4096, 2 * horizon) : max_horizon;
}
void ArcadePlannerSettings::validate() const {
  if (algorithm != 2 && algorithm != 3)
    throw std::invalid_argument("Unknown arcade planner");
  if (horizon < 1 || horizon > 4096)
    throw std::invalid_argument("Search horizon must be between 1 and 4096");
  if (algorithm == 3 && consensus_prefix &&
      (maximum() < horizon || maximum() > 4096))
    throw std::invalid_argument("Maximum search horizon must be at least the normal horizon and at most 4096, or 0 for automatic");
}

namespace {
std::vector<uint32_t> executable_branch(const ExplorationTree& tree, uint32_t leaf) {
  auto path = tree.branch(leaf);
  path.erase(std::remove_if(path.begin(), path.end(), [&](uint32_t id) {
    return tree.node(id).frames == 0;
  }), path.end());
  return path;
}
}  // namespace

ArcadePlan select_arcade_plan(const WalkerState& state, const ExplorationTree& tree,
                              const ArcadePlannerSettings& settings, int depth) {
  settings.validate();
  const bool alive = state.alive_count() > 0;
  if (alive && depth < settings.horizon) return {};
  std::vector<int32_t> candidates;
  for (int32_t i = 0; i < state.N; ++i)
    if (!alive || state.alive(i)) candidates.push_back(i);
  if (candidates.empty() || state.lineage.size() != static_cast<size_t>(state.N))
    throw std::runtime_error("Planner has no action ancestry. Reset the run.");
  auto better = [&](int a, int b) {
    return state.rewards[a] > state.rewards[b] ||
           (state.rewards[a] == state.rewards[b] && a < b);
  };
  int best = candidates.front();
  for (int i : candidates) if (better(i, best)) best = i;
  auto path = executable_branch(tree, state.lineage[best]);
  std::string mode;
  if (!alive) {
    if (path.size() > 1) path.resize(1);
    mode = "all-dead fallback";
  } else if (settings.algorithm == 2) {
    // Vote by discrete action, not by the numerical average of action IDs.
    std::map<int32_t, std::vector<int32_t>> votes;
    for (int i : candidates) {
      const auto branch = executable_branch(tree, state.lineage[i]);
      if (!branch.empty()) votes[static_cast<int32_t>(*tree.action(branch.front()))].push_back(i);
    }
    if (!votes.empty()) {
      auto winner = votes.begin();
      for (auto it = votes.begin(); it != votes.end(); ++it)
        if (it->second.size() > winner->second.size()) winner = it;
      best = winner->second.front();
      for (int i : winner->second) if (better(i, best)) best = i;
      path = executable_branch(tree, state.lineage[best]);
      path.resize(1);
    }
    mode = "first-action vote";
  } else if (settings.consensus_prefix) {
    // Compare ancestry IDs, not merely equal actions at unrelated states.
    auto shared = tree.branch(state.lineage[candidates.front()]);
    for (int i : candidates) {
      const auto other = tree.branch(state.lineage[i]);
      size_t n = 0;
      while (n < shared.size() && n < other.size() && shared[n] == other[n]) ++n;
      shared.resize(n);
    }
    shared.erase(std::remove_if(shared.begin(), shared.end(), [&](uint32_t id) {
      return tree.node(id).frames == 0;
    }), shared.end());
    if (!shared.empty()) {
      path = std::move(shared);
      mode = "shared prefix";
    } else {
      if (depth < settings.maximum()) return {};
      if (path.size() > 1) path.resize(1);
      mode = "horizon fallback";
    }
  } else {
    mode = "full path";
  }
  if (path.empty())
    throw std::runtime_error("Planner found no executable trajectory. Reset or change the search settings.");
  ArcadePlan result;
  result.ready = true;
  result.mode = mode;
  for (uint32_t id : path)
    result.actions.push_back({static_cast<int32_t>(*tree.action(id)),
                              static_cast<int32_t>(tree.node(id).frames)});
  return result;
}

ArcadePlanner::ArcadePlanner(BatchEnv& env, FractalGas& gas, ArcadePlannerSettings settings)
    : env_(env), gas_(gas), settings_(settings) {
  settings_.validate();
  if (gas.params().recording == RecordingMode::Off)
    throw std::invalid_argument("Arcade planning requires pruned action ancestry");
}
void ArcadePlanner::reset() {
  gas_.reset();
  state_ = gas_.walker_state(0);
  obs_.assign(gas_.state().observations.begin(),
              gas_.state().observations.begin() + gas_.state().obs_dim);
  info_ = {};
  has_info_ = false;
  done_ = false;
  played_frames_ = 0;
  score_ = reward_ = 0;
  last_ = {};
  env_.render_frame(state_, frame_);
  invalidate();
}
void ArcadePlanner::configure(ArcadePlannerSettings settings) {
  settings.validate();
  settings_ = settings;
  invalidate();
}
void ArcadePlanner::invalidate() {
  plan_ = {};
  next_action_ = 0;
  depth_ = 0;
  new_search_ = true;
}
StepInfo ArcadePlanner::advance() {
  search_advanced_ = false;
  if (done_) return last_;
  if (next_action_ < plan_.actions.size()) {
    execute(plan_.actions[next_action_]);
    ++next_action_;
    if (done_ || next_action_ == plan_.actions.size()) new_search_ = true;
    return last_;
  }
  if (new_search_) {
    gas_.start_from(state_, obs_, has_info_ ? &info_ : nullptr);
    depth_ = 0;
    plan_ = {};
    next_action_ = 0;
    new_search_ = false;
  }
  last_ = gas_.step();
  search_advanced_ = true;
  ++depth_;
  plan_ = select_arcade_plan(gas_.state(), gas_.exploration_tree(), settings_, depth_);
  return last_;
}
void ArcadePlanner::execute(const ArcadeAction& action) {
  std::vector<std::vector<char>> output(1);
  std::vector<float> observations(static_cast<size_t>(env_.obs_dim())), rewards(1);
  std::vector<uint8_t> dones(1), truncated(1);
  env_.step_batch({state_}, {action.action}, {action.frames}, output, observations,
                  rewards, dones, truncated);
  state_ = std::move(output[0]);
  obs_ = std::move(observations);
  reward_ += rewards[0];
  score_ = env_.has_display_score() ? env_.display_score(0) : reward_;
  has_info_ = env_.has_walker_info();
  if (has_info_) info_ = env_.walker_info(0);
  const int frames = env_.frames_stepped(0);
  played_frames_ += frames >= 0 ? frames : action.frames;
  const bool recoverable = dones[0] && !truncated[0] &&
      env_.has_recoverable_dones() && env_.done_is_recoverable(0);
  done_ = truncated[0] || (dones[0] && !recoverable);
  // A life loss may invalidate the remainder of a planned path.
  if (recoverable) plan_.actions.resize(next_action_ + 1);
  env_.render_frame(state_, frame_);
}
}  // namespace fg
