#include "arcade_planner.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace fg {
namespace {
fractal::PlannerSettings options(const ArcadePlannerSettings& s) {
  fractal::PlannerSettings out;
  out.algorithm = s.algorithm;
  out.horizon = s.horizon;
  out.max_horizon = s.max_horizon;
  out.consensus_prefix = s.consensus_prefix;
  return out;
}
ArcadePlan export_plan(const fractal::Plan<int32_t>& p) {
  ArcadePlan out;
  out.ready = p.ready;
  out.mode = p.mode;
  for (size_t i = 0; i < p.frames.size(); ++i) out.actions.push_back({p.actions[i], p.frames[i]});
  return out;
}
}  // namespace
int ArcadePlannerSettings::maximum() const { return options(*this).maximum(); }
void ArcadePlannerSettings::validate() const { options(*this).validate(); }
ArcadePlan select_arcade_plan(const WalkerState& state, const ExplorationTree& tree,
                              const ArcadePlannerSettings& settings, int depth) {
  return export_plan(fractal::select_plan(state, tree, options(settings), depth));
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
  search_.begin(options(settings_));
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
    search_.begin(options(settings_));
    depth_ = 0;
    plan_ = {};
    next_action_ = 0;
    new_search_ = false;
  }
  search_.advance([&] { last_ = gas_.step(); }, gas_.state(), gas_.exploration_tree());
  search_advanced_ = true;
  depth_ = search_.depth;
  plan_ = export_plan(search_.result);
  return last_;
}
void ArcadePlanner::execute(const ArcadeAction& action) {
  std::vector<std::vector<char>> output(1);
  std::vector<float> observations(static_cast<size_t>(env_.obs_dim())), rewards(1);
  std::vector<uint8_t> dones(1), truncated(1);
  env_.step_batch({state_}, {action.action}, {action.frames}, output, observations, rewards, dones,
                  truncated);
  state_ = std::move(output[0]);
  obs_ = std::move(observations);
  reward_ += rewards[0];
  score_ = env_.has_display_score() ? env_.display_score(0) : reward_;
  has_info_ = env_.has_walker_info();
  if (has_info_) info_ = env_.walker_info(0);
  const int frames = env_.frames_stepped(0);
  played_frames_ += frames >= 0 ? frames : action.frames;
  const bool recoverable =
      dones[0] && !truncated[0] && env_.has_recoverable_dones() && env_.done_is_recoverable(0);
  done_ = truncated[0] || (dones[0] && !recoverable);
  // A life loss may invalidate the remainder of a planned path.
  if (recoverable) plan_.actions.resize(next_action_ + 1);
  env_.render_frame(state_, frame_);
}
}  // namespace fg
