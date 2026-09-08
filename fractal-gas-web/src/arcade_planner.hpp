// Discrete receding-horizon controllers over the arcade's snapshot interface.
#pragma once

#include "fractal_gas.hpp"

#include <string>

namespace fg {
struct ArcadePlannerSettings {
  int algorithm = 2;  // 2 = FMC, 3 = Jump Wave
  int horizon = 32;
  bool consensus_prefix = true;
  int max_horizon = 0;
  int maximum() const;
  void validate() const;
};

struct ArcadeAction {
  int32_t action = 0;
  int32_t frames = 0;
};
struct ArcadePlan {
  bool ready = false;
  std::vector<ArcadeAction> actions;
  std::string mode;
};

// Pure selection over final walker lineage, shared by both controllers.
ArcadePlan select_arcade_plan(const WalkerState& state, const ExplorationTree& tree,
                              const ArcadePlannerSettings& settings, int depth);

class ArcadePlanner {
 public:
  ArcadePlanner(BatchEnv& env, FractalGas& gas, ArcadePlannerSettings settings);
  void reset();
  void configure(ArcadePlannerSettings settings);
  void invalidate();
  // One search iteration or one committed edge, never a whole planning loop.
  StepInfo advance();
  const std::vector<char>& state() const { return state_; }
  const std::vector<uint8_t>& frame() const { return frame_; }
  const WalkerInfo& info() const { return info_; }
  bool has_info() const { return has_info_; }
  float score() const { return score_; }
  float reward() const { return reward_; }
  int64_t played_frames() const { return played_frames_; }
  int depth() const { return depth_; }
  bool done() const { return done_; }
  bool search_advanced() const { return search_advanced_; }
  bool new_search_pending() const { return new_search_ && !done_; }
  bool execution_pending() const { return next_action_ < plan_.actions.size(); }
  const char* phase() const { return done_ ? "ended" : (search_advanced_ ? "planning" : "playing"); }
  const std::string& execution_mode() const { return plan_.mode; }

 private:
  void execute(const ArcadeAction& action);
  BatchEnv& env_;
  FractalGas& gas_;
  ArcadePlannerSettings settings_;
  std::vector<char> state_;
  std::vector<float> obs_;
  std::vector<uint8_t> frame_;
  WalkerInfo info_;
  bool has_info_ = false, done_ = false, new_search_ = true, search_advanced_ = false;
  int depth_ = 0;
  int64_t played_frames_ = 0;
  float score_ = 0, reward_ = 0;
  ArcadePlan plan_;
  size_t next_action_ = 0;
  StepInfo last_;
};
}  // namespace fg
