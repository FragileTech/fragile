#include "arcade_planner.hpp"
#include "mock_env.hpp"
#include "visit_mock_env.hpp"
#include "test_framework.hpp"

#include <cstring>
#include <stdexcept>

using namespace fg;
namespace {
struct BranchFixture {
  ExplorationTree tree;
  WalkerState state;
  uint32_t root;
  BranchFixture() {
    tree.reset(RecordingMode::Pruned, 1, 0);
    float action = 0;
    root = tree.append(0, 0, &action, nullptr, 0, 0, 0, 0);
  }
  uint32_t edge(uint32_t parent, int a, int dt) {
    float action = static_cast<float>(a);
    return tree.append(parent, dt, &action, nullptr, 0, 0, 0, 0);
  }
  void walker(uint32_t leaf, float reward, bool alive = true) {
    ++state.N;
    state.lineage.push_back(leaf);
    state.rewards.push_back(reward);
    state.dones.push_back(!alive);
    state.truncated.push_back(0);
  }
};
FractalGasParams planner_params(int n = 4) {
  FractalGasParams p;
  p.N = n;
  p.dt_min = p.dt_max = 1;
  p.recording = RecordingMode::Pruned;
  p.record_observations = false;
  p.seed = 7;
  return p;
}
// Per-frame deterministic dynamics with a soft death at frame 3 and a hard
// terminal at frame 7. Snapshots include both time and accumulated action.
class TerminalEnv : public BatchEnv {
 public:
  bool soft = false;
  bool zero = false;
  std::vector<int> frames;
  std::vector<bool> recoverable;
  int n_actions() const override { return 4; }
  int obs_dim() const override { return 2; }
  void reset(std::vector<char>& s, std::vector<float>& obs) override {
    int values[2] = {0, 0};
    s.resize(sizeof(values));
    std::memcpy(s.data(), values, sizeof(values));
    obs = {0, 0};
  }
  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions, const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& output, std::vector<float>& obs,
                  std::vector<float>& rewards, std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override {
    frames.assign(states.size(), 0);
    recoverable.assign(states.size(), false);
    for (size_t i = 0; i < states.size(); ++i) {
      int values[2];
      std::memcpy(values, states[i].data(), sizeof(values));
      for (int f = 0; !zero && f < dt[i] && values[0] < 7; ++f) {
        ++values[0]; values[1] += actions[i] + 1; ++frames[i];
        if (soft && values[0] == 3) { recoverable[i] = true; break; }
      }
      output[i].resize(sizeof(values));
      std::memcpy(output[i].data(), values, sizeof(values));
      obs[2 * i] = values[0]; obs[2 * i + 1] = values[1];
      rewards[i] = float(frames[i] * (actions[i] + 1));
      dones[i] = zero || values[0] >= 7 || recoverable[i];
      truncated[i] = 0;
    }
  }
  bool has_recoverable_dones() const override { return soft; }
  bool done_is_recoverable(int i) const override { return recoverable[i]; }
  int frames_stepped(int i) const override { return frames[i]; }
  int frame_width() const override { return 1; }
  int frame_height() const override { return 1; }
  void render_frame(const std::vector<char>& s, std::vector<uint8_t>& frame) override {
    frame = {static_cast<uint8_t>(s[0]), 0, 0, 255};
  }
};
}

TEST_CASE(arcade_fmc_votes_discrete_actions_and_breaks_ties) {
  BranchFixture f;
  f.walker(f.edge(f.root, 2, 4), 99);
  f.walker(f.edge(f.root, 1, 2), 5);
  f.walker(f.edge(f.root, 1, 7), 10);
  f.walker(f.edge(f.root, 2, 8), 100);
  f.walker(f.edge(f.root, 0, 1), 1000, false);
  auto p = select_arcade_plan(f.state, f.tree, {2, 1}, 1);
  CHECK(p.ready && p.actions.size() == 1);
  CHECK(p.actions[0].action == 1 && p.actions[0].frames == 7);
  f.state.rewards[1] = 10;
  p = select_arcade_plan(f.state, f.tree, {2, 1}, 1);
  CHECK(p.actions[0].frames == 2);
  CHECK(!select_arcade_plan(f.state, f.tree, {2, 3}, 2).ready);
}

TEST_CASE(arcade_jump_shared_ancestry_and_full_path) {
  BranchFixture f;
  auto shared = f.edge(f.root, 1, 2);
  f.walker(f.edge(shared, 2, 3), 5);
  f.walker(f.edge(shared, 3, 4), 10);
  auto p = select_arcade_plan(f.state, f.tree, {3, 2}, 2);
  CHECK(p.mode == "shared prefix" && p.actions.size() == 1);
  CHECK(p.actions[0].action == 1);
  p = select_arcade_plan(f.state, f.tree, {3, 2, false}, 2);
  CHECK(p.mode == "full path" && p.actions.size() == 2);
  CHECK(p.actions[1].action == 3);
}

TEST_CASE(arcade_jump_extends_until_limit_and_handles_all_dead) {
  BranchFixture f;
  f.walker(f.edge(f.root, 1, 2), 5);
  f.walker(f.edge(f.root, 1, 3), 10);  // equal actions, different ancestors
  CHECK(!select_arcade_plan(f.state, f.tree, {3, 2}, 2).ready);
  auto p = select_arcade_plan(f.state, f.tree, {3, 2}, 4);
  CHECK(p.mode == "horizon fallback" && p.actions[0].frames == 3);
  f.state.dones = {1, 1};
  p = select_arcade_plan(f.state, f.tree, {3, 32}, 1);
  CHECK(p.mode == "all-dead fallback" && p.actions.size() == 1);
  p = select_arcade_plan(f.state, f.tree, {2, 32}, 1);
  CHECK(p.mode == "all-dead fallback");
}

TEST_CASE(arcade_skips_zero_edges_and_reports_empty_paths) {
  BranchFixture f;
  auto zero = f.edge(f.root, 0, 0);
  f.walker(f.edge(zero, 2, 2), 1);
  auto p = select_arcade_plan(f.state, f.tree, {3, 1}, 1);
  CHECK(p.actions.size() == 1 && p.actions[0].frames == 2);
  f.state.lineage[0] = zero;
  bool threw = false;
  try { select_arcade_plan(f.state, f.tree, {2, 1}, 1); }
  catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
}

TEST_CASE(arcade_lineage_survives_clone_inject_extract) {
  BranchFixture f;
  f.walker(f.edge(f.root, 1, 2), 5);
  f.walker(f.edge(f.root, 3, 4), 10);
  // Use a complete real WalkerState for gather operations.
  MockEnv env;
  FractalGas gas(env, planner_params(2)); gas.reset(); gas.step();
  auto s = gas.state();
  s.lineage = f.state.lineage;
  auto cloned = s.clone({1, 0}, {1, 0});
  CHECK(cloned.lineage[0] == f.state.lineage[1]);
  CHECK(cloned.lineage[1] == f.state.lineage[1]);
  auto elite = WalkerState::extract(s, {1});
  s.inject(elite, 1);
  CHECK(s.lineage[0] == f.state.lineage[1]);
}

TEST_CASE(arcade_commit_replays_winner_exactly_and_replans_without_reset) {
  TerminalEnv env;
  auto params = planner_params(); params.n_elite = 2;
  FractalGas gas(env, params);
  ArcadePlanner planner(env, gas, {3, 3, false}); planner.reset();
  const auto root = planner.state();
  for (int i = 0; i < 3; ++i) {
    planner.advance(); CHECK(planner.state() == root); CHECK(planner.played_frames() == 0);
  }
  int best = gas.get_best_walker().first;
  const auto winner = gas.walker_state(best);
  const auto path = gas.exploration_tree().branch(gas.state().lineage[best]);
  int frames = 0;
  for (auto id : path) if (gas.exploration_tree().node(id).frames) {
    frames += gas.exploration_tree().node(id).frames;
    planner.advance();
    CHECK(!planner.search_advanced());
  }
  CHECK(planner.state() == winner);
  CHECK(planner.played_frames() == frames);
  auto committed = planner.state();
  planner.advance();
  CHECK(planner.state() == committed);
  CHECK(gas.exploration_tree().root_snapshot == std::vector<uint8_t>(committed.begin(), committed.end()));
  CHECK(planner.depth() == 1 && gas.iteration_count() == 4);
  planner.reset();
  for (int i = 0; i < 3; ++i) planner.advance();
  CHECK(gas.walker_state(gas.get_best_walker().first) == winner);
}

TEST_CASE(arcade_actual_terminal_durations_and_committed_end) {
  for (int algorithm : {2, 3}) {
    TerminalEnv env;
    auto params = planner_params(1); params.dt_min = params.dt_max = 10;
    FractalGas gas(env, params);
    ArcadePlanner planner(env, gas, {algorithm, 2}); planner.reset();
    planner.advance();
    const auto& tree = gas.exploration_tree();
    CHECK(tree.node(gas.state().lineage[0]).frames == 7);
    CHECK(!planner.done() && planner.played_frames() == 0);
    const auto winner = gas.walker_state(0);
    planner.advance();
    CHECK(planner.done() && planner.played_frames() == 7);
    CHECK(planner.state() == winner);
    planner.advance(); CHECK(planner.played_frames() == 7);
  }
}

TEST_CASE(arcade_soft_death_replans_and_preserves_game_progress) {
  TerminalEnv env; env.soft = true;
  auto params = planner_params(1); params.dt_min = params.dt_max = 5;
  FractalGas gas(env, params);
  ArcadePlanner planner(env, gas, {3, 1, false}); planner.reset();
  planner.advance(); planner.advance();
  CHECK(!planner.done() && planner.played_frames() == 3);
  planner.advance(); planner.advance();
  CHECK(planner.done() && planner.played_frames() == 7);
}

TEST_CASE(arcade_visits_persist_and_reconfigure_discards_queue) {
  VisitMockEnv env;
  auto params = planner_params(1); params.erase_coef = 0;
  FractalGas gas(env, params);
  ArcadePlanner planner(env, gas, {3, 3, false}); planner.reset();
  for (int i = 0; i < 3; ++i) planner.advance();
  std::vector<int32_t> keys; std::vector<float> sums;
  gas.visit_grid()->export_blocks(keys, sums);
  CHECK(!keys.empty());
  planner.advance();
  auto committed = planner.state();
  planner.configure({2, 1});
  planner.advance();
  CHECK(planner.search_advanced() && planner.state() == committed);
  std::vector<int32_t> next_keys; std::vector<float> next_sums;
  gas.visit_grid()->export_blocks(next_keys, next_sums);
  float before = 0, after = 0;
  for (float v : sums) before += v;
  for (float v : next_sums) after += v;
  CHECK(after >= before);
}

TEST_CASE(arcade_settings_validation) {
  for (auto s : {ArcadePlannerSettings{2, 0}, {3, 4097}, {3, 32, true, 31}, {3, 32, true, -1}}) {
    bool threw = false;
    try { s.validate(); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
  CHECK(ArcadePlannerSettings().maximum() == 64);
}
