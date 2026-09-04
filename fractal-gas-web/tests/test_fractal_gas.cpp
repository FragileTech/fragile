#include <deque>

#include "fractal_gas.hpp"
#include "mock_env.hpp"
#include "test_framework.hpp"
#include "visit_mock_env.hpp"
#include "fixtures/fixtures_generated.hpp"

using namespace fg;

namespace {

/// Cloning operator that replays companion/uniform draws recorded from the
/// Python reference run instead of sampling.
class ReplayCloningOperator final : public FractalCloningOperator {
 public:
  mutable std::deque<std::vector<int32_t>> companions_queue;
  mutable std::deque<std::vector<float>> uniforms_queue;

  std::vector<int32_t> sample_companions(const std::vector<uint8_t>&,
                                         Rng&) const override {
    auto v = companions_queue.front();
    companions_queue.pop_front();
    return v;
  }
  std::vector<float> sample_uniforms(int32_t, Rng&) const override {
    auto v = uniforms_queue.front();
    uniforms_queue.pop_front();
    return v;
  }
};

class ReplayActionOperator final : public RandomActionOperator {
 public:
  mutable std::deque<std::vector<int32_t>> actions_queue;
  mutable std::deque<std::vector<int32_t>> dt_queue;

  std::vector<int32_t> sample_actions(int32_t, int32_t, Rng&) const override {
    auto v = actions_queue.front();
    actions_queue.pop_front();
    return v;
  }
  std::vector<int32_t> sample_dt(int32_t, Rng&) const override {
    auto v = dt_queue.front();
    dt_queue.pop_front();
    return v;
  }
};

}  // namespace

TEST_CASE(full_run_replays_python_reference) {
  MockEnv env;

  auto clone_op = std::make_unique<ReplayCloningOperator>();
  auto kinetic_op = std::make_unique<ReplayActionOperator>();
  for (int it = 0; it < fixtures::kRunIters; ++it) {
    const auto ui = static_cast<size_t>(it);
    // Per step: fitness companions first, then clone companions.
    clone_op->companions_queue.push_back(fixtures::kRunFitCompanions[ui]);
    clone_op->companions_queue.push_back(fixtures::kRunCloneCompanions[ui]);
    clone_op->uniforms_queue.push_back(fixtures::kRunUniforms[ui]);
    kinetic_op->actions_queue.push_back(fixtures::kRunActions[ui]);
    kinetic_op->dt_queue.push_back(fixtures::kRunDts[ui]);
  }

  FractalGasParams params;
  params.N = fixtures::kRunN;
  params.dist_coef = 1.0f;
  params.reward_coef = 1.0f;
  params.use_cumulative_reward = false;
  params.dt_min = 1;
  params.dt_max = 4;
  params.n_elite = fixtures::kRunNElite;

  FractalGas gas(env, params, std::make_unique<Mt19937Rng>(0),
                 std::move(clone_op), std::move(kinetic_op));
  gas.reset();

  for (int it = 0; it < fixtures::kRunIters; ++it) {
    const auto ui = static_cast<size_t>(it);
    const StepInfo info = gas.step();

    int32_t expected_cloned = 0;
    for (const int32_t w : fixtures::kRunExpectedWillClone[ui]) expected_cloned += w;
    CHECK(info.num_cloned == expected_cloned);
    CHECK_CLOSE(info.max_reward, fixtures::kRunExpectedMaxReward[ui], 1e-5);

    const WalkerState& state = gas.state();
    for (int32_t i = 0; i < state.N; ++i) {
      const auto wi = static_cast<size_t>(i);
      CHECK_CLOSE(state.rewards[wi], fixtures::kRunExpectedRewards[ui][wi], 1e-5);
      CHECK_CLOSE(state.virtual_rewards[wi], fixtures::kRunExpectedVr[ui][wi], 1e-5);
    }
  }

  const WalkerState& final_state = gas.state();
  CHECK(final_state.observations.size() == fixtures::kRunFinalObservations.size());
  for (size_t i = 0; i < final_state.observations.size(); ++i) {
    CHECK_CLOSE(final_state.observations[i], fixtures::kRunFinalObservations[i], 1e-5);
  }
}

TEST_CASE(same_seed_is_deterministic) {
  MockEnv env;
  FractalGasParams params;
  params.N = 16;
  params.seed = 1234;
  params.n_elite = 2;

  auto run_once = [&]() {
    FractalGas gas(env, params);
    gas.reset();
    std::vector<float> rewards;
    for (int it = 0; it < 10; ++it) gas.step();
    return gas.state().rewards;
  };
  const std::vector<float> a = run_once();
  const std::vector<float> b = run_once();
  CHECK(a == b);
}

// Mock for the all-dead revive: every walker dies once its step counter
// (s[2], incremented by 1 per step regardless of action) reaches
// kDeathStep, so the whole swarm dies in the same iteration. `mode` picks
// whether those deaths are recoverable (soft, like an Atari life loss),
// hard, or per-walker (even indices soft-die every step while odd ones
// stay alive).
class RecoverableMockEnv final : public BatchEnv {
 public:
  enum class Mode { kAllSoft, kAllHard, kEvenSoft };
  static constexpr int32_t kObsDim = MockEnv::kObsDim;
  static constexpr float kDeathStep = 3.0f;

  explicit RecoverableMockEnv(Mode mode) : mode_(mode) {}

  int32_t n_actions() const override { return 4; }
  int32_t obs_dim() const override { return kObsDim; }

  void reset(std::vector<char>& state, std::vector<float>& obs) override {
    inner_.reset(state, obs);
  }

  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions,
                  const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& new_states,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override {
    inner_.step_batch(states, actions, dt, new_states, observations, rewards,
                      dones, truncated);
    recoverable_.assign(states.size(), 0);
    for (size_t i = 0; i < states.size(); ++i) {
      const float steps = observations[i * kObsDim + 2];
      const bool dies = mode_ == Mode::kEvenSoft ? (i % 2 == 0)
                                                 : steps >= kDeathStep;
      dones[i] = dies ? 1 : 0;
      if (dies && mode_ != Mode::kAllHard) recoverable_[i] = 1;
    }
  }

  bool has_recoverable_dones() const override { return true; }
  bool done_is_recoverable(int32_t i) const override {
    return recoverable_[static_cast<size_t>(i)] != 0;
  }

  void render_frame(const std::vector<char>&,
                    std::vector<uint8_t>& rgba) override {
    rgba.clear();
  }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }

 private:
  Mode mode_;
  MockEnv inner_;
  std::vector<uint8_t> recoverable_;
};

TEST_CASE(all_dead_soft_walkers_are_revived) {
  RecoverableMockEnv env(RecoverableMockEnv::Mode::kAllSoft);
  FractalGasParams params;
  params.N = 16;
  params.seed = 7;

  FractalGas gas(env, params);
  gas.reset();
  bool saw_revive = false;
  for (int it = 0; it < 10; ++it) {
    const StepInfo info = gas.step();
    // Recoverable deaths never leave the swarm empty.
    CHECK(info.alive_count == params.N);
    if (info.num_revived > 0) {
      saw_revive = true;
      CHECK(info.num_revived == params.N);
    }
  }
  CHECK(saw_revive);
}

TEST_CASE(all_dead_hard_walkers_stay_dead) {
  RecoverableMockEnv env(RecoverableMockEnv::Mode::kAllHard);
  FractalGasParams params;
  params.N = 16;
  params.seed = 7;

  FractalGas gas(env, params);
  gas.reset();
  bool died_out = false;
  for (int it = 0; it < 10 && !died_out; ++it) {
    const StepInfo info = gas.step();
    CHECK(info.num_revived == 0);
    died_out = info.alive_count == 0;
  }
  CHECK(died_out);
}

TEST_CASE(soft_deaths_stay_dead_while_others_live) {
  RecoverableMockEnv env(RecoverableMockEnv::Mode::kEvenSoft);
  FractalGasParams params;
  params.N = 16;
  params.seed = 7;

  FractalGas gas(env, params);
  gas.reset();
  for (int it = 0; it < 10; ++it) {
    const StepInfo info = gas.step();
    // With half the swarm alive, soft-dead walkers are NOT revived — they
    // must clone away as usual (the death signal stays intact).
    CHECK(info.num_revived == 0);
    CHECK(info.alive_count == params.N / 2);
  }
}

TEST_CASE(elite_max_reward_never_decreases) {
  MockEnv env;
  FractalGasParams params;
  params.N = 16;
  params.seed = 5;
  params.n_elite = 3;

  FractalGas gas(env, params);
  gas.reset();
  float best = -1e30f;
  for (int it = 0; it < 20; ++it) {
    const StepInfo info = gas.step();
    CHECK(info.max_reward >= best - 1e-6f);
    best = std::max(best, info.max_reward);
  }
}

namespace {
std::vector<float> wave_vr_after(bool visit_reward, bool count_visits, int iters) {
  VisitMockEnv env;
  FractalGasParams params;
  params.N = 12;
  params.seed = 21;
  params.n_elite = 2;
  params.use_cumulative_reward = true;
  params.count_visits = count_visits;
  params.visit_reward = visit_reward;
  FractalGas gas(env, params);
  gas.reset();
  for (int it = 0; it < iters; ++it) gas.step();
  return gas.state().virtual_rewards;
}
}  // namespace

TEST_CASE(wave_visit_term_is_off_by_default_but_counting_runs) {
  VisitMockEnv env;
  FractalGasParams params;
  params.N = 12;
  params.seed = 21;
  FractalGas gas(env, params);
  CHECK(!gas.visit_reward_on());
  CHECK(gas.counting_visits());  // the env has a visit key -> heatmap data
  CHECK(gas.visit_grid() != nullptr);
  gas.reset();
  CHECK(gas.visit_grid()->nonzero_cells() == 0);
  for (int it = 0; it < 5; ++it) gas.step();
  CHECK(gas.visit_grid()->nonzero_cells() > 0);
  // Infos travel with the walkers and are walker-indexed.
  CHECK(gas.state().has_infos);
  CHECK(gas.walker_info(3).visit_plane == gas.state().infos[3].visit_plane);
  // With the term off the dynamics equal a run that never counts at all.
  CHECK(wave_vr_after(false, true, 6) == wave_vr_after(false, false, 6));
  // With the term on they differ (same seed, same draws).
  CHECK(wave_vr_after(true, true, 6) != wave_vr_after(false, true, 6));
}

TEST_CASE(wave_visit_term_switches_live) {
  VisitMockEnv env;
  FractalGasParams params;
  params.N = 12;
  params.seed = 4;
  params.visit_reward = true;
  FractalGas gas(env, params);
  gas.reset();
  for (int it = 0; it < 4; ++it) gas.step();
  gas.set_visit_reward(false);
  CHECK(!gas.visit_reward_on());
  gas.set_agg_block_size(20);
  CHECK(gas.visit_grid()->block_size() == 20);
  gas.set_erase_coef(0.2f);
  CHECK(gas.visit_grid()->erase_coef() == 0.2f);
  const size_t before = gas.visit_grid()->nonzero_cells();
  for (int it = 0; it < 4; ++it) gas.step();
  CHECK(gas.visit_grid()->nonzero_cells() > 0);  // still counting
  (void)before;
  // Elite injection keeps the info consistent with the injected walker.
  const WalkerState& s = gas.state();
  for (int32_t i = 0; i < s.N; ++i) {
    CHECK(s.infos[static_cast<size_t>(i)].visit_x == static_cast<int32_t>(s.observations[static_cast<size_t>(i) * 3]));
  }
}

TEST_CASE(wave_visit_coef_scales_the_visit_term) {
  auto vr_after = [](float coef, bool on) {
    VisitMockEnv env;
    FractalGasParams params;
    params.N = 12;
    params.seed = 21;
    params.use_cumulative_reward = true;
    params.visit_reward = on;
    params.visit_coef = coef;
    FractalGas gas(env, params);
    gas.reset();
    for (int it = 0; it < 6; ++it) gas.step();
    return gas.state().virtual_rewards;
  };
  CHECK(vr_after(0.0f, true) == vr_after(1.0f, false));
  CHECK(vr_after(1.0f, true) != vr_after(2.0f, true));
}

namespace {
/// MockEnv wrapper that reports it ran one frame fewer than requested for
/// odd walkers (like a death mid-step).
class EarlyStopMockEnv final : public BatchEnv {
 public:
  int32_t n_actions() const override { return inner_.n_actions(); }
  int32_t obs_dim() const override { return inner_.obs_dim(); }
  void reset(std::vector<char>& state, std::vector<float>& obs) override { inner_.reset(state, obs); }
  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>& actions, const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& new_states, std::vector<float>& observations,
                  std::vector<float>& rewards, std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override {
    inner_.step_batch(states, actions, dt, new_states, observations, rewards, dones, truncated);
    frames_.resize(states.size());
    for (size_t i = 0; i < states.size(); ++i) frames_[i] = (i % 2) ? dt[i] - 1 : dt[i];
  }
  int32_t frames_stepped(int32_t i) const override { return frames_[static_cast<size_t>(i)]; }
  void render_frame(const std::vector<char>&, std::vector<uint8_t>& rgba) override { rgba.clear(); }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }
 private:
  MockEnv inner_;
  std::vector<int32_t> frames_;
};
}  // namespace

TEST_CASE(wave_total_frames_counts_emulated_frames) {
  MockEnv env;
  FractalGasParams params;
  params.N = 8;
  params.seed = 3;
  FractalGas gas(env, params);
  gas.reset();
  CHECK(gas.total_frames() == 0);
  int64_t expected = 0;
  for (int it = 0; it < 6; ++it) {
    gas.step();
    for (const int32_t d : gas.state().dt) expected += d;  // MockEnv never stops early
    CHECK(gas.total_frames() == expected);
  }
  EarlyStopMockEnv early;
  FractalGas gas2(early, params);
  gas2.reset();
  int64_t expected2 = 0;
  for (int it = 0; it < 6; ++it) {
    gas2.step();
    const auto& dt = gas2.state().dt;
    for (size_t i = 0; i < dt.size(); ++i) expected2 += (i % 2) ? dt[i] - 1 : dt[i];
    CHECK(gas2.total_frames() == expected2);
  }
  CHECK(gas2.total_frames() < gas.total_frames() || params.N == 0);
}
