#include <deque>

#include "fractal_gas.hpp"
#include "mock_env.hpp"
#include "test_framework.hpp"
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
