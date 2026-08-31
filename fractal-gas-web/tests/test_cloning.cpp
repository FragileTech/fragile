#include "cloning.hpp"
#include "test_framework.hpp"
#include "fixtures/fixtures_generated.hpp"

using namespace fg;

namespace {

FractalCloningOperator make_fixture_op() {
  FractalCloningOperator op;
  op.dist_coef = fixtures::kCloneDistCoef;
  op.reward_coef = fixtures::kCloneRewardCoef;
  op.use_cumulative_reward = fixtures::kCloneUseCumulative;
  return op;
}

}  // namespace

TEST_CASE(fitness_matches_python_with_recorded_companions) {
  const FractalCloningOperator op = make_fixture_op();
  const std::vector<float> vr = op.fitness_with_companions(
      fixtures::kCloneObs, fixtures::kCloneN, fixtures::kCloneDim,
      fixtures::kCloneCumRewards, fixtures::kCloneStepRewards,
      fixtures::kCloneFitCompanions);
  CHECK(vr.size() == fixtures::kCloneExpectedVr.size());
  for (size_t i = 0; i < vr.size(); ++i) {
    CHECK_CLOSE(vr[i], fixtures::kCloneExpectedVr[i], 1e-5);
  }
}

TEST_CASE(clone_probs_match_python) {
  const FractalCloningOperator op = make_fixture_op();
  const std::vector<float> probs = op.clone_probs_with_companions(
      fixtures::kCloneExpectedVr, fixtures::kCloneCloneCompanions);
  for (size_t i = 0; i < probs.size(); ++i) {
    CHECK_CLOSE(probs[i], fixtures::kCloneExpectedProbs[i], 1e-5);
  }
}

TEST_CASE(clone_decision_matches_python_with_recorded_uniforms) {
  const FractalCloningOperator op = make_fixture_op();
  const std::vector<float> probs = op.clone_probs_with_companions(
      fixtures::kCloneExpectedVr, fixtures::kCloneCloneCompanions);
  const std::vector<uint8_t> will_clone = op.decide_with_uniforms(
      probs, fixtures::kCloneUniforms, fixtures::kCloneAlive);
  CHECK(will_clone.size() == fixtures::kCloneExpectedWillClone.size());
  for (size_t i = 0; i < will_clone.size(); ++i) {
    CHECK(will_clone[i] == fixtures::kCloneExpectedWillClone[i]);
  }
}

TEST_CASE(dead_walkers_always_clone) {
  const FractalCloningOperator op;
  const std::vector<float> probs = {-1.0f, -1.0f, -1.0f};
  const std::vector<float> uniforms = {0.5f, 0.5f, 0.5f};
  const std::vector<uint8_t> alive = {1, 0, 1};
  const std::vector<uint8_t> will_clone =
      op.decide_with_uniforms(probs, uniforms, alive);
  CHECK(will_clone[0] == 0);
  CHECK(will_clone[1] == 1);  // dead -> forced clone despite negative prob
  CHECK(will_clone[2] == 0);
}
