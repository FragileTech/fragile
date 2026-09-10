#include <cstring>
#include <string>

#include "control/physics.hpp"
#include "test_framework.hpp"

using namespace fg::control;
namespace {
std::shared_ptr<const Scene> pair_scene(const std::string& fields = "") {
  return Scene::compile(R"({"task":"tandem","size":[100,100],
    "environment":{"flight":false},"physics":{"dt":0.1,"substeps":4},
    "bodies":[{"position":[20,20],"controlled":true,"drag":0},
              {"position":[26,20],"controlled":true,"drag":0}])" + fields + "}");
}
float step(Physics& physics, StateBatch& state, int frames = 1) {
  std::vector<float> actions(physics.scene->channels.size(), 0);
  StepResult result;
  physics.step_world(state.row(0), actions.data(), frames, result);
  return result.reward;
}
template <class F>
bool throws(F fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}  // namespace

TEST_CASE(control_formation_pair_score_is_absolute_and_symmetric) {
  const auto scene = pair_scene(R"(,"formation_distance":5,"rewards":{"formation":1})");
  Physics physics(scene);
  StateBatch state(1, *scene);
  state.reset(*scene, 7);
  CHECK_CLOSE(step(physics, state), 5.f / 6, 1e-6);
  position(state.row(0), scene->layout, 1, {24, 20});
  CHECK_CLOSE(step(physics, state), 5.f / 6, 1e-6);
  position(state.row(0), scene->layout, 1, {25, 20});
  CHECK_CLOSE(step(physics, state, 5), 5, 1e-6);
  position(state.row(0), scene->layout, 0, {60, 60});
  position(state.row(0), scene->layout, 1, {60, 65});
  CHECK_CLOSE(step(physics, state), 1, 1e-6);
  position(state.row(0), scene->layout, 1, {60, 95});
  CHECK_CLOSE(step(physics, state), 1.f / 7, 1e-6);
}

TEST_CASE(control_formation_product_uses_all_controlled_pairs_and_scalar_fallback) {
  const auto scene = Scene::compile(R"({"task":"tandem","formation_distance":5,
    "rewards":{"formation":1},
    "formation_pairs":[{"a":2,"b":0,"distance":3},{"a":0,"b":3,"distance":4}],
    "bodies":[{"position":[20,20],"controlled":true},
              {"position":[50,40],"cargo":true},
              {"position":[23,20],"controlled":true},
              {"position":[20,24],"controlled":true}]})");
  CHECK(scene->formation_pairs.size() == 3);
  CHECK(scene->formation_pairs[0].a == 0);
  CHECK(scene->formation_pairs[0].b == 2);
  CHECK(scene->formation_pairs[2].distance == 5);
  Physics physics(scene);
  StateBatch state(1, *scene);
  state.reset(*scene, 7);
  CHECK_CLOSE(step(physics, state), 1, 1e-6);
  position(state.row(0), scene->layout, 2, {26, 20});
  position(state.row(0), scene->layout, 3, {20, 28});
  CHECK_CLOSE(step(physics, state), .125f, 1e-6);
}

TEST_CASE(control_formation_defaults_and_explicit_weights) {
  const auto scene = pair_scene();
  CHECK(scene->formation_distance == 3);
  CHECK(scene->formation_reward == 50);
  CHECK(scene->distance_squared_reward == 1);
  CHECK(scene->wall_collision_penalty == 100);
  CHECK(scene->collision_penalty == 2);
  CHECK(scene->progress_reward == 1);
  CHECK(scene->gate_reward == 30);
  CHECK(scene->pickup_reward == 0);
  CHECK(scene->delivery_reward == 0);
  CHECK(scene->hooked_rock_distance_reward == 0);
  CHECK(scene->catch_reward == 0);
  CHECK(scene->full_reward == 0);
  const auto custom = pair_scene(R"(,"rewards":{"formation":0,"progress":2,
    "gate":17,"pickup":7,"delivery":8,"hooked_rock_distance":9},
    "cargo":{"full_reward":11},"refineries":[{"position":[40,30]}])");
  CHECK(custom->formation_reward == 0);
  CHECK(custom->progress_reward == 2);
  CHECK(custom->gate_reward == 17);
  CHECK(custom->pickup_reward == 7);
  CHECK(custom->delivery_reward == 8);
  CHECK(custom->hooked_rock_distance_reward == 9);
  CHECK(custom->full_reward == 11);
  const auto cargo = pair_scene(R"(,"rewards":{"pickup":7},"cargo":{},
    "refineries":[{"position":[40,30]}])");
  CHECK(cargo->full_reward == 0);
  CHECK(pair_scene(R"(,"rewards":{"formation":100})")->formation_reward == 100);
  CHECK(throws([] { pair_scene(R"(,"rewards":{"formation":100.01})"); }));
}

TEST_CASE(control_formation_accumulates_with_travel_and_is_independent_of_progress) {
  for (int progress : {0, 2}) {
    const auto scene = pair_scene(
        R"(,"formation_distance":5,"rewards":{"formation":1,"progress":)" +
        std::to_string(progress) + "}");
    Physics physics(scene, 2);
    StateBatch state(1, *scene), serial(1, *scene), batch(1, *scene);
    state.reset(*scene, 7);
    velocity(state.row(0), scene->layout, 0, {2, 3});
    velocity(state.row(0), scene->layout, 1, {2, 3});
    std::memcpy(serial.row(0), state.row(0), state.bytes());
    float actions[4] = {};
    int32_t frames = 5;
    StepResult result;
    physics.step(state, nullptr, actions, &frames, batch, &result);
    CHECK_CLOSE(result.reward, 5.f * (5.f / 6 + .13f), 1e-4);
    float separate = 0;
    for (int i = 0; i < frames; ++i) separate += step(physics, serial);
    CHECK_CLOSE(result.reward, separate, 1e-6);
    CHECK(std::memcmp(batch.row(0), serial.row(0), batch.bytes()) == 0);
    // A changing formation with no navigation target must not add potential shaping.
    velocity(serial.row(0), scene->layout, 1, {5, 3});
    const Vec2 before_a = position(serial.row(0), scene->layout, 0);
    const Vec2 before_b = position(serial.row(0), scene->layout, 1);
    const float earned = step(physics, serial);
    const Vec2 a = position(serial.row(0), scene->layout, 0);
    const Vec2 b = position(serial.row(0), scene->layout, 1);
    CHECK_CLOSE(earned, 5.f / (5 + std::abs(5 - length(a - b))) +
                            (length2(a - before_a) + length2(b - before_b)) / 2, 1e-6);
  }
}

TEST_CASE(control_formation_ignores_other_tasks_and_requires_two_vehicles) {
  for (const auto& source : {
           R"({"task":"tandem","bodies":[{"position":[20,20]}]})",
           R"({"task":"tandem","bodies":[{"position":[20,20],"controlled":true}]})",
           R"({"rewards":{"formation":7},"bodies":[{"position":[20,20],"controlled":true},
                {"position":[25,20],"controlled":true}]})"}) {
    const auto scene = Scene::compile(source);
    Physics physics(scene);
    StateBatch state(1, *scene);
    state.reset(*scene, 7);
    CHECK(scene->formation_pairs.empty());
    CHECK(step(physics, state, 3) == 0);
  }
  const auto scene = pair_scene(R"(,"rewards":{"formation":0})");
  Physics physics(scene);
  StateBatch state(1, *scene);
  state.reset(*scene, 7);
  CHECK(step(physics, state, 3) == 0);
}

TEST_CASE(control_tandem_checkpoint_barrier_freezes_stage_and_normalizes_bonus) {
  const auto scene = pair_scene(R"(,"rewards":{"formation":0,"distance_squared":0,"progress":0},
    "gates":[{"position":[20,20],"radius":1},{"position":[26,20],"radius":1}])");
  Physics physics(scene);
  StateBatch state(1, *scene);
  state.reset(*scene, 7);
  auto* row = state.row(0);
  const auto& l = scene->layout;
  CHECK_CLOSE(step(physics, state), 15, 1e-6);
  CHECK(word(row, l.gates) == 1);
  CHECK(word(row, l.gates + 1) == 0);
  position(row, l, 0, {26,20});
  position(row, l, 1, {28,20});
  CHECK_CLOSE(step(physics, state), 0, 1e-6);
  CHECK(word(row, l.gates) == 1);
  position(row, l, 1, {20,20});
  CHECK_CLOSE(step(physics, state), 15, 1e-6);
  CHECK(word(row, l.gates) == 1); // Last arrival cannot unlock an earlier body this frame.
  CHECK(word(row, l.gates + 1) == 1);
  CHECK_CLOSE(step(physics, state), 15, 1e-6);
  CHECK(word(row, l.gates) == 2);
  CHECK(word(row, 6) == 3);
  // A restored leader several stages ahead cannot collect or shape reward.
  word(row, l.gates, 5);
  velocity(row, l, 0, {-2,0});
  CHECK_CLOSE(step(physics, state), 0, 1e-6);
  CHECK(word(row, l.gates) == 5);
}

TEST_CASE(control_tandem_proximity_includes_cleared_agents_and_never_penalizes_retreat) {
  const auto scene = pair_scene(R"(,"rewards":{"formation":0,"distance_squared":0,"gate":0},
    "gates":[{"position":[30,20],"radius":2},{"position":[40,20],"radius":1}])");
  Physics physics(scene);
  StateBatch state(1, *scene);
  state.reset(*scene, 7);
  auto* row = state.row(0);
  const auto& l = scene->layout;
  word(row, l.gates, 1);
  CHECK_CLOSE(step(physics, state), 2.f / 9, 1e-6); // Mean(10,4) = 7.
  position(row, l, 0, {24,20});
  CHECK_CLOSE(step(physics, state), 2.f / 7, 1e-6); // Cleared agent now closer.
  velocity(row, l, 0, {-10,0});
  CHECK_CLOSE(step(physics, state), 2.f / 7.5f, 1e-6);
  CHECK(word(row, l.gates) == 1);
}
