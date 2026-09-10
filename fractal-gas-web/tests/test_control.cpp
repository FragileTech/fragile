#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "control/physics.hpp"
#include "control/wave.hpp"
#include "test_framework.hpp"
using namespace fg::control;
namespace {
const std::string free_scene =
    R"({"version":1,"size":[100,100],"bodies":[{"position":[50,50],"velocity":[2,3],"drag":0,"angular_drag":0,"controlled":true}]})";
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
TEST_CASE(control_state_roundtrip_and_future) {
  auto scene = Scene::compile(free_scene);
  Physics physics(scene, 2);
  StateBatch a(4, *scene), b(4, *scene), c(4, *scene);
  a.reset(*scene, 123);
  std::vector<float> actions(8);
  std::vector<int32_t> dt(4, 6);
  std::vector<StepResult> results(4);
  physics.step(a, nullptr, actions.data(), dt.data(), b, results.data());
  std::vector<uint8_t> snapshot(b.serialized_size());
  b.serialize(snapshot.data(), snapshot.size());
  c.deserialize(snapshot.data(), snapshot.size());
  CHECK(std::memcmp(b.row(0), c.row(0), b.bytes()) == 0);
  CHECK(snapshot.size() == 32 + 4 * scene->layout.words * 4);
  physics.step(b, nullptr, actions.data(), dt.data(), a, results.data());
  physics.step(c, nullptr, actions.data(), dt.data(), b, results.data());
  CHECK(std::memcmp(a.row(0), b.row(0), a.bytes()) == 0);
  snapshot.back() ^= 1;
  CHECK(throws([&] { c.deserialize(snapshot.data(), snapshot.size()); }));
}
TEST_CASE(control_free_flight) {
  auto s = Scene::compile(free_scene);
  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[2] = {0, 0};
  int32_t dt = 60;
  StepResult result;
  p.step(a, nullptr, action, &dt, b, &result);
  CHECK_CLOSE(position(b.row(0), s->layout, 0).x, 52, 1e-5);
  CHECK_CLOSE(position(b.row(0), s->layout, 0).y, 53, 1e-5);
  CHECK(result.frames == 60);
}
TEST_CASE(control_flight_mode_gravity_and_propulsion) {
  const auto falling = Scene::compile(R"({"size":[100,100],"physics":{"dt":0.1,"substeps":1},
    "environment":{"flight":true,"downward_gravity":10},
    "bodies":[{"position":[50,50],"drag":0,"angular_drag":0,
      "controlled":true,"flight_capable":false}]})");
  Physics p(falling);
  StateBatch a(1, *falling), b(1, *falling);
  a.reset(*falling, 0);
  float action[2] = {};
  int32_t frames = 1;
  StepResult result;
  p.step(a, nullptr, action, &frames, b, &result);
  CHECK(falling->flight_mode);
  CHECK_CLOSE(position(b.row(0), falling->layout, 0).x, 50, 1e-5);
  CHECK_CLOSE(position(b.row(0), falling->layout, 0).y, 49.9, 1e-5);
  CHECK_CLOSE(velocity(b.row(0), falling->layout, 0).y, -1, 1e-5);

  const auto rocket = Scene::compile(R"({"size":[100,100],"physics":{"dt":0.1,"substeps":1},
    "environment":{"flight":true,"downward_gravity":9},
    "bodies":[{"position":[50,50],"angle":1.57079632679,"drag":0,
      "angular_drag":0,"controlled":true,"flight_capable":true}]})");
  Physics rocket_physics(rocket);
  StateBatch rocket_a(1, *rocket), rocket_b(1, *rocket);
  rocket_a.reset(*rocket, 0);
  float thrust[2] = {1, 0};
  rocket_physics.step(rocket_a, nullptr, thrust, &frames, rocket_b, &result);
  CHECK(position(rocket_b.row(0), rocket->layout, 0).y > 50);
}
TEST_CASE(control_flight_mode_auto_detection_and_override) {
  const auto automatic = Scene::compile(R"({"bodies":[
    {"position":[10,10],"controlled":true,"flight_capable":true}]})");
  CHECK(automatic->flight_mode);
  const auto passive = Scene::compile(R"({"bodies":[
    {"position":[10,10],"flight_capable":true}]})");
  CHECK(!passive->flight_mode);
  const auto disabled = Scene::compile(R"({"environment":{"flight":false},"bodies":[
    {"position":[10,10],"controlled":true,"flight_capable":true}]})");
  CHECK(!disabled->flight_mode);
  const auto forced = Scene::compile(R"({"environment":{"flight":true},"bodies":[
    {"position":[10,10],"controlled":true}]})");
  CHECK(forced->flight_mode);
}
TEST_CASE(control_flight_mode_is_reported_by_inspection) {
  const auto s = Scene::compile(R"({"size":[100,100],"physics":{"dt":0.1,"substeps":1},
    "environment":{"flight":true,"downward_gravity":10},
    "bodies":[{"position":[50,50],"drag":0}]})");
  Physics p(s);
  StateBatch state(1, *s);
  state.reset(*s, 0);
  const auto rows = p.inspect(state.row(0), nullptr);
  CHECK(rows.size() >= 8);
  CHECK_CLOSE(rows[5], 0, 1e-5);
  CHECK_CLOSE(rows[6], -10, 1e-5);
}
TEST_CASE(control_squared_distance_is_per_frame_and_vehicle_mean) {
  auto s = Scene::compile(R"({"size":[100,100],
    "physics":{"dt":0.1,"substeps":4},
    "rewards":{"progress":0,"distance_squared":2},
    "bodies":[
      {"position":[20,20],"velocity":[2,3],"drag":0,"controlled":true},
      {"position":[60,60],"drag":0,"controlled":true},
      {"position":[80,80],"velocity":[10,0],"drag":0,"cargo":true}]})");
  Physics p(s, 2);
  StateBatch a(1, *s), b(1, *s), serial(1, *s);
  a.reset(*s, 7);
  serial.reset(*s, 7);
  float action[4] = {};
  int32_t frames = 5;
  StepResult batch_result, single_result;
  p.step(a, nullptr, action, &frames, b, &batch_result);
  float total = 0;
  for (int i = 0; i < frames; ++i) {
    p.step_world(serial.row(0), action, 1, single_result);
    total += single_result.reward;
  }
  // 2 * ((0.2² + 0.3²) + 0) / 2 per frame. Cargo movement is excluded.
  CHECK_CLOSE(batch_result.reward, .65f, 1e-4);
  CHECK_CLOSE(batch_result.reward, total, 1e-6);
  CHECK(std::memcmp(b.row(0), serial.row(0), b.bytes()) == 0);
  std::vector<uint8_t> snapshot(b.serialized_size());
  b.serialize(snapshot.data(), snapshot.size());
  serial.deserialize(snapshot.data(), snapshot.size());
  p.step_world(b.row(0), action, 3, batch_result);
  p.step_world(serial.row(0), action, 3, single_result);
  CHECK_CLOSE(batch_result.reward, single_result.reward, 1e-6);
}
TEST_CASE(control_squared_distance_defaults_on_and_validates) {
  auto s = Scene::compile(free_scene);
  CHECK(s->distance_squared_reward == 1);
  Physics p(s);
  StateBatch a(1, *s);
  a.reset(*s, 7);
  float action[2] = {};
  StepResult result;
  p.step_world(a.row(0), action, 10, result);
  CHECK_CLOSE(result.reward, 10.f * 13.f / 3600.f, 1e-4);
  auto disabled = Scene::compile(
      R"({"rewards":{"distance_squared":0},"bodies":[{"controlled":true}]})");
  CHECK(disabled->distance_squared_reward == 0);
  CHECK(throws([] { Scene::compile(R"({"rewards":{"distance_squared":-1}})"); }));
  CHECK(throws([] { Scene::compile(R"({"rewards":{"distance_squared":1001}})"); }));
}
TEST_CASE(control_gather_has_simultaneous_semantics) {
  auto s = Scene::compile(free_scene);
  StateBatch a(3, *s);
  a.reset(*s, 0);
  a.row(0)[8] = 10;
  a.row(1)[8] = 20;
  a.row(2)[8] = 30;
  int32_t ix[3] = {1, 0, 0};
  a.gather(a, ix, 3);
  CHECK(a.row(0)[8] == 20);
  CHECK(a.row(1)[8] == 10);
  CHECK(a.row(2)[8] == 10);
}
TEST_CASE(control_thread_count_does_not_change_state) {
  auto s = Scene::compile(
      R"({"bodies":[{"position":[20,20],"controlled":true},{"position":[24,20],"cargo":true}],"gravity":[{"position":[32,22],"strength":20}],"tethers":[{"a":0,"b":1,"rest_length":3}]})");
  Physics p(s, 1), parallel(s, 4);
  StateBatch a(16, *s), b(16, *s), c(16, *s);
  a.reset(*s, 15);
  std::vector<float> actions(32, .2f);
  std::vector<int32_t> dt(16, 30);
  std::vector<StepResult> r(16), rr(16);
  p.step(a, nullptr, actions.data(), dt.data(), b, r.data());
  parallel.step(a, nullptr, actions.data(), dt.data(), c, rr.data());
  CHECK(std::memcmp(b.row(0), c.row(0), b.bytes()) == 0);
  for (int i = 0; i < 16; ++i) CHECK(r[i].reward == rr[i].reward);
}
TEST_CASE(control_64_threads_complete_sparse_and_large_batches) {
  auto scene = Scene::compile(free_scene);
  Physics serial(scene, 1), parallel(scene, 64);
  for (int worlds : {1, 17, 64, 129}) {
    StateBatch a(worlds, *scene), b(worlds, *scene), c(worlds, *scene);
    a.reset(*scene, 15);
    std::vector<float> actions(worlds * 2, .2f);
    std::vector<int32_t> frames(worlds, 3);
    std::vector<StepResult> r(worlds), rr(worlds);
    for (int step = 0; step < 20; ++step) {
      serial.step(a, nullptr, actions.data(), frames.data(), b, r.data());
      parallel.step(a, nullptr, actions.data(), frames.data(), c, rr.data());
      CHECK(std::memcmp(b.row(0), c.row(0), b.bytes()) == 0);
      for (int i = 0; i < worlds; ++i) CHECK(r[i].reward == rr[i].reward);
      std::memcpy(a.row(0), b.row(0), a.bytes());
    }
  }
}
TEST_CASE(control_high_speed_wall_and_hole) {
  auto s = Scene::compile(
      R"({"physics":{"lethal_walls":true},"holes":[[[30,10],[35,10],[35,30],[30,30]]],"bodies":[{"position":[10,20],"velocity":[3000,0],"drag":0,"controlled":true,"vertices":[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]]}]})");
  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[2] = {};
  int32_t dt = 1;
  StepResult r;
  p.step(a, nullptr, action, &dt, b, &r);
  CHECK(r.dead);
  CHECK(r.collisions > 0);
  CHECK(position(b.row(0), s->layout, 0).x < 30);
}
TEST_CASE(control_scene_validation_and_geometry_independent_snapshot) {
  auto a = Scene::compile(free_scene);
  auto b = Scene::compile(
      R"({"size":[100,100],"holes":[[[2,2],[4,2],[4,4],[2,4]]],"bodies":[{"position":[50,50],"controlled":true}]})");
  CHECK(a->layout.words == b->layout.words);
  CHECK(throws(
      [] { Scene::compile(R"({"bodies":[{"position":[5,5],"mass":0}]})"); }));
  CHECK(throws([] {
    Scene::compile(
        R"({"boundary":[[0,0],[40,40],[40,0],[0,40]],"bodies":[{}]})");
  }));
  CHECK(throws([] { JsonReader("{\"x\":01}").read(); }));
  CHECK(JsonReader("\"\\uD83D\\uDE80\"").read().string == "🚀");
}
TEST_CASE(control_pickup_and_delivery) {
  auto s = Scene::compile(
      R"({"bodies":[{"position":[10,10],"controlled":true},{"position":[30,30],"cargo":true}],"pickups":[{"position":[10,10]}],"bases":[{"position":[30,30],"radius":2}]})");
  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[2] = {};
  int32_t dt = 1;
  StepResult r;
  p.step(a, nullptr, action, &dt, b, &r);
  CHECK(word(b.row(0), 4) == 1);
  CHECK(word(b.row(0), 5) == 1);
  CHECK(r.reward >= 110);
}
TEST_CASE(control_dynamic_ccd_and_restitution) {
  auto s = Scene::compile(
      R"({"size":[100,100],"physics":{"substeps":1},"bodies":[{"position":[20,50],"velocity":[1800,0],"radius":1,"mass":1,"drag":0,"friction":0,"restitution":1,"controlled":true},{"position":[40,50],"radius":1,"mass":1,"drag":0,"friction":0,"restitution":1}]})");
  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[2] = {};
  int32_t dt = 1;
  StepResult r;
  p.step(a, nullptr, action, &dt, b, &r);
  CHECK(r.collisions > 0);
  CHECK(position(b.row(0), s->layout, 0).x <
        position(b.row(0), s->layout, 1).x);
  CHECK_CLOSE(velocity(b.row(0), s->layout, 0).x, 0, 1e-4);
  CHECK_CLOSE(velocity(b.row(0), s->layout, 1).x, 1800, 1e-4);
}
TEST_CASE(control_wave_clone_elite_lineage_replays_exactly) {
  auto s = Scene::compile(
      R"({"bodies":[{"position":[20,20],"controlled":true},{"position":[24,20],"cargo":true}],"pickups":[{"position":[21,20]}],"tethers":[{"a":0,"b":1}]})");
  Physics p(s, 3);
  StateBatch root(1, *s);
  root.reset(*s, 7);
  for (auto mode : {fg::RecordingMode::Full, fg::RecordingMode::Pruned}) {
    WaveConfig config;
    config.walkers = 24;
    config.elites = 3;
    config.recording = mode;
    PackedWave wave(p, config, 42);
    wave.reset(root);
    for (int iteration = 0; iteration < 8; ++iteration) wave.step();
    for (size_t i = 0; i < wave.current.count; ++i) {
      auto replay = wave.replay(wave.node_ids[i]);
      CHECK(std::memcmp(replay.row(0), wave.current.row(i),
                        s->layout.words * 4) == 0);
    }
    auto action = wave.select_action();
    CHECK(action[0] >= 0 && action[0] <= 1);
    CHECK(action[1] >= -1 && action[1] <= 1);
  }
}
TEST_CASE(control_rejects_bad_batch_before_mutation) {
  auto s = Scene::compile(free_scene);
  Physics p(s);
  StateBatch a(2, *s), b(2, *s);
  a.reset(*s, 3);
  b.reset(*s, 4);
  StateBatch before = b;
  std::vector<uint8_t> bytes(b.bytes());
  std::memcpy(bytes.data(), b.row(0), bytes.size());
  float actions[4] = {};
  int32_t frames[2] = {1, -1};
  StepResult r[2];
  CHECK(throws([&] { p.step(a, nullptr, actions, frames, b, r); }));
  CHECK(std::memcmp(bytes.data(), b.row(0), bytes.size()) == 0);
}
TEST_CASE(control_stiff_tether_damps_and_breaks) {
  auto s = Scene::compile(
      R"({"size":[100,100],"bodies":[{"position":[45,50],"controlled":true,"drag":0},{"position":[55,50],"mass":2,"drag":0}],"tethers":[{"a":0,"b":1,"rest_length":3,"stiffness":10000,"damping":100,"break_force":1000000}]})");
  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[2] = {};
  int32_t dt = 120;
  StepResult result;
  p.step(a, nullptr, action, &dt, b, &result);
  CHECK_CLOSE(length(position(b.row(0), s->layout, 0) -
                     position(b.row(0), s->layout, 1)),
              3, .02);
  CHECK_CLOSE(velocity(b.row(0), s->layout, 0).x +
                  2 * velocity(b.row(0), s->layout, 1).x,
              0, .001);
  auto breaking = Scene::compile(
      R"({"bodies":[{"position":[20,20],"controlled":true},{"position":[30,20]}],"tethers":[{"a":0,"b":1,"rest_length":1,"stiffness":100,"break_force":1}]})");
  Physics bp(breaking);
  StateBatch x(1, *breaking), y(1, *breaking);
  x.reset(*breaking, 0);
  dt = 1;
  bp.step(x, nullptr, action, &dt, y, &result);
  CHECK(word(y.row(0), breaking->layout.joints) == 0);
}
TEST_CASE(control_gravity_pickup_rng_and_gate_progress_restore) {
  auto s = Scene::compile(
      R"({"respawn_seconds":0.02,"bodies":[{"position":[15,15],"controlled":true}],"gravity":[{"position":[25,15],"strength":100}],"pickups":[{"position":[15,15]}],"gates":[{"position":[15,15],"radius":2},{"position":[25,15],"radius":2}]})");
  Physics p(s);
  StateBatch a(1, *s), b(1, *s), c(1, *s);
  a.reset(*s, 42);
  float action[2] = {};
  int32_t dt = 10;
  StepResult r;
  p.step(a, nullptr, action, &dt, b, &r);
  CHECK(velocity(b.row(0), s->layout, 0).x > 0);
  CHECK(word(b.row(0), 6) == 1);
  CHECK(word(b.row(0), 5) == 1);
  CHECK(rng_state(b.row(0)) != 42);
  std::vector<uint8_t> snapshot(b.serialized_size());
  b.serialize(snapshot.data(), snapshot.size());
  c.deserialize(snapshot.data(), snapshot.size());
  p.step(b, nullptr, action, &dt, a, &r);
  p.step(c, nullptr, action, &dt, b, &r);
  CHECK(std::memcmp(a.row(0), b.row(0), s->layout.words * 4) == 0);
}

TEST_CASE(control_agent_archetypes_compile_without_state_overhead) {
  const auto typed = Scene::compile(R"({"size":[100,100],"agent_types":{
    "base":{"physics":{"controlled":true,"mass":2,"thrust":8,"drag":0.3}},
    "custom":{"extends":"base","physics":{"torque":4},"visual":{"model":"kit"}}
  },"bodies":[{"agent_type":"custom","position":[50,50],"mass":3}]})");
  const auto explicit_scene = Scene::compile(R"({"size":[100,100],"bodies":[{
    "controlled":true,"position":[50,50],"mass":3,"thrust":8,"drag":0.3,"torque":4}]})");
  CHECK(typed->controlled.size() == 1);
  CHECK(typed->bodies[0].mass == 3);
  CHECK(typed->bodies[0].thrust == 8);
  CHECK(typed->bodies[0].torque == 4);
  CHECK(typed->layout.words == explicit_scene->layout.words);
  Physics a(typed, 2), b(explicit_scene, 2);
  StateBatch x(4, *typed), y(4, *explicit_scene), out_x(4, *typed),
      out_y(4, *explicit_scene);
  x.reset(*typed, 7);
  y.reset(*explicit_scene, 7);
  std::vector<float> actions(8, 0.5f);
  std::vector<int32_t> frames(4, 8);
  std::vector<StepResult> results(4);
  a.step(x, nullptr, actions.data(), frames.data(), out_x, results.data());
  b.step(y, nullptr, actions.data(), frames.data(), out_y, results.data());
  CHECK(std::memcmp(out_x.row(0), out_y.row(0), out_x.bytes()) == 0);
  CHECK(throws([] {
    Scene::compile(
        R"({"agent_types":{"a":{"extends":"b"},"b":{"extends":"a"}},"bodies":[{"position":[2,2]}]})");
  }));
  CHECK(throws([] {
    Scene::compile(R"({"bodies":[{"agent_type":"missing","position":[2,2]}]})");
  }));
  CHECK(throws([] {
    Scene::compile(
        R"({"agent_types":{"a":{"physics":[]}},"bodies":[{"position":[2,2]}]})");
  }));
}

TEST_CASE(control_variable_actuator_channels_and_kart_grip) {
  auto s = Scene::compile(R"({"size":[100,100],"bodies":[
    {"controlled":true,"position":[30,30],"velocity":[6,4],"drag":0,"angular_drag":0,"actuator":{"kind":"kart"}},
    {"controlled":true,"position":[60,60],"drag":0,"actuator":{"kind":"thrusters","thrusters":[
      {"position":[0,1],"direction":[1,0],"force":10},
      {"position":[0,-1],"direction":[1,0],"force":10}]}}
  ]})");
  CHECK(s->channels.size() == 5);
  CHECK(s->channels[0].low == -1);
  CHECK(s->channels[2].low == 0);
  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[5] = {0, 0, 1, 1, 1};
  int32_t frames = 30;
  StepResult result;
  p.step(a, nullptr, action, &frames, b, &result);
  CHECK(std::abs(velocity(b.row(0), s->layout, 0).x) < .01);
  CHECK(std::abs(velocity(b.row(0), s->layout, 0).y) < .02);
  CHECK(velocity(b.row(0), s->layout, 1).x > 0);
  CHECK_CLOSE(omega(b.row(0), s->layout, 1), 0, 1e-5);
  WaveConfig config;
  config.walkers = 8;
  config.elites = 2;
  config.recording = fg::RecordingMode::Full;
  PackedWave wave(p, config, 7);
  wave.reset(a);
  wave.step();
  CHECK(wave.tree.action_dim() == 5);
  CHECK(wave.tree.pose_dim() == 4);
  CHECK(wave.select_action().size() == 5);
  for (size_t i = 0; i < wave.current.count; i++) {
    auto replay = wave.replay(wave.node_ids[i]);
    CHECK(std::memcmp(replay.row(0), wave.current.row(i),
                      s->layout.words * 4) == 0);
  }
}

TEST_CASE(control_action_multipliers_scale_builtin_channels_and_force) {
  auto s = Scene::compile(R"({"size":[100,100],"physics":{"dt":0.1,"substeps":1},
    "bodies":[{"controlled":true,"position":[30,30],"drag":0,"angular_drag":0,
      "thrust":4,"torque":3,"actuator":{"kind":"vector",
      "action_multipliers":{"thrust":2,"torque":0.5}}}]})");
  CHECK(s->channels.size() == 2);
  CHECK(s->channels[0].low == 0);
  CHECK(s->channels[0].high == 2);
  CHECK(s->channels[1].low == -0.5f);
  CHECK(s->channels[1].high == 0.5f);

  Physics p(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 0);
  float action[2] = {2, 0.5f};
  int32_t frames = 1;
  StepResult result;
  p.step(a, nullptr, action, &frames, b, &result);
  CHECK_CLOSE(velocity(b.row(0), s->layout, 0).x, 0.8f, 1e-5);
  CHECK_CLOSE(omega(b.row(0), s->layout, 0), 1.2f, 1e-5);

  auto disabled = Scene::compile(R"({"bodies":[{"controlled":true,
    "actuator":{"kind":"vector","action_multipliers":{"thrust":0,"torque":0}}}]})");
  CHECK(disabled->channels[0].low == disabled->channels[0].high);
  CHECK(disabled->channels[1].low == disabled->channels[1].high);
  CHECK(throws([] {
    Scene::compile(R"({"bodies":[{"controlled":true,"actuator":{
      "kind":"vector","action_multipliers":{"thrust":-0.1}}}]})");
  }));
  CHECK(throws([] {
    Scene::compile(R"({"bodies":[{"controlled":true,"actuator":{
      "kind":"vector","action_multipliers":{"torque":10.1}}}]})");
  }));
}

TEST_CASE(control_custom_actuator_state_clones_and_replays) {
  register_actuator(
      "test_accumulator",
      [](const Json&) {
        ActuatorDef d;
        d.channels = {{"charge", -2, 2}};
        d.initial_state = {0};
        return d;
      },
      [](const BodyDef&, Vec2, float, float, const float* action, float dt,
         float* state) {
        state[0] += action[0] * dt;
        return ActuatorForce{{state[0], 0}, 0};
      });
  auto s = Scene::compile(
      R"({"size":[100,100],"bodies":[{"controlled":true,"position":[50,50],"actuator":{"kind":"test_accumulator"}}]})");
  CHECK(s->channels.size() == 1);
  CHECK(s->layout.auxiliary_words == 1);
  Physics serial(s, 1), parallel(s, 4);
  StateBatch a(8, *s), b(8, *s), c(8, *s);
  a.reset(*s, 7);
  float actions[8] = {1, 20, 1, 2, 1, 2, 1, 2};
  int32_t frames[8] = {10, 10, 10, 10, 10, 10, 10, 10};
  StepResult results[8];
  serial.step(a, nullptr, actions, frames, b, results);
  parallel.step(a, nullptr, actions, frames, c, results);
  CHECK(std::memcmp(b.row(0), c.row(0), b.bytes()) == 0);
  CHECK_CLOSE(b.row(0)[s->layout.auxiliary], 1.f / 6, 1e-5);
  CHECK_CLOSE(b.row(1)[s->layout.auxiliary], b.row(3)[s->layout.auxiliary],
              1e-6);
  auto before = b.row(1)[s->layout.auxiliary];
  int32_t indices[8] = {1, 0, 1, 0, 1, 0, 1, 0};
  b.gather(b, indices, 8);
  CHECK(b.row(0)[s->layout.auxiliary] == before);
  auto debug = serial.inspect(b.row(0), actions);
  CHECK(!debug.empty());
  CHECK(b.row(0)[s->layout.auxiliary] == before);
  std::vector<uint8_t> bytes(b.serialized_size());
  b.serialize(bytes.data(), bytes.size());
  c.deserialize(bytes.data(), bytes.size());
  CHECK(std::memcmp(b.row(0), c.row(0), b.bytes()) == 0);
}

TEST_CASE(control_world_extension_adds_reward_sensor_and_packed_state) {
  register_world_extension("test_counter", [](const Json&) {
    WorldExtension extension;
    extension.initial_state = {2};
    extension.observation_size = 1;
    extension.step = [](const Scene& s, const WorldExtension& e, float* row,
                        const float*, StepResult& result) {
      row[s.layout.auxiliary + e.state_offset] += 1;
      result.reward += 3;
    };
    extension.observe = [](const Scene& s, const WorldExtension& e,
                           const float* row, float* out) {
      out[0] = row[s.layout.auxiliary + e.state_offset];
    };
    return extension;
  });
  auto s = Scene::compile(
      R"({"extensions":[{"kind":"test_counter"}],"bodies":[{"controlled":true,"position":[20,20]}]})");
  Physics p(s, 2);
  StateBatch root(1, *s), out(1, *s);
  root.reset(*s, 7);
  float action[2] = {};
  int32_t frames = 5;
  StepResult result;
  p.step(root, nullptr, action, &frames, out, &result);
  CHECK_CLOSE(result.reward, 15, 1e-6);
  CHECK(p.observation_dim() == 9);
  std::vector<float> observations(p.observation_dim());
  p.observe(out.row(0), observations.data());
  CHECK(observations.back() == 7);
  WaveConfig config;
  config.walkers = 8;
  config.elites = 2;
  config.recording = fg::RecordingMode::Full;
  PackedWave wave(p, config, 7);
  wave.reset(root);
  wave.step();
  for (size_t i = 0; i < wave.current.count; i++) {
    auto replay = wave.replay(wave.node_ids[i]);
    CHECK(std::memcmp(replay.row(0), wave.current.row(i),
                      s->layout.words * 4) == 0);
  }
}

TEST_CASE(control_plugin_bounds_need_not_include_zero) {
  register_actuator(
      "test_positive",
      [](const Json&) {
        ActuatorDef d;
        d.channels = {{"position_target", 2, 3}};
        return d;
      },
      [](const BodyDef&, Vec2, float, float, const float* u, float, float*) {
        return ActuatorForce{{u[0], 0}, 0};
      });
  auto s = Scene::compile(
      R"({"bodies":[{"controlled":true,"position":[20,20],"actuator":{"kind":"test_positive"}}]})");
  Physics p(s);
  StateBatch root(1, *s), a(1, *s), b(1, *s);
  root.reset(*s, 7);
  float low = 2, zero = 0;
  int32_t frame = 1;
  StepResult result;
  p.step(root, nullptr, &low, &frame, a, &result);
  p.step(root, nullptr, &zero, &frame, b, &result);
  CHECK(std::memcmp(a.row(0), b.row(0), a.bytes()) == 0);
  WaveConfig config;
  config.walkers = 8;
  PackedWave wave(p, config, 7);
  wave.reset(root);
  CHECK(wave.select_action()[0] == 2);
  wave.step();
  CHECK(wave.select_action()[0] >= 2);
  CHECK(wave.select_action()[0] <= 3);
}

TEST_CASE(control_mining_heavy_load_and_replenishment) {
  std::ifstream file(std::filesystem::path(__FILE__).parent_path() /
                     "../web/lab/scenarios/mining.json");
  CHECK(file.good());
  std::ostringstream json;
  json << file.rdbuf();
  auto s = Scene::compile(json.str());
  CHECK(s->bodies.size() == 5);
  CHECK_CLOSE(s->bodies[2].mass, .24f, 1e-6);
  CHECK(s->bodies[2].drag == .8f);
  CHECK(s->bodies[2].respawn);
  CHECK(s->keep_delivered_rocks);
  // Keep this overload/coop stress case separate from the liftable preset.
  auto overloaded = std::make_shared<Scene>(*s);
  overloaded->keep_delivered_rocks = false;  // Explicit legacy respawn coverage.
  overloaded->bodies[2].inertia *= 24 / overloaded->bodies[2].mass;
  overloaded->bodies[2].mass = 24;
  s = overloaded;
  Physics p(s);
  StateBatch a(1, *s), b(1, *s), replay(1, *s);
  const auto& l = s->layout;
  float action[4] = {1, 0, 1, 0};
  int32_t frames = 120;
  StepResult result;
  a.reset(*s, 0);
  position(a.row(0), l, 0, {25, 11});
  // Detach the second rocket and move it beyond automatic hook range.
  word(a.row(0), l.joints + 2, 0);
  position(a.row(0), l, 1, {40, 10});
  position(a.row(0), l, s->tethers[1].a, {40, 7});
  p.step(a, nullptr, action, &frames, b, &result);
  const float solo_distance = position(b.row(0), l, 2).x - s->bodies[2].position.x;
  CHECK(solo_distance > .01f);
  CHECK(word(b.row(0), 4) == 0);
  // Exercise two swinging hook assemblies hauling the same heavy rock.
  a.reset(*s, 0);
  position(a.row(0), l, 0, {25, 11});
  position(a.row(0), l, 1, {25, 14});
  p.step(a, nullptr, action, &frames, b, &result);
  const float team_distance = position(b.row(0), l, 2).x - s->bodies[2].position.x;
  CHECK(std::isfinite(team_distance)); // Swinging loads need not move monotonically in x.
  // Repeated deliveries reuse the one cargo slot immediately and replay exactly.
  frames = 1;
  int quadrants[4] = {};
  for (int delivery = 1; delivery <= 256; ++delivery) {
    position(a.row(0), l, 2, s->bases[0].position);
    position(a.row(0), l, 0, {9, 10});
    position(a.row(0), l, 1, {9, 15});
    word(a.row(0), l.joints, 3);
    word(a.row(0), l.joints + 2, 3);
    p.step(a, nullptr, action, &frames, b, &result);
    p.step(a, nullptr, action, &frames, replay, &result);
    CHECK(std::memcmp(b.row(0), replay.row(0), l.words * sizeof(float)) == 0);
    CHECK(word(b.row(0), 4) == uint32_t(delivery));
    CHECK(word(b.row(0), l.flags + 2) == active_flag);
    const Vec2 spawned = position(b.row(0), l, 2);
    CHECK(s->inside(spawned));
    CHECK(length(spawned - s->bodies[2].position) > .01f);
    ++quadrants[(spawned.x >= s->size.x / 2) + 2 * (spawned.y >= s->size.y / 2)];
    for (const auto& edge : s->edges)
      CHECK(length(spawned - closest(spawned, edge.a, edge.b)) > s->bodies[2].radius);
    for (const auto& base : s->bases)
      CHECK(length(spawned - base.position) > s->bodies[2].radius + base.radius);
    for (size_t rocket = 0; rocket < 2; ++rocket)
      CHECK(length(spawned - position(b.row(0), l, rocket)) >
            s->bodies[2].radius + s->bodies[rocket].radius);
    CHECK_CLOSE(length(velocity(b.row(0), l, 2)), 0, 1e-6);
    CHECK_CLOSE(angle(b.row(0), l, 2), s->bodies[2].angle, 1e-6);
    CHECK_CLOSE(omega(b.row(0), l, 2), 0, 1e-6);
    for (size_t tether = 0; tether < s->tethers.size(); ++tether)
      if (!s->tethers[tether].permanent && word(b.row(0), l.joints + 2 * tether))
        CHECK(length(spawned - position(b.row(0), l, s->tethers[tether].a)) <
              s->tethers[tether].hook_range);
    std::memcpy(a.row(0), b.row(0), l.words * sizeof(float));
  }
  for (int count : quadrants) CHECK(count > 10);
}

TEST_CASE(control_cargo_respawn_with_no_free_location_is_bounded) {
  auto s = Scene::compile(
      R"({"size":[20,20],"bodies":[{"position":[2,2],"controlled":true},{"position":[10,10],"cargo":true,"respawn":true}],"bases":[{"position":[10,10],"radius":100}]})");
  Physics p(s);
  StateBatch a(1, *s), b(1, *s), replay(1, *s);
  a.reset(*s, 7);
  float action[2] = {};
  int32_t frames = 10;
  StepResult result;
  p.step(a, nullptr, action, &frames, b, &result);
  p.step(a, nullptr, action, &frames, replay, &result);
  CHECK(word(b.row(0), 0) == 10);
  CHECK(word(b.row(0), 4) == 1);
  CHECK(word(b.row(0), s->layout.flags + 1) == delivered_flag);
  CHECK(std::memcmp(b.row(0), replay.row(0), s->layout.words * sizeof(float)) == 0);
}

TEST_CASE(control_wave_best_leaf_uses_final_rewards_and_stable_ties) {
  auto scene = Scene::compile(free_scene);
  Physics physics(scene, 1);
  StateBatch root(1, *scene);
  root.reset(*scene, 7);
  WaveConfig config;
  config.walkers = 4;
  config.recording = fg::RecordingMode::Pruned;
  PackedWave wave(physics, config, 13);
  wave.reset(root);
  CHECK(throws([&] { wave.best_leaf(); }));
  wave.step();
  wave.step();
  wave.rewards = {-5, -2, -2, -10};
  CHECK(wave.best_leaf() == wave.node_ids[1]);
  word(wave.current.row(3), 7, 1);
  wave.rewards[3] = 4;
  CHECK(wave.best_leaf() == wave.node_ids[1]);
  word(wave.current.row(1), 7, 1);
  CHECK(wave.best_leaf() == wave.node_ids[2]);
  word(wave.current.row(2), 7, 1);
  CHECK(wave.best_leaf() == wave.node_ids[0]);
  word(wave.current.row(0), 7, 1);
  CHECK(wave.best_leaf() == wave.node_ids[3]);
  wave.rewards = {-5, -2, -2, -10};
  CHECK(wave.best_leaf() == wave.node_ids[1]);
  wave.rewards[3] = 4;
  const auto branch = wave.tree.branch(wave.best_leaf());
  CHECK(branch.size() == 3);
  CHECK(wave.tree.node(branch.back()).id == wave.node_ids[3]);
}

TEST_CASE(control_wave_common_ancestor_uses_alive_final_population) {
  auto scene = Scene::compile(free_scene);
  Physics physics(scene, 1);
  StateBatch root(1, *scene);
  root.reset(*scene, 7);
  WaveConfig config;
  config.walkers = 4;
  PackedWave wave(physics, config, 13);
  wave.reset(root);
  CHECK(throws([&] { wave.common_ancestor(); }));
  wave.step();
  const auto root_id = wave.tree.branch(wave.node_ids[0]).front();
  const auto shared = wave.node_ids[0];
  float action[2] = {0, 0}, pose[2] = {0, 0};
  const auto a = wave.tree.append(shared, 2, action, pose, -3, 0, 0, 0);
  const auto b = wave.tree.append(shared, 1, action, pose, -4, 0, 0, 0);
  const auto other = wave.node_ids[1];
  wave.node_ids = {a, b, other, other};
  CHECK(wave.common_ancestor() == root_id);
  word(wave.current.row(2), 7, 1);
  word(wave.current.row(3), 7, 1);
  CHECK(wave.common_ancestor() == shared);
  word(wave.current.row(1), 7, 1);
  CHECK(wave.common_ancestor() == a);
  wave.tree.prune(wave.node_ids);
  CHECK(wave.common_ancestor() == a);
  word(wave.current.row(0), 7, 1);
  CHECK(wave.common_ancestor() == 0);
}

TEST_CASE(control_mining_retained_delivery_keeps_motion_and_releases_both_hooks) {
  std::ifstream file(std::filesystem::path(__FILE__).parent_path() /
                     "../web/lab/scenarios/mining.json");
  CHECK(file.good());
  std::ostringstream json;
  json << file.rdbuf();
  auto s = Scene::compile(json.str());
  CHECK(s->keep_delivered_rocks);
  Physics physics(s);
  StateBatch a(1, *s), b(1, *s);
  a.reset(*s, 7);
  const auto& l = s->layout;
  position(a.row(0), l, 2, s->bases[0].position);
  float action[4] = {};
  int32_t frames = 1;
  StepResult result;
  physics.step(a, nullptr, action, &frames, b, &result);
  CHECK(word(b.row(0), 4) == 1);
  CHECK(word(b.row(0), l.flags + 2) == (active_flag | delivered_flag));
  CHECK(word(b.row(0), l.joints) == 0);
  CHECK(word(b.row(0), l.joints + 2) == 0);
  CHECK(word(b.row(0), l.joints + 4) != 0);
  CHECK(word(b.row(0), l.joints + 6) != 0);
  const float delivered_y = position(b.row(0), l, 2).y;
  frames = 120;
  physics.step(b, nullptr, action, &frames, a, &result);
  CHECK(word(a.row(0), 4) == 1);
  CHECK(word(a.row(0), l.flags + 2) == active_flag);
  CHECK(position(a.row(0), l, 2).y < delivered_y - s->bases[0].radius);
}

TEST_CASE(control_retained_delivery_lock_clones_and_releases) {
  auto s = Scene::compile(R"({"task":"harvest","size":[100,100],
    "keep_delivered_rocks":true,"physics":{"dt":0.1},
    "bodies":[{"controlled":true,"position":[10,10],"drag":0},
      {"cargo":true,"respawn":true,"position":[50,50],"drag":0}],
    "bases":[{"position":[50,50],"radius":4}],
    "tethers":[{"a":0,"b":1,"automatic":true,"stiffness":0,"damping":0}],
    "rewards":{"progress":0,"distance_squared":0,"catch":10}})");
  Physics physics(s);
  StateBatch a(1, *s), b(1, *s), restored(1, *s);
  a.reset(*s, 7);
  float action[2] = {};
  int32_t frames = 1;
  StepResult result;
  const auto& l = s->layout;
  physics.step(a, nullptr, action, &frames, b, &result);
  CHECK(word(b.row(0), l.flags + 1) == (active_flag | delivered_flag));
  CHECK(word(b.row(0), 4) == 1);
  CHECK(word(b.row(0), l.joints) == 0);
  CHECK(word(b.row(0), l.joints + 2) != 0);
  CHECK_CLOSE(position(b.row(0), l, 1).x, 50, 1e-6);
  CHECK_CLOSE(result.reward, 0, 1e-6);
  std::vector<uint8_t> bytes(b.serialized_size());
  b.serialize(bytes.data(), bytes.size());
  restored.deserialize(bytes.data(), bytes.size());
  CHECK(std::memcmp(b.row(0), restored.row(0), b.bytes()) == 0);
  position(restored.row(0), l, 1, {54, 50});
  position(restored.row(0), l, s->tethers[0].a, {53, 50});
  physics.step(restored, nullptr, action, &frames, a, &result);
  CHECK(word(a.row(0), l.flags + 1) == (active_flag | delivered_flag));
  CHECK(word(a.row(0), 4) == 1);
  CHECK(word(a.row(0), l.joints) == 0);
  position(a.row(0), l, 1, {54.01f, 50});
  physics.step(a, nullptr, action, &frames, b, &result);
  CHECK(word(b.row(0), l.flags + 1) == active_flag);
  CHECK(word(b.row(0), l.joints) == 2);
  CHECK_CLOSE(result.reward, 10, 1e-6);
}
