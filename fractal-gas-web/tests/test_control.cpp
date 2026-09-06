#include <cstring>
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
