#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>

#include "control/physics.hpp"
#include "control/wave.hpp"
#include "test_framework.hpp"
using namespace fg::control;

TEST_CASE(control_scheduler_preserves_static_mapping_and_visits_dynamic_once) {
  fg::ThreadPool pool(8);
  std::vector<std::atomic<int>> busy(8);
  for (auto& value : busy) value.store(0);
  for (int n : {0, 1, 3, 17, 128, 1025}) {
    for (int repeat = 0; repeat < 8; ++repeat) {
      std::vector<std::atomic<int>> visits(n);
      for (auto& value : visits) value.store(0);
      std::atomic<bool> exclusive{true};
      pool.parallel_for_dynamic(n, [&](int i, int slot) {
        if (busy[slot].fetch_add(1) != 0) exclusive.store(false);
        visits[i].fetch_add(1);
        if (busy[slot].fetch_sub(1) != 1) exclusive.store(false);
      });
      CHECK(exclusive.load());
      for (auto& value : visits) CHECK(value.load() == 1);
      std::vector<int> assigned(n, -1);
      pool.parallel_for(n, [&](int i, int slot) { assigned[i] = slot; });
      for (int slot = 0; slot < pool.size(); ++slot)
        for (int i = n * slot / pool.size(); i < n * (slot + 1) / pool.size(); ++i)
          CHECK(assigned[i] == slot);
    }
  }
}

TEST_CASE(control_scheduler_can_pass_a_blocked_first_job) {
  fg::ThreadPool pool(4);
  std::mutex mutex;
  std::condition_variable ready;
  bool reached_end = false, timed_out = false;
  // With static blocks, item 1 is trapped behind item 0. Dynamic chunks of
  // one let another slot finish all remaining work while item 0 is blocked.
  std::atomic<int> finished{0};
  pool.parallel_for_dynamic(16, [&](int i, int) {
    if (i == 0) {
      std::unique_lock<std::mutex> lock(mutex);
      timed_out = !ready.wait_for(lock, std::chrono::seconds(5), [&] { return reached_end; });
    } else if (finished.fetch_add(1) == 14) {
      std::lock_guard<std::mutex> lock(mutex);
      reached_end = true;
      ready.notify_one();
    }
  });
  CHECK(!timed_out);
  CHECK(finished.load() == 15);
}

TEST_CASE(control_dynamic_scheduling_preserves_heterogeneous_worlds_and_results) {
  auto scene = Scene::compile(R"({"size":[100,100],"task":"harvest","keep_delivered_rocks":true,"environment":{"flight":true},"physics":{"lethal_walls":true},"bodies":[{"position":[50,15],"controlled":true,"mass":1},{"position":[52,0.5],"radius":0.5,"cargo":true,"vertices":[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]]}],"tethers":[{"a":0,"b":-1,"automatic":true,"rest_length":2.5}],"bases":[{"position":[90,5],"radius":3}]})");
  constexpr int n = 129;
  Physics serial(scene, 1), dynamic(scene, 8);
  StateBatch root(n, *scene), expected(n, *scene), actual(n, *scene);
  root.reset(*scene, 7);
  std::vector<int32_t> sources(n), frames(n);
  std::vector<float> actions(n * scene->channels.size());
  std::vector<StepResult> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    sources[i] = (i * 3) % n; // Includes duplicate parents.
    frames[i] = i % 7 == 0 ? 0 : (i % 4 == 0 ? 36 : 3);
    position(root.row(i), scene->layout, 1, {float(10 + i % 80), i % 2 ? .5f : 30.f});
    velocity(root.row(i), scene->layout, 1, {float(i % 5), 0});
    if (i % 11 == 0) word(root.row(i), 7, 1);
    for (size_t d = 0; d < scene->channels.size(); ++d)
      actions[i * scene->channels.size() + d] = float((i + d) % 7) / 7;
  }
  serial.step(root, sources.data(), actions.data(), frames.data(), expected, a.data());
  for (int repeat = 0; repeat < 12; ++repeat) {
    dynamic.step(root, sources.data(), actions.data(), frames.data(), actual, b.data());
    CHECK(std::memcmp(expected.row(0), actual.row(0), expected.bytes()) == 0);
    for (int i = 0; i < n; ++i) {
      CHECK(a[i].reward == b[i].reward);
      CHECK(a[i].frames == b[i].frames);
      CHECK(a[i].collisions == b[i].collisions);
      CHECK(a[i].ccd_limits == b[i].ccd_limits);
      CHECK(a[i].dead == b[i].dead);
    }
  }
  WaveConfig config;
  config.walkers = 32; config.horizon = 12; config.frames = 3;
  PackedWave wave_a(serial, config, 19), wave_b(dynamic, config, 19);
  wave_a.reset(root, 1); wave_b.reset(root, 1);
  for (int generation = 0; generation < 12; ++generation) {
    wave_a.step(); wave_b.step();
    CHECK(std::memcmp(wave_a.current.row(0), wave_b.current.row(0), wave_a.current.bytes()) == 0);
    CHECK(wave_a.rewards == wave_b.rewards);
    CHECK(wave_a.actions == wave_b.actions);
    CHECK(wave_a.node_ids == wave_b.node_ids);
  }
}
