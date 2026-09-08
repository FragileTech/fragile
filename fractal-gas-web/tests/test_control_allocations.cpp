// Instrument ordinary allocation after the persistent pool and scratch warm up.
#include <atomic>
#include <cstdlib>
#include <iostream>

#include "control/physics.hpp"
#include "control/wave.hpp"
static std::atomic<size_t> allocations{0};
void* operator new(size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  ++allocations;
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void* operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }
using namespace fg::control;
int main() {
  auto scene = Scene::compile(
      R"({"rewards":{"distance_squared":1},"bodies":[{"position":[10,10],"controlled":true},{"position":[13,10],"cargo":true}],"tethers":[{"a":0,"b":1}],"pickups":[{"position":[10,10]}]})");
  for (bool resting : {false, true}) {
    if (resting)
      scene = Scene::compile(
          R"({"environment":{"flight":true},"bodies":[{"position":[10,0.5],"controlled":true,"vertices":[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]]},{"position":[13,0.5],"cargo":true}]})");
    for (int threads : {1, 4}) {
      size_t small = 0;
      for (int worlds : {16, 256}) {
        Physics physics(scene, threads);
        StateBatch a(worlds, *scene), b(worlds, *scene);
        a.reset(*scene, 7);
        std::vector<float> actions(worlds * 2, resting ? 0.f : .2f);
        std::vector<int32_t> frames(worlds, 6);
        std::vector<StepResult> result(worlds);
        physics.step(a, nullptr, actions.data(), frames.data(), b, result.data());
        allocations = 0;
        physics.step(a, nullptr, actions.data(), frames.data(), b, result.data());
        size_t count = allocations.load();
        std::cout << worlds << " worlds, " << threads << " threads: " << count
                  << " allocations per batch\n";
        if (worlds == 16)
          small = count;
        else if (count != small || count > size_t(threads + 2))
          return 1;
      }
    }
  }
  // Count the complete shared Wave lifecycle, including non-cumulative elite
  // restoration and action inheritance, after both state banks are warm.
  for (int threads : {1, 4}) {
    size_t small = 0;
    for (int walkers : {16, 256}) {
      Physics physics(scene, threads);
      StateBatch root(1, *scene);
      root.reset(*scene, 7);
      WaveConfig config;
      config.walkers = walkers;
      config.elites = 4;
      config.cumulative = false;
      config.recording = fg::RecordingMode::Off;
      PackedWave wave(physics, config, 7);
      wave.reset(root);
      for (int i = 0; i < 12; ++i) wave.step();
      allocations = 0;
      for (int i = 0; i < 10; ++i) wave.step();
      size_t count = allocations.load();
      std::cout << walkers << " walkers, " << threads << " threads: " << count / 10.0
                << " allocations per complete Wave iteration\n";
      if (walkers == 16)
        small = count;
      else if (count != small || count > 10 * size_t(threads + 3))
        return 1;
    }
  }
}
