// Same harness links against the pre-refactor and current native libraries.
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "arcade_planner.hpp"
#include "control/wave.hpp"
#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include "mock_env.hpp"
#include "optimization/engine.hpp"

static std::atomic<size_t> calls{0}, bytes{0};
static std::string workload_filter;
void* operator new(size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  ++calls;
  bytes += n;
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void* operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }
template <class Setup>
void bench(const char* label, Setup setup) {
  if (!workload_filter.empty() && workload_filter != label) return;
  std::vector<double> times, counts, sizes;
  times.reserve(9);
  counts.reserve(9);
  sizes.reserve(9);
  for (int repeat = 0; repeat < 9; ++repeat) {
    auto step = setup();
    for (int i = 0; i < 8; ++i) step();
    calls = 0;
    bytes = 0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 40; ++i) step();
    times.push_back(
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
            .count() /
        40);
    counts.push_back(double(calls) / 40);
    sizes.push_back(double(bytes) / 40);
  }
  auto median = [](auto v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << label << ',' << median(times) << ',' << median(counts) << ',' << median(sizes)
            << ',' << usage.ru_maxrss << '\n';
}
int main(int argc, char** argv) {
  if (argc > 1) workload_filter = argv[1];
  using namespace fg;
  std::cout << "workload,us_per_iteration,allocations_per_iteration,allocated_bytes_per_iteration,"
               "process_peak_rss_kib\n";
  for (int mode = 0; mode < 6; ++mode) {
    auto label = std::string("lab-") + std::to_string(mode % 3) + (mode >= 3 ? "-4t" : "");
    bench(label.c_str(), [mode] {
      struct Run {
        std::shared_ptr<const control::Scene> scene = control::Scene::compile(
            R"({"environment":{"flight":true},"bodies":[{"position":[10,10],"controlled":true}]})");
        control::Physics physics;
        control::StateBatch root{1, *scene};
        std::unique_ptr<control::PackedWave> wave;
        explicit Run(int m) : physics(scene, m >= 3 ? 4 : 1) {
          root.reset(*scene, 7);
          control::WaveConfig c;
          c.walkers = 256;
          c.frames = 1;
          c.elites = 4;
          c.recording = RecordingMode(m % 3);
          wave = std::make_unique<control::PackedWave>(physics, c, 7);
          wave->reset(root);
        }
      };
      auto run = std::make_shared<Run>(mode);
      return [run] { run->wave->step(); };
    });
    if (mode >= 3) continue;
    label = std::string("wave-") + std::to_string(mode);
    bench(label.c_str(), [mode] {
      struct Run {
        MockEnv env;
        std::unique_ptr<FractalGas> gas;
        explicit Run(int m) {
          FractalGasParams p;
          p.N = 256;
          p.n_elite = 4;
          p.recording = RecordingMode(m);
          gas = std::make_unique<FractalGas>(env, p);
          gas->reset();
        }
      };
      auto run = std::make_shared<Run>(mode);
      return [run] { run->gas->step(); };
    });
  }
  bench("graph", [] {
    struct Run {
      MockEnv env;
      std::unique_ptr<FractalTree> gas;
      Run() {
        FractalTreeParams p;
        p.start_walkers = 256;
        p.min_leafs = 64;
        p.max_walkers = 512;
        gas = std::make_unique<FractalTree>(env, p);
        gas->reset();
      }
    };
    auto run = std::make_shared<Run>();
    return [run] { run->gas->step(); };
  });
  for (int algorithm : {2, 3}) {
    bench(algorithm == 2 ? "fmc" : "jump", [algorithm] {
      struct Run {
        MockEnv env;
        FractalGas gas;
        ArcadePlanner planner;
        static FractalGasParams params() {
          FractalGasParams p;
          p.N = 256;
          p.recording = RecordingMode::Pruned;
          return p;
        }
        explicit Run(int a)
            : gas(env, params()), planner(env, gas, ArcadePlannerSettings{a, 8, true, 16}) {
          planner.reset();
        }
      };
      auto run = std::make_shared<Run>(algorithm);
      return [run] { run->planner.advance(); };
    });
  }
  bench("lab-fmc-cycle", [] {
    struct Run {
      std::shared_ptr<const control::Scene> scene = control::Scene::compile(
          R"({"size":[1000,1000],"environment":{"flight":true},"bodies":[{"position":[500,500],"controlled":true}]})");
      control::Physics physics{scene, 1};
      control::StateBatch root{1, *scene};
      control::WaveConfig config;
      std::unique_ptr<control::FmcPlanner> planner;
      Run() {
        root.reset(*scene, 7);
        config.walkers = 128;
        config.horizon = 8;
        config.frames = 1;
        config.elites = 4;
        config.recording = RecordingMode::Pruned;
        planner = std::make_unique<control::FmcPlanner>(physics, config, 7);
      }
    };
    auto run = std::make_shared<Run>();
    return [run] {
      run->planner->begin(run->root);
      while (!run->planner->advance()) {
      }
    };
  });
  for (bool force : {true, false})
    bench(force ? "euclidean" : "euclidean-no-force", [force] {
      auto run = std::make_shared<optimization::Session>(
          control::JsonReader(
              std::string(
                  R"({"algorithm":"euclidean","benchmark":"sphere","walkers":128,"dimensions":8,"periodic":true,"companion":"uniform","clone_companion":"uniform","potential_force":)") +
              (force ? "true}" : "false}"))
              .read());
      return [run] { run->step(); };
    });
}
