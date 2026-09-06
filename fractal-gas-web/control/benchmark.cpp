// Repeatable batch benchmark: immutable scene, stable allocation, zero
// rendering.
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>

#include "control/physics.hpp"
using namespace fg::control;
using Clock = std::chrono::steady_clock;
template <class F>
double measure(F fn, int repeats) {
  auto start = Clock::now();
  for (int i = 0; i < repeats; ++i) fn();
  return std::chrono::duration<double>(Clock::now() - start).count() / repeats;
}
int main() {
  std::cout << "worlds,bodies,threads,payload_bytes,stride_bytes,step_worlds_"
               "per_second,step_body_frames_per_second,gather_GB_per_second,"
               "snapshot_MB_per_second\n";
  for (int bodies : {2, 64, 256}) {
    std::ostringstream json;
    json << "{\"size\":[128,128],\"bodies\":[";
    for (int i = 0; i < bodies; ++i) {
      if (i) json << ',';
      json << "{\"position\":[" << 4 + 7 * (i % 16) << ',' << 4 + 7 * (i / 16)
           << "],\"controlled\":true}";
    }
    json << "]}";
    auto scene = Scene::compile(json.str());
    for (int worlds : {64, 256, 1024})
      for (int threads : {1, 2, 4, 8}) {
        Physics physics(scene, threads);
        StateBatch a(worlds, *scene), b(worlds, *scene);
        a.reset(*scene, 7);
        std::vector<float> actions(size_t(worlds) * scene->channels.size(),
                                   .1f);
        std::vector<int32_t> frames(worlds, 1), sources(worlds);
        std::vector<StepResult> results(worlds);
        std::iota(sources.begin(), sources.end(), 0);
        std::rotate(sources.begin(), sources.begin() + worlds / 3,
                    sources.end());
        std::vector<uint8_t> binary(a.serialized_size());
        physics.step(a, sources.data(), actions.data(), frames.data(), b,
                     results.data());
        double step = measure(
            [&] {
              physics.step(a, sources.data(), actions.data(), frames.data(), b,
                           results.data());
            },
            5);
        double gather =
            measure([&] { b.gather(a, sources.data(), worlds); }, 100);
        double snapshot =
            measure([&] { a.serialize(binary.data(), binary.size()); }, 5);
        std::cout << worlds << ',' << bodies << ',' << threads << ','
                  << scene->layout.words * 4 << ',' << scene->layout.stride * 4
                  << ',' << worlds / step << ','
                  << double(worlds) * bodies / step << ','
                  << double(worlds) * scene->layout.words * 4 / gather / 1e9
                  << ',' << binary.size() / snapshot / 1e6 << '\n';
      }
  }
}
