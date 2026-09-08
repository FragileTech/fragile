// Diagnostic build only: counts collision work without changing packed state.
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "control/physics.hpp"
using namespace fg::control;
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: fg_control_collision_benchmark SCENE.json THREADS\n";
    return 2;
  }
  std::ifstream file(argv[1]);
  std::stringstream input;
  input << file.rdbuf();
  auto scene = Scene::compile(input.str());
  int threads = std::stoi(argv[2]);
  std::cout << "seed,threads,median_ms,frames,collisions,edge_queries,"
               "edge_candidates,separation_checks,ccd_iterations\n";
  for (uint32_t seed : {7, 11, 19}) {
    Physics physics(scene, threads);
    StateBatch root(32, *scene), out(32, *scene);
    root.reset(*scene, seed);
    std::vector<float> actions(32 * scene->channels.size());
    std::vector<int32_t> frames(32, 180);
    std::vector<StepResult> results(32);
    physics.step(root, nullptr, actions.data(), frames.data(), out, results.data());
    std::swap(root, out);
    std::fill(frames.begin(), frames.end(), 120);
    std::vector<double> times;
    CollisionWork delta{};
    uint64_t advanced = 0, collisions = 0;
    for (int repeat = 0; repeat < 4; ++repeat) {
      auto before = physics.collision_work();
      auto start = std::chrono::steady_clock::now();
      physics.step(root, nullptr, actions.data(), frames.data(), out, results.data());
      auto elapsed = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start).count();
      if (repeat) times.push_back(elapsed);
      auto after = physics.collision_work();
      delta = {after.edge_queries - before.edge_queries,
               after.edge_candidates - before.edge_candidates,
               after.separation_checks - before.separation_checks,
               after.ccd_iterations - before.ccd_iterations};
      advanced = collisions = 0;
      for (auto result : results) {
        advanced += result.frames;
        collisions += result.collisions;
      }
      if (advanced != 32 * 120) {
        std::cerr << "Fixed workload ended early; disable episode termination.\n";
        return 1;
      }
    }
    std::sort(times.begin(), times.end());
    std::cout << seed << ',' << threads << ',' << times[1] << ',' << advanced
              << ',' << collisions << ',' << delta.edge_queries << ','
              << delta.edge_candidates << ',' << delta.separation_checks << ','
              << delta.ccd_iterations << '\n';
  }
}
