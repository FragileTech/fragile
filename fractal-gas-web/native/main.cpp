// Native CLI: run the fractal gas on a Super Mario Bros ROM and print
// per-iteration metrics plus throughput.
//
//   ./fg_cli --rom smb.nes --n 64 --iters 200 --seed 7 [--threads 8]
//            [--dist-coef 1.0] [--reward-coef 1.0] [--cumulative]
//            [--dt-min 1] [--dt-max 4] [--elite 0]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "fractal_gas.hpp"
#include "nes_env.hpp"

int main(int argc, char** argv) {
  std::string rom;
  fg::FractalGasParams params;
  params.N = 64;
  int iters = 200;
  int threads = 0;
  fg::ObsMode obs_mode = fg::ObsMode::kRam;
  int world = 1;
  int stage = 1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--rom") {
      rom = next();
    } else if (arg == "--n") {
      params.N = std::atoi(next());
    } else if (arg == "--iters") {
      iters = std::atoi(next());
    } else if (arg == "--seed") {
      params.seed = static_cast<uint64_t>(std::atoll(next()));
    } else if (arg == "--threads") {
      threads = std::atoi(next());
    } else if (arg == "--dist-coef") {
      params.dist_coef = static_cast<float>(std::atof(next()));
    } else if (arg == "--reward-coef") {
      params.reward_coef = static_cast<float>(std::atof(next()));
    } else if (arg == "--cumulative") {
      params.use_cumulative_reward = true;
    } else if (arg == "--dt-min") {
      params.dt_min = std::atoi(next());
    } else if (arg == "--dt-max") {
      params.dt_max = std::atoi(next());
    } else if (arg == "--elite") {
      params.n_elite = std::atoi(next());
    } else if (arg == "--world") {
      world = std::atoi(next());
    } else if (arg == "--stage") {
      stage = std::atoi(next());
    } else if (arg == "--obs") {
      const std::string mode = next();
      if (mode == "ram") {
        obs_mode = fg::ObsMode::kRam;
      } else if (mode == "rgb") {
        obs_mode = fg::ObsMode::kRgb;
      } else if (mode == "gray") {
        obs_mode = fg::ObsMode::kGray;
      } else if (mode == "coords") {
        obs_mode = fg::ObsMode::kCoords;
      } else {
        std::fprintf(stderr, "--obs must be ram|rgb|gray|coords\n");
        return 2;
      }
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  if (rom.empty()) {
    std::fprintf(stderr,
                 "Usage: fg_cli --rom smb.nes [--n 64] [--iters 200] "
                 "[--seed 7] [--threads T] [--dist-coef X] [--reward-coef X] "
                 "[--cumulative] [--dt-min 1] [--dt-max 4] [--elite K] "
                 "[--obs ram|rgb|gray|coords] [--world 1-8] [--stage 1-4]\n");
    return 2;
  }
  if (threads <= 0) {
    threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 4;
  }

  try {
    fg::NesMarioEnv env(rom, threads, obs_mode, world, stage);
    std::printf("ROM: %s | N=%d iters=%d threads=%d seed=%llu\n", rom.c_str(),
                params.N, iters, env.threads(),
                static_cast<unsigned long long>(params.seed));

    fg::FractalGas gas(env, params);
    gas.reset();

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 1; it <= iters; ++it) {
      const fg::StepInfo info = gas.step();
      if (it % 10 == 0 || it == 1 || it == iters) {
        std::printf(
            "iter %4d | reward mean %8.1f max %8.1f | vr mean %6.3f | "
            "cloned %3d | alive %3d/%d | show %d-%d\n",
            info.iteration, info.mean_reward, info.max_reward,
            info.mean_virtual_reward, info.num_cloned, info.alive_count,
            params.N, env.walker_world(info.best_walker_idx),
            env.walker_stage(info.best_walker_idx));
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const auto best = gas.get_best_walker();
    std::printf("\nBest walker %d with cumulative reward %.1f\n", best.first,
                best.second);
    std::printf("%d iterations in %.2fs = %.1f it/s (%.0f env frames/s)\n",
                iters, secs, iters / secs,
                static_cast<double>(gas.total_steps()) * 2.5 / secs);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
