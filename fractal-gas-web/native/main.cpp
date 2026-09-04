// Native CLI: run the fractal gas ("wave") or the fractal tree ("graph") on
// a Super Mario Bros ROM and print per-iteration metrics plus throughput.
//
//   ./fg_cli --rom smb.nes --n 64 --iters 200 --seed 7 [--threads 8]
//            [--dist-coef 1.0] [--reward-coef 1.0] [--cumulative]
//            [--dt-min 1] [--dt-max 4] [--elite 0]
//            [--algo wave|graph] [--max-walkers 20000] [--min-leafs K]
//            [--erase-coef 0.05] [--no-visits]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <memory>

#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include "nes_env.hpp"
#include "swarm_algorithm.hpp"

int main(int argc, char** argv) {
  std::string rom;
  fg::FractalGasParams params;
  params.N = 64;
  int iters = 200;
  int threads = 0;
  fg::ObsMode obs_mode = fg::ObsMode::kRam;
  int world = 1;
  int stage = 1;
  bool graph = false;
  fg::FractalTreeParams tree_params;
  tree_params.max_walkers = 20000;
  int min_leafs = -1;

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
    } else if (arg == "--algo") {
      const std::string algo = next();
      if (algo == "wave") {
        graph = false;
      } else if (algo == "graph") {
        graph = true;
      } else {
        std::fprintf(stderr, "--algo must be wave|graph\n");
        return 2;
      }
    } else if (arg == "--max-walkers") {
      tree_params.max_walkers = std::atoi(next());
    } else if (arg == "--min-leafs") {
      min_leafs = std::atoi(next());
    } else if (arg == "--erase-coef") {
      tree_params.erase_coef = static_cast<float>(std::atof(next()));
    } else if (arg == "--no-visits") {
      tree_params.count_visits = false;
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
                 "[--obs ram|rgb|gray|coords] [--world 1-8] [--stage 1-4] "
                 "[--algo wave|graph] [--max-walkers K] [--min-leafs K] "
                 "[--erase-coef X] [--no-visits]\n");
    return 2;
  }
  if (threads <= 0) {
    threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 4;
  }

  try {
    fg::NesMarioEnv env(rom, threads, obs_mode, world, stage);
    std::printf("ROM: %s | algo=%s N=%d iters=%d threads=%d seed=%llu\n",
                rom.c_str(), graph ? "graph" : "wave", params.N, iters,
                env.threads(), static_cast<unsigned long long>(params.seed));

    std::unique_ptr<fg::SwarmAlgorithm> algo;
    if (graph) {
      tree_params.start_walkers = params.N;
      tree_params.min_leafs = min_leafs > 0 ? min_leafs : params.N;
      tree_params.dist_coef = params.dist_coef;
      tree_params.reward_coef = params.reward_coef;
      tree_params.dt_min = params.dt_min;
      tree_params.dt_max = params.dt_max;
      tree_params.seed = params.seed;
      algo = std::make_unique<fg::FractalTree>(env, tree_params);
    } else {
      algo = std::make_unique<fg::FractalGas>(env, params);
    }
    algo->reset();

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 1; it <= iters; ++it) {
      const fg::StepInfo info = algo->step();
      if (it % 10 == 0 || it == 1 || it == iters) {
        const fg::WalkerInfo& show = algo->walker_info(info.best_walker_idx);
        std::printf(
            "iter %4d | reward mean %8.1f max %8.1f | vr mean %6.3f | "
            "stepped %4d | alive %4d/%d | leaves %4d | show %d-%d\n",
            info.iteration, info.mean_reward, info.max_reward,
            info.mean_virtual_reward, info.num_stepped, info.alive_count,
            info.n_walkers, info.n_leaves, show.world, show.stage);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const auto best = algo->get_best_walker();
    std::printf("\nBest walker %d with cumulative reward %.1f (%d walkers)\n",
                best.first, best.second, algo->n_walkers());
    std::printf("%d iterations in %.2fs = %.1f it/s (%lld env frames, %.0f frames/s)\n",
                iters, secs, iters / secs,
                static_cast<long long>(algo->total_frames()),
                static_cast<double>(algo->total_frames()) / secs);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
