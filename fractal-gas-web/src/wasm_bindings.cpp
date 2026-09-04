// Emscripten embind surface. The whole module is meant to run inside a Web
// Worker (worker.js) — fg_step() blocks while the thread pool works, so it
// must never be called on the browser main thread.
//
// Three consoles are supported end-to-end:
//   0 = NES / Super Mario Bros  (nes-py core, pthread-parallel)
//   1 = Atari 2600              (ALE core, pthread-parallel; game 1 =
//       Montezuma's Revenge with dedicated logic)
//   2 = Sega Genesis (Genesis Plus GX statically linked into a shim module
//       instantiated once per plain Web Worker — RetroFarmEnv dispatches
//       walker steps to N such workers in parallel via shared memory)
// and two algorithms (src/swarm_algorithm.hpp):
//   0 = "Wave"  FractalGas   1 = "Graph" FractalTree
#ifdef __EMSCRIPTEN__

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include "atari_env.hpp"
#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include "nes_env.hpp"
#include "retro_farm_env.hpp"
#include "swarm_algorithm.hpp"

namespace {

std::unique_ptr<fg::BatchEnv> g_env;
fg::NesMarioEnv* g_nes = nullptr;  // non-owning; set when console == 0
fg::AtariEnv* g_atari = nullptr;   // non-owning; set when console == 1
std::unique_ptr<fg::SwarmAlgorithm> g_algo;
int g_algorithm = 0;
int g_max_walkers = 0;  // effective Graph cap after the memory clamp
std::string g_last_error;
std::vector<uint8_t> g_frame;

// Scratch buffers for the per-walker arrays (copied to plain JS typed
// arrays before they are returned, so JS never aliases wasm memory).
std::vector<int32_t> g_wx, g_wy, g_ww, g_ws, g_parent;
std::vector<uint8_t> g_alive, g_leaf;
std::vector<int32_t> g_tcx, g_tcy, g_tz, g_ta;
std::vector<int32_t> g_vkeys;
std::vector<float> g_vsums;

struct FgParams {
  int n = 32;         // Wave: N walkers; Graph: start walkers = min leaves
  float distCoef = 1.0f;
  float rewardCoef = 1.0f;
  bool useCumulativeReward = true;
  int dtMin = 1;
  int dtMax = 4;
  int nElite = 2;
  double seed = 0;
  int nThreads = 4;
  int obsMode = 0;   // 0=RAM, 1=RGB, 2=Gray, 3=Coords
  int world = 1;     // NES start level: world 1-8
  int stage = 1;     // NES start level: stage 1-4
  int console = 0;   // 0=NES, 1=Atari, 2=Genesis
  int game = 0;      // Genesis: 0=Airstriker, 1=Sonic; Atari: 0=generic,
                     // 1=Montezuma's Revenge (dedicated RAM logic + map)
  int algorithm = 0;   // 0=Wave (FractalGas), 1=Graph (FractalTree)
  int maxWalkers = 0;  // Graph: population cap, 0 = console default
  float eraseCoef = 0.05f;  // Graph: visit-count decay
  int aggBlock = 5;         // Graph: visit-count pooling window (px), live
  bool visitReward = true;  // visit-count term in the reward (ablation), live
  float visitCoef = 1.0f;   // exponent on the visit-count term, live
  // Genesis core-worker farm, pre-spawned by worker.js (nested workers need
  // the JS event loop, which fg_init blocks): region base pointer, worker
  // count, and the blob size the workers reported.
  double farmPtr = 0;
  int farmWorkers = 0;
  int farmBlobLen = 0;
};

fg::FractalGasParams to_gas_params(const FgParams& p) {
  fg::FractalGasParams params;
  params.N = p.n;
  params.dist_coef = p.distCoef;
  params.reward_coef = p.rewardCoef;
  params.use_cumulative_reward = p.useCumulativeReward;
  params.dt_min = p.dtMin;
  params.dt_max = p.dtMax;
  params.n_elite = p.nElite;
  params.record_frames = true;
  params.seed = static_cast<uint64_t>(p.seed);
  params.count_visits = true;  // effective only with a visit key (Coords)
  params.visit_reward = p.visitReward;
  params.visit_coef = p.visitCoef;
  params.erase_coef = p.eraseCoef;
  params.agg_block_size = p.aggBlock < 1 ? 1 : p.aggBlock;
  return params;
}

/// Graph population cap: the requested / console-default max walkers,
/// clamped to what the fixed 512 MB wasm heap can hold (each walker keeps a
/// full emulator state blob plus its observation row).
int clamp_max_walkers(const FgParams& p) {
  const int defaults[3] = {4000, 20000, 150};  // NES, Atari, Genesis
  const int console = p.console < 0 || p.console > 2 ? 0 : p.console;
  int requested = p.maxWalkers > 0 ? p.maxWalkers : defaults[console];
  std::vector<char> state;
  std::vector<float> obs;
  g_env->reset(state, obs);  // cached by every env, so this is free later
  const double per_walker =
      static_cast<double>(state.size()) + static_cast<double>(obs.size()) * 4.0 + 256.0;
  const double heap = 512.0 * 1024 * 1024;
  double reserve = 160.0 * 1024 * 1024;  // module, emulator instances, frames
  if (console == 2) reserve += static_cast<double>(p.farmWorkers) * 3.7 * 1024 * 1024;
  const double budget = heap - reserve;
  const int cap = static_cast<int>(std::max(1.0, budget / per_walker));
  requested = std::min(requested, cap);
  return std::max(requested, std::max(p.n, 1));
}

fg::FractalTreeParams to_tree_params(const FgParams& p) {
  fg::FractalTreeParams params;
  params.start_walkers = std::max(p.n, 1);
  params.min_leafs = std::max(p.n, 1);
  params.max_walkers = clamp_max_walkers(p);
  params.dist_coef = p.distCoef;
  params.reward_coef = p.rewardCoef;
  params.dt_min = p.dtMin;
  params.dt_max = p.dtMax;
  params.count_visits = true;  // effective only with a visit key (Coords)
  params.erase_coef = p.eraseCoef;
  params.agg_block_size = p.aggBlock < 1 ? 1 : p.aggBlock;
  params.visit_reward = p.visitReward;
  params.visit_coef = p.visitCoef;
  params.record_frames = true;
  params.seed = static_cast<uint64_t>(p.seed);
  return params;
}

void write_bytes(const char* path, emscripten::val data) {
  const std::vector<uint8_t> bytes =
      emscripten::convertJSArrayToNumberVector<uint8_t>(data);
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
}

/// rom: Uint8Array of the ROM for the selected console.
/// aux: Uint8Array of the gpgx.so side module (Genesis only; else empty).
bool fg_init(emscripten::val rom, emscripten::val aux, const FgParams& p) {
  try {
    g_algo.reset();
    g_env.reset();
    g_nes = nullptr;
    g_atari = nullptr;

    const int threads = p.nThreads < 1 ? 1 : (p.nThreads > 8 ? 8 : p.nThreads);
    const int mode_int = p.obsMode < 0 || p.obsMode > 3 ? 0 : p.obsMode;

    switch (p.console) {
      case 1: {  // Atari 2600 via ALE
        write_bytes("/rom.bin", rom);
        auto atari = std::make_unique<fg::AtariEnv>(
            "/rom.bin", threads, static_cast<fg::AtariObsMode>(mode_int),
            p.game == 1 ? fg::AtariGame::kMontezuma : fg::AtariGame::kGeneric);
        g_atari = atari.get();
        g_env = std::move(atari);
        break;
      }
      case 2: {  // Sega Genesis via per-worker shim modules (parallel)
        (void)aux;
        (void)rom;  // the ROM went to the core workers, not this module
        g_env = std::make_unique<fg::RetroFarmEnv>(
            static_cast<uintptr_t>(p.farmPtr), p.farmWorkers,
            static_cast<size_t>(p.farmBlobLen), mode_int,
            p.game == 1 ? fg::RetroGame::kSonic
                        : fg::RetroGame::kAirstriker);
        break;
      }
      default: {  // NES / Super Mario Bros
        write_bytes("/rom.nes", rom);
        auto nes = std::make_unique<fg::NesMarioEnv>(
            "/rom.nes", threads, static_cast<fg::ObsMode>(mode_int), p.world,
            p.stage);
        g_nes = nes.get();
        g_env = std::move(nes);
        break;
      }
    }

    g_algorithm = p.algorithm == 1 ? 1 : 0;
    if (g_algorithm == 1) {
      const fg::FractalTreeParams tp = to_tree_params(p);
      g_max_walkers = tp.max_walkers;
      g_algo = std::make_unique<fg::FractalTree>(*g_env, tp);
    } else {
      g_max_walkers = p.n;
      g_algo = std::make_unique<fg::FractalGas>(*g_env, to_gas_params(p));
    }
    g_algo->reset();
    g_last_error.clear();
    return true;
  } catch (const std::exception& e) {
    g_last_error = e.what();
    g_algo.reset();
    g_env.reset();
    g_nes = nullptr;
    g_atari = nullptr;
    return false;
  }
}

std::string fg_last_error() { return g_last_error; }
int fg_algorithm() { return g_algorithm; }
int fg_max_walkers() { return g_max_walkers; }

template <typename T>
emscripten::val copy_array(const std::vector<T>& v) {
  // A fresh, non-shared typed array (slice of a view into wasm memory).
  return emscripten::val(emscripten::typed_memory_view(v.size(), v.data()))
      .template call<emscripten::val>("slice");
}

emscripten::val fg_step() {
  if (!g_algo) return emscripten::val::null();
  const fg::StepInfo info = g_algo->step();
  emscripten::val out = emscripten::val::object();
  out.set("iteration", info.iteration);
  out.set("numCloned", info.num_cloned);
  out.set("numRevived", info.num_revived);
  out.set("aliveCount", info.alive_count);
  out.set("meanReward", info.mean_reward);
  out.set("maxReward", info.max_reward);
  out.set("minReward", info.min_reward);
  out.set("meanVirtualReward", info.mean_virtual_reward);
  out.set("maxVirtualReward", info.max_virtual_reward);
  out.set("minVirtualReward", info.min_virtual_reward);
  out.set("meanDt", info.mean_dt);
  out.set("bestWalkerIdx", info.best_walker_idx);
  out.set("totalSteps", static_cast<double>(g_algo->total_steps()));
  out.set("totalClones", static_cast<double>(g_algo->total_clones()));
  out.set("totalFrames", static_cast<double>(g_algo->total_frames()));
  out.set("walkerCount", info.n_walkers);
  out.set("nLeaves", info.n_leaves);
  out.set("numStepped", info.num_stepped);
  out.set("algorithm", g_algorithm);

  const int32_t n = g_algo->n_walkers();
  if (g_algo->has_walker_info() && n > 0) {
    // Per-walker swarm data for the map overlays, valid by WALKER index for
    // both algorithms: position, level ids, alive flag, and the tree
    // structure (parent index, leaf flag; the wave reports parent = self).
    const auto un = static_cast<size_t>(n);
    g_wx.resize(un); g_wy.resize(un); g_ww.resize(un); g_ws.resize(un);
    g_parent.resize(un); g_alive.resize(un); g_leaf.resize(un);
    for (int32_t i = 0; i < n; ++i) {
      const fg::WalkerInfo& wi = g_algo->walker_info(i);
      const auto ui = static_cast<size_t>(i);
      g_wx[ui] = wi.x;
      g_wy[ui] = wi.y;
      g_ww[ui] = wi.world;
      g_ws[ui] = wi.stage;
      g_parent[ui] = g_algo->walker_parent(i);
      g_alive[ui] = g_algo->walker_alive(i) ? 1 : 0;
      g_leaf[ui] = g_algo->walker_is_leaf(i) ? 1 : 0;
    }
    out.set("walkerX", copy_array(g_wx));
    out.set("walkerY", copy_array(g_wy));
    out.set("walkerWorld", copy_array(g_ww));
    out.set("walkerStage", copy_array(g_ws));
    out.set("walkerParent", copy_array(g_parent));
    out.set("walkerAlive", copy_array(g_alive));
    out.set("walkerLeaf", copy_array(g_leaf));
    if (dynamic_cast<fg::RetroFarmEnv*>(g_env.get()) != nullptr) {
      // Per-walker camera (Sonic), kept for callers that pair it by walker.
      g_tcx.resize(un); g_tcy.resize(un);
      for (int32_t i = 0; i < n; ++i) {
        const fg::WalkerInfo& wi = g_algo->walker_info(i);
        g_tcx[static_cast<size_t>(i)] = wi.cam_x;
        g_tcy[static_cast<size_t>(i)] = wi.cam_y;
      }
      out.set("walkerCamX", copy_array(g_tcx));
      out.set("walkerCamY", copy_array(g_tcy));
    }
    const fg::WalkerInfo& best = g_algo->walker_info(info.best_walker_idx);
    out.set("world", best.world);
    out.set("level", best.stage);
    if (g_atari) {
      out.set("lives", best.lives);
      out.set("inventory", best.inventory);
    }
  }
  if (auto* farm = dynamic_cast<fg::RetroFarmEnv*>(g_env.get())) {
    // Sonic fog-of-war tiles are indexed by BATCH position of the last
    // step (a subset of the walkers in Graph mode): ship the camera and
    // level of each tile alongside so the UI can pair them.
    const int32_t t = farm->walker_count();
    const auto ut = static_cast<size_t>(t);
    g_tcx.resize(ut); g_tcy.resize(ut); g_tz.resize(ut); g_ta.resize(ut);
    for (int32_t j = 0; j < t; ++j) {
      const auto uj = static_cast<size_t>(j);
      g_tcx[uj] = farm->walker_cam_x(j);
      g_tcy[uj] = farm->walker_cam_y(j);
      g_tz[uj] = farm->walker_zone(j);
      g_ta[uj] = farm->walker_act(j);
    }
    out.set("tileCount", t);
    out.set("tileCamX", copy_array(g_tcx));
    out.set("tileCamY", copy_array(g_tcy));
    out.set("tileZone", copy_array(g_tz));
    out.set("tileAct", copy_array(g_ta));
  }
  return out;
}

/// Fog-of-war tiles: tileCount x 40x28 RGB bytes from the last Genesis step
/// (view into wasm memory — copy on the JS side before using across steps).
emscripten::val fg_get_walker_tiles() {
  auto* farm = dynamic_cast<fg::RetroFarmEnv*>(g_env.get());
  if (!farm || farm->walker_tiles_size() == 0) return emscripten::val::null();
  return emscripten::val(emscripten::typed_memory_view(
      farm->walker_tiles_size(), farm->walker_tiles()));
}

/// RGBA bytes of the showcased walker's frame (view into wasm memory — copy
/// on the JS side before transferring).
emscripten::val fg_get_best_frame() {
  if (!g_algo) return emscripten::val::null();
  g_frame = g_algo->best_frame();
  if (g_frame.empty()) return emscripten::val::null();
  return emscripten::val(
      emscripten::typed_memory_view(g_frame.size(), g_frame.data()));
}

/// RGBA frame of walker i's CURRENT state (view into wasm memory — copy on
/// the JS side before transferring). Used by the Montezuma pyramid map to
/// capture a room image the first time a walker enters it. Null for a tree
/// slot that has no state yet.
emscripten::val fg_render_walker_frame(int i) {
  if (!g_algo || !g_env) return emscripten::val::null();
  if (i < 0 || i >= g_algo->n_walkers()) return emscripten::val::null();
  const std::vector<char>& state = g_algo->walker_state(i);
  if (state.empty()) return emscripten::val::null();
  g_env->render_frame(state, g_frame);
  if (g_frame.empty()) return emscripten::val::null();
  return emscripten::val(
      emscripten::typed_memory_view(g_frame.size(), g_frame.data()));
}

/// Visit counting active (Coords on a game with a map), either algorithm.
bool fg_counting_visits() {
  return g_algo != nullptr && g_algo->counting_visits();
}

/// The Graph's visit-count grid for the map heatmap: every nonzero 5x5
/// block as {count, keys: Int32Array [plane, bx, by] * count, sums:
/// Float32Array, blockSize}. Null when visits are not counted.
emscripten::val fg_get_visit_blocks() {
  if (!fg_counting_visits() || g_algo->visit_grid() == nullptr) {
    return emscripten::val::null();
  }
  const fg::VisitGrid& grid = *g_algo->visit_grid();
  grid.export_blocks(g_vkeys, g_vsums);
  emscripten::val out = emscripten::val::object();
  out.set("count", static_cast<int>(g_vsums.size()));
  out.set("blockSize", grid.block_size());
  out.set("keys", copy_array(g_vkeys));
  out.set("sums", copy_array(g_vsums));
  return out;
}

int fg_frame_width() { return g_env ? g_env->frame_width() : 0; }
int fg_frame_height() { return g_env ? g_env->frame_height() : 0; }
int fg_n_actions() { return g_env ? g_env->n_actions() : 0; }

/// Live-tunable parameters. Changing N/seed/console/obsMode/level/algorithm
/// requires fg_init again.
void fg_set_params(const FgParams& p) {
  if (!g_algo) return;
  g_algo->set_dist_coef(p.distCoef);
  g_algo->set_reward_coef(p.rewardCoef);
  g_algo->set_use_cumulative_reward(p.useCumulativeReward);
  g_algo->set_dt_range(p.dtMin, p.dtMax);
  g_algo->set_n_elite(p.nElite);
  g_algo->set_erase_coef(p.eraseCoef);
  g_algo->set_agg_block_size(p.aggBlock < 1 ? 1 : p.aggBlock);
  g_algo->set_visit_reward(p.visitReward);
  g_algo->set_visit_coef(p.visitCoef);
}

void fg_reset() {
  if (g_algo) g_algo->reset();
}

/// Live-tunable reward term weights, as a JS array of numbers in the
/// console's field order: Mario [x, time, death, clip, flag, area]; Sonic
/// [dx, rings, score, cell, life, boss, act]; Montezuma [score, room].
/// Ignored for generic Atari games (raw score).
void fg_set_reward_weights(emscripten::val weights) {
  if (!g_env) return;
  const int len = weights["length"].as<int>();
  auto at = [&](int i, float fallback) {
    return i < len ? weights[i].as<float>() : fallback;
  };
  if (g_nes) {
    fg::MarioRewardWeights w;
    w.x = at(0, w.x);
    w.time = at(1, w.time);
    w.death = at(2, w.death);
    w.clip = at(3, w.clip);
    w.flag = at(4, w.flag);
    w.area = at(5, w.area);
    g_nes->set_reward_weights(w);
  } else if (auto* farm = dynamic_cast<fg::RetroFarmEnv*>(g_env.get())) {
    fg::SonicRewardWeights w;
    w.dx = at(0, w.dx);
    w.rings = at(1, w.rings);
    w.score = at(2, w.score);
    w.cell = at(3, w.cell);
    w.life = at(4, w.life);
    w.boss = at(5, w.boss);
    w.act = at(6, w.act);
    farm->set_reward_weights(w);
  } else if (g_atari && g_atari->game() == fg::AtariGame::kMontezuma) {
    fg::MontezumaRewardWeights w;
    w.score = at(0, w.score);
    w.room = at(1, w.room);
    g_atari->set_reward_weights(w);
  }
}

}  // namespace

EMSCRIPTEN_BINDINGS(fractal_gas) {
  emscripten::value_object<FgParams>("FgParams")
      .field("n", &FgParams::n)
      .field("distCoef", &FgParams::distCoef)
      .field("rewardCoef", &FgParams::rewardCoef)
      .field("useCumulativeReward", &FgParams::useCumulativeReward)
      .field("dtMin", &FgParams::dtMin)
      .field("dtMax", &FgParams::dtMax)
      .field("nElite", &FgParams::nElite)
      .field("seed", &FgParams::seed)
      .field("nThreads", &FgParams::nThreads)
      .field("obsMode", &FgParams::obsMode)
      .field("world", &FgParams::world)
      .field("stage", &FgParams::stage)
      .field("console", &FgParams::console)
      .field("game", &FgParams::game)
      .field("algorithm", &FgParams::algorithm)
      .field("maxWalkers", &FgParams::maxWalkers)
      .field("eraseCoef", &FgParams::eraseCoef)
      .field("aggBlock", &FgParams::aggBlock)
      .field("visitReward", &FgParams::visitReward)
      .field("visitCoef", &FgParams::visitCoef)
      .field("farmPtr", &FgParams::farmPtr)
      .field("farmWorkers", &FgParams::farmWorkers)
      .field("farmBlobLen", &FgParams::farmBlobLen);

  emscripten::function("init", &fg_init);
  emscripten::function("lastError", &fg_last_error);
  emscripten::function("algorithm", &fg_algorithm);
  emscripten::function("maxWalkers", &fg_max_walkers);
  emscripten::function("step", &fg_step);
  emscripten::function("getBestFrame", &fg_get_best_frame);
  emscripten::function("renderWalkerFrame", &fg_render_walker_frame);
  emscripten::function("getWalkerTiles", &fg_get_walker_tiles);
  emscripten::function("countingVisits", &fg_counting_visits);
  emscripten::function("getVisitBlocks", &fg_get_visit_blocks);
  emscripten::function("frameWidth", &fg_frame_width);
  emscripten::function("frameHeight", &fg_frame_height);
  emscripten::function("nActions", &fg_n_actions);
  emscripten::function("setParams", &fg_set_params);
  emscripten::function("setRewardWeights", &fg_set_reward_weights);
  emscripten::function("reset", &fg_reset);
}

#endif  // __EMSCRIPTEN__
