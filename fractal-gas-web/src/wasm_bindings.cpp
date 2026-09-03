// Emscripten embind surface. The whole module is meant to run inside a Web
// Worker (worker.js) — fg_step() blocks while the thread pool works, so it
// must never be called on the browser main thread.
//
// Three consoles are supported end-to-end:
//   0 = NES / Super Mario Bros  (nes-py core, pthread-parallel)
//   1 = Atari 2600              (ALE core, pthread-parallel)
//   2 = Sega Genesis (Genesis Plus GX statically linked into a shim module
//       instantiated once per plain Web Worker — RetroFarmEnv dispatches
//       walker steps to N such workers in parallel via shared memory)
#ifdef __EMSCRIPTEN__

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include "atari_env.hpp"
#include "fractal_gas.hpp"
#include "nes_env.hpp"
#include "retro_farm_env.hpp"

namespace {

std::unique_ptr<fg::BatchEnv> g_env;
fg::NesMarioEnv* g_nes = nullptr;  // non-owning; set when console == 0
fg::AtariEnv* g_atari = nullptr;   // non-owning; set when console == 1
std::unique_ptr<fg::FractalGas> g_gas;
std::string g_last_error;
std::vector<uint8_t> g_frame;

struct FgParams {
  int n = 32;
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
    g_gas.reset();
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

    g_gas = std::make_unique<fg::FractalGas>(*g_env, to_gas_params(p));
    g_gas->reset();
    g_last_error.clear();
    return true;
  } catch (const std::exception& e) {
    g_last_error = e.what();
    g_gas.reset();
    g_env.reset();
    g_nes = nullptr;
    g_atari = nullptr;
    return false;
  }
}

std::string fg_last_error() { return g_last_error; }

emscripten::val fg_step() {
  if (!g_gas) return emscripten::val::null();
  const fg::StepInfo info = g_gas->step();
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
  out.set("totalSteps", static_cast<double>(g_gas->total_steps()));
  out.set("totalClones", static_cast<double>(g_gas->total_clones()));
  if (g_nes) {
    // World/level of the displayed walker (1-indexed), NES only.
    out.set("world", g_nes->walker_world(info.best_walker_idx));
    out.set("level", g_nes->walker_stage(info.best_walker_idx));
    // Per-walker swarm data for the level-map overlay: level x (world
    // pixels), on-screen y-pixel, world/stage and alive flag.
    const int32_t n = g_gas->params().N;
    const std::vector<uint8_t>& dones = g_gas->state().dones;
    emscripten::val xs = emscripten::val::array();
    emscripten::val ys = emscripten::val::array();
    emscripten::val ws = emscripten::val::array();
    emscripten::val ss = emscripten::val::array();
    emscripten::val alive = emscripten::val::array();
    for (int32_t i = 0; i < n; ++i) {
      xs.set(i, g_nes->walker_x(i));
      ys.set(i, g_nes->walker_y(i));
      ws.set(i, g_nes->walker_world(i));
      ss.set(i, g_nes->walker_stage(i));
      alive.set(i, static_cast<size_t>(i) < dones.size() && !dones[i]);
    }
    out.set("walkerX", xs);
    out.set("walkerY", ys);
    out.set("walkerWorld", ws);
    out.set("walkerStage", ss);
    out.set("walkerAlive", alive);
  } else if (g_atari && g_atari->game() == fg::AtariGame::kMontezuma) {
    // Montezuma pyramid swarm map: per-walker in-room pixel position of
    // Panama Joe (inside the HUD-cropped 160x160 room image), walkerWorld =
    // room number, walkerStage = level (0-based). Generic Atari games emit
    // nothing (no map).
    const int32_t n = g_gas->params().N;
    const std::vector<uint8_t>& dones = g_gas->state().dones;
    emscripten::val xs = emscripten::val::array();
    emscripten::val ys = emscripten::val::array();
    emscripten::val ws = emscripten::val::array();
    emscripten::val ss = emscripten::val::array();
    emscripten::val alive = emscripten::val::array();
    for (int32_t i = 0; i < n; ++i) {
      xs.set(i, g_atari->walker_x(i));
      ys.set(i, g_atari->walker_y(i));
      ws.set(i, g_atari->walker_room(i));
      ss.set(i, g_atari->walker_level(i));
      alive.set(i, static_cast<size_t>(i) < dones.size() && !dones[i]);
    }
    out.set("walkerX", xs);
    out.set("walkerY", ys);
    out.set("walkerWorld", ws);
    out.set("walkerStage", ss);
    out.set("walkerAlive", alive);
    out.set("world", g_atari->walker_room(info.best_walker_idx));
    out.set("level", g_atari->walker_level(info.best_walker_idx));
    out.set("lives", g_atari->walker_lives(info.best_walker_idx));
    out.set("inventory", g_atari->walker_inventory(info.best_walker_idx));
  } else if (auto* farm = dynamic_cast<fg::RetroFarmEnv*>(g_env.get())) {
    // Sonic fog-of-war swarm map: per-walker position/level/camera arrays
    // (same field names as the NES branch where shared; walkerWorld = zone,
    // walkerStage = act, both 0-based internal ids).
    const int32_t n = farm->walker_count();
    const std::vector<uint8_t>& dones = g_gas->state().dones;
    emscripten::val xs = emscripten::val::array();
    emscripten::val ys = emscripten::val::array();
    emscripten::val ws = emscripten::val::array();
    emscripten::val ss = emscripten::val::array();
    emscripten::val cxs = emscripten::val::array();
    emscripten::val cys = emscripten::val::array();
    emscripten::val alive = emscripten::val::array();
    for (int32_t i = 0; i < n; ++i) {
      xs.set(i, farm->walker_x(i));
      ys.set(i, farm->walker_y(i));
      ws.set(i, farm->walker_zone(i));
      ss.set(i, farm->walker_act(i));
      cxs.set(i, farm->walker_cam_x(i));
      cys.set(i, farm->walker_cam_y(i));
      alive.set(i, static_cast<size_t>(i) < dones.size() && !dones[i]);
    }
    out.set("walkerX", xs);
    out.set("walkerY", ys);
    out.set("walkerWorld", ws);
    out.set("walkerStage", ss);
    out.set("walkerCamX", cxs);
    out.set("walkerCamY", cys);
    out.set("walkerAlive", alive);
    if (n > 0) {
      const int32_t best = g_gas->get_best_walker().first;
      out.set("world", farm->walker_zone(best));
      out.set("level", farm->walker_act(best));
    }
  }
  return out;
}

/// Fog-of-war tiles: N x 40x28 RGB bytes from the last Genesis step (view
/// into wasm memory — copy on the JS side before using across steps).
emscripten::val fg_get_walker_tiles() {
  auto* farm = dynamic_cast<fg::RetroFarmEnv*>(g_env.get());
  if (!farm || farm->walker_tiles_size() == 0) return emscripten::val::null();
  return emscripten::val(emscripten::typed_memory_view(
      farm->walker_tiles_size(), farm->walker_tiles()));
}

/// RGBA bytes of the best walker's frame (view into wasm memory — copy on
/// the JS side before transferring).
emscripten::val fg_get_best_frame() {
  if (!g_gas) return emscripten::val::null();
  g_frame = g_gas->best_frame();
  if (g_frame.empty()) return emscripten::val::null();
  return emscripten::val(
      emscripten::typed_memory_view(g_frame.size(), g_frame.data()));
}

/// RGBA frame of walker i's CURRENT state (view into wasm memory — copy on
/// the JS side before transferring). Used by the Montezuma pyramid map to
/// capture a room image the first time a walker enters it.
emscripten::val fg_render_walker_frame(int i) {
  if (!g_gas || !g_env) return emscripten::val::null();
  const std::vector<std::vector<char>>& states = g_gas->state().states;
  if (i < 0 || static_cast<size_t>(i) >= states.size()) {
    return emscripten::val::null();
  }
  g_env->render_frame(states[static_cast<size_t>(i)], g_frame);
  if (g_frame.empty()) return emscripten::val::null();
  return emscripten::val(
      emscripten::typed_memory_view(g_frame.size(), g_frame.data()));
}

int fg_frame_width() { return g_env ? g_env->frame_width() : 0; }
int fg_frame_height() { return g_env ? g_env->frame_height() : 0; }
int fg_n_actions() { return g_env ? g_env->n_actions() : 0; }

/// Live-tunable parameters. Changing N/seed/console/obsMode/level requires
/// fg_init again.
void fg_set_params(const FgParams& p) {
  if (!g_gas) return;
  g_gas->set_dist_coef(p.distCoef);
  g_gas->set_reward_coef(p.rewardCoef);
  g_gas->set_use_cumulative_reward(p.useCumulativeReward);
  g_gas->set_dt_range(p.dtMin, p.dtMax);
  g_gas->set_n_elite(p.nElite);
}

void fg_reset() {
  if (g_gas) g_gas->reset();
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
      .field("farmPtr", &FgParams::farmPtr)
      .field("farmWorkers", &FgParams::farmWorkers)
      .field("farmBlobLen", &FgParams::farmBlobLen);

  emscripten::function("init", &fg_init);
  emscripten::function("lastError", &fg_last_error);
  emscripten::function("step", &fg_step);
  emscripten::function("getBestFrame", &fg_get_best_frame);
  emscripten::function("renderWalkerFrame", &fg_render_walker_frame);
  emscripten::function("getWalkerTiles", &fg_get_walker_tiles);
  emscripten::function("frameWidth", &fg_frame_width);
  emscripten::function("frameHeight", &fg_frame_height);
  emscripten::function("nActions", &fg_n_actions);
  emscripten::function("setParams", &fg_set_params);
  emscripten::function("setRewardWeights", &fg_set_reward_weights);
  emscripten::function("reset", &fg_reset);
}

#endif  // __EMSCRIPTEN__
