// Sega Genesis (Genesis Plus GX via per-instance dlopen'd core copies)
// tests. All assets ship with the repo: the core .so is built by
// cmake/FgRetro.cmake and the Airstriker ROM + Level1 savestate live in the
// stable-retro submodule, so no environment variable is needed. Overrides:
//   FG_RETRO_CORE=/path/to/genesis_plus_gx_libretro.so
//   FG_RETRO_ROM=/path/to/rom.md
//   FG_RETRO_STATE=/path/to/Level1.state
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fractal_gas.hpp"
#include "retro_core.hpp"
#include "retro_env.hpp"
#include "test_framework.hpp"

using namespace fg;

namespace {

std::string asset(const char* env_var, const char* default_path) {
  const char* v = std::getenv(env_var);
  return v && *v ? v : default_path;
}

std::string core_so() {
  return asset("FG_RETRO_CORE", FG_RETRO_CORE_SO_DEFAULT);
}
std::string rom() { return asset("FG_RETRO_ROM", FG_RETRO_ROM_DEFAULT); }
std::string level1() {
  return asset("FG_RETRO_STATE", FG_RETRO_STATE_DEFAULT);
}

bool file_exists(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f) std::fclose(f);
  return f != nullptr;
}

bool skip_if_missing_assets(const char* test_name) {
  for (const std::string& p : {core_so(), rom(), level1()}) {
    if (!file_exists(p)) {
      std::printf("  SKIP %s (missing %s)\n", test_name, p.c_str());
      return true;
    }
  }
  return false;
}

RetroEnv make_env(int threads, ObsMode mode = ObsMode::kRam) {
  return RetroEnv(core_so(), rom(), level1(), threads, mode);
}

}  // namespace

// THE critical isolation test: a libretro core is all globals, so two
// instances only work if each dlopen'd copy has its own. Step instance A
// while B sits untouched; B's full serialized state must be bit-identical
// before and after.
TEST_CASE(retro_two_cores_step_independently) {
  if (skip_if_missing_assets("retro_two_cores_step_independently")) return;
  RetroCore a(core_so());
  RetroCore b(core_so());
  a.load_rom(rom());
  b.load_rom(rom());

  std::vector<char> b_before(b.serialize_size());
  CHECK(b.serialize(b_before.data(), b_before.size()));

  for (int f = 0; f < 10; ++f) a.run_frame(kPadB | kPadRight);

  std::vector<char> a_after(a.serialize_size());
  CHECK(a.serialize(a_after.data(), a_after.size()));
  std::vector<char> b_after(b.serialize_size());
  CHECK(b.serialize(b_after.data(), b_after.size()));

  CHECK(b_before == b_after);   // B never moved
  CHECK(a_after != b_after);    // A did
  // And B still steps normally afterwards (its globals are intact).
  b.run_frame(0);
  CHECK(b.frame_width() > 0);
}

TEST_CASE(retro_reset_reaches_playable_state) {
  if (skip_if_missing_assets("retro_reset_reaches_playable_state")) return;
  RetroEnv env = make_env(2, ObsMode::kCoords);
  CHECK(env.n_actions() == 8);
  CHECK(env.obs_dim() == kRetroCoordsDim);
  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  CHECK(obs.size() == static_cast<size_t>(kRetroCoordsDim));
  CHECK(state.size() > 0xF0000);  // ~1MB savestate (0xfd000) + carry
  // Level1.state starts a fresh game: score 0, 3 lives, not game over.
  CHECK(obs[0] == 0.0f);   // score
  CHECK(obs[1] == 3.0f);   // lives
  CHECK(obs[2] != 1.0f);   // gameover flag not set
  // reset() is cached: a second reset returns the identical state.
  std::vector<char> state2;
  std::vector<float> obs2;
  env.reset(state2, obs2);
  CHECK(state == state2);
  CHECK(obs == obs2);
}

// Same-instance determinism is exact: stepping the same blob with the same
// action twice through the same core gives bit-identical blobs.
TEST_CASE(retro_blob_roundtrip_deterministic_same_instance) {
  if (skip_if_missing_assets("retro_blob_roundtrip_deterministic_same_instance"))
    return;
  RetroEnv env = make_env(1, ObsMode::kCoords);
  std::vector<char> init;
  std::vector<float> obs0;
  env.reset(init, obs0);

  const size_t d = static_cast<size_t>(env.obs_dim());
  auto step_once = [&](const std::vector<char>& s) {
    std::vector<std::vector<char>> states = {s};
    std::vector<std::vector<char>> out(1);
    std::vector<float> obs(d), rewards(1);
    std::vector<uint8_t> dones(1), truncated(1);
    env.step_batch(states, {7}, {5}, out, obs, rewards, dones, truncated);
    return std::make_pair(out[0], obs);
  };
  auto r1 = step_once(init);
  auto r2 = step_once(init);
  CHECK(r1.first == r2.first);
  CHECK(r1.second == r2.second);
}

// Across instances/slots the emulated machine evolves identically (work RAM,
// observations, rewards). The serialized blobs may differ in a few bytes
// that Genesis Plus GX writes to savestates but never reads back (VDP
// FIFO / YM2612 residue), so cross-instance equality is checked on behavior
// and on the walkers' further evolution, not on raw blob bytes.
TEST_CASE(retro_blob_roundtrip_deterministic_across_slots) {
  if (skip_if_missing_assets("retro_blob_roundtrip_deterministic_across_slots"))
    return;
  RetroEnv env = make_env(2, ObsMode::kRam);
  std::vector<char> init;
  std::vector<float> obs0;
  env.reset(init, obs0);

  const size_t d = static_cast<size_t>(env.obs_dim());
  // Two identical walkers; static partitioning maps walker 0 -> slot 0 and
  // walker 1 -> slot 1, i.e. two different core instances.
  std::vector<std::vector<char>> states = {init, init};
  std::vector<std::vector<char>> new_states(2);
  std::vector<float> obs(2 * d), rewards(2);
  std::vector<uint8_t> dones(2), truncated(2);
  const std::vector<int32_t> actions = {6, 6};
  const std::vector<int32_t> dt = {4, 4};
  for (int step = 0; step < 5; ++step) {
    env.step_batch(states, actions, dt, new_states, obs, rewards, dones,
                   truncated);
    states[0] = new_states[0];
    states[1] = new_states[1];
    CHECK(rewards[0] == rewards[1]);
    CHECK(dones[0] == dones[1]);
    for (size_t k = 0; k < d; ++k) {
      if (obs[k] != obs[d + k]) {
        CHECK(obs[k] == obs[d + k]);
        break;  // one failure line is enough
      }
    }
  }
  // Cross-check: swap the blobs between the slots; evolution must not care
  // which instance produced or consumes a blob.
  std::vector<std::vector<char>> swapped = {states[1], states[0]};
  env.step_batch(swapped, actions, dt, new_states, obs, rewards, dones,
                 truncated);
  CHECK(rewards[0] == rewards[1]);
  for (size_t k = 0; k < d; ++k) {
    if (obs[k] != obs[d + k]) {
      CHECK(obs[k] == obs[d + k]);
      break;
    }
  }
}

TEST_CASE(retro_step_batch_8_walkers) {
  if (skip_if_missing_assets("retro_step_batch_8_walkers")) return;
  RetroEnv env = make_env(4, ObsMode::kCoords);
  std::vector<char> init;
  std::vector<float> obs0;
  env.reset(init, obs0);

  const int32_t n = 8;
  const size_t d = static_cast<size_t>(env.obs_dim());
  std::vector<std::vector<char>> states(n, init);
  std::vector<std::vector<char>> new_states(n);
  std::vector<float> obs(n * d), rewards(n);
  std::vector<uint8_t> dones(n), truncated(n);
  std::vector<int32_t> actions, dt;
  for (int32_t i = 0; i < n; ++i) {
    actions.push_back(i % env.n_actions());
    dt.push_back(1 + (i % 4));
  }
  for (int step = 0; step < 3; ++step) {
    env.step_batch(states, actions, dt, new_states, obs, rewards, dones,
                   truncated);
    for (int32_t i = 0; i < n; ++i) {
      states[i] = new_states[i];
      CHECK(new_states[i].size() == init.size());
      CHECK(std::isfinite(rewards[i]));
      CHECK(truncated[i] == 0);
      CHECK(dones[i] == 0 || dones[i] == 1);
      for (size_t k = 0; k < d; ++k) CHECK(std::isfinite(obs[i * d + k]));
    }
    CHECK(env.has_display_score());
    for (int32_t i = 0; i < n; ++i) CHECK(env.display_score(i) >= 0.0f);
  }
}

TEST_CASE(retro_obs_modes) {
  if (skip_if_missing_assets("retro_obs_modes")) return;
  constexpr size_t kPixels =
      static_cast<size_t>(kRetroFrameWidth) * kRetroFrameHeight;

  std::vector<std::vector<float>> obs_by_mode;
  for (int m = 0; m < 4; ++m) {
    RetroEnv env = make_env(1, static_cast<ObsMode>(m));
    std::vector<char> state;
    std::vector<float> obs;
    env.reset(state, obs);
    CHECK(static_cast<int32_t>(obs.size()) == env.obs_dim());
    obs_by_mode.push_back(std::move(obs));
  }

  const auto& ram = obs_by_mode[0];
  const auto& rgb = obs_by_mode[1];
  const auto& gray = obs_by_mode[2];
  const auto& coords = obs_by_mode[3];

  CHECK(ram.size() == 0x10000);
  CHECK(rgb.size() == kPixels * 3);
  CHECK(gray.size() == kPixels);
  CHECK(coords.size() == static_cast<size_t>(kRetroCoordsDim));

  // The reset is deterministic, so every mode saw the same game state.
  // Gray must be the luminance of rgb, and the screen must not be black.
  double rgb_sum = 0.0;
  for (size_t p = 0; p < gray.size(); ++p) {
    const float lum = 0.299f * rgb[p * 3] + 0.587f * rgb[p * 3 + 1] +
                      0.114f * rgb[p * 3 + 2];
    if (std::fabs(gray[p] - lum) > 1e-3f) {
      CHECK_CLOSE(gray[p], lum, 1e-5);
      break;
    }
    rgb_sum += rgb[p * 3] + rgb[p * 3 + 1] + rgb[p * 3 + 2];
  }
  CHECK(rgb_sum > 0.0);

  // Coords must match the same variables recomputed from the RAM
  // observation. The RAM obs holds the raw core buffer (native 16-bit word
  // order), so a big-endian u16 at even 68k offset o is ram[o] | ram[o+1]<<8
  // (see retro_env.cpp).
  auto be_u16 = [&](size_t off) {
    return ram[off] + 256.0f * ram[off + 1];
  };
  const float score =
      be_u16(0x24E) * 65536.0f + be_u16(0x250);          // >u4 @ 0xFF024E
  CHECK(coords[0] == score);
  CHECK(coords[1] == be_u16(0x25A));                     // lives  @ 0xFF025A
  CHECK(coords[2] == be_u16(0x266));                     // gameover @ 0xFF0266
  CHECK(coords[1] == 3.0f);
}

TEST_CASE(retro_render_frame_produces_rgba) {
  if (skip_if_missing_assets("retro_render_frame_produces_rgba")) return;
  RetroEnv env = make_env(1);
  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  CHECK(env.frame_width() == kRetroFrameWidth);
  CHECK(env.frame_height() == kRetroFrameHeight);
  std::vector<uint8_t> rgba;
  env.render_frame(state, rgba);
  CHECK(static_cast<int32_t>(rgba.size()) ==
        env.frame_width() * env.frame_height() * 4);
  int64_t sum = 0;
  bool alpha_ok = true;
  for (size_t i = 0; i < rgba.size(); i += 4) {
    sum += rgba[i] + rgba[i + 1] + rgba[i + 2];
    alpha_ok = alpha_ok && rgba[i + 3] == 0xFF;
  }
  CHECK(sum > 0);
  CHECK(alpha_ok);
}

TEST_CASE(retro_gas_smoke) {
  if (skip_if_missing_assets("retro_gas_smoke")) return;
  RetroEnv env = make_env(4, ObsMode::kRam);
  FractalGasParams params;
  params.N = 8;
  params.seed = 13;
  params.use_cumulative_reward = true;
  params.record_frames = true;

  FractalGas gas(env, params);
  gas.reset();
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < 15; ++it) {
    const StepInfo info = gas.step();
    CHECK(std::isfinite(info.mean_reward));
    CHECK(std::isfinite(info.mean_virtual_reward));
    CHECK(info.alive_count >= 0 && info.alive_count <= params.N);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  CHECK(gas.get_best_walker().second > -1e30f);
  CHECK(!gas.best_frame().empty());
  // total_steps counts walker steps; each runs dt in [dt_min, dt_max] frames.
  std::printf(
      "  retro_gas_smoke: %lld walker steps in %.2fs (%.0f steps/s, dt %d-%d)\n",
      static_cast<long long>(gas.total_steps()), secs,
      secs > 0 ? static_cast<double>(gas.total_steps()) / secs : 0.0,
      params.dt_min, params.dt_max);
}

TEST_CASE(retro_sonic_env) {
  // Needs a user-supplied Sonic The Hedgehog (Genesis) ROM.
  const char* sonic_rom = std::getenv("FG_SONIC_ROM");
  if (!sonic_rom) {
    std::printf("  SKIP retro_sonic_env (set FG_SONIC_ROM=/path/to/sonic.md)\n");
    return;
  }
  const std::string state =
      "third_party/stable-retro/stable_retro/data/stable/"
      "SonicTheHedgehog-Genesis-v0/GreenHillZone.Act1.state";
  RetroEnv env(core_so(), sonic_rom, state, 2, ObsMode::kCoords,
               RetroGame::kSonic);
  CHECK(env.obs_dim() == kSonicCoordsDim);
  std::vector<char> s;
  std::vector<float> o;
  env.reset(s, o);
  CHECK(o.size() == static_cast<size_t>(kSonicCoordsDim));
  CHECK(o[5] == 0.0f);  // progress: Green Hill (zone 0) act 1
  // Hold right for a while: x must increase and pay positive reward.
  std::vector<std::vector<char>> st = {s}, ns(1);
  std::vector<float> obs(static_cast<size_t>(kSonicCoordsDim)), rw(1);
  std::vector<uint8_t> dn(1), tr(1);
  const float x0 = o[0];
  float total = 0.0f;
  float max_x_vel = 0.0f;
  for (int i = 0; i < 40 && !dn[0]; ++i) {
    env.step_batch(st, {2}, {4}, ns, obs, rw, dn, tr);  // action 2 = right
    st[0] = ns[0];
    total += rw[0];
    if (obs[2] > max_x_vel) max_x_vel = obs[2];
  }
  CHECK(obs[0] > x0);
  CHECK(total > 0.0f);
  CHECK(max_x_vel > 0.0f);  // x_vel @0xFFD010 reads real rightward motion
}
