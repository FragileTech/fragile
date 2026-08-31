// NES smoke tests. They need a user-supplied Super Mario Bros ROM:
//   FG_ROM=/path/to/smb.nes ./build/fg_tests
// Without FG_ROM the tests report SKIP and pass.
#include <cstdlib>
#include <string>

#include "fractal_gas.hpp"
#include "mario_reward.hpp"
#include "nes_env.hpp"
#include "test_framework.hpp"

using namespace fg;

namespace {

const char* rom_path() { return std::getenv("FG_ROM"); }

bool skip_if_no_rom(const char* test_name) {
  if (rom_path() == nullptr) {
    std::printf("  SKIP %s (set FG_ROM=/path/to/smb.nes to run)\n", test_name);
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE(nes_reset_reaches_playable_state) {
  if (skip_if_no_rom("nes_reset_reaches_playable_state")) return;
  NesMarioEnv env(rom_path(), 2);
  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  CHECK(obs.size() == 0x800);
  CHECK(state.size() > 0x800);
  // After skip_start_screen the in-game clock must be ticking (nonzero).
  const uint8_t* ram_bytes = nullptr;
  // The observation IS the RAM: reconstruct time from it.
  (void)ram_bytes;
  const int32_t time = 100 * static_cast<int32_t>(obs[0x7F8]) +
                       10 * static_cast<int32_t>(obs[0x7F9]) +
                       static_cast<int32_t>(obs[0x7FA]);
  CHECK(time > 0);
}

TEST_CASE(nes_holding_right_moves_mario_right) {
  if (skip_if_no_rom("nes_holding_right_moves_mario_right")) return;
  NesMarioEnv env(rom_path(), 2);
  std::vector<char> init_state;
  std::vector<float> init_obs;
  env.reset(init_state, init_obs);

  const int32_t x0 = static_cast<int32_t>(init_obs[0x6D]) * 0x100 +
                     static_cast<int32_t>(init_obs[0x86]);

  // Step one walker holding right+B (run) for many frames.
  std::vector<std::vector<char>> states = {init_state};
  std::vector<std::vector<char>> new_states(1);
  std::vector<float> obs(0x800);
  std::vector<float> rewards(1);
  std::vector<uint8_t> dones(1), truncated(1);
  const std::vector<int32_t> actions = {3};  // right + B
  const std::vector<int32_t> dt = {4};
  float total_reward = 0.0f;
  for (int step = 0; step < 30 && !dones[0]; ++step) {
    env.step_batch(states, actions, dt, new_states, obs, rewards, dones,
                   truncated);
    states[0] = new_states[0];
    total_reward += rewards[0];
    CHECK(rewards[0] >= -15.0f * 4 && rewards[0] <= 15.0f * 4);
  }
  const int32_t x1 = static_cast<int32_t>(obs[0x6D]) * 0x100 +
                     static_cast<int32_t>(obs[0x86]);
  CHECK(x1 > x0);
  CHECK(total_reward > 0.0f);
}

TEST_CASE(nes_state_roundtrip_is_deterministic) {
  if (skip_if_no_rom("nes_state_roundtrip_is_deterministic")) return;
  NesMarioEnv env(rom_path(), 2);
  std::vector<char> init_state;
  std::vector<float> init_obs;
  env.reset(init_state, init_obs);

  // Step the SAME blob twice with the same action: results must be
  // bit-identical (this is what makes cloning walkers sound).
  std::vector<std::vector<char>> states = {init_state, init_state};
  std::vector<std::vector<char>> new_states(2);
  std::vector<float> obs(2 * 0x800);
  std::vector<float> rewards(2);
  std::vector<uint8_t> dones(2), truncated(2);
  const std::vector<int32_t> actions = {2, 2};
  const std::vector<int32_t> dt = {3, 3};
  for (int step = 0; step < 10; ++step) {
    env.step_batch(states, actions, dt, new_states, obs, rewards, dones,
                   truncated);
    states[0] = new_states[0];
    states[1] = new_states[1];
  }
  CHECK(states[0] == states[1]);
  CHECK(rewards[0] == rewards[1]);
  for (size_t k = 0; k < 0x800; ++k) CHECK(obs[k] == obs[0x800 + k]);
}

TEST_CASE(nes_render_frame_produces_rgba) {
  if (skip_if_no_rom("nes_render_frame_produces_rgba")) return;
  NesMarioEnv env(rom_path(), 1);
  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  std::vector<uint8_t> rgba;
  env.render_frame(state, rgba);
  CHECK(static_cast<int32_t>(rgba.size()) ==
        env.frame_width() * env.frame_height() * 4);
  // Not all black, and alpha fully opaque.
  int64_t sum = 0;
  for (size_t i = 0; i < rgba.size(); i += 4) {
    sum += rgba[i] + rgba[i + 1] + rgba[i + 2];
    CHECK(rgba[i + 3] == 0xFF);
  }
  CHECK(sum > 0);
}

TEST_CASE(nes_obs_modes) {
  if (skip_if_no_rom("nes_obs_modes")) return;

  // Build one observation per mode from the same freshly-reset state.
  std::vector<std::vector<float>> obs_by_mode;
  for (int m = 0; m < 4; ++m) {
    NesMarioEnv env(rom_path(), 1, static_cast<ObsMode>(m));
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

  CHECK(ram.size() == 0x800);
  CHECK(rgb.size() == 256 * 240 * 3);
  CHECK(gray.size() == 256 * 240);
  CHECK(coords.size() == static_cast<size_t>(kCoordsDim));

  // The reset is deterministic, so all modes saw the same game state.
  // Gray must equal the luminance of rgb.
  double rgb_sum = 0.0;
  for (size_t p = 0; p < gray.size(); ++p) {
    const float lum = 0.299f * rgb[p * 3] + 0.587f * rgb[p * 3 + 1] +
                      0.114f * rgb[p * 3 + 2];
    CHECK_CLOSE(gray[p], lum, 1e-5);
    rgb_sum += rgb[p * 3] + rgb[p * 3 + 1] + rgb[p * 3 + 2];
  }
  CHECK(rgb_sum > 0.0);  // screen not all black

  // Coords entries must match the corresponding RAM fields.
  const float x = ram[0x6D] * 256.0f + ram[0x86];
  const float time =
      100.0f * ram[0x7F8] + 10.0f * ram[0x7F9] + ram[0x7FA];
  CHECK(coords[0] == x);
  CHECK(coords[1] == ram[0x3B8]);  // y pixel
  CHECK(coords[2] == ram[0xB5]);   // y viewport
  CHECK(coords[3] == ram[0x75F]);  // world
  CHECK(coords[4] == ram[0x75C]);  // stage
  CHECK(coords[5] == time);
  CHECK(coords[8] == ram[0x760]);  // sub-area
  CHECK(coords[9] == ram[0x756]);  // power-up
}

TEST_CASE(nes_start_level_selection) {
  if (skip_if_no_rom("nes_start_level_selection")) return;
  const int targets[][2] = {{1, 2}, {2, 1}, {4, 2}, {8, 3}};
  for (const auto& t : targets) {
    NesMarioEnv env(rom_path(), 1, ObsMode::kRam, t[0], t[1]);
    std::vector<char> state;
    std::vector<float> obs;
    env.reset(state, obs);
    // RAM observation: world at 0x75F, stage at 0x75C (0-indexed).
    CHECK(static_cast<int>(obs[0x75F]) == t[0] - 1);
    CHECK(static_cast<int>(obs[0x75C]) == t[1] - 1);
    // Playable state: the clock is ticking.
    const int32_t time = 100 * static_cast<int32_t>(obs[0x7F8]) +
                         10 * static_cast<int32_t>(obs[0x7F9]) +
                         static_cast<int32_t>(obs[0x7FA]);
    CHECK(time > 0);
  }
}

TEST_CASE(nes_coords_gas_smoke) {
  if (skip_if_no_rom("nes_coords_gas_smoke")) return;
  NesMarioEnv env(rom_path(), 4, ObsMode::kCoords);
  FractalGasParams params;
  params.N = 16;
  params.seed = 11;
  params.use_cumulative_reward = true;
  FractalGas gas(env, params);
  gas.reset();
  for (int it = 0; it < 15; ++it) {
    const StepInfo info = gas.step();
    CHECK(std::isfinite(info.mean_virtual_reward));
  }
  CHECK(gas.get_best_walker().second > -1e30f);
}

TEST_CASE(nes_full_gas_smoke) {
  if (skip_if_no_rom("nes_full_gas_smoke")) return;
  NesMarioEnv env(rom_path(), 4);
  FractalGasParams params;
  params.N = 16;
  params.seed = 7;
  params.use_cumulative_reward = true;
  params.record_frames = true;
  params.n_elite = 2;

  FractalGas gas(env, params);
  gas.reset();
  float last_max = -1e30f;
  for (int it = 0; it < 20; ++it) {
    const StepInfo info = gas.step();
    CHECK(std::isfinite(info.mean_reward));
    CHECK(std::isfinite(info.mean_virtual_reward));
    CHECK(info.alive_count >= 0 && info.alive_count <= params.N);
    last_max = info.max_reward;
  }
  CHECK(last_max > -1e30f);
  CHECK(!gas.best_frame().empty());
}
