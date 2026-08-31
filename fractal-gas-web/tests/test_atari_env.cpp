// Atari (ALE) smoke tests. They need a user-supplied Atari 2600 ROM whose
// filename matches a game ALE supports (snake_case, e.g. breakout.bin):
//   FG_ATARI_ROM=/path/to/breakout.bin ./build-atari/fg_atari_tests
// Without FG_ATARI_ROM the tests report SKIP and pass.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "atari_env.hpp"
#include "fractal_gas.hpp"
#include "test_framework.hpp"

using namespace fg;

namespace {

const char* rom_path() { return std::getenv("FG_ATARI_ROM"); }

bool skip_if_no_rom(const char* test_name) {
  if (rom_path() == nullptr) {
    std::printf("  SKIP %s (set FG_ATARI_ROM=/path/to/game.bin to run)\n",
                test_name);
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE(atari_reset_reaches_valid_state) {
  if (skip_if_no_rom("atari_reset_reaches_valid_state")) return;
  AtariEnv env(rom_path(), 2);
  CHECK(env.n_actions() >= 2);          // every minimal set has >= 2 actions
  CHECK(env.frame_width() > 0);
  CHECK(env.frame_height() > 0);
  CHECK(env.obs_dim() == 128);          // default mode is kRam

  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  CHECK(obs.size() == static_cast<size_t>(env.obs_dim()));
  CHECK(state.size() > sizeof(float));  // serialized ALEState + carry
  for (float v : obs) CHECK(v >= 0.0f && v <= 255.0f);

  // A second reset must return the cached initial state bit-identically.
  std::vector<char> state2;
  std::vector<float> obs2;
  env.reset(state2, obs2);
  CHECK(state == state2);
  CHECK(obs == obs2);
}

TEST_CASE(atari_state_roundtrip_is_deterministic) {
  if (skip_if_no_rom("atari_state_roundtrip_is_deterministic")) return;
  AtariEnv env(rom_path(), 2);
  std::vector<char> init_state;
  std::vector<float> init_obs;
  env.reset(init_state, init_obs);

  // Step the SAME blob twice with the same action; with 2 threads the static
  // block partition puts the two walkers on different slots, so this checks
  // both determinism and blob portability across emulator instances.
  const int32_t action = env.n_actions() > 1 ? 1 : 0;
  std::vector<std::vector<char>> states = {init_state, init_state};
  std::vector<std::vector<char>> new_states(2);
  std::vector<float> obs(2 * static_cast<size_t>(env.obs_dim()));
  std::vector<float> rewards(2);
  std::vector<uint8_t> dones(2), truncated(2);
  const std::vector<int32_t> actions = {action, action};
  const std::vector<int32_t> dt = {3, 3};
  for (int step = 0; step < 10; ++step) {
    env.step_batch(states, actions, dt, new_states, obs, rewards, dones,
                   truncated);
    states[0] = new_states[0];
    states[1] = new_states[1];
  }
  CHECK(states[0] == states[1]);
  CHECK(rewards[0] == rewards[1]);
  CHECK(dones[0] == dones[1]);
  const size_t d = static_cast<size_t>(env.obs_dim());
  for (size_t k = 0; k < d; ++k) CHECK(obs[k] == obs[d + k]);
}

TEST_CASE(atari_obs_mode_dims) {
  if (skip_if_no_rom("atari_obs_mode_dims")) return;
  int32_t w = 0;
  int32_t h = 0;
  for (int m = 0; m < 4; ++m) {
    AtariEnv env(rom_path(), 1, static_cast<AtariObsMode>(m));
    w = env.frame_width();
    h = env.frame_height();
    std::vector<char> state;
    std::vector<float> obs;
    env.reset(state, obs);
    CHECK(static_cast<int32_t>(obs.size()) == env.obs_dim());
    switch (static_cast<AtariObsMode>(m)) {
      case AtariObsMode::kRam:
      case AtariObsMode::kCoords:  // documented fallback to RAM
        CHECK(env.obs_dim() == 128);
        break;
      case AtariObsMode::kRgb:
        CHECK(env.obs_dim() == w * h * 3);
        break;
      case AtariObsMode::kGray:
        CHECK(env.obs_dim() == w * h);
        break;
    }
  }
}

TEST_CASE(atari_render_frame_produces_rgba) {
  if (skip_if_no_rom("atari_render_frame_produces_rgba")) return;
  AtariEnv env(rom_path(), 1);
  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  std::vector<uint8_t> rgba;
  env.render_frame(state, rgba);
  CHECK(static_cast<int32_t>(rgba.size()) ==
        env.frame_width() * env.frame_height() * 4);
  int64_t sum = 0;
  for (size_t i = 0; i < rgba.size(); i += 4) {
    sum += rgba[i] + rgba[i + 1] + rgba[i + 2];
    CHECK(rgba[i + 3] == 0xFF);
  }
  CHECK(sum > 0);  // screen not all black
}

TEST_CASE(atari_full_gas_smoke) {
  if (skip_if_no_rom("atari_full_gas_smoke")) return;
  AtariEnv env(rom_path(), 4);
  FractalGasParams params;
  params.N = 16;
  params.seed = 7;
  params.use_cumulative_reward = true;

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
}
