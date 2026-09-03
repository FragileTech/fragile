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

// render_frame must draw the RESTORED walker, not whatever slot 0 emulated
// last: ALE's restoreSystemState() leaves the screen buffer stale, so the
// env redraws one frame after restoring.
TEST_CASE(atari_render_frame_follows_restored_state) {
  if (skip_if_no_rom("atari_render_frame_follows_restored_state")) return;
  AtariEnv env(rom_path(), 1);
  std::vector<char> state_a;
  std::vector<float> obs;
  env.reset(state_a, obs);
  std::vector<uint8_t> frame_a;
  env.render_frame(state_a, frame_a);

  // Play forward on the same emulator slot to some later, different state.
  std::vector<std::vector<char>> states = {state_a};
  std::vector<std::vector<char>> new_states(1);
  std::vector<float> batch_obs(static_cast<size_t>(env.obs_dim()));
  std::vector<float> rewards(1);
  std::vector<uint8_t> dones(1), truncated(1);
  const std::vector<int32_t> actions = {env.n_actions() > 3 ? 3 : 1};
  const std::vector<int32_t> dt = {30};
  for (int i = 0; i < 3; ++i) {
    env.step_batch(states, actions, dt, new_states, batch_obs, rewards, dones,
                   truncated);
    states[0] = new_states[0];
  }
  std::vector<uint8_t> frame_b;
  env.render_frame(states[0], frame_b);
  CHECK(frame_b != frame_a);  // the game moved on

  // Rendering the initial state again must reproduce its frame exactly,
  // even though slot 0 just emulated state B.
  std::vector<uint8_t> frame_a2;
  env.render_frame(state_a, frame_a2);
  CHECK(frame_a2 == frame_a);
}

// Needs a ROM with a life counter (the suggested breakout.bin has 5 lives).
// Holding FIRE launches the ball with the paddle parked, so lives drain one
// by one: each loss must fire done with done_is_recoverable() true (game
// still playable), stepping the post-loss blob must not immediately re-fire
// done, and only the final life reaches the non-recoverable game over.
TEST_CASE(atari_life_loss_is_recoverable_death) {
  if (skip_if_no_rom("atari_life_loss_is_recoverable_death")) return;
  if (std::string(rom_path()).find("breakout") == std::string::npos) {
    std::printf("  SKIP atari_life_loss_is_recoverable_death (needs "
                "breakout.bin: FIRE with a parked paddle drains lives)\n");
    return;
  }
  AtariEnv env(rom_path(), 1);
  std::vector<char> state;
  std::vector<float> init_obs;
  env.reset(state, init_obs);

  const int32_t fire = env.n_actions() > 1 ? 1 : 0;
  std::vector<std::vector<char>> states = {state};
  std::vector<std::vector<char>> new_states(1);
  std::vector<float> obs(static_cast<size_t>(env.obs_dim()));
  std::vector<float> rewards(1);
  std::vector<uint8_t> dones(1), truncated(1);
  const std::vector<int32_t> actions = {fire};
  const std::vector<int32_t> dt = {4};

  int soft_deaths = 0;
  bool hard_death = false;
  bool just_soft_died = false;
  for (int step = 0; step < 20000 && !hard_death; ++step) {
    env.step_batch(states, actions, dt, new_states, obs, rewards, dones,
                   truncated);
    if (just_soft_died) {
      // The revived walker's blob carries the reduced lives count, so the
      // step after a life loss must not re-fire done.
      CHECK(dones[0] == 0);
      just_soft_died = false;
    }
    if (dones[0]) {
      if (env.done_is_recoverable(0)) {
        ++soft_deaths;
        just_soft_died = true;  // keep stepping the post-loss blob = revive
      } else {
        hard_death = true;
      }
    }
    states[0] = new_states[0];
  }
  CHECK(soft_deaths >= 1);  // lives before the last are recoverable deaths
  CHECK(hard_death);        // the final life ends in a real game over
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

// Montezuma's Revenge: dedicated RAM logic (src/montezuma_logic.hpp). Needs
// the Montezuma ROM specifically:
//   FG_ATARI_ROM=web/roms/atari/montezuma_revenge.bin ./build-atari/fg_atari_tests
namespace {
bool rom_is_montezuma() {
  const char* p = rom_path();
  return p != nullptr && std::string(p).find("montezuma") != std::string::npos;
}
}  // namespace

TEST_CASE(montezuma_coords_and_walker_accessors) {
  if (skip_if_no_rom("montezuma_coords_and_walker_accessors")) return;
  if (!rom_is_montezuma()) {
    std::printf("  SKIP montezuma_coords_and_walker_accessors (FG_ATARI_ROM is "
                "not montezuma_revenge.bin)\n");
    return;
  }
  AtariEnv env(rom_path(), 2, AtariObsMode::kCoords, AtariGame::kMontezuma);
  CHECK(env.game() == AtariGame::kMontezuma);
  CHECK(env.obs_dim() == kMontezumaCoordsDim);
  CHECK(env.frame_width() == 160);
  CHECK(env.frame_height() == 210);

  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  CHECK(obs.size() == static_cast<size_t>(kMontezumaCoordsDim));
  // Panama Joe starts on the top platform of room 1 with 5 lives.
  CHECK(obs[2] == 1.0f);
  CHECK(obs[3] == 77.0f);
  CHECK(obs[4] == 235.0f);
  CHECK(obs[5] == 0.0f);
  CHECK(obs[7] == 5.0f);
  CHECK(obs[0] == 4 * 160 + 80);
  CHECK(obs[1] == 28);
  // The reset carry already marks the start room so no bonus is paid for
  // standing still.
  const AtariCarry carry = AtariEnv::read_carry(state);
  CHECK(carry.episode_return == 0.0f);
  CHECK(carry.visited_rooms == (1u << 1));

  const int32_t n = 2;
  std::vector<std::vector<char>> states(n, state);
  std::vector<std::vector<char>> new_states(n);
  std::vector<float> batch_obs(n * static_cast<size_t>(env.obs_dim()));
  std::vector<float> rewards(n);
  std::vector<uint8_t> dones(n), truncated(n);
  const std::vector<int32_t> actions = {0, 0};  // NOOP: stays put, alive
  const std::vector<int32_t> dt = {4, 4};
  for (int step = 0; step < 5; ++step) {
    env.step_batch(states, actions, dt, new_states, batch_obs, rewards, dones,
                   truncated);
    for (int32_t i = 0; i < n; ++i) {
      const float* row = batch_obs.data() + i * env.obs_dim();
      CHECK(env.walker_room(i) == static_cast<int32_t>(row[2]));
      CHECK(env.walker_level(i) == static_cast<int32_t>(row[5]));
      CHECK(env.walker_lives(i) == static_cast<int32_t>(row[7]));
      CHECK(env.walker_inventory(i) == static_cast<int32_t>(row[6]));
      CHECK(env.walker_x(i) == montezuma_room_px(static_cast<int32_t>(row[3])));
      CHECK(env.walker_y(i) == montezuma_room_py(static_cast<int32_t>(row[4])));
      CHECK(std::isfinite(env.display_score(i)));
      CHECK(rewards[i] == 0.0f);  // no score, no new room while idling
      CHECK(dones[i] == 0);
      states[i] = new_states[i];
    }
  }
  CHECK(env.walker_room(0) == 1);
}

TEST_CASE(montezuma_generic_game_keeps_ram_coords) {
  if (skip_if_no_rom("montezuma_generic_game_keeps_ram_coords")) return;
  // Whatever the ROM, the generic game keeps the documented coords==RAM
  // alias and a zeroed carry.
  AtariEnv env(rom_path(), 1, AtariObsMode::kCoords, AtariGame::kGeneric);
  CHECK(env.obs_dim() == 128);
  std::vector<char> state;
  std::vector<float> obs;
  env.reset(state, obs);
  const AtariCarry carry = AtariEnv::read_carry(state);
  CHECK(carry.episode_return == 0.0f);
  CHECK(carry.visited_rooms == 0u);
  CHECK(carry.level_last == 0);
}
