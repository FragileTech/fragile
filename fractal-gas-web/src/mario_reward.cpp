#include "mario_reward.hpp"

#include <algorithm>

namespace fg {

int32_t mario_x_position(const uint8_t* ram) {
  // smb_env.py::_x_position: self.ram[0x6d] * 0x100 + self.ram[0x86]
  return static_cast<int32_t>(ram[0x6D]) * 0x100 + static_cast<int32_t>(ram[0x86]);
}

int32_t mario_time(const uint8_t* ram) {
  // smb_env.py::_time: int(''.join(map(str, self.ram[0x7f8:0x7fb])))
  return 100 * static_cast<int32_t>(ram[0x7F8]) +
         10 * static_cast<int32_t>(ram[0x7F9]) +
         static_cast<int32_t>(ram[0x7FA]);
}

MarioFrameResult mario_frame_update(const uint8_t* ram, MarioCarry& carry,
                                    const MarioRewardWeights& w) {
  // _x_reward: _reward = x - x_last; x_last = x;
  //            if _reward < -5 or _reward > 5: return 0
  const int32_t x = mario_x_position(ram);
  int32_t x_reward = x - carry.x_last;
  carry.x_last = x;
  if (x_reward < -5 || x_reward > 5) x_reward = 0;

  // _time_penalty: _reward = time - time_last; time_last = time;
  //                if _reward > 0: return 0
  const int32_t time = mario_time(ram);
  int32_t time_penalty = time - carry.time_last;
  carry.time_last = time;
  if (time_penalty > 0) time_penalty = 0;

  const uint8_t player_state = ram[0x0E];
  const uint8_t y_viewport = ram[0xB5];
  const bool is_dying = player_state == 0x0B || y_viewport > 1;
  const bool is_dead = player_state == 0x06;
  const bool flag_get = player_state == 0x04 || player_state == 0x05;
  const bool game_over = ram[0x75A] == 0xFF;

  // Deviation from smb_env.py (see header): during the end-of-level
  // sequence the game converts the remaining clock to score at 1 unit per
  // frame. As a raw time_penalty that is a ~-333 cumulative-reward barrier
  // the swarm refuses to cross (deeper countdown = lower reward = cloned
  // away). Mirror the game and count each countdown tick as +1 progress so
  // the reward gradient points through the transition into the next level.
  if ((flag_get || carry.flag_last) && time_penalty < 0) {
    time_penalty = -time_penalty;
  }

  const float death_penalty = (is_dying || is_dead) ? -w.death : 0.0f;

  float reward = std::min(
      w.clip, std::max(-w.clip, w.x * static_cast<float>(x_reward) +
                                    w.time * static_cast<float>(time_penalty) +
                                    death_penalty));

  // Deviation from smb_env.py (see header): one-time flag bonus outside the
  // clip, and flag_get does not end the episode.
  const bool flag_grabbed = flag_get && !carry.flag_last;
  carry.flag_last = flag_get ? 1 : 0;
  if (flag_grabbed) reward += w.flag;

  // Deviation from smb_env.py (see header): one-time bonus for entering a
  // sub-area not visited before in this stage (pipes, warps, bonus rooms).
  // ram[0x0760] = sub-area byte; the visited bitmask resets on stage/world
  // change (a new stage is fresh exploration).
  const int32_t area = ram[0x760];
  const int32_t stage_key = (static_cast<int32_t>(ram[0x75F]) << 8) |
                            static_cast<int32_t>(ram[0x75C]);
  if (stage_key != carry.stage_last) {
    carry.stage_last = stage_key;
    carry.visited_areas = 0;
  }
  const int32_t area_bit = 1 << (area & 31);
  if (area != carry.area_last && !(carry.visited_areas & area_bit) &&
      carry.area_last >= 0) {
    reward += w.area;
  }
  carry.visited_areas |= area_bit;
  carry.area_last = area;

  // Deviation from smb_env.py (see header): +1 per frame while a pipe/area
  // transition animation plays (player_state 0x02 = entering sideways pipe,
  // 0x03 = going down a pipe, 0x07 = entering area). These animations pay
  // no x-progress, and once the swarm has converged the fitness
  // standardization turns any zero-reward stretch into a wall (a walker a
  // few points behind the pack is cloned away with near certainty), so the
  // ~50-frame animation is uncrossable without a positive gradient. The
  // emulator always completes the animation when stepped, so this cannot be
  // farmed indefinitely.
  const bool in_transition =
      player_state == 0x02 || player_state == 0x03 || player_state == 0x07;
  if (in_transition) reward += 1.0f;

  MarioFrameResult result;
  result.reward = reward;
  result.done = is_dying || is_dead || game_over;
  return result;
}

}  // namespace fg
