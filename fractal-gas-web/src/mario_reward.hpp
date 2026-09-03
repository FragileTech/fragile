// Super Mario Bros reward, ported from gym-super-mario-bros
// (Kautenja, smb_env.py) reading NES CPU RAM directly.
//
// The carry variables (x_position_last, time_last) are per-walker episode
// state: they MUST be stored beside the emulator dump in the walker's state
// blob so cloning copies them along with the emulator state.
#ifndef FRACTAL_GAS_MARIO_REWARD_HPP
#define FRACTAL_GAS_MARIO_REWARD_HPP

#include <cstdint>

namespace fg {

struct MarioCarry {
  int32_t x_last = 0;
  int32_t time_last = 0;
  int32_t flag_last = 0;   // was the flag grabbed on the previous frame?
  int32_t area_last = -1;  // sub-area byte (ram[0x0760]) on the previous frame
  int32_t stage_last = -1;    // (world << 8) | stage on the previous frame
  int32_t visited_areas = 0;  // bitmask of sub-areas seen this stage
};

/// One-time bonus for grabbing the flagpole, applied OUTSIDE the +-15 frame
/// clip (the clip would neuter it).
constexpr float kFlagBonus = 500.0f;

/// One-time bonus for entering a sub-area (pipe / warp / bonus room) not
/// visited before in this stage, applied outside the clip. Progress through
/// SMB levels like 1-2 requires entering pipes even though that means moving
/// backwards in x; this pays for the detour. The visited mask (cleared on
/// stage change) prevents farming a pipe back and forth.
constexpr float kAreaBonus = 100.0f;

struct MarioFrameResult {
  float reward = 0.0f;
  bool done = false;
};

/// Live-tunable weights of the reward terms (the web demo's "Reward terms"
/// panel). Defaults reproduce the constants documented below exactly.
struct MarioRewardWeights {
  float x = 1.0f;         // per pixel of x progress
  float time = 1.0f;      // per clock unit lost
  float death = 25.0f;    // penalty on dying (inside the clip)
  float clip = 15.0f;     // |x + time + death| per-frame clip
  float flag = kFlagBonus;
  float area = kAreaBonus;
};

/// smb_env.py: ram[0x6d] * 0x100 + ram[0x86]
int32_t mario_x_position(const uint8_t* ram);

/// smb_env.py: the in-game clock, three decimal digits at 0x7f8..0x7fa.
int32_t mario_time(const uint8_t* ram);

/// Evaluate one emulated frame: updates the carry and returns the frame
/// reward and done flag.
///
/// smb_env.py reference:
///   _x_reward     = x - x_last (then x_last = x); |Δ| > 5 -> 0 (glitch guard)
///   _time_penalty = time - time_last (then time_last = time); > 0 -> 0
///   _death_penalty= -25 if _is_dying or _is_dead else 0
///   _is_dying     = player_state == 0x0b or y_viewport > 1
///   _is_dead      = player_state == 0x06
///   _flag_get     = player_state == 0x04 or player_state == 0x05
///   game over     = lives (ram[0x75a]) == 0xff
///   reward        = clip(_x_reward + _time_penalty + _death_penalty, -15, 15)
///   done          = _is_dying or _is_dead or game over or _flag_get
///
/// DELIBERATE DEVIATIONS from smb_env.py (demo-only, env-side — the fractal
/// gas algorithm itself is untouched):
///   - _flag_get does NOT set done. In the gas, done walkers are forced to
///     clone away, so ending the episode on the flag makes finishing the
///     level indistinguishable from dying; instead the walker keeps playing
///     through the castle walk into the next level.
///   - Grabbing the flag pays kFlagBonus once (on the not-flag -> flag
///     transition), added after the clip, so finishing is the
///     highest-reward event and the swarm is pulled toward it.
///   - During the flag sequence the end-of-level countdown (1 clock unit
///     per frame) counts as +1 progress per tick instead of a -1 time
///     penalty; otherwise it forms a ~-333 reward barrier that cloning
///     refuses to cross and the swarm stalls at the castle door.
///   - Entering a sub-area (ram[0x0760] change) not yet visited this stage
///     pays kAreaBonus once, outside the clip — see kAreaBonus above.
///   - Pipe/area transition animations (player_state 0x02/0x03/0x07) pay
///     +1 per frame so the zero-reward animation is not a fitness wall for
///     a converged swarm.
MarioFrameResult mario_frame_update(const uint8_t* ram, MarioCarry& carry,
                                    const MarioRewardWeights& w = {});

}  // namespace fg

#endif  // FRACTAL_GAS_MARIO_REWARD_HPP
