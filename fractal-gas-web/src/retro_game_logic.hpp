// Per-game logic for the Genesis environments (reward, termination, coords,
// showcase ranking, boot-to-gameplay), shared header-only between:
//   - RetroEnv (native: parallel dlopen-per-copy cores)
//   - the wasm core-worker shim (retro_shim.cpp: one statically linked core
//     per Web Worker)
// so both paths compute byte-identical results.
//
// Integration data sources (stable-retro, addresses are 68k bus addresses;
// work RAM base 0xFF0000):
//   Airstriker-Genesis-v0/data.json:
//     score >u4 @0xFF024E, lives >u2 @0xFF025A, gameover >u2 @0xFF0266
//     scenario.json: reward = delta(score); done = gameover==1 && lives==0
//   SonicTheHedgehog-Genesis-v0/data.json:
//     x >i2 @0xFFD008, y >u2 @0xFFD00C, rings >u2 @0xFFFE20,
//     score >u4 @0xFFFE26, lives |u1 @0xFFFE12, zone |u1 @0xFFFE10,
//     act |u1 @0xFFFE11
#ifndef FRACTAL_GAS_RETRO_GAME_LOGIC_HPP
#define FRACTAL_GAS_RETRO_GAME_LOGIC_HPP

#include <cstdint>
#include <cstring>

#include "retro_core.hpp"

namespace fg {

/// Which game's integration data drives reward / termination / coords /
/// showcase ranking. The emulator core is game-agnostic.
enum class RetroGame : int32_t {
  kAirstriker = 0,  // freeware, ROM bundled
  kSonic = 1,       // user-supplied ROM
};

/// Genesis joypad action sets (libretro RETRO_DEVICE_ID_JOYPAD_* bitmasks).
/// Airstriker: NOOP, left, right, up, down, B, left+B, right+B.
inline constexpr uint32_t kRetroActionMasks[8] = {
    0, kPadLeft, kPadRight, kPadUp, kPadDown,
    kPadB, kPadLeft | kPadB, kPadRight | kPadB,
};
/// Sonic swaps UP (camera tilt only in Sonic 1 — a duplicate NOOP) for
/// RIGHT+DOWN: sustained rolling, which is faster than running downhill
/// and kills badniks on contact (no ring-dump penalty).
inline constexpr uint32_t kSonicActionMasks[8] = {
    0, kPadLeft, kPadRight, kPadRight | kPadDown, kPadDown,
    kPadB, kPadLeft | kPadB, kPadRight | kPadB,
};
inline constexpr int32_t kRetroNumActions = 8;

/// kCoords layout per game:
///   Airstriker: [score, lives, gameover]
///   Sonic:      [x/256, y/256, x_vel/256, y_vel/256, ground_speed/256,
///                (zone*3 + act) * 100]
/// The Sonic tuple feeds the walker L2 distance (the diversity pressure),
/// so it encodes STATE, not reward: position, the momentum that decides
/// what is reachable (loops, ramps), and a single progress coordinate.
/// Score/rings/lives are excluded — they proxy reward, which enters the
/// fitness through its own term. Dimensions are rescaled to comparable
/// magnitude (raw score would otherwise dominate raw zone by ~1e4):
/// /256 puts pixels and subpixel velocities in tile units, and the *100
/// progress scale makes a different act always read as "far".
inline constexpr int32_t kRetroCoordsDim = 3;
inline constexpr int32_t kSonicCoordsDim = 6;
inline constexpr float kSonicCoordScale = 1.0f / 256.0f;
inline constexpr float kSonicProgressScale = 100.0f;

/// Mega Drive native resolution (both games; stable-retro pins the same).
inline constexpr int32_t kRetroFrameWidth = 320;
inline constexpr int32_t kRetroFrameHeight = 224;

/// Sonic reward shaping (delta/potential form: every term pays the signed
/// CHANGE of a carried value, so nothing is farmable by oscillation).
/// Budget per act keeps "finish the game" dominant: completion bonus ~5000
/// >= act x-progress (3000-9000px) >> end-of-act tally * score coef >>
/// rings (~100-200 * coef) >> single pickups. asymmetric_rescale compares
/// walkers per step, so only these relative magnitudes matter.
inline constexpr float kSonicActBonus = 5000.0f;   // signpost > any wandering
inline constexpr float kSonicRingCoef = 3.0f;      // signed: a hit dumps all
                                                   // rings -> damage penalty
inline constexpr float kSonicScoreCoef = 0.5f;     // boss hits (100/hit),
                                                   // badniks, monitors, tally
inline constexpr float kSonicLifeBonus = 1000.0f;  // per gained life (1-ups,
                                                   // 100-ring bonus)
/// Paid per boss hit point removed. Bosses take 8 hits, so a full kill is
/// worth 16000 — dominating everything else the arena offers. While a boss
/// object is loaded the per-frame x-progress term is DISABLED: boss arenas
/// lock the camera, so "move right" is noise there and previously kept
/// walkers glued to the right wall instead of fighting.
inline constexpr float kSonicBossHitBonus = 2000.0f;

/// Exploration bonus (the Mario kAreaBonus idea scaled to a 2-D grid): the
/// act is covered by a coarse cell grid and entering a cell not yet visited
/// by this walker's lineage pays kSonicCellBonus once. The visited bitmask
/// lives in the carry, so a clone inherits its parent's map knowledge and
/// cannot re-farm cells the parent already earned; it is cleared on act
/// change (like Mario's per-stage mask). This is what makes backtracking
/// viable: levels like Marble 1 require going back/down, which the signed
/// x-progress term alone strictly punishes. Budget: a fresh 64px cell pays
/// ~8x the dx earned crossing it, discovery events are rare and lumpy, and
/// the fitness uses cumulative reward — so under asymmetric_rescale a
/// frontier walker stays clone-attractive long after the find. A single
/// step's discovery (usually 1-2 cells) still stays below the act bonus.
inline constexpr float kSonicCellBonus = 500.0f;
inline constexpr int32_t kSonicCellShift = 6;  // 64px cells
inline constexpr int32_t kSonicGridW = 256;    // covers x in [0, 16384)
inline constexpr int32_t kSonicGridH = 32;     // covers y in [0, 2048)
inline constexpr int32_t kSonicVisitedWords = kSonicGridW * kSonicGridH / 64;

/// Sonic 1 boss object ids (per the Sonic 1 disassembly): GHZ Obj3D,
/// MZ Obj73, SYZ Obj75, LZ Obj77, SLZ Obj7A, FZ Eggman Obj85. Each keeps
/// its remaining hit points in the object's collision_property byte
/// (obj+0x21), set to 8 on spawn and decremented per hit.
inline constexpr uint8_t kSonicBossIds[6] = {0x3D, 0x73, 0x75,
                                             0x77, 0x7A, 0x85};

/// Reward carry appended to every state blob so per-step deltas survive
/// walkers hopping between core instances. Airstriker uses score_last;
/// Sonic uses all five.
struct RetroCarry {
  int64_t score_last = 0;
  int32_t x_last = 0;
  int32_t lives_last = 0;
  int32_t rings_last = 0;
  int32_t progress_last = 0;  // zone*3 + act
  int32_t b_held_last = 0;    // B held at the end of the previous step
                              // (Sonic jump-edge handling)
  int32_t boss_hits_last = -1;  // boss hit points at the previous frame;
                                // -1 = no boss loaded (Sonic only)
  uint64_t visited[kSonicVisitedWords] = {};  // per-act visited-cell bitmask
                                              // (Sonic only, see
                                              // kSonicCellBonus)
};
inline constexpr size_t kRetroCarryBytes = sizeof(RetroCarry);
static_assert(sizeof(RetroCarry) == sizeof(int64_t) + 6 * sizeof(int32_t) +
                                        sizeof(uint64_t) * kSonicVisitedWords,
              "RetroCarry must be trivially copyable with no padding");

/// Mark the cell containing (x, y) as visited; returns true when the cell
/// was NEW. Coordinates outside the grid clamp to the border cells, so a
/// glitched position can at worst pay one border cell, never index out of
/// bounds.
inline bool retro_sonic_visit_cell(RetroCarry& carry, int32_t x, int32_t y) {
  int32_t cx = x >> kSonicCellShift;
  int32_t cy = y >> kSonicCellShift;
  if (cx < 0) cx = 0;
  if (cx > kSonicGridW - 1) cx = kSonicGridW - 1;
  if (cy < 0) cy = 0;
  if (cy > kSonicGridH - 1) cy = kSonicGridH - 1;
  const int32_t bit = cy * kSonicGridW + cx;
  uint64_t& word = carry.visited[bit >> 6];
  const uint64_t mask = uint64_t(1) << (bit & 63);
  if (word & mask) return false;
  word |= mask;
  return true;
}

// -- work-RAM readers ---------------------------------------------------------
// Genesis Plus GX exposes work RAM as native-endian 16-bit words (logical
// big-endian 68k bytes swapped pairwise — stable-retro's overlay
// ["=", ">", 2]), so a big-endian u16 at an even 68k offset is exactly a
// little-endian u16 read of the raw buffer; a single byte lives at offset^1.

inline uint32_t retro_read_be_u16(const uint8_t* ram, size_t offset) {
  return static_cast<uint32_t>(ram[offset]) |
         (static_cast<uint32_t>(ram[offset + 1]) << 8);
}
inline uint32_t retro_read_be_u32(const uint8_t* ram, size_t offset) {
  return (retro_read_be_u16(ram, offset) << 16) |
         retro_read_be_u16(ram, offset + 2);
}
inline uint8_t retro_read_u8(const uint8_t* ram, size_t offset) {
  return ram[offset ^ 1];
}

struct SonicVars {
  int32_t x, y, rings, lives;
  int64_t score;
  int32_t zone, act;
  int32_t x_vel, y_vel, ground_speed;
  int32_t cam_x, cam_y;  // viewport top-left (v_screenposx/y)
};

inline SonicVars retro_read_sonic(const RetroCore& core) {
  const uint8_t* ram = core.work_ram();
  SonicVars v;
  v.x = static_cast<int16_t>(retro_read_be_u16(ram, 0xD008));
  v.y = static_cast<int32_t>(retro_read_be_u16(ram, 0xD00C));
  v.rings = static_cast<int32_t>(retro_read_be_u16(ram, 0xFE20));
  v.lives = static_cast<int32_t>(retro_read_u8(ram, 0xFE12));
  v.score = static_cast<int64_t>(retro_read_be_u32(ram, 0xFE26));
  v.zone = static_cast<int32_t>(retro_read_u8(ram, 0xFE10));
  v.act = static_cast<int32_t>(retro_read_u8(ram, 0xFE11));
  // Sonic's object fields (Sonic 1 disassembly; not in stable-retro's
  // data.json): x_vel/y_vel at obj+0x10/0x12, inertia (signed ground
  // speed while on a surface) at obj+0x14 — subpixels per frame, i16.
  v.x_vel = static_cast<int16_t>(retro_read_be_u16(ram, 0xD010));
  v.y_vel = static_cast<int16_t>(retro_read_be_u16(ram, 0xD012));
  v.ground_speed = static_cast<int16_t>(retro_read_be_u16(ram, 0xD014));
  // Camera / screen position (Sonic 1 disasm v_screenposx @ $FFF700 and
  // v_screenposy @ $FFF704) — verified: player - cam stays within the
  // 320x224 viewport. Used by the fog-of-war swarm map.
  v.cam_x = static_cast<int32_t>(retro_read_be_u16(ram, 0xF700));
  v.cam_y = static_cast<int32_t>(retro_read_be_u16(ram, 0xF704));
  return v;
}

/// Scan Sonic 1's object RAM ($FFD000-$FFEFFF, 0x40-byte slots; slot 0 is
/// Sonic himself) for a loaded boss object. Returns its remaining hit
/// points (collision_property, obj+0x21), or -1 when no boss is loaded.
inline int32_t retro_sonic_boss_hits(const RetroCore& core) {
  const uint8_t* ram = core.work_ram();
  for (size_t slot = 0xD040; slot < 0xF000; slot += 0x40) {
    const uint8_t id = retro_read_u8(ram, slot);
    for (const uint8_t boss_id : kSonicBossIds) {
      if (id == boss_id) {
        return static_cast<int32_t>(retro_read_u8(ram, slot + 0x21));
      }
    }
  }
  return -1;
}

struct AirstrikerVars {
  int64_t score;
  int32_t lives, gameover;
};

inline AirstrikerVars retro_read_airstriker(const RetroCore& core) {
  const uint8_t* ram = core.work_ram();
  AirstrikerVars v;
  v.score = static_cast<int64_t>(retro_read_be_u32(ram, 0x24E));
  v.lives = static_cast<int32_t>(retro_read_be_u16(ram, 0x25A));
  v.gameover = static_cast<int32_t>(retro_read_be_u16(ram, 0x266));
  return v;
}

/// Observation dims per (game, mode 0=RAM 1=RGB 2=Gray 3=Coords).
inline int32_t retro_obs_dim(RetroGame game, int32_t mode) {
  constexpr int32_t kPixels = kRetroFrameWidth * kRetroFrameHeight;
  switch (mode) {
    case 1:
      return kPixels * 3;
    case 2:
      return kPixels;
    case 3:
      return game == RetroGame::kSonic ? kSonicCoordsDim : kRetroCoordsDim;
    default:
      return 0x10000;  // the 64KB 68k work RAM, one float per byte
  }
}

/// Fill the observation row for the given mode (see retro_obs_dim).
inline void retro_fill_obs(RetroGame game, int32_t mode, const RetroCore& core,
                           float* obs_row) {
  switch (mode) {
    case 1:
    case 2: {
      const auto& rgb = core.frame_rgb();
      const int32_t w = core.frame_width() < kRetroFrameWidth
                            ? core.frame_width()
                            : kRetroFrameWidth;
      const int32_t h = core.frame_height() < kRetroFrameHeight
                            ? core.frame_height()
                            : kRetroFrameHeight;
      const size_t dim = static_cast<size_t>(retro_obs_dim(game, mode));
      std::memset(obs_row, 0, dim * sizeof(float));
      if (rgb.empty()) break;
      for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src =
            rgb.data() + static_cast<size_t>(y) * core.frame_width() * 3;
        for (int32_t x = 0; x < w; ++x) {
          const float r = static_cast<float>(src[x * 3 + 0]);
          const float g = static_cast<float>(src[x * 3 + 1]);
          const float b = static_cast<float>(src[x * 3 + 2]);
          const size_t p = static_cast<size_t>(y) * kRetroFrameWidth +
                           static_cast<size_t>(x);
          if (mode == 1) {
            obs_row[p * 3 + 0] = r;
            obs_row[p * 3 + 1] = g;
            obs_row[p * 3 + 2] = b;
          } else {
            obs_row[p] = 0.299f * r + 0.587f * g + 0.114f * b;
          }
        }
      }
      break;
    }
    case 3: {
      if (game == RetroGame::kSonic) {
        const SonicVars v = retro_read_sonic(core);
        obs_row[0] = static_cast<float>(v.x) * kSonicCoordScale;
        obs_row[1] = static_cast<float>(v.y) * kSonicCoordScale;
        obs_row[2] = static_cast<float>(v.x_vel) * kSonicCoordScale;
        obs_row[3] = static_cast<float>(v.y_vel) * kSonicCoordScale;
        obs_row[4] = static_cast<float>(v.ground_speed) * kSonicCoordScale;
        obs_row[5] =
            static_cast<float>(v.zone * 3 + v.act) * kSonicProgressScale;
      } else {
        const AirstrikerVars v = retro_read_airstriker(core);
        obs_row[0] = static_cast<float>(v.score);
        obs_row[1] = static_cast<float>(v.lives);
        obs_row[2] = static_cast<float>(v.gameover);
      }
      break;
    }
    default: {
      // Raw buffer bytes (native word order — a fixed permutation, so L2
      // distances are unchanged).
      const uint8_t* ram = core.work_ram();
      for (size_t k = 0; k < 0x10000; ++k) {
        obs_row[k] = static_cast<float>(ram[k]);
      }
      break;
    }
  }
}

/// Run `dt` frames with the given action, updating the carry. Returns the
/// summed reward; sets done and the showcase display score.
///   Airstriker (scenario.json): reward = score delta; done = gameover==1
///   && lives==0; display = score. The gun fires on B press EDGES, so the
///   fire bit toggles across dt frames (autofire) — still a pure function
///   of (state, action, dt).
///   Sonic (contest-style, mirrors the Mario port's design): reward =
///   per-frame x-progress with a +-32px glitch guard (suppressed while a
///   boss object is loaded — boss arenas lock the camera, so rightward
///   drift is noise there; damaging the boss pays kSonicBossHitBonus per
///   hit point removed instead), plus delta-shaped
///   bonuses (see the kSonic* coefficients): a one-time act-completion
///   bonus paid outside the clip (Mario flagpole-style), a one-time
///   exploration bonus per newly visited map cell (kSonicCellBonus — pays
///   for the backtracking detours acts like Marble 1 require), signed ring
///   deltas (a hit dumps all rings -> proportional damage penalty, which
///   also makes shields/invincibility instrumentally valuable), a small
///   score-delta term (boss hits, badniks, monitors, end-of-act tally),
///   and a positive-only lives delta (1-ups). Ring loss at an act
///   transition is resynced, not penalized (rings reset to 0 by design).
///   Done on any life lost; display = zone/act/x. Holding B means a
///   higher jump, so B stays held within the step — but a jump needs a
///   press EDGE, so when the previous step already ended with B held, B
///   is released for this step's first frame; consecutive B draws then
///   chain jumps instead of merging into one long hold. (The one-frame
///   release slightly caps an in-flight ascent — acceptable; the old
///   behavior wasted the whole draw.)
inline float retro_step_frames(RetroGame game, RetroCore& core, int32_t action,
                               int32_t dt, RetroCarry& carry, bool& done,
                               float& display) {
  const bool sonic = game == RetroGame::kSonic;
  const uint32_t buttons = (sonic ? kSonicActionMasks : kRetroActionMasks)[
      action >= 0 && action < kRetroNumActions ? action : 0];
  const bool b_was_held = carry.b_held_last != 0;
  // What the step's FINAL frame holds: with dt == 1 the edge-release frame
  // is the whole step, so B ends up not held despite the B action.
  carry.b_held_last =
      (sonic && (buttons & kPadB) && !(dt == 1 && b_was_held)) ? 1 : 0;
  float total_reward = 0.0f;
  done = false;
  display = 0.0f;

  for (int32_t f = 0; f < dt; ++f) {
    uint32_t frame_buttons = buttons;
    if (sonic) {
      if (f == 0 && b_was_held) frame_buttons &= ~uint32_t(kPadB);
    } else {
      if ((buttons & kPadB) && (f % 2)) frame_buttons &= ~uint32_t(kPadB);
    }
    core.run_frame(frame_buttons);

    if (game == RetroGame::kSonic) {
      const SonicVars v = retro_read_sonic(core);
      const int32_t boss_hits = retro_sonic_boss_hits(core);
      const bool boss_loaded = boss_hits >= 0;

      int32_t dx = v.x - carry.x_last;
      carry.x_last = v.x;
      if (dx < -32 || dx > 32) dx = 0;
      if (!boss_loaded) total_reward += static_cast<float>(dx);

      // Boss damage: pay per hit point removed. Skip the frame the boss
      // spawns (last == -1) so its initial 8 HP isn't misread as a delta.
      if (boss_loaded && carry.boss_hits_last > boss_hits &&
          carry.boss_hits_last >= 0) {
        total_reward += kSonicBossHitBonus *
                        static_cast<float>(carry.boss_hits_last - boss_hits);
      }
      carry.boss_hits_last = boss_hits;

      const int32_t progress = v.zone * 3 + v.act;
      const bool act_changed = progress != carry.progress_last;
      if (progress > carry.progress_last) total_reward += kSonicActBonus;
      carry.progress_last = progress;

      // Exploration: pay once per newly visited map cell (see
      // kSonicCellBonus). On act change the mask resets and the spawn cell
      // is marked silently — arriving somewhere new via the signpost is
      // paid by kSonicActBonus, not double-counted as discovery.
      if (act_changed) {
        std::memset(carry.visited, 0, sizeof(carry.visited));
        retro_sonic_visit_cell(carry, v.x, v.y);
      } else if (retro_sonic_visit_cell(carry, v.x, v.y)) {
        total_reward += kSonicCellBonus;
      }

      // Rings reset to 0 when the next act loads — resync silently there;
      // everywhere else the signed delta pays collection / punishes hits.
      if (!act_changed) {
        total_reward +=
            kSonicRingCoef * static_cast<float>(v.rings - carry.rings_last);
      }
      carry.rings_last = v.rings;

      total_reward +=
          kSonicScoreCoef * static_cast<float>(v.score - carry.score_last);
      carry.score_last = v.score;

      if (v.lives > carry.lives_last) {
        total_reward +=
            kSonicLifeBonus * static_cast<float>(v.lives - carry.lives_last);
      }
      if (v.lives < carry.lives_last || v.lives == 0) done = true;
      carry.lives_last = v.lives;

      display = static_cast<float>(v.zone) * 1000000.0f +
                static_cast<float>(v.act) * 100000.0f +
                static_cast<float>(v.x);
      if (done) break;
    } else {
      const AirstrikerVars v = retro_read_airstriker(core);
      total_reward += static_cast<float>(v.score - carry.score_last);
      carry.score_last = v.score;
      display = static_cast<float>(v.score);
      if (v.gameover == 1 && v.lives == 0) {
        done = true;
        break;  // plangym stops frame-skipping when the episode terminates
      }
    }
  }
  return total_reward;
}

/// Initialize the carry from the core's current (post-boot) state.
inline RetroCarry retro_init_carry(RetroGame game, const RetroCore& core) {
  RetroCarry carry;
  if (game == RetroGame::kSonic) {
    const SonicVars v = retro_read_sonic(core);
    carry.score_last = v.score;
    carry.x_last = v.x;
    carry.lives_last = v.lives;
    carry.rings_last = v.rings;
    carry.progress_last = v.zone * 3 + v.act;
    carry.boss_hits_last = retro_sonic_boss_hits(core);
    retro_sonic_visit_cell(carry, v.x, v.y);  // spawn cell is not a discovery
  } else {
    carry.score_last = retro_read_airstriker(core).score;
  }
  return carry;
}

/// Single logical byte write into the pairwise-swapped work RAM buffer.
inline void retro_write_u8(uint8_t* ram, size_t offset, uint8_t value) {
  ram[offset ^ 1] = value;
}

/// Boot from power-on to active gameplay: alternate START and B taps
/// (Sonic's title wants START; Airstriker needs START then B at its menu),
/// then idle and check the game is running and unpaused.
///   Sonic: the in-level lifetime-timer frame counter (RAM 0xFFFE25)
///   advances every frame only during unpaused gameplay, with lives == 3.
///   Airstriker: lives == 3 once a game has started.
///
/// Sonic level selection (the Mario write_stage pattern): pressing START at
/// the title makes the game write GHZ1 into v_zone/v_act ($FFFE10/$FFFE11)
/// and load that level a few frames later. Poking the target zone/act into
/// those bytes EVERY frame until gameplay is detected wins the race, so the
/// loader picks up our level instead; once the level is running the pokes
/// stop (and until then they simply restate the loaded level's own values).
/// zone is Sonic 1's internal id (GHZ=0 LZ=1 MZ=2 SLZ=3 SYZ=4 SBZ=5), act
/// is 0-based.
/// Returns false when gameplay was never reached (unexpected ROM).
inline bool retro_boot_to_gameplay(RetroGame game, RetroCore& core,
                                   int32_t sonic_zone = 0,
                                   int32_t sonic_act = 0) {
  constexpr int kAttempts = 90;  // ~1.5 emulated minutes worst case
  const bool poke_level =
      game == RetroGame::kSonic && (sonic_zone != 0 || sonic_act != 0);
  const auto poke = [&]() {
    if (!poke_level) return;
    uint8_t* ram = core.work_ram_mut();
    retro_write_u8(ram, 0xFE10, static_cast<uint8_t>(sonic_zone));
    retro_write_u8(ram, 0xFE11, static_cast<uint8_t>(sonic_act));
  };
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    const uint32_t tap = (attempt % 2 == 0) ? kPadStart : kPadB;
    core.run_frame(tap);
    poke();
    core.run_frame(tap);
    poke();
    core.run_frame(0);
    poke();
    const uint8_t timer_before = retro_read_u8(core.work_ram(), 0xFE25);
    for (int i = 0; i < 55; ++i) {
      core.run_frame(0);
      poke();
    }

    bool in_gameplay = false;
    if (game == RetroGame::kSonic) {
      const SonicVars v = retro_read_sonic(core);
      const uint8_t timer_after = retro_read_u8(core.work_ram(), 0xFE25);
      in_gameplay = v.lives == 3 && timer_after != timer_before;
    } else {
      in_gameplay = retro_read_airstriker(core).lives == 3;
    }
    if (in_gameplay) {
      // Settle past any level-intro card with no buttons held.
      for (int i = 0; i < 120; ++i) core.run_frame(0);
      return true;
    }
  }
  return false;
}

/// RGBA conversion of the core's current frame (kRetroFrameWidth x
/// kRetroFrameHeight x 4 bytes).
inline void retro_frame_rgba(const RetroCore& core, uint8_t* rgba) {
  const auto& rgb = core.frame_rgb();
  const size_t n_pixels = static_cast<size_t>(kRetroFrameWidth) *
                          static_cast<size_t>(kRetroFrameHeight);
  std::memset(rgba, 0, n_pixels * 4);
  const int32_t w = core.frame_width() < kRetroFrameWidth ? core.frame_width()
                                                          : kRetroFrameWidth;
  const int32_t h = core.frame_height() < kRetroFrameHeight
                        ? core.frame_height()
                        : kRetroFrameHeight;
  for (int32_t y = 0; y < h; ++y) {
    const uint8_t* src =
        rgb.empty() ? nullptr
                    : rgb.data() + static_cast<size_t>(y) * core.frame_width() * 3;
    for (int32_t x = 0; x < w; ++x) {
      const size_t p =
          static_cast<size_t>(y) * kRetroFrameWidth + static_cast<size_t>(x);
      if (src) {
        rgba[p * 4 + 0] = src[x * 3 + 0];
        rgba[p * 4 + 1] = src[x * 3 + 1];
        rgba[p * 4 + 2] = src[x * 3 + 2];
      }
      rgba[p * 4 + 3] = 0xFF;
    }
  }
  for (size_t p = 0; p < n_pixels; ++p) rgba[p * 4 + 3] = 0xFF;
}

}  // namespace fg

#endif  // FRACTAL_GAS_RETRO_GAME_LOGIC_HPP
