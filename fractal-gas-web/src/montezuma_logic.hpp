// Montezuma's Revenge (Atari 2600) game logic shared by AtariEnv and the
// tests: RAM decoding, the level-1 pyramid layout, the screen projection of
// Panama Joe, the Coords observation tuple and the shaped reward.
//
// Header-only and pure over a 128-byte RAM image so it is unit-testable
// without a ROM (same pattern as retro_game_logic.hpp for Sonic).
//
// RAM map (indices into ALE's ale::ALERAM::array(), i.e. 0x80-based):
//   ram[3]   room number (0..23 on level 1; the pyramid below)
//   ram[42]  player x (0..~152, in screen pixels)
//   ram[43]  player y (grows UPWARD: 235 on the start platform, 148 at the
//            bottom of a room)
//   ram[55]  death timer, non-zero during the dying animation
//   ram[57]  level (0-based)
//   ram[58]  lives (ALE reads 0xBA; drops at the START of the death
//            animation, so ALE's life-loss done already fires immediately)
//   ram[65]  inventory bitmask (keys, torch, sword, hammer)
// Screen projection was calibrated against Panama Joe's face pixels (red
// channel == 228) on the bundled ROM: face_x ~ x + 3, face_row ~ 313 - y in
// the 210-row frame, i.e. 263 - y inside the 160-row room crop below the
// 50-row HUD.
#ifndef FRACTAL_GAS_MONTEZUMA_LOGIC_HPP
#define FRACTAL_GAS_MONTEZUMA_LOGIC_HPP

#include <cstdint>

namespace fg {

// ---- RAM offsets ----------------------------------------------------------
constexpr int kMontezumaRamRoom = 3;
constexpr int kMontezumaRamX = 42;
constexpr int kMontezumaRamY = 43;
constexpr int kMontezumaRamDying = 55;
constexpr int kMontezumaRamLevel = 57;
constexpr int kMontezumaRamLives = 58;
constexpr int kMontezumaRamInventory = 65;

// ---- Pyramid layout (level 1 of the temple: 24 rooms on a 9x4 grid) ------
constexpr int kMontezumaPyramidRows = 4;
constexpr int kMontezumaPyramidCols = 9;
constexpr int kMontezumaRooms = 24;
constexpr int32_t kMontezumaRoomW = 160;  // room image width (frame width)
constexpr int32_t kMontezumaRoomH = 160;  // frame rows 50..209 (HUD cropped)
constexpr int32_t kMontezumaHudRows = 50;

/// Room 8 (level 1, left of room 9) is a dead end the swarm must not spend
/// its budget in: entering it counts as a death, like plangym's
/// death_room_8 option in the old Montezuma demo.
constexpr int32_t kMontezumaDeathRoom = 8;

inline bool montezuma_in_death_room(int32_t room) {
  return room == kMontezumaDeathRoom;
}

constexpr int kMontezumaPyramid[kMontezumaPyramidRows][kMontezumaPyramidCols] = {
    {-1, -1, -1, 0, 1, 2, -1, -1, -1},
    {-1, -1, 3, 4, 5, 6, 7, -1, -1},
    {-1, 8, 9, 10, 11, 12, 13, 14, -1},
    {15, 16, 17, 18, 19, 20, 21, 22, 23},
};

/// Grid column / row of a room, or -1 for rooms outside the level-1 pyramid.
inline int montezuma_room_col(int room) {
  if (room < 0) return -1;  // never match the -1 holes of the table
  for (int r = 0; r < kMontezumaPyramidRows; ++r) {
    for (int c = 0; c < kMontezumaPyramidCols; ++c) {
      if (kMontezumaPyramid[r][c] == room) return c;
    }
  }
  return -1;
}

inline int montezuma_room_row(int room) {
  if (room < 0) return -1;
  for (int r = 0; r < kMontezumaPyramidRows; ++r) {
    for (int c = 0; c < kMontezumaPyramidCols; ++c) {
      if (kMontezumaPyramid[r][c] == room) return r;
    }
  }
  return -1;
}

// ---- RAM decoding ---------------------------------------------------------
struct MontezumaVars {
  int32_t room = 0;
  int32_t x = 0;
  int32_t y = 0;
  int32_t level = 0;
  int32_t lives = 0;
  int32_t inventory = 0;
  int32_t dying = 0;
};

inline MontezumaVars montezuma_read(const uint8_t* ram) {
  MontezumaVars v;
  v.room = ram[kMontezumaRamRoom];
  v.x = ram[kMontezumaRamX];
  v.y = ram[kMontezumaRamY];
  v.level = ram[kMontezumaRamLevel];
  v.lives = ram[kMontezumaRamLives];
  v.inventory = ram[kMontezumaRamInventory];
  v.dying = ram[kMontezumaRamDying];
  return v;
}

// ---- Screen projection ----------------------------------------------------
/// Pixel column of Panama Joe's face inside a room image.
inline int32_t montezuma_room_px(int32_t x) {
  const int32_t px = x + 3;
  return px < 0 ? 0 : (px >= kMontezumaRoomW ? kMontezumaRoomW - 1 : px);
}

/// Pixel row of Panama Joe's face inside the HUD-cropped room image.
inline int32_t montezuma_room_py(int32_t y) {
  const int32_t py = 263 - y;
  return py < 0 ? 0 : (py >= kMontezumaRoomH ? kMontezumaRoomH - 1 : py);
}

/// Pyramid-global pixel coordinates (room cell * room size + in-room pixel),
/// so that adjacent rooms are a room-width apart in observation space.
/// Rooms outside the pyramid are parked beyond its right edge.
inline int32_t montezuma_global_x(const MontezumaVars& v) {
  const int col = montezuma_room_col(v.room);
  const int32_t c = col < 0 ? kMontezumaPyramidCols : col;
  return c * kMontezumaRoomW + montezuma_room_px(v.x);
}

inline int32_t montezuma_global_y(const MontezumaVars& v) {
  const int row = montezuma_room_row(v.room);
  const int32_t r = row < 0 ? 0 : row;
  return r * kMontezumaRoomH + montezuma_room_py(v.y);
}

// ---- Coords observation ---------------------------------------------------
/// Layout: [global_x, global_y, room, x, y, level, inventory, lives]. Raw
/// values (no per-dimension normalization, like the Mario/Sonic tuples).
constexpr int32_t kMontezumaCoordsDim = 8;

inline void montezuma_fill_coords(const MontezumaVars& v, float* out) {
  out[0] = static_cast<float>(montezuma_global_x(v));
  out[1] = static_cast<float>(montezuma_global_y(v));
  out[2] = static_cast<float>(v.room);
  out[3] = static_cast<float>(v.x);
  out[4] = static_cast<float>(v.y);
  out[5] = static_cast<float>(v.level);
  out[6] = static_cast<float>(v.inventory);
  out[7] = static_cast<float>(v.lives);
}

// ---- Reward ---------------------------------------------------------------
/// Live-tunable weights (web demo sliders, field order = JS array order).
struct MontezumaRewardWeights {
  float score = 1.0f;   // multiplier on ALE's score delta
  float room = 500.0f;  // one-off bonus per room new to this walker's lineage
};

/// Per-walker carry appended to every Atari state blob (clones inherit it).
/// Generic games only use episode_return.
struct AtariCarry {
  float episode_return = 0.0f;   // cumulative shaped reward since reset
  uint32_t visited_rooms = 0;    // bit r set = room r entered on level_last
  int32_t level_last = 0;        // level the bitmask belongs to
};

/// Shaped step reward: score delta plus a bonus the first time this lineage
/// enters a pyramid room on the current level. Not paid during the death
/// animation (the respawn does not "enter" the room again), nor for the
/// death room (entering it is a death), and the bitmask resets when the
/// level changes.
inline float montezuma_step_reward(const MontezumaVars& v, float ale_reward,
                                   AtariCarry& carry,
                                   const MontezumaRewardWeights& w) {
  float reward = ale_reward * w.score;
  if (v.level != carry.level_last) {
    carry.level_last = v.level;
    carry.visited_rooms = 0;
  }
  if (v.dying == 0 && v.room >= 0 && v.room < kMontezumaRooms &&
      !montezuma_in_death_room(v.room)) {
    const uint32_t bit = 1u << static_cast<uint32_t>(v.room);
    if ((carry.visited_rooms & bit) == 0) {
      carry.visited_rooms |= bit;
      reward += w.room;
    }
  }
  return reward;
}

}  // namespace fg

#endif  // FRACTAL_GAS_MONTEZUMA_LOGIC_HPP
