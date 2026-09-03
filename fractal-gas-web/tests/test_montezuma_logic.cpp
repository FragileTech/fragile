// Montezuma's Revenge game logic (src/montezuma_logic.hpp): pyramid layout,
// RAM decoding, screen projection, coords tuple and the shaped reward. Pure
// over a synthetic RAM image — no ROM needed.
#include <cstdint>
#include <vector>

#include "montezuma_logic.hpp"
#include "test_framework.hpp"

using namespace fg;

namespace {

// Synthetic 128-byte Atari RAM with Panama Joe at (room, x, y).
std::vector<uint8_t> fake_ram(int room, int x, int y, int level = 0,
                              int lives = 5, int inventory = 0, int dying = 0) {
  std::vector<uint8_t> ram(128, 0);
  ram[kMontezumaRamRoom] = static_cast<uint8_t>(room);
  ram[kMontezumaRamX] = static_cast<uint8_t>(x);
  ram[kMontezumaRamY] = static_cast<uint8_t>(y);
  ram[kMontezumaRamLevel] = static_cast<uint8_t>(level);
  ram[kMontezumaRamLives] = static_cast<uint8_t>(lives);
  ram[kMontezumaRamInventory] = static_cast<uint8_t>(inventory);
  ram[kMontezumaRamDying] = static_cast<uint8_t>(dying);
  return ram;
}

}  // namespace

TEST_CASE(montezuma_pyramid_is_a_bijection_over_24_rooms) {
  // Every room 0..23 sits in exactly one cell, and every non-empty cell
  // holds a room in that range.
  int cells = 0;
  for (int r = 0; r < kMontezumaPyramidRows; ++r) {
    for (int c = 0; c < kMontezumaPyramidCols; ++c) {
      const int room = kMontezumaPyramid[r][c];
      if (room < 0) continue;
      ++cells;
      CHECK(room >= 0 && room < kMontezumaRooms);
      CHECK(montezuma_room_col(room) == c);
      CHECK(montezuma_room_row(room) == r);
    }
  }
  CHECK(cells == kMontezumaRooms);
  // The well-known positions: start room 1 is top-centre, room 15 is the
  // bottom-left corner, room 23 the bottom-right one.
  CHECK(montezuma_room_col(1) == 4 && montezuma_room_row(1) == 0);
  CHECK(montezuma_room_col(15) == 0 && montezuma_room_row(15) == 3);
  CHECK(montezuma_room_col(23) == 8 && montezuma_room_row(23) == 3);
  // Rooms outside the level-1 pyramid are unknown.
  CHECK(montezuma_room_col(24) == -1 && montezuma_room_row(24) == -1);
  CHECK(montezuma_room_col(-1) == -1);
}

TEST_CASE(montezuma_read_decodes_ram) {
  const std::vector<uint8_t> ram = fake_ram(7, 77, 235, 2, 4, 0x0E, 3);
  const MontezumaVars v = montezuma_read(ram.data());
  CHECK(v.room == 7);
  CHECK(v.x == 77);
  CHECK(v.y == 235);
  CHECK(v.level == 2);
  CHECK(v.lives == 4);
  CHECK(v.inventory == 0x0E);
  CHECK(v.dying == 3);
}

TEST_CASE(montezuma_screen_projection_matches_calibration) {
  // Measured against the face pixels on the bundled ROM: start platform
  // (y=235) is row ~28 of the room image, the floor (y=158) row ~105.
  CHECK(montezuma_room_py(235) == 28);
  CHECK(montezuma_room_py(158) == 105);
  CHECK(montezuma_room_px(77) == 80);
  // Clamped to the room image.
  CHECK(montezuma_room_py(255) == 8);
  CHECK(montezuma_room_py(300) == 0);
  CHECK(montezuma_room_py(0) == kMontezumaRoomH - 1);
  CHECK(montezuma_room_px(200) == kMontezumaRoomW - 1);
}

TEST_CASE(montezuma_coords_tuple_layout) {
  const std::vector<uint8_t> ram = fake_ram(1, 77, 235, 0, 5, 0);
  const MontezumaVars v = montezuma_read(ram.data());
  float coords[kMontezumaCoordsDim] = {};
  montezuma_fill_coords(v, coords);
  CHECK(kMontezumaCoordsDim == 8);
  // Room 1 is column 4, row 0: global = cell * 160 + in-room pixel.
  CHECK(coords[0] == 4 * 160 + 80);
  CHECK(coords[1] == 0 * 160 + 28);
  CHECK(coords[2] == 1);
  CHECK(coords[3] == 77);
  CHECK(coords[4] == 235);
  CHECK(coords[5] == 0);
  CHECK(coords[6] == 0);
  CHECK(coords[7] == 5);

  // Adjacent rooms are a room-width apart: room 0 is one column left.
  const std::vector<uint8_t> ram0 = fake_ram(0, 77, 235);
  float coords0[kMontezumaCoordsDim] = {};
  montezuma_fill_coords(montezuma_read(ram0.data()), coords0);
  CHECK(coords[0] - coords0[0] == 160);
  CHECK(coords[1] == coords0[1]);
  // One row down (room 5 sits under room 1).
  const std::vector<uint8_t> ram5 = fake_ram(5, 77, 235);
  float coords5[kMontezumaCoordsDim] = {};
  montezuma_fill_coords(montezuma_read(ram5.data()), coords5);
  CHECK(coords5[0] == coords[0]);
  CHECK(coords5[1] - coords[1] == 160);
}

TEST_CASE(montezuma_room_bonus_paid_once_per_lineage) {
  MontezumaRewardWeights w;
  w.score = 1.0f;
  w.room = 500.0f;
  AtariCarry carry;
  carry.level_last = 0;
  carry.visited_rooms = 1u << 1;  // start room already entered (reset)

  // Standing in the start room: only the (zero) score.
  MontezumaVars v = montezuma_read(fake_ram(1, 77, 235).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == 0.0f);
  // A key: ALE score delta scaled by the score weight.
  w.score = 2.0f;
  CHECK(montezuma_step_reward(v, 100.0f, carry, w) == 200.0f);
  w.score = 1.0f;

  // Entering room 0 pays the bonus once; the next step in it does not.
  v = montezuma_read(fake_ram(0, 150, 200).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == 500.0f);
  CHECK((carry.visited_rooms & (1u << 0)) != 0);
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == 0.0f);
  // Back to room 1: already visited, nothing.
  v = montezuma_read(fake_ram(1, 10, 200).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == 0.0f);
}

TEST_CASE(montezuma_room_bonus_skips_dying_and_resets_on_level_change) {
  MontezumaRewardWeights w;
  AtariCarry carry;
  // Dying inside a new room: no bonus, and the room is NOT marked, so the
  // respawned lineage can still earn it later.
  MontezumaVars v = montezuma_read(fake_ram(4, 50, 200, 0, 4, 0, 6).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == 0.0f);
  CHECK(carry.visited_rooms == 0);
  v = montezuma_read(fake_ram(4, 50, 200, 0, 4, 0, 0).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == w.room);

  // Reaching the next level clears the bitmask: the same room number pays
  // again, and level_last follows the RAM.
  v = montezuma_read(fake_ram(4, 50, 200, 1).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == w.room);
  CHECK(carry.level_last == 1);
  CHECK(carry.visited_rooms == (1u << 4));

  // Rooms outside the pyramid never pay (and never overflow the mask).
  v = montezuma_read(fake_ram(30, 50, 200, 1).data());
  CHECK(montezuma_step_reward(v, 0.0f, carry, w) == 0.0f);
  CHECK(carry.visited_rooms == (1u << 4));
}
