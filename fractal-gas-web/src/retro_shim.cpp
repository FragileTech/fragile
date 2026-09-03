// Core-worker shim: a standalone wasm module holding ONE statically linked
// Genesis Plus GX core plus the shared per-game logic. Each Web Worker in
// the browser instantiates its own copy of this module (own memory, own
// globals), giving the same isolation the native build gets from
// dlopen-per-file-copy — without emscripten dynamic linking.
//
// The worker JS (web/core-worker.js) copies job data between the main
// module's shared memory and this module's heap; all pointers below are in
// THIS module's heap. Built with FG_RETRO_STATIC (retro_core.cpp's
// direct-symbol mode).
#ifdef __EMSCRIPTEN__

#include <emscripten/emscripten.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "retro_core.hpp"
#include "retro_game_logic.hpp"

namespace {

std::unique_ptr<fg::RetroCore> g_core;
fg::RetroGame g_game = fg::RetroGame::kAirstriker;
int32_t g_obs_mode = 0;
int32_t g_zone = 0;  // Sonic start level (internal zone id / 0-based act)
int32_t g_act = 0;
size_t g_serialize_size = 0;
fg::SonicRewardWeights g_weights;
std::string g_error;

}  // namespace

extern "C" {

/// Load the ROM (bytes in this module's heap) and create the core.
/// Returns the walker-blob size (serialize size + carry) or -1 on error.
EMSCRIPTEN_KEEPALIVE
int shim_init(const uint8_t* rom, int rom_len, int game, int obs_mode,
              int zone, int act) {
  try {
    g_core.reset();
    {
      std::ofstream f("/rom.md", std::ios::binary);
      f.write(reinterpret_cast<const char*>(rom), rom_len);
    }
    g_game = game == 1 ? fg::RetroGame::kSonic : fg::RetroGame::kAirstriker;
    g_obs_mode = obs_mode;
    g_zone = zone;
    g_act = act;
    g_core = std::make_unique<fg::RetroCore>("");
    g_core->load_rom("/rom.md");
    g_serialize_size = g_core->serialize_size();
    if (g_serialize_size == 0) throw std::runtime_error("zero serialize size");
    return static_cast<int>(g_serialize_size + fg::kRetroCarryBytes);
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

EMSCRIPTEN_KEEPALIVE
int shim_obs_dim() {
  return static_cast<int>(fg::retro_obs_dim(g_game, g_obs_mode));
}

/// Boot from power-on to gameplay; writes the initial walker blob and
/// observation. Returns 0, or -1 on failure.
EMSCRIPTEN_KEEPALIVE
int shim_boot(uint8_t* blob_out, float* obs_out) {
  try {
    if (!g_core) throw std::runtime_error("shim not initialized");
    if (!fg::retro_boot_to_gameplay(g_game, *g_core, g_zone, g_act)) {
      throw std::runtime_error("could not reach gameplay from boot");
    }
    g_core->run_frame(0);  // framebuffer belongs to the serialized state
    const fg::RetroCarry carry = fg::retro_init_carry(g_game, *g_core);
    if (!g_core->serialize(blob_out, g_serialize_size)) {
      throw std::runtime_error("retro_serialize failed");
    }
    std::memcpy(blob_out + g_serialize_size, &carry, fg::kRetroCarryBytes);
    fg::retro_fill_obs(g_game, g_obs_mode, *g_core, obs_out);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

/// Box-downsample the core's RGB frame 8x to 40x28 RGB (fog-of-war tile).
static void downsample_tile(const fg::RetroCore& core, uint8_t* tile_out) {
  constexpr int kTw = 40, kTh = 28, kF = 8;
  const auto& rgb = core.frame_rgb();
  const int fw = core.frame_width();
  const int fh = core.frame_height();
  std::memset(tile_out, 0, kTw * kTh * 3);
  if (rgb.empty()) return;
  for (int ty = 0; ty < kTh; ++ty) {
    for (int tx = 0; tx < kTw; ++tx) {
      int r = 0, g = 0, b = 0, cnt = 0;
      for (int dy = 0; dy < kF; ++dy) {
        const int sy = ty * kF + dy;
        if (sy >= fh) break;
        const uint8_t* row = rgb.data() + static_cast<size_t>(sy) * fw * 3;
        for (int dx = 0; dx < kF; ++dx) {
          const int sx = tx * kF + dx;
          if (sx >= fw) break;
          r += row[sx * 3 + 0];
          g += row[sx * 3 + 1];
          b += row[sx * 3 + 2];
          ++cnt;
        }
      }
      uint8_t* out = tile_out + (ty * kTw + tx) * 3;
      if (cnt) {
        out[0] = static_cast<uint8_t>(r / cnt);
        out[1] = static_cast<uint8_t>(g / cnt);
        out[2] = static_cast<uint8_t>(b / cnt);
      }
    }
  }
}

/// One walker step: restore blob, run dt frames with the action, write the
/// new blob back in place plus obs/reward/done/display. pos_out (6 ints:
/// x, y, zone, act, cam_x, cam_y) and tile_out (40x28 RGB = 3360 bytes,
/// the frame downsampled 8x) feed the fog-of-war swarm map (zeros for
/// Airstriker positions). Returns 0 or -1.
EMSCRIPTEN_KEEPALIVE
int shim_step(uint8_t* blob_inout, int action, int dt, float* obs_out,
              float* reward_out, int* done_out, float* display_out,
              int* pos_out, uint8_t* tile_out) {
  try {
    if (!g_core->unserialize(blob_inout, g_serialize_size)) {
      throw std::runtime_error("retro_unserialize failed");
    }
    fg::RetroCarry carry;
    std::memcpy(&carry, blob_inout + g_serialize_size, fg::kRetroCarryBytes);

    bool done = false;
    float display = 0.0f;
    const float reward = fg::retro_step_frames(g_game, *g_core, action, dt,
                                               carry, done, display,
                                               g_weights);

    fg::retro_fill_obs(g_game, g_obs_mode, *g_core, obs_out);
    if (!g_core->serialize(blob_inout, g_serialize_size)) {
      throw std::runtime_error("retro_serialize failed");
    }
    std::memcpy(blob_inout + g_serialize_size, &carry, fg::kRetroCarryBytes);
    *reward_out = reward;
    *done_out = done ? 1 : 0;
    *display_out = display;
    if (g_game == fg::RetroGame::kSonic) {
      const fg::SonicVars v = fg::retro_read_sonic(*g_core);
      pos_out[0] = v.x;
      pos_out[1] = v.y;
      pos_out[2] = v.zone;
      pos_out[3] = v.act;
      pos_out[4] = v.cam_x;
      pos_out[5] = v.cam_y;
    } else {
      std::memset(pos_out, 0, 6 * sizeof(int));
    }
    downsample_tile(*g_core, tile_out);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

/// Live-tunable Sonic reward term weights (see SonicRewardWeights; same
/// field order). Applies to every subsequent shim_step.
EMSCRIPTEN_KEEPALIVE
void shim_set_sonic_weights(float dx, float rings, float score, float cell,
                            float life, float boss, float act) {
  g_weights.dx = dx;
  g_weights.rings = rings;
  g_weights.score = score;
  g_weights.cell = cell;
  g_weights.life = life;
  g_weights.boss = boss;
  g_weights.act = act;
}

/// RGBA frame (320x224x4) of the given blob's state. Display-only.
EMSCRIPTEN_KEEPALIVE
int shim_render(const uint8_t* blob, uint8_t* rgba_out) {
  try {
    if (!g_core->unserialize(blob, g_serialize_size)) {
      throw std::runtime_error("retro_unserialize failed");
    }
    g_core->run_frame(0);  // the core only redraws on retro_run
    fg::retro_frame_rgba(*g_core, rgba_out);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

EMSCRIPTEN_KEEPALIVE
const char* shim_error() { return g_error.c_str(); }

}  // extern "C"

#endif  // __EMSCRIPTEN__
