#include "nes_env.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "emulator.hpp"

namespace fg {

const uint8_t kMarioActionMasks[12] = {
    0x00,         // NOOP
    0x80,         // right
    0x80 | 0x01,  // right + A
    0x80 | 0x02,  // right + B
    0x80 | 0x03,  // right + A + B
    0x01,         // A
    0x40,         // left
    0x40 | 0x01,  // left + A
    0x40 | 0x02,  // left + B
    0x40 | 0x03,  // left + A + B
    0x20,         // down
    0x10,         // up
};

namespace {

constexpr size_t kCarryBytes = sizeof(MarioCarry);
static_assert(sizeof(MarioCarry) == 9 * sizeof(int32_t),
              "MarioCarry must be trivially copyable with no padding");
constexpr uint8_t kStartButton = 0x08;  // smb_env.py _frame_advance(8)

void validate_rom(const std::string& path) {
  // nes-py's Cartridge::loadFromFile does no validation (an unsupported
  // mapper yields a null Mapper* and a crash), so validate the iNES header
  // before constructing any Emulator.
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open ROM file: " + path);
  uint8_t header[16];
  f.read(reinterpret_cast<char*>(header), 16);
  if (f.gcount() != 16 || header[0] != 'N' || header[1] != 'E' ||
      header[2] != 'S' || header[3] != 0x1A) {
    throw std::runtime_error("Not a valid iNES ROM (bad header magic)");
  }
  const uint8_t mapper = ((header[6] >> 4) & 0x0F) | (header[7] & 0xF0);
  if (mapper > 3) {
    throw std::runtime_error("Unsupported NES mapper " + std::to_string(mapper) +
                             " (nes-py supports NROM/SxROM/UxROM/CNROM = 0-3)");
  }
}

void frame_advance(NES::Emulator& emu, uint8_t buttons) {
  // smb_env.py::_frame_advance: write the action to the controller, step.
  *emu.get_controller(0) = buttons;
  emu.step();
}

/// smb_env.py::_write_stage: poke the target level into RAM while the game
/// is booting so it loads that level directly. The sub-area byte is not the
/// stage: worlds 1, 2, 4 and 7 have an extra intro sub-area, so stages >= 2
/// there map to area = stage + 1.
void write_stage(NES::Emulator& emu, int32_t world, int32_t stage) {
  int32_t area = stage;
  if ((world == 1 || world == 2 || world == 4 || world == 7) && stage >= 2) {
    area = stage + 1;
  }
  uint8_t* ram = emu.get_memory_buffer();
  ram[0x75F] = static_cast<uint8_t>(world - 1);
  ram[0x75C] = static_cast<uint8_t>(stage - 1);
  ram[0x760] = static_cast<uint8_t>(area - 1);
}

/// smb_env.py::_skip_start_screen: press and release start until the game
/// clock starts, run out the pre-level timer, then idle past the last
/// non-ticking frames. A frame cap guards against non-SMB ROMs where the
/// clock RAM addresses never tick (the Python original loops forever there).
/// The target level is written each loop iteration, as gym's single-stage
/// env does.
void skip_start_screen(NES::Emulator& emu, int32_t world, int32_t stage) {
  constexpr int kMaxFrames = 2000;
  int frames = 0;
  auto guard = [&frames]() {
    if (++frames > kMaxFrames) {
      throw std::runtime_error(
          "skip_start_screen: game clock never started ticking - is this a "
          "Super Mario Bros ROM?");
    }
  };

  // press and release the start button
  frame_advance(emu, kStartButton);
  frame_advance(emu, 0);
  // Press start until the game starts
  while (mario_time(emu.get_memory_buffer()) == 0) {
    guard();
    // press and release the start button, writing the target level while
    // the game boots (smb_env.py order: press, write stage, release)
    frame_advance(emu, kStartButton);
    write_stage(emu, world, stage);
    frame_advance(emu, 0);
    // run-out the prelevel timer to skip the animation
    // (smb_env.py::_runout_prelevel_timer: ram[0x07A0] = 0)
    emu.get_memory_buffer()[0x7A0] = 0;
  }
  // after the start screen, idle to skip some extra frames
  int32_t time_last = mario_time(emu.get_memory_buffer());
  while (mario_time(emu.get_memory_buffer()) >= time_last) {
    guard();
    time_last = mario_time(emu.get_memory_buffer());
    frame_advance(emu, 0);
    frame_advance(emu, 0);
  }
}

}  // namespace

NesMarioEnv::NesMarioEnv(std::string rom_path, int n_threads, ObsMode obs_mode,
                         int32_t start_world, int32_t start_stage)
    : rom_path_(std::move(rom_path)),
      obs_mode_(obs_mode),
      start_world_(start_world < 1 ? 1 : (start_world > 8 ? 8 : start_world)),
      start_stage_(start_stage < 1 ? 1 : (start_stage > 4 ? 4 : start_stage)),
      pool_(n_threads) {
  validate_rom(rom_path_);
  emulators_.reserve(static_cast<size_t>(pool_.size()));
  for (int i = 0; i < pool_.size(); ++i) {
    // screen_in_state=true: state blobs carry the frame buffer, so rendering
    // the best walker is a load_state + buffer copy (no re-simulation).
    emulators_.push_back(
        std::make_unique<NES::Emulator>(rom_path_, /*screen_in_state=*/true));
  }
}

NesMarioEnv::~NesMarioEnv() = default;

int32_t NesMarioEnv::n_actions() const {
  return static_cast<int32_t>(sizeof(kMarioActionMasks));
}

int32_t NesMarioEnv::frame_width() const { return NES::Emulator::WIDTH; }
int32_t NesMarioEnv::frame_height() const { return NES::Emulator::HEIGHT; }

int32_t NesMarioEnv::obs_dim() const {
  constexpr int32_t kPixels = 256 * 240;
  switch (obs_mode_) {
    case ObsMode::kRam:
      return 0x800;
    case ObsMode::kRgb:
      return kPixels * 3;
    case ObsMode::kGray:
      return kPixels;
    case ObsMode::kCoords:
      return kCoordsDim;
  }
  return 0x800;
}

void NesMarioEnv::fill_obs(NES::Emulator& emu, float* obs_row) const {
  const uint8_t* ram = emu.get_memory_buffer();
  switch (obs_mode_) {
    case ObsMode::kRam: {
      for (size_t k = 0; k < 0x800; ++k) obs_row[k] = static_cast<float>(ram[k]);
      break;
    }
    case ObsMode::kRgb: {
      const NES::NES_Pixel* screen = emu.get_screen_buffer();
      const size_t n_pixels = 256 * 240;
      for (size_t p = 0; p < n_pixels; ++p) {
        const uint32_t px = screen[p];  // 0x00RRGGBB
        obs_row[p * 3 + 0] = static_cast<float>((px >> 16) & 0xFF);
        obs_row[p * 3 + 1] = static_cast<float>((px >> 8) & 0xFF);
        obs_row[p * 3 + 2] = static_cast<float>(px & 0xFF);
      }
      break;
    }
    case ObsMode::kGray: {
      const NES::NES_Pixel* screen = emu.get_screen_buffer();
      const size_t n_pixels = 256 * 240;
      for (size_t p = 0; p < n_pixels; ++p) {
        const uint32_t px = screen[p];
        const float r = static_cast<float>((px >> 16) & 0xFF);
        const float g = static_cast<float>((px >> 8) & 0xFF);
        const float b = static_cast<float>(px & 0xFF);
        obs_row[p] = 0.299f * r + 0.587f * g + 0.114f * b;
      }
      break;
    }
    case ObsMode::kCoords: {
      // [x, y_pixel, y_viewport, world, stage, time, h_vel, v_vel,
      //  sub_area, power_up] — fast proxies that distinguish game states.
      obs_row[0] = static_cast<float>(mario_x_position(ram));
      obs_row[1] = static_cast<float>(ram[0x3B8]);  // player y pixel
      obs_row[2] = static_cast<float>(ram[0xB5]);   // y viewport
      obs_row[3] = static_cast<float>(ram[0x75F]);  // world
      obs_row[4] = static_cast<float>(ram[0x75C]);  // stage
      obs_row[5] = static_cast<float>(mario_time(ram));
      obs_row[6] = static_cast<float>(static_cast<int8_t>(ram[0x57]));  // h velocity
      obs_row[7] = static_cast<float>(static_cast<int8_t>(ram[0x9F]));  // v velocity
      obs_row[8] = static_cast<float>(ram[0x760]);  // sub-area byte
      obs_row[9] = static_cast<float>(ram[0x756]);  // power-up state
      break;
    }
  }
}

std::vector<char> NesMarioEnv::dump_with_carry(NES::Emulator& emu,
                                               const MarioCarry& carry) {
  const size_t state_size = emu.state_size();
  std::vector<char> blob(state_size + kCarryBytes);
  emu.dump_state(blob.data());
  std::memcpy(blob.data() + state_size, &carry, kCarryBytes);
  return blob;
}

MarioCarry NesMarioEnv::read_carry(const std::vector<char>& blob) {
  MarioCarry carry;
  std::memcpy(&carry, blob.data() + blob.size() - kCarryBytes, kCarryBytes);
  return carry;
}

void NesMarioEnv::reset(std::vector<char>& state, std::vector<float>& obs) {
  if (has_initial_) {
    state = initial_state_;
    obs = initial_obs_;
    return;
  }

  NES::Emulator& emu = *emulators_[0];
  // Cold boot: SMB checks a RAM signature to detect warm boots and then
  // skips parts of the startup sequence, so clear the RAM first.
  std::memset(emu.get_memory_buffer(), 0, 0x800);
  emu.reset();
  skip_start_screen(emu, start_world_, start_stage_);

  const uint8_t* ram = emu.get_memory_buffer();
  MarioCarry carry;
  carry.x_last = mario_x_position(ram);
  carry.time_last = mario_time(ram);

  state = dump_with_carry(emu, carry);
  obs.resize(static_cast<size_t>(obs_dim()));
  fill_obs(emu, obs.data());

  initial_state_ = state;
  initial_obs_ = obs;
  has_initial_ = true;
}

void NesMarioEnv::step_one(int slot, const std::vector<char>& blob,
                           int32_t action, int32_t dt,
                           std::vector<char>& new_blob, float* obs_row,
                           float& reward, uint8_t& done,
                           DisplayInfo& display) {
  NES::Emulator& emu = *emulators_[static_cast<size_t>(slot)];
  emu.load_state(blob.data());
  MarioCarry carry = read_carry(blob);

  *emu.get_controller(0) = kMarioActionMasks[action];
  float total_reward = 0.0f;
  bool is_done = false;
  for (int32_t f = 0; f < dt; ++f) {
    emu.step();
    const MarioFrameResult r =
        mario_frame_update(emu.get_memory_buffer(), carry, reward_weights_);
    total_reward += r.reward;
    if (r.done) {
      is_done = true;
      break;  // plangym stops frame-skipping when the episode terminates
    }
  }

  fill_obs(emu, obs_row);
  new_blob = dump_with_carry(emu, carry);
  reward = total_reward;
  done = is_done ? 1 : 0;

  // Showcase ranking data, read from RAM regardless of observation mode:
  // highest (world, stage) first, then x within the level.
  const uint8_t* ram = emu.get_memory_buffer();
  display.world = ram[0x75F];
  display.stage = ram[0x75C];
  display.x = mario_x_position(ram);
  display.y = ram[0x3B8];  // player y-pixel on screen
  display.score = static_cast<float>(ram[0x75F]) * 1000000.0f +
                  static_cast<float>(ram[0x75C]) * 100000.0f +
                  static_cast<float>(display.x);
}

void NesMarioEnv::step_batch(const std::vector<std::vector<char>>& states,
                             const std::vector<int32_t>& actions,
                             const std::vector<int32_t>& dt,
                             std::vector<std::vector<char>>& new_states,
                             std::vector<float>& observations,
                             std::vector<float>& rewards,
                             std::vector<uint8_t>& dones,
                             std::vector<uint8_t>& truncated) {
  const auto n = static_cast<int32_t>(states.size());
  const auto d = static_cast<size_t>(obs_dim());
  display_cache_.resize(static_cast<size_t>(n));
  float* obs_base = observations.data();
  pool_.parallel_for(n, [&](int32_t i, int slot) {
    const auto ui = static_cast<size_t>(i);
    step_one(slot, states[ui], actions[ui], dt[ui], new_states[ui],
             obs_base + ui * d, rewards[ui], dones[ui], display_cache_[ui]);
    truncated[ui] = 0;  // the NES env never truncates
  });
}

float NesMarioEnv::display_score(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui].score : 0.0f;
}

int32_t NesMarioEnv::walker_world(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui].world + 1 : 1;
}

int32_t NesMarioEnv::walker_stage(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui].stage + 1 : 1;
}

int32_t NesMarioEnv::walker_x(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui].x : 0;
}

int32_t NesMarioEnv::walker_y(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui].y : 0;
}

void NesMarioEnv::render_frame(const std::vector<char>& state,
                               std::vector<uint8_t>& rgba) {
  NES::Emulator& emu = *emulators_[0];
  emu.load_state(state.data());
  const NES::NES_Pixel* screen = emu.get_screen_buffer();
  const size_t n_pixels = static_cast<size_t>(NES::Emulator::WIDTH) *
                          static_cast<size_t>(NES::Emulator::HEIGHT);
  rgba.resize(n_pixels * 4);
  for (size_t p = 0; p < n_pixels; ++p) {
    const uint32_t px = screen[p];  // 0x00RRGGBB
    rgba[p * 4 + 0] = static_cast<uint8_t>((px >> 16) & 0xFF);  // R
    rgba[p * 4 + 1] = static_cast<uint8_t>((px >> 8) & 0xFF);   // G
    rgba[p * 4 + 2] = static_cast<uint8_t>(px & 0xFF);          // B
    rgba[p * 4 + 3] = 0xFF;
  }
}

}  // namespace fg
