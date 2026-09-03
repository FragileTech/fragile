#include "atari_env.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "ale/ale_interface.hpp"

namespace fg {

namespace {

constexpr size_t kAtariRamBytes = 128;   // ALERAM::kRamSize
constexpr size_t kCarryBytes = sizeof(AtariCarry);

void validate_rom_file(const std::string& path) {
  // ALE's loadROM exits the process (std::exit) on a broken setup instead of
  // throwing, so at least catch the trivial "file missing/unreadable" case
  // with a proper exception before constructing any interface.
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open ROM file: " + path);
  char byte;
  if (!f.read(&byte, 1)) {
    throw std::runtime_error("Empty ROM file: " + path);
  }
}

}  // namespace

AtariEnv::AtariEnv(std::string rom_path, int n_threads, AtariObsMode obs_mode,
                   AtariGame game)
    : rom_path_(std::move(rom_path)),
      obs_mode_(obs_mode),
      game_(game),
      pool_(n_threads) {
  validate_rom_file(rom_path_);
  // Silence the per-instance welcome banner and MD5 warnings.
  ale::Logger::setMode(ale::Logger::Error);

  emulators_.reserve(static_cast<size_t>(pool_.size()));
  for (int i = 0; i < pool_.size(); ++i) {
    auto a = std::make_unique<ale::ALEInterface>();
    // Determinism: sticky actions off, fixed seed, no output devices. The
    // seed only affects the freshly loaded state — every step restores a
    // full system state (including the RNG) from the walker blob, so all
    // instances behave identically given the same blob.
    a->setFloat("repeat_action_probability", 0.0f);
    a->setInt("random_seed", 42);
    a->setBool("sound", false);
    a->setBool("display_screen", false);
    a->loadROM(rom_path_);
    emulators_.push_back(std::move(a));
  }

  scratch_.resize(static_cast<size_t>(pool_.size()));
  const ale::ALEScreen& screen = emulators_[0]->getScreen();
  screen_w_ = static_cast<int32_t>(screen.width());
  screen_h_ = static_cast<int32_t>(screen.height());
  for (ale::Action act : emulators_[0]->getMinimalActionSet()) {
    action_set_.push_back(static_cast<int32_t>(act));
  }
}

AtariEnv::~AtariEnv() = default;

int32_t AtariEnv::n_actions() const {
  return static_cast<int32_t>(action_set_.size());
}

int32_t AtariEnv::obs_dim() const {
  switch (obs_mode_) {
    case AtariObsMode::kCoords:
      if (game_ == AtariGame::kMontezuma) return kMontezumaCoordsDim;
      return static_cast<int32_t>(kAtariRamBytes);  // generic: coords == RAM
    case AtariObsMode::kRam:
      return static_cast<int32_t>(kAtariRamBytes);
    case AtariObsMode::kRgb:
      return screen_w_ * screen_h_ * 3;
    case AtariObsMode::kGray:
      return screen_w_ * screen_h_;
  }
  return static_cast<int32_t>(kAtariRamBytes);
}

void AtariEnv::fill_obs(int slot, float* obs_row) {
  ale::ALEInterface& a = *emulators_[static_cast<size_t>(slot)];
  switch (obs_mode_) {
    case AtariObsMode::kCoords:
      if (game_ == AtariGame::kMontezuma) {
        montezuma_fill_coords(montezuma_read(a.getRAM().array()), obs_row);
        break;
      }
      [[fallthrough]];  // generic games: coords == RAM
    case AtariObsMode::kRam: {
      const ale::ALERAM& ram = a.getRAM();
      const unsigned char* bytes = ram.array();
      for (size_t k = 0; k < kAtariRamBytes; ++k) {
        obs_row[k] = static_cast<float>(bytes[k]);
      }
      break;
    }
    case AtariObsMode::kRgb: {
      std::vector<unsigned char>& buf = scratch_[static_cast<size_t>(slot)];
      a.getScreenRGB(buf);  // resizes to w*h*3, interleaved RGB
      const size_t n = static_cast<size_t>(screen_w_) *
                       static_cast<size_t>(screen_h_) * 3;
      for (size_t k = 0; k < n; ++k) obs_row[k] = static_cast<float>(buf[k]);
      break;
    }
    case AtariObsMode::kGray: {
      std::vector<unsigned char>& buf = scratch_[static_cast<size_t>(slot)];
      a.getScreenGrayscale(buf);  // resizes to w*h
      const size_t n =
          static_cast<size_t>(screen_w_) * static_cast<size_t>(screen_h_);
      for (size_t k = 0; k < n; ++k) obs_row[k] = static_cast<float>(buf[k]);
      break;
    }
  }
}

std::vector<char> AtariEnv::blob_from(ale::ALEInterface& a,
                                      const AtariCarry& carry) {
  // cloneSystemState() = full system state INCLUDING the RNG; serialize()
  // yields a portable byte string any instance can restore.
  ale::ALEState state = a.cloneSystemState();
  const std::string ser = state.serialize();
  std::vector<char> blob(ser.size() + kCarryBytes);
  std::memcpy(blob.data(), ser.data(), ser.size());
  std::memcpy(blob.data() + ser.size(), &carry, kCarryBytes);
  return blob;
}

void AtariEnv::restore_blob(ale::ALEInterface& a,
                            const std::vector<char>& blob) {
  const std::string ser(blob.data(), blob.size() - kCarryBytes);
  a.restoreSystemState(ale::ALEState(ser));
}

AtariCarry AtariEnv::read_carry(const std::vector<char>& blob) {
  AtariCarry carry;
  if (blob.size() >= kCarryBytes) {
    std::memcpy(&carry, blob.data() + blob.size() - kCarryBytes, kCarryBytes);
  }
  return carry;
}

void AtariEnv::reset(std::vector<char>& state, std::vector<float>& obs) {
  if (has_initial_) {
    state = initial_state_;
    obs = initial_obs_;
    return;
  }

  ale::ALEInterface& a = *emulators_[0];
  a.reset_game();
  AtariCarry carry;
  if (game_ == AtariGame::kMontezuma) {
    // The start room is already "entered": no bonus for standing still.
    const MontezumaVars v = montezuma_read(a.getRAM().array());
    carry.level_last = v.level;
    if (v.room >= 0 && v.room < kMontezumaRooms) {
      carry.visited_rooms = 1u << static_cast<uint32_t>(v.room);
    }
  }
  state = blob_from(a, carry);
  obs.resize(static_cast<size_t>(obs_dim()));
  fill_obs(0, obs.data());

  initial_state_ = state;
  initial_obs_ = obs;
  has_initial_ = true;
}

void AtariEnv::step_one(int slot, const std::vector<char>& blob, int32_t action,
                        int32_t dt, std::vector<char>& new_blob, float* obs_row,
                        float& reward, uint8_t& done, uint8_t& trunc,
                        DisplayInfo& display, uint8_t& recoverable) {
  ale::ALEInterface& a = *emulators_[static_cast<size_t>(slot)];
  restore_blob(a, blob);
  AtariCarry carry = read_carry(blob);
  // The lives counter travels inside the ALEState blob (RomSettings state is
  // serialized), so this is the walker's own count. 0 = no life counter.
  const int lives_start = a.lives();

  const ale::Action ale_action =
      static_cast<ale::Action>(action_set_[static_cast<size_t>(action)]);
  float total_reward = 0.0f;
  for (int32_t f = 0; f < dt; ++f) {
    total_reward += static_cast<float>(a.act(ale_action));
    if (a.game_over(/*with_truncation=*/true)) {
      break;  // plangym stops frame-skipping when the episode terminates
    }
    if (lives_start > 0 && a.lives() < lives_start) {
      break;  // stop at the life loss so one step never eats two lives
    }
  }

  float step_reward = total_reward;
  display = DisplayInfo{};
  if (game_ == AtariGame::kMontezuma) {
    const MontezumaVars v = montezuma_read(a.getRAM().array());
    step_reward = montezuma_step_reward(v, total_reward, carry, reward_weights_);
    display.room = v.room;
    display.x = montezuma_room_px(v.x);
    display.y = montezuma_room_py(v.y);
    display.level = v.level;
    display.lives = v.lives;
    display.inventory = v.inventory;
  }
  carry.episode_return += step_reward;

  fill_obs(slot, obs_row);
  new_blob = blob_from(a, carry);
  reward = step_reward;
  const bool game_over = a.game_over(/*with_truncation=*/false);
  const bool life_lost = lives_start > 0 && a.lives() < lives_start;
  done = (game_over || life_lost) ? 1 : 0;
  trunc = a.game_truncated() ? 1 : 0;
  recoverable = (life_lost && !game_over && !trunc) ? 1 : 0;
  display.score = carry.episode_return;
}

void AtariEnv::step_batch(const std::vector<std::vector<char>>& states,
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
  recoverable_cache_.assign(static_cast<size_t>(n), 0);
  info_cache_.resize(static_cast<size_t>(n));
  float* obs_base = observations.data();
  pool_.parallel_for(n, [&](int32_t i, int slot) {
    const auto ui = static_cast<size_t>(i);
    step_one(slot, states[ui], actions[ui], dt[ui], new_states[ui],
             obs_base + ui * d, rewards[ui], dones[ui], truncated[ui],
             display_cache_[ui], recoverable_cache_[ui]);
    const DisplayInfo& di = display_cache_[ui];
    WalkerInfo& wi = info_cache_[ui];
    wi = WalkerInfo{};
    wi.score = di.score;
    wi.x = di.x;
    wi.y = di.y;
    wi.world = di.room;
    wi.stage = di.level;
    wi.lives = di.lives;
    wi.inventory = di.inventory;
    wi.has_visit_key = has_visit_key();
    wi.visit_plane = di.room;
    wi.visit_x = di.x;
    wi.visit_y = di.y;
  });
}

const WalkerInfo& AtariEnv::walker_info(int32_t batch_index) const {
  static const WalkerInfo kEmpty{};
  const auto ui = static_cast<size_t>(batch_index);
  return ui < info_cache_.size() ? info_cache_[ui] : kEmpty;
}

bool AtariEnv::done_is_recoverable(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < recoverable_cache_.size() && recoverable_cache_[ui] != 0;
}

const AtariEnv::DisplayInfo& AtariEnv::display_at(int32_t walker_index) const {
  static const DisplayInfo kEmpty{};
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui] : kEmpty;
}

float AtariEnv::display_score(int32_t walker_index) const {
  return display_at(walker_index).score;
}

int32_t AtariEnv::walker_room(int32_t walker_index) const {
  return display_at(walker_index).room;
}

int32_t AtariEnv::walker_x(int32_t walker_index) const {
  return display_at(walker_index).x;
}

int32_t AtariEnv::walker_y(int32_t walker_index) const {
  return display_at(walker_index).y;
}

int32_t AtariEnv::walker_level(int32_t walker_index) const {
  return display_at(walker_index).level;
}

int32_t AtariEnv::walker_lives(int32_t walker_index) const {
  return display_at(walker_index).lives;
}

int32_t AtariEnv::walker_inventory(int32_t walker_index) const {
  return display_at(walker_index).inventory;
}

void AtariEnv::render_frame(const std::vector<char>& state,
                            std::vector<uint8_t>& rgba) {
  ale::ALEInterface& a = *emulators_[0];
  restore_blob(a, state);
  // restoreSystemState() restores the machine but NOT the screen buffer,
  // which still holds whatever slot 0 emulated last (another walker). One
  // NOOP frame redraws the restored state exactly; the advanced state is
  // discarded (every step restores from a blob), so this is side-effect
  // free. (A game-over state does not emulate and keeps the stale frame.)
  a.act(ale::PLAYER_A_NOOP);
  std::vector<unsigned char>& buf = scratch_[0];
  a.getScreenRGB(buf);
  const size_t n_pixels =
      static_cast<size_t>(screen_w_) * static_cast<size_t>(screen_h_);
  rgba.resize(n_pixels * 4);
  for (size_t p = 0; p < n_pixels; ++p) {
    rgba[p * 4 + 0] = buf[p * 3 + 0];
    rgba[p * 4 + 1] = buf[p * 3 + 1];
    rgba[p * 4 + 2] = buf[p * 3 + 2];
    rgba[p * 4 + 3] = 0xFF;
  }
}

}  // namespace fg
