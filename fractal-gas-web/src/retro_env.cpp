#include "retro_env.hpp"

#include <zlib.h>

#include <cstring>
#include <stdexcept>

namespace fg {

namespace {

std::vector<char> gunzip_file(const std::string& path) {
  gzFile f = gzopen(path.c_str(), "rb");
  if (!f) {
    throw std::runtime_error("RetroEnv: cannot open savestate " + path);
  }
  std::vector<char> out;
  char buf[1 << 16];
  int n;
  while ((n = gzread(f, buf, sizeof(buf))) > 0) {
    out.insert(out.end(), buf, buf + n);
  }
  const bool truncated_read = n < 0;
  gzclose(f);
  if (truncated_read || out.empty()) {
    throw std::runtime_error("RetroEnv: failed to decompress savestate " +
                             path);
  }
  return out;
}

}  // namespace

RetroEnv::RetroEnv(std::string core_so_path, std::string rom_path,
                   std::string state_path, int n_threads, ObsMode obs_mode,
                   RetroGame game)
    : core_so_path_(std::move(core_so_path)),
      rom_path_(std::move(rom_path)),
      state_path_(std::move(state_path)),
      obs_mode_(obs_mode),
      game_(game) {
  const int n_slots = n_threads < 1 ? 1 : n_threads;
  cores_.reserve(static_cast<size_t>(n_slots));
  for (int i = 0; i < n_slots; ++i) {
    auto core = std::make_unique<RetroCore>(core_so_path_);
    core->load_rom(rom_path_);
    cores_.push_back(std::move(core));
  }
  serialize_size_ = cores_[0]->serialize_size();
  if (serialize_size_ == 0) {
    throw std::runtime_error("RetroEnv: core reports zero serialize size");
  }
  pool_ = std::make_unique<ThreadPool>(n_slots);
}

RetroEnv::~RetroEnv() = default;

int32_t RetroEnv::n_actions() const { return kRetroNumActions; }

int32_t RetroEnv::obs_dim() const {
  return retro_obs_dim(game_, static_cast<int32_t>(obs_mode_));
}

std::vector<char> RetroEnv::dump_with_carry(RetroCore& core,
                                            const RetroCarry& carry) {
  std::vector<char> blob(serialize_size_ + kRetroCarryBytes);
  if (!core.serialize(blob.data(), serialize_size_)) {
    throw std::runtime_error("RetroEnv: retro_serialize failed");
  }
  std::memcpy(blob.data() + serialize_size_, &carry, kRetroCarryBytes);
  return blob;
}

RetroCarry RetroEnv::read_carry(const std::vector<char>& blob) {
  RetroCarry carry;
  std::memcpy(&carry, blob.data() + blob.size() - kRetroCarryBytes,
              kRetroCarryBytes);
  return carry;
}

void RetroEnv::load_blob(RetroCore& core, const std::vector<char>& blob) {
  if (blob.size() != serialize_size_ + kRetroCarryBytes ||
      !core.unserialize(blob.data(), serialize_size_)) {
    throw std::runtime_error("RetroEnv: retro_unserialize failed");
  }
}

void RetroEnv::reset(std::vector<char>& state, std::vector<float>& obs) {
  if (has_initial_) {
    state = initial_state_;
    obs = initial_obs_;
    return;
  }

  RetroCore& core = *cores_[0];
  if (state_path_.empty()) {
    if (!retro_boot_to_gameplay(game_, core)) {
      throw std::runtime_error(
          "RetroEnv: could not reach gameplay from boot (unexpected ROM?)");
    }
  } else {
    const std::vector<char> level1 = gunzip_file(state_path_);
    if (!core.unserialize(level1.data(), level1.size())) {
      throw std::runtime_error("RetroEnv: could not restore " + state_path_ +
                               " (savestate/core version mismatch?)");
    }
  }
  // One idle frame so the framebuffer belongs to the current state (the
  // core only redraws on retro_run); the blob is serialized AFTER it so
  // state and observation stay consistent.
  core.run_frame(0);

  const RetroCarry carry = retro_init_carry(game_, core);

  state = dump_with_carry(core, carry);
  obs.resize(static_cast<size_t>(obs_dim()));
  retro_fill_obs(game_, static_cast<int32_t>(obs_mode_), core, obs.data());

  initial_state_ = state;
  initial_obs_ = obs;
  has_initial_ = true;
}

void RetroEnv::step_one(int slot, const std::vector<char>& blob,
                        int32_t action, int32_t dt,
                        std::vector<char>& new_blob, float* obs_row,
                        float& reward, uint8_t& done, float& display) {
  RetroCore& core = *cores_[static_cast<size_t>(slot)];
  load_blob(core, blob);
  RetroCarry carry = read_carry(blob);

  bool episode_done = false;
  reward = retro_step_frames(game_, core, action, dt, carry, episode_done,
                             display);

  retro_fill_obs(game_, static_cast<int32_t>(obs_mode_), core, obs_row);
  new_blob = dump_with_carry(core, carry);
  done = episode_done ? 1 : 0;
}

void RetroEnv::step_batch(const std::vector<std::vector<char>>& states,
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
  pool_->parallel_for(n, [&](int32_t i, int slot) {
    const auto ui = static_cast<size_t>(i);
    step_one(slot, states[ui], actions[ui], dt[ui], new_states[ui],
             obs_base + ui * d, rewards[ui], dones[ui], display_cache_[ui]);
    truncated[ui] = 0;  // the retro env never truncates
  });
}

float RetroEnv::display_score(int32_t walker_index) const {
  const auto ui = static_cast<size_t>(walker_index);
  return ui < display_cache_.size() ? display_cache_[ui] : 0.0f;
}

void RetroEnv::render_frame(const std::vector<char>& state,
                            std::vector<uint8_t>& rgba) {
  RetroCore& core = *cores_[0];
  load_blob(core, state);
  // The core redraws only on retro_run, so advance one idle frame to get
  // the picture. This is display-only: the caller's blob is untouched.
  core.run_frame(0);
  rgba.resize(static_cast<size_t>(kRetroFrameWidth) *
              static_cast<size_t>(kRetroFrameHeight) * 4);
  retro_frame_rgba(core, rgba.data());
}

}  // namespace fg
