#include "retro_core.hpp"

#ifndef FG_RETRO_STATIC
#include <dlfcn.h>
#endif
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

// The libretro API header vendored by stable-retro (read-only use).
#include "libretro.h"

namespace fg {

namespace {

/// The core copy being driven by the calling thread. Set by make_current()
/// before every retro_* entry so the shared callback trampolines dispatch to
/// the right instance (one thread drives one core at a time; see header).
thread_local RetroCore* t_current_core = nullptr;

void cb_log(enum retro_log_level /*level*/, const char* /*fmt*/, ...) {
  // Cores call this unconditionally (Genesis Plus GX logs on every
  // retro_get_memory_data), so it must exist; we keep it silent.
}

#ifndef FG_RETRO_STATIC
/// Copy `src` to a unique temp file (mkstemps keeps the .so suffix so dlopen
/// treats it as a shared object on every platform). Returns the temp path.
std::string copy_to_temp(const std::string& src) {
  const char* tmpdir = std::getenv("TMPDIR");
  std::string templ = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") +
                      "/fg_retro_core_XXXXXX.so";
  std::vector<char> path(templ.begin(), templ.end());
  path.push_back('\0');
  const int fd = mkstemps(path.data(), 3);  // 3 = strlen(".so")
  if (fd < 0) {
    throw std::runtime_error("RetroCore: mkstemps failed for " + templ);
  }
  std::ifstream in(src, std::ios::binary);
  if (!in) {
    close(fd);
    unlink(path.data());
    throw std::runtime_error("RetroCore: cannot read core library " + src);
  }
  char buf[1 << 16];
  while (in) {
    in.read(buf, sizeof(buf));
    const ssize_t n = in.gcount();
    if (n > 0 && write(fd, buf, static_cast<size_t>(n)) != n) {
      close(fd);
      unlink(path.data());
      throw std::runtime_error("RetroCore: short write copying core library");
    }
  }
  close(fd);
  return std::string(path.data());
}
#endif  // !FG_RETRO_STATIC

}  // namespace

struct RetroCore::Fns {
  void (*init)(void);
  void (*deinit)(void);
  void (*run)(void);
  size_t (*serialize_size)(void);
  bool (*serialize)(void*, size_t);
  bool (*unserialize)(const void*, size_t);
  bool (*load_game)(const retro_game_info*);
  void (*unload_game)(void);
  void* (*get_memory_data)(unsigned);
  size_t (*get_memory_size)(unsigned);
  void (*set_environment)(retro_environment_t);
  void (*set_video_refresh)(retro_video_refresh_t);
  void (*set_audio_sample)(retro_audio_sample_t);
  void (*set_audio_sample_batch)(retro_audio_sample_batch_t);
  void (*set_input_poll)(retro_input_poll_t);
  void (*set_input_state)(retro_input_state_t);
};

void RetroCore::make_current() { t_current_core = this; }

bool RetroCore::cb_environment(unsigned cmd, void* data) {
  RetroCore* self = t_current_core;
  switch (cmd) {
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
      self->pixel_format_ = *static_cast<const retro_pixel_format*>(data);
      return self->pixel_format_ == RETRO_PIXEL_FORMAT_RGB565 ||
             self->pixel_format_ == RETRO_PIXEL_FORMAT_XRGB8888;
    case RETRO_ENVIRONMENT_GET_VARIABLE: {
      // Core options, mirroring stable-retro's s_envVariables defaults.
      auto* var = static_cast<retro_variable*>(data);
      if (std::strcmp(var->key, "genesis_plus_gx_bram") == 0) {
        var->value = "per game";
        return true;
      }
      if (std::strcmp(var->key, "genesis_plus_gx_render") == 0) {
        var->value = "single field";
        return true;
      }
      if (std::strcmp(var->key, "genesis_plus_gx_blargg_ntsc_filter") == 0) {
        var->value = "disabled";
        return true;
      }
      return false;
    }
    case RETRO_ENVIRONMENT_GET_CAN_DUPE:
      *static_cast<bool*>(data) = true;
      return true;
    case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
      static_cast<retro_log_callback*>(data)->log = cb_log;
      return true;
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY: {
      static const char* dir = "/tmp";
      *static_cast<const char**>(data) = dir;
      return true;
    }
    default:
      return false;
  }
}

void RetroCore::cb_video_refresh(const void* data, unsigned width,
                                 unsigned height, size_t pitch) {
  RetroCore* self = t_current_core;
  if (!data || !width || !height) return;  // frame dupe: keep the last one
  self->frame_width_ = static_cast<int32_t>(width);
  self->frame_height_ = static_cast<int32_t>(height);
  self->frame_rgb_.resize(static_cast<size_t>(width) * height * 3);
  const auto* src = static_cast<const uint8_t*>(data);
  uint8_t* dst = self->frame_rgb_.data();
  if (self->pixel_format_ == RETRO_PIXEL_FORMAT_XRGB8888) {
    for (unsigned y = 0; y < height; ++y) {
      const auto* row = reinterpret_cast<const uint32_t*>(src + y * pitch);
      for (unsigned x = 0; x < width; ++x) {
        const uint32_t px = row[x];
        *dst++ = static_cast<uint8_t>((px >> 16) & 0xFF);
        *dst++ = static_cast<uint8_t>((px >> 8) & 0xFF);
        *dst++ = static_cast<uint8_t>(px & 0xFF);
      }
    }
  } else {  // RGB565 (Genesis Plus GX with FRONTEND_SUPPORTS_RGB565)
    for (unsigned y = 0; y < height; ++y) {
      const auto* row = reinterpret_cast<const uint16_t*>(src + y * pitch);
      for (unsigned x = 0; x < width; ++x) {
        const uint16_t px = row[x];
        const uint32_t r = (px >> 11) & 0x1F;
        const uint32_t g = (px >> 5) & 0x3F;
        const uint32_t b = px & 0x1F;
        *dst++ = static_cast<uint8_t>((r << 3) | (r >> 2));
        *dst++ = static_cast<uint8_t>((g << 2) | (g >> 4));
        *dst++ = static_cast<uint8_t>((b << 3) | (b >> 2));
      }
    }
  }
}

void RetroCore::cb_audio_sample(int16_t /*left*/, int16_t /*right*/) {}

size_t RetroCore::cb_audio_sample_batch(const int16_t* /*data*/,
                                        size_t frames) {
  return frames;
}

void RetroCore::cb_input_poll() {}

int16_t RetroCore::cb_input_state(unsigned port, unsigned device,
                                  unsigned /*index*/, unsigned id) {
  if (port != 0 || device != RETRO_DEVICE_JOYPAD || id >= 32) return 0;
  return (t_current_core->buttons_ >> id) & 1u ? 1 : 0;
}

#ifdef FG_RETRO_STATIC
// Static-link mode (wasm): the core is linked into the executable, so its
// globals exist exactly once — only ONE RetroCore may exist, and walkers are
// stepped serially through it (every step starts from unserialize, so
// correctness is unchanged; only parallelism is lost).
namespace {
int g_static_core_live_count = 0;
}

RetroCore::RetroCore(const std::string& /*core_so_path*/) {
  if (g_static_core_live_count > 0) {
    throw std::runtime_error(
        "RetroCore: static-link mode supports exactly one live core instance");
  }
  ++g_static_core_live_count;

  fns_ = new Fns();
  fns_->init = &retro_init;
  fns_->deinit = &retro_deinit;
  fns_->run = &retro_run;
  fns_->serialize_size = &retro_serialize_size;
  fns_->serialize = &retro_serialize;
  fns_->unserialize = &retro_unserialize;
  fns_->load_game = &retro_load_game;
  fns_->unload_game = &retro_unload_game;
  fns_->get_memory_data = &retro_get_memory_data;
  fns_->get_memory_size = &retro_get_memory_size;
  fns_->set_environment = &retro_set_environment;
  fns_->set_video_refresh = &retro_set_video_refresh;
  fns_->set_audio_sample = &retro_set_audio_sample;
  fns_->set_audio_sample_batch = &retro_set_audio_sample_batch;
  fns_->set_input_poll = &retro_set_input_poll;
  fns_->set_input_state = &retro_set_input_state;

  make_current();
  fns_->set_environment(cb_environment);
  fns_->set_video_refresh(cb_video_refresh);
  fns_->set_audio_sample(cb_audio_sample);
  fns_->set_audio_sample_batch(cb_audio_sample_batch);
  fns_->set_input_poll(cb_input_poll);
  fns_->set_input_state(cb_input_state);
  fns_->init();
}

RetroCore::~RetroCore() {
  if (fns_) {
    make_current();
    if (rom_loaded_) fns_->unload_game();
    fns_->deinit();
  }
  delete fns_;
  --g_static_core_live_count;
}
#else
RetroCore::RetroCore(const std::string& core_so_path) {
  const std::string copy = copy_to_temp(core_so_path);
  handle_ = dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
  // The mapping stays alive after unlink; nothing is left on disk even if
  // the process dies.
  unlink(copy.c_str());
  if (!handle_) {
    throw std::runtime_error(std::string("RetroCore: dlopen failed: ") +
                             dlerror());
  }

  fns_ = new Fns();
  auto resolve = [this](const char* name) -> void* {
    void* p = dlsym(handle_, name);
    if (!p) {
      throw std::runtime_error(std::string("RetroCore: missing symbol ") +
                               name);
    }
    return p;
  };
  try {
    fns_->init = reinterpret_cast<void (*)()>(resolve("retro_init"));
    fns_->deinit = reinterpret_cast<void (*)()>(resolve("retro_deinit"));
    fns_->run = reinterpret_cast<void (*)()>(resolve("retro_run"));
    fns_->serialize_size =
        reinterpret_cast<size_t (*)()>(resolve("retro_serialize_size"));
    fns_->serialize =
        reinterpret_cast<bool (*)(void*, size_t)>(resolve("retro_serialize"));
    fns_->unserialize = reinterpret_cast<bool (*)(const void*, size_t)>(
        resolve("retro_unserialize"));
    fns_->load_game = reinterpret_cast<bool (*)(const retro_game_info*)>(
        resolve("retro_load_game"));
    fns_->unload_game =
        reinterpret_cast<void (*)()>(resolve("retro_unload_game"));
    fns_->get_memory_data =
        reinterpret_cast<void* (*)(unsigned)>(resolve("retro_get_memory_data"));
    fns_->get_memory_size = reinterpret_cast<size_t (*)(unsigned)>(
        resolve("retro_get_memory_size"));
    fns_->set_environment = reinterpret_cast<void (*)(retro_environment_t)>(
        resolve("retro_set_environment"));
    fns_->set_video_refresh =
        reinterpret_cast<void (*)(retro_video_refresh_t)>(
            resolve("retro_set_video_refresh"));
    fns_->set_audio_sample = reinterpret_cast<void (*)(retro_audio_sample_t)>(
        resolve("retro_set_audio_sample"));
    fns_->set_audio_sample_batch =
        reinterpret_cast<void (*)(retro_audio_sample_batch_t)>(
            resolve("retro_set_audio_sample_batch"));
    fns_->set_input_poll = reinterpret_cast<void (*)(retro_input_poll_t)>(
        resolve("retro_set_input_poll"));
    fns_->set_input_state = reinterpret_cast<void (*)(retro_input_state_t)>(
        resolve("retro_set_input_state"));
  } catch (...) {
    delete fns_;
    dlclose(handle_);
    throw;
  }

  make_current();
  fns_->set_environment(cb_environment);
  fns_->set_video_refresh(cb_video_refresh);
  fns_->set_audio_sample(cb_audio_sample);
  fns_->set_audio_sample_batch(cb_audio_sample_batch);
  fns_->set_input_poll(cb_input_poll);
  fns_->set_input_state(cb_input_state);
  fns_->init();
}

RetroCore::~RetroCore() {
  if (handle_) {
    make_current();
    if (rom_loaded_) fns_->unload_game();
    fns_->deinit();
    dlclose(handle_);
  }
  delete fns_;
}
#endif  // FG_RETRO_STATIC

void RetroCore::load_rom(const std::string& rom_path) {
  std::ifstream in(rom_path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("RetroCore: cannot open ROM " + rom_path);
  rom_data_.resize(static_cast<size_t>(in.tellg()));
  in.seekg(0);
  in.read(rom_data_.data(), static_cast<std::streamsize>(rom_data_.size()));
  rom_path_ = rom_path;

  retro_game_info info{};
  info.path = rom_path_.c_str();
  info.data = rom_data_.data();
  info.size = rom_data_.size();
  make_current();
  if (!fns_->load_game(&info)) {
    throw std::runtime_error("RetroCore: retro_load_game failed for " +
                             rom_path);
  }
  rom_loaded_ = true;
}

void RetroCore::run_frame(uint32_t buttons) {
  make_current();
  buttons_ = buttons;
  fns_->run();
}

size_t RetroCore::serialize_size() {
  make_current();
  return fns_->serialize_size();
}

bool RetroCore::serialize(void* data, size_t size) {
  make_current();
  return fns_->serialize(data, size);
}

bool RetroCore::unserialize(const void* data, size_t size) {
  make_current();
  return fns_->unserialize(data, size);
}

const uint8_t* RetroCore::work_ram() const {
  auto* self = const_cast<RetroCore*>(this);
  self->make_current();
  return static_cast<const uint8_t*>(
      self->fns_->get_memory_data(RETRO_MEMORY_SYSTEM_RAM));
}

uint8_t* RetroCore::work_ram_mut() {
  make_current();
  return static_cast<uint8_t*>(fns_->get_memory_data(RETRO_MEMORY_SYSTEM_RAM));
}

size_t RetroCore::work_ram_size() const {
  auto* self = const_cast<RetroCore*>(this);
  self->make_current();
  return self->fns_->get_memory_size(RETRO_MEMORY_SYSTEM_RAM);
}

}  // namespace fg
