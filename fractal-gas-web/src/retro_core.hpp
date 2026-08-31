// RetroCore — one independent libretro core instance in-process.
//
// A libretro core (here Genesis Plus GX) keeps ALL its state in globals, so
// linking it once gives exactly one emulator per process. Like stable-retro's
// Python side (which pays one process per emulator), we get N independent
// cores by loading N copies of the core dynamic library: each RetroCore
// copies the built .so to a unique temp file and dlopens that copy with
// RTLD_LOCAL | RTLD_NOW, giving it a private set of globals. The copy is
// unlinked right after dlopen so nothing is left behind even on a crash.
//
// The libretro callbacks we register (environment / video / audio / input)
// are functions of THIS executable and therefore shared between all copies.
// They dispatch through a thread_local "current core" pointer that every
// RetroCore entry point sets first. This is safe under fg::ThreadPool's
// static partitioning: each pool slot (and hence each core) is driven by
// exactly one thread at a time.
#ifndef FRACTAL_GAS_RETRO_CORE_HPP
#define FRACTAL_GAS_RETRO_CORE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fg {

/// libretro RETRO_DEVICE_ID_JOYPAD_* bit positions, for building the
/// per-instance input bitmask fed back through the input_state callback.
enum RetroPad : uint32_t {
  kPadB = 1u << 0,
  kPadY = 1u << 1,
  kPadSelect = 1u << 2,
  kPadStart = 1u << 3,
  kPadUp = 1u << 4,
  kPadDown = 1u << 5,
  kPadLeft = 1u << 6,
  kPadRight = 1u << 7,
  kPadA = 1u << 8,
  kPadX = 1u << 9,
};

class RetroCore {
 public:
  /// Copies `core_so_path` to a fresh temp file and dlopens it. Throws
  /// std::runtime_error when the copy / dlopen / symbol lookup fails.
  explicit RetroCore(const std::string& core_so_path);
  ~RetroCore();

  RetroCore(const RetroCore&) = delete;
  RetroCore& operator=(const RetroCore&) = delete;

  /// Loads a ROM (retro_load_game). Throws on failure.
  void load_rom(const std::string& rom_path);

  /// Runs one emulated frame with the given joypad bitmask on port 0.
  void run_frame(uint32_t buttons);

  size_t serialize_size();
  bool serialize(void* data, size_t size);
  bool unserialize(const void* data, size_t size);

  /// RETRO_MEMORY_SYSTEM_RAM (Genesis: the 64KB 68k work RAM). NOTE: on a
  /// little-endian host Genesis Plus GX stores it as native 16-bit words, so
  /// consecutive logical (big-endian 68k) bytes are swapped pairwise; a
  /// big-endian u16 at an even 68k address is simply a little-endian u16
  /// read of this buffer (stable-retro's genesis.json overlay ["=",">",2]).
  const uint8_t* work_ram() const;
  /// Writable view of work RAM (same buffer) — used for boot-time level
  /// selection pokes (the Mario write_stage pattern for Genesis games).
  uint8_t* work_ram_mut();
  size_t work_ram_size() const;

  /// Last video frame, converted to packed RGB888 (width*height*3 bytes) in
  /// the video_refresh callback. Empty until the first run_frame.
  const std::vector<uint8_t>& frame_rgb() const { return frame_rgb_; }
  int32_t frame_width() const { return frame_width_; }
  int32_t frame_height() const { return frame_height_; }

 private:
  struct Fns;  // resolved retro_* symbols

  /// Points the shared callback trampolines at this instance (thread_local).
  void make_current();

  static bool cb_environment(unsigned cmd, void* data);
  static void cb_video_refresh(const void* data, unsigned width,
                               unsigned height, size_t pitch);
  static void cb_audio_sample(int16_t left, int16_t right);
  static size_t cb_audio_sample_batch(const int16_t* data, size_t frames);
  static void cb_input_poll();
  static int16_t cb_input_state(unsigned port, unsigned device, unsigned index,
                                unsigned id);

  void* handle_ = nullptr;
  Fns* fns_ = nullptr;

  std::vector<char> rom_data_;
  std::string rom_path_;
  bool rom_loaded_ = false;

  uint32_t buttons_ = 0;
  int pixel_format_ = -1;  // retro_pixel_format from SET_PIXEL_FORMAT
  std::vector<uint8_t> frame_rgb_;
  int32_t frame_width_ = 0;
  int32_t frame_height_ = 0;
};

}  // namespace fg

#endif  // FRACTAL_GAS_RETRO_CORE_HPP
