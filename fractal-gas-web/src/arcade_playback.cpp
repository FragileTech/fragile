// A single-emulator playback module. No population, planner, or search RNG.
#include <emscripten/bind.h>
#include <fstream>
#include <memory>
#include "nes_env.hpp"
#include "atari_env.hpp"

namespace {
std::unique_ptr<fg::BatchEnv> env;
std::vector<std::vector<char>> current(1), output(1);
std::vector<float> obs, rewards(1);
std::vector<uint8_t> done(1), truncated(1), rgba;
std::vector<int32_t> actions(1), frames(1);
void init(emscripten::val rom, int console, int game, int mode, int world, int stage) {
  auto bytes = emscripten::convertJSArrayToNumberVector<uint8_t>(rom);
  { std::ofstream f("/rom.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }
  if (console == 0)
    env = std::make_unique<fg::NesMarioEnv>("/rom.bin", 1, fg::ObsMode(mode), world, stage);
  else if (console == 1)
    env = std::make_unique<fg::AtariEnv>("/rom.bin", 1, fg::AtariObsMode(mode),
      game == 1 ? fg::AtariGame::kMontezuma : fg::AtariGame::kGeneric);
  else throw std::runtime_error("Unsupported playback console");
  obs.resize(env->obs_dim());
}
void restore(emscripten::val root) {
  auto bytes = emscripten::convertJSArrayToNumberVector<uint8_t>(root);
  current[0].assign(bytes.begin(), bytes.end());
}
void step(int action, int dt) {
  if (dt <= 0) return;
  actions[0] = action; frames[0] = dt;
  env->step_batch(current, actions, frames, output, obs, rewards, done, truncated);
  current.swap(output);
}
emscripten::val frame() {
  env->render_frame(current[0], rgba);
  return emscripten::val::global("Uint8Array").new_(emscripten::val(
      emscripten::typed_memory_view(rgba.size(), rgba.data())));
}
}
EMSCRIPTEN_BINDINGS(arcade_playback) {
  emscripten::function("initPlayback", &init);
  emscripten::function("restorePlayback", &restore);
  emscripten::function("stepPlayback", &step);
  emscripten::function("renderPlayback", &frame);
  emscripten::function("frameWidth", +[]() { return env->frame_width(); });
  emscripten::function("frameHeight", +[]() { return env->frame_height(); });
}
