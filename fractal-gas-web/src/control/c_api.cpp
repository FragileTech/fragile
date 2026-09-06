#include "control/c_api.h"
// A single C ABI serves WebAssembly and Python ctypes. Neither binding owns
// physics.
#include <chrono>
#include <string>

#include "control/wave.hpp"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define FGC_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define FGC_EXPORT
#endif

using namespace fg::control;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
thread_local std::string error;
struct Runtime {
  std::shared_ptr<const Scene> scene;
  Physics physics;
  StateBatch state, next;
  std::vector<float> actions, metrics, selected, tree_values, result_values;
  std::vector<uint32_t> tree_meta;
  std::vector<int32_t> frames;
  std::vector<StepResult> results;
  std::unique_ptr<FmcPlanner> planner;
  double planning_ms = 0;
  std::string planner_settings;
  std::vector<uint8_t> checkpoint;
  std::vector<float> debug;
  std::array<double, 12> profile{};
  Runtime(std::shared_ptr<const Scene> s, int n, int threads)
      : scene(s),
        physics(s, threads),
        state(n, *s),
        next(n, *s),
        actions(size_t(n) * s->channels.size()),
        metrics(16),
        result_values(size_t(n) * 4),
        frames(n, 1),
        results(n) {
    state.reset(*s, 7);
  }
  void writable_next() {
    if (next.storage.use_count() > 1) next = StateBatch(state.count, *scene);
  }
  void update_metrics() {
    metrics.assign(16, 0);
    for (size_t w = 0; w < state.count; ++w) {
      metrics[0] += results[w].reward;
      metrics[1] += float(results[w].frames);
      metrics[2] += float(results[w].collisions);
      metrics[3] += word(state.row(w), 7) != 0;
    }
    metrics[4] = float(word(state.row(0), 0));
    metrics[5] = float(word(state.row(0), 4));
    metrics[6] = float(word(state.row(0), 5));
    metrics[7] = float(word(state.row(0), 6));
    if (planner) {
      const auto& st = planner->wave.stats;
      metrics[8] = float(st.iterations);
      metrics[9] = st.dead_ratio;
      metrics[10] = st.clone_ratio;
      metrics[11] = st.mean_reward;
      metrics[12] = st.max_reward;
      metrics[13] = float(planner->wave.tree.size());
      metrics[14] = float(st.pruned);
      metrics[15] = float(planning_ms);
    }
  }
};
template <class T, class F>
T guard(T fallback, F&& f) {
  try {
    error.clear();
    return f();
  } catch (const std::exception& e) {
    error = e.what();
    return fallback;
  } catch (...) {
    error = "Unknown native control failure";
    return fallback;
  }
}
Runtime& runtime(void* p) {
  if (!p) throw std::invalid_argument("Closed engine");
  return *static_cast<Runtime*>(p);
}
fg::control::WaveConfig config(const char* text, const Scene& s) {
  Json j = JsonReader(std::string(text ? text : "{}")).read();
  WaveConfig c;
  auto integer = [&](const char* key, uint32_t fallback, uint32_t lo,
                     uint32_t hi) {
    double n = j[key].num(fallback);
    if (n < lo || n > hi || n != std::floor(n))
      throw std::invalid_argument(std::string("Invalid planner ") + key);
    return uint32_t(n);
  };
  c.walkers = integer("walkers", 128, 1, 8192);
  c.horizon = integer("horizon", 16, 1, 4096);
  c.frames = integer("frames", 6, 1, 4096);
  c.elites = integer("elites", 0, 0, c.walkers);
  c.distance_coef = float(j["distance_coef"].num(1));
  c.reward_coef = float(j["reward_coef"].num(1));
  c.noise = float(j["noise"].num(.2));
  if (c.distance_coef < 0 || c.distance_coef > 10 || c.reward_coef < 0 ||
      c.reward_coef > 10 || c.noise < 0 || c.noise > 10)
    throw std::invalid_argument("Invalid planner coefficients");
  c.cumulative = j["cumulative"].flag(true);
  c.inertial = j["inertial"].flag(true);
  c.recording = static_cast<fg::RecordingMode>(integer("recording", 2, 0, 2));
  size_t estimated =
      size_t(c.walkers) * (s.layout.stride * 8 + s.channels.size() * 4 * 6 +
                           (s.bodies.size() * 7 + s.controlled.size() +
                            s.tethers.size() * 2 + s.extension_observations +
                            (s.cargo_capacity > 0 ? s.controlled.size() * 4 : 0)) *
                               4);
  if (estimated > 512 * 1024 * 1024)
    throw std::invalid_argument(
        "Planner exceeds the 512 MiB working-memory budget");
  return c;
}
}  // namespace
extern "C" {
FGC_EXPORT const char* fgc_error() { return error.c_str(); }
FGC_EXPORT void* fgc_create(const char* json, int worlds, int threads) {
  return guard<void*>(nullptr, [&]() -> void* {
    if (!json || worlds < 1 || worlds > 8192 || threads < 1 || threads > 64)
      throw std::invalid_argument("Invalid engine dimensions");
    auto s = Scene::compile(json);
    if (size_t(worlds) * s->layout.stride * 8 > 512 * 1024 * 1024)
      throw std::invalid_argument("State banks exceed 512 MiB");
    return new Runtime(s, worlds, threads);
  });
}
FGC_EXPORT void fgc_destroy(void* p) { delete static_cast<Runtime*>(p); }
FGC_EXPORT int fgc_reset(void* p, uint32_t lo, uint32_t hi) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    r.state = StateBatch(r.state.count, *r.scene);
    r.state.reset(*r.scene, lo | (uint64_t(hi) << 32));
    r.planner.reset();
    std::fill(r.results.begin(), r.results.end(), StepResult{});
    std::fill(r.actions.begin(), r.actions.end(), 0);
    std::fill(r.frames.begin(), r.frames.end(), 1);
    r.planning_ms = 0;
    r.update_metrics();
    return 0;
  });
}
FGC_EXPORT uint32_t fgc_hash_lo(void* p) {
  return guard<uint32_t>(
      0, [&] { return uint32_t(runtime(p).scene->fingerprint); });
}
FGC_EXPORT uint32_t fgc_hash_hi(void* p) {
  return guard<uint32_t>(
      0, [&] { return uint32_t(runtime(p).scene->fingerprint >> 32); });
}
FGC_EXPORT int fgc_info(void* p, int field) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    const auto& l = r.scene->layout;
    switch (field) {
      case 0:
        return int(r.state.count);
      case 1:
        return int(l.bodies);
      case 2:
        return int(l.controlled);
      case 3:
        return int(l.stride);
      case 4:
        return int(l.words);
      case 5:
        return int(l.flags);
      case 6:
        return int(l.gates);
      case 7:
        return int(l.joints);
      case 8:
        return int(l.food);
      case 9:
        return int(l.tethers);
      case 10:
        return int(l.pickups);
      case 11:
        return int(r.physics.observation_dim());
      case 12:
        return int(r.scene->channels.size());
      case 13:
        return int(l.auxiliary);
      case 14:
        return int(l.auxiliary_words);
      case 15:
        return int(l.cargo);
      default:
        throw std::invalid_argument("Unknown info field");
    }
  });
}
FGC_EXPORT float fgc_action_bound(void* p, int channel, int upper) {
  return guard(0.f, [&] {
    const auto& c = runtime(p).scene->channels.at(channel);
    return upper ? c.high : c.low;
  });
}
FGC_EXPORT int fgc_action_body(void* p, int channel) {
  return guard(-1, [&] { return runtime(p).scene->channels.at(channel).body; });
}
FGC_EXPORT const char* fgc_action_name(void* p, int channel) {
  return guard<const char*>(nullptr, [&] {
    return runtime(p).scene->channels.at(channel).name.c_str();
  });
}
FGC_EXPORT float* fgc_actions(void* p) {
  return guard<float*>(nullptr, [&] { return runtime(p).actions.data(); });
}
FGC_EXPORT int32_t* fgc_frames(void* p) {
  return guard<int32_t*>(nullptr, [&] { return runtime(p).frames.data(); });
}
FGC_EXPORT float* fgc_states(void* p) {
  return guard<float*>(nullptr, [&] { return runtime(p).state.row(0); });
}
FGC_EXPORT float* fgc_metrics(void* p) {
  return guard<float*>(nullptr, [&] {
    auto& r = runtime(p);
    r.update_metrics();
    return r.metrics.data();
  });
}
FGC_EXPORT double* fgc_profile(void* p) {
  return guard<double*>(nullptr, [&] {
    auto& r = runtime(p);
    size_t memory = r.state.bytes() + r.next.bytes() + 4 * r.actions.capacity();
    if (r.planner) {
      const auto& w = r.planner->wave;
      memory += w.current.bytes() + w.next.bytes() + w.elite.bytes() +
                4 * (w.actions.capacity() + w.root_actions.capacity() +
                     w.observations.capacity());
    }
    r.profile[10] = double(memory);
    r.profile[11] = double(r.state.serialized_size());
    return r.profile.data();
  });
}
FGC_EXPORT int fgc_profile_reset(void* p) {
  return guard(-1, [&] {
    runtime(p).profile.fill(0);
    return 0;
  });
}
FGC_EXPORT int fgc_inspect(void* p) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    r.debug = r.physics.inspect(r.state.row(0), r.actions.data());
    return int(r.debug.size() / 8);
  });
}
FGC_EXPORT float* fgc_inspection(void* p) {
  return guard<float*>(nullptr, [&] { return runtime(p).debug.data(); });
}
FGC_EXPORT float* fgc_results(void* p) {
  return guard<float*>(nullptr, [&] {
    auto& r = runtime(p);
    for (size_t i = 0; i < r.results.size(); ++i) {
      r.result_values[i * 4] = r.results[i].reward;
      r.result_values[i * 4 + 1] = float(r.results[i].frames);
      r.result_values[i * 4 + 2] = r.results[i].dead;
      r.result_values[i * 4 + 3] = float(r.results[i].collisions);
    }
    return r.result_values.data();
  });
}
FGC_EXPORT int fgc_step(void* p) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    r.writable_next();
    const auto started = Clock::now();
    r.physics.step(r.state, nullptr, r.actions.data(), r.frames.data(), r.next,
                   r.results.data());
    r.profile[0] += elapsed(started);
    for (const auto& result : r.results) r.profile[1] += result.frames;
    std::swap(r.state, r.next);
    return 0;
  });
}
FGC_EXPORT int fgc_get_states(void* p, float* out, size_t bytes) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!out || bytes < r.state.bytes())
      throw std::invalid_argument("State output too small");
    const auto started = Clock::now();
    std::memmove(out, r.state.row(0), r.state.bytes());
    r.profile[2] += elapsed(started);
    r.profile[3] += r.state.bytes();
    return 0;
  });
}
FGC_EXPORT int fgc_set_states(void* p, const float* in, size_t bytes) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!in || bytes != r.state.bytes())
      throw std::invalid_argument("State input shape mismatch");
    const auto started = Clock::now();
    for (size_t w = 0; w < r.state.count; ++w)
      r.state.validate_row(in + w * r.state.layout.stride);
    r.writable_next();
    std::memmove(r.next.row(0), in, bytes);
    r.profile[4] += elapsed(started);
    r.profile[5] += bytes;
    std::swap(r.state, r.next);
    return 0;
  });
}
FGC_EXPORT int fgc_broadcast(void* p, const uint8_t* data, size_t size) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    StateBatch source(1, *r.scene);
    source.deserialize(data, size);
    r.writable_next();
    for (size_t w = 0; w < r.state.count; ++w)
      std::memcpy(r.next.row(w), source.row(0), source.layout.words * 4);
    std::swap(r.state, r.next);
    return 0;
  });
}
FGC_EXPORT int fgc_gather(void* p, const int32_t* indices, size_t count) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!indices || count != r.state.count)
      throw std::invalid_argument("Gather shape mismatch");
    r.writable_next();
    const auto started = Clock::now();
    r.next.gather(r.state, indices, count);
    r.profile[6] += elapsed(started);
    r.profile[7] += r.state.bytes();
    std::swap(r.state, r.next);
    return 0;
  });
}
FGC_EXPORT size_t fgc_snapshot_size(void* p) {
  return guard<size_t>(0, [&] { return runtime(p).state.serialized_size(); });
}
FGC_EXPORT int fgc_serialize(void* p, uint8_t* out, size_t bytes) {
  return guard(-1, [&] {
    if (!out) throw std::invalid_argument("Missing snapshot output");
    runtime(p).state.serialize(out, bytes);
    return 0;
  });
}
FGC_EXPORT int fgc_deserialize(void* p, const uint8_t* in, size_t bytes) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!in) throw std::invalid_argument("Missing snapshot");
    r.writable_next();
    r.next.deserialize(in, bytes);
    std::swap(r.state, r.next);
    return 0;
  });
}
FGC_EXPORT void* fgc_borrow(void* p) {
  return guard<void*>(
      nullptr, [&]() -> void* { return new StateBatch(runtime(p).state); });
}
FGC_EXPORT float* fgc_batch_data(void* p) {
  return p ? static_cast<StateBatch*>(p)->row(0) : nullptr;
}
FGC_EXPORT void fgc_release(void* p) { delete static_cast<StateBatch*>(p); }
FGC_EXPORT int fgc_observe(void* p, float* out, size_t count) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!out || count != r.state.count * r.physics.observation_dim())
      throw std::invalid_argument("Observation shape mismatch");
    r.physics.pool.parallel_for(int32_t(r.state.count), [&](int32_t i, int) {
      r.physics.observe(r.state.row(i), out + i * r.physics.observation_dim());
    });
    return 0;
  });
}
FGC_EXPORT int fgc_plan_begin(void* p, const char* settings, uint32_t seed) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    std::string text = settings ? settings : "{}";
    if (!r.planner || text != r.planner_settings) {
      r.planner = std::make_unique<FmcPlanner>(
          r.physics, config(text.c_str(), *r.scene), seed);
      r.planner_settings = text;
    } else
      r.planner->wave.reseed(seed);
    r.planner->begin(r.state);
    r.planning_ms = 0;
    return 0;
  });
}
FGC_EXPORT int fgc_plan_advance(void* p) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!r.planner) throw std::logic_error("Begin a plan first");
    auto start = std::chrono::steady_clock::now();
    const bool advances = !r.planner->ready;
    bool done = r.planner->advance();
    if (advances) r.profile[1] += r.planner->wave.stats.frames;
    r.profile[8] += elapsed(start);
    r.profile[9]++;
    r.planning_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    return int(done);
  });
}
FGC_EXPORT float* fgc_plan_action(void* p) {
  return guard<float*>(nullptr, [&] {
    auto& r = runtime(p);
    if (!r.planner) throw std::logic_error("No plan");
    r.planner->finish();
    return r.planner->selected.data();
  });
}
FGC_EXPORT int fgc_wave_step(void* p) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!r.planner) throw std::logic_error("Begin a Wave run first");
    auto start = std::chrono::steady_clock::now();
    r.planner->wave.step();
    r.profile[1] += r.planner->wave.stats.frames;
    r.profile[8] += elapsed(start);
    r.profile[9]++;
    r.planning_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    return 0;
  });
}
FGC_EXPORT float* fgc_wave_states(void* p) {
  return guard<float*>(nullptr, [&] {
    auto& r = runtime(p);
    if (!r.planner) throw std::logic_error("No Wave run");
    return r.planner->wave.current.row(0);
  });
}
FGC_EXPORT int fgc_tree_export(void* p) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!r.planner) throw std::logic_error("No exploration tree");
    r.planner->wave.tree.export_data(r.tree_meta, r.tree_values);
    return int(r.tree_meta.size() / 5);
  });
}
FGC_EXPORT uint32_t* fgc_tree_meta(void* p) {
  return guard<uint32_t*>(nullptr, [&] { return runtime(p).tree_meta.data(); });
}
FGC_EXPORT float* fgc_tree_values(void* p) {
  return guard<float*>(nullptr, [&] { return runtime(p).tree_values.data(); });
}
FGC_EXPORT uint8_t* fgc_tree_root(void* p) {
  return guard<uint8_t*>(nullptr, [&] {
    auto& r = runtime(p);
    if (!r.planner) throw std::logic_error("No tree");
    return r.planner->wave.tree.root_snapshot.data();
  });
}
FGC_EXPORT size_t fgc_tree_root_size(void* p) {
  return guard<size_t>(0, [&] {
    auto& r = runtime(p);
    return r.planner ? r.planner->wave.tree.root_snapshot.size() : 0;
  });
}
FGC_EXPORT size_t fgc_checkpoint_size(void* p) {
  return guard<size_t>(0, [&] {
    auto& r = runtime(p);
    CheckpointWriter out;
    out.scalar(uint32_t(0x50434746));
    out.scalar(uint32_t(1));
#ifdef __EMSCRIPTEN__
    out.string("wasm-control-2");
#else
    out.string("native-control-2");
#endif
    out.scalar(r.scene->fingerprint);
    out.string(r.planner_settings);
    std::vector<uint8_t> state(r.state.serialized_size());
    r.state.serialize(state.data(), state.size());
    out.vector(state);
    out.vector(r.actions);
    out.vector(r.frames);
    out.scalar(uint32_t(bool(r.planner)));
    if (r.planner) {
      out.scalar(uint32_t(r.planner->ready));
      out.vector(r.planner->selected);
      r.planner->wave.save_checkpoint(out);
    }
    out.scalar(checkpoint_hash(out.data.data(), out.data.size()));
    r.checkpoint = std::move(out.data);
    return r.checkpoint.size();
  });
}
FGC_EXPORT int fgc_checkpoint_write(void* p, uint8_t* out, size_t capacity) {
  return guard(-1, [&] {
    const auto& bytes = runtime(p).checkpoint;
    if (!out || bytes.empty() || capacity < bytes.size())
      throw std::invalid_argument("Prepare checkpoint before copying");
    std::memcpy(out, bytes.data(), bytes.size());
    return 0;
  });
}
FGC_EXPORT int fgc_checkpoint_restore(void* p, const uint8_t* data,
                                      size_t size) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!data || size < 24 || size > 512 * 1024 * 1024)
      throw std::invalid_argument("Invalid checkpoint size");
    uint64_t hash;
    std::memcpy(&hash, data + size - 8, 8);
    if (checkpoint_hash(data, size - 8) != hash)
      throw std::invalid_argument("Checkpoint checksum mismatch");
    CheckpointReader in(data, size - 8);
    if (in.scalar<uint32_t>() != 0x50434746 || in.scalar<uint32_t>() != 1)
      throw std::invalid_argument("Unsupported checkpoint version");
#ifdef __EMSCRIPTEN__
    const std::string backend = "wasm-control-2";
#else
    const std::string backend="native-control-2";
#endif
    if (in.string() != backend || in.scalar<uint64_t>() != r.scene->fingerprint)
      throw std::invalid_argument("Checkpoint backend or scene mismatch");
    auto settings = in.string();
    auto bytes = in.vector<uint8_t>(512 * 1024 * 1024);
    StateBatch restored(r.state.count, *r.scene);
    restored.deserialize(bytes.data(), bytes.size());
    auto actions = in.vector<float>();
    auto frames = in.vector<int32_t>();
    if (actions.size() != r.actions.size() || frames.size() != r.frames.size())
      throw std::invalid_argument("Checkpoint runtime shape mismatch");
    for (float a : actions)
      if (!std::isfinite(a))
        throw std::invalid_argument("Nonfinite checkpoint action");
    for (int32_t f : frames)
      if (f < 0 || f > 4096)
        throw std::invalid_argument("Invalid checkpoint duration");
    const auto has = in.scalar<uint32_t>();
    if (has > 1) throw std::invalid_argument("Invalid planner checkpoint flag");
    std::unique_ptr<FmcPlanner> planner;
    if (has) {
      planner = std::make_unique<FmcPlanner>(
          r.physics, config(settings.c_str(), *r.scene), 0);
      const auto ready = in.scalar<uint32_t>();
      if (ready > 1) throw std::invalid_argument("Invalid planner ready flag");
      planner->ready = ready;
      planner->selected = in.vector<float>();
      if (!planner->selected.empty() &&
          planner->selected.size() != r.scene->channels.size())
        throw std::invalid_argument("Invalid selected action shape");
      if (ready && planner->selected.size() != r.scene->channels.size())
        throw std::invalid_argument("Missing selected action");
      for (float a : planner->selected)
        if (!std::isfinite(a))
          throw std::invalid_argument("Nonfinite selected action");
      planner->wave.load_checkpoint(in);
    }
    in.finish();
    r.state = std::move(restored);
    r.actions = std::move(actions);
    r.frames = std::move(frames);
    r.planner = std::move(planner);
    r.planner_settings = std::move(settings);
    r.planning_ms = 0;
    std::fill(r.results.begin(), r.results.end(), StepResult{});
    return 0;
  });
}
FGC_EXPORT int fgc_replay_node(void* p, uint32_t id) {
  return guard(-1, [&] {
    auto& r = runtime(p);
    if (!r.planner || r.state.count != 1)
      throw std::logic_error("Replay requires a single-world planner");
    r.state = r.planner->wave.replay(id);
    return 0;
  });
}
FGC_EXPORT float fgc_raycast(void* p, float x, float y, float dx, float dy,
                             float distance) {
  return guard(-1.f, [&] {
    auto& r = runtime(p);
    if (!std::isfinite(x + y + dx + dy + distance) || distance < 0)
      throw std::invalid_argument("Invalid ray");
    return r.scene->raycast({x, y}, {dx, dy}, distance);
  });
}
}
