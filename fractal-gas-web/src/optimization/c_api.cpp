#include "optimization/c_api.h"

#include <map>

#include "optimization/engine.hpp"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif
using namespace fg::optimization;
namespace {
std::map<uint32_t, std::unique_ptr<Session>> sessions;
uint32_t next_handle = 1;
thread_local std::string error;
Session& get(uint32_t h) {
  auto it = sessions.find(h);
  if (it == sessions.end())
    throw std::invalid_argument("Invalid optimization session");
  return *it->second;
}
template <class T, class F>
T guard(T fail, F fn) {
  try {
    error.clear();
    return fn();
  } catch (const std::exception& e) {
    error = e.what();
    return fail;
  } catch (...) {
    error = "Unknown optimization engine error";
    return fail;
  }
}
}  // namespace
extern "C" {
EXPORT const char* fgo_catalog() {
  static std::string catalog;
  catalog = discovery_json();
  return catalog.c_str();
}
EXPORT const char* fgo_error() { return error.c_str(); }
EXPORT uint32_t fgo_create(const char* config) {
  return guard<uint32_t>(0, [&] {
    if (!config) throw std::invalid_argument("Missing configuration");
    if (sessions.size() >= 16)
      throw std::invalid_argument("Too many optimization sessions");
    auto session =
        std::make_unique<Session>(JsonReader(std::string(config)).read());
    uint32_t h = next_handle++;
    sessions.emplace(h, std::move(session));
    return h;
  });
}
EXPORT int fgo_destroy(uint32_t h) {
  return guard<int>(-1, [&] {
    get(h);
    sessions.erase(h);
    return 0;
  });
}
EXPORT const char* fgo_config(uint32_t h) {
  return guard<const char*>(nullptr,
                            [&] { return get(h).config_json.c_str(); });
}
EXPORT int fgo_step(uint32_t h) {
  return guard<int>(-1, [&] {
    get(h).step();
    return 0;
  });
}
EXPORT const double* fgo_snapshot(uint32_t h) {
  return guard<const double*>(nullptr, [&] { return get(h).snapshot.data(); });
}
EXPORT int fgo_snapshot_size(uint32_t h) {
  return guard<int>(-1, [&] { return int(get(h).snapshot.size()); });
}
EXPORT int fgo_sample(uint32_t h, const float* x, int n, double* out) {
  return guard<int>(-1, [&] {
    auto& b = get(h).benchmark;
    if (n < 0 || n > 1000000 || !x || !out)
      throw std::invalid_argument("Invalid sampling buffers");
    for (int i = 0; i < n; ++i) out[i] = b.evaluate(x + size_t(i) * b.d);
    return 0;
  });
}
}
