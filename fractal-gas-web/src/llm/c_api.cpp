#include "llm/environment.hpp"
#include "control/json.hpp"
#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include <iomanip>
#include <sstream>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
// One import awaits the entire selected batch, keeping the shared core synchronous.
EM_ASYNC_JS(char*, token_transition, (const char* input), {
  try {
    const result = await Module.llmTransition(JSON.parse(UTF8ToString(input)));
    const text = JSON.stringify(result);
    const ptr = _malloc(lengthBytesUTF8(text) + 1);
    stringToUTF8(text, ptr, lengthBytesUTF8(text) + 1);
    return ptr;
  } catch (error) {
    Module.llmError = error.message;
    return 0;
  }
});
#else
#define EXPORT
static char* token_transition(const char*) { return nullptr; }
#endif
namespace {
using namespace fg;
using namespace fg::llm;
using fg::control::Json;
using fg::control::JsonReader;
std::string error;
int integer(const Json& j, const char* key, int lo, int hi) {
  const double v = j[key].num(-1);
  if (!std::isfinite(v) || v < lo || v > hi || v != std::floor(v))
    throw std::invalid_argument(std::string("Invalid ") + key);
  return int(v);
}
std::vector<Result> transport(const std::vector<Request>& requests) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < requests.size(); ++i) {
    if (i) out << ',';
    const auto& r = requests[i];
    out << "{\"source\":" << r.source.id << ",\"action\":" << r.action
        << ",\"duration\":" << r.duration << '}';
  }
  out << ']';
  char* response = token_transition(out.str().c_str());
  if (!response) throw std::runtime_error("Token provider transition failed");
  std::string text(response); std::free(response);
  auto rows = JsonReader(text).read();
  std::vector<Result> results;
  auto integer = [](const Json& j) {
    double v = j.num(-1);
    if (!std::isfinite(v) || v < 0 || v > UINT32_MAX || v != std::floor(v)) throw std::invalid_argument("Invalid token metadata");
    return uint32_t(v);
  };
  for (const auto& row : rows.items()) {
    Result r;
    r.state.id = integer(row["id"]); r.state.tokens = integer(row["tokens"]);
    r.state.utility = row["utility"].num(NAN);
    r.state.logp = row["logp"].num(); r.state.status = integer(row["status"]);
    r.skipped = row["skipped"].flag(false);
    for (const auto& x : row["embedding"].items()) r.embedding.push_back(float(x.num()));
    results.push_back(std::move(r));
  }
  return results;
}
struct Engine {
  LlmEnvironment env;
  std::unique_ptr<SwarmAlgorithm> swarm;
  std::string output;
  std::vector<int64_t> evaluated;
  bool valid = true;
  Engine(const Json& j) : env(integer(j, "dimensions", 1, 65536), transport) {
    int n = integer(j, "walkers", 2, 1024), dt = integer(j, "chunk_tokens", 1, 4096);
    int seed = integer(j, "seed", 0, 2147483647);
    auto metric = parse_distance_metric(j["distance_metric"].str("l2"));
    float dc = float(j["distance_coef"].num(1)), rc = float(j["reward_coef"].num(1));
    if (!std::isfinite(dc) || !std::isfinite(rc) || dc < 0 || dc > 10 || rc < 0 || rc > 10)
      throw std::invalid_argument("Invalid coefficients");
    if (j["algorithm"].str() == "wave") {
      FractalGasParams p; p.N = n; p.dt_min = p.dt_max = dt; p.seed = seed;
      p.distance_metric = metric; p.dist_coef = dc; p.reward_coef = rc;
      p.use_cumulative_reward = true; p.n_elite = 0; p.count_visits = false;
      auto wave = std::make_unique<FractalGas>(env, p);
      wave->enable_diagnostics(); swarm = std::move(wave);
    } else if (j["algorithm"].str() == "graph") {
      FractalTreeParams p; p.start_walkers = p.min_leafs = n;
      p.max_walkers = integer(j, "max_walkers", n, 4096); p.dt_min = p.dt_max = dt; p.seed = seed;
      p.distance_metric = metric; p.dist_coef = dc; p.reward_coef = rc;
      p.count_visits = p.visit_reward = false;
      auto graph = std::make_unique<FractalTree>(env, p);
      graph->enable_diagnostics(); swarm = std::move(graph);
    } else throw std::invalid_argument("Unknown LLM algorithm");
  }
  void advance(bool reset) {
    evaluated.clear();
    if (reset) swarm->reset();
    if (reset && dynamic_cast<FractalTree*>(swarm.get())) return;
    for (int i = 0; i < swarm->n_walkers(); ++i) {
      const auto& blob = swarm->walker_state(i);
      evaluated.push_back(blob.empty() ? -1 : int64_t(LlmEnvironment::decode(blob).id));
    }
    swarm->step();
  }
  const char* snapshot() {
    std::ostringstream out; out << std::setprecision(10) << "{\"iteration\":" << swarm->iteration_count() << ",\"walkers\":[";
    auto* wave = dynamic_cast<FractalGas*>(swarm.get());
    auto* graph = dynamic_cast<FractalTree*>(swarm.get());
    for (int i = 0; i < swarm->n_walkers(); ++i) {
      if (i) out << ',';
      const auto& blob = swarm->walker_state(i);
      out << "{\"slot\":" << i << ",\"node\":";
      if (blob.empty()) out << "null"; else out << LlmEnvironment::decode(blob).id;
      out << ",\"alive\":" << (swarm->walker_alive(i) ? "true" : "false")
          << ",\"leaf\":" << (swarm->walker_is_leaf(i) ? "true" : "false")
          << ",\"parentSlot\":" << swarm->walker_parent(i)
          << ",\"score\":" << swarm->walker_cum_reward(i);
      int fit, clone; bool cloned; float fitness;
      if (wave) {
        fit = wave->fitness_companions()[i]; clone = wave->clone_companions()[i];
        cloned = wave->clone_mask()[i]; fitness = wave->state().virtual_rewards[i];
      } else {
        const auto& s = graph->state(); fit = s.distance_ix[i]; clone = s.clone_ix[i];
        cloned = s.will_clone[i]; fitness = s.virtual_rewards[i];
      }
      out << ",\"fitnessCompanion\":" << fit << ",\"cloneCompanion\":" << clone
          << ",\"cloned\":" << (cloned ? "true" : "false") << ",\"fitness\":" << fitness << '}';
    }
    out << "],\"decisions\":[";
    const auto& decisions = wave ? wave->diagnostics().decisions : graph->diagnostics().decisions;
    for (size_t i = 0; i < decisions.size() && i < evaluated.size(); ++i) {
      if (i) out << ',';
      const auto& d = decisions[i];
      auto ref = [&](int slot) { if (evaluated[slot] < 0) out << "null"; else out << evaluated[slot]; };
      out << std::boolalpha << "{\"iteration\":" << swarm->iteration_count() << ",\"slot\":" << i
          << ",\"evaluated\":"; ref(int(i));
      out << ",\"companion\":"; ref(d.distance_companion);
      out << ",\"donor\":"; ref(d.clone_donor);
      out << ",\"result\":";
      const auto& blob = swarm->walker_state(int(i));
      if (blob.empty()) out << "null"; else out << LlmEnvironment::decode(blob).id;
      out << ",\"companion_slot\":" << d.distance_companion << ",\"donor_slot\":" << d.clone_donor
          << ",\"distance\":" << d.distance << ",\"distance_norm\":" << d.distance_norm
          << ",\"reward_norm\":" << d.reward_norm << ",\"other\":" << d.other
          << ",\"fitness\":" << d.fitness << ",\"donor_fitness\":" << d.donor_fitness
          << ",\"clone_score\":" << d.clone_score << ",\"draw\":" << d.draw
          << ",\"alive\":" << d.alive << ",\"normalization_leaf\":" << d.normalization_leaf
          << ",\"leaf\":" << d.leaf << ",\"donor_protected\":" << d.donor_protected
          << ",\"best_protected\":" << d.best_protected << ",\"elite_protected\":" << d.elite_protected
          << ",\"invalid_donor\":" << d.invalid_donor << ",\"wanted\":" << d.wanted
          << ",\"cloned\":" << d.cloned << '}';
    }
    out << "]}"; output = out.str(); return output.c_str();
  }
};
template<class T, class F> T guard(T fallback, F fn) {
  try { error.clear(); return fn(); } catch (const std::exception& e) { error = e.what(); return fallback; }
}
}
extern "C" {
EXPORT const char* fgllm_error() { return error.c_str(); }
EXPORT void* fgllm_create(const char* config) {
  return guard<void*>(nullptr, [&]() -> void* {
    if (!config) throw std::invalid_argument("Missing LLM configuration");
    std::string text(config); return new Engine(JsonReader(text).read());
  });
}
EXPORT int fgllm_advance(void* handle, int reset) {
  return guard(-1, [&] {
    auto* e = static_cast<Engine*>(handle);
    if (!e || !e->valid) throw std::invalid_argument("LLM engine needs reset after failure");
    try { e->advance(reset); }
    catch (...) { e->valid = false; throw; }
    return 0;
  });
}
EXPORT const char* fgllm_snapshot(void* h) {
  return guard<const char*>(nullptr, [&] {
    if (!h) throw std::invalid_argument("Missing LLM engine");
    return static_cast<Engine*>(h)->snapshot();
  });
}
EXPORT void fgllm_destroy(void* h) { delete static_cast<Engine*>(h); }
}
