#include "optimization/engine.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>

#include "fractal_gas.hpp"
#include "fractal_tree.hpp"

namespace fg::optimization {
std::unique_ptr<Algorithm> make_euclidean(Benchmark&, const Settings&);
// Opaque BatchEnv state: initialized byte, cached objective, then d
// coordinates. The uninitialized root has score zero. Its first transition
// emits -U(x), subsequent transitions emit cached_U - new_U. Cloning copies
// both fields.
class ObjectiveEnv final : public BatchEnv {
 public:
  Benchmark& b;
  Settings s;
  OptimizationRng rng;
  uint64_t evals = 0;
  ObjectiveEnv(Benchmark& bench, const Settings& settings)
      : b(bench), s(settings), rng(s.seed) {}
  int32_t n_actions() const override { return 1; }
  int32_t obs_dim() const override { return b.d; }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }
  void render_frame(const std::vector<char>&,
                    std::vector<uint8_t>& rgba) override {
    rgba.clear();
  }
  size_t bytes() const { return 1 + sizeof(double) + sizeof(float) * b.d; }
  void reset(std::vector<char>& state, std::vector<float>& obs) override {
    state.assign(bytes(), 0);
    obs.assign(b.d, 0);
    rng = OptimizationRng(s.seed);
    evals = 0;
  }
  double decode(const std::vector<char>& state, float* x) const {
    if (state.size() != bytes()) return INFINITY;
    double u;
    std::memcpy(&u, state.data() + 1, sizeof(u));
    std::memcpy(x, state.data() + 1 + sizeof(u), sizeof(float) * b.d);
    return state[0] ? u : INFINITY;
  }
  void step_batch(const std::vector<std::vector<char>>& states,
                  const std::vector<int32_t>&, const std::vector<int32_t>& dt,
                  std::vector<std::vector<char>>& next,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones,
                  std::vector<uint8_t>& truncated) override {
    for (size_t i = 0; i < states.size(); ++i) {
      auto* x = observations.data() + i * b.d;
      bool initialized = states[i].size() == bytes() && states[i][0];
      double old = initialized ? decode(states[i], x) : 0, u = old;
      if (!initialized) b.initial(x, rng);
      for (int t = 0; t < dt[i]; ++t) {
        if (initialized || t > 0)
          for (int k = 0; k < b.d; ++k)
            x[k] += float(s.proposal * (b.high - b.low) * normal(rng));
        if (s.periodic) b.wrap(x);
        if (!b.valid(x)) break;
      }
      u = b.evaluate(x, &rng);
      ++evals;
      bool valid = b.valid(x) && std::isfinite(u) && std::isfinite(float(u));
      next[i].assign(bytes(), 0);
      next[i][0] = 1;
      std::memcpy(next[i].data() + 1, &u, sizeof(u));
      std::memcpy(next[i].data() + 1 + sizeof(u), x, sizeof(float) * b.d);
      rewards[i] = valid ? float(old - u) : 0;
      dones[i] = !valid;
      truncated[i] = 0;
    }
  }
};
// Capture the existing Wave operator's draws without changing its decisions.
class ObservedCloning final : public FractalCloningOperator {
 public:
  mutable std::vector<std::vector<int32_t>> draws;
  mutable std::vector<uint8_t> alive;
  mutable std::vector<float> uniforms;
  std::vector<int32_t> sample_companions(const std::vector<uint8_t>& mask,
                                         Rng& rng) const override {
    auto result = FractalCloningOperator::sample_companions(mask, rng);
    if (draws.size() == 2) draws.clear();
    draws.push_back(result);
    alive = mask;
    return result;
  }
  std::vector<float> sample_uniforms(int32_t n, Rng& rng) const override {
    uniforms = FractalCloningOperator::sample_uniforms(n, rng);
    return uniforms;
  }
};
class ExistingSwarm final : public Algorithm {
  ObjectiveEnv env;
  Settings s;
  std::unique_ptr<SwarmAlgorithm> swarm;
  Population p;
  ObservedCloning* observed = nullptr;
  void update() {
    p.resize(swarm->n_walkers(), env.b.d);
    p.has_velocity = false;
    auto* wave = dynamic_cast<FractalGas*>(swarm.get());
    auto* graph = dynamic_cast<FractalTree*>(swarm.get());
    for (int i = 0; i < p.n; ++i) {
      p.objective[i] =
          env.decode(swarm->walker_state(i), p.x.data() + size_t(i) * p.d);
      p.alive[i] = swarm->walker_alive(i) &&
                   env.b.valid(p.x.data() + size_t(i) * p.d) &&
                   std::isfinite(p.objective[i]);
      p.leaf[i] = swarm->walker_is_leaf(i);
      p.parent[i] = swarm->walker_parent(i);
      if (wave) {
        auto& st = wave->state();
        if (st.has_virtual_rewards) p.fitness[i] = st.virtual_rewards[i];
        if (observed && observed->draws.size() == 2) {
          p.companions[i] = observed->draws[0][i];
          p.clone_companions[i] = observed->draws[1][i];
          double own = p.fitness[i],
                 other = st.virtual_rewards[p.clone_companions[i]];
          double probability =
              (other - own) / std::max(double(observed->eps), own);
          p.cloned[i] =
              probability > observed->uniforms[i] || !observed->alive[i];
          p.parent[i] = p.cloned[i] ? p.clone_companions[i] : i;
        }
      }
      if (graph) {
        auto& st = graph->state();
        p.fitness[i] = st.virtual_rewards[i];
        p.companions[i] = st.distance_ix[i];
        p.clone_companions[i] = st.clone_ix[i];
        p.cloned[i] = st.will_clone[i];
      }
    }
  }

 public:
  ExistingSwarm(Benchmark& b, const Settings& settings)
      : env(b, settings), s(settings) {
    if (s.algorithm == "wave") {
      FractalGasParams a;
      a.N = s.walkers;
      a.seed = s.seed;
      a.dist_coef = float(s.distance_coef);
      a.reward_coef = float(s.reward_coef);
      a.use_cumulative_reward = true;
      a.dt_min = s.dt_min;
      a.dt_max = s.dt_max;
      a.n_elite = s.elites;
      a.count_visits = false;
      a.recording = RecordingMode::Off;
      auto trace = std::make_unique<ObservedCloning>();
      observed = trace.get();
      swarm = std::make_unique<FractalGas>(
          env, a, std::make_unique<OptimizationRng>(s.seed), std::move(trace));
      swarm->reset();
      swarm->step();
    } else {
      FractalTreeParams a;
      a.start_walkers = s.walkers;
      a.min_leafs = s.walkers;
      a.max_walkers = s.max_walkers;
      a.seed = s.seed;
      a.dist_coef = float(s.distance_coef);
      a.reward_coef = float(s.reward_coef);
      a.dt_min = s.dt_min;
      a.dt_max = s.dt_max;
      a.count_visits = false;
      a.visit_reward = false;
      swarm = std::make_unique<FractalTree>(
          env, a, std::make_unique<OptimizationRng>(s.seed));
      swarm->reset();
    }
    update();
  }
  void step() override {
    swarm->step();
    update();
  }
  const Population& population() const override { return p; }
  uint64_t evaluations() const override { return env.evals; }
  double objective_score(int i) const override {
    return swarm->walker_cum_reward(i);
  }
};
static std::map<std::string, Factory>& factories() {
  static std::map<std::string, Factory> f{
      {"euclidean", make_euclidean},
      {"wave",
       [](Benchmark& b, const Settings& s) {
         return std::make_unique<ExistingSwarm>(b, s);
       }},
      {"graph", [](Benchmark& b, const Settings& s) {
         return std::make_unique<ExistingSwarm>(b, s);
       }}};
  return f;
}
static std::map<std::string, Json>& descriptions() {
  static auto entries = [] {
    std::map<std::string, Json> result;
    const auto catalog = JsonReader(catalog_json()).read();
    for (const auto& entry : catalog["algorithms"].array)
      result.emplace(entry["id"].str(), entry);
    return result;
  }();
  return entries;
}
void register_algorithm(const std::string& id, const std::string& name,
                        bool velocity, Factory factory,
                        const Json& parameters) {
  if (id.empty() || name.empty() || !factory || factories().count(id))
    throw std::invalid_argument("Duplicate or invalid optimization algorithm");
  Json entry;
  entry.kind = Json::Object;
  for (auto pair : {std::make_pair("id", id), std::make_pair("name", name)}) {
    Json v;
    v.kind = Json::String;
    v.string = pair.second;
    entry.object[pair.first] = v;
  }
  Json v;
  v.kind = Json::Boolean;
  v.number = velocity;
  entry.object["velocity"] = v;
  if (parameters.kind == Json::Array) entry.object["parameters"] = parameters;
  descriptions().emplace(id, entry);
  factories().emplace(id, std::move(factory));
}
std::string discovery_json() {
  auto result = JsonReader(catalog_json()).read();
  Json list;
  list.kind = Json::Array;
  for (auto& entry : descriptions()) list.array.push_back(entry.second);
  result.object["algorithms"] = list;
  return stringify(result);
}
Session::Session(const Json& config)
    : benchmark(config), settings(benchmark.config) {
  // Enforce bounded allocations before native or WASM construction.
  uint64_t count =
      settings.algorithm == "graph" ? settings.max_walkers : settings.walkers;
  if (count * uint64_t(benchmark.d) * 32 > 128 * 1024 * 1024)
    throw std::invalid_argument(
        "Swarm exceeds 128 MiB state budget; reduce walkers or dimensions");
  if (settings.algorithm == "euclidean" && settings.walkers > 4096 &&
      (settings.companion != "uniform" ||
       settings.clone_companion != "uniform" || settings.rho > 0))
    throw std::invalid_argument(
        "Use uniform companions and global fitness for more than 4096 walkers");
  auto it = factories().find(settings.algorithm);
  if (it == factories().end())
    throw std::invalid_argument("Unknown optimization algorithm");
  if (benchmark.stochastic) {
    settings.potential_force = false;
    Json off;
    off.kind = Json::Boolean;
    settings.json.object["potential_force"] = off;
  }
  config_json = stringify(settings.json);
  algorithm = it->second(benchmark, settings);
  capture();
}
void Session::step() {
  const auto& p = algorithm->population();
  if (std::none_of(p.alive.begin(), p.alive.end(),
                   [](uint8_t v) { return v != 0; }))
    throw std::runtime_error(
        "All walkers are invalid or outside the domain. "
        "Reset or change bounds/time step.");
  algorithm->step();
  ++iteration;
  capture();
}
void Session::capture() {
  const auto& p = algorithm->population();
  int alive = 0, cloned = 0, best_index = -1;
  double mean = 0, current = INFINITY;
  for (int i = 0; i < p.n; ++i) {
    cloned += p.cloned[i] != 0;
    if (p.alive[i]) {
      ++alive;
      mean += p.objective[i];
      if (p.objective[i] < current) {
        current = p.objective[i];
        best_index = i;
      }
    }
  }
  best = std::min(best, current);
  snapshot = {1,
              double(p.n),
              double(p.d),
              double(p.has_velocity),
              double(iteration),
              double(algorithm->evaluations()),
              double(alive),
              double(cloned),
              current,
              best,
              alive ? mean / alive : INFINITY,
              double(best_index)};
  snapshot.reserve(12 + size_t(p.n) * (2 * p.d + 8));
  for (int i = 0; i < p.n; ++i) {
    for (int k = 0; k < p.d; ++k) snapshot.push_back(p.x[size_t(i) * p.d + k]);
    for (int k = 0; k < p.d; ++k) snapshot.push_back(p.v[size_t(i) * p.d + k]);
    snapshot.insert(
        snapshot.end(),
        {p.objective[i], double(p.fitness[i]), double(p.alive[i]),
         double(p.companions[i]), double(p.clone_companions[i]),
         double(p.parent[i]), double(p.cloned[i]), double(p.leaf[i])});
  }
}
}  // namespace fg::optimization
