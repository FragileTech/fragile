#include "optimization/engine.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>

#include "arcade_planner.hpp"
#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include "optimization/environment.hpp"

namespace fg::optimization {
std::unique_ptr<Algorithm> make_euclidean(Benchmark&, const Settings&);
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
  BenchmarkEnvironment env;
  Settings s;
  std::unique_ptr<SwarmAlgorithm> swarm;
  std::unique_ptr<ArcadePlanner> planner;
  Population p;
  ObservedCloning* observed = nullptr;
  void update() {
    const int count = swarm->n_walkers();
    p.resize(count + (planner ? 1 : 0), env.b.d);
    p.has_velocity = false;
    auto* wave = dynamic_cast<FractalGas*>(swarm.get());
    auto* graph = dynamic_cast<FractalTree*>(swarm.get());
    std::vector<uint8_t> cloned;
    if (wave && wave->state().has_virtual_rewards && observed &&
        observed->draws.size() == 2) {
      const auto probabilities = observed->clone_probs_with_companions(
          wave->state().virtual_rewards, observed->draws[1]);
      cloned = observed->decide_with_uniforms(probabilities, observed->uniforms,
                                              observed->alive);
    }
    for (int i = 0; i < count; ++i) {
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
          p.cloned[i] = cloned.empty() ? 0 : cloned[i];
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
      if (planner) {
        if (!planner->search_advanced()) {
          // Executing the plan leaves the search cloud unchanged.
          p.cloned[i] = 0;
          p.parent[i] = i;
        } else if (planner->depth() == 1) {
          // New searches descend from the committed row, not the old cloud.
          p.parent[i] = count;
        }
      }
    }
    if (planner) {
      p.objective[count] =
          env.decode(planner->state(), p.x.data() + size_t(count) * p.d);
      p.alive[count] = !planner->done() &&
                       env.b.valid(p.x.data() + size_t(count) * p.d) &&
                       std::isfinite(p.objective[count]);
      p.leaf[count] = 0;
      if (planner->done()) std::fill(p.alive.begin(), p.alive.end(), 0);
    }
  }

 public:
  ExistingSwarm(Benchmark& b, const Settings& settings)
      : env(b, settings), s(settings) {
    if (s.algorithm == "wave" || s.planning()) {
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
      a.recording = s.planning() ? RecordingMode::Pruned : RecordingMode::Off;
      a.record_observations = false;
      auto trace = std::make_unique<ObservedCloning>();
      observed = trace.get();
      swarm = std::make_unique<FractalGas>(
          env, a, std::make_unique<OptimizationRng>(s.seed), std::move(trace));
      if (s.planning()) {
        ArcadePlannerSettings options;
        options.algorithm = s.algorithm == "fmc" ? 2 : 3;
        options.horizon = s.horizon;
        options.max_horizon = s.max_horizon;
        options.consensus_prefix = s.consensus_prefix;
        planner = std::make_unique<ArcadePlanner>(
            env, *static_cast<FractalGas*>(swarm.get()), options);
        planner->reset();
      } else {
        swarm->reset();
        swarm->step();
      }
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
    if (planner)
      planner->advance();
    else
      swarm->step();
    update();
  }
  const Population& population() const override { return p; }
  uint64_t evaluations() const override { return env.evals; }
  double objective_score(int i) const override {
    if (!planner) return swarm->walker_cum_reward(i);
    if (i == swarm->n_walkers()) return s.score(p.objective.at(i));
    const auto& bytes =
        static_cast<FractalGas*>(swarm.get())->exploration_tree().root_snapshot;
    std::vector<float> x(env.b.d);
    return swarm->walker_cum_reward(i) +
           s.score(env.decode(std::vector<char>(bytes.begin(), bytes.end()),
                              x.data()));
  }
};
static std::map<std::string, Factory>& factories() {
  static std::map<std::string, Factory> f{
      {"euclidean", make_euclidean},
      {"fmc",
       [](Benchmark& b, const Settings& s) {
         return std::make_unique<ExistingSwarm>(b, s);
       }},
      {"wave_jump",
       [](Benchmark& b, const Settings& s) {
         return std::make_unique<ExistingSwarm>(b, s);
       }},
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
  result.object["perturbations"] = perturbation_catalog();
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
  best = settings.worst();
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
  double mean = 0, current = settings.worst();
  for (int i = 0; i < p.n; ++i) {
    cloned += p.cloned[i] != 0;
    if (p.alive[i]) {
      ++alive;
      mean += p.objective[i];
      if (settings.better(p.objective[i], current)) {
        current = p.objective[i];
        best_index = i;
      }
    }
  }
  if (settings.better(current, best)) best = current;
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
