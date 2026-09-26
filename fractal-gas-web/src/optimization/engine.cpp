#include "optimization/engine.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <set>

#include "arcade_planner.hpp"
#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include "optimization/environment.hpp"
#include "optimization/adaptive.hpp"
#include "optimization/controller.hpp"

namespace fg::optimization {
std::unique_ptr<Algorithm> make_euclidean(Benchmark&, const Settings&);
std::unique_ptr<Algorithm> make_cma(Benchmark&, const Settings&);
std::unique_ptr<Algorithm> make_gas2017(Benchmark&, const Settings&);
class ExistingSwarm final : public Algorithm {
  BenchmarkEnvironment env;
  Settings s;
  std::unique_ptr<SwarmAlgorithm> swarm;
  std::unique_ptr<ArcadePlanner> planner;
  Population p;
  void freeze_proposals() {
    std::vector<std::vector<char>> states;
    for(int i=0;i<swarm->n_walkers();++i) states.push_back(swarm->walker_state(i));
    env.freeze(states);
  }
  void update(bool resized = false) {
    const int count = swarm->n_walkers();
    p.resize(count + (planner ? 1 : 0), env.b.d);
    p.has_velocity = false;
    auto* wave = dynamic_cast<FractalGas*>(swarm.get());
    auto* graph = dynamic_cast<FractalTree*>(swarm.get());
    for (int i = 0; i < count; ++i) {
      p.lineage[i] = env.lineage(swarm->walker_state(i));
      p.objective[i] = env.decode(swarm->walker_state(i), p.x.data() + size_t(i) * p.d);
      p.alive[i] = swarm->walker_alive(i) && env.b.valid(p.x.data() + size_t(i) * p.d) &&
                   std::isfinite(p.objective[i]);
      p.leaf[i] = swarm->walker_is_leaf(i);
      p.parent[i] = swarm->walker_parent(i);
      if (wave) {
        auto& st = wave->state();
        if (st.has_virtual_rewards) p.fitness[i] = st.virtual_rewards[i];
        p.companions[i] = wave->fitness_companions()[i];
        p.clone_companions[i] = wave->clone_companions()[i];
        p.cloned[i] = wave->clone_mask()[i];
        p.parent[i] = p.cloned[i] ? p.clone_companions[i] : i;
      }
      if (graph) {
        auto& st = graph->state();
        p.fitness[i] = st.virtual_rewards[i];
        p.companions[i] = st.distance_ix[i];
        p.clone_companions[i] = st.clone_ix[i];
        p.cloned[i] = st.will_clone[i];
      }
      if (planner) {
        if (resized || !planner->search_advanced()) {
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
      p.objective[count] = env.decode(planner->state(), p.x.data() + size_t(count) * p.d);
      p.alive[count] = !planner->done() && env.b.valid(p.x.data() + size_t(count) * p.d) &&
                       std::isfinite(p.objective[count]);
      p.leaf[count] = 0;
      if (planner->done()) std::fill(p.alive.begin(), p.alive.end(), 0);
    }
  }

 public:
  ExistingSwarm(Benchmark& b, const Settings& settings) : env(b, settings), s(settings) {
    if (s.algorithm == "wave" || s.planning()) {
      FractalGasParams a;
      a.N = s.walkers;
      a.max_walkers = s.max_walkers;
      a.removal_policy = fractal::removal_policy(s.json["removal_policy"].str("virtual_reward"));
      a.seed = s.seed;
      a.distance_metric = s.distance_metric;
      a.dist_coef = float(s.distance_coef);
      a.reward_coef = float(s.reward_coef);
      a.use_cumulative_reward = true;
      a.dt_min = s.dt_min;
      a.dt_max = s.dt_max;
      a.n_elite = s.elites;
      a.count_visits = false;
      a.recording = s.planning() ? RecordingMode::Pruned : RecordingMode::Off;
      a.record_observations = false;
      swarm = std::make_unique<FractalGas>(env, a, std::make_unique<OptimizationRng>(s.seed));
      if (s.planning()) {
        ArcadePlannerSettings options;
        options.algorithm = s.algorithm == "fmc" ? 2 : 3;
        options.horizon = s.horizon;
        options.max_horizon = s.max_horizon;
        options.consensus_prefix = s.consensus_prefix;
        planner =
            std::make_unique<ArcadePlanner>(env, *static_cast<FractalGas*>(swarm.get()), options);
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
      a.freeze_prefix_after = s.freeze_prefix_after;
      a.seed = s.seed;
      a.distance_metric = s.distance_metric;
      a.dist_coef = float(s.distance_coef);
      a.reward_coef = float(s.reward_coef);
      a.dt_min = s.dt_min;
      a.dt_max = s.dt_max;
      a.count_visits = false;
      a.visit_reward = false;
      swarm = std::make_unique<FractalTree>(env, a, std::make_unique<OptimizationRng>(s.seed));
      swarm->reset();
    }
    update();
  }
  void step() override {
    env.begin_geometry_step();
    if (planner) {
      if (planner->new_search_pending()) { env.update_perturbation();freeze_proposals(); }
      env.collect_perturbations(!planner->execution_pending());
      planner->advance();
    } else {
      freeze_proposals();
      swarm->step();
      env.update_perturbation();
    }
    update();
  }
  bool settings_boundary() const override { return !planner || planner->new_search_pending() || planner->done(); }
  bool finished() const override {return planner && planner->done();}
  void validate_settings(const Settings& next) const override {
    if (s.algorithm == "graph" && next.max_walkers < swarm->n_walkers())
      throw std::invalid_argument("Maximum tree population cannot be below stored node count");
    if (planner) {
      ArcadePlannerSettings options;
      options.algorithm = s.algorithm == "fmc" ? 2 : 3;
      options.horizon = next.horizon;
      options.max_horizon = next.max_horizon;
      options.consensus_prefix = next.consensus_prefix;
      options.validate();
    }
    if (!planner && s.algorithm == "wave" && next.walkers > swarm->n_walkers() &&
        !static_cast<FractalGas*>(swarm.get())->state().alive_count())
      throw std::invalid_argument("Cannot grow a population without alive donors");
  }
  void configure(const Settings& next) override {
    validate_settings(next);
    auto proposal = env.prepare_settings(next);
    Settings saved = next, environment = next;
    if (auto* wave = dynamic_cast<FractalGas*>(swarm.get())) {
      wave->configure_population(next.walkers, next.max_walkers, next.elites,
          fractal::removal_policy(next.json["removal_policy"].str()), bool(planner));
    } else {
      static_cast<FractalTree*>(swarm.get())->configure_population(
          next.walkers, next.max_walkers, next.freeze_prefix_after);
    }
    swarm->set_distance_metric(next.distance_metric);
    swarm->set_dist_coef(float(next.distance_coef));
    swarm->set_reward_coef(float(next.reward_coef));
    swarm->set_dt_range(next.dt_min, next.dt_max);
    if (planner && (next.horizon != s.horizon || next.max_horizon != s.max_horizon ||
                    next.consensus_prefix != s.consensus_prefix)) {
      ArcadePlannerSettings options;
      options.algorithm = s.algorithm == "fmc" ? 2 : 3;
      options.horizon = next.horizon;
      options.max_horizon = next.max_horizon;
      options.consensus_prefix = next.consensus_prefix;
      planner->configure(options);
    }
    env.configure(std::move(environment), std::move(proposal));
    const bool resized = s.walkers != next.walkers;
    s = std::move(saved);
    if (resized && !planner) update(true);
  }
  std::unique_ptr<fractal::PopulationMember> exchange_member(const std::string& id,const std::string& key,int count) override {
    if(s.algorithm!="wave") throw std::invalid_argument("Population exchange currently requires Wave");
    auto* wave=static_cast<FractalGas*>(swarm.get());
    auto member=wave->population_member(id,key,count);
    member->imports_enabled=s.json["population_imports"].num(1)>0;
    member->score=[this](int i){return s.score(p.objective.at(i));};
    return member;
  }
  void refresh_exchange() override { update(true); }
  const Population& population() const override { return p; }
  void set_geometry_diagnostics(bool enabled) override {env.set_geometry(enabled);}
  Json movement_geometry() const override {return env.geometry();}
  void restore_movement_geometry(const Json& geometry) override {env.restore_geometry(geometry);}
  Json metadata() const override {
    Json result; result.kind = Json::Object;
    result.object["exploration"]=env.diagnostics();
    result.object["geometry"]=env.visual_geometry();
    auto* graph = dynamic_cast<FractalTree*>(swarm.get());
    if (!graph) {
      const auto status = static_cast<FractalGas*>(swarm.get())->population_status();
      Json pop; pop.kind = Json::Object;
      pop.object["maximum"] = number(status.maximum);
      pop.object["active"] = number(status.active);
      pop.object["requested"] = number(status.requested);
      pop.object["pending"].kind = Json::Boolean;
      pop.object["pending"].number = status.pending();
      pop.object["removal_policy"].kind = Json::String;
      pop.object["removal_policy"].string = fractal::removal_policy_name(status.policy);
      result.object["population"] = std::move(pop);
      return result;
    }
    Json frozen; frozen.kind = Json::Array;
    for (const auto& node : graph->frozen_nodes()) {
      if (!node.prefix) continue;
      Json entry; entry.kind = Json::Object;
      entry.object["id"] = number(static_cast<double>(node.id));
      entry.object["parentId"] = number(static_cast<double>(node.parent_id));
      Json x; x.kind = Json::Array;
      std::vector<float> coordinates(static_cast<size_t>(p.d));
      const double value = env.decode(node.state, coordinates.data());
      for (float v : coordinates) x.array.push_back(number(v));
      entry.object["x"] = std::move(x);
      entry.object["value"] = number(value);
      frozen.array.push_back(std::move(entry));
    }
    result.object["frozen"] = std::move(frozen);
    result.object["activeRootId"] = number(static_cast<double>(graph->active_root_id()));
    result.object["activeRootParentId"] =
        number(static_cast<double>(graph->state().parent_ids[0]));
    return result;
  }
  uint64_t evaluations() const override { return env.b.evaluations; }
  uint64_t next_evaluations_upper_bound() const override {
    const uint64_t cost=evaluated_perturbation(s.perturbation)?adaptive_evaluation_bound(env.b,s.json)+1:1;
    if (planner && planner->execution_pending()) return cost;
    return cost*(uint64_t(planner && planner->new_search_pending() ? s.walkers : swarm->n_walkers()) +
           (s.algorithm == "graph" ? std::min(s.walkers, s.max_walkers - swarm->n_walkers()) : 0));
  }
  uint64_t next_population_size() const override {
    if (planner && planner->new_search_pending()) return uint64_t(s.walkers) + 1;
    return s.algorithm == "graph" ? std::min(s.max_walkers, swarm->n_walkers() + s.walkers) : p.n;
  }
  double objective_score(int i) const override {
    if (!planner) return swarm->walker_cum_reward(i);
    if (i == swarm->n_walkers()) return s.score(p.objective.at(i));
    const auto& bytes = static_cast<FractalGas*>(swarm.get())->exploration_tree().root_snapshot;
    std::vector<float> x(env.b.d);
    return swarm->walker_cum_reward(i) +
           s.score(env.decode(std::vector<char>(bytes.begin(), bytes.end()), x.data()));
  }
};
static std::map<std::string, Factory>& factories() {
  static std::map<std::string, Factory> f{
      {"cmaes_active", make_cma},
      {"cmaes_bipop", make_cma},
      {"gas", make_gas2017},
      {"euclidean", make_euclidean},
      {"fmc",
       [](Benchmark& b, const Settings& s) { return std::make_unique<ExistingSwarm>(b, s); }},
      {"wave_jump",
       [](Benchmark& b, const Settings& s) { return std::make_unique<ExistingSwarm>(b, s); }},
      {"wave",
       [](Benchmark& b, const Settings& s) { return std::make_unique<ExistingSwarm>(b, s); }},
      {"graph",
       [](Benchmark& b, const Settings& s) { return std::make_unique<ExistingSwarm>(b, s); }}};
  return f;
}
static std::map<std::string, Json>& descriptions() {
  static auto entries = [] {
    std::map<std::string, Json> result;
    const auto catalog = JsonReader(catalog_json()).read();
    for (const auto& entry : catalog["algorithms"].array) result.emplace(entry["id"].str(), entry);
    result.emplace("gas", JsonReader(std::string(R"json({
      "id":"gas", "name":"GAS (2017)", "velocity":false,
      "parameters":[
        {"id":"gas_tabu", "label":"Tabu memory", "type":"boolean", "default":true},
        {"id":"gas_local_search", "label":"L-BFGS-B local search", "type":"boolean", "default":true},
        {"id":"gas_local_evaluations", "label":"Evaluations per local search",
         "type":"integer", "default":200, "min":1, "max":1000000}
      ]
    })json"))
                              .read());
    for (auto id : {"cmaes_active", "cmaes_bipop"}) {
      Json entry = JsonReader(std::string(R"json({"velocity":false,"parameters":[
        {"id":"cma_sigma","label":"Initial standard deviation (0 = 20% of width)","type":"number","default":0,"min":0,"max":1000000},
        {"id":"cma_population","label":"Initial population (0 = automatic)","type":"integer","default":0,"min":0,"max":100000}
      ]})json"))
                       .read();
      entry.object["id"].kind = entry.object["name"].kind = Json::String;
      entry.object["id"].string = id;
      entry.object["name"].string =
          std::string(id) == "cmaes_active" ? "Active CMA-ES" : "BIPOP-active CMA-ES";
      if (std::string(id) == "cmaes_bipop")
        entry.object["parameters"].array.push_back(
            JsonReader(
                std::string(
                    R"({"id":"cma_runs","label":"Large-population runs","type":"integer","default":9,"min":1,"max":1000})"))
                .read());
      result.emplace(id, entry);
    }
    return result;
  }();
  return entries;
}
void register_algorithm(const std::string& id, const std::string& name, bool velocity,
                        Factory factory, const Json& parameters) {
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
static void validate_resources(const Benchmark& benchmark, const Settings& settings) {
  // Enforce bounded allocations before native or WASM construction.
  uint64_t count = settings.algorithm == "graph" || settings.algorithm == "wave" || settings.planning()
                       ? settings.max_walkers : settings.walkers;
  const uint64_t state_bytes =
      count * uint64_t(benchmark.d) * (settings.algorithm == "gas" || settings.algorithm == "wave" || settings.planning() ? 64 : 32) +
      (settings.algorithm == "gas" ? uint64_t(benchmark.d) * 4096 + count * 256 : 0) +
      (settings.perturbation == "local_covariance" ? 32 * uint64_t(benchmark.d) * benchmark.d * sizeof(double) : 0) +
      (evaluated_perturbation(settings.perturbation) ?
       32*uint64_t(benchmark.d)*(benchmark.d<=64?benchmark.d:9)*sizeof(double) +
       4096*(uint64_t(benchmark.d)*28+512)+count*(uint64_t(benchmark.d)*4+40) +
       32*uint64_t(benchmark.d)*(benchmark.d<=64?benchmark.d:9)*sizeof(double) : 0) +
      (settings.json["controller_enabled"].flag(false)?
       64*(uint64_t(benchmark.d)*(benchmark.d<=64?benchmark.d:9)+256)*sizeof(Json)+8*1024*1024:0);
  if (!settings.cma() && state_bytes > 128 * 1024 * 1024)
    throw std::invalid_argument(
        "Swarm exceeds 128 MiB state budget; reduce walkers or dimensions");
  if (settings.algorithm == "euclidean" && settings.walkers > 4096 &&
      (settings.companion != "uniform" || settings.clone_companion != "uniform" ||
       settings.rho > 0))
    throw std::invalid_argument(
        "Use uniform companions and global fitness for more than 4096 walkers");
}
int population_capacity(const Benchmark& benchmark,const Settings& settings) {
  if(settings.algorithm!="euclidean" && settings.algorithm!="gas") return settings.max_walkers;
  int low=std::max(2,settings.elites),high=settings.max_walkers,accepted=0;
  while(low<=high) {
    const int candidate=low+(high-low)/2;Settings next=settings;next.walkers=candidate;
    try {validate_resources(benchmark,next);accepted=candidate;low=candidate+1;}
    catch(const std::invalid_argument&) {high=candidate-1;}
  }
  if(!accepted) throw std::invalid_argument("No controller population fits the resource limit");
  return accepted;
}
static uint64_t initialization_cost(const Benchmark& b,const Settings& s) {
  if(s.planning()) return 1;
  uint64_t cost=s.walkers;
  if(s.algorithm=="gas" && s.gas_local_search) cost+=s.gas_local_evaluations;
  if(evaluated_perturbation(s.perturbation) && (s.algorithm=="wave" || s.algorithm=="graph") && s.dt_max>1)
    cost=uint64_t(s.walkers)*(1+adaptive_evaluation_bound(b,s.json));
  return cost;
}
uint64_t Session::initial_evaluation_bound(const Json& config) {
  Benchmark b(config);Settings s(b.config);validate_resources(b,s);return initialization_cost(b,s);
}
std::unique_ptr<fractal::PopulationMember> Session::exchange_member(const std::string& id,int count) {
  if(!controller) throw std::invalid_argument("Population exchange requires Wave");
  controller->collect_basin_events=true;
  const auto archive=JsonReader(controller->archive.export_json()).read();
  return algorithm->exchange_member(id,"optimization-wave-v1:"+stringify(archive["compatibility"]),count);
}
Json Session::take_basin_events() {
  Json result;result.kind=Json::Array;
  if(controller) result.array.swap(controller->basin_events);
  return result;
}
void Session::synchronize_basins(const std::string& data) {
  if(!controller) throw std::invalid_argument("Population exchange requires Wave");
  controller->archive.synchronize_json(data);
}
Session::~Session()=default;
Session::Session(const Json& config) : benchmark(config), settings(benchmark.config) {
  validate_resources(benchmark, settings);
  auto it = factories().find(settings.algorithm);
  if (it == factories().end()) throw std::invalid_argument("Unknown optimization algorithm");
  if (!settings.cma() && settings.max_evaluations &&
      settings.max_evaluations < uint64_t(settings.planning() ? 1 : settings.walkers))
    throw std::invalid_argument("Evaluation budget is too small to initialize the swarm");
  if (benchmark.stochastic) {
    settings.potential_force = false;
    Json off;
    off.kind = Json::Boolean;
    settings.json.object["potential_force"] = off;
    if (settings.algorithm == "gas") {
      settings.gas_local_search = false;
      settings.json.object["gas_local_search"] = off;
    }
  }
  if (settings.algorithm == "gas" && settings.max_evaluations &&
      settings.max_evaluations <
          uint64_t(settings.walkers) +
              (settings.gas_local_search ? uint64_t(settings.gas_local_evaluations) : 0))
    throw std::invalid_argument(
        "Evaluation budget is too small for GAS initialization and local search cap");
  if(!settings.cma() && settings.max_evaluations && initialization_cost(benchmark,settings)>settings.max_evaluations)
    throw std::invalid_argument("Budget is too small for a complete initialization");
  if(!settings.cma()) {
    controller=std::make_unique<RunController>(benchmark,settings);
    benchmark.observed=[this](const double* x,double value){controller->observe(x,value);};
  }
  best = settings.worst();
  config_json = stringify(settings.json);
  algorithm = it->second(benchmark, settings);
  for (const auto& field : algorithm->resolved_config().object)
    settings.json.object[field.first] = field.second;
  config_json = stringify(settings.json);
  if (settings.max_evaluations && benchmark.evaluations > settings.max_evaluations)
    throw std::invalid_argument("Initialization exceeds the evaluation budget");
  capture();
}
void Session::set_geometry_diagnostics(bool enabled) {
  algorithm->set_geometry_diagnostics(enabled);
  settings.json.object["geometry_diagnostics"].kind=Json::Boolean;
  settings.json.object["geometry_diagnostics"].number=enabled;
  if(pending) pending->json.object["geometry_diagnostics"]=settings.json["geometry_diagnostics"];
}
std::string Session::status_json() const {
  Json info = algorithm->metadata();
  info.object["finished"].kind = Json::Boolean;
  info.object["finished"].number = algorithm->finished()&&!restart_ready();
  info.object["next_evaluations"] =
      number(algorithm->finished()&&!restart_ready() ? 0 : next_evaluations());
  info.object["next_population"] = number(restart_ready()?Settings(controller->next(settings).config).walkers+(settings.planning()?1:0):algorithm->next_population_size());
  if(controller) info.object["controller"]=controller->status(settings);
  info.object["best_position"]=array(benchmark.best_position);
  if(settings.json["geometry_diagnostics"].flag(false)) {
    const uint64_t d=benchmark.d;
    uint64_t numbers=settings.cma()?d*d+8*d:
      std::min<uint64_t>(std::max<uint64_t>(262144,d*d+5*d),16*d*d+80*d)+
      2*std::min<uint64_t>(262144,16*d*(d<=64?d:9)+80*d)+
      2*d*std::min<uint64_t>(512,262144/std::max<uint64_t>(1,d));
    info.object["geometry_capacity_bytes"]=number(32*numbers+65536);
  }
  info.object["budget_exhausted"].kind = Json::Boolean;
  info.object["budget_exhausted"].number =
      settings.max_evaluations &&
      (benchmark.evaluations >= settings.max_evaluations ||
       next_evaluations() > settings.max_evaluations - benchmark.evaluations);
  info.object["effective_settings"] = settings.json;
  info.object["pending_settings"] = pending ? pending->json : Json{};
  info.object["settings_revision"] = number(settings_revision);
  if (settings_event.kind == Json::Object && settings_event["iteration"].num(-1) == double(iteration))
    info.object["settings_event"] = settings_event;
  if (pending && info["population"].kind == Json::Object) {
    info.object["population"].object["requested"] = number(pending->walkers);
    info.object["population"].object["pending"].kind = Json::Boolean;
    info.object["population"].object["pending"].number = pending->walkers != settings.walkers;
  }
  return stringify(info);
}
Settings Session::validated_settings(const Json& patch) const {
  if (patch.kind != Json::Object) throw std::invalid_argument("Settings patch must be an object");
  std::set<std::string> allowed{"max_evaluations"};
  auto add = [&](std::initializer_list<const char*> fields) { for (auto key : fields) allowed.insert(key); };
  if (!settings.cma()) {
    add({"walkers", "max_walkers", "removal_policy", "periodic", "boundary", "perturbation",
         "perturbation_std", "covariance_learning_rate", "adaptive_active", "adaptive_paths",
         "adaptive_scale", "adaptive_difference", "adaptive_pairs", "adaptive_mixture", "adaptive_euclidean_mode",
         "adaptive_min_scale", "adaptive_max_scale", "cloning_geometry", "cloning_drift", "cloning_drift_strength",
         "controller_enabled","population_auto","scale_auto","basin_avoidance","restart_token"});
    if (settings.algorithm == "euclidean")
      add({"gamma", "beta", "delta_t", "substeps", "clone_every", "epsilon", "clone_epsilon",
           "lambda_alg", "eta", "sigma_min", "amplitude", "epsilon_dist", "rho", "p_max",
           "epsilon_clone", "sigma_x", "restitution", "potential_force", "cloning", "kinetic",
           "companion", "clone_companion", "reward_coef", "distance_coef"});
    else if (settings.algorithm == "gas") add({"gas_tabu", "gas_local_search", "gas_local_evaluations"});
    else {
      add({"distance_metric", "reward_coef", "distance_coef", "dt_min", "dt_max"});
      if (settings.algorithm == "graph") add({"freeze_prefix_after"});
      else add({"elites"});
      if (settings.planning()) add({"horizon"});
      if (settings.algorithm == "wave_jump") add({"max_horizon", "consensus_prefix"});
    }
  }
  Json merged = pending ? pending->json : settings.json;
  for (const auto& [key, value] : patch.object) {
    if (!allowed.count(key)) throw std::invalid_argument("Setting requires a new run: " + key);
    const auto kind = merged[key].kind;
    if (value.kind != kind && !(key == "removal_policy" && value.kind == Json::String))
      throw std::invalid_argument("Invalid setting type: " + key);
    merged.object[key] = value;
  }
  if(patch.object.count("walkers") && !patch.object.count("population_auto")) {merged.object["population_auto"].kind=Json::Boolean;merged.object["population_auto"].number=0;}
  if(patch.object.count("perturbation_std") && !patch.object.count("scale_auto")) {merged.object["scale_auto"].kind=Json::Boolean;merged.object["scale_auto"].number=0;}
  if(patch.object.count("adaptive_min_scale") || patch.object.count("adaptive_max_scale") || patch.object.count("scale_auto")) {
    if(!patch.object.count("scale_auto")) {merged.object["scale_auto"].kind=Json::Boolean;merged.object["scale_auto"].number=0;}
    merged.object["adaptive_round_fraction"]=number(1);
  }
  if (patch.object.count("periodic") && !patch.object.count("boundary")) {
    merged.object["boundary"].kind=Json::String;
    merged.object["boundary"].string=patch["periodic"].flag() ? "periodic" : "none";
  }
  if (patch.object.count("periodic") && patch.object.count("boundary") &&
      patch["periodic"].flag() != (patch["boundary"].str()=="periodic"))
    throw std::invalid_argument("Conflicting boundary settings");
  Settings next(merged);
  Settings resources=next;
  if (controller && !controller->archive.entries().empty()) {
    resources.json.object["controller_enabled"].kind=Json::Boolean;
    resources.json.object["controller_enabled"].number=1;
  }
  validate_resources(benchmark, resources);
  if (!next.cma()) {
    auto proposal = make_perturbation(benchmark, next.json);
    if (benchmark.stochastic && (next.potential_force || (next.algorithm == "gas" && next.gas_local_search)))
      throw std::invalid_argument("Stochastic objectives do not support gradient forces or local search");
  }
  algorithm->validate_settings(next);
  if (!settings.planning() && next.walkers > algorithm->population().n && next.algorithm != "graph" &&
      std::none_of(algorithm->population().alive.begin(), algorithm->population().alive.end(), [](uint8_t v) { return v != 0; }))
    throw std::invalid_argument("Cannot grow a population without alive donors");
  return next;
}
Json Session::preview_settings(const Json& patch) const { return validated_settings(patch).json; }
void Session::update_settings(const Json& patch) {
  Settings next = validated_settings(patch);
  if (stringify(next.json) == stringify(pending ? pending->json : settings.json)) return;
  Json event; event.kind = Json::Object;
  event.object["iteration"] = number(iteration);
  event.object["revision"] = number(settings_revision + 1);
  event.object["requested"] = next.json;
  event.object["state"].kind = Json::String;
  Json dynamics = next.json;
  dynamics.object["max_evaluations"] = settings.json["max_evaluations"];
  const bool budget_only = stringify(dynamics) == stringify(settings.json);
  if (algorithm->settings_boundary() || budget_only) {
    if (algorithm->settings_boundary()) algorithm->configure(next);
    if(controller) controller->configure(next,settings);
    settings = std::move(next);
    pending.reset();
    event.object["state"].string = "applied";
  } else {
    // The budget is admission control, independent of committed action semantics.
    auto queued = std::make_unique<Settings>(next);
    settings.max_evaluations = next.max_evaluations;
    settings.json.object["max_evaluations"] = number(next.max_evaluations);
    pending = std::move(queued);
    event.object["state"].string = "pending";
  }
  ++settings_revision;
  settings_event = std::move(event);
  config_json = stringify(settings.json);
  capture();
}
void Session::apply_pending() {
  if (!pending || !algorithm->settings_boundary()) return;
  algorithm->configure(*pending);
  if(controller) controller->configure(*pending,settings);
  settings = std::move(*pending);
  pending.reset();
  config_json = stringify(settings.json);
  settings_event.object["iteration"] = number(iteration);
  settings_event.object["state"].string = "applied";
}
void Session::set_population(int count, const std::string& policy) {
  Json patch; patch.kind = Json::Object;
  patch.object["walkers"] = number(count);
  patch.object["removal_policy"].kind = Json::String;
  patch.object["removal_policy"].string = policy;
  update_settings(patch);
}
bool Session::restart_ready() const {
  if(!controller || !algorithm->settings_boundary()) return false;
  const auto& p=algorithm->population();
  return controller->wants_restart(settings,algorithm->finished(),std::any_of(p.alive.begin(),p.alive.end(),[](uint8_t v){return v!=0;}));
}
uint64_t Session::next_evaluations() const {
  if(!restart_ready()) return algorithm->next_evaluations_upper_bound();
  Settings next(controller->next(settings).config);
  return initialization_cost(benchmark,next)+controller->archive.validation_cost(benchmark.stochastic)+(benchmark.stochastic?12:0);
}
void Session::restart() {
  auto choice=controller->next(settings);Settings next(choice.config);
  Settings resources=next;resources.json.object["controller_enabled"].kind=Json::Boolean;resources.json.object["controller_enabled"].number=1;
  validate_resources(benchmark,resources);
  auto prepared=std::make_unique<RunController>(*controller);
  prepared->finish(settings,algorithm->movement_geometry(),algorithm->refinement_results());
  const uint64_t start=benchmark.evaluations;
  prepared->begin(next,choice,start);
  prepared->archive.validate_imports(benchmark,prepared->placement,settings.max_evaluations);
  int refinement=-1;
  if(settings.json["basin_avoidance"].flag(true) && choice.regime=="focused" && !prepared->archive.entries().empty() && prepared->placement.uniform01()<.2) {
    const auto& entries=prepared->archive.entries();
    for(size_t i=0;i<entries.size();++i) if(entries[i].validated &&
      (refinement<0 || settings.better(entries[i].objective,entries[refinement].objective))) refinement=int(i);
  }
  benchmark.initialization=[&](float* x){prepared->archive.place(x,prepared->placement,refinement,settings.json["basin_avoidance"].flag(true));};
  benchmark.observed=[&](const double* x,double value){prepared->observe(x,value);};
  std::unique_ptr<Algorithm> replacement;
  try { replacement=factories().at(settings.algorithm)(benchmark,next);
  Json warm;warm.kind=Json::Array;
  for(const auto& entry:prepared->archive.entries()) if(settings.json["basin_avoidance"].flag(true) && entry.validated && entry.geometry.kind==Json::Object) {
    const auto& population=replacement->population();bool nearby=false;
    for(int i=0;i<population.n && !nearby;++i) {
      std::vector<double> point(population.x.begin()+size_t(i)*population.d,population.x.begin()+size_t(i+1)*population.d);
      nearby=prepared->archive.distance(point,entry.position)<=.1;
    }
    if(nearby) warm.array.push_back(entry.geometry);
  }
  replacement->restore_movement_geometry(warm);
  } catch(...) {benchmark.initialization={};benchmark.observed=[this](const double* x,double value){controller->observe(x,value);};throw;}
  benchmark.initialization={};controller=std::move(prepared);algorithm=std::move(replacement);settings=std::move(next);
  benchmark.observed=[this](const double* x,double value){controller->observe(x,value);};
  config_json=stringify(settings.json);
}
std::string Session::export_basins() const {
  if(!controller) throw std::invalid_argument("CMA-ES does not use the fractal basin archive");
  return controller->archive.export_json();
}
void Session::import_basins(const std::string& text) {
  if(!controller) throw std::invalid_argument("CMA-ES does not use the fractal basin archive");
  Settings resources=settings;resources.json.object["controller_enabled"].kind=Json::Boolean;resources.json.object["controller_enabled"].number=1;
  validate_resources(benchmark,resources);
  controller->archive.import_json(text);
  ++settings_revision;settings_event.kind=Json::Object;
  settings_event.object["iteration"]=number(iteration);settings_event.object["revision"]=number(settings_revision);
  settings_event.object["state"].kind=Json::String;settings_event.object["state"].string="archive_imported";
}
void Session::step() {
  if(restart_ready()) {
    if(settings.max_evaluations && (benchmark.evaluations>=settings.max_evaluations || next_evaluations()>settings.max_evaluations-benchmark.evaluations))
      throw std::runtime_error("Budget cannot admit the complete next round initialization");
    restart();++iteration;capture();return;
  }
  if (algorithm->finished())
    throw std::runtime_error("Optimizer finished; reset to start a new run");
  if (settings.max_evaluations &&
      (benchmark.evaluations >= settings.max_evaluations ||
       next_evaluations() > settings.max_evaluations - benchmark.evaluations))
    throw std::runtime_error(
        "Evaluation budget reached: paused before the next complete step would "
        "exceed it. Apply a larger live budget to continue.");
  const auto& p = algorithm->population();
  if (std::none_of(p.alive.begin(), p.alive.end(), [](uint8_t v) { return v != 0; }))
    throw std::runtime_error(
        "All walkers are invalid or outside the domain. "
        "Reset or change bounds/time step.");
  algorithm->step();
  if(controller) ++controller->round_steps;
  ++iteration;
  apply_pending();
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
  if (settings.better(benchmark.best_observed, best)) best = benchmark.best_observed;
  snapshot = {1,
              double(p.n),
              double(p.d),
              double(p.has_velocity),
              double(iteration),
              double(benchmark.evaluations),
              double(alive),
              double(cloned),
              current,
              best,
              alive ? mean / alive : INFINITY,
              double(best_index)};
  snapshot.reserve(12 + size_t(p.n) * (2 * p.d + 8));
  const double* precise = algorithm->precise_positions();
  for (int i = 0; i < p.n; ++i) {
    for (int k = 0; k < p.d; ++k)
      snapshot.push_back(precise ? precise[size_t(i) * p.d + k] : p.x[size_t(i) * p.d + k]);
    for (int k = 0; k < p.d; ++k) snapshot.push_back(p.v[size_t(i) * p.d + k]);
    snapshot.insert(snapshot.end(), {p.objective[i], double(p.fitness[i]), double(p.alive[i]),
                                     double(p.companions[i]), double(p.clone_companions[i]),
                                     double(p.parent[i]), double(p.cloned[i]), double(p.leaf[i])});
  }
}
}  // namespace fg::optimization
