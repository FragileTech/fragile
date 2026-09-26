#include <algorithm>
#include <limits>
#include <numeric>

#include "optimization/engine.hpp"
#include "optimization/adaptive.hpp"

namespace fg::optimization {
Settings::Settings(const Json& input) : json(input) {
  distance_metric = parse_distance_metric(input["distance_metric"].str("l2"));
  json.object["distance_metric"].kind = Json::String;
  json.object["distance_metric"].string = distance_metric_name(distance_metric);
  auto i = [&](const char* k, int f, int lo, int hi) {
    int v = integer(json[k], f, lo, hi, k);
    json.object[k] = number(v);
    return v;
  };
  auto f = [&](const char* k, double v, double lo, double hi) {
    v = bounded(json[k], v, lo, hi, k);
    json.object[k] = number(v);
    return v;
  };
  auto b = [&](const char* k, bool v) {
    v = json[k].flag(v);
    Json j;
    j.kind = Json::Boolean;
    j.number = v;
    json.object[k] = j;
    return v;
  };
  auto s = [&](const char* k, const char* v) {
    std::string t = json[k].str(v);
    Json j;
    j.kind = Json::String;
    j.string = t;
    json.object[k] = j;
    return t;
  };
  algorithm = s("algorithm", "euclidean");
  double budget = f("max_evaluations", 0, 0, 1e12);
  if (std::floor(budget) != budget)
    throw std::invalid_argument("Evaluation budget must be an integer");
  max_evaluations = uint64_t(budget);
  b("controller_enabled",false);b("population_auto",true);b("scale_auto",true);b("basin_avoidance",true);
  i("restart_token",0,0,1000000000);
  if(json["controller_enabled"].flag() && (!max_evaluations || cma()))
    throw std::invalid_argument("Automatic fractal restarts require a positive evaluation budget and a fractal algorithm");
  objective = s("objective", "minimize");
  if (objective != "minimize" && objective != "maximize")
    throw std::invalid_argument("Objective must be minimize or maximize");
  perturbation = s("perturbation", algorithm == "gas" ? "gas_adaptive" : "gaussian");
  gas_tabu = b("gas_tabu", true);
  gas_local_search = b("gas_local_search", true);
  gas_local_evaluations = i("gas_local_evaluations", 200, 1, 1000000);
  companion = s("companion", "cloning");
  clone_companion = s("clone_companion", "cloning");
  for (auto& name : {companion, clone_companion})
    if (name != "cloning" && name != "softmax" && name != "uniform" && name != "random_pairing" &&
        name != "greedy_pairing")
      throw std::invalid_argument("Unknown companion strategy");
  walkers = i("walkers", 256, 2, 100000);
  const bool live_population = algorithm == "wave" || planning();
  max_walkers = i("max_walkers", live_population ? walkers : std::max(10000, walkers), walkers,
                  live_population ? 100000 : 1000000);
  fractal::removal_policy(s("removal_policy", "virtual_reward"));
  freeze_prefix_after = i("freeze_prefix_after", 0, 0, 1000000);
  seed = i("seed", 7, 0, 2147483647);
  if(json["round_seed"].kind!=Json::Null) seed=integer(json["round_seed"],seed,0,2147483647,"round seed");
  dt_min = i("dt_min", 1, 1, 100);
  dt_max = i("dt_max", 1, dt_min, 100);
  clone_every = i("clone_every", 1, 1, 100000);
  substeps = i("substeps", 1, 1, 100);
  elites = i("elites", 0, 0, walkers);
  horizon = i("horizon", 32, 1, 4096);
  max_horizon = i("max_horizon", 0, 0, 4096);
  consensus_prefix = b("consensus_prefix", true);
  proposal = f("proposal", .025, 0, 1);
  // Preserve the scale of recordings made before perturbations were explicit.
  const double legacy_std = input["proposal"].kind != Json::Null && algorithm != "euclidean"
                                ? proposal * (input["high"].num(5.12) - input["low"].num(-5.12))
                                : 1;
  f("perturbation_std", legacy_std, 0, 1e6);
  f("covariance_learning_rate", .1, 0, 1);
  const bool controlled=b("gas_scale_controlled",json["controller_enabled"].flag(false));
  if(json["controller_enabled"].flag(false)) json.object["gas_scale_controlled"].number=1;
  f("gas_scale_multiplier",1,0,1e6);
  if(controlled || json["controller_enabled"].flag(false)) json.object["gas_scale_multiplier"]=json["perturbation_std"];
  for(auto key:{"adaptive_active","adaptive_paths","adaptive_scale","adaptive_difference","adaptive_pairs","adaptive_mixture"}) b(key,true);
  b("cloning_geometry",true);b("cloning_drift",true);f("cloning_drift_strength",.25,0,1);
  const double minimum_scale=f("adaptive_min_scale",.0001,0,1e6);
  f("adaptive_max_scale",1,minimum_scale,1e6);
  f("adaptive_round_fraction",1,0,1);
  const auto mode=s("adaptive_euclidean_mode","velocity");
  if(mode!="velocity" && mode!="position") throw std::invalid_argument("Unknown Euclidean adaptive movement mode");
  gamma = f("gamma", 1, 1e-9, 1e6);
  beta = f("beta", 1, 1e-9, 1e12);
  delta_t = f("delta_t", .002, 1e-9, 1);
  epsilon = f("epsilon", .1, 1e-9, 1e6);
  clone_epsilon = f("clone_epsilon", .1, 1e-9, 1e6);
  lambda_alg = f("lambda_alg", 0, 0, 1e6);
  reward_coef = f("reward_coef", 1, 0, 10);
  distance_coef = f("distance_coef", 1, 0, 10);
  eta = f("eta", .1, 0, 100);
  sigma_min = f("sigma_min", 1e-8, 1e-12, 1e6);
  amplitude = f("amplitude", 2, 1e-9, 100);
  epsilon_dist = f("epsilon_dist", 1e-8, 0, 100);
  rho = f("rho", 0, 0, 1e6);
  p_max = f("p_max", 1, 1e-9, 1);
  epsilon_clone = f("epsilon_clone", 1e-6, 1e-12, 100);
  sigma_x = f("sigma_x", 1e-6, 0, 1e6);
  restitution = f("restitution", .5, 0, 1);
  periodic = b("periodic", false);
  const auto boundary = s("boundary", periodic ? "periodic" : (cma() ? "cma" : "none"));
  if (boundary!="none" && boundary!="periodic" && boundary!="cma")
    throw std::invalid_argument("Unknown boundary handling");
  if (cma() && boundary!="cma") throw std::invalid_argument("CMA-ES requires bounded domains and its boundary mapping");
  periodic = boundary=="periodic";
  json.object["periodic"].number = periodic;
  potential_force = b("potential_force", input["coco_version"].kind == Json::Null);
  cloning = b("cloning", true);
  kinetic = b("kinetic", true);
}
Population resized_population(const Population& old, const Settings& s, Rng& rng) {
  if (s.walkers == old.n) return old;
  std::vector<int> rows(old.n), donors;
  std::iota(rows.begin(), rows.end(), 0);
  for (int i = 0; i < old.n; ++i) if (old.alive[i]) donors.push_back(i);
  if (s.walkers > old.n && donors.empty())
    throw std::invalid_argument("Cannot grow a population without alive donors");
  if (s.walkers < old.n) {
    const bool fitness = s.json["removal_policy"].str() == "virtual_reward";
    auto score = [&](int i) {
      const double value = fitness ? old.fitness[i] : s.score(old.objective[i]);
      return old.alive[i] && std::isfinite(value) ? value : -INFINITY;
    };
    std::stable_sort(rows.begin(), rows.end(), [&](int a, int b) { return score(a) > score(b); });
  }
  Population result;
  result.resize(s.walkers, old.d);
  result.has_velocity = old.has_velocity;
  for (int i = 0; i < result.n; ++i) {
    const int from = i < old.n ? rows[i] : donors[rng.randint(0, donors.size())];
    std::copy_n(old.x.data() + size_t(from) * old.d, old.d, result.x.data() + size_t(i) * old.d);
    std::copy_n(old.v.data() + size_t(from) * old.d, old.d, result.v.data() + size_t(i) * old.d);
    result.objective[i] = old.objective[from];
    result.lineage[i] = old.lineage[from];
    result.fitness[i] = old.fitness[from];
    result.alive[i] = old.alive[from];
    result.leaf[i] = old.leaf[from];
  }
  return result;
}
namespace {
fractal::EuclideanConfig euclidean_config(const Settings& s, const Benchmark* benchmark = nullptr) {
  fractal::EuclideanConfig c;
  c.walkers = s.walkers;
  c.clone_every = s.clone_every;
  c.substeps = s.substeps;
  c.companion = s.companion;
  c.clone_companion = s.clone_companion;
  c.periodic = s.periodic;
  if (benchmark && s.json["boundary"].str()=="cma")
    c.repair=[benchmark](float* x) { benchmark->boundary(x,"cma"); };
  c.potential_force = s.potential_force;
  c.cloning = s.cloning;
  c.kinetic = s.kinetic;
  c.lambda_alg = s.lambda_alg;
  c.epsilon = s.epsilon;
  c.clone_epsilon = s.clone_epsilon;
  c.epsilon_dist = s.epsilon_dist;
  c.rho = s.rho;
  c.sigma_min = s.sigma_min;
  c.amplitude = s.amplitude;
  c.eta = s.eta;
  c.reward_coef = s.reward_coef;
  c.distance_coef = s.distance_coef;
  c.sigma_x = s.sigma_x;
  c.restitution = s.restitution;
  c.gamma = s.gamma;
  c.delta_t = s.delta_t;
  c.beta = s.beta;
  c.epsilon_clone = s.epsilon_clone;
  c.p_max = s.p_max;
  c.minimize = s.objective == "minimize";
  return c;
}
struct ObjectiveDomain {
  Benchmark& benchmark;
  int d;
  double low, high;
  bool stochastic;
  explicit ObjectiveDomain(Benchmark& b)
      : benchmark(b), d(b.d), low(b.low), high(b.high), stochastic(b.stochastic) {}
  double evaluate(const float* x, Rng* rng) const {
    return benchmark.evaluate_optimization(x, rng);
  }
  void gradient(const float* x, float* out) const { benchmark.gradient(x, out, true); }
  uint64_t gradient_evaluations() const { return benchmark.gradient_evaluations(); }
  uint64_t evaluations() const { return benchmark.evaluations; }
  bool valid(const float* x) const { return benchmark.valid(x); }
  void wrap(float* x) const { benchmark.wrap(x); }
  void initial(float* x, Rng& rng) const { benchmark.initial(x, rng); }
};
// Replay a complete frozen kick sequence in the shared BAOAB integrator.
// The negative branch mirrors only stochastic kicks, never forces or velocity.
class FrozenKicks final : public Perturbation {
  const std::vector<float>& kicks;
  int sign;
  mutable size_t cursor=0;
 public:
  FrozenKicks(const std::vector<float>& values,int branch):kicks(values),sign(branch?-1:1) {}
  void sample(const float*,float* out,int d,Rng&) const override {
    if(cursor+size_t(d)>kicks.size()) throw std::logic_error("Kinetic replay exhausted its frozen kicks");
    for(int j=0;j<d;++j) out[j]=sign*kicks[cursor++];
  }
};
class EuclideanAdapter final : public Algorithm {
  ObjectiveDomain domain;
  Settings settings;
  OptimizationRng rng;
  std::unique_ptr<Perturbation> proposal;
  fractal::Euclidean<ObjectiveDomain> core;
  uint64_t trial_sequence=0;

 public:
  EuclideanAdapter(Benchmark& b, const Settings& s)
      : domain(b),
        settings(s),
        rng(s.seed),
        proposal(make_perturbation(b, s.json)),
        core(domain, euclidean_config(s, &b), rng, *proposal) {}
  void configure(const Settings& next) override {
    auto noise = retune_perturbation(*proposal, domain.benchmark, settings.json, next.json);
    auto random = rng;
    auto population = resized_population(core.population(), next, random);
    auto config = euclidean_config(next, &domain.benchmark);
    if(evaluated_perturbation(next.perturbation) && next.json["adaptive_euclidean_mode"].str()=="position" &&
       (!evaluated_perturbation(settings.perturbation) || settings.json["adaptive_euclidean_mode"].str()!="position"))
      std::fill(population.v.begin(),population.v.end(),0);
    Settings saved = next;
    core.configure(std::move(config), std::move(population), *noise);
    proposal = std::move(noise);
    rng = random;
    settings = std::move(saved);
  }
  void step() override {
    begin_geometry(*proposal);
    const bool direct=settings.json["adaptive_euclidean_mode"].str()=="position";
    if(!proposal->tracks_trials() || (!direct&&!settings.kinetic)) { core.step();if(proposal->observer) proposal->observer->update();return; }
    const auto& frozen=core.population();
    proposal->observed_freeze({frozen.d,frozen.x,frozen.lineage,frozen.alive,frozen.objective});
    core.step_with_movement([&](Population& population) {
      for(int i=0;i<population.n;++i) {
        const uint64_t action=++trial_sequence;
        float* x=population.x.data()+size_t(i)*population.d;
        // Cloning includes position jitter; the stored objective therefore
        // cannot be reused as an evaluation of this exact origin.
        const fractal::TrialIdentity identity{uint64_t(settings.json["round_id"].num()),0,population.lineage[i],action,0};
        EvaluatedProposal result;
        if(direct) result=evaluate_adaptive_position(*proposal,domain.benchmark,settings.json,x,NAN,rng,identity);
        else {
          Population start;start.resize(1,population.d);
          std::copy_n(x,population.d,start.x.data());
          std::copy_n(population.v.data()+size_t(i)*population.d,population.d,start.v.data());
          result=evaluate_adaptive_trial(*proposal,domain.benchmark,settings.json,x,NAN,rng,identity,settings.substeps,
            [&](const std::vector<float>& kicks,int branch) {
              Population candidate=start;FrozenKicks noise(kicks,branch);
              fractal::baoab(candidate,euclidean_config(settings, &domain.benchmark),domain,rng,&noise);
              EvaluatedProposal state;state.position=std::move(candidate.x);state.velocity=std::move(candidate.v);return state;
            });
        }
        std::copy(result.position.begin(),result.position.end(),x);
        if(direct) std::fill_n(population.v.data()+size_t(i)*population.d,population.d,0);
        else std::copy(result.velocity.begin(),result.velocity.end(),population.v.data()+size_t(i)*population.d);
        population.objective[i]=result.objective;population.alive[i]=result.valid;
      }
    });
    proposal->observed_update();
  }
  void set_geometry_diagnostics(bool enabled) override {enable_geometry(*proposal,domain.benchmark,settings.json,enabled);}
  Json movement_geometry() const override {return perturbation_geometry(*proposal);}
  void restore_movement_geometry(const Json& geometry) override {restore_perturbation_geometry(*proposal,geometry);}
  Json metadata() const override {
    Json result;result.kind=Json::Object;result.object["exploration"]=perturbation_diagnostics(*proposal);result.object["geometry"]=geometry_diagnostics(*proposal);return result;
  }
  const Population& population() const override { return core.population(); }
  uint64_t evaluations() const override { return core.evaluations(); }
  uint64_t next_evaluations_upper_bound() const override {
    if(!proposal->tracks_trials()) return core.next_evaluations_upper_bound();
    const bool direct=settings.json["adaptive_euclidean_mode"].str()=="position";
    const uint64_t gradients=!direct&&settings.kinetic&&settings.potential_force&&!domain.stochastic ?
      2*uint64_t(settings.substeps)*domain.gradient_evaluations() : 0;
    return uint64_t(core.population().n)*(adaptive_evaluation_bound(domain.benchmark,settings.json,true)+
      gradients*(1+settings.json["adaptive_pairs"].flag(true)));
  }
  double objective_score(int i) const override { return core.objective_score(i); }
};
}  // namespace
std::vector<int32_t> select_companions(const Population& p, const Settings& s, const Benchmark& b,
                                       const std::string& method, double epsilon, Rng& rng) {
  return fractal::select_companions(p, euclidean_config(s), b, method, epsilon, rng);
}
std::vector<float> fitness(const Population& p, const Settings& s, const Benchmark& b,
                           const std::vector<int32_t>& companions) {
  return fractal::fitness(p, euclidean_config(s), b, companions);
}
void clone_population(Population& p, const Settings& s, const std::vector<int32_t>& companions,
                      const std::vector<uint8_t>& mask, Rng& rng) {
  fractal::clone_population(p, euclidean_config(s), companions, mask, rng);
}
void baoab(Population& p, const Settings& s, const Benchmark& b, Rng& rng,
           const Perturbation* noise) {
  auto& mutable_b = const_cast<Benchmark&>(b);
  ObjectiveDomain domain(mutable_b);
  fractal::baoab(p, euclidean_config(s, &b), domain, rng, noise);
}
std::unique_ptr<Algorithm> make_euclidean(Benchmark& b, const Settings& s) {
  return std::make_unique<EuclideanAdapter>(b, s);
}
}  // namespace fg::optimization
