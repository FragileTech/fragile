#include <algorithm>
#include <limits>
#include <numeric>

#include "optimization/engine.hpp"

namespace fg::optimization {
Settings::Settings(const Json& input) : json(input) {
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
  max_walkers = i("max_walkers", std::max(10000, walkers), walkers, 1000000);
  seed = i("seed", 7, 0, 2147483647);
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
  potential_force = b("potential_force", input["coco_version"].kind == Json::Null);
  cloning = b("cloning", true);
  kinetic = b("kinetic", true);
}
namespace {
fractal::EuclideanConfig euclidean_config(const Settings& s) {
  fractal::EuclideanConfig c;
  c.walkers = s.walkers;
  c.clone_every = s.clone_every;
  c.substeps = s.substeps;
  c.companion = s.companion;
  c.clone_companion = s.clone_companion;
  c.periodic = s.periodic;
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
class EuclideanAdapter final : public Algorithm {
  ObjectiveDomain domain;
  OptimizationRng rng;
  std::unique_ptr<Perturbation> proposal;
  fractal::Euclidean<ObjectiveDomain> core;

 public:
  EuclideanAdapter(Benchmark& b, const Settings& s)
      : domain(b),
        rng(s.seed),
        proposal(make_perturbation(b, s.json)),
        core(domain, euclidean_config(s), rng, *proposal) {}
  void step() override { core.step(); }
  const Population& population() const override { return core.population(); }
  uint64_t evaluations() const override { return core.evaluations(); }
  uint64_t next_evaluations_upper_bound() const override {
    return core.next_evaluations_upper_bound();
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
  fractal::baoab(p, euclidean_config(s), domain, rng, noise);
}
std::unique_ptr<Algorithm> make_euclidean(Benchmark& b, const Settings& s) {
  return std::make_unique<EuclideanAdapter>(b, s);
}
}  // namespace fg::optimization
