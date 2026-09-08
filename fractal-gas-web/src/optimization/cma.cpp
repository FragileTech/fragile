#include "optimization/engine.hpp"

#include <libcmaes/bipopcmastrategy.h>
#include <libcmaes/acovarianceupdate.h>
#include <libcmaes/pwq_bound_strategy.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace fg::optimization {
namespace {
constexpr const char* revision = "6a53cd562c85dc6df3b8df317be3ecb415240c2d";
using Transform = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
using Parameters = libcmaes::CMAParameters<Transform>;
// Upstream Eigen functor copies intentionally share a static RNG. Preserve its
// exact draw semantics while isolating interleaved Lab sessions. Lab calls are
// serialized (one worker); restore external upstream callers' state as well.
struct SamplerScope {
  std::mt19937& owned;
  std::mt19937 previous;
  static std::mt19937& global() { return Eigen::internal::scalar_normal_dist_op<double>::rng; }
  explicit SamplerScope(std::mt19937& state) : owned(state), previous(global()) { global() = owned; }
  ~SamplerScope() { owned = global(); global() = previous; }
};
using Base = libcmaes::BIPOPCMAStrategy<libcmaes::ACovarianceUpdate, Transform>;
// Expose upstream restart primitives, without replacing the sampler or updates.
class Strategy final : public Base {
  uint64_t large_population;
 public:
  Strategy(libcmaes::FitFunc& f, Parameters& p) : Base(f, p), large_population(p.lambda()) {}
  uint64_t largest_restart(bool small) const { return large_population * (small ? 1 : 2); }
  void restart(bool small) {
    if (small) r2(); else { r1(); large_population = get_parameters().lambda(); }
    reset_search_state();
  }
};
void put_string(Json& j, const char* key, const std::string& value) {
  j.object[key].kind = Json::String;
  j.object[key].string = value;
}
class Cma final : public Algorithm {
  Benchmark& benchmark;
  Settings settings;
  OptimizationRng objective_rng;
  std::mt19937 sampler;
  libcmaes::FitFunc function;
  std::unique_ptr<Strategy> strategy;
  Population pop;
  std::vector<double> positions;
  Json resolved = JsonReader(std::string("{}")).read();
  bool bipop, done = false, small_run = false;
  int large_runs = 1, runs_limit, restarts = 0, displayed_restart = 0;
  uint64_t generation = 0, run_evaluations = 0, small_cap = 0;
  uint64_t budgets[2] = {0, 0};
  double displayed_sigma = 0;
  std::string reason;
  int evaluated_row = 0;
  int lambda() const { return strategy->get_parameters().lambda(); }
  void check_memory(uint64_t count) const {
    // Dense eigensolver and covariance workspaces, plus genotype/phenotype rows.
    const uint64_t d = benchmark.d;
    if (count < 2 || 128 * d * d + 64 * d * uint64_t(count) > 128 * 1024 * 1024)
      throw std::invalid_argument("CMA-ES exceeds 128 MiB state budget; reduce dimensions or population");
  }
  void prepare_next() {
    const bool stopped = strategy->stop();
    const bool cap_reached = small_run && run_evaluations + uint64_t(lambda()) > small_cap;
    if (!stopped && !cap_reached) return;
    reason = cap_reached ? "Small-run evaluation allowance reached" : strategy->get_solutions().status_msg();
    if (!bipop) { done = true; return; }
    budgets[small_run ? 1 : 0] += run_evaluations;
    if (large_runs >= runs_limit && !small_run) {
      done = true; reason = "Large-population runs exhausted"; return;
    }
    small_run = budgets[0] > budgets[1];
    check_memory(strategy->largest_restart(small_run));
    strategy->restart(small_run);
    small_cap = budgets[0] / 2;
    // A small allowance cannot be spent on a partial generation. Skip this
    // small phase instead of repeatedly creating runs with zero evaluations.
    if (small_run && uint64_t(lambda()) > small_cap) {
      small_run = false;
      check_memory(strategy->largest_restart(false));
      strategy->restart(false);
    }
    if (!small_run) ++large_runs;
    ++restarts;
    run_evaluations = 0;
    check_memory(lambda());
    reason.clear();
  }
 public:
  Cma(Benchmark& b, const Settings& s)
      : benchmark(b), settings(s), objective_rng(uint64_t(s.seed) ^ 0xd1b54a32d192ed03ULL),
        bipop(s.algorithm == "cmaes_bipop"),
        runs_limit(integer(s.json["cma_runs"], 9, 1, 1000, "large-population runs")) {
    if (s.periodic) throw std::invalid_argument("CMA-ES supports bounded domains only; disable periodic wrapping");
    int count = integer(s.json["cma_population"], 0, 0, 100000, "initial CMA population");
    if (count == 1) throw std::invalid_argument("Initial CMA population must be 0 or at least 2");
    double sigma = bounded(s.json["cma_sigma"], 0, 0, 1e6, "initial CMA standard deviation");
    if (sigma == 0) sigma = .2 * (b.high - b.low);
    displayed_sigma = sigma;
    check_memory(count ? count : 4 + int(3 * std::log(b.d)));
    OptimizationRng initial_rng(s.seed);
    std::vector<float> initial(b.d);
    b.initial(initial.data(), initial_rng);
    std::vector<double> mean(initial.begin(), initial.end()), low(b.d, b.low), high(b.d, b.high);
    Transform transform(low.data(), high.data(), b.d);
    // Upstream uses zero to request wall-clock seeding. Map it explicitly.
    const uint64_t sampler_seed = uint64_t(s.seed) + 1;
    Parameters params(mean, sigma, count ? count : -1, sampler_seed, transform);
    params.set_quiet(true);
    params.set_max_fevals(-1); // Lab enforces global and small-run admission.
    params.set_initial_fvalue(false);
    params.set_mt_feval(false);
    function = [this](const double* x, const int d) {
      const int row = evaluated_row++;
      const double value = benchmark.evaluate_optimization(x, &objective_rng);
      std::copy_n(x, d, positions.data() + size_t(row) * d);
      for (int k = 0; k < d; ++k) pop.x[size_t(row) * d + k] = float(x[k]);
      pop.objective[row] = value;
      pop.alive[row] = benchmark.valid(x) && std::isfinite(value);
      return pop.alive[row] ? -settings.score(value) : std::numeric_limits<double>::max();
    };
    {
      SamplerScope scope(sampler);
      strategy = std::make_unique<Strategy>(function, params);
    }
    check_memory(lambda());
    if (s.max_evaluations && uint64_t(lambda()) > s.max_evaluations)
      throw std::invalid_argument("Evaluation budget is too small for the first complete CMA generation");
    resolved.object["cma_population"] = number(lambda());
    resolved.object["cma_sigma"] = number(sigma);
    resolved.object["cma_runs"] = number(runs_limit);
    resolved.object["cma_initial_mean"] = array(mean);
    resolved.object["cma_sampler_seed"] = number(sampler_seed);
    put_string(resolved, "libcmaes_revision", revision);
    put_string(resolved, "precision", "float64");
    step();
  }
  void step() override {
    if (done) return;
    const int count = lambda();
    if (settings.max_evaluations && uint64_t(count) > settings.max_evaluations - benchmark.evaluations)
      throw std::runtime_error("Evaluation budget reached before a complete CMA generation");
    pop.resize(count, benchmark.d);
    positions.resize(size_t(count) * benchmark.d);
    for (int i = 0; i < count; ++i) {
      pop.companions[i] = pop.clone_companions[i] = pop.parent[i] = i;
      pop.cloned[i] = 0;
    }
    evaluated_row = 0;
    dMat candidates;
    { SamplerScope scope(sampler); candidates = strategy->ask(); }
    auto phenotype = strategy->get_parameters().get_gp().pheno(candidates);
    strategy->eval(candidates, phenotype);
    displayed_restart = restarts;
    ++generation;
    run_evaluations += count;
    if (std::none_of(pop.alive.begin(), pop.alive.end(), [](uint8_t v) { return v != 0; })) {
      done = true; reason = "No valid CMA candidates; model update skipped"; return;
    }
    strategy->tell();
    strategy->inc_iter();
    displayed_sigma = strategy->get_solutions().sigma();
    prepare_next();
  }
  const Population& population() const override { return pop; }
  const double* precise_positions() const override { return positions.data(); }
  uint64_t evaluations() const override { return benchmark.evaluations; }
  bool finished() const override { return done; }
  uint64_t next_evaluations_upper_bound() const override { return done ? 0 : lambda(); }
  uint64_t next_population_size() const override { return done ? pop.n : lambda(); }
  double objective_score(int i) const override { return settings.score(pop.objective.at(i)); }
  Json resolved_config() const override { return resolved; }
  Json metadata() const override {
    Json j = JsonReader(std::string("{}")).read();
    j.object["generation"] = number(generation);
    j.object["population"] = number(pop.n);
    j.object["restarts"] = number(displayed_restart);
    j.object["sigma"] = number(displayed_sigma);
    put_string(j, "stop_reason", reason);
    return j;
  }
};
}  // namespace
std::unique_ptr<Algorithm> make_cma(Benchmark& b, const Settings& s) {
  return std::make_unique<Cma>(b, s);
}
}  // namespace fg::optimization
