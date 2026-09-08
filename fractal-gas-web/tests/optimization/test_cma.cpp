#include "optimization/engine.hpp"
#include <fstream>
#include <sstream>
#include "test_framework.hpp"
#include <libcmaes/cmastrategy.h>
#include <libcmaes/bipopcmastrategy.h>
#include <libcmaes/acovarianceupdate.h>
#include <libcmaes/pwq_bound_strategy.h>
using namespace fg::optimization;
static Json config(const std::string& s) { return JsonReader(s).read(); }

TEST_CASE(cma_double_precision_and_float_compatibility) {
  Benchmark b(config(R"({"benchmark":"rosenbrock","dimensions":2})"));
  double x[] = {1.0 + 1e-10, 1.0};
  float xf[] = {float(x[0]), float(x[1])};
  CHECK(b.evaluate(x) > 0);
  CHECK(b.evaluate(xf) == 0);
  CHECK(b.evaluations == 0);
  CHECK(b.evaluate_optimization(x) == b.evaluate(x));
  CHECK(b.evaluations == 1);
}
TEST_CASE(cma_upstream_active_generation_parity) {
  Session lab(config(R"({"algorithm":"cmaes_active","benchmark":"quadratic","dimensions":3,"seed":0,"periodic":false})"));
  auto resolved = config(lab.config_json);
  std::vector<double> mean;
  for (auto& v : resolved["cma_initial_mean"].array) mean.push_back(v.num());
  std::vector<double> low(3, lab.benchmark.low), high(3, lab.benchmark.high);
  using GP = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
  GP gp(low.data(), high.data(), 3);
  libcmaes::CMAParameters<GP> p(mean, resolved["cma_sigma"].num(), int(resolved["cma_population"].num()), 1, gp);
  p.set_quiet(true); p.set_initial_fvalue(false); p.set_mt_feval(false);
  libcmaes::FitFunc f = [&](const double* x, int) { return lab.benchmark.evaluate(x); };
  libcmaes::CMAStrategy<libcmaes::ACovarianceUpdate, GP> upstream(f, p);
  for (int generation = 0; generation < 20; ++generation) {
    if (generation) lab.step();
    auto genotype = upstream.ask();
    auto phenotype = gp.pheno(genotype);
    upstream.eval(genotype, phenotype); upstream.tell(); upstream.inc_iter();
    const auto& pop = lab.algorithm->population();
    const double* actual = lab.algorithm->precise_positions();
    for (int i = 0; i < pop.n; ++i) {
      for (int k = 0; k < pop.d; ++k) CHECK_CLOSE(actual[i * pop.d + k], phenotype(k, i), 1e-13);
      CHECK_CLOSE(pop.objective[i], lab.benchmark.evaluate(actual + i * pop.d), 1e-13);
    }
    CHECK(lab.benchmark.evaluations == uint64_t((generation + 1) * pop.n));
  }
}
TEST_CASE(cma_reset_bounds_directions_and_generation_budgets) {
  for (auto id : {"cmaes_active", "cmaes_bipop"}) {
    for (auto direction : {"minimize", "maximize"}) {
      auto cfg = config(std::string(R"({"algorithm":")") + id + R"(","objective":")" + direction + R"(","benchmark":"quadratic","seed":0,"dimensions":2,"periodic":false,"cma_population":8,"max_evaluations":24})");
      Session a(cfg), b(cfg);
      CHECK(a.iteration == 0); CHECK(a.benchmark.evaluations == 8);
      for (int step = 0; step < 2; ++step) { a.step(); b.step(); CHECK(a.snapshot == b.snapshot); }
      CHECK(a.benchmark.evaluations == 24);
      CHECK(config(a.status_json())["budget_exhausted"].flag());
      bool rejected = false; try { a.step(); } catch (const std::exception&) { rejected = true; }
      CHECK(rejected); CHECK(a.benchmark.evaluations == 24);
      const auto& pop = a.algorithm->population();
      for (int i = 0; i < pop.n; ++i) CHECK(a.benchmark.valid(a.algorithm->precise_positions() + i * pop.d));
    }
  }
  for (const auto& text : {R"({"algorithm":"cmaes_active","periodic":true})", R"({"algorithm":"cmaes_active","max_evaluations":3})", R"({"algorithm":"cmaes_active","cma_population":1})"}) {
    bool rejected = false; try { Session bad(config(text)); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected);
  }
}
TEST_CASE(cma_bipop_restarts_and_convergence) {
  auto cfg = config(R"({"algorithm":"cmaes_bipop","benchmark":"quadratic","dimensions":2,"seed":4,"cma_runs":2,"max_evaluations":50000})");
  Session a(cfg), b(cfg);
  int largest = 0;
  for (int i = 0; i < 4000 && !a.algorithm->finished(); ++i) {
    a.step(); b.step(); CHECK(a.snapshot == b.snapshot);
    largest = std::max(largest, a.algorithm->population().n);
  }
  CHECK(a.algorithm->finished());
  CHECK(largest >= 12);
  CHECK(a.best < 1e-10);
  CHECK(a.algorithm->next_evaluations_upper_bound() == 0);
  CHECK(config(a.status_json())["stop_reason"].str() == "Large-population runs exhausted");
}

TEST_CASE(cma_seeded_bipop_batch_schedule_parity) {
  Session lab(config(R"({"algorithm":"cmaes_bipop","benchmark":"quadratic","dimensions":2,"seed":0,"cma_runs":3})"));
  auto resolved = config(lab.config_json);
  std::vector<double> mean;
  for (auto& v : resolved["cma_initial_mean"].array) mean.push_back(v.num());
  std::vector<double> low(2, lab.benchmark.low), high(2, lab.benchmark.high);
  using GP = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
  GP gp(low.data(), high.data(), 2);
  libcmaes::CMAParameters<GP> p(mean, resolved["cma_sigma"].num(), int(resolved["cma_population"].num()), 1, gp);
  p.set_quiet(true); p.set_initial_fvalue(false); p.set_mt_feval(false); p.set_restarts(3);
  std::vector<double> reference, actual;
  libcmaes::FitFunc f = [&](const double* x, int d) {
    reference.insert(reference.end(), x, x+d);
    return lab.benchmark.evaluate(x);
  };
  libcmaes::BIPOPCMAStrategy<libcmaes::ACovarianceUpdate, GP> upstream(f, p);
  // Match the Lab's explicit whole-generation admission rule. The upstream
  // batch driver otherwise checks small-run caps only after an overshoot.
  libcmaes::ProgressFunc<libcmaes::CMAParameters<GP>,libcmaes::CMASolutions> admission =
    [](const auto& params, const auto& solution) {
      return params.get_max_fevals() > 0 && solution.nevals() + params.lambda() > params.get_max_fevals() ? 1 : 0;
    };
  upstream.set_progress_func(admission);
  upstream.optimize();
  for (int i = 0; i < 10000; ++i) {
    const auto& pop = lab.algorithm->population();
    const auto* x = lab.algorithm->precise_positions();
    actual.insert(actual.end(), x, x + pop.n * pop.d);
    if (lab.algorithm->finished()) break;
    lab.step();
  }
  CHECK(lab.algorithm->finished());
  CHECK(actual.size() == reference.size());
  CHECK(lab.benchmark.evaluations * 2 == reference.size());
  for (size_t i = 0; i < actual.size(); ++i) CHECK_CLOSE(actual[i], reference[i], 1e-12);
}
TEST_CASE(cma_invalid_generation_and_recorded_coordinates) {
  Session lab(config(R"({"algorithm":"cmaes_active","benchmark":"quadratic","dimensions":2,"seed":0})"));
  const double* x = lab.algorithm->precise_positions();
  for (int i = 0; i < lab.algorithm->population().n; ++i) {
    CHECK(lab.snapshot[12 + i * 12] == x[2*i]);
    CHECK(lab.snapshot[13 + i * 12] == x[2*i+1]);
    CHECK(lab.snapshot[16 + i * 12] == lab.benchmark.evaluate(x + 2*i));
  }
  Session invalid(config(R"({"algorithm":"cmaes_active","benchmark":"gaussian_mixture","dimensions":2,"n_gaussians":1,"centers":[0,0],"stds":[1e-300,1e-300]})"));
  CHECK(invalid.algorithm->finished());
  CHECK(invalid.benchmark.evaluations == uint64_t(invalid.algorithm->population().n));
  CHECK(config(invalid.status_json())["stop_reason"].str() == "No valid CMA candidates; model update skipped");
}

TEST_CASE(cma_native_double_coordinate_fixtures) {
  std::ifstream file(CMA_FIXTURES);
  CHECK(file.good());
  std::stringstream text; text << file.rdbuf();
  for (const auto& fixture : config(text.str()).array) {
    Benchmark benchmark(fixture["config"]);
    std::vector<double> x;
    for (const auto& v : fixture["x"].array) x.push_back(v.num());
    const double expected = fixture["expected"].num();
    CHECK(std::abs(benchmark.evaluate(x.data()) - expected) <= 1e-12 * std::max(1e-25, std::abs(expected)));
  }
}
