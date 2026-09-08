#include <algorithm>

#include "optimization/gas2017.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
namespace {
Json config(const std::string& text) { return JsonReader(text).read(); }
class Draws : public fg::Rng {
 public:
  std::vector<int> integers;
  size_t next = 0;
  float uniform01() override { return 0; }
  int64_t randint(int64_t low, int64_t high) override {
    const int64_t value = next < integers.size() ? integers[next++] : low;
    CHECK(value >= low && value < high);
    return value;
  }
};
class ConstantNoise : public Perturbation {
 public:
  mutable int calls = 0;
  float value = 2;
  void sample(const float*, float* out, int d, fg::Rng&) const override {
    ++calls;
    std::fill_n(out, d, value);
  }
};
Population population() {
  Population p;
  p.resize(3, 1);
  p.x = {0, 1, 3};
  p.objective = {0, 1, 2};
  p.alive = {1, 1, 1};
  return p;
}
}  // namespace
TEST_CASE(gas_normalization_flow_and_snapshot_cloning) {
  Settings s(config("{}"));
  auto p = population();
  CHECK(gas2017::normalize(p, s) == std::vector<double>({0, .5, 1}));
  Settings maximize(config(R"({"objective":"maximize"})"));
  CHECK(gas2017::normalize(p, maximize) == std::vector<double>({1, .5, 0}));
  Draws rng;
  auto f = gas2017::flows(p, s, nullptr, rng);
  CHECK(f == std::vector<double>({1, 2.25, 36}));
  for (int i = 0; i < p.n; ++i) CHECK(p.companions[i] != i);
  auto memory = population();
  memory.x = {0, 0, 0};
  f = gas2017::flows(p, s, &memory, rng);
  CHECK(f == std::vector<double>({1, 2.25, 324}));
  CHECK(gas2017::clone_probability(0, 0) == 0);
  CHECK(gas2017::clone_probability(1, 2) == 0);
  CHECK_CLOSE(gas2017::clone_probability(4, 1), .75, 1e-12);
  rng.integers = {1, 0, 0};
  gas2017::clone(p, {2, 3, 1}, rng);
  CHECK(p.x == std::vector<float>({3, 0, 3}));
  CHECK(p.objective == std::vector<double>({2, 0, 2}));
  CHECK(p.parent == std::vector<int32_t>({2, 0, 2}));
  p.objective = {1, 1, 1};
  CHECK(gas2017::normalize(p, s) == std::vector<double>({0, 0, 0}));
  p.x = {0, 0, 0};
  CHECK(gas2017::flows(p, s, &memory, rng) == std::vector<double>({0, 0, 0}));
  p.alive = {0, 1, 0};
  gas2017::clone(p, {0, 0, 0}, rng);
  CHECK(p.alive == std::vector<uint8_t>({1, 1, 1}));
}
TEST_CASE(gas_adaptive_scale_and_boundaries) {
  auto cfg = config(
      R"({"algorithm":"gas","benchmark":"constant","dimensions":1,"low":-1,"high":1})");
  Benchmark b(cfg);
  Settings s(b.config);
  auto noise = make_perturbation(b, s.json);
  OptimizationRng a(42), c(42);
  float x[] = {0}, low[1], high[1];
  PerturbationContext best{0}, worst{1};
  noise->sample_with_context(x, low, 1, a, &best);
  noise->sample_with_context(x, high, 1, c, &worst);
  CHECK_CLOSE(high[0], low[0] * 10000, 1e-7);
  bool threw = false;
  try {
    noise->sample(x, low, 1, a);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  ConstantNoise fixed;
  PerturbationTransition accepted;
  gas2017::propose(x, high, 1, b, false, fixed, 0, a, &accepted);
  CHECK(high[0] == 1);
  CHECK(fixed.calls == 2);
  CHECK(accepted.draws == 1);
  CHECK(accepted.scale == .5);
  CHECK(accepted.origin == std::vector<float>({0}));
  CHECK(accepted.displacement == std::vector<double>({1}));
  fixed.calls = 0;
  fixed.value = INFINITY;
  gas2017::propose(x, high, 1, b, false, fixed, 0, a, &accepted);
  CHECK(high[0] == x[0]);
  CHECK(fixed.calls == 64);
  CHECK(accepted.draws == 0);
  fixed.value = 3;
  gas2017::propose(x, high, 1, b, true, fixed, 0, a, &accepted);
  CHECK(b.valid(high));
  CHECK(high[0] == -1);
  CHECK(accepted.displacement[0] == 3);
  CHECK(accepted.scale == 1);
}
TEST_CASE(gas_local_search_bounds_and_cap) {
  Benchmark wide(config(R"({"benchmark":"sphere","dimensions":2})"));
  Settings wide_settings(wide.config);
  const auto wide_best = gas2017::local_search(wide, wide_settings, {{800, 500}, 890000});
  CHECK(wide_best.value < 1e-8);
  CHECK(wide.evaluations <= 200);
  auto cfg = config(
      R"({"algorithm":"gas","benchmark":"sphere","dimensions":2,"low":-1,"high":1})");
  Benchmark b(cfg);
  Settings s(b.config);
  auto best = gas2017::local_search(b, s, {{.8f, .5f}, .89});
  CHECK(best.value < 1e-8);
  CHECK(b.valid(best.x.data()));
  CHECK(b.evaluations <= 200);
  cfg.object["objective"].kind = Json::String;
  cfg.object["objective"].string = "maximize";
  Benchmark max_b(cfg);
  Settings max_s(max_b.config);
  best = gas2017::local_search(max_b, max_s, {{.8f, .5f}, .89});
  CHECK_CLOSE(best.value, 2, 1e-6);
  CHECK(max_b.valid(best.x.data()));
  s.gas_local_evaluations = 1;
  const auto before = b.evaluations;
  best = gas2017::local_search(b, s, {{.8f, .5f}, .89});
  CHECK(b.evaluations - before <= 1);
  CHECK(std::isfinite(best.value));
  auto coco_cfg = config(
      R"({"benchmark":"bbob_1","dimensions":2,"algorithm":"gas","gas_local_evaluations":6})");
  Benchmark coco(coco_cfg);
  Settings cs(coco.config);
  gas2017::local_search(coco, cs, {{0, 0}, INFINITY});
  CHECK(coco.evaluations == 5);  // One value and two central differences.
  auto lj_cfg = config(R"({"benchmark":"lennard_jones","n_atoms":2})");
  Benchmark lj(lj_cfg);
  Settings ls(lj.config);
  best = gas2017::local_search(lj, ls, {std::vector<float>(6, 0), INFINITY});
  CHECK(lj.evaluations == 1);  // Singular initial trial stops safely.
}
TEST_CASE(gas_feature_combinations_replay_and_budget) {
  for (bool tabu : {false, true})
    for (bool local : {false, true}) {
      auto cfg = config(
          R"({"algorithm":"gas","benchmark":"sphere","dimensions":3,"walkers":8,"gas_local_evaluations":20,"low":-1,"high":1,"max_evaluations":400})");
      cfg.object["gas_tabu"].kind = cfg.object["gas_local_search"].kind =
          Json::Boolean;
      cfg.object["gas_tabu"].number = tabu;
      cfg.object["gas_local_search"].number = local;
      Session a(cfg), b(cfg);
      CHECK(a.settings.perturbation == "gas_adaptive");
      CHECK(a.benchmark.evaluations <= uint64_t(local ? 28 : 8));
      for (int i = 0; i < 100; ++i) {
        const auto before = a.snapshot;
        const auto count = a.benchmark.evaluations;
        try {
          a.step();
        } catch (const std::runtime_error&) {
          CHECK(a.snapshot == before);
          CHECK(a.benchmark.evaluations == count);
          break;
        }
        b.step();
        CHECK(a.snapshot == b.snapshot);
        CHECK(a.snapshot[6] == 8);
        CHECK(a.benchmark.evaluations <= 400);
      }
      if (local) CHECK(a.best < 1e-6);
    }
  bool threw = false;
  try {
    Session bad(
        config(R"({"algorithm":"gas","walkers":8,"max_evaluations":207})"));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  Session noisy(config(
      R"({"algorithm":"gas","benchmark":"stochastic_gaussian","walkers":8,"max_evaluations":16})"));
  CHECK(!noisy.settings.gas_local_search);
  CHECK(!JsonReader(noisy.config_json).read()["gas_local_search"].flag(true));
  noisy.step();
  CHECK(noisy.benchmark.evaluations == 16);
  for (const char* algorithm :
       {"euclidean", "wave", "graph", "fmc", "wave_jump"}) {
    auto bad = config(R"({"perturbation":"gas_adaptive","walkers":4})");
    bad.object["algorithm"].kind = Json::String;
    bad.object["algorithm"].string = algorithm;
    threw = false;
    try {
      Session invalid(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
}

TEST_CASE(gas_centroid_and_memory_replacement) {
  auto p = population();
  CHECK_CLOSE(gas2017::centroid(p, {0, .5, 1})[0], 7. / 3, 1e-7);
  CHECK_CLOSE(gas2017::centroid(p, {0, 0, 0})[0], 4. / 3, 1e-7);
  // Equal positions make all flows zero, isolating the random replacement.
  p.x = {1, 1, 1};
  Draws rng;
  rng.integers = {1};
  Settings s(config("{}"));
  gas2017::insert_memory(p, s, {{1}, -10}, rng);
  CHECK(p.n == 3);
  CHECK(p.objective == std::vector<double>({0, -10, 2}));
  const auto old = p.objective;
  gas2017::insert_memory(p, s, {{0}, INFINITY}, rng);
  CHECK(p.objective == old);
}

TEST_CASE(gas_local_covariance_learns_without_extra_queries) {
  auto cfg = config(R"({"algorithm":"gas","benchmark":"rosenbrock","dimensions":2,"walkers":32,"periodic":true,"gas_local_search":false,"gas_tabu":false,"perturbation":"local_covariance","perturbation_std":0.2})");
  Session learned(cfg), repeated(cfg);
  cfg.object["covariance_learning_rate"] = number(0);
  Session frozen(cfg);
  for (int i = 0; i < 12; ++i) {
    learned.step(); repeated.step(); frozen.step();
    CHECK(learned.snapshot == repeated.snapshot);
    CHECK(learned.algorithm->population().n == 32);
    CHECK(learned.algorithm->evaluations() == uint64_t(32 * (i + 2)));
  }
  CHECK(learned.algorithm->population().x != frozen.algorithm->population().x);
}

TEST_CASE(gas_local_covariance_retry_scale_does_not_change_learned_shape) {
  auto cfg = config(R"({"algorithm":"gas","benchmark":"sphere","dimensions":2,"perturbation":"local_covariance","covariance_learning_rate":1})");
  Benchmark b(cfg);
  auto full = make_perturbation(b, cfg), retried = make_perturbation(b, cfg);
  for (int i = 0; i < 4; ++i) {
    const double scale = std::pow(.5, i);
    std::vector<double> displacement(2, 0);
    displacement[i % 2] = 1;
    full->observe({{0, 0}, displacement, 1, double(i + 1)});
    displacement[i % 2] *= scale;
    retried->observe({{0, 0}, displacement, 1, double(i + 1), scale});
  }
  full->update(); retried->update();
  OptimizationRng a(8), c(8);
  float origin[2] = {0, 0}, x[2], y[2];
  full->sample(origin, x, 2, a); retried->sample(origin, y, 2, c);
  CHECK(x[0] == y[0]); CHECK(x[1] == y[1]);
}
