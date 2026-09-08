#include <algorithm>

#include "optimization/engine.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
static Json config(const std::string& text) { return JsonReader(text).read(); }
TEST_CASE(optimization_benchmark_values) {
  Benchmark b(config(R"({"benchmark":"rosenbrock","dimensions":2})"));
  float x[] = {0, 0};
  CHECK_CLOSE(b.evaluate(x), 1, 1e-7);
  x[0] = 1.2f;
  x[1] = 1.44f;
  CHECK_CLOSE(b.evaluate(x), .04, 1e-6);
  Benchmark lj(config(R"({"benchmark":"lennard_jones","n_atoms":2})"));
  float atoms[] = {0, 0, 0, float(std::pow(2, 1. / 6)), 0, 0};
  CHECK_CLOSE(lj.evaluate(atoms), -1, 1e-6);
  atoms[3] = 0;
  CHECK(!std::isfinite(lj.evaluate(atoms)));
  CHECK(JsonReader(catalog_json()).read()["benchmarks"].array.size() == 13);
}
TEST_CASE(optimization_gradients) {
  auto catalog = JsonReader(catalog_json()).read();
  for (auto& entry : catalog["benchmarks"].array) {
    auto c = config(R"({"dimensions":2,"n_atoms":2})");
    c.object["benchmark"] = entry["id"];
    Benchmark b(c);
    std::vector<float> x(b.d), g(b.d);
    for (int k = 0; k < b.d; ++k) x[k] = float(.37 + k * .81);
    b.gradient(x.data(), g.data());
    for (int k = 0; k < b.d; ++k) {
      float old = x[k], h = 1e-3f;
      x[k] = old + h;
      double hi = b.evaluate(x.data());
      x[k] = old - h;
      double lo = b.evaluate(x.data());
      x[k] = old;
      CHECK_CLOSE(g[k], (hi - lo) / (2 * h), .003);
    }
  }
}
TEST_CASE(optimization_deterministic_sessions) {
  for (auto name : {"wave", "graph", "euclidean"}) {
    auto c = config(
        R"({"benchmark":"quadratic","walkers":16,"max_walkers":64,"periodic":true})");
    c.object["algorithm"].kind = Json::String;
    c.object["algorithm"].string = name;
    Session a(c), b(c);
    for (int i = 0; i < 20; ++i) {
      CHECK(a.snapshot == b.snapshot);
      a.step();
      b.step();
    }
    CHECK(a.snapshot == b.snapshot);
    CHECK(a.best <= a.snapshot[8]);
    CHECK(a.snapshot.size() ==
          12 + size_t(a.snapshot[1]) * (2 * size_t(a.snapshot[2]) + 8));
  }
}
TEST_CASE(optimization_noise_sampling_is_observational) {
  Session a(config(R"({"benchmark":"stochastic_gaussian","walkers":8})")),
      b(config(R"({"benchmark":"stochastic_gaussian","walkers":8})"));
  float x[] = {1, 2, 3};
  for (int i = 0; i < 100; ++i) CHECK(a.benchmark.evaluate(x) == 0);
  a.step();
  b.step();
  CHECK(a.snapshot == b.snapshot);
  CHECK(!a.settings.potential_force);
}
TEST_CASE(optimization_periodic_and_cloning) {
  Benchmark b(
      config(R"({"benchmark":"constant","dimensions":2,"low":-1,"high":1})"));
  float x[] = {5.5f, -4.5f};
  b.wrap(x);
  CHECK_CLOSE(x[0], -.5, 1e-6);
  CHECK_CLOSE(x[1], -.5, 1e-6);
  Population p;
  p.resize(3, 2);
  p.x = {0, 0, 1, 1, 2, 2};
  p.v = {1, 0, 2, 0, 3, 0};
  Settings s(config(R"({"sigma_x":0,"restitution":0})"));
  fg::Mt19937Rng rng(7);
  clone_population(p, s, {1, 1, 1}, {1, 0, 1}, rng);
  CHECK(p.x == std::vector<float>({1, 1, 1, 1, 1, 1}));
  CHECK(p.v == std::vector<float>({2, 0, 2, 0, 2, 0}));
}
TEST_CASE(optimization_all_dead_and_validation) {
  bool caught = false;
  try {
    Session bad(config(R"({"benchmark":"easom","dimensions":3})"));
  } catch (const std::exception&) {
    caught = true;
  }
  CHECK(caught);
  Population p;
  p.resize(2, 2);
  Benchmark b(config(R"({"dimensions":2})"));
  Settings s(config("{}"));
  fg::Mt19937Rng rng(1);
  caught = false;
  try {
    select_companions(p, s, b, "uniform", 1, rng);
  } catch (const std::exception&) {
    caught = true;
  }
  CHECK(caught);
  auto tiny = config(
      R"({"benchmark":"constant","dimensions":2,"walkers":8,"low":-0.000001,"high":0.000001,"cloning":false,"delta_t":1})");
  Session killed(tiny);
  killed.step();
  CHECK(killed.snapshot[6] == 0);
  auto final_frame = killed.snapshot;
  caught = false;
  try {
    killed.step();
  } catch (const std::exception&) {
    caught = true;
  }
  CHECK(caught);
  CHECK(killed.snapshot == final_frame);
  Session reset(tiny);
  CHECK(reset.snapshot[6] == 8);
  tiny.object["periodic"].kind = Json::Boolean;
  tiny.object["periodic"].number = 1;
  Session wrapped(tiny);
  wrapped.step();
  CHECK(wrapped.snapshot[6] == 8);
}

#include "optimization/fixtures.hpp"
class FixtureRng final : public fg::Rng {
 public:
  float uniform01() override { return .37f; }
  int64_t randint(int64_t lo, int64_t) override { return lo; }
};
static Population fixture_population() {
  namespace f = optimization_fixture;
  Population p;
  p.resize(4, 2);
  p.has_velocity = true;
  p.x.assign(f::positions.begin(), f::positions.end());
  p.v.assign(f::velocities.begin(), f::velocities.end());
  p.objective = f::objectives;
  p.alive.assign(4, 1);
  return p;
}
template <typename T>
static void fixture_equal(const std::vector<T>& a,
                          const std::vector<double>& expected,
                          double tol = 2e-6) {
  CHECK(a.size() == expected.size());
  for (size_t i = 0; i < std::min(a.size(), expected.size()); ++i)
    CHECK_CLOSE(a[i], expected[i], tol);
}
TEST_CASE(optimization_python_operator_fixtures) {
  namespace f = optimization_fixture;
  auto cfg = config(
      R"({"benchmark":"quadratic","dimensions":2,"walkers":4,"lambda_alg":0.2,"gamma":0.8,"beta":1.2,"delta_t":0.01,"sigma_x":0.03,"restitution":0.4})");
  Benchmark b(cfg);
  Settings s(cfg);
  auto p = fixture_population();
  FixtureRng rng;
  std::vector<int32_t> companions{1, 0, 3, 2};
  fixture_equal(fitness(p, s, b, companions), f::fitness_global);
  Benchmark periodic(config(R"({"dimensions":2,"low":-1,"high":1})"));
  s.periodic = true;
  fixture_equal(fitness(p, s, periodic, companions), f::fitness_periodic);
  auto wrapped = p.x;
  for (int i = 0; i < p.n; ++i) periodic.wrap(wrapped.data() + i * p.d);
  fixture_equal(wrapped, f::wrapped_positions);
  s.periodic = false;
  s.rho = 1.5;
  fixture_equal(fitness(p, s, b, companions), f::fitness_local);
  s.rho = 0;
  for (auto method :
       {"uniform", "softmax", "cloning", "random_pairing", "greedy_pairing"}) {
    auto actual = select_companions(p, s, b, method, 2, rng);
    const auto& expected =
        std::string(method) == "uniform"   ? f::companions_uniform
        : std::string(method) == "softmax" ? f::companions_softmax
        : std::string(method) == "cloning" ? f::companions_cloning
        : std::string(method) == "random_pairing"
            ? f::companions_random_pairing
            : f::companions_greedy_pairing;
    fixture_equal(actual, expected, 0);
    const auto& periodic_expected =
        std::string(method) == "uniform"   ? f::periodic_companions_uniform
        : std::string(method) == "softmax" ? f::periodic_companions_softmax
        : std::string(method) == "cloning" ? f::periodic_companions_cloning
        : std::string(method) == "random_pairing"
            ? f::periodic_companions_random_pairing
            : f::periodic_companions_greedy_pairing;
    s.periodic = true;
    fixture_equal(select_companions(p, s, periodic, method, 2, rng),
                  periodic_expected, 0);
    s.periodic = false;
  }
  std::vector<uint8_t> mask(f::clone_mask.begin(), f::clone_mask.end());
  clone_population(p, s, companions, mask, rng);
  fixture_equal(p.x, f::cloned_x);
  fixture_equal(p.v, f::cloned_v);
  p = fixture_population();
  baoab(p, s, b, rng);
  fixture_equal(p.x, f::baoab_x);
  fixture_equal(p.v, f::baoab_v);
}

TEST_CASE(optimization_cumulative_scores_follow_objectives) {
  for (auto algorithm : {"wave", "graph"})
    for (auto benchmark : {"quadratic", "stochastic_gaussian"}) {
      auto c = config(
          R"({"dimensions":3,"walkers":24,"max_walkers":200,"dt_min":1,"dt_max":4,"periodic":true,"elites":2})");
      c.object["algorithm"].kind = Json::String;
      c.object["algorithm"].string = algorithm;
      c.object["benchmark"].kind = Json::String;
      c.object["benchmark"].string = benchmark;
      Session session(c);
      for (int t = 0; t < 40; ++t) {
        auto& p = session.algorithm->population();
        for (int i = 0; i < p.n; ++i)
          if (p.alive[i])
            CHECK_CLOSE(session.algorithm->objective_score(i), -p.objective[i],
                        2e-5);
        session.step();
      }
    }
}
TEST_CASE(optimization_registry_exposes_extensions) {
  register_algorithm(
      "extra", "Extra optimizer", false,
      [](Benchmark&, const Settings&) -> std::unique_ptr<Algorithm> {
        throw std::runtime_error("fixture only");
      },
      config(
          R"([{"id":"learning_rate","label":"Learning rate","type":"number","default":0.1,"min":0,"max":1}])"));
  auto c = JsonReader(discovery_json()).read();
  bool found = false;
  for (auto& a : c["algorithms"].array)
    if (a["id"].str() == "extra") {
      found = true;
      CHECK(a["parameters"].array.size() == 1);
      CHECK(a["parameters"].array[0]["id"].str() == "learning_rate");
    }
  CHECK(found);
}
