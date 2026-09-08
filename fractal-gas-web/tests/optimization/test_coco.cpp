#include <fstream>
#include <sstream>

#include "optimization/coco_benchmark.hpp"
#include "optimization/engine.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
static Json json(const std::string& s) { return JsonReader(s).read(); }

TEST_CASE(coco_upstream_reference_cases) {
  std::ifstream input(COCO_FIXTURES);
  CHECK(input.good());
  std::stringstream buffer;
  buffer << input.rdbuf();
  auto fixtures = json(buffer.str());
  CHECK(fixtures["cases"].array.size() == 1296);
  int function = 0, dimension = 0, instance = 0;
  std::unique_ptr<CocoBenchmark> reference;
  std::unique_ptr<Benchmark> benchmark;
  for (const auto& item : fixtures["cases"].array) {
    auto& c = item.array;
    int f = int(c[0].num()), d = int(c[1].num()), i = int(c[2].num());
    if (f != function || d != dimension || i != instance) {
      function = f;
      dimension = d;
      instance = i;
      reference = std::make_unique<CocoBenchmark>(f, d, i);
      auto config = json("{}");
      config.object["benchmark"].kind = Json::String;
      config.object["benchmark"].string = "bbob_" + std::to_string(f);
      config.object["dimensions"] = number(d);
      config.object["coco_instance"] = number(i);
      benchmark = std::make_unique<Benchmark>(config);
      CHECK_CLOSE(benchmark->config["reference_minimum"].num(),
                  reference->minimum(), 1e-12);
    }
    std::vector<double> x;
    std::vector<float> xf;
    for (const auto& v : fixtures["points"].array[int(c[3].num())].array)
      x.push_back(v.num());
    const double expected = c[4].num(), actual = reference->evaluate(x.data());
    // Upstream test_coco uses a 4e-6 relative tolerance for these rounded
    // values.
    CHECK(std::abs(actual - expected) <=
          4e-6 * std::max(1.0, std::abs(expected)));
    for (int k = 0; k < d; ++k) {
      xf.push_back(float(x[k]));
      x[k] = xf.back();
    }
    CHECK_CLOSE(benchmark->evaluate(xf.data()), reference->evaluate(x.data()),
                1e-12);
    CHECK(benchmark->evaluations == 0);
  }
}

TEST_CASE(coco_instances_validation_and_gradients) {
  Benchmark a(
      json(R"({"benchmark":"bbob_1","dimensions":3,"coco_instance":1})"));
  Benchmark b(
      json(R"({"benchmark":"bbob_1","dimensions":3,"coco_instance":2})"));
  float x[] = {0.37f, 0.83f, -1.1f}, grad[3];
  CHECK(a.evaluate(x) != b.evaluate(x));
  a.gradient(x, grad);
  CHECK(a.evaluations == 0);
  for (int k = 0; k < 3; ++k) {
    float old = x[k];
    x[k] = old + .01f;
    double hi = a.evaluate(x);
    x[k] = old - .01f;
    double lo = a.evaluate(x);
    x[k] = old;
    CHECK_CLOSE(grad[k], (hi - lo) / .02, 2e-4);
  }
  a.gradient(x, grad, true);
  CHECK(a.evaluations == 6);
  Benchmark cusp(json(R"({"benchmark":"holder_table","dimensions":2})"));
  float origin[] = {0, 0};
  cusp.gradient(origin, grad);
  CHECK(cusp.evaluations == 0);
  cusp.gradient(origin, grad, true);
  CHECK(cusp.evaluations == 4);
  CHECK(cusp.gradient_evaluations() == 4);
  for (auto bad : {R"({"benchmark":"bbob_1","dimensions":4})",
                   R"({"benchmark":"bbob_1","coco_instance":0})",
                   R"({"benchmark":"bbob_24","coco_instance":1001})"}) {
    bool caught = false;
    try {
      Benchmark invalid(json(bad));
    } catch (const std::invalid_argument&) {
      caught = true;
    }
    CHECK(caught);
  }
}

TEST_CASE(optimization_fixed_budget_preserves_complete_steps) {
  for (auto name : {"euclidean", "wave", "graph", "fmc", "wave_jump"}) {
    auto config = json(
        R"({"benchmark":"bbob_24","dimensions":3,"walkers":12,"max_walkers":64,"horizon":2,"max_evaluations":73,"perturbation_std":0.1,"periodic":true})");
    config.object["algorithm"].kind = Json::String;
    config.object["algorithm"].string = name;
    Session limited(config);
    CHECK(!limited.settings.potential_force);
    config.object["max_evaluations"] = number(0);
    Session unlimited(config);
    bool stopped = false;
    for (int t = 0; t < 300 && !stopped; ++t) {
      CHECK(limited.snapshot == unlimited.snapshot);
      auto before = limited.snapshot;
      try {
        limited.step();
      } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("Evaluation budget reached") !=
              std::string::npos);
        CHECK(before == limited.snapshot);
        stopped = true;
      }
      if (!stopped) unlimited.step();
      CHECK(limited.benchmark.evaluations <= 73);
    }
    CHECK(stopped);
    float x[] = {1, 2, 3};
    const auto count = limited.benchmark.evaluations;
    const double best = limited.benchmark.best_observed;
    for (int i = 0; i < 100; ++i) limited.benchmark.evaluate(x);
    CHECK(limited.benchmark.evaluations == count);
    CHECK(limited.benchmark.best_observed == best);
  }
}

TEST_CASE(optimization_budget_counts_force_queries) {
  auto config = json(
      R"({"benchmark":"bbob_1","dimensions":3,"walkers":4,"max_evaluations":57,"potential_force":true,"cloning":false,"periodic":true})");
  Session s(config);
  CHECK(s.snapshot[5] == 4);
  // Two B stages, 2d queries per gradient, plus four final positions.
  s.step();
  CHECK(s.snapshot[5] == 56);
  bool stopped = false;
  try {
    s.step();
  } catch (const std::runtime_error&) {
    stopped = true;
  }
  CHECK(stopped);
  CHECK(s.snapshot[5] == 56);
  CHECK(s.snapshot[9] == s.benchmark.best_observed);
  config.object["max_evaluations"] = number(3);
  stopped = false;
  try {
    Session invalid(config);
  } catch (const std::invalid_argument&) {
    stopped = true;
  }
  CHECK(stopped);
}
