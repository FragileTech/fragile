#include "optimization/coco_benchmark.hpp"

#include <algorithm>

#include "coco.h"

namespace fg::optimization {
namespace {
struct Problem {
  std::unique_ptr<coco_suite_t, decltype(&coco_suite_free)> suite{
      nullptr, coco_suite_free};
  coco_problem_t* problem = nullptr;
  Problem(int function, int dimension, int instance) {
    // Library clients own progress reporting; keep stdout suitable for C API
    // tools.
    coco_set_log_level("warning");
    const auto instances = "instances: " + std::to_string(instance);
    const auto options = "function_indices: " + std::to_string(function) +
                         " dimensions: " + std::to_string(dimension);
    suite.reset(coco_suite("bbob", instances.c_str(), options.c_str()));
    if (!suite) throw std::runtime_error("Unable to create COCO suite");
    problem = coco_suite_get_next_problem(suite.get(), nullptr);
    if (!problem) throw std::runtime_error("Unable to create COCO problem");
  }
  double evaluate(const double* x) const {
    double y;
    coco_evaluate_function(problem, x, &y);
    return y;
  }
};
}  // namespace
struct CocoBenchmark::Impl {
  Problem simulation, display;
  double minimum;
  std::string id;
  Impl(int f, int d, int i) : simulation(f, d, i), display(f, d, i) {
    // Optimum access taints only the display problem, never the optimizer's.
    std::vector<double> x(d);
    if (!coco_problem_get_best_parameter(display.problem, x.data()))
      throw std::runtime_error("COCO reference optimum unavailable");
    minimum = display.evaluate(x.data());
    id = coco_problem_get_id(simulation.problem);
  }
};
CocoBenchmark::CocoBenchmark(int f, int d, int i) {
  const std::vector<int> dimensions{2, 3, 5, 10, 20, 40};
  if (f < 1 || f > 24 || i < 1 || i > 1000 ||
      std::find(dimensions.begin(), dimensions.end(), d) == dimensions.end())
    throw std::invalid_argument(
        "COCO BBOB requires dimensions 2, 3, 5, 10, 20, or 40 and instance "
        "1–1000");
  impl = std::make_unique<Impl>(f, d, i);
}
CocoBenchmark::~CocoBenchmark() = default;
double CocoBenchmark::evaluate(const double* x, bool simulation) const {
  return (simulation ? impl->simulation : impl->display).evaluate(x);
}
double CocoBenchmark::minimum() const { return impl->minimum; }
const std::string& CocoBenchmark::problem_id() const { return impl->id; }
void append_coco_catalog(Json& benchmarks) {
  static const char* names[] = {"Sphere",
                                "Ellipsoid separable",
                                "Rastrigin separable",
                                "Bueche–Rastrigin",
                                "Linear slope",
                                "Attractive sector",
                                "Step ellipsoid",
                                "Rosenbrock original",
                                "Rosenbrock rotated",
                                "Ellipsoid rotated",
                                "Discus",
                                "Bent cigar",
                                "Sharp ridge",
                                "Different powers",
                                "Rastrigin rotated",
                                "Weierstrass",
                                "Schaffer F7 (condition 10)",
                                "Schaffer F7 (condition 1000)",
                                "Griewank–Rosenbrock",
                                "Schwefel",
                                "Gallagher 101 peaks",
                                "Gallagher 21 peaks",
                                "Katsuura",
                                "Lunacek bi-Rastrigin"};
  for (int f = 1; f <= 24; ++f) {
    Json entry =
        JsonReader(
            std::string(
                R"json({"suite":"bbob","bounds":[-5,5],"minDimension":2,"maxDimension":40,"dimensions":[2,3,5,10,20,40],"parameters":{"coco_instance":1},"gradient":"central difference (2d objective evaluations)","source":"https://coco-platform.org/testsuites/bbob/overview.html","provider":"COCO 2.8.2","reference":"Shifted/rotated BBOB instance; reference minimum resolved on reset. Dimensions: 2, 3, 5, 10, 20, 40."})json"))
            .read();
    auto string = [&](const char* key, const std::string& v) {
      entry.object[key].kind = Json::String;
      entry.object[key].string = v;
    };
    string("id", "bbob_" + std::to_string(f));
    string("name", "BBOB f" + std::to_string(f) + " · " + names[f - 1]);
    string("group", f <= 5    ? "Separable"
                    : f <= 9  ? "Moderate conditioning"
                    : f <= 14 ? "Ill-conditioned"
                    : f <= 19 ? "Multimodal, global structure"
                              : "Multimodal, weak structure");
    entry.object["function"] = number(f);
    benchmarks.array.push_back(std::move(entry));
  }
}
}  // namespace fg::optimization
