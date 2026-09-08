#include "optimization/perturbation.hpp"

#include <map>

namespace fg::optimization {
namespace {
class GasAdaptive final : public Perturbation {
  double width;

 public:
  explicit GasAdaptive(const Benchmark& b) : width(b.high - b.low) {}
  void sample(const float*, float*, int, Rng&) const override {
    throw std::invalid_argument(
        "GAS adaptive perturbation requires objective context");
  }
  void sample_with_context(const float*, float* delta, int d, Rng& rng,
                           const PerturbationContext* context) const override {
    if (!context || !std::isfinite(context->normalized_objective) ||
        context->normalized_objective < 0 || context->normalized_objective > 1)
      throw std::invalid_argument(
          "GAS adaptive perturbation requires normalized objective in [0,1]");
    const double sigma =
        width * std::pow(10., -5 + 4 * context->normalized_objective);
    for (int k = 0; k < d; ++k) delta[k] = float(sigma * normal(rng));
  }
};
class IndependentNoise final : public Perturbation {
  double stddev;
  bool gaussian;

 public:
  IndependentNoise(const Json& config, bool normal)
      : stddev(bounded(config["perturbation_std"], 1, 0, 1e6,
                       "perturbation standard deviation")),
        gaussian(normal) {}
  void sample(const float*, float* delta, int d, Rng& rng) const override {
    for (int k = 0; k < d; ++k)
      delta[k] = float(stddev *
                       (gaussian ? normal(rng)
                                 : std::sqrt(3.0) * (2 * rng.uniform01() - 1)));
  }
};
struct Entry {
  std::string name;
  PerturbationFactory factory;
  Json parameters;
  Json algorithms;
};
std::map<std::string, Entry>& entries() {
  static const Json parameters =
      JsonReader(
          std::string(
              R"([{"id":"perturbation_std","label":"Standard deviation","type":"number","default":1,"min":0,"max":1000000}])"))
          .read();
  static std::map<std::string, Entry> list{
      {"gas_adaptive",
       {"GAS adaptive Gaussian",
        [](const Benchmark& b, const Json&) {
          return std::make_unique<GasAdaptive>(b);
        },
        JsonReader(std::string("[]")).read(),
        JsonReader(std::string("[\"gas\"]")).read()}},
      {"gaussian",
       {"Gaussian (mean 0)",
        [](const Benchmark&, const Json& c) {
          return std::make_unique<IndependentNoise>(c, true);
        },
        parameters, Json{}}},
      {"uniform",
       {"Uniform (mean 0)",
        [](const Benchmark&, const Json& c) {
          return std::make_unique<IndependentNoise>(c, false);
        },
        parameters, Json{}}}};
  return list;
}
}  // namespace
void register_perturbation(const std::string& id, const std::string& name,
                           PerturbationFactory factory, const Json& parameters,
                           const Json& algorithms) {
  if (id.empty() || name.empty() || !factory || entries().count(id) ||
      parameters.kind != Json::Array)
    throw std::invalid_argument("Duplicate or invalid perturbation strategy");
  if (algorithms.kind != Json::Null && algorithms.kind != Json::Array)
    throw std::invalid_argument("Perturbation algorithms must be an array");
  entries().emplace(id,
                    Entry{name, std::move(factory), parameters, algorithms});
}
std::unique_ptr<Perturbation> make_perturbation(const Benchmark& b,
                                                const Json& config) {
  auto entry = entries().find(config["perturbation"].str("gaussian"));
  if (entry == entries().end())
    throw std::invalid_argument("Unknown perturbation strategy");
  if (entry->second.algorithms.kind == Json::Array) {
    bool supported = false;
    for (const auto& algorithm : entry->second.algorithms.array)
      supported |= algorithm.str() == config["algorithm"].str("euclidean");
    if (!supported)
      throw std::invalid_argument(
          "Perturbation is not supported by this algorithm");
  }
  return entry->second.factory(b, config);
}
Json perturbation_catalog() {
  Json result;
  result.kind = Json::Array;
  for (const auto& [id, entry] : entries()) {
    Json item;
    item.kind = Json::Object;
    item.object["id"].kind = item.object["name"].kind = Json::String;
    item.object["id"].string = id;
    item.object["name"].string = entry.name;
    item.object["parameters"] = entry.parameters;
    if (entry.algorithms.kind == Json::Array)
      item.object["algorithms"] = entry.algorithms;
    result.array.push_back(std::move(item));
  }
  return result;
}
}  // namespace fg::optimization
