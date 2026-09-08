#include "optimization/perturbation.hpp"

#include <map>

namespace fg::optimization {
namespace {
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
};
std::map<std::string, Entry>& entries() {
  static const Json parameters =
      JsonReader(
          std::string(
              R"([{"id":"perturbation_std","label":"Standard deviation","type":"number","default":1,"min":0,"max":1000000}])"))
          .read();
  static std::map<std::string, Entry> list{
      {"gaussian",
       {"Gaussian (mean 0)",
        [](const Benchmark&, const Json& c) {
          return std::make_unique<IndependentNoise>(c, true);
        },
        parameters}},
      {"uniform",
       {"Uniform (mean 0)",
        [](const Benchmark&, const Json& c) {
          return std::make_unique<IndependentNoise>(c, false);
        },
        parameters}}};
  return list;
}
}  // namespace
void register_perturbation(const std::string& id, const std::string& name,
                           PerturbationFactory factory,
                           const Json& parameters) {
  if (id.empty() || name.empty() || !factory || entries().count(id) ||
      parameters.kind != Json::Array)
    throw std::invalid_argument("Duplicate or invalid perturbation strategy");
  entries().emplace(id, Entry{name, std::move(factory), parameters});
}
std::unique_ptr<Perturbation> make_perturbation(const Benchmark& b,
                                                const Json& config) {
  auto entry = entries().find(config["perturbation"].str("gaussian"));
  if (entry == entries().end())
    throw std::invalid_argument("Unknown perturbation strategy");
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
    result.array.push_back(std::move(item));
  }
  return result;
}
}  // namespace fg::optimization
