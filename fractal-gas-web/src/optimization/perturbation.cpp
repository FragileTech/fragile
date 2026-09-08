#include "optimization/perturbation.hpp"

#include <map>
#include <deque>
#include <numeric>
#include <Eigen/Dense>

namespace fg::optimization {
namespace {
// Local proposal geometry, learned only at explicit population/cycle boundaries.
class LocalCovariance final : public Perturbation {
  using Matrix = Eigen::MatrixXd;
  struct Anchor { std::vector<float> x; Matrix covariance, factor; };
  int dimensions;
  double sigma, rate, width;
  bool periodic;
  bool dirty = false;
  std::deque<PerturbationTransition> archive;
  std::vector<Anchor> anchors;
  double distance(const std::vector<float>& a, const std::vector<float>& b) const {
    double sum = 0;
    for (int j = 0; j < dimensions; ++j) {
      double delta = double(a[j]) - b[j];
      if (periodic) delta = std::remainder(delta, width);
      sum += delta * delta;
    }
    return sum;
  }
  const Anchor* nearest(const std::vector<float>& x) const {
    const Anchor* best = nullptr;
    double value = INFINITY;
    for (const auto& a : anchors) {
      double v = distance(x, a.x);
      if (v < value) { value = v; best = &a; }
    }
    return best;
  }
 public:
  LocalCovariance(const Benchmark& b, const Json& config)
      : dimensions(b.d),
        sigma(bounded(config["perturbation_std"], 1, 0, 1e6, "perturbation standard deviation")),
        rate(bounded(config["covariance_learning_rate"], .1, 0, 1, "covariance learning rate")),
        width(b.high - b.low), periodic(config["periodic"].flag(false)) {}
  void sample(const float* position, float* delta, int d, Rng& rng) const override {
    if (sigma == 0) { std::fill(delta, delta + d, 0); return; }
    const auto* a = nearest(std::vector<float>(position, position + d));
    Eigen::VectorXd z(d);
    for (int j = 0; j < d; ++j) z[j] = normal(rng);
    if (a) z = (a->factor * z).eval();
    for (int j = 0; j < d; ++j) delta[j] = float(sigma * z[j]);
  }
  void sample_action(const float* origin, const float*, float* delta, int d, Rng& rng) const override {
    sample(origin, delta, d, rng);
  }
  void observe(const PerturbationTransition& t) override {
    if (sigma == 0 || t.draws <= 0 || !std::isfinite(t.improvement) ||
        !(t.scale > 0) || !std::isfinite(t.scale) ||
        int(t.origin.size()) != dimensions || int(t.displacement.size()) != dimensions)
      return;
    for (int j = 0; j < dimensions; ++j)
      if (!std::isfinite(t.origin[j]) || !std::isfinite(t.displacement[j])) return;
    archive.push_back(t);
    if (archive.size() > 512) archive.pop_front();
    dirty = true;
  }
  void reset() override { archive.clear(); anchors.clear(); dirty = false; }
  void update() override {
    if (!dirty || archive.empty()) return;
    dirty = false;
    std::vector<size_t> selected{archive.size() - 1};
    std::vector<double> distances(archive.size(), INFINITY);
    while (selected.size() < 16) {
      for (size_t i = 0; i < archive.size(); ++i)
        distances[i] = std::min(distances[i], distance(archive[i].origin, archive[selected.back()].origin));
      size_t next = size_t(std::max_element(distances.begin(), distances.end()) - distances.begin());
      if (distances[next] <= 0) break;
      selected.push_back(next);
    }
    std::vector<Anchor> updated;
    const Matrix identity = Matrix::Identity(dimensions, dimensions);
    for (size_t index : selected) {
      Anchor a;
      a.x = archive[index].origin;
      const Anchor* old = nearest(a.x);
      a.covariance = old ? old->covariance : identity;
      std::vector<size_t> neighbors(archive.size());
      std::iota(neighbors.begin(), neighbors.end(), 0);
      std::stable_sort(neighbors.begin(), neighbors.end(), [&](size_t i, size_t j) {
        return distance(a.x, archive[i].origin) < distance(a.x, archive[j].origin);
      });
      if (neighbors.size() > 32) neighbors.resize(32);
      neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(), [&](size_t i) {
        return archive[i].improvement <= 0;
      }), neighbors.end());
      std::stable_sort(neighbors.begin(), neighbors.end(), [&](size_t i, size_t j) {
        return archive[i].improvement / archive[i].draws > archive[j].improvement / archive[j].draws;
      });
      if (neighbors.size() >= 4) {
        Matrix moment = Matrix::Zero(dimensions, dimensions);
        for (size_t k = 0; k < neighbors.size(); ++k) {
          const auto& t = archive[neighbors[k]];
          Eigen::VectorXd y(dimensions);
          for (int j = 0; j < dimensions; ++j)
            y[j] = t.displacement[j] / (sigma * t.scale * std::sqrt(double(t.draws)));
          const double weight = std::log(neighbors.size() + .5) - std::log(k + 1.);
          moment.noalias() += weight * y * y.transpose();
        }
        // Normalizing trace also cancels the common rank-weight normalizer.
        const double trace = moment.trace();
        if (moment.allFinite() && trace > 0 && std::isfinite(trace))
          a.covariance = (1 - rate) * a.covariance + rate * (dimensions / trace) * moment;
      }
      a.covariance = (.5 * (a.covariance + a.covariance.transpose())).eval();
      a.covariance *= dimensions / a.covariance.trace();
      Eigen::LLT<Matrix> chol(.95 * a.covariance + .05 * identity);
      if (chol.info() != Eigen::Success) {
        a.covariance = identity;
        a.factor = identity;
      } else a.factor = chol.matrixL();
      updated.push_back(std::move(a));
    }
    anchors = std::move(updated);
  }
};
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
      {"local_covariance",
       {"Adaptive local Gaussian",
        [](const Benchmark& b, const Json& c) { return std::make_unique<LocalCovariance>(b, c); },
        JsonReader(std::string(R"([{"id":"perturbation_std","label":"Standard deviation","type":"number","default":1,"min":0,"max":1000000},{"id":"covariance_learning_rate","label":"Covariance learning rate","type":"number","default":0.1,"min":0,"max":1}])")).read(),
        JsonReader(std::string(R"(["wave","fmc","wave_jump","gas"])")).read()}},
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
