#include "optimization/perturbation.hpp"
#include "optimization/adaptive.hpp"
#include "optimization/cloning_guided.hpp"

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
  void configure(const Json& config) {
    sigma = config["perturbation_std"].num(1);
    rate = config["covariance_learning_rate"].num(.1);
    const bool wrapping = config["periodic"].flag(false);
    if (wrapping != periodic) reset();
    periodic = wrapping;
  }
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
  Json visual_geometry() const {
    Json list;list.kind=Json::Array;
    const size_t per_model=size_t(dimensions)*dimensions+5*size_t(dimensions);
    const size_t limit=std::max<size_t>(262144,per_model)/per_model;
    for(const auto& a:anchors) {
      if(list.array.size()>=limit) break;
      Json j;j.kind=Json::Object;
      j.object["available_models"]=number(anchors.size());
      j.object["anchor"]=array(std::vector<double>(a.x.begin(),a.x.end()));
      j.object["scale"]=number(sigma);j.object["columns"]=number(dimensions);
      j.object["representation"].kind=Json::String;j.object["representation"].string="dense";
      std::vector<double> values;
      Matrix actual=.95*a.covariance+.05*Matrix::Identity(dimensions,dimensions);
      for(int i=0;i<dimensions;++i) for(int k=0;k<dimensions;++k) values.push_back(actual(i,k));
      j.object["shape"]=array(values);list.array.push_back(std::move(j));
    }
    return list;
  }
  void reset() override { archive.clear(); anchors.clear(); dirty = false; }
  void update() override {
    if (!dirty || archive.empty() || sigma == 0) return;
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
  GasAdaptive(const Benchmark& b,const Json& c) : width((b.high-b.low)*bounded(c["gas_scale_multiplier"],1,0,1e6,"GAS movement multiplier")) {}
  void sample(const float*, float*, int, Rng&) const override {
    throw std::invalid_argument(
        "GAS adaptive perturbation requires objective context");
  }
  void sample_with_context(const float* position, float* delta, int d, Rng& rng,
                           const PerturbationContext* context) const override {
    if (!context || !std::isfinite(context->normalized_objective) ||
        context->normalized_objective < 0 || context->normalized_objective > 1)
      throw std::invalid_argument(
          "GAS adaptive perturbation requires normalized objective in [0,1]");
    const double sigma =
        width * std::pow(10., -5 + 4 * context->normalized_objective);
    if(observer) observer->reference(position,d,sigma);
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
      {"cloning_guided",
       {"Clone-guided adaptive covariance (experimental)",
        [](const Benchmark& b,const Json& c) {return std::make_unique<CloningGuided>(b,c);},
        JsonReader(std::string(R"cg([{"id":"adaptive_min_scale","label":"Minimum movement scale","type":"number","default":0.0001,"min":0,"max":1000000},
          {"id":"adaptive_max_scale","label":"Maximum movement scale","type":"number","default":1,"min":0,"max":1000000},
          {"id":"cloning_geometry","label":"Clone-score covariance","type":"boolean","default":true},
          {"id":"cloning_drift","label":"Clone-score drift","type":"boolean","default":true},
          {"id":"cloning_drift_strength","label":"Maximum drift / noise scale","type":"number","default":0.25,"min":0,"max":1},
          {"id":"adaptive_euclidean_mode","label":"Euclidean movement","type":"enum","default":"velocity","options":[["velocity","Cloning-guided velocity kicks"],["position","Direct position proposals"]]}])cg")).read(),
        JsonReader(std::string(R"(["wave","graph","fmc","wave_jump","euclidean","gas"])")).read()}},
      {"adaptive_fractal",
       {"Adaptive fractal exploration (experimental)",
        [](const Benchmark& b, const Json& c) { return std::make_unique<AdaptiveExploration>(b,c); },
        JsonReader(std::string(R"adaptive([{"id":"adaptive_min_scale","label":"Minimum movement scale","type":"number","default":0.0001,"min":0,"max":1000000},
        {"id":"adaptive_max_scale","label":"Maximum movement scale","type":"number","default":1,"min":0,"max":1000000},
        {"id":"adaptive_active","label":"Active negative updates","type":"boolean","default":true},
        {"id":"adaptive_paths","label":"Evolution paths (shape only)","type":"boolean","default":true},
        {"id":"adaptive_difference","label":"Walker-difference proposals","type":"boolean","default":true},
        {"id":"adaptive_pairs","label":"Paired trials (10%)","type":"boolean","default":true},
        {"id":"adaptive_mixture","label":"Adapt proposal mixture","type":"boolean","default":true},
        {"id":"adaptive_euclidean_mode","label":"Euclidean movement","type":"enum","default":"velocity","options":[["velocity","Adaptive velocity kicks"],["position","Direct position proposals"]]}])adaptive")).read(),
        JsonReader(std::string(R"(["wave","graph","fmc","wave_jump","euclidean","gas"])")).read()}},
      {"local_covariance",
       {"Adaptive local Gaussian",
        [](const Benchmark& b, const Json& c) { return std::make_unique<LocalCovariance>(b, c); },
        JsonReader(std::string(R"([{"id":"perturbation_std","label":"Standard deviation","type":"number","default":1,"min":0,"max":1000000},{"id":"covariance_learning_rate","label":"Covariance learning rate","type":"number","default":0.1,"min":0,"max":1}])")).read(),
        JsonReader(std::string(R"(["wave","fmc","wave_jump","gas"])")).read()}},
      {"gas_adaptive",
       {"GAS adaptive Gaussian",
        [](const Benchmark& b, const Json& c) {
          return std::make_unique<GasAdaptive>(b,c);
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
  auto result=entry->second.factory(b, config);
  if(config["geometry_diagnostics"].flag(false)) enable_geometry(*result,b,config,true);
  return result;
}
std::unique_ptr<Perturbation> retune_perturbation(const Perturbation& old, const Benchmark& b,
                                               const Json& previous, const Json& next) {
  auto replacement = make_perturbation(b, next);
  if(old.observer) enable_geometry(*replacement,b,next,true);  // Validate before copying active learning.
  if (previous["boundary"].str() != next["boundary"].str()) return replacement;
  if (previous["perturbation"].str() == next["perturbation"].str()) {
    if(const auto* guided=dynamic_cast<const CloningGuided*>(&old)) {
      auto copy=std::make_unique<CloningGuided>(*guided);copy->configure(next);inherit_geometry(old,*copy,b,next);return copy;
    }
    if (const auto* adaptive = dynamic_cast<const AdaptiveExploration*>(&old)) {
      auto copy=std::make_unique<AdaptiveExploration>(*adaptive);
      if(previous["adaptive_euclidean_mode"].str("velocity")!=next["adaptive_euclidean_mode"].str("velocity")) copy->reset();
      copy->configure(next);
      inherit_geometry(old,*copy,b,next);
      return copy;
    }
    if (const auto* local = dynamic_cast<const LocalCovariance*>(&old)) {
      auto copy = std::make_unique<LocalCovariance>(*local);
      copy->configure(next);
      inherit_geometry(old,*copy,b,next);
      return copy;
    }
    inherit_geometry(old,*replacement,b,next);
  }
  return replacement;
}
Json proposal_visual_geometry(const Perturbation& proposal) {
  if(const auto* p=dynamic_cast<const CloningGuided*>(&proposal)) return p->visual_geometry();
  if(const auto* p=dynamic_cast<const AdaptiveExploration*>(&proposal)) return p->visual_geometry();
  if(const auto* p=dynamic_cast<const LocalCovariance*>(&proposal)) return p->visual_geometry();
  return Json{};
}
Json perturbation_geometry(const Perturbation& proposal) {
  if(const auto* guided=dynamic_cast<const CloningGuided*>(&proposal)) return guided->geometry();
  if(const auto* adaptive=dynamic_cast<const AdaptiveExploration*>(&proposal)) return adaptive->geometry();
  return Json{};
}
void restore_perturbation_geometry(Perturbation& proposal,const Json& geometry) {
  if(auto* guided=dynamic_cast<CloningGuided*>(&proposal)) guided->restore_geometry(geometry);
  if(auto* adaptive=dynamic_cast<AdaptiveExploration*>(&proposal)) adaptive->restore_geometry(geometry);
}
Json perturbation_diagnostics(const Perturbation& proposal) {
  if(const auto* guided=dynamic_cast<const CloningGuided*>(&proposal)) return guided->diagnostics();
  if(const auto* adaptive=dynamic_cast<const AdaptiveExploration*>(&proposal)) return adaptive->diagnostics();
  return Json{};
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
