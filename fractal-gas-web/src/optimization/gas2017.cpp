#include "optimization/gas2017.hpp"
#include "optimization/adaptive.hpp"

#include <LBFGSB.h>

#include <algorithm>
#include <numeric>

namespace fg::optimization::gas2017 {
std::vector<double> normalize(const Population& p, const Settings& s) {
  double low = INFINITY, high = -INFINITY;
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i]) {
      const double cost = -s.score(p.objective[i]);
      low = std::min(low, cost);
      high = std::max(high, cost);
    }
  std::vector<double> phi(p.n, 1);
  // Scale first to avoid overflowing high-low for large finite objectives.
  const double scale = std::max({1., std::abs(low), std::abs(high)});
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i])
      phi[i] =
          high == low
              ? 0
              : std::clamp((-s.score(p.objective[i]) / scale - low / scale) /
                               (high / scale - low / scale),
                           0., 1.);
  return phi;
}
double clone_score(double flow,double donor_flow) {
  return flow>0 ? 1-donor_flow/flow : 0;
}
double clone_probability(double flow, double donor_flow) {
  return flow > 0 && donor_flow < flow
             ? std::clamp(clone_score(flow,donor_flow), 0., 1.)
             : 0;
}
double distance2(const float* a, const float* b, int d) {
  double result = 0;
  for (int k = 0; k < d; ++k) result += std::pow(double(a[k]) - b[k], 2);
  return result;
}
static std::vector<int32_t> live_rows(const Population& p) {
  std::vector<int32_t> live;
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i]) live.push_back(i);
  if (live.empty())
    throw std::runtime_error(
        "GAS has no finite walkers; reset or change the benchmark");
  return live;
}
static int companion(int i, const std::vector<int32_t>& live,
                     const std::vector<int>& rank, Rng& rng) {
  if (rank[i] < 0 || live.size() == 1) return live[rng.randint(0, live.size())];
  int draw = int(rng.randint(0, live.size() - 1));
  if (draw >= rank[i]) ++draw;
  return live[draw];
}
static std::vector<int> ranks(const Population& p,
                              const std::vector<int32_t>& live) {
  std::vector<int> rank(p.n, -1);
  for (size_t i = 0; i < live.size(); ++i) rank[live[i]] = int(i);
  return rank;
}
std::vector<double> flows(Population& p, const Settings& s,
                          const Population* memory, Rng& rng) {
  const auto phi = normalize(p, s);
  const auto live = live_rows(p);
  const auto rank = ranks(p, live);
  std::vector<double> result(p.n, 0);
  for (int i = 0; i < p.n; ++i) {
    int j = companion(i, live, rank, rng);
    p.companions[i] = j;
    if (!p.alive[i]) continue;
    const float* x = p.x.data() + size_t(i) * p.d;
    double tabu = 1;
    if (memory) {
      int r = int(rng.randint(0, memory->n));
      tabu = distance2(x, memory->x.data() + size_t(r) * p.d, p.d);
      if (tabu == 0) tabu = 1;
    }
    result[i] = std::pow(1 + phi[i], 2) *
                distance2(x, p.x.data() + size_t(j) * p.d, p.d) * tabu;
    // Snapshot fitness is float, but cloning uses the full double flow.
    p.fitness[i] =
        float(std::min(result[i], double(std::numeric_limits<float>::max())));
  }
  return result;
}
void clone(Population& p, const std::vector<double>& flow, Rng& rng) {
  const auto live = live_rows(p);
  const auto rank = ranks(p, live);
  const auto old_x = p.x;
  const auto old_value = p.objective;
  const auto old_lineage = p.lineage;
  const auto old_fitness = p.fitness;
  for (int i = 0; i < p.n; ++i) {
    int donor = companion(i, live, rank, rng);
    p.clone_companions[i] = donor;
    p.cloned[i] = rng.uniform01() < clone_probability(flow[i], flow[donor]) ||
                  !p.alive[i];
    p.parent[i] = p.cloned[i] ? donor : i;
    if (p.cloned[i]) {
      std::copy_n(old_x.data() + size_t(donor) * p.d, p.d,
                  p.x.data() + size_t(i) * p.d);
      p.objective[i] = old_value[donor];
      p.lineage[i] = old_lineage[donor];
      p.fitness[i] = old_fitness[donor];
      p.alive[i] = true;
    }
  }
}
void propose(const float* original, float* out, int d, const Benchmark& b,
             bool periodic, const Perturbation& noise, double phi, Rng& rng,
             PerturbationTransition* accepted, const std::string& boundary) {
  if (accepted) {
    *accepted = {};
    accepted->origin.assign(original, original + d);
  }
  std::vector<float> delta(d);
  PerturbationContext context{phi};
  double scale = 1;
  for (int retry = 0; retry < 64; ++retry, scale *= .5) {
    noise.sample_with_context(original, delta.data(), d, rng, &context);
    for (int k = 0; k < d; ++k) out[k] = float(original[k] + scale * delta[k]);
    b.boundary(out, boundary.empty() ? (periodic ? "periodic" : "none") : boundary);
    if (b.valid(out)) {
      if (accepted) {
        accepted->draws = 1;
        accepted->scale = scale;
        accepted->displacement.resize(d);
        accepted->direction.assign(delta.begin(),delta.end());
        for(auto& v:accepted->direction) v*=scale;
        for (int k = 0; k < d; ++k)
          accepted->displacement[k] = boundary=="cma" ? out[k]-original[k] : scale * delta[k];
      }
      return;
    }
  }
  std::copy_n(original, d, out);
}
namespace {
struct SearchStopped {};
}  // namespace
Candidate local_search(Benchmark& b, const Settings& s, Candidate best) {
  if (!std::isfinite(best.value)) best.value = s.worst();
  const uint64_t start = b.evaluations, cap = uint64_t(s.gas_local_evaluations);
  LBFGSpp::LBFGSBParam<double> options;
  options.max_iterations = 100;
  options.m = 6;
  options.epsilon = options.epsilon_rel = 1e-5;
  LBFGSpp::LBFGSBSolver<double> solver(options);
  Eigen::VectorXd x(b.d), low = Eigen::VectorXd::Constant(b.d, b.low),
                          high = Eigen::VectorXd::Constant(b.d, b.high);
  for (int k = 0; k < b.d; ++k) x[k] = best.x[k];
  std::vector<float> point(b.d), gradient(b.d);
  auto objective = [&](const Eigen::VectorXd& at, Eigen::VectorXd& grad) {
    if (b.evaluations - start + 1 + b.gradient_evaluations() > cap)
      throw SearchStopped{};
    for (int k = 0; k < b.d; ++k) point[k] = float(at[k]);
    if (!b.valid(point.data())) throw SearchStopped{};
    const double value = b.evaluate_optimization(point.data());
    if (!std::isfinite(value)) throw SearchStopped{};
    if (s.better(value, best.value)) best = {point, value};
    b.gradient(point.data(), gradient.data(), true);
    for (int k = 0; k < b.d; ++k) {
      if (!std::isfinite(gradient[k])) throw SearchStopped{};
      grad[k] = -s.score(gradient[k]);
    }
    return -s.score(value);
  };
  try {
    double value;
    solver.minimize(objective, x, value, low, high);
  } catch (const SearchStopped&) {
    // Budget exhaustion or an invalid trial preserves the best valid candidate.
  } catch (const std::runtime_error&) {
    // A line search can fail on nonsmooth or float-quantized benchmarks.
  }
  return best;
}
std::vector<float> centroid(const Population& p,
                            const std::vector<double>& phi) {
  const double mass = std::accumulate(phi.begin(), phi.end(), 0.);
  std::vector<float> center(p.d);
  for (int k = 0; k < p.d; ++k) {
    double sum = 0;
    for (int i = 0; i < p.n; ++i)
      sum += (mass > 0 ? phi[i] : 1.) * p.x[size_t(i) * p.d + k];
    center[k] = float(sum / (mass > 0 ? mass : p.n));
  }
  return center;
}
void insert_memory(Population& memory, const Settings& s,
                   const Candidate& candidate, Rng& rng) {
  if (!std::isfinite(candidate.value)) return;
  int slot = int(rng.randint(0, memory.n));
  std::copy(candidate.x.begin(), candidate.x.end(),
            memory.x.begin() + size_t(slot) * memory.d);
  memory.objective[slot] = candidate.value;
  auto flow = flows(memory, s, nullptr, rng);
  clone(memory, flow, rng);
}
}  // namespace fg::optimization::gas2017

namespace fg::optimization {
class Gas2017 final : public Algorithm {
  Benchmark& b;
  Settings s;
  OptimizationRng rng;
  std::unique_ptr<Perturbation> noise;
  Population p, memory;
  uint64_t trial_sequence = 0;
  std::vector<std::pair<gas2017::Candidate,uint64_t>> refinements;
  gas2017::Candidate best_walker() const {
    int best = -1;
    for (int i = 0; i < p.n; ++i)
      if (p.alive[i] &&
          (best < 0 || s.better(p.objective[i], p.objective[best])))
        best = i;
    if (best < 0)
      throw std::runtime_error(
          "GAS has no finite walkers; reset or change the benchmark");
    // Uniform tie breaking, performed by callers when randomness is needed.
    return {{p.x.begin() + size_t(best) * p.d,
             p.x.begin() + size_t(best + 1) * p.d},
            p.objective[best]};
  }
  gas2017::Candidate choose_best() {
    auto best = best_walker();
    std::vector<int> ties;
    for (int i = 0; i < p.n; ++i)
      if (p.alive[i] && p.objective[i] == best.value) ties.push_back(i);
    int i = ties[rng.randint(0, ties.size())];
    best.x.assign(p.x.begin() + size_t(i) * p.d,
                  p.x.begin() + size_t(i + 1) * p.d);
    return best;
  }
  void insert(const gas2017::Candidate& candidate) {
    gas2017::insert_memory(memory, s, candidate, rng);
  }

 public:
  Gas2017(Benchmark& bench, const Settings& settings)
      : b(bench),
        s(settings),
        rng(s.seed),
        noise(make_perturbation(b, s.json)) {
    p.resize(s.walkers, b.d);
    for (int i = 0; i < p.n; ++i) {
      float* x = p.x.data() + size_t(i) * p.d;
      b.initial(x, rng);
      p.objective[i] = b.evaluate_optimization(x, &rng);
      p.alive[i] = b.valid(x) && std::isfinite(p.objective[i]);
    }
    auto best = choose_best();
    if (s.gas_local_search) {
      const auto start=b.evaluations;best = gas2017::local_search(b, s, std::move(best));
      refinements.push_back({best,b.evaluations-start});
    }
    if (s.gas_tabu) {
      memory.resize(p.n, p.d);
      for (int i = 0; i < p.n; ++i) {
        std::copy(best.x.begin(), best.x.end(),
                  memory.x.begin() + size_t(i) * p.d);
        memory.objective[i] = best.value;
        memory.alive[i] = true;
      }
    }
  }
  void configure(const Settings& next) override {
    auto proposal = retune_perturbation(*noise, b, s.json, next.json);
    auto random = rng;
    auto population = resized_population(p, next, random);
    Population archive;
    if (memory.n) archive = resized_population(memory, next, random);
    else if (next.gas_tabu) {
      archive = population;
      int best = 0;
      for (int i = 0; i < archive.n; ++i)
        if (archive.alive[i] && (!archive.alive[best] || next.better(archive.objective[i], archive.objective[best]))) best = i;
      if (!archive.alive[best]) throw std::invalid_argument("Tabu memory requires an alive walker");
      for (int i = 0; i < archive.n; ++i) {
        std::copy_n(population.x.data() + size_t(best) * archive.d, archive.d, archive.x.data() + size_t(i) * archive.d);
        archive.objective[i] = population.objective[best];
        archive.alive[i] = true;
      }
    }
    Settings saved = next;
    p = std::move(population);
    memory = std::move(archive);
    noise = std::move(proposal);
    rng = random;
    s = std::move(saved);
  }
  const Population& population() const override { return p; }
  uint64_t evaluations() const override { return b.evaluations; }
  double objective_score(int i) const override {
    return s.score(p.objective.at(i));
  }
  uint64_t next_evaluations_upper_bound() const override {
    // The centroid seed is an additional query; each solver has its own cap.
    return uint64_t(p.n)*(noise->tracks_trials()?adaptive_evaluation_bound(b,s.json):1) +
           (s.gas_local_search ? 1 + 2 * uint64_t(s.gas_local_evaluations) : 0);
  }
  Json refinement_results() const override {
    Json result;result.kind=Json::Array;
    for(const auto& record:refinements) {
      Json item;item.kind=Json::Object;item.object["objective"]=number(record.first.value);
      item.object["position"].kind=Json::Array;for(auto x:record.first.x) item.object["position"].array.push_back(number(x));
      item.object["cost"]=number(record.second);result.array.push_back(std::move(item));
    }
    return result;
  }
  void set_geometry_diagnostics(bool enabled) override {enable_geometry(*noise,b,s.json,enabled);}
  Json movement_geometry() const override {return perturbation_geometry(*noise);}
  void restore_movement_geometry(const Json& geometry) override {restore_perturbation_geometry(*noise,geometry);}
  Json metadata() const override {
    Json result;result.kind=Json::Object;
    result.object["exploration"]=perturbation_diagnostics(*noise);result.object["geometry"]=geometry_diagnostics(*noise);return result;
  }
  void step() override {
    begin_geometry(*noise);
    if(noise->tracks_trials() || noise->observer) noise->observed_freeze({p.d,p.x,p.lineage,p.alive,p.objective});
    auto flow = gas2017::flows(p, s, s.gas_tabu ? &memory : nullptr, rng);
    fractal::FrozenPopulation frozen;
    if(noise->uses_cloning_evidence() || noise->observer) frozen={p.d,p.x,p.lineage,p.alive,p.objective};
    gas2017::clone(p, flow, rng);
    if(noise->uses_cloning_evidence() || noise->observer) {
      fractal::SelectionEvidence evidence;
      evidence.donors=p.clone_companions;evidence.sources=p.parent;evidence.score.resize(p.n);
      // GAS compares lower flow against higher flow, before clipping its gate.
      for(int i=0;i<p.n;++i)
        evidence.score[i]=gas2017::clone_score(flow[i],flow[evidence.donors[i]]);
      if(noise->uses_cloning_evidence()) {noise->observe_cloning(frozen,evidence);noise->update();}
      if(noise->observer) {
        noise->observer->cloning(frozen,evidence);
        for(int i=0;i<p.n;++i) if(p.cloned[i]) noise->observer->movement(frozen.positions.data()+size_t(i)*p.d,p.x.data()+size_t(i)*p.d,p.d,frozen.families[i],"cloning");
      }
    }
    const auto phi = gas2017::normalize(p, s);
    auto best = choose_best();
    refinements.clear();
    if (s.gas_local_search) {
      auto refinement_start=b.evaluations;
      gas2017::Candidate center{gas2017::centroid(p, phi), s.worst()};
      center.value = b.evaluate_optimization(center.x.data());
      center = gas2017::local_search(b, s, std::move(center));
      refinements.push_back({center,b.evaluations-refinement_start});
      if (s.gas_tabu) insert(center);
      refinement_start=b.evaluations;
      best = gas2017::local_search(b, s, std::move(best));
      refinements.push_back({best,b.evaluations-refinement_start});
      if (s.gas_tabu) insert(best);
    } else if (s.gas_tabu)
      insert(best);
    std::vector<float> candidate(p.d);
    for (int i = 0; i < p.n; ++i) {
      float* x = p.x.data() + size_t(i) * p.d;
      const double previous = p.objective[i];
      if(noise->tracks_trials()) {
        const uint64_t action=++trial_sequence;
        auto result=evaluate_adaptive_position(*noise,b,s.json,x,previous,rng,{uint64_t(s.json["round_id"].num()),0,p.lineage[i],action,0});
        std::copy(result.position.begin(),result.position.end(),x);
        p.objective[i]=result.objective;p.alive[i]=result.valid;
        p.lineage[i]=fractal::descendant_lineage(p.lineage[i],action);
        continue;
      }
      PerturbationTransition accepted;
      gas2017::propose(x, candidate.data(), p.d, b, s.periodic, *noise, phi[i],
                       rng, &accepted, s.json["boundary"].str());
      if(noise->observer) noise->observer->movement(x,candidate.data(),p.d,p.lineage[i],"proposal");
      std::copy(candidate.begin(), candidate.end(), x);
      p.objective[i] = b.evaluate_optimization(x, &rng);
      p.alive[i] = b.valid(x) && std::isfinite(p.objective[i]);
      if (p.alive[i] && std::isfinite(previous) && accepted.draws > 0) {
        accepted.improvement = s.score(p.objective[i]) - s.score(previous);
        accepted.parent=p.lineage[i];accepted.action=uint64_t(b.evaluations);
        accepted.source_scale=s.perturbation=="gas_adaptive"?(b.high-b.low)*s.json["gas_scale_multiplier"].num(1)*std::pow(10.,-5+4*phi[i]):s.json["perturbation_std"].num(1);
        noise->observed_transition(accepted);
      }
    }
    noise->observed_update();
  }
};
std::unique_ptr<Algorithm> make_gas2017(Benchmark& b, const Settings& s) {
  return std::make_unique<Gas2017>(b, s);
}
}  // namespace fg::optimization
