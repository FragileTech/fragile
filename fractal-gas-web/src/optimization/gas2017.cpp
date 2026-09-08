#include "optimization/gas2017.hpp"

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
double clone_probability(double flow, double donor_flow) {
  return flow > 0 && donor_flow < flow
             ? std::clamp(1 - donor_flow / flow, 0., 1.)
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
      p.fitness[i] = old_fitness[donor];
      p.alive[i] = true;
    }
  }
}
void propose(const float* original, float* out, int d, const Benchmark& b,
             bool periodic, const Perturbation& noise, double phi, Rng& rng,
             PerturbationTransition* accepted) {
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
    if (periodic) b.wrap(out);
    if (b.valid(out)) {
      if (accepted) {
        accepted->draws = 1;
        accepted->scale = scale;
        accepted->displacement.resize(d);
        for (int k = 0; k < d; ++k)
          accepted->displacement[k] = scale * delta[k];
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
    if (s.gas_local_search) best = gas2017::local_search(b, s, std::move(best));
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
  const Population& population() const override { return p; }
  uint64_t evaluations() const override { return b.evaluations; }
  double objective_score(int i) const override {
    return s.score(p.objective.at(i));
  }
  uint64_t next_evaluations_upper_bound() const override {
    // The centroid seed is an additional query; each solver has its own cap.
    return uint64_t(p.n) +
           (s.gas_local_search ? 1 + 2 * uint64_t(s.gas_local_evaluations) : 0);
  }
  void step() override {
    auto flow = gas2017::flows(p, s, s.gas_tabu ? &memory : nullptr, rng);
    gas2017::clone(p, flow, rng);
    const auto phi = gas2017::normalize(p, s);
    auto best = choose_best();
    if (s.gas_local_search) {
      gas2017::Candidate center{gas2017::centroid(p, phi), s.worst()};
      center.value = b.evaluate_optimization(center.x.data());
      center = gas2017::local_search(b, s, std::move(center));
      if (s.gas_tabu) insert(center);
      best = gas2017::local_search(b, s, std::move(best));
      if (s.gas_tabu) insert(best);
    } else if (s.gas_tabu)
      insert(best);
    std::vector<float> candidate(p.d);
    for (int i = 0; i < p.n; ++i) {
      float* x = p.x.data() + size_t(i) * p.d;
      const double previous = p.objective[i];
      PerturbationTransition accepted;
      gas2017::propose(x, candidate.data(), p.d, b, s.periodic, *noise, phi[i],
                       rng, &accepted);
      std::copy(candidate.begin(), candidate.end(), x);
      p.objective[i] = b.evaluate_optimization(x, &rng);
      p.alive[i] = b.valid(x) && std::isfinite(p.objective[i]);
      if (p.alive[i] && std::isfinite(previous) && accepted.draws > 0) {
        accepted.improvement = s.score(p.objective[i]) - s.score(previous);
        noise->observe(accepted);
      }
    }
    noise->update();
  }
};
std::unique_ptr<Algorithm> make_gas2017(Benchmark& b, const Settings& s) {
  return std::make_unique<Gas2017>(b, s);
}
}  // namespace fg::optimization
