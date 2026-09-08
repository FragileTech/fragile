#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include "fractal/proposal.hpp"
namespace fg::fractal {
struct EuclideanConfig {
  int walkers = 256, clone_every = 1, substeps = 1;
  std::string companion = "cloning", clone_companion = "cloning";
  bool periodic = false, potential_force = true, cloning = true, kinetic = true, minimize = true;
  double lambda_alg = 0, epsilon = .1, clone_epsilon = .1, epsilon_dist = 1e-8, rho = 0,
         sigma_min = 1e-8, amplitude = 2, eta = .1, reward_coef = 1, distance_coef = 1,
         sigma_x = 1e-6, restitution = .5, gamma = 1, delta_t = .002, beta = 1,
         epsilon_clone = 1e-6, p_max = 1;
  double score(double value) const { return minimize ? -value : value; }
};
struct EuclideanPopulation {
  int n = 0, d = 0;
  bool has_velocity = false;
  std::vector<float> x, v, fitness;
  std::vector<double> objective;
  std::vector<uint8_t> alive, cloned, leaf;
  std::vector<int32_t> companions, clone_companions, parent;
  void resize(int count, int dimensions);
};
inline double normal(Rng& rng) {
  double u = std::max(1e-12f, rng.uniform01());
  return std::sqrt(-2 * std::log(u)) * std::cos(6.28318530717958647692 * double(rng.uniform01()));
}
inline void EuclideanPopulation::resize(int count, int dimensions) {
  n = count;
  d = dimensions;
  x.resize(size_t(n) * d);
  v.assign(x.size(), 0);
  fitness.assign(n, 0);
  objective.assign(n, INFINITY);
  alive.assign(n, 0);
  cloned.assign(n, 0);
  leaf.assign(n, 1);
  companions.resize(n);
  clone_companions.resize(n);
  parent.resize(n);
  std::iota(companions.begin(), companions.end(), 0);
  clone_companions = parent = companions;
}
template <class Domain>
double distance2(const EuclideanPopulation& p, int i, int j, const EuclideanConfig& s,
                 const Domain& b) {
  double q = 0, L = b.high - b.low;
  for (int k = 0; k < p.d; ++k) {
    double dx = p.x[size_t(i) * p.d + k] - p.x[size_t(j) * p.d + k],
           dv = p.v[size_t(i) * p.d + k] - p.v[size_t(j) * p.d + k];
    if (s.periodic) dx -= L * std::nearbyint(dx / L);
    q += dx * dx + s.lambda_alg * dv * dv;
  }
  return q;
}
struct EuclideanScratch {
  std::vector<int32_t> alive, remaining, choices, order, group;
  std::vector<double> weights, rewards, distances;
  std::vector<float> old_x, old_v, grad, delta;
  std::vector<uint8_t> mask;
};
template <class Domain>
void select_companions_into(const EuclideanPopulation& p, const EuclideanConfig& s,
                            const Domain& b, const std::string& method, double epsilon, Rng& rng,
                            std::vector<int32_t>& result, EuclideanScratch& scratch) {
  auto& alive = scratch.alive;
  alive.clear();
  result.resize(p.n);
  std::iota(result.begin(), result.end(), 0);
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i]) alive.push_back(i);
  if (alive.empty())
    throw std::runtime_error(
        "All walkers are invalid or outside the domain. "
        "Reset or change bounds/time step.");
  auto uniform = [&](const std::vector<int32_t>& candidates) {
    return candidates[size_t(rng.randint(0, candidates.size()))];
  };
  auto weighted = [&](int i, const std::vector<int32_t>& candidates, bool fallback) {
    auto& weights = scratch.weights;
    weights.clear();
    double sum = 0;
    for (int j : candidates) {
      double w = std::exp(-distance2(p, i, j, s, b) / (2 * epsilon * epsilon));
      weights.push_back(w);
      sum += w;
    }
    if (sum < (fallback ? 1e-30 : std::numeric_limits<double>::min()))
      return fallback ? uniform(candidates) : i;
    double u = rng.uniform01() * sum;
    for (size_t j = 0; j < candidates.size(); ++j) {
      u -= weights[j];
      if (u <= 0) return candidates[j];
    }
    return candidates.back();
  };
  if (method == "random_pairing") {
    auto& order = scratch.order;
    rng.permutation_into(int(alive.size()), order);
    for (size_t j = 0; j + 1 < order.size(); j += 2) {
      int a = alive[order[j]], c = alive[order[j + 1]];
      result[a] = c;
      result[c] = a;
    }
    // The Python pairing helper maps dead walkers to themselves. The gas host
    // revives these dead rows from live donors before cloning (see below).
    return;
  }
  if (method == "greedy_pairing") {
    auto& remaining = scratch.remaining;
    remaining = alive;
    for (size_t k = 0; k < alive.size() / 2 && remaining.size() > 1; ++k) {
      int a = remaining[0];
      auto& choices = scratch.choices;
      choices.assign(remaining.begin() + 1, remaining.end());
      int c = weighted(a, choices, false);
      if (c == a) continue;
      result[a] = c;
      result[c] = a;
      remaining.erase(std::remove(remaining.begin(), remaining.end(), a), remaining.end());
      remaining.erase(std::remove(remaining.begin(), remaining.end(), c), remaining.end());
    }
    for (int i = 0; i < p.n; ++i)
      if (!p.alive[i]) result[i] = uniform(alive);
    return;
  }
  for (int i = 0; i < p.n; ++i) {
    if (method == "uniform" || !p.alive[i]) {
      result[i] = uniform(alive);
      continue;
    }
    auto& choices = scratch.choices;
    choices = alive;
    choices.erase(std::remove(choices.begin(), choices.end(), i), choices.end());
    result[i] = choices.empty() ? i : weighted(i, choices, true);
  }
  return;
}
template <class Domain>
std::vector<int32_t> select_companions(const EuclideanPopulation& p, const EuclideanConfig& s,
                                       const Domain& b, const std::string& method, double epsilon,
                                       Rng& rng) {
  EuclideanScratch scratch;
  std::vector<int32_t> out;
  select_companions_into(p, s, b, method, epsilon, rng, out, scratch);
  return out;
}
template <class Domain>
void fitness_into(const EuclideanPopulation& p, const EuclideanConfig& s, const Domain& b,
                  const std::vector<int32_t>& companions, std::vector<float>& out,
                  EuclideanScratch& scratch) {
  auto& rewards = scratch.rewards;
  auto& distances = scratch.distances;
  rewards.resize(p.n);
  distances.resize(p.n);
  out.assign(p.n, 0);
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i]) {
      rewards[i] = s.score(p.objective[i]);
      distances[i] =
          std::sqrt(distance2(p, i, companions[i], s, b) + s.epsilon_dist * s.epsilon_dist);
    }
  auto stats = [&](const std::vector<double>& values, int i, bool local) {
    double mass = 0, sum = 0;
    for (int j = 0; j < p.n; ++j)
      if (p.alive[j] && (!local || i != j)) {
        double w = local ? std::exp(-distance2(p, i, j, s, b) / (2 * s.rho * s.rho)) : 1;
        mass += w;
        sum += w * values[j];
      }
    double mean = mass ? sum / mass : 0, var = 0;
    for (int j = 0; j < p.n; ++j)
      if (p.alive[j] && (!local || i != j)) {
        double w = local ? std::exp(-distance2(p, i, j, s, b) / (2 * s.rho * s.rho)) : 1;
        var += w * std::pow(values[j] - mean, 2);
      }
    return std::array<double, 3>{
        mean, std::sqrt((mass ? var / mass : 0) + s.sigma_min * s.sigma_min), mass};
  };
  auto rg = stats(rewards, 0, false), dg = stats(distances, 0, false);
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i]) {
      auto rs = s.rho > 0 ? stats(rewards, i, true) : rg,
           ds = s.rho > 0 ? stats(distances, i, true) : dg;
      if (rs[2] <= std::numeric_limits<float>::epsilon()) {
        rs = rg;
        ds = dg;
      }
      auto rescale = [&](double v, const std::array<double, 3>& st) {
        double z = (v - st[0]) / st[1];
        z = std::clamp(z, -50.0, 50.0);
        return s.amplitude / (1 + std::exp(-z)) + s.eta;
      };
      out[i] = float(std::pow(rescale(rewards[i], rs), s.reward_coef) *
                     std::pow(rescale(distances[i], ds), s.distance_coef));
    }
}
template <class Domain>
std::vector<float> fitness(const EuclideanPopulation& p, const EuclideanConfig& s, const Domain& b,
                           const std::vector<int32_t>& companions) {
  EuclideanScratch scratch;
  std::vector<float> out;
  fitness_into(p, s, b, companions, out, scratch);
  return out;
}
inline void clone_population(EuclideanPopulation& p, const EuclideanConfig& s,
                             const std::vector<int32_t>& companions,
                             const std::vector<uint8_t>& mask, Rng& rng,
                             EuclideanScratch* workspace = nullptr) {
  EuclideanScratch local;
  auto& scratch = workspace ? *workspace : local;
  auto& old_x = scratch.old_x;
  auto& old_v = scratch.old_v;
  old_x = p.x;
  old_v = p.v;
  for (int i = 0; i < p.n; ++i)
    if (mask[i])
      for (int k = 0; k < p.d; ++k)
        p.x[size_t(i) * p.d + k] =
            old_x[size_t(companions[i]) * p.d + k] + float(s.sigma_x * normal(rng));
  // Sorted donor order and reads from old_v match inelastic_collision_velocity,
  // including overlapping groups in the reference.
  for (int c = 0; c < p.n; ++c) {
    auto& group = scratch.group;
    group.assign(1, c);
    bool used = false;
    for (int i = 0; i < p.n; ++i)
      if (mask[i] && companions[i] == c) {
        used = true;
        if (i != c) group.push_back(i);
      }
    if (!used) continue;
    for (int k = 0; k < p.d; ++k) {
      double mean = 0;
      for (int i : group) mean += old_v[size_t(i) * p.d + k];
      mean /= group.size();
      for (int i : group)
        p.v[size_t(i) * p.d + k] =
            float(mean + s.restitution * (old_v[size_t(i) * p.d + k] - mean));
    }
  }
  p.cloned = mask;
  p.parent = companions;
  for (int i = 0; i < p.n; ++i)
    if (!mask[i]) p.parent[i] = i;
}
template <class Domain>
void baoab(EuclideanPopulation& p, const EuclideanConfig& s, const Domain& b, Rng& rng,
           const Perturbation* noise, EuclideanScratch* workspace = nullptr) {
  EuclideanScratch local;
  auto& scratch = workspace ? *workspace : local;
  auto& grad = scratch.grad;
  grad.assign(p.x.size(), 0);
  double c1 = std::exp(-s.gamma * s.delta_t),
         c2 = std::sqrt(-std::expm1(-2 * s.gamma * s.delta_t) / s.beta);
  auto kick = [&] {
    if (s.potential_force && !b.stochastic)
      for (int i = 0; i < p.n; ++i)
        b.gradient(p.x.data() + size_t(i) * p.d, grad.data() + size_t(i) * p.d);
    for (size_t j = 0; j < p.x.size(); ++j) p.v[j] += float(.5 * s.delta_t * s.score(grad[j]));
  };
  for (int t = 0; t < s.substeps; ++t) {
    kick();
    for (size_t j = 0; j < p.x.size(); ++j) p.x[j] += float(.5 * s.delta_t * p.v[j]);
    if (noise) {
      auto& delta = scratch.delta;
      delta.resize(p.d);
      for (int i = 0; i < p.n; ++i) {
        noise->sample(p.x.data() + size_t(i) * p.d, delta.data(), p.d, rng);
        for (int k = 0; k < p.d; ++k) {
          auto& v = p.v[size_t(i) * p.d + k];
          v = float(c1 * v + c2 * delta[k]);
        }
      }
    } else {
      for (auto& v : p.v) v = float(c1 * v + c2 * normal(rng));
    }
    for (size_t j = 0; j < p.x.size(); ++j) p.x[j] += float(.5 * s.delta_t * p.v[j]);
    kick();
  }
}
template <class Domain>
class Euclidean {
  Domain& b;
  EuclideanConfig s;
  Rng& rng;
  Perturbation& noise;
  EuclideanPopulation p, proposed;
  EuclideanScratch scratch;
  uint64_t ticks = 0;
  void evaluate() {
    for (int i = 0; i < p.n; ++i) {
      float* x = p.x.data() + size_t(i) * p.d;
      if (s.periodic) b.wrap(x);
      p.objective[i] = b.evaluate(x, &rng);
      p.alive[i] = b.valid(x) && std::isfinite(p.objective[i]);
      for (int k = 0; k < p.d; ++k)
        if (!std::isfinite(p.v[size_t(i) * p.d + k])) p.alive[i] = false;
    }
  }

 public:
  Euclidean(Domain& domain, const EuclideanConfig& config, Rng& random, Perturbation& proposal)
      : b(domain), s(config), rng(random), noise(proposal) {
    p.resize(s.walkers, b.d);
    p.has_velocity = true;
    for (int i = 0; i < p.n; ++i) b.initial(p.x.data() + size_t(i) * p.d, rng);
    evaluate();
  }
  const EuclideanPopulation& population() const { return p; }
  uint64_t evaluations() const { return b.evaluations(); }
  uint64_t next_evaluations_upper_bound() const {
    return uint64_t(p.n) * (1 + (s.kinetic && s.potential_force && !b.stochastic
                                     ? 2 * uint64_t(s.substeps) * b.gradient_evaluations()
                                     : 0));
  }
  double objective_score(int i) const { return s.score(p.objective.at(i)); }
  void step() {
    select_companions_into(p, s, b, s.companion, s.epsilon, rng, p.companions, scratch);
    fitness_into(p, s, b, p.companions, p.fitness, scratch);
    std::fill(p.cloned.begin(), p.cloned.end(), 0);
    std::iota(p.parent.begin(), p.parent.end(), 0);
    if (s.cloning) {
      select_companions_into(p, s, b, s.clone_companion, s.clone_epsilon, rng, p.clone_companions,
                             scratch);
      const auto& live = scratch.alive;
      auto& mask = scratch.mask;
      mask.resize(p.n);
      for (int i = 0; i < p.n; ++i) {
        if (!p.alive[i] && !p.alive[p.clone_companions[i]])
          p.clone_companions[i] = live[size_t(rng.randint(0, live.size()))];
        double score =
            (p.fitness[p.clone_companions[i]] - p.fitness[i]) / (p.fitness[i] + s.epsilon_clone);
        mask[i] = rng.uniform01() < std::clamp(score / s.p_max, 0.0, 1.0) || !p.alive[i];
      }
      // As in Python, draws occur even on skipped cloning iterations.
      proposed = p;
      clone_population(proposed, s, p.clone_companions, mask, rng, &scratch);
      if (ticks % uint64_t(s.clone_every) == 0) std::swap(p, proposed);
    }
    if (s.kinetic) baoab(p, s, b, rng, &noise, &scratch);
    ++ticks;
    evaluate();
  }
};
}  // namespace fg::fractal
