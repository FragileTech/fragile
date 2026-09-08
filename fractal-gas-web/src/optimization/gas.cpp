#include <algorithm>
#include <limits>
#include <numeric>

#include "optimization/engine.hpp"

namespace fg::optimization {
Settings::Settings(const Json& input) : json(input) {
  auto i = [&](const char* k, int f, int lo, int hi) {
    int v = integer(json[k], f, lo, hi, k);
    json.object[k] = number(v);
    return v;
  };
  auto f = [&](const char* k, double v, double lo, double hi) {
    v = bounded(json[k], v, lo, hi, k);
    json.object[k] = number(v);
    return v;
  };
  auto b = [&](const char* k, bool v) {
    v = json[k].flag(v);
    Json j;
    j.kind = Json::Boolean;
    j.number = v;
    json.object[k] = j;
    return v;
  };
  auto s = [&](const char* k, const char* v) {
    std::string t = json[k].str(v);
    Json j;
    j.kind = Json::String;
    j.string = t;
    json.object[k] = j;
    return t;
  };
  algorithm = s("algorithm", "euclidean");
  objective = s("objective", "minimize");
  if (objective != "minimize" && objective != "maximize")
    throw std::invalid_argument("Objective must be minimize or maximize");
  perturbation = s("perturbation", "gaussian");
  companion = s("companion", "cloning");
  clone_companion = s("clone_companion", "cloning");
  for (auto& name : {companion, clone_companion})
    if (name != "cloning" && name != "softmax" && name != "uniform" &&
        name != "random_pairing" && name != "greedy_pairing")
      throw std::invalid_argument("Unknown companion strategy");
  walkers = i("walkers", 256, 2, 100000);
  max_walkers = i("max_walkers", std::max(10000, walkers), walkers, 1000000);
  seed = i("seed", 7, 0, 2147483647);
  dt_min = i("dt_min", 1, 1, 100);
  dt_max = i("dt_max", 1, dt_min, 100);
  clone_every = i("clone_every", 1, 1, 100000);
  substeps = i("substeps", 1, 1, 100);
  elites = i("elites", 0, 0, walkers);
  horizon = i("horizon", 32, 1, 4096);
  max_horizon = i("max_horizon", 0, 0, 4096);
  consensus_prefix = b("consensus_prefix", true);
  proposal = f("proposal", .025, 0, 1);
  // Preserve the scale of recordings made before perturbations were explicit.
  const double legacy_std =
      input["proposal"].kind != Json::Null && algorithm != "euclidean"
          ? proposal * (input["high"].num(5.12) - input["low"].num(-5.12))
          : 1;
  f("perturbation_std", legacy_std, 0, 1e6);
  gamma = f("gamma", 1, 1e-9, 1e6);
  beta = f("beta", 1, 1e-9, 1e12);
  delta_t = f("delta_t", .002, 1e-9, 1);
  epsilon = f("epsilon", .1, 1e-9, 1e6);
  clone_epsilon = f("clone_epsilon", .1, 1e-9, 1e6);
  lambda_alg = f("lambda_alg", 0, 0, 1e6);
  reward_coef = f("reward_coef", 1, 0, 10);
  distance_coef = f("distance_coef", 1, 0, 10);
  eta = f("eta", .1, 0, 100);
  sigma_min = f("sigma_min", 1e-8, 1e-12, 1e6);
  amplitude = f("amplitude", 2, 1e-9, 100);
  epsilon_dist = f("epsilon_dist", 1e-8, 0, 100);
  rho = f("rho", 0, 0, 1e6);
  p_max = f("p_max", 1, 1e-9, 1);
  epsilon_clone = f("epsilon_clone", 1e-6, 1e-12, 100);
  sigma_x = f("sigma_x", 1e-6, 0, 1e6);
  restitution = f("restitution", .5, 0, 1);
  periodic = b("periodic", false);
  potential_force = b("potential_force", true);
  cloning = b("cloning", true);
  kinetic = b("kinetic", true);
}
void Population::resize(int count, int dimensions) {
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
static double distance2(const Population& p, int i, int j, const Settings& s,
                        const Benchmark& b) {
  double q = 0, L = b.high - b.low;
  for (int k = 0; k < p.d; ++k) {
    double dx = p.x[size_t(i) * p.d + k] - p.x[size_t(j) * p.d + k],
           dv = p.v[size_t(i) * p.d + k] - p.v[size_t(j) * p.d + k];
    if (s.periodic) dx -= L * std::nearbyint(dx / L);
    q += dx * dx + s.lambda_alg * dv * dv;
  }
  return q;
}
std::vector<int32_t> select_companions(const Population& p, const Settings& s,
                                       const Benchmark& b,
                                       const std::string& method,
                                       double epsilon, Rng& rng) {
  std::vector<int32_t> result(p.n), alive;
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
  auto weighted = [&](int i, const std::vector<int32_t>& candidates,
                      bool fallback) {
    std::vector<double> weights;
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
    auto order = rng.permutation(int(alive.size()));
    for (size_t j = 0; j + 1 < order.size(); j += 2) {
      int a = alive[order[j]], c = alive[order[j + 1]];
      result[a] = c;
      result[c] = a;
    }
    // The Python pairing helper maps dead walkers to themselves. The gas host
    // revives these dead rows from live donors before cloning (see below).
    return result;
  }
  if (method == "greedy_pairing") {
    auto remaining = alive;
    for (size_t k = 0; k < alive.size() / 2 && remaining.size() > 1; ++k) {
      int a = remaining[0];
      std::vector<int32_t> choices(remaining.begin() + 1, remaining.end());
      int c = weighted(a, choices, false);
      if (c == a) continue;
      result[a] = c;
      result[c] = a;
      remaining.erase(std::remove(remaining.begin(), remaining.end(), a),
                      remaining.end());
      remaining.erase(std::remove(remaining.begin(), remaining.end(), c),
                      remaining.end());
    }
    for (int i = 0; i < p.n; ++i)
      if (!p.alive[i]) result[i] = uniform(alive);
    return result;
  }
  for (int i = 0; i < p.n; ++i) {
    if (method == "uniform" || !p.alive[i]) {
      result[i] = uniform(alive);
      continue;
    }
    auto choices = alive;
    choices.erase(std::remove(choices.begin(), choices.end(), i),
                  choices.end());
    result[i] = choices.empty() ? i : weighted(i, choices, true);
  }
  return result;
}
std::vector<float> fitness(const Population& p, const Settings& s,
                           const Benchmark& b,
                           const std::vector<int32_t>& companions) {
  std::vector<double> rewards(p.n), distances(p.n);
  std::vector<float> out(p.n, 0);
  for (int i = 0; i < p.n; ++i)
    if (p.alive[i]) {
      rewards[i] = s.score(p.objective[i]);
      distances[i] = std::sqrt(distance2(p, i, companions[i], s, b) +
                               s.epsilon_dist * s.epsilon_dist);
    }
  auto stats = [&](const std::vector<double>& values, int i, bool local) {
    double mass = 0, sum = 0;
    for (int j = 0; j < p.n; ++j)
      if (p.alive[j] && (!local || i != j)) {
        double w =
            local ? std::exp(-distance2(p, i, j, s, b) / (2 * s.rho * s.rho))
                  : 1;
        mass += w;
        sum += w * values[j];
      }
    double mean = mass ? sum / mass : 0, var = 0;
    for (int j = 0; j < p.n; ++j)
      if (p.alive[j] && (!local || i != j)) {
        double w =
            local ? std::exp(-distance2(p, i, j, s, b) / (2 * s.rho * s.rho))
                  : 1;
        var += w * std::pow(values[j] - mean, 2);
      }
    return std::vector<double>{
        mean, std::sqrt((mass ? var / mass : 0) + s.sigma_min * s.sigma_min),
        mass};
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
      auto rescale = [&](double v, const std::vector<double>& st) {
        double z = (v - st[0]) / st[1];
        z = std::clamp(z, -50.0, 50.0);
        return s.amplitude / (1 + std::exp(-z)) + s.eta;
      };
      out[i] = float(std::pow(rescale(rewards[i], rs), s.reward_coef) *
                     std::pow(rescale(distances[i], ds), s.distance_coef));
    }
  return out;
}
void clone_population(Population& p, const Settings& s,
                      const std::vector<int32_t>& companions,
                      const std::vector<uint8_t>& mask, Rng& rng) {
  const auto old_x = p.x, old_v = p.v;
  for (int i = 0; i < p.n; ++i)
    if (mask[i])
      for (int k = 0; k < p.d; ++k)
        p.x[size_t(i) * p.d + k] = old_x[size_t(companions[i]) * p.d + k] +
                                   float(s.sigma_x * normal(rng));
  // Sorted donor order and reads from old_v match inelastic_collision_velocity,
  // including overlapping groups in the reference.
  for (int c = 0; c < p.n; ++c) {
    std::vector<int> group{c};
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
void baoab(Population& p, const Settings& s, const Benchmark& b, Rng& rng,
           const Perturbation* noise) {
  std::vector<float> grad(p.x.size(), 0);
  double c1 = std::exp(-s.gamma * s.delta_t),
         c2 = std::sqrt(-std::expm1(-2 * s.gamma * s.delta_t) / s.beta);
  auto kick = [&] {
    if (s.potential_force && !b.stochastic)
      for (int i = 0; i < p.n; ++i)
        b.gradient(p.x.data() + size_t(i) * p.d, grad.data() + size_t(i) * p.d);
    for (size_t j = 0; j < p.x.size(); ++j)
      p.v[j] += float(.5 * s.delta_t * s.score(grad[j]));
  };
  for (int t = 0; t < s.substeps; ++t) {
    kick();
    for (size_t j = 0; j < p.x.size(); ++j)
      p.x[j] += float(.5 * s.delta_t * p.v[j]);
    if (noise) {
      std::vector<float> delta(p.d);
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
    for (size_t j = 0; j < p.x.size(); ++j)
      p.x[j] += float(.5 * s.delta_t * p.v[j]);
    kick();
  }
}
class Euclidean final : public Algorithm {
  Benchmark& b;
  Settings s;
  OptimizationRng rng;
  std::unique_ptr<Perturbation> noise;
  Population p;
  uint64_t ticks = 0, evals = 0;
  void evaluate() {
    for (int i = 0; i < p.n; ++i) {
      float* x = p.x.data() + size_t(i) * p.d;
      if (s.periodic) b.wrap(x);
      p.objective[i] = b.evaluate(x, &rng);
      ++evals;
      p.alive[i] = b.valid(x) && std::isfinite(p.objective[i]);
      for (int k = 0; k < p.d; ++k)
        if (!std::isfinite(p.v[size_t(i) * p.d + k])) p.alive[i] = false;
    }
  }

 public:
  Euclidean(Benchmark& bench, const Settings& settings)
      : b(bench),
        s(settings),
        rng(s.seed),
        noise(make_perturbation(b, s.json)) {
    p.resize(s.walkers, b.d);
    p.has_velocity = true;
    for (int i = 0; i < p.n; ++i) b.initial(p.x.data() + size_t(i) * p.d, rng);
    evaluate();
  }
  const Population& population() const override { return p; }
  uint64_t evaluations() const override { return evals; }
  double objective_score(int i) const override {
    return s.score(p.objective.at(i));
  }
  void step() override {
    p.companions = select_companions(p, s, b, s.companion, s.epsilon, rng);
    p.fitness = fitness(p, s, b, p.companions);
    std::fill(p.cloned.begin(), p.cloned.end(), 0);
    std::iota(p.parent.begin(), p.parent.end(), 0);
    if (s.cloning) {
      p.clone_companions =
          select_companions(p, s, b, s.clone_companion, s.clone_epsilon, rng);
      std::vector<int32_t> live;
      for (int i = 0; i < p.n; ++i)
        if (p.alive[i]) live.push_back(i);
      std::vector<uint8_t> mask(p.n);
      for (int i = 0; i < p.n; ++i) {
        if (!p.alive[i] && !p.alive[p.clone_companions[i]])
          p.clone_companions[i] = live[size_t(rng.randint(0, live.size()))];
        double score = (p.fitness[p.clone_companions[i]] - p.fitness[i]) /
                       (p.fitness[i] + s.epsilon_clone);
        mask[i] = rng.uniform01() < std::clamp(score / s.p_max, 0.0, 1.0) ||
                  !p.alive[i];
      }
      // As in Python, draws occur even on skipped cloning iterations.
      Population proposed = p;
      clone_population(proposed, s, p.clone_companions, mask, rng);
      if (ticks % uint64_t(s.clone_every) == 0) p = std::move(proposed);
    }
    if (s.kinetic) baoab(p, s, b, rng, noise.get());
    ++ticks;
    evaluate();
  }
};
std::unique_ptr<Algorithm> make_euclidean(Benchmark& b, const Settings& s) {
  return std::make_unique<Euclidean>(b, s);
}
}  // namespace fg::optimization
