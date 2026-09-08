#include "optimization/environment.hpp"

#include <cstring>

namespace fg::optimization {
BenchmarkEnvironment::BenchmarkEnvironment(Benchmark& bench,
                                           const Settings& settings)
    : b(bench), s(settings), perturbation(make_perturbation(b, s.json)) {}
void BenchmarkEnvironment::encode(std::vector<char>& state, const float* x,
                                  double u) const {
  state.assign(bytes(), 0);
  state[0] = 1;
  std::memcpy(state.data() + 1, &u, sizeof(u));
  std::memcpy(state.data() + 1 + sizeof(u), x, sizeof(float) * b.d);
}
void BenchmarkEnvironment::reset(std::vector<char>& state,
                                 std::vector<float>& obs) {
  perturbation->reset();
  collecting = true;
  state.assign(bytes(), 0);
  obs.assign(b.d, 0);
  if (s.planning()) {
    OptimizationRng rng(s.seed);
    b.initial(obs.data(), rng);
    const double u = b.evaluate_optimization(obs.data(), &rng);
    encode(state, obs.data(), u);
  }
}
double BenchmarkEnvironment::decode(const std::vector<char>& state,
                                    float* x) const {
  if (state.size() != bytes()) return INFINITY;
  double u;
  std::memcpy(&u, state.data() + 1, sizeof(u));
  std::memcpy(x, state.data() + 1 + sizeof(u), sizeof(float) * b.d);
  return state[0] ? u : INFINITY;
}
void BenchmarkEnvironment::step_batch(
    const std::vector<std::vector<char>>& states,
    const std::vector<int32_t>& actions, const std::vector<int32_t>& dt,
    std::vector<std::vector<char>>& next, std::vector<float>& observations,
    std::vector<float>& rewards, std::vector<uint8_t>& dones,
    std::vector<uint8_t>& truncated) {
  std::vector<float> delta(b.d);
  for (size_t i = 0; i < states.size(); ++i) {
    if (actions[i] < 0 || actions[i] >= n_actions())
      throw std::invalid_argument("Invalid perturbation action seed");
    OptimizationRng rng((uint64_t(s.seed) << 24) | uint32_t(actions[i]));
    float* x = observations.data() + i * b.d;
    const bool initialized = states[i].size() == bytes() && states[i][0];
    const double old = initialized ? decode(states[i], x) : 0;
    if (!initialized) b.initial(x, rng);
    PerturbationTransition transition;
    transition.origin.assign(x, x + b.d);
    transition.displacement.assign(b.d, 0);
    for (int t = 0; t < dt[i]; ++t) {
      if (initialized || t > 0) {
        perturbation->sample_action(transition.origin.data(), x, delta.data(), b.d, rng);
        ++transition.draws;
        for (int k = 0; k < b.d; ++k) {
          x[k] += delta[k];
          transition.displacement[k] += delta[k];
        }
      }
      if (s.periodic) b.wrap(x);
      if (!b.valid(x)) break;
    }
    const double u = b.evaluate_optimization(x, &rng);
    const bool valid = b.valid(x) && std::isfinite(u) &&
                       std::isfinite(float(u)) &&
                       std::isfinite(float(s.score(u) - s.score(old)));
    if (collecting && initialized && valid && transition.draws > 0) {
      transition.improvement = s.score(u) - s.score(old);
      perturbation->observe(transition);
    }
    encode(next[i], x, u);
    rewards[i] = valid ? float(s.score(u) - s.score(old)) : 0;
    dones[i] = !valid;
    truncated[i] = 0;
  }
}
}  // namespace fg::optimization
