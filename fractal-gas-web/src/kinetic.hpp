// Translation of RandomActionOperator
// (src/fragile/fractalai/videogames/kinetic.py): uniform random discrete
// actions plus a random frame-skip dt ~ randint[dt_min, dt_max] (inclusive)
// per walker. Sampling methods are virtual so replay tests can inject
// recorded draws. All sampling happens on the caller thread; only
// env.step_batch parallelizes.
#ifndef FRACTAL_GAS_KINETIC_HPP
#define FRACTAL_GAS_KINETIC_HPP

#include <cstdint>
#include <vector>

#include "env.hpp"
#include "rng.hpp"

namespace fg {

class RandomActionOperator {
 public:
  int32_t dt_min = 1;
  int32_t dt_max = 4;  // DEFAULT_DT_RANGE = (1, 4)

  // Mirrors kinetic.py's last_actions / last_dt tracking.
  std::vector<int32_t> last_actions;
  std::vector<int32_t> last_dt;

  virtual ~RandomActionOperator() = default;

  virtual std::vector<int32_t> sample_actions(int32_t n, int32_t n_actions,
                                              Rng& rng) const {
    std::vector<int32_t> actions(static_cast<size_t>(n));
    for (auto& a : actions) a = static_cast<int32_t>(rng.randint(0, n_actions));
    return actions;
  }

  virtual std::vector<int32_t> sample_dt(int32_t n, Rng& rng) const {
    // np.random.randint(dt_min, dt_max + 1)
    std::vector<int32_t> dt(static_cast<size_t>(n));
    for (auto& d : dt) d = static_cast<int32_t>(rng.randint(dt_min, dt_max + 1));
    return dt;
  }

  /// kinetic.py::apply(): sample actions (unless provided), sample dt, step
  /// the env batch. Outputs are resized here.
  void apply(BatchEnv& env, const std::vector<std::vector<char>>& states,
             const std::vector<int32_t>* external_actions, Rng& rng,
             std::vector<std::vector<char>>& new_states,
             std::vector<float>& observations, std::vector<float>& rewards,
             std::vector<uint8_t>& dones, std::vector<uint8_t>& truncated) {
    const auto n = static_cast<int32_t>(states.size());
    // Python order: actions first, then dt.
    last_actions = external_actions ? *external_actions
                                    : sample_actions(n, env.n_actions(), rng);
    last_dt = sample_dt(n, rng);

    new_states.resize(static_cast<size_t>(n));
    observations.resize(static_cast<size_t>(n) *
                        static_cast<size_t>(env.obs_dim()));
    rewards.assign(static_cast<size_t>(n), 0.0f);
    dones.assign(static_cast<size_t>(n), 0);
    truncated.assign(static_cast<size_t>(n), 0);

    env.step_batch(states, last_actions, last_dt, new_states, observations,
                   rewards, dones, truncated);
  }
};

}  // namespace fg

#endif  // FRACTAL_GAS_KINETIC_HPP
