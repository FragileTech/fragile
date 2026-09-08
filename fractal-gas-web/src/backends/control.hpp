#pragma once
#include "control/physics.hpp"
#include "fractal/population.hpp"
#include "rng.hpp"

namespace fg::control {
using PackedPopulation = fractal::Population<StateBatch, StepResult, float>;
struct ControlBackend {
  Physics& physics;
  PackedPopulation scratch;
  explicit ControlBackend(Physics& p) : physics(p) {}
  void resize(PackedPopulation& s, int n) {
    if (s.states.count != uint32_t(n) || !s.states.storage)
      s.states = StateBatch(n, *physics.scene);
  }
  void copy(const PackedPopulation& from, size_t i, PackedPopulation& to, size_t j) {
    std::memcpy(to.states.row(j), from.states.row(i), from.states.layout.words * 4);
  }
  void observe(PackedPopulation& s) {
    for (int i = 0; i < s.N; ++i) s.dones[i] = word(s.states.row(i), 7) != 0;
  }
  void fitness_bonus(const PackedPopulation&, std::vector<float>&) {}
  void transition(const PackedPopulation& from, fractal::Selection rows,
                  const std::vector<float>& actions, const std::vector<int32_t>& dt,
                  PackedPopulation& to) {
    if (actions.size() != rows.count * size_t(to.action_dim) || dt.size() != rows.count ||
        (rows.sources.data && rows.sources.size != rows.count) ||
        (rows.destinations.data && rows.destinations.size != rows.count))
      throw std::invalid_argument("Invalid packed transition shape");
    for (size_t i = 0; i < rows.count; ++i)
      if (rows.sources.index(i) >= from.states.count ||
          rows.destinations.index(i) >= to.states.count)
        throw std::invalid_argument("Invalid packed transition index");
    if (rows.destinations.data) {
      scratch.has_infos = true;
      scratch.resize_metadata(rows.count, to.obs_dim, to.action_dim);
      resize(scratch, rows.count);
      transition(from, fractal::Selection{rows.count, rows.sources, {}}, actions, dt, scratch);
      for (size_t i = 0; i < rows.count; ++i) {
        size_t j = rows.destinations.index(i);
        copy(scratch, i, to, j);
        fractal::copy_transition(scratch, i, to, j);
      }
      return;
    }
    physics.step(from.states, rows.sources.data, actions.data(), dt.data(), to.states,
                 to.infos.data());
    physics.pool.parallel_for(to.N, [&](int i, int) {
      const auto& r = to.infos[i];
      to.step_rewards[i] = r.reward;
      to.dones[i] = r.dead;
      to.truncated[i] = to.recoverable[i] = 0;
      to.actual_dt[i] = r.frames;
      physics.observe(to.states.row(i), to.observations.data() + size_t(i) * to.obs_dim);
    });
  }
  void after_transition(const PackedPopulation&) {}
  void pose(const PackedPopulation& s, size_t i, float* out) {
    for (size_t k = 0; k < physics.scene->controlled.size(); ++k) {
      auto p = position(s.states.row(i), s.states.layout, physics.scene->controlled[k]);
      out[2 * k] = p.x;
      out[2 * k + 1] = p.y;
    }
  }
  uint32_t flags(const PackedPopulation& s, size_t i) {
    uint32_t f = s.alive(i) ? 0 : 1;
    for (size_t t = 0; t < s.states.layout.tethers; ++t)
      if (word(s.states.row(i), s.states.layout.joints + 2 * t)) f |= 2;
    return f;
  }
};
struct ContinuousActions {
  const Scene& scene;
  bool inertial = true;
  float noise = .2f;
  int frames = 6;
  void sample(const PackedPopulation& s, const std::vector<int32_t>& sources, bool first,
              std::vector<float>& actions, std::vector<int32_t>& dt, Rng& rng) {
    const size_t d = scene.channels.size();
    for (int i = 0; i < s.N; ++i) {
      dt[i] = frames;
      for (size_t k = 0; k < d; ++k) {
        float a;
        if (inertial && !first) {
          float u = std::max(1e-7f, rng.uniform01()), v = rng.uniform01();
          a = s.actions[size_t(sources[i]) * d + k] +
              noise * std::sqrt(-2 * std::log(u)) * std::cos(2 * pi * v);
        } else
          a = scene.channels[k].low +
              rng.uniform01() * (scene.channels[k].high - scene.channels[k].low);
        actions[size_t(i) * d + k] = std::clamp(a, scene.channels[k].low, scene.channels[k].high);
      }
    }
  }
};
}  // namespace fg::control
