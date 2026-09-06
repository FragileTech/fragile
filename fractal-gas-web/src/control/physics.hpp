#pragma once
#include <atomic>

#include "control/state.hpp"
#include "thread_pool.hpp"

namespace fg::control {
struct StepResult {
  float reward = 0;
  uint32_t frames = 0, collisions = 0, ccd_limits = 0;
  uint8_t dead = 0;
};
struct Contact {
  int a = -1, b = -1;
  Vec2 normal, point;
  float depth = 0, impulse = 0, target_velocity = 0;
};
struct Scratch {
  std::vector<Contact> contacts;
  std::vector<int> order, edge_ids;
  std::vector<Aabb> bounds;
  std::vector<Vec2> old_positions;
  std::vector<float> old_angles, bounded_actions;
  std::vector<uint32_t> edge_marks;
  uint32_t stamp = 0;
  explicit Scratch(const Scene& s);
};
class Physics {
 public:
  std::shared_ptr<const Scene> scene;
  ThreadPool pool;
  explicit Physics(std::shared_ptr<const Scene> s, int threads = 1);
  void step(const StateBatch& input, const int32_t* sources,
            const float* actions, const int32_t* frames, StateBatch& output,
            StepResult* results);
  void step_world(float* row, const float* actions, int frames,
                  StepResult& result, int slot = 0);
  size_t observation_dim() const;
  void observe(const float* row, float* output) const;
  std::vector<float> inspect(const float* row, const float* actions) const;

 private:
  std::vector<Scratch> scratch_;
  void edge_candidates(Scratch& scratch, Aabb box);
  void substep(float* row, const float* actions, float h, Scratch& scratch,
               StepResult& result);
  float potential(const float* row) const;
  void mechanics(float* row, StepResult& result);
};
}  // namespace fg::control
