#pragma once
#include <atomic>

#include "control/collision_geometry.hpp"
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
  float depth = 0, impulse = 0, target_velocity = 0, tangent_impulse = 0;
  int edge = -1;
  bool counted = false;
};
#ifdef FG_CONTROL_PROFILE
struct CollisionWork {
  uint64_t edge_queries = 0, edge_candidates = 0, separation_checks = 0,
           ccd_iterations = 0;
};
#endif
struct Scratch {
#ifdef FG_CONTROL_PROFILE
  CollisionWork work;
#endif
  std::vector<Contact> contacts, wall_contacts;
  std::vector<geometry::ShapeCache> shapes;
  std::vector<int> order, edge_ids;
  std::vector<Aabb> bounds, empty_edge_bounds;
  std::vector<uint8_t> empty_edge_valid;
  std::vector<Vec2> old_positions, frame_positions, frame_rock_positions;
  std::vector<float> old_angles, bounded_actions;
  std::vector<uint32_t> edge_marks, frame_tethers;
  std::vector<uint8_t> frame_hooked_rocks, frame_wall_contacts;
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
#ifdef FG_CONTROL_PROFILE
  CollisionWork collision_work() const;
#endif
  size_t observation_dim() const;
  void observe(const float* row, float* output) const;
  std::vector<float> inspect(const float* row, const float* actions) const;

 private:
  std::vector<Scratch> scratch_;
  std::vector<geometry::Shape> walls_;
  void edge_candidates(Scratch& scratch, Aabb box);
  void substep(float* row, const float* actions, float h, Scratch& scratch,
               StepResult& result);
  float potential(const float* row, const uint32_t* attachments = nullptr) const;
  void mechanics(float* row, StepResult& result, uint32_t checkpoint_stage);
};
}  // namespace fg::control
