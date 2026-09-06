#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "control/actuators.hpp"
#include "control/extensions.hpp"
#include "control/json.hpp"
#include "control/math.hpp"

namespace fg::control {
struct BodyDef {
  Vec2 position, velocity;
  float angle = 0, omega = 0, radius = .5f, mass = 1, inertia = 1, drag = .15f,
        angular_drag = 2;
  float thrust = 12, torque = 8, restitution = .25f, friction = .3f;
  bool controlled = false, cargo = false;
  std::vector<Vec2> vertices;
  ActuatorDef actuator;
};
struct Edge {
  Vec2 a, b, normal;
  Aabb bounds;
};
struct Gravity {
  Vec2 position;
  float strength = 10, softening = 2;
};
struct Zone {
  Vec2 position;
  float radius = 2;
};
struct TetherDef {
  int a = 0, b = -1;
  float rest = 2, stiffness = 25, damping = 6, break_force = 500,
        hook_range = 2;
  bool automatic = false;
};
struct Layout {
  uint32_t bodies = 0, controlled = 0, tethers = 0, pickups = 0;
  uint32_t flags = 0, gates = 0, joints = 0, food = 0, auxiliary = 0,
           auxiliary_words = 0, words = 0, stride = 0;
  Layout() = default;
  Layout(uint32_t b, uint32_t c, uint32_t t, uint32_t p, uint32_t extra = 0)
      : bodies(b), controlled(c), tethers(t), pickups(p) {
    flags = 8 + 6 * b;
    gates = flags + b;
    joints = gates + c;
    food = joints + 2 * t;
    auxiliary = food + 3 * p;
    auxiliary_words = extra;
    words = auxiliary + extra;
    stride = (words + 15) & ~15u;
  }
};
struct Scene {
  std::string json, name, task;
  std::vector<BodyDef> bodies;
  std::vector<int> controlled;
  std::vector<ActionChannel> channels;
  std::vector<WorldExtension> extensions;
  uint32_t extension_observations = 0;
  std::vector<std::vector<Vec2>> boundaries;
  std::vector<Edge> edges;
  std::vector<Gravity> gravity;
  std::vector<Zone> bases, gates, pickups;
  std::vector<TetherDef> tethers;
  Vec2 size{64, 44};
  float dt = 1.f / 60.f;
  int substeps = 4, solver_iterations = 8;
  float collision_penalty = 2, progress_reward = 1, pickup_reward = 10,
        delivery_reward = 100;
  float gate_reward = 30, formation_reward = .15f, formation_distance = 3,
        respawn_seconds = 4;
  bool lethal_walls = false, lethal_bodies = false;
  Layout layout;
  uint64_t fingerprint = 0;
  // Immutable uniform-grid index for boundary edges; one copy per compiled
  // scene.
  float cell_size = 4;
  int grid_w = 0, grid_h = 0;
  std::vector<std::vector<int>> edge_cells;
  static std::shared_ptr<const Scene> compile(const std::string& json);
  bool inside(Vec2 p) const;
  float raycast(Vec2 origin, Vec2 direction, float distance) const;
};
}  // namespace fg::control
