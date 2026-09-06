#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "control/json.hpp"
#include "control/math.hpp"

namespace fg::control {
struct BodyDef;
struct ThrusterDef {
  Vec2 point{}, direction{1, 0};
  float force = 1;
  bool reversible = false;
};
struct ActuatorDef;
struct ActuatorForce {
  Vec2 force{};
  float torque = 0;
};
using ActuatorEvaluator = ActuatorForce (*)(const BodyDef&, Vec2, float, float,
                                            const float*, float, float*);
struct ChannelSpec {
  std::string name;
  float low = 0, high = 1;
};
struct ActuatorDef {
  enum Kind { Vector, Kart, Thrusters, Holonomic } kind = Vector;
  uint32_t offset = 0, dimensions = 2;
  float wheelbase = 1, steering_limit = .55f, lateral_grip = 12,
        yaw_response = 12, brake_deceleration = 16;
  std::vector<ThrusterDef> thrusters;
  std::vector<ChannelSpec> channels;
  std::vector<float> initial_state;
  std::shared_ptr<const void> extension;
  ActuatorEvaluator evaluate = nullptr;
  uint32_t state_offset = 0;
};
struct ActionChannel {
  int body;
  float low, high;
  std::string name;
};
// Register before compiling a scene. Hot stepping calls a resolved function
// pointer; plugins may keep compiled immutable data in extension and mutable
// per-body data in initial_state. The engine clones/serializes that data too.
using ActuatorCompiler = std::function<ActuatorDef(const Json&)>;
void register_actuator(const std::string& name, ActuatorCompiler compiler,
                       ActuatorEvaluator evaluate);
ActuatorDef compile_actuator(const Json& json);
void append_channels(const ActuatorDef& def, int body,
                     std::vector<ActionChannel>& channels);
ActuatorForce actuator_force(const BodyDef& body, Vec2 velocity, float angle,
                             float omega, const float* actions, float dt,
                             float* state = nullptr);
}  // namespace fg::control
