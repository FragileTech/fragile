#include "control/actuators.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

#include "control/scene.hpp"

namespace fg::control {
namespace {
float parameter(const Json& j, float fallback, float lo, float hi) {
  const double v = j.num(fallback);
  if (!std::isfinite(v) || v < lo || v > hi)
    throw std::invalid_argument("Invalid actuator parameter");
  return float(v);
}
Vec2 vector(const Json& j, Vec2 fallback) {
  if (j.kind == Json::Null) return fallback;
  if (j.kind != Json::Array || j.array.size() != 2)
    throw std::invalid_argument("Actuator vectors require two numbers");
  return {parameter(j.array[0], 0, -100000, 100000),
          parameter(j.array[1], 0, -100000, 100000)};
}
}  // namespace
void builtin_channels(const ActuatorDef&, int, std::vector<ActionChannel>&);
ActuatorDef compile_builtin(const Json& j) {
  if (j.kind != Json::Null && j.kind != Json::Object)
    throw std::invalid_argument("actuator must be an object");
  ActuatorDef a;
  const auto kind = j["kind"].str("vector");
  if (kind == "vector")
    a.kind = ActuatorDef::Vector;
  else if (kind == "kart") {
    a.kind = ActuatorDef::Kart;
    a.dimensions = 3;
    a.wheelbase = parameter(j["wheelbase"], 1, .01f, 100);
    a.steering_limit = parameter(j["steering_limit"], .55f, .01f, 1.4f);
    a.lateral_grip = parameter(j["lateral_grip"], 12, 0, 100);
    a.yaw_response = parameter(j["yaw_response"], 12, 0, 100);
    a.brake_deceleration = parameter(j["brake_deceleration"], 16, 0, 1000);
  } else if (kind == "holonomic") {
    a.kind = ActuatorDef::Holonomic;
    a.dimensions = 3;
  } else if (kind == "thrusters") {
    a.kind = ActuatorDef::Thrusters;
    for (const auto& t : j["thrusters"].items()) {
      ThrusterDef d;
      d.point = vector(t["position"], {});
      d.direction = vector(t["direction"], {1, 0});
      if (length2(d.direction) < 1e-12f)
        throw std::invalid_argument("Thruster direction cannot be zero");
      d.direction = normalized(d.direction);
      d.force = parameter(t["force"], 1, 0, 100000);
      d.reversible = t["reversible"].flag();
      a.thrusters.push_back(d);
    }
    if (a.thrusters.empty() || a.thrusters.size() > 32)
      throw std::invalid_argument("Use 1–32 thrusters per body");
    a.dimensions = uint32_t(a.thrusters.size());
  } else
    throw std::invalid_argument("Unknown actuator kind: " + kind);
  std::vector<ActionChannel> channels;
  builtin_channels(a, 0, channels);
  for (const auto& c : channels) a.channels.push_back({c.name, c.low, c.high});
  return a;
}
void builtin_channels(const ActuatorDef& a, int body,
                      std::vector<ActionChannel>& out) {
  auto add = [&](const std::string& name, float lo) {
    out.push_back({body, lo, 1, name});
  };
  switch (a.kind) {
    case ActuatorDef::Vector:
      add("thrust", 0);
      add("torque", -1);
      break;
    case ActuatorDef::Kart:
      add("throttle", -1);
      add("steering", -1);
      add("brake", 0);
      break;
    case ActuatorDef::Holonomic:
      add("force_x", -1);
      add("force_y", -1);
      add("torque", -1);
      break;
    case ActuatorDef::Thrusters:
      for (size_t i = 0; i < a.thrusters.size(); ++i)
        add("thruster_" + std::to_string(i),
            a.thrusters[i].reversible ? -1 : 0);
      break;
  }
}
ActuatorForce builtin_force(const BodyDef& b, Vec2 v, float theta, float w,
                            const float* u, float h, float*) {
  const auto& a = b.actuator;
  ActuatorForce result;
  Vec2 forward{std::cos(theta), std::sin(theta)}, side = perp(forward);
  auto signed_input = [&](int i) { return std::clamp(u[i], -1.f, 1.f); };
  switch (a.kind) {
    case ActuatorDef::Vector:
      result.force = forward * (b.thrust * std::clamp(u[0], 0.f, 1.f));
      result.torque = b.torque * signed_input(1);
      break;
    case ActuatorDef::Holonomic:
      result.force =
          (forward * signed_input(0) + side * signed_input(1)) * b.thrust;
      result.torque = b.torque * signed_input(2);
      break;
    case ActuatorDef::Thrusters:
      for (size_t i = 0; i < a.thrusters.size(); ++i) {
        const auto& t = a.thrusters[i];
        float power =
            std::clamp(u[i], t.reversible ? -1.f : 0.f, 1.f) * t.force;
        result.force +=
            (forward * t.direction.x + side * t.direction.y) * power;
        result.torque += cross(t.point, t.direction) * power;
      }
      break;
    case ActuatorDef::Kart: {
      const float longitudinal = dot(v, forward), lateral = dot(v, side);
      const float brake = std::clamp(u[2], 0.f, 1.f) * a.brake_deceleration;
      const float braking = std::copysign(
          std::min(std::abs(longitudinal) / h, brake), longitudinal);
      result.force =
          forward * (b.thrust * signed_input(0) - b.mass * braking) -
          side * (b.mass * lateral * (1 - std::exp(-a.lateral_grip * h)) / h);
      const float target = longitudinal / a.wheelbase *
                           std::tan(signed_input(1) * a.steering_limit);
      result.torque =
          b.inertia * (target - w) * (1 - std::exp(-a.yaw_response * h)) / h;
    } break;
  }
  return result;
}

namespace {
struct Plugin {
  ActuatorCompiler compile;
  ActuatorEvaluator evaluate;
};
std::map<std::string, Plugin>& plugins() {
  static std::map<std::string, Plugin> registry{
      {"vector", {compile_builtin, builtin_force}},
      {"kart", {compile_builtin, builtin_force}},
      {"holonomic", {compile_builtin, builtin_force}},
      {"thrusters", {compile_builtin, builtin_force}}};
  return registry;
}
std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}
}  // namespace
void register_actuator(const std::string& name, ActuatorCompiler compiler,
                       ActuatorEvaluator evaluate) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  if (name.empty() || !compiler || !evaluate ||
      !plugins().emplace(name, Plugin{compiler, evaluate}).second)
    throw std::invalid_argument("Invalid or duplicate actuator plugin: " +
                                name);
}
ActuatorDef compile_actuator(const Json& j) {
  const auto name = j["kind"].str("vector");
  Plugin plugin;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto p = plugins().find(name);
    if (p == plugins().end())
      throw std::invalid_argument("Unknown actuator kind: " + name);
    plugin = p->second;
  }
  auto a = plugin.compile(j);
  a.evaluate = plugin.evaluate;
  a.dimensions = uint32_t(a.channels.size());
  if (!a.dimensions || a.dimensions > 64 || a.initial_state.size() > 256)
    throw std::invalid_argument("Invalid actuator plugin dimensions");
  for (const auto& c : a.channels)
    if (c.name.empty() || !std::isfinite(c.low) || !std::isfinite(c.high) ||
        c.low >= c.high)
      throw std::invalid_argument("Invalid actuator bounds");
  for (float v : a.initial_state)
    if (!std::isfinite(v))
      throw std::invalid_argument("Invalid actuator initial state");
  return a;
}
void append_channels(const ActuatorDef& a, int body,
                     std::vector<ActionChannel>& out) {
  for (const auto& c : a.channels) out.push_back({body, c.low, c.high, c.name});
}
ActuatorForce actuator_force(const BodyDef& b, Vec2 v, float theta, float w,
                             const float* u, float h, float* state) {
  return b.actuator.evaluate(b, v, theta, w, u, h, state);
}
}  // namespace fg::control
