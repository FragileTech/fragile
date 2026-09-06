#include "control/scene.hpp"

#include <cstring>
#include <limits>

#include "control/agent_types.hpp"

namespace fg::control {
namespace {
Vec2 vec(const Json& j, Vec2 fallback = {}) {
  if (j.kind == Json::Null) return fallback;
  if (j.kind != Json::Array || j.array.size() != 2)
    throw std::invalid_argument("Vectors require two numbers");
  Vec2 v{float(j.array[0].num()), float(j.array[1].num())};
  if (!std::isfinite(v.x) || !std::isfinite(v.y))
    throw std::invalid_argument("Vector outside float32 range");
  return v;
}
float number(const Json& j, float fallback, float lo, float hi) {
  double n = j.num(fallback);
  if (n < lo || n > hi)
    throw std::invalid_argument("Scene parameter outside supported range");
  return float(n);
}
int integer(const Json& j, int fallback, int lo, int hi) {
  double n = j.num(fallback);
  if (n != std::floor(n) || n < lo || n > hi)
    throw std::invalid_argument("Invalid integer scene parameter");
  return int(n);
}
float area(const std::vector<Vec2>& p) {
  float a = 0;
  for (size_t i = 0; i < p.size(); ++i) a += cross(p[i], p[(i + 1) % p.size()]);
  return a * .5f;
}
bool point_inside(const std::vector<Vec2>& ring, Vec2 p) {
  bool in = false;
  for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
    Vec2 a = ring[i], b = ring[j];
    if ((a.y > p.y) != (b.y > p.y) &&
        p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
      in = !in;
  }
  return in;
}
bool intersect(Vec2 a, Vec2 b, Vec2 c, Vec2 d) {
  float x = cross(b - a, c - a), y = cross(b - a, d - a),
        u = cross(d - c, a - c), v = cross(d - c, b - c);
  auto on = [](Vec2 p, Vec2 q, Vec2 r) {
    return length2(closest(r, p, q) - r) < 1e-10f;
  };
  return (x * y < 0 && u * v < 0) || (std::abs(x) < 1e-6f && on(a, b, c)) ||
         (std::abs(y) < 1e-6f && on(a, b, d)) ||
         (std::abs(u) < 1e-6f && on(c, d, a)) ||
         (std::abs(v) < 1e-6f && on(c, d, b));
}
std::vector<Vec2> ring(const Json& j) {
  std::vector<Vec2> p;
  for (const auto& v : j.items()) p.push_back(vec(v));
  if (p.size() > 3 && length2(p.front() - p.back()) < 1e-12f) p.pop_back();
  if (p.size() < 3 || p.size() > 10000 || std::abs(area(p)) < 1e-5f)
    throw std::invalid_argument(
        "Boundary needs at least three non-collinear vertices");
  for (size_t i = 0; i < p.size(); ++i) {
    if (length2(p[i] - p[(i + 1) % p.size()]) < 1e-10f)
      throw std::invalid_argument("Repeated boundary vertex");
    for (size_t k = i + 1; k < p.size(); ++k)
      if (k != (i + 1) % p.size() && i != (k + 1) % p.size() &&
          intersect(p[i], p[(i + 1) % p.size()], p[k], p[(k + 1) % p.size()]))
        throw std::invalid_argument("Self-intersecting boundary");
  }
  return p;
}
void hash_byte(uint64_t& h, uint8_t b) { h = (h ^ b) * 1099511628211ULL; }
void hash_json(uint64_t& h, const Json& j) {
  hash_byte(h, uint8_t(j.kind));
  if (j.kind == Json::Number || j.kind == Json::Boolean) {
    uint64_t bits;
    std::memcpy(&bits, &j.number, 8);
    for (int i = 0; i < 8; ++i) hash_byte(h, uint8_t(bits >> (i * 8)));
  }
  for (unsigned char c : j.string) hash_byte(h, c);
  for (const auto& v : j.array) hash_json(h, v);
  for (const auto& kv : j.object) {
    for (unsigned char c : kv.first) hash_byte(h, c);
    hash_byte(h, 0);
    hash_json(h, kv.second);
  }
  hash_byte(h, 255);
}
}  // namespace
std::shared_ptr<const Scene> Scene::compile(const std::string& source) {
  if (source.size() > 8 * 1024 * 1024)
    throw std::invalid_argument("Scene JSON exceeds 8 MiB");
  Json root = JsonReader(source).read();
  if (root.kind != Json::Object || root["version"].num(1) != 1)
    throw std::invalid_argument("Unsupported scene version");
  auto s = std::make_shared<Scene>();
  s->json = source;
  s->name = root["name"].str("Untitled experiment");
  s->task = root["task"].str("navigation");
  s->size = vec(root["size"], {64, 44});
  if (s->size.x < 4 || s->size.y < 4 || s->size.x > 4096 || s->size.y > 4096)
    throw std::invalid_argument("Scene size must be between 4 and 4096 metres");
  const auto& physics = root["physics"];
  s->dt = number(physics["dt"], 1.f / 60.f, .0001f, .1f);
  s->substeps = integer(physics["substeps"], 4, 1, 32);
  s->solver_iterations = integer(physics["solver_iterations"], 8, 1, 32);
  s->lethal_walls = physics["lethal_walls"].flag();
  s->lethal_bodies = physics["lethal_bodies"].flag();
  const auto& reward = root["rewards"];
  s->progress_reward = number(reward["progress"], 1, 0, 1000);
  s->collision_penalty = number(reward["collision"], 2, 0, 10000);
  s->pickup_reward = number(reward["pickup"], 10, 0, 10000);
  s->delivery_reward = number(reward["delivery"], 100, 0, 10000);
  s->gate_reward = number(reward["gate"], 30, 0, 10000);
  s->formation_reward = number(reward["formation"], .15f, 0, 1000);
  s->formation_distance = number(root["formation_distance"], 3, .1f, 1000);
  s->respawn_seconds = number(root["respawn_seconds"], 4, 0, 10000);
  if (root["boundary"].kind == Json::Null)
    s->boundaries.push_back({{0, 0}, {s->size.x, 0}, s->size, {0, s->size.y}});
  else
    s->boundaries.push_back(ring(root["boundary"]));
  for (const auto& j : root["holes"].items()) s->boundaries.push_back(ring(j));
  for (size_t ri = 0; ri < s->boundaries.size(); ++ri) {
    auto& r = s->boundaries[ri];
    if ((area(r) > 0) != (ri == 0)) std::reverse(r.begin(), r.end());
    for (Vec2 p : r) {
      if (p.x < 0 || p.y < 0 || p.x > s->size.x || p.y > s->size.y)
        throw std::invalid_argument("Boundary outside scene size");
      if (ri && !point_inside(s->boundaries[0], p))
        throw std::invalid_argument("Hole outside outer boundary");
    }
    for (size_t rj = 0; rj < ri; ++rj) {
      const auto& other = s->boundaries[rj];
      for (size_t i = 0; i < r.size(); ++i)
        for (size_t j = 0; j < other.size(); ++j)
          if (intersect(r[i], r[(i + 1) % r.size()], other[j],
                        other[(j + 1) % other.size()]))
            throw std::invalid_argument("Boundary rings intersect");
      if (rj && (point_inside(other, r[0]) || point_inside(r, other[0])))
        throw std::invalid_argument(
            "Nested holes are not valid free-space boundaries");
    }
    for (size_t i = 0; i < r.size(); ++i) {
      Vec2 a = r[i], b = r[(i + 1) % r.size()];
      s->edges.push_back({a,
                          b,
                          normalized(perp(b - a)),
                          {{std::min(a.x, b.x), std::min(a.y, b.y)},
                           {std::max(a.x, b.x), std::max(a.y, b.y)}}});
    }
  }
  uint32_t auxiliary_words = 0;
  const AgentTypes agent_types(root["agent_types"]);
  for (const auto& instance : root["bodies"].items()) {
    const Json j = agent_types.body(instance);
    BodyDef b;
    b.position = vec(j["position"]);
    b.velocity = vec(j["velocity"]);
    b.angle = number(j["angle"], 0, -10000, 10000);
    b.omega = number(j["omega"], 0, -1000, 1000);
    b.radius = number(j["radius"], .5f, .01f, 100);
    b.mass = number(j["mass"], 1, .001f, 100000);
    b.drag = number(j["drag"], .15f, 0, 100);
    b.angular_drag = number(j["angular_drag"], 2, 0, 100);
    b.thrust = number(j["thrust"], 12, 0, 100000);
    b.torque = number(j["torque"], 8, 0, 100000);
    b.restitution = number(j["restitution"], .25f, 0, 1);
    b.friction = number(j["friction"], .3f, 0, 2);
    b.controlled = j["controlled"].flag();
    b.cargo = j["cargo"].flag();
    b.respawn = j["respawn"].flag();
    if (!s->inside(b.position))
      throw std::invalid_argument("Body centre outside playable region");
    if (j["vertices"].kind != Json::Null) {
      b.vertices = ring(j["vertices"]);
      if (b.vertices.size() > 32)
        throw std::invalid_argument(
            "Dynamic hulls support at most 32 vertices");
      if (area(b.vertices) < 0)
        std::reverse(b.vertices.begin(), b.vertices.end());
      float moment = 0, twice_area = 0;
      b.radius = 0;
      for (size_t i = 0; i < b.vertices.size(); ++i) {
        Vec2 a = b.vertices[i], v = b.vertices[(i + 1) % b.vertices.size()],
             next = b.vertices[(i + 2) % b.vertices.size()];
        if (cross(v - a, next - v) < -1e-6f)
          throw std::invalid_argument("Dynamic hull must be convex");
        float c = cross(a, v);
        twice_area += c;
        moment += c * (dot(a, a) + dot(a, v) + dot(v, v));
        b.radius = std::max(b.radius, length(a));
      }
      b.inertia = b.mass * moment / (6 * twice_area);
    } else
      b.inertia = .5f * b.mass * b.radius * b.radius;
    b.inertia = number(j["inertia"], b.inertia, .000001f, 1e12f);
    if (b.controlled) {
      b.actuator = compile_actuator(j["actuator"]);
      b.actuator.offset = uint32_t(s->channels.size());
      b.actuator.state_offset = auxiliary_words;
      auxiliary_words += uint32_t(b.actuator.initial_state.size());
      append_channels(b.actuator, int(s->bodies.size()), s->channels);
      if (s->channels.size() > 8192)
        throw std::invalid_argument("Scene action dimension exceeds 8192");
      s->controlled.push_back(int(s->bodies.size()));
    }
    s->bodies.push_back(std::move(b));
  }
  if (s->bodies.empty() || s->bodies.size() > 4096)
    throw std::invalid_argument("Scene requires 1–4096 bodies");
  auto zones = [&](const Json& list, std::vector<Zone>& dest) {
    for (const auto& j : list.items()) {
      Zone z{vec(j["position"]), number(j["radius"], 1, .01f, 1000)};
      if (!s->inside(z.position))
        throw std::invalid_argument("Zone outside playable region");
      dest.push_back(z);
    }
  };
  zones(root["bases"], s->bases);
  zones(root["gates"], s->gates);
  zones(root["pickups"], s->pickups);
  zones(root["refineries"], s->refineries);
  if (root["cargo"].kind != Json::Null) {
    if (root["cargo"].kind != Json::Object)
      throw std::invalid_argument("cargo must be an object");
    s->cargo_capacity = float(integer(root["cargo"]["capacity"], 5, 1, 10000));
    s->unload_seconds = number(root["cargo"]["unload_seconds"], 2, .01f, 10000);
    s->full_reward = number(root["cargo"]["full_reward"], s->pickup_reward, 0, 10000);
    if (s->refineries.empty())
      throw std::invalid_argument("Cargo collection requires a refinery zone");
  }
  if (s->pickups.size() > 4096)
    throw std::invalid_argument("Too many pickup slots");
  for (const auto& j : root["gravity"].items())
    s->gravity.push_back({vec(j["position"]),
                          number(j["strength"], 10, -100000, 100000),
                          number(j["softening"], 2, .01f, 1000)});
  for (const auto& j : root["tethers"].items()) {
    TetherDef t;
    t.a = integer(j["a"], 0, 0, int(s->bodies.size()) - 1);
    t.b = integer(j["b"], -1, -1, int(s->bodies.size()) - 1);
    if (t.a == t.b)
      throw std::invalid_argument("A tether cannot join a body to itself");
    t.rest = number(
        j["rest_length"],
        t.b >= 0 ? length(s->bodies[t.a].position - s->bodies[t.b].position)
                 : 2,
        0, 10000);
    t.stiffness = number(j["stiffness"], 25, 0, 1000000);
    t.damping = number(j["damping"], 6, 0, 100000);
    t.break_force = number(j["break_force"], 500, 0, 1e12f);
    t.hook_range = number(j["hook_range"], 2, 0, 1000);
    t.automatic = j["automatic"].flag();
    s->tethers.push_back(t);
  }
  for (const auto& definition : root["extensions"].items()) {
    auto extension = compile_world_extension(definition);
    extension.state_offset = auxiliary_words;
    auxiliary_words += uint32_t(extension.initial_state.size());
    s->extension_observations += extension.observation_size;
    if (auxiliary_words > 65536 || s->extension_observations > 65536 ||
        s->extensions.size() >= 64)
      throw std::invalid_argument("Too many world extension fields");
    s->extensions.push_back(std::move(extension));
  }
  s->layout = Layout(s->bodies.size(), s->controlled.size(), s->tethers.size(),
                     s->pickups.size(), auxiliary_words);
  if (s->cargo_capacity > 0) {
    s->layout.cargo = s->layout.words;
    s->layout.cargo_capacity = s->cargo_capacity;
    // Per controlled vehicle: load, return phase, delivered units, full cycles.
    s->layout.words += 4 * uint32_t(s->controlled.size());
    s->layout.stride = (s->layout.words + 15) & ~15u;
  }
  if (s->layout.words > 100000)
    throw std::invalid_argument("Scene state exceeds 100000 float32 words");
  s->fingerprint = 14695981039346656037ULL;
  hash_json(s->fingerprint, root);
  s->cell_size = std::max(2.f, std::max(s->size.x, s->size.y) / 128.f);
  s->grid_w = int(std::ceil(s->size.x / s->cell_size)) + 1;
  s->grid_h = int(std::ceil(s->size.y / s->cell_size)) + 1;
  s->edge_cells.resize(size_t(s->grid_w) * s->grid_h);
  for (size_t i = 0; i < s->edges.size(); ++i) {
    const Aabb& a = s->edges[i].bounds;
    int x0 = int(a.lo.x / s->cell_size), x1 = int(a.hi.x / s->cell_size),
        y0 = int(a.lo.y / s->cell_size), y1 = int(a.hi.y / s->cell_size);
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x)
        s->edge_cells[size_t(y) * s->grid_w + x].push_back(int(i));
  }
  return s;
}
bool Scene::inside(Vec2 p) const {
  if (!point_inside(boundaries[0], p)) return false;
  for (size_t i = 1; i < boundaries.size(); ++i)
    if (point_inside(boundaries[i], p)) return false;
  return true;
}
float Scene::raycast(Vec2 o, Vec2 direction, float distance) const {
  Vec2 d = normalized(direction);
  for (const auto& edge : edges) {
    Vec2 e = edge.b - edge.a;
    float det = cross(d, e);
    if (std::abs(det) < 1e-8f) continue;
    float t = cross(edge.a - o, e) / det, u = cross(edge.a - o, d) / det;
    if (t >= 0 && u >= 0 && u <= 1) distance = std::min(distance, t);
  }
  return distance;
}
}  // namespace fg::control
