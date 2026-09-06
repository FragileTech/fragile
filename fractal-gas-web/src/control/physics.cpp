#include "control/physics.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace fg::control {
namespace {
bool respawn_cargo(const Scene& s, float* r, size_t b) {
  const auto& l = s.layout;
  const auto& body = s.bodies[b];
  uint64_t seed = rng_state(r);
  // Rejection sampling covers the entire map. Bound work per frame; an
  // overcrowded scene keeps the delivered slot inactive and retries next frame.
  for (int attempt = 0; attempt < 256; ++attempt) {
    Vec2 p{random01(seed) * s.size.x, random01(seed) * s.size.y};
    if (!s.inside(p)) continue;
    bool clear = true;
    for (const auto& edge : s.edges)
      if (length2(p - closest(p, edge.a, edge.b)) <= body.radius * body.radius) {
        clear = false;
        break;
      }
    if (!clear) continue;
    for (const auto& base : s.bases)
      if (length2(p - base.position) <= std::pow(body.radius + base.radius, 2)) {
        clear = false;
        break;
      }
    if (!clear) continue;
    for (size_t other = 0; other < s.bodies.size(); ++other)
      if (other != b && (word(r, l.flags + other) & active_flag) &&
          length2(p - position(r, l, other)) <=
              std::pow(body.radius + s.bodies[other].radius, 2)) {
        clear = false;
        break;
      }
    if (!clear) continue;
    rng_state(r, seed);
    position(r, l, b, p);
    velocity(r, l, b, {});
    angle(r, l, b) = body.angle;
    omega(r, l, b) = 0;
    word(r, l.flags + b, active_flag);
    return true;
  }
  rng_state(r, seed);
  return false;
}

struct Shape {
  Vec2 center;
  float radius = 0;
  int n = 0;
  std::array<Vec2, 32> vertices;
};
Shape shape(const BodyDef& body, Vec2 p, float a) {
  Shape s;
  s.center = p;
  s.radius = body.radius;
  s.n = int(body.vertices.size());
  float c = std::cos(a), sn = std::sin(a);
  for (int i = 0; i < s.n; ++i) {
    Vec2 v = body.vertices[i];
    s.vertices[i] = p + Vec2{v.x * c - v.y * sn, v.x * sn + v.y * c};
  }
  return s;
}
Shape shape(const Edge& e) {
  Shape s;
  s.center = (e.a + e.b) * .5f;
  s.n = 2;
  s.vertices[0] = e.a;
  s.vertices[1] = e.b;
  return s;
}
std::pair<float, float> project(const Shape& s, Vec2 n) {
  if (!s.n) {
    float v = dot(s.center, n);
    return {v - s.radius, v + s.radius};
  }
  float lo = dot(s.vertices[0], n), hi = lo;
  for (int i = 1; i < s.n; ++i) {
    float v = dot(s.vertices[i], n);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  return {lo, hi};
}
// The largest separating-axis gap is a conservative lower bound on distance.
// Together with the translational + rotational speed bound it supports CCD
// without hidden caches or allocation. Negative gap means overlap.
float separation(const Shape& a, const Shape& b, Vec2& normal) {
  float gap = -std::numeric_limits<float>::infinity();
  auto axis = [&](Vec2 n) {
    if (length2(n) < 1e-16f) return;
    n = normalized(n);
    auto pa = project(a, n), pb = project(b, n);
    float g = pb.first - pa.second;
    if (g > gap) {
      gap = g;
      normal = n;
    }
    g = pa.first - pb.second;
    if (g > gap) {
      gap = g;
      normal = -n;
    }
  };
  if (!a.n && !b.n) axis(b.center - a.center);
  for (int i = 0; i < a.n; ++i)
    axis(perp(a.vertices[(i + 1) % a.n] - a.vertices[i]));
  for (int i = 0; i < b.n; ++i)
    axis(perp(b.vertices[(i + 1) % b.n] - b.vertices[i]));
  // A zero-area segment also needs its tangent, especially at end points.
  if (a.n == 2) axis(a.vertices[1] - a.vertices[0]);
  if (b.n == 2) axis(b.vertices[1] - b.vertices[0]);
  if (!a.n && b.n) {
    int k = 0;
    for (int i = 1; i < b.n; ++i)
      if (length2(b.vertices[i] - a.center) < length2(b.vertices[k] - a.center))
        k = i;
    axis(b.vertices[k] - a.center);
  }
  if (!b.n && a.n) {
    int k = 0;
    for (int i = 1; i < a.n; ++i)
      if (length2(a.vertices[i] - b.center) < length2(a.vertices[k] - b.center))
        k = i;
    axis(b.center - a.vertices[k]);
  }
  if (!std::isfinite(gap)) {
    normal = {1, 0};
    gap = -(a.radius + b.radius);
  }
  return gap;
}
Vec2 contact_point(const Shape& a, const Shape& b, Vec2 n) {
  if (!a.n) return a.center + n * a.radius;
  if (!b.n) return b.center - n * b.radius;
  auto pa = project(a, n), pb = project(b, n);
  Vec2 t = perp(n);
  // Clip the contact's tangential interval to the intersecting support faces.
  float amin = 1e30f, amax = -1e30f, bmin = 1e30f, bmax = -1e30f;
  for (int i = 0; i < a.n; ++i)
    if (dot(a.vertices[i], n) >= pa.second - 1e-4f) {
      float v = dot(a.vertices[i], t);
      amin = std::min(amin, v);
      amax = std::max(amax, v);
    }
  for (int i = 0; i < b.n; ++i)
    if (dot(b.vertices[i], n) <= pb.first + 1e-4f) {
      float v = dot(b.vertices[i], t);
      bmin = std::min(bmin, v);
      bmax = std::max(bmax, v);
    }
  float low = std::max(amin, bmin), high = std::min(amax, bmax);
  if (low > high) {
    auto ta = project(a, t), tb = project(b, t);
    low = std::max(ta.first, tb.first);
    high = std::min(ta.second, tb.second);
  }
  return n * ((pa.second + pb.first) * .5f) + t * ((low + high) * .5f);
}
void impulse(float* r, const Scene& s, Contact& c, bool restitution) {
  const auto& l = s.layout;
  const auto& a = s.bodies[c.a];
  Vec2 ra = c.point - position(r, l, c.a),
       va = velocity(r, l, c.a) + perp(ra) * omega(r, l, c.a);
  Vec2 rb{}, vb{};
  float inv_b = 0, inv_ib = 0, e = a.restitution, mu = a.friction;
  if (c.b >= 0) {
    const auto& b = s.bodies[c.b];
    rb = c.point - position(r, l, c.b);
    vb = velocity(r, l, c.b) + perp(rb) * omega(r, l, c.b);
    inv_b = 1 / b.mass;
    inv_ib = 1 / b.inertia;
    e = std::min(e, b.restitution);
    mu = std::sqrt(mu * b.friction);
  }
  float inv_a = 1 / a.mass, inv_ia = 1 / a.inertia;
  float an = cross(ra, c.normal), bn = cross(rb, c.normal),
        den = inv_a + inv_b + an * an * inv_ia + bn * bn * inv_ib;
  float vn = dot(vb - va, c.normal);
  if (restitution) c.target_velocity = vn < -.5f ? -e * vn : 0;
  float j = (c.target_velocity - vn) / den;
  float previous = c.impulse;
  c.impulse = std::max(0.f, previous + j);
  j = c.impulse - previous;
  Vec2 force = c.normal * j;
  Vec2 tangent = perp(c.normal);
  float at = cross(ra, tangent), bt = cross(rb, tangent);
  float jt = -dot(vb - va, tangent) /
             (inv_a + inv_b + at * at * inv_ia + bt * bt * inv_ib);
  jt = std::clamp(jt, -mu * std::abs(j), mu * std::abs(j));
  force += tangent * jt;
  velocity(r, l, c.a, velocity(r, l, c.a) - force * inv_a);
  omega(r, l, c.a) -= cross(ra, force) * inv_ia;
  if (c.b >= 0) {
    velocity(r, l, c.b, velocity(r, l, c.b) + force * inv_b);
    omega(r, l, c.b) += cross(rb, force) * inv_ib;
  }
}
Aabb swept(Vec2 a, Vec2 b, float radius) {
  return {{std::min(a.x, b.x) - radius, std::min(a.y, b.y) - radius},
          {std::max(a.x, b.x) + radius, std::max(a.y, b.y) + radius}};
}
}  // namespace
Scratch::Scratch(const Scene& s) {
  size_t n = s.bodies.size();
  contacts.reserve(std::min(n * (n - 1) / 2, n * 32) + n * 32 + 16);
  order.resize(n);
  bounds.resize(n);
  old_positions.resize(n);
  old_angles.resize(n);
  bounded_actions.resize(s.channels.size());
  edge_marks.resize(s.edges.size());
  edge_ids.reserve(s.edges.size());
}
Physics::Physics(std::shared_ptr<const Scene> s, int threads)
    : scene(std::move(s)), pool(threads) {
  scratch_.reserve(pool.size());
  for (int i = 0; i < pool.size(); ++i) scratch_.emplace_back(*scene);
}
void Physics::edge_candidates(Scratch& q, Aabb box) {
  const auto& s = *scene;
  q.edge_ids.clear();
  if (++q.stamp == 0) {
    std::fill(q.edge_marks.begin(), q.edge_marks.end(), 0);
    ++q.stamp;
  }
  auto cell = [](float v, float size, int limit) {
    if (!std::isfinite(v)) throw std::runtime_error("Non-finite swept bounds");
    return int(std::clamp(std::floor(v / size), 0.f, float(limit - 1)));
  };
  int x0 = cell(box.lo.x, s.cell_size, s.grid_w),
      x1 = cell(box.hi.x, s.cell_size, s.grid_w);
  int y0 = cell(box.lo.y, s.cell_size, s.grid_h),
      y1 = cell(box.hi.y, s.cell_size, s.grid_h);
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
      for (int i : s.edge_cells[size_t(y) * s.grid_w + x])
        if (q.edge_marks[i] != q.stamp) {
          q.edge_marks[i] = q.stamp;
          if (s.edges[i].bounds.overlaps(box)) q.edge_ids.push_back(i);
        }
  std::sort(q.edge_ids.begin(), q.edge_ids.end());
}
void Physics::step(const StateBatch& input, const int32_t* sources,
                   const float* actions, const int32_t* frames,
                   StateBatch& output, StepResult* results) {
  if (input.fingerprint != scene->fingerprint ||
      output.fingerprint != scene->fingerprint ||
      input.storage == output.storage)
    throw std::invalid_argument(
        "Step requires compatible, non-overlapping input/output batches");
  const size_t dims = scene->channels.size();
  for (size_t i = 0; i < output.count; ++i) {
    int ix = sources ? sources[i] : int(i);
    if (ix < 0 || uint32_t(ix) >= input.count)
      throw std::out_of_range("Source row");
    if (frames[i] < 0 || frames[i] > 4096)
      throw std::invalid_argument("Frame repeat must be 0–4096");
  }
  for (size_t i = 0; i < output.count * dims; ++i)
    if (!std::isfinite(actions[i]))
      throw std::invalid_argument("Non-finite action");
  std::atomic<bool> failed{false};
  pool.parallel_for(int32_t(output.count), [&](int32_t i, int slot) {
    try {
      std::memcpy(output.row(i), input.row(sources ? sources[i] : i),
                  input.layout.words * 4);
      step_world(output.row(i), actions + i * dims, frames[i], results[i],
                 slot);
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
    }
  });
  if (failed.load())
    throw std::runtime_error(
        "Physics scratch capacity or numerical limit exceeded; input batch is "
        "unchanged");
}
void Physics::substep(float* r, const float* actions, float h, Scratch& q,
                      StepResult& result) {
  const auto& s = *scene;
  const auto& l = s.layout;
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    q.old_positions[b] = position(r, l, b);
    q.old_angles[b] = angle(r, l, b);
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const auto& def = s.bodies[b];
    Vec2 p = position(r, l, b), v = velocity(r, l, b);
    for (const auto& g : s.gravity) {
      Vec2 d = g.position - p;
      float d2 = length2(d) + g.softening * g.softening;
      v += d * (h * g.strength / (d2 * std::sqrt(d2)));
    }
    velocity(r, l, b, v * std::exp(-def.drag * h));
    omega(r, l, b) *= std::exp(-def.angular_drag * h);
  }
  for (size_t c = 0; c < s.controlled.size(); ++c) {
    int b = s.controlled[c];
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const auto& def = s.bodies[b];
    const auto force =
        actuator_force(def, velocity(r, l, b), angle(r, l, b), omega(r, l, b),
                       actions + def.actuator.offset, h,
                       r + l.auxiliary + def.actuator.state_offset);
    velocity(r, l, b, velocity(r, l, b) + force.force * (h / def.mass));
    omega(r, l, b) += h * force.torque / def.inertia;
  }
  // Implicit spring/damper impulse. No cross-step warm-start cache is needed.
  for (size_t t = 0; t < s.tethers.size(); ++t) {
    const auto& def = s.tethers[t];
    int b = int(word(r, l.joints + 2 * t)) - 1;
    if (b < 0 || !(word(r, l.flags + def.a) & active_flag) ||
        !(word(r, l.flags + b) & active_flag))
      continue;
    Vec2 d = position(r, l, b) - position(r, l, def.a);
    float dist = length(d);
    Vec2 n = normalized(d);
    float inv = 1 / s.bodies[def.a].mass + 1 / s.bodies[b].mass;
    float gamma = h * (def.damping + h * def.stiffness);
    if (gamma <= 0) continue;
    gamma = 1 / gamma;
    float bias = (dist - r[l.joints + 2 * t + 1]) * h * def.stiffness * gamma;
    float j = -(dot(velocity(r, l, b) - velocity(r, l, def.a), n) + bias) /
              (inv + gamma);
    if (std::abs(j) > def.break_force * h) {
      word(r, l.joints + 2 * t, 0);
      continue;
    }
    velocity(r, l, def.a,
             velocity(r, l, def.a) - n * (j / s.bodies[def.a].mass));
    velocity(r, l, b, velocity(r, l, b) + n * (j / s.bodies[b].mass));
  }
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const auto& def = s.bodies[b];
    float remaining = h;
    for (int bounce = 0; bounce < 4 && remaining > 1e-8f; ++bounce) {
      Vec2 start = position(r, l, b), delta = velocity(r, l, b) * remaining;
      float a0 = angle(r, l, b), da = omega(r, l, b) * remaining;
      edge_candidates(q, swept(start, start + delta, def.radius + .002f));
      float best = 1;
      int hit = -1;
      Vec2 hit_n{};
      bool limited = false;
      float bound = length(delta) + std::abs(da) * def.radius;
      if (bound > 1e-9f)
        for (int edge_id : q.edge_ids) {
          const auto& edge = s.edges[edge_id];
          Shape wall = shape(edge);
          float time = 0;
          Vec2 normal{};
          bool found = false;
          for (int iteration = 0; iteration < 64; ++iteration) {
            Shape moving = shape(def, start + delta * time, a0 + da * time);
            float gap = separation(moving, wall, normal);
            if (gap < .0001f) {
              if (time == 0 && dot(delta, normal) <= 0 && std::abs(da) < 1e-5f)
                break;
              found = true;
              break;
            }
            time += .9f * gap / bound;
            if (time >= best) break;
            if (iteration == 63) {
              found = true;
              limited = true;
            }
          }
          if (found && time < best) {
            best = time;
            hit = edge_id;
            hit_n = normal;
          }
        }
      position(r, l, b, start + delta * best);
      angle(r, l, b) = a0 + da * best;
      if (hit < 0) break;
      Shape moving = shape(def, position(r, l, b), angle(r, l, b)),
            wall = shape(s.edges[hit]);
      Contact contact{int(b), -1, hit_n, contact_point(moving, wall, hit_n),
                      0,      0};
      impulse(r, s, contact, true);
      position(r, l, b, position(r, l, b) - hit_n * .0002f);
      ++result.collisions;
      result.reward -= s.collision_penalty;
      if (s.lethal_walls && def.controlled) word(r, 7, 1);
      if (limited) ++result.ccd_limits;
      remaining *= 1 - best;
      if (bounce == 3) {
        ++result.ccd_limits;
        remaining = 0;
      }
    }
    angle(r, l, b) = std::remainder(angle(r, l, b), 2 * pi);
  }
  q.contacts.clear();
  auto append = [&](Contact c) {
    if (q.contacts.size() == q.contacts.capacity())
      throw std::runtime_error("Contact capacity");
    q.contacts.push_back(c);
  };
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    q.order[b] = int(b);
    q.bounds[b] =
        swept(q.old_positions[b], position(r, l, b), s.bodies[b].radius);
    if (!(word(r, l.flags + b) & active_flag)) continue;
    Shape moving = shape(s.bodies[b], position(r, l, b), angle(r, l, b));
    edge_candidates(q,
                    swept(moving.center, moving.center, moving.radius + .002f));
    for (int e : q.edge_ids) {
      Shape wall = shape(s.edges[e]);
      Vec2 n;
      float gap = separation(moving, wall, n);
      if (gap < 0)
        append({int(b), -1, n, contact_point(moving, wall, n), -gap, 0});
    }
  }
  std::sort(q.order.begin(), q.order.end(), [&](int a, int b) {
    return q.bounds[a].lo.x == q.bounds[b].lo.x
               ? a < b
               : q.bounds[a].lo.x < q.bounds[b].lo.x;
  });
  for (size_t ia = 0; ia < q.order.size(); ++ia) {
    int a = q.order[ia];
    if (!(word(r, l.flags + a) & active_flag)) continue;
    for (size_t ib = ia + 1; ib < q.order.size(); ++ib) {
      int b = q.order[ib];
      if (q.bounds[b].lo.x > q.bounds[a].hi.x) break;
      if (!(word(r, l.flags + b) & active_flag) ||
          !q.bounds[a].overlaps(q.bounds[b]))
        continue;
      Shape sa = shape(s.bodies[a], position(r, l, a), angle(r, l, a)),
            sb = shape(s.bodies[b], position(r, l, b), angle(r, l, b));
      Vec2 n;
      float gap = separation(sa, sb, n);
      if (gap > 0) {
        // Conservative advancement catches bodies that crossed between samples.
        Vec2 da = position(r, l, a) - q.old_positions[a],
             db = position(r, l, b) - q.old_positions[b];
        float aa = std::remainder(angle(r, l, a) - q.old_angles[a], 2 * pi),
              ab = std::remainder(angle(r, l, b) - q.old_angles[b], 2 * pi);
        float speed = length(db - da) + std::abs(aa) * s.bodies[a].radius +
                      std::abs(ab) * s.bodies[b].radius;
        float time = 0;
        bool hit = false;
        if (speed > 1e-8f)
          for (int k = 0; k < 64; ++k) {
            sa = shape(s.bodies[a], q.old_positions[a] + da * time,
                       q.old_angles[a] + aa * time);
            sb = shape(s.bodies[b], q.old_positions[b] + db * time,
                       q.old_angles[b] + ab * time);
            gap = separation(sa, sb, n);
            if (gap <= .0001f) {
              hit = time > 0 || dot(db - da, n) < 0;
              break;
            }
            time += .9f * gap / speed;
            if (time >= 1) break;
          }
        if (!hit) continue;
        position(r, l, a, sa.center);
        position(r, l, b, sb.center);
        angle(r, l, a) = q.old_angles[a] + aa * time;
        angle(r, l, b) = q.old_angles[b] + ab * time;
        gap = 0;
      }
      append({a, b, n, contact_point(sa, sb, n), std::max(0.f, -gap), 0});
    }
  }
  // Canonical contact order is independent of worker assignment and broad-phase
  // order.
  std::sort(q.contacts.begin(), q.contacts.end(),
            [](const Contact& a, const Contact& b) {
              if (a.a != b.a) return a.a < b.a;
              if (a.b != b.b) return a.b < b.b;
              if (a.point.x != b.point.x) return a.point.x < b.point.x;
              return a.point.y < b.point.y;
            });
  for (auto& c : q.contacts) {
    ++result.collisions;
    if (s.bodies[c.a].controlled || (c.b >= 0 && s.bodies[c.b].controlled)) {
      result.reward -= s.collision_penalty;
      if ((c.b < 0 && s.lethal_walls) || (c.b >= 0 && s.lethal_bodies))
        word(r, 7, 1);
    }
  }
  for (int iteration = 0; iteration < s.solver_iterations; ++iteration)
    for (auto& c : q.contacts) impulse(r, s, c, iteration == 0);
  for (auto& c : q.contacts) {
    float ia = 1 / s.bodies[c.a].mass,
          ib = c.b < 0 ? 0 : 1 / s.bodies[c.b].mass;
    float correction = .8f * std::max(0.f, c.depth - .0005f) / (ia + ib);
    position(r, l, c.a, position(r, l, c.a) - c.normal * (correction * ia));
    if (c.b >= 0)
      position(r, l, c.b, position(r, l, c.b) + c.normal * (correction * ib));
  }
}
float Physics::potential(const float* r) const {
  const auto& s = *scene;
  const auto& l = s.layout;
  float total = 0;
  for (size_t c = 0; c < s.controlled.size(); ++c) {
    int b = s.controlled[c];
    Vec2 p = position(r, l, b);
    float best = 0;
    if (!s.gates.empty())
      best =
          length(p - s.gates[word(r, l.gates + c) % s.gates.size()].position);
    else if (s.cargo_capacity > 0 && r[l.cargo + 4 * c + 1] > 0) {
      best = 1e6f;
      for (const auto& zone : s.refineries)
        best = std::min(best, std::max(0.f, length(p - zone.position) - zone.radius));
    } else if (!s.pickups.empty()) {
      best = 1e6f;
      for (size_t i = 0; i < s.pickups.size(); ++i)
        if (r[l.food + 3 * i + 2] <= 0)
          best = std::min(
              best, length(p - Vec2{r[l.food + 3 * i], r[l.food + 3 * i + 1]}));
      if (best == 1e6f) best = 0;
    } else if (!s.bases.empty()) {
      int attached = -1;
      for (size_t i = 0; i < s.tethers.size(); ++i)
        if (s.tethers[i].a == b && word(r, l.joints + 2 * i))
          attached = int(word(r, l.joints + 2 * i)) - 1;
      if (attached >= 0) {
        best = 1e6f;
        for (const auto& base : s.bases)
          best =
              std::min(best, length(position(r, l, attached) - base.position));
      } else {
        best = 1e6f;
        for (size_t i = 0; i < s.bodies.size(); ++i)
          if (s.bodies[i].cargo && (word(r, l.flags + i) & active_flag))
            best = std::min(best, length(p - position(r, l, i)));
        if (best == 1e6f) best = 0;
      }
    }
    total -= best;
  }
  if (s.controlled.size() > 1 && s.task == "tandem") {
    Vec2 center{};
    for (int b : s.controlled) center += position(r, l, b);
    center = center / float(s.controlled.size());
    for (int b : s.controlled)
      total -=
          s.formation_reward * std::abs(length(position(r, l, b) - center) -
                                        s.formation_distance * .5f);
  }
  return total / float(std::max(size_t(1), s.controlled.size()));
}
void Physics::mechanics(float* r, StepResult& result) {
  const auto& s = *scene;
  const auto& l = s.layout;
  // Discharge only loads that were already full at the start of this frame.
  // The return phase stays latched through interruptions until completely empty.
  if (s.cargo_capacity > 0)
    for (size_t c = 0; c < s.controlled.size(); ++c) {
      float* cargo = r + l.cargo + 4 * c;
      if (cargo[1] == 0) continue;
      for (const auto& zone : s.refineries)
        if (length2(position(r, l, s.controlled[c]) - zone.position) <=
            zone.radius * zone.radius) {
          float amount = std::min(cargo[0], s.cargo_capacity * s.dt / s.unload_seconds);
          if (cargo[0] - amount < s.cargo_capacity * 1e-6f) amount = cargo[0];
          cargo[0] -= amount;
          cargo[2] += amount;
          result.reward += s.delivery_reward * (amount / s.cargo_capacity);
          if (cargo[0] == 0) {
            cargo[1] = 0;
            word(r, 4, word(r, 4) + 1);
          }
          break;
        }
    }
  for (size_t i = 0; i < s.pickups.size(); ++i) {
    float* f = r + l.food + 3 * i;
    if (f[2] > 0) {
      f[2] = std::max(0.f, f[2] - s.dt);
      if (f[2] == 0) {
        uint64_t seed = rng_state(r);
        Vec2 candidate = s.pickups[i].position;
        for (int k = 0; k < 64; ++k) {
          Vec2 p{1 + random01(seed) * (s.size.x - 2),
                 1 + random01(seed) * (s.size.y - 2)};
          if (s.inside(p)) {
            candidate = p;
            break;
          }
        }
        rng_state(r, seed);
        f[0] = candidate.x;
        f[1] = candidate.y;
      }
      continue;
    }
    for (size_t c = 0; c < s.controlled.size(); ++c) {
      int b = s.controlled[c];
      float* cargo = s.cargo_capacity > 0 ? r + l.cargo + 4 * c : nullptr;
      if (cargo && (cargo[1] > 0 || cargo[0] >= s.cargo_capacity)) continue;
      if (length2(position(r, l, b) - Vec2{f[0], f[1]}) <
          std::pow(s.bodies[b].radius + s.pickups[i].radius, 2)) {
        f[2] = std::max(s.dt, s.respawn_seconds);
        word(r, 5, word(r, 5) + 1);
        result.reward += s.pickup_reward;
        if (cargo) {
          cargo[0] += 1;
          if (cargo[0] >= s.cargo_capacity) {
            cargo[1] = 1;
            cargo[3] += 1;
            result.reward += s.full_reward;
          }
        }
        break;
      }
    }
  }
  for (size_t c = 0; c < s.controlled.size(); ++c)
    if (!s.gates.empty()) {
      int b = s.controlled[c];
      uint32_t gate = word(r, l.gates + c);
      const auto& target = s.gates[gate % s.gates.size()];
      if (length2(position(r, l, b) - target.position) <
          target.radius * target.radius) {
        word(r, l.gates + c, gate + 1);
        word(r, 6, word(r, 6) + 1);
        result.reward += s.gate_reward;
      }
    }
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    if (!s.bodies[b].cargo) continue;
    if (s.bodies[b].respawn && word(r, l.flags + b) == delivered_flag)
      respawn_cargo(s, r, b);
    if (word(r, l.flags + b) & active_flag)
      for (const auto& base : s.bases)
        if (length2(position(r, l, b) - base.position) <
            base.radius * base.radius) {
          word(r, l.flags + b, delivered_flag);
          word(r, 4, word(r, 4) + 1);
          result.reward += s.delivery_reward;
          for (size_t t = 0; t < s.tethers.size(); ++t)
            if (word(r, l.joints + 2 * t) == b + 1)
              word(r, l.joints + 2 * t, 0);
          if (s.bodies[b].respawn) respawn_cargo(s, r, b);
          break;
        }
  }
  for (size_t t = 0; t < s.tethers.size(); ++t) {
    const auto& def = s.tethers[t];
    if (!def.automatic || word(r, l.joints + 2 * t)) continue;
    int best = -1;
    float dist = def.hook_range;
    for (size_t b = 0; b < s.bodies.size(); ++b)
      if (s.bodies[b].cargo && (word(r, l.flags + b) & active_flag)) {
        float d = length(position(r, l, def.a) - position(r, l, b));
        if (d < dist) {
          best = int(b);
          dist = d;
        }
      }
    if (best >= 0) {
      word(r, l.joints + 2 * t, uint32_t(best + 1));
      r[l.joints + 2 * t + 1] = std::max(.1f, dist);
    }
  }
}
void Physics::step_world(float* r, const float* actions, int frames,
                         StepResult& result, int slot) {
  result = {};
  const auto& s = *scene;
  auto& bounded = scratch_[slot].bounded_actions;
  for (size_t i = 0; i < s.channels.size(); ++i)
    bounded[i] = std::clamp(actions[i], s.channels[i].low, s.channels[i].high);
  actions = bounded.data();
  for (int frame = 0; frame < frames && !word(r, 7); ++frame) {
    float before = potential(r);
    for (int k = 0; k < s.substeps; ++k)
      substep(r, actions, s.dt / s.substeps, scratch_[slot], result);
    result.reward += s.progress_reward * (potential(r) - before);
    mechanics(r, result);
    for (const auto& extension : s.extensions)
      if (extension.step) extension.step(s, extension, r, actions, result);
    word(r, 0, word(r, 0) + 1);
    ++result.frames;
    if (word(r, 7)) word(r, 3, word(r, 3) + 1);
  }
  result.dead = uint8_t(word(r, 7));
  for (size_t i = 8; i < s.layout.flags; ++i)
    if (!std::isfinite(r[i]))
      throw std::runtime_error("Non-finite physics state");
  for (size_t i = s.layout.auxiliary; i < s.layout.words; ++i)
    if (!std::isfinite(r[i]))
      throw std::runtime_error("Non-finite extension state");
}
size_t Physics::observation_dim() const {
  return scene->bodies.size() * 7 + scene->controlled.size() +
         scene->tethers.size() * 2 + scene->extension_observations +
         (scene->cargo_capacity > 0 ? scene->controlled.size() * 4 : 0);
}
void Physics::observe(const float* r, float* out) const {
  const auto& s = *scene;
  const auto& l = s.layout;
  size_t at = 0;
  float scale = std::max(s.size.x, s.size.y);
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    Vec2 p = position(r, l, b), v = velocity(r, l, b);
    float a = angle(r, l, b);
    out[at++] = p.x / scale;
    out[at++] = p.y / scale;
    out[at++] = v.x / 20;
    out[at++] = v.y / 20;
    out[at++] = std::cos(a);
    out[at++] = std::sin(a);
    out[at++] = omega(r, l, b) / 10;
  }
  for (size_t c = 0; c < s.controlled.size(); ++c)
    out[at++] = float(word(r, l.gates + c));
  for (size_t t = 0; t < s.tethers.size(); ++t) {
    out[at++] = float(word(r, l.joints + 2 * t));
    out[at++] = r[l.joints + 2 * t + 1] / scale;
  }
  if (s.cargo_capacity > 0)
    for (size_t c = 0; c < s.controlled.size(); ++c) {
      const float* cargo = r + l.cargo + 4 * c;
      out[at++] = cargo[0] / s.cargo_capacity;
      out[at++] = cargo[1];
      Vec2 delta{};
      float best = 1e30f;
      for (const auto& zone : s.refineries) {
        Vec2 candidate = zone.position - position(r, l, s.controlled[c]);
        if (length2(candidate) < best) { best = length2(candidate); delta = candidate; }
      }
      out[at++] = delta.x / scale;
      out[at++] = delta.y / scale;
    }
  for (const auto& extension : s.extensions)
    if (extension.observe) {
      extension.observe(s, extension, r, out + at);
      at += extension.observation_size;
    }
}
std::vector<float> Physics::inspect(const float* r,
                                    const float* actions) const {
  const auto& s = *scene;
  const auto& l = s.layout;
  std::vector<float> bounded(s.channels.size());
  for (size_t i = 0; i < bounded.size(); ++i) {
    if (!std::isfinite(actions[i]))
      throw std::invalid_argument("Non-finite inspection action");
    bounded[i] = std::clamp(actions[i], s.channels[i].low, s.channels[i].high);
  }
  actions = bounded.data();
  std::vector<float> out;
  // Rows: kind, body A, body B (-1 for world), x, y, vector x/y, magnitude.
  auto add = [&](int kind, int a, int b, Vec2 point, Vec2 vector,
                 float magnitude) {
    if (out.size() < 8192 * 8)
      out.insert(out.end(), {float(kind), float(a), float(b), point.x, point.y,
                             vector.x, vector.y, magnitude});
  };
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const auto& def = s.bodies[b];
    const Vec2 p = position(r, l, b), v = velocity(r, l, b);
    Vec2 force = v * (-def.drag * def.mass);
    for (const auto& g : s.gravity) {
      Vec2 d = g.position - p;
      float d2 = length2(d) + g.softening * g.softening;
      force += d * (def.mass * g.strength / (d2 * std::sqrt(d2)));
    }
    if (def.controlled) {
      std::vector<float> aux(def.actuator.initial_state.size());
      std::copy_n(r + l.auxiliary + def.actuator.state_offset, aux.size(),
                  aux.data());
      force += actuator_force(def, v, angle(r, l, b), omega(r, l, b),
                              actions + def.actuator.offset, s.dt / s.substeps,
                              aux.data())
                   .force;
    }
    add(0, b, -1, p, force, length(force));
    const auto body = shape(def, p, angle(r, l, b));
    for (const auto& edge : s.edges) {
      if (!swept(p, p, def.radius + .01f).overlaps(edge.bounds)) continue;
      auto wall = shape(edge);
      Vec2 normal;
      const float gap = separation(body, wall, normal);
      if (gap < .01f)
        add(1, b, -1, contact_point(body, wall, normal), normal,
            std::max(0.f, -gap));
    }
    for (size_t c = b + 1; c < s.bodies.size(); ++c) {
      if (!(word(r, l.flags + c) & active_flag) ||
          length(position(r, l, c) - p) >
              def.radius + s.bodies[c].radius + .01f)
        continue;
      auto other = shape(s.bodies[c], position(r, l, c), angle(r, l, c));
      Vec2 normal;
      float gap = separation(body, other, normal);
      if (gap < .01f)
        add(1, b, c, contact_point(body, other, normal), normal,
            std::max(0.f, -gap));
    }
  }
  for (size_t t = 0; t < s.tethers.size(); ++t) {
    const auto& d = s.tethers[t];
    int b = int(word(r, l.joints + 2 * t)) - 1;
    if (b < 0) continue;
    Vec2 pa = position(r, l, d.a), delta = position(r, l, b) - pa,
         n = normalized(delta);
    float force = d.stiffness * (length(delta) - r[l.joints + 2 * t + 1]) +
                  d.damping * dot(velocity(r, l, b) - velocity(r, l, d.a), n);
    add(2, d.a, b, pa, n * force, force);
  }
  return out;
}
}  // namespace fg::control
