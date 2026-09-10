#include "control/physics.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace fg::control {
namespace {
// Delivered retained cargo stays active for physics, but cannot be a target.
bool available_cargo(const Scene& s, const float* r, size_t b) {
  const auto flags = word(r, s.layout.flags + b);
  return s.bodies[b].cargo && (flags & active_flag) &&
         (!s.keep_delivered_rocks || !(flags & delivered_flag));
}
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

using geometry::Shape;
using geometry::shape;
using geometry::separation;
using geometry::contact_point;
using geometry::swept_near_edge;
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
  // Friction acts on the velocity after the normal constraint update.
  va -= force * inv_a + perp(ra) * (cross(ra, force) * inv_ia);
  vb += force * inv_b + perp(rb) * (cross(rb, force) * inv_ib);
  Vec2 tangent = perp(c.normal);
  float at = cross(ra, tangent), bt = cross(rb, tangent);
  float jt = -dot(vb - va, tangent) /
             (inv_a + inv_b + at * at * inv_ia + bt * bt * inv_ib);
  const float previous_tangent = c.tangent_impulse;
  c.tangent_impulse = std::clamp(previous_tangent + jt, -mu * c.impulse, mu * c.impulse);
  jt = c.tangent_impulse - previous_tangent;
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
  shapes.resize(n);
  empty_edge_bounds.resize(n);
  empty_edge_valid.resize(n);
  wall_contacts.reserve(n * 32 + 16);
  bounds.resize(n);
  old_positions.resize(n);
  frame_positions.resize(s.controlled.size());
  frame_tethers.resize(s.tethers.size());
  frame_rock_positions.resize(n);
  frame_hooked_rocks.resize(n);
  frame_wall_contacts.resize(n);
  old_angles.resize(n);
  bounded_actions.resize(s.channels.size());
  edge_marks.resize(s.edges.size());
  edge_ids.reserve(s.edges.size());
}
Physics::Physics(std::shared_ptr<const Scene> s, int threads)
    : scene(std::move(s)), pool(threads) {
  walls_.reserve(scene->edges.size());
  for (const auto& edge : scene->edges) walls_.push_back(shape(edge));
  scratch_.reserve(pool.size());
  for (int i = 0; i < pool.size(); ++i) scratch_.emplace_back(*scene);
}
void Physics::edge_candidates(Scratch& q, Aabb box) {
  const auto& s = *scene;
#ifdef FG_CONTROL_PROFILE
  ++q.work.edge_queries;
#endif
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
          if (s.edges[i].bounds.overlaps(box)) {
            q.edge_ids.push_back(i);
#ifdef FG_CONTROL_PROFILE
            ++q.work.edge_candidates;
#endif
          }
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
  pool.parallel_for_dynamic(int32_t(output.count), [&](int32_t i, int slot) {
    // Accumulate locally so neighboring dynamically scheduled worlds do not
    // repeatedly write the same cache line in the compact result array.
    StepResult result;
    try {
      std::memcpy(output.row(i), input.row(sources ? sources[i] : i),
                  input.layout.words * 4);
      step_world(output.row(i), actions + i * dims, frames[i], result, slot);
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
    }
    results[i] = result;
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
  q.wall_contacts.clear();
  auto separation = [&](const Shape& a, const Shape& b, Vec2& n) {
#ifdef FG_CONTROL_PROFILE
    ++q.work.separation_checks;
#endif
    return geometry::separation(a, b, n);
  };
  auto body_edges = [&](size_t b, Aabb box) {
    const auto& cached = q.empty_edge_bounds[b];
    if (q.empty_edge_valid[b] && cached.lo.x <= box.lo.x && cached.lo.y <= box.lo.y &&
        cached.hi.x >= box.hi.x && cached.hi.y >= box.hi.y) {
      q.edge_ids.clear();
      return;
    }
    edge_candidates(q, box);
    if (q.edge_ids.empty()) {
      Aabb expanded{box.lo - Vec2{s.cell_size, s.cell_size},
                    box.hi + Vec2{s.cell_size, s.cell_size}};
      edge_candidates(q, expanded);
      q.empty_edge_bounds[b] = q.edge_ids.empty() ? expanded : box;
      q.empty_edge_valid[b] = true;
      q.edge_ids.clear();  // The original query was empty in either case.
    }
  };
  auto body_shape = [&](size_t b, Vec2 p, float a) -> const Shape& {
    return q.shapes[b].get(s.bodies[b], p, a);
  };
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    q.old_positions[b] = position(r, l, b);
    q.old_angles[b] = angle(r, l, b);
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const auto& def = s.bodies[b];
    Vec2 p = position(r, l, b), v = velocity(r, l, b);
    if (s.flight_mode) v.y -= h * s.downward_gravity;
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
    const Vec2 ra = rotate(def.anchor_a, angle(r, l, def.a));
    const Vec2 rb = rotate(def.anchor_b, angle(r, l, b));
    Vec2 d = position(r, l, b) + rb - position(r, l, def.a) - ra;
    float dist = length(d);
    Vec2 n = normalized(d);
    float ca = cross(ra, n), cb = cross(rb, n);
    float inv = 1 / s.bodies[def.a].mass + 1 / s.bodies[b].mass +
                ca * ca / s.bodies[def.a].inertia + cb * cb / s.bodies[b].inertia;
    float gamma = h * (def.damping + h * def.stiffness);
    if (gamma <= 0) continue;
    gamma = 1 / gamma;
    float bias = (dist - r[l.joints + 2 * t + 1]) * h * def.stiffness * gamma;
    const Vec2 relative_velocity = velocity(r, l, b) + perp(rb) * omega(r, l, b) -
                                   velocity(r, l, def.a) - perp(ra) * omega(r, l, def.a);
    float j = -(dot(relative_velocity, n) + bias) / (inv + gamma);
    if (def.permanent) j = std::min(0.f, j);  // A cable pulls; it never pushes.
    if (!def.permanent && std::abs(j) > def.break_force * h) {
      word(r, l.joints + 2 * t, 0);
      continue;
    }
    velocity(r, l, def.a,
             velocity(r, l, def.a) - n * (j / s.bodies[def.a].mass));
    velocity(r, l, b, velocity(r, l, b) + n * (j / s.bodies[b].mass));
    omega(r, l, def.a) -= ca * j / s.bodies[def.a].inertia;
    omega(r, l, b) += cb * j / s.bodies[b].inertia;
  }
  auto wall_contact = [&](int b, int edge, Vec2 normal, Vec2 point) -> Contact& {
    for (auto& c : q.wall_contacts)
      if (c.a == b && c.edge == edge) {
        if (dot(c.normal, normal) < .999f) {
          c.impulse = c.tangent_impulse = 0;
        }
        c.normal = normal;
        c.point = point;
        return c;
      }
    if (q.wall_contacts.size() == q.wall_contacts.capacity())
      throw std::runtime_error("Wall contact capacity");
    Contact c{b, -1, normal, point};
    c.edge = edge;
    q.wall_contacts.push_back(c);
    return q.wall_contacts.back();
  };
  auto count_wall = [&](Contact& c) {
    if (s.bodies[c.a].controlled) {
      // A vehicle pays once per physics frame, including sustained contact.
      // Corners and additional substeps must not multiply its wall penalty.
      if (!q.frame_wall_contacts[c.a]) {
        result.reward -= s.wall_collision_penalty;
        q.frame_wall_contacts[c.a] = 1;
      }
      if (s.lethal_walls) word(r, 7, 1);
    }
    if (c.counted) return;
    c.counted = true;
    ++result.collisions;
  };
  // Resolve low-speed contacts before integrating gravity into another impact.
  // This manifold lasts one substep only; no state is hidden from snapshots.
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const Vec2 p = position(r, l, b);
    // Stationary vehicles still touch walls: reward and death settings apply
    // even when no movement sweep is needed. Passive bodies keep the fast path.
    if (!s.bodies[b].controlled && length2(velocity(r, l, b)) == 0 &&
        omega(r, l, b) == 0) continue;
    body_edges(b, swept(p, p, s.bodies[b].radius + .002f));
    if (q.edge_ids.empty()) continue;
    const Shape& moving = body_shape(b, p, angle(r, l, b));
    for (int e : q.edge_ids) {
      if (!swept_near_edge(moving.center, moving.center, s.edges[e], moving.radius + .002f)) continue;
      Vec2 n;
      const float gap = separation(moving, walls_[e], n);
      if (gap > .0001f) continue;
      const Vec2 point = contact_point(moving, walls_[e], n);
      const float closing = dot(velocity(r, l, b) +
          perp(point - moving.center) * omega(r, l, b), n);
      if (std::abs(closing) > .5f) continue;
      // A skin contact is actionable only if translation actually reaches the
      // wall this substep. Rotating near misses retain the full sweep.
      if (gap > 0 && (closing * h < gap ||
          (!s.bodies[b].vertices.empty() && omega(r, l, b) != 0))) continue;
      auto& c = wall_contact(int(b), e, n, point);
      c.depth = std::max(0.f, -gap);
      count_wall(c);
    }
  }
  for (int iteration = 0; iteration < s.solver_iterations; ++iteration)
    for (auto& c : q.wall_contacts) impulse(r, s, c, false);
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    if (!(word(r, l.flags + b) & active_flag)) continue;
    const auto& def = s.bodies[b];
    float remaining = h;
    for (int bounce = 0; bounce < 4 && remaining > 1e-8f; ++bounce) {
      Vec2 start = position(r, l, b), delta = velocity(r, l, b) * remaining;
      float a0 = angle(r, l, b), da = omega(r, l, b) * remaining;
      body_edges(b, swept(start, start + delta, def.radius + .002f));
      float best = 1;
      int hit = -1;
      Vec2 hit_n{};
      bool limited = false;
      float rotational_bound = def.vertices.empty() ? 0.f : std::abs(da) * def.radius;
      float bound = length(delta) + rotational_bound;
      if (bound > 1e-9f)
        for (int edge_id : q.edge_ids) {
          const auto& edge = s.edges[edge_id];
          if (!swept_near_edge(start, start + delta, edge, def.radius + .002f)) continue;
          const Shape& wall = walls_[edge_id];
          float time = 0;
          Vec2 normal{};
          bool found = false;
          for (int iteration = 0; iteration < 64; ++iteration) {
#ifdef FG_CONTROL_PROFILE
            ++q.work.ccd_iterations;
#endif
            const Shape& moving = body_shape(b, start + delta * time, a0 + da * time);
            float gap = separation(moving, wall, normal);
            if (gap < .0001f) {
              // A fixed separating axis certifies the entire remaining sweep.
              // The rotational support bound also covers segment endpoints.
              if (time == 0 && gap >= -.0005f &&
                  dot(delta, normal) + (def.vertices.empty() ? 0.f : std::abs(da) * def.radius) <= 0)
                break;
              found = true;
              break;
            }
            // Advance along this separating axis, not the total travel speed.
            // Tangential motion cannot close its gap; rotation remains bounded
            // for every hull vertex, so this also certifies no-hit sweeps.
            const float closing_bound = dot(delta, normal) + rotational_bound;
            if (closing_bound <= 0) break;
            time += .9f * gap / closing_bound;
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
      const Shape& moving = body_shape(b, position(r, l, b), angle(r, l, b));
      const Shape& wall = walls_[hit];
      auto& contact = wall_contact(int(b), hit, hit_n, contact_point(moving, wall, hit_n));
      impulse(r, s, contact, !contact.counted);
      if (contact.target_velocity > 0 || (!def.vertices.empty() && omega(r, l, b) != 0))
        position(r, l, b, position(r, l, b) - hit_n * .0002f);
      count_wall(contact);
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
    const Vec2 p = position(r, l, b);
    body_edges(b, swept(p, p, s.bodies[b].radius + .002f));
    if (q.edge_ids.empty()) continue;
    const Shape& moving = body_shape(b, p, angle(r, l, b));
    for (int e : q.edge_ids) {
      if (!swept_near_edge(moving.center, moving.center, s.edges[e], moving.radius + .002f)) continue;
      const Shape& wall = walls_[e];
      Vec2 n;
      float gap = separation(moving, wall, n);
      if (gap < 0) {
        auto& c = wall_contact(int(b), e, n, contact_point(moving, wall, n));
        c.depth = -gap;
        // New high-speed overlaps retain restitution. Continuing contacts keep
        // the target and accumulated impulses from their earlier resolution.
        if (!c.counted) impulse(r, s, c, true);
        count_wall(c);
        append(c);
      }
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
      const Shape& sa = body_shape(a, position(r, l, a), angle(r, l, a));
      const Shape& sb = body_shape(b, position(r, l, b), angle(r, l, b));
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
#ifdef FG_CONTROL_PROFILE
            ++q.work.ccd_iterations;
#endif
            body_shape(a, q.old_positions[a] + da * time, q.old_angles[a] + aa * time);
            body_shape(b, q.old_positions[b] + db * time, q.old_angles[b] + ab * time);
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
    if (c.b < 0) continue;  // Wall contacts were counted once by body/edge.
    ++result.collisions;
    if (s.bodies[c.a].controlled || (c.b >= 0 && s.bodies[c.b].controlled)) {
      result.reward -= s.collision_penalty;
      if ((c.b < 0 && s.lethal_walls) || (c.b >= 0 && s.lethal_bodies))
        word(r, 7, 1);
    }
  }
  for (int iteration = 0; iteration < s.solver_iterations; ++iteration)
    for (auto& c : q.contacts) impulse(r, s, c, iteration == 0 && c.b >= 0);
  for (auto& c : q.contacts) {
    float ia = 1 / s.bodies[c.a].mass,
          ib = c.b < 0 ? 0 : 1 / s.bodies[c.b].mass;
    float correction = .8f * std::max(0.f, c.depth - .0005f) / (ia + ib);
    position(r, l, c.a, position(r, l, c.a) - c.normal * (correction * ia));
    if (c.b >= 0)
      position(r, l, c.b, position(r, l, c.b) + c.normal * (correction * ib));
  }
}
float Physics::potential(const float* r, const uint32_t* attachments) const {
  const auto& s = *scene;
  const auto& l = s.layout;
  float total = 0;
  for (size_t c = 0; c < s.controlled.size(); ++c) {
    int b = s.controlled[c];
    Vec2 p = position(r, l, b);
    float best = 0;
    if (s.task == "harvest") {
      int hook = -1, attached = -1;
      for (size_t t = 0; t < s.tethers.size(); ++t)
        if (s.tethers[t].owner == b) {
          hook = s.tethers[t].a;
          attached = int(attachments ? attachments[t] : word(r, l.joints + 2 * t)) - 1;
          break;
        }
      best = 1e6f;
      if (attached >= 0) {
        for (const auto& base : s.bases)
          best = std::min(best, length(position(r, l, attached) - base.position));
      } else if (hook >= 0) {
        for (size_t i = 0; i < s.bodies.size(); ++i)
          if (available_cargo(s, r, i))
            best = std::min(best, length(position(r, l, hook) - position(r, l, i)));
      }
      if (best == 1e6f) best = 0;
    } else if (!s.gates.empty())
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
      for (size_t i = 0; i < s.tethers.size(); ++i) {
        const uint32_t target =
            attachments ? attachments[i] : word(r, l.joints + 2 * i);
        if (s.tethers[i].a == b && target) attached = int(target) - 1;
      }
      if (attached >= 0) {
        best = 1e6f;
        for (const auto& base : s.bases)
          best =
              std::min(best, length(position(r, l, attached) - base.position));
      } else {
        best = 1e6f;
        for (size_t i = 0; i < s.bodies.size(); ++i)
          if (available_cargo(s, r, i))
            best = std::min(best, length(p - position(r, l, i)));
        if (best == 1e6f) best = 0;
      }
    }
    total -= best;
  }
  return total / float(std::max(size_t(1), s.controlled.size()));
}
void Physics::mechanics(float* r, StepResult& result, uint32_t checkpoint_stage) {
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
      if (s.task == "tandem" && gate != checkpoint_stage) continue;
      const auto& target = s.gates[gate % s.gates.size()];
      if (length2(position(r, l, b) - target.position) <
          target.radius * target.radius) {
        word(r, l.gates + c, gate + 1);
        word(r, 6, word(r, 6) + 1);
        result.reward += s.task == "tandem"
                             ? s.gate_reward / float(s.controlled.size())
                             : s.gate_reward;
      }
    }
  for (size_t b = 0; b < s.bodies.size(); ++b) {
    if (!s.bodies[b].cargo) continue;
    if (s.keep_delivered_rocks &&
        word(r, l.flags + b) == (active_flag | delivered_flag)) {
      const Vec2 p = position(r, l, b);
      const bool outside = std::all_of(s.bases.begin(), s.bases.end(),
          [&](const Zone& base) {
            return length2(p - base.position) > base.radius * base.radius;
          });
      if (outside) word(r, l.flags + b, active_flag);
      // Do not count another delivery while inside either ring. Acquisition
      // below can resume on this frame once all outer zones have been exited.
      continue;
    }
    if (!s.keep_delivered_rocks && s.bodies[b].respawn &&
        word(r, l.flags + b) == delivered_flag)
      respawn_cargo(s, r, b);
    if (word(r, l.flags + b) & active_flag)
      for (const auto& base : s.bases)
        if (length2(position(r, l, b) - base.position) <
            base.radius * base.radius * (s.keep_delivered_rocks ? .25f : 1.f)) {
          word(r, l.flags + b,
               delivered_flag | (s.keep_delivered_rocks ? active_flag : 0));
          word(r, 4, word(r, 4) + 1);
          result.reward += s.delivery_reward;
          for (size_t t = 0; t < s.tethers.size(); ++t)
            if (word(r, l.joints + 2 * t) == b + 1)
              word(r, l.joints + 2 * t, 0);
          if (!s.keep_delivered_rocks && s.bodies[b].respawn) respawn_cargo(s, r, b);
          break;
        }
  }
  for (size_t t = 0; t < s.tethers.size(); ++t) {
    const auto& def = s.tethers[t];
    if (!def.automatic || word(r, l.joints + 2 * t) ||
        !(word(r, l.flags + def.a) & active_flag) ||
        (def.owner >= 0 && !(word(r, l.flags + def.owner) & active_flag)))
      continue;
    int best = -1;
    float dist = def.hook_range;
    for (size_t b = 0; b < s.bodies.size(); ++b)
      if (available_cargo(s, r, b)) {
        float d = length(position(r, l, def.a) - position(r, l, b));
        if (d < dist) {
          best = int(b);
          dist = d;
        }
      }
    if (best >= 0) {
      if (s.task == "harvest") result.reward += s.catch_reward;
      word(r, l.joints + 2 * t, uint32_t(best + 1));
      r[l.joints + 2 * t + 1] = std::max(.1f, dist);
    }
  }
}
void Physics::step_world(float* r, const float* actions, int frames,
                         StepResult& result, int slot) {
  result = {};
  const auto& s = *scene;
  for (auto& cache : scratch_[slot].shapes) cache.valid = false;
  std::fill(scratch_[slot].empty_edge_valid.begin(), scratch_[slot].empty_edge_valid.end(), 0);
  auto& bounded = scratch_[slot].bounded_actions;
  for (size_t i = 0; i < s.channels.size(); ++i)
    bounded[i] = std::clamp(actions[i], s.channels[i].low, s.channels[i].high);
  actions = bounded.data();
  for (int frame = 0; frame < frames && !word(r, 7); ++frame) {
    auto& wall_contacts = scratch_[slot].frame_wall_contacts;
    std::fill(wall_contacts.begin(), wall_contacts.end(), 0);
    // Freeze the stage through movement and all crossings. Counters only change
    // in mechanics, so eligibility and target stay fixed for this entire frame.
    uint32_t checkpoint_stage = std::numeric_limits<uint32_t>::max();
    const bool tandem_checkpoint = s.task == "tandem" && !s.gates.empty();
    if (tandem_checkpoint)
      for (size_t c = 0; c < s.controlled.size(); ++c)
        checkpoint_stage = std::min(checkpoint_stage, word(r, s.layout.gates + c));
    float before = tandem_checkpoint ? 0 : potential(r);
    auto& attachments = scratch_[slot].frame_tethers;
    for (size_t t = 0; t < s.tethers.size(); ++t)
      attachments[t] = word(r, s.layout.joints + 2 * t);
    auto& hooked = scratch_[slot].frame_hooked_rocks;
    auto& rock_starts = scratch_[slot].frame_rock_positions;
    if (s.hooked_rock_distance_reward > 0) {
      std::fill(hooked.begin(), hooked.end(), 0);
      for (size_t t = 0; t < s.tethers.size(); ++t) {
        if (!attachments[t] || !s.bodies[s.tethers[t].a].controlled ||
            !(word(r, s.layout.flags + s.tethers[t].a) & active_flag)) continue;
        const size_t b = attachments[t] - 1;
        if (b < s.bodies.size() && s.bodies[b].cargo &&
            (word(r, s.layout.flags + b) & active_flag) && !hooked[b]) {
          hooked[b] = 1;
          rock_starts[b] = position(r, s.layout, b);
        }
      }
    }
    auto& starts = scratch_[slot].frame_positions;
    if (s.distance_squared_reward > 0)
      for (size_t c = 0; c < s.controlled.size(); ++c)
        starts[c] = position(r, s.layout, s.controlled[c]);
    for (int k = 0; k < s.substeps; ++k)
      substep(r, actions, s.dt / s.substeps, scratch_[slot], result);
    if (tandem_checkpoint) {
      // Everyone remains attracted to the shared checkpoint, including agents
      // that cleared it. Average distances before mapping to a positive score.
      if (s.progress_reward > 0 && !s.controlled.empty()) {
        const auto& target = s.gates[checkpoint_stage % s.gates.size()];
        float distance = 0;
        for (int body : s.controlled)
          distance += length(position(r, s.layout, body) - target.position);
        const float mean_distance = distance / float(s.controlled.size());
        result.reward += s.progress_reward * target.radius / (target.radius + mean_distance);
      }
    } else {
      // Preserve signed target progress in other tasks, keeping hauling targets
      // fixed across broken/re-hooked ropes and evaluating before respawns.
      result.reward += s.progress_reward * (potential(r, attachments.data()) - before);
    }
    // Per-physics-frame displacement, averaged over vehicles. Evaluate before
    // mechanics/extension respawns, so teleportation never earns travel reward.
    if (s.distance_squared_reward > 0 && !s.controlled.empty()) {
      float squared_distance = 0;
      for (size_t c = 0; c < s.controlled.size(); ++c)
        squared_distance += length2(position(r, s.layout, s.controlled[c]) - starts[c]);
      result.reward += s.distance_squared_reward * squared_distance / float(s.controlled.size());
    }
    // Reward the current formation once per frame, independently of target
    // progress. Compiled pairs contain every unordered controlled-body pair.
    if (s.formation_reward > 0 && !s.formation_pairs.empty()) {
      float formation = 1;
      for (const auto& pair : s.formation_pairs) {
        const float distance = length(position(r, s.layout, pair.a) -
                                      position(r, s.layout, pair.b));
        formation *= pair.distance / (pair.distance + std::abs(pair.distance - distance));
      }
      result.reward += s.formation_reward * formation;
    }
    // Sum actual translation of each distinct rock hooked at frame start.
    // Run before delivery/respawn mechanics so teleports never earn reward.
    if (s.hooked_rock_distance_reward > 0) {
      float distance = 0;
      for (size_t b = 0; b < s.bodies.size(); ++b)
        if (hooked[b]) distance += length(position(r, s.layout, b) - rock_starts[b]);
      result.reward += s.hooked_rock_distance_reward * distance;
    }
    mechanics(r, result, checkpoint_stage);
    for (const auto& extension : s.extensions) {
      const float earned = result.reward;
      if (extension.step) extension.step(s, extension, r, actions, result);
      if (s.task == "harvest") result.reward = earned;
    }
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
#ifdef FG_CONTROL_PROFILE
CollisionWork Physics::collision_work() const {
  CollisionWork total;
  for (const auto& q : scratch_) {
    total.edge_queries += q.work.edge_queries;
    total.edge_candidates += q.work.edge_candidates;
    total.separation_checks += q.work.separation_checks;
    total.ccd_iterations += q.work.ccd_iterations;
  }
  return total;
}
#endif
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
    if (s.flight_mode) force.y -= def.mass * s.downward_gravity;
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
      const auto& wall = walls_[size_t(&edge - s.edges.data())];
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
