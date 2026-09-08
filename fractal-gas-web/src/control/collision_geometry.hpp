#pragma once
#include <algorithm>
#include <limits>
#include "control/scene.hpp"

namespace fg::control::geometry {
struct Shape {
  Vec2 center;
  float radius = 0;
  int n = 0;
  std::array<Vec2, 32> vertices, axes;
  int axis_count = 0;
};
inline void build_shape(Shape& s, const BodyDef& body, Vec2 p, float a) {
  s.axis_count = 0;
  s.center = p;
  s.radius = body.radius;
  s.n = int(body.vertices.size());
  if (!s.n) return;
  float c = std::cos(a), sn = std::sin(a);
  for (int i = 0; i < s.n; ++i) {
    Vec2 v = body.vertices[i];
    s.vertices[i] = p + Vec2{v.x * c - v.y * sn, v.x * sn + v.y * c};
  }
  for (int i = 0; i < s.n; ++i) {
    Vec2 axis = perp(s.vertices[(i + 1) % s.n] - s.vertices[i]);
    if (length2(axis) >= 1e-16f) s.axes[s.axis_count++] = normalized(axis);
  }
}
inline Shape shape(const BodyDef& body, Vec2 p, float a) {
  Shape s;
  build_shape(s, body, p, a);
  return s;
}
inline Shape shape(const Edge& e) {
  Shape s;
  s.center = (e.a + e.b) * .5f;
  s.n = 2;
  s.vertices[0] = e.a;
  s.vertices[1] = e.b;
  s.axes[0] = normalized(perp(e.b - e.a));
  s.axes[1] = normalized(e.b - e.a);
  s.axis_count = 2;
  return s;
}
inline std::pair<float, float> project(const Shape& s, Vec2 n) {
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
inline float separation(const Shape& a, const Shape& b, Vec2& normal) {
  float gap = -std::numeric_limits<float>::infinity();
  auto unit_axis = [&](Vec2 n) {
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
  auto axis = [&](Vec2 n) {
    if (length2(n) >= 1e-16f) unit_axis(normalized(n));
  };
  if (!a.n && !b.n) axis(b.center - a.center);
  for (int i = 0; i < a.axis_count; ++i) unit_axis(a.axes[i]);
  for (int i = 0; i < b.axis_count; ++i) unit_axis(b.axes[i]);
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
inline Vec2 contact_point(const Shape& a, const Shape& b, Vec2 n) {
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

// A capsule enclosing the full rotating hull. A negative answer is safe even
// when the polygon spins: BodyDef::radius encloses every local vertex.
inline bool swept_near_edge(Vec2 a, Vec2 b, const Edge& edge, float radius) {
  auto orient = [](Vec2 p, Vec2 q, Vec2 r) {
    return (double(q.x) - p.x) * (double(r.y) - p.y) -
           (double(q.y) - p.y) * (double(r.x) - p.x);
  };
  const double x = orient(a, b, edge.a), y = orient(a, b, edge.b),
               u = orient(edge.a, edge.b, a), v = orient(edge.a, edge.b, b);
  if (((x <= 0 && y >= 0) || (x >= 0 && y <= 0)) &&
      ((u <= 0 && v >= 0) || (u >= 0 && v <= 0))) return true;
  const float d2 = std::min({length2(a - closest(a, edge.a, edge.b)),
                            length2(b - closest(b, edge.a, edge.b)),
                            length2(edge.a - closest(edge.a, a, b)),
                            length2(edge.b - closest(edge.b, a, b))});
  return !(d2 > radius * radius);
}
struct ShapeCache {
  Shape value;
  float angle = 0;
  bool valid = false;
  const Shape& get(const BodyDef& body, Vec2 p, float a) {
    if (!valid || value.center.x != p.x || value.center.y != p.y || angle != a) {
      build_shape(value, body, p, a);
      angle = a;
      valid = true;
    }
    return value;
  }
};

}  // namespace fg::control::geometry
