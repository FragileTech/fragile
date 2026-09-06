#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fg::control {
constexpr float pi = 3.14159265358979323846f;
struct Vec2 {
  float x = 0, y = 0;
  Vec2 operator+(Vec2 b) const { return {x + b.x, y + b.y}; }
  Vec2 operator-(Vec2 b) const { return {x - b.x, y - b.y}; }
  Vec2 operator-() const { return {-x, -y}; }
  Vec2 operator*(float k) const { return {x * k, y * k}; }
  Vec2 operator/(float k) const { return {x / k, y / k}; }
  Vec2& operator+=(Vec2 b) {
    x += b.x;
    y += b.y;
    return *this;
  }
  Vec2& operator-=(Vec2 b) {
    x -= b.x;
    y -= b.y;
    return *this;
  }
};
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
inline Vec2 perp(Vec2 a) { return {-a.y, a.x}; }
inline float length2(Vec2 a) { return dot(a, a); }
inline float length(Vec2 a) { return std::sqrt(length2(a)); }
inline Vec2 normalized(Vec2 a) {
  float n = length(a);
  return n > 1e-8f ? a / n : Vec2{1, 0};
}
inline Vec2 rotate(Vec2 a, float angle) {
  float c = std::cos(angle), s = std::sin(angle);
  return {c * a.x - s * a.y, s * a.x + c * a.y};
}
inline Vec2 closest(Vec2 p, Vec2 a, Vec2 b) {
  Vec2 d = b - a;
  float n = length2(d);
  return a + d * (n > 0 ? std::clamp(dot(p - a, d) / n, 0.f, 1.f) : 0.f);
}
struct Aabb {
  Vec2 lo, hi;
  bool overlaps(const Aabb& b) const {
    return lo.x <= b.hi.x && hi.x >= b.lo.x && lo.y <= b.hi.y && hi.y >= b.lo.y;
  }
};
inline uint64_t mix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
inline float random01(uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  return static_cast<float>(mix64(state) >> 40) * (1.f / 16777216.f);
}
}  // namespace fg::control
