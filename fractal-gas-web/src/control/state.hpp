#pragma once
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "control/scene.hpp"

namespace fg::control {
inline uint32_t word(const float* row, size_t offset) {
  uint32_t v;
  std::memcpy(&v, row + offset, 4);
  return v;
}
inline void word(float* row, size_t offset, uint32_t v) {
  std::memcpy(row + offset, &v, 4);
}
inline uint64_t rng_state(const float* row) {
  return word(row, 1) | (uint64_t(word(row, 2)) << 32);
}
inline void rng_state(float* row, uint64_t v) {
  word(row, 1, uint32_t(v));
  word(row, 2, uint32_t(v >> 32));
}
inline Vec2 position(const float* r, const Layout& l, int b) {
  return {r[8 + b], r[8 + l.bodies + b]};
}
inline Vec2 velocity(const float* r, const Layout& l, int b) {
  return {r[8 + 2 * l.bodies + b], r[8 + 3 * l.bodies + b]};
}
inline void position(float* r, const Layout& l, int b, Vec2 p) {
  r[8 + b] = p.x;
  r[8 + l.bodies + b] = p.y;
}
inline void velocity(float* r, const Layout& l, int b, Vec2 v) {
  r[8 + 2 * l.bodies + b] = v.x;
  r[8 + 3 * l.bodies + b] = v.y;
}
inline float& angle(float* r, const Layout& l, int b) {
  return r[8 + 4 * l.bodies + b];
}
inline float& omega(float* r, const Layout& l, int b) {
  return r[8 + 5 * l.bodies + b];
}
inline float angle(const float* r, const Layout& l, int b) {
  return r[8 + 4 * l.bodies + b];
}
inline float omega(const float* r, const Layout& l, int b) {
  return r[8 + 5 * l.bodies + b];
}
// Active | delivered represents retained cargo locked until it exits a drop zone.
constexpr uint32_t active_flag = 1, delivered_flag = 2;

struct StateBatch {
  struct Storage {
    float* data;
    size_t size;
    explicit Storage(size_t n)
        : data(static_cast<float*>(
              ::operator new[](n * 4, std::align_val_t(64)))),
          size(n) {
      std::memset(data, 0, n * 4);
    }
    ~Storage() { ::operator delete[](data, std::align_val_t(64)); }
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
  };
  std::shared_ptr<Storage> storage;
  uint32_t count = 0;
  Layout layout;
  uint64_t fingerprint = 0;
  StateBatch() = default;
  StateBatch(uint32_t n, const Scene& s)
      : storage(std::make_shared<Storage>(size_t(n) * s.layout.stride)),
        count(n),
        layout(s.layout),
        fingerprint(s.fingerprint) {}
  float* row(size_t i) { return storage->data + i * layout.stride; }
  const float* row(size_t i) const { return storage->data + i * layout.stride; }
  size_t bytes() const { return size_t(count) * layout.stride * 4; }
  void reset(const Scene& scene, uint64_t seed);
  void gather(const StateBatch& source, const int32_t* indices, size_t n);
  size_t serialized_size() const {
    return 32 + size_t(count) * layout.words * 4;
  }
  void serialize(uint8_t* out, size_t capacity) const;
  void deserialize(const uint8_t* in, size_t size);
  void validate_row(const float* r) const;
};
}  // namespace fg::control
