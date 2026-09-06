#include "control/state.hpp"

#include <limits>

namespace fg::control {
namespace {
uint64_t hash_bytes(const uint8_t* p, size_t n) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}
void put(uint8_t* p, uint32_t n) {
  for (int i = 0; i < 4; ++i) p[i] = uint8_t(n >> (8 * i));
}
uint32_t get(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
}  // namespace
void StateBatch::reset(const Scene& s, uint64_t seed) {
  std::memset(storage->data, 0, bytes());
  for (size_t w = 0; w < count; ++w) {
    float* r = row(w);
    rng_state(r, seed);
    for (size_t b = 0; b < s.bodies.size(); ++b) {
      const auto& body = s.bodies[b];
      position(r, layout, b, body.position);
      velocity(r, layout, b, body.velocity);
      angle(r, layout, b) = body.angle;
      omega(r, layout, b) = body.omega;
      word(r, layout.flags + b, active_flag);
      if (body.controlled)
        std::copy(body.actuator.initial_state.begin(),
                  body.actuator.initial_state.end(),
                  r + layout.auxiliary + body.actuator.state_offset);
    }
    for (const auto& extension : s.extensions)
      std::copy(extension.initial_state.begin(), extension.initial_state.end(),
                r + layout.auxiliary + extension.state_offset);
    for (size_t t = 0; t < s.tethers.size(); ++t) {
      word(r, layout.joints + 2 * t, uint32_t(s.tethers[t].b + 1));
      r[layout.joints + 2 * t + 1] = s.tethers[t].rest;
    }
    for (size_t p = 0; p < s.pickups.size(); ++p) {
      r[layout.food + 3 * p] = s.pickups[p].position.x;
      r[layout.food + 3 * p + 1] = s.pickups[p].position.y;
    }
  }
}
void StateBatch::gather(const StateBatch& source, const int32_t* indices,
                        size_t n) {
  if (fingerprint != source.fingerprint || n != count)
    throw std::invalid_argument("Incompatible state batch");
  for (size_t i = 0; i < n; ++i)
    if (indices[i] < 0 || uint32_t(indices[i]) >= source.count)
      throw std::out_of_range("State index");
  if (storage == source.storage) {
    // Explicit overlapping public restores use a temporary; the hot Wave path
    // always uses separate banks.
    std::vector<float> temp(size_t(n) * layout.words);
    for (size_t i = 0; i < n; ++i)
      std::memcpy(temp.data() + i * layout.words, source.row(indices[i]),
                  layout.words * 4);
    for (size_t i = 0; i < n; ++i)
      std::memcpy(row(i), temp.data() + i * layout.words, layout.words * 4);
  } else
    for (size_t i = 0; i < n; ++i)
      std::memcpy(row(i), source.row(indices[i]), layout.words * 4);
}
void StateBatch::validate_row(const float* r) const {
  for (size_t i = 8; i < layout.flags; ++i)
    if (!std::isfinite(r[i]))
      throw std::invalid_argument("Non-finite body state");
  for (size_t b = 0; b < layout.bodies; ++b)
    if (word(r, layout.flags + b) > 3)
      throw std::invalid_argument("Invalid body flags");
  for (size_t t = 0; t < layout.tethers; ++t) {
    if (word(r, layout.joints + 2 * t) > layout.bodies ||
        !std::isfinite(r[layout.joints + 2 * t + 1]) ||
        r[layout.joints + 2 * t + 1] < 0)
      throw std::invalid_argument("Invalid tether state");
  }
  for (size_t i = layout.food; i < layout.words; ++i)
    if (!std::isfinite(r[i]))
      throw std::invalid_argument("Non-finite pickup state");
  if (layout.cargo_capacity > 0)
    for (size_t c = 0; c < layout.controlled; ++c) {
      const float* cargo = r + layout.cargo + 4 * c;
      if (cargo[0] < 0 || cargo[0] > layout.cargo_capacity ||
          (cargo[1] != 0 && cargo[1] != 1) || cargo[2] < 0 || cargo[3] < 0 ||
          std::floor(cargo[3]) != cargo[3] ||
          (cargo[1] == 1 && cargo[0] == 0) ||
          (cargo[1] == 0 && (cargo[0] >= layout.cargo_capacity || std::floor(cargo[0]) != cargo[0])))
        throw std::invalid_argument("Invalid cargo state");
    }
  if (word(r, 7) > 1) throw std::invalid_argument("Invalid terminal state");
}
void StateBatch::serialize(uint8_t* out, size_t capacity) const {
  if (capacity < serialized_size())
    throw std::invalid_argument("Snapshot output too small");
  put(out, 0x53434746);
  put(out + 4, 1);
  put(out + 8, uint32_t(fingerprint));
  put(out + 12, uint32_t(fingerprint >> 32));
  put(out + 16, count);
  put(out + 20, layout.words);
  size_t at = 32;
  for (size_t w = 0; w < count; ++w)
    for (size_t k = 0; k < layout.words; ++k, at += 4)
      put(out + at, word(row(w), k));
  uint64_t hash = hash_bytes(out + 32, at - 32);
  put(out + 24, uint32_t(hash));
  put(out + 28, uint32_t(hash >> 32));
}
void StateBatch::deserialize(const uint8_t* in, size_t size) {
  if (size != serialized_size() || get(in) != 0x53434746 || get(in + 4) != 1 ||
      get(in + 16) != count || get(in + 20) != layout.words ||
      (get(in + 8) | (uint64_t(get(in + 12)) << 32)) != fingerprint)
    throw std::invalid_argument(
        "Snapshot version, scene or batch layout mismatch");
  if (hash_bytes(in + 32, size - 32) !=
      (get(in + 24) | (uint64_t(get(in + 28)) << 32)))
    throw std::invalid_argument("Snapshot checksum mismatch");
  // Validate all rows before committing any state; malformed input cannot
  // partially restore a run.
  std::vector<float> scratch(layout.words);
  for (size_t w = 0; w < count; ++w) {
    for (size_t k = 0; k < layout.words; ++k)
      word(scratch.data(), k, get(in + 32 + 4 * (w * layout.words + k)));
    validate_row(scratch.data());
  }
  for (size_t w = 0; w < count; ++w)
    for (size_t k = 0; k < layout.words; ++k)
      word(row(w), k, get(in + 32 + 4 * (w * layout.words + k)));
}
}  // namespace fg::control
