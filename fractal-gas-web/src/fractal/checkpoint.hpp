#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fg::fractal {
// Checkpoints are build/backend-specific; state snapshots retain their portable
// ABI.
struct CheckpointWriter {
  std::vector<uint8_t> data;
  template <class T>
  void scalar(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto p = reinterpret_cast<const uint8_t*>(&v);
    data.insert(data.end(), p, p + sizeof(T));
  }
  template <class T>
  void vector(const std::vector<T>& v) {
    scalar(uint64_t(v.size()));
    if (!v.empty()) {
      auto p = reinterpret_cast<const uint8_t*>(v.data());
      data.insert(data.end(), p, p + v.size() * sizeof(T));
    }
  }
  void string(const std::string& v) { vector(std::vector<uint8_t>(v.begin(), v.end())); }
};
struct CheckpointReader {
  const uint8_t* data;
  size_t size, at = 0;
  CheckpointReader(const uint8_t* p, size_t n) : data(p), size(n) {}
  template <class T>
  T scalar() {
    static_assert(std::is_trivially_copyable_v<T>);
    if (at > size || sizeof(T) > size - at) throw std::invalid_argument("Truncated checkpoint");
    T v;
    std::memcpy(&v, data + at, sizeof(T));
    at += sizeof(T);
    return v;
  }
  template <class T>
  std::vector<T> vector(size_t limit = 128 * 1024 * 1024) {
    auto n = scalar<uint64_t>();
    if (n > limit / sizeof(T) || n > (size - at) / sizeof(T))
      throw std::invalid_argument("Invalid checkpoint vector size");
    std::vector<T> v(n);
    if (n) std::memcpy(v.data(), data + at, n * sizeof(T));
    at += n * sizeof(T);
    return v;
  }
  std::string string(size_t limit = 1024 * 1024) {
    auto v = vector<uint8_t>(limit);
    return {v.begin(), v.end()};
  }
  void finish() {
    if (at != size) throw std::invalid_argument("Trailing checkpoint data");
  }
};
inline uint64_t checkpoint_hash(const uint8_t* p, size_t n) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}
}  // namespace fg::fractal
