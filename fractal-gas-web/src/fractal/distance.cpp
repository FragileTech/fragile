#include "fractal/distance.hpp"
#include "fractal/tensor_ops.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace fg {
namespace {

/// Squared L2 distance between two rows. Four independent double
/// accumulators let the compiler vectorize (f64x2 with wasm SIMD); this
/// reassociates the previous serial sum, which perturbs results far below
/// the 1e-5 fixture tolerance and identically on every thread count.
double l2_squared(const float* a, const float* b, size_t d) {
  double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
  size_t k = 0;
  for (; k + 4 <= d; k += 4) {
    const double d0 = static_cast<double>(a[k]) - static_cast<double>(b[k]);
    const double d1 = static_cast<double>(a[k + 1]) - static_cast<double>(b[k + 1]);
    const double d2 = static_cast<double>(a[k + 2]) - static_cast<double>(b[k + 2]);
    const double d3 = static_cast<double>(a[k + 3]) - static_cast<double>(b[k + 3]);
    acc0 += d0 * d0;
    acc1 += d1 * d1;
    acc2 += d2 * d2;
    acc3 += d3 * d3;
  }
  for (; k < d; ++k) {
    const double diff = static_cast<double>(a[k]) - static_cast<double>(b[k]);
    acc0 += diff * diff;
  }
  return (acc0 + acc1) + (acc2 + acc3);
}

}  // namespace


DistanceMetric parse_distance_metric(const std::string& name) {
  if (name == "l2") return DistanceMetric::L2;
  if (name == "cosine") return DistanceMetric::Cosine;
  throw std::invalid_argument("Unknown distance metric: " + name);
}
const char* distance_metric_name(DistanceMetric metric) {
  switch (metric) {
    case DistanceMetric::L2: return "l2";
    case DistanceMetric::Cosine: return "cosine";
  }
  throw std::invalid_argument("Invalid distance metric");
}
namespace {
float unchecked_distance(const float* a, const float* b, size_t d, DistanceMetric metric) {
  if (metric == DistanceMetric::L2) return float(std::sqrt(l2_squared(a, b, d)));
  double aa = 0, bb = 0, ab = 0;
  for (size_t k = 0; k < d; ++k) {
    aa += double(a[k]) * a[k]; bb += double(b[k]) * b[k]; ab += double(a[k]) * b[k];
  }
  if (aa == 0 || bb == 0) return aa == bb ? 0.f : 1.f;
  return float(std::clamp(1. - ab / (std::sqrt(aa) * std::sqrt(bb)), 0., 2.));
}
}
float row_distance(const float* a, const float* b, int32_t d, DistanceMetric metric) {
  distance_metric_name(metric);
  if (d < 0 || (d && (!a || !b))) throw std::invalid_argument("Invalid distance row");
  for (int k = 0; k < d; ++k)
    if (!std::isfinite(a[k]) || !std::isfinite(b[k]))
      throw std::invalid_argument("Nonfinite observation");
  return unchecked_distance(a, b, size_t(d), metric);
}
void companion_distances_into(const std::vector<float>& observations,
                              const std::vector<int32_t>& companions, int32_t n, int32_t d,
                              ThreadPool* pool, std::vector<float>& out, DistanceMetric metric) {
  distance_metric_name(metric);
  if (n < 0 || d < 1 || observations.size() != size_t(n) * size_t(d) ||
      companions.size() != size_t(n) || &observations == &out)
    throw std::invalid_argument("Invalid companion distance shape or aliased output");
  for (int c : companions)
    if (c < 0 || c >= n) throw std::invalid_argument("Invalid distance companion");
  for (float x : observations)
    if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite observation");
  out.resize(n);
  auto row = [&](int i, int) {
    out[i] = unchecked_distance(observations.data() + size_t(i) * d,
                               observations.data() + size_t(companions[i]) * d, d, metric);
  };
  if (pool && pool->size() > 1 && int64_t(n) * d >= 65536) pool->parallel_for(n, row);
  else for (int i = 0; i < n; ++i) row(i, 0);
}
std::vector<float> l2_norm_companions(const std::vector<float>& observations,
                                      const std::vector<int32_t>& companions, int32_t n,
                                      int32_t obs_dim, ThreadPool* pool) {
  std::vector<float> distances;
  l2_norm_companions_into(observations, companions, n, obs_dim, pool, distances);
  return distances;
}
void l2_norm_companions_into(const std::vector<float>& observations,
                             const std::vector<int32_t>& companions, int32_t n, int32_t obs_dim,
                             ThreadPool* pool, std::vector<float>& distances) {
  companion_distances_into(observations, companions, n, obs_dim, pool, distances);
}
}
