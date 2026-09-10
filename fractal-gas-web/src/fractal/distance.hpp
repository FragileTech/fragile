#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fg {
class ThreadPool;
enum class DistanceMetric { L2, Cosine };
DistanceMetric parse_distance_metric(const std::string& name);
const char* distance_metric_name(DistanceMetric metric);
// Rows contain d finite floats. Zero-vector cosine convention: 0 for two
// zero vectors, 1 when exactly one vector is zero.
float row_distance(const float* a, const float* b, int32_t d,
                   DistanceMetric metric = DistanceMetric::L2);
void companion_distances_into(const std::vector<float>& observations,
                              const std::vector<int32_t>& companions, int32_t n, int32_t d,
                              ThreadPool* pool, std::vector<float>& out,
                              DistanceMetric metric = DistanceMetric::L2);
}  // namespace fg
