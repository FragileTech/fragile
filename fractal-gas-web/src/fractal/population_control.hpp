#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

namespace fg::fractal {
enum class RemovalPolicy { VirtualReward, CumulativeReward };
inline RemovalPolicy removal_policy(const std::string& value) {
  if (value == "virtual_reward") return RemovalPolicy::VirtualReward;
  if (value == "cumulative_reward") return RemovalPolicy::CumulativeReward;
  throw std::invalid_argument("Unknown population removal policy");
}
inline const char* removal_policy_name(RemovalPolicy value) {
  return value == RemovalPolicy::VirtualReward ? "virtual_reward" : "cumulative_reward";
}
template <class Score>
std::vector<int32_t> retention_order(int n, Score score) {
  std::vector<int32_t> rows(n);
  std::iota(rows.begin(), rows.end(), 0);
  auto finite = [&](int i) { double v = score(i); return std::isfinite(v) ? v : -INFINITY; };
  std::stable_sort(rows.begin(), rows.end(), [&](int a, int b) { return finite(a) > finite(b); });
  return rows;
}
struct PopulationStatus {
  int maximum = 0, active = 0, requested = 0;
  RemovalPolicy policy = RemovalPolicy::VirtualReward;
  bool pending() const { return active != requested; }
};
inline void validate_population(int count, int maximum, int elites, int minimum = 1) {
  if (count < std::max(minimum, elites) || count > maximum)
    throw std::invalid_argument("Active walkers must be within the configured maximum and at least the elite count");
}
}  // namespace fg::fractal
