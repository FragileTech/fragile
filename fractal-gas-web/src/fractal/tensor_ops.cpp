#include "fractal/tensor_ops.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "thread_pool.hpp"

namespace fg {

void relativize_with_stats_into(const std::vector<float>& x, double mean, double stdv,
                                std::vector<float>& out) {
  out.resize(x.size());
  if (stdv == 0.0 || !std::isfinite(stdv) || !std::isfinite(mean)) {
    std::fill(out.begin(), out.end(), 1.f);
    return;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    const double z = (static_cast<double>(x[i]) - mean) / stdv;
    out[i] = static_cast<float>(z > 0.0 ? std::log1p(z) + 1.0 : std::exp(z));
  }
}
std::vector<float> relativize_with_stats(const std::vector<float>& x, double mean, double stdv) {
  std::vector<float> out;
  relativize_with_stats_into(x, mean, stdv, out);
  return out;
}

std::pair<double, double> mean_std_masked(const std::vector<float>& x,
                                          const std::vector<uint8_t>& mask) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  size_t count = 0;
  double mean = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (!mask[i]) continue;
    mean += static_cast<double>(x[i]);
    ++count;
  }
  if (count == 0) return {nan, nan};
  mean /= static_cast<double>(count);
  // torch: std of a single element is NaN.
  if (count < 2) return {mean, nan};
  double var = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (!mask[i]) continue;
    const double d = static_cast<double>(x[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(count - 1);  // Bessel correction, matching torch .std()
  return {mean, std::sqrt(var)};
}

std::vector<float> asymmetric_rescale(const std::vector<float>& x) {
  std::vector<float> out;
  asymmetric_rescale_into(x, out);
  return out;
}
void asymmetric_rescale_into(const std::vector<float>& x, std::vector<float>& out) {
  double mean = 0, var = 0;
  for (float v : x) mean += v;
  if (!x.empty()) mean /= x.size();
  for (float v : x) {
    double d = double(v) - mean;
    var += d * d;
  }
  const double stdv = x.size() < 2 ? 0 : std::sqrt(var / double(x.size() - 1));
  out.resize(x.size());
  if (stdv == 0 || !std::isfinite(stdv) || !std::isfinite(mean)) {
    std::fill(out.begin(), out.end(), 1.f);
    return;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    double z = (double(x[i]) - mean) / stdv;
    out[i] = float(z > 0 ? std::log1p(z) + 1 : std::exp(z));
  }
}

std::vector<int32_t> random_alive_compas(const std::vector<uint8_t>& alive, Rng& rng) {
  std::vector<int32_t> out;
  CompanionScratch scratch;
  random_alive_compas_into(alive, rng, out, scratch);
  return out;
}
void random_alive_compas_into(const std::vector<uint8_t>& alive, Rng& rng,
                              std::vector<int32_t>& out, CompanionScratch& scratch) {
  const auto n = static_cast<int32_t>(alive.size());
  auto& alive_idx = scratch.alive;
  alive_idx.clear();
  alive_idx.reserve(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    if (alive[static_cast<size_t>(i)]) alive_idx.push_back(i);
  }

  auto& pool = scratch.pool;
  if (alive_idx.empty()) {
    // get_alive_indexes: torch.all(oobs) -> arange(N)
    pool.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) pool[static_cast<size_t>(i)] = i;
  } else if (static_cast<int32_t>(alive_idx.size()) == n) {
    // random_choice(arange(N), size=N, replace=False) -> random permutation
    rng.permutation_into(n, pool);
  } else {
    // random_choice(alive_indices, size=N, replace=True)
    pool.resize(static_cast<size_t>(n));
    const auto n_alive = static_cast<int64_t>(alive_idx.size());
    for (int32_t i = 0; i < n; ++i) {
      pool[static_cast<size_t>(i)] = alive_idx[static_cast<size_t>(rng.randint(0, n_alive))];
    }
  }

  // random_alive_compas: compas[torch.randperm(compas.size(0))]
  auto& perm = scratch.permutation;
  rng.permutation_into(n, perm);
  out.resize(n);
  for (int32_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = pool[static_cast<size_t>(perm[static_cast<size_t>(i)])];
  }
}

}  // namespace fg
