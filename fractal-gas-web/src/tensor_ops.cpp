#include "tensor_ops.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "thread_pool.hpp"

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

std::vector<float> relativize_with_stats(const std::vector<float>& x,
                                         double mean, double stdv) {
  const size_t n = x.size();
  std::vector<float> out(n, 1.0f);
  if (stdv == 0.0 || !std::isfinite(stdv) || !std::isfinite(mean)) return out;
  for (size_t i = 0; i < n; ++i) {
    const double s = (static_cast<double>(x[i]) - mean) / stdv;
    out[i] = static_cast<float>(s > 0.0 ? std::log1p(s) + 1.0 : std::exp(s));
  }
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
  // torch: std of a single element (or empty) is NaN -> all-ones branch.
  if (x.size() < 2) return std::vector<float>(x.size(), 1.0f);
  const std::vector<uint8_t> all(x.size(), 1);
  const auto stats = mean_std_masked(x, all);
  return relativize_with_stats(x, stats.first, stats.second);
}

std::vector<float> l2_norm_companions(const std::vector<float>& observations,
                                      const std::vector<int32_t>& companions,
                                      int32_t n, int32_t obs_dim,
                                      ThreadPool* pool) {
  std::vector<float> distances(static_cast<size_t>(n));
  const auto d = static_cast<size_t>(obs_dim);
  const float* obs = observations.data();
  const auto row = [&](int32_t i, int /*slot*/) {
    const auto ii = static_cast<size_t>(i);
    const auto c = static_cast<size_t>(companions[ii]);
    distances[ii] =
        static_cast<float>(std::sqrt(l2_squared(obs + ii * d, obs + c * d, d)));
  };
  // Waking the pool costs more than small batches (RAM obs and below stay
  // borderline; Coords is tiny) — only fan out when there is real work.
  if (pool != nullptr && pool->size() > 1 &&
      static_cast<int64_t>(n) * obs_dim >= 65536) {
    pool->parallel_for(n, row);
  } else {
    for (int32_t i = 0; i < n; ++i) row(i, 0);
  }
  return distances;
}

std::vector<int32_t> random_alive_compas(const std::vector<uint8_t>& alive,
                                         Rng& rng) {
  const auto n = static_cast<int32_t>(alive.size());
  std::vector<int32_t> alive_idx;
  alive_idx.reserve(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    if (alive[static_cast<size_t>(i)]) alive_idx.push_back(i);
  }

  std::vector<int32_t> pool;
  if (alive_idx.empty()) {
    // get_alive_indexes: torch.all(oobs) -> arange(N)
    pool.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) pool[static_cast<size_t>(i)] = i;
  } else if (static_cast<int32_t>(alive_idx.size()) == n) {
    // random_choice(arange(N), size=N, replace=False) -> random permutation
    pool = rng.permutation(n);
  } else {
    // random_choice(alive_indices, size=N, replace=True)
    pool.resize(static_cast<size_t>(n));
    const auto n_alive = static_cast<int64_t>(alive_idx.size());
    for (int32_t i = 0; i < n; ++i) {
      pool[static_cast<size_t>(i)] =
          alive_idx[static_cast<size_t>(rng.randint(0, n_alive))];
    }
  }

  // random_alive_compas: compas[torch.randperm(compas.size(0))]
  const std::vector<int32_t> perm = rng.permutation(n);
  std::vector<int32_t> out(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = pool[static_cast<size_t>(perm[static_cast<size_t>(i)])];
  }
  return out;
}

}  // namespace fg
