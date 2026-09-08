// Direct translations of the tensor helpers in
// src/fragile/fractalai/fractalai.py (asymmetric_rescale, l2_norm,
// random_alive_compas / get_alive_indexes) operating on flat float vectors.
#ifndef FRACTAL_GAS_TENSOR_OPS_HPP
#define FRACTAL_GAS_TENSOR_OPS_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include "fractal/rng.hpp"

namespace fg {

/// Python reference (fractalai.py::asymmetric_rescale):
///   std = x.std()                      # Bessel-corrected (N-1 divisor)
///   if std == 0 or isnan or isinf: return ones_like(x)
///   standard = (x - x.mean()) / std
///   return where(standard > 0, log(1 + standard) + 1, exp(standard))
/// A single-element tensor has std == NaN in torch, so it also maps to ones.
std::vector<float> asymmetric_rescale(const std::vector<float>& x);

/// Python reference (cb9f3296 fractalai.py::relativize(x, std, mean)) with
/// externally supplied statistics (the tree normalizes with leaf-only
/// mean/std):
///   if std == 0 or isnan or isinf: return ones_like(x)
///   standard = (x - mean) / std
///   return where(standard > 0, log(1 + standard) + 1, exp(standard))
/// asymmetric_rescale(x) == relativize_with_stats(x, mean(x), std(x)).
std::vector<float> relativize_with_stats(const std::vector<float>& x, double mean, double stdv);

/// Mean and Bessel-corrected std (torch .mean()/.std()) over the elements
/// where mask != 0. Fewer than two selected elements give std = NaN (and
/// mean = NaN when none), matching torch, so relativize_with_stats falls
/// back to ones.
std::pair<double, double> mean_std_masked(const std::vector<float>& x,
                                          const std::vector<uint8_t>& mask);

class ThreadPool;
struct CompanionScratch {
  std::vector<int32_t> alive, pool, permutation;
};
void random_alive_compas_into(const std::vector<uint8_t>& alive, Rng& rng,
                              std::vector<int32_t>& out, CompanionScratch& scratch);
void relativize_with_stats_into(const std::vector<float>& x, double mean, double stdv,
                                std::vector<float>& out);
void asymmetric_rescale_into(const std::vector<float>& x, std::vector<float>& out);
void l2_norm_companions_into(const std::vector<float>& observations,
                             const std::vector<int32_t>& companions, int32_t n, int32_t obs_dim,
                             ThreadPool* pool, std::vector<float>& distances);

/// Python reference (fractalai.py::l2_norm): row-wise Euclidean distance
/// between observations[i] and observations[companions[i]] over obs_dim
/// components. Rows are independent, so with `pool` large batches are
/// split across it (deterministic: per-row arithmetic never depends on
/// the partition); pass nullptr to force the serial path.
std::vector<float> l2_norm_companions(const std::vector<float>& observations,
                                      const std::vector<int32_t>& companions, int32_t n,
                                      int32_t obs_dim, ThreadPool* pool = nullptr);

/// Python reference (fractalai.py::random_alive_compas + get_alive_indexes),
/// called with oobs = ~alive:
///   - all walkers dead  -> pool = arange(N)
///   - all walkers alive -> pool = random permutation of 0..N-1
///                          (random_choice with replace=False, size N)
///   - some dead         -> pool = N uniform draws WITH replacement from the
///                          alive indices
///   then the pool is shuffled once more by a random permutation.
std::vector<int32_t> random_alive_compas(const std::vector<uint8_t>& alive, Rng& rng);

/// Python reference (fractal_gas.py::_clone_tensor):
///   cloned = data.clone(); cloned[will_clone] = data[companions[will_clone]]
/// Gather semantics: companion values are read from the ORIGINAL array.
template <typename T>
std::vector<T> gather_clone(const std::vector<T>& data, const std::vector<int32_t>& companions,
                            const std::vector<uint8_t>& will_clone) {
  std::vector<T> cloned = data;
  const size_t n = will_clone.size();
  for (size_t i = 0; i < n; ++i) {
    if (will_clone[i]) cloned[i] = data[static_cast<size_t>(companions[i])];
  }
  return cloned;
}

/// gather_clone for row-major [N, dim] data (observations).
template <typename T>
std::vector<T> gather_clone_rows(const std::vector<T>& data,
                                 const std::vector<int32_t>& companions,
                                 const std::vector<uint8_t>& will_clone, int32_t dim) {
  std::vector<T> cloned = data;
  const size_t n = will_clone.size();
  const auto d = static_cast<size_t>(dim);
  for (size_t i = 0; i < n; ++i) {
    if (!will_clone[i]) continue;
    const auto c = static_cast<size_t>(companions[i]);
    for (size_t k = 0; k < d; ++k) cloned[i * d + k] = data[c * d + k];
  }
  return cloned;
}

}  // namespace fg

#endif  // FRACTAL_GAS_TENSOR_OPS_HPP
