// Translation of WalkerState (src/fragile/fractalai/fractal_gas.py) as a
// structure-of-arrays. Environment states are opaque per-walker byte blobs
// (emulator dump + reward-carry tail). `infos` from the Python version is
// dropped — only metrics/frames surface to the caller.
#ifndef FRACTAL_GAS_WALKER_STATE_HPP
#define FRACTAL_GAS_WALKER_STATE_HPP

#include <cstdint>
#include <vector>

#include "env.hpp"
#include "fractal/population.hpp"

namespace fg {

struct WalkerState : fractal::Population<std::vector<std::vector<char>>, WalkerInfo, int32_t> {
  /// Python WalkerState.clone(): for every field,
  /// new[i] = old[companions[i]] if will_clone[i] else old[i], with companion
  /// values read from the original arrays (gather, not sequential in-place).
  WalkerState clone(const std::vector<int32_t>& companions,
                    const std::vector<uint8_t>& will_clone) const;

  /// Python WalkerState.inject(): overwrite walkers [0, count) with
  /// source walkers [0, count) (deep copies). virtual_rewards copied only if
  /// present on both, matching the Python guard.
  void inject(const WalkerState& source, int32_t count);

  /// Python FractalGas._extract_walkers().
  static WalkerState extract(const WalkerState& s, const std::vector<int32_t>& indices);

  /// Python FractalGas._concat_walkers().
  static WalkerState concat(const WalkerState& a, const WalkerState& b);
};

}  // namespace fg

#endif  // FRACTAL_GAS_WALKER_STATE_HPP
