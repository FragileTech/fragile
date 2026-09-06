// Translation of WalkerState (src/fragile/fractalai/fractal_gas.py) as a
// structure-of-arrays. Environment states are opaque per-walker byte blobs
// (emulator dump + reward-carry tail). `infos` from the Python version is
// dropped — only metrics/frames surface to the caller.
#ifndef FRACTAL_GAS_WALKER_STATE_HPP
#define FRACTAL_GAS_WALKER_STATE_HPP

#include <cstdint>
#include <vector>

#include "env.hpp"

namespace fg {

struct WalkerState {
  int32_t N = 0;
  int32_t obs_dim = 0;

  std::vector<std::vector<char>> states;  // [N] opaque env-state blobs
  std::vector<float> observations;        // [N * obs_dim], row-major
  std::vector<float> rewards;             // [N] cumulative
  std::vector<float> step_rewards;        // [N]
  std::vector<uint8_t> dones;             // [N] bool
  std::vector<uint8_t> truncated;         // [N] bool
  std::vector<int32_t> actions;           // [N]
  std::vector<int32_t> dt;                // [N]
  std::vector<float> virtual_rewards;     // [N], valid iff has_virtual_rewards
  bool has_virtual_rewards = false;
  // Per-walker game info copied from the env after each step (valid iff
  // has_infos). Travels with the walker through clone/inject/extract, so
  // it stays walker-indexed even after elite injection.
  std::vector<WalkerInfo> infos;          // [N]
  bool has_infos = false;
  // Optional exploration IDs follow the same gather/inject semantics as state.
  std::vector<uint32_t> lineage;

  bool alive(int32_t i) const {
    return !(dones[static_cast<size_t>(i)] || truncated[static_cast<size_t>(i)]);
  }
  std::vector<uint8_t> alive_mask() const;
  int32_t alive_count() const;

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
  static WalkerState extract(const WalkerState& s,
                             const std::vector<int32_t>& indices);

  /// Python FractalGas._concat_walkers().
  static WalkerState concat(const WalkerState& a, const WalkerState& b);
};

}  // namespace fg

#endif  // FRACTAL_GAS_WALKER_STATE_HPP
