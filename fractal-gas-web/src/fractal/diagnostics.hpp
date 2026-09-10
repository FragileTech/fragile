#pragma once
#include <vector>

namespace fg {
// Optional observations of values already computed by the algorithms. Not part
// of algorithm configuration/checkpoints: enabling this never consumes RNG.
struct CloneDecision {
  int slot = 0, distance_companion = 0, clone_donor = 0;
  float distance = 0, distance_norm = 0, reward_norm = 0, other = 1;
  float fitness = 0, donor_fitness = 0, clone_score = 0, draw = 0;
  bool alive = true, normalization_leaf = true, leaf = true;
  bool donor_protected = false, best_protected = false, elite_protected = false;
  bool invalid_donor = false, wanted = false, cloned = false;
};
struct CloneDiagnostics {
  bool enabled = false;
  std::vector<CloneDecision> decisions;
};
}  // namespace fg
