#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fg::fractal {
// Conditional on the already sampled donor graph. Coupled selection uses
// realized source multiplicities instead of pretending gates are independent.
struct SelectionEvidence {
  // Signed, unclipped comparison used by the adapter before its clone gate.
  std::vector<double> score;
  // Optional population mask: stored Graph ancestors are not moving walkers.
  std::vector<uint8_t> active;
  std::vector<double> fitness, mass, replacement;
  std::vector<int32_t> donors, sources, destinations;
  bool expected = true;
};
inline std::vector<double> expected_clone_mass(const std::vector<int32_t>& donors,
                                              const std::vector<double>& probability) {
  if(donors.size()!=probability.size()) throw std::invalid_argument("Cloning evidence shape");
  std::vector<double> mass(donors.size(),1.);
  for(size_t i=0;i<donors.size();++i) {
    if(donors[i]<0 || size_t(donors[i])>=mass.size() || !std::isfinite(probability[i]) ||
       probability[i]<0 || probability[i]>1) throw std::invalid_argument("Cloning evidence probability");
    mass[i]-=probability[i]; mass[donors[i]]+=probability[i];
  }
  return mass;
}
}
