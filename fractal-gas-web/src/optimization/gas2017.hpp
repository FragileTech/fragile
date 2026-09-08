#pragma once
#include "optimization/engine.hpp"

namespace fg::optimization::gas2017 {
// Pure GAS operators, separate from the Euclidean Gas fitness and collisions.
std::vector<double> normalize(const Population&, const Settings&);
double clone_probability(double flow, double donor_flow);
double distance2(const float*, const float*, int);
void clone(Population&, const std::vector<double>&, Rng&);
std::vector<double> flows(Population&, const Settings&, const Population*,
                          Rng&);
void propose(const float*, float*, int, const Benchmark&, bool,
             const Perturbation&, double phi, Rng&,
             PerturbationTransition* accepted = nullptr);
struct Candidate {
  std::vector<float> x;
  double value;
};
Candidate local_search(Benchmark&, const Settings&, Candidate);
std::vector<float> centroid(const Population&, const std::vector<double>&);
void insert_memory(Population&, const Settings&, const Candidate&, Rng&);
}  // namespace fg::optimization::gas2017
