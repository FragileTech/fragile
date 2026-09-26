#pragma once
#include <vector>
#include <cstdint>
#include <tuple>
#include <memory>
#include <string>

#include "fractal/rng.hpp"
#include "fractal/selection_evidence.hpp"
namespace fg::fractal {
// Deterministic lineage identifiers use a separate stream from optimizer RNG.
// Copies retain the identifier; an evaluated descendant derives a new one.
inline uint64_t descendant_lineage(uint64_t parent,uint64_t action) {
  uint64_t value=parent^(action+0x9e3779b97f4a7c15ULL);
  value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
  value=(value^(value>>27))*0x94d049bb133111ebULL;
  return value^(value>>31);
}
enum class ProposalFamily { Local, Broad, Difference };
struct TrialIdentity {
  uint64_t round = 0, model = 0, parent = 0, action = 0;
  uint32_t branch = 0;
  auto key() const { return std::make_tuple(round, model, parent, action, branch); }
  bool operator<(const TrialIdentity& other) const { return key() < other.key(); }
};
struct FrozenPopulation {
  int dimensions = 0;
  std::vector<float> positions;
  std::vector<uint64_t> families;
  std::vector<uint8_t> valid;
  std::vector<double> objectives;
};
struct ProposalTrial {
  TrialIdentity identity;
  ProposalFamily family = ProposalFamily::Local;
  std::vector<float> origin;
  std::vector<double> direction;
  double scale = 1;
  bool paired = false;
};
struct TrialOutcome {
  ProposalTrial trial;
  std::vector<double> displacement;
  double before = 0, after = 0, improvement = 0, standard_error = 0;
  uint64_t evaluations = 1;
  bool valid = true, boundary_handled = false, replay = false;
  int draws = 1;
};
struct PerturbationContext {
  double normalized_objective;
};
// A strategy draws one vector. The environment adds it to positions; BAOAB
// scales it by the existing thermal-noise coefficient during its O stage.
// Position and benchmark access permit future state-dependent proposals.
struct PerturbationTransition {
  std::vector<float> origin;
  std::vector<double> displacement;
  int draws = 0;
  double improvement = 0;
  double scale = 1;  // Accepted retry scale, independent of covariance shape.
  std::vector<double> direction;
  uint64_t parent = 0, action = 0;
  double source_scale = 1;
};
// Read-only evidence sink: never receives an RNG or objective evaluator.
class ProposalObserver {
 public:
  virtual ~ProposalObserver() = default;
  virtual void begin() {}
  virtual void freeze(const FrozenPopulation&) {}
  virtual void transition(const PerturbationTransition&) {}
  virtual void outcome(const TrialOutcome&) {}
  virtual void cloning(const FrozenPopulation&, const SelectionEvidence&) {}
  virtual void movement(const float*, const float*, int, uint64_t, const std::string&) {}
  virtual void reference(const float*,int,double) {}
  virtual void update() {}
};
class Perturbation {
 public:
  virtual ~Perturbation() = default;
  std::shared_ptr<ProposalObserver> observer;
  void observed_transition(const PerturbationTransition& t) {
    observe(t); if(observer) observer->transition(t);
  }
  void observed_feedback(const TrialOutcome& t) {
    feedback(t); if(observer && !t.replay) observer->outcome(t);
  }
  void observed_freeze(const FrozenPopulation& p) {
    freeze_population(p); if(observer) observer->freeze(p);
  }
  void observed_update() { update(); if(observer) observer->update(); }

  // Observation may collect data, but only update() changes sampling geometry.
  virtual void observe(const PerturbationTransition&) {}
  virtual void update() {}
  virtual void reset() {}
  // Optional evaluated-trial protocol. Legacy callers and strategies keep their
  // original sampling path and consume exactly the same random draws.
  virtual bool tracks_trials() const { return false; }
  virtual void freeze_population(const FrozenPopulation&) {}
  virtual bool uses_cloning_evidence() const { return false; }
  virtual void observe_cloning(const FrozenPopulation&, const SelectionEvidence&) {}
  virtual ProposalTrial propose(const float* position, float* delta, int dimensions,
                                Rng& rng, TrialIdentity identity) const {
    sample(position, delta, dimensions, rng);
    ProposalTrial trial;
    trial.identity = identity;
    trial.origin.assign(position, position + dimensions);
    return trial;
  }
  virtual void feedback(const TrialOutcome&) {}
  // Original objective at the exact origin, compared against frozen context.
  virtual ProposalTrial propose_with_objective(const float* position, float* delta,
      int dimensions, Rng& rng, TrialIdentity identity, double) const {
    return propose(position, delta, dimensions, rng, identity);
  }
  // Deterministic sequence extension within one trial. Adapters may request
  // several independent kicks while retaining the selected proposal family.
  virtual void continue_trial(const ProposalTrial& trial, float* delta, int dimensions,
                              Rng& rng) const {
    sample(trial.origin.data(), delta, dimensions, rng);
  }
  virtual void sample(const float* position, float* delta, int dimensions, Rng& rng) const = 0;
  // The first position is fixed at the action origin; the second is current.
  virtual void sample_action(const float*, const float* position, float* delta, int dimensions,
                             Rng& rng) const {
    sample(position, delta, dimensions, rng);
  }
  virtual void sample_with_context(const float* position, float* delta, int dimensions, Rng& rng,
                                   const PerturbationContext*) const {
    sample(position, delta, dimensions, rng);
  }
};
}  // namespace fg::fractal
