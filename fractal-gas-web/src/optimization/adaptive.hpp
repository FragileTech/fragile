#pragma once
#include "optimization/perturbation.hpp"
#include <functional>

namespace fg::optimization {
struct EvaluatedProposal {
  std::vector<float> position, velocity;
  uint32_t branch = 0;
  double objective = 0;
  bool valid = false;
};
// A movement kernel returns a complete candidate state without evaluating its
// objective. Both branches receive the same frozen draws with opposite signs.
using MovementKernel = std::function<EvaluatedProposal(const std::vector<float>&, int)>;
EvaluatedProposal evaluate_adaptive_trial(Perturbation&, const Benchmark&, const Json&,
    const float*, double, Rng&, fractal::TrialIdentity, int draws,
    const MovementKernel&, bool replay = false);
void validate_adaptive_geometry(const Json&,int dimensions);
uint64_t adaptive_evaluation_bound(const Benchmark&, const Json&, bool baseline = false);
EvaluatedProposal evaluate_adaptive_position(Perturbation&, const Benchmark&, const Json&,
    const float*, double, Rng&, fractal::TrialIdentity, bool replay = false, int draws = 1);
// Experimental rank adaptation. This is not a CMA-ES implementation.
class AdaptiveExploration final : public Perturbation {
 public:
  AdaptiveExploration(const Benchmark&, const Json&);
  AdaptiveExploration(const AdaptiveExploration&);
  ~AdaptiveExploration();
  void configure(const Json&);
  bool tracks_trials() const override { return true; }
  void freeze_population(const fractal::FrozenPopulation&) override;
  fractal::ProposalTrial propose(const float*, float*, int, Rng&,
                                fractal::TrialIdentity) const override;
  fractal::ProposalTrial propose_with_objective(const float*, float*, int, Rng&,
                                fractal::TrialIdentity, double) const override;
  void feedback(const fractal::TrialOutcome&) override;
  void observe_external(fractal::TrialOutcome);
  void continue_trial(const fractal::ProposalTrial&, float*, int, Rng&) const override;
  void sample(const float*, float*, int, Rng&) const override;
  void update() override;
  void reset() override;
  Json diagnostics() const;
  Json geometry() const;
  Json visual_geometry() const;
  double visual_scale(double objective) const;
  void restore_geometry(const Json&);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}  // namespace fg::optimization
