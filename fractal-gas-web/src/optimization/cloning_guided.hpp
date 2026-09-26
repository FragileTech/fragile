#pragma once
#include "optimization/perturbation.hpp"
namespace fg::optimization {
class CloningGuided final : public Perturbation {
 public:
  CloningGuided(const Benchmark&,const Json&);
  CloningGuided(const CloningGuided&);
  ~CloningGuided();
  void configure(const Json&);
  bool tracks_trials() const override {return true;}
  bool uses_cloning_evidence() const override {return true;}
  void freeze_population(const fractal::FrozenPopulation&) override;
  void observe_cloning(const fractal::FrozenPopulation&,const fractal::SelectionEvidence&) override;
  void update() override;
  void reset() override;
  void sample(const float*,float*,int,Rng&) const override;
  fractal::ProposalTrial propose_with_objective(const float*,float*,int,Rng&,fractal::TrialIdentity,double) const override;
  void continue_trial(const fractal::ProposalTrial&,float*,int,Rng&) const override;
  Json diagnostics() const;
  Json geometry() const;
  Json visual_geometry() const;
  double visual_scale(double objective) const;
  void restore_geometry(const Json&);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}
