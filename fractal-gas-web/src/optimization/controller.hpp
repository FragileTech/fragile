#pragma once
#include "optimization/basin.hpp"
#include "optimization/engine.hpp"
namespace fg::optimization {
struct RoundChoice {
  Json config;
  std::string regime;
  OptimizationRng random;
  explicit RoundChoice(const Json& c,const OptimizationRng& r):config(c),random(r) {}
};
// Scheduling owns no optimizer. Session alone replaces adapters at safe boundaries.
class RunController {
 public:
  RunController(Benchmark&,const Settings&);
  void configure(const Settings&,const Settings& previous);
  void observe(const double*,double);
  bool wants_restart(const Settings&,bool finished,bool alive) const;
  RoundChoice next(const Settings&) const;
  void finish(const Settings&,const Json& geometry,const Json& refinements);
  void begin(const Settings&,const RoundChoice&,uint64_t start);
  Json status(const Settings&) const;
  uint64_t round_id() const { return round; }
  uint64_t round_steps=0;
  BasinArchive archive;
  OptimizationRng placement;
 private:
  Benchmark& benchmark;
  OptimizationRng scheduling;
  bool enabled=false,requested=false;
  int baseline_population=0,exploration_population=0;
  double baseline_scale=1,round_best;
  uint64_t experiment_seed=0,repeated_evaluations=0;
  uint64_t round=0,start_evaluations=0,last_improvement=0,exploration_evaluations=0,focused_evaluations=0;
  std::string regime="initial";
  mutable std::string reason;
  std::vector<BasinEntry> candidates;
};
}  // namespace fg::optimization
