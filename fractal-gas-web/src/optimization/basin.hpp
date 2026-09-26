#pragma once
#include "optimization/benchmark.hpp"

namespace fg::optimization {
struct BasinEntry {
  std::vector<double> position;
  double objective=0, uncertainty=0, radius=1e-4, confidence=0;
  uint64_t id=0, discovery_round=0, last_round=0, visits=1, cost=0, refinement_cost=0;
  bool validated=true, refined=false;
  Json settings, geometry;
};
// Archive evidence is deliberately separate from optimizer fitness and memory.
class BasinArchive {
 public:
  explicit BasinArchive(const Benchmark&, bool periodic, size_t capacity=64);
  const std::vector<BasinEntry>& entries() const { return records; }
  double distance(const std::vector<double>&,const std::vector<double>&) const;
  void complete_round(std::vector<BasinEntry> candidates,uint64_t round,bool stalled);
  void place(float*,Rng&,int refinement=-1,bool avoidance=true) const;
  void set_periodic(bool);
  std::string export_json() const;
  void import_json(const std::string&);
  Json summary() const;
  // Trusted evidence exchanged within one active population, already evaluated.
  void synchronize_json(const std::string&);
  void merge_event(const Json&, uint64_t global_round);

  void validate_imports(Benchmark&,Rng&,uint64_t budget);
  uint64_t validation_cost(bool stochastic) const;
 private:
  int dimensions;
  double low,high;
  bool periodic,minimize;
  size_t capacity;
  uint64_t next_id=1;
  Json compatibility;
  std::vector<BasinEntry> records;
  bool better(double a,double b) const { return minimize?a<b:a>b; }
  double penalty(const std::vector<double>&,int ignored) const;
};
}  // namespace fg::optimization
