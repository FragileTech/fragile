#pragma once
#include <set>
#include "optimization/engine.hpp"
#include "optimization/basin.hpp"

namespace fg::optimization {
Json exchange_frame_json(const fractal::MemberFrame&);
fractal::MemberFrame exchange_frame_read(const Json&);
Json exchange_imports_json(const std::vector<fractal::WalkerImport>&);
std::vector<fractal::WalkerImport> exchange_imports_read(const Json&);
Json packet_json(const fractal::WalkerPacket&,bool payload=true);
// The remote mode owns configuration, budget admission, selection and archives,
// while worker-local Sessions own execution. Native mode owns both.
class PopulationExperiment {
 public:
  explicit PopulationExperiment(const Json&,bool remote=false);
  Json config;
  fractal::PopulationController controller;
  std::vector<std::unique_ptr<Session>> sessions;
  Json request(const Json&);
  void step();
  Json status() const;
 private:
  bool remote_,initialized_=false,failed_=false,round_pending_=false,authorized_=false;
  std::set<std::string> seen_events_;
  uint64_t budget_=0,basin_round_=0;
  size_t concurrency_=1;
  std::unique_ptr<Benchmark> task_;
  std::unique_ptr<BasinArchive> archive_;
  std::unique_ptr<ThreadPool> executor_;
  std::unique_ptr<fractal::PopulationController> pending_controller_;
  std::vector<Json> reports_;
  Json report(size_t);
  void accept_reports(const Json&,bool initial);
  Json finish_round(const Json&);
  uint64_t evaluations() const;
  uint64_t next_cost() const;
  void admit() const;
};
} // namespace fg::optimization
