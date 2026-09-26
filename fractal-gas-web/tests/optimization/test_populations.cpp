#include "optimization/populations.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
namespace {
Json parse(const std::string& s){return JsonReader(s).read();}
Json config(int n,int slots=1){
  auto c=parse(R"({"seed":19,"max_evaluations":100000,"exchange_every":1,"global_elites":20,"defaults":{"algorithm":"wave","benchmark":"quadratic","dimensions":2,"walkers":20,"max_walkers":40,"elites":5,"boundary":"periodic","perturbation_std":0.03},"members":[]})");
  c.object["concurrency"]=number(slots);
  for(int i=0;i<n;++i){auto m=parse(R"({"id":"","settings":{}})");m.object["id"].string="swarm-"+std::to_string(i);
    m.object["settings"].object["walkers"]=number(20+i);
    m.object["settings"].object["perturbation_std"]=number(.01*(i+1));
    c.object["members"].array.push_back(m);
  }
  return c;
}
template<class F>bool throws(F f){try{f();return false;}catch(const std::exception&){return true;}}
}
TEST_CASE(populations_configurable_members_and_parallel_reproducibility){
  for(int n:{1,2,4,7}){
    PopulationExperiment serial(config(n)),parallel(config(n,std::min(n,3)));
    for(int k=0;k<3;++k){serial.step();parallel.step();
      CHECK(stringify(serial.status())==stringify(parallel.status()));
      for(int i=0;i<n;++i)CHECK(serial.sessions[i]->snapshot==parallel.sessions[i]->snapshot);
    }
    CHECK(serial.controller.rounds==3);
    CHECK(serial.controller.pool.size()==size_t(n*5));
    CHECK(serial.controller.elites.size()<=20);
    std::set<int> seeds;
    for(int i=0;i<n;++i){
      CHECK(serial.sessions[i]->settings.walkers==20+i);
      CHECK_CLOSE(serial.sessions[i]->settings.json["perturbation_std"].num(),.01*(i+1),1e-12);
      seeds.insert(serial.sessions[i]->settings.seed);
      const auto& e=serial.controller.last_exchange[i];
      CHECK(e.imports.size()==size_t(n==1?0:5));
      std::set<std::pair<std::string,int>> donors;
      for(const auto& imp:e.imports){CHECK(imp.destination>=5);CHECK(imp.walker.source!=e.id);donors.insert({imp.walker.source,imp.walker.row});}
      CHECK(donors.size()==e.imports.size());
    }
    CHECK(seeds.size()==size_t(n));
  }
}
TEST_CASE(populations_budget_and_interval){
  auto c=config(4);c.object["exchange_every"]=number(3);
  PopulationExperiment p(c);p.step();p.step();CHECK(p.controller.exchanges==0);p.step();CHECK(p.controller.exchanges==1);
  auto cost=p.status()["evaluations"].num();CHECK(cost<=100000);
  c=config(4);c.object["max_evaluations"]=number(86);PopulationExperiment limited(c);
  CHECK(throws([&]{limited.step();}));CHECK(limited.controller.rounds==0);CHECK(limited.status()["evaluations"].num()==86);
  c.object["max_evaluations"]=number(85);CHECK(throws([&]{PopulationExperiment invalid(c);}));
}
TEST_CASE(populations_compatibility_and_unequal_counts){
  auto c=config(2);c.object["members"].array[1].object["settings"].object["dimensions"]=number(3);
  CHECK(throws([&]{PopulationExperiment invalid(c);}));
  c=config(2);c.object["members"].array[1].object["settings"].object["elites"]=number(2);
  PopulationExperiment p(c);p.step();CHECK(p.controller.last_exchange[0].shortfall==5);
  CHECK(p.controller.last_exchange[0].imports.empty());CHECK(p.controller.last_exchange[1].imports.size()==2);
  c=config(2);c.object["members"].array[1].object["id"]=c["members"].array[0]["id"];
  CHECK(throws([&]{PopulationExperiment invalid(c);}));
}
TEST_CASE(populations_shared_basin_restarts_are_counted_once){
  auto c=config(2);c.object["defaults"].object["controller_enabled"]=parse("true");
  PopulationExperiment p(c);
  for(auto& s:p.sessions)s->update_settings(parse(R"({"restart_token":1})"));
  p.step();CHECK(!p.status()["basins"].array.empty());
  auto before=p.status()["basins"];p.step();CHECK(stringify(before)==stringify(p.status()["basins"]));
  CHECK(p.status()["evaluations"].num()<=100000);
}

TEST_CASE(populations_native_live_settings_and_single_swarm_without_import_slots){
  auto c=config(1);c.object["defaults"].object["elites"]=number(20);
  PopulationExperiment single(c);single.step();CHECK(single.controller.last_exchange[0].imports.empty());
  PopulationExperiment p(config(2));
  auto request=parse(R"({"op":"settings","member":1,"patch":{"perturbation_std":0.25}})");
  p.request(request);CHECK_CLOSE(p.sessions[1]->settings.json["perturbation_std"].num(),.25,1e-12);
  CHECK_CLOSE(p.sessions[0]->settings.json["perturbation_std"].num(),.01,1e-12);
  request.object["patch"]=parse(R"({"walkers":5})");CHECK(throws([&]{p.request(request);}));
  CHECK(p.sessions[1]->settings.walkers==21);
}
TEST_CASE(populations_shared_basin_events_are_idempotent){
  auto c=config(2);c.object["defaults"].object["controller_enabled"]=parse("true");
  PopulationExperiment coordinator(c,true);
  std::vector<std::unique_ptr<Session>> members;
  auto reports=parse("[]");
  for(const auto& member:coordinator.config["members"].array){
    members.push_back(std::make_unique<Session>(member["settings"]));
    auto& s=*members.back();auto adapter=s.exchange_member(member["id"].str(),5);
    s.update_settings(parse(R"({"restart_token":1})"));s.step();
    adapter=s.exchange_member(member["id"].str(),5);
    auto report=parse("{}");report.object["id"]=member["id"];report.object["settings"]=s.settings.json;
    report.object["evaluations"]=number(s.benchmark.evaluations);report.object["next_evaluations"]=number(s.next_evaluations());
    report.object["events"]=s.take_basin_events();report.object["frame"]=exchange_frame_json(coordinator.controller.capture(*adapter));
    reports.array.push_back(report);
  }
  auto request=parse(R"({"op":"initialize"})");request.object["reports"]=reports;
  coordinator.request(request);auto before=stringify(coordinator.status()["basins"]);
  request.object["op"].string="refresh";coordinator.request(request);
  CHECK(before==stringify(coordinator.status()["basins"]));
}
