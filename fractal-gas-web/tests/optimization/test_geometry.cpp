#include "optimization/engine.hpp"
#include "optimization/adaptive.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
TEST_CASE(geometry_collection_does_not_change_native_trajectory) {
  for(const std::string algorithm:{"wave","gas","euclidean","cmaes_active","cmaes_bipop"}) {
    Json c=JsonReader(std::string(R"({"benchmark":"quadratic","dimensions":3,"walkers":16,"max_walkers":16,"seed":31,"gas_local_search":false,"gas_tabu":false,"potential_force":false,"perturbation":"cloning_guided"})")).read();
    c.object["algorithm"].kind=Json::String;c.object["algorithm"].string=algorithm;
    Session plain(c);
    c.object["geometry_diagnostics"].kind=Json::Boolean;c.object["geometry_diagnostics"].number=1;
    Session observed(c);
    for(int i=0;i<8;++i) {
      CHECK(plain.snapshot==observed.snapshot);
      const auto status=JsonReader(observed.status_json()).read();
      CHECK(status["geometry"]["version"].num()==1);
      CHECK(stringify(status["geometry"]).size()<status["geometry_capacity_bytes"].num());
      if(plain.algorithm->finished()) break;
      plain.step();observed.step();
    }
    auto snapshot=observed.snapshot;
    observed.set_geometry_diagnostics(false);CHECK(observed.snapshot==snapshot);
    CHECK(JsonReader(observed.status_json()).read()["geometry"].kind==Json::Null);
  }
}
TEST_CASE(geometry_bipop_capture_tracks_displayed_restart_not_next_restart) {
  auto c=JsonReader(std::string(R"({"algorithm":"cmaes_bipop","benchmark":"quadratic","dimensions":2,"seed":9,"cma_sigma":1e-14,"cma_runs":3,"geometry_diagnostics":true})")).read();
  Session s(c);bool restarted=false;
  for(int i=0;i<600 && !s.algorithm->finished();++i) {
    auto j=JsonReader(s.status_json()).read();
    CHECK(j["geometry"]["restart"].num()==j["restarts"].num());
    CHECK(j["geometry"]["generation"].num()==j["generation"].num());
    restarted |= j["restarts"].num()>0;s.step();
  }
  CHECK(restarted);
}
