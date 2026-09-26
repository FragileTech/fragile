#include "optimization/engine.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
static Json boundary_config(const std::string& text) { return JsonReader(text).read(); }

TEST_CASE(boundary_mapping_matches_libcmaes_known_values_and_is_idempotent) {
  Benchmark b(boundary_config(R"({"benchmark":"quadratic","dimensions":2,"low":-5,"high":5})"));
  float x[] = {6, -6};
  b.boundary(x, "cma");
  CHECK_CLOSE(x[0], 4.6, 1e-6); CHECK_CLOSE(x[1], -4.6, 1e-6);
  x[0]=5.2f; x[1]=-5.2f;
  b.boundary(x, "cma");
  CHECK_CLOSE(x[0], 5-.01/1.2, 1e-6); CHECK_CLOSE(x[1], -5+.01/1.2, 1e-6);
  const float saved=x[0]; b.boundary(x,"cma"); CHECK(x[0]==saved);
  x[0]=1e30f; x[1]=-1e30f; b.boundary(x,"cma"); CHECK(b.valid(x));
  CHECK(b.evaluations==0);
  x[0]=INFINITY; b.boundary(x,"cma"); CHECK(!b.valid(x));
  x[0]=6; x[1]=-6; b.boundary(x,"none"); CHECK(x[0]==6 && x[1]==-6);
  b.boundary(x,"periodic"); CHECK(x[0]==-4 && x[1]==4);
}

TEST_CASE(boundary_modes_legacy_settings_and_atomic_live_updates) {
  Settings legacy(boundary_config(R"({"algorithm":"wave","periodic":true})"));
  CHECK(legacy.periodic); CHECK(legacy.json["boundary"].str()=="periodic");
  Session s(boundary_config(R"({"algorithm":"wave","dimensions":20,"walkers":8,"elites":5})"));
  auto before=s.snapshot;
  s.update_settings(boundary_config(R"({"boundary":"cma"})"));
  CHECK(s.snapshot==before); CHECK(!s.settings.periodic);
  CHECK(s.settings.json["boundary"].str()=="cma");
  for (auto patch : {R"({"boundary":"unknown"})", R"({"boundary":"cma","periodic":true})"}) {
    bool failed=false; try { s.update_settings(boundary_config(patch)); } catch (...) { failed=true; }
    CHECK(failed); CHECK(s.snapshot==before); CHECK(s.settings.json["boundary"].str()=="cma");
  }
  s.update_settings(boundary_config(R"({"periodic":true})"));
  CHECK(s.settings.periodic); CHECK(s.settings.json["boundary"].str()=="periodic");
  s.update_settings(boundary_config(R"({"boundary":"none"})"));
  CHECK(!s.settings.periodic); CHECK(s.snapshot==before);
}

TEST_CASE(boundary_mapping_repairs_proposals_across_fractal_adapters) {
  for (auto algorithm : {"wave","graph","fmc","wave_jump","gas","euclidean"})
    for (auto strategy : {"gaussian","cloning_guided"}) {
      auto c=boundary_config(R"({"benchmark":"quadratic","dimensions":20,"walkers":8,
        "max_walkers":128,"elites":5,"horizon":2,"seed":7,"boundary":"cma",
        "perturbation_std":1000,"adaptive_min_scale":1000,"adaptive_max_scale":1000,
        "potential_force":false,"gas_local_search":false})");
      c.object["algorithm"].kind=Json::String; c.object["algorithm"].string=algorithm;
      c.object["perturbation"].kind=Json::String; c.object["perturbation"].string=strategy;
      Session s(c);
      for (int step=0;step<6;++step) {
        const auto before=s.benchmark.evaluations, bound=s.next_evaluations();
        s.step();
        CHECK(s.benchmark.evaluations-before<=bound);
        const auto& p=s.algorithm->population();
        for (int i=0;i<p.n;++i) { if (s.settings.algorithm!="graph") CHECK(p.alive[i]); CHECK(s.benchmark.valid(p.x.data()+i*p.d)); }
      }
    }
}

TEST_CASE(boundary_changes_wait_for_committed_planner_actions) {
  for (auto algorithm : {"fmc","wave_jump"}) {
    auto c=boundary_config(R"({"dimensions":20,"walkers":8,"elites":5,"horizon":3,
      "perturbation":"cloning_guided","boundary":"none","adaptive_max_scale":0.01})");
    c.object["algorithm"].kind=Json::String; c.object["algorithm"].string=algorithm;
    Session s(c); s.step();
    s.update_settings(boundary_config(R"({"boundary":"cma"})"));
    CHECK(boundary_config(s.status_json())["pending_settings"]["boundary"].str()=="cma");
    CHECK(s.settings.json["boundary"].str()=="none");
    for (int i=0;i<20 && s.settings.json["boundary"].str()!="cma";++i) s.step();
    CHECK(s.settings.json["boundary"].str()=="cma");
  }
}
