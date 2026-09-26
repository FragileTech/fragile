#include "optimization/controller.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
static Json parse(const std::string& text) {return JsonReader(text).read();}
TEST_CASE(controller_adaptive_rounds_respect_user_scale_bounds_and_pins) {
  Session session(parse(R"({"algorithm":"wave","benchmark":"quadratic","walkers":8,"max_walkers":32,"max_evaluations":10000,"controller_enabled":true,"perturbation":"adaptive_fractal","adaptive_min_scale":0.01,"adaptive_max_scale":0.1})"));
  session.step();
  session.update_settings(parse(R"({"restart_token":1})"));session.step();
  auto status=parse(session.status_json());
  CHECK(status["controller"]["regime"].str()=="focused");
  CHECK(session.settings.json["adaptive_round_fraction"].num()<1);
  CHECK(status["exploration"]["round_scale_max"].num()>=.01);
  CHECK(status["exploration"]["round_scale_max"].num()<=.1);
  session.update_settings(parse(R"({"adaptive_min_scale":0.02,"adaptive_max_scale":0.04})"));
  CHECK(!session.settings.json["scale_auto"].flag());
  CHECK(session.settings.json["adaptive_round_fraction"].num()==1);
  for(int round=2;round<5;++round) {
    Json patch;patch.kind=Json::Object;patch.object["restart_token"]=number(round);
    session.update_settings(patch);session.step();
    CHECK(session.settings.json["adaptive_min_scale"].num()==.02);
    CHECK(session.settings.json["adaptive_max_scale"].num()==.04);
    CHECK(session.settings.json["adaptive_round_fraction"].num()==1);
  }
  session.update_settings(parse(R"({"scale_auto":true})"));
  CHECK(session.settings.json["scale_auto"].flag());
}
TEST_CASE(controller_manual_restart_preserves_global_accounting_and_pins) {
  for(auto name:{"wave","graph","fmc","wave_jump","gas","euclidean"}) {
    auto c=parse(R"({"benchmark":"quadratic","dimensions":2,"walkers":8,"max_walkers":32,"periodic":true,"horizon":2,"gas_local_search":false,"potential_force":false,"max_evaluations":10000,"controller_enabled":true})");
    c.object["algorithm"].kind=Json::String;c.object["algorithm"].string=name;
    Session a(c),b(c);a.step();b.step();
    const auto best=a.best;const auto evals=a.benchmark.evaluations;
    auto patch=parse(R"({"restart_token":1,"walkers":10,"perturbation_std":0.2})");
    a.update_settings(patch);b.update_settings(patch);
    for(int step=0;step<16 && parse(a.status_json())["controller"]["round"].num()==0;++step) {
      a.step();b.step();CHECK(a.snapshot==b.snapshot);
    }
    CHECK(parse(a.status_json())["controller"]["round"].num()==1);
    CHECK(a.benchmark.evaluations>evals);CHECK(a.best<=best);
    CHECK(a.settings.walkers==10);CHECK(a.settings.json["perturbation_std"].num()==.2);
    CHECK(!a.settings.json["population_auto"].flag());CHECK(!a.settings.json["scale_auto"].flag());
    CHECK(a.settings.json["seed"].num()==7);
    auto data=parse(a.export_basins());CHECK(!data["entries"].array.empty());
  }
}
TEST_CASE(controller_requires_budget_and_restart_budget_is_hard) {
  bool failed=false;
  try {Session s(parse(R"({"algorithm":"wave","controller_enabled":true})"));} catch(const std::exception&) {failed=true;}
  CHECK(failed);
  Session s(parse(R"({"algorithm":"wave","benchmark":"quadratic","walkers":4,"max_walkers":16,"max_evaluations":100,"controller_enabled":true})"));
  s.update_settings(parse(R"({"restart_token":1,"max_evaluations":4})"));
  const auto before=s.snapshot;
  failed=false;try{s.step();}catch(const std::exception&){failed=true;}CHECK(failed);CHECK(s.snapshot==before);
  s.update_settings(parse(R"({"max_evaluations":100})"));s.step();
  CHECK(parse(s.status_json())["controller"]["round"].num()==1);
}
TEST_CASE(basin_archive_compatibility_reference_values_and_bounds) {
  Benchmark b(parse(R"({"benchmark":"quadratic","dimensions":2})"));
  BasinArchive archive(b,false,2);
  BasinEntry e;e.position={0,0};e.objective=0;e.radius=.01;
  archive.complete_round({e},0,true);archive.complete_round({e},1,true);
  CHECK(archive.entries()[0].visits==2);CHECK(archive.entries()[0].confidence>0);
  auto exported=archive.export_json();BasinArchive imported(b,false,2);imported.import_json(exported);
  CHECK(!imported.entries()[0].validated);CHECK(b.evaluations==0);
  OptimizationRng rng(1);imported.validate_imports(b,rng,10);CHECK(b.evaluations==1);CHECK(imported.entries()[0].validated);
  Benchmark incompatible(parse(R"({"benchmark":"quadratic","dimensions":3})"));BasinArchive other(incompatible,false);
  bool failed=false;try{other.import_json(exported);}catch(const std::exception&){failed=true;}CHECK(failed);
  imported.set_periodic(true);CHECK(imported.entries()[0].confidence==0);
  float x[2];for(int k=0;k<100;++k){imported.place(x,rng);CHECK(b.valid(x));}
}
TEST_CASE(controller_automatic_rounds_use_complete_operations_and_reproduce) {
  auto c=parse(R"({"algorithm":"wave","benchmark":"constant","dimensions":1,"walkers":4,"max_walkers":16,"max_evaluations":500,"controller_enabled":true,"periodic":true})");
  Session a(c),b(c);
  uint64_t rounds=0;
  for(int i=0;i<100;++i) {
    auto status=parse(a.status_json());if(status["budget_exhausted"].flag()) break;
    auto before=a.benchmark.evaluations;auto bound=a.next_evaluations();a.step();b.step();
    CHECK(a.snapshot==b.snapshot);CHECK(a.benchmark.evaluations-before<=bound);
    CHECK(a.benchmark.evaluations<=500);
    rounds=uint64_t(parse(a.status_json())["controller"]["round"].num());
  }
  CHECK(rounds>=2);
}
TEST_CASE(controller_stochastic_restart_validation_is_admitted) {
  auto c=parse(R"({"algorithm":"wave","benchmark":"stochastic_gaussian","dimensions":2,"walkers":8,"max_walkers":16,"max_evaluations":1000,"controller_enabled":true,"periodic":true,"perturbation":"adaptive_fractal"})");
  Session session(c);session.step();
  session.update_settings(parse(R"({"restart_token":1})"));
  auto before=session.benchmark.evaluations;auto bound=session.next_evaluations();session.step();
  CHECK(session.benchmark.evaluations-before<=bound);
  auto data=parse(session.export_basins());CHECK(!data["entries"].array.empty());
  for(const auto& entry:data["entries"].array) CHECK(entry["uncertainty"].num()>=0);
}
