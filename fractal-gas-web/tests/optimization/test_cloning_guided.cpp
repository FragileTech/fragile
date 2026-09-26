#include "optimization/cloning_guided.hpp"
#include "optimization/adaptive.hpp"
#include "optimization/engine.hpp"
#include "test_framework.hpp"
#include <numeric>
using namespace fg::optimization;
using namespace fg::fractal;
static Json cg_config(const std::string& text) {return JsonReader(text).read();}
static FrozenPopulation cloud(int d,int n=80) {
  FrozenPopulation p;p.dimensions=d;
  for(int i=0;i<n;++i) {
    p.valid.push_back(1);p.families.push_back(i);p.objectives.push_back(i);
    for(int j=0;j<d;++j) p.positions.push_back(float(.03*std::sin(double((i+1)*(j+1))) * (j==0?3:1)));
  }return p;
}
static SelectionEvidence evidence(const FrozenPopulation& p) {
  SelectionEvidence e;e.mass.assign(p.valid.size(),1);
  for(size_t i=0;i<p.valid.size();++i) e.fitness.push_back(std::exp(4*p.positions[i*p.dimensions]));
  for(size_t i=0;i<p.valid.size();++i) {
    e.donors.push_back((i+1)%p.valid.size());
    e.score.push_back((e.fitness[(i+1)%p.valid.size()]-e.fitness[i])/e.fitness[i]);
  }
  return e;
}
TEST_CASE(wave_retains_five_valid_elites_when_all_movement_trials_leave_bounds) {
  for (const auto* strategy : {"gaussian", "cloning_guided"}) {
    auto c = cg_config(R"({"algorithm":"wave","benchmark":"quadratic","dimensions":20,
      "walkers":8,"max_walkers":8,"elites":5,"seed":7,"periodic":false,
      "perturbation_std":1000000,"adaptive_min_scale":1000000,"adaptive_max_scale":1000000})");
    c.object["perturbation"].kind = Json::String;
    c.object["perturbation"].string = strategy;
    Session session(c);
    const auto initial = session.algorithm->population();
    const double best = session.best;
    for (int step = 0; step < 3; ++step) {
      const auto before = session.benchmark.evaluations;
      session.step();
      const auto& p = session.algorithm->population();
      CHECK(session.benchmark.evaluations - before == 8);
      CHECK(session.iteration == uint64_t(step + 1));
      CHECK(session.best == best);
      CHECK(session.snapshot[6] == 5);
      for (int i = 0; i < 5; ++i) {
        CHECK(p.alive[i]);
        CHECK(p.objective[i] == initial.objective[i]);
        CHECK(p.lineage[i] == initial.lineage[i]);
        for (int k = 0; k < p.d; ++k) CHECK(p.x[i * p.d + k] == initial.x[i * p.d + k]);
      }
      for (int i = 5; i < 8; ++i) CHECK(!p.alive[i]);
    }
  }
}
TEST_CASE(cloning_expected_mass_matches_enumerated_gates) {
  std::vector<int32_t> donors{1,2,0};std::vector<double> q{.25,.5,0};
  auto mass=expected_clone_mass(donors,q);std::vector<double> empirical(3,0);
  for(int mask=0;mask<8;++mask) {
    double chance=1;for(int i=0;i<3;++i) chance*=mask&(1<<i)?q[i]:1-q[i];
    for(int i=0;i<3;++i) empirical[mask&(1<<i)?donors[i]:i]+=chance;
  }
  for(int i=0;i<3;++i) CHECK_CLOSE(mass[i],empirical[i],1e-12);
  CHECK_CLOSE(std::accumulate(mass.begin(),mass.end(),0.),3,1e-12);
  CHECK(expected_clone_mass({0},{1})[0]==1);
  bool rejected=false;try {expected_clone_mass({2},{.5});}catch(...) {rejected=true;}CHECK(rejected);
}
TEST_CASE(cloning_geometry_drift_and_lineage_invariance) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":2})"));
  auto c=cg_config(R"({"adaptive_min_scale":1,"adaptive_max_scale":1})");
  CloningGuided model(b,c);auto p=cloud(2);auto e=evidence(p);
  model.freeze_population(p);model.observe_cloning(p,e);model.update();
  CHECK(model.diagnostics()["model_count"].num()==1);
  CHECK(model.diagnostics()["selection_comparisons"].num()==80);
  auto geometry=model.geometry();const auto& shape=geometry.array[0]["shape"].array;
  CHECK(shape[0].num()>shape[3].num()*2);
  CHECK_CLOSE(shape[0].num()+shape[3].num(),2,1e-8);
  auto no_drift=c;
  // Json bool representation is numeric through the existing parser.
  no_drift=cg_config(R"({"adaptive_min_scale":1,"adaptive_max_scale":1,"cloning_drift":false})");
  CloningGuided reference(b,no_drift);reference.observe_cloning(p,e);reference.update();
  OptimizationRng ra(4),rb(4);float origin[2]={0,0},a[2],z[2];
  model.sample(origin,a,2,ra);reference.sample(origin,z,2,rb);
  CHECK(a[0]>z[0]);CHECK(std::abs(a[1]-z[1])<.02);
  CHECK(model.diagnostics()["drift_noise_ratio"].num()>0);
  CloningGuided restored(b,c);restored.restore_geometry(geometry);
  CHECK(stringify(restored.geometry())==stringify(geometry));
  for(auto& f:p.families) f=1;
  model.observe_cloning(p,e);model.update();
  CHECK(model.diagnostics()["selection_comparisons"].num()==80);
  CHECK(stringify(model.geometry())==stringify(geometry));
  CHECK(model.diagnostics()["drift_noise_ratio"].num()>0);
  for(size_t i=0;i<p.families.size();++i) p.families[i]=i;
  e.score.assign(e.score.size(),0);model.observe_cloning(p,e);model.update();
  CHECK(model.diagnostics()["drift_noise_ratio"].num()==0);
}
TEST_CASE(cloning_high_dimension_geometry_and_periodic_reset) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":80})"));
  CloningGuided model(b,cg_config("{}"));auto p=cloud(80,100);model.observe_cloning(p,evidence(p));model.update();
  auto geometry=model.geometry();CHECK(!geometry.array.empty());
  CHECK(geometry.array[0]["columns"].num()<=9);
  validate_adaptive_geometry(geometry.array[0],80);
  CHECK(model.diagnostics()["condition_number"].num()<=1000001);
  OptimizationRng rng(2);std::vector<float> x(80),delta(80);model.sample(x.data(),delta.data(),80,rng);
  for(float v:delta) CHECK(std::isfinite(v));
  model.configure(cg_config(R"({"periodic":true})"));CHECK(model.geometry().array.empty());
}
TEST_CASE(cloning_all_adapters_live_settings_and_budget_accounting) {
  for(auto algorithm:{"wave","graph","fmc","wave_jump","gas","euclidean"}) {
    auto c=cg_config(R"({"benchmark":"quadratic","dimensions":20,"walkers":32,"max_walkers":64,"elites":5,"periodic":true,"horizon":2,"gas_local_search":false,"potential_force":false,"perturbation":"cloning_guided","adaptive_min_scale":0.001,"adaptive_max_scale":0.01})");
    c.object["algorithm"].kind=Json::String;c.object["algorithm"].string=algorithm;
    Session a(c),b(c);
    for(int i=0;i<8;++i) {
      auto before=a.benchmark.evaluations,bound=a.algorithm->next_evaluations_upper_bound();
      a.step();b.step();CHECK(a.snapshot==b.snapshot);CHECK(a.benchmark.evaluations-before<=bound);
    }
    auto status=cg_config(a.status_json());CHECK(status["exploration"]["strategy"].str()=="cloning_guided");
    CHECK(status["exploration"]["model_count"].num()>0);
    auto snapshot=a.snapshot;auto count=a.benchmark.evaluations;
    bool failed=false;try {a.update_settings(cg_config(R"({"cloning_drift_strength":2})"));}catch(...) {failed=true;}
    CHECK(failed);CHECK(a.snapshot==snapshot);CHECK(a.benchmark.evaluations==count);
    a.update_settings(cg_config(R"({"cloning_drift_strength":0.1,"adaptive_min_scale":0.002,"adaptive_max_scale":0.02})"));
    CHECK(a.snapshot==snapshot);CHECK(a.benchmark.evaluations==count);
    for(int i=0;i<10;++i) a.step();
    CHECK(a.settings.json["cloning_drift_strength"].num()==.1);
    if(std::string(algorithm)=="euclidean") {
      a.update_settings(cg_config(R"({"adaptive_euclidean_mode":"position"})"));
      for(float v:a.algorithm->population().v) CHECK(v==0);
      a.step();for(float v:a.algorithm->population().v) CHECK(v==0);
      a.update_settings(cg_config(R"({"adaptive_euclidean_mode":"velocity"})"));a.step();
    }
  }
}
TEST_CASE(cloning_periodic_neighborhood_and_geometry_rotation) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":2,"low":-5,"high":5})"));
  auto p=cloud(2);auto e=evidence(p);
  CloningGuided original(b,cg_config("{}")),rotated(b,cg_config("{}"));
  original.observe_cloning(p,e);original.update();
  const auto shape=original.geometry().array[0]["shape"].array;
  for(size_t i=0;i<p.valid.size();++i) {float x=p.positions[2*i];p.positions[2*i]=-p.positions[2*i+1];p.positions[2*i+1]=x;}
  rotated.observe_cloning(p,e);rotated.update();
  const auto changed=rotated.geometry().array[0]["shape"].array;
  CHECK_CLOSE(shape[0].num(),changed[3].num(),1e-6);
  CHECK_CLOSE(shape[1].num(),-changed[1].num(),1e-6);
  for(size_t i=0;i<p.valid.size();++i) p.positions[2*i]=i%2?4.99f:-4.99f;
  CloningGuided periodic(b,cg_config(R"({"periodic":true})"));periodic.observe_cloning(p,e);periodic.update();
  CHECK(periodic.diagnostics()["model_count"].num()==1);
}
TEST_CASE(cloning_pending_model_noop_updates_and_stochastic_cost) {
  Benchmark b(cg_config(R"({"benchmark":"stochastic_gaussian","dimensions":2})"));
  auto c=cg_config(R"({"perturbation":"cloning_guided","adaptive_min_scale":0.01,"adaptive_max_scale":0.01})");
  CloningGuided model(b,c);auto p=cloud(2);auto e=evidence(p);
  model.observe_cloning(p,e);CHECK(model.geometry().array.empty());model.update();
  CloningGuided reference(model);model.configure(c);
  OptimizationRng a(3),z(3);float x[2]={0,0},u[2],v[2];model.sample(x,u,2,a);reference.sample(x,v,2,z);
  CHECK(u[0]==v[0]);CHECK(u[1]==v[1]);
  const auto before=model.geometry();e.score[0]+=100;
  model.observe_cloning(p,e);CHECK(stringify(before)==stringify(model.geometry()));
  const auto evaluations=b.evaluations;
  evaluate_adaptive_position(model,b,c,x,0,a,{0,0,1,1,0});
  CHECK(b.evaluations-evaluations==6);
  CHECK(adaptive_evaluation_bound(b,c)==6);
}

TEST_CASE(cloning_signed_score_field_without_ancestry_or_mass) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":2,"low":-5,"high":5})"));
  FrozenPopulation p;p.dimensions=2;p.positions={0,0,.1f,0,0,.1f};p.valid={1,1,1};
  // No family IDs, objective values, fitness regression, or offspring mass.
  SelectionEvidence e;e.donors={1,0,0};e.score={1,-1,-1};
  auto c=cg_config(R"({"adaptive_min_scale":1,"adaptive_max_scale":1})");
  auto drift=[&](const SelectionEvidence& evidence) {
    CloningGuided yes(b,c),no(b,cg_config(R"({"adaptive_min_scale":1,"adaptive_max_scale":1,"cloning_drift":false})"));
    yes.observe_cloning(p,evidence);yes.update();no.observe_cloning(p,evidence);no.update();
    OptimizationRng a(7),z(7);float x[2]={0,0},u[2],v[2];yes.sample(x,u,2,a);no.sample(x,v,2,z);
    return std::vector<double>{u[0]-v[0],u[1]-v[1]};
  };
  auto positive=drift(e);CHECK(positive[0]>0);CHECK(positive[1]>0);
  for(auto& score:e.score) score=-score;
  auto negative=drift(e);for(int j=0;j<2;++j) CHECK_CLOSE(negative[j],-positive[j],1e-6);
  e.score={1,-1,-1};auto one=drift(e);e.score[0]=3;auto three=drift(e);
  CHECK(three[0]>one[0]);CHECK(three[1]<one[1]);
  e.score={INFINITY,-INFINITY,-INFINITY};auto extreme=drift(e);
  for(double x:extreme) CHECK(std::isfinite(x));
  CHECK(std::hypot(extreme[0],extreme[1])<=.25*std::sqrt(2.)+1e-6);
  CloningGuided model(b,c);model.observe_cloning(p,e);model.update();auto learned=model.geometry();
  e.score.assign(3,0);model.observe_cloning(p,e);model.update();
  CHECK(stringify(model.geometry())==stringify(learned));
  CHECK(model.diagnostics()["drift_noise_ratio"].num()==0);
  e.donors[0]=3;bool rejected=false;try {model.observe_cloning(p,e);}catch(...) {rejected=true;}CHECK(rejected);
  CHECK(stringify(model.geometry())==stringify(learned));
}
TEST_CASE(cloning_small_population_learns_twenty_dimensional_geometry) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":20})"));
  auto p=cloud(20,8);for(auto& f:p.families) f=1;
  CloningGuided model(b,cg_config("{}"));model.observe_cloning(p,evidence(p));model.update();
  CHECK(model.diagnostics()["condition_number"].num()>1.01);
  CHECK(model.diagnostics()["condition_number"].num()<=1000001);
  auto g=model.geometry();validate_adaptive_geometry(g.array[0],20);
  double trace=0;for(int j=0;j<20;++j) trace+=g.array[0]["shape"].array[j*20+j].num();
  CHECK_CLOSE(trace,20,1e-8);
}

TEST_CASE(cloning_periodic_field_uses_short_direction_and_live_strength) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":2,"low":-5,"high":5})"));
  FrozenPopulation p;p.dimensions=2;p.positions={4.9f,0,-4.9f,0};p.valid={1,1};
  SelectionEvidence e;e.donors={1,0};e.score={1,-1};
  CloningGuided periodic(b,cg_config(R"({"periodic":true,"adaptive_min_scale":1,"adaptive_max_scale":1})"));
  CloningGuided plain(b,cg_config(R"({"adaptive_min_scale":1,"adaptive_max_scale":1})"));
  periodic.observe_cloning(p,e);periodic.update();plain.observe_cloning(p,e);plain.update();
  auto component=[&](CloningGuided& model,bool wrap) {
    auto reference=model;
    reference.configure(cg_config(wrap ? R"({"periodic":true,"cloning_drift":false,"adaptive_min_scale":1,"adaptive_max_scale":1})" : R"({"cloning_drift":false,"adaptive_min_scale":1,"adaptive_max_scale":1})"));
    OptimizationRng a(9),z(9);float u[2],v[2];model.sample(p.positions.data(),u,2,a);reference.sample(p.positions.data(),v,2,z);
    return u[0]-v[0];
  };
  CHECK(component(periodic,true)>0);CHECK(component(plain,false)<0);
  periodic.configure(cg_config(R"({"periodic":true,"cloning_drift_strength":0,"adaptive_min_scale":1,"adaptive_max_scale":1})"));
  CHECK_CLOSE(component(periodic,true),0,1e-6);
  periodic.configure(cg_config(R"({"periodic":true,"cloning_drift_strength":0.5,"adaptive_min_scale":1,"adaptive_max_scale":1})"));
  CHECK(component(periodic,true)>0);
}

TEST_CASE(cloning_inactive_graph_ancestors_do_not_supply_comparisons) {
  Benchmark b(cg_config(R"({"benchmark":"quadratic","dimensions":2})"));
  auto p=cloud(2);auto e=evidence(p);e.active.assign(p.valid.size(),0);e.active[0]=1;
  CloningGuided model(b,cg_config("{}"));model.observe_cloning(p,e);model.update();
  CHECK(model.diagnostics()["selection_comparisons"].num()==1);
  CHECK(model.diagnostics()["model_count"].num()==1);
  CHECK(model.diagnostics()["condition_number"].num()==1);
}
