#include "optimization/adaptive.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
using namespace fg::fractal;
static Json config(const std::string& text) { return JsonReader(text).read(); }
static TrialOutcome outcome(AdaptiveExploration& model, uint64_t parent, uint64_t action,
                            double improvement, OptimizationRng& rng) {
  float x[2]={0,0}, delta[2];
  TrialOutcome out;
  out.trial=model.propose(x,delta,2,rng,{0,0,parent,action,0});
  out.displacement={delta[0],delta[1]}; out.improvement=improvement;
  return out;
}
TEST_CASE(adaptive_trial_identity_and_ancestry_are_independent) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":2})"));
  AdaptiveExploration m(b,config("{}")); OptimizationRng rng(19);
  auto o=outcome(m,1,1,1,rng);
  for(int k=0;k<8;++k) m.feedback(o);
  auto replay=o; replay.trial.identity.action=2; replay.replay=true; m.feedback(replay);
  m.update();
  CHECK(m.diagnostics()["observations"].num()==1);
  CHECK(m.diagnostics()["effective_parents"].num()==0);
  for(int k=0;k<8;++k) m.feedback(outcome(m,1,k+10,1,rng));
  m.update(); CHECK(m.diagnostics()["effective_parents"].num()==0);
}
TEST_CASE(adaptive_probability_floor_and_scale_bounds) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":2})"));
  AdaptiveExploration m(b,config("{}")); OptimizationRng rng(20);
  for(int cycle=0;cycle<20;++cycle) {
    for(int k=0;k<8;++k) m.feedback(outcome(m,k,cycle*8+k,k%2?1:-1,rng));
    m.update();
    auto d=m.diagnostics();
    CHECK(d["effective_parents"].num()>0);
    CHECK(d["scale_min"].num()>=.0001); CHECK(d["scale_max"].num()<=1);
    double sum=0; for(const auto& p:d["proposal_probabilities"].array) {CHECK(p.num()>=.1);sum+=p.num();}
    CHECK_CLOSE(sum,1,1e-12);
  }
  auto count=m.diagnostics()["observations"].num();
  m.configure(config(R"({"perturbation_std":2})")); CHECK(m.diagnostics()["observations"].num()==count);
  m.configure(config(R"({"periodic":true})")); CHECK(m.diagnostics()["model_count"].num()==0);
}
TEST_CASE(adaptive_fitness_scales_use_frozen_ranks_and_bound_all_families) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":2})"));
  auto c=config(R"({"adaptive_min_scale":0.01,"adaptive_max_scale":1,"adaptive_pairs":false})");
  AdaptiveExploration model(b,c);OptimizationRng rng(12);
  FrozenPopulation population{2,{-9,0, 0,0, 9,0},{1,2,3},{1,1,1},{0,10,10000}};
  model.freeze_population(population);
  float x[2]={0,0},delta[2];
  int seen[3]={};
  for(int k=0;k<1000;++k) {
    const int rank=k%3;
    const double expected=rank==0?.01:rank==1?.1:1;
    auto trial=model.propose_with_objective(x,delta,2,rng,{0,0,99,uint64_t(k),0},population.objectives[rank]);
    ++seen[int(trial.family)];
    CHECK_CLOSE(trial.scale,trial.family==ProposalFamily::Broad?std::min(1.,4*expected):expected,1e-12);
    CHECK(trial.scale>=.01 && trial.scale<=1);
    if(trial.family==ProposalFamily::Difference) {
      CHECK_CLOSE(std::hypot(delta[0],delta[1])/std::sqrt(2.),expected,1e-6);
      model.continue_trial(trial,delta,2,rng);
      CHECK_CLOSE(std::hypot(delta[0],delta[1])/std::sqrt(2.),expected,1e-6);
    }
  }
  for(int count:seen) CHECK(count>0);
  c.object["objective"].kind=Json::String;c.object["objective"].string="maximize";
  model.configure(c);model.freeze_population(population);
  for(int k=0;k<100;++k) {
    auto trial=model.propose_with_objective(x,delta,2,rng,{0,0,99,uint64_t(k),0},10000);
    CHECK_CLOSE(trial.scale,trial.family==ProposalFamily::Broad?.04:.01,1e-12);
  }
  population.objectives={4,4,4};model.freeze_population(population);
  for(int k=0;k<100;++k) {
    auto trial=model.propose_with_objective(x,delta,2,rng,{0,0,99,uint64_t(k),0},4);
    CHECK_CLOSE(trial.scale,trial.family==ProposalFamily::Broad?.4:.1,1e-12);
  }
  c.object["adaptive_min_scale"]=number(0);model.configure(c);
  for(int k=0;k<100;++k) {
    const auto trial=model.propose_with_objective(x,delta,2,rng,{0,0,99,uint64_t(k),0},4);
    CHECK_CLOSE(trial.scale,trial.family==ProposalFamily::Broad?1:.5,1e-12);
  }
  c.object["adaptive_min_scale"]=number(.25);c.object["adaptive_max_scale"]=number(.25);model.configure(c);
  for(int k=0;k<100;++k) CHECK_CLOSE(model.propose_with_objective(x,delta,2,rng,{0,0,99,uint64_t(k),0},4).scale,.25,1e-12);
  c.object["adaptive_min_scale"]=number(0);c.object["adaptive_max_scale"]=number(0);model.configure(c);
  for(int k=0;k<100;++k) {model.propose_with_objective(x,delta,2,rng,{0,0,99,uint64_t(k),0},4);CHECK(delta[0]==0 && delta[1]==0);}
}
TEST_CASE(adaptive_missing_donors_fall_back_and_large_shape_is_finite) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":80})"));
  AdaptiveExploration m(b,config("{}")); OptimizationRng rng(21);
  std::vector<float> x(80),delta(80);
  for(int cycle=0;cycle<3;++cycle) {
    for(int k=0;k<8;++k) {
      TrialOutcome out; out.trial=m.propose(x.data(),delta.data(),80,rng,{0,0,uint64_t(k),uint64_t(cycle*8+k),0});
      CHECK(out.trial.family!=ProposalFamily::Difference);
      out.displacement.assign(delta.begin(),delta.end());out.improvement=1;
      m.feedback(out);
    }
    m.update();m.sample(x.data(),delta.data(),80,rng);
    for(float v:delta) CHECK(std::isfinite(v));
  }
}
namespace {
class PairedFixture final : public Perturbation {
 public:
  std::vector<TrialOutcome> outcomes;
  void sample(const float*,float* out,int d,fg::Rng&) const override {
    std::fill(out,out+d,0);out[0]=.5f;
  }
  ProposalTrial propose(const float* x,float* out,int d,fg::Rng& rng,TrialIdentity id) const override {
    sample(x,out,d,rng);ProposalTrial trial;trial.identity=id;trial.origin.assign(x,x+d);
    trial.direction.assign(out,out+d);trial.paired=true;return trial;
  }
  void feedback(const TrialOutcome& o) override { outcomes.push_back(o); }
};
}
TEST_CASE(adaptive_paired_trials_count_and_commit_better_branch) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":2})"));
  OptimizationRng rng(3);PairedFixture p;float x[2]={1,0};
  double before=b.evaluate_optimization(x,&rng);auto count=b.evaluations;
  auto selected=evaluate_adaptive_position(p,b,config("{}"),x,before,rng,{0,0,1,1,0});
  CHECK(b.evaluations-count==2);CHECK(selected.valid);CHECK_CLOSE(selected.position[0],.5,1e-6);
  CHECK(p.outcomes.size()==2);CHECK(p.outcomes[0].trial.identity.branch==0);CHECK(p.outcomes[1].trial.identity.branch==1);
  CHECK(p.outcomes[1].trial.direction[0]==-.5);
  // At a minimum both alternatives worsen. The better valid branch still moves.
  x[0]=0;before=b.evaluate_optimization(x,&rng);
  selected=evaluate_adaptive_position(p,b,config("{}"),x,before,rng,{0,0,1,2,0});
  CHECK(std::abs(selected.position[0])==.5);
}
TEST_CASE(adaptive_stochastic_pairs_charge_all_endpoint_samples) {
  Benchmark b(config(R"({"benchmark":"stochastic_gaussian","dimensions":2})"));
  OptimizationRng rng(5);PairedFixture p;float x[2]={0,0};
  const auto before=b.evaluations;
  auto selected=evaluate_adaptive_position(p,b,config("{}"),x,0,rng,{0,0,1,1,0});
  CHECK(b.evaluations-before==9);CHECK(selected.valid);
  CHECK(p.outcomes.size()==2);
  CHECK(p.outcomes[0].evaluations+p.outcomes[1].evaluations==9);
  CHECK(p.outcomes[0].standard_error>=0);
}
TEST_CASE(adaptive_sparse_evidence_accumulates_without_reusing_trained_trials) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":2})"));
  AdaptiveExploration m(b,config("{}")); OptimizationRng rng(23);
  for(int parent=0;parent<4;++parent) {
    m.feedback(outcome(m,parent,parent,1,rng));m.update();
  }
  CHECK(m.diagnostics()["effective_parents"].num()>0);
  auto before=m.diagnostics();m.update();
  CHECK(stringify(before)==stringify(m.diagnostics()));
}
TEST_CASE(adaptive_high_dimension_negative_only_and_invalid_feedback) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":80})"));
  AdaptiveExploration m(b,config("{}"));OptimizationRng rng(31);
  std::vector<float> x(80),delta(80);
  for(int k=0;k<8;++k) {
    TrialOutcome o;o.trial=m.propose(x.data(),delta.data(),80,rng,{0,0,uint64_t(k),uint64_t(k),0});
    o.displacement.assign(delta.begin(),delta.end());o.improvement=-1;m.feedback(o);
  }
  m.update();m.sample(x.data(),delta.data(),80,rng);
  for(auto v:delta) CHECK(std::isfinite(v));
}

#include "optimization/engine.hpp"
#include "optimization/environment.hpp"
TEST_CASE(adaptive_all_adapters_reproduce_and_respect_step_admission) {
  for(auto algorithm:{"wave","graph","fmc","wave_jump","gas","euclidean"}) {
    auto c=config(R"({"benchmark":"quadratic","dimensions":2,"walkers":12,"max_walkers":24,"periodic":true,"horizon":2,"gas_local_search":false,"potential_force":false,"perturbation":"adaptive_fractal","perturbation_std":0.1})");
    c.object["algorithm"].kind=Json::String;c.object["algorithm"].string=algorithm;
    Session a(c),b(c);
    for(int i=0;i<8;++i) {
      const uint64_t before=a.benchmark.evaluations,bound=a.algorithm->next_evaluations_upper_bound();
      a.step();b.step();CHECK(a.snapshot==b.snapshot);
      CHECK(a.benchmark.evaluations-before<=bound);
    }
    auto status=config(a.status_json());
    CHECK(status["exploration"].kind==Json::Object);
    CHECK(status["exploration"]["observations"].num()>0);
    const auto before=a.benchmark.evaluations;
    a.update_settings(config(R"({"perturbation_std":0.2,"adaptive_active":false})"));
    CHECK(a.benchmark.evaluations==before);
  }
}
TEST_CASE(adaptive_euclidean_modes_commit_velocity_and_budget_gradients) {
  auto c=config(R"({"algorithm":"euclidean","benchmark":"bbob_f1","dimensions":2,"walkers":4,"periodic":true,"potential_force":true,"substeps":2,"perturbation":"adaptive_fractal"})");
  // Use a classic benchmark with counted finite-difference gradients.
  c.object["benchmark"].string="eggholder";
  Session session(c);
  const auto before=session.benchmark.evaluations;
  const auto bound=session.algorithm->next_evaluations_upper_bound();
  session.step();CHECK(session.benchmark.evaluations-before<=bound);
  session.update_settings(config(R"({"adaptive_euclidean_mode":"position"})"));
  for(float v:session.algorithm->population().v) CHECK(v==0);
  session.step();for(float v:session.algorithm->population().v) CHECK(v==0);
  session.update_settings(config(R"({"adaptive_euclidean_mode":"velocity"})"));
  session.step();
  for(float v:session.algorithm->population().v) CHECK(std::isfinite(v));
}
TEST_CASE(adaptive_environment_replay_preserves_branch_and_does_not_learn_twice) {
  auto c=config(R"({"algorithm":"fmc","benchmark":"quadratic","dimensions":2,"walkers":4,"perturbation":"adaptive_fractal","periodic":true})");
  Benchmark b(c);Settings s(b.config);BenchmarkEnvironment env(b,s);
  std::vector<char> state;std::vector<float> x;env.reset(state,x);env.freeze({state});
  std::vector<std::vector<char>> out(1),replayed(1);std::vector<float> obs(2),replay_obs(2),reward(1),replay_reward(1);
  std::vector<uint8_t> done(1),truncated(1);
  env.step_batch({state},{23},{3},out,obs,reward,done,truncated);
  auto count=b.evaluations;env.collect_perturbations(false);
  env.step_batch({state},{23},{3},replayed,replay_obs,replay_reward,done,truncated);
  CHECK(b.evaluations>count);CHECK(out==replayed);CHECK(obs==replay_obs);CHECK(reward==replay_reward);
  env.update_perturbation();CHECK(env.diagnostics()["observations"].num()<=2);
}
TEST_CASE(adaptive_geometry_round_trip_is_validated_and_reproducible) {
  Benchmark b(config(R"({"benchmark":"quadratic","dimensions":2})"));
  AdaptiveExploration source(b,config("{}")),restored(b,config("{}"));OptimizationRng evidence(8);
  for(int k=0;k<8;++k) source.feedback(outcome(source,k,k,k%2?1:-1,evidence));source.update();
  auto geometry=source.geometry();CHECK(!geometry.array.empty());restored.restore_geometry(geometry);
  OptimizationRng a(7),c(7);float x[2]={0,0},u[2],v[2];
  // Mixture adaptation is separate from reusable geometry; compare local factors
  // through the exported shape rather than claiming checkpoint restoration.
  CHECK(stringify(geometry)==stringify(restored.geometry()));
  auto legacy=geometry;legacy.array[0].object["scale"]=number(100);
  restored.restore_geometry(legacy);
  CHECK(stringify(geometry)==stringify(restored.geometry()));
  restored.sample(x,u,2,a);restored.sample(x,v,2,c);CHECK(u[0]==v[0]);CHECK(u[1]==v[1]);
  auto corrupted=geometry;corrupted.array[0].object["shape"].array[0]=number(-1);
  bool failed=false;try{restored.restore_geometry(corrupted);}catch(const std::exception&){failed=true;}
  CHECK(failed);CHECK(stringify(geometry)==stringify(restored.geometry()));
}
