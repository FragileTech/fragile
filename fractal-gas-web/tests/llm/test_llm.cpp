#include "test_framework.hpp"
#include "fractal/distance.hpp"
#include "fractal/tensor_ops.hpp"
#include "fractal_gas.hpp"
#include "fractal_tree.hpp"
#include "llm/environment.hpp"
#include "thread_pool.hpp"
#include <limits>
using namespace fg;
TEST_CASE(distance_api_preserves_l2_and_defines_cosine) {
  float a[] = {3,4}, b[] = {6,8}, c[] = {-3,-4}, z[] = {0,0};
  CHECK(row_distance(a,b,2) == 5);
  CHECK_CLOSE(row_distance(a,b,2,DistanceMetric::Cosine),0,1e-7);
  CHECK_CLOSE(row_distance(a,c,2,DistanceMetric::Cosine),2,1e-7);
  CHECK(row_distance(z,z,2,DistanceMetric::Cosine) == 0);
  CHECK(row_distance(a,z,2,DistanceMetric::Cosine) == 1);
  CHECK(parse_distance_metric(distance_metric_name(DistanceMetric::L2)) == DistanceMetric::L2);
  bool threw = false; try { parse_distance_metric("bad"); } catch (...) { threw = true; } CHECK(threw);
  a[0] = INFINITY; threw = false; try { row_distance(a,b,2); } catch (...) { threw = true; } CHECK(threw);
}
TEST_CASE(distance_batches_are_identical_across_threads) {
  std::vector<float> obs(256*256); std::vector<int32_t> companions(256);
  for (int i=0;i<256;++i) { companions[i]=(i+17)%256;
    for(int k=0;k<256;++k) obs[i*256+k]=float((i+k)%31); }
  ThreadPool pool(3); std::vector<float> a,b;
  for (auto metric : {DistanceMetric::L2,DistanceMetric::Cosine}) {
    companion_distances_into(obs,companions,256,256,nullptr,a,metric);
    companion_distances_into(obs,companions,256,256,&pool,b,metric); CHECK(a==b);
    if(metric==DistanceMetric::L2) CHECK(a==l2_norm_companions(obs,companions,256,256));
  }
}
TEST_CASE(llm_environment_telescopes_objectives_and_preserves_sources) {
  for (bool mean : {false,true}) {
    llm::LlmEnvironment env(2,[mean](const auto& requests) {
      std::vector<llm::Result> out;
      for(const auto& r:requests) {
        auto s=r.source; s.id++; s.tokens+=r.duration; s.logp-=r.duration*0.5;
        s.utility = mean ? s.logp / s.tokens : s.logp;
        if(s.tokens>=4) s.status=2;
        out.push_back({s,{1,2}});
      } return out;
    });
    std::vector<char> root; std::vector<float> obs,rewards; env.reset(root,obs);
    CHECK(!env.best_candidate(root));
    std::vector<std::vector<char>> states={root,root},next;
    std::vector<uint8_t> done,trunc;
    env.step_batch(states,{1,2},{2,2},next,obs,rewards,done,trunc);
    CHECK(llm::LlmEnvironment::decode(states[0]).tokens==0);
    CHECK(env.best_candidate(next[0])); double total=rewards[0]; states=next;
    env.step_batch(states,{3,4},{2,2},next,obs,rewards,done,trunc);
    total+=rewards[0]; CHECK_CLOSE(total,mean?-0.5:-2,1e-7);
    CHECK(trunc[0]); CHECK(env.frames_stepped(0)==2);
  }
}
TEST_CASE(llm_algorithms_share_metric_and_exclude_empty_best) {
  uint32_t id=0;
  llm::LlmEnvironment env(2,[&](const auto& requests) {
    std::vector<llm::Result> out;
    for(const auto& r:requests) { auto s=r.source; s.id=++id; s.tokens++; s.logp-=0.1*id;
      s.utility = s.logp; out.push_back({s,{float(id),1.f}}); } return out;
  });
  FractalGasParams w; w.N=8; w.dt_min=w.dt_max=1; w.distance_metric=DistanceMetric::Cosine;
  w.use_cumulative_reward=true;
  FractalGas wave(env,w); wave.reset(); CHECK(wave.get_best_walker().first==-1); wave.step(); wave.step();
  CHECK(wave.get_best_walker().first>=0); CHECK(wave.params().distance_metric==DistanceMetric::Cosine);
  FractalTreeParams g; g.start_walkers=g.min_leafs=8; g.max_walkers=32; g.dt_min=g.dt_max=1;
  g.distance_metric=DistanceMetric::Cosine; FractalTree graph(env,g); graph.reset();
  CHECK(graph.get_best_walker().first>0);
  for(int i=0;i<4;++i) graph.step(); CHECK(graph.get_best_walker().first>0);
}

TEST_CASE(optional_diagnostics_preserve_seeded_wave_and_graph_results) {
  auto make_env = [] {
    auto next_id = std::make_shared<uint32_t>(0);
    return std::make_unique<llm::LlmEnvironment>(3, [next_id](const auto& requests) {
      std::vector<llm::Result> out;
      for (const auto& r : requests) {
        auto s = r.source;
        if (!s.status) {
          s.id = ++*next_id; s.tokens += r.duration;
          s.logp -= (0.1 + (s.id % 7) * 0.2) * r.duration;
          if (s.tokens >= 12) s.status = 2;
        }
        s.utility = s.logp; out.push_back({s, {float(s.id % 5), float(s.tokens), 1.f}});
      }
      return out;
    });
  };
  for (auto metric : {DistanceMetric::L2, DistanceMetric::Cosine}) {
    auto a = make_env(), b = make_env();
    FractalGasParams p; p.N = 8; p.dt_min = p.dt_max = 2; p.seed = 19;
    p.distance_metric = metric; p.use_cumulative_reward = true;
    FractalGas plain(*a,p), observed(*b,p); observed.enable_diagnostics();
    plain.reset(); observed.reset();
    for (int step = 0; step < 10; ++step) {
      const auto before = observed.state();
      plain.step(); observed.step();
      CHECK(plain.state().states == observed.state().states);
      CHECK(plain.state().observations == observed.state().observations);
      CHECK(plain.state().rewards == observed.state().rewards);
      CHECK(plain.clone_mask() == observed.clone_mask());
      CHECK(plain.diagnostics().decisions.empty());
      CHECK(observed.diagnostics().decisions.size() == 8);
      for (const auto& d : observed.diagnostics().decisions) {
        CHECK(d.distance == row_distance(before.observations.data()+d.slot*3,
            before.observations.data()+d.distance_companion*3,3,metric));
        CHECK_CLOSE(d.fitness, d.distance_norm*d.reward_norm, 1e-6);
        CHECK(d.donor_fitness == observed.state().virtual_rewards[d.clone_donor]);
        CHECK(d.cloned == (d.clone_score > d.draw || !d.alive));
        CHECK(d.cloned == bool(observed.clone_mask()[d.slot]));
      }
    }
    a = make_env(); b = make_env();
    FractalTreeParams g; g.start_walkers=g.min_leafs=8;g.max_walkers=32;
    g.dt_min=g.dt_max=2;g.seed=23;g.distance_metric=metric;g.count_visits=false;
    FractalTree base(*a,g), instrumented(*b,g); instrumented.enable_diagnostics();
    base.reset(); instrumented.reset(); CHECK(instrumented.diagnostics().decisions.empty());
    for(int step=0;step<12;++step){
      const auto before=instrumented.state();
      base.step();instrumented.step();
      CHECK(base.state().states==instrumented.state().states);
      CHECK(base.state().parent==instrumented.state().parent);
      CHECK(base.state().virtual_rewards==instrumented.state().virtual_rewards);
      CHECK(base.state().will_clone==instrumented.state().will_clone);
      CHECK(instrumented.diagnostics().decisions.size()==size_t(before.n));
      for(const auto& d:instrumented.diagnostics().decisions){
        CHECK(d.distance==row_distance(before.observations.data()+d.slot*3,
            before.observations.data()+d.distance_companion*3,3,metric));
        CHECK(d.normalization_leaf==bool(before.is_leaf[d.slot]));
        CHECK_CLOSE(d.fitness,d.distance_norm*d.reward_norm*d.other,1e-5);
        CHECK(d.wanted==(d.clone_score>d.draw||!d.alive));
        CHECK(d.cloned==bool(instrumented.state().will_clone[d.slot]));
        if(!d.leaf||d.donor_protected||d.best_protected||d.invalid_donor) CHECK(!d.cloned);
      }
    }
  }
}

TEST_CASE(llm_terminal_and_skipped_transitions_cannot_change_source) {
  bool skipped = true, mutate = false;
  llm::LlmEnvironment env(2, [&](const auto& requests) {
    std::vector<llm::Result> out;
    for (const auto& r : requests) {
      auto s = r.source;
      if (mutate) { s.id++; s.tokens++; s.logp -= 0.1; }
      s.utility = s.logp; out.push_back({s, {1, 2}, skipped});
    }
    return out;
  });
  for (uint32_t status : {0u, 1u, 2u}) {
    llm::Snapshot source{7, 3, -0.5, -0.5, status};
    std::vector<std::vector<char>> states{llm::LlmEnvironment::encode(source)}, next;
    std::vector<float> obs, reward;
    std::vector<uint8_t> done, truncated;
    skipped = status == 0; mutate = false;
    env.step_batch(states, {1}, {2}, next, obs, reward, done, truncated);
    CHECK(llm::LlmEnvironment::decode(next[0]).id == source.id);
    CHECK(reward[0] == 0); CHECK(env.frames_stepped(0) == 0);
    CHECK(bool(done[0]) == (status == 1)); CHECK(bool(truncated[0]) == (status == 2));
    CHECK(env.best_candidate(states[0]) == (status == 0));
    mutate = true;
    bool threw = false;
    try { env.step_batch(states, {1}, {2}, next, obs, reward, done, truncated); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(llm::LlmEnvironment::decode(next[0]).id == source.id);
  }
}

TEST_CASE(llm_skipped_utility_and_nonfinite_utility_are_rejected_atomically) {
  for (double utility : {1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    llm::LlmEnvironment env(1, [utility](const auto& requests) {
      auto state = requests.front().source; state.utility = utility;
      return std::vector<llm::Result>{{state, {0.f}, true}};
    });
    std::vector<char> root; std::vector<float> obs, reward; env.reset(root, obs);
    std::vector<std::vector<char>> next; std::vector<uint8_t> done, truncated;
    bool threw = false;
    try { env.step_batch({root}, {1}, {2}, next, obs, reward, done, truncated); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw); CHECK(next.empty());
  }
}
