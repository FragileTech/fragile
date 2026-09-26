#include "optimization/engine.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
static Json json(const std::string& text) { return JsonReader(text).read(); }
static Json setup(const std::string& algorithm) {
  auto c = json(R"({"benchmark":"quadratic","dimensions":2,"walkers":8,"max_walkers":24,"periodic":true,"horizon":2,"gas_local_search":false,"potential_force":false})");
  c.object["algorithm"].kind = Json::String;
  c.object["algorithm"].string = algorithm;
  return c;
}
static void rejected(Session& session, const Json& patch) {
  const auto before = session.snapshot;
  const auto status = session.status_json(), config = session.config_json;
  bool failed = false;
  try { session.update_settings(patch); } catch (const std::exception&) { failed = true; }
  CHECK(failed);
  CHECK(session.snapshot == before);
  CHECK(session.status_json() == status);
  CHECK(session.config_json == config);
}
TEST_CASE(live_adaptive_scale_limits_preserve_learning_and_planner_execution) {
  for(auto strategy:{"adaptive_fractal","cloning_guided"})
  for(auto name:{"wave","graph","fmc","wave_jump","euclidean","gas"}) {
    auto c=setup(name);c.object["perturbation"].kind=Json::String;c.object["perturbation"].string=strategy;
    Session changed(c),reference(c);
    changed.step();reference.step();
    rejected(changed,json(R"({"adaptive_min_scale":2,"adaptive_max_scale":1})"));
    rejected(changed,json(R"({"adaptive_min_scale":-1})"));
    rejected(changed,json(R"({"adaptive_round_fraction":0.1})"));
    const auto before=changed.snapshot;
    const auto geometry=stringify(changed.algorithm->movement_geometry());
    changed.update_settings(json(R"({"adaptive_min_scale":0.02,"adaptive_max_scale":0.03})"));
    CHECK(changed.snapshot==before);
    CHECK(stringify(changed.algorithm->movement_geometry())==geometry);
    if(changed.settings.planning()) {
      CHECK(changed.settings.json["adaptive_max_scale"].num()==1);
      bool applied=false;
      for(int i=0;i<20;++i) {
        changed.step();reference.step();CHECK(changed.snapshot==reference.snapshot);
        if(json(changed.status_json())["pending_settings"].kind==Json::Null) {applied=true;break;}
      }
      CHECK(applied);
    }
    CHECK(changed.settings.json["adaptive_min_scale"].num()==.02);
    CHECK(changed.settings.json["adaptive_max_scale"].num()==.03);
    CHECK(!changed.settings.json["scale_auto"].flag());
    changed.step();
    const auto diagnostics=json(changed.status_json())["exploration"];
    CHECK(diagnostics["scale_min"].num()>=.02);
    CHECK(diagnostics["scale_max"].num()<=.03);
  }
}
TEST_CASE(live_settings_noop_preserves_seeded_trajectories) {
  for (auto name : {"wave", "graph", "fmc", "wave_jump", "euclidean", "gas"}) {
    Session a(setup(name)), b(setup(name));
    for (int i = 0; i < 6; ++i) {
      a.update_settings(json(R"({"max_evaluations":0})"));
      CHECK(a.snapshot == b.snapshot);
      a.step(); b.step();
    }
    CHECK(a.snapshot == b.snapshot);
  }
}
TEST_CASE(live_settings_validation_and_budgets_are_atomic) {
  for (auto name : {"wave", "graph", "fmc", "wave_jump", "euclidean", "gas"}) {
    Session session(setup(name));
    rejected(session, json(R"({"seed":3,"perturbation_std":2})"));
    rejected(session, json(R"({"algorithm":"wave"})"));
    rejected(session, json(R"({"walkers":25,"max_walkers":24})"));
    rejected(session, json(R"({"periodic":"true"})"));
    rejected(session, json(R"({"perturbation":"unknown"})"));
    rejected(session, json(R"({"perturbation_std":-1})"));
    rejected(session, json(R"({"walkers":2.5})"));
    auto before = session.snapshot;
    session.update_settings(json(R"({"max_evaluations":1})"));
    CHECK(session.snapshot == before);
    CHECK(json(session.status_json())["budget_exhausted"].flag());
    session.update_settings(json(R"({"max_evaluations":0})"));
    CHECK(!json(session.status_json())["budget_exhausted"].flag());
    session.step();
  }
}
TEST_CASE(live_settings_resize_and_retune_without_reset) {
  for (auto name : {"wave", "euclidean", "gas"}) {
    Session s(setup(name));
    s.step();
    auto before = s.snapshot;
    const auto old = s.algorithm->population();
    s.update_settings(json(R"({"walkers":30,"max_walkers":32,"perturbation":"uniform","perturbation_std":0.2})"));
    CHECK(s.snapshot[1] == 30);
    CHECK(s.snapshot[4] == before[4]);
    CHECK(s.snapshot[5] == before[5]);
    CHECK(s.snapshot[9] == before[9]);
    auto p = s.algorithm->population();
    CHECK(std::equal(old.x.begin(), old.x.end(), p.x.begin()));
    CHECK(std::equal(old.v.begin(), old.v.end(), p.v.begin()));
    s.step();
    s.update_settings(json(R"({"walkers":4,"max_walkers":4,"removal_policy":"cumulative_reward"})"));
    CHECK(s.snapshot[1] == 4);
    for (int i = 0; i < 4; ++i) {
      CHECK(s.algorithm->population().parent[i] == i);
      CHECK(s.algorithm->population().companions[i] == i);
    }
    s.step();
  }
  Session wave(setup("wave"));
  wave.update_settings(json(R"({"walkers":4,"elites":4,"max_walkers":4})"));
  wave.step();
  rejected(wave, json(R"({"walkers":3})"));
  wave.update_settings(json(R"({"walkers":3,"elites":2})"));
  wave.step();
}
TEST_CASE(live_planner_changes_wait_for_completed_execution) {
  for (auto name : {"fmc", "wave_jump"}) {
    Session changed(setup(name)), reference(setup(name));
    changed.step(); reference.step();
    changed.update_settings(json(R"({"walkers":12,"perturbation":"uniform","perturbation_std":0.01,"horizon":3})"));
    CHECK(changed.settings.walkers == 8);
    CHECK(json(changed.status_json())["pending_settings"].kind == Json::Object);
    bool applied = false;
    for (int i = 0; i < 20; ++i) {
      changed.step(); reference.step();
      CHECK(changed.snapshot == reference.snapshot);
      if (json(changed.status_json())["pending_settings"].kind == Json::Null) { applied = true; break; }
    }
    CHECK(applied);
    CHECK(changed.settings.walkers == 12);
    CHECK(changed.settings.horizon == 3);
    CHECK(json(changed.status_json())["settings_event"]["state"].str() == "applied");
    changed.step();
    CHECK(changed.snapshot[1] == 13);
  }
}
TEST_CASE(live_graph_retains_nodes_and_cma_only_accepts_budget) {
  Session graph(setup("graph"));
  graph.step();
  auto before = graph.snapshot;
  rejected(graph, json(R"({"walkers":2,"max_walkers":2})"));
  graph.update_settings(json(R"({"walkers":3,"max_walkers":32,"perturbation_std":0.2,"freeze_prefix_after":10})"));
  CHECK(graph.snapshot == before);
  graph.step();
  auto c = setup("cmaes_active"); c.object["periodic"].number = false;
  Session cma(c);
  rejected(cma, json(R"({"cma_sigma":0.1})"));
  rejected(cma, json(R"({"walkers":12})"));
  cma.update_settings(json(R"({"max_evaluations":1})"));
  CHECK(json(cma.status_json())["budget_exhausted"].flag());
  cma.update_settings(json(R"({"max_evaluations":0})"));
  cma.step();
}
TEST_CASE(live_local_covariance_preserves_learning_and_resets_periodic_geometry) {
  auto c = setup("wave");
  c.object["perturbation"].kind = Json::String; c.object["perturbation"].string = "local_covariance";
  Settings settings(c); Benchmark b(c);
  auto learned = make_perturbation(b, settings.json);
  for (int i = 0; i < 10; ++i) {
    PerturbationTransition t;
    t.origin = {0, 0}; t.displacement = {double(i + 1), .01}; t.draws = 1; t.improvement = i + 1;
    learned->observe(t);
  }
  learned->update();
  auto next = settings.json; next.object["perturbation_std"] = number(2);
  auto tuned = retune_perturbation(*learned, b, settings.json, next);
  float origin[2]{}, a[2], d[2], fresh[2];
  OptimizationRng r1(9), r2(9), r3(9);
  learned->sample(origin, a, 2, r1); tuned->sample(origin, d, 2, r2);
  CHECK_CLOSE(d[0], 2 * a[0], 1e-6); CHECK_CLOSE(d[1], 2 * a[1], 1e-6);
  auto independent = make_perturbation(b, next);
  independent->sample(origin, fresh, 2, r3);
  CHECK(std::abs(fresh[0] - d[0]) > .01);
  next.object["periodic"].number = false;
  auto reset = retune_perturbation(*learned, b, settings.json, next);
  OptimizationRng r4(9); reset->sample(origin, d, 2, r4);
  CHECK(d[0] == fresh[0]); CHECK(d[1] == fresh[1]);
}
TEST_CASE(live_settings_reject_missing_donors_and_unsafe_resources) {
  Session wave(setup("wave"));
  wave.update_settings(json(R"({"periodic":false,"perturbation_std":1000000})"));
  wave.step();
  CHECK(wave.snapshot[6] == 0);
  rejected(wave, json(R"({"walkers":16,"perturbation_std":0.1})"));
  rejected(wave, json(R"({"dt_min":4,"dt_max":2})"));
  auto c = setup("wave"); c.object["dimensions"] = number(1024);
  Session large(c);
  rejected(large, json(R"({"perturbation":"local_covariance"})"));
  Session euclidean(setup("euclidean"));
  rejected(euclidean, json(R"({"walkers":5000,"max_walkers":6000})"));
}
