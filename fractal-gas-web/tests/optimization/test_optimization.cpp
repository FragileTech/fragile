#include <algorithm>

#include "arcade_planner.hpp"
#include "optimization/engine.hpp"
#include "optimization/environment.hpp"
#include "test_framework.hpp"
using namespace fg::optimization;
static Json config(const std::string& text) { return JsonReader(text).read(); }
TEST_CASE(optimization_benchmark_values) {
  Benchmark b(config(R"({"benchmark":"rosenbrock","dimensions":2})"));
  float x[] = {0, 0};
  CHECK_CLOSE(b.evaluate(x), 1, 1e-7);
  x[0] = 1.2f;
  x[1] = 1.44f;
  CHECK_CLOSE(b.evaluate(x), .04, 1e-6);
  Benchmark lj(config(R"({"benchmark":"lennard_jones","n_atoms":2})"));
  float atoms[] = {0, 0, 0, float(std::pow(2, 1. / 6)), 0, 0};
  CHECK_CLOSE(lj.evaluate(atoms), -1, 1e-6);
  atoms[3] = 0;
  CHECK(!std::isfinite(lj.evaluate(atoms)));
  CHECK(JsonReader(catalog_json()).read()["benchmarks"].array.size() == 37);
}
TEST_CASE(optimization_gradients) {
  auto catalog = JsonReader(catalog_json()).read();
  for (auto& entry : catalog["benchmarks"].array) {
    if (entry["suite"].str() == "bbob") continue;
    auto c = config(R"({"dimensions":2,"n_atoms":2})");
    c.object["benchmark"] = entry["id"];
    Benchmark b(c);
    std::vector<float> x(b.d), g(b.d);
    for (int k = 0; k < b.d; ++k) x[k] = float(.37 + k * .81);
    b.gradient(x.data(), g.data());
    for (int k = 0; k < b.d; ++k) {
      float old = x[k], h = 1e-3f;
      x[k] = old + h;
      double hi = b.evaluate(x.data());
      x[k] = old - h;
      double lo = b.evaluate(x.data());
      x[k] = old;
      CHECK_CLOSE(g[k], (hi - lo) / (2 * h), .003);
    }
  }
}
TEST_CASE(optimization_deterministic_sessions) {
  for (auto name : {"wave", "graph", "euclidean", "fmc", "wave_jump"}) {
    auto c = config(
        R"({"benchmark":"quadratic","walkers":16,"max_walkers":64,"periodic":true,"horizon":2})");
    c.object["algorithm"].kind = Json::String;
    c.object["algorithm"].string = name;
    Session a(c), b(c);
    for (int i = 0; i < 20; ++i) {
      CHECK(a.snapshot == b.snapshot);
      a.step();
      b.step();
    }
    CHECK(a.snapshot == b.snapshot);
    CHECK(a.best <= a.snapshot[8]);
    CHECK(a.snapshot.size() ==
          12 + size_t(a.snapshot[1]) * (2 * size_t(a.snapshot[2]) + 8));
  }
}
TEST_CASE(optimization_noise_sampling_is_observational) {
  Session a(config(R"({"benchmark":"stochastic_gaussian","walkers":8})")),
      b(config(R"({"benchmark":"stochastic_gaussian","walkers":8})"));
  float x[] = {1, 2, 3};
  for (int i = 0; i < 100; ++i) CHECK(a.benchmark.evaluate(x) == 0);
  a.step();
  b.step();
  CHECK(a.snapshot == b.snapshot);
  CHECK(!a.settings.potential_force);
}
TEST_CASE(optimization_periodic_and_cloning) {
  Benchmark b(
      config(R"({"benchmark":"constant","dimensions":2,"low":-1,"high":1})"));
  float x[] = {5.5f, -4.5f};
  b.wrap(x);
  CHECK_CLOSE(x[0], -.5, 1e-6);
  CHECK_CLOSE(x[1], -.5, 1e-6);
  Population p;
  p.resize(3, 2);
  p.x = {0, 0, 1, 1, 2, 2};
  p.v = {1, 0, 2, 0, 3, 0};
  Settings s(config(R"({"sigma_x":0,"restitution":0})"));
  fg::Mt19937Rng rng(7);
  clone_population(p, s, {1, 1, 1}, {1, 0, 1}, rng);
  CHECK(p.x == std::vector<float>({1, 1, 1, 1, 1, 1}));
  CHECK(p.v == std::vector<float>({2, 0, 2, 0, 2, 0}));
}
TEST_CASE(optimization_all_dead_and_validation) {
  bool caught = false;
  try {
    Session bad(config(R"({"benchmark":"easom","dimensions":3})"));
  } catch (const std::exception&) {
    caught = true;
  }
  CHECK(caught);
  Population p;
  p.resize(2, 2);
  Benchmark b(config(R"({"dimensions":2})"));
  Settings s(config("{}"));
  fg::Mt19937Rng rng(1);
  caught = false;
  try {
    select_companions(p, s, b, "uniform", 1, rng);
  } catch (const std::exception&) {
    caught = true;
  }
  CHECK(caught);
  auto tiny = config(
      R"({"benchmark":"constant","dimensions":2,"walkers":8,"low":-0.000001,"high":0.000001,"cloning":false,"delta_t":1})");
  Session killed(tiny);
  killed.step();
  CHECK(killed.snapshot[6] == 0);
  auto final_frame = killed.snapshot;
  caught = false;
  try {
    killed.step();
  } catch (const std::exception&) {
    caught = true;
  }
  CHECK(caught);
  CHECK(killed.snapshot == final_frame);
  Session reset(tiny);
  CHECK(reset.snapshot[6] == 8);
  tiny.object["periodic"].kind = Json::Boolean;
  tiny.object["periodic"].number = 1;
  Session wrapped(tiny);
  wrapped.step();
  CHECK(wrapped.snapshot[6] == 8);
}

#include "optimization/fixtures.hpp"
class FixtureRng final : public fg::Rng {
 public:
  float uniform01() override { return .37f; }
  int64_t randint(int64_t lo, int64_t) override { return lo; }
};
static Population fixture_population() {
  namespace f = optimization_fixture;
  Population p;
  p.resize(4, 2);
  p.has_velocity = true;
  p.x.assign(f::positions.begin(), f::positions.end());
  p.v.assign(f::velocities.begin(), f::velocities.end());
  p.objective = f::objectives;
  p.alive.assign(4, 1);
  return p;
}
template <typename T>
static void fixture_equal(const std::vector<T>& a,
                          const std::vector<double>& expected,
                          double tol = 2e-6) {
  CHECK(a.size() == expected.size());
  for (size_t i = 0; i < std::min(a.size(), expected.size()); ++i)
    CHECK_CLOSE(a[i], expected[i], tol);
}
TEST_CASE(optimization_python_operator_fixtures) {
  namespace f = optimization_fixture;
  auto cfg = config(
      R"({"benchmark":"quadratic","dimensions":2,"walkers":4,"lambda_alg":0.2,"gamma":0.8,"beta":1.2,"delta_t":0.01,"sigma_x":0.03,"restitution":0.4})");
  Benchmark b(cfg);
  Settings s(cfg);
  auto p = fixture_population();
  FixtureRng rng;
  std::vector<int32_t> companions{1, 0, 3, 2};
  fixture_equal(fitness(p, s, b, companions), f::fitness_global);
  Benchmark periodic(config(R"({"dimensions":2,"low":-1,"high":1})"));
  s.periodic = true;
  fixture_equal(fitness(p, s, periodic, companions), f::fitness_periodic);
  auto wrapped = p.x;
  for (int i = 0; i < p.n; ++i) periodic.wrap(wrapped.data() + i * p.d);
  fixture_equal(wrapped, f::wrapped_positions);
  s.periodic = false;
  s.rho = 1.5;
  fixture_equal(fitness(p, s, b, companions), f::fitness_local);
  s.rho = 0;
  for (auto method :
       {"uniform", "softmax", "cloning", "random_pairing", "greedy_pairing"}) {
    auto actual = select_companions(p, s, b, method, 2, rng);
    const auto& expected =
        std::string(method) == "uniform"   ? f::companions_uniform
        : std::string(method) == "softmax" ? f::companions_softmax
        : std::string(method) == "cloning" ? f::companions_cloning
        : std::string(method) == "random_pairing"
            ? f::companions_random_pairing
            : f::companions_greedy_pairing;
    fixture_equal(actual, expected, 0);
    const auto& periodic_expected =
        std::string(method) == "uniform"   ? f::periodic_companions_uniform
        : std::string(method) == "softmax" ? f::periodic_companions_softmax
        : std::string(method) == "cloning" ? f::periodic_companions_cloning
        : std::string(method) == "random_pairing"
            ? f::periodic_companions_random_pairing
            : f::periodic_companions_greedy_pairing;
    s.periodic = true;
    fixture_equal(select_companions(p, s, periodic, method, 2, rng),
                  periodic_expected, 0);
    s.periodic = false;
  }
  std::vector<uint8_t> mask(f::clone_mask.begin(), f::clone_mask.end());
  clone_population(p, s, companions, mask, rng);
  fixture_equal(p.x, f::cloned_x);
  fixture_equal(p.v, f::cloned_v);
  p = fixture_population();
  baoab(p, s, b, rng);
  fixture_equal(p.x, f::baoab_x);
  fixture_equal(p.v, f::baoab_v);
}

TEST_CASE(optimization_cumulative_scores_follow_objectives) {
  for (auto algorithm : {"wave", "graph", "fmc", "wave_jump", "euclidean"})
    for (auto direction : {"minimize", "maximize"})
      for (auto benchmark : {"quadratic", "stochastic_gaussian"}) {
        auto c = config(
            R"({"dimensions":3,"walkers":24,"max_walkers":200,"dt_min":1,"dt_max":4,"periodic":true,"elites":2,"horizon":2})");
        c.object["algorithm"].kind = Json::String;
        c.object["algorithm"].string = algorithm;
        c.object["benchmark"].kind = Json::String;
        c.object["benchmark"].string = benchmark;
        c.object["objective"].kind = Json::String;
        c.object["objective"].string = direction;
        Session session(c);
        for (int t = 0; t < 40; ++t) {
          auto& p = session.algorithm->population();
          for (int i = 0; i < p.n; ++i)
            if (p.alive[i])
              CHECK_CLOSE(session.algorithm->objective_score(i),
                          session.settings.score(p.objective[i]), 2e-5);
          double current = session.settings.worst();
          for (int i = 0; i < p.n; ++i)
            if (p.alive[i] && session.settings.better(p.objective[i], current))
              current = p.objective[i];
          CHECK(session.snapshot[8] == current);
          CHECK(session.settings.score(session.best) >=
                session.settings.score(current));
          const double previous = session.best;
          session.step();
          CHECK(session.settings.score(session.best) >=
                session.settings.score(previous));
        }
      }
}
TEST_CASE(optimization_registry_exposes_extensions) {
  register_algorithm(
      "extra", "Extra optimizer", false,
      [](Benchmark&, const Settings&) -> std::unique_ptr<Algorithm> {
        throw std::runtime_error("fixture only");
      },
      config(
          R"([{"id":"learning_rate","label":"Learning rate","type":"number","default":0.1,"min":0,"max":1}])"));
  auto c = JsonReader(discovery_json()).read();
  bool found = false;
  for (auto& a : c["algorithms"].array)
    if (a["id"].str() == "extra") {
      found = true;
      CHECK(a["parameters"].array.size() == 1);
      CHECK(a["parameters"].array[0]["id"].str() == "learning_rate");
    }
  CHECK(found);
}

TEST_CASE(optimization_planners_reuse_arcade_execution) {
  for (auto strategy : {"gaussian", "local_covariance"})
  for (auto name : {"fmc", "wave_jump"}) {
    auto cfg = config(
        R"({"benchmark":"stochastic_gaussian","walkers":12,"horizon":2,"dt_max":3,"periodic":true,"consensus_prefix":false,"objective":"maximize"})");
    cfg.object["perturbation"].kind = Json::String;
    cfg.object["perturbation"].string = strategy;
    cfg.object["algorithm"].kind = Json::String;
    cfg.object["algorithm"].string = name;
    Session session(cfg);
    Benchmark bench(cfg);
    Settings settings(bench.config);
    BenchmarkEnvironment env(bench, settings);
    fg::FractalGasParams params;
    params.N = settings.walkers;
    params.seed = settings.seed;
    params.dt_max = settings.dt_max;
    params.use_cumulative_reward = true;
    params.count_visits = false;
    params.recording = fg::RecordingMode::Pruned;
    params.record_observations = false;
    fg::FractalGas gas(env, params,
                       std::make_unique<OptimizationRng>(settings.seed));
    fg::ArcadePlannerSettings options;
    options.algorithm = settings.algorithm == "fmc" ? 2 : 3;
    options.horizon = settings.horizon;
    options.consensus_prefix = false;
    fg::ArcadePlanner planner(env, gas, options);
    planner.reset();
    int executed = 0;
    for (int t = 0; t < 30; ++t) {
      const auto& p = session.algorithm->population();
      CHECK(p.n == settings.walkers + 1);
      std::vector<float> x(bench.d);
      CHECK(p.objective.back() == env.decode(planner.state(), x.data()));
      CHECK(std::equal(x.begin(), x.end(), p.x.end() - bench.d));
      for (int i = 0; i < settings.walkers; ++i) {
        CHECK(p.objective[i] == env.decode(gas.walker_state(i), x.data()));
        CHECK(
            std::equal(x.begin(), x.end(), p.x.begin() + size_t(i) * bench.d));
      }
      session.step();
      if (planner.new_search_pending()) env.update_perturbation();
      env.collect_perturbations(!planner.execution_pending());
      const auto info = planner.advance();
      CHECK(session.snapshot[7] ==
            (planner.search_advanced() ? info.num_cloned : 0));
      if (planner.search_advanced() && std::string(strategy) == "local_covariance") {
        // Reconstruct a searched lineage with the frozen model, even after
        // covariance has learned across multiple planning cycles.
        const auto& tree = gas.exploration_tree();
        std::vector<std::vector<char>> replay(1);
        replay[0].assign(tree.root_snapshot.begin(), tree.root_snapshot.end());
        std::vector<float> observations(bench.d), rewards(1);
        std::vector<uint8_t> dones(1), truncated(1);
        env.collect_perturbations(false);
        for (auto id : tree.branch(gas.state().lineage[0])) {
          const auto& node = tree.node(id);
          if (node.frames == 0) continue;
          env.step_batch(replay, {int32_t(tree.action(id)[0])}, {int32_t(node.frames)},
                         replay, observations, rewards, dones, truncated);
          CHECK(rewards[0] == node.step_reward);
        }
        CHECK(replay[0] == gas.walker_state(0));
      }
      executed += !planner.search_advanced();
    }
    CHECK(executed > 0);
  }
}

TEST_CASE(optimization_perturbations_replay_actions) {
  for (auto strategy : {"gaussian", "uniform", "local_covariance"}) {
    auto cfg = config(
        R"({"algorithm":"fmc","benchmark":"stochastic_gaussian","dimensions":3,"periodic":true,"perturbation_std":0.3})");
    cfg.object["perturbation"].kind = Json::String;
    cfg.object["perturbation"].string = strategy;
    Benchmark b(cfg);
    Settings s(b.config);
    BenchmarkEnvironment env(b, s);
    std::vector<char> root;
    std::vector<float> obs;
    env.reset(root, obs);
    const std::vector<int32_t> actions{0, 16777215}, frames{1, 4};
    std::vector<std::vector<char>> states(2);
    std::vector<float> x(6), rewards(2);
    std::vector<uint8_t> dones(2), truncated(2);
    env.step_batch({root, root}, actions, frames, states, x, rewards, dones,
                   truncated);
    const auto expected = states;
    const auto expected_rewards = rewards;
    // An unrelated evaluation and step must not alter recorded action
    // semantics.
    b.evaluate(obs.data());
    env.step_batch(states, {42, 53}, {3, 2}, states, x, rewards, dones,
                   truncated);
    env.step_batch({root, root}, actions, frames, states, x, rewards, dones,
                   truncated);
    CHECK(states == expected);
    CHECK(rewards == expected_rewards);
    CHECK(dones == std::vector<uint8_t>({0, 0}));
    auto doubled = s.json;
    doubled.object["perturbation_std"] = number(.6);
    auto a = make_perturbation(b, s.json), c = make_perturbation(b, doubled);
    OptimizationRng ar(7), cr(7);
    std::vector<float> dx(3), twice(3);
    a->sample(obs.data(), dx.data(), 3, ar);
    c->sample(obs.data(), twice.data(), 3, cr);
    for (int k = 0; k < 3; ++k) CHECK_CLOSE(twice[k], 2 * dx[k], 1e-7);
  }
}

class FixedPerturbation final : public Perturbation {
 public:
  void sample(const float*, float* delta, int d, fg::Rng&) const override {
    std::fill(delta, delta + d, .01f);
  }
};
TEST_CASE(optimization_perturbation_extension_and_force_direction) {
  register_perturbation(
      "fixed", "Fixed test step",
      [](const Benchmark&, const Json&) {
        return std::make_unique<FixedPerturbation>();
      },
      config("[]"));
  const auto catalog = JsonReader(discovery_json()).read();
  CHECK(catalog["perturbations"].array.size() == 5);
  auto cfg = config(
      R"({"algorithm":"fmc","benchmark":"quadratic","dimensions":2,"perturbation":"fixed"})");
  Benchmark b(cfg);
  Settings s(b.config);
  BenchmarkEnvironment env(b, s);
  std::vector<char> state;
  std::vector<float> start;
  env.reset(state, start);
  std::vector<std::vector<char>> next(1);
  std::vector<float> x(2), reward(1);
  std::vector<uint8_t> dones(1), trunc(1);
  env.step_batch({state}, {7}, {2}, next, x, reward, dones, trunc);
  for (int k = 0; k < 2; ++k) CHECK_CLOSE(x[k], start[k] + .02, 1e-6);

  auto zero =
      config(R"({"perturbation_std":0,"cloning":false,"delta_t":0.01})");
  auto noise = make_perturbation(b, zero);
  Population down, up;
  down.resize(2, 2);
  down.x.assign(4, 1);
  up = down;
  Settings minimize(zero), maximize(zero);
  maximize.objective = "maximize";
  OptimizationRng rng(7);
  baoab(down, minimize, b, rng, noise.get());
  baoab(up, maximize, b, rng, noise.get());
  CHECK(down.x[0] < 1);
  CHECK(up.x[0] > 1);
  minimize.potential_force = false;
  down.x.assign(4, 1);
  down.v.assign(4, 0);
  baoab(down, minimize, b, rng, noise.get());
  CHECK(down.x == std::vector<float>({1, 1, 1, 1}));

  bool caught = false;
  try {
    Session invalid(config(R"({"objective":"sideways"})"));
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  CHECK(caught);
  caught = false;
  try {
    Session invalid(config(R"({"perturbation":"unknown"})"));
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  CHECK(caught);
}

TEST_CASE(optimization_local_covariance_geometry) {
  auto cfg = config(R"({"algorithm":"wave","benchmark":"quadratic","dimensions":2,"perturbation":"local_covariance","covariance_learning_rate":1})");
  Benchmark b(cfg);
  auto noise = make_perturbation(b, cfg);
  float origin[] = {0, 0}, delta[2];
  // Four independent trials along a rotated valley establish off-diagonal shape.
  for (int i = 0; i < 4; ++i)
    noise->observe({{0, 0}, {1., 1.}, 1, double(i + 1)});
  noise->update();
  OptimizationRng rng(41);
  double xx = 0, yy = 0, xy = 0;
  for (int i = 0; i < 20000; ++i) {
    noise->sample(origin, delta, 2, rng);
    xx += delta[0] * delta[0]; yy += delta[1] * delta[1]; xy += delta[0] * delta[1];
  }
  xx /= 20000; yy /= 20000; xy /= 20000;
  CHECK_CLOSE(xx + yy, 2, .07);
  CHECK(xy > .9);
  CHECK(xx * yy - xy * xy > 0);
  CHECK_CLOSE(xx + yy - 2 * xy, .1, .01);
  // Identical history and seeds reproduce geometry; n draws use sqrt(n).
  auto repeated = make_perturbation(b, cfg);
  for (int i = 0; i < 4; ++i)
    repeated->observe({{0, 0}, {2., 2.}, 4, double(4 * (i + 1))});
  repeated->update();
  OptimizationRng a(8), c(8);
  float other[2];
  noise->sample(origin, delta, 2, a);
  repeated->sample(origin, other, 2, c);
  CHECK_CLOSE(delta[0], other[0], 1e-7);
  CHECK_CLOSE(delta[1], other[1], 1e-7);
  CHECK(b.evaluations == 0);
  noise->reset();
  auto fresh = make_perturbation(b, cfg);
  OptimizationRng reset_rng(19), fresh_rng(19);
  noise->sample(origin, delta, 2, reset_rng);
  fresh->sample(origin, other, 2, fresh_rng);
  CHECK(delta[0] == other[0]); CHECK(delta[1] == other[1]);
}

TEST_CASE(optimization_local_covariance_sparse_and_local) {
  auto cfg = config(R"({"algorithm":"wave","benchmark":"quadratic","dimensions":2,"perturbation":"local_covariance","covariance_learning_rate":1})");
  Benchmark b(cfg);
  auto noise = make_perturbation(b, cfg), fresh = make_perturbation(b, cfg);
  for (int i = 0; i < 3; ++i) noise->observe({{0, 0}, {1., 1.}, 1, 1});
  noise->observe({{0, 0}, {NAN, 0}, 1, 1});
  noise->observe({{0, 0}, {1, 1}, 0, 1});
  noise->update();
  float x[2] = {0, 0}, d[2], e[2];
  OptimizationRng a(5), c(5);
  noise->sample(x, d, 2, a); fresh->sample(x, e, 2, c);
  CHECK(d[0] == e[0]); CHECK(d[1] == e[1]);
  for (int i = 0; i < 40; ++i) {
    noise->observe({{-4, 0}, {1, 1}, 1, 1});
    noise->observe({{4, 0}, {1, -1}, 1, 1});
  }
  noise->update();
  double left = 0, right = 0;
  for (int i = 0; i < 3000; ++i) {
    x[0] = -4; noise->sample(x, d, 2, a); left += d[0] * d[1];
    x[0] = 4; noise->sample(x, d, 2, a); right += d[0] * d[1];
  }
  CHECK(left > 2000); CHECK(right < -2000);
  cfg.object["perturbation_std"] = number(0);
  noise = make_perturbation(b, cfg);
  noise->observe({{0, 0}, {1, 1}, 1, 1}); noise->update();
  noise->sample(x, d, 2, a);
  CHECK(d[0] == 0); CHECK(d[1] == 0);
}

class TransitionProbe final : public Perturbation {
 public:
  std::vector<PerturbationTransition> transitions;
  void sample(const float*, float* d, int n, fg::Rng&) const override {
    std::fill(d, d + n, 20.f);
  }
  void observe(const PerturbationTransition& t) override { transitions.push_back(t); }
};
TEST_CASE(optimization_transition_observation_and_replay) {
  static TransitionProbe* probe = nullptr;
  register_perturbation("transition_probe", "Transition probe",
    [](const Benchmark&, const Json&) {
      auto result = std::make_unique<TransitionProbe>(); probe = result.get(); return result;
    }, config("[]"));
  auto cfg = config(R"({"algorithm":"fmc","benchmark":"quadratic","dimensions":2,"periodic":true,"perturbation":"transition_probe"})");
  Benchmark b(cfg); Settings s(b.config); BenchmarkEnvironment env(b, s);
  std::vector<char> root; std::vector<float> obs;
  env.reset(root, obs);
  std::vector<std::vector<char>> next(1);
  std::vector<float> x(2), rewards(1);
  std::vector<uint8_t> done(1), truncated(1);
  env.step_batch({root}, {42}, {3}, next, x, rewards, done, truncated);
  CHECK(probe->transitions.size() == 1);
  CHECK(probe->transitions[0].origin == obs);
  CHECK(probe->transitions[0].draws == 3);
  CHECK(probe->transitions[0].displacement[0] == 60);
  CHECK_CLOSE(probe->transitions[0].improvement, rewards[0], 1e-4);
  env.collect_perturbations(false);
  const auto expected = next;
  env.step_batch({root}, {42}, {3}, next, x, rewards, done, truncated);
  CHECK(next == expected); CHECK(probe->transitions.size() == 1);
  env.collect_perturbations(true);
  env.step_batch({root}, {42}, {0}, next, x, rewards, done, truncated);
  CHECK(probe->transitions.size() == 1);
  env.s.periodic = false;
  env.step_batch({root}, {42}, {1}, next, x, rewards, done, truncated);
  CHECK(done[0]); CHECK(probe->transitions.size() == 1);
  // The registry keeps no reference to stack storage.
  probe = nullptr;
}
