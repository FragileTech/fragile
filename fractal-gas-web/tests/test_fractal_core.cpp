#include <array>

#include "backends/snapshot.hpp"
#include "fractal/graph.hpp"
#include "fractal/planner.hpp"
#include "fractal/wave.hpp"
#include "fractal_tree.hpp"
#include "mock_env.hpp"
#include "test_framework.hpp"
using namespace fg;
namespace {
struct InjectedCloning : FractalCloningOperator {
  mutable int draws = 0;
  void sample_companions_into(const std::vector<uint8_t>& alive, Rng&,
                              std::vector<int32_t>& out) const override {
    out.resize(alive.size());
    int offset = ++draws % 2 + 1;
    for (size_t i = 0; i < alive.size(); ++i) out[i] = (i + offset) % alive.size();
  }
  void sample_uniforms_into(int32_t n, Rng&, std::vector<float>& out) const override {
    out.assign(n, .125f);
  }
};
template <class A>
struct PackedMock {
  using Storage = std::vector<float>;
  using Info = WalkerInfo;
  using Action = A;
  using Batch = fractal::Population<Storage, Info, A>;
  void resize(Batch& s, int n) { s.states.resize(size_t(n) * 3); }
  void copy(const Batch& from, size_t i, Batch& to, size_t j) {
    std::copy_n(from.states.data() + i * 3, 3, to.states.data() + j * 3);
  }
  void observe(Batch&) {}
  void fitness_bonus(const Batch&, std::vector<float>&) {}
  void after_transition(const Batch&) {}
  template <class Input>
  void transition(const Input& from, fractal::Selection rows, const std::vector<A>& actions,
                  const std::vector<int32_t>& dt, Batch& to) {
    for (size_t j = 0; j < rows.count; ++j) {
      size_t i = rows.destinations.index(j), src = rows.sources.index(j);
      float a = float(actions[j]), d = float(dt[j]);
      float x = from.states[src * 3] + (a + 1) * d, y = from.states[src * 3 + 1] + .5f * d,
            z = from.states[src * 3 + 2] + 1;
      to.states[i * 3] = x;
      to.states[i * 3 + 1] = y;
      to.states[i * 3 + 2] = z;
      std::copy_n(to.states.data() + i * 3, 3, to.observations.data() + i * 3);
      to.step_rewards[i] = .1f * (a - 1) * d + .01f * x;
      to.dones[i] = x > 20;
      to.truncated[i] = to.recoverable[i] = 0;
      to.actual_dt[i] = dt[j];
    }
  }
  void pose(const Batch& s, size_t i, float* out) { std::copy_n(s.states.data() + i * 3, 3, out); }
  uint32_t flags(const Batch& s, size_t i) { return !s.alive(i); }
  int n_actions() const { return 4; }
  int action_dim() const { return 1; }
  int observation_dim() const { return 3; }
  ThreadPool* pool() { return nullptr; }
  bool has_infos() const { return false; }
  bool has_visit_key() const { return false; }
  VisitKey visit_key(const Info&) const { return {}; }
  std::array<float, 3> reset_root() { return {}; }
  void grow(Storage& s, int n) {
    s.resize(size_t(n) * 3, std::numeric_limits<float>::quiet_NaN());
  }
  void broadcast(const std::array<float, 3>&, Storage& s, int n) { s.assign(size_t(n) * 3, 0); }
  bool valid_slot(const Storage& s, size_t i) const { return std::isfinite(s[i * 3]); }
  void commit(Storage& from, size_t i, Storage& to, size_t j) {
    std::copy_n(from.data() + i * 3, 3, to.data() + j * 3);
  }
};
template <class A>
struct MockActions {
  RandomActionOperator sampler;
  template <class S>
  void sample(const S& s, const std::vector<int32_t>&, bool, std::vector<A>& out,
              std::vector<int32_t>& dt, Rng& rng) {
    auto sampled = sampler.sample_actions(s.N, 4, rng);
    out.assign(sampled.begin(), sampled.end());
    dt = sampler.sample_dt(s.N, rng);
  }
};
}  // namespace
TEST_CASE(shared_wave_matches_packed_float_and_opaque_integer_backends) {
  MockEnv env;
  VisitGrid visits;
  SnapshotBackend snapshot(env, visits);
  RandomActionOperator random;
  DiscreteActions discrete{env, random};
  fractal::Wave<WalkerState, SnapshotBackend, DiscreteActions> blobs(snapshot, discrete);
  PackedMock<float> packed;
  MockActions<float> continuous;
  fractal::Wave<PackedMock<float>::Batch, PackedMock<float>, MockActions<float>> flat(packed,
                                                                                      continuous);
  std::vector<char> initial;
  std::vector<float> obs;
  env.reset(initial, obs);
  blobs.reset(12, 3, 1, false);
  flat.reset(12, 3, 1, false);
  blobs.current.states.assign(12, initial);
  std::fill(flat.current.states.begin(), flat.current.states.end(), 0);
  blobs.begin_history(RecordingMode::Pruned, 3, obs.data(), {});
  flat.begin_history(RecordingMode::Pruned, 3, obs.data(), {});
  InjectedCloning a, b;
  a.use_cumulative_reward = b.use_cumulative_reward = false;
  Mt19937Rng r1(89), r2(89);
  for (int step = 0; step < 20; ++step) {
    blobs.step(3, a, r1);
    flat.step(3, b, r2);
    CHECK(a.draws == 2 * (step + 1));
    CHECK(blobs.sources == flat.sources);
    CHECK(blobs.current.rewards == flat.current.rewards);
    CHECK(blobs.current.observations == flat.current.observations);
    CHECK(blobs.current.virtual_rewards == flat.current.virtual_rewards);
    CHECK(blobs.current.lineage == flat.current.lineage);
    CHECK(blobs.current.dones == flat.current.dones);
    for (int i = 0; i < 12; ++i)
      CHECK(float(blobs.current.root_actions[i]) == flat.current.root_actions[i]);
    for (int i = 0; i < flat.elite.N; ++i) {
      const auto& e = flat.elite;
      CHECK_CLOSE(e.step_rewards[i],
                  .1f * (e.actions[i] - 1) * e.dt[i] + .01f * e.states[size_t(i) * 3], 1e-6);
      CHECK(e.observations[size_t(i) * 3] == e.states[size_t(i) * 3]);
      CHECK(blobs.elite.step_rewards[i] == e.step_rewards[i]);
    }
  }
}
TEST_CASE(shared_wave_reads_donor_cycles_and_inherits_root_actions_atomically) {
  PackedMock<float> backend;
  struct InheritedActions {
    void sample(const PackedMock<float>::Batch& s, const std::vector<int32_t>& donors, bool,
                std::vector<float>& actions, std::vector<int32_t>& dt, Rng&) {
      for (int i = 0; i < s.N; ++i) {
        actions[i] = s.actions[donors[i]];
        dt[i] = 1;
      }
    }
  } actions;
  fractal::Wave<PackedMock<float>::Batch, PackedMock<float>, InheritedActions> wave(backend,
                                                                                    actions);
  wave.reset(3, 3, 1, false);
  wave.current.states = {1, 0, 0, 2, 0, 0, 3, 0, 0};
  wave.current.actions = {0, 1, 2};
  wave.current.rewards = {10, 20, 30};
  wave.current.dones.assign(3, 1);
  float pose[3] = {};
  wave.begin_history(RecordingMode::Full, 3, pose, {});
  InjectedCloning cloning;
  Mt19937Rng rng(7);
  wave.step(0, cloning, rng);
  CHECK(wave.sources == std::vector<int32_t>({1, 2, 0}));
  CHECK(wave.current.root_actions == std::vector<float>({1, 2, 0}));
  CHECK(wave.current.states[0] == 4);
  CHECK(wave.current.states[3] == 6);
  CHECK(wave.current.states[6] == 2);
  CHECK_CLOSE(wave.current.rewards[0], 20.04f, 1e-6);
  auto parents = wave.current.lineage;
  wave.current.dones.assign(3, 1);
  wave.step(0, cloning, rng);
  CHECK(wave.current.root_actions == std::vector<float>({2, 0, 1}));
  CHECK(wave.tree.node(wave.current.lineage[0]).parent == parents[1]);
}
TEST_CASE(shared_wave_revives_only_recoverable_nontruncated_deaths_and_records_actual_duration) {
  struct EventBackend : PackedMock<float> {
    void transition(const Batch& from, fractal::Selection rows, const std::vector<float>& actions,
                    const std::vector<int32_t>& dt, Batch& to) {
      PackedMock<float>::transition(from, rows, actions, dt, to);
      to.dones.assign(to.N, 1);
      to.recoverable.assign(to.N, 1);
      to.truncated[1] = 1;
      to.recoverable[2] = 0;
      to.actual_dt = {1, 0, 1};
    }
  } backend;
  MockActions<float> actions;
  fractal::Wave<PackedMock<float>::Batch, EventBackend, MockActions<float>> wave(backend, actions);
  wave.reset(3, 3, 1, false);
  float pose[3] = {};
  wave.begin_history(RecordingMode::Full, 3, pose, {});
  FractalCloningOperator cloning;
  Mt19937Rng rng(7);
  wave.step(0, cloning, rng);
  CHECK(wave.metrics.revived == 1);
  CHECK(wave.current.alive(0));
  CHECK(!wave.current.alive(1));
  CHECK(!wave.current.alive(2));
  CHECK(wave.metrics.frames == 2);
  CHECK(wave.tree.node(wave.current.lineage[1]).frames == 0);
}
TEST_CASE(shared_wave_retains_prior_elites_before_current_rows_on_reward_ties) {
  struct ZeroReward : PackedMock<float> {
    void transition(const Batch& from, fractal::Selection rows, const std::vector<float>& actions,
                    const std::vector<int32_t>& dt, Batch& to) {
      PackedMock<float>::transition(from, rows, actions, dt, to);
      to.step_rewards.assign(to.N, 0);
      to.dones.assign(to.N, 0);
    }
  } backend;
  MockActions<float> actions;
  fractal::Wave<PackedMock<float>::Batch, ZeroReward, MockActions<float>> wave(backend, actions);
  wave.reset(4, 3, 1, false);
  float pose[3] = {};
  wave.begin_history(RecordingMode::Pruned, 3, pose, {});
  FractalCloningOperator cloning;
  Mt19937Rng rng(7);
  wave.step(2, cloning, rng);
  CHECK(wave.elite.lineage[0] == wave.current.lineage[0]);
  CHECK(wave.elite.lineage[1] == wave.current.lineage[1]);
  auto first = wave.elite.lineage;
  auto states = wave.elite.states;
  for (int i = 0; i < 10; ++i) wave.step(2, cloning, rng);
  CHECK(wave.elite.lineage == first);
  CHECK(wave.elite.states == states);
  for (auto id : first) CHECK(wave.tree.node(id).reward == 0);
}
TEST_CASE(shared_graph_matches_packed_and_snapshot_leaf_execution) {
  MockEnv env;
  VisitGrid unused;
  SnapshotGraphBackend opaque(env, unused);
  PackedMock<int32_t> packed;
  FractalTreeSampler a, b;
  Mt19937Rng r1(27), r2(27);
  fractal::GraphConfig c1, c2;
  c1.start_walkers = c2.start_walkers = 8;
  c1.max_walkers = c2.max_walkers = 32;
  c1.min_leafs = c2.min_leafs = 8;
  fractal::Graph<SnapshotGraphBackend, FractalTreeSampler> blobs(opaque, a, r1, c1);
  fractal::Graph<PackedMock<int32_t>, FractalTreeSampler> flat(packed, b, r2, c2);
  blobs.reset();
  flat.reset();
  for (int i = 0; i < 30; ++i) {
    auto x = blobs.step(), y = flat.step();
    CHECK(x.num_stepped == y.num_stepped);
    CHECK(x.n_walkers == y.n_walkers);
    CHECK(blobs.state_.parent == flat.state_.parent);
    CHECK(blobs.state_.cum_rewards == flat.state_.cum_rewards);
    CHECK(blobs.state_.observations == flat.state_.observations);
    CHECK(blobs.state_.will_clone == flat.state_.will_clone);
    CHECK(flat.state_.n <= 32);
    CHECK(blobs.total_frames_ == flat.total_frames_);
  }
}
TEST_CASE(snapshot_backend_scatter_preserves_unselected_rows) {
  MockEnv env;
  VisitGrid visits;
  SnapshotBackend backend(env, visits);
  WalkerState input, output;
  input.resize_metadata(4, 3, 1);
  output.resize_metadata(4, 3, 1);
  backend.resize(input, 4);
  backend.resize(output, 4);
  std::vector<char> initial;
  std::vector<float> obs;
  env.reset(initial, obs);
  input.states.assign(4, initial);
  output.states.assign(4, initial);
  std::vector<int32_t> sources{0, 2}, destinations{3, 1};
  backend.transition(input, fractal::Selection{2, {sources.data(), 2}, {destinations.data(), 2}},
                     {1, 2}, {1, 2}, output);
  CHECK(output.states[0] == initial);
  CHECK(output.states[2] == initial);
  CHECK(output.observations[9] == 2);
  CHECK(output.observations[3] == 6);
}
TEST_CASE(shared_planner_finish_is_atomic_and_does_not_advance_again) {
  PackedMock<float> backend;
  MockActions<float> actions;
  fractal::Wave<PackedMock<float>::Batch, PackedMock<float>, MockActions<float>> wave(backend,
                                                                                      actions);
  wave.reset(4, 3, 1, false);
  float pose[3] = {};
  wave.begin_history(RecordingMode::Full, 3, pose, {});
  FractalCloningOperator cloning;
  Mt19937Rng rng(7);
  fractal::Planner<float> planner;
  fractal::PlannerSettings settings;
  settings.horizon = 12;
  settings.selection = fractal::RootSelection::PopulationMean;
  planner.begin(settings);
  auto advance = [&] { wave.step(0, cloning, rng); };
  planner.advance(advance, wave.current, wave.tree, {0.f});
  CHECK(planner.depth == 1);
  CHECK(!planner.result.ready);
  planner.finish(wave.current, wave.tree, {0.f});
  auto selected = planner.result.actions;
  planner.advance(advance, wave.current, wave.tree, {0.f});
  CHECK(planner.depth == 1);
  CHECK(planner.result.actions == selected);
  CHECK(wave.metrics.iteration == 1);
}
