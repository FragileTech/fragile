// FractalTree ("Graph") tests: replay of the HISTORICAL Python reference
// (cb9f3296, recorded by tests/fixtures/generate_tree_fixtures.py) plus
// structural property tests with the production sampler.
#include <cmath>
#include <deque>
#include <memory>
#include <vector>

#include "fixtures/fixtures_tree_generated.hpp"
#include "fractal_tree.hpp"
#include "mock_env.hpp"
#include "test_framework.hpp"
#include "visit_mock_env.hpp"

using namespace fg;

namespace {

/// Sampler that replays the draws recorded from the Python reference.
class ReplayTreeSampler final : public FractalTreeSampler {
 public:
  mutable std::deque<std::vector<int32_t>> companions_queue;
  mutable std::deque<std::vector<float>> uniforms_queue;
  mutable std::deque<std::vector<int32_t>> actions_queue;
  mutable std::deque<std::vector<int32_t>> dt_queue;

  void sample_companions_into(const std::vector<uint8_t>&, Rng&,
                              std::vector<int32_t>& v) const override {
    v = companions_queue.front();
    companions_queue.pop_front();
  }
  void sample_uniforms_into(int32_t, Rng&, std::vector<float>& v) const override {
    v = uniforms_queue.front();
    uniforms_queue.pop_front();
  }
  void sample_actions_into(int32_t, int32_t, Rng&, std::vector<int32_t>& v) const override {
    v = actions_queue.front();
    actions_queue.pop_front();
  }
  void sample_dt_into(int32_t, int32_t, int32_t, Rng&, std::vector<int32_t>& v) const override {
    v = dt_queue.front();
    dt_queue.pop_front();
  }
};

struct ReplayCase {
  int start, min_leafs, max_walkers, iters;
  const std::vector<int32_t>* reset_actions;
  const std::vector<int32_t>* reset_dts;
  const std::vector<std::vector<int32_t>>* fit;
  const std::vector<std::vector<int32_t>>* clone;
  const std::vector<std::vector<float>>* uniforms;
  const std::vector<std::vector<int32_t>>* actions;
  const std::vector<std::vector<int32_t>>* dts;
  const std::vector<int32_t>* n_before;
  const std::vector<int32_t>* n_after;
  const std::vector<int32_t>* leaves;
  const std::vector<int32_t>* stepped;
  const std::vector<int32_t>* best;
  const std::vector<std::vector<int32_t>>* parent;
  const std::vector<std::vector<int32_t>>* leaf;
  const std::vector<std::vector<int32_t>>* oobs;
  const std::vector<std::vector<int32_t>>* will;
  const std::vector<std::vector<int32_t>>* clone_ix;
  const std::vector<std::vector<float>>* cum;
  const std::vector<std::vector<float>>* vr;
  const std::vector<float>* final_obs;
  int64_t py_total_steps;
  const std::vector<std::vector<float>>* other;  // may be null
};

void run_replay(BatchEnv& env, const ReplayCase& c, bool count_visits,
                const FractalTree** tree_out = nullptr,
                std::unique_ptr<FractalTree>* keep = nullptr) {
  auto sampler = std::make_unique<ReplayTreeSampler>();
  sampler->actions_queue.push_back(*c.reset_actions);
  sampler->dt_queue.push_back(*c.reset_dts);
  for (int it = 0; it < c.iters; ++it) {
    const auto ui = static_cast<size_t>(it);
    // Per step: fitness companions, clone companions, uniforms, then the
    // actions/dt of the stepped walkers (only when some stepped).
    sampler->companions_queue.push_back((*c.fit)[ui]);
    sampler->companions_queue.push_back((*c.clone)[ui]);
    sampler->uniforms_queue.push_back((*c.uniforms)[ui]);
    if ((*c.stepped)[ui] > 0) {
      sampler->actions_queue.push_back((*c.actions)[ui]);
      sampler->dt_queue.push_back((*c.dts)[ui]);
    }
  }

  FractalTreeParams params;
  params.start_walkers = c.start;
  params.min_leafs = c.min_leafs;
  params.max_walkers = c.max_walkers;
  params.dt_min = 1;
  params.dt_max = 4;
  params.count_visits = count_visits;
  params.erase_coef = 0.05f;
  params.agg_block_size = 5;
  auto tree = std::make_unique<FractalTree>(env, params, std::make_unique<Mt19937Rng>(0),
                                            std::move(sampler));
  tree->reset();
  CHECK(tree->n_walkers() == c.start);
  CHECK(tree->walker_parent(0) == 0);
  CHECK(!tree->walker_alive(0));  // the root stays oobs after reset

  int64_t stepped_sum = 0;
  for (int it = 0; it < c.iters; ++it) {
    const auto ui = static_cast<size_t>(it);
    CHECK(tree->n_walkers() == (*c.n_before)[ui]);
    const int32_t n_before = tree->n_walkers();
    const StepInfo info = tree->step();
    const TreeState& s = tree->state();
    CHECK(info.iteration == it + 1);
    CHECK(info.n_walkers == (*c.n_after)[ui]);
    CHECK(info.n_leaves == (*c.leaves)[ui]);
    CHECK(info.num_stepped == (*c.stepped)[ui]);
    CHECK(info.num_cloned == (*c.stepped)[ui]);
    stepped_sum += info.num_stepped;
    CHECK(tree->get_best_walker().first == (*c.best)[ui]);
    if (c.other != nullptr) {
      const auto& other = (*c.other)[ui];
      CHECK(static_cast<int32_t>(other.size()) == n_before);
      for (int32_t i = 0; i < n_before; ++i) {
        CHECK_CLOSE(s.other_rewards[static_cast<size_t>(i)], other[static_cast<size_t>(i)], 1e-5);
      }
    }
    const int32_t n_after = tree->n_walkers();
    for (int32_t i = 0; i < n_after; ++i) {
      const auto wi = static_cast<size_t>(i);
      CHECK(s.parent[wi] == (*c.parent)[ui][wi]);
      CHECK(static_cast<int32_t>(s.is_leaf[wi]) == (*c.leaf)[ui][wi]);
      CHECK(static_cast<int32_t>(s.oobs[wi]) == (*c.oobs)[ui][wi]);
      CHECK_CLOSE(s.cum_rewards[wi], (*c.cum)[ui][wi], 1e-5);
      CHECK_CLOSE(s.virtual_rewards[wi], (*c.vr)[ui][wi], 1e-5);
    }
    for (int32_t i = 0; i < n_before; ++i) {
      const auto wi = static_cast<size_t>(i);
      CHECK(static_cast<int32_t>(s.will_clone[wi]) == (*c.will)[ui][wi]);
      CHECK(s.clone_ix[wi] == (*c.clone_ix)[ui][wi]);
    }
  }
  const TreeState& s = tree->state();
  CHECK(s.observations.size() == c.final_obs->size());
  for (size_t i = 0; i < s.observations.size() && i < c.final_obs->size(); ++i) {
    CHECK_CLOSE(s.observations[i], (*c.final_obs)[i], 1e-5);
  }
  // The reference counts the fresh slots' preallocated will_clone flags;
  // the port counts the walkers that really stepped.
  CHECK(tree->total_steps() == stepped_sum);
  CHECK(stepped_sum <= c.py_total_steps);
  if (tree_out != nullptr) *tree_out = tree.get();
  if (keep != nullptr) *keep = std::move(tree);
}

}  // namespace

TEST_CASE(tree_run_replays_python_reference) {
  MockEnv env;
  const ReplayCase c{fixtures::kTreeStart,
                     fixtures::kTreeMinLeafs,
                     fixtures::kTreeMaxWalkers,
                     fixtures::kTreeIters,
                     &fixtures::kTreeResetActions,
                     &fixtures::kTreeResetDts,
                     &fixtures::kTreeFitCompanions,
                     &fixtures::kTreeCloneCompanions,
                     &fixtures::kTreeUniforms,
                     &fixtures::kTreeActions,
                     &fixtures::kTreeDts,
                     &fixtures::kTreeExpectedNBefore,
                     &fixtures::kTreeExpectedNAfter,
                     &fixtures::kTreeExpectedLeaves,
                     &fixtures::kTreeExpectedStepped,
                     &fixtures::kTreeExpectedBest,
                     &fixtures::kTreeExpectedParent,
                     &fixtures::kTreeExpectedLeaf,
                     &fixtures::kTreeExpectedOobs,
                     &fixtures::kTreeExpectedWillClone,
                     &fixtures::kTreeExpectedCloneIx,
                     &fixtures::kTreeExpectedCum,
                     &fixtures::kTreeExpectedVr,
                     &fixtures::kTreeFinalObservations,
                     fixtures::kTreePyTotalSteps,
                     nullptr};
  run_replay(env, c, /*count_visits=*/true);  // MockEnv has no visit key -> off
}

TEST_CASE(tree_visits_run_replays_python_reference) {
  VisitMockEnv env;
  const ReplayCase c{fixtures::kTVStart,
                     fixtures::kTVMinLeafs,
                     fixtures::kTVMaxWalkers,
                     fixtures::kTVIters,
                     &fixtures::kTVResetActions,
                     &fixtures::kTVResetDts,
                     &fixtures::kTVFitCompanions,
                     &fixtures::kTVCloneCompanions,
                     &fixtures::kTVUniforms,
                     &fixtures::kTVActions,
                     &fixtures::kTVDts,
                     &fixtures::kTVExpectedNBefore,
                     &fixtures::kTVExpectedNAfter,
                     &fixtures::kTVExpectedLeaves,
                     &fixtures::kTVExpectedStepped,
                     &fixtures::kTVExpectedBest,
                     &fixtures::kTVExpectedParent,
                     &fixtures::kTVExpectedLeaf,
                     &fixtures::kTVExpectedOobs,
                     &fixtures::kTVExpectedWillClone,
                     &fixtures::kTVExpectedCloneIx,
                     &fixtures::kTVExpectedCum,
                     &fixtures::kTVExpectedVr,
                     &fixtures::kTVFinalObservations,
                     fixtures::kTVPyTotalSteps,
                     &fixtures::kTVExpectedOther};
  std::unique_ptr<FractalTree> tree;
  run_replay(env, c, /*count_visits=*/true, nullptr, &tree);
  CHECK(tree->counting_visits());
  // The final visit grid equals the reference's nonzero cells exactly.
  const VisitGrid& grid = tree->visits();
  CHECK(grid.nonzero_cells() == fixtures::kTVFinalCellValue.size());
  for (size_t i = 0; i < fixtures::kTVFinalCellValue.size(); ++i) {
    const VisitKey key{fixtures::kTVFinalCellRoom[i], fixtures::kTVFinalCellX[i],
                       fixtures::kTVFinalCellY[i]};
    CHECK_CLOSE(grid.cell(key), fixtures::kTVFinalCellValue[i], 1e-6);
  }
}

TEST_CASE(tree_same_seed_is_deterministic) {
  FractalTreeParams params;
  params.start_walkers = 5;
  params.min_leafs = 5;
  params.max_walkers = 40;
  params.seed = 11;
  std::vector<float> a, b;
  for (int rep = 0; rep < 2; ++rep) {
    MockEnv env;
    FractalTree tree(env, params);
    tree.reset();
    for (int it = 0; it < 12; ++it) tree.step();
    auto& out = rep == 0 ? a : b;
    out = tree.state().cum_rewards;
    out.push_back(static_cast<float>(tree.n_walkers()));
  }
  CHECK(a == b);
}

TEST_CASE(tree_invariants_hold_with_production_sampler) {
  // The visit mock env: walkers die only occasionally (x wraps around), so
  // the tree keeps branching and growing instead of dying out like MockEnv.
  VisitMockEnv env;
  FractalTreeParams params;
  params.start_walkers = 4;
  params.min_leafs = 6;
  params.max_walkers = 30;
  params.dt_min = 1;
  params.dt_max = 4;
  params.seed = 3;
  FractalTree tree(env, params);
  tree.reset();
  CHECK(tree.n_walkers() == 4);
  float best_reward = tree.get_best_walker().second;
  for (int it = 0; it < 40; ++it) {
    // Snapshot the non-leaves: interior nodes never change.
    const TreeState before = tree.state();
    const StepInfo info = tree.step();
    const TreeState& s = tree.state();
    CHECK(s.n <= params.max_walkers);
    CHECK(s.n >= before.n);
    CHECK(info.n_leaves >= 1);
    CHECK(s.parent[0] == 0);
    CHECK(!s.is_leaf[0]);  // the root is its own parent, never a leaf
    for (int32_t i = 0; i < s.n; ++i) {
      const auto wi = static_cast<size_t>(i);
      CHECK(s.parent[wi] >= 0 && s.parent[wi] < s.n);
      if (s.will_clone[wi]) {
        CHECK(s.dt[wi] >= params.dt_min && s.dt[wi] <= params.dt_max);
        CHECK(s.parent[wi] == s.clone_ix[wi]);
      }
    }
    for (int32_t i = 0; i < before.n; ++i) {
      const auto wi = static_cast<size_t>(i);
      if (before.is_leaf[wi] || s.will_clone[wi]) continue;
      // Not a leaf when the step started and did not clone: frozen.
      CHECK(s.cum_rewards[wi] == before.cum_rewards[wi]);
      CHECK(s.states[wi] == before.states[wi]);
      CHECK(s.parent[wi] == before.parent[wi]);
    }
    // Growth tops the leaves up to min_leafs (an iteration with no cloning
    // candidate returns early without growing, like the reference).
    if (s.n > before.n) {
      int32_t leaves = 0;
      for (int32_t i = 0; i < s.n; ++i) leaves += s.is_leaf[static_cast<size_t>(i)];
      CHECK(leaves >= params.min_leafs);
      CHECK(s.n <= params.max_walkers);
    }
    // The best walker is protected: it never clones away, so the maximum
    // cumulative reward never decreases.
    CHECK(tree.get_best_walker().second >= best_reward - 1e-6f);
    best_reward = tree.get_best_walker().second;
    CHECK(info.n_walkers == s.n);
    CHECK(info.num_stepped <= before.n);
  }
  // Keep going until the cap is reached: it must hold and never be crossed.
  for (int it = 0; it < 400 && tree.n_walkers() < params.max_walkers; ++it) tree.step();
  CHECK(tree.n_walkers() == params.max_walkers);
  for (int it = 0; it < 20; ++it) tree.step();
  CHECK(tree.n_walkers() == params.max_walkers);  // the cap holds
  CHECK(tree.total_steps() == tree.total_clones());
}

TEST_CASE(tree_visit_reward_only_when_env_has_key) {
  MockEnv plain;
  VisitMockEnv keyed;
  FractalTreeParams params;
  params.start_walkers = 4;
  params.min_leafs = 4;
  params.max_walkers = 16;
  params.count_visits = true;
  FractalTree a(plain, params);
  FractalTree b(keyed, params);
  CHECK(!a.counting_visits());
  CHECK(b.counting_visits());
  params.count_visits = false;
  FractalTree c(keyed, params);
  CHECK(!c.counting_visits());
  b.reset();
  CHECK(b.visits().nonzero_cells() > 0);  // reset counts every walker
  b.step();
  for (int32_t i = 0; i < b.n_walkers(); ++i) {
    CHECK(std::isfinite(b.state().other_rewards[static_cast<size_t>(i)]));
  }
}

namespace {
/// Every step kills the walker (done always true), so the whole population
/// dies and companions fall back to arange.
class KillerEnv final : public BatchEnv {
 public:
  int32_t n_actions() const override { return 2; }
  int32_t obs_dim() const override { return 1; }
  void reset(std::vector<char>& state, std::vector<float>& obs) override {
    state.assign(4, 0);
    obs.assign(1, 0.0f);
  }
  void step_batch(const std::vector<std::vector<char>>& states, const std::vector<int32_t>&,
                  const std::vector<int32_t>&, std::vector<std::vector<char>>& new_states,
                  std::vector<float>& observations, std::vector<float>& rewards,
                  std::vector<uint8_t>& dones, std::vector<uint8_t>& truncated) override {
    for (size_t i = 0; i < states.size(); ++i) {
      CHECK(!states[i].empty());  // never step an empty blob
      new_states[i] = states[i];
      observations[i] = static_cast<float>(i);
      rewards[i] = 1.0f;
      dones[i] = 1;
      truncated[i] = 0;
    }
  }
  void render_frame(const std::vector<char>&, std::vector<uint8_t>& rgba) override {
    rgba.clear();
  }
  int32_t frame_width() const override { return 0; }
  int32_t frame_height() const override { return 0; }
};
}  // namespace

TEST_CASE(tree_all_dead_does_not_step_empty_states) {
  KillerEnv env;
  FractalTreeParams params;
  params.start_walkers = 3;
  params.min_leafs = 5;
  params.max_walkers = 12;
  params.seed = 5;
  FractalTree tree(env, params);
  tree.reset();
  for (int it = 0; it < 20; ++it) {
    const StepInfo info = tree.step();
    CHECK(info.alive_count == 0);
    CHECK(info.n_walkers <= params.max_walkers);
  }
}

TEST_CASE(tree_visit_reward_switch_is_live_and_keeps_counting) {
  VisitMockEnv env;
  FractalTreeParams params;
  params.start_walkers = 6;
  params.min_leafs = 6;
  params.max_walkers = 40;
  params.seed = 9;
  FractalTree tree(env, params);
  tree.reset();
  CHECK(tree.counting_visits() && tree.visit_reward_on());
  for (int it = 0; it < 5; ++it) tree.step();
  // With the term on, the walkers' other_rewards are not all one.
  bool any_non_one = false;
  for (int32_t i = 0; i < tree.n_walkers(); ++i) {
    any_non_one = any_non_one || tree.state().other_rewards[static_cast<size_t>(i)] != 1.0f;
  }
  CHECK(any_non_one);
  const size_t cells_before = tree.visits().nonzero_cells();
  // Off: the term is exactly one for every walker, and the grid still grows.
  tree.set_visit_reward(false);
  CHECK(!tree.visit_reward_on());
  for (int it = 0; it < 5; ++it) {
    tree.step();
    for (int32_t i = 0; i < tree.n_walkers(); ++i) {
      CHECK(tree.state().other_rewards[static_cast<size_t>(i)] == 1.0f);
    }
  }
  CHECK(tree.visits().nonzero_cells() >= cells_before / 2);  // decays but keeps counting
  CHECK(tree.visits().nonzero_cells() > 0);
  // Back on: the term returns using the accumulated history.
  tree.set_visit_reward(true);
  tree.step();
  any_non_one = false;
  for (int32_t i = 0; i < tree.n_walkers(); ++i) {
    any_non_one = any_non_one || tree.state().other_rewards[static_cast<size_t>(i)] != 1.0f;
  }
  CHECK(any_non_one);
}

TEST_CASE(tree_visit_coef_scales_the_visit_term) {
  auto vr_after = [](float coef, bool on) {
    VisitMockEnv env;
    FractalTreeParams params;
    params.start_walkers = 6;
    params.min_leafs = 6;
    params.max_walkers = 40;
    params.seed = 13;
    params.visit_reward = on;
    params.visit_coef = coef;
    FractalTree tree(env, params);
    tree.reset();
    for (int it = 0; it < 6; ++it) tree.step();
    return tree.state().virtual_rewards;
  };
  // Exponent 0 removes the term exactly like the switch does.
  CHECK(vr_after(0.0f, true) == vr_after(1.0f, false));
  // Exponent 1 is the reference product; 2 changes the dynamics.
  CHECK(vr_after(1.0f, true) != vr_after(2.0f, true));
  CHECK(vr_after(1.0f, true) != vr_after(0.0f, true));
}

TEST_CASE(tree_total_frames_counts_stepped_walkers_only) {
  MockEnv env;
  FractalTreeParams params;
  params.start_walkers = 6;
  params.min_leafs = 6;
  params.max_walkers = 40;
  params.seed = 8;
  FractalTree tree(env, params);
  tree.reset();
  // reset() steps every start walker once.
  int64_t expected = 0;
  for (int32_t i = 0; i < tree.n_walkers(); ++i)
    expected += tree.state().dt[static_cast<size_t>(i)];
  CHECK(tree.total_frames() == expected);
  for (int it = 0; it < 12; ++it) {
    tree.step();
    const TreeState& s = tree.state();
    for (int32_t i = 0; i < s.n; ++i) {
      if (s.will_clone[static_cast<size_t>(i)]) expected += s.dt[static_cast<size_t>(i)];
    }
    CHECK(tree.total_frames() == expected);
  }
  CHECK(tree.total_frames() > tree.total_steps());  // dt >= 1, mostly > 1
}
