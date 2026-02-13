"""End-to-end tests for AtariTreeHistory with AtariFractalGas."""

import tempfile

import numpy as np
import pytest
import torch

from fragile.fractalai.core.tree import DEFAULT_ROOT_ID
from fragile.fractalai.videogames.atari_gas import AtariFractalGas, WalkerState
from fragile.fractalai.videogames.atari_history import AtariHistory
from fragile.fractalai.videogames.atari_tree_history import _node_id, AtariTreeHistory


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------


class MockAtariState:
    """Mock AtariState with rgb_frame attribute."""

    def __init__(self, data: np.ndarray, rgb_frame: np.ndarray | None = None):
        self._data = data
        self.rgb_frame = rgb_frame

    def copy(self):
        return MockAtariState(
            self._data.copy(),
            self.rgb_frame.copy() if self.rgb_frame is not None else None,
        )


class MockEnv:
    """Mock plangym environment returning a 3-tuple from reset()."""

    def __init__(self, obs_shape=(128,), action_space_size=18):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset returning (state, observation, info)."""
        self.step_count = 0
        state = MockAtariState(
            np.zeros(4, dtype=np.float32),
            rgb_frame=np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8),
        )
        observation = np.zeros(self.obs_shape, dtype=np.float32)
        info = {}
        return state, observation, info

    def sample_action(self):
        return np.random.randint(0, self.action_space_size)

    def step_batch(self, states, actions, dt, **kwargs):
        """Mock batch stepping — accepts **kwargs for return_state etc."""
        N = len(states)
        self.step_count += N

        new_states = np.array(
            [
                MockAtariState(
                    np.random.randn(4).astype(np.float32),
                    rgb_frame=np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8),
                )
                for _ in range(N)
            ],
            dtype=object,
        )
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.random.randn(N).astype(np.float32) * 0.1
        dones = np.zeros(N, dtype=bool)
        # Kill ~10% of walkers each step
        n_dead = max(1, N // 10)
        dones[np.random.choice(N, size=n_dead, replace=False)] = True
        truncated = np.zeros(N, dtype=bool)
        infos = [{"step": i} for i in range(N)]

        return new_states, observations, rewards, dones, truncated, infos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_WALKERS = 10
MAX_ITER = 20


@pytest.fixture
def mock_env():
    return MockEnv(obs_shape=(128,))


@pytest.fixture
def gas(mock_env):
    return AtariFractalGas(
        env=mock_env,
        N=N_WALKERS,
        device="cpu",
        seed=42,
    )


@pytest.fixture
def tree(gas):
    """Run a short simulation and return the AtariTreeHistory."""
    return gas.run_with_tree(max_iterations=MAX_ITER, task_label="MockPacman")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunWithTree:
    """Tests for AtariFractalGas.run_with_tree integration."""

    def test_run_with_tree_completes(self, tree):
        assert isinstance(tree, AtariTreeHistory)
        assert tree._n_recorded > 1

    def test_graph_node_count(self, tree):
        tree._n_recorded - 1  # subtract initial state
        expected = 1 + N_WALKERS * tree._n_recorded  # root + N per recorded step
        actual = tree._tree.data.number_of_nodes()
        assert actual == expected, f"Expected {expected} nodes, got {actual}"

    def test_cloning_edges(self, tree):
        """Cloned walkers must have parent pointing to the companion's previous node."""
        nodes = tree._tree.data.nodes
        found_clone = False
        for step in range(1, tree._n_recorded):
            for w in range(N_WALKERS):
                nid = _node_id(step, w, N_WALKERS)
                if nid not in nodes:
                    continue
                node = nodes[nid]
                if node.get("will_clone"):
                    found_clone = True
                    companion = node["clone_companion"]
                    expected_parent = _node_id(step - 1, companion, N_WALKERS)
                    # Check the actual parent edge
                    parents = list(tree._tree.data.predecessors(nid))
                    assert len(parents) == 1
                    assert parents[0] == expected_parent, (
                        f"Step {step}, walker {w}: expected parent "
                        f"{expected_parent} (companion {companion}), got {parents[0]}"
                    )
        # With 20 iterations and 10 walkers, cloning should happen at least once
        assert found_clone, "No cloning events found in the tree"

    def test_per_walker_data(self, tree):
        """Node attributes should be retrievable for any walker at any step."""
        nodes = tree._tree.data.nodes
        # Check a node from step 1
        nid = _node_id(1, 0, N_WALKERS)
        assert nid in nodes
        node = nodes[nid]
        assert "observations" in node
        assert "reward" in node
        assert "alive" in node
        assert isinstance(node["observations"], np.ndarray)
        assert isinstance(node["reward"], float)
        assert isinstance(node["alive"], bool)

    def test_env_state_stored_in_nodes(self, tree):
        """Every node should have an env_state field."""
        nodes = tree._tree.data.nodes
        for step in range(tree._n_recorded):
            for w in range(N_WALKERS):
                nid = _node_id(step, w, N_WALKERS)
                assert nid in nodes
                assert "env_state" in nodes[nid], f"Node (step={step}, w={w}) missing env_state"

    def test_frame_stored_for_all_walkers(self, tree):
        """When states have rgb_frame, every node should have a non-None frame."""
        nodes = tree._tree.data.nodes
        for step in range(tree._n_recorded):
            for w in range(N_WALKERS):
                nid = _node_id(step, w, N_WALKERS)
                assert "frame" in nodes[nid], f"Node (step={step}, w={w}) missing frame field"
                assert (
                    nodes[nid]["frame"] is not None
                ), f"Node (step={step}, w={w}) has frame=None despite rgb_frame"

    def test_get_best_path_frames_all_filled(self, tree):
        """Path frames should all be non-None without needing render_missing_path_frames."""
        frames = tree.get_best_path_frames()
        assert len(frames) > 0
        assert all(f is not None for f in frames), "Expected all path frames to be non-None"


class TestDashboardProperties:
    """Tests for dashboard-compatible read properties."""

    def test_iterations_length(self, tree):
        n_iter = tree._n_recorded - 1
        assert len(tree.iterations) == n_iter
        assert tree.iterations == list(range(n_iter))

    def test_rewards_max_length(self, tree):
        assert len(tree.rewards_max) == len(tree.iterations)
        assert all(isinstance(v, float) for v in tree.rewards_max)

    def test_rewards_mean_length(self, tree):
        assert len(tree.rewards_mean) == len(tree.iterations)

    def test_alive_counts_length(self, tree):
        assert len(tree.alive_counts) == len(tree.iterations)
        assert all(isinstance(v, int) for v in tree.alive_counts)

    def test_num_cloned_length(self, tree):
        assert len(tree.num_cloned) == len(tree.iterations)

    def test_virtual_rewards(self, tree):
        assert len(tree.virtual_rewards_mean) == len(tree.iterations)
        assert len(tree.virtual_rewards_max) == len(tree.iterations)

    def test_best_rewards_and_indices(self, tree):
        assert len(tree.best_rewards) == len(tree.iterations)
        assert len(tree.best_indices) == len(tree.iterations)


class TestConversion:
    """Tests for AtariHistory conversion."""

    def test_to_atari_history(self, tree):
        ah = tree.to_atari_history()
        assert isinstance(ah, AtariHistory)
        assert ah.N == N_WALKERS
        assert ah.game_name == "MockPacman"
        assert ah.max_iterations == len(tree.iterations)
        assert len(ah.rewards_max) == len(tree.rewards_max)
        assert len(ah.alive_counts) == len(tree.alive_counts)

    def test_to_atari_history_metrics_match(self, tree):
        ah = tree.to_atari_history()
        assert ah.rewards_max == tree.rewards_max
        assert ah.rewards_mean == tree.rewards_mean
        assert ah.alive_counts == tree.alive_counts
        assert ah.num_cloned == tree.num_cloned


class TestVisualizerCompatible:
    """Test that the converted history works with AtariGasVisualizer."""

    def test_visualizer_set_history(self, tree):
        """AtariGasVisualizer.set_history should accept the converted history."""
        pn = pytest.importorskip("panel")
        hv = pytest.importorskip("holoviews")
        hv.extension("bokeh")
        pn.extension()
        from fragile.fractalai.videogames.dashboard import AtariGasVisualizer

        ah = tree.to_atari_history()
        viz = AtariGasVisualizer()
        # Should not raise
        viz.set_history(ah)
        assert viz.history is ah


class TestWalkerBranch:
    """Tests for walker lineage traversal."""

    def test_walker_branch_traversal(self, tree):
        branch = tree.get_walker_branch(0)
        assert isinstance(branch, list)
        assert len(branch) >= 2  # at least root -> initial node
        # First node should be root
        assert branch[0] == 0  # DEFAULT_ROOT_ID
        # Last node should be the final-step node for walker 0
        last_nid = _node_id(tree._n_recorded - 1, 0, N_WALKERS)
        assert branch[-1] == last_nid

    def test_all_walkers_have_branches(self, tree):
        for w in range(N_WALKERS):
            branch = tree.get_walker_branch(w)
            assert len(branch) >= 2


class TestPathFrames:
    """Tests for path-based frame reconstruction."""

    def test_get_best_walker_branch_returns_path(self, tree):
        branch = tree.get_best_walker_branch()
        assert isinstance(branch, list)
        assert len(branch) >= 2

    def test_get_best_path_frames_length(self, tree):
        """Path frames should have one entry per node (excluding root)."""
        branch = tree.get_best_walker_branch()
        frames = tree.get_best_path_frames()
        # Branch includes root, frames excludes it
        assert len(frames) == len(branch) - 1

    def test_render_missing_path_frames(self):
        """render_missing_path_frames should call render_fn for nodes with
        env_state but no frame, and store the result."""
        N = 4
        tree = AtariTreeHistory(N=N, game_name="render_test")

        obs = torch.zeros(N, 8)
        state = WalkerState(
            states=np.array([np.ones(4, dtype=np.float32) * i for i in range(N)], dtype=object),
            observations=obs,
            rewards=torch.zeros(N),
            step_rewards=torch.zeros(N),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.zeros(N),
        )
        tree.record_initial_atari_state(state)

        # Step 1 — no frames, no cloning
        state1 = WalkerState(
            states=np.array(
                [np.ones(4, dtype=np.float32) * (i + 10) for i in range(N)], dtype=object
            ),
            observations=torch.randn(N, 8),
            rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            step_rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        tree.record_atari_step(
            state_before=state,
            state_after_clone=state1,
            state_final=state1,
            info={},
            clone_companions=torch.arange(N),
            will_clone=torch.zeros(N, dtype=torch.bool),
            best_frame=None,  # no pre-rendered frame
        )

        # All frames should be None along best path before rendering
        frames_before = tree.get_best_path_frames()
        assert all(f is None for f in frames_before)

        # Render missing frames with a dummy render function
        render_calls = []

        def dummy_render(env_state):
            render_calls.append(env_state)
            return np.ones((4, 4, 3), dtype=np.uint8) * 42

        frames_after = tree.render_missing_path_frames(dummy_render)

        # Should have called render for each node on the path
        assert len(render_calls) > 0
        # All frames should now be filled in
        assert all(f is not None for f in frames_after)
        assert frames_after[0].shape == (4, 4, 3)

    def test_render_preserves_existing_frames(self):
        """Nodes that already have a frame should not be re-rendered."""
        N = 3
        tree = AtariTreeHistory(N=N, game_name="preserve_test")

        obs = torch.zeros(N, 8)
        state = WalkerState(
            states=np.array([np.ones(4, dtype=np.float32) * i for i in range(N)], dtype=object),
            observations=obs,
            rewards=torch.zeros(N),
            step_rewards=torch.zeros(N),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.zeros(N),
        )
        tree.record_initial_atari_state(state)

        existing_frame = np.ones((4, 4, 3), dtype=np.uint8) * 99
        state1 = WalkerState(
            states=np.array(
                [np.ones(4, dtype=np.float32) * (i + 10) for i in range(N)], dtype=object
            ),
            observations=torch.randn(N, 8),
            rewards=torch.tensor([1.0, 5.0, 3.0]),  # walker 1 is best
            step_rewards=torch.tensor([1.0, 5.0, 3.0]),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.tensor([1.0, 5.0, 3.0]),
        )
        tree.record_atari_step(
            state_before=state,
            state_after_clone=state1,
            state_final=state1,
            info={},
            clone_companions=torch.arange(N),
            will_clone=torch.zeros(N, dtype=torch.bool),
            best_frame=existing_frame,  # best walker (1) gets this frame
        )

        render_calls = []

        def dummy_render(env_state):
            render_calls.append(env_state)
            return np.ones((4, 4, 3), dtype=np.uint8) * 77

        frames = tree.render_missing_path_frames(dummy_render)

        # The step-1 node for the best walker already has a frame,
        # so render should NOT be called for it
        # (render_calls should only be for the initial-state node)
        for f in frames:
            assert f is not None

        # Check the existing frame was preserved (last frame on path is step 1)
        # Best walker is 1, so last frame in path should be the existing_frame
        np.testing.assert_array_equal(frames[-1], existing_frame)

    def test_to_atari_history_uses_path_frames(self):
        """Converted history should use path-based frames when available."""
        N = 3
        tree = AtariTreeHistory(N=N, game_name="path_convert")

        obs = torch.zeros(N, 8)
        state = WalkerState(
            states=np.array([np.ones(4, dtype=np.float32) * i for i in range(N)], dtype=object),
            observations=obs,
            rewards=torch.zeros(N),
            step_rewards=torch.zeros(N),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.zeros(N),
        )
        tree.record_initial_atari_state(state)

        frame = np.ones((4, 4, 3), dtype=np.uint8) * 55
        state1 = WalkerState(
            states=np.array(
                [np.ones(4, dtype=np.float32) * (i + 10) for i in range(N)], dtype=object
            ),
            observations=torch.randn(N, 8),
            rewards=torch.tensor([1.0, 5.0, 3.0]),
            step_rewards=torch.tensor([1.0, 5.0, 3.0]),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.tensor([1.0, 5.0, 3.0]),
        )
        tree.record_atari_step(
            state_before=state,
            state_after_clone=state1,
            state_final=state1,
            info={},
            clone_companions=torch.arange(N),
            will_clone=torch.zeros(N, dtype=torch.bool),
            best_frame=frame,
        )

        # Render missing frames to fill the path
        tree.render_missing_path_frames(
            lambda env_state: np.ones((4, 4, 3), dtype=np.uint8) * 88,
        )

        ah = tree.to_atari_history()
        # Path frames should be used: they follow root→best-leaf
        assert ah.has_frames
        # The frames should come from the path, not per-iteration metadata
        path_frames = tree.get_best_path_frames()
        assert len(ah.best_frames) == len(path_frames)


class TestSaveLoad:
    """Tests for serialization roundtrip."""

    def test_save_load_roundtrip(self, tree):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        tree.save(path)
        loaded = AtariTreeHistory.load(path)

        assert loaded.N == tree.N
        assert loaded.game_name == tree.game_name
        assert loaded._n_recorded == tree._n_recorded
        assert loaded._tree.data.number_of_nodes() == tree._tree.data.number_of_nodes()
        assert len(list(loaded._tree.data.edges())) == len(list(tree._tree.data.edges()))
        # Check a node's data survived the roundtrip
        nid = _node_id(1, 0, N_WALKERS)
        orig = tree._tree.data.nodes[nid]
        copy = loaded._tree.data.nodes[nid]
        assert orig["reward"] == copy["reward"]
        np.testing.assert_array_equal(orig["observations"], copy["observations"])

    def test_load_produces_valid_properties(self, tree):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        tree.save(path)
        loaded = AtariTreeHistory.load(path)

        assert loaded.iterations == tree.iterations
        assert loaded.rewards_max == tree.rewards_max
        assert loaded.alive_counts == tree.alive_counts


# ---------------------------------------------------------------------------
# High-death-rate mock for pruning tests
# ---------------------------------------------------------------------------


class HighDeathMockEnv:
    """Mock environment that kills ~50% of walkers per step.

    This ensures plenty of dead walkers available for cloning-over,
    which creates orphaned branches that pruning should remove.
    """

    def __init__(self, obs_shape=(128,), action_space_size=18):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size

    def reset(self, **kwargs):
        state = MockAtariState(
            np.zeros(4, dtype=np.float32),
            rgb_frame=np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8),
        )
        observation = np.zeros(self.obs_shape, dtype=np.float32)
        return state, observation, {}

    def sample_action(self):
        return np.random.randint(0, self.action_space_size)

    def step_batch(self, states, actions, dt, **kwargs):
        N = len(states)
        new_states = np.array(
            [
                MockAtariState(
                    np.random.randn(4).astype(np.float32),
                    rgb_frame=np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8),
                )
                for _ in range(N)
            ],
            dtype=object,
        )
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.random.randn(N).astype(np.float32) * 0.5
        dones = np.zeros(N, dtype=bool)
        # Kill ~50% each step
        n_dead = max(1, N // 2)
        dones[np.random.choice(N, size=n_dead, replace=False)] = True
        truncated = np.zeros(N, dtype=bool)
        infos = [{} for _ in range(N)]
        return new_states, observations, rewards, dones, truncated, infos


# ---------------------------------------------------------------------------
# Pruning tests
# ---------------------------------------------------------------------------


class TestPruning:
    """Tests for dynamic pruning of dead walker branches."""

    # -- Controlled / unit-level tests ----------------------------------

    def test_prune_removes_orphaned_dead_branches(self):
        """Manually build a 4-walker, 3-step tree where walker 2 dies at step 1
        and is cloned-over at step 2.  Pruning should remove the orphaned
        node (step 1, walker 2) and its ancestor (step 0, walker 2) since
        no alive path passes through them.
        """
        N = 4
        tree = AtariTreeHistory(N=N, game_name="unit")

        def _make_obs():
            return np.zeros(8, dtype=np.float32)

        # --- Step 0 (initial): 4 walkers, all alive ---
        obs = torch.zeros(N, 8)
        state0 = WalkerState(
            states=np.zeros((N, 4), dtype=np.float32).view(np.ndarray),
            observations=obs,
            rewards=torch.zeros(N),
            step_rewards=torch.zeros(N),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.zeros(N),
        )
        tree.record_initial_atari_state(state0)
        # nodes: root + 4 = 5
        assert tree._tree.data.number_of_nodes() == 5

        # --- Step 1: walker 2 dies, no cloning yet ---
        dones1 = torch.tensor([False, False, True, False])
        state1 = WalkerState(
            states=np.zeros((N, 4), dtype=np.float32).view(np.ndarray),
            observations=torch.randn(N, 8),
            rewards=torch.tensor([1.0, 2.0, 0.5, 3.0]),
            step_rewards=torch.tensor([1.0, 2.0, 0.5, 3.0]),
            dones=dones1,
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.tensor([1.0, 2.0, 0.5, 3.0]),
        )
        # No cloning in this step
        companions1 = torch.arange(N)
        will_clone1 = torch.zeros(N, dtype=torch.bool)
        tree.record_atari_step(
            state_before=state0,
            state_after_clone=state1,
            state_final=state1,
            info={},
            clone_companions=companions1,
            will_clone=will_clone1,
        )
        # nodes: 5 + 4 = 9
        assert tree._tree.data.number_of_nodes() == 9

        # --- Step 2: walker 2 clones from walker 3 ---
        # This makes node(step=2, w=2) → parent = node(step=1, w=3)
        # So node(step=1, w=2) becomes an orphaned leaf.
        state2 = WalkerState(
            states=np.zeros((N, 4), dtype=np.float32).view(np.ndarray),
            observations=torch.randn(N, 8),
            rewards=torch.tensor([2.0, 4.0, 3.0, 6.0]),
            step_rewards=torch.tensor([1.0, 2.0, 3.0, 3.0]),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.tensor([2.0, 4.0, 3.0, 6.0]),
        )
        companions2 = torch.tensor([0, 1, 3, 3])  # walker 2 clones from 3
        will_clone2 = torch.tensor([False, False, True, False])
        tree.record_atari_step(
            state_before=state1,
            state_after_clone=state2,
            state_final=state2,
            info={},
            clone_companions=companions2,
            will_clone=will_clone2,
        )
        # nodes: 9 + 4 = 13
        assert tree._tree.data.number_of_nodes() == 13

        # node(step=1, w=2) should be an orphaned leaf (no children)
        orphan_nid = _node_id(1, 2, N)
        assert orphan_nid in tree._tree.data.nodes
        assert len(list(tree._tree.data.successors(orphan_nid))) == 0

        # --- Prune ---
        removed = tree.prune_dead_branches()
        assert removed > 0

        # The orphan node(1,2) should be gone
        assert orphan_nid not in tree._tree.data.nodes
        # Its parent node(0,2) should also be gone (no other child)
        parent_nid = _node_id(0, 2, N)
        assert parent_nid not in tree._tree.data.nodes

        # Alive walker leaves should still be intact
        for w in range(N):
            leaf_nid = _node_id(2, w, N)
            assert leaf_nid in tree._tree.data.nodes

        # Branches for alive walkers should still be traceable
        for w in [0, 1, 3]:
            branch = tree.get_walker_branch(w)
            assert len(branch) >= 2
            assert branch[-1] == _node_id(2, w, N)
            # Every consecutive pair is a real edge
            for i in range(len(branch) - 1):
                assert tree._tree.data.has_edge(branch[i], branch[i + 1])

    def test_prune_no_op_when_no_orphans(self):
        """If no walkers are cloned over, pruning should remove nothing."""
        N = 3
        tree = AtariTreeHistory(N=N, game_name="noop")
        obs = torch.zeros(N, 4)
        state = WalkerState(
            states=np.zeros((N, 4), dtype=np.float32).view(np.ndarray),
            observations=obs,
            rewards=torch.zeros(N),
            step_rewards=torch.zeros(N),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.zeros(N),
        )
        tree.record_initial_atari_state(state)

        # Step 1 — no cloning
        tree.record_atari_step(
            state_before=state,
            state_after_clone=state,
            state_final=state,
            info={},
            clone_companions=torch.arange(N),
            will_clone=torch.zeros(N, dtype=torch.bool),
        )

        nodes_before = tree._tree.data.number_of_nodes()
        removed = tree.prune_dead_branches()
        assert removed == 0
        assert tree._tree.data.number_of_nodes() == nodes_before

    def test_prune_preserves_shared_ancestors(self):
        """When two walkers share an ancestor (via cloning), pruning one
        dead branch must not remove the shared parent."""
        N = 3
        tree = AtariTreeHistory(N=N, game_name="shared")
        obs = torch.zeros(N, 4)
        base = WalkerState(
            states=np.zeros((N, 4), dtype=np.float32).view(np.ndarray),
            observations=obs,
            rewards=torch.zeros(N),
            step_rewards=torch.zeros(N),
            dones=torch.zeros(N, dtype=torch.bool),
            truncated=torch.zeros(N, dtype=torch.bool),
            actions=np.zeros(N, dtype=int),
            dt=np.ones(N, dtype=int),
            infos=[{} for _ in range(N)],
            virtual_rewards=torch.zeros(N),
        )

        tree.record_initial_atari_state(base)

        # Step 1: walker 1 clones from 0, walker 2 clones from 0
        # Both (1,1) and (1,2) have parent (0,0)
        tree.record_atari_step(
            state_before=base,
            state_after_clone=base,
            state_final=base,
            info={},
            clone_companions=torch.tensor([0, 0, 0]),
            will_clone=torch.tensor([False, True, True]),
        )
        # node(0,1) and node(0,2) are orphaned leaves
        # But node(0,0) has 3 children: (1,0), (1,1), (1,2) — must survive

        # Step 2: walker 2 clones from 1 → node(1,2) becomes orphan
        tree.record_atari_step(
            state_before=base,
            state_after_clone=base,
            state_final=base,
            info={},
            clone_companions=torch.tensor([0, 1, 1]),
            will_clone=torch.tensor([False, False, True]),
        )

        # Before pruning: node(0,1), node(0,2) are orphan leaves from step 1
        # and node(1,2) is orphan from step 2
        tree._tree.data.number_of_nodes()
        removed = tree.prune_dead_branches()
        assert removed > 0

        # node(0,0) must survive — it's an ancestor of alive walkers 0 and 1
        assert _node_id(0, 0, N) in tree._tree.data.nodes
        # node(1,0) and node(1,1) must survive — they lead to alive step-2 leaves
        assert _node_id(1, 0, N) in tree._tree.data.nodes
        assert _node_id(1, 1, N) in tree._tree.data.nodes
        # All step-2 leaves must survive
        for w in range(N):
            assert _node_id(2, w, N) in tree._tree.data.nodes

    # -- End-to-end through AtariFractalGas -----------------------------

    def test_prune_during_run_reduces_node_count(self):
        """Run the gas step-by-step with pruning after each step.
        The pruned graph should be strictly smaller than the unpruned one.
        """
        N = 8
        n_steps = 15
        env = HighDeathMockEnv()
        gas = AtariFractalGas(env=env, N=N, device="cpu", seed=123)

        state = gas.reset()
        tree = AtariTreeHistory(N=N, game_name="prune_e2e", max_iterations=n_steps)
        tree.record_initial_atari_state(state)

        total_removed = 0
        for _ in range(n_steps):
            prev = state
            state, info = gas.step(prev)
            tree.record_atari_step(
                state_before=prev,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
            )
            total_removed += tree.prune_dead_branches()

        # With 50% death rate and 15 steps, there should be many prune events
        assert total_removed > 0, "Expected at least some pruned nodes"

        # The graph should have strictly fewer nodes than the unpruned max
        unpruned_max = 1 + N * (n_steps + 1)  # root + N per recorded step
        actual = tree._tree.data.number_of_nodes()
        assert actual < unpruned_max, (
            f"Pruned graph ({actual}) should be smaller than " f"unpruned ({unpruned_max})"
        )

    def test_pruned_tree_branches_still_valid(self):
        """After pruning, get_walker_branch must still return a valid
        root-to-leaf path for every walker at the latest step."""
        N = 8
        n_steps = 10
        env = HighDeathMockEnv()
        gas = AtariFractalGas(env=env, N=N, device="cpu", seed=77)

        state = gas.reset()
        tree = AtariTreeHistory(N=N, game_name="branch_valid", max_iterations=n_steps)
        tree.record_initial_atari_state(state)

        for _ in range(n_steps):
            prev = state
            state, info = gas.step(prev)
            tree.record_atari_step(
                state_before=prev,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
            )
            tree.prune_dead_branches()

        for w in range(N):
            branch = tree.get_walker_branch(w)
            assert len(branch) >= 2, f"Walker {w}: branch too short"
            assert branch[-1] == _node_id(n_steps, w, N)
            # Branch must start at root or DEFAULT_FIRST_NODE_ID (get_branch
            # stops at whichever it hits first).
            assert branch[0] in {
                int(DEFAULT_ROOT_ID),
                1,
            }, f"Walker {w}: unexpected branch start {branch[0]}"
            # Every consecutive pair should be a real edge
            for i in range(len(branch) - 1):
                assert tree._tree.data.has_edge(
                    branch[i], branch[i + 1]
                ), f"Missing edge {branch[i]} → {branch[i + 1]} in walker {w}'s branch"

    def test_pruned_tree_to_atari_history(self):
        """Conversion to AtariHistory must succeed after pruning."""
        N = 6
        n_steps = 10
        env = HighDeathMockEnv()
        gas = AtariFractalGas(env=env, N=N, device="cpu", seed=99)

        state = gas.reset()
        tree = AtariTreeHistory(N=N, game_name="prune_convert", max_iterations=n_steps)
        tree.record_initial_atari_state(state)

        for _ in range(n_steps):
            prev = state
            state, info = gas.step(prev)
            tree.record_atari_step(
                state_before=prev,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
            )
            tree.prune_dead_branches()

        ah = tree.to_atari_history()
        assert isinstance(ah, AtariHistory)
        assert ah.max_iterations == n_steps
        assert len(ah.rewards_max) == n_steps

    def test_prune_then_save_load_roundtrip(self):
        """Save/load after pruning must preserve the smaller graph."""
        N = 6
        n_steps = 8
        env = HighDeathMockEnv()
        gas = AtariFractalGas(env=env, N=N, device="cpu", seed=55)

        state = gas.reset()
        tree = AtariTreeHistory(N=N, game_name="prune_save", max_iterations=n_steps)
        tree.record_initial_atari_state(state)

        for _ in range(n_steps):
            prev = state
            state, info = gas.step(prev)
            tree.record_atari_step(
                state_before=prev,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
            )
            tree.prune_dead_branches()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        nodes_before_save = tree._tree.data.number_of_nodes()
        tree.save(path)
        loaded = AtariTreeHistory.load(path)

        assert loaded._tree.data.number_of_nodes() == nodes_before_save
        assert loaded._n_recorded == tree._n_recorded
        # Walker branches should still be valid in the loaded tree
        for w in range(N):
            branch = loaded.get_walker_branch(w)
            assert branch[0] == int(DEFAULT_ROOT_ID)
            assert branch[-1] == _node_id(n_steps, w, N)

    def test_repeated_prune_idempotent(self):
        """Calling prune_dead_branches twice in a row without new recording
        should be a no-op the second time."""
        N = 6
        env = HighDeathMockEnv()
        gas = AtariFractalGas(env=env, N=N, device="cpu", seed=88)

        state = gas.reset()
        tree = AtariTreeHistory(N=N, game_name="idempotent")
        tree.record_initial_atari_state(state)

        for _ in range(5):
            prev = state
            state, info = gas.step(prev)
            tree.record_atari_step(
                state_before=prev,
                state_after_clone=info["_state_after_clone"],
                state_final=state,
                info=info,
                clone_companions=info["clone_companions"],
                will_clone=info["will_clone"],
            )

        tree.prune_dead_branches()
        second_removed = tree.prune_dead_branches()
        assert (
            second_removed == 0
        ), f"Second prune should be no-op but removed {second_removed} nodes"
