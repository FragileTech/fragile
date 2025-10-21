"""Tests for FractalSet data structure."""

import networkx as nx
import pytest
import torch

from fragile.core.fractal_set import FractalSet
from fragile.core.history import RunHistory


@pytest.fixture
def simple_history():
    """Create a simple RunHistory for testing FractalSet construction."""
    N, d = 5, 2
    n_steps = 10
    record_every = 2
    n_recorded = (n_steps // record_every) + 1  # 6 timesteps

    # Create dummy data
    x_before = torch.randn(n_recorded, N, d, dtype=torch.float64)
    v_before = torch.randn(n_recorded, N, d, dtype=torch.float64)
    x_after = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
    v_after = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
    x_final = torch.randn(n_recorded, N, d, dtype=torch.float64)
    v_final = torch.randn(n_recorded, N, d, dtype=torch.float64)

    # Per-step data
    n_alive = torch.ones(n_recorded, dtype=torch.long) * N
    num_cloned = torch.randint(0, N // 2, (n_recorded - 1,))
    step_times = torch.rand(n_recorded - 1, dtype=torch.float64) * 0.01

    # Per-walker per-step data
    fitness = torch.randn(n_recorded - 1, N, dtype=torch.float64)
    rewards = torch.randn(n_recorded - 1, N, dtype=torch.float64)
    cloning_scores = torch.randn(n_recorded - 1, N, dtype=torch.float64)
    cloning_probs = torch.rand(n_recorded - 1, N, dtype=torch.float64)
    will_clone = torch.rand(n_recorded - 1, N) > 0.8
    alive_mask = torch.ones(n_recorded - 1, N, dtype=torch.bool)
    companions_distance = torch.randint(0, N, (n_recorded - 1, N))
    companions_clone = torch.randint(0, N, (n_recorded - 1, N))
    distances = torch.rand(n_recorded - 1, N, dtype=torch.float64)
    z_rewards = torch.randn(n_recorded - 1, N, dtype=torch.float64)
    z_distances = torch.randn(n_recorded - 1, N, dtype=torch.float64)
    pos_squared_differences = torch.rand(n_recorded - 1, N, dtype=torch.float64)
    vel_squared_differences = torch.rand(n_recorded - 1, N, dtype=torch.float64)
    rescaled_rewards = torch.rand(n_recorded - 1, N, dtype=torch.float64)
    rescaled_distances = torch.rand(n_recorded - 1, N, dtype=torch.float64)

    # Localized statistics (per-step, global case rho → ∞)
    mu_rewards = torch.randn(n_recorded - 1, dtype=torch.float64)
    sigma_rewards = torch.rand(n_recorded - 1, dtype=torch.float64) + 0.5
    mu_distances = torch.rand(n_recorded - 1, dtype=torch.float64)
    sigma_distances = torch.rand(n_recorded - 1, dtype=torch.float64) + 0.5

    return RunHistory(
        N=N,
        d=d,
        n_steps=n_steps,
        n_recorded=n_recorded,
        record_every=record_every,
        terminated_early=False,
        final_step=n_steps,
        x_before_clone=x_before,
        v_before_clone=v_before,
        x_after_clone=x_after,
        v_after_clone=v_after,
        x_final=x_final,
        v_final=v_final,
        n_alive=n_alive,
        num_cloned=num_cloned,
        step_times=step_times,
        fitness=fitness,
        rewards=rewards,
        cloning_scores=cloning_scores,
        cloning_probs=cloning_probs,
        will_clone=will_clone,
        alive_mask=alive_mask,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        distances=distances,
        z_rewards=z_rewards,
        z_distances=z_distances,
        pos_squared_differences=pos_squared_differences,
        vel_squared_differences=vel_squared_differences,
        rescaled_rewards=rescaled_rewards,
        rescaled_distances=rescaled_distances,
        mu_rewards=mu_rewards,
        sigma_rewards=sigma_rewards,
        mu_distances=mu_distances,
        sigma_distances=sigma_distances,
        total_time=1.234,
        init_time=0.1,
    )


@pytest.fixture
def simple_fractal_set(simple_history):
    """Create a FractalSet from simple_history."""
    return FractalSet(simple_history)


class TestFractalSetConstruction:
    """Tests for FractalSet construction from RunHistory."""

    def test_initialization(self, simple_fractal_set):
        """Test FractalSet initializes correctly from RunHistory."""
        assert simple_fractal_set.N == 5
        assert simple_fractal_set.d == 2
        assert simple_fractal_set.n_steps == 10
        assert simple_fractal_set.n_recorded == 6
        assert simple_fractal_set.record_every == 2

        # Graph should be created
        assert isinstance(simple_fractal_set.graph, nx.DiGraph)
        assert simple_fractal_set.graph.number_of_nodes() > 0
        assert simple_fractal_set.graph.number_of_edges() > 0

    def test_node_count(self, simple_fractal_set):
        """Test correct number of nodes are created."""
        # Should have N × n_recorded nodes
        expected_nodes = 5 * 6  # 30 nodes
        assert simple_fractal_set.total_nodes == expected_nodes

    def test_node_structure(self, simple_fractal_set):
        """Test node IDs follow (walker_id, timestep) format."""
        # Check a few specific nodes exist
        assert (0, 0) in simple_fractal_set.graph.nodes
        assert (4, 5) in simple_fractal_set.graph.nodes
        assert (2, 3) in simple_fractal_set.graph.nodes

        # Non-existent node
        assert (5, 0) not in simple_fractal_set.graph.nodes  # walker_id out of range

    def test_node_attributes(self, simple_fractal_set):
        """Test nodes have correct attributes."""
        node_data = simple_fractal_set.get_node_data(walker_id=0, timestep=1)

        # Check required attributes
        assert "walker_id" in node_data
        assert "timestep" in node_data
        assert "absolute_step" in node_data
        assert "t" in node_data
        assert "x" in node_data
        assert "v" in node_data
        assert "E_kin" in node_data
        assert "alive" in node_data

        # At t > 0, should have per-step data
        assert "fitness" in node_data
        assert "reward" in node_data
        assert "cloning_score" in node_data

        # Check types
        assert node_data["walker_id"] == 0
        assert node_data["timestep"] == 1
        assert isinstance(node_data["x"], torch.Tensor)
        assert isinstance(node_data["v"], torch.Tensor)
        assert isinstance(node_data["E_kin"], float)

    def test_node_attributes_at_t0(self, simple_fractal_set):
        """Test t=0 nodes don't have per-step data."""
        node_data = simple_fractal_set.get_node_data(walker_id=0, timestep=0)

        # Should have base attributes
        assert "walker_id" in node_data
        assert "x" in node_data
        assert "v" in node_data
        assert "alive" in node_data
        assert node_data["alive"] is True  # All alive at t=0

        # Should NOT have per-step data
        assert "fitness" not in node_data
        assert "reward" not in node_data

    def test_cst_edge_count(self, simple_fractal_set):
        """Test correct number of CST edges are created."""
        # For all walkers alive: N × (n_recorded - 1) CST edges
        expected_cst = 5 * 5  # 25 edges
        assert simple_fractal_set.num_cst_edges == expected_cst

    def test_cst_edge_structure(self, simple_fractal_set):
        """Test CST edges connect correct nodes."""
        # Check walker 0's CST edges
        for t in range(5):
            source = (0, t)
            target = (0, t + 1)
            assert simple_fractal_set.graph.has_edge(source, target)

            edge_data = simple_fractal_set.graph.edges[source, target]
            assert edge_data["edge_type"] == "cst"
            assert edge_data["walker_id"] == 0
            assert edge_data["timestep"] == t

    def test_cst_edge_attributes(self, simple_fractal_set):
        """Test CST edges have correct attributes."""
        edge_data = simple_fractal_set.graph.edges[(0, 0), (0, 1)]

        # Check required attributes
        assert edge_data["edge_type"] == "cst"
        assert "walker_id" in edge_data
        assert "timestep" in edge_data
        assert "v_t" in edge_data
        assert "v_t1" in edge_data
        assert "Delta_v" in edge_data
        assert "Delta_x" in edge_data
        assert "norm_Delta_v" in edge_data
        assert "norm_Delta_x" in edge_data

        # Check types
        assert isinstance(edge_data["v_t"], torch.Tensor)
        assert isinstance(edge_data["Delta_v"], torch.Tensor)
        assert isinstance(edge_data["norm_Delta_v"], float)

    def test_ig_edge_count(self, simple_fractal_set):
        """Test correct number of IG edges are created."""
        # For all walkers alive: sum over t of k(t) * (k(t) - 1)
        # where k(t) = number of alive walkers at timestep t
        # Here all walkers alive at all times (except t=0 has no IG edges)
        # So: (n_recorded - 1) × N × (N - 1) = 5 × 5 × 4 = 100
        expected_ig = 5 * 5 * 4
        assert simple_fractal_set.num_ig_edges == expected_ig

    def test_ig_edge_structure(self, simple_fractal_set):
        """Test IG edges are directed and connect different walkers."""
        # At t=1, should have edges between all walker pairs
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                source = (i, 1)
                target = (j, 1)
                assert simple_fractal_set.graph.has_edge(source, target)

                edge_data = simple_fractal_set.graph.edges[source, target]
                assert edge_data["edge_type"] == "ig"
                assert edge_data["source_walker"] == i
                assert edge_data["target_walker"] == j
                assert edge_data["timestep"] == 1

    def test_ig_edge_antisymmetry(self, simple_fractal_set):
        """Test IG edges encode antisymmetric cloning potential."""
        # Get edges (0, 1) → (1, 1) and (1, 1) → (0, 1)
        edge_01_11 = simple_fractal_set.graph.edges[(0, 1), (1, 1)]
        edge_11_01 = simple_fractal_set.graph.edges[(1, 1), (0, 1)]

        V_clone_0_to_1 = edge_01_11["V_clone"]
        V_clone_1_to_0 = edge_11_01["V_clone"]

        # Should satisfy V_clone(i→j) = -V_clone(j→i)
        # V_clone(0→1) = Φ_1 - Φ_0
        # V_clone(1→0) = Φ_0 - Φ_1 = -V_clone(0→1)
        assert abs(V_clone_0_to_1 + V_clone_1_to_0) < 1e-10

    def test_ig_edge_attributes(self, simple_fractal_set):
        """Test IG edges have correct attributes."""
        edge_data = simple_fractal_set.graph.edges[(0, 1), (1, 1)]

        # Check required attributes
        assert edge_data["edge_type"] == "ig"
        assert "source_walker" in edge_data
        assert "target_walker" in edge_data
        assert "timestep" in edge_data
        assert "Delta_x_ij" in edge_data
        assert "Delta_v_ij" in edge_data
        assert "distance" in edge_data
        assert "V_clone" in edge_data
        assert "fitness_i" in edge_data
        assert "fitness_j" in edge_data

        # Check types
        assert isinstance(edge_data["Delta_x_ij"], torch.Tensor)
        assert isinstance(edge_data["distance"], float)
        assert isinstance(edge_data["V_clone"], float)

    def test_no_ig_edges_at_t0(self, simple_fractal_set):
        """Test no IG edges at t=0 (no fitness data)."""
        # Count IG edges at t=0
        ig_edges_t0 = [
            (u, v)
            for u, v, d in simple_fractal_set.graph.edges(data=True)
            if d["edge_type"] == "ig" and d["timestep"] == 0
        ]
        assert len(ig_edges_t0) == 0


class TestFractalSetQueries:
    """Tests for FractalSet query methods."""

    def test_get_walker_trajectory(self, simple_fractal_set):
        """Test trajectory extraction."""
        traj = simple_fractal_set.get_walker_trajectory(walker_id=0, stage="final")

        assert "x" in traj and "v" in traj
        assert traj["x"].shape == (6, 2)  # n_recorded, d
        assert traj["v"].shape == (6, 2)

    def test_get_cst_subgraph(self, simple_fractal_set):
        """Test CST subgraph extraction."""
        cst = simple_fractal_set.get_cst_subgraph()

        # Should only contain CST edges
        for _, _, data in cst.edges(data=True):
            assert data["edge_type"] == "cst"

        # Should have correct edge count
        assert cst.number_of_edges() == simple_fractal_set.num_cst_edges

    def test_get_ig_subgraph_all(self, simple_fractal_set):
        """Test IG subgraph extraction (all timesteps)."""
        ig = simple_fractal_set.get_ig_subgraph()

        # Should only contain IG edges
        for _, _, data in ig.edges(data=True):
            assert data["edge_type"] == "ig"

        # Should have correct edge count
        assert ig.number_of_edges() == simple_fractal_set.num_ig_edges

    def test_get_ig_subgraph_timestep(self, simple_fractal_set):
        """Test IG subgraph extraction at specific timestep."""
        ig_t1 = simple_fractal_set.get_ig_subgraph(timestep=1)

        # Should only have edges at t=1
        for _, _, data in ig_t1.edges(data=True):
            assert data["edge_type"] == "ig"
            assert data["timestep"] == 1

        # For all alive: N × (N-1) = 5 × 4 = 20 edges
        assert ig_t1.number_of_edges() == 20

    def test_get_cloning_events(self, simple_fractal_set):
        """Test cloning event extraction."""
        events = simple_fractal_set.get_cloning_events()

        # Should be a list of tuples
        assert isinstance(events, list)
        for event in events:
            assert len(event) == 3
            step, cloner_idx, companion_idx = event
            assert isinstance(step, int)
            assert 0 <= cloner_idx < 5
            assert 0 <= companion_idx < 5

    def test_get_alive_walkers(self, simple_fractal_set):
        """Test alive walker extraction."""
        alive = simple_fractal_set.get_alive_walkers(timestep=1)

        # All walkers alive in simple history
        assert len(alive) == 5
        assert alive == [0, 1, 2, 3, 4]

    def test_get_node_data(self, simple_fractal_set):
        """Test node data extraction."""
        data = simple_fractal_set.get_node_data(walker_id=2, timestep=3)

        assert data["walker_id"] == 2
        assert data["timestep"] == 3
        assert "x" in data
        assert "fitness" in data


class TestFractalSetProperties:
    """Tests for FractalSet properties."""

    def test_num_cst_edges(self, simple_fractal_set):
        """Test num_cst_edges property."""
        assert simple_fractal_set.num_cst_edges == 25

    def test_num_ig_edges(self, simple_fractal_set):
        """Test num_ig_edges property."""
        assert simple_fractal_set.num_ig_edges == 100

    def test_total_nodes(self, simple_fractal_set):
        """Test total_nodes property."""
        assert simple_fractal_set.total_nodes == 30


class TestFractalSetSerialization:
    """Tests for FractalSet save/load."""

    def test_save_load(self, simple_fractal_set, simple_history, tmp_path):
        """Test save and load cycle."""
        save_path = tmp_path / "fractal_set.pkl"

        # Save
        simple_fractal_set.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded = FractalSet.load(str(save_path), simple_history)

        # Check metadata
        assert loaded.N == simple_fractal_set.N
        assert loaded.d == simple_fractal_set.d
        assert loaded.n_steps == simple_fractal_set.n_steps
        assert loaded.total_nodes == simple_fractal_set.total_nodes
        assert loaded.num_cst_edges == simple_fractal_set.num_cst_edges
        assert loaded.num_ig_edges == simple_fractal_set.num_ig_edges

        # Check graph structure
        assert nx.is_isomorphic(loaded.graph, simple_fractal_set.graph)


class TestFractalSetSummary:
    """Tests for FractalSet summary and repr."""

    def test_summary(self, simple_fractal_set):
        """Test summary string generation."""
        summary = simple_fractal_set.summary()

        assert isinstance(summary, str)
        assert "10 steps" in summary
        assert "5 walkers" in summary
        assert "2D" in summary
        assert "30 spacetime points" in summary
        assert "25" in summary  # CST edges
        assert "100" in summary  # IG edges

    def test_repr(self, simple_fractal_set):
        """Test __repr__ method."""
        repr_str = repr(simple_fractal_set)

        assert "FractalSet" in repr_str
        assert "N=5" in repr_str
        assert "d=2" in repr_str
        assert "nodes=30" in repr_str
        assert "cst=25" in repr_str
        assert "ig=100" in repr_str


class TestFractalSetWithDeadWalkers:
    """Tests for FractalSet with dead walkers."""

    @pytest.fixture
    def history_with_dead(self):
        """Create RunHistory with some dead walkers."""
        N, d = 5, 2
        n_steps = 10
        record_every = 2
        n_recorded = 6

        x_before = torch.randn(n_recorded, N, d, dtype=torch.float64)
        v_before = torch.randn(n_recorded, N, d, dtype=torch.float64)
        x_after = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
        v_after = torch.randn(n_recorded - 1, N, d, dtype=torch.float64)
        x_final = torch.randn(n_recorded, N, d, dtype=torch.float64)
        v_final = torch.randn(n_recorded, N, d, dtype=torch.float64)

        n_alive = torch.tensor([5, 5, 4, 4, 3, 3], dtype=torch.long)
        num_cloned = torch.zeros(n_recorded - 1, dtype=torch.long)
        step_times = torch.zeros(n_recorded - 1, dtype=torch.float64)

        # Create alive_mask with some dead walkers
        alive_mask = torch.ones(n_recorded - 1, N, dtype=torch.bool)
        alive_mask[1, 4] = False  # Walker 4 dies at t=2
        alive_mask[2:, 4] = False
        alive_mask[3, 3] = False  # Walker 3 dies at t=4
        alive_mask[4, 3] = False

        fitness = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        rewards = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        cloning_scores = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        cloning_probs = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        will_clone = torch.zeros(n_recorded - 1, N, dtype=torch.bool)
        companions_distance = torch.zeros(n_recorded - 1, N, dtype=torch.long)
        companions_clone = torch.zeros(n_recorded - 1, N, dtype=torch.long)
        distances = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        z_rewards = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        z_distances = torch.randn(n_recorded - 1, N, dtype=torch.float64)
        pos_squared_differences = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        vel_squared_differences = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        rescaled_rewards = torch.rand(n_recorded - 1, N, dtype=torch.float64)
        rescaled_distances = torch.rand(n_recorded - 1, N, dtype=torch.float64)

        # Localized statistics
        mu_rewards = torch.randn(n_recorded - 1, dtype=torch.float64)
        sigma_rewards = torch.rand(n_recorded - 1, dtype=torch.float64) + 0.5
        mu_distances = torch.rand(n_recorded - 1, dtype=torch.float64)
        sigma_distances = torch.rand(n_recorded - 1, dtype=torch.float64) + 0.5

        return RunHistory(
            N=N,
            d=d,
            n_steps=n_steps,
            n_recorded=n_recorded,
            record_every=record_every,
            terminated_early=False,
            final_step=n_steps,
            x_before_clone=x_before,
            v_before_clone=v_before,
            x_after_clone=x_after,
            v_after_clone=v_after,
            x_final=x_final,
            v_final=v_final,
            n_alive=n_alive,
            num_cloned=num_cloned,
            step_times=step_times,
            fitness=fitness,
            rewards=rewards,
            cloning_scores=cloning_scores,
            cloning_probs=cloning_probs,
            will_clone=will_clone,
            alive_mask=alive_mask,
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            distances=distances,
            z_rewards=z_rewards,
            z_distances=z_distances,
            pos_squared_differences=pos_squared_differences,
            vel_squared_differences=vel_squared_differences,
            rescaled_rewards=rescaled_rewards,
            rescaled_distances=rescaled_distances,
            mu_rewards=mu_rewards,
            sigma_rewards=sigma_rewards,
            mu_distances=mu_distances,
            sigma_distances=sigma_distances,
            total_time=1.0,
            init_time=0.1,
        )

    def test_cst_edges_skip_dead_walkers(self, history_with_dead):
        """Test CST edges are not created for dead walkers."""
        fs = FractalSet(history_with_dead)

        # alive_mask[i, j] indicates if walker j is alive during transition i→i+1
        # alive_mask[1, 4] = False means walker 4 dies during t=1→2
        # So CST edge (4,1)→(4,2) should not exist
        assert fs.graph.has_edge((4, 0), (4, 1))  # t=0→1 (alive_mask[0,4]=True)
        assert not fs.graph.has_edge((4, 1), (4, 2))  # t=1→2 (alive_mask[1,4]=False)

        # alive_mask[3, 3] = False means walker 3 dies during t=3→4
        assert fs.graph.has_edge((3, 2), (3, 3))  # t=2→3 (alive_mask[2,3]=True)
        assert not fs.graph.has_edge((3, 3), (3, 4))  # t=3→4 (alive_mask[3,3]=False)

    def test_ig_edges_only_alive_walkers(self, history_with_dead):
        """Test IG edges only connect alive walkers."""
        fs = FractalSet(history_with_dead)

        # At t=2, walker 4 is dead (alive_mask[1, 4] = False), so no edges from/to it
        ig_t2 = fs.get_ig_subgraph(timestep=2)
        for u, v in ig_t2.edges():
            assert u[0] != 4 and v[0] != 4

        # At t=4, walkers 3 and 4 are dead
        ig_t4 = fs.get_ig_subgraph(timestep=4)
        for u, v in ig_t4.edges():
            assert u[0] not in [3, 4]
            assert v[0] not in [3, 4]

    def test_get_alive_walkers_with_dead(self, history_with_dead):
        """Test get_alive_walkers correctly identifies alive walkers."""
        fs = FractalSet(history_with_dead)

        # At t=1, all should be alive (alive_mask[0, :] all True)
        alive_t1 = fs.get_alive_walkers(timestep=1)
        assert len(alive_t1) == 5

        # At t=2, walker 4 is dead (alive_mask[1, 4] = False)
        alive_t2 = fs.get_alive_walkers(timestep=2)
        assert 4 not in alive_t2
        assert len(alive_t2) == 4

        # At t=4, walker 3 is dead (alive_mask[3, 3] = False)
        # and walker 4 is still dead (alive_mask[2:, 4] = False)
        alive_t4 = fs.get_alive_walkers(timestep=4)
        assert 3 not in alive_t4
        assert 4 not in alive_t4
        assert len(alive_t4) == 3
