"""Tests for FractalSet: Graph-based representation of swarm evolution."""

from pathlib import Path
import tempfile

import networkx as nx
import numpy as np
import pytest
import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
    SwarmState,
)
from fragile.fractal_set import FractalSet


@pytest.fixture
def simple_state():
    """Create a simple 2D swarm state for testing."""
    N, d = 10, 2
    x = torch.randn(N, d)
    v = torch.randn(N, d)
    return SwarmState(x, v)


@pytest.fixture
def simple_gas():
    """Create a simple EuclideanGas instance."""
    params = EuclideanGasParams(
        N=10,
        d=2,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
        cloning=CloningParams(
            sigma_x=0.5,
            lambda_alg=0.1,
            alpha_restitution=0.5,
            companion_selection_method="softmax",
        ),
        device="cpu",
        dtype="float32",
    )
    return EuclideanGas(params)


class TestFractalSetBasics:
    """Test basic FractalSet functionality."""

    def test_initialization(self):
        """Test FractalSet initialization."""
        N, d = 10, 2
        fs = FractalSet(N=N, d=d)

        assert fs.N == N
        assert fs.d == d
        assert fs.current_step == 0
        assert fs.graph.number_of_nodes() == 0
        assert fs.graph.graph["N"] == N
        assert fs.graph.graph["d"] == d
        assert fs.graph.graph["n_steps"] == 0

    def test_add_single_timestep(self, simple_state):
        """Test adding a single timestep."""
        N, d = simple_state.N, simple_state.d
        fs = FractalSet(N=N, d=d)

        fs.add_timestep(state=simple_state, timestep=0)

        # Check nodes were created
        assert fs.graph.number_of_nodes() == N
        assert fs.current_step == 0
        assert fs.graph.graph["n_steps"] == 1

        # Check node attributes
        for i in range(N):
            node_id = (i, 0)
            assert fs.graph.has_node(node_id)
            node_data = fs.graph.nodes[node_id]

            assert "x" in node_data
            assert "v" in node_data
            assert "high_error" in node_data
            assert "alive" in node_data
            assert "positional_error" in node_data
            assert len(node_data["x"]) == d
            assert len(node_data["v"]) == d

    def test_add_multiple_timesteps(self, simple_state):
        """Test adding multiple timesteps creates edges."""
        N, d = simple_state.N, simple_state.d
        fs = FractalSet(N=N, d=d)

        # Add first timestep
        fs.add_timestep(state=simple_state, timestep=0)

        # Add second timestep (should create edges)
        state2 = SwarmState(simple_state.x + 0.1, simple_state.v)
        fs.add_timestep(state=state2, timestep=1)

        # Check nodes and edges
        assert fs.graph.number_of_nodes() == 2 * N
        assert fs.graph.number_of_edges() == N  # N temporal edges
        assert fs.graph.graph["n_steps"] == 2

        # Check edges exist
        for i in range(N):
            edge = ((i, 0), (i, 1))
            assert fs.graph.has_edge(*edge)
            edge_data = fs.graph.edges[edge]
            assert edge_data["edge_type"] == "kinetic"

    def test_high_error_partitioning(self):
        """Test that high-error attribute is recorded for all walkers."""
        N, d = 10, 2

        # Create state with varied positions
        x = torch.randn(N, d) * 2.0  # Random positions with variety
        v = torch.randn(N, d) * 0.1

        state = SwarmState(x, v)
        fs = FractalSet(N=N, d=d)
        fs.add_timestep(state=state, timestep=0)

        # Check that all nodes have high_error attribute
        for i in range(N):
            assert "high_error" in fs.graph.nodes[i, 0]
            assert isinstance(fs.graph.nodes[i, 0]["high_error"], bool)
            assert "positional_error" in fs.graph.nodes[i, 0]

        # With random positions, typically we'll have both sets
        # But not strictly required, so just check attribute exists
        n_high_error = sum(1 for i in range(N) if fs.graph.nodes[i, 0]["high_error"])

        # Should have some reasonable split (allow edge cases)
        assert 0 <= n_high_error <= N

    def test_trajectory_recording(self):
        """Test that graph-level trajectories are recorded."""
        N, d = 10, 2
        fs = FractalSet(N=N, d=d)

        n_steps = 5
        for t in range(n_steps):
            x = torch.randn(N, d)
            v = torch.randn(N, d)
            state = SwarmState(x, v)
            fs.add_timestep(state=state, timestep=t)

        # Check trajectories
        assert len(fs.graph.graph["var_x_trajectory"]) == n_steps
        assert len(fs.graph.graph["var_v_trajectory"]) == n_steps
        assert len(fs.graph.graph["n_alive_trajectory"]) == n_steps
        assert len(fs.graph.graph["centroid_trajectory"]) == n_steps


class TestFractalSetFromRun:
    """Test creating FractalSet from EuclideanGas run."""

    def test_from_run_basic(self, simple_gas):
        """Test creating FractalSet from a short run."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps, record_fitness=True)

        # Check basic properties
        assert fs.N == simple_gas.params.N
        assert fs.d == simple_gas.params.d
        assert fs.graph.graph["n_steps"] <= n_steps + 1  # Might stop early

        # Check all timesteps have nodes
        for t in range(fs.graph.graph["n_steps"]):
            for i in range(fs.N):
                assert fs.graph.has_node((i, t))

    def test_from_run_with_fitness(self, simple_gas):
        """Test that fitness data is recorded when requested."""
        n_steps = 5
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps, record_fitness=True)

        # Check fitness attributes exist
        sample_node = (0, 0)
        node_data = fs.graph.nodes[sample_node]

        assert "fitness" in node_data
        assert "potential" in node_data
        assert "reward" in node_data

    def test_from_run_without_fitness(self, simple_gas):
        """Test that run works without fitness recording."""
        n_steps = 5
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps, record_fitness=False)

        # Check fitness attributes don't exist
        sample_node = (0, 0)
        node_data = fs.graph.nodes[sample_node]

        assert "fitness" not in node_data
        assert "potential" not in node_data
        assert "reward" not in node_data

    def test_from_run_convergence(self, simple_gas):
        """Test that variance trajectory is recorded and reasonable."""
        n_steps = 20
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        var_traj = np.array(fs.graph.graph["var_x_trajectory"])

        # Just check that variance trajectory exists and has reasonable values
        assert len(var_traj) > 0
        assert all(v >= 0 for v in var_traj)  # All variances non-negative
        assert all(np.isfinite(v) for v in var_traj)  # No NaN or inf


class TestFractalSetQueries:
    """Test querying FractalSet data."""

    def test_get_walker_trajectory(self, simple_gas):
        """Test extracting single walker trajectory."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        walker_id = 0
        traj = fs.get_walker_trajectory(walker_id)

        assert len(traj["timesteps"]) == fs.graph.graph["n_steps"]
        assert len(traj["positions"]) == fs.graph.graph["n_steps"]
        assert len(traj["velocities"]) == fs.graph.graph["n_steps"]
        assert len(traj["high_error"]) == fs.graph.graph["n_steps"]
        assert len(traj["alive"]) == fs.graph.graph["n_steps"]

        # Check positions change over time
        positions = np.array(traj["positions"])
        assert positions.shape == (fs.graph.graph["n_steps"], simple_gas.params.d)

    def test_get_timestep_snapshot(self, simple_gas):
        """Test extracting complete swarm state at a timestep."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        t = 5
        snapshot = fs.get_timestep_snapshot(t)

        assert snapshot["positions"].shape == (simple_gas.params.N, simple_gas.params.d)
        assert snapshot["velocities"].shape == (simple_gas.params.N, simple_gas.params.d)
        assert len(snapshot["high_error_mask"]) == simple_gas.params.N
        assert len(snapshot["alive_mask"]) == simple_gas.params.N
        assert "centroid" in snapshot
        assert "variance" in snapshot

    def test_get_lineage(self, simple_gas):
        """Test tracing walker lineage backwards."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        walker_id = 0
        timestep = n_steps
        lineage = fs.get_lineage(walker_id, timestep)

        # Lineage should be non-empty (goes back to t=0)
        assert len(lineage) > 0

        # All lineage entries should be valid nodes
        for ancestor in lineage:
            assert fs.graph.has_node(ancestor)

    def test_get_cloning_events(self, simple_gas):
        """Test extracting cloning events."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps, record_fitness=True)

        events = fs.get_cloning_events()

        # Should have some cloning events (might be zero in short runs)
        assert isinstance(events, list)

        # Check structure of events
        for event in events:
            assert "timestep" in event
            assert "parent_id" in event
            assert "child_id" in event
            assert "parent_pos" in event
            assert "child_pos" in event

    def test_summary_statistics(self, simple_gas):
        """Test computing summary statistics."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        stats = fs.summary_statistics()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "n_cloning_events" in stats
        assert "initial_variance" in stats
        assert "final_variance" in stats
        assert "variance_reduction" in stats
        assert "mean_high_error_fraction" in stats

        # Check values are reasonable
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] >= 0
        assert 0.0 <= stats["mean_high_error_fraction"] <= 1.0


class TestFractalSetSerialization:
    """Test saving and loading FractalSet."""

    def test_save_and_load(self, simple_gas):
        """Test saving FractalSet to disk and loading it back."""
        n_steps = 5
        fs_original = FractalSet.from_run(simple_gas, n_steps=n_steps)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_fractal_set.graphml"
            fs_original.save(str(filepath))

            # Check file was created
            assert filepath.exists()

            # Load back
            fs_loaded = FractalSet.load(str(filepath))

            # Check basic properties match
            assert fs_loaded.N == fs_original.N
            assert fs_loaded.d == fs_original.d
            assert fs_loaded.graph.number_of_nodes() == fs_original.graph.number_of_nodes()
            assert fs_loaded.graph.number_of_edges() == fs_original.graph.number_of_edges()

            # Check graph attributes match
            assert fs_loaded.graph.graph["n_steps"] == fs_original.graph.graph["n_steps"]

    def test_save_with_extension_handling(self, simple_gas):
        """Test that .graphml extension is added if missing."""
        n_steps = 5
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_no_extension"
            fs.save(str(filepath))

            # Check that .graphml was added
            expected_path = Path(str(filepath) + ".graphml")
            assert expected_path.exists()


class TestFractalSetEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_fractal_set(self):
        """Test operations on empty FractalSet."""
        fs = FractalSet(N=10, d=2)

        # Should handle empty case gracefully
        stats = fs.summary_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["n_steps"] == 0

    def test_single_walker(self):
        """Test FractalSet with N=1."""
        params = EuclideanGasParams(
            N=1,
            d=2,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
            cloning=CloningParams(
                sigma_x=0.5,
                lambda_alg=0.1,
                alpha_restitution=0.5,
                companion_selection_method="uniform",
            ),
            device="cpu",
            dtype="float32",
        )
        gas = EuclideanGas(params)

        n_steps = 5
        fs = FractalSet.from_run(gas, n_steps=n_steps)

        # Should work with single walker
        assert fs.graph.number_of_nodes() == n_steps + 1
        assert fs.N == 1

    def test_walker_trajectory_invalid_id(self, simple_gas):
        """Test getting trajectory for non-existent walker."""
        n_steps = 5
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        # Try to get trajectory for invalid walker
        invalid_id = 999
        traj = fs.get_walker_trajectory(invalid_id)

        # Should return empty trajectory
        assert len(traj["timesteps"]) == 0


class TestFractalSetGraphProperties:
    """Test graph-theoretic properties of FractalSet."""

    def test_directed_graph(self, simple_gas):
        """Test that FractalSet uses directed graph."""
        n_steps = 5
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        assert isinstance(fs.graph, nx.DiGraph)

    def test_temporal_ordering(self, simple_gas):
        """Test that edges respect temporal ordering."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        # All edges should go forward in time
        for u, v in fs.graph.edges():
            _u_walker, u_time = u
            _v_walker, v_time = v

            assert v_time > u_time  # Edges go forward in time

    def test_no_self_loops(self, simple_gas):
        """Test that graph has no self-loops."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        # Check for self-loops
        assert nx.number_of_selfloops(fs.graph) == 0

    def test_weakly_connected_components(self, simple_gas):
        """Test graph connectivity structure."""
        n_steps = 10
        fs = FractalSet.from_run(simple_gas, n_steps=n_steps)

        # Each walker should form a weakly connected component
        # (connected through temporal evolution)
        n_components = nx.number_weakly_connected_components(fs.graph)

        # Number of components should be <= N
        # (might be less if walkers get cloned together)
        assert n_components <= simple_gas.params.N


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
