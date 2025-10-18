"""Tests for FractalSet integration with EuclideanGas."""

import pytest
import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)
from fragile.fractal_set import FractalSet


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


class TestEuclideanGasFractalSetIntegration:
    """Test FractalSet integration with EuclideanGas.run()."""

    def test_run_with_fractal_set_parameter(self, simple_gas):
        """Test that run() accepts and populates FractalSet parameter."""
        # Create FractalSet
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        # Run with FractalSet
        n_steps = 10
        result = simple_gas.run(n_steps=n_steps, fractal_set=fs)

        # Check that FractalSet was populated
        assert fs.graph.number_of_nodes() == simple_gas.params.N * (n_steps + 1)
        assert fs.graph.graph["n_steps"] == n_steps + 1
        assert result["fractal_set"] is fs

    def test_run_with_fitness_recording(self, simple_gas):
        """Test that fitness data is recorded when requested."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 5
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Check fitness attributes exist
        sample_node = (0, 0)
        node_data = fs.graph.nodes[sample_node]

        assert "fitness" in node_data
        assert "potential" in node_data
        assert "reward" in node_data
        assert node_data["fitness"] is not None

    def test_run_without_fitness_recording(self, simple_gas):
        """Test that fitness data is not recorded when not requested."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 5
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=False)

        # Check fitness attributes don't exist or are None
        sample_node = (0, 0)
        node_data = fs.graph.nodes[sample_node]

        # Attributes should be None when record_fitness=False
        assert node_data.get("fitness") is None
        assert node_data.get("potential") is None
        assert node_data.get("reward") is None

    def test_run_backward_compatibility(self, simple_gas):
        """Test that run() works without FractalSet parameter."""
        n_steps = 10
        result = simple_gas.run(n_steps=n_steps)

        # Should work normally without FractalSet
        assert "x" in result
        assert "v" in result
        assert "var_x" in result
        assert "fractal_set" not in result

    def test_fractal_set_trajectories_match_run_output(self, simple_gas):
        """Test that FractalSet trajectories match run() output."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 10
        result = simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Compare trajectories
        x_from_result = result["x"]  # Shape: [n_steps+1, N, d]
        var_x_from_result = result["var_x"]  # Shape: [n_steps+1]

        # Get variance trajectory from FractalSet
        var_x_from_fs = fs.graph.graph["var_x_trajectory"]

        # Should have same length
        assert len(var_x_from_fs) == n_steps + 1

        # Variance should approximately match
        for t in range(n_steps + 1):
            # Compute variance from result positions using correct formula
            # V_Var,x = (1/N) sum_i ||x_i - μ_x||²
            x_t = x_from_result[t]  # [N, d]
            mu_x = torch.mean(x_t, dim=0, keepdim=True)
            computed_var = float(torch.mean(torch.sum((x_t - mu_x) ** 2, dim=-1)))

            # Compare with stored values
            fs_var = var_x_from_fs[t]
            result_var = float(var_x_from_result[t])

            # Both should be close to each other and to computed value
            assert abs(fs_var - result_var) < 1e-4
            assert abs(fs_var - computed_var) < 1e-4

    def test_fractal_set_high_error_mask(self, simple_gas):
        """Test that high-error mask is computed for all timesteps."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 5
        simple_gas.run(n_steps=n_steps, fractal_set=fs)

        # Check all nodes have high_error attribute
        for t in range(n_steps + 1):
            for i in range(simple_gas.params.N):
                node_id = (i, t)
                assert fs.graph.has_node(node_id)
                node_data = fs.graph.nodes[node_id]

                assert "high_error" in node_data
                assert isinstance(node_data["high_error"], bool)
                assert "positional_error" in node_data

    def test_fractal_set_edges_created(self, simple_gas):
        """Test that temporal edges are created between timesteps."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 5
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Should have at least N * n_steps temporal edges
        assert fs.graph.number_of_edges() >= simple_gas.params.N * n_steps

        # Check temporal ordering
        for u, v in fs.graph.edges():
            _u_walker, u_time = u
            _v_walker, v_time = v

            # Edges should go forward in time
            assert v_time > u_time

    def test_fractal_set_cloning_edges(self, simple_gas):
        """Test that cloning edges are marked when recorded."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 10
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Check if any cloning edges exist
        cloning_edges = [
            (u, v)
            for (u, v) in fs.graph.edges()
            if fs.graph.edges[u, v].get("edge_type") == "cloning"
        ]

        # With fitness recording, we might have cloning edges
        # (Not guaranteed in short runs, but structure should be correct)
        for u, v in cloning_edges:
            edge_data = fs.graph.edges[u, v]
            assert "companion_id" in edge_data
            assert "cloning_prob" in edge_data
            assert edge_data["cloning_prob"] > 0.5

    def test_fractal_set_summary_statistics(self, simple_gas):
        """Test that summary statistics work after run."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 10
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        stats = fs.summary_statistics()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "n_steps" in stats
        assert "initial_variance" in stats
        assert "final_variance" in stats

        assert stats["total_nodes"] == simple_gas.params.N * (n_steps + 1)
        assert stats["n_steps"] == n_steps + 1

    def test_fractal_set_query_methods(self, simple_gas):
        """Test that FractalSet query methods work after run."""
        fs = FractalSet(N=simple_gas.params.N, d=simple_gas.params.d)

        n_steps = 10
        simple_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Test get_walker_trajectory
        walker_id = 0
        traj = fs.get_walker_trajectory(walker_id)

        assert len(traj["timesteps"]) == n_steps + 1
        assert len(traj["positions"]) == n_steps + 1
        assert len(traj["velocities"]) == n_steps + 1

        # Test get_timestep_snapshot
        snapshot = fs.get_timestep_snapshot(5)

        assert snapshot["positions"].shape == (simple_gas.params.N, simple_gas.params.d)
        assert "centroid" in snapshot
        assert "variance" in snapshot

        # Test get_lineage
        lineage = fs.get_lineage(walker_id, n_steps)
        assert len(lineage) > 0

        # Test get_cloning_events
        events = fs.get_cloning_events()
        assert isinstance(events, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
