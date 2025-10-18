"""Tests for FractalSet integration with GeometricGas."""

import pytest
import torch

from fragile.euclidean_gas import LangevinParams, SimpleQuadraticPotential
from fragile.fractal_set import FractalSet
from fragile.geometric_gas import (
    AdaptiveParams,
    GeometricGas,
    GeometricGasParams,
    LocalizationKernelParams,
)


@pytest.fixture
def simple_geometric_gas():
    """Create a simple GeometricGas instance."""
    params = GeometricGasParams(
        N=10,
        d=2,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
        localization=LocalizationKernelParams(rho=1.0, kernel_type='gaussian'),
        adaptive=AdaptiveParams(
            epsilon_F=0.1,
            nu=0.05,
            epsilon_Sigma=0.01,
            rescale_amplitude=1.0,
            sigma_var_min=0.1,
            viscous_length_scale=1.0,
        ),
        device='cpu',
        dtype='float32',
        freeze_best=False,
    )
    return GeometricGas(params)


class TestGeometricGasFractalSetIntegration:
    """Test FractalSet integration with GeometricGas.run()."""

    def test_run_with_fractal_set_parameter(self, simple_geometric_gas):
        """Test that run() accepts and populates FractalSet parameter."""
        # Create FractalSet
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        # Run with FractalSet
        n_steps = 10
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs)

        # Check that FractalSet was populated
        assert fs.graph.number_of_nodes() == simple_geometric_gas.params.N * (n_steps + 1)
        assert fs.graph.graph['n_steps'] == n_steps + 1
        assert result['fractal_set'] is fs

    def test_run_with_fitness_recording(self, simple_geometric_gas):
        """Test that fitness data is recorded when requested."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 5
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Check fitness attributes exist
        sample_node = (0, 0)
        node_data = fs.graph.nodes[sample_node]

        assert 'fitness' in node_data
        assert 'potential' in node_data
        assert 'reward' in node_data
        assert node_data['fitness'] is not None

    def test_run_without_fitness_recording(self, simple_geometric_gas):
        """Test that fitness data is not recorded when not requested."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 5
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=False)

        # Check fitness attributes don't exist or are None
        sample_node = (0, 0)
        node_data = fs.graph.nodes[sample_node]

        # Attributes should be None when record_fitness=False
        assert node_data.get('fitness') is None
        assert node_data.get('potential') is None
        assert node_data.get('reward') is None

    def test_run_backward_compatibility(self, simple_geometric_gas):
        """Test that run() works without FractalSet parameter."""
        n_steps = 10
        result = simple_geometric_gas.run(n_steps=n_steps)

        # Should work normally without FractalSet
        assert 'x' in result
        assert 'v' in result
        assert 'fitness' in result
        assert 'fractal_set' not in result

    def test_fractal_set_high_error_mask(self, simple_geometric_gas):
        """Test that high-error mask is computed for all timesteps."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 5
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs)

        # Check all nodes have high_error attribute
        for t in range(n_steps + 1):
            for i in range(simple_geometric_gas.params.N):
                node_id = (i, t)
                assert fs.graph.has_node(node_id)
                node_data = fs.graph.nodes[node_id]

                assert 'high_error' in node_data
                assert isinstance(node_data['high_error'], bool)
                assert 'positional_error' in node_data

    def test_fractal_set_edges_created(self, simple_geometric_gas):
        """Test that temporal edges are created between timesteps."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 5
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Should have at least N * n_steps temporal edges
        assert fs.graph.number_of_edges() >= simple_geometric_gas.params.N * n_steps

        # Check temporal ordering
        for (u, v) in fs.graph.edges():
            u_walker, u_time = u
            v_walker, v_time = v

            # Edges should go forward in time
            assert v_time > u_time

    def test_fractal_set_no_cloning_edges(self, simple_geometric_gas):
        """Test that GeometricGas doesn't create cloning edges (no cloning operator)."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 10
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Check that no cloning edges exist (GeometricGas doesn't use cloning)
        cloning_edges = [
            (u, v) for (u, v) in fs.graph.edges()
            if fs.graph.edges[(u, v)].get('edge_type') == 'cloning'
        ]

        # GeometricGas should have zero cloning edges
        assert len(cloning_edges) == 0

        # All edges should be kinetic
        kinetic_edges = [
            (u, v) for (u, v) in fs.graph.edges()
            if fs.graph.edges[(u, v)].get('edge_type') == 'kinetic'
        ]
        assert len(kinetic_edges) == fs.graph.number_of_edges()

    def test_fractal_set_summary_statistics(self, simple_geometric_gas):
        """Test that summary statistics work after run."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 10
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        stats = fs.summary_statistics()

        assert 'total_nodes' in stats
        assert 'total_edges' in stats
        assert 'n_steps' in stats
        assert 'initial_variance' in stats
        assert 'final_variance' in stats

        assert stats['total_nodes'] == simple_geometric_gas.params.N * (n_steps + 1)
        assert stats['n_steps'] == n_steps + 1
        assert stats['n_cloning_events'] == 0  # GeometricGas has no cloning

    def test_fractal_set_query_methods(self, simple_geometric_gas):
        """Test that FractalSet query methods work after run."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 10
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Test get_walker_trajectory
        walker_id = 0
        traj = fs.get_walker_trajectory(walker_id)

        assert len(traj['timesteps']) == n_steps + 1
        assert len(traj['positions']) == n_steps + 1
        assert len(traj['velocities']) == n_steps + 1

        # Test get_timestep_snapshot
        snapshot = fs.get_timestep_snapshot(5)

        assert snapshot['positions'].shape == (simple_geometric_gas.params.N, simple_geometric_gas.params.d)
        assert 'centroid' in snapshot
        assert 'variance' in snapshot

        # Test get_lineage (should just be linear for GeometricGas)
        lineage = fs.get_lineage(walker_id, n_steps)
        # GeometricGas has no cloning, so lineage should be linear
        # Lineage doesn't include the starting node, so length is n_steps (not n_steps+1)
        assert len(lineage) == n_steps
        assert lineage[0] == (walker_id, n_steps - 1)
        assert lineage[-1] == (walker_id, 0)

        # Test get_cloning_events (should be empty)
        events = fs.get_cloning_events()
        assert len(events) == 0

    def test_fractal_set_fitness_values(self, simple_geometric_gas):
        """Test that fitness values are correctly recorded."""
        fs = FractalSet(N=simple_geometric_gas.params.N, d=simple_geometric_gas.params.d)

        n_steps = 5
        result = simple_geometric_gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

        # Check that fitness values are recorded
        for t in range(n_steps + 1):
            for i in range(simple_geometric_gas.params.N):
                node_id = (i, t)
                node_data = fs.graph.nodes[node_id]

                # Should have fitness value
                assert 'fitness' in node_data
                assert node_data['fitness'] is not None
                assert isinstance(node_data['fitness'], float)

                # Fitness should be non-negative (it's from fitness potential)
                assert node_data['fitness'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
