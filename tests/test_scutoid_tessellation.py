"""Tests for Scutoid Tessellation."""

import pytest
import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)
from fragile.scutoid import ScutoidTessellation


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
            companion_selection_method='softmax'
        ),
        device='cpu',
        dtype='float32'
    )
    return EuclideanGas(params)


class TestScutoidTessellation:
    """Test ScutoidTessellation integration with EuclideanGas."""

    def test_tessellation_initialization(self, simple_gas):
        """Test that ScutoidTessellation can be created."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        assert tessellation.N == 10
        assert tessellation.d == 2
        assert len(tessellation.voronoi_cells) == 0
        assert len(tessellation.scutoid_cells) == 0

    def test_run_with_scutoid_tessellation(self, simple_gas):
        """Test that run() accepts and populates ScutoidTessellation."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 5
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Check that tessellation was populated
        assert len(tessellation.voronoi_cells) == n_steps + 1
        assert len(tessellation.scutoid_cells) == n_steps
        assert tessellation.n_steps == n_steps + 1
        assert result['scutoid_tessellation'] is tessellation

    def test_voronoi_cells_created(self, simple_gas):
        """Test that Voronoi cells are created at each timestep."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 3
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Check each timestep has N Voronoi cells
        for t in range(n_steps + 1):
            voronoi_cells_at_t = tessellation.voronoi_cells[t]
            assert len(voronoi_cells_at_t) == simple_gas.params.N

            # Check each cell has basic attributes
            for cell in voronoi_cells_at_t:
                assert cell.walker_id >= 0
                assert cell.walker_id < simple_gas.params.N
                assert cell.center is not None
                assert len(cell.center) == simple_gas.params.d
                assert cell.t == pytest.approx(t * simple_gas.params.langevin.delta_t, abs=1e-6)

    def test_scutoid_cells_created(self, simple_gas):
        """Test that scutoid cells are created between timesteps."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 3
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Check each interval has N scutoid cells
        for interval in range(n_steps):
            scutoids_at_interval = tessellation.scutoid_cells[interval]
            assert len(scutoids_at_interval) == simple_gas.params.N

            # Check each scutoid has basic attributes
            for scutoid in scutoids_at_interval:
                assert scutoid.walker_id >= 0
                assert scutoid.walker_id < simple_gas.params.N
                assert scutoid.parent_id >= 0
                assert scutoid.parent_id < simple_gas.params.N
                assert scutoid.t_start < scutoid.t_end
                assert scutoid.bottom_center is not None
                assert scutoid.top_center is not None

    def test_scutoid_prism_classification(self, simple_gas):
        """Test that scutoids are classified as prisms or true scutoids."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 3
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Check classification methods work
        for interval in range(n_steps):
            scutoids = tessellation.scutoid_cells[interval]
            for scutoid in scutoids:
                # These methods should not error
                is_prism = scutoid.is_prism()
                shared = scutoid.shared_neighbors()
                lost = scutoid.lost_neighbors()
                gained = scutoid.gained_neighbors()

                assert isinstance(is_prism, bool)
                assert isinstance(shared, list)
                assert isinstance(lost, list)
                assert isinstance(gained, list)

    def test_summary_statistics(self, simple_gas):
        """Test that summary statistics work."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 5
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        stats = tessellation.summary_statistics()

        assert 'n_timesteps' in stats
        assert 'n_intervals' in stats
        assert 'n_prisms' in stats
        assert 'n_scutoids' in stats
        assert 'total_spacetime_volume' in stats
        assert 'N' in stats
        assert 'd' in stats

        assert stats['n_timesteps'] == n_steps + 1
        assert stats['n_intervals'] == n_steps
        assert stats['N'] == simple_gas.params.N
        assert stats['d'] == simple_gas.params.d

    def test_get_scutoid(self, simple_gas):
        """Test retrieving specific scutoid cells."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 3
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Get scutoid for walker 0 at timestep 0
        scutoid = tessellation.get_scutoid(walker_id=0, timestep=0)

        assert scutoid is not None
        assert scutoid.walker_id == 0
        assert scutoid.t_start == pytest.approx(0.0, abs=1e-6)

        # Test invalid queries
        assert tessellation.get_scutoid(walker_id=0, timestep=-1) is None
        assert tessellation.get_scutoid(walker_id=0, timestep=99) is None

    def test_backward_compatibility(self, simple_gas):
        """Test that run() works without ScutoidTessellation."""
        n_steps = 5
        result = simple_gas.run(n_steps=n_steps)

        # Should work normally without scutoid_tessellation
        assert 'x' in result
        assert 'v' in result
        assert 'scutoid_tessellation' not in result

    def test_timestep_tracking(self, simple_gas):
        """Test that timesteps are correctly tracked."""
        tessellation = ScutoidTessellation(
            N=simple_gas.params.N,
            d=simple_gas.params.d,
        )

        n_steps = 4
        dt = simple_gas.params.langevin.delta_t
        result = simple_gas.run(n_steps=n_steps, scutoid_tessellation=tessellation)

        # Check timesteps are recorded correctly
        assert len(tessellation.timesteps) == n_steps + 1
        for i, t in enumerate(tessellation.timesteps):
            assert t == pytest.approx(i * dt, abs=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
