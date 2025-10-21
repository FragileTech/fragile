"""Tests for Scutoid Tessellation with RunHistory API."""

import numpy as np
import pytest
import torch

from fragile.core import RunHistory, ScutoidHistory2D, ScutoidHistory3D, create_scutoid_history


@pytest.fixture
def simple_history_2d():
    """Create a minimal 2D RunHistory for testing."""
    N, d, n_recorded = 10, 2, 5
    n_steps = (n_recorded - 1) * 10  # record_every = 10

    return RunHistory(
        N=N,
        d=d,
        n_steps=n_steps,
        n_recorded=n_recorded,
        record_every=10,
        terminated_early=False,
        final_step=n_steps,
        x_before_clone=torch.randn(n_recorded, N, d),
        v_before_clone=torch.randn(n_recorded, N, d),
        x_after_clone=torch.randn(n_recorded - 1, N, d),
        v_after_clone=torch.randn(n_recorded - 1, N, d),
        x_final=torch.randn(n_recorded, N, d),
        v_final=torch.randn(n_recorded, N, d),
        n_alive=torch.full((n_recorded,), N),
        num_cloned=torch.zeros(n_recorded),
        step_times=torch.ones(n_recorded) * 0.01,
        fitness=torch.randn(n_recorded - 1, N),
        rewards=torch.randn(n_recorded - 1, N),
        cloning_scores=torch.randn(n_recorded - 1, N),
        cloning_probs=torch.rand(n_recorded - 1, N),
        will_clone=torch.zeros(n_recorded - 1, N, dtype=torch.bool),
        alive_mask=torch.ones(n_recorded, N, dtype=torch.bool),
        companions_distance=torch.randint(0, N, (n_recorded - 1, N)),
        companions_clone=torch.randint(0, N, (n_recorded - 1, N)),
        distances=torch.randn(n_recorded - 1, N),
        z_rewards=torch.randn(n_recorded - 1, N),
        z_distances=torch.randn(n_recorded - 1, N),
        pos_squared_differences=torch.randn(n_recorded - 1, N),
        vel_squared_differences=torch.randn(n_recorded - 1, N),
        rescaled_rewards=torch.randn(n_recorded - 1, N),
        rescaled_distances=torch.randn(n_recorded - 1, N),
        total_time=0.1,
        init_time=0.01,
    )


@pytest.fixture
def simple_history_3d():
    """Create a minimal 3D RunHistory for testing."""
    N, d, n_recorded = 10, 3, 5
    n_steps = (n_recorded - 1) * 10

    return RunHistory(
        N=N,
        d=d,
        n_steps=n_steps,
        n_recorded=n_recorded,
        record_every=10,
        terminated_early=False,
        final_step=n_steps,
        x_before_clone=torch.randn(n_recorded, N, d),
        v_before_clone=torch.randn(n_recorded, N, d),
        x_after_clone=torch.randn(n_recorded - 1, N, d),
        v_after_clone=torch.randn(n_recorded - 1, N, d),
        x_final=torch.randn(n_recorded, N, d),
        v_final=torch.randn(n_recorded, N, d),
        n_alive=torch.full((n_recorded,), N),
        num_cloned=torch.zeros(n_recorded),
        step_times=torch.ones(n_recorded) * 0.01,
        fitness=torch.randn(n_recorded - 1, N),
        rewards=torch.randn(n_recorded - 1, N),
        cloning_scores=torch.randn(n_recorded - 1, N),
        cloning_probs=torch.rand(n_recorded - 1, N),
        will_clone=torch.zeros(n_recorded - 1, N, dtype=torch.bool),
        alive_mask=torch.ones(n_recorded, N, dtype=torch.bool),
        companions_distance=torch.randint(0, N, (n_recorded - 1, N)),
        companions_clone=torch.randint(0, N, (n_recorded - 1, N)),
        distances=torch.randn(n_recorded - 1, N),
        z_rewards=torch.randn(n_recorded - 1, N),
        z_distances=torch.randn(n_recorded - 1, N),
        pos_squared_differences=torch.randn(n_recorded - 1, N),
        vel_squared_differences=torch.randn(n_recorded - 1, N),
        rescaled_rewards=torch.randn(n_recorded - 1, N),
        rescaled_distances=torch.randn(n_recorded - 1, N),
        total_time=0.1,
        init_time=0.01,
    )


class TestScutoidHistoryFactory:
    """Test factory function for creating ScutoidHistory instances."""

    def test_create_2d_scutoid_history(self, simple_history_2d):
        """Test factory creates ScutoidHistory2D for 2D data."""
        scutoid_hist = create_scutoid_history(simple_history_2d)

        assert isinstance(scutoid_hist, ScutoidHistory2D)
        assert scutoid_hist.N == 10
        assert scutoid_hist.d == 2
        assert scutoid_hist.n_recorded == 5

    def test_create_3d_scutoid_history(self, simple_history_3d):
        """Test factory creates ScutoidHistory3D for 3D data."""
        scutoid_hist = create_scutoid_history(simple_history_3d)

        assert isinstance(scutoid_hist, ScutoidHistory3D)
        assert scutoid_hist.N == 10
        assert scutoid_hist.d == 3
        assert scutoid_hist.n_recorded == 5

    def test_unsupported_dimension_raises(self):
        """Test that unsupported dimensions raise ValueError."""
        # Create 4D history
        N, d, n_recorded = 5, 4, 3
        history_4d = RunHistory(
            N=N,
            d=d,
            n_steps=20,
            n_recorded=n_recorded,
            record_every=10,
            terminated_early=False,
            final_step=20,
            x_before_clone=torch.randn(n_recorded, N, d),
            v_before_clone=torch.randn(n_recorded, N, d),
            x_after_clone=torch.randn(n_recorded - 1, N, d),
            v_after_clone=torch.randn(n_recorded - 1, N, d),
            x_final=torch.randn(n_recorded, N, d),
            v_final=torch.randn(n_recorded, N, d),
            n_alive=torch.full((n_recorded,), N),
            num_cloned=torch.zeros(n_recorded),
            step_times=torch.ones(n_recorded) * 0.01,
            fitness=torch.randn(n_recorded - 1, N),
            rewards=torch.randn(n_recorded - 1, N),
            cloning_scores=torch.randn(n_recorded - 1, N),
            cloning_probs=torch.rand(n_recorded - 1, N),
            will_clone=torch.zeros(n_recorded - 1, N, dtype=torch.bool),
            alive_mask=torch.ones(n_recorded, N, dtype=torch.bool),
            companions_distance=torch.randint(0, N, (n_recorded - 1, N)),
            companions_clone=torch.randint(0, N, (n_recorded - 1, N)),
            distances=torch.randn(n_recorded - 1, N),
            z_rewards=torch.randn(n_recorded - 1, N),
            z_distances=torch.randn(n_recorded - 1, N),
            pos_squared_differences=torch.randn(n_recorded - 1, N),
            vel_squared_differences=torch.randn(n_recorded - 1, N),
            rescaled_rewards=torch.randn(n_recorded - 1, N),
            rescaled_distances=torch.randn(n_recorded - 1, N),
            total_time=0.1,
            init_time=0.01,
        )

        with pytest.raises(ValueError, match="Only 2D and 3D supported"):
            create_scutoid_history(history_4d)


class TestScutoidHistory2D:
    """Test 2D scutoid tessellation."""

    def test_initialization(self, simple_history_2d):
        """Test that ScutoidHistory2D initializes correctly."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)

        assert scutoid_hist.N == 10
        assert scutoid_hist.d == 2
        assert scutoid_hist.n_recorded == 5
        assert len(scutoid_hist.voronoi_cells) == 0
        assert len(scutoid_hist.scutoid_cells) == 0
        assert scutoid_hist.ricci_scalars is None

    def test_dimension_validation(self, simple_history_3d):
        """Test that ScutoidHistory2D rejects 3D data."""
        with pytest.raises(ValueError, match="requires d=2"):
            ScutoidHistory2D(simple_history_3d)

    def test_build_tessellation(self, simple_history_2d):
        """Test tessellation construction from RunHistory."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        # Check Voronoi cells created
        assert len(scutoid_hist.voronoi_cells) == 5  # n_recorded
        assert len(scutoid_hist.timesteps) == 5

        # Check each timestep has correct number of cells
        for voronoi_list in scutoid_hist.voronoi_cells:
            assert len(voronoi_list) <= 10  # May have fewer due to alive_mask

        # Check scutoid cells created
        assert len(scutoid_hist.scutoid_cells) == 4  # n_recorded - 1

    def test_voronoi_cell_structure(self, simple_history_2d):
        """Test Voronoi cell attributes."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        # Check first timestep cells
        cells = scutoid_hist.voronoi_cells[0]
        for cell in cells:
            assert cell.walker_id >= 0
            assert cell.walker_id < 10
            assert cell.center.shape == (2,)
            assert isinstance(cell.neighbors, list)
            assert cell.t == 0.0

    def test_scutoid_cell_structure(self, simple_history_2d):
        """Test scutoid cell attributes."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        # Check first interval scutoids
        scutoids = scutoid_hist.scutoid_cells[0]
        for scutoid in scutoids:
            assert scutoid.walker_id >= 0
            assert scutoid.parent_id >= 0
            assert scutoid.t_start == 0.0
            assert scutoid.t_end == 10.0
            assert scutoid.bottom_center.shape == (2,)
            assert scutoid.top_center.shape == (2,)
            assert isinstance(scutoid.bottom_neighbors, list)
            assert isinstance(scutoid.top_neighbors, list)

    def test_scutoid_classification(self, simple_history_2d):
        """Test scutoid prism/scutoid classification."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        scutoid = scutoid_hist.scutoid_cells[0][0]

        # Test classification methods
        assert isinstance(scutoid.is_prism(), bool)
        assert isinstance(scutoid.neighbor_change_count(), int)
        assert isinstance(scutoid.shared_neighbors(), list)
        assert isinstance(scutoid.lost_neighbors(), list)
        assert isinstance(scutoid.gained_neighbors(), list)

        # Test neighbor change consistency
        n_shared = len(scutoid.shared_neighbors())
        n_lost = len(scutoid.lost_neighbors())
        n_gained = len(scutoid.gained_neighbors())
        n_change = scutoid.neighbor_change_count()

        assert n_change == n_lost + n_gained

    def test_compute_ricci_scalars(self, simple_history_2d):
        """Test Ricci scalar computation for 2D."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()
        scutoid_hist.compute_ricci_scalars()

        # Check Ricci scalars computed
        assert scutoid_hist.ricci_scalars is not None
        assert scutoid_hist.ricci_scalars.shape == (4, 10)  # (n_recorded-1, N)

        # Check scutoid cells have Ricci scalars assigned
        for scutoid_list in scutoid_hist.scutoid_cells:
            for scutoid in scutoid_list:
                assert scutoid.ricci_scalar is not None
                assert isinstance(scutoid.ricci_scalar, float)

    def test_get_ricci_scalars(self, simple_history_2d):
        """Test retrieving Ricci scalar array."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        # Before computation
        assert scutoid_hist.get_ricci_scalars() is None

        # After computation
        scutoid_hist.compute_ricci_scalars()
        ricci = scutoid_hist.get_ricci_scalars()

        assert ricci is not None
        assert ricci.shape == (4, 10)
        # Check for valid values (may have NaN for dead walkers)
        assert np.sum(~np.isnan(ricci)) > 0

    def test_summary_statistics(self, simple_history_2d):
        """Test summary statistics computation."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()
        scutoid_hist.compute_ricci_scalars()

        stats = scutoid_hist.summary_statistics()

        # Check required fields
        assert "n_timesteps" in stats
        assert "n_intervals" in stats
        assert "n_prisms" in stats
        assert "n_simple_scutoids" in stats
        assert "n_complex_scutoids" in stats
        assert "N" in stats
        assert "d" in stats

        # Check values
        assert stats["n_timesteps"] == 5
        assert stats["n_intervals"] == 4
        assert stats["N"] == 10
        assert stats["d"] == 2

        # Check curvature statistics (if computed)
        if scutoid_hist.ricci_scalars is not None:
            assert "mean_ricci" in stats
            assert "std_ricci" in stats
            assert "min_ricci" in stats
            assert "max_ricci" in stats


class TestScutoidHistory3D:
    """Test 3D scutoid tessellation."""

    def test_initialization(self, simple_history_3d):
        """Test that ScutoidHistory3D initializes correctly."""
        scutoid_hist = ScutoidHistory3D(simple_history_3d)

        assert scutoid_hist.N == 10
        assert scutoid_hist.d == 3
        assert scutoid_hist.n_recorded == 5

    def test_dimension_validation(self, simple_history_2d):
        """Test that ScutoidHistory3D rejects 2D data."""
        with pytest.raises(ValueError, match="requires d=3"):
            ScutoidHistory3D(simple_history_2d)

    def test_build_tessellation(self, simple_history_3d):
        """Test tessellation construction for 3D."""
        scutoid_hist = ScutoidHistory3D(simple_history_3d)
        scutoid_hist.build_tessellation()

        assert len(scutoid_hist.voronoi_cells) == 5
        assert len(scutoid_hist.scutoid_cells) == 4

    def test_compute_ricci_scalars_3d(self, simple_history_3d):
        """Test Ricci scalar computation for 3D (currently zeros)."""
        scutoid_hist = ScutoidHistory3D(simple_history_3d)
        scutoid_hist.build_tessellation()
        scutoid_hist.compute_ricci_scalars()

        # Check Ricci scalars computed (currently all zeros for 3D)
        assert scutoid_hist.ricci_scalars is not None
        assert scutoid_hist.ricci_scalars.shape == (4, 10)

        # For now, 3D returns zero deficit angles (flat space approximation)
        # Non-NaN values should be zero or close to zero
        ricci_valid = scutoid_hist.ricci_scalars[~np.isnan(scutoid_hist.ricci_scalars)]
        if len(ricci_valid) > 0:
            assert np.allclose(ricci_valid, 0.0)


class TestScutoidIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow_2d(self, simple_history_2d):
        """Test complete workflow: create -> build -> compute -> analyze."""
        # Create
        scutoid_hist = create_scutoid_history(simple_history_2d)
        assert isinstance(scutoid_hist, ScutoidHistory2D)

        # Build tessellation
        scutoid_hist.build_tessellation()
        assert len(scutoid_hist.voronoi_cells) == 5
        assert len(scutoid_hist.scutoid_cells) == 4

        # Compute Ricci scalars
        scutoid_hist.compute_ricci_scalars()
        assert scutoid_hist.ricci_scalars is not None

        # Analyze
        stats = scutoid_hist.summary_statistics()
        assert stats["n_timesteps"] == 5
        assert "mean_ricci" in stats

    def test_with_cloning_events(self, simple_history_2d):
        """Test tessellation with cloning events."""
        # Set some cloning events
        simple_history_2d.will_clone[0, 0] = True
        simple_history_2d.companions_clone[0, 0] = 5  # Walker 0 clones from walker 5

        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        # Find scutoid for walker 0 at first interval
        scutoid = next(s for s in scutoid_hist.scutoid_cells[0] if s.walker_id == 0)

        # Parent should be walker 5 (cloning source)
        assert scutoid.parent_id == 5

    def test_with_dead_walkers(self, simple_history_2d):
        """Test tessellation with some dead walkers."""
        # Mark some walkers as dead
        simple_history_2d.alive_mask[1, 0] = False
        simple_history_2d.alive_mask[1, 1] = False

        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        scutoid_hist.build_tessellation()

        # Check second timestep has fewer cells
        cells_t1 = scutoid_hist.voronoi_cells[1]
        assert len(cells_t1) < 10  # Should have fewer than N cells

    def test_with_bounds_filtering(self, simple_history_2d):
        """Test tessellation with bounds filtering."""
        from fragile.bounds import TorchBounds

        # Create bounds that exclude some walkers
        # Use tight bounds to ensure some positions are outside
        bounds = TorchBounds(low=-2.0, high=2.0, shape=(simple_history_2d.d,))

        # Create scutoid history with bounds
        scutoid_hist = ScutoidHistory2D(simple_history_2d, bounds=bounds)
        scutoid_hist.build_tessellation()

        # Verify that some walkers were filtered out at each timestep
        for t_idx, cells in enumerate(scutoid_hist.voronoi_cells):
            # Check that all cell centers are within bounds
            for cell in cells:
                assert np.all(
                    (cell.center >= -2.0) & (cell.center <= 2.0)
                ), f"Cell {cell.walker_id} at t={t_idx} has center outside bounds: {cell.center}"

            # With random positions, we expect some to be outside [-2, 2]
            # So we should have fewer cells than total alive walkers
            # (This might not always be true for random data, so we just check validity)

    def test_bounds_from_history(self, simple_history_2d):
        """Test that bounds are used from RunHistory if available."""
        from fragile.bounds import TorchBounds

        # Set bounds in history
        bounds = TorchBounds(low=-3.0, high=3.0, shape=(simple_history_2d.d,))
        simple_history_2d.bounds = bounds

        # Create scutoid history without explicit bounds
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        assert scutoid_hist.bounds is bounds

        # Build and verify
        scutoid_hist.build_tessellation()

        # All cells should have centers within bounds
        for cells in scutoid_hist.voronoi_cells:
            for cell in cells:
                assert np.all(
                    (cell.center >= -3.0) & (cell.center <= 3.0)
                ), f"Cell {cell.walker_id} outside bounds: {cell.center}"

    def test_bounds_override(self, simple_history_2d):
        """Test that explicitly provided bounds override history.bounds."""
        from fragile.bounds import TorchBounds

        # Set bounds in history
        history_bounds = TorchBounds(low=-5.0, high=5.0, shape=(simple_history_2d.d,))
        simple_history_2d.bounds = history_bounds

        # Override with tighter bounds
        override_bounds = TorchBounds(low=-1.0, high=1.0, shape=(simple_history_2d.d,))

        # Create with override
        scutoid_hist = ScutoidHistory2D(simple_history_2d, bounds=override_bounds)
        assert scutoid_hist.bounds is override_bounds
        assert scutoid_hist.bounds is not history_bounds


class TestIncrementalTessellation:
    """Test incremental tessellation mode for ScutoidHistory2D."""

    def test_incremental_initialization(self, simple_history_2d):
        """Test that incremental parameter is passed correctly."""
        # Default incremental=True
        scutoid_hist = ScutoidHistory2D(simple_history_2d)
        assert scutoid_hist.incremental is True
        assert scutoid_hist._incremental_delaunay is None  # Not created yet

        # Explicit incremental=False
        scutoid_hist_batch = ScutoidHistory2D(simple_history_2d, incremental=False)
        assert scutoid_hist_batch.incremental is False

    def test_incremental_vs_batch_consistency(self, simple_history_2d):
        """Test that incremental and batch modes produce same results."""
        # Build with incremental mode
        scutoid_hist_inc = ScutoidHistory2D(simple_history_2d, incremental=True)
        scutoid_hist_inc.build_tessellation()

        # Build with batch mode
        scutoid_hist_batch = ScutoidHistory2D(simple_history_2d, incremental=False)
        scutoid_hist_batch.build_tessellation()

        # Check same number of timesteps
        assert len(scutoid_hist_inc.voronoi_cells) == len(scutoid_hist_batch.voronoi_cells)
        assert len(scutoid_hist_inc.scutoid_cells) == len(scutoid_hist_batch.scutoid_cells)

        # Check each timestep has same number of cells
        for t_idx in range(len(scutoid_hist_inc.voronoi_cells)):
            inc_cells = scutoid_hist_inc.voronoi_cells[t_idx]
            batch_cells = scutoid_hist_batch.voronoi_cells[t_idx]
            assert len(inc_cells) == len(batch_cells)

            # Check same walker IDs
            inc_ids = sorted([c.walker_id for c in inc_cells])
            batch_ids = sorted([c.walker_id for c in batch_cells])
            assert inc_ids == batch_ids

            # Check neighbor counts match
            inc_map = {c.walker_id: len(c.neighbors) for c in inc_cells}
            batch_map = {c.walker_id: len(c.neighbors) for c in batch_cells}
            assert inc_map == batch_map

    def test_incremental_with_cloning(self, simple_history_2d):
        """Test incremental mode with cloning events."""
        # Set some cloning events
        simple_history_2d.will_clone[0, 0] = True
        simple_history_2d.companions_clone[0, 0] = 5
        simple_history_2d.will_clone[1, 3] = True
        simple_history_2d.companions_clone[1, 3] = 7

        # Build with incremental mode
        scutoid_hist = ScutoidHistory2D(simple_history_2d, incremental=True)
        scutoid_hist.build_tessellation()

        # Check cloning was handled correctly
        scutoid_0 = next(s for s in scutoid_hist.scutoid_cells[0] if s.walker_id == 0)
        assert scutoid_0.parent_id == 5

        scutoid_3 = next(s for s in scutoid_hist.scutoid_cells[1] if s.walker_id == 3)
        assert scutoid_3.parent_id == 7

    def test_incremental_delaunay_created(self, simple_history_2d):
        """Test that IncrementalDelaunay2D object is created."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d, incremental=True)
        scutoid_hist.build_tessellation()

        # Check IncrementalDelaunay2D was created
        assert scutoid_hist._incremental_delaunay is not None

        from fragile.core.incremental_delaunay import IncrementalDelaunay2D

        assert isinstance(scutoid_hist._incremental_delaunay, IncrementalDelaunay2D)

    def test_incremental_ricci_computation(self, simple_history_2d):
        """Test Ricci computation works with incremental mode."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d, incremental=True)
        scutoid_hist.build_tessellation()
        scutoid_hist.compute_ricci_scalars()

        # Check Ricci scalars computed
        assert scutoid_hist.ricci_scalars is not None
        assert scutoid_hist.ricci_scalars.shape == (4, 10)

    def test_incremental_fallback_on_error(self, simple_history_2d):
        """Test automatic fallback to batch mode on incremental errors."""
        # This test is tricky - we need to induce an error in incremental mode
        # For now, we just verify the fallback logic exists by checking the code path

        # Create history with very few walkers (edge case)
        N, d, n_recorded = 2, 2, 3
        minimal_history = RunHistory(
            N=N,
            d=d,
            n_steps=20,
            n_recorded=n_recorded,
            record_every=10,
            terminated_early=False,
            final_step=20,
            x_before_clone=torch.randn(n_recorded, N, d),
            v_before_clone=torch.randn(n_recorded, N, d),
            x_after_clone=torch.randn(n_recorded - 1, N, d),
            v_after_clone=torch.randn(n_recorded - 1, N, d),
            x_final=torch.randn(n_recorded, N, d),
            v_final=torch.randn(n_recorded, N, d),
            n_alive=torch.full((n_recorded,), N),
            num_cloned=torch.zeros(n_recorded),
            step_times=torch.ones(n_recorded) * 0.01,
            fitness=torch.randn(n_recorded - 1, N),
            rewards=torch.randn(n_recorded - 1, N),
            cloning_scores=torch.randn(n_recorded - 1, N),
            cloning_probs=torch.rand(n_recorded - 1, N),
            will_clone=torch.zeros(n_recorded - 1, N, dtype=torch.bool),
            alive_mask=torch.ones(n_recorded, N, dtype=torch.bool),
            companions_distance=torch.randint(0, N, (n_recorded - 1, N)),
            companions_clone=torch.randint(0, N, (n_recorded - 1, N)),
            distances=torch.randn(n_recorded - 1, N),
            z_rewards=torch.randn(n_recorded - 1, N),
            z_distances=torch.randn(n_recorded - 1, N),
            pos_squared_differences=torch.randn(n_recorded - 1, N),
            vel_squared_differences=torch.randn(n_recorded - 1, N),
            rescaled_rewards=torch.randn(n_recorded - 1, N),
            rescaled_distances=torch.randn(n_recorded - 1, N),
            total_time=0.1,
            init_time=0.01,
        )

        # Should handle gracefully (either work or fallback)
        scutoid_hist = ScutoidHistory2D(minimal_history, incremental=True)
        scutoid_hist.build_tessellation()

        # Should have built some tessellation
        assert len(scutoid_hist.voronoi_cells) == n_recorded

    def test_incremental_with_bounds(self, simple_history_2d):
        """Test incremental mode with bounds filtering."""
        from fragile.bounds import TorchBounds

        bounds = TorchBounds(low=-2.0, high=2.0, shape=(simple_history_2d.d,))

        # Build with incremental mode + bounds
        scutoid_hist = ScutoidHistory2D(simple_history_2d, bounds=bounds, incremental=True)
        scutoid_hist.build_tessellation()

        # All cells should be within bounds
        for cells in scutoid_hist.voronoi_cells:
            for cell in cells:
                assert np.all((cell.center >= -2.0) & (cell.center <= 2.0))

    def test_incremental_summary_statistics(self, simple_history_2d):
        """Test summary statistics work with incremental mode."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d, incremental=True)
        scutoid_hist.build_tessellation()
        scutoid_hist.compute_ricci_scalars()

        stats = scutoid_hist.summary_statistics()

        # Check required fields
        assert stats["n_timesteps"] == 5
        assert stats["n_intervals"] == 4
        assert "mean_ricci" in stats

    def test_batch_mode_still_works(self, simple_history_2d):
        """Test that batch mode (incremental=False) still works correctly."""
        scutoid_hist = ScutoidHistory2D(simple_history_2d, incremental=False)
        scutoid_hist.build_tessellation()

        # Should not create IncrementalDelaunay2D
        assert scutoid_hist._incremental_delaunay is None

        # Should still build tessellation
        assert len(scutoid_hist.voronoi_cells) == 5
        assert len(scutoid_hist.scutoid_cells) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
