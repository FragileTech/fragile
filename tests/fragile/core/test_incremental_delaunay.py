"""Tests for incremental 2D Delaunay triangulation.

This module tests the IncrementalDelaunay2D class which provides O(N) amortized
tessellation updates for ScutoidHistory2D.

Test Coverage:
    - Initialization with various walker configurations
    - update_position() for local SDE moves
    - delete_and_insert() for cloning events
    - get_voronoi_cells() extraction and validation
    - Edge cases (degenerate configurations, boundary conditions)
"""

from __future__ import annotations

import numpy as np
import pytest

from fragile.core.incremental_delaunay import IncrementalDelaunay2D, VoronoiCell


class TestIncrementalDelaunay2DInit:
    """Test initialization of IncrementalDelaunay2D."""

    def test_init_valid_2d(self):
        """Test initialization with valid 2D positions."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)

        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        assert inc_del.positions.shape == (10, 2)
        assert inc_del.walker_ids.shape == (10,)
        assert len(inc_del.walker_to_idx) == 10
        assert inc_del.delaunay is not None

    def test_init_invalid_dimension(self):
        """Test that initialization fails for non-2D positions."""
        positions = np.random.rand(10, 3)  # 3D instead of 2D
        walker_ids = np.arange(10)

        with pytest.raises(ValueError, match="IncrementalDelaunay2D requires 2D positions"):
            IncrementalDelaunay2D(positions, walker_ids)

    def test_init_mismatched_sizes(self):
        """Test that initialization fails when position and ID counts don't match."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(5)  # Wrong size

        with pytest.raises(ValueError, match="Position count .* != walker ID count"):
            IncrementalDelaunay2D(positions, walker_ids)

    def test_init_too_few_points(self):
        """Test initialization with too few points for Delaunay."""
        positions = np.random.rand(2, 2)  # Only 2 points
        walker_ids = np.arange(2)

        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Delaunay should be None for < 3 points
        assert inc_del.delaunay is None

    def test_init_copies_data(self):
        """Test that initialization makes copies of input arrays."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)

        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Modify original arrays
        positions[0] = [999.0, 999.0]
        walker_ids[0] = 999

        # Internal data should be unchanged
        assert not np.allclose(inc_del.positions[0], [999.0, 999.0])
        assert inc_del.walker_ids[0] != 999


class TestIncrementalDelaunay2DUpdatePosition:
    """Test update_position() method for local SDE moves."""

    def test_update_position_valid(self):
        """Test updating a single walker position."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Update walker 5
        new_pos = np.array([0.5, 0.5])
        inc_del.update_position(walker_id=5, new_pos=new_pos)

        # Check position was updated
        idx = inc_del.walker_to_idx[5]
        assert np.allclose(inc_del.positions[idx], new_pos)

    def test_update_position_preserves_triangulation(self):
        """Test that update_position maintains a valid Delaunay triangulation."""
        positions = np.random.rand(20, 2)
        walker_ids = np.arange(20)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Update several walkers
        for walker_id in [3, 7, 12]:
            new_pos = np.random.rand(2)
            inc_del.update_position(walker_id, new_pos)

        # Delaunay should still be valid
        assert inc_del.delaunay is not None
        assert inc_del.delaunay.points.shape == (20, 2)

    def test_update_position_invalid_walker_id(self):
        """Test that updating non-existent walker raises error."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        with pytest.raises(ValueError, match="Walker ID .* not found"):
            inc_del.update_position(walker_id=999, new_pos=np.array([0.5, 0.5]))

    def test_update_position_multiple_sequential(self):
        """Test multiple sequential position updates."""
        positions = np.random.rand(15, 2)
        walker_ids = np.arange(15)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Update same walker multiple times
        for _ in range(5):
            new_pos = np.random.rand(2)
            inc_del.update_position(walker_id=7, new_pos=new_pos)
            idx = inc_del.walker_to_idx[7]
            assert np.allclose(inc_del.positions[idx], new_pos)


class TestIncrementalDelaunay2DDeleteAndInsert:
    """Test delete_and_insert() method for cloning events."""

    def test_delete_and_insert_valid(self):
        """Test delete-and-insert operation for cloning."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Clone walker 3 to a new position
        new_pos = np.array([0.8, 0.2])
        inc_del.delete_and_insert(walker_id=3, new_pos=new_pos)

        # Check position was updated
        idx = inc_del.walker_to_idx[3]
        assert np.allclose(inc_del.positions[idx], new_pos)

    def test_delete_and_insert_preserves_triangulation(self):
        """Test that delete_and_insert maintains valid Delaunay triangulation."""
        positions = np.random.rand(20, 2)
        walker_ids = np.arange(20)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Simulate cloning events
        for walker_id in [2, 9, 15]:
            new_pos = np.random.rand(2)
            inc_del.delete_and_insert(walker_id, new_pos)

        # Delaunay should still be valid
        assert inc_del.delaunay is not None
        assert inc_del.delaunay.points.shape == (20, 2)

    def test_delete_and_insert_invalid_walker_id(self):
        """Test that deleting non-existent walker raises error."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        with pytest.raises(ValueError, match="Walker ID .* not found"):
            inc_del.delete_and_insert(walker_id=999, new_pos=np.array([0.5, 0.5]))


class TestIncrementalDelaunay2DGetVoronoiCells:
    """Test get_voronoi_cells() extraction method."""

    def test_get_voronoi_cells_basic(self):
        """Test basic Voronoi cell extraction."""
        positions = np.random.rand(10, 2)
        walker_ids = np.arange(10)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        cells = inc_del.get_voronoi_cells()

        assert len(cells) == 10
        assert all(isinstance(cell, VoronoiCell) for cell in cells)

    def test_get_voronoi_cells_walker_ids_correct(self):
        """Test that extracted cells have correct walker IDs."""
        positions = np.random.rand(15, 2)
        walker_ids = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        cells = inc_del.get_voronoi_cells()

        extracted_ids = sorted([cell.walker_id for cell in cells])
        expected_ids = sorted(walker_ids.tolist())
        assert extracted_ids == expected_ids

    def test_get_voronoi_cells_centers_match_positions(self):
        """Test that Voronoi cell centers match walker positions."""
        positions = np.random.rand(12, 2)
        walker_ids = np.arange(12)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        cells = inc_del.get_voronoi_cells()

        # Build mapping walker_id -> cell
        cell_map = {cell.walker_id: cell for cell in cells}

        for i, walker_id in enumerate(walker_ids):
            cell = cell_map[walker_id]
            assert np.allclose(cell.center, positions[i])

    def test_get_voronoi_cells_neighbors_symmetric(self):
        """Test that neighbor relationships are symmetric."""
        positions = np.random.rand(20, 2)
        walker_ids = np.arange(20)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        cells = inc_del.get_voronoi_cells()
        cell_map = {cell.walker_id: cell for cell in cells}

        # Check symmetry: if j is neighbor of i, then i is neighbor of j
        for cell in cells:
            for neighbor_id in cell.neighbors:
                neighbor_cell = cell_map[neighbor_id]
                assert cell.walker_id in neighbor_cell.neighbors

    def test_get_voronoi_cells_too_few_points(self):
        """Test Voronoi extraction with too few points."""
        positions = np.random.rand(2, 2)
        walker_ids = np.arange(2)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        cells = inc_del.get_voronoi_cells()

        # Should return simple cells with no vertices/neighbors
        assert len(cells) == 2
        assert all(len(cell.vertices) == 0 for cell in cells)
        assert all(len(cell.neighbors) == 0 for cell in cells)

    def test_get_voronoi_cells_after_updates(self):
        """Test Voronoi extraction after position updates."""
        positions = np.random.rand(15, 2)
        walker_ids = np.arange(15)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Update some positions
        inc_del.update_position(walker_id=3, new_pos=np.array([0.1, 0.1]))
        inc_del.update_position(walker_id=7, new_pos=np.array([0.9, 0.9]))

        cells = inc_del.get_voronoi_cells()

        # Should have correct number of cells
        assert len(cells) == 15

        # Updated positions should be reflected in centers
        cell_map = {cell.walker_id: cell for cell in cells}
        assert np.allclose(cell_map[3].center, [0.1, 0.1])
        assert np.allclose(cell_map[7].center, [0.9, 0.9])


class TestIncrementalDelaunay2DEdgeCases:
    """Test edge cases and degenerate configurations."""

    def test_collinear_points(self):
        """Test handling of collinear walker positions."""
        # Create collinear points
        positions = np.array([[i * 0.1, 0.5] for i in range(10)])
        walker_ids = np.arange(10)

        # Collinear points will fail in scipy.spatial.Delaunay
        # This is expected behavior - scipy requires non-degenerate input
        # We just verify it raises the expected error
        with pytest.raises(Exception):  # QhullError or similar
            IncrementalDelaunay2D(positions, walker_ids)

    def test_duplicate_positions(self):
        """Test handling of duplicate walker positions."""
        positions = np.random.rand(5, 2)
        # Duplicate first position
        positions = np.vstack([positions, positions[0:1]])
        walker_ids = np.arange(6)

        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Should handle without crashing
        cells = inc_del.get_voronoi_cells()
        assert len(cells) == 6

    def test_very_close_positions(self):
        """Test handling of very close walker positions."""
        positions = np.random.rand(10, 2)
        # Make two walkers very close
        positions[5] = positions[3] + 1e-10
        walker_ids = np.arange(10)

        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Should handle without crashing
        cells = inc_del.get_voronoi_cells()
        assert len(cells) == 10

    def test_extreme_positions(self):
        """Test handling of extreme coordinate values."""
        positions = np.array([
            [0.0, 0.0],
            [1e6, 1e6],
            [1e-6, 1e-6],
            [0.5, 0.5],
            [0.3, 0.7],
        ])
        walker_ids = np.arange(5)

        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Should handle without crashing
        cells = inc_del.get_voronoi_cells()
        assert len(cells) == 5


class TestIncrementalDelaunay2DIntegration:
    """Integration tests simulating real usage patterns."""

    def test_simulation_workflow(self):
        """Test workflow simulating a typical EuclideanGas simulation."""
        # Initialize
        N = 30
        positions = np.random.rand(N, 2)
        walker_ids = np.arange(N)
        inc_del = IncrementalDelaunay2D(positions, walker_ids)

        # Simulate 10 timesteps
        for t in range(10):
            # Some walkers move locally (SDE)
            for walker_id in range(0, N, 3):
                new_pos = (
                    inc_del.positions[inc_del.walker_to_idx[walker_id]] + np.random.randn(2) * 0.01
                )
                inc_del.update_position(walker_id, new_pos)

            # Some walkers clone (teleport)
            for walker_id in range(1, N, 5):
                new_pos = np.random.rand(2)
                inc_del.delete_and_insert(walker_id, new_pos)

            # Extract Voronoi cells
            cells = inc_del.get_voronoi_cells()
            assert len(cells) == N

    def test_consistency_with_batch_mode(self):
        """Test that incremental mode produces same topology as batch mode."""
        from scipy.spatial import Voronoi

        # Create configuration
        N = 20
        positions = np.random.rand(N, 2)
        walker_ids = np.arange(N)

        # Incremental mode
        inc_del = IncrementalDelaunay2D(positions.copy(), walker_ids.copy())

        # Make some updates
        for walker_id in [3, 7, 12]:
            new_pos = np.random.rand(2)
            inc_del.update_position(walker_id, new_pos)

        incremental_cells = inc_del.get_voronoi_cells()

        # Batch mode (direct scipy)
        batch_vor = Voronoi(inc_del.positions)

        # Extract neighbor counts from batch mode
        batch_neighbor_counts = {}
        for i in range(N):
            neighbors = set()
            for ridge_points in batch_vor.ridge_points:
                if i in ridge_points:
                    other = ridge_points[0] if ridge_points[1] == i else ridge_points[1]
                    neighbors.add(walker_ids[other])
            batch_neighbor_counts[walker_ids[i]] = len(neighbors)

        # Compare neighbor counts
        incremental_neighbor_counts = {
            cell.walker_id: len(cell.neighbors) for cell in incremental_cells
        }

        # Should have same neighbor topology
        for walker_id in walker_ids:
            assert incremental_neighbor_counts[walker_id] == batch_neighbor_counts[walker_id]
