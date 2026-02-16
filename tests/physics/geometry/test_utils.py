"""Tests for fragile.physics.geometry.utils module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.utils import (
    estimate_optimal_step_size,
    find_axial_neighbors,
    validate_finite_difference_inputs,
)


# ---------------------------------------------------------------------------
# TestEstimateOptimalStepSize
# ---------------------------------------------------------------------------


class TestEstimateOptimalStepSize:
    """Tests for estimate_optimal_step_size."""

    def test_output_shape_and_positive(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Output shape is [N] and all values are positive."""
        result = estimate_optimal_step_size(grid_2d_positions, delaunay_2d_edges)
        assert result.shape == (grid_2d_positions.shape[0],)
        assert (result > 0).all()

    def test_fraction_scaling(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """fraction=0.5 gives half the result of fraction=1.0."""
        half = estimate_optimal_step_size(
            grid_2d_positions, delaunay_2d_edges, target_fraction=0.5
        )
        full = estimate_optimal_step_size(
            grid_2d_positions, delaunay_2d_edges, target_fraction=1.0
        )
        torch.testing.assert_close(half, full * 0.5)

    def test_regular_grid_step_size(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """On a regular grid, step ~ target_fraction * grid_spacing (within 50%)."""
        # 10x10 grid on [-2, 2] -> spacing = 4/9
        grid_spacing = 4.0 / 9.0
        target_fraction = 0.5
        result = estimate_optimal_step_size(
            grid_2d_positions, delaunay_2d_edges, target_fraction=target_fraction
        )
        expected = target_fraction * grid_spacing
        # For interior walkers the nearest neighbor is at grid_spacing,
        # so step should be close to expected. Allow 50% tolerance.
        median_step = result.median().item()
        assert abs(median_step - expected) < 0.5 * expected

    def test_alive_mask_dead_walkers_get_nan(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Dead walkers (alive=False) get NaN step sizes."""
        N = grid_2d_positions.shape[0]
        alive = torch.ones(N, dtype=torch.bool)
        alive[0] = False
        alive[5] = False
        result = estimate_optimal_step_size(grid_2d_positions, delaunay_2d_edges, alive=alive)
        assert torch.isnan(result[0])
        assert torch.isnan(result[5])
        # Alive walkers should have finite positive values
        assert (result[alive] > 0).all()
        assert torch.isfinite(result[alive]).all()

    def test_isolated_walker_fallback(self):
        """If a node has no edges, it gets the median edge distance as fallback."""
        # 3 points: 0 at origin, 1 at (1,0), 2 at (3,0)
        # Only edge between 0 and 1 (symmetric)
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        result = estimate_optimal_step_size(positions, edge_index, target_fraction=1.0)
        # Walkers 0 and 1: min neighbor dist = 1.0
        assert abs(result[0].item() - 1.0) < 1e-5
        assert abs(result[1].item() - 1.0) < 1e-5
        # Walker 2 is isolated: gets median edge distance = median([1.0, 1.0]) = 1.0
        assert abs(result[2].item() - 1.0) < 1e-5

    def test_default_fraction(self, random_2d_positions: Tensor, random_2d_edges: Tensor):
        """Default target_fraction=0.5 is applied correctly."""
        result_default = estimate_optimal_step_size(random_2d_positions, random_2d_edges)
        result_explicit = estimate_optimal_step_size(
            random_2d_positions, random_2d_edges, target_fraction=0.5
        )
        torch.testing.assert_close(result_default, result_explicit)


# ---------------------------------------------------------------------------
# TestFindAxialNeighbors
# ---------------------------------------------------------------------------


class TestFindAxialNeighbors:
    """Tests for find_axial_neighbors."""

    def _interior_walker_idx(self, grid_2d_positions: Tensor) -> int:
        """Return the index of a walker near the center of the 10x10 grid.

        Grid layout (ij indexing): index = i*10 + j, so (5, 5) -> 55.
        """
        return 55

    def test_axis0_finds_x_neighbors(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """For an interior walker on a regular grid, axis=0 finds x-neighbors.

        The function computes the angle between the displacement vector and the
        positive axis direction. Only neighbors within max_angle_deg of the
        positive axis are classified. On a regular Delaunay grid, this reliably
        identifies the immediate positive-axis neighbor.
        """
        idx = self._interior_walker_idx(grid_2d_positions)
        pos_neighbors, _neg_neighbors = find_axial_neighbors(
            grid_2d_positions, delaunay_2d_edges, idx, axis=0, max_angle_deg=30.0
        )
        # Should find at least one neighbor in the positive x-direction
        assert len(pos_neighbors) >= 1
        # Verify returned neighbors have positive x-displacement
        center = grid_2d_positions[idx]
        for n_idx in pos_neighbors:
            assert (grid_2d_positions[n_idx, 0] - center[0]).item() > 0

    def test_axis1_finds_y_neighbors(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """For an interior walker on a regular grid, axis=1 finds y-neighbors."""
        idx = self._interior_walker_idx(grid_2d_positions)
        pos_neighbors, _neg_neighbors = find_axial_neighbors(
            grid_2d_positions, delaunay_2d_edges, idx, axis=1, max_angle_deg=30.0
        )
        # Should find at least one neighbor in the positive y-direction
        assert len(pos_neighbors) >= 1
        center = grid_2d_positions[idx]
        for n_idx in pos_neighbors:
            assert (grid_2d_positions[n_idx, 1] - center[1]).item() > 0

    def test_positive_negative_direction(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Positive neighbors have displacement > 0 along axis, negative < 0."""
        idx = self._interior_walker_idx(grid_2d_positions)
        center = grid_2d_positions[idx]

        for axis in [0, 1]:
            pos_neighbors, neg_neighbors = find_axial_neighbors(
                grid_2d_positions, delaunay_2d_edges, idx, axis=axis
            )
            for n_idx in pos_neighbors:
                displacement = grid_2d_positions[n_idx] - center
                assert displacement[axis].item() > 0, (
                    f"Positive neighbor {n_idx} has negative displacement along axis {axis}"
                )
            for n_idx in neg_neighbors:
                displacement = grid_2d_positions[n_idx] - center
                assert displacement[axis].item() < 0, (
                    f"Negative neighbor {n_idx} has positive displacement along axis {axis}"
                )

    def test_tight_angle_threshold(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Tight angle threshold (5 deg) filters out non-axial neighbors more aggressively."""
        idx = self._interior_walker_idx(grid_2d_positions)

        pos_wide, neg_wide = find_axial_neighbors(
            grid_2d_positions, delaunay_2d_edges, idx, axis=0, max_angle_deg=30.0
        )
        pos_tight, neg_tight = find_axial_neighbors(
            grid_2d_positions, delaunay_2d_edges, idx, axis=0, max_angle_deg=5.0
        )
        total_wide = len(pos_wide) + len(neg_wide)
        total_tight = len(pos_tight) + len(neg_tight)
        assert total_tight <= total_wide

    def test_no_neighbors_returns_empty(self, grid_2d_positions: Tensor):
        """Walker with no edges returns empty lists."""
        # Empty edge index means no neighbors for anyone
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        pos_neighbors, neg_neighbors = find_axial_neighbors(
            grid_2d_positions, edge_index, walker_idx=0, axis=0
        )
        assert pos_neighbors == []
        assert neg_neighbors == []

    def test_edge_walker_one_direction(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Boundary walkers may have neighbors in only one direction along an axis."""
        # Walker 0 is at corner (-2, -2) on the 10x10 grid (index=0, i.e. row=0, col=0).
        # Along axis=0, it has only positive (increasing x) neighbors.
        pos_neighbors, neg_neighbors = find_axial_neighbors(
            grid_2d_positions, delaunay_2d_edges, walker_idx=0, axis=0, max_angle_deg=30.0
        )
        # Should have positive neighbors but not necessarily negative ones
        assert len(pos_neighbors) >= 1
        # Corner walker in the minimum-x, minimum-y position: no negative x-neighbors
        assert len(neg_neighbors) == 0


# ---------------------------------------------------------------------------
# TestValidateFiniteDifferenceInputs
# ---------------------------------------------------------------------------


class TestValidateFiniteDifferenceInputs:
    """Tests for validate_finite_difference_inputs."""

    def test_all_valid(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """All walkers valid: valid_walkers all True, n_valid == N."""
        N = grid_2d_positions.shape[0]
        fitness = torch.randn(N)
        result = validate_finite_difference_inputs(
            grid_2d_positions, fitness, delaunay_2d_edges, min_neighbors=1
        )
        assert result["n_valid"] == N
        assert result["n_total"] == N
        assert result["valid_walkers"].all()

    def test_nan_fitness_detected(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Walker with NaN fitness has has_nan_fitness=True and valid_walkers=False."""
        N = grid_2d_positions.shape[0]
        fitness = torch.randn(N)
        fitness[3] = float("nan")
        result = validate_finite_difference_inputs(
            grid_2d_positions, fitness, delaunay_2d_edges, min_neighbors=1
        )
        assert result["has_nan_fitness"][3].item() is True
        assert result["valid_walkers"][3].item() is False
        assert result["n_valid"] == N - 1

    def test_nan_position_detected(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Walker with NaN position has has_nan_position=True."""
        N = grid_2d_positions.shape[0]
        positions = grid_2d_positions.clone()
        positions[7, 0] = float("nan")
        fitness = torch.randn(N)
        result = validate_finite_difference_inputs(
            positions, fitness, delaunay_2d_edges, min_neighbors=1
        )
        assert result["has_nan_position"][7].item() is True
        assert result["valid_walkers"][7].item() is False

    def test_isolated_walkers(self):
        """Walker with fewer than min_neighbors neighbors has is_isolated=True."""
        # 5 walkers, only edges between 0-1, 0-2, 0-3, 0-4 (symmetric)
        positions = torch.randn(5, 2)
        fitness = torch.randn(5)
        edge_index = torch.tensor(
            [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        # Walker 0 has 4 neighbors, walkers 1-4 each have 1 neighbor
        result = validate_finite_difference_inputs(positions, fitness, edge_index, min_neighbors=3)
        # Walker 0 is NOT isolated (4 >= 3)
        assert result["is_isolated"][0].item() is False
        # Walkers 1-4 ARE isolated (1 < 3)
        assert result["is_isolated"][1].item() is True
        assert result["is_isolated"][2].item() is True
        assert result["is_isolated"][3].item() is True
        assert result["is_isolated"][4].item() is True

    def test_alive_mask_applied(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Dead walkers excluded from valid_walkers."""
        N = grid_2d_positions.shape[0]
        fitness = torch.randn(N)
        alive = torch.ones(N, dtype=torch.bool)
        alive[0] = False
        alive[10] = False
        result = validate_finite_difference_inputs(
            grid_2d_positions, fitness, delaunay_2d_edges, alive=alive, min_neighbors=1
        )
        assert result["valid_walkers"][0].item() is False
        assert result["valid_walkers"][10].item() is False
        assert result["n_valid"] == N - 2

    def test_output_keys(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """All expected keys present in the output dict."""
        N = grid_2d_positions.shape[0]
        fitness = torch.randn(N)
        result = validate_finite_difference_inputs(
            grid_2d_positions, fitness, delaunay_2d_edges, min_neighbors=1
        )
        expected_keys = {
            "valid_walkers",
            "num_neighbors",
            "has_nan_fitness",
            "has_nan_position",
            "is_isolated",
            "n_valid",
            "n_total",
        }
        assert set(result.keys()) == expected_keys

    def test_all_walkers_invalid_raises(self):
        """All walkers invalid raises ValueError."""
        # All walkers have NaN fitness
        positions = torch.randn(5, 2)
        fitness = torch.full((5,), float("nan"))
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]],
            dtype=torch.long,
        )
        with pytest.raises(ValueError, match="No valid walkers"):
            validate_finite_difference_inputs(positions, fitness, edge_index, min_neighbors=1)

    def test_no_edge_index_no_num_neighbors_raises(self):
        """No edge_index and no num_neighbors raises ValueError."""
        positions = torch.randn(5, 2)
        fitness = torch.randn(5)
        with pytest.raises(ValueError, match="edge_index is required"):
            validate_finite_difference_inputs(
                positions, fitness, edge_index=None, num_neighbors=None
            )
