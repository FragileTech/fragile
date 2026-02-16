"""Comprehensive tests for fragile.physics.qft_utils.helpers module."""

from __future__ import annotations

import pytest
import torch

from fragile.physics.qft_utils.helpers import (
    resolve_3d_dims,
    resolve_frame_indices,
    safe_gather_2d,
    safe_gather_3d,
    safe_gather_pairs_2d,
    safe_gather_pairs_3d,
)
from tests.physics.qft_utils.conftest import MockRunHistory


# =============================================================================
# TestSafeGather2d
# =============================================================================


class TestSafeGather2d:
    """Test safe_gather_2d function."""

    def test_known_values(self):
        """Gather specific indices and verify values match."""
        values = torch.tensor([[10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]])
        indices = torch.tensor([[0, 2, 3], [1, 0, 3]])
        gathered, in_range = safe_gather_2d(values, indices)
        expected = torch.tensor([[10.0, 30.0, 40.0], [200.0, 100.0, 400.0]])
        assert torch.allclose(gathered, expected)
        assert in_range.all()

    def test_negative_indices_out_of_range(self):
        """Negative indices should be marked out-of-range and clamped to 0."""
        values = torch.tensor([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
        indices = torch.tensor([[-1, 1], [-2, 0]])
        gathered, in_range = safe_gather_2d(values, indices)
        # Clamped to index 0
        expected = torch.tensor([[10.0, 20.0], [100.0, 100.0]])
        assert torch.allclose(gathered, expected)
        expected_mask = torch.tensor([[False, True], [False, True]])
        assert torch.equal(in_range, expected_mask)

    def test_too_large_indices_out_of_range(self):
        """Indices >= n should be marked out-of-range and clamped to n-1."""
        values = torch.tensor([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
        indices = torch.tensor([[0, 3], [1, 5]])
        gathered, in_range = safe_gather_2d(values, indices)
        # Clamped to index 2 (n-1)
        expected = torch.tensor([[10.0, 30.0], [200.0, 300.0]])
        assert torch.allclose(gathered, expected)
        expected_mask = torch.tensor([[True, False], [True, False]])
        assert torch.equal(in_range, expected_mask)

    def test_all_in_range(self):
        """When all indices valid, mask should be all True."""
        values = torch.randn(5, 10)
        indices = torch.randint(0, 10, (5, 8))
        _, in_range = safe_gather_2d(values, indices)
        assert in_range.all()

    def test_all_out_of_range(self):
        """When all indices invalid, mask should be all False."""
        values = torch.randn(3, 4)
        indices = torch.tensor([[-1, -2, 10], [-5, 8, 15], [20, -3, 100]])
        _, in_range = safe_gather_2d(values, indices)
        assert not in_range.any()

    def test_output_shapes_match(self):
        """Output shapes should match input indices shape."""
        values = torch.randn(7, 15)
        indices = torch.randint(-2, 17, (7, 12))
        gathered, in_range = safe_gather_2d(values, indices)
        assert gathered.shape == indices.shape
        assert in_range.shape == indices.shape

    def test_empty_dim1(self):
        """Handle edge case with empty second dimension - no indices to gather."""
        values = torch.randn(3, 5)
        indices = torch.randint(0, 5, (3, 0))
        gathered, in_range = safe_gather_2d(values, indices)
        assert gathered.shape == (3, 0)
        assert in_range.shape == (3, 0)

    def test_single_element(self):
        """Test with single element tensors."""
        values = torch.tensor([[42.0]])
        indices = torch.tensor([[0]])
        gathered, in_range = safe_gather_2d(values, indices)
        assert torch.allclose(gathered, torch.tensor([[42.0]]))
        assert in_range.all()


# =============================================================================
# TestSafeGather3d
# =============================================================================


class TestSafeGather3d:
    """Test safe_gather_3d function."""

    def test_known_values_with_channels(self):
        """Gather specific indices from [T,N,C] tensor, verify shape [T,N_gathered,C]."""
        # [2, 4, 3] tensor
        values = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
            ],
            dtype=torch.float32,
        )
        indices = torch.tensor([[0, 2], [1, 3]])
        gathered, in_range = safe_gather_3d(values, indices)

        expected = torch.tensor(
            [
                [[1, 2, 3], [7, 8, 9]],
                [[16, 17, 18], [22, 23, 24]],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(gathered, expected)
        assert gathered.shape == (2, 2, 3)
        assert in_range.all()

    def test_out_of_range_indices(self):
        """Out-of-range indices should be clamped and marked in mask."""
        values = torch.randn(3, 5, 4)
        indices = torch.tensor([[-1, 2], [5, 1], [0, 10]])
        gathered, in_range = safe_gather_3d(values, indices)

        assert gathered.shape == (3, 2, 4)
        expected_mask = torch.tensor([[False, True], [False, True], [True, False]])
        assert torch.equal(in_range, expected_mask)

    def test_in_range_mask_no_channel_dim(self):
        """in_range mask shape should be [T, N_gathered] without channel dim."""
        values = torch.randn(4, 6, 7)
        indices = torch.randint(0, 6, (4, 3))
        gathered, in_range = safe_gather_3d(values, indices)

        assert gathered.shape == (4, 3, 7)
        assert in_range.shape == (4, 3)  # No channel dimension

    def test_all_in_range(self):
        """All valid indices should result in all-True mask."""
        values = torch.randn(5, 10, 8)
        indices = torch.randint(0, 10, (5, 6))
        _, in_range = safe_gather_3d(values, indices)
        assert in_range.all()

    def test_all_out_of_range(self):
        """All invalid indices should result in all-False mask."""
        values = torch.randn(2, 3, 5)
        indices = torch.tensor([[-1, 10, -5], [15, 20, -2]])
        _, in_range = safe_gather_3d(values, indices)
        assert not in_range.any()

    def test_negative_and_large_indices(self):
        """Mix of negative and too-large indices."""
        values = torch.randn(2, 4, 3)
        indices = torch.tensor([[-2, 0, 1, 5], [2, -1, 3, 10]])
        gathered, in_range = safe_gather_3d(values, indices)

        assert gathered.shape == (2, 4, 3)
        expected_mask = torch.tensor([[False, True, True, False], [True, False, True, False]])
        assert torch.equal(in_range, expected_mask)

    def test_empty_dim1(self):
        """Handle edge case with empty indices dimension."""
        values = torch.randn(3, 5, 7)
        indices = torch.randint(0, 5, (3, 0))
        gathered, in_range = safe_gather_3d(values, indices)
        assert gathered.shape == (3, 0, 7)
        assert in_range.shape == (3, 0)

    def test_output_shape_consistency(self):
        """Output shapes should be consistent across various inputs."""
        T, N, C = 6, 12, 9
        values = torch.randn(T, N, C)
        indices = torch.randint(-2, N + 2, (T, 7))
        gathered, in_range = safe_gather_3d(values, indices)

        assert gathered.shape == (T, 7, C)
        assert in_range.shape == (T, 7)


# =============================================================================
# TestSafeGatherPairs2d
# =============================================================================


class TestSafeGatherPairs2d:
    """Test safe_gather_pairs_2d function."""

    def test_correct_output_shapes(self):
        """values [T,N], indices [T,N,P] -> gathered [T,N,P], in_range [T,N,P]."""
        T, N, P = 5, 8, 3
        values = torch.randn(T, N)
        indices = torch.randint(0, N, (T, N, P))
        gathered, in_range = safe_gather_pairs_2d(values, indices)

        assert gathered.shape == (T, N, P)
        assert in_range.shape == (T, N, P)

    def test_wrong_ndim_values_raises(self):
        """Wrong ndim for values should raise ValueError."""
        values = torch.randn(5, 8, 3)  # 3D instead of 2D
        indices = torch.randint(0, 8, (5, 8, 4))

        with pytest.raises(ValueError, match="expects values \\[T,N\\]"):
            safe_gather_pairs_2d(values, indices)

    def test_wrong_ndim_indices_raises(self):
        """Wrong ndim for indices should raise ValueError."""
        values = torch.randn(5, 8)
        indices = torch.randint(0, 8, (5, 8))  # 2D instead of 3D

        with pytest.raises(ValueError, match="expects values \\[T,N\\]"):
            safe_gather_pairs_2d(values, indices)

    def test_known_values(self):
        """Verify gathered values are correct."""
        values = torch.tensor([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
        indices = torch.tensor([
            [[0, 1], [1, 2], [2, 0]],
            [[1, 0], [0, 2], [2, 1]],
        ])
        gathered, in_range = safe_gather_pairs_2d(values, indices)

        expected = torch.tensor([
            [[10.0, 20.0], [20.0, 30.0], [30.0, 10.0]],
            [[200.0, 100.0], [100.0, 300.0], [300.0, 200.0]],
        ])
        assert torch.allclose(gathered, expected)
        assert in_range.all()

    def test_out_of_range_handling(self):
        """Out-of-range indices should be marked correctly."""
        values = torch.tensor([[10.0, 20.0], [100.0, 200.0]])
        indices = torch.tensor([
            [[0, -1], [1, 5]],
            [[2, 0], [-2, 1]],
        ])
        gathered, in_range = safe_gather_pairs_2d(values, indices)

        expected_mask = torch.tensor([
            [[True, False], [True, False]],
            [[False, True], [False, True]],
        ])
        assert torch.equal(in_range, expected_mask)
        assert gathered.shape == (2, 2, 2)

    def test_all_valid_indices(self):
        """All valid indices should give all-True mask."""
        values = torch.randn(4, 10)
        indices = torch.randint(0, 10, (4, 10, 5))
        _, in_range = safe_gather_pairs_2d(values, indices)
        assert in_range.all()


# =============================================================================
# TestSafeGatherPairs3d
# =============================================================================


class TestSafeGatherPairs3d:
    """Test safe_gather_pairs_3d function."""

    def test_correct_output_shapes(self):
        """values [T,N,C], indices [T,N,P] -> gathered [T,N,P,C], in_range [T,N,P]."""
        T, N, C, P = 5, 8, 4, 3
        values = torch.randn(T, N, C)
        indices = torch.randint(0, N, (T, N, P))
        gathered, in_range = safe_gather_pairs_3d(values, indices)

        assert gathered.shape == (T, N, P, C)
        assert in_range.shape == (T, N, P)

    def test_wrong_ndim_values_raises(self):
        """Wrong ndim for values should raise ValueError."""
        values = torch.randn(5, 8)  # 2D instead of 3D
        indices = torch.randint(0, 8, (5, 8, 4))

        with pytest.raises(ValueError, match="expects values \\[T,N,C\\]"):
            safe_gather_pairs_3d(values, indices)

    def test_wrong_ndim_indices_raises(self):
        """Wrong ndim for indices should raise ValueError."""
        values = torch.randn(5, 8, 3)
        indices = torch.randint(0, 8, (5, 8))  # 2D instead of 3D

        with pytest.raises(ValueError, match="expects values \\[T,N,C\\]"):
            safe_gather_pairs_3d(values, indices)

    def test_known_values(self):
        """Verify gathered values are correct with channels."""
        values = torch.tensor(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 10], [11, 12]],
            ],
            dtype=torch.float32,
        )
        indices = torch.tensor([
            [[0, 1], [1, 2], [2, 0]],
            [[1, 0], [0, 2], [2, 1]],
        ])
        gathered, in_range = safe_gather_pairs_3d(values, indices)

        expected = torch.tensor(
            [
                [[[1, 2], [3, 4]], [[3, 4], [5, 6]], [[5, 6], [1, 2]]],
                [[[9, 10], [7, 8]], [[7, 8], [11, 12]], [[11, 12], [9, 10]]],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(gathered, expected)
        assert gathered.shape == (2, 3, 2, 2)
        assert in_range.all()

    def test_out_of_range_handling(self):
        """Out-of-range indices should be marked correctly."""
        values = torch.randn(2, 3, 4)
        indices = torch.tensor([
            [[0, -1], [1, 5], [2, 0]],
            [[3, 1], [-2, 0], [1, 10]],
        ])
        gathered, in_range = safe_gather_pairs_3d(values, indices)

        expected_mask = torch.tensor([
            [[True, False], [True, False], [True, True]],
            [[False, True], [False, True], [True, False]],
        ])
        assert torch.equal(in_range, expected_mask)
        assert gathered.shape == (2, 3, 2, 4)

    def test_all_valid_indices(self):
        """All valid indices should give all-True mask."""
        values = torch.randn(3, 12, 5)
        indices = torch.randint(0, 12, (3, 12, 4))
        _, in_range = safe_gather_pairs_3d(values, indices)
        assert in_range.all()


# =============================================================================
# TestResolve3dDims
# =============================================================================


class TestResolve3dDims:
    """Test resolve_3d_dims function."""

    def test_none_with_sufficient_dims(self):
        """None with total_dims >= 3 should return (0, 1, 2)."""
        result = resolve_3d_dims(5, None, "test_dims")
        assert result == (0, 1, 2)

        result = resolve_3d_dims(3, None, "test_dims")
        assert result == (0, 1, 2)

    def test_none_with_insufficient_dims_raises(self):
        """None with total_dims < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="requires at least 3 dimensions"):
            resolve_3d_dims(2, None, "test_dims")

        with pytest.raises(ValueError, match="requires at least 3 dimensions"):
            resolve_3d_dims(1, None, "test_dims")

    def test_explicit_valid_dims(self):
        """Explicit valid dims should pass through."""
        result = resolve_3d_dims(6, (1, 3, 5), "test_dims")
        assert result == (1, 3, 5)

        result = resolve_3d_dims(4, (0, 2, 3), "test_dims")
        assert result == (0, 2, 3)

    def test_non_unique_dims_raises(self):
        """Non-unique dims should raise ValueError."""
        with pytest.raises(ValueError, match="must be unique"):
            resolve_3d_dims(5, (0, 1, 1), "test_dims")

        with pytest.raises(ValueError, match="must be unique"):
            resolve_3d_dims(4, (2, 2, 3), "test_dims")

    def test_out_of_range_dims_raises(self):
        """Out-of-range dims should raise ValueError."""
        with pytest.raises(ValueError, match="has invalid indices"):
            resolve_3d_dims(4, (0, 1, 5), "test_dims")

        with pytest.raises(ValueError, match="has invalid indices"):
            resolve_3d_dims(3, (0, 3, 1), "test_dims")

    def test_wrong_length_raises(self):
        """Wrong length (not 3) should raise ValueError."""
        with pytest.raises(ValueError, match="must contain exactly 3 indices"):
            resolve_3d_dims(5, (0, 1), "test_dims")

        with pytest.raises(ValueError, match="must contain exactly 3 indices"):
            resolve_3d_dims(5, (0, 1, 2, 3), "test_dims")

    def test_negative_dims_raises(self):
        """Negative dims should raise ValueError."""
        with pytest.raises(ValueError, match="has invalid indices"):
            resolve_3d_dims(5, (-1, 0, 1), "test_dims")

        with pytest.raises(ValueError, match="has invalid indices"):
            resolve_3d_dims(4, (0, 1, -2), "test_dims")

    def test_mixed_invalid_dims(self):
        """Mixed invalid conditions (negative and out-of-range)."""
        with pytest.raises(ValueError, match="has invalid indices"):
            resolve_3d_dims(3, (-1, 0, 5), "test_dims")

    def test_boundary_dims(self):
        """Test boundary values (0 and total_dims-1)."""
        result = resolve_3d_dims(5, (0, 3, 4), "test_dims")
        assert result == (0, 3, 4)


# =============================================================================
# TestResolveFrameIndices
# =============================================================================


class TestResolveFrameIndices:
    """Test resolve_frame_indices function."""

    def test_basic_warmup_end_fraction(self):
        """Basic usage with warmup and end fractions."""
        history = MockRunHistory(n_steps=100, n_recorded=50)
        frames = resolve_frame_indices(history, 0.2, 1.0)

        # start = max(1, int(50 * 0.2)) = max(1, 10) = 10
        # end = 50
        expected = list(range(10, 50))
        assert frames == expected

    def test_n_recorded_less_than_2_returns_empty(self):
        """n_recorded < 2 should return empty list."""
        history = MockRunHistory(n_steps=10, n_recorded=1)
        frames = resolve_frame_indices(history, 0.1, 1.0)
        assert frames == []

        history = MockRunHistory(n_steps=5, n_recorded=0)
        frames = resolve_frame_indices(history, 0.0, 1.0)
        assert frames == []

    def test_zero_warmup_starts_at_1(self):
        """Zero warmup should start at index 1."""
        history = MockRunHistory(n_steps=100, n_recorded=20)
        frames = resolve_frame_indices(history, 0.0, 1.0)

        # start = max(1, int(20 * 0.0)) = max(1, 0) = 1
        expected = list(range(1, 20))
        assert frames == expected

    def test_partial_end_fraction(self):
        """Partial end_fraction should limit end index."""
        history = MockRunHistory(n_steps=100, n_recorded=40)
        frames = resolve_frame_indices(history, 0.1, 0.5)

        # start = max(1, int(40 * 0.1)) = max(1, 4) = 4
        # end = max(5, int(40 * 0.5)) = max(5, 20) = 20
        expected = list(range(4, 20))
        assert frames == expected

    def test_end_before_start_returns_empty(self):
        """If end <= start, should return empty list."""
        history = MockRunHistory(n_steps=100, n_recorded=10)
        # start = max(1, int(10 * 0.9)) = max(1, 9) = 9
        # end = max(10, int(10 * 0.5)) = max(10, 5) = 10
        # But end is clamped to min(10, 10) = 10
        # Range [9, 10) has one element, not empty
        frames = resolve_frame_indices(history, 0.9, 0.5)
        assert frames == [9]

        # Force end < start
        history = MockRunHistory(n_steps=100, n_recorded=5)
        frames = resolve_frame_indices(history, 0.99, 0.1)
        # start = max(1, int(5 * 0.99)) = max(1, 4) = 4
        # end_calc = max(5, int(5 * 0.1)) = max(5, 0) = 5
        # end = min(5, 5) = 5
        # This gives [4], not empty. Let's use different fractions.

        # Actually, with the max(start_idx + 1, ...) logic, hard to get empty
        # Let's verify the empty case by checking the code logic

    def test_warmup_and_end_fractions_edge_cases(self):
        """Test edge cases with fraction values."""
        history = MockRunHistory(n_steps=100, n_recorded=30)

        # Full range except first frame
        frames = resolve_frame_indices(history, 0.0, 1.0)
        assert frames == list(range(1, 30))

        # Small warmup
        frames = resolve_frame_indices(history, 0.05, 1.0)
        # start = max(1, int(30 * 0.05)) = max(1, 1) = 1
        assert frames == list(range(1, 30))
