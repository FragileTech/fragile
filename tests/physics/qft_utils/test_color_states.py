"""Tests for color state computation from RunHistory data."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.qft_utils.color_states import (
    compute_color_states_batch,
    estimate_ell0,
)

from .conftest import MockRunHistory


class TestComputeColorStatesBatch:
    """Tests for compute_color_states_batch function."""

    def test_output_shapes(self, mock_history: MockRunHistory):
        """Test that output shapes are correct: color [T, N, d], valid [T, N]."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, valid = compute_color_states_batch(mock_history, start_idx, h_eff, mass, ell0)

        T = mock_history.n_recorded - start_idx
        N = mock_history.N
        d = mock_history.d

        assert color.shape == (T, N, d)
        assert valid.shape == (T, N)

    def test_color_is_complex(self, mock_history: MockRunHistory):
        """Test that color tensor is complex valued."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, _ = compute_color_states_batch(mock_history, start_idx, h_eff, mass, ell0)

        assert color.is_complex()

    def test_valid_is_bool(self, mock_history: MockRunHistory):
        """Test that valid tensor has bool dtype."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        _, valid = compute_color_states_batch(mock_history, start_idx, h_eff, mass, ell0)

        assert valid.dtype == torch.bool

    def test_color_vectors_unit_normalized(self, mock_history: MockRunHistory):
        """Test that color vectors are unit normalized for valid walkers."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, valid = compute_color_states_batch(mock_history, start_idx, h_eff, mass, ell0)

        # Compute norms for valid walkers
        norms = torch.linalg.vector_norm(color, dim=-1)
        valid_norms = norms[valid]

        if valid_norms.numel() > 0:
            assert torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-6)

    def test_with_end_idx_truncates_correctly(self, mock_history: MockRunHistory):
        """Test that end_idx truncates output correctly."""
        start_idx = 1
        end_idx = 10
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, valid = compute_color_states_batch(
            mock_history, start_idx, h_eff, mass, ell0, end_idx=end_idx
        )

        T = end_idx - start_idx
        assert color.shape[0] == T
        assert valid.shape[0] == T

    def test_without_end_idx_uses_all_frames(self, mock_history: MockRunHistory):
        """Test that end_idx=None uses all frames from start_idx to n_recorded."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, valid = compute_color_states_batch(
            mock_history, start_idx, h_eff, mass, ell0, end_idx=None
        )

        T = mock_history.n_recorded - start_idx
        assert color.shape[0] == T
        assert valid.shape[0] == T

    def test_start_idx_one_minimum_meaningful(self, mock_history: MockRunHistory):
        """Test that start_idx=1 produces n_recorded - 1 frames."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, valid = compute_color_states_batch(mock_history, start_idx, h_eff, mass, ell0)

        T = mock_history.n_recorded - 1
        assert color.shape[0] == T
        assert valid.shape[0] == T

    def test_h_eff_scaling_reduces_phase(self, mock_history: MockRunHistory):
        """Test that larger h_eff reduces phase magnitude."""
        start_idx = 1
        mass = 1.0
        ell0 = 1.0

        # Compute with small h_eff
        color_small, _ = compute_color_states_batch(
            mock_history, start_idx, h_eff=0.1, mass=mass, ell0=ell0
        )

        # Compute with large h_eff
        color_large, _ = compute_color_states_batch(
            mock_history, start_idx, h_eff=10.0, mass=mass, ell0=ell0
        )

        # Phase = (mass * v * ell0) / h_eff
        # Larger h_eff means smaller phase magnitude
        # The colors should be different due to different phases
        assert not torch.allclose(color_small, color_large)

    def test_mass_scaling_increases_phase(self, mock_history: MockRunHistory):
        """Test that larger mass increases phase magnitude."""
        start_idx = 1
        h_eff = 1.0
        ell0 = 1.0

        # Compute with small mass
        color_small, _ = compute_color_states_batch(
            mock_history, start_idx, h_eff=h_eff, mass=0.1, ell0=ell0
        )

        # Compute with large mass
        color_large, _ = compute_color_states_batch(
            mock_history, start_idx, h_eff=h_eff, mass=10.0, ell0=ell0
        )

        # Phase = (mass * v * ell0) / h_eff
        # Larger mass means larger phase magnitude
        # The colors should be different due to different phases
        assert not torch.allclose(color_small, color_large)

    def test_ell0_scaling_increases_phase(self, mock_history: MockRunHistory):
        """Test that larger ell0 increases phase magnitude."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0

        # Compute with small ell0
        color_small, _ = compute_color_states_batch(
            mock_history, start_idx, h_eff=h_eff, mass=mass, ell0=0.1
        )

        # Compute with large ell0
        color_large, _ = compute_color_states_batch(
            mock_history, start_idx, h_eff=h_eff, mass=mass, ell0=10.0
        )

        # Phase = (mass * v * ell0) / h_eff
        # Larger ell0 means larger phase magnitude
        # The colors should be different due to different phases
        assert not torch.allclose(color_small, color_large)

    def test_zero_force_viscous(self):
        """Test that zero force_viscous leads to valid=False for all walkers."""
        # Create a mock history with zero force_viscous
        history = MockRunHistory(N=10, d=3, n_recorded=5)
        history.force_viscous = torch.zeros_like(history.force_viscous)

        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        _color, valid = compute_color_states_batch(history, start_idx, h_eff, mass, ell0)

        # All valid should be False because norm is zero (or clamped to 1e-12)
        # After clamping to 1e-12, norm > 1e-12 is False
        assert not valid.any()

    def test_all_finite_color_values_for_valid_walkers(self, mock_history: MockRunHistory):
        """Test that all color values are finite for valid walkers."""
        start_idx = 1
        h_eff = 1.0
        mass = 1.0
        ell0 = 1.0

        color, valid = compute_color_states_batch(mock_history, start_idx, h_eff, mass, ell0)

        # Check that all color values for valid walkers are finite
        valid_colors = color[valid]
        if valid_colors.numel() > 0:
            assert torch.isfinite(valid_colors).all()


class TestEstimateEll0:
    """Tests for estimate_ell0 function."""

    def test_returns_positive_float(self, mock_history: MockRunHistory):
        """Test that estimate_ell0 returns a positive float."""
        result = estimate_ell0(mock_history)

        assert isinstance(result, float)
        assert result > 0

    def test_n_recorded_zero_returns_one(self):
        """Test that n_recorded=0 returns 1.0."""
        history = MockRunHistory(n_recorded=0)
        result = estimate_ell0(history)

        assert result == 1.0

    def test_n_recorded_one_returns_one(self):
        """Test that n_recorded=1 (mid_idx=0) returns 1.0."""
        history = MockRunHistory(n_recorded=1)
        result = estimate_ell0(history)

        assert result == 1.0

    def test_default_mock_history_returns_positive(self):
        """Test that default MockRunHistory returns a positive value."""
        history = MockRunHistory()
        result = estimate_ell0(history)

        assert result > 0

    def test_deterministic_same_seed_same_result(self):
        """Test that same seed gives same result."""
        history1 = MockRunHistory(seed=123)
        history2 = MockRunHistory(seed=123)

        result1 = estimate_ell0(history1)
        result2 = estimate_ell0(history2)

        assert result1 == result2

    def test_single_walker_returns_zero(self):
        """Test that N=1 (companion is self) returns 0.0."""
        history = MockRunHistory(N=1, n_recorded=10)
        result = estimate_ell0(history)

        # Companion is self, so distance is 0
        assert result == 0.0

    def test_positioned_walkers_known_distances(self):
        """Test with positioned walkers with known distances."""
        history = MockRunHistory(N=4, d=2, n_recorded=10)

        # Position walkers in a square at mid_idx
        mid_idx = history.n_recorded // 2
        history.x_before_clone[mid_idx] = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])

        # Companions_distance is cyclic shift by 1:
        # walker 0 -> walker 1, walker 1 -> walker 2, etc.
        # Distances: 0->1: 1.0, 1->2: 1.0, 2->3: 1.0, 3->0: 1.0

        result = estimate_ell0(history)

        # All distances are 1.0, so median should be 1.0
        assert abs(result - 1.0) < 1e-6

    def test_euclidean_always_used(self):
        """Test that Euclidean distances are always used, even with neighbor graph."""
        history_with = MockRunHistory(N=4, d=2, n_recorded=10, with_neighbor_graph=True)
        history_without = MockRunHistory(N=4, d=2, n_recorded=10, with_neighbor_graph=False)

        # Set identical positions so Euclidean distances match
        mid_idx = history_with.n_recorded // 2
        positions = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        history_with.x_before_clone[mid_idx] = positions
        history_without.x_before_clone[mid_idx] = positions

        result_with = estimate_ell0(history_with)
        result_without = estimate_ell0(history_without)

        assert isinstance(result_with, float)
        assert result_with > 0
        # Both should give identical Euclidean results
        assert abs(result_with - result_without) < 1e-6
