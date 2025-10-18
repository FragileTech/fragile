"""Tests for VectorizedOps and companion selection functions.

Note: Algorithmic distance and companion selection functions have been moved
to fragile.companion_selection module. These tests now import from there.
"""

import pytest
import torch

from fragile.companion_selection import (
    compute_algorithmic_distance_matrix,
    select_companions_softmax,
)
from fragile.euclidean_gas import SwarmState, VectorizedOps


class TestAlgorithmicDistance:
    """Tests for algorithmic distance calculation."""

    def test_self_distance_is_zero(self):
        """Test that d_alg(i, i) = 0 for all i."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)

        lambda_alg = 1.0
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

        # Diagonal should be zero (within numerical precision)
        assert torch.allclose(torch.diag(dist_sq), torch.zeros(10), atol=1e-5)

    def test_symmetry(self):
        """Test that d_alg(i, j) = d_alg(j, i)."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        lambda_alg = 1.0
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

        # Should be symmetric
        assert torch.allclose(dist_sq, dist_sq.T)

    def test_identical_positions_different_velocities(self):
        """Test distance when positions are identical but velocities differ."""
        x = torch.zeros(5, 2)  # All at origin
        v = torch.randn(5, 2)
        lambda_alg = 2.0
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

        # Distance should be lambda_alg * ||v_i - v_j||^2
        for i in range(5):
            for j in range(5):
                if i != j:
                    expected = lambda_alg * torch.sum((v[i] - v[j]) ** 2)
                    assert torch.allclose(dist_sq[i, j], expected)

    def test_identical_velocities_different_positions(self):
        """Test distance when velocities are identical but positions differ."""
        x = torch.randn(5, 2)
        v = torch.zeros(5, 2)  # All stationary
        lambda_alg = 1.0
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

        # Distance should be ||x_i - x_j||^2
        for i in range(5):
            for j in range(5):
                if i != j:
                    expected = torch.sum((x[i] - x[j]) ** 2)
                    assert torch.allclose(dist_sq[i, j], expected, atol=1e-6)

    @pytest.mark.parametrize("lambda_alg", [0.1, 1.0, 10.0])
    def test_lambda_alg_scaling(self, lambda_alg):
        """Test effect of lambda_alg parameter."""
        x = torch.randn(5, 2)
        v = torch.randn(5, 2)
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

        # Verify formula: ||x_i - x_j||^2 + lambda_alg * ||v_i - v_j||^2
        for i in range(5):
            for j in range(5):
                dx = x[i] - x[j]
                dv = v[i] - v[j]
                expected = torch.sum(dx**2) + lambda_alg * torch.sum(dv**2)
                assert torch.allclose(dist_sq[i, j], expected)

    def test_non_negativity(self):
        """Test that all distances are non-negative (within numerical precision)."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        lambda_alg = 1.0
        dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

        # Allow small numerical errors
        assert torch.all(dist_sq >= -1e-6)

    def test_large_lambda_emphasizes_velocity(self):
        """Test that large lambda_alg emphasizes velocity differences."""
        x = torch.randn(5, 2)
        v = torch.randn(5, 2)
        SwarmState(x, v)

        # Small lambda: position-dominated
        dist_small = compute_algorithmic_distance_matrix(x, v, 0.01)

        # Large lambda: velocity-dominated
        dist_large = compute_algorithmic_distance_matrix(x, v, 100.0)

        # They should give different orderings (in general)
        assert not torch.allclose(dist_small, dist_large)


class TestFindCompanions:
    """Tests for companion finding."""

    def test_excludes_self(self):
        """Test that walkers don't select themselves as companions."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        alive_mask = torch.ones(10, dtype=torch.bool)

        epsilon = 0.5
        companions = select_companions_softmax(
            x, v, alive_mask, epsilon=epsilon, lambda_alg=1.0, exclude_self=True
        )

        # No walker should be its own companion
        for i in range(10):
            assert companions[i] != i

    def test_favors_nearby_companions(self):
        """Test that companion selection favors nearby walkers (stochastically)."""
        # Create simple configuration where nearest neighbors are clear
        # Walkers 0 and 1 are very close, walkers 2 and 3 are far away
        x = torch.tensor([
            [0.0, 0.0],
            [0.01, 0.0],  # Very close to 0
            [10.0, 0.0],  # Far from 0 and 1
            [10.01, 0.0],  # Very close to 2
        ])
        v = torch.zeros(4, 2)
        alive_mask = torch.ones(4, dtype=torch.bool)

        epsilon = 0.5  # Small epsilon means strong preference for nearby

        # Run multiple times to test stochastic behavior
        torch.manual_seed(42)
        counts_0_to_1 = 0
        n_trials = 100

        for _ in range(n_trials):
            companions = select_companions_softmax(
                x, v, alive_mask, epsilon=epsilon, lambda_alg=0.0, exclude_self=True
            )
            if companions[0] == 1:
                counts_0_to_1 += 1

        # With small epsilon and very close neighbors, should almost always choose nearby
        # Allow some randomness but expect > 90% selection of nearest
        assert (
            counts_0_to_1 > 0.9 * n_trials
        ), f"Expected walker 0 to choose walker 1 in >90% of trials, got {counts_0_to_1}/{n_trials}"

    def test_two_walkers(self):
        """Test edge case with only two walkers."""
        x = torch.randn(2, 3)
        v = torch.randn(2, 3)
        alive_mask = torch.ones(x.shape[0], dtype=torch.bool)
        epsilon = 0.5
        companions = select_companions_softmax(
            x, v, alive_mask, epsilon=epsilon, lambda_alg=1.0, exclude_self=True
        )

        # Each must choose the other (only option)
        assert companions[0] == 1
        assert companions[1] == 0

    def test_stochastic_with_seed(self):
        """Test that companion finding is reproducible with same random seed."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        alive_mask = torch.ones(10, dtype=torch.bool)
        epsilon = 0.5

        torch.manual_seed(42)
        companions1 = select_companions_softmax(
            x, v, alive_mask, epsilon=epsilon, lambda_alg=1.0, exclude_self=True
        )

        torch.manual_seed(42)
        companions2 = select_companions_softmax(
            x, v, alive_mask, epsilon=epsilon, lambda_alg=1.0, exclude_self=True
        )

        assert torch.equal(companions1, companions2)

    def test_output_shape(self):
        """Test that output has correct shape."""
        x = torch.randn(20, 5)
        v = torch.randn(20, 5)
        alive_mask = torch.ones(x.shape[0], dtype=torch.bool)
        epsilon = 0.5
        companions = select_companions_softmax(
            x, v, alive_mask, epsilon=epsilon, lambda_alg=1.0, exclude_self=True
        )

        assert companions.shape == (20,)
        assert companions.dtype == torch.long


class TestVariancePosition:
    """Tests for position variance calculation."""

    def test_zero_variance_single_point(self):
        """Test that variance is zero when all walkers at same position."""
        x = torch.ones(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        var_x = VectorizedOps.variance_position(state)
        assert torch.allclose(var_x, torch.tensor(0.0), atol=1e-7)

    def test_variance_from_origin(self):
        """Test variance calculation for points distributed around origin."""
        # Points at distance r from origin
        r = 2.0
        x = torch.tensor([
            [r, 0.0],
            [-r, 0.0],
            [0.0, r],
            [0.0, -r],
        ])
        v = torch.zeros(4, 2)
        state = SwarmState(x, v)

        # Mean is at origin, so variance = (1/4) * 4 * r^2 = r^2
        var_x = VectorizedOps.variance_position(state)
        expected = r**2
        assert torch.allclose(var_x, torch.tensor(expected))

    def test_variance_formula(self):
        """Test that variance matches the formula V = (1/N) sum ||x_i - mu||^2."""
        x = torch.randn(20, 5)
        v = torch.randn(20, 5)
        state = SwarmState(x, v)

        var_x = VectorizedOps.variance_position(state)

        # Manual calculation
        mu = torch.mean(x, dim=0)
        expected = torch.mean(torch.sum((x - mu) ** 2, dim=-1))

        assert torch.allclose(var_x, expected)

    def test_non_negativity(self):
        """Test that variance is always non-negative."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        var_x = VectorizedOps.variance_position(state)
        assert var_x >= 0

    def test_scalar_output(self):
        """Test that variance returns a scalar."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        var_x = VectorizedOps.variance_position(state)
        assert var_x.ndim == 0  # Scalar


class TestVarianceVelocity:
    """Tests for velocity variance calculation."""

    def test_zero_variance_single_velocity(self):
        """Test that variance is zero when all walkers have same velocity."""
        x = torch.randn(10, 3)
        v = torch.ones(10, 3)
        state = SwarmState(x, v)

        var_v = VectorizedOps.variance_velocity(state)
        assert torch.allclose(var_v, torch.tensor(0.0), atol=1e-7)

    def test_stationary_swarm(self):
        """Test variance for stationary swarm."""
        x = torch.randn(10, 3)
        v = torch.zeros(10, 3)
        state = SwarmState(x, v)

        var_v = VectorizedOps.variance_velocity(state)
        assert torch.allclose(var_v, torch.tensor(0.0))

    def test_variance_formula(self):
        """Test that variance matches the formula V = (1/N) sum ||v_i - mu||^2."""
        x = torch.randn(20, 5)
        v = torch.randn(20, 5)
        state = SwarmState(x, v)

        var_v = VectorizedOps.variance_velocity(state)

        # Manual calculation
        mu = torch.mean(v, dim=0)
        expected = torch.mean(torch.sum((v - mu) ** 2, dim=-1))

        assert torch.allclose(var_v, expected)

    def test_non_negativity(self):
        """Test that variance is always non-negative."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        var_v = VectorizedOps.variance_velocity(state)
        assert var_v >= 0

    def test_scalar_output(self):
        """Test that variance returns a scalar."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 3)
        state = SwarmState(x, v)

        var_v = VectorizedOps.variance_velocity(state)
        assert var_v.ndim == 0  # Scalar

    def test_independent_of_position_variance(self):
        """Test that velocity variance is independent of position distribution."""
        x1 = torch.randn(10, 3) * 10  # Large position variance
        x2 = torch.randn(10, 3) * 0.1  # Small position variance
        v = torch.randn(10, 3)

        state1 = SwarmState(x1, v)
        state2 = SwarmState(x2, v)

        var_v1 = VectorizedOps.variance_velocity(state1)
        var_v2 = VectorizedOps.variance_velocity(state2)

        assert torch.allclose(var_v1, var_v2)
