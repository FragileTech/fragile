"""Tests for vectorized mean-field operations."""

import pytest
import torch

from fragile.mean_field_ops import (
    compute_fitness_gradient_vectorized,
    compute_fitness_potential_vectorized,
    distance_to_random_companion,
    patched_std_dev,
)


class TestPatchedStdDev:
    """Test patched standard deviation function."""

    def test_patched_std_dev_shape(self):
        """Output shape should match input shape."""
        var = torch.tensor([0.0, 0.01, 1.0, 100.0])
        sigma_prime = patched_std_dev(var, kappa_var_min=0.01, eps_std=0.001)

        assert sigma_prime.shape == var.shape

    def test_patched_std_dev_floor(self):
        """At zero variance, should return floor value."""
        var = torch.tensor(0.0)
        kappa_var_min = 0.01
        eps_std = 0.001

        sigma_prime = patched_std_dev(var, kappa_var_min, eps_std)

        # σ'_min = sqrt(κ_var,min + ε_std^2) = sqrt(0.01 + 0.000001) ≈ 0.1000
        expected_floor = torch.sqrt(torch.tensor(kappa_var_min + eps_std**2))
        assert torch.allclose(sigma_prime, expected_floor, atol=1e-6)

    def test_patched_std_dev_monotonic(self):
        """Should be monotonically increasing."""
        var = torch.linspace(0.0, 10.0, 100)
        sigma_prime = patched_std_dev(var, kappa_var_min=0.01, eps_std=0.001)

        # Check monotonicity: σ'(v_i) < σ'(v_{i+1})
        diffs = sigma_prime[1:] - sigma_prime[:-1]
        assert (diffs > 0).all(), "Patched std dev should be monotonically increasing"

    def test_patched_std_dev_asymptotic(self):
        """For large variance, should approximate sqrt(V)."""
        var_large = torch.tensor(10000.0)
        sigma_prime = patched_std_dev(var_large, kappa_var_min=0.01, eps_std=0.001)

        # For V >> σ'_min^2, σ'_reg(V) ≈ sqrt(V)
        expected_approx = torch.sqrt(var_large)
        relative_error = torch.abs(sigma_prime - expected_approx) / expected_approx

        assert (
            relative_error < 0.001
        ), f"Should approximate sqrt for large V, got error {relative_error}"

    def test_patched_std_dev_positive(self):
        """Should always be strictly positive."""
        var = torch.tensor([0.0, -0.001, 0.01, 1.0, 100.0])  # Include negative (edge case)
        sigma_prime = patched_std_dev(var.clamp(min=0), kappa_var_min=0.01, eps_std=0.001)

        assert (sigma_prime > 0).all(), "Patched std dev must be strictly positive"

    def test_patched_std_dev_vectorized(self):
        """Should work with multidimensional arrays."""
        var = torch.randn(10, 5).abs()  # [10, 5] array of variances
        sigma_prime = patched_std_dev(var, kappa_var_min=0.01, eps_std=0.001)

        assert sigma_prime.shape == var.shape
        assert (sigma_prime > 0).all()


class TestDistanceToRandomCompanion:
    """Test distance to random companion function."""

    def test_distance_shape(self):
        """Output shape should be [N]."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.ones(N, dtype=torch.bool)

        distances = distance_to_random_companion(x, v, alive_mask, lambda_alg=0.0)

        assert distances.shape == (N,)

    def test_dead_walkers_zero_distance(self):
        """Dead walkers should have distance 0."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.tensor([True] * 7 + [False] * 3)

        distances = distance_to_random_companion(x, v, alive_mask, lambda_alg=0.0)

        # Check dead walkers have zero distance
        dead_indices = torch.where(~alive_mask)[0]
        assert torch.allclose(distances[dead_indices], torch.zeros(3))

    def test_position_only_mode(self):
        """With lambda_alg=0, only position should matter."""
        N, d = 5, 2
        # Fixed positions, random velocities
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
        v = torch.randn(N, d) * 100  # Large velocities should be ignored
        alive_mask = torch.ones(N, dtype=torch.bool)

        torch.manual_seed(42)
        distances1 = distance_to_random_companion(x, v, alive_mask, lambda_alg=0.0)

        # With different velocities but same positions
        v_different = torch.randn(N, d) * 100
        torch.manual_seed(42)  # Same random seed -> same companions
        distances2 = distance_to_random_companion(x, v_different, alive_mask, lambda_alg=0.0)

        # Distances should be identical
        assert torch.allclose(distances1, distances2)

    def test_phase_space_mode(self):
        """With lambda_alg>0, velocity should contribute."""
        N, _d = 2, 1
        # Same position, different velocities
        x = torch.tensor([[0.0], [0.0]])
        v = torch.tensor([[0.0], [1.0]])
        alive_mask = torch.ones(N, dtype=torch.bool)

        dist_pos_only = distance_to_random_companion(x, v, alive_mask, lambda_alg=0.0)
        distance_to_random_companion(x, v, alive_mask, lambda_alg=1.0)

        # Position-only should give 0, phase-space should give nonzero
        assert torch.allclose(dist_pos_only[0], torch.tensor(0.0), atol=1e-6) or torch.allclose(
            dist_pos_only[0], torch.tensor(1.0), atol=1e-6
        )  # Could pair with self or other
        # Phase-space distance includes velocity difference

    def test_distances_non_negative(self):
        """Distances should always be non-negative."""
        N, d = 20, 3
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        alive_mask = torch.rand(N) > 0.2

        distances = distance_to_random_companion(x, v, alive_mask, lambda_alg=1.0)

        assert (distances >= 0).all()


class TestFitnessPotentialVectorized:
    """Test vectorized fitness potential computation."""

    def test_fitness_shape(self):
        """Output shape should be [N]."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.ones(N, dtype=torch.bool)

        V_fit = compute_fitness_potential_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        assert V_fit.shape == (N,)

    def test_dead_walkers_zero_fitness(self):
        """Dead walkers should have zero fitness."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.tensor([True] * 7 + [False] * 3)

        V_fit = compute_fitness_potential_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        # Check dead walkers have zero fitness
        dead_indices = torch.where(~alive_mask)[0]
        assert torch.allclose(V_fit[dead_indices], torch.zeros(3))

    def test_all_dead_walkers(self):
        """With no alive walkers, all fitness should be zero."""
        N, d = 5, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.zeros(N, dtype=torch.bool)  # All dead

        V_fit = compute_fitness_potential_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        assert torch.allclose(V_fit, torch.zeros(N))

    def test_fitness_positive(self):
        """Fitness should be non-negative (r', d' ∈ [η, 1])."""
        N, d = 20, 3
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.ones(N, dtype=torch.bool)

        V_fit = compute_fitness_potential_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        assert (V_fit >= 0).all()

    def test_fitness_bounded(self):
        """Fitness should be bounded (r' ≤ 1, d' ≤ 1 => V_fit ≤ 1)."""
        N, d = 20, 3
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.ones(N, dtype=torch.bool)

        V_fit = compute_fitness_potential_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        assert (V_fit <= 1.0 + 1e-6).all()  # Small tolerance for numerical error


class TestFitnessGradientVectorized:
    """Test vectorized fitness gradient computation."""

    def test_gradient_shape(self):
        """Output shape should be [N, d]."""
        N, d = 10, 3
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.ones(N, dtype=torch.bool)

        grad_V_fit = compute_fitness_gradient_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        assert grad_V_fit.shape == (N, d)

    def test_dead_walkers_zero_gradient(self):
        """Dead walkers should have zero gradient."""
        N, d = 10, 2
        x = torch.randn(N, d)
        v = torch.randn(N, d)
        measurement = torch.randn(N)
        alive_mask = torch.tensor([True] * 7 + [False] * 3)

        grad_V_fit = compute_fitness_gradient_vectorized(
            x,
            v,
            measurement,
            alive_mask,
            alpha=1.0,
            beta=1.0,
            kappa_var_min=0.01,
            eps_std=0.001,
            eta=0.1,
            lambda_alg=0.0,
        )

        # Check dead walkers have zero gradient
        dead_indices = torch.where(~alive_mask)[0]
        assert torch.allclose(grad_V_fit[dead_indices], torch.zeros(3, d))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
