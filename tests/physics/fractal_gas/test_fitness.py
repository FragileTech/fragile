"""Tests for fragile.physics.fractal_gas.fitness module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.fractal_gas.fitness import (
    compute_fitness,
    FitnessOperator,
    global_stats,
    logistic_rescale,
    patched_standardization,
)


# =============================================================================
# TestLogisticRescale
# =============================================================================


class TestLogisticRescale:
    """Tests for the logistic rescale function g_A(z) = A / (1 + exp(-z))."""

    @pytest.mark.parametrize("A", [1.0, 2.0, 5.0, 0.5])
    def test_output_range(self, A: float):
        """Output values lie in [0, A] for various A."""
        z = torch.linspace(-10.0, 10.0, 200)
        result = logistic_rescale(z, A=A)
        assert (result >= 0.0).all(), f"Found values below 0 for A={A}"
        assert (result <= A).all(), f"Found values above A={A}"

    def test_monotone_increasing(self):
        """g_A is monotone increasing: z1 < z2 implies g(z1) < g(z2)."""
        z = torch.linspace(-10.0, 10.0, 500)
        result = logistic_rescale(z, A=2.0)
        diffs = result[1:] - result[:-1]
        assert (diffs >= 0.0).all(), "logistic_rescale is not monotone increasing"

    @pytest.mark.parametrize("A", [1.0, 2.0, 3.5])
    def test_midpoint_property(self, A: float):
        """g_A(0) = A/2 (midpoint of the sigmoid)."""
        z = torch.tensor([0.0])
        result = logistic_rescale(z, A=A)
        torch.testing.assert_close(result, torch.tensor([A / 2.0]))

    def test_large_positive_z_asymptote(self):
        """For large positive z, g_A(z) -> A."""
        z = torch.tensor([40.0])
        result = logistic_rescale(z, A=2.0)
        assert result.item() == pytest.approx(2.0, abs=1e-5)

    def test_large_negative_z_asymptote(self):
        """For large negative z, g_A(z) -> 0."""
        z = torch.tensor([-40.0])
        result = logistic_rescale(z, A=2.0)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_nan_handling(self):
        """NaN values in input are replaced with 0 (mapped to A/2)."""
        z = torch.tensor([float("nan"), 0.0, float("nan")])
        result = logistic_rescale(z, A=2.0)
        assert not torch.isnan(result).any(), "NaN not handled in output"
        # nan -> 0 -> sigmoid(0) * A = A/2
        torch.testing.assert_close(result[0], torch.tensor(1.0))
        torch.testing.assert_close(result[2], torch.tensor(1.0))

    def test_shape_preservation(self):
        """Output tensor has the same shape as input."""
        for shape in [(10,), (3, 4), (2, 3, 5)]:
            z = torch.randn(shape)
            result = logistic_rescale(z, A=1.0)
            assert result.shape == z.shape, f"Shape mismatch for input shape {shape}"


# =============================================================================
# TestGlobalStats
# =============================================================================


class TestGlobalStats:
    """Tests for global_stats (mean and regularized std)."""

    def test_mean_is_correct(self):
        """Mean of global_stats matches torch.mean."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, _ = global_stats(values, sigma_min=1e-8)
        torch.testing.assert_close(mu, values.mean())

    def test_sigma_regularized_lower_bound(self):
        """Regularized sigma >= sigma_min."""
        sigma_min = 0.1
        values = torch.tensor([1.0, 2.0, 3.0])
        _, sigma_reg = global_stats(values, sigma_min=sigma_min)
        assert sigma_reg.item() >= sigma_min

    def test_constant_values_sigma_equals_sigma_min(self):
        """For constant input, variance = 0, so sigma_reg = sigma_min."""
        sigma_min = 0.01
        values = torch.full((10,), 5.0)
        _, sigma_reg = global_stats(values, sigma_min=sigma_min)
        assert sigma_reg.item() == pytest.approx(sigma_min, rel=1e-5)

    def test_output_shapes_scalar(self):
        """Both mu and sigma_reg are scalar tensors."""
        values = torch.randn(20)
        mu, sigma_reg = global_stats(values)
        assert mu.dim() == 0, "mu should be a scalar tensor"
        assert sigma_reg.dim() == 0, "sigma_reg should be a scalar tensor"

    def test_sigma_min_zero_with_variance(self):
        """When sigma_min=0 and input has variance, sigma_reg = std."""
        values = torch.tensor([1.0, 3.0, 5.0, 7.0])
        _, sigma_reg = global_stats(values, sigma_min=0.0)
        expected_sigma = torch.sqrt(values.var())
        torch.testing.assert_close(sigma_reg, expected_sigma)


# =============================================================================
# TestPatchedStandardization
# =============================================================================


class TestPatchedStandardization:
    """Tests for patched_standardization (Z-score computation)."""

    def test_mean_of_zscores_near_zero(self):
        """Mean of Z-scores should be approximately 0."""
        gen = torch.Generator().manual_seed(0)
        values = torch.randn(100, generator=gen) * 5.0 + 3.0
        z = patched_standardization(values, sigma_min=1e-8)
        assert z.mean().abs().item() < 1e-5, f"Mean of Z-scores = {z.mean().item()}"

    def test_std_of_zscores_near_one(self):
        """Std of Z-scores should be approximately 1 when variance >> sigma_min."""
        gen = torch.Generator().manual_seed(1)
        values = torch.randn(1000, generator=gen) * 10.0
        z = patched_standardization(values, sigma_min=1e-12)
        z_std = z.std().item()
        assert z_std == pytest.approx(1.0, abs=0.05), f"Std of Z-scores = {z_std}"

    def test_return_statistics_true(self):
        """return_statistics=True returns (z, mu, sigma) tuple."""
        values = torch.randn(20)
        result = patched_standardization(values, return_statistics=True)
        assert isinstance(result, tuple), "Expected tuple with return_statistics=True"
        assert len(result) == 3, "Expected 3 elements in tuple"
        z, mu, sigma = result
        assert z.shape == values.shape
        assert mu.dim() == 0
        assert sigma.dim() == 0

    def test_return_statistics_false(self):
        """return_statistics=False returns just z tensor."""
        values = torch.randn(20)
        result = patched_standardization(values, return_statistics=False)
        assert isinstance(result, Tensor), "Expected Tensor with return_statistics=False"
        assert result.shape == values.shape

    def test_detach_stats_no_grad(self):
        """detach_stats=True: mu and sigma have no gradient."""
        values = torch.randn(20, requires_grad=True)
        _z, mu, sigma = patched_standardization(
            values,
            detach_stats=True,
            return_statistics=True,
        )
        assert not mu.requires_grad, "mu should be detached"
        assert not sigma.requires_grad, "sigma should be detached"

    def test_constant_input_zscores_zero(self):
        """For constant input, all Z-scores should be zero."""
        values = torch.full((10,), 42.0)
        z = patched_standardization(values, sigma_min=1e-3)
        torch.testing.assert_close(z, torch.zeros(10))


# =============================================================================
# TestComputeFitness
# =============================================================================


class TestComputeFitness:
    """Tests for compute_fitness pipeline."""

    def test_output_shape(self, positions: Tensor, rewards: Tensor, companions: Tensor, N: int):
        """Fitness output shape is [N]."""
        fitness, _info = compute_fitness(positions, rewards, companions)
        assert fitness.shape == (N,)

    def test_info_dict_keys(self, positions: Tensor, rewards: Tensor, companions: Tensor):
        """Info dict contains all expected keys."""
        _, info = compute_fitness(positions, rewards, companions)
        expected_keys = {
            "distances",
            "z_rewards",
            "z_distances",
            "rescaled_rewards",
            "rescaled_distances",
            "mu_rewards",
            "sigma_rewards",
            "mu_distances",
            "sigma_distances",
            "companions",
            "pos_squared_differences",
            "vel_squared_differences",
        }
        assert set(info.keys()) == expected_keys

    def test_fitness_is_positive(self, positions: Tensor, rewards: Tensor, companions: Tensor):
        """Fitness values should be positive with default params (eta=0, A=2)."""
        fitness, _ = compute_fitness(positions, rewards, companions, eta=0.0, A=2.0)
        # With eta=0, rescaled values are in (0, A) so fitness = d^beta * r^alpha > 0
        assert (fitness >= 0.0).all(), "Fitness should be non-negative"

    def test_self_companion_zero_distances(self, N: int, d: int):
        """When companions=arange(N), distances should be zero."""
        gen = torch.Generator().manual_seed(99)
        positions = torch.randn(N, d, generator=gen)
        rewards = torch.randn(N, generator=gen)
        companions = torch.arange(N)
        _, info = compute_fitness(positions, rewards, companions)
        torch.testing.assert_close(
            info["distances"],
            torch.zeros(N),
            atol=1e-7,
            rtol=0.0,
        )

    def test_alpha_zero_independence_from_rewards(self, positions: Tensor, companions: Tensor):
        """With alpha=0, fitness should be independent of reward values."""
        N = positions.shape[0]
        gen1 = torch.Generator().manual_seed(10)
        gen2 = torch.Generator().manual_seed(20)
        rewards_a = torch.randn(N, generator=gen1)
        rewards_b = torch.randn(N, generator=gen2) * 100.0

        fitness_a, _ = compute_fitness(positions, rewards_a, companions, alpha=0.0)
        fitness_b, _ = compute_fitness(positions, rewards_b, companions, alpha=0.0)
        torch.testing.assert_close(fitness_a, fitness_b)

    def test_beta_zero_independence_from_distances(self, rewards: Tensor, N: int, d: int):
        """With beta=0, fitness should be independent of positions (distances)."""
        gen1 = torch.Generator().manual_seed(30)
        gen2 = torch.Generator().manual_seed(40)
        positions_a = torch.randn(N, d, generator=gen1)
        positions_b = torch.randn(N, d, generator=gen2) * 100.0
        companions = torch.arange(N)  # use self to ensure same companion mapping

        fitness_a, _ = compute_fitness(positions_a, rewards, companions, beta=0.0)
        fitness_b, _ = compute_fitness(positions_b, rewards, companions, beta=0.0)
        torch.testing.assert_close(fitness_a, fitness_b)

    def test_eta_floor(self, positions: Tensor, rewards: Tensor, companions: Tensor):
        """With eta > 0, fitness >= eta^(alpha+beta) approximately."""
        eta = 0.1
        alpha, beta = 1.0, 1.0
        fitness, _info = compute_fitness(
            positions,
            rewards,
            companions,
            eta=eta,
            alpha=alpha,
            beta=beta,
        )
        # r_prime >= eta (since logistic >= 0), d_prime >= eta
        # fitness = d_prime^beta * r_prime^alpha >= eta^beta * eta^alpha = eta^(alpha+beta)
        lower_bound = eta ** (alpha + beta)
        assert (
            fitness >= lower_bound - 1e-7
        ).all(), f"Fitness below eta floor: min={fitness.min().item()}, bound={lower_bound}"

    def test_rescaled_values_in_range(
        self, positions: Tensor, rewards: Tensor, companions: Tensor
    ):
        """Rescaled rewards and distances should be in [0, A] (before eta)."""
        A = 3.0
        _, info = compute_fitness(positions, rewards, companions, A=A, eta=0.0)
        assert (info["rescaled_rewards"] >= 0.0).all()
        assert (info["rescaled_rewards"] <= A).all()
        assert (info["rescaled_distances"] >= 0.0).all()
        assert (info["rescaled_distances"] <= A).all()


# =============================================================================
# TestFitnessOperator
# =============================================================================


class TestFitnessOperator:
    """Tests for the FitnessOperator Panel wrapper."""

    def test_callable_returns_tuple(
        self,
        fitness_op: FitnessOperator,
        positions: Tensor,
        rewards: Tensor,
        companions: Tensor,
    ):
        """Calling the operator returns (fitness, info) tuple."""
        result = fitness_op(positions, rewards, companions)
        assert isinstance(result, tuple)
        assert len(result) == 2
        fitness, info = result
        assert isinstance(fitness, Tensor)
        assert isinstance(info, dict)
        assert fitness.shape == (positions.shape[0],)

    def test_parameters_stored_correctly(self):
        """Operator stores custom parameter values."""
        op = FitnessOperator(alpha=2.5, beta=0.5, eta=0.01, sigma_min=1e-4, A=3.0)
        assert op.alpha == 2.5
        assert op.beta == 0.5
        assert op.eta == 0.01
        assert op.sigma_min == 1e-4
        assert op.A == 3.0

    def test_override_defaults(self, positions: Tensor, rewards: Tensor, companions: Tensor):
        """Changing parameters changes fitness output."""
        op_default = FitnessOperator(alpha=1.0, beta=1.0, A=2.0)
        op_custom = FitnessOperator(alpha=2.0, beta=0.5, A=3.0)

        fitness_default, _ = op_default(positions, rewards, companions)
        fitness_custom, _ = op_custom(positions, rewards, companions)

        # Different parameters should give different fitness
        assert not torch.allclose(
            fitness_default, fitness_custom
        ), "Different parameters should produce different fitness"

    def test_widget_parameters_list(self):
        """widget_parameters contains the expected parameter names."""
        op = FitnessOperator()
        params = op.widget_parameters
        expected = {"alpha", "beta", "eta", "sigma_min", "A"}
        assert set(params) == expected
