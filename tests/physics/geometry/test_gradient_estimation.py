"""Tests for fragile.physics.geometry.gradient_estimation module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.gradient_estimation import (
    compute_directional_derivative,
    estimate_gradient_finite_difference,
    estimate_gradient_quality_metrics,
)
from fragile.physics.geometry.neighbors import build_csr_from_coo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interior_mask(positions: Tensor, margin: float = 0.5) -> Tensor:
    """Boolean mask for points away from boundary (avoids FD edge effects)."""
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    return (
        (positions[:, 0] > x_min + margin)
        & (positions[:, 0] < x_max - margin)
        & (positions[:, 1] > y_min + margin)
        & (positions[:, 1] < y_max - margin)
    )


# ===========================================================================
# TestEstimateGradientFiniteDifference
# ===========================================================================


class TestEstimateGradientFiniteDifference:
    """Tests for estimate_gradient_finite_difference."""

    def test_quadratic_gradient_direction(self, quadratic_fitness: dict):
        """Quadratic V=x^2+2y^2: gradient estimates should correlate with [2x, 4y]."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        grad_fn = quadratic_fitness["grad_fn"]

        result = estimate_gradient_finite_difference(pos, fit, ei)
        grad_est = result["gradient"]
        valid = result["valid_mask"]
        interior = _interior_mask(pos)
        sel = valid & interior

        grad_true = grad_fn(pos)
        # For each dimension, check that estimated and true gradients are
        # positively correlated (Pearson r > 0.5). FD on irregular Delaunay
        # grids is approximate, so we check correlation rather than pointwise.
        for dim in range(2):
            est = grad_est[sel, dim]
            true = grad_true[sel, dim]
            # Pearson correlation
            est_centered = est - est.mean()
            true_centered = true - true.mean()
            r = (est_centered * true_centered).sum() / (
                est_centered.norm() * true_centered.norm() + 1e-10
            )
            assert r.item() > 0.5, f"dim={dim}: correlation r={r.item():.3f} too low"

    def test_linear_gradient_accuracy(self, linear_fitness: dict):
        """Linear V=3x+5y: gradient should be nearly exact [3, 5] for interior."""
        pos = linear_fitness["positions"]
        fit = linear_fitness["fitness"]
        ei = linear_fitness["edge_index"]
        grad_true = linear_fitness["grad_true"]

        result = estimate_gradient_finite_difference(pos, fit, ei)
        grad_est = result["gradient"]
        valid = result["valid_mask"]
        interior = _interior_mask(pos)
        sel = valid & interior

        # For a linear function, FD should be very accurate
        for dim in range(2):
            mean_est = grad_est[sel, dim].mean().item()
            true_val = grad_true[0, dim].item()
            assert abs(mean_est - true_val) / abs(true_val) < 0.15, (
                f"dim={dim}: mean est={mean_est:.3f} vs true={true_val:.3f}"
            )

    def test_output_shapes(self, quadratic_fitness: dict):
        """Check that gradient [N, d] and gradient_magnitude [N] have correct shapes."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N, d = pos.shape

        result = estimate_gradient_finite_difference(pos, fit, ei)
        assert result["gradient"].shape == (N, d)
        assert result["gradient_magnitude"].shape == (N,)
        assert result["num_neighbors"].shape == (N,)
        assert result["estimation_quality"].shape == (N,)
        assert result["valid_mask"].shape == (N,)

    def test_output_keys(self, quadratic_fitness: dict):
        """All expected keys must be present."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]

        result = estimate_gradient_finite_difference(pos, fit, ei)
        expected_keys = {
            "gradient",
            "gradient_magnitude",
            "num_neighbors",
            "estimation_quality",
            "valid_mask",
        }
        assert set(result.keys()) == expected_keys

    def test_valid_mask_interior(self, quadratic_fitness: dict):
        """Most interior walkers should have valid_mask=True."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]

        result = estimate_gradient_finite_difference(pos, fit, ei)
        interior = _interior_mask(pos)
        valid_interior = result["valid_mask"][interior]
        # At least 80% of interior points should be valid
        frac = valid_interior.float().mean().item()
        assert frac > 0.80, f"Only {frac:.1%} of interior walkers are valid"

    def test_alive_mask_dead_walkers(self, quadratic_fitness: dict):
        """Dead walkers (alive=False) should have NaN gradients."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        alive = torch.ones(N, dtype=torch.bool)
        alive[0] = False
        alive[5] = False

        result = estimate_gradient_finite_difference(pos, fit, ei, alive=alive)
        assert torch.isnan(result["gradient"][0]).all()
        assert torch.isnan(result["gradient"][5]).all()

    def test_nan_fitness_excluded(self, quadratic_fitness: dict):
        """Walkers with NaN fitness should be marked invalid."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"].clone()
        ei = quadratic_fitness["edge_index"]

        fit[10] = float("nan")
        result = estimate_gradient_finite_difference(pos, fit, ei)
        assert not result["valid_mask"][10]

    def test_csr_input(self, quadratic_fitness: dict):
        """CSR input should give results consistent with COO input."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        # Build CSR with type=0 (all walker-walker edges)
        edge_types = torch.zeros(ei.shape[1], dtype=torch.long)
        csr = build_csr_from_coo(ei, n_nodes=N, edge_types=edge_types)

        result_coo = estimate_gradient_finite_difference(pos, fit, ei)
        result_csr = estimate_gradient_finite_difference(
            pos,
            fit,
            edge_index=None,
            csr_ptr=csr["csr_ptr"],
            csr_indices=csr["csr_indices"],
            csr_types=csr["csr_types"],
        )

        # Both should have the same valid walkers and similar gradients
        both_valid = result_coo["valid_mask"] & result_csr["valid_mask"]
        if both_valid.any():
            torch.testing.assert_close(
                result_coo["gradient"][both_valid],
                result_csr["gradient"][both_valid],
                atol=1e-4,
                rtol=1e-4,
            )

    @pytest.mark.parametrize("weight_mode", ["uniform", "inverse_distance"])
    def test_weight_mode_parametrize(self, quadratic_fitness: dict, weight_mode: str):
        """Both weight modes should produce gradient estimates of correct shape."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N, d = pos.shape

        result = estimate_gradient_finite_difference(pos, fit, ei, weight_mode=weight_mode)
        assert result["gradient"].shape == (N, d)
        assert result["gradient_magnitude"].shape == (N,)

    def test_precomputed_edge_weights(self, quadratic_fitness: dict):
        """Custom edge_weights should be accepted and produce correct-shape output."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N, d = pos.shape
        E = ei.shape[1]

        weights = torch.ones(E, dtype=pos.dtype)
        result = estimate_gradient_finite_difference(pos, fit, ei, edge_weights=weights)
        assert result["gradient"].shape == (N, d)
        assert result["gradient_magnitude"].shape == (N,)


# ===========================================================================
# TestComputeDirectionalDerivative
# ===========================================================================


class TestComputeDirectionalDerivative:
    """Tests for compute_directional_derivative."""

    def test_x_direction_quadratic(self, quadratic_fitness: dict):
        """dV/dx of V=x^2+2y^2 should approximate 2x for valid interior walkers."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]

        direction = torch.tensor([1.0, 0.0])
        dd = compute_directional_derivative(pos, fit, ei, direction)

        interior = _interior_mask(pos)
        valid = torch.isfinite(dd) & interior

        true_dVdx = 2.0 * pos[:, 0]
        # Check sign agreement and rough magnitude for valid interior
        signs_agree = (dd[valid] * true_dVdx[valid]) >= 0
        frac_correct_sign = signs_agree.float().mean().item()
        assert frac_correct_sign > 0.70, (
            f"Only {frac_correct_sign:.1%} of walkers have correct dV/dx sign"
        )

    def test_y_direction_quadratic(self, quadratic_fitness: dict):
        """dV/dy of V=x^2+2y^2 should approximate 4y for valid interior walkers."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]

        direction = torch.tensor([0.0, 1.0])
        dd = compute_directional_derivative(pos, fit, ei, direction)

        interior = _interior_mask(pos)
        valid = torch.isfinite(dd) & interior

        true_dVdy = 4.0 * pos[:, 1]
        signs_agree = (dd[valid] * true_dVdy[valid]) >= 0
        frac_correct_sign = signs_agree.float().mean().item()
        assert frac_correct_sign > 0.70, (
            f"Only {frac_correct_sign:.1%} of walkers have correct dV/dy sign"
        )

    def test_per_walker_direction(self, quadratic_fitness: dict):
        """Per-walker directions [N, d] should produce output shape [N]."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N, d = pos.shape

        # Random per-walker directions
        gen = torch.Generator().manual_seed(7)
        directions = torch.randn(N, d, generator=gen)

        dd = compute_directional_derivative(pos, fit, ei, directions)
        assert dd.shape == (N,)

    def test_single_direction_broadcast(self, quadratic_fitness: dict):
        """Single direction [d] should broadcast to all walkers, output shape [N]."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        direction = torch.tensor([1.0, 0.0])
        dd = compute_directional_derivative(pos, fit, ei, direction)
        assert dd.shape == (N,)

    def test_csr_input(self, quadratic_fitness: dict):
        """CSR input should give consistent results with COO."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        edge_types = torch.zeros(ei.shape[1], dtype=torch.long)
        csr = build_csr_from_coo(ei, n_nodes=N, edge_types=edge_types)

        direction = torch.tensor([1.0, 0.0])

        dd_coo = compute_directional_derivative(pos, fit, ei, direction)
        dd_csr = compute_directional_derivative(
            pos,
            fit,
            edge_index=None,
            direction=direction,
            csr_ptr=csr["csr_ptr"],
            csr_indices=csr["csr_indices"],
            csr_types=csr["csr_types"],
        )

        both_valid = torch.isfinite(dd_coo) & torch.isfinite(dd_csr)
        if both_valid.any():
            torch.testing.assert_close(
                dd_coo[both_valid], dd_csr[both_valid], atol=1e-4, rtol=1e-4
            )

    def test_alive_mask_dead_walkers(self, quadratic_fitness: dict):
        """Dead walkers (alive=False) should get NaN directional derivative."""
        pos = quadratic_fitness["positions"]
        fit = quadratic_fitness["fitness"]
        ei = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        alive = torch.ones(N, dtype=torch.bool)
        alive[0] = False
        alive[3] = False

        direction = torch.tensor([1.0, 0.0])
        dd = compute_directional_derivative(pos, fit, ei, direction, alive=alive)
        assert torch.isnan(dd[0])
        assert torch.isnan(dd[3])


# ===========================================================================
# TestEstimateGradientQualityMetrics
# ===========================================================================


class TestEstimateGradientQualityMetrics:
    """Tests for estimate_gradient_quality_metrics."""

    def _get_gradient_result(self, fitness_data: dict) -> dict:
        """Helper to compute gradient result from a fitness fixture."""
        return estimate_gradient_finite_difference(
            fitness_data["positions"],
            fitness_data["fitness"],
            fitness_data["edge_index"],
        )

    def test_output_keys(self, quadratic_fitness: dict):
        """All expected quality metric keys must be present."""
        result = self._get_gradient_result(quadratic_fitness)
        metrics = estimate_gradient_quality_metrics(
            result,
            quadratic_fitness["positions"],
            quadratic_fitness["fitness"],
            quadratic_fitness["edge_index"],
        )
        expected_keys = {
            "valid_fraction",
            "mean_neighbors",
            "mean_quality",
            "gradient_norm_mean",
            "gradient_norm_std",
        }
        assert set(metrics.keys()) == expected_keys

    def test_valid_fraction_range(self, quadratic_fitness: dict):
        """valid_fraction must be in [0, 1]."""
        result = self._get_gradient_result(quadratic_fitness)
        metrics = estimate_gradient_quality_metrics(
            result,
            quadratic_fitness["positions"],
            quadratic_fitness["fitness"],
            quadratic_fitness["edge_index"],
        )
        assert 0.0 <= metrics["valid_fraction"] <= 1.0

    def test_mean_neighbors_positive(self, quadratic_fitness: dict):
        """mean_neighbors should be > 0 for a connected graph."""
        result = self._get_gradient_result(quadratic_fitness)
        metrics = estimate_gradient_quality_metrics(
            result,
            quadratic_fitness["positions"],
            quadratic_fitness["fitness"],
            quadratic_fitness["edge_index"],
        )
        assert metrics["mean_neighbors"] > 0

    def test_gradient_norm_mean_positive(self, quadratic_fitness: dict):
        """For quadratic fitness (non-zero gradient), gradient_norm_mean > 0."""
        result = self._get_gradient_result(quadratic_fitness)
        metrics = estimate_gradient_quality_metrics(
            result,
            quadratic_fitness["positions"],
            quadratic_fitness["fitness"],
            quadratic_fitness["edge_index"],
        )
        assert metrics["gradient_norm_mean"] > 0
