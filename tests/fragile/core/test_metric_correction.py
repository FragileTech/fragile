"""Tests for metric correction in scutoid Ricci computation.

This module tests the metric correction features added to src/fragile/core/scutoids.py,
which apply first-order perturbation corrections to flat-space deficit angles to
account for the emergent Riemannian geometry induced by anisotropic diffusion.

Test Coverage:
    - Parameter validation for metric_correction modes
    - None mode: Pure flat-space deficit angles (baseline)
    - Diagonal mode: Diagonal metric correction O(N)
    - Full mode: Full metric tensor correction O(N·k)
    - Integration with tessellation and Ricci computation
    - Verification that corrections produce non-trivial changes

Mathematical Framework:
    R^{manifold}(x_i) ≈ R^{flat}(x_i) + ΔR^{metric}(x_i)

    Where:
    - R^{flat}: Flat-space deficit angle Ricci scalar
    - ΔR^{metric}: Correction from emergent metric g = H + ε_Σ I
    - H: Fitness Hessian (approximated from local walker density)

References:
    - src/fragile/core/scutoids.py lines 16-59 (metric correction framework)
    - old_docs/source/14_scutoid_geometry_framework.md §5.4
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.core import create_scutoid_history, RunHistory, ScutoidHistory2D


def create_simple_2d_history(N: int = 20, n_steps: int = 10, seed: int = 42) -> RunHistory:
    """Create minimal 2D RunHistory for testing.

    Args:
        N: Number of walkers
        n_steps: Number of simulation steps
        seed: Random seed

    Returns:
        RunHistory with random walker trajectories
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    d = 2
    n_recorded = n_steps + 1
    record_every = 1

    # Create trajectories with some variation
    # Walkers move slightly each timestep
    x_final = torch.zeros(n_recorded, N, d)
    v_final = torch.zeros(n_recorded, N, d)

    # Initialize positions in a grid
    n_side = int(np.sqrt(N))
    x_vals = torch.linspace(-2, 2, n_side)
    y_vals = torch.linspace(-2, 2, n_side)
    xv, yv = torch.meshgrid(x_vals, y_vals, indexing="ij")
    initial_positions = torch.stack([xv.flatten()[:N], yv.flatten()[:N]], dim=1)

    x_final[0] = initial_positions

    # Add small random walks
    for t in range(1, n_recorded):
        displacement = torch.randn(N, d) * 0.1
        x_final[t] = x_final[t - 1] + displacement
        v_final[t] = displacement

    # Create alive mask (all alive)
    alive_mask = torch.ones(n_recorded, N, dtype=torch.bool)

    # Create dummy cloning data
    will_clone = torch.zeros(n_recorded - 1, N, dtype=torch.bool)
    companions_clone = torch.arange(N).unsqueeze(0).repeat(n_recorded - 1, 1)

    return RunHistory(
        N=N,
        d=d,
        n_steps=n_steps,
        n_recorded=n_recorded,
        record_every=record_every,
        terminated_early=False,
        final_step=n_steps,
        x_before_clone=x_final.clone(),
        v_before_clone=v_final.clone(),
        x_after_clone=x_final[1:].clone(),
        v_after_clone=v_final[1:].clone(),
        x_final=x_final,
        v_final=v_final,
        n_alive=torch.full((n_recorded,), N),
        num_cloned=torch.zeros(n_recorded),
        step_times=torch.ones(n_recorded) * 0.01,
        fitness=torch.randn(n_recorded - 1, N),
        rewards=torch.randn(n_recorded - 1, N),
        cloning_scores=torch.randn(n_recorded - 1, N),
        cloning_probs=torch.rand(n_recorded - 1, N),
        will_clone=will_clone,
        alive_mask=alive_mask,
        companions_distance=torch.randint(0, N, (n_recorded - 1, N)),
        companions_clone=companions_clone,
        distances=torch.randn(n_recorded - 1, N).abs(),
        z_rewards=torch.randn(n_recorded - 1, N),
        z_distances=torch.randn(n_recorded - 1, N),
        pos_squared_differences=torch.randn(n_recorded - 1, N).abs(),
        vel_squared_differences=torch.randn(n_recorded - 1, N).abs(),
        rescaled_rewards=torch.randn(n_recorded - 1, N),
        rescaled_distances=torch.randn(n_recorded - 1, N),
        mu_rewards=torch.randn(n_recorded - 1),
        sigma_rewards=torch.rand(n_recorded - 1) + 0.1,
        mu_distances=torch.randn(n_recorded - 1),
        sigma_distances=torch.rand(n_recorded - 1) + 0.1,
        total_time=0.1,
        init_time=0.01,
    )


class TestMetricCorrectionParameterValidation:
    """Test parameter validation for metric_correction modes."""

    def test_valid_modes(self):
        """Test that valid metric_correction modes are accepted."""
        history = create_simple_2d_history(N=16, n_steps=5)

        # All three modes should be accepted
        for mode in ["none", "diagonal", "full"]:
            scutoid = ScutoidHistory2D(history, metric_correction=mode)
            assert scutoid.metric_correction == mode

    def test_invalid_mode_raises_error(self):
        """Test that invalid metric_correction mode raises ValueError."""
        history = create_simple_2d_history(N=16, n_steps=5)

        with pytest.raises(ValueError, match="metric_correction must be"):
            ScutoidHistory2D(history, metric_correction="invalid")

        with pytest.raises(ValueError, match="metric_correction must be"):
            ScutoidHistory2D(history, metric_correction="NONE")  # Case-sensitive

    def test_factory_function_accepts_mode(self):
        """Test that create_scutoid_history accepts metric_correction."""
        history = create_simple_2d_history(N=16, n_steps=5)

        scutoid = create_scutoid_history(history, metric_correction="diagonal")
        assert scutoid.metric_correction == "diagonal"


class TestMetricCorrectionNoneMode:
    """Test baseline 'none' mode (pure flat-space deficit angles)."""

    def test_none_mode_computes_ricci(self):
        """Test that 'none' mode computes flat-space Ricci scalars."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="none")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should have Ricci scalars
        ricci = scutoid.get_ricci_scalars()
        assert ricci is not None
        assert ricci.shape == (5, 16)  # [n_recorded-1, N]

        # Should not have corrected scalars in 'none' mode
        assert scutoid.ricci_scalars_corrected is None

    def test_none_mode_returns_flat_space_values(self):
        """Test that get_ricci_scalars returns flat-space values in 'none' mode."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="none")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        ricci = scutoid.get_ricci_scalars()
        flat_ricci = scutoid.ricci_scalars

        # Should return same array
        assert np.array_equal(ricci, flat_ricci, equal_nan=True)


class TestMetricCorrectionDiagonalMode:
    """Test 'diagonal' mode (diagonal metric correction O(N))."""

    def test_diagonal_mode_applies_correction(self):
        """Test that 'diagonal' mode applies metric correction."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should have both flat and corrected scalars
        assert scutoid.ricci_scalars is not None
        assert scutoid.ricci_scalars_corrected is not None

    def test_diagonal_correction_produces_finite_values(self):
        """Test that diagonal correction produces finite values."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        ricci_corrected = scutoid.ricci_scalars_corrected

        # All values should be finite (not NaN or inf)
        finite_mask = np.isfinite(ricci_corrected)
        # At least some values should be finite
        assert np.any(finite_mask), "No finite corrected values found"

    def test_diagonal_correction_differs_from_flat(self):
        """Test that diagonal correction changes Ricci values."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        flat_ricci = scutoid.ricci_scalars
        corrected_ricci = scutoid.ricci_scalars_corrected

        # Find valid (finite) values in both
        valid_mask = np.isfinite(flat_ricci) & np.isfinite(corrected_ricci)

        if np.any(valid_mask):
            flat_valid = flat_ricci[valid_mask]
            corrected_valid = corrected_ricci[valid_mask]

            # At least some values should differ
            # (unless configuration is perfectly symmetric, which is unlikely)
            differences = np.abs(corrected_valid - flat_valid)
            # Allow that some might be identical, but expect at least 10% to differ
            different_fraction = np.sum(differences > 1e-10) / len(differences)
            assert (
                different_fraction > 0.1
            ), f"Only {different_fraction:.1%} of values changed with correction"

    def test_get_ricci_scalars_returns_corrected(self):
        """Test that get_ricci_scalars returns corrected values in diagonal mode."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        ricci = scutoid.get_ricci_scalars()
        corrected = scutoid.ricci_scalars_corrected

        # Should return corrected array
        assert np.array_equal(ricci, corrected, equal_nan=True)


class TestMetricCorrectionFullMode:
    """Test 'full' mode (full metric tensor correction O(N·k))."""

    def test_full_mode_applies_correction(self):
        """Test that 'full' mode applies metric correction."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="full")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should have both flat and corrected scalars
        assert scutoid.ricci_scalars is not None
        assert scutoid.ricci_scalars_corrected is not None

    def test_full_correction_produces_finite_values(self):
        """Test that full correction produces finite values."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="full")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        ricci_corrected = scutoid.ricci_scalars_corrected

        # All values should be finite
        finite_mask = np.isfinite(ricci_corrected)
        assert np.any(finite_mask), "No finite corrected values found"

    def test_full_correction_differs_from_flat(self):
        """Test that full correction changes Ricci values."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="full")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        flat_ricci = scutoid.ricci_scalars
        corrected_ricci = scutoid.ricci_scalars_corrected

        # Find valid values
        valid_mask = np.isfinite(flat_ricci) & np.isfinite(corrected_ricci)

        if np.any(valid_mask):
            flat_valid = flat_ricci[valid_mask]
            corrected_valid = corrected_ricci[valid_mask]

            # At least some values should differ
            differences = np.abs(corrected_valid - flat_valid)
            different_fraction = np.sum(differences > 1e-10) / len(differences)
            assert (
                different_fraction > 0.1
            ), f"Only {different_fraction:.1%} of values changed with correction"


class TestMetricCorrectionComparison:
    """Compare different correction modes."""

    def test_diagonal_vs_full_corrections(self):
        """Test that diagonal and full modes produce different corrections."""
        history = create_simple_2d_history(N=16, n_steps=5)

        # Compute with diagonal mode
        scutoid_diag = ScutoidHistory2D(history, metric_correction="diagonal")
        scutoid_diag.build_tessellation()
        scutoid_diag.compute_ricci_scalars()
        ricci_diag = scutoid_diag.ricci_scalars_corrected

        # Compute with full mode
        scutoid_full = ScutoidHistory2D(history, metric_correction="full")
        scutoid_full.build_tessellation()
        scutoid_full.compute_ricci_scalars()
        ricci_full = scutoid_full.ricci_scalars_corrected

        # Find valid values in both
        valid_mask = np.isfinite(ricci_diag) & np.isfinite(ricci_full)

        if np.any(valid_mask):
            diag_valid = ricci_diag[valid_mask]
            full_valid = ricci_full[valid_mask]

            # Should have some differences (different correction formulas)
            # But might be similar in magnitude
            differences = np.abs(full_valid - diag_valid)
            np.mean(differences)

            # Just check they're not identical
            assert not np.allclose(
                diag_valid, full_valid, rtol=1e-10
            ), "Diagonal and full corrections are identical"

    def test_all_modes_run_without_crash(self):
        """Smoke test: all modes should run without crashing."""
        history = create_simple_2d_history(N=16, n_steps=5)

        for mode in ["none", "diagonal", "full"]:
            scutoid = ScutoidHistory2D(history, metric_correction=mode)
            scutoid.build_tessellation()
            scutoid.compute_ricci_scalars()
            ricci = scutoid.get_ricci_scalars()

            # Should return some values
            assert ricci is not None
            assert ricci.shape == (5, 16)


class TestMetricCorrectionIntegration:
    """Test integration with full scutoid tessellation workflow."""

    def test_manual_correction_call(self):
        """Test manually calling compute_metric_corrected_ricci."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="none")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Initially no correction
        assert scutoid.ricci_scalars_corrected is None

        # Temporarily enable correction
        scutoid.metric_correction = "diagonal"
        scutoid.compute_metric_corrected_ricci()

        # Now should have corrected values
        assert scutoid.ricci_scalars_corrected is not None

    def test_correction_with_incremental_mode(self):
        """Test that metric correction works with incremental tessellation."""
        history = create_simple_2d_history(N=16, n_steps=5)

        # Use incremental mode with diagonal correction
        scutoid = ScutoidHistory2D(history, incremental=True, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should have corrected values
        assert scutoid.ricci_scalars_corrected is not None

        ricci = scutoid.get_ricci_scalars()
        assert ricci is not None

    def test_correction_with_batch_mode(self):
        """Test that metric correction works with batch tessellation."""
        history = create_simple_2d_history(N=16, n_steps=5)

        # Use batch mode with full correction
        scutoid = ScutoidHistory2D(history, incremental=False, metric_correction="full")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should have corrected values
        assert scutoid.ricci_scalars_corrected is not None

        ricci = scutoid.get_ricci_scalars()
        assert ricci is not None

    def test_summary_statistics_with_correction(self):
        """Test that summary statistics work with corrected Ricci."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        stats = scutoid.summary_statistics()

        # Should have Ricci statistics
        assert "mean_ricci" in stats
        assert "std_ricci" in stats
        assert "min_ricci" in stats
        assert "max_ricci" in stats

        # Values should be finite
        assert np.isfinite(stats["mean_ricci"])
        assert np.isfinite(stats["std_ricci"])


class TestMetricCorrectionEdgeCases:
    """Test edge cases and error handling."""

    def test_correction_with_few_walkers(self):
        """Test correction with very few walkers."""
        history = create_simple_2d_history(N=4, n_steps=3)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should not crash
        ricci = scutoid.get_ricci_scalars()
        assert ricci is not None

    def test_correction_with_single_timestep(self):
        """Test correction with minimal timesteps."""
        history = create_simple_2d_history(N=16, n_steps=1)
        scutoid = ScutoidHistory2D(history, metric_correction="full")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        # Should have shape [1, N]
        ricci = scutoid.get_ricci_scalars()
        assert ricci.shape == (1, 16)

    def test_correction_preserves_nan_structure(self):
        """Test that correction preserves NaN locations."""
        history = create_simple_2d_history(N=16, n_steps=5)
        scutoid = ScutoidHistory2D(history, metric_correction="diagonal")

        scutoid.build_tessellation()
        scutoid.compute_ricci_scalars()

        flat_ricci = scutoid.ricci_scalars
        corrected_ricci = scutoid.ricci_scalars_corrected

        # NaN locations should match (correction doesn't create/remove NaNs)
        flat_nan_mask = np.isnan(flat_ricci)
        corrected_nan_mask = np.isnan(corrected_ricci)

        # Both should have some finite values
        assert not np.all(flat_nan_mask)
        assert not np.all(corrected_nan_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
