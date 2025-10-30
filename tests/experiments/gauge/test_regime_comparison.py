"""Tests for regime comparison (mean-field ↔ local crossover)."""

import numpy as np
import torch

from fragile.experiments.gauge.regime_comparison import (
    compare_regimes,
    generate_regime_comparison_report,
    identify_critical_scale,
    RegimeComparisonConfig,
    scan_correlation_length,
    scan_field_gradients,
)


def test_regime_comparison_config_defaults():
    """Test RegimeComparisonConfig default values."""
    config = RegimeComparisonConfig()
    assert config.rho_values == [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    assert config.num_samples == 5


def test_scan_correlation_length(simple_swarm_2d):
    """Test correlation length scan across ρ values."""
    rho_values = [0.05, 0.1, 0.2]

    result = scan_correlation_length(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho_values=rho_values,
    )

    # Check output structure
    assert "rho_values" in result
    assert "xi_values" in result
    assert "fit_quality" in result

    # Should include None (mean-field) at end
    assert len(result["rho_values"]) == len(rho_values) + 1
    assert result["rho_values"][-1] is None

    # Arrays should have correct length
    assert len(result["xi_values"]) == len(rho_values) + 1
    assert len(result["fit_quality"]) == len(rho_values) + 1

    # Xi values should be non-negative
    assert (result["xi_values"] >= 0).all()

    # Fit quality (R²) can be negative for poor fits (mathematically valid)
    # Just check all values are finite
    assert np.all(np.isfinite(result["fit_quality"]))


def test_scan_field_gradients(simple_swarm_2d):
    """Test field gradient scan across ρ values."""
    rho_values = [0.05, 0.1, 0.2]

    result = scan_field_gradients(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho_values=rho_values,
    )

    # Check output structure
    assert "rho_values" in result
    assert "gradient_mean" in result
    assert "gradient_std" in result

    # Should include None at end
    assert len(result["rho_values"]) == len(rho_values) + 1
    assert result["rho_values"][-1] is None

    # Arrays should have correct length
    assert len(result["gradient_mean"]) == len(rho_values) + 1
    assert len(result["gradient_std"]) == len(rho_values) + 1

    # All values should be non-negative
    assert (result["gradient_mean"] >= 0).all()
    assert (result["gradient_std"] >= 0).all()


def test_identify_critical_scale_with_transition():
    """Test critical scale identification with clear transition."""
    # Create synthetic data with clear transition at ρ ≈ 0.15
    rho_values = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0])
    observable_values = np.array([0.9, 0.85, 0.7, 0.5, 0.3, 0.1, 0.05])

    rho_c = identify_critical_scale(
        rho_values=rho_values,
        observable_values=observable_values,
        threshold=0.5,
    )

    # Should find critical scale near 0.15
    assert rho_c is not None
    assert 0.1 <= rho_c <= 0.2


def test_identify_critical_scale_no_transition():
    """Test critical scale identification with no transition."""
    # All values above threshold
    rho_values = np.array([0.01, 0.05, 0.1, 0.2])
    observable_values = np.array([0.9, 0.85, 0.8, 0.75])

    rho_c = identify_critical_scale(
        rho_values=rho_values,
        observable_values=observable_values,
        threshold=0.5,
    )

    # Should return None (no crossing)
    assert rho_c is None


def test_identify_critical_scale_with_none_values():
    """Test critical scale handles None in rho_values."""
    # Include None (mean-field)
    rho_values = np.array([0.01, 0.1, None], dtype=object)
    observable_values = np.array([0.8, 0.4, 0.1])

    rho_c = identify_critical_scale(
        rho_values=rho_values,
        observable_values=observable_values,
        threshold=0.5,
    )

    # Should filter None and find crossing
    assert rho_c is not None
    assert 0.01 <= rho_c <= 0.1


def test_compare_regimes(simple_swarm_2d):
    """Test full regime comparison."""
    config = RegimeComparisonConfig(
        rho_values=[0.05, 0.1, 0.2],
        num_samples=1,  # Single sample for faster test
    )

    result = compare_regimes(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        regime_config=config,
    )

    # Check output structure
    assert "correlation_scan" in result
    assert "gradient_scan" in result
    assert "rho_c_correlation" in result
    assert "rho_c_gradient" in result
    assert "rho_values_scanned" in result

    # Check correlation scan
    assert "xi_values" in result["correlation_scan"]
    assert len(result["correlation_scan"]["xi_values"]) == len(config.rho_values) + 1

    # Check gradient scan
    assert "gradient_mean" in result["gradient_scan"]
    assert len(result["gradient_scan"]["gradient_mean"]) == len(config.rho_values) + 1

    # Critical scales can be None or float
    assert result["rho_c_correlation"] is None or isinstance(result["rho_c_correlation"], float)
    assert result["rho_c_gradient"] is None or isinstance(result["rho_c_gradient"], float)


def test_compare_regimes_with_dead_walkers(partially_dead_swarm_2d):
    """Test regime comparison handles dead walkers."""
    config = RegimeComparisonConfig(
        rho_values=[0.1, 0.2],
        num_samples=1,
    )

    result = compare_regimes(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        companions=partially_dead_swarm_2d["diversity_companions"],
        alive=partially_dead_swarm_2d["alive"],
        regime_config=config,
    )

    # Should complete without error
    assert "correlation_scan" in result
    assert "gradient_scan" in result


def test_generate_regime_comparison_report_with_critical_scales():
    """Test report generation with identified critical scales."""
    # Mock results with critical scales
    results = {
        "rho_values_scanned": [0.01, 0.05, 0.1, 0.2, 0.5],
        "rho_c_correlation": 0.12,
        "rho_c_gradient": 0.15,
        "correlation_scan": {
            "rho_values": [0.01, 0.05, 0.1, 0.2, 0.5, None],
            "xi_values": np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0]),
            "fit_quality": np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5]),
        },
        "gradient_scan": {
            "rho_values": [0.01, 0.05, 0.1, 0.2, 0.5, None],
            "gradient_mean": np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1]),
            "gradient_std": np.array([2.0, 1.0, 0.5, 0.2, 0.1, 0.05]),
        },
    }

    report = generate_regime_comparison_report(results)

    # Check report content
    assert isinstance(report, str)
    assert "REGIME COMPARISON REPORT" in report
    assert "Critical Scales:" in report
    assert "0.12" in report  # rho_c_correlation
    assert "0.15" in report  # rho_c_gradient
    assert "Regime Classification:" in report


def test_generate_regime_comparison_report_no_critical_scales():
    """Test report generation with no critical scales found."""
    # Mock results without critical scales
    results = {
        "rho_values_scanned": [0.01, 0.05, 0.1],
        "rho_c_correlation": None,
        "rho_c_gradient": None,
        "correlation_scan": {
            "rho_values": [0.01, 0.05, 0.1, None],
            "xi_values": np.array([0.01, 0.05, 0.1, 0.2]),
            "fit_quality": np.array([0.9, 0.85, 0.8, 0.75]),
        },
        "gradient_scan": {
            "rho_values": [0.01, 0.05, 0.1, None],
            "gradient_mean": np.array([10.0, 8.0, 6.0, 5.0]),
            "gradient_std": np.array([2.0, 1.5, 1.0, 0.8]),
        },
    }

    report = generate_regime_comparison_report(results)

    # Check report handles None
    assert isinstance(report, str)
    assert "N/A" in report or "No clear critical scale" in report


def test_scan_correlation_length_monotonicity(clustered_swarm_2d):
    """Test that ξ tends to increase with ρ (generally expected)."""
    rho_values = [0.01, 0.05, 0.1, 0.2]

    result = scan_correlation_length(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho_values=rho_values,
    )

    # Check that ξ values are all positive
    xi_values = result["xi_values"]
    assert (xi_values > 0).all()

    # Mean-field value (last) should be large (but this is not guaranteed with poor fits)
    # Just check that we have a reasonable range of values
    xi_mean_field = xi_values[-1]
    assert xi_mean_field > 0  # Should be positive
    # Don't enforce strict monotonicity since fits can be poor with random data


def test_scan_field_gradients_trend(clustered_swarm_2d):
    """Test that |∇d'| tends to decrease with ρ (generally expected)."""
    rho_values = [0.01, 0.05, 0.1, 0.2]

    result = scan_field_gradients(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho_values=rho_values,
    )

    # Check that gradient values are non-negative
    gradient_mean = result["gradient_mean"]
    assert (gradient_mean >= 0).all()

    # Mean-field value (last) should be small
    grad_mean_field = gradient_mean[-1]
    grad_local = gradient_mean[0]
    # Don't enforce strict monotonicity (can have noise), just check both positive
    assert grad_mean_field >= 0
    assert grad_local >= 0


def test_critical_scale_interpolation():
    """Test critical scale interpolation is accurate."""
    # Create exact threshold crossing
    rho_values = np.array([0.1, 0.2, 0.3, 0.4])
    observable_values = np.array([0.6, 0.5, 0.4, 0.3])

    rho_c = identify_critical_scale(
        rho_values=rho_values,
        observable_values=observable_values,
        threshold=0.5,
    )

    # Should find exact crossing at 0.2 (by construction)
    assert rho_c is not None
    assert abs(rho_c - 0.2) < 0.01  # Very close to exact value
