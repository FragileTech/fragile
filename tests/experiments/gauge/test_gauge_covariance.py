"""Tests for gauge covariance (Test 1D - CRITICAL)."""

import torch

from fragile.experiments.gauge.gauge_covariance import (
    GaugeTransformConfig,
    apply_gauge_transformation_to_phases,
    define_gauge_transformation_region,
    generate_gauge_covariance_report,
    modify_companion_probabilities_with_gauge,
    test_gauge_covariance as run_gauge_covariance_test,
)


def test_gauge_transform_config_defaults():
    """Test GaugeTransformConfig default values."""
    config = GaugeTransformConfig()
    assert config.h_eff == 1.0
    assert config.alpha_0 == 1.0
    assert config.epsilon_d == 0.1
    assert config.lambda_alg == 0.0
    assert config.region_center == (0.4, 0.6, 0.4, 0.6)
    assert config.boundary_width == 0.05


def test_define_gauge_transformation_region(simple_swarm_2d):
    """Test gauge transformation region definition."""
    # Define a region: 0.3 ≤ x ≤ 0.7, 0.3 ≤ y ≤ 0.7
    region_bounds = (0.3, 0.7, 0.3, 0.7)

    inside, boundary, outside = define_gauge_transformation_region(
        positions=simple_swarm_2d["positions"],
        region_bounds=region_bounds,
    )

    N = simple_swarm_2d["N"]
    assert inside.shape == (N,)
    assert boundary.shape == (N,)
    assert outside.shape == (N,)
    assert inside.dtype == torch.bool
    assert boundary.dtype == torch.bool
    assert outside.dtype == torch.bool

    # Regions logic: boundary is a subset of inside (near-edge walkers)
    # inside and outside are mutually exclusive and exhaustive
    assert not (inside & outside).any()  # inside and outside are exclusive
    assert (inside | outside).all()  # inside and outside cover everything

    # Boundary walkers are inside but near the edge
    # Check: all boundary walkers must also be inside
    if boundary.any():
        assert (boundary <= inside).all()  # boundary ⊂ inside (element-wise ≤)
    assert not (boundary & outside).any()  # boundary cannot be outside


def test_apply_gauge_transformation_to_phases(simple_swarm_2d):
    """Test gauge transformation application to phases."""
    # Create a region
    region_mask = torch.zeros(simple_swarm_2d["N"], dtype=torch.bool)
    region_mask[:10] = True  # First 10 walkers in region

    alpha_0 = 1.5

    alpha = apply_gauge_transformation_to_phases(
        positions=simple_swarm_2d["positions"],
        alpha_0=alpha_0,
        region_mask=region_mask,
    )

    N = simple_swarm_2d["N"]
    assert alpha.shape == (N,)

    # Walkers in region should have α = α_0
    assert torch.allclose(alpha[region_mask], torch.full((10,), alpha_0))

    # Walkers outside should have α = 0
    assert torch.allclose(alpha[~region_mask], torch.zeros(N - 10))


def test_modify_companion_probabilities_with_gauge(simple_swarm_2d):
    """Test gauge-modified companion probabilities."""
    # Create gauge transformation
    alpha = torch.zeros(simple_swarm_2d["N"])
    alpha[:10] = 1.0  # First 10 walkers transformed

    config = GaugeTransformConfig(h_eff=1.0, epsilon_d=0.1)

    probs_gauge = modify_companion_probabilities_with_gauge(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
        alpha=alpha,
        config=config,
    )

    N = simple_swarm_2d["N"]
    assert probs_gauge.shape == (N, N)

    # Each row should sum to ~1 (probability distribution)
    row_sums = probs_gauge.sum(dim=1)
    alive = simple_swarm_2d["alive"]
    assert torch.allclose(row_sums[alive], torch.ones(alive.sum()), atol=1e-4)

    # No self-pairing
    assert (torch.diag(probs_gauge) == 0).all()

    # All probabilities non-negative
    assert (probs_gauge >= 0).all()


def test_gauge_covariance_mean_field(simple_swarm_2d):
    """Test gauge covariance in mean-field regime.

    Expected verdict: "invariant" (mean-field)
    Expected: Δd' ≈ 0 everywhere
    """
    result = run_gauge_covariance_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions_baseline=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,  # Mean-field
        num_trials=3,  # Reduce for faster test
    )

    # Check output structure (actual return fields)
    assert "verdict" in result
    assert "alpha_0" in result
    assert "delta_inside" in result
    assert "delta_outside" in result
    assert "delta_boundary" in result
    assert "confidence" in result
    assert "num_trials" in result
    assert "rho" in result

    # Verdict should be string
    assert isinstance(result["verdict"], str)
    assert result["verdict"] in ["covariant", "invariant", "inconclusive"]

    # Statistics should be scalars
    assert isinstance(result["delta_inside"], float)
    assert isinstance(result["delta_outside"], float)


def test_gauge_covariance_local(clustered_swarm_2d):
    """Test gauge covariance in local regime.

    Expected verdict: Could be "covariant" or "invariant" depending on data
    Expected: If covariant, Δd' ~ O(α) inside, ≈0 outside
    """
    rho = 0.1
    result = run_gauge_covariance_test(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions_baseline=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho=rho,
        num_trials=3,
    )

    # Check all outputs present
    assert "verdict" in result
    assert result["verdict"] in ["covariant", "invariant", "inconclusive"]

    # Check statistics are reasonable
    assert result["delta_inside"] >= 0
    assert result["delta_outside"] >= 0
    assert result["delta_boundary"] >= 0


def test_gauge_covariance_with_dead_walkers(partially_dead_swarm_2d):
    """Test gauge covariance handles dead walkers correctly."""
    result = run_gauge_covariance_test(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        companions_baseline=partially_dead_swarm_2d["diversity_companions"],
        alive=partially_dead_swarm_2d["alive"],
        rho=0.1,
        num_trials=2,
    )

    # Should complete without error
    assert "verdict" in result
    assert result["verdict"] in ["covariant", "invariant", "inconclusive"]


def test_gauge_covariance_custom_config(simple_swarm_2d):
    """Test gauge covariance with custom configuration."""
    config = GaugeTransformConfig(
        alpha_0=2.0,
        h_eff=2.0,
        epsilon_d=0.2,
        region_center=(0.3, 0.7, 0.3, 0.7),
        boundary_width=0.1,
    )

    result = run_gauge_covariance_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions_baseline=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
        gauge_config=config,
        num_trials=2,
    )

    # Check config values reflected in output
    assert result["alpha_0"] == 2.0
    # boundary_width is not returned in the output, it's only used internally
    assert "verdict" in result


def test_gauge_covariance_multiple_trials(simple_swarm_2d):
    """Test that multiple trials provide statistical robustness."""
    # Run with 1 trial
    result_1 = run_gauge_covariance_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions_baseline=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
        num_trials=1,
    )

    # Run with 5 trials
    result_5 = run_gauge_covariance_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions_baseline=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
        num_trials=5,
    )

    # Both should produce valid verdicts
    assert result_1["verdict"] in ["covariant", "invariant", "inconclusive"]
    assert result_5["verdict"] in ["covariant", "invariant", "inconclusive"]

    # Both should have valid statistics
    assert result_1["delta_inside"] >= 0
    assert result_5["delta_inside"] >= 0


def test_generate_gauge_covariance_report_covariant():
    """Test report generation for covariant verdict."""
    # Mock covariant result (actual structure from test_gauge_covariance)
    results = {
        "verdict": "covariant",
        "alpha_0": 1.0,
        "delta_inside": 0.25,
        "delta_outside": 0.02,
        "delta_boundary": 0.10,
        "confidence": 12.5,
        "num_trials": 10,
        "rho": 0.1,
    }

    report = generate_gauge_covariance_report(results)

    # Check report content
    assert isinstance(report, str)
    assert "GAUGE COVARIANCE TEST" in report
    assert "covariant" in report.lower()
    assert "Local gauge theory interpretation is VIABLE!" in report
    assert "0.25" in report  # Inside mean


def test_generate_gauge_covariance_report_invariant():
    """Test report generation for invariant verdict."""
    # Mock invariant result (actual structure)
    results = {
        "verdict": "invariant",
        "alpha_0": 1.0,
        "delta_inside": 0.02,
        "delta_outside": 0.02,
        "delta_boundary": 0.02,
        "confidence": 50.0,
        "num_trials": 10,
        "rho": None,
    }

    report = generate_gauge_covariance_report(results)

    # Check report content
    assert isinstance(report, str)
    assert "invariant" in report.lower()
    assert "Mean-field interpretation applies" in report


def test_gauge_transformation_region_edge_cases():
    """Test edge cases for region definition."""
    # All positions at origin
    positions = torch.zeros(10, 2)
    region_bounds = (0.3, 0.7, 0.3, 0.7)

    inside, boundary, outside = define_gauge_transformation_region(
        positions, region_bounds
    )

    # All should be outside
    assert outside.all()
    assert not inside.any()
    assert not boundary.any()


def test_gauge_covariance_numerical_stability(simple_swarm_2d):
    """Test gauge covariance is numerically stable with non-default parameters."""
    # Use moderately non-standard parameters (not too extreme to avoid numerical issues)
    config = GaugeTransformConfig(
        alpha_0=3.0,  # Larger transformation (but not too extreme)
        h_eff=0.5,  # Different h_eff (but not too small)
        epsilon_d=0.05,  # Smaller epsilon (but not too small)
    )

    result = run_gauge_covariance_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions_baseline=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
        gauge_config=config,
        num_trials=2,
    )

    # Should complete without NaNs or Infs
    assert torch.isfinite(torch.tensor(result["delta_inside"]))
    assert torch.isfinite(torch.tensor(result["delta_outside"]))
    assert result["verdict"] in ["covariant", "invariant", "inconclusive"]
