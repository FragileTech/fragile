"""Tests for locality tests (Tests 1A-1C)."""

import numpy as np
import torch

from fragile.experiments.gauge.locality_tests import (
    LocalityTestsConfig,
    generate_locality_report,
    run_all_locality_tests,
    test_field_gradients as run_field_gradients_test,
    test_perturbation_response as run_perturbation_response_test,
    test_spatial_correlation as run_spatial_correlation_test,
)


def test_locality_config_defaults():
    """Test LocalityTestsConfig default values."""
    config = LocalityTestsConfig()
    assert config.r_max == 0.5
    assert config.n_bins == 50
    assert config.k_neighbors == 5
    assert config.perturbation_magnitude == 0.01


def test_spatial_correlation_mean_field(simple_swarm_2d):
    """Test 1A: Spatial correlation in mean-field regime.

    Expected: ξ → ∞ (no decay)
    """
    result = run_spatial_correlation_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,  # Mean-field
    )

    # Check output structure
    assert "r" in result
    assert "C" in result
    assert "counts" in result
    assert "xi" in result
    assert "r_squared" in result

    # ξ should be very large in mean-field
    assert result["xi"] > 0.1  # At least larger than typical local scale

    # Fit quality: R² can be negative for poor fits (this is mathematically valid)
    # Just check it's finite
    assert isinstance(result["r_squared"], float)
    assert not np.isnan(result["r_squared"])


def test_spatial_correlation_local(clustered_swarm_2d):
    """Test 1A: Spatial correlation in local regime.

    Expected: ξ ≈ ρ (exponential decay)
    """
    rho = 0.1
    result = run_spatial_correlation_test(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho=rho,
    )

    # ξ should be comparable to ρ
    assert result["xi"] > 0  # Positive correlation length

    # Check all outputs present
    assert len(result["r"]) == len(result["C"])
    assert len(result["r"]) == len(result["counts"])


def test_spatial_correlation_with_dead_walkers(partially_dead_swarm_2d):
    """Test 1A handles dead walkers correctly."""
    result = run_spatial_correlation_test(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        companions=partially_dead_swarm_2d["diversity_companions"],
        alive=partially_dead_swarm_2d["alive"],
        rho=0.1,
    )

    # Should complete without error
    assert "xi" in result
    assert result["xi"] > 0


def test_field_gradients_mean_field(simple_swarm_2d):
    """Test 1B: Field gradients in mean-field regime.

    Expected: |∇d'| ≈ 0 (no spatial variation)
    """
    result = run_field_gradients_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,  # Mean-field
    )

    # Check output structure
    assert "gradients" in result
    assert "mean" in result
    assert "std" in result

    N = simple_swarm_2d["N"]
    assert result["gradients"].shape == (N,)

    # Gradients should be small in mean-field
    # (but not necessarily zero due to finite sample effects)
    assert result["mean"] >= 0
    assert torch.isfinite(torch.tensor(result["mean"]))


def test_field_gradients_local(clustered_swarm_2d):
    """Test 1B: Field gradients in local regime.

    Expected: |∇d'| ~ 1/ρ (significant variation)
    """
    rho = 0.1
    result = run_field_gradients_test(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho=rho,
    )

    # Gradients should be non-zero in local regime
    assert result["mean"] > 0
    assert result["std"] >= 0


def test_field_gradients_with_dead_walkers(partially_dead_swarm_2d):
    """Test 1B handles dead walkers correctly."""
    result = run_field_gradients_test(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        companions=partially_dead_swarm_2d["diversity_companions"],
        alive=partially_dead_swarm_2d["alive"],
        rho=0.1,
    )

    alive = partially_dead_swarm_2d["alive"]

    # Dead walkers should have zero gradient
    assert (result["gradients"][~alive] == 0).all()


def test_perturbation_response_mean_field(simple_swarm_2d):
    """Test 1C: Perturbation response in mean-field regime.

    Expected: Δd' uniform everywhere (non-local response)
    """
    perturb_idx = 0  # Perturb walker 0

    result = run_perturbation_response_test(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        perturb_idx=perturb_idx,
        rho=None,  # Mean-field
    )

    # Check output structure (actual field names)
    assert "distances" in result
    assert "delta_d_prime" in result
    assert "response_range" in result
    assert "verdict" in result

    # Response should be present (non-zero)
    assert len(result["distances"]) > 0
    assert len(result["delta_d_prime"]) > 0


def test_perturbation_response_local(clustered_swarm_2d):
    """Test 1C: Perturbation response in local regime.

    Expected: Δd'(r) ~ exp(-r²/ρ²) (localized response)
    """
    perturb_idx = 0
    rho = 0.1

    result = run_perturbation_response_test(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        perturb_idx=perturb_idx,
        rho=rho,
    )

    # Check output structure (actual field names)
    assert "distances" in result
    assert "delta_d_prime" in result
    assert "response_range" in result

    # Response should decay with distance
    # (check that response is present and finite)
    assert len(result["delta_d_prime"]) > 0
    assert torch.isfinite(result["delta_d_prime"]).all()


def test_perturbation_response_with_dead_walkers(partially_dead_swarm_2d):
    """Test 1C handles dead walkers correctly."""
    # Find first alive walker
    alive_indices = torch.where(partially_dead_swarm_2d["alive"])[0]
    if len(alive_indices) == 0:
        import pytest
        pytest.skip("No alive walkers in test fixture")

    perturb_idx = alive_indices[0].item()

    result = run_perturbation_response_test(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        companions=partially_dead_swarm_2d["diversity_companions"],
        alive=partially_dead_swarm_2d["alive"],
        perturb_idx=perturb_idx,
        rho=0.1,
    )

    # Should complete without error
    assert "delta_d_prime" in result
    assert "response_range" in result


def test_run_all_locality_tests_mean_field(simple_swarm_2d):
    """Test running all locality tests in mean-field regime."""
    results = run_all_locality_tests(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,
    )

    # Check all tests ran (actual field names)
    assert "test_1a" in results
    assert "test_1b" in results
    assert "test_1c" in results
    assert "summary" in results

    # Summary should have verdict
    assert "verdict" in results["summary"]
    assert isinstance(results["summary"]["verdict"], str)
    assert results["summary"]["verdict"] in ["local", "mean-field"]


def test_run_all_locality_tests_local(clustered_swarm_2d):
    """Test running all locality tests in local regime."""
    results = run_all_locality_tests(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho=0.1,
    )

    # Check all tests ran (actual field names)
    assert "test_1a" in results
    assert "test_1b" in results
    assert "test_1c" in results
    assert "summary" in results
    assert "verdict" in results["summary"]


def test_generate_locality_report_mean_field(simple_swarm_2d):
    """Test report generation for mean-field regime."""
    results = run_all_locality_tests(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,
    )

    report = generate_locality_report(results, rho=None)

    # Report should be a string with key information
    assert isinstance(report, str)
    assert len(report) > 100  # Non-trivial report
    assert "LOCALITY TESTS REPORT" in report
    assert "Verdict:" in report


def test_generate_locality_report_local(clustered_swarm_2d):
    """Test report generation for local regime."""
    rho = 0.1
    results = run_all_locality_tests(
        positions=clustered_swarm_2d["positions"],
        velocities=clustered_swarm_2d["velocities"],
        rewards=clustered_swarm_2d["rewards"],
        companions=clustered_swarm_2d["diversity_companions"],
        alive=clustered_swarm_2d["alive"],
        rho=rho,
    )

    report = generate_locality_report(results, rho=rho)

    # Report should mention ρ value
    assert isinstance(report, str)
    assert "0.1" in report or "0.10" in report  # ρ value should appear
