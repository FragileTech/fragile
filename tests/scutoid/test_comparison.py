"""Tests for cross-validation between FD and geometric methods."""

import pytest
import torch

from fragile.fractalai.scutoid import (
    compare_estimation_methods,
    validate_on_synthetic_function,
)


def test_compare_methods_basic():
    """Test basic comparison of estimation methods."""
    # Simple quadratic setup
    N = 100
    d = 2
    positions = torch.randn(N, d) * 0.5  # Keep compact for good neighbors

    # Quadratic fitness
    fitness = (positions**2).sum(dim=1)

    # Build graph
    edge_index = _build_knn_graph(positions, k=10)

    # Compare methods
    result = compare_estimation_methods(
        positions,
        fitness,
        edge_index,
        epsilon_sigma=0.1,
        compute_full_hessian=True,
        return_detailed=False,
    )

    # Check all outputs present
    assert "gradient_fd" in result
    assert "hessian_diagonal_fd" in result
    assert "hessian_full_fd" in result
    assert "hessian_geometric" in result
    assert "comparison_metrics" in result

    # Check metrics
    metrics = result["comparison_metrics"]
    assert "eigenvalue_correlation" in metrics
    assert "frobenius_agreement" in metrics
    assert "element_wise_rmse" in metrics
    assert "method_preference" in metrics


def test_compare_methods_quadratic_agreement():
    """Test that FD and geometric methods agree well on quadratic."""

    # Grid for better structure
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Quadratic: V = x² + 2y²
    fitness = positions[:, 0] ** 2 + 2 * positions[:, 1] ** 2

    edge_index = _build_knn_graph(positions, k=12)

    result = compare_estimation_methods(
        positions,
        fitness,
        edge_index,
        epsilon_sigma=0.1,
        compute_full_hessian=True,
    )

    metrics = result["comparison_metrics"]

    # For simple quadratic, methods should agree reasonably
    # (may not be perfect due to non-equilibrium distribution)
    assert metrics["eigenvalue_correlation"] > 0.3, (
        f"Correlation too low: {metrics['eigenvalue_correlation']}"
    )

    assert 0 <= metrics["frobenius_agreement"] <= 1, "Frobenius agreement should be in [0, 1]"


def test_compare_methods_detailed_output():
    """Test detailed per-walker comparison output."""
    N = 50
    d = 2
    positions = torch.randn(N, d) * 0.5
    fitness = (positions**2).sum(dim=1)
    edge_index = _build_knn_graph(positions, k=10)

    result = compare_estimation_methods(
        positions,
        fitness,
        edge_index,
        epsilon_sigma=0.1,
        compute_full_hessian=False,  # Just diagonal
        return_detailed=True,
    )

    # Should have per-walker comparison
    assert "per_walker_comparison" in result

    per_walker = result["per_walker_comparison"]
    assert "eigenvalue_difference" in per_walker
    assert "equilibrium_score" in per_walker

    # Check shapes
    assert per_walker["eigenvalue_difference"].shape == (N,)
    assert per_walker["equilibrium_score"].shape == (N,)


def test_validate_quadratic():
    """Test validation on quadratic function."""
    errors = validate_on_synthetic_function(
        test_function="quadratic",
        n_walkers=100,
        dimensionality=2,
        epsilon_sigma=0.1,
        k_neighbors=10,
    )

    # Check all error metrics present
    assert "gradient_rmse" in errors
    assert "gradient_max_error" in errors
    assert "hessian_frobenius_error" in errors
    assert "hessian_eigenvalue_error" in errors
    assert "geometric_hessian_error" in errors
    assert "passed" in errors

    # Quadratic should have low errors
    assert errors["gradient_rmse"] < 0.01, f"Gradient RMSE too high: {errors['gradient_rmse']}"

    # Hessian may have more error due to second-order FD
    assert errors["hessian_frobenius_error"] < 0.1, (
        f"Hessian error too high: {errors['hessian_frobenius_error']}"
    )


def test_validate_rosenbrock():
    """Test validation on Rosenbrock function."""
    errors = validate_on_synthetic_function(
        test_function="rosenbrock",
        n_walkers=64,  # 8x8 grid
        dimensionality=2,  # Rosenbrock is 2D
        epsilon_sigma=0.1,
        k_neighbors=10,
    )

    # Rosenbrock is harder - allow larger errors
    assert errors["gradient_rmse"] < 0.05, f"Rosenbrock gradient RMSE: {errors['gradient_rmse']}"

    # Just check it completes without crashing
    assert "passed" in errors


def test_validate_rastrigin():
    """Test validation on Rastrigin function (multimodal)."""
    errors = validate_on_synthetic_function(
        test_function="rastrigin",
        n_walkers=100,
        dimensionality=2,
        epsilon_sigma=0.1,
        k_neighbors=10,
    )

    # Rastrigin is very hard - just check it runs
    assert "gradient_rmse" in errors
    assert "hessian_frobenius_error" in errors

    # Errors may be large due to multimodality
    # Just verify non-infinite
    assert torch.isfinite(torch.tensor(errors["gradient_rmse"]))
    assert torch.isfinite(torch.tensor(errors["hessian_frobenius_error"]))


def test_validate_higher_dimensions():
    """Test validation in higher dimensions."""
    errors = validate_on_synthetic_function(
        test_function="quadratic",
        n_walkers=200,
        dimensionality=5,
        epsilon_sigma=0.1,
        k_neighbors=15,
    )

    # Should work in higher dimensions
    assert errors["gradient_rmse"] < 0.02
    assert errors["hessian_frobenius_error"] < 0.15


def test_comparison_metrics_ranges():
    """Test that comparison metrics are in valid ranges."""
    N = 80
    d = 2
    positions = torch.randn(N, d) * 0.5
    fitness = (positions**2).sum(dim=1)
    edge_index = _build_knn_graph(positions, k=10)

    result = compare_estimation_methods(positions, fitness, edge_index, epsilon_sigma=0.1)

    metrics = result["comparison_metrics"]

    # Correlation: should be in [-1, 1], but typically positive
    assert -1 <= metrics["eigenvalue_correlation"] <= 1

    # Frobenius agreement: in [0, 1] by construction
    assert 0 <= metrics["frobenius_agreement"] <= 1

    # RMSE: non-negative
    assert metrics["element_wise_rmse"] >= 0

    # Equilibrium score: in [0, 1]
    assert 0 <= metrics["mean_equilibrium_score"] <= 1

    # Method preference: one of the expected values
    assert metrics["method_preference"] in {"both_agree", "fd", "geometric", "ambiguous"}


def test_comparison_with_different_epsilon():
    """Test that different epsilon_sigma affects geometric method."""
    N = 50
    d = 2
    positions = torch.randn(N, d) * 0.5
    fitness = (positions**2).sum(dim=1)
    edge_index = _build_knn_graph(positions, k=10)

    epsilon_values = [0.05, 0.2]
    results = []

    for eps in epsilon_values:
        result = compare_estimation_methods(
            positions,
            fitness,
            edge_index,
            epsilon_sigma=eps,
            compute_full_hessian=False,
        )
        results.append(result)

    # Geometric Hessians should be different
    H_geo_1 = results[0]["hessian_geometric"]
    H_geo_2 = results[1]["hessian_geometric"]

    diff = torch.norm(H_geo_1 - H_geo_2, p="fro", dim=(1, 2)).mean()

    assert diff > 0.01, "Different epsilon should give different geometric Hessians"

    # But FD Hessians should be similar (epsilon doesn't affect FD)
    H_fd_1 = results[0]["hessian_diagonal_fd"]
    H_fd_2 = results[1]["hessian_diagonal_fd"]

    valid_mask = torch.isfinite(H_fd_1).all(dim=1) & torch.isfinite(H_fd_2).all(dim=1)
    if valid_mask.sum() > 0:
        fd_diff = (H_fd_1[valid_mask] - H_fd_2[valid_mask]).abs().mean()
        # Should be very similar (small numerical differences allowed)
        assert fd_diff < 0.5, "FD Hessian should not depend on epsilon_sigma"


def test_comparison_diagonal_only():
    """Test comparison with diagonal Hessian only."""
    N = 50
    d = 2
    positions = torch.randn(N, d) * 0.5
    fitness = (positions**2).sum(dim=1)
    edge_index = _build_knn_graph(positions, k=10)

    result = compare_estimation_methods(
        positions,
        fitness,
        edge_index,
        epsilon_sigma=0.1,
        compute_full_hessian=False,  # Diagonal only
    )

    # Should not have full Hessian
    assert "hessian_full_fd" not in result

    # Should have diagonal
    assert "hessian_diagonal_fd" in result

    # Should still have geometric (full)
    assert "hessian_geometric" in result

    # Metrics should still be computed
    assert "comparison_metrics" in result


def test_method_preference_logic():
    """Test that method preference is assigned sensibly."""

    # Case 1: Good agreement (quadratic on grid)
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)
    fitness = (positions**2).sum(dim=1)
    edge_index = _build_knn_graph(positions, k=12)

    result = compare_estimation_methods(positions, fitness, edge_index, epsilon_sigma=0.1)

    preference = result["comparison_metrics"]["method_preference"]

    # Should have some preference (not just error)
    assert preference in {"both_agree", "fd", "geometric", "ambiguous"}


def test_validation_passes_on_simple_case():
    """Test that validation passes on simple cases."""
    errors = validate_on_synthetic_function(
        test_function="quadratic",
        n_walkers=100,
        dimensionality=2,
        epsilon_sigma=0.1,
    )

    # Simple quadratic should pass
    assert errors["passed"], "Quadratic validation should pass"


def test_handles_all_nan_gracefully():
    """Test comparison handles edge cases with invalid data."""
    N = 50
    d = 2
    positions = torch.randn(N, d)

    # All NaN fitness
    fitness = torch.full((N,), float("nan"))

    edge_index = _build_knn_graph(positions, k=10)

    # Should not crash
    result = compare_estimation_methods(
        positions,
        fitness,
        edge_index,
        epsilon_sigma=0.1,
        compute_full_hessian=False,
    )

    # Should return some results (likely all invalid)
    assert "comparison_metrics" in result

    # Metrics should indicate no agreement
    metrics = result["comparison_metrics"]
    assert metrics["method_preference"] == "ambiguous"


# ============================================================================
# Helper functions
# ============================================================================


def _build_knn_graph(positions: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-nearest neighbor graph."""
    N = positions.shape[0]
    dist_matrix = torch.cdist(positions, positions)

    _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    indices = indices[:, 1:]

    src = torch.arange(N).unsqueeze(1).expand(-1, k).flatten()
    dst = indices.flatten()

    return torch.stack([src, dst], dim=0)
