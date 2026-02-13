"""Tests for gradient estimation using finite differences."""

import pytest
import torch

from fragile.fractalai.scutoid import (
    compute_directional_derivative,
    estimate_gradient_finite_difference,
    estimate_gradient_quality_metrics,
)
from fragile.fractalai.scutoid.neighbors import build_csr_from_coo
from fragile.fractalai.scutoid.weights import compute_edge_weights


@pytest.fixture
def quadratic_setup():
    """Setup for quadratic function V(x) = (1/2) x^T A x."""
    device = torch.device("cpu")

    # Generate grid of positions
    x = torch.linspace(-2, 2, 10)
    y = torch.linspace(-2, 2, 10)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)  # [100, 2]

    # Quadratic: V(x,y) = x² + 2y²
    A = torch.tensor([[2.0, 0.0], [0.0, 4.0]], device=device)
    fitness = 0.5 * (positions @ A * positions).sum(dim=1)

    # True gradient: ∇V = Ax
    grad_true = positions @ A

    # Build k-NN graph
    edge_index = _build_knn_graph(positions, k=8)

    return {
        "positions": positions,
        "fitness": fitness,
        "edge_index": edge_index,
        "grad_true": grad_true,
        "A": A,
    }


def test_gradient_quadratic_uniform_weighting(quadratic_setup):
    """Test gradient estimation with uniform weighting on quadratic function."""
    edge_weights = compute_edge_weights(
        quadratic_setup["positions"],
        quadratic_setup["edge_index"],
        mode="uniform",
    )

    result = estimate_gradient_finite_difference(
        quadratic_setup["positions"],
        quadratic_setup["fitness"],
        quadratic_setup["edge_index"],
        edge_weights=edge_weights,
    )

    grad_estimated = result["gradient"]
    grad_true = quadratic_setup["grad_true"]

    # Check valid estimates
    valid_mask = result["valid_mask"]
    assert valid_mask.sum() >= 90, "Most walkers should have valid gradients"

    # Compute error
    error = torch.norm(grad_estimated[valid_mask] - grad_true[valid_mask], dim=1)
    relative_error = error.mean() / torch.norm(grad_true[valid_mask], dim=1).mean()

    # Uniform weighting: expect ~10-20% error on irregular grids
    assert relative_error < 0.25, f"Gradient error too large: {relative_error:.2e}"


def test_gradient_quadratic_inverse_distance(quadratic_setup):
    """Test gradient estimation with inverse distance weighting."""
    result = estimate_gradient_finite_difference(
        quadratic_setup["positions"],
        quadratic_setup["fitness"],
        quadratic_setup["edge_index"],
    )

    grad_estimated = result["gradient"]
    grad_true = quadratic_setup["grad_true"]
    valid_mask = result["valid_mask"]

    error = torch.norm(grad_estimated[valid_mask] - grad_true[valid_mask], dim=1)
    relative_error = error.mean() / torch.norm(grad_true[valid_mask], dim=1).mean()

    # Finite-difference on irregular grids: expect ~5-10% error
    # Inverse distance weighting should give good accuracy
    assert relative_error < 0.15, f"Gradient error too large: {relative_error:.2e}"


def test_gradient_quadratic_csr_matches_coo(quadratic_setup):
    """CSR-based gradient estimation should match COO results."""
    positions = quadratic_setup["positions"]
    fitness = quadratic_setup["fitness"]
    edge_index = quadratic_setup["edge_index"]

    csr_data = build_csr_from_coo(edge_index, positions.shape[0])

    result_coo = estimate_gradient_finite_difference(positions, fitness, edge_index)
    result_csr = estimate_gradient_finite_difference(
        positions,
        fitness,
        edge_index=None,
        csr_ptr=csr_data["csr_ptr"],
        csr_indices=csr_data["csr_indices"],
    )

    assert torch.equal(result_coo["valid_mask"], result_csr["valid_mask"])
    valid_mask = result_coo["valid_mask"]
    assert torch.allclose(
        result_coo["gradient"][valid_mask],
        result_csr["gradient"][valid_mask],
        atol=1e-5,
        rtol=1e-4,
    )


def test_gradient_rosenbrock():
    """Test gradient on Rosenbrock function V(x,y) = (1-x)² + 100(y-x²)²."""
    # Generate positions
    n_per_dim = 10
    x = torch.linspace(-1, 1.5, n_per_dim)
    y = torch.linspace(-0.5, 2, n_per_dim)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Fitness
    x1, x2 = positions[:, 0], positions[:, 1]
    fitness = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

    # True gradient
    grad_true = torch.zeros_like(positions)
    grad_true[:, 0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    grad_true[:, 1] = 200 * (x2 - x1**2)

    # Build graph
    edge_index = _build_knn_graph(positions, k=8)

    # Estimate
    result = estimate_gradient_finite_difference(positions, fitness, edge_index)

    grad_estimated = result["gradient"]
    valid_mask = result["valid_mask"]

    # Error (Rosenbrock is harder due to nonlinearity, allow larger error)
    error = torch.norm(grad_estimated[valid_mask] - grad_true[valid_mask], dim=1)
    relative_error = error.mean() / (torch.norm(grad_true[valid_mask], dim=1).mean() + 1e-6)

    # Rosenbrock has strong nonlinearity: expect 15-30% error
    assert relative_error < 0.4, f"Rosenbrock gradient error too large: {relative_error:.2e}"


def test_directional_derivative_quadratic(quadratic_setup):
    """Test directional derivative computation."""
    # Direction: along x-axis
    direction = torch.tensor([1.0, 0.0])

    result = compute_directional_derivative(
        quadratic_setup["positions"],
        quadratic_setup["fitness"],
        quadratic_setup["edge_index"],
        direction,
    )

    # True directional derivative: ∇V · [1, 0] = 2x
    dd_true = 2 * quadratic_setup["positions"][:, 0]

    # Compare
    valid_mask = torch.isfinite(result)
    error = (result[valid_mask] - dd_true[valid_mask]).abs()
    relative_error = error.mean() / (dd_true[valid_mask].abs().mean() + 1e-6)

    # Directional derivatives use simple formula (less accurate than weighted LS)
    assert relative_error < 1.0, f"Directional derivative error: {relative_error:.2e}"


def test_directional_derivative_csr_matches_coo(quadratic_setup):
    """CSR-based directional derivative should match COO results."""
    positions = quadratic_setup["positions"]
    fitness = quadratic_setup["fitness"]
    edge_index = quadratic_setup["edge_index"]
    direction = torch.tensor([1.0, 0.0])

    csr_data = build_csr_from_coo(edge_index, positions.shape[0])

    result_coo = compute_directional_derivative(
        positions,
        fitness,
        edge_index,
        direction,
    )
    result_csr = compute_directional_derivative(
        positions,
        fitness,
        edge_index=None,
        direction=direction,
        csr_ptr=csr_data["csr_ptr"],
        csr_indices=csr_data["csr_indices"],
    )

    valid_mask = torch.isfinite(result_coo) & torch.isfinite(result_csr)
    assert torch.allclose(
        result_coo[valid_mask],
        result_csr[valid_mask],
        atol=1e-5,
        rtol=1e-4,
    )


def test_gradient_quality_metrics(quadratic_setup):
    """Test quality metrics computation."""
    result = estimate_gradient_finite_difference(
        quadratic_setup["positions"],
        quadratic_setup["fitness"],
        quadratic_setup["edge_index"],
    )

    metrics = estimate_gradient_quality_metrics(
        result,
        quadratic_setup["positions"],
        quadratic_setup["fitness"],
        quadratic_setup["edge_index"],
    )

    # Check metrics exist
    assert "valid_fraction" in metrics
    assert "mean_neighbors" in metrics
    assert "mean_quality" in metrics
    assert "gradient_norm_mean" in metrics

    # Sanity checks
    assert metrics["valid_fraction"] > 0.8, "Should have mostly valid estimates"
    assert metrics["mean_neighbors"] > 5, "Should have reasonable neighbor counts"
    assert metrics["gradient_norm_mean"] > 0, "Gradient should be non-zero"


def test_gradient_with_alive_mask(quadratic_setup):
    """Test gradient estimation with alive mask."""
    N = quadratic_setup["positions"].shape[0]

    # Mark some walkers as dead
    alive = torch.ones(N, dtype=torch.bool)
    alive[::3] = False  # Every 3rd walker is dead

    result = estimate_gradient_finite_difference(
        quadratic_setup["positions"],
        quadratic_setup["fitness"],
        quadratic_setup["edge_index"],
        alive=alive,
    )

    grad_estimated = result["gradient"]

    # Dead walkers should have NaN gradients
    assert torch.isnan(grad_estimated[~alive]).all(), "Dead walkers should have NaN"

    # Alive walkers should have finite gradients
    valid_alive = result["valid_mask"] & alive
    assert torch.isfinite(grad_estimated[valid_alive]).all(), "Alive should be finite"


def test_convergence_with_neighbors(quadratic_setup):
    """Test that gradient estimation works with varying neighbor counts."""
    errors = []
    k_values = [5, 10, 20, 40]

    for k in k_values:
        edge_index = _build_knn_graph(quadratic_setup["positions"], k=k)

        result = estimate_gradient_finite_difference(
            quadratic_setup["positions"],
            quadratic_setup["fitness"],
            edge_index,
        )

        grad_estimated = result["gradient"]
        valid_mask = result["valid_mask"]

        error = torch.norm(
            grad_estimated[valid_mask] - quadratic_setup["grad_true"][valid_mask], dim=1
        ).mean()

        errors.append(error.item())

    # All estimates should be reasonable (not just increasing)
    # Note: More neighbors doesn't always mean better for weighted LS due to ill-conditioning
    assert all(e < 5.0 for e in errors), f"Errors too large: {errors}"
    assert errors[1] < 1.0, "Error with k=10 should be reasonable"


def test_gradient_handles_nan_fitness():
    """Test that NaN fitness values are handled gracefully."""
    N = 50
    positions = torch.randn(N, 2)
    fitness = torch.randn(N)

    # Introduce some NaN values
    fitness[::5] = float("nan")

    edge_index = _build_knn_graph(positions, k=8)

    result = estimate_gradient_finite_difference(
        positions, fitness, edge_index, validate_inputs=True
    )

    # Should still return results for valid walkers
    assert result["valid_mask"].any(), "Should have some valid walkers"

    # Walkers with NaN fitness should be invalid
    has_nan = torch.isnan(fitness)
    assert not result["valid_mask"][has_nan].any(), "NaN fitness walkers should be invalid"


# ============================================================================
# Helper functions
# ============================================================================


def _build_knn_graph(positions: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-nearest neighbor graph."""
    N = positions.shape[0]
    dist_matrix = torch.cdist(positions, positions)

    # Find k nearest (excluding self)
    _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    indices = indices[:, 1:]  # Exclude self

    # Build edge list
    src = torch.arange(N).unsqueeze(1).expand(-1, k).flatten()
    dst = indices.flatten()

    return torch.stack([src, dst], dim=0)
