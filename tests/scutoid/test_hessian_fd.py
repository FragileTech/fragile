"""Tests for Hessian estimation using finite differences."""

import pytest
import torch

from fragile.fractalai.scutoid import (
    estimate_gradient_finite_difference,
    estimate_hessian_diagonal_fd,
    estimate_hessian_full_fd,
)
from fragile.fractalai.scutoid.neighbors import build_csr_from_coo


@pytest.fixture
def quadratic_2d_setup():
    """Setup for 2D quadratic function."""
    N = 100
    torch.device("cpu")

    # Grid positions
    x = torch.linspace(-2, 2, 10)
    y = torch.linspace(-2, 2, 10)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Quadratic: V(x,y) = x² + 2y²
    # Hessian: H = [[2, 0], [0, 4]]
    fitness = positions[:, 0] ** 2 + 2 * positions[:, 1] ** 2

    # True Hessian (constant)
    H_true = torch.zeros(N, 2, 2)
    H_true[:, 0, 0] = 2.0
    H_true[:, 1, 1] = 4.0

    edge_index = _build_knn_graph(positions, k=10)

    return {
        "positions": positions,
        "fitness": fitness,
        "edge_index": edge_index,
        "H_true": H_true,
    }


def test_hessian_diagonal_quadratic(quadratic_2d_setup):
    """Test diagonal Hessian estimation on quadratic function."""
    result = estimate_hessian_diagonal_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    H_diag = result["hessian_diagonal"]
    H_true_diag = torch.tensor([[2.0, 4.0]]).expand(100, 2)

    valid_mask = result["valid_mask"]
    assert valid_mask.sum() >= 80, "Most walkers should have valid Hessian"

    # Compute error
    error = (H_diag[valid_mask] - H_true_diag[valid_mask]).abs()
    relative_error = error.mean() / H_true_diag[valid_mask].abs().mean()

    assert relative_error < 0.1, f"Diagonal Hessian error too large: {relative_error:.2e}"


def test_hessian_diagonal_csr_matches_coo(quadratic_2d_setup):
    """CSR-based diagonal Hessian should match COO results."""
    positions = quadratic_2d_setup["positions"]
    fitness = quadratic_2d_setup["fitness"]
    edge_index = quadratic_2d_setup["edge_index"]

    csr_data = build_csr_from_coo(edge_index, positions.shape[0])

    result_coo = estimate_hessian_diagonal_fd(positions, fitness, edge_index)
    result_csr = estimate_hessian_diagonal_fd(
        positions,
        fitness,
        edge_index=None,
        csr_ptr=csr_data["csr_ptr"],
        csr_indices=csr_data["csr_indices"],
    )

    assert torch.equal(result_coo["valid_mask"], result_csr["valid_mask"])
    valid_mask = result_coo["valid_mask"]
    if valid_mask.any():
        assert torch.allclose(
            result_coo["hessian_diagonal"][valid_mask],
            result_csr["hessian_diagonal"][valid_mask],
            atol=1e-5,
            rtol=1e-4,
        )


def test_hessian_full_quadratic_central(quadratic_2d_setup):
    """Test full Hessian with central difference method."""
    # First estimate gradient
    grad_result = estimate_gradient_finite_difference(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    # Then Hessian
    result = estimate_hessian_full_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        grad_result["gradient"],
        quadratic_2d_setup["edge_index"],
        method="central",
        symmetrize=True,
    )

    H_estimated = result["hessian_tensors"]
    H_true = quadratic_2d_setup["H_true"]

    # Check diagonal elements (easier to estimate)
    diag_estimated = torch.diagonal(H_estimated, dim1=1, dim2=2)
    diag_true = torch.diagonal(H_true, dim1=1, dim2=2)

    valid_mask = torch.isfinite(diag_estimated).all(dim=1)

    if valid_mask.sum() > 0:
        error = (diag_estimated[valid_mask] - diag_true[valid_mask]).abs()
        relative_error = error.mean() / diag_true[valid_mask].abs().mean()

        assert relative_error < 0.15, f"Full Hessian diagonal error: {relative_error:.2e}"


def test_hessian_full_central_csr_matches_coo(quadratic_2d_setup):
    """CSR-based central Hessian should match COO results (diagonal)."""
    positions = quadratic_2d_setup["positions"]
    fitness = quadratic_2d_setup["fitness"]
    edge_index = quadratic_2d_setup["edge_index"]

    csr_data = build_csr_from_coo(edge_index, positions.shape[0])

    grad_result = estimate_gradient_finite_difference(positions, fitness, edge_index)
    result_coo = estimate_hessian_full_fd(
        positions,
        fitness,
        grad_result["gradient"],
        edge_index,
        method="central",
        symmetrize=True,
    )
    result_csr = estimate_hessian_full_fd(
        positions,
        fitness,
        grad_result["gradient"],
        edge_index=None,
        method="central",
        symmetrize=True,
        csr_ptr=csr_data["csr_ptr"],
        csr_indices=csr_data["csr_indices"],
    )

    diag_coo = torch.diagonal(result_coo["hessian_tensors"], dim1=1, dim2=2)
    diag_csr = torch.diagonal(result_csr["hessian_tensors"], dim1=1, dim2=2)
    valid_mask = torch.isfinite(diag_coo).all(dim=1) & torch.isfinite(diag_csr).all(dim=1)

    if valid_mask.any():
        assert torch.allclose(
            diag_coo[valid_mask],
            diag_csr[valid_mask],
            atol=1e-5,
            rtol=1e-4,
        )


def test_hessian_full_quadratic_gradient_fd(quadratic_2d_setup):
    """Test full Hessian with gradient finite difference method."""
    # Estimate gradient first
    grad_result = estimate_gradient_finite_difference(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    # Hessian from gradient FD
    result = estimate_hessian_full_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        grad_result["gradient"],
        quadratic_2d_setup["edge_index"],
        method="gradient_fd",
        symmetrize=True,
    )

    result["hessian_tensors"]
    quadratic_2d_setup["H_true"]

    # Check eigenvalues
    eig_estimated = result["hessian_eigenvalues"]
    eig_true = torch.tensor([[4.0, 2.0]]).expand(100, 2)  # Descending order

    valid_mask = torch.isfinite(eig_estimated).all(dim=1)

    if valid_mask.sum() > 10:
        error = (eig_estimated[valid_mask] - eig_true[valid_mask]).abs()
        relative_error = error.mean() / eig_true[valid_mask].abs().mean()

        assert relative_error < 0.2, f"Eigenvalue error: {relative_error:.2e}"


def test_hessian_full_gradient_fd_csr_matches_coo(quadratic_2d_setup):
    """CSR-based gradient-FD Hessian should match COO results."""
    positions = quadratic_2d_setup["positions"]
    fitness = quadratic_2d_setup["fitness"]
    edge_index = quadratic_2d_setup["edge_index"]

    csr_data = build_csr_from_coo(edge_index, positions.shape[0])

    grad_result = estimate_gradient_finite_difference(positions, fitness, edge_index)
    result_coo = estimate_hessian_full_fd(
        positions,
        fitness,
        grad_result["gradient"],
        edge_index,
        method="gradient_fd",
        symmetrize=True,
    )
    result_csr = estimate_hessian_full_fd(
        positions,
        fitness,
        grad_result["gradient"],
        edge_index=None,
        method="gradient_fd",
        symmetrize=True,
        csr_ptr=csr_data["csr_ptr"],
        csr_indices=csr_data["csr_indices"],
    )

    valid_mask = torch.isfinite(result_coo["hessian_tensors"]).all(dim=(1, 2)) & torch.isfinite(
        result_csr["hessian_tensors"]
    ).all(dim=(1, 2))
    if valid_mask.any():
        assert torch.allclose(
            result_coo["hessian_tensors"][valid_mask],
            result_csr["hessian_tensors"][valid_mask],
            atol=1e-5,
            rtol=1e-4,
        )


def test_hessian_symmetry(quadratic_2d_setup):
    """Test that Hessian is symmetric after symmetrization."""
    grad_result = estimate_gradient_finite_difference(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    result = estimate_hessian_full_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        grad_result["gradient"],
        quadratic_2d_setup["edge_index"],
        method="central",
        symmetrize=True,
    )

    H = result["hessian_tensors"]

    # Check symmetry: H should equal H^T
    symmetry_error = torch.norm(H - H.transpose(1, 2), p="fro", dim=(1, 2))

    # After symmetrization, error should be near zero
    assert symmetry_error.max() < 1e-6, "Hessian should be symmetric"


def test_hessian_eigenvalues_sorted(quadratic_2d_setup):
    """Test that eigenvalues are sorted descending."""
    grad_result = estimate_gradient_finite_difference(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    result = estimate_hessian_full_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        grad_result["gradient"],
        quadratic_2d_setup["edge_index"],
        method="central",
        symmetrize=True,
    )

    eigenvalues = result["hessian_eigenvalues"]

    # Check descending order
    for i in range(eigenvalues.shape[0]):
        if torch.isfinite(eigenvalues[i]).all():
            assert (eigenvalues[i, :-1] >= eigenvalues[i, 1:] - 1e-6).all(), (
                "Eigenvalues should be sorted descending"
            )


def test_hessian_rosenbrock():
    """Test Hessian on Rosenbrock function."""
    # Setup
    n_per_dim = 8
    x = torch.linspace(-1, 1.5, n_per_dim)
    y = torch.linspace(-0.5, 2, n_per_dim)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Fitness
    x1, x2 = positions[:, 0], positions[:, 1]
    fitness = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

    # True Hessian
    N = positions.shape[0]
    H_true = torch.zeros(N, 2, 2)
    H_true[:, 0, 0] = 2 - 400 * x2 + 1200 * x1**2
    H_true[:, 0, 1] = -400 * x1
    H_true[:, 1, 0] = -400 * x1
    H_true[:, 1, 1] = 200

    edge_index = _build_knn_graph(positions, k=10)

    # Estimate diagonal
    result = estimate_hessian_diagonal_fd(positions, fitness, edge_index)

    H_diag = result["hessian_diagonal"]
    H_true_diag = torch.stack([H_true[:, 0, 0], H_true[:, 1, 1]], dim=1)

    valid_mask = result["valid_mask"]

    if valid_mask.sum() > 10:
        error = (H_diag[valid_mask] - H_true_diag[valid_mask]).abs()
        relative_error = error.mean() / (H_true_diag[valid_mask].abs().mean() + 1e-6)

        # Rosenbrock is harder - allow larger error
        assert relative_error < 0.3, f"Rosenbrock Hessian error: {relative_error:.2e}"


def test_hessian_psd_fraction():
    """Test that PSD fraction is computed correctly."""
    # Create simple PSD case
    N = 50
    positions = torch.randn(N, 2)

    # Quadratic with positive definite Hessian
    fitness = positions[:, 0] ** 2 + positions[:, 1] ** 2

    edge_index = _build_knn_graph(positions, k=10)

    grad_result = estimate_gradient_finite_difference(positions, fitness, edge_index)

    result = estimate_hessian_full_fd(
        positions, fitness, grad_result["gradient"], edge_index, method="central", symmetrize=True
    )

    # PSD fraction should be high for this simple case
    psd_fraction = result["psd_fraction"]

    assert 0 <= psd_fraction <= 1, "PSD fraction should be in [0, 1]"
    # Allow some numerical error - may not be exactly 1
    assert psd_fraction > 0.5, "Most should be PSD for simple quadratic"


def test_hessian_condition_numbers(quadratic_2d_setup):
    """Test condition number computation."""
    grad_result = estimate_gradient_finite_difference(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    result = estimate_hessian_full_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        grad_result["gradient"],
        quadratic_2d_setup["edge_index"],
        method="central",
        symmetrize=True,
    )

    condition_numbers = result["condition_numbers"]

    # Should be finite and positive
    valid_mask = torch.isfinite(condition_numbers)
    assert valid_mask.any(), "Should have some valid condition numbers"

    if valid_mask.sum() > 0:
        assert (condition_numbers[valid_mask] > 0).all(), "Condition numbers should be positive"

        # For quadratic with H = diag(2, 4), κ = 4/2 = 2
        true_condition = 4.0 / 2.0
        mean_condition = condition_numbers[valid_mask].mean()

        # Allow some error
        assert abs(mean_condition - true_condition) / true_condition < 0.5, (
            f"Condition number error: {mean_condition} vs {true_condition}"
        )


def test_hessian_with_step_size():
    """Test Hessian estimation with manual step size."""
    N = 50
    positions = torch.randn(N, 2)
    fitness = positions[:, 0] ** 2 + 2 * positions[:, 1] ** 2

    edge_index = _build_knn_graph(positions, k=10)

    # Test different step sizes
    step_sizes = [0.1, 0.5, 1.0]

    for h in step_sizes:
        result = estimate_hessian_diagonal_fd(positions, fitness, edge_index, step_size=h)

        # Should complete without error
        assert result["hessian_diagonal"].shape == (N, 2)
        assert result["step_sizes"].shape == (N, 2)


def test_hessian_handles_isolated_walkers():
    """Test Hessian estimation with isolated walkers."""
    N = 20
    positions = torch.randn(N, 2)

    # Create sparse graph with some isolated walkers
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ],
        dtype=torch.long,
    )

    fitness = positions[:, 0] ** 2 + positions[:, 1] ** 2

    result = estimate_hessian_diagonal_fd(positions, fitness, edge_index)

    # Should handle gracefully
    H_diag = result["hessian_diagonal"]

    # Isolated walkers should have NaN
    isolated_walkers = torch.tensor([i for i in range(N) if i not in {0, 1, 2, 3}])

    if len(isolated_walkers) > 0:
        assert torch.isnan(H_diag[isolated_walkers]).any(), "Isolated walkers should have NaN"


def test_axis_quality_scores(quadratic_2d_setup):
    """Test that axis quality scores are computed."""
    result = estimate_hessian_diagonal_fd(
        quadratic_2d_setup["positions"],
        quadratic_2d_setup["fitness"],
        quadratic_2d_setup["edge_index"],
    )

    axis_quality = result["axis_quality"]

    # Should be in [0, 1]
    valid_mask = torch.isfinite(axis_quality)
    if valid_mask.any():
        assert (axis_quality[valid_mask] >= 0).all(), "Quality should be non-negative"
        assert (axis_quality[valid_mask] <= 1).all(), "Quality should be <= 1"


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
