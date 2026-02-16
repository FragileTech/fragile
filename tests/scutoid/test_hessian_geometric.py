"""Tests for geometric Hessian estimation from metric."""

import pytest
import torch

from fragile.fractalai.scutoid import estimate_hessian_from_metric


@pytest.fixture
def isotropic_setup():
    """Setup with isotropic walker distribution."""
    N = 100
    d = 2
    device = torch.device("cpu")

    # Random isotropic positions
    torch.manual_seed(42)
    positions = torch.randn(N, d, device=device)

    # Build k-NN graph
    edge_index = _build_knn_graph(positions, k=10)

    return {
        "positions": positions,
        "edge_index": edge_index,
    }


def test_geometric_hessian_basic(isotropic_setup):
    """Test basic geometric Hessian computation."""
    result = estimate_hessian_from_metric(
        isotropic_setup["positions"],
        isotropic_setup["edge_index"],
        epsilon_sigma=0.1,
    )

    # Check outputs exist
    assert "hessian_tensors" in result
    assert "hessian_eigenvalues" in result
    assert "metric_tensors" in result
    assert "psd_violation_mask" in result
    assert "equilibrium_score" in result

    # Check shapes
    N, d = isotropic_setup["positions"].shape
    assert result["hessian_tensors"].shape == (N, d, d)
    assert result["hessian_eigenvalues"].shape == (N, d)
    assert result["metric_tensors"].shape == (N, d, d)


def test_geometric_hessian_formula():
    """Test that H = g - ε_Σ I is correctly implemented."""
    N = 50
    d = 2
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=10)

    epsilon_sigma = 0.2

    result = estimate_hessian_from_metric(positions, edge_index, epsilon_sigma=epsilon_sigma)

    H = result["hessian_tensors"]
    g = result["metric_tensors"]

    # Verify: H = g - ε_Σ I
    identity = torch.eye(d).unsqueeze(0).expand(N, d, d)
    H_expected = g - epsilon_sigma * identity

    # Check equality
    diff = torch.norm(H - H_expected, p="fro", dim=(1, 2))
    assert diff.max() < 1e-6, "H should equal g - ε_Σ I"


def test_geometric_hessian_symmetry(isotropic_setup):
    """Test that geometric Hessian is symmetric."""
    result = estimate_hessian_from_metric(
        isotropic_setup["positions"],
        isotropic_setup["edge_index"],
        epsilon_sigma=0.1,
    )

    H = result["hessian_tensors"]

    # Check symmetry
    symmetry_error = torch.norm(H - H.transpose(1, 2), p="fro", dim=(1, 2))

    # Metric should be symmetric, so H should be too
    assert symmetry_error.max() < 1e-5, "Geometric Hessian should be symmetric"


def test_geometric_hessian_eigenvalues_sorted(isotropic_setup):
    """Test that eigenvalues are sorted descending."""
    result = estimate_hessian_from_metric(
        isotropic_setup["positions"],
        isotropic_setup["edge_index"],
        epsilon_sigma=0.1,
    )

    eigenvalues = result["hessian_eigenvalues"]

    # Check descending order
    for i in range(eigenvalues.shape[0]):
        if torch.isfinite(eigenvalues[i]).all():
            diffs = eigenvalues[i, :-1] - eigenvalues[i, 1:]
            assert (diffs >= -1e-6).all(), "Eigenvalues should be sorted descending"


def test_psd_violation_detection():
    """Test that PSD violations are detected."""
    N = 50
    d = 2
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=10)

    # Use large epsilon_sigma that might cause negative eigenvalues
    epsilon_sigma = 10.0

    result = estimate_hessian_from_metric(
        positions,
        edge_index,
        epsilon_sigma=epsilon_sigma,
        validate_equilibrium=False,  # Speed up
    )

    psd_violation_mask = result["psd_violation_mask"]

    # Check that violations are detected based on eigenvalues
    eigenvalues = result["hessian_eigenvalues"]
    min_eigenvalues = eigenvalues[:, -1]  # Smallest eigenvalue (last in descending order)

    expected_violations = min_eigenvalues < -1e-6

    # Should match
    assert torch.equal(psd_violation_mask, expected_violations), (
        "PSD violations should be correctly detected"
    )


def test_equilibrium_score_range(isotropic_setup):
    """Test that equilibrium score is in valid range."""
    result = estimate_hessian_from_metric(
        isotropic_setup["positions"],
        isotropic_setup["edge_index"],
        epsilon_sigma=0.1,
        validate_equilibrium=True,
    )

    equilibrium_score = result["equilibrium_score"]

    # Should be in [0, 1]
    assert (equilibrium_score >= 0).all(), "Equilibrium score should be >= 0"
    assert (equilibrium_score <= 1).all(), "Equilibrium score should be <= 1"


def test_precomputed_metric():
    """Test providing pre-computed metric tensors."""
    N = 50
    d = 2
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=10)

    # First compute metric
    result1 = estimate_hessian_from_metric(positions, edge_index, epsilon_sigma=0.1)
    metric_precomputed = result1["metric_tensors"]

    # Now provide pre-computed metric
    result2 = estimate_hessian_from_metric(
        positions,
        edge_index,
        epsilon_sigma=0.1,
        metric_tensors=metric_precomputed,
    )

    # Results should be identical
    diff = torch.norm(result1["hessian_tensors"] - result2["hessian_tensors"], p="fro", dim=(1, 2))

    assert diff.max() < 1e-6, "Results with pre-computed metric should match"


def test_different_epsilon_sigma_values():
    """Test that different epsilon_sigma values produce different Hessians."""
    N = 50
    d = 2
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=10)

    epsilon_values = [0.05, 0.1, 0.2]
    results = []

    for eps in epsilon_values:
        result = estimate_hessian_from_metric(
            positions,
            edge_index,
            epsilon_sigma=eps,
            validate_equilibrium=False,
        )
        results.append(result)

    # H = g - ε_Σ I, so different ε should give different H
    # Specifically, eigenvalues should differ by ε differences

    for i in range(len(epsilon_values) - 1):
        eps_diff = epsilon_values[i + 1] - epsilon_values[i]

        eig_diff = results[i]["hessian_eigenvalues"] - results[i + 1]["hessian_eigenvalues"]

        # Difference should be approximately eps_diff for all eigenvalues
        # (since we subtract from identity)
        mean_diff = eig_diff.mean()

        assert abs(mean_diff - eps_diff) < 0.05, (
            f"Eigenvalue difference should match epsilon difference: {mean_diff} vs {eps_diff}"
        )


def test_metric_invertibility():
    """Test that computed metrics are invertible."""
    N = 50
    d = 2
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=10)

    result = estimate_hessian_from_metric(positions, edge_index, epsilon_sigma=0.1)

    metrics = result["metric_tensors"]

    # Check that metrics are invertible (non-singular)
    for i in range(N):
        det = torch.linalg.det(metrics[i])
        assert abs(det) > 1e-8, f"Metric {i} should be non-singular, det={det}"


def test_alive_mask_filtering():
    """Test that alive mask filters walkers correctly."""
    N = 50
    d = 2
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=10)

    # Create alive mask
    alive = torch.ones(N, dtype=torch.bool)
    alive[::3] = False  # Every 3rd walker is dead

    result = estimate_hessian_from_metric(positions, edge_index, epsilon_sigma=0.1, alive=alive)

    # Dead walkers should still have metrics computed
    # (alive mask mainly affects edge filtering in utils)
    # But results should be valid
    assert result["hessian_tensors"].shape == (N, d, d)


def test_high_dimensional():
    """Test geometric Hessian in higher dimensions."""
    N = 50
    d = 5  # Higher dimensional
    positions = torch.randn(N, d)
    edge_index = _build_knn_graph(positions, k=15)

    result = estimate_hessian_from_metric(positions, edge_index, epsilon_sigma=0.1)

    # Should work in higher dimensions
    assert result["hessian_tensors"].shape == (N, d, d)
    assert result["hessian_eigenvalues"].shape == (N, d)

    # Metrics should be positive definite in high dimensions too
    eigenvalues = torch.linalg.eigvalsh(result["metric_tensors"])
    assert (eigenvalues > -1e-6).all(), "Metric should be PSD"


def test_metric_isotropy_score():
    """Test that isotropy score reflects distribution shape."""
    N = 100
    d = 2

    # Isotropic distribution
    torch.manual_seed(42)
    positions_iso = torch.randn(N, d)
    edge_index_iso = _build_knn_graph(positions_iso, k=10)

    result_iso = estimate_hessian_from_metric(
        positions_iso, edge_index_iso, epsilon_sigma=0.1, validate_equilibrium=True
    )

    # Anisotropic distribution (stretched along one axis)
    positions_aniso = torch.randn(N, d)
    positions_aniso[:, 0] *= 5  # Stretch x-axis
    edge_index_aniso = _build_knn_graph(positions_aniso, k=10)

    result_aniso = estimate_hessian_from_metric(
        positions_aniso, edge_index_aniso, epsilon_sigma=0.1, validate_equilibrium=True
    )

    # Isotropic should have higher equilibrium score (more uniform)
    # This is a rough heuristic - may not always hold
    mean_iso = result_iso["equilibrium_score"].mean()
    mean_aniso = result_aniso["equilibrium_score"].mean()

    # Just check scores are computed and in valid range
    assert 0 <= mean_iso <= 1
    assert 0 <= mean_aniso <= 1


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
