"""Tests for fragile.physics.geometry.hessian_estimation module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.hessian_estimation import (
    compute_emergent_metric,
    estimate_hessian_diagonal_fd,
    estimate_hessian_from_metric,
    estimate_hessian_full_fd,
)
from fragile.physics.geometry.neighbors import build_csr_from_coo


# ---------------------------------------------------------------------------
# TestEstimateHessianDiagonalFd
# ---------------------------------------------------------------------------


class TestEstimateHessianDiagonalFd:
    """Tests for estimate_hessian_diagonal_fd."""

    def test_quadratic_hessian_values(self, quadratic_fitness: dict):
        """V = x^2 + 2y^2 => diagonal Hessian [2, 4]. Check median of valid finite estimates."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_diagonal_fd(pos, fitness, edge_index)
        valid = result["valid_mask"]
        hess = result["hessian_diagonal"]

        assert valid.any(), "Should have at least some valid walkers"
        valid_hess = hess[valid]
        # Some valid walkers may still have NaN on individual axes (only d//2 needed),
        # so filter to finite values per axis.
        finite_h0 = valid_hess[:, 0][torch.isfinite(valid_hess[:, 0])]
        finite_h1 = valid_hess[:, 1][torch.isfinite(valid_hess[:, 1])]
        assert finite_h0.numel() > 0, "Should have finite estimates on axis 0"
        assert finite_h1.numel() > 0, "Should have finite estimates on axis 1"
        median_h00 = finite_h0.median().item()
        median_h11 = finite_h1.median().item()
        assert abs(median_h00 - 2.0) < 1.0, f"Expected ~2.0, got {median_h00}"
        assert abs(median_h11 - 4.0) < 2.0, f"Expected ~4.0, got {median_h11}"

    def test_output_shape(self, quadratic_fitness: dict):
        """hessian_diagonal shape is [N, d], valid_mask shape is [N]."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]
        N, d = pos.shape

        result = estimate_hessian_diagonal_fd(pos, fitness, edge_index)
        assert result["hessian_diagonal"].shape == (N, d)
        assert result["valid_mask"].shape == (N,)
        assert result["valid_mask"].dtype == torch.bool

    def test_linear_fitness_hessian_near_zero(self, linear_fitness: dict):
        """Linear V = 3x + 5y => Hessian diagonal should be near 0."""
        pos = linear_fitness["positions"]
        fitness = linear_fitness["fitness"]
        edge_index = linear_fitness["edge_index"]

        result = estimate_hessian_diagonal_fd(pos, fitness, edge_index)
        valid = result["valid_mask"]
        if valid.any():
            valid_hess = result["hessian_diagonal"][valid]
            finite_vals = valid_hess[torch.isfinite(valid_hess)]
            if finite_vals.numel() > 0:
                median_abs = finite_vals.abs().median().item()
                assert median_abs < 1.0, f"Expected ~0, got median abs {median_abs}"

    def test_output_keys_present(self, quadratic_fitness: dict):
        """All expected keys present in the output dict."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_diagonal_fd(pos, fitness, edge_index)
        expected_keys = {
            "hessian_diagonal",
            "eigenvalues",
            "step_sizes",
            "axis_quality",
            "valid_mask",
        }
        assert set(result.keys()) == expected_keys

    def test_csr_input_same_results(self, quadratic_fitness: dict):
        """CSR input via build_csr_from_coo gives same results as COO edge_index."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        result_coo = estimate_hessian_diagonal_fd(pos, fitness, edge_index)

        csr = build_csr_from_coo(edge_index, n_nodes=N)
        result_csr = estimate_hessian_diagonal_fd(
            pos,
            fitness,
            edge_index=None,
            csr_ptr=csr["csr_ptr"],
            csr_indices=csr["csr_indices"],
        )

        # Both should have the same valid masks and hessian values where valid
        both_valid = result_coo["valid_mask"] & result_csr["valid_mask"]
        if both_valid.any():
            hess_coo = result_coo["hessian_diagonal"][both_valid]
            hess_csr = result_csr["hessian_diagonal"][both_valid]
            # Compare only finite values (some axes may be NaN for boundary walkers)
            finite_mask = torch.isfinite(hess_coo) & torch.isfinite(hess_csr)
            if finite_mask.any():
                torch.testing.assert_close(
                    hess_coo[finite_mask],
                    hess_csr[finite_mask],
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_alive_mask_dead_walkers_nan(self, quadratic_fitness: dict):
        """Dead walkers get NaN hessian diagonal."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]
        N = pos.shape[0]

        alive = torch.ones(N, dtype=torch.bool)
        alive[0] = False
        alive[5] = False
        alive[10] = False

        result = estimate_hessian_diagonal_fd(pos, fitness, edge_index, alive=alive)
        assert result["valid_mask"][0].item() is False
        assert result["valid_mask"][5].item() is False
        assert result["valid_mask"][10].item() is False
        assert torch.isnan(result["hessian_diagonal"][0]).all()
        assert torch.isnan(result["hessian_diagonal"][5]).all()
        assert torch.isnan(result["hessian_diagonal"][10]).all()


# ---------------------------------------------------------------------------
# TestEstimateHessianFullFd
# ---------------------------------------------------------------------------


class TestEstimateHessianFullFd:
    """Tests for estimate_hessian_full_fd."""

    def test_central_quadratic_diagonal(self, quadratic_fitness: dict):
        """Central method on V = x^2 + 2y^2: diag elements near [2, 4], off-diag near 0."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_full_fd(
            pos, fitness, gradient_vectors=None, edge_index=edge_index, method="central"
        )
        valid = result["valid_mask"]
        H = result["hessian_tensors"]

        assert valid.any(), "Should have at least some valid walkers"
        valid_H = H[valid]

        median_h00 = valid_H[:, 0, 0].median().item()
        median_h11 = valid_H[:, 1, 1].median().item()
        median_h01 = valid_H[:, 0, 1].abs().median().item()

        assert abs(median_h00 - 2.0) < 1.0, f"Expected ~2.0, got {median_h00}"
        assert abs(median_h11 - 4.0) < 2.0, f"Expected ~4.0, got {median_h11}"
        assert median_h01 < 1.0, f"Expected off-diag ~0, got {median_h01}"

    def test_symmetry_after_symmetrize(self, quadratic_fitness: dict):
        """With symmetrize=True, H == H^T for all walkers."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_full_fd(
            pos, fitness, gradient_vectors=None, edge_index=edge_index, symmetrize=True
        )
        H = result["hessian_tensors"]
        diff = (H - H.transpose(1, 2)).abs().max().item()
        assert diff < 1e-6, f"Symmetry violated: max |H - H^T| = {diff}"

    def test_psd_fraction_range(self, quadratic_fitness: dict):
        """psd_fraction is in [0, 1]."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_full_fd(
            pos, fitness, gradient_vectors=None, edge_index=edge_index
        )
        assert 0.0 <= result["psd_fraction"] <= 1.0

    def test_condition_numbers_nonnegative(self, quadratic_fitness: dict):
        """Condition numbers are non-negative for valid walkers."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_full_fd(
            pos, fitness, gradient_vectors=None, edge_index=edge_index
        )
        valid = result["valid_mask"]
        if valid.any():
            cond = result["condition_numbers"][valid]
            assert (cond >= 0).all(), "Condition numbers should be non-negative for valid walkers"
            # Most interior walkers should have positive condition numbers
            assert (
                cond > 0
            ).sum() > 0, "At least some walkers should have positive condition numbers"

    def test_gradient_fd_method(self, quadratic_fitness: dict):
        """gradient_fd method with pre-computed gradients produces a Hessian tensor."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]
        grad_fn = quadratic_fitness["grad_fn"]

        gradients = grad_fn(pos)
        result = estimate_hessian_full_fd(
            pos,
            fitness,
            gradient_vectors=gradients,
            edge_index=edge_index,
            method="gradient_fd",
        )
        N, d = pos.shape
        assert result["hessian_tensors"].shape == (N, d, d)
        assert result["valid_mask"].shape == (N,)

    def test_gradient_fd_without_gradients_raises(self, quadratic_fitness: dict):
        """gradient_fd method without gradient_vectors raises ValueError."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        with pytest.raises(ValueError, match="gradient_fd method requires gradient_vectors"):
            estimate_hessian_full_fd(
                pos,
                fitness,
                gradient_vectors=None,
                edge_index=edge_index,
                method="gradient_fd",
            )

    def test_output_keys_present(self, quadratic_fitness: dict):
        """All expected keys present in the output dict."""
        pos = quadratic_fitness["positions"]
        fitness = quadratic_fitness["fitness"]
        edge_index = quadratic_fitness["edge_index"]

        result = estimate_hessian_full_fd(
            pos, fitness, gradient_vectors=None, edge_index=edge_index
        )
        expected_keys = {
            "hessian_tensors",
            "hessian_eigenvalues",
            "condition_numbers",
            "symmetry_error",
            "psd_fraction",
            "valid_mask",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestEstimateHessianFromMetric
# ---------------------------------------------------------------------------


class TestEstimateHessianFromMetric:
    """Tests for estimate_hessian_from_metric."""

    def test_algebraic_identity(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        """H = g - eps*I: verify with known identity metric."""
        pos = grid_2d_positions
        eps = 0.1

        result = estimate_hessian_from_metric(
            pos,
            edge_index=delaunay_2d_edges,
            epsilon_sigma=eps,
            metric_tensors=identity_metric_2d,
            validate_equilibrium=False,
        )
        H = result["hessian_tensors"]
        expected = identity_metric_2d - eps * torch.eye(2).unsqueeze(0).expand_as(
            identity_metric_2d
        )
        torch.testing.assert_close(H, expected, atol=1e-6, rtol=1e-6)

    def test_symmetry(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Output Hessian tensors are symmetric."""
        pos = grid_2d_positions
        result = estimate_hessian_from_metric(
            pos, edge_index=delaunay_2d_edges, validate_equilibrium=False
        )
        H = result["hessian_tensors"]
        diff = (H - H.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5, f"Symmetry violated: max |H - H^T| = {diff}"

    def test_eigenvalues_descending(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Eigenvalues are returned in descending order."""
        pos = grid_2d_positions
        result = estimate_hessian_from_metric(
            pos, edge_index=delaunay_2d_edges, validate_equilibrium=False
        )
        evals = result["hessian_eigenvalues"]
        # Check descending: each column should be >= the next
        for i in range(evals.shape[1] - 1):
            assert (
                evals[:, i] >= evals[:, i + 1] - 1e-6
            ).all(), f"Eigenvalues not descending at column {i}"

    def test_psd_violation_detection(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Large epsilon_sigma forces negative eigenvalues, detected by psd_violation_mask."""
        pos = grid_2d_positions

        with pytest.warns(UserWarning, match="negative eigenvalues"):
            result = estimate_hessian_from_metric(
                pos,
                edge_index=delaunay_2d_edges,
                epsilon_sigma=1e6,
                validate_equilibrium=False,
            )
        assert result[
            "psd_violation_mask"
        ].any(), "Large epsilon_sigma should produce PSD violations"

    def test_precomputed_metric_used(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, diagonal_metric_2d: Tensor
    ):
        """Passing precomputed metric_tensors bypasses internal computation."""
        pos = grid_2d_positions
        eps = 0.05
        metric = diagonal_metric_2d  # diag(2, 0.5)

        result = estimate_hessian_from_metric(
            pos,
            edge_index=delaunay_2d_edges,
            epsilon_sigma=eps,
            metric_tensors=metric,
            validate_equilibrium=False,
        )
        # Verify metric is passed through unchanged
        torch.testing.assert_close(result["metric_tensors"], metric, atol=1e-7, rtol=1e-7)
        # Verify H = g - eps*I
        expected_H = metric - eps * torch.eye(2).unsqueeze(0).expand_as(metric)
        torch.testing.assert_close(result["hessian_tensors"], expected_H, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestComputeEmergentMetric
# ---------------------------------------------------------------------------


class TestComputeEmergentMetric:
    """Tests for compute_emergent_metric."""

    def test_output_shape(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Output shape is [N, d, d]."""
        pos = grid_2d_positions
        N, d = pos.shape
        metric = compute_emergent_metric(pos, delaunay_2d_edges)
        assert metric.shape == (N, d, d)

    def test_symmetry(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Metric is symmetric: g = g^T within numerical precision."""
        pos = grid_2d_positions
        metric = compute_emergent_metric(pos, delaunay_2d_edges)
        diff = (metric - metric.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5, f"Metric not symmetric: max |g - g^T| = {diff}"

    def test_positive_definite(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """All eigenvalues of the metric are positive (SPD)."""
        pos = grid_2d_positions
        metric = compute_emergent_metric(pos, delaunay_2d_edges)
        eigenvalues = torch.linalg.eigvalsh(metric)
        assert (
            eigenvalues > 0
        ).all(), f"Metric not positive definite: min eigenvalue = {eigenvalues.min().item()}"

    def test_isotropic_grid_metric_near_identity_ratio(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Regular grid: metric roughly proportional to I (eigenvalue ratio < 5)."""
        pos = grid_2d_positions
        metric = compute_emergent_metric(pos, delaunay_2d_edges)
        eigenvalues = torch.linalg.eigvalsh(metric)

        # For interior walkers (avoid boundary effects), check eigenvalue ratio
        # Interior walkers on 10x10 grid: rows 2-7, cols 2-7 => indices i*10+j
        interior_indices = []
        for i in range(2, 8):
            for j in range(2, 8):
                interior_indices.append(i * 10 + j)
        interior_indices = torch.tensor(interior_indices)

        interior_evals = eigenvalues[interior_indices]
        ratio = interior_evals[:, -1] / (interior_evals[:, 0] + 1e-10)
        # Regular grid should have relatively isotropic metric
        assert (
            ratio.median().item() < 5.0
        ), f"Eigenvalue ratio too large for regular grid: median = {ratio.median().item()}"

    def test_anisotropic_distribution(self):
        """Points stretched along x-axis produce metric with different eigenvalues."""
        torch.manual_seed(42)
        N = 50
        # Create anisotropic point cloud: stretched along x, compressed along y
        positions = torch.randn(N, 2)
        positions[:, 0] *= 10.0  # stretch x by 10x
        positions[:, 1] *= 0.5  # compress y

        # Build kNN graph
        dists = torch.cdist(positions, positions)
        dists.fill_diagonal_(float("inf"))
        _, indices = dists.topk(8, dim=1, largest=False)
        src = torch.arange(N).unsqueeze(1).expand(-1, 8).reshape(-1)
        dst = indices.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        edge_rev = torch.stack([dst, src], dim=0)
        edge_index = torch.unique(torch.cat([edge_index, edge_rev], dim=1), dim=1)

        metric = compute_emergent_metric(positions, edge_index)
        eigenvalues = torch.linalg.eigvalsh(metric)

        # Metric eigenvalues should differ significantly due to anisotropy
        ratio = eigenvalues[:, -1] / (eigenvalues[:, 0] + 1e-10)
        median_ratio = ratio.median().item()
        assert (
            median_ratio > 2.0
        ), f"Expected anisotropic metric (ratio > 2), got median ratio = {median_ratio}"

    def test_alive_mask_dead_walkers_nan(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Dead walkers get NaN metric."""
        pos = grid_2d_positions
        N = pos.shape[0]
        alive = torch.ones(N, dtype=torch.bool)
        alive[0] = False
        alive[5] = False

        metric = compute_emergent_metric(pos, delaunay_2d_edges, alive=alive)
        assert torch.isnan(metric[0]).all(), "Dead walker 0 should have NaN metric"
        assert torch.isnan(metric[5]).all(), "Dead walker 5 should have NaN metric"
        # Alive walkers should have finite metric
        assert torch.isfinite(metric[alive]).all(), "Alive walkers should have finite metric"
