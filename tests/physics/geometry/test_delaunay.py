"""Tests for fragile.physics.geometry.delaunai module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.delaunai import (
    build_delaunay_edges,
    compute_delaunay_data,
    compute_det_and_volume,
    compute_diffusion_tensor,
    compute_geodesic_distances,
    compute_metric_from_hessian,
    DelaunayGeometryData,
)


# ==========================================================================
# TestBuildDelaunayEdges
# ==========================================================================


class TestBuildDelaunayEdges:
    """Tests for build_delaunay_edges."""

    def test_symmetric_edges(self, grid_2d_positions: Tensor) -> None:
        """Every (i,j) edge should have a corresponding (j,i) edge."""
        edges = build_delaunay_edges(grid_2d_positions.numpy())
        edge_set = set(map(tuple, edges.tolist()))
        for i, j in edges.tolist():
            assert (j, i) in edge_set, f"Missing reverse edge ({j}, {i})"

    def test_no_self_loops(self, grid_2d_positions: Tensor) -> None:
        """No edge should connect a node to itself."""
        edges = build_delaunay_edges(grid_2d_positions.numpy())
        assert (edges[:, 0] != edges[:, 1]).all(), "Self-loop detected"

    def test_no_duplicate_edges(self, grid_2d_positions: Tensor) -> None:
        """All edges should be unique."""
        edges = build_delaunay_edges(grid_2d_positions.numpy())
        unique_edges = np.unique(edges, axis=0)
        assert edges.shape[0] == unique_edges.shape[0], "Duplicate edges found"

    def test_all_nodes_connected(self, grid_2d_positions: Tensor) -> None:
        """Every node should appear as a source in at least one edge."""
        pos_np = grid_2d_positions.numpy()
        edges = build_delaunay_edges(pos_np)
        n = pos_np.shape[0]
        sources = set(edges[:, 0].tolist())
        assert len(sources) == n, f"Only {len(sources)}/{n} nodes appear as source"

    def test_3d_positions(self, grid_3d_positions: Tensor) -> None:
        """build_delaunay_edges works for 3D point clouds."""
        pos_np = grid_3d_positions.numpy()
        edges = build_delaunay_edges(pos_np)
        assert edges.ndim == 2
        assert edges.shape[1] == 2
        assert edges.shape[0] > 0, "Should produce edges for a 3D grid"
        # symmetry check
        edge_set = set(map(tuple, edges.tolist()))
        for i, j in edges.tolist():
            assert (j, i) in edge_set

    def test_too_few_points(self) -> None:
        """A single 2D point cannot form a simplex; returns empty."""
        edges = build_delaunay_edges(np.array([[0.0, 0.0]]))
        assert edges.shape == (0, 2)

    def test_collinear_points(self) -> None:
        """Collinear points in 2D cannot form a 2-simplex; returns empty."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        edges = build_delaunay_edges(points)
        assert edges.shape == (0, 2)

    def test_deterministic(self, grid_2d_positions: Tensor) -> None:
        """Same input gives same output every time."""
        pos_np = grid_2d_positions.numpy()
        edges1 = build_delaunay_edges(pos_np)
        edges2 = build_delaunay_edges(pos_np)
        np.testing.assert_array_equal(edges1, edges2)

    def test_correct_dtype(self, grid_2d_positions: Tensor) -> None:
        """Output dtype should be int64."""
        edges = build_delaunay_edges(grid_2d_positions.numpy())
        assert edges.dtype == np.int64


# ==========================================================================
# TestComputeMetricFromHessian
# ==========================================================================


class TestComputeMetricFromHessian:
    """Tests for compute_metric_from_hessian."""

    def test_shapes(self) -> None:
        """Output shapes should be [N,d,d], [N], [N]."""
        N, d = 10, 3
        hess = torch.zeros(N, d, d)
        metric, det, volume = compute_metric_from_hessian(
            hessian_diag=None,
            hessian_full=hess,
            epsilon_sigma=1.0,
            min_eig=1e-6,
            max_eig=None,
        )
        assert metric.shape == (N, d, d)
        assert det.shape == (N,)
        assert volume.shape == (N,)

    def test_symmetry(self) -> None:
        """Output metric should be symmetric."""
        N, d = 8, 2
        gen = torch.Generator().manual_seed(7)
        hess = torch.randn(N, d, d, generator=gen)
        metric, _, _ = compute_metric_from_hessian(
            hessian_diag=None,
            hessian_full=hess,
            epsilon_sigma=1.0,
            min_eig=1e-6,
            max_eig=None,
        )
        torch.testing.assert_close(metric, metric.transpose(-1, -2))

    def test_identity_case(self) -> None:
        """Zero hessian with eps=1 should give metric ~ I, det ~ 1, volume ~ 1."""
        N, d = 5, 2
        hess = torch.zeros(N, d, d)
        metric, det, volume = compute_metric_from_hessian(
            hessian_diag=None,
            hessian_full=hess,
            epsilon_sigma=1.0,
            min_eig=None,
            max_eig=None,
        )
        eye = torch.eye(d).unsqueeze(0).expand(N, -1, -1)
        torch.testing.assert_close(metric, eye)
        torch.testing.assert_close(det, torch.ones(N))
        torch.testing.assert_close(volume, torch.ones(N))

    def test_diagonal_path(self) -> None:
        """hessian_diag=[a,b] with eps=e -> metric=diag(a+e, b+e)."""
        N = 4
        a, b, e = 2.0, 3.0, 0.5
        hess_diag = torch.tensor([[a, b]] * N)
        metric, det, _volume = compute_metric_from_hessian(
            hessian_diag=hess_diag,
            hessian_full=None,
            epsilon_sigma=e,
            min_eig=None,
            max_eig=None,
        )
        expected_diag = torch.tensor([a + e, b + e])
        expected_metric = torch.diag(expected_diag).unsqueeze(0).expand(N, -1, -1)
        torch.testing.assert_close(metric, expected_metric)
        expected_det = torch.full((N,), (a + e) * (b + e))
        torch.testing.assert_close(det, expected_det)

    def test_min_eig_clamping(self) -> None:
        """Small eigenvalues should be clamped up to min_eig."""
        N, d = 3, 2
        # hessian with very negative diagonal => metric diag would be negative without clamping
        hess = torch.zeros(N, d, d)
        hess[:, 0, 0] = -100.0  # will produce very small eigenvalue
        metric, _det, _volume = compute_metric_from_hessian(
            hessian_diag=None,
            hessian_full=hess,
            epsilon_sigma=0.01,
            min_eig=0.5,
            max_eig=None,
        )
        eigvals = torch.linalg.eigvalsh(metric)
        assert (eigvals >= 0.5 - 1e-6).all(), "Eigenvalues should be clamped to >= min_eig"

    def test_max_eig_clamping(self) -> None:
        """Large eigenvalues should be clamped down to max_eig."""
        N, d = 3, 2
        hess = torch.zeros(N, d, d)
        hess[:, 0, 0] = 1000.0  # will produce very large eigenvalue
        metric, _, _ = compute_metric_from_hessian(
            hessian_diag=None,
            hessian_full=hess,
            epsilon_sigma=1.0,
            min_eig=None,
            max_eig=10.0,
        )
        eigvals = torch.linalg.eigvalsh(metric)
        assert (eigvals <= 10.0 + 1e-6).all(), "Eigenvalues should be clamped to <= max_eig"

    def test_nan_handling(self) -> None:
        """NaN values in hessian should be replaced with 0 before computing metric."""
        N, d = 4, 2
        hess = torch.zeros(N, d, d)
        hess[0, 0, 0] = float("nan")
        hess[1, 1, 1] = float("nan")
        metric, det, volume = compute_metric_from_hessian(
            hessian_diag=None,
            hessian_full=hess,
            epsilon_sigma=1.0,
            min_eig=None,
            max_eig=None,
        )
        assert not torch.isnan(metric).any(), "Metric should not contain NaN"
        assert not torch.isnan(det).any(), "Det should not contain NaN"
        assert not torch.isnan(volume).any(), "Volume should not contain NaN"

    def test_both_none_raises(self) -> None:
        """Passing both hessian_diag=None and hessian_full=None should raise ValueError."""
        with pytest.raises(ValueError, match="Need either"):
            compute_metric_from_hessian(
                hessian_diag=None,
                hessian_full=None,
                epsilon_sigma=1.0,
                min_eig=None,
                max_eig=None,
            )


# ==========================================================================
# TestComputeDetAndVolume
# ==========================================================================


class TestComputeDetAndVolume:
    """Tests for compute_det_and_volume."""

    def test_identity_metrics(self, identity_metric_2d: Tensor) -> None:
        """Identity metrics should have det=1 and volume=1."""
        det, volume = compute_det_and_volume(identity_metric_2d)
        torch.testing.assert_close(det, torch.ones(100), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(volume, torch.ones(100), atol=1e-5, rtol=1e-5)

    def test_diagonal_metrics(self, diagonal_metric_2d: Tensor) -> None:
        """diag(a,b) -> det=a*b, volume=sqrt(a*b)."""
        a, b = 2.0, 0.5
        det, volume = compute_det_and_volume(diagonal_metric_2d)
        expected_det = torch.full((100,), a * b)
        expected_volume = torch.full((100,), float(np.sqrt(a * b)))
        torch.testing.assert_close(det, expected_det, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(volume, expected_volume, atol=1e-5, rtol=1e-5)

    def test_volume_is_sqrt_clamped_det(self) -> None:
        """volume should equal sqrt(clamp(det, min=eps))."""
        N, d = 10, 2
        gen = torch.Generator().manual_seed(55)
        A = torch.randn(N, d, d, generator=gen)
        spd = A @ A.transpose(-1, -2) + 0.1 * torch.eye(d)
        det, volume = compute_det_and_volume(spd, eps=1e-12)
        expected_volume = torch.sqrt(torch.clamp(det, min=1e-12))
        torch.testing.assert_close(volume, expected_volume, atol=1e-6, rtol=1e-6)

    def test_always_positive(self, identity_metric_2d: Tensor) -> None:
        """Det and volume should always be positive."""
        det, volume = compute_det_and_volume(identity_metric_2d)
        assert (det > 0).all(), "Det should be positive"
        assert (volume > 0).all(), "Volume should be positive"


# ==========================================================================
# TestComputeDiffusionTensor
# ==========================================================================


class TestComputeDiffusionTensor:
    """Tests for compute_diffusion_tensor."""

    def test_identity_metric(self, identity_metric_2d: Tensor) -> None:
        """Identity metric -> identity diffusion tensor."""
        sigma = compute_diffusion_tensor(identity_metric_2d)
        expected = torch.eye(2).unsqueeze(0).expand(100, -1, -1)
        torch.testing.assert_close(sigma, expected, atol=1e-5, rtol=1e-5)

    def test_diagonal_metric(self) -> None:
        """diag(a,b) -> Sigma = diag(1/sqrt(a), 1/sqrt(b))."""
        N = 5
        a, b = 4.0, 9.0
        metric = torch.diag(torch.tensor([a, b])).unsqueeze(0).expand(N, -1, -1).clone()
        sigma = compute_diffusion_tensor(metric)
        expected_diag = torch.tensor([1.0 / a**0.5, 1.0 / b**0.5])
        expected = torch.diag(expected_diag).unsqueeze(0).expand(N, -1, -1)
        torch.testing.assert_close(sigma, expected, atol=1e-5, rtol=1e-5)

    def test_symmetry(self, full_spd_metric_2d: Tensor) -> None:
        """Diffusion tensor should be symmetric."""
        sigma = compute_diffusion_tensor(full_spd_metric_2d)
        torch.testing.assert_close(sigma, sigma.transpose(-1, -2), atol=1e-5, rtol=1e-5)

    def test_positive_definite(self, full_spd_metric_2d: Tensor) -> None:
        """Diffusion tensor eigenvalues should all be positive."""
        sigma = compute_diffusion_tensor(full_spd_metric_2d)
        eigvals = torch.linalg.eigvalsh(sigma)
        assert (eigvals > 0).all(), "Diffusion tensor should be positive definite"


# ==========================================================================
# TestComputeGeodesicDistances
# ==========================================================================


class TestComputeGeodesicDistances:
    """Tests for compute_geodesic_distances."""

    def test_identity_metric_equals_euclidean(
        self,
        grid_2d_positions: Tensor,
        delaunay_2d_edges: Tensor,
        identity_metric_2d: Tensor,
    ) -> None:
        """With identity metric, geodesic distances should equal Euclidean distances."""
        geo = compute_geodesic_distances(grid_2d_positions, delaunay_2d_edges, identity_metric_2d)
        src, dst = delaunay_2d_edges
        euclidean = torch.norm(grid_2d_positions[dst] - grid_2d_positions[src], dim=1)
        torch.testing.assert_close(geo, euclidean, atol=1e-5, rtol=1e-5)

    def test_scaled_metric(
        self,
        grid_2d_positions: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """Scaled metric c*I -> geodesic = sqrt(c) * euclidean."""
        c = 4.0
        N = grid_2d_positions.shape[0]
        metric = c * torch.eye(2).unsqueeze(0).expand(N, -1, -1).clone()
        geo = compute_geodesic_distances(grid_2d_positions, delaunay_2d_edges, metric)
        src, dst = delaunay_2d_edges
        euclidean = torch.norm(grid_2d_positions[dst] - grid_2d_positions[src], dim=1)
        expected = np.sqrt(c) * euclidean
        torch.testing.assert_close(geo, expected, atol=1e-5, rtol=1e-5)

    def test_positive_distances(
        self,
        grid_2d_positions: Tensor,
        delaunay_2d_edges: Tensor,
        identity_metric_2d: Tensor,
    ) -> None:
        """All geodesic distances should be positive for non-degenerate edges."""
        geo = compute_geodesic_distances(grid_2d_positions, delaunay_2d_edges, identity_metric_2d)
        assert (geo > 0).all(), "Geodesic distances should be positive"

    def test_empty_edges(self, grid_2d_positions: Tensor, identity_metric_2d: Tensor) -> None:
        """Empty edge index should return empty result."""
        empty_edges = torch.zeros(2, 0, dtype=torch.long)
        geo = compute_geodesic_distances(grid_2d_positions, empty_edges, identity_metric_2d)
        assert geo.shape == (0,)


# ==========================================================================
# TestComputeDelaunayData (integration tests)
# ==========================================================================


class TestComputeDelaunayData:
    """Integration tests for compute_delaunay_data."""

    def test_returns_dataclass(self, quadratic_fitness: dict) -> None:
        """compute_delaunay_data should return a DelaunayGeometryData instance."""
        qf = quadratic_fitness
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
        )
        assert isinstance(result, DelaunayGeometryData)

    def test_field_shapes(self, quadratic_fitness: dict) -> None:
        """All fields should have documented shapes."""
        qf = quadratic_fitness
        pos = qf["positions"]
        N, d = pos.shape

        result = compute_delaunay_data(positions=pos, fitness_values=qf["fitness"])

        assert result.positions.shape == (N, d)
        assert result.fitness.shape == (N,)
        assert result.edge_index.shape[0] == 2
        E = result.edge_index.shape[1]
        assert E > 0
        assert result.csr_ptr.shape == (N + 1,)
        assert result.csr_indices.shape[0] == E
        assert result.neighbor_matrix.shape[0] == N
        assert result.neighbor_mask.shape[0] == N
        assert result.neighbor_counts.shape == (N,)
        assert result.edge_distances.shape == (E,)
        assert result.edge_geodesic_distances.shape == (E,)
        assert result.metric_tensors.shape == (N, d, d)
        assert result.metric_det.shape == (N,)
        assert result.riemannian_volume_weights.shape == (N,)
        assert result.diffusion_tensors.shape == (N, d, d)
        assert result.ricci_proxy.shape == (N,)

    def test_covariance_mode_spd(self, quadratic_fitness: dict) -> None:
        """metric_mode='covariance' should produce SPD metrics (positive eigenvalues)."""
        qf = quadratic_fitness
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
            metric_mode="covariance",
        )
        eigvals = torch.linalg.eigvalsh(result.metric_tensors)
        assert (eigvals > 0).all(), "Covariance metrics should be SPD"

    def test_hessian_full_mode(self, quadratic_fitness: dict) -> None:
        """metric_mode='hessian' + hessian_mode='full' should run and produce hessian_full."""
        qf = quadratic_fitness
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
            metric_mode="hessian",
            hessian_mode="full",
        )
        assert result.hessian_full is not None
        assert result.hessian_full.shape == (qf["positions"].shape[0], 2, 2)

    def test_hessian_diagonal_mode(self, quadratic_fitness: dict) -> None:
        """metric_mode='hessian' + hessian_mode='diagonal' should produce hessian_diag."""
        qf = quadratic_fitness
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
            metric_mode="hessian",
            hessian_mode="diagonal",
        )
        assert result.hessian_diag is not None
        assert result.hessian_diag.shape == (qf["positions"].shape[0], 2)

    def test_invalid_metric_mode_raises(self, quadratic_fitness: dict) -> None:
        """Invalid metric_mode should raise ValueError."""
        qf = quadratic_fitness
        with pytest.raises(ValueError, match="Unknown metric_mode"):
            compute_delaunay_data(
                positions=qf["positions"],
                fitness_values=qf["fitness"],
                metric_mode="invalid_mode",
            )

    def test_empty_positions_raises(self) -> None:
        """Empty positions should raise ValueError."""
        pos = torch.zeros(0, 2)
        fit = torch.zeros(0)
        with pytest.raises(ValueError, match="No walkers"):
            compute_delaunay_data(positions=pos, fitness_values=fit)

    def test_multiple_weight_modes(self, quadratic_fitness: dict) -> None:
        """Passing multiple weight_modes should produce that many entries in edge_weights."""
        qf = quadratic_fitness
        modes = ["uniform", "inverse_distance", "inverse_riemannian_distance"]
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
            weight_modes=modes,
        )
        assert len(result.edge_weights) == len(modes)
        for m in modes:
            assert m in result.edge_weights, f"Missing weight mode '{m}'"
            assert result.edge_weights[m].shape[0] == result.edge_index.shape[1]

    def test_compute_full_ricci(self, quadratic_fitness: dict) -> None:
        """compute_full_ricci=True should populate ricci_proxy_full and ricci_tensor_full."""
        qf = quadratic_fitness
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
            compute_full_ricci=True,
        )
        assert result.ricci_proxy_full is not None
        assert result.ricci_proxy_full.shape == (qf["positions"].shape[0],)
        assert result.ricci_tensor_full is not None
        assert result.ricci_tensor_full.shape == (qf["positions"].shape[0], 2, 2)

    def test_spatial_dims(self, quadratic_fitness: dict) -> None:
        """spatial_dims=1 for 2D positions should use only the first dimension."""
        qf = quadratic_fitness
        result = compute_delaunay_data(
            positions=qf["positions"],
            fitness_values=qf["fitness"],
            spatial_dims=1,
        )
        # With spatial_dims=1, positions used should be [N, 1]
        assert result.positions.shape[1] == 1
        assert result.metric_tensors.shape == (qf["positions"].shape[0], 1, 1)
