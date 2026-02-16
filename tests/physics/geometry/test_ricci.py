"""Tests for fragile.physics.geometry.ricci module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.ricci import (
    compute_ricci_proxy,
    compute_ricci_proxy_full_metric,
    compute_ricci_tensor_proxy,
    compute_ricci_tensor_proxy_full_metric,
)
from fragile.physics.geometry.weights import compute_uniform_weights


# ---------------------------------------------------------------------------
# TestComputeRicciProxy
# ---------------------------------------------------------------------------


class TestComputeRicciProxy:
    """Tests for compute_ricci_proxy."""

    def test_flat_metric_uniform_det(self, delaunay_2d_edges: Tensor) -> None:
        """Uniform det=1.0 means log(det)=0, so u=0 and all diffs vanish -> Ricci=0."""
        N = 100
        metric_det = torch.ones(N)
        edge_weights = compute_uniform_weights(delaunay_2d_edges, n_nodes=N)
        spatial_dim = 2

        result = compute_ricci_proxy(metric_det, delaunay_2d_edges, edge_weights, spatial_dim)

        torch.testing.assert_close(result, torch.zeros(N), atol=1e-6, rtol=0.0)

    def test_output_shape(self, delaunay_2d_edges: Tensor) -> None:
        """Output shape must be [N] matching the number of nodes."""
        N = 100
        metric_det = torch.ones(N) * 2.0
        edge_weights = compute_uniform_weights(delaunay_2d_edges, n_nodes=N)
        spatial_dim = 2

        result = compute_ricci_proxy(metric_det, delaunay_2d_edges, edge_weights, spatial_dim)

        assert result.shape == (N,)

    def test_empty_edge_index(self, empty_edge_index: Tensor) -> None:
        """With no edges, the Laplacian is zero and the result should be zeros."""
        N = 10
        metric_det = torch.ones(N) * 3.0
        edge_weights = torch.empty(0)
        spatial_dim = 2

        result = compute_ricci_proxy(metric_det, empty_edge_index, edge_weights, spatial_dim)

        torch.testing.assert_close(result, torch.zeros(N), atol=1e-12, rtol=0.0)

    def test_empty_metric_det(self, empty_edge_index: Tensor) -> None:
        """Empty metric_det should return an empty tensor."""
        metric_det = torch.empty(0)
        edge_weights = torch.empty(0)
        spatial_dim = 2

        result = compute_ricci_proxy(metric_det, empty_edge_index, edge_weights, spatial_dim)

        assert result.numel() == 0


# ---------------------------------------------------------------------------
# TestComputeRicciProxyFullMetric
# ---------------------------------------------------------------------------


class TestComputeRicciProxyFullMetric:
    """Tests for compute_ricci_proxy_full_metric."""

    def test_identity_metric_near_zero(
        self,
        grid_2d_positions: Tensor,
        identity_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """Identity metric on a regular grid is flat: curvature should be near zero."""
        result = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=identity_metric_2d,
            edge_index=delaunay_2d_edges,
        )

        torch.testing.assert_close(
            result,
            torch.zeros_like(result),
            atol=1e-4,
            rtol=0.0,
        )

    def test_output_shape(
        self,
        grid_2d_positions: Tensor,
        identity_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """Output length must match the number of nodes."""
        N = grid_2d_positions.shape[0]

        result = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=identity_metric_2d,
            edge_index=delaunay_2d_edges,
        )

        assert result.shape == (N,)

    def test_conformal_factor_changes_values(
        self,
        grid_2d_positions: Tensor,
        diagonal_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """include_conformal_factor=True vs False should give different values."""
        kwargs = {
            "positions": grid_2d_positions,
            "metric_tensors": diagonal_metric_2d,
            "edge_index": delaunay_2d_edges,
        }

        compute_ricci_proxy_full_metric(**kwargs, include_conformal_factor=True)
        compute_ricci_proxy_full_metric(**kwargs, include_conformal_factor=False)

        # For uniform diagonal metric, det is constant, so u is constant,
        # and all differences are zero. Both should be zero here.
        # Use a non-uniform metric to see real differences.
        gen = torch.Generator().manual_seed(77)
        N = 100
        A = torch.randn(N, 2, 2, generator=gen)
        varying_metric = A @ A.transpose(-1, -2) + 0.1 * torch.eye(2).unsqueeze(0)

        with_cf_v = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=varying_metric,
            edge_index=delaunay_2d_edges,
            include_conformal_factor=True,
        )
        without_cf_v = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=varying_metric,
            edge_index=delaunay_2d_edges,
            include_conformal_factor=False,
        )

        # They should generally differ for varying metrics
        assert not torch.allclose(with_cf_v, without_cf_v, atol=1e-8)

    def test_auto_computes_metric_det(
        self,
        grid_2d_positions: Tensor,
        full_spd_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """When metric_det=None, the function computes it from metric_tensors.

        Reproduce the same slogdet-based determinant the function uses internally
        so that the results match exactly.
        """
        # Match the internal computation: slogdet -> exp -> zero out negative sign
        sign, logdet = torch.linalg.slogdet(full_spd_metric_2d)
        det = torch.exp(logdet)
        metric_det = torch.where(sign > 0, det, torch.zeros_like(det))

        result_auto = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=full_spd_metric_2d,
            edge_index=delaunay_2d_edges,
            metric_det=None,
        )
        result_explicit = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=full_spd_metric_2d,
            edge_index=delaunay_2d_edges,
            metric_det=metric_det,
        )

        torch.testing.assert_close(result_auto, result_explicit, atol=1e-5, rtol=1e-5)

    def test_empty_edge_index(
        self,
        grid_2d_positions: Tensor,
        identity_metric_2d: Tensor,
        empty_edge_index: Tensor,
    ) -> None:
        """With no edges, the result should be zeros."""
        N = grid_2d_positions.shape[0]

        result = compute_ricci_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=identity_metric_2d,
            edge_index=empty_edge_index,
        )

        torch.testing.assert_close(result, torch.zeros(N), atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# TestComputeRicciTensorProxy
# ---------------------------------------------------------------------------


class TestComputeRicciTensorProxy:
    """Tests for compute_ricci_tensor_proxy."""

    def test_trace_consistency(self, full_spd_metric_2d: Tensor) -> None:
        """tr(g^{-1} @ R_ij) should equal the Ricci scalar R for each walker."""
        N = full_spd_metric_2d.shape[0]
        d = 2
        gen = torch.Generator().manual_seed(55)
        ricci_scalar = torch.randn(N, generator=gen)

        ricci_tensor = compute_ricci_tensor_proxy(
            metric_tensors=full_spd_metric_2d,
            ricci_scalar=ricci_scalar,
            spatial_dim=d,
        )

        g_inv = torch.linalg.inv(full_spd_metric_2d)
        # tr(g^{-1} R) = sum_ij g^{ij} R_{ji}
        trace = torch.einsum("nij,nji->n", g_inv, ricci_tensor)

        torch.testing.assert_close(trace, ricci_scalar, atol=1e-4, rtol=1e-4)

    def test_output_shape(self, identity_metric_2d: Tensor) -> None:
        """Output shape must be [N, d, d]."""
        N = 100
        d = 2
        ricci_scalar = torch.ones(N)

        result = compute_ricci_tensor_proxy(
            metric_tensors=identity_metric_2d,
            ricci_scalar=ricci_scalar,
            spatial_dim=d,
        )

        assert result.shape == (N, d, d)

    def test_symmetry(self, full_spd_metric_2d: Tensor) -> None:
        """R_ij should be symmetric since g_ij is symmetric."""
        N = full_spd_metric_2d.shape[0]
        gen = torch.Generator().manual_seed(33)
        ricci_scalar = torch.randn(N, generator=gen)

        ricci_tensor = compute_ricci_tensor_proxy(
            metric_tensors=full_spd_metric_2d,
            ricci_scalar=ricci_scalar,
        )

        torch.testing.assert_close(
            ricci_tensor, ricci_tensor.transpose(-1, -2), atol=1e-6, rtol=0.0
        )

    def test_zero_scalar_gives_zero_tensor(self, identity_metric_2d: Tensor) -> None:
        """Zero Ricci scalar should produce a zero Ricci tensor."""
        N = 100
        d = 2
        ricci_scalar = torch.zeros(N)

        ricci_tensor = compute_ricci_tensor_proxy(
            metric_tensors=identity_metric_2d,
            ricci_scalar=ricci_scalar,
            spatial_dim=d,
        )

        torch.testing.assert_close(
            ricci_tensor,
            torch.zeros(N, d, d),
            atol=1e-12,
            rtol=0.0,
        )

    def test_algebraic_formula(self, full_spd_metric_2d: Tensor) -> None:
        """Verify R_ij = (R / d) * g_ij exactly."""
        N = full_spd_metric_2d.shape[0]
        d = 2
        gen = torch.Generator().manual_seed(44)
        ricci_scalar = torch.randn(N, generator=gen)

        ricci_tensor = compute_ricci_tensor_proxy(
            metric_tensors=full_spd_metric_2d,
            ricci_scalar=ricci_scalar,
            spatial_dim=d,
        )

        expected = full_spd_metric_2d * (ricci_scalar / float(d))[:, None, None]

        torch.testing.assert_close(ricci_tensor, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestComputeRicciTensorProxyFullMetric
# ---------------------------------------------------------------------------


class TestComputeRicciTensorProxyFullMetric:
    """Tests for compute_ricci_tensor_proxy_full_metric."""

    def test_returns_tuple_with_correct_shapes(
        self,
        grid_2d_positions: Tensor,
        full_spd_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """Should return (tensor [N,d,d], scalar [N])."""
        N = grid_2d_positions.shape[0]
        d = 2

        ricci_tensor, ricci_scalar = compute_ricci_tensor_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=full_spd_metric_2d,
            edge_index=delaunay_2d_edges,
        )

        assert ricci_tensor.shape == (N, d, d)
        assert ricci_scalar.shape == (N,)

    def test_identity_metric_near_zero(
        self,
        grid_2d_positions: Tensor,
        identity_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """Identity metric on a regular grid should give near-zero curvature."""
        N = grid_2d_positions.shape[0]
        d = 2

        ricci_tensor, ricci_scalar = compute_ricci_tensor_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=identity_metric_2d,
            edge_index=delaunay_2d_edges,
        )

        torch.testing.assert_close(
            ricci_scalar,
            torch.zeros(N),
            atol=1e-4,
            rtol=0.0,
        )
        torch.testing.assert_close(
            ricci_tensor,
            torch.zeros(N, d, d),
            atol=1e-4,
            rtol=0.0,
        )

    def test_symmetry(
        self,
        grid_2d_positions: Tensor,
        full_spd_metric_2d: Tensor,
        delaunay_2d_edges: Tensor,
    ) -> None:
        """The output Ricci tensor should be symmetric: R_ij = R_ji."""
        ricci_tensor, _ = compute_ricci_tensor_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=full_spd_metric_2d,
            edge_index=delaunay_2d_edges,
        )

        torch.testing.assert_close(
            ricci_tensor, ricci_tensor.transpose(-1, -2), atol=1e-6, rtol=0.0
        )

    def test_empty_edge_index(
        self,
        grid_2d_positions: Tensor,
        identity_metric_2d: Tensor,
        empty_edge_index: Tensor,
    ) -> None:
        """With no edges, should return zero tensor and zero scalar."""
        N = grid_2d_positions.shape[0]
        d = 2

        ricci_tensor, ricci_scalar = compute_ricci_tensor_proxy_full_metric(
            positions=grid_2d_positions,
            metric_tensors=identity_metric_2d,
            edge_index=empty_edge_index,
        )

        torch.testing.assert_close(
            ricci_tensor,
            torch.zeros(N, d, d),
            atol=1e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            ricci_scalar,
            torch.zeros(N),
            atol=1e-12,
            rtol=0.0,
        )
