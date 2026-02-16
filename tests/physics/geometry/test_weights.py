"""Tests for fragile.physics.geometry.weights module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.weights import (
    clamp_metric_eigenvalues,
    compute_edge_weights,
    compute_gaussian_kernel_weights,
    compute_inverse_distance_weights,
    compute_inverse_riemannian_distance_weights,
    compute_inverse_volume_weights,
    compute_riemannian_kernel_weights,
    compute_riemannian_volumes,
    compute_uniform_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_source_sums(edge_index: Tensor, weights: Tensor, n_nodes: int) -> Tensor:
    """Sum weights per source node."""
    sums = torch.zeros(n_nodes, dtype=weights.dtype)
    sums.scatter_add_(0, edge_index[0], weights)
    return sums


def _source_nodes_with_edges(edge_index: Tensor, n_nodes: int) -> Tensor:
    """Return boolean mask of nodes that appear as source at least once."""
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[edge_index[0].unique()] = True
    return mask


# ===========================================================================
# TestComputeUniformWeights
# ===========================================================================


class TestComputeUniformWeights:
    """Tests for compute_uniform_weights."""

    def test_equal_per_source_when_normalized(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """All edges from the same source should have equal weight."""
        N = grid_2d_positions.shape[0]
        edge_index = delaunay_2d_edges
        w = compute_uniform_weights(edge_index, n_nodes=N, normalize=True)

        src = edge_index[0]
        for node in src.unique().tolist():
            mask = src == node
            node_weights = w[mask]
            assert torch.allclose(
                node_weights, node_weights[0].expand_as(node_weights), atol=1e-6
            ), f"Weights from source {node} are not equal."

    def test_normalized_sum_per_source(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Normalized weights from each source should sum to 1."""
        N = grid_2d_positions.shape[0]
        edge_index = delaunay_2d_edges
        w = compute_uniform_weights(edge_index, n_nodes=N, normalize=True)

        sums = _per_source_sums(edge_index, w, N)
        active = _source_nodes_with_edges(edge_index, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-5)

    def test_alive_mask_zeros_dead_edges(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Edges touching dead nodes should get zero weight."""
        N = grid_2d_positions.shape[0]
        edge_index = delaunay_2d_edges
        alive = torch.ones(N, dtype=torch.bool)
        # Kill node 0
        alive[0] = False

        w = compute_uniform_weights(edge_index, n_nodes=N, alive=alive, normalize=True)

        src, dst = edge_index
        dead_edges = (~alive[src]) | (~alive[dst])
        assert (w[dead_edges] == 0.0).all(), "Dead-endpoint edges should have zero weight."


# ===========================================================================
# TestComputeInverseDistanceWeights
# ===========================================================================


class TestComputeInverseDistanceWeights:
    """Tests for compute_inverse_distance_weights."""

    def test_positive_weights(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        w = compute_inverse_distance_weights(grid_2d_positions, delaunay_2d_edges, normalize=False)
        assert (w > 0).all()

    def test_closer_neighbors_higher_weight(self):
        """Closer neighbor should receive higher weight than farther one."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        # Edges: 0->1 and 0->2
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        w = compute_inverse_distance_weights(positions, edge_index, normalize=False)
        # Edge 0->1 (distance 1) should have higher weight than 0->2 (distance 3)
        assert w[0] > w[1], "Closer neighbor should get higher weight."

    def test_precomputed_distances_match(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Passing precomputed edge_distances should give same result."""
        pos = grid_2d_positions
        ei = delaunay_2d_edges
        src, dst = ei

        dists = torch.norm(pos[dst] - pos[src], dim=1)
        w_auto = compute_inverse_distance_weights(pos, ei, normalize=True)
        w_precomp = compute_inverse_distance_weights(pos, ei, edge_distances=dists, normalize=True)
        assert torch.allclose(w_auto, w_precomp, atol=1e-6)

    def test_normalized_sum_per_source(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        N = grid_2d_positions.shape[0]
        w = compute_inverse_distance_weights(grid_2d_positions, delaunay_2d_edges, normalize=True)
        sums = _per_source_sums(delaunay_2d_edges, w, N)
        active = _source_nodes_with_edges(delaunay_2d_edges, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-5)


# ===========================================================================
# TestComputeInverseVolumeWeights
# ===========================================================================


class TestComputeInverseVolumeWeights:
    """Tests for compute_inverse_volume_weights."""

    def test_positive_weights(self, delaunay_2d_edges: Tensor):
        N = 100
        cell_volumes = torch.rand(N) + 0.1  # positive volumes
        w = compute_inverse_volume_weights(delaunay_2d_edges, cell_volumes, normalize=False)
        assert (w > 0).all()

    def test_smaller_volume_higher_weight(self):
        """Destination with smaller volume should get higher weight."""
        # 3 nodes, edges 0->1 and 0->2
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        cell_volumes = torch.tensor([1.0, 0.5, 2.0])
        w = compute_inverse_volume_weights(edge_index, cell_volumes, normalize=False)
        # vol[1]=0.5 < vol[2]=2.0, so w[0] (0->1) > w[1] (0->2)
        assert w[0] > w[1], "Smaller destination volume should yield higher weight."

    def test_normalized_sum_per_source(self, delaunay_2d_edges: Tensor):
        N = 100
        cell_volumes = torch.rand(N) + 0.1
        w = compute_inverse_volume_weights(delaunay_2d_edges, cell_volumes, normalize=True)
        sums = _per_source_sums(delaunay_2d_edges, w, N)
        active = _source_nodes_with_edges(delaunay_2d_edges, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-5)


# ===========================================================================
# TestClampMetricEigenvalues
# ===========================================================================


class TestClampMetricEigenvalues:
    """Tests for clamp_metric_eigenvalues."""

    def test_noop_when_no_bounds(self, identity_metric_2d: Tensor):
        """With min_eig=None and max_eig=None, returns same tensor."""
        result = clamp_metric_eigenvalues(identity_metric_2d, min_eig=None, max_eig=None)
        assert torch.equal(result, identity_metric_2d)

    def test_min_eig_clamping(self):
        """Eigenvalues should be >= min_eig after clamping."""
        # Construct metric with known small eigenvalue
        m = torch.diag(torch.tensor([0.001, 1.0]))
        metrics = m.unsqueeze(0).expand(5, -1, -1).clone()

        clamped = clamp_metric_eigenvalues(metrics, min_eig=0.1)
        eigvals = torch.linalg.eigvalsh(clamped)
        assert (eigvals >= 0.1 - 1e-6).all(), "All eigenvalues should be >= min_eig."

    def test_max_eig_clamping(self):
        """Eigenvalues should be <= max_eig after clamping."""
        m = torch.diag(torch.tensor([1.0, 100.0]))
        metrics = m.unsqueeze(0).expand(5, -1, -1).clone()

        clamped = clamp_metric_eigenvalues(metrics, max_eig=10.0)
        eigvals = torch.linalg.eigvalsh(clamped)
        assert (eigvals <= 10.0 + 1e-6).all(), "All eigenvalues should be <= max_eig."

    def test_symmetry_preserved(self, full_spd_metric_2d: Tensor):
        """Output should remain symmetric after clamping."""
        clamped = clamp_metric_eigenvalues(full_spd_metric_2d, min_eig=0.01, max_eig=50.0)
        assert torch.allclose(clamped, clamped.transpose(-1, -2), atol=1e-5), (
            "Clamped metrics should be symmetric."
        )


# ===========================================================================
# TestComputeRiemannianVolumes
# ===========================================================================


class TestComputeRiemannianVolumes:
    """Tests for compute_riemannian_volumes."""

    def test_identity_metric_preserves_volume(self, identity_metric_2d: Tensor):
        """With identity metric, Riemannian volume equals Euclidean volume."""
        N = identity_metric_2d.shape[0]
        cell_volumes = torch.rand(N) + 0.1
        vol_r = compute_riemannian_volumes(cell_volumes, identity_metric_2d)
        assert torch.allclose(vol_r, cell_volumes, atol=1e-5)

    def test_scaled_metric(self):
        """With metric c*I, vol_R = c^(d/2) * vol_E."""
        d = 2
        c = 4.0
        N = 10
        metrics = c * torch.eye(d).unsqueeze(0).expand(N, -1, -1)
        cell_volumes = torch.ones(N)

        vol_r = compute_riemannian_volumes(cell_volumes, metrics)
        expected = cell_volumes * (c ** (d / 2))
        assert torch.allclose(vol_r, expected, atol=1e-5)

    def test_always_positive(self, full_spd_metric_2d: Tensor):
        """Riemannian volumes should always be positive for SPD metrics."""
        N = full_spd_metric_2d.shape[0]
        cell_volumes = torch.rand(N) + 0.1
        vol_r = compute_riemannian_volumes(cell_volumes, full_spd_metric_2d)
        assert (vol_r > 0).all()


# ===========================================================================
# TestComputeInverseRiemannianDistanceWeights
# ===========================================================================


class TestComputeInverseRiemannianDistanceWeights:
    """Tests for compute_inverse_riemannian_distance_weights."""

    def test_identity_metric_matches_euclidean(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        """With identity metric, should match Euclidean inverse distance weights."""
        pos = grid_2d_positions
        ei = delaunay_2d_edges

        w_euclid = compute_inverse_distance_weights(pos, ei, normalize=True)
        w_riemann = compute_inverse_riemannian_distance_weights(
            pos,
            ei,
            metric_tensors=identity_metric_2d,
            normalize=True,
            symmetrize_metric=True,
            min_eig=None,
            max_eig=None,
        )
        assert torch.allclose(w_euclid, w_riemann, atol=1e-4)

    def test_positive_weights(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        w = compute_inverse_riemannian_distance_weights(
            grid_2d_positions,
            delaunay_2d_edges,
            metric_tensors=identity_metric_2d,
            normalize=False,
            min_eig=None,
            max_eig=None,
        )
        assert (w > 0).all()

    def test_normalized_per_source(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        N = grid_2d_positions.shape[0]
        w = compute_inverse_riemannian_distance_weights(
            grid_2d_positions,
            delaunay_2d_edges,
            metric_tensors=identity_metric_2d,
            normalize=True,
            min_eig=None,
            max_eig=None,
        )
        sums = _per_source_sums(delaunay_2d_edges, w, N)
        active = _source_nodes_with_edges(delaunay_2d_edges, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-5)


# ===========================================================================
# TestComputeGaussianKernelWeights
# ===========================================================================


class TestComputeGaussianKernelWeights:
    """Tests for compute_gaussian_kernel_weights."""

    def test_positive_weights(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        w = compute_gaussian_kernel_weights(grid_2d_positions, delaunay_2d_edges, normalize=False)
        assert (w > 0).all()

    def test_closer_higher_weight(self):
        """Closer neighbor should receive higher Gaussian weight."""
        positions = torch.tensor([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        w = compute_gaussian_kernel_weights(
            positions, edge_index, length_scale=1.0, normalize=False
        )
        assert w[0] > w[1], "Closer neighbor should get higher Gaussian weight."

    def test_larger_length_scale_more_uniform(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor
    ):
        """Larger length_scale should yield more uniform (less varied) weights."""
        pos = grid_2d_positions
        ei = delaunay_2d_edges

        w_small = compute_gaussian_kernel_weights(pos, ei, length_scale=0.5, normalize=False)
        w_large = compute_gaussian_kernel_weights(pos, ei, length_scale=50.0, normalize=False)

        # Coefficient of variation (std/mean) should be smaller for large scale
        cv_small = w_small.std() / (w_small.mean() + 1e-12)
        cv_large = w_large.std() / (w_large.mean() + 1e-12)
        assert cv_large < cv_small, "Larger length_scale should give more uniform weights."

    def test_normalized_sum_per_source(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        N = grid_2d_positions.shape[0]
        w = compute_gaussian_kernel_weights(grid_2d_positions, delaunay_2d_edges, normalize=True)
        sums = _per_source_sums(delaunay_2d_edges, w, N)
        active = _source_nodes_with_edges(delaunay_2d_edges, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-5)


# ===========================================================================
# TestComputeRiemannianKernelWeights
# ===========================================================================


class TestComputeRiemannianKernelWeights:
    """Tests for compute_riemannian_kernel_weights."""

    def test_identity_metric_matches_euclidean_kernel(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        """With identity metric, should match Euclidean Gaussian kernel."""
        pos = grid_2d_positions
        ei = delaunay_2d_edges
        ls = 1.0

        w_euclid = compute_gaussian_kernel_weights(pos, ei, length_scale=ls, normalize=True)
        w_riemann = compute_riemannian_kernel_weights(
            pos,
            ei,
            length_scale=ls,
            metric_tensors=identity_metric_2d,
            normalize=True,
            symmetrize_metric=True,
            min_eig=None,
            max_eig=None,
        )
        assert torch.allclose(w_euclid, w_riemann, atol=1e-4)

    def test_positive_weights(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        w = compute_riemannian_kernel_weights(
            grid_2d_positions,
            delaunay_2d_edges,
            metric_tensors=identity_metric_2d,
            normalize=False,
            min_eig=None,
            max_eig=None,
        )
        assert (w > 0).all()

    def test_normalized_per_source(
        self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor, identity_metric_2d: Tensor
    ):
        N = grid_2d_positions.shape[0]
        w = compute_riemannian_kernel_weights(
            grid_2d_positions,
            delaunay_2d_edges,
            metric_tensors=identity_metric_2d,
            normalize=True,
            min_eig=None,
            max_eig=None,
        )
        sums = _per_source_sums(delaunay_2d_edges, w, N)
        active = _source_nodes_with_edges(delaunay_2d_edges, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-5)


# ===========================================================================
# TestComputeEdgeWeightsDispatcher
# ===========================================================================


ALL_WEIGHT_MODES = [
    "uniform",
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_volume",
    "inverse_riemannian_distance",
    "kernel",
    "riemannian_kernel",
    "riemannian_kernel_volume",
]

# Modes that require cell_volumes
_VOLUME_MODES = {"inverse_volume", "inverse_riemannian_volume", "riemannian_kernel_volume"}
# Modes that require metric_tensors
_METRIC_MODES = {
    "inverse_riemannian_volume",
    "inverse_riemannian_distance",
    "riemannian_kernel",
    "riemannian_kernel_volume",
}


class TestComputeEdgeWeightsDispatcher:
    """Tests for the compute_edge_weights dispatcher."""

    @pytest.mark.parametrize("mode", ALL_WEIGHT_MODES)
    def test_output_length_equals_num_edges(
        self,
        mode: str,
        grid_2d_positions: Tensor,
        delaunay_2d_edges: Tensor,
        identity_metric_2d: Tensor,
    ):
        """All modes should produce output of length E."""
        N = grid_2d_positions.shape[0]
        E = delaunay_2d_edges.shape[1]
        kwargs: dict = {}
        if mode in _VOLUME_MODES:
            kwargs["cell_volumes"] = torch.ones(N)
        if mode in _METRIC_MODES:
            kwargs["metric_tensors"] = identity_metric_2d
            kwargs["riemannian_min_eig"] = None

        w = compute_edge_weights(grid_2d_positions, delaunay_2d_edges, mode=mode, **kwargs)
        assert w.shape == (E,), f"Expected shape ({E},) but got {w.shape} for mode={mode}."

    def test_unknown_mode_raises(self, grid_2d_positions: Tensor, delaunay_2d_edges: Tensor):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown weight mode"):
            compute_edge_weights(grid_2d_positions, delaunay_2d_edges, mode="nonexistent_mode")

    @pytest.mark.parametrize("mode", ALL_WEIGHT_MODES)
    def test_all_modes_positive_weights(
        self,
        mode: str,
        grid_2d_positions: Tensor,
        delaunay_2d_edges: Tensor,
        identity_metric_2d: Tensor,
    ):
        """All modes should produce positive weights (when normalized, at least non-negative)."""
        N = grid_2d_positions.shape[0]
        kwargs: dict = {"normalize": False}
        if mode in _VOLUME_MODES:
            kwargs["cell_volumes"] = torch.ones(N)
        if mode in _METRIC_MODES:
            kwargs["metric_tensors"] = identity_metric_2d
            kwargs["riemannian_min_eig"] = None

        w = compute_edge_weights(grid_2d_positions, delaunay_2d_edges, mode=mode, **kwargs)
        assert (w > 0).all(), f"All weights should be positive for mode={mode}."

    @pytest.mark.parametrize("mode", ALL_WEIGHT_MODES)
    def test_normalized_per_source_sum(
        self,
        mode: str,
        grid_2d_positions: Tensor,
        delaunay_2d_edges: Tensor,
        identity_metric_2d: Tensor,
    ):
        """Normalized weights from each source should sum to approximately 1."""
        N = grid_2d_positions.shape[0]
        kwargs: dict = {"normalize": True}
        if mode in _VOLUME_MODES:
            kwargs["cell_volumes"] = torch.ones(N)
        if mode in _METRIC_MODES:
            kwargs["metric_tensors"] = identity_metric_2d
            kwargs["riemannian_min_eig"] = None

        w = compute_edge_weights(grid_2d_positions, delaunay_2d_edges, mode=mode, **kwargs)
        sums = _per_source_sums(delaunay_2d_edges, w, N)
        active = _source_nodes_with_edges(delaunay_2d_edges, N)
        assert torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1e-4), (
            f"Per-source sums should be ~1 for mode={mode}."
        )
