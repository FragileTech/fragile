"""Comprehensive tests for fragile.physics.operators.geodesics."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.geodesics import (
    _batched_min_plus_matmul,
    _normalize_edge_weight_mode,
    _normalize_method,
    batched_floyd_warshall,
    batched_tropical_shortest_paths,
    build_adjacency_from_edges,
    compute_smeared_kernels,
    gather_companion_distances,
    select_scales,
)


# ===================================================================
# Helpers
# ===================================================================


def _chain_adjacency(n: int, weight: float = 1.0) -> Tensor:
    """Build adjacency for chain 0-1-2-..-(n-1) with given weight."""
    adj = torch.full((n, n), float("inf"))
    for i in range(n):
        adj[i, i] = 0.0
    for i in range(n - 1):
        adj[i, i + 1] = weight
        adj[i + 1, i] = weight
    return adj


def _chain_shortest(n: int, weight: float = 1.0) -> Tensor:
    """Expected all-pairs shortest paths for an n-node chain."""
    d = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            d[i, j] = abs(i - j) * weight
    return d


def _triangle_adjacency(w01: float = 1.0, w12: float = 1.0, w02: float = 1.0) -> Tensor:
    """Adjacency for a 3-node triangle."""
    adj = torch.full((3, 3), float("inf"))
    adj.diagonal().zero_()
    adj[0, 1] = adj[1, 0] = w01
    adj[1, 2] = adj[2, 1] = w12
    adj[0, 2] = adj[2, 0] = w02
    return adj


# ===================================================================
# _normalize_method
# ===================================================================


class TestNormalizeMethod:
    """Tests for APSP method alias resolution."""

    @pytest.mark.parametrize("alias", ["floyd", "floyd-warshall", "floyd_warshall"])
    def test_floyd_aliases(self, alias: str) -> None:
        assert _normalize_method(alias, torch.device("cpu")) == "floyd-warshall"

    @pytest.mark.parametrize("alias", ["tropical", "min-plus", "min_plus"])
    def test_tropical_aliases(self, alias: str) -> None:
        assert _normalize_method(alias, torch.device("cpu")) == "tropical"

    def test_auto_cpu(self) -> None:
        assert _normalize_method("auto", torch.device("cpu")) == "floyd-warshall"

    def test_auto_case_insensitive(self) -> None:
        assert _normalize_method("  AUTO  ", torch.device("cpu")) == "floyd-warshall"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown APSP method"):
            _normalize_method("dijkstra", torch.device("cpu"))


# ===================================================================
# _normalize_edge_weight_mode
# ===================================================================


class TestNormalizeEdgeWeightMode:
    """Tests for edge weight mode alias resolution."""

    def test_canonical_modes(self) -> None:
        assert _normalize_edge_weight_mode("uniform") == "uniform"
        assert _normalize_edge_weight_mode("riemannian_kernel") == "riemannian_kernel"

    def test_alias_unit(self) -> None:
        assert _normalize_edge_weight_mode("unit") == "uniform"

    def test_alias_typo(self) -> None:
        assert _normalize_edge_weight_mode("riemanian_kernel") == "riemannian_kernel"

    def test_prefix_removal(self) -> None:
        assert _normalize_edge_weight_mode("edge_weight:uniform") == "uniform"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported edge_weight_mode"):
            _normalize_edge_weight_mode("banana")


# ===================================================================
# build_adjacency_from_edges
# ===================================================================


class TestBuildAdjacencyFromEdges:
    """Tests for dense adjacency construction."""

    def test_empty_edges(self) -> None:
        edges = torch.empty((0, 2), dtype=torch.long)
        adj = build_adjacency_from_edges(3, edges)
        assert adj.shape == (3, 3)
        assert adj.diagonal().sum().item() == 0.0
        # Off-diagonal should all be inf
        mask = ~torch.eye(3, dtype=torch.bool)
        assert torch.all(adj[mask] == float("inf"))

    def test_simple_chain(self, simple_edges: Tensor) -> None:
        adj = build_adjacency_from_edges(5, simple_edges, undirected=True)
        assert adj.shape == (5, 5)
        assert adj[0, 0].item() == 0.0
        assert adj[0, 1].item() == 1.0
        assert adj[1, 0].item() == 1.0
        # Non-adjacent nodes should be inf
        assert adj[0, 2].item() == float("inf")

    def test_triangle(self, triangle_edges: Tensor) -> None:
        adj = build_adjacency_from_edges(3, triangle_edges, undirected=True)
        assert adj.shape == (3, 3)
        for i in range(3):
            assert adj[i, i].item() == 0.0
        assert adj[0, 1].item() == 1.0
        assert adj[1, 2].item() == 1.0
        assert adj[0, 2].item() == 1.0

    def test_weighted_edges(self) -> None:
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        weights = torch.tensor([2.5, 3.0])
        adj = build_adjacency_from_edges(3, edges, edge_weights=weights, undirected=True)
        assert adj[0, 1].item() == pytest.approx(2.5)
        assert adj[1, 0].item() == pytest.approx(2.5)
        assert adj[1, 2].item() == pytest.approx(3.0)

    def test_undirected_symmetry(self) -> None:
        edges = torch.tensor([[0, 1]], dtype=torch.long)
        adj = build_adjacency_from_edges(3, edges, undirected=True)
        assert adj[0, 1].item() == adj[1, 0].item()

    def test_directed(self) -> None:
        edges = torch.tensor([[0, 1]], dtype=torch.long)
        adj = build_adjacency_from_edges(3, edges, undirected=False)
        assert adj[0, 1].item() == 1.0
        assert adj[1, 0].item() == float("inf")

    def test_negative_num_nodes_raises(self) -> None:
        edges = torch.empty((0, 2), dtype=torch.long)
        with pytest.raises(ValueError, match="num_nodes must be non-negative"):
            build_adjacency_from_edges(-1, edges)

    def test_edges_shape_validation(self) -> None:
        edges = torch.tensor([0, 1, 2], dtype=torch.long).reshape(3, 1)
        with pytest.raises(ValueError, match="edges must have shape"):
            build_adjacency_from_edges(3, edges)

    def test_duplicate_edges_keep_min(self) -> None:
        edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        weights = torch.tensor([5.0, 2.0])
        adj = build_adjacency_from_edges(3, edges, edge_weights=weights, undirected=False)
        assert adj[0, 1].item() == pytest.approx(2.0)

    def test_self_loops_excluded(self) -> None:
        edges = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        adj = build_adjacency_from_edges(3, edges, undirected=True)
        assert adj[0, 0].item() == 0.0
        assert adj[0, 1].item() == 1.0

    def test_non_finite_weights_excluded(self) -> None:
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        weights = torch.tensor([float("inf"), 1.0])
        adj = build_adjacency_from_edges(3, edges, edge_weights=weights, undirected=True)
        assert adj[0, 1].item() == float("inf")
        assert adj[1, 2].item() == pytest.approx(1.0)

    def test_negative_weights_excluded(self) -> None:
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        weights = torch.tensor([-1.0, 1.0])
        adj = build_adjacency_from_edges(3, edges, edge_weights=weights, undirected=True)
        assert adj[0, 1].item() == float("inf")
        assert adj[1, 2].item() == pytest.approx(1.0)

    def test_zero_nodes(self) -> None:
        edges = torch.empty((0, 2), dtype=torch.long)
        adj = build_adjacency_from_edges(0, edges)
        assert adj.shape == (0, 0)

    def test_device_argument(self) -> None:
        edges = torch.tensor([[0, 1]], dtype=torch.long)
        adj = build_adjacency_from_edges(3, edges, device="cpu")
        assert adj.device.type == "cpu"

    def test_weight_count_mismatch_raises(self) -> None:
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        weights = torch.tensor([1.0])  # only 1, expect 2
        with pytest.raises(ValueError, match="edge_weights must have one value per edge"):
            build_adjacency_from_edges(3, edges, edge_weights=weights)


# ===================================================================
# batched_floyd_warshall
# ===================================================================


class TestBatchedFloydWarshall:
    """Tests for Floyd-Warshall APSP."""

    def test_identity_already_shortest(self) -> None:
        d = _chain_shortest(4).unsqueeze(0)
        result = batched_floyd_warshall(d)
        assert torch.allclose(result, d)

    def test_chain_graph_known_distances(self) -> None:
        adj = _chain_adjacency(5).unsqueeze(0)
        expected = _chain_shortest(5).unsqueeze(0)
        result = batched_floyd_warshall(adj)
        assert torch.allclose(result, expected)

    def test_disconnected_nodes_stay_inf(self) -> None:
        adj = torch.full((1, 3, 3), float("inf"))
        adj[:, 0, 0] = 0.0
        adj[:, 1, 1] = 0.0
        adj[:, 2, 2] = 0.0
        adj[:, 0, 1] = 1.0
        adj[:, 1, 0] = 1.0
        # node 2 disconnected
        result = batched_floyd_warshall(adj)
        assert result[0, 0, 2].item() == float("inf")
        assert result[0, 2, 0].item() == float("inf")
        assert result[0, 0, 1].item() == pytest.approx(1.0)

    def test_batch_dimension(self) -> None:
        adj1 = _chain_adjacency(4)
        adj2 = _triangle_adjacency()
        # Pad adj2 to 4x4
        adj2_padded = torch.full((4, 4), float("inf"))
        adj2_padded[:3, :3] = adj2
        adj2_padded[3, 3] = 0.0
        batch = torch.stack([adj1, adj2_padded], dim=0)
        result = batched_floyd_warshall(batch)
        assert result.shape == (2, 4, 4)
        # First batch: chain distances
        assert result[0, 0, 3].item() == pytest.approx(3.0)
        # Second batch: triangle shortest path 0->2 = 1.0
        assert result[1, 0, 2].item() == pytest.approx(1.0)

    def test_non_square_raises(self) -> None:
        d = torch.zeros(1, 3, 4)
        with pytest.raises(ValueError, match="square"):
            batched_floyd_warshall(d)

    def test_wrong_ndim_raises(self) -> None:
        d = torch.zeros(3, 3)
        with pytest.raises(ValueError, match="shape"):
            batched_floyd_warshall(d)

    def test_single_node(self) -> None:
        d = torch.zeros(1, 1, 1)
        result = batched_floyd_warshall(d)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0].item() == 0.0

    def test_triangle_shortcut(self) -> None:
        """Triangle with long edge 0-2=10 but short path 0-1-2=2."""
        adj = _triangle_adjacency(w01=1.0, w12=1.0, w02=10.0).unsqueeze(0)
        result = batched_floyd_warshall(adj)
        assert result[0, 0, 2].item() == pytest.approx(2.0)


# ===================================================================
# _batched_min_plus_matmul
# ===================================================================


class TestBatchedMinPlusMatmul:
    """Tests for tropical matrix multiplication."""

    def test_identity_element(self) -> None:
        """Multiplying by the tropical identity (0 diagonal, inf elsewhere) leaves matrix unchanged."""
        n = 4
        identity = torch.full((1, n, n), float("inf"))
        identity[:, range(n), range(n)] = 0.0
        adj = _chain_adjacency(n).unsqueeze(0)
        result = _batched_min_plus_matmul(adj, identity)
        assert torch.allclose(result, adj)

    def test_associativity(self) -> None:
        """(A x B) x C == A x (B x C) in tropical algebra."""
        gen = torch.Generator().manual_seed(123)
        A = torch.rand(2, 4, 4, generator=gen) * 10
        B = torch.rand(2, 4, 4, generator=gen) * 10
        C = torch.rand(2, 4, 4, generator=gen) * 10
        left = _batched_min_plus_matmul(_batched_min_plus_matmul(A, B), C)
        right = _batched_min_plus_matmul(A, _batched_min_plus_matmul(B, C))
        assert torch.allclose(left, right, atol=1e-5)

    def test_block_size_parameter(self) -> None:
        adj = _chain_adjacency(5).unsqueeze(0)
        r1 = _batched_min_plus_matmul(adj, adj, block_size=2)
        r2 = _batched_min_plus_matmul(adj, adj, block_size=64)
        assert torch.allclose(r1, r2, atol=1e-5)

    def test_mismatched_shapes_raise(self) -> None:
        a = torch.zeros(1, 3, 3)
        b = torch.zeros(1, 4, 4)
        with pytest.raises(ValueError, match="same shape"):
            _batched_min_plus_matmul(a, b)

    def test_non_positive_block_size_raises(self) -> None:
        a = torch.zeros(1, 3, 3)
        with pytest.raises(ValueError, match="block_size must be positive"):
            _batched_min_plus_matmul(a, a, block_size=0)

    def test_wrong_ndim_raises(self) -> None:
        a = torch.zeros(3, 3)
        b = torch.zeros(3, 3)
        with pytest.raises(ValueError):
            _batched_min_plus_matmul(a, b)

    def test_non_square_raises(self) -> None:
        a = torch.zeros(1, 3, 4)
        with pytest.raises(ValueError, match="square"):
            _batched_min_plus_matmul(a, a)


# ===================================================================
# batched_tropical_shortest_paths
# ===================================================================


class TestBatchedTropicalShortestPaths:
    """Tests for APSP via repeated tropical squaring."""

    def test_matches_floyd_warshall(self) -> None:
        """Tropical and Floyd-Warshall produce the same results."""
        adj = _chain_adjacency(5).unsqueeze(0)
        fw = batched_floyd_warshall(adj)
        tr = batched_tropical_shortest_paths(adj)
        assert torch.allclose(fw, tr, atol=1e-5)

    def test_matches_floyd_warshall_triangle(self) -> None:
        adj = _triangle_adjacency(w01=1.0, w12=1.0, w02=10.0).unsqueeze(0)
        fw = batched_floyd_warshall(adj)
        tr = batched_tropical_shortest_paths(adj)
        assert torch.allclose(fw, tr, atol=1e-5)

    def test_chain_graph(self) -> None:
        adj = _chain_adjacency(6).unsqueeze(0)
        expected = _chain_shortest(6).unsqueeze(0)
        result = batched_tropical_shortest_paths(adj)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_disconnected_nodes(self) -> None:
        adj = torch.full((1, 3, 3), float("inf"))
        adj[:, range(3), range(3)] = 0.0
        adj[:, 0, 1] = 1.0
        adj[:, 1, 0] = 1.0
        result = batched_tropical_shortest_paths(adj)
        assert result[0, 0, 2].item() == float("inf")

    def test_single_node(self) -> None:
        d = torch.zeros(1, 1, 1)
        result = batched_tropical_shortest_paths(d)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0].item() == 0.0

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            batched_tropical_shortest_paths(torch.zeros(3, 3))

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            batched_tropical_shortest_paths(torch.zeros(1, 3, 4))

    def test_matches_floyd_warshall_random(self) -> None:
        """Both methods agree on a random graph."""
        gen = torch.Generator().manual_seed(77)
        adj = torch.full((2, 6, 6), float("inf"))
        for b in range(2):
            adj[b].diagonal().zero_()
        # Add random edges
        for b in range(2):
            for _ in range(10):
                i = int(torch.randint(0, 6, (1,), generator=gen).item())
                j = int(torch.randint(0, 6, (1,), generator=gen).item())
                if i != j:
                    w = float(torch.rand(1, generator=gen).item()) * 5 + 0.1
                    adj[b, i, j] = min(adj[b, i, j].item(), w)
                    adj[b, j, i] = min(adj[b, j, i].item(), w)
        fw = batched_floyd_warshall(adj)
        tr = batched_tropical_shortest_paths(adj)
        assert torch.allclose(fw, tr, atol=1e-4)


# ===================================================================
# select_scales
# ===================================================================


class TestSelectScales:
    """Tests for automatic scale selection."""

    def test_output_length(self) -> None:
        d = torch.rand(2, 5, 5) * 10 + 0.1
        scales = select_scales(d, n_scales=7)
        assert scales.shape == (7,)

    def test_strictly_increasing(self) -> None:
        d = torch.rand(2, 10, 10) * 10 + 0.1
        scales = select_scales(d, n_scales=5)
        for i in range(1, scales.numel()):
            assert scales[i].item() > scales[i - 1].item()

    def test_ge_min_scale(self) -> None:
        d = torch.rand(2, 5, 5) * 10 + 0.1
        ms = 0.01
        scales = select_scales(d, n_scales=5, min_scale=ms)
        assert torch.all(scales >= ms)

    def test_all_finite(self) -> None:
        d = torch.rand(1, 5, 5) * 10 + 0.1
        scales = select_scales(d, n_scales=5)
        assert torch.all(torch.isfinite(scales))

    def test_invalid_n_scales_raises(self) -> None:
        d = torch.rand(1, 3, 3)
        with pytest.raises(ValueError, match="n_scales must be positive"):
            select_scales(d, n_scales=0)

    def test_invalid_quantile_range_raises(self) -> None:
        d = torch.rand(1, 3, 3)
        with pytest.raises(ValueError, match="q_low"):
            select_scales(d, n_scales=3, q_low=0.9, q_high=0.1)

    def test_invalid_min_scale_raises(self) -> None:
        d = torch.rand(1, 3, 3)
        with pytest.raises(ValueError, match="min_scale must be positive"):
            select_scales(d, n_scales=3, min_scale=-1.0)

    def test_all_inf_input_returns_min_scale_fill(self) -> None:
        d = torch.full((1, 3, 3), float("inf"))
        ms = 1e-4
        scales = select_scales(d, n_scales=4, min_scale=ms)
        assert scales.shape == (4,)
        assert torch.allclose(scales, torch.full((4,), ms))

    def test_all_zero_input_returns_min_scale_fill(self) -> None:
        """All zeros are not > 0, so sampled is empty, returns min_scale fill."""
        d = torch.zeros(1, 3, 3)
        ms = 1e-5
        scales = select_scales(d, n_scales=3, min_scale=ms)
        assert scales.shape == (3,)
        # All values should be >= min_scale
        assert torch.all(scales >= ms)


# ===================================================================
# compute_smeared_kernels
# ===================================================================


class TestComputeSmearedKernels:
    """Tests for smeared kernel construction."""

    @pytest.fixture
    def small_distances(self) -> Tensor:
        """Small 3-node distance matrix [1, 3, 3]."""
        return torch.tensor([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]).unsqueeze(0)

    @pytest.fixture
    def scales_2(self) -> Tensor:
        return torch.tensor([0.5, 2.0])

    def test_gaussian_shape(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(small_distances, scales_2, kernel_type="gaussian")
        assert k.shape == (1, 2, 3, 3)

    def test_exponential_shape(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(small_distances, scales_2, kernel_type="exponential")
        assert k.shape == (1, 2, 3, 3)

    def test_tophat_shape(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(small_distances, scales_2, kernel_type="tophat")
        assert k.shape == (1, 2, 3, 3)

    def test_shell_shape(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(small_distances, scales_2, kernel_type="shell")
        assert k.shape == (1, 2, 3, 3)

    def test_zero_diagonal(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(small_distances, scales_2, kernel_type="gaussian")
        for s in range(2):
            diag = k[0, s].diagonal()
            assert torch.allclose(diag, torch.zeros_like(diag))

    def test_row_normalization(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(
            small_distances, scales_2, kernel_type="gaussian", normalize_rows=True
        )
        row_sums = k.sum(dim=-1)
        # Each row with non-zero entries should sum to 1
        for t in range(k.shape[0]):
            for s in range(k.shape[1]):
                for i in range(k.shape[2]):
                    rs = row_sums[t, s, i].item()
                    if rs > 0:
                        assert rs == pytest.approx(1.0, abs=1e-6)

    def test_no_normalization(self, small_distances: Tensor, scales_2: Tensor) -> None:
        k = compute_smeared_kernels(
            small_distances, scales_2, kernel_type="gaussian", normalize_rows=False
        )
        k.sum(dim=-1)
        # With no normalization, rows generally won't sum to 1
        # Just check that values are non-negative
        assert torch.all(k >= 0)

    def test_non_square_raises(self) -> None:
        d = torch.zeros(1, 3, 4)
        s = torch.tensor([1.0])
        with pytest.raises(ValueError, match="square"):
            compute_smeared_kernels(d, s)

    def test_empty_scales_raises(self) -> None:
        d = torch.zeros(1, 3, 3)
        s = torch.tensor([])
        with pytest.raises(ValueError, match="non-empty"):
            compute_smeared_kernels(d, s)

    def test_unknown_kernel_raises(self, small_distances: Tensor) -> None:
        s = torch.tensor([1.0])
        with pytest.raises(ValueError, match="Unknown kernel_type"):
            compute_smeared_kernels(small_distances, s, kernel_type="laplacian")

    def test_inf_distances_produce_zero(self) -> None:
        d = torch.full((1, 3, 3), float("inf"))
        d[:, range(3), range(3)] = 0.0
        s = torch.tensor([1.0])
        k = compute_smeared_kernels(d, s, kernel_type="gaussian", normalize_rows=False)
        # Off-diagonal should all be 0 since distances are inf
        mask = ~torch.eye(3, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        assert torch.allclose(k[mask], torch.zeros_like(k[mask]))

    def test_wrong_ndim_raises(self) -> None:
        d = torch.zeros(3, 3)
        s = torch.tensor([1.0])
        with pytest.raises(ValueError, match="shape"):
            compute_smeared_kernels(d, s)

    def test_negative_scale_raises(self, small_distances: Tensor) -> None:
        s = torch.tensor([-1.0])
        with pytest.raises(ValueError, match="positive"):
            compute_smeared_kernels(small_distances, s)

    def test_tophat_values(self) -> None:
        """Tophat: 1 where dist <= scale, 0 otherwise (except diagonal)."""
        d = torch.tensor([
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ]).unsqueeze(0)
        s = torch.tensor([2.0])
        k = compute_smeared_kernels(d, s, kernel_type="tophat", normalize_rows=False)
        # d[0,1]=1 <= 2 -> 1, d[0,2]=3 > 2 -> 0 (diagonal is forced 0)
        assert k[0, 0, 0, 1].item() == pytest.approx(1.0)
        assert k[0, 0, 0, 2].item() == pytest.approx(0.0)
        assert k[0, 0, 0, 0].item() == pytest.approx(0.0)  # diagonal


# ===================================================================
# gather_companion_distances
# ===================================================================


class TestGatherCompanionDistances:
    """Tests for companion distance extraction."""

    def test_known_values(self) -> None:
        """d_ij, d_ik, d_jk for a known distance matrix."""
        _T, _N = 1, 3
        d = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 3.0],
                    [2.0, 3.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        # companion_distance: walker 0->1, 1->2, 2->0
        comp_dist = torch.tensor([[1, 2, 0]], dtype=torch.long)
        # companion_clone: walker 0->2, 1->0, 2->1
        comp_clone = torch.tensor([[2, 0, 1]], dtype=torch.long)

        d_ij, d_ik, d_jk = gather_companion_distances(d, comp_dist, comp_clone)

        assert d_ij.shape == (1, 3)
        assert d_ik.shape == (1, 3)
        assert d_jk.shape == (1, 3)

        # Walker 0: j=1, k=2 -> d(0,1)=1, d(0,2)=2, d(1,2)=3
        assert d_ij[0, 0].item() == pytest.approx(1.0)
        assert d_ik[0, 0].item() == pytest.approx(2.0)
        assert d_jk[0, 0].item() == pytest.approx(3.0)

        # Walker 1: j=2, k=0 -> d(1,2)=3, d(1,0)=1, d(2,0)=2
        assert d_ij[0, 1].item() == pytest.approx(3.0)
        assert d_ik[0, 1].item() == pytest.approx(1.0)
        assert d_jk[0, 1].item() == pytest.approx(2.0)

        # Walker 2: j=0, k=1 -> d(2,0)=2, d(2,1)=3, d(0,1)=1
        assert d_ij[0, 2].item() == pytest.approx(2.0)
        assert d_ik[0, 2].item() == pytest.approx(3.0)
        assert d_jk[0, 2].item() == pytest.approx(1.0)

    def test_self_companion_gives_zero(self) -> None:
        _T, _N = 1, 3
        d = torch.tensor(
            [
                [
                    [0.0, 5.0, 7.0],
                    [5.0, 0.0, 3.0],
                    [7.0, 3.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        # All companions point to self
        comp = torch.tensor([[0, 1, 2]], dtype=torch.long)
        d_ij, d_ik, d_jk = gather_companion_distances(d, comp, comp)
        # d(i, i) = 0 for all
        assert torch.allclose(d_ij, torch.zeros(1, 3))
        assert torch.allclose(d_ik, torch.zeros(1, 3))
        # d_jk also 0 because j==k==i, so d(i,i)=0
        assert torch.allclose(d_jk, torch.zeros(1, 3))

    def test_shapes_with_conftest_fixtures(
        self,
        T: int,
        N: int,
        companions_distance: Tensor,
        companions_clone: Tensor,
    ) -> None:
        """Use conftest fixtures to verify output shapes."""
        d = torch.rand(T, N, N)
        d_ij, d_ik, d_jk = gather_companion_distances(d, companions_distance, companions_clone)
        assert d_ij.shape == (T, N)
        assert d_ik.shape == (T, N)
        assert d_jk.shape == (T, N)

    def test_batch_consistency(self) -> None:
        """Each time-frame extracts independently."""
        T, N = 2, 3
        d = torch.rand(T, N, N)
        d = (d + d.transpose(-1, -2)) / 2  # symmetrize
        d[:, range(N), range(N)] = 0.0
        comp_dist = torch.tensor([[1, 2, 0], [2, 0, 1]], dtype=torch.long)
        comp_clone = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)
        d_ij, d_ik, d_jk = gather_companion_distances(d, comp_dist, comp_clone)
        # Manually check frame 0, walker 0: j=1, k=2
        assert d_ij[0, 0].item() == pytest.approx(d[0, 0, 1].item())
        assert d_ik[0, 0].item() == pytest.approx(d[0, 0, 2].item())
        assert d_jk[0, 0].item() == pytest.approx(d[0, 1, 2].item())
        # Frame 1, walker 0: j=2, k=1
        assert d_ij[1, 0].item() == pytest.approx(d[1, 0, 2].item())
        assert d_ik[1, 0].item() == pytest.approx(d[1, 0, 1].item())
        assert d_jk[1, 0].item() == pytest.approx(d[1, 2, 1].item())


# ===================================================================
# Integration: build_adjacency -> APSP -> kernels
# ===================================================================


class TestIntegration:
    """End-to-end integration tests combining multiple functions."""

    def test_adjacency_to_distances_to_kernels(self) -> None:
        """Full pipeline: edges -> adjacency -> distances -> scales -> kernels."""
        n = 5
        edges = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            dtype=torch.long,
        )
        adj = build_adjacency_from_edges(n, edges, undirected=True)
        adj_batch = adj.unsqueeze(0)
        distances = batched_floyd_warshall(adj_batch)
        scales = select_scales(distances, n_scales=3)
        kernels = compute_smeared_kernels(distances, scales, kernel_type="gaussian")

        assert kernels.shape == (1, 3, n, n)
        assert torch.all(kernels >= 0)
        # Diagonal should be zero
        for s in range(3):
            diag = kernels[0, s].diagonal()
            assert torch.allclose(diag, torch.zeros_like(diag))

    def test_floyd_vs_tropical_on_random_graph(self) -> None:
        """Both APSP methods agree on a non-trivial graph."""
        gen = torch.Generator().manual_seed(42)
        n = 8
        adj = torch.full((1, n, n), float("inf"))
        adj[:, range(n), range(n)] = 0.0
        for i in range(n - 1):
            w = float(torch.rand(1, generator=gen).item()) * 3 + 0.5
            adj[0, i, i + 1] = w
            adj[0, i + 1, i] = w
        # Add a few cross edges
        for _ in range(5):
            i = int(torch.randint(0, n, (1,), generator=gen).item())
            j = int(torch.randint(0, n, (1,), generator=gen).item())
            if i != j:
                w = float(torch.rand(1, generator=gen).item()) * 5 + 0.1
                adj[0, i, j] = min(adj[0, i, j].item(), w)
                adj[0, j, i] = min(adj[0, j, i].item(), w)

        fw = batched_floyd_warshall(adj)
        tr = batched_tropical_shortest_paths(adj)
        assert torch.allclose(fw, tr, atol=1e-4)
