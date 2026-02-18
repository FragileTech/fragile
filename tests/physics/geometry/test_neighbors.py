"""Tests for fragile.physics.geometry.neighbors module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.geometry.neighbors import (
    _edges_to_neighbor_matrix,
    _select_alive_samples_and_neighbors,
    build_csr_from_coo,
    query_walker_neighbors_vectorized,
)


# =============================================================================
# TestBuildCsrFromCoo
# =============================================================================


class TestBuildCsrFromCoo:
    """Tests for COO -> CSR conversion."""

    def test_triangle_graph_ptr_and_indices(self):
        """Known 3-node triangle: verify ptr and indices exactly."""
        # Triangle: 0-1, 0-2, 1-2 (symmetric)
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.long,
        )
        csr = build_csr_from_coo(edge_index, n_nodes=3)

        # Each node has exactly 2 neighbors
        expected_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.long)
        torch.testing.assert_close(csr["csr_ptr"], expected_ptr)

        # After sorting by source, targets for node 0 are {1,2}, node 1 are {0,2}, node 2 are {0,1}
        assert csr["csr_indices"].shape[0] == 6
        # Check per-node neighbor sets
        for node in range(3):
            start = csr["csr_ptr"][node].item()
            end = csr["csr_ptr"][node + 1].item()
            neighbors = set(csr["csr_indices"][start:end].tolist())
            expected_neighbors = {0, 1, 2} - {node}
            assert (
                neighbors == expected_neighbors
            ), f"Node {node}: {neighbors} != {expected_neighbors}"

    def test_empty_graph(self):
        """Empty graph (no edges): ptr all zeros, indices empty."""
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        csr = build_csr_from_coo(edge_index, n_nodes=4)

        expected_ptr = torch.zeros(5, dtype=torch.long)
        torch.testing.assert_close(csr["csr_ptr"], expected_ptr)
        assert csr["csr_indices"].shape[0] == 0

    def test_disconnected_nodes(self):
        """Nodes without edges have equal consecutive ptr entries."""
        # Only edge: 0 -> 1 and 1 -> 0; nodes 2, 3 are disconnected
        edge_index = torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        )
        csr = build_csr_from_coo(edge_index, n_nodes=4)

        assert csr["csr_ptr"].shape[0] == 5  # N+1
        # Node 0 has 1 neighbor, node 1 has 1 neighbor, nodes 2 and 3 have 0
        assert csr["csr_ptr"][2].item() == csr["csr_ptr"][3].item()
        assert csr["csr_ptr"][3].item() == csr["csr_ptr"][4].item()
        assert csr["csr_indices"].shape[0] == 2

    def test_edge_distances_reordering(self):
        """Edge distances are reordered consistently with sorted edges."""
        # Edges intentionally out of source order: sources are [2, 0, 1]
        edge_index = torch.tensor(
            [[2, 0, 1], [0, 1, 2]],
            dtype=torch.long,
        )
        distances = torch.tensor([10.0, 20.0, 30.0])

        csr = build_csr_from_coo(edge_index, n_nodes=3, edge_distances=distances)

        # After sorting by source: source order becomes [0, 1, 2]
        # Original edge at position 1 (src=0, dist=20) comes first,
        # then position 2 (src=1, dist=30), then position 0 (src=2, dist=10)
        expected_distances = torch.tensor([20.0, 30.0, 10.0])
        torch.testing.assert_close(csr["csr_distances"], expected_distances)

    def test_ptr_length_and_indices_length(self):
        """ptr has length N+1 and indices has length E always."""
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.long,
        )
        n_nodes = 5  # More nodes than used in edges
        csr = build_csr_from_coo(edge_index, n_nodes=n_nodes)

        assert csr["csr_ptr"].shape[0] == n_nodes + 1
        assert csr["csr_indices"].shape[0] == edge_index.shape[1]

    def test_roundtrip_csr_to_query(self):
        """CSR -> query_walker_neighbors -> verify same neighbor sets as original edge_index."""
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.long,
        )
        csr = build_csr_from_coo(edge_index, n_nodes=3)
        neighbors, mask, _counts = query_walker_neighbors_vectorized(
            csr["csr_ptr"],
            csr["csr_indices"],
        )

        for node in range(3):
            valid_neighbors = neighbors[node][mask[node]].tolist()
            # Build expected from original edge_index
            expected = edge_index[1, edge_index[0] == node].tolist()
            assert sorted(valid_neighbors) == sorted(expected)


# =============================================================================
# TestQueryWalkerNeighborsVectorized
# =============================================================================


class TestQueryWalkerNeighborsVectorized:
    """Tests for vectorized neighbor queries from CSR format."""

    def test_known_graph_neighbors(self, simple_csr):
        """Verify neighbor values for each node in the simple_csr fixture."""
        neighbors, mask, _counts = query_walker_neighbors_vectorized(
            simple_csr["csr_ptr"],
            simple_csr["csr_indices"],
        )

        # Graph: 0-1, 0-2, 1-2, 2-3, 3-4 (symmetric)
        expected_neighbors = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1, 3},
            3: {2, 4},
            4: {3},
        }
        for node, expected_set in expected_neighbors.items():
            valid = neighbors[node][mask[node]].tolist()
            assert (
                set(valid) == expected_set
            ), f"Node {node}: got {set(valid)}, expected {expected_set}"

    def test_padding_where_mask_false(self, simple_csr):
        """pad_value=-1 appears where mask is False."""
        neighbors, mask, _counts = query_walker_neighbors_vectorized(
            simple_csr["csr_ptr"],
            simple_csr["csr_indices"],
            pad_value=-1,
        )
        # Where mask is False, neighbors should be -1
        if (~mask).any():
            assert (neighbors[~mask] == -1).all()

    def test_mask_true_iff_not_pad(self, simple_csr):
        """mask[i,j] is True iff neighbors[i,j] != pad_value."""
        pad_value = -1
        neighbors, mask, _counts = query_walker_neighbors_vectorized(
            simple_csr["csr_ptr"],
            simple_csr["csr_indices"],
            pad_value=pad_value,
        )
        expected_mask = neighbors != pad_value
        torch.testing.assert_close(mask, expected_mask)

    def test_counts_equal_mask_sum(self, simple_csr):
        """counts == mask.sum(dim=1) for each node."""
        _neighbors, mask, counts = query_walker_neighbors_vectorized(
            simple_csr["csr_ptr"],
            simple_csr["csr_indices"],
        )
        expected_counts = mask.sum(dim=1)
        torch.testing.assert_close(counts, expected_counts)

    def test_type_filtering(self):
        """Create typed edges, filter by type=0, verify only type-0 neighbors."""
        # 3-node graph: edges 0->1 (type 0), 0->2 (type 1), 1->0 (type 0)
        edge_index = torch.tensor(
            [[0, 0, 1], [1, 2, 0]],
            dtype=torch.long,
        )
        edge_types = torch.tensor([0, 1, 0], dtype=torch.long)
        csr = build_csr_from_coo(edge_index, n_nodes=3, edge_types=edge_types)

        neighbors, mask, counts = query_walker_neighbors_vectorized(
            csr["csr_ptr"],
            csr["csr_indices"],
            csr_types=csr["csr_types"],
            filter_type=0,
        )

        # Node 0: only type-0 neighbor is 1 (edge 0->2 is type 1, filtered out)
        valid_0 = neighbors[0][mask[0]].tolist()
        assert set(valid_0) == {1}
        # Node 1: type-0 neighbor is 0
        valid_1 = neighbors[1][mask[1]].tolist()
        assert set(valid_1) == {0}
        # Node 2: no outgoing edges, so no type-0 neighbors
        assert counts[2].item() == 0

    def test_empty_graph_zero_counts(self):
        """Empty graph: returns empty neighbor matrix, zero counts."""
        csr_ptr = torch.zeros(4, dtype=torch.long)  # 3 nodes, no edges
        csr_indices = torch.empty(0, dtype=torch.long)

        neighbors, mask, counts = query_walker_neighbors_vectorized(csr_ptr, csr_indices)

        assert neighbors.shape[0] == 3
        assert neighbors.shape[1] == 0
        assert mask.shape == neighbors.shape
        torch.testing.assert_close(counts, torch.zeros(3, dtype=torch.long))

    def test_custom_pad_value(self, simple_csr):
        """Use pad_value=-99, verify padding uses that value."""
        neighbors, mask, _counts = query_walker_neighbors_vectorized(
            simple_csr["csr_ptr"],
            simple_csr["csr_indices"],
            pad_value=-99,
        )
        if (~mask).any():
            assert (neighbors[~mask] == -99).all()
        # Valid entries should not be -99
        if mask.any():
            assert (neighbors[mask] != -99).all()


# =============================================================================
# TestEdgesToNeighborMatrix
# =============================================================================


class TestEdgesToNeighborMatrix:
    """Tests for edge list to [N, k] neighbor matrix conversion."""

    def test_basic_conversion(self):
        """Basic conversion from edge list to [N, k] matrix."""
        # Simple chain: 0->1, 1->0, 1->2, 2->1
        edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
        result = _edges_to_neighbor_matrix(edges, N=3, k=2, device=torch.device("cpu"))

        assert result.shape == (3, 2)
        # Node 0 has neighbor 1
        assert 1 in result[0].tolist()
        # Node 1 has neighbors 0 and 2
        assert set(result[1][result[1] >= 0].tolist()) == {0, 2}
        # Node 2 has neighbor 1
        assert 1 in result[2].tolist()

    def test_self_loop_removal(self):
        """Self-loops are removed when skip_self_loops=True."""
        # [E, 2] format: (src, dst) pairs
        edges = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.long)
        result = _edges_to_neighbor_matrix(
            edges,
            N=2,
            k=2,
            device=torch.device("cpu"),
            skip_self_loops=True,
        )

        # Self-loops (0->0) and (1->1) should be removed
        # Node 0 should only have neighbor 1, node 1 should only have neighbor 0
        assert result[0, 0].item() == 1
        assert result[1, 0].item() == 0

    def test_self_loop_preservation(self):
        """Self-loops are kept when skip_self_loops=False."""
        edges = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
        result = _edges_to_neighbor_matrix(
            edges,
            N=2,
            k=3,
            device=torch.device("cpu"),
            skip_self_loops=False,
        )

        # Node 0 has neighbors: self (0) and 1
        node0_valid = result[0][result[0] >= 0].tolist()
        assert 0 in node0_valid  # Self-loop preserved
        assert 1 in node0_valid

    def test_truncation_at_k(self):
        """If a node has more than k neighbors, only first k kept."""
        # Node 0 has 4 neighbors
        edges = torch.tensor(
            [[0, 0, 0, 0], [1, 2, 3, 4]],
            dtype=torch.long,
        ).t()  # [4, 2]
        result = _edges_to_neighbor_matrix(edges, N=5, k=2, device=torch.device("cpu"))

        assert result.shape == (5, 2)
        # Node 0 should have exactly 2 valid neighbors (truncated from 4)
        node0_valid = result[0][result[0] >= 0].tolist()
        assert len(node0_valid) == 2

    def test_padding_for_missing_neighbors(self):
        """Missing neighbors filled with -1."""
        # Only edge: 0->1
        edges = torch.tensor([[0, 1]], dtype=torch.long)
        result = _edges_to_neighbor_matrix(edges, N=3, k=3, device=torch.device("cpu"))

        assert result.shape == (3, 3)
        # Node 0: one neighbor (1) + two padding (-1)
        assert result[0, 0].item() == 1
        assert result[0, 1].item() == -1
        assert result[0, 2].item() == -1
        # Node 1: no outgoing edges (only 0->1 exists), all -1
        assert (result[1] == -1).all()
        # Node 2: no edges, all -1
        assert (result[2] == -1).all()

    def test_empty_edges(self):
        """Empty edge list returns all-(-1) matrix."""
        edges = torch.zeros(0, 2, dtype=torch.long)
        result = _edges_to_neighbor_matrix(edges, N=4, k=3, device=torch.device("cpu"))

        assert result.shape == (4, 3)
        assert (result == -1).all()


# =============================================================================
# TestSelectAliveSamplesAndNeighbors
# =============================================================================


class TestSelectAliveSamplesAndNeighbors:
    """Tests for alive-walker selection and neighbor gathering."""

    def test_all_alive(self):
        """When all walkers are alive, sample_indices are ordered walker indices."""
        T, N, _k = 1, 4, 2
        alive = torch.ones(T, N, dtype=torch.bool)
        neighbor_matrix = torch.tensor([[[1, 2], [0, 2], [0, 1], [0, 1]]], dtype=torch.long)

        sample_indices, _neighbor_indices = _select_alive_samples_and_neighbors(
            alive,
            neighbor_matrix,
            sample_size=N,
            device=torch.device("cpu"),
        )

        assert sample_indices.shape == (T, N)
        # All alive: indices should be [0, 1, 2, 3]
        torch.testing.assert_close(
            sample_indices[0],
            torch.arange(N, dtype=torch.long),
        )

    def test_some_dead_pushed_to_end(self):
        """Dead walkers are pushed to the end (zero-padded)."""
        _T, N, _k = 1, 4, 2
        alive = torch.tensor([[True, False, True, False]])
        neighbor_matrix = torch.tensor([[[1, 2], [0, 2], [0, 1], [0, 1]]], dtype=torch.long)

        sample_indices, _neighbor_indices = _select_alive_samples_and_neighbors(
            alive,
            neighbor_matrix,
            sample_size=N,
            device=torch.device("cpu"),
        )

        # First 2 positions should be alive walkers (0 and 2), last 2 should be 0 (padding)
        assert sample_indices[0, 0].item() == 0
        assert sample_indices[0, 1].item() == 2
        # Padding positions should be zeroed
        assert sample_indices[0, 2].item() == 0
        assert sample_indices[0, 3].item() == 0

    def test_missing_neighbors_replaced_with_self(self):
        """Missing neighbors (-1) are replaced with the walker's own index."""
        T, N, _k = 1, 3, 2
        alive = torch.ones(T, N, dtype=torch.bool)
        # Node 0 has one real neighbor (1) and one missing (-1)
        neighbor_matrix = torch.tensor([[[1, -1], [0, -1], [0, 1]]], dtype=torch.long)

        _sample_indices, neighbor_indices = _select_alive_samples_and_neighbors(
            alive,
            neighbor_matrix,
            sample_size=N,
            device=torch.device("cpu"),
        )

        # Node 0's missing neighbor should be replaced with self (0)
        assert neighbor_indices[0, 0, 0].item() == 1  # Real neighbor
        assert neighbor_indices[0, 0, 1].item() == 0  # Was -1, now self-index

        # Node 1's missing neighbor should be replaced with self (1)
        assert neighbor_indices[0, 1, 0].item() == 0  # Real neighbor
        assert neighbor_indices[0, 1, 1].item() == 1  # Was -1, now self-index

    def test_sample_size_truncation(self):
        """Only keep sample_size walkers when sample_size < N."""
        T, N, k = 1, 5, 2
        alive = torch.ones(T, N, dtype=torch.bool)
        neighbor_matrix = torch.zeros(T, N, k, dtype=torch.long)

        sample_size = 3
        sample_indices, neighbor_indices = _select_alive_samples_and_neighbors(
            alive,
            neighbor_matrix,
            sample_size=sample_size,
            device=torch.device("cpu"),
        )

        assert sample_indices.shape == (T, sample_size)
        assert neighbor_indices.shape == (T, sample_size, k)

    def test_padding_positions_zeroed(self):
        """Neighbor indices at padding positions (beyond alive count) are zeroed."""
        _T, N, _k = 1, 4, 2
        # Only walker 0 is alive
        alive = torch.tensor([[True, False, False, False]])
        neighbor_matrix = torch.tensor([[[1, 2], [0, 2], [0, 1], [0, 1]]], dtype=torch.long)

        _sample_indices, neighbor_indices = _select_alive_samples_and_neighbors(
            alive,
            neighbor_matrix,
            sample_size=N,
            device=torch.device("cpu"),
        )

        # Positions 1, 2, 3 are beyond alive count (only 1 alive walker)
        # Their neighbor indices should be zeroed
        assert (neighbor_indices[0, 1:, :] == 0).all()
