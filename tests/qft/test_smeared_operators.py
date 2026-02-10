"""Tests for graph-distance utilities used by smeared operators."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.smeared_operators import (
    batched_floyd_warshall,
    batched_tropical_shortest_paths,
    build_adjacency_batch_from_history,
    compute_pairwise_distance_matrices_from_history,
    iter_pairwise_distance_batches_from_history,
)


class MockNeighborHistory:
    """Minimal history stub for neighbor-distance tests."""

    def __init__(
        self,
        *,
        n_walkers: int,
        neighbor_edges: list[torch.Tensor],
        geodesic_edge_distances: list[torch.Tensor] | None = None,
        edge_weights: list[dict[str, torch.Tensor]] | None = None,
        device: str = "cpu",
    ) -> None:
        self.N = int(n_walkers)
        self.neighbor_edges = [edges.to(device=device, dtype=torch.long) for edges in neighbor_edges]
        self.geodesic_edge_distances = (
            None
            if geodesic_edge_distances is None
            else [weights.to(device=device, dtype=torch.float32) for weights in geodesic_edge_distances]
        )
        self.edge_weights = edge_weights
        self.x_final = torch.zeros((1, self.N, 1), dtype=torch.float32, device=device)


def _chain_adjacency_batch() -> torch.Tensor:
    """Construct one weighted chain graph as dense adjacency batch [1, 4, 4]."""
    inf = float("inf")
    return torch.tensor(
        [
            [
                [0.0, 1.0, 10.0, inf],
                [1.0, 0.0, 2.0, inf],
                [10.0, 2.0, 0.0, 1.0],
                [inf, inf, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )


def test_batched_floyd_warshall_on_chain_graph() -> None:
    """Floyd-Warshall should recover exact shortest-path distances."""
    adjacency = _chain_adjacency_batch()
    distances = batched_floyd_warshall(adjacency)
    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 3.0, 4.0],
                [1.0, 0.0, 2.0, 3.0],
                [3.0, 2.0, 0.0, 1.0],
                [4.0, 3.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(distances, expected)


def test_tropical_matches_floyd_warshall() -> None:
    """Tropical squaring should match Floyd-Warshall on the same batch."""
    adjacency = _chain_adjacency_batch()
    floyd = batched_floyd_warshall(adjacency)
    tropical = batched_tropical_shortest_paths(adjacency, block_size=2)
    torch.testing.assert_close(tropical, floyd)


def test_build_adjacency_batch_from_history_geodesic() -> None:
    """History geodesic edge weights should populate adjacency entries."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    geodesic = torch.tensor([0.5, 1.5], dtype=torch.float32)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        geodesic_edge_distances=[geodesic],
    )
    adjacency = build_adjacency_batch_from_history(
        history,
        frame_indices=[0],
        weight_mode="geodesic",
        undirected=True,
    )
    assert adjacency.shape == (1, 3, 3)
    assert torch.allclose(adjacency[0, 0, 1], torch.tensor(0.5))
    assert torch.allclose(adjacency[0, 1, 0], torch.tensor(0.5))
    assert torch.allclose(adjacency[0, 1, 2], torch.tensor(1.5))
    assert torch.isinf(adjacency[0, 0, 2])


def test_build_adjacency_batch_from_history_edge_weight_mode() -> None:
    """Named edge weight mode should be read from `history.edge_weights`."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        geodesic_edge_distances=None,
        edge_weights=[{"kernel": torch.tensor([2.0, 4.0], dtype=torch.float32)}],
    )
    adjacency = build_adjacency_batch_from_history(
        history,
        frame_indices=[0],
        weight_mode="edge_weight:kernel",
        undirected=True,
    )
    assert torch.allclose(adjacency[0, 0, 1], torch.tensor(2.0))
    assert torch.allclose(adjacency[0, 1, 2], torch.tensor(4.0))


def test_iter_and_compute_distance_batches_are_consistent() -> None:
    """Iterator batches and stacked convenience API should agree."""
    edges_by_frame = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 2]], dtype=torch.long),
        torch.tensor([[1, 2]], dtype=torch.long),
    ]
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=edges_by_frame,
        geodesic_edge_distances=None,
    )

    iter_frame_ids: list[int] = []
    iter_batches: list[torch.Tensor] = []
    for frame_ids, distance_batch in iter_pairwise_distance_batches_from_history(
        history,
        method="floyd-warshall",
        batch_size=2,
        weight_mode="unit",
    ):
        iter_frame_ids.extend(frame_ids)
        iter_batches.append(distance_batch)
    iter_distances = torch.cat(iter_batches, dim=0)

    full_frame_ids, full_distances = compute_pairwise_distance_matrices_from_history(
        history,
        method="floyd-warshall",
        batch_size=2,
        weight_mode="unit",
    )
    assert iter_frame_ids == full_frame_ids
    torch.testing.assert_close(iter_distances, full_distances)
