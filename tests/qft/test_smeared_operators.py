"""Tests for graph-distance utilities used by smeared operators."""

from __future__ import annotations

import torch

from fragile.fractalai.qft.smeared_operators import (
    batched_floyd_warshall,
    batched_tropical_shortest_paths,
    build_adjacency_batch_from_history,
    compute_pairwise_distance_matrices_from_history,
    compute_smeared_kernels_from_distances,
    compute_smeared_kernels_from_history,
    iter_pairwise_distance_batches_from_history,
    select_interesting_scales_from_distances,
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
        alive_mask: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.N = int(n_walkers)
        self.neighbor_edges = [
            edges.to(device=device, dtype=torch.long) for edges in neighbor_edges
        ]
        self.geodesic_edge_distances = (
            None
            if geodesic_edge_distances is None
            else [
                weights.to(device=device, dtype=torch.float32)
                for weights in geodesic_edge_distances
            ]
        )
        if edge_weights is None:
            self.edge_weights = None
        else:
            self.edge_weights = []
            for frame_dict in edge_weights:
                converted: dict[str, torch.Tensor] = {}
                for key, value in frame_dict.items():
                    converted[str(key)] = torch.as_tensor(
                        value, dtype=torch.float32, device=device
                    ).reshape(-1)
                self.edge_weights.append(converted)
        if alive_mask is None:
            n_alive_rows = max(0, len(self.neighbor_edges) - 1)
            self.alive_mask = torch.ones((n_alive_rows, self.N), dtype=torch.bool, device=device)
        else:
            self.alive_mask = alive_mask.to(device=device, dtype=torch.bool)
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


def test_build_adjacency_batch_from_history_recorded_mode() -> None:
    """Recorded edge-weight modes should populate adjacency entries."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=[{"riemannian_kernel_volume": torch.tensor([0.5, 1.5])}],
    )
    adjacency = build_adjacency_batch_from_history(
        history,
        frame_indices=[0],
        edge_weight_mode="riemannian_kernel_volume",
        undirected=True,
    )
    assert adjacency.shape == (1, 3, 3)
    assert torch.allclose(adjacency[0, 0, 1], torch.tensor(0.5))
    assert torch.allclose(adjacency[0, 1, 0], torch.tensor(0.5))
    assert torch.allclose(adjacency[0, 1, 2], torch.tensor(1.5))
    assert torch.isinf(adjacency[0, 0, 2])


def test_build_adjacency_batch_from_history_edge_weight_mode_legacy_alias() -> None:
    """Legacy `weight_mode` alias should remain compatible for named modes."""
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
        edge_weight_mode="uniform",
    ):
        iter_frame_ids.extend(frame_ids)
        iter_batches.append(distance_batch)
    iter_distances = torch.cat(iter_batches, dim=0)

    full_frame_ids, full_distances = compute_pairwise_distance_matrices_from_history(
        history,
        method="floyd-warshall",
        batch_size=2,
        edge_weight_mode="uniform",
    )
    assert iter_frame_ids == full_frame_ids
    torch.testing.assert_close(iter_distances, full_distances)


def test_alive_filtering_and_assume_all_alive_toggle() -> None:
    """Alive filtering should drop dead-walker edges unless explicitly bypassed."""
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[
            torch.zeros((0, 2), dtype=torch.long),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ],
        edge_weights=[
            {"kernel": torch.zeros((0,), dtype=torch.float32)},
            {"kernel": torch.tensor([1.0, 2.0], dtype=torch.float32)},
        ],
        alive_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
    )
    filtered = build_adjacency_batch_from_history(
        history,
        frame_indices=[1],
        edge_weight_mode="kernel",
        assume_all_alive=False,
    )
    assert torch.allclose(filtered[0, 0, 1], torch.tensor(1.0))
    assert torch.isinf(filtered[0, 1, 2])

    bypassed = build_adjacency_batch_from_history(
        history,
        frame_indices=[1],
        edge_weight_mode="kernel",
        assume_all_alive=True,
    )
    assert torch.allclose(bypassed[0, 1, 2], torch.tensor(2.0))


def test_select_interesting_scales_and_kernel_normalization() -> None:
    """Quantile scales should be monotone and produce normalized kernels."""
    distances = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    scales = select_interesting_scales_from_distances(distances, n_scales=4)
    assert scales.shape == (4,)
    assert bool(torch.all(scales[1:] > scales[:-1]))

    kernels = compute_smeared_kernels_from_distances(
        distances,
        scales,
        kernel_type="gaussian",
        zero_diagonal=True,
        normalize_rows=True,
    )
    assert kernels.shape == (1, 4, 3, 3)
    row_sums = kernels.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)
    diagonal = kernels[..., torch.arange(3), torch.arange(3)]
    torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))


def test_compute_smeared_kernels_from_history_returns_expected_shapes() -> None:
    """History convenience API should return stacked kernels and selected scales."""
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[
            torch.zeros((0, 2), dtype=torch.long),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.tensor([[0, 2]], dtype=torch.long),
        ],
        edge_weights=[
            {"riemannian_kernel_volume": torch.zeros((0,), dtype=torch.float32)},
            {"riemannian_kernel_volume": torch.tensor([1.0, 1.5], dtype=torch.float32)},
            {"riemannian_kernel_volume": torch.tensor([2.0], dtype=torch.float32)},
        ],
    )
    frame_ids, scales, kernels = compute_smeared_kernels_from_history(
        history,
        frame_indices=[1, 2],
        n_scales=3,
        method="floyd-warshall",
        batch_size=1,
        assume_all_alive=True,
    )
    assert frame_ids == [1, 2]
    assert scales.shape == (3,)
    assert kernels.shape == (2, 3, 3, 3)


def test_edge_weight_mode_typo_alias_is_normalized() -> None:
    """Common dashboard typos should resolve to canonical stored mode names."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=[{"riemannian_kernel_volume": torch.tensor([3.0, 5.0], dtype=torch.float32)}],
    )
    adjacency = build_adjacency_batch_from_history(
        history,
        frame_indices=[0],
        edge_weight_mode="riemanian_kernel_volume",
        undirected=True,
    )
    assert torch.allclose(adjacency[0, 0, 1], torch.tensor(3.0))
    assert torch.allclose(adjacency[0, 1, 2], torch.tensor(5.0))


def test_strict_missing_mode_raises_with_context() -> None:
    """Missing recorded mode should fail fast with an informative error."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=[{"kernel": torch.tensor([2.0, 4.0], dtype=torch.float32)}],
    )
    try:
        build_adjacency_batch_from_history(
            history,
            frame_indices=[0],
            edge_weight_mode="riemannian_kernel",
        )
    except ValueError as exc:
        text = str(exc)
        assert "does not contain mode" in text
        assert "Available modes: kernel" in text
    else:
        msg = "Expected strict missing-mode failure."
        raise AssertionError(msg)


def test_strict_size_mismatch_raises() -> None:
    """Recorded edge-weight vectors must align with frame edge count."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=[{"kernel": torch.tensor([1.0], dtype=torch.float32)}],
    )
    try:
        build_adjacency_batch_from_history(
            history,
            frame_indices=[0],
            edge_weight_mode="kernel",
        )
    except ValueError as exc:
        assert "size mismatch with neighbor_edges" in str(exc)
    else:
        msg = "Expected strict size-mismatch failure."
        raise AssertionError(msg)


def test_strict_non_finite_values_raise() -> None:
    """Recorded edge-weight vectors must be finite and non-negative."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=[{"kernel": torch.tensor([1.0, float("nan")], dtype=torch.float32)}],
    )
    try:
        build_adjacency_batch_from_history(
            history,
            frame_indices=[0],
            edge_weight_mode="kernel",
        )
    except ValueError as exc:
        assert "contains invalid values" in str(exc)
    else:
        msg = "Expected strict invalid-values failure."
        raise AssertionError(msg)


def test_uniform_mode_requires_no_recorded_edge_weights() -> None:
    """Uniform mode should work even when history.edge_weights is unavailable."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=None,
    )
    adjacency = build_adjacency_batch_from_history(
        history,
        frame_indices=[0],
        edge_weight_mode="uniform",
    )
    assert torch.allclose(adjacency[0, 0, 1], torch.tensor(1.0))
    assert torch.allclose(adjacency[0, 1, 2], torch.tensor(1.0))


def test_conflicting_legacy_and_new_mode_inputs_raise() -> None:
    """Passing different values in `edge_weight_mode` and `weight_mode` should fail."""
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    history = MockNeighborHistory(
        n_walkers=3,
        neighbor_edges=[edges],
        edge_weights=[{"kernel": torch.tensor([1.0, 2.0], dtype=torch.float32)}],
    )
    try:
        build_adjacency_batch_from_history(
            history,
            frame_indices=[0],
            edge_weight_mode="kernel",
            weight_mode="uniform",
        )
    except ValueError as exc:
        assert "Conflicting mode inputs" in str(exc)
    else:
        msg = "Expected conflicting mode-input failure."
        raise AssertionError(msg)
