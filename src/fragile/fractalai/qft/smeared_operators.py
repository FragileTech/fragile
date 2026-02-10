"""Smeared-operator graph distances from recorded neighbor data.

This module converts recorded sparse neighbor graphs into dense adjacency
matrices and computes all-pairs shortest-path distances per recorded frame.

It provides two APSP backends:
- Batched Floyd-Warshall (memory-stable, O(T * N^3) sequential in N).
- Batched tropical min-plus squaring (fewer sequential steps, larger kernels).

For large runs (for example, many frames), prefer the iterator API to process
frames in chunks and avoid allocating all distance matrices at once.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory


_FLOYD_ALIASES = {"floyd", "floyd-warshall", "floyd_warshall"}
_TROPICAL_ALIASES = {"tropical", "min-plus", "min_plus"}
_AUTO_ALIASES = {"auto"}


def _resolve_default_device(history: RunHistory, device: torch.device | str | None) -> torch.device:
    """Resolve output device from explicit argument or history tensors."""
    if device is not None:
        return torch.device(device)
    x_final = getattr(history, "x_final", None)
    if torch.is_tensor(x_final):
        return x_final.device
    return torch.device("cpu")


def _normalize_method(method: str, device: torch.device) -> str:
    """Normalize method aliases and resolve auto selection."""
    normalized = str(method).strip().lower()
    if normalized in _FLOYD_ALIASES:
        return "floyd-warshall"
    if normalized in _TROPICAL_ALIASES:
        return "tropical"
    if normalized in _AUTO_ALIASES:
        return "floyd-warshall" if device.type == "cpu" else "tropical"
    msg = (
        f"Unknown APSP method {method!r}. "
        "Expected one of: auto, floyd-warshall, tropical."
    )
    raise ValueError(msg)


def resolve_neighbor_frame_indices(
    history: RunHistory,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[int]:
    """Resolve neighbor-frame indices within `history.neighbor_edges`."""
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None:
        return []
    n_frames = len(neighbor_edges)
    if n_frames <= 0:
        return []

    start = int(start_idx)
    if start < 0:
        start += n_frames
    start = max(0, start)

    stop = n_frames if end_idx is None else int(end_idx)
    if stop < 0:
        stop += n_frames
    stop = min(n_frames, max(0, stop))

    if stop <= start:
        return []
    return list(range(start, stop))


def _segment_min_by_flat_index(
    flat_index: Tensor,
    values: Tensor,
    *,
    size: int,
) -> tuple[Tensor, Tensor]:
    """Compute per-index minimum for 1D flat indices."""
    if flat_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=flat_index.device)
        empty_val = torch.empty(0, dtype=values.dtype, device=values.device)
        return empty, empty_val

    if hasattr(torch.Tensor, "scatter_reduce_"):
        reduced = torch.full(
            (size,),
            float("inf"),
            dtype=values.dtype,
            device=values.device,
        )
        reduced.scatter_reduce_(0, flat_index, values, reduce="amin", include_self=True)
        valid = torch.isfinite(reduced)
        unique_flat = torch.nonzero(valid, as_tuple=False).flatten()
        return unique_flat, reduced[unique_flat]

    order = torch.argsort(flat_index)
    flat_sorted = flat_index[order]
    val_sorted = values[order]
    uniq, counts = torch.unique_consecutive(flat_sorted, return_counts=True)
    mins = torch.empty_like(uniq, dtype=values.dtype, device=values.device)
    cursor = 0
    for idx, count in enumerate(counts.tolist()):
        next_cursor = cursor + count
        mins[idx] = val_sorted[cursor:next_cursor].min()
        cursor = next_cursor
    return uniq, mins


def build_adjacency_from_edges(
    *,
    num_nodes: int,
    edges: Tensor,
    edge_weights: Tensor | None = None,
    undirected: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build dense adjacency matrix with `inf` for missing edges and zero diagonal."""
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")
    n = int(num_nodes)
    dev = torch.device(device) if device is not None else edges.device
    adjacency = torch.full((n, n), float("inf"), dtype=dtype, device=dev)
    if n == 0:
        return adjacency
    adjacency.diagonal().zero_()

    if not torch.is_tensor(edges):
        edges_t = torch.as_tensor(edges, dtype=torch.long, device=dev)
    else:
        edges_t = edges.to(device=dev, dtype=torch.long)
    if edges_t.numel() == 0:
        return adjacency
    if edges_t.ndim != 2 or edges_t.shape[1] != 2:
        msg = f"edges must have shape [E, 2], got {tuple(edges_t.shape)}."
        raise ValueError(msg)

    src = edges_t[:, 0]
    dst = edges_t[:, 1]
    valid = (src >= 0) & (dst >= 0) & (src < n) & (dst < n) & (src != dst)
    if not bool(valid.any()):
        return adjacency

    src = src[valid]
    dst = dst[valid]
    if edge_weights is None:
        weights = torch.ones(src.shape[0], dtype=dtype, device=dev)
    else:
        if not torch.is_tensor(edge_weights):
            edge_weights_t = torch.as_tensor(edge_weights, dtype=dtype, device=dev)
        else:
            edge_weights_t = edge_weights.to(device=dev, dtype=dtype)
        if edge_weights_t.numel() != edges_t.shape[0]:
            msg = (
                "edge_weights must have one value per edge. "
                f"Got {edge_weights_t.numel()} values for {edges_t.shape[0]} edges."
            )
            raise ValueError(msg)
        weights = edge_weights_t.reshape(-1)[valid]
    finite = torch.isfinite(weights) & (weights >= 0)
    src = src[finite]
    dst = dst[finite]
    weights = weights[finite]
    if src.numel() == 0:
        return adjacency

    flat_size = n * n
    flat_index = src * n + dst
    uniq_flat, uniq_weight = _segment_min_by_flat_index(flat_index, weights, size=flat_size)
    uniq_src = torch.div(uniq_flat, n, rounding_mode="floor")
    uniq_dst = uniq_flat.remainder(n)
    adjacency[uniq_src, uniq_dst] = uniq_weight

    if undirected:
        rev_flat = dst * n + src
        rev_flat_u, rev_weight_u = _segment_min_by_flat_index(rev_flat, weights, size=flat_size)
        rev_src = torch.div(rev_flat_u, n, rounding_mode="floor")
        rev_dst = rev_flat_u.remainder(n)
        adjacency[rev_src, rev_dst] = torch.minimum(adjacency[rev_src, rev_dst], rev_weight_u)

    return adjacency


def _resolve_frame_edge_weights(
    history: RunHistory,
    *,
    frame_idx: int,
    n_edges: int,
    weight_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Resolve per-edge weights for one frame."""
    mode = str(weight_mode).strip().lower()
    if mode == "unit":
        return None

    if mode == "geodesic":
        geodesic = getattr(history, "geodesic_edge_distances", None)
        if geodesic is None or frame_idx >= len(geodesic):
            return None
        frame_weights = geodesic[frame_idx]
        if not torch.is_tensor(frame_weights) or frame_weights.numel() != n_edges:
            return None
        return frame_weights.to(device=device, dtype=dtype)

    prefix = "edge_weight:"
    if mode.startswith(prefix):
        key = mode[len(prefix) :]
        edge_weights_all = getattr(history, "edge_weights", None)
        if edge_weights_all is None or frame_idx >= len(edge_weights_all):
            return None
        frame_dict = edge_weights_all[frame_idx]
        if not isinstance(frame_dict, dict):
            return None
        frame_weights = frame_dict.get(key)
        if not torch.is_tensor(frame_weights) or frame_weights.numel() != n_edges:
            return None
        return frame_weights.to(device=device, dtype=dtype)

    msg = (
        f"Unknown weight_mode {weight_mode!r}. "
        "Expected unit, geodesic, or edge_weight:<mode>."
    )
    raise ValueError(msg)


def build_adjacency_batch_from_history(
    history: RunHistory,
    *,
    frame_indices: Sequence[int],
    weight_mode: str = "geodesic",
    undirected: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build dense adjacency matrices `[T, N, N]` for requested frames."""
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None:
        msg = (
            "RunHistory.neighbor_edges is None. "
            "Enable neighbor recording during simulation."
        )
        raise RuntimeError(msg)

    n_nodes = int(history.N)
    frame_ids = list(frame_indices)
    dev = _resolve_default_device(history, device)

    adjacency_batch = torch.empty(
        (len(frame_ids), n_nodes, n_nodes),
        dtype=dtype,
        device=dev,
    )
    for out_idx, frame_idx in enumerate(frame_ids):
        if frame_idx < 0 or frame_idx >= len(neighbor_edges):
            raise IndexError(
                f"Frame index {frame_idx} out of range for {len(neighbor_edges)} neighbor frames."
            )
        edges = neighbor_edges[frame_idx]
        if not torch.is_tensor(edges):
            edges_t = torch.empty((0, 2), dtype=torch.long, device=dev)
        else:
            edges_t = edges.to(device=dev, dtype=torch.long)
            if edges_t.ndim != 2 or edges_t.shape[1] != 2:
                msg = (
                    f"neighbor_edges[{frame_idx}] must have shape [E, 2], "
                    f"got {tuple(edges_t.shape)}."
                )
                raise ValueError(msg)

        weights = _resolve_frame_edge_weights(
            history,
            frame_idx=frame_idx,
            n_edges=int(edges_t.shape[0]),
            weight_mode=weight_mode,
            device=dev,
            dtype=dtype,
        )
        adjacency_batch[out_idx] = build_adjacency_from_edges(
            num_nodes=n_nodes,
            edges=edges_t,
            edge_weights=weights,
            undirected=undirected,
            device=dev,
            dtype=dtype,
        )
    return adjacency_batch


def batched_floyd_warshall(distance_batch: Tensor, *, clone: bool = True) -> Tensor:
    """Compute batched all-pairs shortest paths with Floyd-Warshall."""
    if distance_batch.ndim != 3:
        raise ValueError(f"distance_batch must have shape [T, N, N], got {tuple(distance_batch.shape)}.")
    if distance_batch.shape[-1] != distance_batch.shape[-2]:
        msg = "distance_batch must be square on the last two dimensions."
        raise ValueError(msg)

    result = distance_batch.clone() if clone else distance_batch
    _, n_nodes, _ = result.shape
    for k in range(n_nodes):
        via_k = result[:, :, k : k + 1] + result[:, k : k + 1, :]
        torch.minimum(result, via_k, out=result)
    return result


def batched_min_plus_matmul(
    left: Tensor,
    right: Tensor,
    *,
    block_size: int = 64,
) -> Tensor:
    """Compute batched min-plus matrix multiplication in blocks.

    This computes `(left ⊗ right)_{ij} = min_k(left_{ik} + right_{kj})`.
    """
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError(
            f"left/right must have shape [T, N, N], got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.shape != right.shape:
        raise ValueError(f"left and right must have the same shape, got {left.shape} vs {right.shape}.")
    if left.shape[-1] != left.shape[-2]:
        msg = "left/right must be square on the last two dimensions."
        raise ValueError(msg)
    if int(block_size) <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    batch_size, n_nodes, _ = left.shape
    block = min(int(block_size), n_nodes if n_nodes > 0 else 1)
    output = torch.full(
        (batch_size, n_nodes, n_nodes),
        float("inf"),
        dtype=left.dtype,
        device=left.device,
    )

    for col_start in range(0, n_nodes, block):
        col_end = min(col_start + block, n_nodes)
        col_width = col_end - col_start
        col_result = torch.full(
            (batch_size, n_nodes, col_width),
            float("inf"),
            dtype=left.dtype,
            device=left.device,
        )
        right_cols = right[:, :, col_start:col_end]
        for k_start in range(0, n_nodes, block):
            k_end = min(k_start + block, n_nodes)
            left_k = left[:, :, k_start:k_end]
            right_k = right_cols[:, k_start:k_end, :]
            candidate = left_k.unsqueeze(-1) + right_k.unsqueeze(1)
            candidate_min = candidate.min(dim=2).values
            torch.minimum(col_result, candidate_min, out=col_result)
        output[:, :, col_start:col_end] = col_result
    return output


def batched_tropical_shortest_paths(
    distance_batch: Tensor,
    *,
    block_size: int = 64,
    clone: bool = True,
) -> Tensor:
    """Compute APSP by repeated tropical squaring."""
    if distance_batch.ndim != 3:
        raise ValueError(f"distance_batch must have shape [T, N, N], got {tuple(distance_batch.shape)}.")
    if distance_batch.shape[-1] != distance_batch.shape[-2]:
        msg = "distance_batch must be square on the last two dimensions."
        raise ValueError(msg)

    result = distance_batch.clone() if clone else distance_batch
    n_nodes = int(result.shape[-1])
    step = 1
    while step < n_nodes:
        result = batched_min_plus_matmul(result, result, block_size=block_size)
        step *= 2
    return result


def iter_pairwise_distance_batches_from_history(
    history: RunHistory,
    *,
    method: str = "floyd-warshall",
    start_idx: int = 0,
    end_idx: int | None = None,
    frame_indices: Sequence[int] | None = None,
    batch_size: int = 1,
    weight_mode: str = "geodesic",
    undirected: bool = True,
    tropical_block_size: int = 64,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Iterator[tuple[list[int], Tensor]]:
    """Yield shortest-path distance matrices in frame batches.

    Yields tuples `(frame_ids, distance_batch)` with shapes:
    - `frame_ids`: list[int] length `B`
    - `distance_batch`: Tensor `[B, N, N]`
    """
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    dev = _resolve_default_device(history, device)
    resolved_method = _normalize_method(method, dev)
    if frame_indices is None:
        frame_ids = resolve_neighbor_frame_indices(history, start_idx=start_idx, end_idx=end_idx)
    else:
        frame_ids = [int(idx) for idx in frame_indices]

    batch = int(batch_size)
    for offset in range(0, len(frame_ids), batch):
        current_ids = frame_ids[offset : offset + batch]
        adjacency = build_adjacency_batch_from_history(
            history,
            frame_indices=current_ids,
            weight_mode=weight_mode,
            undirected=undirected,
            device=dev,
            dtype=dtype,
        )
        if resolved_method == "floyd-warshall":
            distances = batched_floyd_warshall(adjacency, clone=False)
        else:
            distances = batched_tropical_shortest_paths(
                adjacency,
                block_size=int(tropical_block_size),
                clone=False,
            )
        yield current_ids, distances


def compute_pairwise_distance_matrices_from_history(
    history: RunHistory,
    *,
    method: str = "floyd-warshall",
    start_idx: int = 0,
    end_idx: int | None = None,
    frame_indices: Sequence[int] | None = None,
    batch_size: int = 1,
    weight_mode: str = "geodesic",
    undirected: bool = True,
    tropical_block_size: int = 64,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[int], Tensor]:
    """Compute and stack distance matrices for requested frames.

    Returns:
        `(frame_ids, distances)` where:
        - `frame_ids` is the ordered list of processed frame indices
        - `distances` has shape `[T, N, N]`
    """
    all_frame_ids: list[int] = []
    distance_chunks: list[Tensor] = []
    for frame_ids_chunk, distance_chunk in iter_pairwise_distance_batches_from_history(
        history,
        method=method,
        start_idx=start_idx,
        end_idx=end_idx,
        frame_indices=frame_indices,
        batch_size=batch_size,
        weight_mode=weight_mode,
        undirected=undirected,
        tropical_block_size=tropical_block_size,
        device=device,
        dtype=dtype,
    ):
        all_frame_ids.extend(frame_ids_chunk)
        distance_chunks.append(distance_chunk)

    n_nodes = int(history.N)
    resolved_device = _resolve_default_device(history, device)
    if not distance_chunks:
        empty = torch.empty((0, n_nodes, n_nodes), dtype=dtype, device=resolved_device)
        return all_frame_ids, empty

    distances = torch.cat(distance_chunks, dim=0)
    return all_frame_ids, distances
