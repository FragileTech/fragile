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
RECORDED_EDGE_WEIGHT_MODES = (
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_distance",
    "inverse_riemannian_volume",
    "kernel",
    "riemannian_kernel",
    "riemannian_kernel_volume",
)
EDGE_WEIGHT_MODES = ("uniform", *RECORDED_EDGE_WEIGHT_MODES)
_EDGE_WEIGHT_MODE_ALIASES = {
    "unit": "uniform",
    "riemanian_kernel": "riemannian_kernel",
    "riemanian_kernel_volume": "riemannian_kernel_volume",
    "inv_riemannian_distance": "inverse_riemannian_distance",
    "inv_riemannian_volume": "inverse_riemannian_volume",
}


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


def _normalize_edge_weight_mode(mode: str) -> str:
    """Normalize edge-weight mode names to dashboard-compatible canonical names."""
    normalized = str(mode).strip().lower()
    prefix = "edge_weight:"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    normalized = _EDGE_WEIGHT_MODE_ALIASES.get(normalized, normalized)
    if normalized in EDGE_WEIGHT_MODES:
        return normalized
    msg = (
        f"Unsupported edge_weight_mode {mode!r}. "
        f"Expected one of {EDGE_WEIGHT_MODES}."
    )
    raise ValueError(msg)


def _resolve_edge_weight_mode_inputs(
    *,
    edge_weight_mode: str | None,
    weight_mode: str | None,
) -> str:
    """Resolve canonical edge mode from new and legacy parameter names.

    `weight_mode` is a legacy alias kept for backward compatibility.
    """
    if edge_weight_mode is None and weight_mode is None:
        return "riemannian_kernel_volume"
    if edge_weight_mode is not None and weight_mode is not None:
        resolved_new = _normalize_edge_weight_mode(edge_weight_mode)
        resolved_legacy = _normalize_edge_weight_mode(weight_mode)
        if resolved_new != resolved_legacy:
            msg = (
                "Conflicting mode inputs: "
                f"edge_weight_mode={edge_weight_mode!r} -> {resolved_new!r}, "
                f"weight_mode={weight_mode!r} -> {resolved_legacy!r}."
            )
            raise ValueError(msg)
        return resolved_new
    if edge_weight_mode is not None:
        return _normalize_edge_weight_mode(edge_weight_mode)
    return _normalize_edge_weight_mode(weight_mode)


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
    edge_weight_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Resolve per-edge weights for one frame using strict recorded-mode semantics."""
    mode = _normalize_edge_weight_mode(edge_weight_mode)
    if mode == "uniform":
        return None

    edge_weights_all = getattr(history, "edge_weights", None)
    if edge_weights_all is None:
        msg = (
            f"edge_weight_mode={mode!r} requires RunHistory.edge_weights to be recorded."
        )
        raise ValueError(msg)
    if frame_idx < 0 or frame_idx >= len(edge_weights_all):
        msg = (
            f"Frame {frame_idx} missing in RunHistory.edge_weights "
            f"(available 0..{len(edge_weights_all) - 1})."
        )
        raise ValueError(msg)
    frame_dict = edge_weights_all[frame_idx]
    if not isinstance(frame_dict, dict):
        msg = f"edge_weights[{frame_idx}] is not a dict."
        raise ValueError(msg)
    if mode not in frame_dict:
        available_modes = sorted(str(key) for key in frame_dict)
        available_text = ", ".join(available_modes) if available_modes else "<none>"
        msg = (
            f"edge_weights[{frame_idx}] does not contain mode {mode!r}. "
            f"Available modes: {available_text}."
        )
        raise ValueError(msg)
    frame_weights = frame_dict[mode]
    if frame_weights is None:
        msg = f"edge_weights[{frame_idx}][{mode!r}] is None."
        raise ValueError(msg)
    if not torch.is_tensor(frame_weights):
        frame_weights_t = torch.as_tensor(frame_weights, device=device, dtype=dtype).reshape(-1)
    else:
        frame_weights_t = frame_weights.to(device=device, dtype=dtype).reshape(-1)
    if int(frame_weights_t.numel()) != int(n_edges):
        msg = (
            f"edge_weights[{frame_idx}][{mode!r}] size mismatch with neighbor_edges: "
            f"{int(frame_weights_t.numel())} vs {int(n_edges)}."
        )
        raise ValueError(msg)
    if not torch.isfinite(frame_weights_t).all() or torch.any(frame_weights_t < 0):
        msg = f"edge_weights[{frame_idx}][{mode!r}] contains invalid values."
        raise ValueError(msg)
    return frame_weights_t


def _resolve_alive_mask_for_frame(
    history: RunHistory,
    *,
    frame_idx: int,
    n_nodes: int,
    assume_all_alive: bool,
    device: torch.device,
) -> Tensor:
    """Resolve alive walker mask aligned to a recorded neighbor frame."""
    if assume_all_alive:
        return torch.ones((n_nodes,), dtype=torch.bool, device=device)

    alive = getattr(history, "alive_mask", None)
    if not torch.is_tensor(alive):
        return torch.ones((n_nodes,), dtype=torch.bool, device=device)
    if alive.ndim != 2 or int(alive.shape[-1]) != n_nodes or int(alive.shape[0]) <= 0:
        return torch.ones((n_nodes,), dtype=torch.bool, device=device)

    if frame_idx <= 0:
        return torch.ones((n_nodes,), dtype=torch.bool, device=device)

    # neighbor_edges has n_recorded entries while alive_mask has n_recorded-1 entries.
    alive_idx = min(max(frame_idx - 1, 0), int(alive.shape[0]) - 1)
    return alive[alive_idx].to(device=device, dtype=torch.bool)


def build_adjacency_batch_from_history(
    history: RunHistory,
    *,
    frame_indices: Sequence[int],
    edge_weight_mode: str | None = None,
    weight_mode: str | None = None,
    undirected: bool = True,
    assume_all_alive: bool = False,
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
    resolved_edge_weight_mode = _resolve_edge_weight_mode_inputs(
        edge_weight_mode=edge_weight_mode,
        weight_mode=weight_mode,
    )

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
            edge_weight_mode=resolved_edge_weight_mode,
            device=dev,
            dtype=dtype,
        )
        if edges_t.numel() > 0:
            alive_mask = _resolve_alive_mask_for_frame(
                history,
                frame_idx=frame_idx,
                n_nodes=n_nodes,
                assume_all_alive=assume_all_alive,
                device=dev,
            )
            if not bool(alive_mask.all()):
                src = edges_t[:, 0]
                dst = edges_t[:, 1]
                valid_edges = alive_mask[src] & alive_mask[dst]
                edges_t = edges_t[valid_edges]
                if weights is not None:
                    weights = weights[valid_edges]
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
    edge_weight_mode: str | None = None,
    weight_mode: str | None = None,
    undirected: bool = True,
    assume_all_alive: bool = False,
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
    resolved_edge_weight_mode = _resolve_edge_weight_mode_inputs(
        edge_weight_mode=edge_weight_mode,
        weight_mode=weight_mode,
    )
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
            edge_weight_mode=resolved_edge_weight_mode,
            undirected=undirected,
            assume_all_alive=assume_all_alive,
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
    edge_weight_mode: str | None = None,
    weight_mode: str | None = None,
    undirected: bool = True,
    assume_all_alive: bool = False,
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
        edge_weight_mode=edge_weight_mode,
        weight_mode=weight_mode,
        undirected=undirected,
        assume_all_alive=assume_all_alive,
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


def _sample_finite_positive_distances(
    values: Tensor,
    *,
    max_samples: int,
) -> Tensor:
    """Sample finite positive distances from an arbitrary tensor."""
    if int(max_samples) <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    flat = values.reshape(-1)
    valid = torch.isfinite(flat) & (flat > 0)
    if not bool(valid.any()):
        return torch.empty((0,), dtype=values.dtype, device=values.device)
    selected = flat[valid]
    if selected.numel() <= max_samples:
        return selected
    stride = max(1, int(selected.numel() // max_samples))
    sampled = selected[::stride]
    if sampled.numel() > max_samples:
        sampled = sampled[:max_samples]
    return sampled


def select_interesting_scales_from_distances(
    distances: Tensor,
    *,
    n_scales: int,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_samples: int = 500_000,
    min_scale: float = 1e-6,
) -> Tensor:
    """Pick robust smearing scales from the finite positive distance distribution.

    Scales are selected via quantiles so each scale captures a distinct regime
    of the empirical geodesic-distance distribution.
    """
    if int(n_scales) <= 0:
        raise ValueError(f"n_scales must be positive, got {n_scales}.")
    if not (0.0 <= float(q_low) < float(q_high) <= 1.0):
        raise ValueError(f"Expected 0 <= q_low < q_high <= 1, got {q_low}, {q_high}.")
    if float(min_scale) <= 0.0:
        raise ValueError(f"min_scale must be positive, got {min_scale}.")

    sampled = _sample_finite_positive_distances(distances, max_samples=int(max_samples))
    if sampled.numel() == 0:
        return torch.full(
            (int(n_scales),),
            float(min_scale),
            dtype=distances.dtype,
            device=distances.device,
        )

    probs = torch.linspace(
        float(q_low),
        float(q_high),
        steps=int(n_scales),
        device=sampled.device,
        dtype=sampled.dtype,
    )
    scales = torch.quantile(sampled, probs)
    scales = torch.clamp(scales, min=float(min_scale))
    scales_sorted, _ = torch.sort(scales)

    # Enforce strictly increasing values to avoid duplicate scales after quantile ties.
    delta = max(float(min_scale) * 1e-3, float(torch.finfo(scales_sorted.dtype).eps))
    for idx in range(1, int(scales_sorted.numel())):
        if scales_sorted[idx] <= scales_sorted[idx - 1]:
            scales_sorted[idx] = scales_sorted[idx - 1] + delta
    return scales_sorted


def _select_scale_frame_indices(frame_ids: Sequence[int], *, n_scale_frames: int) -> list[int]:
    """Choose representative frames for scale calibration."""
    ids = [int(v) for v in frame_ids]
    if not ids:
        return []
    if int(n_scale_frames) <= 0:
        raise ValueError(f"n_scale_frames must be positive, got {n_scale_frames}.")
    if len(ids) <= int(n_scale_frames):
        return ids
    picks = torch.linspace(0, len(ids) - 1, steps=int(n_scale_frames), dtype=torch.float64)
    out: list[int] = []
    seen: set[int] = set()
    for raw_idx in picks:
        idx = int(round(float(raw_idx.item())))
        idx = min(max(idx, 0), len(ids) - 1)
        frame_id = ids[idx]
        if frame_id not in seen:
            seen.add(frame_id)
            out.append(frame_id)
    return out


def get_available_edge_weight_modes(
    history: RunHistory,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    frame_indices: Sequence[int] | None = None,
) -> dict[int, list[str]]:
    """Return available recorded edge-weight mode names per requested frame."""
    edge_weights = getattr(history, "edge_weights", None)
    if edge_weights is None:
        return {}
    if frame_indices is None:
        frame_ids = resolve_neighbor_frame_indices(history, start_idx=start_idx, end_idx=end_idx)
    else:
        frame_ids = [int(idx) for idx in frame_indices]
    out: dict[int, list[str]] = {}
    for frame_idx in frame_ids:
        if frame_idx < 0 or frame_idx >= len(edge_weights):
            continue
        frame_dict = edge_weights[frame_idx]
        if isinstance(frame_dict, dict):
            out[frame_idx] = sorted(str(key) for key in frame_dict)
    return out


def select_interesting_scales_from_history(
    history: RunHistory,
    *,
    n_scales: int,
    method: str = "floyd-warshall",
    start_idx: int = 0,
    end_idx: int | None = None,
    frame_indices: Sequence[int] | None = None,
    n_scale_frames: int = 8,
    calibration_batch_size: int = 1,
    edge_weight_mode: str | None = None,
    weight_mode: str | None = None,
    undirected: bool = True,
    assume_all_alive: bool = False,
    tropical_block_size: int = 64,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_samples: int = 500_000,
    min_scale: float = 1e-6,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Estimate `n_scales` robust kernel scales from run neighbor data."""
    dev = _resolve_default_device(history, device)
    if frame_indices is None:
        all_frames = resolve_neighbor_frame_indices(history, start_idx=start_idx, end_idx=end_idx)
    else:
        all_frames = [int(idx) for idx in frame_indices]

    scale_frames = _select_scale_frame_indices(all_frames, n_scale_frames=int(n_scale_frames))
    if not scale_frames:
        return torch.full((int(n_scales),), float(min_scale), dtype=dtype, device=dev)

    sampled_chunks: list[Tensor] = []
    per_batch_budget = max(1, int(max_samples // max(1, len(scale_frames))))
    for _, distance_batch in iter_pairwise_distance_batches_from_history(
        history,
        method=method,
        frame_indices=scale_frames,
        batch_size=int(max(1, calibration_batch_size)),
        edge_weight_mode=edge_weight_mode,
        weight_mode=weight_mode,
        undirected=undirected,
        assume_all_alive=assume_all_alive,
        tropical_block_size=tropical_block_size,
        device=dev,
        dtype=dtype,
    ):
        sample = _sample_finite_positive_distances(distance_batch, max_samples=per_batch_budget)
        if sample.numel() > 0:
            sampled_chunks.append(sample)

    if not sampled_chunks:
        return torch.full((int(n_scales),), float(min_scale), dtype=dtype, device=dev)

    sampled = torch.cat(sampled_chunks, dim=0)
    sampled = sampled.to(device=dev, dtype=dtype)
    if sampled.numel() > int(max_samples):
        stride = max(1, int(sampled.numel() // int(max_samples)))
        sampled = sampled[::stride][: int(max_samples)]
    return select_interesting_scales_from_distances(
        sampled,
        n_scales=int(n_scales),
        q_low=float(q_low),
        q_high=float(q_high),
        max_samples=int(max_samples),
        min_scale=float(min_scale),
    )


def compute_smeared_kernels_from_distances(
    distance_batch: Tensor,
    scales: Tensor | Sequence[float],
    *,
    kernel_type: str = "gaussian",
    shell_sigma_ratio: float = 1.0 / 3.0,
    min_shell_sigma: float = 1e-6,
    zero_diagonal: bool = True,
    normalize_rows: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Build multiscale kernels from pairwise distances.

    Args:
        distance_batch: Tensor `[T, N, N]` with finite distances or `inf` gaps.
        scales: Positive scale values `[S]`.
        kernel_type: One of `gaussian`, `exponential`, `tophat`, `shell`.

    Returns:
        Tensor `[T, S, N, N]` with per-scale smoothing weights.
    """
    if distance_batch.ndim != 3:
        raise ValueError(
            f"distance_batch must have shape [T, N, N], got {tuple(distance_batch.shape)}."
        )
    if distance_batch.shape[-1] != distance_batch.shape[-2]:
        msg = "distance_batch must be square on the last two dimensions."
        raise ValueError(msg)

    scales_t = torch.as_tensor(scales, dtype=distance_batch.dtype, device=distance_batch.device)
    if scales_t.ndim != 1 or scales_t.numel() == 0:
        raise ValueError(f"scales must be a non-empty 1D sequence, got shape {tuple(scales_t.shape)}.")
    if not bool(torch.all(scales_t > 0)):
        msg = "All scales must be strictly positive."
        raise ValueError(msg)

    distances = distance_batch.unsqueeze(1)  # [T, 1, N, N]
    scale_view = scales_t.view(1, -1, 1, 1)  # [1, S, 1, 1]
    finite_mask = torch.isfinite(distances)
    normalized = distances / torch.clamp_min(scale_view, float(eps))

    kernel_name = str(kernel_type).strip().lower()
    if kernel_name == "gaussian":
        kernels = torch.exp(-0.5 * normalized.square())
    elif kernel_name == "exponential":
        kernels = torch.exp(-normalized)
    elif kernel_name == "tophat":
        kernels = (distances <= scale_view).to(distance_batch.dtype)
    elif kernel_name == "shell":
        sigma = torch.clamp_min(scale_view * float(shell_sigma_ratio), float(min_shell_sigma))
        kernels = torch.exp(-0.5 * ((distances - scale_view) / sigma).square())
    else:
        raise ValueError(
            f"Unknown kernel_type {kernel_type!r}. Expected gaussian, exponential, tophat, or shell."
        )

    kernels = torch.where(finite_mask, kernels, torch.zeros_like(kernels))

    if bool(zero_diagonal):
        n_nodes = int(distance_batch.shape[-1])
        diagonal_mask = torch.eye(n_nodes, dtype=torch.bool, device=distance_batch.device).view(
            1, 1, n_nodes, n_nodes
        )
        kernels = kernels.masked_fill(diagonal_mask, 0.0)

    if bool(normalize_rows):
        row_sum = kernels.sum(dim=-1, keepdim=True)
        kernels = kernels / torch.clamp_min(row_sum, float(eps))

    return kernels


def iter_smeared_kernel_batches_from_history(
    history: RunHistory,
    *,
    scales: Tensor | Sequence[float] | None = None,
    n_scales: int = 8,
    method: str = "floyd-warshall",
    start_idx: int = 0,
    end_idx: int | None = None,
    frame_indices: Sequence[int] | None = None,
    batch_size: int = 1,
    edge_weight_mode: str | None = None,
    weight_mode: str | None = None,
    undirected: bool = True,
    assume_all_alive: bool = False,
    tropical_block_size: int = 64,
    n_scale_frames: int = 8,
    calibration_batch_size: int = 1,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_scale_samples: int = 500_000,
    min_scale: float = 1e-6,
    kernel_type: str = "gaussian",
    shell_sigma_ratio: float = 1.0 / 3.0,
    min_shell_sigma: float = 1e-6,
    zero_diagonal: bool = True,
    normalize_rows: bool = True,
    kernel_eps: float = 1e-12,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Iterator[tuple[list[int], Tensor, Tensor, Tensor]]:
    """Yield distance and kernel batches as `(frame_ids, distances, kernels, scales)`."""
    dev = _resolve_default_device(history, device)
    if scales is None:
        scales_t = select_interesting_scales_from_history(
            history,
            n_scales=int(n_scales),
            method=method,
            start_idx=start_idx,
            end_idx=end_idx,
            frame_indices=frame_indices,
            n_scale_frames=int(n_scale_frames),
            calibration_batch_size=int(calibration_batch_size),
            edge_weight_mode=edge_weight_mode,
            weight_mode=weight_mode,
            undirected=undirected,
            assume_all_alive=assume_all_alive,
            tropical_block_size=tropical_block_size,
            q_low=float(q_low),
            q_high=float(q_high),
            max_samples=int(max_scale_samples),
            min_scale=float(min_scale),
            device=dev,
            dtype=dtype,
        )
    else:
        scales_t = torch.as_tensor(scales, dtype=dtype, device=dev)
        if scales_t.ndim != 1 or scales_t.numel() == 0:
            raise ValueError(
                f"scales must be a non-empty 1D sequence, got shape {tuple(scales_t.shape)}."
            )
    for ids, distances in iter_pairwise_distance_batches_from_history(
        history,
        method=method,
        start_idx=start_idx,
        end_idx=end_idx,
        frame_indices=frame_indices,
        batch_size=int(batch_size),
        edge_weight_mode=edge_weight_mode,
        weight_mode=weight_mode,
        undirected=undirected,
        assume_all_alive=assume_all_alive,
        tropical_block_size=tropical_block_size,
        device=dev,
        dtype=dtype,
    ):
        kernels = compute_smeared_kernels_from_distances(
            distances,
            scales_t,
            kernel_type=kernel_type,
            shell_sigma_ratio=float(shell_sigma_ratio),
            min_shell_sigma=float(min_shell_sigma),
            zero_diagonal=bool(zero_diagonal),
            normalize_rows=bool(normalize_rows),
            eps=float(kernel_eps),
        )
        yield ids, distances, kernels, scales_t


def compute_smeared_kernels_from_history(
    history: RunHistory,
    *,
    scales: Tensor | Sequence[float] | None = None,
    n_scales: int = 8,
    method: str = "floyd-warshall",
    start_idx: int = 0,
    end_idx: int | None = None,
    frame_indices: Sequence[int] | None = None,
    batch_size: int = 1,
    edge_weight_mode: str | None = None,
    weight_mode: str | None = None,
    undirected: bool = True,
    assume_all_alive: bool = False,
    tropical_block_size: int = 64,
    n_scale_frames: int = 8,
    calibration_batch_size: int = 1,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_scale_samples: int = 500_000,
    min_scale: float = 1e-6,
    kernel_type: str = "gaussian",
    shell_sigma_ratio: float = 1.0 / 3.0,
    min_shell_sigma: float = 1e-6,
    zero_diagonal: bool = True,
    normalize_rows: bool = True,
    kernel_eps: float = 1e-12,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[int], Tensor, Tensor]:
    """Compute stacked multiscale kernels `[T, S, N, N]` for requested frames."""
    frame_ids_all: list[int] = []
    kernel_chunks: list[Tensor] = []
    scales_out: Tensor | None = None
    for ids, _, kernels, scales_t in iter_smeared_kernel_batches_from_history(
        history,
        scales=scales,
        n_scales=int(n_scales),
        method=method,
        start_idx=start_idx,
        end_idx=end_idx,
        frame_indices=frame_indices,
        batch_size=int(batch_size),
        edge_weight_mode=edge_weight_mode,
        weight_mode=weight_mode,
        undirected=undirected,
        assume_all_alive=assume_all_alive,
        tropical_block_size=int(tropical_block_size),
        n_scale_frames=int(n_scale_frames),
        calibration_batch_size=int(calibration_batch_size),
        q_low=float(q_low),
        q_high=float(q_high),
        max_scale_samples=int(max_scale_samples),
        min_scale=float(min_scale),
        kernel_type=kernel_type,
        shell_sigma_ratio=float(shell_sigma_ratio),
        min_shell_sigma=float(min_shell_sigma),
        zero_diagonal=bool(zero_diagonal),
        normalize_rows=bool(normalize_rows),
        kernel_eps=float(kernel_eps),
        device=device,
        dtype=dtype,
    ):
        frame_ids_all.extend(ids)
        kernel_chunks.append(kernels)
        scales_out = scales_t

    n_nodes = int(history.N)
    resolved_device = _resolve_default_device(history, device)
    if not kernel_chunks:
        if scales_out is None:
            if scales is None:
                scales_out = torch.full((int(n_scales),), float(min_scale), dtype=dtype, device=resolved_device)
            else:
                scales_out = torch.as_tensor(scales, dtype=dtype, device=resolved_device).reshape(-1)
        empty = torch.empty(
            (0, int(scales_out.numel()), n_nodes, n_nodes),
            dtype=dtype,
            device=resolved_device,
        )
        return frame_ids_all, scales_out, empty

    kernels = torch.cat(kernel_chunks, dim=0)
    if scales_out is None:
        scales_out = torch.full((int(n_scales),), float(min_scale), dtype=dtype, device=resolved_device)
    return frame_ids_all, scales_out, kernels
