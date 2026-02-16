"""Geodesic distance computation for the kernels pipeline.

Centralises all pairwise geodesic distance matrix computation, scale
selection, and smeared-kernel construction.  Reuses edge-weight utilities
from :mod:`fragile.physics.geometry.weights` and graph conversion helpers
from :mod:`fragile.physics.geometry.neighbors`.

All public functions accept and return plain :class:`torch.Tensor` objects
(no RunHistory access except in the ``*_from_history`` helpers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.physics.fractal_gas.history import RunHistory


# ---------------------------------------------------------------------------
# Method / mode resolution helpers
# ---------------------------------------------------------------------------

_FLOYD_ALIASES = {"floyd", "floyd-warshall", "floyd_warshall"}
_TROPICAL_ALIASES = {"tropical", "min-plus", "min_plus"}
_AUTO_ALIASES = {"auto"}

EDGE_WEIGHT_MODES = (
    "uniform",
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_distance",
    "inverse_riemannian_volume",
    "kernel",
    "riemannian_kernel",
    "riemannian_kernel_volume",
)

_EDGE_WEIGHT_MODE_ALIASES = {
    "unit": "uniform",
    "riemanian_kernel": "riemannian_kernel",
    "riemanian_kernel_volume": "riemannian_kernel_volume",
    "inv_riemannian_distance": "inverse_riemannian_distance",
    "inv_riemannian_volume": "inverse_riemannian_volume",
}


def _resolve_default_device(
    history: RunHistory, device: torch.device | str | None
) -> torch.device:
    if device is not None:
        return torch.device(device)
    x_final = getattr(history, "x_final", None)
    if torch.is_tensor(x_final):
        return x_final.device
    return torch.device("cpu")


def _normalize_method(method: str, device: torch.device) -> str:
    normalized = str(method).strip().lower()
    if normalized in _FLOYD_ALIASES:
        return "floyd-warshall"
    if normalized in _TROPICAL_ALIASES:
        return "tropical"
    if normalized in _AUTO_ALIASES:
        return "floyd-warshall" if device.type == "cpu" else "tropical"
    msg = f"Unknown APSP method {method!r}. Expected one of: auto, floyd-warshall, tropical."
    raise ValueError(msg)


def _normalize_edge_weight_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    prefix = "edge_weight:"
    normalized = normalized.removeprefix(prefix)
    normalized = _EDGE_WEIGHT_MODE_ALIASES.get(normalized, normalized)
    if normalized in EDGE_WEIGHT_MODES:
        return normalized
    msg = f"Unsupported edge_weight_mode {mode!r}. Expected one of {EDGE_WEIGHT_MODES}."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Segment-min helper (for adjacency construction)
# ---------------------------------------------------------------------------


def _segment_min_by_flat_index(
    flat_index: Tensor, values: Tensor, *, size: int
) -> tuple[Tensor, Tensor]:
    """Compute per-index minimum for 1D flat indices."""
    if flat_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=flat_index.device)
        empty_val = torch.empty(0, dtype=values.dtype, device=values.device)
        return empty, empty_val

    if hasattr(torch.Tensor, "scatter_reduce_"):
        reduced = torch.full((size,), float("inf"), dtype=values.dtype, device=values.device)
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


# ---------------------------------------------------------------------------
# Dense adjacency construction
# ---------------------------------------------------------------------------


def build_adjacency_from_edges(
    num_nodes: int,
    edges: Tensor,
    edge_weights: Tensor | None = None,
    undirected: bool = True,
    device: torch.device | None = None,
) -> Tensor:
    """Build dense adjacency matrix ``[N, N]`` with ``inf`` for missing edges, 0 diagonal."""
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")
    n = int(num_nodes)
    dev = torch.device(device) if device is not None else edges.device
    adjacency = torch.full((n, n), float("inf"), dtype=torch.float32, device=dev)
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
        weights = torch.ones(src.shape[0], dtype=torch.float32, device=dev)
    else:
        if not torch.is_tensor(edge_weights):
            edge_weights_t = torch.as_tensor(edge_weights, dtype=torch.float32, device=dev)
        else:
            edge_weights_t = edge_weights.to(device=dev, dtype=torch.float32)
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


# ---------------------------------------------------------------------------
# All-pairs shortest paths
# ---------------------------------------------------------------------------


def batched_floyd_warshall(distance_batch: Tensor) -> Tensor:
    """All-pairs shortest paths via Floyd-Warshall on ``[T, N, N]``."""
    if distance_batch.ndim != 3:
        raise ValueError(
            f"distance_batch must have shape [T, N, N], got {tuple(distance_batch.shape)}."
        )
    if distance_batch.shape[-1] != distance_batch.shape[-2]:
        msg = "distance_batch must be square on the last two dimensions."
        raise ValueError(msg)

    result = distance_batch.clone()
    _, n_nodes, _ = result.shape
    for k in range(n_nodes):
        via_k = result[:, :, k : k + 1] + result[:, k : k + 1, :]
        torch.minimum(result, via_k, out=result)
    return result


def _batched_min_plus_matmul(left: Tensor, right: Tensor, *, block_size: int = 64) -> Tensor:
    """Batched min-plus matrix multiplication in blocks.

    ``(left ⊗ right)_{ij} = min_k(left_{ik} + right_{kj})``
    """
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError(
            f"left/right must have shape [T, N, N], got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.shape != right.shape:
        raise ValueError(
            f"left and right must have the same shape, got {left.shape} vs {right.shape}."
        )
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


def batched_tropical_shortest_paths(distance_batch: Tensor, *, block_size: int = 64) -> Tensor:
    """APSP via tropical (min-plus) matrix squaring on ``[T, N, N]``."""
    if distance_batch.ndim != 3:
        raise ValueError(
            f"distance_batch must have shape [T, N, N], got {tuple(distance_batch.shape)}."
        )
    if distance_batch.shape[-1] != distance_batch.shape[-2]:
        msg = "distance_batch must be square on the last two dimensions."
        raise ValueError(msg)

    result = distance_batch.clone()
    n_nodes = int(result.shape[-1])
    step = 1
    while step < n_nodes:
        result = _batched_min_plus_matmul(result, result, block_size=block_size)
        step *= 2
    return result


# ---------------------------------------------------------------------------
# History-based adjacency batch construction
# ---------------------------------------------------------------------------


def _resolve_frame_edge_weights(
    history: RunHistory,
    *,
    frame_idx: int,
    n_edges: int,
    edge_weight_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Resolve per-edge weights for one frame using recorded mode semantics."""
    mode = _normalize_edge_weight_mode(edge_weight_mode)
    if mode == "uniform":
        return None

    edge_weights_all = getattr(history, "edge_weights", None)
    if edge_weights_all is None:
        msg = f"edge_weight_mode={mode!r} requires RunHistory.edge_weights to be recorded."
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


def build_adjacency_batch_from_history(
    history: RunHistory,
    frame_indices: list[int],
    edge_weight_mode: str = "riemannian_kernel_volume",
    undirected: bool = True,
) -> Tensor:
    """Build ``[T, N, N]`` adjacency from RunHistory neighbor_edges + edge_weights."""
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None:
        msg = "RunHistory.neighbor_edges is None. Enable neighbor recording during simulation."
        raise RuntimeError(msg)

    n_nodes = int(history.N)
    dev = _resolve_default_device(history, None)
    resolved_mode = _normalize_edge_weight_mode(edge_weight_mode)

    adjacency_batch = torch.empty(
        (len(frame_indices), n_nodes, n_nodes),
        dtype=torch.float32,
        device=dev,
    )
    for out_idx, frame_idx in enumerate(frame_indices):
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
            edge_weight_mode=resolved_mode,
            device=dev,
            dtype=torch.float32,
        )
        adjacency_batch[out_idx] = build_adjacency_from_edges(
            num_nodes=n_nodes,
            edges=edges_t,
            edge_weights=weights,
            undirected=undirected,
            device=dev,
        )
    return adjacency_batch


# ---------------------------------------------------------------------------
# Pairwise distance computation
# ---------------------------------------------------------------------------


def compute_pairwise_distances(
    history: RunHistory,
    frame_indices: list[int],
    method: str = "auto",
    edge_weight_mode: str = "riemannian_kernel_volume",
    batch_size: int = 4,
) -> Tensor:
    """Compute geodesic distance matrices ``[T, N, N]`` for given frames.

    Processes frames in batches for memory efficiency.
    """
    if not frame_indices:
        n_nodes = int(history.N)
        dev = _resolve_default_device(history, None)
        return torch.empty((0, n_nodes, n_nodes), dtype=torch.float32, device=dev)

    dev = _resolve_default_device(history, None)
    resolved_method = _normalize_method(method, dev)
    bs = max(1, int(batch_size))

    distance_chunks: list[Tensor] = []
    for offset in range(0, len(frame_indices), bs):
        current_ids = frame_indices[offset : offset + bs]
        adjacency = build_adjacency_batch_from_history(
            history,
            frame_indices=current_ids,
            edge_weight_mode=edge_weight_mode,
            undirected=True,
        )
        if resolved_method == "floyd-warshall":
            distances = batched_floyd_warshall(adjacency)
        else:
            distances = batched_tropical_shortest_paths(adjacency, block_size=64)
        distance_chunks.append(distances)

    return torch.cat(distance_chunks, dim=0)


# ---------------------------------------------------------------------------
# Scale selection
# ---------------------------------------------------------------------------


def _sample_finite_positive(values: Tensor, *, max_samples: int) -> Tensor:
    """Sample finite positive values from an arbitrary tensor."""
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


def select_scales(
    distances: Tensor,
    n_scales: int,
    q_low: float = 0.05,
    q_high: float = 0.95,
    max_samples: int = 500_000,
    min_scale: float = 1e-6,
) -> Tensor:
    """Auto-select scales ``[S]`` from quantiles of positive finite distances."""
    if int(n_scales) <= 0:
        raise ValueError(f"n_scales must be positive, got {n_scales}.")
    if not (0.0 <= float(q_low) < float(q_high) <= 1.0):
        raise ValueError(f"Expected 0 <= q_low < q_high <= 1, got {q_low}, {q_high}.")
    if float(min_scale) <= 0.0:
        raise ValueError(f"min_scale must be positive, got {min_scale}.")

    sampled = _sample_finite_positive(distances, max_samples=int(max_samples))
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

    # Enforce strictly increasing values
    delta = max(float(min_scale) * 1e-3, float(torch.finfo(scales_sorted.dtype).eps))
    for idx in range(1, int(scales_sorted.numel())):
        if scales_sorted[idx] <= scales_sorted[idx - 1]:
            scales_sorted[idx] = scales_sorted[idx - 1] + delta
    return scales_sorted


# ---------------------------------------------------------------------------
# Smeared kernels
# ---------------------------------------------------------------------------


def compute_smeared_kernels(
    distances: Tensor,
    scales: Tensor,
    kernel_type: str = "gaussian",
    normalize_rows: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Compute row-normalized smearing kernels ``[T, S, N, N]`` from distances."""
    if distances.ndim != 3:
        raise ValueError(f"distances must have shape [T, N, N], got {tuple(distances.shape)}.")
    if distances.shape[-1] != distances.shape[-2]:
        msg = "distances must be square on the last two dimensions."
        raise ValueError(msg)

    scales_t = torch.as_tensor(scales, dtype=distances.dtype, device=distances.device)
    if scales_t.ndim != 1 or scales_t.numel() == 0:
        raise ValueError(
            f"scales must be a non-empty 1D sequence, got shape {tuple(scales_t.shape)}."
        )
    if not bool(torch.all(scales_t > 0)):
        msg = "All scales must be strictly positive."
        raise ValueError(msg)

    dist = distances.unsqueeze(1)  # [T, 1, N, N]
    scale_view = scales_t.view(1, -1, 1, 1)  # [1, S, 1, 1]
    finite_mask = torch.isfinite(dist)
    normalized = dist / torch.clamp_min(scale_view, float(eps))

    kernel_name = str(kernel_type).strip().lower()
    if kernel_name == "gaussian":
        kernels = torch.exp(-0.5 * normalized.square())
    elif kernel_name == "exponential":
        kernels = torch.exp(-normalized)
    elif kernel_name == "tophat":
        kernels = (dist <= scale_view).to(distances.dtype)
    elif kernel_name == "shell":
        sigma = torch.clamp_min(scale_view * (1.0 / 3.0), 1e-6)
        kernels = torch.exp(-0.5 * ((dist - scale_view) / sigma).square())
    else:
        raise ValueError(
            f"Unknown kernel_type {kernel_type!r}. "
            "Expected gaussian, exponential, tophat, or shell."
        )

    kernels = torch.where(finite_mask, kernels, torch.zeros_like(kernels))

    # Zero diagonal
    n_nodes = int(distances.shape[-1])
    diagonal_mask = torch.eye(n_nodes, dtype=torch.bool, device=distances.device).view(
        1, 1, n_nodes, n_nodes
    )
    kernels = kernels.masked_fill(diagonal_mask, 0.0)

    if normalize_rows:
        row_sum = kernels.sum(dim=-1, keepdim=True)
        kernels = kernels / torch.clamp_min(row_sum, float(eps))

    return kernels


# ---------------------------------------------------------------------------
# Companion distance extraction
# ---------------------------------------------------------------------------


def gather_companion_distances(
    distances: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract d(i,j), d(i,k), d(j,k) from the full distance matrix.

    Args:
        distances: ``[T, N, N]`` geodesic distance matrix.
        companions_distance: ``[T, N]`` index j for each walker i.
        companions_clone: ``[T, N]`` index k for each walker i.

    Returns:
        d_ij ``[T, N]``, d_ik ``[T, N]``, d_jk ``[T, N]``
    """
    T, N, _ = distances.shape
    device = distances.device

    # Anchor indices: [T, N]
    anchor = torch.arange(N, device=device).unsqueeze(0).expand(T, N)  # i

    j = companions_distance.to(torch.long).clamp(0, N - 1)
    k = companions_clone.to(torch.long).clamp(0, N - 1)

    # d(i, j): distances[t, i, j]
    d_ij = distances[
        torch.arange(T, device=device).unsqueeze(1).expand(T, N),
        anchor,
        j,
    ]
    # d(i, k): distances[t, i, k]
    d_ik = distances[
        torch.arange(T, device=device).unsqueeze(1).expand(T, N),
        anchor,
        k,
    ]
    # d(j, k): distances[t, j, k]
    d_jk = distances[
        torch.arange(T, device=device).unsqueeze(1).expand(T, N),
        j,
        k,
    ]

    return d_ij, d_ik, d_jk
