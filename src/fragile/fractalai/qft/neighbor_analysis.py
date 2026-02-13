"""Neighbor topology computation for Fractal Gas QFT analysis.

This module handles neighbor computation methods:
- Auto-detection with smart fallbacks (recorded -> companions -> error)
- Pre-recorded neighbor edges (fastest, O(E))
- Companion walker neighbors (fast, O(N))

Completely decoupled from aggregation pipeline. All functions accept
RunHistory as input and return neighbor topology data structures.

Main Functions:
    - compute_neighbor_topology: Main dispatcher for neighbor computation
    - compute_full_neighbor_matrix: For Euclidean time (all walkers)

Usage:
    from fragile.fractalai.qft.neighbor_analysis import (
        compute_neighbor_topology,
        compute_full_neighbor_matrix,
    )

    # Compute neighbors (auto-detect best method)
    sample_idx, neighbor_idx, alive = compute_neighbor_topology(
        history,
        start_idx=1,
        neighbor_method="auto",
        neighbor_k=2,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F


if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory
    from fragile.fractalai.qft.correlator_channels import ChannelConfig


# =============================================================================
# Primary Neighbor Computation Methods
# =============================================================================


def compute_companion_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use stored companion indices as neighbors.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    n_recorded = end_idx if end_idx is not None else history.n_recorded
    T = n_recorded - start_idx
    N = history.N
    k = max(2, int(neighbor_k))
    sample_size = sample_size or N
    device = history.x_final.device

    alive = history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]
    companions_distance = history.companions_distance[start_idx - 1 : n_recorded - 1]
    companions_clone = history.companions_clone[start_idx - 1 : n_recorded - 1]

    all_sample_idx = []
    all_neighbor_idx = []

    for t in range(T):
        alive_t = alive[t]
        alive_indices = torch.where(alive_t)[0]

        if alive_indices.numel() == 0:
            all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
            all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
            continue

        if alive_indices.numel() <= sample_size:
            sample_idx = alive_indices
        else:
            sample_idx = alive_indices[:sample_size]

        actual_sample_size = sample_idx.numel()
        comp_d = companions_distance[t, sample_idx]
        comp_c = companions_clone[t, sample_idx]
        neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)
        neighbor_idx[:, 0] = comp_d
        neighbor_idx[:, 1] = comp_c

        if actual_sample_size < sample_size:
            sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
            neighbor_idx = F.pad(
                neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0
            )

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


def _edges_to_neighbor_matrix(
    edges: Tensor,
    N: int,
    k: int,
    device: torch.device,
) -> Tensor:
    """Convert edge list to dense neighbor matrix using vectorized scatter.

    Args:
        edges: Edge tensor [E, 2] (src, dst pairs).
        N: Number of nodes.
        k: Number of neighbors to keep per node.
        device: Target device.

    Returns:
        Neighbor matrix [N, k] with -1 for missing neighbors.
    """
    edges_d = edges.to(device)
    src, dst = edges_d[:, 0], edges_d[:, 1]

    # Remove self-loops
    not_self = src != dst
    src = src[not_self]
    dst = dst[not_self]

    if src.numel() == 0:
        return torch.full((N, k), -1, dtype=torch.long, device=device)

    # Sort by src to group neighbors together
    sort_order = torch.argsort(src, stable=True)
    src_sorted = src[sort_order]
    dst_sorted = dst[sort_order]

    # Compute per-source offsets via cumulative count within each group
    degree = torch.bincount(src_sorted, minlength=N)
    max_deg = degree.max().item()

    # Build column indices: for each edge, its position within its source's group
    # Use cumsum trick: count occurrences so far for each source
    group_starts = torch.zeros(N, dtype=torch.long, device=device)
    group_starts[1:] = degree[:-1].cumsum(0)
    col_idx = torch.arange(src_sorted.numel(), device=device) - group_starts[src_sorted]

    # Scatter into dense matrix [N, max_deg]
    neighbor_full = torch.full((N, max(max_deg, k)), -1, dtype=torch.long, device=device)
    # Only write where col_idx < max_deg (always true by construction)
    neighbor_full[src_sorted, col_idx] = dst_sorted

    # Truncate/pad to exactly k columns
    return neighbor_full[:, :k]


def compute_recorded_neighbors_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use recorded neighbor edges from RunHistory.

    Vectorized: converts edge lists to dense neighbor matrices via scatter,
    then gathers for sample indices using advanced indexing.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    if history.neighbor_edges is None:
        return compute_companion_batch(
            history,
            start_idx,
            neighbor_k,
            sample_size=sample_size,
            end_idx=end_idx,
        )

    n_recorded = end_idx if end_idx is not None else history.n_recorded
    T = n_recorded - start_idx
    N = history.N
    k = max(1, int(neighbor_k))
    sample_size = sample_size or N
    device = history.x_final.device

    alive = history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]

    all_sample_idx = []
    all_neighbor_idx = []

    for t in range(T):
        alive_t = alive[t]
        alive_indices = torch.where(alive_t)[0]
        if alive_indices.numel() == 0:
            all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
            all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
            continue

        if alive_indices.numel() <= sample_size:
            sample_idx = alive_indices
        else:
            sample_idx = alive_indices[:sample_size]

        actual_sample_size = sample_idx.numel()

        record_idx = start_idx + t
        edges = history.neighbor_edges[record_idx]
        if not torch.is_tensor(edges) or edges.numel() == 0:
            # Fallback to self-padding
            neighbor_idx = sample_idx.unsqueeze(1).expand(-1, k).clone()
        else:
            # Vectorized: scatter edges into dense matrix, then gather for samples
            neighbor_matrix = _edges_to_neighbor_matrix(edges, N, k, device)
            neighbor_idx = neighbor_matrix[sample_idx]  # [S, k] advanced indexing

            # Replace -1 (missing neighbors) with self-index
            missing = neighbor_idx < 0
            if missing.any():
                neighbor_idx[missing] = sample_idx.unsqueeze(1).expand_as(neighbor_idx)[missing]

        if actual_sample_size < sample_size:
            sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
            neighbor_idx = F.pad(
                neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0
            )

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


def compute_neighbors_auto(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Auto-detect and use best available neighbor data.

    Priority:
    1. Recorded neighbors (neighbor_edges) - O(E) lookup
    2. Companions - O(N) lookup

    Args:
        history: RunHistory with neighbor data.
        start_idx: Starting timestep index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    n_recorded = end_idx if end_idx is not None else history.n_recorded
    required_steps = n_recorded - start_idx

    # Check if recorded neighbors are available
    if history.neighbor_edges is not None and len(history.neighbor_edges) > 0:
        available_steps = len(history.neighbor_edges)

        if available_steps >= required_steps:
            return compute_recorded_neighbors_batch(
                history=history,
                start_idx=start_idx,
                neighbor_k=neighbor_k,
                sample_size=sample_size,
                end_idx=end_idx,
            )
        warnings.warn(
            f"Recorded neighbors only available for {available_steps} steps, "
            f"but {required_steps} steps requested. Falling back to companions.",
            UserWarning,
            stacklevel=2,
        )

    # Fallback to companions
    if history.companions_clone is not None:
        return compute_companion_batch(
            history=history,
            start_idx=start_idx,
            neighbor_k=neighbor_k,
            sample_size=sample_size,
            end_idx=end_idx,
        )

    msg = (
        "No neighbor data available. RunHistory has neither neighbor_edges nor "
        "companions_clone. Set neighbor_graph_record=True during simulation."
    )
    raise RuntimeError(msg)


# =============================================================================
# Dispatcher and Full Matrix Computation
# =============================================================================


def compute_neighbor_topology(
    history: RunHistory,
    start_idx: int,
    neighbor_method: str,
    neighbor_k: int,
    sample_size: int | None = None,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute neighbor indices for all timesteps.

    Dispatches to auto/companions/recorded based on method.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_method: Neighbor selection method:
            - "auto": Auto-detect (recorded -> companions)
            - "recorded": Use history.neighbor_edges (fallback to companions)
            - "companions": Use history.companions_clone
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    # Handle deprecated "uniform" alias
    if neighbor_method == "uniform":
        neighbor_method = "companions"

    if neighbor_method == "auto":
        return compute_neighbors_auto(
            history=history,
            start_idx=start_idx,
            neighbor_k=neighbor_k,
            sample_size=sample_size,
            end_idx=end_idx,
        )
    if neighbor_method == "companions":
        return compute_companion_batch(
            history, start_idx, neighbor_k, sample_size, end_idx=end_idx
        )
    if neighbor_method == "recorded":
        return compute_recorded_neighbors_batch(
            history, start_idx, neighbor_k, sample_size, end_idx=end_idx
        )
    raise ValueError(
        f"Unknown neighbor method: {neighbor_method}. "
        f"Valid options: 'auto', 'recorded', 'companions'"
    )


def compute_full_neighbor_matrix(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    alive: Tensor,
    config: ChannelConfig,
) -> Tensor:
    """Compute neighbor indices for ALL walkers (not just samples).

    Used for Euclidean time mode where we need per-walker operators.
    Delegates to compute_recorded_neighbors_batch or compute_companion_batch
    with sample_size=N, then reshapes to [T, N, k].

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per walker.
        alive: Alive mask [T, N].
        config: ChannelConfig with neighbor method.

    Returns:
        Neighbor indices [T, N, k] for all walkers.
    """
    T, N = alive.shape
    k = max(1, int(neighbor_k))

    # Handle deprecated "uniform" alias
    neighbor_method = config.neighbor_method
    if neighbor_method == "uniform":
        neighbor_method = "companions"

    # Delegate to batch functions with sample_size=N to get all walkers
    try:
        if neighbor_method in {"auto", "recorded"}:
            _sample_idx, neighbor_idx, _ = compute_neighbors_auto(
                history=history,
                start_idx=start_idx,
                neighbor_k=k,
                sample_size=N,
            )
        elif neighbor_method == "companions":
            _sample_idx, neighbor_idx, _ = compute_companion_batch(
                history=history,
                start_idx=start_idx,
                neighbor_k=k,
                sample_size=N,
            )
        else:
            raise ValueError(f"Unknown neighbor method: {neighbor_method}")

        # neighbor_idx is [T, S, k] where S=N — already the right shape
        return neighbor_idx

    except RuntimeError:
        # No neighbor data at all — return self-loops
        device = alive.device
        arange = torch.arange(N, device=device).unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        return arange.expand(T, N, k).clone()


# =============================================================================
# Geometric Weight Extraction
# =============================================================================


def extract_geometric_weights(
    history: RunHistory,
    start_idx: int,
    sample_indices: Tensor,
    neighbor_indices: Tensor,
    alive: Tensor,
) -> tuple[Tensor | None, Tensor | None]:
    """Extract pre-computed geometric weights for sample-neighbor pairs.

    Uses geodesic edge distances and Riemannian volume weights stored in
    RunHistory (from scutoid computation) to build per-sample weight tensors.

    Args:
        history: RunHistory with optional geometric data.
        start_idx: Starting time index (matches aggregation).
        sample_indices: Sampled walker indices [T, S].
        neighbor_indices: Neighbor indices [T, S, k].
        alive: Alive mask [T, N].

    Returns:
        Tuple of (edge_weights [T, S] or None, volume_weights [T, S] or None).
        edge_weights: geodesic distance for (sample, first_neighbor) pair.
        volume_weights: sqrt(det g) for sample walkers.
    """
    T, S = sample_indices.shape
    device = sample_indices.device

    # --- Geodesic edge weights (vectorized per-timestep) ---
    edge_weights = None
    if (
        history.neighbor_edges is not None
        and history.geodesic_edge_distances is not None
        and len(history.geodesic_edge_distances) > 0
    ):
        edge_weights = torch.ones(T, S, device=device, dtype=torch.float32)
        first_neighbor = neighbor_indices[:, :, 0]  # [T, S]
        N = alive.shape[1]

        for t in range(T):
            record_idx = start_idx + t
            if record_idx >= len(history.neighbor_edges):
                continue
            edges = history.neighbor_edges[record_idx]
            dists = history.geodesic_edge_distances[record_idx]
            if (
                not torch.is_tensor(edges)
                or edges.numel() == 0
                or not torch.is_tensor(dists)
                or dists.numel() == 0
            ):
                continue

            # Scatter distances into dense [N, N] matrix (replaces dict construction)
            dists_f = dists.float().to(device)
            edges_d = edges.to(device)
            dist_matrix = torch.zeros(N, N, device=device, dtype=torch.float32)
            dist_matrix[edges_d[:, 0], edges_d[:, 1]] = dists_f
            dist_matrix[edges_d[:, 1], edges_d[:, 0]] = dists_f  # symmetric

            # Gather distances for all S sample-neighbor pairs at once
            gathered = dist_matrix[sample_indices[t], first_neighbor[t]]  # [S]
            has_dist = gathered > 0
            edge_weights[t, has_dist] = gathered[has_dist]

    # --- Riemannian volume weights (fully vectorized) ---
    volume_weights = None
    vol = history.riemannian_volume_weights
    if vol is not None and vol.numel() > 0:
        volume_weights = torch.ones(T, S, device=device, dtype=torch.float32)

        # Compute info indices for all timesteps at once (offset by 1)
        info_indices = torch.arange(T, device=device) + start_idx - 1  # [T]
        valid_t_mask = (info_indices >= 0) & (info_indices < vol.shape[0])

        if valid_t_mask.any():
            valid_t = valid_t_mask.nonzero(as_tuple=True)[0]  # indices of valid timesteps
            vol_on_device = vol.to(device)
            vol_slice = vol_on_device[info_indices[valid_t]]  # [T_valid, N]

            # Clamp sample_indices to valid range for gather
            si_valid = sample_indices[valid_t].long()  # [T_valid, S]
            si_clamped = si_valid.clamp(max=vol_slice.shape[1] - 1)
            gathered = torch.gather(vol_slice, 1, si_clamped)  # [T_valid, S]
            gathered = gathered.float().clamp(min=0.0)

            # Only overwrite where volume > 0
            positive = gathered > 0
            volume_weights[valid_t] = torch.where(positive, gathered, volume_weights[valid_t])

    return edge_weights, volume_weights
