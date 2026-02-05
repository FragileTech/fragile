"""Neighbor topology computation for Fractal Gas QFT analysis.

This module handles all neighbor computation methods:
- Auto-detection with smart fallbacks (recorded → companions → voronoi)
- Companion walker neighbors (fast, O(N))
- Pre-recorded neighbor edges (fastest, O(E))
- Voronoi/Delaunay tessellation (slower, O(N log N))

Completely decoupled from aggregation pipeline. All functions accept
RunHistory as input and return neighbor topology data structures.

Main Functions:
    - compute_neighbor_topology: Main dispatcher for neighbor computation
    - compute_full_neighbor_matrix: For Euclidean time (all walkers)
    - check_neighbor_data_availability: Diagnostics

Usage:
    from fragile.fractalai.qft.neighbor_analysis import (
        compute_neighbor_topology,
        check_neighbor_data_availability,
    )

    # Check what data is available
    info = check_neighbor_data_availability(history)

    # Compute neighbors (auto-detect best method)
    sample_idx, neighbor_idx, alive = compute_neighbor_topology(
        history,
        start_idx=1,
        neighbor_method="auto",
        neighbor_k=2,
        voronoi_config={},
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory
    from fragile.fractalai.qft.correlator_channels import ChannelConfig


# =============================================================================
# Diagnostics
# =============================================================================


def check_neighbor_data_availability(history: RunHistory) -> dict[str, Any]:
    """Check what neighbor data is available in RunHistory.

    Useful for diagnostics and determining optimal neighbor method.

    Args:
        history: RunHistory object.

    Returns:
        Dictionary with availability info:
        {
            "has_companions": bool,
            "has_recorded_edges": bool,
            "has_voronoi_regions": bool,
            "recorded_steps": int,
            "total_steps": int,
            "recommended_method": str,
            "coverage_fraction": float,
        }
    """
    has_companions = history.companions_clone is not None
    has_recorded_edges = (
        history.neighbor_edges is not None and
        len(history.neighbor_edges) > 0
    )
    has_voronoi_regions = (
        history.voronoi_regions is not None and
        len(history.voronoi_regions) > 0
    )

    recorded_steps = len(history.neighbor_edges) if has_recorded_edges else 0
    total_steps = len(history.x_before_clone) if history.x_before_clone is not None else 0

    # Recommend method based on availability
    if has_recorded_edges and recorded_steps >= total_steps * 0.9:
        recommended_method = "recorded"
    elif has_companions:
        recommended_method = "companions"
    else:
        recommended_method = "voronoi"

    return {
        "has_companions": has_companions,
        "has_recorded_edges": has_recorded_edges,
        "has_voronoi_regions": has_voronoi_regions,
        "recorded_steps": recorded_steps,
        "total_steps": total_steps,
        "recommended_method": recommended_method,
        "coverage_fraction": recorded_steps / total_steps if total_steps > 0 else 0.0,
    }


# =============================================================================
# Helper Functions
# =============================================================================




def _collect_time_sliced_edges(time_sliced, mode: str) -> np.ndarray:
    """Collect edges from time-sliced Voronoi tessellation.

    Args:
        time_sliced: Time-sliced Voronoi result.
        mode: Edge selection mode ("spacelike", "timelike", "spacelike+timelike").

    Returns:
        Edge array [E, 2].
    """
    edges: list[np.ndarray] = []
    if mode in {"spacelike", "spacelike+timelike"}:
        for bin_result in time_sliced.bins:
            if bin_result.spacelike_edges is not None and bin_result.spacelike_edges.size:
                edges.append(bin_result.spacelike_edges)
    if mode in {"timelike", "spacelike+timelike"}:
        if (
            time_sliced.timelike_edges is not None
            and time_sliced.timelike_edges.size
        ):
            edges.append(time_sliced.timelike_edges)
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.vstack(edges)


def _build_neighbor_lists(edges: np.ndarray, n: int) -> list[list[int]]:
    """Build neighbor lists from edge array.

    Args:
        edges: Edge array [E, 2].
        n: Number of nodes.

    Returns:
        List of neighbor lists for each node.
    """
    neighbors = [[] for _ in range(n)]
    if edges.size == 0:
        return neighbors
    for i, j in edges:
        if i == j:
            continue
        if 0 <= i < n and 0 <= j < n:
            neighbors[i].append(int(j))
    # De-duplicate while preserving order
    for idx, items in enumerate(neighbors):
        if len(items) <= 1:
            continue
        seen: set[int] = set()
        unique = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        neighbors[idx] = unique
    return neighbors


# =============================================================================
# Primary Neighbor Computation Methods
# =============================================================================


def compute_companion_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use stored companion indices as neighbors.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    n_recorded = history.n_recorded
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
            neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


def compute_voronoi_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    voronoi_pbc_mode: str,
    voronoi_exclude_boundary: bool,
    voronoi_boundary_tolerance: float,
    sample_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Voronoi neighbor indices for all timesteps.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        voronoi_pbc_mode: PBC handling mode for Voronoi.
        voronoi_exclude_boundary: Exclude boundary points.
        voronoi_boundary_tolerance: Tolerance for boundary detection.
        sample_size: Number of samples per timestep (None = all walkers).

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    try:
        from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation
    except Exception:
        return compute_companion_batch(
            history, start_idx, neighbor_k, sample_size=sample_size
        )

    n_recorded = history.n_recorded
    T = n_recorded - start_idx
    N = history.N
    k = int(neighbor_k)
    sample_size = sample_size or N
    device = history.x_final.device

    x_pre = history.x_before_clone[start_idx:]  # [T, N, d]
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
        neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)

        vor_data = compute_voronoi_tessellation(
            positions=x_pre[t],
            alive=alive_t,
            bounds=history.bounds,
            pbc=history.pbc,
            pbc_mode=voronoi_pbc_mode,
            exclude_boundary=voronoi_exclude_boundary,
            boundary_tolerance=voronoi_boundary_tolerance,
        )
        neighbor_lists = vor_data.get("neighbor_lists", {})
        index_map = vor_data.get("index_map", {})
        reverse_map = {v: k for k, v in index_map.items()}

        for s_idx, i_idx in enumerate(sample_idx):
            i_orig = int(i_idx.item())
            i_vor = reverse_map.get(i_orig)
            if i_vor is None:
                continue
            neighbors_vor = neighbor_lists.get(i_vor, [])
            if not neighbors_vor:
                continue
            neighbors_orig = [index_map[n] for n in neighbors_vor if n in index_map]
            if not neighbors_orig:
                continue
            chosen = neighbors_orig[:k]
            if len(chosen) < k:
                chosen.extend([i_orig] * (k - len(chosen)))
            neighbor_idx[s_idx] = torch.tensor(chosen, device=device)

        if actual_sample_size < sample_size:
            sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
            neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


def compute_recorded_neighbors_batch(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    sample_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use recorded neighbor edges from RunHistory.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per sample.
        sample_size: Number of samples per timestep (None = all walkers).

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    if history.neighbor_edges is None:
        return compute_companion_batch(
            history, start_idx, neighbor_k, sample_size=sample_size
        )

    n_recorded = history.n_recorded
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
        neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)

        record_idx = start_idx + t
        edges = history.neighbor_edges[record_idx]
        if not torch.is_tensor(edges) or edges.numel() == 0:
            # Fallback to self-padding
            neighbor_idx[:] = sample_idx.unsqueeze(1)
        else:
            edge_list = edges.tolist()
            neighbor_map: dict[int, list[int]] = {}
            for i, j in edge_list:
                if i == j:
                    continue
                if i not in neighbor_map:
                    neighbor_map[i] = [j]
                else:
                    neighbor_map[i].append(j)

            for s_i, i_idx in enumerate(sample_idx.tolist()):
                neighbors = neighbor_map.get(i_idx, [])
                if not neighbors:
                    neighbor_idx[s_i] = i_idx
                    continue
                chosen = neighbors[:k]
                if len(chosen) < k:
                    chosen.extend([i_idx] * (k - len(chosen)))
                neighbor_idx[s_i] = torch.tensor(chosen, device=device)

        if actual_sample_size < sample_size:
            sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
            neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

        all_sample_idx.append(sample_idx)
        all_neighbor_idx.append(neighbor_idx)

    sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
    neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

    return sample_indices, neighbor_indices, alive


def compute_neighbors_auto(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    voronoi_pbc_mode: str = "mirror",
    voronoi_exclude_boundary: bool = True,
    voronoi_boundary_tolerance: float = 1e-6,
    sample_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Auto-detect and use best available neighbor data.

    Priority:
    1. Recorded neighbors (neighbor_edges) - O(E) lookup
    2. Companions - O(N) lookup
    3. Voronoi recomputation - O(N log N) fallback

    Args:
        history: RunHistory with neighbor data.
        start_idx: Starting timestep index.
        neighbor_k: Number of neighbors per sample.
        voronoi_pbc_mode: PBC handling mode for Voronoi (fallback).
        voronoi_exclude_boundary: Exclude boundary points (fallback).
        voronoi_boundary_tolerance: Tolerance for boundary detection (fallback).
        sample_size: Number of samples per timestep (None = all walkers).

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    n_recorded = history.n_recorded
    required_steps = n_recorded - start_idx

    # Check if recorded neighbors are available
    if history.neighbor_edges is not None and len(history.neighbor_edges) > 0:
        # Validate we have enough recorded steps
        available_steps = len(history.neighbor_edges)

        if available_steps >= required_steps:
            # Use recorded neighbors (optimal path)
            return compute_recorded_neighbors_batch(
                history=history,
                start_idx=start_idx,
                neighbor_k=neighbor_k,
                sample_size=sample_size,
            )
        else:
            # Partial recording - warn and fall back
            warnings.warn(
                f"Recorded neighbors only available for {available_steps} steps, "
                f"but {required_steps} steps requested. Falling back to companions.",
                UserWarning,
                stacklevel=2,
            )

    # Check if companions are available
    if history.companions_clone is not None:
        return compute_companion_batch(
            history=history,
            start_idx=start_idx,
            neighbor_k=neighbor_k,
            sample_size=sample_size,
        )

    # Last resort: recompute Voronoi
    warnings.warn(
        "No pre-computed neighbor data found. Recomputing Voronoi tessellation. "
        "This is expensive - consider setting neighbor_graph_record=True during simulation.",
        UserWarning,
        stacklevel=2,
    )
    return compute_voronoi_batch(
        history=history,
        start_idx=start_idx,
        neighbor_k=neighbor_k,
        voronoi_pbc_mode=voronoi_pbc_mode,
        voronoi_exclude_boundary=voronoi_exclude_boundary,
        voronoi_boundary_tolerance=voronoi_boundary_tolerance,
        sample_size=sample_size,
    )


# =============================================================================
# Dispatcher and Full Matrix Computation
# =============================================================================


def compute_neighbor_topology(
    history: RunHistory,
    start_idx: int,
    neighbor_method: str,
    neighbor_k: int,
    voronoi_config: dict,
    sample_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute neighbor indices for all timesteps.

    Dispatches to auto/companions/voronoi/recorded based on method.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_method: Neighbor selection method:
            - "auto": Auto-detect (recorded → companions → voronoi)
            - "recorded": Use history.neighbor_edges (fallback to companions)
            - "companions": Use history.companions_clone (alias "uniform" deprecated)
            - "voronoi": Recompute Voronoi tessellation
        neighbor_k: Number of neighbors per sample.
        voronoi_config: Voronoi configuration dict.
        sample_size: Number of samples per timestep (None = all walkers).

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
            voronoi_pbc_mode=voronoi_config.get("pbc_mode", "mirror"),
            voronoi_exclude_boundary=voronoi_config.get("exclude_boundary", True),
            voronoi_boundary_tolerance=voronoi_config.get("boundary_tolerance", 1e-6),
            sample_size=sample_size,
        )
    elif neighbor_method == "companions":
        return compute_companion_batch(history, start_idx, neighbor_k, sample_size)
    elif neighbor_method == "recorded":
        return compute_recorded_neighbors_batch(history, start_idx, neighbor_k, sample_size)
    elif neighbor_method == "voronoi":
        return compute_voronoi_batch(
            history,
            start_idx,
            neighbor_k,
            voronoi_config.get("pbc_mode", "mirror"),
            voronoi_config.get("exclude_boundary", True),
            voronoi_config.get("boundary_tolerance", 1e-6),
            sample_size,
        )
    else:
        raise ValueError(
            f"Unknown neighbor method: {neighbor_method}. "
            f"Valid options: 'auto', 'recorded', 'companions', 'voronoi'"
        )


def _compute_time_sliced_neighbor_matrix(
    history: RunHistory,
    start_idx: int,
    k: int,
    alive: Tensor,
    config: ChannelConfig,
) -> Tensor:
    """Compute time-sliced Voronoi neighbor matrix for Euclidean time.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        k: Number of neighbors.
        alive: Alive mask [T, N].
        config: ChannelConfig.

    Returns:
        Neighbor indices [T, N, k].
    """
    from fragile.fractalai.qft.aggregation import _resolve_mc_time_index

    T, N = alive.shape
    device = alive.device
    neighbor_indices = torch.zeros((T, N, k), dtype=torch.long, device=device)

    try:
        from fragile.fractalai.qft.voronoi_time_slices import compute_time_sliced_voronoi
    except Exception:
        return neighbor_indices

    if len(history.x_before_clone) == 0:
        return neighbor_indices

    mc_idx = _resolve_mc_time_index(history, config.mc_time_index)
    pos_idx = min(mc_idx, len(history.x_before_clone) - 1)
    positions = history.x_before_clone[pos_idx]
    alive_t = alive[0] if alive.shape[0] else torch.ones(N, dtype=torch.bool, device=device)

    time_sliced = compute_time_sliced_voronoi(
        positions=positions,
        time_dim=int(config.euclidean_time_dim),
        n_bins=int(config.euclidean_time_bins),
        min_walkers_bin=1,
        bounds=history.bounds,
        alive=alive_t,
        pbc=bool(history.pbc),
        pbc_mode=config.voronoi_pbc_mode,
        exclude_boundary=config.voronoi_exclude_boundary,
        boundary_tolerance=config.voronoi_boundary_tolerance,
        compute_curvature=False,
    )

    edges = _collect_time_sliced_edges(time_sliced, config.time_sliced_neighbor_mode)
    neighbor_lists = _build_neighbor_lists(edges, N)
    alive_set = set(int(i) for i in torch.where(alive_t)[0].tolist())

    neighbors_t = torch.zeros((N, k), dtype=torch.long, device=device)
    for i in range(N):
        if i not in alive_set:
            neighbors_t[i] = i
            continue
        choices = [j for j in neighbor_lists[i] if j in alive_set and j != i]
        if not choices:
            neighbors_t[i] = i
            continue
        if len(choices) < k:
            choices = choices + [i] * (k - len(choices))
        else:
            choices = choices[:k]
        neighbors_t[i] = torch.tensor(choices, device=device, dtype=torch.long)

    neighbor_indices[:] = neighbors_t.unsqueeze(0).expand(T, -1, -1)
    return neighbor_indices


def compute_full_neighbor_matrix(
    history: RunHistory,
    start_idx: int,
    neighbor_k: int,
    alive: Tensor,
    config: ChannelConfig,
) -> Tensor:
    """Compute neighbor indices for ALL walkers (not just samples).

    Used for Euclidean time mode where we need per-walker operators.
    Now supports auto-detection of pre-computed neighbors.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_k: Number of neighbors per walker.
        alive: Alive mask [T, N].
        config: ChannelConfig with time-slicing options.

    Returns:
        Neighbor indices [T, N, k] for all walkers.
    """
    T, N = alive.shape
    device = alive.device
    k = max(1, int(neighbor_k))

    neighbor_indices = torch.zeros((T, N, k), dtype=torch.long, device=device)

    # Handle deprecated "uniform" alias
    neighbor_method = config.neighbor_method
    if neighbor_method == "uniform":
        neighbor_method = "companions"

    # Handle auto-detection for recorded neighbors
    if neighbor_method == "auto":
        # Check for recorded neighbors at these timesteps
        if (history.neighbor_edges is not None and
            len(history.neighbor_edges) > start_idx):
            # Use recorded data for these timesteps
            for t in range(T):
                timestep_idx = start_idx + t
                if timestep_idx >= len(history.neighbor_edges):
                    break

                edges = history.neighbor_edges[timestep_idx]
                if not torch.is_tensor(edges) or edges.numel() == 0:
                    continue

                # Convert edge list to neighbor matrix
                edge_list = edges.tolist()
                neighbor_map: dict[int, list[int]] = {}
                for i, j in edge_list:
                    if i == j:
                        continue
                    if i not in neighbor_map:
                        neighbor_map[i] = []
                    neighbor_map[i].append(j)

                for i in range(N):
                    neighbors = neighbor_map.get(i, [])
                    if not neighbors:
                        neighbor_indices[t, i] = i
                        continue
                    chosen = neighbors[:k]
                    if len(chosen) < k:
                        chosen.extend([i] * (k - len(chosen)))
                    neighbor_indices[t, i] = torch.tensor(chosen, device=device, dtype=torch.long)

            # If we filled the matrix from recorded data, return it
            if neighbor_indices.abs().sum() > 0:
                return neighbor_indices

        # Fall back to companions if available
        if history.companions_clone is not None:
            for t in range(T):
                timestep_idx = start_idx + t - 1
                if timestep_idx < 0 or timestep_idx >= len(history.companions_clone):
                    continue

                companions = history.companions_clone[timestep_idx]
                alive_t = alive[min(t, alive.shape[0] - 1)]

                for i in range(N):
                    if not alive_t[i]:
                        neighbor_indices[t, i] = i
                        continue

                    comp = companions[i] if i < len(companions) else torch.tensor([i], device=device)
                    chosen = comp[:k].tolist() if len(comp) >= k else comp.tolist()
                    if len(chosen) < k:
                        chosen.extend([i] * (k - len(chosen)))
                    neighbor_indices[t, i] = torch.tensor(chosen, device=device, dtype=torch.long)

            return neighbor_indices

        # Fall through to voronoi if no pre-computed data

    # Handle time-sliced Voronoi for Euclidean time
    if (
        config.time_axis == "euclidean"
        and neighbor_method in ("voronoi", "auto")
        and config.use_time_sliced_tessellation
    ):
        return _compute_time_sliced_neighbor_matrix(
            history, start_idx, k, alive, config
        )

    # For other methods, use simple approaches
    for t in range(T):
        alive_t = alive[min(t, alive.shape[0] - 1)]
        alive_idx = torch.where(alive_t)[0]

        if len(alive_idx) == 0:
            continue

        # Simple: use companions or random alive walkers
        if history.companions_distance is not None and t + start_idx - 1 < len(history.companions_distance):
            comp_d = history.companions_distance[t + start_idx - 1]
            comp_c = history.companions_clone[t + start_idx - 1] if history.companions_clone is not None else comp_d

            for i in range(N):
                if not alive_t[i]:
                    neighbor_indices[t, i] = i
                    continue

                neighbors = []
                if i < len(comp_d):
                    neighbors.append(int(comp_d[i].item()))
                if i < len(comp_c) and len(neighbors) < k:
                    neighbors.append(int(comp_c[i].item()))

                # Fill remaining with random alive walkers
                while len(neighbors) < k:
                    other_alive = alive_idx[alive_idx != i]
                    if len(other_alive) > 0:
                        idx = torch.randint(0, len(other_alive), (1,), device=device).item()
                        neighbors.append(int(other_alive[idx].item()))
                    else:
                        neighbors.append(i)

                neighbor_indices[t, i] = torch.tensor(neighbors[:k], device=device, dtype=torch.long)
        else:
            # Fallback: random alive neighbors
            for i in range(N):
                if not alive_t[i]:
                    neighbor_indices[t, i] = i
                    continue

                other_alive = alive_idx[alive_idx != i]
                if len(other_alive) >= k:
                    perm = torch.randperm(len(other_alive), device=device)[:k]
                    neighbor_indices[t, i] = other_alive[perm]
                elif len(other_alive) > 0:
                    # Repeat if not enough neighbors
                    indices = other_alive[torch.randint(0, len(other_alive), (k,), device=device)]
                    neighbor_indices[t, i] = indices
                else:
                    neighbor_indices[t, i] = i

    return neighbor_indices
