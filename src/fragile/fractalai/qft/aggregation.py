"""Time series aggregation from RunHistory data.

This module handles preprocessing of Fractal Gas simulation data
into operator time series suitable for QFT channel analysis.

Main workflow:
    RunHistory → color states → neighbor topology → AggregatedTimeSeries

The output AggregatedTimeSeries contains all preprocessed data needed
for channel operator computation without requiring RunHistory access.

Usage:
    from fragile.fractalai.qft.aggregation import (
        aggregate_time_series,
        AggregatedTimeSeries,
    )

    agg_data = aggregate_time_series(history, config)
    # Use agg_data for custom analysis or pass to ChannelCorrelator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory
    from fragile.fractalai.qft.correlator_channels import ChannelConfig


# =============================================================================
# Neighbor Data Diagnostics
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
# Data Structures
# =============================================================================


@dataclass
class AggregatedTimeSeries:
    """Preprocessed data for channel correlator analysis.

    This is the interface between aggregation and analysis modules.
    Contains everything needed to compute channel operators without
    accessing RunHistory.
    """

    # Color state data
    color: Tensor  # [T, N, d] complex color states
    color_valid: Tensor  # [T, N] bool validity mask

    # Neighbor topology
    sample_indices: Tensor  # [T, S] sampled walker indices
    neighbor_indices: Tensor  # [T, S, k] neighbor indices per sample
    alive: Tensor  # [T, N] alive mask

    # Metadata
    n_timesteps: int  # T
    n_walkers: int  # N
    d: int  # spatial dimension
    dt: float  # time step size
    device: torch.device

    # For glueball channel (direct force access)
    force_viscous: Tensor | None = None  # [T, N, d]

    # For Euclidean time mode (per-walker operators)
    time_coords: Tensor | None = None  # bin centers
    full_neighbor_indices: Tensor | None = None  # [T, N, k] for all walkers (Euclidean time)


@dataclass
class OperatorTimeSeries:
    """Complete operator time series for all channels.

    Output of aggregation - everything needed for correlator analysis.
    Time type (MC vs Euclidean) is already handled - series is just series.
    """

    # Per-channel operator series
    operators: dict[str, Tensor]  # channel_name -> [T] series

    # Metadata (common to all channels)
    n_timesteps: int  # T
    dt: float  # effective time step
    time_axis: str  # "mc" or "euclidean" (for info only)
    time_coords: Tensor | None  # For Euclidean: bin centers [T]

    # Original preprocessing data (for diagnostics)
    aggregated_data: AggregatedTimeSeries

    # Per-channel metadata
    channel_metadata: dict[str, dict[str, Any]]  # Extra info per channel

    def get_series(self, channel: str) -> Tensor:
        """Convenience method to get a channel's series."""
        return self.operators[channel]


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


def _normalize_neighbor_method(method: str) -> str:
    """Normalize neighbor method name.

    Args:
        method: Neighbor method ("uniform" or other).

    Returns:
        Normalized method name ("companions" for "uniform", otherwise unchanged).
    """
    if method == "uniform":
        return "companions"
    return method


def _resolve_mc_time_index(history, mc_time_index: int | None) -> int:
    """Resolve a Monte Carlo slice index from either recorded index or step.

    Args:
        history: RunHistory object.
        mc_time_index: Monte Carlo time index (recorded index or step number).

    Returns:
        Resolved recorded index.

    Raises:
        ValueError: If index is out of bounds.
    """
    if history.n_recorded < 2:
        raise ValueError("Need at least 2 recorded timesteps for Euclidean analysis.")
    if mc_time_index is None:
        resolved = history.n_recorded - 1
    else:
        try:
            raw = int(mc_time_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mc_time_index: {mc_time_index}") from exc
        if raw in history.recorded_steps:
            resolved = history.get_step_index(raw)
        else:
            resolved = raw
    if resolved < 1 or resolved >= history.n_recorded:
        msg = (
            f"mc_time_index {resolved} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1} "
            "or a recorded step value)."
        )
        raise ValueError(msg)
    return resolved


# =============================================================================
# Color State Computation
# =============================================================================


def compute_color_states_batch(
    history: RunHistory,
    start_idx: int,
    h_eff: float,
    mass: float,
    ell0: float,
) -> tuple[Tensor, Tensor]:
    """Compute color states for all timesteps from start_idx onward.

    Vectorized across T dimension.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        h_eff: Effective Planck constant.
        mass: Particle mass.
        ell0: Length scale.

    Returns:
        Tuple of (color [T, N, d], valid [T, N]).
    """
    n_recorded = history.n_recorded
    T = n_recorded - start_idx

    # Extract batched tensors
    v_pre = history.v_before_clone[start_idx:]  # [T, N, d]
    force_visc = history.force_viscous[start_idx - 1 : n_recorded - 1]  # [T, N, d]

    # Color state computation (vectorized)
    phase = (mass * v_pre * ell0) / h_eff
    complex_phase = torch.polar(torch.ones_like(phase), phase.float())

    if force_visc.dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64

    tilde = force_visc.to(complex_dtype) * complex_phase.to(complex_dtype)
    norm = torch.linalg.vector_norm(tilde, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    color = tilde / norm
    valid = norm.squeeze(-1) > 1e-12

    return color, valid


# =============================================================================
# Length Scale Estimation
# =============================================================================


def estimate_ell0(history: RunHistory) -> float:
    """Estimate ell0 from median companion distance at mid-point.

    Args:
        history: RunHistory object.

    Returns:
        Estimated ell0 value.
    """
    mid_idx = history.n_recorded // 2
    if mid_idx == 0:
        return 1.0

    x_pre = history.x_before_clone[mid_idx]
    comp_idx = history.companions_distance[mid_idx - 1]
    alive = history.alive_mask[mid_idx - 1]

    # Compute distances
    diff = x_pre - x_pre[comp_idx]
    if history.pbc and history.bounds is not None:
        high = history.bounds.high.to(x_pre)
        low = history.bounds.low.to(x_pre)
        span = high - low
        diff = diff - span * torch.round(diff / span)
    dist = torch.linalg.vector_norm(diff, dim=-1)

    if dist.numel() > 0 and alive.any():
        return float(dist[alive].median().item())
    else:
        return 1.0


# =============================================================================
# Neighbor Topology Computation
# =============================================================================


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
    import warnings

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
            - "companions": Use history.companions_clone
            - "voronoi": Recompute Voronoi tessellation
        neighbor_k: Number of neighbors per sample.
        voronoi_config: Voronoi configuration dict.
        sample_size: Number of samples per timestep (None = all walkers).

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    method = _normalize_neighbor_method(neighbor_method)

    if method == "auto":
        return compute_neighbors_auto(
            history=history,
            start_idx=start_idx,
            neighbor_k=neighbor_k,
            voronoi_pbc_mode=voronoi_config.get("pbc_mode", "mirror"),
            voronoi_exclude_boundary=voronoi_config.get("exclude_boundary", True),
            voronoi_boundary_tolerance=voronoi_config.get("boundary_tolerance", 1e-6),
            sample_size=sample_size,
        )
    elif method == "companions":
        return compute_companion_batch(history, start_idx, neighbor_k, sample_size)
    elif method == "recorded":
        return compute_recorded_neighbors_batch(history, start_idx, neighbor_k, sample_size)
    elif method == "voronoi":
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
            f"Unknown neighbor method: {method}. "
            f"Valid options: 'auto', 'recorded', 'companions', 'voronoi'"
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

    method = _normalize_neighbor_method(config.neighbor_method)

    # Handle auto-detection for recorded neighbors
    if method == "auto":
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
        and method in ("voronoi", "auto")
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


# =============================================================================
# Euclidean Time Binning
# =============================================================================


def bin_by_euclidean_time(
    positions: Tensor,
    operators: Tensor,
    alive: Tensor,
    time_dim: int = 3,
    n_bins: int = 50,
    time_range: tuple[float, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Bin walkers by Euclidean time coordinate and compute mean operator per bin.

    In 4D simulations (3 spatial + 1 Euclidean time), this function treats one
    spatial dimension as a time coordinate and computes operator averages within
    time bins. This enables lattice QFT analysis where correlators are computed
    over spatial separation in the time dimension rather than Monte Carlo timesteps.

    Args:
        positions: Walker positions over MC time [T, N, d]
        operators: Operator values to average [T, N]
        alive: Alive mask [T, N]
        time_dim: Which spatial dimension is Euclidean time (0-indexed, default 3)
        n_bins: Number of time bins
        time_range: (t_min, t_max) or None for auto from data

    Returns:
        time_coords: Bin centers [n_bins]
        operator_series: Mean operator vs Euclidean time [n_bins]

    Example:
        >>> # 4D simulation with d=4, treat 4th dim as time
        >>> positions = history.x_before_clone  # [T, N, 4]
        >>> operators = compute_scalar_operators(...)  # [T, N]
        >>> alive = history.alive_mask  # [T, N]
        >>> time_coords, series = bin_by_euclidean_time(positions, operators, alive)
        >>> correlator = compute_correlator_fft(series, max_lag=40)
    """
    # Extract Euclidean time coordinate
    t_euc = positions[:, :, time_dim]  # [T, N]

    # Flatten over MC time dimension to treat all snapshots as ensemble
    t_euc_flat = t_euc[alive]  # [total_alive_walkers]
    ops_flat = operators[alive]

    if t_euc_flat.numel() == 0:
        # No alive walkers
        device = positions.device
        return torch.zeros(n_bins, device=device), torch.zeros(n_bins, device=device)

    # Determine time range
    if time_range is None:
        t_min, t_max = t_euc_flat.min().item(), t_euc_flat.max().item()
        # Add small padding to avoid edge effects
        padding = (t_max - t_min) * 0.01
        t_min -= padding
        t_max += padding
    else:
        t_min, t_max = time_range

    # Create bins
    edges = torch.linspace(t_min, t_max, n_bins + 1, device=positions.device)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Bin operators using vectorized histogram
    operator_series = torch.zeros(n_bins, device=positions.device)
    counts = torch.zeros(n_bins, device=positions.device)

    for i in range(n_bins):
        mask = (t_euc_flat >= edges[i]) & (t_euc_flat < edges[i + 1])
        count = mask.sum()
        if count > 0:
            operator_series[i] = ops_flat[mask].sum()
            counts[i] = count.float()

    # Handle last bin inclusively
    mask = t_euc_flat == edges[-1]
    if mask.sum() > 0:
        operator_series[-1] += ops_flat[mask].sum()
        counts[-1] += mask.sum().float()

    # Average
    valid = counts > 0
    operator_series[valid] = operator_series[valid] / counts[valid]
    operator_series[~valid] = 0.0

    return bin_centers, operator_series


# =============================================================================
# Main Aggregation Function
# =============================================================================


def aggregate_time_series(
    history: RunHistory,
    config: ChannelConfig,
) -> AggregatedTimeSeries:
    """Main entry point: preprocess RunHistory into aggregated time series.

    This function:
    1. Validates config and estimates ell0 if needed
    2. Computes color states from velocities and forces
    3. Builds neighbor topology (auto/voronoi/companions/recorded)
    4. Packages everything into AggregatedTimeSeries

    Neighbor Method Selection:
        The neighbor_method parameter controls how neighbor topology is computed:

        - "auto" (default): Auto-detects best available method
            1. Uses history.neighbor_edges if available (O(E) lookup)
            2. Falls back to history.companions_clone (O(N) lookup)
            3. Falls back to Voronoi recomputation (O(N log N))

        - "recorded": Explicitly use history.neighbor_edges
            Requires neighbor_graph_record=True during simulation
            Fastest method when available

        - "companions": Use history.companions_clone
            Limited to companion walkers only

        - "voronoi": Recompute Delaunay/Voronoi tessellation
            Most expensive but works without pre-computed data
            Necessary when neighbor_edges not available

    Performance:
        Pre-computed neighbors (recorded) are ~10-100x faster than Voronoi
        recomputation for large walker populations. Always prefer "auto" or
        "recorded" when analyzing simulation runs.

    Used internally by ChannelCorrelator.compute_series()

    Args:
        history: RunHistory object.
        config: Channel configuration.

    Returns:
        AggregatedTimeSeries with all preprocessed data.
    """
    # Determine start index
    start_idx = max(1, int(history.n_recorded * config.warmup_fraction))

    # Estimate ell0 if not provided
    ell0 = config.ell0
    if ell0 is None:
        ell0 = estimate_ell0(history)

    # Compute color states
    color, color_valid = compute_color_states_batch(
        history,
        start_idx,
        config.h_eff,
        config.mass,
        ell0,
    )

    # Compute neighbor topology
    voronoi_config = {
        "pbc_mode": config.voronoi_pbc_mode,
        "exclude_boundary": config.voronoi_exclude_boundary,
        "boundary_tolerance": config.voronoi_boundary_tolerance,
    }

    sample_indices, neighbor_indices, alive = compute_neighbor_topology(
        history,
        start_idx,
        config.neighbor_method,
        config.neighbor_k,
        voronoi_config,
        sample_size=None,
    )

    # Extract force for glueball channel
    n_recorded = history.n_recorded
    force_viscous = history.force_viscous[start_idx - 1 : n_recorded - 1]

    # Compute metadata
    T = color.shape[0]
    N = history.N
    d = history.d
    dt = float(history.delta_t * history.record_every)
    device = color.device

    return AggregatedTimeSeries(
        color=color,
        color_valid=color_valid,
        sample_indices=sample_indices,
        neighbor_indices=neighbor_indices,
        alive=alive,
        n_timesteps=T,
        n_walkers=N,
        d=d,
        dt=dt,
        device=device,
        force_viscous=force_viscous,
        time_coords=None,
    )


# =============================================================================
# Gamma Matrices for Bilinear Projections
# =============================================================================


def build_gamma_matrices(
    d: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Build gamma matrices for bilinear projections.

    Extracted from ChannelCorrelator._build_gamma_matrices().

    Args:
        d: Spatial dimension.
        device: Compute device.

    Returns:
        Dictionary with keys: "1", "5", "5_matrix", "mu", "5mu", "sigma"
    """
    dtype = torch.complex128

    gamma: dict[str, Tensor] = {}

    # Identity (scalar channel)
    gamma["1"] = torch.eye(d, device=device, dtype=dtype)

    # γ₅ diagonal (pseudoscalar) - alternating signs
    gamma5_diag = torch.tensor(
        [(-1.0) ** i for i in range(d)],
        device=device,
        dtype=dtype,
    )
    gamma["5"] = gamma5_diag  # Store just diagonal for efficiency
    gamma["5_matrix"] = torch.diag(gamma5_diag)

    # γ_μ matrices (vector)
    gamma_mu_list = []
    for mu in range(d):
        gamma_mu = torch.zeros(d, d, device=device, dtype=dtype)
        gamma_mu[mu, mu] = 1.0
        if mu > 0:
            gamma_mu[mu, 0] = 0.5j
            gamma_mu[0, mu] = -0.5j
        gamma_mu_list.append(gamma_mu)
    gamma["mu"] = torch.stack(gamma_mu_list, dim=0)  # [d, d, d]

    # γ₅γ_μ matrices (axial vector)
    gamma_5mu_list = []
    for mu in range(d):
        gamma_5mu = gamma["5_matrix"] @ gamma_mu_list[mu]
        gamma_5mu_list.append(gamma_5mu)
    gamma["5mu"] = torch.stack(gamma_5mu_list, dim=0)  # [d, d, d]

    # σ_μν matrices (tensor)
    sigma_list = []
    for mu in range(d):
        for nu in range(mu + 1, d):
            sigma = torch.zeros(d, d, device=device, dtype=dtype)
            sigma[mu, nu] = 1.0j
            sigma[nu, mu] = -1.0j
            sigma_list.append(sigma)
    if sigma_list:
        gamma["sigma"] = torch.stack(sigma_list, dim=0)  # [n_pairs, d, d]
    else:
        gamma["sigma"] = torch.zeros(0, d, d, device=device, dtype=dtype)

    return gamma


# =============================================================================
# Channel-Specific Operator Computation
# =============================================================================


def compute_scalar_operators(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute scalar channel operators: ψ̄_i · ψ_j.

    Extracted from ScalarChannel._compute_operators_vectorized().

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator series [T]
    """
    color = agg_data.color
    valid = agg_data.color_valid
    alive = agg_data.alive
    sample_indices = agg_data.sample_indices
    neighbor_indices = agg_data.neighbor_indices

    T, N, d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states for samples and first neighbors
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]

    # Use first neighbor
    first_neighbor = neighbor_indices[:, :, 0]  # [T, S]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # Identity projection: simple dot product
    op_values = (color_i.conj() * color_j).sum(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Mean over samples per timestep
    counts = valid_mask.sum(dim=1).clamp(min=1)
    series = op_values.sum(dim=1) / counts

    return series


def compute_pseudoscalar_operators(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute pseudoscalar channel operators: ψ̄_i γ₅ ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator series [T]
    """
    color = agg_data.color
    valid = agg_data.color_valid
    alive = agg_data.alive
    sample_indices = agg_data.sample_indices
    neighbor_indices = agg_data.neighbor_indices

    T, N, d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # γ₅ projection: alternating sign dot product
    gamma5_diag = gamma_matrices["5"].to(color_i.device)
    op_values = (color_i.conj() * gamma5_diag * color_j).sum(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Mean over samples
    counts = valid_mask.sum(dim=1).clamp(min=1)
    series = op_values.sum(dim=1) / counts

    return series


def compute_vector_operators(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute vector channel operators: Σ_μ ψ̄_i γ_μ ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator series [T]
    """
    color = agg_data.color
    valid = agg_data.color_valid
    alive = agg_data.alive
    sample_indices = agg_data.sample_indices
    neighbor_indices = agg_data.neighbor_indices

    T, N, d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # γ_μ projection using einsum
    gamma_mu = gamma_matrices["mu"].to(color_i.device, dtype=color_i.dtype)
    result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
    op_values = result.mean(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Mean over samples
    counts = valid_mask.sum(dim=1).clamp(min=1)
    series = op_values.sum(dim=1) / counts

    return series


def compute_axial_vector_operators(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute axial vector channel operators: Σ_μ ψ̄_i γ₅γ_μ ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator series [T]
    """
    color = agg_data.color
    valid = agg_data.color_valid
    alive = agg_data.alive
    sample_indices = agg_data.sample_indices
    neighbor_indices = agg_data.neighbor_indices

    T, N, d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # γ₅γ_μ projection
    gamma_5mu = gamma_matrices["5mu"].to(color_i.device, dtype=color_i.dtype)
    result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_5mu, color_j)
    op_values = result.mean(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Mean over samples
    counts = valid_mask.sum(dim=1).clamp(min=1)
    series = op_values.sum(dim=1) / counts

    return series


def compute_tensor_operators(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute tensor channel operators: Σ_{μ<ν} ψ̄_i σ_μν ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator series [T]
    """
    color = agg_data.color
    valid = agg_data.color_valid
    alive = agg_data.alive
    sample_indices = agg_data.sample_indices
    neighbor_indices = agg_data.neighbor_indices

    T, N, d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # σ_μν projection
    sigma = gamma_matrices["sigma"].to(color_i.device, dtype=color_i.dtype)
    if sigma.shape[0] == 0:
        return torch.zeros(T, device=device)

    result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
    op_values = result.mean(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Mean over samples
    counts = valid_mask.sum(dim=1).clamp(min=1)
    series = op_values.sum(dim=1) / counts

    return series


def compute_nucleon_operators(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute nucleon channel operators: det([ψ_i, ψ_j, ψ_k]).

    Requires d>=3 (uses first 3 spatial components).

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary (unused for nucleon).

    Returns:
        Operator series [T]
    """
    color = agg_data.color
    valid = agg_data.color_valid
    alive = agg_data.alive
    sample_indices = agg_data.sample_indices
    neighbor_indices = agg_data.neighbor_indices

    T, N, d = color.shape
    device = color.device

    if d < 3:
        # Nucleon requires at least 3 spatial dimensions
        return torch.zeros(T, device=device)

    # Use only first 3 components
    color = color[..., :3]

    S = sample_indices.shape[1]
    k = neighbor_indices.shape[2]

    if k < 2:
        return torch.zeros(T, device=device)

    # Gather indices
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)

    # Color states
    color_i = color[t_idx, sample_indices]  # [T, S, 3]
    color_j = color[t_idx, neighbor_indices[:, :, 0]]  # [T, S, 3]
    color_k = color[t_idx, neighbor_indices[:, :, 1]]  # [T, S, 3]

    # Stack to form 3x3 matrix: [T, S, 3, 3]
    matrix = torch.stack([color_i, color_j, color_k], dim=-1)

    # Compute determinant: [T, S]
    det = torch.linalg.det(matrix)

    # Validity mask
    valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    valid_j = valid[t_idx, neighbor_indices[:, :, 0]] & alive[
        t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 0]
    ]
    valid_k = valid[t_idx, neighbor_indices[:, :, 1]] & alive[
        t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 1]
    ]
    valid_mask = valid_i & valid_j & valid_k

    # Mask invalid
    det = torch.where(valid_mask, det, torch.zeros_like(det))

    # Mean over samples
    counts = valid_mask.sum(dim=1).clamp(min=1)
    series = det.sum(dim=1) / counts

    return series.real if series.is_complex() else series


def compute_glueball_operators(
    agg_data: AggregatedTimeSeries,
) -> Tensor:
    """Compute glueball channel operators: ||force||².

    Args:
        agg_data: Aggregated time series data.

    Returns:
        Operator series [T]
    """
    force = agg_data.force_viscous
    alive = agg_data.alive
    T = force.shape[0]
    device = force.device

    # Force squared norm: [T, N]
    force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)

    # Average over alive walkers per timestep
    series = []
    for t in range(T):
        alive_t = alive[t] if t < alive.shape[0] else torch.ones(force.shape[1], dtype=torch.bool, device=device)
        if alive_t.any():
            series.append(force_sq[t, alive_t].mean())
        else:
            series.append(torch.tensor(0.0, device=device))

    return torch.stack(series)


# =============================================================================
# Per-Walker Operator Computation (for Euclidean Time)
# =============================================================================


def _apply_bilinear_projection_per_walker(
    color: Tensor,
    neighbor_indices: Tensor,
    valid: Tensor,
    alive: Tensor,
    gamma_projection_func: callable,
) -> Tensor:
    """Apply bilinear projection for all walkers.

    Helper function for per-walker bilinear operator computation.

    Args:
        color: Color states [T, N, d].
        neighbor_indices: Neighbor indices [T, N, k].
        valid: Valid color flags [T, N].
        alive: Alive walker flags [T, N].
        gamma_projection_func: Function(color_i, color_j) -> operator values.

    Returns:
        Operator values [T, N] for each walker.
    """
    T, N, d = color.shape
    device = color.device

    # Use first neighbor for each walker
    first_neighbor = neighbor_indices[:, :, 0]  # [T, N]

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, N)
    color_i = color  # [T, N, d] (walkers themselves)
    color_j = color[t_idx, first_neighbor]  # [T, N, d] (their neighbors)

    # Validity masks
    valid_i = valid & alive
    valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    self_neighbor = first_neighbor == torch.arange(N, device=device).unsqueeze(0)
    valid_mask = valid_i & valid_j & (~self_neighbor)

    # Apply channel-specific projection
    op_values = gamma_projection_func(color_i, color_j)  # [T, N]

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    return op_values


def compute_scalar_operators_per_walker(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute scalar operators for each walker: ψ̄_i · ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator values [T, N] for each walker.
    """
    if agg_data.full_neighbor_indices is None:
        raise ValueError("full_neighbor_indices required for per-walker computation")

    def scalar_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        return (color_i.conj() * color_j).sum(dim=-1).real

    return _apply_bilinear_projection_per_walker(
        agg_data.color,
        agg_data.full_neighbor_indices,
        agg_data.color_valid,
        agg_data.alive,
        scalar_projection,
    )


def compute_pseudoscalar_operators_per_walker(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute pseudoscalar operators for each walker: ψ̄_i γ₅ ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator values [T, N] for each walker.
    """
    if agg_data.full_neighbor_indices is None:
        raise ValueError("full_neighbor_indices required for per-walker computation")

    gamma5_diag = gamma_matrices["5"]

    def pseudoscalar_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        return (color_i.conj() * gamma5_diag.to(color_i.device) * color_j).sum(dim=-1).real

    return _apply_bilinear_projection_per_walker(
        agg_data.color,
        agg_data.full_neighbor_indices,
        agg_data.color_valid,
        agg_data.alive,
        pseudoscalar_projection,
    )


def compute_vector_operators_per_walker(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute vector operators for each walker: Σ_μ ψ̄_i γ_μ ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator values [T, N] for each walker.
    """
    if agg_data.full_neighbor_indices is None:
        raise ValueError("full_neighbor_indices required for per-walker computation")

    gamma_mu = gamma_matrices["mu"]

    def vector_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu.to(color_i.device, dtype=color_i.dtype), color_j)
        return result.mean(dim=-1).real

    return _apply_bilinear_projection_per_walker(
        agg_data.color,
        agg_data.full_neighbor_indices,
        agg_data.color_valid,
        agg_data.alive,
        vector_projection,
    )


def compute_axial_vector_operators_per_walker(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute axial vector operators for each walker: Σ_μ ψ̄_i γ₅γ_μ ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator values [T, N] for each walker.
    """
    if agg_data.full_neighbor_indices is None:
        raise ValueError("full_neighbor_indices required for per-walker computation")

    gamma_5mu = gamma_matrices["5mu"]

    def axial_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_5mu.to(color_i.device, dtype=color_i.dtype), color_j)
        return result.mean(dim=-1).real

    return _apply_bilinear_projection_per_walker(
        agg_data.color,
        agg_data.full_neighbor_indices,
        agg_data.color_valid,
        agg_data.alive,
        axial_projection,
    )


def compute_tensor_operators_per_walker(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute tensor operators for each walker: Σ_{μ<ν} ψ̄_i σ_μν ψ_j.

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator values [T, N] for each walker.
    """
    if agg_data.full_neighbor_indices is None:
        raise ValueError("full_neighbor_indices required for per-walker computation")

    sigma = gamma_matrices["sigma"]

    if sigma.shape[0] == 0:
        T, N = agg_data.color.shape[:2]
        return torch.zeros(T, N, device=agg_data.device)

    def tensor_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma.to(color_i.device, dtype=color_i.dtype), color_j)
        return result.mean(dim=-1).real

    return _apply_bilinear_projection_per_walker(
        agg_data.color,
        agg_data.full_neighbor_indices,
        agg_data.color_valid,
        agg_data.alive,
        tensor_projection,
    )


def compute_nucleon_operators_per_walker(
    agg_data: AggregatedTimeSeries,
    gamma_matrices: dict[str, Tensor],
) -> Tensor:
    """Compute nucleon operators for each walker: det([ψ_i, ψ_j, ψ_k]).

    Args:
        agg_data: Aggregated time series data.
        gamma_matrices: Gamma matrices dictionary.

    Returns:
        Operator values [T, N] for each walker.
    """
    if agg_data.full_neighbor_indices is None:
        raise ValueError("full_neighbor_indices required for per-walker computation")

    color = agg_data.color
    T, N, d = color.shape
    device = agg_data.device

    if d < 3:
        return torch.zeros(T, N, device=device)

    # Use only first 3 components
    color = color[..., :3]
    neighbor_indices = agg_data.full_neighbor_indices

    if neighbor_indices.shape[2] < 2:
        return torch.zeros(T, N, device=device)

    # Gather indices
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, N)

    # Color states for triplets
    color_i = color  # [T, N, 3]
    color_j = color[t_idx, neighbor_indices[:, :, 0]]  # [T, N, 3]
    color_k = color[t_idx, neighbor_indices[:, :, 1]]  # [T, N, 3]

    # Stack to form 3x3 matrix: [T, N, 3, 3]
    matrix = torch.stack([color_i, color_j, color_k], dim=-1)

    # Compute determinant: [T, N]
    det = torch.linalg.det(matrix)

    # Validity mask
    valid = agg_data.color_valid
    alive = agg_data.alive
    valid_i = valid & alive
    valid_j = valid[t_idx, neighbor_indices[:, :, 0]] & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 0]]
    valid_k = valid[t_idx, neighbor_indices[:, :, 1]] & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 1]]
    valid_mask = valid_i & valid_j & valid_k

    # Mask invalid
    det = torch.where(valid_mask, det, torch.zeros_like(det))

    return det.real if det.is_complex() else det


def compute_glueball_operators_per_walker(
    agg_data: AggregatedTimeSeries,
) -> Tensor:
    """Compute glueball operators for each walker: ||force_i||².

    Args:
        agg_data: Aggregated time series data.

    Returns:
        Operator values [T, N] for each walker.
    """
    force = agg_data.force_viscous
    alive = agg_data.alive

    # Force squared norm: [T, N]
    force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)

    # Mask dead walkers
    force_sq = torch.where(alive, force_sq, torch.zeros_like(force_sq))

    return force_sq


# =============================================================================
# Main Operator Computation Function
# =============================================================================


def compute_all_operator_series(
    history: RunHistory,
    config: ChannelConfig,
    channels: list[str] | None = None,
) -> OperatorTimeSeries:
    """Compute operator series for all channels in ONE PASS.

    This is the main aggregation entry point. Handles both MC and Euclidean time modes.

    Workflow:
        1. Preprocess: aggregate_time_series() → AggregatedTimeSeries
        2. Build gamma matrices (once for all channels)
        3. For MC time: compute averaged operators → series [T]
           For Euclidean time: compute per-walker operators → bin by Euclidean time → series [n_bins]
        4. Package into OperatorTimeSeries

    Args:
        history: RunHistory object.
        config: ChannelConfig (aggregation configuration).
        channels: List of channel names (None = all).

    Returns:
        OperatorTimeSeries with all operator series computed.
    """
    # 1. Preprocess
    agg_data = aggregate_time_series(history, config)

    # 2. Build gamma matrices once
    gamma = build_gamma_matrices(agg_data.d, agg_data.device)

    # 3. Set up channels
    if channels is None:
        channels = ["scalar", "pseudoscalar", "vector", "axial_vector",
                   "tensor", "nucleon", "glueball"]

    # Filter channels based on dimensionality
    if agg_data.d < 3:
        channels = [ch for ch in channels if ch != "nucleon"]

    operators = {}
    channel_metadata = {}

    # Handle MC vs Euclidean time
    if config.time_axis == "euclidean":
        # Euclidean time mode: per-walker operators + binning
        operators, channel_metadata, time_coords, n_timesteps = _compute_euclidean_time_series(
            history, config, agg_data, gamma, channels
        )
    else:
        # MC time mode: averaged operators
        for channel_name in channels:
            if channel_name == "scalar":
                ops = compute_scalar_operators(agg_data, gamma)
            elif channel_name == "pseudoscalar":
                ops = compute_pseudoscalar_operators(agg_data, gamma)
            elif channel_name == "vector":
                ops = compute_vector_operators(agg_data, gamma)
            elif channel_name == "axial_vector":
                ops = compute_axial_vector_operators(agg_data, gamma)
            elif channel_name == "tensor":
                ops = compute_tensor_operators(agg_data, gamma)
            elif channel_name == "nucleon":
                ops = compute_nucleon_operators(agg_data, gamma)
            elif channel_name == "glueball":
                ops = compute_glueball_operators(agg_data)
            else:
                continue

            operators[channel_name] = ops
            channel_metadata[channel_name] = {
                "n_samples": len(ops),
            }

        time_coords = agg_data.time_coords
        n_timesteps = agg_data.n_timesteps

    return OperatorTimeSeries(
        operators=operators,
        n_timesteps=n_timesteps,
        dt=agg_data.dt,
        time_axis=config.time_axis,
        time_coords=time_coords,
        aggregated_data=agg_data,
        channel_metadata=channel_metadata,
    )


def _compute_euclidean_time_series(
    history: RunHistory,
    config: ChannelConfig,
    agg_data: AggregatedTimeSeries,
    gamma: dict[str, Tensor],
    channels: list[str],
) -> tuple[dict[str, Tensor], dict[str, dict], Tensor, int]:
    """Compute operator series for Euclidean time mode.

    Args:
        history: RunHistory object.
        config: ChannelConfig.
        agg_data: Aggregated time series data.
        gamma: Gamma matrices.
        channels: List of channel names.

    Returns:
        Tuple of (operators dict, channel_metadata dict, time_coords, n_timesteps).
    """
    # Check dimension
    if history.d < config.euclidean_time_dim + 1:
        msg = (
            f"Cannot use dimension {config.euclidean_time_dim} as Euclidean time "
            f"(only {history.d} dimensions available)"
        )
        raise ValueError(msg)

    # Resolve MC time index for position extraction
    start_idx = _resolve_mc_time_index(history, config.mc_time_index)

    # Get positions for Euclidean time extraction (single MC timestep)
    positions = history.x_before_clone[start_idx : start_idx + 1]  # [1, N, d]
    alive = history.alive_mask[start_idx - 1 : start_idx]  # [1, N]

    # Compute full neighbor matrix for all walkers
    full_neighbors = compute_full_neighbor_matrix(
        history, start_idx, config.neighbor_k, alive, config
    )

    # Create modified agg_data with full neighbors
    agg_data_full = AggregatedTimeSeries(
        color=agg_data.color[:1],  # Just one timestep
        color_valid=agg_data.color_valid[:1],
        sample_indices=agg_data.sample_indices[:1],
        neighbor_indices=agg_data.neighbor_indices[:1],
        alive=alive,
        n_timesteps=1,
        n_walkers=agg_data.n_walkers,
        d=agg_data.d,
        dt=agg_data.dt,
        device=agg_data.device,
        force_viscous=agg_data.force_viscous[:1] if agg_data.force_viscous is not None else None,
        time_coords=None,
        full_neighbor_indices=full_neighbors[:1],  # [1, N, k]
    )

    operators = {}
    channel_metadata = {}

    # Compute per-walker operators for each channel
    for channel_name in channels:
        if channel_name == "scalar":
            ops_per_walker = compute_scalar_operators_per_walker(agg_data_full, gamma)
        elif channel_name == "pseudoscalar":
            ops_per_walker = compute_pseudoscalar_operators_per_walker(agg_data_full, gamma)
        elif channel_name == "vector":
            ops_per_walker = compute_vector_operators_per_walker(agg_data_full, gamma)
        elif channel_name == "axial_vector":
            ops_per_walker = compute_axial_vector_operators_per_walker(agg_data_full, gamma)
        elif channel_name == "tensor":
            ops_per_walker = compute_tensor_operators_per_walker(agg_data_full, gamma)
        elif channel_name == "nucleon":
            ops_per_walker = compute_nucleon_operators_per_walker(agg_data_full, gamma)
        elif channel_name == "glueball":
            ops_per_walker = compute_glueball_operators_per_walker(agg_data_full)
        else:
            continue

        # Bin by Euclidean time
        time_coords, series = bin_by_euclidean_time(
            positions=positions,
            operators=ops_per_walker,  # [1, N]
            alive=alive,
            time_dim=config.euclidean_time_dim,
            n_bins=config.euclidean_time_bins,
            time_range=config.euclidean_time_range,
        )

        operators[channel_name] = series
        channel_metadata[channel_name] = {
            "n_samples": len(series),
        }

    n_timesteps = len(time_coords)
    return operators, channel_metadata, time_coords, n_timesteps
