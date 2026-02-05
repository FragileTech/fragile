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

    # For Euclidean time mode
    time_coords: Tensor | None = None  # bin centers


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

    Dispatches to companions/voronoi/recorded based on method.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        neighbor_method: Neighbor selection method.
        neighbor_k: Number of neighbors per sample.
        voronoi_config: Voronoi configuration dict.
        sample_size: Number of samples per timestep (None = all walkers).

    Returns:
        Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
    """
    method = _normalize_neighbor_method(neighbor_method)

    if method == "companions":
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
        return compute_companion_batch(history, start_idx, neighbor_k, sample_size)


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
    3. Builds neighbor topology (voronoi/companions/recorded)
    4. Packages everything into AggregatedTimeSeries

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
# Main Operator Computation Function
# =============================================================================


def compute_all_operator_series(
    history: RunHistory,
    config: ChannelConfig,
    channels: list[str] | None = None,
) -> OperatorTimeSeries:
    """Compute operator series for all channels in ONE PASS.

    This is the main aggregation entry point. Replaces the need to
    instantiate individual channel objects.

    Workflow:
        1. Preprocess: aggregate_time_series() → AggregatedTimeSeries
        2. Build gamma matrices (once for all channels)
        3. For each channel, compute operators using channel-specific function
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

    # 3. Compute operators per channel
    if channels is None:
        channels = ["scalar", "pseudoscalar", "vector", "axial_vector",
                   "tensor", "nucleon", "glueball"]

    # Filter channels based on dimensionality
    if agg_data.d < 3:
        channels = [ch for ch in channels if ch != "nucleon"]

    operators = {}
    channel_metadata = {}

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

    return OperatorTimeSeries(
        operators=operators,
        n_timesteps=agg_data.n_timesteps,
        dt=agg_data.dt,
        time_axis=config.time_axis,
        time_coords=agg_data.time_coords,
        aggregated_data=agg_data,
        channel_metadata=channel_metadata,
    )
