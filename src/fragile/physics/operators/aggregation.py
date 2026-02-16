"""Time series aggregation from RunHistory data.

This module handles preprocessing of Fractal Gas simulation data
into operator time series suitable for QFT channel analysis.

Main workflow:
    RunHistory → color states → neighbor topology → AggregatedTimeSeries

The output AggregatedTimeSeries contains all preprocessed data needed
for channel operator computation without requiring RunHistory access.

Neighbor Computation:
    Neighbor topology computation has been moved to neighbor_analysis.py
    for better modularity. This module imports and uses those functions.

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
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F

from fragile.fractalai.qft.neighbor_analysis import (
    compute_full_neighbor_matrix,
    compute_neighbor_topology,
)


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

    # Geometric weighting data (from pre-computed scutoid data)
    sample_edge_weights: Tensor | None = None  # [T, S] geodesic dist for (sample, 1st neighbor)
    sample_volume_weights: Tensor | None = None  # [T, S] sqrt(det g) for sample walkers
    operator_weighting: str = "uniform"  # weighting mode from config

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
        msg = "Need at least 2 recorded timesteps for Euclidean analysis."
        raise ValueError(msg)
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
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute color states for all timesteps from start_idx onward.

    Vectorized across T dimension.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        h_eff: Effective Planck constant.
        mass: Particle mass.
        ell0: Length scale.
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (color [T, N, d], valid [T, N]).
    """
    n_recorded = end_idx if end_idx is not None else history.n_recorded
    n_recorded - start_idx

    # Extract batched tensors
    v_pre = history.v_before_clone[start_idx:n_recorded]  # [T, N, d]
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
    return 1.0


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
    3. Builds neighbor topology (via neighbor_analysis module)
    4. Packages everything into AggregatedTimeSeries

    Neighbor Method Selection:
        Neighbor topology is computed by fragile.fractalai.qft.neighbor_analysis.
        See compute_neighbor_topology() for method details.

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
    # Determine start and end indices
    start_idx = max(1, int(history.n_recorded * config.warmup_fraction))
    end_fraction = getattr(config, "end_fraction", 1.0)
    end_idx = max(start_idx + 1, int(history.n_recorded * end_fraction))

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
        end_idx=end_idx,
    )

    # Compute neighbor topology
    sample_indices, neighbor_indices, alive = compute_neighbor_topology(
        history,
        start_idx,
        config.neighbor_method,
        config.neighbor_k,
        sample_size=None,
        end_idx=end_idx,
    )

    # Extract edge weights when non-uniform weighting is requested
    sample_edge_weights = None
    sample_volume_weights = None
    edge_weight_mode = getattr(config, "edge_weight_mode", "uniform")

    if edge_weight_mode != "uniform":
        sample_edge_weights = extract_precomputed_edge_weights(
            history,
            start_idx,
            sample_indices,
            neighbor_indices,
            alive,
            mode=edge_weight_mode,
        )

    # Extract force for glueball channel
    force_viscous = history.force_viscous[start_idx - 1 : end_idx - 1]

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
        sample_edge_weights=sample_edge_weights,
        sample_volume_weights=sample_volume_weights,
        operator_weighting=edge_weight_mode,
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
# Edge Weight Extraction
# =============================================================================


def extract_precomputed_edge_weights(
    history: RunHistory,
    start_idx: int,
    sample_indices: Tensor,  # [T, S]
    neighbor_indices: Tensor,  # [T, S, k]
    alive: Tensor,  # [T, N]
    mode: str,
) -> Tensor | None:
    """Extract pre-computed edge weights for sample-neighbor pairs.

    Looks up history.edge_weights[step][mode] and gathers values
    for each (sample, first_neighbor) pair using scatter-gather.

    Returns [T, S] tensor of weights, or None if data unavailable.
    """
    edge_weights = getattr(history, "edge_weights", None)
    if edge_weights is None or not edge_weights:
        return None
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None:
        return None

    T, S = sample_indices.shape
    N = alive.shape[1]
    device = sample_indices.device
    result = torch.ones(T, S, device=device, dtype=torch.float32)
    first_neighbor = neighbor_indices[:, :, 0]  # [T, S]
    found_any = False

    for t in range(T):
        record_idx = start_idx + t
        if record_idx >= len(edge_weights):
            continue
        ew_dict = edge_weights[record_idx]
        if mode not in ew_dict:
            continue
        edges = neighbor_edges[record_idx]
        weights = ew_dict[mode]
        if not torch.is_tensor(edges) or edges.numel() == 0:
            continue
        if not torch.is_tensor(weights) or weights.numel() == 0:
            continue

        found_any = True
        edges_d = edges.to(device)
        w_d = weights.float().to(device)
        # Scatter into dense [N, N] matrix
        w_matrix = torch.zeros(N, N, device=device, dtype=torch.float32)
        w_matrix[edges_d[:, 0], edges_d[:, 1]] = w_d
        # Gather for sample-neighbor pairs
        gathered = w_matrix[sample_indices[t], first_neighbor[t]]  # [S]
        has_w = gathered > 0
        result[t, has_w] = gathered[has_w]

    return result if found_any else None


# =============================================================================
# Geometric Weighting Helper
# =============================================================================


def _compute_operator_weights(agg_data: AggregatedTimeSeries, valid_mask: Tensor) -> Tensor:
    """Compute per-sample operator weights based on edge weight mode.

    Args:
        agg_data: Aggregated time series with weighting config and geometric data.
        valid_mask: Boolean validity mask [T, S].

    Returns:
        Weights tensor [T, S] (float). For "uniform" mode, equivalent to valid_mask.float().
    """
    mode = agg_data.operator_weighting
    weights = valid_mask.float()

    if mode == "uniform":
        return weights

    # Use pre-computed edge weights as direct multiplicative weights
    if agg_data.sample_edge_weights is not None:
        ew = agg_data.sample_edge_weights.to(weights.device, dtype=weights.dtype)
        weights = weights * ew

    return weights


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

    T, _N, _d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states for samples and first neighbors
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]

    # Use first neighbor
    first_neighbor = neighbor_indices[:, :, 0]  # [T, S]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = (
        valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # Identity projection: simple dot product
    op_values = (color_i.conj() * color_j).sum(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Weighted mean over samples per timestep
    weights = _compute_operator_weights(agg_data, valid_mask)
    return (op_values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)


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

    T, _N, _d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = (
        valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # γ₅ projection: alternating sign dot product
    gamma5_diag = gamma_matrices["5"].to(color_i.device)
    op_values = (color_i.conj() * gamma5_diag * color_j).sum(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Weighted mean over samples
    weights = _compute_operator_weights(agg_data, valid_mask)
    return (op_values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)


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

    T, _N, _d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = (
        valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # γ_μ projection using einsum
    gamma_mu = gamma_matrices["mu"].to(color_i.device, dtype=color_i.dtype)
    result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
    op_values = result.mean(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Weighted mean over samples
    weights = _compute_operator_weights(agg_data, valid_mask)
    return (op_values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)


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

    T, _N, _d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = (
        valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # γ₅γ_μ projection
    gamma_5mu = gamma_matrices["5mu"].to(color_i.device, dtype=color_i.dtype)
    result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_5mu, color_j)
    op_values = result.mean(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Weighted mean over samples
    weights = _compute_operator_weights(agg_data, valid_mask)
    return (op_values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)


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

    T, _N, _d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    color_i = color[t_idx, sample_indices]
    first_neighbor = neighbor_indices[:, :, 0]
    color_j = color[t_idx, first_neighbor]

    # Validity masks
    valid_i = (
        valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

    # σ_μν projection
    sigma = gamma_matrices["sigma"].to(color_i.device, dtype=color_i.dtype)
    if sigma.shape[0] == 0:
        return torch.zeros(T, device=device)

    result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
    op_values = result.mean(dim=-1).real

    # Mask invalid
    op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

    # Weighted mean over samples
    weights = _compute_operator_weights(agg_data, valid_mask)
    return (op_values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)


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

    T, _N, d = color.shape
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
    valid_i = (
        valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    valid_j = (
        valid[t_idx, neighbor_indices[:, :, 0]]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 0]]
    )
    valid_k = (
        valid[t_idx, neighbor_indices[:, :, 1]]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 1]]
    )
    valid_mask = valid_i & valid_j & valid_k

    # Mask invalid
    det = torch.where(valid_mask, det, torch.zeros_like(det))

    # Weighted mean over samples
    weights = _compute_operator_weights(agg_data, valid_mask)
    if det.is_complex():
        weights_c = weights.to(det.dtype)
        series = (det * weights_c).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)
    else:
        series = (det * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12)

    return series.real if series.is_complex() else series


def compute_glueball_operators(
    agg_data: AggregatedTimeSeries,
) -> Tensor:
    """Compute glueball channel operators: ||force||².

    When volume weighting is enabled, applies Riemannian volume weights
    to per-walker force-squared values.

    Args:
        agg_data: Aggregated time series data.

    Returns:
        Operator series [T]
    """
    force = agg_data.force_viscous
    alive = agg_data.alive
    T = force.shape[0]
    N = force.shape[1]
    device = force.device

    # Force squared norm: [T, N]
    force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)

    # Check if volume weighting should be applied
    use_volume = (
        agg_data.operator_weighting != "uniform" and agg_data.sample_volume_weights is not None
    )

    # Clamp alive mask to match T
    alive_T = (
        alive[:T]
        if alive.shape[0] >= T
        else F.pad(
            alive,
            (0, 0, 0, T - alive.shape[0]),
            value=True,
        )
    )

    if use_volume:
        vol_w = agg_data.sample_volume_weights  # [T, S]
        sample_idx = agg_data.sample_indices  # [T, S]

        # Scatter sample volume weights into per-walker tensor [T, N]
        vol_per_walker = torch.ones(T, N, device=device, dtype=torch.float32)
        if vol_w is not None:
            T_vol = min(T, vol_w.shape[0], sample_idx.shape[0])
            # Use scatter to map sample weights back to walker positions
            idx_clamped = sample_idx[:T_vol].long().clamp(max=N - 1)
            vol_per_walker[:T_vol].scatter_(1, idx_clamped, vol_w[:T_vol].to(device).float())
        vol_per_walker = vol_per_walker.clamp(min=0.0)

        # Weighted mean over alive walkers per timestep
        weights = vol_per_walker * alive_T.float()  # [T, N]
        weighted_sum = (force_sq * weights).sum(dim=1)  # [T]
        w_sum = weights.sum(dim=1).clamp(min=1e-12)  # [T]
        series = weighted_sum / w_sum
    else:
        # Simple mean over alive walkers per timestep
        alive_f = alive_T.float()
        alive_count = alive_f.sum(dim=1).clamp(min=1.0)  # [T]
        series = (force_sq * alive_f).sum(dim=1) / alive_count  # [T]

    return series


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
    T, N, _d = color.shape
    device = color.device

    # Use first neighbor for each walker
    first_neighbor = neighbor_indices[:, :, 0]  # [T, N]

    # Gather color states
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, N)
    color_i = color  # [T, N, d] (walkers themselves)
    color_j = color[t_idx, first_neighbor]  # [T, N, d] (their neighbors)

    # Validity masks
    valid_i = valid & alive
    valid_j = (
        valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
    )
    self_neighbor = first_neighbor == torch.arange(N, device=device).unsqueeze(0)
    valid_mask = valid_i & valid_j & (~self_neighbor)

    # Apply channel-specific projection
    op_values = gamma_projection_func(color_i, color_j)  # [T, N]

    # Mask invalid
    return torch.where(valid_mask, op_values, torch.zeros_like(op_values))


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
        msg = "full_neighbor_indices required for per-walker computation"
        raise ValueError(msg)

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
        msg = "full_neighbor_indices required for per-walker computation"
        raise ValueError(msg)

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
        msg = "full_neighbor_indices required for per-walker computation"
        raise ValueError(msg)

    gamma_mu = gamma_matrices["mu"]

    def vector_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        result = torch.einsum(
            "...i,mij,...j->...m",
            color_i.conj(),
            gamma_mu.to(color_i.device, dtype=color_i.dtype),
            color_j,
        )
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
        msg = "full_neighbor_indices required for per-walker computation"
        raise ValueError(msg)

    gamma_5mu = gamma_matrices["5mu"]

    def axial_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        result = torch.einsum(
            "...i,mij,...j->...m",
            color_i.conj(),
            gamma_5mu.to(color_i.device, dtype=color_i.dtype),
            color_j,
        )
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
        msg = "full_neighbor_indices required for per-walker computation"
        raise ValueError(msg)

    sigma = gamma_matrices["sigma"]

    if sigma.shape[0] == 0:
        T, N = agg_data.color.shape[:2]
        return torch.zeros(T, N, device=agg_data.device)

    def tensor_projection(color_i: Tensor, color_j: Tensor) -> Tensor:
        result = torch.einsum(
            "...i,pij,...j->...p",
            color_i.conj(),
            sigma.to(color_i.device, dtype=color_i.dtype),
            color_j,
        )
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
        msg = "full_neighbor_indices required for per-walker computation"
        raise ValueError(msg)

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
    valid_j = (
        valid[t_idx, neighbor_indices[:, :, 0]]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 0]]
    )
    valid_k = (
        valid[t_idx, neighbor_indices[:, :, 1]]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 1]]
    )
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
    return torch.where(alive, force_sq, torch.zeros_like(force_sq))


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
        channels = [
            "scalar",
            "pseudoscalar",
            "vector",
            "axial_vector",
            "tensor",
            "nucleon",
            "glueball",
        ]

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
