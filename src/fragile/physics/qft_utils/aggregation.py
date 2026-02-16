"""Time series aggregation from RunHistory data.

Ported from fragile.fractalai.qft.aggregation (selected items).

Main workflow:
    RunHistory -> color states -> neighbor topology -> AggregatedTimeSeries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor

from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0
from fragile.physics.qft_utils.neighbors import (
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
