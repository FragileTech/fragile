"""Euclidean-time binning utilities ported from fractalai.qft.aggregation."""

from __future__ import annotations

import torch
from torch import Tensor


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
