"""Euclidean-time binning utilities ported from fractalai.qft.aggregation."""

from __future__ import annotations

import torch
from torch import Tensor


def bin_by_euclidean_time(
    operators: Tensor,  # [T, N]
    n_bins: int = 0,  # kept for API compat, unused
) -> tuple[Tensor, Tensor]:
    """Average operators over walkers per MC timestep.

    All walkers are alive and MC time is the time index, so this reduces to
    a simple mean over the walker dimension at each timestep.

    Args:
        operators: Operator values [T, N].
        n_bins: Unused, kept for API compatibility.

    Returns:
        time_coords: Integer time indices ``arange(T)`` as float tensor.
        operator_series: Mean operator per timestep [T].
    """
    T = operators.shape[0]
    time_coords = torch.arange(T, device=operators.device, dtype=operators.dtype)
    operator_series = operators.mean(dim=1)
    return time_coords, operator_series
