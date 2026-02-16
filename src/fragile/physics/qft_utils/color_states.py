"""Color state computation from RunHistory data.

Ported from fragile.fractalai.qft.aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory


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
