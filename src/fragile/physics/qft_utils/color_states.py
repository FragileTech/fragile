"""Color state computation from RunHistory data.

Ported from fragile.fractalai.qft.aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.physics.fractal_gas.history import RunHistory


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


def estimate_ell0(
    history: RunHistory,
) -> float:
    """Estimate ell0 from median Euclidean companion distance at mid-point.

    Uses Euclidean distances between companion pairs, matching the fractalai
    implementation.  Geodesic (graph-weighted) distances are not suitable here
    because ``ell0`` enters the colour-state phase formula
    ``phase = mass * v * ell0 / h_eff`` and must represent a physical length
    scale, not a graph-metric quantity.

    Args:
        history: RunHistory object.

    Returns:
        Estimated ell0 value.
    """
    mid_idx = history.n_recorded // 2
    if mid_idx == 0:
        return 1.0

    comp_idx = history.companions_distance[mid_idx - 1]
    return _euclidean_companion_distance(history, mid_idx, comp_idx)


def _euclidean_companion_distance(
    history: RunHistory,
    mid_idx: int,
    comp_idx: Tensor,
) -> float:
    """Compute median Euclidean distance to companions at *mid_idx*."""
    x_pre = history.x_before_clone[mid_idx]
    diff = x_pre - x_pre[comp_idx]
    dist = torch.linalg.vector_norm(diff, dim=-1)
    if dist.numel() > 0:
        return float(dist.median().item())
    return 1.0
