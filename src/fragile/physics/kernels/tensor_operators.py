"""Companion-pair spin-2 traceless tensor operators.

Implements a traceless symmetric spin-2 operator from companion pairs:

    O_i^{ab}(t) = Re(c_i^dag c_j) * Q^{ab}(dx_ij)

where Q^{ab} is represented in a 5-component real basis (d=3):
    - q_xy
    - q_xz
    - q_yz
    - q_xx_minus_yy / sqrt(2)
    - q_2zz_minus_xx_minus_yy / sqrt(6)

This module computes *operators only* -- no correlator computation or
RunHistory access.  All frame extraction and color-state preparation is
handled by :mod:`.preparation`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from fragile.fractalai.qft.radial_channels import _apply_pbc_diff_torch, _slice_bounds

from .config import TensorOperatorConfig
from .meson_operators import (
    PAIR_SELECTION_MODES,
    _safe_gather_pairs_2d,
    _safe_gather_pairs_3d,
    build_companion_pair_indices,
)
from .preparation import PreparedChannelData, _safe_gather_2d, _safe_gather_3d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENSOR_COMPONENT_LABELS = (
    "q_xy",
    "q_xz",
    "q_yz",
    "q_xx_minus_yy",
    "q_2zz_minus_xx_minus_yy",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _traceless_tensor_components(dx: Tensor) -> Tensor:
    """Build the 5-component traceless symmetric tensor basis for dx[...,3]."""
    if dx.ndim < 1 or dx.shape[-1] != 3:
        raise ValueError(f"Expected dx[...,3], got {tuple(dx.shape)}.")

    x = dx[..., 0]
    y = dx[..., 1]
    z = dx[..., 2]
    inv_sqrt2 = float(1.0 / math.sqrt(2.0))
    inv_sqrt6 = float(1.0 / math.sqrt(6.0))

    return torch.stack(
        (
            x * y,
            x * z,
            y * z,
            (x * x - y * y) * inv_sqrt2,
            (2.0 * z * z - x * x - y * y) * inv_sqrt6,
        ),
        dim=-1,
    )


def _compute_local_tensor_components(
    *,
    color: Tensor,
    color_valid: Tensor,
    positions: Tensor,
    alive: Tensor,
    pair_indices: Tensor,
    structural_valid: Tensor,
    bounds: object | None,
    pbc: bool,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Compute local 5-component tensor operator per walker and frame.

    Returns:
        local_components: [T, N, 5] local tensor operator per walker.
        valid_walker: [T, N] bool mask of walkers with valid pairs.
        component_counts_per_frame: [T] int64 count of valid walkers per frame.
        n_valid_source_pairs: total valid pair count.
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T,N,3], got {tuple(color.shape)}.")
    if positions.shape != color.shape:
        raise ValueError(
            f"positions must have shape [T,N,3] aligned with color, got {tuple(positions.shape)}."
        )
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T,N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T,N], got {tuple(alive.shape)}.")
    if pair_indices.shape[:2] != color.shape[:2]:
        raise ValueError(
            "pair_indices must have shape [T,N,P] aligned with color, got "
            f"{tuple(pair_indices.shape)}."
        )
    if structural_valid.shape != pair_indices.shape:
        raise ValueError(
            "structural_valid must match pair_indices shape, got "
            f"{tuple(structural_valid.shape)} vs {tuple(pair_indices.shape)}."
        )

    color_j, in_range = _safe_gather_pairs_3d(color, pair_indices)
    alive_j, _ = _safe_gather_pairs_2d(alive, pair_indices)
    valid_j, _ = _safe_gather_pairs_2d(color_valid, pair_indices)
    pos_j, _ = _safe_gather_pairs_3d(positions, pair_indices)

    color_i = color.unsqueeze(2).expand_as(color_j)
    pos_i = positions.unsqueeze(2).expand_as(pos_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1).real.float()
    dx = (pos_j - pos_i).float()
    if pbc and bounds is not None:
        dx = _apply_pbc_diff_torch(dx, bounds)

    finite_inner = torch.isfinite(inner)
    finite_dx = torch.isfinite(dx).all(dim=-1)
    valid = (
        structural_valid
        & in_range
        & alive.unsqueeze(-1)
        & alive_j
        & color_valid.unsqueeze(-1)
        & valid_j
        & finite_inner
        & finite_dx
    )
    if eps > 0:
        valid = valid & (inner.abs() > float(eps))

    components_pair = inner.unsqueeze(-1) * _traceless_tensor_components(dx)  # [T,N,P,5]
    components_pair = torch.where(
        valid.unsqueeze(-1),
        components_pair,
        torch.zeros_like(components_pair),
    )

    pair_weights = valid.to(dtype=torch.float32)
    pair_count_walker = pair_weights.sum(dim=2)  # [T,N]
    valid_walker = pair_count_walker > 0

    local_components = torch.zeros(
        (*pair_count_walker.shape, 5),
        dtype=torch.float32,
        device=color.device,
    )
    if torch.any(valid_walker):
        sums = components_pair.sum(dim=2)  # [T,N,5]
        local_components[valid_walker] = sums[valid_walker] / pair_count_walker[
            valid_walker
        ].unsqueeze(-1).clamp(min=1.0)

    component_counts_per_frame = valid_walker.sum(dim=1).to(torch.int64)
    n_valid_source_pairs = int(valid.sum().item())
    return local_components, valid_walker, component_counts_per_frame, n_valid_source_pairs


def _resolve_positive_length(
    *,
    positions_axis: Tensor,
    valid_walker: Tensor,
    box_length: float | None,
) -> float:
    """Resolve a positive projection length scale for Fourier modes."""
    if box_length is not None and float(box_length) > 0:
        return float(box_length)

    alive_pos = positions_axis[valid_walker]
    if alive_pos.numel() == 0:
        span = float((positions_axis.max() - positions_axis.min()).abs().item())
    else:
        span = float((alive_pos.max() - alive_pos.min()).abs().item())
    return max(span, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_tensor_operators(
    data: PreparedChannelData,
    config: TensorOperatorConfig,
) -> dict[str, Tensor]:
    """Compute spin-2 tensor operator time-series from prepared channel data.

    Returns a dict:
        ``"tensor"`` -- per-frame averaged tensor components [T, 5].

    When momentum projection is configured (``data.positions_axis`` is not
    None), additional entries:
        ``"tensor_momentum_cos_{n}"`` -- cosine-projected operator [T, 5] for mode *n*.
        ``"tensor_momentum_sin_{n}"`` -- sine-projected operator [T, 5] for mode *n*.
    """
    color = data.color
    t_total = int(color.shape[0])
    device = data.device

    if t_total == 0:
        result: dict[str, Tensor] = {
            "tensor": torch.zeros(0, 5, dtype=torch.float32, device=device),
        }
        return result

    if data.positions is None:
        msg = "positions is required for tensor operators."
        raise ValueError(msg)

    # 1. Build companion pair indices
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=str(config.pair_selection),
    )

    # 2. Compute local tensor components per walker
    local_components, valid_walker, component_counts_per_frame, _ = (
        _compute_local_tensor_components(
            color=color,
            color_valid=data.color_valid,
            positions=data.positions,
            alive=data.alive,
            pair_indices=pair_indices,
            structural_valid=structural_valid,
            bounds=data.bounds,
            pbc=data.pbc,
            eps=data.eps,
        )
    )

    # 3. Average per frame -> component_series [T, 5]
    component_series = torch.zeros(
        t_total, 5, dtype=torch.float32, device=device,
    )
    valid_t = component_counts_per_frame > 0
    if torch.any(valid_t):
        sums = (local_components * valid_walker.unsqueeze(-1).to(local_components.dtype)).sum(dim=1)
        component_series[valid_t] = sums[valid_t] / component_counts_per_frame[valid_t].to(
            local_components.dtype
        ).unsqueeze(-1).clamp(min=1.0)

    result: dict[str, Tensor] = {"tensor": component_series}

    # 4. Momentum projection (operator series only, no correlators)
    if data.positions_axis is not None:
        positions_axis = data.positions_axis
        if positions_axis.shape != color.shape[:2]:
            msg = (
                "positions_axis must have shape [T, N], got "
                f"{tuple(positions_axis.shape)} vs {tuple(color.shape[:2])}."
            )
            raise ValueError(msg)

        box_length = _resolve_positive_length(
            positions_axis=positions_axis,
            valid_walker=valid_walker,
            box_length=data.projection_length,
        )

        n_modes = max(0, int(config.momentum_mode_max)) + 1
        n_comp = 5
        modes = torch.arange(n_modes, device=device, dtype=torch.float32)
        k_values = (2.0 * torch.pi / float(box_length)) * modes

        # phase_arg: [n_modes, T, N]
        phase_arg = k_values[:, None, None] * positions_axis[None, :, :].float()
        cos_phase = torch.cos(phase_arg)
        sin_phase = torch.sin(phase_arg)

        weights = valid_walker.to(dtype=torch.float32)
        counts_t = weights.sum(dim=1)
        valid_frames = counts_t > 0

        op_cos = torch.zeros((n_modes, n_comp, t_total), dtype=torch.float32, device=device)
        op_sin = torch.zeros((n_modes, n_comp, t_total), dtype=torch.float32, device=device)
        if torch.any(valid_frames):
            # weighted_local: [T, N, 5]
            weighted_local = local_components * weights.unsqueeze(-1)
            # einsum: [T,N,C] x [M,T,N] -> [M,C,T]
            cos_num = torch.einsum("tnc,mtn->mct", weighted_local, cos_phase)
            sin_num = torch.einsum("tnc,mtn->mct", weighted_local, sin_phase)
            denom = counts_t[valid_frames].to(dtype=torch.float32).clamp(min=1.0)
            op_cos[:, :, valid_frames] = cos_num[:, :, valid_frames] / denom.unsqueeze(0).unsqueeze(
                0
            )
            op_sin[:, :, valid_frames] = sin_num[:, :, valid_frames] / denom.unsqueeze(0).unsqueeze(
                0
            )

        for m in range(n_modes):
            # Each momentum mode entry is [T, 5] (transposed from [5, T])
            result[f"tensor_momentum_cos_{m}"] = op_cos[m].T
            result[f"tensor_momentum_sin_{m}"] = op_sin[m].T

    return result
