"""Shared data preparation for companion-channel operator computation.

Converts a ``RunHistory`` into a ``PreparedChannelData`` dataclass that all
operator modules consume, eliminating duplicated frame-extraction and
color-state logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0

from .config import ChannelConfigBase


# ---------------------------------------------------------------------------
# Helper utilities (originally in baryon_triplet_channels)
# ---------------------------------------------------------------------------


def _resolve_frame_indices(
    history: RunHistory,
    warmup_fraction: float,
    end_fraction: float,
) -> list[int]:
    """Resolve frame indices ``[start_idx, end_idx)`` for correlator analysis."""
    if history.n_recorded < 2:
        return []

    start_idx = max(1, int(history.n_recorded * float(warmup_fraction)))
    end_idx = max(start_idx + 1, int(history.n_recorded * float(end_fraction)))
    end_idx = min(end_idx, history.n_recorded)

    if end_idx <= start_idx:
        return []
    return list(range(start_idx, end_idx))


def _resolve_3d_dims(
    total_dims: int, dims: tuple[int, int, int] | None, name: str
) -> tuple[int, int, int]:
    """Resolve and validate exactly 3 component indices."""
    if dims is None:
        if total_dims < 3:
            raise ValueError(f"{name} requires at least 3 dimensions, got d={total_dims}.")
        return 0, 1, 2
    if len(dims) != 3:
        raise ValueError(f"{name} must contain exactly 3 indices.")
    dims_tuple = tuple(int(d) for d in dims)
    if len(set(dims_tuple)) != 3:
        raise ValueError(f"{name} indices must be unique, got {dims_tuple}.")
    invalid = [d for d in dims_tuple if d < 0 or d >= total_dims]
    if invalid:
        raise ValueError(
            f"{name} has invalid indices {invalid}; valid range is [0, {total_dims - 1}]."
        )
    return dims_tuple


# ---------------------------------------------------------------------------
# PreparedChannelData
# ---------------------------------------------------------------------------


@dataclass
class PreparedChannelData:
    """Pre-extracted tensors shared by all operator modules.

    Created once by :func:`prepare_channel_data` and passed into every
    ``compute_*_operators`` function so that frame selection, color-state
    computation, and companion extraction are never duplicated.
    """

    color: Tensor  # [T, N, 3] complex color states
    color_valid: Tensor  # [T, N] bool
    companions_distance: Tensor  # [T, N] long
    companions_clone: Tensor  # [T, N] long
    scores: Tensor | None  # [T, N] float (for score-directed modes)
    positions: Tensor | None  # [T, N, 3] float (for vector/tensor)
    positions_axis: Tensor | None  # [T, N] float (for momentum projection)
    projection_length: float | None  # Box size along momentum axis
    frame_indices: list[int]
    device: torch.device
    eps: float

    # Multiscale fields (None when n_scales=1)
    scales: Tensor | None = None  # [S] float
    pairwise_distances: Tensor | None = None  # [T, N, N] float geodesic
    kernels: Tensor | None = None  # [T, S, N, N] float smeared
    companion_d_ij: Tensor | None = None  # [T, N] float d(i, companion_j)
    companion_d_ik: Tensor | None = None  # [T, N] float d(i, companion_k)
    companion_d_jk: Tensor | None = None  # [T, N] float d(j, k)


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------


def prepare_channel_data(
    history: RunHistory,
    config: ChannelConfigBase,
    *,
    need_positions: bool = False,
    need_scores: bool = False,
    need_momentum_axis: bool = False,
    momentum_axis: int = 0,
) -> PreparedChannelData:
    """Convert *history* + *config* into a :class:`PreparedChannelData`.

    This is the single point of entry for all companion-channel operators.
    Downstream modules never touch ``RunHistory`` directly.
    """
    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
    )

    device = history.v_before_clone.device
    eps = float(max(config.eps, 0.0))

    if not frame_indices:
        empty_3d = torch.zeros(0, 0, 3, device=device)
        empty_2d_bool = torch.zeros(0, 0, dtype=torch.bool, device=device)
        empty_2d_long = torch.zeros(0, 0, dtype=torch.long, device=device)
        return PreparedChannelData(
            color=empty_3d.to(torch.complex64),
            color_valid=empty_2d_bool,
            companions_distance=empty_2d_long,
            companions_clone=empty_2d_long.clone(),
            scores=None,
            positions=None,
            positions_axis=None,
            projection_length=None,
            frame_indices=[],
            device=device,
            eps=eps,
        )

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1

    # Physics parameters
    h_eff = float(max(config.h_eff, 1e-8))
    mass = float(max(config.mass, 1e-8))
    ell0 = float(config.ell0) if config.ell0 is not None else float(estimate_ell0(history))
    if ell0 <= 0:
        msg = "ell0 must be positive."
        raise ValueError(msg)

    # Color states
    color, color_valid = compute_color_states_batch(
        history=history,
        start_idx=start_idx,
        h_eff=h_eff,
        mass=mass,
        ell0=ell0,
        end_idx=end_idx,
    )
    dims = _resolve_3d_dims(color.shape[-1], config.color_dims, "color_dims")
    color = color[:, :, list(dims)]
    device = color.device

    # Companion arrays (offset by 1 relative to frame indices)
    companions_distance = torch.as_tensor(
        history.companions_distance[start_idx - 1 : end_idx - 1],
        dtype=torch.long,
        device=device,
    )
    companions_clone = torch.as_tensor(
        history.companions_clone[start_idx - 1 : end_idx - 1],
        dtype=torch.long,
        device=device,
    )

    # Optional: scores
    scores: Tensor | None = None
    if need_scores:
        scores = torch.as_tensor(
            history.cloning_scores[start_idx - 1 : end_idx - 1],
            dtype=torch.float32,
            device=device,
        )

    # Optional: positions (3-component)
    positions: Tensor | None = None
    if need_positions:
        pos_dims_cfg = getattr(config, "position_dims", None)
        pos_dims = _resolve_3d_dims(history.d, pos_dims_cfg, "position_dims")
        positions = history.x_before_clone[start_idx:end_idx, :, list(pos_dims)].to(
            device=device, dtype=torch.float32
        )

    # Optional: scalar position axis for momentum projection
    positions_axis: Tensor | None = None
    projection_length: float | None = None
    if need_momentum_axis:
        if momentum_axis < 0 or momentum_axis >= int(history.d):
            raise ValueError(
                f"momentum_axis={momentum_axis} out of range for history.d={history.d}. "
                f"Expected 0..{history.d - 1}."
            )
        positions_axis = history.x_before_clone[start_idx:end_idx, :, momentum_axis].to(
            device=device, dtype=torch.float32
        )
        # Allow explicit override from config
        cfg_length = getattr(config, "projection_length", None)
        if cfg_length is not None:
            projection_length = float(cfg_length)

    return PreparedChannelData(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        scores=scores,
        positions=positions,
        positions_axis=positions_axis,
        projection_length=projection_length,
        frame_indices=frame_indices,
        device=device,
        eps=eps,
    )
