"""Small self-contained helper utilities for QFT analysis.

Ported from fragile.fractalai.qft.radial_channels and
fragile.fractalai.qft.baryon_triplet_channels.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory


# =============================================================================
# PBC Helpers (from radial_channels)
# =============================================================================


def _apply_pbc_diff_torch(diff: Tensor, bounds: Any | None) -> Tensor:
    if bounds is None:
        return diff
    high = bounds.high.to(diff)
    low = bounds.low.to(diff)
    span = high - low
    return diff - span * torch.round(diff / span)


def _slice_bounds(bounds: Any | None, keep_dims: list[int]) -> Any | None:
    if bounds is None:
        return None
    if not hasattr(bounds, "low") or not hasattr(bounds, "high"):
        return bounds
    low = bounds.low[keep_dims]
    high = bounds.high[keep_dims]
    from fragile.fractalai.bounds import TorchBounds

    return TorchBounds(low=low, high=high, shape=low.shape)


# =============================================================================
# Safe Gather Helpers (from baryon_triplet_channels)
# =============================================================================


def _safe_gather_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] and return gathered values + in-range mask."""
    n = values.shape[1]
    in_range = (indices >= 0) & (indices < n)
    idx_safe = indices.clamp(min=0, max=max(n - 1, 0))
    gathered = torch.gather(values, dim=1, index=idx_safe)
    return gathered, in_range


def _safe_gather_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx, :] and return gathered values + in-range mask."""
    n = values.shape[1]
    in_range = (indices >= 0) & (indices < n)
    idx_safe = indices.clamp(min=0, max=max(n - 1, 0))
    gathered = torch.gather(
        values,
        dim=1,
        index=idx_safe.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
    )
    return gathered, in_range


# =============================================================================
# Dimension / Index Resolution (from baryon_triplet_channels)
# =============================================================================


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


def _resolve_mc_time_index(history: RunHistory, mc_time_index: int) -> int:
    """Resolve mc_time_index as recorded index or recorded step."""
    if history.n_recorded < 2:
        msg = "Need at least 2 recorded timesteps."
        raise ValueError(msg)
    raw = int(mc_time_index)
    if raw in history.recorded_steps:
        resolved = int(history.get_step_index(raw))
    else:
        resolved = raw
    if resolved < 1 or resolved >= history.n_recorded:
        raise ValueError(
            f"mc_time_index {resolved} out of bounds (valid recorded index "
            f"1..{history.n_recorded - 1} or a recorded step value)."
        )
    return resolved


def _resolve_frame_indices(
    history: RunHistory,
    warmup_fraction: float,
    end_fraction: float,
    mc_time_index: int | None,
) -> list[int]:
    """Resolve frame indices [start_idx, end_idx) used by correlator analysis."""
    if history.n_recorded < 2:
        return []

    start_idx = max(1, int(history.n_recorded * float(warmup_fraction)))
    end_idx = max(start_idx + 1, int(history.n_recorded * float(end_fraction)))
    end_idx = min(end_idx, history.n_recorded)

    if mc_time_index is not None:
        start_idx = _resolve_mc_time_index(history, int(mc_time_index))
        end_idx = history.n_recorded

    if end_idx <= start_idx:
        return []
    return list(range(start_idx, end_idx))
