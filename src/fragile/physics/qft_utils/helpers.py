"""Small self-contained helper utilities for QFT analysis.

Ported from fragile.fractalai.qft.radial_channels and
fragile.fractalai.qft.baryon_triplet_channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory


# =============================================================================
# Safe Gather Helpers (from baryon_triplet_channels)
# =============================================================================


def safe_gather_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] and return gathered values + in-range mask."""
    n = values.shape[1]
    in_range = (indices >= 0) & (indices < n)
    idx_safe = indices.clamp(min=0, max=max(n - 1, 0))
    gathered = torch.gather(values, dim=1, index=idx_safe)
    return gathered, in_range


def safe_gather_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
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


def safe_gather_pairs_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] for indices [T,N,P] using preparation helpers."""
    if values.ndim != 2 or indices.ndim != 3:
        msg = (
            f"_safe_gather_pairs_2d expects values [T,N] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
        raise ValueError(msg)
    t, n, p = indices.shape
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = safe_gather_2d(values, idx_flat)
    return gathered_flat.reshape(t, n, p), in_range_flat.reshape(t, n, p)


def safe_gather_pairs_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx, :] for indices [T,N,P] using preparation helpers."""
    if values.ndim != 3 or indices.ndim != 3:
        msg = (
            f"_safe_gather_pairs_3d expects values [T,N,C] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
        raise ValueError(msg)
    t, n, p = indices.shape
    c = values.shape[-1]
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = safe_gather_3d(values, idx_flat)
    return (
        gathered_flat.reshape(t, n, p, c),
        in_range_flat.reshape(t, n, p),
    )


# =============================================================================
# Dimension / Index Resolution (from baryon_triplet_channels)
# =============================================================================


def resolve_3d_dims(
    total_dims: int, dims: tuple[int, int, int] | None, name: str
) -> tuple[int, int, int]:
    """Resolve and validate exactly 3 component indices."""
    if dims is None:
        if total_dims < 3:
            msg = f"{name} requires at least 3 dimensions, got d={total_dims}."
            raise ValueError(msg)
        return 0, 1, 2
    if len(dims) != 3:
        msg = f"{name} must contain exactly 3 indices."
        raise ValueError(msg)
    dims_tuple = tuple(int(d) for d in dims)
    if len(set(dims_tuple)) != 3:
        msg = f"{name} indices must be unique, got {dims_tuple}."
        raise ValueError(msg)
    invalid = [d for d in dims_tuple if d < 0 or d >= total_dims]
    if invalid:
        msg = f"{name} has invalid indices {invalid}; valid range is [0, {total_dims - 1}]."
        raise ValueError(msg)
    return dims_tuple


def resolve_frame_indices(
    history: RunHistory,
    warmup_fraction: float,
    end_fraction: float,
) -> list[int]:
    """Resolve frame indices [start_idx, end_idx) used by correlator analysis."""
    if history.n_recorded < 2:
        return []

    start_idx = max(1, int(history.n_recorded * float(warmup_fraction)))
    end_idx = max(start_idx + 1, int(history.n_recorded * float(end_fraction)))
    end_idx = min(end_idx, history.n_recorded)
    steps = getattr(history, "recorded_steps", [])
    if len(steps) == history.n_recorded and len(steps) >= 3:
        if steps[-1] - steps[-2] != getattr(history, "record_every", steps[1] - steps[0]):
            end_idx = min(end_idx, history.n_recorded - 1)

    if end_idx <= start_idx:
        return []
    return list(range(start_idx, end_idx))


def recorded_time_step(history) -> float:
    """Recorded spacing in integrated kinetic time, including legacy histories."""
    params = getattr(history, "params", None) or {}
    dt = float(history.delta_t)
    if params.get("history_conventions", {}).get("delta_t_unit") != "iteration":
        dt *= int(params.get("kinetic", {}).get("n_kinetic_steps", 1))
    return dt * history.record_every
