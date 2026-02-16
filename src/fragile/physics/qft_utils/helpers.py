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
        raise ValueError(
            f"_safe_gather_pairs_2d expects values [T,N] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t, n, p = indices.shape
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = safe_gather_2d(values, idx_flat)
    return gathered_flat.reshape(t, n, p), in_range_flat.reshape(t, n, p)


def safe_gather_pairs_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx, :] for indices [T,N,P] using preparation helpers."""
    if values.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            f"_safe_gather_pairs_3d expects values [T,N,C] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
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

    if end_idx <= start_idx:
        return []
    return list(range(start_idx, end_idx))
