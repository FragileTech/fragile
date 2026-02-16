"""Scale-gating utilities for multiscale operator computation.

Provides functions to expand per-walker validity masks to per-scale masks
based on geodesic distance thresholds, and multiscale-aware per-frame
averaging that produces ``[S, T]`` or ``[S, T, C]`` output tensors.

All operator modules import from this module to add multiscale branching.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Scale gating for pair-based operators
# ---------------------------------------------------------------------------


def gate_pair_validity_by_scale(
    pair_valid: Tensor,
    pair_indices: Tensor,
    distances: Tensor,
    scales: Tensor,
) -> Tensor:
    """Expand pair validity to ``[T, S, N, P]`` by scale gating.

    For each scale s: valid if ``base_valid AND d(anchor_i, pair_j) <= scale_s``.

    Args:
        pair_valid: ``[T, N, P]`` bool base validity.
        pair_indices: ``[T, N, P]`` long companion indices.
        distances: ``[T, N, N]`` geodesic distance matrix.
        scales: ``[S]`` scale thresholds.

    Returns:
        ``[T, S, N, P]`` bool scale-gated validity.
    """
    T, N, P = pair_valid.shape
    scales.shape[0]
    device = pair_valid.device

    # Gather d(i, pair_j) from distance matrix for each pair index
    # anchor i is just the walker index
    t_idx = torch.arange(T, device=device).view(T, 1, 1).expand(T, N, P)
    i_idx = torch.arange(N, device=device).view(1, N, 1).expand(T, N, P)
    j_idx = pair_indices.clamp(0, N - 1)

    d_ij = distances[t_idx, i_idx, j_idx]  # [T, N, P]

    # Scale gating: d_ij <= scale_s for each scale
    # scales: [S] -> [1, S, 1, 1]
    scale_view = scales.view(1, -1, 1, 1)
    d_ij_view = d_ij.unsqueeze(1)  # [T, 1, N, P]
    within_scale = d_ij_view <= scale_view  # [T, S, N, P]

    # Combine with base validity
    base_view = pair_valid.unsqueeze(1)  # [T, 1, N, P]
    return base_view & within_scale


# ---------------------------------------------------------------------------
# Scale gating for triplet-based operators
# ---------------------------------------------------------------------------


def gate_triplet_validity_by_scale(
    triplet_valid: Tensor,
    d_ij: Tensor,
    d_ik: Tensor,
    d_jk: Tensor,
    scales: Tensor,
) -> Tensor:
    """Expand triplet validity to ``[T, S, N]`` by scale gating.

    For each scale s: valid if ``base_valid AND max(d_ij, d_ik, d_jk) <= scale_s``.

    Args:
        triplet_valid: ``[T, N]`` bool base validity.
        d_ij: ``[T, N]`` distance between anchor and companion_j.
        d_ik: ``[T, N]`` distance between anchor and companion_k.
        d_jk: ``[T, N]`` distance between companion_j and companion_k.
        scales: ``[S]`` scale thresholds.

    Returns:
        ``[T, S, N]`` bool scale-gated validity.
    """
    # max of all three pairwise distances
    d_max = torch.max(torch.max(d_ij, d_ik), d_jk)  # [T, N]

    # Scale gating
    scale_view = scales.view(1, -1, 1)  # [1, S, 1]
    d_max_view = d_max.unsqueeze(1)  # [T, 1, N]
    within_scale = d_max_view <= scale_view  # [T, S, N]

    base_view = triplet_valid.unsqueeze(1)  # [T, 1, N]
    return base_view & within_scale


# ---------------------------------------------------------------------------
# Multiscale per-frame averaging
# ---------------------------------------------------------------------------


def per_frame_series_multiscale(
    values: Tensor,
    valid: Tensor,
) -> Tensor:
    """Masked mean per frame per scale.

    Args:
        values: ``[T, N, P]`` or ``[T, N]`` observable values.
        valid: ``[T, S, N, P]`` or ``[T, S, N]`` scale-gated validity mask.

    Returns:
        ``[S, T]`` per-scale per-frame averaged series.
    """
    if values.ndim == 2:
        # Triplet case: values [T, N], valid [T, S, N]
        T, N = values.shape
        S = valid.shape[1]
        values_view = values.unsqueeze(1).expand(T, S, N)  # [T, S, N]
    elif values.ndim == 3:
        # Pair case: values [T, N, P], valid [T, S, N, P]
        T, N, P = values.shape
        S = valid.shape[1]
        values_view = values.unsqueeze(1).expand(T, S, N, P)  # [T, S, N, P]
    else:
        raise ValueError(f"values must be 2D [T,N] or 3D [T,N,P], got {values.ndim}D.")

    weights = valid.to(values.dtype)
    # Sum over spatial dims (everything except T and S)
    spatial_dims = tuple(range(2, weights.ndim))
    counts = valid.sum(dim=spatial_dims).to(torch.float32)  # [T, S]
    weighted_sums = (values_view * weights).sum(dim=spatial_dims)  # [T, S]

    series = torch.zeros(T, S, dtype=torch.float32, device=values.device)
    valid_t = counts > 0
    if torch.any(valid_t):
        series[valid_t] = (weighted_sums[valid_t] / counts[valid_t].to(values.dtype)).float()

    # Transpose to [S, T] (scale-major)
    return series.T


def per_frame_vector_series_multiscale(
    values: Tensor,
    valid: Tensor,
) -> Tensor:
    """Masked mean per frame per scale for multi-component observables.

    Args:
        values: ``[T, N, P, C]`` multi-component observable values.
        valid: ``[T, S, N, P]`` scale-gated validity mask.

    Returns:
        ``[S, T, C]`` per-scale per-frame averaged series.
    """
    if values.ndim != 4:
        raise ValueError(f"values must have shape [T,N,P,C], got {tuple(values.shape)}.")
    if valid.ndim != 4:
        raise ValueError(f"valid must have shape [T,S,N,P], got {tuple(valid.shape)}.")

    T, N, P, C = values.shape
    S = valid.shape[1]

    values_view = values.unsqueeze(1).expand(T, S, N, P, C)  # [T, S, N, P, C]
    weights = valid.to(values.dtype).unsqueeze(-1)  # [T, S, N, P, 1]
    counts = valid.sum(dim=(2, 3)).to(torch.float32)  # [T, S]

    weighted_sums = (values_view * weights).sum(dim=(2, 3))  # [T, S, C]

    series = torch.zeros(T, S, C, dtype=torch.float32, device=values.device)
    valid_t = counts > 0  # [T, S]
    if torch.any(valid_t):
        denom = counts[valid_t].to(values.dtype).unsqueeze(-1)  # [K, 1]
        series[valid_t] = (weighted_sums[valid_t] / denom).float()

    # Transpose to [S, T, C] (scale-major)
    return series.permute(1, 0, 2)
