"""Meson (scalar / pseudoscalar) operator construction from companion pairs.

Computes per-frame meson operator time series from color-singlet pair
inner products:

    z_ij(t) = c_i(t)^dag c_j(t)

Channel decomposition:
- scalar:       Re(z_ij)   (or |z_ij|^2 for abs2_vacsub mode)
- pseudoscalar: Im(z_ij)

This module contains only operator construction; correlator computation
and RunHistory access belong to other modules.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.physics.qft_utils.companions import (
    PAIR_SELECTION_MODES,
    build_companion_pair_indices,
    build_companion_triplets,
)

from .config import MesonOperatorConfig
from .preparation import _safe_gather_2d, _safe_gather_3d, PreparedChannelData


# ---------------------------------------------------------------------------
# Pair-gather helpers (reshape wrappers over _safe_gather_2d / _safe_gather_3d)
# ---------------------------------------------------------------------------


def _safe_gather_pairs_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] for indices [T,N,P] using preparation helpers."""
    if values.ndim != 2 or indices.ndim != 3:
        raise ValueError(
            f"_safe_gather_pairs_2d expects values [T,N] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t, n, p = indices.shape
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = _safe_gather_2d(values, idx_flat)
    return gathered_flat.reshape(t, n, p), in_range_flat.reshape(t, n, p)


def _safe_gather_pairs_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx, :] for indices [T,N,P] using preparation helpers."""
    if values.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            f"_safe_gather_pairs_3d expects values [T,N,C] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t, n, p = indices.shape
    c = values.shape[-1]
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = _safe_gather_3d(values, idx_flat)
    return (
        gathered_flat.reshape(t, n, p, c),
        in_range_flat.reshape(t, n, p),
    )


# ---------------------------------------------------------------------------
# Inner-product computation
# ---------------------------------------------------------------------------


def _compute_inner_products_for_pairs(
    color: Tensor,
    color_valid: Tensor,
    alive: Tensor,
    pair_indices: Tensor,
    structural_valid: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute z_ij = c_i^dag c_j for companion pairs and validity mask."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T, N], got {tuple(alive.shape)}.")
    if pair_indices.shape[:2] != color.shape[:2]:
        raise ValueError(
            f"pair_indices must have shape [T, N, P] aligned with color, got "
            f"{tuple(pair_indices.shape)}."
        )
    if structural_valid.shape != pair_indices.shape:
        raise ValueError(
            "structural_valid must have the same shape as pair_indices, got "
            f"{tuple(structural_valid.shape)} vs {tuple(pair_indices.shape)}."
        )

    color_j, in_range = _safe_gather_pairs_3d(color, pair_indices)
    alive_j, _ = _safe_gather_pairs_2d(alive, pair_indices)
    valid_j, _ = _safe_gather_pairs_2d(color_valid, pair_indices)

    color_i = color.unsqueeze(2).expand_as(color_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1)

    finite = torch.isfinite(inner.real) & torch.isfinite(inner.imag)
    valid = (
        structural_valid
        & in_range
        & alive.unsqueeze(-1)
        & alive_j
        & color_valid.unsqueeze(-1)
        & valid_j
        & finite
    )
    if eps > 0:
        valid = valid & (inner.abs() > eps)

    inner = torch.where(valid, inner, torch.zeros_like(inner))
    return inner, valid


# ---------------------------------------------------------------------------
# Operator mode helpers
# ---------------------------------------------------------------------------


def _resolve_meson_operator_mode(operator_mode: str | None) -> str:
    """Normalize meson operator mode name."""
    if operator_mode is None or not str(operator_mode).strip():
        return "standard"
    mode = str(operator_mode).strip().lower()
    if mode not in {"standard", "score_directed", "score_weighted", "abs2_vacsub"}:
        msg = (
            "operator_mode must be one of "
            "{'standard','score_directed','score_weighted','abs2_vacsub'}."
        )
        raise ValueError(msg)
    return mode


def _orient_inner_products_by_scores(
    *,
    inner: Tensor,
    valid: Tensor,
    scores: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Orient pair inner products uphill according to score differences."""
    score_j, in_range = _safe_gather_pairs_2d(scores, pair_indices)
    score_i = scores.unsqueeze(-1).expand_as(score_j)
    finite_scores = torch.isfinite(score_i) & torch.isfinite(score_j)
    oriented_valid = valid & in_range & finite_scores
    ds = score_j - score_i
    inner_oriented = torch.where(ds >= 0, inner, torch.conj(inner))
    inner_oriented = torch.where(oriented_valid, inner_oriented, torch.zeros_like(inner_oriented))
    return inner_oriented, oriented_valid


def _weight_inner_products_by_score_gap(
    *,
    inner: Tensor,
    valid: Tensor,
    scores: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Weight pair inner products by |delta_score|, preserving pair orientation."""
    score_j, in_range = _safe_gather_pairs_2d(scores, pair_indices)
    score_i = scores.unsqueeze(-1).expand_as(score_j)
    finite_scores = torch.isfinite(score_i) & torch.isfinite(score_j)
    weighted_valid = valid & in_range & finite_scores
    gap = (score_j - score_i).abs()
    inner_weighted = inner * gap.to(dtype=inner.real.dtype).to(dtype=inner.dtype)
    inner_weighted = torch.where(weighted_valid, inner_weighted, torch.zeros_like(inner_weighted))
    return inner_weighted, weighted_valid


# ---------------------------------------------------------------------------
# Per-frame averaging
# ---------------------------------------------------------------------------


def _per_frame_series(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Average pair values per frame with masking."""
    weights = valid.to(values.dtype)
    counts = valid.sum(dim=(1, 2)).to(torch.int64)
    sums = (values * weights).sum(dim=(1, 2))
    series = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
    valid_t = counts > 0
    if torch.any(valid_t):
        series[valid_t] = (sums[valid_t] / counts[valid_t].to(values.dtype)).float()
    return series, counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_meson_operators(
    data: PreparedChannelData,
    config: MesonOperatorConfig,
) -> dict[str, Tensor]:
    """Compute scalar and pseudoscalar meson operator time series.

    Args:
        data: Pre-extracted channel tensors from :func:`prepare_channel_data`.
        config: Meson operator configuration.

    Returns:
        Dictionary with keys ``"scalar"`` and ``"pseudoscalar"``, each a
        ``[T]`` tensor of per-frame averaged operator values.
    """
    device = data.device
    t_total = int(data.color.shape[0])

    if t_total == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return {"scalar": empty, "pseudoscalar": empty.clone()}

    resolved_mode = _resolve_meson_operator_mode(config.operator_mode)
    if resolved_mode in {"score_directed", "score_weighted"}:
        if data.scores is None:
            msg = (
                "scores is required when operator_mode is one of "
                "{'score_directed','score_weighted'}."
            )
            raise ValueError(msg)

    pair_selection = str(config.pair_selection).strip().lower()

    # 1. Build companion pair indices
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=pair_selection,
    )

    # 2. Compute inner products
    inner, valid = _compute_inner_products_for_pairs(
        color=data.color,
        color_valid=data.color_valid,
        alive=data.alive,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=data.eps,
    )

    # 3. Apply score orientation/weighting if configured
    if resolved_mode == "score_directed":
        inner, valid = _orient_inner_products_by_scores(
            inner=inner,
            valid=valid,
            scores=data.scores,
            pair_indices=pair_indices,
        )
    elif resolved_mode == "score_weighted":
        inner, valid = _weight_inner_products_by_score_gap(
            inner=inner,
            valid=valid,
            scores=data.scores,
            pair_indices=pair_indices,
        )

    # 4. Compute scalar and pseudoscalar observables
    if resolved_mode == "abs2_vacsub":
        scalar_obs = inner.abs().square().float()
    else:
        scalar_obs = inner.real.float()
    pseudoscalar_obs = inner.imag.float()

    # 5-6. Average per frame (multiscale or single-scale)
    if data.scales is not None and data.pairwise_distances is not None:
        from .multiscale import gate_pair_validity_by_scale, per_frame_series_multiscale

        valid_ms = gate_pair_validity_by_scale(
            valid, pair_indices, data.pairwise_distances, data.scales,
        )
        scalar_series = per_frame_series_multiscale(scalar_obs, valid_ms)
        pseudoscalar_series = per_frame_series_multiscale(pseudoscalar_obs, valid_ms)
    else:
        scalar_series, _ = _per_frame_series(scalar_obs, valid)
        pseudoscalar_series, _ = _per_frame_series(pseudoscalar_obs, valid)

    # 7. Return operator time series
    return {"scalar": scalar_series, "pseudoscalar": pseudoscalar_series}
