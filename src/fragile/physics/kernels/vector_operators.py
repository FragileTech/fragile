"""Vector / axial-vector meson operator construction from companion pairs.

Builds J=1 meson channels from color-singlet companion pair observables:

    z_ij(t) = c_i(t)^dag c_j(t)
    dx_ij(t) = x_j(t) - x_i(t)

Vector channels:
- vector:       Re(z_ij) * dx_ij
- axial_vector: Im(z_ij) * dx_ij

Score-directed variants are also supported:
- operator_mode="score_directed": orient color phase uphill using cloning scores
- projection_mode in {"full","longitudinal","transverse"}: project displacement
  relative to the local score-gradient direction.

This module contains only operator construction; correlator computation
and RunHistory access belong to other modules.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import VectorOperatorConfig
from .meson_operators import (
    _safe_gather_pairs_2d,
    _safe_gather_pairs_3d,
    build_companion_pair_indices,
)
from .preparation import PreparedChannelData


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_OPERATOR_MODES = ("standard", "score_directed", "score_gradient")
VECTOR_PROJECTION_MODES = ("full", "longitudinal", "transverse")


# ---------------------------------------------------------------------------
# Mode resolvers
# ---------------------------------------------------------------------------


def _resolve_vector_operator_mode(operator_mode: str | None) -> str:
    """Normalize vector meson operator mode."""
    if operator_mode is None or not str(operator_mode).strip():
        return "standard"
    mode = str(operator_mode).strip().lower()
    if mode not in VECTOR_OPERATOR_MODES:
        raise ValueError(f"operator_mode must be one of {VECTOR_OPERATOR_MODES}.")
    return mode


def _resolve_vector_projection_mode(projection_mode: str | None) -> str:
    """Normalize displacement projection mode for vector meson operators."""
    if projection_mode is None or not str(projection_mode).strip():
        return "full"
    mode = str(projection_mode).strip().lower()
    if mode not in VECTOR_PROJECTION_MODES:
        raise ValueError(f"projection_mode must be one of {VECTOR_PROJECTION_MODES}.")
    return mode


# ---------------------------------------------------------------------------
# Pair observable computation (inner products + displacements)
# ---------------------------------------------------------------------------


def _compute_pair_observables(
    color: Tensor,
    color_valid: Tensor,
    positions: Tensor,
    alive: Tensor,
    pair_indices: Tensor,
    structural_valid: Tensor,
    eps: float,
    use_unit_displacement: bool,
    *,
    operator_mode: str,
    projection_mode: str,
    scores: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute pair inner products and displacement vectors with validity mask."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T,N,3], got {tuple(color.shape)}.")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape [T,N,3], got {tuple(positions.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T,N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T,N], got {tuple(alive.shape)}.")
    if pair_indices.shape[:2] != color.shape[:2]:
        raise ValueError(
            f"pair_indices must have shape [T,N,P] aligned with color, got {tuple(pair_indices.shape)}."
        )
    if structural_valid.shape != pair_indices.shape:
        raise ValueError(
            "structural_valid must have the same shape as pair_indices, got "
            f"{tuple(structural_valid.shape)} vs {tuple(pair_indices.shape)}."
        )

    color_j, in_range = _safe_gather_pairs_3d(color, pair_indices)
    alive_j, _ = _safe_gather_pairs_2d(alive, pair_indices)
    valid_j, _ = _safe_gather_pairs_2d(color_valid, pair_indices)
    pos_j, _ = _safe_gather_pairs_3d(positions, pair_indices)

    color_i = color.unsqueeze(2).expand_as(color_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1)

    pos_i = positions.unsqueeze(2).expand_as(pos_j)
    displacement = (pos_j - pos_i).float()

    finite_inner = torch.isfinite(inner.real) & torch.isfinite(inner.imag)
    finite_disp = torch.isfinite(displacement).all(dim=-1)
    valid = (
        structural_valid
        & in_range
        & alive.unsqueeze(-1)
        & alive_j
        & color_valid.unsqueeze(-1)
        & valid_j
        & finite_inner
        & finite_disp
    )
    if eps > 0:
        valid = valid & (inner.abs() > eps)

    if use_unit_displacement:
        disp_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        norm_floor = float(max(eps, 1e-20))
        valid = valid & (disp_norm.squeeze(-1) > norm_floor)
        displacement = displacement / disp_norm.clamp(min=norm_floor)

    resolved_operator_mode = _resolve_vector_operator_mode(operator_mode)
    resolved_projection_mode = _resolve_vector_projection_mode(projection_mode)
    if resolved_operator_mode in {"score_directed", "score_gradient"}:
        if scores is None:
            msg = (
                "scores is required when operator_mode is one of "
                "{'score_directed','score_gradient'}."
            )
            raise ValueError(msg)
        if scores.shape != color.shape[:2]:
            raise ValueError(
                f"scores must have shape [T,N] aligned with color, got {tuple(scores.shape)}."
            )
        score_j, score_in_range = _safe_gather_pairs_2d(scores, pair_indices)
        score_i = scores.unsqueeze(-1).expand_as(score_j)
        finite_scores = torch.isfinite(score_i) & torch.isfinite(score_j)
        valid = valid & score_in_range & finite_scores
        ds = score_j - score_i
        if resolved_operator_mode == "score_directed":
            inner = torch.where(ds >= 0, inner, torch.conj(inner))

        if resolved_operator_mode == "score_gradient":
            norm_floor = float(max(eps, 1e-20))
            disp_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
            unit_disp = displacement / disp_norm.clamp(min=norm_floor)
            grad_valid = valid & (disp_norm.squeeze(-1) > norm_floor)
            grad_weights = grad_valid.to(displacement.dtype).unsqueeze(-1)

            grad_terms = ds.unsqueeze(-1).to(displacement.dtype) * unit_disp
            grad_sum = (grad_terms * grad_weights).sum(dim=2)
            grad_count = grad_weights.sum(dim=2)
            grad = grad_sum / grad_count.clamp(min=1.0)
            displacement = grad.unsqueeze(2).expand_as(displacement)
        elif resolved_projection_mode != "full":
            norm_floor = float(max(eps, 1e-20))
            disp_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
            unit_disp = displacement / disp_norm.clamp(min=norm_floor)
            grad_valid = valid & (disp_norm.squeeze(-1) > norm_floor)
            grad_weights = grad_valid.to(displacement.dtype).unsqueeze(-1)

            grad_terms = ds.unsqueeze(-1).to(displacement.dtype) * unit_disp
            grad_sum = (grad_terms * grad_weights).sum(dim=2)
            grad_count = grad_weights.sum(dim=2)
            grad = grad_sum / grad_count.clamp(min=1.0)
            grad_norm = torch.linalg.vector_norm(grad, dim=-1, keepdim=True)
            n_hat = grad / grad_norm.clamp(min=norm_floor)

            parallel = (displacement * n_hat.unsqueeze(2)).sum(dim=-1, keepdim=True)
            disp_parallel = parallel * n_hat.unsqueeze(2)
            if resolved_projection_mode == "longitudinal":
                displacement = disp_parallel
            else:
                displacement = displacement - disp_parallel

    inner = torch.where(valid, inner, torch.zeros_like(inner))
    displacement = torch.where(valid.unsqueeze(-1), displacement, torch.zeros_like(displacement))
    return inner, displacement, valid


# ---------------------------------------------------------------------------
# Per-frame vector averaging
# ---------------------------------------------------------------------------


def _per_frame_vector_series(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Average vector observables per frame."""
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"values must have shape [T,N,P,3], got {tuple(values.shape)}.")
    if valid.shape != values.shape[:3]:
        raise ValueError(
            f"valid must have shape [T,N,P] matching values, got {tuple(valid.shape)}."
        )

    weights = valid.to(values.dtype).unsqueeze(-1)
    counts = valid.sum(dim=(1, 2)).to(torch.int64)
    sums = (values * weights).sum(dim=(1, 2))
    series = torch.zeros(values.shape[0], 3, dtype=torch.float32, device=values.device)
    valid_t = counts > 0
    if torch.any(valid_t):
        series[valid_t] = (sums[valid_t] / counts[valid_t].to(values.dtype).unsqueeze(-1)).float()
    return series, counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_vector_operators(
    data: PreparedChannelData,
    config: VectorOperatorConfig,
) -> dict[str, Tensor]:
    """Compute vector and axial-vector meson operator time series.

    Args:
        data: Pre-extracted channel tensors from :func:`prepare_channel_data`.
        config: Vector operator configuration.

    Returns:
        Dictionary with keys ``"vector"`` and ``"axial_vector"``, each a
        ``[T, 3]`` tensor of per-frame averaged operator values.
    """
    device = data.device
    t_total = int(data.color.shape[0])

    if t_total == 0:
        empty = torch.zeros(0, 3, dtype=torch.float32, device=device)
        return {"vector": empty, "axial_vector": empty.clone()}

    if data.positions is None:
        raise ValueError("positions must be provided in PreparedChannelData for vector operators.")

    resolved_operator_mode = _resolve_vector_operator_mode(config.operator_mode)
    resolved_projection_mode = _resolve_vector_projection_mode(config.projection_mode)
    if resolved_operator_mode in {"score_directed", "score_gradient"}:
        if data.scores is None:
            msg = (
                "scores is required when operator_mode is one of "
                "{'score_directed','score_gradient'}."
            )
            raise ValueError(msg)

    pair_selection = str(config.pair_selection).strip().lower()

    # 1. Build companion pair indices
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=pair_selection,
    )

    # 2. Compute pair observables (inner products + displacements)
    inner, displacement, valid = _compute_pair_observables(
        color=data.color,
        color_valid=data.color_valid,
        positions=data.positions,
        alive=data.alive,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=data.eps,
        use_unit_displacement=config.use_unit_displacement,
        operator_mode=resolved_operator_mode,
        projection_mode=resolved_projection_mode,
        scores=data.scores,
    )

    # 3. Build vector and axial-vector observables
    vector_obs = inner.real.float().unsqueeze(-1) * displacement
    axial_obs = inner.imag.float().unsqueeze(-1) * displacement

    # 4. Average per frame
    vector_series, _ = _per_frame_vector_series(vector_obs, valid)
    axial_series, _ = _per_frame_vector_series(axial_obs, valid)

    # 5. Return operator time series
    return {"vector": vector_series, "axial_vector": axial_series}
