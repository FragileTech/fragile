"""Baryon (nucleon) operator construction from prepared channel data.

Computes the per-frame baryon operator series B(t) from companion-triplet
determinant observables, without correlator computation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import BaryonOperatorConfig
from .preparation import PreparedChannelData, _safe_gather_2d, _safe_gather_3d


# ---------------------------------------------------------------------------
# Determinant and triplet helpers
# ---------------------------------------------------------------------------


def _det3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Compute 3x3 determinant from column vectors a, b, c."""
    return (
        a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
        - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
        + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    )


def build_companion_triplets(
    companions_distance: Tensor,
    companions_clone: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build companion triplet indices (i, j, k) and structural-valid mask.

    Args:
        companions_distance: Companion distance indices [T, N].
        companions_clone: Companion clone indices [T, N].

    Returns:
        Tuple:
            anchor_idx: [T, N] anchor indices i
            companion_j: [T, N] distance companion j
            companion_k: [T, N] clone companion k
            structural_valid: [T, N] index-range/distinctness validity
    """
    if companions_distance.shape != companions_clone.shape:
        raise ValueError(
            "companions_distance and companions_clone must have the same shape, got "
            f"{tuple(companions_distance.shape)} vs {tuple(companions_clone.shape)}."
        )
    if companions_distance.ndim != 2:
        raise ValueError(
            f"Expected companion arrays with shape [T, N], got {tuple(companions_distance.shape)}."
        )

    t, n = companions_distance.shape
    device = companions_distance.device
    anchor_idx = torch.arange(n, device=device, dtype=torch.long).view(1, n).expand(t, n)
    companion_j = companions_distance.to(torch.long)
    companion_k = companions_clone.to(torch.long)

    in_range_j = (companion_j >= 0) & (companion_j < n)
    in_range_k = (companion_k >= 0) & (companion_k < n)
    distinct = (
        (companion_j != anchor_idx) & (companion_k != anchor_idx) & (companion_j != companion_k)
    )
    structural_valid = in_range_j & in_range_k & distinct
    return anchor_idx, companion_j, companion_k, structural_valid


def _compute_determinants_for_indices(
    vectors: Tensor,
    valid_vectors: Tensor,
    alive: Tensor,
    companion_j: Tensor,
    companion_k: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute determinant observable for (i, companion_j, companion_k) triplets."""
    if vectors.ndim != 3 or vectors.shape[-1] != 3:
        raise ValueError(f"Expected vectors with shape [T, N, 3], got {tuple(vectors.shape)}.")
    if valid_vectors.shape != vectors.shape[:2]:
        raise ValueError(
            f"valid_vectors must have shape [T, N], got {tuple(valid_vectors.shape)}."
        )
    if alive.shape != vectors.shape[:2]:
        raise ValueError(f"alive must have shape [T, N], got {tuple(alive.shape)}.")
    if companion_j.shape != vectors.shape[:2] or companion_k.shape != vectors.shape[:2]:
        msg = "companion indices must have shape [T, N]."
        raise ValueError(msg)

    _, _, structural_valid = build_companion_triplets(companion_j, companion_k)[1:]

    vec_j, in_j = _safe_gather_3d(vectors, companion_j)
    vec_k, in_k = _safe_gather_3d(vectors, companion_k)

    alive_j, _ = _safe_gather_2d(alive, companion_j)
    alive_k, _ = _safe_gather_2d(alive, companion_k)
    valid_j, _ = _safe_gather_2d(valid_vectors, companion_j)
    valid_k, _ = _safe_gather_2d(valid_vectors, companion_k)

    det = _det3(vectors, vec_j, vec_k)
    finite = (
        torch.isfinite(det.real) & torch.isfinite(det.imag)
        if det.is_complex()
        else torch.isfinite(det)
    )

    valid = (
        structural_valid
        & in_j
        & in_k
        & alive
        & alive_j
        & alive_k
        & valid_vectors
        & valid_j
        & valid_k
        & finite
    )
    if eps > 0:
        valid = valid & (det.abs() > eps)

    det = torch.where(valid, det, torch.zeros_like(det))
    return det, valid


def _compute_score_ordered_determinants_for_indices(
    *,
    color: Tensor,
    color_valid: Tensor,
    alive: Tensor,
    scores: Tensor,
    companion_j: Tensor,
    companion_k: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute score-ordered determinant for companion triplets."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T,N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T,N], got {tuple(alive.shape)}.")
    if scores.shape != color.shape[:2]:
        raise ValueError(f"scores must have shape [T,N], got {tuple(scores.shape)}.")
    if companion_j.shape != color.shape[:2] or companion_k.shape != color.shape[:2]:
        msg = "companion indices must have shape [T,N]."
        raise ValueError(msg)

    _, _, structural_valid = build_companion_triplets(companion_j, companion_k)[1:]

    color_j, in_j = _safe_gather_3d(color, companion_j)
    color_k, in_k = _safe_gather_3d(color, companion_k)
    alive_j, _ = _safe_gather_2d(alive, companion_j)
    alive_k, _ = _safe_gather_2d(alive, companion_k)
    valid_j, _ = _safe_gather_2d(color_valid, companion_j)
    valid_k, _ = _safe_gather_2d(color_valid, companion_k)
    score_j, score_j_in_range = _safe_gather_2d(scores, companion_j)
    score_k, score_k_in_range = _safe_gather_2d(scores, companion_k)

    triplet_scores = torch.stack([scores, score_j, score_k], dim=-1)  # [T,N,3]
    triplet_colors = torch.stack([color, color_j, color_k], dim=-2)  # [T,N,3,3]
    order = torch.argsort(triplet_scores, dim=-1)
    ordered_colors = torch.gather(
        triplet_colors,
        dim=-2,
        index=order.unsqueeze(-1).expand(-1, -1, -1, 3),
    )
    det_ordered = _det3(
        ordered_colors[..., 0, :],
        ordered_colors[..., 1, :],
        ordered_colors[..., 2, :],
    )

    finite_det = torch.isfinite(det_ordered.real) & torch.isfinite(det_ordered.imag)
    finite_scores = torch.isfinite(scores) & torch.isfinite(score_j) & torch.isfinite(score_k)
    valid = (
        structural_valid
        & in_j
        & in_k
        & alive
        & alive_j
        & alive_k
        & color_valid
        & valid_j
        & valid_k
        & score_j_in_range
        & score_k_in_range
        & finite_scores
        & finite_det
    )
    if eps > 0:
        valid = valid & (det_ordered.abs() > eps)

    det_ordered = torch.where(valid, det_ordered, torch.zeros_like(det_ordered))
    return det_ordered, valid


def _compute_triplet_plaquette_for_indices(
    *,
    color: Tensor,
    color_valid: Tensor,
    alive: Tensor,
    companion_j: Tensor,
    companion_k: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute companion-triplet plaquette Pi_i and validity mask."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T, N], got {tuple(alive.shape)}.")
    if companion_j.shape != color.shape[:2] or companion_k.shape != color.shape[:2]:
        msg = "companion indices must have shape [T, N]."
        raise ValueError(msg)

    _, _, structural_valid = build_companion_triplets(companion_j, companion_k)[1:]

    color_j, in_j = _safe_gather_3d(color, companion_j)
    color_k, in_k = _safe_gather_3d(color, companion_k)
    alive_j, _ = _safe_gather_2d(alive, companion_j)
    alive_k, _ = _safe_gather_2d(alive, companion_k)
    valid_j, _ = _safe_gather_2d(color_valid, companion_j)
    valid_k, _ = _safe_gather_2d(color_valid, companion_k)

    z_ij = (torch.conj(color) * color_j).sum(dim=-1)
    z_jk = (torch.conj(color_j) * color_k).sum(dim=-1)
    z_ki = (torch.conj(color_k) * color).sum(dim=-1)
    pi = z_ij * z_jk * z_ki

    finite = torch.isfinite(pi.real) & torch.isfinite(pi.imag)
    valid = (
        structural_valid
        & in_j
        & in_k
        & alive
        & alive_j
        & alive_k
        & color_valid
        & valid_j
        & valid_k
        & finite
    )
    if eps > 0:
        valid = valid & (z_ij.abs() > eps) & (z_jk.abs() > eps) & (z_ki.abs() > eps)

    pi = torch.where(valid, pi, torch.zeros_like(pi))
    return pi, valid


def _baryon_flux_weight_from_plaquette(
    *,
    pi: Tensor,
    operator_mode: str,
    flux_exp_alpha: float,
) -> Tensor:
    """Compute gauge-flux weight from plaquette phase."""
    phase = torch.angle(pi)
    if operator_mode == "flux_action":
        return (1.0 - torch.cos(phase)).float()
    if operator_mode == "flux_sin2":
        return torch.sin(phase).square().float()
    if operator_mode == "flux_exp":
        action = 1.0 - torch.cos(phase)
        alpha = float(max(flux_exp_alpha, 0.0))
        return torch.exp(alpha * action).float()
    msg = (
        "Invalid baryon operator_mode. Expected one of "
        "{'det_abs','flux_action','flux_sin2','flux_exp','score_signed','score_abs'}."
    )
    raise ValueError(msg)


def _resolve_baryon_operator_mode(operator_mode: str | None) -> str:
    """Normalize baryon operator mode name."""
    if operator_mode is None or not str(operator_mode).strip():
        return "det_abs"
    mode = str(operator_mode).strip().lower()
    allowed = {
        "det_abs",
        "flux_action",
        "flux_sin2",
        "flux_exp",
        "score_signed",
        "score_abs",
    }
    if mode not in allowed:
        msg = (
            "Invalid baryon operator_mode. Expected one of "
            "{'det_abs','flux_action','flux_sin2','flux_exp','score_signed','score_abs'}."
        )
        raise ValueError(msg)
    return mode


# ---------------------------------------------------------------------------
# Main operator computation
# ---------------------------------------------------------------------------


def compute_baryon_operators(
    data: PreparedChannelData,
    config: BaryonOperatorConfig,
) -> dict[str, Tensor]:
    """Compute per-frame baryon (nucleon) operator series.

    Args:
        data: Pre-extracted channel data from :func:`prepare_channel_data`.
        config: Baryon operator configuration.

    Returns:
        ``{"nucleon": series}`` where *series* is a ``[T]`` real-valued tensor
        giving the frame-averaged baryon observable at each Monte Carlo time.
    """
    resolved_mode = _resolve_baryon_operator_mode(config.operator_mode)
    device = data.device
    t_total = int(data.color.shape[0])

    if t_total == 0:
        return {"nucleon": torch.zeros(0, dtype=torch.float32, device=device)}

    # Validate score requirement
    if resolved_mode in {"score_signed", "score_abs"}:
        if data.scores is None:
            msg = "scores is required when operator_mode is one of {'score_signed','score_abs'}."
            raise ValueError(msg)
        scores = data.scores.to(device=device, dtype=torch.float32)
    else:
        scores = None

    eps = data.eps

    # --- Compute source observable ---
    if resolved_mode in {"score_signed", "score_abs"}:
        source_det, source_valid = _compute_score_ordered_determinants_for_indices(
            color=data.color,
            color_valid=data.color_valid,
            alive=data.alive,
            scores=scores,
            companion_j=data.companions_distance,
            companion_k=data.companions_clone,
            eps=eps,
        )
        source_obs = (
            source_det.real.float()
            if resolved_mode == "score_signed"
            else source_det.abs().float()
        )
    else:
        source_det, source_valid = _compute_determinants_for_indices(
            vectors=data.color,
            valid_vectors=data.color_valid,
            alive=data.alive,
            companion_j=data.companions_distance,
            companion_k=data.companions_clone,
            eps=eps,
        )
        source_obs = source_det.abs().float()

    # --- Apply flux weight for flux modes ---
    if resolved_mode in {"flux_action", "flux_sin2", "flux_exp"}:
        source_pi, source_pi_valid = _compute_triplet_plaquette_for_indices(
            color=data.color,
            color_valid=data.color_valid,
            alive=data.alive,
            companion_j=data.companions_distance,
            companion_k=data.companions_clone,
            eps=eps,
        )
        source_flux_weight = _baryon_flux_weight_from_plaquette(
            pi=source_pi,
            operator_mode=resolved_mode,
            flux_exp_alpha=float(config.flux_exp_alpha),
        )
        source_valid = source_valid & source_pi_valid
        source_obs = source_obs * source_flux_weight
        source_obs = torch.where(source_valid, source_obs, torch.zeros_like(source_obs))

    # --- Per-frame averaging ---
    triplet_counts_per_frame = source_valid.sum(dim=1).to(torch.int64)
    operator_baryon_series = torch.zeros(t_total, dtype=torch.float32, device=device)
    valid_t = triplet_counts_per_frame > 0
    if torch.any(valid_t):
        weight = source_valid.to(dtype=torch.float32)
        sums = (source_obs * weight).sum(dim=1)
        operator_baryon_series[valid_t] = sums[valid_t] / triplet_counts_per_frame[valid_t].to(
            torch.float32
        )

    return {"nucleon": operator_baryon_series}
