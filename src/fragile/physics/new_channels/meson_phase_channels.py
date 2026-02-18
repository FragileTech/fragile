"""Vectorized companion-pair meson phase correlators.

This module computes meson channel correlators from color-singlet pair
inner products:

    z_ij(t) = c_i(t)^† c_j(t)

with companion-defined pairs j in {companions_distance[i], companions_clone[i]}.

Channel decomposition:
- pseudoscalar: Im(z_ij)
- scalar: Re(z_ij)

The implementation is vectorized over time, walkers, and pair slots, with only
a lag loop for temporal correlators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0
from fragile.physics.qft_utils import (
    build_companion_triplets,
    resolve_3d_dims,
    resolve_frame_indices,
    safe_gather_2d,
    safe_gather_3d,
)


PAIR_SELECTION_MODES = ("distance", "clone", "both")


@dataclass
class MesonPhaseCorrelatorConfig:
    """Configuration for companion-pair meson phase correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    pair_selection: str = "both"
    eps: float = 1e-12
    operator_mode: str = "standard"


@dataclass
class MesonPhaseCorrelatorOutput:
    """Meson phase correlator output and diagnostics."""

    pseudoscalar: Tensor
    pseudoscalar_raw: Tensor
    pseudoscalar_connected: Tensor
    scalar: Tensor
    scalar_raw: Tensor
    scalar_connected: Tensor
    counts: Tensor
    frame_indices: list[int]
    pair_counts_per_frame: Tensor
    pair_selection: str
    mean_pseudoscalar: float
    mean_scalar: float
    disconnected_pseudoscalar: float
    disconnected_scalar: float
    n_valid_source_pairs: int
    operator_pseudoscalar_series: Tensor
    operator_scalar_series: Tensor


def _safe_gather_pairs_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] for indices [T,N,P] using baryon helpers."""
    if values.ndim != 2 or indices.ndim != 3:
        raise ValueError(
            f"_safe_gather_pairs_2d expects values [T,N] and indices [T,N,P], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t, n, p = indices.shape
    idx_flat = indices.reshape(t, n * p)
    gathered_flat, in_range_flat = safe_gather_2d(values, idx_flat)
    return gathered_flat.reshape(t, n, p), in_range_flat.reshape(t, n, p)


def _safe_gather_pairs_3d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx, :] for indices [T,N,P] using baryon helpers."""
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


def build_companion_pair_indices(
    companions_distance: Tensor,
    companions_clone: Tensor,
    pair_selection: str = "both",
) -> tuple[Tensor, Tensor]:
    """Build companion pair indices [T,N,P] and structural validity mask.

    Args:
        companions_distance: Distance companion indices [T, N].
        companions_clone: Clone companion indices [T, N].
        pair_selection: One of {"distance", "clone", "both"}.

    Returns:
        pair_indices: Companion indices [T, N, P].
        structural_valid: In-range and non-self mask [T, N, P].
    """
    mode = str(pair_selection).strip().lower()
    if mode not in PAIR_SELECTION_MODES:
        raise ValueError(f"pair_selection must be one of {PAIR_SELECTION_MODES}.")
    anchor_idx, companion_j, companion_k, _ = build_companion_triplets(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
    )
    n = companions_distance.shape[1]
    valid_j = (companion_j >= 0) & (companion_j < n) & (companion_j != anchor_idx)
    valid_k = (companion_k >= 0) & (companion_k < n) & (companion_k != anchor_idx)
    if mode == "distance":
        return companion_j.unsqueeze(-1), valid_j.unsqueeze(-1)
    if mode == "clone":
        return companion_k.unsqueeze(-1), valid_k.unsqueeze(-1)
    pair_indices = torch.stack([companion_j, companion_k], dim=-1)
    structural = torch.stack([valid_j, valid_k], dim=-1)
    return pair_indices, structural


def _compute_inner_products_for_pairs(
    color: Tensor,
    color_valid: Tensor,
    pair_indices: Tensor,
    structural_valid: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute z_ij = c_i^† c_j for companion pairs and validity mask."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
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
    valid_j, _ = _safe_gather_pairs_2d(color_valid, pair_indices)

    color_i = color.unsqueeze(2).expand_as(color_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1)

    finite = torch.isfinite(inner.real) & torch.isfinite(inner.imag)
    valid = (
        structural_valid
        & in_range
        & color_valid.unsqueeze(-1)
        & valid_j
        & finite
    )
    if eps > 0:
        valid = valid & (inner.abs() > eps)

    inner = torch.where(valid, inner, torch.zeros_like(inner))
    return inner, valid


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
    """Weight pair inner products by |Δscore|, preserving pair orientation."""
    score_j, in_range = _safe_gather_pairs_2d(scores, pair_indices)
    score_i = scores.unsqueeze(-1).expand_as(score_j)
    finite_scores = torch.isfinite(score_i) & torch.isfinite(score_j)
    weighted_valid = valid & in_range & finite_scores
    gap = (score_j - score_i).abs()
    inner_weighted = inner * gap.to(dtype=inner.real.dtype).to(dtype=inner.dtype)
    inner_weighted = torch.where(weighted_valid, inner_weighted, torch.zeros_like(inner_weighted))
    return inner_weighted, weighted_valid


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


def compute_meson_phase_correlator_from_color(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_lag: int = 80,
    use_connected: bool = True,
    pair_selection: str = "both",
    eps: float = 1e-12,
    operator_mode: str = "standard",
    scores: Tensor | None = None,
    frame_indices: list[int] | None = None,
) -> MesonPhaseCorrelatorOutput:
    """Compute scalar/pseudoscalar correlators from companion-pair color phases.

    Correlator definition (source-pair propagation):
        C_X(Δt) = < O_X(t, i, j_t(i)) O_X(t+Δt, i, j_t(i)) >
    where X in {scalar, pseudoscalar}, and j_t(i) is from the source frame.
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T, N] aligned with color."
        raise ValueError(msg)
    resolved_operator_mode = _resolve_meson_operator_mode(operator_mode)
    if resolved_operator_mode in {"score_directed", "score_weighted"}:
        if scores is None:
            msg = "scores is required when operator_mode is one of {'score_directed','score_weighted'}."
            raise ValueError(msg)
        if scores.shape != color.shape[:2]:
            raise ValueError(
                f"scores must have shape [T,N] aligned with color, got {tuple(scores.shape)}."
            )
        scores = scores.to(device=color.device, dtype=torch.float32)

    mode = str(pair_selection).strip().lower()
    if mode not in PAIR_SELECTION_MODES:
        raise ValueError(f"pair_selection must be one of {PAIR_SELECTION_MODES}.")

    t_total = int(color.shape[0])
    max_lag = max(0, int(max_lag))
    effective_lag = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1
    device = color.device

    if t_total == 0:
        empty_f = torch.zeros(n_lags, dtype=torch.float32, device=device)
        empty_i = torch.zeros(n_lags, dtype=torch.int64, device=device)
        empty_t = torch.zeros(0, dtype=torch.int64, device=device)
        return MesonPhaseCorrelatorOutput(
            pseudoscalar=empty_f,
            pseudoscalar_raw=empty_f.clone(),
            pseudoscalar_connected=empty_f.clone(),
            scalar=empty_f.clone(),
            scalar_raw=empty_f.clone(),
            scalar_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[] if frame_indices is None else frame_indices,
            pair_counts_per_frame=empty_t,
            pair_selection=mode,
            mean_pseudoscalar=0.0,
            mean_scalar=0.0,
            disconnected_pseudoscalar=0.0,
            disconnected_scalar=0.0,
            n_valid_source_pairs=0,
            operator_pseudoscalar_series=empty_t.float(),
            operator_scalar_series=empty_t.float(),
        )

    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        pair_selection=mode,
    )
    source_inner, source_valid = _compute_inner_products_for_pairs(
        color=color,
        color_valid=color_valid,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=eps,
    )
    if resolved_operator_mode == "score_directed":
        source_inner, source_valid = _orient_inner_products_by_scores(
            inner=source_inner,
            valid=source_valid,
            scores=scores,
            pair_indices=pair_indices,
        )
    elif resolved_operator_mode == "score_weighted":
        source_inner, source_valid = _weight_inner_products_by_score_gap(
            inner=source_inner,
            valid=source_valid,
            scores=scores,
            pair_indices=pair_indices,
        )
    if resolved_operator_mode == "abs2_vacsub":
        source_scalar = source_inner.abs().square().float()
    else:
        source_scalar = source_inner.real.float()
    source_pseudoscalar = source_inner.imag.float()

    operator_scalar_series, pair_counts_per_frame = _per_frame_series(source_scalar, source_valid)
    operator_pseudoscalar_series, _ = _per_frame_series(source_pseudoscalar, source_valid)

    n_valid_source_pairs = int(source_valid.sum().item())
    if n_valid_source_pairs > 0:
        mean_scalar_t = source_scalar[source_valid].mean()
        mean_pseudoscalar_t = source_pseudoscalar[source_valid].mean()
    else:
        mean_scalar_t = torch.zeros((), dtype=torch.float32, device=device)
        mean_pseudoscalar_t = torch.zeros((), dtype=torch.float32, device=device)

    disconnected_scalar = float((mean_scalar_t * mean_scalar_t).item())
    disconnected_pseudoscalar = float((mean_pseudoscalar_t * mean_pseudoscalar_t).item())

    scalar_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    scalar_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    pseudoscalar_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    pseudoscalar_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.int64, device=device)

    for lag in range(effective_lag + 1):
        source_len = t_total - lag
        sink_inner, sink_valid = _compute_inner_products_for_pairs(
            color=color[lag : lag + source_len],
            color_valid=color_valid[lag : lag + source_len],
            pair_indices=pair_indices[:source_len],
            structural_valid=structural_valid[:source_len],
            eps=eps,
        )
        if resolved_operator_mode == "score_directed":
            sink_inner, sink_valid = _orient_inner_products_by_scores(
                inner=sink_inner,
                valid=sink_valid,
                scores=scores[lag : lag + source_len],
                pair_indices=pair_indices[:source_len],
            )
        elif resolved_operator_mode == "score_weighted":
            sink_inner, sink_valid = _weight_inner_products_by_score_gap(
                inner=sink_inner,
                valid=sink_valid,
                scores=scores[lag : lag + source_len],
                pair_indices=pair_indices[:source_len],
            )
        if resolved_operator_mode == "abs2_vacsub":
            sink_scalar = sink_inner.abs().square().float()
        else:
            sink_scalar = sink_inner.real.float()
        sink_pseudoscalar = sink_inner.imag.float()

        valid_pair = source_valid[:source_len] & sink_valid
        count = int(valid_pair.sum().item())
        counts[lag] = count
        if count == 0:
            continue

        src_scalar_l = source_scalar[:source_len]
        src_ps_l = source_pseudoscalar[:source_len]

        scalar_raw_prod = src_scalar_l * sink_scalar
        ps_raw_prod = src_ps_l * sink_pseudoscalar
        scalar_raw[lag] = scalar_raw_prod[valid_pair].mean().float()
        pseudoscalar_raw[lag] = ps_raw_prod[valid_pair].mean().float()

        scalar_conn_prod = (src_scalar_l - mean_scalar_t) * (sink_scalar - mean_scalar_t)
        ps_conn_prod = (src_ps_l - mean_pseudoscalar_t) * (sink_pseudoscalar - mean_pseudoscalar_t)
        scalar_connected[lag] = scalar_conn_prod[valid_pair].mean().float()
        pseudoscalar_connected[lag] = ps_conn_prod[valid_pair].mean().float()

    scalar = scalar_connected if use_connected else scalar_raw
    pseudoscalar = pseudoscalar_connected if use_connected else pseudoscalar_raw
    return MesonPhaseCorrelatorOutput(
        pseudoscalar=pseudoscalar,
        pseudoscalar_raw=pseudoscalar_raw,
        pseudoscalar_connected=pseudoscalar_connected,
        scalar=scalar,
        scalar_raw=scalar_raw,
        scalar_connected=scalar_connected,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
        pair_counts_per_frame=pair_counts_per_frame,
        pair_selection=mode,
        mean_pseudoscalar=float(mean_pseudoscalar_t.item()),
        mean_scalar=float(mean_scalar_t.item()),
        disconnected_pseudoscalar=disconnected_pseudoscalar,
        disconnected_scalar=disconnected_scalar,
        n_valid_source_pairs=n_valid_source_pairs,
        operator_pseudoscalar_series=operator_pseudoscalar_series,
        operator_scalar_series=operator_scalar_series,
    )


def compute_companion_meson_phase_correlator(
    history: RunHistory,
    config: MesonPhaseCorrelatorConfig | None = None,
) -> MesonPhaseCorrelatorOutput:
    """Compute vectorized scalar/pseudoscalar meson phase correlators from RunHistory."""
    config = config or MesonPhaseCorrelatorConfig()
    frame_indices = resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
    )
    n_lags = int(max(0, config.max_lag)) + 1
    if not frame_indices:
        empty_f = torch.zeros(n_lags, dtype=torch.float32)
        empty_i = torch.zeros(n_lags, dtype=torch.int64)
        empty_t = torch.zeros(0, dtype=torch.int64)
        return MesonPhaseCorrelatorOutput(
            pseudoscalar=empty_f,
            pseudoscalar_raw=empty_f.clone(),
            pseudoscalar_connected=empty_f.clone(),
            scalar=empty_f.clone(),
            scalar_raw=empty_f.clone(),
            scalar_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[],
            pair_counts_per_frame=empty_t,
            pair_selection=str(config.pair_selection).strip().lower(),
            mean_pseudoscalar=0.0,
            mean_scalar=0.0,
            disconnected_pseudoscalar=0.0,
            disconnected_scalar=0.0,
            n_valid_source_pairs=0,
            operator_pseudoscalar_series=empty_t.float(),
            operator_scalar_series=empty_t.float(),
        )

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    h_eff = float(max(config.h_eff, 1e-8))
    mass = float(max(config.mass, 1e-8))
    ell0 = float(config.ell0) if config.ell0 is not None else float(estimate_ell0(history))
    if ell0 <= 0:
        msg = "ell0 must be positive."
        raise ValueError(msg)

    color, color_valid = compute_color_states_batch(
        history=history,
        start_idx=start_idx,
        h_eff=h_eff,
        mass=mass,
        ell0=ell0,
        end_idx=end_idx,
    )
    dims = resolve_3d_dims(color.shape[-1], config.color_dims, "color_dims")
    color = color[:, :, list(dims)]

    device = color.device
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
    operator_mode = _resolve_meson_operator_mode(str(config.operator_mode))
    scores: Tensor | None = None
    if operator_mode in {"score_directed", "score_weighted"}:
        if not hasattr(history, "cloning_scores"):
            msg = (
                "RunHistory is missing cloning_scores required for "
                f"operator_mode={operator_mode!r}."
            )
            raise ValueError(msg)
        scores = torch.as_tensor(
            history.cloning_scores[start_idx - 1 : end_idx - 1],
            dtype=torch.float32,
            device=device,
        )

    return compute_meson_phase_correlator_from_color(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        pair_selection=str(config.pair_selection),
        eps=float(max(config.eps, 0.0)),
        operator_mode=operator_mode,
        scores=scores,
        frame_indices=frame_indices,
    )
