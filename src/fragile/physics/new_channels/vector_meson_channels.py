"""Vectorized vector/axial-vector meson correlators from companion pairs.

This module builds J=1 meson channels from color-singlet companion pair observables:

    z_ij(t) = c_i(t)^† c_j(t)
    dx_ij(t) = x_j(t) - x_i(t)

Vector channels:
- vector:       Re(z_ij) * dx_ij
- axial_vector: Im(z_ij) * dx_ij

Temporal correlators are computed from dot products of these 3-vectors at lagged
times, reusing source-frame companion pairs.

Score-directed variants are also supported:
- operator_mode="score_directed": orient color phase uphill using cloning scores
- projection_mode in {"full","longitudinal","transverse"}: project displacement
  relative to the local score-gradient direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0
from fragile.physics.qft_utils import (
    resolve_3d_dims,
    resolve_frame_indices,
    safe_gather_2d,
    safe_gather_3d,
)
from fragile.physics.new_channels.meson_phase_channels import (
    build_companion_pair_indices,
    PAIR_SELECTION_MODES,
)


@dataclass
class VectorMesonCorrelatorConfig:
    """Configuration for companion-pair vector/axial-vector correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    position_dims: tuple[int, int, int] | None = None
    pair_selection: str = "both"
    eps: float = 1e-12
    use_unit_displacement: bool = False
    operator_mode: str = "standard"
    projection_mode: str = "full"


@dataclass
class VectorMesonCorrelatorOutput:
    """Vector/axial-vector correlator output and diagnostics."""

    vector: Tensor
    vector_raw: Tensor
    vector_connected: Tensor
    axial_vector: Tensor
    axial_vector_raw: Tensor
    axial_vector_connected: Tensor
    counts: Tensor
    frame_indices: list[int]
    pair_counts_per_frame: Tensor
    pair_selection: str
    use_unit_displacement: bool
    mean_vector: Tensor
    mean_axial_vector: Tensor
    disconnected_vector: float
    disconnected_axial_vector: float
    n_valid_source_pairs: int
    operator_vector_series: Tensor
    operator_axial_vector_series: Tensor


VECTOR_OPERATOR_MODES = ("standard", "score_directed", "score_gradient")
VECTOR_PROJECTION_MODES = ("full", "longitudinal", "transverse")


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


def _safe_gather_pairs_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] for indices [T,N,P]."""
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
    """Safely gather values[:, idx, :] for indices [T,N,P]."""
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


def _compute_pair_observables(
    color: Tensor,
    color_valid: Tensor,
    positions: Tensor,
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


def compute_vector_meson_correlator_from_color_positions(
    color: Tensor,
    color_valid: Tensor,
    positions: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_lag: int = 80,
    use_connected: bool = True,
    pair_selection: str = "both",
    eps: float = 1e-12,
    use_unit_displacement: bool = False,
    operator_mode: str = "standard",
    projection_mode: str = "full",
    scores: Tensor | None = None,
    frame_indices: list[int] | None = None,
) -> VectorMesonCorrelatorOutput:
    """Compute vector and axial-vector meson correlators from color + positions.

    Correlator definition (source-pair propagation):
        C_X(Δt) = < O_X(t, i, j_t(i)) · O_X(t+Δt, i, j_t(i)) >
    with X in {vector, axial_vector}.
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T,N,3], got {tuple(color.shape)}.")
    if positions.shape != color.shape:
        raise ValueError(
            f"positions must have shape [T,N,3] aligned with color, got {tuple(positions.shape)}."
        )
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T,N], got {tuple(color_valid.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T,N] aligned with color."
        raise ValueError(msg)
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
        scores = scores.to(dtype=torch.float32, device=color.device)

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
        empty_v = torch.zeros(0, 3, dtype=torch.float32, device=device)
        return VectorMesonCorrelatorOutput(
            vector=empty_f,
            vector_raw=empty_f.clone(),
            vector_connected=empty_f.clone(),
            axial_vector=empty_f.clone(),
            axial_vector_raw=empty_f.clone(),
            axial_vector_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[] if frame_indices is None else frame_indices,
            pair_counts_per_frame=empty_t,
            pair_selection=mode,
            use_unit_displacement=bool(use_unit_displacement),
            mean_vector=torch.zeros(3, dtype=torch.float32, device=device),
            mean_axial_vector=torch.zeros(3, dtype=torch.float32, device=device),
            disconnected_vector=0.0,
            disconnected_axial_vector=0.0,
            n_valid_source_pairs=0,
            operator_vector_series=empty_v,
            operator_axial_vector_series=empty_v.clone(),
        )

    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        pair_selection=mode,
    )
    source_inner, source_disp, source_valid = _compute_pair_observables(
        color=color,
        color_valid=color_valid,
        positions=positions,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=eps,
        use_unit_displacement=use_unit_displacement,
        operator_mode=resolved_operator_mode,
        projection_mode=resolved_projection_mode,
        scores=scores,
    )
    source_vector = source_inner.real.float().unsqueeze(-1) * source_disp
    source_axial = source_inner.imag.float().unsqueeze(-1) * source_disp

    operator_vector_series, pair_counts_per_frame = _per_frame_vector_series(
        source_vector, source_valid
    )
    operator_axial_series, _ = _per_frame_vector_series(source_axial, source_valid)

    n_valid_source_pairs = int(source_valid.sum().item())
    if n_valid_source_pairs > 0:
        mean_vector_t = source_vector[source_valid].mean(dim=0)
        mean_axial_t = source_axial[source_valid].mean(dim=0)
    else:
        mean_vector_t = torch.zeros(3, dtype=torch.float32, device=device)
        mean_axial_t = torch.zeros(3, dtype=torch.float32, device=device)

    disconnected_vector = float((mean_vector_t * mean_vector_t).sum().item())
    disconnected_axial = float((mean_axial_t * mean_axial_t).sum().item())

    vector_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    vector_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    axial_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    axial_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.int64, device=device)

    for lag in range(effective_lag + 1):
        source_len = t_total - lag
        sink_inner, sink_disp, sink_valid = _compute_pair_observables(
            color=color[lag : lag + source_len],
            color_valid=color_valid[lag : lag + source_len],
            positions=positions[lag : lag + source_len],
            pair_indices=pair_indices[:source_len],
            structural_valid=structural_valid[:source_len],
            eps=eps,
            use_unit_displacement=use_unit_displacement,
            operator_mode=resolved_operator_mode,
            projection_mode=resolved_projection_mode,
            scores=(None if scores is None else scores[lag : lag + source_len]),
        )
        sink_vector = sink_inner.real.float().unsqueeze(-1) * sink_disp
        sink_axial = sink_inner.imag.float().unsqueeze(-1) * sink_disp

        valid_pair = source_valid[:source_len] & sink_valid
        count = int(valid_pair.sum().item())
        counts[lag] = count
        if count == 0:
            continue

        src_vector_l = source_vector[:source_len]
        src_axial_l = source_axial[:source_len]

        vec_raw_prod = (src_vector_l * sink_vector).sum(dim=-1)
        axial_raw_prod = (src_axial_l * sink_axial).sum(dim=-1)
        vector_raw[lag] = vec_raw_prod[valid_pair].mean().float()
        axial_raw[lag] = axial_raw_prod[valid_pair].mean().float()

        vec_conn_prod = ((src_vector_l - mean_vector_t) * (sink_vector - mean_vector_t)).sum(
            dim=-1
        )
        axial_conn_prod = ((src_axial_l - mean_axial_t) * (sink_axial - mean_axial_t)).sum(dim=-1)
        vector_connected[lag] = vec_conn_prod[valid_pair].mean().float()
        axial_connected[lag] = axial_conn_prod[valid_pair].mean().float()

    vector = vector_connected if use_connected else vector_raw
    axial_vector = axial_connected if use_connected else axial_raw
    return VectorMesonCorrelatorOutput(
        vector=vector,
        vector_raw=vector_raw,
        vector_connected=vector_connected,
        axial_vector=axial_vector,
        axial_vector_raw=axial_raw,
        axial_vector_connected=axial_connected,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
        pair_counts_per_frame=pair_counts_per_frame,
        pair_selection=mode,
        use_unit_displacement=bool(use_unit_displacement),
        mean_vector=mean_vector_t.float(),
        mean_axial_vector=mean_axial_t.float(),
        disconnected_vector=disconnected_vector,
        disconnected_axial_vector=disconnected_axial,
        n_valid_source_pairs=n_valid_source_pairs,
        operator_vector_series=operator_vector_series,
        operator_axial_vector_series=operator_axial_series,
    )


def compute_companion_vector_meson_correlator(
    history: RunHistory,
    config: VectorMesonCorrelatorConfig | None = None,
) -> VectorMesonCorrelatorOutput:
    """Compute vectorized companion-pair vector/axial-vector correlators."""
    config = config or VectorMesonCorrelatorConfig()
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
        empty_v = torch.zeros(0, 3, dtype=torch.float32)
        return VectorMesonCorrelatorOutput(
            vector=empty_f,
            vector_raw=empty_f.clone(),
            vector_connected=empty_f.clone(),
            axial_vector=empty_f.clone(),
            axial_vector_raw=empty_f.clone(),
            axial_vector_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[],
            pair_counts_per_frame=empty_t,
            pair_selection=str(config.pair_selection).strip().lower(),
            use_unit_displacement=bool(config.use_unit_displacement),
            mean_vector=torch.zeros(3, dtype=torch.float32),
            mean_axial_vector=torch.zeros(3, dtype=torch.float32),
            disconnected_vector=0.0,
            disconnected_axial_vector=0.0,
            n_valid_source_pairs=0,
            operator_vector_series=empty_v,
            operator_axial_vector_series=empty_v.clone(),
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
    color_dims = resolve_3d_dims(color.shape[-1], config.color_dims, "color_dims")
    color = color[:, :, list(color_dims)]

    device = color.device
    positions = torch.as_tensor(history.x_before_clone[start_idx:end_idx], device=device)
    pos_dims = resolve_3d_dims(positions.shape[-1], config.position_dims, "position_dims")
    positions = positions[:, :, list(pos_dims)].to(dtype=torch.float32, device=device)

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
    scores = None
    if _resolve_vector_operator_mode(config.operator_mode) in {"score_directed", "score_gradient"}:
        scores = torch.as_tensor(
            history.cloning_scores[start_idx - 1 : end_idx - 1],
            dtype=torch.float32,
            device=device,
        )

    return compute_vector_meson_correlator_from_color_positions(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        positions=positions,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        pair_selection=str(config.pair_selection),
        eps=float(max(config.eps, 0.0)),
        use_unit_displacement=bool(config.use_unit_displacement),
        operator_mode=str(config.operator_mode),
        projection_mode=str(config.projection_mode),
        scores=scores,
        frame_indices=frame_indices,
    )
