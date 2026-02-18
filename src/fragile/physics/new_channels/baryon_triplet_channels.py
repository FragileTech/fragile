"""Vectorized companion-triplet baryon observables and diagnostics.

This module implements:
1. A color-determinant baryon correlator using companion triplets (i, j, k)
   with j=companions_distance[i], k=companions_clone[i].
2. A fast triplet-coherence diagnostic based on velocity determinants.

Both implementations are vectorized over time and walkers and avoid Python loops
over triplets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils import (
    build_companion_triplets,
    resolve_3d_dims,
    resolve_frame_indices,
    safe_gather_2d,
    safe_gather_3d,
)
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0


@dataclass
class BaryonTripletCorrelatorConfig:
    """Configuration for companion-triplet baryon correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    eps: float = 1e-12
    operator_mode: str = "det_abs"
    flux_exp_alpha: float = 1.0


@dataclass
class BaryonTripletCorrelatorOutput:
    """Baryon correlator output and diagnostics."""

    correlator: Tensor
    correlator_raw: Tensor
    correlator_connected: Tensor
    counts: Tensor
    frame_indices: list[int]
    triplet_counts_per_frame: Tensor
    disconnected_contribution: float
    mean_baryon_real: float
    mean_baryon_imag: float
    n_valid_source_triplets: int
    operator_baryon_series: Tensor


@dataclass
class TripletCoherenceConfig:
    """Configuration for companion-chain triplet coherence diagnostics."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_hops: int = 15
    velocity_dims: tuple[int, int, int] | None = None
    eps: float = 1e-12


@dataclass
class TripletCoherenceOutput:
    """Triplet coherence diagnostic output."""

    coherence: Tensor
    counts: Tensor
    frame_indices: list[int]


def _det3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Compute 3x3 determinant from column vectors a, b, c."""
    return (
        a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
        - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
        + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    )


def _compute_determinants_for_indices(
    vectors: Tensor,
    valid_vectors: Tensor,
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
    if companion_j.shape != vectors.shape[:2] or companion_k.shape != vectors.shape[:2]:
        msg = "companion indices must have shape [T, N]."
        raise ValueError(msg)

    _, _, structural_valid = build_companion_triplets(companion_j, companion_k)[1:]

    vec_j, in_j = safe_gather_3d(vectors, companion_j)
    vec_k, in_k = safe_gather_3d(vectors, companion_k)

    valid_j, _ = safe_gather_2d(valid_vectors, companion_j)
    valid_k, _ = safe_gather_2d(valid_vectors, companion_k)

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
    if scores.shape != color.shape[:2]:
        raise ValueError(f"scores must have shape [T,N], got {tuple(scores.shape)}.")
    if companion_j.shape != color.shape[:2] or companion_k.shape != color.shape[:2]:
        msg = "companion indices must have shape [T,N]."
        raise ValueError(msg)

    _, _, structural_valid = build_companion_triplets(companion_j, companion_k)[1:]

    color_j, in_j = safe_gather_3d(color, companion_j)
    color_k, in_k = safe_gather_3d(color, companion_k)
    valid_j, _ = safe_gather_2d(color_valid, companion_j)
    valid_k, _ = safe_gather_2d(color_valid, companion_k)
    score_j, score_j_in_range = safe_gather_2d(scores, companion_j)
    score_k, score_k_in_range = safe_gather_2d(scores, companion_k)

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
    companion_j: Tensor,
    companion_k: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute companion-triplet plaquette Π_i and validity mask."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if companion_j.shape != color.shape[:2] or companion_k.shape != color.shape[:2]:
        msg = "companion indices must have shape [T, N]."
        raise ValueError(msg)

    _, _, structural_valid = build_companion_triplets(companion_j, companion_k)[1:]

    color_j, in_j = safe_gather_3d(color, companion_j)
    color_k, in_k = safe_gather_3d(color, companion_k)
    valid_j, _ = safe_gather_2d(color_valid, companion_j)
    valid_k, _ = safe_gather_2d(color_valid, companion_k)

    z_ij = (torch.conj(color) * color_j).sum(dim=-1)
    z_jk = (torch.conj(color_j) * color_k).sum(dim=-1)
    z_ki = (torch.conj(color_k) * color).sum(dim=-1)
    pi = z_ij * z_jk * z_ki

    finite = torch.isfinite(pi.real) & torch.isfinite(pi.imag)
    valid = (
        structural_valid
        & in_j
        & in_k
        & color_valid
        & valid_j
        & valid_k
        & finite
    )
    if eps > 0:
        valid = valid & (z_ij.abs() > eps) & (z_jk.abs() > eps) & (z_ki.abs() > eps)

    pi = torch.where(valid, pi, torch.zeros_like(pi))
    return pi, valid


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


def compute_baryon_correlator_from_color(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_lag: int = 80,
    use_connected: bool = True,
    eps: float = 1e-12,
    operator_mode: str = "det_abs",
    flux_exp_alpha: float = 1.0,
    scores: Tensor | None = None,
    frame_indices: list[int] | None = None,
) -> BaryonTripletCorrelatorOutput:
    """Compute companion-triplet baryon correlator from precomputed color states.

    Correlator definition (source-triplet propagation):
        C(Δt) = < B*(t, i, j_t(i), k_t(i)) * B(t+Δt, i, j_t(i), k_t(i)) >

    where j_t(i), k_t(i) are defined at source time t from companion arrays.
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T, N] aligned with color."
        raise ValueError(msg)
    resolved_operator_mode = _resolve_baryon_operator_mode(operator_mode)
    if resolved_operator_mode in {"score_signed", "score_abs"}:
        if scores is None:
            msg = "scores is required when operator_mode is one of {'score_signed','score_abs'}."
            raise ValueError(msg)
        if scores.shape != color.shape[:2]:
            raise ValueError(
                f"scores must have shape [T,N] aligned with color, got {tuple(scores.shape)}."
            )
        scores = scores.to(device=color.device, dtype=torch.float32)

    t_total = int(color.shape[0])
    max_lag = max(0, int(max_lag))
    effective_lag = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1
    device = color.device

    if t_total == 0:
        empty_f = torch.zeros(n_lags, dtype=torch.float32, device=device)
        empty_i = torch.zeros(n_lags, dtype=torch.int64, device=device)
        empty_t = torch.zeros(0, dtype=torch.int64, device=device)
        return BaryonTripletCorrelatorOutput(
            correlator=empty_f,
            correlator_raw=empty_f.clone(),
            correlator_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[] if frame_indices is None else frame_indices,
            triplet_counts_per_frame=empty_t,
            disconnected_contribution=0.0,
            mean_baryon_real=0.0,
            mean_baryon_imag=0.0,
            n_valid_source_triplets=0,
            operator_baryon_series=empty_t.float(),
        )

    if resolved_operator_mode in {"score_signed", "score_abs"}:
        source_det, source_valid = _compute_score_ordered_determinants_for_indices(
            color=color,
            color_valid=color_valid,
            scores=scores,
            companion_j=companions_distance,
            companion_k=companions_clone,
            eps=eps,
        )
        source_obs = (
            source_det.real.float()
            if resolved_operator_mode == "score_signed"
            else source_det.abs().float()
        )
    else:
        source_det, source_valid = _compute_determinants_for_indices(
            vectors=color,
            valid_vectors=color_valid,
            companion_j=companions_distance,
            companion_k=companions_clone,
            eps=eps,
        )
        source_obs = source_det.abs().float()
    if resolved_operator_mode in {"flux_action", "flux_sin2", "flux_exp"}:
        source_pi, source_pi_valid = _compute_triplet_plaquette_for_indices(
            color=color,
            color_valid=color_valid,
            companion_j=companions_distance,
            companion_k=companions_clone,
            eps=eps,
        )
        source_flux_weight = _baryon_flux_weight_from_plaquette(
            pi=source_pi,
            operator_mode=resolved_operator_mode,
            flux_exp_alpha=float(flux_exp_alpha),
        )
        source_valid = source_valid & source_pi_valid
        source_obs = source_obs * source_flux_weight
        source_obs = torch.where(source_valid, source_obs, torch.zeros_like(source_obs))

    triplet_counts_per_frame = source_valid.sum(dim=1).to(torch.int64)
    n_valid_source_triplets = int(source_valid.sum().item())
    operator_baryon_series = torch.zeros(t_total, dtype=torch.float32, device=device)
    valid_t = triplet_counts_per_frame > 0
    if torch.any(valid_t):
        weight = source_valid.to(dtype=torch.float32)
        sums = (source_obs * weight).sum(dim=1)
        operator_baryon_series[valid_t] = sums[valid_t] / triplet_counts_per_frame[valid_t].to(
            torch.float32
        )

    if resolved_operator_mode == "det_abs":
        if n_valid_source_triplets > 0:
            mean_baryon = source_det[source_valid].mean()
        else:
            mean_baryon = torch.zeros((), dtype=source_det.dtype, device=device)
        disconnected_contribution = float((mean_baryon.conj() * mean_baryon).real.item())
        source_centered = source_det - mean_baryon
        mean_baryon_real = float(mean_baryon.real.item())
        mean_baryon_imag = float(mean_baryon.imag.item()) if mean_baryon.is_complex() else 0.0
    else:
        if n_valid_source_triplets > 0:
            mean_obs = source_obs[source_valid].mean()
        else:
            mean_obs = torch.zeros((), dtype=torch.float32, device=device)
        disconnected_contribution = float((mean_obs * mean_obs).item())
        source_centered_scalar = source_obs - mean_obs
        mean_baryon_real = float(mean_obs.item())
        mean_baryon_imag = 0.0

    correlator_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    correlator_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.int64, device=device)

    for lag in range(effective_lag + 1):
        source_len = t_total - lag
        if resolved_operator_mode in {"score_signed", "score_abs"}:
            sink_det, sink_valid = _compute_score_ordered_determinants_for_indices(
                color=color[lag : lag + source_len],
                color_valid=color_valid[lag : lag + source_len],
                scores=scores[lag : lag + source_len],
                companion_j=companions_distance[:source_len],
                companion_k=companions_clone[:source_len],
                eps=eps,
            )
            sink_obs = (
                sink_det.real.float()
                if resolved_operator_mode == "score_signed"
                else sink_det.abs().float()
            )
        else:
            sink_det, sink_valid = _compute_determinants_for_indices(
                vectors=color[lag : lag + source_len],
                valid_vectors=color_valid[lag : lag + source_len],
                companion_j=companions_distance[:source_len],
                companion_k=companions_clone[:source_len],
                eps=eps,
            )
            sink_obs = sink_det.abs().float()
        if resolved_operator_mode in {"flux_action", "flux_sin2", "flux_exp"}:
            sink_pi, sink_pi_valid = _compute_triplet_plaquette_for_indices(
                color=color[lag : lag + source_len],
                color_valid=color_valid[lag : lag + source_len],
                companion_j=companions_distance[:source_len],
                companion_k=companions_clone[:source_len],
                eps=eps,
            )
            sink_flux_weight = _baryon_flux_weight_from_plaquette(
                pi=sink_pi,
                operator_mode=resolved_operator_mode,
                flux_exp_alpha=float(flux_exp_alpha),
            )
            sink_valid = sink_valid & sink_pi_valid
            sink_obs = sink_obs * sink_flux_weight
            sink_obs = torch.where(sink_valid, sink_obs, torch.zeros_like(sink_obs))

        valid_pair = source_valid[:source_len] & sink_valid
        count = int(valid_pair.sum().item())
        counts[lag] = count
        if count == 0:
            continue

        if resolved_operator_mode == "det_abs":
            raw_prod = (torch.conj(source_det[:source_len]) * sink_det).real
            correlator_raw[lag] = raw_prod[valid_pair].mean().float()
            conn_prod = (torch.conj(source_centered[:source_len]) * (sink_det - mean_baryon)).real
            correlator_connected[lag] = conn_prod[valid_pair].mean().float()
        else:
            raw_prod = source_obs[:source_len] * sink_obs
            correlator_raw[lag] = raw_prod[valid_pair].mean().float()
            conn_prod = source_centered_scalar[:source_len] * (sink_obs - mean_obs)
            correlator_connected[lag] = conn_prod[valid_pair].mean().float()

    selected = correlator_connected if use_connected else correlator_raw
    return BaryonTripletCorrelatorOutput(
        correlator=selected,
        correlator_raw=correlator_raw,
        correlator_connected=correlator_connected,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
        triplet_counts_per_frame=triplet_counts_per_frame,
        disconnected_contribution=disconnected_contribution,
        mean_baryon_real=mean_baryon_real,
        mean_baryon_imag=mean_baryon_imag,
        n_valid_source_triplets=n_valid_source_triplets,
        operator_baryon_series=operator_baryon_series,
    )


def compute_companion_baryon_correlator(
    history: RunHistory,
    config: BaryonTripletCorrelatorConfig | None = None,
) -> BaryonTripletCorrelatorOutput:
    """Compute vectorized companion-triplet baryon correlator from RunHistory."""
    config = config or BaryonTripletCorrelatorConfig()
    frame_indices = resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
    )
    if not frame_indices:
        n_lags = int(max(0, config.max_lag)) + 1
        empty_f = torch.zeros(n_lags, dtype=torch.float32)
        empty_i = torch.zeros(n_lags, dtype=torch.int64)
        empty_t = torch.zeros(0, dtype=torch.int64)
        return BaryonTripletCorrelatorOutput(
            correlator=empty_f,
            correlator_raw=empty_f.clone(),
            correlator_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[],
            triplet_counts_per_frame=empty_t,
            disconnected_contribution=0.0,
            mean_baryon_real=0.0,
            mean_baryon_imag=0.0,
            n_valid_source_triplets=0,
            operator_baryon_series=empty_t.float(),
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
        device=device,
        dtype=torch.long,
    )
    companions_clone = torch.as_tensor(
        history.companions_clone[start_idx - 1 : end_idx - 1], device=device, dtype=torch.long
    )
    scores = torch.as_tensor(
        history.cloning_scores[start_idx - 1 : end_idx - 1], device=device, dtype=torch.float32
    )

    return compute_baryon_correlator_from_color(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        eps=float(max(config.eps, 0.0)),
        operator_mode=str(config.operator_mode),
        flux_exp_alpha=float(config.flux_exp_alpha),
        scores=scores,
        frame_indices=frame_indices,
    )


def _compute_velocity_det(
    velocities: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    idx_k: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute |det(v_i, v_j, v_k)| with validity mask for index triplets."""
    n = velocities.shape[1]
    in_i = (idx_i >= 0) & (idx_i < n)
    in_j = (idx_j >= 0) & (idx_j < n)
    in_k = (idx_k >= 0) & (idx_k < n)
    distinct = (idx_i != idx_j) & (idx_i != idx_k) & (idx_j != idx_k)

    v_i, _ = safe_gather_3d(velocities, idx_i)
    v_j, _ = safe_gather_3d(velocities, idx_j)
    v_k, _ = safe_gather_3d(velocities, idx_k)
    det = _det3(v_i, v_j, v_k).abs().float()

    finite = torch.isfinite(det)

    valid = in_i & in_j & in_k & distinct & finite & (det > eps)
    det = torch.where(valid, det, torch.zeros_like(det))
    return det, valid


def compute_triplet_coherence_from_velocity(
    velocities: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_hops: int = 15,
    eps: float = 1e-12,
    frame_indices: list[int] | None = None,
) -> TripletCoherenceOutput:
    """Compute companion-chain triplet coherence diagnostic (vectorized)."""
    if velocities.ndim != 3 or velocities.shape[-1] != 3:
        raise ValueError(f"velocities must have shape [T, N, 3], got {tuple(velocities.shape)}.")
    if (
        companions_distance.shape != velocities.shape[:2]
        or companions_clone.shape != velocities.shape[:2]
    ):
        msg = "companion arrays must have shape [T, N] aligned with velocities."
        raise ValueError(msg)

    t_total, n, _ = velocities.shape
    max_hops = max(1, int(max_hops))
    device = velocities.device
    coherence = torch.zeros(max_hops, dtype=torch.float32, device=device)
    counts = torch.zeros(max_hops, dtype=torch.int64, device=device)
    if t_total == 0:
        return TripletCoherenceOutput(
            coherence=coherence,
            counts=counts,
            frame_indices=[] if frame_indices is None else frame_indices,
        )

    anchor = torch.arange(n, device=device, dtype=torch.long).view(1, n).expand(t_total, n)
    companion_j = companions_distance.to(torch.long)
    companion_k = companions_clone.to(torch.long)

    det0, valid0 = _compute_velocity_det(
        velocities=velocities,
        idx_i=anchor,
        idx_j=companion_j,
        idx_k=companion_k,
        eps=eps,
    )
    counts[0] = valid0.sum().to(torch.int64)
    coherence[0] = 1.0 if counts[0].item() > 0 else 0.0

    p1 = anchor
    p2 = companion_j
    p3 = companion_k
    path_valid = valid0.clone()

    for hop in range(1, max_hops):
        p1_next, in1 = safe_gather_2d(companions_distance, p1)
        p2_next, in2 = safe_gather_2d(companions_distance, p2)
        p3_next, in3 = safe_gather_2d(companions_distance, p3)
        p1 = p1_next.to(torch.long)
        p2 = p2_next.to(torch.long)
        p3 = p3_next.to(torch.long)
        path_valid = path_valid & in1 & in2 & in3

        deth, valid_h = _compute_velocity_det(
            velocities=velocities,
            idx_i=p1,
            idx_j=p2,
            idx_k=p3,
            eps=eps,
        )
        valid = path_valid & valid_h & valid0
        count = int(valid.sum().item())
        counts[hop] = count
        if count == 0:
            continue

        ratio = torch.zeros_like(deth)
        ratio[valid] = deth[valid] / det0[valid].clamp(min=float(max(eps, 1e-20)))
        coherence[hop] = ratio[valid].mean().float()

    return TripletCoherenceOutput(
        coherence=coherence,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
    )


def compute_triplet_coherence(
    history: RunHistory,
    config: TripletCoherenceConfig | None = None,
) -> TripletCoherenceOutput:
    """Compute companion-chain triplet coherence from RunHistory."""
    config = config or TripletCoherenceConfig()
    frame_indices = resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
    )
    if not frame_indices:
        max_hops = max(1, int(config.max_hops))
        return TripletCoherenceOutput(
            coherence=torch.zeros(max_hops, dtype=torch.float32),
            counts=torch.zeros(max_hops, dtype=torch.int64),
            frame_indices=[],
        )

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    velocities = history.v_before_clone[start_idx:end_idx]
    dims = resolve_3d_dims(velocities.shape[-1], config.velocity_dims, "velocity_dims")
    velocities = velocities[:, :, list(dims)].float()

    device = velocities.device
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

    return compute_triplet_coherence_from_velocity(
        velocities=velocities,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_hops=int(config.max_hops),
        eps=float(max(config.eps, 0.0)),
        frame_indices=frame_indices,
    )
