"""Vectorized companion-triplet glueball correlators from color plaquettes.

This module computes an SU(3)-inspired glueball observable from companion
triplets (i, j, k), where j=companions_distance[i], k=companions_clone[i]:

    Π_i(t) = (c_i† c_j)(c_j† c_k)(c_k† c_i)

Glueball scalar operator at each source triplet:
- Re(Π_i), or
- 1 - Re(Π_i) (action-style form, configurable).

Temporal correlators are computed by propagating source-frame triplets to sink
times with fixed source indices, matching the override pattern used by other
companion-based anisotropic channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.aggregation import compute_color_states_batch, estimate_ell0
from fragile.fractalai.qft.baryon_triplet_channels import (
    _resolve_3d_dims,
    _resolve_frame_indices,
    _safe_gather_2d,
    _safe_gather_3d,
    build_companion_triplets,
)


@dataclass
class GlueballColorCorrelatorConfig:
    """Configuration for companion-triplet color-plaquette glueball correlator."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    mc_time_index: int | None = None
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    eps: float = 1e-12
    use_action_form: bool = False


@dataclass
class GlueballColorCorrelatorOutput:
    """Glueball color correlator output and diagnostics."""

    correlator: Tensor
    correlator_raw: Tensor
    correlator_connected: Tensor
    counts: Tensor
    frame_indices: list[int]
    triplet_counts_per_frame: Tensor
    disconnected_contribution: float
    mean_glueball: float
    n_valid_source_triplets: int
    operator_glueball_series: Tensor


def _compute_color_plaquette_for_triplets(
    color: Tensor,
    color_valid: Tensor,
    alive: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Compute Π_i for companion triplets and a validity mask.

    Returns:
        pi: Complex color plaquette Π_i [T, N].
        valid: Valid source/sink triplet mask [T, N].
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T, N], got {tuple(alive.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T, N] aligned with color."
        raise ValueError(msg)

    _, companion_j, companion_k, structural_valid = build_companion_triplets(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
    )

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


def compute_glueball_color_correlator_from_color(
    color: Tensor,
    color_valid: Tensor,
    alive: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_lag: int = 80,
    use_connected: bool = True,
    eps: float = 1e-12,
    use_action_form: bool = False,
    frame_indices: list[int] | None = None,
) -> GlueballColorCorrelatorOutput:
    """Compute companion-triplet glueball correlator from precomputed color states."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
    if alive.shape != color.shape[:2]:
        raise ValueError(f"alive must have shape [T, N], got {tuple(alive.shape)}.")
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T, N] aligned with color."
        raise ValueError(msg)

    t_total = int(color.shape[0])
    max_lag = max(0, int(max_lag))
    effective_lag = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1
    device = color.device

    if t_total == 0:
        empty_f = torch.zeros(n_lags, dtype=torch.float32, device=device)
        empty_i = torch.zeros(n_lags, dtype=torch.int64, device=device)
        empty_t = torch.zeros(0, dtype=torch.int64, device=device)
        return GlueballColorCorrelatorOutput(
            correlator=empty_f,
            correlator_raw=empty_f.clone(),
            correlator_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[] if frame_indices is None else frame_indices,
            triplet_counts_per_frame=empty_t,
            disconnected_contribution=0.0,
            mean_glueball=0.0,
            n_valid_source_triplets=0,
            operator_glueball_series=empty_t.float(),
        )

    source_pi, source_valid = _compute_color_plaquette_for_triplets(
        color=color,
        color_valid=color_valid,
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        eps=eps,
    )
    source_obs = source_pi.real.float()
    if use_action_form:
        source_obs = 1.0 - source_obs

    triplet_counts_per_frame = source_valid.sum(dim=1).to(torch.int64)
    n_valid_source_triplets = int(source_valid.sum().item())

    operator_glueball_series = torch.zeros(t_total, dtype=torch.float32, device=device)
    valid_t = triplet_counts_per_frame > 0
    if torch.any(valid_t):
        weight = source_valid.to(dtype=torch.float32)
        sums = (source_obs * weight).sum(dim=1)
        operator_glueball_series[valid_t] = sums[valid_t] / triplet_counts_per_frame[valid_t].to(
            torch.float32
        )

    if n_valid_source_triplets > 0:
        mean_glueball_t = source_obs[source_valid].mean()
    else:
        mean_glueball_t = torch.zeros((), dtype=torch.float32, device=device)
    disconnected = float((mean_glueball_t * mean_glueball_t).item())

    correlator_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    correlator_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.int64, device=device)

    for lag in range(effective_lag + 1):
        source_len = t_total - lag
        sink_pi, sink_valid = _compute_color_plaquette_for_triplets(
            color=color[lag : lag + source_len],
            color_valid=color_valid[lag : lag + source_len],
            alive=alive[lag : lag + source_len],
            companions_distance=companions_distance[:source_len],
            companions_clone=companions_clone[:source_len],
            eps=eps,
        )
        sink_obs = sink_pi.real.float()
        if use_action_form:
            sink_obs = 1.0 - sink_obs

        valid_pair = source_valid[:source_len] & sink_valid
        count = int(valid_pair.sum().item())
        counts[lag] = count
        if count == 0:
            continue

        raw_prod = source_obs[:source_len] * sink_obs
        correlator_raw[lag] = raw_prod[valid_pair].mean().float()

        conn_prod = (source_obs[:source_len] - mean_glueball_t) * (sink_obs - mean_glueball_t)
        correlator_connected[lag] = conn_prod[valid_pair].mean().float()

    correlator = correlator_connected if use_connected else correlator_raw
    return GlueballColorCorrelatorOutput(
        correlator=correlator,
        correlator_raw=correlator_raw,
        correlator_connected=correlator_connected,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
        triplet_counts_per_frame=triplet_counts_per_frame,
        disconnected_contribution=disconnected,
        mean_glueball=float(mean_glueball_t.item()),
        n_valid_source_triplets=n_valid_source_triplets,
        operator_glueball_series=operator_glueball_series,
    )


def compute_companion_glueball_color_correlator(
    history: RunHistory,
    config: GlueballColorCorrelatorConfig | None = None,
) -> GlueballColorCorrelatorOutput:
    """Compute vectorized companion-triplet glueball correlator from RunHistory."""
    config = config or GlueballColorCorrelatorConfig()
    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
        mc_time_index=config.mc_time_index,
    )
    n_lags = int(max(0, config.max_lag)) + 1
    if not frame_indices:
        empty_f = torch.zeros(n_lags, dtype=torch.float32)
        empty_i = torch.zeros(n_lags, dtype=torch.int64)
        empty_t = torch.zeros(0, dtype=torch.int64)
        return GlueballColorCorrelatorOutput(
            correlator=empty_f,
            correlator_raw=empty_f.clone(),
            correlator_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[],
            triplet_counts_per_frame=empty_t,
            disconnected_contribution=0.0,
            mean_glueball=0.0,
            n_valid_source_triplets=0,
            operator_glueball_series=empty_t.float(),
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
    dims = _resolve_3d_dims(color.shape[-1], config.color_dims, "color_dims")
    color = color[:, :, list(dims)]

    device = color.device
    alive = torch.as_tensor(
        history.alive_mask[start_idx - 1 : end_idx - 1], dtype=torch.bool, device=device
    )
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

    return compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        alive=alive,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        eps=float(max(config.eps, 0.0)),
        use_action_form=bool(config.use_action_form),
        frame_indices=frame_indices,
    )
