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
import math

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils import (
    _fft_correlator_batched,
    build_companion_triplets,
    resolve_3d_dims,
    resolve_frame_indices,
    safe_gather_2d,
    safe_gather_3d,
)
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0


@dataclass
class GlueballColorCorrelatorConfig:
    """Configuration for companion-triplet color-plaquette glueball correlator."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    eps: float = 1e-12
    operator_mode: str | None = None
    use_action_form: bool = False
    use_momentum_projection: bool = False
    momentum_axis: int = 0
    momentum_mode_max: int = 3
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


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
    momentum_modes: Tensor | None = None
    momentum_correlator: Tensor | None = None
    momentum_correlator_raw: Tensor | None = None
    momentum_correlator_connected: Tensor | None = None
    momentum_correlator_err: Tensor | None = None
    momentum_operator_cos_series: Tensor | None = None
    momentum_operator_sin_series: Tensor | None = None
    momentum_axis: int | None = None
    momentum_length_scale: float | None = None
    momentum_valid_frames: int = 0


def _resolve_glueball_operator_mode(
    *,
    operator_mode: str | None,
    use_action_form: bool,
) -> str:
    """Resolve glueball operator mode with backward compatibility."""
    if operator_mode is None or not str(operator_mode).strip():
        return "action_re_plaquette" if use_action_form else "re_plaquette"
    mode = str(operator_mode).strip().lower()
    if mode == "action":
        return "action_re_plaquette"
    return mode


def _glueball_observable_from_plaquette(pi: Tensor, *, operator_mode: str) -> Tensor:
    """Build scalar glueball observable from complex plaquette Π."""
    mode = _resolve_glueball_operator_mode(operator_mode=operator_mode, use_action_form=False)
    if mode == "re_plaquette":
        return pi.real.float()
    if mode == "action_re_plaquette":
        return (1.0 - pi.real).float()
    if mode == "phase_action":
        phase = torch.angle(pi)
        return (1.0 - torch.cos(phase)).float()
    if mode == "phase_sin2":
        phase = torch.angle(pi)
        return torch.sin(phase).square().float()
    msg = (
        "Invalid glueball operator_mode. Expected one of "
        "{'re_plaquette','action_re_plaquette','phase_action','phase_sin2'}."
    )
    raise ValueError(msg)


def _compute_color_plaquette_for_triplets(
    color: Tensor,
    color_valid: Tensor,
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
    if companions_distance.shape != color.shape[:2] or companions_clone.shape != color.shape[:2]:
        msg = "companion arrays must have shape [T, N] aligned with color."
        raise ValueError(msg)

    _, companion_j, companion_k, structural_valid = build_companion_triplets(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
    )

    color_j, in_j = safe_gather_3d(color, companion_j)
    color_k, in_k = safe_gather_3d(color, companion_k)
    valid_j, _ = safe_gather_2d(color_valid, companion_j)
    valid_k, _ = safe_gather_2d(color_valid, companion_k)

    z_ij = (torch.conj(color) * color_j).sum(dim=-1)
    z_jk = (torch.conj(color_j) * color_k).sum(dim=-1)
    z_ki = (torch.conj(color_k) * color).sum(dim=-1)
    pi = z_ij * z_jk * z_ki

    finite = torch.isfinite(pi.real) & torch.isfinite(pi.imag)
    valid = structural_valid & in_j & in_k & color_valid & valid_j & valid_k & finite
    if eps > 0:
        valid = valid & (z_ij.abs() > eps) & (z_jk.abs() > eps) & (z_ki.abs() > eps)

    pi = torch.where(valid, pi, torch.zeros_like(pi))
    return pi, valid


def _resolve_positive_length(
    *,
    positions_axis: Tensor,
    box_length: float | None,
) -> float:
    """Resolve a positive projection length scale for Fourier modes."""
    if box_length is not None and float(box_length) > 0:
        return float(box_length)

    span = float((positions_axis.max() - positions_axis.min()).abs().item())
    return max(span, 1.0)


def _extract_axis_bounds(
    bounds: object | None,
    axis: int,
    *,
    device: torch.device | None = None,
) -> tuple[float | None, float | None]:
    """Extract (low, high) for one axis from TorchBounds or array-like bounds."""
    if bounds is None or axis < 0:
        return None, None

    try:
        if hasattr(bounds, "low") and hasattr(bounds, "high"):
            low_vec = torch.as_tensor(bounds.low, dtype=torch.float32, device=device).flatten()
            high_vec = torch.as_tensor(bounds.high, dtype=torch.float32, device=device).flatten()
            if axis < int(low_vec.numel()) and axis < int(high_vec.numel()):
                return float(low_vec[axis].item()), float(high_vec[axis].item())
            return None, None

        bounds_t = torch.as_tensor(bounds, dtype=torch.float32, device=device)
    except Exception:
        return None, None

    if bounds_t.ndim != 2:
        return None, None
    if bounds_t.shape[0] > axis and bounds_t.shape[1] >= 2:
        return float(bounds_t[axis, 0].item()), float(bounds_t[axis, 1].item())
    if bounds_t.shape[0] >= 2 and bounds_t.shape[1] > axis:
        return float(bounds_t[0, axis].item()), float(bounds_t[1, axis].item())
    return None, None


def _compute_momentum_projected_correlators(
    *,
    source_obs: Tensor,
    source_valid: Tensor,
    positions_axis: Tensor,
    max_lag: int,
    use_connected: bool,
    mode_max: int,
    box_length: float,
    compute_bootstrap_errors: bool,
    n_bootstrap: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, int]:
    """Compute batched momentum-projected glueball correlators C_p(Δt)."""
    if source_obs.shape != source_valid.shape or source_obs.shape != positions_axis.shape:
        msg = (
            "source_obs, source_valid, and positions_axis must share shape [T, N], got "
            f"{tuple(source_obs.shape)}, {tuple(source_valid.shape)}, {tuple(positions_axis.shape)}."
        )
        raise ValueError(msg)
    if box_length <= 0:
        msg = "box_length must be positive."
        raise ValueError(msg)

    t_total, _ = source_obs.shape
    n_modes = max(0, int(mode_max)) + 1
    device = source_obs.device

    modes = torch.arange(n_modes, device=device, dtype=torch.float32)
    k_values = (2.0 * torch.pi / float(box_length)) * modes

    phase_arg = k_values[:, None, None] * positions_axis[None, :, :].float()
    cos_phase = torch.cos(phase_arg)
    sin_phase = torch.sin(phase_arg)

    weights = source_valid.to(dtype=torch.float32)
    counts_t = weights.sum(dim=1)
    valid_t = counts_t > 0
    n_valid_frames = int(valid_t.sum().item())

    op_cos = torch.zeros((n_modes, t_total), dtype=torch.float32, device=device)
    op_sin = torch.zeros((n_modes, t_total), dtype=torch.float32, device=device)
    if n_valid_frames > 0:
        weighted_obs = source_obs.float() * weights
        cos_num = (weighted_obs[None, :, :] * cos_phase).sum(dim=2)
        sin_num = (weighted_obs[None, :, :] * sin_phase).sum(dim=2)
        denom = counts_t[valid_t].to(dtype=torch.float32).clamp(min=1.0)
        op_cos[:, valid_t] = cos_num[:, valid_t] / denom.unsqueeze(0)
        op_sin[:, valid_t] = sin_num[:, valid_t] / denom.unsqueeze(0)

    if n_valid_frames == 0:
        n_lags = int(max(0, max_lag)) + 1
        zeros = torch.zeros((n_modes, n_lags), dtype=torch.float32, device=device)
        return modes, zeros, zeros.clone(), zeros.clone(), op_cos, op_sin, None, 0

    series_cos = op_cos[:, valid_t]
    series_sin = op_sin[:, valid_t]

    corr_raw = _fft_correlator_batched(
        series_cos,
        max_lag=int(max_lag),
        use_connected=False,
    ) + _fft_correlator_batched(
        series_sin,
        max_lag=int(max_lag),
        use_connected=False,
    )
    corr_connected = _fft_correlator_batched(
        series_cos,
        max_lag=int(max_lag),
        use_connected=True,
    ) + _fft_correlator_batched(
        series_sin,
        max_lag=int(max_lag),
        use_connected=True,
    )
    correlator = corr_connected if use_connected else corr_raw

    correlator_err: Tensor | None = None
    if bool(compute_bootstrap_errors):
        n_boot = max(1, int(n_bootstrap))
        t_len = int(series_cos.shape[1])
        idx = torch.randint(0, t_len, (n_boot, t_len), device=device)
        idx_expand = idx[:, None, :].expand(n_boot, n_modes, t_len)

        boot_cos = torch.gather(
            series_cos.unsqueeze(0).expand(n_boot, n_modes, t_len),
            dim=2,
            index=idx_expand,
        )
        boot_sin = torch.gather(
            series_sin.unsqueeze(0).expand(n_boot, n_modes, t_len),
            dim=2,
            index=idx_expand,
        )

        boot_cos_flat = boot_cos.reshape(n_boot * n_modes, t_len)
        boot_sin_flat = boot_sin.reshape(n_boot * n_modes, t_len)

        boot_raw = (
            _fft_correlator_batched(boot_cos_flat, max_lag=int(max_lag), use_connected=False)
            + _fft_correlator_batched(boot_sin_flat, max_lag=int(max_lag), use_connected=False)
        ).reshape(n_boot, n_modes, -1)
        boot_connected = (
            _fft_correlator_batched(boot_cos_flat, max_lag=int(max_lag), use_connected=True)
            + _fft_correlator_batched(boot_sin_flat, max_lag=int(max_lag), use_connected=True)
        ).reshape(n_boot, n_modes, -1)
        boot_corr = boot_connected if use_connected else boot_raw
        correlator_err = boot_corr.std(dim=0)

    return (
        modes,
        correlator.float(),
        corr_raw.float(),
        corr_connected.float(),
        op_cos,
        op_sin,
        correlator_err.float() if correlator_err is not None else None,
        n_valid_frames,
    )


def compute_glueball_color_correlator_from_color(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    max_lag: int = 80,
    use_connected: bool = True,
    eps: float = 1e-12,
    operator_mode: str | None = None,
    use_action_form: bool = False,
    frame_indices: list[int] | None = None,
    positions_axis: Tensor | None = None,
    momentum_axis: int | None = None,
    use_momentum_projection: bool = False,
    momentum_mode_max: int = 3,
    projection_length: float | None = None,
    compute_bootstrap_errors: bool = False,
    n_bootstrap: int = 100,
) -> GlueballColorCorrelatorOutput:
    """Compute companion-triplet glueball correlator from precomputed color states."""
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")
    if color_valid.shape != color.shape[:2]:
        raise ValueError(f"color_valid must have shape [T, N], got {tuple(color_valid.shape)}.")
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
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        eps=eps,
    )
    resolved_operator_mode = _resolve_glueball_operator_mode(
        operator_mode=operator_mode,
        use_action_form=use_action_form,
    )
    source_obs = _glueball_observable_from_plaquette(
        source_pi,
        operator_mode=resolved_operator_mode,
    )

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
            companions_distance=companions_distance[:source_len],
            companions_clone=companions_clone[:source_len],
            eps=eps,
        )
        sink_obs = _glueball_observable_from_plaquette(
            sink_pi,
            operator_mode=resolved_operator_mode,
        )

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

    momentum_modes = None
    momentum_correlator = None
    momentum_correlator_raw = None
    momentum_correlator_connected = None
    momentum_correlator_err = None
    momentum_operator_cos_series = None
    momentum_operator_sin_series = None
    momentum_valid_frames = 0
    momentum_length_scale = None

    if use_momentum_projection:
        if positions_axis is None:
            msg = "positions_axis is required when use_momentum_projection=True."
            raise ValueError(msg)
        if positions_axis.shape != source_obs.shape:
            msg = (
                "positions_axis must match source_obs shape [T, N], got "
                f"{tuple(positions_axis.shape)} vs {tuple(source_obs.shape)}."
            )
            raise ValueError(msg)

        resolved_length = _resolve_positive_length(
            positions_axis=positions_axis,
            box_length=projection_length,
        )
        (
            momentum_modes,
            momentum_correlator,
            momentum_correlator_raw,
            momentum_correlator_connected,
            momentum_operator_cos_series,
            momentum_operator_sin_series,
            momentum_correlator_err,
            momentum_valid_frames,
        ) = _compute_momentum_projected_correlators(
            source_obs=source_obs,
            source_valid=source_valid,
            positions_axis=positions_axis,
            max_lag=max_lag,
            use_connected=use_connected,
            mode_max=momentum_mode_max,
            box_length=resolved_length,
            compute_bootstrap_errors=compute_bootstrap_errors,
            n_bootstrap=n_bootstrap,
        )
        momentum_length_scale = float(resolved_length)

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
        momentum_modes=momentum_modes,
        momentum_correlator=momentum_correlator,
        momentum_correlator_raw=momentum_correlator_raw,
        momentum_correlator_connected=momentum_correlator_connected,
        momentum_correlator_err=momentum_correlator_err,
        momentum_operator_cos_series=momentum_operator_cos_series,
        momentum_operator_sin_series=momentum_operator_sin_series,
        momentum_axis=momentum_axis,
        momentum_length_scale=momentum_length_scale,
        momentum_valid_frames=int(momentum_valid_frames),
    )


def compute_companion_glueball_color_correlator(
    history: RunHistory,
    config: GlueballColorCorrelatorConfig | None = None,
) -> GlueballColorCorrelatorOutput:
    """Compute vectorized companion-triplet glueball correlator from RunHistory."""
    config = config or GlueballColorCorrelatorConfig()
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

    use_momentum_projection = bool(config.use_momentum_projection)
    momentum_axis = int(config.momentum_axis)
    positions_axis: Tensor | None = None
    projection_length: float | None = None
    if use_momentum_projection:
        if momentum_axis < 0 or momentum_axis >= int(history.d):
            msg = (
                f"momentum_axis={momentum_axis} out of range for history.d={history.d}. "
                f"Expected 0..{history.d - 1}."
            )
            raise ValueError(msg)
        positions_axis = history.x_before_clone[start_idx:end_idx, :, momentum_axis].to(
            device=device, dtype=torch.float32
        )

        low, high = _extract_axis_bounds(history.bounds, momentum_axis, device=device)
        if (
            low is not None
            and high is not None
            and math.isfinite(low)
            and math.isfinite(high)
            and high > low
        ):
            projection_length = float(high - low)

    return compute_glueball_color_correlator_from_color(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        eps=float(max(config.eps, 0.0)),
        operator_mode=config.operator_mode,
        use_action_form=bool(config.use_action_form),
        frame_indices=frame_indices,
        positions_axis=positions_axis,
        momentum_axis=momentum_axis if use_momentum_projection else None,
        use_momentum_projection=use_momentum_projection,
        momentum_mode_max=int(config.momentum_mode_max),
        projection_length=projection_length,
        compute_bootstrap_errors=bool(config.compute_bootstrap_errors),
        n_bootstrap=int(config.n_bootstrap),
    )
