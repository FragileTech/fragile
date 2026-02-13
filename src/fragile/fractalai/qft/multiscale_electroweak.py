"""Multiscale SU(2) electroweak channel estimation with vectorized PyTorch kernels.

This module mirrors the companion strong-force multiscale workflow for SU(2)
electroweak channels:
- shared scale calibration from recorded neighbor graph distances,
- per-scale correlator extraction,
- best-scale selection under GEVP-quality filters,
- optional vectorized time-bootstrap uncertainty estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.baryon_triplet_channels import _resolve_frame_indices
from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    ChannelCorrelatorResult,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.electroweak_observables import classify_walker_types
from fragile.fractalai.qft.smeared_operators import (
    iter_smeared_kernel_batches_from_history,
    select_interesting_scales_from_history,
)


SU2_BASE_CHANNELS = (
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
)
SU2_DIRECTIONAL_CHANNELS = tuple(f"{name}_directed" for name in SU2_BASE_CHANNELS)
SU2_WALKER_TYPE_CHANNELS = tuple(
    f"{name}_{walker_type}"
    for name in SU2_BASE_CHANNELS
    for walker_type in ("cloner", "resister", "persister")
)
SUPPORTED_CHANNELS = SU2_BASE_CHANNELS + SU2_DIRECTIONAL_CHANNELS + SU2_WALKER_TYPE_CHANNELS
SU2_COMPANION_CHANNEL_MAP: dict[str, str] = {
    name: f"{name}_companion" for name in SUPPORTED_CHANNELS
}
BOOTSTRAP_MODES = ("time", "walker", "hybrid")
KERNEL_TYPES = ("gaussian", "exponential", "tophat", "shell")


@dataclass
class MultiscaleElectroweakConfig:
    """Configuration for multiscale SU(2) electroweak analysis."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    mc_time_index: int | None = None
    h_eff: float = 1.0
    epsilon_c: float | None = None
    epsilon_clone: float | None = None
    lambda_alg: float | None = None
    edge_weight_mode: str = "inverse_riemannian_distance"
    su2_operator_mode: str = "standard"
    enable_walker_type_split: bool = False
    walker_type_scope: str = "frame_global"

    # Scale/kernel controls
    n_scales: int = 8
    kernel_type: str = "gaussian"
    kernel_distance_method: str = "auto"
    kernel_assume_all_alive: bool = True
    kernel_batch_size: int = 1
    kernel_scale_frames: int = 8
    kernel_scale_q_low: float = 0.05
    kernel_scale_q_high: float = 0.95
    kernel_max_scale_samples: int = 500_000
    kernel_min_scale: float = 1e-6

    # Correlator/fitting controls
    max_lag: int = 80
    use_connected: bool = True
    fit_mode: str = "aic"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2
    window_widths: list[int] | None = None

    # Best-scale quality filters (kept aligned with companion GEVP defaults)
    best_min_r2: float = 0.5
    best_min_windows: int = 10
    best_max_error_pct: float = 30.0
    best_remove_artifacts: bool = True

    # Bootstrap controls
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100
    bootstrap_seed: int = 12345
    bootstrap_mode: str = "time"


@dataclass
class MultiscaleElectroweakOutput:
    """Output payload for multiscale SU(2) electroweak analysis."""

    scales: Tensor
    frame_indices: list[int]
    per_scale_results: dict[str, list[ChannelCorrelatorResult]]
    best_results: dict[str, ChannelCorrelatorResult]
    best_scale_index: dict[str, int]
    series_by_channel: dict[str, Tensor]  # [S, T]
    bootstrap_mode_applied: str
    notes: list[str]
    bootstrap_mass_std: dict[str, Tensor] | None = None  # [S]


def _nanstd_compat(values: Tensor, *, dim: int) -> Tensor:
    if hasattr(torch, "nanstd"):
        return torch.nanstd(values, dim=dim)
    finite = torch.isfinite(values)
    count = finite.sum(dim=dim)
    count_f = count.to(dtype=values.dtype).clamp(min=1)
    safe = torch.where(finite, values, torch.zeros_like(values))
    mean = safe.sum(dim=dim) / count_f
    centered = torch.where(finite, values - mean.unsqueeze(dim), torch.zeros_like(values))
    var = centered.square().sum(dim=dim) / count_f
    std = torch.sqrt(torch.clamp_min(var, 0.0))
    nan_fill = torch.full_like(std, float("nan"))
    return torch.where(count > 0, std, nan_fill)


def _nested_param(
    params: dict[str, Any] | None, *keys: str, default: float | None
) -> float | None:
    if params is None:
        return default
    current: Any = params
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _resolve_electroweak_params(
    history: RunHistory,
    cfg: MultiscaleElectroweakConfig,
) -> tuple[float, float, float]:
    params = history.params if isinstance(history.params, dict) else None

    epsilon_c = cfg.epsilon_c
    if epsilon_c is None:
        epsilon_c = _nested_param(params, "companion_selection_clone", "epsilon", default=None)
    if epsilon_c is None:
        epsilon_c = _nested_param(params, "companion_selection", "epsilon", default=None)
    if epsilon_c is None:
        epsilon_c = 1.0

    epsilon_clone = cfg.epsilon_clone
    if epsilon_clone is None:
        epsilon_clone = _nested_param(params, "cloning", "epsilon_clone", default=1e-8)
    if epsilon_clone is None:
        epsilon_clone = 1e-8

    lambda_alg = cfg.lambda_alg
    if lambda_alg is None:
        lambda_alg = _nested_param(params, "companion_selection", "lambda_alg", default=None)
    if lambda_alg is None:
        lambda_alg = _nested_param(params, "fitness", "lambda_alg", default=0.0)
    if lambda_alg is None:
        lambda_alg = 0.0

    return (
        float(max(epsilon_c, 1e-8)),
        float(max(epsilon_clone, 1e-8)),
        float(max(lambda_alg, 0.0)),
    )


def _resolve_su2_operator_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"standard", "score_directed"}:
        msg = "su2_operator_mode must be 'standard' or 'score_directed'."
        raise ValueError(msg)
    return mode_norm


def _resolve_walker_type_scope(scope: str) -> str:
    scope_norm = str(scope).strip().lower()
    if scope_norm != "frame_global":
        msg = "walker_type_scope must be 'frame_global'."
        raise ValueError(msg)
    return scope_norm


def _build_result_from_precomputed_correlator(
    *,
    channel_name: str,
    correlator: Tensor,
    dt: float,
    config: CorrelatorConfig,
    n_samples: int,
    series: Tensor,
    correlator_err: Tensor | None,
) -> ChannelCorrelatorResult:
    corr_t = correlator.float()
    effective_mass = compute_effective_mass_torch(corr_t, dt)
    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(corr_t.abs(), dt, config)
        window_data: dict[str, Any] = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(corr_t, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(corr_t, dt, config)
        window_data = {
            "window_masses": mass_fit.pop("window_masses", None),
            "window_aic": mass_fit.pop("window_aic", None),
            "window_widths": mass_fit.pop("window_widths", None),
            "window_r2": mass_fit.pop("window_r2", None),
        }
    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=corr_t,
        correlator_err=correlator_err.float() if correlator_err is not None else None,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series.float(),
        n_samples=int(n_samples),
        dt=dt,
        **window_data,
    )


def _fit_mass_only(correlator: Tensor, dt: float, config: CorrelatorConfig) -> float:
    corr_t = correlator.float()
    if config.fit_mode == "linear_abs":
        fit = extract_mass_linear(corr_t.abs(), dt, config)
    elif config.fit_mode == "linear":
        fit = extract_mass_linear(corr_t, dt, config)
    else:
        fit = extract_mass_aic(corr_t, dt, config)
    mass = float(fit.get("mass", 0.0))
    if not math.isfinite(mass) or mass <= 0:
        return float("nan")
    return mass


def _extract_n_valid_windows(result: ChannelCorrelatorResult) -> int:
    fit = result.mass_fit or {}
    raw = fit.get("n_valid_windows", None)
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass

    window_masses = getattr(result, "window_masses", None)
    if isinstance(window_masses, Tensor):
        if int(window_masses.numel()) <= 0:
            return 0
        return int(torch.isfinite(window_masses).sum().item())
    if isinstance(window_masses, list | tuple):
        count = 0
        for value in window_masses:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                count += 1
        return count
    return 0


def _passes_best_scale_filters(
    result: ChannelCorrelatorResult,
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
) -> bool:
    fit = result.mass_fit or {}
    mass = float(fit.get("mass", float("nan")))
    if not math.isfinite(mass) or mass <= 0:
        return False

    r2 = float(fit.get("r_squared", float("nan")))
    if math.isfinite(min_r2) and (not math.isfinite(r2) or r2 < min_r2):
        return False

    n_windows = _extract_n_valid_windows(result)
    if n_windows < max(0, int(min_windows)):
        return False

    mass_err = float(fit.get("mass_error", float("nan")))
    if math.isfinite(max_error_pct) and max_error_pct >= 0:
        if math.isfinite(mass_err) and mass_err >= 0 and mass > 0:
            err_pct = abs(mass_err / mass) * 100.0
        else:
            err_pct = float("inf")
        if err_pct > max_error_pct:
            return False

    if remove_artifacts:
        if not math.isfinite(mass_err):
            return False
        if mass_err == 0.0:
            return False
        if mass == 0.0:
            return False
    return True


def _select_best_scale(
    results: list[ChannelCorrelatorResult],
    *,
    min_r2: float = 0.5,
    min_windows: int = 10,
    max_error_pct: float = 30.0,
    remove_artifacts: bool = True,
) -> int | None:
    best_idx: int | None = None
    best_key = (float("inf"), float("inf"), float("inf"))
    for idx, result in enumerate(results):
        if not _passes_best_scale_filters(
            result,
            min_r2=min_r2,
            min_windows=min_windows,
            max_error_pct=max_error_pct,
            remove_artifacts=remove_artifacts,
        ):
            continue
        fit = result.mass_fit or {}
        best_window = (
            fit.get("best_window", {}) if isinstance(fit.get("best_window", {}), dict) else {}
        )
        aic = float(best_window.get("aic", float("inf")))
        if not math.isfinite(aic):
            aic = float("inf")
        r2 = float(fit.get("r_squared", float("nan")))
        r2_penalty = -r2 if math.isfinite(r2) else float("inf")
        mass_err = float(fit.get("mass_error", float("inf")))
        if not math.isfinite(mass_err):
            mass_err = float("inf")
        key = (aic, r2_penalty, mass_err)
        if key < best_key:
            best_key = key
            best_idx = idx
    return best_idx


def _compute_su2_series_from_kernels(
    *,
    positions: Tensor,  # [T,N,d]
    velocities: Tensor,  # [T,N,d]
    fitness: Tensor,  # [T,N]
    alive: Tensor,  # [T,N]
    kernels: Tensor,  # [T,S,N,N]
    h_eff: float,
    epsilon_c: float,
    epsilon_clone: float,
    lambda_alg: float,
    bounds: Any | None,
    pbc: bool,
    will_clone: Tensor | None = None,  # [T,N] bool
    su2_operator_mode: str = "standard",
    enable_walker_type_split: bool = False,
    walker_type_scope: str = "frame_global",
) -> dict[str, Tensor]:
    if kernels.ndim != 4:
        raise ValueError(f"kernels must have shape [T,S,N,N], got {tuple(kernels.shape)}.")
    resolved_mode = _resolve_su2_operator_mode(su2_operator_mode)
    _resolve_walker_type_scope(walker_type_scope)
    t_len, n_scales, n_walkers, _ = kernels.shape
    if t_len <= 0 or n_scales <= 0 or n_walkers <= 0:
        empty = torch.zeros((n_scales, t_len), device=kernels.device, dtype=torch.float32)
        return {
            alias: empty.clone()
            for alias in (SU2_COMPANION_CHANNEL_MAP[name] for name in SUPPORTED_CHANNELS)
        }

    dev = kernels.device
    real_dtype = kernels.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    alive_mask = alive.to(device=dev, dtype=torch.bool)
    alive_f = alive_mask.to(dtype=real_dtype)

    # Row-normalized kernel weights restricted to alive->alive pairs.
    pair_valid = alive_mask[:, None, :, None] & alive_mask[:, None, None, :]
    weights = torch.where(pair_valid, kernels, torch.zeros_like(kernels))
    row_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(
        row_sum > 0, weights / row_sum.clamp(min=1e-12), torch.zeros_like(weights)
    )

    fitness_t = fitness.to(device=dev, dtype=real_dtype)
    fi = fitness_t[:, None, :, None]
    fj = fitness_t[:, None, None, :]
    denom = fi + float(epsilon_clone)
    denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, float(epsilon_clone)), denom)

    su2_phase = ((fj - fi) / denom) / max(float(h_eff), 1e-12)
    weights_c = weights.to(dtype=complex_dtype)
    su2_pair = torch.exp(1j * su2_phase.to(complex_dtype))
    su2_pair_directed = torch.where(su2_phase >= 0, su2_pair, torch.conj(su2_pair))
    su2_phase_exp_standard = (weights_c * su2_pair).sum(dim=-1)
    su2_phase_exp_directed = (weights_c * su2_pair_directed).sum(dim=-1)

    diff_x = positions[:, None, :, None, :] - positions[:, None, None, :, :]
    if pbc and bounds is not None:
        high = bounds.high.to(diff_x)
        low = bounds.low.to(diff_x)
        span = high - low
        diff_x = diff_x - span * torch.round(diff_x / span)
    diff_v = velocities[:, None, :, None, :] - velocities[:, None, None, :, :]
    dist_sq = (diff_x.square()).sum(dim=-1) + float(lambda_alg) * (diff_v.square()).sum(dim=-1)
    su2_weight = torch.exp(-dist_sq / (2.0 * max(float(epsilon_c), 1e-12) ** 2))
    su2_amp = (weights * torch.sqrt(su2_weight)).sum(dim=-1)

    su2_comp_phase_exp_standard = torch.einsum("tsij,tsj->tsi", weights_c, su2_phase_exp_standard)
    su2_comp_phase_exp_directed = torch.einsum("tsij,tsj->tsi", weights_c, su2_phase_exp_directed)
    su2_comp_amp = torch.einsum("tsij,tsj->tsi", weights, su2_amp)

    alive_c = alive_f[:, None, :].to(dtype=complex_dtype)
    su2_amp_c = su2_amp.to(dtype=complex_dtype)
    su2_comp_amp_c = su2_comp_amp.to(dtype=complex_dtype)

    su2_phase_op_standard = su2_phase_exp_standard * alive_c
    su2_component_op_standard = (su2_amp_c * su2_phase_exp_standard) * alive_c
    su2_doublet_op_standard = (
        su2_amp_c * su2_phase_exp_standard + su2_comp_amp_c * su2_comp_phase_exp_standard
    ) * alive_c
    su2_doublet_diff_op_standard = (
        su2_amp_c * su2_phase_exp_standard - su2_comp_amp_c * su2_comp_phase_exp_standard
    ) * alive_c

    su2_phase_op_directed = su2_phase_exp_directed * alive_c
    su2_component_op_directed = (su2_amp_c * su2_phase_exp_directed) * alive_c
    su2_doublet_op_directed = (
        su2_amp_c * su2_phase_exp_directed + su2_comp_amp_c * su2_comp_phase_exp_directed
    ) * alive_c
    su2_doublet_diff_op_directed = (
        su2_amp_c * su2_phase_exp_directed - su2_comp_amp_c * su2_comp_phase_exp_directed
    ) * alive_c

    if resolved_mode == "score_directed":
        su2_phase_op = su2_phase_op_directed
        su2_component_op = su2_component_op_directed
        su2_doublet_op = su2_doublet_op_directed
        su2_doublet_diff_op = su2_doublet_diff_op_directed
    else:
        su2_phase_op = su2_phase_op_standard
        su2_component_op = su2_component_op_standard
        su2_doublet_op = su2_doublet_op_standard
        su2_doublet_diff_op = su2_doublet_diff_op_standard

    counts = alive_f.sum(dim=-1).clamp(min=1.0)[:, None]

    def _series_from_op(op: Tensor) -> Tensor:
        return ((op.real * alive_f[:, None, :]).sum(dim=-1) / counts).transpose(0, 1).contiguous()

    outputs = {
        SU2_COMPANION_CHANNEL_MAP["su2_phase"]: _series_from_op(su2_phase_op),
        SU2_COMPANION_CHANNEL_MAP["su2_component"]: _series_from_op(su2_component_op),
        SU2_COMPANION_CHANNEL_MAP["su2_doublet"]: _series_from_op(su2_doublet_op),
        SU2_COMPANION_CHANNEL_MAP["su2_doublet_diff"]: _series_from_op(su2_doublet_diff_op),
        SU2_COMPANION_CHANNEL_MAP["su2_phase_directed"]: _series_from_op(su2_phase_op_directed),
        SU2_COMPANION_CHANNEL_MAP["su2_component_directed"]: _series_from_op(
            su2_component_op_directed
        ),
        SU2_COMPANION_CHANNEL_MAP["su2_doublet_directed"]: _series_from_op(
            su2_doublet_op_directed
        ),
        SU2_COMPANION_CHANNEL_MAP["su2_doublet_diff_directed"]: _series_from_op(
            su2_doublet_diff_op_directed
        ),
    }

    if bool(enable_walker_type_split):
        if will_clone is not None:
            will_clone_b = will_clone.to(device=dev, dtype=torch.bool)
            if will_clone_b.shape != alive_mask.shape:
                msg = (
                    "will_clone must align with alive [T,N] when walker-type split "
                    f"is enabled, got {tuple(will_clone_b.shape)} vs {tuple(alive_mask.shape)}."
                )
                raise ValueError(msg)
        else:
            will_clone_b = torch.zeros_like(alive_mask)

        cloner_mask = torch.zeros_like(alive_mask)
        resister_mask = torch.zeros_like(alive_mask)
        persister_mask = torch.zeros_like(alive_mask)
        for t_idx in range(t_len):
            c_mask, r_mask, p_mask = classify_walker_types(
                fitness=fitness_t[t_idx],
                alive=alive_mask[t_idx],
                will_clone=will_clone_b[t_idx],
            )
            cloner_mask[t_idx] = c_mask
            resister_mask[t_idx] = r_mask
            persister_mask[t_idx] = p_mask
    else:
        cloner_mask = torch.zeros_like(alive_mask)
        resister_mask = torch.zeros_like(alive_mask)
        persister_mask = torch.zeros_like(alive_mask)

    def _series_from_masked_op(op: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.to(dtype=real_dtype)
        counts_mask = mask_f.sum(dim=-1).clamp(min=1.0)[:, None]
        return (
            ((op.real * mask_f[:, None, :]).sum(dim=-1) / counts_mask).transpose(0, 1).contiguous()
        )

    outputs[SU2_COMPANION_CHANNEL_MAP["su2_phase_cloner"]] = _series_from_masked_op(
        su2_phase_op, cloner_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_phase_resister"]] = _series_from_masked_op(
        su2_phase_op, resister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_phase_persister"]] = _series_from_masked_op(
        su2_phase_op, persister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_component_cloner"]] = _series_from_masked_op(
        su2_component_op, cloner_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_component_resister"]] = _series_from_masked_op(
        su2_component_op, resister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_component_persister"]] = _series_from_masked_op(
        su2_component_op, persister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_doublet_cloner"]] = _series_from_masked_op(
        su2_doublet_op, cloner_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_doublet_resister"]] = _series_from_masked_op(
        su2_doublet_op, resister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_doublet_persister"]] = _series_from_masked_op(
        su2_doublet_op, persister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_doublet_diff_cloner"]] = _series_from_masked_op(
        su2_doublet_diff_op, cloner_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_doublet_diff_resister"]] = _series_from_masked_op(
        su2_doublet_diff_op, resister_mask
    )
    outputs[SU2_COMPANION_CHANNEL_MAP["su2_doublet_diff_persister"]] = _series_from_masked_op(
        su2_doublet_diff_op, persister_mask
    )
    return outputs


def _time_bootstrap_mass_std(
    *,
    series: Tensor,  # [S,T]
    dt: float,
    config: CorrelatorConfig,
    max_lag: int,
    use_connected: bool,
    n_bootstrap: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Return (correlator std [S,L], mass std [S]) via vectorized time bootstrap."""
    s_count, t_len = series.shape
    gen = torch.Generator(device=series.device)
    gen.manual_seed(int(seed))
    idx = torch.randint(0, t_len, (int(n_bootstrap), t_len), generator=gen, device=series.device)
    sampled = torch.gather(
        series.unsqueeze(0).expand(int(n_bootstrap), -1, -1),
        dim=2,
        index=idx.unsqueeze(1).expand(-1, s_count, -1),
    )
    boot_corr = _fft_correlator_batched(
        sampled.reshape(-1, t_len),
        max_lag=int(max_lag),
        use_connected=bool(use_connected),
    ).reshape(int(n_bootstrap), s_count, -1)
    corr_std = boot_corr.std(dim=0)

    mass_samples = torch.full(
        (int(n_bootstrap), s_count),
        float("nan"),
        dtype=torch.float32,
        device=series.device,
    )
    for b_idx in range(int(n_bootstrap)):
        for s_idx in range(s_count):
            mass_samples[b_idx, s_idx] = float(
                _fit_mass_only(boot_corr[b_idx, s_idx], dt=dt, config=config)
            )
    mass_std = _nanstd_compat(mass_samples, dim=0)
    return corr_std, mass_std


def compute_multiscale_electroweak_channels(
    history: RunHistory,
    *,
    config: MultiscaleElectroweakConfig,
    channels: list[str] | None = None,
) -> MultiscaleElectroweakOutput:
    """Compute multiscale SU(2) electroweak channels and select best scales."""
    if config.kernel_type not in KERNEL_TYPES:
        raise ValueError(f"kernel_type must be one of {KERNEL_TYPES}, got {config.kernel_type!r}.")
    if config.bootstrap_mode not in BOOTSTRAP_MODES:
        raise ValueError(
            f"bootstrap_mode must be one of {BOOTSTRAP_MODES}, got {config.bootstrap_mode!r}."
        )
    resolved_mode = _resolve_su2_operator_mode(config.su2_operator_mode)
    resolved_scope = _resolve_walker_type_scope(config.walker_type_scope)

    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
        mc_time_index=config.mc_time_index,
    )
    if not frame_indices:
        empty = torch.empty(0, dtype=torch.float32)
        return MultiscaleElectroweakOutput(
            scales=empty,
            frame_indices=[],
            per_scale_results={},
            best_results={},
            best_scale_index={},
            series_by_channel={},
            bootstrap_mode_applied="none",
            notes=["No valid frame indices for multiscale electroweak analysis."],
            bootstrap_mass_std=None,
        )

    requested = [str(c).strip() for c in (channels or SU2_BASE_CHANNELS) if str(c).strip()]
    requested = [c for c in requested if c in SUPPORTED_CHANNELS]
    if not requested:
        requested = list(SU2_BASE_CHANNELS)

    aliases = [SU2_COMPANION_CHANNEL_MAP[name] for name in requested]

    epsilon_c, epsilon_clone, lambda_alg = _resolve_electroweak_params(history, config)
    h_eff = float(max(config.h_eff, 1e-8))

    frame_ids_t = torch.as_tensor(
        frame_indices, dtype=torch.long, device=history.x_before_clone.device
    )
    positions = history.x_before_clone.index_select(0, frame_ids_t).float()
    velocities = history.v_before_clone.index_select(0, frame_ids_t).float()
    fitness = history.fitness.index_select(0, frame_ids_t - 1).float()
    alive = history.alive_mask.index_select(0, frame_ids_t - 1).to(dtype=torch.bool)
    will_clone_hist = getattr(history, "will_clone", None)
    will_clone = None
    if will_clone_hist is not None:
        if not torch.is_tensor(will_clone_hist):
            will_clone_hist = torch.as_tensor(will_clone_hist, device=frame_ids_t.device)
        will_clone = will_clone_hist.index_select(0, frame_ids_t - 1).to(dtype=torch.bool)

    scales = select_interesting_scales_from_history(
        history,
        n_scales=int(config.n_scales),
        method=str(config.kernel_distance_method),
        frame_indices=frame_indices,
        n_scale_frames=int(config.kernel_scale_frames),
        calibration_batch_size=int(max(1, config.kernel_batch_size)),
        edge_weight_mode=str(config.edge_weight_mode),
        assume_all_alive=bool(config.kernel_assume_all_alive),
        q_low=float(config.kernel_scale_q_low),
        q_high=float(config.kernel_scale_q_high),
        max_samples=int(config.kernel_max_scale_samples),
        min_scale=float(config.kernel_min_scale),
        device=positions.device,
        dtype=torch.float32,
    )

    n_scales = int(scales.numel())
    n_frames = len(frame_indices)
    series_by_channel: dict[str, Tensor] = {
        alias: torch.zeros((n_scales, n_frames), dtype=torch.float32, device=positions.device)
        for alias in aliases
    }

    frame_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(frame_indices)}
    for frame_ids_chunk, _, kernels_chunk, _ in iter_smeared_kernel_batches_from_history(
        history,
        scales=scales,
        method=str(config.kernel_distance_method),
        frame_indices=frame_indices,
        batch_size=int(max(1, config.kernel_batch_size)),
        edge_weight_mode=str(config.edge_weight_mode),
        assume_all_alive=bool(config.kernel_assume_all_alive),
        kernel_type=str(config.kernel_type),
        device=positions.device,
        dtype=torch.float32,
    ):
        pos_idx = [frame_to_pos[int(frame_idx)] for frame_idx in frame_ids_chunk]
        pos_t = torch.as_tensor(pos_idx, dtype=torch.long, device=positions.device)
        chunk_series = _compute_su2_series_from_kernels(
            positions=positions.index_select(0, pos_t),
            velocities=velocities.index_select(0, pos_t),
            fitness=fitness.index_select(0, pos_t),
            alive=alive.index_select(0, pos_t),
            kernels=kernels_chunk,
            h_eff=h_eff,
            epsilon_c=epsilon_c,
            epsilon_clone=epsilon_clone,
            lambda_alg=lambda_alg,
            bounds=history.bounds,
            pbc=bool(history.pbc),
            will_clone=will_clone.index_select(0, pos_t) if will_clone is not None else None,
            su2_operator_mode=resolved_mode,
            enable_walker_type_split=bool(config.enable_walker_type_split),
            walker_type_scope=resolved_scope,
        )
        for base in requested:
            alias = SU2_COMPANION_CHANNEL_MAP[base]
            series_by_channel[alias][:, pos_t] = chunk_series[alias]

    dt = float(history.delta_t * history.record_every)
    correlator_cfg = CorrelatorConfig(
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        window_widths=config.window_widths,
        fit_mode=str(config.fit_mode),
        fit_start=int(config.fit_start),
        fit_stop=config.fit_stop,
        min_fit_points=int(config.min_fit_points),
        compute_bootstrap_errors=False,
        n_bootstrap=int(config.n_bootstrap),
    )

    per_scale_results: dict[str, list[ChannelCorrelatorResult]] = {}
    best_results: dict[str, ChannelCorrelatorResult] = {}
    best_scale_index: dict[str, int] = {}
    bootstrap_mass_std_out: dict[str, Tensor] = {}
    notes: list[str] = []
    bootstrap_mode_applied = "none"
    if bool(config.compute_bootstrap_errors):
        bootstrap_mode_applied = "time"
        if config.bootstrap_mode in {"walker", "hybrid"}:
            notes.append(
                f"bootstrap_mode={config.bootstrap_mode!r} requested; "
                "using vectorized time bootstrap for multiscale SU(2)."
            )

    no_best_channels: list[str] = []
    for alias in aliases:
        series_scales = series_by_channel[alias]  # [S,T]
        s_count, t_len = series_scales.shape
        corr_stack = _fft_correlator_batched(
            series_scales,
            max_lag=int(correlator_cfg.max_lag),
            use_connected=bool(correlator_cfg.use_connected),
        )

        corr_err_stack = None
        mass_std = None
        if bool(config.compute_bootstrap_errors) and int(config.n_bootstrap) > 1 and t_len > 1:
            corr_err_stack, mass_std = _time_bootstrap_mass_std(
                series=series_scales,
                dt=dt,
                config=correlator_cfg,
                max_lag=int(correlator_cfg.max_lag),
                use_connected=bool(correlator_cfg.use_connected),
                n_bootstrap=int(config.n_bootstrap),
                seed=int(config.bootstrap_seed) + hash(alias) % 997,
            )

        results_per_scale: list[ChannelCorrelatorResult] = []
        for s_idx in range(s_count):
            corr_err = corr_err_stack[s_idx] if corr_err_stack is not None else None
            result = _build_result_from_precomputed_correlator(
                channel_name=alias,
                correlator=corr_stack[s_idx],
                dt=dt,
                config=correlator_cfg,
                n_samples=t_len,
                series=series_scales[s_idx],
                correlator_err=corr_err,
            )
            result.mass_fit["scale"] = float(scales[s_idx].item())
            result.mass_fit["scale_index"] = int(s_idx)
            result.mass_fit["source"] = "multiscale_su2"
            if mass_std is not None:
                extra = float(mass_std[s_idx].item())
                if math.isfinite(extra) and extra >= 0:
                    result.mass_fit["bootstrap_mass_error"] = extra
                    base_err = float(result.mass_fit.get("mass_error", float("inf")))
                    if math.isfinite(base_err) and base_err >= 0:
                        result.mass_fit["mass_error"] = float(
                            math.sqrt(base_err * base_err + extra * extra)
                        )
                    else:
                        result.mass_fit["mass_error"] = extra
            results_per_scale.append(result)

        per_scale_results[alias] = results_per_scale

        best_idx = _select_best_scale(
            results_per_scale,
            min_r2=float(config.best_min_r2),
            min_windows=int(config.best_min_windows),
            max_error_pct=float(config.best_max_error_pct),
            remove_artifacts=bool(config.best_remove_artifacts),
        )
        if best_idx is None:
            best_scale_index[alias] = -1
            no_best_channels.append(alias)
        else:
            best_scale_index[alias] = int(best_idx)
            best_results[alias] = results_per_scale[best_idx]

        if mass_std is not None:
            bootstrap_mass_std_out[alias] = mass_std.detach().clone()

    if no_best_channels:
        preview = ", ".join(no_best_channels[:6])
        suffix = " ..." if len(no_best_channels) > 6 else ""
        notes.append(
            "No multiscale_best selected for "
            f"{len(no_best_channels)} channel(s) after best-scale filters "
            f"(min_r2={float(config.best_min_r2):.3g}, "
            f"min_windows={int(config.best_min_windows)}, "
            f"max_error_pct={float(config.best_max_error_pct):.3g}, "
            f"remove_artifacts={bool(config.best_remove_artifacts)}): "
            f"{preview}{suffix}"
        )

    return MultiscaleElectroweakOutput(
        scales=scales.detach().clone(),
        frame_indices=list(frame_indices),
        per_scale_results=per_scale_results,
        best_results=best_results,
        best_scale_index=best_scale_index,
        series_by_channel={k: v.detach().clone() for k, v in series_by_channel.items()},
        bootstrap_mode_applied=bootstrap_mode_applied,
        notes=notes,
        bootstrap_mass_std=bootstrap_mass_std_out or None,
    )
