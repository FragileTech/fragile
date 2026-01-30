"""Electroweak (U1/SU2) channel correlators for Fractal Gas runs.

This module builds U(1) fitness-phase and SU(2) cloning-phase time series and
extracts effective masses using the same correlator pipeline as strong-force
channels (FFT correlators + AIC/linear fits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.correlator_channels import (
    ChannelCorrelatorResult,
    ConvolutionalAICExtractor,
    compute_correlator_fft,
    compute_effective_mass_torch,
)


ELECTROWEAK_CHANNELS = (
    "u1_phase",
    "u1_dressed",
    "u1_phase_q2",
    "u1_dressed_q2",
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
    "ew_mixed",
)


@dataclass
class ElectroweakChannelConfig:
    """Configuration for electroweak channel correlators."""

    warmup_fraction: float = 0.1
    max_lag: int = 80
    h_eff: float = 1.0
    use_connected: bool = True
    neighbor_method: str = "uniform"
    knn_k: int = 1
    voronoi_pbc_mode: str = "mirror"
    voronoi_exclude_boundary: bool = True
    voronoi_boundary_tolerance: float = 1e-6

    # Fit settings
    window_widths: list[int] | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")
    fit_mode: str = "aic"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2

    # Electroweak parameters (None => infer from history.params)
    epsilon_d: float | None = None
    epsilon_c: float | None = None
    epsilon_clone: float | None = None
    lambda_alg: float | None = None


def _nested_param(params: dict[str, Any] | None, *keys: str, default: float | None) -> float | None:
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


def _resolve_electroweak_params(history: RunHistory, cfg: ElectroweakChannelConfig) -> dict[str, float]:
    params = history.params if isinstance(history.params, dict) else None
    epsilon_d = cfg.epsilon_d
    if epsilon_d is None:
        epsilon_d = _nested_param(params, "companion_selection", "epsilon", default=None)
    epsilon_c = cfg.epsilon_c
    if epsilon_c is None:
        epsilon_c = _nested_param(params, "companion_selection_clone", "epsilon", default=None)
    lambda_alg = cfg.lambda_alg
    if lambda_alg is None:
        lambda_alg = _nested_param(params, "companion_selection", "lambda_alg", default=None)
    if lambda_alg is None:
        lambda_alg = _nested_param(params, "fitness", "lambda_alg", default=0.0)
    epsilon_clone = cfg.epsilon_clone
    if epsilon_clone is None:
        epsilon_clone = _nested_param(params, "cloning", "epsilon_clone", default=1e-8)

    if epsilon_d is None:
        epsilon_d = 1.0
    if epsilon_c is None:
        epsilon_c = float(epsilon_d)
    if lambda_alg is None:
        lambda_alg = 0.0
    if epsilon_clone is None:
        epsilon_clone = 1e-8

    epsilon_d = float(max(epsilon_d, 1e-8))
    epsilon_c = float(max(epsilon_c, 1e-8))
    epsilon_clone = float(max(epsilon_clone, 1e-8))
    lambda_alg = float(max(lambda_alg, 0.0))

    return {
        "epsilon_d": epsilon_d,
        "epsilon_c": epsilon_c,
        "epsilon_clone": epsilon_clone,
        "lambda_alg": lambda_alg,
    }


def _apply_pbc_diff(diff: Tensor, bounds) -> Tensor:
    if bounds is None:
        return diff
    high = bounds.high.to(diff)
    low = bounds.low.to(diff)
    span = high - low
    return diff - span * torch.round(diff / span)


def _masked_mean(values: Tensor, alive: Tensor) -> Tensor:
    masked = torch.where(alive, values, torch.zeros_like(values))
    counts = alive.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / counts


def _compute_u1_phase(
    fitness: Tensor, companions: Tensor, alive: Tensor, h_eff: float
) -> Tensor:
    fitness_companion = torch.gather(fitness, 1, companions)
    phases = -(fitness_companion - fitness) / h_eff
    return torch.where(alive, phases, torch.zeros_like(phases))


def _compute_su2_phase(
    fitness: Tensor,
    companions: Tensor,
    alive: Tensor,
    epsilon_clone: float,
    h_eff: float,
) -> Tensor:
    fitness_companion = torch.gather(fitness, 1, companions)
    denom = fitness + epsilon_clone
    denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, epsilon_clone), denom)
    scores = (fitness_companion - fitness) / denom
    phases = scores / h_eff
    return torch.where(alive, phases, torch.zeros_like(phases))


def _compute_companion_dist_sq(
    history: RunHistory,
    start_idx: int,
    lambda_alg: float,
) -> Tensor:
    if history.pos_squared_differences is not None and history.vel_squared_differences is not None:
        pos_sq = history.pos_squared_differences[start_idx - 1 : history.n_recorded - 1]
        vel_sq = history.vel_squared_differences[start_idx - 1 : history.n_recorded - 1]
        return pos_sq + lambda_alg * vel_sq

    x_pre = history.x_before_clone[start_idx:]
    v_pre = history.v_before_clone[start_idx:]
    companions = history.companions_distance[start_idx - 1 : history.n_recorded - 1]
    d = x_pre.shape[-1]
    idx = companions.unsqueeze(-1).expand(-1, -1, d)
    x_comp = torch.gather(x_pre, 1, idx)
    v_comp = torch.gather(v_pre, 1, idx)
    diff = x_pre - x_comp
    if history.pbc:
        diff = _apply_pbc_diff(diff, history.bounds)
    pos_sq = (diff**2).sum(dim=-1)
    vel_sq = (v_pre - v_comp).pow(2).sum(dim=-1)
    return pos_sq + lambda_alg * vel_sq


def _compute_clone_dist_sq(
    history: RunHistory,
    start_idx: int,
    lambda_alg: float,
) -> Tensor:
    x_pre = history.x_before_clone[start_idx:]
    v_pre = history.v_before_clone[start_idx:]
    companions = history.companions_clone[start_idx - 1 : history.n_recorded - 1]
    d = x_pre.shape[-1]
    idx = companions.unsqueeze(-1).expand(-1, -1, d)
    x_comp = torch.gather(x_pre, 1, idx)
    v_comp = torch.gather(v_pre, 1, idx)
    diff = x_pre - x_comp
    if history.pbc:
        diff = _apply_pbc_diff(diff, history.bounds)
    pos_sq = (diff**2).sum(dim=-1)
    vel_sq = (v_pre - v_comp).pow(2).sum(dim=-1)
    return pos_sq + lambda_alg * vel_sq


def _compute_neighbor_dist_sq(
    history: RunHistory,
    start_idx: int,
    neighbor_idx: Tensor,
    lambda_alg: float,
) -> Tensor:
    x_pre = history.x_before_clone[start_idx:]
    v_pre = history.v_before_clone[start_idx:]
    d = x_pre.shape[-1]
    idx = neighbor_idx.unsqueeze(-1).expand(-1, -1, d)
    x_comp = torch.gather(x_pre, 1, idx)
    v_comp = torch.gather(v_pre, 1, idx)
    diff = x_pre - x_comp
    if history.pbc:
        diff = _apply_pbc_diff(diff, history.bounds)
    pos_sq = (diff**2).sum(dim=-1)
    vel_sq = (v_pre - v_comp).pow(2).sum(dim=-1)
    return pos_sq + lambda_alg * vel_sq


def _select_electroweak_neighbors(
    history: RunHistory,
    start_idx: int,
    cfg: ElectroweakChannelConfig,
) -> tuple[Tensor, Tensor]:
    if cfg.neighbor_method == "uniform":
        companions_distance = history.companions_distance[start_idx - 1 : history.n_recorded - 1]
        companions_clone = history.companions_clone[start_idx - 1 : history.n_recorded - 1]
        return companions_distance, companions_clone

    n_recorded = history.n_recorded
    T = n_recorded - start_idx
    N = history.N
    device = history.x_final.device
    neighbors_distance = torch.zeros(T, N, device=device, dtype=torch.long)
    neighbors_clone = torch.zeros_like(neighbors_distance)

    alive = history.alive_mask[start_idx - 1 : n_recorded - 1]
    x_pre = history.x_before_clone[start_idx:]

    if cfg.neighbor_method == "knn":
        from fragile.fractalai.qft.particle_observables import compute_knn_indices

        k = max(1, int(cfg.knn_k))
        for t in range(T):
            alive_t = alive[t]
            if not alive_t.any():
                continue
            alive_idx = torch.where(alive_t)[0]
            try:
                knn = compute_knn_indices(
                    positions=x_pre[t],
                    alive=alive_t,
                    k=k,
                    pbc=history.pbc,
                    bounds=history.bounds,
                    sample_indices=alive_idx,
                )
            except ValueError:
                continue
            if knn.numel() == 0:
                continue
            neighbors_distance[t, alive_idx] = knn[:, 0]
            neighbors_clone[t, alive_idx] = knn[:, 0]
        return neighbors_distance, neighbors_clone

    if cfg.neighbor_method == "voronoi":
        try:
            from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation
        except Exception:
            from fragile.fractalai.qft.particle_observables import compute_knn_indices

            k = max(1, int(cfg.knn_k))
            for t in range(T):
                alive_t = alive[t]
                if not alive_t.any():
                    continue
                alive_idx = torch.where(alive_t)[0]
                try:
                    knn = compute_knn_indices(
                        positions=x_pre[t],
                        alive=alive_t,
                        k=k,
                        pbc=history.pbc,
                        bounds=history.bounds,
                        sample_indices=alive_idx,
                    )
                except ValueError:
                    continue
                if knn.numel() == 0:
                    continue
                neighbors_distance[t, alive_idx] = knn[:, 0]
                neighbors_clone[t, alive_idx] = knn[:, 0]
            return neighbors_distance, neighbors_clone

        for t in range(T):
            alive_t = alive[t]
            if not alive_t.any():
                continue
            alive_idx = torch.where(alive_t)[0]
            vor_data = compute_voronoi_tessellation(
                positions=x_pre[t],
                alive=alive_t,
                bounds=history.bounds,
                pbc=history.pbc,
                pbc_mode=cfg.voronoi_pbc_mode,
                exclude_boundary=cfg.voronoi_exclude_boundary,
                boundary_tolerance=cfg.voronoi_boundary_tolerance,
            )
            neighbor_lists = vor_data.get("neighbor_lists", {})
            index_map = vor_data.get("index_map", {})
            reverse_map = {v: k for k, v in index_map.items()}
            for i_idx in alive_idx:
                i_orig = int(i_idx.item())
                i_vor = reverse_map.get(i_orig)
                if i_vor is None:
                    continue
                neighbors_vor = neighbor_lists.get(i_vor, [])
                if not neighbors_vor:
                    continue
                neighbor_orig = index_map.get(neighbors_vor[0])
                if neighbor_orig is None:
                    continue
                neighbors_distance[t, i_orig] = int(neighbor_orig)
                neighbors_clone[t, i_orig] = int(neighbor_orig)
        return neighbors_distance, neighbors_clone

    msg = "neighbor_method must be 'uniform', 'knn', or 'voronoi'"
    raise ValueError(msg)


def _compute_electroweak_series(
    history: RunHistory,
    cfg: ElectroweakChannelConfig,
) -> dict[str, Tensor]:
    start_idx = max(1, int(history.n_recorded * cfg.warmup_fraction))
    if start_idx >= history.n_recorded:
        return {}

    h_eff = float(max(cfg.h_eff, 1e-8))
    if cfg.neighbor_method not in {"uniform", "knn", "voronoi"}:
        msg = "neighbor_method must be 'uniform', 'knn', or 'voronoi'"
        raise ValueError(msg)
    params = _resolve_electroweak_params(history, cfg)
    epsilon_d = params["epsilon_d"]
    epsilon_c = params["epsilon_c"]
    epsilon_clone = params["epsilon_clone"]
    lambda_alg = params["lambda_alg"]

    fitness = history.fitness[start_idx - 1 : history.n_recorded - 1]
    alive = history.alive_mask[start_idx - 1 : history.n_recorded - 1]
    companions_distance, companions_clone = _select_electroweak_neighbors(history, start_idx, cfg)

    u1_phase = _compute_u1_phase(fitness, companions_distance, alive, h_eff)
    su2_phase = _compute_su2_phase(fitness, companions_clone, alive, epsilon_clone, h_eff)

    if cfg.neighbor_method == "uniform":
        u1_dist_sq = _compute_companion_dist_sq(history, start_idx, lambda_alg)
        su2_dist_sq = _compute_clone_dist_sq(history, start_idx, lambda_alg)
    else:
        u1_dist_sq = _compute_neighbor_dist_sq(history, start_idx, companions_distance, lambda_alg)
        su2_dist_sq = _compute_neighbor_dist_sq(history, start_idx, companions_clone, lambda_alg)

    u1_weights = torch.exp(-u1_dist_sq / (2.0 * epsilon_d**2))
    su2_weights = torch.exp(-su2_dist_sq / (2.0 * epsilon_c**2))

    u1_phase_exp = torch.exp(1j * u1_phase)
    su2_phase_exp = torch.exp(1j * su2_phase)
    u1_phase_q2_exp = torch.exp(1j * (2.0 * u1_phase))

    u1_amp = torch.sqrt(u1_weights)
    su2_amp = torch.sqrt(su2_weights)

    su2_comp_phase = torch.gather(su2_phase, 1, companions_clone)
    su2_comp_amp = torch.gather(su2_amp, 1, companions_clone)
    su2_comp_phase_exp = torch.exp(1j * su2_comp_phase)

    series = {
        "u1_phase": _masked_mean(u1_phase_exp, alive),
        "u1_dressed": _masked_mean(u1_amp * u1_phase_exp, alive),
        "u1_phase_q2": _masked_mean(u1_phase_q2_exp, alive),
        "u1_dressed_q2": _masked_mean(u1_amp * u1_phase_q2_exp, alive),
        "su2_phase": _masked_mean(su2_phase_exp, alive),
        "su2_component": _masked_mean(su2_amp * su2_phase_exp, alive),
        "su2_doublet": _masked_mean(
            su2_amp * su2_phase_exp + su2_comp_amp * su2_comp_phase_exp, alive
        ),
        "su2_doublet_diff": _masked_mean(
            su2_amp * su2_phase_exp - su2_comp_amp * su2_comp_phase_exp, alive
        ),
        "ew_mixed": _masked_mean(
            (u1_amp * su2_amp) * torch.exp(1j * (u1_phase + su2_phase)), alive
        ),
    }

    return series


def _extract_mass_aic(correlator: Tensor, cfg: ElectroweakChannelConfig) -> dict[str, Any]:
    mask = correlator > 0
    if not mask.any():
        return {"mass": 0.0, "mass_error": float("inf"), "n_valid_windows": 0}

    log_corr = torch.full_like(correlator, float("nan"))
    log_corr[mask] = torch.log(correlator[mask])
    log_err = torch.ones_like(log_corr) * 0.1

    finite_mask = torch.isfinite(log_corr)
    if not finite_mask.any():
        return {"mass": 0.0, "mass_error": float("inf"), "n_valid_windows": 0}

    last_valid = finite_mask.nonzero()[-1].item()
    log_corr = log_corr[: last_valid + 1]
    log_err = log_err[: last_valid + 1]

    extractor = ConvolutionalAICExtractor(
        window_widths=cfg.window_widths,
        min_mass=cfg.min_mass,
        max_mass=cfg.max_mass,
    )

    return extractor.fit_all_widths(log_corr, log_err)


def _extract_mass_linear(correlator: Tensor, cfg: ElectroweakChannelConfig) -> dict[str, Any]:
    if correlator.numel() == 0:
        return {"mass": 0.0, "amplitude": 0.0, "r_squared": 0.0, "fit_points": 0.0}

    n = correlator.shape[0]
    fit_start = max(0, int(cfg.fit_start))
    fit_stop = cfg.fit_stop
    if fit_stop is None:
        fit_stop = n - 1
    fit_stop = min(int(fit_stop), n - 1)
    if fit_stop < fit_start:
        return {"mass": 0.0, "amplitude": 0.0, "r_squared": 0.0, "fit_points": 0.0}

    idx = torch.arange(n, device=correlator.device, dtype=torch.float32)
    mask = (idx >= fit_start) & (idx <= fit_stop) & (correlator > 0)
    n_points = int(mask.sum().item())
    if n_points < max(2, int(cfg.min_fit_points)):
        return {"mass": 0.0, "amplitude": 0.0, "r_squared": 0.0, "fit_points": float(n_points)}

    x = idx[mask]
    y = torch.log(correlator[mask])
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xx = (x * x).sum()
    sum_xy = (x * y).sum()
    denom = n_points * sum_xx - sum_x * sum_x
    if denom.abs() < 1e-12:
        return {"mass": 0.0, "amplitude": 0.0, "r_squared": 0.0, "fit_points": float(n_points)}

    slope = (n_points * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_points
    mass = -slope
    amplitude = torch.exp(intercept)

    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mass": float(mass.item()),
        "amplitude": float(amplitude.item()),
        "r_squared": float(r_squared.item()),
        "fit_points": float(n_points),
    }


def _extract_mass_linear_abs(correlator: Tensor, cfg: ElectroweakChannelConfig) -> dict[str, Any]:
    return _extract_mass_linear(correlator.abs(), cfg)


def _build_result(
    history: RunHistory,
    series: Tensor,
    channel_name: str,
    cfg: ElectroweakChannelConfig,
) -> ChannelCorrelatorResult:
    dt = float(history.delta_t * history.record_every)
    if series.numel() == 0:
        return ChannelCorrelatorResult(
            channel_name=channel_name,
            correlator=torch.zeros(cfg.max_lag + 1),
            correlator_err=None,
            effective_mass=torch.zeros(cfg.max_lag),
            mass_fit={"mass": 0.0, "mass_error": float("inf")},
            series=series,
            n_samples=0,
            dt=dt,
            window_masses=None,
            window_aic=None,
            window_widths=None,
            window_r2=None,
        )

    correlator = compute_correlator_fft(
        series.real if series.is_complex() else series,
        max_lag=cfg.max_lag,
        use_connected=cfg.use_connected,
    )
    effective_mass = compute_effective_mass_torch(correlator, dt)

    window_masses = None
    window_aic = None
    window_widths = None
    window_r2 = None

    if cfg.fit_mode == "linear_abs":
        mass_fit = _extract_mass_linear_abs(correlator, cfg)
    elif cfg.fit_mode == "linear":
        mass_fit = _extract_mass_linear(correlator, cfg)
    else:
        mass_fit = _extract_mass_aic(correlator, cfg)
        window_masses = mass_fit.pop("window_masses", None)
        window_aic = mass_fit.pop("window_aic", None)
        window_widths = mass_fit.pop("window_widths", None)
        window_r2 = mass_fit.pop("window_r2", None)

    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=correlator,
        correlator_err=None,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series,
        n_samples=int(series.numel()),
        dt=dt,
        window_masses=window_masses,
        window_aic=window_aic,
        window_widths=window_widths,
        window_r2=window_r2,
    )


def compute_all_electroweak_channels(
    history: RunHistory,
    channels: list[str] | None = None,
    config: ElectroweakChannelConfig | None = None,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute electroweak correlators for multiple channels."""
    config = config or ElectroweakChannelConfig()
    if channels is None:
        channels = list(ELECTROWEAK_CHANNELS)

    series_map = _compute_electroweak_series(history, config)
    results: dict[str, ChannelCorrelatorResult] = {}

    for name in channels:
        series = series_map.get(name)
        if series is None:
            continue
        results[name] = _build_result(history, series, name, config)

    return results
