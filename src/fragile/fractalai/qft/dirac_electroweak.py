"""Unified Dirac/Electroweak observable bundle for dashboard analysis.

This module groups the three complementary electron-sector approaches:
1. Spectral Dirac operator analysis (with color-singlet proxy scoring).
2. Higgs-VEV/Yukawa mass prediction from geometry + fitness gaps.
3. Electroweak correlator proxies (including lower-doublet/electron component).

It reuses existing vectorized kernels and correlator/mass extraction utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.correlator_channels import (
    ChannelCorrelatorResult,
    compute_channel_correlator,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.dirac_spectrum import (
    build_antisymmetric_kernel,
    compute_dirac_spectrum,
    DiracSpectrumConfig,
    DiracSpectrumResult,
)
from fragile.fractalai.qft.electroweak_channels import (
    compute_electroweak_channels,
    ElectroweakChannelConfig,
    ElectroweakChannelOutput,
)
from fragile.fractalai.qft.electroweak_observables import (
    build_phase_space_antisymmetric_kernel,
    compute_fitness_gap_distribution,
    compute_higgs_vev_from_positions,
    predict_yukawa_mass_from_fitness,
)


@dataclass
class DiracElectroweakConfig:
    """Configuration for unified Dirac/Electroweak computations."""

    electroweak: ElectroweakChannelConfig
    dirac: DiracSpectrumConfig
    electroweak_channels: list[str] | None = None
    color_singlet_quantile: float = 0.9
    color_state_mass: float = 1.0
    color_state_ell0: float = 1.0
    sigma_max_lag: int = 80
    sigma_use_connected: bool = True
    sigma_fit_mode: str = "aic"
    sigma_fit_start: int = 2
    sigma_fit_stop: int | None = None
    sigma_min_fit_points: int = 2
    sigma_window_widths: list[int] | None = None
    sigma_compute_bootstrap_errors: bool = False
    sigma_n_bootstrap: int = 100


@dataclass
class DiracColorSingletSpectrum:
    """Dirac spectral modes scored by color-singlet proxy overlap."""

    mode_index: np.ndarray
    masses: np.ndarray
    lepton_scores: np.ndarray
    quark_scores: np.ndarray
    lepton_threshold: float
    electron_mass: float | None
    n_singlet_modes: int


@dataclass
class DiracElectroweakBundle:
    """Unified outputs for new Dirac/Electroweak tab."""

    electroweak_output: ElectroweakChannelOutput
    dirac_result: DiracSpectrumResult
    color_singlet_spectrum: DiracColorSingletSpectrum | None
    electron_component_result: ChannelCorrelatorResult
    higgs_sigma_result: ChannelCorrelatorResult
    higgs_vev: float
    higgs_vev_std: float
    vev_time_mean: float
    vev_time_std: float
    fitness_phi0: float
    fitness_delta_phi_e: float
    yukawa_e: float
    electron_mass_yukawa: float
    frame_indices: list[int]


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


def _resolve_electroweak_params(
    history: RunHistory,
    cfg: ElectroweakChannelConfig,
) -> tuple[float, float, float, float]:
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

    return (
        float(max(epsilon_d, 1e-8)),
        float(max(epsilon_c, 1e-8)),
        float(max(epsilon_clone, 1e-8)),
        float(max(lambda_alg, 0.0)),
    )


def _resolve_series_frame_indices(history: RunHistory, ew_cfg: ElectroweakChannelConfig) -> list[int]:
    start_idx = max(1, int(history.n_recorded * float(ew_cfg.warmup_fraction)))
    end_idx = max(start_idx + 1, int(history.n_recorded * float(ew_cfg.end_fraction)))
    return list(range(start_idx, min(end_idx, history.n_recorded)))


def _resolve_companion_topology(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"distance", "clone", "both"}:
        msg = "companion_topology must be 'distance', 'clone', or 'both'."
        raise ValueError(msg)
    return mode_norm


def _to_correlator_config(cfg: DiracElectroweakConfig) -> CorrelatorConfig:
    return CorrelatorConfig(
        max_lag=int(cfg.sigma_max_lag),
        use_connected=bool(cfg.sigma_use_connected),
        window_widths=cfg.sigma_window_widths,
        fit_mode=str(cfg.sigma_fit_mode),
        fit_start=int(cfg.sigma_fit_start),
        fit_stop=cfg.sigma_fit_stop,
        min_fit_points=int(cfg.sigma_min_fit_points),
        compute_bootstrap_errors=bool(cfg.sigma_compute_bootstrap_errors),
        n_bootstrap=int(cfg.sigma_n_bootstrap),
    )


def _apply_pbc_diff(diff: Tensor, bounds: Any | None) -> Tensor:
    if bounds is None:
        return diff
    high = bounds.high.to(diff)
    low = bounds.low.to(diff)
    span = high - low
    return diff - span * torch.round(diff / span)


def _build_result_from_precomputed(
    channel_name: str,
    series: Tensor,
    correlator: Tensor,
    dt: float,
    config: CorrelatorConfig,
) -> ChannelCorrelatorResult:
    effective_mass = compute_effective_mass_torch(correlator, dt)
    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(correlator.abs(), dt, config)
        window_data = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(correlator, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(correlator, dt, config)
        window_data = {
            "window_masses": mass_fit.pop("window_masses", None),
            "window_aic": mass_fit.pop("window_aic", None),
            "window_widths": mass_fit.pop("window_widths", None),
            "window_r2": mass_fit.pop("window_r2", None),
        }

    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=correlator,
        correlator_err=None,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series,
        n_samples=int(series.numel()),
        dt=dt,
        **window_data,
    )


def _complex_field_correlator(
    field: Tensor,
    valid: Tensor,
    *,
    max_lag: int,
    use_connected: bool,
) -> Tensor:
    """Compute C(τ)=<Re[psi*(t) psi(t+τ)]> across walkers and MC time."""
    if field.numel() == 0 or valid.numel() == 0:
        return torch.zeros(max_lag + 1, dtype=torch.float32)

    x = field
    m = valid.bool()
    if use_connected:
        # Subtract per-walker mean field over valid time samples.
        m_f = m.to(dtype=x.real.dtype)
        denom = m_f.sum(dim=0).clamp(min=1.0)
        mu = torch.where(
            denom > 0,
            (x * m_f).sum(dim=0) / denom,
            torch.zeros_like(denom, dtype=x.dtype),
        )
        x = torch.where(m, x - mu.unsqueeze(0), torch.zeros_like(x))
    else:
        x = torch.where(m, x, torch.zeros_like(x))

    T = int(x.shape[0])
    L = min(int(max_lag), max(T - 1, 0))
    if T <= 0:
        return torch.zeros(max_lag + 1, device=x.device, dtype=torch.float32)

    # Batch over walkers: [N, T]
    x_bt = x.transpose(0, 1).to(torch.complex64)
    m_bt = m.transpose(0, 1).to(torch.float32)
    pad = (0, T)
    x_pad = F.pad(x_bt, pad)
    m_pad = F.pad(m_bt, pad)

    # Numerator: sum_t Re[psi*(t) psi(t+tau)] for each walker.
    fft_x = torch.fft.fft(x_pad, dim=1)
    auto_x = torch.fft.ifft(fft_x * fft_x.conj(), dim=1).real

    # Denominator: valid pair counts for each lag and walker.
    fft_m = torch.fft.fft(m_pad, dim=1)
    auto_m = torch.fft.ifft(fft_m * fft_m.conj(), dim=1).real

    num = auto_x[:, : L + 1].sum(dim=0)
    den = auto_m[:, : L + 1].sum(dim=0)
    corr = torch.where(den > 0.5, num / den.clamp(min=1e-12), torch.zeros_like(num))
    corr = corr.to(torch.float32)
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    corr[0] = torch.clamp(corr[0], min=1e-12)

    if L < max_lag:
        corr = F.pad(corr, (0, max_lag - L))
    return corr


def _compute_doublet_and_sigma_series(
    history: RunHistory,
    cfg: DiracElectroweakConfig,
) -> tuple[Tensor, Tensor, Tensor, list[int], Tensor, Tensor]:
    frame_indices = _resolve_series_frame_indices(history, cfg.electroweak)
    if not frame_indices:
        empty = torch.zeros(0, dtype=torch.float32)
        empty_c = torch.zeros((0, 0), dtype=torch.complex64)
        empty_b = torch.zeros((0, 0), dtype=torch.bool)
        return empty, empty, empty, [], empty_c, empty_b

    h_eff = float(max(cfg.electroweak.h_eff, 1e-12))
    _epsilon_d, epsilon_c, epsilon_clone, lambda_alg = _resolve_electroweak_params(
        history, cfg.electroweak
    )
    topology = _resolve_companion_topology(getattr(cfg.electroweak, "companion_topology", "distance"))
    n_slots = 2 if topology == "both" else 1
    companions_distance = getattr(history, "companions_distance", None)
    companions_clone = getattr(history, "companions_clone", None)

    e_series: list[float] = []
    sigma_series: list[float] = []
    vev_series: list[float] = []
    psi_frames: list[Tensor] = []
    valid_frames: list[Tensor] = []

    for frame_idx in frame_indices:
        info_idx = frame_idx - 1
        if info_idx < 0 or info_idx >= history.alive_mask.shape[0]:
            continue

        positions = history.x_before_clone[frame_idx]
        velocities = history.v_before_clone[frame_idx]
        fitness = history.fitness[info_idx]
        alive = history.alive_mask[info_idx].bool()
        alive_count = int(alive.sum().item())
        if alive_count <= 0:
            n = int(positions.shape[0])
            e_series.append(0.0)
            sigma_series.append(0.0)
            vev_series.append(0.0)
            psi_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.complex64))
            valid_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.bool))
            continue

        # Higgs sigma mode from radial fluctuations around frame centroid.
        center = positions[alive].mean(dim=0)
        radii = torch.linalg.vector_norm(positions[alive] - center.unsqueeze(0), dim=-1)
        vev_t = float(radii.mean().item()) if radii.numel() else 0.0
        sigma = radii - vev_t
        sigma2_t = float((sigma * sigma).mean().item()) if sigma.numel() else 0.0
        sigma_series.append(sigma2_t)
        vev_series.append(vev_t)

        n = int(positions.shape[0])
        if companions_distance is None or info_idx >= companions_distance.shape[0]:
            e_series.append(0.0)
            psi_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.complex64))
            valid_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.bool))
            continue

        comp_d = companions_distance[info_idx]
        if not torch.is_tensor(comp_d):
            comp_d = torch.as_tensor(comp_d)
        comp_d = comp_d.to(device=positions.device, dtype=torch.long)
        if comp_d.numel() != n:
            e_series.append(0.0)
            psi_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.complex64))
            valid_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.bool))
            continue
        comp_d = comp_d.clamp(min=0, max=max(n - 1, 0))

        comp_c: Tensor | None = None
        if companions_clone is not None and info_idx < companions_clone.shape[0]:
            comp_c = companions_clone[info_idx]
            if not torch.is_tensor(comp_c):
                comp_c = torch.as_tensor(comp_c)
            comp_c = comp_c.to(device=positions.device, dtype=torch.long)
            if comp_c.numel() == n:
                comp_c = comp_c.clamp(min=0, max=max(n - 1, 0))
            else:
                comp_c = None

        src = torch.arange(n, device=positions.device, dtype=torch.long)
        if topology == "distance":
            comp_slots = comp_d.unsqueeze(1)
        elif topology == "clone":
            if comp_c is None:
                e_series.append(0.0)
                psi_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.complex64))
                valid_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.bool))
                continue
            comp_slots = comp_c.unsqueeze(1)
        else:
            fallback_clone = comp_c if comp_c is not None else src
            comp_slots = torch.stack([comp_d, fallback_clone], dim=1)

        valid = (
            alive.unsqueeze(1)
            & alive[comp_slots]
            & (comp_slots != src.unsqueeze(1))
        )
        if not torch.any(valid):
            e_series.append(0.0)
            psi_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.complex64))
            valid_frames.append(torch.zeros(n * n_slots, device=positions.device, dtype=torch.bool))
            continue

        diff_x = positions[comp_slots] - positions.unsqueeze(1)
        if bool(history.pbc) and history.bounds is not None:
            diff_x = _apply_pbc_diff(diff_x, history.bounds)
        diff_v = velocities[comp_slots] - velocities.unsqueeze(1)
        d_alg2 = (diff_x * diff_x).sum(dim=-1) + float(lambda_alg) * (diff_v * diff_v).sum(dim=-1)

        p_ij = torch.exp(-d_alg2 / (2.0 * epsilon_c * epsilon_c))
        p_ji = p_ij  # Symmetric distance amplitude.

        fitness_i = fitness.unsqueeze(1)
        fitness_j = fitness[comp_slots]
        denom_ji = torch.where(
            (fitness_j + epsilon_clone).abs() < 1e-12,
            torch.full_like(fitness_j, epsilon_clone),
            fitness_j + epsilon_clone,
        )
        theta_ji = ((fitness_i - fitness_j) / denom_ji) / h_eff

        norm = torch.sqrt(p_ij + p_ji + 1e-12)
        psi_lower = torch.sqrt(p_ji) * torch.exp(1j * theta_ji) / norm
        electron_component = torch.abs(psi_lower).pow(2)
        e_val = float(electron_component[valid].mean().item()) if torch.any(valid) else 0.0
        e_series.append(e_val)
        psi_frames.append(
            torch.where(
                valid,
                psi_lower.to(torch.complex64),
                torch.zeros_like(psi_lower, dtype=torch.complex64),
            ).reshape(-1)
        )
        valid_frames.append(valid.reshape(-1))

    e_t = torch.tensor(e_series, dtype=torch.float32)
    sigma_t = torch.tensor(sigma_series, dtype=torch.float32)
    vev_t = torch.tensor(vev_series, dtype=torch.float32)
    psi_t = torch.stack(psi_frames, dim=0) if psi_frames else torch.zeros((0, 0), dtype=torch.complex64)
    valid_t = torch.stack(valid_frames, dim=0) if valid_frames else torch.zeros((0, 0), dtype=torch.bool)
    return e_t, sigma_t, vev_t, frame_indices[: len(e_series)], psi_t, valid_t


def _compute_color_states_single_frame(
    history: RunHistory,
    info_idx: int,
    alive_indices: np.ndarray,
    *,
    h_eff: float,
    mass: float,
    ell0: float,
) -> np.ndarray:
    if alive_indices.size == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    force_visc = history.force_viscous[info_idx]
    v_pre = history.v_before_clone[info_idx]
    force_np = force_visc.detach().cpu().numpy()
    v_np = v_pre.detach().cpu().numpy()
    phase = (float(mass) * v_np * float(ell0)) / max(float(h_eff), 1e-12)
    complex_phase = np.exp(1j * phase)
    tilde = force_np.astype(np.complex128) * complex_phase.astype(np.complex128)
    norm = np.linalg.norm(tilde, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    color = tilde / norm
    return color[alive_indices]


def _compute_color_singlet_spectrum(
    history: RunHistory,
    cfg: DiracElectroweakConfig,
) -> DiracColorSingletSpectrum | None:
    max_t = max(history.n_recorded - 2, 0)
    info_idx = cfg.dirac.mc_time_index if cfg.dirac.mc_time_index is not None else max_t
    info_idx = int(min(max(info_idx, 0), max_t))

    alive_t = history.alive_mask[info_idx].detach().cpu().numpy().astype(bool)
    fitness_t = history.fitness[info_idx].detach().cpu().numpy().astype(np.float64)
    if int(alive_t.sum()) < 2:
        return None

    mode = str(getattr(cfg.dirac, "kernel_mode", "fitness_ratio")).strip().lower()
    if mode == "phase_space":
        positions_t = history.x_before_clone[info_idx].detach().cpu().numpy().astype(np.float64)
        velocities_t = history.v_before_clone[info_idx].detach().cpu().numpy().astype(np.float64)
        epsilon_c = cfg.dirac.epsilon_c
        if epsilon_c is None:
            params = history.params if isinstance(history.params, dict) else None
            epsilon_c = _nested_param(params, "companion_selection_clone", "epsilon", default=1.0)
        kernel, alive_indices = build_phase_space_antisymmetric_kernel(
            positions=positions_t,
            velocities=velocities_t,
            fitness=fitness_t,
            alive_mask=alive_t,
            epsilon_c=float(max(epsilon_c, 1e-8)),
            lambda_alg=float(cfg.dirac.lambda_alg),
            h_eff=float(cfg.dirac.h_eff),
            epsilon_clone=float(cfg.dirac.epsilon_clone),
            bounds=getattr(history, "bounds", None),
            pbc=bool(getattr(history, "pbc", False)),
            include_phase=bool(cfg.dirac.include_phase),
        )
    else:
        kernel, alive_indices = build_antisymmetric_kernel(
            fitness_t, alive_t, epsilon_clone=float(cfg.dirac.epsilon_clone)
        )

    if kernel.size == 0:
        return None

    ih = 1j * kernel
    ih = 0.5 * (ih + ih.conj().T)
    evals, evecs = np.linalg.eigh(ih)
    masses = np.abs(evals)
    order = np.argsort(masses)
    masses = masses[order]
    evecs = evecs[:, order]
    probs = np.abs(evecs) ** 2

    colors = _compute_color_states_single_frame(
        history,
        info_idx=info_idx,
        alive_indices=alive_indices,
        h_eff=float(cfg.dirac.h_eff),
        mass=cfg.color_state_mass,
        ell0=cfg.color_state_ell0,
    )
    if colors.size == 0:
        return None

    color_mean = colors.mean(axis=0, keepdims=True)
    color_fluct = np.sum(np.abs(colors - color_mean) ** 2, axis=1)
    quark_scores = probs.T @ color_fluct
    quark_norm = quark_scores / max(float(np.mean(color_fluct)), 1e-12)
    lepton_scores = 1.0 / (1.0 + quark_norm)

    q = float(np.clip(cfg.color_singlet_quantile, 0.0, 1.0))
    threshold = float(np.quantile(lepton_scores, q))
    singlet_mask = lepton_scores >= threshold
    electron_mass = float(masses[singlet_mask].min()) if np.any(singlet_mask) else None

    return DiracColorSingletSpectrum(
        mode_index=np.arange(masses.shape[0], dtype=np.int64),
        masses=masses,
        lepton_scores=lepton_scores,
        quark_scores=quark_norm,
        lepton_threshold=threshold,
        electron_mass=electron_mass,
        n_singlet_modes=int(np.sum(singlet_mask)),
    )


def compute_dirac_electroweak_bundle(
    history: RunHistory,
    config: DiracElectroweakConfig,
) -> DiracElectroweakBundle:
    """Compute unified Dirac + electroweak + Higgs/Yukawa observables."""
    ew_output = compute_electroweak_channels(
        history,
        channels=config.electroweak_channels,
        config=config.electroweak,
    )
    dirac_result = compute_dirac_spectrum(history, config=config.dirac)
    color_singlet = _compute_color_singlet_spectrum(history, cfg=config)

    (
        e_series,
        sigma_series,
        vev_series,
        frame_indices,
        psi_lower_series,
        psi_valid,
    ) = _compute_doublet_and_sigma_series(history, config)
    dt = float(history.delta_t * history.record_every)
    corr_cfg = _to_correlator_config(config)
    e_corr = _complex_field_correlator(
        psi_lower_series,
        psi_valid,
        max_lag=int(corr_cfg.max_lag),
        use_connected=bool(corr_cfg.use_connected),
    )
    e_result = _build_result_from_precomputed(
        channel_name="electron_component",
        series=e_series,
        correlator=e_corr,
        dt=dt,
        config=corr_cfg,
    )
    sigma_result = compute_channel_correlator(
        series=sigma_series,
        dt=dt,
        config=corr_cfg,
        channel_name="higgs_sigma",
    )

    # Snapshot-based Higgs VEV + Yukawa prediction at the Dirac analysis frame.
    mc_frame = int(min(max(dirac_result.mc_time_index + 1, 1), history.n_recorded - 1))
    info_idx = mc_frame - 1
    positions = history.x_before_clone[mc_frame]
    fitness = history.fitness[info_idx]
    alive = history.alive_mask[info_idx]
    vev_stats = compute_higgs_vev_from_positions(positions, alive=alive)
    gap_stats = compute_fitness_gap_distribution(fitness, alive=alive)
    yukawa = predict_yukawa_mass_from_fitness(
        v_higgs=float(vev_stats["vev"]),
        fitness=fitness,
        alive=alive,
        phi0=float(gap_stats["phi0"]),
    )

    return DiracElectroweakBundle(
        electroweak_output=ew_output,
        dirac_result=dirac_result,
        color_singlet_spectrum=color_singlet,
        electron_component_result=e_result,
        higgs_sigma_result=sigma_result,
        higgs_vev=float(vev_stats["vev"]),
        higgs_vev_std=float(vev_stats["vev_std"]),
        vev_time_mean=float(vev_series.mean().item()) if vev_series.numel() else 0.0,
        vev_time_std=float(vev_series.std(unbiased=False).item()) if vev_series.numel() else 0.0,
        fitness_phi0=float(yukawa["phi0"]),
        fitness_delta_phi_e=float(yukawa["delta_phi"]),
        yukawa_e=float(yukawa["yukawa"]),
        electron_mass_yukawa=float(yukawa["mass"]),
        frame_indices=frame_indices,
    )
