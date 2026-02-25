"""h_eff parameter sweep: operator variance calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from fragile.physics.app.coupling_diagnostics import (
    _masked_mean,
    _masked_var,
    _resolve_frame_indices,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils.color_states import estimate_ell0_auto


SWEEP_CRITERIA = (
    "variance_ratio",       # Var(Im)/Var(Re) — chiral susceptibility ratio
    "mean_amplitude_ratio",  # |⟨Im⟩|/|⟨Re⟩| — zero-momentum pseudoscalar/scalar
    "r_circ",               # phase concentration — maximize coherence
)

CRITERION_LABELS: dict[str, str] = {
    "variance_ratio": "Var(Im)/Var(Re)",
    "mean_amplitude_ratio": "|⟨pseudoscalar⟩|/|⟨scalar⟩|",
    "r_circ": "R_circ (phase concentration)",
}


@dataclass
class HeffSweepConfig:
    """Configuration for h_eff parameter sweep."""

    h_eff_min: float = 0.01
    h_eff_max: float = 10.0
    n_points: int = 30
    log_scale: bool = True
    criterion: str = "variance_ratio"
    # Shared physics params (h_eff-independent)
    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    ell0_method: str = "companion"
    color_dims: tuple[int, ...] | None = None
    companion_topology: str = "both"
    pair_weighting: str = "uniform"
    eps: float = 1e-12


@dataclass
class HeffSweepResult:
    """Result of h_eff parameter sweep."""

    h_eff_values: np.ndarray  # [K]
    var_re: np.ndarray  # [K]
    var_im: np.ndarray  # [K]
    variance_ratio: np.ndarray  # [K] = var_im / var_re
    mean_amplitude_ratio: np.ndarray  # [K] = |⟨Im⟩| / |⟨Re⟩|
    r_circ: np.ndarray  # [K]
    re_im_asymmetry: np.ndarray  # [K]
    local_phase_coherence: np.ndarray  # [K]
    scalar_mean: np.ndarray  # [K]
    pseudoscalar_mean: np.ndarray  # [K]
    # Autocorrelation times (1/e crossing of R(τ), ~ 1/m_eff)
    tau_re: np.ndarray  # [K] — τ_auto of ⟨Re(s)⟩ (scalar, VEV-subtracted)
    tau_im: np.ndarray  # [K] — τ_auto of ⟨Im(s)⟩ (pseudoscalar, already connected)
    tau_re_v: np.ndarray  # [K] — τ_auto of ⟨|v|·Re(s)⟩ (v-weighted scalar)
    tau_im_v: np.ndarray  # [K] — τ_auto of ⟨|v|·Im(s)⟩ (v-weighted pseudoscalar)
    tau_ratio: np.ndarray  # [K] = tau_im / tau_re (≈ m_σ/m_π)
    tau_ratio_v: np.ndarray  # [K] = tau_im_v / tau_re_v
    criterion: str
    criterion_values: np.ndarray  # [K] — values of selected criterion
    optimal_index: int
    optimal_h_eff: float
    optimal_criterion_value: float
    ell0_used: float
    n_frames: int


# ---------------------------------------------------------------------------
# Autocorrelation helpers
# ---------------------------------------------------------------------------


def _find_1e_crossing(R: np.ndarray) -> float:
    """Find τ where R(τ) first drops below 1/e, with linear interpolation."""
    max_lag = len(R) - 1
    threshold = 1.0 / np.e
    for lag in range(1, max_lag + 1):
        if R[lag] < threshold:
            r_prev = float(R[lag - 1])
            r_curr = float(R[lag])
            denom = r_prev - r_curr
            if denom > 1e-30:
                frac = (r_prev - threshold) / denom
                return float(lag - 1) + frac
            return float(lag)
    return float(max_lag)


def _compute_tau_auto(series: Tensor) -> float:
    """Compute τ_auto via 1/e crossing of the normalized autocorrelation R(τ).

    R(τ) = C(τ)/C(0), where C(τ) = ⟨(x(t)-μ)(x(t+τ)-μ)⟩_t.

    For nonzero-mean fields (Re channel), centering subtracts the VEV.

    Returns τ where R(τ) first drops below 1/e.  Uses FFT for efficiency.
    """
    T = series.numel()
    if T < 4:
        return float("nan")

    centered = (series - series.mean()).float()

    # FFT autocorrelation (zero-padded for linear, not circular, correlation)
    n_fft = 2 * T
    fft_s = torch.fft.rfft(centered, n=n_fft)
    acf = torch.fft.irfft(fft_s * fft_s.conj(), n=n_fft)  # [2T]

    c0 = acf[0].item()
    if c0 < 1e-30:
        return 0.0  # no variance → immediate decorrelation

    max_lag = T // 2
    R = acf[: max_lag + 1].cpu().numpy() / c0  # R[0] = 1.0
    return _find_1e_crossing(R)


def _compute_tau_auto_walkers(field: Tensor, valid: Tensor) -> float:
    """Compute τ_auto from per-walker autocorrelation, averaged over walkers.

    For zero-mean fields (Im channel), the spatial average cancels to ~0 at
    every frame because Im(⟨c_i|c_j⟩) ≈ -Im(⟨c_j|c_i⟩) when companion
    relationships are roughly symmetric.  The autocorrelation of the spatial
    average is therefore ~0, which is NOT the physical pseudoscalar correlator.

    Instead, compute the autocorrelation per-walker and average:
        C_avg(τ) = Σ_i C_i(τ)  /  Σ_i C_i(0)
    where C_i(τ) = Σ_t x_i(t) x_i(t+τ).  This is the correctly weighted
    average (walkers with larger fluctuations contribute more), equivalent to
    the lattice QCD all-to-all correlator.

    Args:
        field: [T, N] per-walker per-frame values (e.g. Im(s_field)).
        valid: [T, N] boolean mask.
    """
    T, N = field.shape
    if T < 4 or N == 0:
        return float("nan")

    # Zero out invalid entries and center each walker independently
    valid_f = valid.float()
    x = torch.where(valid, field, torch.zeros_like(field)).float()  # [T, N]
    n_valid = valid_f.sum(dim=0)  # [N] valid frames per walker
    walker_mean = (x * valid_f).sum(dim=0) / n_valid.clamp(min=1)  # [N]
    x_centered = (x - walker_mean.unsqueeze(0)) * valid_f  # [T, N]

    # Batch FFT autocorrelation over all walkers simultaneously
    # Transpose to [N, T] for FFT along time axis
    n_fft = 2 * T
    fft_s = torch.fft.rfft(x_centered.T, n=n_fft, dim=-1)  # [N, n_fft//2+1]
    acf_all = torch.fft.irfft(fft_s * fft_s.conj(), n=n_fft, dim=-1)  # [N, 2T]

    # C_i(0) = variance of walker i's time series
    c0 = acf_all[:, 0]  # [N]

    # Only keep walkers with enough valid frames and nonzero variance
    min_valid = max(T // 4, 4)
    good = (n_valid >= min_valid) & (c0 > 1e-30)

    if not good.any():
        return 0.0

    max_lag = T // 2

    # Weighted average: R_avg(τ) = Σ_i C_i(τ) / Σ_i C_i(0)
    C_sum = acf_all[good, : max_lag + 1].sum(dim=0)  # [max_lag+1]
    C0_sum = c0[good].sum().item()

    if C0_sum < 1e-30:
        return 0.0

    R = C_sum.cpu().numpy() / C0_sum  # R[0] = 1.0
    return _find_1e_crossing(R)


# ---------------------------------------------------------------------------
# Per-h_eff computation
# ---------------------------------------------------------------------------


def _compute_heff_point_stats(
    color: Tensor,
    color_valid: Tensor,
    pair_indices: Tensor,
    structural_valid: Tensor,
    scores: Tensor,
    pair_weighting: str,
    eps: float,
    v_mag: Tensor,
) -> tuple[float, ...]:
    """Compute per-h_eff inner product + s_field + stats + autocorrelation times.

    Args:
        v_mag: [T, N] velocity magnitude per walker per frame.

    Returns:
        (var_re, var_im, r_circ, re_im_asymmetry, local_phase_coherence,
         scalar_mean, pseudoscalar_mean,
         tau_re, tau_im, tau_re_v, tau_im_v)
    """
    # Gather color pairs
    color_j = torch.gather(
        color.unsqueeze(2).expand(-1, -1, pair_indices.shape[-1], -1),
        1,
        pair_indices.unsqueeze(-1).expand(-1, -1, -1, color.shape[-1]),
    )
    color_i = color.unsqueeze(2).expand_as(color_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1)
    finite_inner = torch.isfinite(inner.real) & torch.isfinite(inner.imag)

    valid = structural_valid & color_valid.unsqueeze(-1) & finite_inner
    if eps > 0:
        valid = valid & (inner.abs() > eps)

    weighting = str(pair_weighting).strip().lower()
    if weighting == "score_abs":
        score_j = torch.gather(scores, 1, pair_indices)
        score_i = scores.unsqueeze(-1).expand_as(score_j)
        finite_scores = torch.isfinite(score_i) & torch.isfinite(score_j)
        valid = valid & finite_scores
        base_weights = (score_j - score_i).abs()
    else:
        base_weights = torch.ones_like(inner.real)

    pair_weights = torch.where(valid, base_weights, torch.zeros_like(base_weights))
    row_sum = pair_weights.sum(dim=-1, keepdim=True)
    row_has_pairs = row_sum.squeeze(-1) > 0
    row_weights = torch.where(
        row_sum > 0,
        pair_weights / row_sum.clamp(min=1e-12),
        torch.zeros_like(pair_weights),
    )

    s_field = (row_weights.to(inner.dtype) * inner).sum(dim=-1)  # [T, N]
    walker_valid = color_valid & row_has_pairs

    # Phase concentration (R_circ)
    phase = torch.angle(s_field)
    phase_vector = torch.where(
        walker_valid,
        torch.exp(1j * phase).to(dtype=s_field.dtype),
        torch.zeros_like(s_field),
    )
    walker_count = walker_valid.sum(dim=1).clamp(min=1)
    phase_mean_complex = phase_vector.sum(dim=1) / walker_count.to(s_field.dtype)
    r_circ_val = torch.abs(phase_mean_complex).float().mean().item()

    # Variance statistics
    re_part = s_field.real.float()  # [T, N]
    im_part = s_field.imag.float()  # [T, N]
    var_re_t = _masked_var(re_part, walker_valid)
    var_im_t = _masked_var(im_part, walker_valid)
    var_re_val = float(var_re_t.mean().item())
    var_im_val = float(var_im_t.mean().item())
    re_im_asym_t = (var_re_t - var_im_t).abs() / (var_re_t + var_im_t + 1e-12)
    re_im_asym_val = float(re_im_asym_t.mean().item())

    # Local phase coherence
    unit_inner = inner / inner.abs().clamp(min=max(eps, 1e-12))
    local_phase = (row_weights.to(inner.dtype) * unit_inner).sum(dim=-1)
    lpc_val = float(_masked_mean(local_phase.abs().float(), walker_valid).mean().item())

    # Spatial averages for scalar/pseudoscalar mean values
    re_mean_t = _masked_mean(re_part, walker_valid)  # [T]
    im_mean_t = _masked_mean(im_part, walker_valid)  # [T]

    scalar_val = float(re_mean_t.mean().item())
    pseudoscalar_val = float(im_mean_t.mean().item())

    # Autocorrelation times (1/e crossing of R(τ))
    # All channels use per-walker autocorrelation averaged over walkers
    # so that ratios τ_Im/τ_Re are computed on equal footing.
    tau_re = _compute_tau_auto_walkers(re_part, walker_valid)
    tau_im = _compute_tau_auto_walkers(im_part, walker_valid)
    tau_re_v = _compute_tau_auto_walkers(v_mag * re_part, walker_valid)
    tau_im_v = _compute_tau_auto_walkers(v_mag * im_part, walker_valid)

    return (
        var_re_val, var_im_val, r_circ_val, re_im_asym_val, lpc_val,
        scalar_val, pseudoscalar_val,
        tau_re, tau_im, tau_re_v, tau_im_v,
    )


def compute_heff_sweep(history: RunHistory, config: HeffSweepConfig) -> HeffSweepResult:
    """Sweep h_eff values and compute operator variance ratio at each point.

    Pre-computes h_eff-independent data once, then loops over h_eff values
    with only the phase division changing.
    """
    # --- h_eff grid ---
    if config.log_scale:
        h_eff_values = np.logspace(
            np.log10(config.h_eff_min),
            np.log10(config.h_eff_max),
            config.n_points,
        )
    else:
        h_eff_values = np.linspace(config.h_eff_min, config.h_eff_max, config.n_points)

    # --- Frame indices (h_eff-independent) ---
    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=config.warmup_fraction,
        end_fraction=config.end_fraction,
    )
    if not frame_indices:
        return _empty_sweep_result(h_eff_values, criterion=config.criterion)

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    n_frames = len(frame_indices)

    # --- ell0 (h_eff-independent) ---
    ell0 = (
        float(config.ell0)
        if config.ell0 is not None
        else float(estimate_ell0_auto(history, config.ell0_method))
    )

    # --- Pre-compute h_eff-independent tensors ---
    v_pre = history.v_before_clone[start_idx:end_idx]  # [T, N, d]
    force_visc = history.force_viscous[start_idx - 1 : end_idx - 1]  # [T, N, d]
    mass = float(max(config.mass, 1e-8))
    base_phase = mass * v_pre * ell0  # [T, N, d] — constant across h_eff

    # Velocity magnitude (h_eff-independent) for velocity-weighted channels
    v_mag = torch.linalg.vector_norm(v_pre, dim=-1).float()  # [T, N]

    # Companion topology (h_eff-independent)
    comp_d = history.companions_distance[start_idx - 1 : end_idx - 1].to(
        dtype=torch.long, device=v_pre.device
    )
    comp_c = history.companions_clone[start_idx - 1 : end_idx - 1].to(
        dtype=torch.long, device=v_pre.device
    )
    scores = history.cloning_scores[start_idx - 1 : end_idx - 1].to(
        dtype=torch.float32, device=v_pre.device
    )

    topology = str(config.companion_topology).strip().lower()
    if topology == "distance":
        pair_indices = comp_d.unsqueeze(-1)
    elif topology == "clone":
        pair_indices = comp_c.unsqueeze(-1)
    else:
        pair_indices = torch.stack([comp_d, comp_c], dim=-1)

    n_walkers = v_pre.shape[1]
    src_idx = torch.arange(n_walkers, device=v_pre.device, dtype=torch.long).view(1, n_walkers, 1)
    structural_valid = pair_indices != src_idx

    eps = float(max(config.eps, 0.0))

    # --- Allocate output arrays ---
    k = len(h_eff_values)
    var_re_arr = np.zeros(k)
    var_im_arr = np.zeros(k)
    r_circ_arr = np.zeros(k)
    re_im_asym_arr = np.zeros(k)
    lpc_arr = np.zeros(k)
    scalar_arr = np.zeros(k)
    pseudoscalar_arr = np.zeros(k)
    tau_re_arr = np.full(k, np.nan)
    tau_im_arr = np.full(k, np.nan)
    tau_re_v_arr = np.full(k, np.nan)
    tau_im_v_arr = np.full(k, np.nan)

    # --- Sweep loop ---
    for i, h_eff in enumerate(h_eff_values):
        h_eff_clamped = float(max(h_eff, 1e-8))
        phase = base_phase / h_eff_clamped  # only this division changes
        complex_phase = torch.polar(torch.ones_like(phase), phase.float())

        if force_visc.dtype == torch.float64:
            complex_dtype = torch.complex128
        else:
            complex_dtype = torch.complex64

        tilde = force_visc.to(complex_dtype) * complex_phase.to(complex_dtype)
        norm = torch.linalg.vector_norm(tilde, dim=-1, keepdim=True).clamp(min=1e-12)
        color = tilde / norm
        color_valid = norm.squeeze(-1) > 1e-12

        # Apply color_dims projection if needed
        if config.color_dims is not None:
            dims = list(config.color_dims)
            color = color[..., dims]
            proj_norm = torch.linalg.vector_norm(color, dim=-1, keepdim=True).clamp(min=1e-12)
            color = color / proj_norm
            color_valid = color_valid & (proj_norm.squeeze(-1) > 1e-12)

        stats = _compute_heff_point_stats(
            color=color,
            color_valid=color_valid,
            pair_indices=pair_indices,
            structural_valid=structural_valid,
            scores=scores,
            pair_weighting=config.pair_weighting,
            eps=eps,
            v_mag=v_mag,
        )
        var_re_arr[i] = stats[0]
        var_im_arr[i] = stats[1]
        r_circ_arr[i] = stats[2]
        re_im_asym_arr[i] = stats[3]
        lpc_arr[i] = stats[4]
        scalar_arr[i] = stats[5]
        pseudoscalar_arr[i] = stats[6]
        tau_re_arr[i] = stats[7]
        tau_im_arr[i] = stats[8]
        tau_re_v_arr[i] = stats[9]
        tau_im_v_arr[i] = stats[10]

    # --- Derived ratios ---
    variance_ratio = var_im_arr / np.maximum(var_re_arr, 1e-30)
    mean_amplitude_ratio = np.abs(pseudoscalar_arr) / (np.abs(scalar_arr) + 1e-30)
    tau_ratio = tau_im_arr / np.where(np.abs(tau_re_arr) > 1e-30, tau_re_arr, np.nan)
    tau_ratio_v = tau_im_v_arr / np.where(np.abs(tau_re_v_arr) > 1e-30, tau_re_v_arr, np.nan)

    # --- Select criterion for optimality ---
    criterion = str(config.criterion).strip().lower()
    if criterion == "mean_amplitude_ratio":
        criterion_values = mean_amplitude_ratio
    elif criterion == "r_circ":
        criterion_values = r_circ_arr
    else:
        criterion = "variance_ratio"
        criterion_values = variance_ratio

    finite_mask = np.isfinite(criterion_values)
    if finite_mask.any():
        optimal_index = int(np.argmax(np.where(finite_mask, criterion_values, -np.inf)))
    else:
        optimal_index = 0

    return HeffSweepResult(
        h_eff_values=h_eff_values,
        var_re=var_re_arr,
        var_im=var_im_arr,
        variance_ratio=variance_ratio,
        mean_amplitude_ratio=mean_amplitude_ratio,
        r_circ=r_circ_arr,
        re_im_asymmetry=re_im_asym_arr,
        local_phase_coherence=lpc_arr,
        scalar_mean=scalar_arr,
        pseudoscalar_mean=pseudoscalar_arr,
        tau_re=tau_re_arr,
        tau_im=tau_im_arr,
        tau_re_v=tau_re_v_arr,
        tau_im_v=tau_im_v_arr,
        tau_ratio=tau_ratio,
        tau_ratio_v=tau_ratio_v,
        criterion=criterion,
        criterion_values=criterion_values,
        optimal_index=optimal_index,
        optimal_h_eff=float(h_eff_values[optimal_index]),
        optimal_criterion_value=float(criterion_values[optimal_index]),
        ell0_used=ell0,
        n_frames=n_frames,
    )


def _empty_sweep_result(
    h_eff_values: np.ndarray,
    criterion: str = "variance_ratio",
) -> HeffSweepResult:
    k = len(h_eff_values)
    zeros = np.zeros(k)
    nans = np.full(k, np.nan)
    return HeffSweepResult(
        h_eff_values=h_eff_values,
        var_re=zeros.copy(),
        var_im=zeros.copy(),
        variance_ratio=zeros.copy(),
        mean_amplitude_ratio=zeros.copy(),
        r_circ=zeros.copy(),
        re_im_asymmetry=zeros.copy(),
        local_phase_coherence=zeros.copy(),
        scalar_mean=zeros.copy(),
        pseudoscalar_mean=zeros.copy(),
        tau_re=nans.copy(),
        tau_im=nans.copy(),
        tau_re_v=nans.copy(),
        tau_im_v=nans.copy(),
        tau_ratio=nans.copy(),
        tau_ratio_v=nans.copy(),
        criterion=criterion,
        criterion_values=zeros.copy(),
        optimal_index=0,
        optimal_h_eff=float(h_eff_values[0]),
        optimal_criterion_value=0.0,
        ell0_used=1.0,
        n_frames=0,
    )


# ---------------------------------------------------------------------------
# Plot / table builders
# ---------------------------------------------------------------------------


def build_heff_sweep_primary_plot(result: HeffSweepResult) -> Any:
    """Selected criterion vs h_eff (log x) with optimal VLine."""
    label = CRITERION_LABELS.get(result.criterion, result.criterion)
    df = pd.DataFrame({
        "h_eff": result.h_eff_values,
        "value": result.criterion_values,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        return hv.Text(0, 0, "No sweep data available").opts(
            width=960, height=320, toolbar=None
        )

    curve = hv.Curve(df, "h_eff", "value").relabel(label).opts(
        color="#4c78a8",
        line_width=2,
        tools=["hover"],
        logx=True,
    )
    vline = hv.VLine(result.optimal_h_eff).opts(
        color="#e45756",
        line_dash="dashed",
        line_width=2,
    )
    return (curve * vline).opts(
        title=f"h_eff Sweep: {label} (maximize)",
        xlabel="h_eff",
        ylabel=label,
        width=960,
        height=320,
        show_grid=True,
    )


def build_heff_sweep_secondary_plot(result: HeffSweepResult) -> Any:
    """R_circ + re_im_asymmetry + local_phase_coherence vs h_eff."""
    df = pd.DataFrame({
        "h_eff": result.h_eff_values,
        "R_circ": result.r_circ,
        "Re/Im asymmetry": result.re_im_asymmetry,
        "local coherence": result.local_phase_coherence,
    })

    curves: list[Any] = []
    for col, color in [
        ("R_circ", "#54a24b"),
        ("Re/Im asymmetry", "#e45756"),
        ("local coherence", "#72b7b2"),
    ]:
        sub = df[["h_eff", col]].rename(columns={col: "value"})
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if not sub.empty:
            curves.append(
                hv.Curve(sub, "h_eff", "value").relabel(col).opts(
                    color=color, line_width=2, tools=["hover"], logx=True,
                )
            )

    if not curves:
        return hv.Text(0, 0, "No sweep data available").opts(
            width=960, height=320, toolbar=None
        )

    plot = curves[0]
    for c in curves[1:]:
        plot = plot * c
    vline = hv.VLine(result.optimal_h_eff).opts(
        color="#e45756", line_dash="dashed", line_width=1,
    )
    return (plot * vline).opts(
        title="h_eff Sweep: Regime Metrics",
        xlabel="h_eff",
        ylabel="dimensionless",
        width=960,
        height=320,
        show_grid=True,
        legend_position="top_left",
    )


def build_heff_sweep_acf_plot(result: HeffSweepResult) -> Any:
    """Autocorrelation times (τ_auto) of 4 channels vs h_eff."""
    series_spec = [
        ("Re(s)", result.tau_re, "#4c78a8"),
        ("Im(s)", result.tau_im, "#e45756"),
        ("|v|Re(s)", result.tau_re_v, "#54a24b"),
        ("|v|Im(s)", result.tau_im_v, "#b279a2"),
    ]
    curves: list[Any] = []
    for label, values, color in series_spec:
        sub = pd.DataFrame({
            "h_eff": result.h_eff_values, "value": values,
        }).replace([np.inf, -np.inf], np.nan).dropna()
        if not sub.empty:
            curves.append(
                hv.Curve(sub, "h_eff", "value").relabel(label).opts(
                    color=color, line_width=2, tools=["hover"], logx=True,
                )
            )
    if not curves:
        return hv.Text(0, 0, "No autocorrelation data available").opts(
            width=960, height=320, toolbar=None,
        )
    plot = curves[0]
    for c in curves[1:]:
        plot = plot * c
    vline = hv.VLine(result.optimal_h_eff).opts(
        color="#e45756", line_dash="dashed", line_width=1,
    )
    return (plot * vline).opts(
        title="h_eff Sweep: Autocorrelation Time (1/e crossing, ~ 1/m_eff)",
        xlabel="h_eff",
        ylabel="tau_auto [frames]",
        width=960,
        height=320,
        show_grid=True,
        legend_position="top_left",
    )


def build_heff_sweep_ratios_plot(result: HeffSweepResult) -> Any:
    """All Im/Re ratios (variance, mean amplitude, tau, tau_v) vs h_eff."""
    series_spec = [
        ("Var(Im)/Var(Re)", result.variance_ratio, "#4c78a8"),
        ("|mean_Im|/|mean_Re|", result.mean_amplitude_ratio, "#f58518"),
        ("tau_Im/tau_Re", result.tau_ratio, "#e45756"),
        ("tau_Im_v/tau_Re_v", result.tau_ratio_v, "#54a24b"),
    ]
    curves: list[Any] = []
    for label, values, color in series_spec:
        sub = pd.DataFrame({
            "h_eff": result.h_eff_values, "value": values,
        }).replace([np.inf, -np.inf], np.nan).dropna()
        if not sub.empty:
            curves.append(
                hv.Curve(sub, "h_eff", "value").relabel(label).opts(
                    color=color, line_width=2, tools=["hover"], logx=True,
                )
            )
    if not curves:
        return hv.Text(0, 0, "No ratio data available").opts(
            width=960, height=320, toolbar=None,
        )
    plot = curves[0]
    for c in curves[1:]:
        plot = plot * c
    vline = hv.VLine(result.optimal_h_eff).opts(
        color="#e45756", line_dash="dashed", line_width=1,
    )
    return (plot * vline).opts(
        title="h_eff Sweep: All Im/Re Channel Ratios",
        xlabel="h_eff",
        ylabel="ratio",
        width=960,
        height=320,
        show_grid=True,
        legend_position="top_left",
    )


def build_heff_sweep_table(result: HeffSweepResult) -> pd.DataFrame:
    """Build sweep data table."""
    return pd.DataFrame({
        "h_eff": result.h_eff_values,
        "var_re": result.var_re,
        "var_im": result.var_im,
        "variance_ratio": result.variance_ratio,
        "mean_amplitude_ratio": result.mean_amplitude_ratio,
        "r_circ": result.r_circ,
        "re_im_asymmetry": result.re_im_asymmetry,
        "local_phase_coherence": result.local_phase_coherence,
        "scalar_mean": result.scalar_mean,
        "pseudoscalar_mean": result.pseudoscalar_mean,
        "tau_re": result.tau_re,
        "tau_im": result.tau_im,
        "tau_re_v": result.tau_re_v,
        "tau_im_v": result.tau_im_v,
        "tau_ratio": result.tau_ratio,
        "tau_ratio_v": result.tau_ratio_v,
    }).replace([np.inf, -np.inf], np.nan)


def _fmt(val: float, digits: int = 6) -> str:
    if not np.isfinite(val):
        return "n/a"
    return f"{val:.{digits}g}"


def build_heff_sweep_summary_text(result: HeffSweepResult) -> str:
    """Build sweep summary markdown."""
    label = CRITERION_LABELS.get(result.criterion, result.criterion)
    idx = result.optimal_index
    lines = [
        "**h_eff Sweep Results:**",
        f"- Criterion: **{label}** (maximize)",
        f"- Optimal h_eff: `{result.optimal_h_eff:.6g}`",
        f"- Optimal {label}: `{result.optimal_criterion_value:.6g}`",
        f"- Sweep range: `[{result.h_eff_values[0]:.4g}, {result.h_eff_values[-1]:.4g}]`"
        f" ({len(result.h_eff_values)} points)",
        f"- ell0 used: `{result.ell0_used:.6g}`",
        f"- Frames analyzed: `{result.n_frames}`",
        "",
        "**Autocorrelation times at optimal h_eff** (τ_auto ~ 1/m_eff, 1/e crossing):",
        f"- Re(s): `{_fmt(result.tau_re[idx])}` frames"
        f" | Im(s): `{_fmt(result.tau_im[idx])}` frames"
        f" | τ_Im/τ_Re: `{_fmt(result.tau_ratio[idx])}`",
        f"- |v|Re(s): `{_fmt(result.tau_re_v[idx])}` frames"
        f" | |v|Im(s): `{_fmt(result.tau_im_v[idx])}` frames"
        f" | τ_Im_v/τ_Re_v: `{_fmt(result.tau_ratio_v[idx])}`",
    ]
    return "\n".join(lines)
