"""GEVP utilities for companion strong-force channel analysis.

This module implements a batched PyTorch generalized eigenvalue workflow for
multi-operator correlator matrices, with basis pruning and bootstrap support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from fragile.fractalai.qft.correlator_channels import (
    ChannelCorrelatorResult,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)


GEVP_BASIS_STRATEGIES = ("base_only", "base_plus_best_scale")
GEVP_BOOTSTRAP_MODES = ("time", "walker", "hybrid")
NUCLEON_GEVP_BASE_CHANNELS = (
    "nucleon",
    "nucleon_flux_action",
    "nucleon_flux_sin2",
    "nucleon_flux_exp",
    "nucleon_score_abs",
)
SCALAR_GEVP_BASE_CHANNELS = (
    "scalar",
    "scalar_score_directed",
    "scalar_score_weighted",
    "scalar_raw",
    "scalar_abs2_vacsub",
)
PSEUDOSCALAR_GEVP_BASE_CHANNELS = (
    "pseudoscalar",
    "pseudoscalar_score_directed",
    "pseudoscalar_score_weighted",
)
GLUEBALL_GEVP_BASE_CHANNELS = (
    "glueball",
    "glueball_phase_action",
    "glueball_phase_sin2",
)
SU2_GEVP_BASE_CHANNELS = (
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
)
COMPANION_GEVP_BASE_CHANNELS: dict[str, tuple[str, ...]] = {
    "nucleon": NUCLEON_GEVP_BASE_CHANNELS,
    "scalar": SCALAR_GEVP_BASE_CHANNELS,
    "pseudoscalar": PSEUDOSCALAR_GEVP_BASE_CHANNELS,
    "glueball": GLUEBALL_GEVP_BASE_CHANNELS,
    "su2": SU2_GEVP_BASE_CHANNELS,
}


@dataclass
class GEVPConfig:
    """Configuration for batched GEVP analysis."""

    t0: int = 2
    max_lag: int = 40
    use_connected: bool = True
    fit_mode: str = "aic"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2
    window_widths: list[int] | None = None
    basis_strategy: str = "base_plus_best_scale"
    max_basis: int = 10
    min_operator_r2: float = -1.0
    min_operator_windows: int = 0
    max_operator_error_pct: float = 30.0
    remove_artifacts: bool = False
    eig_rel_cutoff: float = 1e-2
    cond_limit: float = 1e4
    shrinkage: float = 1e-6
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100
    bootstrap_seed: int = 12345
    bootstrap_mode: str = "time"


@dataclass
class GEVPResult:
    """Computed GEVP payload and diagnostics."""

    result: ChannelCorrelatorResult
    basis_labels: list[str]
    kept_basis_labels: list[str]
    dropped_basis_labels: list[str]
    cond_c0: float
    bootstrap_mode_applied: str
    notes: list[str]


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
    return torch.where(count > 0, std, torch.full_like(std, float("nan")))


def _fft_cross_correlator_lags(
    series: Tensor,
    *,
    max_lag: int,
    use_connected: bool,
) -> Tensor:
    """Compute cross-correlator matrices C_ij(t) from operator series.

    Args:
        series: Operator series [K, T].
        max_lag: Maximum lag to return.
        use_connected: Subtract per-operator mean before correlation.

    Returns:
        Tensor [L, K, K], where L=max_lag+1.
    """
    if series.ndim != 2:
        raise ValueError(f"Expected series [K,T], got {tuple(series.shape)}")
    k_count, t_len = series.shape
    if k_count <= 0 or t_len <= 0:
        return torch.zeros((int(max_lag) + 1, k_count, k_count), dtype=torch.float32, device=series.device)

    work = series.float()
    if use_connected:
        work = work - work.mean(dim=1, keepdim=True)

    padded = F.pad(work, (0, t_len))
    fft_s = torch.fft.fft(padded, dim=1)
    cross_spec = fft_s.unsqueeze(1) * fft_s.conj().unsqueeze(0)
    corr = torch.fft.ifft(cross_spec, dim=-1).real

    effective_lag = min(int(max_lag), t_len - 1)
    counts = torch.arange(
        t_len,
        t_len - effective_lag - 1,
        -1,
        dtype=torch.float32,
        device=series.device,
    )
    out = corr[..., : effective_lag + 1] / counts.view(1, 1, -1)
    if effective_lag < int(max_lag):
        out = F.pad(out, (0, int(max_lag) - effective_lag), value=0.0)

    out = out.permute(2, 0, 1).contiguous()
    return 0.5 * (out + out.transpose(-1, -2))


def _fft_cross_correlator_lags_batched(
    series: Tensor,
    *,
    max_lag: int,
    use_connected: bool,
) -> Tensor:
    """Compute cross-correlator matrices for a bootstrap batch.

    Args:
        series: Operator series [B, K, T].
        max_lag: Maximum lag to return.
        use_connected: Subtract per-sample, per-operator mean.

    Returns:
        Tensor [B, L, K, K], where L=max_lag+1.
    """
    if series.ndim != 3:
        raise ValueError(f"Expected series [B,K,T], got {tuple(series.shape)}")
    b_count, k_count, t_len = series.shape
    if b_count <= 0 or k_count <= 0 or t_len <= 0:
        return torch.zeros(
            (b_count, int(max_lag) + 1, k_count, k_count),
            dtype=torch.float32,
            device=series.device,
        )

    work = series.float()
    if use_connected:
        work = work - work.mean(dim=2, keepdim=True)

    padded = F.pad(work, (0, t_len))
    fft_s = torch.fft.fft(padded, dim=2)
    cross_spec = fft_s.unsqueeze(2) * fft_s.conj().unsqueeze(1)
    corr = torch.fft.ifft(cross_spec, dim=-1).real

    effective_lag = min(int(max_lag), t_len - 1)
    counts = torch.arange(
        t_len,
        t_len - effective_lag - 1,
        -1,
        dtype=torch.float32,
        device=series.device,
    )
    out = corr[..., : effective_lag + 1] / counts.view(1, 1, 1, -1)
    if effective_lag < int(max_lag):
        out = F.pad(out, (0, int(max_lag) - effective_lag), value=0.0)

    out = out.permute(0, 3, 1, 2).contiguous()
    return 0.5 * (out + out.transpose(-1, -2))


def _fit_mass_from_correlator(
    correlator: Tensor,
    *,
    dt: float,
    config: GEVPConfig,
) -> tuple[dict[str, Any], Tensor | None, Tensor | None, list[int] | None, Tensor | None]:
    fit_cfg = CorrelatorConfig(
        max_lag=max(0, int(correlator.numel()) - 1),
        use_connected=False,
        window_widths=config.window_widths,
        fit_mode=str(config.fit_mode),
        fit_start=max(0, int(config.fit_start)),
        fit_stop=config.fit_stop,
        min_fit_points=max(2, int(config.min_fit_points)),
        compute_bootstrap_errors=False,
        n_bootstrap=0,
    )

    if str(config.fit_mode) == "linear_abs":
        fit = extract_mass_linear(correlator.abs(), dt, fit_cfg)
        return fit, None, None, None, None
    if str(config.fit_mode) == "linear":
        fit = extract_mass_linear(correlator, dt, fit_cfg)
        return fit, None, None, None, None

    fit = extract_mass_aic(correlator, dt, fit_cfg)
    return (
        fit,
        fit.pop("window_masses", None),
        fit.pop("window_aic", None),
        fit.pop("window_widths", None),
        fit.pop("window_r2", None),
    )


def _mass_from_fit_mode(correlator: Tensor, *, dt: float, config: GEVPConfig) -> float:
    fit, _, _, _, _ = _fit_mass_from_correlator(correlator, dt=dt, config=config)
    mass = float(fit.get("mass", float("nan")))
    if not math.isfinite(mass) or mass <= 0:
        return float("nan")
    return mass


def _sanitize_mode(mode: str, allowed: tuple[str, ...], fallback: str) -> str:
    value = str(mode).strip().lower()
    return value if value in allowed else fallback


def get_companion_gevp_basis_channels(base_channel: str) -> tuple[str, ...]:
    """Return ordered companion-operator family used to build a GEVP basis."""
    key = str(base_channel).strip().lower()
    channels = COMPANION_GEVP_BASE_CHANNELS.get(key)
    if channels is None:
        supported = ", ".join(sorted(COMPANION_GEVP_BASE_CHANNELS))
        raise ValueError(
            f"Unsupported companion GEVP base channel '{base_channel}'. Supported: {supported}."
        )
    return channels


def _extract_operator_r2(result: ChannelCorrelatorResult) -> float:
    mass_fit = getattr(result, "mass_fit", None)
    if isinstance(mass_fit, dict):
        try:
            return float(mass_fit.get("r_squared", float("nan")))
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _extract_operator_n_windows(result: ChannelCorrelatorResult) -> int:
    mass_fit = getattr(result, "mass_fit", None)
    if isinstance(mass_fit, dict):
        raw = mass_fit.get("n_valid_windows", None)
        if raw is not None:
            try:
                n_valid = int(raw)
                return max(0, n_valid)
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


def _extract_operator_mass_error(result: ChannelCorrelatorResult) -> float:
    mass_fit = getattr(result, "mass_fit", None)
    if isinstance(mass_fit, dict):
        try:
            return float(mass_fit.get("mass_error", float("nan")))
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _extract_operator_mass(result: ChannelCorrelatorResult) -> float:
    mass_fit = getattr(result, "mass_fit", None)
    if isinstance(mass_fit, dict):
        try:
            return float(mass_fit.get("mass", float("nan")))
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _operator_filter_reason(
    result: ChannelCorrelatorResult,
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
) -> str | None:
    reasons: list[str] = []
    r2 = _extract_operator_r2(result)
    n_windows = _extract_operator_n_windows(result)
    mass = _extract_operator_mass(result)
    mass_error = _extract_operator_mass_error(result)

    if math.isfinite(min_r2):
        if not math.isfinite(r2) or r2 < min_r2:
            r2_text = "nan" if not math.isfinite(r2) else f"{r2:.3g}"
            reasons.append(f"r2={r2_text}<{min_r2:.3g}")
    if n_windows < min_windows:
        reasons.append(f"n_windows={n_windows}<{min_windows}")
    if math.isfinite(max_error_pct) and max_error_pct >= 0:
        if math.isfinite(mass) and mass > 0 and math.isfinite(mass_error) and mass_error >= 0:
            err_pct = abs(mass_error / mass) * 100.0
        else:
            err_pct = float("inf")
        if err_pct > max_error_pct:
            err_text = f"{err_pct:.3g}" if math.isfinite(err_pct) else "inf"
            reasons.append(f"err_pct={err_text}>{max_error_pct:.3g}")
    if remove_artifacts:
        if not math.isfinite(mass_error):
            reasons.append("mass_error=nan_or_inf")
        elif mass_error == 0.0:
            reasons.append("mass_error==0")
        if math.isfinite(mass) and mass == 0.0:
            reasons.append("mass==0")

    if reasons:
        return ", ".join(reasons)
    return None


def _build_basis_from_results(
    *,
    base_results: dict[str, ChannelCorrelatorResult],
    multiscale_output: Any | None,
    config: GEVPConfig,
    base_channel: str,
    basis_channels: tuple[str, ...],
) -> tuple[Tensor, list[str], list[str], list[str], list[str]]:
    """Build [K,T] basis matrix from base + optional best multiscale operators."""
    notes: list[str] = []
    basis_entries: list[tuple[str, Tensor]] = []
    min_r2 = float(config.min_operator_r2)
    min_windows = max(0, int(config.min_operator_windows))
    max_error_pct = float(config.max_operator_error_pct)
    remove_artifacts = bool(config.remove_artifacts)
    filtered_out: list[tuple[str, str]] = []

    for channel in basis_channels:
        result = base_results.get(channel)
        if result is None or result.n_samples <= 0 or result.series.numel() <= 0:
            continue
        filter_reason = _operator_filter_reason(
            result,
            min_r2=min_r2,
            min_windows=min_windows,
            max_error_pct=max_error_pct,
            remove_artifacts=remove_artifacts,
        )
        if filter_reason is not None:
            filtered_out.append((channel, filter_reason))
            continue
        basis_entries.append((channel, result.series.float().detach()))

    strategy = _sanitize_mode(str(config.basis_strategy), GEVP_BASIS_STRATEGIES, "base_only")
    if strategy == "base_plus_best_scale" and multiscale_output is not None:
        for channel in basis_channels:
            companion_name = f"{channel}_companion"
            best_idx = int(getattr(multiscale_output, "best_scale_index", {}).get(companion_name, -1))
            per_scale = getattr(multiscale_output, "per_scale_results", {}).get(companion_name, None)
            if not isinstance(per_scale, list):
                continue
            if best_idx < 0 or best_idx >= len(per_scale):
                continue
            best_result = per_scale[best_idx]
            if best_result is None or best_result.n_samples <= 0 or best_result.series.numel() <= 0:
                continue
            label = f"{channel}@best_scale[{best_idx}]"
            filter_reason = _operator_filter_reason(
                best_result,
                min_r2=min_r2,
                min_windows=min_windows,
                max_error_pct=max_error_pct,
                remove_artifacts=remove_artifacts,
            )
            if filter_reason is not None:
                filtered_out.append((label, filter_reason))
                continue
            basis_entries.append((label, best_result.series.float().detach()))

    if not basis_entries:
        if filtered_out:
            preview = ", ".join(
                f"{label}({reason})" for label, reason in filtered_out[:5]
            )
            suffix = " ..." if len(filtered_out) > 5 else ""
            msg = (
                f"No valid {base_channel} operator series found for GEVP basis after operator quality "
                f"filters (min_r2={min_r2:.3g}, min_windows={min_windows}, "
                f"max_error_pct={max_error_pct:.3g}, "
                f"remove_artifacts={remove_artifacts}). "
                f"Filtered: {preview}{suffix}"
            )
        else:
            msg = f"No valid {base_channel} operator series found for GEVP basis."
        raise ValueError(msg)

    if filtered_out:
        preview = ", ".join(
            f"{label}({reason})" for label, reason in filtered_out[:5]
        )
        suffix = " ..." if len(filtered_out) > 5 else ""
        notes.append(
            "Operator quality filter excluded "
            f"{len(filtered_out)} basis vectors (min_r2={min_r2:.3g}, min_windows={min_windows}, "
            f"max_error_pct={max_error_pct:.3g}, "
            f"remove_artifacts={remove_artifacts}): "
            f"{preview}{suffix}"
        )

    # Deduplicate labels while preserving order.
    dedup: dict[str, Tensor] = {}
    for label, series in basis_entries:
        if label not in dedup:
            dedup[label] = series
    basis_entries = list(dedup.items())

    input_labels = [label for label, _ in basis_entries]

    # Keep base operators first, then scale-augmented operators.
    base_first = [item for item in basis_entries if "@best_scale" not in item[0]]
    scales = [item for item in basis_entries if "@best_scale" in item[0]]
    ordered = base_first + scales

    max_basis = max(1, int(config.max_basis))
    if len(ordered) > max_basis:
        notes.append(f"Basis capped to {max_basis} vectors (from {len(ordered)}).")
        ordered = ordered[:max_basis]

    lengths = [int(series.numel()) for _, series in ordered]
    t_min = min(lengths)
    if t_min <= max(2, int(config.t0) + 2):
        raise ValueError(
            "Insufficient time samples for GEVP after basis alignment: "
            f"T={t_min}, t0={int(config.t0)}."
        )
    if len(set(lengths)) > 1:
        notes.append(f"Series length mismatch detected; trimming basis to T={t_min}.")

    aligned = []
    kept_input_labels: list[str] = []
    for label, series in ordered:
        aligned.append(series[:t_min])
        kept_input_labels.append(label)

    dropped_pre_prune = [label for label in input_labels if label not in kept_input_labels]
    basis = torch.stack(aligned, dim=0)
    return basis, input_labels, kept_input_labels, dropped_pre_prune, notes


def _build_whitener(
    c0: Tensor,
    *,
    eig_rel_cutoff: float,
    cond_limit: float,
    shrinkage: float,
) -> tuple[Tensor, Tensor, float]:
    """Construct whitening matrix W from C(t0) with pruning."""
    if c0.ndim != 2 or c0.shape[0] != c0.shape[1]:
        raise ValueError(f"Expected square C(t0), got {tuple(c0.shape)}")

    k_count = c0.shape[0]
    c0_sym = 0.5 * (c0 + c0.transpose(-1, -2))
    trace_scale = float(torch.trace(c0_sym).item()) / max(1, k_count)
    if not math.isfinite(trace_scale) or trace_scale <= 0:
        trace_scale = 1.0
    c0_reg = c0_sym + float(shrinkage) * trace_scale * torch.eye(
        k_count,
        dtype=c0_sym.dtype,
        device=c0_sym.device,
    )

    evals, evecs = torch.linalg.eigh(c0_reg)
    evals = evals.real
    order = torch.argsort(evals, descending=True)
    evals_sorted = evals[order]
    positive = evals_sorted > 0
    if not torch.any(positive):
        msg = "C(t0) is not positive definite after shrinkage."
        raise ValueError(msg)
    evals_pos = evals_sorted[positive]
    order_pos = order[positive]

    max_eval = float(evals_pos[0].item())
    rel_floor = max_eval * max(0.0, float(eig_rel_cutoff))
    keep_count = int((evals_pos >= rel_floor).sum().item())
    keep_count = max(1, keep_count)

    cond_target = max(1.0, float(cond_limit))
    while keep_count > 1:
        min_eval = float(evals_pos[keep_count - 1].item())
        cond = max_eval / max(min_eval, 1e-12)
        if cond <= cond_target:
            break
        keep_count -= 1

    keep_idx = order_pos[:keep_count]
    keep_evals = evals[keep_idx]
    keep_evecs = evecs[:, keep_idx]

    whitener = keep_evecs / torch.sqrt(torch.clamp_min(keep_evals, 1e-12)).unsqueeze(0)
    cond_c0 = float(keep_evals.max().item() / max(float(keep_evals.min().item()), 1e-12))
    return whitener, keep_idx, cond_c0


def _project_and_extract_principal(c_lags: Tensor, whitener: Tensor) -> Tensor:
    """Project C(t) with whitener and extract largest eigenvalue per lag."""
    c_proj = torch.einsum("ki,tkj,jm->tim", whitener, c_lags, whitener)
    c_proj = 0.5 * (c_proj + c_proj.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(c_proj)
    return eigvals[:, -1].real.float()


def compute_companion_channel_gevp(
    *,
    base_results: dict[str, ChannelCorrelatorResult],
    multiscale_output: Any | None,
    config: GEVPConfig,
    base_channel: str,
) -> GEVPResult:
    """Compute a companion GEVP result for one channel family.

    Supported ``base_channel`` values:
    ``nucleon``, ``scalar``, ``pseudoscalar``, ``glueball``, ``su2``.
    """
    canonical_base_channel = str(base_channel).strip().lower()
    basis_channels = get_companion_gevp_basis_channels(canonical_base_channel)
    basis, input_labels, aligned_labels, dropped_pre_prune, notes = _build_basis_from_results(
        base_results=base_results,
        multiscale_output=multiscale_output,
        config=config,
        base_channel=canonical_base_channel,
        basis_channels=basis_channels,
    )

    if basis.ndim != 2:
        raise ValueError(f"GEVP basis must be [K,T], got {tuple(basis.shape)}")

    _, t_len = basis.shape
    max_lag = int(max(1, min(int(config.max_lag), t_len - 1)))
    t0 = int(max(1, config.t0))
    if t0 >= max_lag:
        t0 = max(1, max_lag // 2)
        notes.append(f"t0 adjusted to {t0} to satisfy t0 < max_lag.")

    dt = float("nan")
    for channel in basis_channels:
        result = base_results.get(channel)
        if result is not None and result.dt > 0:
            dt = float(result.dt)
            break
    if not math.isfinite(dt) or dt <= 0:
        dt = 1.0
        notes.append(
            f"dt unavailable from {canonical_base_channel} basis channels; defaulting to dt=1.0."
        )

    c_lags = _fft_cross_correlator_lags(
        basis,
        max_lag=max_lag,
        use_connected=bool(config.use_connected),
    )
    whitener, keep_idx, cond_c0 = _build_whitener(
        c_lags[t0],
        eig_rel_cutoff=float(config.eig_rel_cutoff),
        cond_limit=float(config.cond_limit),
        shrinkage=float(config.shrinkage),
    )

    kept_basis_labels = [aligned_labels[int(i.item())] for i in keep_idx]
    dropped_basis_labels = [label for label in aligned_labels if label not in kept_basis_labels]
    dropped_basis_labels = dropped_pre_prune + dropped_basis_labels
    if dropped_basis_labels:
        notes.append(
            "Dropped basis vectors: " + ", ".join(dropped_basis_labels[:8])
            + (" ..." if len(dropped_basis_labels) > 8 else "")
        )

    principal = _project_and_extract_principal(c_lags, whitener)
    correlator = principal[t0:].clone()
    correlator = torch.where(
        torch.isfinite(correlator) & (correlator > 1e-12),
        correlator,
        torch.full_like(correlator, float("nan")),
    )
    if correlator.numel() <= 2:
        msg = "GEVP principal correlator is too short after t0 truncation."
        raise ValueError(msg)

    fit, window_masses, window_aic, window_widths, window_r2 = _fit_mass_from_correlator(
        correlator,
        dt=dt,
        config=config,
    )
    effective_mass = compute_effective_mass_torch(correlator, dt)

    corr_err: Tensor | None = None
    bootstrap_mode_applied = _sanitize_mode(config.bootstrap_mode, GEVP_BOOTSTRAP_MODES, "time")
    if bool(config.compute_bootstrap_errors) and int(config.n_bootstrap) > 1:
        if bootstrap_mode_applied != "time":
            notes.append(
                f"bootstrap_mode='{bootstrap_mode_applied}' requested; using time bootstrap for GEVP basis."
            )
        gen = torch.Generator(device=basis.device)
        gen.manual_seed(int(config.bootstrap_seed))
        n_boot = int(config.n_bootstrap)
        idx = torch.randint(0, t_len, (n_boot, t_len), generator=gen, device=basis.device)
        sampled = torch.gather(
            basis.unsqueeze(0).expand(n_boot, -1, -1),
            dim=2,
            index=idx.unsqueeze(1).expand(-1, basis.shape[0], -1),
        )
        c_boot = _fft_cross_correlator_lags_batched(
            sampled,
            max_lag=max_lag,
            use_connected=bool(config.use_connected),
        )
        c_proj_boot = torch.einsum("ki,btkj,jm->btim", whitener, c_boot, whitener)
        c_proj_boot = 0.5 * (c_proj_boot + c_proj_boot.transpose(-1, -2))
        evals_boot = torch.linalg.eigvalsh(c_proj_boot)
        principal_boot = evals_boot[..., -1].real.float()[:, t0:]
        principal_boot = torch.where(
            torch.isfinite(principal_boot) & (principal_boot > 1e-12),
            principal_boot,
            torch.full_like(principal_boot, float("nan")),
        )
        corr_err = _nanstd_compat(principal_boot, dim=0)

        mass_samples = torch.full((n_boot,), float("nan"), dtype=torch.float32, device=basis.device)
        for b_idx in range(n_boot):
            mass_samples[b_idx] = float(
                _mass_from_fit_mode(principal_boot[b_idx], dt=dt, config=config)
            )
        mass_std = float(_nanstd_compat(mass_samples, dim=0).item())
        if math.isfinite(mass_std):
            fit["mass_error_bootstrap"] = mass_std

    fit["source"] = f"gevp_{canonical_base_channel}"
    fit["base_channel"] = canonical_base_channel
    fit["gevp_t0"] = int(t0)
    fit["gevp_basis_strategy"] = _sanitize_mode(config.basis_strategy, GEVP_BASIS_STRATEGIES, "base_only")
    fit["gevp_min_operator_r2"] = float(config.min_operator_r2)
    fit["gevp_min_operator_windows"] = int(max(0, config.min_operator_windows))
    fit["gevp_max_operator_error_pct"] = float(config.max_operator_error_pct)
    fit["gevp_remove_artifacts"] = bool(config.remove_artifacts)
    # Backward-compatible metadata key used by older dashboard summaries.
    fit["gevp_exclude_zero_error_operators"] = bool(config.remove_artifacts)
    fit["gevp_n_basis_input"] = int(len(input_labels))
    fit["gevp_n_basis_aligned"] = int(len(aligned_labels))
    fit["gevp_n_basis_kept"] = int(len(kept_basis_labels))
    fit["gevp_basis_labels"] = input_labels
    fit["gevp_basis_labels_kept"] = kept_basis_labels
    fit["gevp_basis_labels_dropped"] = dropped_basis_labels
    fit["gevp_condition_number"] = float(cond_c0)
    fit["gevp_bootstrap_mode_applied"] = bootstrap_mode_applied
    sample_counts = [
        int(base_results[ch].n_samples)
        for ch in basis_channels
        if ch in base_results and base_results[ch].n_samples > 0
    ]
    n_samples = min(sample_counts) if sample_counts else int(correlator.numel())
    if not sample_counts:
        notes.append(
            f"No base {canonical_base_channel} sample count available; "
            "using correlator length for n_samples."
        )
    fit["gevp_notes"] = list(notes)

    result = ChannelCorrelatorResult(
        channel_name=f"{canonical_base_channel}_gevp",
        correlator=correlator,
        correlator_err=corr_err,
        effective_mass=effective_mass,
        mass_fit=fit,
        series=correlator,
        n_samples=n_samples,
        dt=dt,
        window_masses=window_masses,
        window_aic=window_aic,
        window_widths=window_widths,
        window_r2=window_r2,
    )

    return GEVPResult(
        result=result,
        basis_labels=input_labels,
        kept_basis_labels=kept_basis_labels,
        dropped_basis_labels=dropped_basis_labels,
        cond_c0=cond_c0,
        bootstrap_mode_applied=bootstrap_mode_applied,
        notes=notes,
    )


def compute_companion_nucleon_gevp(
    *,
    base_results: dict[str, ChannelCorrelatorResult],
    multiscale_output: Any | None,
    config: GEVPConfig,
) -> GEVPResult:
    """Backward-compatible wrapper for nucleon-family companion GEVP."""
    return compute_companion_channel_gevp(
        base_results=base_results,
        multiscale_output=multiscale_output,
        config=config,
        base_channel="nucleon",
    )
