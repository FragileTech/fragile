"""Multiscale strong-force channel estimation using precomputed smeared kernels.

This module computes strong-force operators across multiple smearing scales
using kernels generated from recorded neighbor/edge-weight data.

Core guarantees:
- One shared kernel family/scales for all channels.
- Shared bootstrap indices across channels/scales.
- Optional walker-bootstrap path with resampled-kernel row renormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.aggregation import compute_color_states_batch, estimate_ell0
from fragile.fractalai.qft.baryon_triplet_channels import (
    _resolve_frame_indices,
    _safe_gather_3d,
    compute_baryon_correlator_from_color,
)
from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    ChannelCorrelatorResult,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.glueball_color_channels import (
    compute_glueball_color_correlator_from_color,
)
from fragile.fractalai.qft.meson_phase_channels import compute_meson_phase_correlator_from_color
from fragile.fractalai.qft.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
    compute_smeared_kernels_from_distances,
    iter_smeared_kernel_batches_from_history,
    select_interesting_scales_from_history,
)
from fragile.fractalai.qft.tensor_momentum_channels import (
    compute_tensor_momentum_correlator_from_color_positions,
)
from fragile.fractalai.qft.vector_meson_channels import (
    compute_vector_meson_correlator_from_color_positions,
)


BASE_CHANNELS = (
    "scalar",
    "pseudoscalar",
    "vector",
    "axial_vector",
    "tensor",
    "tensor_traceless",
    "nucleon",
    "glueball",
)
COMPANION_CHANNEL_MAP: dict[str, str] = {
    "scalar": "scalar_companion",
    "pseudoscalar": "pseudoscalar_companion",
    "scalar_score_directed": "scalar_score_directed_companion",
    "pseudoscalar_score_directed": "pseudoscalar_score_directed_companion",
    "scalar_score_weighted": "scalar_score_weighted_companion",
    "pseudoscalar_score_weighted": "pseudoscalar_score_weighted_companion",
    "scalar_raw": "scalar_raw_companion",
    "scalar_abs2_vacsub": "scalar_abs2_vacsub_companion",
    "vector": "vector_companion",
    "axial_vector": "axial_vector_companion",
    "vector_score_directed": "vector_score_directed_companion",
    "axial_vector_score_directed": "axial_vector_score_directed_companion",
    "vector_score_directed_longitudinal": "vector_score_directed_longitudinal_companion",
    "axial_vector_score_directed_longitudinal": "axial_vector_score_directed_longitudinal_companion",
    "vector_score_directed_transverse": "vector_score_directed_transverse_companion",
    "axial_vector_score_directed_transverse": "axial_vector_score_directed_transverse_companion",
    "vector_score_gradient": "vector_score_gradient_companion",
    "axial_vector_score_gradient": "axial_vector_score_gradient_companion",
    "tensor": "tensor_companion",
    "tensor_traceless": "tensor_traceless_companion",
    "nucleon": "nucleon_companion",
    "nucleon_score_signed": "nucleon_score_signed_companion",
    "nucleon_score_abs": "nucleon_score_abs_companion",
    "glueball": "glueball_companion",
    "nucleon_flux_action": "nucleon_flux_action_companion",
    "nucleon_flux_sin2": "nucleon_flux_sin2_companion",
    "nucleon_flux_exp": "nucleon_flux_exp_companion",
    "glueball_phase_action": "glueball_phase_action_companion",
    "glueball_phase_sin2": "glueball_phase_sin2_companion",
}
SUPPORTED_CHANNELS = BASE_CHANNELS + tuple(COMPANION_CHANNEL_MAP.values())
BOOTSTRAP_MODES = ("time", "walker", "hybrid")
KERNEL_TYPES = ("gaussian", "exponential", "tophat", "shell")


def _nanstd_compat(values: Tensor, *, dim: int) -> Tensor:
    """Compute NaN-aware std with compatibility for older Torch versions."""
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


@dataclass
class MultiscaleStrongForceConfig:
    """Configuration for multiscale strong-force channel estimation."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    mc_time_index: int | None = None
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    edge_weight_mode: str = "riemannian_kernel_volume"

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
    best_min_r2: float = -1.0
    best_min_windows: int = 0
    best_max_error_pct: float = 30.0
    best_remove_artifacts: bool = False

    # Bootstrap controls
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100
    bootstrap_seed: int = 12345
    bootstrap_mode: str = "hybrid"
    walker_bootstrap_max_walkers: int = 512
    walker_bootstrap_max_samples: int = 64
    companion_baryon_flux_exp_alpha: float = 1.0


@dataclass
class MultiscaleStrongForceOutput:
    """Output payload for multiscale strong-force analysis."""

    scales: Tensor
    frame_indices: list[int]
    per_scale_results: dict[str, list[ChannelCorrelatorResult]]
    best_results: dict[str, ChannelCorrelatorResult]
    best_scale_index: dict[str, int]
    series_by_channel: dict[str, Tensor]  # [S, T]
    bootstrap_mode_applied: str
    notes: list[str]
    bootstrap_mass_std: dict[str, Tensor] | None = None  # [S]


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
    """Build a `ChannelCorrelatorResult` from precomputed correlator data."""
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
    """Fit mass from a single correlator for bootstrap spread estimates."""
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
    """Extract valid fit-window count from a channel result."""
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
    """Return True when a result passes active best-scale quality gates."""
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
    min_r2: float = -1.0,
    min_windows: int = 0,
    max_error_pct: float = 30.0,
    remove_artifacts: bool = False,
) -> int | None:
    """Select best scale index using AIC/R²/mass-error fallback under quality filters."""
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


def _masked_mean(values: Tensor, mask: Tensor, *, dim: int = -1) -> Tensor:
    """Compute mean along `dim` under boolean mask with safe denominator."""
    weights = mask.float()
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp(min=1.0)
    return numerator / denominator


def _masked_mean_multi(values: Tensor, mask: Tensor, *, dims: tuple[int, ...]) -> Tensor:
    """Compute masked mean over multiple dimensions."""
    weights = mask.to(dtype=values.dtype)
    numerator = (values * weights).sum(dim=dims)
    denominator = weights.sum(dim=dims).clamp(min=1.0)
    return numerator / denominator


def _safe_gather_2d(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather values[:, idx] and return gathered values plus in-range mask."""
    if values.ndim != 2 or indices.ndim != 2:
        raise ValueError(
            f"_safe_gather_2d expects [T,N] tensors, got {tuple(values.shape)} and {tuple(indices.shape)}."
        )
    if values.shape != indices.shape:
        raise ValueError(
            f"_safe_gather_2d expects aligned shapes, got {tuple(values.shape)} and {tuple(indices.shape)}."
        )
    n = int(values.shape[1])
    in_range = (indices >= 0) & (indices < n)
    idx_safe = indices.clamp(min=0, max=max(n - 1, 0))
    gathered = torch.gather(values, dim=1, index=idx_safe)
    return gathered, in_range


def _safe_gather_4d_by_2d_indices(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Safely gather walker features from [T,S,N,D] using [T,N] indices."""
    if values.ndim != 4 or indices.ndim != 2:
        raise ValueError(
            "_safe_gather_4d_by_2d_indices expects values [T,S,N,D] and indices [T,N], got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}."
        )
    t_len, s_count, n_walkers, n_feat = values.shape
    if indices.shape[0] != t_len or indices.shape[1] != n_walkers:
        raise ValueError(
            "_safe_gather_4d_by_2d_indices expects indices [T,N] aligned with values walker axis, got "
            f"{tuple(indices.shape)} vs [T={t_len},N={n_walkers}]."
        )
    in_range = (indices >= 0) & (indices < n_walkers)
    idx_safe = indices.clamp(min=0, max=max(n_walkers - 1, 0))
    idx_exp = idx_safe[:, None, :, None].expand(t_len, s_count, n_walkers, n_feat)
    gathered = torch.gather(values, dim=2, index=idx_exp)
    return gathered, in_range[:, None, :].expand(t_len, s_count, n_walkers)


def _safe_gather_pairwise_distances(
    distances: Tensor,
    row_idx: Tensor,
    col_idx: Tensor,
) -> tuple[Tensor, Tensor]:
    """Safely gather distances[t, row_idx[t,i], col_idx[t,i]] for [T,N] indices."""
    if distances.ndim != 3:
        raise ValueError(f"distances must have shape [T,N,N], got {tuple(distances.shape)}.")
    if row_idx.ndim != 2 or col_idx.ndim != 2:
        raise ValueError(
            "row_idx and col_idx must have shape [T,N], got "
            f"{tuple(row_idx.shape)} and {tuple(col_idx.shape)}."
        )
    if row_idx.shape != col_idx.shape:
        raise ValueError(
            "row_idx and col_idx must have the same shape, got "
            f"{tuple(row_idx.shape)} and {tuple(col_idx.shape)}."
        )
    t_len, n, n2 = distances.shape
    if n != n2:
        raise ValueError(
            f"distances must be square on last two axes, got {tuple(distances.shape)}."
        )
    if row_idx.shape != (t_len, n):
        raise ValueError(
            "row_idx/col_idx must align with distances [T,N,N], got "
            f"{tuple(row_idx.shape)} vs [T={t_len},N={n}]."
        )
    in_row = (row_idx >= 0) & (row_idx < n)
    in_col = (col_idx >= 0) & (col_idx < n)
    valid = in_row & in_col
    row_safe = row_idx.clamp(min=0, max=max(n - 1, 0))
    col_safe = col_idx.clamp(min=0, max=max(n - 1, 0))
    flat_idx = (row_safe * n + col_safe).clamp(min=0, max=max(n * n - 1, 0))
    gathered = torch.gather(distances.reshape(t_len, n * n), dim=1, index=flat_idx)
    return gathered, valid


def _remap_companion_indices_for_bootstrap(
    companions: Tensor,
    bootstrap_idx: Tensor,
    *,
    n_walkers: int,
) -> Tensor:
    """Remap companion indices to bootstrap-resampled walker indexing.

    Args:
        companions: Companion indices [T,N] in original indexing.
        bootstrap_idx: Resampled walker ids [N] mapping bootstrap position -> original id.
        n_walkers: Number of walkers in the original indexing.

    Returns:
        Remapped companions [T,N] in bootstrap indexing; invalid/unmapped entries are -1.
    """
    if companions.ndim != 2 or bootstrap_idx.ndim != 1:
        raise ValueError(
            "_remap_companion_indices_for_bootstrap expects companions [T,N] and bootstrap_idx [N], got "
            f"{tuple(companions.shape)} and {tuple(bootstrap_idx.shape)}."
        )
    if companions.shape[1] != bootstrap_idx.shape[0]:
        raise ValueError(
            "_remap_companion_indices_for_bootstrap expects aligned walker dimensions, got "
            f"{companions.shape[1]} vs {bootstrap_idx.shape[0]}."
        )
    if int(n_walkers) <= 0:
        return torch.full_like(companions, -1)

    dev = companions.device
    idx = bootstrap_idx.to(device=dev, dtype=torch.long)
    old_to_new = torch.full((int(n_walkers),), -1, dtype=torch.long, device=dev)
    # Keep the first occurrence in case of duplicates.
    rev_pos = torch.arange(idx.shape[0] - 1, -1, -1, device=dev, dtype=torch.long)
    old_to_new[idx.flip(0)] = rev_pos

    comp_old = companions.to(dtype=torch.long)
    comp_clamped = comp_old.clamp(min=0, max=max(int(n_walkers) - 1, 0))
    mapped = old_to_new[comp_clamped]
    valid_old = (comp_old >= 0) & (comp_old < int(n_walkers))
    valid_mapped = valid_old & (mapped >= 0)
    out = torch.full_like(comp_old, -1)
    out[valid_mapped] = mapped[valid_mapped]
    return out


def _compute_channel_series_from_kernels(
    *,
    color: Tensor,  # [T, N, d] complex
    color_valid: Tensor,  # [T, N] bool
    positions: Tensor,  # [T, N, d] float
    alive: Tensor,  # [T, N] bool
    force: Tensor,  # [T, N, d] float
    kernels: Tensor,  # [T, S, N, N] float
    scales: Tensor | None = None,  # [S] float, used by companion hard-threshold channels
    pairwise_distances: Tensor | None = None,  # [T, N, N] float, used by companion channels
    companions_distance: Tensor | None = None,  # [T, N] long
    companions_clone: Tensor | None = None,  # [T, N] long
    cloning_scores: Tensor | None = None,  # [T, N] float
    channels: list[str],
) -> dict[str, Tensor]:
    """Compute multiscale operator series for one frame chunk.

    Returns tensors shaped [S, T_chunk].

    Isotropic channels use kernel-smoothed fields.
    Companion channels use original companion operators with hard-threshold
    geodesic gating at each scale.
    """
    if kernels.ndim != 4:
        raise ValueError(f"kernels must have shape [T,S,N,N], got {tuple(kernels.shape)}.")
    t_len, n_scales, _, _ = kernels.shape
    device = kernels.device
    color = color.to(device=device)
    color_valid = color_valid.to(device=device, dtype=torch.bool)
    positions = positions.to(device=device, dtype=torch.float32)
    alive = alive.to(device=device, dtype=torch.bool)
    force = force.to(device=device, dtype=torch.float32)

    kernels_c = kernels.to(dtype=color.dtype)
    color_sm = torch.einsum("tsij,tjd->tsid", kernels_c, color)  # [T,S,N,d]
    inner = torch.einsum("tnd,tsnd->tsn", torch.conj(color), color_sm)  # [T,S,N]
    inner_re = inner.real.float()
    inner_im = inner.imag.float()

    x_sm = torch.einsum("tsij,tjd->tsid", kernels, positions)  # [T,S,N,d]
    dx = x_sm - positions[:, None, :, :]  # [T,S,N,d]

    mask_color = (alive & color_valid)[:, None, :]  # [T,1,N]
    mask_alive = alive[:, None, :]  # [T,1,N]

    out: dict[str, Tensor] = {}
    if "scalar" in channels:
        out["scalar"] = _masked_mean(inner_re, mask_color, dim=-1).transpose(0, 1).contiguous()
    if "pseudoscalar" in channels:
        out["pseudoscalar"] = (
            _masked_mean(inner_im, mask_color, dim=-1).transpose(0, 1).contiguous()
        )

    if "vector" in channels:
        vec_local = inner_re.unsqueeze(-1) * dx
        vec_mean = _masked_mean(vec_local, mask_color.unsqueeze(-1), dim=-2)
        vec_series = torch.linalg.vector_norm(vec_mean, dim=-1)
        out["vector"] = vec_series.transpose(0, 1).contiguous()

    if "axial_vector" in channels:
        axial_local = inner_im.unsqueeze(-1) * dx
        axial_mean = _masked_mean(axial_local, mask_color.unsqueeze(-1), dim=-2)
        axial_series = torch.linalg.vector_norm(axial_mean, dim=-1)
        out["axial_vector"] = axial_series.transpose(0, 1).contiguous()

    if ("tensor" in channels or "tensor_traceless" in channels) and positions.shape[-1] >= 3:
        dx3 = dx[..., :3]
        x = dx3[..., 0]
        y = dx3[..., 1]
        z = dx3[..., 2]
        inv_sqrt2 = float(1.0 / math.sqrt(2.0))
        inv_sqrt6 = float(1.0 / math.sqrt(6.0))
        q = torch.stack(
            (
                x * y,
                x * z,
                y * z,
                (x * x - y * y) * inv_sqrt2,
                (2.0 * z * z - x * x - y * y) * inv_sqrt6,
            ),
            dim=-1,
        )  # [T,S,N,5]
        tensor_local = inner_re.unsqueeze(-1) * q
        tensor_mean = _masked_mean(tensor_local, mask_color.unsqueeze(-1), dim=-2)  # [T,S,5]
        tensor_series = torch.linalg.vector_norm(tensor_mean, dim=-1)  # [T,S]
        if "tensor" in channels:
            out["tensor"] = tensor_series.transpose(0, 1).contiguous()
        if "tensor_traceless" in channels:
            out["tensor_traceless"] = tensor_series.transpose(0, 1).contiguous()
    else:
        if "tensor" in channels:
            out["tensor"] = torch.zeros((n_scales, t_len), dtype=torch.float32, device=device)
        if "tensor_traceless" in channels:
            out["tensor_traceless"] = torch.zeros(
                (n_scales, t_len), dtype=torch.float32, device=device
            )

    if "nucleon" in channels:
        if color.shape[-1] >= 3:
            c3 = color[..., :3]  # [T,N,3]
            csm3 = color_sm[..., :3]  # [T,S,N,3]
            csm2 = torch.einsum("tsij,tsjd->tsid", kernels_c, csm3)  # [T,S,N,3]
            c3_exp = c3[:, None, :, :].expand(-1, csm3.shape[1], -1, -1)  # [T,S,N,3]
            mat = torch.stack([c3_exp, csm3, csm2], dim=-1)  # [T,S,N,3,3]
            det = torch.abs(torch.linalg.det(mat)).float()  # [T,S,N]
            nucleon_series = _masked_mean(det, mask_color, dim=-1)  # [T,S]
            out["nucleon"] = nucleon_series.transpose(0, 1).contiguous()
        else:
            out["nucleon"] = torch.zeros((n_scales, t_len), dtype=torch.float32, device=device)

    if "glueball" in channels:
        force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)  # [T,N]
        force_sm = torch.einsum("tsij,tj->tsi", kernels, force_sq)  # [T,S,N]
        glueball_series = _masked_mean(force_sm, mask_alive, dim=-1)  # [T,S]
        out["glueball"] = glueball_series.transpose(0, 1).contiguous()

    # ------------------------------------------------------------------
    # Companion-channel multiscale operators (kept separate from base channels)
    # ------------------------------------------------------------------
    requested_companion_channels = [
        name for name in COMPANION_CHANNEL_MAP.values() if name in channels
    ]
    if requested_companion_channels:
        for channel in requested_companion_channels:
            out.setdefault(
                channel, torch.zeros((n_scales, t_len), dtype=torch.float32, device=device)
            )
        if (
            companions_distance is None
            or companions_clone is None
            or pairwise_distances is None
            or scales is None
        ):
            for channel in requested_companion_channels:
                out[channel] = torch.zeros((n_scales, t_len), dtype=torch.float32, device=device)
            return out

        scales_t = torch.as_tensor(scales, device=device, dtype=torch.float32).reshape(-1)
        if int(scales_t.numel()) != int(n_scales):
            raise ValueError(
                "scales must align with kernel scale axis, got "
                f"{int(scales_t.numel())} scales for kernels with S={n_scales}."
            )
        dist = pairwise_distances.to(device=device, dtype=torch.float32)
        n_walkers = int(alive.shape[1])
        if dist.shape != (t_len, n_walkers, n_walkers):
            raise ValueError(
                "pairwise_distances must have shape [T,N,N] aligned with kernels, got "
                f"{tuple(dist.shape)} vs [T={t_len},N={n_walkers},N={n_walkers}]."
            )

        comp_j = companions_distance.to(device=device, dtype=torch.long)
        comp_k = companions_clone.to(device=device, dtype=torch.long)
        if comp_j.shape != alive.shape or comp_k.shape != alive.shape:
            raise ValueError(
                "companion arrays must have shape [T,N] aligned with alive/color tensors, got "
                f"{tuple(comp_j.shape)} and {tuple(comp_k.shape)} vs {tuple(alive.shape)}."
            )
        needs_score_pair_channels = any(
            name in channels
            for name in (
                "scalar_score_directed_companion",
                "pseudoscalar_score_directed_companion",
                "scalar_score_weighted_companion",
                "pseudoscalar_score_weighted_companion",
            )
        )
        needs_score_triplet_channels = any(
            name in channels
            for name in ("nucleon_score_signed_companion", "nucleon_score_abs_companion")
        )
        scores = None
        score_j = None
        score_k = None
        score_j_in_range = None
        score_k_in_range = None
        finite_score_i = None
        finite_score_j = None
        finite_score_k = None
        if needs_score_pair_channels or needs_score_triplet_channels:
            if cloning_scores is None:
                msg = "cloning_scores is required for score-directed companion channels."
                raise ValueError(msg)
            scores = cloning_scores.to(device=device, dtype=torch.float32)
            if scores.shape != alive.shape:
                raise ValueError(
                    "cloning_scores must have shape [T,N] aligned with alive/color tensors, got "
                    f"{tuple(scores.shape)} vs {tuple(alive.shape)}."
                )

        # Companion channels use original (non-smoothed) operators gated by
        # geodesic hard thresholds at each scale.
        c_j, in_j = _safe_gather_3d(color, comp_j)
        c_k, in_k = _safe_gather_3d(color, comp_k)
        x_j, _ = _safe_gather_3d(positions, comp_j)
        x_k, _ = _safe_gather_3d(positions, comp_k)

        alive_j_2d, in_j_alive = _safe_gather_2d(alive, comp_j)
        alive_k_2d, in_k_alive = _safe_gather_2d(alive, comp_k)
        valid_j_2d, in_j_valid = _safe_gather_2d(color_valid, comp_j)
        valid_k_2d, in_k_valid = _safe_gather_2d(color_valid, comp_k)
        if scores is not None:
            score_j, score_j_in_range = _safe_gather_2d(scores, comp_j)
            score_k, score_k_in_range = _safe_gather_2d(scores, comp_k)
            finite_score_i = torch.isfinite(scores)
            finite_score_j = torch.isfinite(score_j)
            finite_score_k = torch.isfinite(score_k)

        anchor_idx = torch.arange(n_walkers, device=device, dtype=torch.long).view(1, -1)
        anchor_rows = anchor_idx.expand(t_len, -1)
        distinct = (comp_j != anchor_idx) & (comp_k != anchor_idx) & (comp_j != comp_k)
        base_anchor_valid = alive & color_valid
        base_pair_j = (
            base_anchor_valid
            & alive_j_2d
            & valid_j_2d
            & in_j_alive
            & in_j_valid
            & (comp_j != anchor_idx)
        )
        base_pair_k = (
            base_anchor_valid
            & alive_k_2d
            & valid_k_2d
            & in_k_alive
            & in_k_valid
            & (comp_k != anchor_idx)
        )
        base_triplet_valid = (
            base_anchor_valid
            & alive_j_2d
            & alive_k_2d
            & valid_j_2d
            & valid_k_2d
            & in_j_alive
            & in_k_alive
            & in_j_valid
            & in_k_valid
            & distinct
        )

        d_ij, in_dij = _safe_gather_pairwise_distances(dist, anchor_rows, comp_j)
        d_ik, in_dik = _safe_gather_pairwise_distances(dist, anchor_rows, comp_k)
        d_jk, in_djk = _safe_gather_pairwise_distances(dist, comp_j, comp_k)
        finite_dij = torch.isfinite(d_ij)
        finite_dik = torch.isfinite(d_ik)
        finite_djk = torch.isfinite(d_jk)

        inner_j = torch.einsum("tnd,tnd->tn", torch.conj(color), c_j)
        inner_k = torch.einsum("tnd,tnd->tn", torch.conj(color), c_k)
        finite_inner_j = torch.isfinite(inner_j.real) & torch.isfinite(inner_j.imag)
        finite_inner_k = torch.isfinite(inner_k.real) & torch.isfinite(inner_k.imag)
        eps = 1e-12
        pair_j_valid_2d = (
            base_pair_j & in_j & in_dij & finite_dij & finite_inner_j & (inner_j.abs() > eps)
        )
        pair_k_valid_2d = (
            base_pair_k & in_k & in_dik & finite_dik & finite_inner_k & (inner_k.abs() > eps)
        )
        triplet_valid_2d = (
            base_triplet_valid
            & in_j
            & in_k
            & in_dij
            & in_dik
            & in_djk
            & finite_dij
            & finite_dik
            & finite_djk
        )

        scales_view = scales_t.view(1, n_scales, 1)
        pair_j_valid = pair_j_valid_2d[:, None, :] & (d_ij[:, None, :] <= scales_view)
        pair_k_valid = pair_k_valid_2d[:, None, :] & (d_ik[:, None, :] <= scales_view)
        triplet_radius = torch.maximum(d_ij, torch.maximum(d_ik, d_jk))
        triplet_valid = triplet_valid_2d[:, None, :] & (triplet_radius[:, None, :] <= scales_view)

        if (
            "scalar_companion" in channels
            or "scalar_raw_companion" in channels
            or "scalar_abs2_vacsub_companion" in channels
            or "pseudoscalar_companion" in channels
        ):
            inner_pair = torch.stack([inner_j, inner_k], dim=-1)[:, None, :, :].expand(
                t_len, n_scales, n_walkers, 2
            )  # [T,S,N,2]
            pair_valid = torch.stack([pair_j_valid, pair_k_valid], dim=-1)  # [T,S,N,2]
            if "scalar_companion" in channels:
                scalar_comp = _masked_mean_multi(
                    inner_pair.real.float(), pair_valid, dims=(-1, -2)
                )
                out["scalar_companion"] = scalar_comp.transpose(0, 1).contiguous()
            if "scalar_raw_companion" in channels:
                scalar_raw_comp = _masked_mean_multi(
                    inner_pair.real.float(), pair_valid, dims=(-1, -2)
                )
                out["scalar_raw_companion"] = scalar_raw_comp.transpose(0, 1).contiguous()
            if "scalar_abs2_vacsub_companion" in channels:
                scalar_abs2_comp = _masked_mean_multi(
                    inner_pair.abs().square().float(), pair_valid, dims=(-1, -2)
                )
                out["scalar_abs2_vacsub_companion"] = scalar_abs2_comp.transpose(0, 1).contiguous()
            if "pseudoscalar_companion" in channels:
                pseu_comp = _masked_mean_multi(inner_pair.imag.float(), pair_valid, dims=(-1, -2))
                out["pseudoscalar_companion"] = pseu_comp.transpose(0, 1).contiguous()

        if needs_score_pair_channels:
            if (
                scores is None
                or score_j is None
                or score_k is None
                or score_j_in_range is None
                or score_k_in_range is None
                or finite_score_i is None
                or finite_score_j is None
                or finite_score_k is None
            ):
                msg = "Internal error: missing score tensors for score-directed channels."
                raise RuntimeError(msg)
            ds_j = score_j - scores
            ds_k = score_k - scores
            inner_j_oriented = torch.where(ds_j >= 0, inner_j, torch.conj(inner_j))
            inner_k_oriented = torch.where(ds_k >= 0, inner_k, torch.conj(inner_k))
            pair_j_score_valid = (
                pair_j_valid
                & score_j_in_range[:, None, :]
                & finite_score_i[:, None, :]
                & finite_score_j[:, None, :]
            )
            pair_k_score_valid = (
                pair_k_valid
                & score_k_in_range[:, None, :]
                & finite_score_i[:, None, :]
                & finite_score_k[:, None, :]
            )
            inner_pair_oriented = torch.stack([inner_j_oriented, inner_k_oriented], dim=-1)[
                :, None, :, :
            ].expand(t_len, n_scales, n_walkers, 2)
            pair_score_valid = torch.stack([pair_j_score_valid, pair_k_score_valid], dim=-1)
            scalar_score = None
            if "scalar_score_directed_companion" in channels:
                scalar_score = _masked_mean_multi(
                    inner_pair_oriented.real.float(), pair_score_valid, dims=(-1, -2)
                )
            if "scalar_score_directed_companion" in channels and scalar_score is not None:
                out["scalar_score_directed_companion"] = scalar_score.transpose(0, 1).contiguous()
            if "pseudoscalar_score_directed_companion" in channels:
                pseudoscalar_score = _masked_mean_multi(
                    inner_pair_oriented.imag.float(), pair_score_valid, dims=(-1, -2)
                )
                out["pseudoscalar_score_directed_companion"] = pseudoscalar_score.transpose(
                    0, 1
                ).contiguous()

        if "vector_companion" in channels or "axial_vector_companion" in channels:
            disp_j = x_j - positions
            disp_k = x_k - positions
            pair_valid = torch.stack([pair_j_valid, pair_k_valid], dim=-1)  # [T,S,N,2]
            if "vector_companion" in channels:
                vec_j = inner_j.real.float().unsqueeze(-1) * disp_j  # [T,N,d]
                vec_k = inner_k.real.float().unsqueeze(-1) * disp_k  # [T,N,d]
                vec_pair = torch.stack([vec_j, vec_k], dim=-2)[:, None, :, :, :].expand(
                    t_len, n_scales, n_walkers, 2, positions.shape[-1]
                )  # [T,S,N,2,d]
                vec_mask = pair_valid.unsqueeze(-1).expand_as(vec_pair)
                vec_mean = _masked_mean_multi(vec_pair, vec_mask, dims=(-2, -3))  # [T,S,d]
                out["vector_companion"] = (
                    torch.linalg.vector_norm(vec_mean, dim=-1).transpose(0, 1).contiguous()
                )
            if "axial_vector_companion" in channels:
                axial_j = inner_j.imag.float().unsqueeze(-1) * disp_j  # [T,N,d]
                axial_k = inner_k.imag.float().unsqueeze(-1) * disp_k  # [T,N,d]
                axial_pair = torch.stack([axial_j, axial_k], dim=-2)[:, None, :, :, :].expand(
                    t_len, n_scales, n_walkers, 2, positions.shape[-1]
                )  # [T,S,N,2,d]
                axial_mask = pair_valid.unsqueeze(-1).expand_as(axial_pair)
                axial_mean = _masked_mean_multi(axial_pair, axial_mask, dims=(-2, -3))  # [T,S,d]
                out["axial_vector_companion"] = (
                    torch.linalg.vector_norm(axial_mean, dim=-1).transpose(0, 1).contiguous()
                )

        if "tensor_companion" in channels:
            if positions.shape[-1] >= 3:
                disp_j = x_j - positions
                disp_k = x_k - positions
                inv_sqrt2 = float(1.0 / math.sqrt(2.0))
                inv_sqrt6 = float(1.0 / math.sqrt(6.0))

                xj = disp_j[..., 0]
                yj = disp_j[..., 1]
                zj = disp_j[..., 2]
                q_j = torch.stack(
                    (
                        xj * yj,
                        xj * zj,
                        yj * zj,
                        (xj * xj - yj * yj) * inv_sqrt2,
                        (2.0 * zj * zj - xj * xj - yj * yj) * inv_sqrt6,
                    ),
                    dim=-1,
                )  # [T,S,N,5]

                xk = disp_k[..., 0]
                yk = disp_k[..., 1]
                zk = disp_k[..., 2]
                q_k = torch.stack(
                    (
                        xk * yk,
                        xk * zk,
                        yk * zk,
                        (xk * xk - yk * yk) * inv_sqrt2,
                        (2.0 * zk * zk - xk * xk - yk * yk) * inv_sqrt6,
                    ),
                    dim=-1,
                )  # [T,S,N,5]

                t_j = inner_j.real.float().unsqueeze(-1) * q_j  # [T,N,5]
                t_k = inner_k.real.float().unsqueeze(-1) * q_k  # [T,N,5]
                tensor_pair = torch.stack([t_j, t_k], dim=-2)[:, None, :, :, :].expand(
                    t_len, n_scales, n_walkers, 2, 5
                )  # [T,S,N,2,5]
                pair_valid = torch.stack([pair_j_valid, pair_k_valid], dim=-1)  # [T,S,N,2]
                tensor_mask = pair_valid.unsqueeze(-1).expand_as(tensor_pair)
                tensor_mean = _masked_mean_multi(
                    tensor_pair, tensor_mask, dims=(-2, -3)
                )  # [T,S,5]
                tensor_series = (
                    torch.linalg.vector_norm(tensor_mean, dim=-1).transpose(0, 1).contiguous()
                )

                out["tensor_companion"] = tensor_series
            else:
                out["tensor_companion"] = torch.zeros(
                    (n_scales, t_len),
                    dtype=torch.float32,
                    device=device,
                )

        nucleon_companion_channels = (
            "nucleon_companion",
            "nucleon_flux_action_companion",
            "nucleon_flux_sin2_companion",
            "nucleon_flux_exp_companion",
        )
        glueball_companion_channels = (
            "glueball_companion",
            "glueball_phase_action_companion",
            "glueball_phase_sin2_companion",
        )
        needs_triplet_plaquette = any(
            name in channels
            for name in (
                "glueball_companion",
                "glueball_phase_action_companion",
                "glueball_phase_sin2_companion",
                "nucleon_flux_action_companion",
                "nucleon_flux_sin2_companion",
                "nucleon_flux_exp_companion",
            )
        )

        plaquette = None
        phase = None
        plaquette_valid = None
        if needs_triplet_plaquette:
            z_ij = inner_j
            z_jk = torch.einsum("tnd,tnd->tn", torch.conj(c_j), c_k)
            z_ki = torch.einsum("tnd,tnd->tn", torch.conj(c_k), color)
            plaquette = z_ij * z_jk * z_ki  # [T,N]
            plaquette_valid = (
                triplet_valid
                & torch.isfinite(plaquette.real).unsqueeze(1)
                & torch.isfinite(plaquette.imag).unsqueeze(1)
                & (z_ij.abs() > eps).unsqueeze(1)
                & (z_jk.abs() > eps).unsqueeze(1)
                & (z_ki.abs() > eps).unsqueeze(1)
            )
            phase = torch.angle(plaquette)

        if any(name in channels for name in nucleon_companion_channels):
            if color.shape[-1] >= 3:
                c_i3 = color[..., :3]
                c_j3 = c_j[..., :3]
                c_k3 = c_k[..., :3]
                det = torch.abs(
                    c_i3[..., 0] * (c_j3[..., 1] * c_k3[..., 2] - c_j3[..., 2] * c_k3[..., 1])
                    - c_i3[..., 1] * (c_j3[..., 0] * c_k3[..., 2] - c_j3[..., 2] * c_k3[..., 0])
                    + c_i3[..., 2] * (c_j3[..., 0] * c_k3[..., 1] - c_j3[..., 1] * c_k3[..., 0])
                ).float()  # [T,N]
                det_valid = (
                    triplet_valid & torch.isfinite(det).unsqueeze(1) & (det > eps).unsqueeze(1)
                )
                det_scales = det[:, None, :].expand(t_len, n_scales, n_walkers)
                if "nucleon_companion" in channels:
                    nucleon_comp = _masked_mean(det_scales, det_valid, dim=-1)
                    out["nucleon_companion"] = nucleon_comp.transpose(0, 1).contiguous()
                if plaquette is not None and phase is not None and plaquette_valid is not None:
                    action = 1.0 - torch.cos(phase)
                    flux_weights: dict[str, Tensor] = {
                        "nucleon_flux_action_companion": action.float(),
                        "nucleon_flux_sin2_companion": torch.sin(phase).square().float(),
                        "nucleon_flux_exp_companion": torch.exp(action).float(),
                    }
                    flux_valid = det_valid & plaquette_valid
                    for channel_name, weight in flux_weights.items():
                        if channel_name not in channels:
                            continue
                        flux_obs = det * weight
                        flux_obs_scales = flux_obs[:, None, :].expand(t_len, n_scales, n_walkers)
                        flux_series = _masked_mean(flux_obs_scales, flux_valid, dim=-1)
                        out[channel_name] = flux_series.transpose(0, 1).contiguous()
            else:
                for channel_name in nucleon_companion_channels:
                    if channel_name in channels:
                        out[channel_name] = torch.zeros(
                            (n_scales, t_len), dtype=torch.float32, device=device
                        )

        score_nucleon_channels = (
            "nucleon_score_signed_companion",
            "nucleon_score_abs_companion",
        )
        if any(name in channels for name in score_nucleon_channels):
            if (
                scores is None
                or score_j is None
                or score_k is None
                or score_j_in_range is None
                or score_k_in_range is None
                or finite_score_i is None
                or finite_score_j is None
                or finite_score_k is None
            ):
                msg = "Internal error: missing score tensors for score-ordered nucleon channels."
                raise RuntimeError(msg)
            if color.shape[-1] >= 3:
                c_i3 = color[..., :3]
                c_j3 = c_j[..., :3]
                c_k3 = c_k[..., :3]
                triplet_scores = torch.stack([scores, score_j, score_k], dim=-1)  # [T,N,3]
                triplet_colors = torch.stack([c_i3, c_j3, c_k3], dim=-2)  # [T,N,3,3]
                order = torch.argsort(triplet_scores, dim=-1)
                ordered_colors = torch.gather(
                    triplet_colors,
                    dim=-2,
                    index=order.unsqueeze(-1).expand(-1, -1, -1, 3),
                )
                det_ordered = (
                    ordered_colors[..., 0, 0]
                    * (
                        ordered_colors[..., 1, 1] * ordered_colors[..., 2, 2]
                        - ordered_colors[..., 1, 2] * ordered_colors[..., 2, 1]
                    )
                    - ordered_colors[..., 0, 1]
                    * (
                        ordered_colors[..., 1, 0] * ordered_colors[..., 2, 2]
                        - ordered_colors[..., 1, 2] * ordered_colors[..., 2, 0]
                    )
                    + ordered_colors[..., 0, 2]
                    * (
                        ordered_colors[..., 1, 0] * ordered_colors[..., 2, 1]
                        - ordered_colors[..., 1, 1] * ordered_colors[..., 2, 0]
                    )
                )
                finite_det = torch.isfinite(det_ordered.real) & torch.isfinite(det_ordered.imag)
                score_triplet_valid = (
                    triplet_valid
                    & score_j_in_range[:, None, :]
                    & score_k_in_range[:, None, :]
                    & finite_score_i[:, None, :]
                    & finite_score_j[:, None, :]
                    & finite_score_k[:, None, :]
                    & finite_det[:, None, :]
                    & (det_ordered.abs() > eps).unsqueeze(1)
                )
                if "nucleon_score_signed_companion" in channels:
                    det_signed = det_ordered.real.float()
                    det_signed_scales = det_signed[:, None, :].expand(t_len, n_scales, n_walkers)
                    det_signed_series = _masked_mean(
                        det_signed_scales, score_triplet_valid, dim=-1
                    )
                    out["nucleon_score_signed_companion"] = det_signed_series.transpose(
                        0, 1
                    ).contiguous()
                if "nucleon_score_abs_companion" in channels:
                    det_abs = det_ordered.abs().float()
                    det_abs_scales = det_abs[:, None, :].expand(t_len, n_scales, n_walkers)
                    det_abs_series = _masked_mean(det_abs_scales, score_triplet_valid, dim=-1)
                    out["nucleon_score_abs_companion"] = det_abs_series.transpose(
                        0, 1
                    ).contiguous()
            else:
                for channel_name in score_nucleon_channels:
                    if channel_name in channels:
                        out[channel_name] = torch.zeros(
                            (n_scales, t_len), dtype=torch.float32, device=device
                        )

        if any(name in channels for name in glueball_companion_channels):
            if plaquette is None or phase is None or plaquette_valid is None:
                for channel_name in glueball_companion_channels:
                    if channel_name in channels:
                        out[channel_name] = torch.zeros(
                            (n_scales, t_len), dtype=torch.float32, device=device
                        )
            else:
                glue_obs: dict[str, Tensor] = {
                    "glueball_companion": plaquette.real.float(),
                    "glueball_phase_action_companion": (1.0 - torch.cos(phase)).float(),
                    "glueball_phase_sin2_companion": torch.sin(phase).square().float(),
                }
                for channel_name, obs in glue_obs.items():
                    if channel_name not in channels:
                        continue
                    obs_scales = obs[:, None, :].expand(t_len, n_scales, n_walkers)
                    glue_series = _masked_mean(obs_scales, plaquette_valid, dim=-1)
                    out[channel_name] = glue_series.transpose(0, 1).contiguous()

    return out


def _time_bootstrap_correlator_errors(
    series_stack: Tensor,  # [C, S, T]
    *,
    max_lag: int,
    use_connected: bool,
    n_bootstrap: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Vectorized bootstrap over time for all channels/scales.

    Returns:
        (correlator_err [C,S,L], mass_std [C,S])
    """
    if series_stack.ndim != 3:
        raise ValueError(f"Expected series_stack [C,S,T], got {tuple(series_stack.shape)}.")
    c_count, s_count, t_len = series_stack.shape
    l_count = int(max_lag) + 1
    corr_err = torch.zeros(
        (c_count, s_count, l_count), dtype=torch.float32, device=series_stack.device
    )
    mass_std = torch.full(
        (c_count, s_count), float("nan"), dtype=torch.float32, device=series_stack.device
    )
    if t_len <= 0 or c_count <= 0 or s_count <= 0 or int(n_bootstrap) <= 0:
        return corr_err, mass_std

    n_boot = int(max(1, n_bootstrap))
    gen = torch.Generator(device=series_stack.device)
    gen.manual_seed(int(seed))
    idx = torch.randint(0, t_len, (n_boot, t_len), generator=gen, device=series_stack.device)
    sampled = torch.gather(
        series_stack.unsqueeze(0).expand(n_boot, -1, -1, -1),
        dim=3,
        index=idx.unsqueeze(1).unsqueeze(1).expand(-1, c_count, s_count, -1),
    )  # [B,C,S,T]
    boot_corr = _fft_correlator_batched(
        sampled.reshape(-1, t_len),
        max_lag=int(max_lag),
        use_connected=bool(use_connected),
    ).reshape(n_boot, c_count, s_count, -1)
    corr_err = boot_corr.std(dim=0)
    return corr_err, mass_std


def _walker_bootstrap_mass_std(
    *,
    color: Tensor,  # [T,N,d]
    color_valid: Tensor,  # [T,N]
    positions: Tensor,  # [T,N,d]
    alive: Tensor,  # [T,N]
    force: Tensor,  # [T,N,d]
    cloning_scores: Tensor | None,  # [T,N]
    companions_distance: Tensor | None,  # [T,N]
    companions_clone: Tensor | None,  # [T,N]
    kernels: Tensor,  # [T,S,N,N]
    scales: Tensor,  # [S]
    pairwise_distances: Tensor,  # [T,N,N]
    channels: list[str],
    dt: float,
    config: CorrelatorConfig,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Tensor]:
    """Compute walker-bootstrap mass std with resampled+renormalized kernels.

    Notes:
    - Intended for moderate N only (caller enforces limits).
    - Keeps kernel family fixed; each sample uses kernel rows/cols gathered by
      bootstrap walker indices, then row-renormalized.
    """
    t_len, s_count, n_walkers, _ = kernels.shape
    if pairwise_distances.shape != (t_len, n_walkers, n_walkers):
        raise ValueError(
            "pairwise_distances must align with kernels [T,N,N], got "
            f"{tuple(pairwise_distances.shape)} vs [T={t_len},N={n_walkers},N={n_walkers}]."
        )
    if int(torch.as_tensor(scales).numel()) != int(s_count):
        raise ValueError(
            "scales must align with kernels scale axis, got "
            f"{int(torch.as_tensor(scales).numel())} vs S={s_count}."
        )
    if t_len <= 0 or s_count <= 0 or n_walkers <= 0 or n_bootstrap <= 0:
        return {
            channel: torch.full((s_count,), float("nan"), device=kernels.device)
            for channel in channels
        }

    gen = torch.Generator(device=kernels.device)
    gen.manual_seed(int(seed))
    idx_boot = torch.randint(
        0, n_walkers, (int(n_bootstrap), n_walkers), generator=gen, device=kernels.device
    )

    mass_samples = {
        channel: torch.full(
            (int(n_bootstrap), s_count), float("nan"), dtype=torch.float32, device=kernels.device
        )
        for channel in channels
    }
    eye = torch.eye(n_walkers, dtype=torch.bool, device=kernels.device).view(
        1, 1, n_walkers, n_walkers
    )
    for b in range(int(n_bootstrap)):
        idx = idx_boot[b]
        k_boot = kernels[:, :, idx][:, :, :, idx].clone()  # [T,S,N,N]
        k_boot = k_boot.masked_fill(eye, 0.0)
        row_sum = k_boot.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        k_boot = k_boot / row_sum
        d_boot = pairwise_distances[:, idx][:, :, idx].clone()  # [T,N,N]

        c_boot = color[:, idx, :]
        valid_boot = color_valid[:, idx]
        pos_boot = positions[:, idx, :]
        alive_boot = alive[:, idx]
        force_boot = force[:, idx, :]
        cloning_scores_boot = cloning_scores[:, idx] if cloning_scores is not None else None
        comp_dist_boot = None
        comp_clone_boot = None
        if companions_distance is not None and companions_clone is not None:
            comp_dist_src = companions_distance[:, idx]
            comp_clone_src = companions_clone[:, idx]
            comp_dist_boot = _remap_companion_indices_for_bootstrap(
                comp_dist_src,
                idx,
                n_walkers=n_walkers,
            )
            comp_clone_boot = _remap_companion_indices_for_bootstrap(
                comp_clone_src,
                idx,
                n_walkers=n_walkers,
            )
        series_boot = _compute_channel_series_from_kernels(
            color=c_boot,
            color_valid=valid_boot,
            positions=pos_boot,
            alive=alive_boot,
            force=force_boot,
            kernels=k_boot,
            scales=scales,
            pairwise_distances=d_boot,
            companions_distance=comp_dist_boot,
            companions_clone=comp_clone_boot,
            cloning_scores=cloning_scores_boot,
            channels=channels,
        )
        for channel in channels:
            series_cs = series_boot[channel]  # [S,T]
            corr_cs = _fft_correlator_batched(
                series_cs.float(),
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
            )
            for s_idx in range(s_count):
                mass_samples[channel][b, s_idx] = float(
                    _fit_mass_only(corr_cs[s_idx], dt=dt, config=config)
                )

    out: dict[str, Tensor] = {}
    for channel in channels:
        vals = mass_samples[channel]
        finite = torch.isfinite(vals)
        if not torch.any(finite):
            out[channel] = torch.full(
                (s_count,), float("nan"), dtype=torch.float32, device=kernels.device
            )
            continue
        vals = torch.where(finite, vals, torch.nan)
        out[channel] = _nanstd_compat(vals, dim=0)
    return out


def _compute_companion_per_scale_results_preserving_original(
    *,
    color: Tensor,  # [T,N,d]
    color_valid: Tensor,  # [T,N]
    positions: Tensor,  # [T,N,d]
    alive: Tensor,  # [T,N]
    cloning_scores: Tensor,  # [T,N]
    companions_distance: Tensor,  # [T,N]
    companions_clone: Tensor,  # [T,N]
    distance_ij: Tensor,  # [T,N]
    distance_ik: Tensor,  # [T,N]
    distance_jk: Tensor,  # [T,N]
    scales: Tensor,  # [S]
    channels: list[str],
    dt: float,
    config: CorrelatorConfig,
    baryon_flux_exp_alpha: float = 1.0,
) -> dict[str, list[ChannelCorrelatorResult]]:
    """Compute companion-channel correlators per scale using original source/sink estimators."""
    requested = [name for name in COMPANION_CHANNEL_MAP.values() if name in channels]
    if not requested:
        return {}

    t_len, n_walkers, d_color = color.shape
    n_scales = int(scales.numel())
    device = color.device
    cloning_scores = cloning_scores.to(device=device, dtype=torch.float32)
    if cloning_scores.shape != color.shape[:2]:
        raise ValueError(
            "cloning_scores must have shape [T,N] aligned with color, got "
            f"{tuple(cloning_scores.shape)} vs {tuple(color.shape[:2])}."
        )
    out: dict[str, list[ChannelCorrelatorResult]] = {name: [] for name in requested}

    if t_len == 0 or n_walkers == 0:
        n_lags = int(max(0, config.max_lag)) + 1
        zero_corr = torch.zeros(n_lags, dtype=torch.float32, device=device)
        zero_series = torch.zeros(0, dtype=torch.float32, device=device)
        for s_idx, scale_value in enumerate(scales.tolist()):
            for channel_name in requested:
                result = _build_result_from_precomputed_correlator(
                    channel_name=channel_name,
                    correlator=zero_corr,
                    dt=dt,
                    config=config,
                    n_samples=0,
                    series=zero_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = float(scale_value)
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out[channel_name].append(result)
        return out

    negative_one = torch.full_like(companions_distance, -1)
    finite_ij = torch.isfinite(distance_ij)
    finite_ik = torch.isfinite(distance_ik)
    finite_jk = torch.isfinite(distance_jk)
    triplet_radius = torch.maximum(distance_ij, torch.maximum(distance_ik, distance_jk))

    use_pair_family = any(
        name in out
        for name in (
            "scalar_companion",
            "scalar_raw_companion",
            "scalar_abs2_vacsub_companion",
            "pseudoscalar_companion",
            "scalar_score_directed_companion",
            "pseudoscalar_score_directed_companion",
            "scalar_score_weighted_companion",
            "pseudoscalar_score_weighted_companion",
            "vector_companion",
            "axial_vector_companion",
            "vector_score_directed_companion",
            "axial_vector_score_directed_companion",
            "vector_score_directed_longitudinal_companion",
            "axial_vector_score_directed_longitudinal_companion",
            "vector_score_directed_transverse_companion",
            "axial_vector_score_directed_transverse_companion",
            "vector_score_gradient_companion",
            "axial_vector_score_gradient_companion",
            "tensor_companion",
            "tensor_traceless_companion",
        )
    )
    use_triplet_family = any(
        name in out
        for name in (
            "nucleon_companion",
            "nucleon_score_signed_companion",
            "nucleon_score_abs_companion",
            "nucleon_flux_action_companion",
            "nucleon_flux_sin2_companion",
            "nucleon_flux_exp_companion",
            "glueball_companion",
            "glueball_phase_action_companion",
            "glueball_phase_sin2_companion",
        )
    )

    if d_color >= 3:
        color3 = color[..., :3]
    else:
        color3 = torch.zeros((t_len, n_walkers, 3), dtype=color.dtype, device=device)
        color3[..., :d_color] = color

    if positions.shape[-1] >= 3:
        positions3 = positions[..., :3]
    else:
        positions3 = torch.zeros((t_len, n_walkers, 3), dtype=torch.float32, device=device)
        positions3[..., : positions.shape[-1]] = positions
    positions_axis = positions3[..., 0]
    axis_extent = float((positions_axis.max() - positions_axis.min()).abs().item())
    if not math.isfinite(axis_extent) or axis_extent <= 0:
        axis_extent = 1.0

    for s_idx in range(n_scales):
        scale_value = float(scales[s_idx].item())
        pair_j_mask = finite_ij & (distance_ij <= scale_value)
        pair_k_mask = finite_ik & (distance_ik <= scale_value)
        triplet_mask = finite_ij & finite_ik & finite_jk & (triplet_radius <= scale_value)

        comp_dist_pair = torch.where(pair_j_mask, companions_distance, negative_one)
        comp_clone_pair = torch.where(pair_k_mask, companions_clone, negative_one)
        comp_dist_triplet = torch.where(triplet_mask, companions_distance, negative_one)
        comp_clone_triplet = torch.where(triplet_mask, companions_clone, negative_one)

        meson_out = None
        if use_pair_family and (
            "scalar_companion" in out
            or "scalar_raw_companion" in out
            or "pseudoscalar_companion" in out
        ):
            meson_out = compute_meson_phase_correlator_from_color(
                color=color3,
                color_valid=color_valid,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                operator_mode="standard",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "scalar_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="scalar_companion",
                    correlator=meson_out.scalar,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_out.n_valid_source_pairs),
                    series=meson_out.operator_scalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["scalar_companion"].append(result)
            if "scalar_raw_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="scalar_raw_companion",
                    correlator=meson_out.scalar_raw,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_out.n_valid_source_pairs),
                    series=meson_out.operator_scalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["scalar_raw_companion"].append(result)
            if "pseudoscalar_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="pseudoscalar_companion",
                    correlator=meson_out.pseudoscalar,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_out.n_valid_source_pairs),
                    series=meson_out.operator_pseudoscalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["pseudoscalar_companion"].append(result)

        if use_pair_family and "scalar_abs2_vacsub_companion" in out:
            meson_abs2_out = compute_meson_phase_correlator_from_color(
                color=color3,
                color_valid=color_valid,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                operator_mode="abs2_vacsub",
                scores=cloning_scores,
                frame_indices=None,
            )
            result = _build_result_from_precomputed_correlator(
                channel_name="scalar_abs2_vacsub_companion",
                correlator=meson_abs2_out.scalar,
                dt=dt,
                config=config,
                n_samples=int(meson_abs2_out.n_valid_source_pairs),
                series=meson_abs2_out.operator_scalar_series,
                correlator_err=None,
            )
            result.mass_fit["scale"] = scale_value
            result.mass_fit["scale_index"] = int(s_idx)
            result.mass_fit["source"] = "scaled_companion_source_sink"
            out["scalar_abs2_vacsub_companion"].append(result)

        if use_pair_family and (
            "scalar_score_directed_companion" in out
            or "pseudoscalar_score_directed_companion" in out
        ):
            meson_score_out = compute_meson_phase_correlator_from_color(
                color=color3,
                color_valid=color_valid,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                operator_mode="score_directed",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "scalar_score_directed_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="scalar_score_directed_companion",
                    correlator=meson_score_out.scalar,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_score_out.n_valid_source_pairs),
                    series=meson_score_out.operator_scalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["scalar_score_directed_companion"].append(result)
            if "pseudoscalar_score_directed_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="pseudoscalar_score_directed_companion",
                    correlator=meson_score_out.pseudoscalar,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_score_out.n_valid_source_pairs),
                    series=meson_score_out.operator_pseudoscalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["pseudoscalar_score_directed_companion"].append(result)

        if use_pair_family and (
            "scalar_score_weighted_companion" in out
            or "pseudoscalar_score_weighted_companion" in out
        ):
            meson_weighted_out = compute_meson_phase_correlator_from_color(
                color=color3,
                color_valid=color_valid,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                operator_mode="score_weighted",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "scalar_score_weighted_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="scalar_score_weighted_companion",
                    correlator=meson_weighted_out.scalar,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_weighted_out.n_valid_source_pairs),
                    series=meson_weighted_out.operator_scalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["scalar_score_weighted_companion"].append(result)
            if "pseudoscalar_score_weighted_companion" in out:
                result = _build_result_from_precomputed_correlator(
                    channel_name="pseudoscalar_score_weighted_companion",
                    correlator=meson_weighted_out.pseudoscalar,
                    dt=dt,
                    config=config,
                    n_samples=int(meson_weighted_out.n_valid_source_pairs),
                    series=meson_weighted_out.operator_pseudoscalar_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["pseudoscalar_score_weighted_companion"].append(result)

        vector_out = None
        if use_pair_family and ("vector_companion" in out or "axial_vector_companion" in out):
            vector_out = compute_vector_meson_correlator_from_color_positions(
                color=color3,
                color_valid=color_valid,
                positions=positions3,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                use_unit_displacement=False,
                frame_indices=None,
            )
            if "vector_companion" in out:
                vec_series = torch.linalg.vector_norm(
                    vector_out.operator_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="vector_companion",
                    correlator=vector_out.vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_out.n_valid_source_pairs),
                    series=vec_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["vector_companion"].append(result)
            if "axial_vector_companion" in out:
                axial_series = torch.linalg.vector_norm(
                    vector_out.operator_axial_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="axial_vector_companion",
                    correlator=vector_out.axial_vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_out.n_valid_source_pairs),
                    series=axial_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["axial_vector_companion"].append(result)

        if use_pair_family and (
            "vector_score_directed_companion" in out
            or "axial_vector_score_directed_companion" in out
        ):
            vector_score_out = compute_vector_meson_correlator_from_color_positions(
                color=color3,
                color_valid=color_valid,
                positions=positions3,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                use_unit_displacement=False,
                operator_mode="score_directed",
                projection_mode="full",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "vector_score_directed_companion" in out:
                vec_series = torch.linalg.vector_norm(
                    vector_score_out.operator_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="vector_score_directed_companion",
                    correlator=vector_score_out.vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_out.n_valid_source_pairs),
                    series=vec_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["vector_score_directed_companion"].append(result)
            if "axial_vector_score_directed_companion" in out:
                axial_series = torch.linalg.vector_norm(
                    vector_score_out.operator_axial_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="axial_vector_score_directed_companion",
                    correlator=vector_score_out.axial_vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_out.n_valid_source_pairs),
                    series=axial_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["axial_vector_score_directed_companion"].append(result)

        if use_pair_family and (
            "vector_score_directed_longitudinal_companion" in out
            or "axial_vector_score_directed_longitudinal_companion" in out
        ):
            vector_score_long = compute_vector_meson_correlator_from_color_positions(
                color=color3,
                color_valid=color_valid,
                positions=positions3,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                use_unit_displacement=False,
                operator_mode="score_directed",
                projection_mode="longitudinal",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "vector_score_directed_longitudinal_companion" in out:
                vec_series = torch.linalg.vector_norm(
                    vector_score_long.operator_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="vector_score_directed_longitudinal_companion",
                    correlator=vector_score_long.vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_long.n_valid_source_pairs),
                    series=vec_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["vector_score_directed_longitudinal_companion"].append(result)
            if "axial_vector_score_directed_longitudinal_companion" in out:
                axial_series = torch.linalg.vector_norm(
                    vector_score_long.operator_axial_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="axial_vector_score_directed_longitudinal_companion",
                    correlator=vector_score_long.axial_vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_long.n_valid_source_pairs),
                    series=axial_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["axial_vector_score_directed_longitudinal_companion"].append(result)

        if use_pair_family and (
            "vector_score_directed_transverse_companion" in out
            or "axial_vector_score_directed_transverse_companion" in out
        ):
            vector_score_trans = compute_vector_meson_correlator_from_color_positions(
                color=color3,
                color_valid=color_valid,
                positions=positions3,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                use_unit_displacement=False,
                operator_mode="score_directed",
                projection_mode="transverse",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "vector_score_directed_transverse_companion" in out:
                vec_series = torch.linalg.vector_norm(
                    vector_score_trans.operator_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="vector_score_directed_transverse_companion",
                    correlator=vector_score_trans.vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_trans.n_valid_source_pairs),
                    series=vec_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["vector_score_directed_transverse_companion"].append(result)
            if "axial_vector_score_directed_transverse_companion" in out:
                axial_series = torch.linalg.vector_norm(
                    vector_score_trans.operator_axial_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="axial_vector_score_directed_transverse_companion",
                    correlator=vector_score_trans.axial_vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_trans.n_valid_source_pairs),
                    series=axial_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["axial_vector_score_directed_transverse_companion"].append(result)

        if use_pair_family and (
            "vector_score_gradient_companion" in out
            or "axial_vector_score_gradient_companion" in out
        ):
            vector_score_grad = compute_vector_meson_correlator_from_color_positions(
                color=color3,
                color_valid=color_valid,
                positions=positions3,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                use_unit_displacement=False,
                operator_mode="score_gradient",
                projection_mode="full",
                scores=cloning_scores,
                frame_indices=None,
            )
            if "vector_score_gradient_companion" in out:
                vec_series = torch.linalg.vector_norm(
                    vector_score_grad.operator_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="vector_score_gradient_companion",
                    correlator=vector_score_grad.vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_grad.n_valid_source_pairs),
                    series=vec_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["vector_score_gradient_companion"].append(result)
            if "axial_vector_score_gradient_companion" in out:
                axial_series = torch.linalg.vector_norm(
                    vector_score_grad.operator_axial_vector_series, dim=-1
                ).float()
                result = _build_result_from_precomputed_correlator(
                    channel_name="axial_vector_score_gradient_companion",
                    correlator=vector_score_grad.axial_vector,
                    dt=dt,
                    config=config,
                    n_samples=int(vector_score_grad.n_valid_source_pairs),
                    series=axial_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out["axial_vector_score_gradient_companion"].append(result)

        if "tensor_companion" in out or "tensor_traceless_companion" in out:
            tensor_out = compute_tensor_momentum_correlator_from_color_positions(
                color=color3,
                color_valid=color_valid,
                positions=positions3,
                positions_axis=positions_axis,
                alive=alive,
                companions_distance=comp_dist_pair,
                companions_clone=comp_clone_pair,
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
                pair_selection="both",
                eps=1e-12,
                momentum_mode_max=0,
                projection_length=axis_extent,
                bounds=None,
                pbc=False,
                compute_bootstrap_errors=False,
                n_bootstrap=0,
                frame_indices=None,
                momentum_axis=0,
            )
            tensor_series = torch.sqrt(
                torch.clamp_min(
                    tensor_out.momentum_operator_cos_series[0].float().pow(2).sum(dim=0)
                    + tensor_out.momentum_operator_sin_series[0].float().pow(2).sum(dim=0),
                    0.0,
                )
            )
            for channel_name in ("tensor_companion", "tensor_traceless_companion"):
                if channel_name not in out:
                    continue
                result = _build_result_from_precomputed_correlator(
                    channel_name=channel_name,
                    correlator=tensor_out.momentum_contracted_correlator[0],
                    dt=dt,
                    config=config,
                    n_samples=int(tensor_out.momentum_valid_frames),
                    series=tensor_series,
                    correlator_err=(
                        tensor_out.momentum_contracted_correlator_err[0]
                        if tensor_out.momentum_contracted_correlator_err is not None
                        else None
                    ),
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out[channel_name].append(result)

        baryon_channel_modes: list[tuple[str, str]] = [
            ("nucleon_companion", "det_abs"),
            ("nucleon_score_signed_companion", "score_signed"),
            ("nucleon_score_abs_companion", "score_abs"),
            ("nucleon_flux_action_companion", "flux_action"),
            ("nucleon_flux_sin2_companion", "flux_sin2"),
            ("nucleon_flux_exp_companion", "flux_exp"),
        ]
        if use_triplet_family and any(name in out for name, _ in baryon_channel_modes):
            for channel_name, operator_mode in baryon_channel_modes:
                if channel_name not in out:
                    continue
                baryon_out = compute_baryon_correlator_from_color(
                    color=color3,
                    color_valid=color_valid,
                    alive=alive,
                    companions_distance=comp_dist_triplet,
                    companions_clone=comp_clone_triplet,
                    max_lag=int(config.max_lag),
                    use_connected=bool(config.use_connected),
                    eps=1e-12,
                    operator_mode=operator_mode,
                    flux_exp_alpha=float(baryon_flux_exp_alpha),
                    scores=cloning_scores,
                    frame_indices=None,
                )
                result = _build_result_from_precomputed_correlator(
                    channel_name=channel_name,
                    correlator=baryon_out.correlator,
                    dt=dt,
                    config=config,
                    n_samples=int(baryon_out.n_valid_source_triplets),
                    series=baryon_out.operator_baryon_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out[channel_name].append(result)

        glueball_channel_modes: list[tuple[str, str]] = [
            ("glueball_companion", "re_plaquette"),
            ("glueball_phase_action_companion", "phase_action"),
            ("glueball_phase_sin2_companion", "phase_sin2"),
        ]
        if use_triplet_family and any(name in out for name, _ in glueball_channel_modes):
            for channel_name, operator_mode in glueball_channel_modes:
                if channel_name not in out:
                    continue
                glue_out = compute_glueball_color_correlator_from_color(
                    color=color3,
                    color_valid=color_valid,
                    alive=alive,
                    companions_distance=comp_dist_triplet,
                    companions_clone=comp_clone_triplet,
                    max_lag=int(config.max_lag),
                    use_connected=bool(config.use_connected),
                    eps=1e-12,
                    operator_mode=operator_mode,
                    use_action_form=False,
                    frame_indices=None,
                    use_momentum_projection=False,
                    compute_bootstrap_errors=False,
                    n_bootstrap=0,
                )
                result = _build_result_from_precomputed_correlator(
                    channel_name=channel_name,
                    correlator=glue_out.correlator,
                    dt=dt,
                    config=config,
                    n_samples=int(glue_out.n_valid_source_triplets),
                    series=glue_out.operator_glueball_series,
                    correlator_err=None,
                )
                result.mass_fit["scale"] = scale_value
                result.mass_fit["scale_index"] = int(s_idx)
                result.mass_fit["source"] = "scaled_companion_source_sink"
                out[channel_name].append(result)

    return out


def compute_multiscale_strong_force_channels(
    history: RunHistory,
    *,
    config: MultiscaleStrongForceConfig,
    channels: list[str] | None = None,
) -> MultiscaleStrongForceOutput:
    """Compute multiscale strong-force channel results and select best scales."""
    if config.kernel_type not in KERNEL_TYPES:
        raise ValueError(f"kernel_type must be one of {KERNEL_TYPES}, got {config.kernel_type!r}.")
    if config.bootstrap_mode not in BOOTSTRAP_MODES:
        raise ValueError(
            f"bootstrap_mode must be one of {BOOTSTRAP_MODES}, got {config.bootstrap_mode!r}."
        )
    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
        mc_time_index=config.mc_time_index,
    )
    if not frame_indices:
        empty = torch.empty(0, dtype=torch.float32)
        return MultiscaleStrongForceOutput(
            scales=empty,
            frame_indices=[],
            per_scale_results={},
            best_results={},
            best_scale_index={},
            series_by_channel={},
            bootstrap_mode_applied="none",
            notes=["No valid frame indices for multiscale analysis."],
            bootstrap_mass_std=None,
        )

    requested = [str(c).strip() for c in (channels or BASE_CHANNELS) if str(c).strip()]
    requested = [c for c in requested if c in SUPPORTED_CHANNELS]
    if not requested:
        requested = list(BASE_CHANNELS)
    # Always include companion-channel measurements for requested override families.
    for base_name, companion_name in COMPANION_CHANNEL_MAP.items():
        if base_name in requested and companion_name not in requested:
            requested.append(companion_name)
    requested_companion_channels = [
        channel_name
        for channel_name in requested
        if channel_name in COMPANION_CHANNEL_MAP.values()
    ]

    if config.ell0 is not None:
        ell0 = float(config.ell0)
    else:
        try:
            ell0 = float(estimate_ell0(history))
        except Exception:
            ell0 = 1.0
    start_idx = int(frame_indices[0])
    end_idx = int(frame_indices[-1]) + 1
    color_full, color_valid_full = compute_color_states_batch(
        history,
        start_idx,
        float(config.h_eff),
        float(config.mass),
        ell0,
        end_idx=end_idx,
    )
    frame_offsets = [int(frame_idx - start_idx) for frame_idx in frame_indices]
    frame_offsets_t = torch.as_tensor(frame_offsets, dtype=torch.long, device=color_full.device)
    color = color_full.index_select(0, frame_offsets_t)
    color_valid = color_valid_full.index_select(0, frame_offsets_t)
    frame_ids_t = torch.as_tensor(
        frame_indices, dtype=torch.long, device=history.x_before_clone.device
    )
    positions = history.x_before_clone.index_select(0, frame_ids_t).float()
    alive = history.alive_mask.index_select(0, frame_ids_t - 1).to(dtype=torch.bool)
    force = history.force_viscous.index_select(0, frame_ids_t - 1).float()
    companions_distance = torch.as_tensor(
        history.companions_distance.index_select(0, frame_ids_t - 1),
        device=color.device,
        dtype=torch.long,
    )
    companions_clone = torch.as_tensor(
        history.companions_clone.index_select(0, frame_ids_t - 1),
        device=color.device,
        dtype=torch.long,
    )
    cloning_scores = torch.as_tensor(
        history.cloning_scores.index_select(0, frame_ids_t - 1),
        device=color.device,
        dtype=torch.float32,
    )

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
        device=color.device,
        dtype=torch.float32,
    )
    n_scales = int(scales.numel())
    n_frames = len(frame_indices)
    n_walkers = int(alive.shape[1])
    series_by_channel: dict[str, Tensor] = {
        channel: torch.zeros((n_scales, n_frames), dtype=torch.float32, device=color.device)
        for channel in requested
    }
    companion_distance_ij = None
    companion_distance_ik = None
    companion_distance_jk = None
    if requested_companion_channels:
        companion_distance_ij = torch.full(
            (n_frames, n_walkers), float("inf"), dtype=torch.float32, device=color.device
        )
        companion_distance_ik = torch.full(
            (n_frames, n_walkers), float("inf"), dtype=torch.float32, device=color.device
        )
        companion_distance_jk = torch.full(
            (n_frames, n_walkers), float("inf"), dtype=torch.float32, device=color.device
        )
    frame_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(frame_indices)}
    for (
        frame_ids_chunk,
        distances_chunk,
        kernels_chunk,
        _,
    ) in iter_smeared_kernel_batches_from_history(
        history,
        scales=scales,
        method=str(config.kernel_distance_method),
        frame_indices=frame_indices,
        batch_size=int(max(1, config.kernel_batch_size)),
        edge_weight_mode=str(config.edge_weight_mode),
        assume_all_alive=bool(config.kernel_assume_all_alive),
        kernel_type=str(config.kernel_type),
        device=color.device,
        dtype=torch.float32,
    ):
        pos_idx = [frame_to_pos[int(frame_idx)] for frame_idx in frame_ids_chunk]
        pos_t = torch.as_tensor(pos_idx, dtype=torch.long, device=color.device)
        if requested_companion_channels:
            comp_j_chunk = companions_distance.index_select(0, pos_t)
            comp_k_chunk = companions_clone.index_select(0, pos_t)
            anchor_rows = (
                torch.arange(n_walkers, device=color.device, dtype=torch.long)
                .view(1, -1)
                .expand(comp_j_chunk.shape[0], -1)
            )
            dist_ij_chunk, _ = _safe_gather_pairwise_distances(
                distances_chunk, anchor_rows, comp_j_chunk
            )
            dist_ik_chunk, _ = _safe_gather_pairwise_distances(
                distances_chunk, anchor_rows, comp_k_chunk
            )
            dist_jk_chunk, _ = _safe_gather_pairwise_distances(
                distances_chunk, comp_j_chunk, comp_k_chunk
            )
            companion_distance_ij.index_copy_(0, pos_t, dist_ij_chunk.float())
            companion_distance_ik.index_copy_(0, pos_t, dist_ik_chunk.float())
            companion_distance_jk.index_copy_(0, pos_t, dist_jk_chunk.float())
        chunk_series = _compute_channel_series_from_kernels(
            color=color.index_select(0, pos_t),
            color_valid=color_valid.index_select(0, pos_t),
            positions=positions.index_select(0, pos_t),
            alive=alive.index_select(0, pos_t),
            force=force.index_select(0, pos_t),
            kernels=kernels_chunk,
            scales=scales,
            pairwise_distances=distances_chunk,
            companions_distance=companions_distance.index_select(0, pos_t),
            companions_clone=companions_clone.index_select(0, pos_t),
            cloning_scores=cloning_scores.index_select(0, pos_t),
            channels=requested,
        )
        for channel in requested:
            series_by_channel[channel][:, pos_t] = chunk_series[channel]

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

    companion_override_results: dict[str, list[ChannelCorrelatorResult]] = {}
    if (
        requested_companion_channels
        and companion_distance_ij is not None
        and companion_distance_ik is not None
        and companion_distance_jk is not None
    ):
        companion_override_results = _compute_companion_per_scale_results_preserving_original(
            color=color,
            color_valid=color_valid,
            positions=positions,
            alive=alive,
            cloning_scores=cloning_scores,
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            distance_ij=companion_distance_ij,
            distance_ik=companion_distance_ik,
            distance_jk=companion_distance_jk,
            scales=scales,
            channels=requested,
            dt=dt,
            config=correlator_cfg,
            baryon_flux_exp_alpha=float(config.companion_baryon_flux_exp_alpha),
        )
        for channel_name, per_scale in companion_override_results.items():
            if per_scale and channel_name in series_by_channel:
                series_by_channel[channel_name] = torch.stack(
                    [result.series.float() for result in per_scale], dim=0
                )

    channel_names = list(series_by_channel.keys())
    stack = torch.stack([series_by_channel[name] for name in channel_names], dim=0)  # [C,S,T]
    c_count, s_count, t_len = stack.shape
    corr_stack = _fft_correlator_batched(
        stack.reshape(c_count * s_count, t_len),
        max_lag=int(correlator_cfg.max_lag),
        use_connected=bool(correlator_cfg.use_connected),
    ).reshape(c_count, s_count, -1)

    corr_err_stack = None
    mass_std_time = None
    if bool(config.compute_bootstrap_errors) and config.bootstrap_mode in {"time", "hybrid"}:
        corr_err_stack, _ = _time_bootstrap_correlator_errors(
            stack,
            max_lag=int(correlator_cfg.max_lag),
            use_connected=bool(correlator_cfg.use_connected),
            n_bootstrap=int(config.n_bootstrap),
            seed=int(config.bootstrap_seed),
        )
        mass_std_time = torch.full(
            (c_count, s_count), float("nan"), dtype=torch.float32, device=stack.device
        )

    notes: list[str] = []
    mass_std_walker_by_channel: dict[str, Tensor] | None = None
    mode_applied = config.bootstrap_mode if bool(config.compute_bootstrap_errors) else "none"
    if bool(config.compute_bootstrap_errors) and config.bootstrap_mode in {"walker", "hybrid"}:
        n_walkers = int(history.N)
        if n_walkers > int(config.walker_bootstrap_max_walkers):
            notes.append(
                "walker bootstrap skipped: "
                f"N={n_walkers} exceeds walker_bootstrap_max_walkers={config.walker_bootstrap_max_walkers}."
            )
            if config.bootstrap_mode == "walker":
                mode_applied = "none"
            else:
                mode_applied = "time"
        else:
            dist_frame_ids, distances_all = compute_pairwise_distance_matrices_from_history(
                history,
                method=str(config.kernel_distance_method),
                frame_indices=frame_indices,
                batch_size=int(max(1, config.kernel_batch_size)),
                edge_weight_mode=str(config.edge_weight_mode),
                assume_all_alive=bool(config.kernel_assume_all_alive),
                device=color.device,
                dtype=torch.float32,
            )
            if dist_frame_ids != frame_indices:
                raise RuntimeError(
                    "Pairwise-distance frame order mismatch during walker bootstrap: "
                    f"{dist_frame_ids} vs {frame_indices}."
                )
            kernels_all = compute_smeared_kernels_from_distances(
                distances_all,
                scales,
                kernel_type=str(config.kernel_type),
            )
            n_walk_boot = int(
                max(1, min(int(config.n_bootstrap), int(config.walker_bootstrap_max_samples)))
            )
            mass_std_walker_by_channel = _walker_bootstrap_mass_std(
                color=color,
                color_valid=color_valid,
                positions=positions,
                alive=alive,
                force=force,
                cloning_scores=cloning_scores,
                companions_distance=companions_distance,
                companions_clone=companions_clone,
                kernels=kernels_all,
                scales=scales,
                pairwise_distances=distances_all,
                channels=channel_names,
                dt=dt,
                config=correlator_cfg,
                n_bootstrap=n_walk_boot,
                seed=int(config.bootstrap_seed) + 17,
            )
            if config.bootstrap_mode == "walker":
                mode_applied = "walker"
            elif config.bootstrap_mode == "hybrid":
                mode_applied = "hybrid"

    per_scale_results: dict[str, list[ChannelCorrelatorResult]] = {}
    best_results: dict[str, ChannelCorrelatorResult] = {}
    best_scale_index: dict[str, int] = {}
    bootstrap_mass_std_out: dict[str, Tensor] = {}
    no_best_channels: list[str] = []
    for c_idx, channel in enumerate(channel_names):
        if companion_override_results.get(channel):
            channel_results = companion_override_results[channel]
        else:
            channel_results = []
            for s_idx in range(s_count):
                corr_err = None
                if corr_err_stack is not None:
                    corr_err = corr_err_stack[c_idx, s_idx]
                result = _build_result_from_precomputed_correlator(
                    channel_name=channel,
                    correlator=corr_stack[c_idx, s_idx],
                    dt=dt,
                    config=correlator_cfg,
                    n_samples=t_len,
                    series=stack[c_idx, s_idx],
                    correlator_err=corr_err,
                )
                result.mass_fit["scale"] = float(scales[s_idx].item())
                result.mass_fit["scale_index"] = int(s_idx)
                channel_results.append(result)
        per_scale_results[channel] = channel_results

        best_idx = _select_best_scale(
            channel_results,
            min_r2=float(config.best_min_r2),
            min_windows=int(config.best_min_windows),
            max_error_pct=float(config.best_max_error_pct),
            remove_artifacts=bool(config.best_remove_artifacts),
        )
        if best_idx is None:
            best_scale_index[channel] = -1
            no_best_channels.append(channel)
            continue
        best_scale_index[channel] = int(best_idx)
        best_result = channel_results[best_idx]

        # Attach bootstrap mass spread if available.
        base_mass_err = float(best_result.mass_fit.get("mass_error", float("inf")))
        comp_mass_err = float("nan")
        if (
            channel not in companion_override_results
            and mass_std_walker_by_channel is not None
            and channel in mass_std_walker_by_channel
        ):
            std_vec = mass_std_walker_by_channel[channel]
            bootstrap_mass_std_out[channel] = std_vec.detach().clone()
            comp_mass_err = float(std_vec[best_idx].item())
        elif channel not in companion_override_results and mass_std_time is not None:
            comp_mass_err = float(mass_std_time[c_idx, best_idx].item())
        if math.isfinite(comp_mass_err) and comp_mass_err >= 0:
            best_result.mass_fit["bootstrap_mass_error"] = comp_mass_err
            if math.isfinite(base_mass_err) and base_mass_err >= 0:
                best_result.mass_fit["mass_error"] = float(
                    math.sqrt(base_mass_err * base_mass_err + comp_mass_err * comp_mass_err)
                )
            else:
                best_result.mass_fit["mass_error"] = comp_mass_err

        best_results[channel] = best_result

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

    if not bootstrap_mass_std_out:
        bootstrap_mass_std_out = None

    return MultiscaleStrongForceOutput(
        scales=scales.detach().clone(),
        frame_indices=list(frame_indices),
        per_scale_results=per_scale_results,
        best_results=best_results,
        best_scale_index=best_scale_index,
        series_by_channel={k: v.detach().clone() for k, v in series_by_channel.items()},
        bootstrap_mode_applied=mode_applied,
        notes=notes,
        bootstrap_mass_std=bootstrap_mass_std_out,
    )
