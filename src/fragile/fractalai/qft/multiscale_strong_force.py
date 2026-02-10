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
)
from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    ChannelCorrelatorResult,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
    compute_smeared_kernels_from_history,
    compute_smeared_kernels_from_distances,
    iter_smeared_kernel_batches_from_history,
    select_interesting_scales_from_history,
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
    "vector": "vector_companion",
    "axial_vector": "axial_vector_companion",
    "tensor": "tensor_companion",
    "tensor_traceless": "tensor_traceless_companion",
    "nucleon": "nucleon_companion",
    "glueball": "glueball_companion",
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

    # Bootstrap controls
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100
    bootstrap_seed: int = 12345
    bootstrap_mode: str = "hybrid"
    walker_bootstrap_max_walkers: int = 512
    walker_bootstrap_max_samples: int = 64


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


def _select_best_scale(results: list[ChannelCorrelatorResult]) -> int:
    """Select best scale index using AIC/R²/mass-error fallback."""
    best_idx = 0
    best_key = (float("inf"), float("inf"), float("inf"))
    for idx, result in enumerate(results):
        fit = result.mass_fit or {}
        mass = float(fit.get("mass", 0.0))
        if not math.isfinite(mass) or mass <= 0:
            continue
        best_window = fit.get("best_window", {}) if isinstance(fit.get("best_window", {}), dict) else {}
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
        raise ValueError(f"distances must be square on last two axes, got {tuple(distances.shape)}.")
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
    flat_idx = row_safe * n + col_safe
    gathered, in_flat = _safe_gather_2d(distances.reshape(t_len, n * n), flat_idx)
    return gathered, valid & in_flat


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
    color: Tensor,        # [T, N, d] complex
    color_valid: Tensor,  # [T, N] bool
    positions: Tensor,    # [T, N, d] float
    alive: Tensor,        # [T, N] bool
    force: Tensor,        # [T, N, d] float
    kernels: Tensor,      # [T, S, N, N] float
    scales: Tensor | None = None,  # [S] float, used by companion hard-threshold channels
    pairwise_distances: Tensor | None = None,  # [T, N, N] float, used by companion channels
    companions_distance: Tensor | None = None,  # [T, N] long
    companions_clone: Tensor | None = None,     # [T, N] long
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
        out["pseudoscalar"] = _masked_mean(inner_im, mask_color, dim=-1).transpose(0, 1).contiguous()

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
    requested_companion_channels = [name for name in COMPANION_CHANNEL_MAP.values() if name in channels]
    if requested_companion_channels:
        if companions_distance is None or companions_clone is None:
            for channel in requested_companion_channels:
                out[channel] = torch.zeros((n_scales, t_len), dtype=torch.float32, device=device)
            return out

        comp_j = companions_distance.to(device=device, dtype=torch.long)
        comp_k = companions_clone.to(device=device, dtype=torch.long)
        if comp_j.shape != alive.shape or comp_k.shape != alive.shape:
            raise ValueError(
                "companion arrays must have shape [T,N] aligned with alive/color tensors, got "
                f"{tuple(comp_j.shape)} and {tuple(comp_k.shape)} vs {tuple(alive.shape)}."
            )

        # Gather smeared companion fields once; all companion channels reuse them.
        c_j, in_j = _safe_gather_4d_by_2d_indices(color_sm, comp_j)
        c_k, in_k = _safe_gather_4d_by_2d_indices(color_sm, comp_k)
        x_j, _ = _safe_gather_4d_by_2d_indices(x_sm, comp_j)
        x_k, _ = _safe_gather_4d_by_2d_indices(x_sm, comp_k)

        alive_j_2d, in_j_alive = _safe_gather_2d(alive, comp_j)
        alive_k_2d, in_k_alive = _safe_gather_2d(alive, comp_k)
        valid_j_2d, in_j_valid = _safe_gather_2d(color_valid, comp_j)
        valid_k_2d, in_k_valid = _safe_gather_2d(color_valid, comp_k)

        anchor_idx = torch.arange(alive.shape[1], device=device, dtype=torch.long).view(1, -1)
        distinct = (comp_j != anchor_idx) & (comp_k != anchor_idx) & (comp_j != comp_k)
        base_anchor_valid = alive & color_valid
        base_pair_j = base_anchor_valid & alive_j_2d & valid_j_2d & in_j_alive & in_j_valid & (comp_j != anchor_idx)
        base_pair_k = base_anchor_valid & alive_k_2d & valid_k_2d & in_k_alive & in_k_valid & (comp_k != anchor_idx)
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

        inner_j = torch.einsum("tsnd,tsnd->tsn", torch.conj(color_sm), c_j)
        inner_k = torch.einsum("tsnd,tsnd->tsn", torch.conj(color_sm), c_k)
        finite_inner_j = torch.isfinite(inner_j.real) & torch.isfinite(inner_j.imag)
        finite_inner_k = torch.isfinite(inner_k.real) & torch.isfinite(inner_k.imag)
        eps = 1e-12
        pair_j_valid = base_pair_j[:, None, :].expand(t_len, n_scales, alive.shape[1])
        pair_k_valid = base_pair_k[:, None, :].expand(t_len, n_scales, alive.shape[1])
        pair_j_valid = pair_j_valid & in_j & finite_inner_j & (inner_j.abs() > eps)
        pair_k_valid = pair_k_valid & in_k & finite_inner_k & (inner_k.abs() > eps)

        if "scalar_companion" in channels or "pseudoscalar_companion" in channels:
            inner_pair = torch.stack([inner_j, inner_k], dim=-1)  # [T,S,N,2]
            pair_valid = torch.stack([pair_j_valid, pair_k_valid], dim=-1)  # [T,S,N,2]
            if "scalar_companion" in channels:
                scalar_comp = _masked_mean_multi(inner_pair.real.float(), pair_valid, dims=(-1, -2))
                out["scalar_companion"] = scalar_comp.transpose(0, 1).contiguous()
            if "pseudoscalar_companion" in channels:
                pseu_comp = _masked_mean_multi(inner_pair.imag.float(), pair_valid, dims=(-1, -2))
                out["pseudoscalar_companion"] = pseu_comp.transpose(0, 1).contiguous()

        if "vector_companion" in channels or "axial_vector_companion" in channels:
            disp_j = x_j - x_sm
            disp_k = x_k - x_sm
            pair_valid = torch.stack([pair_j_valid, pair_k_valid], dim=-1)  # [T,S,N,2]
            if "vector_companion" in channels:
                vec_j = inner_j.real.float().unsqueeze(-1) * disp_j
                vec_k = inner_k.real.float().unsqueeze(-1) * disp_k
                vec_pair = torch.stack([vec_j, vec_k], dim=-2)  # [T,S,N,2,d]
                vec_mask = pair_valid.unsqueeze(-1).expand_as(vec_pair)
                vec_mean = _masked_mean_multi(vec_pair, vec_mask, dims=(-2, -3))  # [T,S,d]
                out["vector_companion"] = torch.linalg.vector_norm(vec_mean, dim=-1).transpose(0, 1).contiguous()
            if "axial_vector_companion" in channels:
                axial_j = inner_j.imag.float().unsqueeze(-1) * disp_j
                axial_k = inner_k.imag.float().unsqueeze(-1) * disp_k
                axial_pair = torch.stack([axial_j, axial_k], dim=-2)  # [T,S,N,2,d]
                axial_mask = pair_valid.unsqueeze(-1).expand_as(axial_pair)
                axial_mean = _masked_mean_multi(axial_pair, axial_mask, dims=(-2, -3))  # [T,S,d]
                out["axial_vector_companion"] = (
                    torch.linalg.vector_norm(axial_mean, dim=-1).transpose(0, 1).contiguous()
                )

        if (
            "tensor_companion" in channels
            or "tensor_traceless_companion" in channels
        ):
            if positions.shape[-1] >= 3:
                disp_j = x_j - x_sm
                disp_k = x_k - x_sm
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

                t_j = inner_j.real.float().unsqueeze(-1) * q_j
                t_k = inner_k.real.float().unsqueeze(-1) * q_k
                tensor_pair = torch.stack([t_j, t_k], dim=-2)  # [T,S,N,2,5]
                pair_valid = torch.stack([pair_j_valid, pair_k_valid], dim=-1)  # [T,S,N,2]
                tensor_mask = pair_valid.unsqueeze(-1).expand_as(tensor_pair)
                tensor_mean = _masked_mean_multi(tensor_pair, tensor_mask, dims=(-2, -3))  # [T,S,5]
                tensor_series = torch.linalg.vector_norm(tensor_mean, dim=-1).transpose(0, 1).contiguous()

                if "tensor_companion" in channels:
                    out["tensor_companion"] = tensor_series
                if "tensor_traceless_companion" in channels:
                    out["tensor_traceless_companion"] = tensor_series
            else:
                if "tensor_companion" in channels:
                    out["tensor_companion"] = torch.zeros(
                        (n_scales, t_len),
                        dtype=torch.float32,
                        device=device,
                    )
                if "tensor_traceless_companion" in channels:
                    out["tensor_traceless_companion"] = torch.zeros(
                        (n_scales, t_len),
                        dtype=torch.float32,
                        device=device,
                    )

        if "nucleon_companion" in channels:
            if color.shape[-1] >= 3:
                c_i3 = color_sm[..., :3]
                c_j3 = c_j[..., :3]
                c_k3 = c_k[..., :3]
                det = torch.abs(
                    c_i3[..., 0] * (c_j3[..., 1] * c_k3[..., 2] - c_j3[..., 2] * c_k3[..., 1])
                    - c_i3[..., 1] * (c_j3[..., 0] * c_k3[..., 2] - c_j3[..., 2] * c_k3[..., 0])
                    + c_i3[..., 2] * (c_j3[..., 0] * c_k3[..., 1] - c_j3[..., 1] * c_k3[..., 0])
                ).float()  # [T,S,N]
                det_valid = base_triplet_valid[:, None, :].expand(t_len, n_scales, alive.shape[1])
                det_valid = det_valid & in_j & in_k & torch.isfinite(det) & (det > eps)
                nucleon_comp = _masked_mean(det, det_valid, dim=-1)
                out["nucleon_companion"] = nucleon_comp.transpose(0, 1).contiguous()
            else:
                out["nucleon_companion"] = torch.zeros((n_scales, t_len), dtype=torch.float32, device=device)

        if "glueball_companion" in channels:
            z_ij = inner_j
            z_jk = torch.einsum("tsnd,tsnd->tsn", torch.conj(c_j), c_k)
            z_ki = torch.einsum("tsnd,tsnd->tsn", torch.conj(c_k), color_sm)
            plaquette = z_ij * z_jk * z_ki
            obs = plaquette.real.float()
            glue_valid = base_triplet_valid[:, None, :].expand(t_len, n_scales, alive.shape[1])
            glue_valid = (
                glue_valid
                & in_j
                & in_k
                & torch.isfinite(plaquette.real)
                & torch.isfinite(plaquette.imag)
                & (z_ij.abs() > eps)
                & (z_jk.abs() > eps)
                & (z_ki.abs() > eps)
            )
            glue_comp = _masked_mean(obs, glue_valid, dim=-1)
            out["glueball_companion"] = glue_comp.transpose(0, 1).contiguous()

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
    corr_err = torch.zeros((c_count, s_count, l_count), dtype=torch.float32, device=series_stack.device)
    mass_std = torch.full((c_count, s_count), float("nan"), dtype=torch.float32, device=series_stack.device)
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
    color: Tensor,        # [T,N,d]
    color_valid: Tensor,  # [T,N]
    positions: Tensor,    # [T,N,d]
    alive: Tensor,        # [T,N]
    force: Tensor,        # [T,N,d]
    companions_distance: Tensor | None,  # [T,N]
    companions_clone: Tensor | None,     # [T,N]
    kernels: Tensor,      # [T,S,N,N]
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
    if t_len <= 0 or s_count <= 0 or n_walkers <= 0 or n_bootstrap <= 0:
        return {channel: torch.full((s_count,), float("nan"), device=kernels.device) for channel in channels}

    gen = torch.Generator(device=kernels.device)
    gen.manual_seed(int(seed))
    idx_boot = torch.randint(0, n_walkers, (int(n_bootstrap), n_walkers), generator=gen, device=kernels.device)

    mass_samples = {
        channel: torch.full((int(n_bootstrap), s_count), float("nan"), dtype=torch.float32, device=kernels.device)
        for channel in channels
    }
    eye = torch.eye(n_walkers, dtype=torch.bool, device=kernels.device).view(1, 1, n_walkers, n_walkers)
    for b in range(int(n_bootstrap)):
        idx = idx_boot[b]
        k_boot = kernels[:, :, idx][:, :, :, idx].clone()  # [T,S,N,N]
        k_boot = k_boot.masked_fill(eye, 0.0)
        row_sum = k_boot.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        k_boot = k_boot / row_sum

        c_boot = color[:, idx, :]
        valid_boot = color_valid[:, idx]
        pos_boot = positions[:, idx, :]
        alive_boot = alive[:, idx]
        force_boot = force[:, idx, :]
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
            companions_distance=comp_dist_boot,
            companions_clone=comp_clone_boot,
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
            out[channel] = torch.full((s_count,), float("nan"), dtype=torch.float32, device=kernels.device)
            continue
        vals = torch.where(finite, vals, torch.nan)
        out[channel] = _nanstd_compat(vals, dim=0)
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
    frame_ids_t = torch.as_tensor(frame_indices, dtype=torch.long, device=history.x_before_clone.device)
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
    series_by_channel: dict[str, Tensor] = {
        channel: torch.zeros((n_scales, n_frames), dtype=torch.float32, device=color.device)
        for channel in requested
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
        device=color.device,
        dtype=torch.float32,
    ):
        pos_idx = [frame_to_pos[int(frame_idx)] for frame_idx in frame_ids_chunk]
        pos_t = torch.as_tensor(pos_idx, dtype=torch.long, device=color.device)
        chunk_series = _compute_channel_series_from_kernels(
            color=color.index_select(0, pos_t),
            color_valid=color_valid.index_select(0, pos_t),
            positions=positions.index_select(0, pos_t),
            alive=alive.index_select(0, pos_t),
            force=force.index_select(0, pos_t),
            kernels=kernels_chunk,
            companions_distance=companions_distance.index_select(0, pos_t),
            companions_clone=companions_clone.index_select(0, pos_t),
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
        mass_std_time = torch.full((c_count, s_count), float("nan"), dtype=torch.float32, device=stack.device)

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
            _, _, kernels_all = compute_smeared_kernels_from_history(
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
            )
            n_walk_boot = int(max(1, min(int(config.n_bootstrap), int(config.walker_bootstrap_max_samples))))
            mass_std_walker_by_channel = _walker_bootstrap_mass_std(
                color=color,
                color_valid=color_valid,
                positions=positions,
                alive=alive,
                force=force,
                companions_distance=companions_distance,
                companions_clone=companions_clone,
                kernels=kernels_all,
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
    for c_idx, channel in enumerate(channel_names):
        channel_results: list[ChannelCorrelatorResult] = []
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

        best_idx = _select_best_scale(channel_results)
        best_scale_index[channel] = int(best_idx)
        best_result = channel_results[best_idx]

        # Attach bootstrap mass spread if available.
        base_mass_err = float(best_result.mass_fit.get("mass_error", float("inf")))
        comp_mass_err = float("nan")
        if mass_std_walker_by_channel is not None and channel in mass_std_walker_by_channel:
            std_vec = mass_std_walker_by_channel[channel]
            bootstrap_mass_std_out[channel] = std_vec.detach().clone()
            comp_mass_err = float(std_vec[best_idx].item())
        elif mass_std_time is not None:
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
