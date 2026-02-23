"""Vectorized coupling diagnostics for fast run-level QCD regime checks.

This module computes quick, mass-free diagnostics from companion-pair color
inner products and optional kernel-scale confinement/topology proxies.

Time-series diagnostics:
- Circular phase concentration ``R_circ``
- Re/Im variance asymmetry
- Local phase coherence
- Global phase drift and drift significance

Kernel diagnostics (snapshot + all-time Polyakov):
- String-tension proxy from coherence-vs-scale decay
- Creutz-ratio and running-coupling proxies
- Color screening-length proxy
- Polyakov-loop proxy from temporal color transport
- Topological flux proxy and aggregate regime score
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from fragile.physics.app.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
    compute_smeared_kernels_from_distances,
    select_interesting_scales_from_history,
)
from fragile.physics.app.spectral_gap import compute_spectral_gap
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils import compute_color_states_batch, estimate_ell0_auto


COMPANION_TOPOLOGY_MODES = ("distance", "clone", "both")
PAIR_WEIGHTING_MODES = ("uniform", "score_abs")
KERNEL_TYPES = ("gaussian", "exponential", "tophat", "shell")
KERNEL_DISTANCE_METHODS = ("auto", "floyd-warshall", "tropical")
EDGE_WEIGHT_MODES = (
    "uniform",
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_distance",
    "inverse_riemannian_volume",
    "kernel",
    "riemannian_kernel",
    "riemannian_kernel_volume",
)


@dataclass
class CouplingDiagnosticsConfig:
    """Configuration for vectorized coupling diagnostics."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    ell0_method: str = "companion"
    color_dims: tuple[int, ...] | None = None
    companion_topology: str = "both"
    pair_weighting: str = "uniform"
    eps: float = 1e-12

    enable_kernel_diagnostics: bool = True
    edge_weight_mode: str = "riemannian_kernel_volume"
    n_scales: int = 8
    kernel_type: str = "gaussian"
    kernel_distance_method: str = "auto"
    # Kept for call compatibility; diagnostics now assume all walkers are alive.
    kernel_assume_all_alive: bool = True
    kernel_scale_frames: int = 8
    kernel_scale_q_low: float = 0.05
    kernel_scale_q_high: float = 0.95
    kernel_max_scale_samples: int = 500_000
    kernel_min_scale: float = 1e-6

    enable_wilson_flow: bool = False
    wilson_flow_n_steps: int = 100
    wilson_flow_step_size: float = 0.02
    wilson_flow_topology: str = "both"
    wilson_flow_t0_reference: float = 0.3
    wilson_flow_w0_reference: float = 0.3


@dataclass
class CouplingDiagnosticsOutput:
    """Per-frame and aggregate coupling diagnostics."""

    frame_indices: list[int]
    phase_mean: Tensor
    phase_mean_unwrapped: Tensor
    phase_concentration: Tensor
    re_im_asymmetry: Tensor
    local_phase_coherence: Tensor
    scalar_mean: Tensor
    pseudoscalar_mean: Tensor
    field_magnitude_mean: Tensor
    valid_pair_counts: Tensor
    valid_walker_counts: Tensor

    scales: Tensor
    coherence_by_scale: Tensor
    phase_spread_by_scale: Tensor
    screening_connected_by_scale: Tensor
    creutz_mid_scales: Tensor
    creutz_ratio_by_mid_scale: Tensor
    running_mid_scales: Tensor
    running_g2_by_mid_scale: Tensor

    snapshot_frame_index: int | None
    string_tension_sigma: float
    screening_length_xi: float
    running_coupling_slope: float
    polyakov_abs: float
    polyakov_phase: float
    polyakov_spread: float
    topological_charge_q: float
    topological_flux_std: float
    regime_score: float
    regime_evidence: list[str]

    spectral_gap_fiedler: float
    spectral_gap_fiedler_std: float
    spectral_gap_autocorrelation: float
    spectral_gap_autocorrelation_tau: float
    spectral_gap_transfer_matrix: float
    spectral_gap_transfer_matrix_std: float

    wilson_flow: object | None

    summary: dict[str, float]


def _resolve_companion_topology(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in COMPANION_TOPOLOGY_MODES:
        msg = f"companion_topology must be one of {COMPANION_TOPOLOGY_MODES}."
        raise ValueError(msg)
    return mode_norm


def _resolve_pair_weighting(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in PAIR_WEIGHTING_MODES:
        msg = f"pair_weighting must be one of {PAIR_WEIGHTING_MODES}."
        raise ValueError(msg)
    return mode_norm


def _resolve_kernel_type(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in KERNEL_TYPES:
        msg = f"kernel_type must be one of {KERNEL_TYPES}."
        raise ValueError(msg)
    return mode_norm


def _resolve_kernel_distance_method(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in KERNEL_DISTANCE_METHODS:
        msg = f"kernel_distance_method must be one of {KERNEL_DISTANCE_METHODS}."
        raise ValueError(msg)
    return mode_norm


def _resolve_edge_weight_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in EDGE_WEIGHT_MODES:
        msg = f"edge_weight_mode must be one of {EDGE_WEIGHT_MODES}."
        raise ValueError(msg)
    return mode_norm


def _resolve_frame_indices(
    history: RunHistory,
    warmup_fraction: float,
    end_fraction: float,
) -> list[int]:
    if history.n_recorded < 2:
        return []
    start_idx = max(1, int(history.n_recorded * float(warmup_fraction)))
    end_idx = max(start_idx + 1, int(history.n_recorded * float(end_fraction)))
    end_idx = min(end_idx, history.n_recorded)
    if end_idx <= start_idx:
        return []
    return list(range(start_idx, end_idx))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / denom


def _masked_var(values: Tensor, mask: Tensor) -> Tensor:
    mean = _masked_mean(values, mask)
    centered = values - mean.unsqueeze(1)
    return _masked_mean(centered * centered, mask)


def _masked_std(values: Tensor, mask: Tensor) -> Tensor:
    return torch.sqrt(_masked_var(values, mask).clamp(min=0.0))


def _linear_fit_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def _empty_output() -> CouplingDiagnosticsOutput:
    empty_f = torch.zeros(0, dtype=torch.float32)
    empty_i = torch.zeros(0, dtype=torch.int64)
    return CouplingDiagnosticsOutput(
        frame_indices=[],
        phase_mean=empty_f,
        phase_mean_unwrapped=empty_f.clone(),
        phase_concentration=empty_f.clone(),
        re_im_asymmetry=empty_f.clone(),
        local_phase_coherence=empty_f.clone(),
        scalar_mean=empty_f.clone(),
        pseudoscalar_mean=empty_f.clone(),
        field_magnitude_mean=empty_f.clone(),
        valid_pair_counts=empty_i,
        valid_walker_counts=empty_i.clone(),
        scales=empty_f.clone(),
        coherence_by_scale=empty_f.clone(),
        phase_spread_by_scale=empty_f.clone(),
        screening_connected_by_scale=empty_f.clone(),
        creutz_mid_scales=empty_f.clone(),
        creutz_ratio_by_mid_scale=empty_f.clone(),
        running_mid_scales=empty_f.clone(),
        running_g2_by_mid_scale=empty_f.clone(),
        snapshot_frame_index=None,
        string_tension_sigma=float("nan"),
        screening_length_xi=float("nan"),
        running_coupling_slope=float("nan"),
        polyakov_abs=float("nan"),
        polyakov_phase=float("nan"),
        polyakov_spread=float("nan"),
        topological_charge_q=float("nan"),
        topological_flux_std=float("nan"),
        regime_score=float("nan"),
        regime_evidence=["No valid frames for diagnostics."],
        spectral_gap_fiedler=float("nan"),
        spectral_gap_fiedler_std=float("nan"),
        spectral_gap_autocorrelation=float("nan"),
        spectral_gap_autocorrelation_tau=float("nan"),
        spectral_gap_transfer_matrix=float("nan"),
        spectral_gap_transfer_matrix_std=float("nan"),
        wilson_flow=None,
        summary={
            "n_frames": 0.0,
            "ell0": float("nan"),
            "h_eff": float("nan"),
            "phase_drift": float("nan"),
            "phase_step_std": float("nan"),
            "phase_drift_sigma": float("nan"),
            "r_circ_mean": float("nan"),
            "re_im_asymmetry_mean": float("nan"),
            "local_phase_coherence_mean": float("nan"),
            "scalar_mean": float("nan"),
            "pseudoscalar_mean": float("nan"),
            "field_magnitude_mean": float("nan"),
            "valid_pairs_mean": 0.0,
            "valid_walkers_mean": 0.0,
            "string_tension_sigma": float("nan"),
            "polyakov_abs": float("nan"),
            "screening_length_xi": float("nan"),
            "running_coupling_slope": float("nan"),
            "topological_flux_std": float("nan"),
            "topological_charge_q": float("nan"),
            "regime_score": float("nan"),
            "kernel_diagnostics_available": 0.0,
            "spectral_gap_fiedler": float("nan"),
            "spectral_gap_autocorrelation": float("nan"),
            "spectral_gap_autocorrelation_tau": float("nan"),
            "spectral_gap_transfer_matrix": float("nan"),
            "wilson_flow_t0": float("nan"),
            "wilson_flow_w0": float("nan"),
            "wilson_flow_sqrt_8t0": float("nan"),
        },
    )


def _compute_regime_score(
    *,
    sigma: float,
    polyakov_abs: float,
    xi: float,
    running_slope: float,
    flux_std: float,
) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []

    if np.isfinite(sigma):
        if sigma > 0.3:
            score += 2.0
            evidence.append(f"String tension sigma={sigma:.4f} indicates confining behavior.")
        elif sigma > 0.1:
            score += 1.0
            evidence.append(f"String tension sigma={sigma:.4f} indicates crossover behavior.")
        else:
            evidence.append(
                f"String tension sigma={sigma:.4f} indicates weak/deconfined behavior."
            )
    else:
        evidence.append("String tension unavailable.")

    if np.isfinite(polyakov_abs):
        if polyakov_abs < 0.05:
            score += 2.0
            evidence.append(f"Polyakov |L|={polyakov_abs:.4f} indicates confinement.")
        elif polyakov_abs < 0.2:
            score += 1.0
            evidence.append(f"Polyakov |L|={polyakov_abs:.4f} indicates crossover.")
        else:
            evidence.append(f"Polyakov |L|={polyakov_abs:.4f} indicates deconfined behavior.")
    else:
        evidence.append("Polyakov loop unavailable.")

    if np.isfinite(xi):
        if 0.15 < xi < 0.6:
            score += 2.0
            evidence.append(f"Screening length xi={xi:.4f} is in target confining range.")
        elif xi > 0.05:
            score += 1.0
            evidence.append(f"Screening length xi={xi:.4f} is marginal.")
        else:
            evidence.append(f"Screening length xi={xi:.4f} is too short.")
    else:
        evidence.append("Screening length unavailable.")

    if np.isfinite(running_slope):
        if running_slope > 0.1:
            score += 2.0
            evidence.append(f"Running coupling slope={running_slope:.4f} grows with scale.")
        elif running_slope > 0.0:
            score += 1.0
            evidence.append(f"Running coupling slope={running_slope:.4f} shows weak growth.")
        else:
            evidence.append(f"Running coupling slope={running_slope:.4f} decreases with scale.")
    else:
        evidence.append("Running coupling slope unavailable.")

    if np.isfinite(flux_std):
        if flux_std > 0.5:
            score += 2.0
            evidence.append(f"Topological flux std={flux_std:.4f} indicates strong topology.")
        elif flux_std > 0.2:
            score += 1.0
            evidence.append(f"Topological flux std={flux_std:.4f} indicates mild topology.")
        else:
            evidence.append(f"Topological flux std={flux_std:.4f} indicates weak topology.")
    else:
        evidence.append("Topological flux unavailable.")

    return score, evidence


def _compute_kernel_scale_diagnostics(
    *,
    history: RunHistory,
    frame_indices: list[int],
    color: Tensor,
    color_valid: Tensor,
    cfg: CouplingDiagnosticsConfig,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    int | None,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    list[str],
]:
    """Compute snapshot kernel diagnostics and all-time Polyakov loop."""
    device = color.device
    empty = torch.zeros(0, dtype=torch.float32, device=device)

    if not bool(cfg.enable_kernel_diagnostics):
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            None,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            ["Kernel diagnostics disabled by configuration."],
        )

    if not frame_indices:
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            None,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            ["No frame indices available for kernel diagnostics."],
        )

    snapshot_frame = int(frame_indices[-1])
    assume_all_alive = True
    try:
        scales = select_interesting_scales_from_history(
            history,
            n_scales=int(max(2, cfg.n_scales)),
            method=str(cfg.kernel_distance_method),
            frame_indices=frame_indices,
            n_scale_frames=int(max(1, cfg.kernel_scale_frames)),
            calibration_batch_size=1,
            edge_weight_mode=str(cfg.edge_weight_mode),
            assume_all_alive=assume_all_alive,
            q_low=float(cfg.kernel_scale_q_low),
            q_high=float(cfg.kernel_scale_q_high),
            max_samples=int(max(1, cfg.kernel_max_scale_samples)),
            min_scale=float(max(1e-12, cfg.kernel_min_scale)),
            device=device,
            dtype=torch.float32,
        )
        frame_ids, distance_batch = compute_pairwise_distance_matrices_from_history(
            history,
            method=str(cfg.kernel_distance_method),
            frame_indices=[snapshot_frame],
            batch_size=1,
            edge_weight_mode=str(cfg.edge_weight_mode),
            assume_all_alive=assume_all_alive,
            device=device,
            dtype=torch.float32,
        )
    except Exception as exc:
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            snapshot_frame,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            [f"Kernel diagnostics unavailable: {exc!s}"],
        )

    if not frame_ids or distance_batch.numel() == 0 or scales.numel() == 0:
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            snapshot_frame,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            ["Kernel diagnostics unavailable: no distance/kernels produced."],
        )

    kernels = compute_smeared_kernels_from_distances(
        distance_batch,
        scales,
        kernel_type=str(cfg.kernel_type),
        zero_diagonal=True,
        normalize_rows=True,
        eps=1e-12,
    )

    t_snapshot = color.shape[0] - 1
    c_snap = color[t_snapshot]
    node_valid = color_valid[t_snapshot].to(torch.bool)

    ip = (torch.conj(c_snap) @ c_snap.transpose(0, 1)).to(torch.complex64)
    abs_ip = ip.abs()
    conn = ip / abs_ip.clamp(min=1e-12)
    finite_ip = torch.isfinite(ip.real) & torch.isfinite(ip.imag)

    n_nodes = int(c_snap.shape[0])
    not_self = ~torch.eye(n_nodes, device=device, dtype=torch.bool)
    pair_valid = node_valid[:, None] & node_valid[None, :] & not_self & finite_ip

    kernel_s = kernels[0].to(dtype=torch.float32)  # [S,N,N]
    masked_kernel = torch.where(pair_valid.unsqueeze(0), kernel_s, torch.zeros_like(kernel_s))
    row_sum = masked_kernel.sum(dim=-1, keepdim=True)
    row_nonzero = row_sum.squeeze(-1) > 0
    kernel_row = torch.where(
        row_nonzero.unsqueeze(-1),
        masked_kernel / row_sum.clamp(min=1e-12),
        torch.zeros_like(masked_kernel),
    )

    u_scale = (kernel_row.to(conn.dtype) * conn.unsqueeze(0)).sum(dim=-1)  # [S,N]
    walker_valid = row_nonzero & node_valid.unsqueeze(0)

    coherence_by_scale = _masked_mean(u_scale.abs().float(), walker_valid)
    phase_spread_by_scale = _masked_std(torch.angle(u_scale).float(), walker_valid)

    kernel_global = masked_kernel
    denom_global = kernel_global.sum(dim=(1, 2)).clamp(min=1e-12)
    mean_mag = (kernel_global * abs_ip.unsqueeze(0)).sum(dim=(1, 2)) / denom_global
    mean_mag2 = (kernel_global * abs_ip.square().unsqueeze(0)).sum(dim=(1, 2)) / denom_global
    screening_connected = mean_mag2 - mean_mag.square()

    scales_np = scales.detach().cpu().numpy().astype(float, copy=False)
    coherence_np = coherence_by_scale.detach().cpu().numpy().astype(float, copy=False)
    screening_np = screening_connected.detach().cpu().numpy().astype(float, copy=False)

    valid_coh = np.isfinite(scales_np) & np.isfinite(coherence_np) & (coherence_np > 1e-3)
    string_sigma = float("nan")
    if np.count_nonzero(valid_coh) >= 3:
        log_coh = np.log(np.clip(coherence_np[valid_coh], 1e-30, None))
        slope, _ = _linear_fit_slope_intercept(scales_np[valid_coh], log_coh)
        if np.isfinite(slope):
            string_sigma = float(-slope)

    valid_screen = np.isfinite(scales_np) & np.isfinite(screening_np) & (screening_np > 1e-8)
    screening_xi = float("nan")
    if np.count_nonzero(valid_screen) >= 3:
        log_screen = np.log(np.clip(screening_np[valid_screen], 1e-30, None))
        slope, _ = _linear_fit_slope_intercept(scales_np[valid_screen], log_screen)
        if np.isfinite(slope) and abs(slope) > 1e-12:
            screening_xi = float(-1.0 / slope)

    # Creutz proxy on adjacent scales.
    if scales.numel() >= 2:
        ds = scales[1:] - scales[:-1]
        ratio = coherence_by_scale[1:] / coherence_by_scale[:-1].clamp(min=1e-12)
        creutz = -torch.log(ratio.clamp(min=1e-12)) / ds.clamp(min=1e-12)
        creutz_mid = 0.5 * (scales[1:] + scales[:-1])
        invalid_creutz = (
            ~torch.isfinite(creutz)
            | ~torch.isfinite(creutz_mid)
            | ~(coherence_by_scale[1:] > 1e-3)
            | ~(coherence_by_scale[:-1] > 1e-3)
        )
        creutz = creutz.masked_fill(invalid_creutz, float("nan"))
    else:
        creutz = empty
        creutz_mid = empty

    # Running coupling proxy with central differences.
    if scales.numel() >= 3:
        log_coh_t = torch.log(coherence_by_scale.clamp(min=1e-12))
        dlog = log_coh_t[2:] - log_coh_t[:-2]
        dscale = scales[2:] - scales[:-2]
        running_mid = scales[1:-1]
        running_g2 = -running_mid * dlog / dscale.clamp(min=1e-12)
        invalid_running = (
            ~torch.isfinite(running_g2)
            | ~torch.isfinite(running_mid)
            | ~(coherence_by_scale[2:] > 1e-3)
            | ~(coherence_by_scale[:-2] > 1e-3)
        )
        running_g2 = running_g2.masked_fill(invalid_running, float("nan"))
    else:
        running_mid = empty
        running_g2 = empty

    running_slope = float("nan")
    if running_mid.numel() >= 3:
        run_x = running_mid.detach().cpu().numpy().astype(float, copy=False)
        run_y = running_g2.detach().cpu().numpy().astype(float, copy=False)
        valid_run = np.isfinite(run_x) & np.isfinite(run_y)
        if np.count_nonzero(valid_run) >= 3:
            running_slope, _ = _linear_fit_slope_intercept(run_x[valid_run], run_y[valid_run])

    # Polyakov-loop proxy across the selected window.
    if color.shape[0] >= 2:
        overlap = (torch.conj(color[:-1]) * color[1:]).sum(dim=-1)  # [T-1,N]
        valid_step = (
            color_valid[:-1]
            & color_valid[1:]
            & torch.isfinite(overlap.real)
            & torch.isfinite(overlap.imag)
        )
        phase_step = torch.where(valid_step, torch.angle(overlap), torch.zeros_like(overlap.real))
        accumulated_phase = phase_step.sum(dim=0)
        walker_valid_poly = valid_step.any(dim=0)
        poly_per_walker = torch.where(
            walker_valid_poly,
            torch.exp(1j * accumulated_phase).to(dtype=torch.complex64),
            torch.zeros_like(accumulated_phase, dtype=torch.complex64),
        )
        n_poly = int(walker_valid_poly.sum().item())
        if n_poly > 0:
            poly_loop = poly_per_walker[walker_valid_poly].mean()
            poly_abs = float(poly_loop.abs().item())
            poly_phase = float(torch.angle(poly_loop).item())
            poly_spread = float(
                poly_per_walker[walker_valid_poly].abs().float().std(unbiased=False).item()
            )
        else:
            poly_abs = float("nan")
            poly_phase = float("nan")
            poly_spread = float("nan")
    else:
        poly_abs = float("nan")
        poly_phase = float("nan")
        poly_spread = float("nan")

    # Topological proxy at median scale.
    top_q = float("nan")
    top_flux_std = float("nan")
    if scales.numel() > 0:
        mid_idx = int(scales.numel() // 2)
        w_mid = kernel_row[mid_idx]
        u_i = (w_mid.to(conn.dtype) * conn).sum(dim=1)
        u_jhop = (w_mid.to(conn.dtype) * conn * u_i.unsqueeze(0)).sum(dim=1)
        holonomy = u_i * u_jhop * torch.conj(u_i)
        flux = torch.angle(holonomy)
        flux_valid = node_valid & row_nonzero[mid_idx] & torch.isfinite(flux)
        if bool(flux_valid.any()):
            flux_sel = flux[flux_valid]
            top_q = float((flux_sel.sum() / (2.0 * math.pi)).item())
            top_flux_std = float(flux_sel.float().std(unbiased=False).item())

    regime_score, regime_evidence = _compute_regime_score(
        sigma=string_sigma,
        polyakov_abs=poly_abs,
        xi=screening_xi,
        running_slope=running_slope,
        flux_std=top_flux_std,
    )

    return (
        scales.float(),
        coherence_by_scale.float(),
        phase_spread_by_scale.float(),
        screening_connected.float(),
        creutz_mid.float(),
        creutz.float(),
        running_mid.float(),
        running_g2.float(),
        snapshot_frame,
        string_sigma,
        screening_xi,
        running_slope,
        poly_abs,
        poly_phase,
        poly_spread,
        top_q,
        top_flux_std,
        regime_score,
        regime_evidence,
    )


def compute_coupling_diagnostics(
    history: RunHistory,
    config: CouplingDiagnosticsConfig | None = None,
) -> CouplingDiagnosticsOutput:
    """Compute fast coupling diagnostics without channel-mass extraction."""
    cfg = config or CouplingDiagnosticsConfig()
    _resolve_companion_topology(cfg.companion_topology)
    _resolve_pair_weighting(cfg.pair_weighting)
    _resolve_kernel_type(cfg.kernel_type)
    _resolve_kernel_distance_method(cfg.kernel_distance_method)
    _resolve_edge_weight_mode(cfg.edge_weight_mode)

    frame_indices = _resolve_frame_indices(
        history=history,
        warmup_fraction=float(cfg.warmup_fraction),
        end_fraction=float(cfg.end_fraction),
    )
    if not frame_indices:
        return _empty_output()

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    ell0 = float(cfg.ell0) if cfg.ell0 is not None else float(estimate_ell0_auto(history, cfg.ell0_method))

    color, color_valid = compute_color_states_batch(
        history=history,
        start_idx=start_idx,
        end_idx=end_idx,
        h_eff=float(max(cfg.h_eff, 1e-8)),
        mass=float(max(cfg.mass, 1e-8)),
        ell0=ell0,
    )
    comp_d = history.companions_distance[start_idx - 1 : end_idx - 1].to(
        dtype=torch.long, device=color.device
    )
    comp_c = history.companions_clone[start_idx - 1 : end_idx - 1].to(
        dtype=torch.long, device=color.device
    )
    scores = history.cloning_scores[start_idx - 1 : end_idx - 1].to(
        dtype=torch.float32, device=color.device
    )

    if cfg.color_dims is not None:
        dims = tuple(int(d) for d in cfg.color_dims)
        if not dims:
            msg = "color_dims must contain at least one dimension."
            raise ValueError(msg)
        d_total = int(color.shape[-1])
        invalid = [d for d in dims if d < 0 or d >= d_total]
        if invalid:
            raise ValueError(
                f"color_dims contains invalid dims {invalid}; valid range [0, {d_total - 1}]."
            )
        color = color[..., list(dims)]
        proj_norm = torch.linalg.vector_norm(color, dim=-1, keepdim=True).clamp(min=1e-12)
        color = color / proj_norm
        color_valid = color_valid & (proj_norm.squeeze(-1) > 1e-12)

    if color.ndim != 3:
        raise ValueError(f"Expected color [T,N,C], got {tuple(color.shape)}.")
    t_len, n_walkers, _ = color.shape
    if t_len == 0 or n_walkers == 0:
        return _empty_output()

    topology = str(cfg.companion_topology).strip().lower()
    if topology == "distance":
        pair_indices = comp_d.unsqueeze(-1)
    elif topology == "clone":
        pair_indices = comp_c.unsqueeze(-1)
    else:
        pair_indices = torch.stack([comp_d, comp_c], dim=-1)

    src_idx = torch.arange(n_walkers, device=color.device, dtype=torch.long).view(1, n_walkers, 1)
    structural_valid = pair_indices != src_idx

    # Pair indices are used directly under the "all walkers in bounds" assumption.
    color_j = torch.gather(
        color.unsqueeze(2).expand(-1, -1, pair_indices.shape[-1], -1),
        1,
        pair_indices.unsqueeze(-1).expand(-1, -1, -1, color.shape[-1]),
    )

    color_i = color.unsqueeze(2).expand_as(color_j)
    inner = (torch.conj(color_i) * color_j).sum(dim=-1)
    finite_inner = torch.isfinite(inner.real) & torch.isfinite(inner.imag)

    valid = structural_valid & color_valid.unsqueeze(-1) & finite_inner
    eps = float(max(cfg.eps, 0.0))
    if eps > 0:
        valid = valid & (inner.abs() > eps)

    weighting = str(cfg.pair_weighting).strip().lower()
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

    s_field = (row_weights.to(inner.dtype) * inner).sum(dim=-1)
    walker_valid = color_valid & row_has_pairs

    phase = torch.angle(s_field)
    phase_vector = torch.where(
        walker_valid,
        torch.exp(1j * phase).to(dtype=s_field.dtype),
        torch.zeros_like(s_field),
    )
    walker_count = walker_valid.sum(dim=1).clamp(min=1)
    phase_mean_complex = phase_vector.sum(dim=1) / walker_count.to(s_field.dtype)
    phase_mean = torch.angle(phase_mean_complex).float()
    phase_concentration = torch.abs(phase_mean_complex).float()

    re_part = s_field.real.float()
    im_part = s_field.imag.float()
    var_re = _masked_var(re_part, walker_valid)
    var_im = _masked_var(im_part, walker_valid)
    re_im_asymmetry = (var_re - var_im).abs() / (var_re + var_im + 1e-12)

    unit_inner = inner / inner.abs().clamp(min=max(eps, 1e-12))
    local_phase = (row_weights.to(inner.dtype) * unit_inner).sum(dim=-1)
    local_phase_coherence = _masked_mean(local_phase.abs().float(), walker_valid)

    scalar_mean = _masked_mean(re_part, walker_valid)
    pseudoscalar_mean = _masked_mean(im_part, walker_valid)
    field_magnitude_mean = _masked_mean(s_field.abs().float(), walker_valid)

    valid_pair_counts = valid.sum(dim=(1, 2)).to(torch.int64)
    valid_walker_counts = walker_valid.sum(dim=1).to(torch.int64)

    phase_mean_np = phase_mean.detach().cpu().numpy()
    if phase_mean_np.size > 0:
        phase_unwrapped_np = np.unwrap(phase_mean_np)
        phase_mean_unwrapped = torch.as_tensor(
            phase_unwrapped_np, dtype=torch.float32, device=phase_mean.device
        )
    else:
        phase_mean_unwrapped = torch.zeros_like(phase_mean)

    if phase_mean_unwrapped.numel() > 1:
        phase_diffs = phase_mean_unwrapped[1:] - phase_mean_unwrapped[:-1]
        phase_step_std = float(phase_diffs.std(unbiased=False).item())
        phase_drift = float((phase_mean_unwrapped[-1] - phase_mean_unwrapped[0]).item())
        if phase_step_std > 0:
            phase_drift_sigma = float(
                abs(phase_drift) / (phase_step_std * math.sqrt(float(phase_diffs.numel())))
            )
        else:
            phase_drift_sigma = float("inf") if abs(phase_drift) > 0 else 0.0
    else:
        phase_step_std = float("nan")
        phase_drift = float("nan")
        phase_drift_sigma = float("nan")

    (
        scales,
        coherence_by_scale,
        phase_spread_by_scale,
        screening_connected_by_scale,
        creutz_mid_scales,
        creutz_ratio_by_mid_scale,
        running_mid_scales,
        running_g2_by_mid_scale,
        snapshot_frame_index,
        string_tension_sigma,
        screening_length_xi,
        running_coupling_slope,
        polyakov_abs,
        polyakov_phase,
        polyakov_spread,
        topological_charge_q,
        topological_flux_std,
        regime_score,
        regime_evidence,
    ) = _compute_kernel_scale_diagnostics(
        history=history,
        frame_indices=frame_indices,
        color=color,
        color_valid=color_valid,
        cfg=cfg,
    )

    kernel_available = float(bool(scales.numel() > 0))

    # Wilson flow
    wilson_flow_output = None
    if cfg.enable_wilson_flow:
        from fragile.physics.new_channels.wilson_flow import compute_wilson_flow, WilsonFlowConfig

        wf_cfg = WilsonFlowConfig(
            n_steps=int(cfg.wilson_flow_n_steps),
            step_size=float(cfg.wilson_flow_step_size),
            topology=str(cfg.wilson_flow_topology),
            t0_reference=float(cfg.wilson_flow_t0_reference),
            w0_reference=float(cfg.wilson_flow_w0_reference),
            eps=float(cfg.eps),
        )
        wilson_flow_output = compute_wilson_flow(
            color=color,
            color_valid=color_valid,
            companions_distance=comp_d,
            companions_clone=comp_c,
            config=wf_cfg,
        )

    # Spectral gap estimation
    sg = compute_spectral_gap(history, warmup_fraction=float(cfg.warmup_fraction))

    summary = {
        "n_frames": float(t_len),
        "ell0": ell0,
        "h_eff": float(cfg.h_eff),
        "phase_drift": phase_drift,
        "phase_step_std": phase_step_std,
        "phase_drift_sigma": phase_drift_sigma,
        "r_circ_mean": float(phase_concentration.mean().item()) if t_len > 0 else float("nan"),
        "re_im_asymmetry_mean": float(re_im_asymmetry.mean().item())
        if t_len > 0
        else float("nan"),
        "local_phase_coherence_mean": (
            float(local_phase_coherence.mean().item()) if t_len > 0 else float("nan")
        ),
        "scalar_mean": float(scalar_mean.mean().item()) if t_len > 0 else float("nan"),
        "pseudoscalar_mean": float(pseudoscalar_mean.mean().item()) if t_len > 0 else float("nan"),
        "field_magnitude_mean": float(field_magnitude_mean.mean().item())
        if t_len > 0
        else float("nan"),
        "valid_pairs_mean": float(valid_pair_counts.float().mean().item()) if t_len > 0 else 0.0,
        "valid_walkers_mean": float(valid_walker_counts.float().mean().item())
        if t_len > 0
        else 0.0,
        "string_tension_sigma": string_tension_sigma,
        "polyakov_abs": polyakov_abs,
        "screening_length_xi": screening_length_xi,
        "running_coupling_slope": running_coupling_slope,
        "topological_flux_std": topological_flux_std,
        "topological_charge_q": topological_charge_q,
        "regime_score": regime_score,
        "kernel_diagnostics_available": kernel_available,
        "spectral_gap_fiedler": sg.fiedler_value,
        "spectral_gap_autocorrelation": sg.autocorrelation_gap,
        "spectral_gap_autocorrelation_tau": sg.autocorrelation_tau,
        "spectral_gap_transfer_matrix": sg.transfer_matrix_gap,
        "wilson_flow_t0": wilson_flow_output.t0
        if wilson_flow_output is not None
        else float("nan"),
        "wilson_flow_w0": wilson_flow_output.w0
        if wilson_flow_output is not None
        else float("nan"),
        "wilson_flow_sqrt_8t0": (
            wilson_flow_output.sqrt_8t0 if wilson_flow_output is not None else float("nan")
        ),
    }

    return CouplingDiagnosticsOutput(
        frame_indices=frame_indices,
        phase_mean=phase_mean,
        phase_mean_unwrapped=phase_mean_unwrapped,
        phase_concentration=phase_concentration,
        re_im_asymmetry=re_im_asymmetry,
        local_phase_coherence=local_phase_coherence,
        scalar_mean=scalar_mean,
        pseudoscalar_mean=pseudoscalar_mean,
        field_magnitude_mean=field_magnitude_mean,
        valid_pair_counts=valid_pair_counts,
        valid_walker_counts=valid_walker_counts,
        scales=scales,
        coherence_by_scale=coherence_by_scale,
        phase_spread_by_scale=phase_spread_by_scale,
        screening_connected_by_scale=screening_connected_by_scale,
        creutz_mid_scales=creutz_mid_scales,
        creutz_ratio_by_mid_scale=creutz_ratio_by_mid_scale,
        running_mid_scales=running_mid_scales,
        running_g2_by_mid_scale=running_g2_by_mid_scale,
        snapshot_frame_index=snapshot_frame_index,
        string_tension_sigma=string_tension_sigma,
        screening_length_xi=screening_length_xi,
        running_coupling_slope=running_coupling_slope,
        polyakov_abs=polyakov_abs,
        polyakov_phase=polyakov_phase,
        polyakov_spread=polyakov_spread,
        topological_charge_q=topological_charge_q,
        topological_flux_std=topological_flux_std,
        regime_score=regime_score,
        regime_evidence=regime_evidence,
        spectral_gap_fiedler=sg.fiedler_value,
        spectral_gap_fiedler_std=sg.fiedler_std,
        spectral_gap_autocorrelation=sg.autocorrelation_gap,
        spectral_gap_autocorrelation_tau=sg.autocorrelation_tau,
        spectral_gap_transfer_matrix=sg.transfer_matrix_gap,
        spectral_gap_transfer_matrix_std=sg.transfer_matrix_gap_std,
        wilson_flow=wilson_flow_output,
        summary=summary,
    )
