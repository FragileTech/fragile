"""Fitness-weighted companion-pair bilinear correlators.

Replaces the frame-averaged fitness pseudoscalar approach with per-pair
operators that preserve spatial coherence through the companion graph.

Three channels, each using companion-pair color inner products z_ij = c_i^dag c_j
weighted by fitness-derived quantities:

- fitness_pseudoscalar: Im(z_ij) * sign(log(f_j / f_i))
  Pseudoscalar parity from relative fitness ordering.

- fitness_scalar_variance: Re(z_ij) * (delta_i - delta_j)^2
  Scalar channel weighted by log-fitness gap squared,
  where delta_k = log(f_k) - <log(f)>_frame.

- fitness_axial: Im(z_ij) * (score_j - score_i)
  Pseudoscalar weighted by cloning-score gradient across pairs.

Source-pair propagation:
    C_X(dt) = <O_X(t, i, j) * O_X(t+dt, i, j)>
where j is the companion from the source frame, tracked across time lags.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.new_channels.meson_phase_channels import (
    _compute_inner_products_for_pairs,
    _per_frame_series,
    _safe_gather_pairs_2d,
    build_companion_pair_indices,
    PAIR_SELECTION_MODES,
)
from fragile.physics.qft_utils import resolve_3d_dims, resolve_frame_indices
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0


@dataclass
class FitnessBilinearConfig:
    """Configuration for fitness-weighted companion-pair correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    pair_selection: str = "both"
    eps: float = 1e-12
    fitness_floor: float = 1e-30


@dataclass
class FitnessBilinearOutput:
    """Output for fitness-weighted companion-pair correlators."""

    # Fitness pseudoscalar: Im(z_ij) * sign(log(f_j/f_i))
    fitness_pseudoscalar: Tensor
    fitness_pseudoscalar_raw: Tensor
    fitness_pseudoscalar_connected: Tensor

    # Fitness scalar variance: Re(z_ij) * (delta_i - delta_j)^2
    fitness_scalar_variance: Tensor
    fitness_scalar_variance_raw: Tensor
    fitness_scalar_variance_connected: Tensor

    # Fitness axial: Im(z_ij) * (score_j - score_i)
    fitness_axial: Tensor
    fitness_axial_raw: Tensor
    fitness_axial_connected: Tensor

    # Pair counts per lag
    counts: Tensor  # [max_lag+1]

    # Metadata
    frame_indices: list[int]
    pair_counts_per_frame: Tensor
    pair_selection: str

    # Statistics
    mean_fitness_pseudoscalar: float
    mean_fitness_scalar_variance: float
    mean_fitness_axial: float
    disconnected_fitness_pseudoscalar: float
    disconnected_fitness_scalar_variance: float
    disconnected_fitness_axial: float
    n_valid_source_pairs: int

    # Operator time series [T]
    operator_fitness_pseudoscalar_series: Tensor
    operator_fitness_scalar_variance_series: Tensor
    operator_fitness_axial_series: Tensor


def _compute_fitness_weights(
    fitness: Tensor,
    cloning_scores: Tensor,
    alive_mask: Tensor,
    pair_indices: Tensor,
    fitness_floor: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute per-pair fitness weights for all three channels.

    Args:
        fitness: Fitness values [T, N].
        cloning_scores: Cloning scores [T, N].
        alive_mask: Boolean alive mask [T, N].
        pair_indices: Companion pair indices [T, N, P].
        fitness_floor: Floor for fitness before log.

    Returns:
        (w_pseudoscalar, w_scalar_var, w_axial, weight_valid)
        Each [T, N, P]. weight_valid masks valid fitness pairs.
    """
    # Log-fitness per walker: [T, N]
    log_f = torch.log(fitness.float().clamp(min=fitness_floor))

    # Per-frame alive-weighted mean log-fitness: [T]
    alive_f = alive_mask.float()
    n_alive = alive_f.sum(dim=1).clamp(min=1.0)
    mean_log_f = (log_f * alive_f).sum(dim=1) / n_alive  # [T]

    # delta_k = log(f_k) - <log(f)>_frame: [T, N]
    delta = log_f - mean_log_f.unsqueeze(1)

    # Gather fitness-derived quantities for companion j
    # log_f_j: [T, N, P]
    log_f_j, in_range_f = _safe_gather_pairs_2d(log_f, pair_indices)
    delta_j, _ = _safe_gather_pairs_2d(delta, pair_indices)
    score_j, in_range_s = _safe_gather_pairs_2d(cloning_scores.float(), pair_indices)
    alive_j, _ = _safe_gather_pairs_2d(alive_mask.float(), pair_indices)

    # Expand i-quantities to pair shape: [T, N, P]
    log_f_i = log_f.unsqueeze(-1).expand_as(log_f_j)
    delta_i = delta.unsqueeze(-1).expand_as(delta_j)
    score_i = cloning_scores.float().unsqueeze(-1).expand_as(score_j)
    alive_i = alive_mask.float().unsqueeze(-1).expand_as(alive_j)

    # Validity: both walkers alive and in-range
    weight_valid = in_range_f & in_range_s & (alive_i > 0.5) & (alive_j > 0.5)

    # Channel 1: sign(log(f_j / f_i)) = sign(log_f_j - log_f_i)
    log_ratio = log_f_j - log_f_i
    w_pseudoscalar = torch.sign(log_ratio)

    # Channel 2: (delta_i - delta_j)^2
    w_scalar_var = (delta_i - delta_j).square()

    # Channel 3: (score_j - score_i)
    w_axial = score_j - score_i

    # Zero out invalid weights
    w_pseudoscalar = torch.where(weight_valid, w_pseudoscalar, torch.zeros_like(w_pseudoscalar))
    w_scalar_var = torch.where(weight_valid, w_scalar_var, torch.zeros_like(w_scalar_var))
    w_axial = torch.where(weight_valid, w_axial, torch.zeros_like(w_axial))

    return w_pseudoscalar, w_scalar_var, w_axial, weight_valid


def compute_fitness_bilinear_from_color(
    color: Tensor,
    color_valid: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    fitness: Tensor,
    cloning_scores: Tensor,
    alive_mask: Tensor,
    *,
    max_lag: int = 80,
    use_connected: bool = True,
    pair_selection: str = "both",
    eps: float = 1e-12,
    fitness_floor: float = 1e-30,
    frame_indices: list[int] | None = None,
) -> FitnessBilinearOutput:
    """Compute fitness-weighted companion-pair correlators from color states.

    Source-pair propagation:
        C_X(dt) = < O_X(t, i, j_t(i)) * O_X(t+dt, i, j_t(i)) >
    where j_t(i) is the companion from the source frame.

    Args:
        color: Complex color states [T, N, 3].
        color_valid: Valid color mask [T, N].
        companions_distance: Distance companion indices [T, N].
        companions_clone: Clone companion indices [T, N].
        fitness: Fitness values [T, N].
        cloning_scores: Cloning scores [T, N].
        alive_mask: Boolean alive mask [T, N].
        max_lag: Maximum lag to compute.
        use_connected: If True, use connected correlators.
        pair_selection: One of {"distance", "clone", "both"}.
        eps: Inner product validity threshold.
        fitness_floor: Floor for fitness before log.
        frame_indices: Frame indices for metadata.

    Returns:
        FitnessBilinearOutput with all correlators and diagnostics.
    """
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color must have shape [T, N, 3], got {tuple(color.shape)}.")

    mode = str(pair_selection).strip().lower()
    if mode not in PAIR_SELECTION_MODES:
        raise ValueError(f"pair_selection must be one of {PAIR_SELECTION_MODES}.")

    t_total = int(color.shape[0])
    max_lag = max(0, int(max_lag))
    effective_lag = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1
    device = color.device

    if t_total == 0:
        empty_f = torch.zeros(n_lags, dtype=torch.float32, device=device)
        empty_i = torch.zeros(n_lags, dtype=torch.int64, device=device)
        empty_t = torch.zeros(0, dtype=torch.int64, device=device)
        return FitnessBilinearOutput(
            fitness_pseudoscalar=empty_f,
            fitness_pseudoscalar_raw=empty_f.clone(),
            fitness_pseudoscalar_connected=empty_f.clone(),
            fitness_scalar_variance=empty_f.clone(),
            fitness_scalar_variance_raw=empty_f.clone(),
            fitness_scalar_variance_connected=empty_f.clone(),
            fitness_axial=empty_f.clone(),
            fitness_axial_raw=empty_f.clone(),
            fitness_axial_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[] if frame_indices is None else frame_indices,
            pair_counts_per_frame=empty_t,
            pair_selection=mode,
            mean_fitness_pseudoscalar=0.0,
            mean_fitness_scalar_variance=0.0,
            mean_fitness_axial=0.0,
            disconnected_fitness_pseudoscalar=0.0,
            disconnected_fitness_scalar_variance=0.0,
            disconnected_fitness_axial=0.0,
            n_valid_source_pairs=0,
            operator_fitness_pseudoscalar_series=empty_t.float(),
            operator_fitness_scalar_variance_series=empty_t.float(),
            operator_fitness_axial_series=empty_t.float(),
        )

    # Build pair indices from companions
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        pair_selection=mode,
    )

    # Compute color inner products z_ij = c_i^dag c_j
    source_inner, source_valid = _compute_inner_products_for_pairs(
        color=color,
        color_valid=color_valid,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=eps,
    )

    # Compute fitness weights for all three channels
    w_ps, w_sv, w_ax, weight_valid = _compute_fitness_weights(
        fitness=fitness,
        cloning_scores=cloning_scores,
        alive_mask=alive_mask,
        pair_indices=pair_indices,
        fitness_floor=fitness_floor,
    )

    # Combined validity: color pairs valid AND fitness weights valid
    combined_valid = source_valid & weight_valid

    # Source operators: [T, N, P]
    source_ps = source_inner.imag.float() * w_ps  # Im(z) * sign(log(f_j/f_i))
    source_sv = source_inner.real.float() * w_sv  # Re(z) * (delta_i - delta_j)^2
    source_ax = source_inner.imag.float() * w_ax  # Im(z) * (score_j - score_i)

    # Per-frame operator series
    op_ps_series, pair_counts_per_frame = _per_frame_series(source_ps, combined_valid)
    op_sv_series, _ = _per_frame_series(source_sv, combined_valid)
    op_ax_series, _ = _per_frame_series(source_ax, combined_valid)

    # Global means (for connected correlators and disconnected parts)
    n_valid_source_pairs = int(combined_valid.sum().item())
    if n_valid_source_pairs > 0:
        mean_ps = source_ps[combined_valid].mean()
        mean_sv = source_sv[combined_valid].mean()
        mean_ax = source_ax[combined_valid].mean()
    else:
        mean_ps = torch.zeros((), dtype=torch.float32, device=device)
        mean_sv = torch.zeros((), dtype=torch.float32, device=device)
        mean_ax = torch.zeros((), dtype=torch.float32, device=device)

    disconnected_ps = float((mean_ps * mean_ps).item())
    disconnected_sv = float((mean_sv * mean_sv).item())
    disconnected_ax = float((mean_ax * mean_ax).item())

    # Allocate correlator arrays
    ps_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    ps_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    sv_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    sv_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    ax_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    ax_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.int64, device=device)

    # Source-pair propagation loop
    for lag in range(effective_lag + 1):
        source_len = t_total - lag

        # Recompute sink inner products using source-frame pair indices
        sink_inner, sink_valid = _compute_inner_products_for_pairs(
            color=color[lag : lag + source_len],
            color_valid=color_valid[lag : lag + source_len],
            pair_indices=pair_indices[:source_len],
            structural_valid=structural_valid[:source_len],
            eps=eps,
        )

        # Recompute sink fitness weights at the time-shifted frame
        sink_w_ps, sink_w_sv, sink_w_ax, sink_weight_valid = _compute_fitness_weights(
            fitness=fitness[lag : lag + source_len],
            cloning_scores=cloning_scores[lag : lag + source_len],
            alive_mask=alive_mask[lag : lag + source_len],
            pair_indices=pair_indices[:source_len],
            fitness_floor=fitness_floor,
        )

        # Sink operators
        sink_ps = sink_inner.imag.float() * sink_w_ps
        sink_sv = sink_inner.real.float() * sink_w_sv
        sink_ax = sink_inner.imag.float() * sink_w_ax

        # Valid pairs: both source and sink valid
        valid_pair = combined_valid[:source_len] & sink_valid & sink_weight_valid
        count = int(valid_pair.sum().item())
        counts[lag] = count
        if count == 0:
            continue

        # Source slices
        src_ps_l = source_ps[:source_len]
        src_sv_l = source_sv[:source_len]
        src_ax_l = source_ax[:source_len]

        # Raw correlators: <O_src * O_sink>
        ps_raw[lag] = (src_ps_l * sink_ps)[valid_pair].mean().float()
        sv_raw[lag] = (src_sv_l * sink_sv)[valid_pair].mean().float()
        ax_raw[lag] = (src_ax_l * sink_ax)[valid_pair].mean().float()

        # Connected correlators: <(O_src - <O>) * (O_sink - <O>)>
        ps_connected[lag] = ((src_ps_l - mean_ps) * (sink_ps - mean_ps))[valid_pair].mean().float()
        sv_connected[lag] = ((src_sv_l - mean_sv) * (sink_sv - mean_sv))[valid_pair].mean().float()
        ax_connected[lag] = ((src_ax_l - mean_ax) * (sink_ax - mean_ax))[valid_pair].mean().float()

    ps_final = ps_connected if use_connected else ps_raw
    sv_final = sv_connected if use_connected else sv_raw
    ax_final = ax_connected if use_connected else ax_raw

    return FitnessBilinearOutput(
        fitness_pseudoscalar=ps_final,
        fitness_pseudoscalar_raw=ps_raw,
        fitness_pseudoscalar_connected=ps_connected,
        fitness_scalar_variance=sv_final,
        fitness_scalar_variance_raw=sv_raw,
        fitness_scalar_variance_connected=sv_connected,
        fitness_axial=ax_final,
        fitness_axial_raw=ax_raw,
        fitness_axial_connected=ax_connected,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
        pair_counts_per_frame=pair_counts_per_frame,
        pair_selection=mode,
        mean_fitness_pseudoscalar=float(mean_ps.item()),
        mean_fitness_scalar_variance=float(mean_sv.item()),
        mean_fitness_axial=float(mean_ax.item()),
        disconnected_fitness_pseudoscalar=disconnected_ps,
        disconnected_fitness_scalar_variance=disconnected_sv,
        disconnected_fitness_axial=disconnected_ax,
        n_valid_source_pairs=n_valid_source_pairs,
        operator_fitness_pseudoscalar_series=op_ps_series,
        operator_fitness_scalar_variance_series=op_sv_series,
        operator_fitness_axial_series=op_ax_series,
    )


def compute_fitness_bilinear_correlator(
    history: RunHistory,
    config: FitnessBilinearConfig | None = None,
) -> FitnessBilinearOutput:
    """Compute fitness-weighted companion-pair correlators from a RunHistory.

    Args:
        history: Run history with color states, fitness, cloning_scores, alive_mask.
        config: Configuration; None uses defaults.

    Returns:
        FitnessBilinearOutput with all correlators and diagnostics.
    """
    config = config or FitnessBilinearConfig()

    frame_indices = resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
    )
    n_lags = int(max(0, config.max_lag)) + 1

    if not frame_indices:
        device = history.x_final.device if hasattr(history, "x_final") else torch.device("cpu")
        empty_f = torch.zeros(n_lags, dtype=torch.float32, device=device)
        empty_i = torch.zeros(n_lags, dtype=torch.int64, device=device)
        empty_t = torch.zeros(0, dtype=torch.int64, device=device)
        return FitnessBilinearOutput(
            fitness_pseudoscalar=empty_f,
            fitness_pseudoscalar_raw=empty_f.clone(),
            fitness_pseudoscalar_connected=empty_f.clone(),
            fitness_scalar_variance=empty_f.clone(),
            fitness_scalar_variance_raw=empty_f.clone(),
            fitness_scalar_variance_connected=empty_f.clone(),
            fitness_axial=empty_f.clone(),
            fitness_axial_raw=empty_f.clone(),
            fitness_axial_connected=empty_f.clone(),
            counts=empty_i,
            frame_indices=[],
            pair_counts_per_frame=empty_t,
            pair_selection=str(config.pair_selection).strip().lower(),
            mean_fitness_pseudoscalar=0.0,
            mean_fitness_scalar_variance=0.0,
            mean_fitness_axial=0.0,
            disconnected_fitness_pseudoscalar=0.0,
            disconnected_fitness_scalar_variance=0.0,
            disconnected_fitness_axial=0.0,
            n_valid_source_pairs=0,
            operator_fitness_pseudoscalar_series=empty_t.float(),
            operator_fitness_scalar_variance_series=empty_t.float(),
            operator_fitness_axial_series=empty_t.float(),
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
    fitness = torch.as_tensor(
        history.fitness[start_idx - 1 : end_idx - 1],
        dtype=torch.float32,
        device=device,
    )
    cloning_scores = torch.as_tensor(
        history.cloning_scores[start_idx - 1 : end_idx - 1],
        dtype=torch.float32,
        device=device,
    )
    alive_mask = torch.as_tensor(
        history.alive_mask[start_idx - 1 : end_idx - 1],
        dtype=torch.bool,
        device=device,
    )

    return compute_fitness_bilinear_from_color(
        color=color,
        color_valid=color_valid.to(dtype=torch.bool, device=device),
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        fitness=fitness,
        cloning_scores=cloning_scores,
        alive_mask=alive_mask,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        pair_selection=str(config.pair_selection),
        eps=float(max(config.eps, 0.0)),
        fitness_floor=float(config.fitness_floor),
        frame_indices=frame_indices,
    )
