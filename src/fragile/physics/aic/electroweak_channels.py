"""Electroweak (U1/SU2) channel correlators for Fractal Gas runs.

This module computes electroweak operator series from simulation-recorded
neighbor/companion data and extracts correlator masses through the shared
correlator pipeline used by other QFT channel modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from fragile.physics.aic.correlator_channels import (
    _fft_correlator_batched,
    ChannelCorrelatorResult,
    compute_channel_correlator,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.physics.electroweak.electroweak_observables import (
    compute_weighted_electroweak_ops_vectorized,
    ELECTROWEAK_OPERATOR_CHANNELS,
    pack_neighbors_from_edges,
    PackedNeighbors,
    PARITY_VELOCITY_CHANNELS,
    SYMMETRY_BREAKING_CHANNELS,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils.aggregation import (
    bin_by_euclidean_time,
)


ELECTROWEAK_BASE_CHANNELS = (
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
ELECTROWEAK_DIRECTIONAL_SU2_CHANNELS = (
    "su2_phase_directed",
    "su2_component_directed",
    "su2_doublet_directed",
    "su2_doublet_diff_directed",
)
ELECTROWEAK_WALKER_TYPE_SU2_CHANNELS = (
    "su2_phase_cloner",
    "su2_phase_resister",
    "su2_phase_persister",
    "su2_component_cloner",
    "su2_component_resister",
    "su2_component_persister",
    "su2_doublet_cloner",
    "su2_doublet_resister",
    "su2_doublet_persister",
    "su2_doublet_diff_cloner",
    "su2_doublet_diff_resister",
    "su2_doublet_diff_persister",
)
ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS = SYMMETRY_BREAKING_CHANNELS
ELECTROWEAK_PARITY_CHANNELS = PARITY_VELOCITY_CHANNELS
ELECTROWEAK_CHANNELS = tuple(ELECTROWEAK_OPERATOR_CHANNELS)


@dataclass
class ElectroweakChannelConfig:
    """Configuration for electroweak channel correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    h_eff: float = 1.0
    use_connected: bool = True
    neighbor_method: str = "auto"
    edge_weight_mode: str = "inverse_riemannian_distance"
    neighbor_weighting: str = "inv_geodesic_full"
    companion_topology: str = "distance"
    companion_topology_u1: str | None = None
    companion_topology_su2: str | None = None
    companion_topology_ew_mixed: str | None = None
    neighbor_k: int = 0
    kernel_length_scale: float = 1.0
    voronoi_pbc_mode: str = "mirror"
    voronoi_exclude_boundary: bool = True
    voronoi_boundary_tolerance: float = 1e-6
    use_time_sliced_tessellation: bool = True
    time_sliced_neighbor_mode: str = "spacelike"

    # Time axis selection (for 4D Euclidean time analysis)
    time_axis: str = "mc"  # "mc" or "euclidean"
    euclidean_time_dim: int = 3  # Which dimension is Euclidean time (0-indexed)
    euclidean_time_bins: int = 50  # Number of time bins for Euclidean analysis
    euclidean_time_range: tuple[float, float] | None = None  # (t_min, t_max) or None for auto
    mc_time_index: int | None = None  # Recorded index for Euclidean slice; None => last

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
    su2_operator_mode: str = "standard"
    enable_walker_type_split: bool = False
    walker_type_scope: str = "frame_global"

    # Bootstrap error estimation
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


@dataclass
class ElectroweakChannelOutput:
    """Computed electroweak correlators and diagnostics."""

    channel_results: dict[str, ChannelCorrelatorResult]
    frame_indices: list[int]
    n_valid_frames: int
    avg_alive_walkers: float
    avg_edges: float


@dataclass
class _ElectroweakSeriesBundle:
    """Internal carrier for per-channel series and frame diagnostics."""

    series_map: dict[str, Tensor]
    frame_indices: list[int]
    n_valid_frames: int
    avg_alive_walkers: float
    avg_edges: float


EDGE_WEIGHT_MODE_ALIASES: dict[str, tuple[str, ...]] = {
    "uniform": ("uniform",),
    "inverse_distance": ("inverse_riemannian_distance", "inv_geodesic_iso"),
    "inverse_volume": ("inverse_riemannian_volume",),
    "kernel": ("kernel", "riemannian_kernel", "riemannian_kernel_volume"),
}


def _resolve_neighbor_method_strict(method: str) -> str:
    method_norm = str(method).strip().lower()
    if method_norm == "uniform":
        method_norm = "companions"
    if method_norm == "voronoi":
        msg = (
            "neighbor_method='voronoi' is disabled for electroweak channels. "
            "Use 'recorded', 'companions', or 'auto' to reuse simulation-recorded data."
        )
        raise ValueError(msg)
    if method_norm not in {"auto", "recorded", "companions"}:
        msg = "neighbor_method must be 'auto', 'recorded', or 'companions'."
        raise ValueError(msg)
    # Electroweak operators are defined on run-selected companion relations.
    # Keep legacy values accepted for compatibility, but route all variants
    # through companion selections.
    return "companions"


def _resolve_companion_topology(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"distance", "clone", "both"}:
        msg = "companion_topology must be 'distance', 'clone', or 'both'."
        raise ValueError(msg)
    return mode_norm


def _resolve_companion_topologies(
    cfg: ElectroweakChannelConfig,
) -> tuple[str, str, str]:
    """Resolve fixed operator-family companion routing.

    U(1) operators are evaluated on diversity (distance) companions,
    SU(2) operators on cloning companions, and EW mixed operators on both.
    """
    _ = cfg
    return ("distance", "clone", "both")


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


def _resolve_edge_weight_mode(
    requested_mode: str,
    edge_dict: dict[str, Tensor],
) -> str | None:
    candidates = [requested_mode]
    candidates.extend(EDGE_WEIGHT_MODE_ALIASES.get(requested_mode, ()))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in edge_dict:
            return candidate
    return None


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
    history: RunHistory, cfg: ElectroweakChannelConfig
) -> dict[str, float]:
    params = history.params if isinstance(history.params, dict) else None
    # Interaction ranges are sourced from run parameters only; they are not
    # user-overridden in the electroweak analysis path.
    epsilon_d = _nested_param(params, "companion_selection", "epsilon", default=None)
    epsilon_c = _nested_param(params, "companion_selection_clone", "epsilon", default=None)
    # Keep velocity-weight contribution pinned off for this pipeline.
    lambda_alg = 0.0
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
    lambda_alg = 0.0

    return {
        "epsilon_d": epsilon_d,
        "epsilon_c": epsilon_c,
        "epsilon_clone": epsilon_clone,
        "lambda_alg": lambda_alg,
    }


def _resolve_lambda_alg(
    history: RunHistory,
    lambda_alg: float | None = None,
) -> float:
    _ = (history, lambda_alg)
    return 0.0


def _resolve_transition_frames(
    history: RunHistory,
    frame_indices: list[int] | None = None,
) -> Tensor:
    device = history.x_before_clone.device
    if frame_indices is None:
        return torch.arange(1, int(history.n_recorded), device=device, dtype=torch.long)
    resolved: list[int] = []
    for raw in frame_indices:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if 1 <= idx < int(history.n_recorded):
            resolved.append(idx)
    if not resolved:
        return torch.zeros(0, device=device, dtype=torch.long)
    return torch.as_tensor(sorted(set(resolved)), device=device, dtype=torch.long)


def _compute_d_alg_sq_for_companions(
    positions: Tensor,
    velocities: Tensor,
    companions: Tensor,
    *,
    lambda_alg: float,
) -> tuple[Tensor, Tensor]:
    n_walkers = int(positions.shape[1])
    companions = companions.to(device=positions.device, dtype=torch.long).clamp(
        min=0, max=max(n_walkers - 1, 0)
    )
    gather_idx = companions.unsqueeze(-1).expand(-1, -1, positions.shape[-1])
    pos_j = torch.gather(positions, dim=1, index=gather_idx)
    vel_j = torch.gather(velocities, dim=1, index=gather_idx)

    diff_x = positions - pos_j
    diff_v = velocities - vel_j
    d_alg_sq = (diff_x**2).sum(dim=-1) + float(lambda_alg) * (diff_v**2).sum(dim=-1)

    src = torch.arange(n_walkers, device=positions.device, dtype=torch.long).unsqueeze(0)
    src = src.expand_as(companions)
    valid = (companions != src) & torch.isfinite(d_alg_sq) & (d_alg_sq > 0)
    return d_alg_sq, valid


def _compute_fitness_gap_abs_for_companions(
    fitness: Tensor,
    companions: Tensor,
    *,
    epsilon_clone: float,
) -> tuple[Tensor, Tensor]:
    n_walkers = int(fitness.shape[1])
    companions = companions.to(device=fitness.device, dtype=torch.long).clamp(
        min=0, max=max(n_walkers - 1, 0)
    )
    fit_j = torch.gather(fitness, dim=1, index=companions)
    fit_i = fitness
    eps = float(epsilon_clone) if epsilon_clone is not None else 1e-8
    eps = max(eps, 1e-12)
    denom = torch.where(
        torch.abs(fit_i) < eps,
        torch.full_like(fit_i, eps),
        fit_i,
    )

    src = torch.arange(n_walkers, device=fitness.device, dtype=torch.long).unsqueeze(0)
    src = src.expand_as(companions)
    gap_abs = torch.abs((fit_j - fit_i) / denom)
    valid = (companions != src) & torch.isfinite(gap_abs)
    return gap_abs, valid


def _compute_offdiag_pairwise_rms_sq(
    matrix: Tensor,
    n_walkers: int,
) -> tuple[float, int]:
    """Return (sum of squares, sample count) for finite positive off-diagonal distances."""
    if matrix.ndim != 2:
        return 0.0, 0
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols or n_rows <= 1:
        return 0.0, 0
    n = min(n_walkers, n_rows)
    if n <= 1:
        return 0.0, 0

    matrix = matrix[:n, :n]
    off_diag = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=matrix.device),
        diagonal=1,
    )
    valid = off_diag & torch.isfinite(matrix) & (matrix > 0)
    flat = matrix[valid]
    if flat.numel() == 0:
        return 0.0, 0
    sum_sq = torch.sum(flat * flat).item()
    return float(sum_sq), int(flat.numel())


def compute_emergent_electroweak_scales(
    history: RunHistory,
    *,
    frame_indices: list[int] | None = None,
    lambda_alg: float | None = None,
    epsilon_clone: float | None = None,
    pairwise_distance_by_frame: dict[int, Tensor] | None = None,
) -> dict[str, float]:
    """Estimate emergent electroweak interaction ranges from recorded trajectories."""
    frames = _resolve_transition_frames(history, frame_indices=frame_indices)
    if frames.numel() == 0:
        return {
            "eps_distance_emergent": float("nan"),
            "eps_geodesic_emergent": float("nan"),
            "eps_clone_emergent": float("nan"),
            "eps_fitness_gap_emergent": float("nan"),
            "n_distance_samples": 0.0,
            "n_geodesic_distance_samples": 0.0,
            "n_clone_samples": 0.0,
            "n_fitness_gap_samples": 0.0,
            "n_frames": 0.0,
            "lambda_alg": float("nan"),
            "epsilon_clone": float("nan"),
        }

    info_idx = frames - 1
    device = history.x_before_clone.device
    positions = history.x_before_clone.index_select(0, frames)
    velocities = history.v_before_clone.index_select(0, frames).to(
        device=positions.device, dtype=positions.dtype
    )
    n_walkers = int(positions.shape[1])
    lambda_alg_resolved = _resolve_lambda_alg(history, lambda_alg=lambda_alg)
    if epsilon_clone is None:
        params = history.params if isinstance(history.params, dict) else None
        epsilon_clone = _nested_param(params, "companion_selection_clone", "epsilon", default=None)
        if epsilon_clone is None:
            epsilon_clone = _nested_param(params, "cloning", "epsilon_clone", default=1e-8)
        if epsilon_clone is None:
            epsilon_clone = 1e-8
    epsilon_clone = float(max(float(epsilon_clone), 1e-12))

    precomputed_distances: dict[int, Tensor] = {}
    if isinstance(pairwise_distance_by_frame, dict):
        for raw_frame, matrix in pairwise_distance_by_frame.items():
            if torch.is_tensor(matrix):
                precomputed_distances[int(raw_frame)] = matrix

    eps_distance = float("nan")
    eps_geodesic = float("nan")
    eps_clone = float("nan")
    eps_fitness_gap = float("nan")
    n_distance_samples = 0
    n_geodesic_distance_samples = 0
    n_clone_samples = 0
    n_fitness_gap_samples = 0

    if precomputed_distances:
        geodesic_sum_sq = 0.0
        geodesic_count = 0
        frame_list = [int(v) for v in frames.tolist()]
        for frame_idx in frame_list:
            matrix = precomputed_distances.get(frame_idx)
            if matrix is None:
                continue
            if not matrix.is_floating_point():
                matrix = matrix.to(dtype=positions.dtype)
            matrix = matrix.to(device=device)
            if matrix.ndim != 2:
                continue
            sum_sq, n_pairs = _compute_offdiag_pairwise_rms_sq(matrix, n_walkers)
            if n_pairs > 0:
                geodesic_sum_sq += sum_sq
                geodesic_count += n_pairs
        if geodesic_count > 0:
            eps_geodesic = float(math.sqrt(geodesic_sum_sq / geodesic_count))
            n_geodesic_distance_samples = geodesic_count

    distances_hist = getattr(history, "distances", None)
    if (
        torch.is_tensor(distances_hist)
        and distances_hist.ndim == 2
        and int(distances_hist.shape[0]) > int(info_idx.max().item())
    ):
        distances = distances_hist.index_select(0, info_idx).to(
            device=device, dtype=positions.dtype
        )
        valid_distance = torch.isfinite(distances) & (distances > 0)
        n_distance_samples = int(valid_distance.sum().item())
        if n_distance_samples > 0:
            eps_distance = float(torch.sqrt((distances[valid_distance] ** 2).mean()).item())

    if not math.isfinite(eps_distance):
        companions_distance = getattr(history, "companions_distance", None)
        if (
            torch.is_tensor(companions_distance)
            and companions_distance.ndim == 2
            and int(companions_distance.shape[0]) > int(info_idx.max().item())
        ):
            d_alg_sq_dist, valid_dist = _compute_d_alg_sq_for_companions(
                positions=positions,
                velocities=velocities,
                companions=companions_distance.index_select(0, info_idx),
                lambda_alg=lambda_alg_resolved,
            )
            n_distance_samples = int(valid_dist.sum().item())
            if n_distance_samples > 0:
                eps_distance = float(torch.sqrt(d_alg_sq_dist[valid_dist].mean()).item())

    companions_clone = getattr(history, "companions_clone", None)
    if (
        torch.is_tensor(companions_clone)
        and companions_clone.ndim == 2
        and int(companions_clone.shape[0]) > int(info_idx.max().item())
    ):
        d_alg_sq_clone, valid_clone = _compute_d_alg_sq_for_companions(
            positions=positions,
            velocities=velocities,
            companions=companions_clone.index_select(0, info_idx),
            lambda_alg=lambda_alg_resolved,
        )
        n_clone_samples = int(valid_clone.sum().item())
        if n_clone_samples > 0:
            eps_clone = float(torch.sqrt(d_alg_sq_clone[valid_clone].mean()).item())

    fitness = getattr(history, "fitness", None)
    if (
        torch.is_tensor(fitness)
        and torch.is_tensor(companions_clone)
        and companions_clone.ndim == 2
        and int(companions_clone.shape[0]) > int(info_idx.max().item())
    ):
        fitness = fitness.index_select(0, info_idx).to(device=device, dtype=positions.dtype)
        fitness_gap_abs, valid_gap = _compute_fitness_gap_abs_for_companions(
            fitness=fitness,
            companions=companions_clone.index_select(0, info_idx),
            epsilon_clone=epsilon_clone,
        )
        n_fitness_gap_samples = int(valid_gap.sum().item())
        if n_fitness_gap_samples > 0:
            eps_fitness_gap = float(torch.sqrt((fitness_gap_abs[valid_gap] ** 2).mean()).item())

    return {
        "eps_distance_emergent": float(eps_distance),
        "eps_geodesic_emergent": float(eps_geodesic),
        "eps_clone_emergent": float(eps_clone),
        "eps_fitness_gap_emergent": float(eps_fitness_gap),
        "n_distance_samples": float(n_distance_samples),
        "n_geodesic_distance_samples": float(n_geodesic_distance_samples),
        "n_clone_samples": float(n_clone_samples),
        "n_fitness_gap_samples": float(n_fitness_gap_samples),
        "n_frames": float(frames.numel()),
        "lambda_alg": float(lambda_alg_resolved),
        "epsilon_clone": float(epsilon_clone),
    }


def compute_electroweak_coupling_constants(
    history: RunHistory | None,
    *,
    h_eff: float,
    frame_indices: list[int] | None = None,
    lambda_alg: float | None = None,
    pairwise_distance_by_frame: dict[int, Tensor] | None = None,
) -> dict[str, float]:
    """Estimate electroweak couplings and emergent Weinberg-angle diagnostics."""
    d = float(history.d) if history is not None else float("nan")
    h_eff = float(max(h_eff, 1e-12))

    if history is not None:
        params = history.params if isinstance(history.params, dict) else None
        nu = _nested_param(params, "kinetic", "nu", default=None)
    else:
        nu = None

    c2d = (d**2 - 1.0) / (2.0 * d) if d > 0 else float("nan")
    c2_2 = 3.0 / 4.0

    mean_force_sq = float("nan")
    if history is not None and getattr(history, "force_viscous", None) is not None:
        force = history.force_viscous
        force_sq = (force**2).sum(dim=-1)
        n_elements = force_sq.numel()
        if n_elements > 0:
            mean_force_sq = float(force_sq.mean().item())

    if nu is None or not math.isfinite(mean_force_sq) or d <= 0:
        g3_est = float("nan")
        kvisc_sq = float("nan")
    else:
        kvisc_sq = float(mean_force_sq / max(float(nu) ** 2, 1e-12))
        g3_sq = (float(nu) ** 2 / h_eff**2) * (d * (d**2 - 1.0) / 12.0) * kvisc_sq
        g3_est = float(max(g3_sq, 0.0) ** 0.5)

    emergent = {
        "eps_distance_emergent": float("nan"),
        "eps_geodesic_emergent": float("nan"),
        "eps_clone_emergent": float("nan"),
        "eps_fitness_gap_emergent": float("nan"),
        "n_distance_samples": 0.0,
        "n_geodesic_distance_samples": 0.0,
        "n_clone_samples": 0.0,
        "n_fitness_gap_samples": 0.0,
        "n_frames": 0.0,
        "lambda_alg": float("nan"),
        "epsilon_clone": float("nan"),
    }
    if history is not None:
        emergent = compute_emergent_electroweak_scales(
            history,
            frame_indices=frame_indices,
            lambda_alg=lambda_alg,
            pairwise_distance_by_frame=pairwise_distance_by_frame,
        )
    eps_distance_em = float(emergent["eps_distance_emergent"])
    eps_geodesic_em = float(emergent["eps_geodesic_emergent"])
    eps_clone_em = float(emergent["eps_clone_emergent"])
    eps_fitness_gap_em = float(emergent["eps_fitness_gap_emergent"])

    if math.isfinite(eps_distance_em) and eps_distance_em > 0:
        g1_em_sq = h_eff / (eps_distance_em**2)
        g1_em = float(max(g1_em_sq, 0.0) ** 0.5)
    else:
        g1_em_sq = float("nan")
        g1_em = float("nan")

    if math.isfinite(eps_clone_em) and eps_clone_em > 0 and math.isfinite(c2d) and c2d > 0:
        g2_em_sq = (2.0 * h_eff / (eps_clone_em**2)) * (c2_2 / c2d)
        g2_em = float(max(g2_em_sq, 0.0) ** 0.5)
    else:
        g2_em_sq = float("nan")
        g2_em = float("nan")

    if (
        math.isfinite(eps_fitness_gap_em)
        and eps_fitness_gap_em > 0
        and math.isfinite(c2d)
        and c2d > 0
    ):
        g2_gap_em_sq = (2.0 * h_eff / (eps_fitness_gap_em**2)) * (c2_2 / c2d)
        g2_gap_em = float(max(g2_gap_em_sq, 0.0) ** 0.5)
    else:
        g2_gap_em_sq = float("nan")
        g2_gap_em = float("nan")

    # Weinberg angle from emergent epsilon scales:
    # sin²(θ_W) = ε_c² / (ε_c² + ε_d²)
    # where ε_c = ε_fitness_gap (SU(2) scale) and ε_d = ε_geodesic (U(1) scale)
    sin2_theta_w_emergent = float("nan")
    tan_theta_w_emergent = float("nan")
    eps_c = eps_fitness_gap_em  # SU(2) coupling scale
    eps_d = eps_geodesic_em  # U(1) hypercharge scale
    if math.isfinite(eps_c) and eps_c > 0 and math.isfinite(eps_d) and eps_d > 0:
        eps_c_sq = eps_c**2
        eps_d_sq = eps_d**2
        denom = eps_c_sq + eps_d_sq
        sin2_theta_w_emergent = float(eps_c_sq / denom)
        tan_theta_w_emergent = float(eps_c / eps_d)

    return {
        "g1_est": float(g1_em),
        "g2_est": float(g2_em),
        "g3_est": float(g3_est),
        "nu": float(nu) if nu is not None else float("nan"),
        "kvisc_sq_proxy": float(kvisc_sq),
        "mean_force_sq": float(mean_force_sq),
        "d": float(d),
        "h_eff": float(h_eff),
        **emergent,
        "g1_est_emergent": float(g1_em),
        "g2_est_emergent": float(g2_em),
        "g2_est_emergent_fitness_gap": float(g2_gap_em),
        "sin2_theta_w_emergent": float(sin2_theta_w_emergent),
        "tan_theta_w_emergent": float(tan_theta_w_emergent),
    }


def _resolve_mc_time_index(history: RunHistory, mc_time_index: int | None) -> int:
    """Resolve a Monte Carlo slice index from either recorded index or step."""
    if history.n_recorded < 2:
        msg = "Need at least 2 recorded timesteps for Euclidean analysis."
        raise ValueError(msg)
    if mc_time_index is None:
        resolved = history.n_recorded - 1
    else:
        try:
            raw = int(mc_time_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mc_time_index: {mc_time_index}") from exc
        if raw in history.recorded_steps:
            resolved = history.get_step_index(raw)
        else:
            resolved = raw
    if resolved < 1 or resolved >= history.n_recorded:
        msg = (
            f"mc_time_index {resolved} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1} "
            "or a recorded step value)."
        )
        raise ValueError(msg)
    return resolved


def _extract_will_clone_for_frame(
    history: RunHistory,
    info_idx: int,
    *,
    n_walkers: int,
    device: torch.device,
) -> Tensor | None:
    will_clone_hist = getattr(history, "will_clone", None)
    if will_clone_hist is None:
        return None
    if info_idx < 0 or info_idx >= int(will_clone_hist.shape[0]):
        return None
    will_clone = will_clone_hist[info_idx]
    if not torch.is_tensor(will_clone):
        will_clone = torch.as_tensor(will_clone)
    will_clone = will_clone.to(device=device, dtype=torch.bool)
    if will_clone.ndim != 1 or int(will_clone.shape[0]) != int(n_walkers):
        return None
    return will_clone


def _build_neighbor_data_from_history(
    history: RunHistory,
    frame_idx: int,
    mode: str,
    n_walkers: int,
    device: torch.device,
    max_neighbors: int = 0,
) -> PackedNeighbors | None:
    """Build per-walker neighbor indices and weights from pre-computed RunHistory data."""
    if (
        history.neighbor_edges is None
        or history.edge_weights is None
        or frame_idx >= len(history.neighbor_edges)
        or frame_idx >= len(history.edge_weights)
    ):
        return None

    edges = history.neighbor_edges[frame_idx]
    ew_dict = history.edge_weights[frame_idx]
    if not torch.is_tensor(edges) or edges.numel() == 0:
        return None
    if mode == "uniform":
        weights_flat = torch.ones(edges.shape[0], device=edges.device, dtype=torch.float32)
    else:
        if not isinstance(ew_dict, dict):
            return None
        resolved_mode = _resolve_edge_weight_mode(mode, ew_dict)
        if resolved_mode is None:
            return None
        weights_flat = ew_dict[resolved_mode]
    all_alive = torch.ones(n_walkers, dtype=torch.bool, device=device)
    return pack_neighbors_from_edges(
        edges=edges,
        edge_weights=weights_flat,
        alive=all_alive,
        n_walkers=n_walkers,
        max_neighbors=int(max_neighbors),
        device=device,
        dtype=torch.float32,
    )


def _build_neighbor_data_from_companions(
    history: RunHistory,
    frame_idx: int,
    n_walkers: int,
    device: torch.device,
    companion_topology: str = "distance",
    max_neighbors: int = 0,
) -> PackedNeighbors | None:
    if frame_idx < 1:
        return None
    mode = _resolve_companion_topology(companion_topology)
    companions_distance = getattr(history, "companions_distance", None)
    companions_clone = getattr(history, "companions_clone", None)
    if companions_distance is None:
        return None
    if mode in {"clone", "both"} and companions_clone is None:
        return None
    info_idx = frame_idx - 1
    if info_idx < 0 or info_idx >= companions_distance.shape[0]:
        return None
    if companions_clone is not None and info_idx >= companions_clone.shape[0]:
        return None

    comp_d = companions_distance[info_idx]
    if not torch.is_tensor(comp_d):
        comp_d = torch.as_tensor(comp_d)
    comp_d = comp_d.to(device=device, dtype=torch.long)

    comp_c: Tensor | None = None
    if companions_clone is not None:
        comp_c = companions_clone[info_idx]
        if not torch.is_tensor(comp_c):
            comp_c = torch.as_tensor(comp_c)
        comp_c = comp_c.to(device=device, dtype=torch.long)

    if comp_d.numel() != n_walkers:
        return None
    if mode in {"clone", "both"} and (comp_c is None or comp_c.numel() != n_walkers):
        return None
    comp_d = comp_d.clamp(min=0, max=max(n_walkers - 1, 0))
    if comp_c is not None:
        comp_c = comp_c.clamp(min=0, max=max(n_walkers - 1, 0))

    src = torch.arange(n_walkers, device=device, dtype=torch.long)
    if mode == "distance":
        dst = comp_d
    elif mode == "clone":
        if comp_c is None:
            return None
        dst = comp_c
    else:
        if comp_c is None:
            return None
        src = torch.cat([src, src], dim=0)
        dst = torch.cat([comp_d, comp_c], dim=0)

    edges = torch.stack([src, dst], dim=1)
    weights = torch.ones(edges.shape[0], device=device, dtype=torch.float32)
    all_alive = torch.ones(n_walkers, dtype=torch.bool, device=device)
    return pack_neighbors_from_edges(
        edges=edges,
        edge_weights=weights,
        alive=all_alive,
        n_walkers=n_walkers,
        max_neighbors=int(max_neighbors),
        device=device,
        dtype=torch.float32,
    )


def _require_recorded_or_companion_neighbors(
    history: RunHistory,
    frame_idx: int,
    n_walkers: int,
    device: torch.device,
    edge_weight_mode: str,
    neighbor_method: str,
    companion_topology: str = "distance",
    max_neighbors: int = 0,
) -> PackedNeighbors:
    method = _resolve_neighbor_method_strict(neighbor_method)
    if method in {"auto", "recorded"}:
        packed = _build_neighbor_data_from_history(
            history=history,
            frame_idx=frame_idx,
            mode=edge_weight_mode,
            n_walkers=n_walkers,
            device=device,
            max_neighbors=max_neighbors,
        )
        if packed is not None:
            return packed
        if method == "recorded":
            edge_weights_history = getattr(history, "edge_weights", None)
            if edge_weights_history is None:
                msg = (
                    "neighbor_method='recorded' requires RunHistory.edge_weights "
                    "to be recorded during simulation."
                )
                raise ValueError(msg)
            if frame_idx >= len(edge_weights_history):
                msg = (
                    f"Recorded frame {frame_idx} missing in edge_weights "
                    f"(available 0..{len(edge_weights_history) - 1})."
                )
                raise ValueError(msg)
            edge_dict = edge_weights_history[frame_idx]
            available = (
                ", ".join(sorted(str(k) for k in edge_dict.keys()))
                if isinstance(edge_dict, dict)
                else ""
            )
            msg = (
                f"edge_weights[{frame_idx}] does not contain mode '{edge_weight_mode}'. "
                f"Available modes: [{available}]"
            )
            raise ValueError(msg)

    if method in {"auto", "companions"}:
        packed = _build_neighbor_data_from_companions(
            history=history,
            frame_idx=frame_idx,
            n_walkers=n_walkers,
            device=device,
            companion_topology=companion_topology,
            max_neighbors=max_neighbors,
        )
        if packed is not None:
            return packed

    msg = (
        "Electroweak channels require simulation-recorded neighbor or companion data. "
        "Enable neighbor graph recording (neighbor_edges + edge_weights) or companions in RunHistory."
    )
    raise ValueError(msg)


def _require_family_neighbors(
    *,
    history: RunHistory,
    frame_idx: int,
    n_walkers: int,
    device: torch.device,
    edge_weight_mode: str,
    neighbor_method: str,
    topology_u1: str,
    topology_su2: str,
    topology_ew_mixed: str,
    max_neighbors: int,
) -> tuple[PackedNeighbors, PackedNeighbors, PackedNeighbors, dict[str, PackedNeighbors]]:
    """Build cached packed neighbors for each electroweak operator family."""
    cache: dict[str, PackedNeighbors] = {}

    def _get(mode: str) -> PackedNeighbors:
        packed = cache.get(mode)
        if packed is None:
            packed = _require_recorded_or_companion_neighbors(
                history=history,
                frame_idx=frame_idx,
                n_walkers=n_walkers,
                device=device,
                edge_weight_mode=edge_weight_mode,
                neighbor_method=neighbor_method,
                companion_topology=mode,
                max_neighbors=max_neighbors,
            )
            cache[mode] = packed
        return packed

    return _get(topology_u1), _get(topology_su2), _get(topology_ew_mixed), cache


def _masked_mean(values: Tensor) -> Tensor:
    return values.mean(dim=1)


def _compute_electroweak_series(
    history: RunHistory,
    cfg: ElectroweakChannelConfig,
) -> _ElectroweakSeriesBundle:
    frame_indices: list[int] = []
    alive_counts: list[float] = []
    edge_counts: list[float] = []

    start_idx = max(1, int(history.n_recorded * cfg.warmup_fraction))
    end_fraction = getattr(cfg, "end_fraction", 1.0)
    end_idx = max(start_idx + 1, int(history.n_recorded * end_fraction))
    if cfg.time_axis == "euclidean":
        start_idx = _resolve_mc_time_index(history, cfg.mc_time_index)
    if start_idx >= end_idx:
        return _ElectroweakSeriesBundle(
            series_map={},
            frame_indices=[],
            n_valid_frames=0,
            avg_alive_walkers=0.0,
            avg_edges=0.0,
        )

    h_eff = float(max(cfg.h_eff, 1e-8))
    edge_weight_mode = getattr(cfg, "edge_weight_mode", "inverse_riemannian_distance")
    su2_operator_mode = _resolve_su2_operator_mode(getattr(cfg, "su2_operator_mode", "standard"))
    walker_type_scope = _resolve_walker_type_scope(
        getattr(cfg, "walker_type_scope", "frame_global")
    )
    topology_u1, topology_su2, topology_ew_mixed = _resolve_companion_topologies(cfg)
    enable_walker_type_split = bool(getattr(cfg, "enable_walker_type_split", False))
    params = _resolve_electroweak_params(history, cfg)
    epsilon_d = float(params["epsilon_d"])
    epsilon_c = float(params["epsilon_c"])
    epsilon_clone = float(params["epsilon_clone"])
    lambda_alg = float(params["lambda_alg"])

    if cfg.time_axis == "euclidean":
        frame_idx = start_idx
        positions = history.x_before_clone[frame_idx]
        velocities = history.v_before_clone[frame_idx]
        info_idx = frame_idx - 1
        fitness = history.fitness[info_idx]
        n_walkers = int(positions.shape[0])
        device = positions.device
        will_clone = _extract_will_clone_for_frame(
            history,
            info_idx,
            n_walkers=n_walkers,
            device=device,
        )
        packed_u1, packed_su2, packed_ew_mixed, neighbor_cache = _require_family_neighbors(
            history=history,
            frame_idx=frame_idx,
            n_walkers=n_walkers,
            device=device,
            edge_weight_mode=edge_weight_mode,
            neighbor_method=cfg.neighbor_method,
            topology_u1=topology_u1,
            topology_su2=topology_su2,
            topology_ew_mixed=topology_ew_mixed,
            max_neighbors=int(cfg.neighbor_k),
        )
        all_alive = torch.ones(n_walkers, dtype=torch.bool, device=device)
        operators = compute_weighted_electroweak_ops_vectorized(
            positions=positions,
            velocities=velocities,
            fitness=fitness,
            alive=all_alive,
            h_eff=h_eff,
            epsilon_d=epsilon_d,
            epsilon_c=epsilon_c,
            epsilon_clone=epsilon_clone,
            lambda_alg=lambda_alg,
            bounds=None,
            pbc=False,
            will_clone=will_clone,
            su2_operator_mode=su2_operator_mode,
            enable_walker_type_split=enable_walker_type_split,
            walker_type_scope=walker_type_scope,
            neighbors_u1=packed_u1,
            neighbors_su2=packed_su2,
            neighbors_ew_mixed=packed_ew_mixed,
        )
        frame_indices.append(int(frame_idx))
        alive_counts.append(float(n_walkers))
        edge_samples = [float(packed.valid.sum().item()) for packed in neighbor_cache.values()]
        edge_counts.append(float(sum(edge_samples) / max(len(edge_samples), 1)))
    else:
        T = end_idx - start_idx
        if T <= 0:
            return _ElectroweakSeriesBundle(
                series_map={},
                frame_indices=[],
                n_valid_frames=0,
                avg_alive_walkers=0.0,
                avg_edges=0.0,
            )
        operators = {name: [] for name in ELECTROWEAK_CHANNELS}
        for frame_idx in range(start_idx, end_idx):
            positions = history.x_before_clone[frame_idx]
            velocities = history.v_before_clone[frame_idx]
            info_idx = frame_idx - 1
            fitness = history.fitness[info_idx]
            n_walkers = int(positions.shape[0])
            device = positions.device
            will_clone = _extract_will_clone_for_frame(
                history,
                info_idx,
                n_walkers=n_walkers,
                device=device,
            )
            packed_u1, packed_su2, packed_ew_mixed, neighbor_cache = _require_family_neighbors(
                history=history,
                frame_idx=frame_idx,
                n_walkers=n_walkers,
                device=device,
                edge_weight_mode=edge_weight_mode,
                neighbor_method=cfg.neighbor_method,
                topology_u1=topology_u1,
                topology_su2=topology_su2,
                topology_ew_mixed=topology_ew_mixed,
                max_neighbors=int(cfg.neighbor_k),
            )
            all_alive = torch.ones(n_walkers, dtype=torch.bool, device=device)
            frame_ops = compute_weighted_electroweak_ops_vectorized(
                positions=positions,
                velocities=velocities,
                fitness=fitness,
                alive=all_alive,
                h_eff=h_eff,
                epsilon_d=epsilon_d,
                epsilon_c=epsilon_c,
                epsilon_clone=epsilon_clone,
                lambda_alg=lambda_alg,
                bounds=None,
                pbc=False,
                will_clone=will_clone,
                su2_operator_mode=su2_operator_mode,
                enable_walker_type_split=enable_walker_type_split,
                walker_type_scope=walker_type_scope,
                neighbors_u1=packed_u1,
                neighbors_su2=packed_su2,
                neighbors_ew_mixed=packed_ew_mixed,
            )
            frame_indices.append(int(frame_idx))
            alive_counts.append(float(n_walkers))
            edge_samples = [float(packed.valid.sum().item()) for packed in neighbor_cache.values()]
            edge_counts.append(float(sum(edge_samples) / max(len(edge_samples), 1)))
            for name in operators:
                operators[name].append(frame_ops[name])
        operators = {name: torch.stack(values, dim=0) for name, values in operators.items()}

    if cfg.time_axis == "euclidean":
        if history.d < cfg.euclidean_time_dim + 1:
            msg = (
                f"Cannot use dimension {cfg.euclidean_time_dim} as Euclidean time "
                f"(only {history.d} dimensions available)"
            )
            raise ValueError(msg)

        positions = history.x_before_clone[start_idx : start_idx + 1]
        n_walkers_slice = int(positions.shape[1])
        alive_slice = torch.ones(1, n_walkers_slice, dtype=torch.bool, device=positions.device)

        series: dict[str, Tensor] = {}
        for name, op_values in operators.items():
            if op_values.dim() == 1:
                op_values = op_values.unsqueeze(0)
            if op_values.is_complex():
                _coords_r, series_real = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values.real[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
                _coords_i, series_imag = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values.imag[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
                series[name] = series_real + 1j * series_imag
            else:
                _coords, series[name] = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
    else:
        series = {name: _masked_mean(op_values) for name, op_values in operators.items()}

    n_valid_frames = len(frame_indices)
    avg_alive_walkers = float(sum(alive_counts) / n_valid_frames) if n_valid_frames > 0 else 0.0
    avg_edges = float(sum(edge_counts) / n_valid_frames) if n_valid_frames > 0 else 0.0
    return _ElectroweakSeriesBundle(
        series_map=series,
        frame_indices=frame_indices,
        n_valid_frames=n_valid_frames,
        avg_alive_walkers=avg_alive_walkers,
        avg_edges=avg_edges,
    )


def _to_correlator_config(cfg: ElectroweakChannelConfig) -> CorrelatorConfig:
    return CorrelatorConfig(
        max_lag=int(cfg.max_lag),
        use_connected=bool(cfg.use_connected),
        window_widths=cfg.window_widths,
        min_mass=float(cfg.min_mass),
        max_mass=float(cfg.max_mass),
        fit_mode=str(cfg.fit_mode),
        fit_start=int(cfg.fit_start),
        fit_stop=cfg.fit_stop,
        min_fit_points=int(cfg.min_fit_points),
        compute_bootstrap_errors=bool(cfg.compute_bootstrap_errors),
        n_bootstrap=int(cfg.n_bootstrap),
    )


def _build_result_from_precomputed(
    channel_name: str,
    series: Tensor,
    correlator: Tensor,
    correlator_err: Tensor | None,
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
        correlator_err=correlator_err,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series,
        n_samples=int(series.numel()),
        dt=dt,
        **window_data,
    )


def _compute_channel_results_batched(
    series_map: dict[str, Tensor],
    dt: float,
    config: CorrelatorConfig,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute correlators and masses using grouped batched FFT operations."""
    results: dict[str, ChannelCorrelatorResult] = {}
    if not series_map:
        return results

    series_buffers: dict[str, Tensor] = {}
    groups: dict[bytes, tuple[Tensor, list[str]]] = {}

    for name, series in series_map.items():
        real_series = series.real if series.is_complex() else series
        real_series = real_series.float()
        valid = torch.isfinite(real_series)
        series_buffers[name] = real_series

        key = valid.detach().cpu().numpy().tobytes()
        if key in groups:
            groups[key][1].append(name)
        else:
            groups[key] = (valid, [name])

    for valid_t, names in groups.values():
        if not bool(torch.any(valid_t)):
            for name in names:
                empty_series = torch.zeros(0, device=valid_t.device, dtype=torch.float32)
                results[name] = compute_channel_correlator(
                    series=empty_series,
                    dt=dt,
                    config=config,
                    channel_name=name,
                )
            continue

        series_stack = torch.stack(
            [series_buffers[name][valid_t] for name in names], dim=0
        ).float()
        correlators = _fft_correlator_batched(
            series_stack,
            max_lag=int(config.max_lag),
            use_connected=bool(config.use_connected),
        )

        correlator_errs: Tensor | None = None
        if config.compute_bootstrap_errors:
            n_bootstrap = int(max(1, config.n_bootstrap))
            t_len = int(series_stack.shape[1])
            idx = torch.randint(0, t_len, (n_bootstrap, t_len), device=series_stack.device)
            idx = idx.unsqueeze(1).expand(-1, series_stack.shape[0], -1)
            sampled = torch.gather(
                series_stack.unsqueeze(0).expand(n_bootstrap, -1, -1),
                dim=2,
                index=idx,
            )
            boot_corr = _fft_correlator_batched(
                sampled.reshape(-1, t_len),
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
            )
            correlator_errs = boot_corr.reshape(n_bootstrap, series_stack.shape[0], -1).std(dim=0)

        for idx_name, name in enumerate(names):
            err = correlator_errs[idx_name] if correlator_errs is not None else None
            results[name] = _build_result_from_precomputed(
                channel_name=name,
                series=series_stack[idx_name],
                correlator=correlators[idx_name],
                correlator_err=err,
                dt=dt,
                config=config,
            )

    return results


def _resolve_requested_channels(channels: list[str] | None) -> list[str]:
    if channels is None:
        return list(ELECTROWEAK_BASE_CHANNELS)
    requested = [str(name) for name in channels]
    unknown = sorted(set(requested) - set(ELECTROWEAK_CHANNELS))
    if unknown:
        msg = (
            f"Unsupported electroweak channels {unknown}; supported: {list(ELECTROWEAK_CHANNELS)}."
        )
        raise ValueError(msg)
    return requested


def compute_electroweak_channels(
    history: RunHistory,
    channels: list[str] | None = None,
    config: ElectroweakChannelConfig | None = None,
) -> ElectroweakChannelOutput:
    """Compute electroweak correlators and return results plus diagnostics."""
    cfg = config or ElectroweakChannelConfig()
    requested_channels = _resolve_requested_channels(channels)
    series_bundle = _compute_electroweak_series(history, cfg)
    selected_series = {
        name: series_bundle.series_map[name]
        for name in requested_channels
        if name in series_bundle.series_map
    }
    correlator_cfg = _to_correlator_config(cfg)
    dt = float(history.delta_t * history.record_every)
    channel_results = _compute_channel_results_batched(
        series_map=selected_series,
        dt=dt,
        config=correlator_cfg,
    )
    return ElectroweakChannelOutput(
        channel_results=channel_results,
        frame_indices=series_bundle.frame_indices,
        n_valid_frames=series_bundle.n_valid_frames,
        avg_alive_walkers=series_bundle.avg_alive_walkers,
        avg_edges=series_bundle.avg_edges,
    )


def compute_all_electroweak_channels(
    history: RunHistory,
    channels: list[str] | None = None,
    config: ElectroweakChannelConfig | None = None,
) -> dict[str, ChannelCorrelatorResult]:
    """Compatibility wrapper returning only channel results."""
    output = compute_electroweak_channels(history, channels=channels, config=config)
    return output.channel_results


def compute_electroweak_snapshot_operators(
    history: RunHistory,
    config: ElectroweakChannelConfig | None = None,
    channels: list[str] | None = None,
    frame_idx: int | None = None,
) -> dict[str, Tensor]:
    """Compute per-walker electroweak operators at a single MC snapshot."""
    cfg = config or ElectroweakChannelConfig()
    requested_channels = _resolve_requested_channels(channels)

    if frame_idx is None:
        frame_idx = _resolve_mc_time_index(history, cfg.mc_time_index)
    if frame_idx < 1 or frame_idx >= history.n_recorded:
        msg = (
            f"frame_idx {frame_idx} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1})."
        )
        raise ValueError(msg)

    h_eff = float(max(cfg.h_eff, 1e-8))
    su2_operator_mode = _resolve_su2_operator_mode(getattr(cfg, "su2_operator_mode", "standard"))
    walker_type_scope = _resolve_walker_type_scope(
        getattr(cfg, "walker_type_scope", "frame_global")
    )
    topology_u1, topology_su2, topology_ew_mixed = _resolve_companion_topologies(cfg)
    enable_walker_type_split = bool(getattr(cfg, "enable_walker_type_split", False))
    params = _resolve_electroweak_params(history, cfg)
    epsilon_d = float(params["epsilon_d"])
    epsilon_c = float(params["epsilon_c"])
    epsilon_clone = float(params["epsilon_clone"])
    lambda_alg = float(params["lambda_alg"])

    positions = history.x_before_clone[frame_idx]
    velocities = history.v_before_clone[frame_idx]
    info_idx = frame_idx - 1
    fitness = history.fitness[info_idx]
    n_walkers = int(positions.shape[0])
    device = positions.device
    will_clone = _extract_will_clone_for_frame(
        history,
        info_idx,
        n_walkers=n_walkers,
        device=device,
    )

    edge_weight_mode = getattr(cfg, "edge_weight_mode", "inverse_riemannian_distance")
    packed_u1, packed_su2, packed_ew_mixed, _neighbor_cache = _require_family_neighbors(
        history=history,
        frame_idx=frame_idx,
        n_walkers=n_walkers,
        device=device,
        edge_weight_mode=edge_weight_mode,
        neighbor_method=cfg.neighbor_method,
        topology_u1=topology_u1,
        topology_su2=topology_su2,
        topology_ew_mixed=topology_ew_mixed,
        max_neighbors=int(cfg.neighbor_k),
    )
    all_alive = torch.ones(n_walkers, dtype=torch.bool, device=device)
    operators = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=all_alive,
        h_eff=h_eff,
        epsilon_d=epsilon_d,
        epsilon_c=epsilon_c,
        epsilon_clone=epsilon_clone,
        lambda_alg=lambda_alg,
        bounds=None,
        pbc=False,
        will_clone=will_clone,
        su2_operator_mode=su2_operator_mode,
        enable_walker_type_split=enable_walker_type_split,
        walker_type_scope=walker_type_scope,
        neighbors_u1=packed_u1,
        neighbors_su2=packed_su2,
        neighbors_ew_mixed=packed_ew_mixed,
    )

    return {name: operators[name] for name in requested_channels if name in operators}
