"""Electroweak (U1/SU2) channel correlators for Fractal Gas runs.

This module builds U(1) fitness-phase and SU(2) cloning-phase time series and
extracts effective masses using the same correlator pipeline as strong-force
channels (FFT correlators + AIC/linear fits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.higgs_observables import (
    compute_emergent_metric,
    compute_geodesic_distances,
)
from fragile.fractalai.qft.aggregation import (
    bin_by_euclidean_time,
)
from fragile.fractalai.qft.correlator_channels import (
    ChannelCorrelatorResult,
    ConvolutionalAICExtractor,
    bootstrap_correlator_error,
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
    neighbor_method: str = "voronoi"
    edge_weight_mode: str = "inverse_riemannian_distance"
    neighbor_weighting: str = "inv_geodesic_full"
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

    # Bootstrap error estimation
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


NEIGHBOR_WEIGHT_MODES = (
    "volume",
    "euclidean",
    "inv_euclidean",
    "inv_geodesic_iso",
    "inv_geodesic_full",
    "kernel",
)


def _collect_time_sliced_edges(time_sliced, mode: str) -> np.ndarray:
    edges: list[np.ndarray] = []
    if mode in {"spacelike", "spacelike+timelike"}:
        for bin_result in time_sliced.bins:
            if bin_result.spacelike_edges is not None and bin_result.spacelike_edges.size:
                edges.append(bin_result.spacelike_edges)
    if mode in {"timelike", "spacelike+timelike"}:
        if (
            time_sliced.timelike_edges is not None
            and time_sliced.timelike_edges.size
        ):
            edges.append(time_sliced.timelike_edges)
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.vstack(edges)


def _build_neighbor_lists(edges: np.ndarray, n: int) -> list[list[int]]:
    neighbors = [[] for _ in range(n)]
    if edges.size == 0:
        return neighbors
    for i, j in edges:
        if i == j:
            continue
        if 0 <= i < n and 0 <= j < n:
            neighbors[i].append(int(j))
    for idx, items in enumerate(neighbors):
        if len(items) <= 1:
            continue
        seen: set[int] = set()
        unique = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        neighbors[idx] = unique
    return neighbors


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


def _resolve_mc_time_index(history: RunHistory, mc_time_index: int | None) -> int:
    """Resolve a Monte Carlo slice index from either recorded index or step."""
    if history.n_recorded < 2:
        raise ValueError("Need at least 2 recorded timesteps for Euclidean analysis.")
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


def _get_volume_weights(history: RunHistory, frame_idx: int) -> Tensor | None:
    weights = getattr(history, "riemannian_volume_weights", None)
    if weights is None or frame_idx < 1:
        return None
    info_idx = min(frame_idx - 1, len(weights) - 1)
    if info_idx < 0:
        return None
    return weights[info_idx]


def _apply_pbc_diff(diff: Tensor, bounds) -> Tensor:
    if bounds is None:
        return diff
    high = bounds.high.to(diff)
    low = bounds.low.to(diff)
    span = high - low
    return diff - span * torch.round(diff / span)


def _validate_neighbor_weighting(weight_mode: str) -> str:
    if weight_mode == "uniform":
        raise ValueError("uniform weighting is disabled; choose a Voronoi weight mode instead")
    if weight_mode not in NEIGHBOR_WEIGHT_MODES:
        raise ValueError(f"neighbor_weighting must be one of {NEIGHBOR_WEIGHT_MODES}")
    return weight_mode


def _map_voronoi_neighbors_to_global(
    neighbor_lists: dict[int, list[int]] | None,
    alive_idx: Tensor,
) -> dict[int, list[int]]:
    if not neighbor_lists:
        return {}
    alive_list = alive_idx.tolist()
    mapped: dict[int, list[int]] = {}
    for i_local, neighbors in neighbor_lists.items():
        if i_local >= len(alive_list):
            continue
        gi = int(alive_list[i_local])
        mapped[gi] = [int(alive_list[j]) for j in neighbors if j < len(alive_list)]
    return mapped


def _build_edge_weight_map(
    edges: np.ndarray,
    weights: np.ndarray,
    alive_idx: Tensor,
) -> dict[tuple[int, int], float]:
    if edges.size == 0 or weights.size == 0:
        return {}
    alive_list = alive_idx.tolist()
    edge_map: dict[tuple[int, int], float] = {}
    for (i, j), w in zip(edges, weights, strict=False):
        if i >= len(alive_list) or j >= len(alive_list):
            continue
        gi = int(alive_list[int(i)])
        gj = int(alive_list[int(j)])
        edge_map[(gi, gj)] = float(w)
        edge_map[(gj, gi)] = float(w)
    return edge_map


def _compute_edge_weights(
    positions: Tensor,
    edges: np.ndarray,
    alive_mask: Tensor,
    bounds,
    weight_mode: str,
    pbc: bool,
) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=float)
    pos = positions.detach().cpu().numpy()
    if weight_mode == "inv_geodesic_iso":
        diff = pos[edges[:, 0]] - pos[edges[:, 1]]
        if pbc and bounds is not None:
            diff = _apply_pbc_diff(torch.as_tensor(diff), bounds).cpu().numpy()
        weights = np.linalg.norm(diff, axis=1)
        return np.where(weights <= 0, 1e-6, weights)

    edge_index = torch.tensor(edges.T, device=positions.device, dtype=torch.long)
    metric = compute_emergent_metric(positions, edge_index, alive_mask)
    geo = compute_geodesic_distances(positions, edge_index, metric, alive_mask)
    return geo.detach().cpu().numpy()


def _build_neighbor_data_from_history(
    history: RunHistory,
    frame_idx: int,
    mode: str,
    alive: Tensor,
    max_neighbors: int = 0,
) -> tuple[list[Tensor], list[Tensor]] | None:
    """Build per-walker neighbor indices and weights from pre-computed RunHistory data.

    Returns (neighbor_indices, neighbor_weights) as lists of N tensors,
    or None if pre-computed data is unavailable.
    """
    if (history.neighbor_edges is None or history.edge_weights is None
            or frame_idx >= len(history.neighbor_edges)
            or frame_idx >= len(history.edge_weights)):
        return None

    edges = history.neighbor_edges[frame_idx]
    ew_dict = history.edge_weights[frame_idx]
    if not torch.is_tensor(edges) or edges.numel() == 0:
        return None
    if mode not in ew_dict:
        return None
    weights_flat = ew_dict[mode]

    device = alive.device
    N = alive.shape[0]
    alive_idx = torch.where(alive)[0]
    alive_set = set(alive_idx.tolist())

    edges_d = edges.to(device)
    weights_d = weights_flat.float().to(device)

    n_total = N
    neighbor_indices: list[Tensor] = [
        torch.tensor([], device=device, dtype=torch.long) for _ in range(n_total)
    ]
    neighbor_weights: list[Tensor] = [
        torch.tensor([], device=device, dtype=torch.float32) for _ in range(n_total)
    ]

    for g_idx in alive_idx.tolist():
        # Find edges where source == g_idx
        mask = edges_d[:, 0] == g_idx
        nbr = edges_d[mask, 1]
        w = weights_d[mask]

        # Filter to alive neighbors only
        alive_mask = torch.tensor(
            [n.item() in alive_set for n in nbr], device=device, dtype=torch.bool
        )
        if alive_mask.any():
            nbr = nbr[alive_mask]
            w = w[alive_mask]

        # Truncate to max_neighbors if set
        if max_neighbors > 0 and len(nbr) > max_neighbors:
            topk = w.topk(max_neighbors)
            nbr = nbr[topk.indices]
            w = topk.values

        # Normalize
        if w.sum() > 0:
            w = w / w.sum()

        neighbor_indices[g_idx] = nbr
        neighbor_weights[g_idx] = w

    return neighbor_indices, neighbor_weights


def _build_voronoi_neighbor_data(
    history: RunHistory,
    positions: Tensor,
    alive: Tensor,
    cfg: ElectroweakChannelConfig,
    volume_weights: Tensor | None = None,
) -> tuple[dict[int, list[int]], np.ndarray, dict[tuple[int, int], float], Tensor]:
    weight_mode = _validate_neighbor_weighting(
        str(getattr(cfg, "neighbor_weighting", "inv_geodesic_full"))
    )
    try:
        from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation
    except Exception as exc:
        raise RuntimeError(f"Voronoi tessellation required for electroweak weighting: {exc}") from exc

    if not alive.any():
        return {}, np.zeros((0,), dtype=float), {}, torch.where(alive)[0]

    vor_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive,
        bounds=history.bounds,
        pbc=history.pbc,
        pbc_mode=cfg.voronoi_pbc_mode,
        exclude_boundary=cfg.voronoi_exclude_boundary,
        boundary_tolerance=cfg.voronoi_boundary_tolerance,
        compute_curvature=False,
    )
    neighbor_lists_local = vor_data.get("neighbor_lists", {}) or {}
    volumes = vor_data.get("volumes", np.ones(len(neighbor_lists_local), dtype=float))
    alive_idx = torch.where(alive)[0]
    if volume_weights is not None and alive_idx.numel() > 0:
        if torch.is_tensor(volume_weights):
            weights_full = volume_weights.detach().cpu().numpy()
        else:
            weights_full = np.asarray(volume_weights)
        alive_np = alive_idx.detach().cpu().numpy()
        if weights_full.size > int(alive_np.max()):
            volumes = weights_full[alive_np]
    neighbor_lists_global = _map_voronoi_neighbors_to_global(neighbor_lists_local, alive_idx)

    edge_weight_map: dict[tuple[int, int], float] = {}
    if weight_mode in {"inv_geodesic_iso", "inv_geodesic_full"}:
        edges = []
        for i, neighbors in neighbor_lists_local.items():
            for j in neighbors:
                edges.append((int(i), int(j)))
        edges_array = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
        positions_alive = positions[alive_idx]
        alive_mask_alive = torch.ones(positions_alive.shape[0], device=positions.device, dtype=torch.bool)
        weights = _compute_edge_weights(
            positions_alive,
            edges_array,
            alive_mask_alive,
            history.bounds,
            weight_mode,
            bool(history.pbc),
        )
        edge_weight_map = _build_edge_weight_map(edges_array, weights, alive_idx)

    return neighbor_lists_global, volumes, edge_weight_map, alive_idx


def _build_neighbor_weights(
    positions: Tensor,
    alive_idx: Tensor,
    neighbor_lists_global: dict[int, list[int]],
    volumes: np.ndarray,
    weight_mode: str,
    edge_weight_map: dict[tuple[int, int], float],
    bounds,
    pbc: bool,
    max_neighbors: int,
    kernel_length_scale: float = 1.0,
) -> tuple[list[Tensor], list[Tensor]]:
    alive_list = alive_idx.tolist()
    global_to_alive = {int(g): i for i, g in enumerate(alive_list)}
    n_total = positions.shape[0]
    neighbors_out: list[Tensor] = [torch.tensor([], device=positions.device, dtype=torch.long) for _ in range(n_total)]
    weights_out: list[Tensor] = [torch.tensor([], device=positions.device, dtype=positions.dtype) for _ in range(n_total)]

    for g_idx in alive_list:
        neighbors_global = neighbor_lists_global.get(int(g_idx), [])
        if not neighbors_global:
            neighbors_global = [int(g_idx)]
        neighbors_tensor = torch.tensor(neighbors_global, device=positions.device, dtype=torch.long)
        if weight_mode == "volume":
            weights = torch.tensor(
                [volumes[global_to_alive[n]] if n in global_to_alive else 0.0 for n in neighbors_global],
                device=positions.device,
                dtype=positions.dtype,
            )
        else:
            pos_i = positions[int(g_idx)]
            pos_j = positions[neighbors_tensor]
            diff = pos_j - pos_i
            if pbc and bounds is not None:
                diff = _apply_pbc_diff(diff, bounds)
            dist = torch.linalg.vector_norm(diff, dim=-1).clamp(min=1e-8)
            if weight_mode == "kernel":
                weights = torch.exp(-(dist ** 2) / (2.0 * kernel_length_scale ** 2))
            elif weight_mode == "euclidean":
                weights = dist
            elif weight_mode == "inv_euclidean":
                weights = 1.0 / dist
            else:
                geo_weights = []
                for nbr_global, base_dist in zip(neighbors_global, dist.tolist(), strict=False):
                    edge_weight = edge_weight_map.get((int(g_idx), int(nbr_global)))
                    if edge_weight is None or not np.isfinite(edge_weight):
                        edge_weight = float(base_dist)
                    geo_weights.append(1.0 / max(edge_weight, 1e-8))
                weights = torch.tensor(geo_weights, device=positions.device, dtype=positions.dtype)

        if weight_mode != "volume" and len(volumes) == len(alive_list):
            vol_tensor = torch.tensor(
                [volumes[global_to_alive[n]] if n in global_to_alive else 0.0 for n in neighbors_global],
                device=positions.device,
                dtype=positions.dtype,
            )
            weights = weights * vol_tensor

        weights = torch.clamp(weights, min=1e-8)
        if max_neighbors and max_neighbors > 0 and weights.numel() > max_neighbors:
            idx = torch.topk(weights, k=max_neighbors, largest=True).indices
            neighbors_tensor = neighbors_tensor[idx]
            weights = weights[idx]

        weights = weights / weights.sum()
        neighbors_out[int(g_idx)] = neighbors_tensor
        weights_out[int(g_idx)] = weights

    return neighbors_out, weights_out


def _compute_weighted_electroweak_ops(
    positions: Tensor,
    velocities: Tensor,
    fitness: Tensor,
    alive: Tensor,
    neighbor_indices: list[Tensor],
    neighbor_weights: list[Tensor],
    h_eff: float,
    epsilon_d: float,
    epsilon_c: float,
    epsilon_clone: float,
    lambda_alg: float,
    bounds,
    pbc: bool,
) -> dict[str, Tensor]:
    N = positions.shape[0]
    device = positions.device
    complex_dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64

    u1_phase_exp = torch.zeros(N, device=device, dtype=complex_dtype)
    u1_phase_q2_exp = torch.zeros_like(u1_phase_exp)
    su2_phase_exp = torch.zeros_like(u1_phase_exp)
    u1_amp = torch.zeros(N, device=device, dtype=positions.dtype)
    su2_amp = torch.zeros_like(u1_amp)

    alive_idx = torch.where(alive)[0].tolist()
    for idx in alive_idx:
        neighbors = neighbor_indices[idx]
        weights = neighbor_weights[idx]
        if neighbors.numel() == 0:
            continue
        w = weights / weights.sum()
        fitness_i = fitness[idx]
        fitness_j = fitness[neighbors]

        phase = -(fitness_j - fitness_i) / h_eff
        phase_exp = torch.exp(1j * phase).to(complex_dtype)
        phase_q2_exp = torch.exp(1j * (2.0 * phase)).to(complex_dtype)
        u1_phase_exp[idx] = (w * phase_exp).sum()
        u1_phase_q2_exp[idx] = (w * phase_q2_exp).sum()

        denom = fitness_i + epsilon_clone
        if torch.is_tensor(denom):
            denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, epsilon_clone), denom)
        scores = (fitness_j - fitness_i) / denom
        su2_phase = scores / h_eff
        su2_phase_exp[idx] = (w * torch.exp(1j * su2_phase).to(complex_dtype)).sum()

        diff_x = positions[idx] - positions[neighbors]
        if pbc and bounds is not None:
            diff_x = _apply_pbc_diff(diff_x, bounds)
        diff_v = velocities[idx] - velocities[neighbors]
        dist_sq = (diff_x**2).sum(dim=-1) + lambda_alg * (diff_v**2).sum(dim=-1)
        u1_weight = torch.exp(-dist_sq / (2.0 * epsilon_d**2))
        su2_weight = torch.exp(-dist_sq / (2.0 * epsilon_c**2))
        u1_amp[idx] = (w * torch.sqrt(u1_weight)).sum()
        su2_amp[idx] = (w * torch.sqrt(su2_weight)).sum()

    su2_comp_phase_exp = torch.zeros_like(su2_phase_exp)
    su2_comp_amp = torch.zeros_like(su2_amp)
    for idx in alive_idx:
        neighbors = neighbor_indices[idx]
        weights = neighbor_weights[idx]
        if neighbors.numel() == 0:
            continue
        w = weights / weights.sum()
        su2_comp_phase_exp[idx] = (w * su2_phase_exp[neighbors]).sum()
        su2_comp_amp[idx] = (w * su2_amp[neighbors]).sum()

    mask = alive.to(u1_phase_exp.dtype)
    u1_amp_c = u1_amp.to(u1_phase_exp.dtype)
    su2_amp_c = su2_amp.to(u1_phase_exp.dtype)
    su2_comp_amp_c = su2_comp_amp.to(u1_phase_exp.dtype)

    operators = {
        "u1_phase": u1_phase_exp * mask,
        "u1_dressed": u1_amp_c * u1_phase_exp * mask,
        "u1_phase_q2": u1_phase_q2_exp * mask,
        "u1_dressed_q2": u1_amp_c * u1_phase_q2_exp * mask,
        "su2_phase": su2_phase_exp * mask,
        "su2_component": su2_amp_c * su2_phase_exp * mask,
        "su2_doublet": (su2_amp_c * su2_phase_exp + su2_comp_amp_c * su2_comp_phase_exp) * mask,
        "su2_doublet_diff": (su2_amp_c * su2_phase_exp - su2_comp_amp_c * su2_comp_phase_exp) * mask,
        "ew_mixed": (u1_amp_c * su2_amp_c) * u1_phase_exp * su2_phase_exp * mask,
    }

    return operators


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


def _compute_neighbor_dist_sq_frame(
    history: RunHistory,
    frame_idx: int,
    neighbor_idx: Tensor,
    lambda_alg: float,
) -> Tensor:
    x_pre = history.x_before_clone[frame_idx]
    v_pre = history.v_before_clone[frame_idx]
    x_comp = x_pre[neighbor_idx]
    v_comp = v_pre[neighbor_idx]
    diff = x_pre - x_comp
    if history.pbc:
        diff = _apply_pbc_diff(diff, history.bounds)
    pos_sq = (diff**2).sum(dim=-1)
    vel_sq = (v_pre - v_comp).pow(2).sum(dim=-1)
    return pos_sq + lambda_alg * vel_sq


def _select_electroweak_neighbors_snapshot(
    history: RunHistory,
    frame_idx: int,
    cfg: ElectroweakChannelConfig,
) -> tuple[Tensor, Tensor]:
    # Handle deprecated "uniform" alias
    neighbor_method = cfg.neighbor_method
    if neighbor_method == "uniform":
        neighbor_method = "companions"

    if neighbor_method == "companions":
        return (
            history.companions_distance[frame_idx - 1],
            history.companions_clone[frame_idx - 1],
        )

    N = history.N
    device = history.x_final.device
    neighbors_distance = torch.arange(N, device=device, dtype=torch.long)
    neighbors_clone = neighbors_distance.clone()

    alive = history.alive_mask[frame_idx - 1]
    x_pre = history.x_before_clone[frame_idx]

    if neighbor_method == "recorded":
        if history.neighbor_edges is None or frame_idx >= len(history.neighbor_edges):
            return (
                history.companions_distance[frame_idx - 1],
                history.companions_clone[frame_idx - 1],
            )
        edges = history.neighbor_edges[frame_idx]
        if not torch.is_tensor(edges) or edges.numel() == 0:
            return neighbors_distance, neighbors_clone
        edge_list = edges.tolist()
        neighbor_map: dict[int, list[int]] = {}
        for i, j in edge_list:
            if i == j:
                continue
            neighbor_map.setdefault(i, []).append(j)
        alive_idx = torch.where(alive)[0].tolist()
        for i_idx in alive_idx:
            neighbors = neighbor_map.get(int(i_idx), [])
            neighbor = int(i_idx) if not neighbors else int(neighbors[0])
            neighbors_distance[i_idx] = neighbor
            neighbors_clone[i_idx] = neighbor
        return neighbors_distance, neighbors_clone

    if neighbor_method == "voronoi":
        try:
            from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation
        except Exception:
            return (
                history.companions_distance[frame_idx - 1],
                history.companions_clone[frame_idx - 1],
            )

        if not alive.any():
            return neighbors_distance, neighbors_clone
        alive_idx = torch.where(alive)[0]
        vor_data = compute_voronoi_tessellation(
            positions=x_pre,
            alive=alive,
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
            neighbors_distance[i_orig] = int(neighbor_orig)
            neighbors_clone[i_orig] = int(neighbor_orig)
        return neighbors_distance, neighbors_clone

    msg = "neighbor_method must be 'companions', 'voronoi', or 'recorded'"
    raise ValueError(msg)


def _select_electroweak_neighbors(
    history: RunHistory,
    start_idx: int,
    cfg: ElectroweakChannelConfig,
) -> tuple[Tensor, Tensor]:
    # Handle deprecated "uniform" alias
    neighbor_method = cfg.neighbor_method
    if neighbor_method == "uniform":
        neighbor_method = "companions"

    if neighbor_method == "companions":
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

    if neighbor_method == "recorded":
        if history.neighbor_edges is None:
            return (
                history.companions_distance[start_idx - 1 : history.n_recorded - 1],
                history.companions_clone[start_idx - 1 : history.n_recorded - 1],
            )
        for t in range(T):
            alive_t = alive[t]
            if not alive_t.any():
                continue
            record_idx = start_idx + t
            if record_idx < 0 or record_idx >= len(history.neighbor_edges):
                continue
            edges = history.neighbor_edges[record_idx]
            if not torch.is_tensor(edges) or edges.numel() == 0:
                continue
            edge_list = edges.tolist()
            neighbor_map: dict[int, list[int]] = {}
            for i, j in edge_list:
                if i == j:
                    continue
                neighbor_map.setdefault(i, []).append(j)
            alive_idx = torch.where(alive_t)[0].tolist()
            for i_idx in alive_idx:
                neighbors = neighbor_map.get(int(i_idx), [])
                neighbor = int(i_idx) if not neighbors else int(neighbors[0])
                neighbors_distance[t, i_idx] = neighbor
                neighbors_clone[t, i_idx] = neighbor
        return neighbors_distance, neighbors_clone

    if (
        neighbor_method == "voronoi"
        and cfg.time_axis == "euclidean"
        and cfg.use_time_sliced_tessellation
    ):
        try:
            from fragile.fractalai.qft.voronoi_time_slices import (
                compute_time_sliced_voronoi,
            )
        except Exception:
            return neighbors_distance, neighbors_clone

        if alive.shape[0] == 0:
            return neighbors_distance, neighbors_clone
        alive_t = alive[0]
        if not alive_t.any():
            return neighbors_distance, neighbors_clone

        time_sliced = compute_time_sliced_voronoi(
            positions=x_pre[0],
            time_dim=int(cfg.euclidean_time_dim),
            n_bins=int(cfg.euclidean_time_bins),
            min_walkers_bin=1,
            bounds=history.bounds,
            alive=alive_t,
            pbc=bool(history.pbc),
            pbc_mode=cfg.voronoi_pbc_mode,
            exclude_boundary=cfg.voronoi_exclude_boundary,
            boundary_tolerance=cfg.voronoi_boundary_tolerance,
            compute_curvature=False,
        )
        edges = _collect_time_sliced_edges(time_sliced, cfg.time_sliced_neighbor_mode)
        neighbor_lists = _build_neighbor_lists(edges, N)
        alive_idx = torch.where(alive_t)[0].tolist()
        alive_set = set(int(i) for i in alive_idx)

        for i in alive_set:
            choices = [j for j in neighbor_lists[i] if j in alive_set and j != i]
            neighbor = i if not choices else choices[0]
            neighbors_distance[0, i] = int(neighbor)
            neighbors_clone[0, i] = int(neighbor)
        return neighbors_distance, neighbors_clone

    if neighbor_method == "voronoi":
        try:
            from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation
        except Exception:
            return (
                history.companions_distance[start_idx - 1 : history.n_recorded - 1],
                history.companions_clone[start_idx - 1 : history.n_recorded - 1],
            )

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

    msg = "neighbor_method must be 'companions', 'voronoi', or 'recorded'"
    raise ValueError(msg)


def _compute_electroweak_series(
    history: RunHistory,
    cfg: ElectroweakChannelConfig,
) -> dict[str, Tensor]:
    start_idx = max(1, int(history.n_recorded * cfg.warmup_fraction))
    if cfg.time_axis == "euclidean":
        start_idx = _resolve_mc_time_index(history, cfg.mc_time_index)
    if start_idx >= history.n_recorded:
        return {}

    h_eff = float(max(cfg.h_eff, 1e-8))
    edge_weight_mode = getattr(cfg, "edge_weight_mode", "inverse_riemannian_distance")
    params = _resolve_electroweak_params(history, cfg)
    epsilon_d = float(params["epsilon_d"])
    epsilon_c = float(params["epsilon_c"])
    epsilon_clone = float(params["epsilon_clone"])
    lambda_alg = float(params["lambda_alg"])

    if cfg.time_axis == "euclidean":
        frame_idx = start_idx
        positions = history.x_before_clone[frame_idx]
        velocities = history.v_before_clone[frame_idx]
        fitness = history.fitness[frame_idx - 1]
        alive = history.alive_mask[frame_idx - 1]

        # Try pre-computed path first
        precomputed = _build_neighbor_data_from_history(
            history, frame_idx, edge_weight_mode, alive,
            max_neighbors=int(cfg.neighbor_k),
        )
        if precomputed is not None:
            neighbor_indices, neighbor_weights = precomputed
        else:
            # Legacy fallback: on-the-fly Voronoi computation
            volume_weights = _get_volume_weights(history, frame_idx)
            neighbor_lists_global, volumes, edge_weight_map, alive_idx = _build_voronoi_neighbor_data(
                history, positions, alive, cfg, volume_weights=volume_weights
            )
            neighbor_indices, neighbor_weights = _build_neighbor_weights(
                positions,
                alive_idx,
                neighbor_lists_global,
                volumes,
                cfg.neighbor_weighting,
                edge_weight_map,
                history.bounds,
                bool(history.pbc),
                int(cfg.neighbor_k),
                kernel_length_scale=cfg.kernel_length_scale,
            )
        operators = _compute_weighted_electroweak_ops(
            positions,
            velocities,
            fitness,
            alive,
            neighbor_indices,
            neighbor_weights,
            h_eff,
            epsilon_d,
            epsilon_c,
            epsilon_clone,
            lambda_alg,
            history.bounds,
            bool(history.pbc),
        )
        alive = alive.unsqueeze(0)
    else:
        T = history.n_recorded - start_idx
        if T <= 0:
            return {}
        operators = {name: [] for name in ELECTROWEAK_CHANNELS}
        alive_series = []
        for frame_idx in range(start_idx, history.n_recorded):
            positions = history.x_before_clone[frame_idx]
            velocities = history.v_before_clone[frame_idx]
            fitness = history.fitness[frame_idx - 1]
            alive = history.alive_mask[frame_idx - 1]

            # Try pre-computed path first
            precomputed = _build_neighbor_data_from_history(
                history, frame_idx, edge_weight_mode, alive,
                max_neighbors=int(cfg.neighbor_k),
            )
            if precomputed is not None:
                neighbor_indices, neighbor_weights = precomputed
            else:
                # Legacy fallback: on-the-fly Voronoi computation
                volume_weights = _get_volume_weights(history, frame_idx)
                neighbor_lists_global, volumes, edge_weight_map, alive_idx = _build_voronoi_neighbor_data(
                    history, positions, alive, cfg, volume_weights=volume_weights
                )
                neighbor_indices, neighbor_weights = _build_neighbor_weights(
                    positions,
                    alive_idx,
                    neighbor_lists_global,
                    volumes,
                    cfg.neighbor_weighting,
                    edge_weight_map,
                    history.bounds,
                    bool(history.pbc),
                    int(cfg.neighbor_k),
                    kernel_length_scale=cfg.kernel_length_scale,
                )
            frame_ops = _compute_weighted_electroweak_ops(
                positions,
                velocities,
                fitness,
                alive,
                neighbor_indices,
                neighbor_weights,
                h_eff,
                epsilon_d,
                epsilon_c,
                epsilon_clone,
                lambda_alg,
                history.bounds,
                bool(history.pbc),
            )
            for name in operators:
                operators[name].append(frame_ops[name])
            alive_series.append(alive)
        operators = {name: torch.stack(values, dim=0) for name, values in operators.items()}
        alive = torch.stack(alive_series, dim=0)

    # Average based on time axis
    if cfg.time_axis == "euclidean":
        # Check dimension
        if history.d < cfg.euclidean_time_dim + 1:
            msg = (
                f"Cannot use dimension {cfg.euclidean_time_dim} as Euclidean time "
                f"(only {history.d} dimensions available)"
            )
            raise ValueError(msg)

        # Get positions for Euclidean time extraction
        positions = history.x_before_clone[start_idx : start_idx + 1]  # [1, N, d]
        alive_slice = alive[:1]

        # Bin each operator by Euclidean time
        series = {}
        for name, op_values in operators.items():
            if op_values.dim() == 1:
                op_values = op_values.unsqueeze(0)
            # For real operators, bin the real part
            if op_values.is_complex():
                # Bin real and imaginary parts separately
                time_coords_real, series_real = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values.real[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
                time_coords_imag, series_imag = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values.imag[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
                series[name] = series_real + 1j * series_imag
            else:
                time_coords, series[name] = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
    else:
        # Monte Carlo time: average over walkers
        series = {
            name: _masked_mean(op_values, alive)
            for name, op_values in operators.items()
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

    real_series = series.real if series.is_complex() else series
    correlator = compute_correlator_fft(
        real_series,
        max_lag=cfg.max_lag,
        use_connected=cfg.use_connected,
    )
    effective_mass = compute_effective_mass_torch(correlator, dt)

    # Compute bootstrap errors if enabled
    correlator_err = None
    if cfg.compute_bootstrap_errors:
        correlator_err = bootstrap_correlator_error(
            real_series,
            max_lag=cfg.max_lag,
            n_bootstrap=cfg.n_bootstrap,
        )

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
        correlator_err=correlator_err,
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


def compute_electroweak_snapshot_operators(
    history: RunHistory,
    config: ElectroweakChannelConfig | None = None,
    channels: list[str] | None = None,
    frame_idx: int | None = None,
) -> dict[str, Tensor]:
    """Compute per-walker electroweak operators at a single MC snapshot."""
    config = config or ElectroweakChannelConfig()
    if channels is None:
        channels = list(ELECTROWEAK_CHANNELS)

    if frame_idx is None:
        frame_idx = _resolve_mc_time_index(history, config.mc_time_index)
    if frame_idx < 1 or frame_idx >= history.n_recorded:
        msg = (
            f"frame_idx {frame_idx} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1})."
        )
        raise ValueError(msg)

    h_eff = float(max(config.h_eff, 1e-8))
    params = _resolve_electroweak_params(history, config)
    epsilon_d = float(params["epsilon_d"])
    epsilon_c = float(params["epsilon_c"])
    epsilon_clone = float(params["epsilon_clone"])
    lambda_alg = float(params["lambda_alg"])

    positions = history.x_before_clone[frame_idx]
    velocities = history.v_before_clone[frame_idx]
    fitness = history.fitness[frame_idx - 1]
    alive = history.alive_mask[frame_idx - 1]

    edge_weight_mode = getattr(config, "edge_weight_mode", "inverse_riemannian_distance")
    precomputed = _build_neighbor_data_from_history(
        history, frame_idx, edge_weight_mode, alive,
        max_neighbors=int(config.neighbor_k),
    )
    if precomputed is not None:
        neighbor_indices, neighbor_weights = precomputed
    else:
        volume_weights = _get_volume_weights(history, frame_idx)
        neighbor_lists_global, volumes, edge_weight_map, alive_idx = _build_voronoi_neighbor_data(
            history, positions, alive, config, volume_weights=volume_weights
        )
        neighbor_indices, neighbor_weights = _build_neighbor_weights(
            positions,
            alive_idx,
            neighbor_lists_global,
            volumes,
            config.neighbor_weighting,
            edge_weight_map,
            history.bounds,
            bool(history.pbc),
            int(config.neighbor_k),
            kernel_length_scale=config.kernel_length_scale,
        )
    operators = _compute_weighted_electroweak_ops(
        positions,
        velocities,
        fitness,
        alive,
        neighbor_indices,
        neighbor_weights,
        h_eff,
        epsilon_d,
        epsilon_c,
        epsilon_clone,
        lambda_alg,
        history.bounds,
        bool(history.pbc),
    )

    return {name: operators[name] for name in channels if name in operators}
