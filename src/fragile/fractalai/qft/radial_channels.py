"""Geometry-aware channel correlators for QFT analysis.

This module supports two analysis axes:
- ``time_axis="mc"``: Monte Carlo-time correlators from geometry-weighted operators.
- ``time_axis="radial"``: single-snapshot screening correlators binned by radial distance.

Radial mode supports:
- 4D radial correlators using Euclidean or graph geodesic distances
- 3D drop-axis correlators averaged across all dropped axes
- Graph geodesics with isotropic (Euclidean edge length) or full (emergent metric) weights

The results reuse ``ChannelCorrelatorResult`` so downstream plotting can mirror
strong-force channel views.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.correlator_channels import (
    bootstrap_correlator_error,
    ChannelCorrelatorResult,
    compute_correlator_fft,
    compute_effective_mass_torch,
    ConvolutionalAICExtractor,
)


DISTANCE_MODES = ("euclidean", "graph_iso", "graph_full")
TIME_AXES = ("mc", "radial")
NEIGHBOR_METHODS = ("recorded",)
RECORDED_EDGE_WEIGHT_MODES = (
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_distance",
    "inverse_riemannian_volume",
    "riemannian_kernel",
    "riemannian_kernel_volume",
)
NEIGHBOR_WEIGHT_MODES = (
    *dict.fromkeys(
        (
            "uniform",
            "volume",
            "euclidean",
            "inv_euclidean",
            "inv_geodesic_iso",
            "inv_geodesic_full",
            "kernel",
            *RECORDED_EDGE_WEIGHT_MODES,
        )
    ),
)


@dataclass
class RadialChannelConfig:
    """Configuration for radial channel correlators."""

    time_axis: str = "mc"  # "mc" (Monte Carlo time) or "radial" (single-slice screening)
    mc_time_index: int | None = None
    warmup_fraction: float = 0.1
    max_lag: int = 80
    use_connected: bool = True
    n_bins: int = 48
    max_pairs: int = 200_000
    distance_mode: str = "graph_full"  # euclidean, graph_iso, graph_full
    neighbor_method: str = "recorded"  # recorded only; reuses simulation Delaunay graph
    neighbor_k: int = 0  # 0 = use all neighbors, >0 = cap neighbor count
    neighbor_weighting: str = "inv_geodesic_full"
    kernel_length_scale: float = 1.0
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    use_volume_weights: bool = True
    apply_power_correction: bool = True
    power_override: float | None = None
    window_widths: list[int] | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")
    random_seed: int | None = None
    drop_axis_average: bool = True
    drop_axes: list[int] | None = None

    # Bootstrap error estimation
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


@dataclass
class RadialChannelOutput:
    """Container for radial channel results."""

    channel_results: dict[str, ChannelCorrelatorResult]
    bin_centers: np.ndarray
    counts: np.ndarray
    pair_count: int
    distance_mode: str
    dimension: int
    dropped_axis: int | None = None


@dataclass
class RadialChannelBundle:
    """Outputs for 4D radial and 3D drop-axis averages."""

    radial_4d: RadialChannelOutput
    radial_3d_avg: RadialChannelOutput | None
    radial_3d_by_axis: dict[int, RadialChannelOutput]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_mc_time_index(history: RunHistory, mc_time_index: int | None) -> int:
    if history.n_recorded < 2:
        msg = "Need at least 2 recorded timesteps for radial analysis."
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


def _apply_pbc_diff(diff: np.ndarray, bounds: Any | None) -> np.ndarray:
    if bounds is None:
        return diff
    high = bounds.high.detach().cpu().numpy() if torch.is_tensor(bounds.high) else np.asarray(bounds.high)
    low = bounds.low.detach().cpu().numpy() if torch.is_tensor(bounds.low) else np.asarray(bounds.low)
    span = high - low
    return diff - span * np.round(diff / span)


def _apply_pbc_diff_torch(diff: Tensor, bounds: Any | None) -> Tensor:
    if bounds is None:
        return diff
    high = bounds.high.to(diff)
    low = bounds.low.to(diff)
    span = high - low
    return diff - span * torch.round(diff / span)


def _slice_bounds(bounds: Any | None, keep_dims: list[int]) -> Any | None:
    if bounds is None:
        return None
    if not hasattr(bounds, "low") or not hasattr(bounds, "high"):
        return bounds
    low = bounds.low[keep_dims]
    high = bounds.high[keep_dims]
    from fragile.fractalai.bounds import TorchBounds

    return TorchBounds(low=low, high=high, shape=low.shape)


def _estimate_ell0(history: RunHistory) -> float:
    mid_idx = history.n_recorded // 2
    if mid_idx == 0:
        return 1.0

    x_pre = history.x_before_clone[mid_idx]
    comp_idx = history.companions_distance[mid_idx - 1]
    alive = history.alive_mask[mid_idx - 1]

    diff = x_pre - x_pre[comp_idx]
    if history.pbc and history.bounds is not None:
        high = history.bounds.high.to(x_pre)
        low = history.bounds.low.to(x_pre)
        span = high - low
        diff = diff - span * torch.round(diff / span)
    dist = torch.linalg.vector_norm(diff, dim=-1)

    if dist.numel() > 0 and alive.any():
        return float(dist[alive].median().item())
    return 1.0


def _build_gamma_matrices(dim: int, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
    gamma: dict[str, Tensor] = {}
    gamma["1"] = torch.eye(dim, device=device, dtype=dtype)

    gamma5_diag = torch.tensor([(-1.0) ** i for i in range(dim)], device=device, dtype=dtype)
    gamma["5"] = gamma5_diag
    gamma["5_matrix"] = torch.diag(gamma5_diag)

    gamma_mu_list = []
    for mu in range(dim):
        gamma_mu = torch.zeros(dim, dim, device=device, dtype=dtype)
        gamma_mu[mu, mu] = 1.0
        if mu > 0:
            gamma_mu[mu, 0] = 0.5j
            gamma_mu[0, mu] = -0.5j
        gamma_mu_list.append(gamma_mu)
    gamma["mu"] = torch.stack(gamma_mu_list, dim=0)

    gamma_5mu_list = []
    for mu in range(dim):
        gamma_5mu = gamma["5_matrix"] @ gamma_mu_list[mu]
        gamma_5mu_list.append(gamma_5mu)
    gamma["5mu"] = torch.stack(gamma_5mu_list, dim=0)

    sigma_list = []
    for mu in range(dim):
        for nu in range(mu + 1, dim):
            sigma = torch.zeros(dim, dim, device=device, dtype=dtype)
            sigma[mu, nu] = 1.0j
            sigma[nu, mu] = -1.0j
            sigma_list.append(sigma)
    if sigma_list:
        gamma["sigma"] = torch.stack(sigma_list, dim=0)
    else:
        gamma["sigma"] = torch.zeros(0, dim, dim, device=device, dtype=dtype)

    return gamma


def _compute_color_states_single(
    history: RunHistory,
    frame_idx: int,
    config: RadialChannelConfig,
    keep_dims: list[int] | None = None,
) -> tuple[Tensor, Tensor]:
    v_pre = history.v_before_clone[frame_idx]
    force_visc = history.force_viscous[frame_idx - 1]

    h_eff = float(max(config.h_eff, 1e-8))
    mass = float(max(config.mass, 1e-8))
    ell0 = config.ell0 if config.ell0 is not None else _estimate_ell0(history)

    phase = (mass * v_pre * ell0) / h_eff
    complex_phase = torch.polar(torch.ones_like(phase), phase.float())

    complex_dtype = torch.complex128 if force_visc.dtype == torch.float64 else torch.complex64
    tilde = force_visc.to(complex_dtype) * complex_phase.to(complex_dtype)
    norm = torch.linalg.vector_norm(tilde, dim=-1, keepdim=True).clamp(min=1e-12)
    color = tilde / norm
    valid = norm.squeeze(-1) > 1e-12

    if keep_dims is None:
        return color, valid

    color = color[:, keep_dims]
    proj_norm = torch.linalg.vector_norm(color, dim=-1, keepdim=True).clamp(min=1e-12)
    color = color / proj_norm
    valid = valid & (proj_norm.squeeze(-1) > 1e-12)
    return color, valid


def _apply_projection(
    channel: str, color_i: Tensor, color_j: Tensor, gamma: dict[str, Tensor]
) -> Tensor:
    if channel == "scalar":
        return (color_i.conj() * color_j).sum(dim=-1).real
    if channel == "pseudoscalar":
        gamma5 = gamma["5"].to(color_i.device)
        return (color_i.conj() * gamma5 * color_j).sum(dim=-1).real
    if channel == "vector":
        gamma_mu = gamma["mu"].to(color_i.device, dtype=color_i.dtype)
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        return result.mean(dim=-1).real
    if channel == "axial_vector":
        gamma_5mu = gamma["5mu"].to(color_i.device, dtype=color_i.dtype)
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_5mu, color_j)
        return result.mean(dim=-1).real
    if channel == "tensor":
        sigma = gamma["sigma"].to(color_i.device, dtype=color_i.dtype)
        if sigma.shape[0] == 0:
            return torch.zeros(color_i.shape[:-1], device=color_i.device)
        result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
        return result.mean(dim=-1).real
    raise ValueError(f"Unsupported channel projection: {channel}")


def _compute_glueball_operator(
    history: RunHistory,
    frame_idx: int,
    alive_idx: Tensor,
    keep_dims: list[int] | None = None,
) -> Tensor:
    force = history.force_viscous[frame_idx - 1]
    if keep_dims is not None:
        force = force[:, keep_dims]
    force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)
    return force_sq[alive_idx]


def _build_edge_weight_map(
    edges: np.ndarray,
    weights: np.ndarray,
    alive_idx: Tensor,
    *,
    symmetric: bool = True,
) -> dict[tuple[int, int], float]:
    if edges.size == 0 or weights.size == 0:
        return {}
    alive_list = alive_idx.tolist()
    edge_map: dict[tuple[int, int], float] = {}
    for (i, j), w in zip(edges, weights):
        if i >= len(alive_list) or j >= len(alive_list):
            continue
        gi = int(alive_list[int(i)])
        gj = int(alive_list[int(j)])
        edge_map[gi, gj] = float(w)
        if symmetric:
            edge_map[gj, gi] = float(w)
    return edge_map


def _build_neighbor_data(
    neighbor_lists_global: dict[int, list[int]] | None,
    alive_idx: Tensor,
    positions: Tensor,
    bounds: Any | None,
    volumes: np.ndarray | None,
    weight_mode: str,
    edge_weight_map: dict[tuple[int, int], float] | None,
    max_neighbors: int,
    pbc: bool,
    kernel_length_scale: float = 1.0,
) -> tuple[list[Tensor], list[Tensor]]:
    if neighbor_lists_global is None:
        msg = "Recorded neighbor lists are required for radial channel analysis."
        raise ValueError(msg)
    alive_list = alive_idx.tolist()
    global_to_alive = {int(g): i for i, g in enumerate(alive_list)}
    n_alive = len(alive_list)
    neighbor_indices: list[Tensor] = []
    neighbor_weights: list[Tensor] = []
    edge_weight_map = edge_weight_map or {}

    for alive_pos, global_idx in enumerate(alive_list):
        neighbors_global = []
        if neighbor_lists_global is not None:
            neighbors_global = neighbor_lists_global.get(int(global_idx), [])

        neighbors_alive: list[int] = []
        neighbors_global_filtered: list[int] = []
        for n in neighbors_global:
            n_int = int(n)
            if n_int not in global_to_alive or n_int == int(global_idx):
                continue
            neighbors_alive.append(global_to_alive[n_int])
            neighbors_global_filtered.append(n_int)

        if not neighbors_alive:
            msg = f"Walker {int(global_idx)} has no recorded alive neighbors."
            raise ValueError(msg)

        if max_neighbors and max_neighbors > 0 and len(neighbors_alive) > max_neighbors:
            neighbors_alive = neighbors_alive[:max_neighbors]
            neighbors_global_filtered = neighbors_global_filtered[:max_neighbors]

        neighbors_tensor = torch.tensor(neighbors_alive, device=positions.device, dtype=torch.long)

        if weight_mode == "uniform":
            weights_tensor = torch.ones(len(neighbors_alive), device=positions.device)
        elif weight_mode == "volume":
            if volumes is None or len(volumes) != n_alive:
                msg = "Volume weighting requires per-alive recorded volume weights."
                raise ValueError(msg)
            weights_tensor = torch.tensor(
                volumes[neighbors_alive],
                device=positions.device,
                dtype=positions.dtype,
            )
        elif weight_mode in {
            "euclidean",
            "inv_euclidean",
            "inv_geodesic_iso",
            "inv_geodesic_full",
            "kernel",
        }:
            pos_i = positions[alive_pos]
            pos_j = positions[neighbors_tensor]
            diff = pos_j - pos_i
            if pbc and bounds is not None:
                diff = _apply_pbc_diff_torch(diff, bounds)
            dist = torch.linalg.vector_norm(diff, dim=-1).clamp(min=1e-8)
            if weight_mode == "kernel":
                weights_tensor = torch.exp(
                    -(dist ** 2) / (2.0 * kernel_length_scale ** 2)
                )
            elif weight_mode == "euclidean":
                weights_tensor = dist
            elif weight_mode == "inv_euclidean":
                weights_tensor = 1.0 / dist
            else:
                geo_weights = []
                for nbr_global, base_dist in zip(
                    neighbors_global_filtered, dist.tolist(), strict=False
                ):
                    edge_weight = edge_weight_map.get(
                        (int(global_idx), int(nbr_global)), None
                    )
                    if (
                        edge_weight is None
                        or not np.isfinite(edge_weight)
                        or float(edge_weight) <= 0
                    ):
                        msg = (
                            "Missing recorded geodesic edge weight for "
                            f"({int(global_idx)} -> {int(nbr_global)})."
                        )
                        raise ValueError(msg)
                    geo_weights.append(1.0 / max(edge_weight, 1e-8))
                if not geo_weights:
                    msg = f"No geometric weights available for walker {int(global_idx)}."
                    raise ValueError(msg)
                weights_tensor = torch.tensor(
                    geo_weights, device=positions.device, dtype=positions.dtype
                )
        elif weight_mode in RECORDED_EDGE_WEIGHT_MODES:
            recorded_weights = []
            for nbr_global in neighbors_global_filtered:
                edge_weight = edge_weight_map.get((int(global_idx), int(nbr_global)), None)
                if edge_weight is None or not np.isfinite(edge_weight):
                    msg = (
                        f"Missing recorded weight mode '{weight_mode}' for "
                        f"edge ({int(global_idx)} -> {int(nbr_global)})."
                    )
                    raise ValueError(msg)
                if float(edge_weight) < 0:
                    msg = (
                        f"Negative recorded weight mode '{weight_mode}' for "
                        f"edge ({int(global_idx)} -> {int(nbr_global)})."
                    )
                    raise ValueError(msg)
                recorded_weights.append(float(edge_weight))
            if not recorded_weights:
                msg = (
                    f"No recorded weights for walker {int(global_idx)} "
                    f"under mode '{weight_mode}'."
                )
                raise ValueError(msg)
            weights_tensor = torch.tensor(
                recorded_weights, device=positions.device, dtype=positions.dtype
            )
        else:
            msg = f"Unsupported neighbor_weighting mode: {weight_mode}"
            raise ValueError(msg)

        if (
            weight_mode != "volume"
            and weight_mode not in RECORDED_EDGE_WEIGHT_MODES
            and volumes is not None
            and len(volumes) == n_alive
        ):
            vol_tensor = torch.tensor(
                volumes[neighbors_alive],
                device=positions.device,
                dtype=positions.dtype,
            )
            weights_tensor = weights_tensor * vol_tensor

        neighbor_indices.append(neighbors_tensor)
        neighbor_weights.append(weights_tensor)

    return neighbor_indices, neighbor_weights


def _build_neighbor_array(
    neighbor_lists: dict[int, list[int]],
    alive_idx: Tensor,
    k: int,
) -> Tensor:
    alive_list = alive_idx.tolist()
    global_to_alive = {int(g): i for i, g in enumerate(alive_list)}
    n_alive = len(alive_list)
    neighbors = torch.full((n_alive, k), -1, dtype=torch.long)
    for alive_pos, global_idx in enumerate(alive_list):
        choices = [
            global_to_alive[c]
            for c in neighbor_lists.get(int(global_idx), [])
            if int(c) in global_to_alive and global_to_alive[int(c)] != alive_pos
        ]
        if not choices:
            neighbors[alive_pos] = alive_pos
            continue
        if len(choices) < k:
            choices = choices + [alive_pos] * (k - len(choices))
        else:
            choices = choices[:k]
        neighbors[alive_pos] = torch.tensor(choices, dtype=torch.long)
    return neighbors


def _sample_pairs(n: int, max_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
        return pairs[:, 0], pairs[:, 1]

    i = rng.integers(0, n, size=max_pairs, dtype=np.int64)
    j = rng.integers(0, n, size=max_pairs, dtype=np.int64)
    mask = i != j
    i = i[mask]
    j = j[mask]
    swap = i > j
    if swap.any():
        i_swap = i[swap].copy()
        i[swap] = j[swap]
        j[swap] = i_swap
    return i, j


def _build_adjacency(edges: np.ndarray, weights: np.ndarray, n: int) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), w in zip(edges, weights):
        if i == j:
            continue
        adjacency[int(i)].append((int(j), float(w)))
    return adjacency


def _dijkstra_targets(
    adjacency: list[list[tuple[int, float]]],
    source: int,
    targets: set[int],
) -> dict[int, float]:
    distances: dict[int, float] = {source: 0.0}
    heap: list[tuple[float, int]] = [(0.0, source)]
    remaining = set(targets)
    remaining.discard(source)
    while heap and remaining:
        dist_u, u = heapq.heappop(heap)
        if dist_u != distances.get(u):
            continue
        if u in remaining:
            remaining.remove(u)
            if not remaining:
                break
        for v, w in adjacency[u]:
            new_dist = dist_u + w
            if new_dist < distances.get(v, float("inf")):
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return {t: distances.get(t, float("inf")) for t in targets}


def _pair_distances_graph(
    adjacency: list[list[tuple[int, float]]],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    pairs_by_source: dict[int, set[int]] = {}
    for i, j in zip(pair_i, pair_j):
        pairs_by_source.setdefault(int(i), set()).add(int(j))

    distances = np.full(pair_i.shape[0], np.inf, dtype=float)
    index_map: dict[tuple[int, int], list[int]] = {}
    for idx, (i, j) in enumerate(zip(pair_i, pair_j)):
        index_map.setdefault((int(i), int(j)), []).append(idx)

    for source, targets in pairs_by_source.items():
        target_dist = _dijkstra_targets(adjacency, source, targets)
        for target, dist in target_dist.items():
            for idx in index_map.get((source, target), []):
                distances[idx] = dist

    return distances


def _pair_distances_euclidean(
    positions: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    bounds: Any | None,
) -> np.ndarray:
    diff = positions[pair_i] - positions[pair_j]
    if bounds is not None:
        diff = _apply_pbc_diff(diff, bounds)
    return np.linalg.norm(diff, axis=1)


def _compute_mass_fit(
    correlator: Tensor,
    config: RadialChannelConfig,
) -> tuple[dict[str, Any], Tensor | None, Tensor | None, list[int], Tensor | None]:
    if correlator.numel() == 0:
        return {"mass": 0.0, "mass_error": float("inf")}, None, None, [], None

    mask = correlator > 0
    if not mask.any():
        return {"mass": 0.0, "mass_error": float("inf")}, None, None, [], None

    log_corr = torch.full_like(correlator, float("nan"))
    log_corr[mask] = torch.log(correlator[mask])
    log_err = torch.ones_like(log_corr) * 0.1

    finite_mask = torch.isfinite(log_corr)
    if not finite_mask.any():
        return {"mass": 0.0, "mass_error": float("inf")}, None, None, [], None

    last_valid = finite_mask.nonzero()[-1].item()
    log_corr = log_corr[: last_valid + 1]
    log_err = log_err[: last_valid + 1]

    extractor = ConvolutionalAICExtractor(
        window_widths=config.window_widths,
        min_mass=config.min_mass,
        max_mass=config.max_mass,
    )
    mass_fit = extractor.fit_all_widths(log_corr, log_err)
    window_masses = mass_fit.pop("window_masses", None)
    window_aic = mass_fit.pop("window_aic", None)
    window_widths = mass_fit.pop("window_widths", None)
    window_r2 = mass_fit.pop("window_r2", None)
    if window_widths is None:
        window_widths = []
    return mass_fit, window_masses, window_aic, window_widths, window_r2


def _build_channel_result(
    correlator: np.ndarray,
    dt: float,
    config: RadialChannelConfig,
    channel_name: str,
    n_samples: int,
    correlator_err: np.ndarray | None = None,
) -> ChannelCorrelatorResult:
    corr_tensor = torch.as_tensor(correlator, dtype=torch.float32)
    effective_mass = compute_effective_mass_torch(corr_tensor, dt)
    mass_fit, window_masses, window_aic, window_widths, window_r2 = _compute_mass_fit(
        corr_tensor, config
    )
    err_tensor = (
        torch.as_tensor(correlator_err, dtype=torch.float32)
        if correlator_err is not None
        else None
    )
    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=corr_tensor,
        correlator_err=err_tensor,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=corr_tensor,
        n_samples=n_samples,
        dt=dt,
        window_masses=window_masses,
        window_aic=window_aic,
        window_widths=window_widths,
        window_r2=window_r2,
    )


def _compute_operator_values(
    channel: str,
    color: Tensor,
    valid: Tensor,
    alive_idx: Tensor,
    neighbor_indices: list[Tensor],
    neighbor_weights: list[Tensor],
    gamma: dict[str, Tensor],
    history: RunHistory,
    frame_idx: int,
    keep_dims: list[int] | None,
) -> Tensor:
    if channel == "glueball":
        return _compute_glueball_operator(history, frame_idx, alive_idx, keep_dims=keep_dims)

    n_alive = color.shape[0]
    outputs = torch.zeros(n_alive, device=color.device)

    if channel == "nucleon" and color.shape[1] < 3:
        return outputs

    for i in range(n_alive):
        neighbors = neighbor_indices[i]
        weights = neighbor_weights[i]
        if neighbors.numel() == 0:
            continue

        if channel == "nucleon":
            if neighbors.numel() < 2:
                continue
            idx_pairs = torch.combinations(neighbors, r=2)
            if idx_pairs.numel() == 0:
                continue
            j = idx_pairs[:, 0]
            k = idx_pairs[:, 1]
            # Baryon proxy uses first 3 components when d > 3.
            color_3 = color[:, :3]
            color_i = color_3[i].unsqueeze(0).expand(j.shape[0], -1)
            color_j = color_3[j]
            color_k = color_3[k]
            matrix = torch.stack([color_i, color_j, color_k], dim=-1)
            det = torch.linalg.det(matrix)
            det = det.real if det.is_complex() else det
            valid_mask = valid[i] & valid[j] & valid[k]
            # Map pair weights by neighbor index
            w_map = {
                int(n.item()): float(w.item())
                for n, w in zip(neighbors, weights, strict=False)
            }
            pair_weights = torch.tensor(
                [
                    w_map[int(jj.item())] * w_map[int(kk.item())]
                    for jj, kk in zip(j, k, strict=False)
                ],
                device=color.device,
                dtype=weights.dtype,
            )
            pair_weights = torch.where(valid_mask, pair_weights, torch.zeros_like(pair_weights))
            denom = pair_weights.sum()
            if denom > 0:
                outputs[i] = (det * pair_weights).sum() / denom
            continue

        color_i = color[i].unsqueeze(0).expand(neighbors.shape[0], -1)
        color_j = color[neighbors]
        op_vals = _apply_projection(channel, color_i, color_j, gamma)
        valid_mask = valid[i] & valid[neighbors] & (neighbors != i)
        weights_eff = torch.where(valid_mask, weights, torch.zeros_like(weights))
        denom = weights_eff.sum()
        if denom > 0:
            outputs[i] = (op_vals * weights_eff).sum() / denom

    return outputs


def _extract_volume_weights_for_frame(
    history: RunHistory,
    frame_idx: int,
    alive_idx: Tensor,
) -> np.ndarray | None:
    """Extract per-alive-walker Riemannian volumes for one frame."""
    volume_history = getattr(history, "riemannian_volume_weights", None)
    if volume_history is None or frame_idx <= 0:
        return None
    if frame_idx - 1 >= len(volume_history):
        return None

    weights_full = volume_history[frame_idx - 1]
    if torch.is_tensor(weights_full):
        weights_full = weights_full.detach().cpu().numpy()
    else:
        weights_full = np.asarray(weights_full)

    alive_np = alive_idx.detach().cpu().numpy()
    if alive_np.size == 0:
        return None
    max_idx = int(alive_np.max())
    if weights_full.size <= max_idx:
        return None
    return np.asarray(weights_full[alive_np], dtype=float)


def _require_recorded_edges_and_geodesic(
    history: RunHistory,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return recorded directed edges and aligned geodesic distances for one frame."""
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None:
        msg = "RunHistory.neighbor_edges is required for radial channel analysis."
        raise ValueError(msg)
    if frame_idx < 0 or frame_idx >= len(neighbor_edges):
        msg = (
            f"Recorded frame {frame_idx} missing in neighbor_edges "
            f"(available 0..{len(neighbor_edges) - 1})."
        )
        raise ValueError(msg)

    edges = neighbor_edges[frame_idx]
    if not torch.is_tensor(edges) or edges.numel() == 0:
        msg = f"neighbor_edges[{frame_idx}] is empty; cannot build radial channels."
        raise ValueError(msg)
    edges_np = edges.detach().cpu().numpy()
    if edges_np.ndim != 2 or edges_np.shape[1] != 2:
        msg = (
            f"neighbor_edges[{frame_idx}] must have shape [E,2], "
            f"got {tuple(edges_np.shape)}."
        )
        raise ValueError(msg)
    edges_np = np.asarray(edges_np, dtype=np.int64)

    geodesic_history = getattr(history, "geodesic_edge_distances", None)
    if geodesic_history is None:
        msg = "RunHistory.geodesic_edge_distances is required for radial channel analysis."
        raise ValueError(msg)
    if frame_idx < 0 or frame_idx >= len(geodesic_history):
        msg = (
            f"Recorded frame {frame_idx} missing in geodesic_edge_distances "
            f"(available 0..{len(geodesic_history) - 1})."
        )
        raise ValueError(msg)
    geodesic = geodesic_history[frame_idx]
    if not torch.is_tensor(geodesic) or geodesic.numel() == 0:
        msg = f"geodesic_edge_distances[{frame_idx}] is empty; cannot build radial channels."
        raise ValueError(msg)
    geodesic_np = np.asarray(geodesic.detach().cpu().numpy(), dtype=float).reshape(-1)
    if geodesic_np.shape[0] != edges_np.shape[0]:
        msg = (
            f"Frame {frame_idx}: edge/geodesic size mismatch "
            f"(E={edges_np.shape[0]}, G={geodesic_np.shape[0]})."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(geodesic_np)):
        msg = f"Frame {frame_idx}: geodesic_edge_distances contain non-finite values."
        raise ValueError(msg)
    return edges_np, geodesic_np


def _recorded_subgraph_for_alive(
    edges_global: np.ndarray,
    geodesic_global: np.ndarray,
    alive_idx: Tensor,
) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    """Project recorded graph/geodesics to alive walkers and local indexing."""
    alive_list = [int(i) for i in alive_idx.tolist()]
    if not alive_list:
        return np.zeros((0, 2), dtype=np.int64), {}, np.zeros((0,), dtype=float)

    alive_set = set(alive_list)
    global_to_local = {g: i for i, g in enumerate(alive_list)}
    neighbor_lists_global: dict[int, list[int]] = {g: [] for g in alive_list}
    local_edge_weights: dict[tuple[int, int], float] = {}

    for (src, dst), geo in zip(edges_global, geodesic_global, strict=False):
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        if src_i not in alive_set or dst_i not in alive_set:
            continue
        weight = float(geo)
        edge_key = (global_to_local[src_i], global_to_local[dst_i])
        if edge_key not in local_edge_weights:
            local_edge_weights[edge_key] = weight
        if dst_i not in neighbor_lists_global[src_i]:
            neighbor_lists_global[src_i].append(dst_i)

    if not local_edge_weights:
        msg = "Recorded neighbor graph has no alive-alive edges for the requested frame."
        raise ValueError(msg)

    missing = [g for g, nbrs in neighbor_lists_global.items() if not nbrs]
    if missing:
        preview = ", ".join(str(m) for m in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        msg = (
            f"Recorded graph has walkers without neighbors in alive set "
            f"({len(missing)} walkers: {preview}{suffix})."
        )
        raise ValueError(msg)

    local_edges = np.asarray(list(local_edge_weights.keys()), dtype=np.int64)
    local_geodesic = np.asarray(list(local_edge_weights.values()), dtype=float)
    return (
        local_edges,
        neighbor_lists_global,
        local_geodesic,
    )


def _extract_recorded_edge_mode_map(
    history: RunHistory,
    frame_idx: int,
    mode: str,
    edges_global: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Extract recorded per-edge weights for a scutoid mode as directed map."""
    edge_weights_history = getattr(history, "edge_weights", None)
    if edge_weights_history is None:
        msg = (
            f"neighbor_weighting='{mode}' requires RunHistory.edge_weights "
            "to be recorded during simulation."
        )
        raise ValueError(msg)
    if frame_idx < 0 or frame_idx >= len(edge_weights_history):
        msg = (
            f"Recorded frame {frame_idx} missing in edge_weights "
            f"(available 0..{len(edge_weights_history) - 1})."
        )
        raise ValueError(msg)

    edge_dict = edge_weights_history[frame_idx]
    if not isinstance(edge_dict, dict):
        msg = f"edge_weights[{frame_idx}] is not a dict."
        raise ValueError(msg)
    if mode not in edge_dict:
        available = ", ".join(sorted(str(k) for k in edge_dict.keys()))
        msg = (
            f"edge_weights[{frame_idx}] does not contain mode '{mode}'. "
            f"Available modes: [{available}]"
        )
        raise ValueError(msg)

    values = edge_dict[mode]
    if not torch.is_tensor(values) or values.numel() == 0:
        msg = f"edge_weights[{frame_idx}]['{mode}'] is empty."
        raise ValueError(msg)
    values_np = np.asarray(values.detach().cpu().numpy(), dtype=float).reshape(-1)
    if values_np.shape[0] != edges_global.shape[0]:
        msg = (
            f"edge_weights[{frame_idx}]['{mode}'] size mismatch with neighbor_edges: "
            f"W={values_np.shape[0]}, E={edges_global.shape[0]}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(values_np)):
        msg = f"edge_weights[{frame_idx}]['{mode}'] contains non-finite values."
        raise ValueError(msg)

    return {
        (int(src), int(dst)): float(weight)
        for (src, dst), weight in zip(edges_global, values_np, strict=False)
    }


def _project_edge_mode_values_to_local_edges(
    edges_local: np.ndarray,
    alive_idx: Tensor,
    edge_mode_map: dict[tuple[int, int], float],
    mode: str,
) -> np.ndarray:
    """Project directed edge-mode map from global ids to local edge array order."""
    alive_list = [int(x) for x in alive_idx.tolist()]
    local_values: list[float] = []
    missing_pairs: list[tuple[int, int]] = []
    for src_local, dst_local in edges_local:
        src_global = int(alive_list[int(src_local)])
        dst_global = int(alive_list[int(dst_local)])
        value = edge_mode_map.get((src_global, dst_global))
        if value is None or not np.isfinite(value):
            missing_pairs.append((src_global, dst_global))
            local_values.append(np.nan)
            continue
        if value < 0:
            msg = (
                f"edge_weights mode '{mode}' produced negative weight "
                f"for edge ({src_global}->{dst_global})."
            )
            raise ValueError(msg)
        local_values.append(float(value))

    if missing_pairs:
        preview = ", ".join(f"{i}->{j}" for i, j in missing_pairs[:8])
        suffix = "..." if len(missing_pairs) > 8 else ""
        msg = (
            f"Missing recorded edge weight mode '{mode}' for projected edges "
            f"({len(missing_pairs)} missing: {preview}{suffix})."
        )
        raise ValueError(msg)
    return np.asarray(local_values, dtype=float)


def _compute_recorded_iso_edge_lengths(
    positions: Tensor,
    edges: np.ndarray,
    bounds: Any | None,
    pbc: bool,
) -> np.ndarray:
    """Compute Euclidean edge lengths on recorded edges only (no graph rebuild)."""
    if edges.size == 0:
        return np.zeros((0,), dtype=float)
    pos = positions.detach().cpu().numpy()
    diff = pos[edges[:, 0]] - pos[edges[:, 1]]
    if pbc and bounds is not None:
        diff = _apply_pbc_diff(diff, bounds)
    lengths = np.linalg.norm(diff, axis=1)
    return np.where(lengths <= 0, 1e-8, lengths)


def _compute_radial_correlator(
    operators: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    distances: np.ndarray,
    bin_edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.asarray(distances)
    pair_i = np.asarray(pair_i)
    pair_j = np.asarray(pair_j)
    min_len = min(distances.shape[0], pair_i.shape[0], pair_j.shape[0])
    if min_len == 0:
        n_bins = len(bin_edges) - 1
        return (
            np.zeros(n_bins, dtype=float),
            np.zeros(n_bins, dtype=float),
            np.zeros(n_bins, dtype=float),
        )
    if distances.shape[0] != min_len or pair_i.shape[0] != min_len or pair_j.shape[0] != min_len:
        distances = distances[:min_len]
        pair_i = pair_i[:min_len]
        pair_j = pair_j[:min_len]
    n_bins = len(bin_edges) - 1
    bins = np.digitize(distances, bin_edges) - 1
    valid = (bins >= 0) & (bins < n_bins) & np.isfinite(distances)
    if weights is None:
        weights = np.ones_like(distances)
    else:
        weights = np.asarray(weights)
        if weights.ndim == 0:
            weights = np.full_like(distances, float(weights))
        elif weights.shape[0] != distances.shape[0]:
            weights = np.ones_like(distances)
    weights = np.where(valid, weights, 0.0)
    bins_safe = np.clip(bins, 0, n_bins - 1)

    op_i = operators[pair_i]
    op_j = operators[pair_j]
    pair_vals = (op_i * np.conjugate(op_j)).real
    pair_vals = np.where(valid, pair_vals * weights, 0.0)

    sum_vals = np.bincount(bins_safe, weights=pair_vals, minlength=n_bins)
    sum_w = np.bincount(bins_safe, weights=weights, minlength=n_bins)
    counts = np.bincount(bins_safe, weights=valid.astype(float), minlength=n_bins)

    correlator = np.zeros(n_bins, dtype=float)
    nonzero = sum_w > 0
    correlator[nonzero] = sum_vals[nonzero] / sum_w[nonzero]
    return correlator, counts, sum_w


def _power_correct(corr: np.ndarray, bin_centers: np.ndarray, power: float) -> np.ndarray:
    corrected = corr.copy()
    if power <= 0:
        return corrected
    scale = np.where(bin_centers > 0, bin_centers**power, 1.0)
    return corrected * scale


def _bootstrap_radial_correlator_error(
    operators: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    distances: np.ndarray,
    bin_edges: np.ndarray,
    weights: np.ndarray | None,
    n_bootstrap: int,
    power: float = 0.0,
    bin_centers: np.ndarray | None = None,
) -> np.ndarray:
    """Compute bootstrap error estimates for radial correlator.

    Resamples pairs with replacement, computes correlators for each bootstrap
    sample, then computes standard deviation.

    Args:
        operators: Operator values for each particle
        pair_i: First particle index of each pair
        pair_j: Second particle index of each pair
        distances: Distance for each pair
        bin_edges: Bin edges for histogram
        weights: Optional weights for each pair
        n_bootstrap: Number of bootstrap resamples
        power: Power correction exponent
        bin_centers: Bin centers for power correction

    Returns:
        Bootstrap std deviation [n_bins]
    """
    n_pairs = len(pair_i)
    if n_pairs == 0:
        n_bins = len(bin_edges) - 1
        return np.zeros(n_bins, dtype=float)

    bootstrap_corrs = []
    rng = np.random.default_rng()

    for _ in range(n_bootstrap):
        # Resample pairs with replacement
        indices = rng.integers(0, n_pairs, size=n_pairs)
        boot_pair_i = pair_i[indices]
        boot_pair_j = pair_j[indices]
        boot_distances = distances[indices]
        boot_weights = weights[indices] if weights is not None else None

        corr, _, _ = _compute_radial_correlator(
            operators,
            boot_pair_i,
            boot_pair_j,
            boot_distances,
            bin_edges,
            weights=boot_weights,
        )

        # Apply power correction if needed
        if power > 0 and bin_centers is not None:
            corr = _power_correct(corr, bin_centers, power)

        bootstrap_corrs.append(corr)

    stacked = np.stack(bootstrap_corrs)  # [n_bootstrap, n_bins]
    return np.std(stacked, axis=0)


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------


def _compute_radial_output(
    history: RunHistory,
    config: RadialChannelConfig,
    channels: list[str],
    positions: Tensor,
    alive_idx: Tensor,
    keep_dims: list[int] | None,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    distances: np.ndarray,
    bin_edges: np.ndarray,
    neighbor_indices: list[Tensor],
    neighbor_weights: list[Tensor],
    volumes: np.ndarray | None,
    operators_override: dict[str, np.ndarray] | None = None,
) -> RadialChannelOutput:
    n_alive = positions.shape[0]
    device = positions.device
    operators_np: dict[str, np.ndarray] = {}
    if operators_override is None:
        gamma = _build_gamma_matrices(positions.shape[1], device, torch.complex128)

        frame_idx = _resolve_mc_time_index(history, config.mc_time_index)
        color_full, valid_full = _compute_color_states_single(
            history,
            frame_idx,
            config,
            keep_dims=keep_dims,
        )
        color = color_full[alive_idx]
        valid = valid_full[alive_idx]

        for channel in channels:
            op = _compute_operator_values(
                channel,
                color,
                valid,
                alive_idx,
                neighbor_indices,
                neighbor_weights,
                gamma,
                history,
                frame_idx,
                keep_dims,
            )
            operators_np[channel] = op.detach().cpu().numpy()
    else:
        for channel in channels:
            values = operators_override.get(channel)
            if values is None:
                operators_np[channel] = np.zeros(n_alive, dtype=float)
                continue
            operators_np[channel] = np.asarray(values)

    weights = None
    if config.use_volume_weights:
        if volumes is None or volumes.size != n_alive:
            msg = (
                "use_volume_weights=True requires per-alive volume weights "
                "for radial correlator binning."
            )
            raise ValueError(msg)
        v = volumes
        weights = v[pair_i] * v[pair_j]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    results: dict[str, ChannelCorrelatorResult] = {}
    counts_out = None
    for channel in channels:
        corr, counts, _sum_w = _compute_radial_correlator(
            operators_np[channel],
            pair_i,
            pair_j,
            distances,
            bin_edges,
            weights=weights,
        )
        if counts_out is None:
            counts_out = counts
        power = 0.0
        if config.apply_power_correction:
            if config.power_override is not None:
                power = float(config.power_override)
            else:
                power = 0.5 * (positions.shape[1] - 1)
        corr_fit = _power_correct(corr, bin_centers, power)

        # Compute bootstrap errors if enabled
        correlator_err = None
        if config.compute_bootstrap_errors:
            correlator_err = _bootstrap_radial_correlator_error(
                operators_np[channel],
                pair_i,
                pair_j,
                distances,
                bin_edges,
                weights,
                n_bootstrap=config.n_bootstrap,
                power=power,
                bin_centers=bin_centers,
            )

        dt = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        results[channel] = _build_channel_result(
            corr_fit,
            dt,
            config,
            channel,
            n_samples=int(np.sum(counts)),
            correlator_err=correlator_err,
        )

    return RadialChannelOutput(
        channel_results=results,
        bin_centers=bin_centers,
        counts=counts_out if counts_out is not None else np.zeros_like(bin_centers),
        pair_count=len(distances),
        distance_mode=config.distance_mode,
        dimension=positions.shape[1],
        dropped_axis=None,
    )


def _compute_distances(
    positions: Tensor,
    bounds: Any | None,
    pbc: bool,
    distance_mode: str,
    edges: np.ndarray | None,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    geodesic_edge_distances: np.ndarray | None = None,
) -> np.ndarray:
    pos_np = positions.detach().cpu().numpy()
    if distance_mode == "euclidean":
        return _pair_distances_euclidean(pos_np, pair_i, pair_j, bounds)

    if edges is None or edges.size == 0:
        msg = "Graph distance modes require recorded neighbor edges for the frame."
        raise ValueError(msg)

    if distance_mode == "graph_iso":
        weights = _compute_recorded_iso_edge_lengths(
            positions=positions,
            edges=edges,
            bounds=bounds,
            pbc=pbc,
        )
    elif distance_mode == "graph_full":
        if geodesic_edge_distances is None:
            msg = "graph_full requires recorded geodesic edge distances."
            raise ValueError(msg)
        weights = np.asarray(geodesic_edge_distances, dtype=float).reshape(-1)
        if weights.shape[0] != edges.shape[0]:
            msg = (
                "graph_full edge/geodesic mismatch: "
                f"E={edges.shape[0]}, G={weights.shape[0]}."
            )
            raise ValueError(msg)
    else:
        msg = f"Unsupported distance_mode: {distance_mode}"
        raise ValueError(msg)

    adjacency = _build_adjacency(edges, weights, positions.shape[0])
    return _pair_distances_graph(adjacency, pair_i, pair_j)


def _compute_mc_time_output(
    history: RunHistory,
    config: RadialChannelConfig,
    channels: list[str],
    keep_dims: list[int] | None = None,
    operators_override: dict[str, np.ndarray | Tensor] | None = None,
) -> RadialChannelOutput:
    """Compute geometry-aware MC-time correlators."""
    if operators_override is not None:
        msg = (
            "time_axis='mc' does not support snapshot operators_override. "
            "Use time_axis='radial' for snapshot-based operators."
        )
        raise ValueError(msg)
    if config.neighbor_method != "recorded":
        msg = (
            "Radial MC-time analysis requires neighbor_method='recorded' "
            "to reuse simulation Delaunay neighbors."
        )
        raise ValueError(msg)

    start_idx = max(1, int(history.n_recorded * float(config.warmup_fraction)))
    if config.mc_time_index is not None:
        start_idx = _resolve_mc_time_index(history, config.mc_time_index)

    frame_indices = list(range(start_idx, history.n_recorded))
    n_frames = len(frame_indices)
    if n_frames == 0:
        empty = np.array([], dtype=float)
        return RadialChannelOutput(
            channel_results={},
            bin_centers=empty,
            counts=empty,
            pair_count=0,
            distance_mode="mc_time",
            dimension=len(keep_dims) if keep_dims is not None else history.d,
            dropped_axis=None,
        )

    device = history.x_before_clone.device
    series_buffers = {
        channel: torch.zeros(n_frames, device=device, dtype=torch.float32)
        for channel in channels
    }
    valid_frames = torch.zeros(n_frames, device=device, dtype=torch.bool)

    for t_idx, frame_idx in enumerate(frame_indices):
        alive_mask_full = history.alive_mask[frame_idx - 1]
        alive_idx = torch.where(alive_mask_full)[0]
        if alive_idx.numel() < 2:
            continue

        positions_full = history.x_before_clone[frame_idx]
        if keep_dims is not None:
            positions_full = positions_full[:, keep_dims]
        positions_alive = positions_full[alive_idx]

        bounds = history.bounds
        if keep_dims is not None:
            bounds = _slice_bounds(bounds, keep_dims)

        volume_weights_alive = _extract_volume_weights_for_frame(history, frame_idx, alive_idx)
        if config.use_volume_weights and volume_weights_alive is None:
            msg = (
                "use_volume_weights=True requires RunHistory.riemannian_volume_weights "
                f"for frame {frame_idx}."
            )
            raise ValueError(msg)
        if config.neighbor_weighting == "volume" and volume_weights_alive is None:
            msg = (
                "neighbor_weighting='volume' requires RunHistory.riemannian_volume_weights "
                f"for frame {frame_idx}."
            )
            raise ValueError(msg)
        volumes = (
            np.asarray(volume_weights_alive, dtype=float)
            if volume_weights_alive is not None
            else np.ones(len(alive_idx), dtype=float)
        )

        edges_global, geodesic_global = _require_recorded_edges_and_geodesic(history, frame_idx)
        edges, neighbor_lists_global, geodesic_edges = _recorded_subgraph_for_alive(
            edges_global,
            geodesic_global,
            alive_idx,
        )

        edge_weight_map: dict[tuple[int, int], float] = {}
        if config.neighbor_weighting == "inv_geodesic_full":
            edge_weight_map = _build_edge_weight_map(edges, geodesic_edges, alive_idx)
        elif config.neighbor_weighting == "inv_geodesic_iso":
            iso_lengths = _compute_recorded_iso_edge_lengths(
                positions=positions_alive,
                edges=edges,
                bounds=bounds,
                pbc=bool(history.pbc),
            )
            edge_weight_map = _build_edge_weight_map(edges, iso_lengths, alive_idx)
        elif config.neighbor_weighting in RECORDED_EDGE_WEIGHT_MODES:
            mode_map = _extract_recorded_edge_mode_map(
                history,
                frame_idx,
                config.neighbor_weighting,
                edges_global,
            )
            mode_local_weights = _project_edge_mode_values_to_local_edges(
                edges,
                alive_idx,
                mode_map,
                config.neighbor_weighting,
            )
            edge_weight_map = _build_edge_weight_map(
                edges,
                mode_local_weights,
                alive_idx,
                symmetric=False,
            )

        neighbor_indices, neighbor_weights = _build_neighbor_data(
            neighbor_lists_global,
            alive_idx,
            positions_alive,
            bounds,
            volumes,
            config.neighbor_weighting,
            edge_weight_map,
            max_neighbors=int(config.neighbor_k),
            pbc=bool(history.pbc),
            kernel_length_scale=config.kernel_length_scale,
        )

        color_full, valid_full = _compute_color_states_single(
            history,
            frame_idx,
            config,
            keep_dims=keep_dims,
        )
        color = color_full[alive_idx]
        valid = valid_full[alive_idx]
        gamma = _build_gamma_matrices(color.shape[1], device, torch.complex128)

        volume_tensor: Tensor | None = None
        if config.use_volume_weights:
            if volumes is None or len(volumes) != len(alive_idx):
                msg = f"Missing per-alive volume weights for frame {frame_idx}."
                raise ValueError(msg)
            volume_tensor = torch.as_tensor(
                volumes,
                device=positions_full.device,
                dtype=torch.float32,
            ).clamp(min=0.0)
            if float(volume_tensor.sum().item()) <= 0:
                msg = f"Non-positive total volume weight at frame {frame_idx}."
                raise ValueError(msg)

        for channel in channels:
            op = _compute_operator_values(
                channel,
                color,
                valid,
                alive_idx,
                neighbor_indices,
                neighbor_weights,
                gamma,
                history,
                frame_idx,
                keep_dims,
            )
            op_real = op.real if op.is_complex() else op
            if volume_tensor is not None and volume_tensor.shape[0] == op_real.shape[0]:
                denom = volume_tensor.sum().clamp(min=1e-12)
                series_buffers[channel][t_idx] = (op_real.float() * volume_tensor).sum() / denom
            else:
                series_buffers[channel][t_idx] = op_real.float().mean()

        valid_frames[t_idx] = True

    valid_indices = torch.where(valid_frames)[0]
    if valid_indices.numel() == 0:
        empty = np.array([], dtype=float)
        return RadialChannelOutput(
            channel_results={},
            bin_centers=empty,
            counts=empty,
            pair_count=0,
            distance_mode="mc_time",
            dimension=len(keep_dims) if keep_dims is not None else history.d,
            dropped_axis=None,
        )

    n_valid = int(valid_indices.numel())
    max_lag = min(int(config.max_lag), n_valid - 1)
    max_lag = max(0, max_lag)
    dt = float(history.delta_t * history.record_every)

    results: dict[str, ChannelCorrelatorResult] = {}
    for channel in channels:
        series = series_buffers[channel][valid_indices]
        corr = compute_correlator_fft(
            series,
            max_lag=max_lag,
            use_connected=bool(config.use_connected),
        )
        correlator_err = None
        if config.compute_bootstrap_errors and series.numel() > 1 and max_lag > 0:
            correlator_err = bootstrap_correlator_error(
                series,
                max_lag=max_lag,
                n_bootstrap=config.n_bootstrap,
                use_connected=bool(config.use_connected),
            )

        results[channel] = _build_channel_result(
            corr.detach().cpu().numpy(),
            dt,
            config,
            channel,
            n_samples=int(series.numel()),
            correlator_err=(
                correlator_err.detach().cpu().numpy()
                if correlator_err is not None
                else None
            ),
        )

    lags = np.arange(max_lag + 1, dtype=float)
    counts = np.arange(n_valid, n_valid - max_lag - 1, -1, dtype=float)
    return RadialChannelOutput(
        channel_results=results,
        bin_centers=lags * dt,
        counts=counts,
        pair_count=0,
        distance_mode="mc_time",
        dimension=len(keep_dims) if keep_dims is not None else history.d,
        dropped_axis=None,
    )


def compute_radial_channels(
    history: RunHistory,
    config: RadialChannelConfig | None = None,
    channels: list[str] | None = None,
    operators_override: dict[str, np.ndarray | Tensor] | None = None,
) -> RadialChannelBundle:
    """Compute geometry-aware channels for MC-time or radial-distance axes."""
    config = config or RadialChannelConfig()
    if config.time_axis not in TIME_AXES:
        raise ValueError(f"time_axis must be one of {TIME_AXES}")
    if config.distance_mode not in DISTANCE_MODES:
        raise ValueError(f"distance_mode must be one of {DISTANCE_MODES}")
    if config.neighbor_method not in NEIGHBOR_METHODS:
        raise ValueError(f"neighbor_method must be one of {NEIGHBOR_METHODS}")
    if config.neighbor_weighting not in NEIGHBOR_WEIGHT_MODES:
        raise ValueError(
            f"neighbor_weighting must be one of {NEIGHBOR_WEIGHT_MODES}"
        )
    if history.neighbor_edges is None:
        msg = "RunHistory.neighbor_edges is required for radial channel analysis."
        raise ValueError(msg)
    if history.geodesic_edge_distances is None:
        msg = "RunHistory.geodesic_edge_distances is required for radial channel analysis."
        raise ValueError(msg)

    channels = channels or [
        "scalar",
        "pseudoscalar",
        "vector",
        "axial_vector",
        "tensor",
        "nucleon",
        "glueball",
    ]

    if config.time_axis == "mc":
        radial_4d = _compute_mc_time_output(
            history,
            config,
            channels,
            keep_dims=None,
            operators_override=operators_override,
        )
        return RadialChannelBundle(
            radial_4d=radial_4d,
            radial_3d_avg=None,
            radial_3d_by_axis={},
        )

    frame_idx = _resolve_mc_time_index(history, config.mc_time_index)
    positions_full = history.x_before_clone[frame_idx]
    alive_mask_full = history.alive_mask[frame_idx - 1]
    alive_idx = torch.where(alive_mask_full)[0]

    if alive_idx.numel() < 2:
        empty_output = RadialChannelOutput(
            channel_results={},
            bin_centers=np.array([]),
            counts=np.array([]),
            pair_count=0,
            distance_mode=config.distance_mode,
            dimension=positions_full.shape[1],
            dropped_axis=None,
        )
        return RadialChannelBundle(
            radial_4d=empty_output,
            radial_3d_avg=None,
            radial_3d_by_axis={},
        )

    positions_alive = positions_full[alive_idx]
    bounds_full = history.bounds

    volume_weights_alive = _extract_volume_weights_for_frame(history, frame_idx, alive_idx)
    if config.use_volume_weights and volume_weights_alive is None:
        msg = (
            "use_volume_weights=True requires RunHistory.riemannian_volume_weights "
            f"for frame {frame_idx}."
        )
        raise ValueError(msg)
    if config.neighbor_weighting == "volume" and volume_weights_alive is None:
        msg = (
            "neighbor_weighting='volume' requires RunHistory.riemannian_volume_weights "
            f"for frame {frame_idx}."
        )
        raise ValueError(msg)
    volumes_full = (
        np.asarray(volume_weights_alive, dtype=float)
        if volume_weights_alive is not None
        else np.ones(len(alive_idx), dtype=float)
    )
    edges_global, geodesic_global = _require_recorded_edges_and_geodesic(history, frame_idx)
    edges_full, neighbor_lists_global, geodesic_full = _recorded_subgraph_for_alive(
        edges_global,
        geodesic_global,
        alive_idx,
    )

    neighbor_indices: list[Tensor]
    neighbor_weights: list[Tensor]
    if operators_override is None:
        edge_weight_map: dict[tuple[int, int], float] = {}
        if config.neighbor_weighting == "inv_geodesic_full":
            edge_weight_map = _build_edge_weight_map(edges_full, geodesic_full, alive_idx)
        elif config.neighbor_weighting == "inv_geodesic_iso":
            iso_lengths = _compute_recorded_iso_edge_lengths(
                positions=positions_alive,
                edges=edges_full,
                bounds=bounds_full,
                pbc=bool(history.pbc),
            )
            edge_weight_map = _build_edge_weight_map(edges_full, iso_lengths, alive_idx)
        elif config.neighbor_weighting in RECORDED_EDGE_WEIGHT_MODES:
            mode_map = _extract_recorded_edge_mode_map(
                history,
                frame_idx,
                config.neighbor_weighting,
                edges_global,
            )
            mode_local_weights = _project_edge_mode_values_to_local_edges(
                edges_full,
                alive_idx,
                mode_map,
                config.neighbor_weighting,
            )
            edge_weight_map = _build_edge_weight_map(
                edges_full,
                mode_local_weights,
                alive_idx,
                symmetric=False,
            )

        neighbor_indices, neighbor_weights = _build_neighbor_data(
            neighbor_lists_global,
            alive_idx,
            positions_alive,
            bounds_full,
            volumes_full,
            config.neighbor_weighting,
            edge_weight_map,
            max_neighbors=int(config.neighbor_k),
            pbc=bool(history.pbc),
            kernel_length_scale=config.kernel_length_scale,
        )
    else:
        empty_idx = torch.zeros(0, dtype=torch.long, device=positions_alive.device)
        empty_w = torch.zeros(0, dtype=positions_alive.dtype, device=positions_alive.device)
        neighbor_indices = [empty_idx for _ in range(len(alive_idx))]
        neighbor_weights = [empty_w for _ in range(len(alive_idx))]

    rng = np.random.default_rng(config.random_seed)
    pair_i, pair_j = _sample_pairs(len(alive_idx), config.max_pairs, rng)

    distances_full = _compute_distances(
        positions_alive,
        bounds_full,
        bool(history.pbc),
        config.distance_mode,
        edges_full,
        pair_i,
        pair_j,
        geodesic_edge_distances=geodesic_full,
    )

    finite_mask = np.isfinite(distances_full)
    if not finite_mask.any():
        msg = "No finite pair distances available from recorded graph for selected pairs."
        raise ValueError(msg)

    dmin = float(np.min(distances_full[finite_mask]))
    dmax = float(np.max(distances_full[finite_mask]))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        dmin = 0.0
        dmax = 1.0
    bin_edges_full = np.linspace(dmin, dmax, config.n_bins + 1)

    operators_alive: dict[str, np.ndarray] | None = None
    if operators_override is not None:
        operators_alive = {}
        for name, values in operators_override.items():
            values_t = values if torch.is_tensor(values) else torch.as_tensor(values)
            if values_t.shape[0] == history.N:
                values_t = values_t[alive_idx]
            elif values_t.shape[0] != len(alive_idx):
                msg = "operators_override must have length N or n_alive."
                raise ValueError(msg)
            operators_alive[name] = values_t.detach().cpu().numpy()

    radial_4d = _compute_radial_output(
        history,
        config,
        channels,
        positions_alive,
        alive_idx,
        keep_dims=None,
        pair_i=pair_i,
        pair_j=pair_j,
        distances=distances_full,
        bin_edges=bin_edges_full,
        neighbor_indices=neighbor_indices,
        neighbor_weights=neighbor_weights,
        volumes=volumes_full,
        operators_override=operators_alive,
    )

    radial_3d_by_axis: dict[int, RadialChannelOutput] = {}
    radial_3d_avg: RadialChannelOutput | None = None

    if config.drop_axis_average and history.d >= 3:
        drop_axes = config.drop_axes or list(range(history.d))
        axis_distances: dict[int, np.ndarray] = {}
        for axis in drop_axes:
            keep_dims = [i for i in range(history.d) if i != axis]
            if len(keep_dims) < 2:
                continue
            positions_proj = positions_full[alive_idx][:, keep_dims]
            bounds_proj = _slice_bounds(bounds_full, keep_dims)
            distances = _compute_distances(
                positions_proj,
                bounds_proj,
                bool(history.pbc),
                config.distance_mode,
                edges_full,
                pair_i,
                pair_j,
                geodesic_edge_distances=geodesic_full,
            )
            finite = np.isfinite(distances)
            if not finite.any():
                msg = f"No finite projected distances for drop axis {axis}."
                raise ValueError(msg)
            axis_distances[axis] = distances

        if axis_distances:
            global_min = min(float(np.min(d[np.isfinite(d)])) for d in axis_distances.values())
            global_max = max(float(np.max(d[np.isfinite(d)])) for d in axis_distances.values())
            if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
                global_min = 0.0
                global_max = 1.0
            bin_edges_3d = np.linspace(global_min, global_max, config.n_bins + 1)

            combined_vals: dict[str, np.ndarray] = {}
            combined_counts = np.zeros(config.n_bins)

            for axis, distances in axis_distances.items():
                keep_dims = [i for i in range(history.d) if i != axis]
                positions_proj = positions_full[alive_idx][:, keep_dims]
                if operators_override is None:
                    edge_weight_map: dict[tuple[int, int], float] = {}
                    if config.neighbor_weighting == "inv_geodesic_full":
                        edge_weight_map = _build_edge_weight_map(
                            edges_full,
                            geodesic_full,
                            alive_idx,
                        )
                    elif config.neighbor_weighting == "inv_geodesic_iso":
                        iso_lengths = _compute_recorded_iso_edge_lengths(
                            positions=positions_proj,
                            edges=edges_full,
                            bounds=_slice_bounds(bounds_full, keep_dims),
                            pbc=bool(history.pbc),
                        )
                        edge_weight_map = _build_edge_weight_map(
                            edges_full,
                            iso_lengths,
                            alive_idx,
                        )
                    elif config.neighbor_weighting in RECORDED_EDGE_WEIGHT_MODES:
                        mode_map = _extract_recorded_edge_mode_map(
                            history,
                            frame_idx,
                            config.neighbor_weighting,
                            edges_global,
                        )
                        mode_local_weights = _project_edge_mode_values_to_local_edges(
                            edges_full,
                            alive_idx,
                            mode_map,
                            config.neighbor_weighting,
                        )
                        edge_weight_map = _build_edge_weight_map(
                            edges_full,
                            mode_local_weights,
                            alive_idx,
                            symmetric=False,
                        )

                    neighbor_indices, neighbor_weights = _build_neighbor_data(
                        neighbor_lists_global,
                        alive_idx,
                        positions_proj,
                        _slice_bounds(bounds_full, keep_dims),
                        volumes_full,
                        config.neighbor_weighting,
                        edge_weight_map,
                        max_neighbors=int(config.neighbor_k),
                        pbc=bool(history.pbc),
                        kernel_length_scale=config.kernel_length_scale,
                    )
                else:
                    empty_idx = torch.zeros(0, dtype=torch.long, device=positions_proj.device)
                    empty_w = torch.zeros(0, dtype=positions_proj.dtype, device=positions_proj.device)
                    neighbor_indices = [empty_idx for _ in range(len(alive_idx))]
                    neighbor_weights = [empty_w for _ in range(len(alive_idx))]
                output = _compute_radial_output(
                    history,
                    config,
                    channels,
                    positions_proj,
                    alive_idx,
                    keep_dims=keep_dims,
                    pair_i=pair_i,
                    pair_j=pair_j,
                    distances=distances,
                    bin_edges=bin_edges_3d,
                    neighbor_indices=neighbor_indices,
                    neighbor_weights=neighbor_weights,
                    volumes=volumes_full,
                    operators_override=operators_alive,
                )
                output.dropped_axis = axis
                radial_3d_by_axis[axis] = output

                for name, result in output.channel_results.items():
                    corr = result.correlator.detach().cpu().numpy()
                    combined_vals.setdefault(name, np.zeros_like(corr))
                    combined_vals[name] += corr * output.counts
                combined_counts += output.counts

            if combined_counts.sum() > 0:
                avg_results: dict[str, ChannelCorrelatorResult] = {}
                for name in combined_vals:
                    avg_corr = np.zeros_like(combined_vals[name])
                    mask = combined_counts > 0
                    avg_corr[mask] = combined_vals[name][mask] / combined_counts[mask]
                    dt = float(bin_edges_3d[1] - bin_edges_3d[0])
                    avg_results[name] = _build_channel_result(
                        avg_corr,
                        dt,
                        config,
                        name,
                        n_samples=int(combined_counts.sum()),
                    )

                radial_3d_avg = RadialChannelOutput(
                    channel_results=avg_results,
                    bin_centers=(bin_edges_3d[:-1] + bin_edges_3d[1:]) / 2.0,
                    counts=combined_counts,
                    pair_count=len(pair_i),
                    distance_mode=config.distance_mode,
                    dimension=history.d - 1,
                    dropped_axis=None,
                )

    return RadialChannelBundle(
        radial_4d=radial_4d,
        radial_3d_avg=radial_3d_avg,
        radial_3d_by_axis=radial_3d_by_axis,
    )
