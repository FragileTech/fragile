"""Axis-free radial channel correlators for QFT analysis.

This module computes screening masses from radial correlators on a single
Monte Carlo snapshot. It supports:
- 4D radial correlators using Euclidean or graph geodesic distances
- 3D drop-axis correlators averaged across all dropped axes
- Graph geodesics with isotropic (Euclidean edge length) or full (emergent metric) weights

The results reuse ChannelCorrelatorResult so downstream plotting can mirror
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
    ChannelCorrelatorResult,
    ConvolutionalAICExtractor,
    compute_effective_mass_torch,
)
from fragile.fractalai.qft.higgs_observables import (
    compute_emergent_metric,
    compute_geodesic_distances,
)
from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation


DISTANCE_MODES = ("euclidean", "graph_iso", "graph_full")
NEIGHBOR_METHODS = ("voronoi", "companions", "recorded")
NEIGHBOR_WEIGHT_MODES = (
    "uniform",
    "volume",
    "euclidean",
    "inv_euclidean",
    "inv_geodesic_iso",
    "inv_geodesic_full",
    "kernel",
)


@dataclass
class RadialChannelConfig:
    """Configuration for radial channel correlators."""

    mc_time_index: int | None = None
    n_bins: int = 48
    max_pairs: int = 200_000
    distance_mode: str = "graph_full"  # euclidean, graph_iso, graph_full
    neighbor_method: str = "voronoi"  # voronoi, companions, recorded
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
        raise ValueError("Need at least 2 recorded timesteps for radial analysis.")
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


def _build_neighbors_from_edges(edges: Any, alive_idx: Tensor) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {int(i): [] for i in alive_idx.tolist()}
    if edges is None:
        return neighbors
    if torch.is_tensor(edges):
        if edges.numel() == 0:
            return neighbors
        edges_np = edges.detach().cpu().numpy()
    else:
        edges_np = np.asarray(edges)
        if edges_np.size == 0:
            return neighbors
    alive_set = set(neighbors.keys())
    for i, j in edges_np:
        if int(i) not in alive_set or int(j) not in alive_set:
            continue
        neighbors[int(i)].append(int(j))
    return neighbors


def _build_companion_neighbor_lists(
    history: RunHistory,
    frame_idx: int,
    alive_idx: Tensor,
) -> dict[int, list[int]]:
    companions_dist = history.companions_distance[frame_idx - 1]
    companions_clone = history.companions_clone[frame_idx - 1]
    neighbors: dict[int, list[int]] = {}
    for idx in alive_idx.tolist():
        companion_list: list[int] = []
        if companions_dist is not None:
            companion_list.append(int(companions_dist[idx]))
        if companions_clone is not None:
            clone_idx = int(companions_clone[idx])
            if clone_idx not in companion_list:
                companion_list.append(clone_idx)
        if not companion_list:
            companion_list = [int(idx)]
        neighbors[int(idx)] = companion_list
    return neighbors


def _map_voronoi_neighbors_to_global(
    neighbor_lists: dict[int, list[int]] | None,
    alive_idx: Tensor,
) -> dict[int, list[int]] | None:
    if not neighbor_lists:
        return None
    alive_list = alive_idx.tolist()
    return {
        int(alive_list[i]): [int(alive_list[j]) for j in neighbors if j < len(alive_list)]
        for i, neighbors in neighbor_lists.items()
        if i < len(alive_list)
    }


def _build_edge_weight_map(
    edges: np.ndarray,
    weights: np.ndarray,
    alive_idx: Tensor,
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
        edge_map[(gi, gj)] = float(w)
        edge_map[(gj, gi)] = float(w)
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
            neighbors_alive = [alive_pos]
            neighbors_global_filtered = [int(global_idx)]

        if max_neighbors and max_neighbors > 0 and len(neighbors_alive) > max_neighbors:
            neighbors_alive = neighbors_alive[:max_neighbors]
            neighbors_global_filtered = neighbors_global_filtered[:max_neighbors]

        neighbors_tensor = torch.tensor(neighbors_alive, device=positions.device, dtype=torch.long)

        if weight_mode == "uniform":
            weights_tensor = torch.ones(len(neighbors_alive), device=positions.device)
        elif weight_mode == "volume" and volumes is not None and len(volumes) == n_alive:
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
                    if edge_weight is None or not np.isfinite(edge_weight):
                        edge_weight = float(base_dist)
                    geo_weights.append(1.0 / max(edge_weight, 1e-8))
                if not geo_weights:
                    weights_tensor = torch.ones(len(neighbors_alive), device=positions.device)
                else:
                    weights_tensor = torch.tensor(
                        geo_weights, device=positions.device, dtype=positions.dtype
                    )
        else:
            weights_tensor = torch.ones(len(neighbors_alive), device=positions.device)

        if weight_mode != "volume" and volumes is not None and len(volumes) == n_alive:
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
    if source in remaining:
        remaining.remove(source)
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

    if channel == "nucleon" and color.shape[1] != 3:
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
            color_i = color[i].unsqueeze(0).expand(j.shape[0], -1)
            color_j = color[j]
            color_k = color[k]
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


def _compute_edges_and_volumes(
    positions: Tensor,
    alive_mask: Tensor,
    history: RunHistory,
    keep_dims: list[int] | None,
    volume_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    bounds = history.bounds
    if keep_dims is not None:
        bounds = _slice_bounds(bounds, keep_dims)
    voronoi = compute_voronoi_tessellation(
        positions=positions,
        alive=alive_mask,
        bounds=bounds,
        pbc=bool(history.pbc),
        pbc_mode="mirror",
        exclude_boundary=True,
        boundary_tolerance=1e-6,
        compute_curvature=False,
        prev_volumes=None,
        dt=float(history.delta_t),
        spatial_dims=positions.shape[1],
    )
    neighbor_lists = voronoi.get("neighbor_lists", {})
    volumes = voronoi.get("volumes", np.ones(len(neighbor_lists), dtype=float))
    if volume_weights is not None and volume_weights.size == len(neighbor_lists):
        volumes = volume_weights
    edges = []
    for i, neighbors in neighbor_lists.items():
        for j in neighbors:
            edges.append((int(i), int(j)))
    edges_array = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    return edges_array, neighbor_lists, volumes


def _compute_graph_weights(
    positions: Tensor,
    edges: np.ndarray,
    alive_mask: Tensor,
    distance_mode: str,
) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=float)

    pos = positions.detach().cpu().numpy()
    if distance_mode == "graph_iso":
        diff = pos[edges[:, 0]] - pos[edges[:, 1]]
        weights = np.linalg.norm(diff, axis=1)
        weights = np.where(weights <= 0, 1e-6, weights)
        return weights

    edge_index = torch.tensor(edges.T, device=positions.device, dtype=torch.long)
    metric = compute_emergent_metric(positions, edge_index, alive_mask)
    geo = compute_geodesic_distances(positions, edge_index, metric, alive_mask)
    return geo.detach().cpu().numpy()


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
    if config.use_volume_weights and volumes is not None and volumes.size == n_alive:
        v = volumes
        weights = v[pair_i] * v[pair_j]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    results: dict[str, ChannelCorrelatorResult] = {}
    counts_out = None
    for channel in channels:
        corr, counts, sum_w = _compute_radial_correlator(
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
        pair_count=int(len(distances)),
        distance_mode=config.distance_mode,
        dimension=positions.shape[1],
        dropped_axis=None,
    )


def _compute_distances(
    positions: Tensor,
    bounds: Any | None,
    distance_mode: str,
    edges: np.ndarray | None,
    alive_mask: Tensor,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    pos_np = positions.detach().cpu().numpy()
    if distance_mode == "euclidean":
        return _pair_distances_euclidean(pos_np, pair_i, pair_j, bounds)

    if edges is None or edges.size == 0:
        return _pair_distances_euclidean(pos_np, pair_i, pair_j, bounds)

    weights = _compute_graph_weights(positions, edges, alive_mask, distance_mode)
    adjacency = _build_adjacency(edges, weights, positions.shape[0])
    return _pair_distances_graph(adjacency, pair_i, pair_j)


def compute_radial_channels(
    history: RunHistory,
    config: RadialChannelConfig | None = None,
    channels: list[str] | None = None,
    operators_override: dict[str, np.ndarray | Tensor] | None = None,
) -> RadialChannelBundle:
    """Compute radial channel correlators for 4D and 3D drop-axis projections."""
    config = config or RadialChannelConfig()
    if config.distance_mode not in DISTANCE_MODES:
        raise ValueError(f"distance_mode must be one of {DISTANCE_MODES}")
    if config.neighbor_method not in NEIGHBOR_METHODS:
        raise ValueError(f"neighbor_method must be one of {NEIGHBOR_METHODS}")
    if config.neighbor_weighting not in NEIGHBOR_WEIGHT_MODES:
        raise ValueError(
            f"neighbor_weighting must be one of {NEIGHBOR_WEIGHT_MODES}"
        )

    channels = channels or ["scalar", "pseudoscalar", "vector", "nucleon", "glueball"]

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
    alive_mask_local = torch.ones(
        len(alive_idx), dtype=torch.bool, device=positions_full.device
    )

    volume_weights_alive = None
    if getattr(history, "riemannian_volume_weights", None) is not None:
        if frame_idx > 0 and frame_idx - 1 < len(history.riemannian_volume_weights):
            weights_full = history.riemannian_volume_weights[frame_idx - 1]
            if torch.is_tensor(weights_full):
                weights_full = weights_full.detach().cpu().numpy()
            else:
                weights_full = np.asarray(weights_full)
            alive_np = alive_idx.detach().cpu().numpy()
            if alive_np.size > 0:
                max_idx = int(alive_np.max())
                if weights_full.size > max_idx:
                    volume_weights_alive = weights_full[alive_np]

    edges_full, neighbor_lists_full, volumes_full = _compute_edges_and_volumes(
        positions_alive,
        alive_mask_local,
        history,
        keep_dims=None,
        volume_weights=volume_weights_alive,
    )

    companion_neighbors = _build_companion_neighbor_lists(history, frame_idx, alive_idx)
    recorded_neighbors: dict[int, list[int]] | None = None
    if history.neighbor_edges is not None and frame_idx < len(history.neighbor_edges):
        recorded_neighbors = _build_neighbors_from_edges(
            history.neighbor_edges[frame_idx], alive_idx
        )
        if recorded_neighbors and not any(recorded_neighbors.values()):
            recorded_neighbors = None

    voronoi_neighbors_global = _map_voronoi_neighbors_to_global(
        neighbor_lists_full, alive_idx
    )

    neighbor_indices: list[Tensor]
    neighbor_weights: list[Tensor]
    if operators_override is None:
        if config.neighbor_method == "voronoi":
            neighbor_lists_global = voronoi_neighbors_global
        elif config.neighbor_method == "recorded":
            neighbor_lists_global = recorded_neighbors or companion_neighbors
        else:
            neighbor_lists_global = companion_neighbors

        edge_weight_map = {}
        if config.neighbor_weighting in {"inv_geodesic_iso", "inv_geodesic_full"}:
            mode = (
                "graph_iso"
                if config.neighbor_weighting == "inv_geodesic_iso"
                else "graph_full"
            )
            edge_weights = _compute_graph_weights(
                positions_alive, edges_full, alive_mask_local, mode
            )
            edge_weight_map = _build_edge_weight_map(edges_full, edge_weights, alive_idx)

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
        config.distance_mode,
        edges_full,
        alive_mask_local,
        pair_i,
        pair_j,
    )

    finite_mask = np.isfinite(distances_full)
    if not finite_mask.any():
        distances_full = _pair_distances_euclidean(
            positions_alive.detach().cpu().numpy(), pair_i, pair_j, bounds_full
        )

    dmin = float(np.nanmin(distances_full))
    dmax = float(np.nanmax(distances_full))
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
                raise ValueError("operators_override must have length N or n_alive.")
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
            edges, neighbor_lists, volumes = _compute_edges_and_volumes(
                positions_proj,
                alive_mask_local,
                history,
                keep_dims=keep_dims,
                volume_weights=volume_weights_alive,
            )
            distances = _compute_distances(
                positions_proj,
                bounds_proj,
                config.distance_mode,
                edges,
                alive_mask_local,
                pair_i,
                pair_j,
            )
            axis_distances[axis] = distances

        if axis_distances:
            global_min = min(float(np.nanmin(d)) for d in axis_distances.values())
            global_max = max(float(np.nanmax(d)) for d in axis_distances.values())
            if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
                global_min = 0.0
                global_max = 1.0
            bin_edges_3d = np.linspace(global_min, global_max, config.n_bins + 1)

            combined_vals: dict[str, np.ndarray] = {}
            combined_counts = np.zeros(config.n_bins)

            for axis, distances in axis_distances.items():
                keep_dims = [i for i in range(history.d) if i != axis]
                positions_proj = positions_full[alive_idx][:, keep_dims]
                edges, neighbor_lists, volumes = _compute_edges_and_volumes(
                    positions_proj,
                    alive_mask_local,
                    history,
                    keep_dims=keep_dims,
                    volume_weights=volume_weights_alive,
                )
                if operators_override is None:
                    if config.neighbor_method == "voronoi":
                        neighbor_lists_global = _map_voronoi_neighbors_to_global(
                            neighbor_lists, alive_idx
                        )
                    elif config.neighbor_method == "recorded":
                        neighbor_lists_global = recorded_neighbors or companion_neighbors
                    else:
                        neighbor_lists_global = companion_neighbors

                    edge_weight_map = {}
                    if config.neighbor_weighting in {"inv_geodesic_iso", "inv_geodesic_full"}:
                        mode = (
                            "graph_iso"
                            if config.neighbor_weighting == "inv_geodesic_iso"
                            else "graph_full"
                        )
                        edge_weights = _compute_graph_weights(
                            positions_proj, edges, alive_mask_local, mode
                        )
                        edge_weight_map = _build_edge_weight_map(edges, edge_weights, alive_idx)

                    neighbor_indices, neighbor_weights = _build_neighbor_data(
                        neighbor_lists_global,
                        alive_idx,
                        positions_proj,
                        _slice_bounds(bounds_full, keep_dims),
                        volumes,
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
                    volumes=volumes,
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
                    pair_count=int(len(pair_i)),
                    distance_mode=config.distance_mode,
                    dimension=history.d - 1,
                    dropped_axis=None,
                )

    return RadialChannelBundle(
        radial_4d=radial_4d,
        radial_3d_avg=radial_3d_avg,
        radial_3d_by_axis=radial_3d_by_axis,
    )
