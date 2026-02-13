"""Anisotropic edge-channel correlators using recorded Delaunay topology.

This module computes MC-time correlators from edge-local observables built on
the simulation-recorded neighbor graph. It never recomputes neighbor topology.

Workflow per frame:
1. Reuse recorded directed Delaunay edges (restricted to alive walkers).
2. Compute channel edge observables on direct neighbors only.
3. Build anisotropic moments by projecting edge observables with directional
   basis functions (axis-square or quadrupole).
4. Correlate each moment along Monte Carlo time with FFT-based correlators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    ChannelCorrelatorResult,
    compute_channel_correlator,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.radial_channels import (
    _apply_pbc_diff_torch,
    _apply_projection,
    _build_gamma_matrices,
    _compute_color_states_batch,
    _compute_recorded_iso_edge_lengths,
    _extract_recorded_edge_mode_values,
    _recorded_subgraph_for_alive,
    _resolve_mc_time_index,
    _slice_bounds,
    RECORDED_EDGE_WEIGHT_MODES,
)


EDGE_WEIGHT_MODES = (
    "uniform",
    "inv_geodesic_iso",
    "inv_geodesic_full",
    *RECORDED_EDGE_WEIGHT_MODES,
)
NUCLEON_TRIPLET_MODES = ("direct_neighbors", "companions")
COMPONENT_MODES = (
    "isotropic",
    "axes",
    "isotropic+axes",
    "quadrupole",
    "isotropic+quadrupole",
)
SUPPORTED_CHANNELS = {
    "scalar",
    "pseudoscalar",
    "vector",
    "axial_vector",
    "tensor",
    "tensor_traceless",
    "nucleon",
    "glueball",
}
PACK_CHUNK_FRAMES_EDGE = 64
PACK_CHUNK_FRAMES_NUCLEON_DIRECT = 16


@dataclass
class AnisotropicEdgeChannelConfig:
    """Configuration for anisotropic edge-channel correlator analysis."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    mc_time_index: int | None = None
    max_lag: int = 80
    use_connected: bool = True
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    keep_dims: list[int] | None = None
    edge_weight_mode: str = "riemannian_kernel_volume"
    use_volume_weights: bool = True
    component_mode: str = "isotropic+axes"
    nucleon_triplet_mode: str = "direct_neighbors"
    window_widths: list[int] | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")
    fit_mode: str = "aic"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


@dataclass
class AnisotropicEdgeChannelOutput:
    """Computed anisotropic edge correlators and diagnostics."""

    channel_results: dict[str, ChannelCorrelatorResult]
    component_labels: list[str]
    frame_indices: list[int]
    n_valid_frames: int
    avg_alive_walkers: float
    avg_edges: float


def _resolve_keep_dims(history: RunHistory, keep_dims: list[int] | None) -> list[int]:
    if keep_dims is None:
        return list(range(history.d))
    dims = sorted({int(dim) for dim in keep_dims})
    if not dims:
        msg = "keep_dims cannot be empty."
        raise ValueError(msg)
    invalid = [dim for dim in dims if dim < 0 or dim >= history.d]
    if invalid:
        raise ValueError(
            f"keep_dims contains invalid indices {invalid}; valid range is [0, {history.d - 1}]."
        )
    return dims


def _component_labels(dim: int, mode: str) -> list[str]:
    labels: list[str] = []
    if mode in {"isotropic", "isotropic+axes", "isotropic+quadrupole"}:
        labels.append("iso")
    if mode in {"axes", "isotropic+axes"}:
        labels.extend([f"axis_{axis}" for axis in range(dim)])
    if mode in {"quadrupole", "isotropic+quadrupole"}:
        labels.extend([f"q{axis}{axis}" for axis in range(dim)])
        for axis_i in range(dim):
            for axis_j in range(axis_i + 1, dim):
                labels.append(f"q{axis_i}{axis_j}")
    return labels


def _component_factors(direction: Tensor, mode: str) -> dict[str, Tensor]:
    dim = int(direction.shape[1])
    factors: dict[str, Tensor] = {}
    if mode in {"isotropic", "isotropic+axes", "isotropic+quadrupole"}:
        factors["iso"] = torch.ones(
            direction.shape[0], device=direction.device, dtype=direction.dtype
        )
    if mode in {"axes", "isotropic+axes"}:
        for axis in range(dim):
            factors[f"axis_{axis}"] = direction[:, axis].pow(2)
    if mode in {"quadrupole", "isotropic+quadrupole"}:
        inv_dim = 1.0 / float(dim)
        for axis in range(dim):
            factors[f"q{axis}{axis}"] = direction[:, axis].pow(2) - inv_dim
        for axis_i in range(dim):
            for axis_j in range(axis_i + 1, dim):
                factors[f"q{axis_i}{axis_j}"] = direction[:, axis_i] * direction[:, axis_j]
    return factors


def _channel_component_name(channel: str, component: str) -> str:
    if component == "iso":
        return channel
    return f"{channel}:{component}"


def _extract_edges_for_frame(
    history: RunHistory,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None:
        msg = "RunHistory.neighbor_edges is required for anisotropic edge analysis."
        raise ValueError(msg)
    if frame_idx < 0 or frame_idx >= len(neighbor_edges):
        raise ValueError(
            f"Recorded frame {frame_idx} missing in neighbor_edges (available 0..{len(neighbor_edges) - 1})."
        )
    edges = neighbor_edges[frame_idx]
    if not torch.is_tensor(edges) or edges.numel() == 0:
        raise ValueError(f"neighbor_edges[{frame_idx}] is empty.")
    edges_np = np.asarray(edges.detach().cpu().numpy(), dtype=np.int64)
    if edges_np.ndim != 2 or edges_np.shape[1] != 2:
        raise ValueError(
            f"neighbor_edges[{frame_idx}] must have shape [E,2], got {edges_np.shape}."
        )

    geodesic_history = getattr(history, "geodesic_edge_distances", None)
    if geodesic_history is None or frame_idx >= len(geodesic_history):
        return edges_np, np.ones(edges_np.shape[0], dtype=float)
    geodesic = geodesic_history[frame_idx]
    if not torch.is_tensor(geodesic) or geodesic.numel() == 0:
        return edges_np, np.ones(edges_np.shape[0], dtype=float)
    geodesic_np = np.asarray(geodesic.detach().cpu().numpy(), dtype=float).reshape(-1)
    if geodesic_np.shape[0] != edges_np.shape[0]:
        raise ValueError(
            f"Frame {frame_idx}: edge/geodesic mismatch (E={edges_np.shape[0]}, G={geodesic_np.shape[0]})."
        )
    if not np.all(np.isfinite(geodesic_np)):
        raise ValueError(f"Frame {frame_idx}: geodesic distances contain non-finite values.")
    return edges_np, geodesic_np


def _extract_volume_weights_alive(
    history: RunHistory,
    frame_idx: int,
    alive_idx: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    volume_history = getattr(history, "riemannian_volume_weights", None)
    if volume_history is None:
        msg = "use_volume_weights=True requires RunHistory.riemannian_volume_weights."
        raise ValueError(msg)
    info_idx = frame_idx - 1
    if info_idx < 0 or info_idx >= len(volume_history):
        raise ValueError(
            f"Frame {frame_idx}: riemannian_volume_weights missing at index {info_idx}."
        )
    vol_full = volume_history[info_idx]
    if not torch.is_tensor(vol_full):
        vol_full = torch.as_tensor(vol_full)
    vol_alive = vol_full.to(device=device, dtype=dtype)[alive_idx]
    vol_alive = torch.clamp(vol_alive, min=0.0)
    if float(vol_alive.sum().item()) <= 0:
        raise ValueError(f"Frame {frame_idx}: non-positive total alive volume weight.")
    return vol_alive


def _resolve_edge_weights(
    history: RunHistory,
    config: AnisotropicEdgeChannelConfig,
    frame_idx: int,
    edges_global: np.ndarray,
    edges_local: np.ndarray,
    geodesic_local: np.ndarray,
    local_edge_indices: np.ndarray,
    positions_alive: Tensor,
    bounds: Any | None,
) -> Tensor:
    mode = str(config.edge_weight_mode)
    device = positions_alive.device
    dtype = positions_alive.dtype

    if mode == "uniform":
        return torch.ones(edges_local.shape[0], device=device, dtype=dtype)

    if mode == "inv_geodesic_full":
        geodesic_history = getattr(history, "geodesic_edge_distances", None)
        if geodesic_history is None or frame_idx >= len(geodesic_history):
            msg = "inv_geodesic_full requires RunHistory.geodesic_edge_distances."
            raise ValueError(msg)
        geodesic_frame = geodesic_history[frame_idx]
        if not torch.is_tensor(geodesic_frame) or geodesic_frame.numel() == 0:
            msg = f"inv_geodesic_full requires geodesic_edge_distances at frame {frame_idx}."
            raise ValueError(msg)
        geo = torch.as_tensor(geodesic_local, device=device, dtype=dtype)
        if not torch.isfinite(geo).all() or torch.any(geo <= 0):
            msg = "inv_geodesic_full requires positive finite geodesic edge distances."
            raise ValueError(msg)
        return 1.0 / geo.clamp(min=1e-8)

    if mode == "inv_geodesic_iso":
        iso = _compute_recorded_iso_edge_lengths(
            positions=positions_alive,
            edges=edges_local,
            bounds=bounds,
            pbc=bool(history.pbc),
        )
        iso_t = torch.as_tensor(iso, device=device, dtype=dtype)
        if not torch.isfinite(iso_t).all() or torch.any(iso_t <= 0):
            msg = "inv_geodesic_iso requires positive finite Euclidean edge lengths."
            raise ValueError(msg)
        return 1.0 / iso_t.clamp(min=1e-8)

    if mode in RECORDED_EDGE_WEIGHT_MODES:
        values_global = _extract_recorded_edge_mode_values(
            history,
            frame_idx,
            mode,
            n_edges=edges_global.shape[0],
        )
        values_local = values_global[local_edge_indices]
        weights = torch.as_tensor(values_local, device=device, dtype=dtype)
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError(f"Recorded edge mode '{mode}' contains invalid values.")
        return weights

    raise ValueError(f"Unsupported edge_weight_mode: {mode}.")


def _compute_nucleon_triplet_values(
    color_alive: Tensor,
    valid_alive: Tensor,
    src: Tensor,
    neigh_j: Tensor,
    neigh_k: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute trilinear nucleon observable over triplets (i,j,k)."""
    if color_alive.shape[1] < 3:
        zeros = torch.zeros(src.shape[0], device=color_alive.device, dtype=torch.float32)
        valid = torch.zeros(src.shape[0], device=color_alive.device, dtype=torch.bool)
        return zeros, valid

    color_3 = color_alive[:, :3]
    color_i = color_3[src]
    color_j = color_3[neigh_j]
    color_k = color_3[neigh_k]

    matrix = torch.stack([color_i, color_j, color_k], dim=-1)
    det = torch.linalg.det(matrix)
    det = det.real if det.is_complex() else det

    valid = (
        (src != neigh_j)
        & (src != neigh_k)
        & (neigh_j != neigh_k)
        & valid_alive[src]
        & valid_alive[neigh_j]
        & valid_alive[neigh_k]
        & torch.isfinite(det)
    )
    return det.float(), valid


def _build_direct_neighbor_triplets(
    src: Tensor,
    dst: Tensor,
    edge_weights: Tensor,
    diff: Tensor,
    n_alive: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build (i,j,k) triplets from direct recorded outgoing neighbors of each i."""
    device = src.device
    dim = int(diff.shape[1])
    if src.numel() == 0 or n_alive <= 0:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    order = torch.argsort(src, stable=True)
    src_sorted = src[order]
    dst_sorted = dst[order]
    weights_sorted = edge_weights[order].float()
    diff_sorted = diff[order].float()

    counts = torch.bincount(src_sorted, minlength=n_alive)
    k = int(counts.max().item())
    if k < 2:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    row_starts = torch.zeros(n_alive, dtype=torch.long, device=device)
    if n_alive > 1:
        row_starts[1:] = torch.cumsum(counts[:-1], dim=0)
    col = (
        torch.arange(src_sorted.shape[0], device=device, dtype=torch.long) - row_starts[src_sorted]
    )

    neighbor_idx = torch.full((n_alive, k), -1, dtype=torch.long, device=device)
    neighbor_weights = torch.zeros((n_alive, k), dtype=torch.float32, device=device)
    neighbor_diff = torch.zeros((n_alive, k, dim), dtype=torch.float32, device=device)
    neighbor_mask = torch.zeros((n_alive, k), dtype=torch.bool, device=device)

    neighbor_idx[src_sorted, col] = dst_sorted
    neighbor_weights[src_sorted, col] = weights_sorted
    neighbor_diff[src_sorted, col] = diff_sorted
    neighbor_mask[src_sorted, col] = True

    pair_idx = torch.triu_indices(k, k, offset=1, device=device)
    if pair_idx.numel() == 0:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    j_idx = neighbor_idx[:, pair_idx[0]]
    k_idx = neighbor_idx[:, pair_idx[1]]
    pair_weights = neighbor_weights[:, pair_idx[0]] * neighbor_weights[:, pair_idx[1]]
    pair_dir = neighbor_diff[:, pair_idx[0], :] + neighbor_diff[:, pair_idx[1], :]
    valid_pairs = (
        neighbor_mask[:, pair_idx[0]]
        & neighbor_mask[:, pair_idx[1]]
        & (j_idx >= 0)
        & (k_idx >= 0)
        & (j_idx != k_idx)
    )
    if not torch.any(valid_pairs):
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    src_grid = torch.arange(n_alive, device=device, dtype=torch.long).unsqueeze(1).expand_as(j_idx)
    return (
        src_grid[valid_pairs],
        j_idx[valid_pairs],
        k_idx[valid_pairs],
        pair_weights[valid_pairs],
        pair_dir[valid_pairs],
    )


def _build_companion_triplets(
    history: RunHistory,
    frame_idx: int,
    alive_idx: Tensor,
    positions_alive: Tensor,
    bounds: Any | None,
    src: Tensor,
    dst: Tensor,
    edge_weights: Tensor,
    volume_weights_alive: Tensor | None,
    edge_weight_mode: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build one (i,j,k) triplet per alive walker from companion choices."""
    device = positions_alive.device
    dim = int(positions_alive.shape[1])
    n_alive = int(alive_idx.numel())
    if n_alive == 0:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    info_idx = frame_idx - 1
    if info_idx < 0:
        msg = "Companion triplets require frame_idx >= 1."
        raise ValueError(msg)
    companions_dist = getattr(history, "companions_distance", None)
    companions_clone = getattr(history, "companions_clone", None)
    if companions_dist is None or companions_clone is None:
        msg = (
            "nucleon_triplet_mode='companions' requires companions_distance and companions_clone."
        )
        raise ValueError(msg)
    if info_idx >= companions_dist.shape[0] or info_idx >= companions_clone.shape[0]:
        msg = f"Companion arrays missing data for frame {frame_idx} (info index {info_idx})."
        raise ValueError(msg)

    comp_dist_full = companions_dist[info_idx]
    comp_clone_full = companions_clone[info_idx]
    if not torch.is_tensor(comp_dist_full):
        comp_dist_full = torch.as_tensor(comp_dist_full)
    if not torch.is_tensor(comp_clone_full):
        comp_clone_full = torch.as_tensor(comp_clone_full)
    comp_dist_full = comp_dist_full.to(device=device, dtype=torch.long)
    comp_clone_full = comp_clone_full.to(device=device, dtype=torch.long)

    src_local = torch.arange(n_alive, device=device, dtype=torch.long)
    map_size = int(max(int(alive_idx.max().item()), int(history.N) - 1) + 1)
    global_to_local = torch.full((map_size,), -1, device=device, dtype=torch.long)
    global_to_local[alive_idx.to(device=device, dtype=torch.long)] = src_local

    comp_dist_global = comp_dist_full[alive_idx].clamp(min=0)
    comp_clone_global = comp_clone_full[alive_idx].clamp(min=0)
    in_range_dist = comp_dist_global < map_size
    in_range_clone = comp_clone_global < map_size
    neigh_j = torch.full_like(src_local, -1)
    neigh_k = torch.full_like(src_local, -1)
    neigh_j[in_range_dist] = global_to_local[comp_dist_global[in_range_dist]]
    neigh_k[in_range_clone] = global_to_local[comp_clone_global[in_range_clone]]

    valid = (
        (neigh_j >= 0)
        & (neigh_k >= 0)
        & (neigh_j != src_local)
        & (neigh_k != src_local)
        & (neigh_j != neigh_k)
    )
    if not torch.any(valid):
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    src_kept = src_local[valid]
    neigh_j_kept = neigh_j[valid]
    neigh_k_kept = neigh_k[valid]

    if edge_weight_mode == "uniform":
        weights = torch.ones(src_kept.shape[0], device=device, dtype=torch.float32)
        if volume_weights_alive is not None:
            vi = volume_weights_alive[src_kept]
            vj = volume_weights_alive[neigh_j_kept]
            vk = volume_weights_alive[neigh_k_kept]
            weights = weights * torch.sqrt(vi * vj).float() * torch.sqrt(vi * vk).float()
    else:
        edge_lookup = torch.zeros(n_alive, n_alive, device=device, dtype=torch.float32)
        edge_lookup[src, dst] = edge_weights.float()
        w_ij = edge_lookup[src_kept, neigh_j_kept]
        w_ik = edge_lookup[src_kept, neigh_k_kept]
        positive = (w_ij > 0) & (w_ik > 0) & torch.isfinite(w_ij) & torch.isfinite(w_ik)
        if not torch.any(positive):
            empty_long = torch.empty(0, device=device, dtype=torch.long)
            empty_w = torch.empty(0, device=device, dtype=torch.float32)
            empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
            return empty_long, empty_long, empty_long, empty_w, empty_dir
        src_kept = src_kept[positive]
        neigh_j_kept = neigh_j_kept[positive]
        neigh_k_kept = neigh_k_kept[positive]
        weights = (w_ij[positive] * w_ik[positive]).float()

    diff_ij = positions_alive[neigh_j_kept] - positions_alive[src_kept]
    diff_ik = positions_alive[neigh_k_kept] - positions_alive[src_kept]
    if bool(history.pbc) and bounds is not None:
        diff_ij = _apply_pbc_diff_torch(diff_ij, bounds)
        diff_ik = _apply_pbc_diff_torch(diff_ik, bounds)
    direction_raw = (diff_ij + diff_ik).float()
    return src_kept, neigh_j_kept, neigh_k_kept, weights, direction_raw


def _accumulate_channel_components(
    *,
    channel: str,
    frame_ids: Tensor,
    values: Tensor,
    weights: Tensor,
    factors: dict[str, Tensor],
    numerators: dict[str, Tensor],
    denominators: dict[str, Tensor],
) -> None:
    """Segmented weighted accumulation keyed by frame index."""
    if frame_ids.numel() == 0:
        return
    for component, factor in factors.items():
        factor_finite = torch.where(torch.isfinite(factor), factor, torch.zeros_like(factor))
        weighted = values * factor_finite * weights
        name = _channel_component_name(channel, component)
        numerators[name].index_add_(0, frame_ids, weighted)
        denominators[name].index_add_(0, frame_ids, weights)


def _clear_chunk(chunk: dict[str, list[Tensor]]) -> None:
    for values in chunk.values():
        values.clear()


def _flush_edge_chunk(
    *,
    chunk: dict[str, list[Tensor]],
    bilinear_channels: list[str],
    include_glueball: bool,
    component_mode: str,
    gamma: dict[str, Tensor],
    numerators: dict[str, Tensor],
    denominators: dict[str, Tensor],
) -> None:
    if not chunk["frame_ids"]:
        return
    frame_ids = torch.cat(chunk["frame_ids"], dim=0)
    weights = torch.cat(chunk["weights"], dim=0).float()
    direction = torch.cat(chunk["direction"], dim=0).float()
    factors = _component_factors(direction, component_mode)

    if bilinear_channels:
        color_i = torch.cat(chunk["color_i"], dim=0)
        color_j = torch.cat(chunk["color_j"], dim=0)
        valid_pair = torch.cat(chunk["valid_pair"], dim=0)
        for channel in bilinear_channels:
            values = _apply_projection(channel, color_i, color_j, gamma).float()
            valid = valid_pair & torch.isfinite(values)
            weights_valid = torch.where(valid, weights, torch.zeros_like(weights))
            if torch.any(weights_valid > 0):
                _accumulate_channel_components(
                    channel=channel,
                    frame_ids=frame_ids,
                    values=values,
                    weights=weights_valid,
                    factors=factors,
                    numerators=numerators,
                    denominators=denominators,
                )

    if include_glueball:
        glueball_values = torch.cat(chunk["glueball_values"], dim=0).float()
        valid = torch.isfinite(glueball_values)
        weights_valid = torch.where(valid, weights, torch.zeros_like(weights))
        if torch.any(weights_valid > 0):
            _accumulate_channel_components(
                channel="glueball",
                frame_ids=frame_ids,
                values=glueball_values,
                weights=weights_valid,
                factors=factors,
                numerators=numerators,
                denominators=denominators,
            )

    _clear_chunk(chunk)


def _flush_nucleon_chunk(
    *,
    chunk: dict[str, list[Tensor]],
    component_mode: str,
    numerators: dict[str, Tensor],
    denominators: dict[str, Tensor],
) -> None:
    if not chunk["frame_ids"]:
        return
    frame_ids = torch.cat(chunk["frame_ids"], dim=0)
    values = torch.cat(chunk["values"], dim=0).float()
    weights = torch.cat(chunk["weights"], dim=0).float()
    direction = torch.cat(chunk["direction"], dim=0).float()
    valid = torch.cat(chunk["valid"], dim=0)

    valid_mask = valid & torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    weights_valid = torch.where(valid_mask, weights, torch.zeros_like(weights))
    if torch.any(weights_valid > 0):
        factors = _component_factors(direction, component_mode)
        _accumulate_channel_components(
            channel="nucleon",
            frame_ids=frame_ids,
            values=values,
            weights=weights_valid,
            factors=factors,
            numerators=numerators,
            denominators=denominators,
        )
    _clear_chunk(chunk)


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


def _compute_traceless_tensor_result(
    *,
    tensor_series: Tensor,
    valid_t: Tensor,
    dt: float,
    config: CorrelatorConfig,
    channel_name: str = "tensor_traceless",
) -> ChannelCorrelatorResult:
    """Compute spin-2 correlator from per-frame symmetric traceless matrices."""
    if not torch.any(valid_t):
        return compute_channel_correlator(
            series=torch.zeros(0, device=tensor_series.device, dtype=torch.float32),
            dt=dt,
            config=config,
            channel_name=channel_name,
        )

    valid_series = tensor_series[valid_t].float()
    n_valid = int(valid_series.shape[0])
    max_lag = max(0, min(int(config.max_lag), n_valid - 1))

    # Flatten μν components and sum component-wise correlators to obtain
    # C(Δt) = <O^{μν}(t) O_{μν}(t+Δt)>.
    flat = valid_series.reshape(n_valid, -1).transpose(0, 1)  # [P, T]
    correlator_components = _fft_correlator_batched(
        flat,
        max_lag=max_lag,
        use_connected=bool(config.use_connected),
    )
    correlator = correlator_components.sum(dim=0)

    correlator_err: Tensor | None = None
    if bool(config.compute_bootstrap_errors):
        n_bootstrap = int(max(1, config.n_bootstrap))
        idx = torch.randint(0, n_valid, (n_bootstrap, n_valid), device=valid_series.device)
        sampled = valid_series[idx]  # [B, T, d, d]
        sampled_flat = sampled.reshape(n_bootstrap, n_valid, -1).permute(0, 2, 1)  # [B, P, T]
        boot_corr = _fft_correlator_batched(
            sampled_flat.reshape(-1, n_valid),
            max_lag=max_lag,
            use_connected=bool(config.use_connected),
        ).reshape(n_bootstrap, sampled_flat.shape[1], -1)
        correlator_err = boot_corr.sum(dim=1).std(dim=0)

    # Scalar diagnostic series used by the standard result container.
    series_scalar = torch.einsum("tij,tij->t", valid_series, valid_series)
    return _build_result_from_precomputed(
        channel_name=channel_name,
        series=series_scalar,
        correlator=correlator.float(),
        correlator_err=correlator_err.float() if correlator_err is not None else None,
        dt=dt,
        config=config,
    )


def _compute_channel_results_batched(
    series_buffers: dict[str, Tensor],
    valid_buffers: dict[str, Tensor],
    dt: float,
    config: CorrelatorConfig,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute correlators/masses with grouped batched FFT + bootstrap."""
    results: dict[str, ChannelCorrelatorResult] = {}
    groups: dict[bytes, tuple[Tensor, list[str]]] = {}

    for name, valid_t in valid_buffers.items():
        key = valid_t.detach().cpu().numpy().tobytes()
        if key in groups:
            groups[key][1].append(name)
        else:
            groups[key] = (valid_t, [name])

    for valid_t, names in groups.values():
        if not torch.any(valid_t):
            for name in names:
                results[name] = compute_channel_correlator(
                    series=torch.zeros(0, device=valid_t.device, dtype=torch.float32),
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
            series = series_stack[idx_name]
            correlator = correlators[idx_name]
            err = correlator_errs[idx_name] if correlator_errs is not None else None
            results[name] = _build_result_from_precomputed(
                channel_name=name,
                series=series,
                correlator=correlator,
                correlator_err=err,
                dt=dt,
                config=config,
            )

    return results


def compute_anisotropic_edge_channels(
    history: RunHistory,
    config: AnisotropicEdgeChannelConfig | None = None,
    channels: list[str] | None = None,
) -> AnisotropicEdgeChannelOutput:
    """Compute anisotropic edge-channel MC-time correlators.

    Uses only simulation-recorded direct Delaunay neighbors and recorded
    geometric edge weights. No neighbor recomputation is performed.
    """
    config = config or AnisotropicEdgeChannelConfig()
    if config.edge_weight_mode not in EDGE_WEIGHT_MODES:
        raise ValueError(f"edge_weight_mode must be one of {EDGE_WEIGHT_MODES}.")
    if config.nucleon_triplet_mode not in NUCLEON_TRIPLET_MODES:
        raise ValueError(f"nucleon_triplet_mode must be one of {NUCLEON_TRIPLET_MODES}.")
    if config.component_mode not in COMPONENT_MODES:
        raise ValueError(f"component_mode must be one of {COMPONENT_MODES}.")
    if history.neighbor_edges is None:
        msg = "RunHistory.neighbor_edges is required for anisotropic edge analysis."
        raise ValueError(msg)

    keep_dims = _resolve_keep_dims(history, config.keep_dims)
    if channels is None:
        channels = [
            "scalar",
            "pseudoscalar",
            "vector",
            "axial_vector",
            "tensor",
            "tensor_traceless",
            "nucleon",
            "glueball",
        ]
    unknown = sorted(set(channels) - SUPPORTED_CHANNELS)
    if unknown:
        raise ValueError(
            f"Unsupported channels {unknown}. Supported channels: {sorted(SUPPORTED_CHANNELS)}."
        )
    include_tensor_traceless = "tensor_traceless" in channels
    component_channels = [channel for channel in channels if channel != "tensor_traceless"]

    start_idx = max(1, int(history.n_recorded * float(config.warmup_fraction)))
    end_idx = max(start_idx + 1, int(history.n_recorded * float(config.end_fraction)))
    if config.mc_time_index is not None:
        start_idx = _resolve_mc_time_index(history, config.mc_time_index)
    frame_indices = list(range(start_idx, end_idx))
    n_frames = len(frame_indices)
    if n_frames == 0:
        return AnisotropicEdgeChannelOutput(
            channel_results={},
            component_labels=[],
            frame_indices=[],
            n_valid_frames=0,
            avg_alive_walkers=0.0,
            avg_edges=0.0,
        )

    color_batch, valid_batch = _compute_color_states_batch(
        history,
        start_idx,
        end_idx,
        config,
        keep_dims=keep_dims,
    )
    device = color_batch.device
    gamma = _build_gamma_matrices(len(keep_dims), device, torch.complex128)
    bounds = _slice_bounds(history.bounds, keep_dims)
    component_labels = _component_labels(len(keep_dims), config.component_mode)

    numerators: dict[str, Tensor] = {}
    denominators: dict[str, Tensor] = {}
    for channel in component_channels:
        for component in component_labels:
            name = _channel_component_name(channel, component)
            numerators[name] = torch.zeros(n_frames, device=device, dtype=torch.float32)
            denominators[name] = torch.zeros(n_frames, device=device, dtype=torch.float32)

    dim_keep = len(keep_dims)
    tensor_traceless_series = torch.zeros(
        (n_frames, dim_keep, dim_keep),
        device=device,
        dtype=torch.float32,
    )
    tensor_traceless_valid = torch.zeros(n_frames, device=device, dtype=torch.bool)
    eye_keep = torch.eye(dim_keep, device=device, dtype=torch.float32)

    bilinear_channels = [
        channel
        for channel in component_channels
        if channel in {"scalar", "pseudoscalar", "vector", "axial_vector", "tensor"}
    ]
    include_glueball = "glueball" in component_channels
    include_nucleon = "nucleon" in component_channels
    use_edge_chunk = bool(bilinear_channels) or include_glueball

    edge_chunk: dict[str, list[Tensor]] = {
        "frame_ids": [],
        "weights": [],
        "direction": [],
        "color_i": [],
        "color_j": [],
        "valid_pair": [],
        "glueball_values": [],
    }
    nucleon_chunk: dict[str, list[Tensor]] = {
        "frame_ids": [],
        "values": [],
        "weights": [],
        "direction": [],
        "valid": [],
    }

    alive_counts: list[float] = []
    edge_counts: list[float] = []

    for t_idx, frame_idx in enumerate(frame_indices):
        alive_mask = history.alive_mask[frame_idx - 1]
        alive_idx = torch.where(alive_mask)[0]
        if alive_idx.numel() < 2:
            continue

        positions_alive = history.x_before_clone[frame_idx][alive_idx][:, keep_dims]
        color_alive = color_batch[t_idx, alive_idx]
        valid_alive = valid_batch[t_idx, alive_idx]

        edges_global, geodesic_global = _extract_edges_for_frame(history, frame_idx)
        edges_local, geodesic_local, local_edge_indices = _recorded_subgraph_for_alive(
            edges_global,
            geodesic_global,
            alive_idx,
        )
        if edges_local.shape[0] == 0:
            continue

        edge_weights_base = _resolve_edge_weights(
            history=history,
            config=config,
            frame_idx=frame_idx,
            edges_global=edges_global,
            edges_local=edges_local,
            geodesic_local=geodesic_local,
            local_edge_indices=local_edge_indices,
            positions_alive=positions_alive,
            bounds=bounds,
        ).float()

        src = torch.as_tensor(edges_local[:, 0], device=device, dtype=torch.long)
        dst = torch.as_tensor(edges_local[:, 1], device=device, dtype=torch.long)

        volume_weights_alive: Tensor | None = None
        edge_weights = edge_weights_base
        if config.use_volume_weights:
            volume_weights_alive = _extract_volume_weights_alive(
                history,
                frame_idx,
                alive_idx,
                device=device,
                dtype=positions_alive.dtype,
            )
            edge_weights = (
                edge_weights
                * torch.sqrt(volume_weights_alive[src] * volume_weights_alive[dst]).float()
            )

        diff = positions_alive[dst] - positions_alive[src]
        if bool(history.pbc) and bounds is not None:
            diff = _apply_pbc_diff_torch(diff, bounds)
        distance = torch.linalg.vector_norm(diff, dim=-1).clamp(min=1e-8)
        direction = diff / distance.unsqueeze(-1)

        if include_tensor_traceless and dim_keep >= 2:
            color_scalar = (color_alive[src].conj() * color_alive[dst]).sum(dim=-1).real.float()
            valid_pair = valid_alive[src] & valid_alive[dst]
            valid_tensor = (
                valid_pair
                & torch.isfinite(color_scalar)
                & torch.isfinite(edge_weights)
                & torch.isfinite(diff).all(dim=-1)
            )
            if torch.any(valid_tensor):
                dx = diff[valid_tensor].float()
                weights_tensor = edge_weights[valid_tensor].float()
                weight_sum = weights_tensor.sum()
                if torch.isfinite(weight_sum) and float(weight_sum.item()) > 0:
                    color_tensor = color_scalar[valid_tensor]
                    r2 = torch.sum(dx * dx, dim=-1)
                    traceless = dx.unsqueeze(-1) * dx.unsqueeze(-2) - eye_keep.unsqueeze(0) * (
                        r2[:, None, None] / float(dim_keep)
                    )
                    coeff = weights_tensor * color_tensor
                    tensor_traceless_series[t_idx] = (coeff[:, None, None] * traceless).sum(
                        dim=0
                    ) / weight_sum.clamp(min=1e-12)
                    tensor_traceless_valid[t_idx] = True

        alive_counts.append(float(alive_idx.numel()))
        edge_counts.append(float(src.numel()))

        if use_edge_chunk:
            frame_ids = torch.full((src.shape[0],), t_idx, device=device, dtype=torch.long)
            edge_chunk["frame_ids"].append(frame_ids)
            edge_chunk["weights"].append(edge_weights.float())
            edge_chunk["direction"].append(direction.float())
            if bilinear_channels:
                edge_chunk["color_i"].append(color_alive[src])
                edge_chunk["color_j"].append(color_alive[dst])
                edge_chunk["valid_pair"].append(valid_alive[src] & valid_alive[dst])
            if include_glueball:
                force = history.force_viscous[frame_idx - 1]
                if keep_dims is not None:
                    force = force[:, keep_dims]
                force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)[alive_idx]
                edge_chunk["glueball_values"].append(
                    (0.5 * (force_sq[src] + force_sq[dst])).float()
                )

        if include_nucleon:
            if config.nucleon_triplet_mode == "companions":
                triplet_src, triplet_j, triplet_k, triplet_weights, direction_raw = (
                    _build_companion_triplets(
                        history=history,
                        frame_idx=frame_idx,
                        alive_idx=alive_idx,
                        positions_alive=positions_alive,
                        bounds=bounds,
                        src=src,
                        dst=dst,
                        edge_weights=edge_weights,
                        volume_weights_alive=volume_weights_alive,
                        edge_weight_mode=str(config.edge_weight_mode),
                    )
                )
            else:
                triplet_src, triplet_j, triplet_k, triplet_weights, direction_raw = (
                    _build_direct_neighbor_triplets(
                        src=src,
                        dst=dst,
                        edge_weights=edge_weights,
                        diff=diff,
                        n_alive=int(alive_idx.numel()),
                    )
                )
            if triplet_src.numel() > 0:
                triplet_values, valid_triplets = _compute_nucleon_triplet_values(
                    color_alive=color_alive,
                    valid_alive=valid_alive,
                    src=triplet_src,
                    neigh_j=triplet_j,
                    neigh_k=triplet_k,
                )
                triplet_norm = torch.linalg.vector_norm(direction_raw, dim=-1).clamp(min=1e-8)
                triplet_dir = direction_raw / triplet_norm.unsqueeze(-1)
                n_triplets = int(triplet_src.shape[0])
                nucleon_chunk["frame_ids"].append(
                    torch.full((n_triplets,), t_idx, device=device, dtype=torch.long)
                )
                nucleon_chunk["values"].append(triplet_values.float())
                nucleon_chunk["weights"].append(triplet_weights.float())
                nucleon_chunk["direction"].append(triplet_dir.float())
                nucleon_chunk["valid"].append(valid_triplets)

        if use_edge_chunk and (t_idx + 1) % PACK_CHUNK_FRAMES_EDGE == 0:
            _flush_edge_chunk(
                chunk=edge_chunk,
                bilinear_channels=bilinear_channels,
                include_glueball=include_glueball,
                component_mode=config.component_mode,
                gamma=gamma,
                numerators=numerators,
                denominators=denominators,
            )
        if include_nucleon:
            flush_stride = (
                PACK_CHUNK_FRAMES_NUCLEON_DIRECT
                if config.nucleon_triplet_mode == "direct_neighbors"
                else PACK_CHUNK_FRAMES_EDGE
            )
            if (t_idx + 1) % flush_stride == 0:
                _flush_nucleon_chunk(
                    chunk=nucleon_chunk,
                    component_mode=config.component_mode,
                    numerators=numerators,
                    denominators=denominators,
                )

    if use_edge_chunk:
        _flush_edge_chunk(
            chunk=edge_chunk,
            bilinear_channels=bilinear_channels,
            include_glueball=include_glueball,
            component_mode=config.component_mode,
            gamma=gamma,
            numerators=numerators,
            denominators=denominators,
        )
    if include_nucleon:
        _flush_nucleon_chunk(
            chunk=nucleon_chunk,
            component_mode=config.component_mode,
            numerators=numerators,
            denominators=denominators,
        )

    series_buffers: dict[str, Tensor] = {}
    valid_buffers: dict[str, Tensor] = {}
    frame_valid_any = torch.zeros(n_frames, device=device, dtype=torch.bool)
    for name, numerator in numerators.items():
        denom = denominators[name]
        valid_t = denom > 0
        series = torch.zeros_like(numerator)
        series[valid_t] = numerator[valid_t] / denom[valid_t].clamp(min=1e-12)
        series_buffers[name] = series
        valid_buffers[name] = valid_t
        frame_valid_any = frame_valid_any | valid_t
    if include_tensor_traceless:
        frame_valid_any = frame_valid_any | tensor_traceless_valid

    dt = float(history.delta_t * history.record_every)
    correlator_config = CorrelatorConfig(
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        window_widths=config.window_widths,
        min_mass=float(config.min_mass),
        max_mass=float(config.max_mass),
        fit_mode=str(config.fit_mode),
        fit_start=int(config.fit_start),
        fit_stop=config.fit_stop,
        min_fit_points=int(config.min_fit_points),
        compute_bootstrap_errors=bool(config.compute_bootstrap_errors),
        n_bootstrap=int(config.n_bootstrap),
    )

    results = _compute_channel_results_batched(
        series_buffers=series_buffers,
        valid_buffers=valid_buffers,
        dt=dt,
        config=correlator_config,
    )
    if include_tensor_traceless:
        if dim_keep < 2:
            results["tensor_traceless"] = compute_channel_correlator(
                series=torch.zeros(0, device=device, dtype=torch.float32),
                dt=dt,
                config=correlator_config,
                channel_name="tensor_traceless",
            )
        else:
            results["tensor_traceless"] = _compute_traceless_tensor_result(
                tensor_series=tensor_traceless_series,
                valid_t=tensor_traceless_valid,
                dt=dt,
                config=correlator_config,
                channel_name="tensor_traceless",
            )

    return AnisotropicEdgeChannelOutput(
        channel_results=results,
        component_labels=component_labels,
        frame_indices=frame_indices,
        n_valid_frames=int(frame_valid_any.sum().item()),
        avg_alive_walkers=float(np.mean(alive_counts)) if alive_counts else 0.0,
        avg_edges=float(np.mean(edge_counts)) if edge_counts else 0.0,
    )
