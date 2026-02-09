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
    ChannelCorrelatorResult,
    compute_channel_correlator,
    CorrelatorConfig,
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
    "nucleon",
    "glueball",
}


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
        factors["iso"] = torch.ones(direction.shape[0], device=direction.device, dtype=direction.dtype)
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
        raise ValueError(f"neighbor_edges[{frame_idx}] must have shape [E,2], got {edges_np.shape}.")

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


def _compute_edge_channel_values(
    history: RunHistory,
    channel: str,
    frame_idx: int,
    alive_idx: Tensor,
    src: Tensor,
    dst: Tensor,
    color_alive: Tensor,
    valid_alive: Tensor,
    gamma: dict[str, Tensor],
    keep_dims: list[int] | None,
) -> tuple[Tensor, Tensor]:
    if channel == "glueball":
        force = history.force_viscous[frame_idx - 1]
        if keep_dims is not None:
            force = force[:, keep_dims]
        force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)[alive_idx]
        edge_vals = 0.5 * (force_sq[src] + force_sq[dst])
        valid_edge = torch.isfinite(edge_vals)
        return edge_vals.float(), valid_edge

    color_i = color_alive[src]
    color_j = color_alive[dst]
    edge_vals = _apply_projection(channel, color_i, color_j, gamma).float()
    valid_edge = valid_alive[src] & valid_alive[dst] & torch.isfinite(edge_vals)
    return edge_vals, valid_edge


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

    src_parts: list[Tensor] = []
    j_parts: list[Tensor] = []
    k_parts: list[Tensor] = []
    weight_parts: list[Tensor] = []
    dir_parts: list[Tensor] = []

    for walker in range(n_alive):
        mask = src == walker
        deg = int(mask.sum().item())
        if deg < 2:
            continue
        neigh = dst[mask]
        weights = edge_weights[mask]
        disp = diff[mask]

        pair_idx = torch.triu_indices(deg, deg, offset=1, device=device)
        if pair_idx.numel() == 0:
            continue
        idx_j = pair_idx[0]
        idx_k = pair_idx[1]
        n_pairs = int(idx_j.numel())
        src_parts.append(torch.full((n_pairs,), walker, device=device, dtype=torch.long))
        j_parts.append(neigh[idx_j])
        k_parts.append(neigh[idx_k])
        weight_parts.append((weights[idx_j] * weights[idx_k]).float())
        dir_parts.append((disp[idx_j] + disp[idx_k]).float())

    if not src_parts:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        empty_dir = torch.empty(0, dim, device=device, dtype=torch.float32)
        return empty_long, empty_long, empty_long, empty_w, empty_dir

    return (
        torch.cat(src_parts, dim=0),
        torch.cat(j_parts, dim=0),
        torch.cat(k_parts, dim=0),
        torch.cat(weight_parts, dim=0),
        torch.cat(dir_parts, dim=0),
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
        msg = "nucleon_triplet_mode='companions' requires companions_distance and companions_clone."
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
    channels = channels or [
        "scalar",
        "pseudoscalar",
        "vector",
        "axial_vector",
        "tensor",
        "nucleon",
        "glueball",
    ]
    unknown = sorted(set(channels) - SUPPORTED_CHANNELS)
    if unknown:
        raise ValueError(
            f"Unsupported channels {unknown}. Supported channels: {sorted(SUPPORTED_CHANNELS)}."
        )

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

    series_buffers: dict[str, Tensor] = {}
    valid_buffers: dict[str, Tensor] = {}
    for channel in channels:
        for component in component_labels:
            name = _channel_component_name(channel, component)
            series_buffers[name] = torch.zeros(n_frames, device=device, dtype=torch.float32)
            valid_buffers[name] = torch.zeros(n_frames, device=device, dtype=torch.bool)

    alive_counts: list[float] = []
    edge_counts: list[float] = []
    frame_valid_any = torch.zeros(n_frames, device=device, dtype=torch.bool)

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
            edge_weights = edge_weights * torch.sqrt(
                volume_weights_alive[src] * volume_weights_alive[dst]
            ).float()

        diff = positions_alive[dst] - positions_alive[src]
        if bool(history.pbc) and bounds is not None:
            diff = _apply_pbc_diff_torch(diff, bounds)
        distance = torch.linalg.vector_norm(diff, dim=-1).clamp(min=1e-8)
        direction = diff / distance.unsqueeze(-1)
        components = _component_factors(direction.float(), config.component_mode)

        alive_counts.append(float(alive_idx.numel()))
        edge_counts.append(float(src.numel()))

        for channel in channels:
            if channel == "nucleon":
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
                if triplet_src.numel() == 0:
                    continue

                triplet_norm = torch.linalg.vector_norm(direction_raw, dim=-1).clamp(min=1e-8)
                triplet_dir = direction_raw / triplet_norm.unsqueeze(-1)
                triplet_components = _component_factors(triplet_dir.float(), config.component_mode)

                triplet_values, valid_triplets = _compute_nucleon_triplet_values(
                    color_alive=color_alive,
                    valid_alive=valid_alive,
                    src=triplet_src,
                    neigh_j=triplet_j,
                    neigh_k=triplet_k,
                )
                valid_triplets = valid_triplets & torch.isfinite(triplet_weights) & (triplet_weights > 0)
                if not torch.any(valid_triplets):
                    continue
                weights_valid = torch.where(
                    valid_triplets,
                    triplet_weights,
                    torch.zeros_like(triplet_weights),
                )
                denom = weights_valid.sum().clamp(min=1e-12)
                if float(denom.item()) <= 0:
                    continue

                for component, factor in triplet_components.items():
                    factor_finite = torch.where(
                        torch.isfinite(factor),
                        factor,
                        torch.zeros_like(factor),
                    )
                    value = (triplet_values * factor_finite * weights_valid).sum() / denom
                    name = _channel_component_name(channel, component)
                    series_buffers[name][t_idx] = value.float()
                    valid_buffers[name][t_idx] = True
                    frame_valid_any[t_idx] = True
                continue

            edge_values, valid_edge = _compute_edge_channel_values(
                history=history,
                channel=channel,
                frame_idx=frame_idx,
                alive_idx=alive_idx,
                src=src,
                dst=dst,
                color_alive=color_alive,
                valid_alive=valid_alive,
                gamma=gamma,
                keep_dims=keep_dims,
            )
            if not torch.any(valid_edge):
                continue

            weights_valid = torch.where(valid_edge, edge_weights, torch.zeros_like(edge_weights))
            denom = weights_valid.sum().clamp(min=1e-12)
            if float(denom.item()) <= 0:
                continue

            for component, factor in components.items():
                factor_finite = torch.where(
                    torch.isfinite(factor),
                    factor,
                    torch.zeros_like(factor),
                )
                value = (edge_values * factor_finite * weights_valid).sum() / denom
                name = _channel_component_name(channel, component)
                series_buffers[name][t_idx] = value.float()
                valid_buffers[name][t_idx] = True
                frame_valid_any[t_idx] = True

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

    results: dict[str, ChannelCorrelatorResult] = {}
    for name, series_full in series_buffers.items():
        valid_t = valid_buffers[name]
        series = series_full[valid_t]
        results[name] = compute_channel_correlator(
            series=series,
            dt=dt,
            config=correlator_config,
            channel_name=name,
        )

    return AnisotropicEdgeChannelOutput(
        channel_results=results,
        component_labels=component_labels,
        frame_indices=frame_indices,
        n_valid_frames=int(frame_valid_any.sum().item()),
        avg_alive_walkers=float(np.mean(alive_counts)) if alive_counts else 0.0,
        avg_edges=float(np.mean(edge_counts)) if edge_counts else 0.0,
    )
