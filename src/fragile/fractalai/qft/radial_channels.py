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
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra
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
    "kernel",
    "riemannian_kernel",
    "riemannian_kernel_volume",
)
NEIGHBOR_WEIGHT_MODES = (
    *dict.fromkeys((
        "uniform",
        "volume",
        "euclidean",
        "inv_euclidean",
        "inv_geodesic_iso",
        "inv_geodesic_full",
        "kernel",
        *RECORDED_EDGE_WEIGHT_MODES,
    )),
)


@dataclass
class RadialChannelConfig:
    """Configuration for radial channel correlators."""

    time_axis: str = "mc"  # "mc" (Monte Carlo time) or "radial" (single-slice screening)
    mc_time_index: int | None = None
    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
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
    high = (
        bounds.high.detach().cpu().numpy()
        if torch.is_tensor(bounds.high)
        else np.asarray(bounds.high)
    )
    low = (
        bounds.low.detach().cpu().numpy()
        if torch.is_tensor(bounds.low)
        else np.asarray(bounds.low)
    )
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


def _compute_color_states_batch(
    history: RunHistory,
    start_idx: int,
    end_idx: int,
    config: RadialChannelConfig,
    keep_dims: list[int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute color states for [start_idx, end_idx) in one batched pass."""
    if end_idx <= start_idx:
        empty_color = history.x_before_clone.new_empty((0, history.N, history.d)).to(
            torch.complex64
        )
        empty_valid = torch.zeros(
            0, history.N, dtype=torch.bool, device=history.x_before_clone.device
        )
        return empty_color, empty_valid

    v_pre = history.v_before_clone[start_idx:end_idx]
    force_visc = history.force_viscous[start_idx - 1 : end_idx - 1]

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

    color = color[:, :, keep_dims]
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


def _build_neighbor_data_dense(
    edges: np.ndarray,
    alive_idx: Tensor,
    positions: Tensor,
    bounds: Any | None,
    volumes: Tensor | None,
    weight_mode: str,
    edge_mode_values: np.ndarray | None,
    max_neighbors: int,
    pbc: bool,
    kernel_length_scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build dense [N, k] neighbor/weight matrices for one frame."""
    if edges.ndim != 2 or edges.shape[1] != 2:
        msg = f"Expected edges with shape [E, 2], got {edges.shape}."
        raise ValueError(msg)
    n_alive = int(positions.shape[0])
    if n_alive <= 0:
        empty_i = torch.empty(0, 0, dtype=torch.long, device=positions.device)
        empty_w = torch.empty(0, 0, dtype=positions.dtype, device=positions.device)
        empty_m = torch.empty(0, 0, dtype=torch.bool, device=positions.device)
        return empty_i, empty_w, empty_m

    edges_t = torch.as_tensor(edges, device=positions.device, dtype=torch.long)
    if edges_t.numel() == 0:
        msg = "Recorded neighbor graph has no alive-alive edges for the requested frame."
        raise ValueError(msg)

    src = edges_t[:, 0]
    dst = edges_t[:, 1]
    valid_edges = (src >= 0) & (src < n_alive) & (dst >= 0) & (dst < n_alive) & (src != dst)
    if not torch.any(valid_edges):
        msg = "Recorded neighbor graph has no valid directed edges after filtering."
        raise ValueError(msg)
    src = src[valid_edges]
    dst = dst[valid_edges]

    volumes_t: Tensor | None = None
    if volumes is not None:
        volumes_t = volumes.to(device=positions.device, dtype=positions.dtype)
        if volumes_t.ndim != 1 or int(volumes_t.shape[0]) != n_alive:
            msg = (
                "Volume weighting requires per-alive recorded volume weights "
                f"of shape [{n_alive}], got {tuple(volumes_t.shape)}."
            )
            raise ValueError(msg)

    edge_mode_t: Tensor | None = None
    if edge_mode_values is not None:
        edge_vals = np.asarray(edge_mode_values, dtype=float).reshape(-1)
        if edge_vals.shape[0] != edges.shape[0]:
            msg = (
                f"Edge weight mode values must align with edges: "
                f"E={edges.shape[0]}, W={edge_vals.shape[0]}."
            )
            raise ValueError(msg)
        edge_vals = edge_vals[valid_edges.detach().cpu().numpy()]
        edge_mode_t = torch.as_tensor(edge_vals, device=positions.device, dtype=positions.dtype)

    if weight_mode == "uniform":
        edge_weights = torch.ones(src.shape[0], device=positions.device, dtype=positions.dtype)
    elif weight_mode == "volume":
        if volumes_t is None:
            msg = "neighbor_weighting='volume' requires per-alive volume weights."
            raise ValueError(msg)
        edge_weights = volumes_t[dst]
    elif weight_mode in {"euclidean", "inv_euclidean", "kernel"}:
        diff = positions[dst] - positions[src]
        if pbc and bounds is not None:
            diff = _apply_pbc_diff_torch(diff, bounds)
        dist = torch.linalg.vector_norm(diff, dim=-1).clamp(min=1e-8)
        if weight_mode == "kernel":
            edge_weights = torch.exp(-(dist**2) / (2.0 * kernel_length_scale**2))
        elif weight_mode == "euclidean":
            edge_weights = dist
        else:
            edge_weights = 1.0 / dist
    elif weight_mode in {"inv_geodesic_iso", "inv_geodesic_full"}:
        if edge_mode_t is None:
            msg = f"neighbor_weighting='{weight_mode}' requires recorded edge distances."
            raise ValueError(msg)
        if not torch.isfinite(edge_mode_t).all() or torch.any(edge_mode_t <= 0):
            msg = (
                f"neighbor_weighting='{weight_mode}' received non-positive/non-finite edge values."
            )
            raise ValueError(msg)
        edge_weights = 1.0 / edge_mode_t.clamp(min=1e-8)
    elif weight_mode in RECORDED_EDGE_WEIGHT_MODES:
        if edge_mode_t is None:
            msg = f"neighbor_weighting='{weight_mode}' requires recorded edge mode values."
            raise ValueError(msg)
        if not torch.isfinite(edge_mode_t).all() or torch.any(edge_mode_t < 0):
            msg = f"neighbor_weighting='{weight_mode}' received negative/non-finite edge values."
            raise ValueError(msg)
        edge_weights = edge_mode_t
    else:
        msg = f"Unsupported neighbor_weighting mode: {weight_mode}"
        raise ValueError(msg)

    # Keep existing semantics: for non-recorded and non-volume modes, multiply by V_j when available.
    if (
        weight_mode != "volume"
        and weight_mode not in RECORDED_EDGE_WEIGHT_MODES
        and volumes_t is not None
    ):
        edge_weights = edge_weights * volumes_t[dst]

    order = torch.argsort(src, stable=True)
    src = src[order]
    dst = dst[order]
    edge_weights = edge_weights[order]

    counts = torch.bincount(src, minlength=n_alive)
    if torch.any(counts == 0):
        missing_local = torch.where(counts == 0)[0]
        if alive_idx.numel() == n_alive:
            missing = alive_idx[missing_local].detach().cpu().tolist()
        else:
            missing = missing_local.detach().cpu().tolist()
        preview = ", ".join(str(int(m)) for m in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        msg = (
            f"Recorded graph has walkers without neighbors in alive set "
            f"({len(missing)} walkers: {preview}{suffix})."
        )
        raise ValueError(msg)

    row_starts = torch.zeros(n_alive, dtype=torch.long, device=positions.device)
    if n_alive > 1:
        row_starts[1:] = torch.cumsum(counts[:-1], dim=0)
    col = torch.arange(src.shape[0], device=positions.device, dtype=torch.long) - row_starts[src]

    if max_neighbors and max_neighbors > 0:
        keep = col < int(max_neighbors)
        src = src[keep]
        dst = dst[keep]
        edge_weights = edge_weights[keep]
        col = col[keep]
        if src.numel() == 0:
            msg = "No neighbors remain after applying neighbor_k cap."
            raise ValueError(msg)

    k = int(col.max().item()) + 1
    rows = (
        torch.arange(n_alive, device=positions.device, dtype=torch.long).unsqueeze(1).expand(-1, k)
    )
    neighbor_indices = rows.clone()
    neighbor_weights = torch.zeros((n_alive, k), device=positions.device, dtype=positions.dtype)
    neighbor_mask = torch.zeros((n_alive, k), device=positions.device, dtype=torch.bool)

    neighbor_indices[src, col] = dst
    neighbor_weights[src, col] = edge_weights
    neighbor_mask[src, col] = True
    return neighbor_indices, neighbor_weights, neighbor_mask


def _sample_pairs(
    n: int, max_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
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


def _build_graph_csr(edges: np.ndarray, weights: np.ndarray, n: int) -> csr_matrix:
    """Build CSR graph for fast shortest-path queries."""
    if edges.ndim != 2 or edges.shape[1] != 2:
        msg = f"Expected edges with shape [E, 2], got {edges.shape}."
        raise ValueError(msg)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != edges.shape[0]:
        msg = f"Edge/weight mismatch: E={edges.shape[0]}, W={w.shape[0]}."
        raise ValueError(msg)
    if not np.all(np.isfinite(w)):
        msg = "Graph edge weights contain non-finite values."
        raise ValueError(msg)
    if np.any(w < 0.0):
        msg = "Graph edge weights must be non-negative for shortest-path computation."
        raise ValueError(msg)

    src = edges[:, 0].astype(np.int64, copy=False)
    dst = edges[:, 1].astype(np.int64, copy=False)
    valid = src != dst
    if not np.any(valid):
        return csr_matrix((n, n), dtype=float)

    graph = csr_matrix(
        (w[valid], (src[valid], dst[valid])),
        shape=(n, n),
        dtype=float,
    )
    graph.sum_duplicates()
    graph.eliminate_zeros()
    return graph


def _pair_distances_graph(
    graph: csr_matrix,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    source_chunk_size: int = 128,
) -> np.ndarray:
    if pair_i.shape != pair_j.shape:
        msg = f"pair_i and pair_j must have same shape, got {pair_i.shape} vs {pair_j.shape}."
        raise ValueError(msg)
    if pair_i.size == 0:
        return np.empty(0, dtype=float)

    pair_i_arr = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pair_j_arr = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    n_nodes = graph.shape[0]
    if np.any(pair_i_arr < 0) or np.any(pair_i_arr >= n_nodes):
        msg = "pair_i contains indices outside graph node range."
        raise ValueError(msg)
    if np.any(pair_j_arr < 0) or np.any(pair_j_arr >= n_nodes):
        msg = "pair_j contains indices outside graph node range."
        raise ValueError(msg)

    distances = np.full(pair_i_arr.shape[0], np.inf, dtype=float)
    order = np.argsort(pair_i_arr, kind="stable")
    src_sorted = pair_i_arr[order]
    dst_sorted = pair_j_arr[order]

    unique_sources, source_starts, source_counts = np.unique(
        src_sorted,
        return_index=True,
        return_counts=True,
    )
    source_ends = source_starts + source_counts
    chunk = max(1, int(source_chunk_size))

    for source_chunk_start in range(0, unique_sources.shape[0], chunk):
        source_chunk_end = min(source_chunk_start + chunk, unique_sources.shape[0])
        chunk_sources = unique_sources[source_chunk_start:source_chunk_end]

        dist_chunk = sparse_dijkstra(
            csgraph=graph,
            directed=True,
            indices=chunk_sources,
            return_predecessors=False,
        )
        if dist_chunk.ndim == 1:
            dist_chunk = dist_chunk[np.newaxis, :]

        pair_start = int(source_starts[source_chunk_start])
        pair_end = int(source_ends[source_chunk_end - 1])
        local_sources = src_sorted[pair_start:pair_end]
        local_targets = dst_sorted[pair_start:pair_end]
        row_idx = np.searchsorted(chunk_sources, local_sources)
        gathered = dist_chunk[row_idx, local_targets]
        distances[order[pair_start:pair_end]] = np.asarray(gathered, dtype=float)

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


def _compute_operator_values_dense(
    channel: str,
    color: Tensor,
    valid: Tensor,
    alive_idx: Tensor,
    neighbor_indices: Tensor,
    neighbor_weights: Tensor,
    neighbor_mask: Tensor,
    gamma: dict[str, Tensor],
    history: RunHistory,
    frame_idx: int,
    keep_dims: list[int] | None,
) -> Tensor:
    """Vectorized per-frame operator computation on dense neighbor tensors."""
    if channel == "glueball":
        return _compute_glueball_operator(history, frame_idx, alive_idx, keep_dims=keep_dims)

    n_alive = int(color.shape[0])
    outputs = torch.zeros(n_alive, device=color.device, dtype=torch.float32)
    if n_alive == 0 or neighbor_indices.numel() == 0:
        return outputs

    if channel == "nucleon" and color.shape[1] < 3:
        return outputs

    row = torch.arange(n_alive, device=color.device, dtype=torch.long).unsqueeze(1)

    if channel == "nucleon":
        k = int(neighbor_indices.shape[1])
        if k < 2:
            return outputs
        pair_idx = torch.triu_indices(k, k, offset=1, device=color.device)
        if pair_idx.numel() == 0:
            return outputs
        j = neighbor_indices[:, pair_idx[0]]
        k_idx = neighbor_indices[:, pair_idx[1]]

        pair_mask = neighbor_mask[:, pair_idx[0]] & neighbor_mask[:, pair_idx[1]]
        valid_mask = (
            pair_mask & (j != row) & (k_idx != row) & valid.unsqueeze(1) & valid[j] & valid[k_idx]
        )

        color_3 = color[:, :3]
        color_i = color_3.unsqueeze(1).expand(-1, pair_idx.shape[1], -1)
        color_j = color_3[j]
        color_k = color_3[k_idx]
        matrix = torch.stack([color_i, color_j, color_k], dim=-1)
        det = torch.linalg.det(matrix)
        det = det.real if det.is_complex() else det

        pair_weights = neighbor_weights[:, pair_idx[0]] * neighbor_weights[:, pair_idx[1]]
        pair_weights = torch.where(valid_mask, pair_weights, torch.zeros_like(pair_weights))
        denom = pair_weights.sum(dim=1)
        num = (det * pair_weights).sum(dim=1)
        out = torch.where(denom > 0, num / denom.clamp(min=1e-12), torch.zeros_like(num))
        return out.float()

    color_i = color.unsqueeze(1).expand(-1, neighbor_indices.shape[1], -1)
    color_j = color[neighbor_indices]
    op_vals = _apply_projection(channel, color_i, color_j, gamma)
    valid_mask = (
        neighbor_mask & valid.unsqueeze(1) & valid[neighbor_indices] & (neighbor_indices != row)
    )
    weights_eff = torch.where(valid_mask, neighbor_weights, torch.zeros_like(neighbor_weights))
    denom = weights_eff.sum(dim=1)
    num = (op_vals * weights_eff).sum(dim=1)
    out = torch.where(denom > 0, num / denom.clamp(min=1e-12), torch.zeros_like(num))
    return out.float()


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
            f"neighbor_edges[{frame_idx}] must have shape [E,2], " f"got {tuple(edges_np.shape)}."
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project recorded graph/geodesics to alive walkers with vectorized indexing.

    Returns:
        - local_edges [E_alive, 2] in local alive indexing
        - local_geodesic [E_alive] aligned with local_edges
        - global_edge_indices [E_alive] indices into original per-frame edge arrays
    """
    alive_np = np.asarray(alive_idx.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    if alive_np.size == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=np.int64),
        )

    if edges_global.ndim != 2 or edges_global.shape[1] != 2:
        msg = f"Expected edges_global shape [E, 2], got {edges_global.shape}."
        raise ValueError(msg)
    if geodesic_global.shape[0] != edges_global.shape[0]:
        msg = (
            "edges_global/geodesic_global mismatch: "
            f"E={edges_global.shape[0]}, G={geodesic_global.shape[0]}."
        )
        raise ValueError(msg)

    max_global = int(
        max(
            np.max(alive_np),
            np.max(edges_global) if edges_global.size else -1,
        )
    )
    map_size = max_global + 1
    global_to_local = np.full(map_size, -1, dtype=np.int64)
    global_to_local[alive_np] = np.arange(alive_np.size, dtype=np.int64)

    src = edges_global[:, 0].astype(np.int64, copy=False)
    dst = edges_global[:, 1].astype(np.int64, copy=False)
    valid_range = (src >= 0) & (src < map_size) & (dst >= 0) & (dst < map_size) & (src != dst)
    if not np.any(valid_range):
        msg = "Recorded neighbor graph has no valid directed edges in frame."
        raise ValueError(msg)

    edge_idx_range = np.nonzero(valid_range)[0]
    src_local = global_to_local[src[valid_range]]
    dst_local = global_to_local[dst[valid_range]]
    keep_alive = (src_local >= 0) & (dst_local >= 0) & (src_local != dst_local)
    if not np.any(keep_alive):
        msg = "Recorded neighbor graph has no alive-alive edges for the requested frame."
        raise ValueError(msg)

    kept_global_indices = edge_idx_range[keep_alive]
    local_edges = np.stack([src_local[keep_alive], dst_local[keep_alive]], axis=1).astype(
        np.int64, copy=False
    )
    local_geodesic = np.asarray(geodesic_global[kept_global_indices], dtype=float)

    # Deduplicate directed local edges, preserving first appearance to keep alignment deterministic.
    edge_keys = local_edges[:, 0] * alive_np.size + local_edges[:, 1]
    _, first_idx = np.unique(edge_keys, return_index=True)
    keep_order = np.sort(first_idx)
    local_edges = local_edges[keep_order]
    local_geodesic = local_geodesic[keep_order]
    kept_global_indices = kept_global_indices[keep_order]

    if local_edges.size == 0:
        msg = "Recorded neighbor graph has no alive-alive edges after deduplication."
        raise ValueError(msg)

    counts = np.bincount(local_edges[:, 0], minlength=alive_np.size)
    missing_local = np.where(counts == 0)[0]
    if missing_local.size > 0:
        missing = alive_np[missing_local].tolist()
        preview = ", ".join(str(int(m)) for m in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        msg = (
            f"Recorded graph has walkers without neighbors in alive set "
            f"({len(missing)} walkers: {preview}{suffix})."
        )
        raise ValueError(msg)

    return local_edges, local_geodesic, kept_global_indices


def _extract_recorded_edge_mode_values(
    history: RunHistory,
    frame_idx: int,
    mode: str,
    n_edges: int,
) -> np.ndarray:
    """Extract recorded per-edge weights for a scutoid mode aligned with neighbor_edges."""
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
    if values_np.shape[0] != n_edges:
        msg = (
            f"edge_weights[{frame_idx}]['{mode}'] size mismatch with neighbor_edges: "
            f"W={values_np.shape[0]}, E={n_edges}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(values_np)):
        msg = f"edge_weights[{frame_idx}]['{mode}'] contains non-finite values."
        raise ValueError(msg)
    return values_np


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
    frame_idx: int,
    keep_dims: list[int] | None,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    distances: np.ndarray,
    bin_edges: np.ndarray,
    neighbor_indices: Tensor,
    neighbor_weights: Tensor,
    neighbor_mask: Tensor,
    volumes: Tensor | np.ndarray | None,
    operators_override: dict[str, np.ndarray] | None = None,
) -> RadialChannelOutput:
    n_alive = positions.shape[0]
    device = positions.device
    operators_np: dict[str, np.ndarray] = {}
    if operators_override is None:
        gamma = _build_gamma_matrices(positions.shape[1], device, torch.complex128)

        color_full, valid_full = _compute_color_states_single(
            history,
            frame_idx,
            config,
            keep_dims=keep_dims,
        )
        color = color_full[alive_idx]
        valid = valid_full[alive_idx]

        for channel in channels:
            op = _compute_operator_values_dense(
                channel,
                color,
                valid,
                alive_idx,
                neighbor_indices,
                neighbor_weights,
                neighbor_mask,
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
        if volumes is None:
            msg = (
                "use_volume_weights=True requires per-alive volume weights "
                "for radial correlator binning."
            )
            raise ValueError(msg)
        v = (
            volumes.detach().cpu().numpy()
            if torch.is_tensor(volumes)
            else np.asarray(volumes, dtype=float)
        )
        if v.size != n_alive:
            msg = (
                "use_volume_weights=True requires per-alive volume weights "
                f"of size {n_alive}, got {v.size}."
            )
            raise ValueError(msg)
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
                "graph_full edge/geodesic mismatch: " f"E={edges.shape[0]}, G={weights.shape[0]}."
            )
            raise ValueError(msg)
    else:
        msg = f"Unsupported distance_mode: {distance_mode}"
        raise ValueError(msg)

    graph = _build_graph_csr(edges, weights, positions.shape[0])
    return _pair_distances_graph(graph, pair_i, pair_j)


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
    end_fraction = getattr(config, "end_fraction", 1.0)
    end_idx = max(start_idx + 1, int(history.n_recorded * end_fraction))
    if config.mc_time_index is not None:
        start_idx = _resolve_mc_time_index(history, config.mc_time_index)

    frame_indices = list(range(start_idx, end_idx))
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
    dim = len(keep_dims) if keep_dims is not None else history.d
    bounds = history.bounds
    if keep_dims is not None:
        bounds = _slice_bounds(bounds, keep_dims)

    positions_batch = history.x_before_clone[start_idx:end_idx]
    if keep_dims is not None:
        positions_batch = positions_batch[:, :, keep_dims]
    alive_batch = history.alive_mask[start_idx - 1 : end_idx - 1]
    color_batch, valid_batch = _compute_color_states_batch(
        history,
        start_idx,
        end_idx,
        config,
        keep_dims=keep_dims,
    )
    gamma = _build_gamma_matrices(dim, device, torch.complex128)

    volume_batch: Tensor | None = None
    volume_history = getattr(history, "riemannian_volume_weights", None)
    if volume_history is not None:
        if torch.is_tensor(volume_history):
            volume_all = volume_history
        else:
            volume_all = torch.as_tensor(volume_history)
        if (
            volume_all.ndim == 2
            and volume_all.shape[0] >= end_idx - 1
            and volume_all.shape[1] >= history.N
        ):
            volume_batch = volume_all[start_idx - 1 : end_idx - 1, : history.N].to(
                device=device,
                dtype=positions_batch.dtype,
            )

    series_buffers = {
        channel: torch.zeros(n_frames, device=device, dtype=torch.float32) for channel in channels
    }
    valid_frames = torch.zeros(n_frames, device=device, dtype=torch.bool)

    for t_idx, frame_idx in enumerate(frame_indices):
        alive_mask_full = alive_batch[t_idx]
        alive_idx = torch.where(alive_mask_full)[0]
        if alive_idx.numel() < 2:
            continue

        positions_alive = positions_batch[t_idx, alive_idx]
        volume_weights_alive = volume_batch[t_idx, alive_idx] if volume_batch is not None else None

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

        edges_global, geodesic_global = _require_recorded_edges_and_geodesic(history, frame_idx)
        edges, geodesic_edges, local_edge_indices = _recorded_subgraph_for_alive(
            edges_global,
            geodesic_global,
            alive_idx,
        )

        edge_mode_values: np.ndarray | None = None
        if config.neighbor_weighting == "inv_geodesic_full":
            edge_mode_values = geodesic_edges
        elif config.neighbor_weighting == "inv_geodesic_iso":
            edge_mode_values = _compute_recorded_iso_edge_lengths(
                positions=positions_alive,
                edges=edges,
                bounds=bounds,
                pbc=bool(history.pbc),
            )
        elif config.neighbor_weighting in RECORDED_EDGE_WEIGHT_MODES:
            mode_values_global = _extract_recorded_edge_mode_values(
                history,
                frame_idx,
                config.neighbor_weighting,
                n_edges=edges_global.shape[0],
            )
            edge_mode_values = mode_values_global[local_edge_indices]

        neighbor_indices, neighbor_weights, neighbor_mask = _build_neighbor_data_dense(
            edges=edges,
            alive_idx=alive_idx,
            positions=positions_alive,
            bounds=bounds,
            volumes=volume_weights_alive,
            weight_mode=config.neighbor_weighting,
            edge_mode_values=edge_mode_values,
            max_neighbors=int(config.neighbor_k),
            pbc=bool(history.pbc),
            kernel_length_scale=config.kernel_length_scale,
        )

        color = color_batch[t_idx, alive_idx]
        valid = valid_batch[t_idx, alive_idx]

        volume_tensor: Tensor | None = None
        if config.use_volume_weights:
            if volume_weights_alive is None or volume_weights_alive.numel() != alive_idx.numel():
                msg = f"Missing per-alive volume weights for frame {frame_idx}."
                raise ValueError(msg)
            volume_tensor = volume_weights_alive.to(
                device=positions_alive.device,
                dtype=torch.float32,
            ).clamp(min=0.0)
            if float(volume_tensor.sum().item()) <= 0:
                msg = f"Non-positive total volume weight at frame {frame_idx}."
                raise ValueError(msg)

        for channel in channels:
            op = _compute_operator_values_dense(
                channel,
                color,
                valid,
                alive_idx,
                neighbor_indices,
                neighbor_weights,
                neighbor_mask,
                gamma,
                history,
                frame_idx,
                keep_dims,
            )
            if volume_tensor is not None and volume_tensor.shape[0] == op.shape[0]:
                denom = volume_tensor.sum().clamp(min=1e-12)
                series_buffers[channel][t_idx] = (op.float() * volume_tensor).sum() / denom
            else:
                series_buffers[channel][t_idx] = op.float().mean()

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
                correlator_err.detach().cpu().numpy() if correlator_err is not None else None
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
        raise ValueError(f"neighbor_weighting must be one of {NEIGHBOR_WEIGHT_MODES}")
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

    volume_weights_alive: Tensor | None = None
    volume_history = getattr(history, "riemannian_volume_weights", None)
    if volume_history is not None and frame_idx > 0 and frame_idx - 1 < len(volume_history):
        vol_full = volume_history[frame_idx - 1]
        if not torch.is_tensor(vol_full):
            vol_full = torch.as_tensor(vol_full)
        volume_weights_alive = vol_full.to(
            device=positions_alive.device,
            dtype=positions_alive.dtype,
        )[alive_idx]

    if config.use_volume_weights and volume_weights_alive is None:
        msg = (
            "use_volume_weights=True requires RunHistory.riemannian_volume_weights "
            f"for frame {frame_idx}."
        )
        raise ValueError(msg)
    if str(config.neighbor_weighting) == "volume" and volume_weights_alive is None:
        msg = (
            "neighbor_weighting='volume' requires RunHistory.riemannian_volume_weights "
            f"for frame {frame_idx}."
        )
        raise ValueError(msg)
    volumes_full = (
        volume_weights_alive
        if volume_weights_alive is not None
        else torch.ones(len(alive_idx), device=positions_alive.device, dtype=positions_alive.dtype)
    )
    edges_global, geodesic_global = _require_recorded_edges_and_geodesic(history, frame_idx)
    edges_full, geodesic_full, local_edge_indices = _recorded_subgraph_for_alive(
        edges_global,
        geodesic_global,
        alive_idx,
    )

    neighbor_indices: Tensor
    neighbor_weights: Tensor
    neighbor_mask: Tensor
    if operators_override is None:
        edge_mode_values: np.ndarray | None = None
        if config.neighbor_weighting == "inv_geodesic_full":
            edge_mode_values = geodesic_full
        elif config.neighbor_weighting == "inv_geodesic_iso":
            edge_mode_values = _compute_recorded_iso_edge_lengths(
                positions=positions_alive,
                edges=edges_full,
                bounds=bounds_full,
                pbc=bool(history.pbc),
            )
        elif config.neighbor_weighting in RECORDED_EDGE_WEIGHT_MODES:
            mode_values_global = _extract_recorded_edge_mode_values(
                history,
                frame_idx,
                config.neighbor_weighting,
                n_edges=edges_global.shape[0],
            )
            edge_mode_values = mode_values_global[local_edge_indices]

        neighbor_indices, neighbor_weights, neighbor_mask = _build_neighbor_data_dense(
            edges=edges_full,
            alive_idx=alive_idx,
            positions=positions_alive,
            bounds=bounds_full,
            volumes=volumes_full,
            weight_mode=config.neighbor_weighting,
            edge_mode_values=edge_mode_values,
            max_neighbors=int(config.neighbor_k),
            pbc=bool(history.pbc),
            kernel_length_scale=config.kernel_length_scale,
        )
    else:
        neighbor_indices = torch.empty(
            len(alive_idx), 0, dtype=torch.long, device=positions_alive.device
        )
        neighbor_weights = torch.empty(
            len(alive_idx), 0, dtype=positions_alive.dtype, device=positions_alive.device
        )
        neighbor_mask = torch.empty(
            len(alive_idx), 0, dtype=torch.bool, device=positions_alive.device
        )

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
        frame_idx=frame_idx,
        keep_dims=None,
        pair_i=pair_i,
        pair_j=pair_j,
        distances=distances_full,
        bin_edges=bin_edges_full,
        neighbor_indices=neighbor_indices,
        neighbor_weights=neighbor_weights,
        neighbor_mask=neighbor_mask,
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
            if config.distance_mode == "graph_full":
                distances = distances_full
            else:
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
            if (
                not np.isfinite(global_min)
                or not np.isfinite(global_max)
                or global_max <= global_min
            ):
                global_min = 0.0
                global_max = 1.0
            bin_edges_3d = np.linspace(global_min, global_max, config.n_bins + 1)

            combined_vals: dict[str, np.ndarray] = {}
            combined_counts = np.zeros(config.n_bins)

            for axis, distances in axis_distances.items():
                keep_dims = [i for i in range(history.d) if i != axis]
                positions_proj = positions_full[alive_idx][:, keep_dims]
                bounds_proj = _slice_bounds(bounds_full, keep_dims)
                if operators_override is None:
                    edge_mode_values: np.ndarray | None = None
                    if config.neighbor_weighting == "inv_geodesic_full":
                        edge_mode_values = geodesic_full
                    elif config.neighbor_weighting == "inv_geodesic_iso":
                        edge_mode_values = _compute_recorded_iso_edge_lengths(
                            positions=positions_proj,
                            edges=edges_full,
                            bounds=bounds_proj,
                            pbc=bool(history.pbc),
                        )
                    elif config.neighbor_weighting in RECORDED_EDGE_WEIGHT_MODES:
                        mode_values_global = _extract_recorded_edge_mode_values(
                            history,
                            frame_idx,
                            config.neighbor_weighting,
                            n_edges=edges_global.shape[0],
                        )
                        edge_mode_values = mode_values_global[local_edge_indices]

                    neighbor_indices_axis, neighbor_weights_axis, neighbor_mask_axis = (
                        _build_neighbor_data_dense(
                            edges=edges_full,
                            alive_idx=alive_idx,
                            positions=positions_proj,
                            bounds=bounds_proj,
                            volumes=volumes_full,
                            weight_mode=config.neighbor_weighting,
                            edge_mode_values=edge_mode_values,
                            max_neighbors=int(config.neighbor_k),
                            pbc=bool(history.pbc),
                            kernel_length_scale=config.kernel_length_scale,
                        )
                    )
                else:
                    neighbor_indices_axis = torch.empty(
                        len(alive_idx), 0, dtype=torch.long, device=positions_proj.device
                    )
                    neighbor_weights_axis = torch.empty(
                        len(alive_idx), 0, dtype=positions_proj.dtype, device=positions_proj.device
                    )
                    neighbor_mask_axis = torch.empty(
                        len(alive_idx), 0, dtype=torch.bool, device=positions_proj.device
                    )
                output = _compute_radial_output(
                    history,
                    config,
                    channels,
                    positions_proj,
                    alive_idx,
                    frame_idx=frame_idx,
                    keep_dims=keep_dims,
                    pair_i=pair_i,
                    pair_j=pair_j,
                    distances=distances,
                    bin_edges=bin_edges_3d,
                    neighbor_indices=neighbor_indices_axis,
                    neighbor_weights=neighbor_weights_axis,
                    neighbor_mask=neighbor_mask_axis,
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
