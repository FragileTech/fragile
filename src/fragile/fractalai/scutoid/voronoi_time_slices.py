"""Time-sliced Voronoi/Delaunay tessellation with homogeneous Euclidean-time bins.

This module enforces uniform time slabs along a chosen Euclidean-time dimension
by binning walkers into contiguous time slices, optionally merging sparse bins
into neighbors, and recomputing spatial Voronoi/Delaunay tessellations per bin.

The resulting structure provides:
- Spacelike neighbors (within-bin Delaunay edges)
- Timelike neighbors (cross-bin Delaunay edges between adjacent bins)
- Per-bin Voronoi data (spatial-only tessellation)

Example:
    >>> result = compute_time_sliced_voronoi(
    ...     positions=x, time_dim=3, n_bins=24, min_walkers_bin=10, bounds=bounds
    ... )
    >>> spacelike_edges_bin0 = result.bins[0].spacelike_edges
    >>> timelike_edges = result.timelike_edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import Delaunay
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation


@dataclass
class TimeSliceResult:
    """Per-bin tessellation outputs."""

    bin_index: int
    t_start: float
    t_end: float
    indices: np.ndarray
    voronoi_data: dict[str, Any]
    spacelike_edges: np.ndarray


@dataclass
class TimeSlicedVoronoi:
    """Aggregated tessellation results for time-sliced Voronoi geometry."""

    bins: list[TimeSliceResult]
    bin_edges: np.ndarray
    walker_bin: np.ndarray
    timelike_edges: np.ndarray
    time_dim: int
    spatial_dims: list[int]
    spatial_bounds: TorchBounds | None


def _resolve_time_range(
    bounds: TorchBounds | None,
    time_dim: int,
    values: np.ndarray,
) -> tuple[float, float]:
    if bounds is not None and hasattr(bounds, "low") and hasattr(bounds, "high"):
        low = bounds.low[time_dim]
        high = bounds.high[time_dim]
        low = float(low.detach().cpu().numpy() if torch.is_tensor(low) else low)
        high = float(high.detach().cpu().numpy() if torch.is_tensor(high) else high)
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return low, high
    data_min = float(np.min(values)) if values.size > 0 else 0.0
    data_max = float(np.max(values)) if values.size > 0 else 1.0
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        data_min = 0.0
        data_max = 1.0
    return data_min, data_max


def _spatial_bounds_from(
    bounds: TorchBounds | None, spatial_dims: list[int]
) -> TorchBounds | None:
    if bounds is None or not hasattr(bounds, "low") or not hasattr(bounds, "high"):
        return None
    low = bounds.low
    high = bounds.high
    low_vals = low[spatial_dims]
    high_vals = high[spatial_dims]
    return TorchBounds(low=low_vals, high=high_vals, shape=low_vals.shape)


def _build_bins(
    time_values: np.ndarray,
    alive_indices: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    n_bins = len(edges) - 1
    if n_bins <= 0 or alive_indices.size == 0:
        return []
    bin_ids = np.digitize(time_values, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        idx = alive_indices[bin_ids == i]
        bins.append({"start": float(edges[i]), "end": float(edges[i + 1]), "indices": idx})
    return bins


def _merge_bins(
    bins: list[dict[str, Any]],
    min_walkers_bin: int,
) -> list[dict[str, Any]]:
    if min_walkers_bin <= 1:
        return bins
    if not bins:
        return []

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(bins):
        current = bins[i]
        indices = current["indices"]
        start = current["start"]
        end = current["end"]

        if i == 0 and indices.size < min_walkers_bin:
            j = i + 1
            while indices.size < min_walkers_bin and j < len(bins):
                end = bins[j]["end"]
                indices = np.concatenate([indices, bins[j]["indices"]])
                j += 1
            merged.append({"start": start, "end": end, "indices": indices})
            i = j
            continue

        if indices.size < min_walkers_bin and merged:
            merged[-1]["end"] = end
            merged[-1]["indices"] = np.concatenate([merged[-1]["indices"], indices])
        else:
            merged.append({"start": start, "end": end, "indices": indices})
        i += 1
    return merged


def _build_delaunay_edges(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0 or points.shape[0] < points.shape[1] + 1:
        return np.zeros((0, 2), dtype=np.int64)
    try:
        delaunay = Delaunay(points)
    except Exception:
        return np.zeros((0, 2), dtype=np.int64)

    edges: set[tuple[int, int]] = set()
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a = int(indices[simplex[i]])
                b = int(indices[simplex[j]])
                edges.add((a, b))
                edges.add((b, a))

    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)


def _build_timelike_edges(
    bins: list[dict[str, Any]],
    spatial_positions: np.ndarray,
) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for i in range(len(bins) - 1):
        idx_a = bins[i]["indices"]
        idx_b = bins[i + 1]["indices"]
        if idx_a.size == 0 or idx_b.size == 0:
            continue
        union_idx = np.concatenate([idx_a, idx_b])
        union_points = spatial_positions[union_idx]
        if union_points.shape[0] < union_points.shape[1] + 1:
            continue
        try:
            delaunay = Delaunay(union_points)
        except Exception:
            continue
        is_b = np.zeros(union_idx.shape[0], dtype=bool)
        is_b[len(idx_a) :] = True
        for simplex in delaunay.simplices:
            for a in range(len(simplex)):
                for b in range(a + 1, len(simplex)):
                    ia = int(simplex[a])
                    ib = int(simplex[b])
                    if is_b[ia] == is_b[ib]:
                        continue
                    ga = int(union_idx[ia])
                    gb = int(union_idx[ib])
                    edges.add((ga, gb))
                    edges.add((gb, ga))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)


def compute_time_sliced_voronoi(
    positions: torch.Tensor,
    time_dim: int,
    n_bins: int,
    min_walkers_bin: int = 1,
    bounds: TorchBounds | None = None,
    alive: torch.Tensor | None = None,
    pbc: bool = False,
    pbc_mode: str = "mirror",
    exclude_boundary: bool = True,
    boundary_tolerance: float = 1e-6,
    compute_curvature: bool = True,
) -> TimeSlicedVoronoi:
    """Recompute Voronoi tessellations with homogeneous Euclidean-time slices.

    Args:
        positions: Walker positions [N, d].
        time_dim: Dimension index to treat as Euclidean time.
        n_bins: Number of uniform time bins before merging.
        min_walkers_bin: Minimum walkers per bin; sparse bins merge with neighbors.
        bounds: Optional bounds (time bin range uses bounds for time_dim if available).
        alive: Optional alive mask [N]; defaults to all True.
        pbc: Whether to use periodic boundaries for spatial tessellation.
        pbc_mode: PBC mode for Voronoi ("mirror", "replicate", "ignore").
        exclude_boundary: Whether to drop boundary cells from neighbor lists.
        boundary_tolerance: Distance threshold for boundary detection.
        compute_curvature: Whether to compute curvature proxies per bin.

    Returns:
        TimeSlicedVoronoi containing per-bin Voronoi data, spacelike edges, and
        timelike edges between adjacent bins.
    """
    if not torch.is_tensor(positions):
        positions = torch.as_tensor(positions)
    n, d = positions.shape
    if time_dim < 0 or time_dim >= d:
        raise ValueError(f"time_dim must be in [0, {d - 1}], got {time_dim}")

    if alive is None:
        alive_mask = torch.ones(n, dtype=torch.bool, device=positions.device)
    else:
        alive_mask = alive.to(dtype=torch.bool, device=positions.device)

    alive_indices = torch.where(alive_mask)[0].detach().cpu().numpy()
    if alive_indices.size == 0:
        return TimeSlicedVoronoi(
            bins=[],
            bin_edges=np.array([]),
            walker_bin=np.full(n, -1, dtype=np.int64),
            timelike_edges=np.zeros((0, 2), dtype=np.int64),
            time_dim=time_dim,
            spatial_dims=[i for i in range(d) if i != time_dim],
            spatial_bounds=_spatial_bounds_from(bounds, [i for i in range(d) if i != time_dim]),
        )

    time_values = positions[alive_indices, time_dim].detach().cpu().numpy()
    low, high = _resolve_time_range(bounds, time_dim, time_values)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(time_values))
        high = float(np.max(time_values))
        if high <= low:
            high = low + 1.0

    edges = np.linspace(low, high, num=max(1, int(n_bins)) + 1)
    bins = _build_bins(time_values, alive_indices, edges)

    spatial_dims = [i for i in range(d) if i != time_dim]
    min_required = max(1, int(min_walkers_bin), len(spatial_dims) + 1)
    bins = _merge_bins(bins, min_required)

    spatial_positions = positions[:, spatial_dims].detach().cpu().numpy()
    spatial_bounds = _spatial_bounds_from(bounds, spatial_dims)

    walker_bin = np.full(n, -1, dtype=np.int64)
    bin_edges: list[float] = []
    slice_results: list[TimeSliceResult] = []

    for bin_index, bin_info in enumerate(bins):
        idx = np.asarray(bin_info["indices"], dtype=np.int64)
        if idx.size > 0:
            walker_bin[idx] = bin_index
        if not bin_edges:
            bin_edges.append(bin_info["start"])
        bin_edges.append(bin_info["end"])

        alive_bin = torch.zeros(n, dtype=torch.bool, device=positions.device)
        if idx.size > 0:
            alive_bin[idx] = True

        voronoi_data = compute_voronoi_tessellation(
            positions=positions[:, spatial_dims],
            alive=alive_bin,
            bounds=spatial_bounds,
            pbc=pbc,
            pbc_mode=pbc_mode,
            exclude_boundary=exclude_boundary,
            boundary_tolerance=boundary_tolerance,
            compute_curvature=compute_curvature,
        )

        points_bin = spatial_positions[idx] if idx.size > 0 else np.zeros((0, len(spatial_dims)))
        spacelike_edges = _build_delaunay_edges(points_bin, idx)

        slice_results.append(
            TimeSliceResult(
                bin_index=bin_index,
                t_start=float(bin_info["start"]),
                t_end=float(bin_info["end"]),
                indices=idx,
                voronoi_data=voronoi_data,
                spacelike_edges=spacelike_edges,
            )
        )

    timelike_edges = _build_timelike_edges(bins, spatial_positions)

    return TimeSlicedVoronoi(
        bins=slice_results,
        bin_edges=np.asarray(bin_edges, dtype=float) if bin_edges else np.array([]),
        walker_bin=walker_bin,
        timelike_edges=timelike_edges,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
        spatial_bounds=spatial_bounds,
    )
