"""Experimental cached-hull Voronoi measures.

This module computes Voronoi cell volumes and ridge facet areas by building
one ConvexHull per bounded cell and reusing its facet decomposition. It is
intended for testing and debugging, and is not wired into the main pipeline.

The behavior matches the existing Voronoi volume and facet computations:
- Unbounded cells (regions with -1 vertices) yield volume 0 (later replaced
  by mean of non-zero volumes).
- Infinite/degenerate ridges default to area 1.0.
- 2D uses shoelace areas and ridge length, no hull caching.
- 3D+ uses ConvexHull for volumes; facet areas come from hull facets when
  possible, with fallback to ridge-vertex computation.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np
import torch
from torch import Tensor
from scipy.spatial import ConvexHull, Voronoi


def _simplex_measure(points: np.ndarray) -> float:
    """Return (d-1)-measure of a facet simplex in d dimensions."""
    if points.shape[0] <= 1:
        return 0.0
    vecs = points[1:] - points[0]
    gram = vecs @ vecs.T
    det = float(np.linalg.det(gram))
    if det < 0.0:
        det = 0.0
    m = points.shape[0] - 1
    return math.sqrt(det) / math.factorial(m)


def _canonical_plane_key(normal: np.ndarray, offset: float, decimals: int) -> tuple[float, ...]:
    """Normalize plane (n, b) and return a sign-invariant rounded key."""
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return tuple([0.0] * (normal.shape[0] + 1))
    normal_u = normal / norm
    offset_u = offset / norm

    idx = np.flatnonzero(np.abs(normal_u) > 0)
    if idx.size > 0 and normal_u[idx[0]] < 0:
        normal_u = -normal_u
        offset_u = -offset_u

    key_vals = np.concatenate([normal_u, [offset_u]])
    return tuple(np.round(key_vals, decimals=decimals))


def _facet_area_map_from_hull(
    vertices: np.ndarray, hull: ConvexHull, decimals: int
) -> dict[tuple[float, ...], float]:
    """Aggregate facet simplex areas by plane key."""
    area_by_plane: dict[tuple[float, ...], float] = {}
    for simplex, equation in zip(hull.simplices, hull.equations, strict=False):
        pts = vertices[simplex]
        area = _simplex_measure(pts)
        key = _canonical_plane_key(equation[:-1], equation[-1], decimals)
        area_by_plane[key] = area_by_plane.get(key, 0.0) + area
    return area_by_plane


def _facet_area_from_vertices(vertices: np.ndarray, d: int) -> float:
    """Fallback facet area matching existing behavior."""
    if d == 2:
        if len(vertices) >= 2:
            return float(np.linalg.norm(vertices[1] - vertices[0]))
        return 1.0
    if d == 3:
        if len(vertices) >= 3:
            v0 = vertices[0]
            total_area = 0.0
            for i in range(1, len(vertices) - 1):
                v1 = vertices[i]
                v2 = vertices[i + 1]
                cross = np.cross(v1 - v0, v2 - v0)
                total_area += 0.5 * float(np.linalg.norm(cross))
            return total_area
        return 1.0

    if len(vertices) < d:
        return 1.0
    try:
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except Exception:
        return 1.0


def _plane_key_from_sites(
    pos_i: np.ndarray, pos_j: np.ndarray, decimals: int
) -> tuple[float, ...] | None:
    """Return canonical plane key for the perpendicular bisector of pos_i/pos_j."""
    diff = pos_j - pos_i
    dist = float(np.linalg.norm(diff))
    if dist == 0.0:
        return None
    normal = diff / dist
    mid = 0.5 * (pos_i + pos_j)
    offset = -float(np.dot(normal, mid))
    return _canonical_plane_key(normal, offset, decimals)


def compute_cached_volumes_and_facet_areas(
    vor: Voronoi,
    positions: np.ndarray,
    ridge_points: np.ndarray,
    ridge_vertices: list[list[int]],
    n_alive: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    plane_decimals: int = 7,
) -> tuple[Tensor, Tensor]:
    """Compute volumes and facet areas using one hull per bounded cell.

    Args:
        vor: scipy Voronoi object
        positions: [N, d] positions used for Voronoi (alive only)
        ridge_points: [n_ridges, 2] neighbor pairs (alive indices)
        ridge_vertices: list of ridge vertex index lists (same order as ridge_points)
        n_alive: number of alive cells
        d: spatial dimension
        device: target device for tensors
        dtype: target dtype for tensors
        plane_decimals: rounding precision for plane key matching

    Returns:
        (cell_volumes, facet_areas) as torch tensors
    """
    volumes = np.zeros(n_alive, dtype=np.float64)
    facet_maps: list[dict[tuple[float, ...], float] | None] = [None] * n_alive

    for i in range(n_alive):
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]

        if -1 in vertices_idx or len(vertices_idx) < d + 1:
            volumes[i] = 0.0
            continue

        try:
            vertices = vor.vertices[vertices_idx]
            if d == 2:
                x = vertices[:, 0]
                y = vertices[:, 1]
                volumes[i] = 0.5 * float(
                    np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                )
            else:
                hull = ConvexHull(vertices)
                volumes[i] = float(hull.volume)
                facet_maps[i] = _facet_area_map_from_hull(vertices, hull, plane_decimals)
        except Exception:
            volumes[i] = 0.0

    mean_vol = volumes[volumes > 0].mean() if (volumes > 0).any() else 1.0
    volumes[volumes == 0] = mean_vol
    volumes_t = torch.from_numpy(volumes).to(device=device, dtype=dtype)

    n_ridges = len(ridge_points)
    areas_half = np.zeros(n_ridges, dtype=np.float64)

    for k, (i, j) in enumerate(ridge_points):
        ridge_v = ridge_vertices[k]
        if -1 in ridge_v or len(ridge_v) < d:
            areas_half[k] = 1.0
            continue

        area = None
        if d >= 3:
            key = _plane_key_from_sites(positions[i], positions[j], plane_decimals)
            if key is not None:
                for cell_idx in (i, j):
                    fmap = facet_maps[cell_idx]
                    if fmap is not None and key in fmap:
                        area = fmap[key]
                        break

        if area is None:
            try:
                vertices = vor.vertices[ridge_v]
                area = _facet_area_from_vertices(vertices, d)
            except Exception:
                area = 1.0

        areas_half[k] = area

    areas = np.concatenate([areas_half, areas_half])
    areas_t = torch.from_numpy(areas).to(device=device, dtype=dtype)
    return volumes_t, areas_t
