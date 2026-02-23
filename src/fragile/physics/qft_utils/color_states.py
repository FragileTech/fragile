"""Color state computation from RunHistory data.

Ported from fragile.fractalai.qft.aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.physics.fractal_gas.history import RunHistory


def compute_color_states_batch(
    history: RunHistory,
    start_idx: int,
    h_eff: float,
    mass: float,
    ell0: float,
    end_idx: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute color states for all timesteps from start_idx onward.

    Vectorized across T dimension.

    Args:
        history: RunHistory object.
        start_idx: Starting time index.
        h_eff: Effective Planck constant.
        mass: Particle mass.
        ell0: Length scale.
        end_idx: Ending time index (exclusive). None = use all recorded frames.

    Returns:
        Tuple of (color [T, N, d], valid [T, N]).
    """
    n_recorded = end_idx if end_idx is not None else history.n_recorded
    n_recorded - start_idx

    # Extract batched tensors
    v_pre = history.v_before_clone[start_idx:n_recorded]  # [T, N, d]
    force_visc = history.force_viscous[start_idx - 1 : n_recorded - 1]  # [T, N, d]

    phase = (mass * v_pre * ell0) / h_eff
    complex_phase = torch.polar(torch.ones_like(phase), phase.float())

    if force_visc.dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64

    tilde = force_visc.to(complex_dtype) * complex_phase.to(complex_dtype)
    norm = torch.linalg.vector_norm(tilde, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    color = tilde / norm
    valid = norm.squeeze(-1) > 1e-12

    return color, valid


def estimate_ell0(
    history: RunHistory,
) -> float:
    """Estimate ell0 from median Euclidean companion distance at mid-point.

    Uses Euclidean distances between companion pairs, matching the fractalai
    implementation.

    Args:
        history: RunHistory object.

    Returns:
        Estimated ell0 value.
    """
    mid_idx = history.n_recorded // 2
    if mid_idx == 0:
        return 1.0

    comp_idx = history.companions_distance[mid_idx - 1]
    return _euclidean_companion_distance(history, mid_idx, comp_idx)


def estimate_ell0_geodesic_edges(history: RunHistory) -> float:
    """Mean geodesic IG edge length averaged over equilibrium frames.

    Uses ``history.geodesic_edge_distances`` (list[Tensor]).
    Averages over second half of recorded frames.  Falls back to
    :func:`estimate_ell0` if geodesic edges are not recorded.

    Args:
        history: RunHistory object.

    Returns:
        Estimated ell0 value.
    """
    geodesic = getattr(history, "geodesic_edge_distances", None)
    if geodesic is None or len(geodesic) == 0:
        return estimate_ell0(history)

    n = len(geodesic)
    start = n // 2
    if start >= n:
        start = 0

    means: list[float] = []
    for i in range(start, n):
        g = geodesic[i]
        if not torch.is_tensor(g) or g.numel() == 0:
            continue
        vals = g.float()
        finite = torch.isfinite(vals) & (vals > 0)
        if finite.any():
            means.append(float(vals[finite].mean().item()))

    if not means:
        return estimate_ell0(history)
    return float(sum(means) / len(means))


def estimate_ell0_euclidean_edges(history: RunHistory) -> float:
    """Mean Euclidean IG edge length averaged over equilibrium frames.

    Uses ``history.neighbor_edges`` and ``history.x_before_clone`` to compute
    Euclidean distances along Delaunay edges.  Averages over second half of
    recorded frames.  Falls back to :func:`estimate_ell0` if neighbor edges
    are not recorded.

    Args:
        history: RunHistory object.

    Returns:
        Estimated ell0 value.
    """
    neighbor_edges = getattr(history, "neighbor_edges", None)
    if neighbor_edges is None or len(neighbor_edges) == 0:
        return estimate_ell0(history)

    n = len(neighbor_edges)
    start = n // 2
    if start >= n:
        start = 0

    means: list[float] = []
    for i in range(start, n):
        edges = neighbor_edges[i]
        if not torch.is_tensor(edges) or edges.numel() == 0:
            continue
        if edges.ndim != 2 or edges.shape[1] != 2:
            continue
        if i >= history.x_before_clone.shape[0]:
            continue
        x = history.x_before_clone[i]
        src = edges[:, 0].long()
        dst = edges[:, 1].long()
        max_idx = x.shape[0]
        valid = (src >= 0) & (src < max_idx) & (dst >= 0) & (dst < max_idx) & (src != dst)
        if not valid.any():
            continue
        diff = x[src[valid]] - x[dst[valid]]
        dist = torch.linalg.vector_norm(diff, dim=-1)
        finite = torch.isfinite(dist) & (dist > 0)
        if finite.any():
            means.append(float(dist[finite].mean().item()))

    if not means:
        return estimate_ell0(history)
    return float(sum(means) / len(means))


_ELL0_METHODS = ("companion", "geodesic_edges", "euclidean_edges")


def estimate_ell0_auto(history: RunHistory, method: str = "companion") -> float:
    """Dispatch to the selected ell0 estimation method.

    Args:
        history: RunHistory object.
        method: One of ``"companion"`` (median companion distance),
            ``"geodesic_edges"`` (mean geodesic IG edge length), or
            ``"euclidean_edges"`` (mean Euclidean IG edge length).

    Returns:
        Estimated ell0 value.
    """
    if method == "geodesic_edges":
        return estimate_ell0_geodesic_edges(history)
    if method == "euclidean_edges":
        return estimate_ell0_euclidean_edges(history)
    return estimate_ell0(history)


def _euclidean_companion_distance(
    history: RunHistory,
    mid_idx: int,
    comp_idx: Tensor,
) -> float:
    """Compute median Euclidean distance to companions at *mid_idx*."""
    x_pre = history.x_before_clone[mid_idx]
    diff = x_pre - x_pre[comp_idx]
    dist = torch.linalg.vector_norm(diff, dim=-1)
    if dist.numel() > 0:
        return float(dist.median().item())
    return 1.0
