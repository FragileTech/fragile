"""Delaunay-based scutoid utilities (no convex hull volumes).

This module computes Delaunay neighbors, fast Hessian estimates, and
Riemannian volume weights from det(g) without Voronoi cell volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.spatial import Delaunay
import torch
from torch import Tensor

from .hessian_estimation import (
    compute_emergent_metric,
    estimate_hessian_diagonal_fd,
    estimate_hessian_full_fd,
)
from .neighbors import build_csr_from_coo, query_walker_neighbors_vectorized
from .ricci import (
    compute_ricci_proxy,
    compute_ricci_tensor_proxy_full_metric,
)
from .weights import clamp_metric_eigenvalues, compute_edge_weights, WeightMode


@dataclass
class DelaunayScutoidData:
    """Container for Delaunay neighbor + metric data."""

    # Inputs (alive-only)
    positions: Tensor  # [N_alive, d]
    fitness: Tensor  # [N_alive]
    alive: Tensor  # [N_alive] (all True)
    alive_indices: Tensor  # [N_alive] original indices

    # Graph
    edge_index: Tensor  # [2, E] alive indexing
    edge_index_full: Tensor  # [2, E] original indexing
    csr_ptr: Tensor  # [N_alive + 1]
    csr_indices: Tensor  # [E]
    neighbor_matrix: Tensor  # [N_alive, K]
    neighbor_mask: Tensor  # [N_alive, K]
    neighbor_counts: Tensor  # [N_alive]

    # Geometry
    edge_distances: Tensor  # [E] Euclidean
    edge_geodesic_distances: Tensor  # [E] Riemannian

    # Hessian / metric
    hessian_diag: Tensor | None  # [N_alive, d]
    hessian_full: Tensor | None  # [N_alive, d, d]
    hessian_valid_mask: Tensor  # [N_alive]
    metric_tensors: Tensor  # [N_alive, d, d]
    metric_det: Tensor  # [N_alive]
    riemannian_volume_weights: Tensor  # [N_alive] = sqrt(det g)
    diffusion_tensors: Tensor  # [N_alive, d, d]

    # Weights on edges
    edge_weights: dict[str, Tensor]

    # Ricci proxy (scalar curvature)
    ricci_proxy: Tensor  # [N_alive]
    ricci_proxy_full: Tensor | None = None  # [N_alive]
    ricci_tensor_full: Tensor | None = None  # [N_alive, d, d] anisotropic proxy


def _build_delaunay_edges(positions: np.ndarray) -> np.ndarray:
    """Compute symmetric Delaunay edges from numpy positions."""
    n, d = positions.shape
    if n < d + 1:
        return np.zeros((0, 2), dtype=np.int64)

    try:
        delaunay = Delaunay(positions)
    except Exception:
        return np.zeros((0, 2), dtype=np.int64)

    simplices = np.asarray(delaunay.simplices, dtype=np.int64)
    if simplices.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Vectorized edge extraction from simplex vertices.
    n_vertices = simplices.shape[1]
    pair_i, pair_j = np.triu_indices(n_vertices, k=1)
    src = simplices[:, pair_i].reshape(-1)
    dst = simplices[:, pair_j].reshape(-1)

    edges = np.stack([src, dst], axis=1)
    edges_rev = edges[:, [1, 0]]
    edges_all = np.concatenate([edges, edges_rev], axis=0)

    # Remove duplicates without Python loops.
    return np.unique(edges_all, axis=0)


def _compute_metric_from_hessian(
    hessian_diag: Tensor | None,
    hessian_full: Tensor | None,
    epsilon_sigma: float,
    min_eig: float | None,
    max_eig: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build metric tensors and determinant from Hessian estimates."""
    if hessian_full is not None:
        hess = torch.nan_to_num(hessian_full, nan=0.0, posinf=0.0, neginf=0.0)
        d = hess.shape[-1]
        eye = torch.eye(d, device=hess.device, dtype=hess.dtype).unsqueeze(0)
        metric = hess + epsilon_sigma * eye
        metric = 0.5 * (metric + metric.transpose(-1, -2))
        metric = clamp_metric_eigenvalues(metric, min_eig=min_eig, max_eig=max_eig)

        det, volume = _compute_det_and_volume(metric)
        return metric, det, volume

    if hessian_diag is None:
        msg = "Need either hessian_diag or hessian_full to build metric."
        raise ValueError(msg)

    hess_diag = torch.nan_to_num(hessian_diag, nan=0.0, posinf=0.0, neginf=0.0)
    metric_diag = hess_diag + epsilon_sigma
    if min_eig is not None:
        metric_diag = torch.clamp(metric_diag, min=min_eig)
    if max_eig is not None:
        metric_diag = torch.clamp(metric_diag, max=max_eig)

    metric = torch.diag_embed(metric_diag)
    det = metric_diag.prod(dim=1)
    volume = torch.sqrt(torch.clamp(det, min=1e-12))
    return metric, det, volume


def _compute_det_and_volume(metric_tensors: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Compute determinant and sqrt(det) volume weights for metric tensors."""
    sign, logdet = torch.linalg.slogdet(metric_tensors)
    if (sign <= 0).any():
        det = torch.det(metric_tensors)
        logdet = torch.where(sign > 0, logdet, torch.log(torch.clamp(det.abs(), min=eps)))
    det = torch.exp(logdet)
    volume = torch.sqrt(torch.clamp(det, min=eps))
    return det, volume


def _compute_diffusion_tensor(metric_tensors: Tensor, min_eig: float = 1e-6) -> Tensor:
    """Compute diffusion tensor Σ = g^{-1/2} (without c2 scaling)."""
    eigvals, eigvecs = torch.linalg.eigh(metric_tensors)
    eigvals = torch.clamp(eigvals, min=min_eig)
    inv_sqrt = torch.rsqrt(eigvals)
    return eigvecs @ torch.diag_embed(inv_sqrt) @ eigvecs.transpose(-1, -2)


def _compute_geodesic_distances(
    positions: Tensor,
    edge_index: Tensor,
    metric_tensors: Tensor,
) -> Tensor:
    """Compute Riemannian geodesic distances along edges (edge-averaged metric)."""
    if edge_index.numel() == 0:
        return torch.empty(0, device=positions.device, dtype=positions.dtype)

    src, dst = edge_index
    delta = positions[dst] - positions[src]
    g_edge = 0.5 * (metric_tensors[src] + metric_tensors[dst])

    d_sq = torch.einsum("ei,eij,ej->e", delta, g_edge, delta)
    return torch.sqrt(torch.clamp(d_sq, min=1e-12))


def compute_delaunay_scutoid(
    positions: Tensor,
    fitness_values: Tensor,
    alive: Tensor | None = None,
    spatial_dims: int | None = None,
    metric_mode: Literal["covariance", "hessian"] = "covariance",
    hessian_mode: Literal["diagonal", "full", "none"] = "none",
    epsilon_sigma: float = 1e-3,
    min_eig: float | None = 1e-6,
    max_eig: float | None = None,
    weight_modes: Iterable[WeightMode] | None = None,
    normalize_weights: bool = True,
    compute_full_ricci: bool = False,
) -> DelaunayScutoidData:
    """Compute Delaunay neighbors, metric tensors, and curvature proxies.

    Defaults:
    - Metric from neighbor covariance (full emergent metric)
    - Neighbor weights from inverse Riemannian (geodesic) distance
    - Riemannian volumes via det(g) (no convex hull volumes)
    - Ricci scalar proxy only (full tensor off by default)
    """
    device = positions.device

    if alive is None:
        alive_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=device)
    else:
        alive_mask = alive

    alive_indices = torch.where(alive_mask)[0]
    if alive_indices.numel() == 0:
        msg = "No alive walkers for Delaunay computation."
        raise ValueError(msg)

    if spatial_dims is not None:
        pos_alive = positions[alive_mask, :spatial_dims]
    else:
        pos_alive = positions[alive_mask]
    fit_alive = fitness_values[alive_mask]

    positions_np = pos_alive.detach().cpu().numpy()
    edges_np = _build_delaunay_edges(positions_np)

    if edges_np.shape[0] == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.as_tensor(edges_np, dtype=torch.long, device=device).t()

    edge_index_full = edge_index.clone()
    if edge_index_full.numel() > 0:
        edge_index_full[0] = alive_indices[edge_index[0]]
        edge_index_full[1] = alive_indices[edge_index[1]]

    csr = build_csr_from_coo(edge_index, n_nodes=pos_alive.shape[0])
    neighbor_matrix, neighbor_mask, neighbor_counts = query_walker_neighbors_vectorized(
        csr["csr_ptr"], csr["csr_indices"]
    )

    # Hessian estimation (fast finite differences on Delaunay neighbors)
    hessian_diag = None
    hessian_full = None
    hessian_valid = torch.ones(pos_alive.shape[0], dtype=torch.bool, device=device)

    if hessian_mode == "full":
        hess_result = estimate_hessian_full_fd(
            pos_alive,
            fit_alive,
            None,
            edge_index,
            method="central",
            symmetrize=True,
        )
        hessian_full = hess_result["hessian_tensors"]
        hessian_valid = hess_result["valid_mask"]
    elif hessian_mode == "diagonal":
        hess_result = estimate_hessian_diagonal_fd(
            pos_alive,
            fit_alive,
            edge_index,
        )
        hessian_diag = hess_result["hessian_diagonal"]
        hessian_valid = hess_result["valid_mask"]
    elif hessian_mode != "none":
        raise ValueError(f"Unknown hessian_mode: {hessian_mode}")

    if metric_mode == "covariance":
        metric_tensors = compute_emergent_metric(pos_alive, edge_index, alive=None)
        metric_tensors = 0.5 * (metric_tensors + metric_tensors.transpose(-1, -2))
        metric_tensors = clamp_metric_eigenvalues(metric_tensors, min_eig=min_eig, max_eig=max_eig)
        metric_det, volume_weights = _compute_det_and_volume(metric_tensors)
    elif metric_mode == "hessian":
        if hessian_mode == "none":
            msg = "hessian_mode must not be 'none' when metric_mode='hessian'."
            raise ValueError(msg)
        metric_tensors, metric_det, volume_weights = _compute_metric_from_hessian(
            hessian_diag=hessian_diag,
            hessian_full=hessian_full,
            epsilon_sigma=epsilon_sigma,
            min_eig=min_eig,
            max_eig=max_eig,
        )
    else:
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    # Edge distances
    if edge_index.numel() == 0:
        edge_distances = torch.empty(0, device=device, dtype=pos_alive.dtype)
    else:
        src, dst = edge_index
        edge_distances = torch.norm(pos_alive[dst] - pos_alive[src], dim=1)

    edge_geodesic = _compute_geodesic_distances(pos_alive, edge_index, metric_tensors)

    diffusion_tensors = _compute_diffusion_tensor(metric_tensors, min_eig=min_eig or 1e-6)

    # Edge weights for requested modes
    if weight_modes is None:
        weight_modes = ("inverse_riemannian_distance",)

    # Use unit Euclidean cell volumes (relative det(g) only)
    cell_volumes = torch.ones(pos_alive.shape[0], device=device, dtype=pos_alive.dtype)

    edge_weights: dict[str, Tensor] = {}
    for mode in weight_modes:
        edge_weights[mode] = compute_edge_weights(
            pos_alive,
            edge_index,
            mode=mode,
            edge_distances=edge_distances,
            cell_volumes=cell_volumes,
            metric_tensors=metric_tensors,
            riemannian_volumes=volume_weights,
            normalize=normalize_weights,
        )

    ricci_weights = edge_weights.get("inverse_riemannian_distance")
    if ricci_weights is None:
        if edge_index.numel() == 0:
            ricci_weights = torch.empty(0, device=device, dtype=pos_alive.dtype)
        else:
            raw = 1.0 / (edge_geodesic + 1e-8)
            src = edge_index[0]
            sums = torch.zeros(pos_alive.shape[0], device=device, dtype=raw.dtype)
            sums.scatter_add_(0, src, raw)
            sums = torch.clamp(sums, min=1e-12)
            ricci_weights = raw / sums[src]

    ricci_proxy = compute_ricci_proxy(
        metric_det=metric_det,
        edge_index=edge_index,
        edge_weights=ricci_weights,
        spatial_dim=pos_alive.shape[1],
    )

    ricci_proxy_full = None
    if compute_full_ricci:
        ricci_tensor_full, ricci_proxy_full = compute_ricci_tensor_proxy_full_metric(
            positions=pos_alive,
            metric_tensors=metric_tensors,
            edge_index=edge_index,
            edge_weights=ricci_weights,
            metric_det=metric_det,
            spatial_dim=pos_alive.shape[1],
        )
    else:
        ricci_tensor_full = None

    return DelaunayScutoidData(
        positions=pos_alive,
        fitness=fit_alive,
        alive=torch.ones(pos_alive.shape[0], dtype=torch.bool, device=device),
        alive_indices=alive_indices,
        edge_index=edge_index,
        edge_index_full=edge_index_full,
        csr_ptr=csr["csr_ptr"],
        csr_indices=csr["csr_indices"],
        neighbor_matrix=neighbor_matrix,
        neighbor_mask=neighbor_mask,
        neighbor_counts=neighbor_counts,
        edge_distances=edge_distances,
        edge_geodesic_distances=edge_geodesic,
        hessian_diag=hessian_diag,
        hessian_full=hessian_full,
        hessian_valid_mask=hessian_valid,
        metric_tensors=metric_tensors,
        metric_det=metric_det,
        riemannian_volume_weights=volume_weights,
        diffusion_tensors=diffusion_tensors,
        edge_weights=edge_weights,
        ricci_proxy=ricci_proxy,
        ricci_proxy_full=ricci_proxy_full,
        ricci_tensor_full=ricci_tensor_full,
    )
