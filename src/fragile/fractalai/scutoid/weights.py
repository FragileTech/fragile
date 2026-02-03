"""Weight computation utilities for scutoid neighbor graphs."""

from typing import Literal

import torch
from torch import Tensor

from .hessian_estimation import compute_emergent_metric


WeightMode = Literal[
    "uniform",
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_volume",
    "inverse_riemannian_distance",
]


def _normalize_edge_weights(
    edge_index: Tensor,
    raw_weights: Tensor,
    n_nodes: int | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize edge weights so they sum to 1 per source node."""
    if raw_weights.numel() == 0:
        return raw_weights

    src = edge_index[0]
    if n_nodes is None:
        n_nodes = int(src.max().item()) + 1 if src.numel() > 0 else 0

    weight_sums = torch.zeros(n_nodes, device=raw_weights.device, dtype=raw_weights.dtype)
    weight_sums.scatter_add_(0, src, raw_weights)
    weight_sums = torch.clamp(weight_sums, min=eps)

    return raw_weights / weight_sums[src]


def _apply_alive_mask(
    edge_index: Tensor,
    raw_weights: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Zero-out weights where either endpoint is not alive."""
    if alive is None or raw_weights.numel() == 0:
        return raw_weights

    src, dst = edge_index
    valid_edges = alive[src] & alive[dst]
    return torch.where(valid_edges, raw_weights, torch.zeros_like(raw_weights))


def compute_uniform_weights(
    edge_index: Tensor,
    n_nodes: int | None = None,
    alive: Tensor | None = None,
    normalize: bool = True,
) -> Tensor:
    """Compute uniform neighbor weights."""
    raw_weights = torch.ones(edge_index.shape[1], device=edge_index.device)
    raw_weights = _apply_alive_mask(edge_index, raw_weights, alive)
    if not normalize:
        return raw_weights
    return _normalize_edge_weights(edge_index, raw_weights, n_nodes=n_nodes)


def compute_inverse_distance_weights(
    positions: Tensor,
    edge_index: Tensor,
    edge_distances: Tensor | None = None,
    alive: Tensor | None = None,
    eps: float = 1e-8,
    normalize: bool = True,
) -> Tensor:
    """Compute weights proportional to inverse Euclidean distance."""
    src, dst = edge_index
    if edge_distances is None:
        edge_distances = torch.norm(positions[dst] - positions[src], dim=1)

    raw_weights = 1.0 / (edge_distances + eps)
    raw_weights = _apply_alive_mask(edge_index, raw_weights, alive)
    if not normalize:
        return raw_weights
    return _normalize_edge_weights(edge_index, raw_weights, n_nodes=positions.shape[0])


def compute_inverse_volume_weights(
    edge_index: Tensor,
    cell_volumes: Tensor,
    alive: Tensor | None = None,
    eps: float = 1e-12,
    normalize: bool = True,
) -> Tensor:
    """Compute weights proportional to inverse destination cell volume."""
    if cell_volumes is None:
        raise ValueError("cell_volumes is required for inverse_volume weights.")

    dst = edge_index[1]
    raw_weights = 1.0 / (cell_volumes[dst] + eps)
    raw_weights = _apply_alive_mask(edge_index, raw_weights, alive)
    if not normalize:
        return raw_weights
    return _normalize_edge_weights(edge_index, raw_weights, n_nodes=cell_volumes.shape[0])


def compute_riemannian_volumes(
    cell_volumes: Tensor,
    metric_tensors: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Compute Riemannian cell volumes: V_i^R = V_i^E * sqrt(det g_i)."""
    if cell_volumes is None:
        raise ValueError("cell_volumes is required for Riemannian volumes.")

    sign, logdet = torch.linalg.slogdet(metric_tensors)
    if (sign <= 0).any():
        det = torch.det(metric_tensors)
        logdet = torch.where(sign > 0, logdet, torch.log(torch.clamp(det.abs(), min=eps)))

    sqrt_det_g = torch.exp(0.5 * logdet)
    return cell_volumes * sqrt_det_g


def clamp_metric_eigenvalues(
    metric_tensors: Tensor,
    min_eig: float | None = None,
    max_eig: float | None = None,
) -> Tensor:
    """Clamp eigenvalues of metric tensors (batched) for stability."""
    if min_eig is None and max_eig is None:
        return metric_tensors

    eigvals, eigvecs = torch.linalg.eigh(metric_tensors)
    if min_eig is not None:
        eigvals = torch.clamp(eigvals, min=min_eig)
    if max_eig is not None:
        eigvals = torch.clamp(eigvals, max=max_eig)

    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)


def compute_inverse_riemannian_volume_weights(
    edge_index: Tensor,
    cell_volumes: Tensor,
    metric_tensors: Tensor | None = None,
    riemannian_volumes: Tensor | None = None,
    positions: Tensor | None = None,
    alive: Tensor | None = None,
    eps: float = 1e-12,
    normalize: bool = True,
    symmetrize_metric: bool = True,
    min_eig: float | None = 1e-6,
    max_eig: float | None = None,
) -> Tensor:
    """Compute weights proportional to inverse Riemannian cell volume."""
    if riemannian_volumes is None:
        if metric_tensors is None:
            if positions is None:
                raise ValueError("metric_tensors or positions is required for Riemannian weights.")
            metric_tensors = compute_emergent_metric(positions, edge_index, alive)
        if symmetrize_metric:
            metric_tensors = 0.5 * (metric_tensors + metric_tensors.transpose(-1, -2))
        metric_tensors = clamp_metric_eigenvalues(
            metric_tensors, min_eig=min_eig, max_eig=max_eig
        )
        riemannian_volumes = compute_riemannian_volumes(cell_volumes, metric_tensors, eps=eps)

    dst = edge_index[1]
    raw_weights = 1.0 / (riemannian_volumes[dst] + eps)
    raw_weights = _apply_alive_mask(edge_index, raw_weights, alive)
    if not normalize:
        return raw_weights
    return _normalize_edge_weights(edge_index, raw_weights, n_nodes=riemannian_volumes.shape[0])


def compute_inverse_riemannian_distance_weights(
    positions: Tensor,
    edge_index: Tensor,
    metric_tensors: Tensor | None = None,
    alive: Tensor | None = None,
    eps: float = 1e-8,
    normalize: bool = True,
    symmetrize_metric: bool = True,
    min_eig: float | None = 1e-6,
    max_eig: float | None = None,
) -> Tensor:
    """Compute weights proportional to inverse Riemannian geodesic distance.

    The distance uses a symmetric edge metric (default):
        g_ij = 0.5 * (g_i + g_j),  d_g(i,j) = sqrt(Δx^T g_ij Δx)

    Eigenvalues are clamped to [min_eig, max_eig] for stability (set min_eig=None
    to skip clamping).
    """
    if metric_tensors is None:
        metric_tensors = compute_emergent_metric(positions, edge_index, alive)
    if symmetrize_metric:
        metric_tensors = 0.5 * (metric_tensors + metric_tensors.transpose(-1, -2))
    metric_tensors = clamp_metric_eigenvalues(metric_tensors, min_eig=min_eig, max_eig=max_eig)

    src, dst = edge_index
    delta = positions[dst] - positions[src]
    g_src = metric_tensors[src]
    g_dst = metric_tensors[dst]
    g_edge = 0.5 * (g_src + g_dst) if symmetrize_metric else g_src

    d_sq = torch.einsum("ei,eij,ej->e", delta, g_edge, delta)
    d_g = torch.sqrt(torch.clamp(d_sq, min=eps))

    raw_weights = 1.0 / (d_g + eps)
    raw_weights = _apply_alive_mask(edge_index, raw_weights, alive)
    if not normalize:
        return raw_weights
    return _normalize_edge_weights(edge_index, raw_weights, n_nodes=positions.shape[0])


def compute_edge_weights(
    positions: Tensor,
    edge_index: Tensor,
    mode: WeightMode = "inverse_distance",
    edge_distances: Tensor | None = None,
    cell_volumes: Tensor | None = None,
    metric_tensors: Tensor | None = None,
    riemannian_volumes: Tensor | None = None,
    alive: Tensor | None = None,
    eps: float = 1e-12,
    normalize: bool = True,
    riemannian_symmetrize: bool = True,
    riemannian_min_eig: float | None = 1e-6,
    riemannian_max_eig: float | None = None,
) -> Tensor:
    """Compute neighbor weights with a selected discrete scheme.

    Riemannian modes clamp metric eigenvalues by default (min_eig=1e-6).
    Pass riemannian_min_eig=None to disable clamping.
    """
    if mode == "uniform":
        return compute_uniform_weights(
            edge_index, n_nodes=positions.shape[0], alive=alive, normalize=normalize
        )
    if mode == "inverse_distance":
        return compute_inverse_distance_weights(
            positions,
            edge_index,
            edge_distances=edge_distances,
            alive=alive,
            eps=max(eps, 1e-8),
            normalize=normalize,
        )
    if mode == "inverse_volume":
        return compute_inverse_volume_weights(
            edge_index,
            cell_volumes=cell_volumes,
            alive=alive,
            eps=eps,
            normalize=normalize,
        )
    if mode == "inverse_riemannian_volume":
        return compute_inverse_riemannian_volume_weights(
            edge_index,
            cell_volumes=cell_volumes,
            metric_tensors=metric_tensors,
            riemannian_volumes=riemannian_volumes,
            positions=positions,
            alive=alive,
            eps=eps,
            normalize=normalize,
            symmetrize_metric=riemannian_symmetrize,
            min_eig=riemannian_min_eig,
            max_eig=riemannian_max_eig,
        )
    if mode == "inverse_riemannian_distance":
        return compute_inverse_riemannian_distance_weights(
            positions,
            edge_index,
            metric_tensors=metric_tensors,
            alive=alive,
            eps=max(eps, 1e-8),
            normalize=normalize,
            symmetrize_metric=riemannian_symmetrize,
            min_eig=riemannian_min_eig,
            max_eig=riemannian_max_eig,
        )

    raise ValueError(f"Unknown weight mode: {mode}")
