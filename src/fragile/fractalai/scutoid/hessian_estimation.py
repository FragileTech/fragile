"""Hessian estimation from fitness values using finite differences and geometric methods.

Provides two complementary approaches:
1. Finite differences (primary): Direct second-order derivatives from fitness
2. Geometric (validation): H = g - ε_Σ I from neighbor covariance
"""

from typing import Literal
import warnings

import torch
from torch import Tensor

from .utils import (
    validate_finite_difference_inputs,
)


# -----------------------------------------------------------------------------
# Neighbor preparation helpers (COO/CSR)
# -----------------------------------------------------------------------------


def _prepare_edge_index(
    positions: Tensor,
    edge_index: Tensor | None,
    csr_ptr: Tensor | None,
    csr_indices: Tensor | None,
    csr_types: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Resolve edge_index and neighbor counts from COO or CSR inputs."""
    device = positions.device
    n_walkers = positions.shape[0]

    if edge_index is not None:
        if csr_ptr is not None or csr_indices is not None:
            raise ValueError("Provide either edge_index or csr_ptr/csr_indices, not both.")
        if edge_index.device != device:
            raise ValueError("edge_index must be on the same device as positions.")
        if edge_index.numel() == 0:
            return edge_index, torch.zeros(n_walkers, dtype=torch.long, device=device)
        if torch.any(edge_index[0] >= n_walkers) or torch.any(edge_index[1] >= n_walkers):
            raise ValueError(
                "edge_index contains indices outside positions. "
                "Filter boundary neighbors before calling."
            )
        num_neighbors = torch.bincount(edge_index[0], minlength=n_walkers)
        return edge_index, num_neighbors

    if csr_ptr is None or csr_indices is None:
        raise ValueError("edge_index or csr_ptr/csr_indices must be provided.")
    if csr_ptr.device != device or csr_indices.device != device:
        raise ValueError("CSR tensors must be on the same device as positions.")
    if csr_ptr.numel() - 1 < n_walkers:
        raise ValueError("csr_ptr has fewer rows than walkers in positions.")

    edge_end = int(csr_ptr[n_walkers].item())
    ptr = csr_ptr[: n_walkers + 1]
    indices = csr_indices[:edge_end]
    types = csr_types[:edge_end] if csr_types is not None else None

    counts = ptr[1:] - ptr[:-1]
    src = torch.repeat_interleave(torch.arange(n_walkers, device=device), counts)

    if types is not None:
        mask = types == 0
        src = src[mask]
        dst = indices[mask]
        num_neighbors = torch.bincount(src, minlength=n_walkers)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, num_neighbors

    if torch.any(indices >= n_walkers):
        raise ValueError(
            "csr_types required to filter boundary neighbors for Hessian estimation."
        )

    edge_index = torch.stack([src, indices], dim=0)
    return edge_index, counts


# -----------------------------------------------------------------------------
# Vectorized neighbor geometry
# -----------------------------------------------------------------------------


def _build_neighbor_matrix(edge_index: Tensor, n_nodes: int) -> tuple[Tensor, Tensor]:
    """Build padded neighbor matrix and mask from edge_index."""
    device = edge_index.device
    if edge_index.numel() == 0:
        neighbors = torch.empty(n_nodes, 0, dtype=torch.long, device=device)
        mask = torch.empty(n_nodes, 0, dtype=torch.bool, device=device)
        return neighbors, mask

    src = edge_index[0]
    dst = edge_index[1]
    sorted_idx = torch.argsort(src)
    src = src[sorted_idx]
    dst = dst[sorted_idx]
    counts = torch.bincount(src, minlength=n_nodes)
    max_degree = int(counts.max().item()) if counts.numel() > 0 else 0

    if max_degree == 0:
        neighbors = torch.empty(n_nodes, 0, dtype=torch.long, device=device)
        mask = torch.empty(n_nodes, 0, dtype=torch.bool, device=device)
        return neighbors, mask

    row_indices = torch.repeat_interleave(torch.arange(n_nodes, device=device), counts)
    edge_pos = torch.arange(dst.numel(), device=device) - torch.repeat_interleave(
        torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=counts.dtype), counts]), dim=0
        )[:-1],
        counts,
    )

    neighbors = torch.full((n_nodes, max_degree), -1, dtype=torch.long, device=device)
    mask = torch.zeros((n_nodes, max_degree), dtype=torch.bool, device=device)
    neighbors[row_indices, edge_pos] = dst
    mask[row_indices, edge_pos] = True
    return neighbors, mask


def _compute_neighbor_geometry(
    positions: Tensor,
    neighbors: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute neighbor displacements, distances, and normalized directions."""
    device = positions.device
    n_nodes, max_degree = neighbors.shape
    if max_degree == 0:
        displacements = torch.empty(n_nodes, 0, positions.shape[1], device=device)
        distances = torch.empty(n_nodes, 0, device=device)
        directions = torch.empty(n_nodes, 0, positions.shape[1], device=device)
        return displacements, distances, directions

    pos_i = positions.unsqueeze(1)  # [N, 1, d]
    safe_neighbors = neighbors.clamp(min=0)
    pos_j = positions[safe_neighbors]  # [N, K, d]

    displacements = pos_j - pos_i
    displacements = torch.where(mask.unsqueeze(-1), displacements, torch.zeros_like(displacements))

    distances = torch.norm(displacements, dim=2)
    directions = displacements / (distances.unsqueeze(-1) + 1e-10)

    return displacements, distances, directions


def _select_axis_neighbors(
    positions: Tensor,
    neighbors: Tensor,
    mask: Tensor,
    directions: Tensor,
    max_angle_deg: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select closest positive and negative neighbors per axis (vectorized)."""
    device = positions.device
    n_nodes, max_degree = neighbors.shape
    d = positions.shape[1]

    if max_degree == 0:
        empty_idx = torch.full((n_nodes, d), -1, dtype=torch.long, device=device)
        empty_delta = torch.full((n_nodes, d), float("nan"), device=device)
        return empty_idx, empty_idx.clone(), empty_delta, empty_delta.clone()

    cos_threshold = torch.cos(torch.tensor(max_angle_deg * torch.pi / 180.0, device=device))

    axis_cos = directions  # [N, K, d]

    aligned = mask.unsqueeze(-1) & (axis_cos.abs() >= cos_threshold)
    positive = aligned & (axis_cos > 0)
    negative = aligned & (axis_cos < 0)

    neighbor_coords = positions[neighbors.clamp(min=0)]  # [N, K, d]
    center_coords = positions.unsqueeze(1)  # [N, 1, d]
    deltas = neighbor_coords - center_coords  # [N, K, d]

    delta_axis = deltas[..., torch.arange(d, device=device)]  # [N, K, d]

    large = torch.full_like(delta_axis, float("inf"))
    pos_cost = torch.where(positive, delta_axis.abs(), large)
    neg_cost = torch.where(negative, delta_axis.abs(), large)

    pos_idx = pos_cost.argmin(dim=1)  # [N, d]
    neg_idx = neg_cost.argmin(dim=1)  # [N, d]

    batch_idx = torch.arange(n_nodes, device=device)[:, None]
    axis_idx = torch.arange(d, device=device)[None, :]

    pos_valid = positive[batch_idx, pos_idx, axis_idx]
    neg_valid = negative[batch_idx, neg_idx, axis_idx]

    pos_neighbors = neighbors[batch_idx, pos_idx]
    neg_neighbors = neighbors[batch_idx, neg_idx]

    pos_neighbors = torch.where(pos_valid, pos_neighbors, torch.full_like(pos_neighbors, -1))
    neg_neighbors = torch.where(neg_valid, neg_neighbors, torch.full_like(neg_neighbors, -1))

    pos_delta = delta_axis[batch_idx, pos_idx, axis_idx]
    neg_delta = delta_axis[batch_idx, neg_idx, axis_idx]

    pos_delta = torch.where(pos_valid, pos_delta, torch.full_like(pos_delta, float("nan")))
    neg_delta = torch.where(neg_valid, neg_delta, torch.full_like(neg_delta, float("nan")))

    return pos_neighbors, neg_neighbors, pos_delta, neg_delta


# -----------------------------------------------------------------------------
# Public APIs
# -----------------------------------------------------------------------------


def estimate_hessian_diagonal_fd(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor | None,
    alive: Tensor | None = None,
    step_size: float | None = None,
    min_neighbors_per_axis: int = 2,
    csr_ptr: Tensor | None = None,
    csr_indices: Tensor | None = None,
    csr_types: Tensor | None = None,
    max_angle_deg: float = 30.0,
) -> dict[str, Tensor]:
    """Estimate diagonal Hessian elements using second-order finite differences.

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness at each position
        edge_index: [2, E] neighbor connectivity (COO). Optional if CSR provided.
        alive: [N] optional validity mask
        step_size: Optional manual step size (used for reporting)
        min_neighbors_per_axis: Minimum neighbors needed along each axis
        csr_ptr: Optional [N+1] CSR row pointers
        csr_indices: Optional [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)
        max_angle_deg: Angular threshold for axial alignment
    """
    N, d = positions.shape
    device = positions.device

    edge_index, num_neighbors = _prepare_edge_index(
        positions, edge_index, csr_ptr, csr_indices, csr_types
    )

    diagnostics = validate_finite_difference_inputs(
        positions,
        fitness_values,
        edge_index,
        alive,
        min_neighbors=min_neighbors_per_axis * d,
        num_neighbors=num_neighbors,
    )

    neighbors, mask = _build_neighbor_matrix(edge_index, N)
    _, _, directions = _compute_neighbor_geometry(positions, neighbors, mask)

    pos_idx, neg_idx, pos_delta, neg_delta = _select_axis_neighbors(
        positions, neighbors, mask, directions, max_angle_deg=max_angle_deg
    )

    hessian_diagonal = torch.full((N, d), float("nan"), device=device, dtype=positions.dtype)
    step_sizes_used = torch.full((N, d), float("nan"), device=device, dtype=positions.dtype)
    axis_quality = torch.zeros((N, d), device=device, dtype=positions.dtype)

    has_pos = pos_idx >= 0
    has_neg = neg_idx >= 0
    valid_axis = has_pos & has_neg

    if valid_axis.any():
        pos_vals = fitness_values[pos_idx.clamp(min=0)]
        neg_vals = fitness_values[neg_idx.clamp(min=0)]
        center_vals = fitness_values.unsqueeze(1)

        h_pos = pos_delta.abs()
        h_neg = neg_delta.abs()
        h_avg = 0.5 * (h_pos + h_neg)

        symmetry = 1.0 - (h_pos - h_neg).abs() / (h_pos + h_neg + 1e-10)

        hessian_est = (pos_vals + neg_vals - 2.0 * center_vals) / (h_avg**2 + 1e-10)

        hessian_diagonal = torch.where(valid_axis, hessian_est, hessian_diagonal)
        if step_size is None:
            step_sizes_used = torch.where(valid_axis, h_avg, step_sizes_used)
        else:
            step_sizes_used = torch.where(
                valid_axis,
                torch.full_like(h_avg, step_size),
                step_sizes_used,
            )
        axis_quality = torch.where(valid_axis, symmetry, axis_quality)

    n_valid_axes = torch.isfinite(hessian_diagonal).sum(dim=1)
    valid_mask = n_valid_axes >= (d // 2)

    if alive is not None:
        valid_mask &= alive

    valid_mask &= diagnostics["valid_walkers"]

    invalid = ~valid_mask
    if invalid.any():
        hessian_diagonal[invalid] = float("nan")
        step_sizes_used[invalid] = float("nan")
        axis_quality[invalid] = float("nan")

    return {
        "hessian_diagonal": hessian_diagonal,
        "eigenvalues": hessian_diagonal,
        "step_sizes": step_sizes_used,
        "axis_quality": axis_quality,
        "valid_mask": valid_mask,
    }


def estimate_hessian_full_fd(
    positions: Tensor,
    fitness_values: Tensor,
    gradient_vectors: Tensor | None,
    edge_index: Tensor | None,
    alive: Tensor | None = None,
    step_size: float | None = None,
    method: Literal["central", "gradient_fd"] = "central",
    symmetrize: bool = True,
    csr_ptr: Tensor | None = None,
    csr_indices: Tensor | None = None,
    csr_types: Tensor | None = None,
    max_angle_deg: float = 30.0,
) -> dict[str, Tensor]:
    """Estimate full Hessian matrix using second-order finite differences.

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness values
        gradient_vectors: [N, d] optional pre-computed gradients
        edge_index: [2, E] neighbor connectivity (COO). Optional if CSR provided.
        alive: [N] optional validity mask
        step_size: Optional manual step size (used for reporting)
        method: "central" or "gradient_fd"
        symmetrize: If True, enforce symmetry by averaging H and H^T
        csr_ptr: Optional [N+1] CSR row pointers
        csr_indices: Optional [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)
        max_angle_deg: Angular threshold for axial alignment
    """
    N, d = positions.shape
    device = positions.device

    if method == "gradient_fd" and gradient_vectors is None:
        raise ValueError("gradient_fd method requires gradient_vectors input")

    edge_index, num_neighbors = _prepare_edge_index(
        positions, edge_index, csr_ptr, csr_indices, csr_types
    )

    if method == "central":
        hessian_tensors = _estimate_hessian_central_differences(
            positions,
            fitness_values,
            edge_index,
            alive,
            max_angle_deg=max_angle_deg,
        )
    else:
        hessian_tensors = _estimate_hessian_from_gradient_fd(
            positions,
            gradient_vectors,
            edge_index,
            alive,
            max_angle_deg=max_angle_deg,
        )

    symmetry_errors = torch.norm(
        hessian_tensors - hessian_tensors.transpose(1, 2),
        p="fro",
        dim=(1, 2),
    )

    if symmetrize:
        hessian_tensors = 0.5 * (hessian_tensors + hessian_tensors.transpose(1, 2))

    eigenvalues = torch.linalg.eigvalsh(hessian_tensors)

    lambda_max = eigenvalues[:, -1].abs()
    lambda_min = eigenvalues[:, 0].abs()
    condition_numbers = lambda_max / (lambda_min + 1e-10)

    psd_mask = (eigenvalues >= -1e-6).all(dim=1)
    psd_fraction = psd_mask.float().mean().item()

    eigenvalues = torch.flip(eigenvalues, dims=[1])

    valid_mask = torch.ones(N, dtype=torch.bool, device=device)
    if alive is not None:
        valid_mask &= alive

    valid_mask &= num_neighbors >= 2

    return {
        "hessian_tensors": hessian_tensors,
        "hessian_eigenvalues": eigenvalues,
        "condition_numbers": condition_numbers,
        "symmetry_error": symmetry_errors,
        "psd_fraction": psd_fraction,
        "valid_mask": valid_mask,
    }


# -----------------------------------------------------------------------------
# Internal Hessian estimators
# -----------------------------------------------------------------------------


def _estimate_hessian_central_differences(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    alive: Tensor | None,
    max_angle_deg: float,
) -> Tensor:
    """Estimate Hessian using central differences (diagonal only)."""
    N, d = positions.shape
    device = positions.device

    neighbors, mask = _build_neighbor_matrix(edge_index, N)
    _, _, directions = _compute_neighbor_geometry(positions, neighbors, mask)

    pos_idx, neg_idx, pos_delta, neg_delta = _select_axis_neighbors(
        positions, neighbors, mask, directions, max_angle_deg=max_angle_deg
    )

    H = torch.zeros(N, d, d, device=device, dtype=positions.dtype)

    has_pos = pos_idx >= 0
    has_neg = neg_idx >= 0
    valid_axis = has_pos & has_neg

    if valid_axis.any():
        pos_vals = fitness_values[pos_idx.clamp(min=0)]
        neg_vals = fitness_values[neg_idx.clamp(min=0)]
        center_vals = fitness_values.unsqueeze(1)

        h_pos = pos_delta.abs()
        h_neg = neg_delta.abs()
        h_avg = 0.5 * (h_pos + h_neg)

        diag_est = (pos_vals + neg_vals - 2.0 * center_vals) / (h_avg**2 + 1e-10)

        diag_values = torch.where(valid_axis, diag_est, torch.zeros_like(diag_est))
        H[:, torch.arange(d), torch.arange(d)] = diag_values

    if alive is not None:
        invalid = ~alive
        H[invalid] = float("nan")

    return H


def _estimate_hessian_from_gradient_fd(
    positions: Tensor,
    gradient_vectors: Tensor,
    edge_index: Tensor,
    alive: Tensor | None,
    max_angle_deg: float,
) -> Tensor:
    """Estimate Hessian from finite differences of gradients (vectorized)."""
    N, d = positions.shape
    device = positions.device

    neighbors, mask = _build_neighbor_matrix(edge_index, N)
    _, _, directions = _compute_neighbor_geometry(positions, neighbors, mask)

    pos_idx, _, pos_delta, _ = _select_axis_neighbors(
        positions, neighbors, mask, directions, max_angle_deg=max_angle_deg
    )

    H = torch.full((N, d, d), float("nan"), device=device, dtype=positions.dtype)

    valid_pos = pos_idx >= 0
    if valid_pos.any():
        grad_j = gradient_vectors[pos_idx.clamp(min=0)]  # [N, d, d]
        grad_i = gradient_vectors.unsqueeze(1).expand(-1, d, -1)

        delta_x = pos_delta  # [N, d]
        delta_x = delta_x.unsqueeze(2)  # [N, d, 1]

        H_est = (grad_j - grad_i) / (delta_x + 1e-10)

        valid_mask = valid_pos.unsqueeze(2)
        H = torch.where(valid_mask, H_est, H)

    if alive is not None:
        H[~alive] = float("nan")

    return H


# -----------------------------------------------------------------------------
# Metric-based Hessian
# -----------------------------------------------------------------------------


def estimate_hessian_from_metric(
    positions: Tensor,
    edge_index: Tensor | None,
    epsilon_sigma: float = 0.1,
    alive: Tensor | None = None,
    metric_tensors: Tensor | None = None,
    validate_equilibrium: bool = True,
    csr_ptr: Tensor | None = None,
    csr_indices: Tensor | None = None,
    csr_types: Tensor | None = None,
) -> dict[str, Tensor]:
    """Estimate Hessian from emergent metric: H = g - ε_Σ I.

    Args:
        positions: [N, d] walker positions
        edge_index: [2, E] neighbor connectivity (COO). Optional if CSR provided.
        epsilon_sigma: Physical spectral floor ε_Σ
        alive: [N] optional validity mask
        metric_tensors: Optional precomputed metrics
        validate_equilibrium: If True, compute equilibrium score
        csr_ptr: Optional [N+1] CSR row pointers
        csr_indices: Optional [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)
    """
    N, d = positions.shape
    device = positions.device

    if metric_tensors is None:
        edge_index, _ = _prepare_edge_index(
            positions, edge_index, csr_ptr, csr_indices, csr_types
        )
        metric_tensors = _compute_emergent_metric(positions, edge_index, alive)

    identity = torch.eye(d, device=device, dtype=positions.dtype).unsqueeze(0).expand(N, d, d)
    hessian_tensors = metric_tensors - epsilon_sigma * identity

    eigenvalues = torch.linalg.eigvalsh(hessian_tensors)

    psd_violation_mask = (eigenvalues[:, 0] < -1e-6)

    eigenvalues = torch.flip(eigenvalues, dims=[1])

    if validate_equilibrium:
        equilibrium_score = _compute_equilibrium_score(positions, edge_index, metric_tensors, alive)
    else:
        equilibrium_score = torch.ones(N, device=device, dtype=positions.dtype)

    if psd_violation_mask.any():
        n_violations = psd_violation_mask.sum().item()
        warnings.warn(
            f"Geometric Hessian has {n_violations}/{N} walkers with negative eigenvalues. "
            "This suggests walkers are not in equilibrium or epsilon_sigma is too large."
        )

    return {
        "hessian_tensors": hessian_tensors,
        "hessian_eigenvalues": eigenvalues,
        "metric_tensors": metric_tensors,
        "psd_violation_mask": psd_violation_mask,
        "equilibrium_score": equilibrium_score,
    }


def compute_emergent_metric(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor | None = None,
) -> Tensor:
    """Public wrapper for emergent metric computation."""
    return _compute_emergent_metric(positions, edge_index, alive)


def _compute_emergent_metric(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Compute emergent metric g from neighbor covariance (vectorized)."""
    N, d = positions.shape
    device = positions.device

    src, dst = edge_index

    displacements = positions[dst] - positions[src]  # [E, d]
    outer_products = displacements[:, :, None] * displacements[:, None, :]  # [E, d, d]

    cov_flat = torch.zeros(N, d * d, device=device, dtype=positions.dtype)
    cov_flat.scatter_add_(0, src[:, None].expand(-1, d * d), outer_products.reshape(-1, d * d))
    covariances = cov_flat.view(N, d, d)

    neighbor_counts = torch.bincount(src, minlength=N).to(positions.dtype)
    neighbor_counts = torch.clamp(neighbor_counts, min=1.0)
    covariances = covariances / neighbor_counts.view(N, 1, 1)

    epsilon_numerical = 1e-5
    identity = torch.eye(d, device=device, dtype=positions.dtype).unsqueeze(0)
    covariances = covariances + epsilon_numerical * identity

    metrics = torch.linalg.pinv(covariances)

    if alive is not None:
        metrics = torch.where(
            alive.view(N, 1, 1), metrics, torch.full_like(metrics, float("nan"))
        )

    return metrics


def _compute_equilibrium_score(
    positions: Tensor,
    edge_index: Tensor,
    metric_tensors: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Compute equilibrium quality score based on metric isotropy."""
    N = positions.shape[0]

    eigenvalues = torch.linalg.eigvalsh(metric_tensors)

    condition_numbers = eigenvalues[:, -1] / (eigenvalues[:, 0] + 1e-10)
    isotropy_scores = 1.0 / (1.0 + condition_numbers)

    if alive is not None:
        isotropy_scores = torch.where(
            alive,
            isotropy_scores,
            torch.full_like(isotropy_scores, float("nan")),
        )

    return isotropy_scores
