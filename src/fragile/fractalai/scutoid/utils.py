"""Utility functions for gradient and Hessian estimation on scutoid geometry.

Provides helper functions for:
- Step size estimation
- Edge weight computation
- Axial neighbor finding
"""

import torch
from torch import Tensor
from typing import Literal


def estimate_optimal_step_size(
    positions: Tensor,
    edge_index: Tensor,
    target_fraction: float = 0.5,
    alive: Tensor | None = None,
) -> Tensor:
    """Estimate per-walker step size as fraction of nearest neighbor distance.

    The step size is crucial for finite-difference accuracy. Too large leads to
    truncation error, too small amplifies numerical noise. This function uses
    the nearest neighbor distance as a natural scale.

    Args:
        positions: [N, d] walker positions
        edge_index: [2, E] neighbor connectivity graph
        target_fraction: Fraction of nearest neighbor distance to use (default: 0.5)
        alive: [N] optional boolean mask for valid walkers

    Returns:
        [N] optimal step size for each walker

    Algorithm:
        For each walker i:
            h_i = target_fraction * min_{j∈N(i)} ||x_j - x_i||

    Typical values:
        - 0.3-0.5: Conservative, good for mixed derivatives
        - 0.5-1.0: Aggressive, good for diagonal Hessian
    """
    N = positions.shape[0]
    device = positions.device

    # Initialize with large value
    min_distances = torch.full((N,), float("inf"), device=device)

    # Compute edge distances
    src, dst = edge_index
    edge_distances = torch.norm(positions[dst] - positions[src], dim=1)

    # Find minimum distance per source walker
    min_distances.scatter_reduce_(
        0, src, edge_distances, reduce="amin", include_self=False
    )

    # Handle isolated walkers (no neighbors)
    isolated_mask = torch.isinf(min_distances)
    if isolated_mask.any():
        # Fallback: use median of all edge distances
        median_dist = torch.median(edge_distances)
        min_distances[isolated_mask] = median_dist

    # Apply alive mask if provided
    if alive is not None:
        min_distances = torch.where(alive, min_distances, torch.nan)

    return target_fraction * min_distances


def compute_edge_weights(
    positions: Tensor,
    edge_index: Tensor,
    mode: Literal["uniform", "inverse_distance", "gaussian"] = "inverse_distance",
    kernel_bandwidth: float = 1.0,
    alive: Tensor | None = None,
) -> Tensor:
    """Compute weights for finite-difference averaging over neighbors.

    Different weighting schemes balance accuracy vs noise:
    - Uniform: Simple average, robust to outliers
    - Inverse distance: Closer neighbors weighted more, good default
    - Gaussian: Smooth falloff, good for irregular spacing

    Args:
        positions: [N, d] walker positions
        edge_index: [2, E] neighbor connectivity
        mode: Weighting scheme to use
        kernel_bandwidth: Bandwidth σ for Gaussian kernel (ignored for other modes)
        alive: [N] optional mask for valid walkers

    Returns:
        [E] normalized weights per edge (sum to 1 per source walker)

    Weighting formulas:
        - uniform: w_ij = 1/k where k = |N(i)|
        - inverse_distance: w_ij ∝ 1/||x_j - x_i||
        - gaussian: w_ij ∝ exp(-||x_j - x_i||²/(2σ²))
    """
    src, dst = edge_index
    E = edge_index.shape[1]
    N = positions.shape[0]
    device = positions.device

    # Compute edge distances
    edge_vectors = positions[dst] - positions[src]  # [E, d]
    edge_distances = torch.norm(edge_vectors, dim=1)  # [E]

    # Identify valid edges (both endpoints alive)
    if alive is not None:
        valid_edges = alive[src] & alive[dst]
    else:
        valid_edges = torch.ones(E, dtype=torch.bool, device=device)

    # Compute raw weights based on mode
    if mode == "uniform":
        raw_weights = torch.ones(E, device=device)
    elif mode == "inverse_distance":
        # Add small epsilon to avoid division by zero
        raw_weights = 1.0 / (edge_distances + 1e-8)
    elif mode == "gaussian":
        raw_weights = torch.exp(-edge_distances**2 / (2 * kernel_bandwidth**2))
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    # Set weights to zero for invalid edges
    raw_weights = torch.where(valid_edges, raw_weights, torch.zeros_like(raw_weights))

    # Normalize weights per source walker (so they sum to 1)
    weight_sums = torch.zeros(N, device=device)
    weight_sums.scatter_add_(0, src, raw_weights)

    # Avoid division by zero for isolated walkers
    weight_sums = torch.clamp(weight_sums, min=1e-8)

    # Normalize
    normalized_weights = raw_weights / weight_sums[src]

    return normalized_weights


def find_axial_neighbors(
    positions: Tensor,
    edge_index: Tensor,
    walker_idx: int,
    axis: int,
    max_angle_deg: float = 30.0,
) -> tuple[list[int], list[int]]:
    """Find neighbors approximately aligned with a coordinate axis.

    Used for computing diagonal Hessian elements via central differences:
        ∂²V/∂x_α² ≈ [V(x+h·e_α) + V(x-h·e_α) - 2V(x)] / h²

    Args:
        positions: [N, d] walker positions
        edge_index: [2, E] neighbor graph
        walker_idx: Index of central walker
        axis: Coordinate axis (0, 1, ..., d-1)
        max_angle_deg: Maximum angle deviation from axis in degrees

    Returns:
        (positive_neighbors, negative_neighbors):
            - positive_neighbors: Indices of walkers in +e_α direction
            - negative_neighbors: Indices of walkers in -e_α direction

    Algorithm:
        For each neighbor j:
            1. Compute displacement: Δx = x_j - x_i
            2. Compute angle with axis: θ = arccos(Δx·e_α / ||Δx||)
            3. If θ < max_angle: classify as positive/negative based on sign(Δx_α)
    """
    src, dst = edge_index
    d = positions.shape[1]

    # Find edges originating from walker_idx
    neighbor_mask = src == walker_idx
    neighbor_indices = dst[neighbor_mask].tolist()

    if len(neighbor_indices) == 0:
        return [], []

    # Get displacements to neighbors
    center_pos = positions[walker_idx]  # [d]
    neighbor_pos = positions[neighbor_indices]  # [k, d]
    displacements = neighbor_pos - center_pos  # [k, d]

    # Normalize displacements
    distances = torch.norm(displacements, dim=1, keepdim=True)  # [k, 1]
    directions = displacements / (distances + 1e-10)  # [k, d]

    # Create unit vector along axis
    axis_vector = torch.zeros(d, device=positions.device)
    axis_vector[axis] = 1.0

    # Compute angles with axis
    cos_angles = directions @ axis_vector  # [k]
    angles_deg = torch.acos(torch.clamp(cos_angles, -1, 1)) * 180 / torch.pi

    # Filter by angle threshold
    aligned_mask = angles_deg <= max_angle_deg

    # Separate into positive and negative direction
    positive_neighbors = []
    negative_neighbors = []

    for i, (idx, is_aligned, displacement) in enumerate(
        zip(neighbor_indices, aligned_mask, displacements)
    ):
        if not is_aligned:
            continue

        # Check sign along axis
        if displacement[axis] > 0:
            positive_neighbors.append(idx)
        else:
            negative_neighbors.append(idx)

    return positive_neighbors, negative_neighbors


def validate_finite_difference_inputs(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    alive: Tensor | None = None,
    min_neighbors: int = 3,
) -> dict[str, Tensor]:
    """Validate inputs for finite-difference estimation and compute diagnostics.

    Checks for common issues:
    - NaN or infinite values
    - Isolated walkers (too few neighbors)
    - Degenerate configurations

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness values
        edge_index: [2, E] neighbor graph
        alive: [N] optional validity mask
        min_neighbors: Minimum neighbors required per walker

    Returns:
        dict with diagnostic information:
            - "valid_walkers": [N] bool mask of walkers suitable for FD
            - "num_neighbors": [N] neighbor count per walker
            - "has_nan_fitness": [N] bool mask
            - "has_nan_position": [N] bool mask
            - "is_isolated": [N] bool mask (too few neighbors)

    Raises:
        ValueError: If critical issues detected (all walkers invalid, etc.)
    """
    N, d = positions.shape
    device = positions.device

    # Initialize masks
    valid_walkers = torch.ones(N, dtype=torch.bool, device=device)

    # Check for NaN/inf in fitness
    has_nan_fitness = ~torch.isfinite(fitness_values)
    valid_walkers &= ~has_nan_fitness

    # Check for NaN/inf in positions
    has_nan_position = ~torch.isfinite(positions).all(dim=1)
    valid_walkers &= ~has_nan_position

    # Count neighbors per walker
    src, dst = edge_index
    num_neighbors = torch.zeros(N, dtype=torch.long, device=device)
    num_neighbors.scatter_add_(0, src, torch.ones_like(src))

    # Check for isolated walkers
    is_isolated = num_neighbors < min_neighbors
    valid_walkers &= ~is_isolated

    # Apply alive mask if provided
    if alive is not None:
        valid_walkers &= alive

    # Check if we have any valid walkers left
    n_valid = valid_walkers.sum().item()
    if n_valid == 0:
        raise ValueError(
            "No valid walkers for finite-difference estimation. "
            f"NaN fitness: {has_nan_fitness.sum()}, "
            f"NaN positions: {has_nan_position.sum()}, "
            f"Isolated: {is_isolated.sum()}"
        )

    return {
        "valid_walkers": valid_walkers,
        "num_neighbors": num_neighbors,
        "has_nan_fitness": has_nan_fitness,
        "has_nan_position": has_nan_position,
        "is_isolated": is_isolated,
        "n_valid": n_valid,
        "n_total": N,
    }
