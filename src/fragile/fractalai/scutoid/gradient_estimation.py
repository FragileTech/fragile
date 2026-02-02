"""Finite-difference gradient estimation from fitness values on scutoid geometry.

Implements weighted finite-difference methods to estimate ∇V_fit from:
- Fitness values V(x_i) at walker positions
- Neighbor graph structure
"""

import torch
from torch import Tensor
from typing import Literal

from .utils import (
    compute_edge_weights,
    validate_finite_difference_inputs,
)


def estimate_gradient_finite_difference(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    alive: Tensor | None = None,
    weighting_mode: Literal["uniform", "inverse_distance", "gaussian"] = "inverse_distance",
    kernel_bandwidth: float = 1.0,
    validate_inputs: bool = True,
) -> dict[str, Tensor]:
    """Estimate fitness gradient using weighted finite differences on neighbors.

    Core algorithm (per walker i):
        ∇V(x_i) ≈ Σ_{j∈N(i)} w_ij · [V(x_j) - V(x_i)] · (x_j - x_i) / ||x_j - x_i||²

    This is a first-order accurate approximation using neighbor fitness differences.
    The weighting scheme controls how different neighbors contribute.

    Args:
        positions: [N, d] walker positions in d-dimensional space
        fitness_values: [N] fitness/potential V(x_i) at each position
        edge_index: [2, E] neighbor connectivity graph (COO format)
        alive: [N] optional boolean mask for valid walkers
        weighting_mode: How to weight neighbor contributions:
            - "uniform": Simple average (robust, lower accuracy)
            - "inverse_distance": Weight by 1/distance (recommended default)
            - "gaussian": Smooth kernel weighting
        kernel_bandwidth: Bandwidth σ for Gaussian weighting (if used)
        validate_inputs: If True, check for NaN/isolated walkers

    Returns:
        dict containing:
            - "gradient": [N, d] estimated ∇V at each walker
            - "gradient_magnitude": [N] ||∇V|| Euclidean norm
            - "num_neighbors": [N] number of neighbors used
            - "estimation_quality": [N] variance of directional derivatives
                (low variance = consistent estimates from all neighbors)
            - "valid_mask": [N] boolean mask of walkers with valid estimates

    Complexity: O(E) = O(N·k) where k = average neighbors per walker

    Example:
        >>> positions = torch.randn(100, 2)
        >>> fitness = (positions**2).sum(dim=1)  # Quadratic potential
        >>> edge_index = build_knn_graph(positions, k=10)
        >>> result = estimate_gradient_finite_difference(
        ...     positions, fitness, edge_index
        ... )
        >>> gradient = result["gradient"]  # [100, 2]
        >>> # For quadratic: ∇V = 2x, so should be ≈ 2*positions
    """
    N, d = positions.shape
    device = positions.device

    # Validate inputs if requested
    if validate_inputs:
        diagnostics = validate_finite_difference_inputs(
            positions, fitness_values, edge_index, alive, min_neighbors=2
        )
        valid_mask = diagnostics["valid_walkers"]
        num_neighbors = diagnostics["num_neighbors"]
    else:
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
        if alive is not None:
            valid_mask &= alive
        # Count neighbors
        src, dst = edge_index
        num_neighbors = torch.zeros(N, dtype=torch.long, device=device)
        num_neighbors.scatter_add_(0, src, torch.ones_like(src))

    # Compute edge weights
    edge_weights = compute_edge_weights(
        positions, edge_index, mode=weighting_mode,
        kernel_bandwidth=kernel_bandwidth, alive=alive
    )

    # Extract source and destination indices
    src, dst = edge_index

    # Compute fitness differences
    fitness_diffs = fitness_values[dst] - fitness_values[src]  # [E]

    # Compute position displacements
    displacements = positions[dst] - positions[src]  # [E, d]
    distances_sq = (displacements**2).sum(dim=1)  # [E], needed for quality metrics

    # For gradient estimation, we use weighted least squares:
    # Minimize Σ w_ij (∇V·Δx_ij - ΔV_ij)²
    # Solution: ∇V = (Σ w_ij Δx Δx^T)^(-1) (Σ w_ij Δx ΔV)

    # Build per-walker systems
    gradient = torch.zeros(N, d, device=device)

    for i in range(N):
        # Find edges from walker i
        walker_edges = src == i
        if walker_edges.sum() < 2:
            continue

        # Get neighbor displacements and fitness differences
        Δx = displacements[walker_edges]  # [k, d]
        ΔV = fitness_diffs[walker_edges]   # [k]
        w = edge_weights[walker_edges]     # [k]

        # Weighted covariance: Σ w_ij Δx Δx^T
        # Shape: [d, d]
        W_diag = torch.diag(w)  # [k, k]
        A = Δx.T @ W_diag @ Δx   # [d, d]

        # Weighted cross-covariance: Σ w_ij Δx ΔV
        # Shape: [d]
        b = Δx.T @ (w * ΔV)      # [d]

        # Solve: A @ grad = b
        # Add small regularization for numerical stability
        A_reg = A + 1e-6 * torch.eye(d, device=device)

        try:
            gradient[i] = torch.linalg.solve(A_reg, b)
        except RuntimeError:
            # Singular matrix - use pseudo-inverse
            gradient[i] = torch.linalg.lstsq(A_reg, b).solution

    # Set invalid walkers to NaN
    gradient[~valid_mask] = float("nan")

    # Compute gradient magnitude
    gradient_magnitude = torch.norm(gradient, dim=1)

    # Estimate quality: variance of directional derivatives
    # For each walker, compute directional derivatives along each neighbor
    # and measure their variance (should be small if consistent)
    directional_derivatives = fitness_diffs / torch.sqrt(distances_sq + 1e-10)  # [E]

    # Compute variance per source walker
    estimation_quality = torch.zeros(N, device=device)

    # For each walker, compute variance of its neighbor derivatives
    for i in range(N):
        if not valid_mask[i]:
            estimation_quality[i] = float("nan")
            continue

        walker_edges = src == i
        if walker_edges.sum() < 2:
            estimation_quality[i] = float("nan")
            continue

        walker_derivatives = directional_derivatives[walker_edges]
        estimation_quality[i] = torch.var(walker_derivatives)

    return {
        "gradient": gradient,
        "gradient_magnitude": gradient_magnitude,
        "num_neighbors": num_neighbors,
        "estimation_quality": estimation_quality,
        "valid_mask": valid_mask,
    }


def compute_directional_derivative(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    direction: Tensor,
    alive: Tensor | None = None,
    weighting_mode: Literal["uniform", "inverse_distance", "gaussian"] = "inverse_distance",
) -> Tensor:
    """Compute directional derivative ∂V/∂n along specified direction(s).

    Uses finite differences projected onto the given direction:
        ∂V/∂n (x_i) ≈ Σ_{j∈N(i)} w_ij · [V(x_j) - V(x_i)] · n̂·(x_j - x_i) / ||x_j - x_i||

    Useful for:
    - Checking gradient quality (compare with gradient·direction)
    - Computing derivatives along specific axes
    - Validating estimation consistency

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness values
        edge_index: [2, E] neighbor graph
        direction: [d] or [N, d] direction vector(s) to compute derivative along
            If [d]: same direction for all walkers
            If [N, d]: per-walker directions (will be normalized)
        alive: [N] optional validity mask
        weighting_mode: Edge weighting scheme

    Returns:
        [N] directional derivatives ∂V/∂n for each walker

    Example:
        >>> # Compute derivative along x-axis
        >>> direction = torch.tensor([1.0, 0.0])
        >>> dV_dx = compute_directional_derivative(
        ...     positions, fitness, edge_index, direction
        ... )
    """
    N, d = positions.shape
    device = positions.device

    # Normalize direction(s)
    if direction.ndim == 1:
        # Single direction for all walkers
        direction = direction / (torch.norm(direction) + 1e-10)
        direction = direction.unsqueeze(0).expand(N, d)  # [N, d]
    else:
        # Per-walker directions
        assert direction.shape == (N, d), f"Direction shape {direction.shape} != {(N, d)}"
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-10)

    # Compute edge weights
    edge_weights = compute_edge_weights(
        positions, edge_index, mode=weighting_mode, alive=alive
    )

    src, dst = edge_index

    # Fitness differences
    fitness_diffs = fitness_values[dst] - fitness_values[src]  # [E]

    # Displacements
    displacements = positions[dst] - positions[src]  # [E, d]
    distances = torch.norm(displacements, dim=1)  # [E]

    # Project displacements onto directions: n̂·Δx
    # direction[src] shape: [E, d]
    # displacements shape: [E, d]
    projections = (direction[src] * displacements).sum(dim=1)  # [E]

    # Directional derivative contributions
    dd_contributions = (
        edge_weights * fitness_diffs * projections / (distances + 1e-10)
    )  # [E]

    # Accumulate per walker
    directional_derivative = torch.zeros(N, device=device)
    directional_derivative.scatter_add_(0, src, dd_contributions)

    # Apply alive mask
    if alive is not None:
        directional_derivative[~alive] = float("nan")

    return directional_derivative


def estimate_gradient_quality_metrics(
    gradient_result: dict[str, Tensor],
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
) -> dict[str, float]:
    """Compute quality metrics for gradient estimation.

    Useful diagnostics:
    - Consistency: How well do directional derivatives match full gradient?
    - Coverage: Fraction of walkers with valid estimates
    - Neighbor distribution: Are walkers sufficiently connected?

    Args:
        gradient_result: Output from estimate_gradient_finite_difference
        positions: [N, d] walker positions
        fitness_values: [N] fitness values
        edge_index: [2, E] neighbor graph

    Returns:
        dict with scalar metrics:
            - "valid_fraction": Fraction of walkers with valid gradients
            - "mean_neighbors": Average neighbors per walker
            - "mean_quality": Average estimation quality (variance)
            - "gradient_norm_mean": Mean ||∇V||
            - "gradient_norm_std": Std of ||∇V||
    """
    valid_mask = gradient_result["valid_mask"]
    num_neighbors = gradient_result["num_neighbors"]
    estimation_quality = gradient_result["estimation_quality"]
    gradient_magnitude = gradient_result["gradient_magnitude"]

    # Valid walkers
    valid_fraction = valid_mask.float().mean().item()

    # Neighbor statistics
    mean_neighbors = num_neighbors[valid_mask].float().mean().item()

    # Quality statistics (lower is better)
    valid_quality = estimation_quality[valid_mask]
    mean_quality = valid_quality[torch.isfinite(valid_quality)].mean().item()

    # Gradient magnitude statistics
    valid_grad_mag = gradient_magnitude[valid_mask]
    valid_grad_mag = valid_grad_mag[torch.isfinite(valid_grad_mag)]
    gradient_norm_mean = valid_grad_mag.mean().item()
    gradient_norm_std = valid_grad_mag.std().item()

    return {
        "valid_fraction": valid_fraction,
        "mean_neighbors": mean_neighbors,
        "mean_quality": mean_quality,
        "gradient_norm_mean": gradient_norm_mean,
        "gradient_norm_std": gradient_norm_std,
    }
