"""Finite-difference gradient estimation from fitness values on scutoid geometry.

Implements weighted finite-difference methods to estimate ∇V_fit from:
- Fitness values V(x_i) at walker positions
- Neighbor graph structure
"""

import torch
from torch import Tensor

from .utils import (
    validate_finite_difference_inputs,
)
from .weights import compute_edge_weights, WeightMode


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
            "csr_types required to filter boundary neighbors for gradient estimation."
        )

    edge_index = torch.stack([src, indices], dim=0)
    return edge_index, counts


def estimate_gradient_finite_difference(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor | None,
    alive: Tensor | None = None,
    edge_weights: Tensor | None = None,
    weight_mode: WeightMode = "inverse_distance",
    cell_volumes: Tensor | None = None,
    metric_tensors: Tensor | None = None,
    riemannian_volumes: Tensor | None = None,
    normalize_weights: bool = True,
    validate_inputs: bool = True,
    csr_ptr: Tensor | None = None,
    csr_indices: Tensor | None = None,
    csr_types: Tensor | None = None,
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
        edge_weights: Optional [E] precomputed neighbor weights
        weight_mode: Weighting scheme for automatic computation if edge_weights is None:
            - "uniform"
            - "inverse_distance"
            - "inverse_volume"
            - "inverse_riemannian_volume"
            - "inverse_riemannian_distance"
        cell_volumes: Optional [N] cell volumes for volume-based weights
        metric_tensors: Optional [N, d, d] emergent metric for Riemannian weights
        riemannian_volumes: Optional [N] precomputed Riemannian volumes
        normalize_weights: If True, normalize weights per source walker
        validate_inputs: If True, check for NaN/isolated walkers
        csr_ptr: Optional [N+1] CSR row pointers
        csr_indices: Optional [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)

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

    edge_index, num_neighbors = _prepare_edge_index(
        positions, edge_index, csr_ptr, csr_indices, csr_types
    )

    # Validate inputs if requested
    if validate_inputs:
        diagnostics = validate_finite_difference_inputs(
            positions,
            fitness_values,
            edge_index,
            alive,
            min_neighbors=2,
            num_neighbors=num_neighbors,
        )
        valid_mask = diagnostics["valid_walkers"]
        num_neighbors = diagnostics["num_neighbors"]
    else:
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
        if alive is not None:
            valid_mask &= alive

    # Compute edge weights
    # Extract source and destination indices
    src, dst = edge_index

    # Compute position displacements
    displacements = positions[dst] - positions[src]  # [E, d]
    distances = torch.norm(displacements, dim=1)  # [E]
    distances_sq = distances**2  # [E], needed for quality metrics

    if edge_weights is None:
        edge_weights = compute_edge_weights(
            positions,
            edge_index,
            mode=weight_mode,
            edge_distances=distances,
            cell_volumes=cell_volumes,
            metric_tensors=metric_tensors,
            riemannian_volumes=riemannian_volumes,
            alive=alive,
            normalize=normalize_weights,
        )
    elif edge_weights.numel() != edge_index.shape[1]:
        raise ValueError("edge_weights must have the same length as edge_index edges.")

    # Compute fitness differences
    fitness_diffs = fitness_values[dst] - fitness_values[src]  # [E]

    # For gradient estimation, we use weighted least squares:
    # Minimize Σ w_ij (∇V·Δx_ij - ΔV_ij)²
    # Solution: ∇V = (Σ w_ij Δx Δx^T)^(-1) (Σ w_ij Δx ΔV)

    # Build per-walker systems (vectorized)
    gradient = torch.zeros(N, d, device=device, dtype=positions.dtype)

    weighted_dx = displacements * edge_weights.unsqueeze(1)  # [E, d]
    b_contrib = weighted_dx * fitness_diffs.unsqueeze(1)  # [E, d]

    b = torch.zeros(N, d, device=device, dtype=positions.dtype)
    b.scatter_add_(0, src[:, None].expand(-1, d), b_contrib)

    outer = displacements[:, :, None] * displacements[:, None, :]  # [E, d, d]
    outer = outer * edge_weights[:, None, None]

    A_flat = torch.zeros(N, d * d, device=device, dtype=positions.dtype)
    A_flat.scatter_add_(0, src[:, None].expand(-1, d * d), outer.reshape(-1, d * d))
    A = A_flat.view(N, d, d)

    A_reg = A + 1e-6 * torch.eye(d, device=device, dtype=positions.dtype)
    gradient = torch.linalg.solve(A_reg, b.unsqueeze(-1)).squeeze(-1)

    # Set invalid walkers to NaN
    gradient[~valid_mask] = float("nan")

    # Compute gradient magnitude
    gradient_magnitude = torch.norm(gradient, dim=1)

    # Estimate quality: variance of directional derivatives
    # For each walker, compute directional derivatives along each neighbor
    # and measure their variance (should be small if consistent)
    directional_derivatives = fitness_diffs / torch.sqrt(distances_sq + 1e-10)  # [E]

    # Compute variance per source walker (vectorized)
    estimation_quality = torch.zeros(N, device=device, dtype=positions.dtype)
    if directional_derivatives.numel() > 0:
        sum_dd = torch.zeros(N, device=device, dtype=positions.dtype)
        sum_dd_sq = torch.zeros(N, device=device, dtype=positions.dtype)
        sum_dd.scatter_add_(0, src, directional_derivatives)
        sum_dd_sq.scatter_add_(0, src, directional_derivatives**2)

        counts = torch.clamp(num_neighbors.to(sum_dd.dtype), min=1.0)
        mean_dd = sum_dd / counts
        var_dd = sum_dd_sq / counts - mean_dd**2
        estimation_quality = var_dd

    estimation_quality[num_neighbors < 2] = float("nan")
    estimation_quality[~valid_mask] = float("nan")

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
    edge_index: Tensor | None,
    direction: Tensor,
    alive: Tensor | None = None,
    edge_weights: Tensor | None = None,
    weight_mode: WeightMode = "inverse_distance",
    cell_volumes: Tensor | None = None,
    metric_tensors: Tensor | None = None,
    riemannian_volumes: Tensor | None = None,
    normalize_weights: bool = True,
    csr_ptr: Tensor | None = None,
    csr_indices: Tensor | None = None,
    csr_types: Tensor | None = None,
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
        edge_weights: Optional [E] precomputed neighbor weights
        weight_mode: Weighting scheme for automatic computation if edge_weights is None
        cell_volumes: Optional [N] cell volumes for volume-based weights
        metric_tensors: Optional [N, d, d] emergent metric for Riemannian weights
        riemannian_volumes: Optional [N] precomputed Riemannian volumes
        normalize_weights: If True, normalize weights per source walker
        csr_ptr: Optional [N+1] CSR row pointers
        csr_indices: Optional [E] CSR column indices
        csr_types: Optional [E] edge types (0=walker, 1=boundary)

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

    edge_index, _ = _prepare_edge_index(positions, edge_index, csr_ptr, csr_indices, csr_types)

    src, dst = edge_index

    # Fitness differences
    fitness_diffs = fitness_values[dst] - fitness_values[src]  # [E]

    # Displacements
    displacements = positions[dst] - positions[src]  # [E, d]
    distances = torch.norm(displacements, dim=1)  # [E]

    # Compute edge weights
    if edge_weights is None:
        edge_weights = compute_edge_weights(
            positions,
            edge_index,
            mode=weight_mode,
            edge_distances=distances,
            cell_volumes=cell_volumes,
            metric_tensors=metric_tensors,
            riemannian_volumes=riemannian_volumes,
            alive=alive,
            normalize=normalize_weights,
        )
    elif edge_weights.numel() != edge_index.shape[1]:
        raise ValueError("edge_weights must have the same length as edge_index edges.")

    # Project displacements onto directions: n̂·Δx
    # direction[src] shape: [E, d]
    # displacements shape: [E, d]
    projections = (direction[src] * displacements).sum(dim=1)  # [E]

    # Directional derivative contributions
    dd_contributions = (
        edge_weights * fitness_diffs * projections / (distances + 1e-10)
    )  # [E]

    # Accumulate per walker
    directional_derivative = torch.zeros(N, device=device, dtype=positions.dtype)
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
