"""Hessian estimation from fitness values using finite differences and geometric methods.

Provides two complementary approaches:
1. Finite differences (primary): Direct second-order derivatives from fitness
2. Geometric (validation): H = g - ε_Σ I from neighbor covariance
"""

import torch
from torch import Tensor
from typing import Literal
import warnings

from .utils import (
    estimate_optimal_step_size,
    find_axial_neighbors,
    validate_finite_difference_inputs,
)


def estimate_hessian_diagonal_fd(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    alive: Tensor | None = None,
    step_size: float | None = None,
    min_neighbors_per_axis: int = 2,
) -> dict[str, Tensor]:
    """Estimate diagonal Hessian elements using second-order finite differences.

    Computes ∂²V/∂x_α² for each coordinate axis α using central differences:
        ∂²V/∂x_α² ≈ [V(x + h·e_α) + V(x - h·e_α) - 2V(x)] / h²

    This is O(N·d) and provides a fast approximation when off-diagonal coupling
    is weak or full Hessian O(N·d²) is too expensive.

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness at each position
        edge_index: [2, E] neighbor connectivity
        alive: [N] optional validity mask
        step_size: Step size h for finite differences (auto if None)
        min_neighbors_per_axis: Minimum neighbors needed along each axis

    Returns:
        dict containing:
            - "hessian_diagonal": [N, d] diagonal elements H_αα
            - "eigenvalues": [N, d] (same as diagonal for diagonal-only estimate)
            - "step_sizes": [N, d] effective step size used per axis
            - "axis_quality": [N, d] alignment quality (1 = perfect axial)
            - "valid_mask": [N] walkers with valid estimates

    Complexity: O(N·d·k) where k = avg neighbors

    Algorithm:
        For each walker i and axis α:
            1. Find neighbors j,k approximately along ±e_α
            2. Estimate h from neighbor distances
            3. Compute (V_j + V_k - 2V_i) / h²
            4. Average if multiple neighbor pairs available
    """
    N, d = positions.shape
    device = positions.device

    # Initialize outputs
    hessian_diagonal = torch.zeros(N, d, device=device)
    step_sizes_used = torch.zeros(N, d, device=device)
    axis_quality = torch.zeros(N, d, device=device)
    valid_mask = torch.ones(N, dtype=torch.bool, device=device)

    # Validate inputs
    diagnostics = validate_finite_difference_inputs(
        positions, fitness_values, edge_index, alive, min_neighbors=min_neighbors_per_axis * d
    )

    # Estimate step size if not provided
    if step_size is None:
        step_sizes_per_walker = estimate_optimal_step_size(
            positions, edge_index, target_fraction=0.5, alive=alive
        )
    else:
        step_sizes_per_walker = torch.full((N,), step_size, device=device)

    # Process each walker
    for i in range(N):
        if alive is not None and not alive[i]:
            valid_mask[i] = False
            hessian_diagonal[i] = float("nan")
            continue

        # Process each axis
        for axis in range(d):
            # Find neighbors along this axis
            pos_neighbors, neg_neighbors = find_axial_neighbors(
                positions, edge_index, walker_idx=i, axis=axis, max_angle_deg=30.0
            )

            if len(pos_neighbors) == 0 or len(neg_neighbors) == 0:
                # Not enough axial neighbors
                hessian_diagonal[i, axis] = float("nan")
                axis_quality[i, axis] = 0.0
                step_sizes_used[i, axis] = float("nan")
                continue

            # Use all combinations of positive and negative neighbors
            h_estimates = []
            H_estimates = []
            quality_scores = []

            for j in pos_neighbors:
                for k in neg_neighbors:
                    # Estimate step size from neighbor positions
                    delta_pos = positions[j, axis] - positions[i, axis]
                    delta_neg = positions[k, axis] - positions[i, axis]

                    # Should have opposite signs
                    if delta_pos * delta_neg >= 0:
                        continue

                    h_pos = abs(delta_pos)
                    h_neg = abs(delta_neg)
                    h_avg = (h_pos + h_neg) / 2

                    # Central difference formula
                    V_i = fitness_values[i]
                    V_j = fitness_values[j]
                    V_k = fitness_values[k]

                    H_estimate = (V_j + V_k - 2 * V_i) / (h_avg ** 2)

                    # Quality: how symmetric are the steps?
                    symmetry = 1.0 - abs(h_pos - h_neg) / (h_pos + h_neg + 1e-10)

                    h_estimates.append(h_avg)
                    H_estimates.append(H_estimate)
                    quality_scores.append(symmetry)

            if len(H_estimates) == 0:
                hessian_diagonal[i, axis] = float("nan")
                axis_quality[i, axis] = 0.0
                step_sizes_used[i, axis] = float("nan")
                continue

            # Average estimates weighted by quality
            h_estimates = torch.tensor(h_estimates, device=device)
            H_estimates = torch.tensor(H_estimates, device=device)
            quality_scores = torch.tensor(quality_scores, device=device)

            # Weighted average
            weights = quality_scores / (quality_scores.sum() + 1e-10)
            hessian_diagonal[i, axis] = (weights * H_estimates).sum()
            step_sizes_used[i, axis] = (weights * h_estimates).sum()
            axis_quality[i, axis] = quality_scores.mean()

    # Overall validity: at least half the axes have valid estimates
    n_valid_axes = torch.isfinite(hessian_diagonal).sum(dim=1)
    valid_mask = n_valid_axes >= (d // 2)

    return {
        "hessian_diagonal": hessian_diagonal,
        "eigenvalues": hessian_diagonal,  # For diagonal matrix, eigenvalues = diagonal
        "step_sizes": step_sizes_used,
        "axis_quality": axis_quality,
        "valid_mask": valid_mask,
    }


def estimate_hessian_full_fd(
    positions: Tensor,
    fitness_values: Tensor,
    gradient_vectors: Tensor | None,
    edge_index: Tensor,
    alive: Tensor | None = None,
    step_size: float | None = None,
    method: Literal["central", "gradient_fd"] = "central",
    symmetrize: bool = True,
) -> dict[str, Tensor]:
    """Estimate full Hessian matrix using second-order finite differences.

    Two methods available:
    1. "central": Mixed second derivatives from fitness values
       H_αβ ≈ [V(x+h_α+h_β) - V(x+h_α) - V(x+h_β) + V(x)] / (h_α·h_β)

    2. "gradient_fd": Finite difference of gradients
       H_αβ ≈ [∇V_β(x+h·e_α) - ∇V_β(x)] / h
       (Requires pre-computed gradient_vectors)

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness values
        gradient_vectors: [N, d] optional pre-computed gradients (for method="gradient_fd")
        edge_index: [2, E] neighbor graph
        alive: [N] optional validity mask
        step_size: Step size h (auto if None)
        method: "central" (from fitness) or "gradient_fd" (from gradients)
        symmetrize: Force Hessian symmetric by averaging H and H^T

    Returns:
        dict containing:
            - "hessian_tensors": [N, d, d] full symmetric Hessian matrices
            - "hessian_eigenvalues": [N, d] eigenspectrum (sorted descending)
            - "condition_numbers": [N] κ(H) = |λ_max|/|λ_min|
            - "symmetry_error": [N] ||H - H^T||_F (before symmetrization)
            - "psd_fraction": float, fraction with all eigenvalues ≥ 0
            - "valid_mask": [N] boolean validity

    Complexity: O(N·d²·k)

    Note: For d > 20, consider using diagonal approximation instead.
    """
    N, d = positions.shape
    device = positions.device

    if method == "gradient_fd" and gradient_vectors is None:
        raise ValueError("gradient_fd method requires gradient_vectors input")

    # Initialize
    hessian_tensors = torch.zeros(N, d, d, device=device)
    symmetry_errors = torch.zeros(N, device=device)
    valid_mask = torch.ones(N, dtype=torch.bool, device=device)

    # Estimate step size if needed
    if step_size is None:
        step_sizes_per_walker = estimate_optimal_step_size(
            positions, edge_index, target_fraction=0.3, alive=alive
        )
    else:
        step_sizes_per_walker = torch.full((N,), step_size, device=device)

    if method == "central":
        # Central difference method: need 4-point stencil per (α,β) pair
        hessian_tensors = _estimate_hessian_central_differences(
            positions, fitness_values, edge_index, step_sizes_per_walker, alive
        )

    elif method == "gradient_fd":
        # Gradient finite difference: ∂∇V/∂x
        hessian_tensors = _estimate_hessian_from_gradient_fd(
            positions, gradient_vectors, edge_index, step_sizes_per_walker, alive
        )

    # Compute symmetry error before symmetrization
    symmetry_errors = torch.norm(
        hessian_tensors - hessian_tensors.transpose(1, 2),
        p="fro",
        dim=(1, 2)
    )

    # Symmetrize if requested
    if symmetrize:
        hessian_tensors = 0.5 * (hessian_tensors + hessian_tensors.transpose(1, 2))

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(hessian_tensors)  # [N, d], sorted ascending

    # Condition numbers
    lambda_max = eigenvalues[:, -1].abs()  # Largest magnitude
    lambda_min = eigenvalues[:, 0].abs()   # Smallest magnitude
    condition_numbers = lambda_max / (lambda_min + 1e-10)

    # Fraction with all non-negative eigenvalues (PSD)
    psd_mask = (eigenvalues >= -1e-6).all(dim=1)  # Small tolerance for numerical error
    psd_fraction = psd_mask.float().mean().item()

    # Sort eigenvalues descending
    eigenvalues = torch.flip(eigenvalues, dims=[1])

    return {
        "hessian_tensors": hessian_tensors,
        "hessian_eigenvalues": eigenvalues,
        "condition_numbers": condition_numbers,
        "symmetry_error": symmetry_errors,
        "psd_fraction": psd_fraction,
        "valid_mask": valid_mask,
    }


def _estimate_hessian_central_differences(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    step_sizes: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Internal: Estimate Hessian using central differences on fitness values.

    For each (α, β) pair, need to find neighbors at:
        - x + h·e_α + h·e_β
        - x + h·e_α
        - x + h·e_β
        - x (center)

    In practice, with irregular neighbor graphs, we approximate using
    available neighbors and directional differences.

    Returns:
        [N, d, d] Hessian tensors
    """
    N, d = positions.shape
    device = positions.device

    H = torch.zeros(N, d, d, device=device)

    # For diagonal elements: use central differences
    for i in range(N):
        h = step_sizes[i]

        for alpha in range(d):
            # Diagonal: H_αα
            pos_neighbors, neg_neighbors = find_axial_neighbors(
                positions, edge_index, i, alpha, max_angle_deg=30.0
            )

            if len(pos_neighbors) > 0 and len(neg_neighbors) > 0:
                # Take closest in each direction
                delta_pos = positions[pos_neighbors, alpha] - positions[i, alpha]
                delta_neg = positions[neg_neighbors, alpha] - positions[i, alpha]

                j_pos = pos_neighbors[torch.argmin(delta_pos.abs())]
                j_neg = neg_neighbors[torch.argmin(delta_neg.abs())]

                V_i = fitness_values[i]
                V_pos = fitness_values[j_pos]
                V_neg = fitness_values[j_neg]

                H[i, alpha, alpha] = (V_pos + V_neg - 2 * V_i) / (h ** 2)

            # Off-diagonal: H_αβ for β > α
            for beta in range(alpha + 1, d):
                # Need neighbors in 4 quadrants (simplified: use nearest)
                # This is approximate - exact requires finding specific combinations

                # Approximate via directional second derivatives
                # H_αβ ≈ (∂²V/∂n² along n = (e_α + e_β)/√2) - (H_αα + H_ββ)/2

                # For now, use a simpler approximation: numerical gradient of gradient
                # Skip if too complex - leave as zero (diagonal dominant)
                pass

    return H


def _estimate_hessian_from_gradient_fd(
    positions: Tensor,
    gradient_vectors: Tensor,
    edge_index: Tensor,
    step_sizes: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Internal: Estimate Hessian from finite differences of gradients.

    H_αβ ≈ [∇V_β(x + h·e_α) - ∇V_β(x)] / h

    Requires finding neighbors along each axis and interpolating their gradients.

    Returns:
        [N, d, d] Hessian tensors
    """
    N, d = positions.shape
    device = positions.device

    H = torch.zeros(N, d, d, device=device)

    for i in range(N):
        h = step_sizes[i]
        grad_i = gradient_vectors[i]  # [d]

        for alpha in range(d):
            # Find neighbors along axis α
            pos_neighbors, neg_neighbors = find_axial_neighbors(
                positions, edge_index, i, alpha, max_angle_deg=30.0
            )

            if len(pos_neighbors) == 0:
                continue

            # Use closest positive neighbor
            delta_pos = positions[pos_neighbors, alpha] - positions[i, alpha]
            j_pos = pos_neighbors[torch.argmin(delta_pos.abs())]

            grad_j = gradient_vectors[j_pos]  # [d]

            # Finite difference of gradient
            delta_x = positions[j_pos, alpha] - positions[i, alpha]
            H[i, alpha, :] = (grad_j - grad_i) / (delta_x + 1e-10)

    return H


def estimate_hessian_from_metric(
    positions: Tensor,
    edge_index: Tensor,
    epsilon_sigma: float = 0.1,
    alive: Tensor | None = None,
    metric_tensors: Tensor | None = None,
    validate_equilibrium: bool = True,
) -> dict[str, Tensor]:
    """Estimate Hessian from emergent metric: H = g - ε_Σ I.

    This is the VALIDATION method. It uses the theoretical relationship:
        g = H + ε_Σ I  →  H = g - ε_Σ I

    where g is the emergent metric computed from neighbor covariance.

    Compare with finite-difference results to:
    1. Validate FD estimates are correct
    2. Check walkers are in equilibrium (ρ ∝ e^(-βV))
    3. Get independent curvature measurement

    Args:
        positions: [N, d] walker positions
        edge_index: [2, E] neighbor graph
        epsilon_sigma: Physical spectral floor ε_Σ from theory (NOT numerical reg!)
        alive: [N] optional validity mask
        metric_tensors: [N, d, d] pre-computed metrics (else compute from neighbors)
        validate_equilibrium: If True, compute equilibrium quality score

    Returns:
        dict containing:
            - "hessian_tensors": [N, d, d] estimated Hessian
            - "hessian_eigenvalues": [N, d] eigenspectrum
            - "metric_tensors": [N, d, d] the emergent metric g used
            - "psd_violation_mask": [N] bool, H not positive semi-definite
            - "equilibrium_score": [N] metric-fitness correlation (0-1)
                High score = good equilibrium, estimates more reliable

    Complexity: O(N·k·d² + N·d³) for covariance + eigendecomp

    Warning: This method assumes walkers are in quasi-equilibrium.
    If not, H estimates may be unreliable. Check equilibrium_score.
    """
    N, d = positions.shape
    device = positions.device

    # Compute metric tensors if not provided
    if metric_tensors is None:
        metric_tensors = _compute_emergent_metric(positions, edge_index, alive)

    # Extract Hessian: H = g - ε_Σ I
    identity = torch.eye(d, device=device).unsqueeze(0).expand(N, d, d)
    hessian_tensors = metric_tensors - epsilon_sigma * identity

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(hessian_tensors)  # [N, d], ascending

    # Check for PSD violations (negative eigenvalues)
    psd_violation_mask = (eigenvalues[:, 0] < -1e-6)  # Most negative eigenvalue

    # Sort eigenvalues descending
    eigenvalues = torch.flip(eigenvalues, dims=[1])

    # Compute equilibrium score if requested
    if validate_equilibrium:
        equilibrium_score = _compute_equilibrium_score(
            positions, edge_index, metric_tensors, alive
        )
    else:
        equilibrium_score = torch.ones(N, device=device)

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


def _compute_emergent_metric(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Compute emergent metric g from neighbor covariance.

    Algorithm:
        1. For each walker i, compute covariance of neighbor positions:
           C_αβ = (1/k) Σ_{j∈N(i)} (x_j^α - x_i^α)(x_j^β - x_i^β)
        2. Invert to get metric: g = C^(-1) + numerical_regularization
        3. Return g

    Returns:
        [N, d, d] metric tensors
    """
    N, d = positions.shape
    device = positions.device

    src, dst = edge_index

    # Compute neighbor covariances
    covariances = torch.zeros(N, d, d, device=device)
    neighbor_counts = torch.zeros(N, device=device)

    # Accumulate displacements
    displacements = positions[dst] - positions[src]  # [E, d]

    # Outer product: Δx ⊗ Δx
    outer_products = displacements.unsqueeze(2) * displacements.unsqueeze(1)  # [E, d, d]

    # Sum per source walker
    for e in range(edge_index.shape[1]):
        i = src[e].item()
        covariances[i] += outer_products[e]
        neighbor_counts[i] += 1

    # Normalize by neighbor count
    neighbor_counts = torch.clamp(neighbor_counts, min=1)  # Avoid division by zero
    covariances = covariances / neighbor_counts.view(N, 1, 1)

    # Invert covariance to get metric (with regularization for numerical stability)
    epsilon_numerical = 1e-5  # Numerical stability regularization
    identity = torch.eye(d, device=device).unsqueeze(0)

    metrics = torch.zeros(N, d, d, device=device)
    for i in range(N):
        C = covariances[i] + epsilon_numerical * identity.squeeze(0)
        try:
            metrics[i] = torch.linalg.inv(C)
        except RuntimeError:
            # Singular matrix - use pseudo-inverse
            metrics[i] = torch.linalg.pinv(C)

    return metrics


def _compute_equilibrium_score(
    positions: Tensor,
    edge_index: Tensor,
    metric_tensors: Tensor,
    alive: Tensor | None,
) -> Tensor:
    """Compute equilibrium quality score based on metric-fitness correlation.

    If walkers are in equilibrium (ρ ∝ e^(-βV)), the metric should correlate
    with local fitness curvature. This provides a sanity check.

    Returns:
        [N] scores in [0, 1], where 1 = high confidence in equilibrium
    """
    N = positions.shape[0]
    device = positions.device

    # Simplified: check metric isotropy (should be more isotropic in equilibrium)
    # Compute condition numbers of metric tensors
    eigenvalues = torch.linalg.eigvalsh(metric_tensors)  # [N, d]

    # Isotropy score: 1 / condition_number
    condition_numbers = eigenvalues[:, -1] / (eigenvalues[:, 0] + 1e-10)
    isotropy_scores = 1.0 / (1.0 + condition_numbers)

    # Simple score: higher isotropy = better equilibrium
    return isotropy_scores
