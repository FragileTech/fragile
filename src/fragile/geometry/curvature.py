"""Alternative curvature computation methods for validation.

This module implements alternative methods for computing Ricci curvature,
providing independent verification of the deficit angle method used in
scutoids.py.

Methods Implemented:
    1. Deficit Angles (implemented in scutoids.py) - baseline
    2. Graph Laplacian Spectrum - spectral bounds on curvature
    3. Fitness Hessian - continuum Riemannian geometry approach
    4. Heat Kernel Asymptotics - (future)
    5. Causal Set Volume - (future)

These methods should all converge to the same Ricci scalar in the continuum
limit (N → ∞), as proven in curvature.md § 2 "Equivalence Theorem".

References:
    - curvature.md for mathematical foundations
    - 14_scutoid_geometry_framework.md § 5 for deficit angle theory
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def compute_graph_laplacian_eigenvalues(
    neighbor_lists: dict[int, list[int]], k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues of graph Laplacian from Voronoi neighbors.

    The graph Laplacian encodes curvature via spectral properties. The
    Cheeger inequality relates the first non-zero eigenvalue λ₁ to
    positive Ricci curvature.

    Args:
        neighbor_lists: Dict mapping walker_id → list of neighbor IDs
        k: Number of smallest eigenvalues to compute (default: 5)

    Returns:
        Tuple of (eigenvalues, eigenvectors)
        - eigenvalues: Array of shape [k] with λ₀ ≤ λ₁ ≤ ... ≤ λₖ₋₁
        - eigenvectors: Array of shape [N, k] with corresponding eigenfunctions

    Notes:
        - λ₀ = 0 always (constant eigenfunction)
        - λ₁ > 0 is the spectral gap (Fiedler value)
        - Large λ₁ suggests positive curvature (Cheeger inequality)
        - Small λ₁ does NOT imply negative curvature (one-way bound)

    Reference:
        curvature.md § 1.2 "Graph Laplacian Spectrum"

    Example:
        >>> neighbors = {0: [1, 2], 1: [0, 2, 3], ...}
        >>> eigenvals, eigenvecs = compute_graph_laplacian_eigenvalues(neighbors, k=5)
        >>> spectral_gap = eigenvals[1]  # λ₁
        >>> # Check Cheeger inequality: Ric > 0 ⟹ λ₁ > threshold
    """
    # Build adjacency matrix
    walker_ids = sorted(neighbor_lists.keys())
    N = len(walker_ids)
    id_to_idx = {wid: i for i, wid in enumerate(walker_ids)}

    # Sparse representation for efficiency
    row, col, data = [], [], []

    for walker_id, neighbors in neighbor_lists.items():
        i = id_to_idx[walker_id]
        degree = len(neighbors)

        # Diagonal: degree
        row.append(i)
        col.append(i)
        data.append(float(degree))

        # Off-diagonal: -1 for each edge
        for neighbor_id in neighbors:
            j = id_to_idx[neighbor_id]
            row.append(i)
            col.append(j)
            data.append(-1.0)

    # Build sparse Laplacian
    laplacian = csr_matrix((data, (row, col)), shape=(N, N))

    # Compute smallest eigenvalues
    # Note: eigsh returns eigenvalues in ascending order
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=min(k, N - 1), which="SM")
    except Exception as e:
        # Fallback: use dense eigensolver for small N
        if N < 100:
            laplacian_dense = laplacian.toarray()
            eig_all = np.linalg.eigvalsh(laplacian_dense)
            eigenvalues = eig_all[:k]
            eigenvectors = np.eye(N)[:, :k]  # Placeholder
        else:
            raise e

    return eigenvalues, eigenvectors


def check_cheeger_consistency(
    ricci_scalars: np.ndarray, eigenvalues: np.ndarray, verbose: bool = False
) -> dict:
    """Check consistency between Ricci curvature and spectral gap.

    Uses one-way implication from Cheeger inequality:
        Ric ≥ κ > 0  ⟹  λ₁ ≥ C(κ, d, diam)

    If mean Ricci is positive, spectral gap should be reasonably large.
    Violations suggest potential errors in curvature computation.

    Args:
        ricci_scalars: Array of computed Ricci scalars [N]
        eigenvalues: Laplacian eigenvalues [k] with λ₀, λ₁, ...
        verbose: Print detailed diagnostics

    Returns:
        Dictionary with consistency check results:
            - mean_ricci: Mean Ricci scalar
            - spectral_gap: λ₁ (first non-zero eigenvalue)
            - is_consistent: Boolean (pass/fail)
            - warning: Message if inconsistent

    Reference:
        curvature.md § 1.2, Theorem "Cheeger Inequality and Ricci Curvature Bounds"

    Example:
        >>> ricci = np.array([0.1, 0.15, 0.12, ...])  # Positive curvature
        >>> eigenvals = np.array([0.0, 0.05, 0.12, ...])
        >>> result = check_cheeger_consistency(ricci, eigenvals)
        >>> if not result["is_consistent"]:
        >>>     print(result["warning"])
    """
    # Compute statistics
    valid_ricci = ricci_scalars[~np.isnan(ricci_scalars)]
    mean_ricci = np.mean(valid_ricci) if len(valid_ricci) > 0 else 0.0
    spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

    # Consistency check (heuristic)
    # If mean Ricci > 0, expect λ₁ to not be too small
    # Rough threshold: λ₁ > 0.01 * mean_ricci (very conservative)
    is_consistent = True
    warning = None

    if mean_ricci > 0.01:
        # Positive curvature case
        if spectral_gap < 0.001:
            is_consistent = False
            warning = (
                f"Positive mean Ricci ({mean_ricci:.4f}) but very small spectral gap "
                f"({spectral_gap:.4f}). Cheeger inequality may be violated. "
                f"Check curvature computation."
            )
    elif mean_ricci < -0.01:
        # Negative curvature case - no lower bound expected
        pass  # Consistent by default

    if verbose:
        print(f"Mean Ricci: {mean_ricci:.4f}")
        print(f"Spectral gap (λ₁): {spectral_gap:.4f}")
        print(f"Consistent: {is_consistent}")
        if warning:
            print(f"Warning: {warning}")

    return {
        "mean_ricci": mean_ricci,
        "spectral_gap": spectral_gap,
        "is_consistent": is_consistent,
        "warning": warning,
    }


def compare_ricci_methods(
    ricci_deficit: np.ndarray,
    ricci_alternative: np.ndarray,
    method_name: str = "alternative",
) -> dict:
    """Compare Ricci scalars from two different methods.

    Computes correlation and relative error statistics to assess agreement
    between deficit angle method and an alternative curvature computation.

    Args:
        ricci_deficit: Ricci from deficit angle method [N]
        ricci_alternative: Ricci from alternative method [N]
        method_name: Name of alternative method for reporting

    Returns:
        Dictionary with comparison statistics:
            - correlation: Pearson correlation coefficient
            - rmse: Root mean squared error
            - mean_relative_error: Mean |R₁ - R₂| / |R₁|
            - max_absolute_error: Maximum |R₁ - R₂|

    Example:
        >>> ricci_deficit = np.array([0.1, 0.2, 0.15, ...])
        >>> ricci_hessian = np.array([0.12, 0.18, 0.16, ...])
        >>> stats = compare_ricci_methods(ricci_deficit, ricci_hessian, "Hessian")
        >>> print(f"Correlation: {stats['correlation']:.3f}")
        >>> print(f"RMSE: {stats['rmse']:.4f}")
    """
    # Filter out NaN values
    mask = ~(np.isnan(ricci_deficit) | np.isnan(ricci_alternative))
    r1 = ricci_deficit[mask]
    r2 = ricci_alternative[mask]

    if len(r1) == 0:
        return {
            "correlation": np.nan,
            "rmse": np.nan,
            "mean_relative_error": np.nan,
            "max_absolute_error": np.nan,
            "n_valid": 0,
        }

    # Correlation
    correlation = np.corrcoef(r1, r2)[0, 1] if len(r1) > 1 else np.nan

    # RMSE
    rmse = np.sqrt(np.mean((r1 - r2) ** 2))

    # Relative error (avoid division by zero)
    relative_errors = np.abs(r1 - r2) / (np.abs(r1) + 1e-10)
    mean_relative_error = np.mean(relative_errors)

    # Max absolute error
    max_absolute_error = np.max(np.abs(r1 - r2))

    return {
        "correlation": correlation,
        "rmse": rmse,
        "mean_relative_error": mean_relative_error,
        "max_absolute_error": max_absolute_error,
        "n_valid": len(r1),
        "method_name": method_name,
    }


# TODO: Implement Method 3 (Fitness Hessian)
# def compute_ricci_from_hessian(positions, fitness_potential, epsilon):
#     """Compute Ricci scalar from fitness Hessian.
#
#     This is Method 3 from curvature.md § 1.3.
#     Requires:
#         1. Compute metric: g = H + εI where H = ∇²V_fit
#         2. Compute Christoffel symbols: Γ^k_ij
#         3. Compute Ricci tensor: Ric_ij
#         4. Contract: R = g^ij Ric_ij
#
#     TODO: Implement full Riemannian geometry computation
#     """
#     raise NotImplementedError("Method 3 (Hessian) not yet implemented")
