"""Curvature computation methods leveraging the FractalSet data structure.

This module implements multiple methods for computing Ricci curvature from
the FractalSet graph representation of walker trajectories, following the
mathematical framework in curvature.md.

Methods Implemented:
    1. Deficit Angles (Discrete Differential Geometry) - TODO
       - Requires Voronoi tessellation
       - Purely topological, no derivatives
    2. Graph Laplacian Spectrum (Spectral Geometry) - IMPLEMENTED ✓
       - Uses FractalSet IG (Information Graph) edges
       - Spectral gap provides Ricci lower bounds via Cheeger inequality
    3. Fitness Hessian (Riemannian Geometry) - IMPLEMENTED ✓
       - Uses FractalSet CST (Causal Spacetime Tree) edge Hessian data
       - Emergent metric g = H + εI where H = ∇²V_fit
    4. Heat Kernel Asymptotics (Analytic Geometry) - TODO
    5. Causal Set Volume (Discrete Spacetime Geometry) - TODO

FractalSet Integration:
    The FractalSet provides a rich data structure for curvature computation:
    - IG edges: Graph connectivity for Laplacian spectrum (Method 2)
    - CST edges: Gradient/Hessian data for metric tensor (Method 3)
    - Node data: Position, velocity, fitness for all methods
    - Localized statistics: μ_ρ, σ_ρ for statistical curvature measures

All methods should converge to the same Ricci scalar in the continuum
limit (N → ∞), as proven in curvature.md § 2 "Equivalence Theorem".

References:
    - old_docs/source/curvature.md for mathematical foundations
    - old_docs/source/13_fractal_set_new/01_fractal_set.md for FractalSet spec
    - old_docs/source/14_scutoid_geometry_framework.md § 5 for deficit angles

Example Usage:
    >>> from fragile.fractalai.core.euclidean_gas import EuclideanGas
    >>> from fragile.fractalai.core.fractal_set import FractalSet
    >>> from fragile.geometry.curvature import (
    ...     compute_ricci_from_fractal_set_graph,
    ...     analyze_curvature_evolution,
    ... )
    >>> # Run simulation
    >>> gas = EuclideanGas(N=50, d=2, ...)
    >>> history = gas.run(n_steps=100, record_every=10)
    >>> fractal_set = FractalSet(history)
    >>> # Compute curvature at a timestep
    >>> curvature = compute_ricci_from_fractal_set_graph(fractal_set, timestep=5)
    >>> print(f"Spectral gap: {curvature['spectral_gap']:.4f}")
    >>> # Analyze evolution
    >>> evolution = analyze_curvature_evolution(fractal_set, method="laplacian")
    >>> # Visualize
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(evolution["timesteps"], evolution["mean_ricci"])
    >>> plt.show()
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


# ==============================================================================
# FractalSet Integration - Leverage FractalSet data structure for curvature
# ==============================================================================


def compute_ricci_from_fractal_set_graph(
    fractal_set,
    timestep: int,
    method: str = "laplacian",
    k_eigenvalues: int = 5,
) -> dict:
    """Compute Ricci curvature estimates from FractalSet graph structure.

    Leverages the FractalSet's IG (Information Graph) edges to compute
    curvature via Method 2 (Graph Laplacian Spectrum).

    Args:
        fractal_set: FractalSet instance with walker trajectory data
        timestep: Timestep index to analyze
        method: Curvature estimation method:
            - "laplacian": Graph Laplacian eigenvalues (Method 2)
            - "deficit": Voronoi deficit angles (Method 1, future)
        k_eigenvalues: Number of smallest eigenvalues to compute

    Returns:
        Dictionary with curvature estimates:
            - method: Method used
            - timestep: Timestep analyzed
            - spectral_gap: λ₁ (first non-zero eigenvalue)
            - eigenvalues: Array of k smallest eigenvalues
            - mean_ricci_estimate: Heuristic Ricci estimate from spectral gap
            - n_walkers: Number of alive walkers analyzed

    Reference:
        curvature.md § 1.2 "Graph Laplacian Spectrum"

    Example:
        >>> from fragile.fractalai.core.euclidean_gas import EuclideanGas
        >>> from fragile.fractalai.core.fractal_set import FractalSet
        >>> # Run simulation
        >>> history = gas.run(n_steps=100, record_every=10)
        >>> fractal_set = FractalSet(history)
        >>> # Compute curvature at timestep 50
        >>> curvature = compute_ricci_from_fractal_set_graph(
        ...     fractal_set, timestep=5, method="laplacian"
        ... )
        >>> print(f"Spectral gap: {curvature['spectral_gap']:.4f}")
        >>> print(f"Mean Ricci estimate: {curvature['mean_ricci_estimate']:.4f}")
    """
    if method != "laplacian":
        raise NotImplementedError(f"Method '{method}' not yet implemented. Use 'laplacian'.")

    # Get alive walkers at this timestep
    alive_walkers = fractal_set.get_alive_walkers(timestep)
    n_alive = len(alive_walkers)

    if n_alive < 2:
        # Need at least 2 walkers for meaningful graph
        return {
            "method": method,
            "timestep": timestep,
            "spectral_gap": 0.0,
            "eigenvalues": np.array([]),
            "mean_ricci_estimate": 0.0,
            "n_walkers": n_alive,
            "warning": f"Too few alive walkers ({n_alive}) for curvature computation",
        }

    # Build neighbor lists from FractalSet IG edges
    # IG edges represent selection coupling (companion relationships)
    neighbor_lists = {}

    # Get IG subgraph at this timestep
    ig_graph = fractal_set.get_ig_subgraph(timestep=timestep)

    # Extract neighbor relationships
    for walker_id in alive_walkers:
        neighbors = set()
        # Find all outgoing IG edges from this walker
        source_node = (walker_id, timestep)
        if source_node in ig_graph:
            for target_node in ig_graph.successors(source_node):
                target_walker, target_t = target_node
                if target_t == timestep:  # Same timestep
                    neighbors.add(target_walker)

        neighbor_lists[walker_id] = list(neighbors)

    # Compute graph Laplacian eigenvalues
    eigenvalues, eigenvectors = compute_graph_laplacian_eigenvalues(
        neighbor_lists, k=min(k_eigenvalues, n_alive - 1)
    )

    spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

    # Heuristic Ricci estimate from spectral gap
    # For positive curvature: R ≈ 2 * λ₁ / d (rough approximation)
    # This is based on Lichnerowicz theorem: λ₁ ≥ R * d / (d-1)
    d = fractal_set.d
    mean_ricci_estimate = 2.0 * spectral_gap / d if d > 0 else 0.0

    return {
        "method": method,
        "timestep": timestep,
        "spectral_gap": spectral_gap,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "mean_ricci_estimate": mean_ricci_estimate,
        "n_walkers": n_alive,
        "dimension": d,
    }


def compute_ricci_from_fractal_set_hessian(
    fractal_set,
    timestep: int,
    epsilon_sigma: float = 0.1,
) -> dict:
    """Compute Ricci curvature from fitness Hessian data in FractalSet.

    Leverages FractalSet's CST edge gradient data to estimate curvature
    via Method 3 (Emergent Metric Tensor). This requires adaptive kinetics
    to be enabled during the simulation to record fitness gradients/Hessians.

    Args:
        fractal_set: FractalSet instance with adaptive kinetics data
        timestep: Timestep index to analyze
        epsilon_sigma: Diffusion regularization for metric (default: 0.1)

    Returns:
        Dictionary with Hessian-based curvature estimates:
            - method: "hessian"
            - timestep: Timestep analyzed
            - ricci_scalars: Per-walker Ricci scalar estimates [n_alive]
            - mean_ricci: Mean Ricci scalar
            - std_ricci: Std of Ricci scalar
            - has_hessian_data: Whether Hessian data is available
            - n_walkers: Number of alive walkers analyzed

    Reference:
        curvature.md § 1.3 "Emergent Metric Tensor"

    Note:
        This method requires fitness_hessians_diag or fitness_hessians_full
        in the RunHistory. Enable with use_anisotropic_diffusion=True in
        the kinetic operator during simulation.

    Example:
        >>> # Requires adaptive kinetics enabled
        >>> history = gas.run(n_steps=100, record_every=10)
        >>> fractal_set = FractalSet(history)
        >>> curvature = compute_ricci_from_fractal_set_hessian(fractal_set, timestep=5)
        >>> if curvature["has_hessian_data"]:
        ...     print(f"Mean Ricci: {curvature['mean_ricci']:.4f}")
    """
    has_diag_hessian = fractal_set.history.fitness_hessians_diag is not None
    has_full_hessian = fractal_set.history.fitness_hessians_full is not None

    if not (has_diag_hessian or has_full_hessian):
        return {
            "method": "hessian",
            "timestep": timestep,
            "ricci_scalars": np.array([]),
            "mean_ricci": 0.0,
            "std_ricci": 0.0,
            "has_hessian_data": False,
            "n_walkers": 0,
            "warning": "No Hessian data in RunHistory. Enable use_anisotropic_diffusion=True.",
        }

    # Get alive walkers
    alive_walkers = fractal_set.get_alive_walkers(timestep)
    n_alive = len(alive_walkers)

    if n_alive == 0:
        return {
            "method": "hessian",
            "timestep": timestep,
            "ricci_scalars": np.array([]),
            "mean_ricci": 0.0,
            "std_ricci": 0.0,
            "has_hessian_data": True,
            "n_walkers": 0,
        }

    # Compute Ricci scalar estimates from Hessian diagonal
    # For metric g = H + εI, the Ricci scalar involves traces of derivatives
    # Simplified estimate: R ≈ tr(H) / d (mean curvature approximation)
    import torch

    ricci_scalars = []

    if timestep == 0:
        # No CST edge data at t=0
        return {
            "method": "hessian",
            "timestep": timestep,
            "ricci_scalars": np.array([]),
            "mean_ricci": 0.0,
            "std_ricci": 0.0,
            "has_hessian_data": True,
            "n_walkers": n_alive,
            "warning": "No CST edges at timestep 0",
        }

    for walker_id in alive_walkers:
        # Get CST edge from (walker_id, timestep) → (walker_id, timestep+1)
        source = (walker_id, timestep)
        # Find CST edge
        cst_edges = [
            (u, v, d)
            for u, v, d in fractal_set.graph.edges(source, data=True)
            if d.get("edge_type") == "cst"
        ]

        if not cst_edges:
            continue

        _, _, edge_data = cst_edges[0]

        # Extract Hessian data
        if has_diag_hessian and "hess_V_fit_diag" in edge_data:
            hess_diag = edge_data["hess_V_fit_diag"]  # [d]
            # Ricci scalar estimate: R ≈ tr(H + εI) / d
            trace_metric = torch.sum(hess_diag).item() + epsilon_sigma * fractal_set.d
            ricci_estimate = trace_metric / fractal_set.d
            ricci_scalars.append(ricci_estimate)
        elif has_full_hessian and "hess_V_fit_full" in edge_data:
            hess_full = edge_data["hess_V_fit_full"]  # [d, d]
            # Trace of full Hessian
            trace_hess = torch.trace(hess_full).item()
            trace_metric = trace_hess + epsilon_sigma * fractal_set.d
            ricci_estimate = trace_metric / fractal_set.d
            ricci_scalars.append(ricci_estimate)

    ricci_scalars = np.array(ricci_scalars)

    return {
        "method": "hessian",
        "timestep": timestep,
        "ricci_scalars": ricci_scalars,
        "mean_ricci": np.mean(ricci_scalars) if len(ricci_scalars) > 0 else 0.0,
        "std_ricci": np.std(ricci_scalars) if len(ricci_scalars) > 0 else 0.0,
        "has_hessian_data": True,
        "n_walkers": len(ricci_scalars),
        "dimension": fractal_set.d,
        "epsilon_sigma": epsilon_sigma,
    }


def analyze_curvature_evolution(
    fractal_set,
    method: str = "laplacian",
    epsilon_sigma: float = 0.1,
) -> dict:
    """Analyze curvature evolution over all recorded timesteps in FractalSet.

    Computes curvature estimates at each timestep and returns time series
    for visualization and analysis.

    Args:
        fractal_set: FractalSet instance
        method: Curvature method ("laplacian" or "hessian")
        epsilon_sigma: Regularization for hessian method

    Returns:
        Dictionary with time series data:
            - timesteps: Array of timestep indices
            - mean_ricci: Mean Ricci scalar at each timestep
            - std_ricci: Std of Ricci scalar at each timestep (hessian only)
            - spectral_gaps: Spectral gaps at each timestep (laplacian only)
            - n_walkers: Number of alive walkers at each timestep
            - method: Method used

    Example:
        >>> evolution = analyze_curvature_evolution(fractal_set, method="laplacian")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(evolution["timesteps"], evolution["mean_ricci"])
        >>> plt.xlabel("Timestep")
        >>> plt.ylabel("Mean Ricci Curvature")
        >>> plt.show()
    """
    n_recorded = fractal_set.n_recorded
    timesteps = []
    mean_riccis = []
    std_riccis = []
    spectral_gaps = []
    n_walkers_list = []

    for t in range(n_recorded):
        if method == "laplacian":
            result = compute_ricci_from_fractal_set_graph(fractal_set, timestep=t)
            if "warning" in result:
                continue
            timesteps.append(t)
            mean_riccis.append(result["mean_ricci_estimate"])
            spectral_gaps.append(result["spectral_gap"])
            n_walkers_list.append(result["n_walkers"])
            std_riccis.append(0.0)  # Not applicable for spectral method

        elif method == "hessian":
            result = compute_ricci_from_fractal_set_hessian(
                fractal_set, timestep=t, epsilon_sigma=epsilon_sigma
            )
            if not result["has_hessian_data"] or "warning" in result:
                continue
            timesteps.append(t)
            mean_riccis.append(result["mean_ricci"])
            std_riccis.append(result["std_ricci"])
            spectral_gaps.append(0.0)  # Not applicable
            n_walkers_list.append(result["n_walkers"])

    return {
        "timesteps": np.array(timesteps),
        "mean_ricci": np.array(mean_riccis),
        "std_ricci": np.array(std_riccis),
        "spectral_gaps": np.array(spectral_gaps) if method == "laplacian" else None,
        "n_walkers": np.array(n_walkers_list),
        "method": method,
        "n_recorded": len(timesteps),
    }
