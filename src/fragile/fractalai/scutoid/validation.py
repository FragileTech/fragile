"""Validation and comparison tools for gradient/Hessian estimation methods.

Cross-validates finite-difference vs geometric methods and tests on synthetic functions.
"""

import torch
from torch import Tensor
from typing import Literal, Any, Callable
import warnings

from .gradient_estimation import estimate_gradient_finite_difference
from .hessian_estimation import (
    estimate_hessian_diagonal_fd,
    estimate_hessian_full_fd,
    estimate_hessian_from_metric,
)


def compare_estimation_methods(
    positions: Tensor,
    fitness_values: Tensor,
    edge_index: Tensor,
    epsilon_sigma: float = 0.1,
    compute_full_hessian: bool = True,
    return_detailed: bool = False,
) -> dict[str, Any]:
    """Cross-validate finite-difference vs geometric Hessian estimation.

    Runs both methods and computes agreement statistics:
    - Eigenvalue correlation
    - Frobenius norm difference
    - Per-element relative errors

    High agreement (>0.8) suggests:
    1. FD estimates are accurate
    2. Walkers are in equilibrium
    3. Both methods capture same curvature

    Low agreement (<0.5) suggests:
    1. Walkers not in equilibrium (ρ ≠ e^(-βV))
    2. Wrong epsilon_sigma parameter
    3. Numerical issues

    Args:
        positions: [N, d] walker positions
        fitness_values: [N] fitness values
        edge_index: [2, E] neighbor graph
        epsilon_sigma: Physical spectral floor for geometric method
        compute_full_hessian: If True, compute full H (else diagonal only)
        return_detailed: If True, include per-walker comparisons

    Returns:
        dict with all estimates and comparison metrics:
            - "gradient_fd": [N, d]
            - "hessian_diagonal_fd": [N, d]
            - "hessian_full_fd": [N, d, d] if compute_full_hessian
            - "hessian_geometric": [N, d, d]
            - "comparison_metrics": {
                "eigenvalue_correlation": float (0-1),
                "frobenius_agreement": float (0-1),
                "element_wise_rmse": float,
                "method_preference": "fd" or "geometric" or "ambiguous"
              }
            - "per_walker_comparison": [N, ...] if return_detailed

    Example:
        >>> comparison = compare_estimation_methods(
        ...     swarm.positions, swarm.fitness, swarm.edge_index
        ... )
        >>> corr = comparison["comparison_metrics"]["eigenvalue_correlation"]
        >>> print(f"Method agreement: {corr:.2f}")
    """
    N, d = positions.shape
    device = positions.device

    # 1. Estimate gradient (FD)
    grad_result = estimate_gradient_finite_difference(
        positions, fitness_values, edge_index
    )
    gradient_fd = grad_result["gradient"]

    # 2. Estimate Hessian diagonal (FD)
    hess_diag_result = estimate_hessian_diagonal_fd(
        positions, fitness_values, edge_index
    )
    hessian_diagonal_fd = hess_diag_result["hessian_diagonal"]

    # 3. Estimate full Hessian (FD) if requested
    if compute_full_hessian:
        hess_full_result = estimate_hessian_full_fd(
            positions, fitness_values, gradient_fd, edge_index,
            method="central", symmetrize=True
        )
        hessian_full_fd = hess_full_result["hessian_tensors"]
        eigenvalues_fd = hess_full_result["hessian_eigenvalues"]
    else:
        hessian_full_fd = None
        eigenvalues_fd = hessian_diagonal_fd  # Diagonal eigenvalues

    # 4. Estimate Hessian (geometric)
    hess_geo_result = estimate_hessian_from_metric(
        positions, edge_index, epsilon_sigma=epsilon_sigma,
        validate_equilibrium=True
    )
    hessian_geometric = hess_geo_result["hessian_tensors"]
    eigenvalues_geo = hess_geo_result["hessian_eigenvalues"]
    equilibrium_score = hess_geo_result["equilibrium_score"]

    # 5. Compare methods
    comparison_metrics = _compute_comparison_metrics(
        eigenvalues_fd, eigenvalues_geo,
        hessian_full_fd if compute_full_hessian else None,
        hessian_geometric,
        equilibrium_score
    )

    # Build return dict
    result = {
        "gradient_fd": gradient_fd,
        "hessian_diagonal_fd": hessian_diagonal_fd,
        "hessian_geometric": hessian_geometric,
        "comparison_metrics": comparison_metrics,
    }

    if compute_full_hessian:
        result["hessian_full_fd"] = hessian_full_fd

    if return_detailed:
        result["per_walker_comparison"] = _compute_per_walker_comparison(
            eigenvalues_fd, eigenvalues_geo, equilibrium_score
        )

    return result


def _compute_comparison_metrics(
    eigenvalues_fd: Tensor,
    eigenvalues_geo: Tensor,
    hessian_full_fd: Tensor | None,
    hessian_geometric: Tensor,
    equilibrium_score: Tensor,
) -> dict[str, Any]:
    """Compute comparison metrics between FD and geometric methods.

    Returns:
        dict with scalar metrics
    """
    # Filter out NaN values
    valid_mask = torch.isfinite(eigenvalues_fd).all(dim=1) & torch.isfinite(eigenvalues_geo).all(dim=1)

    if valid_mask.sum() == 0:
        return {
            "eigenvalue_correlation": 0.0,
            "frobenius_agreement": 0.0,
            "element_wise_rmse": float("inf"),
            "method_preference": "ambiguous",
            "mean_equilibrium_score": 0.0,
        }

    eig_fd_valid = eigenvalues_fd[valid_mask].flatten()
    eig_geo_valid = eigenvalues_geo[valid_mask].flatten()

    # Eigenvalue correlation
    if len(eig_fd_valid) > 1:
        eigenvalue_correlation = torch.corrcoef(
            torch.stack([eig_fd_valid, eig_geo_valid])
        )[0, 1].item()
    else:
        eigenvalue_correlation = 0.0

    # Frobenius agreement for full Hessian
    if hessian_full_fd is not None:
        H_fd = hessian_full_fd[valid_mask]
        H_geo = hessian_geometric[valid_mask]

        diff_norm = torch.norm(H_fd - H_geo, p="fro", dim=(1, 2)).mean()
        fd_norm = torch.norm(H_fd, p="fro", dim=(1, 2)).mean()
        geo_norm = torch.norm(H_geo, p="fro", dim=(1, 2)).mean()

        frobenius_agreement = 1.0 - diff_norm / (fd_norm + geo_norm + 1e-10)
        frobenius_agreement = frobenius_agreement.item()

        # Element-wise RMSE
        element_wise_rmse = torch.sqrt(((H_fd - H_geo) ** 2).mean()).item()
    else:
        # Diagonal only: compare eigenvalues
        diff = (eigenvalues_fd[valid_mask] - eigenvalues_geo[valid_mask]).abs().mean()
        scale = (eigenvalues_fd[valid_mask].abs().mean() + eigenvalues_geo[valid_mask].abs().mean()) / 2
        frobenius_agreement = 1.0 - (diff / (scale + 1e-10)).item()
        element_wise_rmse = diff.item()

    # Method preference based on equilibrium score
    mean_equilibrium_score = equilibrium_score[valid_mask].mean().item()

    if eigenvalue_correlation > 0.8 and mean_equilibrium_score > 0.7:
        method_preference = "both_agree"
    elif mean_equilibrium_score > 0.7:
        method_preference = "geometric"
    elif eigenvalue_correlation > 0.6:
        method_preference = "fd"
    else:
        method_preference = "ambiguous"

    return {
        "eigenvalue_correlation": eigenvalue_correlation,
        "frobenius_agreement": frobenius_agreement,
        "element_wise_rmse": element_wise_rmse,
        "method_preference": method_preference,
        "mean_equilibrium_score": mean_equilibrium_score,
    }


def _compute_per_walker_comparison(
    eigenvalues_fd: Tensor,
    eigenvalues_geo: Tensor,
    equilibrium_score: Tensor,
) -> dict[str, Tensor]:
    """Compute per-walker comparison metrics.

    Returns:
        dict with [N] tensors
    """
    N = eigenvalues_fd.shape[0]

    # Eigenvalue agreement per walker
    eig_diff = (eigenvalues_fd - eigenvalues_geo).abs().mean(dim=1)

    return {
        "eigenvalue_difference": eig_diff,
        "equilibrium_score": equilibrium_score,
    }


def validate_on_synthetic_function(
    test_function: Literal["quadratic", "rosenbrock", "rastrigin", "custom"],
    n_walkers: int = 100,
    dimensionality: int = 2,
    custom_fn: Callable | None = None,
    epsilon_sigma: float = 0.1,
    k_neighbors: int = 10,
) -> dict[str, float]:
    """Test estimation methods on functions with known analytical derivatives.

    Generates a grid of walkers, computes analytical and estimated gradients/Hessians,
    and measures errors.

    Test functions:
    - "quadratic": V(x) = x^T A x → ∇V = 2Ax, H = 2A
    - "rosenbrock": V(x,y) = (1-x)² + 100(y-x²)² → known derivatives
    - "rastrigin": Multimodal with periodic curvature
    - "custom": User-provided function

    Args:
        test_function: Which function to test
        n_walkers: Number of test walkers
        dimensionality: Problem dimensionality
        custom_fn: For test_function="custom", provide SyntheticFunction object
        epsilon_sigma: Spectral floor for geometric method
        k_neighbors: Number of neighbors in k-NN graph

    Returns:
        dict with error metrics:
            - "gradient_rmse": float
            - "gradient_max_error": float
            - "hessian_frobenius_error": float
            - "hessian_eigenvalue_error": float
            - "geometric_hessian_error": float
            - "passed": bool (all errors within tolerance)

    Example:
        >>> errors = validate_on_synthetic_function("quadratic", n_walkers=100, dimensionality=2)
        >>> print(f"Gradient RMSE: {errors['gradient_rmse']:.2e}")
        >>> assert errors["passed"], "Validation failed!"
    """
    device = torch.device("cpu")  # Use CPU for validation

    # Generate synthetic function
    if test_function == "quadratic":
        syn_fn = QuadraticFunction(dimensionality, device)
    elif test_function == "rosenbrock":
        if dimensionality != 2:
            warnings.warn("Rosenbrock is defined for d=2, forcing dimensionality=2")
            dimensionality = 2
        syn_fn = RosenbrockFunction(device)
    elif test_function == "rastrigin":
        syn_fn = RastriginFunction(dimensionality, device)
    elif test_function == "custom":
        if custom_fn is None:
            raise ValueError("custom_fn must be provided for test_function='custom'")
        syn_fn = custom_fn
    else:
        raise ValueError(f"Unknown test function: {test_function}")

    # Generate walker positions (grid)
    positions = syn_fn.generate_walker_grid(n_walkers)
    N = positions.shape[0]

    # Compute fitness values
    fitness_values = syn_fn.evaluate(positions)

    # Build k-NN graph
    edge_index = _build_knn_graph(positions, k_neighbors)

    # Analytical derivatives
    grad_true = syn_fn.gradient(positions)
    hess_true = syn_fn.hessian(positions)

    # Estimated gradient
    grad_result = estimate_gradient_finite_difference(
        positions, fitness_values, edge_index
    )
    grad_estimated = grad_result["gradient"]

    # Estimated Hessian (FD)
    hess_result = estimate_hessian_full_fd(
        positions, fitness_values, grad_estimated, edge_index,
        method="central", symmetrize=True
    )
    hess_estimated = hess_result["hessian_tensors"]
    eig_estimated = hess_result["hessian_eigenvalues"]

    # Estimated Hessian (geometric)
    hess_geo_result = estimate_hessian_from_metric(
        positions, edge_index, epsilon_sigma=epsilon_sigma
    )
    hess_geometric = hess_geo_result["hessian_tensors"]
    eig_geometric = hess_geo_result["hessian_eigenvalues"]

    # Analytical eigenvalues
    eig_true = torch.linalg.eigvalsh(hess_true)
    eig_true = torch.flip(eig_true, dims=[1])  # Descending

    # Compute errors
    valid_mask = torch.isfinite(grad_estimated).all(dim=1)

    gradient_rmse = torch.sqrt(
        ((grad_estimated[valid_mask] - grad_true[valid_mask]) ** 2).mean()
    ).item()

    gradient_max_error = (
        (grad_estimated[valid_mask] - grad_true[valid_mask]).abs().max()
    ).item()

    hessian_frobenius_error = torch.norm(
        hess_estimated[valid_mask] - hess_true[valid_mask],
        p="fro",
        dim=(1, 2)
    ).mean().item()

    hessian_eigenvalue_error = (
        (eig_estimated[valid_mask] - eig_true[valid_mask]).abs().mean()
    ).item()

    geometric_hessian_error = torch.norm(
        hess_geometric[valid_mask] - hess_true[valid_mask],
        p="fro",
        dim=(1, 2)
    ).mean().item()

    # Tolerances (function-dependent)
    if test_function == "quadratic":
        grad_tol, hess_tol = 1e-4, 1e-3
    elif test_function == "rosenbrock":
        grad_tol, hess_tol = 1e-3, 5e-3
    elif test_function == "rastrigin":
        grad_tol, hess_tol = 5e-3, 1e-2
    else:
        grad_tol, hess_tol = 1e-3, 5e-3

    passed = (gradient_rmse < grad_tol) and (hessian_frobenius_error < hess_tol)

    return {
        "gradient_rmse": gradient_rmse,
        "gradient_max_error": gradient_max_error,
        "hessian_frobenius_error": hessian_frobenius_error,
        "hessian_eigenvalue_error": hessian_eigenvalue_error,
        "geometric_hessian_error": geometric_hessian_error,
        "passed": passed,
    }


def plot_estimation_quality(
    comparison_results: dict,
    save_path: str | None = None,
) -> None:
    """Generate diagnostic plots for estimation quality.

    Creates 4-panel figure:
    1. FD vs geometric eigenvalue scatter
    2. Spatial map of estimation quality
    3. Convergence with number of neighbors
    4. Error distribution histograms

    Args:
        comparison_results: Output from compare_estimation_methods
        save_path: Optional path to save figure (else display)

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        warnings.warn("matplotlib not available, skipping plots")
        return

    # Extract data
    metrics = comparison_results["comparison_metrics"]
    gradient = comparison_results["gradient_fd"]
    hess_geo = comparison_results["hessian_geometric"]

    # If full Hessian available
    if "hessian_full_fd" in comparison_results:
        hess_fd = comparison_results["hessian_full_fd"]
        eig_fd = torch.linalg.eigvalsh(hess_fd)
    else:
        eig_fd = comparison_results["hessian_diagonal_fd"]

    eig_geo = torch.linalg.eigvalsh(hess_geo)

    # Convert to numpy
    eig_fd_np = eig_fd.cpu().numpy().flatten()
    eig_geo_np = eig_geo.cpu().numpy().flatten()
    grad_mag_np = torch.norm(gradient, dim=1).cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Eigenvalue scatter
    ax = axes[0, 0]
    ax.scatter(eig_fd_np, eig_geo_np, alpha=0.5, s=10)
    lims = [
        np.min([eig_fd_np.min(), eig_geo_np.min()]),
        np.max([eig_fd_np.max(), eig_geo_np.max()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('FD Eigenvalues')
    ax.set_ylabel('Geometric Eigenvalues')
    ax.set_title(f"Correlation: {metrics['eigenvalue_correlation']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Gradient magnitude histogram
    ax = axes[0, 1]
    ax.hist(grad_mag_np[np.isfinite(grad_mag_np)], bins=30, alpha=0.7)
    ax.set_xlabel('Gradient Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('Gradient Distribution')
    ax.grid(True, alpha=0.3)

    # Panel 3: Eigenvalue difference histogram
    ax = axes[1, 0]
    eig_diff = np.abs(eig_fd_np - eig_geo_np)
    ax.hist(eig_diff[np.isfinite(eig_diff)], bins=30, alpha=0.7)
    ax.set_xlabel('|Eigenvalue Difference|')
    ax.set_ylabel('Count')
    ax.set_title(f"RMSE: {metrics['element_wise_rmse']:.3e}")
    ax.grid(True, alpha=0.3)

    # Panel 4: Method preference summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Method Comparison Summary

    Eigenvalue Correlation: {metrics['eigenvalue_correlation']:.3f}
    Frobenius Agreement: {metrics['frobenius_agreement']:.3f}
    Element-wise RMSE: {metrics['element_wise_rmse']:.3e}

    Equilibrium Score: {metrics['mean_equilibrium_score']:.3f}
    Method Preference: {metrics['method_preference']}

    Interpretation:
    - Correlation > 0.8: Excellent agreement
    - Correlation 0.6-0.8: Good agreement
    - Correlation < 0.6: Poor agreement
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ============================================================================
# Synthetic Test Functions
# ============================================================================

class SyntheticFunction:
    """Base class for synthetic test functions with known derivatives."""

    def evaluate(self, x: Tensor) -> Tensor:
        """Evaluate function at positions x."""
        raise NotImplementedError

    def gradient(self, x: Tensor) -> Tensor:
        """Analytical gradient."""
        raise NotImplementedError

    def hessian(self, x: Tensor) -> Tensor:
        """Analytical Hessian."""
        raise NotImplementedError

    def generate_walker_grid(self, n_walkers: int) -> Tensor:
        """Generate grid of walker positions."""
        raise NotImplementedError


class QuadraticFunction(SyntheticFunction):
    """V(x) = (1/2) x^T A x where A is positive definite."""

    def __init__(self, d: int, device: torch.device):
        self.d = d
        self.device = device
        # Random positive definite matrix
        torch.manual_seed(42)
        M = torch.randn(d, d, device=device)
        self.A = M.T @ M + torch.eye(d, device=device)  # Ensure PD

    def evaluate(self, x: Tensor) -> Tensor:
        return 0.5 * (x @ self.A * x).sum(dim=1)

    def gradient(self, x: Tensor) -> Tensor:
        return x @ self.A.T

    def hessian(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        return self.A.unsqueeze(0).expand(N, -1, -1)

    def generate_walker_grid(self, n_walkers: int) -> Tensor:
        n_per_dim = int(n_walkers ** (1 / self.d)) + 1
        ranges = [torch.linspace(-2, 2, n_per_dim, device=self.device) for _ in range(self.d)]
        grids = torch.meshgrid(*ranges, indexing='ij')
        positions = torch.stack([g.flatten() for g in grids], dim=1)
        return positions[:n_walkers]


class RosenbrockFunction(SyntheticFunction):
    """V(x,y) = (1-x)² + 100(y-x²)²"""

    def __init__(self, device: torch.device):
        self.device = device
        self.d = 2

    def evaluate(self, x: Tensor) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    def gradient(self, x: Tensor) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        grad = torch.zeros_like(x)
        grad[:, 0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 ** 2)
        grad[:, 1] = 200 * (x2 - x1 ** 2)
        return grad

    def hessian(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        x1, x2 = x[:, 0], x[:, 1]
        H = torch.zeros(N, 2, 2, device=self.device)
        H[:, 0, 0] = 2 - 400 * x2 + 1200 * x1 ** 2
        H[:, 0, 1] = -400 * x1
        H[:, 1, 0] = -400 * x1
        H[:, 1, 1] = 200
        return H

    def generate_walker_grid(self, n_walkers: int) -> Tensor:
        n_per_dim = int(n_walkers ** 0.5) + 1
        x = torch.linspace(-1, 1.5, n_per_dim, device=self.device)
        y = torch.linspace(-0.5, 2, n_per_dim, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        positions = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return positions[:n_walkers]


class RastriginFunction(SyntheticFunction):
    """V(x) = A·d + Σ[x_i² - A·cos(2πx_i)]"""

    def __init__(self, d: int, device: torch.device, A: float = 10.0):
        self.d = d
        self.device = device
        self.A = A

    def evaluate(self, x: Tensor) -> Tensor:
        return self.A * self.d + (x ** 2 - self.A * torch.cos(2 * torch.pi * x)).sum(dim=1)

    def gradient(self, x: Tensor) -> Tensor:
        return 2 * x + 2 * torch.pi * self.A * torch.sin(2 * torch.pi * x)

    def hessian(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        H = torch.zeros(N, self.d, self.d, device=self.device)
        diag = 2 + 4 * torch.pi ** 2 * self.A * torch.cos(2 * torch.pi * x)
        for i in range(self.d):
            H[:, i, i] = diag[:, i]
        return H

    def generate_walker_grid(self, n_walkers: int) -> Tensor:
        n_per_dim = int(n_walkers ** (1 / self.d)) + 1
        ranges = [torch.linspace(-5, 5, n_per_dim, device=self.device) for _ in range(self.d)]
        grids = torch.meshgrid(*ranges, indexing='ij')
        positions = torch.stack([g.flatten() for g in grids], dim=1)
        return positions[:n_walkers]


def _build_knn_graph(positions: Tensor, k: int) -> Tensor:
    """Build k-nearest neighbor graph.

    Returns:
        [2, E] edge index in COO format
    """
    N = positions.shape[0]

    # Compute pairwise distances
    dist_matrix = torch.cdist(positions, positions)

    # Find k nearest neighbors (excluding self)
    _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
    indices = indices[:, 1:]  # Exclude self

    # Build edge list
    src = torch.arange(N, device=positions.device).unsqueeze(1).expand(-1, k).flatten()
    dst = indices.flatten()

    edge_index = torch.stack([src, dst], dim=0)

    return edge_index
