"""Common observable measurement functions for gauge symmetry tests.

This module provides utilities for computing physical observables, collective fields,
and statistical measurements used across gauge symmetry tests.

**Framework References:**
- old_docs/source/13_fractal_set_new/04c_test_cases.md - Test measurement procedures
- old_docs/source/13_fractal_set_new/04_symmetry_redefinition_viability_analysis.md - Collective fields
- docs/source/1_euclidean_gas/03_cloning.md - Measurement pipeline

**Key Concepts:**
- Collective fields d'_i, r'_i depend on entire swarm through ρ-localized statistics
- Cloning score S_i(j) compares fitness between walkers
- Observables include correlations, gradients, perturbation responses
"""

import holoviews as hv
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import curve_fit
import torch
from torch import Tensor

from fragile.fractalai.core.companion_selection import compute_algorithmic_distance_matrix
from fragile.fractalai.core.fitness import compute_fitness


class ObservablesConfig(BaseModel):
    """Configuration for observable measurements.

    Attributes:
        h_eff: Effective Planck constant (ℏ_eff) for phase scaling
        epsilon_clone: Regularization for cloning score denominator
        beta: Diversity channel exponent for fitness
        alpha: Reward channel exponent for fitness
        eta: Positivity floor for rescaling
        A: Upper bound for logistic rescale
        lambda_alg: Velocity weight in algorithmic distance
    """

    h_eff: float = Field(default=1.0, gt=0, description="Effective Planck constant (ℏ_eff)")
    epsilon_clone: float = Field(default=1e-8, gt=0, description="Cloning score regularization")
    beta: float = Field(default=1.0, gt=0, description="Diversity channel exponent")
    alpha: float = Field(default=1.0, gt=0, description="Reward channel exponent")
    eta: float = Field(default=0.1, gt=0, description="Positivity floor")
    A: float = Field(default=2.0, gt=0, description="Logistic rescale upper bound")
    lambda_alg: float = Field(default=0.0, ge=0, description="Velocity weight in distance")


def compute_collective_fields(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    alive: Tensor,
    companions: Tensor,
    rho: float | None = None,
    config: ObservablesConfig | None = None,
) -> dict[str, Tensor]:
    """Compute collective field values d'_i and r'_i for all walkers.

    Implements the measurement pipeline from §2.3 in:
    old_docs/source/13_fractal_set_new/04_symmetry_redefinition_viability_analysis.md

    The pipeline:
    1. Raw measurements: d_i, r_i
    2. Swarm aggregation: μ, σ (global or ρ-localized)
    3. Standardization: z-scores
    4. Rescaling: logistic function g_A(z)
    5. Collective fields: d'_i = g_A(z_d,i) + η, r'_i = g_A(z_r,i) + η

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        alive: Boolean mask [N]
        companions: Companion indices for diversity measurement [N]
        rho: Localization scale (None for mean-field, finite for local regime)
        config: Configuration parameters

    Returns:
        Dictionary with keys:
            - "d_prime": Diversity field values [N]
            - "r_prime": Reward field values [N]
            - "z_distances": Distance z-scores [N]
            - "z_rewards": Reward z-scores [N]
            - "mu_distances": Distance mean (scalar or [N] if localized)
            - "sigma_distances": Distance std (scalar or [N] if localized)
            - "mu_rewards": Reward mean (scalar or [N] if localized)
            - "sigma_rewards": Reward std (scalar or [N] if localized)
            - "fitness": Fitness potential [N]

    Example:
        >>> positions = torch.randn(100, 2)
        >>> velocities = torch.randn(100, 2) * 0.1
        >>> rewards = torch.randn(100)
        >>> alive = torch.ones(100, dtype=torch.bool)
        >>> companions = torch.randint(0, 100, (100,))
        >>>
        >>> # Mean-field regime
        >>> fields_mf = compute_collective_fields(positions, velocities, rewards, alive, companions, rho=None)
        >>>
        >>> # Local regime
        >>> fields_local = compute_collective_fields(
        ...     positions, velocities, rewards, alive, companions, rho=0.05
        ... )
    """
    if config is None:
        config = ObservablesConfig()

    # Compute fitness with ρ-localized or global statistics
    fitness, info = compute_fitness(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        alpha=config.alpha,
        beta=config.beta,
        eta=config.eta,
        lambda_alg=config.lambda_alg,
        A=config.A,
        rho=rho,
    )

    return {
        "d_prime": info["rescaled_distances"],  # d'_i
        "r_prime": info["rescaled_rewards"],  # r'_i
        "z_distances": info["z_distances"],
        "z_rewards": info["z_rewards"],
        "mu_distances": info["mu_distances"],
        "sigma_distances": info["sigma_distances"],
        "mu_rewards": info["mu_rewards"],
        "sigma_rewards": info["sigma_rewards"],
        "fitness": fitness,
        "distances": info["distances"],
    }


def compute_cloning_score(
    fitness: Tensor,
    alive: Tensor,
    clone_companions: Tensor,
    epsilon_clone: float = 1e-8,
) -> Tensor:
    """Compute cloning score S_i(j) = (V_j - V_i) / (V_i + ε_clone).

    From {prf:ref}`def-cloning-score-recap` in:
    old_docs/source/13_fractal_set_new/04_symmetry_redefinition_viability_analysis.md

    Args:
        fitness: Fitness potential [N]
        alive: Boolean mask [N]
        clone_companions: Companion indices for cloning [N]
        epsilon_clone: Regularization constant

    Returns:
        Cloning scores [N], zero for dead walkers

    Example:
        >>> fitness = torch.tensor([1.0, 2.0, 1.5, 0.5])
        >>> alive = torch.tensor([True, True, True, False])
        >>> companions = torch.tensor([1, 2, 0, 0])
        >>> scores = compute_cloning_score(fitness, alive, companions)
        >>> # S_0(1) = (2.0 - 1.0) / (1.0 + eps) ≈ 1.0
        >>> # S_1(2) = (1.5 - 2.0) / (2.0 + eps) ≈ -0.25
    """
    # Get companion fitness values
    fitness_companion = fitness[clone_companions]

    # Compute cloning score: (V_j - V_i) / (V_i + ε)
    score = (fitness_companion - fitness) / (fitness + epsilon_clone)

    # Mask dead walkers
    return torch.where(alive, score, torch.zeros_like(score))


def compute_fitness_statistics(fitness: Tensor, alive: Tensor) -> dict[str, float]:
    """Compute statistical summary of fitness distribution.

    Args:
        fitness: Fitness potential [N]
        alive: Boolean mask [N]

    Returns:
        Dictionary with keys: mean, std, min, max, median
    """
    fitness_alive = fitness[alive]

    return {
        "mean": fitness_alive.mean().item(),
        "std": fitness_alive.std().item(),
        "min": fitness_alive.min().item(),
        "max": fitness_alive.max().item(),
        "median": fitness_alive.median().item(),
    }


def bin_by_distance(
    positions: Tensor,
    values: Tensor,
    alive: Tensor,
    r_max: float = 0.5,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin scalar values by pairwise distance for correlation analysis.

    Used for Test 1A (spatial correlation) in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md §1.2

    Args:
        positions: Walker positions [N, d]
        values: Scalar field values [N] (e.g., d'_i)
        alive: Boolean mask [N]
        r_max: Maximum distance to consider
        n_bins: Number of distance bins

    Returns:
        Tuple of (bin_centers, binned_correlations, bin_counts):
            - bin_centers: Distance bin centers [n_bins]
            - binned_correlations: Mean correlation in each bin [n_bins]
            - bin_counts: Number of pairs in each bin [n_bins]

    Example:
        >>> positions = torch.randn(100, 2)
        >>> d_prime = torch.randn(100)
        >>> alive = torch.ones(100, dtype=torch.bool)
        >>> r, C, counts = bin_by_distance(positions, d_prime, alive)
        >>> # C(r) = ⟨d'_i · d'_j⟩ for pairs at distance r
    """
    # Compute pairwise distances [N, N]
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.sqrt((pos_diff**2).sum(dim=-1))

    # Compute pairwise products [N, N]
    products = values.unsqueeze(1) * values.unsqueeze(0)

    # Mask: only alive pairs, exclude diagonal, distance <= r_max
    mask = (
        alive.unsqueeze(1)
        & alive.unsqueeze(0)
        & (torch.eye(len(alive), device=alive.device, dtype=torch.bool).logical_not())
        & (distances <= r_max)
    )

    # Extract valid pairs
    distances_valid = distances[mask].cpu().numpy()
    products_valid = products[mask].cpu().numpy()

    # Bin by distance
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_correlations = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (distances_valid >= bin_edges[i]) & (distances_valid < bin_edges[i + 1])
        if in_bin.sum() > 0:
            binned_correlations[i] = products_valid[in_bin].mean()
            bin_counts[i] = in_bin.sum()

    return bin_centers, binned_correlations, bin_counts


def fit_exponential_decay(
    r: np.ndarray, C: np.ndarray, counts: np.ndarray | None = None
) -> dict[str, float]:
    """Fit correlation function to exponential decay C(r) = C_0 · exp(-r²/ξ²).

    Used for extracting correlation length ξ from Test 1A in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md §1.2

    Args:
        r: Distance values
        C: Correlation values
        counts: Optional weights for fitting (number of pairs per bin)

    Returns:
        Dictionary with keys: C0, xi (correlation length), r_squared (fit quality)

    Example:
        >>> r, C, counts = bin_by_distance(positions, d_prime, alive)
        >>> fit_params = fit_exponential_decay(r, C, counts)
        >>> print(f"Correlation length: ξ = {fit_params['xi']:.4f}")
    """
    # Filter out zero counts
    if counts is not None:
        valid = counts > 0
        r = r[valid]
        C = C[valid]
        weights = counts[valid]
    else:
        weights = None

    # Define exponential decay model
    def model(r_vals, C0, xi):
        return C0 * np.exp(-(r_vals**2) / (xi**2))

    # Initial guess: C0 = max(C), xi = r[C ~ C0/e]
    C0_guess = C.max() if len(C) > 0 else 1.0
    xi_guess = r.max() / 3 if len(r) > 0 else 0.1

    try:
        # Fit with or without weights
        if weights is not None:
            sigma = 1 / np.sqrt(weights + 1)
            popt, _pcov = curve_fit(
                model, r, C, p0=[C0_guess, xi_guess], sigma=sigma, maxfev=10000
            )
        else:
            popt, _pcov = curve_fit(model, r, C, p0=[C0_guess, xi_guess], maxfev=10000)

        # Compute R²
        residuals = C - model(r, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((C - C.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"C0": popt[0], "xi": popt[1], "r_squared": r_squared}

    except (RuntimeError, ValueError):
        # Fit failed
        return {"C0": 0.0, "xi": 0.0, "r_squared": 0.0}


def compute_field_gradients(
    positions: Tensor, field_values: Tensor, alive: Tensor, k_neighbors: int = 5
) -> Tensor:
    """Compute spatial gradient magnitude |∇field| for each walker.

    Used for Test 1B (field gradient magnitude) in:
    old_docs/source/13_fractal_set_new/04c_test_cases.md §1.3

    Approximates gradient using finite differences with k nearest neighbors:
        ∇field(x_i) ≈ (field_j - field_i) / |x_j - x_i| for nearest neighbor j

    Args:
        positions: Walker positions [N, d]
        field_values: Scalar field [N] (e.g., d'_i)
        alive: Boolean mask [N]
        k_neighbors: Number of neighbors to average over

    Returns:
        Gradient magnitudes [N], zero for dead walkers

    Example:
        >>> positions = torch.randn(100, 2)
        >>> d_prime = torch.randn(100)
        >>> alive = torch.ones(100, dtype=torch.bool)
        >>> gradients = compute_field_gradients(positions, d_prime, alive)
        >>> grad_mean = gradients[alive].mean()
        >>> # For local field: expect |∇d'| ~ O(1/ρ)
    """
    N = positions.shape[0]

    # Compute pairwise distances [N, N]
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.sqrt((pos_diff**2).sum(dim=-1))

    # Set diagonal and dead walkers to infinity (to exclude from nearest neighbor search)
    distances_masked = distances.clone()
    distances_masked[~alive, :] = float("inf")
    distances_masked[:, ~alive] = float("inf")
    distances_masked.fill_diagonal_(float("inf"))

    # Find k nearest neighbors for each walker
    _, nearest_idx = torch.topk(distances_masked, k=k_neighbors, dim=1, largest=False)  # [N, k]

    # Compute gradient estimates for each neighbor
    gradients_list = []
    for k in range(k_neighbors):
        neighbor_idx = nearest_idx[:, k]
        field_diff = field_values[neighbor_idx] - field_values
        dist = distances[torch.arange(N), neighbor_idx]
        gradient_magnitude = torch.abs(field_diff) / torch.clamp(dist, min=1e-10)
        gradients_list.append(gradient_magnitude)

    # Average over neighbors
    gradients = torch.stack(gradients_list, dim=1).mean(dim=1)

    # Mask dead walkers
    return torch.where(alive, gradients, torch.zeros_like(gradients))


def plot_field_configuration(
    positions: Tensor,
    field_values: Tensor,
    alive: Tensor,
    field_name: str = "d'",
    title: str = "Field Configuration",
) -> hv.Element:
    """Visualize scalar field configuration in 2D.

    Args:
        positions: Walker positions [N, 2]
        field_values: Scalar field values [N]
        alive: Boolean mask [N]
        field_name: Name for color axis label
        title: Plot title

    Returns:
        HoloViews Points element with Bokeh backend

    Example:
        >>> hv.extension("bokeh")
        >>> positions = torch.randn(100, 2)
        >>> d_prime = torch.randn(100)
        >>> alive = torch.ones(100, dtype=torch.bool)
        >>> plot = plot_field_configuration(positions, d_prime, alive, "d'")
        >>> hv.save(plot, "field_config.html")
    """
    # Extract alive walkers
    pos = positions[alive].cpu().numpy()
    vals = field_values[alive].cpu().numpy()

    # Create scatter plot
    points = hv.Points(
        data={"x": pos[:, 0], "y": pos[:, 1], field_name: vals},
        kdims=["x", "y"],
        vdims=[field_name],
    )

    return points.opts(
        color=field_name,
        cmap="viridis",
        colorbar=True,
        size=5,
        title=title,
        width=600,
        height=500,
        tools=["hover"],
    )


def compute_distance_matrix(
    positions: Tensor, velocities: Tensor, lambda_alg: float = 0.0
) -> Tensor:
    """Compute pairwise algorithmic distance matrix.

    Wrapper around fragile.core.companion_selection.compute_algorithmic_distance_matrix.

    d_alg²(i,j) = ||x_i - x_j||² + λ_alg ||v_i - v_j||²

    Args:
        positions: Positions [N, d]
        velocities: Velocities [N, d]
        lambda_alg: Velocity weight

    Returns:
        Distance matrix [N, N] containing d_alg²(i,j)
    """
    return compute_algorithmic_distance_matrix(positions, velocities, lambda_alg)
