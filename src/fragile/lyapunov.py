"""
Lyapunov Functions for Swarm Convergence Analysis

This module implements Lyapunov functions to numerically verify convergence properties
of Euclidean Gas swarms. It compares two swarm states and returns all relevant terms
as tensors for analysis.

Mathematical Background:
- Lyapunov functions V(t) measure "distance" from equilibrium
- If dV/dt ≤ 0, the system converges
- We decompose V into multiple components:
  * Position variance: spread in state space
  * Velocity variance: kinetic energy distribution
  * Cross-swarm distances: relative positions between swarms
  * Wasserstein-like metrics: distribution matching

References:
- See docs/source/1_fractal_calculus/theorems/ for convergence proofs
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.bounds import TorchBounds
from fragile.euclidean_gas import EuclideanGas, SwarmState, VectorizedOps


class LyapunovTerms:
    """Container for all Lyapunov function terms."""

    def __init__(
        self,
        # Variance terms (within-swarm spread)
        var_x_1: Tensor,
        var_x_2: Tensor,
        var_v_1: Tensor,
        var_v_2: Tensor,
        # Cross-swarm distance terms
        mean_cross_distance_x: Tensor,
        mean_cross_distance_v: Tensor,
        wasserstein_x: Tensor,  # 1-Wasserstein approximation for positions
        wasserstein_v: Tensor,  # 1-Wasserstein approximation for velocities
        # Mean distance terms (center of mass)
        com_distance_x: Tensor,  # ||μ_x^(1) - μ_x^(2)||
        com_distance_v: Tensor,  # ||μ_v^(1) - μ_v^(2)||
        # Total Lyapunov function
        total: Tensor,
    ):
        """
        Initialize Lyapunov terms.

        Args:
            var_x_1: Position variance of swarm 1
            var_x_2: Position variance of swarm 2
            var_v_1: Velocity variance of swarm 1
            var_v_2: Velocity variance of swarm 2
            mean_cross_distance_x: Mean cross-swarm position distance
            mean_cross_distance_v: Mean cross-swarm velocity distance
            wasserstein_x: 1-Wasserstein distance approximation for positions
            wasserstein_v: 1-Wasserstein distance approximation for velocities
            com_distance_x: Distance between position centers of mass
            com_distance_v: Distance between velocity centers of mass
            total: Total Lyapunov function value
        """
        self.var_x_1 = var_x_1
        self.var_x_2 = var_x_2
        self.var_v_1 = var_v_1
        self.var_v_2 = var_v_2
        self.mean_cross_distance_x = mean_cross_distance_x
        self.mean_cross_distance_v = mean_cross_distance_v
        self.wasserstein_x = wasserstein_x
        self.wasserstein_v = wasserstein_v
        self.com_distance_x = com_distance_x
        self.com_distance_v = com_distance_v
        self.total = total

    def to_dict(self) -> dict[str, Tensor]:
        """
        Convert all terms to a dictionary of tensors.

        Returns:
            Dictionary mapping term names to tensor values
        """
        return {
            "var_x_1": self.var_x_1,
            "var_x_2": self.var_x_2,
            "var_v_1": self.var_v_1,
            "var_v_2": self.var_v_2,
            "mean_cross_distance_x": self.mean_cross_distance_x,
            "mean_cross_distance_v": self.mean_cross_distance_v,
            "wasserstein_x": self.wasserstein_x,
            "wasserstein_v": self.wasserstein_v,
            "com_distance_x": self.com_distance_x,
            "com_distance_v": self.com_distance_v,
            "total": self.total,
        }


def compute_wasserstein_1d_approx(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Compute 1-Wasserstein distance approximation between two point clouds.

    Uses the Hungarian algorithm approximation via sorting along each dimension
    and averaging. This is exact for 1D and a good approximation for higher dimensions.

    W_1(μ_1, μ_2) ≈ (1/d) Σ_j W_1(μ_1^j, μ_2^j)

    where μ_i^j is the marginal distribution along dimension j.

    Args:
        x1: Points from distribution 1 [N1, d]
        x2: Points from distribution 2 [N2, d]

    Returns:
        Approximate 1-Wasserstein distance (scalar)
    """
    # Handle different sample sizes by repeating smaller set
    N1, d = x1.shape
    N2 = x2.shape[0]

    if N1 != N2:
        # Match sizes by repeating
        N_max = max(N1, N2)
        if N1 < N_max:
            # Repeat x1
            repeats = (N_max + N1 - 1) // N1  # Ceiling division
            x1 = x1.repeat(repeats, 1)[:N_max]
        if N2 < N_max:
            # Repeat x2
            repeats = (N_max + N2 - 1) // N2
            x2 = x2.repeat(repeats, 1)[:N_max]

    # For each dimension, sort both distributions and compute L1 distance
    wasserstein_per_dim = torch.zeros(d, device=x1.device, dtype=x1.dtype)

    for j in range(d):
        x1_sorted = torch.sort(x1[:, j])[0]
        x2_sorted = torch.sort(x2[:, j])[0]
        wasserstein_per_dim[j] = torch.mean(torch.abs(x1_sorted - x2_sorted))

    # Average over dimensions
    return torch.mean(wasserstein_per_dim)


def compute_cross_distance_mean(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Compute mean pairwise distance between two point clouds.

    D_cross = (1/(N1*N2)) Σ_i Σ_j ||x1_i - x2_j||

    Args:
        x1: Points from cloud 1 [N1, d]
        x2: Points from cloud 2 [N2, d]

    Returns:
        Mean cross-distance (scalar)
    """
    # Pairwise distances [N1, N2]
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # [N1, N2, d]
    distances = torch.norm(diff, dim=2)  # [N1, N2]

    return torch.mean(distances)


def compute_lyapunov(
    state1: SwarmState,
    state2: SwarmState,
    weight_var: float = 1.0,
    weight_cross: float = 1.0,
    weight_wasserstein: float = 1.0,
    weight_com: float = 1.0,
) -> LyapunovTerms:
    """
    Compute all Lyapunov function terms for two swarm states.

    The total Lyapunov function is a weighted sum:

    V(t) = w_var * (var_x_1 + var_x_2 + var_v_1 + var_v_2)
         + w_cross * (D_cross_x + D_cross_v)
         + w_wasserstein * (W_x + W_v)
         + w_com * (||Δμ_x|| + ||Δμ_v||)

    Args:
        state1: First swarm state
        state2: Second swarm state
        weight_var: Weight for variance terms
        weight_cross: Weight for cross-distance terms
        weight_wasserstein: Weight for Wasserstein terms
        weight_com: Weight for center-of-mass distance terms

    Returns:
        LyapunovTerms object containing all terms
    """
    # Within-swarm variance terms
    var_x_1 = VectorizedOps.variance_position(state1)
    var_x_2 = VectorizedOps.variance_position(state2)
    var_v_1 = VectorizedOps.variance_velocity(state1)
    var_v_2 = VectorizedOps.variance_velocity(state2)

    # Cross-swarm distance terms
    mean_cross_distance_x = compute_cross_distance_mean(state1.x, state2.x)
    mean_cross_distance_v = compute_cross_distance_mean(state1.v, state2.v)

    # Wasserstein distance approximations
    wasserstein_x = compute_wasserstein_1d_approx(state1.x, state2.x)
    wasserstein_v = compute_wasserstein_1d_approx(state1.v, state2.v)

    # Center of mass distances
    μ_x_1 = torch.mean(state1.x, dim=0)  # [d]
    μ_x_2 = torch.mean(state2.x, dim=0)  # [d]
    μ_v_1 = torch.mean(state1.v, dim=0)  # [d]
    μ_v_2 = torch.mean(state2.v, dim=0)  # [d]

    com_distance_x = torch.norm(μ_x_1 - μ_x_2)
    com_distance_v = torch.norm(μ_v_1 - μ_v_2)

    # Total Lyapunov function
    total = (
        weight_var * (var_x_1 + var_x_2 + var_v_1 + var_v_2)
        + weight_cross * (mean_cross_distance_x + mean_cross_distance_v)
        + weight_wasserstein * (wasserstein_x + wasserstein_v)
        + weight_com * (com_distance_x + com_distance_v)
    )

    return LyapunovTerms(
        var_x_1=var_x_1,
        var_x_2=var_x_2,
        var_v_1=var_v_1,
        var_v_2=var_v_2,
        mean_cross_distance_x=mean_cross_distance_x,
        mean_cross_distance_v=mean_cross_distance_v,
        wasserstein_x=wasserstein_x,
        wasserstein_v=wasserstein_v,
        com_distance_x=com_distance_x,
        com_distance_v=com_distance_v,
        total=total,
    )


def compute_lyapunov_from_gas(
    gas1: EuclideanGas,
    gas2: EuclideanGas,
    state1: SwarmState | None = None,
    state2: SwarmState | None = None,
    **kwargs,
) -> dict[str, Tensor]:
    """
    Convenience function to compute Lyapunov terms from two EuclideanGas instances.

    Args:
        gas1: First Euclidean Gas instance
        gas2: Second Euclidean Gas instance
        state1: Optional state for gas1 (if None, initializes new state)
        state2: Optional state for gas2 (if None, initializes new state)
        **kwargs: Additional arguments passed to compute_lyapunov (weights, etc.)

    Returns:
        Dictionary of tensors containing all Lyapunov terms

    Example:
        >>> from fragile.euclidean_gas import EuclideanGas, EuclideanGasParams, ...
        >>> from fragile.lyapunov import compute_lyapunov_from_gas
        >>>
        >>> # Create two gas instances with different parameters
        >>> params1 = EuclideanGasParams(...)
        >>> params2 = EuclideanGasParams(...)
        >>> gas1 = EuclideanGas(params1)
        >>> gas2 = EuclideanGas(params2)
        >>>
        >>> # Run both for some steps
        >>> state1 = gas1.initialize_state()
        >>> state2 = gas2.initialize_state()
        >>> for _ in range(100):
        >>>     _, state1 = gas1.step(state1)
        >>>     _, state2 = gas2.step(state2)
        >>>
        >>> # Compare using Lyapunov function
        >>> lyapunov = compute_lyapunov_from_gas(gas1, gas2, state1, state2)
        >>> print(f"Total Lyapunov: {lyapunov['total'].item():.6f}")
    """
    # Initialize states if not provided
    if state1 is None:
        state1 = gas1.initialize_state()
    if state2 is None:
        state2 = gas2.initialize_state()

    # Compute Lyapunov terms
    terms = compute_lyapunov(state1, state2, **kwargs)

    return terms.to_dict()


# =============================================================================
# Cluster Analysis Functions (from docs/source/03_cloning.md)
# =============================================================================


def compute_coupled_alive_partition(
    state1: SwarmState, state2: SwarmState, bounds: TorchBounds | None = None
) -> dict[str, Tensor]:
    """
    Compute the coupled alive/dead status partition for two swarms.

    This function partitions the N walker indices into four disjoint sets based on
    their alive/dead status in each swarm (see docs/source/03_cloning.md § 2.2):

    - I_11: "Stably alive" - walkers alive in both swarms
    - I_10: Alive in swarm 1, dead in swarm 2
    - I_01: Dead in swarm 1, alive in swarm 2
    - I_00: "Stably dead" - walkers dead in both swarms

    The stably alive set I_11 is the most important for convergence analysis, as
    these walkers drive the contractive dynamics of the cloning operator.

    Args:
        state1: First swarm state
        state2: Second swarm state
        bounds: Optional boundary specification. If provided, walkers are considered
                alive only if they satisfy bounds.contains(x). Otherwise, all walkers
                are considered alive.

    Returns:
        Dictionary with keys 'I_11', 'I_10', 'I_01', 'I_00' mapping to boolean masks [N]
        indicating membership in each partition set.

    Example:
        >>> partition = compute_coupled_alive_partition(state1, state2, bounds)
        >>> n_stably_alive = partition['I_11'].sum().item()
        >>> print(f"Stably alive walkers: {n_stably_alive}/{state1.x.shape[0]}")
    """
    N = state1.x.shape[0]
    device = state1.x.device

    # Determine alive status for each swarm
    if bounds is not None:
        alive_1 = bounds.contains(state1.x)  # [N]
        alive_2 = bounds.contains(state2.x)  # [N]
    else:
        # If no bounds, all walkers are alive
        alive_1 = torch.ones(N, dtype=torch.bool, device=device)
        alive_2 = torch.ones(N, dtype=torch.bool, device=device)

    # Compute the four partition sets
    I_11 = alive_1 & alive_2  # Stably alive
    I_10 = alive_1 & ~alive_2  # Alive in 1, dead in 2
    I_01 = ~alive_1 & alive_2  # Dead in 1, alive in 2
    I_00 = ~alive_1 & ~alive_2  # Stably dead

    return {"I_11": I_11, "I_10": I_10, "I_01": I_01, "I_00": I_00}


def compute_algorithmic_distance(
    x_i: Tensor, v_i: Tensor, x_j: Tensor, v_j: Tensor, lambda_alg: float
) -> Tensor:
    """
    Compute algorithmic phase-space distance between two walkers.

    d_alg(i,j)² = ||x_i - x_j||² + λ_alg ||v_i - v_j||²

    Args:
        x_i: Position of walker i [d] or [N, d]
        v_i: Velocity of walker i [d] or [N, d]
        x_j: Position of walker j [d] or [N, d]
        v_j: Velocity of walker j [d] or [N, d]
        lambda_alg: Velocity weight parameter (λ_alg ≥ 0)

    Returns:
        Algorithmic distance (scalar or [N])
    """
    dx = torch.norm(x_i - x_j, dim=-1)
    dv = torch.norm(v_i - v_j, dim=-1)
    return torch.sqrt(dx**2 + lambda_alg * dv**2)


def identify_high_error_clusters(
    state: SwarmState,
    alive_mask: Tensor,
    epsilon: float,
    lambda_alg: float = 1.0,
    c_d: float = 2.0,
    k_min_frac: float = 0.05,
    epsilon_O: float = 0.1,
) -> dict[str, Tensor]:
    """
    Identify high-error and low-error walker clusters based on phase-space geometry.

    This implements the clustering-based approach from docs/source/03_cloning.md § 6.3
    (Definition: The Unified High-Error and Low-Error Sets).

    The algorithm:
    1. Cluster alive walkers using complete-linkage clustering with diameter D_diam = c_d * ε
    2. Filter out invalid clusters (size < k_min)
    3. Identify outlier clusters that contribute most to between-cluster variance
    4. High-error set = outlier clusters + invalid clusters
    5. Low-error set = complement

    Args:
        state: Swarm state
        alive_mask: Boolean mask [N] indicating which walkers are alive
        epsilon: Interaction range parameter (ε)
        lambda_alg: Velocity weight in algorithmic distance (default: 1.0)
        c_d: Diameter constant (default: 2.0)
        k_min_frac: Minimum cluster size as fraction of alive walkers (default: 0.05)
        epsilon_O: Outlier contribution threshold (default: 0.1)

    Returns:
        Dictionary with keys:
        - 'H_k': Boolean mask [N] for high-error walkers
        - 'L_k': Boolean mask [N] for low-error walkers
        - 'cluster_labels': Integer labels [N] for cluster assignment (-1 for high-error)
        - 'n_clusters': Number of valid clusters (scalar)

    Example:
        >>> clusters = identify_high_error_clusters(state, alive_mask, epsilon=1.0)
        >>> n_high_error = clusters['H_k'].sum().item()
        >>> n_low_error = clusters['L_k'].sum().item()
        >>> print(f"High-error: {n_high_error}, Low-error: {n_low_error}")
    """
    N = state.x.shape[0]
    device = state.x.device

    # Extract alive walkers
    x_alive = state.x[alive_mask]  # [k, d]
    v_alive = state.v[alive_mask]  # [k, d]
    k = x_alive.shape[0]

    if k < 2:
        # Trivial case: too few alive walkers
        H_k = torch.zeros(N, dtype=torch.bool, device=device)
        L_k = torch.zeros(N, dtype=torch.bool, device=device)
        H_k[alive_mask] = True  # All alive walkers are high-error
        return {
            "H_k": H_k,
            "L_k": L_k,
            "cluster_labels": torch.full((N,), -1, dtype=torch.long, device=device),
            "n_clusters": torch.tensor(0, device=device),
        }

    # Step 1: Compute pairwise algorithmic distances
    D_diam = c_d * epsilon
    dist_matrix = torch.zeros(k, k, device=device, dtype=state.x.dtype)

    for i in range(k):
        for j in range(i + 1, k):
            d_ij = compute_algorithmic_distance(
                x_alive[i], v_alive[i], x_alive[j], v_alive[j], lambda_alg
            )
            dist_matrix[i, j] = d_ij
            dist_matrix[j, i] = d_ij

    # Step 2: Complete-linkage hierarchical clustering (simplified version)
    # We use a greedy approach: merge clusters until all have diameter > D_diam
    k_min = max(5, int(k_min_frac * k))

    # Initialize: each walker is its own cluster
    cluster_labels_alive = torch.arange(k, device=device, dtype=torch.long)  # [k]

    # Iteratively merge closest clusters
    max_iterations = k  # Upper bound on iterations
    for _ in range(max_iterations):
        # Compute cluster-to-cluster max distances (complete linkage)
        unique_labels = torch.unique(cluster_labels_alive)
        n_clusters = len(unique_labels)

        if n_clusters == 1:
            break  # All merged into one cluster

        # Find the pair of clusters with minimum linkage distance
        min_dist = float("inf")
        merge_pair = None

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                label_i = unique_labels[i]
                label_j = unique_labels[j]
                mask_i = cluster_labels_alive == label_i
                mask_j = cluster_labels_alive == label_j

                # Complete linkage: maximum distance between any pair
                max_dist_ij = dist_matrix[mask_i][:, mask_j].max()

                if max_dist_ij < min_dist:
                    min_dist = max_dist_ij
                    merge_pair = (label_i, label_j)

        # Stop if minimum linkage > D_diam
        if min_dist > D_diam:
            break

        # Merge the closest pair
        if merge_pair is not None:
            label_i, label_j = merge_pair
            cluster_labels_alive[cluster_labels_alive == label_j] = label_i

    # Step 3: Statistical validity constraint
    unique_labels = torch.unique(cluster_labels_alive)
    valid_clusters = []
    invalid_walkers = []

    for label in unique_labels:
        mask = cluster_labels_alive == label
        cluster_size = mask.sum().item()

        if cluster_size >= k_min:
            valid_clusters.append(label)
        else:
            invalid_walkers.append(mask)

    # Step 4: Outlier cluster identification
    # Compute center of mass for each valid cluster and global center
    mu_x = x_alive.mean(dim=0)  # [d]
    mu_v = v_alive.mean(dim=0)  # [d]

    # Compute contribution of each valid cluster
    contributions = []
    for label in valid_clusters:
        mask = cluster_labels_alive == label
        cluster_x = x_alive[mask]
        cluster_v = v_alive[mask]
        cluster_size = mask.sum().item()

        mu_x_cluster = cluster_x.mean(dim=0)
        mu_v_cluster = cluster_v.mean(dim=0)

        # Hypocoercive variance contribution
        contrib = cluster_size * (
            torch.norm(mu_x_cluster - mu_x) ** 2 + lambda_alg * torch.norm(mu_v_cluster - mu_v) ** 2
        )
        contributions.append((label, contrib))

    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x[1], reverse=True)

    # Find outlier clusters: smallest set whose cumulative contribution
    # meets or exceeds (1-ε_O) of total (Definition 6.3, line 2375-2378)
    total_contrib = sum(c[1] for c in contributions)
    target_contrib = (1 - epsilon_O) * total_contrib

    outlier_clusters = []
    cumulative_contrib = 0.0
    for label, contrib in contributions:
        outlier_clusters.append(label)
        cumulative_contrib += contrib
        if cumulative_contrib >= target_contrib:
            break

    # Step 5: Construct high-error and low-error sets
    # Map from alive indices to full indices
    alive_indices = torch.where(alive_mask)[0]

    H_k = torch.zeros(N, dtype=torch.bool, device=device)
    L_k = torch.zeros(N, dtype=torch.bool, device=device)
    cluster_labels_full = torch.full((N,), -1, dtype=torch.long, device=device)

    # Count invalid walkers
    n_invalid = sum(mask.sum().item() for mask in invalid_walkers)

    # Invalid clusters -> high-error
    if invalid_walkers:
        for mask in invalid_walkers:
            H_k[alive_indices[mask]] = True

    # Outlier clusters -> high-error
    n_outlier = 0
    for label in outlier_clusters:
        mask = cluster_labels_alive == label
        n_outlier += mask.sum().item()
        H_k[alive_indices[mask]] = True

    # Low-error set = alive but not high-error
    L_k[alive_mask] = True
    L_k[H_k] = False

    # Store cluster labels for visualization
    cluster_labels_full[alive_indices] = cluster_labels_alive

    # Debug info (can be accessed via return dict)
    n_valid_clusters = len(valid_clusters)
    n_outlier_clusters = len(outlier_clusters)

    return {
        "H_k": H_k,
        "L_k": L_k,
        "cluster_labels": cluster_labels_full,
        "n_clusters": torch.tensor(n_valid_clusters, device=device),
        "n_invalid_walkers": n_invalid,
        "n_outlier_walkers": n_outlier,
        "n_valid_clusters": n_valid_clusters,
        "n_outlier_clusters": n_outlier_clusters,
    }


def compute_cluster_metrics(
    state1: SwarmState,
    state2: SwarmState,
    bounds: TorchBounds | None = None,
    epsilon: float = 1.0,
    lambda_alg: float = 1.0,
) -> dict[str, Tensor]:
    """
    Compute comprehensive cluster-based metrics for two swarms.

    Combines alive/dead partition with high-error/low-error cluster analysis to
    provide a complete characterization of the swarm geometry.

    Args:
        state1: First swarm state
        state2: Second swarm state
        bounds: Optional boundary specification
        epsilon: Interaction range parameter
        lambda_alg: Velocity weight in algorithmic distance

    Returns:
        Dictionary containing:
        - Partition masks: 'I_11', 'I_10', 'I_01', 'I_00' (alive/dead status)
        - Cluster masks for swarm 1: 'H_1', 'L_1', 'cluster_labels_1'
        - Cluster masks for swarm 2: 'H_2', 'L_2', 'cluster_labels_2'
        - Counts: 'n_stably_alive', 'n_high_error_1', 'n_high_error_2', etc.

    Example:
        >>> metrics = compute_cluster_metrics(state1, state2, bounds, epsilon=1.0)
        >>> print(f"Stably alive: {metrics['n_stably_alive']}")
        >>> print(f"High-error in swarm 1: {metrics['n_high_error_1']}")
    """
    # Step 1: Compute alive/dead partition
    partition = compute_coupled_alive_partition(state1, state2, bounds)

    # Step 2: Identify clusters for each swarm
    clusters_1 = identify_high_error_clusters(
        state1, partition["I_11"] | partition["I_10"], epsilon, lambda_alg
    )
    clusters_2 = identify_high_error_clusters(
        state2, partition["I_11"] | partition["I_01"], epsilon, lambda_alg
    )

    # Step 3: Compute counts and statistics
    n_stably_alive = partition["I_11"].sum()
    n_high_error_1 = clusters_1["H_k"].sum()
    n_high_error_2 = clusters_2["H_k"].sum()
    n_low_error_1 = clusters_1["L_k"].sum()
    n_low_error_2 = clusters_2["L_k"].sum()

    # Critical target set: I_11 ∩ H_k (stably alive AND high-error)
    critical_target_1 = partition["I_11"] & clusters_1["H_k"]
    critical_target_2 = partition["I_11"] & clusters_2["H_k"]
    n_critical_1 = critical_target_1.sum()
    n_critical_2 = critical_target_2.sum()

    return {
        # Alive/dead partition
        "I_11": partition["I_11"],
        "I_10": partition["I_10"],
        "I_01": partition["I_01"],
        "I_00": partition["I_00"],
        # Cluster masks
        "H_1": clusters_1["H_k"],
        "L_1": clusters_1["L_k"],
        "cluster_labels_1": clusters_1["cluster_labels"],
        "H_2": clusters_2["H_k"],
        "L_2": clusters_2["L_k"],
        "cluster_labels_2": clusters_2["cluster_labels"],
        # Critical target sets
        "critical_target_1": critical_target_1,
        "critical_target_2": critical_target_2,
        # Counts
        "n_stably_alive": n_stably_alive,
        "n_high_error_1": n_high_error_1,
        "n_high_error_2": n_high_error_2,
        "n_low_error_1": n_low_error_1,
        "n_low_error_2": n_low_error_2,
        "n_critical_1": n_critical_1,
        "n_critical_2": n_critical_2,
        "n_clusters_1": clusters_1["n_clusters"],
        "n_clusters_2": clusters_2["n_clusters"],
    }
