"""Distance-dependent companion selection mechanisms for Euclidean Gas.

This module implements the companion selection operators defined in Chapter 5 and Chapter 9
of docs/source/03_cloning.md. The mechanisms use distance-dependent random matching based
on the algorithmic distance metric d_alg(i,j)^2 = ||x_i - x_j||^2 + λ_alg ||v_i - v_j||^2.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator
import torch


def compute_algorithmic_distance_matrix(
    x: torch.Tensor,
    v: torch.Tensor,
    lambda_alg: float = 0.0,
) -> torch.Tensor:
    """Compute pairwise squared algorithmic distance matrix.

    Definition from docs/source/03_cloning.md § 5.0 ({prf:ref}`def-algorithmic-distance-metric`):

    $$
    d_{alg}(i, j)^2 := ||x_i - x_j||^2 + λ_{alg} ||v_i - v_j||^2
    $$

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        lambda_alg: Weight for velocity contribution (default 0.0 for position-only)

    Returns:
        Squared distance matrix, shape [N, N]
        Entry [i, j] contains d_alg(i, j)^2

    Note:
        - λ_alg = 0: Position-only model (pure Euclidean distance)
        - λ_alg > 0: Fluid dynamics model (phase-space aware)
        - λ_alg = 1: Balanced phase-space model (position and velocity equal weight)
    """
    # Compute position distance matrix: ||x_i - x_j||^2
    # Using (x_i - x_j)^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
    x_norm_sq = torch.sum(x * x, dim=1, keepdim=True)  # [N, 1]
    x_dot = x @ x.T  # [N, N]
    pos_dist_sq = x_norm_sq + x_norm_sq.T - 2 * x_dot  # [N, N]

    if lambda_alg > 0:
        # Add velocity distance: λ_alg * ||v_i - v_j||^2
        v_norm_sq = torch.sum(v * v, dim=1, keepdim=True)  # [N, 1]
        v_dot = v @ v.T  # [N, N]
        vel_dist_sq = v_norm_sq + v_norm_sq.T - 2 * v_dot  # [N, N]
        return pos_dist_sq + lambda_alg * vel_dist_sq
    return pos_dist_sq


def select_companions_softmax(
    x: torch.Tensor,
    v: torch.Tensor,
    alive_mask: torch.Tensor,
    epsilon: float,
    lambda_alg: float = 0.0,
    exclude_self: bool = True,
) -> torch.Tensor:
    """Select companions using distance-dependent softmax distribution.

    From docs/source/03_cloning.md § 9.3.3 ({prf:ref}`def-decision-operator`):

    For alive walkers selecting companions (e.g., for cloning):

    $$
    P(c_i = j) = \\frac{\\exp\\left(-\\frac{d_{alg}(x_i, x_j)^2}{2\\epsilon^2}\\right)}
                      {\\sum_{\\ell \\in \\mathcal{A} \\setminus \\{i\\}}
                       \\exp\\left(-\\frac{d_{alg}(x_i, x_\\ell)^2}{2\\epsilon^2}\\right)}
    $$

    This creates a probability distribution strongly favoring nearby walkers in phase space.

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        alive_mask: Boolean mask indicating alive walkers, shape [N]
        epsilon: Interaction range parameter (ε_c for cloning, ε_d for diversity)
        lambda_alg: Weight for velocity contribution in distance metric
        exclude_self: Whether to exclude self-pairing (default True for alive walkers)

    Returns:
        Companion indices for all alive walkers, shape [N]
        Dead walkers map to -1 (invalid, should use uniform selection instead)

    Note:
        This function only handles alive walkers. Dead walkers should use
        select_companions_uniform() as per the hybrid cloning operator.
    """
    N = x.shape[0]

    # Compute full distance matrix [N, N]
    dist_sq = compute_algorithmic_distance_matrix(x, v, lambda_alg)

    # Compute softmax weights: exp(-d^2 / (2*epsilon^2))
    weights = torch.exp(-dist_sq / (2 * epsilon**2))  # [N, N]

    # Create mask for valid companions
    # Each row i can select from alive walkers, excluding self if requested
    alive_mask_expanded = alive_mask.unsqueeze(0).expand(N, -1)  # [N, N]
    valid_mask = alive_mask_expanded.clone()

    if exclude_self:
        # Exclude self-pairing: set diagonal to False
        valid_mask.fill_diagonal_(fill_value=False)

    # Zero out invalid companions
    weights *= valid_mask.float()

    # Normalize weights to probabilities (vectorized)
    # For each row, divide by row sum
    row_sums = weights.sum(dim=1, keepdim=True)  # [N, 1]

    # Handle numerical underflow: if all weights are ~0, fallback to uniform
    underflow_rows = row_sums.squeeze() < 1e-30
    if underflow_rows.any():
        # For underflow rows, use uniform distribution over valid companions
        # Match dtype of weights to avoid dtype mismatch
        valid_counts = valid_mask.to(weights.dtype).sum(dim=1, keepdim=True).clamp(min=1)
        uniform_probs = valid_mask.to(weights.dtype) / valid_counts
        weights[underflow_rows] = uniform_probs[underflow_rows]
        row_sums = weights.sum(dim=1, keepdim=True)

    # Edge case: If a walker has no valid companions (e.g., single walker with exclude_self=True),
    # allow self-selection for that walker to avoid multinomial error
    no_companion_rows = row_sums.squeeze() < 1e-30
    if no_companion_rows.any():
        # For these rows, allow self-selection
        self_mask = torch.eye(N, dtype=torch.bool, device=x.device)
        weights[no_companion_rows] = self_mask[no_companion_rows].to(weights.dtype)
        row_sums = weights.sum(dim=1, keepdim=True)

    row_sums = torch.clamp(row_sums, min=1e-10)  # Avoid division by zero
    probs = weights / row_sums  # [N, N]

    # Sample companions for all walkers at once (vectorized)
    # multinomial samples one index per row
    companions = torch.multinomial(probs, num_samples=1).squeeze(1)  # [N]

    # Mark dead walkers as -1 (invalid)
    companions[~alive_mask] = -1

    return companions


def select_companions_uniform(
    alive_mask: torch.Tensor,
) -> torch.Tensor:
    """Select companions uniformly at random (O(N) fast baseline).

    This is equivalent to distance-dependent softmax with ε → ∞ .
    Each alive walker gets a random alive companion (possibly self).

    Args:
        alive_mask: Boolean mask indicating alive walkers, shape [N]

    Returns:
        Companion indices for all walkers, shape [N]
        Each alive walker maps to random alive walker (including self)
        Dead walkers map to random alive walker

    Raises:
        ValueError: If no walkers are alive
    """
    N = alive_mask.shape[0]
    device = alive_mask.device

    alive_indices = torch.where(alive_mask)[0]
    n_alive = len(alive_indices)

    if n_alive == 0:
        msg = "No alive walkers available for companion selection"
        raise ValueError(msg)

    # Each walker gets a uniformly random alive companion
    random_positions = torch.randint(0, n_alive, (N,), device=device)
    return alive_indices[random_positions]


def random_pairing_fisher_yates(
    alive_mask: torch.Tensor,
) -> torch.Tensor:
    """Create random mutual pairs using Fisher-Yates shuffle (O(N) algorithm).

    This is an O(N) alternative to distance-dependent pairing. It's equivalent to
    softmax pairing with ε → ∞. The algorithm:
    1. Generate random permutation using PyTorch's randperm
    2. Pair consecutive elements: (perm[0], perm[1]), (perm[2], perm[3]), ...

    Args:
        alive_mask: Boolean mask indicating alive walkers, shape [N]

    Returns:
        Companion map for all walkers, shape [N]
        For paired walkers: mutual pairing c(i) = j and c(j) = i
        For unpaired walkers (dead or singleton if N_alive odd): map to self
        Values are indices into the full walker array [0, N)

    Note:
        If the number of alive walkers is odd, the last walker maps to itself.
        Dead walkers are mapped to themselves.
    """
    N = alive_mask.shape[0]
    device = alive_mask.device

    # Initialize companion map (all map to self)
    companion_map = torch.arange(N, dtype=torch.long, device=device)

    alive_indices = torch.where(alive_mask)[0]
    n_alive = len(alive_indices)

    if n_alive < 2:
        # If 0 or 1 alive walkers, no pairing needed
        return companion_map

    # Generate random permutation using PyTorch's built-in (faster than manual Fisher-Yates)
    permuted = alive_indices[torch.randperm(n_alive, device=device)]

    # Pair consecutive elements: (0,1), (2,3), (4,5), ... (vectorized)
    n_pairs = n_alive // 2
    # Extract pairs: even indices [0, 2, 4, ...] and odd indices [1, 3, 5, ...]
    even_idx = permuted[0 : 2 * n_pairs : 2]  # [n_pairs]
    odd_idx = permuted[1 : 2 * n_pairs : 2]  # [n_pairs]

    # Create mutual pairings (vectorized)
    companion_map[even_idx] = odd_idx
    companion_map[odd_idx] = even_idx

    # If n_alive is odd, the last element already maps to itself

    return companion_map


def select_companions_for_cloning(
    x: torch.Tensor,
    v: torch.Tensor,
    alive_mask: torch.Tensor,
    epsilon_c: float,
    lambda_alg: float = 0.0,
) -> torch.Tensor:
    """Select companions for all walkers using the cloning companion selection operator.

    From docs/source/03_cloning.md § 5.7.1 ({prf:ref}`def-cloning-companion-operator`):

    This is a hybrid operator:
    - Alive walkers: Use distance-dependent softmax over other alive walkers (exclude self)
    - Dead walkers: Use uniform selection over all alive walkers

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        alive_mask: Boolean mask indicating alive walkers, shape [N]
        epsilon_c: Interaction range for cloning
        lambda_alg: Weight for velocity contribution in distance metric

    Returns:
        Companion indices for each walker, shape [N]
        All values are valid indices in [0, N)

    Raises:
        ValueError: If no walkers are alive (cannot select companions)
    """
    device = x.device

    alive_indices = torch.where(alive_mask)[0]
    n_alive = len(alive_indices)

    if n_alive == 0:
        msg = "No alive walkers available for companion selection"
        raise ValueError(msg)

    # Start with softmax selection for alive walkers
    companions = select_companions_softmax(
        x=x,
        v=v,
        alive_mask=alive_mask,
        epsilon=epsilon_c,
        lambda_alg=lambda_alg,
        exclude_self=True,
    )

    # For dead walkers, use uniform selection
    dead_mask = ~alive_mask
    n_dead = dead_mask.sum().item()

    if n_dead > 0:
        # Sample uniformly from alive walkers for each dead walker
        random_positions = torch.randint(0, n_alive, (int(n_dead),), device=device)
        companions[dead_mask] = alive_indices[random_positions]

    return companions


def sequential_greedy_pairing(
    x: torch.Tensor,
    v: torch.Tensor,
    alive_mask: torch.Tensor,
    epsilon_d: float,
    lambda_alg: float = 0.0,
) -> torch.Tensor:
    """Create mutual companion pairs using sequential stochastic greedy algorithm.

    From docs/source/03_cloning.md § 5.1.2 ({prf:ref}`def-greedy-pairing-algorithm`):

    This algorithm builds a mutual pairing iteratively for diversity measurement:
    1. Initialize unpaired set U with all alive walkers
    2. While |U| > 1:
       a. Select and remove walker i from U
       b. Compute softmax weights over remaining walkers in U
       c. Sample companion c_i
       d. Remove c_i from U
       e. Create mutual pairing: c(i) = c_i and c(c_i) = i
    3. If one walker remains (odd N_alive), it maps to itself

    Args:
        x: Positions of all walkers, shape [N, d]
        v: Velocities of all walkers, shape [N, d]
        alive_mask: Boolean mask indicating alive walkers, shape [N]
        epsilon_d: Interaction range for diversity measurement
        lambda_alg: Weight for velocity contribution in distance metric

    Returns:
        Companion map for all walkers, shape [N]
        For paired walkers: mutual pairing c(i) = j and c(j) = i
        For unpaired walkers (dead or singleton): map to self (i -> i)
        Values are indices into the full walker array [0, N)

    Note:
        If the number of alive walkers is odd, one walker will be mapped to itself.
        Dead walkers are mapped to themselves.
    """
    N = x.shape[0]
    device = x.device

    # Initialize companion map (all map to self initially)
    companion_map = torch.arange(N, dtype=torch.long, device=device)

    # Get alive walker indices
    alive_indices = torch.where(alive_mask)[0]
    n_alive = len(alive_indices)

    if n_alive < 2:
        # If 0 or 1 alive walkers, no pairing needed (all map to self)
        return companion_map

    # Compute full distance matrix once (only for alive walkers)
    x_alive = x[alive_mask]  # [n_alive, d]
    v_alive = v[alive_mask]  # [n_alive, d]
    # Compute distance matrix for alive walkers: [n_alive, n_alive]
    dist_sq = compute_algorithmic_distance_matrix(x_alive, v_alive, lambda_alg)

    # Precompute softmax weights for all pairs: [n_alive, n_alive]
    weights = torch.exp(-dist_sq / (2 * epsilon_d**2))

    # Track unpaired status with boolean mask (more efficient than list)
    unpaired_mask = torch.ones(n_alive, dtype=torch.bool, device=device)

    # Iterate through pairs (sequential dependency prevents full vectorization)
    for _ in range(n_alive // 2):
        # Find unpaired walkers
        unpaired_indices = torch.where(unpaired_mask)[0]
        if len(unpaired_indices) < 2:
            break

        # Select first unpaired walker
        i_pos = int(unpaired_indices[0].item())
        i_global = int(alive_indices[i_pos].item())

        # Get weights from i to all unpaired candidates (excluding i)
        candidate_mask = unpaired_mask.clone()
        candidate_mask[i_pos] = False
        candidate_weights = weights[i_pos] * candidate_mask.float()

        # Normalize to probabilities
        total_weight = candidate_weights.sum()
        if total_weight > 0:
            probs = candidate_weights / total_weight
            # Sample companion
            j_pos = int(torch.multinomial(probs, num_samples=1).item())
            j_global = int(alive_indices[j_pos].item())

            # Create mutual pairing
            companion_map[i_global] = j_global
            companion_map[j_global] = i_global

            # Mark both as paired
            unpaired_mask[i_pos] = False
            unpaired_mask[j_pos] = False

    # If one walker remains unpaired (odd number), it already maps to itself

    return companion_map


class CompanionSelection(BaseModel):
    """Pydantic configuration for companion selection mechanisms.

    This class parameterizes all companion selection operators defined in Chapter 5
    of docs/source/03_cloning.md. It provides a unified interface for selecting
    companions based on different strategies (softmax, uniform, pairing, etc.).

    From docs/source/03_cloning.md § 5.0 ({prf:ref}`def-algorithmic-distance-metric`):
    All distance-dependent methods use the algorithmic distance metric:

    $$
    d_{alg}(i, j)^2 := ||x_i - x_j||^2 + λ_{alg} ||v_i - v_j||^2
    $$

    Attributes:
        method: Selection strategy to use. Options:
            - "softmax": Distance-dependent softmax (§ 9.3.3, {prf:ref}`def-decision-operator`)
            - "uniform": Uniform random selection (ε → ∞ limit)
            - "random_pairing": Random mutual pairing via Fisher-Yates
            - "cloning": Hybrid cloning operator (§ 5.7.1, {prf:ref}`def-cloning-companion-operator`)
            - "greedy_pairing": Sequential greedy pairing (§ 5.1.2, {prf:ref}`def-greedy-pairing-algorithm`)
        epsilon: Interaction range parameter (ε_c for cloning, ε_d for diversity).
            Used for "softmax", "cloning", and "greedy_pairing" methods.
        lambda_alg: Weight for velocity contribution in distance metric (default 0.0).
            λ_alg = 0: Position-only model
            λ_alg > 0: Phase-space aware model
            λ_alg = 1: Balanced phase-space model
        exclude_self: Whether to exclude self-pairing in softmax selection (default True).
            For alive walkers, typically True. For dead walkers, typically False.

    Example:
        >>> # Cloning companion selection
        >>> selector = CompanionSelection(method="cloning", epsilon=0.1, lambda_alg=0.0)
        >>> companions = selector(x, v, alive_mask)
        >>>
        >>> # Diversity measurement pairing
        >>> pairing = CompanionSelection(
        ...     method="greedy_pairing", epsilon=0.05, lambda_alg=0.5
        ... )
        >>> pairs = pairing(x, v, alive_mask)
        >>>
        >>> # Fast uniform baseline
        >>> uniform = CompanionSelection(method="uniform")
        >>> companions = uniform(x, v, alive_mask)

    Note:
        The `epsilon` parameter is required for distance-dependent methods
        (softmax, cloning, greedy_pairing) but ignored for uniform and random_pairing.
    """

    method: Literal["softmax", "uniform", "random_pairing", "cloning", "greedy_pairing"] = Field(
        default="cloning",
        description="Companion selection strategy",
    )

    epsilon: float = Field(
        default=0.1,
        gt=0.0,
        description="Interaction range parameter (ε_c or ε_d)",
    )

    lambda_alg: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight for velocity contribution in algorithmic distance",
    )

    exclude_self: bool = Field(
        default=True,
        description="Exclude self-pairing in softmax selection (alive walkers)",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method is a supported selection strategy."""
        valid_methods = {"softmax", "uniform", "random_pairing", "cloning", "greedy_pairing"}
        if v not in valid_methods:
            msg = f"Invalid method '{v}'. Must be one of {valid_methods}"
            raise ValueError(msg)
        return v

    def __call__(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        alive_mask: torch.Tensor,
        lambda_alg: float | None = None,
    ) -> torch.Tensor:
        """Select companions using the configured strategy.

        Args:
            x: Positions of all walkers, shape [N, d]
            v: Velocities of all walkers, shape [N, d]
            alive_mask: Boolean mask indicating alive walkers, shape [N]
            lambda_alg: Optional override for velocity weight in distance metric.
                If None, uses self.lambda_alg.

        Returns:
            Companion indices for each walker, shape [N]
            - For "softmax", "cloning": indices in [0, N), dead walkers may be -1 (softmax)
            - For "uniform": indices in [0, N), all map to alive walkers
            - For "random_pairing", "greedy_pairing": mutual pairing map, shape [N]
              Paired walkers have c(i) = j and c(j) = i
              Unpaired walkers (dead or singleton) map to self

        Raises:
            ValueError: If no walkers are alive (methods that require alive walkers)
            ValueError: If method is not recognized (should not happen after validation)

        Note:
            The return semantics differ between selection methods:
            - Selection methods (softmax, uniform, cloning): Each walker independently
              selects a companion. Multiple walkers can select the same companion.
            - Pairing methods (random_pairing, greedy_pairing): Create mutual pairs
              where c(i) = j implies c(j) = i. Each walker appears in at most one pair.
        """
        lambda_alg = lambda_alg if lambda_alg is not None else self.lambda_alg
        if self.method == "softmax":
            return select_companions_softmax(
                x=x,
                v=v,
                alive_mask=alive_mask,
                epsilon=self.epsilon,
                lambda_alg=lambda_alg,
                exclude_self=self.exclude_self,
            )
        if self.method == "uniform":
            return select_companions_uniform(alive_mask=alive_mask)
        if self.method == "random_pairing":
            return random_pairing_fisher_yates(alive_mask=alive_mask)
        if self.method == "cloning":
            return select_companions_for_cloning(
                x=x,
                v=v,
                alive_mask=alive_mask,
                epsilon_c=self.epsilon,
                lambda_alg=lambda_alg,
            )
        if self.method == "greedy_pairing":
            return sequential_greedy_pairing(
                x=x,
                v=v,
                alive_mask=alive_mask,
                epsilon_d=self.epsilon,
                lambda_alg=lambda_alg,
            )
        # This should never happen due to validator, but needed for type checking
        msg = f"Unknown companion selection method: {self.method}"
        raise ValueError(msg)

    class Config:
        """Pydantic configuration."""

        frozen = False  # Allow mutation for interactive tuning
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Forbid extra fields
