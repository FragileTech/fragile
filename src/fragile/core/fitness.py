from pydantic import BaseModel, Field
import torch
from torch import Tensor


try:
    from torch.func import hessian, jacfwd, jacrev, vmap

    TORCH_FUNC_AVAILABLE = True
except ImportError:
    TORCH_FUNC_AVAILABLE = False

from fragile.core.companion_selection import CompanionSelection


class FitnessParams(BaseModel):
    """Parameters for fitness potential computation.

    Mathematical notation from Definition def-fitness-potential-operator in 03_cloning.md:
    - α (alpha): Reward channel exponent (default: 1.0)
    - β (beta): Diversity channel exponent (default: 1.0)
    - η (eta): Positivity floor parameter (default: 0.1)
    - λ_alg (lambda_alg): Velocity weight in algorithmic distance (default: 0.0)
    - σ_min (sigma_min): Regularization for patched standardization (default: 1e-8)
    - ε_dist (epsilon_dist): Regularization for distance smoothness (default: 1e-8)
    - A: Upper bound for logistic rescale (default: 2.0)

    Reference: Chapter 5, Section 5.6 in 03_cloning.md
    """

    model_config = {"arbitrary_types_allowed": True}

    alpha: float = Field(default=1.0, gt=0, description="Reward channel exponent (α)")
    beta: float = Field(default=1.0, gt=0, description="Diversity channel exponent (β)")
    eta: float = Field(default=0.1, gt=0, description="Positivity floor parameter (η)")
    lambda_alg: float = Field(
        default=0.0, ge=0, description="Velocity weight in algorithmic distance (λ_alg)"
    )
    sigma_min: float = Field(
        default=1e-8, gt=0, description="Regularization for patched standardization (σ_min)"
    )
    epsilon_dist: float = Field(
        default=1e-8,
        gt=0,
        description="Distance regularization for smoothness: d = sqrt(||Δx||² + ε²) ensures C^∞ differentiability",
    )
    A: float = Field(default=2.0, gt=0, description="Upper bound for logistic rescale")


def logistic_rescale(z: Tensor, A: float = 1.0) -> Tensor:
    """Logistic rescale function mapping R -> [0, A].

    Implements g_A(z) = A / (1 + exp(-z)), a smooth, bounded, monotone increasing
    function used in the fitness potential V_fit[f, ρ](x) = g_A(Z_ρ[f, d, x]).

    Reference: Definition def-localized-mean-field-fitness in 11_geometric_gas.md

    Args:
        z: Input tensor (typically Z-scores)
        A: Upper bound of the output range (default: 1.0)

    Returns:
        Tensor with values in [0, A]
    """
    return A / (1.0 + torch.exp(-z))


def patched_standardization(
    values: Tensor,
    alive: Tensor,
    rho: float | None = None,
    sigma_min: float = 1e-8,
    return_statistics: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute Z-scores using only alive walkers for statistics (fully differentiable).

    Implements the patched standardization Z_ρ[f, d, x] where statistics (mean, std)
    are computed only over alive walkers to prevent contamination from dead walkers.

    For the global case (rho=None), computes:
        Z[f, d, x_i] = (d(x_i) - μ[d|alive]) / σ'[d|alive]

    where μ and σ are computed using only alive walkers, and σ' includes regularization:
        σ'[d|alive] = sqrt(σ²[d|alive] + σ²_min)

    Reference: Definition def-unified-z-score in 11_geometric_gas.md

    **Differentiability**: This implementation uses masked element-wise operations instead
    of boolean indexing to preserve second-order gradients for Hessian computation.

    Args:
        values: Tensor of shape [N] containing measurement values for all walkers
        alive: Boolean tensor of shape [N], True for alive walkers
        rho: Localization scale parameter (not yet implemented for finite rho)
        sigma_min: Regularization constant ensuring σ' ≥ σ_min > 0
        return_statistics: If True, return (z_scores, mu, sigma) tuple instead of just z_scores

    Returns:
        If return_statistics=False: Z-scores tensor of shape [N]. Dead walkers receive Z-score of 0.0.
        If return_statistics=True: Tuple of (z_scores [N], mu [scalar], sigma [scalar])
            where mu is the mean over alive walkers and sigma is the regularized std.

    Note:
        Current implementation is for the global case (rho → ∞). For finite rho,
        localization kernel K_ρ(x_i, x_j) would weight contributions from nearby
        alive walkers. See def-localized-mean-field-moments in 11_geometric_gas.md.
    """
    if rho is not None:
        msg = "Localized standardization (finite rho) not yet implemented"
        raise NotImplementedError(msg)

    # Convert boolean mask to float for differentiable operations
    # This preserves gradients where boolean indexing would break them
    alive_mask = alive.float()  # [N], 1.0 for alive, 0.0 for dead

    # Count alive walkers
    n_alive = alive_mask.sum()

    # Handle edge case: no alive walkers (avoiding if statement for vmap compatibility)
    # Clamp to avoid division by zero (if all dead, we'll get NaN which will be masked later)
    n_alive_safe = torch.clamp(n_alive, min=1.0)

    # Compute masked mean over alive walkers
    # μ[alive] = Σ(values_i * mask_i) / Σ(mask_i)
    # Mathematically equivalent to values[alive].mean() but preserves gradients
    mu = (values * alive_mask).sum() / n_alive_safe

    # Compute masked variance over alive walkers
    # σ²[alive] = Σ((values_i - μ)² * mask_i) / Σ(mask_i)
    # Mathematically equivalent to values[alive].var() but preserves gradients
    centered = values - mu
    sigma_sq = ((centered**2) * alive_mask).sum() / n_alive_safe

    # Regularized standard deviation: σ'[d|alive] = sqrt(σ²[d|alive] + σ²_min)
    sigma_reg = torch.sqrt(sigma_sq + sigma_min**2)

    # Compute Z-scores for all walkers
    z_scores = centered / sigma_reg

    # Mask dead walkers (set to 0.0)
    # Using multiplication instead of torch.where to preserve gradients
    z_scores_masked = z_scores * alive_mask

    if return_statistics:
        return z_scores_masked, mu, sigma_reg
    return z_scores_masked


def compute_fitness(
    positions: Tensor,
    velocities: Tensor,
    rewards: Tensor,
    alive: Tensor,
    companions: Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 0.1,
    lambda_alg: float = 0.0,
    sigma_min: float = 1e-8,
    A: float = 2.0,
    epsilon_dist: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute fitness potential using the Euclidean Gas measurement pipeline.

    Implements the complete fitness pipeline from Definition def-fitness-potential-operator
    in 03_cloning.md:

    1. Select random alive companions for diversity measurement
    2. Compute algorithmic distances: d_alg(i,j)² = ||x_i - x_j||² + λ_alg ||v_i - v_j||²
    3. Standardize rewards using patched standardization (only alive walkers)
    4. Standardize distances using patched standardization (only alive walkers)
    5. Apply logistic rescale: g_A(z) = A / (1 + exp(-z))
    6. Add positivity floor η
    7. Combine channels: V_i = (d'_i)^β · (r'_i)^α

    Reference: Chapter 5, Section 5.6 in 03_cloning.md

    Args:
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Raw reward values [N]
        alive: Boolean mask [N], True for alive walkers
        companions: Companion indices [N] (must be provided, not selected here)
        alpha: Reward channel exponent (default: 1.0)
        beta: Diversity channel exponent (default: 1.0)
        eta: Positivity floor parameter (default: 0.1)
        lambda_alg: Algorithmic distance velocity weight (default: 0.0, position-only)
        sigma_min: Regularization for patched standardization (default: 1e-8)
        A: Upper bound for logistic rescale (default: 2.0)
        epsilon_dist: Distance regularization for C^∞ smoothness (default: 1e-8)

    Returns:
        fitness: Fitness potential vector [N], zero for dead walkers
        info: Dictionary with intermediate values for diagnostics and data tracking. Each \
            entry is a tensor of shape [N]. Keys:
            - "distances": Algorithmic distances to companions [N]
            - "companions": Companion indices [N]
            - "z_rewards": Z-scores of rewards [N]
            - "z_distances": Z-scores of distances [N]
            - "pos_squared_differences": Squared position differences ||x_i - x_j||² [N]
            - "vel_squared_differences": Squared velocity differences ||v_i - v_j||² [N]
            - "rescaled_rewards": Rescaled rewards r'_i [N]
            - "rescaled_distances": Rescaled distances d'_i [N]

    Note:
        - Dead walkers receive fitness = 0.0 and are excluded from statistics
        - Distance uses regularization: d = sqrt(||Δx||² + ε²) to ensure smoothness at origin
        - This prevents NaN gradients when walkers are self-paired (companions[i] = i)
        - For λ_alg = 0: position-only distance (spatial proximity)
        - For λ_alg > 0: phase-space distance (kinematic similarity)
        - For λ_alg = 1: balanced phase-space model
    """
    # Step 2: Compute regularized algorithmic distances in phase space
    # Regularized distance: d_alg(i,j) = sqrt(||x_i - x_j||² + λ_alg ||v_i - v_j||² + ε²)
    # The epsilon term ensures C^∞ differentiability at the origin (prevents NaN gradients)
    pos_diff = positions - positions[companions]
    vel_diff = velocities - velocities[companions]
    pos_sq = (pos_diff**2).sum(dim=-1)
    vel_sq = (vel_diff**2).sum(dim=-1)
    distances = torch.sqrt(pos_sq + lambda_alg * vel_sq + epsilon_dist**2)

    # Step 3-4: Patched standardization for both channels (only alive walkers)
    # Get statistics for localized mean-field analysis
    z_rewards, mu_rewards, sigma_rewards = patched_standardization(
        rewards, alive, rho=None, sigma_min=sigma_min, return_statistics=True
    )
    z_distances, mu_distances, sigma_distances = patched_standardization(
        distances, alive, rho=None, sigma_min=sigma_min, return_statistics=True
    )

    # Step 5-6: Logistic rescale + positivity floor
    # r'_i = g_A(z_r,i) + η, d'_i = g_A(z_d,i) + η
    r_prime = logistic_rescale(z_rewards, A=A) + eta
    d_prime = logistic_rescale(z_distances, A=A) + eta

    # Step 7: Combine channels into fitness potential
    # V_i = (d'_i)^β · (r'_i)^α
    fitness = (d_prime**beta) * (r_prime**alpha)

    # Dead walkers receive fitness = 0.0 (they don't participate in cloning)
    fitness = torch.where(alive, fitness, torch.zeros_like(fitness))
    info = {
        "distances": distances,
        "companions": companions,
        "z_rewards": z_rewards,
        "z_distances": z_distances,
        "pos_squared_differences": pos_sq,
        "vel_squared_differences": vel_sq,
        "rescaled_rewards": r_prime,
        "rescaled_distances": d_prime,
        # Localized statistics (global case: rho → ∞)
        "mu_rewards": mu_rewards,      # μ_ρ[r|alive]
        "sigma_rewards": sigma_rewards,  # σ'_ρ[r|alive]
        "mu_distances": mu_distances,    # μ_ρ[d|alive]
        "sigma_distances": sigma_distances,  # σ'_ρ[d|alive]
    }

    return fitness, info


class FitnessOperator:
    """Fitness operator with automatic differentiation for Langevin dynamics.

    This class provides:
    1. Fitness potential V(x, v, rewards, alive, companions)
    2. First derivative ∂V/∂x for fitness-based force in Langevin dynamics
    3. Second derivative ∂²V/∂x² for state-dependent diffusion tensor

    The fitness potential is computed using the Euclidean Gas measurement pipeline
    from Definition def-fitness-potential-operator in 03_cloning.md.

    Reference: Chapter 5, Section 5.6 in 03_cloning.md

    Method Selection Guide (based on benchmarks):
    -----------------------------------------------

    **For GRADIENTS** (first derivative):
        - **Recommended**: `compute_gradient_func()` using torch.func.jacrev
        - ~2x faster than autograd loops
        - Requires PyTorch >= 2.0
        - Fallback: `compute_gradient()` for PyTorch < 2.0

    **For HESSIAN DIAGONAL** (second derivative diagonal):
        - **For small N (<50)**: `compute_hessian(..., diagonal_only=True)` (autograd loops)
        - **For large N (>50)**: `compute_hessian_func(..., diagonal_only=True)` (torch.func)
        - torch.func.hessian gives 10x speedup for N=100, but slower for N=10
        - HVP method is slower and not recommended

    **For FULL HESSIAN** (complete second derivative tensor):
        - **Always use**: `compute_hessian(..., diagonal_only=False)` (autograd loops)
        - Most memory-efficient (O(Nd²) vs O(N²d²))
        - Fastest for small N (typical use case)

    Note:
        Derivatives are computed w.r.t. positions x, treating companions as fixed.
        This gives the "instantaneous force" for the current companion assignment.
        For mean-field forces, expectation over companions would be needed (future work).
    """

    def __init__(
        self,
        params: FitnessParams | None = None,
        companion_selection: CompanionSelection | None = None,
    ):
        """Initialize fitness operator.

        Args:
            params: Fitness parameters (uses defaults if None)
            companion_selection: Companion selection strategy (uses uniform if None)
        """
        self.params = params if params is not None else FitnessParams()
        self.companion_selection = (
            companion_selection
            if companion_selection is not None
            else CompanionSelection(method="uniform")
        )

    def __call__(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute fitness potential using the Euclidean Gas measurement pipeline.

        Wraps the `compute_fitness` function with the operator's parameters.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (optional, will select if None)

        Returns:
            fitness: Fitness potential vector [N], zero for dead walkers
            info: Dictionary with intermediate values for diagnostics and data tracking. Each \
                entry is a tensor of shape [N]. Keys:
                - "distances": Algorithmic distances to companions [N]
                - "companions": Companion indices [N]
                - "z_rewards": Z-scores of rewards [N]
                - "z_distances": Z-scores of distances [N]
                - "pos_squared_differences": Squared position differences ||x_i - x_j||² [N]
                - "vel_squared_differences": Squared velocity differences ||v_i - v_j||² [N]
                - "rescaled_rewards": Rescaled rewards r'_i [N]
                - "rescaled_distances": Rescaled distances d'_i [N]
        """
        # Select companions if not provided
        if companions is None:
            companions = self.companion_selection.select_companions(
                positions, velocities, alive, self.params.lambda_alg
            )

        return compute_fitness(
            positions=positions,
            velocities=velocities,
            rewards=rewards,
            alive=alive,
            companions=companions,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
            lambda_alg=self.params.lambda_alg,
            sigma_min=self.params.sigma_min,
            A=self.params.A,
            epsilon_dist=self.params.epsilon_dist,
        )

    def compute_gradient(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor | None = None,
    ) -> Tensor:
        """Compute gradient ∂V/∂x for fitness-based force in Langevin dynamics.

        Uses automatic differentiation to compute the gradient of the fitness
        potential w.r.t. walker positions. The gradient provides the force term
        for adaptive Langevin dynamics:

            F_fit(x) = -∂V/∂x

        Args:
            positions: Walker positions [N, d] (requires_grad will be enabled)
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (optional, will select if None)

        Returns:
            Gradient tensor [N, d] where grad[i] = ∂V/∂x_i

        Note:
            The gradient is computed treating companions as fixed. This gives
            the instantaneous force for the current companion assignment.
        """
        # Enable gradient tracking on positions
        positions_grad = positions.clone().detach().requires_grad_(True)

        # Select companions if not provided
        if companions is None:
            companions = self.companion_selection.select_companions(
                positions_grad, velocities, alive, self.params.lambda_alg
            )

        # Compute fitness
        fitness, _ = compute_fitness(
            positions=positions_grad,
            velocities=velocities,
            rewards=rewards,
            alive=alive,
            companions=companions,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
            lambda_alg=self.params.lambda_alg,
            sigma_min=self.params.sigma_min,
            A=self.params.A,
            epsilon_dist=self.params.epsilon_dist,
        )

        # Compute gradient: sum fitness to get scalar, then differentiate
        fitness_sum = fitness.sum()
        (grad,) = torch.autograd.grad(
            outputs=fitness_sum,
            inputs=positions_grad,
            create_graph=False,
            retain_graph=False,
        )

        return grad

    def compute_hessian(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor | None = None,
        diagonal_only: bool = True,
    ) -> Tensor:
        """Compute Hessian ∂²V/∂x² for state-dependent diffusion tensor.

        Uses automatic differentiation to compute the Hessian (second derivative)
        of the fitness potential w.r.t. walker positions. The Hessian provides
        the state-dependent diffusion tensor for adaptive Langevin dynamics:

            D(x) = f(∂²V/∂x²)

        where f is some function (e.g., absolute value, eigenvalue decomposition).

        Args:
            positions: Walker positions [N, d] (requires_grad will be enabled)
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (optional, will select if None)
            diagonal_only: If True, return only diagonal elements [N, d].
                          If False, return full Hessian [N, d, d] (expensive!)

        Returns:
            If diagonal_only=True: Diagonal Hessian [N, d] where hess[i, j] = ∂²V_i/∂x_ij²
            If diagonal_only=False: Full Hessian [N, d, d] where hess[i, j, k] = ∂²V_i/∂x_ij∂x_ik

        Note:
            - Full Hessian computation is O(N*d²) and can be very expensive
            - Diagonal approximation is O(N*d) and sufficient for many diffusion models
            - The Hessian is computed treating companions as fixed
        """
        N, d = positions.shape

        # Enable gradient tracking on positions
        positions_grad = positions.clone().detach().requires_grad_(True)

        # Select companions if not provided
        if companions is None:
            companions = self.companion_selection.select_companions(
                positions_grad, velocities, alive, self.params.lambda_alg
            )

        if diagonal_only:
            # Compute diagonal elements efficiently
            hessian_diag = torch.zeros_like(positions_grad)

            # Compute fitness once
            fitness, _ = compute_fitness(
                positions=positions_grad,
                velocities=velocities,
                rewards=rewards,
                alive=alive,
                companions=companions,
                alpha=self.params.alpha,
                beta=self.params.beta,
                eta=self.params.eta,
                lambda_alg=self.params.lambda_alg,
                sigma_min=self.params.sigma_min,
                A=self.params.A,
            )

            # Compute gradient once (with create_graph=True for Hessian)
            fitness_sum = fitness.sum()
            (grad,) = torch.autograd.grad(
                outputs=fitness_sum,
                inputs=positions_grad,
                create_graph=True,  # Need to keep graph for second derivative
                retain_graph=True,
            )

            # Compute diagonal Hessian elements
            # Unfortunately, PyTorch doesn't have a fully vectorized diagonal Hessian
            # We loop over walkers and compute all dimensions at once for each walker
            for i in range(N):
                # For walker i, compute gradient of grad[i, :] w.r.t. positions[i, :]
                # grad[i, :] is [d] vector of ∂(Σfitness)/∂positions[i, :]
                for j in range(d):
                    # Compute ∂(grad[i, j])/∂positions[i, j]
                    (hess_i,) = torch.autograd.grad(
                        outputs=grad[i, j],
                        inputs=positions_grad,
                        create_graph=False,
                        retain_graph=True,
                    )
                    # Extract diagonal: ∂²(Σfitness)/∂positions[i, j]²
                    hessian_diag[i, j] = hess_i[i, j]

            return hessian_diag

        # Compute full Hessian (expensive!)
        hessian_full = torch.zeros(N, d, d, dtype=positions.dtype, device=positions.device)

        # Compute fitness
        fitness, _ = compute_fitness(
            positions=positions_grad,
            velocities=velocities,
            rewards=rewards,
            alive=alive,
            companions=companions,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
            lambda_alg=self.params.lambda_alg,
            sigma_min=self.params.sigma_min,
            A=self.params.A,
            epsilon_dist=self.params.epsilon_dist,
        )

        # Compute gradient
        fitness_sum = fitness.sum()
        (grad,) = torch.autograd.grad(
            outputs=fitness_sum,
            inputs=positions_grad,
            create_graph=True,
            retain_graph=True,
        )

        if TORCH_FUNC_AVAILABLE:
            # Vectorize Hessian block computation using vmap over basis tangents
            basis = torch.eye(N * d, device=positions.device, dtype=positions.dtype)
            tangents = basis.view(N * d, N, d)

            def compute_hvp(tangent: Tensor) -> Tensor:
                (hess_block,) = torch.autograd.grad(
                    outputs=grad,
                    inputs=positions_grad,
                    grad_outputs=tangent,
                    retain_graph=True,
                    create_graph=False,
                )
                return hess_block

            hvps = vmap(compute_hvp)(tangents)  # [N*d, N, d]
            hvps = hvps.view(N, d, N, d)
            hessian_full = torch.diagonal(hvps, dim1=0, dim2=2).permute(2, 0, 1)
        else:
            # Fallback: differentiate each gradient component sequentially
            for walker_idx in range(N):
                for dim_idx in range(d):
                    (hess_block,) = torch.autograd.grad(
                        outputs=grad[walker_idx, dim_idx],
                        inputs=positions_grad,
                        create_graph=False,
                        retain_graph=not (walker_idx == N - 1 and dim_idx == d - 1),
                    )
                    hessian_full[walker_idx, dim_idx, :] = hess_block[walker_idx]

        return hessian_full

    def compute_gradient_func(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor | None = None,
    ) -> Tensor:
        """Compute gradient ∂V/∂x using torch.func.jacrev (fully vectorized, no loops).

        This method uses torch.func.jacrev to compute the full Jacobian of the fitness
        vector w.r.t. all positions, then extracts the diagonal blocks to get each
        walker's gradient w.r.t. its own position.

        This is fully vectorized and typically faster than the autograd loop version
        for moderate numbers of walkers.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (optional, will select if None)

        Returns:
            Gradient tensor [N, d] where grad[i] = ∂fitness_i/∂position_i

        Raises:
            RuntimeError: If torch.func is not available (PyTorch < 2.0)

        Note:
            - Computes full Jacobian [N, N, d] then extracts diagonal [N, d]
            - Memory usage: O(N²d) for Jacobian
            - Suitable for N < ~1000 walkers
            - For very large N, use compute_gradient() which uses O(Nd) memory
        """
        if not TORCH_FUNC_AVAILABLE:
            msg = "torch.func not available. Use compute_gradient() or upgrade PyTorch >= 2.0"
            raise RuntimeError(msg)

        # Define fitness function: positions -> scalar (sum of fitness)
        # This matches the compute_gradient behavior which computes ∂(Σfitness)/∂positions
        def fitness_sum_fn(pos: Tensor) -> Tensor:
            """Compute sum of fitness for all walkers given all positions."""
            fitness, _ = compute_fitness(
                positions=pos,
                velocities=velocities,
                rewards=rewards,
                alive=alive,
                companions=companions,
                alpha=self.params.alpha,
                beta=self.params.beta,
                eta=self.params.eta,
                lambda_alg=self.params.lambda_alg,
                sigma_min=self.params.sigma_min,
                A=self.params.A,
            )
            return fitness.sum()

        # Compute gradient: ∂(Σfitness)/∂positions
        # This gives us the gradient for each position component
        return jacrev(fitness_sum_fn)(positions)  # [N, d]

    def compute_hessian_func(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor | None = None,
        diagonal_only: bool = True,
    ) -> Tensor:
        """Compute Hessian ∂²V/∂x² using torch.func.hessian (fully vectorized, no loops).

        This method uses torch.func.hessian to compute the Hessian of each fitness
        component w.r.t. all positions, then extracts the diagonal blocks.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (optional, will select if None)
            diagonal_only: If True, return only diagonal elements [N, d].
                          If False, return full per-walker Hessian [N, d, d]

        Returns:
            If diagonal_only=True: Diagonal Hessian [N, d]
            If diagonal_only=False: Full per-walker Hessian [N, d, d]

        Raises:
            RuntimeError: If torch.func is not available (PyTorch < 2.0)

        Note:
            - Memory usage: O(N²d²) for full Hessian computation
            - Suitable for small to moderate N (< ~100 walkers)
            - For large N, use compute_hessian() which computes elements iteratively
        """
        if not TORCH_FUNC_AVAILABLE:
            msg = "torch.func not available. Use compute_hessian() or upgrade PyTorch >= 2.0"
            raise RuntimeError(msg)

        # Select companions if not provided
        if companions is None:
            companions = self.companion_selection.select_companions(
                positions, velocities, alive, self.params.lambda_alg
            )

        N, _d = positions.shape

        # Define fitness function: positions -> scalar (sum of fitness)
        # This matches the compute_hessian behavior which computes ∂²(Σfitness)/∂positions²
        def fitness_sum_fn(pos: Tensor) -> Tensor:
            """Compute sum of fitness for all walkers given all positions."""
            fitness, _ = compute_fitness(
                positions=pos,
                velocities=velocities,
                rewards=rewards,
                alive=alive,
                companions=companions,
                alpha=self.params.alpha,
                beta=self.params.beta,
                eta=self.params.eta,
                lambda_alg=self.params.lambda_alg,
                sigma_min=self.params.sigma_min,
                A=self.params.A,
            )
            return fitness.sum()

        # Compute full Hessian of sum of fitness w.r.t. all positions
        # Shape: [N, d, N, d] - full Hessian of scalar function
        hess_full_global = hessian(fitness_sum_fn)(positions)  # [N, d, N, d]

        # Extract diagonal blocks: hess[i, :, :] = ∂²(Σfitness)/∂position_i²
        # This gives us the Hessian block for each walker
        hess_all = []
        for i in range(N):
            hess_i_block = hess_full_global[i, :, i, :]  # [d, d]
            hess_all.append(hess_i_block)

        # Stack into [N, d, d]
        hess_full = torch.stack(hess_all)

        if diagonal_only:
            # Extract diagonal elements [N, d]
            return torch.stack([hess_full[i].diagonal() for i in range(N)])
        return hess_full

    def compute_hessian_hvp(
        self,
        positions: Tensor,
        velocities: Tensor,
        rewards: Tensor,
        alive: Tensor,
        companions: Tensor | None = None,
        diagonal_only: bool = True,
    ) -> Tensor:
        """Compute Hessian ∂²V/∂x² using HVP (Hessian-Vector Products).

        This method uses Hessian-Vector Products to compute only the diagonal blocks
        of the Hessian without materializing the full [N, d, N, d] tensor.
        More memory-efficient than compute_hessian_func, but uses loops.

        Args:
            positions: Walker positions [N, d]
            velocities: Walker velocities [N, d]
            rewards: Raw reward values [N]
            alive: Boolean mask [N], True for alive walkers
            companions: Companion indices [N] (optional, will select if None)
            diagonal_only: If True, return only diagonal elements [N, d].
                          If False, return full per-walker Hessian [N, d, d]

        Returns:
            If diagonal_only=True: Diagonal Hessian [N, d]
            If diagonal_only=False: Full per-walker Hessian [N, d, d]

        Note:
            - Memory usage: O(Nd²) for diagonal blocks only (vs O(N²d²) for full)
            - More efficient than compute_hessian_func for large N
            - Uses loops over walkers and dimensions (not fully vectorized)
        """
        # Select companions if not provided
        if companions is None:
            companions = self.companion_selection.select_companions(
                positions, velocities, alive, self.params.lambda_alg
            )

        N, d = positions.shape

        # Define fitness sum function
        def fitness_sum(pos: Tensor) -> Tensor:
            """Compute sum of fitness for all walkers given all positions."""
            fitness, _ = compute_fitness(
                positions=pos,
                velocities=velocities,
                rewards=rewards,
                alive=alive,
                companions=companions,
                alpha=self.params.alpha,
                beta=self.params.beta,
                eta=self.params.eta,
                lambda_alg=self.params.lambda_alg,
                sigma_min=self.params.sigma_min,
                A=self.params.A,
            )
            return fitness.sum()

        # Compute all Hessian blocks using HVP
        hess_blocks = []

        for i in range(N):
            # Compute Hessian block H[i, :, i, :] using HVPs
            hess_block_i = torch.zeros((d, d), device=positions.device, dtype=positions.dtype)

            for j in range(d):
                # Create tangent vector: only perturb position[i, j]
                tangents = torch.zeros_like(positions)
                tangents[i, j] = 1.0

                # Compute HVP: (H @ e_j) where e_j is the j-th basis vector for walker i
                _, hvp_result = torch.autograd.functional.hvp(
                    fitness_sum, positions, tangents, create_graph=False
                )

                # Extract the i-th walker's response (this is column j of H[i, :, i, :])
                hess_block_i[:, j] = hvp_result[i]

            hess_blocks.append(hess_block_i)

        # Stack into [N, d, d]
        hess_blocks = torch.stack(hess_blocks)

        if diagonal_only:
            # Extract diagonal elements [N, d]
            return torch.stack([hess_blocks[i].diagonal() for i in range(N)])
        return hess_blocks
