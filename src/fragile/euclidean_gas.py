"""
Euclidean Gas: A Fragile Gas Implementation

This module implements the Euclidean Gas algorithm from clean_build/source/02_euclidean_gas.md
and clean_build/source/03_cloning.md using PyTorch for vectorization and Pydantic for
parameter management.

All tensors are vectorized with the first dimension being the number of walkers N.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, Field
from torch import Tensor

if TYPE_CHECKING:
    from fragile.bounds import Bounds, TorchBounds, NumpyBounds


class PotentialParams(BaseModel):
    """Parameters for the target potential U(x)."""

    model_config = {"arbitrary_types_allowed": True}

    def evaluate(self, x: Tensor) -> Tensor:
        """
        Evaluate potential at positions x.

        Args:
            x: Positions [N, d]

        Returns:
            Potential values [N]
        """
        raise NotImplementedError("Subclasses must implement evaluate")


class SimpleQuadraticPotential(PotentialParams):
    """Simple quadratic potential U(x) = 0.5 * ||x||^2."""

    def evaluate(self, x: Tensor) -> Tensor:
        """Evaluate U(x) = 0.5 * ||x||^2."""
        return 0.5 * torch.sum(x**2, dim=-1)


class LangevinParams(BaseModel):
    """Parameters for Langevin dynamics (kinetic operator).

    Mathematical notation:
    - gamma (γ): Friction coefficient
    - beta (β): Inverse temperature 1/(k_B T)
    - delta_t (Δt): Time step size
    """

    model_config = {"arbitrary_types_allowed": True}

    gamma: float = Field(gt=0, description="Friction coefficient (γ)")
    beta: float = Field(gt=0, description="Inverse temperature 1/(k_B T) (β)")
    delta_t: float = Field(gt=0, description="Time step size (Δt)")
    integrator: str = Field("baoab", description="Integration scheme (baoab)")

    def noise_std(self) -> float:
        """Standard deviation for BAOAB noise."""
        return (1.0 - torch.exp(torch.tensor(-2 * self.gamma * self.delta_t))).sqrt().item()


class CloningParams(BaseModel):
    """Parameters for the cloning operator.

    Mathematical notation:
    - sigma_x (σ_x): Positional jitter scale
    - lambda_alg (λ_alg): Velocity weight in algorithmic distance
    """

    model_config = {"arbitrary_types_allowed": True}

    sigma_x: float = Field(gt=0, description="Positional jitter scale (σ_x)")
    lambda_alg: float = Field(gt=0, description="Velocity weight in algorithmic distance (λ_alg)")
    alpha_restitution: float = Field(
        ge=0,
        le=1,
        description="Coefficient of restitution [0,1]: 0=fully inelastic, 1=elastic"
    )
    use_inelastic_collision: bool = Field(
        True, description="Use momentum-conserving inelastic collision for velocity reset"
    )


class EuclideanGasParams(BaseModel):
    """Complete parameter set for Euclidean Gas algorithm."""

    model_config = {"arbitrary_types_allowed": True}

    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Spatial dimension")
    potential: PotentialParams = Field(description="Target potential parameters")
    langevin: LangevinParams = Field(description="Langevin dynamics parameters")
    cloning: CloningParams = Field(description="Cloning operator parameters")
    bounds: Any = Field(None, description="Position bounds (optional, TorchBounds or NumpyBounds)")
    device: str = Field("cpu", description="PyTorch device (cpu/cuda)")
    dtype: str = Field("float32", description="PyTorch dtype (float32/float64)")
    freeze_best: bool = Field(
        False,
        description=(
            "Keep the highest-fitness walker untouched during cloning and kinetic steps."
        ),
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        return torch.float64 if self.dtype == "float64" else torch.float32


class SwarmState:
    """
    Vectorized swarm state with positions and velocities.

    All tensors have shape [N, d] where N is number of walkers, d is dimension.
    """

    def __init__(self, x: Tensor, v: Tensor):
        """
        Initialize swarm state.

        Args:
            x: Positions [N, d]
            v: Velocities [N, d]
        """
        assert x.shape == v.shape, "Position and velocity must have same shape"
        assert len(x.shape) == 2, "Expected shape [N, d]"
        self.x = x
        self.v = v

    @property
    def N(self) -> int:
        """Number of walkers."""
        return self.x.shape[0]

    @property
    def d(self) -> int:
        """Spatial dimension."""
        return self.x.shape[1]

    @property
    def device(self) -> torch.device:
        """Device."""
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type."""
        return self.x.dtype

    def clone(self) -> SwarmState:
        """Create a copy of the state."""
        return SwarmState(self.x.clone(), self.v.clone())

    def copy_from(self, other: SwarmState, mask: Tensor) -> None:
        """
        Copy positions and velocities from another state for masked walkers.

        Args:
            other: Source swarm state
            mask: Boolean tensor indicating walkers to copy
        """
        if not isinstance(other, SwarmState):
            msg = f"Expected SwarmState, got {type(other)}"
            raise TypeError(msg)
        if mask.dtype != torch.bool:
            raise ValueError("Mask must be boolean tensor")
        if mask.shape[0] != self.N:
            raise ValueError("Mask size mismatch")
        if not mask.any():
            return
        indices = torch.where(mask)[0]
        self.x[indices] = other.x[indices]
        self.v[indices] = other.v[indices]


class VectorizedOps:
    """Vectorized operations on swarm states."""

    @staticmethod
    def algorithmic_distance_squared(
        state: SwarmState, lambda_alg: float
    ) -> Tensor:
        """
        Compute pairwise algorithmic distances squared.

        d_alg(i,j)² = ||x_i - x_j||² + λ_alg ||v_i - v_j||²

        Args:
            state: Current swarm state
            lambda_alg: Velocity weight in algorithmic distance (λ_alg)

        Returns:
            Distance matrix [N, N]
        """
        # Position distances [N, N]
        dx = state.x.unsqueeze(1) - state.x.unsqueeze(0)  # [N, N, d]
        dist_x_sq = torch.sum(dx**2, dim=-1)  # [N, N]

        # Velocity distances [N, N]
        dv = state.v.unsqueeze(1) - state.v.unsqueeze(0)  # [N, N, d]
        dist_v_sq = torch.sum(dv**2, dim=-1)  # [N, N]

        return dist_x_sq + lambda_alg * dist_v_sq

    @staticmethod
    def find_companions(dist_sq: Tensor, epsilon: float) -> Tensor:
        """
        Find companions using ε-dependent spatial kernel (softmax with Gaussian weights).

        Implements the Sequential Stochastic Greedy Pairing Operator from 03_cloning.md:
        - Weights: w_ij = exp(-d²_ij / (2ε²))
        - Probability: P(choose j) = w_ij / Σ_l w_il
        - Sample from this distribution for each walker

        Edge case: Single walker returns itself as companion.

        Args:
            dist_sq: Pairwise distance matrix [N, N]
            epsilon: Interaction range parameter (σ_x for cloning, ε_d for diversity)

        Returns:
            Companion indices [N]
        """
        N = dist_sq.shape[0]

        # Edge case: single walker has no companions, returns self
        if N == 1:
            return torch.tensor([0], dtype=torch.long, device=dist_sq.device)

        # Set diagonal to infinity to exclude self
        dist_sq_masked = dist_sq.clone()
        dist_sq_masked.fill_diagonal_(float("inf"))

        # Compute Gaussian spatial weights: w_ij = exp(-d²_ij / (2ε²))
        # Use log-sum-exp trick for numerical stability when distances are large
        exponent = -dist_sq_masked / (2.0 * epsilon**2)  # [N, N]

        # Numerical stability: subtract max per row to avoid underflow
        max_exp = torch.max(exponent, dim=1, keepdim=True)[0]  # [N, 1]
        weights = torch.exp(exponent - max_exp)  # [N, N]

        # Handle case where all weights are zero (all walkers too far)
        # Fall back to uniform distribution
        row_sums = weights.sum(dim=1, keepdim=True)  # [N, 1]
        zero_rows = row_sums.squeeze(1) < 1e-100  # Boolean mask for rows with all zero weights

        if zero_rows.any():
            # For rows with all zero weights, use uniform distribution
            uniform_probs = torch.ones_like(weights) / (N - 1)
            uniform_probs.fill_diagonal_(0.0)  # Still exclude self
            weights = torch.where(zero_rows.unsqueeze(1), uniform_probs, weights)
            row_sums = weights.sum(dim=1, keepdim=True)

        # Convert to probabilities (normalize)
        # Add epsilon to prevent division by zero
        probabilities = weights / (row_sums + 1e-12)  # [N, N]

        # Clamp to [0, 1] to handle floating point errors
        probabilities = torch.clamp(probabilities, min=0.0, max=1.0)

        # Renormalize to ensure valid probability distribution
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

        # Sample companions from the probability distribution
        # Use torch.multinomial which samples from categorical distribution
        companions = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # [N]

        return companions

    @staticmethod
    def find_companions_with_bounds(dist_sq: Tensor, in_bounds: Tensor, epsilon: float) -> Tensor:
        """
        Find companions using ε-dependent spatial kernel, with boundary enforcement.

        CRITICAL: Out-of-bounds walkers can NEVER be chosen as companions by anyone.
        This effectively "kills" them by ensuring no one clones from them.
        All walkers (both in-bounds and out-of-bounds) can only choose in-bounds companions.

        Edge case: If ALL walkers are out of bounds, falls back to nearest neighbor selection.

        Args:
            dist_sq: Pairwise distance matrix [N, N]
            in_bounds: Boolean mask indicating which walkers are in bounds [N]
            epsilon: Interaction range parameter (σ_x for cloning, ε_d for diversity)

        Returns:
            Companion indices [N]
        """
        N = dist_sq.shape[0]
        n_in_bounds = in_bounds.sum().item()

        # Edge case: if ALL or almost all walkers are out of bounds, bypass spatial kernel
        # and use simple uniform distribution (stochastic nearest neighbor)
        if n_in_bounds <= 1:
            # Use the original distance matrix (without bounds masking)
            dist_sq_masked = dist_sq.clone()
            dist_sq_masked.fill_diagonal_(float("inf"))

            # Use log-softmax trick for numerical stability
            exponent = -dist_sq_masked / (2.0 * epsilon**2)
            max_exp = torch.max(exponent, dim=1, keepdim=True)[0]
            weights = torch.exp(exponent - max_exp)

            # Check for zero rows and fall back to uniform if needed
            row_sums = weights.sum(dim=1, keepdim=True)
            zero_rows = row_sums.squeeze(1) < 1e-100

            if zero_rows.any():
                uniform_probs = torch.ones_like(weights) / (N - 1)
                uniform_probs.fill_diagonal_(0.0)
                weights = torch.where(zero_rows.unsqueeze(1), uniform_probs, weights)
                row_sums = weights.sum(dim=1, keepdim=True)

            probabilities = weights / row_sums
            companions = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            return companions

        # Normal case: some walkers are in bounds
        dist_sq_masked = dist_sq.clone()

        # Set diagonal to infinity to exclude self
        dist_sq_masked.fill_diagonal_(float("inf"))

        # Set distance to ALL out-of-bounds walkers to infinity
        out_of_bounds = ~in_bounds
        col_mask = out_of_bounds.unsqueeze(0)  # [1, N]
        dist_sq_masked = torch.where(col_mask, torch.tensor(float("inf")), dist_sq_masked)

        # Compute Gaussian spatial weights with numerical stability
        exponent = -dist_sq_masked / (2.0 * epsilon**2)  # [N, N]

        # Numerical stability: subtract max per row to avoid underflow
        max_exp = torch.max(exponent, dim=1, keepdim=True)[0]  # [N, 1]
        weights = torch.exp(exponent - max_exp)  # [N, N]

        # Handle case where all weights are zero (all walkers too far)
        row_sums = weights.sum(dim=1, keepdim=True)  # [N, 1]
        zero_rows = row_sums.squeeze(1) < 1e-100  # Boolean mask for rows with all zero weights

        if zero_rows.any():
            # Use uniform distribution over in-bounds walkers only
            uniform_probs = torch.ones_like(weights) / (n_in_bounds - 1.0)
            uniform_probs.fill_diagonal_(0.0)
            # Only allow in-bounds companions
            uniform_probs = torch.where(out_of_bounds.unsqueeze(0), torch.tensor(0.0), uniform_probs)

            weights = torch.where(zero_rows.unsqueeze(1), uniform_probs, weights)
            row_sums = weights.sum(dim=1, keepdim=True)

        # Convert to probabilities (normalize)
        probabilities = weights / row_sums  # [N, N]

        # Sample companions from the probability distribution
        companions = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # [N]

        return companions

    @staticmethod
    def variance_position(state: SwarmState) -> Tensor:
        """
        Compute position variance V_Var,x = (1/N) sum_i ||x_i - μ_x||².

        Args:
            state: Current swarm state

        Returns:
            Scalar variance
        """
        μ_x = torch.mean(state.x, dim=0, keepdim=True)  # [1, d]
        return torch.mean(torch.sum((state.x - μ_x) ** 2, dim=-1))

    @staticmethod
    def variance_velocity(state: SwarmState) -> Tensor:
        """
        Compute velocity variance V_Var,v = (1/N) sum_i ||v_i - μ_v||².

        Args:
            state: Current swarm state

        Returns:
            Scalar variance
        """
        μ_v = torch.mean(state.v, dim=0, keepdim=True)  # [1, d]
        return torch.mean(torch.sum((state.v - μ_v) ** 2, dim=-1))


def random_isotropic_rotation(v: Tensor) -> Tensor:
    """
    Apply a random orthogonal rotation to vectors, preserving their magnitudes.

    Uses Householder reflection for efficient isotropic rotation on the (d-1)-sphere.
    This ensures the rotated vector has the same magnitude but uniformly random direction.

    Args:
        v: Input vectors [N, d]

    Returns:
        Rotated vectors [N, d] with same magnitudes as input
    """
    N, d = v.shape
    device, dtype = v.device, v.dtype

    # Generate random unit vectors for Householder reflection
    # For each vector, we reflect it through a random hyperplane
    random_normal = torch.randn(N, d, device=device, dtype=dtype)
    random_unit = random_normal / torch.norm(random_normal, dim=1, keepdim=True)

    # Householder reflection: R(v) = v - 2 * (v · u) * u
    # where u is the unit normal to the reflection plane
    dot_product = torch.sum(v * random_unit, dim=1, keepdim=True)
    v_reflected = v - 2 * dot_product * random_unit

    return v_reflected


class CloningOperator:
    """Cloning operator from Section 4.3 of 02_euclidean_gas.md."""

    def __init__(
        self,
        params: CloningParams,
        device: torch.device,
        dtype: torch.dtype,
        bounds: Bounds | None = None
    ):
        """
        Initialize cloning operator.

        Args:
            params: Cloning parameters
            device: PyTorch device
            dtype: PyTorch dtype
            bounds: Optional position bounds for boundary enforcement
        """
        self.params = params
        self.device = device
        self.dtype = dtype
        self.bounds = bounds

    def apply(self, state: SwarmState) -> SwarmState:
        """
        Apply cloning operator with boundary enforcement.

        For each walker i:
        1. Check if walker is out of bounds (if bounds specified)
        2. Find companion c_clone(i) using d_alg
           - Out-of-bounds walkers are EXCLUDED from being chosen as companions
           - This effectively "kills" them (no one clones from them)
        3. Clone position: x̃_i = x_c + σ_x * ζ_i^x where ζ_i^x ~ N(0, I)
        4. Reset velocity using inelastic collision model

        Args:
            state: Current swarm state

        Returns:
            Updated state after cloning
        """
        N, d = state.N, state.d

        # Check which walkers are out of bounds
        if self.bounds is not None:
            in_bounds = self.bounds.contains(state.x)  # [N]
        else:
            in_bounds = torch.ones(N, dtype=torch.bool, device=self.device)

        # Find companions using algorithmic distance
        dist_sq = VectorizedOps.algorithmic_distance_squared(
            state, self.params.lambda_alg
        )

        # For out-of-bounds walkers, only allow cloning from in-bounds companions
        # Use sigma_x as the interaction range for cloning
        if self.bounds is not None and not in_bounds.all():
            companions = VectorizedOps.find_companions_with_bounds(
                dist_sq, in_bounds, epsilon=self.params.sigma_x
            )
        else:
            companions = VectorizedOps.find_companions(dist_sq, epsilon=self.params.sigma_x)  # [N]

        # Clone positions with jitter
        zeta_x = torch.randn(N, d, device=self.device, dtype=self.dtype)  # [N, d]
        x_companion = state.x[companions]  # [N, d]
        x_new = x_companion + self.params.sigma_x * zeta_x  # [N, d]

        # Reset velocities
        if self.params.use_inelastic_collision:
            v_new = self._inelastic_collision_velocity(state, companions)
        else:
            # Simple velocity reset: just copy companion velocity
            v_new = state.v[companions]

        return SwarmState(x_new, v_new)

    def _inelastic_collision_velocity(
        self, state: SwarmState, companions: Tensor
    ) -> Tensor:
        """
        Compute velocities after multi-body inelastic collision.

        Implements Definition 5.7.4 from 03_cloning.md:
        - Groups walkers by their companion
        - Conserves momentum within each group
        - Applies restitution coefficient to relative velocities
        - Adds random rotation for isotropy

        Physics:
        - alpha_restitution = 0: fully inelastic (all velocities → V_COM)
        - alpha_restitution = 1: perfectly elastic (magnitudes preserved)

        Args:
            state: Current swarm state
            companions: Companion indices [N]

        Returns:
            New velocities [N, d] after collision
        """
        N, d = state.N, state.d
        v_new = state.v.clone()  # Start with original velocities
        alpha = self.params.alpha_restitution

        # For each unique companion, process its collision group
        unique_companions = torch.unique(companions)

        for c_idx in unique_companions:
            # Find all walkers cloning to this companion
            cloners_mask = companions == c_idx  # [N]
            cloner_indices = torch.where(cloners_mask)[0]  # [M]

            # Build collision group: companion + cloners (excluding companion from cloners)
            # This prevents double-counting when a walker clones to itself
            cloner_indices_no_companion = cloner_indices[cloner_indices != c_idx]

            # Collision group: [companion, cloner_1, ..., cloner_M]
            group_indices = torch.cat([c_idx.unsqueeze(0), cloner_indices_no_companion])
            group_velocities = state.v[group_indices]  # [M+1, d] where M is number of OTHER cloners

            # Step 1: Compute center-of-mass velocity (conserved quantity)
            V_COM = torch.mean(group_velocities, dim=0)  # [d]

            # Step 2: Compute relative velocities in COM frame
            u_relative = group_velocities - V_COM.unsqueeze(0)  # [M+1, d]

            # Step 3: Apply restitution (scale relative velocities)
            # Note: We apply restitution without individual rotations to preserve momentum
            # The stochasticity comes from the random companion selection process
            u_new = alpha * u_relative  # [M+1, d]

            # Step 4: Transform back to lab frame
            v_group_new = V_COM.unsqueeze(0) + u_new  # [M+1, d]

            # Step 5: Assign new velocities to all members of collision group
            # No duplicates since we excluded self-cloning
            v_new[group_indices] = v_group_new

        return v_new


class KineticOperator:
    """Kinetic operator using BAOAB integrator for Langevin dynamics."""

    def __init__(
        self,
        params: LangevinParams,
        potential: PotentialParams,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize kinetic operator.

        Args:
            params: Langevin parameters
            potential: Target potential
            device: PyTorch device
            dtype: PyTorch dtype
        """
        self.params = params
        self.potential = potential
        self.device = device
        self.dtype = dtype

        # Precompute BAOAB constants
        self.dt = params.delta_t
        self.gamma = params.gamma
        self.beta = params.beta

        # O-step coefficients
        self.c1 = torch.exp(torch.tensor(-self.gamma * self.dt, dtype=dtype))
        self.c2 = torch.sqrt(
            (1.0 - self.c1**2) / self.beta
        )  # Noise amplitude

    def apply(self, state: SwarmState) -> SwarmState:
        """
        Apply BAOAB integrator for one time step.

        B: v → v - (Δt/2) * ∇U(x)
        A: x → x + (Δt/2) * v
        O: v → c1 * v + c2 * ξ, where ξ ~ N(0,I)
        A: x → x + (Δt/2) * v
        B: v → v - (Δt/2) * ∇U(x)

        Args:
            state: Current swarm state

        Returns:
            Updated state after integration
        """
        x, v = state.x.clone(), state.v.clone()
        N, d = state.N, state.d

        # First B step
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(
            U.sum(), x, create_graph=False
        )[0]  # [N, d]
        v = v - (self.dt / 2) * grad_U
        x.requires_grad_(False)

        # First A step
        x = x + (self.dt / 2) * v

        # O step (Ornstein-Uhlenbeck)
        ξ = torch.randn(N, d, device=self.device, dtype=self.dtype)
        v = self.c1 * v + self.c2 * ξ

        # Second A step
        x = x + (self.dt / 2) * v

        # Second B step
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(
            U.sum(), x, create_graph=False
        )[0]  # [N, d]
        v = v - (self.dt / 2) * grad_U
        x.requires_grad_(False)

        return SwarmState(x, v)


class EuclideanGas:
    """
    Euclidean Gas algorithm implementation.

    Alternates between cloning and kinetic operators.
    """

    def __init__(self, params: EuclideanGasParams):
        """
        Initialize Euclidean Gas.

        Args:
            params: Complete parameter set
        """
        self.params = params
        self.device = torch.device(params.device)
        self.dtype = params.torch_dtype

        # Initialize operators
        self.cloning_op = CloningOperator(
            params.cloning, self.device, self.dtype, params.bounds
        )
        self.kinetic_op = KineticOperator(
            params.langevin, params.potential, self.device, self.dtype
        )

    def get_cumulative_rewards(self, state: SwarmState) -> Tensor | None:
        """Return cumulative rewards associated with the swarm, if available."""
        _ = state
        return None

    def _freeze_mask(self, state: SwarmState) -> Tensor | None:
        if not self.params.freeze_best:
            return None
        rewards = self.get_cumulative_rewards(state)
        if rewards is None:
            raise RuntimeError(
                "freeze_best is enabled but get_cumulative_rewards returned None. "
                "Override get_cumulative_rewards in subclasses to supply cumulative values."
            )
        if rewards.numel() == 0:
            return None
        if not isinstance(rewards, Tensor):
            rewards = torch.as_tensor(rewards, device=state.device)
        else:
            rewards = rewards.to(state.device)
        best_idx = torch.argmax(rewards)
        mask = torch.zeros(state.N, device=state.device, dtype=torch.bool)
        mask[best_idx] = True
        return mask

    def initialize_state(self, x_init: Tensor | None = None, v_init: Tensor | None = None) -> SwarmState:
        """
        Initialize swarm state.

        Args:
            x_init: Initial positions [N, d] (optional, defaults to N(0, I))
            v_init: Initial velocities [N, d] (optional, defaults to N(0, I/β))

        Returns:
            Initial swarm state
        """
        N, d = self.params.N, self.params.d

        if x_init is None:
            x_init = torch.randn(N, d, device=self.device, dtype=self.dtype)

        if v_init is None:
            # Initialize velocities from thermal distribution
            v_std = 1.0 / torch.sqrt(torch.tensor(self.params.langevin.beta, dtype=self.dtype))
            v_init = v_std * torch.randn(N, d, device=self.device, dtype=self.dtype)

        return SwarmState(
            x_init.to(device=self.device, dtype=self.dtype),
            v_init.to(device=self.device, dtype=self.dtype),
        )

    def step(self, state: SwarmState) -> tuple[SwarmState, SwarmState]:
        """
        Perform one full step: cloning followed by kinetic.

        Args:
            state: Current swarm state

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic)
        """
        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        state_cloned = self.cloning_op.apply(state)
        if freeze_mask is not None and freeze_mask.any():
            state_cloned.copy_from(reference_state, freeze_mask)

        state_final = self.kinetic_op.apply(state_cloned)
        if freeze_mask is not None and freeze_mask.any():
            state_final.copy_from(reference_state, freeze_mask)
        return state_cloned, state_final

    def run(self, n_steps: int, x_init: Tensor | None = None, v_init: Tensor | None = None) -> dict:
        """
        Run Euclidean Gas for multiple steps.

        Args:
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)

        Returns:
            Dictionary with trajectory data:
                - 'x': positions [n_steps+1, N, d]
                - 'v': velocities [n_steps+1, N, d]
                - 'var_x': position variance [n_steps+1]
                - 'var_v': velocity variance [n_steps+1]
        """
        state = self.initialize_state(x_init, v_init)

        # Preallocate storage
        N, d = state.N, state.d
        x_traj = torch.zeros(
            n_steps + 1, N, d, device=self.device, dtype=self.dtype
        )
        v_traj = torch.zeros(
            n_steps + 1, N, d, device=self.device, dtype=self.dtype
        )
        var_x_traj = torch.zeros(n_steps + 1, device=self.device, dtype=self.dtype)
        var_v_traj = torch.zeros(n_steps + 1, device=self.device, dtype=self.dtype)

        # Store initial state
        x_traj[0] = state.x
        v_traj[0] = state.v
        var_x_traj[0] = VectorizedOps.variance_position(state)
        var_v_traj[0] = VectorizedOps.variance_velocity(state)

        # Run steps
        for t in range(n_steps):
            _, state = self.step(state)

            x_traj[t + 1] = state.x
            v_traj[t + 1] = state.v
            var_x_traj[t + 1] = VectorizedOps.variance_position(state)
            var_v_traj[t + 1] = VectorizedOps.variance_velocity(state)

        return {
            "x": x_traj,
            "v": v_traj,
            "var_x": var_x_traj,
            "var_v": var_v_traj,
        }
