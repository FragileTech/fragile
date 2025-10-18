"""
Euclidean Gas: A Fragile Gas Implementation

This module implements the Euclidean Gas algorithm from clean_build/source/02_euclidean_gas.md
and clean_build/source/03_cloning.md using PyTorch for vectorization and Pydantic for
parameter management.

All tensors are vectorized with the first dimension being the number of walkers N.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.bounds import TorchBounds
from fragile.companion_selection import (
    select_companions_for_cloning,
    select_companions_softmax,
    select_companions_uniform,
)


if TYPE_CHECKING:
    from fragile.bounds import Bounds


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
        msg = "Subclasses must implement evaluate"
        raise NotImplementedError(msg)


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
    - epsilon_c (ε_c): Companion selection range

    Companion selection methods (see fragile.companion_selection):
    - "hybrid": Alive walkers use softmax (distance-dependent), dead walkers use uniform
    - "softmax": All walkers use distance-dependent softmax selection
    - "uniform": All walkers use uniform random selection
    """

    model_config = {"arbitrary_types_allowed": True}

    sigma_x: float = Field(gt=0, description="Positional jitter scale (σ_x)")
    lambda_alg: float = Field(ge=0, description="Velocity weight in algorithmic distance (λ_alg)")
    epsilon_c: float | None = Field(
        None, description="Companion selection range (ε_c). If None, defaults to sigma_x"
    )
    companion_selection_method: str = Field(
        "hybrid", description="Companion selection method: 'hybrid', 'softmax', or 'uniform'"
    )
    alpha_restitution: float = Field(
        ge=0, le=1, description="Coefficient of restitution [0,1]: 0=fully inelastic, 1=elastic"
    )
    use_inelastic_collision: bool = Field(
        True, description="Use momentum-conserving inelastic collision for velocity reset"
    )

    def get_epsilon_c(self) -> float:
        """Get effective companion selection range, defaulting to sigma_x if not specified."""
        return self.epsilon_c if self.epsilon_c is not None else self.sigma_x


class EuclideanGasParams(BaseModel):
    """Complete parameter set for Euclidean Gas algorithm."""

    model_config = {"arbitrary_types_allowed": True}

    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Spatial dimension")
    potential: PotentialParams = Field(description="Target potential parameters")
    langevin: LangevinParams = Field(description="Langevin dynamics parameters")
    cloning: CloningParams = Field(description="Cloning operator parameters")
    bounds: TorchBounds | None = Field(
        None, description="Position bounds (optional, TorchBounds only)"
    )
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
            msg = "Mask must be boolean tensor"
            raise ValueError(msg)
        if mask.shape[0] != self.N:
            msg = "Mask size mismatch"
            raise ValueError(msg)
        if not mask.any():
            return
        indices = torch.where(mask)[0]
        self.x[indices] = other.x[indices]
        self.v[indices] = other.v[indices]


class VectorizedOps:
    """Vectorized operations on swarm states.

    Note: Companion selection methods have been moved to fragile.companion_selection module.
    This class now only contains variance computation methods.
    """

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
    return v - 2 * dot_product * random_unit


class CloningOperator:
    """Cloning operator from Section 4.3 of 02_euclidean_gas.md."""

    def __init__(
        self,
        params: CloningParams,
        device: torch.device,
        dtype: torch.dtype,
        bounds: Bounds | None = None,
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

    def apply(
        self, state: SwarmState, return_parents: bool = False
    ) -> SwarmState | tuple[SwarmState, Tensor]:
        """
        Apply cloning operator with boundary enforcement.

        Uses companion selection from fragile.companion_selection module.

        For each walker i:
        1. Check if walker is out of bounds (if bounds specified)
        2. Find companion c_clone(i) using configured selection method:
           - "hybrid": Alive walkers use softmax, dead walkers use uniform
           - "softmax": All walkers use distance-dependent selection
           - "uniform": All walkers use uniform random selection
        3. Clone position: x̃_i = x_c + σ_x * ζ_i^x where ζ_i^x ~ N(0, I)
        4. Reset velocity using inelastic collision model

        Args:
            state: Current swarm state
            return_parents: If True, return (new_state, parent_ids) tuple

        Returns:
            Updated state after cloning, or (state, parent_ids) if return_parents=True
        """
        N, d = state.N, state.d

        # Create alive_mask from bounds checking
        # Out-of-bounds walkers are considered "dead" for companion selection
        if self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)  # [N]
        else:
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # SAFETY: If all walkers are dead, revive all within bounds
        # This prevents permanent extinction (cemetery state)
        n_alive = alive_mask.sum().item()
        if n_alive == 0:
            # Resurrect all walkers at random positions within bounds
            if self.bounds is not None:
                # Sample uniform positions within bounds
                low = self.bounds.low.to(device=self.device, dtype=self.dtype)
                high = self.bounds.high.to(device=self.device, dtype=self.dtype)
                x_new = low + (high - low) * torch.rand(N, d, device=self.device, dtype=self.dtype)
                # Reset velocities to small random values
                v_new = torch.randn(N, d, device=self.device, dtype=self.dtype) * 0.1
                return SwarmState(x_new, v_new)
            # No bounds defined - this shouldn't happen, but treat all as alive
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # Get effective epsilon_c (defaults to sigma_x if not specified)
        epsilon_c = self.params.get_epsilon_c()

        # Select companions using configured method
        method = self.params.companion_selection_method
        if method == "hybrid":
            # Hybrid mode: alive use softmax, dead use uniform (matches old behavior)
            companions = select_companions_for_cloning(
                state.x,
                state.v,
                alive_mask,
                epsilon_c=epsilon_c,
                lambda_alg=self.params.lambda_alg,
            )
        elif method == "softmax":
            # All walkers use distance-dependent softmax selection
            companions = select_companions_softmax(
                state.x,
                state.v,
                alive_mask,
                epsilon=epsilon_c,
                lambda_alg=self.params.lambda_alg,
                exclude_self=True,
            )
        elif method == "uniform":
            # All walkers use uniform random selection
            companions = select_companions_uniform(alive_mask)
        else:
            msg = f"Unknown companion_selection_method: {method}"
            raise ValueError(msg)

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

        new_state = SwarmState(x_new, v_new)

        if return_parents:
            return new_state, companions
        return new_state

    def _inelastic_collision_velocity(self, state: SwarmState, companions: Tensor) -> Tensor:
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
        _N, _d = state.N, state.d
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
            group_velocities = state.v[
                group_indices
            ]  # [M+1, d] where M is number of OTHER cloners

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
        self.c2 = torch.sqrt((1.0 - self.c1**2) / self.beta)  # Noise amplitude

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
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
        v -= self.dt / 2 * grad_U
        x.requires_grad_(False)

        # First A step
        x += self.dt / 2 * v

        # O step (Ornstein-Uhlenbeck)
        ξ = torch.randn(N, d, device=self.device, dtype=self.dtype)
        v = self.c1 * v + self.c2 * ξ

        # Second A step
        x += self.dt / 2 * v

        # Second B step
        x.requires_grad_(True)
        U = self.potential.evaluate(x)  # [N]
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]  # [N, d]
        v -= self.dt / 2 * grad_U
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
        self.bounds = params.bounds

        # Initialize operators
        self.cloning_op = CloningOperator(params.cloning, self.device, self.dtype, params.bounds)
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
            msg = (
                "freeze_best is enabled but get_cumulative_rewards returned None. "
                "Override get_cumulative_rewards in subclasses to supply cumulative values."
            )
            raise RuntimeError(msg)
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

    def initialize_state(
        self, x_init: Tensor | None = None, v_init: Tensor | None = None
    ) -> SwarmState:
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

    def step(
        self, state: SwarmState, return_parents: bool = False
    ) -> tuple[SwarmState, SwarmState] | tuple[SwarmState, SwarmState, Tensor]:
        """
        Perform one full step: cloning followed by kinetic.

        Args:
            state: Current swarm state
            return_parents: If True, return parent IDs from cloning

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic), or
            (state_after_cloning, state_after_kinetic, parent_ids) if return_parents=True
        """
        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        if return_parents:
            state_cloned, parent_ids = self.cloning_op.apply(state, return_parents=True)
        else:
            state_cloned = self.cloning_op.apply(state, return_parents=False)

        if freeze_mask is not None and freeze_mask.any():
            state_cloned.copy_from(reference_state, freeze_mask)

        state_final = self.kinetic_op.apply(state_cloned)
        if freeze_mask is not None and freeze_mask.any():
            state_final.copy_from(reference_state, freeze_mask)

        if return_parents:
            return state_cloned, state_final, parent_ids
        return state_cloned, state_final

    def run(
        self,
        n_steps: int,
        x_init: Tensor | None = None,
        v_init: Tensor | None = None,
        fractal_set: FractalSet | None = None,
        record_fitness: bool = False,
        scutoid_tessellation: ScutoidTessellation | None = None,
    ) -> dict:
        """
        Run Euclidean Gas for multiple steps.

        Args:
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)
            fractal_set: Optional FractalSet instance to record simulation data.
                        If provided, all timesteps will be recorded in the graph.
            record_fitness: If True and fractal_set is provided, compute and record
                           fitness, potential, and reward at each step.
            scutoid_tessellation: Optional ScutoidTessellation instance to record
                                 spacetime tessellation. If provided, Voronoi cells
                                 and scutoid cells will be computed at each step.

        Returns:
            Dictionary with trajectory data:
                - 'x': positions [n_steps+1, N, d] or [t_final+1, N, d] if stopped early
                - 'v': velocities [n_steps+1, N, d] or [t_final+1, N, d] if stopped early
                - 'var_x': position variance [n_steps+1] or [t_final+1]
                - 'var_v': velocity variance [n_steps+1] or [t_final+1]
                - 'n_alive': number of alive walkers [n_steps+1] or [t_final+1]
                - 'terminated_early': True if stopped due to all walkers dead
                - 'final_step': actual number of steps completed
                - 'fractal_set': the FractalSet instance (if provided)
                - 'scutoid_tessellation': the ScutoidTessellation instance (if provided)

        Note:
            Run stops early if all walkers become dead (out of bounds).
            If fractal_set is provided, it will be populated during the run.
            If scutoid_tessellation is provided, it will be populated with Voronoi
            and scutoid cells at each timestep.
        """
        state = self.initialize_state(x_init, v_init)

        # Preallocate storage
        N, d = state.N, state.d
        x_traj = torch.zeros(n_steps + 1, N, d, device=self.device, dtype=self.dtype)
        v_traj = torch.zeros(n_steps + 1, N, d, device=self.device, dtype=self.dtype)
        var_x_traj = torch.zeros(n_steps + 1, device=self.device, dtype=self.dtype)
        var_v_traj = torch.zeros(n_steps + 1, device=self.device, dtype=self.dtype)
        n_alive_traj = torch.zeros(n_steps + 1, dtype=torch.long, device=self.device)

        # Check initial alive status
        if self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)
            n_alive = alive_mask.sum().item()
        else:
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            n_alive = N

        # Store initial state
        x_traj[0] = state.x
        v_traj[0] = state.v
        var_x_traj[0] = VectorizedOps.variance_position(state)
        var_v_traj[0] = VectorizedOps.variance_velocity(state)
        n_alive_traj[0] = n_alive

        # Record initial state in FractalSet if provided
        if fractal_set is not None:
            self._record_fractal_set_timestep(
                fractal_set=fractal_set,
                state=state,
                timestep=0,
                alive_mask=alive_mask,
                record_fitness=record_fitness,
            )

        # Record initial state in ScutoidTessellation if provided
        if scutoid_tessellation is not None:
            scutoid_tessellation.add_timestep(
                state=state,
                timestep=0,
                t=0.0,
                parent_ids=None,  # No parents at initial timestep
            )

        # Check if initially all dead
        if n_alive == 0:
            return {
                "x": x_traj[:1],
                "v": v_traj[:1],
                "var_x": var_x_traj[:1],
                "var_v": var_v_traj[:1],
                "n_alive": n_alive_traj[:1],
                "terminated_early": True,
                "final_step": 0,
            }

        # Run steps
        terminated_early = False
        final_step = n_steps

        for t in range(n_steps):
            # Check if all walkers are currently dead BEFORE stepping
            # (step will fail if trying to clone with 0 alive walkers)
            if self.bounds is not None:
                alive_mask = self.bounds.contains(state.x)
                n_alive = alive_mask.sum().item()
                if n_alive == 0:
                    # All walkers died during previous step
                    terminated_early = True
                    final_step = t
                    break

            # Perform step (safe because we know n_alive > 0)
            # Request parent IDs if we need them for scutoid tessellation
            need_parents = scutoid_tessellation is not None and fractal_set is None
            if need_parents:
                _, state, parent_ids_tensor = self.step(state, return_parents=True)
            else:
                _, state = self.step(state, return_parents=False)
                parent_ids_tensor = None

            # Update alive count after step
            if self.bounds is not None:
                alive_mask = self.bounds.contains(state.x)
                n_alive = alive_mask.sum().item()
            else:
                alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)
                n_alive = N

            # Store new state
            x_traj[t + 1] = state.x
            v_traj[t + 1] = state.v
            var_x_traj[t + 1] = VectorizedOps.variance_position(state)
            var_v_traj[t + 1] = VectorizedOps.variance_velocity(state)
            n_alive_traj[t + 1] = n_alive

            # Record in FractalSet if provided
            if fractal_set is not None:
                self._record_fractal_set_timestep(
                    fractal_set=fractal_set,
                    state=state,
                    timestep=t + 1,
                    alive_mask=alive_mask,
                    record_fitness=record_fitness,
                )

            # Record in ScutoidTessellation if provided
            if scutoid_tessellation is not None:
                # Extract parent IDs from FractalSet if available, otherwise from cloning operator
                if fractal_set is not None:
                    # Use FractalSet as source of truth for parent tracking
                    parent_ids_np = fractal_set.get_parent_ids(t + 1)
                    parent_ids = torch.from_numpy(parent_ids_np).to(device=self.device)
                elif parent_ids_tensor is not None:
                    # Use parent IDs from cloning operator
                    parent_ids = parent_ids_tensor
                else:
                    # Fallback: assume no cloning (each walker is its own parent)
                    parent_ids = torch.arange(N, device=self.device)

                scutoid_tessellation.add_timestep(
                    state=state,
                    timestep=t + 1,
                    t=float(t + 1) * self.params.langevin.delta_t,
                    parent_ids=parent_ids,
                )

        # Return only the trajectory up to the final step
        result = {
            "x": x_traj[: final_step + 1],
            "v": v_traj[: final_step + 1],
            "var_x": var_x_traj[: final_step + 1],
            "var_v": var_v_traj[: final_step + 1],
            "n_alive": n_alive_traj[: final_step + 1],
            "terminated_early": terminated_early,
            "final_step": final_step,
        }

        # Include FractalSet in result if provided
        if fractal_set is not None:
            result["fractal_set"] = fractal_set

        # Include ScutoidTessellation in result if provided
        if scutoid_tessellation is not None:
            result["scutoid_tessellation"] = scutoid_tessellation

        return result

    def _record_fractal_set_timestep(
        self,
        fractal_set: FractalSet,
        state: SwarmState,
        timestep: int,
        alive_mask: Tensor,
        record_fitness: bool,
    ) -> None:
        """
        Record current timestep data into FractalSet.

        Args:
            fractal_set: FractalSet instance to record into
            state: Current swarm state
            timestep: Current timestep index
            alive_mask: Boolean mask of alive walkers [N]
            record_fitness: Whether to compute and record fitness metrics
        """
        from fragile.companion_selection import select_companions_softmax

        # Compute high-error mask based on positional error
        mu_x = torch.mean(state.x, dim=0, keepdim=True)
        positional_error = torch.sqrt(torch.sum((state.x - mu_x) ** 2, dim=-1))
        threshold = torch.median(positional_error)
        high_error_mask = positional_error > threshold

        # Compute fitness-related metrics if requested
        if record_fitness:
            # Potential and reward
            potential = self.params.potential.evaluate(state.x)
            reward = -potential

            # Companions and distances
            companions = select_companions_softmax(
                state.x,
                state.v,
                alive_mask,
                epsilon=self.params.cloning.get_epsilon_c(),
                lambda_alg=self.params.cloning.lambda_alg,
                exclude_self=True,
            )

            x_companion = state.x[companions]
            v_companion = state.v[companions]
            pos_diff_sq = torch.sum((state.x - x_companion) ** 2, dim=-1)
            vel_diff_sq = torch.sum((state.v - v_companion) ** 2, dim=-1)
            distances = torch.sqrt(pos_diff_sq + self.params.cloning.lambda_alg * vel_diff_sq)

            # Simplified fitness (inverse distance)
            fitness = 1.0 / (distances + 1e-6)

            # Estimate cloning probabilities (simplified)
            cloning_probs = torch.clamp(fitness / fitness.mean(), 0.0, 1.0)

            # Compute rescaled reward (normalized by mean)
            reward_mean = reward.mean()
            reward_std = reward.std() + 1e-8
            rescaled_reward = (reward - reward_mean) / reward_std

            # Compute rescaled distance (normalized by mean)
            dist_mean = distances.mean()
            dist_std = distances.std() + 1e-8
            rescaled_distance = (distances - dist_mean) / dist_std

            # Generate uniform random samples for cloning decisions
            clone_uniform_sample = torch.rand(state.N, device=self.device, dtype=self.dtype)
        else:
            potential = None
            reward = None
            companions = None
            distances = None
            fitness = None
            cloning_probs = None
            rescaled_reward = None
            rescaled_distance = None
            clone_uniform_sample = None

        # Record timestep in FractalSet
        fractal_set.add_timestep(
            state=state,
            timestep=timestep,
            high_error_mask=high_error_mask,
            alive_mask=alive_mask,
            fitness=fitness,
            potential=potential,
            reward=reward,
            companions=companions,
            cloning_probs=cloning_probs,
            distances=distances,
            rescaled_reward=rescaled_reward,
            rescaled_distance=rescaled_distance,
            clone_uniform_sample=clone_uniform_sample,
        )
