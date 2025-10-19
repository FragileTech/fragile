"""
Euclidean Gas: A Fragile Gas Implementation

This module implements the Euclidean Gas algorithm from clean_build/source/02_euclidean_gas.md
and clean_build/source/03_cloning.md using PyTorch for vectorization and Pydantic for
parameter management.

All tensors are vectorized with the first dimension being the number of walkers N.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
import torch
from torch import Tensor

from fragile.bounds import Bounds, TorchBounds
from fragile.companion_selection import CompanionSelection
from fragile.fractal_set import FractalSet
from fragile.kinetics import KineticOperator, LangevinParams
from fragile.operators import (
    clone_walkers,
    compute_fitness,
)
from fragile.scutoid import ScutoidTessellation


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


class CloningParams(BaseModel):
    """Parameters for the cloning operator.

    Mathematical notation:
    - sigma_x (σ_x): Positional jitter scale
    - lambda_alg (λ_alg): Velocity weight in algorithmic distance
    - p_max: Maximum cloning probability threshold
    - epsilon_clone (ε_clone): Regularization for cloning score
    """

    model_config = {"arbitrary_types_allowed": True}

    sigma_x: float = Field(gt=0, description="Positional jitter scale (σ_x)")
    lambda_alg: float = Field(ge=0, description="Velocity weight in algorithmic distance (λ_alg)")
    alpha_restitution: float = Field(
        default=0.5, ge=0, le=1, description="Coefficient of restitution [0,1]: 0=fully inelastic, 1=elastic"
    )
    p_max: float = Field(default=1.0, gt=0, le=1, description="Maximum cloning probability threshold")
    epsilon_clone: float = Field(default=0.01, gt=0, description="Regularization for cloning score (ε_clone)")
    alpha: float = Field(default=1.0, gt=0, description="Reward channel exponent in fitness")
    beta: float = Field(default=1.0, gt=0, description="Diversity channel exponent in fitness")
    eta: float = Field(default=0.1, gt=0, description="Positivity floor parameter in fitness")
    A: float = Field(default=2.0, gt=0, description="Upper bound for logistic rescale")
    sigma_min: float = Field(default=1e-8, gt=0, description="Regularization for patched standardization")
    companion_selection: CompanionSelection = Field(
        default_factory=lambda: CompanionSelection(method="uniform"),
        description="Companion selection strategy"
    )


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
    """Cloning operator using functions from operators.py."""

    def __init__(
        self,
        params: CloningParams,
        potential: PotentialParams,
        device: torch.device,
        dtype: torch.dtype,
        bounds: Bounds | None = None,
    ):
        """
        Initialize cloning operator.

        Args:
            params: Cloning parameters
            potential: Target potential for reward computation
            device: PyTorch device
            dtype: PyTorch dtype
            bounds: Optional position bounds for boundary enforcement
        """
        self.params = params
        self.potential = potential
        self.device = device
        self.dtype = dtype
        self.bounds = bounds

    def apply(
        self, state: SwarmState, return_parents: bool = False, return_info: bool = False
    ) -> SwarmState | tuple[SwarmState, Tensor] | tuple[SwarmState, Tensor, dict]:
        """
        Apply complete cloning operator using operators.py functions.

        Pipeline:
        1. Compute rewards from potential: R(x) = -U(x)
        2. Determine alive/dead status from bounds
        3. Compute fitness using compute_fitness from operators.py
        4. Execute cloning using clone_walkers from operators.py

        Args:
            state: Current swarm state
            return_parents: If True, return parent IDs (companions)
            return_info: If True, return info dictionary from clone_walkers

        Returns:
            Updated state after cloning, or tuple with companions/info if requested
        """
        N, d = state.N, state.d

        # Step 1: Compute rewards from potential: R(x) = -U(x)
        rewards = -self.potential.evaluate(state.x)  # [N]

        # Step 2: Determine alive status from bounds
        if self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)  # [N]
        else:
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # SAFETY: If all walkers are dead, revive all within bounds
        n_alive = alive_mask.sum().item()
        if n_alive == 0:
            if self.bounds is not None:
                # Sample uniform positions within bounds
                low = self.bounds.low.to(device=self.device, dtype=self.dtype)
                high = self.bounds.high.to(device=self.device, dtype=self.dtype)
                x_new = low + (high - low) * torch.rand(N, d, device=self.device, dtype=self.dtype)
                # Reset velocities to small random values
                v_new = torch.randn(N, d, device=self.device, dtype=self.dtype) * 0.1
                new_state = SwarmState(x_new, v_new)

                if return_info:
                    # Return empty companions and info for consistency
                    companions = torch.arange(N, device=self.device)
                    info = {'cloning_scores': torch.zeros(N, device=self.device),
                            'cloning_probs': torch.ones(N, device=self.device),
                            'will_clone': torch.ones(N, dtype=torch.bool, device=self.device),
                            'companions': companions}
                    return new_state, companions, info
                elif return_parents:
                    companions = torch.arange(N, device=self.device)
                    return new_state, companions
                return new_state
            # No bounds - treat all as alive
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # Step 3: Compute fitness using operators.py
        fitness, distances, companions = compute_fitness(
            positions=state.x,
            velocities=state.v,
            rewards=rewards,
            alive=alive_mask,
            companion_selection=self.params.companion_selection,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
            lambda_alg=self.params.lambda_alg,
            sigma_min=self.params.sigma_min,
            A=self.params.A,
        )

        # Step 4: Execute cloning using operators.py
        x_new, v_new, alive_new, clone_info = clone_walkers(
            positions=state.x,
            velocities=state.v,
            fitness=fitness,
            companions=companions,
            alive=alive_mask,
            p_max=self.params.p_max,
            epsilon_clone=self.params.epsilon_clone,
            sigma_x=self.params.sigma_x,
            alpha_restitution=self.params.alpha_restitution,
        )

        new_state = SwarmState(x_new, v_new)

        # Return based on what was requested
        if return_info:
            # Combine fitness info with cloning info
            # Put clone_info last so it includes num_cloned, will_clone, etc.
            full_info = {
                'fitness': fitness,
                'distances': distances,
                'rewards': rewards,
                **clone_info,  # Includes: cloning_scores, cloning_probs, will_clone, num_cloned, companions
            }
            return new_state, companions, full_info
        elif return_parents:
            return new_state, companions
        return new_state


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
        self.kinetic_op = KineticOperator(
            params.langevin, params.potential, self.device, self.dtype
        )
        self.cloning_op = CloningOperator(
            params.cloning, params.potential, self.device, self.dtype, self.bounds
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
        self, state: SwarmState, return_info: bool = False
    ) -> tuple[SwarmState, SwarmState] | tuple[SwarmState, SwarmState, dict]:
        """
        Perform one full step: compute fitness, clone, then kinetic.

        Uses operators.py functions directly to compute:
        1. Rewards from potential
        2. Fitness using compute_fitness
        3. Cloning using clone_walkers
        4. Kinetic update

        Args:
            state: Current swarm state
            return_info: If True, return full cloning info dictionary

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic), or
            (state_after_cloning, state_after_kinetic, info) if return_info=True

        Note:
            The info dict contains: fitness, distances, companions, rewards,
            cloning_scores, cloning_probs, will_clone, num_cloned
        """
        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        # Step 1: Compute rewards from potential
        rewards = -self.params.potential.evaluate(state.x)  # [N]

        # Step 2: Determine alive status from bounds
        if self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)
        else:
            alive_mask = torch.ones(state.N, dtype=torch.bool, device=self.device)

        # SAFETY: If all walkers are dead, revive all within bounds
        if alive_mask.sum().item() == 0:
            if self.bounds is not None:
                low = self.bounds.low.to(device=self.device, dtype=self.dtype)
                high = self.bounds.high.to(device=self.device, dtype=self.dtype)
                x_new = low + (high - low) * torch.rand(
                    state.N, state.d, device=self.device, dtype=self.dtype
                )
                v_new = torch.randn(state.N, state.d, device=self.device, dtype=self.dtype) * 0.1
                state_cloned = SwarmState(x_new, v_new)
                state_final = self.kinetic_op.apply(state_cloned)

                if return_info:
                    # Return minimal info for resurrection case
                    info = {
                        'fitness': torch.zeros(state.N, device=self.device),
                        'rewards': rewards,
                        'distances': torch.zeros(state.N, device=self.device),
                        'companions': torch.arange(state.N, device=self.device),
                        'cloning_scores': torch.zeros(state.N, device=self.device),
                        'cloning_probs': torch.ones(state.N, device=self.device),
                        'will_clone': torch.ones(state.N, dtype=torch.bool, device=self.device),
                        'num_cloned': state.N,
                    }
                    return state_cloned, state_final, info
                return state_cloned, state_final
            alive_mask = torch.ones(state.N, dtype=torch.bool, device=self.device)

        # Step 3: Compute fitness using operators.py
        fitness, distances, companions = compute_fitness(
            positions=state.x,
            velocities=state.v,
            rewards=rewards,
            alive=alive_mask,
            companion_selection=self.params.cloning.companion_selection,
            alpha=self.params.cloning.alpha,
            beta=self.params.cloning.beta,
            eta=self.params.cloning.eta,
            lambda_alg=self.params.cloning.lambda_alg,
            sigma_min=self.params.cloning.sigma_min,
            A=self.params.cloning.A,
        )

        # Step 4: Execute cloning using operators.py
        x_cloned, v_cloned, _, clone_info = clone_walkers(
            positions=state.x,
            velocities=state.v,
            fitness=fitness,
            companions=companions,
            alive=alive_mask,
            p_max=self.params.cloning.p_max,
            epsilon_clone=self.params.cloning.epsilon_clone,
            sigma_x=self.params.cloning.sigma_x,
            alpha_restitution=self.params.cloning.alpha_restitution,
        )

        state_cloned = SwarmState(x_cloned, v_cloned)

        # Apply freeze mask if needed
        if freeze_mask is not None and freeze_mask.any():
            state_cloned.copy_from(reference_state, freeze_mask)

        # Step 5: Kinetic update
        state_final = self.kinetic_op.apply(state_cloned)
        if freeze_mask is not None and freeze_mask.any():
            state_final.copy_from(reference_state, freeze_mask)

        if return_info:
            # Combine all computed data into info dict
            info = {
                'fitness': fitness,
                'rewards': rewards,
                'distances': distances,
                'companions': companions,
                **clone_info,  # Adds: cloning_scores, cloning_probs, will_clone, num_cloned
            }
            return state_cloned, state_final, info
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
            # For initial state, we need to compute info if record_fitness is True
            if record_fitness:
                # Compute rewards and fitness for initial state
                rewards = -self.params.potential.evaluate(state.x)
                fitness, distances, companions = compute_fitness(
                    positions=state.x,
                    velocities=state.v,
                    rewards=rewards,
                    alive=alive_mask,
                    companion_selection=self.params.cloning.companion_selection,
                    alpha=self.params.cloning.alpha,
                    beta=self.params.cloning.beta,
                    eta=self.params.cloning.eta,
                    lambda_alg=self.params.cloning.lambda_alg,
                    sigma_min=self.params.cloning.sigma_min,
                    A=self.params.cloning.A,
                )
                initial_info = {
                    'fitness': fitness,
                    'rewards': rewards,
                    'distances': distances,
                    'companions': companions,
                }
            else:
                initial_info = None

            self._record_fractal_set_timestep(
                fractal_set=fractal_set,
                state=state,
                timestep=0,
                alive_mask=alive_mask,
                record_fitness=record_fitness,
                info=initial_info,
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

            # Perform step with return_info if we need to record fitness or fractal set
            need_info = fractal_set is not None and record_fitness
            if need_info:
                _, state, info = self.step(state, return_info=True)
            else:
                _, state = self.step(state, return_info=False)
                info = None

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
                    info=info,  # Pass pre-computed info
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
        info: dict | None = None,
    ) -> None:
        """
        Record current timestep data into FractalSet using pre-computed info.

        Args:
            fractal_set: FractalSet instance to record into
            state: Current swarm state
            timestep: Current timestep index
            alive_mask: Boolean mask of alive walkers [N]
            record_fitness: Whether to record fitness metrics
            info: Pre-computed info dict from step() containing fitness, rewards, etc.
                  If None and record_fitness=True, will compute on the fly.
        """
        # Compute high-error mask based on positional error
        mu_x = torch.mean(state.x, dim=0, keepdim=True)
        positional_error = torch.sqrt(torch.sum((state.x - mu_x) ** 2, dim=-1))
        threshold = torch.median(positional_error)
        high_error_mask = positional_error > threshold

        # Use pre-computed fitness metrics if available
        if record_fitness:
            if info is not None:
                # Use pre-computed values from step()
                fitness = info['fitness']
                reward = info['rewards']
                companions = info['companions']
                distances = info['distances']
                potential = -reward  # Reverse the sign

                # Compute cloning scores and probabilities from fitness
                from fragile.operators import compute_cloning_probability, compute_cloning_score

                companion_fitness = fitness[companions]
                cloning_scores = compute_cloning_score(
                    fitness, companion_fitness, epsilon_clone=self.params.cloning.epsilon_clone
                )
                cloning_probs = compute_cloning_probability(
                    cloning_scores, p_max=self.params.cloning.p_max
                )
            else:
                # Fallback: compute if not provided (e.g., for initial state)
                potential = self.params.potential.evaluate(state.x)
                reward = -potential

                # Use operators.py to compute fitness properly
                fitness, distances, companions = compute_fitness(
                    positions=state.x,
                    velocities=state.v,
                    rewards=reward,
                    alive=alive_mask,
                    companion_selection=self.params.cloning.companion_selection,
                    alpha=self.params.cloning.alpha,
                    beta=self.params.cloning.beta,
                    eta=self.params.cloning.eta,
                    lambda_alg=self.params.cloning.lambda_alg,
                    sigma_min=self.params.cloning.sigma_min,
                    A=self.params.cloning.A,
                )

                # Compute cloning scores and probabilities
                from fragile.operators import compute_cloning_probability, compute_cloning_score

                companion_fitness = fitness[companions]
                cloning_scores = compute_cloning_score(
                    fitness, companion_fitness, epsilon_clone=self.params.cloning.epsilon_clone
                )
                cloning_probs = compute_cloning_probability(
                    cloning_scores, p_max=self.params.cloning.p_max
                )

            # Compute rescaled reward using patched standardization
            from fragile.operators import patched_standardization

            rescaled_reward = patched_standardization(
                reward, alive_mask, sigma_min=self.params.cloning.sigma_min
            )

            # Compute rescaled distance using patched standardization
            rescaled_distance = patched_standardization(
                distances, alive_mask, sigma_min=self.params.cloning.sigma_min
            )

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
