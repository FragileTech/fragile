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

from fragile.bounds import TorchBounds
from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.fitness import FitnessOperator
from fragile.core.kinetic_operator import KineticOperator


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


class EuclideanGas(BaseModel):
    """Complete parameter set for Euclidean Gas algorithm."""

    model_config = {"arbitrary_types_allowed": True}

    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Spatial dimension")
    companion_selection: CompanionSelection = Field(
        description="Companion selection strategy for cloning"
    )
    potential: object = Field(
        description=(
            "Target potential function. Must be callable: U(x: [N, d]) -> [N]. "
            "Can be an OptimBenchmark instance (which provides bounds) or any callable."
        )
    )
    kinetic_op: KineticOperator = Field(description="Langevin dynamics parameters")
    cloning: CloneOperator = Field(description="Cloning operator")
    fitness_op: FitnessOperator | None = Field(
        default=None,
        description="Fitness operator (required if using adaptive kinetics features)",
    )
    bounds: TorchBounds | None = Field(
        None,
        description=(
            "Position bounds (optional, TorchBounds only). "
            "If None and potential has a 'bounds' attribute, bounds will be auto-extracted."
        ),
    )
    device: torch.device = Field(torch.device("cpu"), description="PyTorch device (cpu/cuda)")
    dtype: str = Field("float32", description="PyTorch dtype (float32/float64)")
    freeze_best: bool = Field(
        False,
        description=(
            "Keep the highest-fitness walker untouched during cloning and kinetic steps."
        ),
    )
    enable_cloning: bool = Field(
        default=True,
        description="Enable cloning operator (fitness still computed for adaptive forces)",
    )
    enable_kinetic: bool = Field(
        default=True,
        description="Enable kinetic (Langevin dynamics) operator",
    )
    pbc: bool = Field(
        default=False,
        description=(
            "Use periodic boundary conditions. When enabled, walkers that move outside "
            "bounds are wrapped back instead of being marked as dead. Requires bounds to be set. "
            "PBC is applied before computing fitness and after kinetic updates."
        ),
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization: auto-extract bounds from potential if available."""
        # Auto-extract bounds from potential if not provided
        if self.bounds is None and hasattr(self.potential, "bounds"):
            self.bounds = self.potential.bounds

        # Validate PBC requirements
        if self.pbc and self.bounds is None:
            msg = "PBC mode requires bounds to be set"
            raise ValueError(msg)

        # Validate that potential is callable
        if not callable(self.potential):
            msg = f"potential must be callable, got {type(self.potential)}"
            raise TypeError(msg)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        return torch.float64 if self.dtype == "float64" else torch.float32

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
        N, d = self.N, self.d

        if x_init is None:
            x_init = torch.randn(N, d, device=self.device, dtype=self.torch_dtype)

        if v_init is None:
            # Initialize velocities from thermal distribution
            v_std = 1.0 / torch.sqrt(torch.tensor(self.kinetic_op.beta, dtype=self.torch_dtype))
            v_init = v_std * torch.randn(N, d, device=self.device, dtype=self.torch_dtype)

        return SwarmState(
            x_init.to(device=self.device, dtype=self.torch_dtype),
            v_init.to(device=self.device, dtype=self.torch_dtype),
        )

    def _freeze_mask(self, state: SwarmState) -> Tensor | None:
        """Compute mask for walkers that should be frozen (not updated).

        Note: freeze_best feature is currently disabled, so this returns None.
        """
        return None

    def step(
        self, state: SwarmState, return_info: bool = False
    ) -> tuple[SwarmState, SwarmState] | tuple[SwarmState, SwarmState, dict] | None:
        """
        Perform one full step: compute fitness, clone (optional), then kinetic (optional).

        Uses cloning.py functions directly to compute:
        1. Rewards from potential
        2. Fitness using compute_fitness (always computed, even if cloning disabled)
        3. Cloning using clone_walkers (if enable_cloning=True)
        4. Kinetic update (if enable_kinetic=True)

        Args:
            state: Current swarm state
            return_info: If True, return full cloning info dictionary

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic), or
            (state_after_cloning, state_after_kinetic, info) if return_info=True

        Note:
            - The info dict contains: fitness, distances, companions, rewards,
              cloning_scores, cloning_probs, will_clone, num_cloned
            - Fitness is always computed (needed for adaptive forces)
            - If enable_cloning=False, cloning is skipped and state_after_cloning = state
            - If enable_kinetic=False, kinetic is skipped and
              state_after_kinetic = state_after_cloning
        """
        # Apply PBC at start if enabled (ensures positions valid before computing fitness)
        if self.pbc and self.bounds is not None:
            state.x = self.bounds.apply_pbc_to_out_of_bounds(state.x)

        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        # Step 1: Compute rewards from potential
        rewards = -self.potential(state.x)  # [N]

        # Step 2: Determine alive status from bounds
        if self.pbc:
            # PBC mode: all walkers always alive (wrapped back into bounds)
            alive_mask = torch.ones(state.N, dtype=torch.bool, device=self.device)
        elif self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)
        else:
            alive_mask = torch.ones(state.N, dtype=torch.bool, device=self.device)

        # SAFETY: If all walkers are dead, revive all within bounds (only in non-PBC mode)
        if not self.pbc and alive_mask.sum().item() == 0:
            msg = "All walkers are dead (out of bounds); cannot proceed with step."
            raise ValueError(msg)

        # Step 3: Select companions using the companion selection strategy
        companions_distance = self.companion_selection(
            x=state.x,
            v=state.v,
            alive_mask=alive_mask,
            bounds=self.bounds,
            pbc=self.pbc,
        )

        # Step 4: Compute fitness using core.fitness
        # Always compute fitness even if cloning disabled (needed for adaptive forces)
        fitness, fitness_info = self.fitness_op(
            positions=state.x,
            velocities=state.v,
            rewards=rewards,
            alive=alive_mask,
            companions=companions_distance,
            bounds=self.bounds,
            pbc=self.pbc,
        )

        # Step 5: Execute cloning using cloning.py (if enabled)
        if self.enable_cloning:
            # Step 3: Select companions using the companion selection strategy
            companions_clone = self.companion_selection(
                x=state.x,
                v=state.v,
                alive_mask=alive_mask,
                bounds=self.bounds,
                pbc=self.pbc,
            )
            x_cloned, v_cloned, _other_cloned, clone_info = self.cloning(
                positions=state.x,
                velocities=state.v,
                fitness=fitness,
                companions=companions_clone,
                alive=alive_mask,
            )
            state_cloned = SwarmState(x_cloned, v_cloned)
        else:
            # Skip cloning, use current state
            state_cloned = state.clone()
            clone_info = {
                "cloning_scores": torch.zeros(state.N, device=self.device),
                "cloning_probs": torch.ones(state.N, device=self.device),
                "will_clone": torch.zeros(state.N, dtype=torch.bool, device=self.device),
                "num_cloned": 0,
                "companions": torch.zeros_like(companions_distance),
            }

        # # Apply freeze mask if needed
        # if freeze_mask is not None and freeze_mask.any():
        #     state_cloned.copy_from(reference_state, freeze_mask)

        # Step 5: Compute fitness derivatives if needed for adaptive kinetics
        grad_fitness = None
        hess_fitness = None

        if self.fitness_op is not None and self.enable_kinetic:
            # Compute fitness gradient if needed for adaptive force
            if self.kinetic_op.use_fitness_force:
                grad_fitness = self.fitness_op.compute_gradient(
                    state_cloned.x, state_cloned.v, rewards, alive_mask, companions_distance
                )

            # Compute fitness Hessian if needed for anisotropic diffusion
            if self.kinetic_op.use_anisotropic_diffusion:
                hess_fitness = self.fitness_op.compute_hessian(
                    state_cloned.x,
                    state_cloned.v,
                    rewards,
                    alive_mask,
                    companions_distance,
                    diagonal_only=self.kinetic_op.diagonal_diffusion,
                )

            # Step 6: Kinetic update with optional fitness derivatives (if enabled)
            state_final = self.kinetic_op.apply(state_cloned, grad_fitness, hess_fitness)
            if freeze_mask is not None and freeze_mask.any():
                state_final.copy_from(reference_state, freeze_mask)
        else:
            # Skip kinetic update, use cloned state as final
            state_final = state_cloned.clone()

        # Apply PBC after kinetic update (wraps final positions back into bounds)
        if self.pbc and self.bounds is not None:
            state_final.x = self.bounds.apply_pbc_to_out_of_bounds(state_final.x)

        if return_info:
            # Combine all computed data into info dict
            info = {
                "fitness": fitness,
                "rewards": rewards,
                "companions_distance": companions_distance,
                "companions_clone": clone_info["companions"],
                "alive_mask": alive_mask,
                **clone_info,  # Adds: cloning_scores, cloning_probs, will_clone, num_cloned
                **fitness_info,
            }
            return state_cloned, state_final, info
        return state_cloned, state_final

    def run(
        self,
        n_steps: int,
        x_init: Tensor | None = None,
        v_init: Tensor | None = None,
        record_every: int = 1,
    ):
        """
        Run Euclidean Gas for multiple steps and return complete history.

        Args:
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)
            record_every: Record every k-th step (1=all steps, 10=every 10th step).
                         Step 0 (initial) and final step are always recorded.

        Returns:
            RunHistory object with complete execution trace including:
                - States before cloning, after cloning, and after kinetic at each recorded step
                - All fitness, cloning, companion, and alive data
                - Adaptive kinetics data (gradients/Hessians) if computed
                - Timing information

        Note:
            Run stops early if all walkers die (out of bounds).
            Memory scales with n_steps/record_every, so use record_every > 1 for long runs.

        Example:
            >>> gas = EuclideanGas(N=50, d=2, ...)
            >>> history = gas.run(n_steps=1000, record_every=10)  # Record every 10 steps
            >>> print(history.summary())
            >>> history.save("run_001.pt")
        """
        import time

        from fragile.core.history import RunHistory

        # Initialize state with timing
        init_start = time.time()
        state = self.initialize_state(x_init, v_init)

        # Apply PBC to initial state if enabled
        if self.pbc and self.bounds is not None:
            state.x = self.bounds.apply_pbc_to_out_of_bounds(state.x)

        init_time = time.time() - init_start

        N, d = state.N, state.d

        # Calculate number of recorded timesteps
        # Step 0 is always recorded, then every record_every steps,
        # plus final step if not at interval
        recorded_steps = list(range(0, n_steps + 1, record_every))
        if n_steps not in recorded_steps:
            recorded_steps.append(n_steps)
        n_recorded = len(recorded_steps)

        # Preallocate storage for all recorded data
        # States [n_recorded, N, d]
        x_before_clone = torch.zeros(n_recorded, N, d, device=self.device, dtype=self.torch_dtype)
        v_before_clone = torch.zeros(n_recorded, N, d, device=self.device, dtype=self.torch_dtype)
        x_after_clone = torch.zeros(
            n_recorded - 1, N, d, device=self.device, dtype=self.torch_dtype
        )
        v_after_clone = torch.zeros(
            n_recorded - 1, N, d, device=self.device, dtype=self.torch_dtype
        )
        x_final = torch.zeros(n_recorded, N, d, device=self.device, dtype=self.torch_dtype)
        v_final = torch.zeros(n_recorded, N, d, device=self.device, dtype=self.torch_dtype)

        # Per-step scalars [n_recorded] or [n_recorded-1]
        n_alive_traj = torch.zeros(n_recorded, dtype=torch.long, device=self.device)
        num_cloned_traj = torch.zeros(n_recorded - 1, dtype=torch.long, device=self.device)
        step_times = torch.zeros(n_recorded - 1, dtype=torch.float32, device=self.device)

        # Per-walker per-step data [n_recorded-1, N]
        fitness_traj = torch.zeros(n_recorded - 1, N, device=self.device, dtype=self.torch_dtype)
        rewards_traj = torch.zeros(n_recorded - 1, N, device=self.device, dtype=self.torch_dtype)
        cloning_scores_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )
        cloning_probs_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )
        will_clone_traj = torch.zeros(n_recorded - 1, N, dtype=torch.bool, device=self.device)
        alive_mask_traj = torch.zeros(n_recorded - 1, N, dtype=torch.bool, device=self.device)
        companions_distance_traj = torch.zeros(
            n_recorded - 1, N, dtype=torch.long, device=self.device
        )
        companions_clone_traj = torch.zeros(
            n_recorded - 1, N, dtype=torch.long, device=self.device
        )

        # Fitness intermediate values [n_recorded-1, N]
        distances_traj = torch.zeros(n_recorded - 1, N, device=self.device, dtype=self.torch_dtype)
        z_rewards_traj = torch.zeros(n_recorded - 1, N, device=self.device, dtype=self.torch_dtype)
        z_distances_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )
        pos_sq_diff_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )
        vel_sq_diff_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )
        rescaled_rewards_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )
        rescaled_distances_traj = torch.zeros(
            n_recorded - 1, N, device=self.device, dtype=self.torch_dtype
        )

        # Localized statistics [n_recorded-1] (global case: rho → ∞)
        mu_rewards_traj = torch.zeros(n_recorded - 1, device=self.device, dtype=self.torch_dtype)
        sigma_rewards_traj = torch.zeros(
            n_recorded - 1, device=self.device, dtype=self.torch_dtype
        )
        mu_distances_traj = torch.zeros(n_recorded - 1, device=self.device, dtype=self.torch_dtype)
        sigma_distances_traj = torch.zeros(
            n_recorded - 1, device=self.device, dtype=self.torch_dtype
        )

        # Adaptive kinetics data (optional)
        fitness_gradients_traj = None
        fitness_hessians_diag_traj = None
        fitness_hessians_full_traj = None

        if self.fitness_op is not None:
            if self.kinetic_op.use_fitness_force:
                fitness_gradients_traj = torch.zeros(
                    n_recorded - 1, N, d, device=self.device, dtype=self.torch_dtype
                )
            if self.kinetic_op.use_anisotropic_diffusion:
                if self.kinetic_op.diagonal_diffusion:
                    fitness_hessians_diag_traj = torch.zeros(
                        n_recorded - 1, N, d, device=self.device, dtype=self.torch_dtype
                    )
                else:
                    fitness_hessians_full_traj = torch.zeros(
                        n_recorded - 1, N, d, d, device=self.device, dtype=self.torch_dtype
                    )

        # Check initial alive status
        if self.pbc:
            # PBC mode: all walkers always alive
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            n_alive = N
        elif self.bounds is not None:
            alive_mask = self.bounds.contains(state.x)
            n_alive = alive_mask.sum().item()
        else:
            alive_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            n_alive = N

        # Record initial state (t=0)
        x_before_clone[0] = state.x
        v_before_clone[0] = state.v
        x_final[0] = state.x
        v_final[0] = state.v
        n_alive_traj[0] = n_alive

        # Check if initially all dead
        if n_alive == 0:
            return RunHistory(
                N=N,
                d=d,
                n_steps=0,
                n_recorded=1,
                record_every=record_every,
                terminated_early=True,
                final_step=0,
                x_before_clone=x_before_clone[:1],
                v_before_clone=v_before_clone[:1],
                x_after_clone=torch.zeros(0, N, d, device=self.device, dtype=self.torch_dtype),
                v_after_clone=torch.zeros(0, N, d, device=self.device, dtype=self.torch_dtype),
                x_final=x_final[:1],
                v_final=v_final[:1],
                n_alive=n_alive_traj[:1],
                num_cloned=torch.zeros(0, dtype=torch.long, device=self.device),
                step_times=torch.zeros(0, dtype=torch.float32, device=self.device),
                fitness=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                rewards=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                cloning_scores=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                cloning_probs=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                will_clone=torch.zeros(0, N, dtype=torch.bool, device=self.device),
                alive_mask=torch.zeros(0, N, dtype=torch.bool, device=self.device),
                companions_distance=torch.zeros(0, N, dtype=torch.long, device=self.device),
                companions_clone=torch.zeros(0, N, dtype=torch.long, device=self.device),
                distances=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                z_rewards=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                z_distances=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                pos_squared_differences=torch.zeros(
                    0, N, device=self.device, dtype=self.torch_dtype
                ),
                vel_squared_differences=torch.zeros(
                    0, N, device=self.device, dtype=self.torch_dtype
                ),
                rescaled_rewards=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                rescaled_distances=torch.zeros(0, N, device=self.device, dtype=self.torch_dtype),
                mu_rewards=torch.zeros(0, device=self.device, dtype=self.torch_dtype),
                sigma_rewards=torch.zeros(0, device=self.device, dtype=self.torch_dtype),
                mu_distances=torch.zeros(0, device=self.device, dtype=self.torch_dtype),
                sigma_distances=torch.zeros(0, device=self.device, dtype=self.torch_dtype),
                fitness_gradients=fitness_gradients_traj,
                fitness_hessians_diag=fitness_hessians_diag_traj,
                fitness_hessians_full=fitness_hessians_full_traj,
                total_time=0.0,
                init_time=init_time,
                bounds=self.bounds,
            )

        # Run steps with timing
        terminated_early = False
        final_step = n_steps
        recorded_idx = 1  # Index in recorded arrays (0 is initial state)
        total_start = time.time()

        for t in range(1, n_steps + 1):
            step_start = time.time()

            # Check if all walkers are currently dead BEFORE stepping (skip in PBC mode)
            if not self.pbc and self.bounds is not None:
                alive_mask = self.bounds.contains(state.x)
                n_alive = alive_mask.sum().item()
                if n_alive == 0:
                    terminated_early = True
                    final_step = t - 1
                    break

            # Execute step with return_info=True to get all data
            state_cloned, state_final, info = self.step(state, return_info=True)

            # Compute adaptive kinetics derivatives if enabled
            grad_fitness = None
            hess_fitness = None
            if self.fitness_op is not None:
                if self.kinetic_op.use_fitness_force:
                    grad_fitness = self.fitness_op.compute_gradient(
                        positions=state_cloned.x,
                        velocities=state_cloned.v,
                        rewards=info["rewards"],
                        alive=info["alive_mask"],
                        companions=info["companions_distance"],
                    )
                if self.kinetic_op.use_anisotropic_diffusion:
                    hess_fitness = self.fitness_op.compute_hessian(
                        positions=state_cloned.x,
                        velocities=state_cloned.v,
                        rewards=info["rewards"],
                        alive=info["alive_mask"],
                        companions=info["companions_distance"],
                        diagonal_only=self.kinetic_op.diagonal_diffusion,
                    )

            # Determine if this step should be recorded
            should_record = t in recorded_steps

            if should_record:
                # Record states
                x_before_clone[recorded_idx] = state.x
                v_before_clone[recorded_idx] = state.v
                x_after_clone[recorded_idx - 1] = state_cloned.x
                v_after_clone[recorded_idx - 1] = state_cloned.v
                x_final[recorded_idx] = state_final.x
                v_final[recorded_idx] = state_final.v

                # Record scalars
                n_alive_traj[recorded_idx] = info["alive_mask"].sum().item()
                num_cloned_traj[recorded_idx - 1] = info["num_cloned"]
                step_times[recorded_idx - 1] = time.time() - step_start

                # Record per-walker data
                fitness_traj[recorded_idx - 1] = info["fitness"]
                rewards_traj[recorded_idx - 1] = info["rewards"]
                cloning_scores_traj[recorded_idx - 1] = info["cloning_scores"]
                cloning_probs_traj[recorded_idx - 1] = info["cloning_probs"]
                will_clone_traj[recorded_idx - 1] = info["will_clone"]
                alive_mask_traj[recorded_idx - 1] = info["alive_mask"]
                companions_distance_traj[recorded_idx - 1] = info["companions_distance"]
                companions_clone_traj[recorded_idx - 1] = info["companions_clone"]

                # Record fitness intermediate values
                distances_traj[recorded_idx - 1] = info["distances"]
                z_rewards_traj[recorded_idx - 1] = info["z_rewards"]
                z_distances_traj[recorded_idx - 1] = info["z_distances"]
                pos_sq_diff_traj[recorded_idx - 1] = info["pos_squared_differences"]
                vel_sq_diff_traj[recorded_idx - 1] = info["vel_squared_differences"]
                rescaled_rewards_traj[recorded_idx - 1] = info["rescaled_rewards"]
                rescaled_distances_traj[recorded_idx - 1] = info["rescaled_distances"]

                # Record localized statistics
                mu_rewards_traj[recorded_idx - 1] = info["mu_rewards"]
                sigma_rewards_traj[recorded_idx - 1] = info["sigma_rewards"]
                mu_distances_traj[recorded_idx - 1] = info["mu_distances"]
                sigma_distances_traj[recorded_idx - 1] = info["sigma_distances"]

                # Record adaptive kinetics data if computed
                if grad_fitness is not None:
                    fitness_gradients_traj[recorded_idx - 1] = grad_fitness
                if hess_fitness is not None:
                    if self.kinetic_op.diagonal_diffusion:
                        fitness_hessians_diag_traj[recorded_idx - 1] = hess_fitness
                    else:
                        fitness_hessians_full_traj[recorded_idx - 1] = hess_fitness

                recorded_idx += 1

            # Update state for next iteration
            state = state_final

        total_time = time.time() - total_start

        # Trim arrays to actual recorded size
        actual_recorded = recorded_idx

        return RunHistory(
            N=N,
            d=d,
            n_steps=final_step,
            n_recorded=actual_recorded,
            record_every=record_every,
            terminated_early=terminated_early,
            final_step=final_step,
            x_before_clone=x_before_clone[:actual_recorded],
            v_before_clone=v_before_clone[:actual_recorded],
            x_after_clone=x_after_clone[: actual_recorded - 1],
            v_after_clone=v_after_clone[: actual_recorded - 1],
            x_final=x_final[:actual_recorded],
            v_final=v_final[:actual_recorded],
            n_alive=n_alive_traj[:actual_recorded],
            num_cloned=num_cloned_traj[: actual_recorded - 1],
            step_times=step_times[: actual_recorded - 1],
            fitness=fitness_traj[: actual_recorded - 1],
            rewards=rewards_traj[: actual_recorded - 1],
            cloning_scores=cloning_scores_traj[: actual_recorded - 1],
            cloning_probs=cloning_probs_traj[: actual_recorded - 1],
            will_clone=will_clone_traj[: actual_recorded - 1],
            alive_mask=alive_mask_traj[: actual_recorded - 1],
            companions_distance=companions_distance_traj[: actual_recorded - 1],
            companions_clone=companions_clone_traj[: actual_recorded - 1],
            distances=distances_traj[: actual_recorded - 1],
            z_rewards=z_rewards_traj[: actual_recorded - 1],
            z_distances=z_distances_traj[: actual_recorded - 1],
            pos_squared_differences=pos_sq_diff_traj[: actual_recorded - 1],
            vel_squared_differences=vel_sq_diff_traj[: actual_recorded - 1],
            rescaled_rewards=rescaled_rewards_traj[: actual_recorded - 1],
            rescaled_distances=rescaled_distances_traj[: actual_recorded - 1],
            mu_rewards=mu_rewards_traj[: actual_recorded - 1],
            sigma_rewards=sigma_rewards_traj[: actual_recorded - 1],
            mu_distances=mu_distances_traj[: actual_recorded - 1],
            sigma_distances=sigma_distances_traj[: actual_recorded - 1],
            fitness_gradients=fitness_gradients_traj[: actual_recorded - 1]
            if fitness_gradients_traj is not None
            else None,
            fitness_hessians_diag=fitness_hessians_diag_traj[: actual_recorded - 1]
            if fitness_hessians_diag_traj is not None
            else None,
            fitness_hessians_full=fitness_hessians_full_traj[: actual_recorded - 1]
            if fitness_hessians_full_traj is not None
            else None,
            total_time=total_time,
            init_time=init_time,
            bounds=self.bounds,
        )
