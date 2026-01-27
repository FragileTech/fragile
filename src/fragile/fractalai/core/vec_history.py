"""Vectorized history recorder for efficient in-place recording.

This module provides the VectorizedHistoryRecorder class that pre-allocates
all storage arrays and fills them in-place during EuclideanGas execution,
eliminating dynamic allocation overhead and keeping the run() method clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.fractalai.bounds import TorchBounds
    from fragile.fractalai.core.euclidean_gas import SwarmState


class VectorizedHistoryRecorder:
    """Pre-allocated, vectorized history recorder for efficient in-place recording.

    This class pre-allocates all storage arrays at initialization and provides
    a clean API for recording states and info during simulation runs. It acts
    as a builder pattern for RunHistory construction.

    The recorder handles:
    - Pre-allocation of all tensors with shape [n_recorded, N, ...]
    - In-place recording via record_initial_state() and record_step()
    - Automatic trimming to actual_recorded size
    - Construction of final RunHistory object

    Example:
        >>> recorder = VectorizedHistoryRecorder(
        ...     N=50,
        ...     d=2,
        ...     n_recorded=11,
        ...     device="cpu",
        ...     dtype=torch.float64,
        ...     record_gradients=False,
        ...     record_hessians_diag=False,
        ... )
        >>> recorder.record_initial_state(state, n_alive=50)
        >>> for t in range(1, n_steps + 1):
        ...     # ... execute step ...
        ...     recorder.record_step(
        ...         state_before,
        ...         state_cloned,
        ...         state_final,
        ...         info,
        ...         step_time,
        ...         grad_fitness,
        ...         hess_fitness,
        ...     )
        >>> history = recorder.build(
        ...     n_steps=100,
        ...     record_every=10,
        ...     terminated_early=False,
        ...     final_step=100,
        ...     total_time=1.5,
        ...     init_time=0.1,
        ...     bounds=None,
        ... )

    Reference: Replaces inline recording logic in euclidean_gas.py:472-778
    """

    def __init__(
        self,
        N: int,
        d: int,
        n_recorded: int,
        device: torch.device,
        dtype: torch.dtype,
        record_gradients: bool = False,
        record_hessians_diag: bool = False,
        record_hessians_full: bool = False,
        record_sigma_reg_diag: bool = False,
        record_sigma_reg_full: bool = False,
    ):
        """Initialize recorder with pre-allocated arrays.

        Args:
            N: Number of walkers
            d: Spatial dimension
            n_recorded: Number of timesteps to record (including t=0)
            device: Torch device for tensor allocation
            dtype: Data type for floating-point tensors
            record_gradients: Whether to record fitness gradients
            record_hessians_diag: Whether to record diagonal Hessians
            record_hessians_full: Whether to record full Hessians
        """
        self.N = N
        self.d = d
        self.n_recorded = n_recorded
        self.device = device
        self.dtype = dtype
        self.recorded_idx = 1  # Index in recorded arrays (0 is initial state)

        # ====================================================================
        # Preallocate all storage arrays
        # ====================================================================

        # States: Before Cloning [n_recorded, N, d]
        self.x_before_clone = torch.zeros(n_recorded, N, d, device=device, dtype=dtype)
        self.v_before_clone = torch.zeros(n_recorded, N, d, device=device, dtype=dtype)
        self.U_before = torch.zeros(n_recorded, N, device=device, dtype=dtype)

        # States: After Cloning [n_recorded-1, N, d]
        self.x_after_clone = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.v_after_clone = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.U_after_clone = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)

        # States: Final (After Kinetic) [n_recorded, N, d]
        self.x_final = torch.zeros(n_recorded, N, d, device=device, dtype=dtype)
        self.v_final = torch.zeros(n_recorded, N, d, device=device, dtype=dtype)
        self.U_final = torch.zeros(n_recorded, N, device=device, dtype=dtype)

        # Per-step scalars [n_recorded] or [n_recorded-1]
        self.n_alive = torch.zeros(n_recorded, dtype=torch.long, device=device)
        self.num_cloned = torch.zeros(n_recorded - 1, dtype=torch.long, device=device)
        self.step_times = torch.zeros(n_recorded - 1, dtype=torch.float32, device=device)

        # Per-walker per-step data [n_recorded-1, N]
        self.fitness = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.rewards = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.cloning_scores = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.cloning_probs = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.will_clone = torch.zeros(n_recorded - 1, N, dtype=torch.bool, device=device)
        self.alive_mask = torch.zeros(n_recorded - 1, N, dtype=torch.bool, device=device)
        self.companions_distance = torch.zeros(n_recorded - 1, N, dtype=torch.long, device=device)
        self.companions_clone = torch.zeros(n_recorded - 1, N, dtype=torch.long, device=device)
        self.clone_jitter = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.clone_delta_x = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.clone_delta_v = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)

        # Fitness intermediate values [n_recorded-1, N]
        self.distances = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.z_rewards = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.z_distances = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.pos_squared_differences = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.vel_squared_differences = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.rescaled_rewards = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)
        self.rescaled_distances = torch.zeros(n_recorded - 1, N, device=device, dtype=dtype)

        # Localized statistics [n_recorded-1]
        self.mu_rewards = torch.zeros(n_recorded - 1, device=device, dtype=dtype)
        self.sigma_rewards = torch.zeros(n_recorded - 1, device=device, dtype=dtype)
        self.mu_distances = torch.zeros(n_recorded - 1, device=device, dtype=dtype)
        self.sigma_distances = torch.zeros(n_recorded - 1, device=device, dtype=dtype)

        # Adaptive kinetics data (optional) [n_recorded-1, N, d] or [n_recorded-1, N, d, d]
        self.fitness_gradients: Tensor | None = None
        self.fitness_hessians_diag: Tensor | None = None
        self.fitness_hessians_full: Tensor | None = None
        self.sigma_reg_diag: Tensor | None = None
        self.sigma_reg_full: Tensor | None = None

        if record_gradients:
            self.fitness_gradients = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        if record_hessians_diag:
            self.fitness_hessians_diag = torch.zeros(
                n_recorded - 1, N, d, device=device, dtype=dtype
            )
        if record_hessians_full:
            self.fitness_hessians_full = torch.zeros(
                n_recorded - 1, N, d, d, device=device, dtype=dtype
            )
        if record_sigma_reg_diag:
            self.sigma_reg_diag = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        if record_sigma_reg_full:
            self.sigma_reg_full = torch.zeros(n_recorded - 1, N, d, d, device=device, dtype=dtype)

        # Kinetic operator data
        self.force_stable = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.force_adapt = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.force_viscous = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.force_friction = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.force_total = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)
        self.noise = torch.zeros(n_recorded - 1, N, d, device=device, dtype=dtype)

    def record_initial_state(
        self,
        state: SwarmState,
        n_alive: int,
        U_before: Tensor | None = None,
        U_final: Tensor | None = None,
    ) -> None:
        """Record initial state at t=0.

        Args:
            state: Initial swarm state with positions and velocities
            n_alive: Number of alive walkers at t=0
        """
        self.x_before_clone[0] = state.x
        self.v_before_clone[0] = state.v
        self.x_final[0] = state.x
        self.v_final[0] = state.v
        if U_before is not None:
            self.U_before[0] = U_before
        if U_final is not None:
            self.U_final[0] = U_final
        self.n_alive[0] = n_alive

    def record_step(
        self,
        state_before: SwarmState,
        state_cloned: SwarmState,
        state_final: SwarmState,
        info: dict,
        step_time: float,
        grad_fitness: Tensor | None = None,
        hess_fitness: Tensor | None = None,
        is_diagonal_hessian: bool = False,
        kinetic_info: dict | None = None,
    ) -> None:
        """Record a single step in-place.

        This method records all data for one timestep, including states at three
        points (before cloning, after cloning, after kinetic), all fitness/cloning
        info, and optional adaptive kinetics data.

        Args:
            state_before: State before cloning operator
            state_cloned: State after cloning, before kinetic
            state_final: Final state after kinetic update
            info: Info dict from step() with all fitness/cloning/companion data
            step_time: Execution time for this step (seconds)
            grad_fitness: Fitness gradients [N, d] (optional)
            hess_fitness: Fitness Hessians [N, d] or [N, d, d] (optional)
            is_diagonal_hessian: If True, hess_fitness is diagonal [N, d]
        """
        alive_mask = info["alive_mask"]

        def _reduce_stat(value: Tensor | float, alive: Tensor) -> Tensor:
            if isinstance(value, Tensor):
                if value.ndim > 0:
                    if bool(alive.any()):
                        value = value[alive]
                    if value.numel() == 0:
                        return torch.zeros((), device=self.device, dtype=self.dtype)
                    return value.mean()
                return value
            return torch.tensor(value, device=self.device, dtype=self.dtype)

        idx = self.recorded_idx
        idx_minus_1 = idx - 1

        # Record states
        self.x_before_clone[idx] = state_before.x
        self.v_before_clone[idx] = state_before.v
        self.x_after_clone[idx_minus_1] = state_cloned.x
        self.v_after_clone[idx_minus_1] = state_cloned.v
        self.x_final[idx] = state_final.x
        self.v_final[idx] = state_final.v
        self.U_before[idx] = info["U_before"]
        self.U_after_clone[idx_minus_1] = info["U_after_clone"]
        self.U_final[idx] = info["U_final"]

        # Record scalars
        self.n_alive[idx] = info["alive_mask"].sum().item()
        self.num_cloned[idx_minus_1] = info["num_cloned"]
        self.step_times[idx_minus_1] = step_time

        # Record per-walker data
        self.fitness[idx_minus_1] = info["fitness"]
        self.rewards[idx_minus_1] = info["rewards"]
        self.cloning_scores[idx_minus_1] = info["cloning_scores"]
        self.cloning_probs[idx_minus_1] = info["cloning_probs"]
        self.will_clone[idx_minus_1] = info["will_clone"]
        self.alive_mask[idx_minus_1] = alive_mask
        self.companions_distance[idx_minus_1] = info["companions_distance"]
        self.companions_clone[idx_minus_1] = info["companions_clone"]
        self.clone_jitter[idx_minus_1] = info["clone_jitter"]
        self.clone_delta_x[idx_minus_1] = info["clone_delta_x"]
        self.clone_delta_v[idx_minus_1] = info["clone_delta_v"]

        # Record fitness intermediate values
        self.distances[idx_minus_1] = info["distances"]
        self.z_rewards[idx_minus_1] = info["z_rewards"]
        self.z_distances[idx_minus_1] = info["z_distances"]
        self.pos_squared_differences[idx_minus_1] = info["pos_squared_differences"]
        self.vel_squared_differences[idx_minus_1] = info["vel_squared_differences"]
        self.rescaled_rewards[idx_minus_1] = info["rescaled_rewards"]
        self.rescaled_distances[idx_minus_1] = info["rescaled_distances"]

        # Record localized statistics
        # Reduce per-walker localized stats to scalars for history storage.
        self.mu_rewards[idx_minus_1] = _reduce_stat(info["mu_rewards"], alive_mask)
        self.sigma_rewards[idx_minus_1] = _reduce_stat(info["sigma_rewards"], alive_mask)
        self.mu_distances[idx_minus_1] = _reduce_stat(info["mu_distances"], alive_mask)
        self.sigma_distances[idx_minus_1] = _reduce_stat(info["sigma_distances"], alive_mask)

        # Record adaptive kinetics data if provided
        if grad_fitness is not None and self.fitness_gradients is not None:
            self.fitness_gradients[idx_minus_1] = grad_fitness

        if hess_fitness is not None:
            if is_diagonal_hessian and self.fitness_hessians_diag is not None:
                self.fitness_hessians_diag[idx_minus_1] = hess_fitness
            elif not is_diagonal_hessian and self.fitness_hessians_full is not None:
                self.fitness_hessians_full[idx_minus_1] = hess_fitness

        if kinetic_info is not None:
            self.force_stable[idx_minus_1] = kinetic_info["force_stable"]
            self.force_adapt[idx_minus_1] = kinetic_info["force_adapt"]
            self.force_viscous[idx_minus_1] = kinetic_info["force_viscous"]
            self.force_friction[idx_minus_1] = kinetic_info["force_friction"]
            self.force_total[idx_minus_1] = kinetic_info["force_total"]
            self.noise[idx_minus_1] = kinetic_info["noise"]

            if kinetic_info.get("sigma_reg_diag") is not None and self.sigma_reg_diag is not None:
                self.sigma_reg_diag[idx_minus_1] = kinetic_info["sigma_reg_diag"]
            if kinetic_info.get("sigma_reg_full") is not None and self.sigma_reg_full is not None:
                self.sigma_reg_full[idx_minus_1] = kinetic_info["sigma_reg_full"]

        # Increment recorded index for next step
        self.recorded_idx += 1

    def build(
        self,
        n_steps: int,
        record_every: int,
        terminated_early: bool,
        final_step: int,
        total_time: float,
        init_time: float,
        bounds: TorchBounds | None = None,
        recorded_steps: list[int] | None = None,
        delta_t: float | None = None,
        pbc: bool = False,
        params: dict | None = None,
        rng_seed: int | None = None,
        rng_state: dict | None = None,
    ):
        """Construct final RunHistory with trimming to actual recorded size.

        Args:
            n_steps: Total number of steps requested
            record_every: Recording interval
            terminated_early: Whether run stopped early due to all dead
            final_step: Last step completed (may be < n_steps)
            total_time: Total execution time (seconds)
            init_time: Initialization time (seconds)
            bounds: Position bounds used during simulation (optional)

        Returns:
            RunHistory object with complete execution trace
        """
        from fragile.fractalai.core.history import RunHistory

        # Actual recorded size (may be less than n_recorded if terminated early)
        actual_recorded = self.recorded_idx

        return RunHistory(
            N=self.N,
            d=self.d,
            n_steps=final_step,
            n_recorded=actual_recorded,
            record_every=record_every,
            terminated_early=terminated_early,
            final_step=final_step,
            recorded_steps=recorded_steps or [],
            delta_t=delta_t or 0.0,
            pbc=pbc,
            params=params,
            rng_seed=rng_seed,
            rng_state=rng_state,
            x_before_clone=self.x_before_clone[:actual_recorded],
            v_before_clone=self.v_before_clone[:actual_recorded],
            U_before=self.U_before[:actual_recorded],
            x_after_clone=self.x_after_clone[: actual_recorded - 1],
            v_after_clone=self.v_after_clone[: actual_recorded - 1],
            U_after_clone=self.U_after_clone[: actual_recorded - 1],
            x_final=self.x_final[:actual_recorded],
            v_final=self.v_final[:actual_recorded],
            U_final=self.U_final[:actual_recorded],
            n_alive=self.n_alive[:actual_recorded],
            num_cloned=self.num_cloned[: actual_recorded - 1],
            step_times=self.step_times[: actual_recorded - 1],
            fitness=self.fitness[: actual_recorded - 1],
            rewards=self.rewards[: actual_recorded - 1],
            cloning_scores=self.cloning_scores[: actual_recorded - 1],
            cloning_probs=self.cloning_probs[: actual_recorded - 1],
            will_clone=self.will_clone[: actual_recorded - 1],
            alive_mask=self.alive_mask[: actual_recorded - 1],
            companions_distance=self.companions_distance[: actual_recorded - 1],
            companions_clone=self.companions_clone[: actual_recorded - 1],
            clone_jitter=self.clone_jitter[: actual_recorded - 1],
            clone_delta_x=self.clone_delta_x[: actual_recorded - 1],
            clone_delta_v=self.clone_delta_v[: actual_recorded - 1],
            distances=self.distances[: actual_recorded - 1],
            z_rewards=self.z_rewards[: actual_recorded - 1],
            z_distances=self.z_distances[: actual_recorded - 1],
            pos_squared_differences=self.pos_squared_differences[: actual_recorded - 1],
            vel_squared_differences=self.vel_squared_differences[: actual_recorded - 1],
            rescaled_rewards=self.rescaled_rewards[: actual_recorded - 1],
            rescaled_distances=self.rescaled_distances[: actual_recorded - 1],
            mu_rewards=self.mu_rewards[: actual_recorded - 1],
            sigma_rewards=self.sigma_rewards[: actual_recorded - 1],
            mu_distances=self.mu_distances[: actual_recorded - 1],
            sigma_distances=self.sigma_distances[: actual_recorded - 1],
            fitness_gradients=self.fitness_gradients[: actual_recorded - 1]
            if self.fitness_gradients is not None
            else None,
            fitness_hessians_diag=self.fitness_hessians_diag[: actual_recorded - 1]
            if self.fitness_hessians_diag is not None
            else None,
            fitness_hessians_full=self.fitness_hessians_full[: actual_recorded - 1]
            if self.fitness_hessians_full is not None
            else None,
            force_stable=self.force_stable[: actual_recorded - 1],
            force_adapt=self.force_adapt[: actual_recorded - 1],
            force_viscous=self.force_viscous[: actual_recorded - 1],
            force_friction=self.force_friction[: actual_recorded - 1],
            force_total=self.force_total[: actual_recorded - 1],
            noise=self.noise[: actual_recorded - 1],
            sigma_reg_diag=self.sigma_reg_diag[: actual_recorded - 1]
            if self.sigma_reg_diag is not None
            else None,
            sigma_reg_full=self.sigma_reg_full[: actual_recorded - 1]
            if self.sigma_reg_full is not None
            else None,
            total_time=total_time,
            init_time=init_time,
            bounds=bounds,
        )
