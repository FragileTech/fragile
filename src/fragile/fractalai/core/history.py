"""Run history data structure for EuclideanGas execution traces.

This module provides the RunHistory class for storing complete execution traces
of EuclideanGas runs, including all intermediate states, fitness data, cloning
events, and adaptive kinetics information.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
import torch
from torch import Tensor

# Import TorchBounds directly (not in TYPE_CHECKING) for Pydantic
from fragile.fractalai.bounds import TorchBounds


class RunHistory(BaseModel):
    """Complete history of an EuclideanGas run with all intermediate states and info.

    This class stores the full execution trace of a run, including:
    - States at three points per step: before cloning, after cloning, after kinetic
    - All fitness, cloning, and companion data
    - Timing information
    - Adaptive kinetics data (gradients/Hessians) if computed

    All trajectory data has shape [n_recorded, N, ...] where n_recorded is the
    number of recorded timesteps (controlled by record_every parameter).

    Example:
        >>> gas = EuclideanGas(N=50, d=2, ...)
        >>> history = gas.run(n_steps=100, record_every=10)
        >>> print(history.summary())
        >>> history.save("run_data.pt")
        >>> loaded = RunHistory.load("run_data.pt")

    Reference: Euclidean Gas implementation in src/fragile/core/euclidean_gas.py
    """

    model_config = {"arbitrary_types_allowed": True}

    # ========================================================================
    # Metadata
    # ========================================================================
    N: int = Field(description="Number of walkers")
    d: int = Field(description="Spatial dimension")
    n_steps: int = Field(description="Total number of steps executed")
    n_recorded: int = Field(description="Number of timesteps recorded")
    record_every: int = Field(description="Recording interval (every k-th step)")
    terminated_early: bool = Field(description="Whether run stopped early due to all dead")
    final_step: int = Field(description="Last step completed (may be < n_steps)")
    recorded_steps: list[int] = Field(description="Absolute step numbers recorded")
    delta_t: float = Field(description="Algorithmic time step size")
    pbc: bool = Field(description="Whether periodic boundary conditions were used")
    params: dict[str, Any] | None = Field(default=None, description="Run parameter snapshot")
    rng_seed: int | None = Field(default=None, description="Seed used for RNG (if provided)")
    rng_state: dict[str, Any] | None = Field(
        default=None, description="Captured RNG states (optional)"
    )
    bounds: TorchBounds | None = Field(
        None, description="Position bounds used during simulation (optional)"
    )

    # ========================================================================
    # States: Before Cloning [n_recorded, N, d]
    # ========================================================================
    x_before_clone: Tensor = Field(description="Positions before cloning operator")
    v_before_clone: Tensor = Field(description="Velocities before cloning operator")
    U_before: Tensor = Field(description="Potential energy before cloning")

    # ========================================================================
    # States: After Cloning [n_recorded-1, N, d]
    # Note: No "after_clone" state at t=0 (initial state)
    # ========================================================================
    x_after_clone: Tensor = Field(description="Positions after cloning, before kinetic")
    v_after_clone: Tensor = Field(description="Velocities after cloning, before kinetic")
    U_after_clone: Tensor = Field(description="Potential energy after cloning")

    # ========================================================================
    # States: Final (After Kinetic) [n_recorded, N, d]
    # ========================================================================
    x_final: Tensor = Field(description="Final positions after kinetic update")
    v_final: Tensor = Field(description="Final velocities after kinetic update")
    U_final: Tensor = Field(description="Potential energy after kinetic update")

    # ========================================================================
    # Per-Step Scalar Data [n_recorded]
    # ========================================================================
    n_alive: Tensor = Field(description="Number of alive walkers at each recorded step")
    num_cloned: Tensor = Field(description="Number of walkers that cloned at each step")
    step_times: Tensor = Field(description="Execution time for each step (seconds)")

    # ========================================================================
    # Per-Walker Per-Step Data [n_recorded-1, N]
    # Note: No info data at t=0 (initial state has no step)
    # ========================================================================

    # Fitness channel
    fitness: Tensor = Field(description="Fitness potential values V_fit")
    rewards: Tensor = Field(description="Raw reward values (potential or reward 1-form)")

    # Cloning channel
    cloning_scores: Tensor = Field(description="Cloning scores S_i")
    cloning_probs: Tensor = Field(description="Cloning probabilities π(S_i)")
    will_clone: Tensor = Field(description="Boolean mask of walkers that cloned")

    # Alive status
    alive_mask: Tensor = Field(description="Boolean mask of alive walkers")

    # Companion indices
    companions_distance: Tensor = Field(description="Companion indices for diversity (long)")
    companions_clone: Tensor = Field(description="Companion indices for cloning (long)")
    clone_jitter: Tensor = Field(description="Cloning jitter noise (sigma_x * zeta)")
    clone_delta_x: Tensor = Field(description="Position delta from cloning operator")
    clone_delta_v: Tensor = Field(description="Velocity delta from cloning operator")

    # Fitness intermediate values
    distances: Tensor = Field(description="Algorithmic distances d_alg to companions")
    z_rewards: Tensor = Field(description="Z-scores of rewards")
    z_distances: Tensor = Field(description="Z-scores of distances")
    pos_squared_differences: Tensor = Field(description="Squared position differences ||Δx||²")
    vel_squared_differences: Tensor = Field(description="Squared velocity differences ||Δv||²")
    rescaled_rewards: Tensor = Field(description="Rescaled rewards r'_i")
    rescaled_distances: Tensor = Field(description="Rescaled distances d'_i")

    # ========================================================================
    # Per-Step Localized Statistics [n_recorded-1]
    # Note: Global statistics (rho → ∞) computed over alive walkers
    # ========================================================================
    mu_rewards: Tensor = Field(description="Mean of raw rewards μ_ρ[r|alive]")
    sigma_rewards: Tensor = Field(description="Regularized std of rewards σ'_ρ[r|alive]")
    mu_distances: Tensor = Field(description="Mean of algorithmic distances μ_ρ[d|alive]")
    sigma_distances: Tensor = Field(description="Regularized std of distances σ'_ρ[d|alive]")

    # ========================================================================
    # Adaptive Kinetics Data (Optional) [n_recorded-1, N, d] or [n_recorded-1, N, d, d]
    # ========================================================================
    fitness_gradients: Tensor | None = Field(
        default=None,
        description="Fitness gradients ∂V/∂x [n_recorded-1, N, d] if use_fitness_force=True",
    )
    fitness_hessians_diag: Tensor | None = Field(
        default=None,
        description="Diagonal Hessian ∂²V/∂x² [n_recorded-1, N, d] if diagonal_diffusion=True",
    )
    fitness_hessians_full: Tensor | None = Field(
        default=None,
        description="Full Hessian ∂²V/∂x² [n_recorded-1, N, d, d] if anisotropic but not diagonal",
    )

    # ========================================================================
    # Kinetic Operator Data [n_recorded-1, N, d] or [n_recorded-1, N, d, d]
    # ========================================================================
    force_stable: Tensor = Field(description="Stable (potential) force -∇U")
    force_adapt: Tensor = Field(description="Adaptive (fitness) force -ε_F ∇V_fit")
    force_viscous: Tensor = Field(description="Viscous coupling force")
    force_friction: Tensor = Field(description="Friction force -γ v")
    force_total: Tensor = Field(description="Total force sum")
    noise: Tensor = Field(description="Realized stochastic increment Σ_reg ∘ dW")
    sigma_reg_diag: Tensor | None = Field(
        default=None, description="Diagonal diffusion tensor Σ_reg (if diagonal or isotropic)"
    )
    sigma_reg_full: Tensor | None = Field(
        default=None, description="Full diffusion tensor Σ_reg (if anisotropic)"
    )
    riemannian_volume_weights: Tensor | None = Field(
        default=None,
        description="Riemannian volume weights from Voronoi cells [n_recorded-1, N]",
    )

    # ========================================================================
    # Neighbor Graph Data (Optional)
    # ========================================================================
    neighbor_edges: list[Tensor] | None = Field(
        default=None,
        description="Per-recorded-step neighbor edges (directed) used for viscous coupling",
    )
    voronoi_regions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-recorded-step Voronoi regions/vertices metadata (if computed)",
    )

    # ========================================================================
    # Timing Data
    # ========================================================================
    total_time: float = Field(description="Total execution time (seconds)")
    init_time: float = Field(description="Initialization time (seconds)")

    # ========================================================================
    # Methods
    # ========================================================================

    def get_step_index(self, step: int) -> int:
        """Convert absolute step number to recorded index.

        Args:
            step: Absolute step number (0 to n_steps)

        Returns:
            Index in recorded arrays

        Raises:
            ValueError: If step was not recorded
        """
        if step not in self.recorded_steps:
            msg = f"Step {step} was not recorded"
            raise ValueError(msg)
        return self.recorded_steps.index(step)

    def get_walker_trajectory(self, walker_idx: int, stage: str = "final") -> dict:
        """Extract trajectory for a single walker.

        Args:
            walker_idx: Walker index (0 to N-1)
            stage: Which state to extract ("before_clone", "after_clone", "final")

        Returns:
            Dict with x [n_recorded, d] and v [n_recorded, d]

        Raises:
            ValueError: If stage is not recognized
        """
        if stage == "before_clone":
            return {
                "x": self.x_before_clone[:, walker_idx, :],
                "v": self.v_before_clone[:, walker_idx, :],
            }
        if stage == "after_clone":
            return {
                "x": self.x_after_clone[:, walker_idx, :],
                "v": self.v_after_clone[:, walker_idx, :],
            }
        if stage == "final":
            return {
                "x": self.x_final[:, walker_idx, :],
                "v": self.v_final[:, walker_idx, :],
            }
        msg = f"Unknown stage: {stage}. Must be 'before_clone', 'after_clone', or 'final'"
        raise ValueError(msg)

    def get_clone_events(self) -> list[tuple[int, int, int]]:
        """Get list of all cloning events.

        Returns:
            List of (step, cloner_idx, companion_idx) tuples where:
            - step: Absolute step number when cloning occurred
            - cloner_idx: Index of walker that was cloned (replaced)
            - companion_idx: Index of walker that was cloned from (source)
        """
        events = []
        for t in range(self.n_recorded - 1):
            cloners = torch.where(self.will_clone[t])[0]
            for i in cloners:
                companion = self.companions_clone[t, i].item()
                step = self.recorded_steps[t + 1]
                events.append((step, i.item(), companion))
        return events

    def get_alive_walkers(self, step: int) -> Tensor:
        """Get indices of alive walkers at given step.

        Args:
            step: Step number (must be a recorded step)

        Returns:
            Tensor of walker indices [n_alive] (long)

        Raises:
            ValueError: If step was not recorded
        """
        idx = self.get_step_index(step)
        if idx == 0:
            if self.pbc:
                return torch.arange(self.N, device=self.x_before_clone.device)
            if self.bounds is not None:
                return torch.where(self.bounds.contains(self.x_before_clone[0]))[0]
            return torch.arange(self.N, device=self.x_before_clone.device)
        return torch.where(self.alive_mask[idx - 1])[0]

    def to_dict(self) -> dict:
        """Convert to dictionary for saving.

        Returns:
            Dictionary with all fields (excludes None values)
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def save(self, path: str):
        """Save history to disk using torch.save.

        Args:
            path: File path (e.g., "run_history.pt")

        Example:
            >>> history.save("experiment_001.pt")
        """
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> RunHistory:
        """Load history from disk.

        Args:
            path: File path

        Returns:
            RunHistory instance

        Example:
            >>> history = RunHistory.load("experiment_001.pt")
        """
        data = torch.load(path, weights_only=False)
        return cls(**data)

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string

        Example:
            >>> print(history.summary())
            RunHistory: 100 steps, 50 walkers, 2D
              Recorded: 11 timesteps (every 10 steps)
              Final step: 100 (terminated_early=False)
              Total cloning events: 234
              Timing: 1.234s total, 0.0123s/step
        """
        lines = [
            f"RunHistory: {self.n_steps} steps, {self.N} walkers, {self.d}D",
            f"  Recorded: {self.n_recorded} timesteps (every {self.record_every} steps)",
            f"  Final step: {self.final_step} (terminated_early={self.terminated_early})",
            f"  Total cloning events: {self.will_clone.sum().item()}",
            f"  Timing: {self.total_time:.3f}s total, {self.total_time / self.n_steps:.4f}s/step",
        ]
        if self.fitness_gradients is not None:
            lines.append("  Adaptive kinetics: gradients recorded")
        if self.fitness_hessians_diag is not None:
            lines.append("  Adaptive kinetics: Hessian diagonals recorded")
        if self.fitness_hessians_full is not None:
            lines.append("  Adaptive kinetics: Full Hessians recorded")
        return "\n".join(lines)


# Rebuild model after TorchBounds is fully defined
RunHistory.model_rebuild()
