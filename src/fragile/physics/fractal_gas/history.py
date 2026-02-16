"""Vectorized history recorder for efficient in-place recording.

This module provides the VectorizedHistoryRecorder class that pre-allocates
all storage arrays and fills them in-place during EuclideanGas execution,
eliminating dynamic allocation overhead and keeping the run() method clean.

When ``chunk_size`` is set, only a small buffer is kept in memory and full
chunks are flushed to temporary ``.pt`` files on disk, then merged in
``build()`` to produce the same ``RunHistory`` as the unchunked path.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field
import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.physics.fractal_gas.euclidean_gas import SwarmState


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
    params: dict[str, Any] | None = Field(default=None, description="Run parameter snapshot")
    rng_seed: int | None = Field(default=None, description="Seed used for RNG (if provided)")
    rng_state: dict[str, Any] | None = Field(
        default=None, description="Captured RNG states (optional)"
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
    ricci_scalar_proxy: Tensor | None = Field(
        default=None,
        description="Ricci scalar proxy from scutoid data [n_recorded-1, N]",
    )
    geodesic_edge_distances: list[Tensor] | None = Field(
        default=None,
        description="Per-recorded-step geodesic edge distances aligned with neighbor_edges",
    )
    diffusion_tensors_full: Tensor | None = Field(
        default=None,
        description="Full anisotropic diffusion tensor from scutoid metric [n_recorded-1, N, d, d]",
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
    edge_weights: list[dict[str, Tensor]] | None = Field(
        default=None,
        description="Per-recorded-step edge weights dict {mode: Tensor[E]} aligned with neighbor_edges",
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

    When ``chunk_size`` is provided, only a buffer of that many timesteps is
    kept in memory.  Once full the buffer is flushed to a temporary ``.pt``
    file on disk and the buffer is reset.  ``build()`` merges all chunks into
    the same ``RunHistory`` that the unchunked path produces.

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
        ... )

    Reference: Replaces inline recording logic in euclidean_gas.py:472-778
    """

    # ------------------------------------------------------------------
    # Field category constants – used by flush/merge to avoid duplicating
    # the field list in multiple places.
    # ------------------------------------------------------------------

    # Fields indexed with the *full* time axis [buf_cap, ...]
    _FULL_INDEXED_FIELDS = (
        "x_before_clone",
        "v_before_clone",
        "U_before",
        "x_final",
        "v_final",
        "U_final",
        "n_alive",
    )

    # Fields indexed with the *minus-one* time axis [buf_cap - 1, ...]
    _MINUS_ONE_INDEXED_FIELDS = (
        "x_after_clone",
        "v_after_clone",
        "U_after_clone",
        "num_cloned",
        "step_times",
        "fitness",
        "rewards",
        "cloning_scores",
        "cloning_probs",
        "will_clone",
        "alive_mask",
        "companions_distance",
        "companions_clone",
        "clone_jitter",
        "clone_delta_x",
        "clone_delta_v",
        "distances",
        "z_rewards",
        "z_distances",
        "pos_squared_differences",
        "vel_squared_differences",
        "rescaled_rewards",
        "rescaled_distances",
        "mu_rewards",
        "sigma_rewards",
        "mu_distances",
        "sigma_distances",
        "force_stable",
        "force_adapt",
        "force_viscous",
        "force_friction",
        "force_total",
        "noise",
    )

    # Optional minus-one fields (may be None)
    _OPTIONAL_MINUS_ONE_FIELDS = (
        "fitness_gradients",
        "fitness_hessians_diag",
        "fitness_hessians_full",
        "sigma_reg_diag",
        "sigma_reg_full",
        "riemannian_volume_weights",
        "ricci_scalar_proxy",
        "diffusion_tensors_full",
    )

    # List-valued fields (variable-length per step)
    _LIST_FIELDS = (
        "neighbor_edges",
        "geodesic_edge_distances",
        "voronoi_regions",
        "edge_weights",
    )

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
        record_volume_weights: bool = False,
        record_ricci_scalar: bool = False,
        record_geodesic_edges: bool = False,
        record_diffusion_tensors: bool = False,
        record_neighbors: bool = False,
        record_voronoi: bool = False,
        record_edge_weights: bool = False,
        chunk_size: int | None = None,
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
            chunk_size: If set, only allocate buffers for this many timesteps
                and flush to disk when full. Reduces peak memory from
                O(n_recorded) to O(chunk_size). ``build()`` merges all
                chunks transparently.
        """
        self.N = N
        self.d = d
        self.n_recorded = n_recorded
        self.device = device
        self.dtype = dtype
        self.recorded_idx = 1  # Index in recorded arrays (0 is initial state)
        self.record_neighbors = record_neighbors
        self.record_voronoi = record_voronoi
        self.record_geodesic_edges = record_geodesic_edges

        # -- Chunking state --------------------------------------------------
        self._chunk_size = chunk_size
        if chunk_size is not None:
            self._buf_capacity = min(chunk_size, n_recorded)
        else:
            self._buf_capacity = n_recorded
        self._chunk_dir: Path | None = None  # created lazily on first flush
        self._flushed_chunks: list[Path] = []
        self._is_first_chunk = True

        buf_cap = self._buf_capacity

        # ====================================================================
        # Preallocate all storage arrays (sized to buf_cap, not n_recorded)
        # ====================================================================

        # States: Before Cloning [buf_cap, N, d]
        self.x_before_clone = torch.zeros(buf_cap, N, d, device=device, dtype=dtype)
        self.v_before_clone = torch.zeros(buf_cap, N, d, device=device, dtype=dtype)
        self.U_before = torch.zeros(buf_cap, N, device=device, dtype=dtype)

        # States: After Cloning [buf_cap-1, N, d]
        self.x_after_clone = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.v_after_clone = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.U_after_clone = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)

        # States: Final (After Kinetic) [buf_cap, N, d]
        self.x_final = torch.zeros(buf_cap, N, d, device=device, dtype=dtype)
        self.v_final = torch.zeros(buf_cap, N, d, device=device, dtype=dtype)
        self.U_final = torch.zeros(buf_cap, N, device=device, dtype=dtype)

        # Per-step scalars [buf_cap] or [buf_cap-1]
        self.n_alive = torch.zeros(buf_cap, dtype=torch.long, device=device)
        self.num_cloned = torch.zeros(buf_cap - 1, dtype=torch.long, device=device)
        self.step_times = torch.zeros(buf_cap - 1, dtype=torch.float32, device=device)

        # Per-walker per-step data [buf_cap-1, N]
        self.fitness = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.rewards = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.cloning_scores = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.cloning_probs = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.will_clone = torch.zeros(buf_cap - 1, N, dtype=torch.bool, device=device)
        self.alive_mask = torch.zeros(buf_cap - 1, N, dtype=torch.bool, device=device)
        self.companions_distance = torch.zeros(buf_cap - 1, N, dtype=torch.long, device=device)
        self.companions_clone = torch.zeros(buf_cap - 1, N, dtype=torch.long, device=device)
        self.clone_jitter = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.clone_delta_x = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.clone_delta_v = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)

        # Fitness intermediate values [buf_cap-1, N]
        self.distances = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.z_rewards = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.z_distances = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.pos_squared_differences = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.vel_squared_differences = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.rescaled_rewards = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        self.rescaled_distances = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)

        # Localized statistics [buf_cap-1]
        self.mu_rewards = torch.zeros(buf_cap - 1, device=device, dtype=dtype)
        self.sigma_rewards = torch.zeros(buf_cap - 1, device=device, dtype=dtype)
        self.mu_distances = torch.zeros(buf_cap - 1, device=device, dtype=dtype)
        self.sigma_distances = torch.zeros(buf_cap - 1, device=device, dtype=dtype)

        # Adaptive kinetics data (optional) [buf_cap-1, N, d] or [buf_cap-1, N, d, d]
        self.fitness_gradients: Tensor | None = None
        self.fitness_hessians_diag: Tensor | None = None
        self.fitness_hessians_full: Tensor | None = None
        self.sigma_reg_diag: Tensor | None = None
        self.sigma_reg_full: Tensor | None = None
        self.riemannian_volume_weights: Tensor | None = None
        self.ricci_scalar_proxy: Tensor | None = None
        self.diffusion_tensors_full: Tensor | None = None

        if record_gradients:
            self.fitness_gradients = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        if record_hessians_diag:
            self.fitness_hessians_diag = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        if record_hessians_full:
            self.fitness_hessians_full = torch.zeros(
                buf_cap - 1, N, d, d, device=device, dtype=dtype
            )
        if record_sigma_reg_diag:
            self.sigma_reg_diag = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        if record_sigma_reg_full:
            self.sigma_reg_full = torch.zeros(buf_cap - 1, N, d, d, device=device, dtype=dtype)
        if record_volume_weights:
            self.riemannian_volume_weights = torch.zeros(
                buf_cap - 1, N, device=device, dtype=dtype
            )
        if record_ricci_scalar:
            self.ricci_scalar_proxy = torch.zeros(buf_cap - 1, N, device=device, dtype=dtype)
        if record_diffusion_tensors:
            self.diffusion_tensors_full = torch.zeros(
                buf_cap - 1, N, d, d, device=device, dtype=dtype
            )

        # Kinetic operator data
        self.force_stable = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.force_adapt = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.force_viscous = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.force_friction = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.force_total = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)
        self.noise = torch.zeros(buf_cap - 1, N, d, device=device, dtype=dtype)

        # Optional neighbor graph storage (variable-length per step)
        self.neighbor_edges: list[Tensor] | None = [] if record_neighbors else None
        self.geodesic_edge_distances: list[Tensor] | None = [] if record_geodesic_edges else None
        self.voronoi_regions: list[dict] | None = [] if record_voronoi else None
        self.edge_weights: list[dict[str, torch.Tensor]] | None = (
            [] if record_edge_weights else None
        )

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
        if self.neighbor_edges is not None:
            self.neighbor_edges.append(torch.zeros((0, 2), dtype=torch.long, device=self.device))
        if self.geodesic_edge_distances is not None:
            self.geodesic_edge_distances.append(
                torch.zeros((0,), dtype=self.dtype, device=self.device)
            )
        if self.voronoi_regions is not None:
            self.voronoi_regions.append({})
        if self.edge_weights is not None:
            self.edge_weights.append({})

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
            if (
                kinetic_info.get("riemannian_volume_weights") is not None
                and self.riemannian_volume_weights is not None
            ):
                self.riemannian_volume_weights[idx_minus_1] = kinetic_info[
                    "riemannian_volume_weights"
                ]
            if (
                kinetic_info.get("ricci_scalar_proxy") is not None
                and self.ricci_scalar_proxy is not None
            ):
                self.ricci_scalar_proxy[idx_minus_1] = kinetic_info["ricci_scalar_proxy"]
            if (
                kinetic_info.get("diffusion_tensors_full") is not None
                and self.diffusion_tensors_full is not None
            ):
                self.diffusion_tensors_full[idx_minus_1] = kinetic_info["diffusion_tensors_full"]

        if self.neighbor_edges is not None:
            edges = info.get("neighbor_edges")
            if edges is None:
                edges = torch.zeros((0, 2), dtype=torch.long, device=self.device)
            self.neighbor_edges.append(edges)
        if self.geodesic_edge_distances is not None:
            geo = info.get("geodesic_edge_distances")
            if geo is None:
                geo = torch.zeros((0,), dtype=self.dtype, device=self.device)
            self.geodesic_edge_distances.append(geo)
        if self.voronoi_regions is not None:
            regions = info.get("voronoi_regions")
            if regions is None:
                regions = {}
            self.voronoi_regions.append(regions)
        if self.edge_weights is not None:
            ew = info.get("edge_weights")
            if ew is None:
                ew = {}
            self.edge_weights.append(ew)

        # Increment recorded index for next step
        self.recorded_idx += 1

        # Auto-flush when chunk buffer is full
        if self._chunk_size is not None and self.recorded_idx >= self._buf_capacity:
            self._flush_chunk()
            self._reset_buffer()

    # ------------------------------------------------------------------
    # Chunked recording helpers
    # ------------------------------------------------------------------

    def _flush_chunk(self) -> None:
        """Save the current buffer contents to a ``.pt`` file on disk."""
        if self._chunk_dir is None:
            self._chunk_dir = Path(tempfile.mkdtemp(prefix="vechistory_chunks_"))

        idx = self.recorded_idx  # number of full-indexed slots written
        chunk: dict[str, object] = {}

        # Full-indexed fields --------------------------------------------------
        if self._is_first_chunk:
            full_slice = slice(0, idx)
        else:
            # Non-first chunks: slot 0 is an unused dummy → skip it
            full_slice = slice(1, idx)

        for name in self._FULL_INDEXED_FIELDS:
            chunk[name] = getattr(self, name)[full_slice].clone()

        # Minus-one-indexed fields ---------------------------------------------
        minus_one_end = idx - 1
        for name in self._MINUS_ONE_INDEXED_FIELDS:
            chunk[name] = getattr(self, name)[0:minus_one_end].clone()

        # Optional minus-one fields --------------------------------------------
        for name in self._OPTIONAL_MINUS_ONE_FIELDS:
            val = getattr(self, name)
            if val is not None:
                chunk[name] = val[0:minus_one_end].clone()

        # List fields ----------------------------------------------------------
        if self._is_first_chunk:
            list_slice = slice(0, idx)
        else:
            list_slice = slice(1, idx)

        for name in self._LIST_FIELDS:
            val = getattr(self, name)
            if val is not None:
                chunk[name] = list(val[list_slice])

        chunk_path = self._chunk_dir / f"chunk_{len(self._flushed_chunks):04d}.pt"
        torch.save(chunk, str(chunk_path))
        self._flushed_chunks.append(chunk_path)

    def _reset_buffer(self) -> None:
        """Reset buffers after a flush for the next chunk of recording."""
        self.recorded_idx = 1  # slot 0 = unused dummy for non-first chunks
        self._is_first_chunk = False

        # Zero out tensor buffers so stale data doesn't leak
        for name in self._FULL_INDEXED_FIELDS:
            getattr(self, name).zero_()
        for name in self._MINUS_ONE_INDEXED_FIELDS:
            getattr(self, name).zero_()
        for name in self._OPTIONAL_MINUS_ONE_FIELDS:
            val = getattr(self, name)
            if val is not None:
                val.zero_()

        # Clear list fields
        for name in self._LIST_FIELDS:
            val = getattr(self, name)
            if val is not None:
                val.clear()
                # Add dummy placeholder for slot 0 (unused in non-first chunks)
                if name == "neighbor_edges":
                    val.append(torch.zeros((0, 2), dtype=torch.long, device=self.device))
                elif name == "geodesic_edge_distances":
                    val.append(torch.zeros((0,), dtype=self.dtype, device=self.device))
                elif name in {"voronoi_regions", "edge_weights"}:
                    val.append({})

    def build(
        self,
        record_every: int,
        terminated_early: bool,
        final_step: int,
        total_time: float,
        init_time: float,
        recorded_steps: list[int] | None = None,
        delta_t: float | None = None,
        params: dict | None = None,
        rng_seed: int | None = None,
        rng_state: dict | None = None,
    ):
        """Construct final RunHistory with trimming to actual recorded size.

        If chunks have been flushed to disk, the remaining buffer is flushed as
        a final chunk and all chunks are merged via ``torch.cat``.  The merged
        tensors are identical to what the unchunked path would produce.

        Args:
            n_steps: Total number of steps requested
            record_every: Recording interval
            terminated_early: Whether run stopped early due to all dead
            final_step: Last step completed (may be < n_steps)
            total_time: Total execution time (seconds)
            init_time: Initialization time (seconds)

        Returns:
            RunHistory object with complete execution trace
        """

        # ----- Chunked path: merge all flushed chunks -----------------------
        if self._flushed_chunks:
            # Flush the remaining buffer as the final chunk (if any data beyond
            # the dummy slot 0 / the initial state recorded_idx==1 case).
            has_remaining = (self._is_first_chunk and self.recorded_idx > 0) or (
                not self._is_first_chunk and self.recorded_idx > 1
            )
            if has_remaining:
                self._flush_chunk()

            merged = self._merge_chunks()

            # Clean up temp files
            if self._chunk_dir is not None:
                shutil.rmtree(self._chunk_dir, ignore_errors=True)
                self._chunk_dir = None
            self._flushed_chunks.clear()

            actual_recorded = merged[self._FULL_INDEXED_FIELDS[0]].shape[0]

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
                params=params,
                rng_seed=rng_seed,
                rng_state=rng_state,
                total_time=total_time,
                init_time=init_time,
                **merged,
            )

        # ----- Unchunked path (original logic) ------------------------------
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
            riemannian_volume_weights=self.riemannian_volume_weights[: actual_recorded - 1]
            if self.riemannian_volume_weights is not None
            else None,
            ricci_scalar_proxy=self.ricci_scalar_proxy[: actual_recorded - 1]
            if self.ricci_scalar_proxy is not None
            else None,
            geodesic_edge_distances=self.geodesic_edge_distances[:actual_recorded]
            if self.geodesic_edge_distances is not None
            else None,
            diffusion_tensors_full=self.diffusion_tensors_full[: actual_recorded - 1]
            if self.diffusion_tensors_full is not None
            else None,
            neighbor_edges=self.neighbor_edges[:actual_recorded]
            if self.neighbor_edges is not None
            else None,
            voronoi_regions=self.voronoi_regions[:actual_recorded]
            if self.voronoi_regions is not None
            else None,
            edge_weights=self.edge_weights[:actual_recorded]
            if self.edge_weights is not None
            else None,
            total_time=total_time,
            init_time=init_time,
        )

    def _merge_chunks(self) -> dict[str, object]:
        """Load all flushed chunks and concatenate each field along dim 0."""
        all_tensor_fields = (
            self._FULL_INDEXED_FIELDS
            + self._MINUS_ONE_INDEXED_FIELDS
            + self._OPTIONAL_MINUS_ONE_FIELDS
        )

        # Accumulators: tensors → list of tensors, lists → flat list
        accum: dict[str, list] = {name: [] for name in all_tensor_fields}
        list_accum: dict[str, list] = {}
        for name in self._LIST_FIELDS:
            if getattr(self, name) is not None:
                list_accum[name] = []

        for chunk_path in self._flushed_chunks:
            chunk = torch.load(str(chunk_path), map_location="cpu", weights_only=True)
            for name in all_tensor_fields:
                if name in chunk:
                    accum[name].append(chunk[name])
            # List fields are saved as plain lists in the chunk
            for name in list_accum:
                if name in chunk:
                    list_accum[name].extend(chunk[name])
            del chunk  # free memory before loading next

        merged: dict[str, object] = {}
        for name in all_tensor_fields:
            parts = accum[name]
            if parts:
                merged[name] = torch.cat(parts, dim=0)
            else:
                merged[name] = None
        for name in list_accum:
            merged[name] = list_accum[name] or None

        # Ensure list fields that were None stay None
        for name in self._LIST_FIELDS:
            if name not in merged:
                merged[name] = None

        return merged
