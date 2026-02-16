"""
Euclidean Gas: A Fragile Gas Implementation

This module implements the Euclidean Gas algorithm from clean_build/source/02_euclidean_gas.md
and clean_build/source/03_cloning.md using PyTorch for vectorization and Pydantic for
parameter management.

All tensors are vectorized with the first dimension being the number of walkers N.
"""

from __future__ import annotations

from typing import Callable

import panel as pn
import param
import torch
from torch import Tensor

from fragile.physics.fractal_gas.panel_model import INPUT_WIDTH, PanelModel
from fragile.physics.geometry.delaunai import compute_delaunay_data


def random_pairing_fisher_yates(
    n: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create random mutual pairs using Fisher-Yates shuffle (O(N) algorithm).

    All walkers are assumed alive. The algorithm:
    1. Generate random permutation using PyTorch's randperm
    2. Pair consecutive elements: (perm[0], perm[1]), (perm[2], perm[3]), ...

    Args:
        n: Number of walkers.
        device: PyTorch device for the output tensor.

    Returns:
        Companion map [N], where c(i) = j and c(j) = i for each pair.
        If N is odd, the last unpaired walker maps to itself.
    """
    companion_map = torch.arange(n, dtype=torch.long, device=device)
    if n < 2:
        return companion_map

    permuted = torch.randperm(n, device=device)

    n_pairs = n // 2
    even_idx = permuted[0 : 2 * n_pairs : 2]
    odd_idx = permuted[1 : 2 * n_pairs : 2]

    companion_map[even_idx] = odd_idx
    companion_map[odd_idx] = even_idx

    return companion_map


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


class EuclideanGas(PanelModel):
    """Complete parameter set for Euclidean Gas algorithm."""

    _n_widget_columns = param.Integer(default=2, bounds=(1, None), doc="Number of widget columns")
    _max_widget_width = param.Integer(default=800, bounds=(0, None), doc="Maximum widget width")

    N = param.Integer(default=50, bounds=(1, None), softbounds=(50, 500), doc="Number of walkers")
    d = param.Integer(default=3, bounds=(1, None), doc="Spatial dimension")
    eh_scale = param.Number(
        default=1.0,
        bounds=(0, None),
        doc="Scale for the Einstein-Hilbert action reward: reward_i = eh_scale * R_i * sqrt(det g)_i",
    )
    kinetic_op = param.Parameter(default=None, doc="Langevin dynamics parameters")
    cloning = param.Parameter(default=None, doc="Cloning operator")
    fitness_op = param.Parameter(
        default=None,
        allow_None=True,
        doc="Fitness operator (required if using adaptive kinetics features)",
    )
    device = param.Parameter(default=torch.device("cpu"), doc="PyTorch device (cpu/cuda)")
    dtype = param.Selector(
        default="float32", objects=["float32", "float64"], doc="PyTorch dtype (float32/float64)"
    )
    clone_every = param.Integer(
        default=1,
        bounds=(1, None),
        doc="Apply cloning every N steps (scores still computed every step)",
    )
    neighbor_graph_update_every = param.Integer(
        default=1, bounds=(1, None), doc="Recompute neighbor graph every k steps"
    )
    neighbor_weight_modes = param.ListSelector(
        default=["riemannian_kernel_volume"],
        objects=[
            "uniform",
            "inverse_distance",
            "inverse_volume",
            "inverse_riemannian_volume",
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel",
            "riemannian_kernel_volume",
        ],
        doc="Edge weight modes to pre-compute during Voronoi tessellation",
    )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for Euclidean Gas parameters."""
        return {
            "N": {
                "type": pn.widgets.EditableIntSlider,
                "width": INPUT_WIDTH,
                "name": "N (num walkers)",
                "start": 2,
                "end": 10000,
                "step": 1,
            },
            "d": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
                "name": "d (dimension)",
            },
            "dtype": {
                "type": pn.widgets.Select,
                "width": INPUT_WIDTH,
                "name": "Data type",
            },
            "clone_every": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
                "name": "Clone every (steps)",
                "start": 1,
                "end": 10000,
                "step": 1,
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI (excluding nested operators and internal objects)."""
        return [
            "N",
            "d",
            "dtype",
            "clone_every",
        ]

    def __init__(self, **params):
        """Initialize Euclidean Gas with post-initialization validation."""
        super().__init__(**params)

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
            v_init: Initial velocities [N, d] (optional, defaults to 0)

        Returns:
            Initial swarm state
        """
        N, d = self.N, self.d

        if x_init is None:
            x_init = torch.randn(N, d, device=self.device, dtype=self.torch_dtype)

        if v_init is None:
            # Start walkers at rest by default.
            v_init = torch.zeros(N, d, device=self.device, dtype=self.torch_dtype)

        return SwarmState(
            x_init.to(device=self.device, dtype=self.torch_dtype),
            v_init.to(device=self.device, dtype=self.torch_dtype),
        )

    def step(
        self, state: SwarmState, return_info: bool = False
    ) -> tuple[SwarmState, SwarmState] | tuple[SwarmState, SwarmState, dict] | None:
        """
        Perform one full step: tessellation, EH rewards, fitness, clone, kinetic.

        Flow:
        1. Delaunay tessellation on current positions → Ricci, volume, edges
        2. EH rewards from fresh geometry: eh_scale * R_i * √(det g)_i
        3. Fitness from rewards
        4. Cloning (optional)
        5. Kinetic update

        Args:
            state: Current swarm state
            return_info: If True, return full info dictionary

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic), or
            (state_after_cloning, state_after_kinetic, info) if return_info=True
        """
        state.clone()

        # Step 1: Delaunay tessellation on current positions
        delaunay_edges = None
        delaunay_edge_weights = None
        delaunay_all_edge_weights = None
        delaunay_edge_geodesic = None
        delaunay_volume = None
        delaunay_ricci = None
        delaunay_diffusion = None

        update_every = max(1, int(self.neighbor_graph_update_every))
        step_idx = getattr(self, "_current_step", None)
        recompute = step_idx is None or (step_idx % update_every == 0)

        if recompute:
            try:
                spatial_dims = self.d - 1 if self.d >= 3 else None
                delaunay_data = compute_delaunay_data(
                    positions=state.x,
                    fitness_values=torch.zeros(
                        state.N, device=self.device, dtype=self.torch_dtype
                    ),
                    spatial_dims=spatial_dims,
                    weight_modes=tuple(self.neighbor_weight_modes)
                    if self.neighbor_weight_modes
                    else None,
                )
                delaunay_edges = delaunay_data.edge_index.t().contiguous()
                delaunay_edge_geodesic = delaunay_data.edge_geodesic_distances
                viscous_mode = (
                    self.kinetic_op.viscous_neighbor_weighting
                    if self.kinetic_op.use_viscous_coupling
                    else None
                )
                delaunay_edge_weights = (
                    delaunay_data.edge_weights.get(viscous_mode) if viscous_mode else None
                )
                delaunay_all_edge_weights = delaunay_data.edge_weights

                delaunay_volume = delaunay_data.riemannian_volume_weights
                delaunay_ricci = delaunay_data.ricci_proxy
                diffusion_data = delaunay_data.diffusion_tensors
                if diffusion_data.shape[-1] == state.d:
                    delaunay_diffusion = diffusion_data
                else:
                    spatial_d = diffusion_data.shape[-1]
                    delaunay_diffusion = (
                        torch
                        .eye(state.d, device=self.device, dtype=state.x.dtype)
                        .unsqueeze(0)
                        .expand(state.N, state.d, state.d)
                        .clone()
                    )
                    delaunay_diffusion[:, :spatial_d, :spatial_d] = diffusion_data

                self._cached_delaunay_data = {
                    "delaunay_edges": delaunay_edges,
                    "delaunay_edge_weights": delaunay_edge_weights,
                    "delaunay_all_edge_weights": delaunay_all_edge_weights,
                    "delaunay_edge_geodesic": delaunay_edge_geodesic,
                    "delaunay_volume": delaunay_volume,
                    "delaunay_ricci": delaunay_ricci,
                    "delaunay_diffusion": delaunay_diffusion,
                }
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"Delaunay computation failed: {exc}",
                    RuntimeWarning,
                )
        else:
            cached = getattr(self, "_cached_delaunay_data", None)
            if cached is not None:
                delaunay_edges = cached["delaunay_edges"]
                delaunay_edge_weights = cached["delaunay_edge_weights"]
                delaunay_all_edge_weights = cached["delaunay_all_edge_weights"]
                delaunay_edge_geodesic = cached["delaunay_edge_geodesic"]
                delaunay_volume = cached["delaunay_volume"]
                delaunay_ricci = cached["delaunay_ricci"]
                delaunay_diffusion = cached["delaunay_diffusion"]

        neighbor_edges = delaunay_edges

        # Step 2: EH rewards from fresh tessellation
        rewards = self.eh_scale * delaunay_ricci * delaunay_volume
        U_before = -rewards  # U = -reward by convention

        # Step 3: Fitness from rewards
        companions_distance = random_pairing_fisher_yates(self.N, device=self.device)
        fitness, fitness_info = self.fitness_op(
            positions=state.x,
            rewards=rewards,
            companions=companions_distance,
        )

        # Step 4: Cloning

        companions_clone = random_pairing_fisher_yates(self.N, device=self.device)
        clone_tensor_kwargs = {
            "fitness_cloned": fitness,
            "ricci_scalar": delaunay_ricci
            if delaunay_ricci is not None
            else torch.zeros(state.N, device=self.device, dtype=self.torch_dtype),
            "riemannian_volume": delaunay_volume
            if delaunay_volume is not None
            else torch.ones(state.N, device=self.device, dtype=self.torch_dtype),
        }

        x_cloned, v_cloned, other_cloned, clone_info = self.cloning(
            positions=state.x,
            velocities=state.v,
            fitness=fitness,
            companions=companions_clone,
            **clone_tensor_kwargs,
        )
        clone_every = max(1, int(self.clone_every))
        apply_clone = step_idx is None or clone_every <= 1 or (step_idx % clone_every == 0)
        if apply_clone:
            state_cloned = SwarmState(x_cloned, v_cloned)
        else:
            state_cloned = state.clone()
        clone_info["cloning_applied"] = apply_clone
        fitness = other_cloned.get("fitness_cloned", fitness)

        # Remap edges for cloned walkers: use companion's neighborhood
        # so they receive the same viscous force as the walker they copied.
        if (
            neighbor_edges is not None
            and clone_info.get("cloning_applied")
            and clone_info["num_cloned"] > 0
        ):
            will_clone = clone_info["will_clone"]
            if will_clone.any():
                src = neighbor_edges[:, 0]

                # companion→clone map (each companion maps to the walker that copied it)
                cloned_idx = torch.where(will_clone)[0]
                comp_idx = clone_info["companions"][cloned_idx]
                comp_to_clone = torch.full((state.N,), -1, dtype=torch.long, device=self.device)
                comp_to_clone[comp_idx] = cloned_idx

                # Keep all edges NOT from cloned walkers
                keep = ~will_clone[src]

                # Duplicate companion's outgoing edges with source remapped to clone
                has_clone = comp_to_clone[src] >= 0
                from_comp = has_clone & keep
                dup_edges = neighbor_edges[from_comp].clone()
                dup_edges[:, 0] = comp_to_clone[dup_edges[:, 0]]

                neighbor_edges = torch.cat([neighbor_edges[keep], dup_edges], dim=0)
                if delaunay_edge_weights is not None:
                    delaunay_edge_weights = torch.cat(
                        [delaunay_edge_weights[keep], delaunay_edge_weights[from_comp]],
                        dim=0,
                    )
                if delaunay_edge_geodesic is not None:
                    delaunay_edge_geodesic = torch.cat(
                        [delaunay_edge_geodesic[keep], delaunay_edge_geodesic[from_comp]],
                        dim=0,
                    )
                if delaunay_all_edge_weights is not None:
                    delaunay_all_edge_weights = {
                        k: torch.cat([w[keep], w[from_comp]], dim=0)
                        for k, w in delaunay_all_edge_weights.items()
                    }

        # Step 5: Kinetic update
        n_kinetic_steps = max(1, int(getattr(self.kinetic_op, "n_kinetic_steps", 1)))
        state_final = state_cloned
        kinetic_info = {}
        for _ in range(n_kinetic_steps):
            state_final, kinetic_info = self.kinetic_op.apply(
                state_final,
                neighbor_edges=neighbor_edges,
                edge_weights=delaunay_edge_weights,
                return_info=True,
            )

        if delaunay_volume is not None:
            kinetic_info["riemannian_volume_weights"] = delaunay_volume
        if delaunay_ricci is not None:
            kinetic_info["ricci_scalar_proxy"] = delaunay_ricci
        if delaunay_diffusion is not None:
            kinetic_info["diffusion_tensors_full"] = delaunay_diffusion

        # U_after uses clone-aware geometry (cloned walkers inherit companion's values)
        ricci_final = other_cloned.get("ricci_scalar", delaunay_ricci)
        volume_final = other_cloned.get("riemannian_volume", delaunay_volume)
        if ricci_final is not None and volume_final is not None:
            eh_value = -(self.eh_scale * ricci_final * volume_final)
        else:
            eh_value = torch.zeros(state.N, device=self.device, dtype=self.torch_dtype)
        U_after_clone = eh_value
        U_final = eh_value

        if return_info:
            info = {
                "fitness": fitness,
                "rewards": rewards,
                "companions_distance": companions_distance,
                "companions_clone": clone_info["companions"],
                **clone_info,
                **fitness_info,
                "U_before": U_before,
                "U_after_clone": U_after_clone,
                "U_final": U_final,
                "kinetic_info": kinetic_info,
            }
            if neighbor_edges is not None:
                info["neighbor_edges"] = neighbor_edges
            if delaunay_edge_geodesic is not None:
                info["geodesic_edge_distances"] = delaunay_edge_geodesic
            if delaunay_all_edge_weights is not None:
                info["edge_weights"] = delaunay_all_edge_weights
            return state_cloned, state_final, info
        return state_cloned, state_final

    def run(
        self,
        n_steps: int,
        x_init: Tensor | None = None,
        v_init: Tensor | None = None,
        record_every: int = 1,
        seed: int | None = None,
        record_rng_state: bool = False,
        show_progress: bool = False,
        progress_callback: Callable[[int, int, float], None] | None = None,
        chunk_size: int | None = None,
    ):
        """
        Run Euclidean Gas for multiple steps and return complete history.

        Args:
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)
            record_every: Record every k-th step (1=all steps, 10=every 10th step).
                         Step 0 (initial) and final step are always recorded.
            show_progress: Show tqdm progress bar during simulation.
            progress_callback: Optional callback invoked as (step, total_steps, elapsed_seconds).
            chunk_size: If set, the history recorder keeps only this many
                timesteps in memory and flushes full chunks to disk. Reduces
                peak memory from O(n_recorded) to O(chunk_size).

        Returns:
            RunHistory object with complete execution trace including:
                - States before cloning, after cloning, and after kinetic at each recorded step
                - Fitness, cloning, and companion data
                - Timing information

        Note:
            All walkers are always alive. Memory scales with n_steps/record_every,
            so use record_every > 1 for long runs.

        Example:
            >>> gas = EuclideanGas(N=50, d=2, ...)
            >>> history = gas.run(n_steps=1000, record_every=10)  # Record every 10 steps
            >>> print(history.summary())
            >>> history.save("run_001.pt")
        """
        import time

        # Initialize state with timing
        init_start = time.time()
        if seed is not None:
            torch.manual_seed(seed)
            try:
                import numpy as np

                np.random.seed(seed)
            except ImportError:
                pass
        state = self.initialize_state(x_init, v_init)

        init_time = time.time() - init_start

        N, d = state.N, state.d

        # Calculate number of recorded timesteps
        # Step 0 is always recorded, then every record_every steps,
        # plus final step if not at interval
        recorded_steps = list(range(0, n_steps + 1, record_every))
        if n_steps not in recorded_steps:
            recorded_steps.append(n_steps)
        n_recorded = len(recorded_steps)

        def _capture_rng_state() -> dict[str, object] | None:
            if not record_rng_state:
                return None
            rng_state: dict[str, object] = {"torch": torch.get_rng_state()}
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            try:
                import numpy as np

                rng_state["numpy"] = np.random.get_state()
            except ImportError:
                pass
            return rng_state

        def _build_params() -> dict[str, dict[str, object]]:
            return {
                "gas": {
                    "N": self.N,
                    "d": self.d,
                    "dtype": self.dtype,
                    "eh_scale": self.eh_scale,
                    "clone_every": self.clone_every,
                },
                "cloning": {
                    "p_max": self.cloning.p_max if self.cloning else None,
                    "epsilon_clone": self.cloning.epsilon_clone if self.cloning else None,
                    "sigma_x": self.cloning.sigma_x if self.cloning else None,
                    "alpha_restitution": self.cloning.alpha_restitution if self.cloning else None,
                },
                "kinetic": {
                    "gamma": self.kinetic_op.gamma,
                    "beta": self.kinetic_op.beta,
                    "delta_t": self.kinetic_op.delta_t,
                    "n_kinetic_steps": getattr(self.kinetic_op, "n_kinetic_steps", 1),
                    "nu": self.kinetic_op.nu,
                    "use_viscous_coupling": self.kinetic_op.use_viscous_coupling,
                    "viscous_neighbor_weighting": self.kinetic_op.viscous_neighbor_weighting,
                    "beta_curl": self.kinetic_op.beta_curl,
                },
                "fitness": {
                    "alpha": self.fitness_op.alpha if self.fitness_op else None,
                    "beta": self.fitness_op.beta if self.fitness_op else None,
                    "eta": self.fitness_op.eta if self.fitness_op else None,
                    "sigma_min": self.fitness_op.sigma_min if self.fitness_op else None,
                    "A": self.fitness_op.A if self.fitness_op else None,
                },
                "neighbor_graph": {
                    "update_every": self.neighbor_graph_update_every,
                    "weight_modes": list(self.neighbor_weight_modes),
                },
            }

        # Initialize vectorized history recorder with pre-allocated arrays
        from fragile.physics.fractal_gas.history import VectorizedHistoryRecorder

        recorder = VectorizedHistoryRecorder(
            N=N,
            d=d,
            n_recorded=n_recorded,
            device=self.device,
            dtype=self.torch_dtype,
            record_volume_weights=True,
            record_ricci_scalar=True,
            record_geodesic_edges=True,
            record_diffusion_tensors=True,
            record_neighbors=True,
            record_voronoi=True,
            record_edge_weights=True,
            chunk_size=chunk_size,
        )
        self._cached_delaunay_data = None

        # All walkers always alive
        n_alive = N

        # Record initial state (t=0)
        U_initial = torch.zeros(state.N, device=self.device, dtype=self.torch_dtype)
        recorder.record_initial_state(state, n_alive, U_before=U_initial, U_final=U_initial)

        # Run steps with timing
        terminated_early = False
        final_step = n_steps
        total_start = time.time()

        if progress_callback is not None:
            progress_callback(0, n_steps, 0.0)

        # Set up progress bar if requested
        step_iterator = range(1, n_steps + 1)
        if show_progress:
            try:
                from tqdm import tqdm

                step_iterator = tqdm(step_iterator, desc="Running simulation", unit="step")
            except ImportError:
                pass  # tqdm not available, run without progress bar

        for t in step_iterator:
            step_start = time.time()

            try:
                # Execute step with return_info=True to get all data
                self._current_step = t
                state_cloned, state_final, info = self.step(state, return_info=True)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                terminated_early = True
                final_step = t - 1
                print(f"Simulation terminated early at step {t}: {exc}")
                break

            # Determine if this step should be recorded
            should_record = t in recorded_steps

            if should_record:
                # Physics gas has no alive mask — all walkers are always alive.
                info.setdefault(
                    "alive_mask", torch.ones(state.N, dtype=torch.bool, device=self.device)
                )
                recorder.record_step(
                    state_before=state,
                    state_cloned=state_cloned,
                    state_final=state_final,
                    info=info,
                    step_time=time.time() - step_start,
                    kinetic_info=info.get("kinetic_info"),
                )

            # Update state for next iteration
            state = state_final

            if progress_callback is not None:
                progress_callback(t, n_steps, time.time() - total_start)

        total_time = time.time() - total_start

        if progress_callback is not None:
            progress_callback(final_step, n_steps, total_time)

        # Build final RunHistory with automatic trimming to actual recorded size
        recorded_steps = recorded_steps[: recorder.recorded_idx]
        rng_state = _capture_rng_state()
        params = _build_params()

        return recorder.build(
            record_every=record_every,
            terminated_early=terminated_early,
            final_step=final_step,
            total_time=total_time,
            init_time=init_time,
            recorded_steps=recorded_steps,
            delta_t=self.kinetic_op.delta_t,
            params=params,
            rng_seed=seed,
            rng_state=rng_state,
        )
