"""
Euclidean Gas: A Fragile Gas Implementation

This module implements the Euclidean Gas algorithm from clean_build/source/02_euclidean_gas.md
and clean_build/source/03_cloning.md using PyTorch for vectorization and Pydantic for
parameter management.

All tensors are vectorized with the first dimension being the number of walkers N.
"""

from __future__ import annotations

import itertools
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
    potential = param.Parameter(
        default=None,
        doc=(
            "Target potential function. Must be callable: U(x: [N, d]) -> [N]. "
            "Can be an OptimBenchmark instance (which provides bounds) or any callable."
        ),
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

        # Validate that potential is callable
        if self.potential is not None and not callable(self.potential):
            msg = f"potential must be callable, got {type(self.potential)}"
            raise TypeError(msg)

        # Cached scutoid values from previous step for cloning
        self._cached_ricci_scalar: Tensor | None = None
        self._cached_riemannian_volume: Tensor | None = None

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

    def _compute_rewards(self, state: SwarmState) -> Tensor:
        """Compute per-walker rewards from reward 1-form or potential."""
        return -self.potential(state.x)

    def _compute_neighbor_graph(
        self,
        x: Tensor,
    ) -> dict[str, object]:
        """Compute Delaunay neighbor graph. All walkers are assumed alive."""
        import numpy as np

        if x.shape[1] < 2:
            empty = torch.zeros((0, 2), dtype=torch.long, device=x.device)
            return {"neighbor_edges": empty, "updated": True}

        points = x.detach().cpu().numpy()
        try:
            simplices = Delaunay(points).simplices  # [n_simplices, d+1]
        except Exception:
            empty = torch.zeros((0, 2), dtype=torch.long, device=x.device)
            return {"neighbor_edges": empty, "updated": True}

        # Extract all pairwise edges from simplices vectorized
        n_verts = simplices.shape[1]
        pairs = np.array(list(itertools.combinations(range(n_verts), 2)))
        src = simplices[:, pairs[:, 0]].ravel()
        dst = simplices[:, pairs[:, 1]].ravel()
        # Stack both directions and deduplicate
        all_src = np.concatenate([src, dst])
        all_dst = np.concatenate([dst, src])
        edge_array = np.unique(np.stack([all_src, all_dst], axis=1), axis=0)

        neighbor_edges = torch.as_tensor(edge_array, dtype=torch.long, device=x.device)

        return {
            "neighbor_edges": neighbor_edges,
            "updated": True,
        }

    def _maybe_update_neighbor_cache(
        self,
        x: Tensor,
        step_idx: int | None = None,
    ) -> dict[str, object] | None:
        """Update neighbor graph on schedule and cache results."""
        update_every = max(1, int(self.neighbor_graph_update_every))
        cache = getattr(self, "_neighbor_cache", None)
        if step_idx is None:
            step_idx = getattr(self, "_current_step", None)

        if cache is not None and step_idx is not None and (step_idx % update_every != 0):
            return {**cache, "updated": False}

        result = self._compute_neighbor_graph(x)
        self._neighbor_cache = result
        return result

    def step(
        self, state: SwarmState, return_info: bool = False
    ) -> tuple[SwarmState, SwarmState] | tuple[SwarmState, SwarmState, dict] | None:
        """
        Perform one full step: compute fitness, clone (optional), then kinetic (optional).

        Uses cloning.py functions directly to compute:
        1. Rewards from potential
        2. Fitness using compute_fitness (always computed, even if cloning disabled)
        3. Cloning using clone_walkers every n_clone_steps
        4. Kinetic update

        Args:
            state: Current swarm state
            return_info: If True, return full cloning info dictionary

        Returns:
            Tuple of (state_after_cloning, state_after_kinetic), or
            (state_after_cloning, state_after_kinetic, info) if return_info=True

        Note:
            - The info dict contains: fitness, distances, companions, rewards,
              cloning_scores, cloning_probs, will_clone, num_cloned
            - Fitness is always computed (needed for viscous forces)
            - If clone_every > 1, cloning is applied every clone_every steps
            - If enable_kinetic=False, kinetic is skipped and
              state_after_kinetic = state_after_cloning
        """
        if self.potential is not None and hasattr(self.potential, "set_step_id"):
            self.potential.set_step_id(getattr(self, "_current_step", None))

        reference_state = state.clone()

        # Step 1: Compute rewards from reward 1-form or potential
        rewards = self._compute_rewards(state)  # [N]
        U_before = self.potential(state.x) if self.potential is not None else None

        # Select companions using random pairing
        companions_distance = random_pairing_fisher_yates(self.N, device=self.device)

        # Step 4: Compute fitness using core.fitness
        # Always compute fitness even if cloning disabled (needed for adaptive forces)
        fitness, fitness_info = self.fitness_op(
            positions=state.x,
            rewards=rewards,
            companions=companions_distance,
        )

        # Step 5: Execute cloning using cloning.py (if enabled)
        if self.enable_cloning:
            companions_clone = random_pairing_fisher_yates(self.N, device=self.device)
            # Prepare cached values to be cloned from companion
            clone_tensor_kwargs = {"fitness_cloned": fitness}
            if self._cached_ricci_scalar is not None:
                clone_tensor_kwargs["ricci_scalar"] = self._cached_ricci_scalar
            else:
                clone_tensor_kwargs["ricci_scalar"] = torch.zeros(
                    state.N, device=self.device, dtype=self.torch_dtype
                )
            if self._cached_riemannian_volume is not None:
                clone_tensor_kwargs["riemannian_volume"] = self._cached_riemannian_volume
            else:
                clone_tensor_kwargs["riemannian_volume"] = torch.ones(
                    state.N, device=self.device, dtype=self.torch_dtype
                )

            x_cloned, v_cloned, other_cloned, clone_info = self.cloning(
                positions=state.x,
                velocities=state.v,
                fitness=fitness,
                companions=companions_clone,
                **clone_tensor_kwargs,
            )
            step_idx = getattr(self, "_current_step", None)
            clone_every = max(1, int(self.clone_every))
            apply_clone = step_idx is None or clone_every <= 1 or (step_idx % clone_every == 0)
            if apply_clone:
                state_cloned = SwarmState(x_cloned, v_cloned)
            else:
                state_cloned = state.clone()
            clone_info["cloning_applied"] = apply_clone
        else:
            # Skip cloning, use current state
            state_cloned = state.clone()
            clone_info = {
                "cloning_scores": torch.zeros(state.N, device=self.device),
                "cloning_probs": torch.ones(state.N, device=self.device),
                "will_clone": torch.zeros(state.N, dtype=torch.bool, device=self.device),
                "num_cloned": 0,
                "companions": torch.zeros_like(companions_distance),
                "clone_jitter": torch.zeros_like(state.x),
                "clone_delta_x": torch.zeros_like(state.x),
                "clone_delta_v": torch.zeros_like(state.v),
                "cloning_applied": False,
            }
            # No cloning happened, so no cloned tensors
            other_cloned = {}

        neighbor_edges = None

        neighbor_info = self._maybe_update_neighbor_cache(state_cloned.x)
        if neighbor_info is not None:
            neighbor_edges = neighbor_info.get("neighbor_edges")

        # Step 5: Compute fitness derivatives if needed for adaptive kinetics
        grad_fitness = None
        hess_fitness = None
        is_diagonal_hessian = False
        voronoi_data = None
        scutoid_data = None
        scutoid_edges = None
        scutoid_edge_weights = None
        scutoid_all_edge_weights = None
        scutoid_edge_geodesic = None
        scutoid_volume_full = None
        scutoid_ricci_full = None
        scutoid_diffusion_full = None


        try:


            spatial_dims = self.d - 1 if self.d >= 3 else None
            scutoid_data = compute_delaunay_data(
                positions=state_cloned.x,
                fitness_values=fitness,
                spatial_dims=spatial_dims,
                weight_modes=tuple(self.neighbor_weight_modes)
                if self.neighbor_weight_modes
                else None,
            )
            scutoid_edges = scutoid_data.edge_index.t().contiguous()
            scutoid_edge_geodesic = scutoid_data.edge_geodesic_distances
            viscous_mode = (
                self.kinetic_op.viscous_neighbor_weighting
                if self.kinetic_op.use_viscous_coupling
                else None
            )
            scutoid_edge_weights = (
                scutoid_data.edge_weights.get(viscous_mode) if viscous_mode else None
            )
            scutoid_all_edge_weights = scutoid_data.edge_weights

            # All walkers alive — results are already full-N
            scutoid_volume_full = scutoid_data.riemannian_volume_weights
            scutoid_ricci_full = scutoid_data.ricci_proxy
            diffusion_data = scutoid_data.diffusion_tensors
            if diffusion_data.shape[-1] == state.d:
                scutoid_diffusion_full = diffusion_data
            else:
                spatial_d = diffusion_data.shape[-1]
                scutoid_diffusion_full = (
                    torch.eye(state.d, device=self.device, dtype=state_cloned.x.dtype)
                    .unsqueeze(0)
                    .expand(state.N, state.d, state.d)
                    .clone()
                )
                scutoid_diffusion_full[:, :spatial_d, :spatial_d] = diffusion_data
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Delaunay scutoid computation failed: {exc}",
                RuntimeWarning,
            )

        if scutoid_edges is not None:
            neighbor_edges = scutoid_edges

        # Step 6: Kinetic update with optional fitness derivatives (if enabled)
        n_kinetic_steps = max(1, int(getattr(self.kinetic_op, "n_kinetic_steps", 1)))
        state_final = state_cloned
        kinetic_info = {}
        for _ in range(n_kinetic_steps):
            state_final, kinetic_info = self.kinetic_op.apply(
                state_final,
                grad_fitness,
                hess_fitness,
                neighbor_edges=neighbor_edges,
                voronoi_data=voronoi_data,
                edge_weights=scutoid_edge_weights,
                volume_weights=scutoid_volume_full,
                diffusion_tensors=scutoid_diffusion_full,
                return_info=True,
            )

        if scutoid_volume_full is not None:
            kinetic_info["riemannian_volume_weights"] = scutoid_volume_full
        if scutoid_ricci_full is not None:
            kinetic_info["ricci_scalar_proxy"] = scutoid_ricci_full
        if scutoid_diffusion_full is not None:
            kinetic_info["diffusion_tensors_full"] = scutoid_diffusion_full

        # Cache scutoid values for next step's cloning
        if scutoid_ricci_full is not None:
            self._cached_ricci_scalar = scutoid_ricci_full.clone()
        if scutoid_volume_full is not None:
            self._cached_riemannian_volume = scutoid_volume_full.clone()

        if voronoi_data is not None:
            volume_weights = kinetic_info.get("riemannian_volume_weights")
            if volume_weights is not None:
                voronoi_data["riemannian_volume_weights"] = volume_weights

        if self.potential is not None and hasattr(self.potential, "update_voronoi_cache"):
            step_id = getattr(self, "_current_step", None)
            if voronoi_data is not None:
                self.potential.update_voronoi_cache(
                    voronoi_data,
                    state_cloned.x,
                    step_id,
                    dt=self.kinetic_op.delta_t,
                )
            elif (
                neighbor_info is not None
                and neighbor_info.get("updated")
                and neighbor_info.get("voronoi_regions") is not None
            ):
                self.potential.update_voronoi_cache(
                    neighbor_info["voronoi_regions"],
                    state_cloned.x,
                    step_id,
                    dt=self.kinetic_op.delta_t,
                )

        if self.potential is not None and hasattr(self.potential, "update_scutoid_cache"):
            if scutoid_volume_full is not None or scutoid_ricci_full is not None:
                # Use cloned values where available so cloned walkers get
                # their companion's precomputed values
                ricci_for_cache = other_cloned.get("ricci_scalar", scutoid_ricci_full)
                volume_for_cache = other_cloned.get("riemannian_volume", scutoid_volume_full)
                scutoid_cache = {
                    "riemannian_volume_weights": volume_for_cache,
                    "ricci_scalar": ricci_for_cache,
                }
                step_id = getattr(self, "_current_step", None)
                self.potential.update_scutoid_cache(
                    scutoid_cache,
                    state_cloned.x,
                    step_id,
                    dt=self.kinetic_op.delta_t,
                )

        U_after_clone = self.potential(state_cloned.x) if self.potential is not None else None
        U_final = self.potential(state_final.x) if self.potential is not None else None

        if return_info:
            # Combine all computed data into info dict
            info = {
                "fitness": fitness,
                "rewards": rewards,
                "companions_distance": companions_distance,
                "companions_clone": clone_info["companions"],
                **clone_info,  # Adds: cloning_scores, cloning_probs, will_clone, num_cloned
                **fitness_info,
                "U_before": U_before if U_before is not None else torch.zeros_like(rewards),
                "U_after_clone": U_after_clone
                if U_after_clone is not None
                else torch.zeros_like(rewards),
                "U_final": U_final if U_final is not None else torch.zeros_like(rewards),
                "kinetic_info": kinetic_info,
            }
            if neighbor_info is not None:
                info["neighbor_edges"] = neighbor_edges
                info["voronoi_regions"] = neighbor_info.get("voronoi_regions")
                info["neighbor_updated"] = neighbor_info.get("updated")
            elif neighbor_edges is not None:
                info["neighbor_edges"] = neighbor_edges
            if scutoid_edge_geodesic is not None:
                info["geodesic_edge_distances"] = scutoid_edge_geodesic
            if scutoid_all_edge_weights is not None:
                info["edge_weights"] = scutoid_all_edge_weights
            if grad_fitness is not None:
                info["grad_fitness"] = grad_fitness
            if hess_fitness is not None:
                info["hess_fitness"] = hess_fitness
                info["is_diagonal_hessian"] = is_diagonal_hessian
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
        use_tree_history: bool = False,
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
            use_tree_history: If True, use a graph-backed TreeHistory recorder
                instead of the default VectorizedHistoryRecorder.

        Returns:
            RunHistory object with complete execution trace including:
                - States before cloning, after cloning, and after kinetic at each recorded step
                - Fitness, cloning, and companion data
                - Adaptive kinetics data (gradients/Hessians) if computed
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
                    "freeze_best": self.freeze_best,
                    "enable_cloning": self.enable_cloning,
                    "clone_every": self.clone_every,
                    "enable_kinetic": self.enable_kinetic,
                    "pbc": False,
                    "pbc_fitness_only": False,
                },
                "companion_selection": {
                    "method": self.companion_selection.method
                    if self.companion_selection
                    else None,
                    "epsilon": self.companion_selection.epsilon
                    if self.companion_selection
                    else None,
                    "lambda_alg": self.companion_selection.lambda_alg
                    if self.companion_selection
                    else None,
                    "exclude_self": self.companion_selection.exclude_self
                    if self.companion_selection
                    else None,
                },
                "companion_selection_clone": {
                    "method": self.companion_selection_clone.method
                    if self.companion_selection_clone
                    else None,
                    "epsilon": self.companion_selection_clone.epsilon
                    if self.companion_selection_clone
                    else None,
                    "lambda_alg": self.companion_selection_clone.lambda_alg
                    if self.companion_selection_clone
                    else None,
                    "exclude_self": self.companion_selection_clone.exclude_self
                    if self.companion_selection_clone
                    else None,
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
                    "auto_thermostat": getattr(self.kinetic_op, "auto_thermostat", False),
                    "sigma_v": getattr(self.kinetic_op, "sigma_v", None),
                    "beta_effective": (
                        float(self.kinetic_op.effective_beta())
                        if hasattr(self.kinetic_op, "effective_beta")
                        else float(self.kinetic_op.beta)
                    ),
                    "delta_t": self.kinetic_op.delta_t,
                    "n_kinetic_steps": getattr(self.kinetic_op, "n_kinetic_steps", 1),
                    "epsilon_F": self.kinetic_op.epsilon_F,
                    "use_fitness_force": self.kinetic_op.use_fitness_force,
                    "use_potential_force": self.kinetic_op.use_potential_force,
                    "use_anisotropic_diffusion": self.kinetic_op.use_anisotropic_diffusion,
                    "diagonal_diffusion": self.kinetic_op.diagonal_diffusion,
                    "epsilon_Sigma": self.kinetic_op.epsilon_Sigma,
                    "nu": self.kinetic_op.nu,
                    "use_viscous_coupling": self.kinetic_op.use_viscous_coupling,
                    "viscous_length_scale": self.kinetic_op.viscous_length_scale,
                    "viscous_neighbor_weighting": self.kinetic_op.viscous_neighbor_weighting,
                    "viscous_neighbor_threshold": self.kinetic_op.viscous_neighbor_threshold,
                    "viscous_neighbor_penalty": self.kinetic_op.viscous_neighbor_penalty,
                    "viscous_degree_cap": self.kinetic_op.viscous_degree_cap,
                    "viscous_volume_weighting": self.kinetic_op.viscous_volume_weighting,
                    "compute_volume_weights": self.kinetic_op.compute_volume_weights,
                    "beta_curl": self.kinetic_op.beta_curl,
                    "use_velocity_squashing": self.kinetic_op.use_velocity_squashing,
                    "V_alg": self.kinetic_op.V_alg,
                },
                "fitness": {
                    "alpha": self.fitness_op.alpha if self.fitness_op else None,
                    "beta": self.fitness_op.beta if self.fitness_op else None,
                    "eta": self.fitness_op.eta if self.fitness_op else None,
                    "lambda_alg": self.fitness_op.lambda_alg if self.fitness_op else None,
                    "sigma_min": self.fitness_op.sigma_min if self.fitness_op else None,
                    "epsilon_dist": self.fitness_op.epsilon_dist if self.fitness_op else None,
                    "A": self.fitness_op.A if self.fitness_op else None,
                    "rho": self.fitness_op.rho if self.fitness_op else None,
                },
                "potential": {
                    "name": type(self.potential).__name__ if self.potential is not None else None
                },
                "reward_1form": {
                    "name": type(self.reward_1form).__name__
                    if self.reward_1form is not None
                    else None
                },
                "neighbor_graph": {
                    "method": self.neighbor_graph_method,
                    "update_every": self.neighbor_graph_update_every,
                    "record": self.neighbor_graph_record,
                },
            }

        # Initialize vectorized history recorder with pre-allocated arrays
        from fragile.fractalai.core.vec_history import VectorizedHistoryRecorder

        record_gradients = False
        record_hessians_diag = False
        record_hessians_full = False

        if self.fitness_op is not None:
            needs_grad = self.kinetic_op.use_fitness_force or (
                self.kinetic_op.use_anisotropic_diffusion
                and getattr(self.kinetic_op, "diffusion_mode", "hessian") == "grad_proxy"
            )
            record_gradients = needs_grad
            if (
                self.kinetic_op.use_anisotropic_diffusion
                and getattr(self.kinetic_op, "diffusion_mode", "hessian") == "hessian"
            ):
                record_hessians_diag = self.kinetic_op.diagonal_diffusion
                record_hessians_full = not self.kinetic_op.diagonal_diffusion

        record_sigma_reg_diag = False
        record_sigma_reg_full = False
        if self.kinetic_op.use_anisotropic_diffusion:
            record_sigma_reg_diag = self.kinetic_op.diagonal_diffusion
            record_sigma_reg_full = not self.kinetic_op.diagonal_diffusion
        record_scutoid = self.neighbor_graph_record or (
            self.fitness_op is not None
            and self.enable_kinetic
            and (
                self.kinetic_op.use_viscous_coupling
                or self.kinetic_op.use_anisotropic_diffusion
                or bool(getattr(self.kinetic_op, "compute_volume_weights", False))
            )
        )
        record_volume_weights = bool(
            getattr(self.kinetic_op, "compute_volume_weights", False) or record_scutoid
        )
        record_edge_weights = (
            self.neighbor_graph_record
            and self.neighbor_graph_method != "none"
            and bool(self.neighbor_weight_modes)
        )

        if use_tree_history:
            from fragile.fractalai.core.tree_history import TreeHistory

            recorder = TreeHistory(
                N=N,
                d=d,
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            recorder = VectorizedHistoryRecorder(
                N=N,
                d=d,
                n_recorded=n_recorded,
                device=self.device,
                dtype=self.torch_dtype,
                record_gradients=record_gradients,
                record_hessians_diag=record_hessians_diag,
                record_hessians_full=record_hessians_full,
                record_sigma_reg_diag=record_sigma_reg_diag,
                record_sigma_reg_full=record_sigma_reg_full,
                record_volume_weights=record_volume_weights,
                record_ricci_scalar=record_scutoid,
                record_geodesic_edges=record_scutoid,
                record_diffusion_tensors=record_scutoid,
                record_neighbors=self.neighbor_graph_record
                and self.neighbor_graph_method != "none",
                record_voronoi=self.neighbor_graph_record and self.neighbor_graph_method != "none",
                record_edge_weights=record_edge_weights,
                chunk_size=chunk_size,
            )
        self._neighbor_cache = None

        # All walkers always alive
        n_alive = N

        # Record initial state (t=0)
        U_initial = self.potential(state.x) if self.potential is not None else None
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

            # Execute step with return_info=True to get all data
            self._current_step = t
            state_cloned, state_final, info = self.step(state, return_info=True)

            # Compute adaptive kinetics derivatives if enabled
            grad_fitness = info.get("grad_fitness")
            hess_fitness = info.get("hess_fitness")
            is_diagonal_hessian = info.get("is_diagonal_hessian", False)

            # Determine if this step should be recorded
            should_record = t in recorded_steps

            if should_record:
                # Record all data for this step using recorder
                recorder.record_step(
                    state_before=state,
                    state_cloned=state_cloned,
                    state_final=state_final,
                    info=info,
                    step_time=time.time() - step_start,
                    grad_fitness=grad_fitness,
                    hess_fitness=hess_fitness,
                    is_diagonal_hessian=is_diagonal_hessian,
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
            n_steps=final_step,
            record_every=record_every,
            terminated_early=terminated_early,
            final_step=final_step,
            total_time=total_time,
            init_time=init_time,
            bounds=None,
            recorded_steps=recorded_steps,
            delta_t=self.kinetic_op.delta_t,
            pbc=False,
            params=params,
            rng_seed=seed,
            rng_state=rng_state,
        )
