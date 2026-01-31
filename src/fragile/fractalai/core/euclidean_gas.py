"""
Euclidean Gas: A Fragile Gas Implementation

This module implements the Euclidean Gas algorithm from clean_build/source/02_euclidean_gas.md
and clean_build/source/03_cloning.md using PyTorch for vectorization and Pydantic for
parameter management.

All tensors are vectorized with the first dimension being the number of walkers N.
"""

from __future__ import annotations

from typing import Callable

import itertools
import panel as pn
import param
import torch
from torch import Tensor

from fragile.fractalai.core.panel_model import INPUT_WIDTH, PanelModel


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
    d = param.Integer(default=2, bounds=(1, None), doc="Spatial dimension")
    companion_selection = param.Parameter(
        default=None, doc="Companion selection strategy for diversity/fitness"
    )
    companion_selection_clone = param.Parameter(
        default=None, doc="Companion selection strategy for cloning (optional override)"
    )
    potential = param.Parameter(
        default=None,
        doc=(
            "Target potential function. Must be callable: U(x: [N, d]) -> [N]. "
            "Can be an OptimBenchmark instance (which provides bounds) or any callable."
        ),
    )
    reward_1form = param.Parameter(
        default=None,
        doc=(
            "Optional reward 1-form field. Must be callable: R(x: [N, d]) -> [N, d]. "
            "If provided, rewards are computed as <R(x), v> per walker."
        ),
    )
    kinetic_op = param.Parameter(default=None, doc="Langevin dynamics parameters")
    cloning = param.Parameter(default=None, doc="Cloning operator")
    fitness_op = param.Parameter(
        default=None,
        allow_None=True,
        doc="Fitness operator (required if using adaptive kinetics features)",
    )
    bounds = param.Parameter(
        default=None,
        allow_None=True,
        doc=(
            "Position bounds (optional, TorchBounds only). "
            "If None and potential has a 'bounds' attribute, bounds will be auto-extracted."
        ),
    )
    device = param.Parameter(default=torch.device("cpu"), doc="PyTorch device (cpu/cuda)")
    dtype = param.Selector(
        default="float32", objects=["float32", "float64"], doc="PyTorch dtype (float32/float64)"
    )
    freeze_best = param.Boolean(
        default=False,
        doc="Keep the highest-fitness walker untouched during cloning and kinetic steps.",
    )
    enable_cloning = param.Boolean(
        default=True,
        doc="Enable cloning operator (fitness still computed for adaptive forces)",
    )
    enable_kinetic = param.Boolean(
        default=True,
        doc="Enable kinetic (Langevin dynamics) operator",
    )
    pbc = param.Boolean(
        default=False,
        doc=(
            "Use periodic boundary conditions. When enabled, walkers that move outside "
            "bounds are wrapped back instead of being marked as dead. Requires bounds to be set. "
            "PBC is applied before computing fitness and after kinetic updates."
        ),
    )
    neighbor_graph_method = param.Selector(
        default="delaunay",
        objects=["none", "delaunay", "voronoi"],
        doc=(
            "Neighbor graph backend (SciPy): 'delaunay' for simplicial neighbors, "
            "'voronoi' for Voronoi ridge neighbors, or 'none' to disable."
        ),
    )
    neighbor_graph_update_every = param.Integer(
        default=1, bounds=(1, None), doc="Recompute neighbor graph every k steps"
    )
    neighbor_graph_record = param.Boolean(
        default=True,
        doc="Record neighbor edges and Voronoi regions into RunHistory",
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
            "freeze_best": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Freeze best walker",
            },
            "enable_cloning": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Enable cloning",
            },
            "enable_kinetic": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Enable kinetic",
            },
            "pbc": {
                "type": pn.widgets.Checkbox,
                "width": INPUT_WIDTH,
                "name": "Periodic BC",
            },
        }

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI (excluding nested operators and internal objects)."""
        return ["N", "d", "dtype", "freeze_best", "enable_cloning", "enable_kinetic", "pbc"]

    def __init__(self, **params):
        """Initialize Euclidean Gas with post-initialization validation."""
        super().__init__(**params)

        # Post-initialization: auto-extract bounds from potential if available
        if (
            self.bounds is None
            and self.potential is not None
            and hasattr(self.potential, "bounds")
        ):
            self.bounds = self.potential.bounds

        # Validate PBC requirements
        if self.pbc and self.bounds is None:
            msg = "PBC mode requires bounds to be set"
            raise ValueError(msg)

        # Validate that potential is callable
        if self.potential is not None and not callable(self.potential):
            msg = f"potential must be callable, got {type(self.potential)}"
            raise TypeError(msg)

        # Validate that reward 1-form is callable
        if self.reward_1form is not None and not callable(self.reward_1form):
            msg = f"reward_1form must be callable, got {type(self.reward_1form)}"
            raise TypeError(msg)

        if self.companion_selection_clone is None:
            self.companion_selection_clone = self.companion_selection

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

    def _compute_rewards(self, state: SwarmState) -> Tensor:
        """Compute per-walker rewards from reward 1-form or potential."""
        if self.reward_1form is not None:
            reward_field = self.reward_1form(state.x)
            if reward_field.shape != state.x.shape:
                msg = (
                    "reward_1form must return shape [N, d] matching positions; "
                    f"got {tuple(reward_field.shape)} vs {tuple(state.x.shape)}"
                )
                raise ValueError(msg)
            return (reward_field * state.v).sum(dim=-1)

        if self.potential is None:
            msg = "potential required when reward_1form is None"
            raise ValueError(msg)
        return -self.potential(state.x)

    def _compute_neighbor_graph(
        self,
        x: Tensor,
        alive_mask: Tensor,
    ) -> dict[str, object]:
        """Compute Delaunay/Voronoi neighbor graph for alive walkers."""
        if self.neighbor_graph_method == "none":
            return {"neighbor_edges": None, "voronoi_regions": None, "updated": False}

        alive_idx = torch.where(alive_mask)[0]
        if alive_idx.numel() == 0:
            empty = torch.zeros((0, 2), dtype=torch.long, device=x.device)
            return {"neighbor_edges": empty, "voronoi_regions": None, "updated": True}

        if x.shape[1] < 2:
            empty = torch.zeros((0, 2), dtype=torch.long, device=x.device)
            return {"neighbor_edges": empty, "voronoi_regions": None, "updated": True}

        points = x[alive_idx].detach().cpu().numpy()
        try:
            from scipy.spatial import Delaunay, Voronoi
        except Exception as exc:
            msg = f"SciPy required for neighbor graph computation: {exc}"
            raise RuntimeError(msg) from exc

        edges: set[tuple[int, int]] = set()
        vor_data: dict[str, object] | None = None

        if self.neighbor_graph_method == "delaunay":
            try:
                delaunay = Delaunay(points)
                simplices = delaunay.simplices
                for simplex in simplices:
                    for i, j in itertools.combinations(simplex.tolist(), 2):
                        a = int(alive_idx[i].item())
                        b = int(alive_idx[j].item())
                        edges.add((a, b))
                        edges.add((b, a))
            except Exception:
                pass

            if self.neighbor_graph_record:
                try:
                    vor = Voronoi(points)
                except Exception as exc:
                    vor = None
                    vor_data = {"error": str(exc)}
                if vor is not None:
                    vertices = torch.as_tensor(
                        vor.vertices, device=x.device, dtype=x.dtype
                    )
                    point_region = vor.point_region.tolist()
                    regions = [vor.regions[idx] for idx in point_region]
                    regions_full: list[list[int] | None] = [None] * self.N
                    for local_idx, orig_idx in enumerate(alive_idx.tolist()):
                        regions_full[int(orig_idx)] = regions[local_idx]
                    vor_data = {
                        "vertices": vertices,
                        "regions": regions_full,
                        "point_region": point_region,
                        "alive_indices": alive_idx.tolist(),
                    }

        elif self.neighbor_graph_method == "voronoi":
            try:
                vor = Voronoi(points)
            except Exception as exc:
                vor = None
                vor_data = {"error": str(exc)}

            if vor is not None:
                ridge_points = vor.ridge_points
                for i, j in ridge_points.tolist():
                    a = int(alive_idx[i].item())
                    b = int(alive_idx[j].item())
                    edges.add((a, b))
                    edges.add((b, a))

                if self.neighbor_graph_record:
                    vertices = torch.as_tensor(
                        vor.vertices, device=x.device, dtype=x.dtype
                    )
                    point_region = vor.point_region.tolist()
                    regions = [vor.regions[idx] for idx in point_region]
                    regions_full = [None] * self.N
                    for local_idx, orig_idx in enumerate(alive_idx.tolist()):
                        regions_full[int(orig_idx)] = regions[local_idx]
                    vor_data = {
                        "vertices": vertices,
                        "regions": regions_full,
                        "point_region": point_region,
                        "alive_indices": alive_idx.tolist(),
                    }

        if not edges:
            neighbor_edges = torch.zeros((0, 2), dtype=torch.long, device=x.device)
        else:
            edge_list = list(edges)
            neighbor_edges = torch.tensor(edge_list, dtype=torch.long, device=x.device)

        return {
            "neighbor_edges": neighbor_edges,
            "voronoi_regions": vor_data if self.neighbor_graph_record else None,
            "updated": True,
        }

    def _maybe_update_neighbor_cache(
        self,
        x: Tensor,
        alive_mask: Tensor,
        step_idx: int | None = None,
    ) -> dict[str, object] | None:
        """Update neighbor graph on schedule and cache results."""
        if self.neighbor_graph_method == "none":
            return None

        update_every = max(1, int(self.neighbor_graph_update_every))
        cache = getattr(self, "_neighbor_cache", None)
        if step_idx is None:
            step_idx = getattr(self, "_current_step", None)

        if cache is not None and step_idx is not None and (step_idx % update_every != 0):
            return {**cache, "updated": False}

        result = self._compute_neighbor_graph(x, alive_mask)
        self._neighbor_cache = result
        return result

    def step(
        self, state: SwarmState, return_info: bool = False
    ) -> tuple[SwarmState, SwarmState] | tuple[SwarmState, SwarmState, dict] | None:
        """
        Perform one full step: compute fitness, clone (optional), then kinetic (optional).

        Uses cloning.py functions directly to compute:
        1. Rewards from reward 1-form or potential
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
            - If kinetic_op.n_kinetic_steps > 1, the kinetic operator is applied
              repeatedly using the same fitness derivatives for each substep
        """
        # Apply PBC at start if enabled (ensures positions valid before computing fitness)
        if self.pbc and self.bounds is not None:
            state.x = self.bounds.apply_pbc_to_out_of_bounds(state.x)

        if self.potential is not None and hasattr(self.potential, "set_step_id"):
            self.potential.set_step_id(getattr(self, "_current_step", None))

        freeze_mask = self._freeze_mask(state)
        reference_state = state.clone() if freeze_mask is not None else None

        # Step 1: Compute rewards from reward 1-form or potential
        rewards = self._compute_rewards(state)  # [N]
        U_before = self.potential(state.x) if self.potential is not None else None

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
            # Step 3: Select companions using the cloning companion selection strategy
            clone_selector = (
                self.companion_selection_clone
                if self.companion_selection_clone is not None
                else self.companion_selection
            )
            companions_clone = clone_selector(
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
                "clone_jitter": torch.zeros_like(state.x),
                "clone_delta_x": torch.zeros_like(state.x),
                "clone_delta_v": torch.zeros_like(state.v),
            }

        neighbor_info = None
        neighbor_edges = None
        if self.neighbor_graph_method != "none":
            neighbor_info = self._maybe_update_neighbor_cache(state_cloned.x, alive_mask)
            if neighbor_info is not None:
                neighbor_edges = neighbor_info.get("neighbor_edges")

        # # Apply freeze mask if needed
        # if freeze_mask is not None and freeze_mask.any():
        #     state_cloned.copy_from(reference_state, freeze_mask)

        # Step 5: Compute fitness derivatives if needed for adaptive kinetics
        grad_fitness = None
        hess_fitness = None
        is_diagonal_hessian = False
        voronoi_data = None

        if self.fitness_op is not None and self.enable_kinetic:
            # Compute fitness gradient if needed for adaptive force or diffusion proxy
            needs_grad = self.kinetic_op.use_fitness_force or (
                self.kinetic_op.use_anisotropic_diffusion
                and getattr(self.kinetic_op, "diffusion_mode", "hessian") == "grad_proxy"
            )
            if needs_grad:
                grad_fitness = self.fitness_op.compute_gradient(
                    state_cloned.x, state_cloned.v, rewards, alive_mask, companions_distance
                )

            # Compute fitness Hessian if needed for anisotropic diffusion
            needs_hess = (
                self.kinetic_op.use_anisotropic_diffusion
                and getattr(self.kinetic_op, "diffusion_mode", "hessian") == "hessian"
            )
            if needs_hess:
                hess_fitness = self.fitness_op.compute_hessian(
                    state_cloned.x,
                    state_cloned.v,
                    rewards,
                    alive_mask,
                    companions_distance,
                    diagonal_only=self.kinetic_op.diagonal_diffusion,
                )
                is_diagonal_hessian = self.kinetic_op.diagonal_diffusion

            # Compute Voronoi data if needed for voronoi_proxy diffusion
            needs_voronoi = (
                self.kinetic_op.use_anisotropic_diffusion
                and getattr(self.kinetic_op, "diffusion_mode", "hessian") == "voronoi_proxy"
            )
            if needs_voronoi:
                try:
                    from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation

                    voronoi_data = compute_voronoi_tessellation(
                        positions=state_cloned.x,
                        alive=alive_mask,
                        bounds=self.bounds,
                        pbc=self.pbc,
                        pbc_mode="mirror" if self.pbc else "ignore",
                        exclude_boundary=not self.pbc,  # Exclude boundary only if not using PBC
                        boundary_tolerance=1e-6,
                        compute_curvature=False,  # Not needed for diffusion
                    )
                except Exception as e:
                    # Fall back to isotropic diffusion if Voronoi fails
                    import warnings
                    warnings.warn(
                        f"Voronoi tessellation failed: {e}. Falling back to isotropic diffusion.",
                        RuntimeWarning,
                    )
                    voronoi_data = None

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
                    return_info=True,
                )
                if self.pbc and self.bounds is not None:
                    state_final.x = self.bounds.apply_pbc_to_out_of_bounds(state_final.x)
                if freeze_mask is not None and freeze_mask.any():
                    state_final.copy_from(reference_state, freeze_mask)
        else:
            # Skip kinetic update, use cloned state as final
            state_final = state_cloned.clone()
            kinetic_info = {
                "force_stable": torch.zeros_like(state.v),
                "force_adapt": torch.zeros_like(state.v),
                "force_viscous": torch.zeros_like(state.v),
                "force_friction": torch.zeros_like(state.v),
                "force_total": torch.zeros_like(state.v),
                "noise": torch.zeros_like(state.v),
                "sigma_reg_diag": None,
                "sigma_reg_full": None,
            }

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

        U_after_clone = self.potential(state_cloned.x) if self.potential is not None else None

        # Apply PBC after kinetic update (wraps final positions back into bounds)
        if self.pbc and self.bounds is not None:
            state_final.x = self.bounds.apply_pbc_to_out_of_bounds(state_final.x)

        U_final = self.potential(state_final.x) if self.potential is not None else None

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
                    "enable_kinetic": self.enable_kinetic,
                    "pbc": self.pbc,
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
                    "viscous_neighbor_mode": self.kinetic_op.viscous_neighbor_mode,
                    "viscous_neighbor_weighting": self.kinetic_op.viscous_neighbor_weighting,
                    "viscous_neighbor_threshold": self.kinetic_op.viscous_neighbor_threshold,
                    "viscous_neighbor_penalty": self.kinetic_op.viscous_neighbor_penalty,
                    "viscous_degree_cap": self.kinetic_op.viscous_degree_cap,
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
            record_neighbors=self.neighbor_graph_record and self.neighbor_graph_method != "none",
            record_voronoi=self.neighbor_graph_record and self.neighbor_graph_method != "none",
        )
        self._neighbor_cache = None

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
        U_initial = self.potential(state.x) if self.potential is not None else None
        recorder.record_initial_state(state, n_alive, U_before=U_initial, U_final=U_initial)

        # Check if initially all dead
        if n_alive == 0:
            if progress_callback is not None:
                progress_callback(0, n_steps, 0.0)
            recorded_steps = recorded_steps[: recorder.recorded_idx]
            return recorder.build(
                n_steps=0,
                record_every=record_every,
                terminated_early=True,
                final_step=0,
                total_time=0.0,
                init_time=init_time,
                bounds=self.bounds,
                recorded_steps=recorded_steps,
                delta_t=self.kinetic_op.delta_t,
                pbc=self.pbc,
                params=_build_params(),
                rng_seed=seed,
                rng_state=_capture_rng_state(),
            )

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

            # Check if all walkers are currently dead BEFORE stepping (skip in PBC mode)
            if not self.pbc and self.bounds is not None:
                alive_mask = self.bounds.contains(state.x)
                n_alive = alive_mask.sum().item()
                if n_alive == 0:
                    terminated_early = True
                    final_step = t - 1
                    break

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
            bounds=self.bounds,
            recorded_steps=recorded_steps,
            delta_t=self.kinetic_op.delta_t,
            pbc=self.pbc,
            params=params,
            rng_seed=seed,
            rng_state=rng_state,
        )
