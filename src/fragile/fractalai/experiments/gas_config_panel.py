"""Modern parameter configuration dashboard using operator PanelModel interfaces.

This module provides a Panel-based dashboard that leverages the __panel__() methods
of EuclideanGas and its nested operators, replacing the manual GasConfig approach.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import panel as pn
import panel.widgets as pnw
import param
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.core.benchmarks import BENCHMARK_NAMES, prepare_benchmark_for_explorer
from fragile.fractalai.core.cloning import CloneOperator
from fragile.fractalai.core.companion_selection import CompanionSelection
from fragile.fractalai.core.euclidean_gas import EuclideanGas
from fragile.fractalai.core.fitness import FitnessOperator
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.core.kinetic_operator import KineticOperator


__all__ = ["GasConfigPanel"]


class GasConfigPanel(param.Parameterized):
    """Modern configuration dashboard using operator PanelModel interfaces.

    This class provides a Panel-based UI that uses the __panel__() methods from
    EuclideanGas and its nested operators (KineticOperator, CloneOperator, etc.)
    to create an organized accordion-based parameter dashboard.

    The UI organization matches GasConfig but leverages the operator's own widget
    definitions instead of manually duplicating parameters.

    Example:
        >>> config = GasConfigPanel(dims=2)
        >>> dashboard = config.panel()
        >>> dashboard.show()  # Interactive parameter selection
        >>> history = config.history  # Access result after running
    """

    # Spatial dimension selection
    spatial_dims = param.ObjectSelector(
        default=3,
        objects=[2, 3],
        doc="Number of spatial dimensions (Euclidean time will be added as extra dimension)",
    )

    # Benchmark selection
    benchmark_name = param.ObjectSelector(
        default="Mixture of Gaussians",
        objects=list(BENCHMARK_NAMES.keys()),
        doc="Select benchmark potential function",
    )
    n_gaussians = param.Integer(default=3, bounds=(1, 10), doc="Number of Gaussian modes (MoG)")
    benchmark_seed = param.Integer(default=42, bounds=(0, 9999), doc="Random seed (MoG)")
    n_atoms = param.Integer(default=10, bounds=(2, 30), doc="Number of atoms (Lennard-Jones)")

    # Riemannian Mix parameters
    riemannian_volume_weight = param.Number(
        default=1.0,
        bounds=(-10.0, 10.0),
        softbounds=(-2.0, 2.0),
        doc="Volume element weight for Riemannian Mix",
    )
    riemannian_ricci_weight = param.Number(
        default=1.0,
        bounds=(-10.0, 10.0),
        softbounds=(-2.0, 2.0),
        doc="Ricci scalar weight for Riemannian Mix",
    )

    # Simulation controls
    n_steps = param.Integer(
        default=240, bounds=(10, 10000), softbounds=(50, 1000), doc="Simulation steps"
    )
    record_every = param.Integer(
        default=1,
        bounds=(1, 1000),
        softbounds=(1, 200),
        doc="Record every k-th step (1=all steps)",
    )
    neighbor_graph_method = param.ObjectSelector(
        default="delaunay",
        objects=["none", "delaunay", "voronoi"],
        doc="Neighbor graph backend (SciPy) for true-neighbor coupling",
    )
    neighbor_graph_update_every = param.Integer(
        default=1,
        bounds=(1, 1000),
        softbounds=(1, 200),
        doc="Recompute neighbor graph every k steps",
    )
    neighbor_graph_record = param.Boolean(
        default=True,
        doc="Record neighbor edges + Voronoi regions into RunHistory",
    )
    hide_viscous_kernel_widgets = param.Boolean(
        default=False,
        doc="Hide viscous kernel-only widgets (deprecated).",
    )

    # Initialization controls
    init_offset = param.Number(default=0.0, bounds=(-6.0, 6.0), doc="Initial position offset")
    init_spread = param.Number(default=10.0, bounds=(0.0, 50.0), doc="Initial position spread")
    init_velocity_scale = param.Number(
        default=5.0,
        bounds=(0.0, None),
        softbounds=(0.01, 2.0),
        doc="Initial velocity scale",
    )
    bounds_extent = param.Number(default=3.0, bounds=(1, 12), doc="Spatial bounds half-width")

    # Benchmark visualization controls
    show_optimum = param.Boolean(default=True, doc="Show global optimum marker on benchmark plot")
    show_density = param.Boolean(default=True, doc="Show density heatmap on benchmark plot")
    show_contours = param.Boolean(default=True, doc="Show contour lines on benchmark plot")
    viz_n_cells = param.Integer(
        default=200, bounds=(50, 500), doc="Grid resolution for benchmark visualization"
    )

    def __init__(self, dims: int = 2, spatial_dims: int | None = None, **params):
        """Initialize GasConfigPanel.

        Args:
            dims: Spatial dimension (default: 2) - DEPRECATED, use spatial_dims instead
            spatial_dims: Number of spatial dimensions (2 or 3). If None, uses dims parameter.
            **params: Override default parameter values
        """
        # Handle backward compatibility: spatial_dims takes precedence over dims
        if spatial_dims is not None:
            dims = spatial_dims
        elif "spatial_dims" in params:
            dims = params["spatial_dims"]

        super().__init__(**params)
        # dims/spatial_dims are interpreted as spatial dimension count
        spatial_count = int(dims)
        # Sync the spatial_dims parameter
        if self.spatial_dims != spatial_count:
            self.spatial_dims = spatial_count
        # Total dimension includes Euclidean time
        self.dims = spatial_count + 1
        self.history: RunHistory | None = None

        # Create default operators with sensible defaults
        self._create_default_operators()

        # Create benchmark
        self._update_benchmark()

        # Watch for benchmark parameter changes
        self.param.watch(
            self._on_benchmark_change,
            ["benchmark_name", "n_gaussians", "benchmark_seed", "n_atoms",
             "riemannian_volume_weight", "riemannian_ricci_weight"],
        )

        # Watch for spatial_dims changes to update dims
        self.param.watch(self._on_spatial_dims_change, "spatial_dims")

        # Create UI components
        self.run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
        self.run_button.sizing_mode = "stretch_width"
        self.run_button.on_click(self._on_run_clicked)

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Widget overrides for special cases
        self._widget_overrides: dict[str, pn.widgets.Widget] = {
            "n_gaussians": pnw.EditableIntSlider(
                name="n_gaussians", start=1, end=10, value=self.n_gaussians, step=1
            ),
            "benchmark_seed": pnw.EditableIntSlider(
                name="benchmark_seed", start=0, end=9999, value=self.benchmark_seed, step=1
            ),
            "n_atoms": pnw.EditableIntSlider(
                name="n_atoms", start=2, end=30, value=self.n_atoms, step=1
            ),
            "n_steps": pnw.EditableIntSlider(
                name="n_steps", start=10, end=10000, value=self.n_steps, step=1
            ),
            "record_every": pnw.EditableIntSlider(
                name="record_every", start=1, end=1000, value=self.record_every, step=1
            ),
            "viz_n_cells": pnw.EditableIntSlider(
                name="viz_n_cells (resolution)", start=50, end=500, value=self.viz_n_cells, step=10
            ),
            "init_offset": pnw.EditableIntSlider(
                name="init_offset", start=-6, end=6, value=int(self.init_offset), step=1
            ),
            "init_spread": pnw.EditableIntSlider(
                name="init_spread", start=0, end=50, value=int(self.init_spread), step=1
            ),
            "init_velocity_scale": pnw.EditableIntSlider(
                name="init_velocity_scale", start=0, end=50, value=int(self.init_velocity_scale), step=1
            ),
            "riemannian_volume_weight": pnw.EditableFloatSlider(
                name="Volume Weight",
                start=-10.0,
                end=10.0,
                value=self.riemannian_volume_weight,
                step=0.1,
            ),
            "riemannian_ricci_weight": pnw.EditableFloatSlider(
                name="Ricci Weight",
                start=-10.0,
                end=10.0,
                value=self.riemannian_ricci_weight,
                step=0.1,
            ),
        }
        self._widget_links: set[str] = set()

        self.progress_label = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.progress_bar = pn.indicators.Progress(
            name="Simulation progress",
            value=0,
            max=max(1, int(self.n_steps)),
            bar_color="primary",
            sizing_mode="stretch_width",
        )

        self._progress_update_interval = 0.2
        self._progress_last_emit = 0.0
        self._simulation_thread: threading.Thread | None = None

        self.param.watch(self._on_n_steps_change, "n_steps")
        n_steps_widget = self._widget_overrides.get("n_steps")
        if n_steps_widget is not None:
            n_steps_widget.param.watch(
                lambda e: self._sync_progress_total(e.new),
                "value",
            )
        self._sync_progress_total(self.n_steps)

        # Callbacks for external listeners
        self._on_simulation_complete: list[Callable[[RunHistory], None]] = []
        self._on_benchmark_updated: list[Callable[[object, object, object], None]] = []

    @staticmethod
    def create_qft_config(
        spatial_dims: int = 3, bounds_extent: float = 3.0, dims: int | None = None
    ) -> GasConfigPanel:
        """Create GasConfigPanel with QFT calibration defaults.

        These parameters match the calibrated simulation from
        08_qft_calibration_notebook.ipynb.

        Args:
            spatial_dims: Number of spatial dimensions (2 or 3, default: 3)
            bounds_extent: Half-width of spatial domain (default: 3.0)
            dims: DEPRECATED - use spatial_dims instead

        Returns:
            GasConfigPanel configured with QFT parameters
        """
        # Backward compatibility
        if dims is not None:
            spatial_dims = dims
        config = GasConfigPanel(spatial_dims=spatial_dims)

        # Benchmark
        config.bounds_extent = bounds_extent
        config.benchmark_name = "Voronoi Ricci Scalar"

        # Simulation
        config.n_steps = 5000
        config.gas_params["N"] = 200
        config.gas_params["dtype"] = "float32"
        config.gas_params["pbc"] = False
        config.neighbor_graph_method = "delaunay"
        config.neighbor_graph_update_every = 1
        config.neighbor_graph_record = True

        # Initialization (match EuclideanGas defaults used in QFT calibration)
        config.init_offset = 0.0
        config.init_spread = 1.0
        config.init_velocity_scale = 1.0

        # Kinetic operator (Langevin + viscous coupling)
        config.kinetic_op.gamma = 1.0
        config.kinetic_op.beta = 1.0
        config.kinetic_op.delta_t = 0.1005
        config.kinetic_op.epsilon_F = 38.6373
        config.kinetic_op.use_fitness_force = False
        config.kinetic_op.use_potential_force = False
        config.kinetic_op.use_anisotropic_diffusion = True
        config.kinetic_op.diagonal_diffusion = False
        config.kinetic_op.nu = 1.10
        config.kinetic_op.use_viscous_coupling = True
        config.kinetic_op.viscous_length_scale = 0.251372
        config.kinetic_op.viscous_neighbor_mode = "nearest"
        config.kinetic_op.viscous_neighbor_weighting = "geodesic"
        config.kinetic_op.viscous_neighbor_threshold = 0.75
        config.kinetic_op.viscous_neighbor_penalty = 1.1

        # Companion selection (separate epsilon for diversity and cloning)
        config.companion_selection.method = "uniform"
        config.companion_selection.epsilon = 2.80  # epsilon_d
        config.companion_selection.lambda_alg = 1.0
        config.companion_selection.exclude_self = True
        config.companion_selection_clone.method = "uniform"
        config.companion_selection_clone.epsilon = 1.68419  # epsilon_clone
        config.companion_selection_clone.lambda_alg = 1.0
        config.companion_selection_clone.exclude_self = True

        # Cloning operator
        config.cloning.p_max = 1.0
        config.cloning.epsilon_clone = 0.01
        config.cloning.sigma_x = 0.1
        config.cloning.alpha_restitution = 0.5

        # Fitness operator
        config.fitness_op.alpha = 1.0
        config.fitness_op.beta = 1.0
        config.fitness_op.eta = 0.1
        config.fitness_op.lambda_alg = 1.0
        config.fitness_op.sigma_min = 1e-8
        config.fitness_op.epsilon_dist = 1e-8
        config.fitness_op.A = 2.0
        # Tuned for improved electroweak probe ratios (Jan 2026 calibration sweep).
        config.fitness_op.rho = 0.1

        return config

    # Backward compatibility properties for components expecting old GasConfig interface
    @property
    def gamma(self):
        """Backward compatibility: delegate to kinetic_op.gamma"""
        return self.kinetic_op.gamma

    @property
    def beta(self):
        """Backward compatibility: delegate to kinetic_op.beta"""
        return self.kinetic_op.beta

    @property
    def delta_t(self):
        """Backward compatibility: delegate to kinetic_op.delta_t"""
        return self.kinetic_op.delta_t

    @property
    def epsilon_F(self):
        """Backward compatibility: delegate to kinetic_op.epsilon_F"""
        return self.kinetic_op.epsilon_F

    @property
    def use_fitness_force(self):
        """Backward compatibility: delegate to kinetic_op.use_fitness_force"""
        return self.kinetic_op.use_fitness_force

    @property
    def use_potential_force(self):
        """Backward compatibility: delegate to kinetic_op.use_potential_force"""
        return self.kinetic_op.use_potential_force

    @property
    def lambda_alg(self):
        """Backward compatibility: delegate to fitness_op.lambda_alg"""
        return self.fitness_op.lambda_alg

    @property
    def N(self):
        """Backward compatibility: delegate to gas_params['N']"""
        return self.gas_params["N"]

    @property
    def pbc(self):
        """Backward compatibility: delegate to gas_params['pbc']"""
        return self.gas_params["pbc"]

    def _create_default_operators(self):
        """Create default operator instances with sensible defaults for multimodal exploration."""
        # Companion selection
        self.companion_selection = CompanionSelection(
            method="uniform",
            epsilon=0.5,
            lambda_alg=0.2,
        )

        # Companion selection for cloning (separate instance, allows different epsilon)
        self.companion_selection_clone = CompanionSelection(
            method="uniform",
            epsilon=0.5,
            lambda_alg=0.2,
        )

        # Kinetic operator (Langevin dynamics)
        self.kinetic_op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.05,
            integrator="boris-baoab",
            epsilon_F=0.15,
            use_fitness_force=False,
            use_potential_force=False,
            epsilon_Sigma=0.1,
            use_anisotropic_diffusion=True,
            diagonal_diffusion=False,
            nu=0.1,
            use_viscous_coupling=True,
            viscous_length_scale=1.0,
            viscous_neighbor_mode="nearest",
            viscous_neighbor_weighting="geodesic",
            viscous_neighbor_penalty=1.1,
            V_alg=10.0,
            use_velocity_squashing=False,
        )

        # Cloning operator
        self.cloning = CloneOperator(
            sigma_x=0.5,
            alpha_restitution=0.6,
            p_max=1.0,
            epsilon_clone=0.005,
        )

        # Fitness operator (tuned for multimodal: β >> α for diversity)
        self.fitness_op = FitnessOperator(
            alpha=0.4,  # Reward exponent
            beta=2.5,  # Diversity exponent (higher promotes exploration)
            eta=0.003,  # Positivity floor
            lambda_alg=0.2,  # Velocity weight
            sigma_min=1e-8,  # Standardization regularization
            A=3.5,  # Logistic rescale amplitude
        )

        # EuclideanGas top-level params
        self.gas_params = {
            "N": 160,
            "d": self.dims,
            "freeze_best": False,
            "enable_cloning": True,
            "clone_every": 1,
            "enable_kinetic": True,
            "pbc": False,
            "dtype": "float32",
        }

    def add_completion_callback(self, callback: Callable[[RunHistory], None]):
        """Register a callback to be called when simulation completes.

        Args:
            callback: Function that takes RunHistory as argument
        """
        self._on_simulation_complete.append(callback)

    def add_benchmark_callback(self, callback: Callable[[object, object, object], None]):
        """Register a callback to be called when benchmark updates.

        Args:
            callback: Function that takes (potential, background, mode_points) as arguments
        """
        self._on_benchmark_updated.append(callback)

    def _update_benchmark(self):
        """Create benchmark from current benchmark parameters."""
        bounds_range = (-float(self.bounds_extent), float(self.bounds_extent))

        # Prepare benchmark-specific kwargs
        benchmark_kwargs = {}
        if self.benchmark_name == "Mixture of Gaussians":
            benchmark_kwargs["n_gaussians"] = self.n_gaussians
            benchmark_kwargs["seed"] = self.benchmark_seed
        elif self.benchmark_name == "Lennard-Jones":
            benchmark_kwargs["n_atoms"] = self.n_atoms
        elif self.benchmark_name == "Riemannian Mix":
            benchmark_kwargs["volume_weight"] = self.riemannian_volume_weight
            benchmark_kwargs["ricci_weight"] = self.riemannian_ricci_weight

        # Create benchmark with background and mode_points
        benchmark, background, mode_points = prepare_benchmark_for_explorer(
            benchmark_name=self.benchmark_name,
            dims=self.dims,
            bounds_range=bounds_range,
            resolution=100,
            **benchmark_kwargs,
        )

        # Store all benchmark components
        self.potential = benchmark
        self.background = background
        self.mode_points = mode_points

    def _on_benchmark_change(self, *_):
        """Handle benchmark parameter changes."""
        self._update_benchmark()
        if self.benchmark_name == "Voronoi Cell Volume":
            self._apply_voronoi_volume_preset()
        elif self.benchmark_name == "Riemannian Mix":
            self._apply_riemannian_mix_preset()
        else:
            self.status_pane.object = f"**Benchmark updated:** {self.benchmark_name}"

        # Notify listeners
        for callback in self._on_benchmark_updated:
            callback(self.potential, self.background, self.mode_points)

    def _on_spatial_dims_change(self, event):
        """Handle spatial dimensions parameter changes."""
        new_spatial_dims = event.new
        self.dims = int(new_spatial_dims) + 1
        # Update benchmark with new dimensions
        self._update_benchmark()
        self.status_pane.object = (
            f"**Spatial dimensions updated:** {new_spatial_dims}D spatial + 1D time "
            f"= {new_spatial_dims + 1}D total"
        )

    def _apply_fast_fitness_preset(self, *_):
        """Apply fast fitness-force settings (approximate gradients)."""
        self.kinetic_op.use_fitness_force = True
        self.kinetic_op.use_anisotropic_diffusion = False
        self.kinetic_op.diagonal_diffusion = False
        self.kinetic_op.epsilon_F = 5.0
        self.fitness_op.grad_mode = "sum"
        self.fitness_op.detach_stats = True
        self.fitness_op.detach_companions = True
        self.fitness_op.sigma_min = 1e-1
        self.status_pane.object = "**Applied fast fitness preset (approximate gradients).**"

    def _apply_voronoi_volume_preset(self) -> None:
        """Apply Voronoi volume benchmark preset settings."""
        self.neighbor_graph_method = "voronoi"
        self.neighbor_graph_update_every = 1
        self.neighbor_graph_record = True
        self.record_every = 1
        self.kinetic_op.use_viscous_coupling = True
        self.kinetic_op.viscous_neighbor_weighting = "uniform"
        self.kinetic_op.use_potential_force = False
        self.kinetic_op.use_fitness_force = False
        self.kinetic_op.use_anisotropic_diffusion = True
        self.kinetic_op.diagonal_diffusion = False
        self.fitness_op.grad_mode = "sum"
        self.fitness_op.detach_stats = True
        self.fitness_op.detach_companions = True
        self.status_pane.object = (
            "**Applied Voronoi cell volume preset (record every step, "
            "use Voronoi neighbors for viscous coupling).**"
        )

    def _apply_riemannian_mix_preset(self) -> None:
        """Apply Riemannian Mix benchmark preset settings."""
        self.neighbor_graph_method = "delaunay"
        self.neighbor_graph_update_every = 1
        self.neighbor_graph_record = True
        self.record_every = 1
        self.kinetic_op.use_viscous_coupling = True
        self.status_pane.object = (
            "**Applied Riemannian Mix preset (Delaunay neighbors, "
            "record every step).**"
        )

    def _format_eta(self, seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        seconds = max(0.0, seconds)
        total_seconds = int(round(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    def _progress_text(self, step: int, total_steps: int, eta_seconds: float | None) -> str:
        total_steps = max(1, int(total_steps))
        step = max(0, min(int(step), total_steps))
        percent = (step / total_steps) * 100.0 if total_steps else 0.0
        eta_str = self._format_eta(eta_seconds)
        return (
            f"**Progress:** {step}/{total_steps} ({percent:.1f}%) "
            f"| **Remaining:** {eta_str}"
        )

    def _sync_progress_total(self, total_steps: int) -> None:
        total_steps = max(1, int(total_steps))
        current = min(int(self.progress_bar.value), total_steps)
        self.progress_bar.max = total_steps
        self.progress_bar.value = current
        self.progress_label.object = self._progress_text(current, total_steps, None)

    def _on_n_steps_change(self, event) -> None:
        self._sync_progress_total(event.new)

    def _schedule_ui_update(self, callback: Callable[[], None]) -> None:
        doc = pn.state.curdoc
        if doc is None:
            callback()
        else:
            doc.add_next_tick_callback(callback)

    def _update_progress_display(
        self, step: int, total_steps: int, eta_seconds: float | None
    ) -> None:
        total_steps = max(1, int(total_steps))
        step = max(0, min(int(step), total_steps))
        self.progress_bar.max = total_steps
        self.progress_bar.value = step
        self.progress_label.object = self._progress_text(step, total_steps, eta_seconds)

    def _progress_callback(self, step: int, total_steps: int, elapsed: float) -> None:
        now = time.perf_counter()
        if step < total_steps and (
            now - self._progress_last_emit
        ) < self._progress_update_interval:
            return
        self._progress_last_emit = now
        eta = None
        if step > 0 and total_steps > step:
            eta = (elapsed / step) * (total_steps - step)
        self._schedule_ui_update(
            lambda: self._update_progress_display(step, total_steps, eta)
        )

    def _current_steps_value(self) -> int:
        widget = self._widget_overrides.get("n_steps")
        if widget is not None and hasattr(widget, "value"):
            return int(widget.value)
        return int(self.n_steps)

    def _run_simulation_worker(self) -> None:
        history: RunHistory | None = None
        error: Exception | None = None
        try:
            history = self.run_simulation(progress_callback=self._progress_callback)
        except Exception as exc:
            error = exc

        def _finalize() -> None:
            self.run_button.disabled = False
            if error is not None:
                self.status_pane.object = f"**Error:** {error!s}"
                self.progress_bar.bar_color = "danger"
                return

            if history is None:
                self.status_pane.object = "**Error:** simulation failed without history."
                self.progress_bar.bar_color = "danger"
                return

            self.status_pane.object = (
                f"**Simulation complete!** {history.n_steps} steps, "
                f"{history.n_recorded} recorded timesteps"
            )
            self.progress_bar.bar_color = (
                "warning" if history.terminated_early else "success"
            )
            self._update_progress_display(history.final_step, self.progress_bar.max, 0.0)

        self._schedule_ui_update(_finalize)

    def _on_run_clicked(self, *_):
        """Handle Run button click."""
        if self._simulation_thread is not None and self._simulation_thread.is_alive():
            return

        total_steps = self._current_steps_value()
        self._sync_progress_total(total_steps)
        self._update_progress_display(0, self.progress_bar.max, None)
        self.progress_bar.bar_color = "primary"
        self.status_pane.object = "**Running simulation...**"
        self.run_button.disabled = True
        self._progress_last_emit = 0.0

        self._simulation_thread = threading.Thread(
            target=self._run_simulation_worker,
            name="gas-simulation-thread",
            daemon=True,
        )
        self._simulation_thread.start()

    def run_simulation(
        self,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> RunHistory:
        """Run EuclideanGas simulation with current parameters.

        Returns:
            RunHistory object containing complete execution trace

        Raises:
            ValueError: If parameters are invalid
        """
        for name in ("n_steps", "record_every"):
            widget = self._widget_overrides.get(name)
            if widget is not None and hasattr(widget, "value"):
                setattr(self, name, widget.value)

        # Create bounds
        bounds_extent = float(self.bounds_extent)
        low = torch.full((self.dims,), -bounds_extent, dtype=torch.float32)
        high = torch.full((self.dims,), bounds_extent, dtype=torch.float32)
        bounds = TorchBounds(low=low, high=high)

        # Update kinetic operator's potential and bounds
        self.kinetic_op.potential = self.potential
        self.kinetic_op.bounds = bounds
        self.kinetic_op.pbc = self.gas_params["pbc"]

        # Create EuclideanGas using current operator instances
        gas = EuclideanGas(
            N=int(self.gas_params["N"]),
            d=self.dims,
            companion_selection=self.companion_selection,
            companion_selection_clone=self.companion_selection_clone,
            potential=self.potential,
            kinetic_op=self.kinetic_op,
            cloning=self.cloning,
            fitness_op=self.fitness_op,
            bounds=bounds,
            device=torch.device("cpu"),
            dtype=self.gas_params["dtype"],
            freeze_best=self.gas_params["freeze_best"],
            enable_cloning=self.gas_params["enable_cloning"],
            clone_every=int(self.gas_params.get("clone_every", 1)),
            enable_kinetic=self.gas_params["enable_kinetic"],
            pbc=self.gas_params["pbc"],
            neighbor_graph_method=self.neighbor_graph_method,
            neighbor_graph_update_every=int(self.neighbor_graph_update_every),
            neighbor_graph_record=self.neighbor_graph_record,
        )

        # Initialize state
        offset = torch.full((self.dims,), float(self.init_offset), dtype=torch.float32)
        x_init = torch.randn(self.gas_params["N"], self.dims) * float(self.init_spread) + offset
        x_init = torch.clamp(x_init, min=low, max=high)
        v_init = torch.randn(self.gas_params["N"], self.dims) * float(self.init_velocity_scale)

        # Run simulation
        history = gas.run(
            self.n_steps,
            x_init=x_init,
            v_init=v_init,
            record_every=int(self.record_every),
            progress_callback=progress_callback,
        )

        # Store history and notify listeners
        self.history = history
        for callback in self._on_simulation_complete:
            self._schedule_ui_update(lambda cb=callback, hist=self.history: cb(hist))

        return history

    def _build_param_panel(self, names: list[str]) -> pn.Param:
        """Build parameter panel with custom widgets."""
        widgets = {
            name: self._widget_overrides[name] for name in names if name in self._widget_overrides
        }
        def _coerce_widget_value(widget_ref: pn.widgets.Widget, value):
            if isinstance(widget_ref, pnw.EditableIntSlider):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return value
            return value

        for name, widget in widgets.items():
            if hasattr(widget, "value"):
                widget.value = _coerce_widget_value(widget, getattr(self, name))
                if name not in self._widget_links:
                    widget.param.watch(
                        lambda e, param_name=name: setattr(self, param_name, e.new),
                        "value",
                    )
                    self.param.watch(
                        lambda e, widget_ref=widget: (
                            None
                            if getattr(widget_ref, "value", None) == e.new
                            else setattr(widget_ref, "value", _coerce_widget_value(widget_ref, e.new))
                        ),
                        name,
                    )
                    self._widget_links.add(name)
        return pn.Param(
            self.param,
            parameters=names,
            widgets=widgets,
            show_name=False,
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.Column:
        """Create Panel dashboard using operator __panel__() methods.

        Returns:
            Panel Column with organized parameter sections and Run button
        """
        # === Benchmark Panel ===
        benchmark_params_base = ["benchmark_name"]

        def get_benchmark_specific_params(benchmark_name):
            """Return parameter panel for benchmark-specific settings."""
            if benchmark_name == "Mixture of Gaussians":
                return self._build_param_panel(["n_gaussians", "benchmark_seed"])
            if benchmark_name == "Lennard-Jones":
                return self._build_param_panel(["n_atoms"])
            if benchmark_name == "Riemannian Mix":
                return self._build_param_panel(["riemannian_volume_weight", "riemannian_ricci_weight"])
            return pn.pane.Markdown("*No additional parameters*", sizing_mode="stretch_width")

        benchmark_specific = pn.bind(get_benchmark_specific_params, self.param.benchmark_name)

        # Visualization controls
        viz_controls = pn.Column(
            pn.pane.Markdown("#### Visualization Options"),
            self._build_param_panel([
                "show_optimum",
                "show_density",
                "show_contours",
                "viz_n_cells",
            ]),
            sizing_mode="stretch_width",
        )

        # Spatial dimensions selector (added before benchmark)
        spatial_dims_description = pn.pane.Markdown(
            """
**Spatial Dimensions**: Choose 2D or 3D spatial configuration.
Euclidean time will be added as an additional dimension for QFT analysis.
- **2D spatial + time** = 3D total
- **3D spatial + time** = 4D total (default)

*Note: 1D spatial is not supported due to Voronoi tessellation requirements.*
            """,
            sizing_mode="stretch_width",
        )

        benchmark_panel = pn.Column(
            pn.pane.Markdown("### Spatial Dimensions"),
            spatial_dims_description,
            self._build_param_panel(["spatial_dims"]),
            pn.layout.Divider(),
            pn.pane.Markdown("### Potential Function"),
            self._build_param_panel(benchmark_params_base),
            benchmark_specific,
            viz_controls,
            sizing_mode="stretch_width",
        )

        # === General Panel ===
        # Create widgets first (separate from watchers to avoid rendering Watcher objects)
        n_slider = pn.widgets.EditableIntSlider(
            name="N (walkers)",
            value=self.gas_params["N"],
            start=2,
            end=10000,
            step=1,
        )
        enable_cloning_cb = pn.widgets.Checkbox(
            name="Enable cloning",
            value=self.gas_params["enable_cloning"],
        )
        enable_kinetic_cb = pn.widgets.Checkbox(
            name="Enable kinetic",
            value=self.gas_params["enable_kinetic"],
        )
        pbc_cb = pn.widgets.Checkbox(
            name="PBC (periodic bounds)",
            value=self.gas_params["pbc"],
        )

        # Set up watchers separately (watch() returns Watcher, not widget)
        n_slider.param.watch(lambda e: self.gas_params.update({"N": e.new}), "value")
        enable_cloning_cb.param.watch(
            lambda e: self.gas_params.update({"enable_cloning": e.new}), "value"
        )
        enable_kinetic_cb.param.watch(
            lambda e: self.gas_params.update({"enable_kinetic": e.new}), "value"
        )
        pbc_cb.param.watch(lambda e: self.gas_params.update({"pbc": e.new}), "value")

        # Add widgets to column
        neighbor_params = [
            "neighbor_graph_method",
            "neighbor_graph_update_every",
            "neighbor_graph_record",
        ]
        general_panel = pn.Column(
            n_slider,
            self._build_param_panel(["n_steps", "record_every"]),
            enable_cloning_cb,
            enable_kinetic_cb,
            pbc_cb,
            self._build_param_panel(neighbor_params),
            sizing_mode="stretch_width",
        )

        # === Operator Panels using __panel__() methods ===
        langevin_params = list(self.kinetic_op.widget_parameters)
        kinetic_widgets = {
            name: (dict(cfg) if isinstance(cfg, dict) else cfg)
            for name, cfg in self.kinetic_op.widgets.items()
        }
        if self.hide_viscous_kernel_widgets:
            weighting_widget = kinetic_widgets.get("viscous_neighbor_weighting")
            if isinstance(weighting_widget, dict):
                options = [opt for opt in weighting_widget.get("options", []) if opt != "kernel"]
                if options:
                    weighting_widget["options"] = options
                    if self.kinetic_op.viscous_neighbor_weighting not in options:
                        self.kinetic_op.viscous_neighbor_weighting = options[0]
            hidden_params = {
                "viscous_length_scale",
                "viscous_neighbor_penalty",
                "viscous_neighbor_threshold",
                "viscous_degree_cap",
            }
            langevin_params = [name for name in langevin_params if name not in hidden_params]
            for name in hidden_params:
                kinetic_widgets.pop(name, None)
        langevin_panel = pn.Param(
            self.kinetic_op,
            show_name=False,
            parameters=langevin_params,
            widgets=self.kinetic_op.process_widgets(kinetic_widgets),
            default_layout=self.kinetic_op.default_layout,
        )
        fast_fitness_button = pn.widgets.Button(
            name="Apply fast fitness preset",
            button_type="primary",
        )
        fast_fitness_button.on_click(self._apply_fast_fitness_preset)
        clone_every_input = pn.widgets.IntInput(
            name="Clone every (steps)",
            value=int(self.gas_params.get("clone_every", 1)),
            start=1,
            end=10000,
            step=1,
        )
        def _set_clone_every(event):
            try:
                value = int(event.new)
            except (TypeError, ValueError):
                value = 1
            value = max(1, value)
            self.gas_params["clone_every"] = value
            if getattr(clone_every_input, "value", None) != value:
                clone_every_input.value = value

        clone_every_input.param.watch(_set_clone_every, "value")
        cloning_panel_combined = pn.Column(
            pn.pane.Markdown("#### Cloning Operator"),
            clone_every_input,
            self.cloning.__panel__(),
            pn.pane.Markdown("#### Fitness Potential"),
            fast_fitness_button,
            self.fitness_op.__panel__(),
            pn.pane.Markdown("#### Companion Selection (distance)"),
            self.companion_selection.__panel__(),
            pn.pane.Markdown("#### Companion Selection (clone)"),
            self.companion_selection_clone.__panel__(),
            sizing_mode="stretch_width",
        )

        # === Initialization Panel ===
        init_panel = self._build_param_panel([
            "init_offset",
            "init_spread",
            "init_velocity_scale",
            "bounds_extent",
        ])

        # === Accordion Organization ===
        accordion = pn.Accordion(
            ("Benchmark", benchmark_panel),
            ("General", general_panel),
            ("Langevin Dynamics", langevin_panel),
            ("Cloning & Fitness", cloning_panel_combined),
            ("Initialization", init_panel),
            sizing_mode="stretch_width",
        )
        # Open benchmark and general sections by default
        accordion.active = [0, 1]

        return pn.Column(
            pn.pane.Markdown("## Simulation Parameters"),
            accordion,
            self.progress_label,
            self.progress_bar,
            self.run_button,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=380,
        )
