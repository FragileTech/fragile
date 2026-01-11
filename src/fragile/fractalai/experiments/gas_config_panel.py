"""Modern parameter configuration dashboard using operator PanelModel interfaces.

This module provides a Panel-based dashboard that leverages the __panel__() methods
of EuclideanGas and its nested operators, replacing the manual GasConfig approach.
"""

from __future__ import annotations

from typing import Callable

import panel as pn
import panel.widgets as pnw
import param
import torch

from fragile.bounds import TorchBounds
from fragile.core.benchmarks import BENCHMARK_NAMES, prepare_benchmark_for_explorer
from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import EuclideanGas
from fragile.core.fitness import FitnessOperator
from fragile.core.history import RunHistory
from fragile.core.kinetic_operator import KineticOperator


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

    # Benchmark selection
    benchmark_name = param.ObjectSelector(
        default="Mixture of Gaussians",
        objects=list(BENCHMARK_NAMES.keys()),
        doc="Select benchmark potential function",
    )
    n_gaussians = param.Integer(default=3, bounds=(1, 10), doc="Number of Gaussian modes (MoG)")
    benchmark_seed = param.Integer(default=42, bounds=(0, 9999), doc="Random seed (MoG)")
    n_atoms = param.Integer(default=10, bounds=(2, 30), doc="Number of atoms (Lennard-Jones)")

    # Simulation controls
    n_steps = param.Integer(
        default=240, bounds=(10, 10000), softbounds=(50, 1000), doc="Simulation steps"
    )

    # Initialization controls
    init_offset = param.Number(default=4.5, bounds=(-6.0, 6.0), doc="Initial position offset")
    init_spread = param.Number(default=0.5, bounds=(0.1, 3.0), doc="Initial position spread")
    init_velocity_scale = param.Number(
        default=0.1, bounds=(0.01, 0.8), doc="Initial velocity scale"
    )
    bounds_extent = param.Number(default=6.0, bounds=(1, 12), doc="Spatial bounds half-width")

    # Benchmark visualization controls
    show_optimum = param.Boolean(default=True, doc="Show global optimum marker on benchmark plot")
    show_density = param.Boolean(default=True, doc="Show density heatmap on benchmark plot")
    show_contours = param.Boolean(default=True, doc="Show contour lines on benchmark plot")
    viz_n_cells = param.Integer(
        default=200, bounds=(50, 500), doc="Grid resolution for benchmark visualization"
    )

    def __init__(self, dims: int = 2, **params):
        """Initialize GasConfigPanel.

        Args:
            dims: Spatial dimension (default: 2)
            **params: Override default parameter values
        """
        super().__init__(**params)
        self.dims = dims
        self.history: RunHistory | None = None

        # Create default operators with sensible defaults
        self._create_default_operators()

        # Create benchmark
        self._update_benchmark()

        # Watch for benchmark parameter changes
        self.param.watch(
            self._on_benchmark_change,
            ["benchmark_name", "n_gaussians", "benchmark_seed", "n_atoms"],
        )

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
            "viz_n_cells": pnw.EditableIntSlider(
                name="viz_n_cells (resolution)", start=50, end=500, value=self.viz_n_cells, step=10
            ),
        }

        # Callbacks for external listeners
        self._on_simulation_complete: list[Callable[[RunHistory], None]] = []
        self._on_benchmark_updated: list[Callable[[object, object, object], None]] = []

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

        # Kinetic operator (Langevin dynamics)
        self.kinetic_op = KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.05,
            integrator="baoab",
            epsilon_F=0.15,
            use_fitness_force=False,
            use_potential_force=False,
            epsilon_Sigma=0.1,
            use_anisotropic_diffusion=False,
            diagonal_diffusion=True,
            nu=0.0,
            use_viscous_coupling=False,
            viscous_length_scale=1.0,
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
        self.status_pane.object = f"**Benchmark updated:** {self.benchmark_name}"

        # Notify listeners
        for callback in self._on_benchmark_updated:
            callback(self.potential, self.background, self.mode_points)

    def _on_run_clicked(self, *_):
        """Handle Run button click."""
        self.status_pane.object = "**Running simulation...**"
        self.run_button.disabled = True

        try:
            self.run_simulation()
            self.status_pane.object = (
                f"**Simulation complete!** "
                f"{self.history.n_steps} steps, "
                f"{self.history.n_recorded} recorded timesteps"
            )

        except Exception as e:
            self.status_pane.object = f"**Error:** {e!s}"
        finally:
            self.run_button.disabled = False

    def run_simulation(self) -> RunHistory:
        """Run EuclideanGas simulation with current parameters.

        Returns:
            RunHistory object containing complete execution trace

        Raises:
            ValueError: If parameters are invalid
        """
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
            potential=self.potential,
            kinetic_op=self.kinetic_op,
            cloning=self.cloning,
            fitness_op=self.fitness_op,
            bounds=bounds,
            device=torch.device("cpu"),
            dtype=self.gas_params["dtype"],
            freeze_best=self.gas_params["freeze_best"],
            enable_cloning=self.gas_params["enable_cloning"],
            enable_kinetic=self.gas_params["enable_kinetic"],
            pbc=self.gas_params["pbc"],
        )

        # Initialize state
        offset = torch.full((self.dims,), float(self.init_offset), dtype=torch.float32)
        x_init = torch.randn(self.gas_params["N"], self.dims) * float(self.init_spread) + offset
        x_init = torch.clamp(x_init, min=low, max=high)
        v_init = torch.randn(self.gas_params["N"], self.dims) * float(self.init_velocity_scale)

        # Run simulation
        history = gas.run(self.n_steps, x_init=x_init, v_init=v_init)

        # Store history and notify listeners
        self.history = history
        for callback in self._on_simulation_complete:
            callback(self.history)

        return history

    def _build_param_panel(self, names: list[str]) -> pn.Param:
        """Build parameter panel with custom widgets."""
        widgets = {
            name: self._widget_overrides[name] for name in names if name in self._widget_overrides
        }
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

        benchmark_panel = pn.Column(
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
        general_panel = pn.Column(
            n_slider,
            self._build_param_panel(["n_steps"]),
            enable_cloning_cb,
            enable_kinetic_cb,
            pbc_cb,
            sizing_mode="stretch_width",
        )

        # === Operator Panels using __panel__() methods ===
        langevin_panel = self.kinetic_op.__panel__()
        cloning_panel_combined = pn.Column(
            pn.pane.Markdown("#### Cloning Operator"),
            self.cloning.__panel__(),
            pn.pane.Markdown("#### Fitness Potential"),
            self.fitness_op.__panel__(),
            pn.pane.Markdown("#### Companion Selection"),
            self.companion_selection.__panel__(),
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
            self.run_button,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=380,
        )
