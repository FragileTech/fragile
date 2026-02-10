"""Reusable parameter configuration dashboard for EuclideanGas simulations.

DEPRECATED: This module is deprecated in favor of gas_config_panel.GasConfigPanel,
which uses the operator PanelModel interfaces instead of manually duplicating parameters.

New code should use:
    from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

This module provides a Panel-based dashboard for configuring simulation parameters
and running EuclideanGas simulations. It returns RunHistory objects that can be
visualized or analyzed separately.
"""

from __future__ import annotations

from typing import Callable, Iterable
import warnings

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


__all__ = ["GasConfig"]


class GasConfig(param.Parameterized):
    """DEPRECATED: Use gas_config_panel.GasConfigPanel instead.

    This class manually duplicates operator parameters. The new GasConfigPanel
    uses operator.__panel__() methods for better maintainability and consistency.
    """

    # Note: __init__ defined below includes deprecation warning

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
    N = param.Integer(default=160, bounds=(2, 10000), doc="Number of walkers")
    n_steps = param.Integer(
        default=240, bounds=(10, 10000), softbounds=(50, 1000), doc="Simulation steps"
    )

    # Langevin parameters
    gamma = param.Number(
        default=1.0, bounds=(0.0001, 10.0), softbounds=(0.05, 5.0), doc="Friction γ"
    )
    beta = param.Number(
        default=1.0, bounds=(0.0001, 10000.0), softbounds=(0.01, 10.0), doc="Inverse temperature β"
    )
    delta_t = param.Number(
        default=0.05, bounds=(0.000001, 1.0), softbounds=(0.01, 0.1), doc="Time step Δt"
    )
    epsilon_F = param.Number(
        default=0.15,
        bounds=(0.0, 2.0),
        softbounds=(0.0, 0.5),
        doc="Fitness force rate ε_F (for active exploration)",
    )
    use_fitness_force = param.Boolean(default=False, doc="Enable fitness-driven force")
    use_potential_force = param.Boolean(default=False, doc="Enable potential force")
    epsilon_Sigma = param.Number(default=0.1, bounds=(0.0, 1.0), doc="Hessian regularisation ε_Σ")
    use_anisotropic_diffusion = param.Boolean(default=False, doc="Enable anisotropic diffusion")
    diagonal_diffusion = param.Boolean(default=True, doc="Use diagonal diffusion tensor")
    nu = param.Number(default=0.0, bounds=(0.0, 10.0), doc="Viscous coupling strength ν")
    use_viscous_coupling = param.Boolean(default=False, doc="Enable viscous coupling")
    viscous_length_scale = param.Number(
        default=1.0, bounds=(0.1, 5.0), doc="Viscous kernel length scale l"
    )
    V_alg = param.Number(default=10.0, bounds=(0.1, 100.0), doc="Algorithmic velocity bound V_alg")
    use_velocity_squashing = param.Boolean(default=False, doc="Enable velocity squashing map ψ_v")

    # Cloning parameters (defaults tuned for multimodal exploration: β >> α for diversity)
    sigma_x = param.Number(
        default=0.5, bounds=(0.00000001, 10.0), softbounds=(0.01, 1.0), doc="Cloning jitter σ_x"
    )
    lambda_alg = param.Number(
        default=0.2,
        bounds=(0.0, 10.0),
        softbounds=(0.0, 1.0),
        doc="Algorithmic distance weight λ_alg",
    )
    alpha_restitution = param.Number(
        default=0.6, bounds=(0.0, 1.0), softbounds=(0.0, 1.0), doc="Restitution α_rest"
    )
    alpha_fit = param.Number(
        default=0.4, bounds=(0.000001, 5.0), softbounds=(0.01, 5.0), doc="Reward exponent α"
    )
    beta_fit = param.Number(
        default=2.5, bounds=(0.000001, 5.0), softbounds=(0.01, 5.0), doc="Diversity exponent β"
    )
    eta = param.Number(
        default=0.003, bounds=(0.001, 10), softbounds=(0.001, 0.5), doc="Positivity floor η"
    )
    A = param.Number(default=3.5, bounds=(0.5, 5.0), doc="Logistic rescale amplitude A")
    sigma_min = param.Number(default=1e-8, bounds=(1e-9, 1e-3), doc="Standardisation σ_min")
    p_max = param.Number(default=1.0, bounds=(0.2, 10.0), doc="Maximum cloning probability p_max")
    epsilon_clone = param.Number(default=0.005, bounds=(1e-4, 0.05), doc="Cloning score ε_clone")
    companion_method = param.ObjectSelector(
        default="uniform",
        objects=("uniform", "softmax", "cloning", "random_pairing"),
        doc="Companion selection method",
    )
    companion_epsilon = param.Number(
        default=0.5, bounds=(0.0001, 1000), softbounds=(0.01, 5.0), doc="Companion ε"
    )
    integrator = param.ObjectSelector(
        default="boris-baoab",
        objects=("boris-baoab", "baoab"),
        doc="Integrator",
    )

    # Algorithm control
    enable_cloning = param.Boolean(default=True, doc="Enable cloning operator")
    enable_kinetic = param.Boolean(default=True, doc="Enable kinetic (Langevin) operator")
    pbc = param.Boolean(default=False, doc="Use periodic boundary conditions (wrap walkers)")

    # Initialisation controls
    init_offset = param.Number(default=4.5, bounds=(-6.0, 6.0), doc="Initial position offset")
    init_spread = param.Number(default=0.5, bounds=(0.1, 3.0), doc="Initial position spread")
    init_velocity_scale = param.Number(
        default=0.1, bounds=(0.01, 0.8), doc="Initial velocity scale"
    )
    bounds_extent = param.Number(default=6.0, bounds=(1, 12), doc="Spatial bounds half-width")

    def __init__(self, potential: object | None = None, dims: int = 2, **params):
        """Initialize GasConfig with optional potential function.

        Args:
            potential: Optional potential function object with evaluate() method.
                      If None, potential is created from benchmark_name parameter.
            dims: Spatial dimension (default: 2)
            **params: Override default parameter values
        """
        warnings.warn(
            "GasConfig is deprecated and will be removed in a future version. "
            "Use gas_config_panel.GasConfigPanel instead, which uses operator "
            "PanelModel interfaces for better maintainability.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**params)
        self.dims = dims
        self.history: RunHistory | None = None

        # Create or use provided potential
        if potential is None:
            self._update_benchmark()
        else:
            # Store provided potential directly (should be callable)
            self.potential = potential

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

        # Widget overrides for better UX
        self._widget_overrides: dict[str, pn.widgets.Widget] = {
            "sigma_min": pnw.EditableFloatSlider(
                name="sigma_min", start=1e-9, end=1e-3, value=self.sigma_min, step=1e-9
            ),
            "epsilon_clone": pnw.EditableFloatSlider(
                name="epsilon_clone", start=1e-4, end=0.05, value=self.epsilon_clone, step=1e-4
            ),
            "gamma": pnw.EditableFloatSlider(name="gamma", start=0.05, end=5.0, step=0.05),
            "beta": pnw.EditableFloatSlider(name="beta", start=0.1, end=5.0, step=0.05),
            "delta_t": pnw.EditableFloatSlider(name="delta_t", start=0.0001, end=0.2, step=0.0005),
            "lambda_alg": pnw.EditableFloatSlider(
                name="lambda_alg", start=0.0, end=3.0, step=0.05
            ),
            "nu": pnw.EditableFloatSlider(name="nu", start=0.0, end=10.0, step=0.1),
            "viscous_length_scale": pnw.EditableFloatSlider(
                name="viscous_length_scale", start=0.1, end=5.0, step=0.1
            ),
            # Fitness parameters (key for multimodal exploration tuning)
            "alpha_fit": pnw.EditableFloatSlider(
                name="alpha_fit (reward)", start=0.1, end=5.0, step=0.05, value=self.alpha_fit
            ),
            "beta_fit": pnw.EditableFloatSlider(
                name="beta_fit (diversity)", start=0.5, end=5.0, step=0.1, value=self.beta_fit
            ),
            "eta": pnw.EditableFloatSlider(
                name="eta", start=0.001, end=0.1, step=0.001, value=self.eta
            ),
            "A": pnw.EditableFloatSlider(name="A", start=1.0, end=5.0, step=0.1, value=self.A),
            "sigma_x": pnw.EditableFloatSlider(
                name="sigma_x", start=0.05, end=2.0, step=0.05, value=self.sigma_x
            ),
            "epsilon_F": pnw.EditableFloatSlider(
                name="epsilon_F", start=0.0, end=0.5, step=0.01, value=self.epsilon_F
            ),
            # Integer sliders
            "n_gaussians": pnw.EditableIntSlider(
                name="n_gaussians", start=1, end=10, value=self.n_gaussians, step=1
            ),
            "benchmark_seed": pnw.EditableIntSlider(
                name="benchmark_seed", start=0, end=9999, value=self.benchmark_seed, step=1
            ),
            "n_atoms": pnw.EditableIntSlider(
                name="n_atoms", start=2, end=30, value=self.n_atoms, step=1
            ),
            "N": pnw.EditableIntSlider(name="N", start=2, end=10000, value=self.N, step=1),
            "n_steps": pnw.EditableIntSlider(
                name="n_steps", start=10, end=10000, value=self.n_steps, step=1
            ),
        }

        # Callbacks for external listeners
        self._on_simulation_complete: list[Callable[[RunHistory], None]] = []
        self._on_benchmark_updated: list[Callable[[object, object, object], None]] = []

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
            resolution=100,  # Default resolution for background
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

        # Notify listeners with updated benchmark components
        for callback in self._on_benchmark_updated:
            callback(self.potential, self.background, self.mode_points)

    def _on_run_clicked(self, *_):
        """Handle Run button click."""
        self.status_pane.object = "**Running simulation...**"
        self.run_button.disabled = True

        try:
            # run_simulation() now handles history storage and callbacks
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

        # Create companion selection
        companion_selection = CompanionSelection(
            method=self.companion_method,
            epsilon=float(self.companion_epsilon),
            lambda_alg=float(self.lambda_alg),
        )

        # Create CloneOperator
        clone_op = CloneOperator(
            sigma_x=float(self.sigma_x),
            alpha_restitution=float(self.alpha_restitution),
            p_max=float(self.p_max),
            epsilon_clone=float(self.epsilon_clone),
        )

        # Create KineticOperator
        kinetic_op = KineticOperator(
            gamma=float(self.gamma),
            beta=float(self.beta),
            delta_t=float(self.delta_t),
            integrator=self.integrator,
            epsilon_F=float(self.epsilon_F),
            use_fitness_force=bool(self.use_fitness_force),
            use_potential_force=bool(self.use_potential_force),
            epsilon_Sigma=float(self.epsilon_Sigma),
            use_anisotropic_diffusion=bool(self.use_anisotropic_diffusion),
            diagonal_diffusion=bool(self.diagonal_diffusion),
            nu=float(self.nu),
            use_viscous_coupling=bool(self.use_viscous_coupling),
            viscous_length_scale=float(self.viscous_length_scale),
            V_alg=float(self.V_alg),
            use_velocity_squashing=bool(self.use_velocity_squashing),
            potential=self.potential,
            device=torch.device("cpu"),
            dtype=torch.float32,
            bounds=bounds,  # Pass bounds for periodic distance calculations
            pbc=bool(self.pbc),  # Enable periodic boundary conditions
        )

        # Create FitnessOperator with parameters directly
        fitness_op = FitnessOperator(
            alpha=float(self.alpha_fit),
            beta=float(self.beta_fit),
            eta=float(self.eta),
            lambda_alg=float(self.lambda_alg),
            sigma_min=float(self.sigma_min),
            A=float(self.A),
        )

        # Store operators for visualizer
        self.companion_selection = companion_selection
        self.clone_op = clone_op
        self.fitness_op = fitness_op

        # Create EuclideanGas
        self.gas = EuclideanGas(
            N=int(self.N),
            d=self.dims,
            companion_selection=companion_selection,
            potential=self.potential,
            kinetic_op=kinetic_op,
            cloning=clone_op,
            fitness_op=fitness_op,
            bounds=bounds,
            device=torch.device("cpu"),
            dtype="float32",
            enable_cloning=bool(self.enable_cloning),
            enable_kinetic=bool(self.enable_kinetic),
            pbc=bool(self.pbc),
        )

        # Initialize state
        offset = torch.full((self.dims,), float(self.init_offset), dtype=torch.float32)
        x_init = torch.randn(self.N, self.dims) * float(self.init_spread) + offset
        x_init = torch.clamp(x_init, min=low, max=high)
        v_init = torch.randn(self.N, self.dims) * float(self.init_velocity_scale)

        # Run simulation
        history = self.gas.run(self.n_steps, x_init=x_init, v_init=v_init)

        # Store history and notify listeners
        self.history = history
        for callback in self._on_simulation_complete:
            callback(self.history)

        return history

    def _build_param_panel(self, names: Iterable[str]) -> pn.Param:
        """Build parameter panel with custom widgets."""
        widgets = {
            name: self._widget_overrides[name] for name in names if name in self._widget_overrides
        }
        return pn.Param(
            self.param,
            parameters=list(names),
            widgets=widgets,
            show_name=False,
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.Column:
        """Create Panel dashboard for parameter configuration.

        Returns:
            Panel Column with parameter controls and Run button
        """
        # Benchmark configuration
        benchmark_params_base = ["benchmark_name"]

        # Dynamic benchmark-specific parameters
        def get_benchmark_specific_params(benchmark_name):
            """Return parameter panel for benchmark-specific settings."""
            if benchmark_name == "Mixture of Gaussians":
                return pn.Param(
                    self.param,
                    parameters=["n_gaussians", "benchmark_seed"],
                    show_name=False,
                    sizing_mode="stretch_width",
                )
            if benchmark_name == "Lennard-Jones":
                return pn.Param(
                    self.param,
                    parameters=["n_atoms"],
                    show_name=False,
                    sizing_mode="stretch_width",
                )
            return pn.pane.Markdown("*No additional parameters*", sizing_mode="stretch_width")

        benchmark_specific = pn.bind(get_benchmark_specific_params, self.param.benchmark_name)

        benchmark_panel = pn.Column(
            pn.pane.Markdown("### Potential Function"),
            self._build_param_panel(benchmark_params_base),
            benchmark_specific,
            sizing_mode="stretch_width",
        )

        general_params = (
            "N",
            "n_steps",
            "enable_cloning",
            "enable_kinetic",
            "pbc",
        )
        langevin_params = (
            "gamma",
            "beta",
            "delta_t",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diagonal_diffusion",
            "nu",
            "use_viscous_coupling",
            "viscous_length_scale",
            "V_alg",
            "use_velocity_squashing",
        )
        cloning_params = (
            "sigma_x",
            "lambda_alg",
            "alpha_restitution",
            "alpha_fit",
            "beta_fit",
            "eta",
            "A",
            "sigma_min",
            "p_max",
            "epsilon_clone",
            "companion_method",
            "companion_epsilon",
        )
        init_params = ("init_offset", "init_spread", "init_velocity_scale", "bounds_extent")

        accordion = pn.Accordion(
            ("Potential Function", benchmark_panel),
            ("General", self._build_param_panel(general_params)),
            ("Langevin Dynamics", self._build_param_panel(langevin_params)),
            ("Cloning & Selection", self._build_param_panel(cloning_params)),
            ("Initialization", self._build_param_panel(init_params)),
            sizing_mode="stretch_width",
        )
        accordion.active = [0, 1]  # Open first two sections by default

        return pn.Column(
            pn.pane.Markdown("## Simulation Parameters"),
            accordion,
            self.run_button,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=380,
        )
