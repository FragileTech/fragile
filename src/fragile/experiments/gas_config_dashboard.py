"""Reusable parameter configuration dashboard for EuclideanGas simulations.

This module provides a Panel-based dashboard for configuring simulation parameters
and running EuclideanGas simulations. It returns RunHistory objects that can be
visualized or analyzed separately.
"""

from __future__ import annotations

from typing import Callable, Iterable

import panel as pn
import panel.widgets as pnw
import param
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import CloningParams, EuclideanGas
from fragile.core.fitness import FitnessOperator, FitnessParams
from fragile.core.history import RunHistory
from fragile.core.kinetic_operator import KineticOperator, LangevinParams


__all__ = ["GasConfig"]


class GasConfig(param.Parameterized):
    """Reusable configuration dashboard for EuclideanGas simulations.

    This class provides a Panel-based UI for configuring all simulation parameters
    and running EuclideanGas. After running, it provides a RunHistory object that
    can be visualized or analyzed.

    Example:
        >>> potential, _, _ = prepare_background(dims=2)
        >>> config = GasConfig(potential=potential, dims=2)
        >>> dashboard = config.panel()
        >>> dashboard.show()  # Interactive parameter selection
        >>> history = config.history  # Access result after running
    """

    # Simulation controls
    N = param.Integer(default=160, bounds=(10, 1000), doc="Number of walkers")
    n_steps = param.Integer(default=240, bounds=(50, 1000), doc="Simulation steps")

    # Langevin parameters
    gamma = param.Number(default=1.0, bounds=(0.05, 5.0), doc="Friction γ")
    beta = param.Number(default=1.0, bounds=(0.01, 10.0), doc="Inverse temperature β")
    delta_t = param.Number(default=0.05, bounds=(0.01, 0.2), doc="Time step Δt")
    epsilon_F = param.Number(default=0.0, bounds=(0.0, 0.5), doc="Fitness force rate ε_F")
    use_fitness_force = param.Boolean(default=False, doc="Enable fitness-driven force")
    use_potential_force = param.Boolean(default=False, doc="Enable potential force")
    epsilon_Sigma = param.Number(default=0.1, bounds=(0.0, 1.0), doc="Hessian regularisation ε_Σ")
    use_anisotropic_diffusion = param.Boolean(default=False, doc="Enable anisotropic diffusion")
    diagonal_diffusion = param.Boolean(default=True, doc="Use diagonal diffusion tensor")

    # Cloning parameters
    sigma_x = param.Number(default=0.15, bounds=(0.01, 1.0), doc="Cloning jitter σ_x")
    lambda_alg = param.Number(
        default=0.5, bounds=(0.0, 3.0), doc="Algorithmic distance weight λ_alg"
    )
    alpha_restitution = param.Number(default=0.6, bounds=(0.0, 1.0), doc="Restitution α_rest")
    alpha_fit = param.Number(default=0.7, bounds=(0.01, 5.0), doc="Reward exponent α")
    beta_fit = param.Number(default=1.3, bounds=(0.01, 5.0), doc="Diversity exponent β")
    eta = param.Number(default=0.01, bounds=(0.001, 0.5), doc="Positivity floor η")
    A = param.Number(default=2.0, bounds=(0.5, 5.0), doc="Logistic rescale amplitude A")
    sigma_min = param.Number(default=1e-8, bounds=(1e-9, 1e-3), doc="Standardisation σ_min")
    p_max = param.Number(default=1.0, bounds=(0.2, 10.0), doc="Maximum cloning probability p_max")
    epsilon_clone = param.Number(default=0.005, bounds=(1e-4, 0.05), doc="Cloning score ε_clone")
    companion_method = param.ObjectSelector(
        default="random_pairing",
        objects=("uniform", "softmax", "cloning", "random_pairing"),
        doc="Companion selection method",
    )
    companion_epsilon = param.Number(default=0.5, bounds=(0.01, 5.0), doc="Companion ε")
    integrator = param.ObjectSelector(default="baoab", objects=("baoab",), doc="Integrator")

    # Algorithm control
    enable_cloning = param.Boolean(default=True, doc="Enable cloning operator")
    enable_kinetic = param.Boolean(default=True, doc="Enable kinetic (Langevin) operator")

    # Initialisation controls
    init_offset = param.Number(default=4.5, bounds=(-6.0, 6.0), doc="Initial position offset")
    init_spread = param.Number(default=0.5, bounds=(0.1, 3.0), doc="Initial position spread")
    init_velocity_scale = param.Number(
        default=0.1, bounds=(0.01, 0.8), doc="Initial velocity scale"
    )
    bounds_extent = param.Number(default=6.0, bounds=(1, 12), doc="Spatial bounds half-width")

    def __init__(self, potential: object, dims: int = 2, **params):
        """Initialize GasConfig with a potential function.

        Args:
            potential: Potential function object with evaluate() method
            dims: Spatial dimension (default: 2)
            **params: Override default parameter values
        """
        super().__init__(**params)
        self.potential = potential
        self.dims = dims
        self.history: RunHistory | None = None

        # Create UI components
        self.run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
        self.run_button.sizing_mode = "stretch_width"
        self.run_button.on_click(self._on_run_clicked)

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Widget overrides for better UX
        self._widget_overrides: dict[str, pn.widgets.Widget] = {
            "sigma_min": pnw.FloatSlider(
                name="sigma_min", start=1e-9, end=1e-3, value=self.sigma_min, step=1e-9
            ),
            "epsilon_clone": pnw.FloatSlider(
                name="epsilon_clone", start=1e-4, end=0.05, value=self.epsilon_clone, step=1e-4
            ),
            "gamma": pnw.FloatSlider(name="gamma", start=0.05, end=5.0, step=0.05),
            "beta": pnw.FloatSlider(name="beta", start=0.1, end=5.0, step=0.05),
            "delta_t": pnw.FloatSlider(name="delta_t", start=0.01, end=0.2, step=0.005),
            "lambda_alg": pnw.FloatSlider(name="lambda_alg", start=0.0, end=3.0, step=0.1),
        }

        # Callbacks for external listeners
        self._on_simulation_complete: list[Callable[[RunHistory], None]] = []

    def add_completion_callback(self, callback: Callable[[RunHistory], None]):
        """Register a callback to be called when simulation completes.

        Args:
            callback: Function that takes RunHistory as argument
        """
        self._on_simulation_complete.append(callback)

    def _on_run_clicked(self, *_):
        """Handle Run button click."""
        self.status_pane.object = "**Running simulation...**"
        self.run_button.disabled = True

        try:
            self.history = self.run_simulation()
            self.status_pane.object = (
                f"**Simulation complete!** "
                f"{self.history.n_steps} steps, "
                f"{self.history.n_recorded} recorded timesteps"
            )

            # Notify listeners
            for callback in self._on_simulation_complete:
                callback(self.history)

        except Exception as e:
            self.status_pane.object = f"**Error:** {str(e)}"
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

        # Create Langevin parameters
        langevin_params = LangevinParams(
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
        )

        # Create cloning parameters
        cloning_params = CloningParams(
            sigma_x=float(self.sigma_x),
            lambda_alg=float(self.lambda_alg),
            alpha_restitution=float(self.alpha_restitution),
            alpha=float(self.alpha_fit),
            beta=float(self.beta_fit),
            eta=float(self.eta),
            A=float(self.A),
            sigma_min=float(self.sigma_min),
            p_max=float(self.p_max),
            epsilon_clone=float(self.epsilon_clone),
            companion_selection=companion_selection,
        )

        # Create KineticOperator
        kinetic_op = KineticOperator(
            gamma=langevin_params.gamma,
            beta=langevin_params.beta,
            delta_t=langevin_params.delta_t,
            integrator=langevin_params.integrator,
            epsilon_F=langevin_params.epsilon_F,
            use_fitness_force=langevin_params.use_fitness_force,
            use_potential_force=langevin_params.use_potential_force,
            epsilon_Sigma=langevin_params.epsilon_Sigma,
            use_anisotropic_diffusion=langevin_params.use_anisotropic_diffusion,
            diagonal_diffusion=langevin_params.diagonal_diffusion,
            potential=self.potential,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Create FitnessOperator
        fitness_params = FitnessParams(
            alpha=float(self.alpha_fit),
            beta=float(self.beta_fit),
            eta=float(self.eta),
            lambda_alg=float(self.lambda_alg),
            sigma_min=float(self.sigma_min),
            A=float(self.A),
        )
        fitness_op = FitnessOperator(
            params=fitness_params,
            companion_selection=companion_selection,
        )

        # Create EuclideanGas
        gas = EuclideanGas(
            N=int(self.N),
            d=self.dims,
            companion_selection=companion_selection,
            potential=self.potential,
            kinetic_op=kinetic_op,
            cloning=cloning_params,
            fitness_op=fitness_op,
            bounds=bounds,
            device=torch.device("cpu"),
            dtype="float32",
            enable_cloning=bool(self.enable_cloning),
            enable_kinetic=bool(self.enable_kinetic),
        )

        # Initialize state
        offset = torch.full((self.dims,), float(self.init_offset), dtype=torch.float32)
        x_init = torch.randn(self.N, self.dims) * float(self.init_spread) + offset
        x_init = torch.clamp(x_init, min=low, max=high)
        v_init = torch.randn(self.N, self.dims) * float(self.init_velocity_scale)

        # Run simulation
        history = gas.run(self.n_steps, x_init=x_init, v_init=v_init)

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
        general_params = (
            "N",
            "n_steps",
            "enable_cloning",
            "enable_kinetic",
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
            ("General", self._build_param_panel(general_params)),
            ("Langevin Dynamics", self._build_param_panel(langevin_params)),
            ("Cloning & Selection", self._build_param_panel(cloning_params)),
            ("Initialization", self._build_param_panel(init_params)),
            sizing_mode="stretch_width",
        )
        accordion.active = [0]

        return pn.Column(
            pn.pane.Markdown("## Simulation Parameters"),
            accordion,
            self.run_button,
            self.status_pane,
            sizing_mode="stretch_width",
            min_width=380,
        )
