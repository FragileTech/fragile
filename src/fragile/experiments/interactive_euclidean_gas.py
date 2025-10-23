"""Interactive Euclidean Gas explorer with integrated configuration and visualization.

This module provides the integrated SwarmExplorer dashboard that combines
parameter configuration (GasConfig) and visualization (GasVisualizer) into
a unified interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import panel as pn

from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.gas_config_dashboard import GasConfig


if TYPE_CHECKING:
    pass


__all__ = [
    "SwarmExplorer",
    "create_dashboard",
]


class SwarmExplorer:
    """Integrated dashboard combining parameter configuration and visualization.

    This class provides the complete interactive dashboard for exploring
    EuclideanGas dynamics by composing GasConfig (parameters) and GasVisualizer
    (display) into a unified interface.

    Example:
        >>> from fragile.core.benchmarks import prepare_benchmark_for_explorer
        >>> potential, background, mode_points = prepare_benchmark_for_explorer(
        ...     benchmark_name="Mixture of Gaussians",
        ...     dims=2,
        ...     bounds_range=(-6.0, 6.0),
        ... )
        >>> explorer = SwarmExplorer(potential, background, mode_points, dims=2)
        >>> dashboard = explorer.panel()
        >>> dashboard.show()  # Launch interactive dashboard
    """

    def __init__(
        self,
        potential: object,
        background: hv.Image,
        mode_points: hv.Points,
        dims: int = 2,
        **params,
    ):
        """Initialize SwarmExplorer with potential and background visuals.

        Args:
            potential: Potential function object with evaluate() method
            background: HoloViews Image for background visualization
            mode_points: HoloViews Points showing target modes
            dims: Spatial dimension (default: 2)
            **params: Override default parameter values (passed to GasConfig)
        """
        # Lazy import to avoid circular dependency
        from fragile.experiments.gas_visualization_dashboard import GasVisualizer

        self.potential = potential
        self.background = background
        self.mode_points = mode_points
        self.dims = dims

        # Extract display parameters from params (if any)
        display_params = {}
        config_param_names = {
            "N",
            "n_steps",
            "gamma",
            "beta",
            "delta_t",
            "epsilon_F",
            "use_fitness_force",
            "use_potential_force",
            "epsilon_Sigma",
            "use_anisotropic_diffusion",
            "diagonal_diffusion",
            "V_alg",
            "use_velocity_squashing",
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
            "integrator",
            "enable_cloning",
            "enable_kinetic",
            "init_offset",
            "init_spread",
            "init_velocity_scale",
            "bounds_extent",
        }

        config_params = {k: v for k, v in params.items() if k in config_param_names}
        display_params = {k: v for k, v in params.items() if k not in config_param_names}

        # Create configuration dashboard
        self.config = GasConfig(potential=potential, dims=dims, **config_params)

        # Create visualization dashboard (initially without history)
        self.visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            companion_selection=None,  # Will be set after run
            fitness_op=None,  # Will be set after run
            bounds_extent=self.config.bounds_extent,
            epsilon_F=self.config.epsilon_F,
            use_fitness_force=self.config.use_fitness_force,
            use_potential_force=self.config.use_potential_force,
            **display_params,
        )

        # Wire up callbacks
        # When simulation completes -> update visualizer with history
        self.config.add_completion_callback(self._on_simulation_complete)
        # When benchmark changes -> update visualizer with new background/mode_points
        self.config.add_benchmark_callback(self._on_benchmark_update)

    def _on_simulation_complete(self, history):
        """Handle simulation completion - update visualizer with new history.

        Args:
            history: RunHistory from completed simulation
        """
        # Get parameters for visualization
        from fragile.core.companion_selection import CompanionSelection
        from fragile.core.fitness import FitnessOperator

        companion_selection = CompanionSelection(
            method=self.config.companion_method,
            epsilon=float(self.config.companion_epsilon),
            lambda_alg=float(self.config.lambda_alg),
        )

        # Create FitnessOperator with parameters directly
        fitness_op = FitnessOperator(
            alpha=float(self.config.alpha_fit),
            beta=float(self.config.beta_fit),
            eta=float(self.config.eta),
            lambda_alg=float(self.config.lambda_alg),
            sigma_min=float(self.config.sigma_min),
            A=float(self.config.A),
        )

        # Update visualizer settings
        self.visualizer.companion_selection = companion_selection
        self.visualizer.fitness_op = fitness_op
        self.visualizer.bounds_extent = self.config.bounds_extent
        self.visualizer.epsilon_F = self.config.epsilon_F
        self.visualizer.use_fitness_force = self.config.use_fitness_force
        self.visualizer.use_potential_force = self.config.use_potential_force

        # Load new history into visualizer
        self.visualizer.set_history(history)

    def _on_benchmark_update(
        self, potential: object, background: hv.Image, mode_points: hv.Points
    ):
        """Handle benchmark update - update visualizer with new background/mode_points.

        Args:
            potential: New potential function
            background: New HoloViews Image for background
            mode_points: New HoloViews Points for target modes
        """
        # Update stored references
        self.potential = potential
        self.background = background
        self.mode_points = mode_points

        # Update visualizer
        self.visualizer.update_benchmark(potential, background, mode_points)

    def panel(self) -> pn.Row:
        """Create the complete dashboard panel.

        Returns:
            Panel Row containing configuration (left) and visualization (right)
        """
        return pn.Row(
            self.config.panel(),
            self.visualizer.panel(),
            sizing_mode="stretch_width",
        )


def create_dashboard(
    potential: object | None = None,
    background: hv.Image | None = None,
    mode_points: hv.Points | None = None,
    *,
    dims: int = 2,
    explorer_params: dict | None = None,
) -> tuple[SwarmExplorer, pn.Row]:
    """Factory function for creating SwarmExplorer dashboard.

    Args:
        potential: Potential function (if None, creates default multimodal)
        background: Background image (if None, creates from potential)
        mode_points: Mode markers (if None, creates from potential)
        dims: Spatial dimension
        explorer_params: Override default SwarmExplorer parameters

    Returns:
        Tuple of (explorer, panel) for the interactive dashboard

    Example:
        >>> explorer, dashboard = create_dashboard(dims=2)
        >>> dashboard.show()
    """
    if potential is None or background is None or mode_points is None:
        potential, background, mode_points = prepare_benchmark_for_explorer(
            benchmark_name="Mixture of Gaussians",
            dims=dims,
            bounds_range=(-6.0, 6.0),
            resolution=200,
        )

    explorer_params = explorer_params or {}
    explorer = SwarmExplorer(
        potential=potential,
        background=background,
        mode_points=mode_points,
        dims=dims,
        **explorer_params,
    )
    return explorer, explorer.panel()
