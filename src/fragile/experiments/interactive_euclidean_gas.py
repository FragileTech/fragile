"""Interactive Euclidean Gas explorer with integrated configuration and visualization.

This module provides the integrated SwarmExplorer dashboard that combines
parameter configuration (GasConfig) and visualization (GasVisualizer) into
a unified interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import holoviews as hv
import panel as pn

from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.gas_config_dashboard import GasConfig


if TYPE_CHECKING:
    pass


__all__ = [
    "SwarmExplorer",
    "create_dashboard",
    "prepare_background",
]


def prepare_background(
    dims: int = 2,
    n_gaussians: int = 3,
    bounds_range: tuple[float, float] = (-6.0, 6.0),
    seed: int = 42,
    resolution: int = 200,
) -> tuple[object, hv.Image, hv.Points]:
    """Pre-compute potential, density backdrop, and mode markers for the explorer.

    .. deprecated:: 0.1.0
        Use :func:`fragile.core.benchmarks.prepare_benchmark_for_explorer` instead.
        This function only supports MixtureOfGaussians. The new function in
        benchmarks.py supports all benchmark types (Sphere, Rastrigin, etc.).

    Args:
        dims: Spatial dimension (must be 2 for visualization)
        n_gaussians: Number of Gaussian modes in mixture
        bounds_range: (min, max) bounds for spatial domain
        seed: Random seed for reproducible potential
        resolution: Grid resolution for background density

    Returns:
        Tuple of (potential, background_image, mode_points)

    Example:
        Old usage::

            from fragile.experiments.interactive_euclidean_gas import prepare_background

            potential, background, mode_points = prepare_background(dims=2)

        New usage::

            from fragile.core.benchmarks import prepare_benchmark_for_explorer

            potential, benchmark, background, mode_points = prepare_benchmark_for_explorer(
                benchmark_name="Mixture of Gaussians",
                dims=2,
                bounds_range=(-6.0, 6.0),
                resolution=200,
                n_gaussians=3,
                seed=42,
            )
    """
    warnings.warn(
        "prepare_background() is deprecated and will be removed in a future version. "
        "Use fragile.core.benchmarks.prepare_benchmark_for_explorer() instead, "
        "which supports all benchmark types (not just MixtureOfGaussians).",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use the new unified function
    potential, _benchmark, background, mode_points = prepare_benchmark_for_explorer(
        benchmark_name="Mixture of Gaussians",
        dims=dims,
        bounds_range=bounds_range,
        resolution=resolution,
        n_gaussians=n_gaussians,
        seed=seed,
    )

    return potential, background, mode_points


class SwarmExplorer:
    """Integrated dashboard combining parameter configuration and visualization.

    This class provides the complete interactive dashboard for exploring
    EuclideanGas dynamics by composing GasConfig (parameters) and GasVisualizer
    (display) into a unified interface.

    Example:
        >>> potential, background, mode_points = prepare_background(dims=2)
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
            cloning_params=None,  # Will be set after run
            bounds_extent=self.config.bounds_extent,
            epsilon_F=self.config.epsilon_F,
            use_fitness_force=self.config.use_fitness_force,
            use_potential_force=self.config.use_potential_force,
            **display_params,
        )

        # Wire up callback: when simulation completes -> update visualizer
        self.config.add_completion_callback(self._on_simulation_complete)

    def _on_simulation_complete(self, history):
        """Handle simulation completion - update visualizer with new history.

        Args:
            history: RunHistory from completed simulation
        """
        # Get parameters for visualization
        from fragile.core.companion_selection import CompanionSelection
        from fragile.core.euclidean_gas import CloningParams

        companion_selection = CompanionSelection(
            method=self.config.companion_method,
            epsilon=float(self.config.companion_epsilon),
            lambda_alg=float(self.config.lambda_alg),
        )

        cloning_params = CloningParams(
            sigma_x=float(self.config.sigma_x),
            lambda_alg=float(self.config.lambda_alg),
            alpha_restitution=float(self.config.alpha_restitution),
            alpha=float(self.config.alpha_fit),
            beta=float(self.config.beta_fit),
            eta=float(self.config.eta),
            A=float(self.config.A),
            sigma_min=float(self.config.sigma_min),
            p_max=float(self.config.p_max),
            epsilon_clone=float(self.config.epsilon_clone),
            companion_selection=companion_selection,
        )

        # Update visualizer settings
        self.visualizer.companion_selection = companion_selection
        self.visualizer.cloning_params = cloning_params
        self.visualizer.bounds_extent = self.config.bounds_extent
        self.visualizer.epsilon_F = self.config.epsilon_F
        self.visualizer.use_fitness_force = self.config.use_fitness_force
        self.visualizer.use_potential_force = self.config.use_potential_force

        # Load new history into visualizer
        self.visualizer.set_history(history)

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
        potential, background, mode_points = prepare_background(dims=dims)

    explorer_params = explorer_params or {}
    explorer = SwarmExplorer(
        potential=potential,
        background=background,
        mode_points=mode_points,
        dims=dims,
        **explorer_params,
    )
    return explorer, explorer.panel()
