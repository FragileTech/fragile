"""Dashboard assembly for Gas algorithm visualization.

This module assembles the complete visualization dashboard by combining:
- GasConfigPanel: Modern simulation parameter configuration UI using operator panels
- GasVisualizer: N-particle swarm evolution visualization
- ConvergencePanel: Convergence analysis (KL, Lyapunov)
- ConvergenceBoundsPanel: Theoretical bounds and parameter optimization

Can be run as a standalone script to spawn an interactive dashboard:
    python -m fragile.experiments.gas_visualization_dashboard
"""

from __future__ import annotations

import holoviews as hv
import panel as pn

from fragile.fractalai.experiments.convergence import ConvergencePanel
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel
from fragile.fractalai.experiments.n_particle_swarm import GasVisualizer
from fragile.fractalai.experiments.parameter_optimization import ConvergenceBoundsPanel


__all__ = ["create_app"]


def create_app(dims: int = 2, n_gaussians: int = 3, bounds_extent: float = 6.0):
    """Create the Panel app for standalone usage.

    Args:
        dims: Spatial dimension
        n_gaussians: Number of Gaussian modes in potential
        bounds_extent: Spatial bounds half-width

    Returns:
        Panel template ready to serve
    """
    # Initialize extensions
    hv.extension("bokeh")
    pn.extension()

    # Create gas configuration dashboard (creates its own benchmark)
    gas_config = GasConfigPanel(dims=dims)

    # Get potential and mode_points from gas_config
    potential = gas_config.potential
    mode_points = gas_config.mode_points

    # Generate rich background from benchmark.show() instead of simple background
    background = potential.show(
        show_optimum=gas_config.show_optimum,
        show_density=gas_config.show_density,
        show_contours=gas_config.show_contours,
        n_cells=gas_config.viz_n_cells,
    )

    # Create visualizer (initially with no history)
    visualizer = GasVisualizer(
        history=None,
        potential=potential,
        background=background,
        mode_points=mode_points,
        bounds_extent=bounds_extent,
        pbc=gas_config.gas_params["pbc"],
    )

    # Create convergence panel (initially with no history)
    convergence_panel = ConvergencePanel(
        history=None,
        potential=potential,
        benchmark=potential,  # Pass benchmark for best_state access
        bounds_extent=bounds_extent,
    )

    # Create convergence bounds diagnostics panel
    bounds_diagnostics_panel = ConvergenceBoundsPanel(
        gas_config=gas_config,
        potential=potential,
        bounds_extent=bounds_extent,
    )

    # Connect simulation completion to visualizer and convergence panel
    def on_simulation_complete(history):
        """Update visualizer and convergence panel when simulation completes."""
        # Extract parameters from kinetic_op for force visualization
        visualizer.epsilon_F = float(gas_config.kinetic_op.epsilon_F)
        visualizer.use_fitness_force = bool(gas_config.kinetic_op.use_fitness_force)
        visualizer.use_potential_force = bool(gas_config.kinetic_op.use_potential_force)

        # Set companion selection and fitness params from gas_config operators
        visualizer.companion_selection = gas_config.companion_selection
        visualizer.fitness_op = gas_config.fitness_op

        # Load the history in a separate thread to avoid blocking UI
        # This is critical: _process_history() contains expensive computations
        # that would freeze the UI if run synchronously
        def _update_history():
            visualizer.set_history(history)
            convergence_panel.set_history(history)  # Also update convergence panel
            bounds_diagnostics_panel.set_history(history)  # Also update diagnostics panel

        import threading

        thread = threading.Thread(target=_update_history, daemon=True)
        thread.start()

    gas_config.add_completion_callback(on_simulation_complete)

    def on_benchmark_change(potential, background, mode_points):
        """Update visualizer, convergence panel, and diagnostics when benchmark changes."""
        # Generate rich background visualization
        rich_background = potential.show(
            show_optimum=gas_config.show_optimum,
            show_density=gas_config.show_density,
            show_contours=gas_config.show_contours,
            n_cells=gas_config.viz_n_cells,
        )
        visualizer.update_benchmark(potential, rich_background, mode_points)
        convergence_panel.update_benchmark(potential)  # Update convergence benchmark
        bounds_diagnostics_panel.update_benchmark(potential)  # Update diagnostics benchmark

    gas_config.add_benchmark_callback(on_benchmark_change)

    def update_benchmark_viz():
        """Update benchmark background in visualizer when visualization parameters change."""
        try:
            rich_background = gas_config.potential.show(
                show_optimum=gas_config.show_optimum,
                show_density=gas_config.show_density,
                show_contours=gas_config.show_contours,
                n_cells=gas_config.viz_n_cells,
            )
            # Update visualizer's background
            visualizer.update_benchmark(
                gas_config.potential, rich_background, gas_config.mode_points
            )
        except Exception as e:
            print(f"Error updating benchmark visualization: {e}")

    # Watch for visualization parameter changes to update background
    gas_config.param.watch(
        lambda *_: update_benchmark_viz(),
        [
            "show_optimum",
            "show_density",
            "show_contours",
            "viz_n_cells",
        ],
    )

    # Create tabbed layout
    tabs = pn.Tabs(
        (
            "Evolution",
            pn.Column(
                pn.pane.Markdown("## Swarm Evolution on Potential Landscape"),
                visualizer.panel(),
                sizing_mode="stretch_width",
            ),
        ),
        (
            "Convergence",
            pn.Column(
                pn.pane.Markdown("## Convergence Analysis"),
                convergence_panel.panel(),
                sizing_mode="stretch_width",
            ),
        ),
        (
            "Diagnostics",
            pn.Column(
                bounds_diagnostics_panel.panel(),
                sizing_mode="stretch_width",
            ),
        ),
        dynamic=True,
        sizing_mode="stretch_width",
    )

    # Create layout using FastListTemplate
    return pn.template.FastListTemplate(
        title="Gas Visualization Dashboard",
        sidebar=[
            pn.pane.Markdown("## Simulation Control"),
            gas_config.panel(),
        ],
        main=[tabs],
        sidebar_width=400,
        main_max_width="100%",
    )


if __name__ == "__main__":
    # Create and serve the app without opening browser
    app = create_app()
    app.show(port=5007, open=False)
    print("Gas Visualization Dashboard running at http://localhost:5007")
