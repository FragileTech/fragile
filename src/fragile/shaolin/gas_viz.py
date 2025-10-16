"""
Canonical Gas Visualization for Euclidean Gas.

This module provides streaming visualization of swarm dynamics with:
- Position scatter plots with customizable colors
- Velocity vector fields
- Alive walker tracking
- Real-time updates using HoloViews streaming
"""

import pandas as pd
import panel as pn

from fragile.euclidean_gas import SwarmState
from fragile.shaolin.stream_plots import Curve, Scatter, VectorField


class GasVisualization:
    """
    Canonical visualization for Euclidean Gas swarm dynamics.

    Displays:
    - Positions as scatter plot (configurable color)
    - Velocities as vector field (configurable color)
    - Number of alive walkers over time
    """

    def __init__(
        self,
        bounds=None,
        position_color: str = "blue",
        velocity_color: str = "cyan",
        velocity_scale: float = 0.5,
        plot_size: int = 600,
        track_alive: bool = True,
    ):
        """
        Initialize gas visualization.

        Args:
            bounds: Optional Bounds object for axis limits
            position_color: Color for walker positions
            velocity_color: Color for velocity vectors
            velocity_scale: Scaling factor for velocity arrows
            plot_size: Size of position/velocity plot
            track_alive: Whether to track alive walkers
        """
        self.bounds = bounds
        self.position_color = position_color
        self.velocity_color = velocity_color
        self.velocity_scale = velocity_scale
        self.plot_size = plot_size
        self.track_alive = track_alive

        # Initialize step counter
        self.step = 0
        self.alive_history = []
        self.cloned_history = []

        # Initialize streaming plots
        self._init_plots()

    def _init_plots(self):
        """Initialize all streaming plot components."""
        # Position scatter
        self.position_stream = Scatter(
            data=pd.DataFrame({"x": [], "y": []}),
            bokeh_opts={
                "color": self.position_color,
                "size": 8,
                "alpha": 0.7,
                "height": self.plot_size,
                "width": self.plot_size,
                "tools": ["hover", "box_zoom", "wheel_zoom", "reset"],
            },
        )

        # Velocity vector field
        self.velocity_stream = VectorField(
            data=pd.DataFrame({"x0": [], "y0": [], "x1": [], "y1": []}),
            scale=self.velocity_scale,
            bokeh_opts={
                "line_color": self.velocity_color,
                "line_width": 2,
                "alpha": 0.5,
                "height": self.plot_size,
                "width": self.plot_size,
            },
        )

        # Alive walkers curve
        if self.track_alive:
            self.alive_stream = Curve(
                data=pd.DataFrame({"step": [], "alive": []}),
                bokeh_opts={
                    "color": "green",
                    "line_width": 3,
                    "height": 300,
                    "width": 800,
                    "ylabel": "Walker Count",
                    "xlabel": "Step",
                    "title": "Alive vs Cloned Walkers",
                },
            )

            # Cloned walkers curve (overlaid)
            self.cloned_stream = Curve(
                data=pd.DataFrame({"step": [], "cloned": []}),
                bokeh_opts={
                    "color": "red",
                    "line_width": 3,
                    "line_dash": "dashed",
                    "height": 300,
                    "width": 800,
                },
            )

    def update(self, state: SwarmState, n_alive: int | None = None):
        """
        Update visualization with new swarm state.

        Args:
            state: Current SwarmState
            n_alive: Optional explicit alive count (if None, assumes all alive)
        """
        # Convert tensors to numpy
        x_np = state.x.detach().cpu().numpy()
        v_np = state.v.detach().cpu().numpy()

        # Determine number of alive walkers
        if n_alive is None:
            n_alive = state.N

        # Update positions
        pos_data = pd.DataFrame({
            "x": x_np[:, 0],
            "y": x_np[:, 1] if x_np.shape[1] > 1 else x_np[:, 0],
        })
        self.position_stream.send(pos_data)

        # Update velocities (create segments)
        vel_data = pd.DataFrame({
            "x0": x_np[:, 0],
            "y0": x_np[:, 1] if x_np.shape[1] > 1 else x_np[:, 0],
            "x1": x_np[:, 0] + v_np[:, 0] * self.velocity_scale,
            "y1": (x_np[:, 1] if x_np.shape[1] > 1 else x_np[:, 0])
            + (v_np[:, 1] if v_np.shape[1] > 1 else v_np[:, 0]) * self.velocity_scale,
        })
        self.velocity_stream.send(vel_data)

        # Update alive count
        if self.track_alive:
            # Calculate number of cloned walkers (dead/out-of-bounds)
            n_cloned = state.N - n_alive

            # Send only the NEW data point to Buffer (not the full history)
            alive_point = pd.DataFrame({"step": [self.step], "alive": [n_alive]})
            self.alive_stream.send(alive_point)

            cloned_point = pd.DataFrame({"step": [self.step], "cloned": [n_cloned]})
            self.cloned_stream.send(cloned_point)

            # Keep history for reference
            self.alive_history.append({"step": self.step, "alive": n_alive})
            self.cloned_history.append({"step": self.step, "cloned": n_cloned})

        self.step += 1

    def create_layout(self):
        """
        Create the complete visualization layout.

        Returns:
            Panel layout with all plots
        """
        # Overlay positions and velocities
        main_plot = (self.position_stream.plot * self.velocity_stream.plot).opts(
            title=f"Euclidean Gas Swarm (Step {self.step})",
            xlabel="x₁",
            ylabel="x₂",
        )

        # Apply bounds if specified
        if self.bounds is not None:
            xlim = (self.bounds.low[0].item(), self.bounds.high[0].item())
            ylim = (
                self.bounds.low[1].item() if len(self.bounds.low) > 1 else xlim[0],
                self.bounds.high[1].item() if len(self.bounds.high) > 1 else xlim[1],
            )
            main_plot = main_plot.opts(xlim=xlim, ylim=ylim)

        # Create layout
        if self.track_alive:
            # Overlay alive and cloned curves
            alive_plot = (self.alive_stream.plot * self.cloned_stream.plot).opts(
                legend_position="top_right",
            )

            layout = pn.Column(
                pn.pane.Markdown("# Euclidean Gas Visualization"),
                main_plot,
                pn.pane.Markdown("---"),
                pn.pane.Markdown(
                    "**Green (solid)** = Alive walkers | **Red (dashed)** = Cloned walkers"
                ),
                alive_plot,
            )
        else:
            layout = pn.Column(
                pn.pane.Markdown("# Euclidean Gas Visualization"),
                main_plot,
            )

        return layout

    def reset(self):
        """Reset the visualization state."""
        self.step = 0
        self.alive_history = []
        self.cloned_history = []


class BoundaryGasVisualization(GasVisualization):
    """
    Gas visualization with boundary enforcement tracking.

    Extends GasVisualization to:
    - Color in-bounds walkers differently from out-of-bounds
    - Track alive (in-bounds) vs dead (out-of-bounds) walkers
    """

    def __init__(
        self,
        bounds,
        in_bounds_color: str = "blue",
        out_bounds_color: str = "red",
        velocity_color: str = "cyan",
        velocity_scale: float = 0.5,
        plot_size: int = 600,
    ):
        """
        Initialize boundary gas visualization.

        Args:
            bounds: Bounds object (required)
            in_bounds_color: Color for in-bounds walkers
            out_bounds_color: Color for out-of-bounds walkers
            velocity_color: Color for velocity vectors
            velocity_scale: Scaling factor for velocity arrows
            plot_size: Size of plots
        """
        assert bounds is not None, "BoundaryGasVisualization requires bounds"

        self.in_bounds_color = in_bounds_color
        self.out_bounds_color = out_bounds_color

        super().__init__(
            bounds=bounds,
            position_color=in_bounds_color,  # Default, will override per update
            velocity_color=velocity_color,
            velocity_scale=velocity_scale,
            plot_size=plot_size,
            track_alive=True,
        )

    def update(self, state: SwarmState):
        """
        Update visualization with boundary information.

        Args:
            state: Current SwarmState
        """
        # Convert tensors to numpy
        x_np = state.x.detach().cpu().numpy()
        v_np = state.v.detach().cpu().numpy()

        # Check which walkers are in bounds
        in_bounds = self.bounds.contains(state.x)
        in_bounds_np = in_bounds.cpu().numpy()
        n_alive = in_bounds.sum().item()

        # Create position data with colors
        colors = [self.in_bounds_color if ib else self.out_bounds_color for ib in in_bounds_np]

        pos_data = pd.DataFrame({
            "x": x_np[:, 0],
            "y": x_np[:, 1] if x_np.shape[1] > 1 else x_np[:, 0],
            "color": colors,
        })

        # Update scatter with colors
        self.position_stream.send(pos_data)
        # Update color column mapping
        self.position_stream.plot = self.position_stream.plot.opts(
            color="color",
            cmap={
                self.in_bounds_color: self.in_bounds_color,
                self.out_bounds_color: self.out_bounds_color,
            },
        )

        # Update velocities (only for in-bounds walkers)
        in_bounds_mask = in_bounds_np
        vel_data = pd.DataFrame({
            "x0": x_np[in_bounds_mask, 0],
            "y0": x_np[in_bounds_mask, 1] if x_np.shape[1] > 1 else x_np[in_bounds_mask, 0],
            "x1": x_np[in_bounds_mask, 0] + v_np[in_bounds_mask, 0] * self.velocity_scale,
            "y1": (x_np[in_bounds_mask, 1] if x_np.shape[1] > 1 else x_np[in_bounds_mask, 0])
            + (v_np[in_bounds_mask, 1] if v_np.shape[1] > 1 else v_np[in_bounds_mask, 0])
            * self.velocity_scale,
        })
        self.velocity_stream.send(vel_data)

        # Update alive count
        # Calculate number of cloned walkers (dead/out-of-bounds)
        n_cloned = state.N - n_alive

        # Send only the NEW data point to Buffer (not the full history)
        alive_point = pd.DataFrame({"step": [self.step], "alive": [n_alive]})
        self.alive_stream.send(alive_point)

        cloned_point = pd.DataFrame({"step": [self.step], "cloned": [n_cloned]})
        self.cloned_stream.send(cloned_point)

        # Keep history for reference
        self.alive_history.append({"step": self.step, "alive": n_alive})
        self.cloned_history.append({"step": self.step, "cloned": n_cloned})

        self.step += 1
