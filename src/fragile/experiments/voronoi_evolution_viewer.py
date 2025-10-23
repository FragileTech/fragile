"""Interactive Voronoi tessellation evolution viewer.

This module provides an interactive dashboard for visualizing how Voronoi
tessellations evolve during EuclideanGas simulations. It combines:
- Gas configuration panel (left) for parameter control
- Voronoi tessellation plot (center) showing spatial structure
- Playback controls (bottom) with play/pause, time slider, and speed control
- Statistics panel (right) showing tessellation metrics and Ricci scalars

The visualization uses the new ScutoidHistory API to compute tessellations
from RunHistory data.

Features:
- Watch tessellations evolve in real-time (1-30 fps)
- Manually navigate to specific timesteps
- Toggle visualization options (walkers, edges, centers, fitness coloring)
- View Ricci scalar curvature statistics

Usage:
    python -m fragile.experiments.voronoi_evolution_viewer
"""

from __future__ import annotations

import holoviews as hv
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import panel as pn
from scipy.spatial import Voronoi

from fragile.bounds import TorchBounds
from fragile.core import create_scutoid_history, ScutoidHistory2D
from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.gas_config_dashboard import GasConfig


hv.extension("bokeh")
pn.extension()


class VoronoiEvolutionViewer:
    """Interactive viewer for Voronoi tessellation evolution.

    Displays how the spatial tessellation changes over time during a
    EuclideanGas simulation run.
    """

    def __init__(self, dims: int = 2):
        """Initialize viewer.

        Args:
            dims: Spatial dimension (must be 2 for Voronoi visualization)
        """
        if dims != 2:
            msg = "VoronoiEvolutionViewer currently only supports 2D"
            raise ValueError(msg)

        self.dims = dims
        self.scutoid_history: ScutoidHistory2D | None = None

        # Prepare potential (do this after hv.extension is loaded)
        self.potential = None
        self.background = None
        self.mode_points = None

        # Performance caches
        self._cached_voronoi: dict[int, Voronoi] = {}
        self._cached_polygon_data: dict[int, list] = {}
        self._cached_edges: dict[int, list] = {}
        self._cached_delaunay_edges: dict[int, list] = {}
        self._static_background_plot: hv.Overlay | None = None

        # Create gas configuration dashboard (will init potential in panel())
        self.gas_config = None

        # Time control with play/pause functionality
        self.time_player = pn.widgets.Player(
            name="Time Step",
            start=0,
            end=1,  # Initial value, will be updated after simulation
            value=0,
            step=1,
            interval=200,  # 200ms between frames (5 fps)
            loop_policy="loop",
            disabled=True,
        )
        self.time_player.sizing_mode = "stretch_width"
        self.time_player.param.watch(self._on_time_changed, "value")

        # Speed control
        self.speed_slider = pn.widgets.IntSlider(
            name="Playback Speed (fps)",
            start=1,
            end=30,
            value=5,
            step=1,
        )
        self.speed_slider.param.watch(self._on_speed_changed, "value")

        # Visualization controls
        self.show_walkers = pn.widgets.Checkbox(name="Show Walkers", value=True)
        self.show_voronoi = pn.widgets.Checkbox(name="Show Voronoi Edges", value=True)
        self.show_voronoi_vertices = pn.widgets.Checkbox(name="Show Voronoi Vertices", value=False)
        self.show_delaunay = pn.widgets.Checkbox(name="Show Delaunay Dual", value=False)

        # Color mode radio buttons
        self.color_mode = pn.widgets.RadioButtonGroup(
            name="Color by",
            options=["None", "Fitness", "Ricci Curvature"],
            value="None",
            button_type="default",
        )

        # Bind visualization controls
        for widget in [
            self.show_walkers,
            self.show_voronoi,
            self.show_voronoi_vertices,
            self.show_delaunay,
            self.color_mode,
        ]:
            widget.param.watch(lambda _: self._update_plot(), "value")

        # Main plot pane (will be initialized in panel())
        self.plot_pane = None

        # Statistics pane
        self.stats_pane = pn.pane.Markdown(
            "Run a simulation to see statistics",
            sizing_mode="stretch_width",
        )

    def _initialize_potential(self):
        """Initialize potential and background (call after hv.extension)."""
        if self.potential is None:
            self.potential, self.background, self.mode_points = prepare_benchmark_for_explorer(
                benchmark_name="Mixture of Gaussians",
                dims=self.dims,
                bounds_range=(-6.0, 6.0),
                resolution=100,
                n_gaussians=3,
            )
            # Create bounds matching the visualization domain
            self.bounds = TorchBounds(low=-6.0, high=6.0, shape=(self.dims,))
            self.gas_config = GasConfig(potential=self.potential, dims=self.dims)
            self.gas_config.add_completion_callback(self._on_simulation_complete)

    def _create_empty_plot(self) -> hv.Overlay:
        """Create empty plot with background (cached for performance)."""
        # Return cached static background if available
        if self._static_background_plot is not None:
            return self._static_background_plot

        # Use the background image from prepare_background
        if self.background is None:
            # Return empty plot
            plot = hv.Curve([]).opts(
                width=700, height=700, xlabel="x", ylabel="y", xlim=(-6, 6), ylim=(-6, 6)
            )
            self._static_background_plot = plot
            return plot

        # Background is already an hv.Image - just overlay modes
        plot = self.background

        # Add mode points
        if self.mode_points is not None:
            plot *= self.mode_points

        # Apply axis limits matching the bounds domain
        plot = plot.opts(xlim=(-6, 6), ylim=(-6, 6))

        # Cache the static background
        self._static_background_plot = plot

        return plot

    def _precompute_voronoi_cache(self):
        """Pre-compute and cache Voronoi tessellations for all timesteps.

        This dramatically improves playback performance by eliminating O(N log N)
        Voronoi computation during rendering. Caches:
        - scipy Voronoi objects
        - Polygon vertex coordinates
        - Edge coordinates
        - Delaunay dual edges
        """
        if self.scutoid_history is None:
            return

        print("Pre-computing Voronoi cache for fast playback...")

        for t_idx in range(self.scutoid_history.n_recorded):
            voronoi_cells = self.scutoid_history.voronoi_cells[t_idx]

            # Get walker positions
            positions = np.array([cell.center for cell in voronoi_cells])

            if len(positions) < 3:
                # Too few points for Voronoi
                self._cached_voronoi[t_idx] = None
                self._cached_polygon_data[t_idx] = []
                self._cached_edges[t_idx] = []
                self._cached_delaunay_edges[t_idx] = []
                continue

            # Compute Voronoi diagram once
            try:
                vor = Voronoi(positions)
            except Exception:
                self._cached_voronoi[t_idx] = None
                self._cached_polygon_data[t_idx] = []
                self._cached_edges[t_idx] = []
                self._cached_delaunay_edges[t_idx] = []
                continue

            # Cache the Voronoi object
            self._cached_voronoi[t_idx] = vor

            # Pre-extract polygon data (for Ricci coloring)
            polygon_data = []
            for i in range(len(positions)):
                region_idx = vor.point_region[i]
                region_vertices_indices = vor.regions[region_idx]

                # Only cache bounded regions
                if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
                    polygon_vertices = vor.vertices[region_vertices_indices]
                    # Close the polygon
                    closed_polygon = np.vstack([polygon_vertices, polygon_vertices[0]])
                    polygon_data.append((i, closed_polygon))

            self._cached_polygon_data[t_idx] = polygon_data

            # Pre-extract edge coordinates
            edge_coords = []
            for simplex in vor.ridge_vertices:
                if -1 not in simplex:  # Skip infinite ridges
                    v0 = vor.vertices[simplex[0]]
                    v1 = vor.vertices[simplex[1]]
                    edge_coords.append((v0, v1))

            self._cached_edges[t_idx] = edge_coords

            # Pre-extract Delaunay dual edges
            delaunay_edge_coords = []
            for p0, p1 in vor.ridge_points:
                delaunay_edge_coords.append((positions[p0], positions[p1]))

            self._cached_delaunay_edges[t_idx] = delaunay_edge_coords

        print(f"✓ Cached {len(self._cached_voronoi)} timesteps for fast rendering")

    def _on_simulation_complete(self, history):
        """Handle simulation completion."""
        # Build scutoid tessellation with bounds filtering
        # Only include cells with centers inside the visualization domain [-6, 6]²
        self.scutoid_history = create_scutoid_history(history)

        # Override history bounds with viewer bounds to filter tessellation
        # This ensures only walkers inside the visualization domain are included
        if hasattr(self, "bounds"):
            self.scutoid_history.bounds = self.bounds

        self.scutoid_history.build_tessellation()
        self.scutoid_history.compute_ricci_scalars()

        # Pre-compute Voronoi cache for fast playback
        self._precompute_voronoi_cache()

        # Enable time player
        self.time_player.end = self.scutoid_history.n_recorded - 1
        self.time_player.value = 0
        self.time_player.disabled = False

        # Update display
        self._update_plot()
        self._update_statistics()

    def _on_time_changed(self, _event):
        """Handle time slider change."""
        self._update_plot()
        self._update_statistics()

    def _on_speed_changed(self, event):
        """Handle speed slider change."""
        fps = event.new
        interval_ms = int(1000 / fps)
        self.time_player.interval = interval_ms

    def _get_valid_time_index(self) -> int | None:
        """Return current time index if within bounds, otherwise clamp Player."""
        if self.scutoid_history is None:
            return None

        total_frames = len(self.scutoid_history.voronoi_cells)
        if total_frames == 0:
            return None

        max_idx = total_frames - 1
        current_idx = int(self.time_player.value)

        if current_idx < 0:
            self.time_player.value = 0
            return None
        if current_idx > max_idx:
            self.time_player.value = max_idx
            return None
        return current_idx

    def _update_plot(self):
        """Update the visualization plot."""
        if self.scutoid_history is None or self.plot_pane is None:
            return

        t_idx = self._get_valid_time_index()
        if t_idx is None:
            return
        voronoi_cells = self.scutoid_history.voronoi_cells[t_idx]

        # Start with background
        plot = self._create_empty_plot()

        # Get walker positions
        positions = np.array([cell.center for cell in voronoi_cells])

        if len(positions) < 3:
            # Not enough points for Voronoi
            self.plot_pane.object = plot
            return

        # Get cached Voronoi diagram (or compute if not cached)
        vor = self._cached_voronoi.get(t_idx)
        if vor is None:
            # Fallback: compute on-the-fly if cache missed
            try:
                vor = Voronoi(positions)
            except Exception:
                self.plot_pane.object = plot
                return

        # Get color data based on color mode
        color_mode = self.color_mode.value
        color_data = None
        color_label = None
        cmap = None

        if color_mode == "Fitness" and hasattr(self.scutoid_history.history, "fitness"):
            # Fitness coloring
            fitness_array = self.scutoid_history.history.fitness
            if t_idx > 0 and t_idx - 1 < len(fitness_array):
                fitness_vals = fitness_array[t_idx - 1].detach().cpu().numpy()
                walker_ids = np.array([cell.walker_id for cell in voronoi_cells])
                if fitness_vals.size > 0:
                    max_idx = fitness_vals.shape[0] - 1
                    safe_ids = np.clip(walker_ids, 0, max_idx)
                    color_data = fitness_vals[safe_ids]
                    color_label = "fitness"
                    cmap = "coolwarm"

        elif color_mode == "Ricci Curvature" and self.scutoid_history.ricci_scalars is not None:
            # Ricci curvature coloring
            if t_idx > 0 and t_idx - 1 < len(self.scutoid_history.ricci_scalars):
                ricci_array = self.scutoid_history.ricci_scalars[t_idx - 1]
                walker_ids = np.array([cell.walker_id for cell in voronoi_cells])
                color_data = ricci_array[walker_ids]
                color_label = "ricci"
                cmap = "RdBu_r"  # Diverging colormap: red (negative) to blue (positive)

        # Layer 1: Voronoi cell polygons (filled, colored by Ricci if enabled)
        if color_mode == "Ricci Curvature" and color_data is not None:
            # Use cached polygon data if available
            cached_polygons = self._cached_polygon_data.get(t_idx, [])

            if cached_polygons:
                # Extract polygons and corresponding Ricci values
                polygons_list = []
                ricci_values_list = []

                for i, closed_polygon in cached_polygons:
                    ricci_val = color_data[i]
                    # Skip NaN values
                    if not np.isnan(ricci_val):
                        polygons_list.append(closed_polygon)
                        ricci_values_list.append(ricci_val)

                if polygons_list and ricci_values_list:
                    # Vectorized color mapping
                    ricci_array = np.array(ricci_values_list)
                    ricci_min = np.min(ricci_array)
                    ricci_max = np.max(ricci_array)

                    # Normalize all values at once
                    if ricci_max > ricci_min:
                        norm_values = (ricci_array - ricci_min) / (ricci_max - ricci_min)
                    else:
                        norm_values = np.full_like(ricci_array, 0.5)

                    # Vectorized colormap lookup
                    cmap_obj = mpl.colormaps[cmap]
                    rgba_colors = cmap_obj(norm_values)  # Vectorized!
                    hex_colors = [mcolors.rgb2hex(rgba[:3]) for rgba in rgba_colors]

                    # Create paths with pre-computed colors
                    polygon_paths = [
                        hv.Path([poly]).opts(color=color, alpha=0.6, line_alpha=0)
                        for poly, color in zip(polygons_list, hex_colors)
                    ]

                    # Overlay all polygons
                    if polygon_paths:
                        voronoi_polygons = hv.Overlay(polygon_paths)
                        plot *= voronoi_polygons

        # Layer 2: Voronoi edges
        if self.show_voronoi.value:
            # Use cached edges if available
            edge_points = self._cached_edges.get(t_idx, [])

            if not edge_points:
                # Fallback: extract edges on-the-fly
                for simplex in vor.ridge_vertices:
                    if -1 not in simplex:  # Skip infinite ridges
                        v0 = vor.vertices[simplex[0]]
                        v1 = vor.vertices[simplex[1]]
                        edge_points.append((v0, v1))

            if edge_points:
                # Create path segments
                segments = hv.Overlay([
                    hv.Curve([v0, v1]).opts(color="black", line_width=1.5, alpha=0.7)
                    for v0, v1 in edge_points
                ])
                plot *= segments

        # Layer 3: Delaunay dual (edges between walkers sharing Voronoi edge)
        if self.show_delaunay.value:
            # Use cached Delaunay edges if available
            delaunay_edges = self._cached_delaunay_edges.get(t_idx, [])

            if not delaunay_edges:
                # Fallback: extract on-the-fly
                for p0, p1 in vor.ridge_points:
                    delaunay_edges.append((positions[p0], positions[p1]))

            if delaunay_edges:
                dual_segments = hv.Overlay([
                    hv.Curve([e0, e1]).opts(
                        color="blue", line_width=1.0, alpha=0.5, line_dash="dashed"
                    )
                    for e0, e1 in delaunay_edges
                ])
                plot *= dual_segments

        # Layer 4: Voronoi vertices
        if self.show_voronoi_vertices.value and len(vor.vertices) > 0:
            # Color Voronoi vertices by Ricci if enabled
            if color_mode == "Ricci Curvature" and color_data is not None:
                # For now, just show vertices without coloring (they don't have walker IDs)
                vertices = hv.Points(
                    vor.vertices,
                    label="Voronoi Vertices",
                ).opts(
                    color="orange",
                    size=6,
                    alpha=0.8,
                )
            else:
                vertices = hv.Points(
                    vor.vertices,
                    label="Voronoi Vertices",
                ).opts(
                    color="orange",
                    size=6,
                    alpha=0.8,
                )
            plot *= vertices

        # Layer 5: Walker positions (Delaunay vertices / Voronoi cell centers)
        if self.show_walkers.value:
            if color_data is not None:
                # Color walkers by selected mode
                walkers = hv.Points(
                    (positions[:, 0], positions[:, 1], color_data),
                    vdims=[color_label],
                    label="Walkers",
                ).opts(
                    color=color_label,
                    cmap=cmap,
                    size=10,
                    alpha=0.8,
                    colorbar=True,
                    clim=(np.nanmin(color_data), np.nanmax(color_data)),
                )
            else:
                # No coloring
                walkers = hv.Points(positions, label="Walkers").opts(
                    color="cyan", size=10, alpha=0.8
                )

            plot *= walkers

        # Apply axis limits matching the bounds domain
        plot = plot.opts(xlim=(-6, 6), ylim=(-6, 6))

        self.plot_pane.object = plot

    def _update_statistics(self):
        """Update statistics panel."""
        if self.scutoid_history is None:
            return

        t_idx = self._get_valid_time_index()
        if t_idx is None:
            return
        voronoi_cells = self.scutoid_history.voronoi_cells[t_idx]

        # Basic stats
        n_cells = len(voronoi_cells)
        time_value = self.scutoid_history.timesteps[t_idx]

        stats_md = f"""
### Tessellation Statistics

**Time Step:** {t_idx} / {self.scutoid_history.n_recorded - 1}
**Time Value:** {time_value:.2f}
**Number of Cells:** {n_cells}
"""

        # Cell neighbor statistics
        if n_cells > 0:
            neighbor_counts = [len(cell.neighbors) for cell in voronoi_cells]
            avg_neighbors = np.mean(neighbor_counts)
            max_neighbors = np.max(neighbor_counts)
            min_neighbors = np.min(neighbor_counts)

            stats_md += f"""
**Neighbor Statistics:**
- Average: {avg_neighbors:.2f}
- Min: {min_neighbors}
- Max: {max_neighbors}
"""

        # Scutoid statistics (if not first timestep)
        if (
            t_idx > 0
            and self.scutoid_history.scutoid_cells
            and (t_idx - 1) < len(self.scutoid_history.scutoid_cells)
        ):
            scutoids = self.scutoid_history.scutoid_cells[t_idx - 1]

            n_prisms = sum(1 for s in scutoids if s.is_prism())
            n_scutoids = len(scutoids) - n_prisms

            stats_md += f"""
### Scutoid Cell Statistics

**Total Cells:** {len(scutoids)}
**Prisms:** {n_prisms} ({100 * n_prisms / len(scutoids):.1f}%)
**Scutoids:** {n_scutoids} ({100 * n_scutoids / len(scutoids):.1f}%)
"""

            # Ricci scalar statistics
            if self.scutoid_history.ricci_scalars is not None and (t_idx - 1) < len(
                self.scutoid_history.ricci_scalars
            ):
                ricci_vals = self.scutoid_history.ricci_scalars[t_idx - 1]
                valid_ricci = ricci_vals[~np.isnan(ricci_vals)]

                if len(valid_ricci) > 0:
                    stats_md += f"""
**Ricci Scalar Curvature:**
- Mean: {np.mean(valid_ricci):.4f}
- Std: {np.std(valid_ricci):.4f}
- Min: {np.min(valid_ricci):.4f}
- Max: {np.max(valid_ricci):.4f}
"""

        self.stats_pane.object = stats_md

    def panel(self) -> pn.Template:
        """Create complete dashboard.

        Returns:
            Panel template with all components
        """
        # Initialize potential (now that hv.extension is loaded)
        self._initialize_potential()

        # Initialize plot pane
        if self.plot_pane is None:
            self.plot_pane = pn.pane.HoloViews(
                self._create_empty_plot(),
                sizing_mode="stretch_both",
                min_height=600,
            )

        # Visualization controls
        viz_controls = pn.Card(
            self.show_walkers,
            self.show_voronoi,
            self.show_voronoi_vertices,
            self.show_delaunay,
            pn.pane.Markdown("---"),  # Separator
            self.color_mode,
            title="Visualization Options",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        # Left sidebar
        sidebar = pn.Column(
            self.gas_config.panel(),
            viz_controls,
            sizing_mode="stretch_width",
            width=400,
        )

        # Playback controls
        playback_controls = pn.Card(
            self.time_player,
            self.speed_slider,
            title="Playback Controls",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        # Main content
        main = pn.Column(
            pn.pane.Markdown("## Voronoi Tessellation Evolution"),
            self.plot_pane,
            playback_controls,
            sizing_mode="stretch_both",
        )

        # Right sidebar with statistics
        stats_sidebar = pn.Column(
            pn.pane.Markdown("## Statistics"),
            self.stats_pane,
            sizing_mode="stretch_width",
            width=300,
        )

        # Create template
        template = pn.template.FastListTemplate(
            title="Voronoi Tessellation Evolution Viewer",
            sidebar=[sidebar],
            main=[main],
            sidebar_width=400,
            main_max_width="100%",
        )

        # Add stats as modal or separate column
        template.modal.append(stats_sidebar)

        return template


def create_app():
    """Create the Panel app.

    Returns:
        Panel template ready to serve
    """
    viewer = VoronoiEvolutionViewer(dims=2)
    return viewer.panel()


if __name__ == "__main__":
    # Create and serve the app
    app = create_app()
    app.show(port=5006)
