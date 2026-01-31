"""QFT dashboard with simulation and analysis tabs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.core.benchmarks import prepare_benchmark_for_explorer
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel
from fragile.fractalai.qft import analysis as qft_analysis
from fragile.fractalai.qft.correlator_channels import (
    ChannelConfig,
    ChannelCorrelatorResult,
    compute_all_channels,
    CHANNEL_REGISTRY,
)
from fragile.fractalai.qft.electroweak_channels import (
    ELECTROWEAK_CHANNELS,
    ElectroweakChannelConfig,
    compute_all_electroweak_channels,
)
from fragile.fractalai.qft.plotting import (
    build_all_channels_overlay,
    build_correlation_decay_plot,
    build_correlator_plot,
    build_effective_mass_plot,
    build_effective_mass_plateau_plot,
    build_lyapunov_plot,
    build_mass_spectrum_bar,
    build_window_heatmap,
    build_wilson_histogram_plot,
    build_wilson_timeseries_plot,
)


# Prevent Plotly from probing the system browser during import.
os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

__all__ = ["create_app"]


class SwarmConvergence3D(param.Parameterized):
    """3D swarm convergence viewer using Plotly backend."""

    # Dimension mapping parameters
    x_axis_dim = param.ObjectSelector(
        default="dim_0",
        objects=["dim_0", "dim_1", "dim_2", "mc_time", "euclidean_time"],
        doc="Dimension to map to X axis",
    )
    y_axis_dim = param.ObjectSelector(
        default="dim_1",
        objects=["dim_0", "dim_1", "dim_2", "mc_time", "euclidean_time"],
        doc="Dimension to map to Y axis",
    )
    z_axis_dim = param.ObjectSelector(
        default="dim_2",
        objects=["dim_0", "dim_1", "dim_2", "mc_time", "euclidean_time"],
        doc="Dimension to map to Z axis",
    )
    time_iteration = param.ObjectSelector(
        default="euclidean",
        objects=["monte_carlo", "euclidean", "spatial"],
        doc="Player axis: Monte Carlo time, Euclidean time, or spatial slices",
    )
    mc_time_index = param.Integer(
        default=None,
        bounds=(0, None),
        allow_None=True,
        doc="Monte Carlo slice (recorded step or index) for Euclidean/spatial visualization",
    )
    spatial_iteration_dim = param.ObjectSelector(
        default="dim_0",
        objects=["dim_0", "dim_1", "dim_2"],
        doc="Spatial dimension to iterate over when using spatial slices",
    )
    euclidean_time_dim = param.Integer(
        default=3,
        bounds=(0, 10),
        doc="Spatial dimension index for Euclidean time slices (0-indexed)",
    )
    euclidean_time_bins = param.Integer(
        default=50,
        bounds=(5, 500),
        doc="Number of Euclidean/spatial slices",
    )

    # Appearance parameters
    point_size = param.Number(default=4, bounds=(1, 20), doc="Walker point size")
    point_alpha = param.Number(default=0.85, bounds=(0.05, 1.0), doc="Marker opacity")
    color_metric = param.ObjectSelector(
        default="fitness",
        objects=[
            "constant",
            "fitness",
            "reward",
            "radius",
            "dim_0",
            "dim_1",
            "dim_2",
            "mc_time",
            "euclidean_time",
        ],
        doc="Color encoding for walkers",
    )
    fix_axes = param.Boolean(default=True, doc="Fix axis ranges to bounds extent")
    show_delaunay = param.Boolean(default=False, doc="Show Delaunay neighbor graph")
    line_color = param.Color(default="#2b2b2b", doc="Delaunay line color")
    line_style = param.ObjectSelector(
        default="solid",
        objects=["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"],
        doc="Delaunay line style",
    )
    line_width = param.Number(default=1.2, bounds=(0.1, 8.0), doc="Delaunay line width")
    line_alpha = param.Number(default=0.35, bounds=(0.05, 1.0), doc="Delaunay line alpha")

    def __init__(self, history: RunHistory | None, bounds_extent: float = 10.0, **params):
        super().__init__(**params)
        self.history = history
        self.bounds_extent = float(bounds_extent)

        self._x = None
        self._fitness = None
        self._rewards = None
        self._alive = None
        self._neighbor_edges = None
        self._neighbor_graph_method = None
        self._last_mc_frame = 0
        self._mc_slice_controls = None
        self._spatial_slice_controls = None
        self._time_bin_controls = None
        self._time_distribution_pane = None

        self.time_player = pn.widgets.Player(
            name="frame",
            start=0,
            end=0,
            value=0,
            step=1,
            interval=150,
            loop_policy="loop",
        )
        self.time_player.disabled = True
        self.time_player.sizing_mode = "stretch_width"
        self.time_player.param.watch(self._sync_frame, "value")

        self.status_pane = pn.pane.Markdown(
            "**Status:** Load or run a simulation to view 3D convergence.",
            sizing_mode="stretch_width",
        )

        self.plot_pane = pn.pane.Plotly(
            self._make_figure(0),
            sizing_mode="stretch_width",
            height=720,
        )

        self.param.watch(
            self._refresh_frame,
            [
                "x_axis_dim",
                "y_axis_dim",
                "z_axis_dim",
                "point_size",
                "point_alpha",
                "color_metric",
                "fix_axes",
                "show_delaunay",
                "line_color",
                "line_style",
                "line_width",
                "line_alpha",
                "euclidean_time_dim",
                "euclidean_time_bins",
                "mc_time_index",
                "spatial_iteration_dim",
            ],
        )
        self.param.watch(self._on_time_iteration_change, ["time_iteration", "euclidean_time_bins"])

        if history is not None:
            self.set_history(history)

    def set_history(self, history: RunHistory):
        """Load RunHistory data into the viewer."""
        self.history = history
        self._x = history.x_final.detach().cpu().numpy()
        self._fitness = history.fitness.detach().cpu().numpy()
        self._rewards = history.rewards.detach().cpu().numpy()
        self._alive = history.alive_mask.detach().cpu().numpy().astype(bool)
        self._neighbor_edges = history.neighbor_edges
        if history.params is not None:
            self._neighbor_graph_method = history.params.get("neighbor_graph", {}).get("method")

        # Update dimension options based on history.d
        d = history.d
        dim_options = [f"dim_{i}" for i in range(d)] + ["mc_time", "euclidean_time"]

        # Update parameter objects dynamically
        self.param.x_axis_dim.objects = dim_options
        self.param.y_axis_dim.objects = dim_options
        self.param.z_axis_dim.objects = dim_options
        spatial_options = [f"dim_{i}" for i in range(d)]
        time_dim_idx = self._resolve_euclidean_dim(d)
        time_dim_label = f"dim_{time_dim_idx}"
        if d > 3 and time_dim_label in spatial_options:
            spatial_options.remove(time_dim_label)
        if not spatial_options:
            spatial_options = [f"dim_{i}" for i in range(d)]
        self.param.spatial_iteration_dim.objects = spatial_options

        # For color, include all dimensions plus existing metrics
        color_options = dim_options + ["fitness", "reward", "radius", "constant"]
        self.param.color_metric.objects = color_options

        # Reset to defaults if current selection no longer valid
        if self.x_axis_dim not in dim_options:
            self.x_axis_dim = "dim_0"
        if self.y_axis_dim not in dim_options:
            self.y_axis_dim = "dim_1" if d > 1 else "dim_0"
        if self.z_axis_dim not in dim_options:
            self.z_axis_dim = "dim_2" if d > 2 else "dim_0"
        if self.color_metric not in color_options:
            self.color_metric = "fitness"
        if self.spatial_iteration_dim not in spatial_options:
            self.spatial_iteration_dim = spatial_options[0]

        self._last_mc_frame = max(0, history.n_recorded - 1)
        self.time_player.disabled = False
        self._update_time_player_range(reset_value=True)

        self.status_pane.object = (
            f"**RunHistory loaded:** N={history.N}, "
            f"steps={history.n_steps}, "
            f"recorded={history.n_recorded}, "
            f"dims={d}"
        )
        if self._neighbor_edges is None:
            self.status_pane.object += (
                " | Neighbor graph not recorded (rerun with neighbor_graph_method='delaunay' and "
                "neighbor_graph_record=True to show edges)"
            )
        elif self._neighbor_graph_method and self._neighbor_graph_method != "delaunay":
            self.status_pane.object += (
                f" | Neighbor graph method: {self._neighbor_graph_method}"
            )
        self._refresh_frame()

    def _sync_frame(self, event):
        if self.history is None:
            return
        if self.time_iteration == "monte_carlo":
            self._last_mc_frame = int(np.clip(event.new, 0, self.history.n_recorded - 1))
        self._update_plot(int(event.new))

    def _refresh_frame(self, *_):
        if self.history is None:
            return
        self._update_time_distribution()
        self._update_plot(int(self.time_player.value))

    def _on_time_iteration_change(self, event):
        if self.history is None:
            return
        if event.name == "time_iteration":
            if event.new in {"euclidean", "spatial"}:
                self._last_mc_frame = int(
                    np.clip(self.time_player.value, 0, self.history.n_recorded - 1)
                )
                self._update_time_player_range(reset_value=True)
            elif event.new == "monte_carlo":
                self._update_time_player_range(reset_value=False)
                self.time_player.value = int(
                    np.clip(self._last_mc_frame, 0, self.history.n_recorded - 1)
                )
        elif (
            event.name == "euclidean_time_bins"
            and self.time_iteration in {"euclidean", "spatial"}
        ):
            self._update_time_player_range(reset_value=False)
        self._sync_mc_slice_visibility()
        self._sync_spatial_slice_visibility()
        self._sync_time_bin_visibility()
        self._refresh_frame()

    def _update_time_player_range(self, reset_value: bool) -> None:
        if self.history is None:
            self.time_player.start = 0
            self.time_player.end = 0
            self.time_player.value = 0
            return
        if self.time_iteration == "monte_carlo":
            end = max(0, self.history.n_recorded - 1)
            self.time_player.start = 0
            self.time_player.end = end
            if reset_value:
                self.time_player.value = 0
            else:
                self.time_player.value = int(np.clip(self.time_player.value, 0, end))
        else:
            bins = max(1, int(self.euclidean_time_bins))
            end = max(0, bins - 1)
            self.time_player.start = 0
            self.time_player.end = end
            if reset_value:
                self.time_player.value = 0
            else:
                self.time_player.value = int(np.clip(self.time_player.value, 0, end))

    def _sync_mc_slice_visibility(self) -> None:
        if self._mc_slice_controls is None:
            return
        self._mc_slice_controls.visible = self.time_iteration in {"euclidean", "spatial"}

    def _sync_spatial_slice_visibility(self) -> None:
        if self._spatial_slice_controls is None:
            return
        self._spatial_slice_controls.visible = self.time_iteration == "spatial"

    def _sync_time_bin_visibility(self) -> None:
        if self._time_bin_controls is None:
            return
        self._time_bin_controls.visible = self.time_iteration in {"euclidean", "spatial"}
        if self._time_distribution_pane is not None:
            self._time_distribution_pane.visible = self._time_bin_controls.visible

    def _build_time_distribution(self):
        if self.history is None or self._x is None:
            return hv.Text(0, 0, "No history loaded").opts(height=160)

        mc_frame = self._resolve_mc_frame(int(self.time_player.value))
        mc_frame = int(np.clip(mc_frame, 0, self._x.shape[0] - 1))
        positions = self._x[mc_frame]
        if positions.size == 0:
            return hv.Text(0, 0, "No samples").opts(height=160)

        dim_idx = self._resolve_euclidean_dim(positions.shape[1])
        values = positions[:, dim_idx]
        bins = max(1, int(self.euclidean_time_bins))
        extent = float(self.bounds_extent)
        if np.isfinite(extent) and extent > 0:
            hist_range = (-extent, extent)
        else:
            hist_range = (float(np.min(values)), float(np.max(values)))
        counts, edges = np.histogram(values, bins=bins, range=hist_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bars = hv.Bars((centers, counts), kdims=["t"], vdims=["count"]).opts(
            height=160,
            width=420,
            color="#4c78a8",
            line_color="#2a4a6d",
            xlabel="Euclidean time",
            ylabel="samples",
            title="Euclidean time samples",
        )
        if self.time_iteration in {"euclidean", "spatial"} and bins > 0:
            idx = int(np.clip(self.time_player.value, 0, bins - 1))
            low = edges[idx]
            high = edges[idx + 1]
            span = hv.VSpan(low, high).opts(color="#f28e2b", alpha=0.25)
            return bars * span
        return bars

    def _update_time_distribution(self) -> None:
        if self._time_distribution_pane is None:
            return
        self._time_distribution_pane.object = self._build_time_distribution()

    def _get_alive_mask(self, frame: int) -> np.ndarray:
        if self._alive is None:
            return np.ones(self._x.shape[1], dtype=bool)
        if frame == 0:
            return np.ones(self._x.shape[1], dtype=bool)
        idx = min(frame - 1, self._alive.shape[0] - 1)
        return self._alive[idx]

    def _resolve_mc_frame(self, frame: int) -> int:
        if self.history is None:
            return 0
        if self.time_iteration == "monte_carlo":
            return int(np.clip(frame, 0, self.history.n_recorded - 1))
        return self._resolve_mc_time_index()

    def _resolve_mc_time_index(self) -> int:
        if self.history is None:
            return 0
        n_recorded = max(1, int(self.history.n_recorded))
        if self.mc_time_index is None:
            return n_recorded - 1
        try:
            raw = int(self.mc_time_index)
        except (TypeError, ValueError):
            return n_recorded - 1
        if raw in self.history.recorded_steps:
            try:
                resolved = self.history.get_step_index(raw)
            except ValueError:
                resolved = raw
        else:
            resolved = raw
        if resolved < 0 or resolved >= n_recorded:
            return n_recorded - 1
        return resolved

    def _resolve_spatial_dim(self, d: int) -> int:
        if isinstance(self.spatial_iteration_dim, str) and self.spatial_iteration_dim.startswith(
            "dim_"
        ):
            try:
                dim_idx = int(self.spatial_iteration_dim.split("_")[1])
            except (TypeError, ValueError):
                dim_idx = 0
        else:
            dim_idx = 0
        if dim_idx >= d:
            return max(0, d - 1)
        if dim_idx < 0:
            return 0
        return dim_idx

    def _resolve_euclidean_dim(self, d: int) -> int:
        dim = int(self.euclidean_time_dim)
        if dim >= d:
            return max(0, d - 1)
        if dim < 0:
            return 0
        return dim

    def _get_slice_mask(
        self, positions_all: np.ndarray, slice_index: int, dim_idx: int
    ) -> tuple[np.ndarray, tuple[float, float], int]:
        n = positions_all.shape[0]
        bins = max(1, int(self.euclidean_time_bins))
        idx = int(np.clip(slice_index, 0, bins - 1))
        extent = float(self.bounds_extent)
        edges = np.linspace(-extent, extent, bins + 1)
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if positions_all.shape[1] == 0:
            return np.zeros(n, dtype=bool), (low, high), dim_idx
        values = positions_all[:, dim_idx]
        mask = (values >= low) & (values < high)
        return mask, (low, high), dim_idx

    def _frame_title(
        self,
        mc_frame: int,
        slice_index: int | None = None,
        slice_bounds: tuple[float, float] | None = None,
        slice_dim: int | None = None,
        slice_mode: str | None = None,
    ) -> str:
        if self.history is None:
            return "QFT Swarm Convergence"
        if not self.history.recorded_steps:
            return "QFT Swarm Convergence"
        safe_idx = int(np.clip(mc_frame, 0, len(self.history.recorded_steps) - 1))
        step = self.history.recorded_steps[safe_idx]
        if slice_index is None:
            return f"QFT Swarm Convergence (frame {mc_frame}, step {step})"
        label = None
        if slice_dim is not None:
            label = self._axis_label(f"dim_{slice_dim}")
        if slice_bounds is None:
            return (
                f"QFT Swarm Convergence (frame {mc_frame}, step {step}, slice {slice_index})"
            )
        low, high = slice_bounds
        label_text = f"{label} in [{low:.2f}, {high:.2f})" if label else (
            f"time in [{low:.2f}, {high:.2f})"
        )
        mode = f", {slice_mode}" if slice_mode else ""
        return (
            f"QFT Swarm Convergence (frame {mc_frame}, step {step}, slice {slice_index}{mode}, "
            f"{label_text})"
        )

    def _get_delaunay_edges(self, frame: int) -> np.ndarray | None:
        if self.history is None or self._neighbor_edges is None:
            return None
        if frame < 0 or frame >= len(self._neighbor_edges):
            return None
        edges = self._neighbor_edges[frame]
        if edges is None:
            return None
        if isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
        else:
            edges = np.asarray(edges)
        if edges.size == 0:
            return None
        edges = edges.reshape(-1, 2)
        edges = edges[edges[:, 0] != edges[:, 1]]
        if edges.size == 0:
            return None
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    @staticmethod
    def _rgba_from_color(color: str, alpha: float) -> str:
        if color is None:
            return f"rgba(0, 0, 0, {alpha})"
        color = color.strip()
        if color.startswith("rgba(") and color.endswith(")"):
            parts = color[5:-1].split(",")
            if len(parts) >= 3:
                r, g, b = (p.strip() for p in parts[:3])
                return f"rgba({r}, {g}, {b}, {alpha})"
        if color.startswith("rgb(") and color.endswith(")"):
            parts = color[4:-1].split(",")
            if len(parts) >= 3:
                r, g, b = (p.strip() for p in parts[:3])
                return f"rgba({r}, {g}, {b}, {alpha})"
        if color.startswith("#"):
            hex_value = color[1:]
            if len(hex_value) == 3:
                hex_value = "".join([c * 2 for c in hex_value])
            if len(hex_value) == 6:
                r = int(hex_value[0:2], 16)
                g = int(hex_value[2:4], 16)
                b = int(hex_value[4:6], 16)
                return f"rgba({r}, {g}, {b}, {alpha})"
        return color

    def _extract_dimension(
        self, dim_spec: str, frame: int, positions_all: np.ndarray, alive: np.ndarray
    ) -> np.ndarray:
        """Extract coordinate values based on dimension specification.

        Args:
            dim_spec: Dimension specifier ("dim_0", "dim_1", ..., "mc_time")
            frame: Current frame index
            positions_all: All walker positions [N, d]
            alive: Alive mask [N]

        Returns:
            Coordinate values for alive walkers [N_alive]
        """
        if dim_spec == "mc_time":
            # All walkers get same MC time value (frame index)
            n_alive = alive.sum()
            return np.full(n_alive, frame, dtype=float)
        if dim_spec == "euclidean_time":
            dim_idx = self._resolve_euclidean_dim(positions_all.shape[1])
            if dim_idx >= positions_all.shape[1]:
                return np.zeros(alive.sum())
            return positions_all[alive, dim_idx]

        elif dim_spec.startswith("dim_"):
            # Extract spatial dimension
            dim_idx = int(dim_spec.split("_")[1])
            if dim_idx >= positions_all.shape[1]:
                # Dimension not available, return zeros
                return np.zeros(alive.sum())
            return positions_all[alive, dim_idx]

        else:
            # Invalid spec, return zeros
            return np.zeros(alive.sum())

    def _get_color_values(self, frame: int, positions_all: np.ndarray, alive: np.ndarray):
        """Extract color values based on color_metric selection.

        Returns:
            (colors, showscale, colorbar) tuple
        """
        metric = self.color_metric

        # Handle dimension-based coloring
        if metric.startswith("dim_"):
            dim_idx = int(metric.split("_")[1])
            if dim_idx < positions_all.shape[1]:
                colors = positions_all[alive, dim_idx]
            else:
                colors = np.zeros(alive.sum())
            return colors, True, {"title": self._axis_label(metric)}

        if metric == "euclidean_time":
            dim_idx = self._resolve_euclidean_dim(positions_all.shape[1])
            if dim_idx < positions_all.shape[1]:
                colors = positions_all[alive, dim_idx]
            else:
                colors = np.zeros(alive.sum())
            return colors, True, {"title": "Euclidean Time"}

        elif metric == "mc_time":
            # Color by MC time (constant for this frame)
            colors = np.full(alive.sum(), frame, dtype=float)
            return colors, True, {"title": "MC Time (frame)"}

        # Existing metrics
        elif metric == "fitness":
            if frame == 0 or self._fitness is None:
                colors = "#1f77b4"
                return colors, False, None
            idx = min(frame - 1, len(self._fitness) - 1)
            colors = self._fitness[idx][alive]
            return colors, True, {"title": "Fitness"}

        elif metric == "reward":
            if frame == 0 or self._rewards is None:
                colors = "#1f77b4"
                return colors, False, None
            idx = min(frame - 1, len(self._rewards) - 1)
            colors = self._rewards[idx][alive]
            return colors, True, {"title": "Reward"}

        elif metric == "radius":
            # Compute radius from original positions (first 3 dims)
            positions_filtered = positions_all[alive][:, : min(3, positions_all.shape[1])]
            colors = np.linalg.norm(positions_filtered, axis=1)
            return colors, True, {"title": "Radius"}

        else:  # "constant"
            return "#1f77b4", False, None

    def _get_axis_ranges(self, frame: int):
        """Determine axis ranges based on dimension mappings."""

        def get_range(dim_spec: str):
            if dim_spec == "mc_time":
                return [0, len(self._x) - 1]
            if dim_spec == "euclidean_time":
                extent = self.bounds_extent
                return [-extent, extent]
            elif dim_spec.startswith("dim_"):
                # Use bounds extent for spatial dimensions
                extent = self.bounds_extent
                return [-extent, extent]
            else:
                return [-10, 10]  # Default

        return {
            "xaxis": {"range": get_range(self.x_axis_dim)},
            "yaxis": {"range": get_range(self.y_axis_dim)},
            "zaxis": {"range": get_range(self.z_axis_dim)},
        }

    def _axis_label(self, dim_spec: str) -> str:
        """Generate axis label from dimension specification."""
        if dim_spec == "mc_time":
            return "Monte Carlo Time (frame)"
        if dim_spec == "euclidean_time":
            dim_idx = 0
            if self.history is not None:
                dim_idx = self._resolve_euclidean_dim(self.history.d)
            return f"Euclidean Time (dim_{dim_idx})"
        elif dim_spec.startswith("dim_"):
            dim_idx = int(dim_spec.split("_")[1])
            labels = ["X", "Y", "Z", "T"]
            if dim_idx < len(labels):
                return f"Dimension {dim_idx} ({labels[dim_idx]})"
            return f"Dimension {dim_idx}"
        return dim_spec

    def _build_delaunay_trace_mapped(
        self,
        frame: int,
        positions_all: np.ndarray,
        alive: np.ndarray,
        positions_mapped: np.ndarray,
    ):
        """Build Delaunay edges using mapped 3D coordinates."""
        import plotly.graph_objects as go

        # Get edges from history
        edges = self._get_delaunay_edges(frame)
        if edges is None or positions_mapped.size == 0:
            return None

        # Filter edges to alive walkers only
        alive_indices = np.where(alive)[0]
        alive_set = set(alive_indices)

        valid_edges = []
        for i, j in edges:
            if i in alive_set and j in alive_set and i != j:
                valid_edges.append((i, j))

        if not valid_edges:
            return None

        # Build edge coordinates using mapped positions
        x_edges, y_edges, z_edges = [], [], []
        for i, j in valid_edges:
            i_local = np.where(alive_indices == i)[0][0]
            j_local = np.where(alive_indices == j)[0][0]

            x_edges.extend([positions_mapped[i_local, 0], positions_mapped[j_local, 0], None])
            y_edges.extend([positions_mapped[i_local, 1], positions_mapped[j_local, 1], None])
            z_edges.extend([positions_mapped[i_local, 2], positions_mapped[j_local, 2], None])

        line_color = self._rgba_from_color(self.line_color, float(self.line_alpha))
        return go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line={
                "color": line_color,
                "width": float(self.line_width),
                "dash": self.line_style,
            },
            hoverinfo="skip",
            name="Delaunay edges",
            showlegend=False,
        )

    def _build_delaunay_trace(
        self, frame: int, positions: np.ndarray, alive_mask: np.ndarray
    ):
        import plotly.graph_objects as go

        edges = self._get_delaunay_edges(frame)
        if edges is None or positions.size == 0:
            return None
        if alive_mask is not None and alive_mask.size:
            mask = alive_mask[edges[:, 0]] & alive_mask[edges[:, 1]]
            edges = edges[mask]
        if edges.size == 0:
            return None
        n = positions.shape[0]
        mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < n) & (edges[:, 1] < n)
        edges = edges[mask]
        if edges.size == 0:
            return None

        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for i, j in edges:
            p0 = positions[i]
            p1 = positions[j]
            xs.extend([float(p0[0]), float(p1[0]), None])
            ys.extend([float(p0[1]), float(p1[1]), None])
            zs.extend([float(p0[2]), float(p1[2]), None])

        line_color = self._rgba_from_color(self.line_color, float(self.line_alpha))
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={
                "color": line_color,
                "width": float(self.line_width),
                "dash": self.line_style,
            },
            hoverinfo="skip",
            name="Delaunay edges",
            showlegend=False,
        )

    def _make_figure(self, frame: int):
        import plotly.graph_objects as go

        if self.history is None or self._x is None:
            fig = go.Figure()
            fig.update_layout(
                title="QFT Swarm Convergence",
                height=720,
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
            )
            return fig

        mc_frame = self._resolve_mc_frame(frame)
        positions_all = self._x[mc_frame]
        alive = self._get_alive_mask(mc_frame)
        slice_index = None
        slice_bounds = None
        slice_dim = None
        slice_mode = None
        if self.time_iteration in {"euclidean", "spatial"}:
            slice_index = int(np.clip(frame, 0, max(0, int(self.euclidean_time_bins) - 1)))
            if self.time_iteration == "euclidean":
                slice_dim = self._resolve_euclidean_dim(positions_all.shape[1])
                slice_mode = "euclidean"
            else:
                slice_dim = self._resolve_spatial_dim(positions_all.shape[1])
                slice_mode = "spatial"
            slice_mask, slice_bounds, slice_dim = self._get_slice_mask(
                positions_all, slice_index, slice_dim
            )
            alive = alive & slice_mask
        n_alive = alive.sum()

        if n_alive == 0:
            fig = go.Figure()
            fig.update_layout(
                title=self._frame_title(
                    mc_frame,
                    slice_index=slice_index,
                    slice_bounds=slice_bounds,
                    slice_dim=slice_dim,
                    slice_mode=slice_mode,
                ),
                height=720,
            )
            return fig

        # Extract coordinates based on dimension mapping
        x_coords = self._extract_dimension(
            self.x_axis_dim, mc_frame, positions_all, alive
        )
        y_coords = self._extract_dimension(
            self.y_axis_dim, mc_frame, positions_all, alive
        )
        z_coords = self._extract_dimension(
            self.z_axis_dim, mc_frame, positions_all, alive
        )

        # Extract color values
        colors, showscale, colorbar = self._get_color_values(
            mc_frame, positions_all, alive
        )

        # Create scatter trace
        scatter = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            marker={
                "size": self.point_size,
                "color": colors,
                "colorscale": "Viridis" if showscale else None,
                "opacity": self.point_alpha,
                "showscale": showscale,
                "colorbar": colorbar,
            },
            hovertemplate=(
                f"X ({self.x_axis_dim}): %{{x:.3f}}<br>"
                f"Y ({self.y_axis_dim}): %{{y:.3f}}<br>"
                f"Z ({self.z_axis_dim}): %{{z:.3f}}<br>"
                "<extra></extra>"
            ),
        )

        # Add Delaunay edges if enabled
        traces = []
        if self.show_delaunay:
            # Build mapped positions for edge rendering
            positions_mapped = np.column_stack([x_coords, y_coords, z_coords])
            line_trace = self._build_delaunay_trace_mapped(
                mc_frame, positions_all, alive, positions_mapped
            )
            if line_trace is not None:
                traces.append(line_trace)
        traces.append(scatter)

        # Create figure
        fig = go.Figure(data=traces)

        # Update layout with dimension-aware axis labels
        fig.update_layout(
            title=self._frame_title(
                mc_frame,
                slice_index=slice_index,
                slice_bounds=slice_bounds,
                slice_dim=slice_dim,
                slice_mode=slice_mode,
            ),
            height=720,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            scene={
                "xaxis": {"title": self._axis_label(self.x_axis_dim)},
                "yaxis": {"title": self._axis_label(self.y_axis_dim)},
                "zaxis": {"title": self._axis_label(self.z_axis_dim)},
                "aspectmode": "cube" if self.fix_axes else "auto",
            },
        )

        # Set axis ranges
        if self.fix_axes:
            axis_ranges = self._get_axis_ranges(frame)
            fig.update_layout(scene=axis_ranges)

        return fig

    def _update_plot(self, frame: int):
        self.plot_pane.object = self._make_figure(frame)

    def panel(self) -> pn.Column:
        """Return the Panel layout for the 3D convergence viewer."""
        # Row 1: Dimension mapping and point appearance
        controls_row1 = pn.Param(
            self,
            parameters=[
                "x_axis_dim",
                "y_axis_dim",
                "z_axis_dim",
                "point_size",
                "point_alpha",
                "color_metric",
            ],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "x_axis_dim": {"type": pn.widgets.Select, "name": "X Axis"},
                "y_axis_dim": {"type": pn.widgets.Select, "name": "Y Axis"},
                "z_axis_dim": {"type": pn.widgets.Select, "name": "Z Axis"},
                "color_metric": {"type": pn.widgets.Select, "name": "Color By"},
            },
        )

        # Row 2: Axis settings and Delaunay graph
        controls_row2 = pn.Param(
            self,
            parameters=[
                "fix_axes",
                "show_delaunay",
                "line_color",
                "line_style",
                "line_width",
                "line_alpha",
            ],
            sizing_mode="stretch_width",
            show_name=False,
        )

        dimension_info = pn.pane.Alert(
            """
            **Dimension Mapping:** Map spatial dimensions (dim_0, dim_1, dim_2, dim_3), Euclidean time, or Monte Carlo time to plot axes.
            - **Spatial dims**: Position coordinates from simulation
            - **MC time**: Current frame index (useful for temporal evolution visualization)
            - **Player mode**: Choose Monte Carlo, Euclidean, or spatial slices
            """,
            alert_type="info",
            sizing_mode="stretch_width",
        )
        time_toggle = pn.widgets.RadioButtonGroup(
            name="Iterate by",
            options={
                "Monte Carlo": "monte_carlo",
                "Euclidean": "euclidean",
                "Spatial": "spatial",
            },
            value=self.time_iteration,
            sizing_mode="stretch_width",
        )
        time_toggle.param.watch(
            lambda e: setattr(self, "time_iteration", e.new), "value"
        )
        self.param.watch(
            lambda e, widget_ref=time_toggle: (
                None
                if widget_ref.value == e.new
                else setattr(widget_ref, "value", e.new)
            ),
            "time_iteration",
        )
        mc_slice_controls = pn.Param(
            self,
            parameters=["mc_time_index"],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "mc_time_index": {
                    "type": pn.widgets.IntInput,
                    "name": "MC slice (step or idx; blank=last)",
                }
            },
        )
        time_bin_controls = pn.Param(
            self,
            parameters=["euclidean_time_dim", "euclidean_time_bins"],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "euclidean_time_dim": {
                    "type": pn.widgets.IntInput,
                    "name": "Euclidean time dim (0-indexed)",
                },
                "euclidean_time_bins": {
                    "type": pn.widgets.IntSlider,
                    "name": "Time bins (Euclidean/Spatial)",
                    "start": 5,
                    "end": 500,
                    "step": 1,
                }
            },
        )
        spatial_slice_controls = pn.Param(
            self,
            parameters=["spatial_iteration_dim"],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "spatial_iteration_dim": {
                    "type": pn.widgets.Select,
                    "name": "Slice dimension",
                }
            },
        )
        self._mc_slice_controls = mc_slice_controls
        self._spatial_slice_controls = spatial_slice_controls
        self._time_bin_controls = time_bin_controls
        self._time_distribution_pane = pn.pane.HoloViews(
            self._build_time_distribution(),
            sizing_mode="stretch_width",
            height=180,
        )
        self._sync_mc_slice_visibility()
        self._sync_spatial_slice_visibility()
        self._sync_time_bin_visibility()

        return pn.Column(
            pn.pane.Markdown("## 3D Swarm Convergence (Plotly)"),
            dimension_info,
            time_toggle,
            mc_slice_controls,
            time_bin_controls,
            self._time_distribution_pane,
            spatial_slice_controls,
            self.time_player,
            pn.Spacer(height=10),
            pn.Row(controls_row1, controls_row2, sizing_mode="stretch_width"),
            pn.layout.Divider(),
            self.plot_pane,
            self.status_pane,
            sizing_mode="stretch_width",
        )


class AnalysisSettings(param.Parameterized):
    analysis_time_index = param.Integer(default=None, bounds=(0, None), allow_None=True)
    analysis_step = param.Integer(default=None, bounds=(0, None), allow_None=True)
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    correlation_r_max = param.Number(default=0.5, bounds=(1e-6, None))
    correlation_bins = param.Integer(default=50, bounds=(1, None))
    gradient_neighbors = param.Integer(default=5, bounds=(1, None))
    build_fractal_set = param.Boolean(default=False)
    fractal_set_stride = param.Integer(default=10, bounds=(1, None))

    use_local_fields = param.Boolean(default=False)
    use_connected = param.Boolean(default=False)
    density_sigma = param.Number(default=0.5, bounds=(1e-6, None))

    compute_particles = param.Boolean(default=False)
    particle_operators = param.String(default="baryon,meson,glueball")
    particle_max_lag = param.Integer(default=80, bounds=(1, None), allow_None=True)
    particle_fit_start = param.Integer(default=7, bounds=(0, None))
    particle_fit_stop = param.Integer(default=16, bounds=(0, None))
    particle_fit_mode = param.ObjectSelector(
        default="window",
        objects=["window", "plateau", "auto"],
    )
    particle_plateau_min_points = param.Integer(default=3, bounds=(1, None))
    particle_plateau_max_points = param.Integer(default=None, bounds=(1, None), allow_None=True)
    particle_plateau_max_cv = param.Number(default=0.2, bounds=(1e-6, None), allow_None=True)
    particle_mass = param.Number(default=1.0, bounds=(1e-6, None))
    particle_ell0 = param.Number(default=None, bounds=(1e-6, None), allow_None=True)
    particle_use_connected = param.Boolean(default=True)
    particle_neighbor_method = param.ObjectSelector(default="voronoi", objects=["companion", "knn", "voronoi"])
    particle_knn_k = param.Integer(default=4, bounds=(1, None))
    particle_knn_sample = param.Integer(default=512, bounds=(1, None), allow_None=True)
    particle_meson_reduce = param.ObjectSelector(default="mean", objects=["mean", "first"])
    particle_baryon_pairs = param.Integer(default=None, bounds=(1, None), allow_None=True)
    # Voronoi-specific parameters
    particle_voronoi_weight = param.ObjectSelector(default="facet_area", objects=["facet_area", "volume", "combined"])
    particle_voronoi_pbc_mode = param.ObjectSelector(default="mirror", objects=["mirror", "replicate", "ignore"])
    particle_voronoi_normalize = param.Boolean(default=True)
    particle_voronoi_max_triplets = param.Integer(default=100, bounds=(1, None), allow_None=True)
    particle_voronoi_exclude_boundary = param.Boolean(default=True)
    particle_voronoi_boundary_tolerance = param.Number(default=1e-6, bounds=(1e-9, 1e-3))
    
    # Curvature proxy parameters (computed automatically when Voronoi is active)
    compute_curvature_proxies = param.Boolean(
        default=True,
        doc="Compute fast O(N) curvature proxies from Voronoi geometry (volume variance, Graph Laplacian, Raychaudhuri)",
    )
    curvature_compute_interval = param.Integer(
        default=1,
        bounds=(1, None),
        doc="Compute curvature every N timesteps (1=every step, 10=every 10th step)",
    )

    compute_string_tension = param.Boolean(default=False)
    string_tension_max_triangles = param.Integer(default=20000, bounds=(1, None))
    string_tension_bins = param.Integer(default=20, bounds=(2, None))

    def to_cli_args(self, history_path: Path, output_dir: Path, analysis_id: str) -> list[str]:
        args = [
            "analyze_fractal_gas_qft",
            "--history-path",
            str(history_path),
            "--output-dir",
            str(output_dir),
            "--analysis-id",
            analysis_id,
            "--warmup-fraction",
            str(self.warmup_fraction),
            "--h-eff",
            str(self.h_eff),
            "--correlation-r-max",
            str(self.correlation_r_max),
            "--correlation-bins",
            str(self.correlation_bins),
            "--gradient-neighbors",
            str(self.gradient_neighbors),
            "--fractal-set-stride",
            str(self.fractal_set_stride),
            "--density-sigma",
            str(self.density_sigma),
            "--particle-operators",
            self.particle_operators,
            "--particle-fit-start",
            str(self.particle_fit_start),
            "--particle-fit-stop",
            str(self.particle_fit_stop),
            "--particle-fit-mode",
            str(self.particle_fit_mode),
            "--particle-plateau-min-points",
            str(self.particle_plateau_min_points),
            "--particle-mass",
            str(self.particle_mass),
            "--particle-neighbor-method",
            str(self.particle_neighbor_method),
            "--particle-knn-k",
            str(self.particle_knn_k),
            "--particle-meson-reduce",
            str(self.particle_meson_reduce),
            "--particle-voronoi-weight",
            str(self.particle_voronoi_weight),
            "--particle-voronoi-pbc-mode",
            str(self.particle_voronoi_pbc_mode),
            "--particle-voronoi-max-triplets",
            str(self.particle_voronoi_max_triplets),
            "--particle-voronoi-boundary-tolerance",
            str(self.particle_voronoi_boundary_tolerance),
            "--string-tension-max-triangles",
            str(self.string_tension_max_triangles),
            "--string-tension-bins",
            str(self.string_tension_bins),
        ]

        # Add boolean flags
        if self.particle_voronoi_normalize:
            args.append("--particle-voronoi-normalize")
        else:
            args.append("--no-particle-voronoi-normalize")

        if self.particle_voronoi_exclude_boundary:
            args.append("--particle-voronoi-exclude-boundary")
        else:
            args.append("--no-particle-voronoi-exclude-boundary")
        
        if self.compute_curvature_proxies:
            args.append("--compute-curvature-proxies")
        else:
            args.append("--no-compute-curvature-proxies")
        
        if self.curvature_compute_interval != 1:
            args.extend(["--curvature-compute-interval", str(self.curvature_compute_interval)])

        if self.analysis_time_index is not None:
            args.extend(["--analysis-time-index", str(self.analysis_time_index)])
        if self.analysis_step is not None:
            args.extend(["--analysis-step", str(self.analysis_step)])
        if self.build_fractal_set:
            args.append("--build-fractal-set")
        if self.use_local_fields:
            args.append("--use-local-fields")
        if self.use_connected:
            args.append("--use-connected")
        if self.compute_particles:
            args.append("--compute-particles")
        if self.particle_max_lag is not None:
            args.extend(["--particle-max-lag", str(self.particle_max_lag)])
        if self.particle_plateau_max_points is not None:
            args.extend(["--particle-plateau-max-points", str(self.particle_plateau_max_points)])
        if self.particle_plateau_max_cv is not None:
            args.extend(["--particle-plateau-max-cv", str(self.particle_plateau_max_cv)])
        if self.particle_ell0 is not None:
            args.extend(["--particle-ell0", str(self.particle_ell0)])
        if self.particle_use_connected:
            args.append("--particle-use-connected")
        else:
            args.append("--no-particle-use-connected")
        if self.particle_knn_sample is not None:
            args.extend(["--particle-knn-sample", str(self.particle_knn_sample)])
        if self.particle_baryon_pairs is not None:
            args.extend(["--particle-baryon-pairs", str(self.particle_baryon_pairs)])
        if self.compute_string_tension:
            args.append("--compute-string-tension")

        return args


class ChannelSettings(param.Parameterized):
    """Settings for the new Channels tab (independent of old particle analysis)."""

    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.5))
    max_lag = param.Integer(default=80, bounds=(10, 200))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-6, None), allow_None=True)
    neighbor_method = param.ObjectSelector(
        default="recorded",
        objects=("uniform", "knn", "voronoi", "recorded"),
    )
    knn_k = param.Integer(default=4, bounds=(1, 20))
    knn_sample = param.Integer(default=512, bounds=(1, None), allow_None=True)
    use_connected = param.Boolean(default=True)

    # User-friendly time dimension selection
    time_dimension = param.ObjectSelector(
        default="t",
        objects=["t", "x", "y", "z", "monte_carlo"],
        doc=(
            "Time axis for correlator analysis:\n"
            "  - 't': Euclidean time dimension (default, spatial dim 3)\n"
            "  - 'x': X spatial dimension (dim 0)\n"
            "  - 'y': Y spatial dimension (dim 1)\n"
            "  - 'z': Z spatial dimension (dim 2)\n"
            "  - 'monte_carlo': Monte Carlo timesteps (ignores spatial dims)"
        ),
    )
    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc=(
            "Recorded Monte Carlo index or step to use as the 4D slice for Euclidean "
            "analysis. None uses the last recorded slice."
        ),
    )

    # Time axis selection (for 4D Euclidean time analysis)
    time_axis = param.ObjectSelector(
        default="mc",
        objects=("mc", "euclidean"),
        doc="Time axis: 'mc' (Monte Carlo timesteps) or 'euclidean' (spatial dimension as time)",
    )
    euclidean_time_dim = param.Integer(
        default=3,
        bounds=(0, 10),
        doc="Spatial dimension index to use as Euclidean time (0-indexed, default 3 = 4th dimension)",
    )
    euclidean_time_bins = param.Integer(
        default=50,
        bounds=(10, 500),
        doc="Number of time bins for Euclidean time analysis",
    )

    channel_list = param.String(default="scalar,pseudoscalar,vector,nucleon,glueball")
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    scalar_fit_mode = param.ObjectSelector(
        default="default",
        objects=("default", "aic", "linear", "linear_abs"),
    )
    pseudoscalar_fit_mode = param.ObjectSelector(
        default="default",
        objects=("default", "aic", "linear", "linear_abs"),
    )
    vector_fit_mode = param.ObjectSelector(
        default="default",
        objects=("default", "aic", "linear", "linear_abs"),
    )
    nucleon_fit_mode = param.ObjectSelector(
        default="default",
        objects=("default", "aic", "linear", "linear_abs"),
    )
    glueball_fit_mode = param.ObjectSelector(
        default="default",
        objects=("default", "aic", "linear", "linear_abs"),
    )
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))


class ElectroweakSettings(param.Parameterized):
    """Settings for electroweak (U1/SU2) channel correlators."""

    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.5))
    max_lag = param.Integer(default=80, bounds=(10, 200))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    use_connected = param.Boolean(default=True)
    neighbor_method = param.ObjectSelector(
        default="uniform",
        objects=("uniform", "knn", "voronoi"),
    )
    knn_k = param.Integer(default=1, bounds=(1, 20))

    # User-friendly time dimension selection
    time_dimension = param.ObjectSelector(
        default="t",
        objects=["t", "x", "y", "z", "monte_carlo"],
        doc=(
            "Time axis for correlator analysis:\n"
            "  - 't': Euclidean time dimension (default, spatial dim 3)\n"
            "  - 'x': X spatial dimension (dim 0)\n"
            "  - 'y': Y spatial dimension (dim 1)\n"
            "  - 'z': Z spatial dimension (dim 2)\n"
            "  - 'monte_carlo': Monte Carlo timesteps (ignores spatial dims)"
        ),
    )
    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc=(
            "Recorded Monte Carlo index or step to use as the 4D slice for Euclidean "
            "analysis. None uses the last recorded slice."
        ),
    )

    # Time axis selection (for 4D Euclidean time analysis)
    time_axis = param.ObjectSelector(
        default="mc",
        objects=("mc", "euclidean"),
        doc="Time axis: 'mc' (Monte Carlo timesteps) or 'euclidean' (spatial dimension as time)",
    )
    euclidean_time_dim = param.Integer(
        default=3,
        bounds=(0, 10),
        doc="Spatial dimension index to use as Euclidean time (0-indexed, default 3 = 4th dimension)",
    )
    euclidean_time_bins = param.Integer(
        default=50,
        bounds=(10, 500),
        doc="Number of time bins for Euclidean time analysis",
    )

    channel_list = param.String(default=",".join(ELECTROWEAK_CHANNELS))
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))

    # Electroweak parameters (override history params if provided)
    epsilon_d = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    epsilon_c = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    epsilon_clone = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    lambda_alg = param.Number(default=None, bounds=(0.0, None), allow_None=True)


def _parse_window_widths(spec: str) -> list[int]:
    """Parse '5-50' or '5,10,15,20' into list of ints."""
    if "-" in spec and "," not in spec:
        parts = spec.split("-")
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
                return list(range(start, end + 1))
            except ValueError:
                pass
    try:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError:
        return list(range(5, 51))


def _map_time_dimension(time_dimension: str, history_d: int | None = None) -> tuple[str, int]:
    """Map user-friendly time dimension name to backend parameters.

    Args:
        time_dimension: User selection ("t", "x", "y", "z", "monte_carlo")
        history_d: Optional spatial dimensionality for validation/fallback.

    Returns:
        (time_axis, euclidean_time_dim) tuple:
        - time_axis: "mc" or "euclidean"
        - euclidean_time_dim: 0-3 (dimension index, clamped for "t" if needed)
    """
    mapping = {
        "monte_carlo": ("mc", 3),      # MC time, dim doesn't matter
        "x": ("euclidean", 0),         # X spatial dimension
        "y": ("euclidean", 1),         # Y spatial dimension
        "z": ("euclidean", 2),         # Z spatial dimension
        "t": ("euclidean", 3),         # Euclidean time (default)
    }

    if time_dimension not in mapping:
        msg = f"Invalid time_dimension: {time_dimension}. Must be one of {list(mapping.keys())}"
        raise ValueError(msg)

    time_axis, euclidean_time_dim = mapping[time_dimension]
    if history_d is not None and time_axis == "euclidean":
        if euclidean_time_dim >= history_d:
            if time_dimension == "t":
                euclidean_time_dim = max(0, history_d - 1)
            else:
                msg = (
                    f"Cannot use dimension {euclidean_time_dim} as Euclidean time "
                    f"(only {history_d} dimensions available)."
                )
                raise ValueError(msg)

    return time_axis, euclidean_time_dim


def _compute_channels_vectorized(
    history: RunHistory,
    settings: ChannelSettings,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute channels using vectorized correlator_channels."""
    # Map user-friendly dimension selection to backend parameters
    time_axis, euclidean_time_dim = _map_time_dimension(
        settings.time_dimension, history_d=history.d
    )

    base_config = ChannelConfig(
        warmup_fraction=settings.warmup_fraction,
        max_lag=settings.max_lag,
        h_eff=settings.h_eff,
        mass=settings.mass,
        ell0=settings.ell0,
        neighbor_method=settings.neighbor_method,
        knn_k=settings.knn_k,
        knn_sample=settings.knn_sample,
        use_connected=settings.use_connected,
        mc_time_index=settings.mc_time_index,
        time_axis=time_axis,
        euclidean_time_dim=euclidean_time_dim,
        euclidean_time_bins=settings.euclidean_time_bins,
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=settings.fit_mode,
        fit_start=settings.fit_start,
        fit_stop=settings.fit_stop,
        min_fit_points=settings.min_fit_points,
    )
    channels = [c.strip() for c in settings.channel_list.split(",") if c.strip()]

    per_channel = {
        "scalar": settings.scalar_fit_mode,
        "pseudoscalar": settings.pseudoscalar_fit_mode,
        "vector": settings.vector_fit_mode,
        "nucleon": settings.nucleon_fit_mode,
        "glueball": settings.glueball_fit_mode,
    }

    results: dict[str, ChannelCorrelatorResult] = {}
    for channel in channels:
        override = per_channel.get(channel, "default")
        if override and override != "default":
            config = replace(base_config, fit_mode=str(override))
        else:
            config = base_config
        results.update(compute_all_channels(history, channels=[channel], config=config))
    return results


def _compute_electroweak_channels(
    history: RunHistory,
    settings: ElectroweakSettings,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute electroweak channels using electroweak_channels module."""
    # Map user-friendly dimension selection to backend parameters
    time_axis, euclidean_time_dim = _map_time_dimension(
        settings.time_dimension, history_d=history.d
    )

    config = ElectroweakChannelConfig(
        warmup_fraction=settings.warmup_fraction,
        max_lag=settings.max_lag,
        h_eff=settings.h_eff,
        use_connected=settings.use_connected,
        neighbor_method=settings.neighbor_method,
        knn_k=settings.knn_k,
        mc_time_index=settings.mc_time_index,
        time_axis=time_axis,
        euclidean_time_dim=euclidean_time_dim,
        euclidean_time_bins=settings.euclidean_time_bins,
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=settings.fit_mode,
        fit_start=settings.fit_start,
        fit_stop=settings.fit_stop,
        min_fit_points=settings.min_fit_points,
        epsilon_d=settings.epsilon_d,
        epsilon_c=settings.epsilon_c,
        epsilon_clone=settings.epsilon_clone,
        lambda_alg=settings.lambda_alg,
    )
    channels = [c.strip() for c in settings.channel_list.split(",") if c.strip()]
    return compute_all_electroweak_channels(history, channels=channels, config=config)


@contextmanager
def _temporary_argv(args: list[str]):
    original = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original


def _format_analysis_summary(metrics: dict[str, Any]) -> str:
    observables = metrics.get("observables", {})
    d_fit = observables.get("d_prime_correlation", {})
    r_fit = observables.get("r_prime_correlation", {})
    ew = metrics.get("electroweak_proxy", {})
    qsd = metrics.get("qsd_variance", {})

    lines = [
        "## Analysis Summary",
        f"- d_prime ξ: {d_fit.get('xi', 0.0):.4f} (R² {d_fit.get('r_squared', 0.0):.3f})",
        f"- r_prime ξ: {r_fit.get('xi', 0.0):.4f} (R² {r_fit.get('r_squared', 0.0):.3f})",
        f"- sin²θw proxy: {ew.get('sin2_theta_w_proxy', 0.0):.4f}",
        f"- QSD scaling exponent: {qsd.get('scaling_exponent', 0.0):.4f}",
    ]

    local_fields = metrics.get("local_fields")
    if local_fields:
        lines.append("\n### Local Field Fits")
        for name, info in local_fields.items():
            fit = info.get("fit", {})
            lines.append(
                f"- {name}: ξ={fit.get('xi', 0.0):.4f}, R²={fit.get('r_squared', 0.0):.3f}"
            )

    particle = metrics.get("particle_observables", {}) or {}
    operators = particle.get("operators") if isinstance(particle, dict) else None
    if operators:
        lines.append("\n### Particle Mass Estimates")
        for name, data in operators.items():
            fit = data.get("fit", {})
            lines.append(
                f"- {name}: m={fit.get('mass', 0.0):.4f} (R² {fit.get('r_squared', 0.0):.3f})"
            )

    return "\n".join(lines)


def _format_electroweak_summary(metrics: dict[str, Any]) -> str:
    u1 = metrics.get("u1", {}) if isinstance(metrics, dict) else {}
    su2 = metrics.get("su2", {}) if isinstance(metrics, dict) else {}
    ew = metrics.get("electroweak_proxy", {}) if isinstance(metrics, dict) else {}

    lines = [
        "## Electroweak Summary",
        f"- g1 proxy: {ew.get('g1_proxy', 0.0):.4f}",
        f"- g2 proxy: {ew.get('g2_proxy', 0.0):.4f}",
        f"- sin²θw proxy: {ew.get('sin2_theta_w_proxy', 0.0):.4f}",
        f"- tanθw proxy: {ew.get('tan_theta_w_proxy', 0.0):.4f}",
    ]

    if u1:
        lines.append("\n### U1 Phase")
        lines.append(f"- mean: {u1.get('phase_mean', 0.0):.4f}")
        lines.append(f"- std: {u1.get('phase_std', 0.0):.4f}")
        lines.append(f"- gauge norm: {u1.get('gauge_invariant_norm', 0.0):.4f}")
    if su2:
        lines.append("\n### SU2 Phase")
        lines.append(f"- mean: {su2.get('phase_mean', 0.0):.4f}")
        lines.append(f"- std: {su2.get('phase_std', 0.0):.4f}")
        lines.append(f"- gauge norm: {su2.get('gauge_invariant_norm', 0.0):.4f}")

    return "\n".join(lines)


def _resolve_history_param(
    history: RunHistory | None,
    *keys: str,
    default: float | None = None,
) -> float | None:
    if history is None or not isinstance(history.params, dict):
        return default
    current: Any = history.params
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _compute_force_stats(history: RunHistory | None) -> dict[str, float]:
    if history is None or history.force_viscous is None:
        return {"mean_force_sq": float("nan")}

    force = history.force_viscous
    alive = history.alive_mask
    if alive is None:
        alive = torch.ones(force.shape[:-1], dtype=torch.bool, device=force.device)
    if alive.shape[0] != force.shape[0]:
        min_len = min(alive.shape[0], force.shape[0])
        alive = alive[:min_len]
        force = force[:min_len]

    force_sq = (force**2).sum(dim=-1)
    if alive.numel() == 0:
        return {"mean_force_sq": float("nan")}
    masked = torch.where(alive, force_sq, torch.zeros_like(force_sq))
    counts = alive.sum().clamp(min=1)
    mean_force_sq = float((masked.sum() / counts).item())
    return {"mean_force_sq": mean_force_sq}


def _compute_coupling_constants(
    history: RunHistory | None,
    h_eff: float,
    epsilon_d: float | None = None,
    epsilon_c: float | None = None,
) -> dict[str, float]:
    d = float(history.d) if history is not None else float("nan")
    h_eff = float(max(h_eff, 1e-12))

    epsilon_d = epsilon_d or _resolve_history_param(history, "companion_selection", "epsilon", default=None)
    epsilon_c = epsilon_c or _resolve_history_param(
        history, "companion_selection_clone", "epsilon", default=None
    )
    nu = _resolve_history_param(history, "kinetic", "nu", default=None)

    if epsilon_d is None or epsilon_d <= 0:
        g1_est = float("nan")
    else:
        g1_est = (h_eff / (epsilon_d**2)) ** 0.5

    c2d = (d**2 - 1.0) / (2.0 * d) if d > 0 else float("nan")
    c2_2 = 3.0 / 4.0
    if epsilon_c is None or epsilon_c <= 0 or not np.isfinite(c2d) or c2d <= 0:
        g2_est = float("nan")
    else:
        g2_sq = (2.0 * h_eff / (epsilon_c**2)) * (c2_2 / c2d)
        g2_est = float(max(g2_sq, 0.0) ** 0.5)

    force_stats = _compute_force_stats(history)
    mean_force_sq = force_stats["mean_force_sq"]
    if nu is None or not np.isfinite(mean_force_sq) or d <= 0:
        g3_est = float("nan")
        kvisc_sq = float("nan")
    else:
        kvisc_sq = mean_force_sq / max(float(nu) ** 2, 1e-12)
        g3_sq = (float(nu) ** 2 / h_eff**2) * (d * (d**2 - 1.0) / 12.0) * kvisc_sq
        g3_est = float(max(g3_sq, 0.0) ** 0.5)

    return {
        "g1_est": float(g1_est),
        "g2_est": float(g2_est),
        "g3_est": float(g3_est),
        "epsilon_d": float(epsilon_d) if epsilon_d is not None else float("nan"),
        "epsilon_c": float(epsilon_c) if epsilon_c is not None else float("nan"),
        "nu": float(nu) if nu is not None else float("nan"),
        "kvisc_sq_proxy": float(kvisc_sq),
        "mean_force_sq": float(mean_force_sq),
        "d": float(d),
        "h_eff": float(h_eff),
    }


def _build_coupling_rows(
    couplings: dict[str, float],
    proxies: dict[str, float] | None = None,
    include_strong: bool = False,
    refs: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    proxies = proxies or {}
    refs = refs or {}
    rows = [
        {"name": "g1_est (N1=1)", "value": couplings.get("g1_est"), "note": "from ε_d, h_eff"},
        {"name": "g2_est", "value": couplings.get("g2_est"), "note": "from ε_c, h_eff, C2"},
    ]
    if "g1_proxy" in proxies:
        rows.append({"name": "g1_proxy", "value": proxies.get("g1_proxy"), "note": "phase std"})
    if "g2_proxy" in proxies:
        rows.append({"name": "g2_proxy", "value": proxies.get("g2_proxy"), "note": "phase std"})
    if "sin2_theta_w_proxy" in proxies:
        rows.append(
            {
                "name": "sin2θw proxy",
                "value": proxies.get("sin2_theta_w_proxy"),
                "note": "from phase stds",
            }
        )
    if "tan_theta_w_proxy" in proxies:
        rows.append(
            {
                "name": "tanθw proxy",
                "value": proxies.get("tan_theta_w_proxy"),
                "note": "from phase stds",
            }
        )
    if include_strong:
        rows.append(
            {
                "name": "g3_est",
                "value": couplings.get("g3_est"),
                "note": "from ν, h_eff, <K_visc^2>",
            }
        )
        rows.append(
            {
                "name": "<K_visc^2> proxy",
                "value": couplings.get("kvisc_sq_proxy"),
                "note": "mean ||F_visc||^2 / ν^2",
            }
        )

    for row in rows:
        name = row.get("name")
        value = row.get("value")
        ref_values = refs.get(name, {})
        for column in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS:
            observed = ref_values.get(column)
            error_pct = None
            if observed is not None and observed > 0 and value is not None and np.isfinite(value):
                error_pct = (float(value) - observed) / observed * 100.0
            row[column] = observed
            row[f"error_pct_{column}"] = error_pct
    return rows


def _build_strong_coupling_rows(couplings: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "name": "g3_est",
            "value": couplings.get("g3_est"),
            "note": "from ν, h_eff, <K_visc^2>",
        },
        {
            "name": "<K_visc^2> proxy",
            "value": couplings.get("kvisc_sq_proxy"),
            "note": "mean ||F_visc||^2 / ν^2",
        },
        {"name": "nu", "value": couplings.get("nu"), "note": "viscous coupling"},
        {"name": "h_eff", "value": couplings.get("h_eff"), "note": "phase scale"},
    ]


def _build_analysis_plots(metrics: dict[str, Any], arrays: dict[str, Any]) -> list[Any]:
    plots: list[Any] = []

    observables = metrics.get("observables", {})
    d_fit = observables.get("d_prime_correlation", {})
    r_fit = observables.get("r_prime_correlation", {})

    if "d_prime_bins" in arrays:
        plot = build_correlation_decay_plot(
            arrays["d_prime_bins"],
            arrays["d_prime_correlation"],
            arrays["d_prime_counts"],
            d_fit,
            "Diversity Correlation Decay",
        )
        if plot is not None:
            plots.append(plot)

    if "r_prime_bins" in arrays:
        plot = build_correlation_decay_plot(
            arrays["r_prime_bins"],
            arrays["r_prime_correlation"],
            arrays["r_prime_counts"],
            r_fit,
            "Reward Correlation Decay",
        )
        if plot is not None:
            plots.append(plot)

    if "lyapunov_time" in arrays:
        plot = build_lyapunov_plot(
            arrays["lyapunov_time"],
            arrays["lyapunov_total"],
            arrays["lyapunov_var_x"],
            arrays["lyapunov_var_v"],
        )
        plots.append(plot)

    if "wilson_time_index" in arrays:
        plot = build_wilson_timeseries_plot(
            arrays["wilson_time_index"],
            arrays["wilson_action_mean"],
        )
        if plot is not None:
            plots.append(plot)

    wilson = metrics.get("wilson_loops")
    if wilson and "wilson_values" in wilson:
        plot = build_wilson_histogram_plot(
            np.asarray(wilson["wilson_values"], dtype=float),
            "Wilson Loop Distribution",
        )
        if plot is not None:
            plots.append(plot)

    local_fields = metrics.get("local_fields") or {}
    for field_name, info in local_fields.items():
        bins_key = f"{field_name}_bins"
        corr_key = f"{field_name}_correlation"
        counts_key = f"{field_name}_counts"
        if bins_key not in arrays:
            continue
        plot = build_correlation_decay_plot(
            arrays[bins_key],
            arrays[corr_key],
            arrays[counts_key],
            info.get("fit", {}),
            f"{field_name} Correlation",
        )
        if plot is not None:
            plots.append(plot)

    return plots


BARYON_REFS = {
    "proton": 0.938272,
    "neutron": 0.939565,
    "delta": 1.232,
    "lambda": 1.115683,
    "sigma0": 1.192642,
    "xi0": 1.31486,
    "omega-": 1.67245,
}

MESON_REFS = {
    "pion": 0.13957,
    "kaon": 0.493677,
    "eta": 0.547862,
    "rho": 0.77526,
    "omega": 0.78265,
    "phi": 1.01946,
    "jpsi": 3.0969,
    "upsilon": 9.4603,
}

# Default electroweak mass references (GeV) mapped to proxy channels.
DEFAULT_ELECTROWEAK_REFS = {
    "u1_phase": 0.000511,   # electron
    "u1_dressed": 0.105658, # muon
    "su2_phase": 80.379,    # W boson
    "su2_doublet": 91.1876, # Z boson
    "ew_mixed": 1.77686,    # tau
}

ELECTROWEAK_COUPLING_NAMES = (
    "g1_est (N1=1)",
    "g2_est",
    "g1_proxy",
    "g2_proxy",
    "sin2θw proxy",
    "tanθw proxy",
)

ELECTROWEAK_COUPLING_REFERENCE_COLUMNS = ("observed_mZ", "observed_GUT")

DEFAULT_ELECTROWEAK_COUPLING_REFS = {
    "g1_est (N1=1)": {"observed_mZ": 0.357468, "observed_GUT": 0.560499},
    "g2_est": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "g1_proxy": {"observed_mZ": 0.357468, "observed_GUT": 0.560499},
    "g2_proxy": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "sin2θw proxy": {"observed_mZ": 0.23129, "observed_GUT": 0.375},
    "tanθw proxy": {"observed_mZ": 0.548526, "observed_GUT": 0.774597},
}


def _format_ref_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.6f}"

# Channel-to-family mapping for physics comparisons
CHANNEL_FAMILY_MAP = {
    "scalar": "meson",  # σ (sigma)
    "pseudoscalar": "meson",  # π (pion)
    "vector": "meson",  # ρ (rho)
    "axial_vector": "meson",  # a₁
    "tensor": "meson",  # f₂
    "nucleon": "baryon",  # N (proton/neutron)
    "glueball": "glueball",  # 0^++
}


def _closest_reference(value: float, refs: dict[str, float]) -> tuple[str, float, float]:
    name, ref = min(refs.items(), key=lambda kv: abs(value - kv[1]))
    err = (value - ref) / ref * 100.0
    return name, ref, err


def _best_fit_scale(
    masses: dict[str, float], anchors: list[tuple[str, float, str]]
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for _label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            continue
        numerator += alg_mass * mass_phys
        denominator += alg_mass**2
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_closest(value: float | None, refs: dict[str, float]) -> str:
    if value is None or value <= 0:
        return "n/a"
    name, ref, err = _closest_reference(value, refs)
    return f"{name} {ref:.3f} ({err:+.1f}%)"


def _extract_particle_masses(metrics: dict[str, Any]) -> dict[str, float]:
    particle = metrics.get("particle_observables") or {}
    operators = particle.get("operators") or {}
    masses: dict[str, float] = {}
    for name in ("baryon", "meson", "glueball"):
        fit = operators.get(name, {}).get("fit")
        if fit and isinstance(fit.get("mass"), int | float):
            masses[name] = float(fit["mass"])

    string_tension = metrics.get("string_tension")
    if isinstance(string_tension, dict):
        sigma = string_tension.get("sigma")
        if isinstance(sigma, int | float) and sigma > 0:
            masses["sqrt_sigma"] = float(sigma) ** 0.5

    return masses


def _extract_particle_r2(metrics: dict[str, Any]) -> dict[str, float]:
    particle = metrics.get("particle_observables") or {}
    operators = particle.get("operators") or {}
    r2s: dict[str, float] = {}
    for name in ("baryon", "meson", "glueball"):
        fit = operators.get(name, {}).get("fit")
        if fit and isinstance(fit.get("r_squared"), int | float):
            r2s[name] = float(fit["r_squared"])
    return r2s


def _build_algorithmic_mass_rows(
    masses: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    r2s = r2s or {}
    for name in ("baryon", "meson", "glueball", "sqrt_sigma"):
        if name in masses:
            r2 = r2s.get(name)
            rows.append({
                "operator": name,
                "mass_alg": masses[name],
                "r2": r2 if r2 is not None and np.isfinite(r2) else None,
            })
    return rows


def _format_mass_ratio(masses: dict[str, float]) -> str:
    baryon = masses.get("baryon")
    meson = masses.get("meson")
    if baryon is None or meson is None or baryon <= 0 or meson <= 0:
        return "**Baryon/Meson ratio:** n/a"
    ratio = baryon / meson
    inv_ratio = meson / baryon
    return f"**Baryon/Meson ratio:** {ratio:.3f}  \n" f"**Meson/Baryon ratio:** {inv_ratio:.3f}"


def _build_best_fit_rows(
    masses: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    anchors: list[tuple[str, float, str]] = []
    anchors.extend((f"baryon->{name}", mass, "baryon") for name, mass in BARYON_REFS.items())
    anchors.extend((f"meson->{name}", mass, "meson") for name, mass in MESON_REFS.items())
    baryon_anchors = [a for a in anchors if a[2] == "baryon"]
    meson_anchors = [a for a in anchors if a[2] == "meson"]
    combined_anchors = baryon_anchors + meson_anchors

    r2s = r2s or {}
    baryon_r2 = r2s.get("baryon")
    meson_r2 = r2s.get("meson")

    rows: list[dict[str, Any]] = []
    for label, anchor_list in (
        ("baryon refs", baryon_anchors),
        ("meson refs", meson_anchors),
        ("baryon+meson refs", combined_anchors),
    ):
        scale = _best_fit_scale(masses, anchor_list)
        if scale is None:
            rows.append({"fit_mode": label, "scale_GeV_per_alg": None})
            continue
        pred_b = masses.get("baryon", 0.0) * scale
        pred_m = masses.get("meson", 0.0) * scale
        rows.append({
            "fit_mode": label,
            "scale_GeV_per_alg": scale,
            "baryon_pred_GeV": pred_b,
            "closest_baryon": _format_closest(pred_b, BARYON_REFS),
            "baryon_r2": baryon_r2 if baryon_r2 is not None and np.isfinite(baryon_r2) else None,
            "meson_pred_GeV": pred_m,
            "closest_meson": _format_closest(pred_m, MESON_REFS),
            "meson_r2": meson_r2 if meson_r2 is not None and np.isfinite(meson_r2) else None,
        })
    return rows


def _build_anchor_rows(
    masses: dict[str, float],
    glueball_ref: tuple[str, float] | None,
    sqrt_sigma_ref: float | None,
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    anchors: list[tuple[str, float, str]] = []
    anchors.extend((f"baryon->{name}", mass, "baryon") for name, mass in BARYON_REFS.items())
    anchors.extend((f"meson->{name}", mass, "meson") for name, mass in MESON_REFS.items())
    if glueball_ref is not None and masses.get("glueball", 0.0) > 0:
        label, mass = glueball_ref
        anchors.append((f"glueball->{label}", mass, "glueball"))
    if sqrt_sigma_ref is not None and masses.get("sqrt_sigma", 0.0) > 0:
        anchors.append((f"sqrt_sigma->{sqrt_sigma_ref:.3f}", sqrt_sigma_ref, "sqrt_sigma"))

    glueball_refs: dict[str, float] = {}
    if glueball_ref is not None:
        label, mass = glueball_ref
        glueball_refs[label] = mass

    r2s = r2s or {}
    baryon_r2 = r2s.get("baryon")
    meson_r2 = r2s.get("meson")
    glueball_r2 = r2s.get("glueball")

    rows: list[dict[str, Any]] = []
    for label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            rows.append({"anchor": label})
            continue
        scale = mass_phys / alg_mass
        pred_b = masses.get("baryon", 0.0) * scale
        pred_m = masses.get("meson", 0.0) * scale
        row = {
            "anchor": label,
            "scale_GeV_per_alg": scale,
            "baryon_pred_GeV": pred_b,
            "closest_baryon": _format_closest(pred_b, BARYON_REFS),
            "baryon_r2": baryon_r2 if baryon_r2 is not None and np.isfinite(baryon_r2) else None,
            "meson_pred_GeV": pred_m,
            "closest_meson": _format_closest(pred_m, MESON_REFS),
            "meson_r2": meson_r2 if meson_r2 is not None and np.isfinite(meson_r2) else None,
        }
        if masses.get("glueball") and glueball_ref is not None:
            pred_g = masses.get("glueball", 0.0) * scale
            row["glueball_pred_GeV"] = pred_g
            row["closest_glueball"] = _format_closest(pred_g, glueball_refs)
            row["glueball_r2"] = (
                glueball_r2 if glueball_r2 is not None and np.isfinite(glueball_r2) else None
            )
        if masses.get("sqrt_sigma") and sqrt_sigma_ref is not None:
            pred_s = masses.get("sqrt_sigma", 0.0) * scale
            row["sqrt_sigma_pred_GeV"] = pred_s
            row["closest_sqrt_sigma"] = f"{sqrt_sigma_ref:.3f}"
        rows.append(row)
    return rows


def _get_channel_mass(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    """Extract mass from a channel result based on mode."""
    if mode == "AIC-Weighted":
        return result.mass_fit.get("mass", 0.0)
    else:  # Best Window
        best_window = result.mass_fit.get("best_window", {})
        return best_window.get("mass", 0.0)


def _get_channel_r2(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    if mode == "AIC-Weighted":
        return result.mass_fit.get("r_squared", float("nan"))
    best_window = result.mass_fit.get("best_window", {})
    return best_window.get("r2", float("nan"))


def _extract_channel_masses(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
) -> dict[str, float]:
    """Extract masses from channel results, mapped to families.

    Uses pseudoscalar as the primary meson reference (maps to pion).
    Returns dict like {"baryon": nucleon_mass, "meson": pseudoscalar_mass, "glueball": glueball_mass}.
    """
    masses: dict[str, float] = {}

    # Use pseudoscalar as the primary meson mass (maps to pion)
    if "pseudoscalar" in results:
        ps_mass = _get_channel_mass(results["pseudoscalar"], mode)
        if ps_mass > 0:
            masses["meson"] = ps_mass

    # Use nucleon channel for baryon mass
    if "nucleon" in results:
        nuc_mass = _get_channel_mass(results["nucleon"], mode)
        if nuc_mass > 0:
            masses["baryon"] = nuc_mass

    # Use glueball channel directly
    if "glueball" in results:
        glue_mass = _get_channel_mass(results["glueball"], mode)
        if glue_mass > 0:
            masses["glueball"] = glue_mass

    return masses


def _extract_channel_r2(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
) -> dict[str, float]:
    r2s: dict[str, float] = {}

    if "pseudoscalar" in results:
        ps_r2 = _get_channel_r2(results["pseudoscalar"], mode)
        if np.isfinite(ps_r2):
            r2s["meson"] = ps_r2

    if "nucleon" in results:
        nuc_r2 = _get_channel_r2(results["nucleon"], mode)
        if np.isfinite(nuc_r2):
            r2s["baryon"] = nuc_r2

    if "glueball" in results:
        glue_r2 = _get_channel_r2(results["glueball"], mode)
        if np.isfinite(glue_r2):
            r2s["glueball"] = glue_r2

    return r2s


def _format_channel_ratios(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
) -> str:
    """Format mass ratios: nucleon/pseudoscalar, vector/pseudoscalar, etc."""
    lines = ["**Mass Ratios:**"]

    # Get pseudoscalar mass as denominator
    ps_mass = 0.0
    if "pseudoscalar" in results:
        ps_mass = _get_channel_mass(results["pseudoscalar"], mode)

    if ps_mass <= 0:
        return "**Mass Ratios:** n/a (no pseudoscalar mass)"

    # Nucleon / Pseudoscalar (proton/pion ≈ 6.7)
    if "nucleon" in results:
        nuc_mass = _get_channel_mass(results["nucleon"], mode)
        if nuc_mass > 0:
            ratio = nuc_mass / ps_mass
            lines.append(f"- nucleon/pseudoscalar: **{ratio:.3f}** (proton/pion ≈ 6.7)")

    # Vector / Pseudoscalar (rho/pion ≈ 5.5)
    if "vector" in results:
        vec_mass = _get_channel_mass(results["vector"], mode)
        if vec_mass > 0:
            ratio = vec_mass / ps_mass
            lines.append(f"- vector/pseudoscalar: **{ratio:.3f}** (rho/pion ≈ 5.5)")

    # Scalar / Pseudoscalar (sigma/pion ≈ 3.5-7.0 depending on interpretation)
    if "scalar" in results:
        scalar_mass = _get_channel_mass(results["scalar"], mode)
        if scalar_mass > 0:
            ratio = scalar_mass / ps_mass
            lines.append(f"- scalar/pseudoscalar: **{ratio:.3f}** (sigma/pion)")

    # Glueball / Pseudoscalar
    if "glueball" in results:
        glue_mass = _get_channel_mass(results["glueball"], mode)
        if glue_mass > 0:
            ratio = glue_mass / ps_mass
            lines.append(f"- glueball/pseudoscalar: **{ratio:.3f}**")

    if len(lines) == 1:
        return "**Mass Ratios:** n/a (no valid channel masses)"

    return "  \n".join(lines)


def _format_electroweak_ratios(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
) -> str:
    """Format proxy ratios for electroweak channels."""
    if not results:
        return "**Electroweak Ratios:** n/a"

    base_name = "u1_dressed" if "u1_dressed" in results else "u1_phase"
    base_result = results.get(base_name)
    if base_result is None:
        return "**Electroweak Ratios:** n/a (no U1 mass)"
    base_mass = _get_channel_mass(base_result, mode)
    if base_mass <= 0:
        return "**Electroweak Ratios:** n/a (no U1 mass)"

    lines = ["**Electroweak Ratios (proxy):**"]
    for name in sorted(results.keys()):
        if name == base_name:
            continue
        mass = _get_channel_mass(results[name], mode)
        if mass > 0:
            ratio = mass / base_mass
            lines.append(f"- {name}/{base_name}: **{ratio:.3f}**")

    return "  \n".join(lines)


def _extract_electroweak_masses(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
) -> dict[str, float]:
    masses: dict[str, float] = {}
    for name, result in results.items():
        if result.n_samples == 0:
            continue
        mass = _get_channel_mass(result, mode)
        if mass > 0:
            masses[name] = mass
    return masses


def _extract_electroweak_r2(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
) -> dict[str, float]:
    r2s: dict[str, float] = {}
    for name, result in results.items():
        if result.n_samples == 0:
            continue
        r2 = _get_channel_r2(result, mode)
        if np.isfinite(r2):
            r2s[name] = r2
    return r2s


def _build_electroweak_best_fit_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    r2s = r2s or {}
    anchors = [(f"{name}->{mass:.6f}", mass, name) for name, mass in refs.items()]
    scale = _best_fit_scale(masses, anchors)
    if scale is None:
        return [{"fit_mode": "electroweak refs", "scale_GeV_per_alg": None}]

    row: dict[str, Any] = {
        "fit_mode": "electroweak refs",
        "scale_GeV_per_alg": scale,
    }
    for name, alg_mass in masses.items():
        row[f"{name}_pred_GeV"] = alg_mass * scale
        if name in r2s:
            row[f"{name}_r2"] = r2s[name]
    return [row]


def _build_electroweak_anchor_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    r2s = r2s or {}
    rows: list[dict[str, Any]] = []
    for name, mass_phys in refs.items():
        alg_mass = masses.get(name)
        if alg_mass is None or alg_mass <= 0:
            rows.append({"anchor": f"{name}->{mass_phys:.6f}"})
            continue
        scale = mass_phys / alg_mass
        row: dict[str, Any] = {
            "anchor": f"{name}->{mass_phys:.6f}",
            "scale_GeV_per_alg": scale,
        }
        for ch_name, alg in masses.items():
            row[f"{ch_name}_pred_GeV"] = alg * scale
            if ch_name in r2s:
                row[f"{ch_name}_r2"] = r2s[ch_name]
        rows.append(row)
    return rows


def _build_electroweak_comparison_rows(
    masses: dict[str, float],
    refs: dict[str, float],
) -> list[dict[str, Any]]:
    anchors = [(f"{name}->{mass:.6f}", mass, name) for name, mass in refs.items()]
    scale = _best_fit_scale(masses, anchors)
    if scale is None:
        return []

    rows: list[dict[str, Any]] = []
    for name, alg_mass in masses.items():
        if alg_mass <= 0:
            continue
        obs = refs.get(name)
        pred = alg_mass * scale
        err_pct = None
        if obs is not None and obs > 0:
            err_pct = (pred - obs) / obs * 100.0
        rows.append(
            {
                "channel": name,
                "alg_mass": alg_mass,
                "obs_mass_GeV": obs,
                "pred_mass_GeV": pred,
                "error_pct": err_pct,
            }
        )
    return rows


def _build_electroweak_ratio_rows(
    masses: dict[str, float],
    base_name: str,
    refs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    refs = refs or {}
    rows: list[dict[str, Any]] = []
    base_mass = masses.get(base_name)
    if base_mass is None or base_mass <= 0:
        return rows
    for name, mass in masses.items():
        if name == base_name or mass <= 0:
            continue
        measured = mass / base_mass
        observed = None
        error_pct = None
        obs_num = refs.get(name)
        obs_den = refs.get(base_name)
        if obs_num is not None and obs_den is not None and obs_den > 0:
            observed = obs_num / obs_den
            if observed > 0:
                error_pct = (measured - observed) / observed * 100.0
        rows.append(
            {
                "ratio": f"{name}/{base_name}",
                "measured": measured,
                "observed": observed,
                "error_pct": error_pct,
            }
        )
    return rows


SWEEP_PARAM_DEFS: dict[str, dict[str, Any]] = {
    "density_sigma": {"label": "density_sigma", "type": float},
    "correlation_r_max": {"label": "correlation_r_max", "type": float},
    "correlation_bins": {"label": "correlation_bins", "type": int},
    "gradient_neighbors": {"label": "gradient_neighbors", "type": int},
    "warmup_fraction": {"label": "warmup_fraction", "type": float},
    "fractal_set_stride": {"label": "fractal_set_stride", "type": int},
    "particle_fit_start": {"label": "particle_fit_start", "type": int},
    "particle_fit_stop": {"label": "particle_fit_stop", "type": int},
    "particle_max_lag": {"label": "particle_max_lag", "type": int},
    "particle_knn_k": {"label": "particle_knn_k", "type": int},
    "particle_knn_sample": {"label": "particle_knn_sample", "type": int},
    "particle_mass": {"label": "particle_mass", "type": float},
    "particle_ell0": {"label": "particle_ell0", "type": float},
}

SWEEP_METRICS: dict[str, str] = {
    "baryon mass": "particle_baryon_mass",
    "baryon R²": "particle_baryon_r2",
    "meson mass": "particle_meson_mass",
    "meson R²": "particle_meson_r2",
    "glueball mass": "particle_glueball_mass",
    "glueball R²": "particle_glueball_r2",
    "d_prime ξ": "d_prime_xi",
    "d_prime R²": "d_prime_r2",
    "r_prime ξ": "r_prime_xi",
    "r_prime R²": "r_prime_r2",
    "string tension σ": "string_tension_sigma",
}


def _coerce_sweep_value(param_name: str, value: float) -> Any:
    param_def = SWEEP_PARAM_DEFS.get(param_name, {})
    cast = param_def.get("type", float)
    return cast(value)


def _build_sweep_values(param_name: str, min_val: float, max_val: float, steps: int) -> list[Any]:
    steps = max(1, int(steps))
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    if steps == 1:
        values = [min_val]
    else:
        values = np.linspace(float(min_val), float(max_val), steps)
    param_type = SWEEP_PARAM_DEFS.get(param_name, {}).get("type", float)
    if param_type is int:
        cast_values = [int(round(v)) for v in values]
        return sorted(set(cast_values))
    return [float(v) for v in values]


def _resolve_fit_window(
    analysis_settings: AnalysisSettings,
    x_param: str,
    x_val: Any,
    y_param: str | None,
    y_val: Any | None,
) -> tuple[int | None, int | None]:
    fit_start = analysis_settings.particle_fit_start
    fit_stop = analysis_settings.particle_fit_stop
    if x_param == "particle_fit_start":
        fit_start = int(x_val)
    elif x_param == "particle_fit_stop":
        fit_stop = int(x_val)
    if y_param == "particle_fit_start" and y_val is not None:
        fit_start = int(y_val)
    elif y_param == "particle_fit_stop" and y_val is not None:
        fit_stop = int(y_val)
    return fit_start, fit_stop


def _metric_operator(metric_key: str) -> str | None:
    if metric_key.startswith("particle_baryon"):
        return "baryon"
    if metric_key.startswith("particle_meson"):
        return "meson"
    if metric_key.startswith("particle_glueball"):
        return "glueball"
    return None


def _extract_fit_metadata(
    metrics: dict[str, Any], operator: str
) -> tuple[int | None, int | None, str | None]:
    fit = (
        metrics.get("particle_observables", {})
        .get("operators", {})
        .get(operator, {})
        .get("fit", {})
    )
    fit_start = fit.get("fit_start")
    fit_stop = fit.get("fit_stop")
    fit_mode = fit.get("fit_mode")
    return fit_start, fit_stop, fit_mode


def _extract_metric(metrics: dict[str, Any], key: str) -> float:
    if key == "d_prime_xi":
        return float(
            metrics.get("observables", {}).get("d_prime_correlation", {}).get("xi", np.nan)
        )
    if key == "d_prime_r2":
        return float(
            metrics.get("observables", {}).get("d_prime_correlation", {}).get("r_squared", np.nan)
        )
    if key == "r_prime_xi":
        return float(
            metrics.get("observables", {}).get("r_prime_correlation", {}).get("xi", np.nan)
        )
    if key == "r_prime_r2":
        return float(
            metrics.get("observables", {}).get("r_prime_correlation", {}).get("r_squared", np.nan)
        )
    if key == "string_tension_sigma":
        return float(metrics.get("string_tension", {}).get("sigma", np.nan))

    particle = metrics.get("particle_observables", {}) or {}
    operators = particle.get("operators", {}) or {}
    if key == "particle_baryon_mass":
        return float(operators.get("baryon", {}).get("fit", {}).get("mass", np.nan))
    if key == "particle_baryon_r2":
        return float(operators.get("baryon", {}).get("fit", {}).get("r_squared", np.nan))
    if key == "particle_meson_mass":
        return float(operators.get("meson", {}).get("fit", {}).get("mass", np.nan))
    if key == "particle_meson_r2":
        return float(operators.get("meson", {}).get("fit", {}).get("r_squared", np.nan))
    if key == "particle_glueball_mass":
        return float(operators.get("glueball", {}).get("fit", {}).get("mass", np.nan))
    if key == "particle_glueball_r2":
        return float(operators.get("glueball", {}).get("fit", {}).get("r_squared", np.nan))

    return float(np.nan)


def _extract_particle_fit_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "baryon_mass": _extract_metric(metrics, "particle_baryon_mass"),
        "baryon_r2": _extract_metric(metrics, "particle_baryon_r2"),
        "meson_mass": _extract_metric(metrics, "particle_meson_mass"),
        "meson_r2": _extract_metric(metrics, "particle_meson_r2"),
        "glueball_mass": _extract_metric(metrics, "particle_glueball_mass"),
        "glueball_r2": _extract_metric(metrics, "particle_glueball_r2"),
    }


def _build_sweep_plot(
    dataframe: pd.DataFrame, x_param: str, metric_key: str, y_param: str | None = None
) -> hv.Element | None:
    if dataframe.empty:
        return None
    if y_param is None:
        return hv.Curve(dataframe, kdims=[x_param], vdims=[metric_key]).opts(
            xlabel=x_param,
            ylabel=metric_key,
            title=f"{metric_key} vs {x_param}",
            width=700,
            height=400,
        )
    return hv.HeatMap(dataframe, kdims=[x_param, y_param], vdims=[metric_key]).opts(
        xlabel=x_param,
        ylabel=y_param,
        title=f"{metric_key} sweep",
        width=700,
        height=450,
        colorbar=True,
    )


def create_app() -> pn.template.FastListTemplate:
    """Create the QFT convergence + analysis dashboard."""
    debug = os.environ.get("QFT_DASH_DEBUG", "").lower() in {"1", "true", "yes"}
    skip_sidebar = os.environ.get("QFT_DASH_SKIP_SIDEBAR", "").lower() in {"1", "true", "yes"}
    skip_visual = os.environ.get("QFT_DASH_SKIP_VIS", "").lower() in {"1", "true", "yes"}

    def _debug(msg: str):
        if debug:
            print(f"[qft-dashboard] {msg}", flush=True)

    sidebar = pn.Column(
        pn.pane.Markdown("## QFT Dashboard"),
        pn.pane.Markdown("Starting dashboard..."),
        sizing_mode="stretch_width",
    )
    main = pn.Column(
        pn.pane.Markdown("Loading visualization..."),
        sizing_mode="stretch_both",
    )

    template = pn.template.FastListTemplate(
        title="QFT Swarm Convergence Dashboard",
        sidebar=[sidebar],
        main=[main],
        sidebar_width=435,
        main_max_width="100%",
    )

    def _build_ui():
        start_total = time.time()
        _debug("initializing extensions")
        pn.extension("plotly", "tabulator")

        _debug("building config + visualizer")
        gas_config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)
        gas_config.hide_viscous_kernel_widgets = True
        # Override with the best stable calibration settings found in QFT tuning.
        # This matches weak_potential_fit1_aniso_stable2 (200 walkers, 300 steps).
        gas_config.n_steps = 300
        gas_config.gas_params["N"] = 200
        gas_config.gas_params["dtype"] = "float32"
        gas_config.gas_params["pbc"] = False
        gas_config.neighbor_graph_method = "voronoi"
        gas_config.neighbor_graph_record = True
        gas_config.init_offset = 0.0
        gas_config.init_spread = 10.0
        gas_config.init_velocity_scale = 5.0

        # Quadratic well with calibrated curvature.
        benchmark, background, mode_points = prepare_benchmark_for_explorer(
            benchmark_name="Quadratic Well",
            dims=gas_config.dims,
            bounds_range=(-gas_config.bounds_extent, gas_config.bounds_extent),
            resolution=100,
            alpha=0.02,
        )
        gas_config.potential = benchmark
        gas_config.background = background
        gas_config.mode_points = mode_points

        # Kinetic operator (Langevin + viscous coupling).
        gas_config.kinetic_op.gamma = 1.0
        gas_config.kinetic_op.beta = 1.0
        gas_config.kinetic_op.delta_t = 0.05
        gas_config.kinetic_op.epsilon_F = 1.0
        gas_config.kinetic_op.use_fitness_force = False
        gas_config.kinetic_op.use_potential_force = False
        gas_config.kinetic_op.use_anisotropic_diffusion = True
        gas_config.kinetic_op.diagonal_diffusion = False
        gas_config.kinetic_op.diffusion_mode = "voronoi_proxy"
        gas_config.kinetic_op.diffusion_grad_scale = 30.0
        gas_config.kinetic_op.epsilon_Sigma = 0.5
        gas_config.kinetic_op.nu = 1.10
        gas_config.kinetic_op.beta_curl = 1.0
        gas_config.kinetic_op.use_viscous_coupling = True
        gas_config.kinetic_op.viscous_length_scale = 0.251372
        gas_config.kinetic_op.viscous_neighbor_mode = "all"
        gas_config.kinetic_op.viscous_neighbor_weighting = "uniform"
        gas_config.kinetic_op.viscous_neighbor_threshold = None
        gas_config.kinetic_op.viscous_neighbor_penalty = 0.0
        gas_config.kinetic_op.viscous_degree_cap = None
        gas_config.kinetic_op.use_velocity_squashing = True

        # Companion selection (diversity + cloning).
        gas_config.companion_selection.method = "uniform"
        gas_config.companion_selection.epsilon = 2.80
        gas_config.companion_selection.lambda_alg = 1.0
        gas_config.companion_selection.exclude_self = True
        gas_config.companion_selection_clone.method = "uniform"
        gas_config.companion_selection_clone.epsilon = 1.68419
        gas_config.companion_selection_clone.lambda_alg = 1.0
        gas_config.companion_selection_clone.exclude_self = True

        # Cloning operator.
        gas_config.cloning.p_max = 1.0
        gas_config.cloning.epsilon_clone = 0.01
        gas_config.cloning.sigma_x = 0.1
        gas_config.cloning.alpha_restitution = 1.0

        # Fitness operator.
        gas_config.fitness_op.alpha = 1.0
        gas_config.fitness_op.beta = 1.0
        gas_config.fitness_op.eta = 0.1
        gas_config.fitness_op.lambda_alg = 1.0
        gas_config.fitness_op.sigma_min = 0.1
        gas_config.fitness_op.epsilon_dist = 1e-8
        gas_config.fitness_op.A = 2.0
        gas_config.fitness_op.rho = 0.1
        gas_config.fitness_op.grad_mode = "sum"
        gas_config.fitness_op.detach_stats = True
        gas_config.fitness_op.detach_companions = True
        visualizer = SwarmConvergence3D(history=None, bounds_extent=gas_config.bounds_extent)
        analysis_settings = AnalysisSettings()

        state: dict[str, Any] = {
            "history": None,
            "history_path": None,
            "analysis_metrics": None,
            "analysis_arrays": None,
            "electroweak_results": None,
        }

        _debug("setting up history controls")
        repo_root = Path(__file__).resolve().parents[4]
        qft_run_id = "qft_penalty_thr0p75_pen0p9_m354_ed2p80_nu1p10_N200_long"
        qft_history_path = repo_root / "outputs" / "qft_calibrated" / f"{qft_run_id}_history.pt"
        qft_history_dir = qft_history_path.parent
        qft_history_dir.mkdir(parents=True, exist_ok=True)
        history_dir = qft_history_dir
        history_path_input = pn.widgets.TextInput(
            name="QFT RunHistory path",
            value=str(qft_history_path),
            width=335,
            sizing_mode="stretch_width",
        )
        browse_button = pn.widgets.Button(
            name="Browse files...",
            button_type="default",
            width=335,
            sizing_mode="stretch_width",
        )
        file_selector_container = pn.Column(sizing_mode="stretch_width")

        load_button = pn.widgets.Button(
            name="Load RunHistory",
            button_type="primary",
            width=335,
            sizing_mode="stretch_width",
        )
        load_status = pn.pane.Markdown(
            "**Load a history**: paste a *_history.pt path or browse and click Load.",
            sizing_mode="stretch_width",
        )

        analysis_status_sidebar = pn.pane.Markdown(
            "**Analysis:** run a simulation or load a RunHistory.",
            sizing_mode="stretch_width",
        )
        analysis_status_main = pn.pane.Markdown(
            "**Analysis:** run a simulation or load a RunHistory.",
            sizing_mode="stretch_width",
        )
        analysis_summary = pn.pane.Markdown(
            "## Analysis Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        analysis_json = pn.pane.JSON({}, depth=2, sizing_mode="stretch_width")
        analysis_plots = pn.Column(sizing_mode="stretch_width")

        analysis_output_dir = pn.widgets.TextInput(
            name="Analysis output dir",
            value="outputs/qft_dashboard_analysis",
            width=335,
            sizing_mode="stretch_width",
        )
        analysis_id_input = pn.widgets.TextInput(
            name="Analysis id",
            value="",
            width=335,
            sizing_mode="stretch_width",
            placeholder="Optional (defaults to timestamp)",
        )
        run_analysis_button = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            width=335,
            sizing_mode="stretch_width",
            disabled=True,
        )
        run_analysis_button_main = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )

        particle_status = pn.pane.Markdown(
            "**Particles:** run particle analysis to populate tables.",
            sizing_mode="stretch_width",
        )
        particle_run_button = pn.widgets.Button(
            name="Compute Particles",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )
        glueball_label_input = pn.widgets.TextInput(
            name="Glueball label",
            value="glueball",
            width=200,
            sizing_mode="fixed",
        )
        glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            width=200,
            sizing_mode="fixed",
        )
        sqrt_sigma_ref_input = pn.widgets.FloatInput(
            name="sqrt(sigma) ref (GeV)",
            value=None,
            step=0.01,
            width=200,
            sizing_mode="fixed",
        )

        particle_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        particle_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        particle_ratio_pane = pn.pane.Markdown(
            "**Baryon/Meson ratio:** n/a",
            sizing_mode="stretch_width",
        )
        particle_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        sweep_enable_2d = pn.widgets.Checkbox(name="2D sweep", value=False)
        sweep_param_x = pn.widgets.Select(
            name="Sweep param (X)",
            options={v["label"]: k for k, v in SWEEP_PARAM_DEFS.items()},
            value="density_sigma",
        )
        sweep_param_y = pn.widgets.Select(
            name="Sweep param (Y)",
            options={v["label"]: k for k, v in SWEEP_PARAM_DEFS.items()},
            value="particle_knn_k",
        )
        sweep_metric = pn.widgets.Select(
            name="Metric",
            options=dict(SWEEP_METRICS.items()),
            value="particle_baryon_mass",
        )
        sweep_min_x = pn.widgets.FloatInput(name="X min", value=0.1, step=0.1, width=120)
        sweep_max_x = pn.widgets.FloatInput(name="X max", value=1.0, step=0.1, width=120)
        sweep_steps_x = pn.widgets.IntInput(name="X steps", value=5, step=1, width=120)
        sweep_min_y = pn.widgets.FloatInput(name="Y min", value=1.0, step=1.0, width=120)
        sweep_max_y = pn.widgets.FloatInput(name="Y max", value=5.0, step=1.0, width=120)
        sweep_steps_y = pn.widgets.IntInput(name="Y steps", value=4, step=1, width=120)
        sweep_run_button = pn.widgets.Button(
            name="Run Sweep",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )
        sweep_status = pn.pane.Markdown(
            "**Sweep:** configure parameters and run to see results.",
            sizing_mode="stretch_width",
        )
        sweep_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        sweep_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)

        # =====================================================================
        # New "Channels" tab components (independent vectorized analysis)
        # =====================================================================
        channel_settings = ChannelSettings()

        channels_status = pn.pane.Markdown(
            "**Strong Force:** Load a RunHistory and click Compute to analyze.",
            sizing_mode="stretch_width",
        )
        channels_run_button = pn.widgets.Button(
            name="Compute Channels",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )

        channel_settings_panel = pn.Param(
            channel_settings,
            parameters=[
                "warmup_fraction",
                "max_lag",
                "h_eff",
                "mass",
                "ell0",
                "time_dimension",
                "mc_time_index",
                "euclidean_time_bins",
                "neighbor_method",
                "knn_k",
                "knn_sample",
                "use_connected",
                "channel_list",
                "window_widths_spec",
                "fit_mode",
                "scalar_fit_mode",
                "pseudoscalar_fit_mode",
                "vector_fit_mode",
                "nucleon_fit_mode",
                "glueball_fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
            ],
            show_name=False,
            widgets={
                "time_dimension": {
                    "type": pn.widgets.Select,
                    "name": "Time Axis",
                },
                "mc_time_index": {
                    "name": "MC time slice (step or idx; blank=last)",
                },
                "euclidean_time_bins": {
                    "name": "Time Bins (Euclidean only)",
                },
            },
        )

        # Plot containers for channel tab
        channel_plots_correlator = pn.Column(sizing_mode="stretch_width")
        channel_plots_effective_mass = pn.Column(sizing_mode="stretch_width")
        channel_plots_spectrum = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        channel_plots_overlay_corr = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        channel_plots_overlay_meff = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        strong_coupling_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        # New plot containers for plateau and heatmap visualizations
        channel_plateau_plots = pn.Column(sizing_mode="stretch_width")
        channel_heatmap_plots = pn.Column(sizing_mode="stretch_width")
        heatmap_color_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Color",
            options=["mass", "aic", "r2"],
            value="mass",
            button_type="default",
        )
        heatmap_alpha_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Opacity",
            options=["aic", "mass", "r2"],
            value="aic",
            button_type="default",
        )
        channel_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )

        # Toggle for AIC-weighted vs best-window masses
        channel_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )

        # New tables for channel comparison (mirrors Particles tab)
        channel_ratio_pane = pn.pane.Markdown(
            "**Mass Ratios:** Compute channels to see ratios.",
            sizing_mode="stretch_width",
        )
        channel_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        channel_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        # Optional: Channel-specific reference inputs
        channel_glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            width=200,
            sizing_mode="fixed",
        )

        # =====================================================================
        # Electroweak tab components (U1/SU2 phase channels)
        # =====================================================================
        electroweak_settings = ElectroweakSettings()
        electroweak_status = pn.pane.Markdown(
            "**Electroweak:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        electroweak_run_button = pn.widgets.Button(
            name="Compute Electroweak",
            button_type="primary",
            width=240,
            sizing_mode="fixed",
            disabled=True,
        )
        electroweak_summary = pn.pane.Markdown(
            "## Electroweak Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        electroweak_coupling_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        electroweak_coupling_ref_table = pn.widgets.Tabulator(
            pd.DataFrame(
                {
                    "name": list(ELECTROWEAK_COUPLING_NAMES),
                    "observed_mZ": [
                        _format_ref_value(
                            DEFAULT_ELECTROWEAK_COUPLING_REFS.get(name, {}).get(
                                "observed_mZ"
                            )
                        )
                        for name in ELECTROWEAK_COUPLING_NAMES
                    ],
                    "observed_GUT": [
                        _format_ref_value(
                            DEFAULT_ELECTROWEAK_COUPLING_REFS.get(name, {}).get(
                                "observed_GUT"
                            )
                        )
                        for name in ELECTROWEAK_COUPLING_NAMES
                    ],
                }
            ),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors={column: "input" for column in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS},
        )
        electroweak_phase_plot = pn.pane.HTML(
            "<p><em>Phase histograms will appear after analysis.</em></p>",
            sizing_mode="stretch_width",
            height=420,
        )

        electroweak_settings_panel = pn.Param(
            electroweak_settings,
            parameters=[
                "warmup_fraction",
                "max_lag",
                "h_eff",
                "time_dimension",
                "mc_time_index",
                "euclidean_time_bins",
                "use_connected",
                "neighbor_method",
                "knn_k",
                "channel_list",
                "window_widths_spec",
                "fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
                "epsilon_d",
                "epsilon_c",
                "epsilon_clone",
                "lambda_alg",
            ],
            show_name=False,
            widgets={
                "time_dimension": {
                    "type": pn.widgets.Select,
                    "name": "Time Axis",
                },
                "mc_time_index": {
                    "name": "MC time slice (step or idx; blank=last)",
                },
                "euclidean_time_bins": {
                    "name": "Time Bins (Euclidean only)",
                },
            },
        )

        electroweak_plots_correlator = pn.Column(sizing_mode="stretch_width")
        electroweak_plots_effective_mass = pn.Column(sizing_mode="stretch_width")
        electroweak_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        electroweak_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        electroweak_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        electroweak_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        electroweak_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )
        electroweak_ratio_pane = pn.pane.Markdown(
            "**Electroweak Ratios:** Compute channels to see ratios.",
            sizing_mode="stretch_width",
        )
        electroweak_ratio_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        electroweak_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        electroweak_compare_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        electroweak_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        electroweak_ref_table = pn.widgets.Tabulator(
            pd.DataFrame(
                {
                    "channel": list(ELECTROWEAK_CHANNELS),
                    "mass_ref_GeV": [
                        _format_ref_value(DEFAULT_ELECTROWEAK_REFS.get(name))
                        for name in ELECTROWEAK_CHANNELS
                    ],
                }
            ),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors={"mass_ref_GeV": "input"},
        )

        def _set_analysis_status(message: str) -> None:
            analysis_status_sidebar.object = message
            analysis_status_main.object = message

        def _set_particle_status(message: str) -> None:
            particle_status.object = message

        def _update_particle_tables(metrics: dict[str, Any]) -> None:
            masses = _extract_particle_masses(metrics)
            r2s = _extract_particle_r2(metrics)
            if "baryon" not in masses or "meson" not in masses:
                particle_mass_table.value = pd.DataFrame()
                particle_fit_table.value = pd.DataFrame()
                particle_anchor_table.value = pd.DataFrame()
                particle_ratio_pane.object = "**Baryon/Meson ratio:** n/a"
                _set_particle_status(
                    "**Particles:** missing baryon/meson masses. Enable compute_particles."
                )
                return

            glueball_ref = None
            if glueball_ref_input.value is not None:
                glueball_ref = (glueball_label_input.value, float(glueball_ref_input.value))

            sqrt_sigma_ref = None
            if sqrt_sigma_ref_input.value is not None:
                sqrt_sigma_ref = float(sqrt_sigma_ref_input.value)

            particle_mass_table.value = pd.DataFrame(_build_algorithmic_mass_rows(masses, r2s))
            particle_fit_table.value = pd.DataFrame(_build_best_fit_rows(masses, r2s))
            particle_ratio_pane.object = _format_mass_ratio(masses)
            particle_anchor_table.value = pd.DataFrame(
                _build_anchor_rows(masses, glueball_ref, sqrt_sigma_ref, r2s)
            )
            _set_particle_status("**Particles:** tables updated.")

        def _default_sweep_range(param_name: str) -> tuple[float, float, int]:
            current = getattr(analysis_settings, param_name, 1.0)
            if current is None:
                current = 1.0
            param_type = SWEEP_PARAM_DEFS.get(param_name, {}).get("type", float)
            if param_type is int:
                base = int(current)
                min_v = max(1, base - 5)
                max_v = max(min_v + 1, base + 5)
            else:
                base = float(current)
                if base == 0:
                    base = 1.0
                min_v = max(1e-6, base * 0.5)
                max_v = max(min_v * 1.1, base * 1.5)
            return float(min_v), float(max_v), 5

        def _sync_sweep_bounds(param_name: str, min_w, max_w, steps_w) -> None:
            min_v, max_v, steps = _default_sweep_range(param_name)
            min_w.value = min_v
            max_w.value = max_v
            steps_w.value = steps

        def _toggle_sweep_controls(event) -> None:
            enabled = bool(event.new)
            sweep_param_y.visible = enabled
            sweep_min_y.visible = enabled
            sweep_max_y.visible = enabled
            sweep_steps_y.visible = enabled

        def _on_sweep_param_x(event) -> None:
            _sync_sweep_bounds(event.new, sweep_min_x, sweep_max_x, sweep_steps_x)

        def _on_sweep_param_y(event) -> None:
            _sync_sweep_bounds(event.new, sweep_min_y, sweep_max_y, sweep_steps_y)

        _sync_sweep_bounds(sweep_param_x.value, sweep_min_x, sweep_max_x, sweep_steps_x)
        _sync_sweep_bounds(sweep_param_y.value, sweep_min_y, sweep_max_y, sweep_steps_y)
        for widget in (sweep_param_y, sweep_min_y, sweep_max_y, sweep_steps_y):
            widget.visible = sweep_enable_2d.value

        def set_history(history: RunHistory, history_path: Path | None = None) -> None:
            state["history"] = history
            state["history_path"] = history_path
            visualizer.bounds_extent = float(gas_config.bounds_extent)
            visualizer.set_history(history)
            _set_analysis_status("**Analysis ready:** click Run Analysis.")
            run_analysis_button.disabled = False
            run_analysis_button_main.disabled = False
            particle_run_button.disabled = False
            _set_particle_status("**Particles ready:** click Compute Particles.")
            sweep_run_button.disabled = False
            # Enable channels tab
            channels_run_button.disabled = False
            channels_status.object = "**Strong Force ready:** click Compute Channels."
            # Enable electroweak tab
            electroweak_run_button.disabled = False
            electroweak_status.object = "**Electroweak ready:** click Compute Electroweak."

        def on_simulation_complete(history: RunHistory):
            set_history(history)

        def _infer_bounds_extent(history: RunHistory) -> float | None:
            if history.bounds is None:
                return None
            high = history.bounds.high.detach().cpu().abs().max().item()
            low = history.bounds.low.detach().cpu().abs().max().item()
            return float(max(high, low))

        def _sync_history_path(value):
            if value:
                history_path_input.value = str(value[0])

        def _ensure_file_selector() -> pn.widgets.FileSelector:
            if file_selector_container.objects:
                return file_selector_container.objects[0]
            selector = pn.widgets.FileSelector(
                name="Select RunHistory",
                directory=str(history_dir),
                file_pattern="*_history.pt",
                only_files=True,
            )
            if qft_history_path.exists():
                selector.value = [str(qft_history_path)]
            selector.param.watch(lambda e: _sync_history_path(e.new), "value")
            file_selector_container.objects = [selector]
            return selector

        def _on_browse_clicked(_):
            _ensure_file_selector()

        def on_load_clicked(_):
            history_path = Path(history_path_input.value).expanduser()
            if not history_path.exists():
                load_status.object = "**Error:** History path does not exist."
                return
            try:
                history = RunHistory.load(str(history_path))
                inferred_extent = _infer_bounds_extent(history)
                if inferred_extent is not None:
                    visualizer.bounds_extent = inferred_extent
                    gas_config.bounds_extent = inferred_extent
                set_history(history, history_path)
                load_status.object = f"**Loaded:** `{history_path}`"
            except Exception as exc:
                load_status.object = f"**Error loading history:** {exc!s}"

        def on_bounds_change(event):
            visualizer.bounds_extent = float(event.new)
            visualizer._refresh_frame()

        def _run_analysis(force_particles: bool) -> tuple[dict[str, Any], dict[str, Any]] | None:
            history = state.get("history")
            if history is None:
                _set_analysis_status("**Error:** load or run a simulation first.")
                _set_particle_status("**Error:** load or run a simulation first.")
                return None

            if force_particles:
                analysis_settings.compute_particles = True
                if glueball_ref_input.value is not None:
                    analysis_settings.build_fractal_set = True
                if sqrt_sigma_ref_input.value is not None:
                    analysis_settings.compute_string_tension = True
                    analysis_settings.build_fractal_set = True

            output_dir = Path(analysis_output_dir.value)
            output_dir.mkdir(parents=True, exist_ok=True)
            analysis_id = analysis_id_input.value.strip() or datetime.utcnow().strftime(
                "%Y%m%d_%H%M%S"
            )
            analysis_id_input.value = analysis_id

            history_path = state.get("history_path")
            if history_path is None or not history_path.exists():
                history_path = output_dir / f"{analysis_id}_history.pt"
                history.save(str(history_path))
                state["history_path"] = history_path

            args = analysis_settings.to_cli_args(history_path, output_dir, analysis_id)
            _set_analysis_status("**Running analysis...**")
            _set_particle_status("**Running analysis...**")

            try:
                with _temporary_argv(args):
                    qft_analysis.main()
            except Exception as exc:
                _set_analysis_status(f"**Error:** {exc!s}")
                _set_particle_status(f"**Error:** {exc!s}")
                return None

            metrics_path = output_dir / f"{analysis_id}_metrics.json"
            arrays_path = output_dir / f"{analysis_id}_arrays.npz"
            if not metrics_path.exists():
                _set_analysis_status("**Error:** analysis metrics file missing.")
                _set_particle_status("**Error:** analysis metrics file missing.")
                return None

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if arrays_path.exists():
                with np.load(arrays_path) as data:
                    arrays = {key: data[key] for key in data.files}
            else:
                arrays = {}

            state["analysis_metrics"] = metrics
            state["analysis_arrays"] = arrays
            return metrics, arrays

        def _extract_coupling_refs() -> dict[str, dict[str, float]]:
            refs: dict[str, dict[str, float]] = {}
            ref_df = electroweak_coupling_ref_table.value
            if isinstance(ref_df, pd.DataFrame):
                for _, row in ref_df.iterrows():
                    name = str(row.get("name", "")).strip()
                    if not name:
                        continue
                    ref_values: dict[str, float] = {}
                    for column in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS:
                        raw = row.get(column)
                        if isinstance(raw, str):
                            raw = raw.strip()
                            if raw == "":
                                continue
                        try:
                            value = float(raw)
                        except (TypeError, ValueError):
                            continue
                        if value > 0:
                            ref_values[column] = value
                    if ref_values:
                        refs[name] = ref_values
            return refs

        def _update_electroweak_summary(metrics: dict[str, Any]) -> None:
            electroweak_summary.object = _format_electroweak_summary(metrics)
            history = state.get("history")
            proxies = metrics.get("electroweak_proxy", {}) if isinstance(metrics, dict) else {}
            couplings = _compute_coupling_constants(
                history,
                h_eff=float(electroweak_settings.h_eff),
                epsilon_d=electroweak_settings.epsilon_d,
                epsilon_c=electroweak_settings.epsilon_c,
            )
            electroweak_coupling_table.value = pd.DataFrame(
                _build_coupling_rows(
                    couplings,
                    proxies,
                    include_strong=False,
                    refs=_extract_coupling_refs(),
                )
            )
            plots = metrics.get("plots", {}) if isinstance(metrics, dict) else {}
            phase_path = plots.get("phase_histograms")
            if phase_path and Path(phase_path).exists():
                electroweak_phase_plot.object = Path(phase_path).read_text()
            else:
                electroweak_phase_plot.object = (
                    "<p><em>Phase histograms not available. "
                    "Run analysis with gauge phase plots enabled.</em></p>"
                )

        def _update_analysis_outputs(metrics: dict[str, Any], arrays: dict[str, Any]) -> None:
            analysis_summary.object = _format_analysis_summary(metrics)
            analysis_json.object = metrics
            analysis_plots.objects = [
                pn.pane.HoloViews(plot, sizing_mode="stretch_width", linked_axes=False)
                for plot in _build_analysis_plots(metrics, arrays)
            ]
            _update_electroweak_summary(metrics)

        def on_run_analysis(_):
            result = _run_analysis(force_particles=False)
            if result is None:
                return
            metrics, arrays = result
            _update_analysis_outputs(metrics, arrays)
            _set_analysis_status("**Analysis complete.**")
            _update_particle_tables(metrics)

        def on_run_particles(_):
            result = _run_analysis(force_particles=True)
            if result is None:
                return
            metrics, arrays = result
            _update_analysis_outputs(metrics, arrays)
            _set_analysis_status("**Analysis complete.**")
            _update_particle_tables(metrics)

        def on_run_sweep(_):
            history = state.get("history")
            if history is None:
                _set_analysis_status("**Error:** load or run a simulation first.")
                _set_particle_status("**Error:** load or run a simulation first.")
                return

            x_param = sweep_param_x.value
            y_param = sweep_param_y.value if sweep_enable_2d.value else None
            metric_key = sweep_metric.value

            steps_x = max(1, int(sweep_steps_x.value))
            x_values = _build_sweep_values(
                x_param,
                float(sweep_min_x.value),
                float(sweep_max_x.value),
                steps_x,
            )
            if y_param:
                steps_y = max(1, int(sweep_steps_y.value))
                y_values = _build_sweep_values(
                    y_param,
                    float(sweep_min_y.value),
                    float(sweep_max_y.value),
                    steps_y,
                )
            else:
                y_values = [None]

            original_values = {
                x_param: getattr(analysis_settings, x_param),
                "compute_particles": analysis_settings.compute_particles,
                "build_fractal_set": analysis_settings.build_fractal_set,
                "compute_string_tension": analysis_settings.compute_string_tension,
            }
            if y_param:
                original_values[y_param] = getattr(analysis_settings, y_param)

            original_analysis_id = analysis_id_input.value
            base_id = original_analysis_id.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            results: list[dict[str, Any]] = []
            skipped = 0
            attempted = 0
            sweep_status.object = "**Sweep:** running..."
            include_fit_meta = {
                "particle_fit_start",
                "particle_fit_stop",
            }.intersection({x_param, y_param or ""})
            fit_operator = _metric_operator(metric_key)

            try:
                for x_val in x_values:
                    x_val_cast = _coerce_sweep_value(x_param, x_val)
                    setattr(analysis_settings, x_param, x_val_cast)
                    for y_val in y_values:
                        suffix = f"{x_param}_{x_val_cast}"
                        if y_param and y_val is not None:
                            y_val_cast = _coerce_sweep_value(y_param, y_val)
                            setattr(analysis_settings, y_param, y_val_cast)
                            suffix = f"{suffix}__{y_param}_{y_val_cast}"
                        else:
                            y_val_cast = None

                        fit_start, fit_stop = _resolve_fit_window(
                            analysis_settings, x_param, x_val_cast, y_param, y_val_cast
                        )
                        if fit_start is not None and fit_stop is not None and fit_stop < fit_start:
                            skipped += 1
                            row = {
                                x_param: x_val_cast,
                                metric_key: np.nan,
                                "status": "invalid_fit_window",
                                "fit_start": fit_start,
                                "fit_stop": fit_stop,
                            }
                            if y_param:
                                row[y_param] = y_val_cast
                            row.update({
                                "baryon_mass": np.nan,
                                "baryon_r2": np.nan,
                                "meson_mass": np.nan,
                                "meson_r2": np.nan,
                                "glueball_mass": np.nan,
                                "glueball_r2": np.nan,
                            })
                            results.append(row)
                            continue

                        analysis_id_input.value = f"{base_id}_sweep_{suffix}"
                        attempted += 1
                        result = _run_analysis(force_particles=True)
                        if result is None:
                            skipped += 1
                            row = {
                                x_param: x_val_cast,
                                metric_key: np.nan,
                                "status": "analysis_error",
                                "fit_start": fit_start,
                                "fit_stop": fit_stop,
                            }
                            if y_param:
                                row[y_param] = y_val_cast
                            row.update({
                                "baryon_mass": np.nan,
                                "baryon_r2": np.nan,
                                "meson_mass": np.nan,
                                "meson_r2": np.nan,
                                "glueball_mass": np.nan,
                                "glueball_r2": np.nan,
                            })
                            results.append(row)
                            continue
                        metrics, _arrays = result
                        value = _extract_metric(metrics, metric_key)
                        row = {x_param: x_val_cast, metric_key: value}
                        if y_param:
                            row[y_param] = y_val_cast
                        row.update(_extract_particle_fit_metrics(metrics))
                        if fit_start is not None:
                            row["fit_start"] = fit_start
                        if fit_stop is not None:
                            row["fit_stop"] = fit_stop
                        if include_fit_meta and fit_operator is not None:
                            fit_start_used, fit_stop_used, fit_mode_used = _extract_fit_metadata(
                                metrics, fit_operator
                            )
                            row["fit_start_used"] = fit_start_used
                            row["fit_stop_used"] = fit_stop_used
                            row["fit_mode_used"] = fit_mode_used
                        results.append(row)
            finally:
                for name, value in original_values.items():
                    setattr(analysis_settings, name, value)
                analysis_id_input.value = original_analysis_id

            if not results:
                sweep_status.object = "**Sweep:** no results."
                sweep_table.value = pd.DataFrame()
                sweep_plot.object = None
                return

            df = pd.DataFrame(results)
            sweep_table.value = df
            plot = _build_sweep_plot(df, x_param, metric_key, y_param=y_param)
            sweep_plot.object = plot
            if skipped:
                sweep_status.object = f"**Sweep:** complete ({attempted} runs, {skipped} skipped)."
            else:
                sweep_status.object = f"**Sweep:** complete ({attempted} runs)."

        # =====================================================================
        # Channels tab callbacks (vectorized correlator_channels)
        # =====================================================================

        def _update_channel_plots(results: dict[str, ChannelCorrelatorResult]) -> None:
            """Update all channel plots from computed results."""
            corr_plots = []
            meff_plots = []
            plateau_plots = []
            heatmap_plots = []

            for name, result in results.items():
                if result.n_samples == 0:
                    continue

                # Convert tensors to numpy
                corr = result.correlator.cpu().numpy() if hasattr(result.correlator, "cpu") else np.asarray(result.correlator)
                meff = result.effective_mass.cpu().numpy() if hasattr(result.effective_mass, "cpu") else np.asarray(result.effective_mass)
                lag_times = np.arange(len(corr)) * result.dt
                meff_times = np.arange(len(meff)) * result.dt

                corr_plot = build_correlator_plot(lag_times, corr, result.mass_fit, name)
                if corr_plot is not None:
                    corr_plots.append(
                        pn.pane.HoloViews(corr_plot, sizing_mode="stretch_width", linked_axes=False)
                    )

                meff_plot = build_effective_mass_plot(meff_times, meff, result.mass_fit, name)
                if meff_plot is not None:
                    meff_plots.append(
                        pn.pane.HoloViews(meff_plot, sizing_mode="stretch_width", linked_axes=False)
                    )

                # Build enhanced plateau plot (2-panel layout)
                plateau_panels = build_effective_mass_plateau_plot(
                    lag_times, corr, meff, result.mass_fit, name, dt=result.dt
                )
                if plateau_panels is not None:
                    left_panel, right_panel = plateau_panels
                    plateau_plots.append(pn.Row(
                        pn.pane.HoloViews(
                            left_panel,
                            sizing_mode="stretch_width",
                            linked_axes=False,
                        ),
                        pn.pane.HoloViews(
                            right_panel,
                            sizing_mode="stretch_width",
                            linked_axes=False,
                        ),
                        sizing_mode="stretch_width",
                    ))

                # Build window heatmap if window data is available
                if result.window_masses is not None and result.window_aic is not None:
                    window_masses = result.window_masses.cpu().numpy() if hasattr(result.window_masses, "cpu") else np.asarray(result.window_masses)
                    window_aic = result.window_aic.cpu().numpy() if hasattr(result.window_aic, "cpu") else np.asarray(result.window_aic)
                    window_r2 = None
                    if result.window_r2 is not None:
                        window_r2 = result.window_r2.cpu().numpy() if hasattr(result.window_r2, "cpu") else np.asarray(result.window_r2)
                    best_window = result.mass_fit.get("best_window", {})

                    heatmap_plot = build_window_heatmap(
                        window_masses,
                        window_aic,
                        result.window_widths or [],
                        best_window,
                        name,
                        window_r2=window_r2,
                        color_metric=str(heatmap_color_metric.value),
                        alpha_metric=str(heatmap_alpha_metric.value),
                    )
                    if heatmap_plot is not None:
                        heatmap_plots.append(pn.pane.HoloViews(
                            heatmap_plot,
                            sizing_mode="stretch_width",
                            linked_axes=False,  # Prevent axis sharing between plots
                        ))

            channel_plots_correlator.objects = corr_plots if corr_plots else [
                pn.pane.Markdown("_No correlator plots available._")
            ]
            channel_plots_effective_mass.objects = meff_plots if meff_plots else [
                pn.pane.Markdown("_No effective mass plots available._")
            ]
            channel_plateau_plots.objects = plateau_plots if plateau_plots else [
                pn.pane.Markdown("_No plateau plots available._")
            ]
            channel_heatmap_plots.objects = heatmap_plots if heatmap_plots else [
                pn.pane.Markdown("_No window heatmaps available._")
            ]

            # Mass spectrum bar chart
            spectrum = build_mass_spectrum_bar(results)
            channel_plots_spectrum.object = spectrum

            # Overlay plots
            overlay_corr = build_all_channels_overlay(results, plot_type="correlator")
            channel_plots_overlay_corr.object = overlay_corr

            overlay_meff = build_all_channels_overlay(results, plot_type="effective_mass")
            channel_plots_overlay_meff.object = overlay_meff

        def _update_channel_mass_table(
            results: dict[str, ChannelCorrelatorResult],
            mode: str = "AIC-Weighted",
        ) -> None:
            """Update the channel mass table with extracted masses."""
            rows = []
            for name, result in results.items():
                if result.n_samples == 0:
                    continue

                if mode == "AIC-Weighted":
                    mass = result.mass_fit.get("mass", 0.0)
                    mass_error = result.mass_fit.get("mass_error", float("inf"))
                    r2 = result.mass_fit.get("r_squared", float("nan"))
                else:  # Best Window
                    best_window = result.mass_fit.get("best_window", {})
                    mass = best_window.get("mass", 0.0)
                    mass_error = 0.0  # No error for single window
                    r2 = best_window.get("r2", float("nan"))

                n_windows = result.mass_fit.get("n_valid_windows", 0)
                rows.append({
                    "channel": name,
                    "mass": f"{mass:.6f}" if mass > 0 else "n/a",
                    "mass_error": f"{mass_error:.6f}" if mass_error < float("inf") else "n/a",
                    "r2": f"{r2:.4f}" if np.isfinite(r2) else "n/a",
                    "n_windows": n_windows,
                    "n_samples": result.n_samples,
                })
            channel_mass_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        def _update_channel_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            """Update all channel tables from computed results."""
            if mode is None:
                mode = channel_mass_mode.value

            # 1. Update raw mass table (existing)
            _update_channel_mass_table(results, mode)

            # 2. Update ratio pane
            channel_ratio_pane.object = _format_channel_ratios(results, mode)

            # 3. Extract family-mapped masses
            channel_masses = _extract_channel_masses(results, mode)
            channel_r2 = _extract_channel_r2(results, mode)

            if not channel_masses:
                channel_fit_table.value = pd.DataFrame()
                channel_anchor_table.value = pd.DataFrame()
                return

            # 4. Update best-fit table (reuse existing function)
            fit_rows = _build_best_fit_rows(channel_masses, channel_r2)
            channel_fit_table.value = pd.DataFrame(fit_rows)

            # 5. Update anchor table
            glueball_ref = None
            if channel_glueball_ref_input.value is not None:
                glueball_ref = ("glueball", float(channel_glueball_ref_input.value))

            # sqrt_sigma not available from channels, pass None
            anchor_rows = _build_anchor_rows(
                channel_masses,
                glueball_ref,
                sqrt_sigma_ref=None,
                r2s=channel_r2,
            )
            channel_anchor_table.value = pd.DataFrame(anchor_rows)

        def _on_channel_mass_mode_change(event):
            """Handle mass mode toggle changes - updates all tables."""
            if "channel_results" in state:
                _update_channel_tables(state["channel_results"], event.new)

        channel_mass_mode.param.watch(_on_channel_mass_mode_change, "value")

        def _on_heatmap_metric_change(_event):
            if "channel_results" in state:
                _update_channel_plots(state["channel_results"])

        heatmap_color_metric.param.watch(_on_heatmap_metric_change, "value")
        heatmap_alpha_metric.param.watch(_on_heatmap_metric_change, "value")

        def _update_strong_couplings(history: RunHistory | None) -> None:
            couplings = _compute_coupling_constants(
                history,
                h_eff=float(channel_settings.h_eff),
            )
            strong_coupling_table.value = pd.DataFrame(_build_strong_coupling_rows(couplings))

        def on_run_channels(_):
            """Compute channels using vectorized correlator_channels (new code)."""
            history = state.get("history")
            if history is None:
                channels_status.object = "**Error:** Load a RunHistory first."
                return

            channels_status.object = "**Computing channels (vectorized)...**"

            try:
                results = _compute_channels_vectorized(history, channel_settings)
                state["channel_results"] = results

                # Update plots
                _update_channel_plots(results)

                # Update all tables (mass table, ratios, best-fit, anchored)
                _update_channel_tables(results)
                _update_strong_couplings(history)

                n_channels = len([r for r in results.values() if r.n_samples > 0])
                channels_status.object = f"**Complete:** {n_channels} channels computed."
            except Exception as e:
                channels_status.object = f"**Error:** {e}"
                import traceback
                traceback.print_exc()

        # =====================================================================
        # Electroweak tab callbacks (U1/SU2 correlators)
        # =====================================================================

        def _update_electroweak_plots(results: dict[str, ChannelCorrelatorResult]) -> None:
            corr_plots = []
            meff_plots = []

            for name, result in results.items():
                if result.n_samples == 0:
                    continue

                corr = (
                    result.correlator.cpu().numpy()
                    if hasattr(result.correlator, "cpu")
                    else np.asarray(result.correlator)
                )
                meff = (
                    result.effective_mass.cpu().numpy()
                    if hasattr(result.effective_mass, "cpu")
                    else np.asarray(result.effective_mass)
                )
                lag_times = np.arange(len(corr)) * result.dt
                meff_times = np.arange(len(meff)) * result.dt

                corr_plot = build_correlator_plot(lag_times, corr, result.mass_fit, name)
                if corr_plot is not None:
                    corr_plots.append(
                        pn.pane.HoloViews(corr_plot, sizing_mode="stretch_width", linked_axes=False)
                    )

                meff_plot = build_effective_mass_plot(meff_times, meff, result.mass_fit, name)
                if meff_plot is not None:
                    meff_plots.append(
                        pn.pane.HoloViews(meff_plot, sizing_mode="stretch_width", linked_axes=False)
                    )

            electroweak_plots_correlator.objects = corr_plots if corr_plots else [
                pn.pane.Markdown("_No correlator plots available._")
            ]
            electroweak_plots_effective_mass.objects = meff_plots if meff_plots else [
                pn.pane.Markdown("_No effective mass plots available._")
            ]

            electroweak_plots_spectrum.object = build_mass_spectrum_bar(results)
            electroweak_plots_overlay_corr.object = build_all_channels_overlay(
                results, plot_type="correlator"
            )
            electroweak_plots_overlay_meff.object = build_all_channels_overlay(
                results, plot_type="effective_mass"
            )

        def _update_electroweak_mass_table(
            results: dict[str, ChannelCorrelatorResult],
            mode: str = "AIC-Weighted",
        ) -> None:
            rows = []
            for name, result in results.items():
                if result.n_samples == 0:
                    continue

                if mode == "AIC-Weighted":
                    mass = result.mass_fit.get("mass", 0.0)
                    mass_error = result.mass_fit.get("mass_error", float("inf"))
                    r2 = result.mass_fit.get("r_squared", float("nan"))
                else:
                    best_window = result.mass_fit.get("best_window", {})
                    mass = best_window.get("mass", 0.0)
                    mass_error = 0.0
                    r2 = best_window.get("r2", float("nan"))

                n_windows = result.mass_fit.get("n_valid_windows", 0)
                rows.append(
                    {
                        "channel": name,
                        "mass": f"{mass:.6f}" if mass > 0 else "n/a",
                        "mass_error": (
                            f"{mass_error:.6f}" if mass_error < float("inf") else "n/a"
                        ),
                        "r2": f"{r2:.4f}" if np.isfinite(r2) else "n/a",
                        "n_windows": n_windows,
                        "n_samples": result.n_samples,
                    }
                )
            electroweak_mass_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        def _update_electroweak_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = electroweak_mass_mode.value
            _update_electroweak_mass_table(results, mode)
            electroweak_ratio_pane.object = _format_electroweak_ratios(results, mode)

            masses = _extract_electroweak_masses(results, mode)
            r2s = _extract_electroweak_r2(results, mode)

            base_name = "u1_dressed" if "u1_dressed" in masses else "u1_phase"

            refs_df = electroweak_ref_table.value
            refs: dict[str, float] = {}
            if isinstance(refs_df, pd.DataFrame):
                for _, row in refs_df.iterrows():
                    name = str(row.get("channel", "")).strip()
                    try:
                        mass_raw = row.get("mass_ref_GeV")
                        if isinstance(mass_raw, str):
                            mass_raw = mass_raw.strip()
                            if mass_raw == "":
                                continue
                        mass_ref = float(mass_raw)
                    except (TypeError, ValueError):
                        continue
                    if name and mass_ref > 0:
                        refs[name] = mass_ref

            ratio_rows = _build_electroweak_ratio_rows(masses, base_name, refs=refs)
            electroweak_ratio_table.value = pd.DataFrame(ratio_rows) if ratio_rows else pd.DataFrame()

            if not masses or not refs:
                electroweak_fit_table.value = pd.DataFrame()
                electroweak_anchor_table.value = pd.DataFrame()
                electroweak_compare_table.value = pd.DataFrame()
                return

            fit_rows = _build_electroweak_best_fit_rows(masses, refs, r2s)
            electroweak_fit_table.value = pd.DataFrame(fit_rows)

            anchor_rows = _build_electroweak_anchor_rows(masses, refs, r2s)
            electroweak_anchor_table.value = pd.DataFrame(anchor_rows)

            compare_rows = _build_electroweak_comparison_rows(masses, refs)
            electroweak_compare_table.value = pd.DataFrame(compare_rows)

        def _on_electroweak_mass_mode_change(event) -> None:
            if "electroweak_results" in state:
                _update_electroweak_tables(state["electroweak_results"], event.new)

        electroweak_mass_mode.param.watch(_on_electroweak_mass_mode_change, "value")

        def on_run_electroweak(_):
            history = state.get("history")
            if history is None:
                electroweak_status.object = "**Error:** Load a RunHistory first."
                return

            electroweak_status.object = "**Computing electroweak channels...**"
            try:
                results = _compute_electroweak_channels(history, electroweak_settings)
                state["electroweak_results"] = results

                _update_electroweak_plots(results)
                _update_electroweak_tables(results)
                couplings = _compute_coupling_constants(
                    history,
                    h_eff=float(electroweak_settings.h_eff),
                    epsilon_d=electroweak_settings.epsilon_d,
                    epsilon_c=electroweak_settings.epsilon_c,
                )
                electroweak_coupling_table.value = pd.DataFrame(
                    _build_coupling_rows(
                        couplings,
                        proxies=None,
                        include_strong=False,
                        refs=_extract_coupling_refs(),
                    )
                )

                n_channels = len([r for r in results.values() if r.n_samples > 0])
                electroweak_status.object = (
                    f"**Complete:** {n_channels} electroweak channels computed."
                )
            except Exception as e:
                electroweak_status.object = f"**Error:** {e}"
                import traceback
                traceback.print_exc()

        browse_button.on_click(_on_browse_clicked)
        load_button.on_click(on_load_clicked)
        gas_config.add_completion_callback(on_simulation_complete)
        gas_config.param.watch(on_bounds_change, "bounds_extent")
        run_analysis_button.on_click(on_run_analysis)
        run_analysis_button_main.on_click(on_run_analysis)
        particle_run_button.on_click(on_run_particles)
        sweep_run_button.on_click(on_run_sweep)
        channels_run_button.on_click(on_run_channels)
        electroweak_run_button.on_click(on_run_electroweak)
        sweep_enable_2d.param.watch(_toggle_sweep_controls, "value")
        sweep_param_x.param.watch(_on_sweep_param_x, "value")
        sweep_param_y.param.watch(_on_sweep_param_y, "value")

        visualization_controls = pn.Param(
            visualizer,
            parameters=["point_size", "point_alpha", "color_metric", "fix_axes"],
            show_name=False,
        )

        analysis_core = pn.Param(
            analysis_settings,
            parameters=[
                "analysis_time_index",
                "analysis_step",
                "warmup_fraction",
                "h_eff",
                "correlation_r_max",
                "correlation_bins",
                "gradient_neighbors",
                "build_fractal_set",
                "fractal_set_stride",
            ],
            show_name=False,
        )
        analysis_local = pn.Param(
            analysis_settings,
            parameters=["use_local_fields", "use_connected", "density_sigma"],
            show_name=False,
        )
        analysis_particles = pn.Param(
            analysis_settings,
            parameters=[
                "compute_particles",
                "particle_operators",
                "particle_max_lag",
                "particle_fit_start",
                "particle_fit_stop",
                "particle_fit_mode",
                "particle_plateau_min_points",
                "particle_plateau_max_points",
                "particle_plateau_max_cv",
                "particle_mass",
                "particle_ell0",
                "particle_use_connected",
                "particle_neighbor_method",
                "particle_knn_k",
                "particle_knn_sample",
                "particle_meson_reduce",
                "particle_baryon_pairs",
                "particle_voronoi_weight",
                "particle_voronoi_pbc_mode",
                "particle_voronoi_normalize",
                "particle_voronoi_max_triplets",
                "particle_voronoi_exclude_boundary",
                "particle_voronoi_boundary_tolerance",
                "compute_curvature_proxies",
                "curvature_compute_interval",
            ],
            show_name=False,
        )
        analysis_string = pn.Param(
            analysis_settings,
            parameters=[
                "compute_string_tension",
                "string_tension_max_triangles",
                "string_tension_bins",
            ],
            show_name=False,
        )

        analysis_output = pn.Column(
            analysis_output_dir,
            analysis_id_input,
            run_analysis_button,
            analysis_status_sidebar,
            sizing_mode="stretch_width",
        )
        particle_anchor_controls = pn.Column(
            glueball_label_input,
            glueball_ref_input,
            sqrt_sigma_ref_input,
            sizing_mode="stretch_width",
        )
        sweep_controls = pn.Column(
            pn.pane.Markdown("### Sweep Controls"),
            pn.Row(sweep_enable_2d, sweep_metric, sizing_mode="stretch_width"),
            pn.Row(
                sweep_param_x,
                sweep_min_x,
                sweep_max_x,
                sweep_steps_x,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                sweep_param_y,
                sweep_min_y,
                sweep_max_y,
                sweep_steps_y,
                sizing_mode="stretch_width",
            ),
            pn.Row(sweep_run_button, sizing_mode="stretch_width"),
            sweep_status,
            pn.layout.Divider(),
            sweep_plot,
            sweep_table,
            sizing_mode="stretch_width",
        )

        if skip_sidebar:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\n" "Sidebar disabled via QFT_DASH_SKIP_SIDEBAR=1."
                ),
                pn.pane.Markdown("### Load QFT RunHistory"),
                history_path_input,
                browse_button,
                file_selector_container,
                load_button,
                load_status,
            ]
        else:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\n"
                    "Run a simulation or load a RunHistory, then analyze results."
                ),
                pn.Accordion(
                    (
                        "RunHistory",
                        pn.Column(
                            history_path_input,
                            browse_button,
                            file_selector_container,
                            load_button,
                            load_status,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    ("Simulation", gas_config.panel()),
                    ("Visualization", visualization_controls),
                    ("Analysis: Core", analysis_core),
                    ("Analysis: Local Fields", analysis_local),
                    ("Analysis: Particles", analysis_particles),
                    ("Analysis: String Tension", analysis_string),
                    ("Analysis: Output", analysis_output),
                    ("Particle Anchors", particle_anchor_controls),
                    ("Electroweak Channels", electroweak_settings_panel),
                    sizing_mode="stretch_width",
                ),
            ]

        if skip_visual:
            main.objects = [
                pn.pane.Markdown(
                    "Visualization disabled via QFT_DASH_SKIP_VIS=1.",
                    sizing_mode="stretch_both",
                )
            ]
        else:
            simulation_tab = pn.Column(visualizer.panel(), sizing_mode="stretch_both")
            analysis_tab = pn.Column(
                analysis_status_main,
                pn.Row(run_analysis_button_main, sizing_mode="stretch_width"),
                analysis_summary,
                pn.layout.Divider(),
                analysis_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("## Raw Metrics"),
                analysis_json,
                sizing_mode="stretch_both",
            )
            particle_tab = pn.Column(
                particle_status,
                pn.Row(particle_run_button, sizing_mode="stretch_width"),
                sweep_controls,
                pn.pane.Markdown("### Algorithmic Masses"),
                particle_mass_table,
                particle_ratio_pane,
                pn.pane.Markdown("### Best-Fit Scales"),
                particle_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                particle_anchor_table,
                sizing_mode="stretch_both",
            )

            # New Channels tab (vectorized correlator_channels)
            # Informational alert for time dimension selection
            time_dimension_info = pn.pane.Alert(
                """
**Time Axis Selection:**
- **t (default)**: Use Euclidean time dimension (4th spatial dimension)
- **x, y, z**: Use spatial dimensions as time (correlators along that axis)
- **monte_carlo**: Use Monte Carlo timesteps (standard time evolution)

When using spatial dimensions (t, x, y, z), the analysis ignores the Monte Carlo
dimension and bins walkers by their spatial coordinate.
Set **MC time slice** in the settings to choose which recorded Monte Carlo
configuration to analyze (recorded step or index; blank = last recorded slice).
                """,
                alert_type="info",
                sizing_mode="stretch_width",
            )

            channels_tab = pn.Column(
                channels_status,
                time_dimension_info,
                pn.Row(channels_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Channel Settings", channel_settings_panel),
                    ("Reference Anchors", pn.Column(channel_glueball_ref_input)),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                channel_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                channel_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Mass Spectrum"),
                channel_plots_spectrum,
                pn.pane.Markdown("### Strong Coupling Constants"),
                strong_coupling_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Extracted Masses"),
                channel_mass_mode,
                channel_mass_table,
                channel_ratio_pane,
                pn.pane.Markdown("### Best-Fit Scales"),
                channel_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                channel_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Effective Mass Plateaus"),
                pn.pane.Markdown(
                    "_Two-panel view: correlator decay (left) + effective mass with best window "
                    "region (green) and error band (red)._",
                ),
                channel_plateau_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("### Window Heatmaps"),
                pn.pane.Markdown(
                    "_2D matrix of start × end positions. Color/opacity map to mass, AIC, or R². "
                    "Hover shows mass, AIC, R². Red marker = best window._",
                ),
                pn.Row(
                    heatmap_color_metric,
                    heatmap_alpha_metric,
                    sizing_mode="stretch_width",
                ),
                channel_heatmap_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("### Individual Channel Correlators C(t)"),
                channel_plots_correlator,
                pn.pane.Markdown("### Individual Channel Effective Masses m_eff(t)"),
                channel_plots_effective_mass,
                sizing_mode="stretch_both",
            )

            # Informational alert for time dimension selection
            electroweak_time_dimension_info = pn.pane.Alert(
                """
**Time Axis Selection:**
- **t (default)**: Use Euclidean time dimension (4th spatial dimension)
- **x, y, z**: Use spatial dimensions as time (correlators along that axis)
- **monte_carlo**: Use Monte Carlo timesteps (standard time evolution)

When using spatial dimensions (t, x, y, z), the analysis ignores the Monte Carlo
dimension and bins walkers by their spatial coordinate.
Set **MC time slice** in the settings to choose which recorded Monte Carlo
configuration to analyze (recorded step or index; blank = last recorded slice).
                """,
                alert_type="info",
                sizing_mode="stretch_width",
            )

            electroweak_tab = pn.Column(
                electroweak_status,
                electroweak_time_dimension_info,
                pn.Row(electroweak_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Electroweak Settings", electroweak_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown(
                    "_Electroweak channels are proxy observables built from U(1)/SU(2) phases "
                    "defined in `docs/source/3_fractal_gas/2_fractal_set/04_standard_model.md`._"
                ),
                pn.layout.Divider(),
                electroweak_summary,
                pn.pane.Markdown("### Electroweak Couplings"),
                electroweak_coupling_table,
                pn.pane.Markdown("### Electroweak Coupling References"),
                pn.pane.Markdown(
                    "_Set observed values to compute error percentages for couplings._"
                ),
                electroweak_coupling_ref_table,
                pn.pane.Markdown("### Gauge Phase Histograms"),
                electroweak_phase_plot,
                pn.layout.Divider(),
                pn.pane.Markdown("### Electroweak Mass Spectrum"),
                electroweak_plots_spectrum,
                pn.pane.Markdown("### Extracted Masses"),
                electroweak_mass_mode,
                electroweak_mass_table,
                electroweak_ratio_pane,
                pn.pane.Markdown("### Electroweak Ratios"),
                electroweak_ratio_table,
                pn.pane.Markdown("### Electroweak Reference Masses (GeV)"),
                pn.pane.Markdown(
                    "_Edit the table below to set observed masses for calibration._"
                ),
                electroweak_ref_table,
                pn.pane.Markdown("### Best-Fit Scales"),
                electroweak_fit_table,
                pn.pane.Markdown("### Measured vs Observed"),
                pn.pane.Markdown(
                    "_Best-fit scale applied to all electroweak channels; "
                    "error shows percent deviation from observed masses._"
                ),
                electroweak_compare_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                electroweak_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                electroweak_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                electroweak_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Individual Channel Correlators C(t)"),
                electroweak_plots_correlator,
                pn.pane.Markdown("### Individual Channel Effective Masses m_eff(t)"),
                electroweak_plots_effective_mass,
                sizing_mode="stretch_both",
            )

            main.objects = [
                pn.Tabs(
                    ("Simulation", simulation_tab),
                    ("Analysis", analysis_tab),
                    ("Particles", particle_tab),
                    ("Strong Force", channels_tab),
                    ("Electroweak", electroweak_tab),
                )
            ]

        _debug(f"ui ready ({time.time() - start_total:.2f}s)")

    pn.state.onload(_build_ui)
    return template


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--open", action="store_true", help="Open browser on launch")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting QFT Swarm Convergence Dashboard...", flush=True)
    app = create_app()
    print(
        f"QFT Swarm Convergence Dashboard running at http://localhost:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    app.show(port=args.port, open=args.open)
