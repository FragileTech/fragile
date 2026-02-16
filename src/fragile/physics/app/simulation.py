"""Simulation tab: 3D swarm convergence viewer and RunHistory I/O controls."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import holoviews as hv
import numpy as np
import panel as pn
import param
import torch

from fragile.physics.app.gas_config_panel import GasConfigPanel
from fragile.physics.fractal_gas.history import RunHistory


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
        default="monte_carlo",
        objects=["monte_carlo", "euclidean"],
        doc="Player axis: Monte Carlo time or Euclidean time slices",
    )
    mc_time_index = param.Integer(
        default=None,
        bounds=(0, None),
        allow_None=True,
        doc="Monte Carlo slice (recorded step or index) for Euclidean visualization",
    )
    euclidean_time_dim = param.Integer(
        default=3,
        bounds=(0, 10),
        doc="Spatial dimension index for Euclidean time slices (0-indexed)",
    )
    euclidean_time_bins = param.Integer(
        default=50,
        bounds=(5, 500),
        doc="Number of Euclidean time slices",
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
            "riemannian_volume",
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
    line_color_metric = param.ObjectSelector(
        default="constant",
        objects=["constant", "distance", "geodesic"],
        doc="Delaunay edge color metric",
    )
    line_colorscale = param.ObjectSelector(
        default="Viridis",
        objects=["Viridis", "Plasma", "Cividis", "Turbo", "Magma", "Inferno"],
        doc="Colorscale for distance-colored edges",
    )
    def __init__(self, history: RunHistory | None, bounds_extent: float = 10.0, **params):
        super().__init__(**params)
        self.history = history
        self.bounds_extent = float(bounds_extent)

        self._x = None
        self._fitness = None
        self._rewards = None
        self._volume_weights = None
        self._alive = None
        self._neighbor_edges = None
        self._geodesic_edge_distances = None
        self._neighbor_graph_method = None
        self._data_d = None
        self._last_mc_frame = 0
        self._mc_slice_controls = None
        self._time_bin_controls = None
        self._time_distribution_pane = None
        self._time_distribution_container = None
        self._player_mode_pane = None
        self._camera_state = None

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
        self.plot_pane.param.watch(self._on_plot_relayout, "relayout_data")

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
                "line_color_metric",
                "line_colorscale",
                "euclidean_time_dim",
                "euclidean_time_bins",
                "mc_time_index",
            ],
        )
        self.param.watch(
            self._sync_player_range,
            ["x_axis_dim", "y_axis_dim", "z_axis_dim", "euclidean_time_dim"],
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
        self._volume_weights = (
            history.riemannian_volume_weights.detach().cpu().numpy()
            if getattr(history, "riemannian_volume_weights", None) is not None
            else None
        )
        self._alive = history.alive_mask.detach().cpu().numpy().astype(bool)
        self._neighbor_edges = history.neighbor_edges
        self._geodesic_edge_distances = history.geodesic_edge_distances
        self._data_d = int(self._x.shape[2]) if self._x is not None and self._x.ndim >= 3 else None
        if history.params is not None:
            self._neighbor_graph_method = history.params.get("neighbor_graph", {}).get("method")

        # Update dimension options based on history.d
        d = self._data_d if self._data_d is not None else history.d
        dim_options = [f"dim_{i}" for i in range(d)] + ["mc_time", "euclidean_time"]

        # Update parameter objects dynamically
        self.param.x_axis_dim.objects = dim_options
        self.param.y_axis_dim.objects = dim_options
        self.param.z_axis_dim.objects = dim_options
        # For color, include all dimensions plus existing metrics
        color_options = [
            *dim_options,
            "fitness",
            "reward",
            "riemannian_volume",
            "radius",
            "constant",
        ]
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
        self._last_mc_frame = max(0, history.n_recorded - 1)
        self.time_player.disabled = False
        self._update_time_player_range(reset_value=True)

        if self._data_d is not None and self._data_d != history.d:
            dim_note = f"{history.d} (history) / {self._data_d} (data)"
        else:
            dim_note = f"{d}"
        self.status_pane.object = (
            f"**RunHistory loaded:** N={history.N}, "
            f"steps={history.n_steps}, "
            f"recorded={history.n_recorded}, "
            f"dims={dim_note}"
        )
        if self._neighbor_edges is None:
            self.status_pane.object += (
                " | Neighbor graph not recorded (rerun with neighbor_graph_method='delaunay' and "
                "neighbor_graph_record=True to show edges)"
            )
        elif self._neighbor_graph_method and self._neighbor_graph_method != "delaunay":
            self.status_pane.object += f" | Neighbor graph method: {self._neighbor_graph_method}"
        self._refresh_frame()

    def _sync_frame(self, event):
        if self.history is None:
            return
        if self._use_mc_time_player():
            self._last_mc_frame = int(np.clip(event.new, 0, self.history.n_recorded - 1))
        self._update_plot(int(event.new))

    def _sync_player_range(self, *_):
        if self.history is None:
            return
        self._update_time_player_range(reset_value=False)
        self._update_player_mode()

    def _update_player_mode(self) -> None:
        if self._player_mode_pane is None:
            return
        if self.history is None:
            self._player_mode_pane.object = "**Player:** idle"
            return
        d = self._data_d if self._data_d is not None else self.history.d
        if self._use_mc_time_player():
            if self.time_iteration == "monte_carlo":
                mode = "Monte Carlo frames"
            else:
                mode = "Monte Carlo frames (slice dim not shown)"
        elif self._slice_dim_displayed(d):
            mode = f"{self.time_iteration.capitalize()} bins (slice highlighted)"
        else:
            mode = "Euclidean bins (slice dim not shown)"
        if self._axis_uses_mc_time():
            mode = f"{mode} | MC time plotted across frames"
        self._player_mode_pane.object = f"**Player:** {mode}"

    def _refresh_frame(self, *_):
        if self.history is None:
            return
        self._update_time_distribution()
        self._update_player_mode()
        self._update_plot(int(self.time_player.value))

    def _on_time_iteration_change(self, event):
        if self.history is None:
            return
        if event.name == "time_iteration":
            if event.new == "euclidean":
                self._last_mc_frame = int(
                    np.clip(self.time_player.value, 0, self.history.n_recorded - 1)
                )
                self._update_time_player_range(reset_value=True)
            elif event.new == "monte_carlo":
                self._update_time_player_range(reset_value=False)
                self.time_player.value = int(
                    np.clip(self._last_mc_frame, 0, self.history.n_recorded - 1)
                )
        elif event.name == "euclidean_time_bins" and self.time_iteration == "euclidean":
            self._update_time_player_range(reset_value=False)
        self._sync_mc_slice_visibility()
        self._sync_time_bin_visibility()
        self._refresh_frame()

    def _update_time_player_range(self, reset_value: bool) -> None:
        if self.history is None:
            self.time_player.start = 0
            self.time_player.end = 0
            self.time_player.value = 0
            return
        if self._use_mc_time_player():
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
        self._mc_slice_controls.visible = self.time_iteration == "euclidean"

    def _sync_time_bin_visibility(self) -> None:
        if self._time_bin_controls is None:
            return
        visible = self.time_iteration == "euclidean"
        self._time_bin_controls.visible = visible
        # Toggle an outer container instead of the HoloViews pane itself.
        # This avoids forwarding unsupported layout props to underlying Bokeh figures
        # on some Panel/Bokeh version combinations.
        if self._time_distribution_container is not None:
            self._time_distribution_container.visible = visible

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
        if self.time_iteration == "euclidean" and not self._use_mc_time_player() and bins > 0:
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
        if self._use_mc_time_player():
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

    def _resolve_euclidean_dim(self, d: int) -> int:
        dim = int(self.euclidean_time_dim)
        if dim >= d:
            return max(0, d - 1)
        if dim < 0:
            return 0
        return dim

    def _axis_uses_dim(self, dim_spec: str, dim_idx: int, d: int) -> bool:
        if dim_spec == "euclidean_time":
            return dim_idx == self._resolve_euclidean_dim(d)
        if dim_spec.startswith("dim_"):
            try:
                return int(dim_spec.split("_")[1]) == dim_idx
            except (TypeError, ValueError):
                return False
        return False

    def _axis_uses_mc_time(self) -> bool:
        return (
            self.x_axis_dim == "mc_time"
            or self.y_axis_dim == "mc_time"
            or self.z_axis_dim == "mc_time"
        )

    def _slice_dim_displayed(self, d: int) -> bool:
        if self.time_iteration != "euclidean":
            return False
        slice_dim = self._resolve_euclidean_dim(d)
        for dim_spec in (self.x_axis_dim, self.y_axis_dim, self.z_axis_dim):
            if self._axis_uses_dim(dim_spec, slice_dim, d):
                return True
        return False

    def _use_mc_time_player(self) -> bool:
        if self.history is None:
            return True
        if self.time_iteration == "monte_carlo":
            return True
        d = self._data_d if self._data_d is not None else self.history.d
        return not self._slice_dim_displayed(d)

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
            return f"QFT Swarm Convergence (frame {mc_frame}, step {step}, slice {slice_index})"
        low, high = slice_bounds
        label_text = (
            f"{label} in [{low:.2f}, {high:.2f})"
            if label
            else (f"time in [{low:.2f}, {high:.2f})")
        )
        mode = f", {slice_mode}" if slice_mode else ""
        return (
            f"QFT Swarm Convergence (frame {mc_frame}, step {step}, slice {slice_index}{mode}, "
            f"{label_text})"
        )

    @staticmethod
    def _normalize_edges(edges: np.ndarray | None) -> np.ndarray:
        if edges is None:
            return np.zeros((0, 2), dtype=np.int64)
        edges = np.asarray(edges)
        if edges.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        edges = edges.reshape(-1, 2)
        edges = edges[edges[:, 0] != edges[:, 1]]
        if edges.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    def _get_delaunay_edges(self, frame: int, normalize: bool = True) -> np.ndarray | None:
        if self.history is None or self._neighbor_edges is None:
            return None
        if frame < 0 or frame >= len(self._neighbor_edges):
            return None
        edges = self._neighbor_edges[frame]
        if edges is None:
            return None
        if isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
        if normalize:
            edges = self._normalize_edges(edges)
        return edges if edges.size else None

    def _get_geodesic_edge_distances(self, frame: int) -> np.ndarray | None:
        if self.history is None or self._geodesic_edge_distances is None:
            return None
        if frame < 0 or frame >= len(self._geodesic_edge_distances):
            return None
        distances = self._geodesic_edge_distances[frame]
        if distances is None:
            return None
        if torch.is_tensor(distances):
            distances = distances.detach().cpu().numpy()
        distances = np.asarray(distances)
        return distances if distances.size else None

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
        if dim_spec == "mc_time":
            n_alive = alive.sum()
            return np.full(n_alive, frame, dtype=float)
        if dim_spec == "euclidean_time":
            dim_idx = self._resolve_euclidean_dim(positions_all.shape[1])
            if dim_idx >= positions_all.shape[1]:
                return np.zeros(alive.sum())
            return positions_all[alive, dim_idx]

        if dim_spec.startswith("dim_"):
            dim_idx = int(dim_spec.split("_")[1])
            if dim_idx >= positions_all.shape[1]:
                return np.zeros(alive.sum())
            return positions_all[alive, dim_idx]

        return np.zeros(alive.sum())

    def _get_color_values(self, frame: int, positions_all: np.ndarray, alive: np.ndarray):
        metric = self.color_metric

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

        if metric == "mc_time":
            colors = np.full(alive.sum(), frame, dtype=float)
            return colors, True, {"title": "MC Time (frame)"}

        if metric == "fitness":
            if frame == 0 or self._fitness is None:
                colors = "#1f77b4"
                return colors, False, None
            idx = min(frame - 1, len(self._fitness) - 1)
            colors = self._fitness[idx][alive]
            return colors, True, {"title": "Fitness"}

        if metric == "reward":
            if frame == 0 or self._rewards is None:
                colors = "#1f77b4"
                return colors, False, None
            idx = min(frame - 1, len(self._rewards) - 1)
            colors = self._rewards[idx][alive]
            return colors, True, {"title": "Reward"}

        if metric == "riemannian_volume":
            if frame == 0 or self._volume_weights is None:
                colors = "#1f77b4"
                return colors, False, None
            idx = min(frame - 1, len(self._volume_weights) - 1)
            colors = self._volume_weights[idx][alive]
            return colors, True, {"title": "Riemannian Volume"}

        if metric == "radius":
            positions_filtered = positions_all[alive][:, : min(3, positions_all.shape[1])]
            colors = np.linalg.norm(positions_filtered, axis=1)
            return colors, True, {"title": "Radius"}

        # "constant"
        return "#1f77b4", False, None

    def _get_color_values_all_frames(
        self,
        n_frames: int,
    ) -> tuple[np.ndarray | str, bool, dict[str, Any] | None]:
        metric = self.color_metric
        if metric == "constant":
            return "#1f77b4", False, None

        colors: list[np.ndarray] = []
        for frame in range(n_frames):
            positions_all = self._x[frame]
            alive = self._get_alive_mask(frame)

            if metric.startswith("dim_"):
                dim_idx = int(metric.split("_")[1])
                if dim_idx < positions_all.shape[1]:
                    colors.append(positions_all[alive, dim_idx])
                else:
                    colors.append(np.zeros(alive.sum()))
                continue

            if metric == "euclidean_time":
                dim_idx = self._resolve_euclidean_dim(positions_all.shape[1])
                if dim_idx < positions_all.shape[1]:
                    colors.append(positions_all[alive, dim_idx])
                else:
                    colors.append(np.zeros(alive.sum()))
                continue

            if metric == "mc_time":
                colors.append(np.full(alive.sum(), frame, dtype=float))
                continue

            if metric == "fitness":
                if frame == 0 or self._fitness is None:
                    colors.append(np.full(alive.sum(), np.nan))
                else:
                    idx = min(frame - 1, len(self._fitness) - 1)
                    colors.append(self._fitness[idx][alive])
                continue

            if metric == "reward":
                if frame == 0 or self._rewards is None:
                    colors.append(np.full(alive.sum(), np.nan))
                else:
                    idx = min(frame - 1, len(self._rewards) - 1)
                    colors.append(self._rewards[idx][alive])
                continue

            if metric == "riemannian_volume":
                if frame == 0 or self._volume_weights is None:
                    colors.append(np.full(alive.sum(), np.nan))
                else:
                    idx = min(frame - 1, len(self._volume_weights) - 1)
                    colors.append(self._volume_weights[idx][alive])
                continue

            if metric == "radius":
                positions_filtered = positions_all[alive][:, : min(3, positions_all.shape[1])]
                colors.append(np.linalg.norm(positions_filtered, axis=1))
                continue

            colors.append(np.zeros(alive.sum()))

        if not colors:
            return "#1f77b4", False, None
        return np.concatenate(colors), True, {"title": self._axis_label(metric)}

    def _get_axis_ranges(self, frame: int):
        def get_range(dim_spec: str):
            if dim_spec == "mc_time":
                return [0, len(self._x) - 1]
            if dim_spec == "euclidean_time":
                extent = self.bounds_extent
                return [-extent, extent]
            if dim_spec.startswith("dim_"):
                extent = self.bounds_extent
                return [-extent, extent]
            return [-10, 10]

        return {
            "xaxis": {"range": get_range(self.x_axis_dim)},
            "yaxis": {"range": get_range(self.y_axis_dim)},
            "zaxis": {"range": get_range(self.z_axis_dim)},
        }

    def _axis_label(self, dim_spec: str) -> str:
        if dim_spec == "mc_time":
            return "Monte Carlo Time (frame)"
        if dim_spec == "euclidean_time":
            dim_idx = 0
            if self.history is not None:
                d = self._data_d if self._data_d is not None else self.history.d
                dim_idx = self._resolve_euclidean_dim(d)
            return f"Euclidean Time (dim_{dim_idx})"
        if dim_spec == "riemannian_volume":
            return "Riemannian Volume"
        if dim_spec.startswith("dim_"):
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
        use_geodesic = self.line_color_metric == "geodesic"
        edges = self._get_delaunay_edges(frame, normalize=not use_geodesic)
        edge_values = None
        if use_geodesic:
            edge_values = self._get_geodesic_edge_distances(frame)
            if (
                edge_values is not None
                and edges is not None
                and edge_values.shape[0] != edges.shape[0]
            ):
                edge_values = None
        if edges is None or positions_mapped.size == 0:
            return None
        return self._build_delaunay_trace_mapped_from_edges(
            edges,
            positions_all,
            alive,
            positions_mapped,
            edge_values=edge_values,
            trace_name="Delaunay edges",
        )

    def _build_delaunay_trace_mapped_from_edges(
        self,
        edges: np.ndarray,
        positions_all: np.ndarray,
        alive: np.ndarray,
        positions_mapped: np.ndarray,
        *,
        edge_values: np.ndarray | None = None,
        color_override: str | None = None,
        color_mode: str | None = None,
        trace_name: str = "Delaunay edges",
        showlegend: bool = False,
    ):
        import plotly.graph_objects as go

        if edges is None or edges.size == 0 or positions_mapped.size == 0:
            return None

        alive_indices = np.where(alive)[0]
        if alive_indices.size == 0:
            return None
        alive_map = {int(idx): pos for pos, idx in enumerate(alive_indices)}

        valid_edges: list[tuple[int, int, int, int]] = []
        values: list[float] | None = [] if edge_values is not None else None
        seen: set[tuple[int, int]] = set()
        for idx, (i, j) in enumerate(edges):
            if i == j:
                continue
            i_local = alive_map.get(int(i))
            j_local = alive_map.get(int(j))
            if i_local is None or j_local is None:
                continue
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            valid_edges.append((int(i), int(j), i_local, j_local))
            if values is not None:
                values.append(float(edge_values[idx]))

        if not valid_edges:
            return None

        x_edges, y_edges, z_edges = [], [], []
        metric = color_mode or self.line_color_metric
        label_metric = metric
        edge_values_list: list[float] | None = None
        raw_values = None
        distances = None
        if metric == "distance":
            edge_values_list = []
            pairs = np.array([(i, j) for i, j, _, _ in valid_edges], dtype=np.int64)
            deltas = positions_all[pairs[:, 0]] - positions_all[pairs[:, 1]]
            distances = np.linalg.norm(deltas, axis=1)
            raw_values = distances
        elif metric == "geodesic":
            edge_values_list = []
            if values is not None and len(values) == len(valid_edges):
                raw_values = np.asarray(values, dtype=float)
            else:
                label_metric = "distance"
                pairs = np.array([(i, j) for i, j, _, _ in valid_edges], dtype=np.int64)
                deltas = positions_all[pairs[:, 0]] - positions_all[pairs[:, 1]]
                distances = np.linalg.norm(deltas, axis=1)
                raw_values = distances

        for idx, (i, j, i_local, j_local) in enumerate(valid_edges):
            x_edges.extend([positions_mapped[i_local, 0], positions_mapped[j_local, 0], None])
            y_edges.extend([positions_mapped[i_local, 1], positions_mapped[j_local, 1], None])
            z_edges.extend([positions_mapped[i_local, 2], positions_mapped[j_local, 2], None])
            if edge_values_list is not None:
                value = float(raw_values[idx]) if raw_values is not None else float("nan")
                edge_values_list.extend([value, value, np.nan])

        if edge_values_list is not None:
            line = {
                "color": edge_values_list,
                "width": float(self.line_width),
                "dash": self.line_style,
                "colorscale": self.line_colorscale,
                "showscale": True,
                "colorbar": {
                    "title": "Geodesic distance"
                    if label_metric == "geodesic"
                    else "Euclidean distance"
                },
            }
            if raw_values is not None and np.asarray(raw_values).size:
                line["cmin"] = float(np.nanmin(raw_values))
                line["cmax"] = float(np.nanmax(raw_values))
            return go.Scatter3d(
                x=x_edges,
                y=y_edges,
                z=z_edges,
                mode="lines",
                line=line,
                opacity=float(self.line_alpha),
                hoverinfo="skip",
                name=trace_name,
                showlegend=showlegend,
            )

        line_color = self._rgba_from_color(
            color_override or self.line_color, float(self.line_alpha)
        )
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
            name=trace_name,
            showlegend=showlegend,
        )

    def _build_delaunay_trace(self, frame: int, positions: np.ndarray, alive_mask: np.ndarray):
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
        use_all_frames = self._axis_uses_mc_time()
        slice_index = None
        slice_bounds = None
        slice_dim = None
        slice_mode = None
        slice_mask = None
        slice_dim_visible = False
        if self.time_iteration == "euclidean":
            slice_index = int(np.clip(frame, 0, max(0, int(self.euclidean_time_bins) - 1)))
            slice_dim = self._resolve_euclidean_dim(positions_all.shape[1])
            slice_mode = "euclidean"
            slice_dim_visible = (
                self._axis_uses_dim(self.x_axis_dim, slice_dim, positions_all.shape[1])
                or self._axis_uses_dim(self.y_axis_dim, slice_dim, positions_all.shape[1])
                or self._axis_uses_dim(self.z_axis_dim, slice_dim, positions_all.shape[1])
            )
            if slice_dim_visible:
                slice_mask, slice_bounds, slice_dim = self._get_slice_mask(
                    positions_all, slice_index, slice_dim
                )
            else:
                slice_index = None
                slice_bounds = None
                slice_dim = None
                slice_mode = None
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

        if use_all_frames:
            n_frames = self._x.shape[0]
            x_all: list[np.ndarray] = []
            y_all: list[np.ndarray] = []
            z_all: list[np.ndarray] = []
            for idx in range(n_frames):
                pos = self._x[idx]
                alive_f = self._get_alive_mask(idx)
                x_all.append(self._extract_dimension(self.x_axis_dim, idx, pos, alive_f))
                y_all.append(self._extract_dimension(self.y_axis_dim, idx, pos, alive_f))
                z_all.append(self._extract_dimension(self.z_axis_dim, idx, pos, alive_f))
            x_coords = np.concatenate(x_all) if x_all else np.array([])
            y_coords = np.concatenate(y_all) if y_all else np.array([])
            z_coords = np.concatenate(z_all) if z_all else np.array([])
            colors, showscale, colorbar = self._get_color_values_all_frames(n_frames)
        else:
            x_coords = self._extract_dimension(self.x_axis_dim, mc_frame, positions_all, alive)
            y_coords = self._extract_dimension(self.y_axis_dim, mc_frame, positions_all, alive)
            z_coords = self._extract_dimension(self.z_axis_dim, mc_frame, positions_all, alive)
            colors, showscale, colorbar = self._get_color_values(mc_frame, positions_all, alive)

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

        traces = []
        if self.show_delaunay:
            positions_mapped = np.column_stack([x_coords, y_coords, z_coords])
            line_trace = self._build_delaunay_trace_mapped(
                mc_frame, positions_all, alive, positions_mapped
            )
            if line_trace is not None:
                traces.append(line_trace)
        traces.append(scatter)
        if slice_mask is not None:
            highlight_alive = alive & slice_mask
            if highlight_alive.any():
                highlight_x = self._extract_dimension(
                    self.x_axis_dim, mc_frame, positions_all, highlight_alive
                )
                highlight_y = self._extract_dimension(
                    self.y_axis_dim, mc_frame, positions_all, highlight_alive
                )
                highlight_z = self._extract_dimension(
                    self.z_axis_dim, mc_frame, positions_all, highlight_alive
                )
                highlight_size = float(self.point_size) + 2.0
                highlight = go.Scatter3d(
                    x=highlight_x,
                    y=highlight_y,
                    z=highlight_z,
                    mode="markers",
                    marker={
                        "size": highlight_size,
                        "color": "rgba(242, 142, 43, 0.9)",
                        "opacity": min(1.0, float(self.point_alpha) + 0.2),
                    },
                    hoverinfo="skip",
                    name="Slice highlight",
                    showlegend=False,
                )
                traces.append(highlight)
        if use_all_frames:
            if self._use_mc_time_player():
                highlight_frame = int(np.clip(self.time_player.value, 0, self._x.shape[0] - 1))
                highlight_name = "MC frame highlight"
                highlight_color = "rgba(54, 162, 235, 0.6)"
            else:
                highlight_frame = int(
                    np.clip(self._resolve_mc_time_index(), 0, self._x.shape[0] - 1)
                )
                highlight_name = "MC slice highlight"
                highlight_color = "rgba(0, 166, 140, 0.55)"
            alive_h = self._get_alive_mask(highlight_frame)
            pos_h = self._x[highlight_frame]
            if alive_h.any():
                hx = self._extract_dimension(self.x_axis_dim, highlight_frame, pos_h, alive_h)
                hy = self._extract_dimension(self.y_axis_dim, highlight_frame, pos_h, alive_h)
                hz = self._extract_dimension(self.z_axis_dim, highlight_frame, pos_h, alive_h)
                mc_highlight = go.Scatter3d(
                    x=hx,
                    y=hy,
                    z=hz,
                    mode="markers",
                    marker={
                        "size": float(self.point_size) + 1.0,
                        "color": highlight_color,
                        "opacity": min(1.0, float(self.point_alpha) + 0.1),
                    },
                    hoverinfo="skip",
                    name=highlight_name,
                    showlegend=False,
                )
                traces.append(mc_highlight)

        fig = go.Figure(data=traces)

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

        if self.fix_axes:
            axis_ranges = self._get_axis_ranges(frame)
            fig.update_layout(scene=axis_ranges)

        if self._camera_state:
            fig.update_layout(scene={"camera": self._camera_state})

        return fig

    def _update_plot(self, frame: int):
        self.plot_pane.object = self._make_figure(frame)

    def _on_plot_relayout(self, event) -> None:
        data = event.new
        if not isinstance(data, dict):
            return
        camera = data.get("scene.camera")
        if isinstance(camera, dict):
            self._camera_state = camera
            return
        for key, value in data.items():
            if not isinstance(key, str) or not key.startswith("scene.camera"):
                continue
            if self._camera_state is None:
                self._camera_state = {}
            if key == "scene.camera":
                if isinstance(value, dict):
                    self._camera_state = value
                continue
            if not key.startswith("scene.camera."):
                continue
            parts = key.split(".")[2:]
            target = self._camera_state
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    def panel(self) -> pn.Column:
        """Return the Panel layout for the 3D convergence viewer."""
        controls_dim = pn.Param(
            self,
            parameters=[
                "x_axis_dim",
                "y_axis_dim",
                "z_axis_dim",
            ],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "x_axis_dim": {"type": pn.widgets.Select, "name": "X Axis"},
                "y_axis_dim": {"type": pn.widgets.Select, "name": "Y Axis"},
                "z_axis_dim": {"type": pn.widgets.Select, "name": "Z Axis"},
            },
        )

        controls_points = pn.Param(
            self,
            parameters=[
                "point_size",
                "point_alpha",
                "color_metric",
            ],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "color_metric": {"type": pn.widgets.Select, "name": "Color By"},
            },
        )

        controls_axis = pn.Param(
            self,
            parameters=[
                "fix_axes",
                "show_delaunay",
                "line_color",
            ],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "line_color": {"type": pn.widgets.ColorPicker, "name": "Edge color"},
            },
        )

        controls_line = pn.Param(
            self,
            parameters=[
                "line_style",
                "line_width",
                "line_alpha",
            ],
            sizing_mode="stretch_width",
            show_name=False,
        )

        edge_color_metric = pn.widgets.Select.from_param(
            self.param.line_color_metric,
            name="Edge color by",
            sizing_mode="stretch_width",
        )
        edge_colorscale = pn.widgets.Select.from_param(
            self.param.line_colorscale,
            name="Edge colorscale",
            sizing_mode="stretch_width",
        )
        controls_edge_colors = pn.Column(
            edge_color_metric,
            edge_colorscale,
            sizing_mode="stretch_width",
        )

        controls_row = pn.Row(
            controls_dim,
            controls_points,
            controls_axis,
            controls_line,
            controls_edge_colors,
            sizing_mode="stretch_width",
        )

        dimension_info = pn.pane.Alert(
            """
            **Dimension Mapping:** Map spatial dimensions (dim_0, dim_1, dim_2, dim_3), Euclidean time, or Monte Carlo time to plot axes.
            - **Spatial dims**: Position coordinates from simulation
            - **MC time**: Current frame index (useful for temporal evolution visualization)
            - **Player mode**: Choose Monte Carlo or Euclidean slices
            """,
            alert_type="info",
            sizing_mode="stretch_width",
        )
        euclidean_dim_note = pn.pane.Markdown(
            "**Euclidean time dimension:** dim_0",
            sizing_mode="stretch_width",
        )
        euclidean_dim_warning = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )
        time_toggle = pn.widgets.RadioButtonGroup(
            name="Iterate by",
            options={
                "Monte Carlo": "monte_carlo",
                "Euclidean": "euclidean",
            },
            value=self.time_iteration,
            sizing_mode="stretch_width",
        )
        time_toggle.param.watch(lambda e: setattr(self, "time_iteration", e.new), "value")
        self.param.watch(
            lambda e, widget_ref=time_toggle: (
                None if widget_ref.value == e.new else setattr(widget_ref, "value", e.new)
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
            parameters=[
                "euclidean_time_dim",
                "euclidean_time_bins",
            ],
            sizing_mode="stretch_width",
            show_name=False,
            widgets={
                "euclidean_time_dim": {
                    "type": pn.widgets.IntInput,
                    "name": "Euclidean time dim (0-indexed)",
                },
                "euclidean_time_bins": {
                    "type": pn.widgets.EditableIntSlider,
                    "name": "Time bins (Euclidean)",
                    "start": 5,
                    "end": 500,
                    "step": 1,
                },
            },
        )
        self._mc_slice_controls = mc_slice_controls
        self._time_bin_controls = time_bin_controls
        self._time_distribution_pane = pn.pane.HoloViews(
            self._build_time_distribution(),
            sizing_mode="stretch_width",
            height=180,
        )
        self._time_distribution_container = pn.Column(
            self._time_distribution_pane,
            sizing_mode="stretch_width",
        )
        self._player_mode_pane = pn.pane.Markdown(
            "**Player:** ready",
            sizing_mode="stretch_width",
        )
        self._sync_mc_slice_visibility()
        self._sync_time_bin_visibility()
        self._update_player_mode()

        def _update_euclidean_dim_note(*_):
            if self.history is None:
                euclidean_dim_note.object = "**Euclidean time dimension:** dim_0"
                euclidean_dim_warning.object = ""
                return
            d = self._data_d if self._data_d is not None else self.history.d
            dim_idx = self._resolve_euclidean_dim(d)
            euclidean_dim_note.object = f"**Euclidean time dimension:** dim_{dim_idx}"
            raw_dim = int(self.euclidean_time_dim)
            if raw_dim != dim_idx or d < 4:
                euclidean_dim_warning.object = (
                    f"\u26a0\ufe0f Euclidean time clamped to dim_{dim_idx} because d={d}."
                )
            else:
                euclidean_dim_warning.object = ""

        self.param.watch(_update_euclidean_dim_note, ["euclidean_time_dim"])
        _update_euclidean_dim_note()

        return pn.Column(
            pn.pane.Markdown("## 3D Swarm Convergence (Plotly)"),
            dimension_info,
            euclidean_dim_note,
            euclidean_dim_warning,
            time_toggle,
            mc_slice_controls,
            time_bin_controls,
            self._time_distribution_container,
            self._player_mode_pane,
            self.time_player,
            pn.Spacer(height=10),
            self.plot_pane,
            pn.layout.Divider(),
            controls_row,
            pn.layout.Divider(),
            self.status_pane,
            sizing_mode="stretch_width",
        )


class SimulationTab:
    """Encapsulates the Simulation tab: gas config, 3D visualizer, and RunHistory I/O."""

    def __init__(self, bounds_extent: float = 30.0, debug_fn: Callable | None = None):
        self._debug = debug_fn or (lambda *_args, **_kwargs: None)
        self._history_changed_callbacks: list[Callable] = []

        self._debug("building config + visualizer")
        self._gas_config = GasConfigPanel.create_qft_config(bounds_extent=bounds_extent)
        # Override with the best stable calibration settings found in QFT tuning.
        self._gas_config.n_steps = 750
        self._gas_config.gas_params["N"] = 500
        self._gas_config.gas_params["dtype"] = "float32"
        self._gas_config.gas_params["clone_every"] = 1
        self._gas_config.neighbor_weight_modes = [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ]
        self._gas_config.init_offset = 0.0
        self._gas_config.init_spread = 0.0
        self._gas_config.init_velocity_scale = 0.0

        # Kinetic operator (Langevin + viscous coupling).
        self._gas_config.kinetic_op.gamma = 1.0
        self._gas_config.kinetic_op.beta = 1.0
        self._gas_config.kinetic_op.auto_thermostat = True
        self._gas_config.kinetic_op.delta_t = 0.01
        self._gas_config.kinetic_op.sigma_v = 1.0
        self._gas_config.kinetic_op.nu = 1.0
        self._gas_config.kinetic_op.beta_curl = 1.0
        self._gas_config.kinetic_op.use_viscous_coupling = True
        self._gas_config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel_volume"
        self._gas_config.kinetic_op.viscous_length_scale = 1.0

        # Cloning operator.
        self._gas_config.cloning.p_max = 1.0
        self._gas_config.cloning.epsilon_clone = 1e-6
        self._gas_config.cloning.sigma_x = 0.01
        self._gas_config.cloning.alpha_restitution = 1.0

        # Fitness operator.
        self._gas_config.fitness_op.alpha = 1.0
        self._gas_config.fitness_op.beta = 1.0
        self._gas_config.fitness_op.eta = 0.0
        self._gas_config.fitness_op.sigma_min = 0.0
        self._gas_config.fitness_op.A = 2.0

        self._visualizer = SwarmConvergence3D(
            history=None, bounds_extent=self._gas_config.bounds_extent
        )

        # Internal state
        self._history: RunHistory | None = None
        self._history_path: Path | None = None

        # Default history path
        repo_root = Path(__file__).resolve().parents[4]
        qft_run_id = "qft_penalty_thr0p75_pen0p9_m354_ed2p80_nu1p10_N200_long"
        self._qft_history_path = (
            repo_root / "outputs" / "qft_calibrated" / f"{qft_run_id}_history.pt"
        )
        self._history_dir = self._qft_history_path.parent
        self._history_dir.mkdir(parents=True, exist_ok=True)

        # Build widgets
        self._debug("setting up history controls")
        self._history_path_input = pn.widgets.TextInput(
            name="QFT RunHistory path",
            value=str(self._qft_history_path),
            min_width=335,
            sizing_mode="stretch_width",
        )
        self._browse_button = pn.widgets.Button(
            name="Browse files...",
            button_type="default",
            min_width=335,
            sizing_mode="stretch_width",
        )
        self._file_selector_container = pn.Column(sizing_mode="stretch_width")
        self._load_button = pn.widgets.Button(
            name="Load RunHistory",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
        )
        self._save_button = pn.widgets.Button(
            name="Save RunHistory",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
            disabled=True,
        )
        self._load_status = pn.pane.Markdown(
            "**Load a history**: paste a *_history.pt path or browse and click Load.",
            sizing_mode="stretch_width",
        )
        self._save_status = pn.pane.Markdown(
            "**Save a history**: run a simulation or load a RunHistory first.",
            sizing_mode="stretch_width",
        )
        self._simulation_compute_status = pn.pane.Markdown(
            "**Simulation:** run or load a RunHistory, then click Compute Simulation.",
            sizing_mode="stretch_width",
        )
        self._simulation_compute_button = pn.widgets.Button(
            name="Compute Simulation",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )

        # Wire internal events
        self._browse_button.on_click(self._on_browse_clicked)
        self._load_button.on_click(self._on_load_clicked)
        self._save_button.on_click(self._on_save_clicked)
        self._simulation_compute_button.on_click(self._on_compute_simulation_clicked)
        self._gas_config.add_completion_callback(self._on_simulation_complete)
        self._gas_config.param.watch(self._on_bounds_change, "bounds_extent")

    # -- Public properties --

    @property
    def gas_config(self) -> GasConfigPanel:
        return self._gas_config

    @property
    def visualizer(self) -> SwarmConvergence3D:
        return self._visualizer

    @property
    def history(self) -> RunHistory | None:
        return self._history

    @property
    def history_path(self) -> Path | None:
        return self._history_path

    # -- Public methods --

    def set_history(
        self,
        history: RunHistory,
        history_path: Path | None = None,
        defer_dashboard_updates: bool = False,
    ) -> None:
        """Update internal state and simulation widgets, then notify listeners."""
        self._history = history
        self._history_path = history_path

        if not defer_dashboard_updates:
            self._visualizer.bounds_extent = float(self._gas_config.bounds_extent)
            self._visualizer.set_history(history)

        self._simulation_compute_button.disabled = False
        self._simulation_compute_status.object = (
            "**Simulation:** click Compute Simulation to visualize this RunHistory."
        )

        if defer_dashboard_updates:
            self._visualizer.status_pane.object = (
                "**Simulation complete:** history captured; click a Compute button to "
                "run post-processing."
            )
            self._save_button.disabled = False
            self._save_status.object = "**Save a history**: choose a path and click Save."
        else:
            self._save_button.disabled = False
            self._save_status.object = "**Save a history**: choose a path and click Save."

        # Notify registered listeners
        for cb in self._history_changed_callbacks:
            cb(history, history_path, defer_dashboard_updates)

    def on_history_changed(self, callback: Callable) -> None:
        """Register callback(history: RunHistory, path: Path|None, defer: bool)."""
        self._history_changed_callbacks.append(callback)

    def build_run_history_panel(self) -> pn.Column:
        """Return the RunHistory sidebar section (path input, browse, load, save)."""
        return pn.Column(
            self._history_path_input,
            self._browse_button,
            self._file_selector_container,
            self._load_button,
            self._load_status,
            self._save_button,
            self._save_status,
            sizing_mode="stretch_width",
        )

    def build_simulation_sidebar_panel(self) -> pn.Column:
        """Return gas_config.panel() for the sidebar Simulation accordion."""
        return self._gas_config.panel()

    def build_visualization_controls(self) -> pn.Param:
        """Return visualization controls Param panel."""
        return pn.Param(
            self._visualizer,
            parameters=["point_size", "point_alpha", "color_metric", "fix_axes"],
            show_name=False,
        )

    def build_tab(self) -> pn.Column:
        """Return the main Simulation tab panel."""
        return pn.Column(
            self._simulation_compute_status,
            pn.Row(self._simulation_compute_button, sizing_mode="stretch_width"),
            self._visualizer.panel(),
            sizing_mode="stretch_both",
        )

    # -- Internal callbacks --

    def _on_simulation_complete(self, history: RunHistory) -> None:
        self.set_history(history, defer_dashboard_updates=True)

    def _sync_history_path(self, value):
        if value:
            self._history_path_input.value = str(value[0])

    def _ensure_file_selector(self) -> pn.widgets.FileSelector:
        if self._file_selector_container.objects:
            return self._file_selector_container.objects[0]
        selector = pn.widgets.FileSelector(
            name="Select RunHistory",
            directory=str(self._history_dir),
            file_pattern="*_history.pt",
            only_files=True,
        )
        if self._qft_history_path.exists():
            selector.value = [str(self._qft_history_path)]
        selector.param.watch(lambda e: self._sync_history_path(e.new), "value")
        self._file_selector_container.objects = [selector]
        return selector

    def _on_browse_clicked(self, _):
        self._ensure_file_selector()

    def _on_load_clicked(self, _):
        history_path = Path(self._history_path_input.value).expanduser()
        if not history_path.exists():
            self._load_status.object = "**Error:** History path does not exist."
            return
        try:
            history = RunHistory.load(str(history_path))
            self.set_history(history, history_path, defer_dashboard_updates=True)
            self._load_status.object = f"**Loaded:** `{history_path}`"
        except Exception as exc:
            self._load_status.object = f"**Error loading history:** {exc!s}"

    def _on_compute_simulation_clicked(self, _):
        if self._gas_config.run_button.disabled:
            self._simulation_compute_status.object = (
                "**Simulation:** simulation is currently running.\n\n"
                "Wait for completion before recomputing visualization."
            )
            return

        history = self._history
        if history is None:
            self._simulation_compute_status.object = (
                "**Error:** run a simulation or load a RunHistory first."
            )
            return

        self._simulation_compute_status.object = "**Computing Simulation...**"
        try:
            self._visualizer.set_history(history)
            self._simulation_compute_status.object = (
                f"**Simulation ready:** {history.n_steps} steps / "
                f"{history.n_recorded} recorded frames."
            )
        except Exception as exc:
            self._simulation_compute_status.object = f"**Error:** {exc!s}"

    def _on_save_clicked(self, _):
        history = self._history
        if history is None:
            self._save_status.object = "**Error:** run a simulation or load a RunHistory first."
            return
        raw_path = self._history_path_input.value.strip()
        if raw_path:
            history_path = Path(raw_path).expanduser()
        else:
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            history_path = self._history_dir / f"qft_{stamp}_history.pt"
        if history_path.exists() and history_path.is_dir():
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            history_path = history_path / f"qft_{stamp}_history.pt"
        elif history_path.suffix != ".pt":
            history_path = history_path.with_suffix(".pt")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            history.save(str(history_path))
            self._history_path = history_path
            self._history_path_input.value = str(history_path)
            self._save_status.object = f"**Saved:** `{history_path}`"
        except Exception as exc:
            self._save_status.object = f"**Error saving history:** {exc!s}"

    def _on_bounds_change(self, event):
        self._visualizer.bounds_extent = float(event.new)
        self._visualizer._refresh_frame()
