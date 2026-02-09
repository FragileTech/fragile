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
    CorrelatorConfig,
    compute_channel_correlator,
)
from fragile.fractalai.qft.aggregation import compute_all_operator_series
from fragile.fractalai.qft.electroweak_channels import (
    ELECTROWEAK_CHANNELS,
    ElectroweakChannelConfig,
    compute_all_electroweak_channels,
    compute_electroweak_snapshot_operators,
)
from fragile.fractalai.qft.higgs_observables import (
    HiggsConfig,
    HiggsObservables,
    compute_higgs_observables,
)
from fragile.fractalai.qft.radial_channels import (
    RadialChannelBundle,
    RadialChannelConfig,
    compute_radial_channels,
)
from fragile.fractalai.qft.anisotropic_edge_channels import (
    AnisotropicEdgeChannelConfig,
    AnisotropicEdgeChannelOutput,
    compute_anisotropic_edge_channels,
)
from fragile.fractalai.qft.higgs_plotting import build_all_higgs_plots
from fragile.fractalai.qft.dirac_spectrum import (
    DiracSpectrumConfig,
    compute_dirac_spectrum,
    build_fermion_comparison,
    build_fermion_ratio_comparison,
)
from fragile.fractalai.qft.dirac_spectrum_plotting import build_all_dirac_plots
from fragile.fractalai.qft.einstein_equations import (
    EinsteinConfig,
    compute_einstein_test,
)
from fragile.fractalai.qft.einstein_equations_plotting import build_all_einstein_plots
from fragile.fractalai.qft.isospin_channels import (
    ISOSPIN_CHANNEL_SPLITTINGS,
    ISOSPIN_MASS_RATIOS,
    IsospinChannelResult,
    compute_isospin_channels,
)
from fragile.fractalai.qft.quantum_gravity import (
    QuantumGravityConfig,
    QuantumGravityObservables,
    QuantumGravityTimeSeries,
    compute_quantum_gravity_observables,
    compute_quantum_gravity_time_evolution,
)
from fragile.fractalai.qft.quantum_gravity_plotting import (
    build_all_gravity_plots,
    build_all_quantum_gravity_time_series_plots,
)
from fragile.fractalai.qft.plotting import (
    CHANNEL_COLORS,
    ChannelPlot,
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
    use_time_sliced_tessellation = param.Boolean(
        default=True,
        doc="Use time-sliced tessellation for Euclidean Delaunay edges",
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
        objects=["constant", "distance", "geodesic", "timelike/spacelike"],
        doc="Delaunay edge color metric",
    )
    line_colorscale = param.ObjectSelector(
        default="Viridis",
        objects=["Viridis", "Plasma", "Cividis", "Turbo", "Magma", "Inferno"],
        doc="Colorscale for distance-colored edges",
    )
    line_spacelike_color = param.Color(
        default="#4e79a7",
        doc="Spacelike edge color (time-sliced tessellation)",
    )
    line_timelike_color = param.Color(
        default="#e15759",
        doc="Timelike edge color (time-sliced tessellation)",
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
        self._player_mode_pane = None
        self._time_sliced_cache_key = None
        self._time_sliced_cache_edges = None
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
                "line_spacelike_color",
                "line_timelike_color",
                "euclidean_time_dim",
                "euclidean_time_bins",
                "use_time_sliced_tessellation",
                "mc_time_index",
            ],
        )
        self.param.watch(
            self._invalidate_time_sliced_cache,
            ["euclidean_time_dim", "euclidean_time_bins", "use_time_sliced_tessellation"],
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
        self._time_sliced_cache_key = None
        self._time_sliced_cache_edges = None
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
        color_options = dim_options + [
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
            self.status_pane.object += (
                f" | Neighbor graph method: {self._neighbor_graph_method}"
            )
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

    def _invalidate_time_sliced_cache(self, *_):
        self._time_sliced_cache_key = None
        self._time_sliced_cache_edges = None

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
        self._time_bin_controls.visible = self.time_iteration == "euclidean"
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
        return self.x_axis_dim == "mc_time" or self.y_axis_dim == "mc_time" or self.z_axis_dim == "mc_time"

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

    def _get_time_sliced_edges(self, frame: int, time_dim: int) -> dict[str, np.ndarray] | None:
        if self.history is None or self._x is None or not self.use_time_sliced_tessellation:
            return None
        if frame < 0 or frame >= self._x.shape[0]:
            return None
        key = (
            int(frame),
            int(time_dim),
            int(self.euclidean_time_bins),
            float(self.bounds_extent),
            bool(self.history.pbc),
        )
        if key == self._time_sliced_cache_key and self._time_sliced_cache_edges is not None:
            cached = self._time_sliced_cache_edges
            if cached.get("combined") is None or cached["combined"].size == 0:
                return None
            return cached

        try:
            from fragile.fractalai.qft.voronoi_time_slices import (
                compute_time_sliced_voronoi,
            )
        except Exception:
            return None

        positions = torch.as_tensor(self._x[frame])
        alive_mask = self._get_alive_mask(frame)
        alive = torch.as_tensor(alive_mask, dtype=torch.bool, device=positions.device)

        try:
            time_sliced = compute_time_sliced_voronoi(
                positions=positions,
                time_dim=int(time_dim),
                n_bins=int(self.euclidean_time_bins),
                min_walkers_bin=1,
                bounds=self.history.bounds,
                alive=alive,
                pbc=bool(self.history.pbc),
                pbc_mode="mirror",
                exclude_boundary=True,
                boundary_tolerance=1e-6,
                compute_curvature=False,
            )
        except Exception:
            self._time_sliced_cache_key = key
            self._time_sliced_cache_edges = {
                "spacelike": np.zeros((0, 2), dtype=np.int64),
                "timelike": np.zeros((0, 2), dtype=np.int64),
                "combined": np.zeros((0, 2), dtype=np.int64),
            }
            return None

        spacelike_edges: list[np.ndarray] = []
        for bin_result in time_sliced.bins:
            if bin_result.spacelike_edges is not None and bin_result.spacelike_edges.size:
                spacelike_edges.append(bin_result.spacelike_edges)
        if spacelike_edges:
            spacelike_array = np.vstack(spacelike_edges)
        else:
            spacelike_array = np.zeros((0, 2), dtype=np.int64)
        spacelike_array = self._normalize_edges(spacelike_array)

        timelike_array = self._normalize_edges(time_sliced.timelike_edges)

        combined_candidates = []
        if spacelike_array.size:
            combined_candidates.append(spacelike_array)
        if timelike_array.size:
            combined_candidates.append(timelike_array)
        if combined_candidates:
            combined = self._normalize_edges(np.vstack(combined_candidates))
        else:
            combined = np.zeros((0, 2), dtype=np.int64)

        edge_sets = {
            "spacelike": spacelike_array,
            "timelike": timelike_array,
            "combined": combined,
        }
        self._time_sliced_cache_key = key
        self._time_sliced_cache_edges = edge_sets
        return edge_sets if combined.size else None

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

        elif metric == "riemannian_volume":
            if frame == 0 or self._volume_weights is None:
                colors = "#1f77b4"
                return colors, False, None
            idx = min(frame - 1, len(self._volume_weights) - 1)
            colors = self._volume_weights[idx][alive]
            return colors, True, {"title": "Riemannian Volume"}

        elif metric == "radius":
            # Compute radius from original positions (first 3 dims)
            positions_filtered = positions_all[alive][:, : min(3, positions_all.shape[1])]
            colors = np.linalg.norm(positions_filtered, axis=1)
            return colors, True, {"title": "Radius"}

        else:  # "constant"
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
                d = self._data_d if self._data_d is not None else self.history.d
                dim_idx = self._resolve_euclidean_dim(d)
            return f"Euclidean Time (dim_{dim_idx})"
        if dim_spec == "riemannian_volume":
            return "Riemannian Volume"
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
        # Get edges from history
        use_geodesic = self.line_color_metric == "geodesic"
        edges = self._get_delaunay_edges(frame, normalize=not use_geodesic)
        edge_values = None
        if use_geodesic:
            edge_values = self._get_geodesic_edge_distances(frame)
            if edge_values is not None and edges is not None and edge_values.shape[0] != edges.shape[0]:
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
        """Build Delaunay edges from a provided edge list."""
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
            slice_dim_visible = self._axis_uses_dim(
                self.x_axis_dim, slice_dim, positions_all.shape[1]
            ) or self._axis_uses_dim(
                self.y_axis_dim, slice_dim, positions_all.shape[1]
            ) or self._axis_uses_dim(
                self.z_axis_dim, slice_dim, positions_all.shape[1]
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

        # Extract coordinates based on dimension mapping
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
            line_traces: list[go.Scatter3d] = []
            if (
                self.use_time_sliced_tessellation
                and self.time_iteration == "euclidean"
                and slice_dim_visible
                and slice_dim is not None
            ):
                edge_sets = self._get_time_sliced_edges(mc_frame, slice_dim)
                if edge_sets is not None:
                    if self.line_color_metric == "timelike/spacelike":
                        spacelike_trace = self._build_delaunay_trace_mapped_from_edges(
                            edge_sets["spacelike"],
                            positions_all,
                            alive,
                            positions_mapped,
                            color_override=self.line_spacelike_color,
                            trace_name="Spacelike edges",
                        )
                        timelike_trace = self._build_delaunay_trace_mapped_from_edges(
                            edge_sets["timelike"],
                            positions_all,
                            alive,
                            positions_mapped,
                            color_override=self.line_timelike_color,
                            trace_name="Timelike edges",
                        )
                        if spacelike_trace is not None:
                            line_traces.append(spacelike_trace)
                        if timelike_trace is not None:
                            line_traces.append(timelike_trace)
                    else:
                        line_trace = self._build_delaunay_trace_mapped_from_edges(
                            edge_sets["combined"],
                            positions_all,
                            alive,
                            positions_mapped,
                        )
                        if line_trace is not None:
                            line_traces.append(line_trace)
            if not line_traces:
                line_trace = self._build_delaunay_trace_mapped(
                    mc_frame, positions_all, alive, positions_mapped
                )
                if line_trace is not None:
                    line_traces.append(line_trace)
            traces.extend(line_traces)
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
                highlight_frame = int(np.clip(self._resolve_mc_time_index(), 0, self._x.shape[0] - 1))
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
        # Dimension mapping
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

        # Point size + color
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

        # Axis settings + Delaunay toggle + base color
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

        # Line styling
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

        # Edge color controls
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
        edge_spacelike = pn.widgets.ColorPicker.from_param(
            self.param.line_spacelike_color,
            name="Spacelike color",
            sizing_mode="stretch_width",
        )
        edge_timelike = pn.widgets.ColorPicker.from_param(
            self.param.line_timelike_color,
            name="Timelike color",
            sizing_mode="stretch_width",
        )
        controls_edge_colors = pn.Column(
            edge_color_metric,
            edge_colorscale,
            pn.Row(edge_spacelike, edge_timelike, sizing_mode="stretch_width"),
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
            parameters=[
                "euclidean_time_dim",
                "euclidean_time_bins",
                "use_time_sliced_tessellation",
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
                "use_time_sliced_tessellation": {
                    "type": pn.widgets.Checkbox,
                    "name": "Use sliced tessellation",
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
                    f"⚠️ Euclidean time clamped to dim_{dim_idx} because d={d}."
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
            self._time_distribution_pane,
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
            "--string-tension-max-triangles",
            str(self.string_tension_max_triangles),
            "--string-tension-bins",
            str(self.string_tension_bins),
        ]

        # Add boolean flags
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
        if self.compute_string_tension:
            args.append("--compute-string-tension")

        return args


class ChannelSettings(param.Parameterized):
    """Settings for the new Channels tab (independent of old particle analysis)."""

    simulation_range = param.Range(
        default=(0.1, 1.0), bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end). Trims both warmup and late-time frames.",
    )
    max_lag = param.Integer(default=80, bounds=(10, 200))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-6, None), allow_None=True)
    neighbor_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "recorded", "companions", "voronoi"),
        doc="Neighbor topology source ('voronoi' kept as legacy alias for 'auto').",
    )
    edge_weight_mode = param.ObjectSelector(
        default="uniform",
        objects=[
            "uniform",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "riemannian_kernel",
            "riemannian_kernel_volume",
            "inverse_distance",
            "inverse_volume",
            "kernel",
        ],
        doc="Edge weight mode for operator averaging (from pre-computed scutoid weights)",
    )
    use_connected = param.Boolean(default=True)

    # User-friendly time dimension selection
    time_dimension = param.ObjectSelector(
        default="monte_carlo",
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
    use_time_sliced_tessellation = param.Boolean(
        default=True,
        doc="Use time-sliced tessellation for Euclidean neighbor selection",
    )
    time_sliced_neighbor_mode = param.ObjectSelector(
        default="spacelike",
        objects=("spacelike", "spacelike+timelike", "timelike"),
        doc="Neighbor mode when using time-sliced tessellation",
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

    # Bootstrap error estimation
    compute_bootstrap_errors = param.Boolean(
        default=False,
        doc="Enable bootstrap resampling for correlator error estimation",
    )
    n_bootstrap = param.Integer(
        default=100,
        bounds=(10, 1000),
        doc="Number of bootstrap resamples for error estimation",
    )


class RadialSettings(param.Parameterized):
    """Settings for radial channel correlators (axis-free)."""

    time_axis = param.ObjectSelector(
        default="mc",
        objects=("mc", "radial"),
        doc=(
            "Analysis axis: 'mc' computes correlator decay across Monte Carlo time; "
            "'radial' uses a single snapshot binned by radial distance."
        ),
    )
    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc=(
            "For time_axis='radial': recorded index/step for the snapshot. "
            "For time_axis='mc': optional starting recorded index/step (None uses simulation_range)."
        ),
    )
    simulation_range = param.Range(
        default=(0.1, 1.0), bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end). Trims both warmup and late-time frames.",
    )
    max_lag = param.Integer(
        default=80,
        bounds=(10, 500),
        doc="Maximum MC lag used when time_axis='mc'.",
    )
    use_connected = param.Boolean(
        default=True,
        doc="Use connected correlators C(τ)=<OO>-<O>² when time_axis='mc'.",
    )
    n_bins = param.Integer(default=48, bounds=(10, 200))
    max_pairs = param.Integer(default=200_000, bounds=(10_000, 2_000_000))
    distance_mode = param.ObjectSelector(
        default="graph_full",
        objects=("euclidean", "graph_iso", "graph_full"),
    )
    neighbor_method = param.ObjectSelector(
        default="recorded",
        objects=("recorded",),
        doc="Reuse simulation-recorded Delaunay neighbors (no recomputation).",
    )
    neighbor_weighting = param.ObjectSelector(
        default="inv_geodesic_full",
        objects=(
            "volume",
            "euclidean",
            "inv_euclidean",
            "inv_geodesic_iso",
            "inv_geodesic_full",
            "kernel",
            "uniform",
            "inverse_distance",
            "inverse_volume",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "riemannian_kernel",
            "riemannian_kernel_volume",
        ),
        doc=(
            "Neighbor weighting mode. Supports recorded scutoid edge weights "
            "(e.g. inverse_riemannian_distance, riemannian_kernel, "
            "riemannian_kernel_volume)."
        ),
    )
    neighbor_k = param.Integer(
        default=0,
        bounds=(0, 50),
        doc="Maximum neighbors per walker (0 = use all).",
    )
    kernel_length_scale = param.Number(
        default=1.0,
        bounds=(1e-6, None),
        doc="Length scale for Gaussian kernel weighting: exp(-d²/(2l²))",
    )
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-6, None), allow_None=True)
    use_volume_weights = param.Boolean(default=True)
    apply_power_correction = param.Boolean(default=True)
    power_override = param.Number(default=None, allow_None=True)
    window_widths_spec = param.String(default="5-50")
    channel_list = param.String(
        default="scalar,pseudoscalar,vector,axial_vector,tensor,nucleon,glueball"
    )
    drop_axis_average = param.Boolean(default=True)

    # Bootstrap error estimation
    compute_bootstrap_errors = param.Boolean(
        default=False,
        doc="Enable bootstrap resampling for correlator error estimation",
    )
    n_bootstrap = param.Integer(
        default=100,
        bounds=(10, 1000),
        doc="Number of bootstrap resamples for error estimation",
    )


class AnisotropicEdgeSettings(param.Parameterized):
    """Settings for anisotropic edge-channel MC-time correlators."""

    simulation_range = param.Range(
        default=(0.1, 1.0), bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end).",
    )
    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Optional starting recorded index/step (None uses simulation_range start).",
    )
    max_lag = param.Integer(default=80, bounds=(10, 500))
    use_connected = param.Boolean(default=True)
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-6, None), allow_None=True)
    spatial_dims_spec = param.String(
        default="",
        doc="Comma-separated spatial dims to use (blank = all). Example: '0,1,2'.",
    )
    edge_weight_mode = param.ObjectSelector(
        default="riemannian_kernel_volume",
        objects=(
            "uniform",
            "inv_geodesic_iso",
            "inv_geodesic_full",
            "inverse_distance",
            "inverse_volume",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "kernel",
            "riemannian_kernel",
            "riemannian_kernel_volume",
        ),
        doc="Recorded edge weighting mode from simulation scutoid pipeline.",
    )
    use_volume_weights = param.Boolean(
        default=True,
        doc="Multiply edge weights by sqrt(V_i V_j) from recorded Riemannian volumes.",
    )
    component_mode = param.ObjectSelector(
        default="isotropic+axes",
        objects=("isotropic", "axes", "isotropic+axes", "quadrupole", "isotropic+quadrupole"),
        doc="Directional basis used to build anisotropic edge moments.",
    )
    nucleon_triplet_mode = param.ObjectSelector(
        default="direct_neighbors",
        objects=("direct_neighbors", "companions"),
        doc=(
            "Nucleon triplet construction: "
            "'direct_neighbors' uses Delaunay-neighbor triplets, "
            "'companions' uses (distance companion, clone companion)."
        ),
    )
    channel_list = param.String(
        default="scalar,pseudoscalar,vector,axial_vector,tensor,nucleon,glueball"
    )
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))
    compute_bootstrap_errors = param.Boolean(default=False)
    n_bootstrap = param.Integer(default=100, bounds=(10, 1000))


class RadialElectroweakSettings(param.Parameterized):
    """Settings for radial electroweak channel correlators (axis-free)."""

    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Recorded Monte Carlo index or step to use as the 4D slice (None = last).",
    )
    n_bins = param.Integer(default=48, bounds=(10, 200))
    max_pairs = param.Integer(default=200_000, bounds=(10_000, 2_000_000))
    distance_mode = param.ObjectSelector(
        default="graph_full",
        objects=("euclidean", "graph_iso", "graph_full"),
    )
    neighbor_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "recorded", "companions", "voronoi"),
        doc="Neighbor topology source ('voronoi' kept as legacy; 'auto' tries recorded → companions).",
    )
    neighbor_weighting = param.ObjectSelector(
        default="inv_geodesic_full",
        objects=(
            "volume",
            "euclidean",
            "inv_euclidean",
            "inv_geodesic_iso",
            "inv_geodesic_full",
            "kernel",
        ),
        doc="Neighbor weighting for legacy Voronoi fallback path",
    )
    edge_weight_mode = param.ObjectSelector(
        default="inverse_riemannian_distance",
        objects=[
            "uniform",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "riemannian_kernel",
            "riemannian_kernel_volume",
            "inverse_distance",
            "inverse_volume",
            "kernel",
        ],
        doc="Edge weight mode for neighbor weighting (from pre-computed scutoid weights)",
    )
    neighbor_k = param.Integer(
        default=0,
        bounds=(0, 50),
        doc="Maximum neighbors per walker (0 = use all).",
    )
    kernel_length_scale = param.Number(
        default=1.0,
        bounds=(1e-6, None),
        doc="Length scale for Gaussian kernel weighting: exp(-d²/(2l²))",
    )
    use_volume_weights = param.Boolean(default=True)
    apply_power_correction = param.Boolean(default=True)
    power_override = param.Number(default=None, allow_None=True)
    window_widths_spec = param.String(default="5-50")

    channel_list = param.String(default=",".join(ELECTROWEAK_CHANNELS))
    drop_axis_average = param.Boolean(default=True)
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    epsilon_d = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    epsilon_c = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    epsilon_clone = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    lambda_alg = param.Number(default=None, bounds=(0.0, None), allow_None=True)

    # Bootstrap error estimation
    compute_bootstrap_errors = param.Boolean(
        default=False,
        doc="Enable bootstrap resampling for correlator error estimation",
    )
    n_bootstrap = param.Integer(
        default=100,
        bounds=(10, 1000),
        doc="Number of bootstrap resamples for error estimation",
    )


class ElectroweakSettings(param.Parameterized):
    """Settings for electroweak (U1/SU2) channel correlators."""

    simulation_range = param.Range(
        default=(0.1, 1.0), bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end). Trims both warmup and late-time frames.",
    )
    max_lag = param.Integer(default=80, bounds=(10, 200))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    use_connected = param.Boolean(default=True)
    neighbor_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "recorded", "companions", "voronoi"),
        doc="Neighbor topology source ('voronoi' kept as legacy; 'auto' tries recorded → companions).",
    )
    neighbor_weighting = param.ObjectSelector(
        default="inv_geodesic_full",
        objects=(
            "volume",
            "euclidean",
            "inv_euclidean",
            "inv_geodesic_iso",
            "inv_geodesic_full",
            "kernel",
        ),
        doc="Neighbor weighting for legacy Voronoi fallback path",
    )
    edge_weight_mode = param.ObjectSelector(
        default="inverse_riemannian_distance",
        objects=[
            "uniform",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "riemannian_kernel",
            "riemannian_kernel_volume",
            "inverse_distance",
            "inverse_volume",
            "kernel",
        ],
        doc="Edge weight mode for neighbor weighting (from pre-computed scutoid weights)",
    )
    neighbor_k = param.Integer(
        default=0,
        bounds=(0, 50),
        doc="Maximum neighbors per walker (0 = use all).",
    )
    kernel_length_scale = param.Number(
        default=1.0,
        bounds=(1e-6, None),
        doc="Length scale for Gaussian kernel weighting: exp(-d²/(2l²))",
    )

    # User-friendly time dimension selection
    time_dimension = param.ObjectSelector(
        default="monte_carlo",
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
    use_time_sliced_tessellation = param.Boolean(
        default=True,
        doc="Use time-sliced tessellation for Euclidean neighbor selection",
    )
    time_sliced_neighbor_mode = param.ObjectSelector(
        default="spacelike",
        objects=("spacelike", "spacelike+timelike", "timelike"),
        doc="Neighbor mode when using time-sliced tessellation",
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

    # Bootstrap error estimation
    compute_bootstrap_errors = param.Boolean(
        default=False,
        doc="Enable bootstrap resampling for correlator error estimation",
    )
    n_bootstrap = param.Integer(
        default=100,
        bounds=(10, 1000),
        doc="Number of bootstrap resamples for error estimation",
    )


class HiggsSettings(param.Parameterized):
    """Settings for Higgs field observable computation."""

    # Scalar field source
    scalar_field_source = param.ObjectSelector(
        default="fitness",
        objects=["fitness", "reward", "radius"],
        doc="Which field to use as the Higgs scalar field φ",
    )

    # Time slice selection
    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Recorded Monte Carlo index or step (None = last slice)",
    )

    # Physical parameters
    h_eff = param.Number(default=1.0, bounds=(1e-6, None), doc="Effective Planck constant ℏ")
    mu_sq = param.Number(default=1.0, doc="Higgs potential parameter μ² (can be negative)")
    lambda_higgs = param.Number(
        default=0.5, bounds=(1e-6, None), doc="Higgs potential parameter λ"
    )
    alpha_gravity = param.Number(default=0.1, bounds=(0.0, None), doc="Gravity coupling α")

    # Voronoi parameters (reused from existing implementation)
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.5))
    compute_curvature = param.Boolean(default=True, doc="Compute Ricci scalar curvature")
    compute_action = param.Boolean(default=True, doc="Compute Higgs action")

    # Metric tensor visualization
    metric_component_x = param.Integer(default=0, bounds=(0, 10), doc="Metric tensor row index")
    metric_component_y = param.Integer(default=0, bounds=(0, 10), doc="Metric tensor col index")


class QuantumGravitySettings(param.Parameterized):
    """Settings for quantum gravity analysis."""

    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Recorded Monte Carlo index or step (None = last slice)",
    )
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.5))
    analysis_dim_0 = param.Integer(
        default=0,
        bounds=(0, 10),
        doc="First dimension index used for quantum gravity analysis",
    )
    analysis_dim_1 = param.Integer(
        default=1,
        bounds=(0, 10),
        doc="Second dimension index used for quantum gravity analysis",
    )
    analysis_dim_2 = param.Integer(
        default=2,
        bounds=(0, 10),
        doc="Third dimension index used for quantum gravity analysis",
    )

    # Regge calculus
    use_metric_correction = param.ObjectSelector(
        default="full",
        objects=["none", "diagonal", "full"],
        doc="Metric correction mode for deficit angles",
    )

    # Spectral dimension
    diffusion_time_steps = param.Integer(
        default=100, bounds=(10, 500), doc="Number of diffusion time steps"
    )
    max_diffusion_time = param.Number(
        default=10.0, bounds=(0.1, 100.0), doc="Maximum diffusion time"
    )

    # Hausdorff dimension
    n_radial_bins = param.Integer(default=50, bounds=(10, 200), doc="Number of radial bins")

    # Causal structure
    light_speed = param.Number(default=1.0, bounds=(0.1, 10.0), doc="Speed of light (c)")
    euclidean_time_dim = param.Integer(
        default=3,
        bounds=(0, 10),
        doc="Spatial dimension index for Euclidean time (0-indexed)",
    )
    euclidean_time_bins = param.Integer(
        default=50, bounds=(10, 500), doc="Number of time bins for causal structure"
    )

    # Holographic entropy
    planck_length = param.Number(default=1.0, bounds=(1e-6, 10.0), doc="Planck length")

    # Time evolution (4D spacetime block analysis)
    compute_time_evolution = param.Boolean(
        default=True,
        doc="Compute observables over all MC frames (slower but shows time evolution)",
    )
    frame_stride = param.Integer(
        default=1,
        bounds=(1, 100),
        doc="Compute every N frames (for efficiency)",
    )


class FractalSetSettings(param.Parameterized):
    """Settings for Fractal Set IG/CST area-law measurements."""

    warmup_fraction = param.Number(
        default=0.1,
        bounds=(0.0, 0.95),
        doc="Fraction of early recorded transitions to discard before measuring.",
    )
    frame_stride = param.Integer(
        default=1,
        bounds=(1, 100),
        doc="Analyze every Nth recorded transition after warmup.",
    )
    max_frames = param.Integer(
        default=120,
        bounds=(1, None),
        doc="Maximum number of transitions to analyze (evenly subsampled).",
    )
    n_cut_samples = param.Integer(
        default=50,
        bounds=(5, 200),
        doc="Number of boundary samples for hyperplane/spherical sweeps.",
    )
    partition_family = param.ObjectSelector(
        default="all",
        objects=["all", "spatial", "graph", "random"],
        doc="Partition generator family.",
    )
    partition_axis = param.Integer(
        default=0,
        bounds=(0, 10),
        doc="Coordinate axis used for hyperplane/median cuts.",
    )
    cut_geometry = param.ObjectSelector(
        default="all",
        objects=["all", "hyperplane", "spherical", "median"],
        doc="Boundary geometry to evaluate.",
    )
    graph_cut_source = param.ObjectSelector(
        default="all",
        objects=["all", "distance", "fitness", "both"],
        doc="Companion graph used for spectral graph cuts.",
    )
    min_partition_size = param.Integer(
        default=5,
        bounds=(1, None),
        doc="Minimum walkers per side for graph/random cuts.",
    )
    random_partitions = param.Integer(
        default=50,
        bounds=(1, 500),
        doc="Number of random baseline partitions per analyzed transition.",
    )
    random_balanced = param.Boolean(
        default=True,
        doc="Use balanced random partitions (|A| ~= N/2).",
    )
    random_seed = param.Integer(
        default=12345,
        bounds=(0, None),
        doc="Seed for random baseline partition sampling.",
    )
    use_geometry_correction = param.Boolean(
        default=True,
        doc="Apply Riemannian distance and volume corrections to IG/CST measures.",
    )
    metric_display = param.ObjectSelector(
        default="both",
        objects=["raw", "geometry", "both"],
        doc="Which metric family to show in regressions and plots.",
    )
    geometry_kernel_length_scale = param.Number(
        default=None,
        bounds=(1e-6, None),
        allow_None=True,
        doc="Length scale for geometric kernel exp(-d_g^2/(2 l^2)); None uses history/default.",
    )
    geometry_min_eig = param.Number(
        default=1e-6,
        bounds=(0.0, None),
        doc="Minimum eigenvalue clamp for local metric tensors.",
    )
    geometry_use_volume = param.Boolean(
        default=True,
        doc="Multiply edge kernels by destination Riemannian volume weights.",
    )
    geometry_correct_area = param.Boolean(
        default=True,
        doc="Use volume-weighted lineage crossing area instead of pure lineage counts.",
    )


class EinsteinTestSettings(param.Parameterized):
    """Settings for Einstein equation verification test."""

    mc_time_index = param.Integer(
        default=None,
        allow_None=True,
        bounds=(0, None),
        doc="MC frame to analyze (None=last)",
    )
    regularization = param.Number(
        default=1e-6,
        bounds=(1e-12, 1.0),
        doc="Min eigenvalue for metric positivity",
    )
    stress_energy_mode = param.ObjectSelector(
        default="full",
        objects=["fitness_only", "kinetic_pressure", "full"],
        doc="T_uv components to include",
    )
    bulk_fraction = param.Number(
        default=0.8,
        bounds=(0.01, 0.99),
        doc="Fraction of walkers considered bulk",
    )
    g_newton_metric = param.ObjectSelector(
        default="s_total_geom",
        objects=["s_total_geom", "s_total", "s_dist_geom", "s_dist", "manual"],
        doc="Area-law metric for G_N extraction",
    )
    g_newton_manual = param.Number(
        default=1.0,
        bounds=(1e-6, None),
        doc="Manual G_N (when g_newton_metric='manual')",
    )


FRACTAL_SET_CUT_TYPES = ("hyperplane", "spherical", "median")
FRACTAL_SET_GRAPH_CUT_TYPES = ("spectral_distance", "spectral_fitness", "spectral_both")
FRACTAL_SET_RANDOM_CUT_TYPE = "random_baseline"
FRACTAL_SET_METRICS_RAW = {
    "s_dist": "S_dist",
    "s_fit": "S_fit",
    "s_total": "S_total",
}
FRACTAL_SET_METRICS_GEOM = {
    "s_dist_geom": "S_dist_geom",
    "s_fit_geom": "S_fit_geom",
    "s_total_geom": "S_total_geom",
}


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


def _parse_dims_spec(spec: str, history_d: int) -> list[int] | None:
    """Parse a comma-separated dimension list. Blank means all dimensions."""
    spec_clean = str(spec).strip()
    if not spec_clean:
        return None
    try:
        dims = sorted({int(item.strip()) for item in spec_clean.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"Invalid spatial_dims_spec: {spec!r}. Expected comma-separated integers.") from exc
    if not dims:
        return None
    invalid = [dim for dim in dims if dim < 0 or dim >= history_d]
    if invalid:
        raise ValueError(
            f"spatial_dims_spec contains invalid dims {invalid}; valid range is [0, {history_d - 1}]."
        )
    return dims


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

    neighbor_method = "auto" if settings.neighbor_method == "voronoi" else settings.neighbor_method

    channel_config = ChannelConfig(
        warmup_fraction=settings.simulation_range[0],
        end_fraction=settings.simulation_range[1],
        h_eff=settings.h_eff,
        mass=settings.mass,
        ell0=settings.ell0,
        neighbor_method=neighbor_method,
        edge_weight_mode=settings.edge_weight_mode,
        mc_time_index=settings.mc_time_index,
        time_axis=time_axis,
        euclidean_time_dim=euclidean_time_dim,
        euclidean_time_bins=settings.euclidean_time_bins,
    )
    correlator_config = CorrelatorConfig(
        max_lag=settings.max_lag,
        use_connected=settings.use_connected,
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=settings.fit_mode,
        fit_start=settings.fit_start,
        fit_stop=settings.fit_stop,
        min_fit_points=settings.min_fit_points,
        compute_bootstrap_errors=settings.compute_bootstrap_errors,
        n_bootstrap=settings.n_bootstrap,
    )
    channels = [c.strip() for c in settings.channel_list.split(",") if c.strip()]

    per_channel = {
        "scalar": settings.scalar_fit_mode,
        "pseudoscalar": settings.pseudoscalar_fit_mode,
        "vector": settings.vector_fit_mode,
        "nucleon": settings.nucleon_fit_mode,
        "glueball": settings.glueball_fit_mode,
    }

    # Determine spatial dimensions (for filtering baryon channels in 2D mode).
    # Only subtract 1 for Euclidean time axis (a spatial dim is consumed as time).
    # For MC time, all d dimensions are spatial.
    if time_axis == "euclidean":
        spatial_dims = history.d - 1 if history.d >= 3 else history.d
    else:
        spatial_dims = history.d

    # Mirror compute_all_channels() filtering in lower-dimensional settings.
    if spatial_dims < 3:
        channels = [ch for ch in channels if ch != "nucleon"]

    # Compute operator series once, then run per-channel correlator analysis.
    operator_series = compute_all_operator_series(history, channel_config, channels=channels)

    results: dict[str, ChannelCorrelatorResult] = {}
    for channel in channels:
        if channel not in operator_series.operators:
            continue
        override = per_channel.get(channel, "default")
        if override and override != "default":
            config = replace(correlator_config, fit_mode=str(override))
        else:
            config = correlator_config
        results[channel] = compute_channel_correlator(
            series=operator_series.operators[channel],
            dt=operator_series.dt,
            config=config,
            channel_name=channel,
        )
    return results


def _compute_anisotropic_edge_bundle(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> AnisotropicEdgeChannelOutput:
    """Compute anisotropic direct-edge correlators from recorded geometry."""
    config = AnisotropicEdgeChannelConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=int(settings.max_lag),
        use_connected=bool(settings.use_connected),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        keep_dims=_parse_dims_spec(settings.spatial_dims_spec, history.d),
        edge_weight_mode=str(settings.edge_weight_mode),
        use_volume_weights=bool(settings.use_volume_weights),
        component_mode=str(settings.component_mode),
        nucleon_triplet_mode=str(settings.nucleon_triplet_mode),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    channels = [c.strip() for c in settings.channel_list.split(",") if c.strip()]
    return compute_anisotropic_edge_channels(history, config=config, channels=channels)


def _compute_radial_channels_bundle(
    history: RunHistory,
    settings: RadialSettings,
) -> RadialChannelBundle:
    """Compute geometry-aware strong-force channels (MC-time or radial mode)."""
    config = RadialChannelConfig(
        time_axis=str(settings.time_axis),
        mc_time_index=settings.mc_time_index,
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        max_lag=int(settings.max_lag),
        use_connected=bool(settings.use_connected),
        n_bins=int(settings.n_bins),
        max_pairs=int(settings.max_pairs),
        distance_mode=str(settings.distance_mode),
        neighbor_method=str(settings.neighbor_method),
        neighbor_weighting=str(settings.neighbor_weighting),
        neighbor_k=int(settings.neighbor_k),
        kernel_length_scale=float(settings.kernel_length_scale),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        use_volume_weights=bool(settings.use_volume_weights),
        apply_power_correction=bool(settings.apply_power_correction),
        power_override=settings.power_override,
        window_widths=_parse_window_widths(settings.window_widths_spec),
        drop_axis_average=bool(settings.drop_axis_average),
        compute_bootstrap_errors=settings.compute_bootstrap_errors,
        n_bootstrap=settings.n_bootstrap,
    )
    channels = [c.strip() for c in settings.channel_list.split(",") if c.strip()]
    return compute_radial_channels(history, config=config, channels=channels)


def _compute_radial_electroweak_bundle(
    history: RunHistory,
    settings: RadialElectroweakSettings,
) -> RadialChannelBundle:
    config = RadialChannelConfig(
        time_axis="radial",
        mc_time_index=settings.mc_time_index,
        n_bins=int(settings.n_bins),
        max_pairs=int(settings.max_pairs),
        distance_mode=str(settings.distance_mode),
        neighbor_method="recorded",
        neighbor_weighting=str(settings.neighbor_weighting),
        neighbor_k=int(settings.neighbor_k),
        kernel_length_scale=float(settings.kernel_length_scale),
        h_eff=float(settings.h_eff),
        mass=1.0,
        ell0=None,
        use_volume_weights=bool(settings.use_volume_weights),
        apply_power_correction=bool(settings.apply_power_correction),
        power_override=settings.power_override,
        window_widths=_parse_window_widths(settings.window_widths_spec),
        drop_axis_average=bool(settings.drop_axis_average),
        compute_bootstrap_errors=settings.compute_bootstrap_errors,
        n_bootstrap=settings.n_bootstrap,
    )
    channels = [c.strip() for c in settings.channel_list.split(",") if c.strip()]

    ew_config = ElectroweakChannelConfig(
        h_eff=settings.h_eff,
        neighbor_method=settings.neighbor_method,
        neighbor_weighting=str(settings.neighbor_weighting),
        edge_weight_mode=getattr(settings, "edge_weight_mode", "inverse_riemannian_distance"),
        neighbor_k=settings.neighbor_k,
        kernel_length_scale=float(settings.kernel_length_scale),
        epsilon_d=settings.epsilon_d,
        epsilon_c=settings.epsilon_c,
        epsilon_clone=settings.epsilon_clone,
        lambda_alg=settings.lambda_alg,
        mc_time_index=settings.mc_time_index,
    )
    operators = compute_electroweak_snapshot_operators(
        history, config=ew_config, channels=channels
    )
    return compute_radial_channels(
        history, config=config, channels=channels, operators_override=operators
    )


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
        warmup_fraction=settings.simulation_range[0],
        end_fraction=settings.simulation_range[1],
        max_lag=settings.max_lag,
        h_eff=settings.h_eff,
        use_connected=settings.use_connected,
        neighbor_method=settings.neighbor_method,
        neighbor_weighting=str(settings.neighbor_weighting),
        edge_weight_mode=settings.edge_weight_mode,
        neighbor_k=settings.neighbor_k,
        kernel_length_scale=float(settings.kernel_length_scale),
        mc_time_index=settings.mc_time_index,
        time_axis=time_axis,
        euclidean_time_dim=euclidean_time_dim,
        euclidean_time_bins=settings.euclidean_time_bins,
        use_time_sliced_tessellation=settings.use_time_sliced_tessellation,
        time_sliced_neighbor_mode=settings.time_sliced_neighbor_mode,
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=settings.fit_mode,
        fit_start=settings.fit_start,
        fit_stop=settings.fit_stop,
        min_fit_points=settings.min_fit_points,
        epsilon_d=settings.epsilon_d,
        epsilon_c=settings.epsilon_c,
        epsilon_clone=settings.epsilon_clone,
        lambda_alg=settings.lambda_alg,
        compute_bootstrap_errors=settings.compute_bootstrap_errors,
        n_bootstrap=settings.n_bootstrap,
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


def _to_numpy(t):
    """Convert a tensor or array-like to a numpy array."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


def _select_fractal_set_frames(
    n_transitions: int,
    warmup_fraction: float,
    frame_stride: int,
    max_frames: int | None,
) -> list[int]:
    """Select transition indices after warmup with optional subsampling."""
    if n_transitions <= 0:
        return []

    warmup = float(np.clip(warmup_fraction, 0.0, 0.95))
    start_idx = int(np.floor(warmup * n_transitions))
    start_idx = int(np.clip(start_idx, 0, max(n_transitions - 1, 0)))
    stride = max(1, int(frame_stride))
    frame_ids = list(range(start_idx, n_transitions, stride))
    if not frame_ids:
        return []

    if max_frames is not None and max_frames > 0 and len(frame_ids) > max_frames:
        pick = np.linspace(0, len(frame_ids) - 1, num=int(max_frames), dtype=int)
        frame_ids = [frame_ids[int(i)] for i in pick]
    return frame_ids


def _build_lineage_by_transition(
    companions_fit: np.ndarray,
    will_clone: np.ndarray,
    n_walkers: int,
) -> np.ndarray:
    """Build lineage labels per transition index from clone parent mapping."""
    n_transitions = int(min(companions_fit.shape[0], will_clone.shape[0]))
    lineage = np.zeros((n_transitions, n_walkers), dtype=np.int64)
    current = np.arange(n_walkers, dtype=np.int64)

    for info_idx in range(n_transitions):
        lineage[info_idx] = current
        parent = np.arange(n_walkers, dtype=np.int64)
        clone_mask = np.asarray(will_clone[info_idx], dtype=bool)
        clone_src = np.asarray(companions_fit[info_idx], dtype=np.int64)
        clone_src = np.clip(clone_src, 0, n_walkers - 1)
        parent[clone_mask] = clone_src[clone_mask]
        current = current[parent]

    return lineage


def _build_region_masks(
    positions: np.ndarray,
    axis: int,
    cut_type: str,
    n_cut_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build binary masks A for a given boundary geometry."""
    n_walkers = positions.shape[0]
    if n_walkers <= 1:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    safe_axis = int(np.clip(axis, 0, max(positions.shape[1] - 1, 0)))
    coord = positions[:, safe_axis]
    sample_count = int(max(2, n_cut_samples))

    if cut_type == "hyperplane":
        low, high = float(coord.min()), float(coord.max())
        if np.isclose(low, high):
            cuts = np.array([low], dtype=float)
        else:
            cuts = np.linspace(low, high, num=sample_count, dtype=float)
        masks = coord[None, :] > cuts[:, None]
    elif cut_type == "spherical":
        center = positions.mean(axis=0, keepdims=True)
        radii = np.linalg.norm(positions - center, axis=1)
        max_radius = float(radii.max())
        if np.isclose(max_radius, 0.0):
            cuts = np.array([0.0], dtype=float)
        else:
            cuts = np.linspace(0.0, max_radius, num=sample_count, dtype=float)
        masks = radii[None, :] <= cuts[:, None]
    elif cut_type == "median":
        order = np.argsort(coord, kind="mergesort")
        mask = np.zeros(n_walkers, dtype=bool)
        mask[order[n_walkers // 2 :]] = True
        cuts = np.array([float(np.median(coord))], dtype=float)
        masks = mask[None, :]
    else:
        msg = f"Unsupported cut type: {cut_type}"
        raise ValueError(msg)

    valid = np.logical_and(masks.any(axis=1), (~masks).any(axis=1))
    return masks[valid], cuts[valid]


def _build_spectral_sweep_masks(
    companions: np.ndarray,
    n_cut_samples: int,
    min_partition_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build graph-theoretic cuts using a Fiedler-vector sweep."""
    companions_idx = np.asarray(companions, dtype=np.int64)
    if companions_idx.ndim == 1:
        companions_idx = companions_idx[None, :]
    if companions_idx.ndim != 2:
        return np.zeros((0, 0), dtype=bool), np.zeros(0, dtype=float)

    n_walkers = int(companions_idx.shape[1])
    if n_walkers <= 2:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    companions_idx = np.clip(companions_idx, 0, n_walkers - 1)
    adjacency = np.zeros((n_walkers, n_walkers), dtype=float)
    walker = np.arange(n_walkers, dtype=np.int64)
    for row in companions_idx:
        adjacency[walker, row] += 1.0
        adjacency[row, walker] += 1.0
    np.fill_diagonal(adjacency, 0.0)

    degree = adjacency.sum(axis=1)
    if np.allclose(degree, 0.0):
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    # Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    inv_sqrt_degree = np.zeros_like(degree)
    positive = degree > 1e-12
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(n_walkers, dtype=float) - (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )

    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        if eigvecs.shape[1] < 2:
            return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)
        # First eigenvector is near-constant; second gives Fiedler direction.
        fiedler = eigvecs[:, 1]
        if not np.isfinite(eigvals[1]):
            return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)
    except np.linalg.LinAlgError:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    order = np.argsort(fiedler, kind="mergesort")
    min_size = int(np.clip(min_partition_size, 1, max(1, n_walkers // 2)))
    k_values = np.linspace(
        min_size,
        n_walkers - min_size,
        num=max(2, int(n_cut_samples)),
        dtype=int,
    )
    k_values = np.unique(k_values)
    k_values = k_values[(k_values > 0) & (k_values < n_walkers)]
    if k_values.size == 0:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    masks = np.zeros((k_values.size, n_walkers), dtype=bool)
    for idx, k in enumerate(k_values.tolist()):
        masks[idx, order[: int(k)]] = True

    cuts = k_values.astype(float) / float(n_walkers)
    valid = np.logical_and(masks.any(axis=1), (~masks).any(axis=1))
    return masks[valid], cuts[valid]


def _build_random_partition_masks(
    n_walkers: int,
    n_partitions: int,
    min_partition_size: int,
    balanced: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build random partition baseline masks."""
    if n_walkers <= 2 or n_partitions <= 0:
        return np.zeros((0, n_walkers), dtype=bool), np.zeros(0, dtype=float)

    min_size = int(np.clip(min_partition_size, 1, max(1, n_walkers // 2)))
    masks = np.zeros((int(n_partitions), n_walkers), dtype=bool)
    sizes = np.zeros(int(n_partitions), dtype=int)
    for idx in range(int(n_partitions)):
        if balanced:
            k = n_walkers // 2
            k = int(np.clip(k, min_size, n_walkers - min_size))
        else:
            low = min_size
            high = n_walkers - min_size
            if low >= high:
                k = n_walkers // 2
            else:
                k = int(rng.integers(low=low, high=high + 1))
        selected = rng.choice(n_walkers, size=k, replace=False)
        masks[idx, selected] = True
        sizes[idx] = k

    valid = np.logical_and(masks.any(axis=1), (~masks).any(axis=1))
    cuts = sizes.astype(float) / float(n_walkers)
    return masks[valid], cuts[valid]


def _compute_companion_geometric_weights(
    positions: np.ndarray,
    companions: np.ndarray,
    metric_tensors: np.ndarray | None,
    volume_weights: np.ndarray | None,
    length_scale: float,
    min_eig: float,
    use_volume: bool,
    pbc: bool,
    bounds_low: np.ndarray | None,
    bounds_high: np.ndarray | None,
) -> np.ndarray:
    """Compute unnormalized Riemannian-kernel-volume edge weights for companion edges."""
    n_walkers = int(positions.shape[0])
    if n_walkers == 0:
        return np.zeros(0, dtype=float)

    comp_idx = np.asarray(companions, dtype=np.int64)
    comp_idx = np.clip(comp_idx, 0, n_walkers - 1)
    delta = positions[comp_idx] - positions
    if pbc and bounds_low is not None and bounds_high is not None:
        span = bounds_high - bounds_low
        safe_span = np.where(np.abs(span) > 1e-12, span, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = delta - span * np.round(delta / safe_span)

    d = int(positions.shape[1])
    metric_arr = None if metric_tensors is None else np.asarray(metric_tensors, dtype=float)
    if (
        metric_arr is None
        or metric_arr.ndim != 3
        or metric_arr.shape != (n_walkers, d, d)
        or not np.all(np.isfinite(metric_arr))
    ):
        g = np.broadcast_to(np.eye(d, dtype=float), (n_walkers, d, d)).copy()
    else:
        g = metric_arr

    try:
        eigvals, eigvecs = np.linalg.eigh(g)
        eigvals = np.clip(eigvals, float(min_eig), None)
        g = np.einsum("nik,nk,njk->nij", eigvecs, eigvals, eigvecs)
    except (np.linalg.LinAlgError, ValueError):
        g = np.broadcast_to(np.eye(d, dtype=float), (n_walkers, d, d)).copy()

    g_edge = 0.5 * (g + g[comp_idx])
    d_sq = np.einsum("ni,nij,nj->n", delta, g_edge, delta)
    d_sq = np.clip(d_sq, 0.0, None)
    l = float(max(length_scale, 1e-6))
    kernel = np.exp(-d_sq / (2.0 * l * l))

    if use_volume and volume_weights is not None and volume_weights.shape[0] == n_walkers:
        vol = np.asarray(volume_weights, dtype=float)
        vol = np.clip(vol, 1e-12, None)
        kernel = kernel * vol[comp_idx]
    return kernel


def _count_cross_boundary(masks: np.ndarray, companions: np.ndarray) -> np.ndarray:
    """Count cross-boundary companion edges for each mask in a sweep."""
    if masks.size == 0:
        return np.zeros(0, dtype=np.int64)
    companion_idx = np.asarray(companions, dtype=np.int64)
    companion_idx = np.clip(companion_idx, 0, masks.shape[1] - 1)
    return np.sum(masks ^ masks[:, companion_idx], axis=1, dtype=np.int64)


def _count_crossing_lineages(masks: np.ndarray, lineage_ids: np.ndarray) -> np.ndarray:
    """Count lineages with descendants on both sides of each boundary mask."""
    if masks.size == 0:
        return np.zeros(0, dtype=np.int64)

    labels = np.asarray(lineage_ids, dtype=np.int64)
    areas = np.zeros(masks.shape[0], dtype=np.int64)
    for idx, mask in enumerate(masks):
        lineages_a = np.unique(labels[mask])
        lineages_ac = np.unique(labels[~mask])
        areas[idx] = np.intersect1d(lineages_a, lineages_ac, assume_unique=True).size
    return areas


def _sum_cross_boundary_weights(
    masks: np.ndarray,
    companions: np.ndarray,
    edge_weights: np.ndarray,
) -> np.ndarray:
    """Sum weighted companion edges crossing the boundary for each mask."""
    if masks.size == 0:
        return np.zeros(0, dtype=float)
    comp_idx = np.asarray(companions, dtype=np.int64)
    comp_idx = np.clip(comp_idx, 0, masks.shape[1] - 1)
    cross = masks ^ masks[:, comp_idx]
    return np.sum(cross * edge_weights[None, :], axis=1, dtype=float)


def _count_crossing_lineages_weighted(
    masks: np.ndarray,
    lineage_ids: np.ndarray,
    volume_weights: np.ndarray | None,
    use_weighted_area: bool,
) -> np.ndarray:
    """Count/weight crossing lineages for each mask."""
    if masks.size == 0:
        return np.zeros(0, dtype=float)
    if not use_weighted_area or volume_weights is None:
        return _count_crossing_lineages(masks, lineage_ids).astype(float)

    labels = np.asarray(lineage_ids, dtype=np.int64)
    vols = np.asarray(volume_weights, dtype=float)
    if vols.shape[0] != labels.shape[0]:
        return _count_crossing_lineages(masks, lineage_ids).astype(float)
    vols = np.clip(vols, 1e-12, None)

    values = np.zeros(masks.shape[0], dtype=float)
    for idx, mask in enumerate(masks):
        lineages_a = np.unique(labels[mask])
        lineages_ac = np.unique(labels[~mask])
        crossing = np.intersect1d(lineages_a, lineages_ac, assume_unique=True)
        if crossing.size == 0:
            continue
        total = 0.0
        for lineage in crossing:
            in_a = np.logical_and(mask, labels == lineage)
            in_b = np.logical_and(~mask, labels == lineage)
            if not in_a.any() or not in_b.any():
                continue
            va = float(np.mean(vols[in_a]))
            vb = float(np.mean(vols[in_b]))
            total += np.sqrt(max(va, 1e-12) * max(vb, 1e-12))
        values[idx] = total
    return values


def _fit_linear_relation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = alpha*x + b and return (alpha, b, R2)."""
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]):
        return float("nan"), float("nan"), float("nan")

    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), r2


def _select_fractal_set_metric_map(settings: FractalSetSettings) -> dict[str, str]:
    """Select which metric family should be regressed/plotted."""
    include_geom = bool(settings.use_geometry_correction)
    mode = str(settings.metric_display)

    metric_map: dict[str, str] = {}
    if mode in {"raw", "both"} or not include_geom:
        metric_map.update(FRACTAL_SET_METRICS_RAW)
    if include_geom and mode in {"geometry", "both"}:
        metric_map.update(FRACTAL_SET_METRICS_GEOM)
    if not metric_map:
        metric_map.update(FRACTAL_SET_METRICS_RAW)
    return metric_map


def _compute_fractal_set_measurements(
    history: RunHistory,
    settings: FractalSetSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute IG/CST boundary measurements from a RunHistory."""
    companions_dist = _to_numpy(history.companions_distance).astype(np.int64, copy=False)
    companions_fit = _to_numpy(history.companions_clone).astype(np.int64, copy=False)
    will_clone = _to_numpy(history.will_clone).astype(bool, copy=False)
    x_pre = _to_numpy(history.x_before_clone)
    volume_series = (
        _to_numpy(history.riemannian_volume_weights)
        if getattr(history, "riemannian_volume_weights", None) is not None
        else None
    )
    metric_series = (
        _to_numpy(history.diffusion_tensors_full)
        if getattr(history, "diffusion_tensors_full", None) is not None
        else None
    )

    n_transitions = int(
        min(
            companions_dist.shape[0],
            companions_fit.shape[0],
            will_clone.shape[0],
            max(0, x_pre.shape[0] - 1),
        )
    )
    n_walkers = int(history.N)

    if n_transitions <= 0 or n_walkers <= 1:
        empty = pd.DataFrame()
        return empty, empty, empty

    lineage_by_transition = _build_lineage_by_transition(
        companions_fit[:n_transitions],
        will_clone[:n_transitions],
        n_walkers=n_walkers,
    )
    frame_ids = _select_fractal_set_frames(
        n_transitions=n_transitions,
        warmup_fraction=float(settings.warmup_fraction),
        frame_stride=int(settings.frame_stride),
        max_frames=int(settings.max_frames) if settings.max_frames is not None else None,
    )
    if not frame_ids:
        empty = pd.DataFrame()
        return empty, empty, empty

    kernel_length = settings.geometry_kernel_length_scale
    if kernel_length is None:
        params = history.params if isinstance(history.params, dict) else {}
        kinetic = params.get("kinetic", {}) if isinstance(params.get("kinetic", {}), dict) else {}
        kernel_length = kinetic.get("viscous_length_scale", 1.0)
    try:
        kernel_length = float(max(float(kernel_length), 1e-6))
    except (TypeError, ValueError):
        kernel_length = 1.0

    use_geom = bool(settings.use_geometry_correction)
    pbc = bool(getattr(history, "pbc", False))
    if pbc and history.bounds is not None:
        bounds_low = _to_numpy(history.bounds.low).astype(float)
        bounds_high = _to_numpy(history.bounds.high).astype(float)
    else:
        bounds_low = None
        bounds_high = None

    include_spatial = settings.partition_family in {"all", "spatial"}
    include_graph = settings.partition_family in {"all", "graph"}
    include_random = settings.partition_family in {"all", "random"}

    if include_spatial:
        if settings.cut_geometry == "all":
            spatial_cut_types = FRACTAL_SET_CUT_TYPES
        else:
            spatial_cut_types = (str(settings.cut_geometry),)
    else:
        spatial_cut_types = tuple()

    graph_specs: list[tuple[str, str]] = []
    if include_graph:
        graph_source = str(settings.graph_cut_source)
        if graph_source == "all":
            graph_specs = [
                ("spectral_distance", "distance"),
                ("spectral_fitness", "fitness"),
                ("spectral_both", "both"),
            ]
        elif graph_source == "distance":
            graph_specs = [("spectral_distance", "distance")]
        elif graph_source == "fitness":
            graph_specs = [("spectral_fitness", "fitness")]
        elif graph_source == "both":
            graph_specs = [("spectral_both", "both")]

    rng = np.random.default_rng(int(settings.random_seed))

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for info_idx in frame_ids:
        positions = np.asarray(x_pre[info_idx + 1], dtype=float)
        if positions.ndim != 2 or positions.shape[0] != n_walkers:
            continue
        axis = int(np.clip(settings.partition_axis, 0, max(positions.shape[1] - 1, 0)))
        labels = lineage_by_transition[info_idx]
        step = int(history.recorded_steps[min(info_idx + 1, len(history.recorded_steps) - 1)])
        vol_frame = None
        if volume_series is not None and info_idx < volume_series.shape[0]:
            vol_frame = np.asarray(volume_series[info_idx], dtype=float)
        metric_frame = None
        if metric_series is not None and info_idx < metric_series.shape[0]:
            metric_frame = np.asarray(metric_series[info_idx], dtype=float)

        if use_geom:
            w_dist = _compute_companion_geometric_weights(
                positions=positions,
                companions=companions_dist[info_idx],
                metric_tensors=metric_frame,
                volume_weights=vol_frame,
                length_scale=kernel_length,
                min_eig=float(settings.geometry_min_eig),
                use_volume=bool(settings.geometry_use_volume),
                pbc=pbc,
                bounds_low=bounds_low,
                bounds_high=bounds_high,
            )
            w_fit = _compute_companion_geometric_weights(
                positions=positions,
                companions=companions_fit[info_idx],
                metric_tensors=metric_frame,
                volume_weights=vol_frame,
                length_scale=kernel_length,
                min_eig=float(settings.geometry_min_eig),
                use_volume=bool(settings.geometry_use_volume),
                pbc=pbc,
                bounds_low=bounds_low,
                bounds_high=bounds_high,
            )
        else:
            w_dist = np.ones(n_walkers, dtype=float)
            w_fit = np.ones(n_walkers, dtype=float)

        def _append_partition_measurements(
            masks: np.ndarray,
            cut_values: np.ndarray,
            cut_type: str,
            partition_family: str,
        ) -> None:
            if masks.size == 0:
                return
            s_dist = _count_cross_boundary(masks, companions_dist[info_idx])
            s_fit = _count_cross_boundary(masks, companions_fit[info_idx])
            s_total = s_dist + s_fit
            area_cst = _count_crossing_lineages(masks, labels)
            s_dist_geom = _sum_cross_boundary_weights(
                masks, companions_dist[info_idx], w_dist
            )
            s_fit_geom = _sum_cross_boundary_weights(
                masks, companions_fit[info_idx], w_fit
            )
            s_total_geom = s_dist_geom + s_fit_geom
            area_cst_geom = _count_crossing_lineages_weighted(
                masks=masks,
                lineage_ids=labels,
                volume_weights=vol_frame,
                use_weighted_area=bool(settings.geometry_correct_area and use_geom),
            )
            region_size = masks.sum(axis=1, dtype=np.int64)

            for local_idx, cut_value in enumerate(cut_values.tolist()):
                rows.append(
                    {
                        "info_idx": int(info_idx),
                        "recorded_step": step,
                        "partition_family": partition_family,
                        "cut_type": cut_type,
                        "cut_value": float(cut_value),
                        "region_size": int(region_size[local_idx]),
                        "area_cst": int(area_cst[local_idx]),
                        "area_cst_geom": float(area_cst_geom[local_idx]),
                        "s_dist": int(s_dist[local_idx]),
                        "s_fit": int(s_fit[local_idx]),
                        "s_total": int(s_total[local_idx]),
                        "s_dist_geom": float(s_dist_geom[local_idx]),
                        "s_fit_geom": float(s_fit_geom[local_idx]),
                        "s_total_geom": float(s_total_geom[local_idx]),
                    }
                )

            frame_rows.append(
                {
                    "info_idx": int(info_idx),
                    "recorded_step": step,
                    "partition_family": partition_family,
                    "cut_type": cut_type,
                    "n_partitions": int(masks.shape[0]),
                    "mean_area_cst": float(np.mean(area_cst)),
                    "mean_area_cst_geom": float(np.mean(area_cst_geom)),
                    "mean_s_dist": float(np.mean(s_dist)),
                    "mean_s_fit": float(np.mean(s_fit)),
                    "mean_s_total": float(np.mean(s_total)),
                    "mean_s_dist_geom": float(np.mean(s_dist_geom)),
                    "mean_s_fit_geom": float(np.mean(s_fit_geom)),
                    "mean_s_total_geom": float(np.mean(s_total_geom)),
                }
            )

        for cut_type in spatial_cut_types:
            masks, cut_values = _build_region_masks(
                positions=positions,
                axis=axis,
                cut_type=cut_type,
                n_cut_samples=int(settings.n_cut_samples),
            )
            _append_partition_measurements(
                masks=masks,
                cut_values=cut_values,
                cut_type=cut_type,
                partition_family="spatial",
            )

        for cut_type, source in graph_specs:
            if source == "distance":
                companions_for_graph = companions_dist[info_idx]
            elif source == "fitness":
                companions_for_graph = companions_fit[info_idx]
            else:
                companions_for_graph = np.stack(
                    [companions_dist[info_idx], companions_fit[info_idx]],
                    axis=0,
                )
            masks, cut_values = _build_spectral_sweep_masks(
                companions=companions_for_graph,
                n_cut_samples=int(settings.n_cut_samples),
                min_partition_size=int(settings.min_partition_size),
            )
            _append_partition_measurements(
                masks=masks,
                cut_values=cut_values,
                cut_type=cut_type,
                partition_family="graph",
            )

        if include_random:
            masks, cut_values = _build_random_partition_masks(
                n_walkers=n_walkers,
                n_partitions=int(settings.random_partitions),
                min_partition_size=int(settings.min_partition_size),
                balanced=bool(settings.random_balanced),
                rng=rng,
            )
            _append_partition_measurements(
                masks=masks,
                cut_values=cut_values,
                cut_type=FRACTAL_SET_RANDOM_CUT_TYPE,
                partition_family="random",
            )

    points_df = pd.DataFrame(rows)
    frame_df = pd.DataFrame(frame_rows)
    if points_df.empty:
        empty = pd.DataFrame()
        return points_df, empty, frame_df

    metric_map = _select_fractal_set_metric_map(settings)
    regression_rows: list[dict[str, Any]] = []
    for metric_key, metric_label in metric_map.items():
        area_key = "area_cst_geom" if metric_key.endswith("_geom") else "area_cst"
        x_all = points_df[area_key].to_numpy(dtype=float)
        y_all = points_df[metric_key].to_numpy(dtype=float)
        slope, intercept, r2 = _fit_linear_relation(x_all, y_all)
        regression_rows.append(
            {
                "metric": metric_key,
                "metric_label": metric_label,
                "area_key": area_key,
                "cut_type": "all",
                "partition_family": "all",
                "n_points": int(x_all.size),
                "slope_alpha": slope,
                "intercept": intercept,
                "r2": r2,
            }
        )

        for cut_type in sorted(points_df["cut_type"].unique()):
            subset = points_df[points_df["cut_type"] == cut_type]
            x = subset[area_key].to_numpy(dtype=float)
            y = subset[metric_key].to_numpy(dtype=float)
            slope, intercept, r2 = _fit_linear_relation(x, y)
            families = subset["partition_family"].unique().tolist()
            family = str(families[0]) if len(families) == 1 else "mixed"
            regression_rows.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "area_key": area_key,
                    "cut_type": str(cut_type),
                    "partition_family": family,
                    "n_points": int(x.size),
                    "slope_alpha": slope,
                    "intercept": intercept,
                    "r2": r2,
                }
            )

        for family in sorted(points_df["partition_family"].unique()):
            subset = points_df[points_df["partition_family"] == family]
            x = subset[area_key].to_numpy(dtype=float)
            y = subset[metric_key].to_numpy(dtype=float)
            slope, intercept, r2 = _fit_linear_relation(x, y)
            regression_rows.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "area_key": area_key,
                    "cut_type": f"family:{family}",
                    "partition_family": str(family),
                    "n_points": int(x.size),
                    "slope_alpha": slope,
                    "intercept": intercept,
                    "r2": r2,
                }
            )

    regression_df = pd.DataFrame(regression_rows)
    return points_df, regression_df, frame_df


def _build_fractal_set_scatter_plot(
    points_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    metric_key: str,
    title: str,
) -> Any:
    """Build scatter + linear fits for S_IG metric versus Area_CST."""
    if points_df.empty:
        return hv.Text(0, 0, "No Fractal Set measurements")

    colors = {
        "hyperplane": "#4c78a8",
        "spherical": "#f58518",
        "median": "#54a24b",
        "spectral_distance": "#b279a2",
        "spectral_fitness": "#e45756",
        "spectral_both": "#72b7b2",
        FRACTAL_SET_RANDOM_CUT_TYPE: "#7f7f7f",
    }
    overlays: list[Any] = []

    cut_types = sorted(points_df["cut_type"].unique().tolist())
    for cut_type in cut_types:
        subset = points_df[points_df["cut_type"] == cut_type]
        if subset.empty:
            continue
        x_key = "area_cst_geom" if metric_key.endswith("_geom") else "area_cst"

        label = str(cut_type).replace("_", " ").title()
        color = colors.get(cut_type, "#4c78a8")
        overlays.append(
            hv.Scatter(
                subset,
                kdims=[x_key],
                vdims=[
                    metric_key,
                    "recorded_step",
                    "cut_value",
                    "region_size",
                    "partition_family",
                ],
                label=label,
            ).opts(
                color=color,
                alpha=0.55,
                size=5,
                tools=["hover"],
                marker="circle",
            )
        )

        fit_rows = regression_df[
            (regression_df["metric"] == metric_key) & (regression_df["cut_type"] == cut_type)
        ]
        if fit_rows.empty:
            continue
        fit = fit_rows.iloc[0]
        slope = float(fit.get("slope_alpha", float("nan")))
        intercept = float(fit.get("intercept", float("nan")))
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue

        x_min = float(subset[x_key].min())
        x_max = float(subset[x_key].max())
        if np.isclose(x_min, x_max):
            continue
        x_line = np.linspace(x_min, x_max, num=80)
        y_line = slope * x_line + intercept
        overlays.append(
            hv.Curve((x_line, y_line), label=f"{label} fit").opts(
                color=color,
                line_width=2,
                line_dash="dashed",
            )
        )

    if not overlays:
        return hv.Text(0, 0, "No valid Fractal Set data")

    y_label = (
        FRACTAL_SET_METRICS_RAW.get(metric_key)
        or FRACTAL_SET_METRICS_GEOM.get(metric_key)
        or metric_key
    )
    x_label = "Area_CST_geom(A)" if metric_key.endswith("_geom") else "Area_CST(A)"
    return hv.Overlay(overlays).opts(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        width=900,
        height=360,
        legend_position="top_left",
        show_grid=True,
        toolbar="above",
    )


def _format_fractal_set_summary(
    points_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    frame_df: pd.DataFrame,
) -> str:
    """Summarize Fractal Set boundary measurements and area-law fits."""
    if points_df.empty:
        return "## Fractal Set Summary\n_No valid measurements. Load a RunHistory and run compute._"

    lines = [
        "## Fractal Set Summary",
        f"- Transitions analyzed: {int(points_df['info_idx'].nunique())}",
        f"- Boundary samples: {int(len(points_df))}",
        f"- Geometries: {', '.join(sorted(points_df['cut_type'].unique()))}",
        f"- Partition families: {', '.join(sorted(points_df['partition_family'].unique()))}",
    ]

    all_rows = regression_df[regression_df["cut_type"] == "all"].copy()
    for _, fit in all_rows.sort_values("metric").iterrows():
        metric_key = str(fit.get("metric", ""))
        area_key = str(fit.get("area_key", "area_cst"))
        area_label = "Area_CST_geom" if area_key == "area_cst_geom" else "Area_CST"
        metric_label = str(
            fit.get("metric_label")
            or FRACTAL_SET_METRICS_RAW.get(metric_key)
            or FRACTAL_SET_METRICS_GEOM.get(metric_key)
            or metric_key
        )
        slope = float(fit.get("slope_alpha", float("nan")))
        r2 = float(fit.get("r2", float("nan")))
        n_points = int(fit.get("n_points", 0))
        if np.isfinite(slope):
            lines.append(
                f"- {metric_label} vs {area_label}: alpha={slope:.6f}, "
                f"R2={r2:.4f}, n={n_points}"
            )
        else:
            lines.append(
                f"- {metric_label} vs {area_label}: insufficient area variation for fit (n={n_points})"
            )
        baseline = regression_df[
            (regression_df["metric"] == metric_key)
            & (regression_df["cut_type"] == FRACTAL_SET_RANDOM_CUT_TYPE)
        ]
        if not baseline.empty:
            b = baseline.iloc[0]
            b_alpha = float(b.get("slope_alpha", float("nan")))
            b_r2 = float(b.get("r2", float("nan")))
            if np.isfinite(b_alpha) and np.isfinite(b_r2):
                lines.append(
                    f"  random baseline: alpha={b_alpha:.6f}, R2={b_r2:.4f}"
                )

    if not frame_df.empty:
        lines.append(f"- Mean per-frame Area_CST: {float(frame_df['mean_area_cst'].mean()):.3f}")
        if "mean_area_cst_geom" in frame_df:
            lines.append(
                f"- Mean per-frame Area_CST_geom: "
                f"{float(frame_df['mean_area_cst_geom'].mean()):.3f}"
            )
        lines.append(f"- Mean per-frame S_total: {float(frame_df['mean_s_total'].mean()):.3f}")
        if "mean_s_total_geom" in frame_df:
            lines.append(
                f"- Mean per-frame S_total_geom: "
                f"{float(frame_df['mean_s_total_geom'].mean()):.3f}"
            )

    lines.append("")
    lines.append(
        "N-scaling and geodesic/flat ablation are cross-run tests and should be done across "
        "multiple histories."
    )
    return "\n".join(lines)


def _build_fractal_set_baseline_comparison(regression_df: pd.DataFrame) -> pd.DataFrame:
    """Compare each cut family against the random partition baseline."""
    if regression_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_keys = sorted(regression_df["metric"].dropna().unique().tolist())
    for metric_key in metric_keys:
        metric_df = regression_df[regression_df["metric"] == metric_key]
        if metric_df.empty:
            continue
        metric_label = str(
            metric_df.iloc[0].get("metric_label")
            or FRACTAL_SET_METRICS_RAW.get(metric_key)
            or FRACTAL_SET_METRICS_GEOM.get(metric_key)
            or metric_key
        )
        baseline_row = metric_df[metric_df["cut_type"] == FRACTAL_SET_RANDOM_CUT_TYPE]
        baseline_r2 = float("nan")
        baseline_alpha = float("nan")
        if not baseline_row.empty:
            baseline_r2 = float(baseline_row.iloc[0]["r2"])
            baseline_alpha = float(baseline_row.iloc[0]["slope_alpha"])

        for _, fit in metric_df.iterrows():
            cut_type = str(fit.get("cut_type", ""))
            if cut_type in {"all", FRACTAL_SET_RANDOM_CUT_TYPE}:
                continue
            if cut_type.startswith("family:"):
                continue
            r2 = float(fit.get("r2", float("nan")))
            alpha = float(fit.get("slope_alpha", float("nan")))
            rows.append(
                {
                    "metric": metric_label,
                    "cut_type": cut_type,
                    "partition_family": str(fit.get("partition_family", "")),
                    "n_points": int(fit.get("n_points", 0)),
                    "alpha": alpha,
                    "r2": r2,
                    "baseline_alpha_random": baseline_alpha,
                    "baseline_r2_random": baseline_r2,
                    "delta_r2_vs_random": (
                        r2 - baseline_r2
                        if np.isfinite(r2) and np.isfinite(baseline_r2)
                        else float("nan")
                    ),
                    "delta_alpha_vs_random": (
                        alpha - baseline_alpha
                        if np.isfinite(alpha) and np.isfinite(baseline_alpha)
                        else float("nan")
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()
    comparison = pd.DataFrame(rows)
    return comparison.sort_values(["metric", "delta_r2_vs_random"], ascending=[True, False])


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


# Reference masses for hadrons (GeV)
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


def _build_hadron_reference_table() -> pd.DataFrame:
    rows = []
    for name, mass in BARYON_REFS.items():
        rows.append({"family": "baryon", "name": name, "mass_ref_GeV": mass})
    for name, mass in MESON_REFS.items():
        rows.append({"family": "meson", "name": name, "mass_ref_GeV": mass})
    return pd.DataFrame(rows)


def _closest_reference(value: float, refs: dict[str, float]) -> tuple[str, float, float]:
    """Find closest reference mass and compute percent error."""
    name, ref = min(refs.items(), key=lambda kv: abs(value - kv[1]))
    err = (value - ref) / ref * 100.0
    return name, ref, err


def _best_fit_scale(
    masses: dict[str, float], anchors: list[tuple[str, float, str]]
) -> float | None:
    """Compute best-fit scale from algorithmic masses to physical masses."""
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
    """Format closest reference mass with percent error."""
    if value is None or value <= 0:
        return "n/a"
    name, ref, err = _closest_reference(value, refs)
    return f"{name} {ref:.3f} ({err:+.1f}%)"


def _build_best_fit_rows_generic(
    masses: dict[str, float],
    ref_groups: list[tuple[str, dict[str, float]]],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build best-fit scale rows from reference groups.

    *ref_groups* is a list of ``(group_name, {ref_name: mass_GeV})`` tuples.
    For strong force this is ``[("baryon", BARYON_REFS), ("meson", MESON_REFS)]``;
    for electroweak it is ``[("electroweak", user_refs)]``.
    """
    r2s = r2s or {}

    # Build per-group anchor lists
    group_anchors: list[tuple[str, list[tuple[str, float, str]]]] = []
    for group_name, refs in ref_groups:
        anchors = [(f"{group_name}->{name}", mass, group_name) for name, mass in refs.items()]
        group_anchors.append((group_name, anchors))

    # Build combined anchor list
    all_anchors: list[tuple[str, float, str]] = []
    for _group_name, anchors in group_anchors:
        all_anchors.extend(anchors)

    # If only one group, just compute a single fit row using its label
    if len(ref_groups) == 1:
        group_name, refs = ref_groups[0]
        anchors_for_fit = [(f"{name}->{mass:.6f}", mass, name) for name, mass in refs.items()]
        scale = _best_fit_scale(masses, anchors_for_fit)
        if scale is None:
            return [{"fit_mode": f"{group_name} refs", "scale_GeV_per_alg": None}]
        row: dict[str, Any] = {
            "fit_mode": f"{group_name} refs",
            "scale_GeV_per_alg": scale,
        }
        for name, alg_mass in masses.items():
            row[f"{name}_pred_GeV"] = alg_mass * scale
            if name in r2s:
                row[f"{name}_r2"] = r2s[name]
        return [row]

    # Multiple groups: per-group fits + combined fit
    rows: list[dict[str, Any]] = []
    fit_sets: list[tuple[str, list[tuple[str, float, str]]]] = []
    for group_name, anchors in group_anchors:
        fit_sets.append((f"{group_name} refs", anchors))
    combined_label = "+".join(g for g, _ in ref_groups) + " refs"
    fit_sets.append((combined_label, all_anchors))

    # Collect all mass keys that predictions should cover
    mass_keys = list(masses.keys())
    # Collect all reference dicts for closest matching
    all_refs: dict[str, dict[str, float]] = {g: r for g, r in ref_groups}

    for label, anchor_list in fit_sets:
        scale = _best_fit_scale(masses, anchor_list)
        if scale is None:
            rows.append({"fit_mode": label, "scale_GeV_per_alg": None})
            continue
        row = {
            "fit_mode": label,
            "scale_GeV_per_alg": scale,
        }
        for key in mass_keys:
            pred = masses.get(key, 0.0) * scale
            row[f"{key}_pred_GeV"] = pred
            # Find matching ref group for closest formatting
            if key in all_refs:
                row[f"closest_{key}"] = _format_closest(pred, all_refs[key])
            key_r2 = r2s.get(key)
            row[f"{key}_r2"] = key_r2 if key_r2 is not None and np.isfinite(key_r2) else None
        rows.append(row)
    return rows


def _build_anchor_rows_generic(
    masses: dict[str, float],
    anchors: list[tuple[str, float, str]],
    r2s: dict[str, float] | None = None,
    closest_refs: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Build anchored mass table using individual reference masses.

    *anchors* is ``[(label, physical_mass, mass_key)]``.
    *closest_refs* maps mass keys to reference dicts for closest-match formatting
    (e.g. ``{"baryon": BARYON_REFS, "meson": MESON_REFS}``).
    When *closest_refs* is ``None``, no closest matching is performed and all
    mass keys get ``{key}_pred_GeV`` columns.
    """
    r2s = r2s or {}
    closest_refs = closest_refs or {}
    mass_keys = list(masses.keys())

    rows: list[dict[str, Any]] = []
    for label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            rows.append({"anchor": label})
            continue
        scale = mass_phys / alg_mass
        row: dict[str, Any] = {
            "anchor": label,
            "scale_GeV_per_alg": scale,
        }
        for key in mass_keys:
            pred = masses.get(key, 0.0) * scale
            row[f"{key}_pred_GeV"] = pred
            if key in closest_refs:
                row[f"closest_{key}"] = _format_closest(pred, closest_refs[key])
            key_r2 = r2s.get(key)
            if key_r2 is not None and np.isfinite(key_r2):
                row[f"{key}_r2"] = key_r2
        rows.append(row)
    return rows


def _build_best_fit_rows(
    masses: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build best-fit scale rows using baryon/meson references."""
    return _build_best_fit_rows_generic(
        masses, [("baryon", BARYON_REFS), ("meson", MESON_REFS)], r2s,
    )


def _build_anchor_rows(
    masses: dict[str, float],
    glueball_ref: tuple[str, float] | None,
    sqrt_sigma_ref: float | None,
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build anchored mass table using individual reference masses."""
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

    closest_refs: dict[str, dict[str, float]] = {
        "baryon": BARYON_REFS,
        "meson": MESON_REFS,
    }
    if glueball_refs:
        closest_refs["glueball"] = glueball_refs
    if sqrt_sigma_ref is not None:
        closest_refs["sqrt_sigma"] = {f"{sqrt_sigma_ref:.3f}": sqrt_sigma_ref}

    return _build_anchor_rows_generic(masses, anchors, r2s, closest_refs)


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


_STRONG_FAMILY_MAP: dict[str, str] = {
    "pseudoscalar": "meson",
    "nucleon": "baryon",
    "glueball": "glueball",
}

STRONG_FORCE_RATIO_SPECS: list[tuple[str, str, str]] = [
    ("nucleon", "pseudoscalar", "proton/pion ≈ 6.7"),
    ("vector", "pseudoscalar", "rho/pion ≈ 5.5"),
    ("scalar", "pseudoscalar", "sigma/pion"),
    ("glueball", "pseudoscalar", ""),
]

RADIAL_STRONG_FORCE_RATIO_SPECS: list[tuple[str, str, str]] = [
    ("nucleon", "pseudoscalar", "proton/pion ≈ 6.7"),
    ("vector", "pseudoscalar", "rho/pion ≈ 5.5"),
    ("glueball", "pseudoscalar", "glueball/pion ≈ 10"),
]

ANISOTROPIC_EDGE_RATIO_SPECS: list[tuple[str, str, str]] = [
    ("nucleon", "pseudoscalar", "proton/pion ≈ 6.7"),
    ("vector", "pseudoscalar", "rho/pion ≈ 5.5"),
    ("scalar", "pseudoscalar", "sigma/pion"),
    ("glueball", "pseudoscalar", "glueball/pion ≈ 10"),
]


def _extract_masses(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
    mass_getter=None,
    family_map: dict[str, str] | None = _STRONG_FAMILY_MAP,
) -> dict[str, float]:
    """Extract masses from channel results.

    When *family_map* is provided, only channels present in the map are used and
    the returned keys are the mapped family names (e.g. "meson", "baryon").
    When *family_map* is ``None``, all channels with positive mass are returned
    using their original channel names (electroweak mode).
    """
    if mass_getter is None:
        mass_getter = _get_channel_mass
    masses: dict[str, float] = {}
    if family_map is not None:
        for channel, family in family_map.items():
            if channel in results:
                mass = mass_getter(results[channel], mode)
                if mass > 0:
                    masses[family] = mass
    else:
        for name, result in results.items():
            if result.n_samples == 0:
                continue
            mass = mass_getter(result, mode)
            if mass > 0:
                masses[name] = mass
    return masses


def _extract_r2(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
    family_map: dict[str, str] | None = _STRONG_FAMILY_MAP,
) -> dict[str, float]:
    """Extract R² values from channel results, using the same mapping logic as _extract_masses."""
    r2s: dict[str, float] = {}
    if family_map is not None:
        for channel, family in family_map.items():
            if channel in results:
                r2 = _get_channel_r2(results[channel], mode)
                if np.isfinite(r2):
                    r2s[family] = r2
    else:
        for name, result in results.items():
            if result.n_samples == 0:
                continue
            r2 = _get_channel_r2(result, mode)
            if np.isfinite(r2):
                r2s[name] = r2
    return r2s


def _format_ratios(
    results: dict[str, ChannelCorrelatorResult],
    mode: str = "AIC-Weighted",
    mass_getter=None,
    ratio_specs: list[tuple[str, str, str]] | None = None,
    title: str = "Mass Ratios",
) -> str:
    """Format mass ratios.

    In **spec-based** mode (*ratio_specs* given), each spec is
    ``(numerator_channel, denominator_channel, annotation)``.

    In **generic** mode (*ratio_specs* is ``None``), auto-selects the best
    denominator and computes all-vs-base ratios (electroweak style).
    """
    if mass_getter is None:
        mass_getter = _get_channel_mass

    if ratio_specs is not None:
        lines = [f"**{title}:**"]
        # Gather all unique denominators to check availability
        denom_masses: dict[str, float] = {}
        for _num, denom, _ann in ratio_specs:
            if denom not in denom_masses and denom in results:
                denom_masses[denom] = mass_getter(results[denom], mode)

        # Check first spec's denominator as the primary one
        primary_denom = ratio_specs[0][1]
        primary_mass = denom_masses.get(primary_denom, 0.0)
        if primary_mass <= 0:
            return f"**{title}:** n/a (no {primary_denom} mass)"

        for num, denom, annotation in ratio_specs:
            denom_mass = denom_masses.get(denom, 0.0)
            if denom_mass <= 0 or num not in results:
                continue
            num_mass = mass_getter(results[num], mode)
            if num_mass > 0:
                ratio = num_mass / denom_mass
                suffix = f" ({annotation})" if annotation else ""
                lines.append(f"- {num}/{denom}: **{ratio:.3f}**{suffix}")

        if len(lines) == 1:
            return f"**{title}:** n/a (no valid channel masses)"
        return "  \n".join(lines)

    # Generic mode (electroweak)
    if not results:
        return f"**{title}:** n/a"

    base_name = "u1_dressed" if "u1_dressed" in results else "u1_phase"
    base_result = results.get(base_name)
    if base_result is None:
        return f"**{title}:** n/a (no U1 mass)"
    base_mass = mass_getter(base_result, mode)
    if base_mass <= 0:
        return f"**{title}:** n/a (no U1 mass)"

    lines = [f"**{title} (proxy):**"]
    for name in sorted(results.keys()):
        if name == base_name:
            continue
        mass = mass_getter(results[name], mode)
        if mass > 0:
            ratio = mass / base_mass
            lines.append(f"- {name}/{base_name}: **{ratio:.3f}**")
    return "  \n".join(lines)


def _get_radial_mass(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    mass = _get_channel_mass(result, mode)
    dt = result.dt if result.dt and result.dt > 0 else 1.0
    return mass / dt


def _get_radial_mass_error(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    if mode == "AIC-Weighted":
        mass_error = result.mass_fit.get("mass_error", float("inf"))
    else:
        mass_error = 0.0
    dt = result.dt if result.dt and result.dt > 0 else 1.0
    return mass_error / dt




def _build_electroweak_best_fit_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    return _build_best_fit_rows_generic(masses, [("electroweak", refs)], r2s)


def _build_electroweak_anchor_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    r2s: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    anchors = [(f"{name}->{mass:.6f}", mass, name) for name, mass in refs.items()]
    return _build_anchor_rows_generic(masses, anchors, r2s)


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

    return float(np.nan)


def _update_mass_table(
    results: dict[str, ChannelCorrelatorResult],
    table,
    mode: str = "AIC-Weighted",
    mass_getter=None,
    error_getter=None,
) -> None:
    """Update a mass table widget with extracted masses (module-level, no closure)."""
    if mass_getter is None:
        mass_getter = _get_channel_mass
    if error_getter is None:
        # Default: extract mass_error from mass_fit directly
        def error_getter(result, m):
            if m == "AIC-Weighted":
                return result.mass_fit.get("mass_error", float("inf"))
            return 0.0

    rows = []
    for name, result in results.items():
        if result.n_samples == 0:
            continue

        mass = mass_getter(result, mode)
        mass_error = error_getter(result, mode)
        if mode == "AIC-Weighted":
            r2 = result.mass_fit.get("r_squared", float("nan"))
        else:
            best_window = result.mass_fit.get("best_window", {})
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
    table.value = pd.DataFrame(rows) if rows else pd.DataFrame()


def _update_correlator_plots(
    results: dict[str, ChannelCorrelatorResult],
    plateau_container,
    spectrum_pane,
    overlay_corr_pane,
    overlay_meff_pane,
    heatmap_container=None,
    heatmap_color_metric_widget=None,
    heatmap_alpha_metric_widget=None,
    spectrum_builder=None,
) -> None:
    """Update all correlator plots for a tab (module-level, no closure).

    Parameters
    ----------
    spectrum_builder : callable, optional
        Function to build the mass spectrum bar chart.  Signature is
        ``(results) -> hv.Bars | None``.  Defaults to ``build_mass_spectrum_bar``.
    """
    channel_plot_items = []
    heatmap_items = []

    for name, result in results.items():
        if result.n_samples == 0:
            continue

        channel_plot = ChannelPlot(result, logy=True, width=400, height=350)
        side_by_side = channel_plot.side_by_side()
        if side_by_side is not None:
            channel_plot_items.append(side_by_side)

        if (
            heatmap_container is not None
            and result.window_masses is not None
            and result.window_aic is not None
        ):
            window_masses = _to_numpy(result.window_masses)
            window_aic = _to_numpy(result.window_aic)
            window_r2 = _to_numpy(result.window_r2) if result.window_r2 is not None else None
            best_window = result.mass_fit.get("best_window", {})
            color_metric = str(heatmap_color_metric_widget.value) if heatmap_color_metric_widget else "mass"
            alpha_metric = str(heatmap_alpha_metric_widget.value) if heatmap_alpha_metric_widget else "aic"
            heatmap_plot = build_window_heatmap(
                window_masses,
                window_aic,
                result.window_widths or [],
                best_window,
                name,
                window_r2=window_r2,
                color_metric=color_metric,
                alpha_metric=alpha_metric,
            )
            if heatmap_plot is not None:
                heatmap_items.append(
                    pn.pane.HoloViews(heatmap_plot, sizing_mode="stretch_width", linked_axes=False)
                )

    plateau_container.objects = channel_plot_items if channel_plot_items else [
        pn.pane.Markdown("_No channel plots available._")
    ]

    if heatmap_container is not None:
        heatmap_container.objects = heatmap_items if heatmap_items else [
            pn.pane.Markdown("_No window heatmaps available._")
        ]

    if spectrum_builder is None:
        spectrum_builder = build_mass_spectrum_bar
    spectrum_pane.object = spectrum_builder(results)
    overlay_corr_pane.object = build_all_channels_overlay(results, plot_type="correlator")
    overlay_meff_pane.object = build_all_channels_overlay(results, plot_type="effective_mass")


def _update_strong_tables(
    results: dict[str, ChannelCorrelatorResult],
    mode: str,
    mass_table,
    ratio_pane,
    fit_table,
    anchor_table,
    glueball_ref_input,
    mass_getter=None,
    error_getter=None,
    ratio_specs=None,
) -> None:
    """Orchestrate all strong-force table updates (module-level, no closure)."""
    if ratio_specs is None:
        ratio_specs = STRONG_FORCE_RATIO_SPECS

    _update_mass_table(results, mass_table, mode, mass_getter=mass_getter, error_getter=error_getter)
    ratio_pane.object = _format_ratios(results, mode, mass_getter=mass_getter, ratio_specs=ratio_specs)

    channel_masses = _extract_masses(results, mode, mass_getter=mass_getter)
    channel_r2 = _extract_r2(results, mode)

    if not channel_masses:
        fit_table.value = pd.DataFrame()
        anchor_table.value = pd.DataFrame()
        return

    fit_rows = _build_best_fit_rows(channel_masses, channel_r2)
    fit_table.value = pd.DataFrame(fit_rows)

    glueball_ref = None
    if glueball_ref_input.value is not None:
        glueball_ref = ("glueball", float(glueball_ref_input.value))

    anchor_rows = _build_anchor_rows(
        channel_masses, glueball_ref, sqrt_sigma_ref=None, r2s=channel_r2,
    )
    anchor_table.value = pd.DataFrame(anchor_rows)


def _run_tab_computation(state, status_pane, label, compute_fn):
    """Run a correlator tab computation with shared guard/try/except pattern."""
    history = state.get("history")
    if history is None:
        status_pane.object = "**Error:** Load a RunHistory first."
        return
    status_pane.object = f"**Computing {label}...**"
    try:
        compute_fn(history)
    except Exception as e:
        status_pane.object = f"**Error:** {e}"
        import traceback
        traceback.print_exc()


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
        gas_config = GasConfigPanel.create_qft_config(spatial_dims=2, bounds_extent=12.0)
        gas_config.hide_viscous_kernel_widgets = False
        # Override with the best stable calibration settings found in QFT tuning.
        # This matches weak_potential_fit1_aniso_stable2 (200 walkers, 300 steps).
        gas_config.n_steps = 300
        gas_config.gas_params["N"] = 200
        gas_config.gas_params["dtype"] = "float32"
        gas_config.gas_params["pbc"] = False
        gas_config.gas_params["clone_every"] = 1
        gas_config.neighbor_graph_method = "delaunay"
        gas_config.neighbor_graph_record = True
        gas_config.neighbor_weight_modes = ["inverse_riemannian_distance", "kernel", "riemannian_kernel_volume"]
        gas_config.init_offset = 0.0
        gas_config.init_spread = 0.0
        gas_config.init_velocity_scale = 10.0

        # Riemannian Mix benchmark.
        benchmark, background, mode_points = prepare_benchmark_for_explorer(
            benchmark_name="Riemannian Mix",
            dims=gas_config.dims,
            bounds_range=(-gas_config.bounds_extent, gas_config.bounds_extent),
            resolution=100,
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
        gas_config.kinetic_op.nu = 1.0
        gas_config.kinetic_op.beta_curl = 1.0
        gas_config.kinetic_op.use_viscous_coupling = True
        gas_config.kinetic_op.viscous_length_scale = 1.0
        gas_config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel"
        gas_config.kinetic_op.viscous_neighbor_threshold = None
        gas_config.kinetic_op.viscous_neighbor_penalty = 0.0
        gas_config.kinetic_op.viscous_degree_cap = None
        gas_config.kinetic_op.use_velocity_squashing = True

        # Companion selection (diversity + cloning).
        gas_config.companion_selection.method = "uniform"
        gas_config.companion_selection.epsilon = 2.80
        gas_config.companion_selection.lambda_alg = 1.0
        gas_config.companion_selection.exclude_self = True
        gas_config.companion_selection_clone.method = "random_pairing"
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
            "fractal_set_points": None,
            "fractal_set_regressions": None,
            "fractal_set_frame_summary": None,
            "electroweak_results": None,
            "radial_results_4d": None,
            "radial_results_3d": None,
            "radial_results_3d_axes": None,
            "radial_ew_results_4d": None,
            "radial_ew_results_3d": None,
            "radial_ew_results_3d_axes": None,
            "anisotropic_edge_results": None,
            "einstein_test_result": None,
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
            min_width=335,
            sizing_mode="stretch_width",
        )
        browse_button = pn.widgets.Button(
            name="Browse files...",
            button_type="default",
            min_width=335,
            sizing_mode="stretch_width",
        )
        file_selector_container = pn.Column(sizing_mode="stretch_width")

        load_button = pn.widgets.Button(
            name="Load RunHistory",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
        )
        save_button = pn.widgets.Button(
            name="Save RunHistory",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
            disabled=True,
        )
        load_status = pn.pane.Markdown(
            "**Load a history**: paste a *_history.pt path or browse and click Load.",
            sizing_mode="stretch_width",
        )
        save_status = pn.pane.Markdown(
            "**Save a history**: run a simulation or load a RunHistory first.",
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
            min_width=335,
            sizing_mode="stretch_width",
        )
        analysis_id_input = pn.widgets.TextInput(
            name="Analysis id",
            value="",
            min_width=335,
            sizing_mode="stretch_width",
            placeholder="Optional (defaults to timestamp)",
        )
        run_analysis_button = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            min_width=335,
            sizing_mode="stretch_width",
            disabled=True,
        )
        run_analysis_button_main = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )

        # =====================================================================
        # Fractal Set tab components (IG/CST area-law measurements)
        # =====================================================================
        fractal_set_settings = FractalSetSettings()
        fractal_set_status = pn.pane.Markdown(
            "**Fractal Set:** Load a RunHistory and click Compute Fractal Set.",
            sizing_mode="stretch_width",
        )
        fractal_set_run_button = pn.widgets.Button(
            name="Compute Fractal Set",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        fractal_set_settings_panel = pn.Param(
            fractal_set_settings,
            parameters=[
                "warmup_fraction",
                "frame_stride",
                "max_frames",
                "partition_family",
                "n_cut_samples",
                "partition_axis",
                "cut_geometry",
                "graph_cut_source",
                "min_partition_size",
                "random_partitions",
                "random_balanced",
                "random_seed",
                "use_geometry_correction",
                "metric_display",
                "geometry_kernel_length_scale",
                "geometry_min_eig",
                "geometry_use_volume",
                "geometry_correct_area",
            ],
            show_name=False,
            widgets={
                "partition_family": {
                    "type": pn.widgets.Select,
                    "name": "Partition family",
                },
                "cut_geometry": {
                    "type": pn.widgets.Select,
                    "name": "Boundary geometry",
                },
                "graph_cut_source": {
                    "type": pn.widgets.Select,
                    "name": "Graph cut source",
                },
                "partition_axis": {
                    "name": "Partition axis",
                },
                "metric_display": {
                    "type": pn.widgets.Select,
                    "name": "Metric display",
                },
                "geometry_kernel_length_scale": {
                    "name": "Geometry kernel length scale",
                },
                "geometry_min_eig": {
                    "name": "Geometry min eigenvalue",
                },
                "geometry_use_volume": {
                    "name": "Use volume in edge weights",
                },
                "geometry_correct_area": {
                    "name": "Use volume-weighted CST area",
                },
            },
            default_layout=type("FractalSetSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        fractal_set_summary = pn.pane.Markdown(
            "## Fractal Set Summary\n_Compute Fractal Set to populate._",
            sizing_mode="stretch_width",
        )
        fractal_set_plot_dist = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        fractal_set_plot_fit = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        fractal_set_plot_total = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        fractal_set_plot_dist_geom = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        fractal_set_plot_fit_geom = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        fractal_set_plot_total_geom = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        fractal_set_regression_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        fractal_set_baseline_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        fractal_set_frame_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_width",
        )
        fractal_set_points_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_width",
        )

        # =====================================================================
        # Einstein equation test widgets (embedded in Fractal Set tab)
        # =====================================================================
        einstein_settings = EinsteinTestSettings()
        einstein_status = pn.pane.Markdown(
            "**Einstein Test:** Run Fractal Set first, then click.",
            sizing_mode="stretch_width",
        )
        einstein_run_button = pn.widgets.Button(
            name="Run Einstein Test",
            button_type="success",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        einstein_settings_panel = pn.Param(
            einstein_settings,
            show_name=False,
            parameters=[
                "mc_time_index", "regularization", "stress_energy_mode",
                "bulk_fraction", "g_newton_metric", "g_newton_manual",
            ],
        )
        einstein_summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        einstein_scalar_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        )
        einstein_scalar_log_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        )
        einstein_tensor_table = pn.pane.HoloViews(linked_axes=False)
        einstein_curvature_hist = pn.pane.HoloViews(linked_axes=False)
        einstein_residual_map = pn.pane.HoloViews(linked_axes=False)
        einstein_crosscheck_plot = pn.pane.HoloViews(linked_axes=False)
        einstein_bulk_boundary = pn.pane.Markdown("", sizing_mode="stretch_width")

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
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )

        channel_settings_panel = pn.Param(
            channel_settings,
            parameters=[
                "simulation_range",
                "max_lag",
                "h_eff",
                "mass",
                "ell0",
                "time_dimension",
                "mc_time_index",
                "euclidean_time_bins",
                "use_time_sliced_tessellation",
                "time_sliced_neighbor_mode",
                "neighbor_method",
                "edge_weight_mode",
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
                "compute_bootstrap_errors",
                "n_bootstrap",
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
                    "type": pn.widgets.EditableIntSlider,
                    "name": "Time Bins (Euclidean only)",
                    "start": 10,
                    "end": 500,
                    "step": 1,
                },
                "use_time_sliced_tessellation": {
                    "name": "Use sliced tessellation",
                },
                "time_sliced_neighbor_mode": {
                    "type": pn.widgets.Select,
                    "name": "Neighbor mode (sliced)",
                },
            },
            default_layout=type("ChannelSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        # Plot containers for channel tab
        # Channel plots container removed - now using channel_plateau_plots with ChannelPlot
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
        channel_ref_table = pn.widgets.Tabulator(
            _build_hadron_reference_table(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
        )

        # Optional: Channel-specific reference inputs
        channel_glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            min_width=200,
            sizing_mode="stretch_width",
        )

        # =====================================================================
        # Radial channels tab components (axis-free correlators)
        # =====================================================================
        radial_settings = RadialSettings()
        radial_status = pn.pane.Markdown(
            "**Radial Strong Force:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        radial_run_button = pn.widgets.Button(
            name="Compute Radial Strong Force",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        radial_settings_panel = pn.Param(
            radial_settings,
            parameters=[
                "time_axis",
                "mc_time_index",
                "simulation_range",
                "max_lag",
                "use_connected",
                "n_bins",
                "max_pairs",
                "distance_mode",
                "neighbor_method",
                "neighbor_weighting",
                "neighbor_k",
                "h_eff",
                "mass",
                "ell0",
                "use_volume_weights",
                "apply_power_correction",
                "power_override",
                "window_widths_spec",
                "channel_list",
                "drop_axis_average",
                "compute_bootstrap_errors",
                "n_bootstrap",
            ],
            show_name=False,
            widgets={
                "time_axis": {"type": pn.widgets.Select, "name": "Analysis Axis"},
                "mc_time_index": {"name": "MC start/slice (step or idx)"},
            },
            default_layout=type("RadialSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        radial_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )

        radial_heatmap_color_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Color",
            options=["mass", "aic", "r2"],
            value="mass",
            button_type="default",
        )
        radial_heatmap_alpha_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Opacity",
            options=["aic", "mass", "r2"],
            value="aic",
            button_type="default",
        )

        # Plot containers (4D) - using ChannelPlot for side-by-side display
        radial4d_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial4d_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial4d_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial4d_plateau_plots = pn.Column(sizing_mode="stretch_width")
        radial4d_heatmap_plots = pn.Column(sizing_mode="stretch_width")
        radial4d_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial4d_ratio_pane = pn.pane.Markdown(
            "**Mass Ratios:** Compute radial strong force channels to see ratios.",
            sizing_mode="stretch_width",
        )
        radial4d_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial4d_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ref_table = pn.widgets.Tabulator(
            _build_hadron_reference_table(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
        )

        radial_glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            min_width=200,
            sizing_mode="stretch_width",
        )

        # =====================================================================
        # Anisotropic edge channels tab components (direct recorded neighbors)
        # =====================================================================
        anisotropic_edge_settings = AnisotropicEdgeSettings()
        anisotropic_edge_status = pn.pane.Markdown(
            "**Anisotropic Edge Channels:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_run_button = pn.widgets.Button(
            name="Compute Anisotropic Edge Channels",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        anisotropic_edge_settings_panel = pn.Param(
            anisotropic_edge_settings,
            parameters=[
                "simulation_range",
                "mc_time_index",
                "max_lag",
                "use_connected",
                "h_eff",
                "mass",
                "ell0",
                "spatial_dims_spec",
                "edge_weight_mode",
                "use_volume_weights",
                "component_mode",
                "nucleon_triplet_mode",
                "channel_list",
                "window_widths_spec",
                "fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
                "compute_bootstrap_errors",
                "n_bootstrap",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC start (step or idx)"},
                "spatial_dims_spec": {"name": "Spatial dims (blank=all)"},
            },
            default_layout=type("AnisotropicEdgeSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        anisotropic_edge_summary = pn.pane.Markdown(
            "## Anisotropic Edge Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )
        anisotropic_edge_heatmap_color_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Color",
            options=["mass", "aic", "r2"],
            value="mass",
            button_type="default",
        )
        anisotropic_edge_heatmap_alpha_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Opacity",
            options=["aic", "mass", "r2"],
            value="aic",
            button_type="default",
        )
        anisotropic_edge_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        anisotropic_edge_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        anisotropic_edge_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        anisotropic_edge_plateau_plots = pn.Column(sizing_mode="stretch_width")
        anisotropic_edge_heatmap_plots = pn.Column(sizing_mode="stretch_width")
        anisotropic_edge_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_ratio_pane = pn.pane.Markdown(
            "**Mass Ratios:** Compute anisotropic edge channels to see ratios.",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_ref_table = pn.widgets.Tabulator(
            _build_hadron_reference_table(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
        )
        anisotropic_edge_glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            min_width=200,
            sizing_mode="stretch_width",
        )

        # =====================================================================
        # Radial Electroweak tab components (axis-free electroweak correlators)
        # =====================================================================
        radial_ew_settings = RadialElectroweakSettings()
        radial_ew_status = pn.pane.Markdown(
            "**Radial Electroweak:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        radial_ew_run_button = pn.widgets.Button(
            name="Compute Radial Electroweak",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        radial_ew_settings_panel = pn.Param(
            radial_ew_settings,
            parameters=[
                "mc_time_index",
                "n_bins",
                "max_pairs",
                "distance_mode",
                "neighbor_method",
                "edge_weight_mode",
                "neighbor_k",
                "use_volume_weights",
                "apply_power_correction",
                "power_override",
                "window_widths_spec",
                "channel_list",
                "drop_axis_average",
                "h_eff",
                "epsilon_d",
                "epsilon_c",
                "epsilon_clone",
                "lambda_alg",
                "compute_bootstrap_errors",
                "n_bootstrap",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC time slice (blank=last)"},
            },
            default_layout=type("RadialEWSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        radial_ew_summary = pn.pane.Markdown(
            "## Radial Electroweak Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        radial_ew_coupling_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew_coupling_ref_table = pn.widgets.Tabulator(
            pd.DataFrame(
                {
                    "name": list(ELECTROWEAK_COUPLING_NAMES),
                    **{
                        col: [
                            _format_ref_value(
                                DEFAULT_ELECTROWEAK_COUPLING_REFS.get(name, {}).get(col)
                            )
                            for name in ELECTROWEAK_COUPLING_NAMES
                        ]
                        for col in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS
                    },
                }
            ),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors={column: "input" for column in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS},
        )
        radial_ew_phase_plot = pn.pane.HTML(
            "<p><em>Phase histograms will appear after analysis.</em></p>",
            sizing_mode="stretch_width",
        )

        radial_ew_ref_table = pn.widgets.Tabulator(
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

        radial_ew_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )

        # Plot containers (4D) - using ChannelPlot for side-by-side display
        radial_ew4d_channel_plots = pn.Column(sizing_mode="stretch_width")
        radial_ew4d_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial_ew4d_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial_ew4d_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial_ew4d_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew4d_ratio_pane = pn.pane.Markdown(
            "**Electroweak Ratios:** Compute channels to see ratios.",
            sizing_mode="stretch_width",
        )
        radial_ew4d_ratio_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew4d_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew4d_compare_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew4d_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        # Plot containers (3D Avg)
        # 3D containers - using ChannelPlot for side-by-side display
        radial_ew3d_channel_plots = pn.Column(sizing_mode="stretch_width")
        radial_ew3d_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial_ew3d_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial_ew3d_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial_ew3d_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew3d_ratio_pane = pn.pane.Markdown(
            "**Electroweak Ratios:** Compute channels to see ratios.",
            sizing_mode="stretch_width",
        )
        radial_ew3d_ratio_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew3d_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew3d_compare_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial_ew3d_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        # Plot containers (3D average) - using ChannelPlot for side-by-side display
        radial3d_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial3d_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial3d_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        radial3d_plateau_plots = pn.Column(sizing_mode="stretch_width")
        radial3d_heatmap_plots = pn.Column(sizing_mode="stretch_width")
        radial3d_axis_grid = pn.GridBox(ncols=2, sizing_mode="stretch_width")
        radial3d_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial3d_ratio_pane = pn.pane.Markdown(
            "**Mass Ratios:** Compute radial strong force channels to see ratios.",
            sizing_mode="stretch_width",
        )
        radial3d_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        radial3d_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
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
            min_width=240,
            sizing_mode="stretch_width",
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
                "simulation_range",
                "max_lag",
                "h_eff",
                "time_dimension",
                "mc_time_index",
                "euclidean_time_bins",
                "use_time_sliced_tessellation",
                "time_sliced_neighbor_mode",
                "use_connected",
                "neighbor_method",
                "edge_weight_mode",
                "neighbor_k",
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
                "compute_bootstrap_errors",
                "n_bootstrap",
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
                    "type": pn.widgets.EditableIntSlider,
                    "name": "Time Bins (Euclidean only)",
                    "start": 10,
                    "end": 500,
                    "step": 1,
                },
                "use_time_sliced_tessellation": {
                    "name": "Use sliced tessellation",
                },
                "time_sliced_neighbor_mode": {
                    "type": pn.widgets.Select,
                    "name": "Neighbor mode (sliced)",
                },
            },
            default_layout=type("ElectroweakSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        # Electroweak channel plots - using ChannelPlot for side-by-side display
        electroweak_channel_plots = pn.Column(sizing_mode="stretch_width")
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

        # =====================================================================
        # Higgs Field tab components (emergent manifold geometry)
        # =====================================================================
        higgs_settings = HiggsSettings()
        higgs_status = pn.pane.Markdown(
            "**Higgs Field:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        higgs_run_button = pn.widgets.Button(
            name="Compute Higgs Observables",
            button_type="primary",
            width=240,
            height=40,
            sizing_mode="fixed",
            disabled=True,
        )
        higgs_settings_panel = pn.Param(
            higgs_settings,
            parameters=[
                "scalar_field_source",
                "mc_time_index",
                "h_eff",
                "mu_sq",
                "lambda_higgs",
                "alpha_gravity",
                "warmup_fraction",
                "compute_curvature",
                "compute_action",
                "metric_component_x",
                "metric_component_y",
            ],
            show_name=False,
            widgets={
                "scalar_field_source": {
                    "type": pn.widgets.Select,
                    "name": "Scalar Field Source",
                },
                "mc_time_index": {
                    "name": "MC time slice (blank=last)",
                },
            },
            default_layout=type("HiggsSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        # Higgs plot containers
        higgs_action_summary = pn.pane.Markdown(
            "**Action Summary:** _Compute observables to populate._",
            sizing_mode="stretch_width",
        )
        higgs_metric_heatmap = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        higgs_centroid_field = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        higgs_ricci_histogram = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        higgs_geodesic_scatter = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        higgs_volume_curvature = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        higgs_scalar_field_map = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        higgs_eigenvalue_dist = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )

        # =====================================================================
        # Dynamics tab components (forces, cloning, fitness distributions)
        # =====================================================================
        dynamics_status = pn.pane.Markdown(
            "**Dynamics:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        dynamics_run_button = pn.widgets.Button(
            name="Compute Dynamics",
            button_type="primary",
            width=240,
            height=40,
            sizing_mode="fixed",
            disabled=True,
        )
        dynamics_mc_slider = pn.widgets.IntSlider(
            name="MC time index",
            start=0,
            end=0,
            value=0,
            sizing_mode="stretch_width",
        )
        dirac_time_avg_checkbox = pn.widgets.Checkbox(
            name="Time-average spectrum", value=False,
        )
        dirac_warmup_slider = pn.widgets.FloatSlider(
            name="Warmup fraction",
            start=0.0,
            end=0.5,
            value=0.1,
            step=0.05,
            sizing_mode="stretch_width",
        )
        dirac_max_frames = pn.widgets.IntInput(
            name="Max avg frames",
            value=80,
            start=10,
            step=10,
            width=130,
        )
        dirac_threshold_mode = pn.widgets.Select(
            name="Color threshold",
            options={"Manual": "manual", "Median": "median"},
            value="manual",
            width=130,
        )
        dirac_threshold_value = pn.widgets.FloatInput(
            name="||F_visc|| threshold",
            value=1.0,
            start=0.0,
            step=0.1,
            width=130,
        )

        # Plot panes (HoloViews)
        dynamics_scatter_pane = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dynamics_fitness_hist = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dynamics_cloning_hist = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dynamics_force_hist = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dynamics_momentum_hist = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )

        # Dirac spectrum plot panes (attached to dynamics tab)
        dirac_full_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dirac_sector_up = pn.pane.HoloViews(linked_axes=False)
        dirac_sector_down = pn.pane.HoloViews(linked_axes=False)
        dirac_sector_nu = pn.pane.HoloViews(linked_axes=False)
        dirac_sector_lep = pn.pane.HoloViews(linked_axes=False)
        dirac_walker_classification = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dirac_mass_hierarchy = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dirac_chiral_density = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        dirac_generation_ratios = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )

        # Dirac fermion comparison widgets
        dirac_comparison_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False,
            sizing_mode="stretch_width",
        )
        dirac_ratio_comparison_pane = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        )
        dirac_summary_pane = pn.pane.Markdown(
            "", sizing_mode="stretch_width",
        )

        # =====================================================================
        # Quantum Gravity tab components (10 quantum gravity analyses)
        # =====================================================================
        qg_settings = QuantumGravitySettings()
        qg_status = pn.pane.Markdown(
            "**Quantum Gravity:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        qg_run_button = pn.widgets.Button(
            name="Compute Quantum Gravity",
            button_type="primary",
            width=240,
            height=40,
            sizing_mode="fixed",
            disabled=True,
        )
        qg_settings_panel = pn.Param(
            qg_settings,
            parameters=[
                "mc_time_index",
                "warmup_fraction",
                "analysis_dim_0",
                "analysis_dim_1",
                "analysis_dim_2",
                "use_metric_correction",
                "diffusion_time_steps",
                "max_diffusion_time",
                "n_radial_bins",
                "light_speed",
                "euclidean_time_dim",
                "euclidean_time_bins",
                "planck_length",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {
                    "name": "MC time slice (blank=last)",
                },
                "analysis_dim_0": {"name": "Analysis dim 1"},
                "analysis_dim_1": {"name": "Analysis dim 2"},
                "analysis_dim_2": {"name": "Analysis dim 3"},
            },
            default_layout=type("QGSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        # Quantum Gravity plot containers (20+ plots for 10 analyses)
        qg_summary_panel = pn.pane.Markdown(
            "**Summary:** _Compute observables to populate._",
            sizing_mode="stretch_width",
        )
        # 1. Regge Calculus
        qg_regge_action_density = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_deficit_angle_dist = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        # 2. Einstein-Hilbert
        qg_ricci_landscape = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_action_decomposition = pn.Column(sizing_mode="stretch_width")
        # 3. ADM Energy
        qg_adm_summary = pn.pane.Markdown(sizing_mode="stretch_width")
        qg_energy_density_dist = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        # 4. Spectral Dimension
        qg_spectral_dim_curve = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_heat_kernel_trace = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        # 5. Hausdorff Dimension
        qg_hausdorff_scaling = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_local_hausdorff_map = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        # 6. Causal Structure
        qg_causal_diamond = pn.Column(sizing_mode="stretch_width")
        qg_causal_violations = pn.pane.Markdown(sizing_mode="stretch_width")
        # 7. Holographic Entropy
        qg_holographic_summary = pn.pane.Markdown(sizing_mode="stretch_width")
        # 8. Spin Network
        qg_spin_distribution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_spin_network_summary = pn.pane.Markdown(sizing_mode="stretch_width")
        # 9. Raychaudhuri Expansion
        qg_expansion_field = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_convergence_regions = pn.pane.Markdown(sizing_mode="stretch_width")
        # 10. Geodesic Deviation
        qg_tidal_eigenvalues = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_tidal_summary = pn.pane.Markdown(sizing_mode="stretch_width")

        # Time Evolution (4D Spacetime Block Analysis)
        qg_time_summary = pn.pane.Markdown("**Time Evolution:** _Enable 'Compute Time Evolution' and run to populate._", sizing_mode="stretch_width")
        qg_regge_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_adm_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_spectral_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_hausdorff_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_holographic_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_raychaudhuri_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_causal_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_spin_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        qg_tidal_evolution = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)

        # =====================================================================
        # Weak Isospin tab components
        # =====================================================================
        isospin_status = pn.pane.Markdown(
            "**Weak Isospin:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        isospin_run_button = pn.widgets.Button(
            name="Compute Isospin Channels",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        # Plot panes — up-type
        isospin_up_spectrum = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        isospin_up_overlay_corr = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        isospin_up_overlay_meff = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        isospin_up_plateau = pn.Column()
        # Comparison tables — up-type
        isospin_up_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False, sizing_mode="stretch_width",
        )
        isospin_up_ratio_pane = pn.pane.Markdown(
            "**Mass Ratios (Up):** Compute to see ratios.", sizing_mode="stretch_width",
        )
        isospin_up_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False, sizing_mode="stretch_width",
        )
        isospin_up_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="remote", page_size=20,
            show_index=False, sizing_mode="stretch_width",
        )
        # Plot panes — down-type
        isospin_down_spectrum = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        isospin_down_overlay_corr = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        isospin_down_overlay_meff = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
        isospin_down_plateau = pn.Column()
        # Comparison tables — down-type
        isospin_down_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False, sizing_mode="stretch_width",
        )
        isospin_down_ratio_pane = pn.pane.Markdown(
            "**Mass Ratios (Down):** Compute to see ratios.", sizing_mode="stretch_width",
        )
        isospin_down_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False, sizing_mode="stretch_width",
        )
        isospin_down_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="remote", page_size=20,
            show_index=False, sizing_mode="stretch_width",
        )
        # Mass mode toggle (shared for both up/down)
        isospin_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )
        # Isospin splitting comparison
        isospin_split_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False, sizing_mode="stretch_width",
        )
        isospin_ratio_pane = pn.pane.Markdown("")

        def _set_analysis_status(message: str) -> None:
            analysis_status_sidebar.object = message
            analysis_status_main.object = message

        def set_history(history: RunHistory, history_path: Path | None = None) -> None:
            state["history"] = history
            state["history_path"] = history_path
            visualizer.bounds_extent = float(gas_config.bounds_extent)
            visualizer.set_history(history)
            _set_analysis_status("**Analysis ready:** click Run Analysis.")
            run_analysis_button.disabled = False
            run_analysis_button_main.disabled = False
            save_button.disabled = False
            save_status.object = "**Save a history**: choose a path and click Save."
            # Enable fractal set tab
            fractal_set_run_button.disabled = False
            fractal_set_status.object = "**Fractal Set ready:** click Compute Fractal Set."
            # Enable einstein test
            einstein_run_button.disabled = False
            einstein_status.object = "**Einstein Test ready:** run Fractal Set for G_N, then click."
            # Enable channels tab
            channels_run_button.disabled = False
            channels_status.object = "**Strong Force ready:** click Compute Channels."
            # Enable radial tab
            radial_run_button.disabled = False
            radial_status.object = "**Radial Strong Force ready:** click Compute Radial Strong Force."
            # Enable anisotropic edge tab
            anisotropic_edge_run_button.disabled = False
            anisotropic_edge_status.object = (
                "**Anisotropic Edge Channels ready:** click Compute Anisotropic Edge Channels."
            )
            # Enable radial electroweak tab
            radial_ew_run_button.disabled = False
            radial_ew_status.object = "**Radial Electroweak ready:** click Compute Radial Electroweak."
            # Enable electroweak tab
            electroweak_run_button.disabled = False
            electroweak_status.object = "**Electroweak ready:** click Compute Electroweak."
            # Enable higgs tab
            higgs_run_button.disabled = False
            higgs_status.object = "**Higgs Field ready:** click Compute Higgs Observables."
            # Enable quantum gravity tab
            qg_run_button.disabled = False
            qg_status.object = "**Quantum Gravity ready:** click Compute Quantum Gravity."
            # Enable isospin tab
            isospin_run_button.disabled = False
            isospin_status.object = "**Weak Isospin ready:** click Compute Isospin Channels."
            # Enable dynamics tab
            dynamics_run_button.disabled = False
            dynamics_mc_slider.end = max(0, history.n_recorded - 2)
            dynamics_mc_slider.value = max(0, history.n_recorded - 2)
            dynamics_status.object = "**Dynamics ready:** click Compute Dynamics."

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
                    gas_config.bounds_extent = int(inferred_extent)
                set_history(history, history_path)
                load_status.object = f"**Loaded:** `{history_path}`"
            except Exception as exc:
                load_status.object = f"**Error loading history:** {exc!s}"

        def on_save_clicked(_):
            history = state.get("history")
            if history is None:
                save_status.object = "**Error:** run a simulation or load a RunHistory first."
                return
            raw_path = history_path_input.value.strip()
            if raw_path:
                history_path = Path(raw_path).expanduser()
            else:
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                history_path = history_dir / f"qft_{stamp}_history.pt"
            if history_path.exists() and history_path.is_dir():
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                history_path = history_path / f"qft_{stamp}_history.pt"
            elif history_path.suffix != ".pt":
                history_path = history_path.with_suffix(".pt")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                history.save(str(history_path))
                state["history_path"] = history_path
                history_path_input.value = str(history_path)
                save_status.object = f"**Saved:** `{history_path}`"
            except Exception as exc:
                save_status.object = f"**Error saving history:** {exc!s}"

        def on_bounds_change(event):
            visualizer.bounds_extent = float(event.new)
            visualizer._refresh_frame()

        def _run_analysis() -> tuple[dict[str, Any], dict[str, Any]] | None:
            history = state.get("history")
            if history is None:
                _set_analysis_status("**Error:** load or run a simulation first.")
                return None

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

            try:
                with _temporary_argv(args):
                    qft_analysis.main()
            except Exception as exc:
                _set_analysis_status(f"**Error:** {exc!s}")
                return None

            metrics_path = output_dir / f"{analysis_id}_metrics.json"
            arrays_path = output_dir / f"{analysis_id}_arrays.npz"
            if not metrics_path.exists():
                _set_analysis_status("**Error:** analysis metrics file missing.")
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

        def _extract_coupling_refs(
            ref_table: pn.widgets.Tabulator,
        ) -> dict[str, dict[str, float]]:
            refs: dict[str, dict[str, float]] = {}
            ref_df = ref_table.value
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
                    refs=_extract_coupling_refs(electroweak_coupling_ref_table),
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

        def _update_radial_ew_summary(metrics: dict[str, Any]) -> None:
            radial_ew_summary.object = _format_electroweak_summary(metrics)
            history = state.get("history")
            proxies = metrics.get("electroweak_proxy", {}) if isinstance(metrics, dict) else {}
            couplings = _compute_coupling_constants(
                history,
                h_eff=float(radial_ew_settings.h_eff),
                epsilon_d=radial_ew_settings.epsilon_d,
                epsilon_c=radial_ew_settings.epsilon_c,
            )
            radial_ew_coupling_table.value = pd.DataFrame(
                _build_coupling_rows(
                    couplings,
                    proxies,
                    include_strong=False,
                    refs=_extract_coupling_refs(radial_ew_coupling_ref_table),
                )
            )
            plots = metrics.get("plots", {}) if isinstance(metrics, dict) else {}
            phase_path = plots.get("phase_histograms")
            if phase_path and Path(phase_path).exists():
                radial_ew_phase_plot.object = Path(phase_path).read_text()
            else:
                radial_ew_phase_plot.object = (
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
            _update_radial_ew_summary(metrics)

        def on_run_analysis(_):
            result = _run_analysis()
            if result is None:
                return
            metrics, arrays = result
            _update_analysis_outputs(metrics, arrays)
            _set_analysis_status("**Analysis complete.**")

        # =====================================================================
        # Fractal Set tab callbacks
        # =====================================================================

        def on_run_fractal_set(_):
            """Compute IG/CST area-law measurements from recorded companion traces."""

            def _compute(history):
                points_df, regression_df, frame_df = _compute_fractal_set_measurements(
                    history,
                    fractal_set_settings,
                )

                if points_df.empty:
                    fractal_set_summary.object = (
                        "## Fractal Set Summary\n"
                        "_No valid measurements for the selected settings._"
                    )
                    fractal_set_regression_table.value = pd.DataFrame()
                    fractal_set_baseline_table.value = pd.DataFrame()
                    fractal_set_frame_table.value = pd.DataFrame()
                    fractal_set_points_table.value = pd.DataFrame()
                    fractal_set_plot_dist.object = hv.Text(0, 0, "No data")
                    fractal_set_plot_fit.object = hv.Text(0, 0, "No data")
                    fractal_set_plot_total.object = hv.Text(0, 0, "No data")
                    fractal_set_plot_dist_geom.object = hv.Text(0, 0, "No data")
                    fractal_set_plot_fit_geom.object = hv.Text(0, 0, "No data")
                    fractal_set_plot_total_geom.object = hv.Text(0, 0, "No data")
                    fractal_set_status.object = (
                        "**Error:** Could not build any non-trivial boundary partitions."
                    )
                    return

                state["fractal_set_points"] = points_df
                state["fractal_set_regressions"] = regression_df
                state["fractal_set_frame_summary"] = frame_df

                display_regression = regression_df.copy()
                if not display_regression.empty:
                    if "metric_label" in display_regression.columns:
                        display_regression["metric"] = display_regression["metric_label"]
                    for column in ("slope_alpha", "intercept", "r2"):
                        display_regression[column] = pd.to_numeric(
                            display_regression[column], errors="coerce"
                        ).round(6)
                fractal_set_regression_table.value = display_regression
                fractal_set_baseline_table.value = _build_fractal_set_baseline_comparison(
                    regression_df
                )
                fractal_set_frame_table.value = frame_df.sort_values(
                    ["recorded_step", "partition_family", "cut_type"]
                ).reset_index(drop=True)
                fractal_set_points_table.value = points_df.sort_values(
                    ["recorded_step", "partition_family", "cut_type", "cut_value"]
                ).reset_index(drop=True)

                fractal_set_summary.object = _format_fractal_set_summary(
                    points_df,
                    regression_df,
                    frame_df,
                )
                show_geom = bool(fractal_set_settings.use_geometry_correction) and (
                    fractal_set_settings.metric_display in {"geometry", "both"}
                )
                show_raw = (
                    fractal_set_settings.metric_display in {"raw", "both"}
                    or not bool(fractal_set_settings.use_geometry_correction)
                )

                if show_raw:
                    fractal_set_plot_dist.object = _build_fractal_set_scatter_plot(
                        points_df,
                        regression_df,
                        metric_key="s_dist",
                        title="S_dist vs Area_CST",
                    )
                    fractal_set_plot_fit.object = _build_fractal_set_scatter_plot(
                        points_df,
                        regression_df,
                        metric_key="s_fit",
                        title="S_fit vs Area_CST",
                    )
                    fractal_set_plot_total.object = _build_fractal_set_scatter_plot(
                        points_df,
                        regression_df,
                        metric_key="s_total",
                        title="S_total vs Area_CST",
                    )
                else:
                    fractal_set_plot_dist.object = hv.Text(0, 0, "Raw metrics hidden")
                    fractal_set_plot_fit.object = hv.Text(0, 0, "Raw metrics hidden")
                    fractal_set_plot_total.object = hv.Text(0, 0, "Raw metrics hidden")

                if show_geom:
                    fractal_set_plot_dist_geom.object = _build_fractal_set_scatter_plot(
                        points_df,
                        regression_df,
                        metric_key="s_dist_geom",
                        title="S_dist_geom vs Area_CST_geom",
                    )
                    fractal_set_plot_fit_geom.object = _build_fractal_set_scatter_plot(
                        points_df,
                        regression_df,
                        metric_key="s_fit_geom",
                        title="S_fit_geom vs Area_CST_geom",
                    )
                    fractal_set_plot_total_geom.object = _build_fractal_set_scatter_plot(
                        points_df,
                        regression_df,
                        metric_key="s_total_geom",
                        title="S_total_geom vs Area_CST_geom",
                    )
                else:
                    fractal_set_plot_dist_geom.object = hv.Text(0, 0, "Geometry metrics hidden")
                    fractal_set_plot_fit_geom.object = hv.Text(0, 0, "Geometry metrics hidden")
                    fractal_set_plot_total_geom.object = hv.Text(0, 0, "Geometry metrics hidden")

                n_frames = int(points_df["info_idx"].nunique())
                n_samples = int(len(points_df))
                fractal_set_status.object = (
                    f"**Complete:** {n_samples} boundary samples from "
                    f"{n_frames} recorded transitions."
                )

            _run_tab_computation(state, fractal_set_status, "fractal set", _compute)

        # =====================================================================
        # Einstein equation test callback
        # =====================================================================

        def on_run_einstein_test(_):
            """Compute Einstein equation verification."""

            def _compute(history):
                config = EinsteinConfig(
                    mc_time_index=einstein_settings.mc_time_index,
                    regularization=einstein_settings.regularization,
                    stress_energy_mode=einstein_settings.stress_energy_mode,
                    bulk_fraction=einstein_settings.bulk_fraction,
                )
                g_metric = einstein_settings.g_newton_metric
                g_manual = einstein_settings.g_newton_manual
                result = compute_einstein_test(
                    history,
                    config,
                    fractal_set_regressions=state.get("fractal_set_regressions"),
                    g_newton_metric=g_metric,
                    g_newton_manual=g_manual,
                )
                state["einstein_test_result"] = result

                plots = build_all_einstein_plots(result)

                einstein_summary.object = plots["summary"]
                einstein_scalar_plot.object = plots["scalar_test"]
                einstein_scalar_log_plot.object = plots["scalar_test_log"]
                einstein_tensor_table.object = plots["tensor_r2"]
                einstein_curvature_hist.object = plots["curvature_dist"]
                einstein_residual_map.object = plots["residual_map"]
                if plots.get("crosscheck") is not None:
                    einstein_crosscheck_plot.object = plots["crosscheck"]
                einstein_bulk_boundary.object = plots["bulk_boundary"]

                full_volume_status = (
                    f"Full-volume R\u00b2={result.scalar_r2_full_volume:.4f}, "
                    if result.scalar_r2_full_volume is not None
                    else ""
                )
                einstein_status.object = (
                    f"**Complete:** {result.n_walkers} walkers, d={result.spatial_dim}. "
                    f"Ricci source={result.ricci_scalar_source}, "
                    f"Scalar R\u00b2={result.scalar_r2:.4f}, "
                    f"{full_volume_status}"
                    f"Tensor R\u00b2={result.tensor_r2:.4f}, "
                    f"G_N ratio={result.g_newton_ratio:.3f}"
                )

            _run_tab_computation(state, einstein_status, "Einstein equation test", _compute)

        # =====================================================================
        # Channels tab callbacks (vectorized correlator_channels)
        # =====================================================================

        def _update_channel_plots(results: dict[str, ChannelCorrelatorResult]) -> None:
            _update_correlator_plots(
                results,
                channel_plateau_plots,
                channel_plots_spectrum,
                channel_plots_overlay_corr,
                channel_plots_overlay_meff,
                heatmap_container=channel_heatmap_plots,
                heatmap_color_metric_widget=heatmap_color_metric,
                heatmap_alpha_metric_widget=heatmap_alpha_metric,
            )

        def _update_channel_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = channel_mass_mode.value
            _update_strong_tables(
                results,
                mode,
                channel_mass_table,
                channel_ratio_pane,
                channel_fit_table,
                channel_anchor_table,
                channel_glueball_ref_input,
            )

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
            def _compute(history):
                results = _compute_channels_vectorized(history, channel_settings)
                state["channel_results"] = results
                _update_channel_plots(results)
                _update_channel_tables(results)
                _update_strong_couplings(history)
                n_channels = len([r for r in results.values() if r.n_samples > 0])
                channels_status.object = f"**Complete:** {n_channels} channels computed."

            _run_tab_computation(state, channels_status, "channels (vectorized)", _compute)

        # =====================================================================
        # Weak Isospin tab callbacks
        # =====================================================================

        def _update_isospin_tables(
            results: dict[str, ChannelCorrelatorResult],
            mass_table,
            ratio_pane,
            fit_table,
            anchor_table,
            mode: str | None = None,
        ) -> None:
            """Update strong-force-style tables for one isospin component."""
            if mode is None:
                mode = isospin_mass_mode.value
            _update_strong_tables(
                results,
                mode,
                mass_table,
                ratio_pane,
                fit_table,
                anchor_table,
                channel_glueball_ref_input,
            )

        def _build_isospin_comparison(
            iso_result: IsospinChannelResult,
            mode: str | None = None,
        ) -> None:
            """Build splitting table: measured up/down ratios vs PDG per channel."""
            if mode is None:
                mode = isospin_mass_mode.value
            rows = []
            for ch in iso_result.up_results:
                up_r = iso_result.up_results.get(ch)
                down_r = iso_result.down_results.get(ch)
                if up_r is None or down_r is None:
                    continue
                m_up = _get_channel_mass(up_r, mode)
                m_down = _get_channel_mass(down_r, mode)
                measured_ratio = m_up / m_down if m_down and m_down != 0 else float("nan")

                # PDG observed splitting for this channel
                pdg = ISOSPIN_CHANNEL_SPLITTINGS.get(ch)
                if pdg is not None:
                    up_name, down_name, m_up_pdg, m_down_pdg, pdg_ratio = pdg
                else:
                    up_name, down_name = "?", "?"
                    m_up_pdg = m_down_pdg = pdg_ratio = float("nan")

                rows.append({
                    "Channel": ch,
                    "Up particle": up_name,
                    "Down particle": down_name,
                    "m_up (alg)": f"{m_up:.6f}" if m_up > 0 else "n/a",
                    "m_down (alg)": f"{m_down:.6f}" if m_down > 0 else "n/a",
                    "up/down (measured)": f"{measured_ratio:.4f}" if np.isfinite(measured_ratio) else "n/a",
                    "up/down (PDG)": f"{pdg_ratio:.6f}",
                    "m_up (PDG GeV)": f"{m_up_pdg:.6f}",
                    "m_down (PDG GeV)": f"{m_down_pdg:.6f}",
                })
            isospin_split_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

            # Quark-level reference
            lines = ["**PDG Quark Mass Ratios (up-type / down-type):**"]
            for label, val in ISOSPIN_MASS_RATIOS.items():
                lines.append(f"- **{label}**: {val:.4f}")
            isospin_ratio_pane.object = "  \n".join(lines)

        def _on_isospin_mass_mode_change(event):
            """Handle mass mode toggle — refresh all isospin tables."""
            iso = state.get("isospin_results")
            if iso is None:
                return
            _update_isospin_tables(
                iso.up_results, isospin_up_mass_table, isospin_up_ratio_pane,
                isospin_up_fit_table, isospin_up_anchor_table, event.new,
            )
            _update_isospin_tables(
                iso.down_results, isospin_down_mass_table, isospin_down_ratio_pane,
                isospin_down_fit_table, isospin_down_anchor_table, event.new,
            )
            _build_isospin_comparison(iso, event.new)

        isospin_mass_mode.param.watch(_on_isospin_mass_mode_change, "value")

        def on_run_isospin(_):
            """Compute isospin-split correlator channels."""
            def _compute(history):
                time_axis, euclidean_time_dim = _map_time_dimension(
                    channel_settings.time_dimension, history_d=history.d,
                )
                neighbor_method = (
                    "auto" if channel_settings.neighbor_method == "voronoi"
                    else channel_settings.neighbor_method
                )
                channel_config = ChannelConfig(
                    warmup_fraction=channel_settings.simulation_range[0],
                    end_fraction=channel_settings.simulation_range[1],
                    h_eff=channel_settings.h_eff,
                    mass=channel_settings.mass,
                    ell0=channel_settings.ell0,
                    neighbor_method=neighbor_method,
                    edge_weight_mode=channel_settings.edge_weight_mode,
                    mc_time_index=channel_settings.mc_time_index,
                    time_axis=time_axis,
                    euclidean_time_dim=euclidean_time_dim,
                    euclidean_time_bins=channel_settings.euclidean_time_bins,
                )
                correlator_config = CorrelatorConfig(
                    max_lag=channel_settings.max_lag,
                    use_connected=channel_settings.use_connected,
                    window_widths=_parse_window_widths(channel_settings.window_widths_spec),
                    fit_mode=channel_settings.fit_mode,
                    fit_start=channel_settings.fit_start,
                    fit_stop=channel_settings.fit_stop,
                    min_fit_points=channel_settings.min_fit_points,
                    compute_bootstrap_errors=channel_settings.compute_bootstrap_errors,
                    n_bootstrap=channel_settings.n_bootstrap,
                )
                channels = [c.strip() for c in channel_settings.channel_list.split(",") if c.strip()]

                iso_result = compute_isospin_channels(
                    history, channel_config, correlator_config, channels=channels,
                )
                state["isospin_results"] = iso_result

                # Update up-type plots + tables
                _update_correlator_plots(
                    iso_result.up_results,
                    isospin_up_plateau,
                    isospin_up_spectrum,
                    isospin_up_overlay_corr,
                    isospin_up_overlay_meff,
                )
                _update_isospin_tables(
                    iso_result.up_results, isospin_up_mass_table, isospin_up_ratio_pane,
                    isospin_up_fit_table, isospin_up_anchor_table,
                )
                # Update down-type plots + tables
                _update_correlator_plots(
                    iso_result.down_results,
                    isospin_down_plateau,
                    isospin_down_spectrum,
                    isospin_down_overlay_corr,
                    isospin_down_overlay_meff,
                )
                _update_isospin_tables(
                    iso_result.down_results, isospin_down_mass_table, isospin_down_ratio_pane,
                    isospin_down_fit_table, isospin_down_anchor_table,
                )
                # Build isospin splitting comparison
                _build_isospin_comparison(iso_result)

                n_up = len([r for r in iso_result.up_results.values() if r.n_samples > 0])
                n_down = len([r for r in iso_result.down_results.values() if r.n_samples > 0])
                isospin_status.object = (
                    f"**Complete:** {n_up} up-type + {n_down} down-type channels computed."
                )

            _run_tab_computation(state, isospin_status, "isospin channels", _compute)

        isospin_run_button.on_click(on_run_isospin)

        # =====================================================================
        # Radial channels tab callbacks
        # =====================================================================

        def _radial_spectrum_builder(results):
            return build_mass_spectrum_bar(
                results,
                mass_getter=_get_radial_mass,
                error_getter=_get_radial_mass_error,
                title="Geometry-Aware Channel Mass Spectrum",
                ylabel="Mass (physical units)",
            )

        def _update_radial_plots(
            results: dict[str, ChannelCorrelatorResult],
            plots_spectrum,
            plots_overlay_corr,
            plots_overlay_meff,
            channel_plots_container,
            heatmap_plots_container,
        ) -> None:
            _update_correlator_plots(
                results,
                channel_plots_container,
                plots_spectrum,
                plots_overlay_corr,
                plots_overlay_meff,
                heatmap_container=heatmap_plots_container,
                heatmap_color_metric_widget=radial_heatmap_color_metric,
                heatmap_alpha_metric_widget=radial_heatmap_alpha_metric,
                spectrum_builder=_radial_spectrum_builder,
            )

        def _update_radial_tables(
            results: dict[str, ChannelCorrelatorResult],
            table,
            ratio_pane,
            fit_table,
            anchor_table,
            glueball_input,
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = radial_mass_mode.value
            _update_strong_tables(
                results,
                mode,
                table,
                ratio_pane,
                fit_table,
                anchor_table,
                glueball_input,
                mass_getter=_get_radial_mass,
                error_getter=_get_radial_mass_error,
                ratio_specs=RADIAL_STRONG_FORCE_RATIO_SPECS,
            )

        def _on_radial_mass_mode_change(event):
            if state.get("radial_results_4d") is not None:
                _update_radial_tables(
                    state["radial_results_4d"],
                    radial4d_mass_table,
                    radial4d_ratio_pane,
                    radial4d_fit_table,
                    radial4d_anchor_table,
                    radial_glueball_ref_input,
                    event.new,
                )
            if state.get("radial_results_3d") is not None:
                _update_radial_tables(
                    state["radial_results_3d"],
                    radial3d_mass_table,
                    radial3d_ratio_pane,
                    radial3d_fit_table,
                    radial3d_anchor_table,
                    radial_glueball_ref_input,
                    event.new,
                )

        radial_mass_mode.param.watch(_on_radial_mass_mode_change, "value")

        def _on_radial_heatmap_metric_change(_event):
            if state.get("radial_results_4d") is not None:
                _update_radial_plots(
                    state["radial_results_4d"],
                    radial4d_plots_spectrum,
                    radial4d_plots_overlay_corr,
                    radial4d_plots_overlay_meff,
                    radial4d_plateau_plots,
                    radial4d_heatmap_plots,
                )
            if state.get("radial_results_3d") is not None:
                _update_radial_plots(
                    state["radial_results_3d"],
                    radial3d_plots_spectrum,
                    radial3d_plots_overlay_corr,
                    radial3d_plots_overlay_meff,
                    radial3d_plateau_plots,
                    radial3d_heatmap_plots,
                )

        radial_heatmap_color_metric.param.watch(_on_radial_heatmap_metric_change, "value")
        radial_heatmap_alpha_metric.param.watch(_on_radial_heatmap_metric_change, "value")

        def on_run_radial_channels(_):
            def _compute(history):
                bundle = _compute_radial_channels_bundle(history, radial_settings)
                state["radial_results_4d"] = bundle.radial_4d.channel_results
                state["radial_results_3d"] = (
                    bundle.radial_3d_avg.channel_results if bundle.radial_3d_avg else {}
                )
                state["radial_results_3d_axes"] = {
                    axis: output.channel_results
                    for axis, output in (bundle.radial_3d_by_axis or {}).items()
                }

                def _build_axis_panel(title: str, results: dict[str, ChannelCorrelatorResult]):
                    overlay_corr = build_all_channels_overlay(results, plot_type="correlator")
                    overlay_meff = build_all_channels_overlay(results, plot_type="effective_mass")
                    panel_items = [pn.pane.Markdown(f"#### {title}")]
                    if overlay_corr is not None:
                        panel_items.append(
                            pn.pane.HoloViews(
                                overlay_corr, sizing_mode="stretch_width", linked_axes=False
                            )
                        )
                    else:
                        panel_items.append(pn.pane.Markdown("_No correlator overlay._"))
                    if overlay_meff is not None:
                        panel_items.append(
                            pn.pane.HoloViews(
                                overlay_meff, sizing_mode="stretch_width", linked_axes=False
                            )
                        )
                    else:
                        panel_items.append(pn.pane.Markdown("_No effective-mass overlay._"))
                    return pn.Column(*panel_items, sizing_mode="stretch_width")

                _update_radial_plots(
                    bundle.radial_4d.channel_results,
                    radial4d_plots_spectrum,
                    radial4d_plots_overlay_corr,
                    radial4d_plots_overlay_meff,
                    radial4d_plateau_plots,
                    radial4d_heatmap_plots,
                )
                _update_radial_tables(
                    bundle.radial_4d.channel_results,
                    radial4d_mass_table,
                    radial4d_ratio_pane,
                    radial4d_fit_table,
                    radial4d_anchor_table,
                    radial_glueball_ref_input,
                )

                if bundle.radial_3d_avg is not None:
                    axis_panels = []
                    axis_panels.append(
                        _build_axis_panel("Average (drop-axis)", bundle.radial_3d_avg.channel_results)
                    )
                    axis_dict = bundle.radial_3d_by_axis or {}
                    if len(axis_dict) >= 4:
                        for axis in sorted(axis_dict.keys())[:3]:
                            axis_panels.append(
                                _build_axis_panel(
                                    f"Drop axis {axis}",
                                    axis_dict[axis].channel_results,
                                )
                            )
                    radial3d_axis_grid.objects = axis_panels

                    _update_radial_plots(
                        bundle.radial_3d_avg.channel_results,
                        radial3d_plots_spectrum,
                        radial3d_plots_overlay_corr,
                        radial3d_plots_overlay_meff,
                        radial3d_plateau_plots,
                        radial3d_heatmap_plots,
                    )
                    _update_radial_tables(
                        bundle.radial_3d_avg.channel_results,
                        radial3d_mass_table,
                        radial3d_ratio_pane,
                        radial3d_fit_table,
                        radial3d_anchor_table,
                        radial_glueball_ref_input,
                    )
                else:
                    radial3d_axis_grid.objects = [pn.pane.Markdown("_No 3D averages available._")]

                n_channels = len(
                    [r for r in bundle.radial_4d.channel_results.values() if r.n_samples > 0]
                )
                radial_status.object = (
                    f"**Complete:** {n_channels} radial strong force channels computed."
                )

            _run_tab_computation(state, radial_status, "radial strong force channels", _compute)

        # =====================================================================
        # Anisotropic edge channels tab callbacks
        # =====================================================================

        def _anisotropic_edge_spectrum_builder(results):
            return build_mass_spectrum_bar(
                results,
                title="Anisotropic Edge Channel Mass Spectrum",
                ylabel="Mass (algorithmic units)",
            )

        def _update_anisotropic_edge_plots(
            results: dict[str, ChannelCorrelatorResult],
        ) -> None:
            _update_correlator_plots(
                results,
                anisotropic_edge_plateau_plots,
                anisotropic_edge_plots_spectrum,
                anisotropic_edge_plots_overlay_corr,
                anisotropic_edge_plots_overlay_meff,
                heatmap_container=anisotropic_edge_heatmap_plots,
                heatmap_color_metric_widget=anisotropic_edge_heatmap_color_metric,
                heatmap_alpha_metric_widget=anisotropic_edge_heatmap_alpha_metric,
                spectrum_builder=_anisotropic_edge_spectrum_builder,
            )

        def _update_anisotropic_edge_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = anisotropic_edge_mass_mode.value
            _update_strong_tables(
                results,
                mode,
                anisotropic_edge_mass_table,
                anisotropic_edge_ratio_pane,
                anisotropic_edge_fit_table,
                anisotropic_edge_anchor_table,
                anisotropic_edge_glueball_ref_input,
                ratio_specs=ANISOTROPIC_EDGE_RATIO_SPECS,
            )

        def _on_anisotropic_edge_mass_mode_change(event):
            if state.get("anisotropic_edge_results") is None:
                return
            _update_anisotropic_edge_tables(state["anisotropic_edge_results"], event.new)

        anisotropic_edge_mass_mode.param.watch(_on_anisotropic_edge_mass_mode_change, "value")

        def _on_anisotropic_edge_heatmap_metric_change(_event):
            if state.get("anisotropic_edge_results") is None:
                return
            _update_anisotropic_edge_plots(state["anisotropic_edge_results"])

        anisotropic_edge_heatmap_color_metric.param.watch(
            _on_anisotropic_edge_heatmap_metric_change,
            "value",
        )
        anisotropic_edge_heatmap_alpha_metric.param.watch(
            _on_anisotropic_edge_heatmap_metric_change,
            "value",
        )

        def on_run_anisotropic_edge_channels(_):
            def _compute(history):
                output = _compute_anisotropic_edge_bundle(history, anisotropic_edge_settings)
                results = output.channel_results
                state["anisotropic_edge_results"] = results
                _update_anisotropic_edge_plots(results)
                _update_anisotropic_edge_tables(results)
                anisotropic_edge_summary.object = (
                    "## Anisotropic Edge Summary\n"
                    f"- Components: `{', '.join(output.component_labels)}`\n"
                    f"- Frames used: `{output.n_valid_frames}/{len(output.frame_indices)}`\n"
                    f"- Mean alive walkers/frame: `{output.avg_alive_walkers:.2f}`\n"
                    f"- Mean directed edges/frame: `{output.avg_edges:.2f}`\n"
                )
                n_channels = len([res for res in results.values() if res.n_samples > 0])
                anisotropic_edge_status.object = (
                    f"**Complete:** {n_channels} anisotropic edge channels computed."
                )

            _run_tab_computation(
                state,
                anisotropic_edge_status,
                "anisotropic edge channels",
                _compute,
            )

        # =====================================================================
        # Electroweak tab callbacks (U1/SU2 correlators)
        # =====================================================================

        def _update_electroweak_plots_generic(
            results: dict[str, ChannelCorrelatorResult],
            channel_plots_container,
            plots_spectrum,
            plots_overlay_corr,
            plots_overlay_meff,
        ) -> None:
            _update_correlator_plots(
                results,
                channel_plots_container,
                plots_spectrum,
                plots_overlay_corr,
                plots_overlay_meff,
            )

        def _update_electroweak_tables_generic(
            results: dict[str, ChannelCorrelatorResult],
            mode: str,
            mass_table,
            ratio_pane,
            ratio_table,
            fit_table,
            anchor_table,
            compare_table,
            ref_table,
        ) -> None:
            _update_mass_table(results, mass_table, mode)
            ratio_pane.object = _format_ratios(results, mode, title="Electroweak Ratios")

            masses = _extract_masses(results, mode, family_map=None)
            r2s = _extract_r2(results, mode, family_map=None)

            base_name = "u1_dressed" if "u1_dressed" in masses else "u1_phase"

            refs_df = ref_table.value
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
            ratio_table.value = pd.DataFrame(ratio_rows) if ratio_rows else pd.DataFrame()

            if not masses or not refs:
                fit_table.value = pd.DataFrame()
                anchor_table.value = pd.DataFrame()
                compare_table.value = pd.DataFrame()
                return

            fit_rows = _build_electroweak_best_fit_rows(masses, refs, r2s)
            fit_table.value = pd.DataFrame(fit_rows)

            anchor_rows = _build_electroweak_anchor_rows(masses, refs, r2s)
            anchor_table.value = pd.DataFrame(anchor_rows)

            compare_rows = _build_electroweak_comparison_rows(masses, refs)
            compare_table.value = pd.DataFrame(compare_rows)

        def _update_electroweak_plots(results: dict[str, ChannelCorrelatorResult]) -> None:
            _update_electroweak_plots_generic(
                results,
                electroweak_channel_plots,
                electroweak_plots_spectrum,
                electroweak_plots_overlay_corr,
                electroweak_plots_overlay_meff,
            )

        def _update_electroweak_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = electroweak_mass_mode.value
            _update_electroweak_tables_generic(
                results,
                mode,
                electroweak_mass_table,
                electroweak_ratio_pane,
                electroweak_ratio_table,
                electroweak_fit_table,
                electroweak_anchor_table,
                electroweak_compare_table,
                electroweak_ref_table,
            )

        def _on_electroweak_mass_mode_change(event) -> None:
            if "electroweak_results" in state:
                _update_electroweak_tables(state["electroweak_results"], event.new)

        electroweak_mass_mode.param.watch(_on_electroweak_mass_mode_change, "value")

        def on_run_electroweak(_):
            def _compute(history):
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
                        refs=_extract_coupling_refs(electroweak_coupling_ref_table),
                    )
                )
                n_channels = len([r for r in results.values() if r.n_samples > 0])
                electroweak_status.object = (
                    f"**Complete:** {n_channels} electroweak channels computed."
                )

            _run_tab_computation(state, electroweak_status, "electroweak channels", _compute)

        # =====================================================================
        # Radial electroweak tab callbacks
        # =====================================================================

        def _update_radial_ew_plots(
            results: dict[str, ChannelCorrelatorResult],
            channel_plots_container: pn.Column,
            plots_spectrum: pn.pane.HoloViews,
            plots_overlay_corr: pn.pane.HoloViews,
            plots_overlay_meff: pn.pane.HoloViews,
        ) -> None:
            _update_electroweak_plots_generic(
                results,
                channel_plots_container,
                plots_spectrum,
                plots_overlay_corr,
                plots_overlay_meff,
            )

        def _update_radial_ew_tables(
            results: dict[str, ChannelCorrelatorResult],
            mass_table: pn.widgets.Tabulator,
            ratio_pane: pn.pane.Markdown,
            ratio_table: pn.widgets.Tabulator,
            fit_table: pn.widgets.Tabulator,
            anchor_table: pn.widgets.Tabulator,
            compare_table: pn.widgets.Tabulator,
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = radial_ew_mass_mode.value
            _update_electroweak_tables_generic(
                results,
                mode,
                mass_table,
                ratio_pane,
                ratio_table,
                fit_table,
                anchor_table,
                compare_table,
                radial_ew_ref_table,
            )

        def _on_radial_ew_mass_mode_change(event) -> None:
            if state.get("radial_ew_results_4d") is not None:
                _update_radial_ew_tables(
                    state["radial_ew_results_4d"],
                    radial_ew4d_mass_table,
                    radial_ew4d_ratio_pane,
                    radial_ew4d_ratio_table,
                    radial_ew4d_fit_table,
                    radial_ew4d_anchor_table,
                    radial_ew4d_compare_table,
                    event.new,
                )
            if state.get("radial_ew_results_3d") is not None:
                _update_radial_ew_tables(
                    state["radial_ew_results_3d"],
                    radial_ew3d_mass_table,
                    radial_ew3d_ratio_pane,
                    radial_ew3d_ratio_table,
                    radial_ew3d_fit_table,
                    radial_ew3d_anchor_table,
                    radial_ew3d_compare_table,
                    event.new,
                )

        radial_ew_mass_mode.param.watch(_on_radial_ew_mass_mode_change, "value")

        def on_run_radial_electroweak(_):
            def _compute(history):
                bundle = _compute_radial_electroweak_bundle(history, radial_ew_settings)
                state["radial_ew_results_4d"] = bundle.radial_4d.channel_results
                state["radial_ew_results_3d"] = (
                    bundle.radial_3d_avg.channel_results if bundle.radial_3d_avg else {}
                )
                state["radial_ew_results_3d_axes"] = {
                    axis: output.channel_results
                    for axis, output in (bundle.radial_3d_by_axis or {}).items()
                }

                results_4d = state["radial_ew_results_4d"] or {}
                _update_radial_ew_plots(
                    results_4d,
                    radial_ew4d_channel_plots,
                    radial_ew4d_plots_spectrum,
                    radial_ew4d_plots_overlay_corr,
                    radial_ew4d_plots_overlay_meff,
                )
                _update_radial_ew_tables(
                    results_4d,
                    radial_ew4d_mass_table,
                    radial_ew4d_ratio_pane,
                    radial_ew4d_ratio_table,
                    radial_ew4d_fit_table,
                    radial_ew4d_anchor_table,
                    radial_ew4d_compare_table,
                )

                results_3d = state["radial_ew_results_3d"] or {}
                _update_radial_ew_plots(
                    results_3d,
                    radial_ew3d_channel_plots,
                    radial_ew3d_plots_spectrum,
                    radial_ew3d_plots_overlay_corr,
                    radial_ew3d_plots_overlay_meff,
                )
                _update_radial_ew_tables(
                    results_3d,
                    radial_ew3d_mass_table,
                    radial_ew3d_ratio_pane,
                    radial_ew3d_ratio_table,
                    radial_ew3d_fit_table,
                    radial_ew3d_anchor_table,
                    radial_ew3d_compare_table,
                )

                couplings = _compute_coupling_constants(
                    history,
                    h_eff=float(radial_ew_settings.h_eff),
                    epsilon_d=radial_ew_settings.epsilon_d,
                    epsilon_c=radial_ew_settings.epsilon_c,
                )
                radial_ew_coupling_table.value = pd.DataFrame(
                    _build_coupling_rows(
                        couplings,
                        proxies=None,
                        include_strong=False,
                        refs=_extract_coupling_refs(radial_ew_coupling_ref_table),
                    )
                )

                n_channels = len([r for r in results_4d.values() if r.n_samples > 0])
                radial_ew_status.object = (
                    f"**Complete:** {n_channels} radial electroweak channels computed."
                )

            _run_tab_computation(state, radial_ew_status, "radial electroweak channels", _compute)

        def on_run_higgs(_):
            history = state.get("history")
            if history is None:
                higgs_status.object = "**Error:** Load a RunHistory first."
                return

            higgs_status.object = "**Computing Higgs field observables...**"
            try:
                # Build HiggsConfig from settings
                config = HiggsConfig(
                    mc_time_index=higgs_settings.mc_time_index,
                    h_eff=higgs_settings.h_eff,
                    mu_sq=higgs_settings.mu_sq,
                    lambda_higgs=higgs_settings.lambda_higgs,
                    alpha_gravity=higgs_settings.alpha_gravity,
                    warmup_fraction=higgs_settings.warmup_fraction,
                    compute_curvature=higgs_settings.compute_curvature,
                    compute_action=higgs_settings.compute_action,
                )

                # Compute observables
                observables = compute_higgs_observables(
                    history,
                    config=config,
                    scalar_field_source=higgs_settings.scalar_field_source,
                )

                state["higgs_observables"] = observables

                # Get positions for plotting (use same mc_frame as observables)
                mc_frame = config.mc_time_index if config.mc_time_index is not None else history.n_recorded - 1
                mc_frame = min(mc_frame, history.n_recorded - 1)
                positions = history.x_final[mc_frame].detach().cpu().numpy()

                # Build all plots using the plotting module
                plots = build_all_higgs_plots(
                    observables,
                    positions=positions,
                    metric_component=(
                        higgs_settings.metric_component_x,
                        higgs_settings.metric_component_y,
                    ),
                )

                # Update plot panes
                higgs_metric_heatmap.object = plots["metric_tensor_heatmap"]
                higgs_centroid_field.object = plots["centroid_vector_field"]
                higgs_ricci_histogram.object = plots["ricci_scalar_distribution"]
                higgs_geodesic_scatter.object = plots["geodesic_distance_scatter"]
                higgs_volume_curvature.object = plots["volume_vs_curvature_scatter"]
                higgs_scalar_field_map.object = plots["scalar_field_map"]
                higgs_eigenvalue_dist.object = plots["metric_eigenvalues_distribution"]

                # Update action summary
                gravity_term_str = f"{observables.gravity_term:.6f}" if observables.gravity_term is not None else "N/A"
                action_md = (
                    "**Higgs Action Summary**\n\n"
                    f"- **Kinetic Term:** {observables.kinetic_term:.6f}\n"
                    f"- **Potential Term:** {observables.potential_term:.6f}\n"
                    f"- **Gravity Term:** {gravity_term_str}\n"
                    f"- **Total Action:** {observables.total_action:.6f}\n\n"
                    "**Geometry Statistics**\n"
                    f"- **N Walkers:** {observables.n_walkers}\n"
                    f"- **N Edges:** {observables.n_edges}\n"
                    f"- **Spatial Dims:** {observables.spatial_dims}\n"
                    f"- **Volume Variance:** {observables.volume_variance:.6f}\n"
                )
                higgs_action_summary.object = action_md

                higgs_status.object = (
                    f"**Complete:** Computed observables for {observables.n_walkers} walkers, "
                    f"{observables.n_edges} edges. Total action: {observables.total_action:.6f}"
                )
            except IndexError as e:
                higgs_status.object = (
                    f"**Error (IndexError):** {e}. "
                    "This usually indicates dimension mismatch or invalid indexing. "
                    "Check spatial dimensions and data shapes."
                )
                import traceback
                traceback.print_exc()
            except ValueError as e:
                higgs_status.object = f"**Error (ValueError):** {e}"
                import traceback
                traceback.print_exc()
            except Exception as e:
                higgs_status.object = f"**Error:** {e}"
                import traceback
                traceback.print_exc()

        def on_run_dynamics(_):
            history = state.get("history")
            if history is None:
                dynamics_status.object = "**Error:** Load a RunHistory first."
                return

            dynamics_status.object = "**Computing dynamics plots...**"
            try:
                t = dynamics_mc_slider.value
                t = min(t, history.n_recorded - 2)  # info fields are [n_recorded-1]
                t = max(t, 0)

                alive = _to_numpy(history.alive_mask[t]).astype(bool)

                # Viscous force modulus
                if history.force_viscous is not None:
                    fv = _to_numpy(history.force_viscous[t])  # [N, d]
                    force_mod = np.linalg.norm(fv[alive], axis=1)
                else:
                    force_mod = np.zeros(alive.sum())

                # Cloning scores
                cs = _to_numpy(history.cloning_scores[t])[alive]

                # Fitness
                fit = _to_numpy(history.fitness[t])[alive]

                # Momentum modulus (||v||)
                v = _to_numpy(history.v_final[t + 1])  # v_final is [n_recorded, N, d]
                mom_mod = np.linalg.norm(v[alive], axis=1)

                # === 1. Scatter: |F_visc| vs cloning score ===
                pos_mask = cs >= 0
                neg_mask = ~pos_mask

                scatter_pos = hv.Points(
                    pd.DataFrame({
                        "cloning_score": cs[pos_mask],
                        "force_modulus": force_mod[pos_mask],
                    }),
                    kdims=["cloning_score", "force_modulus"],
                ).opts(
                    color="cloning_score", cmap="Greens",
                    colorbar=False, alpha=0.7, size=5,
                )

                scatter_neg = hv.Points(
                    pd.DataFrame({
                        "cloning_score": cs[neg_mask],
                        "force_modulus": force_mod[neg_mask],
                    }),
                    kdims=["cloning_score", "force_modulus"],
                ).opts(
                    color="cloning_score", cmap="Reds_r",
                    colorbar=False, alpha=0.7, size=5,
                )

                scatter_plot = (scatter_neg * scatter_pos).opts(
                    xlabel="Cloning Score",
                    ylabel="|Viscous Force|",
                    title=f"|Viscous Force| vs Cloning Score (MC step {t})",
                    width=700,
                    height=450,
                    logx=True,
                    logy=True if force_mod.max() > 0 else False,
                    bgcolor="#1a1a1a",
                )
                dynamics_scatter_pane.object = scatter_plot

                # === 2. Distributions ===
                n_bins = 50

                dynamics_fitness_hist.object = hv.Histogram(
                    np.histogram(fit, bins=n_bins)
                ).opts(
                    color="#4c78a8", alpha=0.8, xlabel="Fitness", ylabel="Count",
                    title="Fitness Distribution", width=500, height=300,
                )

                cs_pos = cs[cs > 0]
                if len(cs_pos) > 1:
                    cs_bins = np.geomspace(cs_pos.min(), cs_pos.max(), n_bins + 1)
                    cs_hist_data = np.histogram(cs_pos, bins=cs_bins)
                else:
                    cs_hist_data = np.histogram(cs[cs > 0] if (cs > 0).any() else cs, bins=n_bins)
                dynamics_cloning_hist.object = hv.Histogram(cs_hist_data).opts(
                    color="#f58518", alpha=0.8, xlabel="Cloning Score", ylabel="Count",
                    title="Cloning Score Distribution", width=500, height=300,
                    logx=True,
                )

                dynamics_force_hist.object = hv.Histogram(
                    np.histogram(force_mod[force_mod > 0], bins=n_bins)
                    if (force_mod > 0).any()
                    else np.histogram(force_mod, bins=n_bins)
                ).opts(
                    color="#e45756", alpha=0.8, xlabel="|Viscous Force|", ylabel="Count",
                    title="Viscous Force Modulus Distribution", width=500, height=300,
                )

                dynamics_momentum_hist.object = hv.Histogram(
                    np.histogram(mom_mod, bins=n_bins)
                ).opts(
                    color="#72b7b2", alpha=0.8, xlabel="|Momentum|", ylabel="Count",
                    title="Momentum Modulus Distribution", width=500, height=300,
                )

                n_alive = alive.sum()

                # === 3. Dirac spectrum analysis ===
                try:
                    _color_thresh = (
                        dirac_threshold_value.value
                        if dirac_threshold_mode.value == "manual"
                        else "median"
                    )
                    dirac_config = DiracSpectrumConfig(
                        mc_time_index=t,
                        time_average=dirac_time_avg_checkbox.value,
                        warmup_fraction=dirac_warmup_slider.value,
                        max_avg_frames=dirac_max_frames.value,
                        color_threshold=_color_thresh,
                    )
                    dirac_result = compute_dirac_spectrum(history, dirac_config)
                    dirac_plots = build_all_dirac_plots(dirac_result)
                    dirac_full_spectrum.object = dirac_plots["full_spectrum"]
                    _sector_plots = dirac_plots["sector_spectra"]
                    dirac_sector_up.object = _sector_plots.get("up_quark")
                    dirac_sector_down.object = _sector_plots.get("down_quark")
                    dirac_sector_nu.object = _sector_plots.get("neutrino")
                    dirac_sector_lep.object = _sector_plots.get("charged_lepton")
                    dirac_walker_classification.object = dirac_plots["walker_classification"]
                    dirac_mass_hierarchy.object = dirac_plots["mass_hierarchy"]
                    dirac_chiral_density.object = dirac_plots["chiral_density"]
                    dirac_generation_ratios.object = dirac_plots["generation_ratios"]

                    # Fermion comparison tables
                    comp_rows, best_scale = build_fermion_comparison(dirac_result)
                    if comp_rows:
                        dirac_comparison_table.value = pd.DataFrame(comp_rows)
                    ratio_rows = build_fermion_ratio_comparison(dirac_result)
                    dirac_ratio_comparison_pane.object = dirac_plots["fermion_ratio_comparison"]

                    # Summary markdown
                    sector_counts = {
                        name: spec.n_walkers
                        for name, spec in dirac_result.sectors.items()
                    }
                    summary_lines = [
                        f"**Best-fit scale:** {best_scale:.6g} GeV/σ" if best_scale else "**Best-fit scale:** N/A",
                        f"**Chiral condensate:** ⟨ψ̄ψ⟩ ≈ {dirac_result.chiral_condensate:.4f}",
                        f"**Sector walkers:** " + ", ".join(
                            f"{k}: {v}" for k, v in sector_counts.items()
                        ),
                    ]
                    dirac_summary_pane.object = "  \n".join(summary_lines)
                except Exception as dirac_err:
                    import traceback
                    traceback.print_exc()
                    dirac_full_spectrum.object = hv.Text(
                        0, 0, f"Dirac error: {dirac_err}"
                    )

                dynamics_status.object = (
                    f"**Complete:** Dynamics for MC step {t}, "
                    f"{n_alive} alive walkers."
                )
            except Exception as e:
                dynamics_status.object = f"**Error:** {e}"
                import traceback
                traceback.print_exc()

        def on_run_quantum_gravity(_):
            history = state.get("history")
            if history is None:
                qg_status.object = "**Error:** Load a RunHistory first."
                return

            qg_status.object = "**Computing quantum gravity observables...**"
            try:
                # Build config from settings
                config = QuantumGravityConfig(
                    mc_time_index=qg_settings.mc_time_index,
                    warmup_fraction=qg_settings.warmup_fraction,
                    analysis_dims=(
                        qg_settings.analysis_dim_0,
                        qg_settings.analysis_dim_1,
                        qg_settings.analysis_dim_2,
                    ),
                    use_metric_correction=qg_settings.use_metric_correction,
                    diffusion_time_steps=qg_settings.diffusion_time_steps,
                    max_diffusion_time=qg_settings.max_diffusion_time,
                    n_radial_bins=qg_settings.n_radial_bins,
                    light_speed=qg_settings.light_speed,
                    euclidean_time_dim=qg_settings.euclidean_time_dim,
                    euclidean_time_bins=qg_settings.euclidean_time_bins,
                    planck_length=qg_settings.planck_length,
                    compute_all=True,
                )

                # Compute single-frame observables
                observables = compute_quantum_gravity_observables(history, config)
                state["quantum_gravity_observables"] = observables

                # Get positions
                mc_frame = config.mc_time_index if config.mc_time_index is not None else history.n_recorded - 1
                mc_frame = min(mc_frame, history.n_recorded - 1)
                positions = history.x_final[mc_frame].detach().cpu().numpy()

                # Validate and slice positions if needed
                analysis_dims_input = config.analysis_dims or (0, 1, 2)
                analysis_dims = [int(d) for d in analysis_dims_input]

                # Check for invalid dimensions
                invalid_dims = [d for d in analysis_dims if d < 0 or d >= positions.shape[1]]
                if invalid_dims:
                    qg_status.object = (
                        f"**Error:** analysis_dims {invalid_dims} invalid for "
                        f"{positions.shape[1]}D data (valid range: 0..{positions.shape[1]-1})"
                    )
                    return

                # Filter to valid unique dimensions
                analysis_dims = [d for d in analysis_dims if 0 <= d < positions.shape[1]]
                if analysis_dims:
                    # Defensive: ensure positions is 2D
                    if positions.ndim != 2:
                        qg_status.object = (
                            f"**Error:** Expected 2D positions, got shape {positions.shape}"
                        )
                        return

                    positions = positions[:, analysis_dims]

                # Build all plots
                plots = build_all_gravity_plots(observables, positions)

                # Update plot panes (20 total plots for 10 analyses)
                qg_summary_panel.object = plots["summary_panel"]

                # 1. Regge Calculus
                qg_regge_action_density.object = plots["regge_action_density"]
                qg_deficit_angle_dist.object = plots["deficit_angle_dist"]

                # 2. Einstein-Hilbert
                qg_ricci_landscape.object = plots["ricci_landscape"]
                qg_action_decomposition.object = plots["action_decomposition"]

                # 3. ADM Energy
                qg_adm_summary.object = plots["adm_summary"]
                qg_energy_density_dist.object = plots["energy_density_dist"]

                # 4. Spectral Dimension
                qg_spectral_dim_curve.object = plots["spectral_dimension_curve"]
                qg_heat_kernel_trace.object = plots["heat_kernel_trace"]

                # 5. Hausdorff Dimension
                qg_hausdorff_scaling.object = plots["hausdorff_scaling"]
                qg_local_hausdorff_map.object = plots["local_hausdorff_map"]

                # 6. Causal Structure
                qg_causal_diamond.object = plots["causal_diamond"]
                qg_causal_violations.object = plots["causal_violations"]

                # 7. Holographic Entropy
                qg_holographic_summary.object = plots["holographic_summary"]

                # 8. Spin Network
                qg_spin_distribution.object = plots["spin_distribution"]
                qg_spin_network_summary.object = plots["spin_network_summary"]

                # 9. Raychaudhuri Expansion
                qg_expansion_field.object = plots["expansion_field"]
                qg_convergence_regions.object = plots["convergence_regions"]

                # 10. Geodesic Deviation
                qg_tidal_eigenvalues.object = plots["tidal_eigenvalues"]
                qg_tidal_summary.object = plots["tidal_summary"]

                status_msg = (
                    f"**Complete:** Computed {observables.n_walkers} walkers, "
                    f"{observables.n_edges} edges. "
                    f"Spectral dim (Planck): {observables.spectral_dimension_planck:.2f}, "
                    f"Hausdorff dim: {observables.hausdorff_dimension:.2f}"
                )

                # Time evolution (4D spacetime block analysis)
                if qg_settings.compute_time_evolution:
                    qg_status.object = "**Computing quantum gravity time evolution...**"

                    time_series = compute_quantum_gravity_time_evolution(
                        history,
                        config,
                        frame_stride=qg_settings.frame_stride,
                    )

                    state["quantum_gravity_time_series"] = time_series

                    # Build time evolution plots
                    time_plots = build_all_quantum_gravity_time_series_plots(time_series)

                    # Update time evolution plot panes
                    qg_time_summary.object = time_plots["time_series_summary"]
                    qg_regge_evolution.object = time_plots["regge_action_evolution"]
                    qg_adm_evolution.object = time_plots["adm_mass_evolution"]
                    qg_spectral_evolution.object = time_plots["spectral_dimension_evolution"]
                    qg_hausdorff_evolution.object = time_plots["hausdorff_dimension_evolution"]
                    qg_holographic_evolution.object = time_plots["holographic_entropy_evolution"]
                    qg_raychaudhuri_evolution.object = time_plots["raychaudhuri_expansion_evolution"]
                    qg_causal_evolution.object = time_plots["causal_structure_evolution"]
                    qg_spin_evolution.object = time_plots["spin_network_evolution"]
                    qg_tidal_evolution.object = time_plots["tidal_strength_evolution"]

                    status_msg += f" | Time evolution: {time_series.n_frames} frames analyzed."

                qg_status.object = status_msg

            except IndexError as e:
                qg_status.object = (
                    f"**Error (IndexError):** {e}. "
                    "This usually indicates dimension mismatch or invalid indexing. "
                    "Check spatial dimensions, analysis_dims, and data shapes."
                )
                import traceback
                traceback.print_exc()
            except ValueError as e:
                qg_status.object = f"**Error (ValueError):** {e}"
                import traceback
                traceback.print_exc()
            except Exception as e:
                qg_status.object = f"**Error:** {e}"
                import traceback
                traceback.print_exc()

        browse_button.on_click(_on_browse_clicked)
        load_button.on_click(on_load_clicked)
        save_button.on_click(on_save_clicked)
        gas_config.add_completion_callback(on_simulation_complete)
        gas_config.param.watch(on_bounds_change, "bounds_extent")
        run_analysis_button.on_click(on_run_analysis)
        run_analysis_button_main.on_click(on_run_analysis)
        fractal_set_run_button.on_click(on_run_fractal_set)
        einstein_run_button.on_click(on_run_einstein_test)
        channels_run_button.on_click(on_run_channels)
        radial_run_button.on_click(on_run_radial_channels)
        anisotropic_edge_run_button.on_click(on_run_anisotropic_edge_channels)
        radial_ew_run_button.on_click(on_run_radial_electroweak)
        electroweak_run_button.on_click(on_run_electroweak)
        higgs_run_button.on_click(on_run_higgs)
        qg_run_button.on_click(on_run_quantum_gravity)
        dynamics_run_button.on_click(on_run_dynamics)
        dynamics_mc_slider.param.watch(
            lambda _: on_run_dynamics(None), "value_throttled",
        )

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
                save_button,
                save_status,
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
                            save_button,
                            save_status,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    ("Simulation", gas_config.panel()),
                    ("Visualization", visualization_controls),
                    ("Analysis: Core", analysis_core),
                    ("Analysis: Local Fields", analysis_local),
                    ("Analysis: String Tension", analysis_string),
                    ("Analysis: Output", analysis_output),
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

            fractal_set_note = pn.pane.Alert(
                """
**Fractal Set Protocol**
- IG measurements: cross-boundary companion counts from `companions_distance` and
  `companions_clone` (`S_dist`, `S_fit`, `S_total`).
- CST measurement: number of ancestral lineages with descendants on both sides of
  the same partition (`Area_CST`).
- Partition generators:
  spatial boundaries (hyperplane/spherical/median),
  spectral graph cuts from companion graphs, and
  random partition baseline.
- Geometry correction (optional):
  Riemannian-kernel edge lengths and volume-weighted CST area
  (`S_*_geom`, `Area_CST_geom`).
                """,
                alert_type="info",
                sizing_mode="stretch_width",
            )

            fractal_set_tab = pn.Column(
                fractal_set_status,
                fractal_set_note,
                pn.Row(fractal_set_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Fractal Set Settings", fractal_set_settings_panel),
                    active=[0],
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                fractal_set_summary,
                pn.pane.Markdown("### Linear Fits"),
                fractal_set_regression_table,
                pn.pane.Markdown("### Compare Vs Random Baseline"),
                fractal_set_baseline_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### S_dist vs Area_CST"),
                fractal_set_plot_dist,
                pn.pane.Markdown("### S_fit vs Area_CST"),
                fractal_set_plot_fit,
                pn.pane.Markdown("### S_total vs Area_CST"),
                fractal_set_plot_total,
                pn.layout.Divider(),
                pn.pane.Markdown("### S_dist_geom vs Area_CST_geom"),
                fractal_set_plot_dist_geom,
                pn.pane.Markdown("### S_fit_geom vs Area_CST_geom"),
                fractal_set_plot_fit_geom,
                pn.pane.Markdown("### S_total_geom vs Area_CST_geom"),
                fractal_set_plot_total_geom,
                pn.layout.Divider(),
                pn.pane.Markdown("### Per-Frame Means"),
                fractal_set_frame_table,
                pn.Accordion(
                    ("Raw Boundary Samples", fractal_set_points_table),
                    active=[],
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("## Einstein Equation Test: G_uv + \u039b g_uv = 8\u03c0 G_N T_uv"),
                einstein_status,
                pn.Row(einstein_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Einstein Test Settings", einstein_settings_panel),
                    active=[],
                    sizing_mode="stretch_width",
                ),
                einstein_summary,
                pn.pane.Markdown("### Scalar Test: R vs \u03c1"),
                einstein_scalar_plot,
                pn.pane.Markdown("### Scalar Test: R vs log\u2081\u2080(\u03c1)"),
                einstein_scalar_log_plot,
                pn.pane.Markdown("### Tensor Component R\u00b2"),
                einstein_tensor_table,
                pn.pane.Markdown("### Curvature Distribution"),
                einstein_curvature_hist,
                pn.pane.Markdown("### Residual Map"),
                einstein_residual_map,
                pn.pane.Markdown("### Cross-Check: Full Ricci vs Proxy"),
                einstein_crosscheck_plot,
                pn.pane.Markdown("### Bulk vs Boundary"),
                einstein_bulk_boundary,
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
                    (
                        "Reference Anchors",
                        pn.Column(
                            channel_glueball_ref_input,
                            pn.pane.Markdown("### Observed Mass Anchors (GeV)"),
                            channel_ref_table,
                        ),
                    ),
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
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                pn.pane.Markdown(
                    "_Side-by-side view: correlator C(t) with exponential fit (left) and effective mass "
                    "m_eff(t) with plateau and best window region (right). Error bars shown when bootstrap "
                    "estimation is enabled._",
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
                sizing_mode="stretch_both",
            )

            radial_tab_note = pn.pane.Alert(
                """**Geometry-Aware Strong Force:** Choose **time_axis=mc** (default) to fit
correlator decay across Monte Carlo time using geometry-weighted operators.
Choose **time_axis=radial** to recover single-snapshot radial screening
analysis (4D radial + optional 3D drop-axis averages with r^p correction).""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            radial_4d_section = pn.Column(
                pn.pane.Markdown("### Primary Geometry-Aware Strong Force Output"),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                radial4d_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                radial4d_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Mass Spectrum"),
                radial4d_plots_spectrum,
                pn.layout.Divider(),
                pn.pane.Markdown("### Extracted Masses"),
                radial4d_mass_table,
                radial4d_ratio_pane,
                pn.pane.Markdown("### Best-Fit Scales"),
                radial4d_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                radial4d_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                pn.pane.Markdown(
                    "_Side-by-side view: correlator C(·) (left) and effective mass m_eff(·) (right). "
                    "Error bars shown when bootstrap estimation is enabled._"
                ),
                radial4d_plateau_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("### Window Heatmaps"),
                pn.Row(
                    radial_heatmap_color_metric,
                    radial_heatmap_alpha_metric,
                    sizing_mode="stretch_width",
                ),
                radial4d_heatmap_plots,
                sizing_mode="stretch_both",
            )

            radial_3d_section = pn.Column(
                pn.pane.Markdown("### 3D Drop-Axis Average (Radial Mode)"),
                pn.pane.Markdown("### 3D Per-Axis Summary (Avg + up to 3 axes)"),
                radial3d_axis_grid,
                pn.layout.Divider(),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                radial3d_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                radial3d_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Mass Spectrum"),
                radial3d_plots_spectrum,
                pn.layout.Divider(),
                pn.pane.Markdown("### Extracted Masses"),
                radial3d_mass_table,
                radial3d_ratio_pane,
                pn.pane.Markdown("### Best-Fit Scales"),
                radial3d_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                radial3d_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                pn.pane.Markdown(
                    "_Side-by-side view: correlator C(·) (left) and effective mass m_eff(·) (right). "
                    "Error bars shown when bootstrap estimation is enabled._"
                ),
                radial3d_plateau_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("### Window Heatmaps"),
                radial3d_heatmap_plots,
                sizing_mode="stretch_both",
            )

            radial_ew_note = pn.pane.Alert(
                """**Radial Electroweak:** Electroweak proxy correlators computed from a single
Monte Carlo slice and binned by radial distance (4D and 3D drop-axis averages).
Use **MC time slice** to select the snapshot.""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            radial_ew_4d_section = pn.Column(
                pn.pane.Markdown("### 4D Radial Electroweak"),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                radial_ew4d_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                radial_ew4d_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Electroweak Mass Spectrum"),
                radial_ew4d_plots_spectrum,
                pn.pane.Markdown("### Extracted Masses"),
                radial_ew4d_mass_table,
                radial_ew4d_ratio_pane,
                pn.pane.Markdown("### Electroweak Ratios"),
                radial_ew4d_ratio_table,
                pn.pane.Markdown("### Best-Fit Scales"),
                radial_ew4d_fit_table,
                pn.pane.Markdown("### Measured vs Observed"),
                pn.pane.Markdown(
                    "_Best-fit scale applied to all electroweak channels; "
                    "error shows percent deviation from observed masses._"
                ),
                radial_ew4d_compare_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                radial_ew4d_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                pn.pane.Markdown(
                    "_Side-by-side view: correlator C(r) (left) and effective mass m_eff(r) (right). "
                    "Error bars shown when bootstrap estimation is enabled._"
                ),
                radial_ew4d_channel_plots,
                sizing_mode="stretch_both",
            )

            radial_ew_3d_section = pn.Column(
                pn.pane.Markdown("### 3D Drop-Axis Average"),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                radial_ew3d_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                radial_ew3d_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Electroweak Mass Spectrum"),
                radial_ew3d_plots_spectrum,
                pn.pane.Markdown("### Extracted Masses"),
                radial_ew3d_mass_table,
                radial_ew3d_ratio_pane,
                pn.pane.Markdown("### Electroweak Ratios"),
                radial_ew3d_ratio_table,
                pn.pane.Markdown("### Best-Fit Scales"),
                radial_ew3d_fit_table,
                pn.pane.Markdown("### Measured vs Observed"),
                pn.pane.Markdown(
                    "_Best-fit scale applied to all electroweak channels; "
                    "error shows percent deviation from observed masses._"
                ),
                radial_ew3d_compare_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                radial_ew3d_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                pn.pane.Markdown(
                    "_Side-by-side view: correlator C(r) (left) and effective mass m_eff(r) (right). "
                    "Error bars shown when bootstrap estimation is enabled._"
                ),
                radial_ew3d_channel_plots,
                sizing_mode="stretch_both",
            )

            radial_tab = pn.Column(
                radial_status,
                radial_tab_note,
                pn.Row(radial_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Radial Settings", radial_settings_panel),
                    (
                        "Reference Anchors",
                        pn.Column(
                            radial_glueball_ref_input,
                            pn.pane.Markdown("### Observed Mass Anchors (GeV)"),
                            radial_ref_table,
                        ),
                    ),
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown("### Mass Display Mode"),
                radial_mass_mode,
                pn.layout.Divider(),
                pn.Tabs(
                    ("4D Radial", radial_4d_section),
                    ("3D Avg", radial_3d_section),
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            )

            anisotropic_edge_note = pn.pane.Alert(
                """**Anisotropic Edge Channels:** Computes MC-time correlators from direct
recorded Delaunay neighbors only (no tessellation recomputation). Local edge
observables are projected onto directional components to preserve anisotropy
under symmetry breaking. Nucleon supports two triplet modes:
direct-neighbor triplets or (distance companion, clone companion).""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            anisotropic_edge_tab = pn.Column(
                anisotropic_edge_status,
                anisotropic_edge_note,
                pn.Row(anisotropic_edge_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Anisotropic Edge Settings", anisotropic_edge_settings_panel),
                    (
                        "Reference Anchors",
                        pn.Column(
                            anisotropic_edge_glueball_ref_input,
                            pn.pane.Markdown("### Observed Mass Anchors (GeV)"),
                            anisotropic_edge_ref_table,
                        ),
                    ),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                anisotropic_edge_summary,
                pn.pane.Markdown("### Mass Display Mode"),
                anisotropic_edge_mass_mode,
                pn.layout.Divider(),
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                anisotropic_edge_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                anisotropic_edge_plots_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("### Mass Spectrum"),
                anisotropic_edge_plots_spectrum,
                pn.layout.Divider(),
                pn.pane.Markdown("### Extracted Masses"),
                anisotropic_edge_mass_table,
                anisotropic_edge_ratio_pane,
                pn.pane.Markdown("### Best-Fit Scales"),
                anisotropic_edge_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                anisotropic_edge_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                anisotropic_edge_plateau_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("### Window Heatmaps"),
                pn.Row(
                    anisotropic_edge_heatmap_color_metric,
                    anisotropic_edge_heatmap_alpha_metric,
                    sizing_mode="stretch_width",
                ),
                anisotropic_edge_heatmap_plots,
                sizing_mode="stretch_both",
            )

            radial_ew_tab = pn.Column(
                radial_ew_status,
                radial_ew_note,
                pn.Row(radial_ew_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Radial Electroweak Settings", radial_ew_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                radial_ew_summary,
                pn.pane.Markdown("### Electroweak Couplings"),
                radial_ew_coupling_table,
                pn.pane.Markdown("### Electroweak Coupling References"),
                pn.pane.Markdown(
                    "_Set observed values to compute error percentages for couplings._"
                ),
                radial_ew_coupling_ref_table,
                pn.pane.Markdown("### Gauge Phase Histograms"),
                radial_ew_phase_plot,
                pn.layout.Divider(),
                pn.pane.Markdown("### Electroweak Reference Masses (GeV)"),
                pn.pane.Markdown(
                    "_Edit the table below to set observed masses for calibration._"
                ),
                radial_ew_ref_table,
                pn.pane.Markdown("### Mass Display Mode"),
                radial_ew_mass_mode,
                pn.layout.Divider(),
                pn.Tabs(
                    ("4D Radial", radial_ew_4d_section),
                    ("3D Avg", radial_ew_3d_section),
                    sizing_mode="stretch_both",
                ),
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
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                pn.pane.Markdown(
                    "_Side-by-side view: correlator C(t) with exponential fit (left) "
                    "and effective mass m_eff(t) with plateau (right). Error bars shown when "
                    "bootstrap estimation is enabled._"
                ),
                electroweak_channel_plots,
                sizing_mode="stretch_both",
            )

            higgs_tab = pn.Column(
                higgs_status,
                pn.Row(higgs_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Higgs Settings", higgs_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown(
                    "_Higgs field observables computed from emergent manifold geometry. "
                    "Voronoi cell volumes encode spacetime density, neighbor covariance defines "
                    "the metric tensor g_μν, and centroid displacement (Lloyd vectors) acts as "
                    "gauge field/drift._"
                ),
                pn.layout.Divider(),
                higgs_action_summary,
                pn.layout.Divider(),
                pn.pane.Markdown("### Emergent Metric Tensor g_μν"),
                pn.pane.Markdown(
                    "_Heatmap shows selected component of the metric tensor computed from "
                    "neighbor covariance. Use settings to select which (μ,ν) component to visualize._"
                ),
                higgs_metric_heatmap,
                pn.layout.Divider(),
                pn.pane.Markdown("### Centroid Displacement Field (Lloyd Vectors)"),
                pn.pane.Markdown(
                    "_Vector field showing displacement from each walker to its neighbor centroid. "
                    "Acts as a gauge field/drift in the emergent geometry._"
                ),
                higgs_centroid_field,
                pn.layout.Divider(),
                pn.pane.Markdown("### Ricci Scalar Curvature Distribution"),
                pn.pane.Markdown(
                    "_Histogram of Ricci scalar values estimated from volume distortion and "
                    "Raychaudhuri expansion. Positive values indicate clustering, negative values "
                    "indicate expansion._"
                ),
                higgs_ricci_histogram,
                pn.layout.Divider(),
                pn.pane.Markdown("### Geodesic vs Euclidean Distances"),
                pn.pane.Markdown(
                    "_Scatter plot comparing Euclidean distances to geodesic distances computed "
                    "using the emergent metric tensor. Deviations indicate anisotropic geometry._"
                ),
                higgs_geodesic_scatter,
                pn.layout.Divider(),
                pn.pane.Markdown("### Cell Volume vs Curvature"),
                pn.pane.Markdown(
                    "_Relationship between Voronoi cell volume and local curvature. "
                    "Shows how spacetime density relates to curvature._"
                ),
                higgs_volume_curvature,
                pn.layout.Divider(),
                pn.pane.Markdown("### Scalar Field Configuration φ"),
                pn.pane.Markdown(
                    "_Spatial distribution of the Higgs field values. Source field selected in settings "
                    "(fitness, reward, or radius)._"
                ),
                higgs_scalar_field_map,
                pn.layout.Divider(),
                pn.pane.Markdown("### Metric Eigenvalue Distribution"),
                pn.pane.Markdown(
                    "_Histogram of metric tensor eigenvalues. Measures anisotropy in the emergent "
                    "geometry. Uniform eigenvalues → isotropic, spread eigenvalues → anisotropic._"
                ),
                higgs_eigenvalue_dist,
                sizing_mode="stretch_both",
            )

            dynamics_tab = pn.Column(
                dynamics_status,
                pn.Row(dynamics_run_button, sizing_mode="stretch_width"),
                dynamics_mc_slider,
                pn.layout.Divider(),
                pn.pane.Markdown("### |Viscous Force| vs Cloning Score"),
                dynamics_scatter_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("### Distributions"),
                pn.Row(
                    dynamics_fitness_hist,
                    dynamics_cloning_hist,
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    dynamics_force_hist,
                    dynamics_momentum_hist,
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("### Dirac Spectrum (Antisymmetric Kernel SVD)"),
                pn.Row(
                    dirac_time_avg_checkbox,
                    dirac_warmup_slider,
                    dirac_max_frames,
                    dirac_threshold_mode,
                    dirac_threshold_value,
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    dirac_full_spectrum,
                    dirac_walker_classification,
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("### Sector-Projected Spectra"),
                pn.GridBox(
                    dirac_sector_up, dirac_sector_down,
                    dirac_sector_nu, dirac_sector_lep,
                    ncols=2,
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("### Mass Hierarchy & Chiral Condensate"),
                pn.Row(
                    dirac_mass_hierarchy,
                    dirac_chiral_density,
                    sizing_mode="stretch_width",
                ),
                dirac_generation_ratios,
                pn.layout.Divider(),
                pn.pane.Markdown("### Fermion Mass Comparison (Extracted vs PDG)"),
                pn.pane.Markdown(
                    "_Best-fit scale maps algorithmic singular values to physical masses (GeV). "
                    "Error shows percent deviation from PDG values._"
                ),
                dirac_summary_pane,
                dirac_comparison_table,
                pn.pane.Markdown("### Inter-Generation Mass Ratios (Measured vs Observed)"),
                dirac_ratio_comparison_pane,
                sizing_mode="stretch_both",
            )

            quantum_gravity_tab = pn.Column(
                qg_status,
                pn.Row(qg_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Quantum Gravity Settings", qg_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown(
                    "_Quantum Gravity analyses using scutoid geometry and Voronoi tessellation. "
                    "Reproduces 10 famous quantum gravity signatures from the emergent spacetime._"
                ),
                pn.layout.Divider(),

                # Summary statistics
                pn.pane.Markdown("### Overall Summary"),
                qg_summary_panel,

                pn.layout.Divider(),
                pn.pane.Markdown("### 1. Regge Calculus (Deficit Angle Gravity)"),
                pn.pane.Markdown(
                    "_First practical approach to numerical GR (Tullio Regge, 1961). "
                    "Discretizes general relativity using deficit angles around edges._"
                ),
                qg_regge_action_density,
                qg_deficit_angle_dist,

                pn.layout.Divider(),
                pn.pane.Markdown("### 2. Einstein-Hilbert Action (Continuous Limit)"),
                pn.pane.Markdown(
                    "_The fundamental action of general relativity: S = ∫ R √g d⁴x. "
                    "Starting point of all gravitational theories._"
                ),
                qg_ricci_landscape,
                qg_action_decomposition,

                pn.layout.Divider(),
                pn.pane.Markdown("### 3. ADM Energy (Hamiltonian Formalism)"),
                pn.pane.Markdown(
                    "_Arnowitt-Deser-Misner mass from spatial hypersurface. "
                    "Canonical formulation of general relativity._"
                ),
                qg_adm_summary,
                qg_energy_density_dist,

                pn.layout.Divider(),
                pn.pane.Markdown("### 4. Spectral Dimension (Dimension Reduction)"),
                pn.pane.Markdown(
                    "_Predicts dimension reduction at Planck scale (Lauscher-Reuter, Ambjørn-Jurkiewicz-Loll CDT). "
                    "Key signature of asymptotic safety and causal dynamical triangulations._"
                ),
                qg_spectral_dim_curve,
                qg_heat_kernel_trace,

                pn.layout.Divider(),
                pn.pane.Markdown("### 5. Hausdorff Dimension (Fractal Geometry)"),
                pn.pane.Markdown(
                    "_Measures intrinsic manifold dimensionality from volume scaling N(r) ~ r^{d_H}. "
                    "Universal tool for fractal spacetimes._"
                ),
                qg_hausdorff_scaling,
                qg_local_hausdorff_map,

                pn.layout.Divider(),
                pn.pane.Markdown("### 6. Causal Set Structure (Discrete Spacetime)"),
                pn.pane.Markdown(
                    "_Rafael Sorkin's approach to quantum gravity. Partially ordered set of events "
                    "with causal relations (spacelike/timelike edges)._"
                ),
                qg_causal_diamond,
                qg_causal_violations,

                pn.layout.Divider(),
                pn.pane.Markdown("### 7. Holographic Entropy (AdS/CFT & Black Holes)"),
                pn.pane.Markdown(
                    "_Bekenstein-Hawking formula S = A/(4G ℏ). Holographic principle: "
                    "entropy proportional to boundary area, not volume._"
                ),
                qg_holographic_summary,

                pn.layout.Divider(),
                pn.pane.Markdown("### 8. Spin Network States (Loop Quantum Gravity)"),
                pn.pane.Markdown(
                    "_Ashtekar-Rovelli-Smolin formalism. Graph-based quantum geometry "
                    "with SU(2) labels on edges (quantized areas) and quantized volumes at vertices._"
                ),
                qg_spin_distribution,
                qg_spin_network_summary,

                pn.layout.Divider(),
                pn.pane.Markdown("### 9. Raychaudhuri Expansion (Singularity Theorem)"),
                pn.pane.Markdown(
                    "_Cornerstone of Hawking-Penrose singularity theorems. "
                    "Volume expansion rate θ = (1/V) dV/dt predicts singularities when θ → -∞._"
                ),
                qg_expansion_field,
                qg_convergence_regions,

                pn.layout.Divider(),
                pn.pane.Markdown("### 10. Geodesic Deviation (Tidal Forces)"),
                pn.pane.Markdown(
                    "_Operational definition of spacetime curvature (Einstein 1916). "
                    "Relative acceleration of nearby geodesics ∝ Riemann tensor._"
                ),
                qg_tidal_eigenvalues,
                qg_tidal_summary,

                pn.layout.Divider(),
                pn.pane.Markdown("## 🕐 Time Evolution (4D Spacetime Block Analysis)"),
                pn.pane.Markdown(
                    "_Enable 'Compute Time Evolution' in settings to analyze all MC frames. "
                    "Shows dimension reduction, ADM conservation, entropy growth, and singularity formation over time._"
                ),
                pn.Accordion(
                    ("Time Evolution Plots", pn.Column(
                        qg_time_summary,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Regge Action Evolution"),
                        qg_regge_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### ADM Mass Evolution (Energy Conservation)"),
                        qg_adm_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Spectral Dimension Evolution (Dimension Reduction)"),
                        qg_spectral_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Hausdorff Dimension Evolution (Fractal → Manifold)"),
                        qg_hausdorff_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Holographic Entropy Evolution (2nd Law)"),
                        qg_holographic_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Raychaudhuri Expansion Evolution (Singularity Predictor)"),
                        qg_raychaudhuri_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Causal Structure Evolution"),
                        qg_causal_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Spin Network Evolution"),
                        qg_spin_evolution,
                        pn.layout.Divider(),
                        pn.pane.Markdown("#### Tidal Strength Evolution"),
                        qg_tidal_evolution,
                    )),
                    active=[],  # Collapsed by default
                    sizing_mode="stretch_width",
                ),

                sizing_mode="stretch_both",
            )

            isospin_tab = pn.Column(
                isospin_status,
                pn.Row(isospin_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Channel Settings (shared with Strong Force)", channel_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                # --- Up-type section ---
                pn.pane.Markdown("### Up-Type (Cloners: will_clone=True)"),
                pn.pane.Markdown("#### Mass Spectrum"),
                isospin_up_spectrum,
                pn.pane.Markdown("#### All Channels Overlay - Correlators"),
                isospin_up_overlay_corr,
                pn.pane.Markdown("#### All Channels Overlay - Effective Masses"),
                isospin_up_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Extracted Masses (Up)"),
                isospin_mass_mode,
                isospin_up_mass_table,
                isospin_up_ratio_pane,
                pn.pane.Markdown("#### Best-Fit Scales (Up)"),
                isospin_up_fit_table,
                pn.pane.Markdown("#### Anchored Mass Table (Up)"),
                isospin_up_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Channel Plots (Correlator + Effective Mass)"),
                isospin_up_plateau,
                pn.layout.Divider(),
                # --- Down-type section ---
                pn.pane.Markdown("### Down-Type (Persisters: will_clone=False)"),
                pn.pane.Markdown("#### Mass Spectrum"),
                isospin_down_spectrum,
                pn.pane.Markdown("#### All Channels Overlay - Correlators"),
                isospin_down_overlay_corr,
                pn.pane.Markdown("#### All Channels Overlay - Effective Masses"),
                isospin_down_overlay_meff,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Extracted Masses (Down)"),
                isospin_down_mass_table,
                isospin_down_ratio_pane,
                pn.pane.Markdown("#### Best-Fit Scales (Down)"),
                isospin_down_fit_table,
                pn.pane.Markdown("#### Anchored Mass Table (Down)"),
                isospin_down_anchor_table,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Channel Plots (Correlator + Effective Mass)"),
                isospin_down_plateau,
                pn.layout.Divider(),
                # --- Isospin splitting comparison ---
                pn.pane.Markdown("### Isospin Mass Splitting (Up / Down)"),
                pn.pane.Markdown(
                    "_Per-channel comparison of up-type vs down-type extracted masses, "
                    "with PDG observed isospin splittings for the corresponding hadrons._"
                ),
                isospin_split_table,
                isospin_ratio_pane,
                sizing_mode="stretch_both",
            )

            main.objects = [
                pn.Tabs(
                    ("Simulation", simulation_tab),
                    ("Dynamics", dynamics_tab),
                    ("Analysis", analysis_tab),
                    ("Fractal Set", fractal_set_tab),
                    ("Strong Force", channels_tab),
                    ("Weak Isospin", isospin_tab),
                    ("Radial Strong Force", radial_tab),
                    ("Anisotropic Edge", anisotropic_edge_tab),
                    ("Radial Electroweak", radial_ew_tab),
                    ("Electroweak", electroweak_tab),
                    ("Higgs Field", higgs_tab),
                    ("Quantum Gravity", quantum_gravity_tab),
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
    print(
        f"QFT Swarm Convergence Dashboard running at http://localhost:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    pn.serve(create_app, port=args.port, show=args.open, title="QFT Swarm Convergence Dashboard")
