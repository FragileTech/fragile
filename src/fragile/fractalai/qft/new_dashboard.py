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
    _fft_correlator_batched,
    ChannelConfig,
    ChannelCorrelatorResult,
    CorrelatorConfig,
    compute_effective_mass_torch,
    compute_channel_correlator,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.aggregation import compute_all_operator_series
from fragile.fractalai.qft.baryon_triplet_channels import (
    BaryonTripletCorrelatorConfig,
    compute_companion_baryon_correlator,
)
from fragile.fractalai.qft.meson_phase_channels import (
    MesonPhaseCorrelatorConfig,
    compute_companion_meson_phase_correlator,
)
from fragile.fractalai.qft.vector_meson_channels import (
    VectorMesonCorrelatorConfig,
    compute_companion_vector_meson_correlator,
)
from fragile.fractalai.qft.glueball_color_channels import (
    GlueballColorCorrelatorConfig,
    compute_companion_glueball_color_correlator,
)
from fragile.fractalai.qft.tensor_momentum_channels import (
    TensorMomentumCorrelatorConfig,
    compute_companion_tensor_momentum_correlator,
)
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
    _apply_pbc_diff_torch,
    _compute_color_states_single,
    _recorded_subgraph_for_alive,
    _slice_bounds,
    RadialChannelBundle,
    RadialChannelConfig,
    compute_radial_channels,
)
from fragile.fractalai.qft.anisotropic_edge_channels import (
    _extract_edges_for_frame,
    _resolve_edge_weights,
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
from fragile.fractalai.qft.dirac_electroweak import (
    compute_dirac_electroweak_bundle,
    DiracElectroweakConfig,
)
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
        self._time_distribution_container = None
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
    use_companion_baryon_triplet = param.Boolean(
        default=True,
        doc="Use companion-triplet baryon channel implementation for nucleon channel.",
    )
    baryon_use_connected = param.Boolean(
        default=True,
        doc="Use connected baryon correlator C_B(τ)-<B>² for nucleon channel.",
    )
    baryon_max_lag = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Override max lag for baryon channel (None uses max_lag).",
    )
    baryon_color_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for baryon determinant (blank uses first 3).",
    )
    baryon_eps = param.Number(
        default=1e-12,
        bounds=(0.0, None),
        doc="Minimum |det| threshold for valid baryon triplets.",
    )
    use_companion_meson_phase = param.Boolean(
        default=True,
        doc="Use companion-pair color-phase implementation for scalar/pseudoscalar channels.",
    )
    meson_use_connected = param.Boolean(
        default=True,
        doc="Use connected meson phase correlator C(τ)-<O>² for scalar/pseudoscalar channels.",
    )
    meson_max_lag = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Override max lag for meson phase channels (None uses max_lag).",
    )
    meson_pair_selection = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion-pair source for meson phase channels.",
    )
    meson_color_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for meson color inner products (blank uses first 3).",
    )
    meson_eps = param.Number(
        default=1e-12,
        bounds=(0.0, None),
        doc="Minimum |c_i†c_j| threshold for valid meson pairs.",
    )
    use_companion_vector_meson = param.Boolean(
        default=True,
        doc="Use companion-pair color-displacement implementation for vector/axial channels.",
    )
    vector_meson_use_connected = param.Boolean(
        default=True,
        doc="Use connected vector meson correlator C(τ)-<O>² for vector/axial channels.",
    )
    vector_meson_max_lag = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Override max lag for vector meson channels (None uses max_lag).",
    )
    vector_meson_pair_selection = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion-pair source for vector meson channels.",
    )
    vector_meson_color_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for vector meson color inner products (blank uses first 3).",
    )
    vector_meson_position_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for vector meson displacements (blank uses first 3).",
    )
    vector_meson_eps = param.Number(
        default=1e-12,
        bounds=(0.0, None),
        doc="Minimum |c_i†c_j| threshold for valid vector meson pairs.",
    )
    vector_meson_use_unit_displacement = param.Boolean(
        default=False,
        doc="Normalize pair displacement to unit vectors before vector meson projection.",
    )
    use_companion_tensor_momentum = param.Boolean(
        default=True,
        doc=(
            "Compute companion-pair spin-2 tensor channels using old strong-force style "
            "and momentum-projected traceless components."
        ),
    )
    tensor_momentum_use_connected = param.Boolean(
        default=True,
        doc="Use connected tensor momentum correlators C(τ)-<O>².",
    )
    tensor_momentum_max_lag = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Override max lag for tensor momentum channels (None uses max_lag).",
    )
    tensor_momentum_pair_selection = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion-pair source for tensor momentum channels.",
    )
    tensor_momentum_color_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for tensor momentum color inner products (blank uses first 3).",
    )
    tensor_momentum_position_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for tensor momentum displacements (blank uses first 3).",
    )
    tensor_momentum_axis = param.Integer(
        default=0,
        bounds=(0, 10),
        doc="Global spatial axis index used for Fourier momentum projection.",
    )
    tensor_momentum_mode_max = param.Integer(
        default=4,
        bounds=(0, 32),
        doc="Maximum integer momentum mode n (computes p_n for n=0..n_max).",
    )
    tensor_momentum_eps = param.Number(
        default=1e-12,
        bounds=(0.0, None),
        doc="Minimum |c_i†c_j| threshold for valid tensor momentum pairs.",
    )
    use_companion_glueball_color = param.Boolean(
        default=True,
        doc="Use companion-triplet color-plaquette implementation for glueball channel.",
    )
    glueball_use_connected = param.Boolean(
        default=True,
        doc="Use connected glueball correlator C(τ)-<O>² for glueball channel.",
    )
    glueball_max_lag = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Override max lag for glueball channel (None uses max_lag).",
    )
    glueball_color_dims_spec = param.String(
        default="0,1,2",
        doc="Comma-separated 3 dims for glueball color plaquette (blank uses first 3).",
    )
    glueball_eps = param.Number(
        default=1e-12,
        bounds=(0.0, None),
        doc="Minimum |c_i†c_j| threshold for valid glueball triplets.",
    )
    glueball_use_action_form = param.Boolean(
        default=False,
        doc="Use action form 1-Re(Π) instead of Re(Π) for glueball operator.",
    )
    glueball_use_momentum_projection = param.Boolean(
        default=False,
        doc="Compute momentum-projected SU(3) glueball correlators C_p(Δt) from Fourier modes.",
    )
    glueball_momentum_axis = param.Integer(
        default=0,
        bounds=(0, 10),
        doc="Spatial axis index used for Fourier momentum projection.",
    )
    glueball_momentum_mode_max = param.Integer(
        default=3,
        bounds=(0, 32),
        doc="Maximum integer momentum mode n (computes p_n = 2πn/L for n=0..n_max).",
    )
    channel_list = param.String(
        default=(
            "scalar,pseudoscalar,vector,axial_vector,tensor,tensor_traceless,nucleon,glueball"
        )
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


class NewDiracElectroweakSettings(param.Parameterized):
    """Settings for the unified Dirac/Electroweak analysis tab."""

    simulation_range = param.Range(
        default=(0.1, 1.0),
        bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end).",
    )
    max_lag = param.Integer(default=80, bounds=(10, 200))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    use_connected = param.Boolean(default=True)
    neighbor_method = param.ObjectSelector(
        default="companions",
        objects=("auto", "recorded", "companions", "voronoi"),
    )
    companion_topology = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion topology when companion neighbors are used.",
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
    )
    neighbor_k = param.Integer(default=0, bounds=(0, 50))
    channel_list = param.String(default=",".join(ELECTROWEAK_CHANNELS))
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))
    epsilon_d = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    epsilon_c = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    epsilon_clone = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    lambda_alg = param.Number(default=None, bounds=(0.0, None), allow_None=True)
    compute_bootstrap_errors = param.Boolean(default=False)
    n_bootstrap = param.Integer(default=100, bounds=(10, 1000))

    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Recorded MC index/step for snapshot analyses (blank=last).",
    )
    dirac_kernel_mode = param.ObjectSelector(
        default="phase_space",
        objects=("phase_space", "fitness_ratio"),
        doc="Dirac kernel used for spectral analysis.",
    )
    dirac_time_average = param.Boolean(default=False)
    dirac_warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.5))
    dirac_max_avg_frames = param.Integer(default=80, bounds=(10, 400))
    dirac_color_threshold_mode = param.ObjectSelector(
        default="median",
        objects=("median", "manual"),
    )
    dirac_color_threshold_value = param.Number(default=1.0, bounds=(0.0, None))
    color_singlet_quantile = param.Number(default=0.9, bounds=(0.0, 1.0))


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
    scalar_density_mode = param.ObjectSelector(
        default="volume",
        objects=["volume", "knn"],
        doc="Density estimator for scalar Einstein regression.",
    )
    knn_k = param.Integer(
        default=10,
        bounds=(1, 256),
        doc="k for KNN density when scalar_density_mode='knn'.",
    )
    coarse_grain_bins = param.Integer(
        default=0,
        bounds=(0, 512),
        doc="Number of radial coarse-graining bins (0 disables coarse fit).",
    )
    coarse_grain_min_points = param.Integer(
        default=5,
        bounds=(1, 512),
        doc="Minimum walkers per coarse bin used in regression.",
    )
    temporal_average_enabled = param.Boolean(
        default=False,
        doc="Average scalar Einstein regression inputs across recent frames.",
    )
    temporal_window_frames = param.Integer(
        default=8,
        bounds=(1, 512),
        doc="Number of frames included in temporal averaging window.",
    )
    temporal_stride = param.Integer(
        default=1,
        bounds=(1, 128),
        doc="Stride between frames in the temporal averaging window.",
    )
    bootstrap_samples = param.Integer(
        default=0,
        bounds=(0, 5000),
        doc="Bootstrap resamples for scalar regression uncertainty (0 disables).",
    )
    bootstrap_confidence = param.Number(
        default=0.95,
        bounds=(0.5, 0.999),
        doc="Confidence level for bootstrap/jackknife intervals.",
    )
    bootstrap_seed = param.Integer(
        default=12345,
        bounds=(0, None),
        doc="Random seed for scalar bootstrap resampling.",
    )
    bootstrap_frame_block_size = param.Integer(
        default=1,
        bounds=(1, 128),
        doc="Temporal block size for frame bootstrap sampling.",
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


def _parse_triplet_dims_spec(spec: str, history_d: int) -> tuple[int, int, int] | None:
    """Parse exactly three dimensions for baryon triplet color determinant."""
    dims = _parse_dims_spec(spec, history_d)
    if dims is None:
        return None
    if len(dims) != 3:
        raise ValueError(
            "baryon_color_dims_spec must contain exactly 3 dims; "
            f"received {dims}."
        )
    return int(dims[0]), int(dims[1]), int(dims[2])


def _build_result_from_precomputed_correlator(
    channel_name: str,
    correlator: torch.Tensor,
    dt: float,
    config: CorrelatorConfig,
    *,
    n_samples: int,
    series: torch.Tensor | None = None,
    correlator_err: torch.Tensor | None = None,
) -> ChannelCorrelatorResult:
    """Build ChannelCorrelatorResult from a precomputed correlator."""
    corr_t = correlator.float()
    effective_mass = compute_effective_mass_torch(corr_t, dt)
    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(corr_t.abs(), dt, config)
        window_data: dict[str, Any] = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(corr_t, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(corr_t, dt, config)
        window_data = {
            "window_masses": mass_fit.pop("window_masses", None),
            "window_aic": mass_fit.pop("window_aic", None),
            "window_widths": mass_fit.pop("window_widths", None),
            "window_r2": mass_fit.pop("window_r2", None),
        }
    series_t = corr_t if series is None else series.float()
    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=corr_t,
        correlator_err=correlator_err,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series_t,
        n_samples=int(n_samples),
        dt=dt,
        **window_data,
    )


def _bootstrap_errors_batched_scalar_series(
    series_stack: torch.Tensor,
    *,
    valid_t: torch.Tensor | None,
    max_lag: int,
    use_connected: bool,
    n_bootstrap: int,
) -> torch.Tensor:
    """Compute vectorized bootstrap correlator errors for multiple scalar series [B, T]."""
    if series_stack.ndim != 2:
        raise ValueError(f"Expected series_stack [B, T], got {tuple(series_stack.shape)}.")
    n_channels, n_time = series_stack.shape
    errors = torch.zeros(
        (n_channels, int(max_lag) + 1), dtype=torch.float32, device=series_stack.device
    )
    if n_channels == 0 or n_time == 0:
        return errors

    if valid_t is None:
        valid_mask = torch.ones(n_time, dtype=torch.bool, device=series_stack.device)
    else:
        valid_mask = valid_t.to(dtype=torch.bool, device=series_stack.device)
        if valid_mask.shape != (n_time,):
            raise ValueError(
                f"valid_t must have shape [{n_time}], got {tuple(valid_mask.shape)}."
            )
    if not torch.any(valid_mask):
        return errors

    trimmed = series_stack[:, valid_mask].float()
    t_len = int(trimmed.shape[1])
    if t_len == 0:
        return errors

    n_boot = int(max(1, n_bootstrap))
    sample_idx = torch.randint(0, t_len, (n_boot, t_len), device=trimmed.device)
    sampled = torch.gather(
        trimmed.unsqueeze(0).expand(n_boot, -1, -1),
        dim=2,
        index=sample_idx.unsqueeze(1).expand(-1, trimmed.shape[0], -1),
    )
    boot_corr = _fft_correlator_batched(
        sampled.reshape(-1, t_len),
        max_lag=int(max_lag),
        use_connected=bool(use_connected),
    )
    return boot_corr.reshape(n_boot, trimmed.shape[0], -1).std(dim=0)


def _bootstrap_error_vector_dot_series(
    vector_series: torch.Tensor,
    *,
    valid_t: torch.Tensor | None,
    max_lag: int,
    use_connected: bool,
    n_bootstrap: int,
) -> torch.Tensor:
    """Bootstrap error for dot-product correlator from vector operator series [T, D]."""
    if vector_series.ndim != 2:
        raise ValueError(f"Expected vector_series [T, D], got {tuple(vector_series.shape)}.")
    n_time, _ = vector_series.shape
    errors = torch.zeros(int(max_lag) + 1, dtype=torch.float32, device=vector_series.device)
    if n_time == 0:
        return errors

    if valid_t is None:
        valid_mask = torch.ones(n_time, dtype=torch.bool, device=vector_series.device)
    else:
        valid_mask = valid_t.to(dtype=torch.bool, device=vector_series.device)
        if valid_mask.shape != (n_time,):
            raise ValueError(
                f"valid_t must have shape [{n_time}], got {tuple(valid_mask.shape)}."
            )
    if not torch.any(valid_mask):
        return errors

    trimmed = vector_series[valid_mask].float()  # [T_valid, D]
    t_len = int(trimmed.shape[0])
    if t_len == 0:
        return errors

    # Correlator is sum over component auto-correlators: <V(t)·V(t+lag)>.
    comp_stack = trimmed.transpose(0, 1).contiguous()  # [D, T_valid]
    n_boot = int(max(1, n_bootstrap))
    sample_idx = torch.randint(0, t_len, (n_boot, t_len), device=comp_stack.device)
    sampled = torch.gather(
        comp_stack.unsqueeze(0).expand(n_boot, -1, -1),
        dim=2,
        index=sample_idx.unsqueeze(1).expand(-1, comp_stack.shape[0], -1),
    )  # [B, D, T_valid]
    boot_corr_comp = _fft_correlator_batched(
        sampled.reshape(-1, t_len),
        max_lag=int(max_lag),
        use_connected=bool(use_connected),
    ).reshape(n_boot, comp_stack.shape[0], -1)
    boot_corr_total = boot_corr_comp.sum(dim=1)
    return boot_corr_total.std(dim=0)


def _compute_anisotropic_baryon_triplet_result(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> tuple[ChannelCorrelatorResult, int]:
    """Compute nucleon channel using companion-triplet baryon correlator."""
    baryon_max_lag = int(settings.baryon_max_lag or settings.max_lag)
    baryon_cfg = BaryonTripletCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=baryon_max_lag,
        use_connected=bool(settings.baryon_use_connected),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.baryon_color_dims_spec, history.d),
        eps=float(settings.baryon_eps),
    )
    baryon_out = compute_companion_baryon_correlator(history, baryon_cfg)
    dt = float(history.delta_t * history.record_every)
    fit_cfg = CorrelatorConfig(
        max_lag=baryon_max_lag,
        use_connected=bool(settings.baryon_use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    correlator_err = None
    if bool(settings.compute_bootstrap_errors):
        baryon_err = _bootstrap_errors_batched_scalar_series(
            baryon_out.operator_baryon_series.unsqueeze(0),
            valid_t=(baryon_out.triplet_counts_per_frame > 0),
            max_lag=baryon_max_lag,
            use_connected=bool(settings.baryon_use_connected),
            n_bootstrap=int(settings.n_bootstrap),
        )
        correlator_err = baryon_err[0]
    result = _build_result_from_precomputed_correlator(
        channel_name="nucleon",
        correlator=baryon_out.correlator,
        dt=dt,
        config=fit_cfg,
        n_samples=int(baryon_out.n_valid_source_triplets),
        series=baryon_out.operator_baryon_series,
        correlator_err=correlator_err,
    )
    valid_frames = int((baryon_out.triplet_counts_per_frame > 0).sum().item())
    return result, valid_frames


def _compute_anisotropic_meson_phase_results(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    requested_channels: set[str],
) -> tuple[dict[str, ChannelCorrelatorResult], int]:
    """Compute scalar/pseudoscalar channels using companion-pair meson phases."""
    meson_max_lag = int(settings.meson_max_lag or settings.max_lag)
    meson_cfg = MesonPhaseCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=meson_max_lag,
        use_connected=bool(settings.meson_use_connected),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.meson_color_dims_spec, history.d),
        pair_selection=str(settings.meson_pair_selection),
        eps=float(settings.meson_eps),
    )
    meson_out = compute_companion_meson_phase_correlator(history, meson_cfg)
    dt = float(history.delta_t * history.record_every)
    fit_cfg = CorrelatorConfig(
        max_lag=meson_max_lag,
        use_connected=bool(settings.meson_use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    results: dict[str, ChannelCorrelatorResult] = {}
    n_samples = int(meson_out.n_valid_source_pairs)
    pseudoscalar_err: torch.Tensor | None = None
    scalar_err: torch.Tensor | None = None
    if bool(settings.compute_bootstrap_errors):
        meson_err = _bootstrap_errors_batched_scalar_series(
            torch.stack(
                [meson_out.operator_pseudoscalar_series, meson_out.operator_scalar_series], dim=0
            ),
            valid_t=(meson_out.pair_counts_per_frame > 0),
            max_lag=meson_max_lag,
            use_connected=bool(settings.meson_use_connected),
            n_bootstrap=int(settings.n_bootstrap),
        )
        pseudoscalar_err = meson_err[0]
        scalar_err = meson_err[1]
    if "pseudoscalar" in requested_channels:
        results["pseudoscalar"] = _build_result_from_precomputed_correlator(
            channel_name="pseudoscalar",
            correlator=meson_out.pseudoscalar,
            dt=dt,
            config=fit_cfg,
            n_samples=n_samples,
            series=meson_out.operator_pseudoscalar_series,
            correlator_err=pseudoscalar_err,
        )
    if "scalar" in requested_channels:
        results["scalar"] = _build_result_from_precomputed_correlator(
            channel_name="scalar",
            correlator=meson_out.scalar,
            dt=dt,
            config=fit_cfg,
            n_samples=n_samples,
            series=meson_out.operator_scalar_series,
            correlator_err=scalar_err,
        )
    valid_frames = int((meson_out.pair_counts_per_frame > 0).sum().item())
    return results, valid_frames


def _compute_anisotropic_vector_meson_results(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    requested_channels: set[str],
) -> tuple[dict[str, ChannelCorrelatorResult], int]:
    """Compute vector/axial-vector channels using companion-pair vector mesons."""
    vector_max_lag = int(settings.vector_meson_max_lag or settings.max_lag)
    vector_cfg = VectorMesonCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=vector_max_lag,
        use_connected=bool(settings.vector_meson_use_connected),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.vector_meson_color_dims_spec, history.d),
        position_dims=_parse_triplet_dims_spec(settings.vector_meson_position_dims_spec, history.d),
        pair_selection=str(settings.vector_meson_pair_selection),
        eps=float(settings.vector_meson_eps),
        use_unit_displacement=bool(settings.vector_meson_use_unit_displacement),
    )
    vector_out = compute_companion_vector_meson_correlator(history, vector_cfg)
    dt = float(history.delta_t * history.record_every)
    fit_cfg = CorrelatorConfig(
        max_lag=vector_max_lag,
        use_connected=bool(settings.vector_meson_use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    results: dict[str, ChannelCorrelatorResult] = {}
    n_samples = int(vector_out.n_valid_source_pairs)
    vector_err: torch.Tensor | None = None
    axial_err: torch.Tensor | None = None
    if bool(settings.compute_bootstrap_errors):
        valid_t = vector_out.pair_counts_per_frame > 0
        vector_err = _bootstrap_error_vector_dot_series(
            vector_out.operator_vector_series,
            valid_t=valid_t,
            max_lag=vector_max_lag,
            use_connected=bool(settings.vector_meson_use_connected),
            n_bootstrap=int(settings.n_bootstrap),
        )
        axial_err = _bootstrap_error_vector_dot_series(
            vector_out.operator_axial_vector_series,
            valid_t=valid_t,
            max_lag=vector_max_lag,
            use_connected=bool(settings.vector_meson_use_connected),
            n_bootstrap=int(settings.n_bootstrap),
        )
    if "vector" in requested_channels:
        results["vector"] = _build_result_from_precomputed_correlator(
            channel_name="vector",
            correlator=vector_out.vector,
            dt=dt,
            config=fit_cfg,
            n_samples=n_samples,
            series=vector_out.vector,
            correlator_err=vector_err,
        )
    if "axial_vector" in requested_channels:
        results["axial_vector"] = _build_result_from_precomputed_correlator(
            channel_name="axial_vector",
            correlator=vector_out.axial_vector,
            dt=dt,
            config=fit_cfg,
            n_samples=n_samples,
            series=vector_out.axial_vector,
            correlator_err=axial_err,
        )
    valid_frames = int((vector_out.pair_counts_per_frame > 0).sum().item())
    return results, valid_frames


def _compute_anisotropic_glueball_color_result(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    *,
    force_momentum_projection: bool | None = None,
    momentum_mode_max: int | None = None,
) -> tuple[dict[str, ChannelCorrelatorResult], int]:
    """Compute glueball channel using companion-triplet color plaquettes."""
    glueball_max_lag = int(settings.glueball_max_lag or settings.max_lag)
    use_momentum_projection = (
        bool(force_momentum_projection)
        if force_momentum_projection is not None
        else bool(settings.glueball_use_momentum_projection)
    )
    mode_max = (
        int(momentum_mode_max)
        if momentum_mode_max is not None
        else int(settings.glueball_momentum_mode_max)
    )
    glueball_cfg = GlueballColorCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=glueball_max_lag,
        use_connected=bool(settings.glueball_use_connected),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.glueball_color_dims_spec, history.d),
        eps=float(settings.glueball_eps),
        use_action_form=bool(settings.glueball_use_action_form),
        use_momentum_projection=use_momentum_projection,
        momentum_axis=int(settings.glueball_momentum_axis),
        momentum_mode_max=mode_max,
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    glueball_out = compute_companion_glueball_color_correlator(history, glueball_cfg)
    dt = float(history.delta_t * history.record_every)
    fit_cfg = CorrelatorConfig(
        max_lag=glueball_max_lag,
        use_connected=bool(settings.glueball_use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    correlator_err = None
    if bool(settings.compute_bootstrap_errors):
        glueball_err = _bootstrap_errors_batched_scalar_series(
            glueball_out.operator_glueball_series.unsqueeze(0),
            valid_t=(glueball_out.triplet_counts_per_frame > 0),
            max_lag=glueball_max_lag,
            use_connected=bool(settings.glueball_use_connected),
            n_bootstrap=int(settings.n_bootstrap),
        )
        correlator_err = glueball_err[0]
    result = _build_result_from_precomputed_correlator(
        channel_name="glueball",
        correlator=glueball_out.correlator,
        dt=dt,
        config=fit_cfg,
        n_samples=int(glueball_out.n_valid_source_triplets),
        series=glueball_out.operator_glueball_series,
        correlator_err=correlator_err,
    )
    results: dict[str, ChannelCorrelatorResult] = {"glueball": result}

    if (
        use_momentum_projection
        and glueball_out.momentum_modes is not None
        and glueball_out.momentum_correlator is not None
    ):
        momentum_modes = glueball_out.momentum_modes
        momentum_corr = glueball_out.momentum_correlator
        momentum_err = glueball_out.momentum_correlator_err
        momentum_cos = glueball_out.momentum_operator_cos_series
        momentum_sin = glueball_out.momentum_operator_sin_series
        n_mode = int(momentum_modes.shape[0])
        n_samples_momentum = int(glueball_out.momentum_valid_frames)
        for mode_idx in range(n_mode):
            mode_n = int(round(float(momentum_modes[mode_idx].item())))
            channel_name = f"glueball_momentum_p{mode_n}"
            mode_series = None
            if momentum_cos is not None and momentum_sin is not None:
                mode_series = torch.sqrt(
                    torch.clamp(
                        momentum_cos[mode_idx].float() ** 2 + momentum_sin[mode_idx].float() ** 2,
                        min=0.0,
                    )
                )
            mode_err = None
            if momentum_err is not None and mode_idx < int(momentum_err.shape[0]):
                mode_err = momentum_err[mode_idx]
            results[channel_name] = _build_result_from_precomputed_correlator(
                channel_name=channel_name,
                correlator=momentum_corr[mode_idx],
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples_momentum,
                series=mode_series,
                correlator_err=mode_err,
            )

    valid_frames = int((glueball_out.triplet_counts_per_frame > 0).sum().item())
    valid_frames = max(valid_frames, int(getattr(glueball_out, "momentum_valid_frames", 0)))
    return results, valid_frames


def _compute_anisotropic_edge_isotropic_glueball_result(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> ChannelCorrelatorResult | None:
    """Compute isotropic-edge glueball estimator (no SU(3) override)."""
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
        component_mode="isotropic",
        nucleon_triplet_mode=str(settings.nucleon_triplet_mode),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    output = compute_anisotropic_edge_channels(history, config=config, channels=["glueball"])
    return output.channel_results.get("glueball")


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
    use_baryon_triplet = bool(settings.use_companion_baryon_triplet) and ("nucleon" in channels)
    meson_override_targets = {"scalar", "pseudoscalar"}
    requested_meson_targets = set(channels) & meson_override_targets
    use_meson_phase = bool(settings.use_companion_meson_phase) and bool(requested_meson_targets)
    vector_override_targets = {"vector", "axial_vector"}
    requested_vector_targets = set(channels) & vector_override_targets
    use_vector_meson = bool(settings.use_companion_vector_meson) and bool(requested_vector_targets)
    use_glueball_color = bool(settings.use_companion_glueball_color) and ("glueball" in channels)

    override_channels: set[str] = set()
    if use_baryon_triplet:
        override_channels.add("nucleon")
    if use_meson_phase:
        override_channels.update(requested_meson_targets)
    if use_vector_meson:
        override_channels.update(requested_vector_targets)
    if use_glueball_color:
        override_channels.add("glueball")
    anisotropic_channels = [channel for channel in channels if channel not in override_channels]

    output = compute_anisotropic_edge_channels(history, config=config, channels=anisotropic_channels)
    if not use_baryon_triplet and not use_meson_phase and not use_vector_meson and not use_glueball_color:
        return output

    merged_results = dict(output.channel_results)
    valid_frame_counts = [int(output.n_valid_frames)]
    if use_baryon_triplet:
        baryon_result, baryon_valid_frames = _compute_anisotropic_baryon_triplet_result(history, settings)
        merged_results["nucleon"] = baryon_result
        valid_frame_counts.append(int(baryon_valid_frames))
    if use_meson_phase:
        meson_results, meson_valid_frames = _compute_anisotropic_meson_phase_results(
            history,
            settings,
            requested_channels=requested_meson_targets,
        )
        merged_results.update(meson_results)
        valid_frame_counts.append(int(meson_valid_frames))
    if use_vector_meson:
        vector_results, vector_valid_frames = _compute_anisotropic_vector_meson_results(
            history,
            settings,
            requested_channels=requested_vector_targets,
        )
        merged_results.update(vector_results)
        valid_frame_counts.append(int(vector_valid_frames))
    if use_glueball_color:
        glueball_results, glueball_valid_frames = _compute_anisotropic_glueball_color_result(
            history, settings
        )
        merged_results.update(glueball_results)
        valid_frame_counts.append(int(glueball_valid_frames))

    return AnisotropicEdgeChannelOutput(
        channel_results=merged_results,
        component_labels=output.component_labels,
        frame_indices=output.frame_indices,
        n_valid_frames=max(valid_frame_counts) if valid_frame_counts else 0,
        avg_alive_walkers=output.avg_alive_walkers,
        avg_edges=output.avg_edges,
    )


def _compute_strong_glueball_for_anisotropic_edge(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> ChannelCorrelatorResult:
    """Compute the strong-force tab glueball channel for cross-checking.

    This reuses the same operator/correlator pipeline as the strong-force channels:
    - glueball operator from force norm squared
    - FFT correlator + configured mass extraction
    """
    channel_config = ChannelConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        time_axis="mc",
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        neighbor_method="auto",
        edge_weight_mode=str(settings.edge_weight_mode),
    )
    correlator_config = CorrelatorConfig(
        max_lag=int(settings.max_lag),
        use_connected=bool(settings.use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )

    operator_series = compute_all_operator_series(
        history,
        channel_config,
        channels=["glueball"],
    )
    glueball_series = operator_series.operators.get("glueball", torch.zeros(0, dtype=torch.float32))
    return compute_channel_correlator(
        series=glueball_series,
        dt=operator_series.dt,
        config=correlator_config,
        channel_name="glueball_strong_force",
    )


def _compute_strong_tensor_for_anisotropic_edge(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> ChannelCorrelatorResult:
    """Compute legacy strong-force tensor channel for cross-checking."""
    channel_config = ChannelConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        time_axis="mc",
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        neighbor_method="auto",
        edge_weight_mode=str(settings.edge_weight_mode),
    )
    correlator_config = CorrelatorConfig(
        max_lag=int(settings.max_lag),
        use_connected=bool(settings.use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )

    operator_series = compute_all_operator_series(
        history,
        channel_config,
        channels=["tensor"],
    )
    tensor_series = operator_series.operators.get("tensor", torch.zeros(0, dtype=torch.float32))
    return compute_channel_correlator(
        series=tensor_series,
        dt=operator_series.dt,
        config=correlator_config,
        channel_name="tensor_strong_force",
    )


def _compute_tensor_momentum_for_anisotropic_edge(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> tuple[dict[str, ChannelCorrelatorResult], dict[str, Any]]:
    """Compute momentum-projected tensor correlators and build fit results."""
    if not bool(settings.use_companion_tensor_momentum):
        return {}, {"component_labels": tuple(), "momentum_axis": int(settings.tensor_momentum_axis)}

    tensor_max_lag = int(settings.tensor_momentum_max_lag or settings.max_lag)
    tensor_cfg = TensorMomentumCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=tensor_max_lag,
        use_connected=bool(settings.tensor_momentum_use_connected),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.tensor_momentum_color_dims_spec, history.d),
        position_dims=_parse_triplet_dims_spec(settings.tensor_momentum_position_dims_spec, history.d),
        pair_selection=str(settings.tensor_momentum_pair_selection),
        eps=float(settings.tensor_momentum_eps),
        momentum_axis=int(settings.tensor_momentum_axis),
        momentum_mode_max=int(settings.tensor_momentum_mode_max),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )
    tensor_out = compute_companion_tensor_momentum_correlator(history, tensor_cfg)

    dt = float(history.delta_t * history.record_every)
    fit_cfg = CorrelatorConfig(
        max_lag=tensor_max_lag,
        use_connected=bool(settings.tensor_momentum_use_connected),
        window_widths=_parse_window_widths(settings.window_widths_spec),
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=settings.fit_stop,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )

    results: dict[str, ChannelCorrelatorResult] = {}
    component_labels = tuple(str(label) for label in tensor_out.component_labels)
    n_samples = int(tensor_out.momentum_valid_frames)
    momentum_modes = tensor_out.momentum_modes
    n_modes = int(momentum_modes.shape[0])

    for mode_idx in range(n_modes):
        mode_n = int(round(float(momentum_modes[mode_idx].item())))

        contracted_series = torch.sqrt(
            torch.clamp(
                tensor_out.momentum_operator_cos_series[mode_idx].float().pow(2).sum(dim=0)
                + tensor_out.momentum_operator_sin_series[mode_idx].float().pow(2).sum(dim=0),
                min=0.0,
            )
        )
        channel_name = f"tensor_momentum_p{mode_n}"
        corr_err = None
        if tensor_out.momentum_contracted_correlator_err is not None:
            corr_err = tensor_out.momentum_contracted_correlator_err[mode_idx]
        results[channel_name] = _build_result_from_precomputed_correlator(
            channel_name=channel_name,
            correlator=tensor_out.momentum_contracted_correlator[mode_idx],
            dt=dt,
            config=fit_cfg,
            n_samples=n_samples,
            series=contracted_series,
            correlator_err=corr_err,
        )

        for comp_idx, comp_label in enumerate(component_labels):
            comp_series = torch.sqrt(
                torch.clamp(
                    tensor_out.momentum_operator_cos_series[mode_idx, comp_idx].float() ** 2
                    + tensor_out.momentum_operator_sin_series[mode_idx, comp_idx].float() ** 2,
                    min=0.0,
                )
            )
            comp_name = f"tensor_momentum_p{mode_n}_{comp_label}"
            comp_err = None
            if tensor_out.momentum_correlator_err is not None:
                comp_err = tensor_out.momentum_correlator_err[mode_idx, comp_idx]
            results[comp_name] = _build_result_from_precomputed_correlator(
                channel_name=comp_name,
                correlator=tensor_out.momentum_correlator[mode_idx, comp_idx],
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples,
                series=comp_series,
                correlator_err=comp_err,
            )

    metadata = {
        "component_labels": component_labels,
        "momentum_axis": int(tensor_out.momentum_axis),
        "momentum_modes": momentum_modes.detach().cpu(),
        "momentum_length_scale": float(tensor_out.momentum_length_scale),
        "n_valid_frames": int(tensor_out.momentum_valid_frames),
        "pair_selection": str(tensor_out.pair_selection),
    }
    return results, metadata


def _resolve_lorentz_mc_time_index(history: RunHistory, mc_time_index: int | None) -> int:
    """Resolve radial-snapshot index for Lorentz glueball checks."""
    if history.n_recorded < 2:
        msg = "Need at least 2 recorded timesteps for glueball Lorentz check."
        raise ValueError(msg)
    if mc_time_index is None:
        resolved = history.n_recorded - 1
    else:
        try:
            raw = int(mc_time_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mc_time_index: {mc_time_index}") from exc
        if raw in history.recorded_steps:
            resolved = history.get_step_index(raw)
        else:
            resolved = raw
    if resolved < 1 or resolved >= history.n_recorded:
        msg = (
            f"mc_time_index {resolved} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1} "
            "or a recorded step value)."
        )
        raise ValueError(msg)
    return resolved


def _estimate_frame_geodesic_edge_spacing(history: RunHistory, frame_idx: int) -> float | None:
    """Estimate IG spatial spacing rho as median geodesic edge length on one frame."""
    geodesic_history = getattr(history, "geodesic_edge_distances", None)
    if geodesic_history is None or frame_idx < 0 or frame_idx >= len(geodesic_history):
        return None
    geodesic_frame = geodesic_history[frame_idx]
    if geodesic_frame is None:
        return None
    geodesic_tensor = (
        geodesic_frame
        if torch.is_tensor(geodesic_frame)
        else torch.as_tensor(geodesic_frame, dtype=torch.float32)
    )
    if geodesic_tensor.numel() == 0:
        return None
    flat = geodesic_tensor.detach().reshape(-1).float().cpu()
    mask = torch.isfinite(flat) & (flat > 0)
    if not bool(mask.any()):
        return None
    return float(torch.median(flat[mask]).item())


def _build_lorentz_radial_config(
    anisotropic_settings: AnisotropicEdgeSettings,
    radial_settings: RadialSettings,
) -> RadialChannelConfig:
    """Build a radial snapshot config used by Lorentz checks."""
    return RadialChannelConfig(
        time_axis="radial",
        mc_time_index=anisotropic_settings.mc_time_index,
        warmup_fraction=float(radial_settings.simulation_range[0]),
        end_fraction=float(radial_settings.simulation_range[1]),
        max_lag=int(radial_settings.max_lag),
        use_connected=bool(radial_settings.use_connected),
        n_bins=int(radial_settings.n_bins),
        max_pairs=int(radial_settings.max_pairs),
        distance_mode=str(radial_settings.distance_mode),
        neighbor_method=str(radial_settings.neighbor_method),
        neighbor_weighting=str(radial_settings.neighbor_weighting),
        neighbor_k=int(radial_settings.neighbor_k),
        kernel_length_scale=float(radial_settings.kernel_length_scale),
        h_eff=float(radial_settings.h_eff),
        mass=float(radial_settings.mass),
        ell0=radial_settings.ell0,
        use_volume_weights=bool(radial_settings.use_volume_weights),
        apply_power_correction=bool(radial_settings.apply_power_correction),
        power_override=radial_settings.power_override,
        window_widths=_parse_window_widths(radial_settings.window_widths_spec),
        drop_axis_average=False,
        compute_bootstrap_errors=bool(radial_settings.compute_bootstrap_errors),
        n_bootstrap=int(radial_settings.n_bootstrap),
    )


def _compute_tensor_traceless_snapshot_operator(
    history: RunHistory,
    frame_idx: int,
    anisotropic_settings: AnisotropicEdgeSettings,
) -> torch.Tensor:
    """Compute per-walker symmetric-traceless tensor scalar at one recorded frame."""
    if frame_idx <= 0 or frame_idx >= history.n_recorded:
        msg = (
            f"frame_idx {frame_idx} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1})."
        )
        raise ValueError(msg)

    alive_idx = torch.where(history.alive_mask[frame_idx - 1])[0]
    n_alive = int(alive_idx.numel())
    if n_alive == 0:
        return torch.zeros(0, dtype=torch.float32, device=history.x_before_clone.device)

    keep_dims = _parse_dims_spec(anisotropic_settings.spatial_dims_spec, history.d)
    if keep_dims is None:
        keep_dims = list(range(int(history.d)))
    dim = len(keep_dims)
    if dim < 2:
        return torch.zeros(n_alive, dtype=torch.float32, device=history.x_before_clone.device)

    positions_alive = history.x_before_clone[frame_idx][alive_idx][:, keep_dims]

    color_config = RadialChannelConfig(
        h_eff=float(anisotropic_settings.h_eff),
        mass=float(anisotropic_settings.mass),
        ell0=anisotropic_settings.ell0,
    )
    color_full, valid_full = _compute_color_states_single(
        history,
        frame_idx,
        color_config,
        keep_dims=keep_dims,
    )
    color_alive = color_full[alive_idx]
    valid_alive = valid_full[alive_idx]

    edges_global, geodesic_global = _extract_edges_for_frame(history, frame_idx)
    edges_local, geodesic_local, local_edge_indices = _recorded_subgraph_for_alive(
        edges_global,
        geodesic_global,
        alive_idx,
    )
    if edges_local.shape[0] == 0:
        return torch.zeros(n_alive, dtype=torch.float32, device=positions_alive.device)

    bounds = _slice_bounds(history.bounds, keep_dims)
    edge_cfg = AnisotropicEdgeChannelConfig(
        edge_weight_mode=str(anisotropic_settings.edge_weight_mode),
    )
    edge_weights = _resolve_edge_weights(
        history=history,
        config=edge_cfg,
        frame_idx=frame_idx,
        edges_global=edges_global,
        edges_local=edges_local,
        geodesic_local=geodesic_local,
        local_edge_indices=local_edge_indices,
        positions_alive=positions_alive,
        bounds=bounds,
    ).float()

    src = torch.as_tensor(edges_local[:, 0], device=positions_alive.device, dtype=torch.long)
    dst = torch.as_tensor(edges_local[:, 1], device=positions_alive.device, dtype=torch.long)

    if bool(anisotropic_settings.use_volume_weights):
        volume_history = getattr(history, "riemannian_volume_weights", None)
        if volume_history is None or frame_idx - 1 < 0 or frame_idx - 1 >= len(volume_history):
            msg = (
                "use_volume_weights=True requires RunHistory.riemannian_volume_weights "
                f"for frame {frame_idx}."
            )
            raise ValueError(msg)
        volume_full = volume_history[frame_idx - 1]
        if not torch.is_tensor(volume_full):
            volume_full = torch.as_tensor(volume_full)
        volume_alive = (
            volume_full.to(device=positions_alive.device, dtype=positions_alive.dtype)[alive_idx]
            .clamp(min=0.0)
        )
        if float(volume_alive.sum().item()) <= 0:
            msg = f"Non-positive total alive volume weight at frame {frame_idx}."
            raise ValueError(msg)
        edge_weights = edge_weights * torch.sqrt(
            volume_alive[src].float() * volume_alive[dst].float()
        )

    diff = positions_alive[dst] - positions_alive[src]
    if bool(history.pbc) and bounds is not None:
        diff = _apply_pbc_diff_torch(diff, bounds)

    color_scalar = (color_alive[src].conj() * color_alive[dst]).sum(dim=-1).real.float()
    valid_edge = (
        valid_alive[src]
        & valid_alive[dst]
        & torch.isfinite(edge_weights)
        & (edge_weights > 0)
        & torch.isfinite(color_scalar)
        & torch.isfinite(diff).all(dim=-1)
    )
    if not torch.any(valid_edge):
        return torch.zeros(n_alive, dtype=torch.float32, device=positions_alive.device)

    src_valid = src[valid_edge]
    w_valid = edge_weights[valid_edge].float()
    c_valid = color_scalar[valid_edge]
    dx_valid = diff[valid_edge].float()

    r2 = torch.sum(dx_valid * dx_valid, dim=-1)
    eye = torch.eye(dim, device=positions_alive.device, dtype=torch.float32)
    traceless = (
        dx_valid.unsqueeze(-1) * dx_valid.unsqueeze(-2)
        - eye.unsqueeze(0) * (r2[:, None, None] / float(dim))
    )

    weighted_tensor = (w_valid * c_valid)[:, None, None] * traceless
    tensor_sum = torch.zeros((n_alive, dim * dim), device=positions_alive.device, dtype=torch.float32)
    tensor_sum.index_add_(0, src_valid, weighted_tensor.reshape(-1, dim * dim))
    weight_sum = torch.zeros(n_alive, device=positions_alive.device, dtype=torch.float32)
    weight_sum.index_add_(0, src_valid, w_valid)

    local_tensor = torch.zeros_like(tensor_sum)
    has_weight = weight_sum > 0
    if torch.any(has_weight):
        local_tensor[has_weight] = tensor_sum[has_weight] / weight_sum[has_weight, None].clamp(min=1e-12)
    local_tensor = local_tensor.reshape(n_alive, dim, dim)

    # Scalar contraction per walker for radial correlation binning.
    return torch.einsum("nij,nij->n", local_tensor, local_tensor).float()


def _compute_spatial_glueball_and_tensor_for_anisotropic_edge(
    history: RunHistory,
    anisotropic_settings: AnisotropicEdgeSettings,
    radial_settings: RadialSettings,
) -> tuple[ChannelCorrelatorResult | None, ChannelCorrelatorResult | None, int, float | None]:
    """Compute spatial glueball/tensor correlators in one batched radial pass."""
    radial_config = _build_lorentz_radial_config(anisotropic_settings, radial_settings)
    bundle = compute_radial_channels(history, config=radial_config, channels=["glueball", "tensor"])
    frame_idx = _resolve_lorentz_mc_time_index(history, radial_config.mc_time_index)
    rho_edge = _estimate_frame_geodesic_edge_spacing(history, frame_idx)
    return (
        bundle.radial_4d.channel_results.get("glueball"),
        bundle.radial_4d.channel_results.get("tensor"),
        frame_idx,
        rho_edge,
    )


def _compute_spatial_tensor_traceless_for_anisotropic_edge(
    history: RunHistory,
    anisotropic_settings: AnisotropicEdgeSettings,
    radial_settings: RadialSettings,
) -> tuple[ChannelCorrelatorResult | None, int, float | None]:
    """Compute spatial radial correlator for the new symmetric-traceless tensor channel."""
    radial_config = _build_lorentz_radial_config(anisotropic_settings, radial_settings)
    frame_idx = _resolve_lorentz_mc_time_index(history, radial_config.mc_time_index)
    tensor_operator = _compute_tensor_traceless_snapshot_operator(
        history=history,
        frame_idx=frame_idx,
        anisotropic_settings=anisotropic_settings,
    )
    bundle = compute_radial_channels(
        history,
        config=radial_config,
        channels=["tensor_traceless"],
        operators_override={"tensor_traceless": tensor_operator},
    )
    rho_edge = _estimate_frame_geodesic_edge_spacing(history, frame_idx)
    return bundle.radial_4d.channel_results.get("tensor_traceless"), frame_idx, rho_edge


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
    neighbor_method = "auto" if settings.neighbor_method == "voronoi" else settings.neighbor_method

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
        neighbor_method=neighbor_method,
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
    neighbor_method = "auto" if settings.neighbor_method == "voronoi" else settings.neighbor_method

    config = ElectroweakChannelConfig(
        warmup_fraction=settings.simulation_range[0],
        end_fraction=settings.simulation_range[1],
        max_lag=settings.max_lag,
        h_eff=settings.h_eff,
        use_connected=settings.use_connected,
        neighbor_method=neighbor_method,
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


def _extract_axis_extent_from_bounds(bounds: Any | None, axis: int) -> float:
    """Return positive axis extent from bounds or NaN when unavailable."""
    if bounds is None or axis < 0:
        return float("nan")
    try:
        if hasattr(bounds, "low") and hasattr(bounds, "high"):
            low_vec = np.asarray(_to_numpy(bounds.low), dtype=float).reshape(-1)
            high_vec = np.asarray(_to_numpy(bounds.high), dtype=float).reshape(-1)
            if axis < low_vec.size and axis < high_vec.size:
                low = float(low_vec[axis])
                high = float(high_vec[axis])
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    return float(high - low)
                return float("nan")

        bounds_arr = np.asarray(_to_numpy(bounds), dtype=float)
    except Exception:
        return float("nan")

    if bounds_arr.ndim != 2:
        return float("nan")
    if bounds_arr.shape[0] > axis and bounds_arr.shape[1] >= 2:
        low = float(bounds_arr[axis, 0])
        high = float(bounds_arr[axis, 1])
    elif bounds_arr.shape[0] >= 2 and bounds_arr.shape[1] > axis:
        low = float(bounds_arr[0, axis])
        high = float(bounds_arr[1, axis])
    else:
        return float("nan")
    if np.isfinite(low) and np.isfinite(high) and high > low:
        return float(high - low)
    return float("nan")


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


def _algorithm_placeholder_plot(message: str) -> hv.Text:
    return hv.Text(0, 0, message).opts(
        width=960,
        height=280,
        toolbar=None,
    )


def _history_transition_steps(history: RunHistory, n_steps: int) -> np.ndarray:
    """Return step axis for transition-indexed arrays [n_recorded-1, ...]."""
    recorded = np.asarray(history.recorded_steps, dtype=float)
    if recorded.size >= n_steps + 1:
        return recorded[1 : n_steps + 1]
    record_every = float(max(1, int(history.record_every)))
    return np.arange(1, n_steps + 1, dtype=float) * record_every


def _compute_masked_mean_p95(values: np.ndarray, alive_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step mean and p95 over alive walkers (vectorized)."""
    arr = np.asarray(values, dtype=float)
    mask = np.asarray(alive_mask, dtype=bool)
    if arr.ndim != 2 or mask.ndim != 2:
        msg = "Expected 2D [time, walkers] arrays for masked statistics."
        raise ValueError(msg)

    n_steps = min(arr.shape[0], mask.shape[0])
    arr = arr[:n_steps]
    mask = mask[:n_steps]

    masked = np.where(mask, arr, np.nan)
    counts = mask.sum(axis=1)
    sums = np.nansum(masked, axis=1)
    mean = np.divide(
        sums,
        counts,
        out=np.full(n_steps, np.nan, dtype=float),
        where=counts > 0,
    )
    p95 = np.full(n_steps, np.nan, dtype=float)
    valid = counts > 0
    if np.any(valid):
        p95[valid] = np.nanpercentile(masked[valid], 95, axis=1)
    return mean, p95


def _compute_vectorized_lyapunov(history: RunHistory) -> dict[str, np.ndarray]:
    """Compute Lyapunov trajectory from recorded states using vectorized numpy ops."""
    x = _to_numpy(history.x_final).astype(float, copy=False)
    v = _to_numpy(history.v_final).astype(float, copy=False)
    if x.ndim != 3 or v.ndim != 3:
        msg = "Expected x_final/v_final with shape [n_recorded, N, d]."
        raise ValueError(msg)

    n_steps = min(int(history.n_recorded), x.shape[0], v.shape[0])
    if n_steps <= 0:
        msg = "No recorded states available for Lyapunov diagnostics."
        raise ValueError(msg)

    x = x[:n_steps]
    v = v[:n_steps]
    n_walkers = int(x.shape[1])

    alive = np.ones((n_steps, n_walkers), dtype=bool)
    if getattr(history, "alive_mask", None) is not None:
        alive_info = _to_numpy(history.alive_mask).astype(bool, copy=False)
        info_len = min(alive_info.shape[0], max(n_steps - 1, 0))
        if info_len > 0:
            alive[1 : 1 + info_len] = alive_info[:info_len]

    alive_3d = alive[..., None]
    counts = alive.sum(axis=1).astype(float)
    safe_counts = np.clip(counts, a_min=1.0, a_max=None)

    x_mean = np.where(alive_3d, x, 0.0).sum(axis=1) / safe_counts[:, None]
    v_mean = np.where(alive_3d, v, 0.0).sum(axis=1) / safe_counts[:, None]

    x_sq = np.sum((x - x_mean[:, None, :]) ** 2, axis=-1)
    v_sq = np.sum((v - v_mean[:, None, :]) ** 2, axis=-1)

    var_x = np.where(alive, x_sq, 0.0).sum(axis=1) / safe_counts
    var_v = np.where(alive, v_sq, 0.0).sum(axis=1) / safe_counts
    v_total = var_x + var_v

    recorded = np.asarray(history.recorded_steps, dtype=float)
    if recorded.size >= n_steps:
        time = recorded[:n_steps]
    else:
        record_every = float(max(1, int(history.record_every)))
        time = np.arange(n_steps, dtype=float) * record_every

    return {
        "time": time,
        "V_total": v_total,
        "V_var_x": var_x,
        "V_var_v": var_v,
    }


def _build_timeseries_mean_p95_plot(
    *,
    step: np.ndarray,
    mean: np.ndarray,
    p95: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
) -> hv.Overlay | hv.Text:
    err95 = np.clip(np.asarray(p95, dtype=float) - np.asarray(mean, dtype=float), 0.0, None)
    frame = pd.DataFrame({"step": step, "mean": mean, "err95": err95}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if frame.empty:
        return _algorithm_placeholder_plot("No data available")

    curve = hv.Curve(frame, "step", "mean").opts(color=color, line_width=2)
    errorbars = hv.ErrorBars(frame, "step", ["mean", "err95"]).opts(
        color=color,
        alpha=0.45,
        line_width=1,
    )
    return (errorbars * curve).opts(
        title=title,
        xlabel="Recorded step",
        ylabel=ylabel,
        width=960,
        height=320,
        show_grid=True,
    )


def _build_timeseries_mean_error_plot(
    *,
    step: np.ndarray,
    mean: np.ndarray,
    error: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
) -> hv.Overlay | hv.Text:
    frame = pd.DataFrame({"step": step, "mean": mean, "error": error}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if frame.empty:
        return _algorithm_placeholder_plot("No data available")

    curve = hv.Curve(frame, "step", "mean").opts(color=color, line_width=2)
    errorbars = hv.ErrorBars(frame, "step", ["mean", "error"]).opts(
        color=color,
        alpha=0.45,
        line_width=1,
    )
    return (errorbars * curve).opts(
        title=title,
        xlabel="Recorded step",
        ylabel=ylabel,
        width=960,
        height=320,
        show_grid=True,
    )


def _compute_interwalker_distance_stats(
    positions: np.ndarray,
    alive_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of pairwise inter-walker distances per step (vectorized)."""
    x = np.asarray(positions, dtype=float)
    alive = np.asarray(alive_mask, dtype=bool)
    if x.ndim != 3 or alive.ndim != 2:
        msg = "Expected positions [time, walkers, dim] and alive_mask [time, walkers]."
        raise ValueError(msg)

    n_steps = min(x.shape[0], alive.shape[0])
    x = x[:n_steps]
    alive = alive[:n_steps]
    n_walkers = int(x.shape[1])
    if n_walkers < 2:
        return np.full(n_steps, np.nan), np.full(n_steps, np.nan)

    sq_norm = np.sum(x * x, axis=-1)
    gram = np.einsum("tid,tjd->tij", x, x)
    dist_sq = sq_norm[:, :, None] + sq_norm[:, None, :] - 2.0 * gram
    np.maximum(dist_sq, 0.0, out=dist_sq)
    distances = np.sqrt(dist_sq)

    valid = alive[:, :, None] & alive[:, None, :]
    upper = np.triu(np.ones((n_walkers, n_walkers), dtype=bool), k=1)
    valid &= upper[None, :, :]

    counts = valid.sum(axis=(1, 2)).astype(float)
    sums = np.where(valid, distances, 0.0).sum(axis=(1, 2))
    means = np.divide(
        sums,
        counts,
        out=np.full(n_steps, np.nan, dtype=float),
        where=counts > 0,
    )

    sq_sums = np.where(valid, distances * distances, 0.0).sum(axis=(1, 2))
    second_moment = np.divide(
        sq_sums,
        counts,
        out=np.full(n_steps, np.nan, dtype=float),
        where=counts > 0,
    )
    variances = np.maximum(second_moment - means * means, 0.0)
    stds = np.sqrt(variances)
    return means, stds


def _build_companion_distance_plot(
    *,
    step: np.ndarray,
    clone_mean: np.ndarray,
    clone_p95: np.ndarray,
    random_mean: np.ndarray,
    random_p95: np.ndarray,
) -> hv.Overlay | hv.Text:
    clone_err = np.clip(np.asarray(clone_p95) - np.asarray(clone_mean), 0.0, None)
    random_err = np.clip(np.asarray(random_p95) - np.asarray(random_mean), 0.0, None)

    clone_df = pd.DataFrame({"step": step, "mean": clone_mean, "err95": clone_err}).replace(
        [np.inf, -np.inf], np.nan
    )
    random_df = pd.DataFrame({"step": step, "mean": random_mean, "err95": random_err}).replace(
        [np.inf, -np.inf], np.nan
    )
    clone_df = clone_df.dropna()
    random_df = random_df.dropna()

    overlays: list[Any] = []
    if not clone_df.empty:
        overlays.append(
            hv.ErrorBars(clone_df, "step", ["mean", "err95"])
            .relabel("Clone p95")
            .opts(color="#e45756", alpha=0.4, line_width=1)
        )
        overlays.append(
            hv.Curve(clone_df, "step", "mean")
            .relabel("Clone mean")
            .opts(color="#e45756", line_width=2)
        )
    if not random_df.empty:
        overlays.append(
            hv.ErrorBars(random_df, "step", ["mean", "err95"])
            .relabel("Random p95")
            .opts(color="#4c78a8", alpha=0.4, line_width=1)
        )
        overlays.append(
            hv.Curve(random_df, "step", "mean")
            .relabel("Random mean")
            .opts(color="#4c78a8", line_width=2)
        )

    if not overlays:
        return _algorithm_placeholder_plot("No companion-distance data available")

    plot = overlays[0]
    for overlay in overlays[1:]:
        plot = plot * overlay
    return plot.opts(
        title="Companion Distances Over Time (mean with p95 error bars)",
        xlabel="Recorded step",
        ylabel="Distance",
        width=960,
        height=340,
        legend_position="top_left",
        show_grid=True,
    )


def _build_algorithm_diagnostics(history: RunHistory) -> dict[str, Any]:
    """Build vectorized algorithm diagnostics from collected run traces."""
    alive = _to_numpy(history.alive_mask).astype(bool, copy=False)
    will_clone = _to_numpy(history.will_clone).astype(bool, copy=False)
    fitness = _to_numpy(history.fitness).astype(float, copy=False)
    rewards = _to_numpy(history.rewards).astype(float, copy=False)
    x_pre = _to_numpy(history.x_before_clone).astype(float, copy=False)
    companions_clone = _to_numpy(history.companions_clone).astype(np.int64, copy=False)
    companions_random = _to_numpy(history.companions_distance).astype(np.int64, copy=False)

    n_steps = min(
        alive.shape[0],
        will_clone.shape[0],
        fitness.shape[0],
        rewards.shape[0],
        x_pre.shape[0],
        companions_clone.shape[0],
        companions_random.shape[0],
    )
    if n_steps <= 0:
        msg = "No transition frames found in RunHistory."
        raise ValueError(msg)

    alive = alive[:n_steps]
    will_clone = will_clone[:n_steps]
    fitness = fitness[:n_steps]
    rewards = rewards[:n_steps]
    x_pre = x_pre[:n_steps]
    companions_clone = companions_clone[:n_steps]
    companions_random = companions_random[:n_steps]

    if x_pre.ndim != 3:
        msg = "Expected x_before_clone with shape [time, walkers, dim]."
        raise ValueError(msg)

    step = _history_transition_steps(history, n_steps)
    n_walkers = int(x_pre.shape[1])
    companions_clone = np.clip(companions_clone, 0, n_walkers - 1)
    companions_random = np.clip(companions_random, 0, n_walkers - 1)
    time_index = np.arange(n_steps, dtype=np.int64)[:, None]

    clone_dist = np.linalg.norm(
        x_pre - x_pre[time_index, companions_clone],
        axis=-1,
    )
    random_dist = np.linalg.norm(
        x_pre - x_pre[time_index, companions_random],
        axis=-1,
    )

    alive_counts = alive.sum(axis=1).astype(float)
    clone_counts = np.logical_and(will_clone, alive).sum(axis=1).astype(float)
    clone_pct = np.divide(
        100.0 * clone_counts,
        alive_counts,
        out=np.zeros_like(alive_counts),
        where=alive_counts > 0,
    )

    fit_mean, fit_p95 = _compute_masked_mean_p95(fitness, alive)
    rew_mean, rew_p95 = _compute_masked_mean_p95(rewards, alive)
    clone_dist_mean, clone_dist_p95 = _compute_masked_mean_p95(clone_dist, alive)
    random_dist_mean, random_dist_p95 = _compute_masked_mean_p95(random_dist, alive)

    clone_frame = pd.DataFrame({"step": step, "pct_clone": clone_pct}).replace(
        [np.inf, -np.inf], np.nan
    )
    clone_frame = clone_frame.dropna()
    if clone_frame.empty:
        clone_plot: hv.Curve | hv.Text = _algorithm_placeholder_plot("No clone data available")
    else:
        clone_plot = hv.Curve(clone_frame, "step", "pct_clone").opts(
            title="Percentage of Clones Over Time",
            xlabel="Recorded step",
            ylabel="% cloned (alive)",
            width=960,
            height=300,
            color="#f58518",
            line_width=2,
            ylim=(0, 100),
            show_grid=True,
        )

    fitness_plot = _build_timeseries_mean_p95_plot(
        step=step,
        mean=fit_mean,
        p95=fit_p95,
        title="Mean Fitness Over Time (p95 error bars)",
        ylabel="Fitness",
        color="#4c78a8",
    )
    reward_plot = _build_timeseries_mean_p95_plot(
        step=step,
        mean=rew_mean,
        p95=rew_p95,
        title="Mean Reward Over Time (p95 error bars)",
        ylabel="Reward",
        color="#72b7b2",
    )
    companion_plot = _build_companion_distance_plot(
        step=step,
        clone_mean=clone_dist_mean,
        clone_p95=clone_dist_p95,
        random_mean=random_dist_mean,
        random_p95=random_dist_p95,
    )
    inter_mean, inter_std = _compute_interwalker_distance_stats(x_pre, alive)
    interwalker_plot = _build_timeseries_mean_error_plot(
        step=step,
        mean=inter_mean,
        error=inter_std,
        title="Average Inter-Walker Distance Over Time (mean ± 1σ)",
        ylabel="Pairwise distance",
        color="#54a24b",
    )

    lyapunov = _compute_vectorized_lyapunov(history)
    lyapunov_plot = build_lyapunov_plot(
        lyapunov["time"],
        lyapunov["V_total"],
        lyapunov["V_var_x"],
        lyapunov["V_var_v"],
    )

    return {
        "clone_plot": clone_plot,
        "fitness_plot": fitness_plot,
        "reward_plot": reward_plot,
        "companion_plot": companion_plot,
        "interwalker_plot": interwalker_plot,
        "lyapunov_plot": lyapunov_plot,
        "n_transition_steps": int(n_steps),
        "n_lyapunov_steps": int(len(lyapunov["time"])),
    }


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


def _build_single_scale_anchor_row(
    label: str,
    scale: float,
    masses: dict[str, float],
    r2s: dict[str, float] | None = None,
    closest_refs: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build one anchored row using a single scale applied to all masses."""
    r2s = r2s or {}
    closest_refs = closest_refs or {}
    row: dict[str, Any] = {
        "anchor": label,
        "scale_GeV_per_alg": scale,
    }
    for key, alg_mass in masses.items():
        pred = alg_mass * scale
        row[f"{key}_pred_GeV"] = pred
        if key in closest_refs:
            row[f"closest_{key}"] = _format_closest(pred, closest_refs[key])
        key_r2 = r2s.get(key)
        if key_r2 is not None and np.isfinite(key_r2):
            row[f"{key}_r2"] = key_r2
    return row


def _build_family_fixed_anchor_rows(
    masses: dict[str, float],
    anchors: list[tuple[str, float, str]],
    r2s: dict[str, float] | None = None,
    closest_refs: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Build one anchored row with one fixed scale per family."""
    r2s = r2s or {}
    closest_refs = closest_refs or {}

    families = sorted({family for _, _, family in anchors})
    scales_by_family: dict[str, float] = {}
    for family in families:
        family_anchors = [anchor for anchor in anchors if anchor[2] == family]
        scale = _best_fit_scale(masses, family_anchors)
        if scale is not None:
            scales_by_family[family] = scale

    if not scales_by_family:
        return []

    global_scale = _best_fit_scale(masses, anchors)
    row: dict[str, Any] = {
        "anchor": "family-fixed",
        "scale_GeV_per_alg": None,
    }
    for family, scale in scales_by_family.items():
        row[f"{family}_scale_GeV_per_alg"] = scale

    for key, alg_mass in masses.items():
        scale = scales_by_family.get(key)
        if scale is None:
            scale = global_scale
        if scale is None:
            continue
        pred = alg_mass * scale
        row[f"{key}_pred_GeV"] = pred
        if key in closest_refs:
            row[f"closest_{key}"] = _format_closest(pred, closest_refs[key])
        key_r2 = r2s.get(key)
        if key_r2 is not None and np.isfinite(key_r2):
            row[f"{key}_r2"] = key_r2
    return [row]


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
    anchor_mode: str = "per_anchor_row",
) -> list[dict[str, Any]]:
    """Build anchored mass table with selectable calibration mode.

    Modes:
    - ``per_anchor_row``: one scale per row/anchor.
    - ``family_fixed``: one fitted scale per family (baryon/meson/...) in one row.
    - ``global_fixed``: one fitted scale over all anchors in one row.
    """
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

    mode = str(anchor_mode).strip().lower()
    if mode not in {"per_anchor_row", "family_fixed", "global_fixed"}:
        mode = "per_anchor_row"
    if mode == "family_fixed":
        return _build_family_fixed_anchor_rows(masses, anchors, r2s, closest_refs)
    if mode == "global_fixed":
        scale = _best_fit_scale(masses, anchors)
        if scale is None:
            return []
        return [
            _build_single_scale_anchor_row(
                "global-fixed",
                scale,
                masses,
                r2s=r2s,
                closest_refs=closest_refs,
            )
        ]
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
    if mode == "Best Window":
        best_window = result.mass_fit.get("best_window", {})
        return best_window.get("mass", 0.0)
    return result.mass_fit.get("mass", 0.0)


def _get_channel_mass_error(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    if mode == "Best Window":
        return float("nan")
    mass_error = float(result.mass_fit.get("mass_error", float("inf")))
    return mass_error if np.isfinite(mass_error) and mass_error >= 0 else float("inf")


def _get_channel_r2(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    if mode == "Best Window":
        best_window = result.mass_fit.get("best_window", {})
        return best_window.get("r2", float("nan"))
    return result.mass_fit.get("r_squared", float("nan"))


def _resolve_base_mass_mode(mode: str) -> str:
    """Map display mode to an underlying fit mode."""
    return "Best Window" if mode == "Best Window" else "AIC-Weighted"


def _select_mode_result(
    mode_results: dict[str, ChannelCorrelatorResult] | None,
    *,
    prefix: str,
) -> ChannelCorrelatorResult | None:
    """Select p0-like mode result by prefix, falling back to lowest available mode."""
    if not mode_results:
        return None
    p0_name = f"{prefix}0"
    if p0_name in mode_results:
        return mode_results[p0_name]

    def _mode_key(name: str) -> int:
        if not name.startswith(prefix):
            return 10**9
        raw = name[len(prefix) :]
        if "_" in raw:
            raw = raw.split("_", 1)[0]
        try:
            return int(raw)
        except ValueError:
            return 10**9

    candidates = [name for name in mode_results if name.startswith(prefix)]
    if not candidates:
        return None
    first = min(candidates, key=_mode_key)
    return mode_results.get(first)


def _compute_tensor_correction_payload(
    *,
    results: dict[str, ChannelCorrelatorResult],
    tensor_strong_result: ChannelCorrelatorResult | None,
    tensor_momentum_results: dict[str, ChannelCorrelatorResult] | None,
    mode: str,
) -> dict[str, Any]:
    """Compute tensor-channel calibration/correction from multiple estimators.

    Uses a vectorized weighted-consensus over:
    - anisotropic-edge tensor
    - anisotropic-edge tensor_traceless
    - strong-force tensor
    - tensor momentum p0 (or lowest available mode)
    """
    base_mode = _resolve_base_mass_mode(mode)
    momentum_p0 = _select_mode_result(
        tensor_momentum_results,
        prefix="tensor_momentum_p",
    )

    estimators: list[tuple[str, ChannelCorrelatorResult | None]] = [
        ("tensor", results.get("tensor")),
        ("tensor_traceless", results.get("tensor_traceless")),
        ("tensor_strong_force", tensor_strong_result),
        ("tensor_momentum_p0", momentum_p0),
    ]
    labels = np.asarray([name for name, _ in estimators], dtype=object)
    masses = np.asarray(
        [
            _get_channel_mass(result, base_mode)
            if result is not None and result.n_samples > 0
            else np.nan
            for _, result in estimators
        ],
        dtype=float,
    )
    errors = np.asarray(
        [
            _get_channel_mass_error(result, base_mode)
            if result is not None and result.n_samples > 0
            else np.nan
            for _, result in estimators
        ],
        dtype=float,
    )
    valid = np.isfinite(masses) & (masses > 0)
    weighted = valid & np.isfinite(errors) & (errors > 0)

    consensus_mass = float("nan")
    consensus_err = float("nan")
    if bool(np.any(weighted)):
        mass_w = masses[weighted]
        err_w = errors[weighted]
        weights = 1.0 / np.maximum(err_w, 1e-12) ** 2
        consensus_mass = float(np.sum(weights * mass_w) / np.sum(weights))
        consensus_err = float(np.sqrt(1.0 / np.sum(weights)))
    elif bool(np.any(valid)):
        mass_v = masses[valid]
        consensus_mass = float(np.mean(mass_v))
        if mass_v.size > 1:
            consensus_err = float(np.std(mass_v, ddof=1) / np.sqrt(float(mass_v.size)))

    spread = (
        float(np.std(masses[valid], ddof=1))
        if int(np.count_nonzero(valid)) > 1
        else 0.0
    )

    base_idx = int(np.where(labels == "tensor")[0][0])
    if not bool(valid[base_idx]) and bool(np.any(valid)):
        base_idx = int(np.flatnonzero(valid)[0])
    base_mass = masses[base_idx] if bool(valid[base_idx]) else float("nan")
    base_label = str(labels[base_idx])

    correction_scale = float("nan")
    correction_scale_err = float("nan")
    if np.isfinite(consensus_mass) and consensus_mass > 0 and np.isfinite(base_mass) and base_mass > 0:
        correction_scale = float(consensus_mass / base_mass)
        base_err = errors[base_idx]
        if np.isfinite(consensus_err) and np.isfinite(base_err):
            rel_cons = consensus_err / max(consensus_mass, 1e-12)
            rel_base = base_err / max(base_mass, 1e-12)
            correction_scale_err = abs(correction_scale) * float(np.sqrt(rel_cons**2 + rel_base**2))

    per_estimator_scale = np.full_like(masses, np.nan, dtype=float)
    per_estimator_scale[valid & np.isfinite(consensus_mass) & (consensus_mass > 0)] = (
        consensus_mass / masses[valid & np.isfinite(consensus_mass) & (consensus_mass > 0)]
    )

    return {
        "base_mode": base_mode,
        "labels": labels,
        "masses": masses,
        "errors": errors,
        "valid_mask": valid,
        "weighted_mask": weighted,
        "consensus_mass": consensus_mass,
        "consensus_err": consensus_err,
        "spread": spread,
        "base_label": base_label,
        "base_mass": base_mass,
        "correction_scale": correction_scale,
        "correction_scale_err": correction_scale_err,
        "per_estimator_scale": per_estimator_scale,
    }


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
    ("tensor_traceless", "tensor", "new spin-2 / legacy tensor"),
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
    correlator_logy: bool = True,
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

        channel_plot = ChannelPlot(result, logy=correlator_logy, width=400, height=350)
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
    overlay_corr_pane.object = build_all_channels_overlay(
        results,
        plot_type="correlator",
        correlator_logy=correlator_logy,
    )
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
    anchor_mode: str = "per_anchor_row",
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
        channel_masses,
        glueball_ref,
        sqrt_sigma_ref=None,
        r2s=channel_r2,
        anchor_mode=anchor_mode,
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
    # Load JS extensions before any document/template objects are built.
    # Calling this inside onload can race with frontend model initialization,
    # leaving window.Plotly undefined in Panel resize handlers.
    pn.extension("plotly", "tabulator")

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

        _debug("building config + visualizer")
        gas_config = GasConfigPanel.create_qft_config(spatial_dims=2, bounds_extent=12.0)
        gas_config.hide_viscous_kernel_widgets = False
        gas_config.benchmark_name = "Riemannian Mix"
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
        gas_config.kinetic_op.delta_t = 0.01
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
        gas_config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel_volume"
        gas_config.kinetic_op.viscous_neighbor_threshold = None
        gas_config.kinetic_op.viscous_neighbor_penalty = 0.0
        gas_config.kinetic_op.viscous_degree_cap = None
        gas_config.kinetic_op.use_velocity_squashing = True

        # Companion selection (diversity + cloning).
        gas_config.companion_selection.method = "random_pairing"
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
            "anisotropic_edge_glueball_strong_result": None,
            "anisotropic_edge_glueball_edge_iso_result": None,
            "anisotropic_edge_glueball_su3_result": None,
            "anisotropic_edge_glueball_momentum_results": None,
            "anisotropic_edge_glueball_systematics_error": None,
            "anisotropic_edge_glueball_spatial_result": None,
            "anisotropic_edge_glueball_spatial_frame_idx": None,
            "anisotropic_edge_glueball_spatial_rho_edge": None,
            "anisotropic_edge_glueball_spatial_error": None,
            "anisotropic_edge_tensor_spatial_result": None,
            "anisotropic_edge_tensor_spatial_frame_idx": None,
            "anisotropic_edge_tensor_spatial_rho_edge": None,
            "anisotropic_edge_tensor_spatial_error": None,
            "anisotropic_edge_tensor_strong_result": None,
            "anisotropic_edge_tensor_momentum_results": None,
            "anisotropic_edge_tensor_momentum_meta": None,
            "anisotropic_edge_tensor_systematics_error": None,
            "anisotropic_edge_tensor_traceless_spatial_result": None,
            "anisotropic_edge_tensor_traceless_spatial_frame_idx": None,
            "anisotropic_edge_tensor_traceless_spatial_rho_edge": None,
            "anisotropic_edge_tensor_traceless_spatial_error": None,
            "new_dirac_ew_bundle": None,
            "new_dirac_ew_results": None,
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

        algorithm_status = pn.pane.Markdown(
            "**Algorithm:** run a simulation or load a RunHistory, then click Run Algorithm Analysis.",
            sizing_mode="stretch_width",
        )
        algorithm_run_button = pn.widgets.Button(
            name="Run Algorithm Analysis",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )
        algorithm_clone_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show clone percentage."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_fitness_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show fitness trend."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_reward_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show reward trend."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_companion_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show companion distances."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_interwalker_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show inter-walker distances."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_lyapunov_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show Lyapunov convergence."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )

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
                "bulk_fraction", "scalar_density_mode", "knn_k",
                "coarse_grain_bins", "coarse_grain_min_points",
                "temporal_average_enabled", "temporal_window_frames", "temporal_stride",
                "bootstrap_samples", "bootstrap_confidence", "bootstrap_seed",
                "bootstrap_frame_block_size",
                "g_newton_metric", "g_newton_manual",
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
        channel_anchor_mode = pn.widgets.RadioButtonGroup(
            name="Anchor Calibration",
            options={
                "Per-anchor rows": "per_anchor_row",
                "Family-fixed scales": "family_fixed",
                "Global fixed scale": "global_fixed",
            },
            value="per_anchor_row",
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
        anisotropic_edge_baryon_panel = pn.Param(
            anisotropic_edge_settings,
            parameters=[
                "use_companion_baryon_triplet",
                "baryon_use_connected",
                "baryon_max_lag",
                "baryon_color_dims_spec",
                "baryon_eps",
            ],
            show_name=False,
            widgets={
                "baryon_max_lag": {"name": "Baryon max lag (blank=use max_lag)"},
                "baryon_color_dims_spec": {"name": "Baryon color dims (3 dims)"},
            },
            default_layout=type("AnisotropicEdgeBaryonGrid", (pn.GridBox,), {"ncols": 1}),
        )
        anisotropic_edge_meson_panel = pn.Param(
            anisotropic_edge_settings,
            parameters=[
                "use_companion_meson_phase",
                "meson_use_connected",
                "meson_max_lag",
                "meson_pair_selection",
                "meson_color_dims_spec",
                "meson_eps",
            ],
            show_name=False,
            widgets={
                "meson_max_lag": {"name": "Meson max lag (blank=use max_lag)"},
                "meson_pair_selection": {"name": "Meson pair selection"},
                "meson_color_dims_spec": {"name": "Meson color dims (3 dims)"},
            },
            default_layout=type("AnisotropicEdgeMesonGrid", (pn.GridBox,), {"ncols": 1}),
        )
        anisotropic_edge_vector_panel = pn.Param(
            anisotropic_edge_settings,
            parameters=[
                "use_companion_vector_meson",
                "vector_meson_use_connected",
                "vector_meson_max_lag",
                "vector_meson_pair_selection",
                "vector_meson_color_dims_spec",
                "vector_meson_position_dims_spec",
                "vector_meson_eps",
                "vector_meson_use_unit_displacement",
            ],
            show_name=False,
            widgets={
                "vector_meson_max_lag": {"name": "Vector max lag (blank=use max_lag)"},
                "vector_meson_pair_selection": {"name": "Vector pair selection"},
                "vector_meson_color_dims_spec": {"name": "Vector color dims (3 dims)"},
                "vector_meson_position_dims_spec": {"name": "Vector position dims (3 dims)"},
            },
            default_layout=type("AnisotropicEdgeVectorGrid", (pn.GridBox,), {"ncols": 1}),
        )
        anisotropic_edge_tensor_momentum_panel = pn.Param(
            anisotropic_edge_settings,
            parameters=[
                "use_companion_tensor_momentum",
                "tensor_momentum_use_connected",
                "tensor_momentum_max_lag",
                "tensor_momentum_pair_selection",
                "tensor_momentum_color_dims_spec",
                "tensor_momentum_position_dims_spec",
                "tensor_momentum_axis",
                "tensor_momentum_mode_max",
                "tensor_momentum_eps",
            ],
            show_name=False,
            widgets={
                "tensor_momentum_max_lag": {"name": "Tensor momentum max lag (blank=use max_lag)"},
                "tensor_momentum_pair_selection": {"name": "Tensor momentum pair selection"},
                "tensor_momentum_color_dims_spec": {"name": "Tensor momentum color dims (3 dims)"},
                "tensor_momentum_position_dims_spec": {
                    "name": "Tensor momentum position dims (3 dims)"
                },
                "tensor_momentum_axis": {"name": "Tensor momentum axis"},
                "tensor_momentum_mode_max": {"name": "Tensor momentum n_max"},
            },
            default_layout=type(
                "AnisotropicEdgeTensorMomentumGrid",
                (pn.GridBox,),
                {"ncols": 1},
            ),
        )
        anisotropic_edge_glueball_panel = pn.Param(
            anisotropic_edge_settings,
            parameters=[
                "use_companion_glueball_color",
                "glueball_use_connected",
                "glueball_max_lag",
                "glueball_color_dims_spec",
                "glueball_eps",
                "glueball_use_action_form",
                "glueball_use_momentum_projection",
                "glueball_momentum_axis",
                "glueball_momentum_mode_max",
            ],
            show_name=False,
            widgets={
                "glueball_max_lag": {"name": "Glueball max lag (blank=use max_lag)"},
                "glueball_color_dims_spec": {"name": "Glueball color dims (3 dims)"},
                "glueball_momentum_axis": {"name": "Momentum axis"},
                "glueball_momentum_mode_max": {"name": "Momentum n_max"},
            },
            default_layout=type("AnisotropicEdgeGlueballGrid", (pn.GridBox,), {"ncols": 1}),
        )
        anisotropic_edge_base_settings_row = pn.Row(
            anisotropic_edge_settings_panel,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_override_settings_row = pn.Row(
            pn.Column(
                pn.pane.Markdown("### Baryon Triplet Settings"),
                anisotropic_edge_baryon_panel,
                pn.layout.Divider(),
                pn.pane.Markdown("### Meson Phase Settings"),
                anisotropic_edge_meson_panel,
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.pane.Markdown("### Vector Meson Settings"),
                anisotropic_edge_vector_panel,
                pn.layout.Divider(),
                pn.pane.Markdown("### Tensor Momentum Settings"),
                anisotropic_edge_tensor_momentum_panel,
                pn.layout.Divider(),
                pn.pane.Markdown("### Glueball Color Settings"),
                anisotropic_edge_glueball_panel,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_settings_layout = pn.Column(
            pn.pane.Markdown("### Base Channel Settings"),
            anisotropic_edge_base_settings_row,
            pn.layout.Divider(),
            pn.pane.Markdown("### Override Parameters"),
            anisotropic_edge_override_settings_row,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_summary = pn.pane.Markdown(
            "## Anisotropic Edge Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window", "Tensor-Corrected"],
            value="AIC-Weighted",
            button_type="default",
        )
        anisotropic_edge_anchor_mode = pn.widgets.RadioButtonGroup(
            name="Anchor Calibration",
            options={
                "Per-anchor rows": "per_anchor_row",
                "Family-fixed scales": "family_fixed",
                "Global fixed scale": "global_fixed",
            },
            value="per_anchor_row",
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
        anisotropic_edge_glueball_compare_ratio = pn.pane.Markdown(
            (
                "**Glueball Cross-Check (Edge-Isotropic / Strong-Isotropic):** "
                "run anisotropic channels to compare."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_lorentz_ratio = pn.pane.Markdown(
            (
                "**Glueball Lorentz Check (4 Estimators + Momentum Dispersion):** "
                "run anisotropic channels to compare all glueball estimators and fit `E(p)^2`."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_approach_summary = pn.pane.Markdown(
            (
                "**Glueball Approach Comparison:** run anisotropic channels to compare "
                "strong-force isotropic, anisotropic-edge isotropic, SU(3) plaquette, "
                "and momentum-projected estimates."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_approach_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_pairwise_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_consensus_summary = pn.pane.Markdown(
            (
                "**Glueball Consensus / Systematics:** run anisotropic channels "
                "to compute weighted consensus and discrepancy-based systematics."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_systematics_badge = pn.pane.Alert(
            "Systematics verdict: run anisotropic channels to evaluate agreement.",
            alert_type="secondary",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_consensus_plot = pn.Column(
            pn.pane.Markdown("_Run anisotropic channels to populate estimator comparison plot._"),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_dispersion_plot = pn.Column(
            pn.pane.Markdown(
                "_Run anisotropic channels with glueball momentum projection to populate this plot._"
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_systematics_badge = pn.pane.Alert(
            "Tensor systematics verdict: run anisotropic channels to evaluate agreement.",
            alert_type="secondary",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_approach_summary = pn.pane.Markdown(
            (
                "**Tensor Approach Comparison:** run anisotropic channels to compare "
                "anisotropic-edge tensor, legacy strong-force tensor, and momentum-projected tensor."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_approach_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_dispersion_plot = pn.Column(
            pn.pane.Markdown(
                "_Run anisotropic channels with tensor momentum projection to populate this plot._"
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_component_dispersion_plot = pn.Column(
            pn.pane.Markdown(
                "_Run anisotropic channels with tensor momentum projection to populate component dispersions._"
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_lorentz_ratio = pn.pane.Markdown(
            (
                "**Tensor Lorentz Check (Legacy isotropic tensor):** "
                "spatial tensor check is disabled in this tab."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_tensor_traceless_lorentz_ratio = pn.pane.Markdown(
            (
                "**Tensor Lorentz Check (Traceless tensor):** "
                "spatial tensor check is disabled in this tab."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_lorentz_correction_summary = pn.pane.Markdown(
            (
                "**Lorentz/Anisotropy Correction Factors (Glueball 4-way):** "
                "run anisotropic channels to compute consistency and estimator correction factors."
            ),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_aniso_plot = pn.Column(
            pn.pane.Markdown("_Run anisotropic channels to populate this plot._"),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_strong_plot = pn.Column(
            pn.pane.Markdown("_Run anisotropic channels to populate this plot._"),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_plaquette_plot = pn.Column(
            pn.pane.Markdown("_Run anisotropic channels to populate this plot._"),
            sizing_mode="stretch_width",
        )
        anisotropic_edge_glueball_momentum_p0_plot = pn.Column(
            pn.pane.Markdown("_Run anisotropic channels to populate this plot._"),
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
        # New Dirac/Electroweak unified tab components
        # =====================================================================
        new_dirac_ew_settings = NewDiracElectroweakSettings()
        new_dirac_ew_status = pn.pane.Markdown(
            "**Electroweak:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        new_dirac_ew_run_button = pn.widgets.Button(
            name="Compute Electroweak",
            button_type="primary",
            min_width=260,
            sizing_mode="stretch_width",
            disabled=True,
        )
        new_dirac_ew_summary = pn.pane.Markdown(
            "## Electroweak Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        new_dirac_ew_settings_panel = pn.Param(
            new_dirac_ew_settings,
            parameters=[
                "simulation_range",
                "max_lag",
                "h_eff",
                "mc_time_index",
                "use_connected",
                "neighbor_method",
                "companion_topology",
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
                "dirac_kernel_mode",
                "dirac_time_average",
                "dirac_warmup_fraction",
                "dirac_max_avg_frames",
                "dirac_color_threshold_mode",
                "dirac_color_threshold_value",
                "color_singlet_quantile",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC time slice (step or idx; blank=last)"},
                "dirac_color_threshold_mode": {"name": "Dirac color threshold"},
                "dirac_color_threshold_value": {"name": "||F_visc|| threshold"},
            },
            default_layout=type("NewDiracEWSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        new_dirac_ew_coupling_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_coupling_ref_table = pn.widgets.Tabulator(
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

        new_dirac_ew_channel_plots = pn.Column(sizing_mode="stretch_width")
        new_dirac_ew_plots_overlay_corr = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_plots_overlay_meff = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )
        new_dirac_ew_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_ratio_pane = pn.pane.Markdown(
            "**Electroweak Ratios:** Compute channels to see ratios.",
            sizing_mode="stretch_width",
        )
        new_dirac_ew_ratio_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_compare_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_anchor_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_ref_table = pn.widgets.Tabulator(
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

        new_dirac_ew_dirac_full = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_dirac_walker = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_dirac_up = pn.pane.HoloViews(linked_axes=False)
        new_dirac_ew_dirac_down = pn.pane.HoloViews(linked_axes=False)
        new_dirac_ew_dirac_nu = pn.pane.HoloViews(linked_axes=False)
        new_dirac_ew_dirac_lep = pn.pane.HoloViews(linked_axes=False)
        new_dirac_ew_dirac_mass_hierarchy = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_dirac_chiral = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_dirac_generation_ratios = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_dirac_summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        new_dirac_ew_dirac_comparison = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_dirac_ratio = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        new_dirac_ew_color_singlet_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        new_dirac_ew_electron_plot = pn.Column(sizing_mode="stretch_width")
        new_dirac_ew_sigma_plot = pn.Column(sizing_mode="stretch_width")
        new_dirac_ew_observable_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_ratio_table_extra = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )

        new_dirac_ew_observed_table = pn.widgets.Tabulator(
            pd.DataFrame(
                {
                    "observable": [
                        "electron_dirac",
                        "electron_yukawa",
                        "electron_component",
                        "u1_phase",
                        "u1_dressed",
                        "su2_phase",
                        "su2_doublet",
                        "ew_mixed",
                        "higgs_sigma",
                    ],
                    "observed_GeV": [
                        "0.000511",
                        "0.000511",
                        "0.000511",
                        "0.000511",
                        "0.105658",
                        "80.379",
                        "91.1876",
                        "1.77686",
                        "125.10",
                    ],
                }
            ),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors={"observed_GeV": "input"},
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
            algorithm_status.object = (
                "**Algorithm ready:** click Run Algorithm Analysis."
            )
            algorithm_run_button.disabled = False
            algorithm_clone_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show clone percentage."
            )
            algorithm_fitness_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show fitness trend."
            )
            algorithm_reward_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show reward trend."
            )
            algorithm_companion_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show companion distances."
            )
            algorithm_interwalker_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show inter-walker distances."
            )
            algorithm_lyapunov_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show Lyapunov convergence."
            )
            _set_analysis_status("**Analysis ready:** click Run Analysis.")
            run_analysis_button.disabled = False
            run_analysis_button_main.disabled = False
            save_button.disabled = False
            save_status.object = "**Save a history**: choose a path and click Save."
            # Enable fractal set tab
            fractal_set_run_button.disabled = False
            fractal_set_status.object = "**Holographic Principle ready:** click Compute Fractal Set."
            # Enable einstein test
            einstein_run_button.disabled = False
            einstein_status.object = (
                "**Einstein Test ready:** run Holographic Principle for G_N, then click."
            )
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
            # Enable unified dirac/electroweak tab
            new_dirac_ew_run_button.disabled = False
            new_dirac_ew_status.object = "**Electroweak ready:** click Compute Electroweak."
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

        def _update_algorithm_outputs(history: RunHistory) -> None:
            try:
                diagnostics = _build_algorithm_diagnostics(history)
            except Exception as exc:
                algorithm_status.object = f"**Algorithm error:** {exc!s}"
                fallback = _algorithm_placeholder_plot("Algorithm diagnostics unavailable.")
                algorithm_clone_plot.object = fallback
                algorithm_fitness_plot.object = fallback
                algorithm_reward_plot.object = fallback
                algorithm_companion_plot.object = fallback
                algorithm_interwalker_plot.object = fallback
                algorithm_lyapunov_plot.object = fallback
                return

            algorithm_clone_plot.object = diagnostics["clone_plot"]
            algorithm_fitness_plot.object = diagnostics["fitness_plot"]
            algorithm_reward_plot.object = diagnostics["reward_plot"]
            algorithm_companion_plot.object = diagnostics["companion_plot"]
            algorithm_interwalker_plot.object = diagnostics["interwalker_plot"]
            algorithm_lyapunov_plot.object = diagnostics["lyapunov_plot"]
            algorithm_status.object = (
                "**Algorithm diagnostics updated:** "
                f"{diagnostics['n_transition_steps']} transition frames, "
                f"{diagnostics['n_lyapunov_steps']} Lyapunov frames."
            )

        def on_run_analysis(_):
            result = _run_analysis()
            if result is None:
                return
            metrics, arrays = result
            _update_analysis_outputs(metrics, arrays)
            _set_analysis_status("**Analysis complete.**")

        def on_run_algorithm_analysis(_):
            history = state.get("history")
            if history is None:
                algorithm_status.object = "**Error:** load or run a simulation first."
                return
            algorithm_status.object = "**Running algorithm diagnostics...**"
            _update_algorithm_outputs(history)

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
                    scalar_density_mode=einstein_settings.scalar_density_mode,
                    knn_k=einstein_settings.knn_k,
                    coarse_grain_bins=einstein_settings.coarse_grain_bins,
                    coarse_grain_min_points=einstein_settings.coarse_grain_min_points,
                    temporal_average_enabled=einstein_settings.temporal_average_enabled,
                    temporal_window_frames=einstein_settings.temporal_window_frames,
                    temporal_stride=einstein_settings.temporal_stride,
                    bootstrap_samples=einstein_settings.bootstrap_samples,
                    bootstrap_confidence=einstein_settings.bootstrap_confidence,
                    bootstrap_seed=einstein_settings.bootstrap_seed,
                    bootstrap_frame_block_size=einstein_settings.bootstrap_frame_block_size,
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
                coarse_status = (
                    f"Coarse R\u00b2={result.scalar_r2_coarse:.4f}, "
                    if result.scalar_r2_coarse is not None
                    else ""
                )
                temporal_status = (
                    f"temporal={result.temporal_frame_count} frame(s), "
                    if result.temporal_average_enabled
                    else ""
                )
                bootstrap_status = (
                    f"bootstrap={result.scalar_bootstrap_samples}, "
                    if result.scalar_bootstrap_samples > 0
                    else ""
                )
                einstein_status.object = (
                    f"**Complete:** {result.n_walkers} walkers, d={result.spatial_dim}. "
                    f"Ricci source={result.ricci_scalar_source}, "
                    f"density={result.scalar_density_mode}, "
                    f"Scalar R\u00b2={result.scalar_r2:.4f}, "
                    f"{temporal_status}"
                    f"{bootstrap_status}"
                    f"{coarse_status}"
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
            anchor_mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = channel_mass_mode.value
            if anchor_mode is None:
                anchor_mode = channel_anchor_mode.value
            _update_strong_tables(
                results,
                mode,
                channel_mass_table,
                channel_ratio_pane,
                channel_fit_table,
                channel_anchor_table,
                channel_glueball_ref_input,
                anchor_mode=str(anchor_mode),
            )

        def _on_channel_mass_mode_change(event):
            """Handle mass mode toggle changes - updates all tables."""
            if "channel_results" in state:
                _update_channel_tables(state["channel_results"], event.new)

        channel_mass_mode.param.watch(_on_channel_mass_mode_change, "value")

        def _on_channel_anchor_mode_change(_event):
            if "channel_results" in state:
                _update_channel_tables(state["channel_results"])

        channel_anchor_mode.param.watch(_on_channel_anchor_mode_change, "value")

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
                correlator_logy=False,
            )

        def _update_anisotropic_edge_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
            anchor_mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = anisotropic_edge_mass_mode.value
            if anchor_mode is None:
                anchor_mode = anisotropic_edge_anchor_mode.value
            display_mode = str(mode)
            table_mode = _resolve_base_mass_mode(display_mode)
            mass_getter = None
            error_getter = None

            tensor_payload = _compute_tensor_correction_payload(
                results=results,
                tensor_strong_result=state.get("anisotropic_edge_tensor_strong_result"),
                tensor_momentum_results=state.get("anisotropic_edge_tensor_momentum_results"),
                mode=table_mode,
            )
            state["anisotropic_edge_tensor_correction_payload"] = tensor_payload
            state["anisotropic_edge_tensor_correction_applied"] = False

            if display_mode == "Tensor-Corrected":
                correction_scale = float(tensor_payload.get("correction_scale", float("nan")))
                if np.isfinite(correction_scale) and correction_scale > 0:

                    def mass_getter(result, _mode):
                        raw_mass = _get_channel_mass(result, table_mode)
                        if not np.isfinite(raw_mass) or raw_mass <= 0:
                            return raw_mass
                        return raw_mass * correction_scale

                    def error_getter(result, _mode):
                        raw_error = _get_channel_mass_error(result, table_mode)
                        if np.isfinite(raw_error) and raw_error >= 0:
                            return abs(correction_scale) * raw_error
                        return raw_error

                    state["anisotropic_edge_tensor_correction_applied"] = True

            _update_strong_tables(
                results,
                table_mode,
                anisotropic_edge_mass_table,
                anisotropic_edge_ratio_pane,
                anisotropic_edge_fit_table,
                anisotropic_edge_anchor_table,
                anisotropic_edge_glueball_ref_input,
                mass_getter=mass_getter,
                error_getter=error_getter,
                ratio_specs=ANISOTROPIC_EDGE_RATIO_SPECS,
                anchor_mode=str(anchor_mode),
            )

        def _update_anisotropic_edge_glueball_crosscheck(
            results: dict[str, ChannelCorrelatorResult],
            strong_result: ChannelCorrelatorResult | None,
            edge_iso_result: ChannelCorrelatorResult | None,
            su3_result: ChannelCorrelatorResult | None,
            momentum_results: dict[str, ChannelCorrelatorResult] | None,
            spatial_result: ChannelCorrelatorResult | None,
            history: RunHistory | None,
            spatial_frame_idx: int | None,
            spatial_rho_edge: float | None,
            spatial_error: str | None = None,
            tensor_spatial_result: ChannelCorrelatorResult | None = None,
            tensor_spatial_frame_idx: int | None = None,
            tensor_spatial_rho_edge: float | None = None,
            tensor_spatial_error: str | None = None,
            tensor_strong_result: ChannelCorrelatorResult | None = None,
            tensor_momentum_results: dict[str, ChannelCorrelatorResult] | None = None,
            tensor_momentum_meta: dict[str, Any] | None = None,
            tensor_systematics_error: str | None = None,
            tensor_traceless_spatial_result: ChannelCorrelatorResult | None = None,
            tensor_traceless_spatial_frame_idx: int | None = None,
            tensor_traceless_spatial_rho_edge: float | None = None,
            tensor_traceless_spatial_error: str | None = None,
            glueball_systematics_error: str | None = None,
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = anisotropic_edge_mass_mode.value
            display_mode = str(mode)
            base_mode = _resolve_base_mass_mode(display_mode)

            tensor_payload_display = _compute_tensor_correction_payload(
                results=results,
                tensor_strong_result=tensor_strong_result,
                tensor_momentum_results=tensor_momentum_results,
                mode=base_mode,
            )
            state["anisotropic_edge_tensor_correction_payload"] = tensor_payload_display
            correction_scale_display = float(
                tensor_payload_display.get("correction_scale", float("nan"))
            )
            apply_tensor_correction = (
                display_mode == "Tensor-Corrected"
                and np.isfinite(correction_scale_display)
                and correction_scale_display > 0
            )

            def _display_mass(result_obj: ChannelCorrelatorResult) -> float:
                mass_value = _get_channel_mass(result_obj, base_mode)
                if (
                    apply_tensor_correction
                    and np.isfinite(mass_value)
                    and mass_value > 0
                ):
                    return mass_value * correction_scale_display
                return mass_value

            def _display_mass_error(result_obj: ChannelCorrelatorResult) -> float:
                mass_err = _get_channel_mass_error(result_obj, base_mode)
                if (
                    apply_tensor_correction
                    and np.isfinite(mass_err)
                    and mass_err >= 0
                ):
                    return abs(correction_scale_display) * mass_err
                return mass_err

            def _display_r2(result_obj: ChannelCorrelatorResult) -> float:
                return _get_channel_r2(result_obj, base_mode)

            if edge_iso_result is not None and edge_iso_result.n_samples > 0:
                edge_plot = ChannelPlot(
                    edge_iso_result,
                    logy=False,
                    width=420,
                    height=320,
                ).side_by_side()
                anisotropic_edge_glueball_aniso_plot.objects = [edge_plot]
            else:
                anisotropic_edge_glueball_aniso_plot.objects = [
                    pn.pane.Markdown("_No anisotropic-edge isotropic glueball result._")
                ]

            if strong_result is not None and strong_result.n_samples > 0:
                strong_plot = ChannelPlot(
                    strong_result,
                    logy=False,
                    width=420,
                    height=320,
                ).side_by_side()
                anisotropic_edge_glueball_strong_plot.objects = [strong_plot]
            else:
                anisotropic_edge_glueball_strong_plot.objects = [
                    pn.pane.Markdown("_No strong-force glueball result._")
                ]

            if su3_result is not None and su3_result.n_samples > 0:
                su3_plot = ChannelPlot(
                    su3_result,
                    logy=False,
                    width=420,
                    height=320,
                ).side_by_side()
                anisotropic_edge_glueball_plaquette_plot.objects = [su3_plot]
            else:
                anisotropic_edge_glueball_plaquette_plot.objects = [
                    pn.pane.Markdown("_No SU(3) plaquette glueball result._")
                ]

            compare_ready = (
                edge_iso_result is not None
                and edge_iso_result.n_samples > 0
                and strong_result is not None
                and strong_result.n_samples > 0
            )
            if not compare_ready:
                anisotropic_edge_glueball_compare_ratio.object = (
                    "**Glueball Cross-Check (Edge-Isotropic / Strong-Isotropic):** "
                    "n/a (missing one or both isotropic estimators)."
                )
            else:
                mass_edge_iso = _display_mass(edge_iso_result)
                mass_strong_iso = _display_mass(strong_result)
                if mass_edge_iso <= 0 or mass_strong_iso <= 0:
                    anisotropic_edge_glueball_compare_ratio.object = (
                        "**Glueball Cross-Check (Edge-Isotropic / Strong-Isotropic):** "
                        "n/a (non-positive fitted mass)."
                    )
                else:
                    ratio = mass_edge_iso / mass_strong_iso
                    delta_pct = (ratio - 1.0) * 100.0
                    r2_edge = _display_r2(edge_iso_result)
                    r2_strong = _display_r2(strong_result)
                    lines = [
                        "**Glueball Cross-Check (Edge-Isotropic / Strong-Isotropic):**",
                        f"- Mass ratio: **{ratio:.4f}**",
                        f"- Δ% vs strong isotropic: `{delta_pct:+.2f}%`",
                        f"- Edge-isotropic mass: `{mass_edge_iso:.6g}`",
                        f"- Strong-isotropic mass: `{mass_strong_iso:.6g}`",
                    ]
                    if np.isfinite(r2_edge):
                        lines.append(f"- Edge-isotropic fit R²: `{r2_edge:.4f}`")
                    if np.isfinite(r2_strong):
                        lines.append(f"- Strong-isotropic fit R²: `{r2_strong:.4f}`")
                    anisotropic_edge_glueball_compare_ratio.object = "  \n".join(lines)

            momentum_axis = int(anisotropic_edge_settings.glueball_momentum_axis)
            momentum_length = float("nan")
            if history is not None and 0 <= momentum_axis < int(history.d):
                momentum_length = _extract_axis_extent_from_bounds(
                    history.bounds, momentum_axis
                )

            momentum_channels = dict(momentum_results or {})
            if not momentum_channels:
                momentum_prefix = "glueball_momentum_p"
                momentum_channels = {
                    name: value
                    for name, value in results.items()
                    if name.startswith(momentum_prefix)
                }
            momentum_p0_result = momentum_channels.get("glueball_momentum_p0")
            if momentum_p0_result is None and momentum_channels:
                def _mode_sort_key(name: str) -> int:
                    try:
                        return int(name[len("glueball_momentum_p") :])
                    except Exception:
                        return 10**9

                first_mode_name = sorted(momentum_channels.keys(), key=_mode_sort_key)[0]
                momentum_p0_result = momentum_channels.get(first_mode_name)
            if momentum_p0_result is not None and momentum_p0_result.n_samples > 0:
                momentum_plot = ChannelPlot(
                    momentum_p0_result,
                    logy=False,
                    width=420,
                    height=320,
                ).side_by_side()
                anisotropic_edge_glueball_momentum_p0_plot.objects = [momentum_plot]
            else:
                anisotropic_edge_glueball_momentum_p0_plot.objects = [
                    pn.pane.Markdown(
                        "_No SU(3) momentum-projected glueball result (`p0`) available._"
                    )
                ]

            def _delta_percent(value: float, reference: float) -> float:
                if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
                    return float("nan")
                return (value / reference - 1.0) * 100.0

            estimator_rows: list[dict[str, Any]] = []
            core_rows: list[dict[str, Any]] = []
            momentum_rows: list[dict[str, Any]] = []

            def _append_estimator_row(
                *,
                estimator: str,
                approach: str,
                channel_name: str,
                result_obj: ChannelCorrelatorResult | None,
                is_core: bool,
                n_mode: int | None = None,
            ) -> None:
                if result_obj is None or result_obj.n_samples <= 0:
                    return
                mass_value = _display_mass(result_obj)
                if not np.isfinite(mass_value) or mass_value <= 0:
                    return
                mass_err = _display_mass_error(result_obj)
                r2_value = _display_r2(result_obj)
                p_value = float("nan")
                p2_value = float("nan")
                if n_mode is not None and np.isfinite(momentum_length) and momentum_length > 0:
                    p_value = (2.0 * np.pi * float(n_mode)) / float(momentum_length)
                    p2_value = p_value * p_value

                row = {
                    "estimator": estimator,
                    "approach": approach,
                    "channel": channel_name,
                    "n_mode": n_mode if n_mode is not None else "",
                    "mass": float(mass_value),
                    "mass_error": float(mass_err) if np.isfinite(mass_err) and mass_err >= 0 else np.nan,
                    "r2": float(r2_value) if np.isfinite(r2_value) else np.nan,
                    "p": p_value,
                    "p2": p2_value,
                    "core_estimator": bool(is_core),
                }
                estimator_rows.append(row)
                if is_core:
                    core_rows.append(row)
                if n_mode is not None and np.isfinite(p2_value):
                    momentum_rows.append(
                        {
                            "estimator": estimator,
                            "channel": channel_name,
                            "n_mode": int(n_mode),
                            "p": float(p_value),
                            "p2": float(p2_value),
                            "mass": float(mass_value),
                            "mass_error": (
                                float(mass_err)
                                if np.isfinite(mass_err) and mass_err >= 0
                                else np.nan
                            ),
                        }
                    )

            _append_estimator_row(
                estimator="Strong-force isotropic",
                approach="isotropic",
                channel_name="glueball_strong_force",
                result_obj=strong_result,
                is_core=True,
            )
            _append_estimator_row(
                estimator="Anisotropic-edge isotropic",
                approach="isotropic",
                channel_name="glueball_edge_isotropic",
                result_obj=edge_iso_result,
                is_core=True,
            )
            _append_estimator_row(
                estimator="SU(3) plaquette",
                approach="su3_plaquette",
                channel_name="glueball_su3_plaquette",
                result_obj=su3_result,
                is_core=True,
            )

            momentum_prefix = "glueball_momentum_p"

            def _momentum_mode_from_name(channel_name: str) -> int | None:
                if not channel_name.startswith(momentum_prefix):
                    return None
                raw = channel_name[len(momentum_prefix) :]
                try:
                    return int(raw)
                except ValueError:
                    return None

            for channel_name in sorted(
                momentum_channels.keys(),
                key=lambda name: (
                    _momentum_mode_from_name(name)
                    if _momentum_mode_from_name(name) is not None
                    else 10**9
                ),
            ):
                mode_n = _momentum_mode_from_name(channel_name)
                if mode_n is None:
                    continue
                _append_estimator_row(
                    estimator=f"SU(3) momentum p{mode_n}",
                    approach="su3_momentum",
                    channel_name=channel_name,
                    result_obj=momentum_channels[channel_name],
                    is_core=(mode_n == 0),
                    n_mode=mode_n,
                )

            consensus_mass = float("nan")
            consensus_stat = float("nan")
            consensus_syst = float("nan")
            consensus_weighting = "n/a"
            chi2 = float("nan")
            ndof = 1

            if core_rows:
                core_mass = np.asarray([float(row["mass"]) for row in core_rows], dtype=float)
                core_err = np.asarray([float(row["mass_error"]) for row in core_rows], dtype=float)
                finite_weight_mask = np.isfinite(core_err) & (core_err > 0)
                if bool(np.any(finite_weight_mask)):
                    weighted_mass = core_mass[finite_weight_mask]
                    weighted_err = core_err[finite_weight_mask]
                    weights = 1.0 / np.maximum(weighted_err, 1e-12) ** 2
                    consensus_mass = float(np.sum(weights * weighted_mass) / np.sum(weights))
                    consensus_stat = float(np.sqrt(1.0 / np.sum(weights)))
                    n_eff = int(weighted_mass.size)
                    ndof = max(n_eff - 1, 1)
                    chi2 = float(np.sum(weights * (weighted_mass - consensus_mass) ** 2))
                    consensus_weighting = "inverse-variance weighted"
                else:
                    consensus_mass = float(np.mean(core_mass))
                    consensus_stat = (
                        float(np.std(core_mass, ddof=1) / np.sqrt(float(core_mass.size)))
                        if core_mass.size > 1
                        else float("nan")
                    )
                    consensus_weighting = "unweighted mean"
                consensus_syst = (
                    float(np.std(core_mass, ddof=1)) if core_mass.size > 1 else 0.0
                )

            strong_mass = float("nan")
            if strong_result is not None and strong_result.n_samples > 0:
                strong_mass = _display_mass(strong_result)
            su3_mass = float("nan")
            if su3_result is not None and su3_result.n_samples > 0:
                su3_mass = _display_mass(su3_result)

            for row in estimator_rows:
                mass_value = float(row["mass"])
                row["delta_vs_consensus_pct"] = _delta_percent(mass_value, consensus_mass)
                row["delta_vs_strong_pct"] = _delta_percent(mass_value, strong_mass)
                row["delta_vs_su3_plaquette_pct"] = _delta_percent(mass_value, su3_mass)

            approach_order = {
                "Strong-force isotropic": 0,
                "Anisotropic-edge isotropic": 1,
                "SU(3) plaquette": 2,
            }
            if estimator_rows:
                table_df = pd.DataFrame(estimator_rows)
                table_df["approach_order"] = (
                    table_df["estimator"].map(approach_order).fillna(3).astype(int)
                )
                table_df["n_mode_sort"] = (
                    pd.to_numeric(table_df["n_mode"], errors="coerce").fillna(-1).astype(int)
                )
                table_df = table_df.sort_values(
                    ["approach_order", "n_mode_sort", "channel"]
                ).drop(columns=["approach_order", "n_mode_sort"], errors="ignore")
                anisotropic_edge_glueball_approach_table.value = table_df
            else:
                anisotropic_edge_glueball_approach_table.value = pd.DataFrame()

            pairwise_rows: list[dict[str, Any]] = []
            for idx_a in range(len(core_rows)):
                for idx_b in range(idx_a + 1, len(core_rows)):
                    row_a = core_rows[idx_a]
                    row_b = core_rows[idx_b]
                    mass_a = float(row_a["mass"])
                    mass_b = float(row_b["mass"])
                    err_a = float(row_a["mass_error"])
                    err_b = float(row_b["mass_error"])
                    ratio = mass_a / mass_b if mass_b > 0 else float("nan")
                    delta_pct = (ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan")
                    abs_diff = abs(mass_a - mass_b)
                    comb_err = float("nan")
                    pull_sigma = float("nan")
                    if np.isfinite(err_a) and np.isfinite(err_b):
                        comb_err = float(np.sqrt(max(err_a, 0.0) ** 2 + max(err_b, 0.0) ** 2))
                        if comb_err > 0:
                            pull_sigma = abs_diff / comb_err
                    pairwise_rows.append(
                        {
                            "estimator_a": str(row_a["estimator"]),
                            "estimator_b": str(row_b["estimator"]),
                            "mass_a": mass_a,
                            "mass_b": mass_b,
                            "ratio_a_over_b": ratio,
                            "delta_pct": delta_pct,
                            "abs_delta_pct": abs(delta_pct) if np.isfinite(delta_pct) else np.nan,
                            "abs_mass_diff": abs_diff,
                            "combined_error": comb_err,
                            "pull_sigma": pull_sigma,
                        }
                    )
            if pairwise_rows:
                pairwise_df = pd.DataFrame(pairwise_rows).sort_values(
                    "abs_delta_pct", ascending=False
                )
                anisotropic_edge_glueball_pairwise_table.value = pairwise_df
            else:
                anisotropic_edge_glueball_pairwise_table.value = pd.DataFrame()

            expected_core = {
                "Strong-force isotropic",
                "Anisotropic-edge isotropic",
                "SU(3) plaquette",
                "SU(3) momentum p0",
            }
            observed_core = {str(row["estimator"]) for row in core_rows}
            missing_core = sorted(expected_core - observed_core)

            verdict_label = "insufficient data"
            verdict_type = "secondary"
            verdict_details = "Need at least two core glueball estimators with valid fits."
            if pairwise_rows:
                abs_delta_values = np.asarray(
                    [float(row.get("abs_delta_pct", np.nan)) for row in pairwise_rows],
                    dtype=float,
                )
                pull_values = np.asarray(
                    [float(row.get("pull_sigma", np.nan)) for row in pairwise_rows], dtype=float
                )
                finite_abs_delta = abs_delta_values[np.isfinite(abs_delta_values)]
                finite_pull = pull_values[np.isfinite(pull_values)]
                max_abs_delta = (
                    float(np.max(finite_abs_delta)) if finite_abs_delta.size > 0 else float("nan")
                )
                max_pull = float(np.max(finite_pull)) if finite_pull.size > 0 else float("nan")

                if np.isfinite(max_abs_delta):
                    verdict_details = f"max |Δ%| = {max_abs_delta:.2f}%"
                else:
                    verdict_details = "max |Δ%| = n/a"
                if np.isfinite(max_pull):
                    verdict_details += f", max pull = {max_pull:.2f}σ"
                else:
                    verdict_details += ", max pull = n/a"

                if (
                    np.isfinite(max_abs_delta)
                    and max_abs_delta <= 5.0
                    and (not np.isfinite(max_pull) or max_pull <= 1.5)
                ):
                    verdict_label = "consistent"
                    verdict_type = "success"
                elif (
                    np.isfinite(max_abs_delta)
                    and max_abs_delta <= 15.0
                    and (not np.isfinite(max_pull) or max_pull <= 3.0)
                ):
                    verdict_label = "mild tension"
                    verdict_type = "warning"
                else:
                    verdict_label = "tension"
                    verdict_type = "danger"

            if missing_core:
                verdict_details += f"; missing: {', '.join(missing_core)}"
            if glueball_systematics_error:
                verdict_details += f"; warnings: {glueball_systematics_error}"
            anisotropic_edge_glueball_systematics_badge.object = (
                f"Systematics verdict: {verdict_label}. {verdict_details}"
            )
            anisotropic_edge_glueball_systematics_badge.alert_type = verdict_type

            summary_lines = ["**Glueball Approach Comparison (4 estimators):**"]
            if core_rows:
                core_sorted = sorted(
                    core_rows,
                    key=lambda row: approach_order.get(str(row["estimator"]), 3),
                )
                for row in core_sorted:
                    mass_value = float(row["mass"])
                    mass_err = float(row["mass_error"])
                    delta_cons = float(row["delta_vs_consensus_pct"])
                    entry = f"- {row['estimator']}: `{mass_value:.6g}`"
                    if np.isfinite(mass_err):
                        entry += f" ± `{mass_err:.2g}`"
                    if np.isfinite(delta_cons):
                        entry += f", Δ vs consensus `{delta_cons:+.2f}%`"
                    summary_lines.append(entry)
            else:
                summary_lines.append("- No valid glueball estimators available.")
            if missing_core:
                summary_lines.append(f"- Missing estimators: `{', '.join(missing_core)}`")
            if momentum_rows:
                summary_lines.append(f"- Momentum modes available: `{len(momentum_rows)}`")
                if np.isfinite(momentum_length) and momentum_length > 0:
                    summary_lines.append(
                        f"- Momentum axis/length: axis `{momentum_axis}`, `L={momentum_length:.6g}`"
                    )
            if glueball_systematics_error:
                summary_lines.append(
                    f"- Estimator warnings: `{glueball_systematics_error}`"
                )
            anisotropic_edge_glueball_approach_summary.object = "  \n".join(summary_lines)

            consensus_lines = ["**Glueball Consensus / Systematics:**"]
            if np.isfinite(consensus_mass) and consensus_mass > 0:
                line = (
                    f"- Consensus mass ({consensus_weighting}): `{consensus_mass:.6g}`"
                )
                if np.isfinite(consensus_stat):
                    line += f" ± `{consensus_stat:.2g}` (stat)"
                if np.isfinite(consensus_syst):
                    line += f" ± `{consensus_syst:.2g}` (syst)"
                consensus_lines.append(line)
                if np.isfinite(consensus_syst) and consensus_mass > 0:
                    consensus_lines.append(
                        f"- Relative systematic spread: `{(consensus_syst / consensus_mass) * 100.0:.2f}%`"
                    )
                if np.isfinite(chi2):
                    red_chi2 = chi2 / max(ndof, 1)
                    consensus_lines.append(
                        f"- Core-estimator consistency (`chi2/ndof`): `{chi2:.4g}/{ndof}` = `{red_chi2:.4g}`"
                    )
            else:
                consensus_lines.append("- n/a (not enough core estimators).")

            if pairwise_rows:
                max_pair = max(
                    pairwise_rows,
                    key=lambda row: float(row.get("abs_delta_pct", float("-inf"))),
                )
                if np.isfinite(float(max_pair["abs_delta_pct"])):
                    consensus_lines.append(
                        "- Largest pairwise discrepancy: "
                        f"`{max_pair['estimator_a']}` vs `{max_pair['estimator_b']}` "
                        f"({float(max_pair['delta_pct']):+.2f}%, pull `{float(max_pair['pull_sigma']):.2f}σ`)"
                    )
            anisotropic_edge_glueball_consensus_summary.object = "  \n".join(consensus_lines)

            if core_rows:
                core_sorted = sorted(
                    core_rows,
                    key=lambda row: approach_order.get(str(row["estimator"]), 3),
                )
                core_df = pd.DataFrame(core_sorted).reset_index(drop=True)
                core_df["x"] = core_df.index.astype(float)
                scatter = hv.Scatter(
                    core_df,
                    kdims=[("x", "Estimator index")],
                    vdims=[("mass", "Mass"), ("estimator", "Estimator"), ("mass_error", "Mass Error")],
                ).opts(
                    width=760,
                    height=320,
                    size=11,
                    color="#4c78a8",
                    marker="circle",
                    tools=["hover"],
                    xlabel="Estimator",
                    ylabel="Mass (index units)",
                    title="Glueball Estimator Consensus",
                )
                overlay = scatter
                err_df = core_df[
                    np.isfinite(core_df["mass_error"].to_numpy())
                    & (core_df["mass_error"].to_numpy() > 0)
                ][["x", "mass", "mass_error"]]
                if not err_df.empty:
                    overlay *= hv.ErrorBars(
                        err_df,
                        kdims=[("x", "Estimator index")],
                        vdims=[("mass", "Mass"), ("mass_error", "Mass Error")],
                    ).opts(color="#4c78a8", alpha=0.9, line_width=1)
                if np.isfinite(consensus_mass):
                    overlay *= hv.HLine(float(consensus_mass)).opts(
                        color="#2ca02c",
                        line_width=2,
                    )
                if (
                    np.isfinite(consensus_mass)
                    and np.isfinite(consensus_syst)
                    and consensus_syst > 0
                ):
                    overlay *= hv.HSpan(
                        float(consensus_mass - consensus_syst),
                        float(consensus_mass + consensus_syst),
                    ).opts(color="#2ca02c", alpha=0.12)
                xticks = [(float(i), str(label)) for i, label in enumerate(core_df["estimator"])]
                overlay = overlay.opts(
                    xlim=(-0.5, float(len(core_df) - 0.5)),
                    xticks=xticks,
                    xrotation=20,
                    show_grid=True,
                )
                anisotropic_edge_glueball_consensus_plot.objects = [overlay]
            else:
                anisotropic_edge_glueball_consensus_plot.objects = [
                    pn.pane.Markdown(
                        "_Need at least one valid core glueball estimator to populate this plot._"
                    )
                ]

            dispersion_m0 = float("nan")
            dispersion_ceff = float("nan")
            if momentum_rows:
                momentum_df = pd.DataFrame(momentum_rows).sort_values("n_mode")
                scatter = hv.Scatter(
                    momentum_df,
                    kdims=[("p2", "p^2")],
                    vdims=[("mass", "Mass"), ("n_mode", "n"), ("estimator", "Estimator")],
                ).opts(
                    width=760,
                    height=320,
                    size=9,
                    color="#4c78a8",
                    tools=["hover"],
                    xlabel="p^2",
                    ylabel="Mass (index units)",
                    title="SU(3) Momentum-Projected Glueball Dispersion",
                )
                overlay = scatter
                p0_df = momentum_df[momentum_df["n_mode"] == 0]
                if not p0_df.empty:
                    overlay *= hv.Scatter(
                        p0_df,
                        kdims=[("p2", "p^2")],
                        vdims=[("mass", "Mass"), ("n_mode", "n"), ("estimator", "Estimator")],
                    ).opts(size=12, color="#d62728", marker="diamond", tools=["hover"])
                err_df = momentum_df[
                    np.isfinite(momentum_df["mass_error"].to_numpy())
                    & (momentum_df["mass_error"].to_numpy() > 0)
                ][["p2", "mass", "mass_error"]]
                if not err_df.empty:
                    overlay *= hv.ErrorBars(
                        err_df,
                        kdims=[("p2", "p^2")],
                        vdims=[("mass", "Mass"), ("mass_error", "Mass Error")],
                    ).opts(color="#4c78a8", line_width=1, alpha=0.9)

                fit_df = momentum_df[
                    np.isfinite(momentum_df["p2"].to_numpy())
                    & np.isfinite(momentum_df["mass"].to_numpy())
                    & (momentum_df["mass"].to_numpy() > 0)
                ][["p2", "mass", "mass_error"]]
                if len(fit_df) >= 2:
                    p2 = fit_df["p2"].to_numpy(dtype=float)
                    mass = fit_df["mass"].to_numpy(dtype=float)
                    y = mass * mass
                    y_err = 2.0 * mass * fit_df["mass_error"].to_numpy(dtype=float)
                    design = np.stack([np.ones_like(p2), p2], axis=1)
                    fit_mask = np.isfinite(y_err) & (y_err > 0)
                    if np.count_nonzero(fit_mask) >= 2:
                        w = 1.0 / np.maximum(y_err[fit_mask], 1e-12) ** 2
                        lhs = design[fit_mask] * np.sqrt(w)[:, None]
                        rhs = y[fit_mask] * np.sqrt(w)
                        beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
                    else:
                        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                    intercept = float(beta[0])
                    slope = float(beta[1])
                    p2_grid = np.linspace(float(np.min(p2)), float(np.max(p2)), 200)
                    fit_mass = np.sqrt(np.clip(intercept + slope * p2_grid, a_min=0.0, a_max=None))
                    overlay *= hv.Curve(
                        (p2_grid, fit_mass),
                        kdims=[("p2", "p^2")],
                        vdims=[("mass_fit", "Fit mass")],
                    ).opts(color="#ff7f0e", line_width=2, line_dash="dashed")
                    dispersion_m0 = np.sqrt(intercept) if intercept > 0 else float("nan")
                    dispersion_ceff = np.sqrt(slope) if slope > 0 else float("nan")

                if np.isfinite(strong_mass) and strong_mass > 0:
                    overlay *= hv.HLine(float(strong_mass)).opts(
                        color="#7f7f7f",
                        line_dash="dashed",
                        line_width=2,
                    )
                if np.isfinite(su3_mass) and su3_mass > 0:
                    overlay *= hv.HLine(float(su3_mass)).opts(
                        color="#2ca02c",
                        line_dash="dotdash",
                        line_width=2,
                    )
                if np.isfinite(consensus_mass) and consensus_mass > 0:
                    overlay *= hv.HLine(float(consensus_mass)).opts(
                        color="#9467bd",
                        line_dash="solid",
                        line_width=2,
                    )
                anisotropic_edge_glueball_dispersion_plot.objects = [overlay]
            else:
                anisotropic_edge_glueball_dispersion_plot.objects = [
                    pn.pane.Markdown(
                        "_No momentum-projected glueball channels available. "
                        "Enable momentum projection in Glueball Color Settings._"
                    )
                ]

            if np.isfinite(dispersion_m0) and np.isfinite(dispersion_ceff):
                consensus_lines = [anisotropic_edge_glueball_consensus_summary.object]
                consensus_lines.append(
                    "  \n"
                    + "- Dispersion fit (`m(p)^2 = m0^2 + c_eff^2 p^2`): "
                    + f"`m0={dispersion_m0:.6g}`, `c_eff={dispersion_ceff:.6g}`"
                )
                anisotropic_edge_glueball_consensus_summary.object = "".join(consensus_lines)

            tensor_momentum_channels = dict(tensor_momentum_results or {})
            if not tensor_momentum_channels:
                tensor_momentum_prefix = "tensor_momentum_p"
                tensor_momentum_channels = {
                    name: value
                    for name, value in results.items()
                    if name.startswith(tensor_momentum_prefix)
                }
            tensor_component_labels = tuple(
                str(label) for label in (tensor_momentum_meta or {}).get("component_labels", ())
            )
            if not tensor_component_labels:
                tensor_component_labels = (
                    "q_xy",
                    "q_xz",
                    "q_yz",
                    "q_xx_minus_yy",
                    "q_2zz_minus_xx_minus_yy",
                )
            tensor_momentum_axis = int(
                (tensor_momentum_meta or {}).get(
                    "momentum_axis", int(anisotropic_edge_settings.tensor_momentum_axis)
                )
            )
            tensor_momentum_length = float(
                (tensor_momentum_meta or {}).get("momentum_length_scale", float("nan"))
            )
            if (
                not np.isfinite(tensor_momentum_length)
                or tensor_momentum_length <= 0
            ) and history is not None:
                tensor_momentum_length = _extract_axis_extent_from_bounds(
                    history.bounds, tensor_momentum_axis
                )

            tensor_rows: list[dict[str, Any]] = []
            tensor_core_rows: list[dict[str, Any]] = []
            tensor_momentum_rows: list[dict[str, Any]] = []
            tensor_component_rows: list[dict[str, Any]] = []

            tensor_momentum_prefix = "tensor_momentum_p"

            def _parse_tensor_momentum_name(channel_name: str) -> tuple[int | None, str | None]:
                if not channel_name.startswith(tensor_momentum_prefix):
                    return None, None
                raw = channel_name[len(tensor_momentum_prefix) :]
                if "_" in raw:
                    mode_raw, component = raw.split("_", 1)
                else:
                    mode_raw, component = raw, None
                try:
                    return int(mode_raw), component
                except ValueError:
                    return None, None

            def _append_tensor_row(
                *,
                estimator: str,
                approach: str,
                channel_name: str,
                result_obj: ChannelCorrelatorResult | None,
                is_core: bool,
                n_mode: int | None = None,
                component: str | None = None,
            ) -> None:
                if result_obj is None or result_obj.n_samples <= 0:
                    return
                mass_value = _display_mass(result_obj)
                if not np.isfinite(mass_value) or mass_value <= 0:
                    return
                mass_err = _display_mass_error(result_obj)
                r2_value = _display_r2(result_obj)
                p_value = float("nan")
                p2_value = float("nan")
                if (
                    n_mode is not None
                    and np.isfinite(tensor_momentum_length)
                    and tensor_momentum_length > 0
                ):
                    p_value = (2.0 * np.pi * float(n_mode)) / float(tensor_momentum_length)
                    p2_value = p_value * p_value

                row = {
                    "estimator": estimator,
                    "approach": approach,
                    "channel": channel_name,
                    "component": component if component is not None else "",
                    "n_mode": n_mode if n_mode is not None else "",
                    "mass": float(mass_value),
                    "mass_error": (
                        float(mass_err) if np.isfinite(mass_err) and mass_err >= 0 else np.nan
                    ),
                    "r2": float(r2_value) if np.isfinite(r2_value) else np.nan,
                    "p": p_value,
                    "p2": p2_value,
                    "core_estimator": bool(is_core),
                }
                tensor_rows.append(row)
                if is_core:
                    tensor_core_rows.append(row)
                if n_mode is not None and np.isfinite(p2_value):
                    if component is None:
                        tensor_momentum_rows.append(
                            {
                                "estimator": estimator,
                                "channel": channel_name,
                                "n_mode": int(n_mode),
                                "p": float(p_value),
                                "p2": float(p2_value),
                                "mass": float(mass_value),
                                "mass_error": (
                                    float(mass_err)
                                    if np.isfinite(mass_err) and mass_err >= 0
                                    else np.nan
                                ),
                            }
                        )
                    else:
                        tensor_component_rows.append(
                            {
                                "estimator": estimator,
                                "channel": channel_name,
                                "component": str(component),
                                "n_mode": int(n_mode),
                                "p": float(p_value),
                                "p2": float(p2_value),
                                "mass": float(mass_value),
                                "mass_error": (
                                    float(mass_err)
                                    if np.isfinite(mass_err) and mass_err >= 0
                                    else np.nan
                                ),
                            }
                        )

            _append_tensor_row(
                estimator="Anisotropic-edge tensor",
                approach="anisotropic_edge",
                channel_name="tensor",
                result_obj=results.get("tensor"),
                is_core=True,
            )
            _append_tensor_row(
                estimator="Anisotropic-edge tensor traceless",
                approach="anisotropic_edge_traceless",
                channel_name="tensor_traceless",
                result_obj=results.get("tensor_traceless"),
                is_core=True,
            )
            _append_tensor_row(
                estimator="Strong-force tensor",
                approach="strong_force",
                channel_name="tensor_strong_force",
                result_obj=tensor_strong_result,
                is_core=True,
            )

            def _tensor_component_sort_key(component: str | None) -> tuple[int, int]:
                if component is None:
                    return 0, -1
                try:
                    return 1, int(tensor_component_labels.index(component))
                except ValueError:
                    return 1, int(len(tensor_component_labels))

            for channel_name in sorted(
                tensor_momentum_channels.keys(),
                key=lambda name: (
                    _parse_tensor_momentum_name(name)[0]
                    if _parse_tensor_momentum_name(name)[0] is not None
                    else 10**9,
                    _tensor_component_sort_key(_parse_tensor_momentum_name(name)[1]),
                    name,
                ),
            ):
                mode_n, component_name = _parse_tensor_momentum_name(channel_name)
                if mode_n is None:
                    continue
                if component_name is None:
                    _append_tensor_row(
                        estimator=f"Tensor momentum p{mode_n}",
                        approach="momentum_contracted",
                        channel_name=channel_name,
                        result_obj=tensor_momentum_channels[channel_name],
                        is_core=(mode_n == 0),
                        n_mode=mode_n,
                        component=None,
                    )
                else:
                    _append_tensor_row(
                        estimator=f"Tensor momentum p{mode_n} {component_name}",
                        approach="momentum_component",
                        channel_name=channel_name,
                        result_obj=tensor_momentum_channels[channel_name],
                        is_core=False,
                        n_mode=mode_n,
                        component=component_name,
                    )

            tensor_consensus_mass = float("nan")
            tensor_consensus_stat = float("nan")
            tensor_consensus_syst = float("nan")
            tensor_consensus_weighting = "n/a"
            if tensor_core_rows:
                core_mass = np.asarray(
                    [float(row["mass"]) for row in tensor_core_rows],
                    dtype=float,
                )
                core_err = np.asarray(
                    [float(row["mass_error"]) for row in tensor_core_rows],
                    dtype=float,
                )
                finite_weight_mask = np.isfinite(core_err) & (core_err > 0)
                if bool(np.any(finite_weight_mask)):
                    weighted_mass = core_mass[finite_weight_mask]
                    weighted_err = core_err[finite_weight_mask]
                    weights = 1.0 / np.maximum(weighted_err, 1e-12) ** 2
                    tensor_consensus_mass = float(np.sum(weights * weighted_mass) / np.sum(weights))
                    tensor_consensus_stat = float(np.sqrt(1.0 / np.sum(weights)))
                    tensor_consensus_weighting = "inverse-variance weighted"
                else:
                    tensor_consensus_mass = float(np.mean(core_mass))
                    tensor_consensus_stat = (
                        float(np.std(core_mass, ddof=1) / np.sqrt(float(core_mass.size)))
                        if core_mass.size > 1
                        else float("nan")
                    )
                    tensor_consensus_weighting = "unweighted mean"
                tensor_consensus_syst = (
                    float(np.std(core_mass, ddof=1)) if core_mass.size > 1 else 0.0
                )

            for row in tensor_rows:
                mass_value = float(row["mass"])
                row["delta_vs_consensus_pct"] = _delta_percent(mass_value, tensor_consensus_mass)

            tensor_approach_order = {
                "Anisotropic-edge tensor": 0,
                "Anisotropic-edge tensor traceless": 1,
                "Strong-force tensor": 2,
                "Tensor momentum p0": 3,
            }
            if tensor_rows:
                table_df = pd.DataFrame(tensor_rows)
                table_df["approach_order"] = (
                    table_df["estimator"].map(tensor_approach_order).fillna(4).astype(int)
                )
                table_df["component_sort"] = table_df["component"].apply(
                    lambda value: 0 if str(value) == "" else 1
                )
                table_df["n_mode_sort"] = (
                    pd.to_numeric(table_df["n_mode"], errors="coerce").fillna(-1).astype(int)
                )
                table_df = table_df.sort_values(
                    ["approach_order", "n_mode_sort", "component_sort", "channel"]
                ).drop(
                    columns=["approach_order", "component_sort", "n_mode_sort"],
                    errors="ignore",
                )
                anisotropic_edge_tensor_approach_table.value = table_df
            else:
                anisotropic_edge_tensor_approach_table.value = pd.DataFrame()

            tensor_pairwise_rows: list[dict[str, Any]] = []
            for idx_a in range(len(tensor_core_rows)):
                for idx_b in range(idx_a + 1, len(tensor_core_rows)):
                    row_a = tensor_core_rows[idx_a]
                    row_b = tensor_core_rows[idx_b]
                    mass_a = float(row_a["mass"])
                    mass_b = float(row_b["mass"])
                    err_a = float(row_a["mass_error"])
                    err_b = float(row_b["mass_error"])
                    ratio = mass_a / mass_b if mass_b > 0 else float("nan")
                    delta_pct = (ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan")
                    abs_diff = abs(mass_a - mass_b)
                    combined_err = float("nan")
                    pull_sigma = float("nan")
                    if np.isfinite(err_a) and np.isfinite(err_b):
                        combined_err = float(np.sqrt(max(err_a, 0.0) ** 2 + max(err_b, 0.0) ** 2))
                        if combined_err > 0:
                            pull_sigma = abs_diff / combined_err
                    tensor_pairwise_rows.append(
                        {
                            "estimator_a": str(row_a["estimator"]),
                            "estimator_b": str(row_b["estimator"]),
                            "delta_pct": delta_pct,
                            "abs_delta_pct": (
                                abs(delta_pct) if np.isfinite(delta_pct) else np.nan
                            ),
                            "pull_sigma": pull_sigma,
                        }
                    )

            expected_tensor_core = {
                "Anisotropic-edge tensor",
                "Anisotropic-edge tensor traceless",
                "Strong-force tensor",
                "Tensor momentum p0",
            }
            observed_tensor_core = {str(row["estimator"]) for row in tensor_core_rows}
            missing_tensor_core = sorted(expected_tensor_core - observed_tensor_core)

            tensor_verdict_label = "insufficient data"
            tensor_verdict_type = "secondary"
            tensor_verdict_details = "Need at least two core tensor estimators with valid fits."
            if tensor_pairwise_rows:
                abs_delta_values = np.asarray(
                    [
                        float(row.get("abs_delta_pct", np.nan))
                        for row in tensor_pairwise_rows
                    ],
                    dtype=float,
                )
                pull_values = np.asarray(
                    [float(row.get("pull_sigma", np.nan)) for row in tensor_pairwise_rows],
                    dtype=float,
                )
                finite_abs_delta = abs_delta_values[np.isfinite(abs_delta_values)]
                finite_pull = pull_values[np.isfinite(pull_values)]
                max_abs_delta = (
                    float(np.max(finite_abs_delta)) if finite_abs_delta.size > 0 else float("nan")
                )
                max_pull = float(np.max(finite_pull)) if finite_pull.size > 0 else float("nan")

                if np.isfinite(max_abs_delta):
                    tensor_verdict_details = f"max |Δ%| = {max_abs_delta:.2f}%"
                else:
                    tensor_verdict_details = "max |Δ%| = n/a"
                if np.isfinite(max_pull):
                    tensor_verdict_details += f", max pull = {max_pull:.2f}σ"
                else:
                    tensor_verdict_details += ", max pull = n/a"

                if (
                    np.isfinite(max_abs_delta)
                    and max_abs_delta <= 7.5
                    and (not np.isfinite(max_pull) or max_pull <= 1.5)
                ):
                    tensor_verdict_label = "consistent"
                    tensor_verdict_type = "success"
                elif (
                    np.isfinite(max_abs_delta)
                    and max_abs_delta <= 20.0
                    and (not np.isfinite(max_pull) or max_pull <= 3.0)
                ):
                    tensor_verdict_label = "mild tension"
                    tensor_verdict_type = "warning"
                else:
                    tensor_verdict_label = "tension"
                    tensor_verdict_type = "danger"

            if missing_tensor_core:
                tensor_verdict_details += f"; missing: {', '.join(missing_tensor_core)}"
            if tensor_systematics_error:
                tensor_verdict_details += f"; warnings: {tensor_systematics_error}"
            anisotropic_edge_tensor_systematics_badge.object = (
                f"Tensor systematics verdict: {tensor_verdict_label}. {tensor_verdict_details}"
            )
            anisotropic_edge_tensor_systematics_badge.alert_type = tensor_verdict_type

            tensor_dispersion_m0 = float("nan")
            tensor_dispersion_ceff = float("nan")
            if tensor_momentum_rows:
                momentum_df = pd.DataFrame(tensor_momentum_rows).sort_values("n_mode")
                scatter = hv.Scatter(
                    momentum_df,
                    kdims=[("p2", "p^2")],
                    vdims=[("mass", "Mass"), ("n_mode", "n"), ("estimator", "Estimator")],
                ).opts(
                    width=760,
                    height=320,
                    size=9,
                    color="#4c78a8",
                    tools=["hover"],
                    xlabel="p^2",
                    ylabel="Mass (index units)",
                    title="Tensor Momentum Dispersion (Contracted Spin-2)",
                )
                overlay = scatter
                p0_df = momentum_df[momentum_df["n_mode"] == 0]
                if not p0_df.empty:
                    overlay *= hv.Scatter(
                        p0_df,
                        kdims=[("p2", "p^2")],
                        vdims=[("mass", "Mass"), ("n_mode", "n"), ("estimator", "Estimator")],
                    ).opts(size=12, color="#d62728", marker="diamond", tools=["hover"])
                err_df = momentum_df[
                    np.isfinite(momentum_df["mass_error"].to_numpy())
                    & (momentum_df["mass_error"].to_numpy() > 0)
                ][["p2", "mass", "mass_error"]]
                if not err_df.empty:
                    overlay *= hv.ErrorBars(
                        err_df,
                        kdims=[("p2", "p^2")],
                        vdims=[("mass", "Mass"), ("mass_error", "Mass Error")],
                    ).opts(color="#4c78a8", line_width=1, alpha=0.9)

                fit_df = momentum_df[
                    np.isfinite(momentum_df["p2"].to_numpy())
                    & np.isfinite(momentum_df["mass"].to_numpy())
                    & (momentum_df["mass"].to_numpy() > 0)
                ][["p2", "mass", "mass_error"]]
                if len(fit_df) >= 2:
                    p2 = fit_df["p2"].to_numpy(dtype=float)
                    mass = fit_df["mass"].to_numpy(dtype=float)
                    y = mass * mass
                    y_err = 2.0 * mass * fit_df["mass_error"].to_numpy(dtype=float)
                    design = np.stack([np.ones_like(p2), p2], axis=1)
                    fit_mask = np.isfinite(y_err) & (y_err > 0)
                    if np.count_nonzero(fit_mask) >= 2:
                        w = 1.0 / np.maximum(y_err[fit_mask], 1e-12) ** 2
                        lhs = design[fit_mask] * np.sqrt(w)[:, None]
                        rhs = y[fit_mask] * np.sqrt(w)
                        beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
                    else:
                        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                    intercept = float(beta[0])
                    slope = float(beta[1])
                    p2_grid = np.linspace(float(np.min(p2)), float(np.max(p2)), 200)
                    fit_mass = np.sqrt(np.clip(intercept + slope * p2_grid, a_min=0.0, a_max=None))
                    overlay *= hv.Curve(
                        (p2_grid, fit_mass),
                        kdims=[("p2", "p^2")],
                        vdims=[("mass_fit", "Fit mass")],
                    ).opts(color="#ff7f0e", line_width=2, line_dash="dashed")
                    tensor_dispersion_m0 = np.sqrt(intercept) if intercept > 0 else float("nan")
                    tensor_dispersion_ceff = np.sqrt(slope) if slope > 0 else float("nan")

                tensor_mass = float("nan")
                tensor_temporal_result = results.get("tensor")
                if tensor_temporal_result is not None and tensor_temporal_result.n_samples > 0:
                    tensor_mass = _display_mass(tensor_temporal_result)
                tensor_traceless_mass = float("nan")
                tensor_traceless_temporal_result = results.get("tensor_traceless")
                if (
                    tensor_traceless_temporal_result is not None
                    and tensor_traceless_temporal_result.n_samples > 0
                ):
                    tensor_traceless_mass = _display_mass(tensor_traceless_temporal_result)
                strong_tensor_mass = float("nan")
                if tensor_strong_result is not None and tensor_strong_result.n_samples > 0:
                    strong_tensor_mass = _display_mass(tensor_strong_result)
                if np.isfinite(tensor_mass) and tensor_mass > 0:
                    overlay *= hv.HLine(float(tensor_mass)).opts(
                        color="#2ca02c", line_dash="dashed", line_width=2
                    )
                if np.isfinite(tensor_traceless_mass) and tensor_traceless_mass > 0:
                    overlay *= hv.HLine(float(tensor_traceless_mass)).opts(
                        color="#9467bd", line_dash="dotdash", line_width=2
                    )
                if np.isfinite(strong_tensor_mass) and strong_tensor_mass > 0:
                    overlay *= hv.HLine(float(strong_tensor_mass)).opts(
                        color="#7f7f7f", line_dash="dotted", line_width=2
                    )
                if np.isfinite(tensor_consensus_mass) and tensor_consensus_mass > 0:
                    overlay *= hv.HLine(float(tensor_consensus_mass)).opts(
                        color="#d62728", line_dash="solid", line_width=2
                    )
                anisotropic_edge_tensor_dispersion_plot.objects = [overlay]
            else:
                anisotropic_edge_tensor_dispersion_plot.objects = [
                    pn.pane.Markdown(
                        "_No momentum-projected tensor channels available. "
                        "Enable Tensor Momentum Settings._"
                    )
                ]

            tensor_component_spread_pct = float("nan")
            if tensor_component_rows:
                comp_df = pd.DataFrame(tensor_component_rows).sort_values(["component", "n_mode"])
                component_overlay = None
                component_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                ]
                for comp_idx, component_name in enumerate(tensor_component_labels):
                    sub = comp_df[comp_df["component"] == component_name]
                    if sub.empty:
                        continue
                    color = component_colors[comp_idx % len(component_colors)]
                    layer = hv.Scatter(
                        sub,
                        kdims=[("p2", "p^2")],
                        vdims=[
                            ("mass", "Mass"),
                            ("n_mode", "n"),
                            ("component", "Component"),
                            ("mass_error", "Mass Error"),
                        ],
                        label=component_name,
                    ).opts(size=7, color=color, tools=["hover"])
                    err_sub = sub[
                        np.isfinite(sub["mass_error"].to_numpy())
                        & (sub["mass_error"].to_numpy() > 0)
                    ][["p2", "mass", "mass_error"]]
                    if not err_sub.empty:
                        layer *= hv.ErrorBars(
                            err_sub,
                            kdims=[("p2", "p^2")],
                            vdims=[("mass", "Mass"), ("mass_error", "Mass Error")],
                        ).opts(color=color, line_width=1, alpha=0.8)
                    fit_sub = sub[
                        np.isfinite(sub["p2"].to_numpy())
                        & np.isfinite(sub["mass"].to_numpy())
                        & (sub["mass"].to_numpy() > 0)
                    ][["p2", "mass", "mass_error"]]
                    if len(fit_sub) >= 2:
                        p2 = fit_sub["p2"].to_numpy(dtype=float)
                        mass = fit_sub["mass"].to_numpy(dtype=float)
                        y = mass * mass
                        y_err = 2.0 * mass * fit_sub["mass_error"].to_numpy(dtype=float)
                        design = np.stack([np.ones_like(p2), p2], axis=1)
                        fit_mask = np.isfinite(y_err) & (y_err > 0)
                        if np.count_nonzero(fit_mask) >= 2:
                            w = 1.0 / np.maximum(y_err[fit_mask], 1e-12) ** 2
                            lhs = design[fit_mask] * np.sqrt(w)[:, None]
                            rhs = y[fit_mask] * np.sqrt(w)
                            beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
                        else:
                            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                        intercept = float(beta[0])
                        slope = float(beta[1])
                        p2_grid = np.linspace(float(np.min(p2)), float(np.max(p2)), 160)
                        fit_mass = np.sqrt(
                            np.clip(intercept + slope * p2_grid, a_min=0.0, a_max=None)
                        )
                        layer *= hv.Curve(
                            (p2_grid, fit_mass),
                            kdims=[("p2", "p^2")],
                            vdims=[("mass_fit", "Fit mass")],
                            label=f"{component_name} fit",
                        ).opts(color=color, line_dash="dashed", line_width=1.5)
                    component_overlay = (
                        layer if component_overlay is None else component_overlay * layer
                    )

                p0_components = comp_df[comp_df["n_mode"] == 0]
                if len(p0_components) >= 2:
                    p0_mass = p0_components["mass"].to_numpy(dtype=float)
                    p0_mass = p0_mass[np.isfinite(p0_mass) & (p0_mass > 0)]
                    if p0_mass.size >= 2:
                        mean_mass = float(np.mean(p0_mass))
                        if mean_mass > 0:
                            tensor_component_spread_pct = (
                                float(np.max(p0_mass) - np.min(p0_mass)) / mean_mass * 100.0
                            )

                if component_overlay is not None:
                    anisotropic_edge_tensor_component_dispersion_plot.objects = [
                        component_overlay.opts(
                            width=760,
                            height=360,
                            show_legend=True,
                            legend_position="right",
                            xlabel="p^2",
                            ylabel="Mass (index units)",
                            title="Tensor Momentum Component Dispersions",
                        )
                    ]
                else:
                    anisotropic_edge_tensor_component_dispersion_plot.objects = [
                        pn.pane.Markdown(
                            "_Tensor momentum component channels are present, but no valid fits "
                            "were extracted for plotting._"
                        )
                    ]
            else:
                anisotropic_edge_tensor_component_dispersion_plot.objects = [
                    pn.pane.Markdown(
                        "_No tensor momentum component channels available. "
                        "Enable Tensor Momentum Settings._"
                    )
                ]

            tensor_summary_lines = ["**Tensor Approach Comparison:**"]
            if tensor_core_rows:
                core_sorted = sorted(
                    tensor_core_rows,
                    key=lambda row: tensor_approach_order.get(str(row["estimator"]), 4),
                )
                for row in core_sorted:
                    mass_value = float(row["mass"])
                    mass_err = float(row["mass_error"])
                    delta_cons = float(row.get("delta_vs_consensus_pct", np.nan))
                    entry = f"- {row['estimator']}: `{mass_value:.6g}`"
                    if np.isfinite(mass_err):
                        entry += f" ± `{mass_err:.2g}`"
                    if np.isfinite(delta_cons):
                        entry += f", Δ vs consensus `{delta_cons:+.2f}%`"
                    tensor_summary_lines.append(entry)
                if np.isfinite(tensor_consensus_mass) and tensor_consensus_mass > 0:
                    consensus_entry = (
                        f"- Tensor consensus mass ({tensor_consensus_weighting}): "
                        f"`{tensor_consensus_mass:.6g}`"
                    )
                    if np.isfinite(tensor_consensus_stat):
                        consensus_entry += f" ± `{tensor_consensus_stat:.2g}` (stat)"
                    if np.isfinite(tensor_consensus_syst):
                        consensus_entry += f" ± `{tensor_consensus_syst:.2g}` (syst)"
                    tensor_summary_lines.append(consensus_entry)
            else:
                tensor_summary_lines.append("- No valid tensor estimators available.")
            if missing_tensor_core:
                tensor_summary_lines.append(f"- Missing estimators: `{', '.join(missing_tensor_core)}`")
            if tensor_momentum_rows:
                mode_count = len({int(row["n_mode"]) for row in tensor_momentum_rows})
                tensor_summary_lines.append(f"- Contracted momentum modes available: `{mode_count}`")
                if np.isfinite(tensor_momentum_length) and tensor_momentum_length > 0:
                    tensor_summary_lines.append(
                        f"- Momentum axis/length: axis `{tensor_momentum_axis}`, "
                        f"`L={tensor_momentum_length:.6g}`"
                    )
            if np.isfinite(tensor_component_spread_pct):
                tensor_summary_lines.append(
                    f"- p0 component splitting (max-min)/mean: `{tensor_component_spread_pct:.2f}%`"
                )
            if np.isfinite(tensor_dispersion_m0) and np.isfinite(tensor_dispersion_ceff):
                tensor_summary_lines.append(
                    "- Contracted dispersion fit (`m(p)^2 = m0^2 + c_eff^2 p^2`): "
                    f"`m0={tensor_dispersion_m0:.6g}`, `c_eff={tensor_dispersion_ceff:.6g}`"
                )
            if tensor_systematics_error:
                tensor_summary_lines.append(f"- Estimator warnings: `{tensor_systematics_error}`")
            anisotropic_edge_tensor_approach_summary.object = "  \n".join(tensor_summary_lines)

            anisotropic_edge_tensor_lorentz_ratio.object = (
                "**Tensor Lorentz Check (Legacy isotropic tensor):** "
                "n/a (spatial tensor estimator disabled in this tab)."
            )
            anisotropic_edge_tensor_traceless_lorentz_ratio.object = (
                "**Tensor Lorentz Check (Traceless tensor):** "
                "n/a (spatial tensor estimator disabled in this tab)."
            )

            glueball_lorentz_rows: list[dict[str, Any]] = []

            def _append_glueball_lorentz_row(
                estimator: str,
                result_obj: ChannelCorrelatorResult | None,
            ) -> None:
                if result_obj is None or result_obj.n_samples <= 0:
                    return
                mass_value = _display_mass(result_obj)
                if not np.isfinite(mass_value) or mass_value <= 0:
                    return
                mass_err = _display_mass_error(result_obj)
                r2_value = _display_r2(result_obj)
                glueball_lorentz_rows.append(
                    {
                        "estimator": estimator,
                        "mass": float(mass_value),
                        "mass_error": (
                            float(mass_err)
                            if np.isfinite(mass_err) and mass_err >= 0
                            else np.nan
                        ),
                        "r2": float(r2_value) if np.isfinite(r2_value) else np.nan,
                    }
                )

            _append_glueball_lorentz_row("Anisotropic-edge isotropic", edge_iso_result)
            _append_glueball_lorentz_row("Strong-force isotropic", strong_result)
            _append_glueball_lorentz_row("SU(3) plaquette", su3_result)
            _append_glueball_lorentz_row("SU(3) momentum p0", momentum_p0_result)

            consensus_mass_lorentz = float("nan")
            consensus_mass_lorentz_err = float("nan")
            chi2_lorentz = float("nan")
            ndof_lorentz = 1
            if glueball_lorentz_rows:
                masses = np.asarray([row["mass"] for row in glueball_lorentz_rows], dtype=float)
                errors = np.asarray([row["mass_error"] for row in glueball_lorentz_rows], dtype=float)
                weighted_mask = np.isfinite(errors) & (errors > 0)
                if bool(np.any(weighted_mask)):
                    mass_w = masses[weighted_mask]
                    err_w = errors[weighted_mask]
                    weights = 1.0 / np.maximum(err_w, 1e-12) ** 2
                    consensus_mass_lorentz = float(np.sum(weights * mass_w) / np.sum(weights))
                    consensus_mass_lorentz_err = float(np.sqrt(1.0 / np.sum(weights)))
                    ndof_lorentz = max(int(mass_w.size - 1), 1)
                    chi2_lorentz = float(np.sum(weights * (mass_w - consensus_mass_lorentz) ** 2))
                else:
                    consensus_mass_lorentz = float(np.mean(masses))
                    if masses.size > 1:
                        consensus_mass_lorentz_err = float(
                            np.std(masses, ddof=1) / np.sqrt(float(masses.size))
                        )

            max_abs_delta_pct = float("nan")
            if (
                np.isfinite(consensus_mass_lorentz)
                and consensus_mass_lorentz > 0
                and glueball_lorentz_rows
            ):
                delta_values = []
                for row in glueball_lorentz_rows:
                    delta = (float(row["mass"]) / consensus_mass_lorentz - 1.0) * 100.0
                    row["delta_vs_consensus_pct"] = delta
                    delta_values.append(abs(delta))
                max_abs_delta_pct = float(np.max(np.asarray(delta_values, dtype=float)))

            dispersion_m0 = float("nan")
            dispersion_c2 = float("nan")
            dispersion_c4 = float("nan")
            dispersion_ceff = float("nan")
            dispersion_r2 = float("nan")
            dispersion_fit_order = 0
            dispersion_rel_c4 = float("nan")
            if momentum_rows:
                momentum_df = (
                    pd.DataFrame(momentum_rows)
                    .sort_values("n_mode")
                    .drop_duplicates(subset=["n_mode"], keep="first")
                )
                fit_df = momentum_df[
                    np.isfinite(momentum_df["p2"].to_numpy())
                    & np.isfinite(momentum_df["mass"].to_numpy())
                    & (momentum_df["mass"].to_numpy() > 0)
                ][["p2", "mass", "mass_error"]]
                if len(fit_df) >= 2:
                    p2 = fit_df["p2"].to_numpy(dtype=float)
                    mass = fit_df["mass"].to_numpy(dtype=float)
                    y = mass * mass
                    y_err = 2.0 * mass * fit_df["mass_error"].to_numpy(dtype=float)
                    if len(fit_df) >= 3:
                        design = np.stack([np.ones_like(p2), p2, p2 * p2], axis=1)
                        dispersion_fit_order = 2
                    else:
                        design = np.stack([np.ones_like(p2), p2], axis=1)
                        dispersion_fit_order = 1
                    fit_mask = np.isfinite(y_err) & (y_err > 0)
                    try:
                        if np.count_nonzero(fit_mask) >= design.shape[1]:
                            w = 1.0 / np.maximum(y_err[fit_mask], 1e-12) ** 2
                            lhs = design[fit_mask] * np.sqrt(w)[:, None]
                            rhs = y[fit_mask] * np.sqrt(w)
                            beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
                            y_fit = design @ beta
                        else:
                            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                            y_fit = design @ beta
                        sse = float(np.sum((y - y_fit) ** 2))
                        sst = float(np.sum((y - np.mean(y)) ** 2))
                        dispersion_r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
                        dispersion_m0_sq = float(beta[0])
                        dispersion_c2 = float(beta[1]) if beta.size > 1 else float("nan")
                        dispersion_c4 = float(beta[2]) if beta.size > 2 else 0.0
                        dispersion_m0 = (
                            float(np.sqrt(dispersion_m0_sq))
                            if np.isfinite(dispersion_m0_sq) and dispersion_m0_sq > 0
                            else float("nan")
                        )
                        dispersion_ceff = (
                            float(np.sqrt(dispersion_c2))
                            if np.isfinite(dispersion_c2) and dispersion_c2 > 0
                            else float("nan")
                        )
                        if np.isfinite(dispersion_c2) and abs(dispersion_c2) > 1e-12:
                            dispersion_rel_c4 = abs(dispersion_c4) / max(
                                dispersion_c2 * dispersion_c2, 1e-12
                            )
                    except Exception:
                        pass

            missing_lorentz_estimators = sorted(
                {
                    "Anisotropic-edge isotropic",
                    "Strong-force isotropic",
                    "SU(3) plaquette",
                    "SU(3) momentum p0",
                }
                - {str(row["estimator"]) for row in glueball_lorentz_rows}
            )
            glueball_lorentz_lines = [
                "**Glueball Lorentz Check (4 Estimators + Momentum Dispersion):**"
            ]
            if glueball_lorentz_rows:
                for row in glueball_lorentz_rows:
                    entry = f"- {row['estimator']}: `m={row['mass']:.6g}`"
                    if np.isfinite(float(row["mass_error"])):
                        entry += f" ± `{float(row['mass_error']):.2g}`"
                    if np.isfinite(float(row["r2"])):
                        entry += f", `R²={float(row['r2']):.4f}`"
                    if np.isfinite(float(row.get("delta_vs_consensus_pct", np.nan))):
                        entry += (
                            f", Δ vs consensus `{float(row['delta_vs_consensus_pct']):+.2f}%`"
                        )
                    glueball_lorentz_lines.append(entry)
            else:
                glueball_lorentz_lines.append("- No valid glueball estimators available.")

            if np.isfinite(consensus_mass_lorentz) and consensus_mass_lorentz > 0:
                consensus_line = (
                    f"- Consensus mass: `{consensus_mass_lorentz:.6g}`"
                    + (
                        f" ± `{consensus_mass_lorentz_err:.2g}`"
                        if np.isfinite(consensus_mass_lorentz_err)
                        else ""
                    )
                )
                glueball_lorentz_lines.append(consensus_line)
            if np.isfinite(max_abs_delta_pct):
                glueball_lorentz_lines.append(
                    f"- Four-estimator spread: max |Δ| vs consensus `{max_abs_delta_pct:.2f}%`"
                )
            if np.isfinite(chi2_lorentz):
                glueball_lorentz_lines.append(
                    f"- Four-estimator consistency (`chi2/ndof`): "
                    f"`{chi2_lorentz:.4g}/{ndof_lorentz}` = `{(chi2_lorentz / max(ndof_lorentz, 1)):.4g}`"
                )
            if np.isfinite(dispersion_m0) and np.isfinite(dispersion_ceff):
                if dispersion_fit_order >= 2:
                    glueball_lorentz_lines.append(
                        "- Dispersion fit (`E(p)^2 = m0^2 + c2 p^2 + c4 p^4`): "
                        f"`m0={dispersion_m0:.6g}`, `c_eff=sqrt(c2)={dispersion_ceff:.6g}`, "
                        f"`c4={dispersion_c4:.6g}`, `R²={dispersion_r2:.4f}`"
                    )
                else:
                    glueball_lorentz_lines.append(
                        "- Dispersion fit (`E(p)^2 = m0^2 + c2 p^2`): "
                        f"`m0={dispersion_m0:.6g}`, `c_eff=sqrt(c2)={dispersion_ceff:.6g}`, "
                        f"`R²={dispersion_r2:.4f}`"
                    )
            else:
                glueball_lorentz_lines.append(
                    "- Dispersion fit: n/a (need at least two valid momentum modes with finite masses)."
                )
            if np.isfinite(dispersion_rel_c4):
                glueball_lorentz_lines.append(
                    f"- Lorentz-violation proxy `|c4|/c2²`: `{dispersion_rel_c4:.3g}`"
                )
            if missing_lorentz_estimators:
                glueball_lorentz_lines.append(
                    f"- Missing estimators: `{', '.join(missing_lorentz_estimators)}`"
                )
            if glueball_systematics_error:
                glueball_lorentz_lines.append(
                    f"- Estimator warnings: `{glueball_systematics_error}`"
                )
            anisotropic_edge_glueball_lorentz_ratio.object = "  \n".join(glueball_lorentz_lines)

            correction_lines = ["**Lorentz/Anisotropy Correction Factors (Glueball 4-way):**"]
            if np.isfinite(consensus_mass_lorentz) and consensus_mass_lorentz > 0:
                for row in glueball_lorentz_rows:
                    mass_value = float(row["mass"])
                    if not np.isfinite(mass_value) or mass_value <= 0:
                        continue
                    correction_factor = consensus_mass_lorentz / mass_value
                    correction_line = (
                        f"- `{row['estimator']}` correction to consensus: "
                        f"`{correction_factor:.6g}`"
                    )
                    if np.isfinite(float(row["mass_error"])) and np.isfinite(consensus_mass_lorentz_err):
                        rel_cons = consensus_mass_lorentz_err / max(consensus_mass_lorentz, 1e-12)
                        rel_row = float(row["mass_error"]) / max(mass_value, 1e-12)
                        corr_err = abs(correction_factor) * np.sqrt(rel_cons * rel_cons + rel_row * rel_row)
                        correction_line += f" ± `{corr_err:.2g}`"
                    correction_lines.append(correction_line)
            else:
                correction_lines.append("- n/a (consensus glueball mass unavailable).")

            if np.isfinite(dispersion_ceff) and dispersion_ceff > 0:
                correction_lines.append(
                    f"- Dispersion-derived `c_eff`: `{dispersion_ceff:.6g}`"
                )
                correction_lines.append(
                    f"- Absolute-scale correction proxy (`1/c_eff`): `{(1.0 / dispersion_ceff):.6g}`"
                )
            if np.isfinite(dispersion_rel_c4):
                correction_lines.append(
                    f"- Continuum-quality proxy (`|c4|/c2²`): `{dispersion_rel_c4:.3g}`"
                )
            if missing_lorentz_estimators:
                correction_lines.append(
                    f"- Missing estimators: `{', '.join(missing_lorentz_estimators)}`"
                )
            if glueball_systematics_error:
                correction_lines.append(
                    f"- Estimator warnings: `{glueball_systematics_error}`"
                )

            tensor_payload = tensor_payload_display
            state["anisotropic_edge_tensor_correction_payload"] = tensor_payload
            tensor_consensus_mass = float(tensor_payload.get("consensus_mass", float("nan")))
            tensor_consensus_err = float(tensor_payload.get("consensus_err", float("nan")))
            tensor_spread = float(tensor_payload.get("spread", float("nan")))
            tensor_base_label = str(tensor_payload.get("base_label", "tensor"))
            tensor_base_mass = float(tensor_payload.get("base_mass", float("nan")))
            tensor_scale = float(tensor_payload.get("correction_scale", float("nan")))
            tensor_scale_err = float(
                tensor_payload.get("correction_scale_err", float("nan"))
            )
            tensor_mode = str(tensor_payload.get("base_mode", _resolve_base_mass_mode(mode)))
            tensor_labels = np.asarray(tensor_payload.get("labels", []), dtype=object)
            tensor_valid = np.asarray(tensor_payload.get("valid_mask", []), dtype=bool)
            tensor_missing = [
                str(label)
                for label, is_valid in zip(tensor_labels.tolist(), tensor_valid.tolist(), strict=False)
                if not bool(is_valid)
            ]
            correction_lines.append("- Tensor calibration (4-way consensus):")
            correction_lines.append(f"- Tensor calibration mode: `{tensor_mode}`")
            if np.isfinite(tensor_consensus_mass) and tensor_consensus_mass > 0:
                tensor_consensus_line = (
                    f"- Tensor consensus mass: `{tensor_consensus_mass:.6g}`"
                )
                if np.isfinite(tensor_consensus_err):
                    tensor_consensus_line += f" ± `{tensor_consensus_err:.2g}` (stat)"
                correction_lines.append(tensor_consensus_line)
            else:
                correction_lines.append("- Tensor consensus mass: n/a")
            if np.isfinite(tensor_scale) and tensor_scale > 0:
                tensor_scale_line = (
                    f"- Global tensor correction scale (`consensus/{tensor_base_label}`): "
                    f"`{tensor_scale:.6g}`"
                )
                if np.isfinite(tensor_scale_err):
                    tensor_scale_line += f" ± `{tensor_scale_err:.2g}`"
                correction_lines.append(tensor_scale_line)
                if mode == "Tensor-Corrected":
                    correction_lines.append(
                        "- Tensor-corrected mode is active: this scale is applied to all table masses."
                    )
            elif mode == "Tensor-Corrected":
                correction_lines.append(
                    "- Tensor-corrected mode is active but no valid tensor scale could be computed."
                )
            if np.isfinite(tensor_base_mass) and tensor_base_mass > 0:
                correction_lines.append(
                    f"- Tensor base estimator `{tensor_base_label}` mass: `{tensor_base_mass:.6g}`"
                )
            if (
                np.isfinite(tensor_spread)
                and np.isfinite(tensor_consensus_mass)
                and tensor_consensus_mass > 0
            ):
                spread_pct = abs(tensor_spread) / tensor_consensus_mass * 100.0
                correction_lines.append(
                    f"- Tensor inter-estimator spread (1σ): `{tensor_spread:.6g}` ({spread_pct:.2f}%)"
                )
            if tensor_missing:
                correction_lines.append(
                    f"- Tensor missing estimators: `{', '.join(tensor_missing)}`"
                )
            if tensor_systematics_error:
                correction_lines.append(
                    f"- Tensor estimator warnings: `{tensor_systematics_error}`"
                )

            anisotropic_edge_lorentz_correction_summary.object = "  \n".join(correction_lines)

        def _on_anisotropic_edge_mass_mode_change(event):
            if state.get("anisotropic_edge_results") is None:
                return
            _update_anisotropic_edge_tables(state["anisotropic_edge_results"], event.new)
            _update_anisotropic_edge_glueball_crosscheck(
                results=state["anisotropic_edge_results"],
                strong_result=state.get("anisotropic_edge_glueball_strong_result"),
                edge_iso_result=state.get("anisotropic_edge_glueball_edge_iso_result"),
                su3_result=state.get("anisotropic_edge_glueball_su3_result"),
                momentum_results=state.get("anisotropic_edge_glueball_momentum_results"),
                spatial_result=state.get("anisotropic_edge_glueball_spatial_result"),
                history=state.get("history"),
                spatial_frame_idx=state.get("anisotropic_edge_glueball_spatial_frame_idx"),
                spatial_rho_edge=state.get("anisotropic_edge_glueball_spatial_rho_edge"),
                spatial_error=state.get("anisotropic_edge_glueball_spatial_error"),
                tensor_spatial_result=state.get("anisotropic_edge_tensor_spatial_result"),
                tensor_spatial_frame_idx=state.get("anisotropic_edge_tensor_spatial_frame_idx"),
                tensor_spatial_rho_edge=state.get("anisotropic_edge_tensor_spatial_rho_edge"),
                tensor_spatial_error=state.get("anisotropic_edge_tensor_spatial_error"),
                tensor_strong_result=state.get("anisotropic_edge_tensor_strong_result"),
                tensor_momentum_results=state.get("anisotropic_edge_tensor_momentum_results"),
                tensor_momentum_meta=state.get("anisotropic_edge_tensor_momentum_meta"),
                tensor_systematics_error=state.get("anisotropic_edge_tensor_systematics_error"),
                tensor_traceless_spatial_result=state.get(
                    "anisotropic_edge_tensor_traceless_spatial_result"
                ),
                tensor_traceless_spatial_frame_idx=state.get(
                    "anisotropic_edge_tensor_traceless_spatial_frame_idx"
                ),
                tensor_traceless_spatial_rho_edge=state.get(
                    "anisotropic_edge_tensor_traceless_spatial_rho_edge"
                ),
                tensor_traceless_spatial_error=state.get(
                    "anisotropic_edge_tensor_traceless_spatial_error"
                ),
                glueball_systematics_error=state.get("anisotropic_edge_glueball_systematics_error"),
                mode=event.new,
            )

        anisotropic_edge_mass_mode.param.watch(_on_anisotropic_edge_mass_mode_change, "value")

        def _on_anisotropic_edge_anchor_mode_change(_event):
            if state.get("anisotropic_edge_results") is None:
                return
            _update_anisotropic_edge_tables(state["anisotropic_edge_results"])

        anisotropic_edge_anchor_mode.param.watch(
            _on_anisotropic_edge_anchor_mode_change,
            "value",
        )

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
                strong_glueball_result = _compute_strong_glueball_for_anisotropic_edge(
                    history, anisotropic_edge_settings
                )
                strong_tensor_result = None
                tensor_momentum_results: dict[str, ChannelCorrelatorResult] = {}
                tensor_momentum_meta: dict[str, Any] | None = None
                tensor_systematics_error = None
                try:
                    strong_tensor_result = _compute_strong_tensor_for_anisotropic_edge(
                        history, anisotropic_edge_settings
                    )
                except Exception as exc:
                    tensor_systematics_error = f"strong-force tensor estimator failed: {exc}"
                try:
                    tensor_momentum_results, tensor_momentum_meta = (
                        _compute_tensor_momentum_for_anisotropic_edge(
                            history,
                            anisotropic_edge_settings,
                        )
                    )
                except Exception as exc:
                    tensor_msg = f"tensor momentum estimators failed: {exc}"
                    tensor_systematics_error = (
                        f"{tensor_systematics_error}; {tensor_msg}"
                        if tensor_systematics_error
                        else tensor_msg
                    )
                edge_iso_glueball_result = None
                su3_glueball_result = None
                momentum_glueball_results: dict[str, ChannelCorrelatorResult] = {}
                glueball_systematics_error = None
                try:
                    edge_iso_glueball_result = _compute_anisotropic_edge_isotropic_glueball_result(
                        history,
                        anisotropic_edge_settings,
                    )
                except Exception as exc:
                    glueball_systematics_error = (
                        f"isotropic-edge glueball estimator failed: {exc}"
                    )
                try:
                    su3_glueball_bundle, _ = _compute_anisotropic_glueball_color_result(
                        history,
                        anisotropic_edge_settings,
                        force_momentum_projection=True,
                        momentum_mode_max=max(
                            1, int(anisotropic_edge_settings.glueball_momentum_mode_max)
                        ),
                    )
                    su3_glueball_result = su3_glueball_bundle.get("glueball")
                    momentum_glueball_results = {
                        name: result
                        for name, result in su3_glueball_bundle.items()
                        if name.startswith("glueball_momentum_p")
                    }
                except Exception as exc:
                    su3_msg = f"SU(3) glueball estimators failed: {exc}"
                    glueball_systematics_error = (
                        f"{glueball_systematics_error}; {su3_msg}"
                        if glueball_systematics_error
                        else su3_msg
                    )
                spatial_glueball_result = None
                spatial_frame_idx = None
                spatial_rho_edge = None
                spatial_error = None
                spatial_tensor_result = None
                spatial_tensor_frame_idx = None
                spatial_tensor_rho_edge = None
                spatial_tensor_error = (
                    "disabled (spatial radial estimator removed from anisotropic-edge tab)"
                )
                spatial_tensor_traceless_result = None
                spatial_tensor_traceless_frame_idx = None
                spatial_tensor_traceless_rho_edge = None
                spatial_tensor_traceless_error = (
                    "disabled (spatial radial estimator removed from anisotropic-edge tab)"
                )
                spatial_error = (
                    "disabled (spatial radial estimator removed from anisotropic-edge tab)"
                )
                results = output.channel_results
                state["anisotropic_edge_results"] = results
                state["anisotropic_edge_glueball_strong_result"] = strong_glueball_result
                state["anisotropic_edge_glueball_edge_iso_result"] = edge_iso_glueball_result
                state["anisotropic_edge_glueball_su3_result"] = su3_glueball_result
                state["anisotropic_edge_glueball_momentum_results"] = momentum_glueball_results
                state["anisotropic_edge_glueball_systematics_error"] = glueball_systematics_error
                state["anisotropic_edge_glueball_spatial_result"] = spatial_glueball_result
                state["anisotropic_edge_glueball_spatial_frame_idx"] = spatial_frame_idx
                state["anisotropic_edge_glueball_spatial_rho_edge"] = spatial_rho_edge
                state["anisotropic_edge_glueball_spatial_error"] = spatial_error
                state["anisotropic_edge_tensor_spatial_result"] = spatial_tensor_result
                state["anisotropic_edge_tensor_spatial_frame_idx"] = spatial_tensor_frame_idx
                state["anisotropic_edge_tensor_spatial_rho_edge"] = spatial_tensor_rho_edge
                state["anisotropic_edge_tensor_spatial_error"] = spatial_tensor_error
                state["anisotropic_edge_tensor_strong_result"] = strong_tensor_result
                state["anisotropic_edge_tensor_momentum_results"] = tensor_momentum_results
                state["anisotropic_edge_tensor_momentum_meta"] = tensor_momentum_meta
                state["anisotropic_edge_tensor_systematics_error"] = tensor_systematics_error
                state["anisotropic_edge_tensor_traceless_spatial_result"] = (
                    spatial_tensor_traceless_result
                )
                state["anisotropic_edge_tensor_traceless_spatial_frame_idx"] = (
                    spatial_tensor_traceless_frame_idx
                )
                state["anisotropic_edge_tensor_traceless_spatial_rho_edge"] = (
                    spatial_tensor_traceless_rho_edge
                )
                state["anisotropic_edge_tensor_traceless_spatial_error"] = (
                    spatial_tensor_traceless_error
                )
                _update_anisotropic_edge_plots(results)
                _update_anisotropic_edge_tables(results)
                _update_anisotropic_edge_glueball_crosscheck(
                    results=results,
                    strong_result=strong_glueball_result,
                    edge_iso_result=edge_iso_glueball_result,
                    su3_result=su3_glueball_result,
                    momentum_results=momentum_glueball_results,
                    spatial_result=spatial_glueball_result,
                    history=history,
                    spatial_frame_idx=spatial_frame_idx,
                    spatial_rho_edge=spatial_rho_edge,
                    spatial_error=spatial_error,
                    tensor_spatial_result=spatial_tensor_result,
                    tensor_spatial_frame_idx=spatial_tensor_frame_idx,
                    tensor_spatial_rho_edge=spatial_tensor_rho_edge,
                    tensor_spatial_error=spatial_tensor_error,
                    tensor_strong_result=strong_tensor_result,
                    tensor_momentum_results=tensor_momentum_results,
                    tensor_momentum_meta=tensor_momentum_meta,
                    tensor_systematics_error=tensor_systematics_error,
                    tensor_traceless_spatial_result=spatial_tensor_traceless_result,
                    tensor_traceless_spatial_frame_idx=spatial_tensor_traceless_frame_idx,
                    tensor_traceless_spatial_rho_edge=spatial_tensor_traceless_rho_edge,
                    tensor_traceless_spatial_error=spatial_tensor_traceless_error,
                    glueball_systematics_error=glueball_systematics_error,
                )
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
                # Match anisotropic-edge visualization: show C(t) on linear axis
                # so fitted decays appear as exponentials rather than straight lines.
                correlator_logy=False,
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

        # =====================================================================
        # New Dirac/Electroweak tab callbacks (unified observables)
        # =====================================================================

        def _extract_observed_refs_from_table(
            table: pn.widgets.Tabulator,
            key_col: str = "observable",
            value_col: str = "observed_GeV",
        ) -> dict[str, float]:
            refs: dict[str, float] = {}
            df = table.value
            if not isinstance(df, pd.DataFrame):
                return refs
            for _, row in df.iterrows():
                key = str(row.get(key_col, "")).strip()
                if not key:
                    continue
                raw = row.get(value_col)
                if isinstance(raw, str):
                    raw = raw.strip()
                    if raw == "":
                        continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    refs[key] = val
            return refs

        def _update_new_dirac_ew_electroweak_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = new_dirac_ew_mass_mode.value
            _update_electroweak_tables_generic(
                results,
                mode,
                new_dirac_ew_mass_table,
                new_dirac_ew_ratio_pane,
                new_dirac_ew_ratio_table,
                new_dirac_ew_fit_table,
                new_dirac_ew_anchor_table,
                new_dirac_ew_compare_table,
                new_dirac_ew_ref_table,
            )

        def _update_new_dirac_ew_derived_tables(
            bundle,
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = new_dirac_ew_mass_mode.value

            observed = _extract_observed_refs_from_table(new_dirac_ew_observed_table)
            ew_results = bundle.electroweak_output.channel_results
            ew_masses = _extract_masses(ew_results, mode=mode, family_map=None)
            electron_component_mass = _get_channel_mass(bundle.electron_component_result, mode=mode)
            higgs_sigma_mass = _get_channel_mass(bundle.higgs_sigma_result, mode=mode)

            color_singlet = bundle.color_singlet_spectrum
            electron_dirac = color_singlet.electron_mass if color_singlet is not None else None

            rows: list[dict[str, Any]] = []
            rows.append(
                {
                    "observable": "electron_dirac",
                    "measured": electron_dirac,
                    "observed_GeV": observed.get("electron_dirac"),
                    "error_pct": (
                        ((electron_dirac - observed["electron_dirac"]) / observed["electron_dirac"] * 100.0)
                        if electron_dirac is not None and observed.get("electron_dirac", 0) > 0
                        else None
                    ),
                    "note": "Dirac spectral color-singlet proxy",
                }
            )
            rows.append(
                {
                    "observable": "electron_yukawa",
                    "measured": bundle.electron_mass_yukawa,
                    "observed_GeV": observed.get("electron_yukawa"),
                    "error_pct": (
                        (bundle.electron_mass_yukawa - observed["electron_yukawa"])
                        / observed["electron_yukawa"]
                        * 100.0
                        if observed.get("electron_yukawa", 0) > 0
                        else None
                    ),
                    "note": "Yukawa prediction y_e * v",
                }
            )
            rows.append(
                {
                    "observable": "electron_component",
                    "measured": electron_component_mass,
                    "observed_GeV": observed.get("electron_component"),
                    "error_pct": (
                        (electron_component_mass - observed["electron_component"])
                        / observed["electron_component"]
                        * 100.0
                        if observed.get("electron_component", 0) > 0
                        else None
                    ),
                    "note": "Lower SU(2) doublet correlator mass",
                }
            )
            rows.append(
                {
                    "observable": "higgs_sigma",
                    "measured": higgs_sigma_mass,
                    "observed_GeV": observed.get("higgs_sigma"),
                    "error_pct": (
                        (higgs_sigma_mass - observed["higgs_sigma"]) / observed["higgs_sigma"] * 100.0
                        if observed.get("higgs_sigma", 0) > 0
                        else None
                    ),
                    "note": "Radial fluctuation (sigma-mode) correlator mass",
                }
            )
            for channel, mass in sorted(ew_masses.items()):
                obs = observed.get(channel)
                rows.append(
                    {
                        "observable": channel,
                        "measured": mass,
                        "observed_GeV": obs,
                        "error_pct": (
                            (mass - obs) / obs * 100.0 if obs is not None and obs > 0 else None
                        ),
                        "note": "Electroweak proxy channel",
                    }
                )
            new_dirac_ew_observable_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

            ratio_rows: list[dict[str, Any]] = []
            m_w = ew_masses.get("su2_phase")
            m_z = ew_masses.get("su2_doublet")
            m_h = higgs_sigma_mass if higgs_sigma_mass > 0 else None
            if m_w is not None and m_h is not None and m_h > 0:
                measured = m_w / m_h
                obs = None
                if observed.get("su2_phase", 0) > 0 and observed.get("higgs_sigma", 0) > 0:
                    obs = observed["su2_phase"] / observed["higgs_sigma"]
                ratio_rows.append(
                    {
                        "ratio": "mW/mH",
                        "measured": measured,
                        "observed": obs,
                        "error_pct": ((measured - obs) / obs * 100.0 if obs and obs > 0 else None),
                    }
                )
            if m_z is not None and m_h is not None and m_h > 0:
                measured = m_z / m_h
                obs = None
                if observed.get("su2_doublet", 0) > 0 and observed.get("higgs_sigma", 0) > 0:
                    obs = observed["su2_doublet"] / observed["higgs_sigma"]
                ratio_rows.append(
                    {
                        "ratio": "mZ/mH",
                        "measured": measured,
                        "observed": obs,
                        "error_pct": ((measured - obs) / obs * 100.0 if obs and obs > 0 else None),
                    }
                )
            if bundle.electron_mass_yukawa > 0 and electron_component_mass > 0:
                ratio_rows.append(
                    {
                        "ratio": "m(e_component)/m(e_yukawa)",
                        "measured": electron_component_mass / bundle.electron_mass_yukawa,
                        "observed": 1.0,
                        "error_pct": (
                            (electron_component_mass / bundle.electron_mass_yukawa - 1.0) * 100.0
                        ),
                    }
                )
            new_dirac_ew_ratio_table_extra.value = (
                pd.DataFrame(ratio_rows) if ratio_rows else pd.DataFrame()
            )

            summary_lines = [
                "## Electroweak Summary",
                f"- Frames used: `{len(bundle.frame_indices)}`",
                f"- Electroweak channels with samples: "
                f"`{len([r for r in ew_results.values() if r.n_samples > 0])}`",
                f"- Higgs VEV (snapshot): `{bundle.higgs_vev:.6f}` ± `{bundle.higgs_vev_std:.6f}`",
                f"- Higgs VEV (time mean): `{bundle.vev_time_mean:.6f}` ± `{bundle.vev_time_std:.6f}`",
                f"- Yukawa prediction: `m_e={bundle.electron_mass_yukawa:.6f}`, "
                f"`y_e={bundle.yukawa_e:.6f}`, `ΔΦ_e={bundle.fitness_delta_phi_e:.6f}`, "
                f"`Φ_0={bundle.fitness_phi0:.6f}`",
            ]
            if color_singlet is not None:
                summary_lines.append(
                    f"- Dirac color-singlet threshold q={new_dirac_ew_settings.color_singlet_quantile:.2f}: "
                    f"`{color_singlet.lepton_threshold:.6f}` "
                    f"(modes={color_singlet.n_singlet_modes}/{len(color_singlet.masses)})"
                )
                if color_singlet.electron_mass is not None:
                    summary_lines.append(
                        f"- Dirac electron proxy mass: `{color_singlet.electron_mass:.6f}`"
                    )
            new_dirac_ew_summary.object = "\n".join(summary_lines)

        def _on_new_dirac_ew_mass_mode_change(event) -> None:
            results = state.get("new_dirac_ew_results")
            bundle = state.get("new_dirac_ew_bundle")
            if results is None or bundle is None:
                return
            _update_new_dirac_ew_electroweak_tables(results, event.new)
            _update_new_dirac_ew_derived_tables(bundle, event.new)

        new_dirac_ew_mass_mode.param.watch(_on_new_dirac_ew_mass_mode_change, "value")

        def on_run_new_dirac_electroweak(_):
            def _compute(history):
                neighbor_method = (
                    "auto"
                    if new_dirac_ew_settings.neighbor_method == "voronoi"
                    else new_dirac_ew_settings.neighbor_method
                )
                ew_cfg = ElectroweakChannelConfig(
                    warmup_fraction=new_dirac_ew_settings.simulation_range[0],
                    end_fraction=new_dirac_ew_settings.simulation_range[1],
                    max_lag=new_dirac_ew_settings.max_lag,
                    h_eff=new_dirac_ew_settings.h_eff,
                    use_connected=new_dirac_ew_settings.use_connected,
                    neighbor_method=neighbor_method,
                    companion_topology=str(new_dirac_ew_settings.companion_topology),
                    edge_weight_mode=new_dirac_ew_settings.edge_weight_mode,
                    neighbor_k=new_dirac_ew_settings.neighbor_k,
                    window_widths=_parse_window_widths(new_dirac_ew_settings.window_widths_spec),
                    fit_mode=new_dirac_ew_settings.fit_mode,
                    fit_start=new_dirac_ew_settings.fit_start,
                    fit_stop=new_dirac_ew_settings.fit_stop,
                    min_fit_points=new_dirac_ew_settings.min_fit_points,
                    epsilon_d=new_dirac_ew_settings.epsilon_d,
                    epsilon_c=new_dirac_ew_settings.epsilon_c,
                    epsilon_clone=new_dirac_ew_settings.epsilon_clone,
                    lambda_alg=new_dirac_ew_settings.lambda_alg,
                    mc_time_index=new_dirac_ew_settings.mc_time_index,
                    compute_bootstrap_errors=new_dirac_ew_settings.compute_bootstrap_errors,
                    n_bootstrap=new_dirac_ew_settings.n_bootstrap,
                )
                threshold = (
                    new_dirac_ew_settings.dirac_color_threshold_value
                    if new_dirac_ew_settings.dirac_color_threshold_mode == "manual"
                    else "median"
                )
                dirac_idx = None
                if new_dirac_ew_settings.mc_time_index is not None:
                    try:
                        raw_idx = int(new_dirac_ew_settings.mc_time_index)
                    except (TypeError, ValueError):
                        raw_idx = None
                    if raw_idx is not None:
                        if raw_idx in history.recorded_steps:
                            raw_idx = int(history.get_step_index(raw_idx))
                        dirac_idx = max(raw_idx - 1, 0)
                dirac_cfg = DiracSpectrumConfig(
                    mc_time_index=dirac_idx,
                    epsilon_clone=(
                        float(new_dirac_ew_settings.epsilon_clone)
                        if new_dirac_ew_settings.epsilon_clone is not None
                        else 0.01
                    ),
                    kernel_mode=str(new_dirac_ew_settings.dirac_kernel_mode),
                    epsilon_c=new_dirac_ew_settings.epsilon_c,
                    lambda_alg=(
                        float(new_dirac_ew_settings.lambda_alg)
                        if new_dirac_ew_settings.lambda_alg is not None
                        else 1.0
                    ),
                    h_eff=float(new_dirac_ew_settings.h_eff),
                    include_phase=True,
                    color_threshold=threshold,
                    time_average=bool(new_dirac_ew_settings.dirac_time_average),
                    warmup_fraction=float(new_dirac_ew_settings.dirac_warmup_fraction),
                    max_avg_frames=int(new_dirac_ew_settings.dirac_max_avg_frames),
                )
                bundle_cfg = DiracElectroweakConfig(
                    electroweak=ew_cfg,
                    electroweak_channels=[
                        c.strip()
                        for c in str(new_dirac_ew_settings.channel_list).split(",")
                        if c.strip()
                    ],
                    dirac=dirac_cfg,
                    color_singlet_quantile=float(new_dirac_ew_settings.color_singlet_quantile),
                    sigma_max_lag=int(new_dirac_ew_settings.max_lag),
                    sigma_use_connected=bool(new_dirac_ew_settings.use_connected),
                    sigma_fit_mode=str(new_dirac_ew_settings.fit_mode),
                    sigma_fit_start=int(new_dirac_ew_settings.fit_start),
                    sigma_fit_stop=new_dirac_ew_settings.fit_stop,
                    sigma_min_fit_points=int(new_dirac_ew_settings.min_fit_points),
                    sigma_window_widths=_parse_window_widths(new_dirac_ew_settings.window_widths_spec),
                    sigma_compute_bootstrap_errors=bool(new_dirac_ew_settings.compute_bootstrap_errors),
                    sigma_n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                )
                bundle = compute_dirac_electroweak_bundle(history, config=bundle_cfg)
                state["new_dirac_ew_bundle"] = bundle
                results = bundle.electroweak_output.channel_results
                state["new_dirac_ew_results"] = results

                _update_electroweak_plots_generic(
                    results,
                    new_dirac_ew_channel_plots,
                    new_dirac_ew_plots_spectrum,
                    new_dirac_ew_plots_overlay_corr,
                    new_dirac_ew_plots_overlay_meff,
                )
                _update_new_dirac_ew_electroweak_tables(results)

                couplings = _compute_coupling_constants(
                    history,
                    h_eff=float(new_dirac_ew_settings.h_eff),
                    epsilon_d=new_dirac_ew_settings.epsilon_d,
                    epsilon_c=new_dirac_ew_settings.epsilon_c,
                )
                new_dirac_ew_coupling_table.value = pd.DataFrame(
                    _build_coupling_rows(
                        couplings,
                        proxies=None,
                        include_strong=False,
                        refs=_extract_coupling_refs(new_dirac_ew_coupling_ref_table),
                    )
                )

                dirac_plots = build_all_dirac_plots(bundle.dirac_result)
                new_dirac_ew_dirac_full.object = dirac_plots["full_spectrum"]
                sector_plots = dirac_plots["sector_spectra"]
                new_dirac_ew_dirac_up.object = sector_plots.get("up_quark")
                new_dirac_ew_dirac_down.object = sector_plots.get("down_quark")
                new_dirac_ew_dirac_nu.object = sector_plots.get("neutrino")
                new_dirac_ew_dirac_lep.object = sector_plots.get("charged_lepton")
                new_dirac_ew_dirac_walker.object = dirac_plots["walker_classification"]
                new_dirac_ew_dirac_mass_hierarchy.object = dirac_plots["mass_hierarchy"]
                new_dirac_ew_dirac_chiral.object = dirac_plots["chiral_density"]
                new_dirac_ew_dirac_generation_ratios.object = dirac_plots["generation_ratios"]
                comp_rows, best_scale = build_fermion_comparison(bundle.dirac_result)
                new_dirac_ew_dirac_comparison.value = (
                    pd.DataFrame(comp_rows) if comp_rows else pd.DataFrame()
                )
                new_dirac_ew_dirac_ratio.object = dirac_plots["fermion_ratio_comparison"]
                sector_counts = {name: spec.n_walkers for name, spec in bundle.dirac_result.sectors.items()}
                new_dirac_ew_dirac_summary.object = "  \n".join(
                    [
                        (
                            f"**Best-fit scale:** {best_scale:.6g} GeV/σ"
                            if best_scale
                            else "**Best-fit scale:** N/A"
                        ),
                        f"**Chiral condensate:** ⟨ψ̄ψ⟩ ≈ {bundle.dirac_result.chiral_condensate:.4f}",
                        "**Sector walkers:** " + ", ".join(f"{k}: {v}" for k, v in sector_counts.items()),
                    ]
                )

                color_singlet = bundle.color_singlet_spectrum
                if color_singlet is None or len(color_singlet.masses) == 0:
                    new_dirac_ew_color_singlet_table.value = pd.DataFrame()
                else:
                    rows = []
                    max_rows = min(len(color_singlet.masses), 300)
                    for i in range(max_rows):
                        ls = float(color_singlet.lepton_scores[i])
                        rows.append(
                            {
                                "mode": int(color_singlet.mode_index[i]),
                                "mass": float(color_singlet.masses[i]),
                                "lepton_score": ls,
                                "quark_score": float(color_singlet.quark_scores[i]),
                                "is_singlet": bool(ls >= color_singlet.lepton_threshold),
                            }
                        )
                    new_dirac_ew_color_singlet_table.value = pd.DataFrame(rows)

                e_plot = ChannelPlot(
                    bundle.electron_component_result,
                    logy=False,
                    width=420,
                    height=320,
                ).side_by_side()
                sigma_plot = ChannelPlot(
                    bundle.higgs_sigma_result,
                    logy=False,
                    width=420,
                    height=320,
                ).side_by_side()
                new_dirac_ew_electron_plot.objects = [
                    e_plot
                    if e_plot is not None
                    else pn.pane.Markdown("_No electron-component data._")
                ]
                new_dirac_ew_sigma_plot.objects = [
                    sigma_plot
                    if sigma_plot is not None
                    else pn.pane.Markdown("_No sigma-mode data._")
                ]

                _update_new_dirac_ew_derived_tables(bundle)

                n_channels = len([r for r in results.values() if r.n_samples > 0])
                new_dirac_ew_status.object = (
                    f"**Complete:** Electroweak computed "
                    f"({n_channels} electroweak channels; Dirac tab now updated too)."
                )

            _run_tab_computation(
                state,
                new_dirac_ew_status,
                "new Dirac/electroweak observables",
                _compute,
            )

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
        algorithm_run_button.on_click(on_run_algorithm_analysis)
        fractal_set_run_button.on_click(on_run_fractal_set)
        einstein_run_button.on_click(on_run_einstein_test)
        channels_run_button.on_click(on_run_channels)
        radial_run_button.on_click(on_run_radial_channels)
        anisotropic_edge_run_button.on_click(on_run_anisotropic_edge_channels)
        radial_ew_run_button.on_click(on_run_radial_electroweak)
        electroweak_run_button.on_click(on_run_electroweak)
        new_dirac_ew_run_button.on_click(on_run_new_dirac_electroweak)
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

            algorithm_tab = pn.Column(
                algorithm_status,
                pn.Row(algorithm_run_button, sizing_mode="stretch_width"),
                pn.pane.Markdown(
                    "_Vectorized diagnostics computed directly from collected RunHistory traces._"
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("### Percentage of Clones Over Time"),
                algorithm_clone_plot,
                pn.pane.Markdown("### Mean Fitness Over Time (p95 error bars)"),
                algorithm_fitness_plot,
                pn.pane.Markdown("### Mean Rewards Over Time (p95 error bars)"),
                algorithm_reward_plot,
                pn.pane.Markdown("### Companion Distances (Clone vs Random, p95 error bars)"),
                algorithm_companion_plot,
                pn.pane.Markdown("### Average Inter-Walker Distance Over Time (mean ± 1σ)"),
                algorithm_interwalker_plot,
                pn.pane.Markdown("### Lyapunov Convergence Over Time"),
                algorithm_lyapunov_plot,
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
                sizing_mode="stretch_both",
            )

            einstein_tab = pn.Column(
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
                channel_anchor_mode,
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
direct-neighbor triplets or (distance companion, clone companion). Enable
**Baryon Triplet Settings** to override nucleon with companion-triplet
color-determinant baryon correlator. Enable **Meson Phase Settings** to
override scalar/pseudoscalar with companion-pair color-phase correlators
Re(c_i†c_j) / Im(c_i†c_j). Enable **Vector Meson Settings** to override
vector/axial_vector with companion-pair color-displacement correlators
Re(c_i†c_j)Δx / Im(c_i†c_j)Δx. Enable **Glueball Color Settings** momentum
projection to tune `glueball_momentum_pn` channels (n=0..n_max). The four
glueball estimators (anisotropic-edge isotropic, strong-force isotropic, SU(3)
plaquette, SU(3) momentum-projected) are always computed for comparison. Enable
**Tensor Momentum Settings** to compute `tensor_momentum_pn` and per-component
spin-2 momentum channels for polarization/dispersion tests.""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            anisotropic_edge_tab = pn.Column(
                anisotropic_edge_status,
                anisotropic_edge_note,
                pn.Row(anisotropic_edge_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Anisotropic Edge Settings", anisotropic_edge_settings_layout),
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
                pn.pane.Markdown("### Glueball Cross-Check (Edge-Isotropic vs Strong-Isotropic)"),
                anisotropic_edge_glueball_compare_ratio,
                pn.pane.Markdown("### Glueball Approach Comparison"),
                anisotropic_edge_glueball_approach_summary,
                anisotropic_edge_glueball_approach_table,
                pn.pane.Markdown("### Glueball Pairwise Discrepancies"),
                anisotropic_edge_glueball_pairwise_table,
                pn.pane.Markdown("### Glueball Consensus / Systematics"),
                anisotropic_edge_glueball_systematics_badge,
                anisotropic_edge_glueball_consensus_summary,
                anisotropic_edge_glueball_consensus_plot,
                pn.pane.Markdown("### Glueball Momentum Dispersion"),
                anisotropic_edge_glueball_dispersion_plot,
                pn.pane.Markdown("### Tensor Approach Comparison"),
                anisotropic_edge_tensor_systematics_badge,
                anisotropic_edge_tensor_approach_summary,
                anisotropic_edge_tensor_approach_table,
                pn.pane.Markdown("### Tensor Momentum Dispersion"),
                anisotropic_edge_tensor_dispersion_plot,
                pn.pane.Markdown("### Tensor Component Dispersion"),
                anisotropic_edge_tensor_component_dispersion_plot,
                pn.pane.Markdown("### Glueball Lorentz Check (4 Estimators + Dispersion)"),
                anisotropic_edge_glueball_lorentz_ratio,
                pn.pane.Markdown("### Tensor Lorentz Check (Legacy isotropic tensor)"),
                anisotropic_edge_tensor_lorentz_ratio,
                pn.pane.Markdown("### Tensor Lorentz Check (Traceless tensor)"),
                anisotropic_edge_tensor_traceless_lorentz_ratio,
                pn.pane.Markdown("### Lorentz/Anisotropy Corrections (Glueball 4-way)"),
                anisotropic_edge_lorentz_correction_summary,
                pn.pane.Markdown("### Glueball Estimator Plots"),
                pn.GridBox(
                    pn.Column(
                        pn.pane.Markdown("#### Anisotropic Edge Glueball"),
                        anisotropic_edge_glueball_aniso_plot,
                        sizing_mode="stretch_width",
                    ),
                    pn.Column(
                        pn.pane.Markdown("#### Strong-Force Glueball"),
                        anisotropic_edge_glueball_strong_plot,
                        sizing_mode="stretch_width",
                    ),
                    pn.Column(
                        pn.pane.Markdown("#### SU(3) Plaquette Glueball"),
                        anisotropic_edge_glueball_plaquette_plot,
                        sizing_mode="stretch_width",
                    ),
                    pn.Column(
                        pn.pane.Markdown("#### SU(3) Momentum Glueball (p0)"),
                        anisotropic_edge_glueball_momentum_p0_plot,
                        sizing_mode="stretch_width",
                    ),
                    ncols=2,
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown("### Best-Fit Scales"),
                anisotropic_edge_fit_table,
                pn.pane.Markdown("### Anchored Mass Table"),
                anisotropic_edge_anchor_mode,
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

            new_dirac_ew_note = pn.pane.Alert(
                """**Electroweak:** Proxy channels and coupled observables from the unified run.
Dirac-sector analyses are shown in the dedicated Dirac tab.""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            new_dirac_ew_tab = pn.Column(
                new_dirac_ew_status,
                new_dirac_ew_note,
                pn.Row(new_dirac_ew_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Electroweak Settings", new_dirac_ew_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                new_dirac_ew_summary,
                pn.pane.Markdown("### Electroweak Couplings"),
                new_dirac_ew_coupling_table,
                pn.pane.Markdown("### Electroweak Coupling References"),
                new_dirac_ew_coupling_ref_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Electroweak Proxy Channels"),
                new_dirac_ew_mass_mode,
                new_dirac_ew_plots_spectrum,
                new_dirac_ew_mass_table,
                new_dirac_ew_ratio_pane,
                new_dirac_ew_ratio_table,
                new_dirac_ew_fit_table,
                new_dirac_ew_compare_table,
                new_dirac_ew_anchor_table,
                pn.pane.Markdown("### Electroweak Reference Masses (GeV)"),
                new_dirac_ew_ref_table,
                pn.pane.Markdown("### All Channels Overlay - Correlators"),
                new_dirac_ew_plots_overlay_corr,
                pn.pane.Markdown("### All Channels Overlay - Effective Masses"),
                new_dirac_ew_plots_overlay_meff,
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                new_dirac_ew_channel_plots,
                sizing_mode="stretch_both",
            )

            dirac_tab = pn.Column(
                pn.pane.Markdown(
                    "_Dirac outputs are computed together with Electroweak observables. "
                    "Run the Electroweak tab to refresh these plots and tables._"
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("### Dirac Spectrum (Antisymmetric Kernel)"),
                pn.Row(
                    new_dirac_ew_dirac_full,
                    new_dirac_ew_dirac_walker,
                    sizing_mode="stretch_width",
                ),
                pn.GridBox(
                    new_dirac_ew_dirac_up,
                    new_dirac_ew_dirac_down,
                    new_dirac_ew_dirac_nu,
                    new_dirac_ew_dirac_lep,
                    ncols=2,
                ),
                pn.Row(
                    new_dirac_ew_dirac_mass_hierarchy,
                    new_dirac_ew_dirac_chiral,
                    sizing_mode="stretch_width",
                ),
                new_dirac_ew_dirac_generation_ratios,
                new_dirac_ew_dirac_summary,
                new_dirac_ew_dirac_comparison,
                new_dirac_ew_dirac_ratio,
                pn.layout.Divider(),
                pn.pane.Markdown("### Color-Singlet Mode Projection"),
                new_dirac_ew_color_singlet_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Electron/Higgs Proxy Correlators"),
                pn.Row(new_dirac_ew_electron_plot, new_dirac_ew_sigma_plot, sizing_mode="stretch_width"),
                pn.pane.Markdown("### Observed Mass Inputs (GeV)"),
                new_dirac_ew_observed_table,
                pn.pane.Markdown("### Derived Observable Comparison"),
                new_dirac_ew_observable_table,
                pn.pane.Markdown("### Cross-Sector Ratios"),
                new_dirac_ew_ratio_table_extra,
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
                    ("Algorithm", algorithm_tab),
                    ("Holographic Principle", fractal_set_tab),
                    ("Strong Force", anisotropic_edge_tab),
                    ("Electroweak", new_dirac_ew_tab),
                    ("Einsten Equation", einstein_tab),
                    ("Dirac", dirac_tab),
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
