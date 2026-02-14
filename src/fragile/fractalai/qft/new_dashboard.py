"""QFT dashboard with simulation and analysis tabs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import itertools
import os
from pathlib import Path
import re
import time
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.fractalai.core.benchmarks import prepare_benchmark_for_explorer
from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel
from fragile.fractalai.qft.aggregation import compute_all_operator_series
from fragile.fractalai.qft.anisotropic_edge_channels import (
    _extract_edges_for_frame,
    _resolve_edge_weights,
    AnisotropicEdgeChannelConfig,
    AnisotropicEdgeChannelOutput,
    compute_anisotropic_edge_channels,
)
from fragile.fractalai.qft.baryon_triplet_channels import (
    _resolve_frame_indices as _resolve_baryon_frame_indices,
    BaryonTripletCorrelatorConfig,
    compute_companion_baryon_correlator,
)
from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    ChannelConfig,
    ChannelCorrelatorResult,
    compute_channel_correlator,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.coupling_diagnostics import (
    compute_coupling_diagnostics,
    CouplingDiagnosticsConfig,
)
from fragile.fractalai.qft.dashboard.channel_dashboard import (
    build_multiscale_tab_layout,
    clear_multiscale_tab,
    create_multiscale_widgets,
    update_multiscale_tab,
)
from fragile.fractalai.qft.dashboard.tensor_gevp_dashboard import (
    build_tensor_gevp_calibration_tab_layout,
    clear_tensor_gevp_calibration_tab,
    create_tensor_gevp_calibration_widgets,
    GEVP_DIRTY_STATE_KEY as TENSOR_GEVP_DIRTY_STATE_KEY,
    update_tensor_gevp_calibration_tab,
)
from fragile.fractalai.qft.dirac_electroweak import (
    compute_dirac_electroweak_bundle,
    DiracElectroweakConfig,
)
from fragile.fractalai.qft.dirac_spectrum import (
    build_fermion_comparison,
    DiracSpectrumConfig,
)
from fragile.fractalai.qft.dirac_spectrum_plotting import build_all_dirac_plots
from fragile.fractalai.qft.einstein_equations import (
    compute_einstein_test,
    EinsteinConfig,
)
from fragile.fractalai.qft.einstein_equations_plotting import build_all_einstein_plots
from fragile.fractalai.qft.electroweak_channels import (
    compute_electroweak_channels,
    compute_electroweak_coupling_constants,
    ELECTROWEAK_CHANNELS,
    ElectroweakChannelConfig,
    ELECTROWEAK_PARITY_CHANNELS,
    ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
)
from fragile.fractalai.qft.dashboard.gevp_dashboard import (
    GEVPDashboardWidgets,
    build_gevp_dashboard_sections,
    clear_gevp_dashboard,
    create_gevp_dashboard_widgets,
    update_gevp_dashboard,
)
from fragile.fractalai.qft.gevp_channels import (
    compute_companion_channel_gevp,
    EW_MIXED_GEVP_BASE_CHANNELS,
    get_companion_gevp_basis_channels,
    GEVPConfig,
    U1_GEVP_BASE_CHANNELS,
)
from fragile.fractalai.qft.glueball_color_channels import (
    compute_companion_glueball_color_correlator,
    GlueballColorCorrelatorConfig,
)
from fragile.fractalai.qft.meson_phase_channels import (
    compute_companion_meson_phase_correlator,
    MesonPhaseCorrelatorConfig,
)
from fragile.fractalai.qft.multiscale_electroweak import (
    compute_multiscale_electroweak_channels,
    EW_MIXED_BASE_CHANNELS,
    MultiscaleElectroweakConfig,
    MultiscaleElectroweakOutput,
    SU2_BASE_CHANNELS,
    SU2_DIRECTIONAL_CHANNELS,
    SU2_WALKER_TYPE_CHANNELS,
    U1_BASE_CHANNELS,
)
from fragile.fractalai.qft.multiscale_strong_force import (
    COMPANION_CHANNEL_MAP,
    compute_multiscale_strong_force_channels,
    MultiscaleStrongForceConfig,
    MultiscaleStrongForceOutput,
)
from fragile.fractalai.qft.plotting import (
    build_all_channels_overlay,
    build_lyapunov_plot,
    build_mass_spectrum_bar,
    build_window_heatmap,
    CHANNEL_COLORS,
    ChannelPlot,
)
from fragile.fractalai.qft.radial_channels import (
    _apply_pbc_diff_torch,
    _compute_color_states_single,
    _recorded_subgraph_for_alive,
    _slice_bounds,
    compute_radial_channels,
    RadialChannelConfig,
)
from fragile.fractalai.qft.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
)
from fragile.fractalai.qft.tensor_momentum_channels import (
    compute_companion_tensor_momentum_correlator,
    TensorMomentumCorrelatorConfig,
)
from fragile.fractalai.qft.vector_meson_channels import (
    compute_companion_vector_meson_correlator,
    VectorMesonCorrelatorConfig,
)


# Prevent Plotly from probing the system browser during import.
os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

COMPANION_CHANNEL_VARIANTS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "pseudoscalar": (
        "pseudoscalar",
        "pseudoscalar_score_weighted",
    ),
    "scalar": (
        "scalar",
        "scalar_score_directed",
        "scalar_score_weighted",
        "scalar_raw",
        "scalar_abs2_vacsub",
    ),
    "vector": (
        "vector",
        "vector_score_directed",
        "vector_score_gradient",
        "vector_score_directed_longitudinal",
        "vector_score_directed_transverse",
    ),
    "axial_vector": (
        "axial_vector",
        "axial_vector_score_directed",
        "axial_vector_score_gradient",
        "axial_vector_score_directed_longitudinal",
        "axial_vector_score_directed_transverse",
    ),
    "nucleon": (
        "nucleon",
        "nucleon_flux_action",
        "nucleon_flux_sin2",
        "nucleon_flux_exp",
        "nucleon_score_signed",
        "nucleon_score_abs",
    ),
    "glueball": (
        "glueball",
        "glueball_phase_action",
        "glueball_phase_sin2",
    ),
    "tensor": ("tensor",),
}

DEFAULT_COMPANION_CHANNEL_VARIANT_SELECTION: dict[str, tuple[str, ...]] = {
    "pseudoscalar": ("pseudoscalar", "pseudoscalar_score_weighted"),
    "scalar": ("scalar", "scalar_raw", "scalar_abs2_vacsub"),
    "vector": (
        "vector",
        "vector_score_directed",
        "vector_score_gradient",
        "vector_score_directed_longitudinal",
        "vector_score_directed_transverse",
    ),
    "axial_vector": ("axial_vector",),
    "nucleon": (
        "nucleon",
        "nucleon_flux_action",
        "nucleon_flux_sin2",
        "nucleon_flux_exp",
        "nucleon_score_signed",
        "nucleon_score_abs",
    ),
    "glueball": ("glueball", "glueball_phase_action", "glueball_phase_sin2"),
    "tensor": ("tensor",),
}

ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "u1": (
        "u1_phase",
        "u1_dressed",
        "u1_phase_q2",
        "u1_dressed_q2",
    ),
    "su2": SU2_BASE_CHANNELS,
    "su2_directional": SU2_DIRECTIONAL_CHANNELS,
    "su2_walker_type": SU2_WALKER_TYPE_CHANNELS,
    "ew_mixed": ("ew_mixed",),
    "symmetry_breaking": ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
    "parity_velocity": ELECTROWEAK_PARITY_CHANNELS,
}

DEFAULT_ELECTROWEAK_CHANNEL_VARIANT_SELECTION: dict[str, tuple[str, ...]] = {
    "u1": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["u1"],
    "su2": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["su2"],
    "su2_directional": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["su2_directional"],
    "su2_walker_type": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["su2_walker_type"],
    "ew_mixed": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["ew_mixed"],
    "symmetry_breaking": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["symmetry_breaking"],
    "parity_velocity": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["parity_velocity"],
}

ELECTROWEAK_CANONICAL_CHANNEL_ORDER = (
    "u1_phase",
    "u1_dressed",
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
    "ew_mixed",
    "fitness_phase",
    "clone_indicator",
)

DEFAULT_ELECTROWEAK_CHANNELS_FOR_DIRAC = list(ELECTROWEAK_CHANNELS)


def _collect_multiselect_values(
    selectors: dict[str, pn.widgets.MultiSelect],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for selector in selectors.values():
        for value in selector.value or []:
            channel_name = str(value).strip()
            if not channel_name:
                continue
            if channel_name in seen:
                continue
            seen.add(channel_name)
            selected.append(channel_name)
    return selected


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

        if dim_spec.startswith("dim_"):
            # Extract spatial dimension
            dim_idx = int(dim_spec.split("_")[1])
            if dim_idx >= positions_all.shape[1]:
                # Dimension not available, return zeros
                return np.zeros(alive.sum())
            return positions_all[alive, dim_idx]

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

        if metric == "mc_time":
            # Color by MC time (constant for this frame)
            colors = np.full(alive.sum(), frame, dtype=float)
            return colors, True, {"title": "MC Time (frame)"}

        # Existing metrics
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
            # Compute radius from original positions (first 3 dims)
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
        """Determine axis ranges based on dimension mappings."""

        def get_range(dim_spec: str):
            if dim_spec == "mc_time":
                return [0, len(self._x) - 1]
            if dim_spec == "euclidean_time":
                extent = self.bounds_extent
                return [-extent, extent]
            if dim_spec.startswith("dim_"):
                # Use bounds extent for spatial dimensions
                extent = self.bounds_extent
                return [-extent, extent]
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
        """Build Delaunay edges using mapped 3D coordinates."""
        # Get edges from history
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
            x_coords = self._extract_dimension(self.x_axis_dim, mc_frame, positions_all, alive)
            y_coords = self._extract_dimension(self.y_axis_dim, mc_frame, positions_all, alive)
            z_coords = self._extract_dimension(self.z_axis_dim, mc_frame, positions_all, alive)

            # Extract color values
            colors, showscale, colorbar = self._get_color_values(mc_frame, positions_all, alive)

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


class AnisotropicEdgeSettings(param.Parameterized):
    """Settings for anisotropic edge-channel MC-time correlators."""

    simulation_range = param.Range(
        default=(0.2, 1.0),
        bounds=(0.0, 1.0),
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
    h_eff_mode = param.ObjectSelector(
        default="manual",
        objects=("manual", "auto_sigma_s"),
        doc=(
            "How to determine h_eff: "
            "'manual' uses the h_eff value; "
            "'auto_sigma_s' sets h_eff = sigma_S/2 from cloning score std dev."
        ),
    )
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
    baryon_operator_mode = param.ObjectSelector(
        default="det_abs",
        objects=("det_abs", "flux_action", "flux_sin2", "flux_exp", "score_signed", "score_abs"),
        doc="Primary baryon operator mode.",
    )
    baryon_flux_exp_alpha = param.Number(
        default=1.0,
        bounds=(0.0, None),
        doc="Exponent coefficient α for baryon flux_exp mode.",
    )
    use_companion_nucleon_gevp = param.Boolean(
        default=True,
        doc=(
            "Compute GEVP-combined companion channels "
            "(nucleon, scalar, pseudoscalar, glueball; vector excluded)."
        ),
    )
    gevp_basis_strategy = param.ObjectSelector(
        default="base_plus_best_scale",
        objects=("base_only", "base_plus_best_scale"),
        doc=(
            "GEVP basis strategy: "
            "'base_only' uses original nucleon operators, "
            "'base_plus_best_scale' augments each operator with its best multiscale estimator when available."
        ),
    )
    gevp_t0 = param.Integer(
        default=2,
        bounds=(1, None),
        doc="Reference lag t0 for generalized eigenvalue analysis.",
    )
    gevp_max_basis = param.Integer(
        default=10,
        bounds=(1, 64),
        doc="Maximum number of basis vectors retained before GEVP pruning.",
    )
    gevp_min_operator_r2 = param.Number(
        default=0.5,
        bounds=(-1.0, 1.0),
        doc=(
            "Minimum operator fit R² required for inclusion in GEVP basis. "
            "Set to -1 to disable R² filtering."
        ),
    )
    gevp_min_operator_windows = param.Integer(
        default=5,
        bounds=(0, None),
        doc=(
            "Minimum number of valid fit windows required for inclusion in GEVP basis. "
            "Set to 0 to disable window-count filtering."
        ),
    )
    gevp_max_operator_error_pct = param.Number(
        default=60.0,
        bounds=(0.0, None),
        doc=(
            "Maximum operator mass-error percentage (100*mass_error/mass) allowed in GEVP "
            "basis. Set to a large value to effectively disable this filter."
        ),
    )
    gevp_remove_artifacts = param.Boolean(
        default=True,
        doc=(
            "Remove artifact operators from GEVP basis "
            "(mass_error == 0, mass_error is NaN/Inf, or mass == 0)."
        ),
    )
    gevp_eig_rel_cutoff = param.Number(
        default=1e-2,
        bounds=(1e-8, 1.0),
        doc="Drop GEVP C(t0) modes with eigenvalue < cutoff * max_eigenvalue.",
    )
    gevp_cond_limit = param.Number(
        default=1e4,
        bounds=(1.0, None),
        doc="Maximum allowed condition number for the kept GEVP basis at t0.",
    )
    gevp_shrinkage = param.Number(
        default=1e-6,
        bounds=(0.0, 1.0),
        doc="Diagonal shrinkage fraction applied to C(t0) before eigendecomposition.",
    )
    gevp_bootstrap_mode = param.ObjectSelector(
        default="time",
        objects=("time", "walker", "hybrid"),
        doc="Bootstrap mode for GEVP uncertainty estimation.",
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
    glueball_operator_mode = param.ObjectSelector(
        default="auto",
        objects=("auto", "re_plaquette", "action_re_plaquette", "phase_action", "phase_sin2"),
        doc=(
            "Primary glueball operator mode. 'auto' keeps legacy behavior from "
            "glueball_use_action_form."
        ),
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
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))
    compute_bootstrap_errors = param.Boolean(default=False)
    n_bootstrap = param.Integer(default=100, bounds=(10, 1000))
    use_multiscale_kernels = param.Boolean(
        default=False,
        doc=(
            "Compute channels at multiple smearing scales with kernels from recorded "
            "neighbor/edge-weight data and auto-select best scale per channel."
        ),
    )
    n_scales = param.Integer(default=8, bounds=(2, 32))
    kernel_type = param.ObjectSelector(
        default="gaussian",
        objects=("gaussian", "exponential", "tophat", "shell"),
    )
    kernel_distance_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "floyd-warshall", "tropical"),
    )
    kernel_assume_all_alive = param.Boolean(
        default=True,
        doc="Assume all walkers alive when constructing kernel distances (fast path).",
    )
    kernel_batch_size = param.Integer(
        default=1,
        bounds=(1, 64),
        doc="Frame batch size for kernel construction.",
    )
    kernel_scale_frames = param.Integer(
        default=8,
        bounds=(1, 128),
        doc="Number of representative frames used to calibrate scales.",
    )
    kernel_scale_q_low = param.Number(default=0.05, bounds=(0.0, 0.95))
    kernel_scale_q_high = param.Number(default=0.95, bounds=(0.05, 1.0))
    kernel_bootstrap_mode = param.ObjectSelector(
        default="hybrid",
        objects=("time", "walker", "hybrid"),
        doc="Bootstrap strategy for multiscale kernels.",
    )
    kernel_bootstrap_max_walkers = param.Integer(
        default=512,
        bounds=(16, None),
        doc="Skip walker bootstrap when N exceeds this threshold.",
    )


class NewDiracElectroweakSettings(param.Parameterized):
    """Settings for the unified Dirac/Electroweak analysis tab."""

    simulation_range = param.Range(
        default=(0.2, 1.0),
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
    companion_topology_u1 = param.ObjectSelector(
        default="distance",
        objects=("distance", "clone", "both"),
        doc="Companion topology used by U(1) operator family.",
    )
    companion_topology_su2 = param.ObjectSelector(
        default="clone",
        objects=("distance", "clone", "both"),
        doc="Companion topology used by SU(2) operator family.",
    )
    companion_topology_ew_mixed = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion topology used by EW mixed operator family.",
    )
    edge_weight_mode = param.ObjectSelector(
        default="riemannian_kernel_volume",
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
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))
    epsilon_clone = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    lambda_alg = param.Number(default=None, bounds=(0.0, None), allow_None=True)
    su2_operator_mode = param.ObjectSelector(
        default="standard",
        objects=("standard", "score_directed"),
        doc="SU(2) operator branch (standard or score-directed).",
    )
    enable_walker_type_split = param.Boolean(
        default=False,
        doc="Emit cloner/resister/persister split SU(2) channels.",
    )
    walker_type_scope = param.ObjectSelector(
        default="frame_global",
        objects=("frame_global",),
        doc="Walker-type classification scope.",
    )
    compute_bootstrap_errors = param.Boolean(default=False)
    n_bootstrap = param.Integer(default=100, bounds=(10, 1000))
    use_multiscale_kernels = param.Boolean(
        default=True,
        doc=(
            "Compute multiscale SU(2) channels using recorded kernel distances and "
            "promote best-scale estimators."
        ),
    )
    n_scales = param.Integer(default=8, bounds=(2, 32))
    kernel_type = param.ObjectSelector(
        default="gaussian",
        objects=("gaussian", "exponential", "tophat", "shell"),
    )
    kernel_distance_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "floyd-warshall", "tropical"),
    )
    kernel_assume_all_alive = param.Boolean(default=True)
    kernel_batch_size = param.Integer(default=1, bounds=(1, 64))
    kernel_scale_frames = param.Integer(default=8, bounds=(1, 128))
    kernel_scale_q_low = param.Number(default=0.05, bounds=(0.0, 0.95))
    kernel_scale_q_high = param.Number(default=0.95, bounds=(0.05, 1.0))
    kernel_bootstrap_mode = param.ObjectSelector(
        default="time",
        objects=("time", "walker", "hybrid"),
        doc="Bootstrap strategy for multiscale SU(2) channels.",
    )

    use_su2_gevp = param.Boolean(
        default=True,
        doc="Compute GEVP-combined SU(2) channel from SU(2) operator family.",
    )
    use_u1_gevp = param.Boolean(default=True, doc="Compute GEVP-combined U(1) channel.")
    use_ew_mixed_gevp = param.Boolean(default=True, doc="Compute GEVP-combined EW mixed channel.")
    gevp_basis_strategy = param.ObjectSelector(
        default="base_plus_best_scale",
        objects=("base_only", "base_plus_best_scale"),
    )
    gevp_t0 = param.Integer(default=2, bounds=(1, None))
    gevp_max_basis = param.Integer(default=10, bounds=(1, 64))
    gevp_min_operator_r2 = param.Number(
        default=0.5,
        bounds=(-1.0, 1.0),
        doc="Minimum operator fit R² required for GEVP basis inclusion.",
    )
    gevp_min_operator_windows = param.Integer(
        default=5,
        bounds=(0, None),
        doc="Minimum valid fit-window count required for GEVP basis inclusion.",
    )
    gevp_max_operator_error_pct = param.Number(
        default=60.0,
        bounds=(0.0, None),
        doc="Maximum allowed operator mass error percentage for GEVP basis inclusion.",
    )
    gevp_remove_artifacts = param.Boolean(
        default=True,
        doc=(
            "Remove artifact operators (mass_error==0, mass_error NaN/Inf, or mass==0) "
            "from GEVP basis and displayed tables."
        ),
    )
    gevp_eig_rel_cutoff = param.Number(default=1e-2, bounds=(1e-8, 1.0))
    gevp_cond_limit = param.Number(default=1e4, bounds=(1.0, None))
    gevp_shrinkage = param.Number(default=1e-6, bounds=(0.0, 1.0))
    gevp_bootstrap_mode = param.ObjectSelector(
        default="time",
        objects=("time", "walker", "hybrid"),
    )

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


class CouplingDiagnosticsSettings(param.Parameterized):
    """Settings for quick mass-free coupling diagnostics."""

    simulation_range = param.Range(
        default=(0.2, 1.0),
        bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end).",
    )
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(
        default=None,
        bounds=(1e-8, None),
        allow_None=True,
        doc="Color-state length scale ell0. Blank uses estimated value.",
    )
    companion_topology = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion topology used for pair diagnostics.",
    )
    pair_weighting = param.ObjectSelector(
        default="uniform",
        objects=("uniform", "score_abs"),
        doc="Pair weighting scheme for local color fields.",
    )
    color_dims_spec = param.String(
        default="",
        doc="Optional comma-separated color dims (blank uses all available dims).",
    )
    eps = param.Number(
        default=1e-12,
        bounds=(0.0, None),
        doc="Minimum |inner product| for valid pair contributions.",
    )
    enable_kernel_diagnostics = param.Boolean(
        default=True,
        doc=(
            "Enable multiscale kernel diagnostics (string tension, Polyakov, screening, "
            "running coupling, topology)."
        ),
    )
    edge_weight_mode = param.ObjectSelector(
        default="riemannian_kernel_volume",
        objects=(
            "uniform",
            "inverse_distance",
            "inverse_volume",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "kernel",
            "riemannian_kernel",
            "riemannian_kernel_volume",
        ),
    )
    n_scales = param.Integer(default=8, bounds=(2, 32))
    kernel_type = param.ObjectSelector(
        default="gaussian",
        objects=("gaussian", "exponential", "tophat", "shell"),
    )
    kernel_distance_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "floyd-warshall", "tropical"),
    )
    kernel_assume_all_alive = param.Boolean(default=True)
    kernel_scale_frames = param.Integer(default=8, bounds=(1, 64))
    kernel_scale_q_low = param.Number(default=0.05, bounds=(0.0, 0.99))
    kernel_scale_q_high = param.Number(default=0.95, bounds=(0.01, 1.0))
    kernel_max_scale_samples = param.Integer(default=500_000, bounds=(1_000, 5_000_000))
    kernel_min_scale = param.Number(default=1e-6, bounds=(1e-12, None))


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
        raise ValueError(
            f"Invalid spatial_dims_spec: {spec!r}. Expected comma-separated integers."
        ) from exc
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
            "baryon_color_dims_spec must contain exactly 3 dims; " f"received {dims}."
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
            raise ValueError(f"valid_t must have shape [{n_time}], got {tuple(valid_mask.shape)}.")
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
            raise ValueError(f"valid_t must have shape [{n_time}], got {tuple(valid_mask.shape)}.")
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


def _resolve_h_eff(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> tuple[float, str]:
    """Resolve h_eff: manual value or auto sigma_S/2."""
    if str(settings.h_eff_mode) != "auto_sigma_s":
        val = float(settings.h_eff)
        return val, f"manual: {val:.6g}"
    frame_indices = _resolve_baryon_frame_indices(
        history=history,
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
    )
    if len(frame_indices) < 2:
        val = float(settings.h_eff)
        return val, f"auto fallback (too few frames): {val:.6g}"
    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    scores = history.cloning_scores[start_idx - 1 : end_idx - 1]
    alive = history.alive_mask[start_idx - 1 : end_idx - 1]
    alive_scores = scores[alive]
    if alive_scores.numel() < 2:
        val = float(settings.h_eff)
        return val, f"auto fallback (no alive scores): {val:.6g}"
    sigma_s = float(alive_scores.float().std().item())
    if sigma_s < 1e-12:
        val = float(settings.h_eff)
        return val, f"auto fallback (sigma_S~0): {val:.6g}"
    h_eff_auto = sigma_s / 2.0
    return h_eff_auto, f"auto: {h_eff_auto:.6g} (sigma_S={sigma_s:.6g})"


def _compute_anisotropic_baryon_triplet_result(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    *,
    h_eff: float,
    operator_mode: str | None = None,
    channel_name: str = "nucleon",
) -> tuple[ChannelCorrelatorResult, int]:
    """Compute nucleon channel using companion-triplet baryon correlator."""
    baryon_max_lag = int(settings.baryon_max_lag or settings.max_lag)
    baryon_cfg = BaryonTripletCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=baryon_max_lag,
        use_connected=bool(settings.baryon_use_connected),
        h_eff=h_eff,
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.baryon_color_dims_spec, history.d),
        eps=float(settings.baryon_eps),
        operator_mode=(
            str(operator_mode) if operator_mode is not None else str(settings.baryon_operator_mode)
        ),
        flux_exp_alpha=float(settings.baryon_flux_exp_alpha),
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
        channel_name=str(channel_name),
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
    *,
    h_eff: float,
) -> tuple[dict[str, ChannelCorrelatorResult], int]:
    """Compute meson channels using companion-pair meson phases."""
    meson_max_lag = int(settings.meson_max_lag or settings.max_lag)
    meson_cfg = MesonPhaseCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=meson_max_lag,
        use_connected=bool(settings.meson_use_connected),
        h_eff=h_eff,
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.meson_color_dims_spec, history.d),
        pair_selection=str(settings.meson_pair_selection),
        eps=float(settings.meson_eps),
        operator_mode="standard",
    )
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
    valid_frames = 0

    standard_requested = requested_channels & {"scalar", "scalar_raw", "pseudoscalar"}
    if standard_requested:
        meson_out = compute_companion_meson_phase_correlator(history, meson_cfg)
        n_samples = int(meson_out.n_valid_source_pairs)
        pseudoscalar_err: torch.Tensor | None = None
        scalar_err: torch.Tensor | None = None
        scalar_raw_err: torch.Tensor | None = None
        if bool(settings.compute_bootstrap_errors):
            meson_err = _bootstrap_errors_batched_scalar_series(
                torch.stack(
                    [meson_out.operator_pseudoscalar_series, meson_out.operator_scalar_series],
                    dim=0,
                ),
                valid_t=(meson_out.pair_counts_per_frame > 0),
                max_lag=meson_max_lag,
                use_connected=bool(settings.meson_use_connected),
                n_bootstrap=int(settings.n_bootstrap),
            )
            pseudoscalar_err = meson_err[0]
            scalar_err = meson_err[1]
            if "scalar_raw" in requested_channels:
                meson_raw_err = _bootstrap_errors_batched_scalar_series(
                    torch.stack(
                        [meson_out.operator_pseudoscalar_series, meson_out.operator_scalar_series],
                        dim=0,
                    ),
                    valid_t=(meson_out.pair_counts_per_frame > 0),
                    max_lag=meson_max_lag,
                    use_connected=False,
                    n_bootstrap=int(settings.n_bootstrap),
                )
                scalar_raw_err = meson_raw_err[1]
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
        if "scalar_raw" in requested_channels:
            results["scalar_raw"] = _build_result_from_precomputed_correlator(
                channel_name="scalar_raw",
                correlator=meson_out.scalar_raw,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples,
                series=meson_out.operator_scalar_series,
                correlator_err=scalar_raw_err,
            )
        valid_frames = max(valid_frames, int((meson_out.pair_counts_per_frame > 0).sum().item()))

    score_requested = requested_channels & {"scalar_score_directed"}
    if score_requested:
        meson_score_cfg = MesonPhaseCorrelatorConfig(
            warmup_fraction=float(settings.simulation_range[0]),
            end_fraction=float(settings.simulation_range[1]),
            mc_time_index=settings.mc_time_index,
            max_lag=meson_max_lag,
            use_connected=bool(settings.meson_use_connected),
            h_eff=h_eff,
            mass=float(settings.mass),
            ell0=settings.ell0,
            color_dims=_parse_triplet_dims_spec(settings.meson_color_dims_spec, history.d),
            pair_selection=str(settings.meson_pair_selection),
            eps=float(settings.meson_eps),
            operator_mode="score_directed",
        )
        meson_score_out = compute_companion_meson_phase_correlator(history, meson_score_cfg)
        n_samples_score = int(meson_score_out.n_valid_source_pairs)
        scalar_score_err: torch.Tensor | None = None
        if bool(settings.compute_bootstrap_errors):
            meson_err_score = _bootstrap_errors_batched_scalar_series(
                torch.stack(
                    [
                        meson_score_out.operator_pseudoscalar_series,
                        meson_score_out.operator_scalar_series,
                    ],
                    dim=0,
                ),
                valid_t=(meson_score_out.pair_counts_per_frame > 0),
                max_lag=meson_max_lag,
                use_connected=bool(settings.meson_use_connected),
                n_bootstrap=int(settings.n_bootstrap),
            )
            scalar_score_err = meson_err_score[1]
        if "scalar_score_directed" in requested_channels:
            results["scalar_score_directed"] = _build_result_from_precomputed_correlator(
                channel_name="scalar_score_directed",
                correlator=meson_score_out.scalar,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples_score,
                series=meson_score_out.operator_scalar_series,
                correlator_err=scalar_score_err,
            )
        valid_frames = max(
            valid_frames, int((meson_score_out.pair_counts_per_frame > 0).sum().item())
        )

    weighted_requested = requested_channels & {
        "scalar_score_weighted",
        "pseudoscalar_score_weighted",
    }
    if weighted_requested:
        meson_weighted_cfg = MesonPhaseCorrelatorConfig(
            warmup_fraction=float(settings.simulation_range[0]),
            end_fraction=float(settings.simulation_range[1]),
            mc_time_index=settings.mc_time_index,
            max_lag=meson_max_lag,
            use_connected=bool(settings.meson_use_connected),
            h_eff=h_eff,
            mass=float(settings.mass),
            ell0=settings.ell0,
            color_dims=_parse_triplet_dims_spec(settings.meson_color_dims_spec, history.d),
            pair_selection=str(settings.meson_pair_selection),
            eps=float(settings.meson_eps),
            operator_mode="score_weighted",
        )
        meson_weighted_out = compute_companion_meson_phase_correlator(history, meson_weighted_cfg)
        n_samples_weighted = int(meson_weighted_out.n_valid_source_pairs)
        pseudoscalar_weighted_err: torch.Tensor | None = None
        scalar_weighted_err: torch.Tensor | None = None
        if bool(settings.compute_bootstrap_errors):
            meson_err_weighted = _bootstrap_errors_batched_scalar_series(
                torch.stack(
                    [
                        meson_weighted_out.operator_pseudoscalar_series,
                        meson_weighted_out.operator_scalar_series,
                    ],
                    dim=0,
                ),
                valid_t=(meson_weighted_out.pair_counts_per_frame > 0),
                max_lag=meson_max_lag,
                use_connected=bool(settings.meson_use_connected),
                n_bootstrap=int(settings.n_bootstrap),
            )
            pseudoscalar_weighted_err = meson_err_weighted[0]
            scalar_weighted_err = meson_err_weighted[1]
        if "pseudoscalar_score_weighted" in requested_channels:
            results["pseudoscalar_score_weighted"] = _build_result_from_precomputed_correlator(
                channel_name="pseudoscalar_score_weighted",
                correlator=meson_weighted_out.pseudoscalar,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples_weighted,
                series=meson_weighted_out.operator_pseudoscalar_series,
                correlator_err=pseudoscalar_weighted_err,
            )
        if "scalar_score_weighted" in requested_channels:
            results["scalar_score_weighted"] = _build_result_from_precomputed_correlator(
                channel_name="scalar_score_weighted",
                correlator=meson_weighted_out.scalar,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples_weighted,
                series=meson_weighted_out.operator_scalar_series,
                correlator_err=scalar_weighted_err,
            )
        valid_frames = max(
            valid_frames, int((meson_weighted_out.pair_counts_per_frame > 0).sum().item())
        )

    abs2_requested = requested_channels & {"scalar_abs2_vacsub"}
    if abs2_requested:
        meson_abs2_cfg = MesonPhaseCorrelatorConfig(
            warmup_fraction=float(settings.simulation_range[0]),
            end_fraction=float(settings.simulation_range[1]),
            mc_time_index=settings.mc_time_index,
            max_lag=meson_max_lag,
            use_connected=bool(settings.meson_use_connected),
            h_eff=h_eff,
            mass=float(settings.mass),
            ell0=settings.ell0,
            color_dims=_parse_triplet_dims_spec(settings.meson_color_dims_spec, history.d),
            pair_selection=str(settings.meson_pair_selection),
            eps=float(settings.meson_eps),
            operator_mode="abs2_vacsub",
        )
        meson_abs2_out = compute_companion_meson_phase_correlator(history, meson_abs2_cfg)
        n_samples_abs2 = int(meson_abs2_out.n_valid_source_pairs)
        scalar_abs2_err: torch.Tensor | None = None
        if bool(settings.compute_bootstrap_errors):
            meson_err_abs2 = _bootstrap_errors_batched_scalar_series(
                torch.stack(
                    [
                        meson_abs2_out.operator_pseudoscalar_series,
                        meson_abs2_out.operator_scalar_series,
                    ],
                    dim=0,
                ),
                valid_t=(meson_abs2_out.pair_counts_per_frame > 0),
                max_lag=meson_max_lag,
                use_connected=bool(settings.meson_use_connected),
                n_bootstrap=int(settings.n_bootstrap),
            )
            scalar_abs2_err = meson_err_abs2[1]
        if "scalar_abs2_vacsub" in requested_channels:
            results["scalar_abs2_vacsub"] = _build_result_from_precomputed_correlator(
                channel_name="scalar_abs2_vacsub",
                correlator=meson_abs2_out.scalar,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples_abs2,
                series=meson_abs2_out.operator_scalar_series,
                correlator_err=scalar_abs2_err,
            )
        valid_frames = max(
            valid_frames, int((meson_abs2_out.pair_counts_per_frame > 0).sum().item())
        )

    return results, valid_frames


def _compute_anisotropic_vector_meson_results(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    requested_channels: set[str],
    *,
    h_eff: float,
) -> tuple[dict[str, ChannelCorrelatorResult], int]:
    """Compute vector/axial-vector channels using companion-pair vector mesons."""
    vector_max_lag = int(settings.vector_meson_max_lag or settings.max_lag)
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

    variant_specs: list[tuple[str, str, str, str]] = [
        ("standard", "full", "vector", "axial_vector"),
        ("score_directed", "full", "vector_score_directed", "axial_vector_score_directed"),
        ("score_gradient", "full", "vector_score_gradient", "axial_vector_score_gradient"),
        (
            "score_directed",
            "longitudinal",
            "vector_score_directed_longitudinal",
            "axial_vector_score_directed_longitudinal",
        ),
        (
            "score_directed",
            "transverse",
            "vector_score_directed_transverse",
            "axial_vector_score_directed_transverse",
        ),
    ]

    results: dict[str, ChannelCorrelatorResult] = {}
    valid_frames = 0
    for operator_mode, projection_mode, vector_name, axial_name in variant_specs:
        if vector_name not in requested_channels and axial_name not in requested_channels:
            continue
        vector_cfg = VectorMesonCorrelatorConfig(
            warmup_fraction=float(settings.simulation_range[0]),
            end_fraction=float(settings.simulation_range[1]),
            mc_time_index=settings.mc_time_index,
            max_lag=vector_max_lag,
            use_connected=bool(settings.vector_meson_use_connected),
            h_eff=h_eff,
            mass=float(settings.mass),
            ell0=settings.ell0,
            color_dims=_parse_triplet_dims_spec(settings.vector_meson_color_dims_spec, history.d),
            position_dims=_parse_triplet_dims_spec(
                settings.vector_meson_position_dims_spec, history.d
            ),
            pair_selection=str(settings.vector_meson_pair_selection),
            eps=float(settings.vector_meson_eps),
            use_unit_displacement=bool(settings.vector_meson_use_unit_displacement),
            operator_mode=operator_mode,
            projection_mode=projection_mode,
        )
        vector_out = compute_companion_vector_meson_correlator(history, vector_cfg)
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
        if vector_name in requested_channels:
            results[vector_name] = _build_result_from_precomputed_correlator(
                channel_name=vector_name,
                correlator=vector_out.vector,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples,
                series=vector_out.vector,
                correlator_err=vector_err,
            )
        if axial_name in requested_channels:
            results[axial_name] = _build_result_from_precomputed_correlator(
                channel_name=axial_name,
                correlator=vector_out.axial_vector,
                dt=dt,
                config=fit_cfg,
                n_samples=n_samples,
                series=vector_out.axial_vector,
                correlator_err=axial_err,
            )
        valid_frames = max(valid_frames, int((vector_out.pair_counts_per_frame > 0).sum().item()))
    return results, valid_frames


def _compute_anisotropic_glueball_color_result(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    *,
    h_eff: float,
    operator_mode: str | None = None,
    channel_name: str = "glueball",
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
    resolved_glueball_mode = (
        str(operator_mode) if operator_mode is not None else str(settings.glueball_operator_mode)
    )
    glueball_cfg = GlueballColorCorrelatorConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
        max_lag=glueball_max_lag,
        use_connected=bool(settings.glueball_use_connected),
        h_eff=h_eff,
        mass=float(settings.mass),
        ell0=settings.ell0,
        color_dims=_parse_triplet_dims_spec(settings.glueball_color_dims_spec, history.d),
        eps=float(settings.glueball_eps),
        operator_mode=(None if resolved_glueball_mode == "auto" else resolved_glueball_mode),
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
        channel_name=str(channel_name),
        correlator=glueball_out.correlator,
        dt=dt,
        config=fit_cfg,
        n_samples=int(glueball_out.n_valid_source_triplets),
        series=glueball_out.operator_glueball_series,
        correlator_err=correlator_err,
    )
    results: dict[str, ChannelCorrelatorResult] = {str(channel_name): result}

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
        "monte_carlo": ("mc", 3),  # MC time, dim doesn't matter
        "x": ("euclidean", 0),  # X spatial dimension
        "y": ("euclidean", 1),  # Y spatial dimension
        "z": ("euclidean", 2),  # Z spatial dimension
        "t": ("euclidean", 3),  # Euclidean time (default)
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


def _compute_anisotropic_edge_bundle(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> AnisotropicEdgeChannelOutput:
    """Compute non-companion anisotropic direct-edge correlators from recorded geometry."""
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


def _compute_anisotropic_edge_tensor_only_results(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute only non-companion tensor channels for tensor calibration tab."""
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
    output = compute_anisotropic_edge_channels(
        history,
        config=config,
        channels=["tensor", "tensor_traceless"],
    )
    return dict(output.channel_results)


def _compute_companion_strong_force_bundle(
    history: RunHistory,
    settings: AnisotropicEdgeSettings,
    requested_channels: list[str] | None = None,
) -> tuple[AnisotropicEdgeChannelOutput, float, str]:
    """Compute companion-only strong-force correlators."""
    resolved_h_eff, h_eff_desc = _resolve_h_eff(history, settings)
    if requested_channels is None:
        channels = [
            channel
            for family in DEFAULT_COMPANION_CHANNEL_VARIANT_SELECTION.values()
            for channel in family
        ]
    else:
        channels = list(requested_channels)
    requested = {str(c).strip() for c in channels if str(c).strip()}
    requested_meson_channels = requested & {
        "scalar",
        "scalar_raw",
        "scalar_abs2_vacsub",
        "pseudoscalar",
        "scalar_score_directed",
        "scalar_score_weighted",
        "pseudoscalar_score_weighted",
    }
    requested_nucleon_channels = requested & {
        "nucleon",
        "nucleon_flux_action",
        "nucleon_flux_sin2",
        "nucleon_flux_exp",
        "nucleon_score_signed",
        "nucleon_score_abs",
    }
    requested_glueball_channels = requested & {
        "glueball",
        "glueball_phase_action",
        "glueball_phase_sin2",
    }
    requested_vector_channels = requested & {
        "vector",
        "axial_vector",
        "vector_score_directed",
        "vector_score_directed_longitudinal",
        "axial_vector_score_directed_longitudinal",
        "vector_score_directed_transverse",
        "axial_vector_score_directed_transverse",
        "vector_score_gradient",
        "axial_vector_score_gradient",
    }
    results: dict[str, ChannelCorrelatorResult] = {}
    valid_frame_counts: list[int] = []

    if bool(settings.use_companion_baryon_triplet) and requested_nucleon_channels:
        baryon_mode_primary = str(settings.baryon_operator_mode)
        baryon_variants: list[tuple[str, str]] = [
            (baryon_mode_primary, "nucleon"),
            ("flux_action", "nucleon_flux_action"),
            ("flux_sin2", "nucleon_flux_sin2"),
            ("flux_exp", "nucleon_flux_exp"),
            ("score_signed", "nucleon_score_signed"),
            ("score_abs", "nucleon_score_abs"),
        ]
        for variant_mode, variant_channel in baryon_variants:
            if variant_channel != "nucleon" and variant_channel not in requested_nucleon_channels:
                continue
            variant_result, variant_valid_frames = _compute_anisotropic_baryon_triplet_result(
                history,
                settings,
                h_eff=resolved_h_eff,
                operator_mode=variant_mode,
                channel_name=variant_channel,
            )
            results[variant_channel] = variant_result
            valid_frame_counts.append(int(variant_valid_frames))

    meson_targets = requested_meson_channels & {
        "scalar",
        "scalar_raw",
        "scalar_abs2_vacsub",
        "pseudoscalar",
        "scalar_score_directed",
        "scalar_score_weighted",
        "pseudoscalar_score_weighted",
    }
    if bool(settings.use_companion_meson_phase) and meson_targets:
        meson_results, meson_valid_frames = _compute_anisotropic_meson_phase_results(
            history,
            settings,
            requested_channels=meson_targets,
            h_eff=resolved_h_eff,
        )
        results.update(meson_results)
        valid_frame_counts.append(int(meson_valid_frames))

    vector_targets = requested_vector_channels & {
        "vector",
        "axial_vector",
        "vector_score_directed",
        "vector_score_directed_longitudinal",
        "axial_vector_score_directed_longitudinal",
        "vector_score_directed_transverse",
        "axial_vector_score_directed_transverse",
        "vector_score_gradient",
        "axial_vector_score_gradient",
    }
    if bool(settings.use_companion_vector_meson) and vector_targets:
        vector_results, vector_valid_frames = _compute_anisotropic_vector_meson_results(
            history,
            settings,
            requested_channels=vector_targets,
            h_eff=resolved_h_eff,
        )
        results.update(vector_results)
        valid_frame_counts.append(int(vector_valid_frames))

    if bool(settings.use_companion_glueball_color) and requested_glueball_channels:
        glueball_mode_primary = str(settings.glueball_operator_mode)
        glueball_variants: list[tuple[str, str, bool]] = [
            (glueball_mode_primary, "glueball", bool(settings.glueball_use_momentum_projection)),
            ("phase_action", "glueball_phase_action", False),
            ("phase_sin2", "glueball_phase_sin2", False),
        ]
        for variant_mode, variant_channel, allow_momentum in glueball_variants:
            if variant_channel != "glueball":
                if variant_channel not in requested_glueball_channels:
                    continue
            variant_results, variant_valid_frames = _compute_anisotropic_glueball_color_result(
                history,
                settings,
                h_eff=resolved_h_eff,
                operator_mode=variant_mode,
                channel_name=variant_channel,
                force_momentum_projection=allow_momentum,
            )
            variant_result = variant_results.get(variant_channel)
            if variant_result is not None:
                results[variant_channel] = variant_result
            valid_frame_counts.append(int(variant_valid_frames))

    if bool(settings.use_companion_tensor_momentum) and "tensor" in requested:
        tensor_results, tensor_meta = _compute_tensor_momentum_for_anisotropic_edge(
            history, settings
        )
        p0_result = tensor_results.get("tensor_momentum_p0")
        if p0_result is not None:
            results["tensor"] = p0_result
        valid_frame_counts.append(int(tensor_meta.get("n_valid_frames", 0)))

    frame_indices = _resolve_baryon_frame_indices(
        history=history,
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        mc_time_index=settings.mc_time_index,
    )
    avg_alive = 0.0
    if frame_indices:
        frame_ids = torch.as_tensor(
            frame_indices,
            dtype=torch.long,
            device=history.alive_mask.device,
        )
        alive = history.alive_mask.index_select(0, frame_ids - 1)
        if alive.numel() > 0:
            avg_alive = float(alive.float().sum(dim=1).mean().item())

    return (
        AnisotropicEdgeChannelOutput(
            channel_results=results,
            component_labels=["companion"],
            frame_indices=list(frame_indices),
            n_valid_frames=max(valid_frame_counts) if valid_frame_counts else 0,
            avg_alive_walkers=avg_alive,
            avg_edges=float("nan"),
        ),
        resolved_h_eff,
        h_eff_desc,
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
    glueball_series = operator_series.operators.get(
        "glueball", torch.zeros(0, dtype=torch.float32)
    )
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
        return {}, {"component_labels": (), "momentum_axis": int(settings.tensor_momentum_axis)}

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
        position_dims=_parse_triplet_dims_spec(
            settings.tensor_momentum_position_dims_spec, history.d
        ),
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
        volume_alive = volume_full.to(device=positions_alive.device, dtype=positions_alive.dtype)[
            alive_idx
        ].clamp(min=0.0)
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
    traceless = dx_valid.unsqueeze(-1) * dx_valid.unsqueeze(-2) - eye.unsqueeze(0) * (
        r2[:, None, None] / float(dim)
    )

    weighted_tensor = (w_valid * c_valid)[:, None, None] * traceless
    tensor_sum = torch.zeros(
        (n_alive, dim * dim), device=positions_alive.device, dtype=torch.float32
    )
    tensor_sum.index_add_(0, src_valid, weighted_tensor.reshape(-1, dim * dim))
    weight_sum = torch.zeros(n_alive, device=positions_alive.device, dtype=torch.float32)
    weight_sum.index_add_(0, src_valid, w_valid)

    local_tensor = torch.zeros_like(tensor_sum)
    has_weight = weight_sum > 0
    if torch.any(has_weight):
        local_tensor[has_weight] = tensor_sum[has_weight] / weight_sum[has_weight, None].clamp(
            min=1e-12
        )
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
    bundle = compute_radial_channels(
        history, config=radial_config, channels=["glueball", "tensor"]
    )
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
        spatial_cut_types = ()

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
            s_dist_geom = _sum_cross_boundary_weights(masks, companions_dist[info_idx], w_dist)
            s_fit_geom = _sum_cross_boundary_weights(masks, companions_fit[info_idx], w_fit)
            s_total_geom = s_dist_geom + s_fit_geom
            area_cst_geom = _count_crossing_lineages_weighted(
                masks=masks,
                lineage_ids=labels,
                volume_weights=vol_frame,
                use_weighted_area=bool(settings.geometry_correct_area and use_geom),
            )
            region_size = masks.sum(axis=1, dtype=np.int64)

            for local_idx, cut_value in enumerate(cut_values.tolist()):
                rows.append({
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
                })

            frame_rows.append({
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
            })

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
        regression_rows.append({
            "metric": metric_key,
            "metric_label": metric_label,
            "area_key": area_key,
            "cut_type": "all",
            "partition_family": "all",
            "n_points": int(x_all.size),
            "slope_alpha": slope,
            "intercept": intercept,
            "r2": r2,
        })

        for cut_type in sorted(points_df["cut_type"].unique()):
            subset = points_df[points_df["cut_type"] == cut_type]
            x = subset[area_key].to_numpy(dtype=float)
            y = subset[metric_key].to_numpy(dtype=float)
            slope, intercept, r2 = _fit_linear_relation(x, y)
            families = subset["partition_family"].unique().tolist()
            family = str(families[0]) if len(families) == 1 else "mixed"
            regression_rows.append({
                "metric": metric_key,
                "metric_label": metric_label,
                "area_key": area_key,
                "cut_type": str(cut_type),
                "partition_family": family,
                "n_points": int(x.size),
                "slope_alpha": slope,
                "intercept": intercept,
                "r2": r2,
            })

        for family in sorted(points_df["partition_family"].unique()):
            subset = points_df[points_df["partition_family"] == family]
            x = subset[area_key].to_numpy(dtype=float)
            y = subset[metric_key].to_numpy(dtype=float)
            slope, intercept, r2 = _fit_linear_relation(x, y)
            regression_rows.append({
                "metric": metric_key,
                "metric_label": metric_label,
                "area_key": area_key,
                "cut_type": f"family:{family}",
                "partition_family": str(family),
                "n_points": int(x.size),
                "slope_alpha": slope,
                "intercept": intercept,
                "r2": r2,
            })

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
        return (
            "## Fractal Set Summary\n_No valid measurements. Load a RunHistory and run compute._"
        )

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
                lines.append(f"  random baseline: alpha={b_alpha:.6f}, R2={b_r2:.4f}")

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
            rows.append({
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
            })

    if not rows:
        return pd.DataFrame()
    comparison = pd.DataFrame(rows)
    return comparison.sort_values(["metric", "delta_r2_vs_random"], ascending=[True, False])


def _compute_coupling_constants(
    history: RunHistory | None,
    h_eff: float,
    frame_indices: list[int] | None = None,
    lambda_alg: float | None = None,
    pairwise_distance_by_frame: dict[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    return compute_electroweak_coupling_constants(
        history,
        h_eff=h_eff,
        frame_indices=frame_indices,
        lambda_alg=lambda_alg,
        pairwise_distance_by_frame=pairwise_distance_by_frame,
    )


def _resolve_electroweak_geodesic_matrices(
    history: RunHistory | None,
    frame_indices: list[int] | None,
    state: dict[str, Any],
    *,
    method: str,
    edge_weight_mode: str,
    assume_all_alive: bool,
) -> dict[int, torch.Tensor] | None:
    if not isinstance(state, dict):
        return None

    requested_frames: list[int] = []
    for frame in frame_indices or []:
        try:
            frame_idx = int(frame)
        except (TypeError, ValueError):
            continue
        if frame_idx not in requested_frames:
            requested_frames.append(frame_idx)
    if not requested_frames:
        return None

    cached = state.get("_multiscale_geodesic_distance_by_frame")
    merged: dict[int, torch.Tensor] = {}
    if isinstance(cached, dict):
        for raw_key, value in cached.items():
            try:
                frame = int(raw_key)
            except (TypeError, ValueError):
                continue
            if torch.is_tensor(value):
                merged[frame] = value.detach().to(dtype=torch.float32, device="cpu")

    missing_frames = [frame for frame in requested_frames if frame not in merged]
    if not missing_frames or history is None:
        return merged or None

    try:
        frame_ids, distance_batch = compute_pairwise_distance_matrices_from_history(
            history,
            method=method,
            frame_indices=missing_frames,
            batch_size=1,
            edge_weight_mode=edge_weight_mode,
            assume_all_alive=bool(assume_all_alive),
            device=None,
            dtype=torch.float32,
        )
    except Exception:
        return merged or None

    for local_idx, frame_id in enumerate(frame_ids):
        if local_idx >= int(distance_batch.shape[0]):
            break
        matrix = distance_batch[local_idx]
        if torch.is_tensor(matrix):
            merged[int(frame_id)] = matrix.detach().to(dtype=torch.float32, device="cpu")

    if merged:
        state["_multiscale_geodesic_distance_by_frame"] = merged
        return merged
    return None


def _build_coupling_rows(
    couplings: dict[str, float],
    proxies: dict[str, float] | None = None,
    include_strong: bool = False,
    refs: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    proxies = proxies or {}
    refs = refs or {}
    rows = [
        {
            "name": "ε_distance emergent",
            "value": couplings.get("eps_distance_emergent"),
            "note": "RMS d_alg to diversity companions",
        },
        {
            "name": "ε_clone emergent",
            "value": couplings.get("eps_clone_emergent"),
            "note": "RMS d_alg to cloning companions",
        },
        {
            "name": "ε_fitness_gap emergent",
            "value": couplings.get("eps_fitness_gap_emergent"),
            "note": "RMS |S_i| to random cloning companion",
        },
        {
            "name": "g1_est emergent",
            "value": couplings.get("g1_est_emergent"),
            "note": "from ε_distance emergent",
        },
        {
            "name": "g2_est emergent",
            "value": couplings.get("g2_est_emergent"),
            "note": "from ε_clone emergent",
        },
        {
            "name": "ε_geodesic emergent",
            "value": couplings.get("eps_geodesic_emergent"),
            "note": "RMS geodesic pairwise distance across walkers",
        },
        {
            "name": "g2_est emergent fitness-gap",
            "value": couplings.get("g2_est_emergent_fitness_gap"),
            "note": "from ε_fitness_gap emergent",
        },
        {
            "name": "sin²θ_W emergent",
            "value": couplings.get("sin2_theta_w_emergent"),
            "note": "ε_c²/(ε_c²+ε_d²), ε_c=fitness_gap, ε_d=geodesic",
        },
        {
            "name": "tanθ_W emergent",
            "value": couplings.get("tan_theta_w_emergent"),
            "note": "ε_c/ε_d, ε_c=fitness_gap, ε_d=geodesic",
        },
    ]
    if "g1_proxy" in proxies:
        rows.append({"name": "g1_proxy", "value": proxies.get("g1_proxy"), "note": "phase std"})
    if "g2_proxy" in proxies:
        rows.append({"name": "g2_proxy", "value": proxies.get("g2_proxy"), "note": "phase std"})
    if "sin2_theta_w_proxy" in proxies:
        rows.append({
            "name": "sin2θw proxy",
            "value": proxies.get("sin2_theta_w_proxy"),
            "note": "from phase stds",
        })
    if "tan_theta_w_proxy" in proxies:
        rows.append({
            "name": "tanθw proxy",
            "value": proxies.get("tan_theta_w_proxy"),
            "note": "from phase stds",
        })
    if include_strong:
        rows.append({
            "name": "g3_est",
            "value": couplings.get("g3_est"),
            "note": "from ν, h_eff, <K_visc^2>",
        })
        rows.append({
            "name": "<K_visc^2> proxy",
            "value": couplings.get("kvisc_sq_proxy"),
            "note": "mean ||F_visc||^2 / ν^2",
        })

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


def _compute_masked_mean_p95(
    values: np.ndarray, alive_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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

    curve = hv.Curve(frame, "step", "mean").opts(color=color, line_width=2, tools=["hover"])
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

    curve = hv.Curve(frame, "step", "mean").opts(color=color, line_width=2, tools=["hover"])
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
            .opts(color="#e45756", line_width=2, tools=["hover"])
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
            .opts(color="#4c78a8", line_width=2, tools=["hover"])
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
    v_pre = _to_numpy(history.v_before_clone).astype(float, copy=False)
    companions_clone = _to_numpy(history.companions_clone).astype(np.int64, copy=False)
    companions_random = _to_numpy(history.companions_distance).astype(np.int64, copy=False)

    n_steps = min(
        alive.shape[0],
        will_clone.shape[0],
        fitness.shape[0],
        rewards.shape[0],
        x_pre.shape[0],
        v_pre.shape[0],
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
    v_pre = v_pre[:n_steps]
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
            tools=["hover"],
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

    # Average walker speed (||v||) over time with p95 error bars.
    speed = np.linalg.norm(v_pre, axis=-1)  # [n_steps, N]
    speed_mean, speed_p95 = _compute_masked_mean_p95(speed, alive)
    velocity_plot = _build_timeseries_mean_p95_plot(
        step=step,
        mean=speed_mean,
        p95=speed_p95,
        title="Mean Walker Speed Over Time (p95 error bars)",
        ylabel="Speed (||v||)",
        color="#e45756",
    )

    # Geodesic edge distances and riemannian_kernel_volume weights over time.
    geodesic_plot = _algorithm_placeholder_plot("No geodesic edge distance data available.")
    rkv_plot = _algorithm_placeholder_plot("No riemannian_kernel_volume data available.")
    geo_list = getattr(history, "geodesic_edge_distances", None)
    ew_list = getattr(history, "edge_weights", None)
    recorded_steps = np.asarray(history.recorded_steps, dtype=float)
    if geo_list is not None and len(geo_list) > 0:
        geo_means = np.full(len(geo_list), np.nan)
        geo_stds = np.full(len(geo_list), np.nan)
        for t, geo_t in enumerate(geo_list):
            vals = _to_numpy(geo_t).astype(float).ravel()
            if vals.size > 0:
                geo_means[t] = float(np.mean(vals))
                geo_stds[t] = float(np.std(vals))
        geo_step = (
            recorded_steps[: len(geo_list)]
            if recorded_steps.size >= len(geo_list)
            else np.arange(len(geo_list), dtype=float)
        )
        geodesic_plot = _build_timeseries_mean_error_plot(
            step=geo_step,
            mean=geo_means,
            error=geo_stds,
            title="Mean Geodesic Edge Distance Over Time (mean ± 1σ)",
            ylabel="Geodesic distance",
            color="#9d755d",
        )
    if ew_list is not None and len(ew_list) > 0:
        rkv_means = np.full(len(ew_list), np.nan)
        rkv_stds = np.full(len(ew_list), np.nan)
        for t, ew_t in enumerate(ew_list):
            if isinstance(ew_t, dict) and "riemannian_kernel_volume" in ew_t:
                vals = _to_numpy(ew_t["riemannian_kernel_volume"]).astype(float).ravel()
                if vals.size > 0:
                    rkv_means[t] = float(np.mean(vals))
                    rkv_stds[t] = float(np.std(vals))
        rkv_step = (
            recorded_steps[: len(ew_list)]
            if recorded_steps.size >= len(ew_list)
            else np.arange(len(ew_list), dtype=float)
        )
        rkv_plot = _build_timeseries_mean_error_plot(
            step=rkv_step,
            mean=rkv_means,
            error=rkv_stds,
            title="Riemannian Kernel Volume Weights Over Time (mean ± 1σ)",
            ylabel="Weight",
            color="#b279a2",
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
        "velocity_plot": velocity_plot,
        "geodesic_plot": geodesic_plot,
        "rkv_plot": rkv_plot,
        "lyapunov_plot": lyapunov_plot,
        "n_transition_steps": int(n_steps),
        "n_lyapunov_steps": int(len(lyapunov["time"])),
    }


def _build_coupling_diagnostics_summary_table(summary: dict[str, float]) -> pd.DataFrame:
    """Build a compact summary table for coupling diagnostics."""
    metrics = [
        ("n_frames", summary.get("n_frames")),
        ("phase_drift", summary.get("phase_drift")),
        ("phase_step_std", summary.get("phase_step_std")),
        ("phase_drift_sigma", summary.get("phase_drift_sigma")),
        ("r_circ_mean", summary.get("r_circ_mean")),
        ("re_im_asymmetry_mean", summary.get("re_im_asymmetry_mean")),
        ("local_phase_coherence_mean", summary.get("local_phase_coherence_mean")),
        ("scalar_mean", summary.get("scalar_mean")),
        ("pseudoscalar_mean", summary.get("pseudoscalar_mean")),
        ("field_magnitude_mean", summary.get("field_magnitude_mean")),
        ("valid_pairs_mean", summary.get("valid_pairs_mean")),
        ("valid_walkers_mean", summary.get("valid_walkers_mean")),
        ("string_tension_sigma", summary.get("string_tension_sigma")),
        ("polyakov_abs", summary.get("polyakov_abs")),
        ("screening_length_xi", summary.get("screening_length_xi")),
        ("running_coupling_slope", summary.get("running_coupling_slope")),
        ("topological_flux_std", summary.get("topological_flux_std")),
        ("topological_charge_q", summary.get("topological_charge_q")),
        ("regime_score", summary.get("regime_score")),
        ("kernel_diagnostics_available", summary.get("kernel_diagnostics_available")),
    ]
    rows = [{"metric": name, "value": value} for name, value in metrics]
    frame = pd.DataFrame(rows)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _build_coupling_diagnostics_frame_table(
    step_axis: np.ndarray,
    output: Any,
) -> pd.DataFrame:
    """Build per-frame diagnostics table."""
    if len(step_axis) == 0:
        return pd.DataFrame()

    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(float, copy=False)

    table = pd.DataFrame({
        "step": step_axis,
        "phase_mean": _tensor_to_numpy(output.phase_mean),
        "phase_mean_unwrapped": _tensor_to_numpy(output.phase_mean_unwrapped),
        "r_circ": _tensor_to_numpy(output.phase_concentration),
        "re_im_asymmetry": _tensor_to_numpy(output.re_im_asymmetry),
        "local_phase_coherence": _tensor_to_numpy(output.local_phase_coherence),
        "scalar_mean": _tensor_to_numpy(output.scalar_mean),
        "pseudoscalar_mean": _tensor_to_numpy(output.pseudoscalar_mean),
        "field_magnitude_mean": _tensor_to_numpy(output.field_magnitude_mean),
        "valid_pairs": output.valid_pair_counts.detach().cpu().numpy().astype(int, copy=False),
        "valid_walkers": output.valid_walker_counts.detach().cpu().numpy().astype(int, copy=False),
    })
    return table.replace([np.inf, -np.inf], np.nan)


def _build_coupling_diagnostics_scale_table(output: Any) -> pd.DataFrame:
    """Build one-row-per-scale diagnostics table."""
    if getattr(output, "scales", None) is None or int(output.scales.numel()) == 0:
        return pd.DataFrame()

    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(float, copy=False)

    table = pd.DataFrame({
        "scale": _to_numpy(output.scales),
        "coherence": _to_numpy(output.coherence_by_scale),
        "phase_spread": _to_numpy(output.phase_spread_by_scale),
        "screening_connected": _to_numpy(output.screening_connected_by_scale),
    })
    return table.replace([np.inf, -np.inf], np.nan)


def _build_coupling_diagnostics_overlay(
    *,
    step_axis: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    ylabel: str,
) -> hv.Overlay | hv.Text:
    """Build an overlay plot from multiple diagnostics series."""
    overlays: list[Any] = []
    for label, values, color in series:
        frame = pd.DataFrame({"step": step_axis, "value": values}).replace(
            [np.inf, -np.inf], np.nan
        )
        frame = frame.dropna()
        if frame.empty:
            continue
        overlays.append(
            hv.Curve(frame, "step", "value").relabel(label).opts(color=color, line_width=2)
        )
    if not overlays:
        return _algorithm_placeholder_plot("No diagnostics data available")
    plot = overlays[0]
    for overlay in overlays[1:]:
        plot = plot * overlay
    return plot.opts(
        title=title,
        xlabel="Recorded step",
        ylabel=ylabel,
        width=960,
        height=320,
        legend_position="top_left",
        show_grid=True,
    )


def _build_coupling_diagnostics_scale_overlay(
    *,
    scale_axis: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    ylabel: str,
) -> hv.Overlay | hv.Text:
    """Build an overlay plot on the scale axis."""
    overlays: list[Any] = []
    for label, values, color in series:
        frame = pd.DataFrame({"scale": scale_axis, "value": values}).replace(
            [np.inf, -np.inf], np.nan
        )
        frame = frame.dropna()
        if frame.empty:
            continue
        overlays.append(
            hv.Curve(frame, "scale", "value").relabel(label).opts(color=color, line_width=2)
        )
    if not overlays:
        return _algorithm_placeholder_plot("No kernel-scale diagnostics available")
    plot = overlays[0]
    for overlay in overlays[1:]:
        plot = plot * overlay
    return plot.opts(
        title=title,
        xlabel="Scale",
        ylabel=ylabel,
        width=960,
        height=320,
        legend_position="top_left",
        show_grid=True,
    )


def _build_coupling_diagnostics_plots(
    step_axis: np.ndarray,
    output: Any,
) -> dict[str, hv.Overlay | hv.Text]:
    """Build coupling diagnostics plot bundle."""
    if len(step_axis) == 0:
        placeholder = _algorithm_placeholder_plot("No diagnostics data available")
        return {
            "phase": placeholder,
            "regime": placeholder,
            "fields": placeholder,
            "coverage": placeholder,
        }

    phase = output.phase_mean.detach().cpu().numpy().astype(float, copy=False)
    phase_unwrapped = output.phase_mean_unwrapped.detach().cpu().numpy().astype(float, copy=False)
    r_circ = output.phase_concentration.detach().cpu().numpy().astype(float, copy=False)
    asym = output.re_im_asymmetry.detach().cpu().numpy().astype(float, copy=False)
    coherence = output.local_phase_coherence.detach().cpu().numpy().astype(float, copy=False)
    scalar = output.scalar_mean.detach().cpu().numpy().astype(float, copy=False)
    pseudoscalar = output.pseudoscalar_mean.detach().cpu().numpy().astype(float, copy=False)
    magnitude = output.field_magnitude_mean.detach().cpu().numpy().astype(float, copy=False)
    valid_pairs = output.valid_pair_counts.detach().cpu().numpy().astype(float, copy=False)
    valid_walkers = output.valid_walker_counts.detach().cpu().numpy().astype(float, copy=False)

    return {
        "phase": _build_coupling_diagnostics_overlay(
            step_axis=step_axis,
            series=[
                ("phase_mean", phase, "#4c78a8"),
                ("phase_mean_unwrapped", phase_unwrapped, "#f58518"),
            ],
            title="Global Phase Trend",
            ylabel="phase [rad]",
        ),
        "regime": _build_coupling_diagnostics_overlay(
            step_axis=step_axis,
            series=[
                ("R_circ", r_circ, "#54a24b"),
                ("Re/Im asymmetry", asym, "#e45756"),
                ("local coherence", coherence, "#72b7b2"),
            ],
            title="Coupling Regime Diagnostics",
            ylabel="dimensionless",
        ),
        "fields": _build_coupling_diagnostics_overlay(
            step_axis=step_axis,
            series=[
                ("scalar_mean", scalar, "#9d755d"),
                ("pseudoscalar_mean", pseudoscalar, "#b279a2"),
                ("field_magnitude_mean", magnitude, "#4c78a8"),
            ],
            title="Local Color Field Means",
            ylabel="operator value",
        ),
        "coverage": _build_coupling_diagnostics_overlay(
            step_axis=step_axis,
            series=[
                ("valid_pairs", valid_pairs, "#f58518"),
                ("valid_walkers", valid_walkers, "#54a24b"),
            ],
            title="Diagnostics Coverage",
            ylabel="count",
        ),
    }


def _build_coupling_diagnostics_kernel_plots(output: Any) -> dict[str, hv.Overlay | hv.Text]:
    """Build kernel-scale diagnostics plots."""
    if getattr(output, "scales", None) is None or int(output.scales.numel()) == 0:
        placeholder = _algorithm_placeholder_plot("No kernel-scale diagnostics available")
        return {"scale": placeholder, "running": placeholder}

    scales = output.scales.detach().cpu().numpy().astype(float, copy=False)
    coherence = output.coherence_by_scale.detach().cpu().numpy().astype(float, copy=False)
    phase_spread = output.phase_spread_by_scale.detach().cpu().numpy().astype(float, copy=False)
    screening = (
        output.screening_connected_by_scale.detach().cpu().numpy().astype(float, copy=False)
    )
    creutz_mid = output.creutz_mid_scales.detach().cpu().numpy().astype(float, copy=False)
    creutz = output.creutz_ratio_by_mid_scale.detach().cpu().numpy().astype(float, copy=False)
    running_mid = output.running_mid_scales.detach().cpu().numpy().astype(float, copy=False)
    running_g2 = output.running_g2_by_mid_scale.detach().cpu().numpy().astype(float, copy=False)

    scale_plot = _build_coupling_diagnostics_scale_overlay(
        scale_axis=scales,
        series=[
            ("coherence", coherence, "#4c78a8"),
            ("phase_spread", phase_spread, "#f58518"),
            ("screening_connected", screening, "#54a24b"),
        ],
        title="Kernel-Scale Diagnostics",
        ylabel="dimensionless",
    )

    running_curves: list[Any] = []
    running_frame = (
        pd.DataFrame({"scale": running_mid, "value": running_g2})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not running_frame.empty:
        running_curves.append(
            hv.Curve(running_frame, "scale", "value")
            .relabel("running_g2")
            .opts(color="#e45756", line_width=2)
        )
    creutz_frame = (
        pd.DataFrame({"scale": creutz_mid, "value": creutz})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not creutz_frame.empty:
        running_curves.append(
            hv.Curve(creutz_frame, "scale", "value")
            .relabel("creutz")
            .opts(color="#72b7b2", line_width=2)
        )
    if running_curves:
        running_plot = running_curves[0]
        for curve in running_curves[1:]:
            running_plot = running_plot * curve
        running_plot = running_plot.opts(
            title="Running Coupling / Creutz Proxies",
            xlabel="Scale",
            ylabel="value",
            width=960,
            height=320,
            legend_position="top_left",
            show_grid=True,
        )
    else:
        running_plot = _algorithm_placeholder_plot("No running/Creutz diagnostics available")

    return {"scale": scale_plot, "running": running_plot}


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
    all_refs: dict[str, dict[str, float]] = dict(ref_groups)

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
        masses,
        [("baryon", BARYON_REFS), ("meson", MESON_REFS)],
        r2s,
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
    "u1_phase": 0.000511,  # electron
    "u1_dressed": 0.105658,  # muon
    "su2_phase": 80.379,  # W boson
    "su2_doublet": 91.1876,  # Z boson
    "ew_mixed": 1.77686,  # tau
    "fitness_phase": 91.1876,  # Z boson (massive mode of fitness phase)
    "clone_indicator": 80.379,  # W boson (cloning decay rate)
}

ELECTROWEAK_COUPLING_NAMES = (
    "ε_distance emergent",
    "ε_clone emergent",
    "ε_fitness_gap emergent",
    "ε_geodesic emergent",
    "g1_est emergent",
    "g2_est emergent",
    "g2_est emergent fitness-gap",
    "sin²θ_W emergent",
    "tanθ_W emergent",
    "g1_proxy",
    "g2_proxy",
    "sin2θw proxy",
    "tanθw proxy",
)

ELECTROWEAK_COUPLING_REFERENCE_COLUMNS = ("observed_mZ", "observed_GUT")

DEFAULT_ELECTROWEAK_COUPLING_REFS = {
    "g1_est emergent": {"observed_mZ": 0.357468, "observed_GUT": 0.560499},
    "g2_est emergent": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "g2_est emergent fitness-gap": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "sin²θ_W emergent": {"observed_mZ": 0.23129, "observed_GUT": 0.375},
    "tanθ_W emergent": {"observed_mZ": 0.548526, "observed_GUT": 0.774597},
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

    spread = float(np.std(masses[valid], ddof=1)) if int(np.count_nonzero(valid)) > 1 else 0.0

    base_idx = int(np.where(labels == "tensor")[0][0])
    if not bool(valid[base_idx]) and bool(np.any(valid)):
        base_idx = int(np.flatnonzero(valid)[0])
    base_mass = masses[base_idx] if bool(valid[base_idx]) else float("nan")
    base_label = str(labels[base_idx])

    correction_scale = float("nan")
    correction_scale_err = float("nan")
    if (
        np.isfinite(consensus_mass)
        and consensus_mass > 0
        and np.isfinite(base_mass)
        and base_mass > 0
    ):
        correction_scale = float(consensus_mass / base_mass)
        base_err = errors[base_idx]
        if np.isfinite(consensus_err) and np.isfinite(base_err):
            rel_cons = consensus_err / max(consensus_mass, 1e-12)
            rel_base = base_err / max(base_mass, 1e-12)
            correction_scale_err = abs(correction_scale) * float(
                np.sqrt(rel_cons**2 + rel_base**2)
            )

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

STRONG_FORCE_RATIO_REFERENCE_VALUES: dict[tuple[str, str], float] = {
    ("nucleon", "pseudoscalar"): float(BARYON_REFS["proton"] / MESON_REFS["pion"]),
    ("vector", "pseudoscalar"): float(MESON_REFS["rho"] / MESON_REFS["pion"]),
    ("scalar", "pseudoscalar"): float(0.500 / MESON_REFS["pion"]),
    ("glueball", "pseudoscalar"): 10.0,
}

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

ANISOTROPIC_EDGE_COMPANION_FAMILY_MAP: dict[str, str] = {
    "pseudoscalar_companion": "meson",
    "pseudoscalar_score_directed_companion": "meson",
    "nucleon_companion": "baryon",
    "nucleon_score_signed_companion": "baryon",
    "nucleon_score_abs_companion": "baryon",
    "glueball_companion": "glueball",
}

ANISOTROPIC_EDGE_COMPANION_RATIO_SPECS: list[tuple[str, str, str]] = [
    ("nucleon_companion", "pseudoscalar_companion", "proton/pion ≈ 6.7"),
    ("vector_companion", "pseudoscalar_companion", "rho/pion ≈ 5.5"),
    ("scalar_companion", "pseudoscalar_companion", "sigma/pion"),
    ("glueball_companion", "pseudoscalar_companion", "glueball/pion ≈ 10"),
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
        rows.append({
            "channel": name,
            "alg_mass": alg_mass,
            "obs_mass_GeV": obs,
            "pred_mass_GeV": pred,
            "error_pct": err_pct,
        })
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
        rows.append({
            "ratio": f"{name}/{base_name}",
            "measured": measured,
            "observed": observed,
            "error_pct": error_pct,
        })
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


def _extract_n_windows_for_filter(result: ChannelCorrelatorResult) -> int:
    """Extract valid window count consistently for companion GEVP table filtering."""
    mass_fit = result.mass_fit if isinstance(result.mass_fit, dict) else {}
    raw = mass_fit.get("n_valid_windows", None)
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass

    window_masses = getattr(result, "window_masses", None)
    if isinstance(window_masses, torch.Tensor):
        if int(window_masses.numel()) <= 0:
            return 0
        return int(torch.isfinite(window_masses).sum().item())
    if isinstance(window_masses, list | tuple):
        count = 0
        for value in window_masses:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                count += 1
        return count
    return 0


def _companion_gevp_filter_reason(
    result: ChannelCorrelatorResult,
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
) -> str | None:
    """Return exclusion reason when a result fails active GEVP-quality filters."""
    mass_fit = result.mass_fit if isinstance(result.mass_fit, dict) else {}
    mass = float(mass_fit.get("mass", float("nan")))
    mass_error = float(mass_fit.get("mass_error", float("nan")))
    r2 = float(mass_fit.get("r_squared", float("nan")))
    n_windows = _extract_n_windows_for_filter(result)

    reasons: list[str] = []
    if np.isfinite(min_r2):
        if not np.isfinite(r2) or r2 < min_r2:
            r2_text = "nan" if not np.isfinite(r2) else f"{r2:.3g}"
            reasons.append(f"r2={r2_text}<{min_r2:.3g}")
    if n_windows < int(max(0, min_windows)):
        reasons.append(f"n_windows={n_windows}<{int(max(0, min_windows))}")

    if np.isfinite(max_error_pct) and max_error_pct >= 0:
        if np.isfinite(mass) and mass > 0 and np.isfinite(mass_error) and mass_error >= 0:
            err_pct = abs(mass_error / mass) * 100.0
        else:
            err_pct = float("inf")
        if err_pct > max_error_pct:
            err_text = f"{err_pct:.3g}" if np.isfinite(err_pct) else "inf"
            reasons.append(f"err_pct={err_text}>{max_error_pct:.3g}")

    if remove_artifacts:
        if not np.isfinite(mass_error):
            reasons.append("mass_error=nan_or_inf")
        elif mass_error == 0.0:
            reasons.append("mass_error==0")
        if np.isfinite(mass) and mass == 0.0:
            reasons.append("mass==0")

    return ", ".join(reasons) if reasons else None


def _split_results_by_companion_gevp_filters(
    results: dict[str, ChannelCorrelatorResult],
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
    keep_gevp_results: bool = False,
) -> tuple[dict[str, ChannelCorrelatorResult], dict[str, tuple[ChannelCorrelatorResult, str]]]:
    """Split results into passing and filtered-out sets under active GEVP-quality filters."""
    kept: dict[str, ChannelCorrelatorResult] = {}
    filtered_out: dict[str, tuple[ChannelCorrelatorResult, str]] = {}
    for name, result in results.items():
        if result is None or int(getattr(result, "n_samples", 0)) <= 0:
            continue
        mass_fit = result.mass_fit if isinstance(result.mass_fit, dict) else {}
        source = str(mass_fit.get("source", "")).strip()
        is_gevp_result = str(name).endswith("_gevp") or source.startswith("gevp_")
        if keep_gevp_results and is_gevp_result:
            kept[name] = result
            continue
        reason = _companion_gevp_filter_reason(
            result,
            min_r2=min_r2,
            min_windows=min_windows,
            max_error_pct=max_error_pct,
            remove_artifacts=remove_artifacts,
        )
        if reason is None:
            kept[name] = result
        else:
            filtered_out[name] = (result, reason)
    return kept, filtered_out


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
        mass_error_pct = float("nan")
        if (
            np.isfinite(float(mass))
            and float(mass) > 0
            and np.isfinite(float(mass_error))
            and float(mass_error) >= 0
            and float(mass_error) < float("inf")
        ):
            mass_error_pct = abs(float(mass_error) / float(mass)) * 100.0
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
            "mass_error_pct": (f"{mass_error_pct:.2f}%" if np.isfinite(mass_error_pct) else "n/a"),
            "r2": f"{r2:.4f}" if np.isfinite(r2) else "n/a",
            "n_windows": n_windows,
            "n_samples": result.n_samples,
        })
    table.value = pd.DataFrame(rows) if rows else pd.DataFrame()


def _update_correlator_plots(
    results: dict[str, ChannelCorrelatorResult],
    plateau_container,
    spectrum_pane,
    overlay_corr_pane: pn.pane.HoloViews | None,
    overlay_meff_pane: pn.pane.HoloViews | None,
    heatmap_container=None,
    heatmap_color_metric_widget=None,
    heatmap_alpha_metric_widget=None,
    spectrum_builder=None,
    correlator_logy: bool = True,
    channels_per_row: int = 1,
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
            color_metric = (
                str(heatmap_color_metric_widget.value) if heatmap_color_metric_widget else "mass"
            )
            alpha_metric = (
                str(heatmap_alpha_metric_widget.value) if heatmap_alpha_metric_widget else "aic"
            )
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

    if channel_plot_items:
        ncols = max(1, int(channels_per_row))
        if ncols == 1:
            plateau_container.objects = channel_plot_items
        else:
            rows = [
                pn.Row(*channel_plot_items[i : i + ncols], sizing_mode="stretch_width")
                for i in range(0, len(channel_plot_items), ncols)
            ]
            plateau_container.objects = rows
    else:
        plateau_container.objects = [pn.pane.Markdown("_No channel plots available._")]

    if heatmap_container is not None:
        heatmap_container.objects = heatmap_items or [
            pn.pane.Markdown("_No window heatmaps available._")
        ]

    if spectrum_builder is None:
        spectrum_builder = build_mass_spectrum_bar
    spectrum_pane.object = spectrum_builder(results)
    if overlay_corr_pane is not None:
        overlay_corr_pane.object = build_all_channels_overlay(
            results,
            plot_type="correlator",
            correlator_logy=correlator_logy,
        )
    if overlay_meff_pane is not None:
        overlay_meff_pane.object = build_all_channels_overlay(results, plot_type="effective_mass")


def _update_strong_tables(
    results: dict[str, ChannelCorrelatorResult],
    mode: str,
    mass_table,
    ratio_pane,
    fit_table,
    anchor_table=None,
    glueball_ref_input=None,
    mass_getter=None,
    error_getter=None,
    ratio_specs=None,
    anchor_mode: str = "per_anchor_row",
    calibration_family_map: dict[str, str] | None = None,
    calibration_ratio_specs: list[tuple[str, str, str]] | None = None,
    comparison_channel_overrides: dict[str, str] | None = None,
) -> None:
    """Orchestrate all strong-force table updates (module-level, no closure)."""
    if ratio_specs is None:
        ratio_specs = STRONG_FORCE_RATIO_SPECS
    ratio_specs_for_table = (
        calibration_ratio_specs if calibration_ratio_specs is not None else ratio_specs
    )
    family_map = (
        calibration_family_map if calibration_family_map is not None else _STRONG_FAMILY_MAP
    )

    _update_mass_table(
        results, mass_table, mode, mass_getter=mass_getter, error_getter=error_getter
    )
    comparison_results = dict(results)
    if comparison_channel_overrides:
        for canonical_name, selected_name in comparison_channel_overrides.items():
            key = str(canonical_name).strip()
            selected_key = str(selected_name).strip()
            if not key or not selected_key:
                continue
            selected_result = results.get(selected_key)
            if selected_result is None:
                continue
            comparison_results[key] = selected_result
    ratio_pane.object = _format_ratios(
        comparison_results,
        mode,
        mass_getter=mass_getter,
        ratio_specs=ratio_specs_for_table,
    )

    channel_masses = _extract_masses(
        comparison_results,
        mode,
        mass_getter=mass_getter,
        family_map=family_map,
    )
    channel_r2 = _extract_r2(
        comparison_results,
        mode,
        family_map=family_map,
    )

    if not channel_masses:
        fit_table.value = pd.DataFrame()
        if anchor_table is not None:
            anchor_table.value = pd.DataFrame()
        return

    fit_rows = _build_best_fit_rows(channel_masses, channel_r2)
    fit_table.value = pd.DataFrame(fit_rows)

    glueball_ref = None
    if glueball_ref_input is not None and glueball_ref_input.value is not None:
        glueball_ref = ("glueball", float(glueball_ref_input.value))

    if anchor_table is None:
        return

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
        gas_config = GasConfigPanel.create_qft_config(spatial_dims=2, bounds_extent=30.0)
        gas_config.hide_viscous_kernel_widgets = False
        gas_config.hide_compute_volume_weight_widget = True
        gas_config.hide_viscous_coupling_widget = False
        gas_config.hide_diffusion_widgets = True
        gas_config.hide_integrator_widget = True
        gas_config.hide_localization_scale_widget = False
        gas_config.hide_distance_reg_widget = True
        gas_config.hide_companion_interaction_range_widgets = True
        gas_config.hide_lambda_alg_widgets = True
        gas_config.benchmark_name = "Riemannian Mix"
        # Override with the best stable calibration settings found in QFT tuning.
        # This matches weak_potential_fit1_aniso_stable2 (200 walkers, 300 steps).
        gas_config.n_steps = 750
        gas_config.gas_params["N"] = 500
        gas_config.gas_params["dtype"] = "float32"
        gas_config.gas_params["pbc"] = False
        gas_config.gas_params["clone_every"] = 3
        gas_config.neighbor_graph_method = "delaunay"
        gas_config.neighbor_graph_record = True
        gas_config.neighbor_weight_modes = [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ]
        gas_config.init_offset = 0.0
        gas_config.init_spread = 0.0
        gas_config.init_velocity_scale = 0.0

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
        gas_config.kinetic_op.integrator = "boris-baoab"
        gas_config.kinetic_op.auto_thermostat = True
        gas_config.kinetic_op.delta_t = 0.01
        gas_config.kinetic_op.epsilon_F = 1.0
        gas_config.kinetic_op.use_fitness_force = False
        gas_config.kinetic_op.use_potential_force = False
        gas_config.kinetic_op.use_anisotropic_diffusion = False
        gas_config.kinetic_op.diagonal_diffusion = False
        gas_config.kinetic_op.diffusion_mode = "voronoi_proxy"
        gas_config.kinetic_op.diffusion_grad_scale = 30.0
        gas_config.kinetic_op.epsilon_Sigma = 0.5
        gas_config.kinetic_op.nu = 1.0
        gas_config.kinetic_op.beta_curl = 1.0
        gas_config.kinetic_op.use_viscous_coupling = True
        gas_config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel_volume"
        gas_config.kinetic_op.viscous_length_scale = 0.5
        gas_config.kinetic_op.viscous_neighbor_penalty = 0.0
        gas_config.kinetic_op.compute_volume_weights = False
        gas_config.kinetic_op.use_velocity_squashing = False

        # Companion selection (diversity + cloning).
        gas_config.companion_selection.method = "random_pairing"
        gas_config.companion_selection.epsilon = 2.80
        gas_config.companion_selection.lambda_alg = 0.0
        gas_config.companion_selection.exclude_self = True
        gas_config.companion_selection_clone.method = "random_pairing"
        gas_config.companion_selection_clone.epsilon = 1.68419
        gas_config.companion_selection_clone.lambda_alg = 0.0
        gas_config.companion_selection_clone.exclude_self = True

        # Cloning operator.
        gas_config.cloning.p_max = 1.0
        gas_config.cloning.epsilon_clone = 1e-6
        gas_config.cloning.sigma_x = 0.01
        gas_config.cloning.alpha_restitution = 1.0

        # Fitness operator.
        gas_config.fitness_op.alpha = 1.0
        gas_config.fitness_op.beta = 1.0
        gas_config.fitness_op.eta = 0.0
        gas_config.fitness_op.lambda_alg = 0.0
        gas_config.fitness_op.sigma_min = 0.0
        gas_config.fitness_op.epsilon_dist = 1e-8
        gas_config.fitness_op.A = 2.0
        gas_config.fitness_op.rho = None
        gas_config.fitness_op.grad_mode = "sum"
        gas_config.fitness_op.detach_stats = True
        gas_config.fitness_op.detach_companions = True
        visualizer = SwarmConvergence3D(history=None, bounds_extent=gas_config.bounds_extent)

        state: dict[str, Any] = {
            "history": None,
            "history_path": None,
            "fractal_set_points": None,
            "fractal_set_regressions": None,
            "fractal_set_frame_summary": None,
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
            "anisotropic_edge_multiscale_output": None,
            "anisotropic_edge_multiscale_error": None,
            "companion_strong_force_results": None,
            "companion_strong_force_multiscale_output": None,
            "companion_strong_force_multiscale_error": None,
            "companion_strong_force_gevp_error": None,
            "companion_strong_force_plots_unlocked": False,
            "tensor_calibration_base_results": None,
            "tensor_calibration_strong_result": None,
            "tensor_calibration_momentum_results": None,
            "tensor_calibration_momentum_meta": None,
            "tensor_calibration_noncomp_multiscale_output": None,
            "tensor_calibration_companion_multiscale_output": None,
            "tensor_calibration_payload": None,
            "new_dirac_ew_bundle": None,
            "new_dirac_ew_results": None,
            "new_dirac_ew_multiscale_output": None,
            "new_dirac_ew_multiscale_error": None,
            "new_dirac_ew_gevp_error": None,
            "new_dirac_ew_comparison_overrides": None,
            "new_dirac_ew_ratio_specs": None,
            "_multiscale_geodesic_distance_by_frame": None,
            "new_dirac_bundle": None,
            "coupling_diagnostics_output": None,
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
        algorithm_velocity_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show walker speed."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_geodesic_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show geodesic edge distances."),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_rkv_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot(
                "Load RunHistory to show riemannian kernel volume weights."
            ),
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        algorithm_lyapunov_plot = pn.pane.HoloViews(
            _algorithm_placeholder_plot("Load RunHistory to show Lyapunov convergence."),
            linked_axes=False,
            sizing_mode="stretch_width",
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
        fractal_set_plot_dist_geom = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        fractal_set_plot_fit_geom = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        fractal_set_plot_total_geom = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
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
                "mc_time_index",
                "regularization",
                "stress_energy_mode",
                "bulk_fraction",
                "scalar_density_mode",
                "knn_k",
                "coarse_grain_bins",
                "coarse_grain_min_points",
                "temporal_average_enabled",
                "temporal_window_frames",
                "temporal_stride",
                "bootstrap_samples",
                "bootstrap_confidence",
                "bootstrap_seed",
                "bootstrap_frame_block_size",
                "g_newton_metric",
                "g_newton_manual",
            ],
        )
        einstein_summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        einstein_scalar_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        einstein_scalar_log_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        einstein_tensor_table = pn.pane.HoloViews(linked_axes=False)
        einstein_curvature_hist = pn.pane.HoloViews(linked_axes=False)
        einstein_residual_map = pn.pane.HoloViews(linked_axes=False)
        einstein_crosscheck_plot = pn.pane.HoloViews(linked_axes=False)
        einstein_bulk_boundary = pn.pane.Markdown("", sizing_mode="stretch_width")

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
                "use_multiscale_kernels",
                "n_scales",
                "kernel_type",
                "kernel_distance_method",
                "kernel_assume_all_alive",
                "kernel_batch_size",
                "kernel_scale_frames",
                "kernel_scale_q_low",
                "kernel_scale_q_high",
                "kernel_bootstrap_mode",
                "kernel_bootstrap_max_walkers",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC start (step or idx)"},
                "spatial_dims_spec": {"name": "Spatial dims (blank=all)"},
                "use_multiscale_kernels": {"name": "Enable multiscale kernels"},
                "n_scales": {"name": "Number of scales"},
                "kernel_type": {"name": "Kernel shape"},
                "kernel_distance_method": {"name": "Distance solver"},
                "kernel_assume_all_alive": {"name": "Assume all walkers alive"},
                "kernel_batch_size": {"name": "Kernel frame batch size"},
                "kernel_scale_frames": {"name": "Scale calibration frames"},
                "kernel_scale_q_low": {"name": "Scale quantile low"},
                "kernel_scale_q_high": {"name": "Scale quantile high"},
                "kernel_bootstrap_mode": {"name": "Kernel bootstrap mode"},
                "kernel_bootstrap_max_walkers": {
                    "name": "Kernel bootstrap max walkers",
                },
            },
            default_layout=type("AnisotropicEdgeSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        anisotropic_edge_base_settings_row = pn.Row(
            anisotropic_edge_settings_panel,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_settings_layout = pn.Column(
            pn.pane.Markdown("### Base Channel Settings"),
            anisotropic_edge_base_settings_row,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_summary = pn.pane.Markdown(
            "## Anisotropic Edge Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_multiscale_summary = pn.pane.Markdown(
            "### Multiscale Kernel Summary\n_Multiscale kernels disabled._",
            sizing_mode="stretch_width",
        )
        anisotropic_edge_multiscale_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        anisotropic_edge_multiscale_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False
        )
        _msw = create_multiscale_widgets()
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
        # Companion strong-force tab components (companion-only operators)
        # =====================================================================
        companion_strong_force_settings = AnisotropicEdgeSettings()
        companion_strong_force_settings.max_lag = 40
        companion_strong_force_settings.h_eff_mode = "auto_sigma_s"
        companion_strong_force_status = pn.pane.Markdown(
            "**Companion Strong Force:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        companion_strong_force_run_button = pn.widgets.Button(
            name="Compute Companion Strong Force Channels",
            button_type="primary",
            min_width=260,
            sizing_mode="stretch_width",
            disabled=True,
        )
        companion_strong_force_display_plots_button = pn.widgets.Button(
            name="Display Plots",
            button_type="default",
            min_width=180,
            sizing_mode="stretch_width",
            disabled=True,
        )
        companion_strong_force_plot_gate_note = pn.pane.Markdown(
            "_Plots are hidden. Click `Display Plots` after computing channels._",
            sizing_mode="stretch_width",
        )
        companion_strong_force_settings_panel = pn.Param(
            companion_strong_force_settings,
            parameters=[
                "simulation_range",
                "mc_time_index",
                "max_lag",
                "use_connected",
                "h_eff",
                "h_eff_mode",
                "mass",
                "ell0",
                "edge_weight_mode",
                "window_widths_spec",
                "fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
                "compute_bootstrap_errors",
                "n_bootstrap",
                "use_multiscale_kernels",
                "n_scales",
                "kernel_type",
                "kernel_distance_method",
                "kernel_assume_all_alive",
                "kernel_batch_size",
                "kernel_scale_frames",
                "kernel_scale_q_low",
                "kernel_scale_q_high",
                "kernel_bootstrap_mode",
                "kernel_bootstrap_max_walkers",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC start (step or idx)"},
                "h_eff_mode": {"name": "h_eff mode"},
                "use_multiscale_kernels": {"name": "Enable multiscale kernels"},
                "n_scales": {"name": "Number of scales"},
                "kernel_type": {"name": "Kernel shape"},
                "kernel_distance_method": {"name": "Distance solver"},
                "kernel_assume_all_alive": {"name": "Assume all walkers alive"},
                "kernel_batch_size": {"name": "Kernel frame batch size"},
                "kernel_scale_frames": {"name": "Scale calibration frames"},
                "kernel_scale_q_low": {"name": "Scale quantile low"},
                "kernel_scale_q_high": {"name": "Scale quantile high"},
                "kernel_bootstrap_mode": {"name": "Kernel bootstrap mode"},
                "kernel_bootstrap_max_walkers": {"name": "Kernel bootstrap max walkers"},
            },
            default_layout=type("CompanionStrongForceSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        companion_strong_force_baryon_panel = pn.Param(
            companion_strong_force_settings,
            parameters=[
                "use_companion_baryon_triplet",
                "baryon_use_connected",
                "baryon_max_lag",
                "baryon_color_dims_spec",
                "baryon_eps",
                "baryon_operator_mode",
                "baryon_flux_exp_alpha",
                "use_companion_nucleon_gevp",
                "gevp_basis_strategy",
                "gevp_t0",
                "gevp_max_basis",
                "gevp_min_operator_r2",
                "gevp_min_operator_windows",
                "gevp_max_operator_error_pct",
                "gevp_remove_artifacts",
                "gevp_eig_rel_cutoff",
                "gevp_cond_limit",
                "gevp_shrinkage",
                "gevp_bootstrap_mode",
            ],
            show_name=False,
            widgets={
                "baryon_max_lag": {"name": "Baryon max lag (blank=use max_lag)"},
                "baryon_color_dims_spec": {"name": "Baryon color dims (3 dims)"},
                "baryon_operator_mode": {"name": "Baryon operator mode"},
                "baryon_flux_exp_alpha": {"name": "Flux exp α"},
                "use_companion_nucleon_gevp": {"name": "Enable companion GEVP"},
                "gevp_basis_strategy": {"name": "GEVP basis strategy"},
                "gevp_t0": {"name": "GEVP t0"},
                "gevp_max_basis": {"name": "GEVP max basis"},
                "gevp_min_operator_r2": {"name": "GEVP min operator R²"},
                "gevp_min_operator_windows": {"name": "GEVP min operator windows"},
                "gevp_max_operator_error_pct": {"name": "GEVP max operator error %"},
                "gevp_remove_artifacts": {"name": "GEVP remove artifacts"},
                "gevp_eig_rel_cutoff": {"name": "GEVP rel eig cutoff"},
                "gevp_cond_limit": {"name": "GEVP cond limit"},
                "gevp_shrinkage": {"name": "GEVP shrinkage"},
                "gevp_bootstrap_mode": {"name": "GEVP bootstrap mode"},
            },
            default_layout=type("CompanionStrongForceBaryonGrid", (pn.GridBox,), {"ncols": 1}),
        )
        companion_strong_force_meson_panel = pn.Param(
            companion_strong_force_settings,
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
            default_layout=type("CompanionStrongForceMesonGrid", (pn.GridBox,), {"ncols": 1}),
        )
        companion_strong_force_vector_panel = pn.Param(
            companion_strong_force_settings,
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
            default_layout=type("CompanionStrongForceVectorGrid", (pn.GridBox,), {"ncols": 1}),
        )
        companion_strong_force_glueball_panel = pn.Param(
            companion_strong_force_settings,
            parameters=[
                "use_companion_glueball_color",
                "glueball_use_connected",
                "glueball_max_lag",
                "glueball_color_dims_spec",
                "glueball_eps",
                "glueball_use_action_form",
                "glueball_operator_mode",
                "glueball_use_momentum_projection",
                "glueball_momentum_axis",
                "glueball_momentum_mode_max",
            ],
            show_name=False,
            widgets={
                "glueball_max_lag": {"name": "Glueball max lag (blank=use max_lag)"},
                "glueball_color_dims_spec": {"name": "Glueball color dims (3 dims)"},
                "glueball_operator_mode": {"name": "Glueball operator mode"},
                "glueball_momentum_axis": {"name": "Momentum axis"},
                "glueball_momentum_mode_max": {"name": "Momentum n_max"},
            },
            default_layout=type("CompanionStrongForceGlueballGrid", (pn.GridBox,), {"ncols": 1}),
        )
        companion_strong_force_channel_family_selectors: dict[str, pn.widgets.MultiSelect] = {
            family: pn.widgets.MultiSelect(
                name=f"{family.replace('_', ' ').title()} Variants",
                options=list(variants),
                value=list(DEFAULT_COMPANION_CHANNEL_VARIANT_SELECTION.get(family, ())),
                size=6,
            )
            for family, variants in COMPANION_CHANNEL_VARIANTS_BY_FAMILY.items()
        }
        companion_strong_force_channel_family_selector_grid = pn.GridBox(
            *companion_strong_force_channel_family_selectors.values(),
            ncols=2,
            sizing_mode="stretch_width",
        )
        companion_strong_force_channel_family_selector_layout = pn.Column(
            pn.pane.Markdown("### Companion Channel Family Selection"),
            companion_strong_force_channel_family_selector_grid,
            sizing_mode="stretch_width",
        )
        companion_strong_force_settings_layout = pn.Column(
            pn.pane.Markdown("### Companion Channel Settings"),
            companion_strong_force_settings_panel,
            pn.layout.Divider(),
            companion_strong_force_channel_family_selector_layout,
            pn.layout.Divider(),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Baryon Triplet Settings"),
                    companion_strong_force_baryon_panel,
                    pn.layout.Divider(),
                    pn.pane.Markdown("### Meson Phase Settings"),
                    companion_strong_force_meson_panel,
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    pn.pane.Markdown("### Vector Meson Settings"),
                    companion_strong_force_vector_panel,
                    pn.layout.Divider(),
                    pn.pane.Markdown("### Glueball Color Settings"),
                    companion_strong_force_glueball_panel,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
        companion_strong_force_summary = pn.pane.Markdown(
            "## Companion Strong Force Summary\n_Run analysis to populate._",
            sizing_mode="stretch_width",
        )
        companion_strong_force_multiscale_summary = pn.pane.Markdown(
            "### Multiscale Kernel Summary\n_Multiscale kernels disabled (original companion estimators only)._",
            sizing_mode="stretch_width",
        )
        companion_strong_force_multiscale_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_multiscale_per_scale_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_multiscale_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        companion_strong_force_mass_mode = pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
        )
        companion_strong_force_variant_pseudoscalar = pn.widgets.Select(
            name="Pseudoscalar Variant",
            options=["pseudoscalar"],
            value="pseudoscalar",
            sizing_mode="stretch_width",
        )
        companion_strong_force_variant_nucleon = pn.widgets.Select(
            name="Nucleon Variant",
            options=["nucleon"],
            value="nucleon",
            sizing_mode="stretch_width",
        )
        companion_strong_force_variant_glueball = pn.widgets.Select(
            name="Glueball Variant",
            options=["glueball"],
            value="glueball",
            sizing_mode="stretch_width",
        )
        companion_strong_force_variant_scalar = pn.widgets.Select(
            name="Scalar Variant",
            options=["scalar"],
            value="scalar",
            sizing_mode="stretch_width",
        )
        companion_strong_force_variant_vector = pn.widgets.Select(
            name="Vector Variant",
            options=["vector"],
            value="vector",
            sizing_mode="stretch_width",
        )
        companion_strong_force_variant_axial_vector = pn.widgets.Select(
            name="Axial Vector Variant",
            options=["axial_vector"],
            value="axial_vector",
            sizing_mode="stretch_width",
        )
        companion_strong_force_variant_selectors: dict[str, pn.widgets.Select] = {
            "pseudoscalar": companion_strong_force_variant_pseudoscalar,
            "nucleon": companion_strong_force_variant_nucleon,
            "glueball": companion_strong_force_variant_glueball,
            "scalar": companion_strong_force_variant_scalar,
            "vector": companion_strong_force_variant_vector,
            "axial_vector": companion_strong_force_variant_axial_vector,
        }
        companion_strong_force_variant_sync = {"active": False}
        companion_strong_force_heatmap_color_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Color",
            options=["mass", "aic", "r2"],
            value="mass",
            button_type="default",
        )
        companion_strong_force_heatmap_alpha_metric = pn.widgets.RadioButtonGroup(
            name="Heatmap Opacity",
            options=["aic", "mass", "r2"],
            value="aic",
            button_type="default",
        )
        companion_strong_force_plots_spectrum = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        companion_strong_force_plateau_plots = pn.Column(sizing_mode="stretch_width")
        companion_strong_force_heatmap_plots = pn.Column(sizing_mode="stretch_width")
        companion_strong_force_plateau_plots.objects = [
            pn.pane.Markdown(
                "_Plots are hidden. Click `Display Plots` to render channel plots._",
                sizing_mode="stretch_width",
            )
        ]
        companion_strong_force_heatmap_plots.objects = [
            pn.pane.Markdown(
                "_Plots are hidden. Click `Display Plots` to render heatmaps._",
                sizing_mode="stretch_width",
            )
        ]
        companion_strong_force_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_filtered_summary = pn.pane.Markdown(
            "**Filtered-out candidates:** run companion channels to populate.",
            sizing_mode="stretch_width",
        )
        companion_strong_force_filtered_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_ratio_pane = pn.pane.Markdown(
            (
                "**Mass Ratios (Best Global Combo):** compute companion channels to "
                "populate best-combination ratios."
            ),
            sizing_mode="stretch_width",
        )
        companion_strong_force_ratio_tables = pn.Column(
            pn.pane.Markdown("_Per-ratio variant tables will appear after computation._"),
            sizing_mode="stretch_width",
        )
        companion_strong_force_best_combo_summary = pn.pane.Markdown(
            (
                "**Best global variant combo:** compute companion channels to evaluate "
                "the top 5 one-variant-per-channel combinations by total absolute % ratio error."
            ),
            sizing_mode="stretch_width",
        )
        companion_strong_force_best_combo_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_fit_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_cross_ratio_summary = pn.pane.Markdown(
            (
                "**Cross-channel ratio debug:** run companion channels to populate all "
                "cross-channel variant ratios."
            ),
            sizing_mode="stretch_width",
        )
        companion_strong_force_cross_ratio_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=30,
            show_index=False,
            sizing_mode="stretch_width",
        )
        companion_strong_force_ref_table = pn.widgets.Tabulator(
            _build_hadron_reference_table(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
        )
        companion_strong_force_glueball_ref_input = pn.widgets.FloatInput(
            name="Glueball ref (GeV)",
            value=None,
            step=0.01,
            min_width=200,
            sizing_mode="stretch_width",
        )

        # =====================================================================
        # Tensor calibration tab components
        # =====================================================================
        tensor_calibration_settings = AnisotropicEdgeSettings()
        tensor_calibration_run_button = pn.widgets.Button(
            name="Compute Tensor Calibration",
            button_type="primary",
            min_width=260,
            sizing_mode="stretch_width",
            disabled=True,
        )
        tensor_calibration_settings_panel = pn.Param(
            tensor_calibration_settings,
            parameters=[
                "simulation_range",
                "mc_time_index",
                "max_lag",
                "use_connected",
                "h_eff",
                "mass",
                "ell0",
                "edge_weight_mode",
                "use_volume_weights",
                "component_mode",
                "window_widths_spec",
                "fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
                "compute_bootstrap_errors",
                "n_bootstrap",
                "use_companion_tensor_momentum",
                "tensor_momentum_use_connected",
                "tensor_momentum_max_lag",
                "tensor_momentum_pair_selection",
                "tensor_momentum_color_dims_spec",
                "tensor_momentum_position_dims_spec",
                "tensor_momentum_axis",
                "tensor_momentum_mode_max",
                "tensor_momentum_eps",
                "n_scales",
                "kernel_type",
                "kernel_distance_method",
                "kernel_assume_all_alive",
                "kernel_batch_size",
                "kernel_scale_frames",
                "kernel_scale_q_low",
                "kernel_scale_q_high",
                "kernel_bootstrap_mode",
                "kernel_bootstrap_max_walkers",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC start (step or idx)"},
                "tensor_momentum_max_lag": {"name": "Tensor momentum max lag"},
                "tensor_momentum_pair_selection": {"name": "Tensor momentum pair selection"},
                "tensor_momentum_color_dims_spec": {"name": "Tensor momentum color dims (3 dims)"},
                "tensor_momentum_position_dims_spec": {
                    "name": "Tensor momentum position dims (3 dims)"
                },
                "tensor_momentum_axis": {"name": "Tensor momentum axis"},
                "tensor_momentum_mode_max": {"name": "Tensor momentum n_max"},
                "n_scales": {"name": "Number of scales"},
                "kernel_type": {"name": "Kernel shape"},
                "kernel_distance_method": {"name": "Distance solver"},
                "kernel_assume_all_alive": {"name": "Assume all walkers alive"},
                "kernel_batch_size": {"name": "Kernel frame batch size"},
                "kernel_scale_frames": {"name": "Scale calibration frames"},
                "kernel_scale_q_low": {"name": "Scale quantile low"},
                "kernel_scale_q_high": {"name": "Scale quantile high"},
                "kernel_bootstrap_mode": {"name": "Kernel bootstrap mode"},
                "kernel_bootstrap_max_walkers": {"name": "Kernel bootstrap max walkers"},
            },
            default_layout=type("TensorCalibrationSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        tensor_calibration_settings_layout = pn.Column(
            tensor_calibration_settings_panel,
            sizing_mode="stretch_width",
        )
        _tcw = create_tensor_gevp_calibration_widgets()

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
                "companion_topology_u1",
                "companion_topology_su2",
                "companion_topology_ew_mixed",
                "edge_weight_mode",
                "neighbor_k",
                "window_widths_spec",
                "fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
                "epsilon_clone",
                "lambda_alg",
                "su2_operator_mode",
                "enable_walker_type_split",
                "walker_type_scope",
                "compute_bootstrap_errors",
                "n_bootstrap",
                "use_multiscale_kernels",
                "n_scales",
                "kernel_type",
                "kernel_distance_method",
                "kernel_assume_all_alive",
                "kernel_batch_size",
                "kernel_scale_frames",
                "kernel_scale_q_low",
                "kernel_scale_q_high",
                "kernel_bootstrap_mode",
                "use_su2_gevp",
                "gevp_basis_strategy",
                "gevp_t0",
                "gevp_max_basis",
                "gevp_min_operator_r2",
                "gevp_min_operator_windows",
                "gevp_max_operator_error_pct",
                "gevp_remove_artifacts",
                "gevp_eig_rel_cutoff",
                "gevp_cond_limit",
                "gevp_shrinkage",
                "gevp_bootstrap_mode",
            ],
            show_name=False,
            widgets={
                "mc_time_index": {"name": "MC time slice (step or idx; blank=last)"},
                "kernel_type": {"name": "Kernel shape"},
                "kernel_distance_method": {"name": "Distance solver"},
                "kernel_assume_all_alive": {"name": "Assume all walkers alive"},
                "kernel_batch_size": {"name": "Kernel frame batch size"},
                "kernel_scale_frames": {"name": "Scale calibration frames"},
                "kernel_scale_q_low": {"name": "Scale quantile low"},
                "kernel_scale_q_high": {"name": "Scale quantile high"},
                "kernel_bootstrap_mode": {"name": "Kernel bootstrap mode"},
                "su2_operator_mode": {"name": "SU(2) operator mode"},
                "enable_walker_type_split": {"name": "Enable walker-type split"},
                "walker_type_scope": {"name": "Walker-type scope"},
                "use_su2_gevp": {"name": "Enable SU(2) GEVP"},
                "gevp_basis_strategy": {"name": "GEVP basis strategy"},
                "gevp_t0": {"name": "GEVP t0"},
                "gevp_max_basis": {"name": "GEVP max basis"},
                "gevp_min_operator_r2": {"name": "GEVP min operator R²"},
                "gevp_min_operator_windows": {"name": "GEVP min operator windows"},
                "gevp_max_operator_error_pct": {"name": "GEVP max operator error %"},
                "gevp_remove_artifacts": {"name": "GEVP remove artifacts"},
                "gevp_eig_rel_cutoff": {"name": "GEVP rel eig cutoff"},
                "gevp_cond_limit": {"name": "GEVP cond limit"},
                "gevp_shrinkage": {"name": "GEVP shrinkage"},
                "gevp_bootstrap_mode": {"name": "GEVP bootstrap mode"},
            },
            default_layout=type("NewDiracEWSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        new_dirac_ew_channel_family_selectors: dict[str, pn.widgets.MultiSelect] = {
            family: pn.widgets.MultiSelect(
                name=f"{family.replace('_', ' ').title()} Variants",
                options=list(variants),
                value=list(DEFAULT_ELECTROWEAK_CHANNEL_VARIANT_SELECTION.get(family, ())),
                size=6,
            )
            for family, variants in ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY.items()
        }
        new_dirac_ew_channel_family_selector_grid = pn.GridBox(
            *new_dirac_ew_channel_family_selectors.values(),
            ncols=2,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_channel_family_selector_layout = pn.Column(
            pn.pane.Markdown("### Electroweak Channel Family Selection"),
            new_dirac_ew_channel_family_selector_grid,
            sizing_mode="stretch_width",
        )
        new_dirac_status = pn.pane.Markdown(
            "**Dirac:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        new_dirac_run_button = pn.widgets.Button(
            name="Compute Dirac",
            button_type="primary",
            min_width=260,
            sizing_mode="stretch_width",
            disabled=True,
        )
        new_dirac_settings_panel = pn.Param(
            new_dirac_ew_settings,
            parameters=[
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
                "dirac_color_threshold_mode": {"name": "Dirac color threshold"},
                "dirac_color_threshold_value": {"name": "||F_visc|| threshold"},
                "color_singlet_quantile": {"name": "Color-singlet quantile"},
            },
            default_layout=type("NewDiracSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )

        new_dirac_ew_coupling_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_coupling_ref_table = pn.widgets.Tabulator(
            pd.DataFrame({
                "name": list(ELECTROWEAK_COUPLING_NAMES),
                **{
                    col: [
                        _format_ref_value(DEFAULT_ELECTROWEAK_COUPLING_REFS.get(name, {}).get(col))
                        for name in ELECTROWEAK_COUPLING_NAMES
                    ]
                    for col in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS
                },
            }),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors=dict.fromkeys(ELECTROWEAK_COUPLING_REFERENCE_COLUMNS, "input"),
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
        new_dirac_ew_best_combo_summary = pn.pane.Markdown(
            (
                "**Best global variant combo:** compute electroweak channels to evaluate "
                "the top 5 one-variant-per-channel combinations by total absolute % ratio error."
            ),
            sizing_mode="stretch_width",
        )
        new_dirac_ew_best_combo_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_best_combo_ratio_pane = pn.pane.Markdown(
            (
                "**Mass Ratios (Best Global Combo):** compute electroweak channels to "
                "populate best-combination ratios."
            ),
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
            pd.DataFrame({
                "channel": list(ELECTROWEAK_CHANNELS),
                "mass_ref_GeV": [
                    _format_ref_value(DEFAULT_ELECTROWEAK_REFS.get(name))
                    for name in ELECTROWEAK_CHANNELS
                ],
            }),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors={"mass_ref_GeV": "input"},
        )
        new_dirac_ew_filtered_summary = pn.pane.Markdown(
            "**Filtered-out candidates:** none.",
            sizing_mode="stretch_width",
        )
        new_dirac_ew_filtered_mass_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_symmetry_breaking_summary = pn.pane.Markdown(
            "**Symmetry Breaking:** _run electroweak analysis to populate._",
            sizing_mode="stretch_width",
        )
        new_dirac_ew_multiscale_summary = pn.pane.Markdown(
            "### SU(2) Multiscale Summary\n_Multiscale kernels disabled._",
            sizing_mode="stretch_width",
        )
        new_dirac_ew_multiscale_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_multiscale_per_scale_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        new_dirac_ew_su2_gevp_widgets = create_gevp_dashboard_widgets()
        new_dirac_ew_u1_gevp_widgets = create_gevp_dashboard_widgets()
        new_dirac_ew_ew_mixed_gevp_widgets = create_gevp_dashboard_widgets()

        new_dirac_ew_dirac_full = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
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
            pd.DataFrame({
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
            }),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
            selectable=False,
            configuration={"editable": True},
            editors={"observed_GeV": "input"},
        )

        # =====================================================================
        # Coupling diagnostics tab components (quick, mass-free)
        # =====================================================================
        coupling_diagnostics_settings = CouplingDiagnosticsSettings()
        coupling_diagnostics_status = pn.pane.Markdown(
            "**Coupling Diagnostics:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )
        coupling_diagnostics_run_button = pn.widgets.Button(
            name="Compute Coupling Diagnostics",
            button_type="primary",
            min_width=260,
            sizing_mode="stretch_width",
            disabled=True,
        )
        coupling_diagnostics_settings_panel = pn.Param(
            coupling_diagnostics_settings,
            parameters=[
                "simulation_range",
                "h_eff",
                "mass",
                "ell0",
                "companion_topology",
                "pair_weighting",
                "color_dims_spec",
                "eps",
                "enable_kernel_diagnostics",
                "edge_weight_mode",
                "kernel_distance_method",
                "kernel_type",
                "n_scales",
                "kernel_scale_frames",
                "kernel_scale_q_low",
                "kernel_scale_q_high",
                "kernel_max_scale_samples",
                "kernel_min_scale",
                "kernel_assume_all_alive",
            ],
            show_name=False,
            widgets={
                "color_dims_spec": {"name": "Color dims (optional)"},
            },
            default_layout=type("CouplingDiagnosticsSettingsGrid", (pn.GridBox,), {"ncols": 2}),
        )
        coupling_diagnostics_summary = pn.pane.Markdown(
            "## Coupling Diagnostics Summary\n_Run diagnostics to populate._",
            sizing_mode="stretch_width",
        )
        coupling_diagnostics_regime_evidence = pn.pane.Markdown(
            "_Regime evidence will appear after running diagnostics._",
            sizing_mode="stretch_width",
        )
        coupling_diagnostics_summary_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        coupling_diagnostics_frame_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )
        coupling_diagnostics_scale_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        )
        coupling_diagnostics_phase_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        coupling_diagnostics_regime_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        coupling_diagnostics_fields_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        coupling_diagnostics_coverage_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        coupling_diagnostics_scale_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        coupling_diagnostics_running_plot = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        )
        coupling_diagnostics_phase_plot.object = _algorithm_placeholder_plot(
            "Run diagnostics to show phase trend."
        )
        coupling_diagnostics_regime_plot.object = _algorithm_placeholder_plot(
            "Run diagnostics to show regime metrics."
        )
        coupling_diagnostics_fields_plot.object = _algorithm_placeholder_plot(
            "Run diagnostics to show field means."
        )
        coupling_diagnostics_coverage_plot.object = _algorithm_placeholder_plot(
            "Run diagnostics to show pair/walker coverage."
        )
        coupling_diagnostics_scale_plot.object = _algorithm_placeholder_plot(
            "Run diagnostics to show kernel-scale diagnostics."
        )
        coupling_diagnostics_running_plot.object = _algorithm_placeholder_plot(
            "Run diagnostics to show running/Creutz diagnostics."
        )

        def set_history(
            history: RunHistory,
            history_path: Path | None = None,
            defer_dashboard_updates: bool = False,
        ) -> None:
            state["history"] = history
            state["history_path"] = history_path
            state["_multiscale_geodesic_distance_by_frame"] = None
            state["new_dirac_ew_comparison_overrides"] = None
            state["new_dirac_ew_ratio_specs"] = None
            if not defer_dashboard_updates:
                visualizer.bounds_extent = float(gas_config.bounds_extent)
                visualizer.set_history(history)
            algorithm_status.object = "**Algorithm ready:** click Run Algorithm Analysis."
            algorithm_run_button.disabled = False
            if defer_dashboard_updates:
                visualizer.status_pane.object = (
                    "**Simulation complete:** history captured; click a Compute button to "
                    "run post-processing."
                )
                save_button.disabled = False
                save_status.object = "**Save a history**: choose a path and click Save."
                fractal_set_run_button.disabled = False
                fractal_set_status.object = (
                    "**Holographic Principle ready:** click Compute Fractal Set."
                )
                einstein_run_button.disabled = False
                einstein_status.object = (
                    "**Einstein Test ready:** run Holographic Principle for G_N, then click."
                )
                anisotropic_edge_run_button.disabled = False
                anisotropic_edge_status.object = (
                    "**Anisotropic Edge Channels ready:** click Compute Anisotropic Edge Channels."
                )
                companion_strong_force_run_button.disabled = False
                companion_strong_force_display_plots_button.disabled = True
                companion_strong_force_display_plots_button.button_type = "default"
                state["companion_strong_force_plots_unlocked"] = False
                companion_strong_force_plot_gate_note.object = (
                    "_Plots are hidden. Click `Display Plots` after computing channels._"
                )
                companion_strong_force_status.object = "**Companion Strong Force ready:** click Compute Companion Strong Force Channels."
                tensor_calibration_run_button.disabled = False
                _tcw.status.object = (
                    "**Tensor Calibration ready:** click Compute Tensor Calibration."
                )
                new_dirac_ew_run_button.disabled = False
                new_dirac_ew_status.object = "**Electroweak ready:** click Compute Electroweak."
                new_dirac_run_button.disabled = False
                new_dirac_status.object = "**Dirac ready:** click Compute Dirac."
                coupling_diagnostics_run_button.disabled = False
                coupling_diagnostics_status.object = (
                    "**Coupling Diagnostics ready:** click Compute Coupling Diagnostics."
                )
                return
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
            algorithm_velocity_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show walker speed."
            )
            algorithm_geodesic_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show geodesic edge distances."
            )
            algorithm_rkv_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show riemannian kernel volume weights."
            )
            algorithm_lyapunov_plot.object = _algorithm_placeholder_plot(
                "Click Run Algorithm Analysis to show Lyapunov convergence."
            )
            save_button.disabled = False
            save_status.object = "**Save a history**: choose a path and click Save."
            # Enable fractal set tab
            fractal_set_run_button.disabled = False
            fractal_set_status.object = (
                "**Holographic Principle ready:** click Compute Fractal Set."
            )
            # Enable einstein test
            einstein_run_button.disabled = False
            einstein_status.object = (
                "**Einstein Test ready:** run Holographic Principle for G_N, then click."
            )
            # Enable anisotropic edge tab
            anisotropic_edge_run_button.disabled = False
            anisotropic_edge_status.object = (
                "**Anisotropic Edge Channels ready:** click Compute Anisotropic Edge Channels."
            )
            companion_strong_force_run_button.disabled = False
            companion_strong_force_display_plots_button.disabled = True
            companion_strong_force_display_plots_button.button_type = "default"
            state["companion_strong_force_plots_unlocked"] = False
            companion_strong_force_plot_gate_note.object = (
                "_Plots are hidden. Click `Display Plots` after computing channels._"
            )
            companion_strong_force_plots_spectrum.object = None
            companion_strong_force_multiscale_plot.object = None
            companion_strong_force_plateau_plots.objects = [
                pn.pane.Markdown(
                    "_Plots are hidden. Click `Display Plots` to render channel plots._",
                    sizing_mode="stretch_width",
                )
            ]
            companion_strong_force_heatmap_plots.objects = [
                pn.pane.Markdown(
                    "_Plots are hidden. Click `Display Plots` to render heatmaps._",
                    sizing_mode="stretch_width",
                )
            ]
            companion_strong_force_status.object = (
                "**Companion Strong Force ready:** click Compute Companion Strong Force Channels."
            )
            tensor_calibration_run_button.disabled = False
            clear_tensor_gevp_calibration_tab(
                _tcw,
                "**Tensor Calibration ready:** click Compute Tensor Calibration.",
                state=state,
            )
            new_dirac_ew_run_button.disabled = False
            new_dirac_ew_status.object = "**Electroweak ready:** click Compute Electroweak."
            new_dirac_run_button.disabled = False
            new_dirac_status.object = "**Dirac ready:** click Compute Dirac."
            new_dirac_ew_summary.object = "## Electroweak Summary\n_Run analysis to populate._"
            new_dirac_ew_multiscale_summary.object = "### SU(2) Multiscale Summary\n_Multiscale kernels disabled (original estimators only)._"
            clear_gevp_dashboard(new_dirac_ew_su2_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_u1_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_ew_mixed_gevp_widgets)
            new_dirac_ew_filtered_summary.object = "**Filtered-out candidates:** none."
            new_dirac_ew_mass_table.value = pd.DataFrame()
            new_dirac_ew_filtered_mass_table.value = pd.DataFrame()
            new_dirac_ew_ratio_table.value = pd.DataFrame()
            new_dirac_ew_fit_table.value = pd.DataFrame()
            new_dirac_ew_compare_table.value = pd.DataFrame()
            new_dirac_ew_anchor_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
            coupling_diagnostics_run_button.disabled = False
            coupling_diagnostics_status.object = (
                "**Coupling Diagnostics ready:** click Compute Coupling Diagnostics."
            )
            state["coupling_diagnostics_output"] = None
            coupling_diagnostics_summary.object = (
                "## Coupling Diagnostics Summary\n_Run diagnostics to populate._"
            )
            coupling_diagnostics_regime_evidence.object = (
                "_Regime evidence will appear after running diagnostics._"
            )
            coupling_diagnostics_summary_table.value = pd.DataFrame()
            coupling_diagnostics_frame_table.value = pd.DataFrame()
            coupling_diagnostics_scale_table.value = pd.DataFrame()
            coupling_diagnostics_phase_plot.object = _algorithm_placeholder_plot(
                "Run diagnostics to show phase trend."
            )
            coupling_diagnostics_regime_plot.object = _algorithm_placeholder_plot(
                "Run diagnostics to show regime metrics."
            )
            coupling_diagnostics_fields_plot.object = _algorithm_placeholder_plot(
                "Run diagnostics to show field means."
            )
            coupling_diagnostics_coverage_plot.object = _algorithm_placeholder_plot(
                "Run diagnostics to show pair/walker coverage."
            )
            coupling_diagnostics_scale_plot.object = _algorithm_placeholder_plot(
                "Run diagnostics to show kernel-scale diagnostics."
            )
            coupling_diagnostics_running_plot.object = _algorithm_placeholder_plot(
                "Run diagnostics to show running/Creutz diagnostics."
            )

        def on_simulation_complete(history: RunHistory):
            set_history(history, defer_dashboard_updates=True)

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
                    gas_config.bounds_extent = float(inferred_extent)
                set_history(history, history_path, defer_dashboard_updates=True)
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
                algorithm_velocity_plot.object = fallback
                algorithm_geodesic_plot.object = fallback
                algorithm_rkv_plot.object = fallback
                algorithm_lyapunov_plot.object = fallback
                return

            algorithm_clone_plot.object = diagnostics["clone_plot"]
            algorithm_fitness_plot.object = diagnostics["fitness_plot"]
            algorithm_reward_plot.object = diagnostics["reward_plot"]
            algorithm_companion_plot.object = diagnostics["companion_plot"]
            algorithm_interwalker_plot.object = diagnostics["interwalker_plot"]
            algorithm_velocity_plot.object = diagnostics["velocity_plot"]
            algorithm_geodesic_plot.object = diagnostics["geodesic_plot"]
            algorithm_rkv_plot.object = diagnostics["rkv_plot"]
            algorithm_lyapunov_plot.object = diagnostics["lyapunov_plot"]
            algorithm_status.object = (
                "**Algorithm diagnostics updated:** "
                f"{diagnostics['n_transition_steps']} transition frames, "
                f"{diagnostics['n_lyapunov_steps']} Lyapunov frames."
            )

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
                fractal_set_frame_table.value = frame_df.sort_values([
                    "recorded_step",
                    "partition_family",
                    "cut_type",
                ]).reset_index(drop=True)
                fractal_set_points_table.value = points_df.sort_values([
                    "recorded_step",
                    "partition_family",
                    "cut_type",
                    "cut_value",
                ]).reset_index(drop=True)

                fractal_set_summary.object = _format_fractal_set_summary(
                    points_df,
                    regression_df,
                    frame_df,
                )
                show_geom = bool(fractal_set_settings.use_geometry_correction) and (
                    fractal_set_settings.metric_display in {"geometry", "both"}
                )
                show_raw = fractal_set_settings.metric_display in {"raw", "both"} or not bool(
                    fractal_set_settings.use_geometry_correction
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

        def _update_anisotropic_edge_multiscale_views(
            output: MultiscaleStrongForceOutput | None,
            error: str | None = None,
            original_results: dict[str, Any] | None = None,
        ) -> None:
            # --- Error / None guard (touches both strong-force & multiscale tabs) ---
            if error:
                anisotropic_edge_multiscale_summary.object = (
                    "### Multiscale Kernel Summary\n" f"- Status: failed\n" f"- Error: `{error}`"
                )
                anisotropic_edge_multiscale_table.value = pd.DataFrame()
                anisotropic_edge_multiscale_plot.object = None
                clear_multiscale_tab(
                    _msw,
                    "## Multiscale\n" f"**Status:** failed. `{error}`",
                )
                return

            if output is None:
                anisotropic_edge_multiscale_summary.object = (
                    "### Multiscale Kernel Summary\n" "_Multiscale kernels disabled._"
                )
                anisotropic_edge_multiscale_table.value = pd.DataFrame()
                anisotropic_edge_multiscale_plot.object = None
                clear_multiscale_tab(
                    _msw,
                    "## Multiscale\n"
                    "_Run Companion Strong Force with **Enable multiscale kernels** to populate plots._",
                )
                return

            # --- Part A: Strong-force tab summary (stays in closure) ---
            scale_values = (
                output.scales.detach().cpu().numpy() if output.scales.numel() else np.array([])
            )
            lines = [
                "### Multiscale Kernel Summary",
                f"- Scales: `{len(scale_values)}`",
                f"- Frames: `{len(output.frame_indices)}`",
                f"- Bootstrap mode: `{output.bootstrap_mode_applied}`",
            ]
            if scale_values.size > 0:
                lines.append(
                    "- Scale range: "
                    f"`[{float(scale_values.min()):.4g}, {float(scale_values.max()):.4g}]`"
                )
            if output.notes:
                for note in output.notes:
                    lines.append(f"- Note: {note}")
            anisotropic_edge_multiscale_summary.object = "  \n".join(lines)

            rows: list[dict[str, Any]] = []
            curves: list[hv.Element] = []
            for channel, results_per_scale in output.per_scale_results.items():
                display_channel = _display_channel_name(str(channel))
                measurement_group = (
                    "companion" if str(channel).endswith("_companion") else "non_companion"
                )
                masses = [
                    float(res.mass_fit.get("mass", float("nan")))
                    if res is not None
                    else float("nan")
                    for res in results_per_scale
                ]
                best_idx = int(output.best_scale_index.get(channel, 0))
                best_mass = masses[best_idx] if 0 <= best_idx < len(masses) else float("nan")
                best_scale = (
                    float(scale_values[best_idx])
                    if scale_values.size > best_idx >= 0
                    else float("nan")
                )
                best_err = float("nan")
                if channel in output.best_results:
                    best_err = float(
                        output.best_results[channel].mass_fit.get("mass_error", float("nan"))
                    )
                rows.append({
                    "channel": display_channel,
                    "source_channel": display_channel,
                    "measurement_group": measurement_group,
                    "best_scale_idx": best_idx,
                    "best_scale": best_scale,
                    "mass": best_mass,
                    "mass_error": best_err,
                })
                if scale_values.size > 0 and len(masses) == len(scale_values):
                    y = np.asarray(masses, dtype=float)
                    if np.isfinite(y).any():
                        color = CHANNEL_COLORS.get(_channel_color_key(str(channel)), None)
                        curve = hv.Curve(
                            (scale_values, y),
                            kdims=["scale"],
                            vdims=["mass"],
                            label=display_channel,
                        )
                        if color is not None:
                            curve = curve.opts(color=color)
                        curves.append(curve)

            anisotropic_edge_multiscale_table.value = (
                pd.DataFrame(rows).sort_values("channel") if rows else pd.DataFrame()
            )
            if curves:
                overlay = curves[0]
                for curve in curves[1:]:
                    overlay *= curve
                anisotropic_edge_multiscale_plot.object = overlay.opts(
                    width=900,
                    height=320,
                    title="Multiscale Mass Curves",
                    show_grid=True,
                    legend_position="right",
                )
            else:
                anisotropic_edge_multiscale_plot.object = None

            # --- Part B: Dedicated Multiscale tab (delegated to module) ---
            update_multiscale_tab(
                _msw,
                output,
                scale_values,
                state,
                history=state.get("history"),
                original_results=original_results,
                kernel_distance_method=str(anisotropic_edge_settings.kernel_distance_method),
                edge_weight_mode=str(anisotropic_edge_settings.edge_weight_mode),
                kernel_assume_all_alive=bool(anisotropic_edge_settings.kernel_assume_all_alive),
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

            tensor_payload = state.get("tensor_calibration_payload")
            use_cached_payload = (
                isinstance(tensor_payload, dict)
                and np.isfinite(float(tensor_payload.get("correction_scale", float("nan"))))
                and str(tensor_payload.get("mode", table_mode)) == str(table_mode)
            )
            if not use_cached_payload:
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
                calibration_family_map=ANISOTROPIC_EDGE_COMPANION_FAMILY_MAP,
                calibration_ratio_specs=ANISOTROPIC_EDGE_COMPANION_RATIO_SPECS,
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
                if apply_tensor_correction and np.isfinite(mass_value) and mass_value > 0:
                    return mass_value * correction_scale_display
                return mass_value

            def _display_mass_error(result_obj: ChannelCorrelatorResult) -> float:
                mass_err = _get_channel_mass_error(result_obj, base_mode)
                if apply_tensor_correction and np.isfinite(mass_err) and mass_err >= 0:
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
                momentum_length = _extract_axis_extent_from_bounds(history.bounds, momentum_axis)

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

                first_mode_name = min(momentum_channels.keys(), key=_mode_sort_key)
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
                    "mass_error": float(mass_err)
                    if np.isfinite(mass_err) and mass_err >= 0
                    else np.nan,
                    "r2": float(r2_value) if np.isfinite(r2_value) else np.nan,
                    "p": p_value,
                    "p2": p2_value,
                    "core_estimator": bool(is_core),
                }
                estimator_rows.append(row)
                if is_core:
                    core_rows.append(row)
                if n_mode is not None and np.isfinite(p2_value):
                    momentum_rows.append({
                        "estimator": estimator,
                        "channel": channel_name,
                        "n_mode": int(n_mode),
                        "p": float(p_value),
                        "p2": float(p2_value),
                        "mass": float(mass_value),
                        "mass_error": (
                            float(mass_err) if np.isfinite(mass_err) and mass_err >= 0 else np.nan
                        ),
                    })

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
                consensus_syst = float(np.std(core_mass, ddof=1)) if core_mass.size > 1 else 0.0

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
                table_df = table_df.sort_values(["approach_order", "n_mode_sort", "channel"]).drop(
                    columns=["approach_order", "n_mode_sort"], errors="ignore"
                )
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
                    pairwise_rows.append({
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
                    })
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
                summary_lines.append(f"- Estimator warnings: `{glueball_systematics_error}`")
            anisotropic_edge_glueball_approach_summary.object = "  \n".join(summary_lines)

            consensus_lines = ["**Glueball Consensus / Systematics:**"]
            if np.isfinite(consensus_mass) and consensus_mass > 0:
                line = f"- Consensus mass ({consensus_weighting}): `{consensus_mass:.6g}`"
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
                    vdims=[
                        ("mass", "Mass"),
                        ("estimator", "Estimator"),
                        ("mass_error", "Mass Error"),
                    ],
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
                not np.isfinite(tensor_momentum_length) or tensor_momentum_length <= 0
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
                        tensor_momentum_rows.append({
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
                        })
                    else:
                        tensor_component_rows.append({
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
                        })

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
                    tensor_consensus_mass = float(
                        np.sum(weights * weighted_mass) / np.sum(weights)
                    )
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
                table_df = table_df.sort_values([
                    "approach_order",
                    "n_mode_sort",
                    "component_sort",
                    "channel",
                ]).drop(
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
                    tensor_pairwise_rows.append({
                        "estimator_a": str(row_a["estimator"]),
                        "estimator_b": str(row_b["estimator"]),
                        "delta_pct": delta_pct,
                        "abs_delta_pct": (abs(delta_pct) if np.isfinite(delta_pct) else np.nan),
                        "pull_sigma": pull_sigma,
                    })

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
                    [float(row.get("abs_delta_pct", np.nan)) for row in tensor_pairwise_rows],
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
                tensor_summary_lines.append(
                    f"- Missing estimators: `{', '.join(missing_tensor_core)}`"
                )
            if tensor_momentum_rows:
                mode_count = len({int(row["n_mode"]) for row in tensor_momentum_rows})
                tensor_summary_lines.append(
                    f"- Contracted momentum modes available: `{mode_count}`"
                )
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
                glueball_lorentz_rows.append({
                    "estimator": estimator,
                    "mass": float(mass_value),
                    "mass_error": (
                        float(mass_err) if np.isfinite(mass_err) and mass_err >= 0 else np.nan
                    ),
                    "r2": float(r2_value) if np.isfinite(r2_value) else np.nan,
                })

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
                errors = np.asarray(
                    [row["mass_error"] for row in glueball_lorentz_rows], dtype=float
                )
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
                        entry += f", Δ vs consensus `{float(row['delta_vs_consensus_pct']):+.2f}%`"
                    glueball_lorentz_lines.append(entry)
            else:
                glueball_lorentz_lines.append("- No valid glueball estimators available.")

            if np.isfinite(consensus_mass_lorentz) and consensus_mass_lorentz > 0:
                consensus_line = f"- Consensus mass: `{consensus_mass_lorentz:.6g}`" + (
                    f" ± `{consensus_mass_lorentz_err:.2g}`"
                    if np.isfinite(consensus_mass_lorentz_err)
                    else ""
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
                    if np.isfinite(float(row["mass_error"])) and np.isfinite(
                        consensus_mass_lorentz_err
                    ):
                        rel_cons = consensus_mass_lorentz_err / max(consensus_mass_lorentz, 1e-12)
                        rel_row = float(row["mass_error"]) / max(mass_value, 1e-12)
                        corr_err = abs(correction_factor) * np.sqrt(
                            rel_cons * rel_cons + rel_row * rel_row
                        )
                        correction_line += f" ± `{corr_err:.2g}`"
                    correction_lines.append(correction_line)
            else:
                correction_lines.append("- n/a (consensus glueball mass unavailable).")

            if np.isfinite(dispersion_ceff) and dispersion_ceff > 0:
                correction_lines.append(f"- Dispersion-derived `c_eff`: `{dispersion_ceff:.6g}`")
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
                correction_lines.append(f"- Estimator warnings: `{glueball_systematics_error}`")

            tensor_payload = tensor_payload_display
            state["anisotropic_edge_tensor_correction_payload"] = tensor_payload
            tensor_consensus_mass = float(tensor_payload.get("consensus_mass", float("nan")))
            tensor_consensus_err = float(tensor_payload.get("consensus_err", float("nan")))
            tensor_spread = float(tensor_payload.get("spread", float("nan")))
            tensor_base_label = str(tensor_payload.get("base_label", "tensor"))
            tensor_base_mass = float(tensor_payload.get("base_mass", float("nan")))
            tensor_scale = float(tensor_payload.get("correction_scale", float("nan")))
            tensor_scale_err = float(tensor_payload.get("correction_scale_err", float("nan")))
            tensor_mode = str(tensor_payload.get("base_mode", _resolve_base_mass_mode(mode)))
            tensor_labels = np.asarray(tensor_payload.get("labels", []), dtype=object)
            tensor_valid = np.asarray(tensor_payload.get("valid_mask", []), dtype=bool)
            tensor_missing = [
                str(label)
                for label, is_valid in zip(
                    tensor_labels.tolist(), tensor_valid.tolist(), strict=False
                )
                if not bool(is_valid)
            ]
            correction_lines.append("- Tensor calibration (4-way consensus):")
            correction_lines.append(f"- Tensor calibration mode: `{tensor_mode}`")
            if np.isfinite(tensor_consensus_mass) and tensor_consensus_mass > 0:
                tensor_consensus_line = f"- Tensor consensus mass: `{tensor_consensus_mass:.6g}`"
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
                glueball_systematics_error=state.get(
                    "anisotropic_edge_glueball_systematics_error"
                ),
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
                tensor_systematics_error = (
                    "companion tensor momentum estimators moved to Companion Strong Force tab"
                )
                try:
                    strong_tensor_result = _compute_strong_tensor_for_anisotropic_edge(
                        history, anisotropic_edge_settings
                    )
                except Exception as exc:
                    tensor_systematics_error = f"strong-force tensor estimator failed: {exc}"
                edge_iso_glueball_result = None
                su3_glueball_result = None
                momentum_glueball_results: dict[str, ChannelCorrelatorResult] = {}
                glueball_systematics_error = (
                    "companion SU(3) glueball estimators moved to Companion Strong Force tab"
                )
                try:
                    edge_iso_glueball_result = _compute_anisotropic_edge_isotropic_glueball_result(
                        history,
                        anisotropic_edge_settings,
                    )
                except Exception as exc:
                    glueball_systematics_error = f"isotropic-edge glueball estimator failed: {exc}"
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
                multiscale_output: MultiscaleStrongForceOutput | None = None
                multiscale_error: str | None = None
                results = dict(output.channel_results)
                if bool(anisotropic_edge_settings.use_multiscale_kernels):
                    try:
                        multiscale_cfg = MultiscaleStrongForceConfig(
                            warmup_fraction=float(anisotropic_edge_settings.simulation_range[0]),
                            end_fraction=float(anisotropic_edge_settings.simulation_range[1]),
                            mc_time_index=anisotropic_edge_settings.mc_time_index,
                            h_eff=float(anisotropic_edge_settings.h_eff),
                            mass=float(anisotropic_edge_settings.mass),
                            ell0=anisotropic_edge_settings.ell0,
                            edge_weight_mode=str(anisotropic_edge_settings.edge_weight_mode),
                            n_scales=int(anisotropic_edge_settings.n_scales),
                            kernel_type=str(anisotropic_edge_settings.kernel_type),
                            kernel_distance_method=str(
                                anisotropic_edge_settings.kernel_distance_method
                            ),
                            kernel_assume_all_alive=bool(
                                anisotropic_edge_settings.kernel_assume_all_alive
                            ),
                            kernel_batch_size=int(anisotropic_edge_settings.kernel_batch_size),
                            kernel_scale_frames=int(anisotropic_edge_settings.kernel_scale_frames),
                            kernel_scale_q_low=float(anisotropic_edge_settings.kernel_scale_q_low),
                            kernel_scale_q_high=float(
                                anisotropic_edge_settings.kernel_scale_q_high
                            ),
                            max_lag=int(anisotropic_edge_settings.max_lag),
                            use_connected=bool(anisotropic_edge_settings.use_connected),
                            fit_mode=str(anisotropic_edge_settings.fit_mode),
                            fit_start=int(anisotropic_edge_settings.fit_start),
                            fit_stop=anisotropic_edge_settings.fit_stop,
                            min_fit_points=int(anisotropic_edge_settings.min_fit_points),
                            window_widths=_parse_window_widths(
                                anisotropic_edge_settings.window_widths_spec
                            ),
                            compute_bootstrap_errors=bool(
                                anisotropic_edge_settings.compute_bootstrap_errors
                            ),
                            n_bootstrap=int(anisotropic_edge_settings.n_bootstrap),
                            bootstrap_mode=str(anisotropic_edge_settings.kernel_bootstrap_mode),
                            walker_bootstrap_max_walkers=int(
                                anisotropic_edge_settings.kernel_bootstrap_max_walkers
                            ),
                        )
                        requested_multiscale_channels = [
                            c.strip()
                            for c in str(anisotropic_edge_settings.channel_list).split(",")
                            if c.strip()
                        ]
                        multiscale_output = compute_multiscale_strong_force_channels(
                            history,
                            config=multiscale_cfg,
                            channels=requested_multiscale_channels,
                        )
                        results.update(multiscale_output.best_results)
                    except Exception as exc:
                        multiscale_error = str(exc)
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
                state["anisotropic_edge_multiscale_output"] = multiscale_output
                state["anisotropic_edge_multiscale_error"] = multiscale_error
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
                    f"- Multiscale kernels: `{'on' if anisotropic_edge_settings.use_multiscale_kernels else 'off'}`\n"
                )
                n_channels = len([res for res in results.values() if res.n_samples > 0])
                if multiscale_error:
                    anisotropic_edge_status.object = (
                        f"**Complete with multiscale error:** {n_channels} channels computed. "
                        f"`{multiscale_error}`"
                    )
                else:
                    anisotropic_edge_status.object = (
                        f"**Complete:** {n_channels} anisotropic edge channels computed."
                    )

            _run_tab_computation(
                state,
                anisotropic_edge_status,
                "anisotropic edge channels",
                _compute,
            )

        def _base_channel_name(name: str) -> str:
            raw = str(name)
            if raw.endswith("_companion"):
                return raw[: -len("_companion")]
            if raw.startswith("spatial_"):
                return raw[len("spatial_") :]
            return raw

        def _display_channel_name(raw_name: str) -> str:
            raw = str(raw_name)
            base = _base_channel_name(raw)
            if raw.endswith("_companion"):
                return base
            spatial_name = f"spatial_{base}"
            spatial_canonical_label_map = {
                "spatial_nucleon_score_abs": "nucleon",
                "spatial_pseudoscalar_score_directed": "pseudoscalar",
            }
            return spatial_canonical_label_map.get(spatial_name, spatial_name)

        def _display_companion_channel_name(raw_name: str) -> str:
            """Companion tab display uses canonical channel names (no spatial prefix)."""
            return _base_channel_name(str(raw_name))

        def _channel_color_key(raw_name: str) -> str:
            return _base_channel_name(str(raw_name))

        def _comparison_variant_roots(base_name: str) -> list[str]:
            return [str(base_name)]

        def _variant_names_for_base_in_results(
            results: dict[str, ChannelCorrelatorResult],
            base_name: str,
        ) -> list[str]:
            variant_roots = _comparison_variant_roots(str(base_name))
            names = [
                str(name)
                for name, result in results.items()
                if isinstance(result, ChannelCorrelatorResult)
                and result.n_samples > 0
                and any(
                    str(name) == root or str(name).startswith(f"{root}_") for root in variant_roots
                )
            ]
            return sorted(
                names,
                key=lambda name: (
                    0 if name == base_name else 1 if not name.endswith("_multiscale_best") else 2,
                    name,
                ),
            )

        def _sync_companion_variant_selectors(
            results: dict[str, ChannelCorrelatorResult],
        ) -> None:
            companion_strong_force_variant_sync["active"] = True
            try:
                for base_name, selector in companion_strong_force_variant_selectors.items():
                    names = _variant_names_for_base_in_results(results, str(base_name))
                    if not names:
                        names = [str(base_name)]
                    selector.options = names
                    if str(selector.value) not in names:
                        selector.value = names[0]
            finally:
                companion_strong_force_variant_sync["active"] = False

        def _companion_comparison_overrides(
            results: dict[str, ChannelCorrelatorResult],
        ) -> dict[str, str]:
            overrides: dict[str, str] = {}
            for canonical_name, selector in companion_strong_force_variant_selectors.items():
                selected_name = str(selector.value).strip()
                if not selected_name:
                    continue
                if selected_name not in results:
                    continue
                overrides[str(canonical_name)] = selected_name
            return overrides

        def _hide_companion_strong_force_plots(message: str) -> None:
            companion_strong_force_plots_spectrum.object = None
            companion_strong_force_multiscale_plot.object = None
            companion_strong_force_plateau_plots.objects = [
                pn.pane.Markdown(
                    f"_{message}_",
                    sizing_mode="stretch_width",
                )
            ]
            companion_strong_force_heatmap_plots.objects = [
                pn.pane.Markdown(
                    f"_{message}_",
                    sizing_mode="stretch_width",
                )
            ]

        def _update_companion_strong_force_plots(
            results: dict[str, ChannelCorrelatorResult],
        ) -> None:
            if not bool(state.get("companion_strong_force_plots_unlocked", False)):
                _hide_companion_strong_force_plots(
                    "Plots are hidden. Click `Display Plots` to render."
                )
                companion_strong_force_plot_gate_note.object = (
                    "_Plots are hidden. Click `Display Plots` to render them._"
                )
                return
            companion_strong_force_plot_gate_note.object = (
                "_Plots are visible for the current computed results._"
            )
            _update_correlator_plots(
                results,
                companion_strong_force_plateau_plots,
                companion_strong_force_plots_spectrum,
                None,
                None,
                heatmap_container=companion_strong_force_heatmap_plots,
                heatmap_color_metric_widget=companion_strong_force_heatmap_color_metric,
                heatmap_alpha_metric_widget=companion_strong_force_heatmap_alpha_metric,
                correlator_logy=False,
            )

        def _update_companion_strong_force_tables(
            results: dict[str, ChannelCorrelatorResult],
            mode: str | None = None,
        ) -> None:
            if mode is None:
                mode = companion_strong_force_mass_mode.value
            try:
                gevp_min_r2 = float(companion_strong_force_settings.gevp_min_operator_r2)
            except (TypeError, ValueError):
                gevp_min_r2 = 0.5
            try:
                gevp_min_windows = max(
                    0,
                    int(companion_strong_force_settings.gevp_min_operator_windows),
                )
            except (TypeError, ValueError):
                gevp_min_windows = 10
            try:
                gevp_max_error_pct = float(
                    companion_strong_force_settings.gevp_max_operator_error_pct
                )
            except (TypeError, ValueError):
                gevp_max_error_pct = 30.0
            gevp_remove_artifacts = bool(companion_strong_force_settings.gevp_remove_artifacts)

            filtered_results, filtered_out_results = _split_results_by_companion_gevp_filters(
                results,
                min_r2=gevp_min_r2,
                min_windows=gevp_min_windows,
                max_error_pct=gevp_max_error_pct,
                remove_artifacts=gevp_remove_artifacts,
                keep_gevp_results=True,
            )
            _sync_companion_variant_selectors(filtered_results)
            comparison_overrides = _companion_comparison_overrides(filtered_results)
            _update_strong_tables(
                filtered_results,
                str(mode),
                companion_strong_force_mass_table,
                companion_strong_force_ratio_pane,
                companion_strong_force_fit_table,
                anchor_table=None,
                glueball_ref_input=companion_strong_force_glueball_ref_input,
                ratio_specs=STRONG_FORCE_RATIO_SPECS,
                comparison_channel_overrides=comparison_overrides,
            )
            filtered_out_dict = {
                name: payload[0] for name, payload in filtered_out_results.items()
            }
            _update_mass_table(
                filtered_out_dict,
                companion_strong_force_filtered_mass_table,
                str(mode),
            )
            min_r2_str = f"{gevp_min_r2:.3g}" if np.isfinite(gevp_min_r2) else "off"
            max_error_pct_str = (
                f"{gevp_max_error_pct:.3g}%" if np.isfinite(gevp_max_error_pct) else "off"
            )
            if filtered_out_results:
                preview = ", ".join(
                    (f"{_display_companion_channel_name(name)}" f"({reason})")
                    for name, (_, reason) in list(filtered_out_results.items())[:6]
                )
                suffix = " ..." if len(filtered_out_results) > 6 else ""
                companion_strong_force_filtered_summary.object = (
                    f"**Filtered-out candidates:** `{len(filtered_out_results)}` excluded; "
                    f"`{len(filtered_results)}` shown.  \n"
                    f"_Filters:_ `R² >= {min_r2_str}`, `n_windows >= {gevp_min_windows}`, "
                    f"`error % <= {max_error_pct_str}`, "
                    f"`remove artifacts={'on' if gevp_remove_artifacts else 'off'}`.  \n"
                    f"_Preview:_ `{preview}{suffix}`"
                )
            else:
                companion_strong_force_filtered_summary.object = (
                    "**Filtered-out candidates:** none.  \n"
                    f"_Filters:_ `R² >= {min_r2_str}`, `n_windows >= {gevp_min_windows}`, "
                    f"`error % <= {max_error_pct_str}`, "
                    f"`remove artifacts={'on' if gevp_remove_artifacts else 'off'}`."
                )

            def _variant_names_for_base(base_name: str) -> list[str]:
                return _variant_names_for_base_in_results(filtered_results, base_name)

            def _ratio_reference_value(
                numerator_base: str,
                denominator_base: str,
                reference_label: str,
            ) -> float:
                pair_key = (str(numerator_base), str(denominator_base))
                if pair_key in STRONG_FORCE_RATIO_REFERENCE_VALUES:
                    return float(STRONG_FORCE_RATIO_REFERENCE_VALUES[pair_key])
                match = re.search(r"≈\s*([0-9]+(?:\.[0-9]+)?)", str(reference_label))
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return float("nan")
                return float("nan")

            companion_reference_masses: dict[str, float] = {
                "pseudoscalar": float(MESON_REFS["pion"]),
                "nucleon": float(BARYON_REFS["proton"]),
                "vector": float(MESON_REFS["rho"]),
                "scalar": 0.500,
            }
            if companion_strong_force_glueball_ref_input.value is not None:
                try:
                    glueball_ref_value = float(companion_strong_force_glueball_ref_input.value)
                except (TypeError, ValueError):
                    glueball_ref_value = float("nan")
                if np.isfinite(glueball_ref_value) and glueball_ref_value > 0:
                    companion_reference_masses["glueball"] = glueball_ref_value

            canonical_channel_order = (
                "pseudoscalar",
                "nucleon",
                "vector",
                "scalar",
                "axial_vector",
                "glueball",
                "tensor_traceless",
                "tensor",
            )

            def _canonical_companion_channel(variant_name: str) -> str:
                base = _base_channel_name(str(variant_name))
                for canonical in canonical_channel_order:
                    if base == canonical or base.startswith(f"{canonical}_"):
                        return canonical
                return base

            ratio_mass_entries: list[dict[str, Any]] = []
            for variant_name, variant_result in filtered_results.items():
                if not isinstance(variant_result, ChannelCorrelatorResult):
                    continue
                if variant_result.n_samples == 0:
                    continue
                variant_mass = _get_channel_mass(variant_result, str(mode))
                if not np.isfinite(variant_mass) or variant_mass <= 0:
                    continue
                canonical_channel = _canonical_companion_channel(str(variant_name))
                reference_mass = companion_reference_masses.get(canonical_channel, float("nan"))
                ratio_mass_entries.append({
                    "variant_key": str(variant_name),
                    "variant_display": _display_companion_channel_name(str(variant_name)),
                    "canonical_channel": canonical_channel,
                    "mass": float(variant_mass),
                    "reference_mass": float(reference_mass)
                    if np.isfinite(reference_mass)
                    else float("nan"),
                })

            cross_ratio_rows: list[dict[str, Any]] = []
            for numerator_entry in ratio_mass_entries:
                for denominator_entry in ratio_mass_entries:
                    if numerator_entry["variant_key"] == denominator_entry["variant_key"]:
                        continue
                    if (
                        numerator_entry["canonical_channel"]
                        == denominator_entry["canonical_channel"]
                    ):
                        continue
                    ratio_value = float(numerator_entry["mass"] / denominator_entry["mass"])
                    measured_ratio = float("nan")
                    numerator_reference_mass = float(numerator_entry["reference_mass"])
                    denominator_reference_mass = float(denominator_entry["reference_mass"])
                    if (
                        np.isfinite(numerator_reference_mass)
                        and np.isfinite(denominator_reference_mass)
                        and denominator_reference_mass > 0
                    ):
                        measured_ratio = float(
                            numerator_reference_mass / denominator_reference_mass
                        )
                    delta_pct = float("nan")
                    if np.isfinite(measured_ratio) and measured_ratio > 0:
                        delta_pct = float((ratio_value / measured_ratio - 1.0) * 100.0)
                    cross_ratio_rows.append({
                        "numerator_variant": numerator_entry["variant_display"],
                        "numerator_channel": numerator_entry["canonical_channel"],
                        "numerator_mass": numerator_entry["mass"],
                        "denominator_variant": denominator_entry["variant_display"],
                        "denominator_channel": denominator_entry["canonical_channel"],
                        "denominator_mass": denominator_entry["mass"],
                        "ratio_alg": ratio_value,
                        "measured_ratio": measured_ratio,
                        "delta_pct": delta_pct,
                        "abs_delta_pct": abs(delta_pct) if np.isfinite(delta_pct) else np.nan,
                        "numerator_measured_mass": numerator_reference_mass,
                        "denominator_measured_mass": denominator_reference_mass,
                        "has_measured_ratio": bool(np.isfinite(measured_ratio)),
                    })

            if cross_ratio_rows:
                cross_ratio_df = pd.DataFrame(cross_ratio_rows).sort_values(
                    [
                        "has_measured_ratio",
                        "abs_delta_pct",
                        "numerator_variant",
                        "denominator_variant",
                    ],
                    ascending=[False, False, True, True],
                    kind="stable",
                )
                measured_mask = np.isfinite(
                    cross_ratio_df["measured_ratio"].to_numpy(dtype=float, copy=False)
                )
                n_measured_pairs = int(np.sum(measured_mask))
                measured_df = cross_ratio_df[measured_mask]
                if not measured_df.empty:
                    worst_row = measured_df.iloc[0]
                    companion_strong_force_cross_ratio_summary.object = (
                        f"**Cross-channel ratio debug:** `{len(cross_ratio_df)}` ordered "
                        f"cross-channel pairs; `{n_measured_pairs}` have measured targets.  \n"
                        f"_Largest measured mismatch:_ "
                        f"`{worst_row['numerator_variant']}/{worst_row['denominator_variant']}` = "
                        f"`{float(worst_row['ratio_alg']):.6g}` vs measured "
                        f"`{float(worst_row['measured_ratio']):.6g}` "
                        f"(`{float(worst_row['delta_pct']):+.2f}%`)."
                    )
                else:
                    companion_strong_force_cross_ratio_summary.object = (
                        f"**Cross-channel ratio debug:** `{len(cross_ratio_df)}` ordered "
                        "cross-channel pairs, but none has a measured-ratio target with the "
                        "current reference map."
                    )
                companion_strong_force_cross_ratio_table.value = cross_ratio_df.drop(
                    columns=["has_measured_ratio"],
                    errors="ignore",
                )
            else:
                companion_strong_force_cross_ratio_summary.object = (
                    "**Cross-channel ratio debug:** no valid cross-channel pairs from the "
                    "currently filtered companion results."
                )
                companion_strong_force_cross_ratio_table.value = pd.DataFrame()

            ratio_specs_with_targets: list[tuple[str, str, float]] = []
            for numerator_base, denominator_base, reference_label in STRONG_FORCE_RATIO_SPECS:
                reference_value = _ratio_reference_value(
                    str(numerator_base),
                    str(denominator_base),
                    str(reference_label),
                )
                if np.isfinite(reference_value) and reference_value > 0:
                    ratio_specs_with_targets.append((
                        str(numerator_base),
                        str(denominator_base),
                        float(reference_value),
                    ))

            combo_channels: list[str] = []
            for numerator_base, denominator_base, _reference_value in ratio_specs_with_targets:
                if numerator_base not in combo_channels:
                    combo_channels.append(numerator_base)
                if denominator_base not in combo_channels:
                    combo_channels.append(denominator_base)

            variant_options_by_channel: dict[str, list[str]] = {
                channel: _variant_names_for_base(channel) for channel in combo_channels
            }
            missing_combo_channels = [
                channel for channel, options in variant_options_by_channel.items() if not options
            ]
            variant_mass_lookup: dict[str, float] = {}
            for variant_name, variant_result in filtered_results.items():
                if not isinstance(variant_result, ChannelCorrelatorResult):
                    continue
                if variant_result.n_samples == 0:
                    continue
                variant_mass = _get_channel_mass(variant_result, str(mode))
                if np.isfinite(variant_mass) and variant_mass > 0:
                    variant_mass_lookup[str(variant_name)] = float(variant_mass)

            combo_rows: list[dict[str, Any]] = []
            combo_truncated = False
            combos_considered = 0
            total_combinations = 0
            max_combo_evaluations = 100000
            evaluation_options = {
                channel: list(options) for channel, options in variant_options_by_channel.items()
            }
            if combo_channels and not missing_combo_channels:
                total_combinations = int(
                    np.prod(
                        np.asarray(
                            [len(evaluation_options[channel]) for channel in combo_channels],
                            dtype=np.int64,
                        )
                    )
                )
                if total_combinations > max_combo_evaluations:
                    per_channel_cap = max(
                        1,
                        int(np.floor(max_combo_evaluations ** (1.0 / float(len(combo_channels))))),
                    )
                    for channel in combo_channels:
                        if len(evaluation_options[channel]) > per_channel_cap:
                            evaluation_options[channel] = evaluation_options[channel][
                                :per_channel_cap
                            ]
                            combo_truncated = True
                    total_combinations = int(
                        np.prod(
                            np.asarray(
                                [len(evaluation_options[channel]) for channel in combo_channels],
                                dtype=np.int64,
                            )
                        )
                    )

                option_lists = [evaluation_options[channel] for channel in combo_channels]
                for selected_variants in itertools.product(*option_lists):
                    combos_considered += 1
                    selection = {
                        channel: str(variant_name)
                        for channel, variant_name in zip(combo_channels, selected_variants)
                    }
                    per_channel_error = dict.fromkeys(combo_channels, 0.0)
                    per_ratio_error: dict[str, float] = {}
                    total_abs_pct_error = 0.0
                    used_ratio_count = 0
                    valid_combo = True
                    for (
                        numerator_base,
                        denominator_base,
                        reference_value,
                    ) in ratio_specs_with_targets:
                        numerator_variant = selection.get(numerator_base)
                        denominator_variant = selection.get(denominator_base)
                        if numerator_variant is None or denominator_variant is None:
                            valid_combo = False
                            break
                        numerator_mass = variant_mass_lookup.get(numerator_variant, float("nan"))
                        denominator_mass = variant_mass_lookup.get(
                            denominator_variant, float("nan")
                        )
                        if (
                            not np.isfinite(numerator_mass)
                            or not np.isfinite(denominator_mass)
                            or denominator_mass <= 0
                        ):
                            valid_combo = False
                            break
                        ratio_value = float(numerator_mass / denominator_mass)
                        abs_pct_error = float(abs((ratio_value / reference_value - 1.0) * 100.0))
                        per_ratio_error[
                            f"error_{numerator_base}_over_{denominator_base}_abs_pct"
                        ] = abs_pct_error
                        per_channel_error[numerator_base] += abs_pct_error
                        per_channel_error[denominator_base] += abs_pct_error
                        total_abs_pct_error += abs_pct_error
                        used_ratio_count += 1

                    if not valid_combo or used_ratio_count != len(ratio_specs_with_targets):
                        continue

                    combo_row: dict[str, Any] = {
                        "total_abs_pct_error": total_abs_pct_error,
                        "mean_abs_pct_error": total_abs_pct_error / float(used_ratio_count),
                        "n_ratio_targets": int(used_ratio_count),
                    }
                    for channel in combo_channels:
                        combo_row[f"variant_{channel}"] = selection[channel]
                        combo_row[f"error_{channel}_abs_pct"] = float(per_channel_error[channel])
                    combo_row.update(per_ratio_error)
                    combo_rows.append(combo_row)

            if combo_rows:
                combo_df = pd.DataFrame(combo_rows).sort_values(
                    ["total_abs_pct_error", "mean_abs_pct_error"],
                    ascending=[True, True],
                    kind="stable",
                )
                top_combo_df = combo_df.head(5).reset_index(drop=True)
                companion_strong_force_best_combo_table.value = top_combo_df
                best_row = top_combo_df.iloc[0]
                best_selection = {
                    channel: str(best_row[f"variant_{channel}"])
                    for channel in combo_channels
                    if f"variant_{channel}" in best_row
                }
                truncation_note = (
                    " Evaluated a capped subset of combinations." if combo_truncated else ""
                )
                companion_strong_force_best_combo_summary.object = (
                    f"**Best global variant combo (top 5):** "
                    f"`{len(combo_df)}` valid combinations ranked by summed absolute `%` error "
                    f"across `{len(ratio_specs_with_targets)}` ratio targets.  \n"
                    f"_Best total error:_ `{float(best_row['total_abs_pct_error']):.4g}%` "
                    f"(mean `{float(best_row['mean_abs_pct_error']):.4g}%`).  \n"
                    f"_Combinations considered:_ `{combos_considered}`.{truncation_note}"
                )
                best_ratio_lines = ["**Mass Ratios (Best Global Combo):**"]
                for numerator_base, denominator_base, reference_label in STRONG_FORCE_RATIO_SPECS:
                    numerator_variant = best_selection.get(str(numerator_base))
                    denominator_variant = best_selection.get(str(denominator_base))
                    if not numerator_variant or not denominator_variant:
                        continue
                    numerator_mass = variant_mass_lookup.get(numerator_variant, float("nan"))
                    denominator_mass = variant_mass_lookup.get(denominator_variant, float("nan"))
                    if (
                        not np.isfinite(numerator_mass)
                        or not np.isfinite(denominator_mass)
                        or denominator_mass <= 0
                    ):
                        continue
                    ratio_value = float(numerator_mass / denominator_mass)
                    ref_value = _ratio_reference_value(
                        str(numerator_base),
                        str(denominator_base),
                        str(reference_label),
                    )
                    abs_pct_error = float("nan")
                    if np.isfinite(ref_value) and ref_value > 0:
                        abs_pct_error = float(abs((ratio_value / ref_value - 1.0) * 100.0))
                    if str(reference_label).strip():
                        annotation = str(reference_label).strip()
                    elif np.isfinite(ref_value) and ref_value > 0:
                        annotation = f"target ≈ {ref_value:.4g}"
                    else:
                        annotation = "no external target"
                    if np.isfinite(abs_pct_error):
                        annotation = f"{annotation}; abs error {abs_pct_error:.2f}%"
                    best_ratio_lines.append(
                        f"- {numerator_base}/{denominator_base}: **{ratio_value:.3f}** ({annotation})"
                    )
                if len(best_ratio_lines) == 1:
                    best_ratio_lines.append("- n/a (missing valid masses for selected variants)")
                companion_strong_force_ratio_pane.object = "  \n".join(best_ratio_lines)
            else:
                if not ratio_specs_with_targets:
                    reason = "no finite measured ratio targets are configured"
                elif missing_combo_channels:
                    reason = "missing variants for: " + ", ".join(
                        f"`{channel}`" for channel in missing_combo_channels
                    )
                elif total_combinations == 0:
                    reason = "no variant combinations available"
                else:
                    reason = (
                        "no complete variant combination has valid masses for all ratio targets"
                    )
                companion_strong_force_best_combo_summary.object = (
                    f"**Best global variant combo:** unavailable ({reason})."
                )
                companion_strong_force_best_combo_table.value = pd.DataFrame()
                companion_strong_force_ratio_pane.object = (
                    f"**Mass Ratios (Best Global Combo):** unavailable ({reason})."
                )

            ratio_blocks: list[Any] = []
            for numerator_base, denominator_base, reference_label in STRONG_FORCE_RATIO_SPECS:
                numerator_variants = _variant_names_for_base(str(numerator_base))
                denominator_variants = _variant_names_for_base(str(denominator_base))
                if not numerator_variants or not denominator_variants:
                    continue

                ref_value = _ratio_reference_value(
                    str(numerator_base),
                    str(denominator_base),
                    str(reference_label),
                )
                rows: list[dict[str, Any]] = []
                for num_name in numerator_variants:
                    num_result = filtered_results.get(num_name)
                    if num_result is None:
                        continue
                    num_mass = _get_channel_mass(num_result, str(mode))
                    if not np.isfinite(num_mass) or num_mass <= 0:
                        continue
                    for den_name in denominator_variants:
                        den_result = filtered_results.get(den_name)
                        if den_result is None:
                            continue
                        den_mass = _get_channel_mass(den_result, str(mode))
                        if not np.isfinite(den_mass) or den_mass <= 0:
                            continue
                        ratio_value = float(num_mass / den_mass)
                        delta_pct = float("nan")
                        if np.isfinite(ref_value) and ref_value > 0:
                            delta_pct = (ratio_value / ref_value - 1.0) * 100.0
                        rows.append({
                            "numerator_variant": _display_companion_channel_name(num_name),
                            "denominator_variant": _display_companion_channel_name(den_name),
                            "ratio": ratio_value,
                            "reference": ref_value if np.isfinite(ref_value) else np.nan,
                            "delta_pct": delta_pct,
                        })

                if not rows:
                    continue

                ratio_df = pd.DataFrame(rows).sort_values(
                    ["numerator_variant", "denominator_variant"],
                    kind="stable",
                )
                if np.isfinite(ref_value):
                    header_ref = f"{ref_value:.4f}"
                else:
                    header_ref = "n/a"
                ratio_blocks.extend([
                    pn.pane.Markdown(
                        f"#### {numerator_base}/{denominator_base} variants"
                        f" (reference: `{header_ref}`; {reference_label or 'no external target'})",
                        sizing_mode="stretch_width",
                    ),
                    pn.widgets.Tabulator(
                        ratio_df,
                        pagination=None,
                        show_index=False,
                        sizing_mode="stretch_width",
                    ),
                ])

            if ratio_blocks:
                companion_strong_force_ratio_tables.objects = ratio_blocks
            else:
                companion_strong_force_ratio_tables.objects = [
                    pn.pane.Markdown(
                        "_No variant ratios available for the currently computed channels._",
                        sizing_mode="stretch_width",
                    )
                ]

        def _update_companion_strong_force_multiscale_views(
            output: MultiscaleStrongForceOutput | None,
            *,
            original_results: dict[str, ChannelCorrelatorResult] | None = None,
            error: str | None = None,
        ) -> None:
            if error:
                companion_strong_force_multiscale_summary.object = (
                    "### Multiscale Kernel Summary\n" "- Status: failed\n" f"- Error: `{error}`"
                )
                companion_strong_force_multiscale_table.value = pd.DataFrame()
                companion_strong_force_multiscale_per_scale_table.value = pd.DataFrame()
                companion_strong_force_multiscale_plot.object = None
                return

            if output is None:
                companion_strong_force_multiscale_summary.object = (
                    "### Multiscale Kernel Summary\n"
                    "_Multiscale kernels disabled (original companion estimators only)._"
                )
                companion_strong_force_multiscale_table.value = pd.DataFrame()
                companion_strong_force_multiscale_per_scale_table.value = pd.DataFrame()
                companion_strong_force_multiscale_plot.object = None
                return

            scale_values = (
                output.scales.detach().cpu().numpy() if output.scales.numel() else np.array([])
            )
            lines = [
                "### Multiscale Kernel Summary",
                f"- Scales: `{len(scale_values)}`",
                f"- Frames: `{len(output.frame_indices)}`",
                f"- Bootstrap mode: `{output.bootstrap_mode_applied}`",
            ]
            if scale_values.size > 0:
                lines.append(
                    "- Scale range: "
                    f"`[{float(scale_values.min()):.4g}, {float(scale_values.max()):.4g}]`"
                )
            if output.notes:
                for note in output.notes:
                    lines.append(f"- Note: {note}")
            companion_strong_force_multiscale_summary.object = "  \n".join(lines)

            mode = str(companion_strong_force_mass_mode.value)
            try:
                gevp_min_r2 = float(companion_strong_force_settings.gevp_min_operator_r2)
            except (TypeError, ValueError):
                gevp_min_r2 = 0.5
            try:
                gevp_min_windows = max(
                    0, int(companion_strong_force_settings.gevp_min_operator_windows)
                )
            except (TypeError, ValueError):
                gevp_min_windows = 10
            try:
                gevp_max_error_pct = float(
                    companion_strong_force_settings.gevp_max_operator_error_pct
                )
            except (TypeError, ValueError):
                gevp_max_error_pct = 30.0
            gevp_remove_artifacts = bool(companion_strong_force_settings.gevp_remove_artifacts)
            rows: list[dict[str, Any]] = []
            per_scale_rows: list[dict[str, Any]] = []
            curves: list[hv.Element] = []
            for channel_name, results_per_scale in output.per_scale_results.items():
                display_name = _display_companion_channel_name(str(channel_name))
                original_mass = float("nan")
                original_result = None
                if isinstance(original_results, dict):
                    base_name = _base_channel_name(str(channel_name))
                    if str(channel_name).endswith("_companion"):
                        candidate_keys = [display_name, base_name, str(channel_name)]
                    else:
                        candidate_keys = [
                            display_name,
                            f"spatial_{base_name}",
                            base_name,
                            str(channel_name),
                        ]
                    for key in candidate_keys:
                        maybe = original_results.get(key)
                        if maybe is not None:
                            original_result = maybe
                            break
                if (
                    original_result is not None
                    and _companion_gevp_filter_reason(
                        original_result,
                        min_r2=gevp_min_r2,
                        min_windows=gevp_min_windows,
                        max_error_pct=gevp_max_error_pct,
                        remove_artifacts=gevp_remove_artifacts,
                    )
                    is None
                ):
                    original_mass = float(_get_channel_mass(original_result, mode))
                else:
                    original_result = None

                valid_scale_idx: list[int] = []
                masses: list[float] = []
                for scale_idx, res in enumerate(results_per_scale):
                    if res is None:
                        masses.append(float("nan"))
                        continue
                    reason = _companion_gevp_filter_reason(
                        res,
                        min_r2=gevp_min_r2,
                        min_windows=gevp_min_windows,
                        max_error_pct=gevp_max_error_pct,
                        remove_artifacts=gevp_remove_artifacts,
                    )
                    if reason is None:
                        valid_scale_idx.append(int(scale_idx))
                        masses.append(float(_get_channel_mass(res, mode)))
                    else:
                        masses.append(float("nan"))
                if not valid_scale_idx and original_result is None:
                    continue

                best_idx_from_output = int(output.best_scale_index.get(channel_name, -1))
                if best_idx_from_output in valid_scale_idx:
                    best_idx = best_idx_from_output
                elif valid_scale_idx:
                    best_idx = int(valid_scale_idx[0])
                else:
                    best_idx = -1
                best_mass = masses[best_idx] if 0 <= best_idx < len(masses) else float("nan")
                best_scale = (
                    float(scale_values[best_idx])
                    if scale_values.size > best_idx >= 0
                    else float("nan")
                )
                best_err = float("nan")
                best_r2 = float("nan")
                if channel_name in output.best_results:
                    best_result = output.best_results[channel_name]
                    best_reason = _companion_gevp_filter_reason(
                        best_result,
                        min_r2=gevp_min_r2,
                        min_windows=gevp_min_windows,
                        max_error_pct=gevp_max_error_pct,
                        remove_artifacts=gevp_remove_artifacts,
                    )
                    if best_reason is None:
                        best_err = float(_get_channel_mass_error(best_result, mode))
                        best_r2 = float(_get_channel_r2(best_result, mode))
                best_err_pct = (
                    abs(best_err / best_mass) * 100.0
                    if np.isfinite(best_mass)
                    and best_mass > 0
                    and np.isfinite(best_err)
                    and best_err >= 0
                    else float("nan")
                )
                rows.append({
                    "channel": display_name,
                    "source_channel": display_name,
                    "original_mass": original_mass,
                    "full_scale_mass": original_mass,
                    "best_scale_idx": best_idx,
                    "best_scale": best_scale,
                    "mass": best_mass,
                    "delta_vs_original_pct": (
                        ((best_mass - original_mass) / original_mass) * 100.0
                        if np.isfinite(original_mass)
                        and original_mass > 0
                        and np.isfinite(best_mass)
                        else float("nan")
                    ),
                    "mass_error": best_err,
                    "mass_error_pct": best_err_pct,
                    "r2": best_r2,
                })
                if scale_values.size > 0 and len(masses) == len(scale_values):
                    y = np.asarray(masses, dtype=float)
                    scale_fit = list(zip(scale_values.tolist(), results_per_scale, strict=False))
                    for scale_idx, (scale_value, result_obj) in enumerate(scale_fit):
                        if result_obj is None:
                            continue
                        filter_reason = _companion_gevp_filter_reason(
                            result_obj,
                            min_r2=gevp_min_r2,
                            min_windows=gevp_min_windows,
                            max_error_pct=gevp_max_error_pct,
                            remove_artifacts=gevp_remove_artifacts,
                        )
                        if filter_reason is not None:
                            continue
                        mass_value = (
                            float(_get_channel_mass(result_obj, mode))
                            if result_obj is not None
                            else float("nan")
                        )
                        mass_error = (
                            float(_get_channel_mass_error(result_obj, mode))
                            if result_obj is not None
                            else float("nan")
                        )
                        mass_error_pct = (
                            abs(mass_error / mass_value) * 100.0
                            if np.isfinite(mass_value)
                            and mass_value > 0
                            and np.isfinite(mass_error)
                            and mass_error >= 0
                            else float("nan")
                        )
                        r2_value = (
                            float(_get_channel_r2(result_obj, mode))
                            if result_obj is not None
                            else float("nan")
                        )
                        per_scale_rows.append({
                            "channel": display_name,
                            "source_channel": display_name,
                            "scale_label": f"s{int(scale_idx)}",
                            "scale_idx": int(scale_idx),
                            "scale": float(scale_value),
                            "mass": mass_value,
                            "mass_error": mass_error,
                            "mass_error_pct": mass_error_pct,
                            "r2": r2_value,
                            "is_best": bool(scale_idx == best_idx),
                            "is_full_scale": False,
                            "delta_vs_original_pct": (
                                ((mass_value - original_mass) / original_mass) * 100.0
                                if np.isfinite(original_mass)
                                and original_mass > 0
                                and np.isfinite(mass_value)
                                else float("nan")
                            ),
                        })
                    if original_result is not None:
                        full_mass = float(_get_channel_mass(original_result, mode))
                        full_err = float(_get_channel_mass_error(original_result, mode))
                        full_err_pct = (
                            abs(full_err / full_mass) * 100.0
                            if np.isfinite(full_mass)
                            and full_mass > 0
                            and np.isfinite(full_err)
                            and full_err >= 0
                            else float("nan")
                        )
                        full_r2 = float(_get_channel_r2(original_result, mode))
                        per_scale_rows.append({
                            "channel": display_name,
                            "source_channel": display_name,
                            "scale_label": "full_original_no_threshold",
                            "scale_idx": -1,
                            "scale": float("nan"),
                            "mass": full_mass,
                            "mass_error": full_err,
                            "mass_error_pct": full_err_pct,
                            "r2": full_r2,
                            "is_best": False,
                            "is_full_scale": True,
                            "delta_vs_original_pct": 0.0,
                        })
                    if np.isfinite(y).any():
                        color = CHANNEL_COLORS.get(_channel_color_key(str(channel_name)), None)
                        curve = hv.Curve(
                            (scale_values, y),
                            kdims=["scale"],
                            vdims=["mass"],
                            label=display_name,
                        )
                        if color is not None:
                            curve = curve.opts(color=color)
                        curves.append(curve)
                        if np.isfinite(original_mass) and original_mass > 0:
                            reference_curve = hv.Curve(
                                (
                                    np.asarray(
                                        [float(scale_values[0]), float(scale_values[-1])],
                                        dtype=float,
                                    ),
                                    np.asarray([original_mass, original_mass], dtype=float),
                                ),
                                kdims=["scale"],
                                vdims=["mass"],
                                label=f"{display_name} (original)",
                            ).opts(line_dash="dashed", line_width=1.5, alpha=0.65)
                            if color is not None:
                                reference_curve = reference_curve.opts(color=color)
                            curves.append(reference_curve)

            companion_strong_force_multiscale_table.value = (
                pd.DataFrame(rows).sort_values("channel") if rows else pd.DataFrame()
            )
            companion_strong_force_multiscale_per_scale_table.value = (
                pd.DataFrame(per_scale_rows).sort_values(
                    ["channel", "is_full_scale", "scale_idx"],
                    ascending=[True, True, True],
                )
                if per_scale_rows
                else pd.DataFrame()
            )
            if curves and bool(state.get("companion_strong_force_plots_unlocked", False)):
                overlay = curves[0]
                for curve in curves[1:]:
                    overlay *= curve
                companion_strong_force_multiscale_plot.object = overlay.opts(
                    width=900,
                    height=320,
                    title="Companion Multiscale Mass Curves (with original references)",
                    show_grid=True,
                    legend_position="right",
                )
            else:
                companion_strong_force_multiscale_plot.object = None

        def _on_companion_strong_force_mass_mode_change(event):
            if state.get("companion_strong_force_results") is None:
                return
            _update_companion_strong_force_tables(
                state["companion_strong_force_results"],
                mode=event.new,
            )
            _update_companion_strong_force_multiscale_views(
                state.get("companion_strong_force_multiscale_output"),
                original_results={
                    name: result
                    for name, result in state["companion_strong_force_results"].items()
                    if isinstance(result, ChannelCorrelatorResult)
                    and str(result.mass_fit.get("source", "original_companion"))
                    == "original_companion"
                },
                error=state.get("companion_strong_force_multiscale_error"),
            )

        companion_strong_force_mass_mode.param.watch(
            _on_companion_strong_force_mass_mode_change,
            "value",
        )

        def _on_companion_strong_force_variant_change(_event):
            if companion_strong_force_variant_sync["active"]:
                return
            if state.get("companion_strong_force_results") is None:
                return
            _update_companion_strong_force_tables(state["companion_strong_force_results"])

        for variant_selector in companion_strong_force_variant_selectors.values():
            variant_selector.param.watch(
                _on_companion_strong_force_variant_change,
                "value",
            )

        def _on_companion_strong_force_heatmap_metric_change(_event):
            if state.get("companion_strong_force_results") is None:
                return
            _update_companion_strong_force_plots(state["companion_strong_force_results"])

        companion_strong_force_heatmap_color_metric.param.watch(
            _on_companion_strong_force_heatmap_metric_change,
            "value",
        )
        companion_strong_force_heatmap_alpha_metric.param.watch(
            _on_companion_strong_force_heatmap_metric_change,
            "value",
        )

        def _on_companion_strong_force_display_plots_click(_event) -> None:
            results = state.get("companion_strong_force_results")
            if results is None:
                companion_strong_force_status.object = "**No companion results yet:** run Compute Companion Strong Force Channels first."
                return
            state["companion_strong_force_plots_unlocked"] = True
            companion_strong_force_display_plots_button.button_type = "success"
            _update_companion_strong_force_plots(results)
            _update_companion_strong_force_multiscale_views(
                state.get("companion_strong_force_multiscale_output"),
                original_results={
                    name: result
                    for name, result in results.items()
                    if isinstance(result, ChannelCorrelatorResult)
                    and str(result.mass_fit.get("source", "original_companion"))
                    == "original_companion"
                },
                error=state.get("companion_strong_force_multiscale_error"),
            )

        def on_run_companion_strong_force_channels(_):
            def _compute(history):
                requested_companion_channels = _collect_multiselect_values(
                    companion_strong_force_channel_family_selectors
                )
                if not requested_companion_channels:
                    requested_companion_channels = [
                        channel
                        for channels in DEFAULT_COMPANION_CHANNEL_VARIANT_SELECTION.values()
                        for channel in channels
                    ]
                output, resolved_h_eff, h_eff_desc = _compute_companion_strong_force_bundle(
                    history,
                    companion_strong_force_settings,
                    requested_channels=requested_companion_channels,
                )
                base_results = dict(output.channel_results)
                results = dict(base_results)
                for channel_name, channel_result in base_results.items():
                    if isinstance(channel_result.mass_fit, dict):
                        channel_result.mass_fit.setdefault("source", "original_companion")
                        channel_result.mass_fit.setdefault("base_channel", str(channel_name))

                multiscale_output: MultiscaleStrongForceOutput | None = None
                multiscale_error: str | None = None
                gevp_results: dict[str, ChannelCorrelatorResult] = {}
                gevp_errors: dict[str, str] = {}
                gevp_error: str | None = None
                if bool(companion_strong_force_settings.use_multiscale_kernels):
                    try:
                        requested_companion_multiscale_channels = sorted({
                            str(COMPANION_CHANNEL_MAP[ch])
                            for ch in requested_companion_channels
                            if ch in COMPANION_CHANNEL_MAP
                        })
                        if requested_companion_multiscale_channels:
                            multiscale_cfg = MultiscaleStrongForceConfig(
                                warmup_fraction=float(
                                    companion_strong_force_settings.simulation_range[0]
                                ),
                                end_fraction=float(
                                    companion_strong_force_settings.simulation_range[1]
                                ),
                                mc_time_index=companion_strong_force_settings.mc_time_index,
                                h_eff=resolved_h_eff,
                                mass=float(companion_strong_force_settings.mass),
                                ell0=companion_strong_force_settings.ell0,
                                edge_weight_mode=str(
                                    companion_strong_force_settings.edge_weight_mode
                                ),
                                n_scales=int(companion_strong_force_settings.n_scales),
                                kernel_type=str(companion_strong_force_settings.kernel_type),
                                kernel_distance_method=str(
                                    companion_strong_force_settings.kernel_distance_method
                                ),
                                kernel_assume_all_alive=bool(
                                    companion_strong_force_settings.kernel_assume_all_alive
                                ),
                                kernel_batch_size=int(
                                    companion_strong_force_settings.kernel_batch_size
                                ),
                                kernel_scale_frames=int(
                                    companion_strong_force_settings.kernel_scale_frames
                                ),
                                kernel_scale_q_low=float(
                                    companion_strong_force_settings.kernel_scale_q_low
                                ),
                                kernel_scale_q_high=float(
                                    companion_strong_force_settings.kernel_scale_q_high
                                ),
                                max_lag=int(companion_strong_force_settings.max_lag),
                                use_connected=bool(companion_strong_force_settings.use_connected),
                                fit_mode=str(companion_strong_force_settings.fit_mode),
                                fit_start=int(companion_strong_force_settings.fit_start),
                                fit_stop=companion_strong_force_settings.fit_stop,
                                min_fit_points=int(companion_strong_force_settings.min_fit_points),
                                window_widths=_parse_window_widths(
                                    companion_strong_force_settings.window_widths_spec
                                ),
                                best_min_r2=float(
                                    companion_strong_force_settings.gevp_min_operator_r2
                                ),
                                best_min_windows=int(
                                    companion_strong_force_settings.gevp_min_operator_windows
                                ),
                                best_max_error_pct=float(
                                    companion_strong_force_settings.gevp_max_operator_error_pct
                                ),
                                best_remove_artifacts=bool(
                                    companion_strong_force_settings.gevp_remove_artifacts
                                ),
                                compute_bootstrap_errors=bool(
                                    companion_strong_force_settings.compute_bootstrap_errors
                                ),
                                n_bootstrap=int(companion_strong_force_settings.n_bootstrap),
                                bootstrap_mode=str(
                                    companion_strong_force_settings.kernel_bootstrap_mode
                                ),
                                walker_bootstrap_max_walkers=int(
                                    companion_strong_force_settings.kernel_bootstrap_max_walkers
                                ),
                                companion_baryon_flux_exp_alpha=float(
                                    companion_strong_force_settings.baryon_flux_exp_alpha
                                ),
                            )
                            multiscale_output = compute_multiscale_strong_force_channels(
                                history,
                                config=multiscale_cfg,
                                channels=requested_companion_multiscale_channels,
                            )
                            for channel_name, result in multiscale_output.best_results.items():
                                display_name = _display_companion_channel_name(str(channel_name))
                                tagged_name = f"{display_name}_multiscale_best"
                                tagged_result = replace(result, channel_name=tagged_name)
                                if isinstance(tagged_result.mass_fit, dict):
                                    tagged_result.mass_fit["source"] = "multiscale_best"
                                    tagged_result.mass_fit["base_channel"] = display_name
                                results[tagged_name] = tagged_result
                    except Exception as exc:
                        multiscale_error = str(exc)

                if bool(companion_strong_force_settings.use_companion_nucleon_gevp):
                    channel_connected = {
                        "nucleon": bool(companion_strong_force_settings.baryon_use_connected),
                        "scalar": bool(companion_strong_force_settings.meson_use_connected),
                        "pseudoscalar": bool(companion_strong_force_settings.meson_use_connected),
                        "glueball": bool(companion_strong_force_settings.glueball_use_connected),
                    }
                    for gevp_base_channel in ("nucleon", "scalar", "pseudoscalar", "glueball"):
                        basis_channels = get_companion_gevp_basis_channels(gevp_base_channel)
                        has_channel_series = any(
                            (
                                ch in base_results
                                and base_results[ch] is not None
                                and int(base_results[ch].n_samples) > 0
                                and int(base_results[ch].series.numel()) > 0
                            )
                            for ch in basis_channels
                        )
                        if not has_channel_series:
                            continue
                        try:
                            gevp_cfg = GEVPConfig(
                                t0=int(companion_strong_force_settings.gevp_t0),
                                max_lag=int(companion_strong_force_settings.max_lag),
                                use_connected=bool(channel_connected[gevp_base_channel]),
                                fit_mode=str(companion_strong_force_settings.fit_mode),
                                fit_start=int(companion_strong_force_settings.fit_start),
                                fit_stop=companion_strong_force_settings.fit_stop,
                                min_fit_points=int(companion_strong_force_settings.min_fit_points),
                                window_widths=_parse_window_widths(
                                    companion_strong_force_settings.window_widths_spec
                                ),
                                basis_strategy=str(
                                    companion_strong_force_settings.gevp_basis_strategy
                                ),
                                max_basis=int(companion_strong_force_settings.gevp_max_basis),
                                min_operator_r2=float(
                                    companion_strong_force_settings.gevp_min_operator_r2
                                ),
                                min_operator_windows=int(
                                    companion_strong_force_settings.gevp_min_operator_windows
                                ),
                                max_operator_error_pct=float(
                                    companion_strong_force_settings.gevp_max_operator_error_pct
                                ),
                                remove_artifacts=bool(
                                    companion_strong_force_settings.gevp_remove_artifacts
                                ),
                                eig_rel_cutoff=float(
                                    companion_strong_force_settings.gevp_eig_rel_cutoff
                                ),
                                cond_limit=float(companion_strong_force_settings.gevp_cond_limit),
                                shrinkage=float(companion_strong_force_settings.gevp_shrinkage),
                                compute_bootstrap_errors=bool(
                                    companion_strong_force_settings.compute_bootstrap_errors
                                ),
                                n_bootstrap=int(companion_strong_force_settings.n_bootstrap),
                                bootstrap_mode=str(
                                    companion_strong_force_settings.gevp_bootstrap_mode
                                ),
                            )
                            gevp_payload = compute_companion_channel_gevp(
                                base_results=base_results,
                                multiscale_output=multiscale_output,
                                config=gevp_cfg,
                                base_channel=gevp_base_channel,
                            )
                            channel_gevp_result = gevp_payload.result
                            if isinstance(channel_gevp_result.mass_fit, dict):
                                channel_gevp_result.mass_fit.setdefault(
                                    "source",
                                    f"gevp_{gevp_base_channel}",
                                )
                                channel_gevp_result.mass_fit.setdefault(
                                    "base_channel",
                                    gevp_base_channel,
                                )
                            results[channel_gevp_result.channel_name] = channel_gevp_result
                            gevp_results[gevp_base_channel] = channel_gevp_result
                        except Exception as exc:
                            gevp_errors[gevp_base_channel] = str(exc)
                    if gevp_errors:
                        gevp_error = "; ".join(
                            f"{channel}: {message}" for channel, message in gevp_errors.items()
                        )

                state["companion_strong_force_results"] = results
                state["companion_strong_force_multiscale_output"] = multiscale_output
                state["companion_strong_force_multiscale_error"] = multiscale_error
                state["companion_strong_force_gevp_error"] = gevp_error
                state["_multiscale_gevp_min_operator_r2"] = float(
                    companion_strong_force_settings.gevp_min_operator_r2
                )
                state["_multiscale_gevp_min_operator_windows"] = int(
                    companion_strong_force_settings.gevp_min_operator_windows
                )
                state["_multiscale_gevp_max_operator_error_pct"] = float(
                    companion_strong_force_settings.gevp_max_operator_error_pct
                )
                state["_multiscale_gevp_remove_artifacts"] = bool(
                    companion_strong_force_settings.gevp_remove_artifacts
                )
                state["companion_strong_force_plots_unlocked"] = False
                companion_strong_force_display_plots_button.disabled = False
                companion_strong_force_display_plots_button.button_type = "primary"

                _update_companion_strong_force_plots(results)
                _update_companion_strong_force_tables(results)
                _update_companion_strong_force_multiscale_views(
                    multiscale_output,
                    original_results=base_results,
                    error=multiscale_error,
                )
                _update_anisotropic_edge_multiscale_views(
                    multiscale_output,
                    multiscale_error,
                    original_results=base_results,
                )

                summary_lines = [
                    "## Companion Strong Force Summary",
                    f"- Frames used: `{output.n_valid_frames}/{len(output.frame_indices)}`",
                    f"- Mean alive walkers/frame: `{output.avg_alive_walkers:.2f}`",
                    f"- h_eff: `{h_eff_desc}`",
                    (
                        "- Multiscale kernels: "
                        f"`{'on' if companion_strong_force_settings.use_multiscale_kernels else 'off'}`"
                    ),
                    f"- Channels computed: `{len([res for res in results.values() if res.n_samples > 0])}`",
                    (
                        "- Companion GEVP channels: "
                        f"`{'on' if companion_strong_force_settings.use_companion_nucleon_gevp else 'off'}`"
                    ),
                ]
                filters_reported = False
                for gevp_base_channel in ("nucleon", "scalar", "pseudoscalar", "glueball"):
                    channel_gevp_result = gevp_results.get(gevp_base_channel)
                    if channel_gevp_result is None or not isinstance(
                        channel_gevp_result.mass_fit, dict
                    ):
                        continue
                    n_input = int(channel_gevp_result.mass_fit.get("gevp_n_basis_input", 0))
                    n_kept = int(channel_gevp_result.mass_fit.get("gevp_n_basis_kept", 0))
                    cond_c0 = float(
                        channel_gevp_result.mass_fit.get("gevp_condition_number", float("nan"))
                    )
                    cond_str = f"{cond_c0:.3g}" if np.isfinite(cond_c0) else "n/a"
                    summary_lines.append(
                        f"- GEVP `{gevp_base_channel}` basis kept: `{n_kept}/{n_input}` "
                        f"(cond `{cond_str}`)"
                    )
                    if not filters_reported:
                        min_r2 = float(
                            channel_gevp_result.mass_fit.get("gevp_min_operator_r2", float("nan"))
                        )
                        min_windows = int(
                            channel_gevp_result.mass_fit.get("gevp_min_operator_windows", 0)
                        )
                        max_error_pct = float(
                            channel_gevp_result.mass_fit.get(
                                "gevp_max_operator_error_pct",
                                float("nan"),
                            )
                        )
                        remove_artifacts = bool(
                            channel_gevp_result.mass_fit.get(
                                "gevp_remove_artifacts",
                                channel_gevp_result.mass_fit.get(
                                    "gevp_exclude_zero_error_operators",
                                    False,
                                ),
                            )
                        )
                        min_r2_str = f"{min_r2:.3g}" if np.isfinite(min_r2) else "off"
                        max_error_pct_str = (
                            f"{max_error_pct:.3g}%" if np.isfinite(max_error_pct) else "off"
                        )
                        summary_lines.append(
                            f"- GEVP operator filters: `R² >= {min_r2_str}`, "
                            f"`n_windows >= {min_windows}`, "
                            f"`error % <= {max_error_pct_str}`, "
                            f"`remove artifacts={'on' if remove_artifacts else 'off'}`"
                        )
                        filters_reported = True
                if gevp_error:
                    summary_lines.append(f"- GEVP error: `{gevp_error}`")
                companion_strong_force_summary.object = "\n".join(summary_lines)

                error_parts = []
                if multiscale_error:
                    error_parts.append(f"multiscale: `{multiscale_error}`")
                if gevp_error:
                    error_parts.append(f"GEVP: `{gevp_error}`")
                if error_parts:
                    companion_strong_force_status.object = (
                        "**Complete with errors:** " + "; ".join(error_parts)
                    )
                else:
                    companion_strong_force_status.object = (
                        "**Complete:** companion strong-force channels computed."
                    )

            _run_tab_computation(
                state,
                companion_strong_force_status,
                "companion strong-force channels",
                _compute,
            )

        def _build_multiscale_cfg_from_tensor_settings() -> MultiscaleStrongForceConfig:
            return MultiscaleStrongForceConfig(
                warmup_fraction=float(tensor_calibration_settings.simulation_range[0]),
                end_fraction=float(tensor_calibration_settings.simulation_range[1]),
                mc_time_index=tensor_calibration_settings.mc_time_index,
                h_eff=float(tensor_calibration_settings.h_eff),
                mass=float(tensor_calibration_settings.mass),
                ell0=tensor_calibration_settings.ell0,
                edge_weight_mode=str(tensor_calibration_settings.edge_weight_mode),
                n_scales=int(tensor_calibration_settings.n_scales),
                kernel_type=str(tensor_calibration_settings.kernel_type),
                kernel_distance_method=str(tensor_calibration_settings.kernel_distance_method),
                kernel_assume_all_alive=bool(tensor_calibration_settings.kernel_assume_all_alive),
                kernel_batch_size=int(tensor_calibration_settings.kernel_batch_size),
                kernel_scale_frames=int(tensor_calibration_settings.kernel_scale_frames),
                kernel_scale_q_low=float(tensor_calibration_settings.kernel_scale_q_low),
                kernel_scale_q_high=float(tensor_calibration_settings.kernel_scale_q_high),
                max_lag=int(tensor_calibration_settings.max_lag),
                use_connected=bool(tensor_calibration_settings.use_connected),
                fit_mode=str(tensor_calibration_settings.fit_mode),
                fit_start=int(tensor_calibration_settings.fit_start),
                fit_stop=tensor_calibration_settings.fit_stop,
                min_fit_points=int(tensor_calibration_settings.min_fit_points),
                window_widths=_parse_window_widths(tensor_calibration_settings.window_widths_spec),
                compute_bootstrap_errors=bool(
                    tensor_calibration_settings.compute_bootstrap_errors
                ),
                n_bootstrap=int(tensor_calibration_settings.n_bootstrap),
                bootstrap_mode=str(tensor_calibration_settings.kernel_bootstrap_mode),
                walker_bootstrap_max_walkers=int(
                    tensor_calibration_settings.kernel_bootstrap_max_walkers
                ),
            )

        def on_run_tensor_calibration(_):
            def _compute(history):
                enabled_estimators = {str(v) for v in (_tcw.estimator_toggles.value or [])}
                if not enabled_estimators:
                    clear_tensor_gevp_calibration_tab(
                        _tcw,
                        "## Tensor Calibration\nSelect at least one estimator and rerun.",
                        state=state,
                    )
                    state["tensor_calibration_payload"] = None
                    return

                status_lines = [
                    "## Tensor Calibration",
                    f"- Enabled estimators: `{', '.join(sorted(enabled_estimators))}`",
                ]
                base_results: dict[str, ChannelCorrelatorResult] = {}
                strong_tensor_result: ChannelCorrelatorResult | None = None
                tensor_momentum_results: dict[str, ChannelCorrelatorResult] = {}
                tensor_momentum_meta: dict[str, Any] | None = None
                noncomp_multiscale_output: MultiscaleStrongForceOutput | None = None
                companion_multiscale_output: MultiscaleStrongForceOutput | None = None
                warnings: list[str] = []

                if (
                    "anisotropic_edge" in enabled_estimators
                    or "anisotropic_edge_traceless" in enabled_estimators
                ):
                    try:
                        base_results = _compute_anisotropic_edge_tensor_only_results(
                            history,
                            tensor_calibration_settings,
                        )
                    except Exception as exc:
                        warnings.append(f"anisotropic tensor failed: {exc}")

                if "strong_force" in enabled_estimators:
                    try:
                        strong_tensor_result = _compute_strong_tensor_for_anisotropic_edge(
                            history,
                            tensor_calibration_settings,
                        )
                    except Exception as exc:
                        warnings.append(f"strong-force tensor failed: {exc}")

                if (
                    "momentum_contracted" in enabled_estimators
                    or "momentum_components" in enabled_estimators
                ):
                    if bool(tensor_calibration_settings.use_companion_tensor_momentum):
                        try:
                            tensor_momentum_results, tensor_momentum_meta = (
                                _compute_tensor_momentum_for_anisotropic_edge(
                                    history,
                                    tensor_calibration_settings,
                                )
                            )
                        except Exception as exc:
                            warnings.append(f"tensor momentum failed: {exc}")
                    else:
                        warnings.append(
                            "tensor momentum disabled by settings (enable companion tensor momentum)."
                        )

                if "multiscale_non_companion" in enabled_estimators:
                    try:
                        noncomp_multiscale_channels: list[str] = []
                        if "anisotropic_edge" in enabled_estimators:
                            noncomp_multiscale_channels.append("tensor")
                        if "anisotropic_edge_traceless" in enabled_estimators:
                            noncomp_multiscale_channels.append("tensor_traceless")
                        if not noncomp_multiscale_channels:
                            noncomp_multiscale_channels = ["tensor", "tensor_traceless"]
                        noncomp_multiscale_channels = list(
                            dict.fromkeys(noncomp_multiscale_channels)
                        )
                        noncomp_multiscale_output = compute_multiscale_strong_force_channels(
                            history,
                            config=_build_multiscale_cfg_from_tensor_settings(),
                            channels=noncomp_multiscale_channels,
                        )
                        status_lines.append(
                            "- Non-companion multiscale channels: "
                            f"`{', '.join(noncomp_multiscale_channels)}`"
                        )
                    except Exception as exc:
                        warnings.append(f"non-companion multiscale failed: {exc}")

                if "multiscale_companion" in enabled_estimators:
                    try:
                        companion_multiscale_channels: list[str] = []
                        if (
                            "anisotropic_edge" in enabled_estimators
                            or "strong_force" in enabled_estimators
                            or "momentum_contracted" in enabled_estimators
                            or "momentum_components" in enabled_estimators
                        ):
                            companion_multiscale_channels.append("tensor_companion")
                        if "anisotropic_edge_traceless" in enabled_estimators:
                            companion_multiscale_channels.append("tensor_traceless_companion")
                        if not companion_multiscale_channels:
                            companion_multiscale_channels = [
                                "tensor_companion",
                                "tensor_traceless_companion",
                            ]
                        companion_multiscale_channels = list(
                            dict.fromkeys(companion_multiscale_channels)
                        )
                        companion_multiscale_output = compute_multiscale_strong_force_channels(
                            history,
                            config=_build_multiscale_cfg_from_tensor_settings(),
                            channels=companion_multiscale_channels,
                        )
                        status_lines.append(
                            "- Companion multiscale channels: "
                            f"`{', '.join(companion_multiscale_channels)}`"
                        )
                    except Exception as exc:
                        warnings.append(f"companion multiscale failed: {exc}")

                if warnings:
                    for warning in warnings:
                        status_lines.append(f"- Warning: `{warning}`")

                state[TENSOR_GEVP_DIRTY_STATE_KEY] = False
                payload = update_tensor_gevp_calibration_tab(
                    _tcw,
                    base_results=base_results,
                    strong_tensor_result=strong_tensor_result,
                    tensor_momentum_results=tensor_momentum_results,
                    tensor_momentum_meta=tensor_momentum_meta,
                    noncomp_multiscale_output=noncomp_multiscale_output,
                    companion_multiscale_output=companion_multiscale_output,
                    status_lines=status_lines,
                    state=state,
                    force_gevp=True,
                )
                state["tensor_calibration_base_results"] = base_results
                state["tensor_calibration_strong_result"] = strong_tensor_result
                state["tensor_calibration_momentum_results"] = tensor_momentum_results
                state["tensor_calibration_momentum_meta"] = tensor_momentum_meta
                state["tensor_calibration_noncomp_multiscale_output"] = noncomp_multiscale_output
                state["tensor_calibration_companion_multiscale_output"] = (
                    companion_multiscale_output
                )
                state["tensor_calibration_payload"] = payload
                if isinstance(payload, dict):
                    state["anisotropic_edge_tensor_correction_payload"] = payload

            _run_tab_computation(
                state,
                _tcw.status,
                "tensor calibration",
                _compute,
            )

        def _refresh_tensor_calibration_views(*, force_gevp: bool = False) -> None:
            if state.get("history") is None:
                return
            if state.get("tensor_calibration_base_results") is None:
                return
            payload = update_tensor_gevp_calibration_tab(
                _tcw,
                base_results=state.get("tensor_calibration_base_results") or {},
                strong_tensor_result=state.get("tensor_calibration_strong_result"),
                tensor_momentum_results=state.get("tensor_calibration_momentum_results"),
                tensor_momentum_meta=state.get("tensor_calibration_momentum_meta"),
                noncomp_multiscale_output=state.get(
                    "tensor_calibration_noncomp_multiscale_output"
                ),
                companion_multiscale_output=state.get(
                    "tensor_calibration_companion_multiscale_output"
                ),
                state=state,
                force_gevp=force_gevp,
            )
            state["tensor_calibration_payload"] = payload
            if isinstance(payload, dict):
                state["anisotropic_edge_tensor_correction_payload"] = payload

        def _on_tensor_calibration_mode_change(_event):
            _refresh_tensor_calibration_views()

        def _on_tensor_calibration_estimator_toggle_change(_event):
            state[TENSOR_GEVP_DIRTY_STATE_KEY] = False
            _refresh_tensor_calibration_views()

        def _on_tensor_calibration_gevp_selection_change(_event):
            state[TENSOR_GEVP_DIRTY_STATE_KEY] = True
            _refresh_tensor_calibration_views(force_gevp=False)

        def _on_tensor_calibration_gevp_compute_click(_event):
            state[TENSOR_GEVP_DIRTY_STATE_KEY] = False
            _refresh_tensor_calibration_views(force_gevp=True)

        def _on_tensor_calibration_gevp_family_select_all_click(_event):
            _tcw.gevp_family_select.value = [str(v) for v in list(_tcw.gevp_family_select.options)]

        def _on_tensor_calibration_gevp_family_clear_click(_event):
            _tcw.gevp_family_select.value = []

        def _on_tensor_calibration_gevp_scale_select_all_click(_event):
            _tcw.gevp_scale_select.value = [str(v) for v in list(_tcw.gevp_scale_select.options)]

        def _on_tensor_calibration_gevp_scale_clear_click(_event):
            _tcw.gevp_scale_select.value = []

        _tcw.mass_mode.param.watch(_on_tensor_calibration_mode_change, "value")
        _tcw.estimator_toggles.param.watch(
            _on_tensor_calibration_estimator_toggle_change,
            "value",
        )
        _tcw.gevp_family_select.param.watch(
            _on_tensor_calibration_gevp_selection_change,
            "value",
        )
        _tcw.gevp_scale_select.param.watch(
            _on_tensor_calibration_gevp_selection_change,
            "value",
        )
        _tcw.gevp_compute_button.param.watch(
            _on_tensor_calibration_gevp_compute_click,
            "clicks",
        )
        _tcw.gevp_family_select_all_button.param.watch(
            _on_tensor_calibration_gevp_family_select_all_click,
            "clicks",
        )
        _tcw.gevp_family_clear_button.param.watch(
            _on_tensor_calibration_gevp_family_clear_click,
            "clicks",
        )
        _tcw.gevp_scale_select_all_button.param.watch(
            _on_tensor_calibration_gevp_scale_select_all_click,
            "clicks",
        )
        _tcw.gevp_scale_clear_button.param.watch(
            _on_tensor_calibration_gevp_scale_clear_click,
            "clicks",
        )

        # =====================================================================
        # New Dirac/Electroweak tab callbacks (unified observables)
        # =====================================================================

        def _update_electroweak_plots_generic(
            results: dict[str, ChannelCorrelatorResult],
            channel_plots_container,
            plots_spectrum,
            plots_overlay_corr,
            plots_overlay_meff,
        ) -> None:
            mandatory_channels = (
                "su2_doublet",
                "ew_mixed",
                "su2_component",
                "su2_phase",
            )
            filtered_results: dict[str, ChannelCorrelatorResult] = {}
            seen: set[str] = set()

            for name in mandatory_channels:
                result = results.get(name)
                if result is None or result.n_samples <= 0 or name in seen:
                    continue
                filtered_results[name] = result
                seen.add(name)

            for name, result in results.items():
                if name in seen or result.n_samples <= 0:
                    continue
                if name.endswith("_gevp"):
                    filtered_results[name] = result
                    seen.add(name)
                    continue
                if "_multiscale" in name:
                    continue
                filtered_results[name] = result
                seen.add(name)

            _update_correlator_plots(
                filtered_results,
                channel_plots_container,
                plots_spectrum,
                plots_overlay_corr,
                plots_overlay_meff,
                correlator_logy=False,
                channels_per_row=2,
            )

        def _canonicalize_electroweak_channel_name(raw_name: str) -> str:
            channel_name = str(raw_name).strip()
            if not channel_name:
                return ""
            while channel_name.endswith("_q2"):
                channel_name = channel_name[:-3]
            while channel_name.endswith("_companion"):
                channel_name = channel_name[:-10]
            changed = True
            while changed:
                changed = False
                for suffix in ("_directed", "_cloner", "_resister", "_persister"):
                    if channel_name.endswith(suffix):
                        channel_name = channel_name[: -len(suffix)]
                        changed = True
            return channel_name

        def _extract_electroweak_refs_from_table(table: pn.widgets.Tabulator) -> dict[str, float]:
            refs: dict[str, float] = {}
            df = table.value
            if not isinstance(df, pd.DataFrame):
                return refs
            for _, row in df.iterrows():
                channel = str(row.get("channel", "")).strip()
                if not channel:
                    continue
                raw_mass = row.get("mass_ref_GeV")
                if isinstance(raw_mass, str):
                    raw_mass = raw_mass.strip()
                    if raw_mass == "":
                        continue
                try:
                    mass_ref = float(raw_mass)
                except (TypeError, ValueError):
                    continue
                if mass_ref > 0:
                    refs[channel] = mass_ref
            return refs

        def _build_electroweak_ratio_specs(
            refs: dict[str, float],
        ) -> list[tuple[str, str, str]]:
            ordered_channels: list[str] = []
            for channel in ELECTROWEAK_CANONICAL_CHANNEL_ORDER:
                if channel in refs:
                    ordered_channels.append(channel)
            for channel in sorted(refs):
                if channel not in ordered_channels:
                    ordered_channels.append(channel)

            ratio_specs: list[tuple[str, str, str]] = []
            for numerator in ordered_channels:
                num_ref = refs.get(numerator, float("nan"))
                if not np.isfinite(num_ref) or num_ref <= 0:
                    continue
                for denominator in ordered_channels:
                    if numerator == denominator:
                        continue
                    den_ref = refs.get(denominator, float("nan"))
                    if not np.isfinite(den_ref) or den_ref <= 0:
                        continue
                    ratio_specs.append((
                        numerator,
                        denominator,
                        f"target ≈ {num_ref / den_ref:.5g}",
                    ))
            return ratio_specs

        def _extract_electroweak_ratio_reference(spec_annotation: str) -> float:
            match = re.search(r"≈\s*([0-9]+(?:\.[0-9]+)?)", str(spec_annotation))
            if match is None:
                return float("nan")
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return float("nan")

        def _electroweak_variant_options(
            results: dict[str, ChannelCorrelatorResult],
        ) -> dict[str, list[str]]:
            options: dict[str, list[str]] = {}
            for name, result in results.items():
                if int(getattr(result, "n_samples", 0)) <= 0:
                    continue
                canonical = _canonicalize_electroweak_channel_name(name)
                if not canonical:
                    continue
                options.setdefault(canonical, [])
                name_str = str(name)
                if name_str not in options[canonical]:
                    options[canonical].append(name_str)
            return options

        def _electroweak_best_variant_overrides(
            results: dict[str, ChannelCorrelatorResult],
            ratio_specs: list[tuple[str, str, str]] | None,
            mode: str = "AIC-Weighted",
        ) -> dict[str, str]:
            if not ratio_specs:
                new_dirac_ew_best_combo_summary.object = (
                    "**Best global variant combo:** unavailable "
                    "(no finite measured ratio targets are configured)."
                )
                new_dirac_ew_best_combo_table.value = pd.DataFrame()
                new_dirac_ew_best_combo_ratio_pane.object = (
                    "**Mass Ratios (Best Global Combo):** unavailable "
                    "(no finite measured ratio targets are configured)."
                )
                return {}

            variant_options = _electroweak_variant_options(results)
            combo_channels: list[str] = []
            for numerator, denominator, _spec in ratio_specs:
                if numerator not in combo_channels:
                    combo_channels.append(numerator)
                if denominator not in combo_channels:
                    combo_channels.append(denominator)

            available = {
                channel: list(variants)
                for channel, variants in variant_options.items()
                if variants
            }
            missing_channels = [channel for channel in combo_channels if channel not in available]
            if missing_channels:
                reason = "missing variants for: " + ", ".join(
                    f"`{ch}`" for ch in missing_channels
                )
                new_dirac_ew_best_combo_summary.object = (
                    f"**Best global variant combo:** unavailable ({reason})."
                )
                new_dirac_ew_best_combo_table.value = pd.DataFrame()
                new_dirac_ew_best_combo_ratio_pane.object = (
                    f"**Mass Ratios (Best Global Combo):** unavailable ({reason})."
                )
                return {}

            variant_mass_lookup: dict[str, dict[str, float]] = {
                channel: {
                    variant: _get_channel_mass(results[variant], mode=mode)
                    if variant in results
                    else float("nan")
                    for variant in variants
                }
                for channel, variants in available.items()
            }

            combo_rows: list[dict[str, Any]] = []
            option_lists = [available[channel] for channel in combo_channels]
            combos_considered = 0
            combo_truncated = False
            max_combo_evaluations = 100000
            total_combinations = int(
                np.prod([len(options) for options in option_lists], dtype=np.int64)
            )
            evaluation_options = {channel: list(options) for channel, options in available.items()}
            if total_combinations > max_combo_evaluations:
                if len(combo_channels) <= 0:
                    return {}
                per_channel_cap = max(
                    1,
                    int(np.floor(max_combo_evaluations ** (1.0 / float(len(combo_channels))))),
                )
                for channel in combo_channels:
                    if len(evaluation_options[channel]) > per_channel_cap:
                        evaluation_options[channel] = evaluation_options[channel][:per_channel_cap]
                        combo_truncated = True
                option_lists = [evaluation_options[channel] for channel in combo_channels]

            for selected_variants in itertools.product(*option_lists):
                combos_considered += 1
                selection = {
                    channel: str(variant_name)
                    for channel, variant_name in zip(
                        combo_channels, selected_variants, strict=False
                    )
                }
                per_channel_error = dict.fromkeys(combo_channels, 0.0)
                per_ratio_error: dict[str, float] = {}
                total_abs_pct_error = 0.0
                used_ratio_count = 0
                valid_combo = True

                for numerator, denominator, annotation in ratio_specs:
                    num_variant = selection.get(numerator)
                    den_variant = selection.get(denominator)
                    if num_variant is None or den_variant is None:
                        valid_combo = False
                        break
                    num_mass = variant_mass_lookup.get(numerator, {}).get(
                        num_variant, float("nan")
                    )
                    den_mass = variant_mass_lookup.get(denominator, {}).get(
                        den_variant, float("nan")
                    )
                    if not np.isfinite(num_mass) or not np.isfinite(den_mass) or den_mass <= 0:
                        valid_combo = False
                        break

                    reference = _extract_electroweak_ratio_reference(annotation)
                    if not np.isfinite(reference) or reference <= 0:
                        continue
                    ratio = float(num_mass / den_mass)
                    abs_pct_error = float(abs((ratio / reference - 1.0) * 100.0))
                    per_ratio_error[
                        f"error_{numerator}_over_{denominator}_abs_pct"
                    ] = abs_pct_error
                    per_channel_error[numerator] += abs_pct_error
                    per_channel_error[denominator] += abs_pct_error
                    total_abs_pct_error += abs_pct_error
                    used_ratio_count += 1

                if not valid_combo or used_ratio_count == 0:
                    continue

                combo_row: dict[str, Any] = {
                    "total_abs_pct_error": total_abs_pct_error,
                    "mean_abs_pct_error": total_abs_pct_error / float(used_ratio_count),
                    "n_ratio_targets": int(used_ratio_count),
                }
                for channel in combo_channels:
                    combo_row[f"variant_{channel}"] = selection[channel]
                    combo_row[f"error_{channel}_abs_pct"] = float(per_channel_error[channel])
                combo_row.update(per_ratio_error)
                combo_rows.append(combo_row)

            if combo_rows:
                combo_df = pd.DataFrame(combo_rows).sort_values(
                    ["total_abs_pct_error", "mean_abs_pct_error"],
                    ascending=[True, True],
                    kind="stable",
                )
                top_combo_df = combo_df.head(5).reset_index(drop=True)
                new_dirac_ew_best_combo_table.value = top_combo_df
                best_row = top_combo_df.iloc[0]
                best_selection = {
                    channel: str(best_row[f"variant_{channel}"])
                    for channel in combo_channels
                    if f"variant_{channel}" in best_row
                }
                truncation_note = (
                    " Evaluated a capped subset of combinations." if combo_truncated else ""
                )
                new_dirac_ew_best_combo_summary.object = (
                    f"**Best global variant combo (top 5):** "
                    f"`{len(combo_df)}` valid combinations ranked by summed absolute `%` error "
                    f"across `{len(ratio_specs)}` ratio targets.  \n"
                    f"_Best total error:_ `{float(best_row['total_abs_pct_error']):.4g}%` "
                    f"(mean `{float(best_row['mean_abs_pct_error']):.4g}%`).  \n"
                    f"_Combinations considered:_ `{combos_considered}`.{truncation_note}"
                )
                best_ratio_lines = ["**Mass Ratios (Best Global Combo):**"]
                for numerator, denominator, annotation in ratio_specs:
                    num_variant = best_selection.get(numerator)
                    den_variant = best_selection.get(denominator)
                    if not num_variant or not den_variant:
                        continue
                    num_mass = variant_mass_lookup.get(numerator, {}).get(
                        num_variant, float("nan")
                    )
                    den_mass = variant_mass_lookup.get(denominator, {}).get(
                        den_variant, float("nan")
                    )
                    if (
                        not np.isfinite(num_mass)
                        or not np.isfinite(den_mass)
                        or den_mass <= 0
                    ):
                        continue
                    ratio_value = float(num_mass / den_mass)
                    ref_value = _extract_electroweak_ratio_reference(annotation)
                    abs_pct_err = float("nan")
                    if np.isfinite(ref_value) and ref_value > 0:
                        abs_pct_err = float(abs((ratio_value / ref_value - 1.0) * 100.0))
                    ann = str(annotation).strip()
                    if not ann:
                        if np.isfinite(ref_value) and ref_value > 0:
                            ann = f"target ≈ {ref_value:.4g}"
                        else:
                            ann = "no external target"
                    if np.isfinite(abs_pct_err):
                        ann = f"{ann}; abs error {abs_pct_err:.2f}%"
                    best_ratio_lines.append(
                        f"- {numerator}/{denominator}: **{ratio_value:.3f}** ({ann})"
                    )
                if len(best_ratio_lines) == 1:
                    best_ratio_lines.append(
                        "- n/a (missing valid masses for selected variants)"
                    )
                new_dirac_ew_best_combo_ratio_pane.object = "  \n".join(best_ratio_lines)
                return best_selection
            else:
                if total_combinations == 0:
                    reason = "no variant combinations available"
                else:
                    reason = (
                        "no complete variant combination has valid masses "
                        "for all ratio targets"
                    )
                new_dirac_ew_best_combo_summary.object = (
                    f"**Best global variant combo:** unavailable ({reason})."
                )
                new_dirac_ew_best_combo_table.value = pd.DataFrame()
                new_dirac_ew_best_combo_ratio_pane.object = (
                    f"**Mass Ratios (Best Global Combo):** unavailable ({reason})."
                )
                return {}

        def _apply_electroweak_comparison_overrides(
            results: dict[str, ChannelCorrelatorResult],
            comparison_channel_overrides: dict[str, str] | None,
        ) -> dict[str, ChannelCorrelatorResult]:
            if not comparison_channel_overrides:
                comparison_channel_overrides = {}

            variant_options = _electroweak_variant_options(results)
            canonical_order = list(ELECTROWEAK_CANONICAL_CHANNEL_ORDER)
            for channel in sorted(variant_options):
                if channel not in canonical_order:
                    canonical_order.append(channel)

            projected: dict[str, ChannelCorrelatorResult] = {}
            for canonical_name in canonical_order:
                options = variant_options.get(canonical_name)
                if not options:
                    continue
                selected_name = comparison_channel_overrides.get(canonical_name)
                chosen_name = selected_name if selected_name in options else options[0]
                chosen_result = results.get(chosen_name)
                if chosen_result is not None:
                    projected[canonical_name] = chosen_result

            for name, result in results.items():
                canonical_name = _canonicalize_electroweak_channel_name(name)
                if canonical_name in projected:
                    continue
                projected[str(name)] = result

            return projected

        def _build_electroweak_gevp_input_results(
            base_results: dict[str, ChannelCorrelatorResult],
            comparison_channel_overrides: dict[str, str] | None,
        ) -> dict[str, ChannelCorrelatorResult]:
            gevp_results = dict(base_results)
            if not comparison_channel_overrides:
                return gevp_results
            for base_channel in list(SU2_BASE_CHANNELS) + list(U1_BASE_CHANNELS) + list(EW_MIXED_BASE_CHANNELS):
                selected = str(
                    comparison_channel_overrides.get(base_channel, base_channel)
                ).strip()
                selected_result = base_results.get(selected)
                if selected_result is not None:
                    gevp_results[base_channel] = selected_result
            return gevp_results

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
            comparison_channel_overrides: dict[str, str] | None = None,
            ratio_specs: list[tuple[str, str, str]] | None = None,
        ) -> None:
            if ratio_specs is None:
                ratio_specs = _build_electroweak_ratio_specs(
                    _extract_electroweak_refs_from_table(ref_table)
                )
            comparison_results = dict(results)
            if comparison_channel_overrides:
                comparison_results = _apply_electroweak_comparison_overrides(
                    results,
                    comparison_channel_overrides,
                )
            _update_mass_table(comparison_results, mass_table, mode)
            ratio_pane.object = _format_ratios(
                comparison_results,
                mode,
                title="Electroweak Ratios",
                ratio_specs=ratio_specs,
            )

            masses = _extract_masses(comparison_results, mode, family_map=None)
            r2s = _extract_r2(comparison_results, mode, family_map=None)
            base_name = "u1_dressed" if "u1_dressed" in masses else "u1_phase"

            refs = _extract_electroweak_refs_from_table(ref_table)

            ratio_rows = _build_electroweak_ratio_rows(masses, base_name, refs=refs)
            ratio_table.value = pd.DataFrame(ratio_rows) if ratio_rows else pd.DataFrame()

            if not masses or not refs:
                fit_table.value = pd.DataFrame()
                anchor_table.value = pd.DataFrame()
                compare_table.value = pd.DataFrame()
                return

            fit_table.value = pd.DataFrame(_build_electroweak_best_fit_rows(masses, refs, r2s))
            anchor_table.value = pd.DataFrame(_build_electroweak_anchor_rows(masses, refs, r2s))
            compare_table.value = pd.DataFrame(_build_electroweak_comparison_rows(masses, refs))

        def _resolve_new_dirac_ew_filter_values() -> tuple[float, int, float, bool]:
            try:
                min_r2 = float(new_dirac_ew_settings.gevp_min_operator_r2)
            except (TypeError, ValueError):
                min_r2 = 0.5
            try:
                min_windows = max(0, int(new_dirac_ew_settings.gevp_min_operator_windows))
            except (TypeError, ValueError):
                min_windows = 10
            try:
                max_error_pct = float(new_dirac_ew_settings.gevp_max_operator_error_pct)
            except (TypeError, ValueError):
                max_error_pct = 30.0
            remove_artifacts = bool(new_dirac_ew_settings.gevp_remove_artifacts)
            return min_r2, min_windows, max_error_pct, remove_artifacts

        def _update_ew_gevp_displays(
            multiscale_output: MultiscaleElectroweakOutput | None,
            results: dict[str, ChannelCorrelatorResult],
        ) -> None:
            min_r2, min_windows, max_error_pct, remove_artifacts = _resolve_new_dirac_ew_filter_values()
            t0 = int(new_dirac_ew_settings.gevp_t0)
            eig_rel_cutoff = float(new_dirac_ew_settings.gevp_eig_rel_cutoff)
            per_scale = multiscale_output.per_scale_results if multiscale_output else {}
            gevp_results = {k: v for k, v in results.items() if k.endswith("_gevp")}
            for family, widgets in (
                ("su2", new_dirac_ew_su2_gevp_widgets),
                ("u1", new_dirac_ew_u1_gevp_widgets),
                ("ew_mixed", new_dirac_ew_ew_mixed_gevp_widgets),
            ):
                try:
                    update_gevp_dashboard(
                        widgets,
                        selected_channel_name=f"{family.upper()} GEVP",
                        raw_channel_name=f"{family}_companion" if family != "ew_mixed" else "ew_mixed_companion",
                        per_scale_results=per_scale,
                        original_results=results,
                        companion_gevp_results=gevp_results,
                        min_r2=min_r2,
                        min_windows=min_windows,
                        max_error_pct=max_error_pct,
                        remove_artifacts=remove_artifacts,
                        selected_families=[family],
                        t0=t0,
                        eig_rel_cutoff=eig_rel_cutoff,
                        use_connected=bool(new_dirac_ew_settings.use_connected),
                    )
                except Exception:
                    clear_gevp_dashboard(widgets)

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
            comparison_channel_overrides: dict[str, str] | None = None,
            ratio_specs: list[tuple[str, str, str]] | None = None,
        ) -> None:
            if mode is None:
                mode = new_dirac_ew_mass_mode.value
            min_r2, min_windows, max_error_pct, remove_artifacts = (
                _resolve_new_dirac_ew_filter_values()
            )
            filtered_results, filtered_out_results = _split_results_by_companion_gevp_filters(
                results,
                min_r2=min_r2,
                min_windows=min_windows,
                max_error_pct=max_error_pct,
                remove_artifacts=remove_artifacts,
            )
            _update_electroweak_tables_generic(
                filtered_results,
                mode,
                new_dirac_ew_mass_table,
                new_dirac_ew_ratio_pane,
                new_dirac_ew_ratio_table,
                new_dirac_ew_fit_table,
                new_dirac_ew_anchor_table,
                new_dirac_ew_compare_table,
                new_dirac_ew_ref_table,
                comparison_channel_overrides=comparison_channel_overrides,
                ratio_specs=ratio_specs,
            )
            filtered_out_dict = {
                name: payload[0] for name, payload in filtered_out_results.items()
            }
            _update_mass_table(
                filtered_out_dict,
                new_dirac_ew_filtered_mass_table,
                str(mode),
            )

            min_r2_str = f"{min_r2:.3g}" if np.isfinite(min_r2) else "off"
            max_error_pct_str = f"{max_error_pct:.3g}%" if np.isfinite(max_error_pct) else "off"
            if filtered_out_results:
                preview = ", ".join(
                    f"{name}({reason})"
                    for name, (_, reason) in list(filtered_out_results.items())[:6]
                )
                suffix = " ..." if len(filtered_out_results) > 6 else ""
                new_dirac_ew_filtered_summary.object = (
                    f"**Filtered-out candidates:** `{len(filtered_out_results)}` excluded; "
                    f"`{len(filtered_results)}` shown.  \n"
                    f"_Filters:_ `R² >= {min_r2_str}`, `n_windows >= {min_windows}`, "
                    f"`error % <= {max_error_pct_str}`, "
                    f"`remove artifacts={'on' if remove_artifacts else 'off'}`.  \n"
                    f"_Preview:_ `{preview}{suffix}`"
                )
            else:
                new_dirac_ew_filtered_summary.object = (
                    "**Filtered-out candidates:** none.  \n"
                    f"_Filters:_ `R² >= {min_r2_str}`, `n_windows >= {min_windows}`, "
                    f"`error % <= {max_error_pct_str}`, "
                    f"`remove artifacts={'on' if remove_artifacts else 'off'}`."
                )
            _update_ew_gevp_displays(
                state.get("new_dirac_ew_multiscale_output"),
                results,
            )

            # --- Symmetry-breaking derived observables ---
            import math as _math

            sb_lines = ["**Symmetry Breaking Observables:**"]
            ew_masses = _extract_masses(filtered_results, mode=str(mode), family_map=None)
            m_z_proxy = ew_masses.get("fitness_phase")
            m_w_proxy = ew_masses.get("clone_indicator")
            if m_z_proxy and m_z_proxy > 0 and m_w_proxy and m_w_proxy > 0:
                cos_tw = min(m_w_proxy / m_z_proxy, 1.0)
                sin2_tw = 1.0 - cos_tw ** 2
                sb_lines.append(
                    f"- **M_Z proxy** (fitness_phase): `{m_z_proxy:.6g}`"
                )
                sb_lines.append(
                    f"- **M_W proxy** (clone_indicator): `{m_w_proxy:.6g}`"
                )
                theta_w_deg = _math.degrees(_math.acos(cos_tw))
                sb_lines.append(
                    f"- **\u03b8_W** = `{theta_w_deg:.2f}\u00b0` "
                    f"(SM: `{_math.degrees(_math.acos(80.379 / 91.1876)):.2f}\u00b0`)"
                )
                sb_lines.append(
                    f"- **sin\u00b2(\u03b8_W)** = `{sin2_tw:.5f}` (SM: `0.23129`; "
                    f"error: `{(sin2_tw / 0.23129 - 1.0) * 100:.1f}%`)"
                )
            elif m_z_proxy and m_z_proxy > 0:
                sb_lines.append(f"- **M_Z proxy** (fitness_phase): `{m_z_proxy:.6g}`")
                sb_lines.append("- **M_W proxy** (clone_indicator): _not available_")
            elif m_w_proxy and m_w_proxy > 0:
                sb_lines.append("- **M_Z proxy** (fitness_phase): _not available_")
                sb_lines.append(f"- **M_W proxy** (clone_indicator): `{m_w_proxy:.6g}`")
            else:
                sb_lines.append(
                    "- _Fitness phase and clone indicator channels not computed._"
                )

            # Parity violation: compare velocity-norm masses across walker types
            parity_masses = {}
            for label in ("cloner", "resister", "persister"):
                m = ew_masses.get(f"velocity_norm_{label}")
                if m and m > 0:
                    parity_masses[label] = m
            if len(parity_masses) >= 2:
                sb_lines.append("- **Parity test** (velocity norm by walker type):")
                types = sorted(parity_masses.keys())
                for i, t1 in enumerate(types):
                    for t2 in types[i + 1 :]:
                        ratio = parity_masses[t1] / parity_masses[t2]
                        sb_lines.append(
                            f"  - m({t1})/m({t2}) = `{ratio:.4f}` "
                            f"(parity conservation \u2192 1.0; "
                            f"deviation: `{(ratio - 1.0) * 100:+.1f}%`)"
                        )
            else:
                sb_lines.append(
                    "- **Parity test:** _enable walker type split and select "
                    "parity_velocity channels._"
                )
            new_dirac_ew_symmetry_breaking_summary.object = "  \n".join(sb_lines)

        def _update_new_dirac_ew_multiscale_views(
            output: MultiscaleElectroweakOutput | None,
            *,
            original_results: dict[str, ChannelCorrelatorResult] | None = None,
            error: str | None = None,
        ) -> None:
            if error:
                new_dirac_ew_multiscale_summary.object = (
                    "### SU(2) Multiscale Summary\n" "- Status: failed\n" f"- Error: `{error}`"
                )
                new_dirac_ew_multiscale_table.value = pd.DataFrame()
                new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
                clear_gevp_dashboard(new_dirac_ew_su2_gevp_widgets)
                clear_gevp_dashboard(new_dirac_ew_u1_gevp_widgets)
                clear_gevp_dashboard(new_dirac_ew_ew_mixed_gevp_widgets)
                return

            if output is None:
                new_dirac_ew_multiscale_summary.object = (
                    "### SU(2) Multiscale Summary\n"
                    "_Multiscale kernels disabled (original estimators only)._"
                )
                new_dirac_ew_multiscale_table.value = pd.DataFrame()
                new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
                return

            scale_values = (
                output.scales.detach().cpu().numpy() if output.scales.numel() else np.array([])
            )
            summary_lines = [
                "### SU(2) Multiscale Summary",
                f"- Scales: `{len(scale_values)}`",
                f"- Frames: `{len(output.frame_indices)}`",
                f"- Bootstrap mode: `{output.bootstrap_mode_applied}`",
            ]
            if scale_values.size > 0:
                summary_lines.append(
                    f"- Scale range: `[{float(scale_values.min()):.4g}, {float(scale_values.max()):.4g}]`"
                )
            if output.notes:
                for note in output.notes:
                    summary_lines.append(f"- Note: {note}")
            new_dirac_ew_multiscale_summary.object = "  \n".join(summary_lines)

            mode = str(new_dirac_ew_mass_mode.value)
            min_r2, min_windows, max_error_pct, remove_artifacts = (
                _resolve_new_dirac_ew_filter_values()
            )
            rows: list[dict[str, Any]] = []
            per_scale_rows: list[dict[str, Any]] = []
            for alias, results_per_scale in output.per_scale_results.items():
                base_name = str(alias).replace("_companion", "")
                best_idx_raw = int(output.best_scale_index.get(alias, -1))
                best_idx = best_idx_raw if 0 <= best_idx_raw < len(results_per_scale) else -1
                best_result = results_per_scale[best_idx] if best_idx >= 0 else None
                if best_result is not None:
                    best_reason = _companion_gevp_filter_reason(
                        best_result,
                        min_r2=min_r2,
                        min_windows=min_windows,
                        max_error_pct=max_error_pct,
                        remove_artifacts=remove_artifacts,
                    )
                    if best_reason is not None:
                        best_result = None
                        best_idx = -1

                original_mass = float("nan")
                if isinstance(original_results, dict):
                    original_result = original_results.get(base_name)
                    if original_result is not None:
                        original_reason = _companion_gevp_filter_reason(
                            original_result,
                            min_r2=min_r2,
                            min_windows=min_windows,
                            max_error_pct=max_error_pct,
                            remove_artifacts=remove_artifacts,
                        )
                        if original_reason is None:
                            original_mass = float(_get_channel_mass(original_result, mode))

                for scale_idx, res in enumerate(results_per_scale):
                    if res is None:
                        continue
                    reason = _companion_gevp_filter_reason(
                        res,
                        min_r2=min_r2,
                        min_windows=min_windows,
                        max_error_pct=max_error_pct,
                        remove_artifacts=remove_artifacts,
                    )
                    if reason is not None:
                        continue
                    fit = res.mass_fit if isinstance(res.mass_fit, dict) else {}
                    mass = float(_get_channel_mass(res, mode))
                    mass_error = float(_get_channel_mass_error(res, mode))
                    err_pct = (
                        abs(mass_error / mass) * 100.0
                        if np.isfinite(mass) and mass > 0 and np.isfinite(mass_error)
                        else float("nan")
                    )
                    per_scale_rows.append({
                        "channel": base_name,
                        "scale_idx": int(scale_idx),
                        "scale": (
                            float(scale_values[scale_idx])
                            if scale_values.size > scale_idx
                            else float("nan")
                        ),
                        "mass": mass,
                        "mass_error": mass_error,
                        "mass_error_pct": err_pct,
                        "r2": float(_get_channel_r2(res, mode)),
                        "n_windows": _extract_n_windows_for_filter(res),
                        "source": str(fit.get("source", "multiscale")),
                    })

                if best_result is None:
                    continue
                best_mass = float(_get_channel_mass(best_result, mode))
                best_error = float(_get_channel_mass_error(best_result, mode))
                best_err_pct = (
                    abs(best_error / best_mass) * 100.0
                    if np.isfinite(best_mass) and best_mass > 0 and np.isfinite(best_error)
                    else float("nan")
                )
                rows.append({
                    "channel": base_name,
                    "best_scale_idx": int(best_idx),
                    "best_scale": (
                        float(scale_values[best_idx])
                        if scale_values.size > best_idx >= 0
                        else float("nan")
                    ),
                    "mass": best_mass,
                    "mass_error": best_error,
                    "mass_error_pct": best_err_pct,
                    "r2": float(_get_channel_r2(best_result, mode)),
                    "n_windows": _extract_n_windows_for_filter(best_result),
                    "original_mass": original_mass,
                    "delta_vs_original_pct": (
                        ((best_mass - original_mass) / original_mass) * 100.0
                        if np.isfinite(original_mass)
                        and original_mass > 0
                        and np.isfinite(best_mass)
                        else float("nan")
                    ),
                })

            new_dirac_ew_multiscale_table.value = (
                pd.DataFrame(rows).sort_values(["channel"], kind="stable")
                if rows
                else pd.DataFrame()
            )
            new_dirac_ew_multiscale_per_scale_table.value = (
                pd.DataFrame(per_scale_rows).sort_values(["channel", "scale_idx"], kind="stable")
                if per_scale_rows
                else pd.DataFrame()
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
            electron_component_mass = _get_channel_mass(
                bundle.electron_component_result, mode=mode
            )
            higgs_sigma_mass = _get_channel_mass(bundle.higgs_sigma_result, mode=mode)

            color_singlet = bundle.color_singlet_spectrum
            electron_dirac = color_singlet.electron_mass if color_singlet is not None else None

            rows: list[dict[str, Any]] = []
            rows.append({
                "observable": "electron_dirac",
                "measured": electron_dirac,
                "observed_GeV": observed.get("electron_dirac"),
                "error_pct": (
                    (
                        (electron_dirac - observed["electron_dirac"])
                        / observed["electron_dirac"]
                        * 100.0
                    )
                    if electron_dirac is not None and observed.get("electron_dirac", 0) > 0
                    else None
                ),
                "note": "Dirac spectral color-singlet proxy",
            })
            rows.append({
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
            })
            rows.append({
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
            })
            rows.append({
                "observable": "higgs_sigma",
                "measured": higgs_sigma_mass,
                "observed_GeV": observed.get("higgs_sigma"),
                "error_pct": (
                    (higgs_sigma_mass - observed["higgs_sigma"]) / observed["higgs_sigma"] * 100.0
                    if observed.get("higgs_sigma", 0) > 0
                    else None
                ),
                "note": "Radial fluctuation (sigma-mode) correlator mass",
            })
            for channel, mass in sorted(ew_masses.items()):
                obs = observed.get(channel)
                rows.append({
                    "observable": channel,
                    "measured": mass,
                    "observed_GeV": obs,
                    "error_pct": (
                        (mass - obs) / obs * 100.0 if obs is not None and obs > 0 else None
                    ),
                    "note": "Electroweak proxy channel",
                })
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
                ratio_rows.append({
                    "ratio": "mW/mH",
                    "measured": measured,
                    "observed": obs,
                    "error_pct": ((measured - obs) / obs * 100.0 if obs and obs > 0 else None),
                })
            if m_z is not None and m_h is not None and m_h > 0:
                measured = m_z / m_h
                obs = None
                if observed.get("su2_doublet", 0) > 0 and observed.get("higgs_sigma", 0) > 0:
                    obs = observed["su2_doublet"] / observed["higgs_sigma"]
                ratio_rows.append({
                    "ratio": "mZ/mH",
                    "measured": measured,
                    "observed": obs,
                    "error_pct": ((measured - obs) / obs * 100.0 if obs and obs > 0 else None),
                })
            if bundle.electron_mass_yukawa > 0 and electron_component_mass > 0:
                ratio_rows.append({
                    "ratio": "m(e_component)/m(e_yukawa)",
                    "measured": electron_component_mass / bundle.electron_mass_yukawa,
                    "observed": 1.0,
                    "error_pct": (
                        (electron_component_mass / bundle.electron_mass_yukawa - 1.0) * 100.0
                    ),
                })
            new_dirac_ew_ratio_table_extra.value = (
                pd.DataFrame(ratio_rows) if ratio_rows else pd.DataFrame()
            )

        def _on_new_dirac_ew_mass_mode_change(event) -> None:
            results = state.get("new_dirac_ew_results")
            if results is not None:
                _update_new_dirac_ew_electroweak_tables(
                    results,
                    event.new,
                    comparison_channel_overrides=state.get("new_dirac_ew_comparison_overrides"),
                    ratio_specs=state.get("new_dirac_ew_ratio_specs"),
                )
                _update_new_dirac_ew_multiscale_views(
                    state.get("new_dirac_ew_multiscale_output"),
                    original_results=results,
                    error=state.get("new_dirac_ew_multiscale_error"),
                )
            bundle = state.get("new_dirac_bundle")
            if bundle is not None:
                _update_new_dirac_ew_derived_tables(bundle, event.new)

        new_dirac_ew_mass_mode.param.watch(_on_new_dirac_ew_mass_mode_change, "value")

        def _build_new_dirac_ew_channel_config() -> ElectroweakChannelConfig:
            neighbor_method = (
                "auto"
                if new_dirac_ew_settings.neighbor_method == "voronoi"
                else new_dirac_ew_settings.neighbor_method
            )
            return ElectroweakChannelConfig(
                warmup_fraction=new_dirac_ew_settings.simulation_range[0],
                end_fraction=new_dirac_ew_settings.simulation_range[1],
                max_lag=new_dirac_ew_settings.max_lag,
                h_eff=new_dirac_ew_settings.h_eff,
                use_connected=new_dirac_ew_settings.use_connected,
                neighbor_method=neighbor_method,
                companion_topology=str(new_dirac_ew_settings.companion_topology_u1),
                companion_topology_u1=str(new_dirac_ew_settings.companion_topology_u1),
                companion_topology_su2=str(new_dirac_ew_settings.companion_topology_su2),
                companion_topology_ew_mixed=str(new_dirac_ew_settings.companion_topology_ew_mixed),
                edge_weight_mode=new_dirac_ew_settings.edge_weight_mode,
                neighbor_k=new_dirac_ew_settings.neighbor_k,
                window_widths=_parse_window_widths(new_dirac_ew_settings.window_widths_spec),
                fit_mode=new_dirac_ew_settings.fit_mode,
                fit_start=new_dirac_ew_settings.fit_start,
                fit_stop=new_dirac_ew_settings.fit_stop,
                min_fit_points=new_dirac_ew_settings.min_fit_points,
                epsilon_clone=new_dirac_ew_settings.epsilon_clone,
                lambda_alg=new_dirac_ew_settings.lambda_alg,
                su2_operator_mode=str(new_dirac_ew_settings.su2_operator_mode),
                enable_walker_type_split=bool(new_dirac_ew_settings.enable_walker_type_split),
                walker_type_scope=str(new_dirac_ew_settings.walker_type_scope),
                mc_time_index=new_dirac_ew_settings.mc_time_index,
                compute_bootstrap_errors=new_dirac_ew_settings.compute_bootstrap_errors,
                n_bootstrap=new_dirac_ew_settings.n_bootstrap,
            )

        def on_run_new_dirac_electroweak(_):
            def _compute(history):
                state["new_dirac_ew_comparison_overrides"] = None
                state["new_dirac_ew_ratio_specs"] = None
                ew_cfg = _build_new_dirac_ew_channel_config()
                requested_electroweak_channels = _collect_multiselect_values(
                    new_dirac_ew_channel_family_selectors
                )
                if not requested_electroweak_channels:
                    requested_electroweak_channels = [
                        channel
                        for channels in DEFAULT_ELECTROWEAK_CHANNEL_VARIANT_SELECTION.values()
                        for channel in channels
                    ]
                ew_output = compute_electroweak_channels(
                    history,
                    channels=requested_electroweak_channels,
                    config=ew_cfg,
                )
                base_results = dict(ew_output.channel_results)
                results = dict(base_results)
                ratio_specs = _build_electroweak_ratio_specs(
                    _extract_electroweak_refs_from_table(new_dirac_ew_ref_table)
                )
                comparison_overrides = _electroweak_best_variant_overrides(
                    base_results,
                    ratio_specs,
                )
                state["new_dirac_ew_ratio_specs"] = ratio_specs
                state["new_dirac_ew_comparison_overrides"] = comparison_overrides

                multiscale_output: MultiscaleElectroweakOutput | None = None
                multiscale_error: str | None = None
                gevp_error: str | None = None
                gevp_input_results = _build_electroweak_gevp_input_results(
                    base_results=base_results,
                    comparison_channel_overrides=comparison_overrides,
                )

                multiscale_supported = (
                    tuple(SU2_BASE_CHANNELS)
                    + tuple(SU2_DIRECTIONAL_CHANNELS)
                    + tuple(SU2_WALKER_TYPE_CHANNELS)
                    + tuple(U1_BASE_CHANNELS)
                    + tuple(EW_MIXED_BASE_CHANNELS)
                )
                multiscale_requested = [
                    name
                    for name in requested_electroweak_channels
                    if name in multiscale_supported
                ]
                if bool(new_dirac_ew_settings.use_multiscale_kernels) and multiscale_requested:
                    try:
                        ms_cfg = MultiscaleElectroweakConfig(
                            warmup_fraction=float(new_dirac_ew_settings.simulation_range[0]),
                            end_fraction=float(new_dirac_ew_settings.simulation_range[1]),
                            mc_time_index=new_dirac_ew_settings.mc_time_index,
                            h_eff=float(new_dirac_ew_settings.h_eff),
                            epsilon_clone=new_dirac_ew_settings.epsilon_clone,
                            lambda_alg=new_dirac_ew_settings.lambda_alg,
                            su2_operator_mode=str(new_dirac_ew_settings.su2_operator_mode),
                            enable_walker_type_split=bool(
                                new_dirac_ew_settings.enable_walker_type_split
                            ),
                            walker_type_scope=str(new_dirac_ew_settings.walker_type_scope),
                            edge_weight_mode=str(new_dirac_ew_settings.edge_weight_mode),
                            n_scales=int(new_dirac_ew_settings.n_scales),
                            kernel_type=str(new_dirac_ew_settings.kernel_type),
                            kernel_distance_method=str(
                                new_dirac_ew_settings.kernel_distance_method
                            ),
                            kernel_assume_all_alive=bool(
                                new_dirac_ew_settings.kernel_assume_all_alive
                            ),
                            kernel_batch_size=int(new_dirac_ew_settings.kernel_batch_size),
                            kernel_scale_frames=int(new_dirac_ew_settings.kernel_scale_frames),
                            kernel_scale_q_low=float(new_dirac_ew_settings.kernel_scale_q_low),
                            kernel_scale_q_high=float(new_dirac_ew_settings.kernel_scale_q_high),
                            max_lag=int(new_dirac_ew_settings.max_lag),
                            use_connected=bool(new_dirac_ew_settings.use_connected),
                            fit_mode=str(new_dirac_ew_settings.fit_mode),
                            fit_start=int(new_dirac_ew_settings.fit_start),
                            fit_stop=new_dirac_ew_settings.fit_stop,
                            min_fit_points=int(new_dirac_ew_settings.min_fit_points),
                            window_widths=_parse_window_widths(
                                new_dirac_ew_settings.window_widths_spec
                            ),
                            best_min_r2=float(new_dirac_ew_settings.gevp_min_operator_r2),
                            best_min_windows=int(new_dirac_ew_settings.gevp_min_operator_windows),
                            best_max_error_pct=float(
                                new_dirac_ew_settings.gevp_max_operator_error_pct
                            ),
                            best_remove_artifacts=bool(
                                new_dirac_ew_settings.gevp_remove_artifacts
                            ),
                            compute_bootstrap_errors=bool(
                                new_dirac_ew_settings.compute_bootstrap_errors
                            ),
                            n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                            bootstrap_mode=str(new_dirac_ew_settings.kernel_bootstrap_mode),
                        )
                        multiscale_output = compute_multiscale_electroweak_channels(
                            history,
                            config=ms_cfg,
                            channels=multiscale_requested,
                        )
                        min_r2, min_windows, max_error_pct, remove_artifacts = (
                            _resolve_new_dirac_ew_filter_values()
                        )
                        for alias, result in multiscale_output.best_results.items():
                            reason = _companion_gevp_filter_reason(
                                result,
                                min_r2=min_r2,
                                min_windows=min_windows,
                                max_error_pct=max_error_pct,
                                remove_artifacts=remove_artifacts,
                            )
                            if reason is not None:
                                continue
                            base_name = str(alias).replace("_companion", "")
                            tagged_name = f"{base_name}_multiscale_best"
                            tagged_result = replace(result, channel_name=tagged_name)
                            if isinstance(tagged_result.mass_fit, dict):
                                tagged_result.mass_fit["source"] = "multiscale_best"
                                tagged_result.mass_fit["base_channel"] = base_name
                            results[tagged_name] = tagged_result
                    except Exception as exc:
                        multiscale_error = str(exc)

                gevp_cfg = GEVPConfig(
                    t0=int(new_dirac_ew_settings.gevp_t0),
                    max_lag=int(new_dirac_ew_settings.max_lag),
                    use_connected=bool(new_dirac_ew_settings.use_connected),
                    fit_mode=str(new_dirac_ew_settings.fit_mode),
                    fit_start=int(new_dirac_ew_settings.fit_start),
                    fit_stop=new_dirac_ew_settings.fit_stop,
                    min_fit_points=int(new_dirac_ew_settings.min_fit_points),
                    window_widths=_parse_window_widths(
                        new_dirac_ew_settings.window_widths_spec
                    ),
                    basis_strategy=str(new_dirac_ew_settings.gevp_basis_strategy),
                    max_basis=int(new_dirac_ew_settings.gevp_max_basis),
                    min_operator_r2=float(new_dirac_ew_settings.gevp_min_operator_r2),
                    min_operator_windows=int(
                        new_dirac_ew_settings.gevp_min_operator_windows
                    ),
                    max_operator_error_pct=float(
                        new_dirac_ew_settings.gevp_max_operator_error_pct
                    ),
                    remove_artifacts=bool(new_dirac_ew_settings.gevp_remove_artifacts),
                    eig_rel_cutoff=float(new_dirac_ew_settings.gevp_eig_rel_cutoff),
                    cond_limit=float(new_dirac_ew_settings.gevp_cond_limit),
                    shrinkage=float(new_dirac_ew_settings.gevp_shrinkage),
                    compute_bootstrap_errors=bool(
                        new_dirac_ew_settings.compute_bootstrap_errors
                    ),
                    n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                    bootstrap_mode=str(new_dirac_ew_settings.gevp_bootstrap_mode),
                )
                for _gevp_family, _gevp_use_flag in (
                    ("su2", bool(new_dirac_ew_settings.use_su2_gevp)),
                    ("u1", bool(new_dirac_ew_settings.use_u1_gevp)),
                    ("ew_mixed", bool(new_dirac_ew_settings.use_ew_mixed_gevp)),
                ):
                    if not _gevp_use_flag:
                        continue
                    _gevp_basis = get_companion_gevp_basis_channels(_gevp_family)
                    _gevp_has_series = any(
                        (
                            ch in gevp_input_results
                            and gevp_input_results[ch] is not None
                            and int(gevp_input_results[ch].n_samples) > 0
                            and int(gevp_input_results[ch].series.numel()) > 0
                        )
                        for ch in _gevp_basis
                    )
                    if _gevp_has_series:
                        try:
                            _gevp_payload = compute_companion_channel_gevp(
                                base_results=gevp_input_results,
                                multiscale_output=multiscale_output,
                                config=gevp_cfg,
                                base_channel=_gevp_family,
                            )
                            results[_gevp_payload.result.channel_name] = _gevp_payload.result
                        except Exception as exc:
                            if _gevp_family == "su2":
                                gevp_error = str(exc)

                state["new_dirac_ew_results"] = results
                state["new_dirac_ew_multiscale_output"] = multiscale_output
                state["new_dirac_ew_multiscale_error"] = multiscale_error
                state["new_dirac_ew_gevp_error"] = gevp_error

                _update_electroweak_plots_generic(
                    results,
                    new_dirac_ew_channel_plots,
                    new_dirac_ew_plots_spectrum,
                    new_dirac_ew_plots_overlay_corr,
                    new_dirac_ew_plots_overlay_meff,
                )
                _update_new_dirac_ew_electroweak_tables(
                    results,
                    comparison_channel_overrides=comparison_overrides,
                    ratio_specs=ratio_specs,
                )
                _update_new_dirac_ew_multiscale_views(
                    multiscale_output,
                    original_results=base_results,
                    error=multiscale_error,
                )

                pairwise_distance_by_frame = _resolve_electroweak_geodesic_matrices(
                    history,
                    ew_output.frame_indices,
                    state,
                    method=str(new_dirac_ew_settings.kernel_distance_method),
                    edge_weight_mode=str(new_dirac_ew_settings.edge_weight_mode),
                    assume_all_alive=bool(new_dirac_ew_settings.kernel_assume_all_alive),
                )
                couplings = _compute_coupling_constants(
                    history,
                    h_eff=float(new_dirac_ew_settings.h_eff),
                    frame_indices=ew_output.frame_indices,
                    lambda_alg=new_dirac_ew_settings.lambda_alg,
                    pairwise_distance_by_frame=pairwise_distance_by_frame,
                )
                new_dirac_ew_coupling_table.value = pd.DataFrame(
                    _build_coupling_rows(
                        couplings,
                        proxies=None,
                        include_strong=False,
                        refs=_extract_coupling_refs(new_dirac_ew_coupling_ref_table),
                    )
                )

                n_channels = len([r for r in results.values() if r.n_samples > 0])
                summary_lines = [
                    "## Electroweak Summary",
                    f"- Frames used: `{len(ew_output.frame_indices)}`",
                    f"- Electroweak channels with samples: `{n_channels}`",
                    f"- Neighbor source mode: `{new_dirac_ew_settings.neighbor_method}`",
                    (
                        "- Companion routing (U(1), SU(2), EW mixed): "
                        f"`{new_dirac_ew_settings.companion_topology_u1}`, "
                        f"`{new_dirac_ew_settings.companion_topology_su2}`, "
                        f"`{new_dirac_ew_settings.companion_topology_ew_mixed}`"
                    ),
                    f"- SU(2) operator mode: `{new_dirac_ew_settings.su2_operator_mode}`",
                    (
                        "- Walker-type split: "
                        f"`{'on' if new_dirac_ew_settings.enable_walker_type_split else 'off'}`"
                    ),
                    (
                        "- SU(2) multiscale kernels: "
                        f"`{'on' if new_dirac_ew_settings.use_multiscale_kernels else 'off'}`"
                    ),
                    (
                        "- SU(2) GEVP: "
                        f"`{'on' if new_dirac_ew_settings.use_su2_gevp else 'off'}`"
                    ),
                ]
                eps_distance_em = float(couplings.get("eps_distance_emergent", float("nan")))
                eps_clone_em = float(couplings.get("eps_clone_emergent", float("nan")))
                eps_fitness_gap_em = float(couplings.get("eps_fitness_gap_emergent", float("nan")))
                eps_geodesic_em = float(couplings.get("eps_geodesic_emergent", float("nan")))
                sin2_theta_w_em = float(couplings.get("sin2_theta_w_emergent", float("nan")))
                tan_theta_w_em = float(couplings.get("tan_theta_w_emergent", float("nan")))
                if np.isfinite(eps_distance_em):
                    summary_lines.append(f"- ε_distance emergent: `{eps_distance_em:.6g}`")
                if np.isfinite(eps_clone_em):
                    summary_lines.append(f"- ε_clone emergent: `{eps_clone_em:.6g}`")
                if np.isfinite(eps_geodesic_em):
                    summary_lines.append(f"- ε_geodesic emergent: `{eps_geodesic_em:.6g}`")
                if np.isfinite(eps_fitness_gap_em):
                    summary_lines.append(f"- ε_fitness_gap emergent: `{eps_fitness_gap_em:.6g}`")
                if np.isfinite(sin2_theta_w_em):
                    theta_w_deg = float(np.degrees(np.arcsin(np.sqrt(sin2_theta_w_em))))
                    summary_lines.append(
                        f"- sin²θ_W emergent: `{sin2_theta_w_em:.6g}` "
                        f"(θ_W ≈ {theta_w_deg:.1f}°, "
                        f"observed@M_Z: 0.231, GUT: 0.375)"
                    )
                if np.isfinite(tan_theta_w_em):
                    summary_lines.append(f"- tanθ_W emergent: `{tan_theta_w_em:.6g}`")
                if multiscale_error:
                    summary_lines.append(f"- Multiscale error: `{multiscale_error}`")
                if gevp_error:
                    summary_lines.append(f"- GEVP error: `{gevp_error}`")
                new_dirac_ew_summary.object = "\n".join(summary_lines)

                error_parts = []
                if multiscale_error:
                    error_parts.append(f"multiscale: `{multiscale_error}`")
                if gevp_error:
                    error_parts.append(f"GEVP: `{gevp_error}`")
                if error_parts:
                    new_dirac_ew_status.object = "**Complete with errors:** " + "; ".join(
                        error_parts
                    )
                else:
                    new_dirac_ew_status.object = (
                        f"**Complete:** Electroweak computed ({n_channels} channels)."
                    )

            _run_tab_computation(
                state,
                new_dirac_ew_status,
                "electroweak observables",
                _compute,
            )

        def on_run_new_dirac(_):
            def _compute(history):
                ew_cfg = _build_new_dirac_ew_channel_config()
                requested_electroweak_channels = _collect_multiselect_values(
                    new_dirac_ew_channel_family_selectors
                )
                if not requested_electroweak_channels:
                    requested_electroweak_channels = list(DEFAULT_ELECTROWEAK_CHANNELS_FOR_DIRAC)
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
                    electroweak_channels=requested_electroweak_channels,
                    dirac=dirac_cfg,
                    color_singlet_quantile=float(new_dirac_ew_settings.color_singlet_quantile),
                    sigma_max_lag=int(new_dirac_ew_settings.max_lag),
                    sigma_use_connected=bool(new_dirac_ew_settings.use_connected),
                    sigma_fit_mode=str(new_dirac_ew_settings.fit_mode),
                    sigma_fit_start=int(new_dirac_ew_settings.fit_start),
                    sigma_fit_stop=new_dirac_ew_settings.fit_stop,
                    sigma_min_fit_points=int(new_dirac_ew_settings.min_fit_points),
                    sigma_window_widths=_parse_window_widths(
                        new_dirac_ew_settings.window_widths_spec
                    ),
                    sigma_compute_bootstrap_errors=bool(
                        new_dirac_ew_settings.compute_bootstrap_errors
                    ),
                    sigma_n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                )
                bundle = compute_dirac_electroweak_bundle(history, config=bundle_cfg)
                state["new_dirac_bundle"] = bundle
                state["new_dirac_ew_bundle"] = bundle

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
                sector_counts = {
                    name: spec.n_walkers for name, spec in bundle.dirac_result.sectors.items()
                }
                new_dirac_ew_dirac_summary.object = "  \n".join([
                    (
                        f"**Best-fit scale:** {best_scale:.6g} GeV/σ"
                        if best_scale
                        else "**Best-fit scale:** N/A"
                    ),
                    f"**Chiral condensate:** ⟨ψ̄ψ⟩ ≈ {bundle.dirac_result.chiral_condensate:.4f}",
                    "**Sector walkers:** "
                    + ", ".join(f"{k}: {v}" for k, v in sector_counts.items()),
                ])

                color_singlet = bundle.color_singlet_spectrum
                if color_singlet is None or len(color_singlet.masses) == 0:
                    new_dirac_ew_color_singlet_table.value = pd.DataFrame()
                else:
                    rows = []
                    max_rows = min(len(color_singlet.masses), 300)
                    for i in range(max_rows):
                        ls = float(color_singlet.lepton_scores[i])
                        rows.append({
                            "mode": int(color_singlet.mode_index[i]),
                            "mass": float(color_singlet.masses[i]),
                            "lepton_score": ls,
                            "quark_score": float(color_singlet.quark_scores[i]),
                            "is_singlet": bool(ls >= color_singlet.lepton_threshold),
                        })
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
                new_dirac_status.object = "**Complete:** Dirac analysis computed."

            _run_tab_computation(
                state,
                new_dirac_status,
                "dirac observables",
                _compute,
            )

        def on_run_coupling_diagnostics(_):
            def _compute(history):
                color_dims = _parse_dims_spec(
                    coupling_diagnostics_settings.color_dims_spec, history.d
                )
                cfg = CouplingDiagnosticsConfig(
                    warmup_fraction=float(coupling_diagnostics_settings.simulation_range[0]),
                    end_fraction=float(coupling_diagnostics_settings.simulation_range[1]),
                    h_eff=float(coupling_diagnostics_settings.h_eff),
                    mass=float(coupling_diagnostics_settings.mass),
                    ell0=(
                        float(coupling_diagnostics_settings.ell0)
                        if coupling_diagnostics_settings.ell0 is not None
                        else None
                    ),
                    color_dims=tuple(color_dims) if color_dims is not None else None,
                    companion_topology=str(coupling_diagnostics_settings.companion_topology),
                    pair_weighting=str(coupling_diagnostics_settings.pair_weighting),
                    eps=float(coupling_diagnostics_settings.eps),
                    enable_kernel_diagnostics=bool(
                        coupling_diagnostics_settings.enable_kernel_diagnostics
                    ),
                    edge_weight_mode=str(coupling_diagnostics_settings.edge_weight_mode),
                    n_scales=int(coupling_diagnostics_settings.n_scales),
                    kernel_type=str(coupling_diagnostics_settings.kernel_type),
                    kernel_distance_method=str(
                        coupling_diagnostics_settings.kernel_distance_method
                    ),
                    kernel_assume_all_alive=bool(
                        coupling_diagnostics_settings.kernel_assume_all_alive
                    ),
                    kernel_scale_frames=int(coupling_diagnostics_settings.kernel_scale_frames),
                    kernel_scale_q_low=float(coupling_diagnostics_settings.kernel_scale_q_low),
                    kernel_scale_q_high=float(coupling_diagnostics_settings.kernel_scale_q_high),
                    kernel_max_scale_samples=int(
                        coupling_diagnostics_settings.kernel_max_scale_samples
                    ),
                    kernel_min_scale=float(coupling_diagnostics_settings.kernel_min_scale),
                )
                output = compute_coupling_diagnostics(history, config=cfg)
                state["coupling_diagnostics_output"] = output

                frame_indices = output.frame_indices
                recorded = np.asarray(history.recorded_steps, dtype=float)
                if frame_indices and recorded.size > max(frame_indices):
                    step_axis = recorded[np.asarray(frame_indices, dtype=int)]
                elif frame_indices:
                    step_axis = _history_transition_steps(history, len(frame_indices))
                else:
                    step_axis = np.zeros(0, dtype=float)

                summary_df = _build_coupling_diagnostics_summary_table(output.summary)
                frame_df = _build_coupling_diagnostics_frame_table(step_axis, output)
                scale_df = _build_coupling_diagnostics_scale_table(output)
                plots = _build_coupling_diagnostics_plots(step_axis, output)
                kernel_plots = _build_coupling_diagnostics_kernel_plots(output)

                coupling_diagnostics_summary_table.value = summary_df
                coupling_diagnostics_frame_table.value = frame_df
                coupling_diagnostics_scale_table.value = scale_df
                coupling_diagnostics_phase_plot.object = plots["phase"]
                coupling_diagnostics_regime_plot.object = plots["regime"]
                coupling_diagnostics_fields_plot.object = plots["fields"]
                coupling_diagnostics_coverage_plot.object = plots["coverage"]
                coupling_diagnostics_scale_plot.object = kernel_plots["scale"]
                coupling_diagnostics_running_plot.object = kernel_plots["running"]

                evidence = [
                    str(item) for item in (output.regime_evidence or []) if str(item).strip()
                ]
                if evidence:
                    coupling_diagnostics_regime_evidence.object = "\n".join(
                        ["### Regime Evidence"] + [f"- {line}" for line in evidence]
                    )
                else:
                    coupling_diagnostics_regime_evidence.object = (
                        "_Regime evidence unavailable for this run._"
                    )

                n_frames = int(output.summary.get("n_frames", 0.0) or 0)
                r_circ = output.summary.get("r_circ_mean")
                asym = output.summary.get("re_im_asymmetry_mean")
                coherence = output.summary.get("local_phase_coherence_mean")
                drift_sigma = output.summary.get("phase_drift_sigma")
                sigma = output.summary.get("string_tension_sigma")
                poly = output.summary.get("polyakov_abs")
                xi = output.summary.get("screening_length_xi")
                running_slope = output.summary.get("running_coupling_slope")
                flux_std = output.summary.get("topological_flux_std")
                regime_score = output.summary.get("regime_score")
                snapshot = output.snapshot_frame_index

                def _fmt(value: Any, digits: int = 6) -> str:
                    if value is None:
                        return "n/a"
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        return "n/a"
                    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"

                coupling_diagnostics_summary.object = "\n".join([
                    "## Coupling Diagnostics Summary",
                    f"- Frames analyzed: `{n_frames}`",
                    f"- Mean R_circ: `{_fmt(r_circ)}`",
                    f"- Mean Re/Im asymmetry: `{_fmt(asym)}`",
                    f"- Mean local phase coherence: `{_fmt(coherence)}`",
                    f"- Phase drift significance: `{_fmt(drift_sigma, 3)}σ`",
                    f"- String tension proxy σ: `{_fmt(sigma, 6)}`",
                    f"- Polyakov loop |L|: `{_fmt(poly, 6)}`",
                    f"- Screening length ξ: `{_fmt(xi, 6)}`",
                    f"- Running coupling slope: `{_fmt(running_slope, 6)}`",
                    f"- Topological flux std: `{_fmt(flux_std, 6)}`",
                    f"- Regime score: `{_fmt(regime_score, 2)}` / 10",
                    (
                        f"- Kernel snapshot frame index: `{int(snapshot)}`"
                        if snapshot is not None
                        else "- Kernel snapshot frame index: `n/a`"
                    ),
                    "- This tab computes only fast regime diagnostics (no channel masses).",
                ])
                coupling_diagnostics_status.object = (
                    f"**Complete:** Coupling diagnostics computed ({n_frames} frames). "
                    f"Kernel scales: {int(output.scales.numel())}."
                )

            _run_tab_computation(
                state,
                coupling_diagnostics_status,
                "coupling diagnostics",
                _compute,
            )

        browse_button.on_click(_on_browse_clicked)
        load_button.on_click(on_load_clicked)
        save_button.on_click(on_save_clicked)
        gas_config.add_completion_callback(on_simulation_complete)
        gas_config.param.watch(on_bounds_change, "bounds_extent")
        algorithm_run_button.on_click(on_run_algorithm_analysis)
        fractal_set_run_button.on_click(on_run_fractal_set)
        einstein_run_button.on_click(on_run_einstein_test)
        anisotropic_edge_run_button.on_click(on_run_anisotropic_edge_channels)
        companion_strong_force_run_button.on_click(on_run_companion_strong_force_channels)
        companion_strong_force_display_plots_button.on_click(
            _on_companion_strong_force_display_plots_click
        )
        tensor_calibration_run_button.on_click(on_run_tensor_calibration)
        new_dirac_ew_run_button.on_click(on_run_new_dirac_electroweak)
        new_dirac_run_button.on_click(on_run_new_dirac)
        coupling_diagnostics_run_button.on_click(on_run_coupling_diagnostics)

        visualization_controls = pn.Param(
            visualizer,
            parameters=["point_size", "point_alpha", "color_metric", "fix_axes"],
            show_name=False,
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
                pn.pane.Markdown("### Mean Walker Speed Over Time (p95 error bars)"),
                algorithm_velocity_plot,
                pn.pane.Markdown("### Mean Geodesic Edge Distance Over Time (mean ± 1σ)"),
                algorithm_geodesic_plot,
                pn.pane.Markdown("### Riemannian Kernel Volume Weights Over Time (mean ± 1σ)"),
                algorithm_rkv_plot,
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
                pn.pane.Markdown(
                    "## Einstein Equation Test: G_uv + \u039b g_uv = 8\u03c0 G_N T_uv"
                ),
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

            anisotropic_edge_note = pn.pane.Alert(
                """**Anisotropic Edge Channels:** Computes MC-time correlators from direct
recorded Delaunay neighbors only (no tessellation recomputation). This tab is
now non-companion only; companion-pair/triplet operators are available in the
dedicated **Companion Strong Force** tab. Tensor estimator comparison and
calibration are available in the dedicated **Tensor Calibration** tab.""",
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
                anisotropic_edge_multiscale_summary,
                pn.pane.Markdown("### Multiscale Mass Selection"),
                anisotropic_edge_multiscale_table,
                anisotropic_edge_multiscale_plot,
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
                pn.pane.Markdown("### Glueball Lorentz Check (4 Estimators + Dispersion)"),
                anisotropic_edge_glueball_lorentz_ratio,
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

            companion_strong_force_note = pn.pane.Alert(
                """**Companion Strong Force Channels:** Computes only companion-based operators
and keeps them fully decoupled from anisotropic direct-edge estimators. This tab
uses companion triplets/pairs for baryon, meson, vector/axial, glueball color
plaquette, and a single companion tensor channel, with independent settings and execution.""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            companion_strong_force_tab = pn.Column(
                companion_strong_force_status,
                companion_strong_force_note,
                pn.Row(
                    companion_strong_force_run_button,
                    companion_strong_force_display_plots_button,
                    sizing_mode="stretch_width",
                ),
                companion_strong_force_plot_gate_note,
                pn.Accordion(
                    ("Companion Strong Force Settings", companion_strong_force_settings_layout),
                    (
                        "Reference Anchors",
                        pn.Column(
                            companion_strong_force_glueball_ref_input,
                            pn.pane.Markdown("### Observed Mass Anchors (GeV)"),
                            companion_strong_force_ref_table,
                        ),
                    ),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                companion_strong_force_summary,
                companion_strong_force_multiscale_summary,
                pn.pane.Markdown("### Multiscale Mass Selection"),
                pn.pane.Markdown(
                    "_Top table reports best-scale masses and their delta versus the original "
                    "non-smoothed companion estimators. Per-scale table includes all scales plus "
                    "a `full_original_no_threshold` row for each channel._"
                ),
                companion_strong_force_multiscale_table,
                pn.pane.Markdown("### Per-Scale Companion Values (Mass + R²)"),
                companion_strong_force_multiscale_per_scale_table,
                companion_strong_force_multiscale_plot,
                pn.pane.Markdown("### Mass Display Mode"),
                companion_strong_force_mass_mode,
                pn.layout.Divider(),
                pn.layout.Divider(),
                pn.pane.Markdown("### Mass Spectrum"),
                companion_strong_force_plots_spectrum,
                pn.layout.Divider(),
                pn.pane.Markdown("### Extracted Masses"),
                companion_strong_force_mass_table,
                pn.pane.Markdown("### Filtered-Out Candidates"),
                companion_strong_force_filtered_summary,
                companion_strong_force_filtered_mass_table,
                pn.pane.Markdown("### Ratio Tables by Operator Pair"),
                companion_strong_force_ratio_tables,
                pn.pane.Markdown("### Best Global Variant Combinations (Top 5)"),
                companion_strong_force_best_combo_summary,
                companion_strong_force_best_combo_table,
                pn.pane.Markdown("### Mass Ratios (From Best Global Combination)"),
                companion_strong_force_ratio_pane,
                pn.pane.Markdown("### Calibration/PDG Variant Selection"),
                pn.Row(
                    companion_strong_force_variant_pseudoscalar,
                    companion_strong_force_variant_nucleon,
                    companion_strong_force_variant_glueball,
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    companion_strong_force_variant_scalar,
                    companion_strong_force_variant_vector,
                    companion_strong_force_variant_axial_vector,
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown("### Best-Fit Scales"),
                companion_strong_force_fit_table,
                pn.pane.Markdown("### Cross-Channel Ratio Debug (All Variant Pairs)"),
                companion_strong_force_cross_ratio_summary,
                companion_strong_force_cross_ratio_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                companion_strong_force_plateau_plots,
                pn.layout.Divider(),
                pn.pane.Markdown("### Window Heatmaps"),
                pn.Row(
                    companion_strong_force_heatmap_color_metric,
                    companion_strong_force_heatmap_alpha_metric,
                    sizing_mode="stretch_width",
                ),
                companion_strong_force_heatmap_plots,
                sizing_mode="stretch_both",
            )

            tensor_calibration_tab = build_tensor_gevp_calibration_tab_layout(
                _tcw,
                run_button=tensor_calibration_run_button,
                settings_layout=tensor_calibration_settings_layout,
            )

            multiscale_tab = build_multiscale_tab_layout(_msw)

            # Informational alert for time dimension selection

            new_dirac_ew_note = pn.pane.Alert(
                """**Electroweak:** Proxy channels only (no Dirac operators are computed here).
This tab supports SU(2) multiscale kernels, filtering, and GEVP diagnostics.""",
                alert_type="info",
                sizing_mode="stretch_width",
            )

            coupling_diagnostics_note = pn.pane.Alert(
                (
                    "**Quick Coupling Diagnostics:** Vectorized regime metrics from run traces "
                    "(phase concentration, asymmetry, coherence, drift, coverage, plus "
                    "kernel-scale confinement/topology proxies). No mass-channel extraction "
                    "is performed."
                ),
                alert_type="info",
                sizing_mode="stretch_width",
            )

            coupling_diagnostics_tab = pn.Column(
                coupling_diagnostics_status,
                coupling_diagnostics_note,
                pn.Row(coupling_diagnostics_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Diagnostics Settings", coupling_diagnostics_settings_panel),
                    sizing_mode="stretch_width",
                ),
                pn.layout.Divider(),
                coupling_diagnostics_summary,
                coupling_diagnostics_regime_evidence,
                pn.pane.Markdown("### Summary Metrics"),
                coupling_diagnostics_summary_table,
                pn.pane.Markdown("### Per-Frame Metrics"),
                coupling_diagnostics_frame_table,
                pn.pane.Markdown("### Kernel-Scale Metrics"),
                coupling_diagnostics_scale_table,
                pn.layout.Divider(),
                pn.pane.Markdown("### Global Phase Trend"),
                coupling_diagnostics_phase_plot,
                pn.pane.Markdown("### Coupling Regime Metrics"),
                coupling_diagnostics_regime_plot,
                pn.pane.Markdown("### Local Color Field Means"),
                coupling_diagnostics_fields_plot,
                pn.pane.Markdown("### Coverage (Valid Pairs/Walkers)"),
                coupling_diagnostics_coverage_plot,
                pn.pane.Markdown("### Kernel-Scale Curves"),
                coupling_diagnostics_scale_plot,
                pn.pane.Markdown("### Running Coupling / Creutz"),
                coupling_diagnostics_running_plot,
                sizing_mode="stretch_both",
            )

            new_dirac_ew_tab = pn.Column(
                new_dirac_ew_status,
                new_dirac_ew_note,
                pn.Row(new_dirac_ew_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Electroweak Settings", new_dirac_ew_settings_panel),
                    (
                        "Electroweak Channel Family Selection",
                        new_dirac_ew_channel_family_selector_layout,
                    ),
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
                pn.pane.Markdown("### Filtered-Out Candidates"),
                new_dirac_ew_filtered_summary,
                new_dirac_ew_filtered_mass_table,
                new_dirac_ew_ratio_pane,
                new_dirac_ew_ratio_table,
                pn.pane.Markdown("### Best Global Variant Combinations (Top 5)"),
                new_dirac_ew_best_combo_summary,
                new_dirac_ew_best_combo_table,
                pn.pane.Markdown("### Mass Ratios (From Best Global Combination)"),
                new_dirac_ew_best_combo_ratio_pane,
                pn.pane.Markdown("### Electroweak Symmetry Breaking"),
                new_dirac_ew_symmetry_breaking_summary,
                new_dirac_ew_fit_table,
                new_dirac_ew_compare_table,
                new_dirac_ew_anchor_table,
                pn.layout.Divider(),
                new_dirac_ew_multiscale_summary,
                pn.pane.Markdown("### SU(2) Multiscale Best-Scale Selection"),
                new_dirac_ew_multiscale_table,
                pn.pane.Markdown("### SU(2) Per-Scale Candidates"),
                new_dirac_ew_multiscale_per_scale_table,
                pn.pane.Markdown("### SU(2) GEVP Analysis"),
                *build_gevp_dashboard_sections(new_dirac_ew_su2_gevp_widgets),
                pn.pane.Markdown("### U(1) GEVP Analysis"),
                *build_gevp_dashboard_sections(new_dirac_ew_u1_gevp_widgets),
                pn.pane.Markdown("### EW Mixed GEVP Analysis"),
                *build_gevp_dashboard_sections(new_dirac_ew_ew_mixed_gevp_widgets),
                pn.pane.Markdown("### Electroweak Reference Masses (GeV)"),
                new_dirac_ew_ref_table,
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                new_dirac_ew_channel_plots,
                sizing_mode="stretch_both",
            )

            dirac_tab = pn.Column(
                new_dirac_status,
                pn.Row(new_dirac_run_button, sizing_mode="stretch_width"),
                pn.Accordion(
                    ("Dirac Settings", new_dirac_settings_panel),
                    sizing_mode="stretch_width",
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
                pn.Row(
                    new_dirac_ew_electron_plot,
                    new_dirac_ew_sigma_plot,
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown("### Observed Mass Inputs (GeV)"),
                new_dirac_ew_observed_table,
                pn.pane.Markdown("### Derived Observable Comparison"),
                new_dirac_ew_observable_table,
                pn.pane.Markdown("### Cross-Sector Ratios"),
                new_dirac_ew_ratio_table_extra,
                sizing_mode="stretch_both",
            )

            main.objects = [
                pn.Tabs(
                    ("Simulation", simulation_tab),
                    ("Algorithm", algorithm_tab),
                    ("Holographic Principle", fractal_set_tab),
                    ("Strong Force", anisotropic_edge_tab),
                    ("Tensor Calibration", tensor_calibration_tab),
                    ("Companion Strong Force", companion_strong_force_tab),
                    ("Multiscale", multiscale_tab),
                    ("Coupling Diagnostics", coupling_diagnostics_tab),
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
    parser.add_argument("--address", type=str, default="0.0.0.0", help="Bind address")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting QFT Swarm Convergence Dashboard...", flush=True)
    print(
        f"QFT Swarm Convergence Dashboard running at http://{args.address}:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    pn.serve(
        create_app,
        port=args.port,
        address=args.address,
        show=args.open,
        title="QFT Swarm Convergence Dashboard",
        websocket_origin="*",
    )
