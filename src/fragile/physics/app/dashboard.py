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
from fragile.fractalai.qft.dashboard.gevp_dashboard import (
    build_gevp_dashboard_sections,
    clear_gevp_dashboard,
    create_gevp_dashboard_widgets,
    update_gevp_dashboard,
)
from fragile.fractalai.qft.dashboard.gevp_mass_dashboard import (
    build_gevp_mass_spectrum_sections,
    clear_gevp_mass_spectrum,
    create_gevp_mass_spectrum_widgets,
    update_gevp_mass_spectrum,
)
from fragile.fractalai.qft.electroweak_channels import (
    compute_electroweak_channels,
    compute_electroweak_coupling_constants,
    ELECTROWEAK_CHANNELS,
    ELECTROWEAK_PARITY_CHANNELS,
    ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
    ElectroweakChannelConfig,
)
from fragile.fractalai.qft.gevp_channels import (
    compute_companion_channel_gevp,
    get_companion_gevp_basis_channels,
    GEVPConfig,
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
from fragile.fractalai.qft.tensor_momentum_channels import (
    compute_companion_tensor_momentum_correlator,
    TensorMomentumCorrelatorConfig,
)
from fragile.fractalai.qft.vector_meson_channels import (
    compute_companion_vector_meson_correlator,
    VectorMesonCorrelatorConfig,
)
from fragile.physics.app.algorithm import (
    _algorithm_placeholder_plot,
    build_algorithm_diagnostics_tab,
)
from fragile.physics.app.gas_config_panel import GasConfigPanel
from fragile.physics.app.diagnostics import build_coupling_diagnostics_tab
from fragile.physics.app.gravity import build_holographic_principle_tab
from fragile.physics.fractal_gas.history import RunHistory


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
    gevp_compute_multimode = param.Boolean(
        default=False,
        doc="Extract multi-mode GEVP mass spectrum (ground + excited states).",
    )
    gevp_tau_diag = param.Integer(
        default=None,
        bounds=(2, None),
        allow_None=True,
        doc="Fixed τ_diag for multi-mode GEVP (None = auto-select).",
    )
    gevp_plateau_min_length = param.Integer(
        default=3,
        bounds=(2, None),
        doc="Minimum plateau length for multi-mode mass extraction.",
    )
    gevp_plateau_max_slope = param.Number(
        default=0.3,
        bounds=(0.0, None),
        doc="Maximum relative slope for plateau acceptance.",
    )
    gevp_t0_sweep_spec = param.String(
        default="",
        doc="t0 sweep: '2-6' or '2,3,5'. Empty=disabled.",
    )
    gevp_multimode_fit_start = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Exp fit start lag (None=auto).",
    )
    gevp_multimode_fit_stop = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Exp fit stop lag (None=all).",
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
    gevp_compute_multimode = param.Boolean(
        default=False,
        doc="Extract multi-mode GEVP mass spectrum (ground + excited states).",
    )
    gevp_tau_diag = param.Integer(
        default=None,
        bounds=(2, None),
        allow_None=True,
        doc="Fixed τ_diag for multi-mode GEVP (None = auto-select).",
    )
    gevp_plateau_min_length = param.Integer(
        default=3,
        bounds=(2, None),
        doc="Minimum plateau length for multi-mode mass extraction.",
    )
    gevp_plateau_max_slope = param.Number(
        default=0.3,
        bounds=(0.0, None),
        doc="Maximum relative slope for plateau acceptance.",
    )
    gevp_t0_sweep_spec = param.String(
        default="",
        doc="t0 sweep: '2-6' or '2,3,5'. Empty=disabled.",
    )
    gevp_multimode_fit_start = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Exp fit start lag (None=auto).",
    )
    gevp_multimode_fit_stop = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Exp fit stop lag (None=all).",
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


def _parse_t0_sweep_spec(spec: str) -> list[int] | None:
    """Parse '2-6' or '2,3,5' into list of ints, or None if empty."""
    spec = str(spec).strip()
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        parts = spec.split("-")
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
                return list(range(start, end + 1))
            except ValueError:
                return None
    try:
        values = [int(x.strip()) for x in spec.split(",") if x.strip()]
        return values or None
    except ValueError:
        return None


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
        raise ValueError(f"baryon_color_dims_spec must contain exactly 3 dims; received {dims}.")
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
            mode_n = round(float(momentum_modes[mode_idx].item()))
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
        neighbor_method="recorded",
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
        mode_n = round(float(momentum_modes[mode_idx].item()))

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


def _compute_coupling_constants(
    history: RunHistory | None,
    h_eff: float,
    frame_indices: list[int] | None = None,
    pairwise_distance_by_frame: dict[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    return compute_electroweak_coupling_constants(
        history,
        h_eff=h_eff,
        frame_indices=frame_indices,
        lambda_alg=0.0,
        pairwise_distance_by_frame=pairwise_distance_by_frame,
    )


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


# NOTE: _build_strong_coupling_rows and _build_coupling_diagnostics_* functions
# have been moved to fragile.physics.app.diagnostics.


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


def _get_channel_bootstrap_mass_error(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> float:
    if mode == "Best Window":
        return float("nan")
    raw = result.mass_fit.get("bootstrap_mass_error", float("nan"))
    try:
        bootstrap_error = float(raw)
    except (TypeError, ValueError):
        bootstrap_error = float("nan")
    return (
        bootstrap_error if np.isfinite(bootstrap_error) and bootstrap_error >= 0 else float("nan")
    )


def _get_channel_error_components(
    result: ChannelCorrelatorResult,
    mode: str = "AIC-Weighted",
) -> tuple[float, float, float]:
    """Return (aic_error, bootstrap_error, total_error)."""
    total_error = float(_get_channel_mass_error(result, mode))
    if not np.isfinite(total_error) or total_error < 0:
        total_error = float("nan")
    bootstrap_error = float(_get_channel_bootstrap_mass_error(result, mode))

    if np.isfinite(total_error):
        if np.isfinite(bootstrap_error):
            aic_sq = max(total_error**2 - bootstrap_error**2, 0.0)
            aic_error = float(np.sqrt(aic_sq))
        else:
            aic_error = float(total_error)
    else:
        aic_error = float("nan")
    return aic_error, bootstrap_error, total_error


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
    include_error_breakdown: bool = False,
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
        aic_error = float("nan")
        bootstrap_error = float("nan")
        total_error = float("nan")
        if include_error_breakdown:
            aic_error, bootstrap_error, total_error = _get_channel_error_components(result, mode)
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
        row = {
            "channel": name,
            "mass": f"{mass:.6f}" if mass > 0 else "n/a",
            "mass_error": f"{mass_error:.6f}" if mass_error < float("inf") else "n/a",
            "mass_error_pct": (f"{mass_error_pct:.2f}%" if np.isfinite(mass_error_pct) else "n/a"),
            "r2": f"{r2:.4f}" if np.isfinite(r2) else "n/a",
            "n_windows": n_windows,
            "n_samples": result.n_samples,
        }
        if include_error_breakdown:
            total_error_pct = (
                abs(total_error / float(mass)) * 100.0
                if np.isfinite(total_error) and np.isfinite(float(mass)) and float(mass) > 0
                else float("nan")
            )
            row["aic_error"] = f"{aic_error:.6f}" if np.isfinite(aic_error) else "n/a"
            row["bootstrap_error"] = (
                f"{bootstrap_error:.6f}" if np.isfinite(bootstrap_error) else "n/a"
            )
            row["total_error"] = f"{total_error:.6f}" if np.isfinite(total_error) else "n/a"
            row["total_error_pct"] = (
                f"{total_error_pct:.2f}%" if np.isfinite(total_error_pct) else "n/a"
            )
        rows.append(row)
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
    include_error_breakdown: bool = False,
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
        results,
        mass_table,
        mode,
        mass_getter=mass_getter,
        error_getter=error_getter,
        include_error_breakdown=include_error_breakdown,
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
        gas_config = GasConfigPanel.create_qft_config(bounds_extent=30.0)
        # Override with the best stable calibration settings found in QFT tuning.
        # This matches weak_potential_fit1_aniso_stable2 (200 walkers, 300 steps).
        gas_config.n_steps = 750
        gas_config.gas_params["N"] = 500
        gas_config.gas_params["dtype"] = "float32"
        gas_config.gas_params["clone_every"] = 1
        gas_config.neighbor_weight_modes = [
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ]
        gas_config.init_offset = 0.0
        gas_config.init_spread = 0.0
        gas_config.init_velocity_scale = 0.0

        # Kinetic operator (Langevin + viscous coupling).
        gas_config.kinetic_op.gamma = 1.0
        gas_config.kinetic_op.beta = 1.0
        gas_config.kinetic_op.auto_thermostat = True
        gas_config.kinetic_op.delta_t = 0.01
        gas_config.kinetic_op.sigma_v = 1.0
        gas_config.kinetic_op.nu = 1.0
        gas_config.kinetic_op.beta_curl = 1.0
        gas_config.kinetic_op.use_viscous_coupling = True
        gas_config.kinetic_op.viscous_neighbor_weighting = "riemannian_kernel_volume"
        gas_config.kinetic_op.viscous_length_scale = 1.0

        # Cloning operator.
        gas_config.cloning.p_max = 1.0
        gas_config.cloning.epsilon_clone = 1e-6
        gas_config.cloning.sigma_x = 0.01
        gas_config.cloning.alpha_restitution = 1.0

        # Fitness operator.
        gas_config.fitness_op.alpha = 1.0
        gas_config.fitness_op.beta = 1.0
        gas_config.fitness_op.eta = 0.0
        gas_config.fitness_op.sigma_min = 0.0
        gas_config.fitness_op.A = 2.0
        visualizer = SwarmConvergence3D(history=None, bounds_extent=gas_config.bounds_extent)

        state: dict[str, Any] = {
            "history": None,
            "history_path": None,
            "fractal_set_points": None,
            "fractal_set_regressions": None,
            "fractal_set_frame_summary": None,
            "companion_strong_force_results": None,
            "companion_strong_force_multiscale_output": None,
            "companion_strong_force_multiscale_error": None,
            "companion_strong_force_gevp_error": None,
            "companion_strong_force_plots_unlocked": False,
            "new_dirac_ew_bundle": None,
            "new_dirac_ew_results": None,
            "new_dirac_ew_multiscale_output": None,
            "new_dirac_ew_multiscale_error": None,
            "new_dirac_ew_gevp_error": None,
            "new_dirac_ew_comparison_overrides": None,
            "new_dirac_ew_ratio_specs": None,
            "_multiscale_geodesic_distance_by_frame": None,
            "_multiscale_geodesic_distribution": None,
            "coupling_diagnostics_output": None,
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
        simulation_compute_status = pn.pane.Markdown(
            "**Simulation:** run or load a RunHistory, then click Compute Simulation.",
            sizing_mode="stretch_width",
        )
        simulation_compute_button = pn.widgets.Button(
            name="Compute Simulation",
            button_type="primary",
            min_width=240,
            sizing_mode="stretch_width",
            disabled=True,
        )

        algorithm_section = build_algorithm_diagnostics_tab(state)

        # =====================================================================
        # Holographic principle sections (moved to gravity.py)
        # =====================================================================
        # Built after NewDirac settings are initialized below.

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
        # New Dirac/Electroweak unified tab components
        # =====================================================================
        new_dirac_ew_settings = NewDiracElectroweakSettings()
        new_dirac_ew_status = pn.pane.Markdown(
            "**Electroweak:** Load a RunHistory and click Compute.",
            sizing_mode="stretch_width",
        )

        holographic_section = build_holographic_principle_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
            new_dirac_ew_settings=new_dirac_ew_settings,
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
                "edge_weight_mode",
                "neighbor_k",
                "window_widths_spec",
                "fit_mode",
                "fit_start",
                "fit_stop",
                "min_fit_points",
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
        new_dirac_ew_su2_mass_spectrum_widgets = create_gevp_mass_spectrum_widgets()
        new_dirac_ew_u1_mass_spectrum_widgets = create_gevp_mass_spectrum_widgets()
        new_dirac_ew_ew_mixed_mass_spectrum_widgets = create_gevp_mass_spectrum_widgets()

        # --- GEVP Mass Spectrum tab widgets (consolidated view of all 7 families) ---
        gevp_tab_status = pn.pane.Markdown(
            "**GEVP Mass Spectrum:** _run Strong Force or Electroweak analysis to populate._",
            sizing_mode="stretch_width",
        )
        gevp_tab_nucleon_ms_widgets = create_gevp_mass_spectrum_widgets()
        gevp_tab_scalar_ms_widgets = create_gevp_mass_spectrum_widgets()
        gevp_tab_pseudoscalar_ms_widgets = create_gevp_mass_spectrum_widgets()
        gevp_tab_glueball_ms_widgets = create_gevp_mass_spectrum_widgets()
        gevp_tab_su2_ms_widgets = create_gevp_mass_spectrum_widgets()
        gevp_tab_u1_ms_widgets = create_gevp_mass_spectrum_widgets()
        gevp_tab_ew_mixed_ms_widgets = create_gevp_mass_spectrum_widgets()

        coupling_section = build_coupling_diagnostics_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
            parse_dims_spec=_parse_dims_spec,
        )

        def set_history(
            history: RunHistory,
            history_path: Path | None = None,
            defer_dashboard_updates: bool = False,
        ) -> None:
            state["history"] = history
            state["history_path"] = history_path
            state["_multiscale_geodesic_distance_by_frame"] = None
            state["_multiscale_geodesic_distribution"] = None
            state["new_dirac_ew_comparison_overrides"] = None
            state["new_dirac_ew_ratio_specs"] = None
            if not defer_dashboard_updates:
                visualizer.bounds_extent = float(gas_config.bounds_extent)
                visualizer.set_history(history)
            algorithm_section.on_history_ready()
            simulation_compute_button.disabled = False
            simulation_compute_status.object = (
                "**Simulation:** click Compute Simulation to visualize this RunHistory."
            )
            if defer_dashboard_updates:
                visualizer.status_pane.object = (
                    "**Simulation complete:** history captured; click a Compute button to "
                    "run post-processing."
                )
                save_button.disabled = False
                save_status.object = "**Save a history**: choose a path and click Save."
                holographic_section.on_history_changed(defer_dashboard_updates)
                companion_strong_force_run_button.disabled = False
                companion_strong_force_display_plots_button.disabled = True
                companion_strong_force_display_plots_button.button_type = "default"
                state["companion_strong_force_plots_unlocked"] = False
                companion_strong_force_plot_gate_note.object = (
                    "_Plots are hidden. Click `Display Plots` after computing channels._"
                )
                companion_strong_force_status.object = "**Companion Strong Force ready:** click Compute Companion Strong Force Channels."
                new_dirac_ew_run_button.disabled = False
                new_dirac_ew_status.object = "**Electroweak ready:** click Compute Electroweak."
                coupling_section.on_history_changed(True)
                return
            algorithm_section.reset_plots()
            save_button.disabled = False
            save_status.object = "**Save a history**: choose a path and click Save."
            holographic_section.on_history_changed(defer_dashboard_updates)
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
            new_dirac_ew_run_button.disabled = False
            new_dirac_ew_status.object = "**Electroweak ready:** click Compute Electroweak."
            new_dirac_ew_summary.object = "## Electroweak Summary\n_Run analysis to populate._"
            new_dirac_ew_multiscale_summary.object = "### SU(2) Multiscale Summary\n_Multiscale kernels disabled (original estimators only)._"
            clear_gevp_dashboard(new_dirac_ew_su2_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_u1_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_ew_mixed_gevp_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_su2_mass_spectrum_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_u1_mass_spectrum_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_ew_mixed_mass_spectrum_widgets)
            # Clear consolidated GEVP Mass Spectrum tab
            for gevp_tab_w in (
                gevp_tab_nucleon_ms_widgets,
                gevp_tab_scalar_ms_widgets,
                gevp_tab_pseudoscalar_ms_widgets,
                gevp_tab_glueball_ms_widgets,
                gevp_tab_su2_ms_widgets,
                gevp_tab_u1_ms_widgets,
                gevp_tab_ew_mixed_ms_widgets,
            ):
                clear_gevp_mass_spectrum(gevp_tab_w)
            gevp_tab_status.object = (
                "**GEVP Mass Spectrum:** _run Strong Force or Electroweak analysis to populate._"
            )
            new_dirac_ew_filtered_summary.object = "**Filtered-out candidates:** none."
            new_dirac_ew_mass_table.value = pd.DataFrame()
            new_dirac_ew_filtered_mass_table.value = pd.DataFrame()
            new_dirac_ew_ratio_table.value = pd.DataFrame()
            new_dirac_ew_fit_table.value = pd.DataFrame()
            new_dirac_ew_compare_table.value = pd.DataFrame()
            new_dirac_ew_anchor_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
            coupling_section.on_history_changed(False)

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

        def on_compute_simulation_clicked(_):
            if gas_config.run_button.disabled:
                simulation_compute_status.object = (
                    "**Simulation:** simulation is currently running.\n\n"
                    "Wait for completion before recomputing visualization."
                )
                return

            history = state.get("history")
            if history is None:
                simulation_compute_status.object = (
                    "**Error:** run a simulation or load a RunHistory first."
                )
                return

            simulation_compute_status.object = "**Computing Simulation...**"
            try:
                inferred_extent = _infer_bounds_extent(history)
                if inferred_extent is not None:
                    visualizer.bounds_extent = inferred_extent
                    gas_config.bounds_extent = float(inferred_extent)
                visualizer.set_history(history)
                simulation_compute_status.object = (
                    f"**Simulation ready:** {history.n_steps} steps / "
                    f"{history.n_recorded} recorded frames."
                )
            except Exception as exc:
                simulation_compute_status.object = f"**Error:** {exc!s}"

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
            if not bool(state.get("companion_strong_force_plots_unlocked")):
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
                include_error_breakdown=True,
            )
            filtered_out_dict = {
                name: payload[0] for name, payload in filtered_out_results.items()
            }
            _update_mass_table(
                filtered_out_dict,
                companion_strong_force_filtered_mass_table,
                str(mode),
                include_error_breakdown=True,
            )
            min_r2_str = f"{gevp_min_r2:.3g}" if np.isfinite(gevp_min_r2) else "off"
            max_error_pct_str = (
                f"{gevp_max_error_pct:.3g}%" if np.isfinite(gevp_max_error_pct) else "off"
            )
            if filtered_out_results:
                preview = ", ".join(
                    (f"{_display_companion_channel_name(name)}({reason})")
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

            def _mass_error_components(
                result_obj: ChannelCorrelatorResult,
            ) -> tuple[float, float, float]:
                """Return (aic_error, bootstrap_error, total_error) for one channel mass."""
                return _get_channel_error_components(result_obj, str(mode))

            def _ratio_sigma(
                *,
                ratio: float,
                numerator_mass: float,
                denominator_mass: float,
                numerator_err: float,
                denominator_err: float,
            ) -> float:
                if (
                    not np.isfinite(ratio)
                    or not np.isfinite(numerator_mass)
                    or not np.isfinite(denominator_mass)
                    or denominator_mass <= 0
                    or not np.isfinite(numerator_err)
                    or not np.isfinite(denominator_err)
                    or numerator_err < 0
                    or denominator_err < 0
                ):
                    return float("nan")
                rel_num = numerator_err / max(numerator_mass, 1e-12)
                rel_den = denominator_err / max(denominator_mass, 1e-12)
                return float(abs(ratio) * np.sqrt(rel_num**2 + rel_den**2))

            def _interval_with_target_status(
                center: float,
                sigma: float,
                target: float,
            ) -> str:
                if not np.isfinite(sigma) or sigma <= 0:
                    return "interval unavailable"
                low = center - sigma
                high = center + sigma
                status = "n/a"
                if np.isfinite(target):
                    status = "yes" if (low <= target <= high) else "no"
                return f"[{low:.3f}, {high:.3f}], target_in={status}"

            variant_options_by_channel: dict[str, list[str]] = {
                channel: _variant_names_for_base(channel) for channel in combo_channels
            }
            missing_combo_channels = [
                channel for channel, options in variant_options_by_channel.items() if not options
            ]
            variant_mass_lookup: dict[str, float] = {}
            variant_aic_error_lookup: dict[str, float] = {}
            variant_bootstrap_error_lookup: dict[str, float] = {}
            variant_total_error_lookup: dict[str, float] = {}
            for variant_name, variant_result in filtered_results.items():
                if not isinstance(variant_result, ChannelCorrelatorResult):
                    continue
                if variant_result.n_samples == 0:
                    continue
                variant_mass = _get_channel_mass(variant_result, str(mode))
                if np.isfinite(variant_mass) and variant_mass > 0:
                    variant_mass_lookup[str(variant_name)] = float(variant_mass)
                    aic_err, boot_err, total_err = _mass_error_components(variant_result)
                    variant_aic_error_lookup[str(variant_name)] = float(aic_err)
                    variant_bootstrap_error_lookup[str(variant_name)] = float(boot_err)
                    variant_total_error_lookup[str(variant_name)] = float(total_err)

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
                    numerator_aic_err = variant_aic_error_lookup.get(
                        numerator_variant, float("nan")
                    )
                    denominator_aic_err = variant_aic_error_lookup.get(
                        denominator_variant, float("nan")
                    )
                    numerator_boot_err = variant_bootstrap_error_lookup.get(
                        numerator_variant, float("nan")
                    )
                    denominator_boot_err = variant_bootstrap_error_lookup.get(
                        denominator_variant, float("nan")
                    )
                    numerator_total_err = variant_total_error_lookup.get(
                        numerator_variant, float("nan")
                    )
                    denominator_total_err = variant_total_error_lookup.get(
                        denominator_variant, float("nan")
                    )
                    ratio_sigma_aic = _ratio_sigma(
                        ratio=ratio_value,
                        numerator_mass=numerator_mass,
                        denominator_mass=denominator_mass,
                        numerator_err=numerator_aic_err,
                        denominator_err=denominator_aic_err,
                    )
                    ratio_sigma_boot = _ratio_sigma(
                        ratio=ratio_value,
                        numerator_mass=numerator_mass,
                        denominator_mass=denominator_mass,
                        numerator_err=numerator_boot_err,
                        denominator_err=denominator_boot_err,
                    )
                    ratio_sigma_total = _ratio_sigma(
                        ratio=ratio_value,
                        numerator_mass=numerator_mass,
                        denominator_mass=denominator_mass,
                        numerator_err=numerator_total_err,
                        denominator_err=denominator_total_err,
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
                    aic_interval = _interval_with_target_status(
                        center=ratio_value,
                        sigma=ratio_sigma_aic,
                        target=ref_value,
                    )
                    boot_interval = _interval_with_target_status(
                        center=ratio_value,
                        sigma=ratio_sigma_boot,
                        target=ref_value,
                    )
                    total_interval = _interval_with_target_status(
                        center=ratio_value,
                        sigma=ratio_sigma_total,
                        target=ref_value,
                    )
                    best_ratio_lines.append(
                        f"- {numerator_base}/{denominator_base}: **{ratio_value:.3f}** "
                        f"({annotation}; total_1sigma {total_interval}; AIC_1sigma {aic_interval}; "
                        f"bootstrap_1sigma {boot_interval})"
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
                    f"### Multiscale Kernel Summary\n- Status: failed\n- Error: `{error}`"
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
            if curves and bool(state.get("companion_strong_force_plots_unlocked")):
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
                                compute_multimode=bool(
                                    companion_strong_force_settings.gevp_compute_multimode
                                ),
                                tau_diag=companion_strong_force_settings.gevp_tau_diag,
                                plateau_min_length=int(
                                    companion_strong_force_settings.gevp_plateau_min_length
                                ),
                                plateau_max_slope=float(
                                    companion_strong_force_settings.gevp_plateau_max_slope
                                ),
                                multimode_fit_start=companion_strong_force_settings.gevp_multimode_fit_start,
                                multimode_fit_stop=companion_strong_force_settings.gevp_multimode_fit_stop,
                                t0_sweep_values=_parse_t0_sweep_spec(
                                    companion_strong_force_settings.gevp_t0_sweep_spec
                                ),
                            )
                            gevp_payload = compute_companion_channel_gevp(
                                base_results=base_results,
                                multiscale_output=multiscale_output,
                                config=gevp_cfg,
                                base_channel=gevp_base_channel,
                            )
                            if gevp_payload.mass_spectrum is not None:
                                state[f"_gevp_mass_spectrum_{gevp_base_channel}"] = (
                                    gevp_payload.mass_spectrum
                                )
                            if gevp_payload.t0_sweep is not None:
                                state[f"_gevp_t0_sweep_{gevp_base_channel}"] = (
                                    gevp_payload.t0_sweep
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
                _update_gevp_mass_spectrum_tab()

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
                reason = "missing variants for: " + ", ".join(f"`{ch}`" for ch in missing_channels)
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
                    per_ratio_error[f"error_{numerator}_over_{denominator}_abs_pct"] = (
                        abs_pct_error
                    )
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
                    if not np.isfinite(num_mass) or not np.isfinite(den_mass) or den_mass <= 0:
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
                    best_ratio_lines.append("- n/a (missing valid masses for selected variants)")
                new_dirac_ew_best_combo_ratio_pane.object = "  \n".join(best_ratio_lines)
                return best_selection
            if total_combinations == 0:
                reason = "no variant combinations available"
            else:
                reason = "no complete variant combination has valid masses for all ratio targets"
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
            for base_channel in (
                list(SU2_BASE_CHANNELS) + list(U1_BASE_CHANNELS) + list(EW_MIXED_BASE_CHANNELS)
            ):
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
            min_r2, min_windows, max_error_pct, remove_artifacts = (
                _resolve_new_dirac_ew_filter_values()
            )
            t0 = int(new_dirac_ew_settings.gevp_t0)
            eig_rel_cutoff = float(new_dirac_ew_settings.gevp_eig_rel_cutoff)
            per_scale = multiscale_output.per_scale_results if multiscale_output else {}
            gevp_results = {k: v for k, v in results.items() if k.endswith("_gevp")}
            for family, widgets, ms_widgets in (
                ("su2", new_dirac_ew_su2_gevp_widgets, new_dirac_ew_su2_mass_spectrum_widgets),
                ("u1", new_dirac_ew_u1_gevp_widgets, new_dirac_ew_u1_mass_spectrum_widgets),
                (
                    "ew_mixed",
                    new_dirac_ew_ew_mixed_gevp_widgets,
                    new_dirac_ew_ew_mixed_mass_spectrum_widgets,
                ),
            ):
                try:
                    update_gevp_dashboard(
                        widgets,
                        selected_channel_name=f"{family.upper()} GEVP",
                        raw_channel_name=f"{family}_companion"
                        if family != "ew_mixed"
                        else "ew_mixed_companion",
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
                # Update multi-mode mass spectrum widgets
                spectrum = state.get(f"_ew_gevp_mass_spectrum_{family}")
                t0_sweep = state.get(f"_ew_gevp_t0_sweep_{family}")
                try:
                    dt_val = (
                        float(new_dirac_ew_settings.dt)
                        if hasattr(new_dirac_ew_settings, "dt")
                        else 1.0
                    )
                except (TypeError, ValueError):
                    dt_val = 1.0
                update_gevp_mass_spectrum(
                    ms_widgets,
                    spectrum=spectrum,
                    family_label=family.upper(),
                    dt=dt_val,
                    t0_sweep=t0_sweep,
                )

        def _update_gevp_mass_spectrum_tab() -> None:
            """Refresh the consolidated GEVP Mass Spectrum tab for all 7 families."""
            strong_families = [
                ("nucleon", "Nucleon", gevp_tab_nucleon_ms_widgets),
                ("scalar", "Scalar", gevp_tab_scalar_ms_widgets),
                ("pseudoscalar", "Pseudoscalar", gevp_tab_pseudoscalar_ms_widgets),
                ("glueball", "Glueball", gevp_tab_glueball_ms_widgets),
            ]
            ew_families = [
                ("su2", "SU(2)", gevp_tab_su2_ms_widgets),
                ("u1", "U(1)", gevp_tab_u1_ms_widgets),
                ("ew_mixed", "EW Mixed", gevp_tab_ew_mixed_ms_widgets),
            ]
            populated = []
            # Strong-force families
            for family_key, label, widgets in strong_families:
                spectrum = state.get(f"_gevp_mass_spectrum_{family_key}")
                t0_sweep = state.get(f"_gevp_t0_sweep_{family_key}")
                update_gevp_mass_spectrum(
                    widgets,
                    spectrum=spectrum,
                    family_label=label,
                    dt=1.0,
                    t0_sweep=t0_sweep,
                )
                if spectrum is not None:
                    populated.append(label)
            # EW families
            try:
                dt_val = (
                    float(new_dirac_ew_settings.dt)
                    if hasattr(new_dirac_ew_settings, "dt")
                    else 1.0
                )
            except (TypeError, ValueError):
                dt_val = 1.0
            for family_key, label, widgets in ew_families:
                spectrum = state.get(f"_ew_gevp_mass_spectrum_{family_key}")
                t0_sweep = state.get(f"_ew_gevp_t0_sweep_{family_key}")
                update_gevp_mass_spectrum(
                    widgets,
                    spectrum=spectrum,
                    family_label=label,
                    dt=dt_val,
                    t0_sweep=t0_sweep,
                )
                if spectrum is not None:
                    populated.append(label)
            if populated:
                gevp_tab_status.object = (
                    f"**GEVP Mass Spectrum:** {', '.join(populated)} populated."
                )
            else:
                gevp_tab_status.object = "**GEVP Mass Spectrum:** _run Strong Force or Electroweak analysis to populate._"

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
            _update_gevp_mass_spectrum_tab()

            # --- Symmetry-breaking derived observables ---
            import math as _math

            sb_lines = ["**Symmetry Breaking Observables:**"]
            ew_masses = _extract_masses(filtered_results, mode=str(mode), family_map=None)
            m_z_proxy = ew_masses.get("fitness_phase")
            m_w_proxy = ew_masses.get("clone_indicator")
            if m_z_proxy and m_z_proxy > 0 and m_w_proxy and m_w_proxy > 0:
                cos_tw = min(m_w_proxy / m_z_proxy, 1.0)
                sin2_tw = 1.0 - cos_tw**2
                sb_lines.append(f"- **M_Z proxy** (fitness_phase): `{m_z_proxy:.6g}`")
                sb_lines.append(f"- **M_W proxy** (clone_indicator): `{m_w_proxy:.6g}`")
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
                sb_lines.append("- _Fitness phase and clone indicator channels not computed._")

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
                    f"### SU(2) Multiscale Summary\n- Status: failed\n- Error: `{error}`"
                )
                new_dirac_ew_multiscale_table.value = pd.DataFrame()
                new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
                clear_gevp_dashboard(new_dirac_ew_su2_gevp_widgets)
                clear_gevp_dashboard(new_dirac_ew_u1_gevp_widgets)
                clear_gevp_dashboard(new_dirac_ew_ew_mixed_gevp_widgets)
                clear_gevp_mass_spectrum(new_dirac_ew_su2_mass_spectrum_widgets)
                clear_gevp_mass_spectrum(new_dirac_ew_u1_mass_spectrum_widgets)
                clear_gevp_mass_spectrum(new_dirac_ew_ew_mixed_mass_spectrum_widgets)
                # Clear EW families in consolidated GEVP tab
                for gevp_tab_w in (
                    gevp_tab_su2_ms_widgets,
                    gevp_tab_u1_ms_widgets,
                    gevp_tab_ew_mixed_ms_widgets,
                ):
                    clear_gevp_mass_spectrum(gevp_tab_w)
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

        new_dirac_ew_mass_mode.param.watch(_on_new_dirac_ew_mass_mode_change, "value")

        def _build_new_dirac_ew_channel_config() -> ElectroweakChannelConfig:
            return ElectroweakChannelConfig(
                warmup_fraction=new_dirac_ew_settings.simulation_range[0],
                end_fraction=new_dirac_ew_settings.simulation_range[1],
                max_lag=new_dirac_ew_settings.max_lag,
                h_eff=new_dirac_ew_settings.h_eff,
                use_connected=new_dirac_ew_settings.use_connected,
                neighbor_method="companions",
                companion_topology="distance",
                companion_topology_u1="distance",
                companion_topology_su2="clone",
                companion_topology_ew_mixed="both",
                edge_weight_mode=new_dirac_ew_settings.edge_weight_mode,
                neighbor_k=new_dirac_ew_settings.neighbor_k,
                window_widths=_parse_window_widths(new_dirac_ew_settings.window_widths_spec),
                fit_mode=new_dirac_ew_settings.fit_mode,
                fit_start=new_dirac_ew_settings.fit_start,
                fit_stop=new_dirac_ew_settings.fit_stop,
                min_fit_points=new_dirac_ew_settings.min_fit_points,
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
                    name for name in requested_electroweak_channels if name in multiscale_supported
                ]
                if bool(new_dirac_ew_settings.use_multiscale_kernels) and multiscale_requested:
                    try:
                        ms_cfg = MultiscaleElectroweakConfig(
                            warmup_fraction=float(new_dirac_ew_settings.simulation_range[0]),
                            end_fraction=float(new_dirac_ew_settings.simulation_range[1]),
                            mc_time_index=new_dirac_ew_settings.mc_time_index,
                            h_eff=float(new_dirac_ew_settings.h_eff),
                            lambda_alg=0.0,
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
                    window_widths=_parse_window_widths(new_dirac_ew_settings.window_widths_spec),
                    basis_strategy=str(new_dirac_ew_settings.gevp_basis_strategy),
                    max_basis=int(new_dirac_ew_settings.gevp_max_basis),
                    min_operator_r2=float(new_dirac_ew_settings.gevp_min_operator_r2),
                    min_operator_windows=int(new_dirac_ew_settings.gevp_min_operator_windows),
                    max_operator_error_pct=float(
                        new_dirac_ew_settings.gevp_max_operator_error_pct
                    ),
                    remove_artifacts=bool(new_dirac_ew_settings.gevp_remove_artifacts),
                    eig_rel_cutoff=float(new_dirac_ew_settings.gevp_eig_rel_cutoff),
                    cond_limit=float(new_dirac_ew_settings.gevp_cond_limit),
                    shrinkage=float(new_dirac_ew_settings.gevp_shrinkage),
                    compute_bootstrap_errors=bool(new_dirac_ew_settings.compute_bootstrap_errors),
                    n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                    bootstrap_mode=str(new_dirac_ew_settings.gevp_bootstrap_mode),
                    compute_multimode=bool(new_dirac_ew_settings.gevp_compute_multimode),
                    tau_diag=new_dirac_ew_settings.gevp_tau_diag,
                    plateau_min_length=int(new_dirac_ew_settings.gevp_plateau_min_length),
                    plateau_max_slope=float(new_dirac_ew_settings.gevp_plateau_max_slope),
                    multimode_fit_start=new_dirac_ew_settings.gevp_multimode_fit_start,
                    multimode_fit_stop=new_dirac_ew_settings.gevp_multimode_fit_stop,
                    t0_sweep_values=_parse_t0_sweep_spec(new_dirac_ew_settings.gevp_t0_sweep_spec),
                )
                for gevp_family, gevp_use_flag in (
                    ("su2", bool(new_dirac_ew_settings.use_su2_gevp)),
                    ("u1", bool(new_dirac_ew_settings.use_u1_gevp)),
                    ("ew_mixed", bool(new_dirac_ew_settings.use_ew_mixed_gevp)),
                ):
                    if not gevp_use_flag:
                        continue
                    gevp_basis = get_companion_gevp_basis_channels(gevp_family)
                    gevp_has_series = any(
                        (
                            ch in gevp_input_results
                            and gevp_input_results[ch] is not None
                            and int(gevp_input_results[ch].n_samples) > 0
                            and int(gevp_input_results[ch].series.numel()) > 0
                        )
                        for ch in gevp_basis
                    )
                    if gevp_has_series:
                        try:
                            gevp_payload = compute_companion_channel_gevp(
                                base_results=gevp_input_results,
                                multiscale_output=multiscale_output,
                                config=gevp_cfg,
                                base_channel=gevp_family,
                            )
                            results[gevp_payload.result.channel_name] = gevp_payload.result
                            if gevp_payload.mass_spectrum is not None:
                                state[f"_ew_gevp_mass_spectrum_{gevp_family}"] = (
                                    gevp_payload.mass_spectrum
                                )
                            if gevp_payload.t0_sweep is not None:
                                state[f"_ew_gevp_t0_sweep_{gevp_family}"] = gevp_payload.t0_sweep
                        except Exception as exc:
                            if gevp_family == "su2":
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

                couplings = _compute_coupling_constants(
                    history,
                    h_eff=float(new_dirac_ew_settings.h_eff),
                    frame_indices=ew_output.frame_indices,
                    pairwise_distance_by_frame=state.get("_multiscale_geodesic_distance_by_frame"),
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
                    "- Neighbor source mode: `companions` (run-selected pairs)",
                    "- Companion routing (U(1), SU(2), EW mixed): `distance`, `clone`, `both`",
                    f"- SU(2) operator mode: `{new_dirac_ew_settings.su2_operator_mode}`",
                    (
                        "- Walker-type split: "
                        f"`{'on' if new_dirac_ew_settings.enable_walker_type_split else 'off'}`"
                    ),
                    (
                        "- SU(2) multiscale kernels: "
                        f"`{'on' if new_dirac_ew_settings.use_multiscale_kernels else 'off'}`"
                    ),
                    (f"- SU(2) GEVP: `{'on' if new_dirac_ew_settings.use_su2_gevp else 'off'}`"),
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

        browse_button.on_click(_on_browse_clicked)
        load_button.on_click(on_load_clicked)
        save_button.on_click(on_save_clicked)
        simulation_compute_button.on_click(on_compute_simulation_clicked)
        gas_config.add_completion_callback(on_simulation_complete)
        gas_config.param.watch(on_bounds_change, "bounds_extent")
        holographic_section.fractal_set_run_button.on_click(holographic_section.on_run_fractal_set)
        companion_strong_force_run_button.on_click(on_run_companion_strong_force_channels)
        companion_strong_force_display_plots_button.on_click(
            _on_companion_strong_force_display_plots_click
        )
        new_dirac_ew_run_button.on_click(on_run_new_dirac_electroweak)

        visualization_controls = pn.Param(
            visualizer,
            parameters=["point_size", "point_alpha", "color_metric", "fix_axes"],
            show_name=False,
        )

        if skip_sidebar:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\nSidebar disabled via QFT_DASH_SKIP_SIDEBAR=1."
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
            simulation_tab = pn.Column(
                simulation_compute_status,
                pn.Row(simulation_compute_button, sizing_mode="stretch_width"),
                visualizer.panel(),
                sizing_mode="stretch_both",
            )

            algorithm_tab = algorithm_section.tab

            fractal_set_tab = holographic_section.fractal_set_tab

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

            # Informational alert for time dimension selection

            new_dirac_ew_note = pn.pane.Alert(
                """**Electroweak:** Proxy channels only (no Dirac operators are computed here).
This tab supports SU(2) multiscale kernels, filtering, and GEVP diagnostics.
Operator routing is fixed to run-selected companions: U(1)→distance, SU(2)→clone, EW mixed→both (with λ_alg=0).""",
                alert_type="info",
                sizing_mode="stretch_width",
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
                *build_gevp_mass_spectrum_sections(new_dirac_ew_su2_mass_spectrum_widgets),
                pn.pane.Markdown("### U(1) GEVP Analysis"),
                *build_gevp_dashboard_sections(new_dirac_ew_u1_gevp_widgets),
                *build_gevp_mass_spectrum_sections(new_dirac_ew_u1_mass_spectrum_widgets),
                pn.pane.Markdown("### EW Mixed GEVP Analysis"),
                *build_gevp_dashboard_sections(new_dirac_ew_ew_mixed_gevp_widgets),
                *build_gevp_mass_spectrum_sections(new_dirac_ew_ew_mixed_mass_spectrum_widgets),
                pn.pane.Markdown("### Electroweak Reference Masses (GeV)"),
                new_dirac_ew_ref_table,
                pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
                new_dirac_ew_channel_plots,
                sizing_mode="stretch_both",
            )

            gevp_mass_spectrum_tab = pn.Column(
                gevp_tab_status,
                pn.layout.Divider(),
                pn.pane.Markdown("## Strong Force"),
                pn.pane.Markdown("### Nucleon"),
                *build_gevp_mass_spectrum_sections(gevp_tab_nucleon_ms_widgets),
                pn.pane.Markdown("### Scalar"),
                *build_gevp_mass_spectrum_sections(gevp_tab_scalar_ms_widgets),
                pn.pane.Markdown("### Pseudoscalar"),
                *build_gevp_mass_spectrum_sections(gevp_tab_pseudoscalar_ms_widgets),
                pn.pane.Markdown("### Glueball"),
                *build_gevp_mass_spectrum_sections(gevp_tab_glueball_ms_widgets),
                pn.layout.Divider(),
                pn.pane.Markdown("## Electroweak"),
                pn.pane.Markdown("### SU(2)"),
                *build_gevp_mass_spectrum_sections(gevp_tab_su2_ms_widgets),
                pn.pane.Markdown("### U(1)"),
                *build_gevp_mass_spectrum_sections(gevp_tab_u1_ms_widgets),
                pn.pane.Markdown("### EW Mixed"),
                *build_gevp_mass_spectrum_sections(gevp_tab_ew_mixed_ms_widgets),
                sizing_mode="stretch_both",
            )

            main.objects = [
                pn.Tabs(
                    ("Simulation", simulation_tab),
                    ("Algorithm", algorithm_tab),
                    ("Holographic Principle", fractal_set_tab),
                    ("Companion Strong Force", companion_strong_force_tab),
                    ("GEVP Mass Spectrum", gevp_mass_spectrum_tab),
                    ("Coupling Diagnostics", coupling_section.coupling_diagnostics_tab),
                    ("Electroweak", new_dirac_ew_tab),
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
