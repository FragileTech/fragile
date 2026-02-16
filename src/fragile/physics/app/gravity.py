"""Gravitational / holographic dashboard building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import holoviews as hv
import pandas as pd
import panel as pn

from fragile.physics.app._fractal_set import (
    FractalSetSettings,
    build_fractal_set_baseline_comparison,
    build_fractal_set_scatter_plot,
    build_geodesic_distance_distribution_by_frame,
    build_geodesic_distribution_plot,
    compute_fractal_set_measurements,
    format_fractal_set_summary,
    resolve_electroweak_geodesic_matrices,
)
from fragile.physics.app.qft.einstein_equations import compute_einstein_test, EinsteinConfig
from fragile.physics.app.qft.einstein_equations_plotting import build_scalar_test_log_plot
from fragile.physics.fractal_gas.history import RunHistory


@dataclass
class HolographicPrincipleSection:
    """Container for the holographic dashboard section."""

    fractal_set_tab: pn.layout.base.Column
    fractal_set_status: pn.pane.Markdown
    fractal_set_run_button: pn.widgets.Button
    on_run_fractal_set: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


def build_holographic_principle_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
    new_dirac_ew_settings: Any = None,
) -> HolographicPrincipleSection:
    """Build Holographic Principle tab with callbacks."""

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
    fractal_set_geodesic_distribution_plot = pn.pane.HoloViews(
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    einstein_scalar_log_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
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

    def on_run_fractal_set(_: Any) -> None:
        """Compute IG/CST area-law measurements from recorded companion traces."""

        def _compute(history: RunHistory) -> None:
            points_df, regression_df, frame_df = compute_fractal_set_measurements(
                history,
                fractal_set_settings,
            )

            if points_df.empty:
                fractal_set_summary.object = (
                    "## Fractal Set Summary\n_No valid measurements for the selected settings._"
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
            fractal_set_baseline_table.value = build_fractal_set_baseline_comparison(regression_df)
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

            fractal_set_summary.object = format_fractal_set_summary(
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
                fractal_set_plot_dist.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_dist",
                    title="S_dist vs Area_CST",
                )
                fractal_set_plot_fit.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_fit",
                    title="S_fit vs Area_CST",
                )
                fractal_set_plot_total.object = build_fractal_set_scatter_plot(
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
                fractal_set_plot_dist_geom.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_dist_geom",
                    title="S_dist_geom vs Area_CST_geom",
                )
                fractal_set_plot_fit_geom.object = build_fractal_set_scatter_plot(
                    points_df,
                    regression_df,
                    metric_key="s_fit_geom",
                    title="S_fit_geom vs Area_CST_geom",
                )
                fractal_set_plot_total_geom.object = build_fractal_set_scatter_plot(
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
            n_samples = len(points_df)
            all_frames = list(range(1, int(getattr(history, "n_recorded", 0))))
            if new_dirac_ew_settings is not None:
                kernel_distance_method = str(new_dirac_ew_settings.kernel_distance_method)
                edge_weight_mode = str(new_dirac_ew_settings.edge_weight_mode)
                kernel_assume_all_alive = bool(new_dirac_ew_settings.kernel_assume_all_alive)
            else:
                kernel_distance_method = "auto"
                edge_weight_mode = "riemannian_kernel_volume"
                kernel_assume_all_alive = True
            precomputed_pairs = resolve_electroweak_geodesic_matrices(
                history,
                all_frames,
                state,
                method=kernel_distance_method,
                edge_weight_mode=edge_weight_mode,
                assume_all_alive=kernel_assume_all_alive,
            )
            distribution = build_geodesic_distance_distribution_by_frame(precomputed_pairs)
            state["_multiscale_geodesic_distribution"] = distribution
            fractal_set_geodesic_distribution_plot.object = build_geodesic_distribution_plot(
                distribution
            )
            g_newton_metric = (
                "s_total_geom" if bool(fractal_set_settings.use_geometry_correction) else "s_total"
            )
            try:
                einstein_result = compute_einstein_test(
                    history,
                    EinsteinConfig(),
                    fractal_set_regressions=regression_df,
                    g_newton_metric=g_newton_metric,
                )
                state["einstein_test_result"] = einstein_result
                einstein_scalar_log_plot.object = build_scalar_test_log_plot(einstein_result)
            except Exception as exc:
                einstein_scalar_log_plot.object = hv.Text(
                    0,
                    0,
                    f"Einstein test unavailable: {exc!s}",
                ).opts(title="Einstein Scalar Test")
            fractal_set_status.object = (
                f"**Complete:** {n_samples} boundary samples from {n_frames} recorded transitions."
            )
            if precomputed_pairs is not None:
                geodesic_samples = int(
                    state.get("_multiscale_geodesic_distribution", {}).get(
                        "n_geodesic_samples",
                        0,
                    )
                    or 0
                )
                geodesic_frames = int(
                    state.get("_multiscale_geodesic_distribution", {}).get(
                        "n_frames_with_samples",
                        0,
                    )
                    or 0
                )
                fractal_set_status.object = (
                    f"{fractal_set_status.object}"
                    f"  Geodesic pairwise matrix cache: {geodesic_samples} samples from "
                    f"{geodesic_frames} frames."
                )

        run_tab_computation(state, fractal_set_status, "fractal set", _compute)

    def on_history_changed(defer_dashboard_updates: bool) -> None:
        """Update holographic-section controls when a new history is loaded."""
        fractal_set_run_button.disabled = False
        fractal_set_status.object = "**Holographic Principle ready:** click Compute Fractal Set."

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
        pn.pane.Markdown("### Geodesic Pairwise Distance Distribution"),
        fractal_set_geodesic_distribution_plot,
        pn.pane.Markdown("### Einstein Scalar Test: R vs log10(rho)"),
        einstein_scalar_log_plot,
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

    return HolographicPrincipleSection(
        fractal_set_tab=fractal_set_tab,
        fractal_set_status=fractal_set_status,
        fractal_set_run_button=fractal_set_run_button,
        on_run_fractal_set=on_run_fractal_set,
        on_history_changed=on_history_changed,
    )
