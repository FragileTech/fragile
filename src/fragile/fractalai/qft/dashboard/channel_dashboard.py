"""Multiscale tab: widgets, layout, and update logic extracted from the dashboard closure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import torch

from fragile.fractalai.qft.dashboard.gevp_dashboard import (
    build_gevp_dashboard_sections,
    channel_family_key,
    clear_gevp_dashboard,
    create_gevp_dashboard_widgets,
    GEVPDashboardWidgets,
    update_gevp_dashboard,
)
from fragile.fractalai.qft.multiscale_analysis import (
    analyze_channel_across_scales,
    build_estimator_table_rows,
    build_pairwise_table_rows,
    format_consensus_summary,
)
from fragile.fractalai.qft.multiscale_strong_force import MultiscaleStrongForceOutput
from fragile.fractalai.qft.operator_analysis import (
    build_consensus_plot,
    build_mass_vs_scale_plot,
    build_multiscale_correlator_plot,
    build_multiscale_effective_mass_plot,
    build_per_scale_channel_plots,
)
from fragile.fractalai.qft.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
)


COMPANION_SUFFIX = "_companion"
SPATIAL_PREFIX = "spatial_"
SPATIAL_CANONICAL_LABEL_MAP: dict[str, str] = {
    "spatial_nucleon_score_abs": "nucleon",
    "spatial_pseudoscalar_score_directed": "pseudoscalar",
}


def _is_companion_channel(name: str) -> bool:
    return str(name).endswith(COMPANION_SUFFIX)


def _base_channel_name(name: str) -> str:
    raw = str(name)
    if raw.endswith(COMPANION_SUFFIX):
        return raw[: -len(COMPANION_SUFFIX)]
    if raw.startswith(SPATIAL_PREFIX):
        return raw[len(SPATIAL_PREFIX) :]
    return raw


def _display_channel_name(raw_name: str) -> str:
    base = _base_channel_name(raw_name)
    if _is_companion_channel(raw_name):
        return base
    spatial_name = f"{SPATIAL_PREFIX}{base}"
    return SPATIAL_CANONICAL_LABEL_MAP.get(spatial_name, spatial_name)


def _display_channel_name_for_original(raw_name: str, result: Any) -> str:
    raw = str(raw_name)
    if raw.startswith(SPATIAL_PREFIX) or raw.endswith(COMPANION_SUFFIX):
        return _display_channel_name(raw)
    base = _base_channel_name(raw)
    if _is_companion_original_result(raw, result):
        return base
    spatial_name = f"{SPATIAL_PREFIX}{base}"
    return SPATIAL_CANONICAL_LABEL_MAP.get(spatial_name, spatial_name)


def _is_companion_original_result(name: str, result: Any) -> bool:
    if _is_companion_channel(name):
        return True
    mass_fit = getattr(result, "mass_fit", None)
    if not isinstance(mass_fit, dict):
        return False
    if str(mass_fit.get("source", "")).strip().lower() == "original_companion":
        return True
    base_channel = str(mass_fit.get("base_channel", "")).strip()
    return _is_companion_channel(base_channel)


# ---------------------------------------------------------------------------
# Dataclass holding multiscale-tab widget references
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleTabWidgets:
    """All Panel widgets that belong to the dedicated Multiscale tab."""

    status: pn.pane.Markdown
    geodesic_heatmap: pn.pane.HoloViews
    geodesic_histogram: pn.pane.HoloViews
    channel_select: pn.widgets.Select
    scale_plot_family_select: pn.widgets.Select
    corr_plot: pn.pane.HoloViews
    meff_plot: pn.pane.HoloViews
    mass_vs_scale_plot: pn.pane.HoloViews
    estimator_table: pn.widgets.Tabulator
    pairwise_table: pn.widgets.Tabulator
    consensus_summary: pn.pane.Markdown
    systematics_badge: pn.pane.Alert
    consensus_plot: pn.pane.HoloViews
    gevp_family_select: pn.widgets.MultiChoice
    gevp_family_select_all_button: pn.widgets.Button
    gevp_family_clear_button: pn.widgets.Button
    gevp_scale_select: pn.widgets.MultiChoice
    gevp_scale_select_all_button: pn.widgets.Button
    gevp_scale_clear_button: pn.widgets.Button
    gevp_compute_button: pn.widgets.Button
    gevp: GEVPDashboardWidgets
    per_scale_plots: pn.Column


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_multiscale_widgets() -> MultiscaleTabWidgets:
    """Create and return all multiscale-tab widgets with default values."""
    return MultiscaleTabWidgets(
        status=pn.pane.Markdown(
            "**Multiscale:** enable multiscale kernels in Companion Strong Force settings, "
            "then run Compute Companion Strong Force Channels.",
            sizing_mode="stretch_width",
        ),
        geodesic_heatmap=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        geodesic_histogram=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        channel_select=pn.widgets.Select(
            name="Channel",
            options=["nucleon"],
            value="nucleon",
            sizing_mode="stretch_width",
        ),
        scale_plot_family_select=pn.widgets.Select(
            name="Scale Plot Family",
            options=["nucleon"],
            value="nucleon",
            sizing_mode="stretch_width",
        ),
        corr_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        meff_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        mass_vs_scale_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        estimator_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        pairwise_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        consensus_summary=pn.pane.Markdown(
            "**Scale-as-Estimator Consensus:** run multiscale analysis to populate.",
            sizing_mode="stretch_width",
        ),
        systematics_badge=pn.pane.Alert(
            "Systematics verdict: run multiscale analysis to evaluate.",
            alert_type="secondary",
            sizing_mode="stretch_width",
        ),
        consensus_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width", linked_axes=False,
        ),
        gevp_family_select=pn.widgets.MultiChoice(
            name="GEVP Families",
            options=["nucleon"],
            value=["nucleon"],
            sizing_mode="stretch_width",
        ),
        gevp_family_select_all_button=pn.widgets.Button(
            name="All Families",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_family_clear_button=pn.widgets.Button(
            name="Clear Families",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_scale_select=pn.widgets.MultiChoice(
            name="GEVP Scales",
            options=["s0"],
            value=["s0"],
            sizing_mode="stretch_width",
        ),
        gevp_scale_select_all_button=pn.widgets.Button(
            name="All Scales",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_scale_clear_button=pn.widgets.Button(
            name="Clear Scales",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_compute_button=pn.widgets.Button(
            name="Recompute GEVP",
            button_type="primary",
            sizing_mode="stretch_width",
        ),
        gevp=create_gevp_dashboard_widgets(),
        per_scale_plots=pn.Column(sizing_mode="stretch_width"),
    )


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------


def build_multiscale_tab_layout(w: MultiscaleTabWidgets) -> pn.Column:
    """Return the ``pn.Column`` that forms the Multiscale tab content."""
    return pn.Column(
        w.status,
        pn.Row(w.channel_select, w.scale_plot_family_select, sizing_mode="stretch_width"),
        pn.pane.Markdown("### Full Geodesic Distance Matrix"),
        pn.pane.Markdown(
            "_Computed from recorded neighbor graph and selected edge-weight mode "
            "on the representative frame used by the current multiscale run._"
        ),
        w.geodesic_heatmap,
        pn.pane.Markdown("### Geodesic Distance Distribution"),
        w.geodesic_histogram,
        pn.layout.Divider(),
        pn.pane.Markdown("### Correlator Across Scales"),
        pn.pane.Markdown(
            "_One color per scale; scatter+errorbars from the same multiscale kernel family._"
        ),
        w.corr_plot,
        pn.pane.Markdown("### Effective Mass Across Scales"),
        w.meff_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass vs Scale"),
        w.mass_vs_scale_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Scale-as-Estimator Analysis"),
        w.estimator_table,
        pn.pane.Markdown("### Pairwise Discrepancies"),
        w.pairwise_table,
        pn.pane.Markdown("### Consensus / Systematics"),
        w.systematics_badge,
        w.consensus_summary,
        w.consensus_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### GEVP Controls"),
        pn.Row(
            w.gevp_family_select,
            w.gevp_scale_select,
            w.gevp_compute_button,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            w.gevp_family_select_all_button,
            w.gevp_family_clear_button,
            w.gevp_scale_select_all_button,
            w.gevp_scale_clear_button,
            sizing_mode="stretch_width",
        ),
        *build_gevp_dashboard_sections(w.gevp),
        pn.layout.Divider(),
        pn.pane.Markdown("### Per-Scale Channel Plots"),
        w.per_scale_plots,
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Clear helper
# ---------------------------------------------------------------------------


def clear_multiscale_tab(w: MultiscaleTabWidgets, status_text: str) -> None:
    """Reset every multiscale-tab widget to its empty/default state."""
    w.status.object = status_text
    w.geodesic_heatmap.object = None
    w.geodesic_histogram.object = None
    w.scale_plot_family_select.options = []
    w.scale_plot_family_select.value = None
    w.corr_plot.object = None
    w.meff_plot.object = None
    w.mass_vs_scale_plot.object = None
    w.estimator_table.value = pd.DataFrame()
    w.pairwise_table.value = pd.DataFrame()
    w.consensus_summary.object = (
        "**Scale-as-Estimator Consensus:** run multiscale analysis to populate."
    )
    w.systematics_badge.object = (
        "Systematics verdict: run multiscale analysis to evaluate."
    )
    w.systematics_badge.alert_type = "secondary"
    w.consensus_plot.object = None
    w.gevp_family_select.options = []
    w.gevp_family_select.value = []
    w.gevp_scale_select.options = []
    w.gevp_scale_select.value = []
    clear_gevp_dashboard(w.gevp)
    w.per_scale_plots.objects = []


# ---------------------------------------------------------------------------
# Main update (the "Part B" of the old _update_anisotropic_edge_multiscale_views)
# ---------------------------------------------------------------------------


def update_multiscale_tab(
    w: MultiscaleTabWidgets,
    output: MultiscaleStrongForceOutput,
    scale_values: np.ndarray,
    state: dict,
    *,
    history: Any | None = None,
    original_results: dict[str, Any] | None = None,
    kernel_distance_method: str = "auto",
    edge_weight_mode: str = "riemannian_kernel_volume",
    kernel_assume_all_alive: bool = True,
) -> None:
    """Populate the dedicated Multiscale tab from a completed multiscale run.

    Parameters
    ----------
    w:
        Widget dataclass returned by :func:`create_multiscale_widgets`.
    output:
        The ``MultiscaleStrongForceOutput`` from the latest computation.
    scale_values:
        1-D numpy array of scale values (already detached/cpu).
    state:
        The shared dashboard ``state`` dict – used to stash output for the
        channel-selector callback.
    history:
        Optional ``RunHistory`` for geodesic distance computation.
    kernel_distance_method, edge_weight_mode, kernel_assume_all_alive:
        Settings forwarded to ``compute_pairwise_distance_matrices_from_history``.
    """
    # -- Status header -------------------------------------------------------
    status_lines = [
        "## Multiscale",
        f"- Scales available: `{len(scale_values)}`",
        f"- Frames analyzed: `{len(output.frame_indices)}`",
        f"- Bootstrap mode: `{output.bootstrap_mode_applied}`",
    ]
    base_channel_count = sum(
        1
        for channel_name in output.per_scale_results
        if not _is_companion_channel(str(channel_name))
    )
    companion_channel_count = sum(
        1
        for channel_name in output.per_scale_results
        if _is_companion_channel(str(channel_name))
    )
    status_lines.append(
        f"- Measurement groups: `non_companion={base_channel_count}`, "
        f"`companion={companion_channel_count}`"
    )

    # -- 1) Full geodesic distance matrix heatmap ----------------------------
    geodesic_error: str | None = None
    geodesic_frame_idx: int | None = None
    geodesic_max_scale = float("nan")
    try:
        if history is not None and output.frame_indices:
            geodesic_frame_idx = int(output.frame_indices[-1])
            _, distance_batch = compute_pairwise_distance_matrices_from_history(
                history,
                method=kernel_distance_method,
                frame_indices=[geodesic_frame_idx],
                batch_size=1,
                edge_weight_mode=edge_weight_mode,
                assume_all_alive=kernel_assume_all_alive,
                device=None,
                dtype=torch.float32,
            )
            if distance_batch.numel() > 0:
                distance_matrix = distance_batch[0].detach().cpu().numpy().astype(
                    np.float64, copy=False
                )
                finite_geodesics = distance_matrix[np.isfinite(distance_matrix) & (distance_matrix > 0)]
                if finite_geodesics.size > 0:
                    geodesic_max_scale = float(np.nanmax(finite_geodesics))
                n_walkers = int(distance_matrix.shape[0])
                if n_walkers == 0 or distance_matrix.shape[1] != n_walkers:
                    w.geodesic_heatmap.object = None
                    geodesic_error = "distance matrix is not square"
                else:
                    matrix = np.array(distance_matrix, dtype=np.float64, copy=True)
                    matrix[~np.isfinite(matrix)] = np.nan
                    matrix = np.nanmean(np.stack([matrix, matrix.T], axis=0), axis=0)
                    finite_positive = np.isfinite(matrix) & (matrix > 0)
                    if np.any(finite_positive):
                        fill_value = float(np.nanpercentile(matrix[finite_positive], 99.0))
                        if not np.isfinite(fill_value) or fill_value <= 0.0:
                            fill_value = float(np.nanmax(matrix[finite_positive]))
                    else:
                        fill_value = 1.0
                    if not np.isfinite(fill_value) or fill_value <= 0.0:
                        fill_value = 1.0
                    matrix = np.where(np.isfinite(matrix), matrix, fill_value)
                    matrix = np.maximum(matrix, 0.0)
                    np.fill_diagonal(matrix, 0.0)

                    clustering_note = "not clustered"
                    order = np.arange(n_walkers, dtype=np.int64)
                    try:
                        from scipy.cluster.hierarchy import leaves_list, linkage
                        from scipy.spatial.distance import squareform

                        if n_walkers >= 2:
                            condensed = squareform(matrix, checks=False)
                            linkage_mat = linkage(
                                condensed,
                                method="average",
                                optimal_ordering=True,
                            )
                            order = leaves_list(linkage_mat).astype(np.int64, copy=False)
                            clustering_note = "average-linkage"
                        else:
                            clustering_note = "single walker"
                    except Exception as cluster_exc:
                        clustering_note = f"fallback (no clustering): {cluster_exc!s}"

                    matrix_ordered = matrix[np.ix_(order, order)].astype(np.float64, copy=False)
                    log_floor = np.finfo(np.float64).tiny
                    log_matrix = np.log10(np.maximum(matrix_ordered, log_floor))
                    np.fill_diagonal(log_matrix, np.nan)
                    w.geodesic_heatmap.object = hv.QuadMesh(
                        (order, order, log_matrix.astype(np.float32, copy=False)),
                        kdims=["walker_j", "walker_i"],
                        vdims=["log10_geodesic_distance"],
                    ).opts(
                        width=900,
                        height=420,
                        cmap="Viridis",
                        colorbar=True,
                        xlabel="Walker index j (clustered order)",
                        ylabel="Walker index i (clustered order)",
                        title=(
                            f"Geodesic Distance Heatmap (log₁₀) "
                            f"(frame={geodesic_frame_idx}, {clustering_note})"
                        ),
                        tools=["hover"],
                        show_grid=False,
                    )
                    # Histogram of off-diagonal geodesic distances.
                    upper_tri = matrix_ordered[np.triu_indices(n_walkers, k=1)]
                    valid_dists = upper_tri[np.isfinite(upper_tri) & (upper_tri > 0)]
                    if valid_dists.size >= 2:
                        log_dists = np.log10(valid_dists)
                        hist_freq, hist_edges = np.histogram(log_dists, bins=60)
                        w.geodesic_histogram.object = hv.Histogram(
                            (hist_freq, hist_edges),
                            kdims=["log10_distance"],
                            vdims=["count"],
                        ).opts(
                            width=900,
                            height=260,
                            xlabel="log₁₀(geodesic distance)",
                            ylabel="Pair count",
                            title="Geodesic Distance Distribution",
                            color="#4c78a8",
                            line_color="white",
                            show_grid=True,
                        )
                    else:
                        w.geodesic_histogram.object = None
                    status_lines.append(f"- Heatmap clustering: `{clustering_note}`")
            else:
                w.geodesic_heatmap.object = None
                geodesic_error = "distance matrix is empty"
        else:
            w.geodesic_heatmap.object = None
            geodesic_error = "history or frame indices unavailable"
    except Exception as exc:
        w.geodesic_heatmap.object = None
        geodesic_error = str(exc)

    if geodesic_frame_idx is not None:
        status_lines.append(f"- Heatmap frame: `{geodesic_frame_idx}`")
    if math.isfinite(geodesic_max_scale) and geodesic_max_scale > 0:
        status_lines.append(f"- Estimated full-scale radius (max geodesic): `{geodesic_max_scale:.6g}`")
    if geodesic_error:
        status_lines.append(f"- Heatmap note: `{geodesic_error}`")

    # -- 2) Populate channel selector and store output for callback ----------
    raw_channels = sorted(
        output.per_scale_results.keys(),
        key=lambda name: (0 if _is_companion_channel(str(name)) else 1, _base_channel_name(str(name))),
    )
    display_to_raw: dict[str, str] = {}
    display_to_original: dict[str, str] = {}
    for raw_name in raw_channels:
        display_name = _display_channel_name(str(raw_name))
        if display_name not in display_to_raw:
            display_to_raw[display_name] = str(raw_name)
            display_to_original.setdefault(display_name, str(raw_name))
    if isinstance(original_results, dict):
        for raw_name, result in original_results.items():
            raw_name_s = str(raw_name)
            if raw_name_s.endswith("_multiscale_best"):
                continue
            display_name = _display_channel_name_for_original(raw_name_s, result)
            display_to_original.setdefault(display_name, raw_name_s)
    available_channels = list(display_to_raw.keys())
    for display_name in display_to_original:
        if display_name not in available_channels:
            available_channels.append(display_name)
    available_channels.sort(
        key=lambda name: (
            0 if not str(name).startswith(SPATIAL_PREFIX) else 1,
            _base_channel_name(str(name)),
        )
    )

    w.channel_select.options = available_channels
    if "nucleon" in available_channels:
        default_channel = "nucleon"
    elif "spatial_nucleon_score_abs" in available_channels:
        default_channel = "spatial_nucleon_score_abs"
    elif "spatial_nucleon" in available_channels:
        default_channel = "spatial_nucleon"
    else:
        default_channel = available_channels[0] if available_channels else ""
    w.channel_select.value = default_channel

    available_families = sorted({channel_family_key(str(raw_name)) for raw_name in raw_channels})
    if not available_families:
        available_families = ["nucleon"]
    default_family = channel_family_key(display_to_raw.get(default_channel, default_channel))
    if default_family not in available_families:
        default_family = available_families[0]

    current_plot_family = str(getattr(w.scale_plot_family_select, "value", "") or "")
    w.scale_plot_family_select.options = available_families
    w.scale_plot_family_select.value = (
        current_plot_family if current_plot_family in available_families else default_family
    )

    current_gevp_families = [str(v) for v in getattr(w.gevp_family_select, "value", [])]
    w.gevp_family_select.options = available_families
    filtered_gevp_families = [fam for fam in current_gevp_families if fam in available_families]
    if not filtered_gevp_families:
        filtered_gevp_families = [default_family]
    w.gevp_family_select.value = filtered_gevp_families

    scale_option_values = [f"s{i}" for i in range(max(0, len(scale_values)))]
    if isinstance(original_results, dict) and len(original_results) > 0:
        scale_option_values.append("full_original_no_threshold")
    if not scale_option_values:
        scale_option_values = ["s0"]
    current_gevp_scales = [str(v) for v in getattr(w.gevp_scale_select, "value", [])]
    w.gevp_scale_select.options = scale_option_values
    filtered_gevp_scales = [sc for sc in current_gevp_scales if sc in scale_option_values]
    if not filtered_gevp_scales:
        filtered_gevp_scales = list(scale_option_values)
    w.gevp_scale_select.value = filtered_gevp_scales

    state["_multiscale_output"] = output
    state["_multiscale_scale_values"] = scale_values
    state["_multiscale_original_results"] = original_results
    state["_multiscale_display_to_raw"] = display_to_raw
    state["_multiscale_display_to_original"] = display_to_original
    state["_multiscale_geodesic_max_scale"] = geodesic_max_scale
    state.setdefault("_multiscale_gevp_dirty", False)

    # -- 3) Inner function: update channel-specific views --------------------
    def _update_channel_views(channel_name: str, *, force_gevp: bool = False) -> None:
        if not channel_name:
            return
        ms_output = state.get("_multiscale_output")
        sv = state.get("_multiscale_scale_values")
        display_to_raw_map = state.get("_multiscale_display_to_raw", {})
        display_to_original_map = state.get("_multiscale_display_to_original", {})
        raw_channel_name = str(display_to_raw_map.get(channel_name, channel_name))
        display_channel_name = str(channel_name)
        if ms_output is None or sv is None:
            return
        channel_results = ms_output.per_scale_results.get(raw_channel_name, [])

        bundle = (
            analyze_channel_across_scales(channel_results, sv, display_channel_name)
            if channel_results
            else None
        )

        # -- Look up original (no-scale-filter) mass for reference overlay --
        original_mass: float | None = None
        original_error: float = float("nan")
        original_r2: float = float("nan")
        original_result_obj: Any | None = None
        original_scale_estimate = float(state.get("_multiscale_geodesic_max_scale", float("nan")))
        if not (math.isfinite(original_scale_estimate) and original_scale_estimate > 0):
            sv_arr = np.asarray(sv, dtype=float)
            finite_scales = sv_arr[np.isfinite(sv_arr) & (sv_arr > 0)]
            if finite_scales.size > 0:
                original_scale_estimate = float(np.max(finite_scales))
        orig_results = state.get("_multiscale_original_results")
        if isinstance(orig_results, dict):
            mapped_key = display_to_original_map.get(display_channel_name)
            if mapped_key is not None:
                orig_result = orig_results.get(mapped_key)
                if orig_result is not None:
                    original_result_obj = orig_result
                    mass_fit = getattr(orig_result, "mass_fit", None)
                    if isinstance(mass_fit, dict):
                        original_mass_candidate = float(mass_fit.get("mass", float("nan")))
                        if math.isfinite(original_mass_candidate) and original_mass_candidate > 0:
                            original_mass = original_mass_candidate
                            original_error = float(mass_fit.get("mass_error", float("nan")))
                            original_r2 = float(mass_fit.get("r_squared", float("nan")))
            base_name = _base_channel_name(raw_channel_name)
            is_companion_selection = _is_companion_channel(raw_channel_name)
            if is_companion_selection:
                candidate_keys = [
                    base_name,
                    raw_channel_name,
                    display_channel_name,
                    f"{base_name}{COMPANION_SUFFIX}",
                ]
            else:
                candidate_keys = [
                    display_channel_name,
                    raw_channel_name,
                    f"{SPATIAL_PREFIX}{base_name}",
                    base_name,
                ]
            for key in candidate_keys:
                orig_result = orig_results.get(key)
                if orig_result is None:
                    continue
                if _is_companion_original_result(str(key), orig_result) != is_companion_selection:
                    continue
                original_result_obj = orig_result
                mass_fit = getattr(orig_result, "mass_fit", None)
                if not isinstance(mass_fit, dict):
                    continue
                original_mass_candidate = float(mass_fit.get("mass", float("nan")))
                if not (math.isfinite(original_mass_candidate) and original_mass_candidate > 0):
                    continue
                original_mass = original_mass_candidate
                original_error = float(mass_fit.get("mass_error", float("nan")))
                original_r2 = float(mass_fit.get("r_squared", float("nan")))
                break

        try:
            gevp_min_r2 = float(state.get("_multiscale_gevp_min_operator_r2", 0.5))
        except (TypeError, ValueError):
            gevp_min_r2 = 0.5
        try:
            gevp_min_windows = max(0, int(state.get("_multiscale_gevp_min_operator_windows", 10)))
        except (TypeError, ValueError):
            gevp_min_windows = 10
        try:
            gevp_max_error_pct = float(state.get("_multiscale_gevp_max_operator_error_pct", 30.0))
        except (TypeError, ValueError):
            gevp_max_error_pct = 30.0
        gevp_remove_artifacts = bool(
            state.get(
                "_multiscale_gevp_remove_artifacts",
                state.get("_multiscale_gevp_exclude_zero_error_operators", True),
            )
        )
        companion_gevp_results = state.get("companion_strong_force_results")
        gevp_selected_families = [str(v) for v in getattr(w.gevp_family_select, "value", [])]
        gevp_selected_scales = [str(v) for v in getattr(w.gevp_scale_select, "value", [])]
        if force_gevp or not bool(state.get("_multiscale_gevp_dirty")):
            gevp_payload = update_gevp_dashboard(
                w.gevp,
                selected_channel_name=display_channel_name,
                raw_channel_name=raw_channel_name,
                per_scale_results=ms_output.per_scale_results,
                original_results=orig_results if isinstance(orig_results, dict) else None,
                companion_gevp_results=(
                    companion_gevp_results if isinstance(companion_gevp_results, dict) else None
                ),
                min_r2=gevp_min_r2,
                min_windows=gevp_min_windows,
                max_error_pct=gevp_max_error_pct,
                remove_artifacts=gevp_remove_artifacts,
                selected_families=gevp_selected_families,
                selected_scale_labels=gevp_selected_scales,
            )
            state["_multiscale_gevp_dirty"] = False
            gevp_mass = gevp_payload.get("gevp_mass")
            gevp_error = float(gevp_payload.get("gevp_error", float("nan")))
        else:
            clear_gevp_dashboard(w.gevp)
            families_txt = ", ".join(gevp_selected_families) if gevp_selected_families else "none"
            scales_txt = ", ".join(gevp_selected_scales) if gevp_selected_scales else "none"
            w.gevp.summary.object = (
                "**GEVP pending recompute:** selectors changed.  \n"
                f"_Pending scope:_ `families={families_txt}`, `scales={scales_txt}`.  \n"
                "Click `Recompute GEVP` to update diagnostics."
            )
            gevp_mass = None
            gevp_error = float("nan")

        def _render_per_scale_family_plots() -> None:
            # Per-scale ChannelPlot layouts (2 scales per row => 4 plots/row).
            selected_plot_family = str(getattr(w.scale_plot_family_select, "value", "") or "")
            family_candidates = [
                str(name)
                for name in ms_output.per_scale_results
                if channel_family_key(str(name)) == selected_plot_family
            ]
            family_candidates.sort(
                key=lambda name: (0 if _is_companion_channel(name) else 1, _base_channel_name(name))
            )
            chosen_scale_plot_channel: str | None = None
            for candidate in family_candidates:
                if _display_channel_name(candidate) == selected_plot_family:
                    chosen_scale_plot_channel = candidate
                    break
            if chosen_scale_plot_channel is None and family_candidates:
                chosen_scale_plot_channel = family_candidates[0]

            if chosen_scale_plot_channel is not None:
                scale_plot_results = ms_output.per_scale_results.get(chosen_scale_plot_channel, [])
                per_scale_layouts = build_per_scale_channel_plots(scale_plot_results, sv)
                if per_scale_layouts:
                    display_scale_operator = _display_channel_name(chosen_scale_plot_channel)
                    rows: list[Any] = [
                        pn.pane.Markdown(
                            f"_Scale plot family:_ `{selected_plot_family}`  \n"
                            f"_Operator used:_ `{display_scale_operator}`",
                            sizing_mode="stretch_width",
                        )
                    ]
                    for idx in range(0, len(per_scale_layouts), 2):
                        row_panels: list[Any] = []
                        for label, layout in per_scale_layouts[idx : idx + 2]:
                            row_panels.append(
                                pn.Column(
                                    pn.pane.Markdown(f"#### {label}"),
                                    layout,
                                    sizing_mode="stretch_width",
                                )
                            )
                        rows.append(pn.Row(*row_panels, sizing_mode="stretch_width"))
                    w.per_scale_plots.objects = rows
                else:
                    w.per_scale_plots.objects = [
                        pn.pane.Markdown(
                            f"_No per-scale plots available for family `{selected_plot_family}`._",
                            sizing_mode="stretch_width",
                        )
                    ]
            else:
                w.per_scale_plots.objects = [
                    pn.pane.Markdown(
                        f"_No channels available for family `{selected_plot_family}`._",
                        sizing_mode="stretch_width",
                    )
                ]

        if bundle is None:
            w.corr_plot.object = build_multiscale_correlator_plot(
                [],
                np.asarray([], dtype=float),
                display_channel_name,
                reference_result=original_result_obj,
                reference_label="original (no filter)",
                reference_scale=original_scale_estimate,
            )
            w.meff_plot.object = build_multiscale_effective_mass_plot(
                [],
                np.asarray([], dtype=float),
                display_channel_name,
                reference_result=original_result_obj,
                reference_label="original (no filter)",
                reference_scale=original_scale_estimate,
            )
            w.mass_vs_scale_plot.object = build_mass_vs_scale_plot(
                [],
                display_channel_name,
                reference_mass=original_mass,
                reference_scale=original_scale_estimate,
                gevp_mass=gevp_mass,
                gevp_error=gevp_error,
                gevp_scale=original_scale_estimate,
            )
            est_rows = build_estimator_table_rows(
                [],
                original_mass=original_mass,
                original_error=original_error,
                original_r2=original_r2,
                original_scale=original_scale_estimate,
            )
            if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
                gevp_row: dict[str, float | str] = {
                    "scale": "GEVP (final)",
                    "scale_value": float("nan"),
                    "mass": float(gevp_mass),
                    "mass_error": float(gevp_error),
                    "mass_error_pct": (
                        abs(float(gevp_error) / float(gevp_mass)) * 100.0
                        if math.isfinite(gevp_error) and gevp_error > 0
                        else float("nan")
                    ),
                    "r_squared": float("nan"),
                }
                if original_mass is not None and math.isfinite(original_mass) and original_mass > 0:
                    gevp_row["delta_vs_original_pct"] = (
                        (float(gevp_mass) - float(original_mass)) / float(original_mass) * 100.0
                    )
                est_rows.append(gevp_row)
            w.estimator_table.value = pd.DataFrame(est_rows) if est_rows else pd.DataFrame()
            if original_mass is not None:
                w.consensus_summary.object = (
                    f"**{display_channel_name}:** original-only result available; "
                    "no multiscale sweep was produced for this channel in the current run."
                )
            else:
                w.consensus_summary.object = (
                    f"**{display_channel_name}:** no per-scale or original results available."
                )
            if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
                gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
                if math.isfinite(gevp_error) and gevp_error > 0:
                    gevp_line += f" ± `{gevp_error:.2g}`"
                w.consensus_summary.object += f"  \n{gevp_line}"
            w.pairwise_table.value = pd.DataFrame()
            w.systematics_badge.object = (
                "Systematics verdict: no multiscale sweep available for this channel."
            )
            w.systematics_badge.alert_type = "secondary"
            w.consensus_plot.object = None
            _render_per_scale_family_plots()
            return

        # Correlator plot (scatter+errorbars per scale).
        w.corr_plot.object = build_multiscale_correlator_plot(
            channel_results,
            sv,
            display_channel_name,
            reference_result=original_result_obj,
            reference_label="original (no filter)",
            reference_scale=original_scale_estimate,
        )
        # Effective mass plot.
        w.meff_plot.object = build_multiscale_effective_mass_plot(
            channel_results,
            sv,
            display_channel_name,
            reference_result=original_result_obj,
            reference_label="original (no filter)",
            reference_scale=original_scale_estimate,
        )
        # Mass vs scale with consensus overlay.
        w.mass_vs_scale_plot.object = build_mass_vs_scale_plot(
            bundle.measurements,
            display_channel_name,
            consensus=bundle.consensus,
            reference_mass=original_mass,
            reference_scale=original_scale_estimate,
            gevp_mass=gevp_mass,
            gevp_error=gevp_error,
            gevp_scale=original_scale_estimate,
        )
        est_rows = build_estimator_table_rows(
            bundle.measurements,
            original_mass=original_mass,
            original_error=original_error,
            original_r2=original_r2,
            original_scale=original_scale_estimate,
        )
        if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
            gevp_row: dict[str, float | str] = {
                "scale": "GEVP (final)",
                "scale_value": float("nan"),
                "mass": float(gevp_mass),
                "mass_error": float(gevp_error),
                "mass_error_pct": (
                    abs(float(gevp_error) / float(gevp_mass)) * 100.0
                    if math.isfinite(gevp_error) and gevp_error > 0
                    else float("nan")
                ),
                "r_squared": float("nan"),
            }
            if original_mass is not None and math.isfinite(original_mass) and original_mass > 0:
                gevp_row["delta_vs_original_pct"] = (
                    (float(gevp_mass) - float(original_mass)) / float(original_mass) * 100.0
                )
            est_rows.append(gevp_row)
        w.estimator_table.value = (
            pd.DataFrame(est_rows) if est_rows else pd.DataFrame()
        )
        # Pairwise discrepancy table.
        pw_rows = build_pairwise_table_rows(bundle.discrepancies)
        w.pairwise_table.value = (
            pd.DataFrame(pw_rows).sort_values("abs_delta_pct", ascending=False)
            if pw_rows
            else pd.DataFrame()
        )
        # Consensus summary + badge.
        w.consensus_summary.object = format_consensus_summary(
            bundle.consensus, bundle.discrepancies, display_channel_name,
            reference_mass=original_mass,
        )
        if gevp_mass is not None and math.isfinite(gevp_mass) and gevp_mass > 0:
            gevp_line = f"- Final GEVP mass: `{gevp_mass:.6g}`"
            if math.isfinite(gevp_error) and gevp_error > 0:
                gevp_line += f" ± `{gevp_error:.2g}`"
            if math.isfinite(bundle.consensus.mass) and bundle.consensus.mass > 0:
                delta_gevp_pct = (
                    (float(bundle.consensus.mass) - float(gevp_mass)) / float(gevp_mass) * 100.0
                )
                gevp_line += f" (consensus delta `{delta_gevp_pct:+.2f}%`)"
            w.consensus_summary.object += f"  \n{gevp_line}"
        w.systematics_badge.object = (
            f"Systematics verdict: {bundle.verdict.label}. {bundle.verdict.details}"
        )
        w.systematics_badge.alert_type = bundle.verdict.alert_type
        # Consensus plot.
        w.consensus_plot.object = build_consensus_plot(
            bundle.measurements, bundle.consensus, display_channel_name,
            reference_mass=original_mass,
            gevp_mass=gevp_mass,
            gevp_error=gevp_error,
        )
        _render_per_scale_family_plots()

    # -- Wire channel selector callback --------------------------------------
    def _on_channel_change(event: Any) -> None:
        _update_channel_views(str(event.new), force_gevp=False)

    def _on_scale_plot_family_change(event: Any) -> None:
        current = str(w.channel_select.value or "")
        _update_channel_views(current, force_gevp=False)

    def _on_gevp_selection_change(event: Any) -> None:
        state["_multiscale_gevp_dirty"] = True
        current = str(w.channel_select.value or "")
        _update_channel_views(current, force_gevp=False)

    def _on_gevp_compute_click(event: Any) -> None:
        if int(getattr(event, "new", 0)) <= 0:
            return
        state["_multiscale_gevp_dirty"] = False
        current = str(w.channel_select.value or "")
        _update_channel_views(current, force_gevp=True)

    def _on_gevp_family_select_all_click(event: Any) -> None:
        if int(getattr(event, "new", 0)) <= 0:
            return
        w.gevp_family_select.value = [str(v) for v in list(w.gevp_family_select.options)]

    def _on_gevp_family_clear_click(event: Any) -> None:
        if int(getattr(event, "new", 0)) <= 0:
            return
        w.gevp_family_select.value = []

    def _on_gevp_scale_select_all_click(event: Any) -> None:
        if int(getattr(event, "new", 0)) <= 0:
            return
        w.gevp_scale_select.value = [str(v) for v in list(w.gevp_scale_select.options)]

    def _on_gevp_scale_clear_click(event: Any) -> None:
        if int(getattr(event, "new", 0)) <= 0:
            return
        w.gevp_scale_select.value = []

    # Rebind watchers on each update to avoid duplicate callbacks.
    old_watchers = state.get("_multiscale_tab_watchers", [])
    if isinstance(old_watchers, list):
        for widget, watcher in old_watchers:
            try:
                widget.param.unwatch(watcher)
            except Exception:
                continue
    new_watchers: list[tuple[Any, Any]] = [
        (w.channel_select, w.channel_select.param.watch(_on_channel_change, "value")),
        (
            w.scale_plot_family_select,
            w.scale_plot_family_select.param.watch(_on_scale_plot_family_change, "value"),
        ),
        (w.gevp_family_select, w.gevp_family_select.param.watch(_on_gevp_selection_change, "value")),
        (w.gevp_scale_select, w.gevp_scale_select.param.watch(_on_gevp_selection_change, "value")),
        (
            w.gevp_family_select_all_button,
            w.gevp_family_select_all_button.param.watch(_on_gevp_family_select_all_click, "clicks"),
        ),
        (
            w.gevp_family_clear_button,
            w.gevp_family_clear_button.param.watch(_on_gevp_family_clear_click, "clicks"),
        ),
        (
            w.gevp_scale_select_all_button,
            w.gevp_scale_select_all_button.param.watch(_on_gevp_scale_select_all_click, "clicks"),
        ),
        (
            w.gevp_scale_clear_button,
            w.gevp_scale_clear_button.param.watch(_on_gevp_scale_clear_click, "clicks"),
        ),
        (w.gevp_compute_button, w.gevp_compute_button.param.watch(_on_gevp_compute_click, "clicks")),
    ]
    state["_multiscale_tab_watchers"] = new_watchers

    # -- Initial render for default channel ----------------------------------
    _update_channel_views(default_channel, force_gevp=True)

    w.status.object = "  \n".join(status_lines)
