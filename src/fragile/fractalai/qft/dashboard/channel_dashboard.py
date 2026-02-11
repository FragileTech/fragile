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
# Dataclass holding the 13 multiscale-tab widget references
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleTabWidgets:
    """All Panel widgets that belong to the dedicated Multiscale tab."""

    status: pn.pane.Markdown
    geodesic_heatmap: pn.pane.HoloViews
    geodesic_histogram: pn.pane.HoloViews
    channel_select: pn.widgets.Select
    corr_plot: pn.pane.HoloViews
    meff_plot: pn.pane.HoloViews
    mass_vs_scale_plot: pn.pane.HoloViews
    estimator_table: pn.widgets.Tabulator
    pairwise_table: pn.widgets.Tabulator
    consensus_summary: pn.pane.Markdown
    systematics_badge: pn.pane.Alert
    consensus_plot: pn.pane.HoloViews
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
        per_scale_plots=pn.Column(sizing_mode="stretch_width"),
    )


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------


def build_multiscale_tab_layout(w: MultiscaleTabWidgets) -> pn.Column:
    """Return the ``pn.Column`` that forms the Multiscale tab content."""
    return pn.Column(
        w.status,
        pn.Row(w.channel_select, sizing_mode="stretch_width"),
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
    state["_multiscale_output"] = output
    state["_multiscale_scale_values"] = scale_values
    state["_multiscale_original_results"] = original_results
    state["_multiscale_display_to_raw"] = display_to_raw
    state["_multiscale_display_to_original"] = display_to_original
    state["_multiscale_geodesic_max_scale"] = geodesic_max_scale

    # -- 3) Inner function: update channel-specific views --------------------
    def _update_channel_views(channel_name: str) -> None:
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
                    _mfit = getattr(orig_result, "mass_fit", None)
                    if isinstance(_mfit, dict):
                        _om = float(_mfit.get("mass", float("nan")))
                        if math.isfinite(_om) and _om > 0:
                            original_mass = _om
                            original_error = float(_mfit.get("mass_error", float("nan")))
                            original_r2 = float(_mfit.get("r_squared", float("nan")))
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
                _mfit = getattr(orig_result, "mass_fit", None)
                if not isinstance(_mfit, dict):
                    continue
                _om = float(_mfit.get("mass", float("nan")))
                if not (math.isfinite(_om) and _om > 0):
                    continue
                original_mass = _om
                original_error = float(_mfit.get("mass_error", float("nan")))
                original_r2 = float(_mfit.get("r_squared", float("nan")))
                break

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
            )
            if original_mass is not None:
                w.estimator_table.value = pd.DataFrame(
                    build_estimator_table_rows(
                        [],
                        original_mass=original_mass,
                        original_error=original_error,
                        original_r2=original_r2,
                        original_scale=original_scale_estimate,
                    )
                )
                w.consensus_summary.object = (
                    f"**{display_channel_name}:** original-only result available; "
                    "no multiscale sweep was produced for this channel in the current run."
                )
            else:
                w.estimator_table.value = pd.DataFrame()
                w.consensus_summary.object = (
                    f"**{display_channel_name}:** no per-scale or original results available."
                )
            w.pairwise_table.value = pd.DataFrame()
            w.systematics_badge.object = (
                "Systematics verdict: no multiscale sweep available for this channel."
            )
            w.systematics_badge.alert_type = "secondary"
            w.consensus_plot.object = None
            w.per_scale_plots.objects = []
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
        )
        est_rows = build_estimator_table_rows(
            bundle.measurements,
            original_mass=original_mass,
            original_error=original_error,
            original_r2=original_r2,
            original_scale=original_scale_estimate,
        )
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
        w.systematics_badge.object = (
            f"Systematics verdict: {bundle.verdict.label}. {bundle.verdict.details}"
        )
        w.systematics_badge.alert_type = bundle.verdict.alert_type
        # Consensus plot.
        w.consensus_plot.object = build_consensus_plot(
            bundle.measurements, bundle.consensus, display_channel_name,
            reference_mass=original_mass,
        )
        # Per-scale ChannelPlot layouts.
        per_scale_layouts = build_per_scale_channel_plots(channel_results, sv)
        if per_scale_layouts:
            children: list[Any] = []
            for label, layout in per_scale_layouts:
                children.append(pn.pane.Markdown(f"#### {label}"))
                children.append(layout)
            w.per_scale_plots.objects = children
        else:
            w.per_scale_plots.objects = []

    # -- Wire channel selector callback --------------------------------------
    def _on_channel_change(event: Any) -> None:
        _update_channel_views(str(event.new))

    w.channel_select.param.watch(_on_channel_change, "value")

    # -- Initial render for default channel ----------------------------------
    _update_channel_views(default_channel)

    w.status.object = "  \n".join(status_lines)
