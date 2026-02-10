"""Multiscale tab: widgets, layout, and update logic extracted from the dashboard closure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import torch

from fragile.fractalai.qft.multiscale_strong_force import MultiscaleStrongForceOutput
from fragile.fractalai.qft.operator_analysis import (
    analyze_channel_across_scales,
    build_consensus_plot,
    build_mass_vs_scale_plot,
    build_multiscale_correlator_plot,
    build_multiscale_effective_mass_plot,
    build_per_scale_channel_plots,
    format_consensus_summary,
)
from fragile.fractalai.qft.smeared_operators import (
    compute_pairwise_distance_matrices_from_history,
)


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
        if not str(channel_name).endswith("_companion")
    )
    companion_channel_count = sum(
        1
        for channel_name in output.per_scale_results
        if str(channel_name).endswith("_companion")
    )
    status_lines.append(
        f"- Measurement groups: `non_companion={base_channel_count}`, "
        f"`companion={companion_channel_count}`"
    )

    # -- 1) Full geodesic distance matrix heatmap ----------------------------
    geodesic_error: str | None = None
    geodesic_frame_idx: int | None = None
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
    if geodesic_error:
        status_lines.append(f"- Heatmap note: `{geodesic_error}`")

    # -- 2) Populate channel selector and store output for callback ----------
    available_channels = sorted(
        output.per_scale_results.keys(),
        key=lambda name: (1 if str(name).endswith("_companion") else 0, str(name)),
    )
    w.channel_select.options = available_channels
    default_channel = "nucleon" if "nucleon" in available_channels else (
        available_channels[0] if available_channels else ""
    )
    w.channel_select.value = default_channel
    state["_multiscale_output"] = output
    state["_multiscale_scale_values"] = scale_values
    state["_multiscale_original_results"] = original_results

    # -- 3) Inner function: update channel-specific views --------------------
    def _update_channel_views(channel_name: str) -> None:
        if not channel_name:
            return
        ms_output = state.get("_multiscale_output")
        sv = state.get("_multiscale_scale_values")
        if ms_output is None or sv is None:
            return
        channel_results = ms_output.per_scale_results.get(channel_name, [])
        if not channel_results:
            w.corr_plot.object = None
            w.meff_plot.object = None
            w.mass_vs_scale_plot.object = None
            w.estimator_table.value = pd.DataFrame()
            w.pairwise_table.value = pd.DataFrame()
            w.consensus_summary.object = (
                f"**{channel_name}:** no per-scale results available."
            )
            w.systematics_badge.object = "Systematics verdict: no data."
            w.systematics_badge.alert_type = "secondary"
            w.consensus_plot.object = None
            w.per_scale_plots.objects = []
            return

        bundle = analyze_channel_across_scales(channel_results, sv, channel_name)

        # -- Look up original (no-scale-filter) mass for reference overlay --
        original_mass: float | None = None
        original_error: float = float("nan")
        original_r2: float = float("nan")
        orig_results = state.get("_multiscale_original_results")
        if isinstance(orig_results, dict):
            orig_result = orig_results.get(channel_name)
            if orig_result is not None:
                _mfit = getattr(orig_result, "mass_fit", None)
                if isinstance(_mfit, dict):
                    _om = float(_mfit.get("mass", float("nan")))
                    if math.isfinite(_om) and _om > 0:
                        original_mass = _om
                        original_error = float(_mfit.get("mass_error", float("nan")))
                        original_r2 = float(_mfit.get("r_squared", float("nan")))

        # Correlator plot (scatter+errorbars per scale).
        w.corr_plot.object = build_multiscale_correlator_plot(
            channel_results, sv, channel_name,
        )
        # Effective mass plot.
        w.meff_plot.object = build_multiscale_effective_mass_plot(
            channel_results, sv, channel_name,
        )
        # Mass vs scale with consensus overlay.
        w.mass_vs_scale_plot.object = build_mass_vs_scale_plot(
            bundle.measurements, channel_name, consensus=bundle.consensus,
            reference_mass=original_mass,
        )
        # Estimator table.
        est_rows: list[dict[str, Any]] = []
        if original_mass is not None:
            est_rows.append({
                "scale": "original (no filter)",
                "scale_value": float("nan"),
                "mass": original_mass,
                "mass_error": original_error,
                "r_squared": original_r2,
                "delta_vs_original_pct": 0.0,
            })
        for m in bundle.measurements:
            row: dict[str, Any] = {
                "scale": m.label,
                "scale_value": m.scale,
                "mass": m.mass,
                "mass_error": m.mass_error,
                "r_squared": m.r_squared,
            }
            if original_mass is not None and math.isfinite(m.mass) and m.mass > 0:
                row["delta_vs_original_pct"] = (
                    (m.mass - original_mass) / original_mass * 100.0
                )
            elif original_mass is not None:
                row["delta_vs_original_pct"] = float("nan")
            est_rows.append(row)
        w.estimator_table.value = (
            pd.DataFrame(est_rows) if est_rows else pd.DataFrame()
        )
        # Pairwise discrepancy table.
        pw_rows = [
            {
                "scale_a": d.label_a,
                "scale_b": d.label_b,
                "mass_a": d.mass_a,
                "mass_b": d.mass_b,
                "ratio": d.ratio,
                "delta_pct": d.delta_pct,
                "abs_delta_pct": d.abs_delta_pct,
                "combined_error": d.combined_error,
                "pull_sigma": d.pull_sigma,
            }
            for d in bundle.discrepancies
        ]
        w.pairwise_table.value = (
            pd.DataFrame(pw_rows).sort_values("abs_delta_pct", ascending=False)
            if pw_rows
            else pd.DataFrame()
        )
        # Consensus summary + badge.
        w.consensus_summary.object = format_consensus_summary(
            bundle.consensus, bundle.discrepancies, channel_name,
            reference_mass=original_mass,
        )
        w.systematics_badge.object = (
            f"Systematics verdict: {bundle.verdict.label}. {bundle.verdict.details}"
        )
        w.systematics_badge.alert_type = bundle.verdict.alert_type
        # Consensus plot.
        w.consensus_plot.object = build_consensus_plot(
            bundle.measurements, bundle.consensus, channel_name,
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
