"""Coupling diagnostics dashboard building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.fractalai.qft.coupling_diagnostics import (
    compute_coupling_diagnostics,
    CouplingDiagnosticsConfig,
)
from fragile.physics.app.algorithm import _algorithm_placeholder_plot, _history_transition_steps
from fragile.physics.fractal_gas.history import RunHistory


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


@dataclass
class CouplingDiagnosticsSection:
    """Container for the coupling diagnostics dashboard section."""

    coupling_diagnostics_tab: pn.Column
    coupling_diagnostics_status: pn.pane.Markdown
    coupling_diagnostics_run_button: pn.widgets.Button
    on_run_coupling_diagnostics: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


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
        pd
        .DataFrame({"scale": running_mid, "value": running_g2})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not running_frame.empty:
        running_curves.append(
            hv
            .Curve(running_frame, "scale", "value")
            .relabel("running_g2")
            .opts(color="#e45756", line_width=2)
        )
    creutz_frame = (
        pd
        .DataFrame({"scale": creutz_mid, "value": creutz})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not creutz_frame.empty:
        running_curves.append(
            hv
            .Curve(creutz_frame, "scale", "value")
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


def build_coupling_diagnostics_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
) -> CouplingDiagnosticsSection:
    """Build Coupling Diagnostics tab with callbacks."""

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

    # ----- callback -----

    def on_run_coupling_diagnostics(_):
        def _compute(history):
            color_dims = _parse_dims_spec(coupling_diagnostics_settings.color_dims_spec, history.d)
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
                kernel_distance_method=str(coupling_diagnostics_settings.kernel_distance_method),
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

            evidence = [str(item) for item in (output.regime_evidence or []) if str(item).strip()]
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

        run_tab_computation(
            state,
            coupling_diagnostics_status,
            "coupling diagnostics",
            _compute,
        )

    # ----- event wiring -----

    coupling_diagnostics_run_button.on_click(on_run_coupling_diagnostics)

    # ----- tab layout -----

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

    # ----- on_history_changed -----

    def on_history_changed(defer: bool) -> None:
        coupling_diagnostics_run_button.disabled = False
        coupling_diagnostics_status.object = (
            "**Coupling Diagnostics ready:** click Compute Coupling Diagnostics."
        )
        if defer:
            return
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

    return CouplingDiagnosticsSection(
        coupling_diagnostics_tab=coupling_diagnostics_tab,
        coupling_diagnostics_status=coupling_diagnostics_status,
        coupling_diagnostics_run_button=coupling_diagnostics_run_button,
        on_run_coupling_diagnostics=on_run_coupling_diagnostics,
        on_history_changed=on_history_changed,
    )
