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

from fragile.physics.app.algorithm import _algorithm_placeholder_plot, _history_transition_steps
from fragile.physics.app.coupling_diagnostics import (
    compute_coupling_diagnostics,
    CouplingDiagnosticsConfig,
)
from fragile.physics.fractal_gas.history import RunHistory


def _parse_dims_spec(spec: str, history_d: int) -> list[int] | None:
    """Parse a comma-separated dimension list."""
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
            "spatial_dims_spec contains invalid dims "
            f"{invalid}; valid range is [0, {history_d - 1}]."
        )

    return dims


def _tensor_to_numpy(tensor: torch.Tensor, dtype: type = float) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


def _format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


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
    ell0_method = param.ObjectSelector(
        default="companion",
        objects=("companion", "geodesic_edges", "euclidean_edges"),
        doc="Automatic ell0 estimation method when ell0 is blank.",
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
    kernel_scale_frames = param.Integer(default=8, bounds=(1, 64))
    kernel_scale_q_low = param.Number(default=0.05, bounds=(0.0, 0.99))
    kernel_scale_q_high = param.Number(default=0.95, bounds=(0.01, 1.0))
    kernel_max_scale_samples = param.Integer(default=500_000, bounds=(1_000, 5_000_000))
    kernel_min_scale = param.Number(default=1e-6, bounds=(1e-12, None))
    enable_wilson_flow = param.Boolean(
        default=False,
        doc="Enable Wilson flow (gradient flow) analysis for scale extraction.",
    )
    wilson_flow_n_steps = param.Integer(
        default=100,
        bounds=(10, 1000),
        doc="Number of Wilson flow diffusion steps.",
    )
    wilson_flow_step_size = param.Number(
        default=0.02,
        bounds=(0.001, 0.5),
        step=0.01,
        doc="Wilson flow diffusion step size (epsilon).",
    )
    wilson_flow_topology = param.ObjectSelector(
        default="both",
        objects=("distance", "clone", "both"),
        doc="Companion topology for Wilson flow diffusion.",
    )
    wilson_flow_t0_reference = param.Number(
        default=0.3,
        bounds=(0.01, 10.0),
        step=0.01,
        doc="Reference value for t0 extraction (t^2<E> = ref).",
    )
    wilson_flow_w0_reference = param.Number(
        default=0.3,
        bounds=(0.01, 10.0),
        step=0.01,
        doc="Reference value for w0 extraction (d/dt[t^2<E>] = ref).",
    )


@dataclass
class CouplingDiagnosticsSection:
    """Container for the coupling diagnostics dashboard section."""

    coupling_diagnostics_tab: pn.Column
    coupling_diagnostics_status: pn.pane.Markdown
    coupling_diagnostics_run_button: pn.widgets.Button
    on_run_coupling_diagnostics: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


@dataclass
class _CouplingDiagnosticsWidgets:
    summary: pn.pane.Markdown
    regime_evidence: pn.pane.Markdown
    summary_table: pn.widgets.Tabulator
    frame_table: pn.widgets.Tabulator
    scale_table: pn.widgets.Tabulator
    phase_plot: pn.pane.HoloViews
    regime_plot: pn.pane.HoloViews
    fields_plot: pn.pane.HoloViews
    coverage_plot: pn.pane.HoloViews
    scale_plot: pn.pane.HoloViews
    running_plot: pn.pane.HoloViews
    wilson_flow_plot: pn.pane.HoloViews
    wilson_t2e_plot: pn.pane.HoloViews
    wilson_derivative_plot: pn.pane.HoloViews
    wilson_summary: pn.pane.Markdown


def _build_coupling_diagnostics_settings_panel(
    settings: CouplingDiagnosticsSettings,
) -> pn.Param:
    return pn.Param(
        settings,
        parameters=[
            "simulation_range",
            "h_eff",
            "mass",
            "ell0",
            "ell0_method",
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
            "enable_wilson_flow",
            "wilson_flow_n_steps",
            "wilson_flow_step_size",
            "wilson_flow_topology",
            "wilson_flow_t0_reference",
            "wilson_flow_w0_reference",
        ],
        show_name=False,
        widgets={
            "color_dims_spec": {"name": "Color dims (optional)"},
            "wilson_flow_step_size": {"type": pn.widgets.EditableFloatSlider, "step": 0.01},
            "wilson_flow_t0_reference": {"type": pn.widgets.EditableFloatSlider, "step": 0.01},
            "wilson_flow_w0_reference": {"type": pn.widgets.EditableFloatSlider, "step": 0.01},
        },
        default_layout=type("CouplingDiagnosticsSettingsGrid", (pn.GridBox,), {"ncols": 2}),
    )


def _build_coupling_diagnostics_widgets() -> _CouplingDiagnosticsWidgets:
    summary = pn.pane.Markdown(
        "## Coupling Diagnostics Summary\n_Run diagnostics to populate._",
        sizing_mode="stretch_width",
    )
    regime_evidence = pn.pane.Markdown(
        "_Regime evidence will appear after running diagnostics._",
        sizing_mode="stretch_width",
    )
    summary_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    frame_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=20,
        show_index=False,
        sizing_mode="stretch_width",
    )
    scale_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    phase_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    regime_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    fields_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    coverage_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    scale_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    running_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    wilson_flow_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    wilson_t2e_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    wilson_derivative_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    wilson_summary = pn.pane.Markdown(
        "_Enable Wilson flow and run diagnostics to populate._",
        sizing_mode="stretch_width",
    )

    _set_plot_placeholders(
        phase_plot,
        regime_plot,
        fields_plot,
        coverage_plot,
        scale_plot,
        running_plot,
        wilson_flow_plot,
        wilson_t2e_plot,
        wilson_derivative_plot,
    )

    return _CouplingDiagnosticsWidgets(
        summary=summary,
        regime_evidence=regime_evidence,
        summary_table=summary_table,
        frame_table=frame_table,
        scale_table=scale_table,
        phase_plot=phase_plot,
        regime_plot=regime_plot,
        fields_plot=fields_plot,
        coverage_plot=coverage_plot,
        scale_plot=scale_plot,
        running_plot=running_plot,
        wilson_flow_plot=wilson_flow_plot,
        wilson_t2e_plot=wilson_t2e_plot,
        wilson_derivative_plot=wilson_derivative_plot,
        wilson_summary=wilson_summary,
    )


def _set_plot_placeholders(*plots: pn.pane.HoloViews) -> None:
    placeholders = (
        "Run diagnostics to show phase trend.",
        "Run diagnostics to show regime metrics.",
        "Run diagnostics to show field means.",
        "Run diagnostics to show pair/walker coverage.",
        "Run diagnostics to show kernel-scale diagnostics.",
        "Run diagnostics to show running/Creutz diagnostics.",
        "Enable Wilson flow to show action density E(t).",
        "Enable Wilson flow to show t^2 E(t) and t0.",
        "Enable Wilson flow to show d/dt[t^2 E] and w0.",
    )
    for plot, message in zip(plots, placeholders):
        plot.object = _algorithm_placeholder_plot(message)


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
        ("spectral_gap_fiedler", summary.get("spectral_gap_fiedler")),
        ("ell0", summary.get("ell0")),
        ("h_eff", summary.get("h_eff")),
        ("spectral_gap_autocorrelation", summary.get("spectral_gap_autocorrelation")),
        ("spectral_gap_autocorrelation_tau", summary.get("spectral_gap_autocorrelation_tau")),
        ("spectral_gap_transfer_matrix", summary.get("spectral_gap_transfer_matrix")),
        ("wilson_flow_t0", summary.get("wilson_flow_t0")),
        ("wilson_flow_w0", summary.get("wilson_flow_w0")),
        ("wilson_flow_sqrt_8t0", summary.get("wilson_flow_sqrt_8t0")),
    ]
    frame = pd.DataFrame([{"metric": name, "value": value} for name, value in metrics])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _build_coupling_diagnostics_frame_table(
    step_axis: np.ndarray,
    output: Any,
) -> pd.DataFrame:
    """Build per-frame diagnostics table."""
    if len(step_axis) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "step": step_axis,
        "phase_mean": _tensor_to_numpy(output.phase_mean),
        "phase_mean_unwrapped": _tensor_to_numpy(output.phase_mean_unwrapped),
        "r_circ": _tensor_to_numpy(output.phase_concentration),
        "re_im_asymmetry": _tensor_to_numpy(output.re_im_asymmetry),
        "local_phase_coherence": _tensor_to_numpy(output.local_phase_coherence),
        "scalar_mean": _tensor_to_numpy(output.scalar_mean),
        "pseudoscalar_mean": _tensor_to_numpy(output.pseudoscalar_mean),
        "field_magnitude_mean": _tensor_to_numpy(output.field_magnitude_mean),
        "valid_pairs": _tensor_to_numpy(output.valid_pair_counts, int),
        "valid_walkers": _tensor_to_numpy(output.valid_walker_counts, int),
    }).replace([np.inf, -np.inf], np.nan)


def _build_coupling_diagnostics_scale_table(output: Any) -> pd.DataFrame:
    """Build one-row-per-scale diagnostics table."""
    scales = getattr(output, "scales", None)
    if scales is None or int(scales.numel()) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "scale": _tensor_to_numpy(output.scales),
        "coherence": _tensor_to_numpy(output.coherence_by_scale),
        "phase_spread": _tensor_to_numpy(output.phase_spread_by_scale),
        "screening_connected": _tensor_to_numpy(output.screening_connected_by_scale),
    }).replace([np.inf, -np.inf], np.nan)


def _build_overlay_plot(
    *,
    axis_name: str,
    axis: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    ylabel: str,
    placeholder: str,
) -> hv.Overlay | hv.Text:
    overlays: list[Any] = []
    for label, values, color in series:
        frame = pd.DataFrame({axis_name: axis, "value": values}).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        frame = frame.dropna()
        if frame.empty:
            continue
        overlays.append(
            hv.Curve(frame, axis_name, "value")
            .relabel(label)
            .opts(
                color=color,
                line_width=2,
                tools=["hover"],
            )
        )

    if not overlays:
        return _algorithm_placeholder_plot(placeholder)

    plot = overlays[0]
    for overlay in overlays[1:]:
        plot = plot * overlay

    xlabel = "Recorded step" if axis_name == "step" else "Scale"
    opts_kw: dict[str, Any] = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "width": 960,
        "height": 320,
        "show_grid": True,
    }
    if len(overlays) > 1:
        opts_kw["legend_position"] = "top_left"
    return plot.opts(**opts_kw)


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

    return {
        "phase": _build_overlay_plot(
            axis_name="step",
            axis=step_axis,
            series=[
                ("phase_mean", _tensor_to_numpy(output.phase_mean), "#4c78a8"),
                (
                    "phase_mean_unwrapped",
                    _tensor_to_numpy(output.phase_mean_unwrapped),
                    "#f58518",
                ),
            ],
            title="Global Phase Trend",
            ylabel="phase [rad]",
            placeholder="No diagnostics data available",
        ),
        "regime": _build_overlay_plot(
            axis_name="step",
            axis=step_axis,
            series=[
                ("R_circ", _tensor_to_numpy(output.phase_concentration), "#54a24b"),
                ("Re/Im asymmetry", _tensor_to_numpy(output.re_im_asymmetry), "#e45756"),
                ("local coherence", _tensor_to_numpy(output.local_phase_coherence), "#72b7b2"),
            ],
            title="Coupling Regime Diagnostics",
            ylabel="dimensionless",
            placeholder="No diagnostics data available",
        ),
        "fields": _build_overlay_plot(
            axis_name="step",
            axis=step_axis,
            series=[
                ("scalar_mean", _tensor_to_numpy(output.scalar_mean), "#9d755d"),
                (
                    "pseudoscalar_mean",
                    _tensor_to_numpy(output.pseudoscalar_mean),
                    "#b279a2",
                ),
                (
                    "field_magnitude_mean",
                    _tensor_to_numpy(output.field_magnitude_mean),
                    "#4c78a8",
                ),
            ],
            title="Local Color Field Means",
            ylabel="operator value",
            placeholder="No diagnostics data available",
        ),
        "coverage": _build_overlay_plot(
            axis_name="step",
            axis=step_axis,
            series=[
                ("valid_pairs", _tensor_to_numpy(output.valid_pair_counts, int), "#f58518"),
                ("valid_walkers", _tensor_to_numpy(output.valid_walker_counts, int), "#54a24b"),
            ],
            title="Diagnostics Coverage",
            ylabel="count",
            placeholder="No diagnostics data available",
        ),
    }


def _build_coupling_diagnostics_kernel_plots(output: Any) -> dict[str, hv.Overlay | hv.Text]:
    """Build kernel-scale diagnostics plots."""
    scales = getattr(output, "scales", None)
    if scales is None or int(scales.numel()) == 0:
        placeholder = _algorithm_placeholder_plot("No kernel-scale diagnostics available")
        return {"scale": placeholder, "running": placeholder}

    scale_plot = _build_overlay_plot(
        axis_name="scale",
        axis=_tensor_to_numpy(output.scales),
        series=[
            ("coherence", _tensor_to_numpy(output.coherence_by_scale), "#4c78a8"),
            ("phase_spread", _tensor_to_numpy(output.phase_spread_by_scale), "#f58518"),
            (
                "screening_connected",
                _tensor_to_numpy(output.screening_connected_by_scale),
                "#54a24b",
            ),
        ],
        title="Kernel-Scale Diagnostics",
        ylabel="dimensionless",
        placeholder="No kernel-scale diagnostics available",
    )

    running_curves: list[Any] = []
    running_frame = (
        pd.DataFrame({
            "scale": _tensor_to_numpy(output.running_mid_scales),
            "value": _tensor_to_numpy(output.running_g2_by_mid_scale),
        })
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
        pd.DataFrame({
            "scale": _tensor_to_numpy(output.creutz_mid_scales),
            "value": _tensor_to_numpy(output.creutz_ratio_by_mid_scale),
        })
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
            logy=True,
        )
    else:
        running_plot = _algorithm_placeholder_plot("No running/Creutz diagnostics available")

    return {"scale": scale_plot, "running": running_plot}


def _build_wilson_flow_plots(
    wf_output: Any,
) -> dict[str, hv.Overlay | hv.Text]:
    """Build Wilson flow diagnostic plots."""
    if wf_output is None:
        placeholder = _algorithm_placeholder_plot("Wilson flow disabled")
        return {"flow": placeholder, "t2e": placeholder, "derivative": placeholder}

    flow_times = _tensor_to_numpy(wf_output.flow_times)
    action_density = _tensor_to_numpy(wf_output.action_density)
    t2_action = _tensor_to_numpy(wf_output.t2_action)

    # 1. Action density E(t) vs flow time
    flow_plot = _build_overlay_plot(
        axis_name="flow_time",
        axis=flow_times,
        series=[("E(t)", action_density, "#4c78a8")],
        title="Wilson Flow: Action Density E(t)",
        ylabel="E(t)",
        placeholder="No Wilson flow data available",
    )
    if not isinstance(flow_plot, hv.Text):
        flow_plot = flow_plot.opts(xlabel="Flow time t")

    # 2. t^2 * E(t) vs flow time + t0 marker
    t2e_curves: list[Any] = []
    t2e_frame = (
        pd.DataFrame({"flow_time": flow_times, "value": t2_action})
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )
    if not t2e_frame.empty:
        t2e_curves.append(
            hv.Curve(t2e_frame, "flow_time", "value")
            .relabel("t^2 E(t)")
            .opts(color="#4c78a8", line_width=2, tools=["hover"])
        )
        t2e_curves.append(
            hv.HLine(wf_output.config.t0_reference).opts(
                color="#e45756",
                line_dash="dashed",
                line_width=1,
            )
        )
        if np.isfinite(wf_output.t0):
            t2e_curves.append(
                hv.VLine(wf_output.t0).opts(
                    color="#54a24b",
                    line_dash="dotted",
                    line_width=1,
                )
            )

    if t2e_curves:
        t2e_plot = t2e_curves[0]
        for c in t2e_curves[1:]:
            t2e_plot = t2e_plot * c
        t2e_plot = t2e_plot.opts(
            title="Wilson Flow: t^2 E(t) (t0 extraction)",
            xlabel="Flow time t",
            ylabel="t^2 E(t)",
            width=960,
            height=320,
            show_grid=True,
        )
    else:
        t2e_plot = _algorithm_placeholder_plot("No t^2 E(t) data available")

    # 3. d/dt[t^2 E] vs midpoint flow time + w0 marker
    dt2_times = _tensor_to_numpy(wf_output.dt2_action_times)
    dt2_values = _tensor_to_numpy(wf_output.dt2_action)
    deriv_curves: list[Any] = []
    deriv_frame = (
        pd.DataFrame({"flow_time": dt2_times, "value": dt2_values})
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )
    if not deriv_frame.empty:
        deriv_curves.append(
            hv.Curve(deriv_frame, "flow_time", "value")
            .relabel("d/dt[t^2 E]")
            .opts(color="#4c78a8", line_width=2, tools=["hover"])
        )
        deriv_curves.append(
            hv.HLine(wf_output.config.w0_reference).opts(
                color="#e45756",
                line_dash="dashed",
                line_width=1,
            )
        )
        if np.isfinite(wf_output.w0):
            deriv_curves.append(
                hv.VLine(wf_output.w0).opts(
                    color="#54a24b",
                    line_dash="dotted",
                    line_width=1,
                )
            )

    if deriv_curves:
        deriv_plot = deriv_curves[0]
        for c in deriv_curves[1:]:
            deriv_plot = deriv_plot * c
        deriv_plot = deriv_plot.opts(
            title="Wilson Flow: d/dt[t^2 E(t)] (w0 extraction)",
            xlabel="Flow time t",
            ylabel="d/dt[t^2 E]",
            width=960,
            height=320,
            show_grid=True,
        )
    else:
        deriv_plot = _algorithm_placeholder_plot("No derivative data available")

    return {"flow": flow_plot, "t2e": t2e_plot, "derivative": deriv_plot}


def _build_wilson_flow_summary_text(wf_output: Any) -> str:
    """Build Wilson flow summary markdown."""
    if wf_output is None:
        return "_Wilson flow disabled._"
    return "\n".join([
        "**Wilson Flow Scale Extraction:**",
        f"- t0 (t^2<E> = {wf_output.config.t0_reference}): `{_format_metric(wf_output.t0, 6)}`",
        f"- w0 (d/dt[t^2<E>] = {wf_output.config.w0_reference}): `{_format_metric(wf_output.w0, 6)}`",
        f"- sqrt(8 t0): `{_format_metric(wf_output.sqrt_8t0, 6)}`",
        f"- Flow steps: `{wf_output.config.n_steps}`, step size: `{wf_output.config.step_size}`",
    ])


def _build_coupling_diagnostics_summary_text(output: Any) -> str:
    summary = output.summary
    n_frames = int(summary.get("n_frames", 0.0) or 0)
    return "\n".join([
        "## Coupling Diagnostics Summary",
        f"- Frames analyzed: `{n_frames}`",
        f"- ℓ₀ (estimated): `{_format_metric(summary.get('ell0'), 6)}`",
        f"- h_eff: `{_format_metric(summary.get('h_eff'), 6)}`",
        f"- Mean R_circ: `{_format_metric(summary.get('r_circ_mean'))}`",
        f"- Mean Re/Im asymmetry: `{_format_metric(summary.get('re_im_asymmetry_mean'))}`",
        (
            "- Mean local phase coherence: "
            f"`{_format_metric(summary.get('local_phase_coherence_mean'))}`"
        ),
        (
            "- Phase drift significance: "
            f"`{_format_metric(summary.get('phase_drift_sigma'), 3)}σ`"
        ),
        (
            "- String tension proxy σ: "
            f"`{_format_metric(summary.get('string_tension_sigma'), 6)}`"
        ),
        f"- Polyakov loop |L|: `{_format_metric(summary.get('polyakov_abs'), 6)}`",
        f"- Screening length ξ: `{_format_metric(summary.get('screening_length_xi'), 6)}`",
        (
            "- Running coupling slope: "
            f"`{_format_metric(summary.get('running_coupling_slope'), 6)}`"
        ),
        f"- Topological flux std: `{_format_metric(summary.get('topological_flux_std'), 6)}`",
        f"- Regime score: `{_format_metric(summary.get('regime_score'), 2)}` / 10",
        "",
        "**Spectral Gap Estimates:**",
        (
            "- Fiedler value (graph Laplacian λ₂): "
            f"`{_format_metric(summary.get('spectral_gap_fiedler'), 6)}`"
        ),
        (
            "- Autocorrelation gap (1/τ): "
            f"`{_format_metric(summary.get('spectral_gap_autocorrelation'), 6)}`"
            f" (τ = `{_format_metric(summary.get('spectral_gap_autocorrelation_tau'), 2)}`)"
        ),
        (
            "- Transfer-matrix gap: "
            f"`{_format_metric(summary.get('spectral_gap_transfer_matrix'), 6)}`"
        ),
        "",
        (
            f"- Kernel snapshot frame index: `{int(output.snapshot_frame_index)}`"
            if output.snapshot_frame_index is not None
            else "- Kernel snapshot frame index: `n/a`"
        ),
        "",
        "**Wilson Flow:**",
        f"- t0: `{_format_metric(summary.get('wilson_flow_t0'), 6)}`",
        f"- w0: `{_format_metric(summary.get('wilson_flow_w0'), 6)}`",
        f"- sqrt(8 t0): `{_format_metric(summary.get('wilson_flow_sqrt_8t0'), 6)}`",
        "",
        "- This tab computes only fast regime diagnostics (no channel masses).",
    ])


def _build_regime_evidence_markdown(output: Any) -> str:
    evidence = [str(item) for item in (output.regime_evidence or []) if str(item).strip()]
    if evidence:
        return "\n".join(["### Regime Evidence"] + [f"- {line}" for line in evidence])
    return "_Regime evidence unavailable for this run._"


def _build_step_axis(history: RunHistory, frame_indices: Any) -> np.ndarray:
    recorded_steps = np.asarray(history.recorded_steps, dtype=float)
    if frame_indices is None:
        return np.zeros(0, dtype=float)

    indices = np.atleast_1d(np.asarray(frame_indices, dtype=int))
    if indices.size == 0:
        return np.zeros(0, dtype=float)

    max_index = int(indices.max())
    if recorded_steps.size > max_index:
        return recorded_steps[indices]

    return _history_transition_steps(history, len(indices))


def _build_coupling_diagnostics_config(
    settings: CouplingDiagnosticsSettings,
    history_d: int,
) -> CouplingDiagnosticsConfig:
    color_dims = _parse_dims_spec(settings.color_dims_spec, history_d)
    return CouplingDiagnosticsConfig(
        warmup_fraction=float(settings.simulation_range[0]),
        end_fraction=float(settings.simulation_range[1]),
        h_eff=float(settings.h_eff),
        mass=float(settings.mass),
        ell0=float(settings.ell0) if settings.ell0 is not None else None,
        ell0_method=str(settings.ell0_method),
        color_dims=tuple(color_dims) if color_dims is not None else None,
        companion_topology=str(settings.companion_topology),
        pair_weighting=str(settings.pair_weighting),
        eps=float(settings.eps),
        enable_kernel_diagnostics=bool(settings.enable_kernel_diagnostics),
        edge_weight_mode=str(settings.edge_weight_mode),
        n_scales=int(settings.n_scales),
        kernel_type=str(settings.kernel_type),
        kernel_distance_method=str(settings.kernel_distance_method),
        kernel_scale_frames=int(settings.kernel_scale_frames),
        kernel_scale_q_low=float(settings.kernel_scale_q_low),
        kernel_scale_q_high=float(settings.kernel_scale_q_high),
        kernel_max_scale_samples=int(settings.kernel_max_scale_samples),
        kernel_min_scale=float(settings.kernel_min_scale),
        kernel_assume_all_alive=True,
        enable_wilson_flow=bool(settings.enable_wilson_flow),
        wilson_flow_n_steps=int(settings.wilson_flow_n_steps),
        wilson_flow_step_size=float(settings.wilson_flow_step_size),
        wilson_flow_topology=str(settings.wilson_flow_topology),
        wilson_flow_t0_reference=float(settings.wilson_flow_t0_reference),
        wilson_flow_w0_reference=float(settings.wilson_flow_w0_reference),
    )


def _apply_diagnostics_outputs(
    *,
    output: Any,
    step_axis: np.ndarray,
    widgets: _CouplingDiagnosticsWidgets,
) -> None:
    widgets.summary_table.value = _build_coupling_diagnostics_summary_table(output.summary)
    widgets.frame_table.value = _build_coupling_diagnostics_frame_table(step_axis, output)
    widgets.scale_table.value = _build_coupling_diagnostics_scale_table(output)

    plots = _build_coupling_diagnostics_plots(step_axis, output)
    kernel_plots = _build_coupling_diagnostics_kernel_plots(output)

    widgets.phase_plot.object = plots["phase"]
    widgets.regime_plot.object = plots["regime"]
    widgets.fields_plot.object = plots["fields"]
    widgets.coverage_plot.object = plots["coverage"]
    widgets.scale_plot.object = kernel_plots["scale"]
    widgets.running_plot.object = kernel_plots["running"]

    wilson_plots = _build_wilson_flow_plots(output.wilson_flow)
    widgets.wilson_flow_plot.object = wilson_plots["flow"]
    widgets.wilson_t2e_plot.object = wilson_plots["t2e"]
    widgets.wilson_derivative_plot.object = wilson_plots["derivative"]
    widgets.wilson_summary.object = _build_wilson_flow_summary_text(output.wilson_flow)


def _make_status_text(output: Any) -> str:
    scales = getattr(output, "scales", None)
    n_scales = int(scales.numel()) if scales is not None else 0
    n_frames = int(output.summary.get("n_frames", 0.0) or 0)
    return (
        f"**Complete:** Coupling diagnostics computed ({n_frames} frames). "
        f"Kernel scales: {n_scales}."
    )


def _build_layout(
    status: pn.pane.Markdown,
    settings_panel: pn.Param,
    run_button: pn.widgets.Button,
    run_note: pn.pane.Alert,
    widgets: _CouplingDiagnosticsWidgets,
) -> pn.Column:
    return pn.Column(
        status,
        run_note,
        pn.Row(run_button, sizing_mode="stretch_width"),
        pn.Accordion(("Diagnostics Settings", settings_panel), active=[0], sizing_mode="stretch_width"),
        pn.layout.Divider(),
        widgets.summary,
        widgets.regime_evidence,
        pn.pane.Markdown("### Summary Metrics"),
        widgets.summary_table,
        pn.pane.Markdown("### Per-Frame Metrics"),
        widgets.frame_table,
        pn.pane.Markdown("### Kernel-Scale Metrics"),
        widgets.scale_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### Global Phase Trend"),
        widgets.phase_plot,
        pn.pane.Markdown("### Coupling Regime Metrics"),
        widgets.regime_plot,
        pn.pane.Markdown("### Local Color Field Means"),
        widgets.fields_plot,
        pn.pane.Markdown("### Coverage (Valid Pairs/Walkers)"),
        widgets.coverage_plot,
        pn.pane.Markdown("### Kernel-Scale Curves"),
        widgets.scale_plot,
        pn.pane.Markdown("### Running Coupling / Creutz"),
        widgets.running_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Wilson Flow"),
        widgets.wilson_summary,
        widgets.wilson_flow_plot,
        pn.pane.Markdown("### Wilson Flow: t^2 E(t) (t0 extraction)"),
        widgets.wilson_t2e_plot,
        pn.pane.Markdown("### Wilson Flow: d/dt[t^2 E(t)] (w0 extraction)"),
        widgets.wilson_derivative_plot,
        sizing_mode="stretch_both",
    )


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
    coupling_diagnostics_settings_panel = _build_coupling_diagnostics_settings_panel(
        coupling_diagnostics_settings,
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
    widgets = _build_coupling_diagnostics_widgets()

    def on_run_coupling_diagnostics(_):
        def _compute(history: RunHistory):
            config = _build_coupling_diagnostics_config(
                coupling_diagnostics_settings,
                history.d,
            )
            output = compute_coupling_diagnostics(history, config=config)
            state["coupling_diagnostics_output"] = output

            step_axis = _build_step_axis(history, output.frame_indices)
            _apply_diagnostics_outputs(output=output, step_axis=step_axis, widgets=widgets)

            widgets.summary.object = _build_coupling_diagnostics_summary_text(output)
            widgets.regime_evidence.object = _build_regime_evidence_markdown(output)
            coupling_diagnostics_status.object = _make_status_text(output)

        run_tab_computation(
            state,
            coupling_diagnostics_status,
            "coupling diagnostics",
            _compute,
        )

    def on_history_changed(defer: bool) -> None:
        coupling_diagnostics_run_button.disabled = False
        coupling_diagnostics_status.object = (
            "**Coupling Diagnostics ready:** click Compute Coupling Diagnostics."
        )
        if defer:
            return

        state["coupling_diagnostics_output"] = None
        widgets.summary.object = "## Coupling Diagnostics Summary\n_Run diagnostics to populate._"
        widgets.regime_evidence.object = "_Regime evidence will appear after running diagnostics._"
        widgets.summary_table.value = pd.DataFrame()
        widgets.frame_table.value = pd.DataFrame()
        widgets.scale_table.value = pd.DataFrame()
        _set_plot_placeholders(
            widgets.phase_plot,
            widgets.regime_plot,
            widgets.fields_plot,
            widgets.coverage_plot,
            widgets.scale_plot,
            widgets.running_plot,
            widgets.wilson_flow_plot,
            widgets.wilson_t2e_plot,
            widgets.wilson_derivative_plot,
        )
        widgets.wilson_summary.object = "_Enable Wilson flow and run diagnostics to populate._"

    coupling_diagnostics_run_button.on_click(on_run_coupling_diagnostics)

    coupling_diagnostics_tab = _build_layout(
        status=coupling_diagnostics_status,
        settings_panel=coupling_diagnostics_settings_panel,
        run_button=coupling_diagnostics_run_button,
        run_note=coupling_diagnostics_note,
        widgets=widgets,
    )

    return CouplingDiagnosticsSection(
        coupling_diagnostics_tab=coupling_diagnostics_tab,
        coupling_diagnostics_status=coupling_diagnostics_status,
        coupling_diagnostics_run_button=coupling_diagnostics_run_button,
        on_run_coupling_diagnostics=on_run_coupling_diagnostics,
        on_history_changed=on_history_changed,
    )
