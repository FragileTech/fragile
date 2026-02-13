"""Tensor calibration tab widgets and analysis helpers.

This module owns the Tensor Calibration tab:
- estimator selection controls
- tensor estimator comparison tables/plots
- multiscale calibration surface fitting and cross-validation
- correction payload for downstream mass-table calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from fragile.fractalai.qft.correlator_channels import ChannelCorrelatorResult
from fragile.fractalai.qft.multiscale_strong_force import MultiscaleStrongForceOutput


CORE_ESTIMATOR_ORDER = (
    "Anisotropic-edge tensor",
    "Anisotropic-edge tensor traceless",
    "Strong-force tensor",
    "Tensor momentum p0",
)
COMPONENT_DEFAULT_LABELS = (
    "q_xy",
    "q_xz",
    "q_yz",
    "q_xx_minus_yy",
    "q_2zz_minus_xx_minus_yy",
)
ESTIMATOR_TOGGLE_OPTIONS = [
    ("anisotropic_edge", "Anisotropic-edge tensor"),
    ("anisotropic_edge_traceless", "Anisotropic-edge traceless"),
    ("strong_force", "Strong-force tensor"),
    ("momentum_contracted", "Momentum contracted"),
    ("momentum_components", "Momentum components"),
    ("multiscale_non_companion", "Multiscale non-companion"),
    ("multiscale_companion", "Multiscale companion"),
]
DEFAULT_ESTIMATOR_TOGGLES = [key for key, _ in ESTIMATOR_TOGGLE_OPTIONS]


@dataclass
class TensorCalibrationWidgets:
    """Panel widgets for Tensor Calibration tab."""

    status: pn.pane.Markdown
    estimator_toggles: pn.widgets.CheckButtonGroup
    mass_mode: pn.widgets.RadioButtonGroup
    estimator_table: pn.widgets.Tabulator
    pairwise_table: pn.widgets.Tabulator
    systematics_badge: pn.pane.Alert
    summary: pn.pane.Markdown
    correction_summary: pn.pane.Markdown
    dispersion_plot: pn.pane.HoloViews
    component_dispersion_plot: pn.pane.HoloViews
    anisotropy_summary: pn.pane.Markdown
    anisotropy_table: pn.widgets.Tabulator
    anisotropy_plot: pn.pane.HoloViews
    multiscale_plot: pn.pane.HoloViews
    calibration_surface_plot: pn.pane.HoloViews
    calibration_residual_plot: pn.pane.HoloViews
    cv_summary: pn.pane.Markdown


def create_tensor_calibration_widgets() -> TensorCalibrationWidgets:
    """Create Tensor Calibration widgets."""
    return TensorCalibrationWidgets(
        status=pn.pane.Markdown(
            "**Tensor Calibration:** load a RunHistory and click Compute Tensor Calibration.",
            sizing_mode="stretch_width",
        ),
        estimator_toggles=pn.widgets.CheckButtonGroup(
            name="Tensor estimators",
            value=list(DEFAULT_ESTIMATOR_TOGGLES),
            options={label: key for key, label in ESTIMATOR_TOGGLE_OPTIONS},
            button_type="default",
            sizing_mode="stretch_width",
        ),
        mass_mode=pn.widgets.RadioButtonGroup(
            name="Mass Display",
            options=["AIC-Weighted", "Best Window"],
            value="AIC-Weighted",
            button_type="default",
            sizing_mode="stretch_width",
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
        systematics_badge=pn.pane.Alert(
            "Systematics verdict: run tensor calibration to evaluate.",
            alert_type="secondary",
            sizing_mode="stretch_width",
        ),
        summary=pn.pane.Markdown(
            "**Tensor estimator summary:** run tensor calibration to populate.",
            sizing_mode="stretch_width",
        ),
        correction_summary=pn.pane.Markdown(
            "**Tensor correction:** run tensor calibration to compute correction scale.",
            sizing_mode="stretch_width",
        ),
        dispersion_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        component_dispersion_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        anisotropy_summary=pn.pane.Markdown(
            "**Tensor anisotropy coefficients:** run tensor calibration to populate.",
            sizing_mode="stretch_width",
        ),
        anisotropy_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        anisotropy_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        multiscale_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        calibration_surface_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        calibration_residual_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        cv_summary=pn.pane.Markdown(
            "**Calibration cross-validation:** run tensor calibration to populate.",
            sizing_mode="stretch_width",
        ),
    )


def build_tensor_calibration_tab_layout(
    w: TensorCalibrationWidgets,
    *,
    run_button: pn.widgets.Button,
    settings_layout: pn.layout.Panel | None = None,
) -> pn.Column:
    """Build Tensor Calibration tab layout."""
    sections: list[Any] = [
        w.status,
        pn.Row(run_button, sizing_mode="stretch_width"),
    ]
    if settings_layout is not None:
        sections.extend(
            [
                pn.Accordion(
                    ("Tensor Calibration Settings", settings_layout),
                    sizing_mode="stretch_width",
                ),
            ]
        )
    sections.extend(
        [
            pn.pane.Markdown("### Estimator Selection"),
            w.estimator_toggles,
            pn.pane.Markdown("### Mass Display Mode"),
            w.mass_mode,
            pn.layout.Divider(),
            pn.pane.Markdown("### Tensor Estimator Comparison"),
            w.systematics_badge,
            w.summary,
            w.estimator_table,
            pn.pane.Markdown("### Tensor Pairwise Discrepancies"),
            w.pairwise_table,
            pn.pane.Markdown("### Tensor Correction"),
            w.correction_summary,
            pn.layout.Divider(),
            pn.pane.Markdown("### Tensor Momentum Dispersion (Contracted)"),
            w.dispersion_plot,
            pn.pane.Markdown("### Tensor Momentum Dispersion (Components)"),
            w.component_dispersion_plot,
            pn.pane.Markdown("### Tensor Anisotropy Coefficients"),
            w.anisotropy_summary,
            w.anisotropy_table,
            w.anisotropy_plot,
            pn.layout.Divider(),
            pn.pane.Markdown("### Tensor Mass vs Scale"),
            w.multiscale_plot,
            pn.pane.Markdown("### Tensor Calibration Surface"),
            w.calibration_surface_plot,
            pn.pane.Markdown("### Tensor Calibration Residuals"),
            w.calibration_residual_plot,
            w.cv_summary,
        ]
    )
    return pn.Column(*sections, sizing_mode="stretch_both")


def clear_tensor_calibration_tab(w: TensorCalibrationWidgets, status_text: str) -> None:
    """Reset Tensor Calibration tab to empty/default state."""
    w.status.object = status_text
    w.estimator_table.value = pd.DataFrame()
    w.pairwise_table.value = pd.DataFrame()
    w.systematics_badge.object = "Systematics verdict: run tensor calibration to evaluate."
    w.systematics_badge.alert_type = "secondary"
    w.summary.object = "**Tensor estimator summary:** run tensor calibration to populate."
    w.correction_summary.object = (
        "**Tensor correction:** run tensor calibration to compute correction scale."
    )
    w.dispersion_plot.object = None
    w.component_dispersion_plot.object = None
    w.anisotropy_summary.object = "**Tensor anisotropy coefficients:** run tensor calibration to populate."
    w.anisotropy_table.value = pd.DataFrame()
    w.anisotropy_plot.object = None
    w.multiscale_plot.object = None
    w.calibration_surface_plot.object = None
    w.calibration_residual_plot.object = None
    w.cv_summary.object = "**Calibration cross-validation:** run tensor calibration to populate."


def _get_channel_mass(result: ChannelCorrelatorResult, mode: str) -> float:
    if mode == "Best Window":
        best_window = result.mass_fit.get("best_window", {})
        return float(best_window.get("mass", float("nan")))
    return float(result.mass_fit.get("mass", float("nan")))


def _get_channel_mass_error(result: ChannelCorrelatorResult, mode: str) -> float:
    if mode == "Best Window":
        return float("nan")
    mass_error = float(result.mass_fit.get("mass_error", float("nan")))
    return mass_error if np.isfinite(mass_error) and mass_error >= 0 else float("nan")


def _get_channel_r2(result: ChannelCorrelatorResult, mode: str) -> float:
    if mode == "Best Window":
        best_window = result.mass_fit.get("best_window", {})
        return float(best_window.get("r2", float("nan")))
    return float(result.mass_fit.get("r_squared", float("nan")))


def _parse_tensor_momentum_name(channel_name: str) -> tuple[int | None, str | None]:
    prefix = "tensor_momentum_p"
    if not channel_name.startswith(prefix):
        return None, None
    raw = channel_name[len(prefix) :]
    if "_" in raw:
        mode_raw, component = raw.split("_", 1)
    else:
        mode_raw, component = raw, None
    try:
        return int(mode_raw), component
    except ValueError:
        return None, None


def _delta_percent(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return (value / reference - 1.0) * 100.0


def _fit_dispersion(momentum_rows: list[dict[str, Any]]) -> tuple[float, float]:
    if len(momentum_rows) < 2:
        return float("nan"), float("nan")
    df = pd.DataFrame(momentum_rows)
    fit_df = df[
        np.isfinite(df["p2"].to_numpy())
        & np.isfinite(df["mass"].to_numpy())
        & (df["mass"].to_numpy() > 0)
    ]
    if len(fit_df) < 2:
        return float("nan"), float("nan")
    p2 = fit_df["p2"].to_numpy(dtype=float)
    mass = fit_df["mass"].to_numpy(dtype=float)
    y = mass * mass
    design = np.stack([np.ones_like(p2), p2], axis=1)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1])
    m0 = np.sqrt(intercept) if intercept > 0 else float("nan")
    c_eff = np.sqrt(slope) if slope > 0 else float("nan")
    return m0, c_eff


def _build_dispersion_plot(momentum_rows: list[dict[str, Any]]) -> hv.Overlay | None:
    if not momentum_rows:
        return None
    df = pd.DataFrame(momentum_rows).sort_values("n_mode")
    scatter = hv.Scatter(
        df,
        kdims=[("p2", "p^2")],
        vdims=[("mass", "Mass"), ("n_mode", "n"), ("estimator", "Estimator")],
    ).opts(
        width=860,
        height=320,
        size=8,
        color="#4c78a8",
        tools=["hover"],
        xlabel="p^2",
        ylabel="Mass (index units)",
        title="Tensor momentum dispersion (contracted)",
        show_grid=True,
    )
    overlay = scatter
    err_df = df[np.isfinite(df["mass_error"].to_numpy()) & (df["mass_error"].to_numpy() > 0)][
        ["p2", "mass", "mass_error"]
    ]
    if not err_df.empty:
        overlay *= hv.ErrorBars(
            err_df,
            kdims=[("p2", "p^2")],
            vdims=[("mass", "Mass"), ("mass_error", "Mass Error")],
        ).opts(color="#4c78a8", line_width=1, alpha=0.9)

    m0, c_eff = _fit_dispersion(momentum_rows)
    if np.isfinite(m0) and np.isfinite(c_eff):
        p2_grid = np.linspace(float(np.nanmin(df["p2"])), float(np.nanmax(df["p2"])), 200)
        fit_mass = np.sqrt(np.clip(m0 * m0 + c_eff * c_eff * p2_grid, a_min=0.0, a_max=None))
        overlay *= hv.Curve(
            (p2_grid, fit_mass),
            kdims=[("p2", "p^2")],
            vdims=[("mass_fit", "Fit mass")],
        ).opts(color="#ff7f0e", line_dash="dashed", line_width=2)
    return overlay


def _build_component_dispersion_plot(component_rows: list[dict[str, Any]]) -> hv.Overlay | None:
    if not component_rows:
        return None
    comp_df = pd.DataFrame(component_rows).sort_values(["component", "n_mode"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    overlay = None
    for idx, component in enumerate(sorted(comp_df["component"].unique())):
        sub = comp_df[comp_df["component"] == component]
        if sub.empty:
            continue
        color = colors[idx % len(colors)]
        layer = hv.Scatter(
            sub,
            kdims=[("p2", "p^2")],
            vdims=[("mass", "Mass"), ("n_mode", "n"), ("component", "Component")],
            label=str(component),
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
            ).opts(color=color, line_width=1, alpha=0.75)
        overlay = layer if overlay is None else overlay * layer
    if overlay is None:
        return None
    return overlay.opts(
        width=860,
        height=360,
        title="Tensor momentum dispersion (components)",
        xlabel="p^2",
        ylabel="Mass (index units)",
        show_grid=True,
        show_legend=True,
        legend_position="right",
    )


def _robust_positive_reference(values: np.ndarray) -> float:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _compute_component_anisotropy(
    component_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not component_rows:
        return [], {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in component_rows:
        component = str(row.get("component", "")).strip()
        if not component:
            continue
        grouped.setdefault(component, []).append(row)
    if not grouped:
        return [], {}

    rows: list[dict[str, Any]] = []
    for component in sorted(grouped):
        samples = grouped[component]
        m0, c_eff = _fit_dispersion(samples)
        mode_values: list[int] = []
        n_fit_points = 0
        p0_values: list[float] = []
        for sample in samples:
            try:
                mode_n = int(sample.get("n_mode", -1))
            except (TypeError, ValueError):
                mode_n = -1
            if mode_n >= 0:
                mode_values.append(mode_n)
            mass = float(sample.get("mass", float("nan")))
            p2 = float(sample.get("p2", float("nan")))
            if np.isfinite(p2) and np.isfinite(mass) and mass > 0:
                n_fit_points += 1
            if mode_n == 0 and np.isfinite(mass) and mass > 0:
                p0_values.append(mass)
        p0_mass = float(np.mean(np.asarray(p0_values, dtype=float))) if p0_values else float("nan")
        rows.append(
            {
                "component": component,
                "n_points": len(samples),
                "n_fit_points": n_fit_points,
                "n_modes": ",".join(str(v) for v in sorted(set(mode_values))),
                "p0_mass": p0_mass,
                "m0": m0,
                "c_eff": c_eff,
            }
        )

    c_eff_ref = _robust_positive_reference(np.asarray([float(r["c_eff"]) for r in rows], dtype=float))
    m0_ref = _robust_positive_reference(np.asarray([float(r["m0"]) for r in rows], dtype=float))
    p0_ref = _robust_positive_reference(np.asarray([float(r["p0_mass"]) for r in rows], dtype=float))

    for row in rows:
        c_eff = float(row["c_eff"])
        m0 = float(row["m0"])
        p0_mass = float(row["p0_mass"])
        row["xi_c_eff"] = c_eff / c_eff_ref if np.isfinite(c_eff_ref) and c_eff_ref > 0 else np.nan
        row["xi_m0"] = m0 / m0_ref if np.isfinite(m0_ref) and m0_ref > 0 else np.nan
        row["xi_p0"] = p0_mass / p0_ref if np.isfinite(p0_ref) and p0_ref > 0 else np.nan
        row["delta_c_eff_pct"] = _delta_percent(c_eff, c_eff_ref)
        row["delta_m0_pct"] = _delta_percent(m0, m0_ref)
        row["delta_p0_pct"] = _delta_percent(p0_mass, p0_ref)

    xi_c_eff = np.asarray([float(r.get("xi_c_eff", np.nan)) for r in rows], dtype=float)
    xi_m0 = np.asarray([float(r.get("xi_m0", np.nan)) for r in rows], dtype=float)
    valid_xi_c_eff = xi_c_eff[np.isfinite(xi_c_eff) & (xi_c_eff > 0)]
    valid_xi_m0 = xi_m0[np.isfinite(xi_m0) & (xi_m0 > 0)]
    stats = {
        "component_count": float(len(rows)),
        "c_eff_ref": float(c_eff_ref),
        "m0_ref": float(m0_ref),
        "p0_ref": float(p0_ref),
        "xi_c_eff_rms_pct": float(np.sqrt(np.mean((valid_xi_c_eff - 1.0) ** 2)) * 100.0)
        if valid_xi_c_eff.size > 0
        else float("nan"),
        "xi_c_eff_max_abs_pct": float(np.max(np.abs(valid_xi_c_eff - 1.0)) * 100.0)
        if valid_xi_c_eff.size > 0
        else float("nan"),
        "xi_m0_rms_pct": float(np.sqrt(np.mean((valid_xi_m0 - 1.0) ** 2)) * 100.0)
        if valid_xi_m0.size > 0
        else float("nan"),
        "xi_m0_max_abs_pct": float(np.max(np.abs(valid_xi_m0 - 1.0)) * 100.0)
        if valid_xi_m0.size > 0
        else float("nan"),
    }
    return rows, stats


def _build_anisotropy_plot(anisotropy_rows: list[dict[str, Any]]) -> hv.Overlay | None:
    if not anisotropy_rows:
        return None
    frame = pd.DataFrame(anisotropy_rows).sort_values("component")
    overlay: hv.Overlay | None = None

    c_eff_frame = frame[np.isfinite(frame["xi_c_eff"].to_numpy())]
    if not c_eff_frame.empty:
        c_eff_scatter = hv.Scatter(
            c_eff_frame,
            kdims=[("component", "Component")],
            vdims=[("xi_c_eff", "xi c_eff")],
            label="xi c_eff",
        ).opts(size=9, marker="square", color="#1f77b4", tools=["hover"])
        overlay = c_eff_scatter if overlay is None else overlay * c_eff_scatter

    m0_frame = frame[np.isfinite(frame["xi_m0"].to_numpy())]
    if not m0_frame.empty:
        m0_scatter = hv.Scatter(
            m0_frame,
            kdims=[("component", "Component")],
            vdims=[("xi_m0", "xi m0")],
            label="xi m0",
        ).opts(size=9, marker="triangle", color="#ff7f0e", tools=["hover"])
        overlay = m0_scatter if overlay is None else overlay * m0_scatter

    if overlay is None:
        return None
    overlay *= hv.HLine(1.0).opts(color="#2ca02c", line_dash="dashed", line_width=1.5)
    return overlay.opts(
        width=860,
        height=300,
        title="Tensor anisotropy coefficients by component",
        xlabel="Tensor component",
        ylabel="Relative coefficient (xi)",
        show_grid=True,
        show_legend=True,
        legend_position="right",
    )


def _build_multiscale_curve_plot(
    noncomp: MultiscaleStrongForceOutput | None,
    comp: MultiscaleStrongForceOutput | None,
    mode: str,
) -> hv.Overlay | None:
    def _channel_sort_key(channel_name: str) -> tuple[int, int, str]:
        is_companion = channel_name.endswith("_companion")
        is_traceless = "traceless" in channel_name
        return (1 if is_companion else 0, 1 if is_traceless else 0, channel_name)

    def _tensor_channels_for_output(
        output: MultiscaleStrongForceOutput | None,
        *,
        companion_only: bool,
    ) -> list[str]:
        if output is None or output.scales.numel() == 0:
            return []
        channels: list[str] = []
        for channel_name, per_scale in output.per_scale_results.items():
            channel_name_s = str(channel_name)
            if "tensor" not in channel_name_s:
                continue
            if channel_name_s.endswith("_companion") != companion_only:
                continue
            if not per_scale:
                continue
            channels.append(channel_name_s)
        return sorted(channels, key=_channel_sort_key)

    curves: list[hv.Curve] = []
    line_dashes = ("solid", "dashed", "dotted", "dotdash", "longdash", "dashdot")
    for output, companion_only in ((noncomp, False), (comp, True)):
        channels = _tensor_channels_for_output(output, companion_only=companion_only)
        if output is None or output.scales.numel() == 0:
            continue
        scales = output.scales.detach().cpu().numpy().astype(float, copy=False)
        for idx, channel in enumerate(channels):
            per_scale = output.per_scale_results.get(channel, [])
            if len(per_scale) != len(scales):
                continue
            masses = np.asarray([_get_channel_mass(res, mode) for res in per_scale], dtype=float)
            valid = np.isfinite(scales) & np.isfinite(masses) & (masses > 0)
            if not np.any(valid):
                continue
            label = (
                f"{channel.removesuffix('_companion')} (companion)"
                if channel.endswith("_companion")
                else f"{channel} (non-companion)"
            )
            curve = hv.Curve(
                (scales[valid], masses[valid]),
                kdims=[("scale", "Scale")],
                vdims=[("mass", "Mass")],
                label=label,
            ).opts(line_dash=line_dashes[idx % len(line_dashes)], line_width=2)
            curves.append(curve)
    if not curves:
        return None
    overlay = curves[0]
    for curve in curves[1:]:
        overlay *= curve
    return overlay.opts(
        width=860,
        height=320,
        title="Tensor mass vs scale",
        xlabel="Kernel scale",
        ylabel="Mass (index units)",
        show_grid=True,
        legend_position="right",
    )


def _collect_surface_grid(
    noncomp: MultiscaleStrongForceOutput | None,
    comp: MultiscaleStrongForceOutput | None,
    mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    def _channel_sort_key(channel_name: str) -> tuple[int, int, str]:
        is_companion = channel_name.endswith("_companion")
        is_traceless = "traceless" in channel_name
        return (1 if is_companion else 0, 1 if is_traceless else 0, channel_name)

    def _tensor_rows_for_output(
        output: MultiscaleStrongForceOutput | None,
        *,
        companion_only: bool,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        rows_local: list[tuple[str, np.ndarray, np.ndarray]] = []
        if output is None or output.scales.numel() == 0:
            return rows_local
        scales = output.scales.detach().cpu().numpy().astype(float, copy=False)
        channels = sorted(
            [
                str(channel_name)
                for channel_name, per_scale in output.per_scale_results.items()
                if "tensor" in str(channel_name)
                and str(channel_name).endswith("_companion") == companion_only
                and per_scale
            ],
            key=_channel_sort_key,
        )
        for channel in channels:
            per_scale = output.per_scale_results.get(channel, [])
            if len(per_scale) != len(scales):
                continue
            masses = np.asarray([_get_channel_mass(res, mode) for res in per_scale], dtype=float)
            rows_local.append((channel, scales, masses))
        return rows_local

    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    rows.extend(_tensor_rows_for_output(noncomp, companion_only=False))
    rows.extend(_tensor_rows_for_output(comp, companion_only=True))

    if not rows:
        return None, None, []

    all_scales = np.unique(np.concatenate([scale for _, scale, _ in rows], axis=0))
    op_names = [name for name, _, _ in rows]
    grid = np.full((len(op_names), len(all_scales)), np.nan, dtype=float)
    for op_idx, (_, op_scales, op_masses) in enumerate(rows):
        for s, m in zip(op_scales, op_masses, strict=False):
            loc = int(np.argmin(np.abs(all_scales - s)))
            if np.isfinite(m) and m > 0:
                grid[op_idx, loc] = float(m)
    return all_scales, grid, op_names


def _fit_shared_exp_surface(
    scales: np.ndarray,
    grid: np.ndarray,
) -> dict[str, Any] | None:
    """Fit m(op, R) = m_inf(op) + A(op) exp(-R / r_scale)."""
    n_ops, _ = grid.shape
    valid_points = np.argwhere(np.isfinite(grid) & (grid > 0))
    if valid_points.shape[0] < max(6, 2 * n_ops):
        return None

    scale_min = float(np.nanmin(scales))
    scale_max = float(np.nanmax(scales))
    if not (np.isfinite(scale_min) and np.isfinite(scale_max) and scale_max > scale_min):
        return None
    r_candidates = np.logspace(
        np.log10(max(scale_min * 0.2, 1e-6)),
        np.log10(max(scale_max * 5.0, 1e-5)),
        80,
    )

    y = np.asarray([grid[i, j] for i, j in valid_points], dtype=float)
    best: dict[str, Any] | None = None
    best_sse = float("inf")

    for r_scale in r_candidates:
        x = np.zeros((valid_points.shape[0], 2 * n_ops), dtype=float)
        for row_idx, (op_idx, scale_idx) in enumerate(valid_points):
            x[row_idx, op_idx] = 1.0
            x[row_idx, n_ops + op_idx] = float(np.exp(-scales[scale_idx] / r_scale))
        theta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ theta
        residual = y - pred
        sse = float(np.sum(residual * residual))
        if sse < best_sse:
            best_sse = sse
            best = {
                "r_scale": float(r_scale),
                "theta": theta,
                "sse": sse,
                "n_obs": int(valid_points.shape[0]),
            }

    if best is None:
        return None

    theta = best["theta"]
    m_inf = theta[:n_ops]
    amp = theta[n_ops:]
    pred_grid = np.full_like(grid, np.nan, dtype=float)
    for i in range(n_ops):
        pred_grid[i, :] = m_inf[i] + amp[i] * np.exp(-scales / best["r_scale"])
    residual_grid = grid - pred_grid
    return {
        "r_scale": float(best["r_scale"]),
        "m_inf": m_inf,
        "amp": amp,
        "pred_grid": pred_grid,
        "residual_grid": residual_grid,
        "sse": float(best["sse"]),
        "n_obs": int(best["n_obs"]),
    }


def _predict_single_point(
    scales: np.ndarray,
    op_idx: int,
    scale_idx: int,
    fit: dict[str, Any],
) -> float:
    return float(fit["m_inf"][op_idx] + fit["amp"][op_idx] * np.exp(-scales[scale_idx] / fit["r_scale"]))


def _cross_validate_surface(scales: np.ndarray, grid: np.ndarray) -> dict[str, Any]:
    valid_points = np.argwhere(np.isfinite(grid) & (grid > 0))
    if len(valid_points) < 3:
        return {"n_points": 0, "mean_frac_error_pct": float("nan"), "rms_error": float("nan")}
    frac_errors: list[float] = []
    abs_errors: list[float] = []
    for op_idx, scale_idx in valid_points:
        reduced = np.array(grid, copy=True)
        held = float(reduced[op_idx, scale_idx])
        reduced[op_idx, scale_idx] = np.nan
        fit = _fit_shared_exp_surface(scales, reduced)
        if fit is None:
            continue
        pred = _predict_single_point(scales, int(op_idx), int(scale_idx), fit)
        if np.isfinite(pred) and pred > 0 and np.isfinite(held) and held > 0:
            abs_errors.append(abs(pred - held))
            frac_errors.append(abs(pred - held) / held)
    if not frac_errors:
        return {"n_points": 0, "mean_frac_error_pct": float("nan"), "rms_error": float("nan")}
    frac_arr = np.asarray(frac_errors, dtype=float)
    abs_arr = np.asarray(abs_errors, dtype=float)
    return {
        "n_points": int(frac_arr.size),
        "mean_frac_error_pct": float(np.mean(frac_arr) * 100.0),
        "median_frac_error_pct": float(np.median(frac_arr) * 100.0),
        "p95_frac_error_pct": float(np.percentile(frac_arr, 95.0) * 100.0),
        "rms_error": float(np.sqrt(np.mean(abs_arr * abs_arr))),
    }


def _grid_heatmap(
    scales: np.ndarray,
    op_names: list[str],
    values: np.ndarray,
    *,
    title: str,
    value_label: str,
) -> hv.HeatMap:
    records: list[tuple[float, str, float]] = []
    for op_idx, op_name in enumerate(op_names):
        for scale_idx, scale in enumerate(scales):
            value = float(values[op_idx, scale_idx])
            if np.isfinite(value):
                records.append((float(scale), str(op_name), value))
    frame = pd.DataFrame(records, columns=["scale", "operator", value_label])
    return hv.HeatMap(
        frame,
        kdims=[("scale", "Scale"), ("operator", "Operator")],
        vdims=[(value_label, value_label)],
    ).opts(
        width=860,
        height=300,
        cmap="Viridis",
        colorbar=True,
        xlabel="Kernel scale",
        ylabel="Tensor estimator",
        title=title,
        tools=["hover"],
    )


def update_tensor_calibration_tab(
    w: TensorCalibrationWidgets,
    *,
    base_results: dict[str, ChannelCorrelatorResult],
    strong_tensor_result: ChannelCorrelatorResult | None,
    tensor_momentum_results: dict[str, ChannelCorrelatorResult] | None,
    tensor_momentum_meta: dict[str, Any] | None,
    noncomp_multiscale_output: MultiscaleStrongForceOutput | None,
    companion_multiscale_output: MultiscaleStrongForceOutput | None,
    status_lines: list[str] | None = None,
) -> dict[str, Any]:
    """Populate tab from tensor computation outputs and return correction payload."""
    mode = str(w.mass_mode.value)
    enabled = {str(v) for v in (w.estimator_toggles.value or [])}

    momentum_axis = int((tensor_momentum_meta or {}).get("momentum_axis", 0))
    momentum_length = float((tensor_momentum_meta or {}).get("momentum_length_scale", float("nan")))
    momentum_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    estimator_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []

    def append_row(
        *,
        estimator: str,
        approach: str,
        result_obj: ChannelCorrelatorResult | None,
        is_core: bool,
        n_mode: int | None = None,
        component: str | None = None,
    ) -> None:
        if result_obj is None or result_obj.n_samples <= 0:
            return
        mass = _get_channel_mass(result_obj, mode)
        if not np.isfinite(mass) or mass <= 0:
            return
        err = _get_channel_mass_error(result_obj, mode)
        r2 = _get_channel_r2(result_obj, mode)
        p_value = float("nan")
        p2_value = float("nan")
        if n_mode is not None and np.isfinite(momentum_length) and momentum_length > 0:
            p_value = (2.0 * np.pi * float(n_mode)) / float(momentum_length)
            p2_value = p_value * p_value
        row = {
            "estimator": estimator,
            "approach": approach,
            "component": component or "",
            "n_mode": n_mode if n_mode is not None else "",
            "mass": float(mass),
            "mass_error": float(err) if np.isfinite(err) else np.nan,
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "p": float(p_value) if np.isfinite(p_value) else np.nan,
            "p2": float(p2_value) if np.isfinite(p2_value) else np.nan,
            "core_estimator": bool(is_core),
        }
        estimator_rows.append(row)
        if is_core:
            core_rows.append(row)
        if n_mode is not None and np.isfinite(p2_value):
            target = component_rows if component else momentum_rows
            target.append(
                {
                    "estimator": estimator,
                    "component": component or "",
                    "n_mode": int(n_mode),
                    "p": float(p_value),
                    "p2": float(p2_value),
                    "mass": float(mass),
                    "mass_error": float(err) if np.isfinite(err) else np.nan,
                }
            )

    if "anisotropic_edge" in enabled:
        append_row(
            estimator="Anisotropic-edge tensor",
            approach="anisotropic_edge",
            result_obj=base_results.get("tensor"),
            is_core=True,
        )
    if "anisotropic_edge_traceless" in enabled:
        append_row(
            estimator="Anisotropic-edge tensor traceless",
            approach="anisotropic_edge_traceless",
            result_obj=base_results.get("tensor_traceless"),
            is_core=True,
        )
    if "strong_force" in enabled:
        append_row(
            estimator="Strong-force tensor",
            approach="strong_force",
            result_obj=strong_tensor_result,
            is_core=True,
        )
    momentum_map = dict(tensor_momentum_results or {})
    if "momentum_contracted" in enabled:
        for channel_name in sorted(momentum_map.keys()):
            mode_n, component = _parse_tensor_momentum_name(channel_name)
            if mode_n is None or component is not None:
                continue
            append_row(
                estimator=f"Tensor momentum p{mode_n}",
                approach="momentum_contracted",
                result_obj=momentum_map[channel_name],
                is_core=(mode_n == 0),
                n_mode=mode_n,
                component=None,
            )
    if "momentum_components" in enabled:
        for channel_name in sorted(momentum_map.keys()):
            mode_n, component = _parse_tensor_momentum_name(channel_name)
            if mode_n is None or component is None:
                continue
            append_row(
                estimator=f"Tensor momentum p{mode_n} {component}",
                approach="momentum_component",
                result_obj=momentum_map[channel_name],
                is_core=False,
                n_mode=mode_n,
                component=component,
            )

    consensus_mass = float("nan")
    consensus_stat = float("nan")
    consensus_syst = float("nan")
    if core_rows:
        masses = np.asarray([float(row["mass"]) for row in core_rows], dtype=float)
        errs = np.asarray([float(row["mass_error"]) for row in core_rows], dtype=float)
        valid_w = np.isfinite(errs) & (errs > 0)
        if np.any(valid_w):
            wgt = 1.0 / np.maximum(errs[valid_w], 1e-12) ** 2
            consensus_mass = float(np.sum(wgt * masses[valid_w]) / np.sum(wgt))
            consensus_stat = float(np.sqrt(1.0 / np.sum(wgt)))
        else:
            consensus_mass = float(np.mean(masses))
            consensus_stat = (
                float(np.std(masses, ddof=1) / np.sqrt(float(masses.size)))
                if masses.size > 1
                else float("nan")
            )
        consensus_syst = float(np.std(masses, ddof=1)) if masses.size > 1 else 0.0

    for row in estimator_rows:
        row["delta_vs_consensus_pct"] = _delta_percent(float(row["mass"]), consensus_mass)

    if estimator_rows:
        table_df = pd.DataFrame(estimator_rows)
        table_df["approach_order"] = (
            table_df["estimator"]
            .map({name: i for i, name in enumerate(CORE_ESTIMATOR_ORDER)})
            .fillna(10)
            .astype(int)
        )
        table_df["mode_sort"] = pd.to_numeric(table_df["n_mode"], errors="coerce").fillna(-1).astype(int)
        table_df["component_sort"] = table_df["component"].apply(lambda c: 0 if not str(c) else 1)
        table_df = table_df.sort_values(
            ["approach_order", "mode_sort", "component_sort", "estimator"]
        ).drop(columns=["approach_order", "mode_sort", "component_sort"], errors="ignore")
        w.estimator_table.value = table_df
    else:
        w.estimator_table.value = pd.DataFrame()

    pairwise_rows: list[dict[str, Any]] = []
    for i in range(len(core_rows)):
        for j in range(i + 1, len(core_rows)):
            a = core_rows[i]
            b = core_rows[j]
            mass_a = float(a["mass"])
            mass_b = float(b["mass"])
            err_a = float(a["mass_error"])
            err_b = float(b["mass_error"])
            ratio = mass_a / mass_b if mass_b > 0 else float("nan")
            delta_pct = (ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan")
            pull = float("nan")
            if np.isfinite(err_a) and np.isfinite(err_b):
                comb = np.sqrt(max(err_a, 0.0) ** 2 + max(err_b, 0.0) ** 2)
                if comb > 0:
                    pull = abs(mass_a - mass_b) / comb
            pairwise_rows.append(
                {
                    "estimator_a": a["estimator"],
                    "estimator_b": b["estimator"],
                    "delta_pct": delta_pct,
                    "abs_delta_pct": abs(delta_pct) if np.isfinite(delta_pct) else np.nan,
                    "pull_sigma": pull,
                }
            )
    if pairwise_rows:
        w.pairwise_table.value = pd.DataFrame(pairwise_rows).sort_values(
            "abs_delta_pct", ascending=False
        )
    else:
        w.pairwise_table.value = pd.DataFrame()

    verdict = "insufficient data"
    verdict_type = "secondary"
    details = "Need at least two core tensor estimators."
    if pairwise_rows:
        abs_vals = np.asarray([r["abs_delta_pct"] for r in pairwise_rows], dtype=float)
        pull_vals = np.asarray([r["pull_sigma"] for r in pairwise_rows], dtype=float)
        max_abs = float(np.nanmax(abs_vals)) if np.isfinite(abs_vals).any() else float("nan")
        max_pull = float(np.nanmax(pull_vals)) if np.isfinite(pull_vals).any() else float("nan")
        details = f"max |Δ%|={max_abs:.2f}%, max pull={max_pull:.2f}σ"
        if np.isfinite(max_abs) and max_abs <= 7.5 and (not np.isfinite(max_pull) or max_pull <= 1.5):
            verdict = "consistent"
            verdict_type = "success"
        elif np.isfinite(max_abs) and max_abs <= 20.0 and (not np.isfinite(max_pull) or max_pull <= 3.0):
            verdict = "mild tension"
            verdict_type = "warning"
        else:
            verdict = "tension"
            verdict_type = "danger"
    w.systematics_badge.object = f"Systematics verdict: {verdict}. {details}"
    w.systematics_badge.alert_type = verdict_type

    summary_lines = ["**Tensor estimator summary:**"]
    if core_rows:
        for row in core_rows:
            mass = float(row["mass"])
            err = float(row["mass_error"])
            delta = float(row.get("delta_vs_consensus_pct", np.nan))
            line = f"- {row['estimator']}: `{mass:.6g}`"
            if np.isfinite(err):
                line += f" ± `{err:.2g}`"
            if np.isfinite(delta):
                line += f", Δ vs consensus `{delta:+.2f}%`"
            summary_lines.append(line)
    else:
        summary_lines.append("- No valid core tensor estimators.")
    if np.isfinite(consensus_mass) and consensus_mass > 0:
        line = f"- Consensus mass: `{consensus_mass:.6g}`"
        if np.isfinite(consensus_stat):
            line += f" ± `{consensus_stat:.2g}` (stat)"
        if np.isfinite(consensus_syst):
            line += f" ± `{consensus_syst:.2g}` (syst)"
        summary_lines.append(line)
    if np.isfinite(momentum_length) and momentum_length > 0:
        summary_lines.append(
            f"- Momentum axis/length: axis `{momentum_axis}`, `L={momentum_length:.6g}`"
        )
    w.summary.object = "  \n".join(summary_lines)

    contracted_disp = _build_dispersion_plot(momentum_rows)
    w.dispersion_plot.object = contracted_disp
    comp_disp = _build_component_dispersion_plot(component_rows)
    w.component_dispersion_plot.object = comp_disp
    anisotropy_rows, anisotropy_stats = _compute_component_anisotropy(component_rows)
    if anisotropy_rows:
        anisotropy_df = pd.DataFrame(anisotropy_rows).sort_values("component")
        anisotropy_cols = [
            "component",
            "n_points",
            "n_fit_points",
            "n_modes",
            "p0_mass",
            "m0",
            "c_eff",
            "xi_p0",
            "xi_m0",
            "xi_c_eff",
            "delta_p0_pct",
            "delta_m0_pct",
            "delta_c_eff_pct",
        ]
        present_cols = [col for col in anisotropy_cols if col in anisotropy_df.columns]
        w.anisotropy_table.value = anisotropy_df[present_cols]
        anisotropy_lines = ["**Tensor anisotropy coefficients:**"]
        c_eff_ref = float(anisotropy_stats.get("c_eff_ref", float("nan")))
        m0_ref = float(anisotropy_stats.get("m0_ref", float("nan")))
        p0_ref = float(anisotropy_stats.get("p0_ref", float("nan")))
        if np.isfinite(c_eff_ref):
            anisotropy_lines.append(f"- Isotropic reference `c_eff`: `{c_eff_ref:.6g}`")
        if np.isfinite(m0_ref):
            anisotropy_lines.append(f"- Isotropic reference `m0`: `{m0_ref:.6g}`")
        if np.isfinite(p0_ref):
            anisotropy_lines.append(f"- Isotropic reference `p0 mass`: `{p0_ref:.6g}`")
        xi_c_eff_rms = float(anisotropy_stats.get("xi_c_eff_rms_pct", float("nan")))
        xi_c_eff_max = float(anisotropy_stats.get("xi_c_eff_max_abs_pct", float("nan")))
        xi_m0_rms = float(anisotropy_stats.get("xi_m0_rms_pct", float("nan")))
        xi_m0_max = float(anisotropy_stats.get("xi_m0_max_abs_pct", float("nan")))
        if np.isfinite(xi_c_eff_rms):
            anisotropy_lines.append(
                f"- `xi_c_eff` RMS anisotropy: `{xi_c_eff_rms:.2f}%`, max deviation `{xi_c_eff_max:.2f}%`"
            )
        if np.isfinite(xi_m0_rms):
            anisotropy_lines.append(
                f"- `xi_m0` RMS anisotropy: `{xi_m0_rms:.2f}%`, max deviation `{xi_m0_max:.2f}%`"
            )
        w.anisotropy_summary.object = "  \n".join(anisotropy_lines)
        w.anisotropy_plot.object = _build_anisotropy_plot(anisotropy_rows)
    else:
        w.anisotropy_summary.object = (
            "**Tensor anisotropy coefficients:** insufficient component momentum data."
        )
        w.anisotropy_table.value = pd.DataFrame()
        w.anisotropy_plot.object = None
    w.multiscale_plot.object = _build_multiscale_curve_plot(
        noncomp_multiscale_output if "multiscale_non_companion" in enabled else None,
        companion_multiscale_output if "multiscale_companion" in enabled else None,
        mode,
    )

    scales, grid, op_names = _collect_surface_grid(
        noncomp_multiscale_output if "multiscale_non_companion" in enabled else None,
        companion_multiscale_output if "multiscale_companion" in enabled else None,
        mode,
    )
    fit = None
    cv = None
    if scales is not None and grid is not None and len(op_names) > 0:
        w.calibration_surface_plot.object = _grid_heatmap(
            scales,
            op_names,
            grid,
            title="Tensor calibration surface m(op, R)",
            value_label="mass",
        )
        fit = _fit_shared_exp_surface(scales, grid)
        if fit is not None:
            w.calibration_residual_plot.object = _grid_heatmap(
                scales,
                op_names,
                fit["residual_grid"],
                title="Tensor calibration residuals",
                value_label="residual",
            )
            cv = _cross_validate_surface(scales, grid)
        else:
            w.calibration_residual_plot.object = None
    else:
        w.calibration_surface_plot.object = None
        w.calibration_residual_plot.object = None

    if cv is not None and cv.get("n_points", 0) > 0:
        w.cv_summary.object = (
            "**Calibration cross-validation:** "
            f"`n={cv['n_points']}`, mean frac error `{cv['mean_frac_error_pct']:.2f}%`, "
            f"median `{cv['median_frac_error_pct']:.2f}%`, p95 `{cv['p95_frac_error_pct']:.2f}%`, "
            f"RMS abs error `{cv['rms_error']:.4g}`"
        )
    else:
        w.cv_summary.object = (
            "**Calibration cross-validation:** insufficient multiscale tensor grid for LOOCV."
        )

    reference_mass = float("nan")
    tensor_base = base_results.get("tensor")
    if tensor_base is not None:
        reference_mass = _get_channel_mass(tensor_base, mode)
    correction_scale = float("nan")
    if np.isfinite(consensus_mass) and consensus_mass > 0 and np.isfinite(reference_mass) and reference_mass > 0:
        correction_scale = float(consensus_mass / reference_mass)

    m0, c_eff = _fit_dispersion(momentum_rows)
    correction_lines = ["**Tensor correction:**"]
    if np.isfinite(correction_scale) and correction_scale > 0:
        correction_lines.append(
            f"- Global correction scale (consensus / anisotropic tensor): `{correction_scale:.6g}`"
        )
    else:
        correction_lines.append("- Global correction scale: n/a")
    if fit is not None:
        correction_lines.append(f"- Shared multiscale decay length `r_scale`: `{fit['r_scale']:.6g}`")
    if np.isfinite(m0):
        correction_lines.append(f"- Momentum-dispersion intercept `m0`: `{m0:.6g}`")
    if np.isfinite(c_eff):
        correction_lines.append(f"- Momentum-dispersion `c_eff`: `{c_eff:.6g}`")
    w.correction_summary.object = "  \n".join(correction_lines)

    status_parts = status_lines or []
    if status_parts:
        w.status.object = "  \n".join(status_parts)

    payload = {
        "correction_scale": correction_scale,
        "consensus_mass": consensus_mass,
        "consensus_stat_error": consensus_stat,
        "consensus_syst_error": consensus_syst,
        "verdict": verdict,
        "verdict_details": details,
        "mode": mode,
        "dispersion_m0": m0,
        "dispersion_ceff": c_eff,
        "anisotropy_component_count": len(anisotropy_rows),
        "anisotropy_xi_ceff_rms_pct": float(
            anisotropy_stats.get("xi_c_eff_rms_pct", float("nan"))
        )
        if anisotropy_rows
        else float("nan"),
        "anisotropy_xi_ceff_max_abs_pct": float(
            anisotropy_stats.get("xi_c_eff_max_abs_pct", float("nan"))
        )
        if anisotropy_rows
        else float("nan"),
        "anisotropy_xi_m0_rms_pct": float(anisotropy_stats.get("xi_m0_rms_pct", float("nan")))
        if anisotropy_rows
        else float("nan"),
        "anisotropy_xi_m0_max_abs_pct": float(
            anisotropy_stats.get("xi_m0_max_abs_pct", float("nan"))
        )
        if anisotropy_rows
        else float("nan"),
    }
    if fit is not None:
        payload["r_scale"] = float(fit["r_scale"])
    if cv is not None:
        payload["cv_mean_frac_error_pct"] = float(cv.get("mean_frac_error_pct", float("nan")))
    return payload
