"""Strong Force AIC dashboard tab.

Feeds companion-correlator PipelineResult into the AIC windowed fitter,
or runs direct multiscale strong-force analysis, then displays extracted
masses, window heatmaps, mass ratios, and per-channel diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.physics.aic.mass_extraction_adapter import _fit_channel
from fragile.physics.aic.multiscale_strong_force import (
    compute_multiscale_strong_force_channels,
    MultiscaleStrongForceConfig,
    MultiscaleStrongForceOutput,
)
from fragile.physics.aic.plotting import (
    build_mass_spectrum_bar,
    build_window_heatmap,
    ChannelPlot,
    CHANNEL_COLORS,
)
from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.new_channels.correlator_channels import (
    ChannelCorrelatorResult,
    CorrelatorConfig,
)
from fragile.physics.operators.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class StrongForceAICSection:
    """Container for the strong-force AIC dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]
    on_correlators_ready: Callable[[], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class StrongForceAICSettings(param.Parameterized):
    """Settings for strong-force AIC mass extraction."""

    # AIC fitting
    window_widths_spec = param.String(
        default="5-50",
        doc="Window widths: range 'lo-hi' or comma-separated list.",
    )
    fit_mode = param.ObjectSelector(
        default="aic",
        objects=["aic", "linear", "linear_abs"],
    )
    fit_start = param.Integer(default=2, bounds=(0, 100))
    fit_stop = param.Integer(
        default=0,
        bounds=(0, 500),
        doc="Fit stop lag (0 = auto).",
    )
    min_fit_points = param.Integer(default=2, bounds=(1, 50))

    # Quality filters
    best_min_r2 = param.Number(default=-1.0)
    best_min_windows = param.Integer(default=0, bounds=(0, 100))
    best_max_error_pct = param.Number(default=30.0, bounds=(0.0, None))
    best_remove_artifacts = param.Boolean(default=False)

    # Bootstrap
    compute_bootstrap_errors = param.Boolean(default=False)
    n_bootstrap = param.Integer(default=100, bounds=(10, 10000))

    # Multiscale settings
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.95))
    end_fraction = param.Number(default=1.0, bounds=(0.05, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    max_lag = param.Integer(default=80, bounds=(1, 500))
    use_connected = param.Boolean(default=True)
    n_scales = param.Integer(default=8, bounds=(2, 32))
    kernel_type = param.ObjectSelector(
        default="gaussian",
        objects=["gaussian", "exponential", "tophat", "shell"],
    )
    kernel_batch_size = param.Integer(default=1, bounds=(1, 8))
    edge_weight_mode = param.ObjectSelector(
        default="riemannian_kernel_volume",
        objects=[
            "uniform",
            "inverse_distance",
            "inverse_volume",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "kernel",
            "riemannian_kernel",
            "riemannian_kernel_volume",
        ],
    )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _parse_window_widths(spec: str) -> list[int] | None:
    """Parse window widths specification.

    Accepts ``"5-50"`` (range) or ``"5,10,15,20"`` (explicit list).
    Returns ``None`` for empty/blank input.
    """
    spec = spec.strip()
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        parts = spec.split("-")
        lo, hi = int(parts[0]), int(parts[1])
        return list(range(lo, hi + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _build_correlator_config(settings: StrongForceAICSettings) -> CorrelatorConfig:
    """Build a CorrelatorConfig from the current settings."""
    widths = _parse_window_widths(str(settings.window_widths_spec))
    fit_stop = int(settings.fit_stop)
    return CorrelatorConfig(
        max_lag=int(settings.max_lag),
        use_connected=bool(settings.use_connected),
        window_widths=widths,
        fit_mode=str(settings.fit_mode),
        fit_start=int(settings.fit_start),
        fit_stop=fit_stop if fit_stop > 0 else None,
        min_fit_points=int(settings.min_fit_points),
        compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
        n_bootstrap=int(settings.n_bootstrap),
    )


def _get_mass(result: ChannelCorrelatorResult, mode: str) -> float:
    """Extract mass from a result, respecting display mode."""
    fit = result.mass_fit or {}
    if mode == "Best Window":
        bw = fit.get("best_window", {})
        if isinstance(bw, dict):
            return float(bw.get("mass", float("nan")))
    return float(fit.get("mass", float("nan")))


def _get_mass_error(result: ChannelCorrelatorResult, mode: str) -> float:
    """Extract mass error from a result, respecting display mode."""
    fit = result.mass_fit or {}
    if mode == "Best Window":
        bw = fit.get("best_window", {})
        if isinstance(bw, dict):
            return float(bw.get("mass_error", float("inf")))
    return float(fit.get("mass_error", float("inf")))


def _get_r2(result: ChannelCorrelatorResult, mode: str) -> float:
    """Extract R-squared from a result, respecting display mode."""
    fit = result.mass_fit or {}
    if mode == "Best Window":
        bw = fit.get("best_window", {})
        if isinstance(bw, dict):
            return float(bw.get("r_squared", float("nan")))
    return float(fit.get("r_squared", float("nan")))


def _build_aic_mass_table(
    results: dict[str, ChannelCorrelatorResult],
    mode: str,
) -> pd.DataFrame:
    """Build a mass summary DataFrame from AIC results."""
    rows = []
    for ch_name, result in results.items():
        mass = _get_mass(result, mode)
        error = _get_mass_error(result, mode)
        r2 = _get_r2(result, mode)
        err_pct = abs(error / mass) * 100.0 if mass != 0 and math.isfinite(mass) else float("inf")
        n_windows = 0
        wm = result.window_masses
        if wm is not None:
            if isinstance(wm, torch.Tensor):
                n_windows = int(torch.isfinite(wm).sum().item())
            elif isinstance(wm, np.ndarray):
                n_windows = int(np.isfinite(wm).sum())
        rows.append({
            "Channel": ch_name,
            "Mass": f"{mass:.6f}" if math.isfinite(mass) else "N/A",
            "Error": f"{error:.6f}" if math.isfinite(error) else "N/A",
            "Error %": f"{err_pct:.2f}%" if math.isfinite(err_pct) else "N/A",
            "R2": f"{r2:.4f}" if math.isfinite(r2) else "N/A",
            "Windows": n_windows,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _split_results_by_quality(
    results: dict[str, ChannelCorrelatorResult],
    mode: str,
    *,
    min_r2: float,
    min_windows: int,
    max_error_pct: float,
    remove_artifacts: bool,
) -> tuple[dict[str, ChannelCorrelatorResult], dict[str, ChannelCorrelatorResult]]:
    """Split results into kept and filtered-out dicts based on quality criteria."""
    kept: dict[str, ChannelCorrelatorResult] = {}
    filtered: dict[str, ChannelCorrelatorResult] = {}
    for ch_name, result in results.items():
        mass = _get_mass(result, mode)
        error = _get_mass_error(result, mode)
        r2 = _get_r2(result, mode)

        if not math.isfinite(mass) or mass <= 0:
            filtered[ch_name] = result
            continue

        if math.isfinite(min_r2) and min_r2 > -1.0:
            if not math.isfinite(r2) or r2 < min_r2:
                filtered[ch_name] = result
                continue

        # Count valid windows
        n_windows = 0
        wm = result.window_masses
        if wm is not None:
            if isinstance(wm, torch.Tensor):
                n_windows = int(torch.isfinite(wm).sum().item())
            elif isinstance(wm, np.ndarray):
                n_windows = int(np.isfinite(wm).sum())
        if n_windows < max(0, int(min_windows)):
            filtered[ch_name] = result
            continue

        if math.isfinite(max_error_pct) and max_error_pct >= 0:
            err_pct = abs(error / mass) * 100.0 if mass > 0 else float("inf")
            if err_pct > max_error_pct:
                filtered[ch_name] = result
                continue

        if remove_artifacts:
            if not math.isfinite(error) or error == 0.0 or mass == 0.0:
                filtered[ch_name] = result
                continue

        kept[ch_name] = result

    return kept, filtered


def _build_multiscale_best_table(output: MultiscaleStrongForceOutput) -> pd.DataFrame:
    """Build a DataFrame showing the best scale per channel."""
    rows = []
    for ch_name, result in output.best_results.items():
        fit = result.mass_fit or {}
        mass = float(fit.get("mass", float("nan")))
        error = float(fit.get("mass_error", float("inf")))
        r2 = float(fit.get("r_squared", float("nan")))
        scale_idx = output.best_scale_index.get(ch_name, -1)
        scale_val = float("nan")
        if scale_idx >= 0 and scale_idx < len(output.scales):
            scale_val = float(output.scales[scale_idx].item())
        rows.append({
            "Channel": ch_name,
            "Best Scale Idx": scale_idx,
            "Scale": f"{scale_val:.6f}" if math.isfinite(scale_val) else "N/A",
            "Mass": f"{mass:.6f}" if math.isfinite(mass) else "N/A",
            "Error": f"{error:.6f}" if math.isfinite(error) else "N/A",
            "R2": f"{r2:.4f}" if math.isfinite(r2) else "N/A",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_multiscale_per_scale_table(output: MultiscaleStrongForceOutput) -> pd.DataFrame:
    """Build a DataFrame with all scales x channels."""
    rows = []
    n_scales = int(output.scales.numel())
    for ch_name, scale_results in output.per_scale_results.items():
        for si in range(min(n_scales, len(scale_results))):
            result = scale_results[si]
            fit = result.mass_fit or {}
            mass = float(fit.get("mass", float("nan")))
            error = float(fit.get("mass_error", float("inf")))
            r2 = float(fit.get("r_squared", float("nan")))
            scale_val = float(output.scales[si].item())
            rows.append({
                "Channel": ch_name,
                "Scale Idx": si,
                "Scale": f"{scale_val:.6f}",
                "Mass": f"{mass:.6f}" if math.isfinite(mass) else "N/A",
                "Error": f"{error:.6f}" if math.isfinite(error) else "N/A",
                "R2": f"{r2:.4f}" if math.isfinite(r2) else "N/A",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_multiscale_mass_curves(output: MultiscaleStrongForceOutput) -> hv.Overlay | None:
    """Build an overlay of mass-vs-scale curves for each channel."""
    if output.scales.numel() == 0:
        return None

    scales_np = output.scales.cpu().numpy()
    overlays = []
    for ch_name, scale_results in output.per_scale_results.items():
        masses = []
        errors = []
        scale_vals = []
        for si, result in enumerate(scale_results):
            if si >= len(scales_np):
                break
            fit = result.mass_fit or {}
            m = float(fit.get("mass", float("nan")))
            e = float(fit.get("mass_error", float("nan")))
            if math.isfinite(m) and m > 0:
                masses.append(m)
                errors.append(e if math.isfinite(e) else 0.0)
                scale_vals.append(float(scales_np[si]))
        if not masses:
            continue
        color = CHANNEL_COLORS.get(ch_name, "#1f77b4")
        curve = hv.Curve(
            (scale_vals, masses), "Scale", "Mass", label=ch_name,
        ).opts(color=color, line_width=2)
        scatter = hv.Scatter(
            (scale_vals, masses), "Scale", "Mass",
        ).opts(color=color, size=6)
        overlays.append(curve * scatter)

    if not overlays:
        return None
    combined = overlays[0]
    for ov in overlays[1:]:
        combined = combined * ov
    return combined.opts(
        width=700,
        height=350,
        title="Mass vs Scale",
        show_legend=True,
        legend_position="top_right",
    )


def _build_ratio_table(
    results: dict[str, ChannelCorrelatorResult],
    mode: str,
) -> pd.DataFrame:
    """Build a mass ratio DataFrame between all channel pairs."""
    names = sorted(results.keys())
    rows = []
    for i, name_a in enumerate(names):
        mass_a = _get_mass(results[name_a], mode)
        for name_b in names[i + 1:]:
            mass_b = _get_mass(results[name_b], mode)
            if mass_a > 0 and mass_b > 0 and math.isfinite(mass_a) and math.isfinite(mass_b):
                ratio = mass_a / mass_b
                rows.append({
                    "Channel A": name_a,
                    "Channel B": name_b,
                    "Mass A": f"{mass_a:.6f}",
                    "Mass B": f"{mass_b:.6f}",
                    "Ratio A/B": f"{ratio:.6f}",
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_strong_force_aic_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
    on_aic_computed: Callable[[], None] | None = None,
) -> StrongForceAICSection:
    """Build the Strong Force AIC tab with callbacks."""

    settings = StrongForceAICSettings()
    status = pn.pane.Markdown(
        "**Strong Force AIC:** Run Companion Correlators first, "
        "then click *AIC from Correlators*; or click *Direct Multiscale*.",
        sizing_mode="stretch_width",
    )

    # -- Buttons --
    run_from_correlators_btn = pn.widgets.Button(
        name="AIC from Correlators",
        button_type="primary",
        min_width=200,
        sizing_mode="stretch_width",
        disabled=True,
    )
    run_multiscale_btn = pn.widgets.Button(
        name="Direct Multiscale",
        button_type="warning",
        min_width=200,
        sizing_mode="stretch_width",
        disabled=True,
    )
    display_plots_btn = pn.widgets.Button(
        name="Display Plots",
        button_type="default",
        min_width=160,
        sizing_mode="stretch_width",
    )

    # -- Display widgets --
    mass_mode_selector = pn.widgets.RadioButtonGroup(
        name="Mass mode",
        options=["AIC-Weighted", "Best Window"],
        value="AIC-Weighted",
    )
    summary_md = pn.pane.Markdown("", sizing_mode="stretch_width")
    mass_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    filtered_summary = pn.pane.Markdown("", sizing_mode="stretch_width")
    filtered_mass_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    spectrum_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run extraction to show mass spectrum."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )

    # -- Multiscale widgets --
    multiscale_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    per_scale_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=20,
        show_index=False,
        sizing_mode="stretch_width",
    )
    mass_curves_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run multiscale to show mass curves."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )

    # -- Ratio table --
    ratio_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    # -- Heatmap widgets --
    heatmap_color_metric = pn.widgets.Select(
        name="Color metric", options=["mass", "aic", "r2"], value="mass",
    )
    heatmap_alpha_metric = pn.widgets.Select(
        name="Alpha metric", options=["mass", "aic", "r2"], value="aic",
    )
    heatmap_container = pn.Column(sizing_mode="stretch_width")

    # -- Per-channel plateau plots --
    plateau_container = pn.Column(sizing_mode="stretch_width")

    # -- Settings panels --
    aic_settings_panel = pn.Param(
        settings,
        parameters=[
            "window_widths_spec",
            "fit_mode",
            "fit_start",
            "fit_stop",
            "min_fit_points",
            "best_min_r2",
            "best_min_windows",
            "best_max_error_pct",
            "best_remove_artifacts",
            "compute_bootstrap_errors",
            "n_bootstrap",
        ],
        show_name=False,
        default_layout=type("AICGrid", (pn.GridBox,), {"ncols": 2}),
    )
    multiscale_settings_panel = pn.Param(
        settings,
        parameters=[
            "warmup_fraction",
            "end_fraction",
            "h_eff",
            "mass",
            "ell0",
            "max_lag",
            "use_connected",
            "n_scales",
            "kernel_type",
            "kernel_batch_size",
            "edge_weight_mode",
        ],
        show_name=False,
        default_layout=type("MultiscaleGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Refresh helpers --

    def _refresh_tables():
        results: dict[str, ChannelCorrelatorResult] | None = state.get(
            "strong_force_aic_output",
        )
        if results is None:
            return

        mode = str(mass_mode_selector.value)

        # Split by quality
        kept, filtered = _split_results_by_quality(
            results,
            mode,
            min_r2=float(settings.best_min_r2),
            min_windows=int(settings.best_min_windows),
            max_error_pct=float(settings.best_max_error_pct),
            remove_artifacts=bool(settings.best_remove_artifacts),
        )

        # Mass table (kept channels)
        mass_table.value = _build_aic_mass_table(kept, mode)
        summary_md.object = f"**{len(kept)}** channels passed quality filters ({mode} mode)."

        # Filtered table
        if filtered:
            filtered_summary.object = (
                f"**{len(filtered)}** channels filtered out."
            )
            filtered_mass_table.value = _build_aic_mass_table(filtered, mode)
        else:
            filtered_summary.object = "*No channels filtered out.*"
            filtered_mass_table.value = pd.DataFrame()

        # Spectrum bar
        bar = build_mass_spectrum_bar(
            kept,
            mass_getter=lambda r, _m=mode: _get_mass(r, _m),
            error_getter=lambda r, _m=mode: _get_mass_error(r, _m),
            title="Strong Force Mass Spectrum",
        )
        spectrum_plot.object = bar if bar is not None else _algorithm_placeholder_plot(
            "No valid masses to display.",
        )

        # Ratio table
        ratio_table.value = _build_ratio_table(kept, mode)

    def _refresh_multiscale():
        ms_output: MultiscaleStrongForceOutput | None = state.get(
            "strong_force_aic_multiscale_output",
        )
        if ms_output is None:
            return
        multiscale_table.value = _build_multiscale_best_table(ms_output)
        per_scale_table.value = _build_multiscale_per_scale_table(ms_output)

        curves = _build_multiscale_mass_curves(ms_output)
        mass_curves_plot.object = curves if curves is not None else _algorithm_placeholder_plot(
            "No multiscale mass curves to display.",
        )

    def _refresh_heatmaps():
        results: dict[str, ChannelCorrelatorResult] | None = state.get(
            "strong_force_aic_output",
        )
        if results is None:
            return
        heatmap_container.clear()
        color_met = str(heatmap_color_metric.value)
        alpha_met = str(heatmap_alpha_metric.value)
        for ch_name, result in results.items():
            wm = result.window_masses
            wa = result.window_aic
            ww = result.window_widths
            if wm is None or wa is None or ww is None:
                continue
            wm_np = wm.cpu().numpy() if isinstance(wm, torch.Tensor) else np.asarray(wm)
            wa_np = wa.cpu().numpy() if isinstance(wa, torch.Tensor) else np.asarray(wa)
            wr2 = result.window_r2
            wr2_np = None
            if wr2 is not None:
                wr2_np = wr2.cpu().numpy() if isinstance(wr2, torch.Tensor) else np.asarray(wr2)
            bw = (result.mass_fit or {}).get("best_window", {})
            if not isinstance(bw, dict):
                bw = {}
            hm = build_window_heatmap(
                window_masses=wm_np,
                window_aic=wa_np,
                window_widths=list(ww),
                best_window=bw,
                channel_name=ch_name,
                window_r2=wr2_np,
                color_metric=color_met,
                alpha_metric=alpha_met,
            )
            if hm is not None:
                heatmap_container.append(pn.pane.Markdown(f"#### {ch_name}"))
                heatmap_container.append(
                    pn.pane.HoloViews(hm, sizing_mode="stretch_width", linked_axes=False),
                )

    def _refresh_plateau_plots():
        results: dict[str, ChannelCorrelatorResult] | None = state.get(
            "strong_force_aic_output",
        )
        if results is None:
            return
        plateau_container.clear()
        for ch_name, result in results.items():
            cp = ChannelPlot(result)
            row = cp.side_by_side()
            if row is not None:
                plateau_container.append(pn.pane.Markdown(f"#### {ch_name}"))
                plateau_container.append(row)

    # -- Watch callbacks --

    mass_mode_selector.param.watch(lambda _: _refresh_tables(), "value")
    heatmap_color_metric.param.watch(lambda _: _refresh_heatmaps(), "value")
    heatmap_alpha_metric.param.watch(lambda _: _refresh_heatmaps(), "value")

    def _on_display_plots(_event):
        _refresh_plateau_plots()
        _refresh_heatmaps()

    display_plots_btn.on_click(_on_display_plots)

    # -- Compute: AIC from Correlators --

    def _on_run_from_correlators(_event):
        def _compute(_history: RunHistory):
            pipeline_result: PipelineResult | None = state.get("companion_correlator_output")
            if pipeline_result is None:
                status.object = "**Error:** Run Companion Correlators first."
                return

            config = _build_correlator_config(settings)
            dt = 1.0
            results: dict[str, ChannelCorrelatorResult] = {}

            for ch_name, correlator in pipeline_result.correlators.items():
                series = pipeline_result.operators.get(ch_name)
                if series is None:
                    continue

                # Handle multiscale (2D correlator [S, max_lag+1])
                if correlator.dim() == 2:
                    # Use the first scale (scale 0) as default
                    correlator_1d = correlator[0]
                    series_1d = series[0] if series.dim() == 2 else series
                else:
                    correlator_1d = correlator
                    series_1d = series

                result = _fit_channel(
                    channel_name=ch_name,
                    correlator=correlator_1d,
                    series=series_1d,
                    dt=dt,
                    config=config,
                )
                results[ch_name] = result

            state["strong_force_aic_output"] = results

            _refresh_tables()

            n_ch = len(results)
            n_valid = sum(
                1
                for r in results.values()
                if math.isfinite(_get_mass(r, "AIC-Weighted")) and _get_mass(r, "AIC-Weighted") > 0
            )
            status.object = (
                f"**AIC Complete:** {n_valid}/{n_ch} channels with valid masses."
            )
            if on_aic_computed is not None:
                on_aic_computed()

        run_tab_computation(state, status, "strong force AIC extraction", _compute)

    run_from_correlators_btn.on_click(_on_run_from_correlators)

    # -- Compute: Direct Multiscale --

    def _on_run_multiscale(_event):
        def _compute(history: RunHistory):
            widths = _parse_window_widths(str(settings.window_widths_spec))
            fit_stop = int(settings.fit_stop)
            ms_config = MultiscaleStrongForceConfig(
                warmup_fraction=float(settings.warmup_fraction),
                end_fraction=float(settings.end_fraction),
                h_eff=float(settings.h_eff),
                mass=float(settings.mass),
                ell0=float(settings.ell0) if settings.ell0 is not None else None,
                edge_weight_mode=str(settings.edge_weight_mode),
                n_scales=int(settings.n_scales),
                kernel_type=str(settings.kernel_type),
                kernel_batch_size=int(settings.kernel_batch_size),
                max_lag=int(settings.max_lag),
                use_connected=bool(settings.use_connected),
                fit_mode=str(settings.fit_mode),
                fit_start=int(settings.fit_start),
                fit_stop=fit_stop if fit_stop > 0 else None,
                min_fit_points=int(settings.min_fit_points),
                window_widths=widths,
                best_min_r2=float(settings.best_min_r2),
                best_min_windows=int(settings.best_min_windows),
                best_max_error_pct=float(settings.best_max_error_pct),
                best_remove_artifacts=bool(settings.best_remove_artifacts),
                compute_bootstrap_errors=bool(settings.compute_bootstrap_errors),
                n_bootstrap=int(settings.n_bootstrap),
            )

            ms_output = compute_multiscale_strong_force_channels(
                history,
                config=ms_config,
            )
            state["strong_force_aic_multiscale_output"] = ms_output

            # Populate AIC output from best results
            state["strong_force_aic_output"] = dict(ms_output.best_results)

            _refresh_tables()
            _refresh_multiscale()

            n_ch = len(ms_output.best_results)
            n_s = int(ms_output.scales.numel())
            status.object = (
                f"**Multiscale Complete:** {n_ch} channels across "
                f"{n_s} scales ({settings.kernel_type} kernel)."
            )
            if on_aic_computed is not None:
                on_aic_computed()

        run_tab_computation(state, status, "direct multiscale strong force", _compute)

    run_multiscale_btn.on_click(_on_run_multiscale)

    # -- on_history_changed --

    def on_history_changed(defer: bool) -> None:
        run_from_correlators_btn.disabled = True
        run_multiscale_btn.disabled = False
        status.object = (
            "**Strong Force AIC:** Run Companion Correlators first, "
            "then click *AIC from Correlators*; or click *Direct Multiscale*."
        )
        if defer:
            return
        state["strong_force_aic_output"] = None
        state["strong_force_aic_multiscale_output"] = None
        summary_md.object = ""
        mass_table.value = pd.DataFrame()
        filtered_summary.object = ""
        filtered_mass_table.value = pd.DataFrame()
        spectrum_plot.object = _algorithm_placeholder_plot(
            "Run extraction to show mass spectrum.",
        )
        multiscale_table.value = pd.DataFrame()
        per_scale_table.value = pd.DataFrame()
        mass_curves_plot.object = _algorithm_placeholder_plot(
            "Run multiscale to show mass curves.",
        )
        ratio_table.value = pd.DataFrame()
        heatmap_container.clear()
        plateau_container.clear()

    # -- on_correlators_ready --

    def on_correlators_ready() -> None:
        run_from_correlators_btn.disabled = False
        status.object = (
            "**Strong Force AIC ready:** Companion correlators available. "
            "Click *AIC from Correlators* or *Direct Multiscale*."
        )

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            "**Strong Force AIC:** Performs AIC windowed mass extraction "
            "on companion-correlator output, or runs direct multiscale "
            "strong-force analysis.  Displays extracted masses, window "
            "heatmaps, mass ratios, and per-channel diagnostics."
        ),
        alert_type="info",
        sizing_mode="stretch_width",
    )

    tab = pn.Column(
        status,
        info_note,
        pn.Row(
            run_from_correlators_btn,
            run_multiscale_btn,
            display_plots_btn,
            sizing_mode="stretch_width",
        ),
        pn.Accordion(
            ("AIC Fitting Settings", aic_settings_panel),
            ("Multiscale Settings", multiscale_settings_panel),
            active=[],
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.Row(mass_mode_selector, sizing_mode="stretch_width"),
        summary_md,
        pn.pane.Markdown("### Mass Table"),
        mass_table,
        pn.pane.Markdown("### Filtered-Out Channels"),
        filtered_summary,
        filtered_mass_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Spectrum"),
        spectrum_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Multiscale Analysis"),
        multiscale_table,
        per_scale_table,
        mass_curves_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Ratios"),
        ratio_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### Window Heatmaps"),
        pn.Row(heatmap_color_metric, heatmap_alpha_metric, sizing_mode="stretch_width"),
        heatmap_container,
        pn.layout.Divider(),
        pn.pane.Markdown("### Per-Channel Correlator & Effective Mass"),
        plateau_container,
        sizing_mode="stretch_both",
    )

    return StrongForceAICSection(
        tab=tab,
        status=status,
        run_button=run_from_correlators_btn,
        on_run=_on_run_from_correlators,
        on_history_changed=on_history_changed,
        on_correlators_ready=on_correlators_ready,
    )
