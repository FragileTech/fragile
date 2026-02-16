"""Mass Extraction dashboard tab.

Feeds the strong-correlator PipelineResult into the Bayesian
multi-exponential fitter, displays extracted masses, fit diagnostics,
effective-mass plots, and correlator-vs-fit overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gvar
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.mass_extraction import (
    MassExtractionConfig,
    MassExtractionResult,
    ChannelFitConfig,
    CovarianceConfig,
    FitConfig,
    PriorConfig,
    extract_masses,
)


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class MassExtractionSection:
    """Container for the mass extraction dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]
    on_correlators_ready: Callable[[], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class MassExtractionSettings(param.Parameterized):
    """Settings for the mass extraction pipeline."""

    covariance_method = param.ObjectSelector(
        default="uncorrelated",
        objects=["uncorrelated", "block_jackknife", "bootstrap"],
    )
    nexp = param.Integer(default=2, bounds=(1, 6))
    tmin = param.Integer(default=2, bounds=(1, 20))
    tmax_frac = param.Number(
        default=1.0,
        bounds=(0.1, 1.0),
        doc="Fraction of max_lag for tmax (1.0 = full range).",
    )
    svdcut = param.Number(default=1e-4)
    use_log_dE = param.Boolean(default=True)
    use_fastfit_seeding = param.Boolean(default=True)
    effective_mass_method = param.ObjectSelector(
        default="log_ratio",
        objects=["log_ratio", "cosh"],
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _build_meff_plot(
    result: MassExtractionResult,
    channel_name: str,
    channel_result: Any,
) -> hv.Overlay:
    """Build effective-mass scatter + error bars with fitted mass overlay."""
    # Collect effective-mass entries for this channel's variant keys
    variant_keys = channel_result.variant_keys
    curves = []
    all_means: list[np.ndarray] = []  # central values for y-range

    for vk in variant_keys:
        em = result.effective_masses.get(vk)
        if em is None or len(em.m_eff) == 0:
            continue

        t = em.t_values.astype(float)
        means = np.array([gvar.mean(m) for m in em.m_eff])
        errs = np.array([gvar.sdev(m) for m in em.m_eff])

        # Filter out zero-error / zero-mean placeholder points
        mask = errs > 0
        if not mask.any():
            continue

        t, means, errs = t[mask], means[mask], errs[mask]
        all_means.append(means)

        scatter = hv.Scatter(
            (t, means), kdims="t", vdims="m_eff", label=vk,
        )
        ebars = hv.ErrorBars(
            (t, means, errs), kdims="t", vdims=["m_eff", "yerr"],
        )
        curves.append(scatter * ebars)

    if not curves:
        return _algorithm_placeholder_plot(f"No effective mass data for {channel_name}")

    overlay = curves[0]
    for c in curves[1:]:
        overlay = overlay * c

    # Fitted ground-state mass ± error band
    gs = channel_result.ground_state_mass
    gs_mean = float(gvar.mean(gs))
    gs_err = float(gvar.sdev(gs))

    hline = hv.HLine(gs_mean).opts(color="red", line_dash="dashed", line_width=1.5)
    band = hv.HSpan(gs_mean - gs_err, gs_mean + gs_err).opts(
        color="red", alpha=0.15,
    )

    # Compute y-range from central values only (ignore error bars)
    all_vals = np.concatenate(all_means) if all_means else np.array([gs_mean])
    all_vals = np.concatenate([all_vals, [gs_mean]])
    ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    margin = max(0.1 * (ymax - ymin), 1e-6)
    ylim = (ymin - margin, ymax + margin)

    overlay = overlay * band * hline
    overlay = overlay.opts(
        hv.opts.Scatter(size=6, width=700, height=350),
        hv.opts.ErrorBars(line_width=1.5),
        hv.opts.Overlay(
            title=f"{channel_name} Effective Mass",
            legend_position="top_right",
            ylim=ylim,
        ),
    )
    return overlay


def _build_correlator_fit_plot(
    result: MassExtractionResult,
    channel_name: str,
    channel_result: Any,
) -> hv.Overlay:
    """Build correlator data + fit curve overlay (log Y)."""
    variant_keys = channel_result.variant_keys
    curves = []

    for vk in variant_keys:
        corr = result.data.get(vk)
        if corr is None or len(corr) == 0:
            continue

        t = np.arange(len(corr), dtype=float)
        means = np.array([gvar.mean(c) for c in corr])
        errs = np.array([gvar.sdev(c) for c in corr])

        # Only plot positive values for log scale
        mask = means > 0
        if not mask.any():
            continue

        scatter = hv.Scatter(
            (t[mask], means[mask]), kdims="t", vdims="C(t)", label=vk,
        )
        ebars = hv.ErrorBars(
            (t[mask], means[mask], errs[mask]), kdims="t", vdims=["C(t)", "yerr"],
        )
        curves.append(scatter * ebars)

    if not curves:
        return _algorithm_placeholder_plot(f"No correlator data for {channel_name}")

    overlay = curves[0]
    for c in curves[1:]:
        overlay = overlay * c

    overlay = overlay.opts(
        hv.opts.Scatter(size=5, width=700, height=350),
        hv.opts.ErrorBars(line_width=1.5),
        hv.opts.Overlay(
            title=f"{channel_name} Correlator",
            legend_position="top_right",
            logy=True,
        ),
    )
    return overlay


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_mass_extraction_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
) -> MassExtractionSection:
    """Build the Mass Extraction tab with callbacks."""

    settings = MassExtractionSettings()
    status = pn.pane.Markdown(
        "**Mass Extraction:** Run Strong Correlators first, then click Extract Masses.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name="Extract Masses",
        button_type="primary",
        min_width=260,
        sizing_mode="stretch_width",
        disabled=True,
    )

    # -- Widgets --
    mass_summary_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    diagnostics_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

    channel_selector = pn.widgets.Select(
        name="Channel",
        options=[],
        sizing_mode="stretch_width",
    )
    channel_detail_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    placeholder = _algorithm_placeholder_plot("Run extraction to show plots.")
    meff_plot = pn.pane.HoloViews(placeholder, sizing_mode="stretch_width", linked_axes=False)
    correlator_fit_plot = pn.pane.HoloViews(
        placeholder, sizing_mode="stretch_width", linked_axes=False,
    )

    # -- Settings panel --
    settings_panel = pn.Param(
        settings,
        parameters=[
            "covariance_method",
            "nexp",
            "tmin",
            "tmax_frac",
            "svdcut",
            "use_log_dE",
            "use_fastfit_seeding",
            "effective_mass_method",
        ],
        show_name=False,
        default_layout=type("FitGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Refresh helpers --

    def _refresh_all():
        result: MassExtractionResult | None = state.get("mass_extraction_output")
        if result is None:
            return

        # Mass summary table
        rows = []
        for name, ch in result.channels.items():
            gs_mean = float(gvar.mean(ch.ground_state_mass))
            gs_err = float(gvar.sdev(ch.ground_state_mass))
            rows.append({
                "Channel": ch.name,
                "Type": ch.channel_type,
                "Ground Mass": f"{gs_mean:.6f}",
                "Error": f"{gs_err:.6f}",
                "chi2/dof": f"{result.diagnostics.chi2_per_dof:.3f}",
                "Q": f"{result.diagnostics.Q:.4f}",
                "Variants": ", ".join(ch.variant_keys),
            })
        mass_summary_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Diagnostics
        d = result.diagnostics
        diagnostics_pane.object = (
            f"**chi2** = {d.chi2:.2f} &nbsp;|&nbsp; "
            f"**dof** = {d.dof} &nbsp;|&nbsp; "
            f"**chi2/dof** = {d.chi2_per_dof:.3f} &nbsp;|&nbsp; "
            f"**Q** = {d.Q:.4f} &nbsp;|&nbsp; "
            f"**logGBF** = {d.logGBF:.2f} &nbsp;|&nbsp; "
            f"**nit** = {d.nit} &nbsp;|&nbsp; "
            f"**svdcut** = {d.svdcut:.1e}"
        )

        # Channel selector
        channel_names = list(result.channels.keys())
        channel_selector.options = channel_names
        if channel_names:
            channel_selector.value = channel_names[0]

        _refresh_channel_detail()

    def _refresh_channel_detail(*_args):
        result: MassExtractionResult | None = state.get("mass_extraction_output")
        if result is None:
            return

        selected = channel_selector.value
        if selected is None or selected not in result.channels:
            channel_detail_table.value = pd.DataFrame()
            meff_plot.object = _algorithm_placeholder_plot("Select a channel.")
            correlator_fit_plot.object = _algorithm_placeholder_plot("Select a channel.")
            return

        ch = result.channels[selected]

        # Energy level detail table
        rows = []
        for i, E_n in enumerate(ch.energy_levels):
            row = {
                "Level": i,
                "E_n": f"{gvar.mean(E_n):.6f}",
                "Error": f"{gvar.sdev(E_n):.6f}",
            }
            if ch.dE is not None and i < len(ch.dE):
                row["dE_n"] = f"{gvar.mean(ch.dE[i]):.6f}"
                row["dE Error"] = f"{gvar.sdev(ch.dE[i]):.6f}"
            rows.append(row)
        channel_detail_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Effective mass plot
        meff_plot.object = _build_meff_plot(result, selected, ch)

        # Correlator + fit plot
        correlator_fit_plot.object = _build_correlator_fit_plot(result, selected, ch)

    channel_selector.param.watch(_refresh_channel_detail, "value")

    # -- Compute callback --

    def on_run(_event):
        def _compute(_history: RunHistory):
            pipeline_result = state.get("strong_correlator_output")
            if pipeline_result is None:
                status.object = "**Error:** Run Strong Correlators first."
                return

            # Build config from settings
            tmax = None
            if settings.tmax_frac < 1.0:
                # Will be resolved per-channel inside the pipeline using
                # the default ChannelFitConfig.tmax = None behavior; we set
                # tmax only when fraction < 1.
                # Approximate from the first correlator length
                first_key = next(iter(pipeline_result.correlators), None)
                if first_key is not None:
                    corr = pipeline_result.correlators[first_key]
                    max_len = corr.shape[-1] if hasattr(corr, "shape") else len(corr)
                    tmax = max(settings.tmin + 1, int(max_len * settings.tmax_frac))

            config = MassExtractionConfig(
                covariance=CovarianceConfig(method=str(settings.covariance_method)),
                fit=FitConfig(svdcut=float(settings.svdcut)),
                compute_effective_mass=True,
                effective_mass_method=str(settings.effective_mass_method),
            )

            # Apply per-channel fit defaults via channel_groups auto-detection
            # We leave channel_groups empty for auto-detection and override
            # the default ChannelFitConfig values via a monkey-patch on the
            # auto-detected groups after the first call.  Simpler: just call
            # extract_masses and let it auto-detect, then the per-channel
            # configs will use defaults.  We'll set defaults on ChannelFitConfig.
            ChannelFitConfig.__init__.__defaults__  # noqa: B018 – reference only
            # Actually we cannot cleanly override the auto-detected defaults
            # without modifying config.  Instead, run extraction, then if the
            # user wants custom tmin/nexp we need groups.  Let's auto-detect
            # first, patch, then run.
            from fragile.physics.mass_extraction.pipeline import _auto_detect_channel_groups

            groups = _auto_detect_channel_groups(list(pipeline_result.correlators.keys()))
            for g in groups:
                g.fit = ChannelFitConfig(
                    tmin=int(settings.tmin),
                    tmax=tmax,
                    nexp=int(settings.nexp),
                    use_log_dE=bool(settings.use_log_dE),
                )
                g.prior = PriorConfig(use_fastfit_seeding=bool(settings.use_fastfit_seeding))
            config.channel_groups = groups

            result = extract_masses(pipeline_result, config)
            state["mass_extraction_output"] = result

            _refresh_all()

            n_ch = len(result.channels)
            n_eff = len(result.effective_masses)
            status.object = (
                f"**Complete:** {n_ch} channel groups, "
                f"{n_eff} effective-mass curves.  "
                f"chi2/dof = {result.diagnostics.chi2_per_dof:.3f}, "
                f"Q = {result.diagnostics.Q:.4f}."
            )

        run_tab_computation(state, status, "mass extraction", _compute)

    run_button.on_click(on_run)

    # -- on_history_changed --

    def on_history_changed(defer: bool) -> None:
        run_button.disabled = True
        status.object = (
            "**Mass Extraction:** Run Strong Correlators first, "
            "then click Extract Masses."
        )
        if defer:
            return
        state["mass_extraction_output"] = None
        mass_summary_table.value = pd.DataFrame()
        diagnostics_pane.object = ""
        channel_selector.options = []
        channel_detail_table.value = pd.DataFrame()
        placeholder = _algorithm_placeholder_plot("Run extraction to show plots.")
        meff_plot.object = placeholder
        correlator_fit_plot.object = placeholder

    # -- on_correlators_ready --

    def on_correlators_ready() -> None:
        run_button.disabled = False
        status.object = (
            "**Mass Extraction ready:** Strong Correlators available. "
            "Click Extract Masses."
        )

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            "**Mass Extraction:** Performs Bayesian multi-exponential fits "
            "on the strong-correlator output to extract particle masses "
            "with error bars.  Displays effective-mass plateaus, fit "
            "quality diagnostics, and per-channel energy levels."
        ),
        alert_type="info",
        sizing_mode="stretch_width",
    )

    tab = pn.Column(
        status,
        info_note,
        pn.Row(run_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Fit Settings", settings_panel),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Summary"),
        mass_summary_table,
        pn.pane.Markdown("### Fit Diagnostics"),
        diagnostics_pane,
        pn.layout.Divider(),
        pn.pane.Markdown("### Channel Detail"),
        channel_selector,
        channel_detail_table,
        meff_plot,
        correlator_fit_plot,
        sizing_mode="stretch_both",
    )

    return MassExtractionSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
        on_correlators_ready=on_correlators_ready,
    )
