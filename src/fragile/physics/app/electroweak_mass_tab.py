"""Electroweak Mass Extraction dashboard tab.

Feeds the electroweak-correlator PipelineResult into the Bayesian
multi-exponential fitter, displays extracted masses, fit diagnostics,
effective-mass plots, and correlator-vs-fit overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gvar
import pandas as pd
import panel as pn
import param

from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.app.mass_extraction_tab import (
    _build_amplitude_table,
    _build_correlator_fit_plot,
    _build_cross_channel_ratio_table,
    _build_diagnostics_container,
    _build_intra_channel_ratio_table,
    _build_mass_spectrum_bar,
    _build_meff_plot,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.mass_extraction import (
    ChannelFitConfig,
    CovarianceConfig,
    extract_masses,
    FitConfig,
    MassExtractionConfig,
    MassExtractionResult,
    PriorConfig,
)


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class ElectroweakMassSection:
    """Container for the electroweak mass extraction dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]
    on_correlators_ready: Callable[[], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class ElectroweakMassSettings(param.Parameterized):
    """Settings for the electroweak mass extraction pipeline."""

    covariance_method = param.ObjectSelector(
        default="uncorrelated",
        objects=["uncorrelated", "block_jackknife", "bootstrap"],
    )
    nexp = param.Integer(default=2, bounds=(1, 6))
    tmin = param.Integer(default=2, bounds=(1, 20))
    tmax = param.Integer(
        default=0,
        bounds=(0, 500),
        doc="Global tmax (0 = full range).",
    )
    svdcut = param.Number(default=1e-4)
    use_log_dE = param.Boolean(default=True)
    use_fastfit_seeding = param.Boolean(default=True)
    effective_mass_method = param.ObjectSelector(
        default="log_ratio",
        objects=["log_ratio", "cosh"],
    )
    include_multiscale = param.Boolean(default=True)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_electroweak_mass_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
) -> ElectroweakMassSection:
    """Build the Electroweak Mass Extraction tab with callbacks."""

    settings = ElectroweakMassSettings()
    status = pn.pane.Markdown(
        "**Electroweak Mass:** Run Electroweak Correlators first, "
        "then click Extract Electroweak Masses.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name="Extract Electroweak Masses",
        button_type="primary",
        min_width=260,
        sizing_mode="stretch_width",
        disabled=True,
    )

    # -- Widgets --
    mass_spectrum_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run extraction to show mass spectrum."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    mass_summary_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    diagnostics_container = pn.Column(
        pn.pane.Markdown("", sizing_mode="stretch_width"),
        pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    channel_details_container = pn.Column(sizing_mode="stretch_width")
    cross_ratio_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    # -- Channel key selection widgets --
    channel_key_selectors: dict[str, pn.widgets.MultiSelect] = {}

    channel_key_selection_container = pn.Column(
        pn.pane.Markdown(
            "*Run Electroweak Correlators to populate channel operator selections.*",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    # -- Settings panel --
    settings_panel = pn.Param(
        settings,
        parameters=[
            "covariance_method",
            "nexp",
            "tmin",
            "tmax",
            "svdcut",
            "use_log_dE",
            "use_fastfit_seeding",
            "effective_mass_method",
            "include_multiscale",
        ],
        show_name=False,
        default_layout=type("FitGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Refresh helpers --

    def _refresh_all():
        result: MassExtractionResult | None = state.get("electroweak_mass_output")
        if result is None:
            return

        # Mass spectrum bar chart
        mass_spectrum_plot.object = _build_mass_spectrum_bar(result)

        # Mass summary table
        rows = []
        for ch in result.channels.values():
            gs_mean = float(gvar.mean(ch.ground_state_mass))
            gs_err = float(gvar.sdev(ch.ground_state_mass))
            err_pct = (gs_err / gs_mean * 100) if gs_mean != 0 else float("inf")
            rows.append({
                "Channel": ch.name,
                "Type": ch.channel_type,
                "Ground Mass": f"{gs_mean:.6f}",
                "Error": f"{gs_err:.6f}",
                "Error %": f"{err_pct:.2f}%",
                "chi2/dof": f"{result.diagnostics.chi2_per_dof:.3f}",
                "Q": f"{result.diagnostics.Q:.4f}",
                "Variants": ", ".join(ch.variant_keys),
            })
        mass_summary_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Diagnostics container
        diag_parts = _build_diagnostics_container(result)
        diagnostics_container[0].object = diag_parts[0]  # global markdown
        diagnostics_container[1].value = diag_parts[1]  # per-channel table

        _refresh_all_channel_details()
        cross_ratio_table.value = _build_cross_channel_ratio_table(result)

    def _refresh_all_channel_details():
        """Build per-channel detail sections for all channels at once."""
        result: MassExtractionResult | None = state.get("electroweak_mass_output")
        channel_details_container.clear()
        if result is None or not result.channels:
            return

        for ch_name, ch in result.channels.items():
            # Energy level detail table
            rows = []
            for i, E_n in enumerate(ch.energy_levels):
                e_mean = gvar.mean(E_n)
                e_err = gvar.sdev(E_n)
                e_err_pct = abs(e_err / e_mean) * 100.0 if e_mean != 0 else float("inf")
                row = {
                    "Level": i,
                    "E_n": f"{e_mean:.6f}",
                    "Error": f"{e_err:.6f}",
                    "Error (%)": f"{e_err_pct:.2f}",
                }
                if ch.dE is not None and i < len(ch.dE):
                    de_mean = gvar.mean(ch.dE[i])
                    de_err = gvar.sdev(ch.dE[i])
                    de_err_pct = abs(de_err / de_mean) * 100.0 if de_mean != 0 else float("inf")
                    row["dE_n"] = f"{de_mean:.6f}"
                    row["dE Error"] = f"{de_err:.6f}"
                    row["dE Error (%)"] = f"{de_err_pct:.2f}"
                rows.append(row)
            detail_df = pd.DataFrame(rows) if rows else pd.DataFrame()
            amp_df = _build_amplitude_table(ch)
            intra_df = _build_intra_channel_ratio_table(ch)

            meff = _build_meff_plot(result, ch_name, ch)
            corr_fit = _build_correlator_fit_plot(result, ch_name, ch)

            channel_details_container.append(
                pn.Column(
                    pn.pane.Markdown(f"#### {ch_name}"),
                    pn.Row(
                        pn.pane.HoloViews(
                            corr_fit, sizing_mode="stretch_width", linked_axes=False
                        ),
                        pn.pane.HoloViews(meff, sizing_mode="stretch_width", linked_axes=False),
                        sizing_mode="stretch_width",
                    ),
                    pn.Row(
                        pn.Column(
                            pn.pane.Markdown("**Energy Levels**"),
                            pn.widgets.Tabulator(
                                detail_df,
                                pagination=None,
                                show_index=False,
                                sizing_mode="stretch_width",
                            ),
                            sizing_mode="stretch_width",
                        ),
                        pn.Column(
                            pn.pane.Markdown("**Amplitudes**"),
                            pn.widgets.Tabulator(
                                amp_df,
                                pagination=None,
                                show_index=False,
                                sizing_mode="stretch_width",
                            ),
                            sizing_mode="stretch_width",
                        ),
                        pn.Column(
                            pn.pane.Markdown("**Energy Level Ratios**"),
                            pn.widgets.Tabulator(
                                intra_df,
                                pagination=None,
                                show_index=False,
                                sizing_mode="stretch_width",
                            ),
                            sizing_mode="stretch_width",
                        ),
                        sizing_mode="stretch_width",
                    ),
                    pn.layout.Divider(),
                    sizing_mode="stretch_width",
                )
            )

    # -- Compute callback --

    def on_run(_event):
        def _compute(_history: RunHistory):
            pipeline_result = state.get("electroweak_correlator_output")
            if pipeline_result is None:
                status.object = "**Error:** Run Electroweak Correlators first."
                return

            # Build config from settings
            tmax_val = int(settings.tmax)
            tmax = None if tmax_val <= 0 else max(int(settings.tmin) + 1, tmax_val)

            config = MassExtractionConfig(
                covariance=CovarianceConfig(method=str(settings.covariance_method)),
                fit=FitConfig(svdcut=float(settings.svdcut)),
                compute_effective_mass=True,
                effective_mass_method=str(settings.effective_mass_method),
                include_multiscale=bool(settings.include_multiscale),
            )

            from fragile.physics.mass_extraction.pipeline import _auto_detect_channel_groups

            groups = _auto_detect_channel_groups(
                list(pipeline_result.correlators.keys()),
                include_multiscale=bool(settings.include_multiscale),
            )

            # Filter correlator keys per channel using selector widgets
            for g in groups:
                selector = channel_key_selectors.get(g.name)
                if selector is not None:
                    selected = set(selector.value)
                    g.correlator_keys = [k for k in g.correlator_keys if k in selected]
            groups = [g for g in groups if g.correlator_keys]

            if not groups:
                status.object = (
                    "**Error:** No correlator keys selected. "
                    "Open *Channel Operator Selection* and select at least "
                    "one key per channel."
                )
                return

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
            state["electroweak_mass_output"] = result

            _refresh_all()

            n_ch = len(result.channels)
            n_eff = len(result.effective_masses)
            status.object = (
                f"**Complete:** {n_ch} channel groups, "
                f"{n_eff} effective-mass curves.  "
                f"chi2/dof = {result.diagnostics.chi2_per_dof:.3f}, "
                f"Q = {result.diagnostics.Q:.4f}."
            )

        run_tab_computation(state, status, "electroweak mass extraction", _compute)

    run_button.on_click(on_run)

    # -- on_history_changed --

    def on_history_changed(defer: bool) -> None:
        run_button.disabled = True
        status.object = (
            "**Electroweak Mass:** Run Electroweak Correlators first, "
            "then click Extract Electroweak Masses."
        )
        # Always reset channel key selectors when history changes
        channel_key_selectors.clear()
        channel_key_selection_container.clear()
        channel_key_selection_container.append(
            pn.pane.Markdown(
                "*Run Electroweak Correlators to populate channel operator selections.*",
                sizing_mode="stretch_width",
            ),
        )
        if defer:
            return
        state["electroweak_mass_output"] = None
        mass_spectrum_plot.object = _algorithm_placeholder_plot(
            "Run extraction to show mass spectrum.",
        )
        mass_summary_table.value = pd.DataFrame()
        diagnostics_container[0].object = ""
        diagnostics_container[1].value = pd.DataFrame()
        channel_details_container.clear()
        cross_ratio_table.value = pd.DataFrame()

    # -- on_correlators_ready --

    def on_correlators_ready() -> None:
        run_button.disabled = False
        status.object = (
            "**Electroweak Mass ready:** Electroweak Correlators available. "
            "Click Extract Electroweak Masses."
        )

        # Populate channel key selection widgets from pipeline result
        pipeline_result = state.get("electroweak_correlator_output")
        if pipeline_result is not None:
            from fragile.physics.mass_extraction.pipeline import (
                _auto_detect_channel_groups,
            )

            groups = _auto_detect_channel_groups(
                list(pipeline_result.correlators.keys()),
                include_multiscale=True,
            )
            channel_key_selectors.clear()
            widgets: list[pn.widgets.MultiSelect] = []
            for g in groups:
                keys = sorted(g.correlator_keys)
                if not keys:
                    continue
                selector = pn.widgets.MultiSelect(
                    name=g.name,
                    options=keys,
                    value=keys,
                    size=min(len(keys), 8),
                    sizing_mode="stretch_width",
                )
                channel_key_selectors[g.name] = selector
                widgets.append(selector)

            channel_key_selection_container.clear()
            if widgets:
                # Lay out in rows of 3
                for i in range(0, len(widgets), 3):
                    channel_key_selection_container.append(
                        pn.Row(*widgets[i : i + 3], sizing_mode="stretch_width"),
                    )
            else:
                channel_key_selection_container.append(
                    pn.pane.Markdown(
                        "*No channel groups detected.*",
                        sizing_mode="stretch_width",
                    ),
                )

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            "**Electroweak Mass Extraction:** Performs Bayesian multi-exponential fits "
            "on the electroweak-correlator output to extract electroweak masses "
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
            ("Channel Operator Selection", channel_key_selection_container),
            active=[0, 1],
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Spectrum"),
        mass_spectrum_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Summary"),
        mass_summary_table,
        pn.pane.Markdown("### Cross-Channel Mass Ratios"),
        cross_ratio_table,
        pn.pane.Markdown("### Fit Diagnostics"),
        diagnostics_container,
        pn.layout.Divider(),
        pn.pane.Markdown("### Channel Details"),
        channel_details_container,
        sizing_mode="stretch_both",
    )

    return ElectroweakMassSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
        on_correlators_ready=on_correlators_ready,
    )
