"""Electroweak Correlators dashboard tab.

Provides a UI to drive the electroweak companion-channel pipeline:
select channels, configure operator parameters, run the pipeline,
and visualize correlator curves, effective masses, and operator time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import panel as pn
import param

from fragile.physics.app.correlator_plots import (
    build_correlator_table,
    build_grouped_correlator_plot,
    build_grouped_meff_plot,
    build_summary_table,
    group_electroweak_correlator_keys,
)
from fragile.physics.electroweak.electroweak_channels import (
    compute_electroweak_channels,
    ElectroweakChannelConfig,
    ElectroweakChannelOutput,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.operators import PipelineResult


# ---------------------------------------------------------------------------
# Channel family constants
# ---------------------------------------------------------------------------

EW_U1_CHANNELS: tuple[str, ...] = (
    "u1_phase",
    "u1_dressed",
    "u1_phase_q2",
    "u1_dressed_q2",
)
EW_SU2_BASE_CHANNELS: tuple[str, ...] = (
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
)
EW_SU2_DIRECTED_CHANNELS: tuple[str, ...] = (
    "su2_phase_directed",
    "su2_component_directed",
    "su2_doublet_directed",
    "su2_doublet_diff_directed",
)
EW_SU2_WALKER_TYPE_CHANNELS: tuple[str, ...] = (
    "su2_phase_cloner",
    "su2_component_cloner",
    "su2_doublet_cloner",
    "su2_doublet_diff_cloner",
    "su2_phase_resister",
    "su2_component_resister",
    "su2_doublet_resister",
    "su2_doublet_diff_resister",
    "su2_phase_persister",
    "su2_component_persister",
    "su2_doublet_persister",
    "su2_doublet_diff_persister",
)
EW_MIXED_CHANNELS: tuple[str, ...] = ("ew_mixed",)
EW_SYMMETRY_BREAKING_CHANNELS: tuple[str, ...] = (
    "fitness_phase",
    "clone_indicator",
)
EW_PARITY_VELOCITY_CHANNELS: tuple[str, ...] = (
    "velocity_norm_cloner",
    "velocity_norm_resister",
    "velocity_norm_persister",
)


# ---------------------------------------------------------------------------
# Output adapter
# ---------------------------------------------------------------------------


def _electroweak_output_to_pipeline_result(ew_output: ElectroweakChannelOutput) -> PipelineResult:
    """Convert ``ElectroweakChannelOutput`` to ``PipelineResult`` for downstream consumers."""
    return PipelineResult(
        operators={name: cr.series for name, cr in ew_output.channel_results.items()},
        correlators={name: cr.correlator for name, cr in ew_output.channel_results.items()},
        prepared_data=None,
        scales=None,
    )


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class ElectroweakCorrelatorSection:
    """Container for the electroweak correlators dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class ElectroweakCorrelatorSettings(param.Parameterized):
    """Settings for the electroweak correlator pipeline."""

    # -- Common --
    warmup_fraction = param.Number(default=0.2, bounds=(0.0, 0.95))
    end_fraction = param.Number(default=1.0, bounds=(0.05, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))

    # -- Correlator --
    max_lag = param.Integer(default=40, bounds=(1, 500))
    use_connected = param.Boolean(default=True)

    # -- Electroweak-specific --
    epsilon_d = param.Number(
        default=None,
        bounds=(1e-12, None),
        allow_None=True,
        doc="Distance epsilon (None = auto-resolve).",
    )
    epsilon_clone = param.Number(
        default=None,
        bounds=(1e-12, None),
        allow_None=True,
        doc="Clone epsilon (None = auto-resolve).",
    )
    lambda_alg = param.Number(default=0.0, bounds=(0.0, None))
    su2_operator_mode = param.ObjectSelector(
        default="standard",
        objects=("standard", "score_directed"),
    )

    # -- Family toggles --
    enable_directed_variants = param.Boolean(default=True)
    enable_walker_type_split = param.Boolean(default=False)
    enable_parity_velocity = param.Boolean(default=True)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_electroweak_correlator_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
) -> ElectroweakCorrelatorSection:
    """Build the Electroweak Correlators tab with callbacks."""

    settings = ElectroweakCorrelatorSettings()
    status = pn.pane.Markdown(
        "**Electroweak Correlators:** Load a RunHistory and click Compute.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name="Compute Electroweak Correlators",
        button_type="primary",
        min_width=260,
        sizing_mode="stretch_width",
        disabled=True,
    )

    # -- Visualization widgets --
    log_scale_toggle = pn.widgets.Toggle(
        name="Log scale Y",
        value=True,
        button_type="default",
        width=120,
    )

    # -- Per-channel-group container --
    per_channel_container = pn.Column(sizing_mode="stretch_width")

    # -- Tables --
    summary_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    correlator_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=10,
        show_index=False,
        sizing_mode="stretch_width",
    )

    # -- Settings panels --
    common_settings_panel = pn.Param(
        settings,
        parameters=[
            "warmup_fraction",
            "end_fraction",
            "h_eff",
            "max_lag",
            "use_connected",
        ],
        show_name=False,
        default_layout=type("CommonGrid", (pn.GridBox,), {"ncols": 2}),
    )

    electroweak_settings_panel = pn.Param(
        settings,
        parameters=[
            "epsilon_d",
            "epsilon_clone",
            "lambda_alg",
            "su2_operator_mode",
            "enable_directed_variants",
            "enable_walker_type_split",
            "enable_parity_velocity",
        ],
        show_name=False,
        default_layout=type("EWGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Channel selection (MultiSelect per family) --
    u1_selector = pn.widgets.MultiSelect(
        name="U(1)",
        options=list(EW_U1_CHANNELS),
        value=list(EW_U1_CHANNELS),
        size=len(EW_U1_CHANNELS),
        sizing_mode="stretch_width",
    )
    su2_base_selector = pn.widgets.MultiSelect(
        name="SU(2) Base",
        options=list(EW_SU2_BASE_CHANNELS),
        value=list(EW_SU2_BASE_CHANNELS),
        size=len(EW_SU2_BASE_CHANNELS),
        sizing_mode="stretch_width",
    )
    su2_directed_selector = pn.widgets.MultiSelect(
        name="SU(2) Directed",
        options=list(EW_SU2_DIRECTED_CHANNELS),
        value=list(EW_SU2_DIRECTED_CHANNELS),
        size=len(EW_SU2_DIRECTED_CHANNELS),
        sizing_mode="stretch_width",
    )
    su2_walker_type_selector = pn.widgets.MultiSelect(
        name="SU(2) Walker-Type",
        options=list(EW_SU2_WALKER_TYPE_CHANNELS),
        value=[],
        size=min(8, len(EW_SU2_WALKER_TYPE_CHANNELS)),
        sizing_mode="stretch_width",
    )
    mixed_selector = pn.widgets.MultiSelect(
        name="EW Mixed",
        options=list(EW_MIXED_CHANNELS),
        value=list(EW_MIXED_CHANNELS),
        size=len(EW_MIXED_CHANNELS),
        sizing_mode="stretch_width",
    )
    symmetry_breaking_selector = pn.widgets.MultiSelect(
        name="Symmetry Breaking",
        options=list(EW_SYMMETRY_BREAKING_CHANNELS),
        value=list(EW_SYMMETRY_BREAKING_CHANNELS),
        size=len(EW_SYMMETRY_BREAKING_CHANNELS),
        sizing_mode="stretch_width",
    )
    parity_velocity_selector = pn.widgets.MultiSelect(
        name="Parity Velocity",
        options=list(EW_PARITY_VELOCITY_CHANNELS),
        value=list(EW_PARITY_VELOCITY_CHANNELS),
        size=len(EW_PARITY_VELOCITY_CHANNELS),
        sizing_mode="stretch_width",
    )

    # Gate visibility of directed/walker-type/parity selectors by their toggles
    def _on_directed_toggle(event):
        su2_directed_selector.visible = event.new

    def _on_walker_type_toggle(event):
        su2_walker_type_selector.visible = event.new

    def _on_parity_toggle(event):
        parity_velocity_selector.visible = event.new

    settings.param.watch(_on_directed_toggle, "enable_directed_variants")
    settings.param.watch(_on_walker_type_toggle, "enable_walker_type_split")
    settings.param.watch(_on_parity_toggle, "enable_parity_velocity")

    # Initial visibility
    su2_directed_selector.visible = settings.enable_directed_variants
    su2_walker_type_selector.visible = settings.enable_walker_type_split
    parity_velocity_selector.visible = settings.enable_parity_velocity

    # -- Refresh plots helper --

    def _refresh_overlay():
        """Refresh summary tables and per-channel-group plots."""
        result: PipelineResult | None = state.get("electroweak_correlator_output")
        if result is None:
            return
        ls = bool(log_scale_toggle.value)
        summary_table.value = build_summary_table(result)
        correlator_table.value = build_correlator_table(result)

        # Per-channel group plots
        groups = group_electroweak_correlator_keys(result.correlators.keys())
        per_channel_container.clear()
        for group_name, keys in groups.items():
            corr_overlay = build_grouped_correlator_plot(
                result,
                group_name,
                keys,
                0,
                ls,
            )
            meff_overlay = build_grouped_meff_plot(
                result,
                group_name,
                keys,
                0,
            )
            per_channel_container.append(pn.pane.Markdown(f"#### {group_name}"))
            per_channel_container.append(
                pn.Row(
                    pn.pane.HoloViews(
                        corr_overlay, sizing_mode="stretch_width", linked_axes=False
                    ),
                    pn.pane.HoloViews(
                        meff_overlay, sizing_mode="stretch_width", linked_axes=False
                    ),
                    sizing_mode="stretch_width",
                )
            )

    log_scale_toggle.param.watch(lambda _: _refresh_overlay(), "value")

    # -- Compute callback --

    def on_run(_):
        def _compute(history: RunHistory):
            # Collect user-selected channels from the MultiSelect widgets
            selected_channels: list[str] = []
            selected_channels.extend(u1_selector.value)
            selected_channels.extend(su2_base_selector.value)
            if settings.enable_directed_variants:
                selected_channels.extend(su2_directed_selector.value)
            if settings.enable_walker_type_split:
                selected_channels.extend(su2_walker_type_selector.value)
            selected_channels.extend(mixed_selector.value)
            selected_channels.extend(symmetry_breaking_selector.value)
            if settings.enable_parity_velocity:
                selected_channels.extend(parity_velocity_selector.value)

            if not selected_channels:
                status.object = "**Error:** No channels selected."
                return

            cfg = ElectroweakChannelConfig(
                warmup_fraction=float(settings.warmup_fraction),
                end_fraction=float(settings.end_fraction),
                h_eff=float(settings.h_eff),
                max_lag=int(settings.max_lag),
                use_connected=bool(settings.use_connected),
                epsilon_d=(float(settings.epsilon_d) if settings.epsilon_d is not None else None),
                epsilon_clone=(
                    float(settings.epsilon_clone) if settings.epsilon_clone is not None else None
                ),
                lambda_alg=float(settings.lambda_alg),
                su2_operator_mode=str(settings.su2_operator_mode),
                enable_walker_type_split=bool(settings.enable_walker_type_split),
            )

            ew_output = compute_electroweak_channels(
                history, channels=selected_channels, config=cfg
            )
            result = _electroweak_output_to_pipeline_result(ew_output)
            state["electroweak_correlator_output"] = result

            _refresh_overlay()

            n_channels = len(result.correlators)
            status.object = (
                f"**Complete:** {n_channels} correlators " f"({ew_output.n_valid_frames} frames)."
            )

        run_tab_computation(
            state,
            status,
            "electroweak correlators",
            _compute,
        )

    run_button.on_click(on_run)

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            "**Electroweak Correlators:** Select electroweak channel families below. "
            "Toggle directed variants, walker-type splits, and parity velocity "
            "channels using the settings panel. Uses the electroweak operator "
            "pipeline (U(1) hypercharge + SU(2) isospin)."
        ),
        alert_type="info",
        sizing_mode="stretch_width",
    )

    channel_selection_panel = pn.Column(
        pn.Row(
            u1_selector,
            su2_base_selector,
            mixed_selector,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            su2_directed_selector,
            su2_walker_type_selector,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            symmetry_breaking_selector,
            parity_velocity_selector,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    tab = pn.Column(
        status,
        info_note,
        pn.Row(run_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Common Settings", common_settings_panel),
            ("Channel Selection", channel_selection_panel),
            ("Electroweak Settings", electroweak_settings_panel),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Correlator Summary"),
        summary_table,
        pn.Row(log_scale_toggle, sizing_mode="stretch_width"),
        pn.pane.Markdown("### Per-Channel Correlators"),
        per_channel_container,
        pn.layout.Divider(),
        pn.pane.Markdown("### Full Correlator Table"),
        correlator_table,
        sizing_mode="stretch_both",
    )

    # -- on_history_changed --

    def on_history_changed(defer: bool) -> None:
        run_button.disabled = False
        status.object = "**Electroweak Correlators ready:** click Compute Electroweak Correlators."
        if defer:
            return
        state["electroweak_correlator_output"] = None
        summary_table.value = pd.DataFrame()
        correlator_table.value = pd.DataFrame()
        per_channel_container.clear()

    return ElectroweakCorrelatorSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
    )
