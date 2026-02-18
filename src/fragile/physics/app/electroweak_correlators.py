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

from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.app.correlator_plots import (
    build_all_channels_correlator_overlay,
    build_all_channels_meff_overlay,
    build_correlator_table,
    build_grouped_correlator_plot,
    build_grouped_meff_plot,
    build_single_correlator_plot,
    build_single_effective_mass_plot,
    build_single_operator_series_plot,
    build_summary_table,
    get_correlator_array,
    get_operator_series_array,
    group_electroweak_correlator_keys,
)
from fragile.physics.app.strong_correlators import _parse_color_dims
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.operators import (
    ChannelConfigBase,
    compute_strong_force_pipeline,
    CorrelatorConfig,
    ElectroweakOperatorConfig,
    MultiscaleConfig,
    PipelineConfig,
    PipelineResult,
)


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

    # -- Common (ChannelConfigBase) --
    warmup_fraction = param.Number(default=0.2, bounds=(0.0, 0.95))
    end_fraction = param.Number(default=1.0, bounds=(0.05, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    color_dims_spec = param.String(
        default="",
        doc="Comma-separated color dims (blank = all available).",
    )
    eps = param.Number(default=1e-12, bounds=(0.0, None))
    pair_selection = param.ObjectSelector(
        default="both",
        objects=("both", "distance", "clone"),
    )

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

    # -- Multiscale --
    n_scales = param.Integer(
        default=1,
        bounds=(1, 32),
        doc="1 = single-scale (no multiscale).",
    )
    kernel_type = param.ObjectSelector(
        default="gaussian",
        objects=("gaussian", "exponential", "tophat", "shell"),
    )
    distance_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "floyd-warshall", "tropical"),
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
    scale_q_low = param.Number(default=0.05, bounds=(0.0, 0.99))
    scale_q_high = param.Number(default=0.95, bounds=(0.01, 1.0))
    max_scale_samples = param.Integer(default=500_000, bounds=(1_000, 5_000_000))
    min_scale = param.Number(default=1e-6, bounds=(1e-12, None))


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
    scale_selector = pn.widgets.IntSlider(
        name="Scale index",
        value=0,
        start=0,
        end=0,
        visible=False,
        sizing_mode="stretch_width",
    )
    ALL_CHANNELS = "(all channels)"
    channel_selector = pn.widgets.Select(
        name="Channel",
        options=[ALL_CHANNELS],
        value=ALL_CHANNELS,
        sizing_mode="stretch_width",
    )

    # -- Plot panes --
    overlay_correlator_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    overlay_meff_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    single_correlator_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    single_meff_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    single_operator_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)

    placeholder = _algorithm_placeholder_plot("Run pipeline to show plots.")
    overlay_correlator_plot.object = placeholder
    overlay_meff_plot.object = placeholder
    single_correlator_plot.object = placeholder
    single_meff_plot.object = placeholder
    single_operator_plot.object = placeholder

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
            "mass",
            "ell0",
            "eps",
            "pair_selection",
            "color_dims_spec",
            "max_lag",
            "use_connected",
        ],
        show_name=False,
        widgets={"color_dims_spec": {"name": "Color dims (optional)"}},
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

    multiscale_settings_panel = pn.Param(
        settings,
        parameters=[
            "n_scales",
            "kernel_type",
            "distance_method",
            "edge_weight_mode",
            "scale_q_low",
            "scale_q_high",
            "max_scale_samples",
            "min_scale",
        ],
        show_name=False,
        default_layout=type("MultiscaleGrid", (pn.GridBox,), {"ncols": 2}),
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
        result: PipelineResult | None = state.get("electroweak_correlator_output")
        if result is None:
            return
        si = int(scale_selector.value)
        ls = bool(log_scale_toggle.value)
        overlay_correlator_plot.object = build_all_channels_correlator_overlay(
            result,
            logy=ls,
            scale_index=si,
        )
        overlay_meff_plot.object = build_all_channels_meff_overlay(
            result,
            scale_index=si,
        )
        summary_table.value = build_summary_table(result, scale_index=si)
        correlator_table.value = build_correlator_table(result, scale_index=si)

        # Per-channel group plots
        groups = group_electroweak_correlator_keys(result.correlators.keys())
        per_channel_container.clear()
        for group_name, keys in groups.items():
            corr_overlay = build_grouped_correlator_plot(
                result,
                group_name,
                keys,
                si,
                ls,
            )
            meff_overlay = build_grouped_meff_plot(
                result,
                group_name,
                keys,
                si,
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

    def _refresh_single():
        result: PipelineResult | None = state.get("electroweak_correlator_output")
        if result is None:
            return
        selected = str(channel_selector.value)
        si = int(scale_selector.value)
        ls = bool(log_scale_toggle.value)

        if selected == ALL_CHANNELS or selected not in result.correlators:
            single_correlator_plot.object = _algorithm_placeholder_plot(
                "Select a channel above to see its individual plots."
            )
            single_meff_plot.object = _algorithm_placeholder_plot("Select a channel above.")
            single_operator_plot.object = _algorithm_placeholder_plot("Select a channel above.")
            return

        corr_arr = get_correlator_array(result, selected, si)
        op_arr = get_operator_series_array(result, selected, si)

        if len(corr_arr) > 0:
            single_correlator_plot.object = build_single_correlator_plot(
                corr_arr,
                selected,
                logy=ls,
            )
            single_meff_plot.object = build_single_effective_mass_plot(
                corr_arr,
                selected,
            )
        else:
            single_correlator_plot.object = _algorithm_placeholder_plot(
                f"No correlator data for {selected}"
            )
            single_meff_plot.object = _algorithm_placeholder_plot(f"No m_eff data for {selected}")

        if len(op_arr) > 0:
            single_operator_plot.object = build_single_operator_series_plot(
                op_arr,
                selected,
            )
        else:
            single_operator_plot.object = _algorithm_placeholder_plot(
                f"No operator data for {selected}"
            )

    def _refresh_all():
        _refresh_overlay()
        _refresh_single()

    log_scale_toggle.param.watch(lambda _: _refresh_all(), "value")
    scale_selector.param.watch(lambda _: _refresh_all(), "value")
    channel_selector.param.watch(lambda _: _refresh_single(), "value")

    # -- Compute callback --

    def on_run(_):
        def _compute(history: RunHistory):
            d = history.d
            color_dims = _parse_color_dims(settings.color_dims_spec, d)

            base_kwargs: dict[str, Any] = {
                "warmup_fraction": float(settings.warmup_fraction),
                "end_fraction": float(settings.end_fraction),
                "h_eff": float(settings.h_eff),
                "mass": float(settings.mass),
                "ell0": float(settings.ell0) if settings.ell0 is not None else None,
                "color_dims": color_dims,
                "eps": float(settings.eps),
                "pair_selection": str(settings.pair_selection),
            }

            ew_config = ElectroweakOperatorConfig(
                epsilon_d=(float(settings.epsilon_d) if settings.epsilon_d is not None else None),
                epsilon_clone=(
                    float(settings.epsilon_clone) if settings.epsilon_clone is not None else None
                ),
                lambda_alg=float(settings.lambda_alg),
                su2_operator_mode=str(settings.su2_operator_mode),
                enable_directed_variants=bool(settings.enable_directed_variants),
                enable_walker_type_split=bool(settings.enable_walker_type_split),
                enable_parity_velocity=bool(settings.enable_parity_velocity),
                **base_kwargs,
            )

            correlator_cfg = CorrelatorConfig(
                max_lag=int(settings.max_lag),
                use_connected=bool(settings.use_connected),
            )
            multiscale_cfg = MultiscaleConfig(
                n_scales=int(settings.n_scales),
                kernel_type=str(settings.kernel_type),
                distance_method=str(settings.distance_method),
                edge_weight_mode=str(settings.edge_weight_mode),
                scale_q_low=float(settings.scale_q_low),
                scale_q_high=float(settings.scale_q_high),
                max_scale_samples=int(settings.max_scale_samples),
                min_scale=float(settings.min_scale),
            )

            pipeline_config = PipelineConfig(
                base=ChannelConfigBase(**base_kwargs),
                electroweak=ew_config,
                correlator=correlator_cfg,
                multiscale=multiscale_cfg,
                channels=["electroweak"],
            )
            result = compute_strong_force_pipeline(history, pipeline_config)

            # Collect all user-selected channels from the MultiSelect widgets
            selected_channels: set[str] = set()
            selected_channels.update(u1_selector.value)
            selected_channels.update(su2_base_selector.value)
            if settings.enable_directed_variants:
                selected_channels.update(su2_directed_selector.value)
            if settings.enable_walker_type_split:
                selected_channels.update(su2_walker_type_selector.value)
            selected_channels.update(mixed_selector.value)
            selected_channels.update(symmetry_breaking_selector.value)
            if settings.enable_parity_velocity:
                selected_channels.update(parity_velocity_selector.value)

            if not selected_channels:
                status.object = "**Error:** No channels selected."
                return

            # Filter result to only user-selected channels
            result.correlators = {
                k: v for k, v in result.correlators.items() if k in selected_channels
            }
            result.operators = {
                k: v for k, v in result.operators.items() if k in selected_channels
            }

            state["electroweak_correlator_output"] = result

            # Update scale selector
            if result.scales is not None and result.scales.numel() > 1:
                scale_selector.end = int(result.scales.numel()) - 1
                scale_selector.value = 0
                scale_selector.visible = True
            else:
                scale_selector.end = 0
                scale_selector.value = 0
                scale_selector.visible = False

            # Update channel selector
            channel_names = [ALL_CHANNELS, *list(result.correlators.keys())]
            channel_selector.options = channel_names
            channel_selector.value = ALL_CHANNELS

            _refresh_all()

            n_channels = len(result.correlators)
            n_ops = len(result.operators)
            n_frames = 0
            data = result.prepared_data
            if data is not None and data.frame_indices:
                n_frames = len(data.frame_indices)
            ms_info = ""
            if result.scales is not None and result.scales.numel() > 1:
                ms_info = f" | {int(result.scales.numel())} scales"
            status.object = (
                f"**Complete:** {n_channels} correlators from {n_ops} operators "
                f"({n_frames} frames){ms_info}."
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
            ("Multiscale Settings", multiscale_settings_panel),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Correlator Summary"),
        summary_table,
        pn.pane.Markdown("### All Channels Overlay"),
        pn.Row(log_scale_toggle, scale_selector, sizing_mode="stretch_width"),
        overlay_correlator_plot,
        overlay_meff_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Per-Channel Correlators"),
        per_channel_container,
        pn.layout.Divider(),
        pn.pane.Markdown("### Single Channel View"),
        channel_selector,
        single_correlator_plot,
        single_meff_plot,
        single_operator_plot,
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
        channel_selector.options = [ALL_CHANNELS]
        channel_selector.value = ALL_CHANNELS
        ph = _algorithm_placeholder_plot("Run pipeline to show plots.")
        overlay_correlator_plot.object = ph
        overlay_meff_plot.object = ph
        single_correlator_plot.object = ph
        single_meff_plot.object = ph
        single_operator_plot.object = ph
        per_channel_container.clear()
        scale_selector.visible = False

    return ElectroweakCorrelatorSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
    )
