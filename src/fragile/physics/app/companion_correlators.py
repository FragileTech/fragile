"""Companion Correlators dashboard tab.

Provides a UI to drive the ``new_channels`` companion-based compute functions,
collect results via per-type extractors, and visualize correlator
curves, effective masses, and operator time series using the shared
:mod:`correlator_plots` utilities.
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
    group_strong_correlator_keys,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.new_channels.baryon_triplet_channels import (
    BaryonTripletCorrelatorConfig,
    compute_companion_baryon_correlator,
)
from fragile.physics.new_channels.glueball_color_channels import (
    GlueballColorCorrelatorConfig,
    compute_companion_glueball_color_correlator,
)
from fragile.physics.new_channels.mass_extraction_adapter import (
    extract_baryon,
    extract_glueball,
    extract_meson_phase,
    extract_tensor_momentum,
    extract_vector_meson,
)
from fragile.physics.new_channels.meson_phase_channels import (
    MesonPhaseCorrelatorConfig,
    compute_companion_meson_phase_correlator,
)
from fragile.physics.new_channels.tensor_momentum_channels import (
    TensorMomentumCorrelatorConfig,
    compute_companion_tensor_momentum_correlator,
)
from fragile.physics.new_channels.vector_meson_channels import (
    VectorMesonCorrelatorConfig,
    compute_companion_vector_meson_correlator,
)
from fragile.physics.operators.pipeline import PipelineResult

# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompanionCorrelatorSection:
    """Container for the companion correlators dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MESON_MODES = ("standard", "score_directed", "score_weighted", "abs2_vacsub")
BARYON_MODES = (
    "det_abs", "flux_action", "flux_sin2", "flux_exp",
    "score_signed", "score_abs",
)
VECTOR_MODES = ("standard", "score_directed", "score_gradient")
VECTOR_PROJECTIONS = ("full", "longitudinal", "transverse")
GLUEBALL_MODES = (
    "re_plaquette", "action_re_plaquette", "phase_action", "phase_sin2",
)


class CompanionCorrelatorSettings(param.Parameterized):
    """Settings for the companion-channel correlator computations."""

    # -- Common --
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.95))
    end_fraction = param.Number(default=1.0, bounds=(0.05, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    max_lag = param.Integer(default=80, bounds=(1, 500))
    use_connected = param.Boolean(default=True)
    color_dims_spec = param.String(
        default="",
        doc="Comma-separated color dims (blank = all available).",
    )
    pair_selection = param.ObjectSelector(
        default="both",
        objects=("both", "distance", "clone"),
    )
    eps = param.Number(default=1e-12, bounds=(0.0, None))

    # -- Tensor checkbox (no mode variants) --
    compute_tensor = param.Boolean(default=False)

    # -- Per-channel operator settings (non-mode) --
    baryon_flux_exp_alpha = param.Number(default=1.0, bounds=(0.0, None))

    vector_unit_displacement = param.Boolean(default=False)

    glueball_momentum_projection = param.Boolean(default=False)
    glueball_momentum_axis = param.Integer(default=0, bounds=(0, 3))
    glueball_momentum_mode_max = param.Integer(default=3, bounds=(0, 8))

    tensor_momentum_axis = param.Integer(default=0, bounds=(0, 3))
    tensor_momentum_mode_max = param.Integer(default=4, bounds=(0, 8))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_color_dims(spec: str, d: int) -> tuple[int, int, int] | None:
    """Parse color_dims_spec into a tuple or None."""
    spec = spec.strip()
    if not spec:
        return None
    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"color_dims_spec must have exactly 3 integers, got {parts}")
    for p in parts:
        if p < 0 or p >= d:
            raise ValueError(f"color dim {p} out of range [0, {d - 1}]")
    return (parts[0], parts[1], parts[2])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_companion_correlator_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
) -> CompanionCorrelatorSection:
    """Build the Companion Correlators tab with callbacks."""

    settings = CompanionCorrelatorSettings()
    status = pn.pane.Markdown(
        "**Companion Correlators:** Load a RunHistory and click Compute.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name="Compute Companion Correlators",
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

    # -- Per-channel mode selectors (MultiSelect) --
    meson_mode_selector = pn.widgets.MultiSelect(
        name="Meson",
        options=list(MESON_MODES),
        value=["standard"],
        size=len(MESON_MODES),
        sizing_mode="stretch_width",
    )
    baryon_mode_selector = pn.widgets.MultiSelect(
        name="Baryon",
        options=list(BARYON_MODES),
        value=["det_abs"],
        size=len(BARYON_MODES),
        sizing_mode="stretch_width",
    )
    vector_mode_selector = pn.widgets.MultiSelect(
        name="Vector",
        options=list(VECTOR_MODES),
        value=["standard"],
        size=len(VECTOR_MODES),
        sizing_mode="stretch_width",
    )
    vector_projection_selector = pn.widgets.MultiSelect(
        name="Vector Projection",
        options=list(VECTOR_PROJECTIONS),
        value=["full"],
        size=len(VECTOR_PROJECTIONS),
        sizing_mode="stretch_width",
    )
    glueball_mode_selector = pn.widgets.MultiSelect(
        name="Glueball",
        options=list(GLUEBALL_MODES),
        value=["re_plaquette"],
        size=len(GLUEBALL_MODES),
        sizing_mode="stretch_width",
    )

    channel_mode_selectors: dict[str, pn.widgets.MultiSelect] = {
        "meson": meson_mode_selector,
        "baryon": baryon_mode_selector,
        "vector": vector_mode_selector,
        "glueball": glueball_mode_selector,
    }

    channel_selection_panel = pn.Column(
        pn.Row(
            meson_mode_selector,
            baryon_mode_selector,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            vector_mode_selector,
            vector_projection_selector,
            glueball_mode_selector,
            sizing_mode="stretch_width",
        ),
        pn.Param(
            settings,
            parameters=["compute_tensor"],
            show_name=False,
        ),
        sizing_mode="stretch_width",
    )

    operator_config_panel = pn.Param(
        settings,
        parameters=[
            "baryon_flux_exp_alpha",
            "vector_unit_displacement",
            "glueball_momentum_projection",
            "glueball_momentum_axis",
            "glueball_momentum_mode_max",
            "tensor_momentum_axis",
            "tensor_momentum_mode_max",
        ],
        show_name=False,
        default_layout=type("OperatorGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Refresh plots helper --

    def _refresh_overlay():
        """Refresh the all-channels overlay plots."""
        result = state.get("companion_correlator_output")
        if result is None:
            return
        si = int(scale_selector.value)
        ls = bool(log_scale_toggle.value)
        overlay_correlator_plot.object = build_all_channels_correlator_overlay(
            result, logy=ls, scale_index=si,
        )
        overlay_meff_plot.object = build_all_channels_meff_overlay(
            result, scale_index=si,
        )
        summary_table.value = build_summary_table(result, scale_index=si)
        correlator_table.value = build_correlator_table(result, scale_index=si)

        # Per-channel group plots
        groups = group_strong_correlator_keys(result.correlators.keys())
        per_channel_container.clear()
        for group_name, keys in groups.items():
            corr_overlay = build_grouped_correlator_plot(
                result, group_name, keys, si, ls,
            )
            meff_overlay = build_grouped_meff_plot(
                result, group_name, keys, si,
            )
            per_channel_container.append(
                pn.pane.Markdown(f"#### {group_name}")
            )
            per_channel_container.append(
                pn.Row(
                    pn.pane.HoloViews(corr_overlay, sizing_mode="stretch_width", linked_axes=False),
                    pn.pane.HoloViews(meff_overlay, sizing_mode="stretch_width", linked_axes=False),
                    sizing_mode="stretch_width",
                )
            )

    def _refresh_single():
        """Refresh the single-channel plots based on channel_selector."""
        result = state.get("companion_correlator_output")
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
                corr_arr, selected, logy=ls,
            )
            single_meff_plot.object = build_single_effective_mass_plot(
                corr_arr, selected,
            )
        else:
            single_correlator_plot.object = _algorithm_placeholder_plot(
                f"No correlator data for {selected}"
            )
            single_meff_plot.object = _algorithm_placeholder_plot(
                f"No m_eff data for {selected}"
            )

        if len(op_arr) > 0:
            single_operator_plot.object = build_single_operator_series_plot(
                op_arr, selected,
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
            use_connected = bool(settings.use_connected)

            # Shared config kwargs
            common_kw: dict[str, Any] = {
                "warmup_fraction": float(settings.warmup_fraction),
                "end_fraction": float(settings.end_fraction),
                "max_lag": int(settings.max_lag),
                "use_connected": use_connected,
                "h_eff": float(settings.h_eff),
                "mass": float(settings.mass),
                "ell0": float(settings.ell0) if settings.ell0 is not None else None,
                "color_dims": color_dims,
                "eps": float(settings.eps),
            }

            # Collect selected modes per channel from MultiSelect widgets
            selected_modes: dict[str, list[str]] = {}
            for family, selector in channel_mode_selectors.items():
                modes = [str(v) for v in selector.value]
                if modes:
                    selected_modes[family] = modes

            has_tensor = bool(settings.compute_tensor)

            if not selected_modes and not has_tensor:
                status.object = "**Error:** No channels selected."
                return

            merged_correlators: dict[str, Any] = {}
            merged_operators: dict[str, Any] = {}
            channel_labels: list[str] = []

            # -- Meson: iterate over selected modes --
            for mode in selected_modes.get("meson", []):
                meson_cfg = MesonPhaseCorrelatorConfig(
                    pair_selection=str(settings.pair_selection),
                    operator_mode=mode,
                    **common_kw,
                )
                out = compute_companion_meson_phase_correlator(history, meson_cfg)
                corrs, ops = extract_meson_phase(
                    out, use_connected=use_connected, prefix="",
                )
                # Suffix keys with mode
                for key, val in corrs.items():
                    merged_correlators[f"{key}_{mode}"] = val
                for key, val in ops.items():
                    merged_operators[f"{key}_{mode}"] = val
            if "meson" in selected_modes:
                channel_labels.append("meson")

            # -- Baryon: iterate over selected modes --
            for mode in selected_modes.get("baryon", []):
                baryon_cfg = BaryonTripletCorrelatorConfig(
                    operator_mode=mode,
                    flux_exp_alpha=float(settings.baryon_flux_exp_alpha),
                    **common_kw,
                )
                out = compute_companion_baryon_correlator(history, baryon_cfg)
                corrs, ops = extract_baryon(
                    out, use_connected=use_connected, prefix="",
                )
                for key, val in corrs.items():
                    merged_correlators[f"{key}_{mode}"] = val
                for key, val in ops.items():
                    merged_operators[f"{key}_{mode}"] = val
            if "baryon" in selected_modes:
                channel_labels.append("baryon")

            # -- Vector: iterate over (mode x projection) --
            selected_projections = [str(v) for v in vector_projection_selector.value]
            if not selected_projections:
                selected_projections = ["full"]

            for mode in selected_modes.get("vector", []):
                for proj in selected_projections:
                    vector_cfg = VectorMesonCorrelatorConfig(
                        pair_selection=str(settings.pair_selection),
                        use_unit_displacement=bool(settings.vector_unit_displacement),
                        operator_mode=mode,
                        projection_mode=proj,
                        **common_kw,
                    )
                    out = compute_companion_vector_meson_correlator(history, vector_cfg)
                    corrs, ops = extract_vector_meson(
                        out, use_connected=use_connected, prefix="",
                    )
                    suffix = f"{mode}_{proj}"
                    for key, val in corrs.items():
                        merged_correlators[f"{key}_{suffix}"] = val
                    for key, val in ops.items():
                        merged_operators[f"{key}_{suffix}"] = val
            if "vector" in selected_modes:
                channel_labels.append("vector")

            # -- Glueball: iterate over selected modes --
            for mode in selected_modes.get("glueball", []):
                glueball_cfg = GlueballColorCorrelatorConfig(
                    operator_mode=mode,
                    use_momentum_projection=bool(settings.glueball_momentum_projection),
                    momentum_axis=int(settings.glueball_momentum_axis),
                    momentum_mode_max=int(settings.glueball_momentum_mode_max),
                    **common_kw,
                )
                out = compute_companion_glueball_color_correlator(history, glueball_cfg)
                corrs, ops = extract_glueball(
                    out, use_connected=use_connected, prefix="",
                )
                for key, val in corrs.items():
                    merged_correlators[f"{key}_{mode}"] = val
                for key, val in ops.items():
                    merged_operators[f"{key}_{mode}"] = val
            if "glueball" in selected_modes:
                channel_labels.append("glueball")

            # -- Tensor: no mode variants, single compute --
            if has_tensor:
                tensor_cfg = TensorMomentumCorrelatorConfig(
                    pair_selection=str(settings.pair_selection),
                    momentum_axis=int(settings.tensor_momentum_axis),
                    momentum_mode_max=int(settings.tensor_momentum_mode_max),
                    **common_kw,
                )
                out = compute_companion_tensor_momentum_correlator(history, tensor_cfg)
                corrs, ops = extract_tensor_momentum(
                    out, use_connected=use_connected,
                )
                # No suffix for tensor (no mode variants)
                merged_correlators.update(corrs)
                merged_operators.update(ops)
                channel_labels.append("tensor")

            if not merged_correlators:
                status.object = "**Error:** No correlators produced."
                return

            # Build PipelineResult directly
            result = PipelineResult(
                correlators=merged_correlators,
                operators=merged_operators,
            )
            state["companion_correlator_output"] = result

            # Update channel selector
            channel_names = [ALL_CHANNELS, *list(result.correlators.keys())]
            channel_selector.options = channel_names
            channel_selector.value = ALL_CHANNELS

            _refresh_all()

            n_channels = len(result.correlators)
            n_ops = len(result.operators)
            status.object = (
                f"**Complete:** {n_channels} correlators from {n_ops} operators "
                f"({', '.join(channel_labels)})."
            )

        run_tab_computation(
            state,
            status,
            "companion correlators",
            _compute,
        )

    run_button.on_click(on_run)

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            "**Companion Correlators:** Uses the ``new_channels`` companion-based "
            "compute functions. Select operator modes per channel below. "
            "Multiple modes can be selected simultaneously to compare results."
        ),
        alert_type="info",
        sizing_mode="stretch_width",
    )

    tab = pn.Column(
        status,
        info_note,
        pn.Row(run_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Common Settings", common_settings_panel),
            ("Channel & Mode Selection", channel_selection_panel),
            ("Operator Settings", operator_config_panel),
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
        status.object = "**Companion Correlators ready:** click Compute Companion Correlators."
        if defer:
            return
        state["companion_correlator_output"] = None
        summary_table.value = pd.DataFrame()
        correlator_table.value = pd.DataFrame()
        channel_selector.options = [ALL_CHANNELS]
        channel_selector.value = ALL_CHANNELS
        placeholder = _algorithm_placeholder_plot("Run pipeline to show plots.")
        overlay_correlator_plot.object = placeholder
        overlay_meff_plot.object = placeholder
        single_correlator_plot.object = placeholder
        single_meff_plot.object = placeholder
        single_operator_plot.object = placeholder
        per_channel_container.clear()
        scale_selector.visible = False

    return CompanionCorrelatorSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
    )
