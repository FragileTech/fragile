"""Strong Correlators dashboard tab.

Provides a UI to drive the strong-force companion-channel pipeline:
select channels, configure operator parameters, run the pipeline,
and visualize correlator curves, effective masses, and operator time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.app.correlator_plots import (
    build_all_channels_correlator_overlay,
    build_all_channels_meff_overlay,
    build_single_correlator_plot,
    build_single_effective_mass_plot,
    build_single_operator_series_plot,
    get_correlator_array,
    get_operator_series_array,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.operators import (
    BaryonOperatorConfig,
    ChannelConfigBase,
    compute_baryon_operators,
    compute_correlators_batched,
    compute_glueball_operators,
    compute_meson_operators,
    compute_propagator_pipeline,
    compute_strong_force_pipeline,
    compute_tensor_operators,
    compute_vector_operators,
    CorrelatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    MultiscaleConfig,
    PipelineConfig,
    PipelineResult,
    TensorOperatorConfig,
    VectorOperatorConfig,
)

_MESON_REGULAR: tuple[str, ...] = (
    "standard", "score_directed", "score_weighted", "abs2_vacsub",
)
_VECTOR_REGULAR: tuple[str, ...] = (
    "standard", "score_directed", "score_gradient",
)
_BARYON_REGULAR: tuple[str, ...] = (
    "det_abs", "flux_action", "flux_sin2", "flux_exp", "score_signed", "score_abs",
)
_GLUEBALL_REGULAR: tuple[str, ...] = (
    "re_plaquette", "action_re_plaquette", "phase_action", "phase_sin2",
)


def _with_propagator(modes: tuple[str, ...]) -> tuple[str, ...]:
    """Append ``<mode>_propagator`` variants after regular modes."""
    return modes + tuple(f"{m}_propagator" for m in modes)


MESON_MODES: tuple[str, ...] = _with_propagator(_MESON_REGULAR)
VECTOR_MODES: tuple[str, ...] = _with_propagator(_VECTOR_REGULAR)
BARYON_MODES: tuple[str, ...] = _with_propagator(_BARYON_REGULAR)
GLUEBALL_MODES: tuple[str, ...] = _with_propagator(_GLUEBALL_REGULAR)
TENSOR_MODES: tuple[str, ...] = ("standard",)

# Modes that require scores to be prepared in PreparedChannelData
_SCORE_MODES: dict[str, set[str]] = {
    "meson": {"score_directed", "score_weighted"},
    "vector": {"score_directed", "score_gradient"},
    "baryon": {"score_signed", "score_abs"},
}


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class StrongCorrelatorSection:
    """Container for the strong correlators dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class StrongCorrelatorSettings(param.Parameterized):
    """Settings for the strong-force correlator pipeline."""

    # -- Common (ChannelConfigBase) --
    warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.95))
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
    max_lag = param.Integer(default=80, bounds=(1, 500))
    use_connected = param.Boolean(default=True)

    # -- Per-channel settings --
    vector_projection = param.ObjectSelector(
        default="full",
        objects=("full", "longitudinal", "transverse"),
    )
    vector_unit_displacement = param.Boolean(default=False)
    baryon_flux_exp_alpha = param.Number(default=1.0, bounds=(0.0, None))
    glueball_momentum_projection = param.Boolean(default=False)
    glueball_momentum_axis = param.Integer(default=0, bounds=(0, 3))
    glueball_momentum_mode_max = param.Integer(default=3, bounds=(0, 8))
    tensor_momentum_axis = param.Integer(default=0, bounds=(0, 3))
    tensor_momentum_mode_max = param.Integer(default=4, bounds=(0, 8))

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
# Helpers: dim parsing
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
# Helpers: summary / correlator tables
# ---------------------------------------------------------------------------


def _build_summary_table(
    result: PipelineResult,
    scale_index: int = 0,
) -> pd.DataFrame:
    """Build one-row-per-channel summary table."""
    rows: list[dict[str, Any]] = []
    for name in result.correlators:
        arr = get_correlator_array(result, name, scale_index)
        if len(arr) == 0:
            continue
        op_arr = get_operator_series_array(result, name, scale_index)
        n_frames = max(0, len(op_arr))

        c0 = float(arr[0]) if len(arr) > 0 else np.nan
        c1 = float(arr[1]) if len(arr) > 1 else np.nan
        c_last = float(arr[-1]) if len(arr) > 0 else np.nan
        m_eff1 = np.nan
        if len(arr) > 1 and abs(c0) > 1e-30 and c1 / c0 > 0:
            m_eff1 = np.log(c0 / c1)

        row: dict[str, Any] = {
            "channel": name,
            "C(0)": c0,
            "C(1)": c1,
            f"C({len(arr) - 1})": c_last,
            "m_eff(1)": m_eff1,
            "n_frames": n_frames,
        }
        if result.scales is not None:
            row["scale_index"] = scale_index
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_correlator_table(
    result: PipelineResult,
    scale_index: int = 0,
) -> pd.DataFrame:
    """Build full correlator table: rows = channels, columns = lag values."""
    if not result.correlators:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name in result.correlators:
        arr = get_correlator_array(result, name, scale_index)
        if len(arr) == 0:
            continue
        row: dict[str, Any] = {"channel": name}
        for lag_i, val in enumerate(arr):
            row[f"lag_{lag_i}"] = float(val)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_strong_correlator_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
) -> StrongCorrelatorSection:
    """Build the Strong Correlators tab with callbacks."""

    settings = StrongCorrelatorSettings()
    status = pn.pane.Markdown(
        "**Strong Correlators:** Load a RunHistory and click Compute.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name="Compute Strong Correlators",
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
    # Overlay (all channels)
    overlay_correlator_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    overlay_meff_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    # Single channel (selected via widget)
    single_correlator_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    single_meff_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    single_operator_plot = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)

    placeholder = _algorithm_placeholder_plot("Run pipeline to show plots.")
    overlay_correlator_plot.object = placeholder
    overlay_meff_plot.object = placeholder
    single_correlator_plot.object = placeholder
    single_meff_plot.object = placeholder
    single_operator_plot.object = placeholder

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

    # -- Per-channel mode selectors --
    meson_mode_selector = pn.widgets.MultiSelect(
        name="Meson",
        options=list(MESON_MODES),
        value=["standard"],
        size=len(MESON_MODES),
        sizing_mode="stretch_width",
    )
    vector_mode_selector = pn.widgets.MultiSelect(
        name="Vector",
        options=list(VECTOR_MODES),
        value=["standard"],
        size=len(VECTOR_MODES),
        sizing_mode="stretch_width",
    )
    baryon_mode_selector = pn.widgets.MultiSelect(
        name="Baryon",
        options=list(BARYON_MODES),
        value=["det_abs"],
        size=len(BARYON_MODES),
        sizing_mode="stretch_width",
    )
    glueball_mode_selector = pn.widgets.MultiSelect(
        name="Glueball",
        options=list(GLUEBALL_MODES),
        value=["re_plaquette"],
        size=len(GLUEBALL_MODES),
        sizing_mode="stretch_width",
    )
    tensor_mode_selector = pn.widgets.MultiSelect(
        name="Tensor",
        options=list(TENSOR_MODES),
        value=[],
        size=len(TENSOR_MODES),
        sizing_mode="stretch_width",
    )
    channel_mode_selectors: dict[str, pn.widgets.MultiSelect] = {
        "meson": meson_mode_selector,
        "vector": vector_mode_selector,
        "baryon": baryon_mode_selector,
        "glueball": glueball_mode_selector,
        "tensor": tensor_mode_selector,
    }

    operator_config_panel = pn.Param(
        settings,
        parameters=[
            "vector_projection",
            "vector_unit_displacement",
            "baryon_flux_exp_alpha",
            "glueball_momentum_projection",
            "glueball_momentum_axis",
            "glueball_momentum_mode_max",
            "tensor_momentum_axis",
            "tensor_momentum_mode_max",
        ],
        show_name=False,
        default_layout=type("OperatorGrid", (pn.GridBox,), {"ncols": 2}),
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

    # -- Refresh plots helper --

    def _refresh_overlay():
        """Refresh the all-channels overlay plots."""
        result: PipelineResult | None = state.get("strong_correlator_output")
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
        summary_table.value = _build_summary_table(result, scale_index=si)
        correlator_table.value = _build_correlator_table(result, scale_index=si)

    def _refresh_single():
        """Refresh the single-channel plots based on channel_selector."""
        result: PipelineResult | None = state.get("strong_correlator_output")
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

            # Build per-channel configs from settings
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

            # Collect selected modes per channel
            selected_modes: dict[str, list[str]] = {}
            for family, selector in channel_mode_selectors.items():
                modes = [str(v) for v in selector.value]
                if modes:
                    selected_modes[family] = modes

            if not selected_modes:
                status.object = "**Error:** No channels selected."
                return

            # Separate regular modes from propagator modes per channel
            regular_modes: dict[str, list[str]] = {}
            propagator_modes: dict[str, list[str]] = {}
            for family, modes in selected_modes.items():
                regs = [m for m in modes if not m.endswith("_propagator")]
                props = [
                    m.removesuffix("_propagator")
                    for m in modes if m.endswith("_propagator")
                ]
                if regs:
                    regular_modes[family] = regs
                if props:
                    propagator_modes[family] = props

            # Determine which channels the pipeline must prepare data for
            pipeline_channels = list(
                dict.fromkeys(list(regular_modes) + list(propagator_modes))
            )
            if not pipeline_channels:
                status.object = "**Error:** No channels selected."
                return

            # Pick first mode per channel for the initial pipeline run.
            # Prefer a score mode if any is selected (regular OR propagator),
            # so PreparedChannelData includes scores for subsequent calls.
            def _pick_first(family: str, modes: list[str]) -> str:
                score_set = _SCORE_MODES.get(family, set())
                for m in modes:
                    if m in score_set:
                        return m
                return modes[0]

            first_modes: dict[str, str] = {}
            _defaults = {"meson": "standard", "vector": "standard",
                         "baryon": "det_abs", "glueball": "re_plaquette",
                         "tensor": "standard"}
            for family in pipeline_channels:
                # Combine regular + propagator base modes for score detection
                all_base = regular_modes.get(family, []) + propagator_modes.get(family, [])
                if all_base:
                    first_modes[family] = _pick_first(family, all_base)
                else:
                    first_modes[family] = _defaults.get(family, "standard")

            # Build operator configs using first modes
            meson_cfg = MesonOperatorConfig(
                operator_mode=first_modes.get("meson", "standard"),
                **base_kwargs,
            )
            vector_cfg = VectorOperatorConfig(
                operator_mode=first_modes.get("vector", "standard"),
                projection_mode=str(settings.vector_projection),
                use_unit_displacement=bool(settings.vector_unit_displacement),
                **base_kwargs,
            )
            baryon_cfg = BaryonOperatorConfig(
                operator_mode=first_modes.get("baryon", "det_abs"),
                flux_exp_alpha=float(settings.baryon_flux_exp_alpha),
                **base_kwargs,
            )
            glueball_cfg = GlueballOperatorConfig(
                operator_mode=first_modes.get("glueball", "re_plaquette"),
                use_momentum_projection=bool(settings.glueball_momentum_projection),
                momentum_axis=int(settings.glueball_momentum_axis),
                momentum_mode_max=int(settings.glueball_momentum_mode_max),
                **base_kwargs,
            )
            tensor_cfg = TensorOperatorConfig(
                momentum_axis=int(settings.tensor_momentum_axis),
                momentum_mode_max=int(settings.tensor_momentum_mode_max),
                **base_kwargs,
            )

            # Run the pipeline once to get prepared_data + first-mode results
            pipeline_config = PipelineConfig(
                base=ChannelConfigBase(**base_kwargs),
                meson=meson_cfg,
                vector=vector_cfg,
                baryon=baryon_cfg,
                glueball=glueball_cfg,
                tensor=tensor_cfg,
                correlator=correlator_cfg,
                multiscale=multiscale_cfg,
                channels=pipeline_channels,
            )
            result = compute_strong_force_pipeline(history, pipeline_config)

            effective_n_scales = (
                multiscale_cfg.n_scales
                if result.scales is not None
                else 1
            )

            # Rename first-mode correlators/operators with mode suffix
            merged_correlators: dict[str, Any] = {}
            merged_operators: dict[str, Any] = {}

            # Map operator keys back to their channel family
            _family_of_key: dict[str, str] = {}
            _FAMILY_PREFIXES = {
                "meson": ("scalar", "pseudoscalar"),
                "vector": ("vector", "axial_vector"),
                "baryon": ("nucleon",),
                "glueball": ("glueball",),
                "tensor": ("tensor",),
            }
            for fam, prefixes in _FAMILY_PREFIXES.items():
                for prefix in prefixes:
                    _family_of_key[prefix] = fam
                    # Also handle momentum variants like glueball_momentum_*
            for key in list(result.correlators):
                for fam, prefixes in _FAMILY_PREFIXES.items():
                    if any(key == p or key.startswith(p + "_momentum") for p in prefixes):
                        _family_of_key[key] = fam
                        break

            for key, val in result.correlators.items():
                fam = _family_of_key.get(key)
                mode = first_modes.get(fam, "") if fam else ""
                merged_correlators[f"{key}_{mode}"] = val
            for key, val in result.operators.items():
                fam = _family_of_key.get(key)
                mode = first_modes.get(fam, "") if fam else ""
                merged_operators[f"{key}_{mode}"] = val

            # Compute additional modes directly using prepared_data
            _OPERATOR_FN = {
                "meson": compute_meson_operators,
                "vector": compute_vector_operators,
                "baryon": compute_baryon_operators,
                "glueball": compute_glueball_operators,
                "tensor": compute_tensor_operators,
            }

            def _make_config(family: str, mode: str):
                if family == "meson":
                    return MesonOperatorConfig(operator_mode=mode, **base_kwargs)
                if family == "vector":
                    return VectorOperatorConfig(
                        operator_mode=mode,
                        projection_mode=str(settings.vector_projection),
                        use_unit_displacement=bool(settings.vector_unit_displacement),
                        **base_kwargs,
                    )
                if family == "baryon":
                    return BaryonOperatorConfig(
                        operator_mode=mode,
                        flux_exp_alpha=float(settings.baryon_flux_exp_alpha),
                        **base_kwargs,
                    )
                if family == "glueball":
                    return GlueballOperatorConfig(
                        operator_mode=mode,
                        use_momentum_projection=bool(
                            settings.glueball_momentum_projection
                        ),
                        momentum_axis=int(settings.glueball_momentum_axis),
                        momentum_mode_max=int(settings.glueball_momentum_mode_max),
                        **base_kwargs,
                    )
                # tensor has no operator_mode
                return TensorOperatorConfig(
                    momentum_axis=int(settings.tensor_momentum_axis),
                    momentum_mode_max=int(settings.tensor_momentum_mode_max),
                    **base_kwargs,
                )

            data = result.prepared_data
            for family, modes in regular_modes.items():
                extra = [m for m in modes if m != first_modes.get(family)]
                op_fn = _OPERATOR_FN.get(family)
                if not extra or op_fn is None or data is None:
                    continue
                for mode in extra:
                    cfg = _make_config(family, mode)
                    ops = op_fn(data, cfg)
                    corrs = compute_correlators_batched(
                        ops,
                        max_lag=correlator_cfg.max_lag,
                        use_connected=correlator_cfg.use_connected,
                        n_scales=effective_n_scales,
                    )
                    for key, val in corrs.items():
                        merged_correlators[f"{key}_{mode}"] = val
                    for key, val in ops.items():
                        merged_operators[f"{key}_{mode}"] = val

            # Propagator correlators (frozen-source topology), per mode
            if propagator_modes and data is not None:
                for family, prop_mode_list in propagator_modes.items():
                    for pmode in prop_mode_list:
                        pcfg = _make_config(family, pmode)
                        # Build kwargs for compute_propagator_pipeline
                        prop_kw: dict[str, Any] = {
                            f"{family}_config": pcfg,
                        }
                        prop_result = compute_propagator_pipeline(
                            data,
                            channels=[family],
                            max_lag=int(settings.max_lag),
                            use_connected=bool(settings.use_connected),
                            **prop_kw,
                        )
                        for prop_name, prop_ch in prop_result.channels.items():
                            merged_correlators[
                                f"{prop_name}_{pmode}_propagator"
                            ] = prop_ch.connected

            result.correlators = merged_correlators
            result.operators = merged_operators
            state["strong_correlator_output"] = result

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
            "strong correlators",
            _compute,
        )

    run_button.on_click(on_run)

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            "**Strong Correlators:** Select operator modes per channel below. "
            "Multiple modes can be selected simultaneously to compare across "
            "scales. The 'propagator' option computes the frozen-source topology."
        ),
        alert_type="info",
        sizing_mode="stretch_width",
    )

    channel_selection_panel = pn.Column(
        pn.Row(
            meson_mode_selector,
            vector_mode_selector,
            baryon_mode_selector,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            glueball_mode_selector,
            tensor_mode_selector,
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
            ("Channel & Mode Selection", channel_selection_panel),
            ("Operator Settings", operator_config_panel),
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
        status.object = "**Strong Correlators ready:** click Compute Strong Correlators."
        if defer:
            return
        state["strong_correlator_output"] = None
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
        scale_selector.visible = False

    return StrongCorrelatorSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
    )
