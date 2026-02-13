"""Tensor calibration tab extended with reusable multiscale GEVP diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import panel as pn

from fragile.fractalai.qft.correlator_channels import ChannelCorrelatorResult
from fragile.fractalai.qft.dashboard.gevp_dashboard import (
    build_gevp_dashboard_sections,
    channel_family_key,
    clear_gevp_dashboard,
    create_gevp_dashboard_widgets,
    extract_final_gevp_mass,
    GEVPDashboardWidgets,
    update_gevp_dashboard,
)
from fragile.fractalai.qft.dashboard.tensor_calibration import (
    build_tensor_calibration_tab_layout,
    clear_tensor_calibration_tab,
    create_tensor_calibration_widgets,
    TensorCalibrationWidgets,
    update_tensor_calibration_tab,
)
from fragile.fractalai.qft.multiscale_analysis import (
    analyze_channel_across_scales,
    build_estimator_table_rows,
    build_pairwise_table_rows,
    format_consensus_summary,
)
from fragile.fractalai.qft.multiscale_strong_force import MultiscaleStrongForceOutput
from fragile.fractalai.qft.operator_analysis import (
    build_consensus_plot,
    build_mass_vs_scale_plot,
    build_multiscale_correlator_plot,
    build_multiscale_effective_mass_plot,
)


GEVP_DIRTY_STATE_KEY = "_tensor_calibration_gevp_dirty"
FULL_ORIGINAL_SCALE_LABEL = "full_original_no_threshold"
TENSOR_VARIANT_TABS: tuple[tuple[str, str], ...] = (
    ("tensor", "Tensor"),
    ("tensor_traceless", "Tensor Traceless"),
    ("tensor_companion", "Tensor Companion"),
    ("tensor_traceless_companion", "Tensor Traceless Companion"),
)


@dataclass
class TensorGEVPCalibrationWidgets(TensorCalibrationWidgets):
    """Tensor calibration widgets plus reusable GEVP diagnostics controls."""

    multiscale_variant_tabs: pn.Tabs
    gevp_family_select: pn.widgets.MultiChoice
    gevp_family_select_all_button: pn.widgets.Button
    gevp_family_clear_button: pn.widgets.Button
    gevp_scale_select: pn.widgets.MultiChoice
    gevp_scale_select_all_button: pn.widgets.Button
    gevp_scale_clear_button: pn.widgets.Button
    gevp_compute_button: pn.widgets.Button
    gevp: GEVPDashboardWidgets


def create_tensor_gevp_calibration_widgets() -> TensorGEVPCalibrationWidgets:
    """Create tensor calibration widgets with embedded GEVP diagnostics widgets."""
    base = create_tensor_calibration_widgets()
    initial_tabs = pn.Tabs(sizing_mode="stretch_width", dynamic=False)
    for _, label in TENSOR_VARIANT_TABS:
        initial_tabs.append((
            label,
            pn.pane.Markdown(
                "_Run tensor calibration to populate this multiscale analysis tab._",
                sizing_mode="stretch_width",
            ),
        ))
    return TensorGEVPCalibrationWidgets(
        status=base.status,
        estimator_toggles=base.estimator_toggles,
        mass_mode=base.mass_mode,
        estimator_table=base.estimator_table,
        pairwise_table=base.pairwise_table,
        systematics_badge=base.systematics_badge,
        summary=base.summary,
        correction_summary=base.correction_summary,
        dispersion_plot=base.dispersion_plot,
        component_dispersion_plot=base.component_dispersion_plot,
        anisotropy_summary=base.anisotropy_summary,
        anisotropy_table=base.anisotropy_table,
        anisotropy_plot=base.anisotropy_plot,
        multiscale_plot=base.multiscale_plot,
        calibration_surface_plot=base.calibration_surface_plot,
        calibration_residual_plot=base.calibration_residual_plot,
        cv_summary=base.cv_summary,
        multiscale_variant_tabs=initial_tabs,
        gevp_family_select=pn.widgets.MultiChoice(
            name="GEVP Families",
            options=["tensor"],
            value=["tensor"],
            sizing_mode="stretch_width",
        ),
        gevp_family_select_all_button=pn.widgets.Button(
            name="All Families",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_family_clear_button=pn.widgets.Button(
            name="Clear Families",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_scale_select=pn.widgets.MultiChoice(
            name="GEVP Scales",
            options=["s0"],
            value=["s0"],
            sizing_mode="stretch_width",
        ),
        gevp_scale_select_all_button=pn.widgets.Button(
            name="All Scales",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_scale_clear_button=pn.widgets.Button(
            name="Clear Scales",
            button_type="default",
            sizing_mode="stretch_width",
        ),
        gevp_compute_button=pn.widgets.Button(
            name="Recompute GEVP",
            button_type="primary",
            sizing_mode="stretch_width",
        ),
        gevp=create_gevp_dashboard_widgets(),
    )


def build_tensor_gevp_calibration_tab_layout(
    w: TensorGEVPCalibrationWidgets,
    *,
    run_button: pn.widgets.Button,
    settings_layout: pn.layout.Panel | None = None,
) -> pn.Column:
    """Build tensor calibration layout and append reusable GEVP diagnostics sections."""
    base_layout = build_tensor_calibration_tab_layout(
        w,
        run_button=run_button,
        settings_layout=settings_layout,
    )
    base_layout.objects.extend([
        pn.layout.Divider(),
        pn.pane.Markdown("### Multiscale Analysis By Tensor Variant"),
        pn.pane.Markdown(
            "_One tab per tensor-family channel, with per-scale correlator, effective-mass, "
            "mass-vs-scale, consensus, and discrepancy diagnostics._"
        ),
        w.multiscale_variant_tabs,
        pn.layout.Divider(),
        pn.pane.Markdown("### GEVP Controls"),
        pn.Row(
            w.gevp_family_select,
            w.gevp_scale_select,
            w.gevp_compute_button,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            w.gevp_family_select_all_button,
            w.gevp_family_clear_button,
            w.gevp_scale_select_all_button,
            w.gevp_scale_clear_button,
            sizing_mode="stretch_width",
        ),
        *build_gevp_dashboard_sections(w.gevp),
    ])
    return base_layout


def clear_tensor_gevp_calibration_tab(
    w: TensorGEVPCalibrationWidgets,
    status_text: str,
    *,
    state: dict[str, Any] | None = None,
) -> None:
    """Reset tensor calibration and GEVP sections to defaults."""
    clear_tensor_calibration_tab(w, status_text)
    w.multiscale_variant_tabs.clear()
    for _, label in TENSOR_VARIANT_TABS:
        w.multiscale_variant_tabs.append((
            label,
            pn.pane.Markdown(
                "_Run tensor calibration to populate this multiscale analysis tab._",
                sizing_mode="stretch_width",
            ),
        ))
    w.gevp_family_select.options = []
    w.gevp_family_select.value = []
    w.gevp_scale_select.options = []
    w.gevp_scale_select.value = []
    clear_gevp_dashboard(w.gevp)
    if isinstance(state, dict):
        state[GEVP_DIRTY_STATE_KEY] = False


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


def _is_valid_result(result: ChannelCorrelatorResult | None) -> bool:
    return result is not None and int(getattr(result, "n_samples", 0)) > 0


def _collect_tensor_original_results(
    *,
    base_results: Mapping[str, ChannelCorrelatorResult],
    strong_tensor_result: ChannelCorrelatorResult | None,
    tensor_momentum_results: Mapping[str, ChannelCorrelatorResult] | None,
) -> dict[str, ChannelCorrelatorResult]:
    original: dict[str, ChannelCorrelatorResult] = {}
    candidate = base_results.get("tensor")
    if _is_valid_result(candidate):
        original["tensor"] = candidate
    candidate = base_results.get("tensor_traceless")
    if _is_valid_result(candidate):
        original["tensor_traceless"] = candidate
    if _is_valid_result(strong_tensor_result):
        original["tensor_companion"] = strong_tensor_result

    if tensor_momentum_results is not None:
        for raw_name, result in tensor_momentum_results.items():
            if not _is_valid_result(result):
                continue
            raw_name_s = str(raw_name)
            mode_n, _ = _parse_tensor_momentum_name(raw_name_s)
            if mode_n is None:
                continue
            original[raw_name_s] = result
    return original


def _collect_tensor_per_scale_results(
    *,
    noncomp_multiscale_output: MultiscaleStrongForceOutput | None,
    companion_multiscale_output: MultiscaleStrongForceOutput | None,
) -> dict[str, Sequence[ChannelCorrelatorResult]]:
    per_scale: dict[str, Sequence[ChannelCorrelatorResult]] = {}
    if noncomp_multiscale_output is not None and int(noncomp_multiscale_output.scales.numel()) > 0:
        for raw_name, series in noncomp_multiscale_output.per_scale_results.items():
            per_scale[str(raw_name)] = series
    if (
        companion_multiscale_output is not None
        and int(companion_multiscale_output.scales.numel()) > 0
    ):
        for raw_name, series in companion_multiscale_output.per_scale_results.items():
            per_scale[str(raw_name)] = series
    return per_scale


def _resolve_gevp_filter_values(
    state: Mapping[str, Any] | None,
) -> tuple[float, int, float, bool]:
    default = (0.5, 10, 30.0, True)
    if not isinstance(state, Mapping):
        return default
    try:
        min_r2 = float(state.get("_multiscale_gevp_min_operator_r2", default[0]))
    except (TypeError, ValueError):
        min_r2 = default[0]
    try:
        min_windows = max(0, int(state.get("_multiscale_gevp_min_operator_windows", default[1])))
    except (TypeError, ValueError):
        min_windows = default[1]
    try:
        max_error_pct = float(state.get("_multiscale_gevp_max_operator_error_pct", default[2]))
    except (TypeError, ValueError):
        max_error_pct = default[2]
    remove_artifacts = bool(
        state.get(
            "_multiscale_gevp_remove_artifacts",
            state.get("_multiscale_gevp_exclude_zero_error_operators", default[3]),
        )
    )
    return min_r2, min_windows, max_error_pct, remove_artifacts


def _configure_gevp_controls(
    w: TensorGEVPCalibrationWidgets,
    *,
    per_scale_results: Mapping[str, Sequence[ChannelCorrelatorResult]],
    original_results: Mapping[str, ChannelCorrelatorResult],
) -> None:
    families = sorted({
        channel_family_key(str(raw_name))
        for raw_name in list(per_scale_results.keys()) + list(original_results.keys())
    })
    if not families:
        families = ["tensor"]

    selected_families = [str(v) for v in (w.gevp_family_select.value or [])]
    selected_families = [name for name in selected_families if name in families]
    if not selected_families:
        selected_families = [families[0]]
    w.gevp_family_select.options = families
    w.gevp_family_select.value = selected_families

    max_scales = 0
    for series in per_scale_results.values():
        max_scales = max(max_scales, len(series))
    scale_options = [f"s{i}" for i in range(max(0, max_scales))]
    if len(original_results) > 0:
        scale_options.append(FULL_ORIGINAL_SCALE_LABEL)
    if not scale_options:
        scale_options = ["s0"]

    selected_scales = [str(v) for v in (w.gevp_scale_select.value or [])]
    selected_scales = [scale for scale in selected_scales if scale in scale_options]
    if not selected_scales:
        selected_scales = list(scale_options)
    w.gevp_scale_select.options = scale_options
    w.gevp_scale_select.value = selected_scales


def _extract_reference_stats(
    result: ChannelCorrelatorResult | None,
) -> tuple[float | None, float, float]:
    if not _is_valid_result(result):
        return None, float("nan"), float("nan")
    mass_fit = getattr(result, "mass_fit", None)
    if not isinstance(mass_fit, Mapping):
        return None, float("nan"), float("nan")
    mass = float(mass_fit.get("mass", float("nan")))
    if not (math.isfinite(mass) and mass > 0):
        return None, float("nan"), float("nan")
    return (
        mass,
        float(mass_fit.get("mass_error", float("nan"))),
        float(mass_fit.get("r_squared", float("nan"))),
    )


def _reference_result_for_variant(
    channel_name: str,
    *,
    base_results: Mapping[str, ChannelCorrelatorResult],
    strong_tensor_result: ChannelCorrelatorResult | None,
) -> ChannelCorrelatorResult | None:
    if channel_name == "tensor":
        return base_results.get("tensor")
    if channel_name == "tensor_traceless":
        return base_results.get("tensor_traceless")
    if channel_name == "tensor_companion":
        return strong_tensor_result
    if channel_name == "tensor_traceless_companion":
        return base_results.get("tensor_traceless")
    return None


def _output_for_variant(
    channel_name: str,
    *,
    noncomp_multiscale_output: MultiscaleStrongForceOutput | None,
    companion_multiscale_output: MultiscaleStrongForceOutput | None,
) -> tuple[list[ChannelCorrelatorResult], np.ndarray]:
    candidate_outputs: list[MultiscaleStrongForceOutput] = []
    if noncomp_multiscale_output is not None and int(noncomp_multiscale_output.scales.numel()) > 0:
        candidate_outputs.append(noncomp_multiscale_output)
    if (
        companion_multiscale_output is not None
        and int(companion_multiscale_output.scales.numel()) > 0
    ):
        candidate_outputs.append(companion_multiscale_output)
    for output in candidate_outputs:
        if channel_name not in output.per_scale_results:
            continue
        per_scale = output.per_scale_results.get(channel_name, [])
        if not per_scale:
            continue
        scales = output.scales.detach().cpu().numpy().astype(float, copy=False)
        if len(per_scale) != len(scales):
            continue
        return list(per_scale), scales
    return [], np.asarray([], dtype=float)


def _plot_or_markdown(
    plot_obj: Any,
    *,
    empty_text: str,
) -> pn.viewable.Viewable:
    if plot_obj is None:
        return pn.pane.Markdown(empty_text, sizing_mode="stretch_width")
    return pn.pane.HoloViews(plot_obj, sizing_mode="stretch_width", linked_axes=False)


def _build_tensor_variant_multiscale_panel(
    *,
    channel_name: str,
    label: str,
    per_scale_results: list[ChannelCorrelatorResult],
    scales: np.ndarray,
    reference_result: ChannelCorrelatorResult | None,
    companion_gevp_results: Mapping[str, Any] | None,
) -> pn.Column:
    if not per_scale_results or scales.size == 0:
        return pn.Column(
            pn.pane.Markdown(
                "_No multiscale output for this variant in the current tensor run. "
                "Enable the matching multiscale estimator and recompute._",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    bundle = analyze_channel_across_scales(per_scale_results, scales, channel_name)
    reference_mass, reference_error, reference_r2 = _extract_reference_stats(reference_result)
    finite_scales = scales[np.isfinite(scales)]
    reference_scale = float(np.max(finite_scales)) if finite_scales.size > 0 else float("nan")
    gevp_mass, gevp_error = extract_final_gevp_mass(
        channel_family_key(channel_name),
        companion_gevp_results,
    )

    estimator_rows = build_estimator_table_rows(
        bundle.measurements,
        original_mass=reference_mass,
        original_error=reference_error,
        original_r2=reference_r2,
        original_scale=reference_scale,
    )
    pairwise_rows = build_pairwise_table_rows(bundle.discrepancies)
    summary_md = format_consensus_summary(
        bundle.consensus,
        bundle.discrepancies,
        channel_name,
        reference_mass=reference_mass,
    )

    verdict_text = f"Systematics verdict: {bundle.verdict.label}. {bundle.verdict.details}"
    verdict_badge = pn.pane.Alert(
        verdict_text,
        alert_type=bundle.verdict.alert_type,
        sizing_mode="stretch_width",
    )

    correlator_plot = build_multiscale_correlator_plot(
        per_scale_results,
        scales,
        channel_name,
        reference_result=reference_result if reference_mass is not None else None,
        reference_scale=reference_scale,
    )
    effective_mass_plot = build_multiscale_effective_mass_plot(
        per_scale_results,
        scales,
        channel_name,
        reference_result=reference_result if reference_mass is not None else None,
        reference_scale=reference_scale,
    )
    mass_vs_scale_plot = build_mass_vs_scale_plot(
        bundle.measurements,
        channel_name,
        bundle.consensus,
        reference_mass=reference_mass,
        reference_scale=reference_scale,
        gevp_mass=gevp_mass,
        gevp_error=gevp_error,
        gevp_scale=reference_scale,
    )
    consensus_plot = build_consensus_plot(
        bundle.measurements,
        bundle.consensus,
        channel_name,
        reference_mass=reference_mass,
        gevp_mass=gevp_mass,
        gevp_error=gevp_error,
    )

    return pn.Column(
        pn.pane.Markdown(f"#### {label}", sizing_mode="stretch_width"),
        verdict_badge,
        pn.pane.Markdown(summary_md, sizing_mode="stretch_width"),
        pn.pane.Markdown("##### Correlator Across Scales"),
        _plot_or_markdown(
            correlator_plot,
            empty_text="_No correlator plot available for this variant._",
        ),
        pn.pane.Markdown("##### Effective Mass Across Scales"),
        _plot_or_markdown(
            effective_mass_plot,
            empty_text="_No effective-mass plot available for this variant._",
        ),
        pn.pane.Markdown("##### Mass vs Scale"),
        _plot_or_markdown(
            mass_vs_scale_plot,
            empty_text="_No mass-vs-scale plot available for this variant._",
        ),
        pn.pane.Markdown("##### Consensus"),
        _plot_or_markdown(
            consensus_plot,
            empty_text="_No consensus plot available for this variant._",
        ),
        pn.pane.Markdown("##### Scale-As-Estimator Table"),
        pn.widgets.Tabulator(
            pd.DataFrame(estimator_rows) if estimator_rows else pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        pn.pane.Markdown("##### Pairwise Discrepancies"),
        pn.widgets.Tabulator(
            pd.DataFrame(pairwise_rows) if pairwise_rows else pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def _update_tensor_multiscale_variant_tabs(
    w: TensorGEVPCalibrationWidgets,
    *,
    base_results: Mapping[str, ChannelCorrelatorResult],
    strong_tensor_result: ChannelCorrelatorResult | None,
    noncomp_multiscale_output: MultiscaleStrongForceOutput | None,
    companion_multiscale_output: MultiscaleStrongForceOutput | None,
    companion_gevp_results: Mapping[str, Any] | None,
) -> None:
    w.multiscale_variant_tabs.clear()
    for channel_name, label in TENSOR_VARIANT_TABS:
        per_scale_results, scales = _output_for_variant(
            channel_name,
            noncomp_multiscale_output=noncomp_multiscale_output,
            companion_multiscale_output=companion_multiscale_output,
        )
        reference_result = _reference_result_for_variant(
            channel_name,
            base_results=base_results,
            strong_tensor_result=strong_tensor_result,
        )
        panel = _build_tensor_variant_multiscale_panel(
            channel_name=channel_name,
            label=label,
            per_scale_results=per_scale_results,
            scales=scales,
            reference_result=reference_result,
            companion_gevp_results=companion_gevp_results,
        )
        w.multiscale_variant_tabs.append((label, panel))


def update_tensor_gevp_calibration_tab(
    w: TensorGEVPCalibrationWidgets,
    *,
    base_results: dict[str, ChannelCorrelatorResult],
    strong_tensor_result: ChannelCorrelatorResult | None,
    tensor_momentum_results: dict[str, ChannelCorrelatorResult] | None,
    tensor_momentum_meta: dict[str, Any] | None,
    noncomp_multiscale_output: MultiscaleStrongForceOutput | None,
    companion_multiscale_output: MultiscaleStrongForceOutput | None,
    status_lines: list[str] | None = None,
    state: dict[str, Any] | None = None,
    force_gevp: bool = False,
) -> dict[str, Any]:
    """Update tensor calibration plus the reusable GEVP diagnostics section."""
    payload = update_tensor_calibration_tab(
        w,
        base_results=base_results,
        strong_tensor_result=strong_tensor_result,
        tensor_momentum_results=tensor_momentum_results,
        tensor_momentum_meta=tensor_momentum_meta,
        noncomp_multiscale_output=noncomp_multiscale_output,
        companion_multiscale_output=companion_multiscale_output,
        status_lines=status_lines,
    )

    per_scale_results = _collect_tensor_per_scale_results(
        noncomp_multiscale_output=noncomp_multiscale_output,
        companion_multiscale_output=companion_multiscale_output,
    )
    original_results = _collect_tensor_original_results(
        base_results=base_results,
        strong_tensor_result=strong_tensor_result,
        tensor_momentum_results=tensor_momentum_results,
    )
    _configure_gevp_controls(
        w,
        per_scale_results=per_scale_results,
        original_results=original_results,
    )

    min_r2, min_windows, max_error_pct, remove_artifacts = _resolve_gevp_filter_values(state)
    selected_families = [str(v) for v in (w.gevp_family_select.value or [])]
    selected_scales = [str(v) for v in (w.gevp_scale_select.value or [])]
    dirty = bool(state.get(GEVP_DIRTY_STATE_KEY, False)) if isinstance(state, dict) else False
    companion_gevp_results = None
    if isinstance(state, Mapping):
        raw_gevp = state.get("companion_strong_force_results")
        if isinstance(raw_gevp, Mapping):
            companion_gevp_results = raw_gevp

    _update_tensor_multiscale_variant_tabs(
        w,
        base_results=base_results,
        strong_tensor_result=strong_tensor_result,
        noncomp_multiscale_output=noncomp_multiscale_output,
        companion_multiscale_output=companion_multiscale_output,
        companion_gevp_results=companion_gevp_results,
    )

    if force_gevp or not dirty:
        gevp_payload = update_gevp_dashboard(
            w.gevp,
            selected_channel_name="tensor",
            raw_channel_name="tensor",
            per_scale_results=per_scale_results,
            original_results=original_results,
            companion_gevp_results=companion_gevp_results,
            min_r2=min_r2,
            min_windows=min_windows,
            max_error_pct=max_error_pct,
            remove_artifacts=remove_artifacts,
            selected_families=selected_families,
            selected_scale_labels=selected_scales,
        )
        if isinstance(state, dict):
            state[GEVP_DIRTY_STATE_KEY] = False
    else:
        clear_gevp_dashboard(w.gevp)
        families_txt = ", ".join(selected_families) if selected_families else "none"
        scales_txt = ", ".join(selected_scales) if selected_scales else "none"
        w.gevp.summary.object = (
            "**Tensor GEVP pending recompute.**  \n"
            f"_Selections:_ families `{families_txt}`, scales `{scales_txt}`.  \n"
            "Click **Recompute GEVP** to refresh diagnostics."
        )
        gevp_payload = None

    payload["gevp_payload"] = gevp_payload
    payload["gevp_selected_families"] = selected_families
    payload["gevp_selected_scales"] = selected_scales
    return payload
