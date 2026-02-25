"""Companion Correlators dashboard tab.

Provides a UI to drive the ``new_channels`` companion-based compute functions,
collect results via per-type extractors, and visualize correlator
curves, effective masses, and operator time series using the shared
:mod:`correlator_plots` utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import pandas as pd
import panel as pn
import param
import torch

from fragile.physics.app.correlator_plots import (
    build_correlator_table,
    build_grouped_correlator_plot,
    build_grouped_meff_plot,
    build_summary_table,
    group_strong_correlator_keys,
)
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.new_channels.baryon_triplet_channels import (
    compute_baryon_correlator_from_color,
)
from fragile.physics.new_channels.fitness_bilinear_channels import (
    compute_fitness_bilinear_from_color,
)
from fragile.physics.new_channels.glueball_color_channels import (
    _extract_axis_bounds as _glueball_extract_axis_bounds,
    compute_glueball_color_correlator_from_color,
)
from fragile.physics.new_channels.mass_extraction_adapter import (
    extract_baryon,
    extract_fitness_bilinear,
    extract_glueball,
    extract_meson_phase,
    extract_vector_meson,
)
from fragile.physics.new_channels.meson_phase_channels import (
    compute_meson_phase_correlator_from_color,
)
from fragile.physics.new_channels.multiscale_strong_force import (
    compute_multiscale_strong_force_channels,
    MultiscaleStrongForceConfig,
)
from fragile.physics.operators.tensor_operators import _build_sigma_matrices
from fragile.physics.new_channels.vector_meson_channels import (
    compute_vector_meson_correlator_from_color_positions,
)
from fragile.physics.operators.pipeline import PipelineResult
from fragile.physics.qft_utils import resolve_3d_dims, resolve_frame_indices
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0_auto


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

SCALAR_MODES = ("standard", "score_directed", "score_weighted", "abs2_vacsub", "dirac")
PSEUDOSCALAR_MODES = (
    "standard",
    "score_weighted",
    "standard_raw",
    "score_weighted_raw",
    "fitness_pseudoscalar",
    "fitness_scalar_variance",
    "fitness_axial",
    "dirac",
    "pion_tensor",
)
# Meson modes whose correlator should use raw (not vacuum-subtracted) C(lag).
_RAW_PSEUDOSCALAR_SUFFIX = "_raw"
FITNESS_PSEUDOSCALAR_MODES = frozenset({
    "fitness_pseudoscalar",
    "fitness_scalar_variance",
    "fitness_axial",
})
_DIRAC_MODE = "dirac"
BARYON_MODES = (
    "det_abs",
    "flux_action",
    "flux_sin2",
    "flux_exp",
    "score_signed",
    "score_abs",
)
VECTOR_MODES = ("standard", "score_directed", "score_gradient", "dirac")
VECTOR_PROJECTIONS = ("full", "longitudinal", "transverse")
GLUEBALL_MODES = (
    "re_plaquette",
    "action_re_plaquette",
    "phase_action",
    "phase_sin2",
)
AXIAL_VECTOR_MODES = ("dirac",)
TENSOR_MODES = ("standard", "dirac")


class CompanionCorrelatorSettings(param.Parameterized):
    """Settings for the companion-channel correlator computations."""

    # -- Common --
    warmup_fraction = param.Number(default=0.3, bounds=(0.0, 0.95))
    end_fraction = param.Number(default=1.0, bounds=(0.05, 1.0))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    mass = param.Number(default=1.0, bounds=(1e-6, None))
    ell0 = param.Number(default=None, bounds=(1e-8, None), allow_None=True)
    ell0_method = param.ObjectSelector(
        default="companion",
        objects=("companion", "geodesic_edges", "euclidean_edges"),
        doc="Automatic ell0 estimation method when ell0 is blank.",
    )
    max_lag = param.Integer(default=40, bounds=(1, 500))
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

    # -- Per-channel operator settings (non-mode) --
    baryon_flux_exp_alpha = param.Number(default=1.0, bounds=(0.0, None))

    vector_unit_displacement = param.Boolean(default=True)

    glueball_momentum_projection = param.Boolean(default=False)
    glueball_momentum_axis = param.Integer(default=0, bounds=(0, 3))
    glueball_momentum_mode_max = param.Integer(default=3, bounds=(0, 8))

    # -- Multiscale --
    n_scales = param.Integer(default=4, bounds=(2, 16))
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
    multiscale_button = pn.widgets.Button(
        name="Run Multiscale",
        button_type="warning",
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
            "ell0_method",
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
    scalar_mode_selector = pn.widgets.MultiSelect(
        name="Scalar",
        options=list(SCALAR_MODES),
        value=list(SCALAR_MODES),
        size=len(SCALAR_MODES),
        sizing_mode="stretch_width",
    )
    pseudoscalar_mode_selector = pn.widgets.MultiSelect(
        name="Pseudoscalar",
        options=list(PSEUDOSCALAR_MODES),
        value=["standard", "score_weighted"],
        size=len(PSEUDOSCALAR_MODES),
        sizing_mode="stretch_width",
    )
    baryon_mode_selector = pn.widgets.MultiSelect(
        name="Baryon",
        options=list(BARYON_MODES),
        value=["det_abs", "flux_action", "flux_sin2", "flux_exp"],
        size=len(BARYON_MODES),
        sizing_mode="stretch_width",
    )
    vector_mode_selector = pn.widgets.MultiSelect(
        name="Vector",
        options=list(VECTOR_MODES),
        value=list(VECTOR_MODES),
        size=len(VECTOR_MODES),
        sizing_mode="stretch_width",
    )
    vector_projection_selector = pn.widgets.MultiSelect(
        name="Vector Projection",
        options=list(VECTOR_PROJECTIONS),
        value=list(VECTOR_PROJECTIONS),
        size=len(VECTOR_PROJECTIONS),
        sizing_mode="stretch_width",
    )
    glueball_mode_selector = pn.widgets.MultiSelect(
        name="Glueball",
        options=list(GLUEBALL_MODES),
        value=list(GLUEBALL_MODES),
        size=len(GLUEBALL_MODES),
        sizing_mode="stretch_width",
    )
    axial_vector_mode_selector = pn.widgets.MultiSelect(
        name="Axial Vector",
        options=list(AXIAL_VECTOR_MODES),
        value=[],
        size=max(len(AXIAL_VECTOR_MODES), 2),
        sizing_mode="stretch_width",
    )
    tensor_mode_selector = pn.widgets.MultiSelect(
        name="Tensor",
        options=list(TENSOR_MODES),
        value=[],
        size=max(len(TENSOR_MODES), 2),
        sizing_mode="stretch_width",
    )

    channel_mode_selectors: dict[str, pn.widgets.MultiSelect] = {
        "scalar": scalar_mode_selector,
        "pseudoscalar": pseudoscalar_mode_selector,
        "baryon": baryon_mode_selector,
        "vector": vector_mode_selector,
        "glueball": glueball_mode_selector,
        "axial_vector": axial_vector_mode_selector,
        "tensor": tensor_mode_selector,
    }

    channel_selection_panel = pn.Column(
        pn.Row(
            scalar_mode_selector,
            pseudoscalar_mode_selector,
            baryon_mode_selector,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            vector_mode_selector,
            vector_projection_selector,
            glueball_mode_selector,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            axial_vector_mode_selector,
            tensor_mode_selector,
            sizing_mode="stretch_width",
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
        ],
        show_name=False,
        default_layout=type("OperatorGrid", (pn.GridBox,), {"ncols": 2}),
    )

    multiscale_settings_panel = pn.Param(
        settings,
        parameters=["n_scales", "kernel_type", "kernel_batch_size", "edge_weight_mode"],
        show_name=False,
        default_layout=type("MultiscaleGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Refresh plots helper --

    def _refresh_overlay():
        """Refresh summary tables and per-channel-group plots."""
        result = state.get("companion_correlator_output")
        if result is None:
            return
        si = int(scale_selector.value)
        ls = bool(log_scale_toggle.value)
        summary_table.value = build_summary_table(result, scale_index=si)
        correlator_table.value = build_correlator_table(result, scale_index=si)

        # Per-channel group plots
        groups = group_strong_correlator_keys(result.correlators.keys())
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

    log_scale_toggle.param.watch(lambda _: _refresh_overlay(), "value")
    scale_selector.param.watch(lambda _: _refresh_overlay(), "value")

    # -- Compute callback --

    def on_run(_):
        def _compute(history: RunHistory):
            d = history.d
            color_dims = _parse_color_dims(settings.color_dims_spec, d)
            use_connected = bool(settings.use_connected)
            max_lag = int(settings.max_lag)
            eps = float(settings.eps)

            # Collect selected modes per channel from MultiSelect widgets
            selected_modes: dict[str, list[str]] = {}
            for family, selector in channel_mode_selectors.items():
                modes = [str(v) for v in selector.value]
                if modes:
                    selected_modes[family] = modes

            if not selected_modes:
                status.object = "**Error:** No channels selected."
                return

            # ---- Shared data prep (computed ONCE) ----
            frame_indices = resolve_frame_indices(
                history=history,
                warmup_fraction=float(settings.warmup_fraction),
                end_fraction=float(settings.end_fraction),
            )
            if not frame_indices:
                status.object = "**Error:** No valid frames after warmup/end filtering."
                return

            start_idx = frame_indices[0]
            end_idx = frame_indices[-1] + 1
            h_eff = float(max(settings.h_eff, 1e-8))
            mass_val = float(max(settings.mass, 1e-8))
            ell0_val = (
                float(settings.ell0)
                if settings.ell0 is not None
                else float(estimate_ell0_auto(history, settings.ell0_method))
            )
            if ell0_val <= 0:
                status.object = "**Error:** ell0 must be positive."
                return

            # Compute color states ONCE
            color, color_valid = compute_color_states_batch(
                history=history,
                start_idx=start_idx,
                h_eff=h_eff,
                mass=mass_val,
                ell0=ell0_val,
                end_idx=end_idx,
            )
            if color_dims is not None:
                dims = resolve_3d_dims(color.shape[-1], color_dims, "color_dims")
                color = color[:, :, list(dims)]

            device = color.device
            color_valid = color_valid.to(dtype=torch.bool, device=device)

            # Gather companions ONCE
            companions_distance = torch.as_tensor(
                history.companions_distance[start_idx - 1 : end_idx - 1],
                dtype=torch.long,
                device=device,
            )
            companions_clone = torch.as_tensor(
                history.companions_clone[start_idx - 1 : end_idx - 1],
                dtype=torch.long,
                device=device,
            )

            # Scores (needed by meson, baryon, vector score modes)
            scores = torch.as_tensor(
                history.cloning_scores[start_idx - 1 : end_idx - 1],
                dtype=torch.float32,
                device=device,
            )

            # Positions (needed by vector, glueball momentum)
            needs_positions = (
                "vector" in selected_modes
                or ("glueball" in selected_modes and bool(settings.glueball_momentum_projection))
            )
            positions = None
            if needs_positions:
                positions = torch.as_tensor(
                    history.x_before_clone[start_idx:end_idx],
                    device=device,
                    dtype=torch.float32,
                )

            merged_correlators: dict[str, Any] = {}
            merged_operators: dict[str, Any] = {}
            channel_labels: list[str] = []

            # -- Meson output cache (keyed by operator_mode) --
            meson_cache: dict[str, Any] = {}

            def _get_meson_output(operator_mode: str):
                if operator_mode not in meson_cache:
                    meson_cache[operator_mode] = compute_meson_phase_correlator_from_color(
                        color=color,
                        color_valid=color_valid,
                        companions_distance=companions_distance,
                        companions_clone=companions_clone,
                        max_lag=max_lag,
                        use_connected=use_connected,
                        pair_selection=str(settings.pair_selection),
                        eps=eps,
                        operator_mode=operator_mode,
                        scores=scores,
                        frame_indices=frame_indices,
                    )
                return meson_cache[operator_mode]

            # -- Lazy Dirac computation (shared across channels) --
            _dirac_result = None

            def _get_dirac_result():
                nonlocal _dirac_result
                if _dirac_result is None:
                    from fragile.physics.new_channels.dirac_spinors import (
                        compute_dirac_operator_series,
                    )

                    alive_mask = torch.as_tensor(
                        history.alive_mask[start_idx - 1 : end_idx - 1],
                        dtype=torch.bool,
                        device=device,
                    )
                    sample_indices = torch.arange(
                        color.shape[1], device=device,
                    ).unsqueeze(0).expand(color.shape[0], -1)
                    neighbor_indices = companions_distance.unsqueeze(-1)

                    _dirac_result = compute_dirac_operator_series(
                        color=color,
                        color_valid=color_valid,
                        sample_indices=sample_indices,
                        neighbor_indices=neighbor_indices,
                        alive=alive_mask,
                    )
                return _dirac_result

            def _add_dirac_channel(field_name: str, key: str):
                """Extract one Dirac bilinear field and compute its FFT correlator."""
                if color.shape[-1] != 3:
                    return
                from fragile.physics.qft_utils import _fft_correlator_batched

                dr = _get_dirac_result()
                op_series = getattr(dr, field_name)  # [T]
                merged_operators[key] = op_series
                corr = _fft_correlator_batched(
                    op_series.unsqueeze(0),  # [1, T]
                    max_lag=max_lag,
                    use_connected=use_connected,
                )
                merged_correlators[key] = corr.squeeze(0)  # [max_lag+1]

            # -- Scalar: iterate over selected scalar modes --
            for mode in selected_modes.get("scalar", []):
                if mode == _DIRAC_MODE:
                    _add_dirac_channel("scalar", "scalar_dirac")
                    continue
                out = _get_meson_output(mode)
                corrs, ops = extract_meson_phase(
                    out,
                    use_connected=use_connected,
                    prefix="",
                )
                for key, val in corrs.items():
                    if key.startswith("scalar"):
                        merged_correlators[f"{key}_{mode}"] = val
                for key, val in ops.items():
                    if key.startswith("scalar"):
                        merged_operators[f"{key}_{mode}"] = val
            if "scalar" in selected_modes:
                channel_labels.append("scalar")

            # -- Pseudoscalar: meson-based + fitness + dirac modes --
            ps_modes = selected_modes.get("pseudoscalar", [])
            meson_ps_modes = [
                m for m in ps_modes
                if m not in FITNESS_PSEUDOSCALAR_MODES and m != _DIRAC_MODE
                and m != "pion_tensor"
            ]
            fitness_ps_modes = [m for m in ps_modes if m in FITNESS_PSEUDOSCALAR_MODES]
            if _DIRAC_MODE in ps_modes:
                _add_dirac_channel("pseudoscalar", "pseudoscalar_dirac")
            if "pion_tensor" in ps_modes:
                _add_dirac_channel("tensor_0k", "pseudoscalar_pion_tensor")

            for mode in meson_ps_modes:
                is_raw = mode.endswith(_RAW_PSEUDOSCALAR_SUFFIX)
                base_mode = mode[: -len(_RAW_PSEUDOSCALAR_SUFFIX)] if is_raw else mode
                out = _get_meson_output(base_mode)
                corrs, ops = extract_meson_phase(
                    out,
                    use_connected=not is_raw and use_connected,
                    prefix="",
                )
                for key, val in corrs.items():
                    if key.startswith("pseudoscalar"):
                        merged_correlators[f"{key}_{mode}"] = val
                for key, val in ops.items():
                    if key.startswith("pseudoscalar"):
                        merged_operators[f"{key}_{mode}"] = val

            if fitness_ps_modes:
                # Gather fitness and alive_mask tensors (same slice as scores)
                fitness_tensor = torch.as_tensor(
                    history.fitness[start_idx - 1 : end_idx - 1],
                    dtype=torch.float32,
                    device=device,
                )
                alive_mask = torch.as_tensor(
                    history.alive_mask[start_idx - 1 : end_idx - 1],
                    dtype=torch.bool,
                    device=device,
                )
                fitness_output = compute_fitness_bilinear_from_color(
                    color=color,
                    color_valid=color_valid,
                    companions_distance=companions_distance,
                    companions_clone=companions_clone,
                    fitness=fitness_tensor,
                    cloning_scores=scores,
                    alive_mask=alive_mask,
                    max_lag=max_lag,
                    use_connected=use_connected,
                    pair_selection=str(settings.pair_selection),
                    eps=eps,
                    frame_indices=frame_indices,
                )
                all_fitness_corrs, all_fitness_ops = extract_fitness_bilinear(
                    fitness_output,
                    use_connected=use_connected,
                )
                for mode in fitness_ps_modes:
                    if mode in all_fitness_corrs:
                        merged_correlators[f"pseudoscalar_{mode}"] = all_fitness_corrs[mode]
                    if mode in all_fitness_ops:
                        merged_operators[f"pseudoscalar_{mode}"] = all_fitness_ops[mode]

            if ps_modes:
                channel_labels.append("pseudoscalar")

            # -- Baryon: iterate over selected modes --
            for mode in selected_modes.get("baryon", []):
                out = compute_baryon_correlator_from_color(
                    color=color,
                    color_valid=color_valid,
                    companions_distance=companions_distance,
                    companions_clone=companions_clone,
                    max_lag=max_lag,
                    use_connected=use_connected,
                    eps=eps,
                    operator_mode=mode,
                    flux_exp_alpha=float(settings.baryon_flux_exp_alpha),
                    scores=scores,
                    frame_indices=frame_indices,
                )
                corrs, ops = extract_baryon(
                    out,
                    use_connected=use_connected,
                    prefix="",
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
                if mode == _DIRAC_MODE:
                    _add_dirac_channel("vector", "vector_dirac")
                    continue
                for proj in selected_projections:
                    out = compute_vector_meson_correlator_from_color_positions(
                        color=color,
                        color_valid=color_valid,
                        positions=positions,
                        companions_distance=companions_distance,
                        companions_clone=companions_clone,
                        max_lag=max_lag,
                        use_connected=use_connected,
                        pair_selection=str(settings.pair_selection),
                        eps=eps,
                        use_unit_displacement=bool(settings.vector_unit_displacement),
                        operator_mode=mode,
                        projection_mode=proj,
                        scores=scores,
                        frame_indices=frame_indices,
                    )
                    corrs, ops = extract_vector_meson(
                        out,
                        use_connected=use_connected,
                        prefix="",
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
                use_momentum = bool(settings.glueball_momentum_projection)
                momentum_axis = int(settings.glueball_momentum_axis)
                positions_axis = None
                projection_length = None
                if use_momentum:
                    positions_axis = torch.as_tensor(
                        history.x_before_clone[start_idx:end_idx, :, momentum_axis],
                        device=device,
                        dtype=torch.float32,
                    )
                    low, high = _glueball_extract_axis_bounds(
                        history.bounds,
                        momentum_axis,
                        device=device,
                    )
                    if (
                        low is not None
                        and high is not None
                        and math.isfinite(low)
                        and math.isfinite(high)
                        and high > low
                    ):
                        projection_length = float(high - low)

                out = compute_glueball_color_correlator_from_color(
                    color=color,
                    color_valid=color_valid,
                    companions_distance=companions_distance,
                    companions_clone=companions_clone,
                    max_lag=max_lag,
                    use_connected=use_connected,
                    eps=eps,
                    operator_mode=mode,
                    frame_indices=frame_indices,
                    positions_axis=positions_axis,
                    momentum_axis=momentum_axis if use_momentum else None,
                    use_momentum_projection=use_momentum,
                    momentum_mode_max=int(settings.glueball_momentum_mode_max),
                    projection_length=projection_length,
                )
                corrs, ops = extract_glueball(
                    out,
                    use_connected=use_connected,
                    prefix="",
                )
                for key, val in corrs.items():
                    merged_correlators[f"{key}_{mode}"] = val
                for key, val in ops.items():
                    merged_operators[f"{key}_{mode}"] = val
            if "glueball" in selected_modes:
                channel_labels.append("glueball")

            # -- Axial Vector: only dirac mode --
            for mode in selected_modes.get("axial_vector", []):
                if mode == _DIRAC_MODE:
                    _add_dirac_channel("axial_vector", "axial_vector_dirac")
            if "axial_vector" in selected_modes:
                channel_labels.append("axial_vector")

            # -- Tensor: iterate over selected modes --
            for mode in selected_modes.get("tensor", []):
                if mode == _DIRAC_MODE:
                    _add_dirac_channel("tensor", "tensor_dirac")
                    continue
                # mode == "standard": bilinear sigma_{mu,nu} tensor
                from fragile.physics.qft_utils import _fft_correlator_batched
                from fragile.physics.qft_utils.companions import build_companion_pair_indices
                from fragile.physics.qft_utils.helpers import (
                    safe_gather_pairs_2d,
                    safe_gather_pairs_3d,
                )

                d = color.shape[-1]
                sigma = _build_sigma_matrices(d, device).to(dtype=color.dtype)
                pair_indices, structural_valid = build_companion_pair_indices(
                    companions_distance=companions_distance,
                    companions_clone=companions_clone,
                    pair_selection=str(settings.pair_selection),
                )
                color_j, in_range = safe_gather_pairs_3d(color, pair_indices)
                valid_j, _ = safe_gather_pairs_2d(color_valid, pair_indices)
                color_i = color.unsqueeze(2).expand_as(color_j)
                finite_i = torch.isfinite(color_i.real) & torch.isfinite(color_i.imag)
                finite_j = torch.isfinite(color_j.real) & torch.isfinite(color_j.imag)
                valid = (
                    structural_valid & in_range
                    & color_valid.unsqueeze(-1) & valid_j
                    & finite_i.all(dim=-1) & finite_j.all(dim=-1)
                )
                if sigma.shape[0] > 0:
                    result_t = torch.einsum(
                        "...i,pij,...j->...p", color_i.conj(), sigma, color_j,
                    )
                    op_tensor = result_t.mean(dim=-1).imag.float()
                else:
                    op_tensor = torch.zeros(color_i.shape[:-1], device=device)
                op_tensor = torch.where(valid, op_tensor, torch.zeros_like(op_tensor))
                # Per-frame average
                weights = valid.to(op_tensor.dtype)
                counts = valid.sum(dim=(1, 2)).to(torch.int64)
                sums = (op_tensor * weights).sum(dim=(1, 2))
                tensor_series = torch.zeros(
                    op_tensor.shape[0], device=device, dtype=torch.float32,
                )
                valid_t = counts > 0
                if torch.any(valid_t):
                    tensor_series[valid_t] = (
                        sums[valid_t] / counts[valid_t].to(op_tensor.dtype)
                    ).float()
                key = "tensor_standard"
                merged_operators[key] = tensor_series
                corr = _fft_correlator_batched(
                    tensor_series.unsqueeze(0),
                    max_lag=max_lag,
                    use_connected=use_connected,
                )
                merged_correlators[key] = corr.squeeze(0)
            if "tensor" in selected_modes:
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

            _refresh_overlay()

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

    # -- Multiscale compute callback --

    def on_multiscale_run(_):
        def _compute(history: RunHistory):
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
                fit_mode="aic",
                companion_baryon_flux_exp_alpha=float(settings.baryon_flux_exp_alpha),
            )

            ms_output = compute_multiscale_strong_force_channels(
                history,
                config=ms_config,
            )

            # Convert MultiscaleStrongForceOutput → PipelineResult
            merged_correlators: dict[str, torch.Tensor] = {}
            merged_operators: dict[str, torch.Tensor] = {}
            for ch_name, scale_results in ms_output.per_scale_results.items():
                corrs = torch.stack([r.correlator for r in scale_results])  # [S, max_lag+1]
                merged_correlators[ch_name] = corrs
                if ch_name in ms_output.series_by_channel:
                    merged_operators[ch_name] = ms_output.series_by_channel[ch_name]  # [S, T]

            result = PipelineResult(
                correlators=merged_correlators,
                operators=merged_operators,
                scales=ms_output.scales,
            )
            state["companion_correlator_output"] = result

            # Activate scale selector
            n_s = int(ms_output.scales.numel())
            scale_selector.start = 0
            scale_selector.end = max(0, n_s - 1)
            scale_selector.value = 0
            scale_selector.visible = n_s > 1

            _refresh_overlay()

            n_ch = len(ms_output.best_results)
            status.object = (
                f"**Multiscale complete:** {n_ch} channels across "
                f"{n_s} scales ({settings.kernel_type} kernel)."
            )

        run_tab_computation(state, status, "multiscale analysis", _compute)

    multiscale_button.on_click(on_multiscale_run)

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
        pn.Row(run_button, multiscale_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Common Settings", common_settings_panel),
            ("Channel & Mode Selection", channel_selection_panel),
            ("Operator Settings", operator_config_panel),
            ("Multiscale Settings", multiscale_settings_panel),
            active=[0, 1, 2, 3],
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Correlator Summary"),
        summary_table,
        pn.Row(log_scale_toggle, scale_selector, sizing_mode="stretch_width"),
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
        multiscale_button.disabled = False
        status.object = "**Companion Correlators ready:** click Compute Companion Correlators."
        if defer:
            return
        state["companion_correlator_output"] = None
        summary_table.value = pd.DataFrame()
        correlator_table.value = pd.DataFrame()
        per_channel_container.clear()
        scale_selector.visible = False

    return CompanionCorrelatorSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
    )
