"""Electroweak dashboard tab building blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math as _math
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.physics.app.qft.correlator_channels import ChannelCorrelatorResult
from fragile.physics.app.qft.dashboard.gevp_dashboard import (
    build_gevp_dashboard_sections,
    clear_gevp_dashboard,
    create_gevp_dashboard_widgets,
    update_gevp_dashboard,
)
from fragile.physics.app.qft.dashboard.gevp_mass_dashboard import (
    build_gevp_mass_spectrum_sections,
    clear_gevp_mass_spectrum,
    create_gevp_mass_spectrum_widgets,
    update_gevp_mass_spectrum,
)
from fragile.physics.app.qft.electroweak_channels import (
    compute_electroweak_channels,
    ELECTROWEAK_CHANNELS,
    ELECTROWEAK_PARITY_CHANNELS,
    ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
    ElectroweakChannelConfig,
)
from fragile.physics.app.qft.gevp_channels import (
    compute_companion_channel_gevp,
    get_companion_gevp_basis_channels,
    GEVPConfig,
)
from fragile.physics.app.qft.multiscale_electroweak import (
    compute_multiscale_electroweak_channels,
    EW_MIXED_BASE_CHANNELS,
    MultiscaleElectroweakConfig,
    MultiscaleElectroweakOutput,
    SU2_BASE_CHANNELS,
    SU2_DIRECTIONAL_CHANNELS,
    SU2_WALKER_TYPE_CHANNELS,
    U1_BASE_CHANNELS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "u1": (
        "u1_phase",
        "u1_dressed",
        "u1_phase_q2",
        "u1_dressed_q2",
    ),
    "su2": SU2_BASE_CHANNELS,
    "su2_directional": SU2_DIRECTIONAL_CHANNELS,
    "su2_walker_type": SU2_WALKER_TYPE_CHANNELS,
    "ew_mixed": ("ew_mixed",),
    "symmetry_breaking": ELECTROWEAK_SYMMETRY_BREAKING_CHANNELS,
    "parity_velocity": ELECTROWEAK_PARITY_CHANNELS,
}

DEFAULT_ELECTROWEAK_CHANNEL_VARIANT_SELECTION: dict[str, tuple[str, ...]] = {
    "u1": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["u1"],
    "su2": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["su2"],
    "su2_directional": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["su2_directional"],
    "su2_walker_type": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["su2_walker_type"],
    "ew_mixed": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["ew_mixed"],
    "symmetry_breaking": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["symmetry_breaking"],
    "parity_velocity": ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY["parity_velocity"],
}

ELECTROWEAK_CANONICAL_CHANNEL_ORDER = (
    "u1_phase",
    "u1_dressed",
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
    "ew_mixed",
    "fitness_phase",
    "clone_indicator",
)

DEFAULT_ELECTROWEAK_CHANNELS_FOR_DIRAC = list(ELECTROWEAK_CHANNELS)

# Default electroweak mass references (GeV) mapped to proxy channels.
DEFAULT_ELECTROWEAK_REFS = {
    "u1_phase": 0.000511,  # electron
    "u1_dressed": 0.105658,  # muon
    "su2_phase": 80.379,  # W boson
    "su2_doublet": 91.1876,  # Z boson
    "ew_mixed": 1.77686,  # tau
    "fitness_phase": 91.1876,  # Z boson (massive mode of fitness phase)
    "clone_indicator": 80.379,  # W boson (cloning decay rate)
}

ELECTROWEAK_COUPLING_NAMES = (
    "ε_distance emergent",
    "ε_clone emergent",
    "ε_fitness_gap emergent",
    "ε_geodesic emergent",
    "g1_est emergent",
    "g2_est emergent",
    "g2_est emergent fitness-gap",
    "sin²θ_W emergent",
    "tanθ_W emergent",
    "g1_proxy",
    "g2_proxy",
    "sin2θw proxy",
    "tanθw proxy",
)

ELECTROWEAK_COUPLING_REFERENCE_COLUMNS = ("observed_mZ", "observed_GUT")

DEFAULT_ELECTROWEAK_COUPLING_REFS = {
    "g1_est emergent": {"observed_mZ": 0.357468, "observed_GUT": 0.560499},
    "g2_est emergent": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "g2_est emergent fitness-gap": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "sin²θ_W emergent": {"observed_mZ": 0.23129, "observed_GUT": 0.375},
    "tanθ_W emergent": {"observed_mZ": 0.548526, "observed_GUT": 0.774597},
    "g1_proxy": {"observed_mZ": 0.357468, "observed_GUT": 0.560499},
    "g2_proxy": {"observed_mZ": 0.651689, "observed_GUT": 0.723601},
    "sin2θw proxy": {"observed_mZ": 0.23129, "observed_GUT": 0.375},
    "tanθw proxy": {"observed_mZ": 0.548526, "observed_GUT": 0.774597},
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class NewDiracElectroweakSettings(param.Parameterized):
    """Settings for the unified Dirac/Electroweak analysis tab."""

    simulation_range = param.Range(
        default=(0.2, 1.0),
        bounds=(0.0, 1.0),
        doc="Fraction of simulation timeline to use (start, end).",
    )
    max_lag = param.Integer(default=80, bounds=(10, 200))
    h_eff = param.Number(default=1.0, bounds=(1e-6, None))
    use_connected = param.Boolean(default=True)
    edge_weight_mode = param.ObjectSelector(
        default="riemannian_kernel_volume",
        objects=[
            "uniform",
            "inverse_riemannian_distance",
            "inverse_riemannian_volume",
            "riemannian_kernel",
            "riemannian_kernel_volume",
            "inverse_distance",
            "inverse_volume",
            "kernel",
        ],
    )
    neighbor_k = param.Integer(default=0, bounds=(0, 50))
    window_widths_spec = param.String(default="5-50")
    fit_mode = param.ObjectSelector(default="aic", objects=("aic", "linear", "linear_abs"))
    fit_start = param.Integer(default=2, bounds=(0, None))
    fit_stop = param.Integer(default=None, bounds=(1, None), allow_None=True)
    min_fit_points = param.Integer(default=2, bounds=(2, None))
    su2_operator_mode = param.ObjectSelector(
        default="standard",
        objects=("standard", "score_directed"),
        doc="SU(2) operator branch (standard or score-directed).",
    )
    enable_walker_type_split = param.Boolean(
        default=False,
        doc="Emit cloner/resister/persister split SU(2) channels.",
    )
    walker_type_scope = param.ObjectSelector(
        default="frame_global",
        objects=("frame_global",),
        doc="Walker-type classification scope.",
    )
    compute_bootstrap_errors = param.Boolean(default=False)
    n_bootstrap = param.Integer(default=100, bounds=(10, 1000))
    use_multiscale_kernels = param.Boolean(
        default=True,
        doc=(
            "Compute multiscale SU(2) channels using recorded kernel distances and "
            "promote best-scale estimators."
        ),
    )
    n_scales = param.Integer(default=8, bounds=(2, 32))
    kernel_type = param.ObjectSelector(
        default="gaussian",
        objects=("gaussian", "exponential", "tophat", "shell"),
    )
    kernel_distance_method = param.ObjectSelector(
        default="auto",
        objects=("auto", "floyd-warshall", "tropical"),
    )
    kernel_assume_all_alive = param.Boolean(default=True)
    kernel_batch_size = param.Integer(default=1, bounds=(1, 64))
    kernel_scale_frames = param.Integer(default=8, bounds=(1, 128))
    kernel_scale_q_low = param.Number(default=0.05, bounds=(0.0, 0.95))
    kernel_scale_q_high = param.Number(default=0.95, bounds=(0.05, 1.0))
    kernel_bootstrap_mode = param.ObjectSelector(
        default="time",
        objects=("time", "walker", "hybrid"),
        doc="Bootstrap strategy for multiscale SU(2) channels.",
    )

    use_su2_gevp = param.Boolean(
        default=True,
        doc="Compute GEVP-combined SU(2) channel from SU(2) operator family.",
    )
    use_u1_gevp = param.Boolean(default=True, doc="Compute GEVP-combined U(1) channel.")
    use_ew_mixed_gevp = param.Boolean(default=True, doc="Compute GEVP-combined EW mixed channel.")
    gevp_basis_strategy = param.ObjectSelector(
        default="base_plus_best_scale",
        objects=("base_only", "base_plus_best_scale"),
    )
    gevp_t0 = param.Integer(default=2, bounds=(1, None))
    gevp_max_basis = param.Integer(default=10, bounds=(1, 64))
    gevp_min_operator_r2 = param.Number(
        default=0.5,
        bounds=(-1.0, 1.0),
        doc="Minimum operator fit R² required for GEVP basis inclusion.",
    )
    gevp_min_operator_windows = param.Integer(
        default=5,
        bounds=(0, None),
        doc="Minimum valid fit-window count required for GEVP basis inclusion.",
    )
    gevp_max_operator_error_pct = param.Number(
        default=60.0,
        bounds=(0.0, None),
        doc="Maximum allowed operator mass error percentage for GEVP basis inclusion.",
    )
    gevp_remove_artifacts = param.Boolean(
        default=True,
        doc=(
            "Remove artifact operators (mass_error==0, mass_error NaN/Inf, or mass==0) "
            "from GEVP basis and displayed tables."
        ),
    )
    gevp_eig_rel_cutoff = param.Number(default=1e-2, bounds=(1e-8, 1.0))
    gevp_cond_limit = param.Number(default=1e4, bounds=(1.0, None))
    gevp_shrinkage = param.Number(default=1e-6, bounds=(0.0, 1.0))
    gevp_bootstrap_mode = param.ObjectSelector(
        default="time",
        objects=("time", "walker", "hybrid"),
    )
    gevp_compute_multimode = param.Boolean(
        default=False,
        doc="Extract multi-mode GEVP mass spectrum (ground + excited states).",
    )
    gevp_tau_diag = param.Integer(
        default=None,
        bounds=(2, None),
        allow_None=True,
        doc="Fixed τ_diag for multi-mode GEVP (None = auto-select).",
    )
    gevp_plateau_min_length = param.Integer(
        default=3,
        bounds=(2, None),
        doc="Minimum plateau length for multi-mode mass extraction.",
    )
    gevp_plateau_max_slope = param.Number(
        default=0.3,
        bounds=(0.0, None),
        doc="Maximum relative slope for plateau acceptance.",
    )
    gevp_t0_sweep_spec = param.String(
        default="",
        doc="t0 sweep: '2-6' or '2,3,5'. Empty=disabled.",
    )
    gevp_multimode_fit_start = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Exp fit start lag (None=auto).",
    )
    gevp_multimode_fit_stop = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Exp fit stop lag (None=all).",
    )

    mc_time_index = param.Integer(
        default=None,
        bounds=(1, None),
        allow_None=True,
        doc="Recorded MC index/step for snapshot analyses (blank=last).",
    )
    dirac_kernel_mode = param.ObjectSelector(
        default="phase_space",
        objects=("phase_space", "fitness_ratio"),
        doc="Dirac kernel used for spectral analysis.",
    )
    dirac_time_average = param.Boolean(default=False)
    dirac_warmup_fraction = param.Number(default=0.1, bounds=(0.0, 0.5))
    dirac_max_avg_frames = param.Integer(default=80, bounds=(10, 400))
    dirac_color_threshold_mode = param.ObjectSelector(
        default="median",
        objects=("median", "manual"),
    )
    dirac_color_threshold_value = param.Number(default=1.0, bounds=(0.0, None))
    color_singlet_quantile = param.Number(default=0.9, bounds=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_electroweak_best_fit_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    r2s: dict[str, float] | None = None,
    *,
    build_best_fit_rows_generic: Callable,
) -> list[dict[str, Any]]:
    return build_best_fit_rows_generic(masses, [("electroweak", refs)], r2s)


def _build_electroweak_anchor_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    r2s: dict[str, float] | None = None,
    *,
    build_anchor_rows_generic: Callable,
) -> list[dict[str, Any]]:
    anchors = [(f"{name}->{mass:.6f}", mass, name) for name, mass in refs.items()]
    return build_anchor_rows_generic(masses, anchors, r2s)


def _build_electroweak_comparison_rows(
    masses: dict[str, float],
    refs: dict[str, float],
    *,
    best_fit_scale: Callable,
) -> list[dict[str, Any]]:
    anchors = [(f"{name}->{mass:.6f}", mass, name) for name, mass in refs.items()]
    scale = best_fit_scale(masses, anchors)
    if scale is None:
        return []

    rows: list[dict[str, Any]] = []
    for name, alg_mass in masses.items():
        if alg_mass <= 0:
            continue
        obs = refs.get(name)
        pred = alg_mass * scale
        err_pct = None
        if obs is not None and obs > 0:
            err_pct = (pred - obs) / obs * 100.0
        rows.append({
            "channel": name,
            "alg_mass": alg_mass,
            "obs_mass_GeV": obs,
            "pred_mass_GeV": pred,
            "error_pct": err_pct,
        })
    return rows


def _build_electroweak_ratio_rows(
    masses: dict[str, float],
    base_name: str,
    refs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    refs = refs or {}
    rows: list[dict[str, Any]] = []
    base_mass = masses.get(base_name)
    if base_mass is None or base_mass <= 0:
        return rows
    for name, mass in masses.items():
        if name == base_name or mass <= 0:
            continue
        measured = mass / base_mass
        observed = None
        error_pct = None
        obs_num = refs.get(name)
        obs_den = refs.get(base_name)
        if obs_num is not None and obs_den is not None and obs_den > 0:
            observed = obs_num / obs_den
            if observed > 0:
                error_pct = (measured - observed) / observed * 100.0
        rows.append({
            "ratio": f"{name}/{base_name}",
            "measured": measured,
            "observed": observed,
            "error_pct": error_pct,
        })
    return rows


# ---------------------------------------------------------------------------
# Dataclass returned by builder
# ---------------------------------------------------------------------------


@dataclass
class ElectroweakSection:
    """Container for the electroweak dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    settings: NewDiracElectroweakSettings
    on_history_changed: Callable[[bool], None]
    # Widgets needed externally by set_history() for clearing:
    summary: pn.pane.Markdown
    multiscale_summary: pn.pane.Markdown
    su2_gevp_widgets: dict
    u1_gevp_widgets: dict
    ew_mixed_gevp_widgets: dict
    su2_mass_spectrum_widgets: dict
    u1_mass_spectrum_widgets: dict
    ew_mixed_mass_spectrum_widgets: dict
    filtered_summary: pn.pane.Markdown
    mass_table: pn.widgets.Tabulator
    filtered_mass_table: pn.widgets.Tabulator
    ratio_table: pn.widgets.Tabulator
    fit_table: pn.widgets.Tabulator
    compare_table: pn.widgets.Tabulator
    anchor_table: pn.widgets.Tabulator
    multiscale_table: pn.widgets.Tabulator
    multiscale_per_scale_table: pn.widgets.Tabulator


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_electroweak_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable,
    # Shared helpers from dashboard (passed as dependencies):
    update_correlator_plots: Callable,
    update_mass_table: Callable,
    format_ratios: Callable,
    extract_masses: Callable,
    extract_r2: Callable,
    build_best_fit_rows_generic: Callable,
    build_anchor_rows_generic: Callable,
    best_fit_scale: Callable,
    get_channel_mass: Callable,
    get_channel_mass_error: Callable,
    get_channel_r2: Callable,
    split_results_by_companion_gevp_filters: Callable,
    companion_gevp_filter_reason: Callable,
    compute_coupling_constants_fn: Callable,
    build_coupling_rows: Callable,
    extract_coupling_refs: Callable,
    collect_multiselect_values: Callable,
    parse_window_widths: Callable,
    parse_t0_sweep_spec: Callable,
    format_ref_value: Callable,
    extract_n_windows_for_filter: Callable,
) -> ElectroweakSection:
    """Build the Electroweak tab with all widgets, callbacks, and layout."""

    # =====================================================================
    # Widget creation
    # =====================================================================

    new_dirac_ew_settings = NewDiracElectroweakSettings()
    new_dirac_ew_status = pn.pane.Markdown(
        "**Electroweak:** Load a RunHistory and click Compute.",
        sizing_mode="stretch_width",
    )
    new_dirac_ew_run_button = pn.widgets.Button(
        name="Compute Electroweak",
        button_type="primary",
        min_width=260,
        sizing_mode="stretch_width",
        disabled=True,
    )
    new_dirac_ew_summary = pn.pane.Markdown(
        "## Electroweak Summary\n_Run analysis to populate._",
        sizing_mode="stretch_width",
    )
    new_dirac_ew_settings_panel = pn.Param(
        new_dirac_ew_settings,
        parameters=[
            "simulation_range",
            "max_lag",
            "h_eff",
            "mc_time_index",
            "use_connected",
            "edge_weight_mode",
            "neighbor_k",
            "window_widths_spec",
            "fit_mode",
            "fit_start",
            "fit_stop",
            "min_fit_points",
            "su2_operator_mode",
            "enable_walker_type_split",
            "walker_type_scope",
            "compute_bootstrap_errors",
            "n_bootstrap",
            "use_multiscale_kernels",
            "n_scales",
            "kernel_type",
            "kernel_distance_method",
            "kernel_assume_all_alive",
            "kernel_batch_size",
            "kernel_scale_frames",
            "kernel_scale_q_low",
            "kernel_scale_q_high",
            "kernel_bootstrap_mode",
            "use_su2_gevp",
            "gevp_basis_strategy",
            "gevp_t0",
            "gevp_max_basis",
            "gevp_min_operator_r2",
            "gevp_min_operator_windows",
            "gevp_max_operator_error_pct",
            "gevp_remove_artifacts",
            "gevp_eig_rel_cutoff",
            "gevp_cond_limit",
            "gevp_shrinkage",
            "gevp_bootstrap_mode",
        ],
        show_name=False,
        widgets={
            "mc_time_index": {"name": "MC time slice (step or idx; blank=last)"},
            "kernel_type": {"name": "Kernel shape"},
            "kernel_distance_method": {"name": "Distance solver"},
            "kernel_assume_all_alive": {"name": "Assume all walkers alive"},
            "kernel_batch_size": {"name": "Kernel frame batch size"},
            "kernel_scale_frames": {"name": "Scale calibration frames"},
            "kernel_scale_q_low": {"name": "Scale quantile low"},
            "kernel_scale_q_high": {"name": "Scale quantile high"},
            "kernel_bootstrap_mode": {"name": "Kernel bootstrap mode"},
            "su2_operator_mode": {"name": "SU(2) operator mode"},
            "enable_walker_type_split": {"name": "Enable walker-type split"},
            "walker_type_scope": {"name": "Walker-type scope"},
            "use_su2_gevp": {"name": "Enable SU(2) GEVP"},
            "gevp_basis_strategy": {"name": "GEVP basis strategy"},
            "gevp_t0": {"name": "GEVP t0"},
            "gevp_max_basis": {"name": "GEVP max basis"},
            "gevp_min_operator_r2": {"name": "GEVP min operator R²"},
            "gevp_min_operator_windows": {"name": "GEVP min operator windows"},
            "gevp_max_operator_error_pct": {"name": "GEVP max operator error %"},
            "gevp_remove_artifacts": {"name": "GEVP remove artifacts"},
            "gevp_eig_rel_cutoff": {"name": "GEVP rel eig cutoff"},
            "gevp_cond_limit": {"name": "GEVP cond limit"},
            "gevp_shrinkage": {"name": "GEVP shrinkage"},
            "gevp_bootstrap_mode": {"name": "GEVP bootstrap mode"},
        },
        default_layout=type("NewDiracEWSettingsGrid", (pn.GridBox,), {"ncols": 2}),
    )
    new_dirac_ew_channel_family_selectors: dict[str, pn.widgets.MultiSelect] = {
        family: pn.widgets.MultiSelect(
            name=f"{family.replace('_', ' ').title()} Variants",
            options=list(variants),
            value=list(DEFAULT_ELECTROWEAK_CHANNEL_VARIANT_SELECTION.get(family, ())),
            size=6,
        )
        for family, variants in ELECTROWEAK_CHANNEL_VARIANTS_BY_FAMILY.items()
    }
    new_dirac_ew_channel_family_selector_grid = pn.GridBox(
        *new_dirac_ew_channel_family_selectors.values(),
        ncols=2,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_channel_family_selector_layout = pn.Column(
        pn.pane.Markdown("### Electroweak Channel Family Selection"),
        new_dirac_ew_channel_family_selector_grid,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_coupling_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_coupling_ref_table = pn.widgets.Tabulator(
        pd.DataFrame({
            "name": list(ELECTROWEAK_COUPLING_NAMES),
            **{
                col: [
                    format_ref_value(DEFAULT_ELECTROWEAK_COUPLING_REFS.get(name, {}).get(col))
                    for name in ELECTROWEAK_COUPLING_NAMES
                ]
                for col in ELECTROWEAK_COUPLING_REFERENCE_COLUMNS
            },
        }),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
        selectable=False,
        configuration={"editable": True},
        editors=dict.fromkeys(ELECTROWEAK_COUPLING_REFERENCE_COLUMNS, "input"),
    )

    new_dirac_ew_channel_plots = pn.Column(sizing_mode="stretch_width")
    new_dirac_ew_plots_overlay_corr = pn.pane.HoloViews(
        sizing_mode="stretch_width", linked_axes=False
    )
    new_dirac_ew_plots_overlay_meff = pn.pane.HoloViews(
        sizing_mode="stretch_width", linked_axes=False
    )
    new_dirac_ew_plots_spectrum = pn.pane.HoloViews(sizing_mode="stretch_width", linked_axes=False)
    new_dirac_ew_mass_mode = pn.widgets.RadioButtonGroup(
        name="Mass Display",
        options=["AIC-Weighted", "Best Window"],
        value="AIC-Weighted",
        button_type="default",
    )
    new_dirac_ew_mass_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_ratio_pane = pn.pane.Markdown(
        "**Electroweak Ratios:** Compute channels to see ratios.",
        sizing_mode="stretch_width",
    )
    new_dirac_ew_ratio_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_best_combo_summary = pn.pane.Markdown(
        (
            "**Best global variant combo:** compute electroweak channels to evaluate "
            "the top 5 one-variant-per-channel combinations by total absolute % ratio error."
        ),
        sizing_mode="stretch_width",
    )
    new_dirac_ew_best_combo_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_best_combo_ratio_pane = pn.pane.Markdown(
        (
            "**Mass Ratios (Best Global Combo):** compute electroweak channels to "
            "populate best-combination ratios."
        ),
        sizing_mode="stretch_width",
    )
    new_dirac_ew_fit_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_compare_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_anchor_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=20,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_ref_table = pn.widgets.Tabulator(
        pd.DataFrame({
            "channel": list(ELECTROWEAK_CHANNELS),
            "mass_ref_GeV": [
                format_ref_value(DEFAULT_ELECTROWEAK_REFS.get(name))
                for name in ELECTROWEAK_CHANNELS
            ],
        }),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
        selectable=False,
        configuration={"editable": True},
        editors={"mass_ref_GeV": "input"},
    )
    new_dirac_ew_filtered_summary = pn.pane.Markdown(
        "**Filtered-out candidates:** none.",
        sizing_mode="stretch_width",
    )
    new_dirac_ew_filtered_mass_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_symmetry_breaking_summary = pn.pane.Markdown(
        "**Symmetry Breaking:** _run electroweak analysis to populate._",
        sizing_mode="stretch_width",
    )
    new_dirac_ew_multiscale_summary = pn.pane.Markdown(
        "### SU(2) Multiscale Summary\n_Multiscale kernels disabled._",
        sizing_mode="stretch_width",
    )
    new_dirac_ew_multiscale_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_multiscale_per_scale_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination="remote",
        page_size=20,
        show_index=False,
        sizing_mode="stretch_width",
    )
    new_dirac_ew_su2_gevp_widgets = create_gevp_dashboard_widgets()
    new_dirac_ew_u1_gevp_widgets = create_gevp_dashboard_widgets()
    new_dirac_ew_ew_mixed_gevp_widgets = create_gevp_dashboard_widgets()
    new_dirac_ew_su2_mass_spectrum_widgets = create_gevp_mass_spectrum_widgets()
    new_dirac_ew_u1_mass_spectrum_widgets = create_gevp_mass_spectrum_widgets()
    new_dirac_ew_ew_mixed_mass_spectrum_widgets = create_gevp_mass_spectrum_widgets()

    # =====================================================================
    # Callbacks
    # =====================================================================

    def _update_electroweak_plots_generic(
        results: dict[str, ChannelCorrelatorResult],
        channel_plots_container,
        plots_spectrum,
        plots_overlay_corr,
        plots_overlay_meff,
    ) -> None:
        mandatory_channels = (
            "su2_doublet",
            "ew_mixed",
            "su2_component",
            "su2_phase",
        )
        filtered_results: dict[str, ChannelCorrelatorResult] = {}
        seen: set[str] = set()

        for name in mandatory_channels:
            result = results.get(name)
            if result is None or result.n_samples <= 0 or name in seen:
                continue
            filtered_results[name] = result
            seen.add(name)

        for name, result in results.items():
            if name in seen or result.n_samples <= 0:
                continue
            if name.endswith("_gevp"):
                filtered_results[name] = result
                seen.add(name)
                continue
            if "_multiscale" in name:
                continue
            filtered_results[name] = result
            seen.add(name)

        update_correlator_plots(
            filtered_results,
            channel_plots_container,
            plots_spectrum,
            plots_overlay_corr,
            plots_overlay_meff,
            correlator_logy=False,
            channels_per_row=2,
        )

    def _canonicalize_electroweak_channel_name(raw_name: str) -> str:
        channel_name = str(raw_name).strip()
        if not channel_name:
            return ""
        while channel_name.endswith("_q2"):
            channel_name = channel_name[:-3]
        while channel_name.endswith("_companion"):
            channel_name = channel_name[:-10]
        changed = True
        while changed:
            changed = False
            for suffix in ("_directed", "_cloner", "_resister", "_persister"):
                if channel_name.endswith(suffix):
                    channel_name = channel_name[: -len(suffix)]
                    changed = True
        return channel_name

    def _extract_electroweak_refs_from_table(table: pn.widgets.Tabulator) -> dict[str, float]:
        refs: dict[str, float] = {}
        df = table.value
        if not isinstance(df, pd.DataFrame):
            return refs
        for _, row in df.iterrows():
            channel = str(row.get("channel", "")).strip()
            if not channel:
                continue
            raw_mass = row.get("mass_ref_GeV")
            if isinstance(raw_mass, str):
                raw_mass = raw_mass.strip()
                if raw_mass == "":
                    continue
            try:
                mass_ref = float(raw_mass)
            except (TypeError, ValueError):
                continue
            if mass_ref > 0:
                refs[channel] = mass_ref
        return refs

    def _build_electroweak_ratio_specs(
        refs: dict[str, float],
    ) -> list[tuple[str, str, str]]:
        ordered_channels: list[str] = []
        for channel in ELECTROWEAK_CANONICAL_CHANNEL_ORDER:
            if channel in refs:
                ordered_channels.append(channel)
        for channel in sorted(refs):
            if channel not in ordered_channels:
                ordered_channels.append(channel)

        ratio_specs: list[tuple[str, str, str]] = []
        for numerator in ordered_channels:
            num_ref = refs.get(numerator, float("nan"))
            if not np.isfinite(num_ref) or num_ref <= 0:
                continue
            for denominator in ordered_channels:
                if numerator == denominator:
                    continue
                den_ref = refs.get(denominator, float("nan"))
                if not np.isfinite(den_ref) or den_ref <= 0:
                    continue
                ratio_specs.append((
                    numerator,
                    denominator,
                    f"target ≈ {num_ref / den_ref:.5g}",
                ))
        return ratio_specs

    def _extract_electroweak_ratio_reference(spec_annotation: str) -> float:
        match = re.search(r"≈\s*([0-9]+(?:\.[0-9]+)?)", str(spec_annotation))
        if match is None:
            return float("nan")
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return float("nan")

    def _electroweak_variant_options(
        results: dict[str, ChannelCorrelatorResult],
    ) -> dict[str, list[str]]:
        options: dict[str, list[str]] = {}
        for name, result in results.items():
            if int(getattr(result, "n_samples", 0)) <= 0:
                continue
            canonical = _canonicalize_electroweak_channel_name(name)
            if not canonical:
                continue
            options.setdefault(canonical, [])
            name_str = str(name)
            if name_str not in options[canonical]:
                options[canonical].append(name_str)
        return options

    def _electroweak_best_variant_overrides(
        results: dict[str, ChannelCorrelatorResult],
        ratio_specs: list[tuple[str, str, str]] | None,
        mode: str = "AIC-Weighted",
    ) -> dict[str, str]:
        if not ratio_specs:
            new_dirac_ew_best_combo_summary.object = (
                "**Best global variant combo:** unavailable "
                "(no finite measured ratio targets are configured)."
            )
            new_dirac_ew_best_combo_table.value = pd.DataFrame()
            new_dirac_ew_best_combo_ratio_pane.object = (
                "**Mass Ratios (Best Global Combo):** unavailable "
                "(no finite measured ratio targets are configured)."
            )
            return {}

        variant_options = _electroweak_variant_options(results)
        combo_channels: list[str] = []
        for numerator, denominator, _spec in ratio_specs:
            if numerator not in combo_channels:
                combo_channels.append(numerator)
            if denominator not in combo_channels:
                combo_channels.append(denominator)

        available = {
            channel: list(variants) for channel, variants in variant_options.items() if variants
        }
        missing_channels = [channel for channel in combo_channels if channel not in available]
        if missing_channels:
            reason = "missing variants for: " + ", ".join(f"`{ch}`" for ch in missing_channels)
            new_dirac_ew_best_combo_summary.object = (
                f"**Best global variant combo:** unavailable ({reason})."
            )
            new_dirac_ew_best_combo_table.value = pd.DataFrame()
            new_dirac_ew_best_combo_ratio_pane.object = (
                f"**Mass Ratios (Best Global Combo):** unavailable ({reason})."
            )
            return {}

        variant_mass_lookup: dict[str, dict[str, float]] = {
            channel: {
                variant: get_channel_mass(results[variant], mode=mode)
                if variant in results
                else float("nan")
                for variant in variants
            }
            for channel, variants in available.items()
        }

        combo_rows: list[dict[str, Any]] = []
        option_lists = [available[channel] for channel in combo_channels]
        combos_considered = 0
        combo_truncated = False
        max_combo_evaluations = 100000
        total_combinations = int(
            np.prod([len(options) for options in option_lists], dtype=np.int64)
        )
        evaluation_options = {channel: list(options) for channel, options in available.items()}
        if total_combinations > max_combo_evaluations:
            if len(combo_channels) <= 0:
                return {}
            per_channel_cap = max(
                1,
                int(np.floor(max_combo_evaluations ** (1.0 / float(len(combo_channels))))),
            )
            for channel in combo_channels:
                if len(evaluation_options[channel]) > per_channel_cap:
                    evaluation_options[channel] = evaluation_options[channel][:per_channel_cap]
                    combo_truncated = True
            option_lists = [evaluation_options[channel] for channel in combo_channels]

        for selected_variants in itertools.product(*option_lists):
            combos_considered += 1
            selection = {
                channel: str(variant_name)
                for channel, variant_name in zip(combo_channels, selected_variants, strict=False)
            }
            per_channel_error = dict.fromkeys(combo_channels, 0.0)
            per_ratio_error: dict[str, float] = {}
            total_abs_pct_error = 0.0
            used_ratio_count = 0
            valid_combo = True

            for numerator, denominator, annotation in ratio_specs:
                num_variant = selection.get(numerator)
                den_variant = selection.get(denominator)
                if num_variant is None or den_variant is None:
                    valid_combo = False
                    break
                num_mass = variant_mass_lookup.get(numerator, {}).get(num_variant, float("nan"))
                den_mass = variant_mass_lookup.get(denominator, {}).get(den_variant, float("nan"))
                if not np.isfinite(num_mass) or not np.isfinite(den_mass) or den_mass <= 0:
                    valid_combo = False
                    break

                reference = _extract_electroweak_ratio_reference(annotation)
                if not np.isfinite(reference) or reference <= 0:
                    continue
                ratio = float(num_mass / den_mass)
                abs_pct_error = float(abs((ratio / reference - 1.0) * 100.0))
                per_ratio_error[f"error_{numerator}_over_{denominator}_abs_pct"] = abs_pct_error
                per_channel_error[numerator] += abs_pct_error
                per_channel_error[denominator] += abs_pct_error
                total_abs_pct_error += abs_pct_error
                used_ratio_count += 1

            if not valid_combo or used_ratio_count == 0:
                continue

            combo_row: dict[str, Any] = {
                "total_abs_pct_error": total_abs_pct_error,
                "mean_abs_pct_error": total_abs_pct_error / float(used_ratio_count),
                "n_ratio_targets": int(used_ratio_count),
            }
            for channel in combo_channels:
                combo_row[f"variant_{channel}"] = selection[channel]
                combo_row[f"error_{channel}_abs_pct"] = float(per_channel_error[channel])
            combo_row.update(per_ratio_error)
            combo_rows.append(combo_row)

        if combo_rows:
            combo_df = pd.DataFrame(combo_rows).sort_values(
                ["total_abs_pct_error", "mean_abs_pct_error"],
                ascending=[True, True],
                kind="stable",
            )
            top_combo_df = combo_df.head(5).reset_index(drop=True)
            new_dirac_ew_best_combo_table.value = top_combo_df
            best_row = top_combo_df.iloc[0]
            best_selection = {
                channel: str(best_row[f"variant_{channel}"])
                for channel in combo_channels
                if f"variant_{channel}" in best_row
            }
            truncation_note = (
                " Evaluated a capped subset of combinations." if combo_truncated else ""
            )
            new_dirac_ew_best_combo_summary.object = (
                f"**Best global variant combo (top 5):** "
                f"`{len(combo_df)}` valid combinations ranked by summed absolute `%` error "
                f"across `{len(ratio_specs)}` ratio targets.  \n"
                f"_Best total error:_ `{float(best_row['total_abs_pct_error']):.4g}%` "
                f"(mean `{float(best_row['mean_abs_pct_error']):.4g}%`).  \n"
                f"_Combinations considered:_ `{combos_considered}`.{truncation_note}"
            )
            best_ratio_lines = ["**Mass Ratios (Best Global Combo):**"]
            for numerator, denominator, annotation in ratio_specs:
                num_variant = best_selection.get(numerator)
                den_variant = best_selection.get(denominator)
                if not num_variant or not den_variant:
                    continue
                num_mass = variant_mass_lookup.get(numerator, {}).get(num_variant, float("nan"))
                den_mass = variant_mass_lookup.get(denominator, {}).get(den_variant, float("nan"))
                if not np.isfinite(num_mass) or not np.isfinite(den_mass) or den_mass <= 0:
                    continue
                ratio_value = float(num_mass / den_mass)
                ref_value = _extract_electroweak_ratio_reference(annotation)
                abs_pct_err = float("nan")
                if np.isfinite(ref_value) and ref_value > 0:
                    abs_pct_err = float(abs((ratio_value / ref_value - 1.0) * 100.0))
                ann = str(annotation).strip()
                if not ann:
                    if np.isfinite(ref_value) and ref_value > 0:
                        ann = f"target ≈ {ref_value:.4g}"
                    else:
                        ann = "no external target"
                if np.isfinite(abs_pct_err):
                    ann = f"{ann}; abs error {abs_pct_err:.2f}%"
                best_ratio_lines.append(
                    f"- {numerator}/{denominator}: **{ratio_value:.3f}** ({ann})"
                )
            if len(best_ratio_lines) == 1:
                best_ratio_lines.append("- n/a (missing valid masses for selected variants)")
            new_dirac_ew_best_combo_ratio_pane.object = "  \n".join(best_ratio_lines)
            return best_selection
        if total_combinations == 0:
            reason = "no variant combinations available"
        else:
            reason = "no complete variant combination has valid masses for all ratio targets"
        new_dirac_ew_best_combo_summary.object = (
            f"**Best global variant combo:** unavailable ({reason})."
        )
        new_dirac_ew_best_combo_table.value = pd.DataFrame()
        new_dirac_ew_best_combo_ratio_pane.object = (
            f"**Mass Ratios (Best Global Combo):** unavailable ({reason})."
        )
        return {}

    def _apply_electroweak_comparison_overrides(
        results: dict[str, ChannelCorrelatorResult],
        comparison_channel_overrides: dict[str, str] | None,
    ) -> dict[str, ChannelCorrelatorResult]:
        if not comparison_channel_overrides:
            comparison_channel_overrides = {}

        variant_options = _electroweak_variant_options(results)
        canonical_order = list(ELECTROWEAK_CANONICAL_CHANNEL_ORDER)
        for channel in sorted(variant_options):
            if channel not in canonical_order:
                canonical_order.append(channel)

        projected: dict[str, ChannelCorrelatorResult] = {}
        for canonical_name in canonical_order:
            options = variant_options.get(canonical_name)
            if not options:
                continue
            selected_name = comparison_channel_overrides.get(canonical_name)
            chosen_name = selected_name if selected_name in options else options[0]
            chosen_result = results.get(chosen_name)
            if chosen_result is not None:
                projected[canonical_name] = chosen_result

        for name, result in results.items():
            canonical_name = _canonicalize_electroweak_channel_name(name)
            if canonical_name in projected:
                continue
            projected[str(name)] = result

        return projected

    def _build_electroweak_gevp_input_results(
        base_results: dict[str, ChannelCorrelatorResult],
        comparison_channel_overrides: dict[str, str] | None,
    ) -> dict[str, ChannelCorrelatorResult]:
        gevp_results = dict(base_results)
        if not comparison_channel_overrides:
            return gevp_results
        for base_channel in (
            list(SU2_BASE_CHANNELS) + list(U1_BASE_CHANNELS) + list(EW_MIXED_BASE_CHANNELS)
        ):
            selected = str(comparison_channel_overrides.get(base_channel, base_channel)).strip()
            selected_result = base_results.get(selected)
            if selected_result is not None:
                gevp_results[base_channel] = selected_result
        return gevp_results

    def _update_electroweak_tables_generic(
        results: dict[str, ChannelCorrelatorResult],
        mode: str,
        mass_table,
        ratio_pane,
        ratio_table,
        fit_table,
        anchor_table,
        compare_table,
        ref_table,
        comparison_channel_overrides: dict[str, str] | None = None,
        ratio_specs: list[tuple[str, str, str]] | None = None,
    ) -> None:
        if ratio_specs is None:
            ratio_specs = _build_electroweak_ratio_specs(
                _extract_electroweak_refs_from_table(ref_table)
            )
        comparison_results = dict(results)
        if comparison_channel_overrides:
            comparison_results = _apply_electroweak_comparison_overrides(
                results,
                comparison_channel_overrides,
            )
        update_mass_table(comparison_results, mass_table, mode)
        ratio_pane.object = format_ratios(
            comparison_results,
            mode,
            title="Electroweak Ratios",
            ratio_specs=ratio_specs,
        )

        masses = extract_masses(comparison_results, mode, family_map=None)
        r2s = extract_r2(comparison_results, mode, family_map=None)
        base_name = "u1_dressed" if "u1_dressed" in masses else "u1_phase"

        refs = _extract_electroweak_refs_from_table(ref_table)

        ratio_rows = _build_electroweak_ratio_rows(masses, base_name, refs=refs)
        ratio_table.value = pd.DataFrame(ratio_rows) if ratio_rows else pd.DataFrame()

        if not masses or not refs:
            fit_table.value = pd.DataFrame()
            anchor_table.value = pd.DataFrame()
            compare_table.value = pd.DataFrame()
            return

        fit_table.value = pd.DataFrame(
            _build_electroweak_best_fit_rows(
                masses, refs, r2s, build_best_fit_rows_generic=build_best_fit_rows_generic
            )
        )
        anchor_table.value = pd.DataFrame(
            _build_electroweak_anchor_rows(
                masses, refs, r2s, build_anchor_rows_generic=build_anchor_rows_generic
            )
        )
        compare_table.value = pd.DataFrame(
            _build_electroweak_comparison_rows(masses, refs, best_fit_scale=best_fit_scale)
        )

    def _resolve_new_dirac_ew_filter_values() -> tuple[float, int, float, bool]:
        try:
            min_r2 = float(new_dirac_ew_settings.gevp_min_operator_r2)
        except (TypeError, ValueError):
            min_r2 = 0.5
        try:
            min_windows = max(0, int(new_dirac_ew_settings.gevp_min_operator_windows))
        except (TypeError, ValueError):
            min_windows = 10
        try:
            max_error_pct = float(new_dirac_ew_settings.gevp_max_operator_error_pct)
        except (TypeError, ValueError):
            max_error_pct = 30.0
        remove_artifacts = bool(new_dirac_ew_settings.gevp_remove_artifacts)
        return min_r2, min_windows, max_error_pct, remove_artifacts

    def _update_ew_gevp_displays(
        multiscale_output: MultiscaleElectroweakOutput | None,
        results: dict[str, ChannelCorrelatorResult],
    ) -> None:
        min_r2, min_windows, max_error_pct, remove_artifacts = (
            _resolve_new_dirac_ew_filter_values()
        )
        t0 = int(new_dirac_ew_settings.gevp_t0)
        eig_rel_cutoff = float(new_dirac_ew_settings.gevp_eig_rel_cutoff)
        per_scale = multiscale_output.per_scale_results if multiscale_output else {}
        gevp_results = {k: v for k, v in results.items() if k.endswith("_gevp")}
        for family, widgets, ms_widgets in (
            ("su2", new_dirac_ew_su2_gevp_widgets, new_dirac_ew_su2_mass_spectrum_widgets),
            ("u1", new_dirac_ew_u1_gevp_widgets, new_dirac_ew_u1_mass_spectrum_widgets),
            (
                "ew_mixed",
                new_dirac_ew_ew_mixed_gevp_widgets,
                new_dirac_ew_ew_mixed_mass_spectrum_widgets,
            ),
        ):
            try:
                update_gevp_dashboard(
                    widgets,
                    selected_channel_name=f"{family.upper()} GEVP",
                    raw_channel_name=f"{family}_companion"
                    if family != "ew_mixed"
                    else "ew_mixed_companion",
                    per_scale_results=per_scale,
                    original_results=results,
                    companion_gevp_results=gevp_results,
                    min_r2=min_r2,
                    min_windows=min_windows,
                    max_error_pct=max_error_pct,
                    remove_artifacts=remove_artifacts,
                    selected_families=[family],
                    t0=t0,
                    eig_rel_cutoff=eig_rel_cutoff,
                    use_connected=bool(new_dirac_ew_settings.use_connected),
                )
            except Exception:
                clear_gevp_dashboard(widgets)
            # Update multi-mode mass spectrum widgets
            spectrum = state.get(f"_ew_gevp_mass_spectrum_{family}")
            t0_sweep = state.get(f"_ew_gevp_t0_sweep_{family}")
            try:
                dt_val = (
                    float(new_dirac_ew_settings.dt)
                    if hasattr(new_dirac_ew_settings, "dt")
                    else 1.0
                )
            except (TypeError, ValueError):
                dt_val = 1.0
            update_gevp_mass_spectrum(
                ms_widgets,
                spectrum=spectrum,
                family_label=family.upper(),
                dt=dt_val,
                t0_sweep=t0_sweep,
            )

    def _update_new_dirac_ew_electroweak_tables(
        results: dict[str, ChannelCorrelatorResult],
        mode: str | None = None,
        comparison_channel_overrides: dict[str, str] | None = None,
        ratio_specs: list[tuple[str, str, str]] | None = None,
    ) -> None:
        if mode is None:
            mode = new_dirac_ew_mass_mode.value
        min_r2, min_windows, max_error_pct, remove_artifacts = (
            _resolve_new_dirac_ew_filter_values()
        )
        filtered_results, filtered_out_results = split_results_by_companion_gevp_filters(
            results,
            min_r2=min_r2,
            min_windows=min_windows,
            max_error_pct=max_error_pct,
            remove_artifacts=remove_artifacts,
        )
        _update_electroweak_tables_generic(
            filtered_results,
            mode,
            new_dirac_ew_mass_table,
            new_dirac_ew_ratio_pane,
            new_dirac_ew_ratio_table,
            new_dirac_ew_fit_table,
            new_dirac_ew_anchor_table,
            new_dirac_ew_compare_table,
            new_dirac_ew_ref_table,
            comparison_channel_overrides=comparison_channel_overrides,
            ratio_specs=ratio_specs,
        )
        filtered_out_dict = {name: payload[0] for name, payload in filtered_out_results.items()}
        update_mass_table(
            filtered_out_dict,
            new_dirac_ew_filtered_mass_table,
            str(mode),
        )

        min_r2_str = f"{min_r2:.3g}" if np.isfinite(min_r2) else "off"
        max_error_pct_str = f"{max_error_pct:.3g}%" if np.isfinite(max_error_pct) else "off"
        if filtered_out_results:
            preview = ", ".join(
                f"{name}({reason})" for name, (_, reason) in list(filtered_out_results.items())[:6]
            )
            suffix = " ..." if len(filtered_out_results) > 6 else ""
            new_dirac_ew_filtered_summary.object = (
                f"**Filtered-out candidates:** `{len(filtered_out_results)}` excluded; "
                f"`{len(filtered_results)}` shown.  \n"
                f"_Filters:_ `R² >= {min_r2_str}`, `n_windows >= {min_windows}`, "
                f"`error % <= {max_error_pct_str}`, "
                f"`remove artifacts={'on' if remove_artifacts else 'off'}`.  \n"
                f"_Preview:_ `{preview}{suffix}`"
            )
        else:
            new_dirac_ew_filtered_summary.object = (
                "**Filtered-out candidates:** none.  \n"
                f"_Filters:_ `R² >= {min_r2_str}`, `n_windows >= {min_windows}`, "
                f"`error % <= {max_error_pct_str}`, "
                f"`remove artifacts={'on' if remove_artifacts else 'off'}`."
            )
        _update_ew_gevp_displays(
            state.get("new_dirac_ew_multiscale_output"),
            results,
        )

        # --- Symmetry-breaking derived observables ---
        sb_lines = ["**Symmetry Breaking Observables:**"]
        ew_masses = extract_masses(filtered_results, mode=str(mode), family_map=None)
        m_z_proxy = ew_masses.get("fitness_phase")
        m_w_proxy = ew_masses.get("clone_indicator")
        if m_z_proxy and m_z_proxy > 0 and m_w_proxy and m_w_proxy > 0:
            cos_tw = min(m_w_proxy / m_z_proxy, 1.0)
            sin2_tw = 1.0 - cos_tw**2
            sb_lines.append(f"- **M_Z proxy** (fitness_phase): `{m_z_proxy:.6g}`")
            sb_lines.append(f"- **M_W proxy** (clone_indicator): `{m_w_proxy:.6g}`")
            theta_w_deg = _math.degrees(_math.acos(cos_tw))
            sb_lines.append(
                f"- **\u03b8_W** = `{theta_w_deg:.2f}\u00b0` "
                f"(SM: `{_math.degrees(_math.acos(80.379 / 91.1876)):.2f}\u00b0`)"
            )
            sb_lines.append(
                f"- **sin\u00b2(\u03b8_W)** = `{sin2_tw:.5f}` (SM: `0.23129`; "
                f"error: `{(sin2_tw / 0.23129 - 1.0) * 100:.1f}%`)"
            )
        elif m_z_proxy and m_z_proxy > 0:
            sb_lines.append(f"- **M_Z proxy** (fitness_phase): `{m_z_proxy:.6g}`")
            sb_lines.append("- **M_W proxy** (clone_indicator): _not available_")
        elif m_w_proxy and m_w_proxy > 0:
            sb_lines.append("- **M_Z proxy** (fitness_phase): _not available_")
            sb_lines.append(f"- **M_W proxy** (clone_indicator): `{m_w_proxy:.6g}`")
        else:
            sb_lines.append("- _Fitness phase and clone indicator channels not computed._")

        # Parity violation: compare velocity-norm masses across walker types
        parity_masses = {}
        for label in ("cloner", "resister", "persister"):
            m = ew_masses.get(f"velocity_norm_{label}")
            if m and m > 0:
                parity_masses[label] = m
        if len(parity_masses) >= 2:
            sb_lines.append("- **Parity test** (velocity norm by walker type):")
            types = sorted(parity_masses.keys())
            for i, t1 in enumerate(types):
                for t2 in types[i + 1 :]:
                    ratio = parity_masses[t1] / parity_masses[t2]
                    sb_lines.append(
                        f"  - m({t1})/m({t2}) = `{ratio:.4f}` "
                        f"(parity conservation \u2192 1.0; "
                        f"deviation: `{(ratio - 1.0) * 100:+.1f}%`)"
                    )
        else:
            sb_lines.append(
                "- **Parity test:** _enable walker type split and select "
                "parity_velocity channels._"
            )
        new_dirac_ew_symmetry_breaking_summary.object = "  \n".join(sb_lines)

    def _update_new_dirac_ew_multiscale_views(
        output: MultiscaleElectroweakOutput | None,
        *,
        original_results: dict[str, ChannelCorrelatorResult] | None = None,
        error: str | None = None,
    ) -> None:
        if error:
            new_dirac_ew_multiscale_summary.object = (
                f"### SU(2) Multiscale Summary\n- Status: failed\n- Error: `{error}`"
            )
            new_dirac_ew_multiscale_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
            clear_gevp_dashboard(new_dirac_ew_su2_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_u1_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_ew_mixed_gevp_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_su2_mass_spectrum_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_u1_mass_spectrum_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_ew_mixed_mass_spectrum_widgets)
            return

        if output is None:
            new_dirac_ew_multiscale_summary.object = (
                "### SU(2) Multiscale Summary\n"
                "_Multiscale kernels disabled (original estimators only)._"
            )
            new_dirac_ew_multiscale_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()
            return

        scale_values = (
            output.scales.detach().cpu().numpy() if output.scales.numel() else np.array([])
        )
        summary_lines = [
            "### SU(2) Multiscale Summary",
            f"- Scales: `{len(scale_values)}`",
            f"- Frames: `{len(output.frame_indices)}`",
            f"- Bootstrap mode: `{output.bootstrap_mode_applied}`",
        ]
        if scale_values.size > 0:
            summary_lines.append(
                f"- Scale range: `[{float(scale_values.min()):.4g}, {float(scale_values.max()):.4g}]`"
            )
        if output.notes:
            for note in output.notes:
                summary_lines.append(f"- Note: {note}")
        new_dirac_ew_multiscale_summary.object = "  \n".join(summary_lines)

        mode = str(new_dirac_ew_mass_mode.value)
        min_r2, min_windows, max_error_pct, remove_artifacts = (
            _resolve_new_dirac_ew_filter_values()
        )
        rows: list[dict[str, Any]] = []
        per_scale_rows: list[dict[str, Any]] = []
        for alias, results_per_scale in output.per_scale_results.items():
            base_name = str(alias).replace("_companion", "")
            best_idx_raw = int(output.best_scale_index.get(alias, -1))
            best_idx = best_idx_raw if 0 <= best_idx_raw < len(results_per_scale) else -1
            best_result = results_per_scale[best_idx] if best_idx >= 0 else None
            if best_result is not None:
                best_reason = companion_gevp_filter_reason(
                    best_result,
                    min_r2=min_r2,
                    min_windows=min_windows,
                    max_error_pct=max_error_pct,
                    remove_artifacts=remove_artifacts,
                )
                if best_reason is not None:
                    best_result = None
                    best_idx = -1

            original_mass = float("nan")
            if isinstance(original_results, dict):
                original_result = original_results.get(base_name)
                if original_result is not None:
                    original_reason = companion_gevp_filter_reason(
                        original_result,
                        min_r2=min_r2,
                        min_windows=min_windows,
                        max_error_pct=max_error_pct,
                        remove_artifacts=remove_artifacts,
                    )
                    if original_reason is None:
                        original_mass = float(get_channel_mass(original_result, mode))

            for scale_idx, res in enumerate(results_per_scale):
                if res is None:
                    continue
                reason = companion_gevp_filter_reason(
                    res,
                    min_r2=min_r2,
                    min_windows=min_windows,
                    max_error_pct=max_error_pct,
                    remove_artifacts=remove_artifacts,
                )
                if reason is not None:
                    continue
                fit = res.mass_fit if isinstance(res.mass_fit, dict) else {}
                mass = float(get_channel_mass(res, mode))
                mass_error = float(get_channel_mass_error(res, mode))
                err_pct = (
                    abs(mass_error / mass) * 100.0
                    if np.isfinite(mass) and mass > 0 and np.isfinite(mass_error)
                    else float("nan")
                )
                per_scale_rows.append({
                    "channel": base_name,
                    "scale_idx": int(scale_idx),
                    "scale": (
                        float(scale_values[scale_idx])
                        if scale_values.size > scale_idx
                        else float("nan")
                    ),
                    "mass": mass,
                    "mass_error": mass_error,
                    "mass_error_pct": err_pct,
                    "r2": float(get_channel_r2(res, mode)),
                    "n_windows": extract_n_windows_for_filter(res),
                    "source": str(fit.get("source", "multiscale")),
                })

            if best_result is None:
                continue
            best_mass = float(get_channel_mass(best_result, mode))
            best_error = float(get_channel_mass_error(best_result, mode))
            best_err_pct = (
                abs(best_error / best_mass) * 100.0
                if np.isfinite(best_mass) and best_mass > 0 and np.isfinite(best_error)
                else float("nan")
            )
            rows.append({
                "channel": base_name,
                "best_scale_idx": int(best_idx),
                "best_scale": (
                    float(scale_values[best_idx])
                    if scale_values.size > best_idx >= 0
                    else float("nan")
                ),
                "mass": best_mass,
                "mass_error": best_error,
                "mass_error_pct": best_err_pct,
                "r2": float(get_channel_r2(best_result, mode)),
                "n_windows": extract_n_windows_for_filter(best_result),
                "original_mass": original_mass,
                "delta_vs_original_pct": (
                    ((best_mass - original_mass) / original_mass) * 100.0
                    if np.isfinite(original_mass) and original_mass > 0 and np.isfinite(best_mass)
                    else float("nan")
                ),
            })

        new_dirac_ew_multiscale_table.value = (
            pd.DataFrame(rows).sort_values(["channel"], kind="stable") if rows else pd.DataFrame()
        )
        new_dirac_ew_multiscale_per_scale_table.value = (
            pd.DataFrame(per_scale_rows).sort_values(["channel", "scale_idx"], kind="stable")
            if per_scale_rows
            else pd.DataFrame()
        )

    def _on_new_dirac_ew_mass_mode_change(event) -> None:
        results = state.get("new_dirac_ew_results")
        if results is not None:
            _update_new_dirac_ew_electroweak_tables(
                results,
                event.new,
                comparison_channel_overrides=state.get("new_dirac_ew_comparison_overrides"),
                ratio_specs=state.get("new_dirac_ew_ratio_specs"),
            )
            _update_new_dirac_ew_multiscale_views(
                state.get("new_dirac_ew_multiscale_output"),
                original_results=results,
                error=state.get("new_dirac_ew_multiscale_error"),
            )

    new_dirac_ew_mass_mode.param.watch(_on_new_dirac_ew_mass_mode_change, "value")

    def _build_new_dirac_ew_channel_config() -> ElectroweakChannelConfig:
        return ElectroweakChannelConfig(
            warmup_fraction=new_dirac_ew_settings.simulation_range[0],
            end_fraction=new_dirac_ew_settings.simulation_range[1],
            max_lag=new_dirac_ew_settings.max_lag,
            h_eff=new_dirac_ew_settings.h_eff,
            use_connected=new_dirac_ew_settings.use_connected,
            neighbor_method="companions",
            companion_topology="distance",
            companion_topology_u1="distance",
            companion_topology_su2="clone",
            companion_topology_ew_mixed="both",
            edge_weight_mode=new_dirac_ew_settings.edge_weight_mode,
            neighbor_k=new_dirac_ew_settings.neighbor_k,
            window_widths=parse_window_widths(new_dirac_ew_settings.window_widths_spec),
            fit_mode=new_dirac_ew_settings.fit_mode,
            fit_start=new_dirac_ew_settings.fit_start,
            fit_stop=new_dirac_ew_settings.fit_stop,
            min_fit_points=new_dirac_ew_settings.min_fit_points,
            su2_operator_mode=str(new_dirac_ew_settings.su2_operator_mode),
            enable_walker_type_split=bool(new_dirac_ew_settings.enable_walker_type_split),
            walker_type_scope=str(new_dirac_ew_settings.walker_type_scope),
            mc_time_index=new_dirac_ew_settings.mc_time_index,
            compute_bootstrap_errors=new_dirac_ew_settings.compute_bootstrap_errors,
            n_bootstrap=new_dirac_ew_settings.n_bootstrap,
        )

    def on_run_new_dirac_electroweak(_):
        def _compute(history):
            state["new_dirac_ew_comparison_overrides"] = None
            state["new_dirac_ew_ratio_specs"] = None
            ew_cfg = _build_new_dirac_ew_channel_config()
            requested_electroweak_channels = collect_multiselect_values(
                new_dirac_ew_channel_family_selectors
            )
            if not requested_electroweak_channels:
                requested_electroweak_channels = [
                    channel
                    for channels in DEFAULT_ELECTROWEAK_CHANNEL_VARIANT_SELECTION.values()
                    for channel in channels
                ]
            ew_output = compute_electroweak_channels(
                history,
                channels=requested_electroweak_channels,
                config=ew_cfg,
            )
            base_results = dict(ew_output.channel_results)
            results = dict(base_results)
            ratio_specs = _build_electroweak_ratio_specs(
                _extract_electroweak_refs_from_table(new_dirac_ew_ref_table)
            )
            comparison_overrides = _electroweak_best_variant_overrides(
                base_results,
                ratio_specs,
            )
            state["new_dirac_ew_ratio_specs"] = ratio_specs
            state["new_dirac_ew_comparison_overrides"] = comparison_overrides

            multiscale_output: MultiscaleElectroweakOutput | None = None
            multiscale_error: str | None = None
            gevp_error: str | None = None
            gevp_input_results = _build_electroweak_gevp_input_results(
                base_results=base_results,
                comparison_channel_overrides=comparison_overrides,
            )

            multiscale_supported = (
                tuple(SU2_BASE_CHANNELS)
                + tuple(SU2_DIRECTIONAL_CHANNELS)
                + tuple(SU2_WALKER_TYPE_CHANNELS)
                + tuple(U1_BASE_CHANNELS)
                + tuple(EW_MIXED_BASE_CHANNELS)
            )
            multiscale_requested = [
                name for name in requested_electroweak_channels if name in multiscale_supported
            ]
            if bool(new_dirac_ew_settings.use_multiscale_kernels) and multiscale_requested:
                try:
                    ms_cfg = MultiscaleElectroweakConfig(
                        warmup_fraction=float(new_dirac_ew_settings.simulation_range[0]),
                        end_fraction=float(new_dirac_ew_settings.simulation_range[1]),
                        mc_time_index=new_dirac_ew_settings.mc_time_index,
                        h_eff=float(new_dirac_ew_settings.h_eff),
                        lambda_alg=0.0,
                        su2_operator_mode=str(new_dirac_ew_settings.su2_operator_mode),
                        enable_walker_type_split=bool(
                            new_dirac_ew_settings.enable_walker_type_split
                        ),
                        walker_type_scope=str(new_dirac_ew_settings.walker_type_scope),
                        edge_weight_mode=str(new_dirac_ew_settings.edge_weight_mode),
                        n_scales=int(new_dirac_ew_settings.n_scales),
                        kernel_type=str(new_dirac_ew_settings.kernel_type),
                        kernel_distance_method=str(new_dirac_ew_settings.kernel_distance_method),
                        kernel_assume_all_alive=bool(
                            new_dirac_ew_settings.kernel_assume_all_alive
                        ),
                        kernel_batch_size=int(new_dirac_ew_settings.kernel_batch_size),
                        kernel_scale_frames=int(new_dirac_ew_settings.kernel_scale_frames),
                        kernel_scale_q_low=float(new_dirac_ew_settings.kernel_scale_q_low),
                        kernel_scale_q_high=float(new_dirac_ew_settings.kernel_scale_q_high),
                        max_lag=int(new_dirac_ew_settings.max_lag),
                        use_connected=bool(new_dirac_ew_settings.use_connected),
                        fit_mode=str(new_dirac_ew_settings.fit_mode),
                        fit_start=int(new_dirac_ew_settings.fit_start),
                        fit_stop=new_dirac_ew_settings.fit_stop,
                        min_fit_points=int(new_dirac_ew_settings.min_fit_points),
                        window_widths=parse_window_widths(
                            new_dirac_ew_settings.window_widths_spec
                        ),
                        best_min_r2=float(new_dirac_ew_settings.gevp_min_operator_r2),
                        best_min_windows=int(new_dirac_ew_settings.gevp_min_operator_windows),
                        best_max_error_pct=float(
                            new_dirac_ew_settings.gevp_max_operator_error_pct
                        ),
                        best_remove_artifacts=bool(new_dirac_ew_settings.gevp_remove_artifacts),
                        compute_bootstrap_errors=bool(
                            new_dirac_ew_settings.compute_bootstrap_errors
                        ),
                        n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                        bootstrap_mode=str(new_dirac_ew_settings.kernel_bootstrap_mode),
                    )
                    multiscale_output = compute_multiscale_electroweak_channels(
                        history,
                        config=ms_cfg,
                        channels=multiscale_requested,
                    )
                    min_r2, min_windows, max_error_pct, remove_artifacts = (
                        _resolve_new_dirac_ew_filter_values()
                    )
                    for alias, result in multiscale_output.best_results.items():
                        reason = companion_gevp_filter_reason(
                            result,
                            min_r2=min_r2,
                            min_windows=min_windows,
                            max_error_pct=max_error_pct,
                            remove_artifacts=remove_artifacts,
                        )
                        if reason is not None:
                            continue
                        base_name = str(alias).replace("_companion", "")
                        tagged_name = f"{base_name}_multiscale_best"
                        tagged_result = replace(result, channel_name=tagged_name)
                        if isinstance(tagged_result.mass_fit, dict):
                            tagged_result.mass_fit["source"] = "multiscale_best"
                            tagged_result.mass_fit["base_channel"] = base_name
                        results[tagged_name] = tagged_result
                except Exception as exc:
                    multiscale_error = str(exc)

            gevp_cfg = GEVPConfig(
                t0=int(new_dirac_ew_settings.gevp_t0),
                max_lag=int(new_dirac_ew_settings.max_lag),
                use_connected=bool(new_dirac_ew_settings.use_connected),
                fit_mode=str(new_dirac_ew_settings.fit_mode),
                fit_start=int(new_dirac_ew_settings.fit_start),
                fit_stop=new_dirac_ew_settings.fit_stop,
                min_fit_points=int(new_dirac_ew_settings.min_fit_points),
                window_widths=parse_window_widths(new_dirac_ew_settings.window_widths_spec),
                basis_strategy=str(new_dirac_ew_settings.gevp_basis_strategy),
                max_basis=int(new_dirac_ew_settings.gevp_max_basis),
                min_operator_r2=float(new_dirac_ew_settings.gevp_min_operator_r2),
                min_operator_windows=int(new_dirac_ew_settings.gevp_min_operator_windows),
                max_operator_error_pct=float(new_dirac_ew_settings.gevp_max_operator_error_pct),
                remove_artifacts=bool(new_dirac_ew_settings.gevp_remove_artifacts),
                eig_rel_cutoff=float(new_dirac_ew_settings.gevp_eig_rel_cutoff),
                cond_limit=float(new_dirac_ew_settings.gevp_cond_limit),
                shrinkage=float(new_dirac_ew_settings.gevp_shrinkage),
                compute_bootstrap_errors=bool(new_dirac_ew_settings.compute_bootstrap_errors),
                n_bootstrap=int(new_dirac_ew_settings.n_bootstrap),
                bootstrap_mode=str(new_dirac_ew_settings.gevp_bootstrap_mode),
                compute_multimode=bool(new_dirac_ew_settings.gevp_compute_multimode),
                tau_diag=new_dirac_ew_settings.gevp_tau_diag,
                plateau_min_length=int(new_dirac_ew_settings.gevp_plateau_min_length),
                plateau_max_slope=float(new_dirac_ew_settings.gevp_plateau_max_slope),
                multimode_fit_start=new_dirac_ew_settings.gevp_multimode_fit_start,
                multimode_fit_stop=new_dirac_ew_settings.gevp_multimode_fit_stop,
                t0_sweep_values=parse_t0_sweep_spec(new_dirac_ew_settings.gevp_t0_sweep_spec),
            )
            for gevp_family, gevp_use_flag in (
                ("su2", bool(new_dirac_ew_settings.use_su2_gevp)),
                ("u1", bool(new_dirac_ew_settings.use_u1_gevp)),
                ("ew_mixed", bool(new_dirac_ew_settings.use_ew_mixed_gevp)),
            ):
                if not gevp_use_flag:
                    continue
                gevp_basis = get_companion_gevp_basis_channels(gevp_family)
                gevp_has_series = any(
                    (
                        ch in gevp_input_results
                        and gevp_input_results[ch] is not None
                        and int(gevp_input_results[ch].n_samples) > 0
                        and int(gevp_input_results[ch].series.numel()) > 0
                    )
                    for ch in gevp_basis
                )
                if gevp_has_series:
                    try:
                        gevp_payload = compute_companion_channel_gevp(
                            base_results=gevp_input_results,
                            multiscale_output=multiscale_output,
                            config=gevp_cfg,
                            base_channel=gevp_family,
                        )
                        results[gevp_payload.result.channel_name] = gevp_payload.result
                        if gevp_payload.mass_spectrum is not None:
                            state[f"_ew_gevp_mass_spectrum_{gevp_family}"] = (
                                gevp_payload.mass_spectrum
                            )
                        if gevp_payload.t0_sweep is not None:
                            state[f"_ew_gevp_t0_sweep_{gevp_family}"] = gevp_payload.t0_sweep
                    except Exception as exc:
                        if gevp_family == "su2":
                            gevp_error = str(exc)

            state["new_dirac_ew_results"] = results
            state["new_dirac_ew_multiscale_output"] = multiscale_output
            state["new_dirac_ew_multiscale_error"] = multiscale_error
            state["new_dirac_ew_gevp_error"] = gevp_error

            _update_electroweak_plots_generic(
                results,
                new_dirac_ew_channel_plots,
                new_dirac_ew_plots_spectrum,
                new_dirac_ew_plots_overlay_corr,
                new_dirac_ew_plots_overlay_meff,
            )
            _update_new_dirac_ew_electroweak_tables(
                results,
                comparison_channel_overrides=comparison_overrides,
                ratio_specs=ratio_specs,
            )
            _update_new_dirac_ew_multiscale_views(
                multiscale_output,
                original_results=base_results,
                error=multiscale_error,
            )

            couplings = compute_coupling_constants_fn(
                history,
                h_eff=float(new_dirac_ew_settings.h_eff),
                frame_indices=ew_output.frame_indices,
                pairwise_distance_by_frame=state.get("_multiscale_geodesic_distance_by_frame"),
            )
            new_dirac_ew_coupling_table.value = pd.DataFrame(
                build_coupling_rows(
                    couplings,
                    proxies=None,
                    include_strong=False,
                    refs=extract_coupling_refs(new_dirac_ew_coupling_ref_table),
                )
            )

            n_channels = len([r for r in results.values() if r.n_samples > 0])
            summary_lines = [
                "## Electroweak Summary",
                f"- Frames used: `{len(ew_output.frame_indices)}`",
                f"- Electroweak channels with samples: `{n_channels}`",
                "- Neighbor source mode: `companions` (run-selected pairs)",
                "- Companion routing (U(1), SU(2), EW mixed): `distance`, `clone`, `both`",
                f"- SU(2) operator mode: `{new_dirac_ew_settings.su2_operator_mode}`",
                (
                    "- Walker-type split: "
                    f"`{'on' if new_dirac_ew_settings.enable_walker_type_split else 'off'}`"
                ),
                (
                    "- SU(2) multiscale kernels: "
                    f"`{'on' if new_dirac_ew_settings.use_multiscale_kernels else 'off'}`"
                ),
                (f"- SU(2) GEVP: `{'on' if new_dirac_ew_settings.use_su2_gevp else 'off'}`"),
            ]
            eps_distance_em = float(couplings.get("eps_distance_emergent", float("nan")))
            eps_clone_em = float(couplings.get("eps_clone_emergent", float("nan")))
            eps_fitness_gap_em = float(couplings.get("eps_fitness_gap_emergent", float("nan")))
            eps_geodesic_em = float(couplings.get("eps_geodesic_emergent", float("nan")))
            sin2_theta_w_em = float(couplings.get("sin2_theta_w_emergent", float("nan")))
            tan_theta_w_em = float(couplings.get("tan_theta_w_emergent", float("nan")))
            if np.isfinite(eps_distance_em):
                summary_lines.append(f"- ε_distance emergent: `{eps_distance_em:.6g}`")
            if np.isfinite(eps_clone_em):
                summary_lines.append(f"- ε_clone emergent: `{eps_clone_em:.6g}`")
            if np.isfinite(eps_geodesic_em):
                summary_lines.append(f"- ε_geodesic emergent: `{eps_geodesic_em:.6g}`")
            if np.isfinite(eps_fitness_gap_em):
                summary_lines.append(f"- ε_fitness_gap emergent: `{eps_fitness_gap_em:.6g}`")
            if np.isfinite(sin2_theta_w_em):
                theta_w_deg = float(np.degrees(np.arcsin(np.sqrt(sin2_theta_w_em))))
                summary_lines.append(
                    f"- sin²θ_W emergent: `{sin2_theta_w_em:.6g}` "
                    f"(θ_W ≈ {theta_w_deg:.1f}°, "
                    f"observed@M_Z: 0.231, GUT: 0.375)"
                )
            if np.isfinite(tan_theta_w_em):
                summary_lines.append(f"- tanθ_W emergent: `{tan_theta_w_em:.6g}`")
            if multiscale_error:
                summary_lines.append(f"- Multiscale error: `{multiscale_error}`")
            if gevp_error:
                summary_lines.append(f"- GEVP error: `{gevp_error}`")
            new_dirac_ew_summary.object = "\n".join(summary_lines)

            error_parts = []
            if multiscale_error:
                error_parts.append(f"multiscale: `{multiscale_error}`")
            if gevp_error:
                error_parts.append(f"GEVP: `{gevp_error}`")
            if error_parts:
                new_dirac_ew_status.object = "**Complete with errors:** " + "; ".join(error_parts)
            else:
                new_dirac_ew_status.object = (
                    f"**Complete:** Electroweak computed ({n_channels} channels)."
                )

        run_tab_computation(
            state,
            new_dirac_ew_status,
            "electroweak observables",
            _compute,
        )

    new_dirac_ew_run_button.on_click(on_run_new_dirac_electroweak)

    # =====================================================================
    # on_history_changed callback
    # =====================================================================

    def _on_history_changed(defer: bool) -> None:
        new_dirac_ew_run_button.disabled = False
        new_dirac_ew_status.object = "**Electroweak ready:** click Compute Electroweak."
        if not defer:
            new_dirac_ew_summary.object = "## Electroweak Summary\n_Run analysis to populate._"
            new_dirac_ew_multiscale_summary.object = (
                "### SU(2) Multiscale Summary\n"
                "_Multiscale kernels disabled (original estimators only)._"
            )
            clear_gevp_dashboard(new_dirac_ew_su2_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_u1_gevp_widgets)
            clear_gevp_dashboard(new_dirac_ew_ew_mixed_gevp_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_su2_mass_spectrum_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_u1_mass_spectrum_widgets)
            clear_gevp_mass_spectrum(new_dirac_ew_ew_mixed_mass_spectrum_widgets)
            new_dirac_ew_filtered_summary.object = "**Filtered-out candidates:** none."
            new_dirac_ew_mass_table.value = pd.DataFrame()
            new_dirac_ew_filtered_mass_table.value = pd.DataFrame()
            new_dirac_ew_ratio_table.value = pd.DataFrame()
            new_dirac_ew_fit_table.value = pd.DataFrame()
            new_dirac_ew_compare_table.value = pd.DataFrame()
            new_dirac_ew_anchor_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_table.value = pd.DataFrame()
            new_dirac_ew_multiscale_per_scale_table.value = pd.DataFrame()

    # =====================================================================
    # Tab layout
    # =====================================================================

    new_dirac_ew_note = pn.pane.Alert(
        """**Electroweak:** Proxy channels only (no Dirac operators are computed here).
This tab supports SU(2) multiscale kernels, filtering, and GEVP diagnostics.
Operator routing is fixed to run-selected companions: U(1)\u2192distance, SU(2)\u2192clone, EW mixed\u2192both (with \u03bb_alg=0).""",
        alert_type="info",
        sizing_mode="stretch_width",
    )

    new_dirac_ew_tab = pn.Column(
        new_dirac_ew_status,
        new_dirac_ew_note,
        pn.Row(new_dirac_ew_run_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Electroweak Settings", new_dirac_ew_settings_panel),
            (
                "Electroweak Channel Family Selection",
                new_dirac_ew_channel_family_selector_layout,
            ),
            active=[0, 1],
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        new_dirac_ew_summary,
        pn.pane.Markdown("### Electroweak Couplings"),
        new_dirac_ew_coupling_table,
        pn.pane.Markdown("### Electroweak Coupling References"),
        new_dirac_ew_coupling_ref_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### Electroweak Proxy Channels"),
        new_dirac_ew_mass_mode,
        new_dirac_ew_plots_spectrum,
        new_dirac_ew_mass_table,
        pn.pane.Markdown("### Filtered-Out Candidates"),
        new_dirac_ew_filtered_summary,
        new_dirac_ew_filtered_mass_table,
        new_dirac_ew_ratio_pane,
        new_dirac_ew_ratio_table,
        pn.pane.Markdown("### Best Global Variant Combinations (Top 5)"),
        new_dirac_ew_best_combo_summary,
        new_dirac_ew_best_combo_table,
        pn.pane.Markdown("### Mass Ratios (From Best Global Combination)"),
        new_dirac_ew_best_combo_ratio_pane,
        pn.pane.Markdown("### Electroweak Symmetry Breaking"),
        new_dirac_ew_symmetry_breaking_summary,
        new_dirac_ew_fit_table,
        new_dirac_ew_compare_table,
        new_dirac_ew_anchor_table,
        pn.layout.Divider(),
        new_dirac_ew_multiscale_summary,
        pn.pane.Markdown("### SU(2) Multiscale Best-Scale Selection"),
        new_dirac_ew_multiscale_table,
        pn.pane.Markdown("### SU(2) Per-Scale Candidates"),
        new_dirac_ew_multiscale_per_scale_table,
        pn.pane.Markdown("### SU(2) GEVP Analysis"),
        *build_gevp_dashboard_sections(new_dirac_ew_su2_gevp_widgets),
        *build_gevp_mass_spectrum_sections(new_dirac_ew_su2_mass_spectrum_widgets),
        pn.pane.Markdown("### U(1) GEVP Analysis"),
        *build_gevp_dashboard_sections(new_dirac_ew_u1_gevp_widgets),
        *build_gevp_mass_spectrum_sections(new_dirac_ew_u1_mass_spectrum_widgets),
        pn.pane.Markdown("### EW Mixed GEVP Analysis"),
        *build_gevp_dashboard_sections(new_dirac_ew_ew_mixed_gevp_widgets),
        *build_gevp_mass_spectrum_sections(new_dirac_ew_ew_mixed_mass_spectrum_widgets),
        pn.pane.Markdown("### Electroweak Reference Masses (GeV)"),
        new_dirac_ew_ref_table,
        pn.pane.Markdown("### Channel Plots (Correlator + Effective Mass)"),
        new_dirac_ew_channel_plots,
        sizing_mode="stretch_both",
    )

    return ElectroweakSection(
        tab=new_dirac_ew_tab,
        status=new_dirac_ew_status,
        run_button=new_dirac_ew_run_button,
        settings=new_dirac_ew_settings,
        on_history_changed=_on_history_changed,
        summary=new_dirac_ew_summary,
        multiscale_summary=new_dirac_ew_multiscale_summary,
        su2_gevp_widgets=new_dirac_ew_su2_gevp_widgets,
        u1_gevp_widgets=new_dirac_ew_u1_gevp_widgets,
        ew_mixed_gevp_widgets=new_dirac_ew_ew_mixed_gevp_widgets,
        su2_mass_spectrum_widgets=new_dirac_ew_su2_mass_spectrum_widgets,
        u1_mass_spectrum_widgets=new_dirac_ew_u1_mass_spectrum_widgets,
        ew_mixed_mass_spectrum_widgets=new_dirac_ew_ew_mixed_mass_spectrum_widgets,
        filtered_summary=new_dirac_ew_filtered_summary,
        mass_table=new_dirac_ew_mass_table,
        filtered_mass_table=new_dirac_ew_filtered_mass_table,
        ratio_table=new_dirac_ew_ratio_table,
        fit_table=new_dirac_ew_fit_table,
        compare_table=new_dirac_ew_compare_table,
        anchor_table=new_dirac_ew_anchor_table,
        multiscale_table=new_dirac_ew_multiscale_table,
        multiscale_per_scale_table=new_dirac_ew_multiscale_per_scale_table,
    )
