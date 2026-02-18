"""Scatter-style correlator and effective mass plots for PipelineResult.

Each function produces a single-channel plot with scatter points (dots)
on a log-scale Y axis, matching the style of the old dashboard's
``ChannelPlot`` from ``fragile.physics.app.qft.plotting``.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import torch

from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.operators.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Channel colors
# ---------------------------------------------------------------------------

CHANNEL_COLORS: dict[str, str] = {
    "meson_scalar": "#4c78a8",
    "meson_pseudoscalar": "#72b7b2",
    "meson_score_directed": "#4c78a8",
    "meson_score_weighted": "#4c78a8",
    "meson_abs2_vacsub": "#4c78a8",
    "vector_full": "#f58518",
    "vector_longitudinal": "#e45756",
    "vector_transverse": "#9d755d",
    "vector_score_directed": "#f58518",
    "vector_score_gradient": "#f58518",
    "baryon_nucleon": "#54a24b",
    "glueball_plaquette": "#b279a2",
    "tensor_traceless": "#ff9da6",
    "meson_scalar_propagator": "#7b9ec9",
    "meson_pseudoscalar_propagator": "#a3d4d0",
    "vector_full_propagator": "#f9a84f",
    "axial_full_propagator": "#e8817f",
    "baryon_nucleon_propagator": "#7ec47b",
    "glueball_plaquette_propagator": "#c9a0bf",
    # Electroweak channels
    "u1_phase": "#1b9e77",
    "u1_dressed": "#d95f02",
    "u1_phase_q2": "#1b9e77",
    "u1_dressed_q2": "#d95f02",
    "su2_phase": "#7570b3",
    "su2_component": "#e7298a",
    "su2_doublet": "#66a61e",
    "su2_doublet_diff": "#e6ab02",
    "su2_phase_directed": "#9e9ac8",
    "su2_component_directed": "#f768a1",
    "su2_doublet_directed": "#a1d99b",
    "su2_doublet_diff_directed": "#fec44f",
    "ew_mixed": "#a6761d",
    "fitness_phase": "#666666",
    "clone_indicator": "#999999",
    "velocity_norm_cloner": "#e41a1c",
    "velocity_norm_resister": "#377eb8",
    "velocity_norm_persister": "#4daf4a",
}

_DEFAULT_COLOR = "#1f77b4"


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(float, copy=False)


# ---------------------------------------------------------------------------
# Single-channel correlator scatter plot
# ---------------------------------------------------------------------------


def build_single_correlator_plot(
    correlator: np.ndarray,
    channel_name: str,
    logy: bool = True,
) -> hv.Overlay | hv.Text:
    """Scatter plot of C(lag) for one channel.

    Args:
        correlator: 1-D array of correlator values, indexed by lag.
        channel_name: Channel name for title and color.
        logy: Use log-scale Y axis.

    Returns:
        HoloViews Overlay (scatter points), or placeholder if no data.
    """
    if logy:
        mask = np.isfinite(correlator) & (correlator > 0)
    else:
        mask = np.isfinite(correlator)
    if mask.sum() < 2:
        return _algorithm_placeholder_plot(f"No valid C(lag) data for {channel_name}")

    lags = np.arange(len(correlator))
    t_plot = lags[mask]
    c_plot = correlator[mask]

    color = CHANNEL_COLORS.get(channel_name, _DEFAULT_COLOR)
    scatter = (
        hv
        .Scatter((t_plot, c_plot), "lag", "C(lag)")
        .opts(color=color, size=6, alpha=0.8, tools=["hover"])
        .relabel(f"{channel_name} C(lag)")
    )

    opts_kwargs: dict[str, Any] = {
        "xlabel": "lag",
        "ylabel": "C(lag)",
        "title": f"{channel_name.replace('_', ' ').title()} Correlator",
        "width": 700,
        "height": 400,
        "show_legend": True,
        "show_grid": True,
        "shared_axes": False,
    }
    if logy:
        opts_kwargs["logy"] = True
        y_min = float(np.min(c_plot))
        y_max = float(np.max(c_plot))
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min and y_min > 0:
            opts_kwargs["ylim"] = (max(np.finfo(float).tiny, y_min * 0.8), y_max * 1.2)

    return scatter.opts(**opts_kwargs)


# ---------------------------------------------------------------------------
# Single-channel effective mass scatter plot
# ---------------------------------------------------------------------------


def build_single_effective_mass_plot(
    correlator: np.ndarray,
    channel_name: str,
) -> hv.Overlay | hv.Text:
    """Scatter plot of m_eff(t) = log(C(t)/C(t+1)) for one channel.

    Args:
        correlator: 1-D array of correlator values, indexed by lag.
        channel_name: Channel name for title and color.

    Returns:
        HoloViews Overlay (scatter points), or placeholder if no data.
    """
    if len(correlator) < 2:
        return _algorithm_placeholder_plot(f"Not enough data for {channel_name} m_eff")

    c0 = correlator[:-1]
    c1 = correlator[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((c0 > 0) & (c1 > 0), c0 / c1, np.nan)
        m_eff = np.log(ratio)

    mask = np.isfinite(m_eff) & (m_eff > 0)
    if mask.sum() < 2:
        return _algorithm_placeholder_plot(f"No valid m_eff data for {channel_name}")

    lags = np.arange(len(m_eff))
    t_plot = lags[mask]
    m_plot = m_eff[mask]

    color = CHANNEL_COLORS.get(channel_name, _DEFAULT_COLOR)
    scatter = (
        hv
        .Scatter((t_plot, m_plot), "t", "m_eff(t)")
        .opts(color=color, size=6, alpha=0.8, tools=["hover"])
        .relabel(f"{channel_name} m_eff")
    )

    y_lo, y_hi = float(np.min(m_plot)), float(np.max(m_plot))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05 * max(abs(y_hi), 1.0)

    return scatter.opts(
        xlabel="t (lag)",
        ylabel="m_eff(t)",
        title=f"{channel_name.replace('_', ' ').title()} Effective Mass",
        width=700,
        height=400,
        show_legend=True,
        show_grid=True,
        shared_axes=False,
        ylim=(y_lo - margin, y_hi + margin),
    )


# ---------------------------------------------------------------------------
# Single-channel operator time series scatter plot
# ---------------------------------------------------------------------------


def build_single_operator_series_plot(
    series: np.ndarray,
    channel_name: str,
) -> hv.Overlay | hv.Text:
    """Scatter plot of operator value vs frame index for one channel.

    Args:
        series: 1-D array of per-frame operator values.
        channel_name: Channel name for title and color.

    Returns:
        HoloViews Scatter, or placeholder if no data.
    """
    mask = np.isfinite(series)
    if mask.sum() < 2:
        return _algorithm_placeholder_plot(f"No valid operator data for {channel_name}")

    frames = np.arange(len(series))
    color = CHANNEL_COLORS.get(channel_name, _DEFAULT_COLOR)
    scatter = (
        hv
        .Scatter((frames[mask], series[mask]), "frame", "value")
        .opts(color=color, size=4, alpha=0.6, tools=["hover"])
        .relabel(channel_name)
    )

    return scatter.opts(
        xlabel="Frame index",
        ylabel="Operator value",
        title=f"{channel_name.replace('_', ' ').title()} Operator Series",
        width=700,
        height=350,
        show_grid=True,
        shared_axes=False,
    )


# ---------------------------------------------------------------------------
# All-channel overlay (scatter)
# ---------------------------------------------------------------------------


def build_all_channels_correlator_overlay(
    result: PipelineResult,
    logy: bool = True,
    scale_index: int = 0,
) -> hv.Overlay | hv.Text:
    """Overlay scatter plot of C(lag) for all channels."""
    if not result.correlators:
        return _algorithm_placeholder_plot("No correlator data")

    scatters: list[Any] = []
    y_min_positive = float("inf")
    y_max_positive = 0.0

    for name, corr in result.correlators.items():
        if corr.numel() == 0:
            continue
        arr = _to_numpy(corr)
        if arr.ndim == 2:
            idx = min(scale_index, arr.shape[0] - 1)
            arr = arr[idx]

        if logy:
            mask = np.isfinite(arr) & (arr > 0)
        else:
            mask = np.isfinite(arr)
        if mask.sum() < 2:
            continue

        lags = np.arange(len(arr))
        t_plot = lags[mask]
        c_plot = arr[mask]

        if logy:
            y_min_positive = min(y_min_positive, float(np.min(c_plot)))
            y_max_positive = max(y_max_positive, float(np.max(c_plot)))

        color = CHANNEL_COLORS.get(name, _DEFAULT_COLOR)
        scatters.append(
            hv
            .Scatter((t_plot, c_plot), "lag", "C(lag)")
            .opts(color=color, size=5, alpha=0.7, tools=["hover"])
            .relabel(name)
        )

    if not scatters:
        return _algorithm_placeholder_plot("All correlator channels are empty")

    overlay = scatters[0]
    for s in scatters[1:]:
        overlay = overlay * s

    opts_kwargs: dict[str, Any] = {
        "logy": logy,
        "xlabel": "lag",
        "ylabel": "C(lag)",
        "title": "All Channel Correlators",
        "width": 960,
        "height": 400,
        "show_legend": True,
        "show_grid": True,
        "shared_axes": False,
    }
    if (
        logy
        and np.isfinite(y_min_positive)
        and y_min_positive > 0
        and y_max_positive > y_min_positive
    ):
        opts_kwargs["ylim"] = (
            max(np.finfo(float).tiny, y_min_positive * 0.8),
            y_max_positive * 1.2,
        )

    return overlay.opts(**opts_kwargs)


def build_all_channels_meff_overlay(
    result: PipelineResult,
    scale_index: int = 0,
) -> hv.Overlay | hv.Text:
    """Overlay scatter plot of m_eff(t) for all channels."""
    if not result.correlators:
        return _algorithm_placeholder_plot("No correlator data for effective mass")

    scatters: list[Any] = []
    for name, corr in result.correlators.items():
        if corr.numel() == 0:
            continue
        arr = _to_numpy(corr)
        if arr.ndim == 2:
            idx = min(scale_index, arr.shape[0] - 1)
            arr = arr[idx]
        if len(arr) < 2:
            continue

        c0 = arr[:-1]
        c1 = arr[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((c0 > 0) & (c1 > 0), c0 / c1, np.nan)
            m_eff = np.log(ratio)

        mask = np.isfinite(m_eff) & (m_eff > 0)
        if mask.sum() < 2:
            continue

        lags = np.arange(len(m_eff))
        color = CHANNEL_COLORS.get(name, _DEFAULT_COLOR)
        scatters.append(
            hv
            .Scatter((lags[mask], m_eff[mask]), "t", "m_eff(t)")
            .opts(color=color, size=5, alpha=0.7, tools=["hover"])
            .relabel(name)
        )

    if not scatters:
        return _algorithm_placeholder_plot("No valid effective mass data")

    overlay = scatters[0]
    for s in scatters[1:]:
        overlay = overlay * s

    return overlay.opts(
        xlabel="t (lag)",
        ylabel="m_eff(t)",
        title="All Channel Effective Masses",
        width=960,
        height=400,
        show_legend=True,
        show_grid=True,
        shared_axes=False,
    )


# ---------------------------------------------------------------------------
# Helpers for extracting 1-D arrays from PipelineResult
# ---------------------------------------------------------------------------


def get_correlator_array(
    result: PipelineResult,
    channel: str,
    scale_index: int = 0,
) -> np.ndarray:
    """Extract 1-D correlator array for a channel."""
    corr = result.correlators.get(channel)
    if corr is None or corr.numel() == 0:
        return np.array([])
    arr = _to_numpy(corr)
    if arr.ndim == 2:
        idx = min(scale_index, arr.shape[0] - 1)
        arr = arr[idx]
    return arr


def get_operator_series_array(
    result: PipelineResult,
    channel: str,
    scale_index: int = 0,
) -> np.ndarray:
    """Extract 1-D operator time-series array for a channel."""
    series = result.operators.get(channel)
    if series is None or series.numel() == 0:
        return np.array([])
    arr = _to_numpy(series)
    if arr.ndim == 3:
        idx = min(scale_index, arr.shape[0] - 1)
        arr = arr[idx].mean(axis=-1)
    elif arr.ndim == 2:
        if result.scales is not None and arr.shape[0] <= 32:
            idx = min(scale_index, arr.shape[0] - 1)
            arr = arr[idx]
        else:
            arr = arr.mean(axis=-1)
    return arr


# ---------------------------------------------------------------------------
# Summary / correlator tables (shared by strong & electroweak tabs)
# ---------------------------------------------------------------------------


def build_summary_table(
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


def build_correlator_table(
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
# Per-channel-group correlator / effective-mass plots
# ---------------------------------------------------------------------------

_VARIANT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

# Strong-force channel prefixes → display names.
_STRONG_PREFIX_DISPLAY: dict[str, str] = {
    "axial_vector": "Axial Vector",
    "pseudoscalar": "Pseudoscalar",
    "scalar": "Scalar",
    "vector": "Vector",
    "nucleon": "Nucleon",
    "glueball": "Glueball",
    "tensor": "Tensor",
}

_STRONG_GROUP_ORDER = [
    "Scalar", "Pseudoscalar", "Vector", "Axial Vector",
    "Nucleon", "Glueball", "Tensor",
]


def group_strong_correlator_keys(keys) -> dict[str, list[str]]:
    """Group strong-force correlator keys by channel prefix.

    Returns an ordered dict of ``{display_name: [key, ...]}``.
    """
    groups: dict[str, list[str]] = {}
    for key in keys:
        if key.startswith("glueball_momentum_"):
            groups.setdefault("Glueball", []).append(key)
            continue
        matched = False
        for prefix, display in _STRONG_PREFIX_DISPLAY.items():
            if key.startswith(prefix + "_") or key == prefix:
                groups.setdefault(display, []).append(key)
                matched = True
                break
        if not matched:
            groups.setdefault("Other", []).append(key)

    ordered: dict[str, list[str]] = {}
    for name in _STRONG_GROUP_ORDER:
        if name in groups:
            ordered[name] = sorted(groups.pop(name))
    for name in sorted(groups):
        ordered[name] = sorted(groups[name])
    return ordered


_EW_GROUP_ORDER = [
    "U(1)", "SU(2) Base", "SU(2) Directed", "SU(2) Walker-Type",
    "EW Mixed", "Symmetry Breaking", "Parity Velocity",
]


def group_electroweak_correlator_keys(keys) -> dict[str, list[str]]:
    """Group electroweak correlator keys by family.

    Returns an ordered dict of ``{display_name: [key, ...]}``.
    """
    _rules = [
        ("U(1)", lambda k: k.startswith("u1_")),
        ("SU(2) Directed", lambda k: k.startswith("su2_") and k.endswith("_directed")),
        (
            "SU(2) Walker-Type",
            lambda k: k.startswith("su2_")
            and any(k.endswith(s) for s in ("_cloner", "_resister", "_persister")),
        ),
        ("SU(2) Base", lambda k: k.startswith("su2_")),
        ("EW Mixed", lambda k: k == "ew_mixed"),
        ("Symmetry Breaking", lambda k: k in ("fitness_phase", "clone_indicator")),
        ("Parity Velocity", lambda k: k.startswith("velocity_norm_")),
    ]
    groups: dict[str, list[str]] = {}
    for key in keys:
        matched = False
        for group_name, predicate in _rules:
            if predicate(key):
                groups.setdefault(group_name, []).append(key)
                matched = True
                break
        if not matched:
            groups.setdefault("Other", []).append(key)

    ordered: dict[str, list[str]] = {}
    for name in _EW_GROUP_ORDER:
        if name in groups:
            ordered[name] = sorted(groups.pop(name))
    for name in sorted(groups):
        ordered[name] = sorted(groups[name])
    return ordered


def build_grouped_correlator_plot(
    result: PipelineResult,
    group_name: str,
    keys: list[str],
    scale_index: int = 0,
    logy: bool = True,
) -> hv.Overlay | hv.Text:
    """Build a correlator overlay for one channel group."""
    scatters: list[Any] = []
    y_min_pos = float("inf")
    y_max_pos = 0.0

    for idx, name in enumerate(keys):
        arr = get_correlator_array(result, name, scale_index)
        if len(arr) == 0:
            continue
        mask = (np.isfinite(arr) & (arr > 0)) if logy else np.isfinite(arr)
        if mask.sum() < 2:
            continue

        lags = np.arange(len(arr))
        t_plot, c_plot = lags[mask], arr[mask]
        if logy:
            y_min_pos = min(y_min_pos, float(np.min(c_plot)))
            y_max_pos = max(y_max_pos, float(np.max(c_plot)))

        color = _VARIANT_PALETTE[idx % len(_VARIANT_PALETTE)]
        scatters.append(
            hv.Scatter((t_plot, c_plot), "lag", "C(lag)")
            .opts(color=color, size=5, alpha=0.7, tools=["hover"])
            .relabel(name)
        )

    if not scatters:
        return _algorithm_placeholder_plot(f"No correlator data for {group_name}")

    overlay = scatters[0]
    for s in scatters[1:]:
        overlay = overlay * s

    opts_kw: dict[str, Any] = {
        "logy": logy,
        "xlabel": "lag",
        "ylabel": "C(lag)",
        "title": f"{group_name} Correlator",
        "width": 500,
        "height": 350,
        "show_legend": True,
        "show_grid": True,
        "shared_axes": False,
    }
    if (
        logy
        and np.isfinite(y_min_pos)
        and y_min_pos > 0
        and y_max_pos > y_min_pos
    ):
        opts_kw["ylim"] = (
            max(np.finfo(float).tiny, y_min_pos * 0.8),
            y_max_pos * 1.2,
        )
    return overlay.opts(**opts_kw)


def build_grouped_meff_plot(
    result: PipelineResult,
    group_name: str,
    keys: list[str],
    scale_index: int = 0,
) -> hv.Overlay | hv.Text:
    """Build an effective-mass overlay for one channel group."""
    scatters: list[Any] = []

    for idx, name in enumerate(keys):
        arr = get_correlator_array(result, name, scale_index)
        if len(arr) < 2:
            continue
        c0, c1 = arr[:-1], arr[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((c0 > 0) & (c1 > 0), c0 / c1, np.nan)
            m_eff = np.log(ratio)

        mask = np.isfinite(m_eff) & (m_eff > 0)
        if mask.sum() < 2:
            continue

        lags = np.arange(len(m_eff))
        color = _VARIANT_PALETTE[idx % len(_VARIANT_PALETTE)]
        scatters.append(
            hv.Scatter((lags[mask], m_eff[mask]), "t", "m_eff(t)")
            .opts(color=color, size=5, alpha=0.7, tools=["hover"])
            .relabel(name)
        )

    if not scatters:
        return _algorithm_placeholder_plot(f"No m_eff data for {group_name}")

    overlay = scatters[0]
    for s in scatters[1:]:
        overlay = overlay * s

    return overlay.opts(
        xlabel="t (lag)",
        ylabel="m_eff(t)",
        title=f"{group_name} Effective Mass",
        width=500,
        height=350,
        show_legend=True,
        show_grid=True,
        shared_axes=False,
    )
