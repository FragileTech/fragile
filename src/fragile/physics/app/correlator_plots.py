"""Scatter-style correlator and effective mass plots for PipelineResult.

Each function produces a single-channel plot with scatter points (dots)
on a log-scale Y axis, matching the style of the old dashboard's
``ChannelPlot`` from ``fragile.fractalai.qft.plotting``.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
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
        .opts(color=color, size=6, alpha=0.8)
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
        .opts(color=color, size=6, alpha=0.8)
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
        .opts(color=color, size=4, alpha=0.6)
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
            .opts(color=color, size=5, alpha=0.7)
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
            .opts(color=color, size=5, alpha=0.7)
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
