"""Plotting helpers for QFT analysis outputs.

2D plots are rendered with Holoviews using the Bokeh backend.
3D plots are rendered with Plotly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import holoviews as hv
import numpy as np
import plotly.graph_objects as go
import torch


hv.extension("bokeh")


def _ensure_html_path(path: Path) -> Path:
    if path.suffix.lower() != ".html":
        return path.with_suffix(".html")
    return path


def _save_holoviews(plot: Any, path: Path) -> Path:
    html_path = _ensure_html_path(path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    hv.save(plot, str(html_path), backend="bokeh")
    return html_path


def _save_plotly(fig: go.Figure, path: Path) -> Path:
    html_path = _ensure_html_path(path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    return html_path


def build_correlation_decay_plot(
    r: np.ndarray,
    C: np.ndarray,
    counts: np.ndarray,
    fit: dict[str, float],
    title: str,
) -> hv.Overlay | None:
    mask = (counts > 0) & (C > 0)
    if mask.sum() < 2:
        return None

    r_plot = r[mask]
    c_plot = C[mask]

    scatter = hv.Scatter((r_plot, c_plot), "r", "C(r)").relabel("C(r)")
    overlays = scatter

    if fit.get("xi", 0.0) > 0 and fit.get("C0", 0.0) > 0:
        r_line = np.linspace(float(r_plot.min()), float(r_plot.max()), 200)
        c_line = fit["C0"] * np.exp(-(r_line**2) / (fit["xi"] ** 2))
        curve = hv.Curve((r_line, c_line), "r", "C(r)").relabel("exp fit")
        overlays = overlays * curve

    return overlays.opts(
        logy=True,
        xlabel="r",
        ylabel="C(r)",
        title=title,
        width=600,
        height=400,
        legend_position="top_left",
    )


def plot_correlation_decay(
    r: np.ndarray,
    C: np.ndarray,
    counts: np.ndarray,
    fit: dict[str, float],
    title: str,
    path: Path,
) -> Path | None:
    plot = build_correlation_decay_plot(r, C, counts, fit, title)
    if plot is None:
        return None
    return _save_holoviews(plot, path)


def build_lyapunov_plot(
    time: np.ndarray,
    V_total: np.ndarray,
    V_var_x: np.ndarray,
    V_var_v: np.ndarray,
) -> hv.Overlay:
    curve_total = hv.Curve((time, V_total), "step", "Lyapunov").relabel("V_total")
    curve_var_x = hv.Curve((time, V_var_x), "step", "Lyapunov").relabel("V_var_x")
    curve_var_v = hv.Curve((time, V_var_v), "step", "Lyapunov").relabel("V_var_v")

    return (curve_total * curve_var_x * curve_var_v).opts(
        logy=True,
        xlabel="step",
        ylabel="Lyapunov",
        title="Lyapunov Components",
        width=600,
        height=400,
        legend_position="top_left",
    )


def plot_lyapunov(
    time: np.ndarray,
    V_total: np.ndarray,
    V_var_x: np.ndarray,
    V_var_v: np.ndarray,
    path: Path,
) -> Path | None:
    plot = build_lyapunov_plot(time, V_total, V_var_x, V_var_v)
    return _save_holoviews(plot, path)


def build_phase_histograms_plot(
    u1_phases: torch.Tensor,
    su2_phases: torch.Tensor,
    alive: torch.Tensor,
) -> hv.Overlay | None:
    u1_vals = u1_phases[alive].cpu().numpy()
    su2_vals = su2_phases[alive].cpu().numpy()
    if u1_vals.size == 0 or su2_vals.size == 0:
        return None

    u1_hist = np.histogram(u1_vals, bins=50)
    su2_hist = np.histogram(su2_vals, bins=50)

    u1_plot = hv.Histogram(u1_hist).relabel("U1 phase").opts(alpha=0.6)
    su2_plot = hv.Histogram(su2_hist).relabel("SU2 phase").opts(alpha=0.6)

    return (u1_plot * su2_plot).opts(
        xlabel="phase",
        ylabel="count",
        title="Gauge Phase Distributions",
        width=600,
        height=400,
        legend_position="top_left",
    )


def plot_phase_histograms(
    u1_phases: torch.Tensor,
    su2_phases: torch.Tensor,
    alive: torch.Tensor,
    path: Path,
) -> Path | None:
    plot = build_phase_histograms_plot(u1_phases, su2_phases, alive)
    if plot is None:
        return None
    return _save_holoviews(plot, path)


def build_wilson_histogram_plot(values: np.ndarray, title: str) -> hv.Histogram | None:
    if values.size == 0:
        return None
    hist = np.histogram(values, bins=50)
    return (
        hv.Histogram(hist)
        .relabel("Wilson loop")
        .opts(
            xlabel="Wilson loop (Re)",
            ylabel="count",
            title=title,
            width=600,
            height=400,
        )
    )


def plot_wilson_histogram(values: np.ndarray, title: str, path: Path) -> Path | None:
    plot = build_wilson_histogram_plot(values, title)
    if plot is None:
        return None
    return _save_holoviews(plot, path)


def build_wilson_timeseries_plot(
    time_index: np.ndarray, action_mean: np.ndarray
) -> hv.Curve | None:
    if time_index.size == 0:
        return None
    return hv.Curve((time_index, action_mean), "time index", "Wilson action").opts(
        xlabel="time index",
        ylabel="Wilson action (mean)",
        title="Wilson Action Over Time",
        width=600,
        height=400,
    )


def plot_wilson_timeseries(
    time_index: np.ndarray, action_mean: np.ndarray, path: Path
) -> Path | None:
    plot = build_wilson_timeseries_plot(time_index, action_mean)
    if plot is None:
        return None
    return _save_holoviews(plot, path)


def plot_scatter3d(points: np.ndarray, title: str, path: Path) -> Path | None:
    if points.size == 0:
        return None
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 3, "opacity": 0.8},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
        width=700,
        height=500,
    )
    return _save_plotly(fig, path)
