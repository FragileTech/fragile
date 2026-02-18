"""Shared plot helpers for Atari and Robotic dashboards."""

import holoviews as hv
import numpy as np
import pandas as pd


def build_minmax_error_plot(
    *,
    step: list | np.ndarray,
    mean: list | np.ndarray,
    vmin: list | np.ndarray,
    vmax: list | np.ndarray,
    title: str,
    ylabel: str,
    color: str,
) -> hv.Overlay:
    """Mean line with asymmetric min/max error bars.

    Uses hv.ErrorBars with (mean, lower, upper) to show the range.
    """
    step = np.asarray(step, dtype=float)
    mean = np.asarray(mean, dtype=float)
    vmin = np.asarray(vmin, dtype=float)
    vmax = np.asarray(vmax, dtype=float)

    frame = pd.DataFrame({
        "step": step,
        "mean": mean,
        "lower": vmin,
        "upper": vmax,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if frame.empty:
        return hv.Text(0, 0, "No data").opts(height=320)

    curve = hv.Curve(frame, "step", "mean").opts(
        color=color, line_width=2, tools=["hover"],
    )
    errorbars = hv.ErrorBars(frame, "step", ["mean", "lower", "upper"]).opts(
        color=color, alpha=0.35, line_width=1,
    )
    return (errorbars * curve).opts(
        title=title,
        xlabel="Iteration",
        ylabel=ylabel,
        responsive=True,
        height=320,
        show_grid=True,
    )


def build_line_plot(
    *,
    step: list | np.ndarray,
    values: list | np.ndarray,
    title: str,
    ylabel: str,
    color: str,
) -> hv.Curve:
    """Simple line plot for clone % and alive count."""
    frame = pd.DataFrame({
        "step": np.asarray(step, dtype=float),
        "value": np.asarray(values, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if frame.empty:
        return hv.Text(0, 0, "No data").opts(height=320)

    return hv.Curve(frame, "step", "value").opts(
        title=title,
        xlabel="Iteration",
        ylabel=ylabel,
        color=color,
        line_width=2,
        tools=["hover"],
        responsive=True,
        height=320,
        show_grid=True,
    )
