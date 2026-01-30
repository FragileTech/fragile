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


# =============================================================================
# Channel Correlator Plotting
# =============================================================================

# Consistent colors for channel plots
CHANNEL_COLORS = {
    "scalar": "#1f77b4",        # Blue
    "pseudoscalar": "#ff7f0e",  # Orange
    "vector": "#2ca02c",        # Green
    "axial_vector": "#d62728",  # Red
    "tensor": "#9467bd",        # Purple
    "nucleon": "#8c564b",       # Brown
    "glueball": "#e377c2",      # Pink
    # Electroweak proxy channels
    "u1_phase": "#17becf",      # Cyan
    "u1_dressed": "#bcbd22",    # Olive
    "u1_phase_q2": "#00b5ad",   # Teal
    "u1_dressed_q2": "#c7d11b", # Yellow-green
    "su2_phase": "#ff9896",     # Light red
    "su2_component": "#ff7b7b", # Soft red
    "su2_doublet": "#c49c94",   # Tan
    "su2_doublet_diff": "#b07c7c", # Dusky red
    "ew_mixed": "#7f7f7f",      # Gray
}


def build_correlator_plot(
    lag_times: np.ndarray,
    correlator: np.ndarray,
    mass_fit: dict[str, Any],
    channel_name: str,
    logy: bool = True,
) -> hv.Overlay | None:
    """Build correlator C(t) plot with exponential fit overlay.

    Args:
        lag_times: Time lag values [T].
        correlator: Correlator C(t) values [T].
        mass_fit: Dict with mass, mass_error from AIC fitting.
        channel_name: Name of the channel for labeling.
        logy: Use logarithmic y-axis.

    Returns:
        HoloViews Overlay with scatter and fit curve, or None if no data.
    """
    mask = (correlator > 0) & np.isfinite(correlator)
    if mask.sum() < 2:
        return None

    t_plot = lag_times[mask]
    c_plot = correlator[mask]

    color = CHANNEL_COLORS.get(channel_name, "#1f77b4")
    scatter = hv.Scatter(
        (t_plot, c_plot), "t", "C(t)"
    ).opts(
        color=color,
        size=6,
        alpha=0.8,
    ).relabel(f"{channel_name} C(t)")

    overlays = scatter

    mass = mass_fit.get("mass", 0.0)
    best_window = mass_fit.get("best_window", {})
    t_start = best_window.get("t_start", 0)
    dt_scale = float(t_plot[1] - t_plot[0]) if len(t_plot) > 1 else 1.0
    mass_scaled = mass / dt_scale if mass > 0 and dt_scale > 0 else 0.0

    if mass_scaled > 0:
        # Exponential fit: C(t) = C(t_start) * exp(-m*(t - t_start))
        # Anchor at best window start, not t=0 (which is often an outlier)
        t_line = np.linspace(float(t_plot.min()), float(t_plot.max()), 200)

        # Find correlator value at t_start for anchoring
        t_anchor = t_start * dt_scale
        # Find closest point to t_anchor in the data
        anchor_idx = np.argmin(np.abs(t_plot - t_anchor))
        c_anchor = c_plot[anchor_idx] if anchor_idx < len(c_plot) else c_plot[0]

        if c_anchor > 0:
            c_line = c_anchor * np.exp(-mass_scaled * (t_line - t_anchor))
            curve = hv.Curve(
                (t_line, c_line), "t", "C(t)"
            ).opts(
                color=color,
                line_dash="dashed",
                line_width=2,
            ).relabel(f"fit m={mass_scaled:.4f}")
            overlays = overlays * curve

    return overlays.opts(
        logy=logy,
        xlabel="t (time lag)",
        ylabel="C(t)",
        title=f"{channel_name.replace('_', ' ').title()} Correlator",
        width=600,
        height=350,
        legend_position="top_right",
    )


def build_effective_mass_plot(
    lag_times: np.ndarray,
    effective_mass: np.ndarray,
    mass_fit: dict[str, Any],
    channel_name: str,
) -> hv.Overlay | None:
    """Build effective mass m_eff(t) plot with fitted mass line.

    Args:
        lag_times: Time lag values [T-1].
        effective_mass: Effective mass m_eff(t) values [T-1].
        mass_fit: Dict with mass, mass_error from AIC fitting.
        channel_name: Name of the channel for labeling.

    Returns:
        HoloViews Overlay with scatter and plateau line, or None if no data.
    """
    mask = np.isfinite(effective_mass) & (effective_mass > 0)
    if mask.sum() < 2:
        return None

    t_plot = lag_times[mask]
    m_plot = effective_mass[mask]

    color = CHANNEL_COLORS.get(channel_name, "#1f77b4")
    scatter = hv.Scatter(
        (t_plot, m_plot), "t", "m_eff(t)"
    ).opts(
        color=color,
        size=6,
        alpha=0.8,
    ).relabel(f"{channel_name} m_eff")

    overlays = scatter

    mass = mass_fit.get("mass", 0.0)
    mass_error = mass_fit.get("mass_error", 0.0)
    if mass > 0:
        # Horizontal line at fitted mass
        t_line = np.array([float(t_plot.min()), float(t_plot.max())])
        m_line = np.array([mass, mass])
        curve = hv.Curve(
            (t_line, m_line), "t", "m_eff(t)"
        ).opts(
            color=color,
            line_dash="dashed",
            line_width=2,
        ).relabel(f"m={mass:.4f}±{mass_error:.4f}")
        overlays = overlays * curve

        # Error band
        if mass_error > 0 and mass_error < mass:
            band = hv.Area(
                (t_line, [mass - mass_error, mass - mass_error], [mass + mass_error, mass + mass_error]),
                kdims=["t"],
                vdims=["lower", "upper"],
            ).opts(
                color=color,
                alpha=0.2,
            )
            overlays = overlays * band

    return overlays.opts(
        xlabel="t (time lag)",
        ylabel="m_eff(t)",
        title=f"{channel_name.replace('_', ' ').title()} Effective Mass",
        width=600,
        height=350,
        legend_position="top_right",
    )


def build_mass_spectrum_bar(
    channel_results: dict[str, Any],
) -> hv.Bars | None:
    """Build bar chart of extracted masses across channels.

    Args:
        channel_results: Dict mapping channel names to ChannelCorrelatorResult.

    Returns:
        HoloViews Bars chart, or None if no valid masses.
    """
    bars_data = []
    for name, result in channel_results.items():
        if result.n_samples == 0:
            continue
        mass = result.mass_fit.get("mass", 0.0)
        mass_error = result.mass_fit.get("mass_error", float("inf"))
        if mass > 0 and mass_error < float("inf"):
            bars_data.append((name, mass, mass_error))

    if not bars_data:
        return None

    # Sort by mass
    bars_data.sort(key=lambda x: x[1])
    names = [d[0] for d in bars_data]
    masses = [d[1] for d in bars_data]
    errors = [d[2] for d in bars_data]
    colors = [CHANNEL_COLORS.get(n, "#1f77b4") for n in names]

    bars = hv.Bars(
        list(zip(names, masses)),
        kdims=["channel"],
        vdims=["mass"],
    ).opts(
        xlabel="Channel",
        ylabel="Mass (algorithmic units)",
        title="Channel Mass Spectrum",
        width=600,
        height=350,
        color=hv.dim("channel").categorize(dict(zip(names, colors))),
        xrotation=45,
    )

    # Add error bars using Scatter
    error_data = [(n, m, e) for n, m, e in zip(names, masses, errors)]
    errorbars = hv.ErrorBars(
        error_data,
        kdims=["channel"],
        vdims=["mass", "error"],
    ).opts(
        color="black",
        line_width=2,
    )

    return bars * errorbars


def build_window_heatmap(
    window_masses: np.ndarray,
    window_aic: np.ndarray,
    window_widths: list[int],
    best_window: dict[str, Any],
    channel_name: str,
    window_r2: np.ndarray | None = None,
    color_metric: str = "mass",
    alpha_metric: str = "aic",
) -> hv.Overlay | None:
    """Build window heatmap with mass as color and AIC weight as alpha.

    Creates a 2D heatmap where:
    - X-axis: window start index
    - Y-axis: window end index (= start + width - 1)
    - Color: selected metric (mass, AIC, or R²)
    - Alpha: selected metric (mass, AIC, or R²) normalized to [0.2, 1.0]

    Args:
        window_masses: Mass values [num_widths, max_positions].
        window_aic: AIC values [num_widths, max_positions].
        window_widths: List of window widths used.
        best_window: Dict with width, t_start for best window.
        channel_name: Name of the channel for labeling.
        window_r2: Optional R² values [num_widths, max_positions].
        color_metric: Metric for color mapping ("mass", "aic", "r2").
        alpha_metric: Metric for alpha mapping ("mass", "aic", "r2").

    Returns:
        HoloViews Overlay with heatmap and best window marker, or None if no data.
    """
    if window_masses is None or window_aic is None or len(window_widths) == 0:
        return None
    if color_metric not in {"mass", "aic", "r2"}:
        color_metric = "mass"
    if alpha_metric not in {"mass", "aic", "r2"}:
        alpha_metric = "aic"

    # Convert (width_idx, start) grid to (start, end) coordinates
    records = []
    num_widths, max_pos = window_masses.shape

    for w_idx, width in enumerate(window_widths):
        for start in range(max_pos):
            mass = float(window_masses[w_idx, start])
            aic = float(window_aic[w_idx, start])
            r2_val = float(window_r2[w_idx, start]) if window_r2 is not None else float("nan")

            # Skip invalid entries
            if not np.isfinite(mass) or not np.isfinite(aic) or mass <= 0:
                continue
            if aic == float("inf"):
                continue

            end = start + width - 1
            records.append({
                "start": start,
                "end": end,
                "mass": mass,
                "aic": aic,
                "width": width,
                "r2": r2_val,
            })

    if not records:
        return None

    # Convert to numpy arrays for vectorized operations
    starts = np.array([r["start"] for r in records])
    ends = np.array([r["end"] for r in records])
    masses = np.array([r["mass"] for r in records])
    aics = np.array([r["aic"] for r in records])
    widths = np.array([r["width"] for r in records])
    r2_vals = np.array([r["r2"] for r in records])
    if window_r2 is None and (color_metric == "r2" or alpha_metric == "r2"):
        color_metric = "mass" if color_metric == "r2" else color_metric
        alpha_metric = "mass" if alpha_metric == "r2" else alpha_metric

    def _alpha_from_values(values: np.ndarray, alpha_min: float = 0.2) -> np.ndarray:
        finite = np.isfinite(values)
        if not finite.any():
            return np.full_like(values, alpha_min, dtype=float)
        vmin = np.nanmin(values[finite])
        vmax = np.nanmax(values[finite])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-12:
            return np.full_like(values, (alpha_min + 1.0) * 0.5, dtype=float)
        norm = (values - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        return alpha_min + (1.0 - alpha_min) * norm

    # Precompute AIC weights (lower AIC => higher alpha)
    aic_min = np.nanmin(aics)
    aic_weights = np.exp(-0.5 * (aics - aic_min))
    aic_weights = np.nan_to_num(aic_weights, nan=0.0)

    alpha_min = 0.2
    if alpha_metric == "aic":
        max_weight = np.nanmax(aic_weights) if np.nanmax(aic_weights) > 0 else 1.0
        weights_norm = aic_weights / max_weight
        alpha = alpha_min + (1.0 - alpha_min) * weights_norm
    elif alpha_metric == "r2":
        alpha = _alpha_from_values(r2_vals, alpha_min=alpha_min)
    else:  # mass
        alpha = _alpha_from_values(masses, alpha_min=alpha_min)

    # Build DataFrame for HoloViews
    import pandas as pd
    df = pd.DataFrame({
        "start": starts,
        "end": ends,
        "mass": masses,
        "alpha": alpha,
        "width": widths,
        "r2": r2_vals,
        "aic": aics,
    })

    # Create the heatmap using Points with color and alpha mapping
    # Using Scatter/Points allows per-point alpha control
    points = hv.Points(
        df,
        kdims=["start", "end"],
        vdims=["mass", "aic", "r2", "alpha", "width"],
    ).opts(
        color=color_metric,
        alpha="alpha",
        cmap="viridis",
        colorbar=True,
        size=8,
        marker="square",
        tools=["hover"],
        width=500,
        height=450,
        xlabel="Window Start",
        ylabel="Window End",
        title=f"{channel_name.replace('_', ' ').title()} Window Heatmap",
        clabel=color_metric.upper(),
    )

    overlays = points

    # Mark best window with a red marker
    best_t_start = best_window.get("t_start", -1)
    best_width = best_window.get("width", 0)
    if best_t_start >= 0 and best_width > 0:
        best_end = best_t_start + best_width - 1
        best_mass = best_window.get("mass", 0.0)
        best_marker = hv.Points(
            [(best_t_start, best_end, best_mass)],
            kdims=["start", "end"],
            vdims=["mass"],
        ).opts(
            color="red",
            size=15,
            marker="circle",
            line_color="white",
            line_width=2,
            alpha=1.0,
        ).relabel("Best window")
        overlays = overlays * best_marker

    return overlays


def build_effective_mass_plateau_plot(
    lag_times: np.ndarray,
    correlator: np.ndarray,
    effective_mass: np.ndarray,
    mass_fit: dict[str, Any],
    channel_name: str,
    dt: float = 1.0,
) -> tuple[hv.Element, hv.Element] | None:
    """Build 2-panel correlator + effective mass plateau plot.

    Left panel: C(t) with log scale, scatter + fit line
    Right panel: Effective mass with best window shading and error band

    Args:
        lag_times: Time lag values for correlator [T].
        correlator: Correlator C(t) values [T].
        effective_mass: Effective mass m_eff(t) values [T-1].
        mass_fit: Dict with mass, mass_error, best_window from AIC fitting.
        channel_name: Name of the channel for labeling.
        dt: Time step for scaling.

    Returns:
        Tuple of (left_panel, right_panel) or None if insufficient data.
    """
    # Filter valid correlator data
    corr_mask = (correlator > 0) & np.isfinite(correlator)
    if corr_mask.sum() < 2:
        return None

    # Filter valid effective mass data
    meff_mask = np.isfinite(effective_mass) & (effective_mass > 0)
    if meff_mask.sum() < 2:
        return None

    color = CHANNEL_COLORS.get(channel_name, "#1f77b4")
    mass = mass_fit.get("mass", 0.0)
    mass_error = mass_fit.get("mass_error", 0.0)
    best_window = mass_fit.get("best_window", {})
    t_start = best_window.get("t_start", 0)
    width = best_window.get("width", 0)
    dt_scale = dt if dt > 0 else 1.0
    mass_scaled = mass / dt_scale if mass > 0 else 0.0
    mass_error_scaled = mass_error / dt_scale if mass_error > 0 else 0.0

    # Left panel: Correlator C(t)
    t_corr = lag_times[corr_mask]
    c_plot = correlator[corr_mask]

    corr_scatter = hv.Scatter(
        (t_corr, c_plot), "t", "C(t)"
    ).opts(
        color=color,
        size=6,
        alpha=0.8,
    ).relabel("C(t)")

    left_overlays = corr_scatter

    if mass_scaled > 0:
        # Anchor fit at best window start, not t=0
        t_line = np.linspace(float(t_corr.min()), float(t_corr.max()), 200)
        t_anchor = t_start * dt_scale
        # Find closest point to t_anchor in the data
        anchor_idx = np.argmin(np.abs(t_corr - t_anchor))
        c_anchor = c_plot[anchor_idx] if anchor_idx < len(c_plot) else c_plot[0]

        if c_anchor > 0:
            c_line = c_anchor * np.exp(-mass_scaled * (t_line - t_anchor))
            corr_curve = hv.Curve(
                (t_line, c_line), "t", "C(t)"
            ).opts(
                color=color,
                line_dash="dashed",
                line_width=2,
            ).relabel(f"fit m={mass:.4f}")
            left_overlays = left_overlays * corr_curve

    left_panel = left_overlays.opts(
        logy=True,
        xlabel="t (time lag)",
        ylabel="C(t)",
        title=f"{channel_name.replace('_', ' ').title()} Correlator",
        width=400,
        height=350,
        legend_position="top_right",
    )

    # Right panel: Effective mass plateau
    meff_times = lag_times[:len(effective_mass)]
    t_meff = meff_times[meff_mask]
    m_plot = effective_mass[meff_mask]

    meff_scatter = hv.Scatter(
        (t_meff, m_plot), "t", "m_eff(t)"
    ).opts(
        color=color,
        size=6,
        alpha=0.8,
    ).relabel("m_eff(t)")

    right_overlays = meff_scatter

    # Add best window shading (green region)
    if width > 0 and t_start >= 0:
        t_window_start = t_start * dt_scale
        t_window_end = (t_start + width - 1) * dt_scale
        # Use VSpan for vertical shading
        window_shade = hv.VSpan(t_window_start, t_window_end).opts(
            color="green",
            alpha=0.15,
        )
        right_overlays = window_shade * right_overlays

    # Add fitted mass line and error band
    if mass_scaled > 0:
        t_line = np.array([float(t_meff.min()), float(t_meff.max())])
        m_line = np.array([mass_scaled, mass_scaled])
        mass_curve = hv.Curve(
            (t_line, m_line), "t", "m_eff(t)"
        ).opts(
            color="red",
            line_dash="dashed",
            line_width=2,
        ).relabel(f"M={mass_scaled:.4f}±{mass_error_scaled:.4f}")
        right_overlays = right_overlays * mass_curve

        # Error band (pink region)
        if mass_error_scaled > 0 and mass_error_scaled < mass_scaled:
            # Create error band using Area
            error_band = hv.Area(
                (t_line, [mass_scaled - mass_error_scaled, mass_scaled - mass_error_scaled],
                 [mass_scaled + mass_error_scaled, mass_scaled + mass_error_scaled]),
                kdims=["t"],
                vdims=["lower", "upper"],
            ).opts(
                color="red",
                alpha=0.15,
            )
            right_overlays = error_band * right_overlays

    # Add annotation for best window and data point count
    n_valid_corr = int(corr_mask.sum())
    n_valid_meff = int(meff_mask.sum())

    if width > 0 and t_start >= 0:
        window_label = f"Best: [{t_start}, {t_start + width - 1}]"
    else:
        window_label = "No valid window"

    best_r2 = best_window.get("r2", float("nan"))
    r2_label = f"R²={best_r2:.4f}" if np.isfinite(best_r2) else "R²=n/a"

    # Combine window info with data point counts
    info_label = (
        f"{window_label}\n{r2_label}\nC(t): {n_valid_corr} pts, m_eff: {n_valid_meff} pts"
    )

    # Use Text annotation at top of plot
    y_max = float(m_plot.max()) if len(m_plot) > 0 else mass_scaled
    annotation = hv.Text(
        float(t_meff.min()) + 0.05 * (float(t_meff.max()) - float(t_meff.min())),
        y_max * 0.95,
        info_label,
        halign="left",
        valign="top",
    ).opts(
        color="green",
        fontsize=9,
    )
    right_overlays = right_overlays * annotation

    right_panel = right_overlays.opts(
        xlabel="t (time lag)",
        ylabel="m_eff(t)",
        title=f"{channel_name.replace('_', ' ').title()} Effective Mass Plateau",
        width=400,
        height=350,
        legend_position="top_right",
    )

    return left_panel, right_panel


def build_all_channels_overlay(
    channel_results: dict[str, Any],
    plot_type: str = "correlator",
) -> hv.Overlay | None:
    """Build overlay of all channels on single plot.

    Args:
        channel_results: Dict mapping channel names to ChannelCorrelatorResult.
        plot_type: "correlator" or "effective_mass".

    Returns:
        HoloViews Overlay with all channels, or None if no data.
    """
    curves = []
    for name, result in channel_results.items():
        if result.n_samples == 0:
            continue

        color = CHANNEL_COLORS.get(name, "#1f77b4")
        dt = result.dt

        if plot_type == "correlator":
            corr = result.correlator.cpu().numpy() if hasattr(result.correlator, "cpu") else np.asarray(result.correlator)
            mask = (corr > 0) & np.isfinite(corr)
            if mask.sum() < 2:
                continue
            t = np.arange(len(corr)) * dt
            curve = hv.Curve(
                (t[mask], corr[mask]), "t", "C(t)"
            ).opts(
                color=color,
                line_width=2,
            ).relabel(name)
            curves.append(curve)
        else:  # effective_mass
            meff = result.effective_mass.cpu().numpy() if hasattr(result.effective_mass, "cpu") else np.asarray(result.effective_mass)
            mask = np.isfinite(meff) & (meff > 0)
            if mask.sum() < 2:
                continue
            t = np.arange(len(meff)) * dt
            curve = hv.Curve(
                (t[mask], meff[mask]), "t", "m_eff(t)"
            ).opts(
                color=color,
                line_width=2,
            ).relabel(name)
            curves.append(curve)

    if not curves:
        return None

    overlay = curves[0]
    for c in curves[1:]:
        overlay = overlay * c

    ylabel = "C(t)" if plot_type == "correlator" else "m_eff(t)"
    title = "All Channel Correlators" if plot_type == "correlator" else "All Channel Effective Masses"

    return overlay.opts(
        logy=(plot_type == "correlator"),
        xlabel="t (time lag)",
        ylabel=ylabel,
        title=title,
        width=700,
        height=400,
        legend_position="top_right",
    )
