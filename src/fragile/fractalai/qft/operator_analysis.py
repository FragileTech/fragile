"""Reusable multiscale operator analysis: dataclasses, computation, and HoloViews plots.

Provides a channel-agnostic pipeline that treats each smearing scale as an
"estimator" of the channel mass.  The analysis mirrors the glueball
calibration/correction pipeline (consensus mass, pairwise discrepancies,
systematics verdict) and can be driven from any set of per-scale
``ChannelCorrelatorResult`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import holoviews as hv
import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScaleMeasurement:
    """Single mass measurement at one smearing scale."""

    scale: float
    scale_index: int
    mass: float
    mass_error: float
    r_squared: float
    label: str


@dataclass
class PairwiseDiscrepancy:
    """Pairwise comparison between two scale measurements."""

    label_a: str
    label_b: str
    mass_a: float
    mass_b: float
    ratio: float
    delta_pct: float
    abs_delta_pct: float
    combined_error: float
    pull_sigma: float


@dataclass
class ConsensusResult:
    """Inverse-variance weighted consensus mass from multiple scales."""

    mass: float
    stat_error: float
    systematic_spread: float
    chi2: float
    ndof: int
    weighting: str
    measurements: list[ScaleMeasurement] = field(default_factory=list)


@dataclass
class SystematicsVerdict:
    """Summary verdict on cross-scale consistency."""

    label: str  # "consistent" | "mild tension" | "tension" | "insufficient data"
    alert_type: str  # "success" | "warning" | "danger" | "secondary"
    details: str


@dataclass
class MultiscaleAnalysisBundle:
    """Full analysis payload for one channel across scales."""

    channel_name: str
    measurements: list[ScaleMeasurement]
    consensus: ConsensusResult
    discrepancies: list[PairwiseDiscrepancy]
    verdict: SystematicsVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(t: Any) -> np.ndarray:
    """Convert a Tensor (or array-like) to a numpy array."""
    if hasattr(t, "cpu"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def scale_color_hex(scale_idx: int, n_scales: int) -> str:
    """Return a gradient hex color for *scale_idx* out of *n_scales*."""
    if n_scales <= 1:
        return "#1f77b4"
    frac = float(scale_idx) / float(max(1, n_scales - 1))
    r = int(round(35 + 210 * frac))
    g = int(round(160 - 90 * abs(2.0 * frac - 1.0)))
    b = int(round(235 - 190 * frac))
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


# ---------------------------------------------------------------------------
# Pure computation functions
# ---------------------------------------------------------------------------


def extract_scale_measurements(
    results: list[Any],
    scales: np.ndarray,
) -> list[ScaleMeasurement]:
    """Build a :class:`ScaleMeasurement` for each per-scale result.

    *results* is ``list[ChannelCorrelatorResult]`` — one per scale.
    """
    measurements: list[ScaleMeasurement] = []
    for s_idx, result in enumerate(results):
        if result is None or getattr(result, "n_samples", 0) <= 0:
            continue
        fit = getattr(result, "mass_fit", None) or {}
        mass = float(fit.get("mass", float("nan")))
        mass_error = float(fit.get("mass_error", float("nan")))
        r2 = float(fit.get("r_squared", float("nan")))
        scale_val = float(scales[s_idx]) if s_idx < len(scales) else float("nan")
        measurements.append(
            ScaleMeasurement(
                scale=scale_val,
                scale_index=s_idx,
                mass=mass,
                mass_error=mass_error,
                r_squared=r2,
                label=f"R={scale_val:.4g}",
            )
        )
    return measurements


def compute_consensus_mass(
    measurements: list[ScaleMeasurement],
) -> ConsensusResult:
    """Compute inverse-variance weighted consensus from scale measurements."""
    valid = [
        m for m in measurements if math.isfinite(m.mass) and m.mass > 0
    ]
    if not valid:
        return ConsensusResult(
            mass=float("nan"),
            stat_error=float("nan"),
            systematic_spread=float("nan"),
            chi2=float("nan"),
            ndof=1,
            weighting="n/a",
            measurements=list(measurements),
        )

    masses = np.asarray([m.mass for m in valid], dtype=float)
    errors = np.asarray([m.mass_error for m in valid], dtype=float)
    finite_weight_mask = np.isfinite(errors) & (errors > 0)

    if np.any(finite_weight_mask):
        w_masses = masses[finite_weight_mask]
        w_errors = errors[finite_weight_mask]
        weights = 1.0 / np.maximum(w_errors, 1e-12) ** 2
        consensus_mass = float(np.sum(weights * w_masses) / np.sum(weights))
        stat_error = float(np.sqrt(1.0 / np.sum(weights)))
        n_eff = int(w_masses.size)
        ndof = max(n_eff - 1, 1)
        chi2 = float(np.sum(weights * (w_masses - consensus_mass) ** 2))
        weighting = "inverse-variance weighted"
    else:
        consensus_mass = float(np.mean(masses))
        stat_error = (
            float(np.std(masses, ddof=1) / np.sqrt(float(masses.size)))
            if masses.size > 1
            else float("nan")
        )
        ndof = max(int(masses.size) - 1, 1)
        chi2 = float("nan")
        weighting = "unweighted mean"

    systematic_spread = (
        float(np.std(masses, ddof=1)) if masses.size > 1 else 0.0
    )

    return ConsensusResult(
        mass=consensus_mass,
        stat_error=stat_error,
        systematic_spread=systematic_spread,
        chi2=chi2,
        ndof=ndof,
        weighting=weighting,
        measurements=list(measurements),
    )


def compute_pairwise_discrepancies(
    measurements: list[ScaleMeasurement],
) -> list[PairwiseDiscrepancy]:
    """All-pairs ratio, delta%, pull_sigma between valid measurements."""
    valid = [
        m for m in measurements if math.isfinite(m.mass) and m.mass > 0
    ]
    rows: list[PairwiseDiscrepancy] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            ratio = a.mass / b.mass if b.mass > 0 else float("nan")
            delta_pct = (ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan")
            abs_diff = abs(a.mass - b.mass)
            comb_err = float("nan")
            pull_sigma = float("nan")
            if np.isfinite(a.mass_error) and np.isfinite(b.mass_error):
                comb_err = float(
                    np.sqrt(max(a.mass_error, 0.0) ** 2 + max(b.mass_error, 0.0) ** 2)
                )
                if comb_err > 0:
                    pull_sigma = abs_diff / comb_err
            rows.append(
                PairwiseDiscrepancy(
                    label_a=a.label,
                    label_b=b.label,
                    mass_a=a.mass,
                    mass_b=b.mass,
                    ratio=ratio,
                    delta_pct=delta_pct,
                    abs_delta_pct=abs(delta_pct) if np.isfinite(delta_pct) else float("nan"),
                    combined_error=comb_err,
                    pull_sigma=pull_sigma,
                )
            )
    return rows


def evaluate_systematics_verdict(
    discrepancies: list[PairwiseDiscrepancy],
) -> SystematicsVerdict:
    """Classify cross-scale agreement as consistent / tension / insufficient."""
    if not discrepancies:
        return SystematicsVerdict(
            label="insufficient data",
            alert_type="secondary",
            details="Need at least two scales with valid fits.",
        )
    abs_deltas = np.asarray(
        [d.abs_delta_pct for d in discrepancies], dtype=float
    )
    pulls = np.asarray([d.pull_sigma for d in discrepancies], dtype=float)
    finite_abs = abs_deltas[np.isfinite(abs_deltas)]
    finite_pull = pulls[np.isfinite(pulls)]
    max_abs_delta = float(np.max(finite_abs)) if finite_abs.size > 0 else float("nan")
    max_pull = float(np.max(finite_pull)) if finite_pull.size > 0 else float("nan")

    details = ""
    if np.isfinite(max_abs_delta):
        details = f"max |Δ%| = {max_abs_delta:.2f}%"
    else:
        details = "max |Δ%| = n/a"
    if np.isfinite(max_pull):
        details += f", max pull = {max_pull:.2f}σ"
    else:
        details += ", max pull = n/a"

    if (
        np.isfinite(max_abs_delta)
        and max_abs_delta <= 5.0
        and (not np.isfinite(max_pull) or max_pull <= 1.5)
    ):
        return SystematicsVerdict(label="consistent", alert_type="success", details=details)
    elif (
        np.isfinite(max_abs_delta)
        and max_abs_delta <= 15.0
        and (not np.isfinite(max_pull) or max_pull <= 3.0)
    ):
        return SystematicsVerdict(label="mild tension", alert_type="warning", details=details)
    else:
        return SystematicsVerdict(label="tension", alert_type="danger", details=details)


def analyze_channel_across_scales(
    results: list[Any],
    scales: np.ndarray,
    channel_name: str,
) -> MultiscaleAnalysisBundle:
    """Full pipeline: extract → consensus → pairwise → verdict."""
    measurements = extract_scale_measurements(results, scales)
    consensus = compute_consensus_mass(measurements)
    discrepancies = compute_pairwise_discrepancies(measurements)
    verdict = evaluate_systematics_verdict(discrepancies)
    return MultiscaleAnalysisBundle(
        channel_name=channel_name,
        measurements=measurements,
        consensus=consensus,
        discrepancies=discrepancies,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# HoloViews plot builders
# ---------------------------------------------------------------------------


def _errorbars_cap_hook(color: str):
    """Return a Bokeh hook that colors error-bar caps to match *color*."""
    def hook(plot, element):
        whisker = plot.handles.get("glyph")
        if whisker is not None:
            for attr in ("upper_head", "lower_head"):
                head = getattr(whisker, attr, None)
                if head is not None:
                    head.line_color = color
    return hook


def build_multiscale_correlator_plot(
    results: list[Any],
    scales: np.ndarray,
    channel_name: str,
    *,
    width: int = 900,
    height: int = 340,
) -> hv.Overlay | None:
    """Per-scale scatter+errorbars for the correlator C(t), with AIC fit lines."""
    n_scales = len(results)
    elements: list[hv.Element] = []
    for s_idx, result in enumerate(results):
        if result is None or getattr(result, "n_samples", 0) <= 0:
            continue
        color = scale_color_hex(s_idx, n_scales)
        scale_val = float(scales[s_idx]) if s_idx < len(scales) else s_idx
        label = f"R={scale_val:.4g}"
        corr = _to_numpy(result.correlator).astype(float, copy=False)
        dt = float(result.dt) if hasattr(result, "dt") else 1.0
        t_arr = np.arange(corr.shape[0], dtype=float) * dt
        mask = np.isfinite(corr) & (corr > 0)
        if np.count_nonzero(mask) < 2:
            continue
        t_plot = t_arr[mask]
        c_plot = corr[mask]
        scatter = hv.Scatter(
            (t_plot, c_plot),
            kdims=["lag"],
            vdims=["C(t)"],
            label=label,
        ).opts(color=color, size=5, alpha=0.85)
        elements.append(scatter)
        # Error bars.
        corr_err = getattr(result, "correlator_err", None)
        if corr_err is not None:
            err_np = _to_numpy(corr_err).astype(float, copy=False)
            err_mask = mask & (np.arange(len(corr)) < len(err_np))
            if np.count_nonzero(err_mask) >= 2:
                valid_err = err_np[: len(corr)][err_mask]
                elements.append(
                    hv.ErrorBars(
                        (t_arr[err_mask], corr[err_mask], valid_err),
                        kdims=["lag"],
                        vdims=["C(t)", "err"],
                    ).opts(color=color, alpha=0.7, line_width=1, hooks=[_errorbars_cap_hook(color)])
                )
        # AIC exponential fit line: C(t) = C(t_anchor) * exp(-m_scaled * (t - t_anchor)).
        mass_fit = getattr(result, "mass_fit", None) or {}
        mass = float(mass_fit.get("mass", 0.0))
        best_window = mass_fit.get("best_window", {})
        if not isinstance(best_window, dict):
            best_window = {}
        t_start = best_window.get("t_start", 0)
        dt_scale = dt if dt > 0 else 1.0
        mass_scaled = mass / dt_scale if mass > 0 and dt_scale > 0 else 0.0
        if mass_scaled > 0 and len(t_plot) >= 2:
            t_line = np.linspace(float(t_plot.min()), float(t_plot.max()), 200)
            t_anchor = t_start * dt_scale
            anchor_idx = int(np.argmin(np.abs(t_plot - t_anchor)))
            c_anchor = float(c_plot[anchor_idx]) if anchor_idx < len(c_plot) else float(c_plot[0])
            if c_anchor > 0:
                c_line = c_anchor * np.exp(-mass_scaled * (t_line - t_anchor))
                c_line = np.maximum(c_line, np.finfo(float).tiny)
                elements.append(
                    hv.Curve(
                        (t_line, c_line),
                        kdims=["lag"],
                        vdims=["C(t)"],
                        label=f"fit m={mass_scaled:.4f}",
                    ).opts(color=color, line_dash="dashed", line_width=1.5, alpha=0.8)
                )
    if not elements:
        return None
    overlay = elements[0]
    for el in elements[1:]:
        overlay *= el
    return overlay.opts(
        width=width,
        height=height,
        logy=True,
        show_grid=True,
        legend_position="right",
        xlabel="t (time lag)",
        ylabel="C(t)",
        title=f"{channel_name.replace('_', ' ').title()} Correlator Across Scales",
    )


def build_multiscale_effective_mass_plot(
    results: list[Any],
    scales: np.ndarray,
    channel_name: str,
    *,
    width: int = 900,
    height: int = 340,
) -> hv.Overlay | None:
    """Per-scale scatter+errorbars for the effective mass m_eff(t), with fitted mass lines.

    Uses a log y-axis.  Error-bar lower bounds are clamped to ``_MEFF_FLOOR``
    so they never cross zero (which would be invisible on a log scale).
    """
    _MEFF_FLOOR = 1e-17
    n_scales = len(results)
    elements: list[hv.Element] = []
    for s_idx, result in enumerate(results):
        if result is None or getattr(result, "n_samples", 0) <= 0:
            continue
        color = scale_color_hex(s_idx, n_scales)
        scale_val = float(scales[s_idx]) if s_idx < len(scales) else s_idx
        label = f"R={scale_val:.4g}"
        meff = _to_numpy(result.effective_mass).astype(float, copy=False)
        dt = float(result.dt) if hasattr(result, "dt") else 1.0
        t_arr = np.arange(meff.shape[0], dtype=float) * dt
        mask = np.isfinite(meff) & (meff > 0)
        if np.count_nonzero(mask) < 2:
            continue
        t_plot = t_arr[mask]
        m_plot = meff[mask]
        scatter = hv.Scatter(
            (t_plot, m_plot),
            kdims=["lag"],
            vdims=["m_eff"],
            label=label,
        ).opts(color=color, size=5, alpha=0.85)
        elements.append(scatter)
        # Error propagation: meff_err = sqrt((dC_t/C_t)^2 + (dC_{t+1}/C_{t+1})^2) / dt
        corr_err = getattr(result, "correlator_err", None)
        if corr_err is not None:
            err_np = _to_numpy(corr_err).astype(float, copy=False)
            corr_np = _to_numpy(result.correlator).astype(float, copy=False)
            n = min(len(meff), len(corr_np) - 1, len(err_np) - 1)
            if n >= 2:
                c_t = corr_np[:n]
                c_tp1 = corr_np[1 : n + 1]
                e_t = err_np[:n]
                e_tp1 = err_np[1 : n + 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_t = np.where(np.abs(c_t) > 0, e_t / np.abs(c_t), np.nan)
                    rel_tp1 = np.where(np.abs(c_tp1) > 0, e_tp1 / np.abs(c_tp1), np.nan)
                    meff_err = np.sqrt(rel_t**2 + rel_tp1**2) / max(dt, 1e-12)
                meff_mask = mask[:n] & np.isfinite(meff_err) & (meff_err > 0)
                if np.count_nonzero(meff_mask) >= 2:
                    eb_y = meff[:n][meff_mask]
                    eb_e = meff_err[meff_mask]
                    # Clamp the lower extent so it never goes below _MEFF_FLOOR.
                    eb_e = np.minimum(eb_e, eb_y - _MEFF_FLOOR)
                    eb_e = np.maximum(eb_e, 0.0)
                    elements.append(
                        hv.ErrorBars(
                            (t_arr[:n][meff_mask], eb_y, eb_e),
                            kdims=["lag"],
                            vdims=["m_eff", "err"],
                        ).opts(color=color, alpha=0.7, line_width=1, hooks=[_errorbars_cap_hook(color)])
                    )
        # AIC fitted mass horizontal line.
        mass_fit = getattr(result, "mass_fit", None) or {}
        mass = float(mass_fit.get("mass", 0.0))
        if mass > 0 and len(t_plot) >= 2:
            t_line = np.array([float(t_plot.min()), float(t_plot.max())])
            elements.append(
                hv.Curve(
                    (t_line, [mass, mass]),
                    kdims=["lag"],
                    vdims=["m_eff"],
                    label=f"M={mass:.4f}",
                ).opts(color=color, line_dash="dashed", line_width=1.5, alpha=0.8)
            )
    if not elements:
        return None
    overlay = elements[0]
    for el in elements[1:]:
        overlay *= el
    return overlay.opts(
        width=width,
        height=height,
        logy=True,
        show_grid=True,
        legend_position="right",
        xlabel="t (time lag)",
        ylabel="m_eff(t)",
        title=f"{channel_name.replace('_', ' ').title()} Effective Mass Across Scales",
    )


def build_mass_vs_scale_plot(
    measurements: list[ScaleMeasurement],
    channel_name: str,
    consensus: ConsensusResult | None = None,
    *,
    width: int = 900,
    height: int = 320,
) -> hv.Overlay | None:
    """Per-scale colored scatter+errorbars of mass vs kernel scale."""
    valid = [
        m for m in measurements if math.isfinite(m.mass) and m.mass > 0
    ]
    if not valid:
        return None
    n_scales = max(m.scale_index for m in measurements) + 1 if measurements else 1
    elements: list[hv.Element] = []
    for m in valid:
        color = scale_color_hex(m.scale_index, n_scales)
        elements.append(
            hv.Scatter(
                [(m.scale, m.mass)],
                kdims=["scale"],
                vdims=["mass"],
                label=m.label,
            ).opts(color=color, size=8)
        )
        if math.isfinite(m.mass_error) and m.mass_error > 0:
            elements.append(
                hv.ErrorBars(
                    [(m.scale, m.mass, m.mass_error)],
                    kdims=["scale"],
                    vdims=["mass", "mass_error"],
                ).opts(
                    color=color, line_width=1.5, alpha=0.8,
                    hooks=[_errorbars_cap_hook(color)],
                )
            )
    if consensus is not None and math.isfinite(consensus.mass) and consensus.mass > 0:
        elements.append(
            hv.HLine(consensus.mass).opts(color="#2ca02c", line_width=2)
        )
        if math.isfinite(consensus.systematic_spread) and consensus.systematic_spread > 0:
            elements.append(
                hv.HSpan(
                    consensus.mass - consensus.systematic_spread,
                    consensus.mass + consensus.systematic_spread,
                ).opts(color="#2ca02c", alpha=0.12)
            )
    if not elements:
        return None
    overlay = elements[0]
    for el in elements[1:]:
        overlay *= el
    pretty = channel_name.replace("_", " ").title()
    return overlay.opts(
        width=width,
        height=height,
        show_grid=True,
        xlabel="Kernel scale",
        ylabel=f"Extracted {pretty} mass",
        title=f"{pretty} Mass vs Scale",
    )


def build_consensus_plot(
    measurements: list[ScaleMeasurement],
    consensus: ConsensusResult,
    channel_name: str,
    *,
    width: int = 760,
    height: int = 320,
) -> hv.Overlay | None:
    """Scatter per scale at x=index with consensus HLine + systematic HSpan."""
    valid = [
        m for m in measurements if math.isfinite(m.mass) and m.mass > 0
    ]
    if not valid:
        return None
    import pandas as pd

    n_scales = max(m.scale_index for m in measurements) + 1 if measurements else 1
    rows = []
    for i, m in enumerate(valid):
        rows.append(
            {
                "x": float(i),
                "mass": m.mass,
                "mass_error": m.mass_error,
                "label": m.label,
                "color": scale_color_hex(m.scale_index, n_scales),
            }
        )
    df = pd.DataFrame(rows)
    scatter = hv.Scatter(
        df,
        kdims=[("x", "Scale index")],
        vdims=[("mass", "Mass"), ("label", "Scale"), ("mass_error", "Mass Error")],
    ).opts(
        width=width,
        height=height,
        size=11,
        color="#4c78a8",
        marker="circle",
        tools=["hover"],
        xlabel="Scale",
        ylabel="Mass (index units)",
        title=f"{channel_name.replace('_', ' ').title()} Scale-as-Estimator Consensus",
    )
    overlay: hv.Element = scatter
    err_df = df[np.isfinite(df["mass_error"].to_numpy()) & (df["mass_error"].to_numpy() > 0)][
        ["x", "mass", "mass_error"]
    ]
    if not err_df.empty:
        overlay *= hv.ErrorBars(
            err_df,
            kdims=[("x", "Scale index")],
            vdims=[("mass", "Mass"), ("mass_error", "Mass Error")],
        ).opts(
            color="#4c78a8", alpha=0.9, line_width=1,
            hooks=[_errorbars_cap_hook("#4c78a8")],
        )
    if math.isfinite(consensus.mass):
        overlay *= hv.HLine(float(consensus.mass)).opts(color="#2ca02c", line_width=2)
    if (
        math.isfinite(consensus.mass)
        and math.isfinite(consensus.systematic_spread)
        and consensus.systematic_spread > 0
    ):
        overlay *= hv.HSpan(
            float(consensus.mass - consensus.systematic_spread),
            float(consensus.mass + consensus.systematic_spread),
        ).opts(color="#2ca02c", alpha=0.12)
    xticks = [(float(i), m.label) for i, m in enumerate(valid)]
    overlay = overlay.opts(
        xlim=(-0.5, float(len(valid) - 0.5)),
        xticks=xticks,
        xrotation=20,
        show_grid=True,
    )
    return overlay


def build_per_scale_channel_plots(
    results: list[Any],
    scales: np.ndarray,
) -> list[tuple[str, Any]]:
    """Build per-scale ChannelPlot side-by-side layouts.

    Returns list of ``(label, layout)`` tuples.
    """
    from fragile.fractalai.qft.plotting import ChannelPlot

    layouts: list[tuple[str, Any]] = []
    for s_idx, result in enumerate(results):
        if result is None or getattr(result, "n_samples", 0) <= 0:
            continue
        scale_val = float(scales[s_idx]) if s_idx < len(scales) else s_idx
        label = f"Scale R={scale_val:.4g}"
        try:
            cp = ChannelPlot(result)
            layout = cp.side_by_side()
            if layout is not None:
                layouts.append((label, layout))
        except Exception:
            pass
    return layouts


# ---------------------------------------------------------------------------
# Markdown summary formatter
# ---------------------------------------------------------------------------


def format_consensus_summary(
    consensus: ConsensusResult,
    discrepancies: list[PairwiseDiscrepancy],
    channel_name: str,
) -> str:
    """Markdown summary matching the glueball format."""
    pretty = channel_name.replace("_", " ").title()
    lines = [f"**{pretty} Scale-as-Estimator Consensus:**"]
    if math.isfinite(consensus.mass) and consensus.mass > 0:
        line = f"- Consensus mass ({consensus.weighting}): `{consensus.mass:.6g}`"
        if math.isfinite(consensus.stat_error):
            line += f" ± `{consensus.stat_error:.2g}` (stat)"
        if math.isfinite(consensus.systematic_spread):
            line += f" ± `{consensus.systematic_spread:.2g}` (syst)"
        lines.append(line)
        if math.isfinite(consensus.systematic_spread) and consensus.mass > 0:
            lines.append(
                f"- Relative systematic spread: "
                f"`{(consensus.systematic_spread / consensus.mass) * 100.0:.2f}%`"
            )
        if math.isfinite(consensus.chi2):
            red_chi2 = consensus.chi2 / max(consensus.ndof, 1)
            lines.append(
                f"- Scale consistency (χ²/ndof): "
                f"`{consensus.chi2:.4g}/{consensus.ndof}` = `{red_chi2:.4g}`"
            )
    else:
        lines.append("- n/a (not enough scales with valid fits).")
    if discrepancies:
        max_pair = max(
            discrepancies,
            key=lambda d: d.abs_delta_pct if math.isfinite(d.abs_delta_pct) else float("-inf"),
        )
        if math.isfinite(max_pair.abs_delta_pct):
            lines.append(
                "- Largest pairwise discrepancy: "
                f"`{max_pair.label_a}` vs `{max_pair.label_b}` "
                f"({max_pair.delta_pct:+.2f}%, pull `{max_pair.pull_sigma:.2f}σ`)"
            )
    return "  \n".join(lines)
