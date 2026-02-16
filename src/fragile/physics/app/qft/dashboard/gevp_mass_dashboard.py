"""Reusable GEVP multi-mode mass spectrum widgets and update logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from fragile.physics.app.qft.gevp_mass_extraction import GEVPMassSpectrum, T0SweepResult


@dataclass
class GEVPMassSpectrumWidgets:
    """Panel widgets for multi-mode GEVP mass spectrum visualization."""

    summary: pn.pane.Markdown
    effective_mass_plot: pn.pane.HoloViews
    mass_spectrum_bar: pn.pane.HoloViews
    eigenvalue_decay_plot: pn.pane.HoloViews
    mode_table: pn.widgets.Tabulator
    t0_comparison_plot: pn.pane.HoloViews
    t0_comparison_table: pn.widgets.Tabulator


def create_gevp_mass_spectrum_widgets() -> GEVPMassSpectrumWidgets:
    """Create GEVP mass spectrum widgets."""
    return GEVPMassSpectrumWidgets(
        summary=pn.pane.Markdown(
            "**GEVP Multi-Mode Spectrum:** _run analysis to populate._",
            sizing_mode="stretch_width",
        ),
        effective_mass_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        mass_spectrum_bar=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        eigenvalue_decay_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        mode_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
        t0_comparison_plot=pn.pane.HoloViews(
            sizing_mode="stretch_width",
            linked_axes=False,
        ),
        t0_comparison_table=pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination=None,
            show_index=False,
            sizing_mode="stretch_width",
        ),
    )


def clear_gevp_mass_spectrum(w: GEVPMassSpectrumWidgets) -> None:
    """Reset GEVP mass spectrum widgets."""
    w.summary.object = "**GEVP Multi-Mode Spectrum:** _run analysis to populate._"
    w.effective_mass_plot.object = None
    w.mass_spectrum_bar.object = None
    w.eigenvalue_decay_plot.object = None
    w.mode_table.value = pd.DataFrame()
    w.t0_comparison_plot.object = None
    w.t0_comparison_table.value = pd.DataFrame()


def build_gevp_mass_spectrum_sections(w: GEVPMassSpectrumWidgets) -> list[Any]:
    """Return reusable GEVP mass spectrum section blocks for tab composition."""
    return [
        pn.layout.Divider(),
        pn.pane.Markdown("### GEVP Multi-Mode Mass Spectrum"),
        pn.pane.Markdown(
            "_Fixed-eigenvector projection extracts ground + excited state masses "
            "from the GEVP correlator matrix._"
        ),
        w.summary,
        w.effective_mass_plot,
        w.mass_spectrum_bar,
        w.eigenvalue_decay_plot,
        pn.pane.Markdown("### Mode Summary Table"),
        w.mode_table,
        pn.pane.Markdown("### t0 Sweep Comparison"),
        w.t0_comparison_plot,
        w.t0_comparison_table,
    ]


MODE_COLORS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
]


def build_effective_mass_plateau_plot(
    spectrum: GEVPMassSpectrum,
    *,
    family_label: str = "",
    dt: float = 1.0,
) -> hv.Overlay | None:
    """Create scatter + plateau band per mode for effective masses."""
    m_eff = spectrum.effective_masses
    if m_eff.numel() == 0:
        return None

    m_np = m_eff.detach().cpu().numpy()
    n_modes, n_lags = m_np.shape
    tau = np.arange(n_lags, dtype=float) * dt

    overlays = []
    for mode_idx in range(min(n_modes, len(MODE_COLORS))):
        row = m_np[mode_idx]
        finite = np.isfinite(row) & (row > 0)
        if not np.any(finite):
            continue

        color = MODE_COLORS[mode_idx % len(MODE_COLORS)]
        label = f"mode {mode_idx}"

        # Scatter points
        df = pd.DataFrame({
            "tau": tau[finite],
            "m_eff": row[finite],
            "mode": label,
        })
        scatter = hv.Scatter(df, kdims=["tau"], vdims=["m_eff", "mode"]).opts(
            color=color,
            size=6,
            tools=["hover"],
        )
        overlays.append(scatter)

        # Plateau band
        plateau_mass = float(spectrum.plateau_masses[mode_idx].item())
        plateau_err = float(spectrum.plateau_errors[mode_idx].item())
        p_start, p_end = spectrum.plateau_ranges[mode_idx]
        if math.isfinite(plateau_mass) and plateau_mass > 0 and p_end > p_start:
            if not math.isfinite(plateau_err) or plateau_err < 0:
                plateau_err = 0.0
            x_lo = float(p_start) * dt
            x_hi = float(p_end - 1) * dt
            band = hv.Area(
                ([x_lo, x_hi], [plateau_mass - plateau_err] * 2, [plateau_mass + plateau_err] * 2),
                kdims=["tau"],
                vdims=["y_lo", "y_hi"],
            ).opts(color=color, alpha=0.2)
            hline = hv.Curve(
                ([x_lo, x_hi], [plateau_mass, plateau_mass]),
                kdims=["tau"],
                vdims=["m_eff"],
            ).opts(color=color, line_dash="dashed", line_width=1.5)
            overlays.extend([band, hline])

    if not overlays:
        return None

    overlay = hv.Overlay(overlays)
    title = f"Effective Mass Plateau: {family_label}" if family_label else "Effective Mass Plateau"
    return overlay.opts(
        width=900,
        height=350,
        xlabel="lag τ",
        ylabel="m_eff(τ)",
        title=title,
        show_grid=True,
        legend_position="top_right",
    )


def build_gevp_mass_bar_chart(
    spectrum: GEVPMassSpectrum,
    *,
    family_label: str = "",
) -> hv.Bars | None:
    """Create a bar chart of all mode masses with error bars."""
    masses = spectrum.plateau_masses.detach().cpu().numpy()
    errors = spectrum.plateau_errors.detach().cpu().numpy()
    n_modes = len(masses)
    if n_modes == 0:
        return None

    rows = []
    for i in range(n_modes):
        m = float(masses[i])
        e = float(errors[i])
        if not math.isfinite(m) or m <= 0:
            continue
        if not math.isfinite(e):
            e = 0.0
        rows.append({
            "mode": f"mode {i}",
            "mass": m,
            "error": e,
            "ratio": m / float(masses[0])
            if math.isfinite(float(masses[0])) and masses[0] > 0
            else float("nan"),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    title = f"GEVP Mass Spectrum: {family_label}" if family_label else "GEVP Mass Spectrum"
    bars = hv.Bars(df, kdims=["mode"], vdims=["mass", "error", "ratio"]).opts(
        width=600,
        height=300,
        xlabel="Mode",
        ylabel="Mass",
        title=title,
        color="#4c78a8",
        tools=["hover"],
        show_grid=True,
    )
    # Error bars
    err_data = []
    for row in rows:
        err_data.append((row["mode"], row["mass"] - row["error"], row["mass"] + row["error"]))
    if err_data:
        err_df = pd.DataFrame(err_data, columns=["mode", "y_lo", "y_hi"])
        err_seg = hv.Segments(err_df, kdims=["mode", "y_lo"], vdims=["mode", "y_hi"]).opts(
            color="black", line_width=2
        )
        return (bars * err_seg).opts(title=title)

    return bars


def build_eigenvalue_decay_plot(
    spectrum: GEVPMassSpectrum,
    *,
    family_label: str = "",
    dt: float = 1.0,
) -> hv.Overlay | None:
    """Create log-scale λ_n(τ) vs τ curves per mode with optional exp fit overlay."""
    eigenvalues = spectrum.mode_eigenvalues
    if eigenvalues.numel() == 0:
        return None

    ev_np = eigenvalues.detach().cpu().numpy()
    n_modes, n_lags = ev_np.shape
    tau = np.arange(n_lags, dtype=float) * dt

    # Pre-compute exp fit params if available
    has_exp = spectrum.exp_masses is not None and spectrum.exp_r2 is not None
    if has_exp:
        exp_m = spectrum.exp_masses.detach().cpu().numpy()
        exp_r2_np = spectrum.exp_r2.detach().cpu().numpy()

    overlays = []
    for mode_idx in range(min(n_modes, len(MODE_COLORS))):
        row = ev_np[mode_idx]
        finite = np.isfinite(row) & (row > 0)
        if not np.any(finite):
            continue

        color = MODE_COLORS[mode_idx % len(MODE_COLORS)]
        label = f"mode {mode_idx}"
        df = pd.DataFrame({
            "tau": tau[finite],
            "lambda": row[finite],
            "mode": label,
        })
        curve = hv.Curve(df, kdims=["tau"], vdims=["lambda", "mode"]).opts(
            color=color,
            line_width=2,
            tools=["hover"],
        )
        overlays.append(curve)

        # Overlay exponential fit line if available
        if has_exp and math.isfinite(float(exp_m[mode_idx])) and float(exp_r2_np[mode_idx]) > 0:
            mass_val = float(exp_m[mode_idx])
            # Scale amplitude A to match data at first finite point
            first_finite_idx = int(np.where(finite)[0][0])
            a_fit = float(row[first_finite_idx]) * np.exp(mass_val * tau[first_finite_idx])
            fit_tau = tau[finite]
            fit_lambda = a_fit * np.exp(-mass_val * fit_tau)
            fit_df = pd.DataFrame({
                "tau": fit_tau,
                "lambda": fit_lambda,
                "mode": f"mode {mode_idx} exp fit",
            })
            fit_curve = hv.Curve(fit_df, kdims=["tau"], vdims=["lambda", "mode"]).opts(
                color=color,
                line_dash="dashed",
                line_width=1.5,
            )
            overlays.append(fit_curve)

    if not overlays:
        return None

    overlay = hv.Overlay(overlays)
    title = f"Eigenvalue Decay: {family_label}" if family_label else "Eigenvalue Decay"
    return overlay.opts(
        width=900,
        height=350,
        xlabel="lag τ",
        ylabel="λ_n(τ)",
        title=title,
        logy=True,
        show_grid=True,
        legend_position="top_right",
    )


def build_mode_summary_table(
    spectrum: GEVPMassSpectrum,
) -> pd.DataFrame:
    """Build a DataFrame summarizing each mode's extracted mass."""
    masses = spectrum.plateau_masses.detach().cpu().numpy()
    errors = spectrum.plateau_errors.detach().cpu().numpy()
    n_modes = len(masses)
    if n_modes == 0:
        return pd.DataFrame()

    has_exp = spectrum.exp_masses is not None
    if has_exp:
        exp_m = spectrum.exp_masses.detach().cpu().numpy()
        exp_e = (
            spectrum.exp_errors.detach().cpu().numpy()
            if spectrum.exp_errors is not None
            else np.full(n_modes, np.nan)
        )
        exp_r2 = (
            spectrum.exp_r2.detach().cpu().numpy()
            if spectrum.exp_r2 is not None
            else np.full(n_modes, np.nan)
        )

    m0 = float(masses[0]) if math.isfinite(float(masses[0])) and masses[0] > 0 else float("nan")

    rows = []
    for i in range(n_modes):
        m = float(masses[i])
        e = float(errors[i])
        p_start, p_end = spectrum.plateau_ranges[i]
        row = {
            "mode": i,
            "mass": m if math.isfinite(m) else float("nan"),
            "error": e if math.isfinite(e) else float("nan"),
            "plateau_start": p_start,
            "plateau_end": p_end,
            "m_n / m_0": m / m0
            if math.isfinite(m) and math.isfinite(m0) and m0 > 0
            else float("nan"),
        }
        if has_exp:
            row["exp_mass"] = float(exp_m[i]) if math.isfinite(float(exp_m[i])) else float("nan")
            row["exp_error"] = float(exp_e[i]) if math.isfinite(float(exp_e[i])) else float("nan")
            row["exp_r2"] = float(exp_r2[i]) if math.isfinite(float(exp_r2[i])) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_t0_mass_comparison_plot(
    t0_sweep: T0SweepResult,
    *,
    family_label: str = "",
) -> hv.Overlay | None:
    """Create scatter plot of mass vs t0 per mode with consensus bands."""
    if not t0_sweep.spectra:
        return None

    max_modes = max(s.n_modes for s in t0_sweep.spectra.values())
    overlays = []

    for mode_idx in range(min(max_modes, len(MODE_COLORS))):
        color = MODE_COLORS[mode_idx % len(MODE_COLORS)]
        t0s = []
        masses = []
        errors = []
        for t0_val, spectrum in sorted(t0_sweep.spectra.items()):
            if mode_idx >= spectrum.n_modes:
                continue
            m = float(spectrum.plateau_masses[mode_idx].item())
            e = float(spectrum.plateau_errors[mode_idx].item())
            if not math.isfinite(m) or m <= 0:
                continue
            t0s.append(t0_val)
            masses.append(m)
            errors.append(e if math.isfinite(e) else 0.0)

        if not t0s:
            continue

        label = f"mode {mode_idx}"
        df = pd.DataFrame({"t0": t0s, "mass": masses, "error": errors, "mode": label})
        scatter = hv.Scatter(df, kdims=["t0"], vdims=["mass", "error", "mode"]).opts(
            color=color,
            size=8,
            tools=["hover"],
        )
        overlays.append(scatter)

        # Error bars
        for t0_val, m, e in zip(t0s, masses, errors):
            if e > 0:
                seg = hv.Curve(
                    [(t0_val, m - e), (t0_val, m + e)],
                    kdims=["t0"],
                    vdims=["mass"],
                ).opts(color=color, line_width=1.5)
                overlays.append(seg)

        # Consensus band
        if t0_sweep.consensus_masses is not None and mode_idx < t0_sweep.consensus_masses.shape[0]:
            cm = float(t0_sweep.consensus_masses[mode_idx].item())
            ce = (
                float(t0_sweep.consensus_errors[mode_idx].item())
                if t0_sweep.consensus_errors is not None
                else 0.0
            )
            if math.isfinite(cm) and cm > 0:
                if not math.isfinite(ce):
                    ce = 0.0
                x_lo = min(t0s) - 0.5
                x_hi = max(t0s) + 0.5
                band = hv.Area(
                    ([x_lo, x_hi], [cm - ce] * 2, [cm + ce] * 2),
                    kdims=["t0"],
                    vdims=["y_lo", "y_hi"],
                ).opts(color=color, alpha=0.15)
                hline = hv.Curve(
                    ([x_lo, x_hi], [cm, cm]),
                    kdims=["t0"],
                    vdims=["mass"],
                ).opts(color=color, line_dash="dotdash", line_width=1)
                overlays.extend([band, hline])

    if not overlays:
        return None

    overlay = hv.Overlay(overlays)
    title = (
        f"t0 Sweep Mass Stability: {family_label}" if family_label else "t0 Sweep Mass Stability"
    )
    return overlay.opts(
        width=900,
        height=350,
        xlabel="t0",
        ylabel="mass",
        title=title,
        show_grid=True,
        legend_position="top_right",
    )


def build_t0_comparison_table(t0_sweep: T0SweepResult) -> pd.DataFrame:
    """Build a table: rows=t0 values, columns=per-mode masses + tau_diag."""
    if not t0_sweep.spectra:
        return pd.DataFrame()

    max_modes = max(s.n_modes for s in t0_sweep.spectra.values())
    rows = []
    for t0_val in sorted(t0_sweep.spectra):
        spectrum = t0_sweep.spectra[t0_val]
        row: dict[str, Any] = {"t0": t0_val, "tau_diag": spectrum.tau_diag}
        for mode_idx in range(max_modes):
            if mode_idx < spectrum.n_modes:
                m = float(spectrum.plateau_masses[mode_idx].item())
                row[f"mode_{mode_idx}_mass"] = m if math.isfinite(m) else float("nan")
            else:
                row[f"mode_{mode_idx}_mass"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def update_gevp_mass_spectrum(
    w: GEVPMassSpectrumWidgets,
    *,
    spectrum: GEVPMassSpectrum | None,
    family_label: str = "",
    dt: float = 1.0,
    t0_sweep: T0SweepResult | None = None,
) -> None:
    """Update GEVP mass spectrum widgets with results."""
    if spectrum is None:
        clear_gevp_mass_spectrum(w)
        return

    n_modes = spectrum.n_modes
    masses_np = spectrum.plateau_masses.detach().cpu().numpy()
    finite_masses = [
        f"`{float(masses_np[i]):.6g}`"
        for i in range(n_modes)
        if math.isfinite(float(masses_np[i])) and masses_np[i] > 0
    ]
    mass_str = ", ".join(finite_masses) if finite_masses else "none extracted"

    # Include exp fit masses in summary if available
    exp_str = ""
    if spectrum.exp_masses is not None:
        exp_np = spectrum.exp_masses.detach().cpu().numpy()
        finite_exp = [
            f"`{float(exp_np[i]):.6g}`"
            for i in range(n_modes)
            if math.isfinite(float(exp_np[i])) and exp_np[i] > 0
        ]
        if finite_exp:
            exp_str = f"  \nExp-fit masses: {', '.join(finite_exp)}"

    w.summary.object = (
        f"**{family_label} GEVP Multi-Mode Spectrum:** "
        f"`{n_modes}` modes, τ_diag=`{spectrum.tau_diag}`.  \n"
        f"Plateau masses: {mass_str}{exp_str}"
    )

    w.effective_mass_plot.object = build_effective_mass_plateau_plot(
        spectrum, family_label=family_label, dt=dt
    )
    w.mass_spectrum_bar.object = build_gevp_mass_bar_chart(spectrum, family_label=family_label)
    w.eigenvalue_decay_plot.object = build_eigenvalue_decay_plot(
        spectrum, family_label=family_label, dt=dt
    )
    w.mode_table.value = build_mode_summary_table(spectrum)

    # t0 sweep widgets
    if t0_sweep is not None and t0_sweep.spectra:
        w.t0_comparison_plot.object = build_t0_mass_comparison_plot(
            t0_sweep,
            family_label=family_label,
        )
        w.t0_comparison_table.value = build_t0_comparison_table(t0_sweep)
    else:
        w.t0_comparison_plot.object = None
        w.t0_comparison_table.value = pd.DataFrame()
