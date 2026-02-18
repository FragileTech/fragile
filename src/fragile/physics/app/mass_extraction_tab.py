"""Mass Extraction dashboard tab.

Feeds the strong-correlator PipelineResult into the Bayesian
multi-exponential fitter, displays extracted masses, fit diagnostics,
effective-mass plots, and correlator-vs-fit overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gvar
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.physics.app.algorithm import _algorithm_placeholder_plot
from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.mass_extraction import (
    MassExtractionConfig,
    MassExtractionResult,
    ChannelFitConfig,
    CovarianceConfig,
    FitConfig,
    PriorConfig,
    extract_masses,
)


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------


@dataclass
class MassExtractionSection:
    """Container for the mass extraction dashboard section."""

    tab: pn.Column
    status: pn.pane.Markdown
    run_button: pn.widgets.Button
    on_run: Callable[[Any], None]
    on_history_changed: Callable[[bool], None]
    on_correlators_ready: Callable[[], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class MassExtractionSettings(param.Parameterized):
    """Settings for the mass extraction pipeline."""

    covariance_method = param.ObjectSelector(
        default="uncorrelated",
        objects=["uncorrelated", "block_jackknife", "bootstrap"],
    )
    nexp = param.Integer(default=2, bounds=(1, 6))
    tmin = param.Integer(default=2, bounds=(1, 20))
    tmax_frac = param.Number(
        default=1.0,
        bounds=(0.1, 1.0),
        doc="Fraction of max_lag for tmax (1.0 = full range).",
    )
    svdcut = param.Number(default=1e-4)
    use_log_dE = param.Boolean(default=True)
    use_fastfit_seeding = param.Boolean(default=True)
    effective_mass_method = param.ObjectSelector(
        default="log_ratio",
        objects=["log_ratio", "cosh"],
    )
    include_multiscale = param.Boolean(default=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _build_meff_plot(
    result: MassExtractionResult,
    channel_name: str,
    channel_result: Any,
) -> hv.Overlay:
    """Build effective-mass scatter + error bars with fitted mass overlay."""
    # Collect effective-mass entries for this channel's variant keys
    variant_keys = channel_result.variant_keys
    curves = []
    all_means: list[np.ndarray] = []  # central values for y-range

    for vk in variant_keys:
        em = result.effective_masses.get(vk)
        if em is None or len(em.m_eff) == 0:
            continue

        t = em.t_values.astype(float)
        means = np.array([gvar.mean(m) for m in em.m_eff])
        errs = np.array([gvar.sdev(m) for m in em.m_eff])

        # Filter out zero-error / zero-mean placeholder points
        mask = errs > 0
        if not mask.any():
            continue

        t, means, errs = t[mask], means[mask], errs[mask]
        all_means.append(means)

        scatter = hv.Scatter(
            (t, means), kdims="t", vdims="m_eff", label=vk,
        )
        ebars = hv.ErrorBars(
            (t, means, errs), kdims="t", vdims=["m_eff", "yerr"],
        )
        curves.append(scatter * ebars)

    if not curves:
        return _algorithm_placeholder_plot(f"No effective mass data for {channel_name}")

    overlay = curves[0]
    for c in curves[1:]:
        overlay = overlay * c

    # Fitted ground-state mass ± error band
    gs = channel_result.ground_state_mass
    gs_mean = float(gvar.mean(gs))
    gs_err = float(gvar.sdev(gs))

    hline = hv.HLine(gs_mean).opts(color="red", line_dash="dashed", line_width=1.5)
    band = hv.HSpan(gs_mean - gs_err, gs_mean + gs_err).opts(
        color="red", alpha=0.15,
    )

    # Compute y-range from central values only (ignore error bars)
    all_vals = np.concatenate(all_means) if all_means else np.array([gs_mean])
    all_vals = np.concatenate([all_vals, [gs_mean]])
    ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    margin = max(0.1 * (ymax - ymin), 1e-6)
    ylim = (ymin - margin, ymax + margin)

    overlay = overlay * band * hline
    overlay = overlay.opts(
        hv.opts.Scatter(size=6, width=500, height=350),
        hv.opts.ErrorBars(line_width=1.5),
        hv.opts.Overlay(
            title=f"{channel_name} Effective Mass",
            legend_position="top_right",
            ylim=ylim,
        ),
    )
    return overlay


def _build_correlator_fit_plot(
    result: MassExtractionResult,
    channel_name: str,
    channel_result: Any,
) -> hv.Overlay:
    """Build correlator data + fit curve overlay (log Y)."""
    variant_keys = channel_result.variant_keys
    curves = []
    fit = getattr(result, "fit", None)

    for idx, vk in enumerate(variant_keys):
        corr = result.data.get(vk)
        if corr is None or len(corr) == 0:
            continue

        t = np.arange(len(corr), dtype=float)
        means = np.array([gvar.mean(c) for c in corr])
        errs = np.array([gvar.sdev(c) for c in corr])

        # Only plot positive values for log scale
        mask = means > 0
        if not mask.any():
            continue

        color = _CHANNEL_PALETTE[idx % len(_CHANNEL_PALETTE)]
        scatter = hv.Scatter(
            (t[mask], means[mask]), kdims="t", vdims="C(t)", label=vk,
        ).opts(color=color)
        ebars = hv.ErrorBars(
            (t[mask], means[mask], errs[mask]), kdims="t", vdims=["C(t)", "yerr"],
        ).opts(color=color)
        curves.append(scatter * ebars)

        # Overlay multi-exponential fit curve from posterior parameters
        if fit is not None and getattr(fit, "p", None) is not None:
            try:
                dE = fit.p[f"{channel_name}.dE"]
                a = fit.p[f"{channel_name}.{vk}.a"]
                b = fit.p[f"{channel_name}.{vk}.b"]
                E = np.cumsum([gvar.mean(x) for x in dE])
                t_max = float(t[mask][-1])
                t_fine = np.linspace(1, t_max, 200)
                n_states = min(len(E), len(a), len(b))
                C_fit = sum(
                    gvar.mean(a[n]) * gvar.mean(b[n]) * np.exp(-E[n] * t_fine)
                    for n in range(n_states)
                )
                # Only plot positive fit values for log scale
                fit_mask = C_fit > 0
                if np.any(fit_mask):
                    fit_curve = hv.Curve(
                        (t_fine[fit_mask], C_fit[fit_mask]),
                        kdims="t",
                        vdims="C(t)",
                        label=f"{vk} fit",
                    ).opts(color=color, line_dash="dashed", line_width=2)
                    curves.append(fit_curve)
            except KeyError:
                pass

    if not curves:
        return _algorithm_placeholder_plot(f"No correlator data for {channel_name}")

    overlay = curves[0]
    for c in curves[1:]:
        overlay = overlay * c

    overlay = overlay.opts(
        hv.opts.Scatter(size=5, width=500, height=350),
        hv.opts.ErrorBars(line_width=1.5),
        hv.opts.Overlay(
            title=f"{channel_name} Correlator",
            legend_position="top_right",
            logy=True,
        ),
    )
    return overlay


_CHANNEL_TYPE_COLORS = {
    "meson": "#1f77b4",
    "baryon": "#2ca02c",
    "glueball": "#9467bd",
    "tensor": "#e377c2",
}

# PDG reference masses (GeV) per channel group name.
# Group names match _auto_detect_channel_groups output.
PDG_REFERENCES: dict[str, tuple[str, float]] = {
    "scalar": ("\u03c3/f\u2080(500)", 0.500),
    "pseudoscalar": ("\u03c0 (0.140 GeV)", 0.13957),
    "vector": ("\u03c1 (0.775 GeV)", 0.77526),
    "axial_vector": ("a\u2081 (1.260 GeV)", 1.260),
    "nucleon": ("p (0.938 GeV)", 0.938272),
    "glueball": ("0\u207a\u207a (1.710 GeV)", 1.710),
    "tensor": ("a\u2082 (1.318 GeV)", 1.3183),
}


_CHANNEL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _build_mass_spectrum_bar(result: MassExtractionResult) -> hv.Overlay:
    """Build a scatter+errorbars chart of ground-state masses (log y-axis).

    Channels are ordered lightest-to-heaviest using PDG reference masses.
    Channels without a PDG entry are appended at the end sorted by extracted mass.
    """
    if not result.channels:
        return _algorithm_placeholder_plot("No channels to display.")

    # Collect per-channel data
    entries: list[tuple[str, float, float]] = []
    for name, ch in result.channels.items():
        gs_mean = float(gvar.mean(ch.ground_state_mass))
        gs_err = float(gvar.sdev(ch.ground_state_mass))
        entries.append((name, gs_mean, gs_err))

    # Sort by PDG mass (lightest first); channels without PDG go last by extracted mass
    def _sort_key(entry: tuple[str, float, float]) -> tuple[int, float]:
        name, gs_mean, _ = entry
        if name in PDG_REFERENCES:
            return (0, PDG_REFERENCES[name][1])
        return (1, gs_mean)

    entries.sort(key=_sort_key)

    names = [e[0] for e in entries]
    means = [e[1] for e in entries]
    errs = [e[2] for e in entries]
    colors = [_CHANNEL_PALETTE[i % len(_CHANNEL_PALETTE)] for i in range(len(entries))]

    scatter = hv.Scatter(
        list(zip(names, means, colors)),
        kdims="Channel",
        vdims=["Mass", "color"],
    ).opts(color="color", size=10, width=700, height=300, xrotation=45, logy=True)
    ebars = hv.ErrorBars(
        list(zip(names, means, errs)),
        kdims="Channel",
        vdims=["Mass", "yerr"],
    ).opts(line_width=2)
    return (scatter * ebars).opts(
        hv.opts.Overlay(title="Ground-State Mass Spectrum"),
    )


def _build_pdg_comparison_plot(
    result: MassExtractionResult,
    anchor_channel: str,
) -> hv.Overlay:
    """Bar chart of predicted masses (GeV) using *anchor_channel* for scale setting.

    For each channel with a PDG reference, compute
    ``m_predicted = m_lattice * (m_PDG_anchor / m_lattice_anchor)``
    and overlay the PDG reference values as diamond markers.
    """
    if anchor_channel not in result.channels or anchor_channel not in PDG_REFERENCES:
        return _algorithm_placeholder_plot(
            f"Anchor '{anchor_channel}' not available in extraction results.",
        )

    anchor_mass = result.channels[anchor_channel].ground_state_mass
    anchor_mean = float(gvar.mean(anchor_mass))
    if anchor_mean <= 0:
        return _algorithm_placeholder_plot("Anchor mass is non-positive.")

    _, anchor_pdg = PDG_REFERENCES[anchor_channel]
    scale = anchor_pdg / anchor_mass  # gvar propagates

    names, pred_means, pred_errs, pdg_vals, colors = [], [], [], [], []
    for name, ch in result.channels.items():
        if name not in PDG_REFERENCES:
            continue
        m_lat = ch.ground_state_mass
        if float(gvar.mean(m_lat)) <= 0:
            continue
        m_pred = m_lat * scale
        label, pdg_mass = PDG_REFERENCES[name]
        names.append(label)
        pred_means.append(float(gvar.mean(m_pred)))
        pred_errs.append(float(gvar.sdev(m_pred)))
        pdg_vals.append(pdg_mass)
        colors.append(_CHANNEL_TYPE_COLORS.get(ch.channel_type, "#7f7f7f"))

    if not names:
        return _algorithm_placeholder_plot("No channels with PDG references found.")

    bars = hv.Bars(
        list(zip(names, pred_means, colors)),
        kdims="Channel",
        vdims=["Mass (GeV)", "color"],
    ).opts(color="color", width=700, height=350, xrotation=45)
    ebars = hv.ErrorBars(
        list(zip(names, pred_means, pred_errs)),
        kdims="Channel",
        vdims=["Mass (GeV)", "yerr"],
    ).opts(line_width=2)
    pdg_scatter = hv.Scatter(
        list(zip(names, pdg_vals)),
        kdims="Channel",
        vdims="Mass (GeV)",
        label="PDG",
    ).opts(color="red", marker="diamond", size=12)

    anchor_label, _ = PDG_REFERENCES[anchor_channel]
    return (bars * ebars * pdg_scatter).opts(
        hv.opts.Overlay(
            title=f"PDG Comparison (anchor: {anchor_label})",
            legend_position="top_right",
        ),
    )


def _build_ratio_comparison(
    result: MassExtractionResult,
) -> tuple[hv.Overlay, pd.DataFrame]:
    """Scatter of extracted vs PDG mass ratios for all unique channel pairs.

    Points on the diagonal ``y = x`` indicate perfect agreement.
    Returns the plot overlay and a summary DataFrame.
    """
    matched = {
        name: ch
        for name, ch in result.channels.items()
        if name in PDG_REFERENCES and float(gvar.mean(ch.ground_state_mass)) > 0
    }
    names = sorted(matched)

    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m_a = matched[a].ground_state_mass
            m_b = matched[b].ground_state_mass
            ratio_ext = m_a / m_b  # gvar propagates
            _, pdg_a = PDG_REFERENCES[a]
            _, pdg_b = PDG_REFERENCES[b]
            ratio_pdg = pdg_a / pdg_b
            ext_mean = float(gvar.mean(ratio_ext))
            ext_err = float(gvar.sdev(ratio_ext))
            tension = abs(ext_mean - ratio_pdg) / ext_err if ext_err > 0 else float("inf")
            error_pct = (ext_mean - ratio_pdg) / ratio_pdg * 100.0 if ratio_pdg != 0 else float("inf")
            matches = "YES" if abs(ext_mean - ratio_pdg) <= ext_err else "NO"
            rows.append({
                "Ratio": f"{a}/{b}",
                "Extracted": round(ext_mean, 4),
                "Error": round(ext_err, 4),
                "PDG": round(ratio_pdg, 4),
                "Error (%)": round(error_pct, 2),
                "Matches": matches,
                "Tension (\u03c3)": round(tension, 2),
            })

    cols = ["Ratio", "Extracted", "Error", "PDG", "Error (%)", "Matches", "Tension (\u03c3)"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    if not rows:
        return _algorithm_placeholder_plot("Need \u22652 channels with PDG refs for ratios."), df

    pdg_vals = [r["PDG"] for r in rows]
    ext_vals = [r["Extracted"] for r in rows]
    ext_errs = [r["Error"] for r in rows]
    labels = [r["Ratio"] for r in rows]

    scatter = hv.Scatter(
        list(zip(pdg_vals, ext_vals, labels)),
        kdims="PDG ratio",
        vdims=["Extracted ratio", "label"],
    ).opts(size=8, color="#1f77b4", tools=["hover"])
    ebars = hv.ErrorBars(
        list(zip(pdg_vals, ext_vals, ext_errs)),
        kdims="PDG ratio",
        vdims=["Extracted ratio", "yerr"],
    ).opts(line_width=1.5)

    lo = min(min(pdg_vals), min(ext_vals)) * 0.9
    hi = max(max(pdg_vals), max(ext_vals)) * 1.1
    diagonal = hv.Curve([(lo, lo), (hi, hi)], kdims="PDG ratio", vdims="Extracted ratio").opts(
        color="gray", line_dash="dashed", line_width=1,
    )
    overlay = (diagonal * scatter * ebars).opts(
        hv.opts.Overlay(
            title="Mass Ratios: Extracted vs PDG",
            width=500,
            height=400,
        ),
    )
    return overlay, df


def _build_anchor_spread_analysis(
    result: MassExtractionResult,
) -> tuple[hv.Overlay, pd.DataFrame, pd.DataFrame]:
    """Scatter plot + tables showing how predicted GeV masses vary across anchors.

    For every (channel, anchor) pair where both appear in *PDG_REFERENCES*,
    compute ``m_pred = m_lattice_channel * (pdg_anchor / m_lattice_anchor)``
    and measure the deviation from the PDG value.

    Returns ``(overlay, summary_df, detail_df)``.
    """
    matched = {
        name: ch
        for name, ch in result.channels.items()
        if name in PDG_REFERENCES and float(gvar.mean(ch.ground_state_mass)) > 0
    }

    detail_cols = [
        "Channel", "Anchor", "Predicted (GeV)", "Error (GeV)",
        "PDG (GeV)", "Deviation (%)",
    ]
    summary_cols = [
        "Channel", "PDG (GeV)", "Mean Pred (GeV)", "Std Across Anchors (GeV)",
        "Spread (%)", "Best Anchor", "Worst Anchor",
    ]
    empty_overlay = _algorithm_placeholder_plot("Need ≥2 channels with PDG refs.")
    empty_detail = pd.DataFrame(columns=detail_cols)
    empty_summary = pd.DataFrame(columns=summary_cols)

    if len(matched) < 2:
        return empty_overlay, empty_summary, empty_detail

    detail_rows: list[dict[str, Any]] = []
    # {channel_name: [(anchor_name, pred_mean, pred_err, deviation_pct)]}
    per_channel: dict[str, list[tuple[str, float, float, float]]] = {}

    for ch_name, ch in matched.items():
        for anc_name, anc in matched.items():
            if ch_name == anc_name:
                continue
            _, pdg_anc = PDG_REFERENCES[anc_name]
            _, pdg_ch = PDG_REFERENCES[ch_name]
            scale = pdg_anc / anc.ground_state_mass  # gvar
            m_pred = ch.ground_state_mass * scale     # gvar
            pred_mean = float(gvar.mean(m_pred))
            pred_err = float(gvar.sdev(m_pred))
            dev_pct = (pred_mean - pdg_ch) / pdg_ch * 100.0

            detail_rows.append({
                "Channel": ch_name,
                "Anchor": anc_name,
                "Predicted (GeV)": round(pred_mean, 4),
                "Error (GeV)": round(pred_err, 4),
                "PDG (GeV)": round(pdg_ch, 4),
                "Deviation (%)": round(dev_pct, 2),
            })
            per_channel.setdefault(ch_name, []).append(
                (anc_name, pred_mean, pred_err, dev_pct),
            )

    detail_df = pd.DataFrame(detail_rows, columns=detail_cols) if detail_rows else empty_detail

    # --- Summary (one row per channel) ---
    summary_rows: list[dict[str, Any]] = []
    for ch_name, entries in per_channel.items():
        _, pdg_ch = PDG_REFERENCES[ch_name]
        pred_means = np.array([e[1] for e in entries])
        mean_pred = float(np.mean(pred_means))
        std_pred = float(np.std(pred_means, ddof=0))
        spread_pct = std_pred / pdg_ch * 100.0
        abs_devs = [abs(e[3]) for e in entries]
        best_idx = int(np.argmin(abs_devs))
        worst_idx = int(np.argmax(abs_devs))
        summary_rows.append({
            "Channel": ch_name,
            "PDG (GeV)": round(pdg_ch, 4),
            "Mean Pred (GeV)": round(mean_pred, 4),
            "Std Across Anchors (GeV)": round(std_pred, 4),
            "Spread (%)": round(spread_pct, 2),
            "Best Anchor": entries[best_idx][0],
            "Worst Anchor": entries[worst_idx][0],
        })
    summary_df = (
        pd.DataFrame(summary_rows, columns=summary_cols)
        if summary_rows else empty_summary
    )

    # --- Plot: one colour per anchor, PDG as red diamonds ---
    anchor_names = sorted({r["Anchor"] for r in detail_rows})
    anchor_color = {
        a: _CHANNEL_PALETTE[i % len(_CHANNEL_PALETTE)]
        for i, a in enumerate(anchor_names)
    }

    scatter_layers = []
    for anc in anchor_names:
        subset = [r for r in detail_rows if r["Anchor"] == anc]
        scatter_layers.append(
            hv.Scatter(
                [(r["Channel"], r["Predicted (GeV)"]) for r in subset],
                kdims="Channel",
                vdims="Mass (GeV)",
                label=anc,
            ).opts(
                color=anchor_color[anc],
                size=8,
                jitter=0.3,
            )
        )

    # PDG reference diamonds
    pdg_points = [
        (ch_name, PDG_REFERENCES[ch_name][1])
        for ch_name in per_channel
    ]
    pdg_scatter = hv.Scatter(
        pdg_points, kdims="Channel", vdims="Mass (GeV)", label="PDG",
    ).opts(color="red", marker="diamond", size=12)

    overlay = scatter_layers[0]
    for s in scatter_layers[1:]:
        overlay = overlay * s
    overlay = overlay * pdg_scatter
    overlay = overlay.opts(
        hv.opts.Overlay(
            title="Anchor Spread: Predicted Mass by Anchor",
            legend_position="top_right",
            width=700,
            height=400,
            logy=True,
        ),
        hv.opts.Scatter(xrotation=45),
    )

    return overlay, summary_df, detail_df


def _build_amplitude_table(channel_result: Any) -> pd.DataFrame:
    """Build a DataFrame of per-variant amplitudes for a channel."""
    rows = []
    for vk in channel_result.variant_keys:
        amps = channel_result.amplitudes.get(vk)
        if amps is None or len(amps) == 0:
            continue
        a0 = amps[0]
        a0_mean = float(gvar.mean(a0))
        a0_err = float(gvar.sdev(a0))
        significance = abs(a0_mean) / a0_err if a0_err > 0 else 0.0
        rows.append({
            "Variant": vk,
            "a_0": f"{a0_mean:.6f}",
            "Error": f"{a0_err:.6f}",
            "Significance (σ)": f"{significance:.1f}",
        })
    return pd.DataFrame(rows)


def _build_intra_channel_ratio_table(ch: Any) -> pd.DataFrame:
    """Build a DataFrame of excited-to-ground-state mass ratios within a channel."""
    cols = ["Ratio", "Value", "Error", "Error (%)"]
    if len(ch.energy_levels) < 2:
        return pd.DataFrame(columns=cols)
    E_0 = ch.energy_levels[0]
    rows = []
    for i in range(1, len(ch.energy_levels)):
        ratio = ch.energy_levels[i] / E_0
        mean = float(gvar.mean(ratio))
        err = float(gvar.sdev(ratio))
        pct = abs(err / mean) * 100.0 if mean != 0 else float("inf")
        rows.append({
            "Ratio": f"E_{i}/E_0",
            "Value": f"{mean:.6f}",
            "Error": f"{err:.6f}",
            "Error (%)": f"{pct:.2f}",
        })
    return pd.DataFrame(rows, columns=cols)


def _build_cross_channel_ratio_table(result: MassExtractionResult) -> pd.DataFrame:
    """Build a DataFrame of ground-state mass ratios across channels."""
    cols = ["Ratio", "Value", "Error", "Error (%)"]
    mesons = [
        (name, ch) for name, ch in result.channels.items()
        if ch.channel_type == "meson"
    ]
    baryons = [
        (name, ch) for name, ch in result.channels.items()
        if ch.channel_type == "baryon"
    ]
    rows = []

    def _add_pair(a_name, a_ch, b_name, b_ch):
        m_b = float(gvar.mean(b_ch.ground_state_mass))
        if m_b == 0:
            return
        ratio = a_ch.ground_state_mass / b_ch.ground_state_mass
        mean = float(gvar.mean(ratio))
        err = float(gvar.sdev(ratio))
        pct = abs(err / mean) * 100.0 if mean != 0 else float("inf")
        rows.append({
            "Ratio": f"{a_name}/{b_name}",
            "Value": f"{mean:.6f}",
            "Error": f"{err:.6f}",
            "Error (%)": f"{pct:.2f}",
        })

    # Meson/baryon pairs first
    for a_name, a_ch in mesons:
        for b_name, b_ch in baryons:
            _add_pair(a_name, a_ch, b_name, b_ch)
    # Meson/meson pairs
    for i, (a_name, a_ch) in enumerate(mesons):
        for b_name, b_ch in mesons[i + 1:]:
            _add_pair(a_name, a_ch, b_name, b_ch)
    # Baryon/baryon pairs
    for i, (a_name, a_ch) in enumerate(baryons):
        for b_name, b_ch in baryons[i + 1:]:
            _add_pair(a_name, a_ch, b_name, b_ch)

    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _build_diagnostics_container(result: MassExtractionResult) -> list:
    """Build diagnostics widgets: global fit quality + per-channel summary."""
    d = result.diagnostics
    global_md = (
        f"**chi2** = {d.chi2:.2f} &nbsp;|&nbsp; "
        f"**dof** = {d.dof} &nbsp;|&nbsp; "
        f"**chi2/dof** = {d.chi2_per_dof:.3f} &nbsp;|&nbsp; "
        f"**Q** = {d.Q:.4f} &nbsp;|&nbsp; "
        f"**logGBF** = {d.logGBF:.2f} &nbsp;|&nbsp; "
        f"**nit** = {d.nit} &nbsp;|&nbsp; "
        f"**svdcut** = {d.svdcut:.1e}"
    )

    rows = []
    for name, ch in result.channels.items():
        rows.append({
            "Group": name,
            "Type": ch.channel_type,
            "Ground Mass": f"{gvar.mean(ch.ground_state_mass):.6f}",
            "Levels": len(ch.energy_levels),
            "#Variants": len(ch.variant_keys),
            "Variant keys": ", ".join(ch.variant_keys),
        })

    return [global_md, pd.DataFrame(rows)]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_mass_extraction_tab(
    *,
    state: dict[str, Any],
    run_tab_computation: Callable[
        [dict[str, Any], pn.pane.Markdown, str, Callable[[RunHistory], None]], None
    ],
    correlator_state_key: str = "strong_correlator_output",
    output_state_key: str = "mass_extraction_output",
    tab_label: str = "Strong Force Mass",
    button_label: str = "Extract Strong Masses",
    source_label: str = "Strong Correlators",
    computation_label: str = "strong force mass extraction",
) -> MassExtractionSection:
    """Build the Mass Extraction tab with callbacks."""

    settings = MassExtractionSettings()
    status = pn.pane.Markdown(
        f"**{tab_label}:** Run {source_label} first, then click {button_label}.",
        sizing_mode="stretch_width",
    )
    run_button = pn.widgets.Button(
        name=button_label,
        button_type="primary",
        min_width=260,
        sizing_mode="stretch_width",
        disabled=True,
    )

    # -- Widgets --
    mass_spectrum_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run extraction to show mass spectrum."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    mass_summary_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    diagnostics_container = pn.Column(
        pn.pane.Markdown("", sizing_mode="stretch_width"),
        pn.widgets.Tabulator(
            pd.DataFrame(), pagination=None, show_index=False,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    channel_selector = pn.widgets.Select(
        name="Channel",
        options=[],
        sizing_mode="stretch_width",
    )
    channel_detail_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    amplitude_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    intra_ratio_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    cross_ratio_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    placeholder = _algorithm_placeholder_plot("Run extraction to show plots.")
    meff_plot = pn.pane.HoloViews(placeholder, sizing_mode="stretch_width", linked_axes=False)
    correlator_fit_plot = pn.pane.HoloViews(
        placeholder, sizing_mode="stretch_width", linked_axes=False,
    )

    # -- PDG comparison widgets --
    anchor_selector = pn.widgets.Select(
        name="Scale Anchor",
        options=list(PDG_REFERENCES.keys()),
        value="pseudoscalar",
        sizing_mode="stretch_width",
    )
    pdg_comparison_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run extraction for PDG comparison."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    ratio_comparison_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run extraction for ratio comparison."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    ratio_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    # -- Anchor spread widgets --
    anchor_spread_plot = pn.pane.HoloViews(
        _algorithm_placeholder_plot("Run extraction for anchor spread analysis."),
        sizing_mode="stretch_width",
        linked_axes=False,
    )
    anchor_spread_summary = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )
    anchor_spread_detail = pn.widgets.Tabulator(
        pd.DataFrame(),
        pagination=None,
        show_index=False,
        sizing_mode="stretch_width",
    )

    # -- Channel key selection widgets --
    channel_key_selectors: dict[str, pn.widgets.MultiSelect] = {}
    channel_tmax_frac_sliders: dict[str, pn.widgets.FloatSlider] = {}

    channel_key_selection_container = pn.Column(
        pn.pane.Markdown(
            f"*Run {source_label} to populate channel operator selections.*",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    # -- Settings panel --
    settings_panel = pn.Param(
        settings,
        parameters=[
            "covariance_method",
            "nexp",
            "tmin",
            "tmax_frac",
            "svdcut",
            "use_log_dE",
            "use_fastfit_seeding",
            "effective_mass_method",
            "include_multiscale",
        ],
        show_name=False,
        default_layout=type("FitGrid", (pn.GridBox,), {"ncols": 2}),
    )

    # -- Refresh helpers --

    def _refresh_all():
        result: MassExtractionResult | None = state.get(output_state_key)
        if result is None:
            return

        # Mass spectrum bar chart
        mass_spectrum_plot.object = _build_mass_spectrum_bar(result)

        # Mass summary table
        rows = []
        for name, ch in result.channels.items():
            gs_mean = float(gvar.mean(ch.ground_state_mass))
            gs_err = float(gvar.sdev(ch.ground_state_mass))
            err_pct = (gs_err / gs_mean * 100) if gs_mean != 0 else float("inf")
            rows.append({
                "Channel": ch.name,
                "Type": ch.channel_type,
                "Ground Mass": f"{gs_mean:.6f}",
                "Error": f"{gs_err:.6f}",
                "Error %": f"{err_pct:.2f}%",
                "chi2/dof": f"{result.diagnostics.chi2_per_dof:.3f}",
                "Q": f"{result.diagnostics.Q:.4f}",
                "Variants": ", ".join(ch.variant_keys),
            })
        mass_summary_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Diagnostics container
        diag_parts = _build_diagnostics_container(result)
        diagnostics_container[0].object = diag_parts[0]  # global markdown
        diagnostics_container[1].value = diag_parts[1]   # per-channel table

        # Channel selector
        channel_names = list(result.channels.keys())
        channel_selector.options = channel_names
        if channel_names:
            channel_selector.value = channel_names[0]

        _refresh_channel_detail()
        _refresh_pdg_comparison()
        cross_ratio_table.value = _build_cross_channel_ratio_table(result)

    def _refresh_channel_detail(*_args):
        result: MassExtractionResult | None = state.get(output_state_key)
        if result is None:
            return

        selected = channel_selector.value
        if selected is None or selected not in result.channels:
            channel_detail_table.value = pd.DataFrame()
            amplitude_table.value = pd.DataFrame()
            intra_ratio_table.value = pd.DataFrame()
            meff_plot.object = _algorithm_placeholder_plot("Select a channel.")
            correlator_fit_plot.object = _algorithm_placeholder_plot("Select a channel.")
            return

        ch = result.channels[selected]

        # Energy level detail table
        rows = []
        for i, E_n in enumerate(ch.energy_levels):
            e_mean = gvar.mean(E_n)
            e_err = gvar.sdev(E_n)
            e_err_pct = abs(e_err / e_mean) * 100.0 if e_mean != 0 else float("inf")
            row = {
                "Level": i,
                "E_n": f"{e_mean:.6f}",
                "Error": f"{e_err:.6f}",
                "Error (%)": f"{e_err_pct:.2f}",
            }
            if ch.dE is not None and i < len(ch.dE):
                de_mean = gvar.mean(ch.dE[i])
                de_err = gvar.sdev(ch.dE[i])
                de_err_pct = abs(de_err / de_mean) * 100.0 if de_mean != 0 else float("inf")
                row["dE_n"] = f"{de_mean:.6f}"
                row["dE Error"] = f"{de_err:.6f}"
                row["dE Error (%)"] = f"{de_err_pct:.2f}"
            rows.append(row)
        channel_detail_table.value = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Amplitude table
        amplitude_table.value = _build_amplitude_table(ch)

        # Intra-channel energy level ratios
        intra_ratio_table.value = _build_intra_channel_ratio_table(ch)

        # Effective mass plot
        meff_plot.object = _build_meff_plot(result, selected, ch)

        # Correlator + fit plot
        correlator_fit_plot.object = _build_correlator_fit_plot(result, selected, ch)

    channel_selector.param.watch(_refresh_channel_detail, "value")

    # -- PDG comparison refresh --

    def _refresh_pdg_comparison(*_args):
        result: MassExtractionResult | None = state.get(output_state_key)
        if result is None:
            return
        pdg_comparison_plot.object = _build_pdg_comparison_plot(
            result, anchor_selector.value,
        )
        ratio_overlay, ratio_df = _build_ratio_comparison(result)
        ratio_comparison_plot.object = ratio_overlay
        ratio_table.value = ratio_df
        spread_overlay, spread_summary_df, spread_detail_df = (
            _build_anchor_spread_analysis(result)
        )
        anchor_spread_plot.object = spread_overlay
        anchor_spread_summary.value = spread_summary_df
        anchor_spread_detail.value = spread_detail_df

    anchor_selector.param.watch(_refresh_pdg_comparison, "value")

    # -- Compute callback --

    def on_run(_event):
        def _compute(_history: RunHistory):
            pipeline_result = state.get(correlator_state_key)
            if pipeline_result is None:
                status.object = f"**Error:** Run {source_label} first for {tab_label}."
                return

            # Build config from settings

            config = MassExtractionConfig(
                covariance=CovarianceConfig(method=str(settings.covariance_method)),
                fit=FitConfig(svdcut=float(settings.svdcut)),
                compute_effective_mass=True,
                effective_mass_method=str(settings.effective_mass_method),
                include_multiscale=bool(settings.include_multiscale),
            )

            # Apply per-channel fit defaults via channel_groups auto-detection
            # We leave channel_groups empty for auto-detection and override
            # the default ChannelFitConfig values via a monkey-patch on the
            # auto-detected groups after the first call.  Simpler: just call
            # extract_masses and let it auto-detect, then the per-channel
            # configs will use defaults.  We'll set defaults on ChannelFitConfig.
            ChannelFitConfig.__init__.__defaults__  # noqa: B018 – reference only
            # Actually we cannot cleanly override the auto-detected defaults
            # without modifying config.  Instead, run extraction, then if the
            # user wants custom tmin/nexp we need groups.  Let's auto-detect
            # first, patch, then run.
            from fragile.physics.mass_extraction.pipeline import _auto_detect_channel_groups

            groups = _auto_detect_channel_groups(
                list(pipeline_result.correlators.keys()),
                include_multiscale=bool(settings.include_multiscale),
            )

            # Filter correlator keys per channel using selector widgets
            for g in groups:
                selector = channel_key_selectors.get(g.name)
                if selector is not None:
                    selected = set(selector.value)
                    g.correlator_keys = [
                        k for k in g.correlator_keys if k in selected
                    ]
            groups = [g for g in groups if g.correlator_keys]

            if not groups:
                status.object = (
                    "**Error:** No correlator keys selected. "
                    "Open *Channel Operator Selection* and select at least "
                    "one key per channel."
                )
                return

            for g in groups:
                # Per-channel tmax_frac from slider, fallback to global setting
                ch_tmax_frac = float(settings.tmax_frac)
                slider = channel_tmax_frac_sliders.get(g.name)
                if slider is not None:
                    ch_tmax_frac = float(slider.value)

                ch_tmax = None
                if ch_tmax_frac < 1.0:
                    for key in g.correlator_keys:
                        if key in pipeline_result.correlators:
                            corr = pipeline_result.correlators[key]
                            clen = corr.shape[-1] if hasattr(corr, "shape") else len(corr)
                            candidate = max(int(settings.tmin) + 1, int(clen * ch_tmax_frac))
                            if ch_tmax is None or candidate > ch_tmax:
                                ch_tmax = candidate

                g.fit = ChannelFitConfig(
                    tmin=int(settings.tmin),
                    tmax=ch_tmax,
                    nexp=int(settings.nexp),
                    use_log_dE=bool(settings.use_log_dE),
                )
                g.prior = PriorConfig(use_fastfit_seeding=bool(settings.use_fastfit_seeding))
            config.channel_groups = groups

            result = extract_masses(pipeline_result, config)
            state[output_state_key] = result

            _refresh_all()

            n_ch = len(result.channels)
            n_eff = len(result.effective_masses)
            status.object = (
                f"**Complete:** {n_ch} channel groups, "
                f"{n_eff} effective-mass curves.  "
                f"chi2/dof = {result.diagnostics.chi2_per_dof:.3f}, "
                f"Q = {result.diagnostics.Q:.4f}."
            )

        run_tab_computation(state, status, computation_label, _compute)

    run_button.on_click(on_run)

    # -- on_history_changed --

    def on_history_changed(defer: bool) -> None:
        run_button.disabled = True
        status.object = (
            f"**{tab_label}:** Run {source_label} first, "
            f"then click {button_label}."
        )
        # Always reset channel key selectors when history changes
        channel_key_selectors.clear()
        channel_tmax_frac_sliders.clear()
        channel_key_selection_container.clear()
        channel_key_selection_container.append(
            pn.pane.Markdown(
                f"*Run {source_label} to populate channel operator selections.*",
                sizing_mode="stretch_width",
            ),
        )
        if defer:
            return
        state[output_state_key] = None
        mass_spectrum_plot.object = _algorithm_placeholder_plot(
            "Run extraction to show mass spectrum.",
        )
        mass_summary_table.value = pd.DataFrame()
        diagnostics_container[0].object = ""
        diagnostics_container[1].value = pd.DataFrame()
        channel_selector.options = []
        channel_detail_table.value = pd.DataFrame()
        amplitude_table.value = pd.DataFrame()
        intra_ratio_table.value = pd.DataFrame()
        cross_ratio_table.value = pd.DataFrame()
        placeholder = _algorithm_placeholder_plot("Run extraction to show plots.")
        meff_plot.object = placeholder
        correlator_fit_plot.object = placeholder
        pdg_comparison_plot.object = _algorithm_placeholder_plot(
            "Run extraction for PDG comparison.",
        )
        ratio_comparison_plot.object = _algorithm_placeholder_plot(
            "Run extraction for ratio comparison.",
        )
        ratio_table.value = pd.DataFrame()
        anchor_spread_plot.object = _algorithm_placeholder_plot(
            "Run extraction for anchor spread analysis.",
        )
        anchor_spread_summary.value = pd.DataFrame()
        anchor_spread_detail.value = pd.DataFrame()

    # -- on_correlators_ready --

    def on_correlators_ready() -> None:
        run_button.disabled = False
        status.object = (
            f"**{tab_label} ready:** {source_label} available. "
            f"Click {button_label}."
        )

        # Populate channel key selection widgets from pipeline result
        pipeline_result = state.get(correlator_state_key)
        if pipeline_result is not None:
            from fragile.physics.mass_extraction.pipeline import (
                _auto_detect_channel_groups,
            )

            groups = _auto_detect_channel_groups(
                list(pipeline_result.correlators.keys()),
                include_multiscale=True,
            )
            channel_key_selectors.clear()
            channel_tmax_frac_sliders.clear()
            widget_groups: list[pn.Column] = []
            for g in groups:
                keys = sorted(g.correlator_keys)
                if not keys:
                    continue
                selector = pn.widgets.MultiSelect(
                    name=g.name,
                    options=keys,
                    value=keys,
                    size=min(len(keys), 8),
                    sizing_mode="stretch_width",
                )
                channel_key_selectors[g.name] = selector
                tmax_slider = pn.widgets.FloatSlider(
                    name=f"{g.name} tmax_frac",
                    start=0.1,
                    end=1.0,
                    step=0.05,
                    value=float(settings.tmax_frac),
                    sizing_mode="stretch_width",
                )
                channel_tmax_frac_sliders[g.name] = tmax_slider
                widget_groups.append(
                    pn.Column(selector, tmax_slider, sizing_mode="stretch_width"),
                )

            channel_key_selection_container.clear()
            if widget_groups:
                # Lay out in rows of 3
                for i in range(0, len(widget_groups), 3):
                    channel_key_selection_container.append(
                        pn.Row(*widget_groups[i : i + 3], sizing_mode="stretch_width"),
                    )
            else:
                channel_key_selection_container.append(
                    pn.pane.Markdown(
                        "*No channel groups detected.*",
                        sizing_mode="stretch_width",
                    ),
                )

    # -- Tab layout --

    info_note = pn.pane.Alert(
        (
            f"**{tab_label} Extraction:** Performs Bayesian multi-exponential fits "
            f"on the {source_label.lower()} output to extract particle masses "
            "with error bars.  Displays effective-mass plateaus, fit "
            "quality diagnostics, and per-channel energy levels."
        ),
        alert_type="info",
        sizing_mode="stretch_width",
    )

    tab = pn.Column(
        status,
        info_note,
        pn.Row(run_button, sizing_mode="stretch_width"),
        pn.Accordion(
            ("Fit Settings", settings_panel),
            ("Channel Operator Selection", channel_key_selection_container),
            sizing_mode="stretch_width",
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Spectrum"),
        mass_spectrum_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### PDG Mass Comparison"),
        pn.Row(anchor_selector, sizing_mode="stretch_width"),
        pdg_comparison_plot,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Ratios vs PDG"),
        ratio_comparison_plot,
        ratio_table,
        pn.layout.Divider(),
        pn.pane.Markdown("### Anchor Spread (Systematic Error)"),
        anchor_spread_plot,
        anchor_spread_summary,
        pn.pane.Markdown("#### Per-Anchor Detail"),
        anchor_spread_detail,
        pn.layout.Divider(),
        pn.pane.Markdown("### Mass Summary"),
        mass_summary_table,
        pn.pane.Markdown("### Cross-Channel Mass Ratios"),
        cross_ratio_table,
        pn.pane.Markdown("### Fit Diagnostics"),
        diagnostics_container,
        pn.layout.Divider(),
        pn.pane.Markdown("### Channel Detail"),
        channel_selector,
        channel_detail_table,
        pn.pane.Markdown("#### Variant Amplitudes"),
        amplitude_table,
        pn.pane.Markdown("#### Energy Level Ratios"),
        intra_ratio_table,
        pn.Row(correlator_fit_plot, meff_plot, sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )

    return MassExtractionSection(
        tab=tab,
        status=status,
        run_button=run_button,
        on_run=on_run,
        on_history_changed=on_history_changed,
        on_correlators_ready=on_correlators_ready,
    )
