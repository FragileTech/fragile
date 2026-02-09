"""Plotting functions for Dirac spectrum analysis.

Builds HoloViews plots from :class:`DiracSpectrumResult`.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd

from fragile.fractalai.qft.dirac_spectrum import (
    DiracSpectrumResult,
    SECTOR_NAMES,
    build_fermion_comparison,
    build_fermion_ratio_comparison,
)


SECTOR_COLORS = {
    "up_quark": "#e45756",
    "down_quark": "#4c78a8",
    "neutrino": "#f58518",
    "charged_lepton": "#72b7b2",
}

SECTOR_LABELS = {
    "up_quark": "Up Quark (cloner + strong visc.)",
    "down_quark": "Down Quark (persister + strong visc.)",
    "neutrino": "Neutrino (cloner + weak visc.)",
    "charged_lepton": "Charged Lepton (persister + weak visc.)",
}

GENERATION_COLORS = ["#e45756", "#4c78a8", "#f58518", "#72b7b2", "#b07aa1"]


def _color_by_generation(
    singular_values: np.ndarray,
    boundaries: list[int],
) -> np.ndarray:
    """Assign a generation index to each singular value given boundaries."""
    gen = np.zeros(len(singular_values), dtype=int)
    for b in boundaries:
        gen[b:] += 1
    return gen


# ---------------------------------------------------------------------------
# Individual plot builders
# ---------------------------------------------------------------------------


def build_full_spectrum_plot(result: DiracSpectrumResult) -> hv.Element:
    """Scatter plot of all singular values colored by generation cluster."""
    sv = result.full_singular_values
    if len(sv) == 0:
        return hv.Text(0, 0, "No singular values").opts(title="Full Spectrum")

    # Filter to positive values for log-scale safety
    pos_mask = sv > 0
    sv_pos = sv[pos_mask]
    idx_pos = np.where(pos_mask)[0]
    if len(sv_pos) == 0:
        return hv.Text(0, 0, "All singular values are zero").opts(title="Full Spectrum")

    gen = _color_by_generation(sv, result.full_generation_boundaries)
    gen_pos = gen[pos_mask]
    n_gen = gen_pos.max() + 1

    overlays = []
    for g in range(n_gen):
        mask = gen_pos == g
        color = GENERATION_COLORS[g % len(GENERATION_COLORS)]
        df = pd.DataFrame({"index": idx_pos[mask], "sigma": sv_pos[mask]})
        pts = hv.Scatter(df, kdims=["index"], vdims=["sigma"]).opts(
            color=color, alpha=0.8, size=5, tools=["hover"],
        )
        overlays.append(pts)

    plot = hv.Overlay(overlays).opts(
        width=650, height=400,
        title=f"Full K̃ Spectrum ({len(sv)} singular values)",
        xlabel="Singular value index",
        ylabel="σ",
        logy=True,
    )
    return plot


def build_sector_spectra_plots(result: DiracSpectrumResult) -> dict[str, hv.Element]:
    """Individual sector SVD spectrum plots, keyed by sector name."""
    plots = {}
    for sector_name in SECTOR_NAMES.values():
        spec = result.sectors.get(sector_name)
        if spec is None or len(spec.singular_values) == 0:
            n_w = spec.n_walkers if spec is not None else 0
            plots[sector_name] = hv.Text(
                0, 0, f"No data ({sector_name}, {n_w} walkers)"
            ).opts(title=SECTOR_LABELS.get(sector_name, sector_name))
            continue

        sv = spec.singular_values
        pos_mask = sv > 0
        sv_pos = sv[pos_mask]
        idx_pos = np.where(pos_mask)[0]
        if len(sv_pos) == 0:
            plots[sector_name] = hv.Text(
                0, 0, f"All zeros ({sector_name})"
            ).opts(title=SECTOR_LABELS.get(sector_name, sector_name))
            continue

        gen = _color_by_generation(sv, spec.generation_boundaries)
        gen_pos = gen[pos_mask]
        n_gen = gen_pos.max() + 1

        overlays = []
        for g in range(n_gen):
            mask = gen_pos == g
            color = GENERATION_COLORS[g % len(GENERATION_COLORS)]
            df = pd.DataFrame({"index": idx_pos[mask], "sigma": sv_pos[mask]})
            pts = hv.Scatter(df, kdims=["index"], vdims=["sigma"]).opts(
                color=color, alpha=0.8, size=5, tools=["hover"],
            )
            overlays.append(pts)

        plot = hv.Overlay(overlays).opts(
            width=500, height=350,
            title=SECTOR_LABELS.get(sector_name, sector_name),
            xlabel="Index", ylabel="σ",
            logy=True,
        )
        plots[sector_name] = plot

    return plots


def build_walker_classification_plot(result: DiracSpectrumResult) -> hv.Element:
    """Scatter: x=||F_visc|| (log), y=fitness, color=sector."""
    cm = result.color_magnitude
    fit = result.walker_fitness
    sec = result.sector_assignment

    if len(cm) == 0:
        return hv.Text(0, 0, "No alive walkers").opts(title="Walker Classification")

    overlays = []
    for sector_idx, sector_name in SECTOR_NAMES.items():
        mask = sec == sector_idx
        if not mask.any():
            continue
        color = SECTOR_COLORS[sector_name]
        marker = "circle" if sector_idx in (0, 2) else "triangle"  # up-type vs down-type
        df = pd.DataFrame({
            "force_viscous_norm": cm[mask],
            "fitness": fit[mask],
        })
        pts = hv.Points(df, kdims=["force_viscous_norm", "fitness"]).opts(
            color=color, marker=marker, size=8, alpha=0.85,
            tools=["hover"],
        )
        overlays.append(pts)

    if not overlays:
        return hv.Text(0, 0, "No data").opts(title="Walker Classification")

    plot = hv.Overlay(overlays).opts(
        width=650, height=400,
        title=f"Walker Classification ({result.n_alive} alive, MC step {result.mc_time_index})",
        xlabel="‖F_visc‖",
        ylabel="Fitness",
        logx=True if cm.max() > 0 else False,
    )
    return plot


def build_mass_hierarchy_plot(result: DiracSpectrumResult) -> hv.Element:
    """Grouped bar chart: generation x sector, showing median sigma per cluster."""
    rows = []
    for sector_name, spec in result.sectors.items():
        if not spec.generation_masses:
            continue
        for g_idx, mass in enumerate(spec.generation_masses):
            rows.append({
                "sector": SECTOR_LABELS.get(sector_name, sector_name),
                "generation": f"Gen {g_idx + 1}",
                "mass": mass,
            })

    # Filter out zero/negative masses (invisible on log scale)
    rows = [r for r in rows if r["mass"] > 0]
    if not rows:
        return hv.Text(0, 0, "No generation masses").opts(title="Mass Hierarchy")

    df = pd.DataFrame(rows)
    bars = hv.Bars(df, kdims=["sector", "generation"], vdims=["mass"]).opts(
        width=650, height=400,
        title="Mass Hierarchy (median σ per generation per sector)",
        ylabel="Median σ",
        xrotation=45,
        color="generation",
        cmap="Category10",
        tools=["hover"],
        logy=True,
    )
    return bars


def build_chiral_density_plot(result: DiracSpectrumResult) -> hv.Element:
    """Histogram of near-zero singular values with rho(0) estimate."""
    sv = result.full_singular_values
    if len(sv) == 0:
        return hv.Text(0, 0, "No singular values").opts(title="Chiral Density")

    # Focus on the lower end of the spectrum
    sv_max = sv.max()
    if sv_max <= 0:
        return hv.Text(0, 0, "All singular values are zero").opts(title="Chiral Density")

    threshold = sv_max * 0.3
    near = sv[sv < threshold]
    if len(near) < 2:
        near = sv  # fallback to full spectrum

    hist = hv.Histogram(np.histogram(near, bins=40)).opts(
        width=500, height=350,
        title="Singular Value Density (near-zero region)",
        xlabel="σ", ylabel="Count",
        color="#b07aa1", alpha=0.8,
        tools=["hover"],
    )

    # Add condensate annotation
    annotation = hv.Text(
        float(np.median(near)) if len(near) > 0 else 0,
        0,
        f"⟨ψ̄ψ⟩ ≈ {result.chiral_condensate:.3f}\n"
        f"({result.near_zero_count} near-zero modes)",
    ).opts(text_font_size="9pt", text_align="left")

    return hist * annotation


def build_generation_ratios_plot(result: DiracSpectrumResult) -> hv.Element:
    """Bar chart of mass ratios between generations within each sector."""
    rows = []
    for sector_name, spec in result.sectors.items():
        masses = spec.generation_masses
        if len(masses) < 2:
            continue
        for i in range(len(masses) - 1):
            if masses[i + 1] > 0:
                ratio = masses[i] / masses[i + 1]
            else:
                ratio = 0.0
            rows.append({
                "sector": sector_name,
                "ratio_label": f"Gen{i + 1}/Gen{i + 2}",
                "ratio": ratio,
            })

    if not rows:
        return hv.Text(0, 0, "Need ≥ 2 generations for ratios").opts(
            title="Generation Ratios"
        )

    df = pd.DataFrame(rows)
    bars = hv.Bars(df, kdims=["sector", "ratio_label"], vdims=["ratio"]).opts(
        width=650, height=350,
        title="Inter-generation Mass Ratios (σ_gen_i / σ_gen_{{i+1}})",
        ylabel="Mass Ratio",
        xrotation=45,
        color="ratio_label",
        cmap="Set2",
        tools=["hover"],
        logy=True,
    )
    return bars


# ---------------------------------------------------------------------------
# Fermion comparison plots
# ---------------------------------------------------------------------------


def build_fermion_comparison_table(result: DiracSpectrumResult, refs=None) -> hv.Table:
    """Table showing extracted vs PDG masses with error %."""
    rows, scale = build_fermion_comparison(result, refs)
    if not rows:
        return hv.Text(0, 0, "No comparison data").opts(title="Fermion Comparison")
    df = pd.DataFrame(rows)
    cols = ["sector", "generation", "particle", "alg_mass", "obs_mass_GeV",
            "pred_mass_GeV", "error_pct"]
    df = df[[c for c in cols if c in df.columns]]
    return hv.Table(df).opts(width=750, height=350, title="Extracted vs PDG Fermion Masses")


def build_fermion_ratio_comparison_plot(result: DiracSpectrumResult, refs=None) -> hv.Element:
    """Grouped bar chart comparing measured vs observed inter-generation ratios."""
    rows = build_fermion_ratio_comparison(result, refs)
    if not rows:
        return hv.Text(0, 0, "Need ≥ 2 generations for ratio comparison").opts(
            title="Fermion Ratio Comparison"
        )
    df = pd.DataFrame(rows)
    # Melt measured/observed into a single "source" column for grouped bars
    df_melt = df.melt(
        id_vars=["sector", "ratio_label"],
        value_vars=["measured", "observed"],
        var_name="source",
        value_name="ratio",
    ).dropna(subset=["ratio"])
    if df_melt.empty:
        return hv.Text(0, 0, "No valid ratios").opts(title="Fermion Ratio Comparison")
    # Use log scale since ratios can span orders of magnitude
    bars = hv.Bars(
        df_melt, kdims=["sector", "source"], vdims=["ratio"],
    ).opts(
        width=650, height=400,
        title="Inter-Generation Mass Ratios: Measured vs Observed (PDG)",
        ylabel="Mass Ratio",
        xrotation=45,
        color="source",
        cmap={"measured": "#4c78a8", "observed": "#e45756"},
        logy=True,
    )
    return bars


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------


def build_all_dirac_plots(result: DiracSpectrumResult) -> dict[str, Any]:
    """Build all Dirac spectrum plots from a result.

    Returns:
        Dict mapping plot name to HoloViews element.
    """
    return {
        "full_spectrum": build_full_spectrum_plot(result),
        "sector_spectra": build_sector_spectra_plots(result),
        "walker_classification": build_walker_classification_plot(result),
        "mass_hierarchy": build_mass_hierarchy_plot(result),
        "chiral_density": build_chiral_density_plot(result),
        "generation_ratios": build_generation_ratios_plot(result),
        "fermion_comparison": build_fermion_comparison_table(result),
        "fermion_ratio_comparison": build_fermion_ratio_comparison_plot(result),
    }
