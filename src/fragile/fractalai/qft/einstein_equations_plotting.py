"""Plotting functions for Einstein equation verification.

Builds HoloViews plots from :class:`EinsteinTestResult`.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd

from fragile.fractalai.qft.einstein_equations import EinsteinTestResult


# ---------------------------------------------------------------------------
# Individual plot builders
# ---------------------------------------------------------------------------


def build_scalar_test_plot(result: EinsteinTestResult) -> hv.Element:
    """R vs rho scatter + regression line + R2/G_N annotation."""
    valid = result.valid_mask & (result.volumes > 0)
    if valid.sum() < 3:
        return hv.Text(0, 0, "Insufficient valid data").opts(title="Scalar Test")

    rho = 1.0 / result.volumes[valid]
    R = result.ricci_scalar[valid]
    finite = np.isfinite(rho) & np.isfinite(R)
    if finite.sum() < 3:
        return hv.Text(0, 0, "No finite data").opts(title="Scalar Test")

    rho_f, R_f = rho[finite], R[finite]

    scatter = hv.Scatter(
        pd.DataFrame({"rho": rho_f, "R": R_f}),
        kdims=["rho"], vdims=["R"],
    ).opts(
        color="#4c78a8", alpha=0.6, size=4, tools=["hover"],
    )

    # Regression line
    rho_range = np.linspace(rho_f.min(), rho_f.max(), 50)
    R_fit = result.scalar_slope * rho_range + result.scalar_intercept
    line = hv.Curve(
        pd.DataFrame({"rho": rho_range, "R": R_fit}),
        kdims=["rho"], vdims=["R"],
    ).opts(color="#e45756", line_width=2)

    annotation = hv.Text(
        float(np.median(rho_f)),
        float(np.max(R_f) * 0.9) if len(R_f) > 0 else 0,
        (
            f"R\u00b2 = {result.scalar_r2:.4f}\n"
            f"G_N(Einstein) = {result.g_newton_einstein:.4e}\n"
            f"\u039b = {result.lambda_measured:.4e}"
        ),
    ).opts(text_font_size="9pt", text_align="left")

    return (scatter * line * annotation).opts(
        width=650, height=400,
        title=f"Scalar Test: R vs \u03c1 (R\u00b2={result.scalar_r2:.4f})",
        xlabel="\u03c1 = 1/V", ylabel="R (Ricci scalar)",
    )


def build_tensor_r2_table(result: EinsteinTestResult) -> hv.Table:
    """Per-component R2 values table."""
    rows = []
    for label, r2 in zip(result.component_labels, result.tensor_r2_per_component):
        rows.append({"component": label, "R2": float(r2)})
    rows.append({"component": "overall", "R2": float(result.tensor_r2)})

    if not rows:
        return hv.Text(0, 0, "No tensor test data").opts(title="Tensor R\u00b2")

    df = pd.DataFrame(rows)
    return hv.Table(df).opts(width=400, height=250, title="Tensor Component R\u00b2")


def build_curvature_histogram(result: EinsteinTestResult) -> hv.Element:
    """Distribution of Ricci scalar R."""
    R = result.ricci_scalar[result.valid_mask]
    R = R[np.isfinite(R)]
    if len(R) < 2:
        return hv.Text(0, 0, "No valid curvature data").opts(title="Curvature Distribution")

    # Clip outliers for better visualization
    p1, p99 = np.percentile(R, [1, 99])
    R_clip = R[(R >= p1) & (R <= p99)]
    if len(R_clip) < 2:
        R_clip = R

    hist = hv.Histogram(np.histogram(R_clip, bins=40)).opts(
        width=500, height=350,
        title=f"Ricci Scalar Distribution (N={len(R)} valid)",
        xlabel="R (Ricci scalar)", ylabel="Count",
        color="#72b7b2", alpha=0.8,
        tools=["hover"],
    )
    return hist


def build_residual_scatter(result: EinsteinTestResult) -> hv.Element:
    """Spatial map colored by ||G - 8piG_N*T||_F."""
    valid = result.valid_mask
    if valid.sum() < 2 or result.spatial_dim < 2:
        return hv.Text(0, 0, "Insufficient data for residual map").opts(
            title="Residual Map"
        )

    G = result.einstein_tensor[valid]
    T = result.stress_energy_tensor[valid]
    g_n = result.g_newton_area_law

    residual = G - 8.0 * np.pi * g_n * T
    residual_norm = np.sqrt(np.sum(residual ** 2, axis=(1, 2)))

    pos = result.positions[valid]
    df = pd.DataFrame({
        "x": pos[:, 0],
        "y": pos[:, 1],
        "residual": residual_norm,
    })

    scatter = hv.Points(df, kdims=["x", "y"], vdims=["residual"]).opts(
        color="residual", cmap="inferno", colorbar=True,
        size=6, alpha=0.8, tools=["hover"],
        width=550, height=450,
        title=f"Residual ||G - 8\u03c0G_N T||_F (G_N={g_n:.3e})",
        xlabel="x\u2080", ylabel="x\u2081",
    )
    return scatter


def build_crosscheck_plot(result: EinsteinTestResult) -> hv.Element | None:
    """Full Ricci vs proxy scatter."""
    if result.ricci_proxy is None or result.proxy_vs_full_r2 is None:
        return None

    valid = result.valid_mask & np.isfinite(result.ricci_proxy)
    if valid.sum() < 3:
        return hv.Text(0, 0, "Insufficient cross-check data")

    proxy = result.ricci_proxy[valid]
    full = result.ricci_scalar[valid]
    finite = np.isfinite(proxy) & np.isfinite(full)
    if finite.sum() < 3:
        return None

    proxy_f, full_f = proxy[finite], full[finite]

    scatter = hv.Scatter(
        pd.DataFrame({"proxy": proxy_f, "full": full_f}),
        kdims=["proxy"], vdims=["full"],
    ).opts(color="#f58518", alpha=0.5, size=4, tools=["hover"])

    # 1:1 line
    vmin = min(proxy_f.min(), full_f.min())
    vmax = max(proxy_f.max(), full_f.max())
    diag = hv.Curve([(vmin, vmin), (vmax, vmax)]).opts(
        color="gray", line_dash="dashed", line_width=1,
    )

    return (scatter * diag).opts(
        width=550, height=400,
        title=f"Full Ricci vs Proxy (R\u00b2={result.proxy_vs_full_r2:.4f})",
        xlabel="Ricci Proxy (conformal)", ylabel="Full Ricci Scalar",
    )


def build_summary_markdown(result: EinsteinTestResult) -> str:
    """Formatted interpretation summary."""
    lines = [
        "## Einstein Test Summary",
        "",
        f"| Quantity | Value |",
        f"|----------|-------|",
        f"| N walkers | {result.n_walkers} |",
        f"| Spatial dim | {result.spatial_dim} |",
        f"| MC frame | {result.mc_frame} |",
        f"| Valid walkers | {result.valid_mask.sum()} |",
        f"| Bulk walkers | {result.bulk_mask.sum()} |",
        "",
        "### Scalar Test (R vs rho)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| R\u00b2 (all) | {result.scalar_r2:.4f} |",
        f"| R\u00b2 (bulk) | {result.bulk_scalar_r2:.4f} |",
        f"| R\u00b2 (boundary) | {result.boundary_scalar_r2:.4f} |",
        f"| slope | {result.scalar_slope:.6e} |",
        f"| G_N (Einstein) | {result.g_newton_einstein:.6e} |",
        f"| Lambda | {result.lambda_measured:.6e} |",
        "",
        "### Tensor Test (G_uv vs T_uv)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| R\u00b2 (overall) | {result.tensor_r2:.4f} |",
        f"| slope (8piG_N) | {result.tensor_slope:.6e} |",
        "",
        "### G_N Cross-Reference",
        "",
        f"| Source | Value |",
        f"|--------|-------|",
        f"| G_N (area law) | {result.g_newton_area_law:.6e} ({result.g_newton_source}) |",
        f"| G_N (Einstein) | {result.g_newton_einstein:.6e} |",
        f"| Ratio (Einstein/area) | {result.g_newton_ratio:.4f} |",
    ]

    if result.proxy_vs_full_r2 is not None:
        lines.extend([
            "",
            "### Ricci Proxy Cross-Check",
            "",
            f"| R\u00b2 (proxy vs full) | {result.proxy_vs_full_r2:.4f} |",
        ])

    return "\n".join(lines)


def build_bulk_boundary_markdown(result: EinsteinTestResult) -> str:
    """Bulk vs boundary comparison."""
    return (
        f"**Bulk** (inner {result.config.bulk_fraction:.0%}): "
        f"R\u00b2 = {result.bulk_scalar_r2:.4f} "
        f"({int(result.bulk_mask.sum())} walkers)\n\n"
        f"**Boundary** (outer {1 - result.config.bulk_fraction:.0%}): "
        f"R\u00b2 = {result.boundary_scalar_r2:.4f} "
        f"({int((~result.bulk_mask).sum())} walkers)"
    )


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------


def build_all_einstein_plots(result: EinsteinTestResult) -> dict[str, Any]:
    """Build all Einstein test plots from a result.

    Returns:
        Dict mapping plot name to HoloViews element or markdown string.
    """
    crosscheck = build_crosscheck_plot(result)

    return {
        "summary": build_summary_markdown(result),
        "scalar_test": build_scalar_test_plot(result),
        "tensor_r2": build_tensor_r2_table(result),
        "curvature_dist": build_curvature_histogram(result),
        "residual_map": build_residual_scatter(result),
        "crosscheck": crosscheck,
        "bulk_boundary": build_bulk_boundary_markdown(result),
    }
