"""Plotting functions for Einstein equation verification.

Builds HoloViews plots from :class:`EinsteinTestResult`.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
from scipy import stats

from fragile.fractalai.qft.einstein_equations import EinsteinTestResult


# ---------------------------------------------------------------------------
# Individual plot builders
# ---------------------------------------------------------------------------


def _ricci_source_label(result: EinsteinTestResult) -> str:
    if result.ricci_scalar_source == "riemannian_mix_proxy":
        return "RiemannianMix proxy"
    return "full derivative pipeline"


def _fmt_optional(value: float | None, fmt: str = ".4f", na: str = "n/a") -> str:
    if value is None:
        return na
    return format(float(value), fmt)


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
            "Volume: determinant approximation\n"
            f"Ricci source: {_ricci_source_label(result)}\n"
            f"R\u00b2 = {result.scalar_r2:.4f}\n"
            f"G_N(Einstein) = {result.g_newton_einstein:.4e}\n"
            f"\u039b = {result.lambda_measured:.4e}"
        ),
    ).opts(text_font_size="9pt", text_align="left")

    return (scatter * line * annotation).opts(
        width=650, height=400,
        title=f"Scalar Test (det approx): R vs \u03c1 (R\u00b2={result.scalar_r2:.4f})",
        xlabel="\u03c1 = 1/V", ylabel="R (Ricci scalar)",
    )


def build_scalar_test_log_plot(result: EinsteinTestResult) -> hv.Element:
    """Semi-log R vs rho scatter: log10(rho) on x-axis, linear R on y-axis."""
    valid = result.valid_mask & (result.volumes > 0)
    if valid.sum() < 3:
        return hv.Text(0, 0, "Insufficient valid data").opts(title="Scalar Test (semi-log)")

    rho = 1.0 / result.volumes[valid]
    R = result.ricci_scalar[valid]
    # rho must be positive for log; R can be any sign
    finite = np.isfinite(rho) & np.isfinite(R) & (rho > 0)
    if finite.sum() < 3:
        return hv.Text(0, 0, "Too few finite points").opts(title="Scalar Test (semi-log)")

    rho_f, R_f = rho[finite], R[finite]
    log_rho = np.log10(rho_f)

    # Fit in semi-log space: R = slope * log10(rho) + intercept
    slope, intercept, r_value, _p, _se = stats.linregress(log_rho, R_f)
    r2_log = r_value ** 2

    scatter = hv.Scatter(
        pd.DataFrame({"log10_rho": log_rho, "R": R_f}),
        kdims=["log10_rho"], vdims=["R"],
    ).opts(
        color="#4c78a8", alpha=0.6, size=4, tools=["hover"],
    )

    # Regression line in semi-log space
    log_rho_range = np.linspace(log_rho.min(), log_rho.max(), 50)
    R_fit = slope * log_rho_range + intercept
    line = hv.Curve(
        pd.DataFrame({"log10_rho": log_rho_range, "R": R_fit}),
        kdims=["log10_rho"], vdims=["R"],
    ).opts(color="#e45756", line_width=2)

    annotation = hv.Text(
        float(np.median(log_rho)),
        float(np.max(R_f) * 0.9) if len(R_f) > 0 else 0,
        (
            f"Semi-log fit: R = {slope:.3f} \u00b7 log\u2081\u2080(\u03c1) + {intercept:.3f}\n"
            f"R\u00b2 (semi-log) = {r2_log:.4f}\n"
            f"R\u00b2 (linear) = {result.scalar_r2:.4f}\n"
            f"N points = {finite.sum()}"
        ),
    ).opts(text_font_size="9pt", text_align="left")

    return (scatter * line * annotation).opts(
        width=650, height=400,
        title=f"Scalar Test (semi-log): R vs log\u2081\u2080(\u03c1) (R\u00b2={r2_log:.4f})",
        xlabel="log\u2081\u2080(\u03c1)", ylabel="R (Ricci scalar)",
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
        title=f"Ricci Scalar Distribution ({_ricci_source_label(result)}, N={len(R)} valid)",
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


def _symlog(x: np.ndarray) -> np.ndarray:
    """Symmetric log transform: sign(x) * log10(1 + |x|)."""
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def build_crosscheck_plot(result: EinsteinTestResult) -> hv.Element | None:
    """Full Ricci vs proxy scatter with symlog axes, colored by bulk/boundary."""
    if (
        result.ricci_proxy is None
        or result.ricci_scalar_full is None
        or result.proxy_vs_full_r2 is None
    ):
        return None

    valid = result.valid_mask & np.isfinite(result.ricci_proxy)
    if valid.sum() < 3:
        return hv.Text(0, 0, "Insufficient cross-check data")

    proxy = result.ricci_proxy[valid]
    full = result.ricci_scalar_full[valid]
    is_bulk = result.bulk_mask[valid]
    finite = np.isfinite(proxy) & np.isfinite(full)
    if finite.sum() < 3:
        return None

    proxy_f, full_f = proxy[finite], full[finite]
    is_bulk_f = is_bulk[finite]

    # Symlog transform
    proxy_sl = _symlog(proxy_f)
    full_sl = _symlog(full_f)

    # R² on symlog-transformed values
    ss_res = np.sum((full_sl - proxy_sl) ** 2)
    ss_tot = np.sum((full_sl - np.mean(full_sl)) ** 2)
    r2_symlog = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # R² in symlog space via linear regression
    sl_slope, sl_intercept, sl_r, _p, _se = stats.linregress(proxy_sl, full_sl)
    r2_symlog_fit = sl_r ** 2

    # Build dataframe with region labels
    df = pd.DataFrame({
        "proxy_symlog": proxy_sl,
        "full_symlog": full_sl,
        "region": np.where(is_bulk_f, "bulk", "boundary"),
    })

    scatter_bulk = hv.Scatter(
        df[df["region"] == "bulk"],
        kdims=["proxy_symlog"], vdims=["full_symlog"],
        label="Bulk",
    ).opts(color="#4c78a8", alpha=0.5, size=4, tools=["hover"])

    scatter_boundary = hv.Scatter(
        df[df["region"] == "boundary"],
        kdims=["proxy_symlog"], vdims=["full_symlog"],
        label="Boundary",
    ).opts(color="#e45756", alpha=0.5, size=4, tools=["hover"])

    # 1:1 line
    vmin = min(proxy_sl.min(), full_sl.min())
    vmax = max(proxy_sl.max(), full_sl.max())
    diag = hv.Curve([(vmin, vmin), (vmax, vmax)]).opts(
        color="gray", line_dash="dashed", line_width=1,
    )

    # Regression line in symlog space
    fit_x = np.linspace(proxy_sl.min(), proxy_sl.max(), 50)
    fit_y = sl_slope * fit_x + sl_intercept
    reg_line = hv.Curve(
        pd.DataFrame({"proxy_symlog": fit_x, "full_symlog": fit_y}),
        kdims=["proxy_symlog"], vdims=["full_symlog"],
    ).opts(color="#54a24b", line_width=2, line_dash="solid")

    annotation = hv.Text(
        float(np.percentile(proxy_sl, 10)),
        float(np.percentile(full_sl, 95)),
        (
            f"R\u00b2 (raw) = {result.proxy_vs_full_r2:.4f}\n"
            f"R\u00b2 (symlog, 1:1) = {r2_symlog:.4f}\n"
            f"R\u00b2 (symlog, fit) = {r2_symlog_fit:.4f}\n"
            f"fit slope = {sl_slope:.3f}"
        ),
    ).opts(text_font_size="9pt", text_align="left")

    return (scatter_bulk * scatter_boundary * diag * reg_line * annotation).opts(
        width=600, height=450,
        title=f"Full Ricci vs Proxy \u2014 symlog (R\u00b2={r2_symlog_fit:.4f})",
        xlabel="symlog(Ricci Proxy)", ylabel="symlog(Full Ricci Scalar)",
        legend_position="bottom_right",
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
        "### Scalar Test (Determinant Approximation)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| R\u00b2 (all) | {result.scalar_r2:.4f} |",
        f"| R\u00b2 (bulk) | {result.bulk_scalar_r2:.4f} |",
        f"| R\u00b2 (boundary) | {result.boundary_scalar_r2:.4f} |",
        f"| Ricci source | {_ricci_source_label(result)} |",
        f"| slope | {result.scalar_slope:.6e} |",
        f"| G_N (Einstein) | {result.g_newton_einstein:.6e} |",
        f"| Lambda | {result.lambda_measured:.6e} |",
        "",
    ]

    if result.scalar_r2_full_volume is not None:
        lines.extend([
            "### Scalar Test (Full Volume Element)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| R\u00b2 (all) | {_fmt_optional(result.scalar_r2_full_volume, '.4f')} |",
            f"| R\u00b2 (bulk) | {_fmt_optional(result.bulk_scalar_r2_full_volume, '.4f')} |",
            f"| R\u00b2 (boundary) | {_fmt_optional(result.boundary_scalar_r2_full_volume, '.4f')} |",
            f"| slope | {_fmt_optional(result.scalar_slope_full_volume, '.6e')} |",
            f"| G_N (Einstein) | {_fmt_optional(result.g_newton_einstein_full_volume, '.6e')} |",
            f"| Lambda | {_fmt_optional(result.lambda_measured_full_volume, '.6e')} |",
            "",
            "### Volume Comparison",
            "",
            f"| Metric | Det approx | Full volume |",
            f"|--------|------------|-------------|",
            f"| R\u00b2 | {_fmt_optional(result.scalar_r2, '.4f')} | {_fmt_optional(result.scalar_r2_full_volume, '.4f')} |",
            (
                f"| G_N | {_fmt_optional(result.g_newton_einstein, '.6e')} | "
                f"{_fmt_optional(result.g_newton_einstein_full_volume, '.6e')} |"
            ),
            (
                f"| Lambda | {_fmt_optional(result.lambda_measured, '.6e')} | "
                f"{_fmt_optional(result.lambda_measured_full_volume, '.6e')} |"
            ),
            "",
        ])
    else:
        lines.extend([
            "### Scalar Test (Full Volume Element)",
            "",
            "_Unavailable: run with `neighbor_graph_record=True` so Voronoi dual volumes are recorded._",
            "",
        ])

    lines.extend([
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
    ])

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
    text = (
        f"**Bulk** (inner {result.config.bulk_fraction:.0%}): "
        f"R\u00b2 = {result.bulk_scalar_r2:.4f} "
        f"({int(result.bulk_mask.sum())} walkers)\n\n"
        f"**Boundary** (outer {1 - result.config.bulk_fraction:.0%}): "
        f"R\u00b2 = {result.boundary_scalar_r2:.4f} "
        f"({int((~result.bulk_mask).sum())} walkers)"
    )
    if result.scalar_r2_full_volume is not None:
        text += (
            "\n\n"
            "**Full volume element (theory-aligned):**\n\n"
            f"Bulk R\u00b2 = {_fmt_optional(result.bulk_scalar_r2_full_volume, '.4f')}  \n"
            f"Boundary R\u00b2 = {_fmt_optional(result.boundary_scalar_r2_full_volume, '.4f')}"
        )
    return text


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
        "scalar_test_log": build_scalar_test_log_plot(result),
        "tensor_r2": build_tensor_r2_table(result),
        "curvature_dist": build_curvature_histogram(result),
        "residual_map": build_residual_scatter(result),
        "crosscheck": crosscheck,
        "bulk_boundary": build_bulk_boundary_markdown(result),
    }
