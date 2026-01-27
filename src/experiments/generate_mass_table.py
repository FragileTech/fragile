"""
Generate an anchored mass table from particle observables.

Usage:
    python src/experiments/generate_mass_table.py
    python src/experiments/generate_mass_table.py --metrics-path outputs/..._metrics.json
    python src/experiments/generate_mass_table.py --output-path outputs/mass_table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BARYON_REFS = {
    "proton": 0.938272,
    "neutron": 0.939565,
    "delta": 1.232,
    "lambda": 1.115683,
    "sigma0": 1.192642,
    "xi0": 1.31486,
    "omega-": 1.67245,
}

MESON_REFS = {
    "pion": 0.13957,
    "kaon": 0.493677,
    "eta": 0.547862,
    "rho": 0.77526,
    "omega": 0.78265,
    "phi": 1.01946,
    "jpsi": 3.0969,
    "upsilon": 9.4603,
}


def _find_latest_metrics(search_dir: Path) -> Path:
    candidates = list(search_dir.glob("*_metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No metrics files found in {search_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_metrics(metrics_path: Path) -> tuple[dict[str, float], dict | None]:
    metrics = json.loads(metrics_path.read_text())
    particle = metrics.get("particle_observables") or {}
    operators = particle.get("operators") or {}
    masses: dict[str, float] = {}
    for name in ("baryon", "meson"):
        fit = operators.get(name, {}).get("fit")
        if not fit or "mass" not in fit:
            raise ValueError(f"Missing {name} fit mass in {metrics_path}")
        masses[name] = float(fit["mass"])
    glue_fit = operators.get("glueball", {}).get("fit")
    if glue_fit and "mass" in glue_fit:
        masses["glueball"] = float(glue_fit["mass"])

    string_tension = metrics.get("string_tension")
    if isinstance(string_tension, dict):
        sigma = string_tension.get("sigma")
        if sigma is not None and sigma > 0:
            masses["sqrt_sigma"] = float(sigma) ** 0.5
    else:
        string_tension = None
    return masses, string_tension


def _closest_reference(value: float, refs: dict[str, float]) -> tuple[str, float, float]:
    name, ref = min(refs.items(), key=lambda kv: abs(value - kv[1]))
    err = (value - ref) / ref * 100.0
    return name, ref, err


def _fmt(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _fmt_scale(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _best_fit_scale(
    masses: dict[str, float], anchors: list[tuple[str, float, str]]
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for _label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            continue
        numerator += alg_mass * mass_phys
        denominator += alg_mass**2
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_scale_summary(scale: float | None, masses: dict[str, float]) -> str:
    if scale is None:
        return "n/a"
    pred_b = masses["baryon"] * scale
    pred_m = masses["meson"] * scale
    b_name, b_val, b_err = _closest_reference(pred_b, BARYON_REFS)
    m_name, m_val, m_err = _closest_reference(pred_m, MESON_REFS)
    return (
        f"{_fmt_scale(scale)} -> baryon {_fmt(pred_b)} "
        f"({b_name} {b_val:.3f}, {b_err:+.1f}%), "
        f"meson {_fmt(pred_m)} ({m_name} {m_val:.3f}, {m_err:+.1f}%)"
    )


def build_table(
    masses: dict[str, float],
    fit_mode: str = "none",
    glueball_ref: tuple[str, float] | None = None,
    sqrt_sigma_ref: float | None = None,
) -> str:
    anchors: list[tuple[str, float, str]] = []
    anchors.extend((f"baryon->{name}", mass, "baryon") for name, mass in BARYON_REFS.items())
    anchors.extend((f"meson->{name}", mass, "meson") for name, mass in MESON_REFS.items())
    if glueball_ref is not None and masses.get("glueball", 0.0) > 0:
        label, mass = glueball_ref
        anchors.append((f"glueball->{label}", mass, "glueball"))
    if sqrt_sigma_ref is not None and masses.get("sqrt_sigma", 0.0) > 0:
        anchors.append((f"sqrt_sigma->{sqrt_sigma_ref:.3f}", sqrt_sigma_ref, "sqrt_sigma"))

    glueball_refs: dict[str, float] = {}
    if glueball_ref is not None:
        label, mass = glueball_ref
        glueball_refs[label] = mass

    lines: list[str] = []
    if fit_mode != "none":
        baryon_anchors = [a for a in anchors if a[2] == "baryon"]
        meson_anchors = [a for a in anchors if a[2] == "meson"]
        combined_anchors = baryon_anchors + meson_anchors
        if fit_mode in {"baryon", "both"}:
            scale_b = _best_fit_scale(masses, baryon_anchors)
            lines.append(f"best-fit scale (baryon refs): {_format_scale_summary(scale_b, masses)}")
        if fit_mode in {"meson", "both"}:
            scale_m = _best_fit_scale(masses, meson_anchors)
            lines.append(f"best-fit scale (meson refs): {_format_scale_summary(scale_m, masses)}")
        if fit_mode == "both":
            scale_all = _best_fit_scale(masses, combined_anchors)
            lines.append(
                f"best-fit scale (baryon+meson refs): {_format_scale_summary(scale_all, masses)}"
            )
        lines.append("")

    include_glueball = masses.get("glueball", 0.0) > 0
    rows = []
    for label, mass_phys, family in anchors:
        alg_mass = masses.get(family)
        if alg_mass is None or alg_mass <= 0:
            if include_glueball:
                rows.append((label, None, None, "n/a", None, "n/a", None, "n/a"))
            else:
                rows.append((label, None, None, "n/a", None, "n/a"))
            continue

        scale = mass_phys / alg_mass
        pred_b = masses["baryon"] * scale
        pred_m = masses["meson"] * scale
        pred_g = None
        g_cell = "n/a"
        if include_glueball:
            pred_g = masses["glueball"] * scale
            if pred_g > 0 and glueball_refs:
                g_name, g_val, g_err = _closest_reference(pred_g, glueball_refs)
                g_cell = f"{g_name} {g_val:.3f} ({g_err:+.1f}%)"

        b_cell = "n/a"
        m_cell = "n/a"
        if pred_b > 0:
            b_name, b_val, b_err = _closest_reference(pred_b, BARYON_REFS)
            b_cell = f"{b_name} {b_val:.3f} ({b_err:+.1f}%)"
        if pred_m > 0:
            m_name, m_val, m_err = _closest_reference(pred_m, MESON_REFS)
            m_cell = f"{m_name} {m_val:.3f} ({m_err:+.1f}%)"

        if include_glueball:
            rows.append((
                label,
                scale,
                pred_b if pred_b > 0 else None,
                b_cell,
                pred_m if pred_m > 0 else None,
                m_cell,
                pred_g if pred_g and pred_g > 0 else None,
                g_cell,
            ))
        else:
            rows.append((
                label,
                scale,
                pred_b if pred_b > 0 else None,
                b_cell,
                pred_m if pred_m > 0 else None,
                m_cell,
            ))

    header = (
        "Anchor | scale (GeV/alg) | baryon pred (GeV) | closest baryon | meson pred (GeV) | "
        "closest meson"
    )
    if include_glueball:
        header += " | glueball pred (GeV) | glueball ref"
    separator = "-" * len(header)
    lines.extend([header, separator])
    for row in rows:
        if include_glueball:
            label, scale, pred_b, b_cell, pred_m, m_cell, pred_g, g_cell = row
            lines.append(
                f"{label:<16} | {_fmt_scale(scale):>14} | {_fmt(pred_b):>15} | "
                f"{b_cell:<20} | {_fmt(pred_m):>14} | {m_cell:<20} | "
                f"{_fmt(pred_g):>17} | {g_cell}"
            )
        else:
            label, scale, pred_b, b_cell, pred_m, m_cell = row
            lines.append(
                f"{label:<16} | {_fmt_scale(scale):>14} | {_fmt(pred_b):>15} | "
                f"{b_cell:<20} | {_fmt(pred_m):>14} | {m_cell}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anchored mass table.")
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Path to a *_metrics.json file (defaults to newest in outputs/qft_calibrated_analysis).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the table.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("none", "baryon", "meson", "both"),
        default="none",
        help="Add best-fit scale summary using baryon, meson, or combined references.",
    )
    parser.add_argument(
        "--glueball-ref",
        type=float,
        default=None,
        help="Optional physical glueball mass in GeV for anchoring/prediction.",
    )
    parser.add_argument(
        "--glueball-label",
        type=str,
        default="glueball",
        help="Label to use for glueball reference.",
    )
    parser.add_argument(
        "--sqrt-sigma-ref",
        type=float,
        default=None,
        help="Optional physical sqrt(sigma) in GeV for string-tension anchoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metrics_path is None:
        metrics_path = _find_latest_metrics(Path("outputs/qft_calibrated_analysis"))
    else:
        metrics_path = Path(args.metrics_path)

    masses, string_tension = _load_metrics(metrics_path)
    glueball_ref = None
    if args.glueball_ref is not None:
        glueball_ref = (args.glueball_label, args.glueball_ref)
    table = build_table(
        masses,
        fit_mode=args.fit_mode,
        glueball_ref=glueball_ref,
        sqrt_sigma_ref=args.sqrt_sigma_ref,
    )
    print(f"metrics: {metrics_path}")
    alg_parts = [f"baryon={masses['baryon']:.6f}", f"meson={masses['meson']:.6f}"]
    if masses.get("glueball") is not None:
        alg_parts.append(f"glueball={masses['glueball']:.6f}")
    if masses.get("sqrt_sigma") is not None:
        alg_parts.append(f"sqrt_sigma={masses['sqrt_sigma']:.6f}")
    print(f"algorithmic masses: {', '.join(alg_parts)}")
    if isinstance(string_tension, dict) and string_tension.get("sigma"):
        sigma = float(string_tension["sigma"])
        sqrt_sigma = sigma**0.5
        r2 = string_tension.get("r_squared")
        n_used = string_tension.get("n_used")
        r2_str = f"{r2:.3f}" if isinstance(r2, float | int) else "n/a"
        n_used_str = f"{n_used}" if n_used is not None else "n/a"
        print(
            "string tension: "
            f"sigma={sigma:.6f}, sqrt_sigma={sqrt_sigma:.6f}, r2={r2_str}, n_used={n_used_str}"
        )
    print(table)

    if args.output_path:
        Path(args.output_path).write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
