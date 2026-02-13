"""Multiscale estimator analysis helpers for dashboard integration.

This module centralizes the "scale-as-estimator" analysis used by the
Multiscale dashboard tab:

- extraction of per-scale mass measurements
- consensus mass estimation
- pairwise discrepancy diagnostics
- systematics verdict classification
- standardized table-row construction for UI rendering
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


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

    label: str
    alert_type: str
    details: str


@dataclass
class MultiscaleAnalysisBundle:
    """Full analysis payload for one channel across scales."""

    channel_name: str
    measurements: list[ScaleMeasurement]
    consensus: ConsensusResult
    discrepancies: list[PairwiseDiscrepancy]
    verdict: SystematicsVerdict


def _relative_error_pct(error: float, value: float) -> float:
    """Return |error/value| in percent, or NaN when undefined."""
    err = float(error)
    val = float(value)
    if not (math.isfinite(err) and err >= 0.0 and math.isfinite(val) and val != 0.0):
        return float("nan")
    return abs(err / val) * 100.0


def extract_scale_measurements(results: list[Any], scales: np.ndarray) -> list[ScaleMeasurement]:
    """Build one measurement per valid per-scale correlator result."""
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


def compute_consensus_mass(measurements: list[ScaleMeasurement]) -> ConsensusResult:
    """Compute an inverse-variance consensus from valid scale measurements."""
    valid = [m for m in measurements if math.isfinite(m.mass) and m.mass > 0]
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
        w_errors = np.maximum(errors[finite_weight_mask], 1e-12)
        weights = 1.0 / (w_errors**2)
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

    systematic_spread = float(np.std(masses, ddof=1)) if masses.size > 1 else 0.0
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
    """Compute all pairwise discrepancy diagnostics with vectorized numerics."""
    valid = [m for m in measurements if math.isfinite(m.mass) and m.mass > 0]
    n_valid = len(valid)
    if n_valid < 2:
        return []

    labels = np.asarray([m.label for m in valid], dtype=object)
    masses = np.asarray([m.mass for m in valid], dtype=float)
    errors = np.asarray([m.mass_error for m in valid], dtype=float)

    idx_a, idx_b = np.triu_indices(n_valid, k=1)
    mass_a = masses[idx_a]
    mass_b = masses[idx_b]
    label_a = labels[idx_a]
    label_b = labels[idx_b]

    ratio = np.full(mass_a.shape, np.nan, dtype=float)
    valid_ratio = mass_b > 0.0
    ratio[valid_ratio] = mass_a[valid_ratio] / mass_b[valid_ratio]
    delta_pct = np.full_like(ratio, np.nan, dtype=float)
    finite_ratio = np.isfinite(ratio)
    delta_pct[finite_ratio] = (ratio[finite_ratio] - 1.0) * 100.0
    abs_delta_pct = np.abs(delta_pct)

    err_a = np.where(np.isfinite(errors[idx_a]), np.maximum(errors[idx_a], 0.0), np.nan)
    err_b = np.where(np.isfinite(errors[idx_b]), np.maximum(errors[idx_b], 0.0), np.nan)
    combined_error = np.sqrt(err_a**2 + err_b**2)

    abs_diff = np.abs(mass_a - mass_b)
    pull_sigma = np.full_like(combined_error, np.nan, dtype=float)
    valid_pull = np.isfinite(combined_error) & (combined_error > 0.0)
    pull_sigma[valid_pull] = abs_diff[valid_pull] / combined_error[valid_pull]

    return [
        PairwiseDiscrepancy(
            label_a=str(label_a[k]),
            label_b=str(label_b[k]),
            mass_a=float(mass_a[k]),
            mass_b=float(mass_b[k]),
            ratio=float(ratio[k]),
            delta_pct=float(delta_pct[k]),
            abs_delta_pct=float(abs_delta_pct[k]),
            combined_error=float(combined_error[k]),
            pull_sigma=float(pull_sigma[k]),
        )
        for k in range(len(idx_a))
    ]


def evaluate_systematics_verdict(discrepancies: list[PairwiseDiscrepancy]) -> SystematicsVerdict:
    """Classify cross-scale agreement as consistent / mild tension / tension."""
    if not discrepancies:
        return SystematicsVerdict(
            label="insufficient data",
            alert_type="secondary",
            details="Need at least two scales with valid fits.",
        )

    abs_deltas = np.asarray([d.abs_delta_pct for d in discrepancies], dtype=float)
    pulls = np.asarray([d.pull_sigma for d in discrepancies], dtype=float)
    finite_abs = abs_deltas[np.isfinite(abs_deltas)]
    finite_pull = pulls[np.isfinite(pulls)]
    max_abs_delta = float(np.max(finite_abs)) if finite_abs.size > 0 else float("nan")
    max_pull = float(np.max(finite_pull)) if finite_pull.size > 0 else float("nan")

    details = (
        f"max |Δ%| = {max_abs_delta:.2f}%" if np.isfinite(max_abs_delta) else "max |Δ%| = n/a"
    )
    details += f", max pull = {max_pull:.2f}σ" if np.isfinite(max_pull) else ", max pull = n/a"

    if (
        np.isfinite(max_abs_delta)
        and max_abs_delta <= 5.0
        and (not np.isfinite(max_pull) or max_pull <= 1.5)
    ):
        return SystematicsVerdict(label="consistent", alert_type="success", details=details)
    if (
        np.isfinite(max_abs_delta)
        and max_abs_delta <= 15.0
        and (not np.isfinite(max_pull) or max_pull <= 3.0)
    ):
        return SystematicsVerdict(label="mild tension", alert_type="warning", details=details)
    return SystematicsVerdict(label="tension", alert_type="danger", details=details)


def analyze_channel_across_scales(
    results: list[Any],
    scales: np.ndarray,
    channel_name: str,
) -> MultiscaleAnalysisBundle:
    """Run the full multiscale analysis pipeline for a single channel."""
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


def build_estimator_table_rows(
    measurements: list[ScaleMeasurement],
    *,
    original_mass: float | None = None,
    original_error: float = float("nan"),
    original_r2: float = float("nan"),
    original_scale: float = float("nan"),
) -> list[dict[str, float | str]]:
    """Create standardized estimator-table rows for the Multiscale tab."""
    rows: list[dict[str, float | str]] = []
    if original_mass is not None and math.isfinite(original_mass) and original_mass > 0.0:
        rows.append({
            "scale": "original (no filter)",
            "scale_value": float(original_scale)
            if math.isfinite(float(original_scale))
            else float("nan"),
            "mass": float(original_mass),
            "mass_error": float(original_error),
            "mass_error_pct": _relative_error_pct(float(original_error), float(original_mass)),
            "r_squared": float(original_r2),
            "delta_vs_original_pct": 0.0,
        })

    for measurement in measurements:
        row: dict[str, float | str] = {
            "scale": measurement.label,
            "scale_value": float(measurement.scale),
            "mass": float(measurement.mass),
            "mass_error": float(measurement.mass_error),
            "mass_error_pct": _relative_error_pct(
                float(measurement.mass_error),
                float(measurement.mass),
            ),
            "r_squared": float(measurement.r_squared),
        }
        if original_mass is not None and math.isfinite(original_mass) and original_mass > 0.0:
            if math.isfinite(measurement.mass) and measurement.mass > 0.0:
                row["delta_vs_original_pct"] = (
                    (float(measurement.mass) - float(original_mass)) / float(original_mass) * 100.0
                )
            else:
                row["delta_vs_original_pct"] = float("nan")
        rows.append(row)
    return rows


def build_pairwise_table_rows(
    discrepancies: list[PairwiseDiscrepancy],
) -> list[dict[str, float | str]]:
    """Create standardized pairwise discrepancy-table rows for the Multiscale tab."""
    rows: list[dict[str, float | str]] = []
    for discrepancy in discrepancies:
        mass_diff = float(discrepancy.mass_a) - float(discrepancy.mass_b)
        rows.append({
            "scale_a": discrepancy.label_a,
            "scale_b": discrepancy.label_b,
            "mass_a": float(discrepancy.mass_a),
            "mass_b": float(discrepancy.mass_b),
            "ratio": float(discrepancy.ratio),
            "delta_pct": float(discrepancy.delta_pct),
            "abs_delta_pct": float(discrepancy.abs_delta_pct),
            "combined_error": float(discrepancy.combined_error),
            "combined_error_pct": _relative_error_pct(
                float(discrepancy.combined_error),
                mass_diff,
            ),
            "pull_sigma": float(discrepancy.pull_sigma),
        })
    return rows


def format_consensus_summary(
    consensus: ConsensusResult,
    discrepancies: list[PairwiseDiscrepancy],
    channel_name: str,
    *,
    reference_mass: float | None = None,
) -> str:
    """Format a markdown summary for multiscale consensus and discrepancies."""
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
                "- Relative systematic spread: "
                f"`{(consensus.systematic_spread / consensus.mass) * 100.0:.2f}%`"
            )
        if math.isfinite(consensus.chi2):
            red_chi2 = consensus.chi2 / max(consensus.ndof, 1)
            lines.append(
                f"- Scale consistency (χ²/ndof): `{consensus.chi2:.4g}/{consensus.ndof}` "
                f"= `{red_chi2:.4g}`"
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

    if reference_mass is not None and math.isfinite(reference_mass) and reference_mass > 0:
        lines.append(f"- Original (no scale filter): `{reference_mass:.6g}`")
        if math.isfinite(consensus.mass) and consensus.mass > 0 and reference_mass > 0:
            delta_pct = (consensus.mass - reference_mass) / reference_mass * 100.0
            lines.append(f"- Δ vs original: `{delta_pct:+.2f}%`")
    return "  \n".join(lines)
