"""Tests for vectorized multiscale analysis helpers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from fragile.fractalai.qft.multiscale_analysis import (
    analyze_channel_across_scales,
    build_estimator_table_rows,
    build_pairwise_table_rows,
)


def _result(mass: float, mass_error: float, *, r_squared: float = 0.9, n_samples: int = 16):
    return SimpleNamespace(
        n_samples=n_samples,
        mass_fit={
            "mass": mass,
            "mass_error": mass_error,
            "r_squared": r_squared,
        },
    )


def test_analyze_channel_across_scales_computes_expected_consensus_and_discrepancies() -> None:
    """Consensus and pairwise diagnostics should be numerically consistent."""
    scales = np.asarray([0.1, 0.2, 0.4], dtype=float)
    results = [
        _result(1.0, 0.1, r_squared=0.95),
        _result(1.2, 0.2, r_squared=0.85),
        _result(float("nan"), float("nan"), n_samples=0),
    ]

    bundle = analyze_channel_across_scales(results, scales, "nucleon")

    assert len(bundle.measurements) == 2
    assert len(bundle.discrepancies) == 1

    expected_mass = (100.0 * 1.0 + 25.0 * 1.2) / 125.0
    expected_stat_error = math.sqrt(1.0 / 125.0)
    assert math.isclose(bundle.consensus.mass, expected_mass, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(
        bundle.consensus.stat_error, expected_stat_error, rel_tol=1e-12, abs_tol=1e-12
    )

    discrepancy = bundle.discrepancies[0]
    assert math.isclose(discrepancy.combined_error, math.sqrt(0.1**2 + 0.2**2), rel_tol=1e-12)
    assert math.isclose(
        discrepancy.pull_sigma,
        abs(1.0 - 1.2) / math.sqrt(0.1**2 + 0.2**2),
        rel_tol=1e-12,
    )
    assert bundle.verdict.label == "tension"


def test_table_row_builders_include_relative_error_columns() -> None:
    """Estimator and pairwise rows should expose percent-relative error columns."""
    scales = np.asarray([0.1, 0.2], dtype=float)
    results = [_result(1.0, 0.1), _result(1.2, 0.2)]
    bundle = analyze_channel_across_scales(results, scales, "glueball")

    estimator_rows = build_estimator_table_rows(
        bundle.measurements,
        original_mass=1.1,
        original_error=0.11,
        original_r2=0.91,
        original_scale=0.9,
    )
    assert len(estimator_rows) == 3
    assert math.isclose(float(estimator_rows[0]["mass_error_pct"]), 10.0, rel_tol=1e-12)
    for row in estimator_rows[1:]:
        assert "mass_error_pct" in row
        assert float(row["mass_error_pct"]) >= 0.0

    pairwise_rows = build_pairwise_table_rows(bundle.discrepancies)
    assert len(pairwise_rows) == 1
    row = pairwise_rows[0]
    assert "combined_error_pct" in row
    expected_combined_error_pct = abs(math.sqrt(0.1**2 + 0.2**2) / (1.0 - 1.2)) * 100.0
    assert math.isclose(
        float(row["combined_error_pct"]),
        expected_combined_error_pct,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
