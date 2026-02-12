"""Tests for multiscale operator-quality diagnostics in the channel dashboard."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from fragile.fractalai.qft.dashboard.channel_dashboard import (
    _analyze_operator_quality_vectorized,
    _apply_gevp_entry_filters,
    _build_eigenspectrum_plot,
    _collect_family_operator_entries,
)


def _result(
    series: list[float],
    *,
    mass: float,
    mass_error: float,
    r2: float = 0.9,
    n_valid_windows: int = 6,
    n_samples: int = 12,
):
    return SimpleNamespace(
        series=torch.tensor(series, dtype=torch.float32),
        n_samples=n_samples,
        mass_fit={
            "mass": mass,
            "mass_error": mass_error,
            "r_squared": r2,
            "n_valid_windows": n_valid_windows,
        },
    )


def test_collect_family_operator_entries_includes_multiscale_and_original() -> None:
    """Family collection should aggregate multiscale scales plus original channels."""
    output = SimpleNamespace(
        per_scale_results={
            "nucleon": [
                _result([1.0, 0.8, 0.65, 0.53, 0.43], mass=0.55, mass_error=0.06),
                _result([1.0, 0.76, 0.59, 0.47, 0.38], mass=0.57, mass_error=0.07),
            ],
            "nucleon_flux_exp": [
                _result([1.0, 0.82, 0.66, 0.54, 0.44], mass=0.58, mass_error=0.08),
            ],
            "scalar": [
                _result([1.0, 0.9, 0.85, 0.8, 0.76], mass=0.21, mass_error=0.03),
            ],
        }
    )
    original_results = {
        "nucleon": _result([1.0, 0.79, 0.63, 0.51, 0.41], mass=0.56, mass_error=0.06),
        "nucleon_multiscale_best": _result(
            [1.0, 0.79, 0.63, 0.51, 0.41], mass=0.56, mass_error=0.06
        ),
        "scalar": _result([1.0, 0.9, 0.85, 0.8, 0.76], mass=0.21, mass_error=0.03),
    }

    entries = _collect_family_operator_entries(output, "nucleon", original_results=original_results)

    assert len(entries) == 4
    assert all("nucleon" in str(entry["source_channel"]) for entry in entries)
    assert all(int(entry["series"].numel()) > 0 for entry in entries)


def test_analyze_operator_quality_vectorized_builds_rows_and_spectrum() -> None:
    """Vectorized analysis should produce ranked rows and an eigenspectrum plot."""
    entries = [
        {
            "operator_label": "nucleon@s0",
            "source_channel": "nucleon",
            "scale_label": "s0",
            "series": torch.tensor([1.0, 0.83, 0.68, 0.56, 0.46, 0.38], dtype=torch.float32),
            "mass": 0.55,
            "mass_error": 0.05,
            "r2": 0.95,
            "n_samples": 20,
            "source_kind": "multiscale",
        },
        {
            "operator_label": "nucleon_flux_exp@s2",
            "source_channel": "nucleon_flux_exp",
            "scale_label": "s2",
            "series": torch.tensor([1.0, 0.81, 0.66, 0.54, 0.45, 0.37], dtype=torch.float32),
            "mass": 0.57,
            "mass_error": 0.06,
            "r2": 0.91,
            "n_samples": 18,
            "source_kind": "multiscale",
        },
        {
            "operator_label": "nucleon_noise@full",
            "source_channel": "nucleon_noise",
            "scale_label": "full_original_no_threshold",
            "series": torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01, -0.01], dtype=torch.float32),
            "mass": float("nan"),
            "mass_error": float("nan"),
            "r2": float("nan"),
            "n_samples": 8,
            "source_kind": "original",
        },
    ]

    analysis = _analyze_operator_quality_vectorized(
        entries,
        t0=1,
        eig_rel_cutoff=1e-2,
        importance_cutoff=0.05,
        use_connected=True,
    )

    assert analysis is not None
    assert int(analysis["n_operators"]) == 3
    assert 1 <= int(analysis["n_significant"]) <= 3
    assert len(list(analysis["rows"])) == 3

    rows_by_name = {row["operator"]: row for row in analysis["rows"]}
    assert rows_by_name["nucleon_noise@full"]["suggestion"] == "low_signal_candidate"

    plot = _build_eigenspectrum_plot(analysis, family_label="nucleon")
    assert plot is not None


def test_apply_gevp_entry_filters_matches_gevp_quality_criteria() -> None:
    """Table filters should exclude entries that fail active GEVP thresholds."""
    entries = [
        {
            "operator_label": "keep",
            "mass": 0.58,
            "r2": 0.92,
            "n_valid_windows": 5,
            "mass_error": 0.10,
        },
        {
            "operator_label": "low_r2",
            "mass": 0.61,
            "r2": 0.2,
            "n_valid_windows": 6,
            "mass_error": 0.09,
        },
        {
            "operator_label": "low_windows",
            "mass": 0.63,
            "r2": 0.95,
            "n_valid_windows": 1,
            "mass_error": 0.11,
        },
        {
            "operator_label": "zero_error",
            "mass": 0.67,
            "r2": 0.94,
            "n_valid_windows": 8,
            "mass_error": 0.0,
        },
        {
            "operator_label": "nan_error",
            "mass": 0.66,
            "r2": 0.94,
            "n_valid_windows": 8,
            "mass_error": float("nan"),
        },
        {
            "operator_label": "inf_error",
            "mass": 0.65,
            "r2": 0.94,
            "n_valid_windows": 8,
            "mass_error": float("inf"),
        },
        {
            "operator_label": "zero_mass",
            "mass": 0.0,
            "r2": 0.94,
            "n_valid_windows": 8,
            "mass_error": 0.2,
        },
        {
            "operator_label": "high_err_pct",
            "mass": 0.5,
            "r2": 0.98,
            "n_valid_windows": 8,
            "mass_error": 0.25,
        },
    ]

    kept, excluded = _apply_gevp_entry_filters(
        entries,
        min_r2=0.8,
        min_windows=3,
        max_error_pct=30.0,
        remove_artifacts=True,
    )

    assert [entry["operator_label"] for entry in kept] == ["keep"]
    excluded_by_name = dict(excluded)
    assert "r2=" in excluded_by_name["low_r2"]
    assert "n_windows=1<3" in excluded_by_name["low_windows"]
    assert "mass_error==0" in excluded_by_name["zero_error"]
    assert "mass_error=nan_or_inf" in excluded_by_name["nan_error"]
    assert "mass_error=nan_or_inf" in excluded_by_name["inf_error"]
    assert "mass==0" in excluded_by_name["zero_mass"]
    assert "err_pct=50>30" in excluded_by_name["high_err_pct"]
