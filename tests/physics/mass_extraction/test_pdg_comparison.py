"""Tests for PDG comparison and ratio analysis helpers."""

from __future__ import annotations

import gvar
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from fragile.physics.app.mass_extraction_tab import (
    PDG_REFERENCES,
    _build_pdg_comparison_plot,
    _build_ratio_comparison,
)
from fragile.physics.mass_extraction.results import (
    ChannelMassResult,
    FitDiagnostics,
    MassExtractionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(name: str, mass_mean: float, mass_err: float, channel_type: str = "meson"):
    return ChannelMassResult(
        name=name,
        channel_type=channel_type,
        ground_state_mass=gvar.gvar(mass_mean, mass_err),
        variant_keys=[f"{name}_ps"],
    )


def _make_result(channels: dict[str, ChannelMassResult]) -> MassExtractionResult:
    return MassExtractionResult(
        channels=channels,
        diagnostics=FitDiagnostics(chi2=10.0, dof=8, Q=0.5, logGBF=1.0, nit=50, svdcut=1e-4),
    )


# ---------------------------------------------------------------------------
# PDG_REFERENCES
# ---------------------------------------------------------------------------


def test_pdg_references_complete():
    """All 7 expected channel groups are present with positive masses."""
    expected = {"scalar", "pseudoscalar", "vector", "axial_vector", "nucleon", "glueball", "tensor"}
    assert set(PDG_REFERENCES.keys()) == expected
    for key, (label, mass) in PDG_REFERENCES.items():
        assert mass > 0, f"{key} has non-positive PDG mass"
        assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# _build_pdg_comparison_plot
# ---------------------------------------------------------------------------


def test_build_pdg_comparison_plot_smoke():
    """Smoke test: returns an hv overlay when given valid channels."""
    channels = {
        "pseudoscalar": _make_channel("pseudoscalar", 0.07, 0.005),
        "vector": _make_channel("vector", 0.40, 0.02),
        "nucleon": _make_channel("nucleon", 0.50, 0.03, channel_type="baryon"),
    }
    result = _make_result(channels)
    plot = _build_pdg_comparison_plot(result, "vector")
    assert isinstance(plot, (hv.Overlay, hv.NdOverlay))


def test_missing_anchor_returns_placeholder():
    """When the anchor channel is absent, return a placeholder."""
    channels = {
        "pseudoscalar": _make_channel("pseudoscalar", 0.07, 0.005),
    }
    result = _make_result(channels)
    plot = _build_pdg_comparison_plot(result, "nucleon")
    # Placeholder is an hv.Text
    assert isinstance(plot, hv.Text)


def test_nonpositive_anchor_returns_placeholder():
    """Non-positive anchor mass → placeholder."""
    channels = {
        "vector": _make_channel("vector", -0.1, 0.02),
    }
    result = _make_result(channels)
    plot = _build_pdg_comparison_plot(result, "vector")
    assert isinstance(plot, hv.Text)


# ---------------------------------------------------------------------------
# _build_ratio_comparison
# ---------------------------------------------------------------------------


def test_build_ratio_comparison_smoke():
    """3 channels → 3 unique pairs in the DataFrame."""
    channels = {
        "pseudoscalar": _make_channel("pseudoscalar", 0.07, 0.005),
        "vector": _make_channel("vector", 0.40, 0.02),
        "nucleon": _make_channel("nucleon", 0.50, 0.03, channel_type="baryon"),
    }
    result = _make_result(channels)
    overlay, df = _build_ratio_comparison(result)
    assert isinstance(overlay, (hv.Overlay, hv.NdOverlay))
    assert len(df) == 3  # C(3,2) = 3
    assert set(df.columns) == {"Ratio", "Extracted", "Error", "PDG", "Tension (\u03c3)"}


def test_ratio_values_match_gvar():
    """Extracted ratio values should match gvar.mean(m_A / m_B)."""
    m_pseudoscalar = gvar.gvar(0.07, 0.005)
    m_vector = gvar.gvar(0.40, 0.02)
    channels = {
        "pseudoscalar": ChannelMassResult(
            name="pseudoscalar", channel_type="meson",
            ground_state_mass=m_pseudoscalar, variant_keys=["pseudoscalar_ps"],
        ),
        "vector": ChannelMassResult(
            name="vector", channel_type="meson",
            ground_state_mass=m_vector, variant_keys=["vector_v"],
        ),
    }
    result = _make_result(channels)
    _, df = _build_ratio_comparison(result)
    assert len(df) == 1
    row = df.iloc[0]

    expected_ratio = gvar.mean(m_pseudoscalar / m_vector)
    assert abs(row["Extracted"] - expected_ratio) < 1e-6


def test_single_channel_no_ratios():
    """With only 1 channel, no ratios can be formed → placeholder + empty df."""
    channels = {
        "pseudoscalar": _make_channel("pseudoscalar", 0.07, 0.005),
    }
    result = _make_result(channels)
    overlay, df = _build_ratio_comparison(result)
    # Should be a placeholder (hv.Text) since <2 channels
    assert isinstance(overlay, hv.Text)
    assert len(df) == 0


def test_channels_without_pdg_refs_excluded():
    """Channels not in PDG_REFERENCES are silently skipped."""
    channels = {
        "pseudoscalar": _make_channel("pseudoscalar", 0.07, 0.005),
        "vector": _make_channel("vector", 0.40, 0.02),
        "exotic_X": _make_channel("exotic_X", 1.5, 0.1),
    }
    result = _make_result(channels)
    _, df = _build_ratio_comparison(result)
    # Only pseudoscalar and vector match PDG → 1 pair
    assert len(df) == 1
    assert "exotic_X" not in df["Ratio"].values[0]
