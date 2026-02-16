"""Tests for the top-level mass extraction pipeline."""

from __future__ import annotations

import gvar
import numpy as np
import torch

from fragile.physics.mass_extraction.config import (
    ChannelGroupConfig,
    MassExtractionConfig,
)
from fragile.physics.mass_extraction.pipeline import (
    _auto_detect_channel_groups,
    extract_masses,
)

from .conftest import KNOWN_MASS_0, MockPipelineResult, make_synthetic_correlator


def test_auto_detect_channel_groups():
    keys = ["scalar", "pseudoscalar", "vector", "nucleon", "glueball_plaq"]
    groups = _auto_detect_channel_groups(keys)
    names = {g.name for g in groups}
    assert "pion" in names
    assert "rho" in names
    assert "nucleon" in names
    assert "glueball" in names


def test_auto_detect_multiscale():
    keys = ["scalar", "scalar_scale_0", "scalar_scale_1"]
    groups = _auto_detect_channel_groups(keys)
    # All should map to pion
    pion_groups = [g for g in groups if g.name == "pion"]
    assert len(pion_groups) == 1
    assert "scalar" in pion_groups[0].correlator_keys


def test_extract_masses_end_to_end():
    """End-to-end: synthetic data -> extract_masses -> verify recovered mass."""
    # Create clean synthetic data with known mass
    mass0 = KNOWN_MASS_0
    corr = make_synthetic_correlator(mass0=mass0, mass1=0.8, noise_level=1e-5)

    pr = MockPipelineResult(
        correlators={"scalar": corr},
        operators={},
    )

    config = MassExtractionConfig(
        channel_groups=[
            ChannelGroupConfig(
                name="pion",
                correlator_keys=["scalar"],
            ),
        ],
        compute_effective_mass=True,
    )

    result = extract_masses(pr, config)

    assert "pion" in result.channels
    extracted = gvar.mean(result.channels["pion"].ground_state_mass)
    # Should recover ground state mass within ~30%
    assert abs(extracted - mass0) / mass0 < 0.3, (
        f"Extracted mass {extracted:.4f} too far from true {mass0}"
    )
    assert result.diagnostics.chi2_per_dof > 0


def test_extract_masses_auto_detect():
    """Pipeline with auto-detected channel groups."""
    pr = MockPipelineResult(
        correlators={
            "scalar": make_synthetic_correlator(seed=1),
            "pseudoscalar": make_synthetic_correlator(seed=2),
        },
        operators={},
    )

    result = extract_masses(pr)
    assert len(result.channels) > 0


def test_extract_masses_empty():
    """Empty pipeline result should return empty results."""
    pr = MockPipelineResult(correlators={}, operators={})
    result = extract_masses(pr)
    assert len(result.channels) == 0
