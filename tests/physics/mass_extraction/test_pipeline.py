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

from .conftest import KNOWN_MASS_0, make_synthetic_correlator, MockPipelineResult


def test_auto_detect_channel_groups():
    keys = ["scalar", "pseudoscalar", "vector", "nucleon", "glueball_plaq"]
    groups = _auto_detect_channel_groups(keys)
    names = {g.name for g in groups}
    assert "scalar" in names
    assert "pseudoscalar" in names
    assert "vector" in names
    assert "nucleon" in names
    assert "glueball" in names


def test_auto_detect_multiscale():
    keys = ["scalar", "scalar_scale_0", "scalar_scale_1"]
    groups = _auto_detect_channel_groups(keys)
    # All should map to scalar
    scalar_groups = [g for g in groups if g.name == "scalar"]
    assert len(scalar_groups) == 1
    assert "scalar" in scalar_groups[0].correlator_keys


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
                name="scalar",
                correlator_keys=["scalar"],
            ),
        ],
        compute_effective_mass=True,
    )

    result = extract_masses(pr, config)

    assert "scalar" in result.channels
    extracted = gvar.mean(result.channels["scalar"].ground_state_mass)
    # Should recover ground state mass within ~30%
    assert (
        abs(extracted - mass0) / mass0 < 0.3
    ), f"Extracted mass {extracted:.4f} too far from true {mass0}"
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


# ---------------------------------------------------------------------------
# Mode-suffixed key tests
# ---------------------------------------------------------------------------


def test_auto_detect_mode_suffixed_keys():
    """Keys like scalar_standard, nucleon_flux_action → correct groups."""
    keys = [
        "scalar_standard",
        "scalar_flux_action",
        "nucleon_flux_action",
        "glueball_re_plaquette",
        "vector_standard",
    ]
    groups = _auto_detect_channel_groups(keys)
    by_name = {g.name: g for g in groups}

    assert "scalar" in by_name
    assert "scalar_standard" in by_name["scalar"].correlator_keys
    assert "scalar_flux_action" in by_name["scalar"].correlator_keys

    assert "nucleon" in by_name
    assert "nucleon_flux_action" in by_name["nucleon"].correlator_keys

    assert "glueball" in by_name
    assert "glueball_re_plaquette" in by_name["glueball"].correlator_keys

    assert "vector" in by_name
    assert "vector_standard" in by_name["vector"].correlator_keys


def test_auto_detect_propagator_variants():
    """Keys like scalar_standard_propagator → scalar, pseudoscalar → pseudoscalar."""
    keys = ["scalar_standard_propagator", "pseudoscalar_flux_action_propagator"]
    groups = _auto_detect_channel_groups(keys)
    by_name = {g.name: g for g in groups}
    assert "scalar" in by_name
    assert "scalar_standard_propagator" in by_name["scalar"].correlator_keys
    assert "pseudoscalar" in by_name
    assert "pseudoscalar_flux_action_propagator" in by_name["pseudoscalar"].correlator_keys


def test_auto_detect_include_multiscale_false():
    """Scale keys excluded when include_multiscale=False."""
    keys = ["scalar", "scalar_scale_0", "scalar_scale_1", "vector"]
    groups = _auto_detect_channel_groups(keys, include_multiscale=False)
    all_keys = []
    for g in groups:
        all_keys.extend(g.correlator_keys)
    assert "scalar_scale_0" not in all_keys
    assert "scalar_scale_1" not in all_keys
    assert "scalar" in all_keys
    assert "vector" in all_keys


def test_axial_vector_before_vector():
    """axial_vector_standard → axial_vector, vector_standard → vector."""
    keys = ["axial_vector_standard", "vector_standard"]
    groups = _auto_detect_channel_groups(keys)
    by_name = {g.name: g for g in groups}
    assert "axial_vector" in by_name
    assert "axial_vector_standard" in by_name["axial_vector"].correlator_keys
    assert "vector" in by_name
    assert "vector_standard" in by_name["vector"].correlator_keys


def test_auto_detect_propagator_prefixes():
    """Propagator-style prefixes map to the correct channel groups."""
    keys = [
        "meson_scalar_standard_propagator",
        "axial_full_standard_propagator",
        "baryon_nucleon_flux_action_propagator",
    ]
    groups = _auto_detect_channel_groups(keys)
    by_name = {g.name: g for g in groups}
    assert "scalar" in by_name
    assert "meson_scalar_standard_propagator" in by_name["scalar"].correlator_keys
    assert "axial_vector" in by_name
    assert "axial_full_standard_propagator" in by_name["axial_vector"].correlator_keys
    assert "nucleon" in by_name
    assert "baryon_nucleon_flux_action_propagator" in by_name["nucleon"].correlator_keys


def test_extract_masses_mode_suffixed():
    """End-to-end with mode-suffixed correlators."""
    pr = MockPipelineResult(
        correlators={
            "scalar_standard": make_synthetic_correlator(seed=1),
            "scalar_flux_action": make_synthetic_correlator(seed=2),
            "vector_standard": make_synthetic_correlator(
                mass0=0.5,
                mass1=1.0,
                seed=3,
            ),
        },
        operators={},
    )

    result = extract_masses(pr)
    assert "scalar" in result.channels
    assert "vector" in result.channels
    # scalar group should have 2 variant keys
    assert len(result.channels["scalar"].variant_keys) == 2
