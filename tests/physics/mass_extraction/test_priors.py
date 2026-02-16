"""Tests for prior construction."""

from __future__ import annotations

import gvar
import numpy as np

from fragile.physics.mass_extraction.config import ChannelGroupConfig
from fragile.physics.mass_extraction.priors import (
    build_combined_prior,
    build_prior_for_group,
)


def _make_gvar_data():
    """Create simple gvar correlator data for testing priors."""
    t = np.arange(41)
    means = np.exp(-0.3 * t)
    errors = np.abs(means) * 0.05
    return {"scalar": gvar.gvar(means, errors)}


def test_build_prior_for_group():
    data = _make_gvar_data()
    group = ChannelGroupConfig(
        name="pion",
        correlator_keys=["scalar"],
    )
    prior = build_prior_for_group(group, data, ["scalar"])
    assert "log(pion.dE)" in prior
    assert "pion.scalar.a" in prior
    assert "pion.scalar.b" in prior
    assert len(prior["log(pion.dE)"]) == 2  # nexp default is 2
    assert len(prior["pion.scalar.a"]) == 2


def test_build_prior_no_log():
    data = _make_gvar_data()
    group = ChannelGroupConfig(
        name="pion",
        correlator_keys=["scalar"],
    )
    group.fit.use_log_dE = False
    prior = build_prior_for_group(group, data, ["scalar"])
    assert "pion.dE" in prior


def test_build_combined_prior():
    data = _make_gvar_data()
    groups = [
        ChannelGroupConfig(name="pion", correlator_keys=["scalar"]),
    ]
    keys_per_group = {"pion": ["scalar"]}
    prior = build_combined_prior(groups, data, keys_per_group)
    assert "log(pion.dE)" in prior
    assert "pion.scalar.a" in prior


def test_fastfit_seeding():
    """When fastfit seeding is enabled, priors should still work even if fastfit fails."""
    data = _make_gvar_data()
    group = ChannelGroupConfig(
        name="pion",
        correlator_keys=["scalar"],
    )
    group.prior.use_fastfit_seeding = True
    prior = build_prior_for_group(group, data, ["scalar"])
    # Should succeed regardless of whether fastfit worked
    assert "log(pion.dE)" in prior
