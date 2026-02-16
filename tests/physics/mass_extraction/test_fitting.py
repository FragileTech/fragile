"""Tests for the fitting engine."""

from __future__ import annotations

import gvar
import numpy as np

from fragile.physics.mass_extraction.config import (
    ChannelGroupConfig,
    FitConfig,
)
from fragile.physics.mass_extraction.fitting import run_fit
from fragile.physics.mass_extraction.models import build_all_models
from fragile.physics.mass_extraction.priors import build_combined_prior


def _make_fit_data():
    """Create gvar data and groups for fitting tests."""
    mass0, mass1 = 0.3, 0.8
    amp0, amp1 = 1.0, 0.3
    max_lag = 40
    t = np.arange(max_lag + 1, dtype=np.float64)
    signal = amp0**2 * np.exp(-mass0 * t) + amp1**2 * np.exp(-mass1 * t)
    errors = np.abs(signal) * 0.02
    data = {"scalar": gvar.gvar(signal, errors)}

    groups = [
        ChannelGroupConfig(
            name="pion",
            correlator_keys=["scalar"],
        ),
    ]
    keys_per_group = {"pion": ["scalar"]}
    max_lags = {"scalar": max_lag}
    return data, groups, keys_per_group, max_lags, mass0


def test_run_fit_converges():
    data, groups, keys_per_group, max_lags, true_mass = _make_fit_data()

    models = build_all_models(groups, list(data.keys()), max_lags)
    prior = build_combined_prior(groups, data, keys_per_group)
    fit = run_fit(data, models, prior)

    # Fit should converge
    assert fit.Q > 0.001  # reasonable fit quality

    # Extracted mass should be close to true value
    dE = fit.p["pion.dE"]
    extracted_mass = gvar.mean(dE[0])
    assert abs(extracted_mass - true_mass) < 0.1


def test_run_fit_with_svdcut():
    data, groups, keys_per_group, max_lags, _ = _make_fit_data()

    models = build_all_models(groups, list(data.keys()), max_lags)
    prior = build_combined_prior(groups, data, keys_per_group)
    config = FitConfig(svdcut=1e-2)
    fit = run_fit(data, models, prior, config)
    assert fit is not None
