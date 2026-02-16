"""Tests for corrfitter model construction."""

from __future__ import annotations

import corrfitter as cf

from fragile.physics.mass_extraction.config import ChannelFitConfig, ChannelGroupConfig
from fragile.physics.mass_extraction.models import (
    build_all_models,
    build_corr2_model,
    get_parameter_keys,
)


def test_build_corr2_model():
    model = build_corr2_model("scalar", "pion", ChannelFitConfig(), max_lag=40)
    assert isinstance(model, cf.Corr2)
    assert model.datatag == "scalar"


def test_build_corr2_model_with_tp():
    cfg = ChannelFitConfig(tp=80)
    model = build_corr2_model("scalar", "pion", cfg, max_lag=40)
    assert isinstance(model, cf.Corr2)


def test_build_all_models():
    groups = [
        ChannelGroupConfig(
            name="pion",
            correlator_keys=["scalar", "pseudoscalar"],
        ),
        ChannelGroupConfig(
            name="rho",
            correlator_keys=["vector"],
        ),
    ]
    available = ["scalar", "pseudoscalar", "vector"]
    max_lags = {"scalar": 40, "pseudoscalar": 40, "vector": 40}
    models = build_all_models(groups, available, max_lags)
    assert len(models) == 3


def test_build_all_models_missing_key():
    groups = [
        ChannelGroupConfig(
            name="pion",
            correlator_keys=["scalar", "missing_key"],
        ),
    ]
    models = build_all_models(groups, ["scalar"], {"scalar": 40})
    assert len(models) == 1


def test_get_parameter_keys():
    groups = [
        ChannelGroupConfig(
            name="pion",
            correlator_keys=["scalar", "pseudoscalar"],
        ),
    ]
    keys = get_parameter_keys(groups, ["scalar", "pseudoscalar"])
    assert "pion" in keys
    assert keys["pion"]["dE_key"] == "pion.dE"
    assert "scalar" in keys["pion"]["amplitude_keys"]
