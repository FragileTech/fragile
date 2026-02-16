"""Tests for mass extraction configuration dataclasses."""

from __future__ import annotations

from fragile.physics.mass_extraction.config import (
    ChannelFitConfig,
    ChannelGroupConfig,
    CovarianceConfig,
    FitConfig,
    MassExtractionConfig,
    PriorConfig,
)


def test_default_config():
    """MassExtractionConfig() works with all defaults."""
    cfg = MassExtractionConfig()
    assert cfg.channel_groups == []
    assert cfg.compute_effective_mass is True
    assert cfg.fit.svdcut == 1e-4


def test_covariance_config_defaults():
    cfg = CovarianceConfig()
    assert cfg.method == "uncorrelated"
    assert cfg.n_bootstrap == 200


def test_channel_fit_config_defaults():
    cfg = ChannelFitConfig()
    assert cfg.tmin == 2
    assert cfg.nexp == 2
    assert cfg.use_log_dE is True


def test_prior_config_defaults():
    cfg = PriorConfig()
    assert cfg.dE_ground == "0.5(5)"
    assert cfg.use_fastfit_seeding is True


def test_fit_config_defaults():
    cfg = FitConfig()
    assert cfg.maxit == 2000
    assert cfg.use_chained is False


def test_channel_group_config():
    cfg = ChannelGroupConfig(
        name="pion",
        correlator_keys=["scalar", "pseudoscalar"],
        channel_type="meson",
    )
    assert cfg.name == "pion"
    assert len(cfg.correlator_keys) == 2
