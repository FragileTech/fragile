"""Tests for data preparation (torch -> gvar conversion)."""

from __future__ import annotations

import gvar
import numpy as np
import torch

from fragile.physics.mass_extraction.config import CovarianceConfig, MassExtractionConfig
from fragile.physics.mass_extraction.data_preparation import (
    correlator_tensor_to_numpy,
    correlators_to_gvar,
    multi_run_correlators_to_gvar,
    operator_series_to_correlator_samples,
)


def test_correlator_tensor_to_numpy():
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = correlator_tensor_to_numpy(t)
    assert arr.dtype == np.float64
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_correlator_tensor_to_numpy_2d():
    t = torch.randn(3, 5)
    arr = correlator_tensor_to_numpy(t)
    assert arr.shape == (3, 5)
    assert arr.dtype == np.float64


def test_correlators_to_gvar_uncorrelated(synthetic_correlators):
    cfg = MassExtractionConfig(covariance=CovarianceConfig(method="uncorrelated"))
    data = correlators_to_gvar(synthetic_correlators, config=cfg)
    assert "scalar" in data
    assert "pseudoscalar" in data
    assert "vector" in data
    # Check gvar output
    for key, gv_arr in data.items():
        assert len(gv_arr) > 0
        assert isinstance(gv_arr[0], gvar.GVar)


def test_correlators_to_gvar_multiscale(multiscale_correlators):
    cfg = MassExtractionConfig(covariance=CovarianceConfig(method="uncorrelated"))
    data = correlators_to_gvar(multiscale_correlators, config=cfg)
    # Should expand [3, L] into scalar_scale_0, scalar_scale_1, scalar_scale_2
    assert "scalar_scale_0" in data
    assert "scalar_scale_1" in data
    assert "scalar_scale_2" in data


def test_operator_series_to_correlator_samples_jackknife(synthetic_operator_series):
    samples = operator_series_to_correlator_samples(
        synthetic_operator_series,
        max_lag=20,
        method="block_jackknife",
        block_size=50,
    )
    assert samples.ndim == 2
    assert samples.shape[1] == 21  # max_lag + 1
    assert samples.shape[0] > 1


def test_operator_series_to_correlator_samples_bootstrap(synthetic_operator_series):
    samples = operator_series_to_correlator_samples(
        synthetic_operator_series,
        max_lag=20,
        method="bootstrap",
        n_bootstrap=50,
        block_size=50,
    )
    assert samples.shape == (50, 21)


def test_multi_run_correlators_to_gvar(synthetic_correlators):
    # Use the same correlators with slight noise as "multiple runs"
    run1 = synthetic_correlators
    run2 = {k: v + 0.001 * torch.randn_like(v) for k, v in run1.items()}
    data = multi_run_correlators_to_gvar([run1, run2])
    assert "scalar" in data
    assert isinstance(data["scalar"][0], gvar.GVar)


def test_multiscale_operator_scale_slicing():
    """Verify that [S,T] operator + _scale_N key → resampled covariance."""
    from fragile.physics.mass_extraction.data_preparation import _single_correlator_to_gvar

    n_scales = 3
    T = 500
    max_lag = 20

    # Create [S, T] operator series
    rng = np.random.default_rng(42)
    series_np = rng.normal(size=(n_scales, T))
    operators = {"scalar": torch.from_numpy(series_np).float()}

    # Create a 1D correlator (as if already expanded from [S,L])
    corr_np = np.abs(rng.normal(size=max_lag + 1)) + 0.1

    cov_config = CovarianceConfig(method="block_jackknife", block_size=25)

    # With _scale_1 key, should slice series[1] and use resampling
    result = _single_correlator_to_gvar("scalar_scale_1", corr_np, operators, cov_config)
    assert len(result) == max_lag + 1
    assert isinstance(result[0], gvar.GVar)

    # Errors should NOT be the 10% fallback
    for i in range(len(result)):
        sdev = gvar.sdev(result[i])
        fallback_err = abs(corr_np[i]) * 0.1 + 1e-15
        # The resampled error should differ from the naive 10% fallback
        assert abs(sdev - fallback_err) / max(fallback_err, 1e-15) > 0.01, (
            f"Error at lag {i} looks like 10% fallback — slicing may not have worked"
        )
