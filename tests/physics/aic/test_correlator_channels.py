"""AIC parity tests for correlator_channels module.

Verifies that the AIC copy (fragile.physics.aic.correlator_channels) produces
identical outputs to the original (fragile.fractalai.qft.correlator_channels).
Since the AIC files are verbatim copies that still import from the original
path, all outputs must be bit-for-bit identical.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.fractalai.qft.correlator_channels import (
    bootstrap_correlator_error,
    CHANNEL_REGISTRY,
    compute_channel_correlator,
    compute_correlator_fft,
    compute_effective_mass_torch,
    ConvolutionalAICExtractor,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
    get_channel_class,
)
from fragile.physics.aic.correlator_channels import (
    bootstrap_correlator_error as new_bootstrap,
    CHANNEL_REGISTRY as NEW_CHANNEL_REGISTRY,
    compute_channel_correlator as new_compute_channel,
    compute_correlator_fft as new_fft,
    compute_effective_mass_torch as new_eff_mass,
    ConvolutionalAICExtractor as NewAICExtractor,
    CorrelatorConfig as NewCorrelatorConfig,
    extract_mass_aic as new_extract_aic,
    extract_mass_linear as new_extract_linear,
    get_channel_class as new_get_channel_class,
)
from tests.physics.aic.conftest import (
    assert_mass_fit_equal,
    assert_outputs_equal,
    assert_tensor_or_nan_equal,
)


def _assert_channel_result_equal(old_out, new_out) -> None:
    """NaN-tolerant comparison of ChannelCorrelatorResult dataclasses.

    Like assert_outputs_equal but uses assert_tensor_or_nan_equal for tensor
    fields (effective_mass, window_masses, etc. may contain NaN).
    """
    import dataclasses

    old_fields = {f.name for f in dataclasses.fields(old_out)}
    new_fields = {f.name for f in dataclasses.fields(new_out)}
    assert old_fields == new_fields, f"Field mismatch: {old_fields ^ new_fields}"

    for f in dataclasses.fields(old_out):
        old_val = getattr(old_out, f.name)
        new_val = getattr(new_out, f.name)
        if old_val is None and new_val is None:
            continue
        if isinstance(old_val, Tensor):
            assert isinstance(
                new_val, Tensor
            ), f"Field {f.name}: old is Tensor, new is {type(new_val)}"
            assert_tensor_or_nan_equal(old_val, new_val, label=f"Field {f.name}")
        elif isinstance(old_val, dict) and isinstance(new_val, dict):
            assert_mass_fit_equal(old_val, new_val, label=f"Field {f.name}")
        else:
            assert old_val == new_val, f"Field {f.name}: {old_val!r} != {new_val!r}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEEDS = [42, 99, 2025]
ALL_CHANNELS = [
    "scalar",
    "pseudoscalar",
    "vector",
    "axial_vector",
    "tensor",
    "nucleon",
    "glueball",
]


def _make_series_1d(seed: int, T: int = 200) -> Tensor:
    """Create a 1-D operator time series [T]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(T, generator=gen)


def _make_series_2d(seed: int, B: int = 8, T: int = 200) -> Tensor:
    """Create a batched series [B, T]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, generator=gen)


def _make_exponential_correlator(
    seed: int, length: int = 81, mass: float = 0.3, dt: float = 1.0
) -> Tensor:
    """Create a synthetic exponential correlator C(t) = A * exp(-m * t) + noise."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    amplitude = 2.0 + torch.rand(1, generator=gen).item()
    correlator = amplitude * torch.exp(-mass * dt * t)
    noise = torch.randn(length, generator=gen) * 0.001
    return correlator + noise.abs()  # keep positive


def _make_log_corr_err(seed: int, T: int = 100) -> tuple[Tensor, Tensor]:
    """Create synthetic log-correlator and log-error tensors."""
    gen = torch.Generator().manual_seed(seed)
    # Synthetic decaying log-correlator: log(A) - m*t + noise
    t = torch.arange(T, dtype=torch.float32)
    mass = 0.2 + 0.1 * torch.rand(1, generator=gen).item()
    log_A = 1.0 + torch.rand(1, generator=gen).item()
    log_corr = log_A - mass * t + torch.randn(T, generator=gen) * 0.02
    log_err = 0.05 + 0.05 * torch.rand(T, generator=gen)
    return log_corr, log_err


# ===========================================================================
# Test classes
# ===========================================================================


class TestParityFFTCorrelator:
    """Parity tests for compute_correlator_fft."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_1d_series(self, seed: int) -> None:
        series = _make_series_1d(seed)
        max_lag = 40
        old = compute_correlator_fft(series, max_lag, use_connected=True)
        new = new_fft(series, max_lag, use_connected=True)
        assert torch.equal(old, new), f"seed={seed}: 1D FFT outputs differ"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_1d_disconnected(self, seed: int) -> None:
        series = _make_series_1d(seed)
        max_lag = 40
        old = compute_correlator_fft(series, max_lag, use_connected=False)
        new = new_fft(series, max_lag, use_connected=False)
        assert torch.equal(old, new), f"seed={seed}: 1D disconnected FFT outputs differ"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_2d_series(self, seed: int) -> None:
        """Test that the underlying batched FFT path gives identical results.

        compute_correlator_fft takes [T], so we manually call the internal
        _fft_correlator_batched via the public function on each row to verify
        consistency. Here we just verify the 1D path is identical for multiple
        independent series.
        """
        series_batch = _make_series_2d(seed)
        max_lag = 30
        for i in range(series_batch.shape[0]):
            row = series_batch[i]
            old = compute_correlator_fft(row, max_lag)
            new = new_fft(row, max_lag)
            assert torch.equal(old, new), f"seed={seed}, row={i}: batched FFT outputs differ"


class TestParityEffectiveMass:
    """Parity tests for compute_effective_mass_torch."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_effective_mass(self, seed: int) -> None:
        correlator = _make_exponential_correlator(seed)
        dt = 1.0
        old = compute_effective_mass_torch(correlator, dt)
        new = new_eff_mass(correlator, dt)
        assert_tensor_or_nan_equal(old, new, label=f"effective_mass seed={seed}")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_effective_mass_dt05(self, seed: int) -> None:
        correlator = _make_exponential_correlator(seed, mass=0.5, dt=0.5)
        dt = 0.5
        old = compute_effective_mass_torch(correlator, dt)
        new = new_eff_mass(correlator, dt)
        assert_tensor_or_nan_equal(old, new, label=f"effective_mass dt=0.5 seed={seed}")

    def test_effective_mass_empty(self) -> None:
        correlator = torch.tensor([])
        old = compute_effective_mass_torch(correlator, 1.0)
        new = new_eff_mass(correlator, 1.0)
        assert torch.equal(old, new)

    def test_effective_mass_single(self) -> None:
        correlator = torch.tensor([1.0])
        old = compute_effective_mass_torch(correlator, 1.0)
        new = new_eff_mass(correlator, 1.0)
        assert torch.equal(old, new)


class TestParityBootstrap:
    """Parity tests for bootstrap_correlator_error.

    Uses torch.manual_seed before each call to ensure identical random state.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bootstrap_error(self, seed: int) -> None:
        series = _make_series_1d(seed, T=150)
        max_lag = 30
        n_bootstrap = 50

        torch.manual_seed(seed)
        old = bootstrap_correlator_error(series, max_lag, n_bootstrap=n_bootstrap)

        torch.manual_seed(seed)
        new = new_bootstrap(series, max_lag, n_bootstrap=n_bootstrap)

        assert_tensor_or_nan_equal(old, new, label=f"bootstrap seed={seed}")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bootstrap_disconnected(self, seed: int) -> None:
        series = _make_series_1d(seed, T=150)
        max_lag = 30
        n_bootstrap = 50

        torch.manual_seed(seed)
        old = bootstrap_correlator_error(
            series, max_lag, n_bootstrap=n_bootstrap, use_connected=False
        )

        torch.manual_seed(seed)
        new = new_bootstrap(series, max_lag, n_bootstrap=n_bootstrap, use_connected=False)

        assert_tensor_or_nan_equal(old, new, label=f"bootstrap disconnected seed={seed}")

    def test_bootstrap_empty_series(self) -> None:
        series = torch.tensor([])
        old = bootstrap_correlator_error(series, 10, n_bootstrap=5)
        new = new_bootstrap(series, 10, n_bootstrap=5)
        assert torch.equal(old, new)


class TestParityAICExtractor:
    """Parity tests for ConvolutionalAICExtractor."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_single_width(self, seed: int) -> None:
        log_corr, log_err = _make_log_corr_err(seed, T=80)
        # Reshape to [1, 1, T] as required by fit_single_width
        lc = log_corr.view(1, 1, -1)
        le = log_err.view(1, 1, -1)
        W = 10

        old_ext = ConvolutionalAICExtractor()
        new_ext = NewAICExtractor()

        old_mass, old_aic, old_r2 = old_ext.fit_single_width(lc, le, W)
        new_mass, new_aic, new_r2 = new_ext.fit_single_width(lc, le, W)

        assert torch.equal(old_mass, new_mass), f"seed={seed}: mass tensors differ"
        assert torch.equal(old_aic, new_aic), f"seed={seed}: AIC tensors differ"
        assert_tensor_or_nan_equal(old_r2, new_r2, label=f"fit_single_width r2 seed={seed}")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_single_width_various_W(self, seed: int) -> None:
        log_corr, log_err = _make_log_corr_err(seed, T=80)
        lc = log_corr.view(1, 1, -1)
        le = log_err.view(1, 1, -1)

        old_ext = ConvolutionalAICExtractor()
        new_ext = NewAICExtractor()

        for W in [5, 15, 25]:
            old_mass, old_aic, old_r2 = old_ext.fit_single_width(lc, le, W)
            new_mass, new_aic, new_r2 = new_ext.fit_single_width(lc, le, W)

            assert torch.equal(old_mass, new_mass), f"seed={seed}, W={W}: mass differs"
            assert torch.equal(old_aic, new_aic), f"seed={seed}, W={W}: AIC differs"
            assert_tensor_or_nan_equal(old_r2, new_r2, label=f"fit_single_width W={W} seed={seed}")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_all_widths(self, seed: int) -> None:
        log_corr, log_err = _make_log_corr_err(seed, T=100)
        widths = [5, 10, 15, 20]

        old_ext = ConvolutionalAICExtractor(window_widths=widths)
        new_ext = NewAICExtractor(window_widths=widths)

        old_result = old_ext.fit_all_widths(log_corr, log_err)
        new_result = new_ext.fit_all_widths(log_corr, log_err)

        # Compare scalar fields
        assert old_result["mass"] == new_result["mass"], f"seed={seed}: mass differs"
        assert (
            old_result["mass_error"] == new_result["mass_error"]
        ), f"seed={seed}: mass_error differs"
        assert (
            old_result["n_valid_windows"] == new_result["n_valid_windows"]
        ), f"seed={seed}: n_valid_windows differs"
        assert (
            old_result["window_widths"] == new_result["window_widths"]
        ), f"seed={seed}: window_widths differs"

        # Compare tensor fields
        for key in ["window_masses", "window_aic", "window_r2"]:
            old_t = old_result.get(key)
            new_t = new_result.get(key)
            if old_t is None and new_t is None:
                continue
            assert old_t is not None and new_t is not None, f"seed={seed}: {key} None mismatch"
            assert_tensor_or_nan_equal(old_t, new_t, label=f"fit_all_widths {key} seed={seed}")

        # Compare best_window dict if present
        if "best_window" in old_result and "best_window" in new_result:
            old_bw = old_result["best_window"]
            new_bw = new_result["best_window"]
            for k in old_bw:
                assert old_bw[k] == new_bw[k], f"seed={seed}: best_window[{k}] differs"

    def test_fit_all_widths_degenerate(self) -> None:
        """Series too short for any window width."""
        log_corr = torch.randn(3)
        log_err = torch.ones(3) * 0.1
        widths = [5, 10]

        old_ext = ConvolutionalAICExtractor(window_widths=widths)
        new_ext = NewAICExtractor(window_widths=widths)

        old_result = old_ext.fit_all_widths(log_corr, log_err)
        new_result = new_ext.fit_all_widths(log_corr, log_err)

        assert old_result["mass"] == new_result["mass"]
        assert old_result["n_valid_windows"] == new_result["n_valid_windows"]
        assert old_result["window_widths"] == new_result["window_widths"]


class TestParityMassExtraction:
    """Parity tests for extract_mass_aic and extract_mass_linear."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extract_mass_aic(self, seed: int) -> None:
        correlator = _make_exponential_correlator(seed, length=81)
        dt = 1.0
        config = CorrelatorConfig(window_widths=[5, 10, 15, 20])
        new_config = NewCorrelatorConfig(window_widths=[5, 10, 15, 20])

        old = extract_mass_aic(correlator, dt, config)
        new = new_extract_aic(correlator, dt, new_config)

        assert_mass_fit_equal(old, new, label=f"extract_mass_aic seed={seed}")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extract_mass_linear(self, seed: int) -> None:
        correlator = _make_exponential_correlator(seed, length=81)
        dt = 1.0
        config = CorrelatorConfig(fit_mode="linear", fit_start=2, fit_stop=40)
        new_config = NewCorrelatorConfig(fit_mode="linear", fit_start=2, fit_stop=40)

        old = extract_mass_linear(correlator, dt, config)
        new = new_extract_linear(correlator, dt, new_config)

        assert_mass_fit_equal(old, new, label=f"extract_mass_linear seed={seed}")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_extract_mass_linear_abs(self, seed: int) -> None:
        correlator = _make_exponential_correlator(seed, length=81)
        dt = 1.0
        config = CorrelatorConfig(fit_mode="linear_abs", fit_start=2, fit_stop=40)
        new_config = NewCorrelatorConfig(fit_mode="linear_abs", fit_start=2, fit_stop=40)

        old = extract_mass_linear(correlator.abs(), dt, config)
        new = new_extract_linear(correlator.abs(), dt, new_config)

        assert_mass_fit_equal(old, new, label=f"extract_mass_linear_abs seed={seed}")

    def test_extract_mass_aic_all_negative(self) -> None:
        """Correlator with no positive values should return degenerate result."""
        correlator = -torch.ones(50)
        dt = 1.0
        config = CorrelatorConfig()
        new_config = NewCorrelatorConfig()

        old = extract_mass_aic(correlator, dt, config)
        new = new_extract_aic(correlator, dt, new_config)

        assert_mass_fit_equal(old, new, label="extract_mass_aic all_negative")

    def test_extract_mass_linear_empty(self) -> None:
        """Empty correlator should return zero mass."""
        correlator = torch.tensor([])
        dt = 1.0
        config = CorrelatorConfig()
        new_config = NewCorrelatorConfig()

        old = extract_mass_linear(correlator, dt, config)
        new = new_extract_linear(correlator, dt, new_config)

        assert_mass_fit_equal(old, new, label="extract_mass_linear empty")


class TestParityComputeChannelCorrelator:
    """Parity tests for compute_channel_correlator."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_aic_mode(self, seed: int) -> None:
        series = _make_series_1d(seed, T=200)
        dt = 1.0
        config = CorrelatorConfig(
            max_lag=40,
            use_connected=True,
            window_widths=[5, 10, 15],
            fit_mode="aic",
        )
        new_config = NewCorrelatorConfig(
            max_lag=40,
            use_connected=True,
            window_widths=[5, 10, 15],
            fit_mode="aic",
        )

        old = compute_channel_correlator(series, dt, config, channel_name="test_ch")
        new = new_compute_channel(series, dt, new_config, channel_name="test_ch")

        _assert_channel_result_equal(old, new)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_linear_mode(self, seed: int) -> None:
        series = _make_series_1d(seed, T=200)
        dt = 1.0
        config = CorrelatorConfig(
            max_lag=40,
            use_connected=True,
            fit_mode="linear",
            fit_start=2,
            fit_stop=30,
        )
        new_config = NewCorrelatorConfig(
            max_lag=40,
            use_connected=True,
            fit_mode="linear",
            fit_start=2,
            fit_stop=30,
        )

        old = compute_channel_correlator(series, dt, config, channel_name="linear_ch")
        new = new_compute_channel(series, dt, new_config, channel_name="linear_ch")

        _assert_channel_result_equal(old, new)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bootstrap_errors(self, seed: int) -> None:
        series = _make_series_1d(seed, T=150)
        dt = 1.0
        config = CorrelatorConfig(
            max_lag=30,
            compute_bootstrap_errors=True,
            n_bootstrap=20,
            window_widths=[5, 10],
        )
        new_config = NewCorrelatorConfig(
            max_lag=30,
            compute_bootstrap_errors=True,
            n_bootstrap=20,
            window_widths=[5, 10],
        )

        torch.manual_seed(seed)
        old = compute_channel_correlator(series, dt, config, channel_name="boot_ch")

        torch.manual_seed(seed)
        new = new_compute_channel(series, dt, new_config, channel_name="boot_ch")

        _assert_channel_result_equal(old, new)

    def test_empty_series(self) -> None:
        series = torch.tensor([])
        dt = 1.0
        config = CorrelatorConfig(max_lag=10)
        new_config = NewCorrelatorConfig(max_lag=10)

        old = compute_channel_correlator(series, dt, config, channel_name="empty")
        new = new_compute_channel(series, dt, new_config, channel_name="empty")

        _assert_channel_result_equal(old, new)


class TestParityRegistry:
    """Parity tests for CHANNEL_REGISTRY and get_channel_class."""

    def test_registry_keys_identical(self) -> None:
        assert (
            set(CHANNEL_REGISTRY.keys()) == set(NEW_CHANNEL_REGISTRY.keys())
        ), f"Registry keys differ: {set(CHANNEL_REGISTRY.keys()) ^ set(NEW_CHANNEL_REGISTRY.keys())}"

    def test_registry_has_all_expected_channels(self) -> None:
        for ch in ALL_CHANNELS:
            assert ch in CHANNEL_REGISTRY, f"Missing from old registry: {ch}"
            assert ch in NEW_CHANNEL_REGISTRY, f"Missing from new registry: {ch}"

    @pytest.mark.parametrize("channel", ALL_CHANNELS)
    def test_get_channel_class_parity(self, channel: str) -> None:
        old_cls = get_channel_class(channel)
        new_cls = new_get_channel_class(channel)

        assert (
            old_cls.channel_name == new_cls.channel_name
        ), f"{channel}: channel_name mismatch: {old_cls.channel_name} vs {new_cls.channel_name}"
        # Both should be the same class since the AIC copy imports from the same module
        assert (
            old_cls.__name__ == new_cls.__name__
        ), f"{channel}: class name mismatch: {old_cls.__name__} vs {new_cls.__name__}"

    def test_registry_length(self) -> None:
        assert len(CHANNEL_REGISTRY) == len(
            NEW_CHANNEL_REGISTRY
        ), f"Registry lengths differ: {len(CHANNEL_REGISTRY)} vs {len(NEW_CHANNEL_REGISTRY)}"
        assert len(CHANNEL_REGISTRY) == 7, f"Expected 7 channels, got {len(CHANNEL_REGISTRY)}"
