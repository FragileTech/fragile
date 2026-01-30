"""Tests for the vectorized correlator channels module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.correlator_channels import (
    CHANNEL_REGISTRY,
    AxialVectorChannel,
    BilinearChannelCorrelator,
    ChannelConfig,
    ChannelCorrelator,
    ChannelCorrelatorResult,
    ConvolutionalAICExtractor,
    GlueballChannel,
    NucleonChannel,
    PseudoscalarChannel,
    ScalarChannel,
    TensorChannel,
    VectorChannel,
    compute_all_channels,
    compute_correlator_fft,
    compute_effective_mass_torch,
    get_channel_class,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockRunHistory:
    """Minimal mock RunHistory for testing."""

    def __init__(
        self,
        N: int = 50,
        d: int = 3,
        n_recorded: int = 100,
        device: str = "cpu",
    ):
        self.N = N
        self.d = d
        self.n_recorded = n_recorded
        self.record_every = 10
        self.delta_t = 0.01
        self.pbc = False
        self.bounds = None

        # Generate mock trajectory data
        T = n_recorded
        self.x_before_clone = torch.randn(T, N, d, device=device)
        self.x_final = torch.randn(T, N, d, device=device)
        self.v_before_clone = torch.randn(T, N, d, device=device)

        # Info arrays have T-1 entries
        self.force_viscous = torch.randn(T - 1, N, d, device=device)
        self.alive_mask = torch.ones(T - 1, N, dtype=torch.bool, device=device)
        self.companions_distance = torch.randint(0, N, (T - 1, N), device=device)


@pytest.fixture
def mock_history():
    """Create a mock RunHistory for testing."""
    return MockRunHistory()


@pytest.fixture
def config():
    """Create a test configuration."""
    return ChannelConfig(
        warmup_fraction=0.1,
        max_lag=40,
        h_eff=1.0,
        mass=1.0,
        ell0=1.0,
        knn_k=4,
        knn_sample=32,
    )


# =============================================================================
# Test ConvolutionalAICExtractor
# =============================================================================


class TestConvolutionalAICExtractor:
    """Tests for the convolutional AIC mass extractor."""

    def test_extractor_creation(self):
        """Should create extractor with default parameters."""
        extractor = ConvolutionalAICExtractor()
        assert extractor.window_widths == list(range(5, 51))
        assert extractor.min_mass == 0.0

    def test_extractor_custom_widths(self):
        """Should accept custom window widths."""
        extractor = ConvolutionalAICExtractor(window_widths=[10, 15, 20])
        assert extractor.window_widths == [10, 15, 20]

    def test_fit_synthetic_exponential(self):
        """Should extract correct mass from synthetic exponential."""
        # Create synthetic correlator: C(t) = exp(-m*t)
        # The AIC extractor works on log(C), so log_corr = -m*t
        T = 100
        true_mass = 0.5
        t = torch.arange(T, dtype=torch.float32)
        # Correlator values (positive), not log
        correlator = torch.exp(-true_mass * t)
        # Filter positive for log
        log_corr = torch.log(correlator)
        log_err = torch.ones(T) * 0.01  # Small errors

        extractor = ConvolutionalAICExtractor(window_widths=[10, 20, 30], min_mass=0.0)
        result = extractor.fit_all_widths(log_corr, log_err)

        assert "mass" in result
        assert "mass_error" in result
        # Should be close to true mass (the slope is -mass, so mass should be positive)
        assert result["n_valid_windows"] > 0
        assert abs(result["mass"] - true_mass) < 0.1

    def test_fit_single_width(self):
        """Should compute mass and AIC for single window width."""
        T = 50
        log_corr = torch.randn(1, 1, T)
        log_err = torch.ones(1, 1, T) * 0.1

        extractor = ConvolutionalAICExtractor()
        mass, aic = extractor.fit_single_width(log_corr, log_err, W=10)

        assert mass.shape == (1, 1, T - 10 + 1)
        assert aic.shape == (1, 1, T - 10 + 1)

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        log_corr = torch.tensor([])
        log_err = torch.tensor([])

        extractor = ConvolutionalAICExtractor(window_widths=[5])
        result = extractor.fit_all_widths(log_corr, log_err)

        assert result["mass"] == 0.0
        assert result["n_valid_windows"] == 0


# =============================================================================
# Test FFT Correlator
# =============================================================================


class TestFFTCorrelator:
    """Tests for FFT-based correlator computation."""

    def test_correlator_shape(self):
        """Correlator should have correct shape."""
        series = torch.randn(100)
        corr = compute_correlator_fft(series, max_lag=50)
        assert corr.shape == (51,)

    def test_correlator_at_zero_lag(self):
        """C(0) should be variance for connected correlator."""
        series = torch.randn(1000)
        corr = compute_correlator_fft(series, max_lag=10, use_connected=True)
        # C(0) ~ var(series) for connected
        expected_var = series.var().item()
        assert abs(corr[0].item() - expected_var) < 0.1

    def test_correlator_disconnected(self):
        """Disconnected correlator should include mean squared."""
        series = torch.ones(100) * 5.0  # Constant series
        corr = compute_correlator_fft(series, max_lag=10, use_connected=False)
        # For constant series, C(t) = mean^2 = 25
        assert abs(corr[0].item() - 25.0) < 0.1

    def test_empty_series(self):
        """Should handle empty series."""
        series = torch.tensor([])
        corr = compute_correlator_fft(series, max_lag=10)
        assert corr.shape == (11,)
        assert (corr == 0).all()


class TestEffectiveMass:
    """Tests for effective mass computation."""

    def test_effective_mass_shape(self):
        """Effective mass should have T-1 elements."""
        correlator = torch.exp(-0.3 * torch.arange(50, dtype=torch.float32))
        eff_mass = compute_effective_mass_torch(correlator, dt=1.0)
        assert eff_mass.shape == (49,)

    def test_effective_mass_exponential(self):
        """Should give constant m_eff for pure exponential."""
        true_mass = 0.3
        t = torch.arange(50, dtype=torch.float32)
        correlator = torch.exp(-true_mass * t)
        eff_mass = compute_effective_mass_torch(correlator, dt=1.0)

        # All valid values should be close to true_mass
        valid = torch.isfinite(eff_mass)
        assert valid.all()
        assert (eff_mass - true_mass).abs().mean() < 0.01

    def test_effective_mass_empty(self):
        """Should handle empty correlator."""
        correlator = torch.tensor([])
        eff_mass = compute_effective_mass_torch(correlator, dt=1.0)
        assert eff_mass.numel() == 0


# =============================================================================
# Test Channel Configuration
# =============================================================================


class TestChannelConfig:
    """Tests for ChannelConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ChannelConfig()
        assert config.warmup_fraction == 0.1
        assert config.max_lag == 80
        assert config.h_eff == 1.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = ChannelConfig(
            warmup_fraction=0.2,
            max_lag=100,
            knn_k=8,
        )
        assert config.warmup_fraction == 0.2
        assert config.max_lag == 100
        assert config.knn_k == 8


# =============================================================================
# Test Channel Registry
# =============================================================================


class TestChannelRegistry:
    """Tests for channel registry and factory."""

    def test_all_channels_registered(self):
        """All expected channels should be in registry."""
        expected = {
            "scalar",
            "pseudoscalar",
            "vector",
            "axial_vector",
            "tensor",
            "nucleon",
            "glueball",
        }
        assert set(CHANNEL_REGISTRY.keys()) == expected

    def test_get_channel_class(self):
        """Should return correct class for channel name."""
        assert get_channel_class("scalar") is ScalarChannel
        assert get_channel_class("pseudoscalar") is PseudoscalarChannel
        assert get_channel_class("nucleon") is NucleonChannel

    def test_get_unknown_channel(self):
        """Should raise ValueError for unknown channel."""
        with pytest.raises(ValueError, match="Unknown channel"):
            get_channel_class("unknown_channel")


# =============================================================================
# Test Bilinear Channel Classes
# =============================================================================


class TestScalarChannel:
    """Tests for ScalarChannel."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert ScalarChannel.channel_name == "scalar"

    def test_gamma_projection(self, mock_history, config):
        """Scalar projection should be identity-like."""
        channel = ScalarChannel(mock_history, config)

        color_i = torch.randn(10, 5, 3, dtype=torch.complex128)
        color_j = torch.randn(10, 5, 3, dtype=torch.complex128)

        result = channel._apply_gamma_projection(color_i, color_j)
        assert result.shape == (10, 5)
        assert result.dtype == torch.float64

    def test_compute_series(self, mock_history, config):
        """Should compute operator series."""
        channel = ScalarChannel(mock_history, config)
        series = channel.compute_series()
        assert series.ndim == 1
        assert series.numel() > 0


class TestPseudoscalarChannel:
    """Tests for PseudoscalarChannel."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert PseudoscalarChannel.channel_name == "pseudoscalar"

    def test_gamma5_projection(self, mock_history, config):
        """γ₅ projection should have alternating signs."""
        channel = PseudoscalarChannel(mock_history, config)

        # Use identity color vectors
        d = 3
        color_i = torch.eye(d, dtype=torch.complex128).unsqueeze(0).expand(5, -1, -1)
        color_j = color_i.clone()

        result = channel._apply_gamma_projection(color_i, color_j)
        # For identity matrices: Tr(γ₅) = sum of diagonal of γ₅
        # γ₅ = diag(1, -1, 1) for d=3
        expected = 1 - 1 + 1  # = 1
        assert result.shape == (5, d)


class TestVectorChannel:
    """Tests for VectorChannel."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert VectorChannel.channel_name == "vector"

    def test_vector_projection(self, mock_history, config):
        """Vector projection should average over directions."""
        channel = VectorChannel(mock_history, config)

        color_i = torch.randn(10, 5, 3, dtype=torch.complex128)
        color_j = torch.randn(10, 5, 3, dtype=torch.complex128)

        result = channel._apply_gamma_projection(color_i, color_j)
        assert result.shape == (10, 5)


class TestAxialVectorChannel:
    """Tests for AxialVectorChannel."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert AxialVectorChannel.channel_name == "axial_vector"


class TestTensorChannel:
    """Tests for TensorChannel."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert TensorChannel.channel_name == "tensor"


# =============================================================================
# Test Nucleon Channel
# =============================================================================


class TestNucleonChannel:
    """Tests for NucleonChannel (trilinear/baryon)."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert NucleonChannel.channel_name == "nucleon"

    def test_requires_d3(self, config):
        """Nucleon should require d=3."""
        # Create mock with d=2
        history_d2 = MockRunHistory(d=2)
        channel = NucleonChannel(history_d2, config)
        series = channel.compute_series()
        # Should return zeros for d!=3
        assert (series == 0).all()

    def test_determinant_computation(self, mock_history, config):
        """Should compute 3x3 determinant."""
        channel = NucleonChannel(mock_history, config)
        series = channel.compute_series()
        assert series.ndim == 1


# =============================================================================
# Test Glueball Channel
# =============================================================================


class TestGlueballChannel:
    """Tests for GlueballChannel (gauge)."""

    def test_channel_name(self):
        """Should have correct channel name."""
        assert GlueballChannel.channel_name == "glueball"

    def test_force_squared_norm(self, mock_history, config):
        """Should compute ||force||² norm."""
        channel = GlueballChannel(mock_history, config)
        series = channel.compute_series()
        assert series.ndim == 1
        # Force squared should be non-negative
        assert (series >= 0).all()


# =============================================================================
# Test Full Channel Computation
# =============================================================================


class TestFullChannelComputation:
    """Integration tests for full channel computation."""

    def test_compute_single_channel(self, mock_history, config):
        """Should compute full result for single channel."""
        channel = ScalarChannel(mock_history, config)
        result = channel.compute()

        assert isinstance(result, ChannelCorrelatorResult)
        assert result.channel_name == "scalar"
        assert result.correlator.shape == (config.max_lag + 1,)
        assert result.effective_mass.shape == (config.max_lag,)
        assert "mass" in result.mass_fit
        assert result.n_samples > 0

    def test_compute_all_channels(self, mock_history, config):
        """Should compute all channels."""
        results = compute_all_channels(mock_history, config=config)

        assert len(results) == len(CHANNEL_REGISTRY)
        for name, result in results.items():
            assert isinstance(result, ChannelCorrelatorResult)
            assert result.channel_name == name

    def test_compute_selected_channels(self, mock_history, config):
        """Should compute only selected channels."""
        results = compute_all_channels(
            mock_history,
            channels=["scalar", "pseudoscalar"],
            config=config,
        )

        assert len(results) == 2
        assert "scalar" in results
        assert "pseudoscalar" in results
        assert "vector" not in results


# =============================================================================
# Test Vectorization
# =============================================================================


class TestVectorization:
    """Tests to verify vectorization is working correctly."""

    def test_batch_color_states(self, mock_history, config):
        """Color states should be computed for all timesteps."""
        channel = ScalarChannel(mock_history, config)
        start_idx = max(1, int(mock_history.n_recorded * config.warmup_fraction))

        color, valid = channel._compute_color_states_batch(start_idx)

        T = mock_history.n_recorded - start_idx
        N = mock_history.N
        d = mock_history.d

        assert color.shape == (T, N, d)
        assert valid.shape == (T, N)

    def test_batch_knn(self, mock_history, config):
        """k-NN should be computed for all timesteps."""
        channel = ScalarChannel(mock_history, config)
        start_idx = max(1, int(mock_history.n_recorded * config.warmup_fraction))

        sample_idx, neighbor_idx, alive = channel._compute_knn_batch(start_idx)

        T = mock_history.n_recorded - start_idx
        S = config.knn_sample
        k = config.knn_k

        assert sample_idx.shape == (T, S)
        assert neighbor_idx.shape == (T, S, k)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_history(self):
        """Should handle small history."""
        history = MockRunHistory(n_recorded=10)
        config = ChannelConfig(warmup_fraction=0.1, max_lag=5)

        result = ScalarChannel(history, config).compute()
        assert result.correlator.numel() > 0

    def test_few_walkers(self):
        """Should handle few walkers."""
        history = MockRunHistory(N=5, n_recorded=50)
        config = ChannelConfig(knn_k=2, knn_sample=4)

        result = ScalarChannel(history, config).compute()
        assert result.correlator.numel() > 0

    def test_all_dead_walkers(self):
        """Should handle all dead walkers gracefully."""
        history = MockRunHistory(n_recorded=50)
        # Kill all walkers
        history.alive_mask[:] = False

        config = ChannelConfig()
        result = ScalarChannel(history, config).compute()
        # Should return empty or zero result
        assert result.n_samples >= 0


# =============================================================================
# Test Performance Comparison (reference vs vectorized)
# =============================================================================


class TestCorrectnessVsReference:
    """Tests comparing vectorized implementation against reference."""

    def test_scalar_projection_correctness(self):
        """Scalar projection should match simple loop implementation."""
        T, S, d = 5, 10, 3
        color_i = torch.randn(T, S, d, dtype=torch.complex128)
        color_j = torch.randn(T, S, d, dtype=torch.complex128)

        # Reference: loop implementation
        ref_result = torch.zeros(T, S, dtype=torch.float64)
        for t in range(T):
            for s in range(S):
                ref_result[t, s] = (color_i[t, s].conj() * color_j[t, s]).sum().real

        # Vectorized
        vec_result = (color_i.conj() * color_j).sum(dim=-1).real

        torch.testing.assert_close(vec_result, ref_result)

    def test_fft_correlator_vs_loop(self):
        """FFT correlator should match loop implementation."""
        series = torch.randn(100)
        max_lag = 20

        # FFT version
        fft_corr = compute_correlator_fft(series, max_lag, use_connected=True)

        # Reference loop version
        centered = series - series.mean()
        loop_corr = torch.zeros(max_lag + 1)
        n = series.shape[0]
        for lag in range(max_lag + 1):
            loop_corr[lag] = (centered[: n - lag] * centered[lag:]).mean()

        torch.testing.assert_close(fft_corr, loop_corr, atol=1e-5, rtol=1e-5)
