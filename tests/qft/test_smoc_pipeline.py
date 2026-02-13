"""
Tests for the SMoC (Standard Model of Cognition) simulation and analysis pipeline.

Tests cover all 6 phases:
- Phase 1: Simulation (SMoCSimulator)
- Phase 2: Channel Projection (ChannelProjector)
- Phase 3: Correlation Calculation (CorrelatorComputer)
- Phase 4: Statistical Aggregation (aggregate_correlators)
- Phase 5-6: Mass Extraction (MassExtractor)
- Full Pipeline: run_smoc_pipeline

Also includes validation tests for physics consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.smoc_pipeline import (
    aggregate_correlators,
    AggregatedCorrelator,
    ChannelProjector,
    compute_smoc_correlators_from_history,
    CorrelatorComputer,
    CorrelatorConfig,
    MassExtractionConfig,
    MassExtractor,
    ProjectorConfig,
    run_smoc_pipeline,
    SimulationConfig,
    SMoCPipelineConfig,
    SMoCSimulator,
)


# =============================================================================
# Phase 1: Simulation Tests
# =============================================================================


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.batch_size == 1000
        assert config.grid_size == 64
        assert config.internal_dim == 4
        assert config.t_thermalization == 500
        assert config.t_measurement == 1000
        assert config.init_mode == "hot"
        assert config.use_pbc is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            batch_size=100,
            grid_size=32,
            internal_dim=8,
            init_mode="cold",
        )
        assert config.batch_size == 100
        assert config.grid_size == 32
        assert config.internal_dim == 8
        assert config.init_mode == "cold"


class TestSMoCSimulator:
    """Tests for SMoCSimulator class."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for fast tests."""
        return SimulationConfig(
            batch_size=4,
            grid_size=8,
            internal_dim=4,
            t_thermalization=10,
            t_measurement=20,
            seed=42,
        )

    def test_initialization_hot(self, small_config):
        """Test hot start initialization."""
        small_config.init_mode = "hot"
        sim = SMoCSimulator(small_config)

        assert sim.agents.shape == (4, 8, 4)
        # Hot start should have unit norm vectors
        norms = torch.linalg.vector_norm(sim.agents, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_initialization_cold(self, small_config):
        """Test cold start initialization."""
        small_config.init_mode = "cold"
        sim = SMoCSimulator(small_config)

        assert sim.agents.shape == (4, 8, 4)
        # Cold start should have all agents pointing in first direction
        assert torch.allclose(sim.agents[..., 0], torch.ones(4, 8))
        assert torch.allclose(sim.agents[..., 1:], torch.zeros(4, 8, 3))

    def test_reproducibility_with_seed(self, small_config):
        """Test that seed produces reproducible results."""
        small_config.seed = 123
        sim1 = SMoCSimulator(small_config)
        agents1 = sim1.agents.clone()

        small_config.seed = 123
        sim2 = SMoCSimulator(small_config)
        agents2 = sim2.agents.clone()

        assert torch.allclose(agents1, agents2)

    def test_run_thermalization(self, small_config):
        """Test thermalization phase."""
        sim = SMoCSimulator(small_config)
        initial_agents = sim.agents.clone()

        sim.run_thermalization()

        # Agents should have changed
        assert not torch.allclose(sim.agents, initial_agents)
        # Should still be normalized
        norms = torch.linalg.vector_norm(sim.agents, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_run_measurement(self, small_config):
        """Test measurement phase."""
        sim = SMoCSimulator(small_config)
        sim.run_thermalization()

        history = sim.run_measurement()

        assert history.shape == (4, 20, 8, 4)
        assert sim.history is not None
        assert torch.equal(history, sim.history)

    def test_full_run(self, small_config):
        """Test full simulation run."""
        sim = SMoCSimulator(small_config)
        history = sim.run()

        assert history.shape == (4, 20, 8, 4)
        # All states should be normalized
        norms = torch.linalg.vector_norm(history, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_pbc_vs_non_pbc(self, small_config):
        """Test periodic vs non-periodic boundary conditions."""
        small_config.seed = 42
        small_config.use_pbc = True
        sim_pbc = SMoCSimulator(small_config)
        history_pbc = sim_pbc.run()

        small_config.seed = 42
        small_config.use_pbc = False
        sim_no_pbc = SMoCSimulator(small_config)
        history_no_pbc = sim_no_pbc.run()

        # Results should differ due to boundary treatment
        assert not torch.allclose(history_pbc, history_no_pbc)


# =============================================================================
# Phase 2: Channel Projection Tests
# =============================================================================


class TestProjectorConfig:
    """Tests for ProjectorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProjectorConfig()
        assert config.internal_dim == 4
        assert config.device == "cpu"


class TestChannelProjector:
    """Tests for ChannelProjector class."""

    @pytest.fixture
    def projector(self):
        """Default projector for tests."""
        config = ProjectorConfig(internal_dim=4)
        return ChannelProjector(config)

    @pytest.fixture
    def sample_history(self):
        """Sample history tensor for tests."""
        torch.manual_seed(42)
        return torch.randn(2, 10, 5, 4)  # batch=2, time=10, grid=5, dim=4

    def test_projector_initialization(self, projector):
        """Test projector builds all channel matrices."""
        expected_channels = {"scalar", "pion", "rho", "sigma", "eta", "nucleon"}
        assert set(projector.projectors.keys()) == expected_channels

    def test_projector_shapes(self, projector):
        """Test projector matrices have correct shapes."""
        for name, matrix in projector.projectors.items():
            assert matrix.shape == (4, 4), f"Channel {name} has wrong shape"

    def test_scalar_projection(self, projector, sample_history):
        """Test scalar (identity) projection."""
        field = projector.project(sample_history, "scalar")

        assert field.shape == (2, 10, 5)
        # Scalar should sum all components
        expected = sample_history.sum(dim=-1)
        assert torch.allclose(field, expected)

    def test_pion_projection(self, projector, sample_history):
        """Test pion (γ₅) projection."""
        field = projector.project(sample_history, "pion")

        assert field.shape == (2, 10, 5)
        # Pion uses alternating signs
        expected = (
            sample_history[..., 0]
            - sample_history[..., 1]
            + sample_history[..., 2]
            - sample_history[..., 3]
        )
        assert torch.allclose(field, expected)

    def test_unknown_channel_raises(self, projector, sample_history):
        """Test that unknown channel raises ValueError."""
        with pytest.raises(ValueError, match="Unknown channel"):
            projector.project(sample_history, "unknown_particle")

    def test_project_all(self, projector, sample_history):
        """Test projecting all channels at once."""
        all_fields = projector.project_all(sample_history)

        assert len(all_fields) == 6
        for field in all_fields.values():
            assert field.shape == (2, 10, 5)


# =============================================================================
# Phase 3: Correlation Calculation Tests
# =============================================================================


class TestCorrelatorConfig:
    """Tests for CorrelatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = CorrelatorConfig()
        assert config.use_connected is True
        assert config.normalize is True


class TestCorrelatorComputer:
    """Tests for CorrelatorComputer class."""

    @pytest.fixture
    def computer(self):
        """Default correlator computer."""
        return CorrelatorComputer()

    @pytest.fixture
    def sample_field(self):
        """Sample field tensor."""
        torch.manual_seed(42)
        return torch.randn(4, 50, 10)  # batch=4, time=50, grid=10

    def test_spatial_average(self, computer, sample_field):
        """Test spatial averaging."""
        avg = computer.spatial_average(sample_field)

        assert avg.shape == (4, 50)
        expected = sample_field.mean(dim=-1)
        assert torch.allclose(avg, expected)

    def test_autocorrelation_shape(self, computer):
        """Test autocorrelation output shape."""
        signal = torch.randn(4, 100)
        corr = computer.compute_autocorrelation_fft(signal)

        assert corr.shape == (4, 100)

    def test_autocorrelation_normalized(self, computer):
        """Test that normalized correlator starts at 1."""
        signal = torch.randn(4, 100)
        corr = computer.compute_autocorrelation_fft(signal)

        # C(0) should be 1 when normalized
        assert torch.allclose(corr[:, 0], torch.ones(4), atol=1e-5)

    def test_autocorrelation_not_normalized(self):
        """Test correlator without normalization."""
        config = CorrelatorConfig(normalize=False)
        computer = CorrelatorComputer(config)
        signal = torch.randn(4, 100)
        corr = computer.compute_autocorrelation_fft(signal)

        # C(0) should not be 1
        assert not torch.allclose(corr[:, 0], torch.ones(4))

    def test_autocorrelation_decay(self, computer):
        """Test that correlator decays over time."""
        # Create signal with known correlation structure
        torch.manual_seed(42)
        t = torch.arange(100, dtype=torch.float32)
        signal = torch.exp(-0.05 * t) + 0.1 * torch.randn(100)
        signal = signal.unsqueeze(0)

        corr = computer.compute_autocorrelation_fft(signal)

        # Correlator should generally decrease
        assert corr[0, 0] >= corr[0, 10]
        assert corr[0, 10] >= corr[0, 30]

    def test_compute_correlator_full(self, computer, sample_field):
        """Test full correlator computation."""
        corr = computer.compute_correlator(sample_field)

        assert corr.shape == (4, 50)
        assert torch.allclose(corr[:, 0], torch.ones(4), atol=1e-5)


# =============================================================================
# Phase 4: Statistical Aggregation Tests
# =============================================================================


class TestAggregateCorrelators:
    """Tests for aggregate_correlators function."""

    @pytest.fixture
    def sample_correlators(self):
        """Sample correlator data."""
        torch.manual_seed(42)
        # Simulate correlators that decay exponentially with noise
        t = torch.arange(50, dtype=torch.float32)
        base = torch.exp(-0.1 * t)
        noise = 0.05 * torch.randn(100, 50)
        return base.unsqueeze(0) + noise

    def test_aggregation_shapes(self, sample_correlators):
        """Test aggregated output shapes."""
        agg = aggregate_correlators(sample_correlators)

        assert agg.mean.shape == (50,)
        assert agg.std.shape == (50,)
        assert agg.std_err.shape == (50,)
        assert agg.n_samples == 100

    def test_aggregation_statistics(self, sample_correlators):
        """Test aggregation statistics are correct."""
        agg = aggregate_correlators(sample_correlators)

        expected_mean = sample_correlators.mean(dim=0)
        expected_std = sample_correlators.std(dim=0)
        expected_err = expected_std / (100**0.5)

        assert torch.allclose(agg.mean, expected_mean)
        assert torch.allclose(agg.std, expected_std)
        assert torch.allclose(agg.std_err, expected_err)

    def test_aggregation_keep_raw(self, sample_correlators):
        """Test keeping raw data."""
        agg_with_raw = aggregate_correlators(sample_correlators, keep_raw=True)
        agg_no_raw = aggregate_correlators(sample_correlators, keep_raw=False)

        assert agg_with_raw.raw is not None
        assert torch.equal(agg_with_raw.raw, sample_correlators)
        assert agg_no_raw.raw is None


# =============================================================================
# Phase 5-6: Mass Extraction Tests
# =============================================================================


class TestMassExtractionConfig:
    """Tests for MassExtractionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = MassExtractionConfig()
        assert config.min_window_length == 5
        assert config.min_t_start == 2
        assert config.min_mass == 0.0


class TestMassExtractor:
    """Tests for MassExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Default mass extractor."""
        config = MassExtractionConfig(
            min_window_length=3,
            min_t_start=1,
        )
        return MassExtractor(config)

    @pytest.fixture
    def exponential_correlator(self):
        """Create correlator with known exponential decay."""
        torch.manual_seed(42)
        t = torch.arange(50, dtype=torch.float32)
        true_mass = 0.15
        true_amp = 1.0

        # Generate many samples
        base = true_amp * torch.exp(-true_mass * t)
        noise = 0.02 * torch.randn(200, 50)
        correlators = base.unsqueeze(0) + noise

        agg = aggregate_correlators(correlators)
        return agg, true_mass

    def test_generate_windows(self, extractor):
        """Test window generation."""
        t_starts, t_ends = extractor.generate_windows(t_max=20)

        assert len(t_starts) == len(t_ends)
        assert len(t_starts) > 0

        # All windows should have minimum length
        lengths = t_ends - t_starts
        assert (lengths >= 3).all()

        # All starts should be >= min_t_start
        assert (t_starts >= 1).all()

    def test_extract_mass_exponential(self, extractor, exponential_correlator):
        """Test mass extraction from exponential decay."""
        agg, true_mass = exponential_correlator
        result = extractor.extract_mass(agg)

        assert "mass" in result
        assert "mass_error" in result
        assert "n_valid_windows" in result
        assert "best_window" in result

        # Extracted mass should be close to true mass
        assert abs(result["mass"] - true_mass) < 0.05
        assert result["n_valid_windows"] > 0

    def test_extract_mass_with_constraints(self, exponential_correlator):
        """Test mass extraction with mass constraints."""
        agg, _true_mass = exponential_correlator

        # Set minimum mass well above true mass
        config = MassExtractionConfig(
            min_mass=1.0,  # Very high threshold
            min_window_length=3,
            min_t_start=1,
        )
        extractor = MassExtractor(config)
        result = extractor.extract_mass(agg)

        # Should find few or no valid windows (true mass is ~0.15)
        # Some noisy samples might still exceed threshold, so we allow a few
        assert result["n_valid_windows"] < 10

    def test_aic_weights_sum_to_one(self, extractor):
        """Test that AIC weights are properly normalized."""
        aic = torch.tensor([10.0, 12.0, 15.0, 20.0])
        valid = torch.tensor([True, True, True, True])

        weights = extractor.compute_aic_weights(aic, valid)

        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_aic_weights_prefer_lower_aic(self, extractor):
        """Test that lower AIC gets higher weight."""
        aic = torch.tensor([10.0, 20.0, 30.0])
        valid = torch.tensor([True, True, True])

        weights = extractor.compute_aic_weights(aic, valid)

        # Weights should be ordered by AIC (lower AIC = higher weight)
        assert weights[0] > weights[1] > weights[2]


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestRunSMoCPipeline:
    """Tests for the full run_smoc_pipeline function."""

    def test_pipeline_runs(self):
        """Test that pipeline completes without errors."""
        result = run_smoc_pipeline(
            batch_size=8,
            grid_size=8,
            internal_dim=4,
            t_thermalization=20,
            t_measurement=50,
            channels=("scalar", "pion"),
            seed=42,
            verbose=False,
        )

        assert result is not None
        assert "scalar" in result.masses
        assert "pion" in result.masses

    def test_pipeline_output_structure(self):
        """Test pipeline output structure."""
        result = run_smoc_pipeline(
            batch_size=4,
            grid_size=8,
            internal_dim=4,
            t_thermalization=10,
            t_measurement=30,
            channels=("scalar", "pion", "rho"),
            seed=42,
            keep_history=True,
            verbose=False,
        )

        # Check history
        assert result.history is not None
        assert result.history.shape == (4, 30, 8, 4)

        # Check projected fields
        assert len(result.projected_fields) == 3
        for ch in ("scalar", "pion", "rho"):
            assert ch in result.projected_fields
            assert result.projected_fields[ch].shape == (4, 30, 8)

        # Check correlators
        assert len(result.correlators) == 3
        for ch in ("scalar", "pion", "rho"):
            assert ch in result.correlators
            assert result.correlators[ch].shape == (4, 30)

        # Check aggregated
        assert len(result.aggregated) == 3
        for ch in ("scalar", "pion", "rho"):
            assert ch in result.aggregated
            assert isinstance(result.aggregated[ch], AggregatedCorrelator)

        # Check masses
        assert len(result.masses) == 3
        for ch in ("scalar", "pion", "rho"):
            assert ch in result.masses
            assert "mass" in result.masses[ch]
            assert "mass_error" in result.masses[ch]

    def test_pipeline_no_history(self):
        """Test pipeline without keeping history."""
        result = run_smoc_pipeline(
            batch_size=4,
            grid_size=8,
            internal_dim=4,
            t_thermalization=10,
            t_measurement=30,
            keep_history=False,
            verbose=False,
        )

        assert result.history is None

    def test_pipeline_reproducibility(self):
        """Test pipeline reproducibility with seed."""
        kwargs = {
            "batch_size": 4,
            "grid_size": 8,
            "internal_dim": 4,
            "t_thermalization": 10,
            "t_measurement": 30,
            "channels": ("scalar",),
            "seed": 123,
            "verbose": False,
        }

        result1 = run_smoc_pipeline(**kwargs)
        result2 = run_smoc_pipeline(**kwargs)

        assert result1.masses["scalar"]["mass"] == result2.masses["scalar"]["mass"]


# =============================================================================
# Integration with Fractal Gas Tests
# =============================================================================


class TestComputeSMoCCorrelatorsFromHistory:
    """Tests for compute_smoc_correlators_from_history function."""

    @pytest.fixture
    def fractal_gas_history(self):
        """Simulate Fractal Gas history data."""
        torch.manual_seed(42)
        t_steps = 100
        n_walkers = 50
        dims = 3

        positions = torch.randn(t_steps, n_walkers, dims)
        velocities = torch.randn(t_steps, n_walkers, dims)
        alive = torch.ones(t_steps, n_walkers, dtype=torch.bool)

        return positions, velocities, alive

    def test_basic_computation(self, fractal_gas_history):
        """Test basic correlator computation from Fractal Gas data."""
        positions, velocities, alive = fractal_gas_history

        results = compute_smoc_correlators_from_history(
            positions,
            velocities,
            alive,
            channels=("scalar", "pion"),
            max_lag=50,
        )

        assert "scalar" in results
        assert "pion" in results

        for ch in ("scalar", "pion"):
            assert "correlator" in results[ch]
            assert "lags" in results[ch]
            if results[ch]["correlator"] is not None:
                assert len(results[ch]["correlator"]) == 50
                assert len(results[ch]["lags"]) == 50

    def test_with_partial_alive(self, fractal_gas_history):
        """Test with some walkers dead."""
        positions, velocities, alive = fractal_gas_history

        # Kill some walkers
        alive[:, 10:20] = False

        results = compute_smoc_correlators_from_history(
            positions,
            velocities,
            alive,
            channels=("scalar",),
        )

        assert results["scalar"]["correlator"] is not None


# =============================================================================
# Physics Validation Tests
# =============================================================================


class TestPhysicsConsistency:
    """Tests for physics consistency of the pipeline."""

    def test_correlator_positivity(self):
        """Test that correlators are positive at t=0."""
        result = run_smoc_pipeline(
            batch_size=16,
            grid_size=16,
            internal_dim=4,
            t_thermalization=50,
            t_measurement=100,
            channels=("scalar", "pion", "rho"),
            seed=42,
            verbose=False,
        )

        for ch in ("scalar", "pion", "rho"):
            agg = result.aggregated[ch]
            # C(0) should be positive (and ~1 due to normalization)
            assert agg.mean[0] > 0

    def test_correlator_decay(self):
        """Test that correlators decay over time."""
        result = run_smoc_pipeline(
            batch_size=32,
            grid_size=32,
            internal_dim=4,
            t_thermalization=100,
            t_measurement=200,
            channels=("scalar",),
            seed=42,
            verbose=False,
        )

        corr = result.aggregated["scalar"].mean

        # Correlator should generally decrease (with some noise)
        # Check that average of first quarter > average of last quarter
        n = len(corr)
        first_quarter = corr[: n // 4].mean()
        last_quarter = corr[3 * n // 4 :].mean()
        assert first_quarter > last_quarter

    def test_mass_is_positive(self):
        """Test that extracted masses are non-negative."""
        result = run_smoc_pipeline(
            batch_size=32,
            grid_size=32,
            internal_dim=4,
            t_thermalization=100,
            t_measurement=200,
            channels=("scalar", "pion", "rho"),
            seed=42,
            verbose=False,
        )

        for ch in ("scalar", "pion", "rho"):
            mass = result.masses[ch]["mass"]
            assert mass >= 0, f"Mass for {ch} should be non-negative"

    def test_error_decreases_with_statistics(self):
        """Test that mass error decreases with more samples."""
        result_small = run_smoc_pipeline(
            batch_size=16,
            grid_size=16,
            internal_dim=4,
            t_thermalization=50,
            t_measurement=100,
            channels=("scalar",),
            seed=42,
            verbose=False,
        )

        result_large = run_smoc_pipeline(
            batch_size=64,
            grid_size=16,
            internal_dim=4,
            t_thermalization=50,
            t_measurement=100,
            channels=("scalar",),
            seed=42,
            verbose=False,
        )

        # Larger batch should have smaller error (statistically)
        # Note: This is probabilistic, so we use a loose bound
        err_small = result_small.masses["scalar"]["mass_error"]
        err_large = result_large.masses["scalar"]["mass_error"]

        # Allow for statistical fluctuations but expect trend
        # Error should scale roughly as 1/sqrt(N)
        assert err_large < err_small * 1.5


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_batch(self):
        """Test with batch_size=1."""
        result = run_smoc_pipeline(
            batch_size=1,
            grid_size=8,
            internal_dim=4,
            t_thermalization=10,
            t_measurement=30,
            channels=("scalar",),
            verbose=False,
        )

        assert result.masses["scalar"]["mass"] >= 0

    def test_minimal_grid(self):
        """Test with minimal grid size."""
        result = run_smoc_pipeline(
            batch_size=4,
            grid_size=2,
            internal_dim=4,
            t_thermalization=10,
            t_measurement=30,
            channels=("scalar",),
            verbose=False,
        )

        assert result is not None

    def test_short_measurement(self):
        """Test with short measurement time."""
        result = run_smoc_pipeline(
            batch_size=4,
            grid_size=8,
            internal_dim=4,
            t_thermalization=5,
            t_measurement=10,
            channels=("scalar",),
            verbose=False,
        )

        # May not find valid windows, but should not crash
        assert result is not None

    def test_large_internal_dim(self):
        """Test with larger internal dimension."""
        result = run_smoc_pipeline(
            batch_size=4,
            grid_size=8,
            internal_dim=16,
            t_thermalization=10,
            t_measurement=30,
            channels=("scalar", "pion"),
            verbose=False,
        )

        assert result.projected_fields["scalar"].shape == (4, 30, 8)


# =============================================================================
# Performance Tests (Optional, marked slow)
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests (run with pytest -m slow)."""

    def test_large_batch(self):
        """Test with large batch size."""
        result = run_smoc_pipeline(
            batch_size=500,
            grid_size=64,
            internal_dim=4,
            t_thermalization=100,
            t_measurement=200,
            channels=("scalar", "pion", "rho"),
            verbose=False,
        )

        assert result is not None

    def test_cuda_if_available(self):
        """Test CUDA execution if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        result = run_smoc_pipeline(
            batch_size=100,
            grid_size=32,
            internal_dim=4,
            t_thermalization=50,
            t_measurement=100,
            channels=("scalar",),
            device="cuda",
            verbose=False,
        )

        assert result is not None
