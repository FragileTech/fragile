"""Comprehensive end-to-end and integration tests for the operators pipeline
and preparation modules.

Tests cover:
- _resolve_3d_dims helper
- _resolve_frame_indices helper
- prepare_channel_data
- PipelineConfig
- compute_strong_force_pipeline (full end-to-end)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.config import (
    ChannelConfigBase,
    CorrelatorConfig,
)
from fragile.physics.operators.pipeline import (
    compute_strong_force_pipeline,
    PipelineConfig,
    PipelineResult,
)
from fragile.physics.operators.preparation import (
    _resolve_3d_dims,
    _resolve_frame_indices,
    prepare_channel_data,
    PreparedChannelData,
)


# ---------------------------------------------------------------------------
# Mock RunHistory for unit tests of _resolve_frame_indices
# ---------------------------------------------------------------------------


class _SimpleHistoryStub:
    """Minimal stub with only n_recorded and recorded_steps for frame-index tests."""

    def __init__(self, n_recorded: int, n_steps: int = 100):
        self.n_recorded = n_recorded
        self.n_steps = n_steps
        self.recorded_steps = list(range(0, n_steps, max(1, n_steps // max(n_recorded, 1))))[
            :n_recorded
        ]

    def get_step_index(self, step: int) -> int:
        return self.recorded_steps.index(step)


# ---------------------------------------------------------------------------
# Full MockRunHistory for integration tests
# ---------------------------------------------------------------------------


class MockRunHistory:
    """Full mock that provides all attributes needed by the pipeline.

    Mimics a RunHistory with the fields accessed by:
    - prepare_channel_data (via compute_color_states_batch, estimate_ell0)
    - pipeline (positions, velocities, companions, scores, force_viscous)
    """

    def __init__(
        self,
        N: int = 20,
        d: int = 5,
        n_steps: int = 100,
        n_recorded: int = 50,
    ):
        self.N = N
        self.d = d
        self.n_steps = n_steps
        self.n_recorded = n_recorded

        gen = torch.Generator().manual_seed(42)

        # Positions [n_recorded, N, d]
        self.x_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        # Velocities [n_recorded, N, d]
        self.v_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        # Force viscous [n_recorded - 1, N, d] (used by compute_color_states_batch)
        self.force_viscous = torch.randn(n_recorded - 1, N, d, generator=gen)

        # Companion indices [n_recorded - 1, N]
        comp_dist = torch.arange(N).roll(-1).unsqueeze(0).expand(n_recorded - 1, -1)
        self.companions_distance = comp_dist.clone()
        self.companions_clone = (
            torch.arange(N).roll(-2).unsqueeze(0).expand(n_recorded - 1, -1).clone()
        )

        # Scores [n_recorded - 1, N]
        self.cloning_scores = torch.randn(n_recorded - 1, N, generator=gen)

        # Recorded steps
        step_gap = max(1, n_steps // n_recorded)
        self.recorded_steps = [i * step_gap for i in range(n_recorded)]

        # x_final for estimate_ell0 fallback
        self.x_final = self.x_before_clone[-1]

    def get_step_index(self, step: int) -> int:
        return self.recorded_steps.index(step)


# =========================================================================
# TestResolve3dDims
# =========================================================================


class TestResolve3dDims:
    """Tests for _resolve_3d_dims(total_dims, dims, name)."""

    def test_none_with_enough_dims_returns_default(self):
        result = _resolve_3d_dims(5, None, "test")
        assert result == (0, 1, 2)

    def test_none_with_exactly_3_dims(self):
        result = _resolve_3d_dims(3, None, "test")
        assert result == (0, 1, 2)

    def test_none_with_too_few_dims_raises(self):
        with pytest.raises(ValueError, match="requires at least 3 dimensions"):
            _resolve_3d_dims(2, None, "test")

    def test_explicit_valid_dims(self):
        result = _resolve_3d_dims(10, (3, 5, 7), "test")
        assert result == (3, 5, 7)

    def test_explicit_dims_boundary(self):
        result = _resolve_3d_dims(5, (0, 2, 4), "test")
        assert result == (0, 2, 4)

    def test_non_unique_dims_raises(self):
        with pytest.raises(ValueError, match="must be unique"):
            _resolve_3d_dims(5, (1, 1, 2), "color_dims")

    def test_out_of_range_dims_raises(self):
        with pytest.raises(ValueError, match="invalid indices"):
            _resolve_3d_dims(3, (0, 1, 5), "color_dims")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 3 indices"):
            _resolve_3d_dims(5, (0, 1), "color_dims")

    def test_negative_dim_raises(self):
        with pytest.raises(ValueError, match="invalid indices"):
            _resolve_3d_dims(5, (-1, 0, 1), "test")


# =========================================================================
# TestResolveFrameIndices
# =========================================================================


class TestResolveFrameIndices:
    """Tests for _resolve_frame_indices(history, warmup_fraction, end_fraction)."""

    def test_basic_warmup_and_end(self):
        history = _SimpleHistoryStub(n_recorded=100)
        indices = _resolve_frame_indices(history, 0.1, 1.0)
        # start_idx = max(1, int(100 * 0.1)) = 10
        # end_idx = int(100 * 1.0) = 100
        assert indices[0] == 10
        assert indices[-1] == 99
        assert len(indices) == 90

    def test_zero_warmup(self):
        history = _SimpleHistoryStub(n_recorded=50)
        indices = _resolve_frame_indices(history, 0.0, 1.0)
        # start_idx = max(1, 0) = 1
        assert indices[0] == 1
        assert indices[-1] == 49

    def test_n_recorded_less_than_2_returns_empty(self):
        history = _SimpleHistoryStub(n_recorded=1)
        indices = _resolve_frame_indices(history, 0.1, 1.0)
        assert indices == []

    def test_n_recorded_zero_returns_empty(self):
        history = _SimpleHistoryStub(n_recorded=0)
        indices = _resolve_frame_indices(history, 0.1, 1.0)
        assert indices == []

    def test_partial_end_fraction(self):
        history = _SimpleHistoryStub(n_recorded=100)
        indices = _resolve_frame_indices(history, 0.0, 0.5)
        # start_idx = 1, end_idx = max(2, int(100*0.5)) = 50
        assert indices[0] == 1
        assert indices[-1] == 49


# =========================================================================
# TestPrepareChannelData
# =========================================================================


class TestPrepareChannelData:
    """Tests for prepare_channel_data using MockRunHistory."""

    @pytest.fixture
    def history(self) -> MockRunHistory:
        return MockRunHistory(N=20, d=5, n_steps=100, n_recorded=50)

    @pytest.fixture
    def base_config(self) -> ChannelConfigBase:
        return ChannelConfigBase(
            warmup_fraction=0.1,
            end_fraction=1.0,
            ell0=1.0,  # explicit to avoid estimate_ell0 edge cases
        )

    def test_basic_returns_prepared_data(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        assert isinstance(data, PreparedChannelData)

    def test_color_shape_complex(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        T = len(data.frame_indices)
        N = history.N
        assert data.color.shape == (T, N, 3)
        assert data.color.is_complex()

    def test_color_valid_shape_bool(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        T = len(data.frame_indices)
        N = history.N
        assert data.color_valid.shape == (T, N)
        assert data.color_valid.dtype == torch.bool

    def test_companions_shapes_long(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        T = len(data.frame_indices)
        N = history.N
        assert data.companions_distance.shape == (T, N)
        assert data.companions_distance.dtype == torch.long
        assert data.companions_clone.shape == (T, N)
        assert data.companions_clone.dtype == torch.long

    def test_need_positions(self, history, base_config):
        data = prepare_channel_data(history, base_config, need_positions=True)
        T = len(data.frame_indices)
        N = history.N
        assert data.positions is not None
        assert data.positions.shape == (T, N, 3)
        assert data.positions.dtype == torch.float32

    def test_need_scores(self, history, base_config):
        data = prepare_channel_data(history, base_config, need_scores=True)
        T = len(data.frame_indices)
        N = history.N
        assert data.scores is not None
        assert data.scores.shape == (T, N)
        assert data.scores.dtype == torch.float32

    def test_no_positions_by_default(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        assert data.positions is None

    def test_no_scores_by_default(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        assert data.scores is None

    def test_empty_frames_returns_empty_tensors(self):
        """History with n_recorded < 2 produces empty PreparedChannelData."""
        history = MockRunHistory(N=10, d=5, n_steps=1, n_recorded=1)
        config = ChannelConfigBase(ell0=1.0)
        data = prepare_channel_data(history, config)
        assert data.frame_indices == []
        assert data.color.shape[0] == 0
        assert data.color_valid.shape[0] == 0

    def test_custom_color_dims(self):
        history = MockRunHistory(N=15, d=6, n_steps=100, n_recorded=50)
        config = ChannelConfigBase(ell0=1.0, color_dims=(1, 3, 5))
        data = prepare_channel_data(history, config)
        T = len(data.frame_indices)
        assert data.color.shape == (T, 15, 3)

    def test_frame_indices_populated(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        assert len(data.frame_indices) > 0
        assert all(isinstance(i, int) for i in data.frame_indices)

    def test_device_is_cpu(self, history, base_config):
        data = prepare_channel_data(history, base_config)
        assert data.device == torch.device("cpu")


# =========================================================================
# TestPipelineConfig
# =========================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig defaults and channel selection."""

    def test_default_has_all_sub_configs(self):
        cfg = PipelineConfig()
        assert cfg.base is not None
        assert cfg.meson is not None
        assert cfg.vector is not None
        assert cfg.baryon is not None
        assert cfg.glueball is not None
        assert cfg.tensor is not None
        assert cfg.correlator is not None
        assert cfg.multiscale is not None

    def test_channels_none_means_all(self):
        cfg = PipelineConfig()
        assert cfg.channels is None

    def test_channels_selective(self):
        cfg = PipelineConfig(channels=["meson"])
        assert cfg.channels == ["meson"]

    def test_default_correlator_max_lag(self):
        cfg = PipelineConfig()
        assert cfg.correlator.max_lag == 80


# =========================================================================
# TestComputeStrongForcePipeline
# =========================================================================


class TestComputeStrongForcePipeline:
    """End-to-end integration tests for compute_strong_force_pipeline."""

    @pytest.fixture
    def history(self) -> MockRunHistory:
        return MockRunHistory(N=20, d=5, n_steps=100, n_recorded=50)

    @pytest.fixture
    def default_config(self) -> PipelineConfig:
        return PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
        )

    def test_full_pipeline_returns_result(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        assert isinstance(result, PipelineResult)

    def test_operators_dict_populated(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        assert isinstance(result.operators, dict)
        assert len(result.operators) > 0

    def test_correlators_dict_populated(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        assert isinstance(result.correlators, dict)
        assert len(result.correlators) > 0

    def test_operator_keys_match_expected_channels(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        expected_keys = {
            "scalar",
            "pseudoscalar",
            "vector",
            "axial_vector",
            "nucleon",
            "glueball",
            "tensor",
        }
        assert expected_keys.issubset(set(result.operators.keys()))

    def test_correlator_keys_match_operator_keys(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        assert set(result.correlators.keys()) == set(result.operators.keys())

    def test_scalar_operator_shape(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        T = len(result.prepared_data.frame_indices)
        assert result.operators["scalar"].shape == (T,)

    def test_vector_operator_shape(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        T = len(result.prepared_data.frame_indices)
        assert result.operators["vector"].shape == (T, 3)

    def test_nucleon_operator_shape(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        T = len(result.prepared_data.frame_indices)
        assert result.operators["nucleon"].shape == (T,)

    def test_glueball_operator_shape(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        T = len(result.prepared_data.frame_indices)
        assert result.operators["glueball"].shape == (T,)

    def test_tensor_operator_shape(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        T = len(result.prepared_data.frame_indices)
        assert result.operators["tensor"].shape == (T,)

    def test_correlator_shape(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        max_lag = default_config.correlator.max_lag
        assert result.correlators["scalar"].shape == (max_lag + 1,)

    def test_selective_channels_meson_only(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
            channels=["meson"],
        )
        result = compute_strong_force_pipeline(history, config)
        # meson produces scalar and pseudoscalar
        assert "scalar" in result.operators
        assert "pseudoscalar" in result.operators
        # Other channels should not be present
        assert "vector" not in result.operators
        assert "nucleon" not in result.operators
        assert "glueball" not in result.operators
        assert "tensor" not in result.operators

    def test_config_none_uses_defaults(self, history):
        result = compute_strong_force_pipeline(history, config=None)
        assert isinstance(result, PipelineResult)
        assert len(result.operators) > 0

    def test_prepared_data_is_populated(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        assert result.prepared_data is not None
        assert isinstance(result.prepared_data, PreparedChannelData)
        assert len(result.prepared_data.frame_indices) > 0

    def test_selective_channels_baryon_only(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=5),
            channels=["baryon"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "nucleon" in result.operators
        assert "scalar" not in result.operators

    def test_selective_channels_vector_only(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=5),
            channels=["vector"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "vector" in result.operators
        assert "axial_vector" in result.operators
        assert "scalar" not in result.operators

    def test_selective_channels_glueball_only(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=5),
            channels=["glueball"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "glueball" in result.operators
        assert "scalar" not in result.operators

    def test_selective_channels_tensor_only(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=5),
            channels=["tensor"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "tensor" in result.operators
        assert "scalar" not in result.operators

    def test_operators_are_finite(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        for name, op in result.operators.items():
            assert torch.isfinite(op).all(), f"Operator '{name}' contains non-finite values"

    def test_correlators_are_finite(self, history, default_config):
        result = compute_strong_force_pipeline(history, default_config)
        for name, corr in result.correlators.items():
            assert torch.isfinite(corr).all(), f"Correlator '{name}' contains non-finite values"
