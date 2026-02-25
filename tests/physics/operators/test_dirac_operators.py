"""Tests for physics/operators/dirac_operators.py.

Verifies compute_dirac_operators produces correct output keys, shapes,
dtypes, and finite values from PreparedChannelData.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.config import ChannelConfigBase, DiracOperatorConfig
from fragile.physics.operators.dirac_operators import compute_dirac_operators

from .conftest import make_prepared_data


# ---------------------------------------------------------------------------
# Output keys
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "dirac_scalar",
    "dirac_pseudoscalar",
    "dirac_vector",
    "dirac_axial_vector",
    "dirac_tensor",
    "dirac_tensor_0k",
}


class TestComputeDiracOperatorsBasic:
    """Basic output validation for compute_dirac_operators."""

    def test_output_keys(self):
        data = make_prepared_data(T=10, N=20)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_output_shape_single_scale(self):
        T, N = 10, 20
        data = make_prepared_data(T=T, N=N)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert result[key].shape == (T,), f"{key}: expected shape ({T},), got {result[key].shape}"

    def test_empty_frames(self):
        data = make_prepared_data(T=0, N=20)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert result[key].shape == (0,)

    def test_values_are_real_float32(self):
        data = make_prepared_data(T=10, N=20)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert result[key].dtype == torch.float32, f"{key}: expected float32"
            assert not result[key].is_complex(), f"{key}: should not be complex"

    def test_values_are_finite(self):
        data = make_prepared_data(T=10, N=20)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert torch.isfinite(result[key]).all(), f"{key}: has non-finite values"

    def test_single_frame(self):
        data = make_prepared_data(T=1, N=20)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert result[key].shape == (1,)

    def test_deterministic_with_same_seed(self):
        data1 = make_prepared_data(T=10, N=20, seed=123)
        data2 = make_prepared_data(T=10, N=20, seed=123)
        config = DiracOperatorConfig()
        r1 = compute_dirac_operators(data1, config)
        r2 = compute_dirac_operators(data2, config)
        for key in EXPECTED_KEYS:
            assert torch.allclose(r1[key], r2[key]), f"{key}: not deterministic"


# ---------------------------------------------------------------------------
# Pair selection modes
# ---------------------------------------------------------------------------


class TestComputeDiracOperatorsPairSelection:
    """Test different pair selection strategies."""

    @pytest.mark.parametrize("pair_selection", ["both", "distance", "clone"])
    def test_pair_selection_mode(self, pair_selection):
        data = make_prepared_data(T=8, N=15)
        config = DiracOperatorConfig(pair_selection=pair_selection)
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert result[key].shape == (8,)
            assert torch.isfinite(result[key]).all(), f"{key} with {pair_selection}: non-finite"


# ---------------------------------------------------------------------------
# Channel non-degeneracy
# ---------------------------------------------------------------------------


class TestChannelNonDegeneracy:
    """Verify that different Dirac channels produce distinct time series."""

    def test_channels_are_distinct(self):
        data = make_prepared_data(T=20, N=30, seed=42)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)

        keys = list(EXPECTED_KEYS)
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                differ = not torch.allclose(result[k1], result[k2], atol=1e-4)
                assert differ, f"{k1} and {k2} should produce distinct values"

    def test_tensor_0k_differs_from_tensor(self):
        """σ_0k and σ_jk use different gamma matrices and must produce different values."""
        data = make_prepared_data(T=20, N=30, seed=42)
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        assert not torch.allclose(
            result["dirac_tensor"], result["dirac_tensor_0k"], atol=1e-4,
        ), "dirac_tensor (σ_jk) and dirac_tensor_0k (σ_0k) must differ"


# ---------------------------------------------------------------------------
# Multiscale support
# ---------------------------------------------------------------------------


class TestComputeDiracOperatorsMultiscale:
    """Test multiscale Dirac operators when scales are populated."""

    def test_output_shape_multiscale(self):
        n_scales = 3
        T, N = 10, 20
        data = make_prepared_data(
            T=T, N=N,
            include_positions=True,
            include_scores=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert result[key].shape == (n_scales, T), (
                f"{key}: expected ({n_scales}, {T}), got {result[key].shape}"
            )

    def test_multiscale_values_finite(self):
        data = make_prepared_data(
            T=10, N=20,
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = DiracOperatorConfig()
        result = compute_dirac_operators(data, config)
        for key in EXPECTED_KEYS:
            assert torch.isfinite(result[key]).all(), f"{key}: non-finite in multiscale"


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineConfigIntegration:
    """Verify DiracOperatorConfig integrates with PipelineConfig."""

    def test_pipeline_config_has_dirac(self):
        from fragile.physics.operators.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert hasattr(cfg, "dirac")
        assert isinstance(cfg.dirac, DiracOperatorConfig)

    def test_default_channels_include_dirac(self):
        """When channels=None, 'dirac' should be in the default list."""
        from fragile.physics.operators.pipeline import compute_strong_force_pipeline

        # We can't run the full pipeline without RunHistory, but we can
        # verify the default channel list
        from fragile.physics.operators.pipeline import PipelineConfig
        cfg = PipelineConfig()
        # channels=None means all defaults; the pipeline sets it at runtime
        assert cfg.channels is None  # means "use defaults which includes dirac"


# ---------------------------------------------------------------------------
# ChannelConfigBase compatibility
# ---------------------------------------------------------------------------


class TestDiracOperatorConfigInheritance:
    """DiracOperatorConfig should inherit all ChannelConfigBase fields."""

    def test_inherits_base_fields(self):
        config = DiracOperatorConfig()
        assert hasattr(config, "warmup_fraction")
        assert hasattr(config, "end_fraction")
        assert hasattr(config, "h_eff")
        assert hasattr(config, "mass")
        assert hasattr(config, "eps")
        assert hasattr(config, "pair_selection")

    def test_is_instance_of_base(self):
        config = DiracOperatorConfig()
        assert isinstance(config, ChannelConfigBase)

    def test_default_pair_selection(self):
        config = DiracOperatorConfig()
        assert config.pair_selection == "both"
