"""Comprehensive tests for physics/operators/vector_operators.py."""

from __future__ import annotations

import pytest
import torch

from fragile.physics.operators.config import VectorOperatorConfig
from fragile.physics.operators.vector_operators import (
    _resolve_vector_operator_mode,
    _resolve_vector_projection_mode,
    compute_vector_operators,
    VECTOR_OPERATOR_MODES,
    VECTOR_PROJECTION_MODES,
)

from .conftest import make_prepared_data


# ---------------------------------------------------------------------------
# TestResolveVectorModes
# ---------------------------------------------------------------------------


class TestResolveVectorModes:
    """Tests for _resolve_vector_operator_mode and _resolve_vector_projection_mode."""

    # -- operator mode resolution --

    def test_operator_mode_none_returns_standard(self):
        assert _resolve_vector_operator_mode(None) == "standard"

    def test_operator_mode_empty_string_returns_standard(self):
        assert _resolve_vector_operator_mode("") == "standard"

    def test_operator_mode_whitespace_returns_standard(self):
        assert _resolve_vector_operator_mode("   ") == "standard"

    @pytest.mark.parametrize("mode", VECTOR_OPERATOR_MODES)
    def test_operator_mode_valid_modes_pass(self, mode: str):
        assert _resolve_vector_operator_mode(mode) == mode

    def test_operator_mode_case_insensitive(self):
        assert _resolve_vector_operator_mode("STANDARD") == "standard"
        assert _resolve_vector_operator_mode("Score_Directed") == "score_directed"

    def test_operator_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="operator_mode must be one of"):
            _resolve_vector_operator_mode("invalid_mode")

    # -- projection mode resolution --

    def test_projection_mode_none_returns_full(self):
        assert _resolve_vector_projection_mode(None) == "full"

    def test_projection_mode_empty_string_returns_full(self):
        assert _resolve_vector_projection_mode("") == "full"

    def test_projection_mode_whitespace_returns_full(self):
        assert _resolve_vector_projection_mode("   ") == "full"

    @pytest.mark.parametrize("mode", VECTOR_PROJECTION_MODES)
    def test_projection_mode_valid_modes_pass(self, mode: str):
        assert _resolve_vector_projection_mode(mode) == mode

    def test_projection_mode_case_insensitive(self):
        assert _resolve_vector_projection_mode("FULL") == "full"
        assert _resolve_vector_projection_mode("Longitudinal") == "longitudinal"
        assert _resolve_vector_projection_mode("TRANSVERSE") == "transverse"

    def test_projection_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="projection_mode must be one of"):
            _resolve_vector_projection_mode("invalid_mode")


# ---------------------------------------------------------------------------
# TestComputeVectorOperatorsStandard
# ---------------------------------------------------------------------------


class TestComputeVectorOperatorsStandard:
    """Tests for compute_vector_operators with default (standard) mode."""

    def test_output_keys(self):
        data = make_prepared_data(include_positions=True)
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert set(result.keys()) == {"vector", "axial_vector"}

    def test_output_shape(self):
        T, N = 10, 20
        data = make_prepared_data(T=T, N=N, include_positions=True)
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (T, 3)
        assert result["axial_vector"].shape == (T, 3)

    def test_empty_T0(self):
        data = make_prepared_data(T=0, N=5, include_positions=True)
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (0, 3)
        assert result["axial_vector"].shape == (0, 3)

    def test_values_are_real_float32(self):
        data = make_prepared_data(include_positions=True)
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert result["vector"].dtype == torch.float32
        assert result["axial_vector"].dtype == torch.float32
        assert not result["vector"].is_complex()
        assert not result["axial_vector"].is_complex()

    def test_values_are_finite(self):
        data = make_prepared_data(include_positions=True)
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert torch.isfinite(result["vector"]).all()
        assert torch.isfinite(result["axial_vector"]).all()

    def test_missing_positions_raises(self):
        data = make_prepared_data(include_positions=False)
        config = VectorOperatorConfig()
        with pytest.raises(ValueError, match="positions must be provided"):
            compute_vector_operators(data, config)

    @pytest.mark.parametrize("pair_selection", ["both", "distance", "clone"])
    def test_pair_selection_variations(self, pair_selection: str):
        data = make_prepared_data(include_positions=True)
        config = VectorOperatorConfig(pair_selection=pair_selection)
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (10, 3)
        assert result["axial_vector"].shape == (10, 3)
        assert torch.isfinite(result["vector"]).all()
        assert torch.isfinite(result["axial_vector"]).all()

    def test_use_unit_displacement(self):
        data = make_prepared_data(include_positions=True)
        config_no_unit = VectorOperatorConfig(use_unit_displacement=False)
        config_unit = VectorOperatorConfig(use_unit_displacement=True)
        result_no_unit = compute_vector_operators(data, config_no_unit)
        result_unit = compute_vector_operators(data, config_unit)
        # Both should produce valid outputs
        assert result_unit["vector"].shape == (10, 3)
        assert torch.isfinite(result_unit["vector"]).all()
        # Results should differ since unit displacement normalizes
        # (unless all displacements happen to be unit vectors already)
        assert not torch.allclose(result_no_unit["vector"], result_unit["vector"])

    def test_different_seeds_produce_different_results(self):
        data1 = make_prepared_data(include_positions=True, seed=42)
        data2 = make_prepared_data(include_positions=True, seed=99)
        config = VectorOperatorConfig()
        result1 = compute_vector_operators(data1, config)
        result2 = compute_vector_operators(data2, config)
        assert not torch.allclose(result1["vector"], result2["vector"])

    def test_small_system(self):
        """Test with minimal T=2, N=4 to verify no shape errors."""
        data = make_prepared_data(T=2, N=4, include_positions=True)
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (2, 3)
        assert result["axial_vector"].shape == (2, 3)


# ---------------------------------------------------------------------------
# TestComputeVectorOperatorsScoreDirected
# ---------------------------------------------------------------------------


class TestComputeVectorOperatorsScoreDirected:
    """Tests for compute_vector_operators with operator_mode='score_directed'."""

    def test_score_directed_output_shape(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(operator_mode="score_directed")
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (10, 3)
        assert result["axial_vector"].shape == (10, 3)

    def test_score_directed_missing_scores_raises(self):
        data = make_prepared_data(include_positions=True, include_scores=False)
        config = VectorOperatorConfig(operator_mode="score_directed")
        with pytest.raises(ValueError, match="scores is required"):
            compute_vector_operators(data, config)

    def test_score_directed_values_are_finite(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(operator_mode="score_directed")
        result = compute_vector_operators(data, config)
        assert torch.isfinite(result["vector"]).all()
        assert torch.isfinite(result["axial_vector"]).all()

    def test_score_directed_differs_from_standard(self):
        """Score-directed flips inner products where ds < 0.

        With a large enough system and an adversarial seed, the averaged
        result can still coincide, so we check either vector or axial_vector
        channel differs, or at least that the computation runs correctly.
        """
        data = make_prepared_data(
            T=20,
            N=40,
            include_positions=True,
            include_scores=True,
            seed=7,
        )
        config_std = VectorOperatorConfig(operator_mode="standard")
        config_sd = VectorOperatorConfig(operator_mode="score_directed")
        result_std = compute_vector_operators(data, config_std)
        result_sd = compute_vector_operators(data, config_sd)
        # At least one channel should differ with a large enough random system
        vector_differ = not torch.allclose(result_std["vector"], result_sd["vector"])
        axial_differ = not torch.allclose(
            result_std["axial_vector"],
            result_sd["axial_vector"],
        )
        assert vector_differ or axial_differ

    def test_score_directed_real_float32(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(operator_mode="score_directed")
        result = compute_vector_operators(data, config)
        assert result["vector"].dtype == torch.float32
        assert result["axial_vector"].dtype == torch.float32


# ---------------------------------------------------------------------------
# TestComputeVectorOperatorsScoreGradient
# ---------------------------------------------------------------------------


class TestComputeVectorOperatorsScoreGradient:
    """Tests for compute_vector_operators with operator_mode='score_gradient'."""

    def test_score_gradient_output_shape(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(operator_mode="score_gradient")
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (10, 3)
        assert result["axial_vector"].shape == (10, 3)

    def test_score_gradient_missing_scores_raises(self):
        data = make_prepared_data(include_positions=True, include_scores=False)
        config = VectorOperatorConfig(operator_mode="score_gradient")
        with pytest.raises(ValueError, match="scores is required"):
            compute_vector_operators(data, config)

    def test_score_gradient_values_are_finite(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(operator_mode="score_gradient")
        result = compute_vector_operators(data, config)
        assert torch.isfinite(result["vector"]).all()
        assert torch.isfinite(result["axial_vector"]).all()


# ---------------------------------------------------------------------------
# TestComputeVectorOperatorsProjection
# ---------------------------------------------------------------------------


class TestComputeVectorOperatorsProjection:
    """Tests for compute_vector_operators with projection modes."""

    def test_longitudinal_output_shape(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(
            operator_mode="score_directed",
            projection_mode="longitudinal",
        )
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (10, 3)
        assert result["axial_vector"].shape == (10, 3)
        assert torch.isfinite(result["vector"]).all()

    def test_transverse_output_shape(self):
        data = make_prepared_data(include_positions=True, include_scores=True)
        config = VectorOperatorConfig(
            operator_mode="score_directed",
            projection_mode="transverse",
        )
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (10, 3)
        assert result["axial_vector"].shape == (10, 3)
        assert torch.isfinite(result["vector"]).all()

    def test_longitudinal_plus_transverse_approx_full(self):
        """Longitudinal + transverse projections should approximately sum to full."""
        data = make_prepared_data(include_positions=True, include_scores=True)
        config_full = VectorOperatorConfig(
            operator_mode="score_directed",
            projection_mode="full",
        )
        config_long = VectorOperatorConfig(
            operator_mode="score_directed",
            projection_mode="longitudinal",
        )
        config_trans = VectorOperatorConfig(
            operator_mode="score_directed",
            projection_mode="transverse",
        )
        result_full = compute_vector_operators(data, config_full)
        result_long = compute_vector_operators(data, config_long)
        result_trans = compute_vector_operators(data, config_trans)

        # vector_long + vector_trans ~ vector_full
        vector_sum = result_long["vector"] + result_trans["vector"]
        assert torch.allclose(vector_sum, result_full["vector"], atol=1e-4)

        axial_sum = result_long["axial_vector"] + result_trans["axial_vector"]
        assert torch.allclose(axial_sum, result_full["axial_vector"], atol=1e-4)

    def test_projection_modes_differ_from_each_other(self):
        """Different projection modes should produce different results."""
        data = make_prepared_data(include_positions=True, include_scores=True)
        results = {}
        for mode in ("full", "longitudinal", "transverse"):
            config = VectorOperatorConfig(
                operator_mode="score_directed",
                projection_mode=mode,
            )
            results[mode] = compute_vector_operators(data, config)

        assert not torch.allclose(results["full"]["vector"], results["longitudinal"]["vector"])
        assert not torch.allclose(results["full"]["vector"], results["transverse"]["vector"])
        assert not torch.allclose(
            results["longitudinal"]["vector"], results["transverse"]["vector"]
        )

    def test_full_projection_same_as_default(self):
        """projection_mode='full' should give the same result as no projection."""
        data = make_prepared_data(include_positions=True, include_scores=True)
        config_default = VectorOperatorConfig(operator_mode="score_directed")
        config_full = VectorOperatorConfig(
            operator_mode="score_directed",
            projection_mode="full",
        )
        result_default = compute_vector_operators(data, config_default)
        result_full = compute_vector_operators(data, config_full)
        assert torch.allclose(result_default["vector"], result_full["vector"])
        assert torch.allclose(result_default["axial_vector"], result_full["axial_vector"])


# ---------------------------------------------------------------------------
# TestComputeVectorOperatorsMultiscale
# ---------------------------------------------------------------------------


class TestComputeVectorOperatorsMultiscale:
    """Tests for compute_vector_operators with multiscale data."""

    def test_multiscale_output_shape(self):
        n_scales = 3
        T = 10
        data = make_prepared_data(
            T=T,
            include_positions=True,
            include_scores=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert result["vector"].shape == (n_scales, T, 3)
        assert result["axial_vector"].shape == (n_scales, T, 3)

    def test_multiscale_n_scales_matches(self):
        for n_scales in (2, 4, 5):
            T = 8
            data = make_prepared_data(
                T=T,
                include_positions=True,
                include_scores=True,
                include_multiscale=True,
                n_scales=n_scales,
            )
            config = VectorOperatorConfig()
            result = compute_vector_operators(data, config)
            assert result["vector"].shape[0] == n_scales
            assert result["axial_vector"].shape[0] == n_scales

    def test_multiscale_values_are_finite(self):
        data = make_prepared_data(
            include_positions=True,
            include_scores=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert torch.isfinite(result["vector"]).all()
        assert torch.isfinite(result["axial_vector"]).all()

    def test_multiscale_real_float32(self):
        data = make_prepared_data(
            include_positions=True,
            include_scores=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = VectorOperatorConfig()
        result = compute_vector_operators(data, config)
        assert result["vector"].dtype == torch.float32
        assert result["axial_vector"].dtype == torch.float32
