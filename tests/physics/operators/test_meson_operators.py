"""Comprehensive tests for physics/operators/meson_operators.py."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.config import MesonOperatorConfig
from fragile.physics.operators.meson_operators import (
    _compute_inner_products_for_pairs,
    _orient_inner_products_by_scores,
    _per_frame_series,
    _resolve_meson_operator_mode,
    _weight_inner_products_by_score_gap,
    compute_meson_operators,
)
from fragile.physics.operators.preparation import PreparedChannelData

from .conftest import make_prepared_data


# ---------------------------------------------------------------------------
# TestResolveMesonOperatorMode
# ---------------------------------------------------------------------------


class TestResolveMesonOperatorMode:
    """Tests for _resolve_meson_operator_mode normalization."""

    def test_none_returns_standard(self):
        assert _resolve_meson_operator_mode(None) == "standard"

    def test_empty_string_returns_standard(self):
        assert _resolve_meson_operator_mode("") == "standard"

    def test_whitespace_only_returns_standard(self):
        assert _resolve_meson_operator_mode("   ") == "standard"

    @pytest.mark.parametrize(
        "mode",
        ["standard", "score_directed", "score_weighted", "abs2_vacsub"],
    )
    def test_valid_modes_pass_through(self, mode: str):
        assert _resolve_meson_operator_mode(mode) == mode

    @pytest.mark.parametrize(
        "mode",
        ["STANDARD", "Score_Directed", "SCORE_WEIGHTED", "ABS2_VACSUB"],
    )
    def test_case_insensitive(self, mode: str):
        result = _resolve_meson_operator_mode(mode)
        assert result == mode.strip().lower()

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="operator_mode must be one of"):
            _resolve_meson_operator_mode("nonexistent_mode")

    def test_another_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _resolve_meson_operator_mode("connected")


# ---------------------------------------------------------------------------
# TestComputeInnerProductsForPairs
# ---------------------------------------------------------------------------


class TestComputeInnerProductsForPairs:
    """Tests for _compute_inner_products_for_pairs shape validation and logic."""

    def test_basic_shape(self):
        T, N, P = 5, 10, 2
        color = torch.randn(T, N, 3, dtype=torch.cfloat)
        color_valid = torch.ones(T, N, dtype=torch.bool)
        pair_indices = torch.randint(0, N, (T, N, P))
        structural_valid = torch.ones(T, N, P, dtype=torch.bool)

        inner, valid = _compute_inner_products_for_pairs(
            color,
            color_valid,
            pair_indices,
            structural_valid,
            eps=0.0,
        )
        assert inner.shape == (T, N, P)
        assert valid.shape == (T, N, P)
        assert inner.is_complex()
        assert valid.dtype == torch.bool

    def test_color_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="color must have shape"):
            _compute_inner_products_for_pairs(
                color=torch.randn(5, 10, dtype=torch.cfloat),
                color_valid=torch.ones(5, 10, dtype=torch.bool),
                pair_indices=torch.zeros(5, 10, 2, dtype=torch.long),
                structural_valid=torch.ones(5, 10, 2, dtype=torch.bool),
                eps=0.0,
            )

    def test_color_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="color must have shape"):
            _compute_inner_products_for_pairs(
                color=torch.randn(5, 10, 4, dtype=torch.cfloat),
                color_valid=torch.ones(5, 10, dtype=torch.bool),
                pair_indices=torch.zeros(5, 10, 2, dtype=torch.long),
                structural_valid=torch.ones(5, 10, 2, dtype=torch.bool),
                eps=0.0,
            )

    def test_color_valid_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="color_valid must have shape"):
            _compute_inner_products_for_pairs(
                color=torch.randn(5, 10, 3, dtype=torch.cfloat),
                color_valid=torch.ones(5, 8, dtype=torch.bool),
                pair_indices=torch.zeros(5, 10, 2, dtype=torch.long),
                structural_valid=torch.ones(5, 10, 2, dtype=torch.bool),
                eps=0.0,
            )

    def test_pair_indices_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="pair_indices must have shape"):
            _compute_inner_products_for_pairs(
                color=torch.randn(5, 10, 3, dtype=torch.cfloat),
                color_valid=torch.ones(5, 10, dtype=torch.bool),
                pair_indices=torch.zeros(4, 10, 2, dtype=torch.long),
                structural_valid=torch.ones(4, 10, 2, dtype=torch.bool),
                eps=0.0,
            )

    def test_structural_valid_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="structural_valid must have the same shape"):
            _compute_inner_products_for_pairs(
                color=torch.randn(5, 10, 3, dtype=torch.cfloat),
                color_valid=torch.ones(5, 10, dtype=torch.bool),
                pair_indices=torch.zeros(5, 10, 2, dtype=torch.long),
                structural_valid=torch.ones(5, 10, 3, dtype=torch.bool),
                eps=0.0,
            )

    def test_invalid_walkers_masked_out(self):
        T, N, P = 3, 5, 1
        color = torch.randn(T, N, 3, dtype=torch.cfloat)
        color_valid = torch.zeros(T, N, dtype=torch.bool)  # all invalid
        pair_indices = torch.zeros(T, N, P, dtype=torch.long)
        structural_valid = torch.ones(T, N, P, dtype=torch.bool)

        inner, valid = _compute_inner_products_for_pairs(
            color,
            color_valid,
            pair_indices,
            structural_valid,
            eps=0.0,
        )
        # No valid pairs when all walkers are invalid
        assert not valid.any()
        # Masked entries should be zero
        assert (inner == 0).all()

    def test_eps_filters_small_inner_products(self):
        T, N, P = 2, 4, 1
        # Make nearly-orthogonal color states so inner products are small
        color = torch.zeros(T, N, 3, dtype=torch.cfloat)
        color[:, 0, 0] = 1.0
        color[:, 1, 1] = 1.0  # orthogonal to walker 0
        color[:, 2, 0] = 1.0
        color[:, 3, 0] = 1.0
        color_valid = torch.ones(T, N, dtype=torch.bool)
        # Pair walker 0 with walker 1 (orthogonal -> inner = 0)
        pair_indices = torch.ones(T, N, P, dtype=torch.long)
        pair_indices[:, 0, 0] = 1  # pair 0 with 1
        structural_valid = torch.ones(T, N, P, dtype=torch.bool)

        _, valid_strict = _compute_inner_products_for_pairs(
            color,
            color_valid,
            pair_indices,
            structural_valid,
            eps=0.5,
        )
        # Walker 0 paired with walker 1: inner product = 0, should be filtered by eps=0.5
        assert not valid_strict[:, 0, 0].any()


# ---------------------------------------------------------------------------
# TestOrientInnerProductsByScores
# ---------------------------------------------------------------------------


class TestOrientInnerProductsByScores:
    """Tests for _orient_inner_products_by_scores."""

    def test_basic_orientation(self):
        T, N, P = 3, 5, 2
        inner = torch.randn(T, N, P, dtype=torch.cfloat)
        valid = torch.ones(T, N, P, dtype=torch.bool)
        scores = torch.randn(T, N)
        pair_indices = torch.randint(0, N, (T, N, P))

        out, out_valid = _orient_inner_products_by_scores(
            inner=inner,
            valid=valid,
            scores=scores,
            pair_indices=pair_indices,
        )
        assert out.shape == (T, N, P)
        assert out_valid.shape == (T, N, P)
        assert out.is_complex()

    def test_non_finite_scores_invalidated(self):
        T, N, P = 2, 4, 1
        inner = torch.ones(T, N, P, dtype=torch.cfloat)
        valid = torch.ones(T, N, P, dtype=torch.bool)
        scores = torch.ones(T, N)
        scores[0, 0] = float("inf")
        pair_indices = torch.zeros(T, N, P, dtype=torch.long)
        pair_indices[:, :, 0] = 1

        _, out_valid = _orient_inner_products_by_scores(
            inner=inner,
            valid=valid,
            scores=scores,
            pair_indices=pair_indices,
        )
        # Walker 0 at t=0 has inf score, so that pair should be invalidated
        assert not out_valid[0, 0, 0].item()


# ---------------------------------------------------------------------------
# TestWeightInnerProductsByScoreGap
# ---------------------------------------------------------------------------


class TestWeightInnerProductsByScoreGap:
    """Tests for _weight_inner_products_by_score_gap."""

    def test_basic_weighting(self):
        T, N, P = 3, 5, 2
        inner = torch.ones(T, N, P, dtype=torch.cfloat)
        valid = torch.ones(T, N, P, dtype=torch.bool)
        scores = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(T, -1)
        pair_indices = torch.randint(0, N, (T, N, P))

        out, out_valid = _weight_inner_products_by_score_gap(
            inner=inner,
            valid=valid,
            scores=scores,
            pair_indices=pair_indices,
        )
        assert out.shape == (T, N, P)
        assert out_valid.shape == (T, N, P)

    def test_equal_scores_give_zero_weight(self):
        T, N, P = 2, 4, 1
        inner = torch.ones(T, N, P, dtype=torch.cfloat)
        valid = torch.ones(T, N, P, dtype=torch.bool)
        scores = torch.ones(T, N)  # all equal
        pair_indices = torch.zeros(T, N, P, dtype=torch.long)
        pair_indices[:, :, 0] = 1

        out, out_valid = _weight_inner_products_by_score_gap(
            inner=inner,
            valid=valid,
            scores=scores,
            pair_indices=pair_indices,
        )
        # Gap is zero everywhere, so weighted inner products should be zero
        assert torch.allclose(out[out_valid].abs(), torch.zeros(1))


# ---------------------------------------------------------------------------
# TestPerFrameSeries
# ---------------------------------------------------------------------------


class TestPerFrameSeries:
    """Tests for _per_frame_series averaging."""

    def test_basic_averaging(self):
        T, N, P = 4, 3, 2
        values = torch.ones(T, N, P)
        valid = torch.ones(T, N, P, dtype=torch.bool)

        series, counts = _per_frame_series(values, valid)
        assert series.shape == (T,)
        assert counts.shape == (T,)
        assert series.dtype == torch.float32
        # All ones averaged should give 1.0
        assert torch.allclose(series, torch.ones(T))

    def test_no_valid_gives_zero(self):
        T, N, P = 3, 4, 2
        values = torch.randn(T, N, P)
        valid = torch.zeros(T, N, P, dtype=torch.bool)

        series, counts = _per_frame_series(values, valid)
        assert (series == 0).all()
        assert (counts == 0).all()

    def test_partial_validity(self):
        T, N, P = 2, 3, 1
        values = torch.tensor([[[2.0], [4.0], [6.0]], [[1.0], [3.0], [5.0]]])
        valid = torch.ones(T, N, P, dtype=torch.bool)
        valid[0, 2, 0] = False  # mask out the 6.0

        series, counts = _per_frame_series(values, valid)
        # Frame 0: (2+4)/2 = 3.0; Frame 1: (1+3+5)/3 = 3.0
        assert counts[0].item() == 2
        assert counts[1].item() == 3
        assert torch.isclose(series[0], torch.tensor(3.0))
        assert torch.isclose(series[1], torch.tensor(3.0))


# ---------------------------------------------------------------------------
# TestComputeMesonOperatorsStandard
# ---------------------------------------------------------------------------


class TestComputeMesonOperatorsStandard:
    """Tests for compute_meson_operators with standard mode."""

    def test_output_keys(self):
        data = make_prepared_data(T=10, N=20)
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert set(result.keys()) == {"scalar", "pseudoscalar"}

    def test_output_shape_single_scale(self):
        T, N = 10, 20
        data = make_prepared_data(T=T, N=N)
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (T,)
        assert result["pseudoscalar"].shape == (T,)

    def test_empty_frames(self):
        data = make_prepared_data(T=0, N=20)
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (0,)
        assert result["pseudoscalar"].shape == (0,)

    def test_values_are_real_float32(self):
        data = make_prepared_data(T=10, N=20)
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert result["scalar"].dtype == torch.float32
        assert result["pseudoscalar"].dtype == torch.float32
        assert not result["scalar"].is_complex()
        assert not result["pseudoscalar"].is_complex()

    def test_values_are_finite(self):
        data = make_prepared_data(T=10, N=20)
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert torch.isfinite(result["scalar"]).all()
        assert torch.isfinite(result["pseudoscalar"]).all()

    def test_pair_selection_distance(self):
        data = make_prepared_data(T=8, N=15)
        config = MesonOperatorConfig(pair_selection="distance")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (8,)
        assert result["pseudoscalar"].shape == (8,)
        assert torch.isfinite(result["scalar"]).all()

    def test_pair_selection_clone(self):
        data = make_prepared_data(T=8, N=15)
        config = MesonOperatorConfig(pair_selection="clone")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (8,)
        assert result["pseudoscalar"].shape == (8,)
        assert torch.isfinite(result["scalar"]).all()

    def test_pair_selection_both(self):
        data = make_prepared_data(T=8, N=15)
        config = MesonOperatorConfig(pair_selection="both")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (8,)
        assert result["pseudoscalar"].shape == (8,)
        assert torch.isfinite(result["scalar"]).all()

    def test_single_frame(self):
        data = make_prepared_data(T=1, N=20)
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (1,)
        assert result["pseudoscalar"].shape == (1,)

    def test_deterministic_with_same_seed(self):
        data1 = make_prepared_data(T=10, N=20, seed=123)
        data2 = make_prepared_data(T=10, N=20, seed=123)
        config = MesonOperatorConfig()
        r1 = compute_meson_operators(data1, config)
        r2 = compute_meson_operators(data2, config)
        assert torch.allclose(r1["scalar"], r2["scalar"])
        assert torch.allclose(r1["pseudoscalar"], r2["pseudoscalar"])


# ---------------------------------------------------------------------------
# TestComputeMesonOperatorsScoreDirected
# ---------------------------------------------------------------------------


class TestComputeMesonOperatorsScoreDirected:
    """Tests for compute_meson_operators with score_directed mode."""

    def test_requires_scores(self):
        data = make_prepared_data(T=10, N=20, include_scores=False)
        config = MesonOperatorConfig(operator_mode="score_directed")
        with pytest.raises(ValueError, match="scores is required"):
            compute_meson_operators(data, config)

    def test_with_scores_succeeds(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config = MesonOperatorConfig(operator_mode="score_directed")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (10,)
        assert result["pseudoscalar"].shape == (10,)

    def test_output_is_real_float32(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config = MesonOperatorConfig(operator_mode="score_directed")
        result = compute_meson_operators(data, config)
        assert result["scalar"].dtype == torch.float32
        assert result["pseudoscalar"].dtype == torch.float32

    def test_output_is_finite(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config = MesonOperatorConfig(operator_mode="score_directed")
        result = compute_meson_operators(data, config)
        assert torch.isfinite(result["scalar"]).all()
        assert torch.isfinite(result["pseudoscalar"]).all()

    def test_differs_from_standard(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config_std = MesonOperatorConfig(operator_mode="standard")
        config_sd = MesonOperatorConfig(operator_mode="score_directed")
        r_std = compute_meson_operators(data, config_std)
        r_sd = compute_meson_operators(data, config_sd)
        # Score-directed orientation generally changes values
        # (unless all score differences are non-negative, which is unlikely)
        differs_scalar = not torch.allclose(r_std["scalar"], r_sd["scalar"])
        differs_pseudo = not torch.allclose(r_std["pseudoscalar"], r_sd["pseudoscalar"])
        assert differs_scalar or differs_pseudo


# ---------------------------------------------------------------------------
# TestComputeMesonOperatorsScoreWeighted
# ---------------------------------------------------------------------------


class TestComputeMesonOperatorsScoreWeighted:
    """Tests for compute_meson_operators with score_weighted mode."""

    def test_requires_scores(self):
        data = make_prepared_data(T=10, N=20, include_scores=False)
        config = MesonOperatorConfig(operator_mode="score_weighted")
        with pytest.raises(ValueError, match="scores is required"):
            compute_meson_operators(data, config)

    def test_with_scores_succeeds(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config = MesonOperatorConfig(operator_mode="score_weighted")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (10,)
        assert result["pseudoscalar"].shape == (10,)

    def test_output_is_real_float32(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config = MesonOperatorConfig(operator_mode="score_weighted")
        result = compute_meson_operators(data, config)
        assert result["scalar"].dtype == torch.float32
        assert result["pseudoscalar"].dtype == torch.float32

    def test_output_is_finite(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config = MesonOperatorConfig(operator_mode="score_weighted")
        result = compute_meson_operators(data, config)
        assert torch.isfinite(result["scalar"]).all()
        assert torch.isfinite(result["pseudoscalar"]).all()

    def test_differs_from_standard(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        config_std = MesonOperatorConfig(operator_mode="standard")
        config_sw = MesonOperatorConfig(operator_mode="score_weighted")
        r_std = compute_meson_operators(data, config_std)
        r_sw = compute_meson_operators(data, config_sw)
        differs_scalar = not torch.allclose(r_std["scalar"], r_sw["scalar"])
        differs_pseudo = not torch.allclose(r_std["pseudoscalar"], r_sw["pseudoscalar"])
        assert differs_scalar or differs_pseudo


# ---------------------------------------------------------------------------
# TestComputeMesonOperatorsAbs2
# ---------------------------------------------------------------------------


class TestComputeMesonOperatorsAbs2:
    """Tests for compute_meson_operators with abs2_vacsub mode."""

    def test_output_shape(self):
        data = make_prepared_data(T=10, N=20)
        config = MesonOperatorConfig(operator_mode="abs2_vacsub")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (10,)
        assert result["pseudoscalar"].shape == (10,)

    def test_scalar_non_negative(self):
        data = make_prepared_data(T=10, N=20)
        config = MesonOperatorConfig(operator_mode="abs2_vacsub")
        result = compute_meson_operators(data, config)
        # |z|^2 averaged per frame should be >= 0
        assert (result["scalar"] >= 0).all()

    def test_output_is_finite(self):
        data = make_prepared_data(T=10, N=20)
        config = MesonOperatorConfig(operator_mode="abs2_vacsub")
        result = compute_meson_operators(data, config)
        assert torch.isfinite(result["scalar"]).all()
        assert torch.isfinite(result["pseudoscalar"]).all()


# ---------------------------------------------------------------------------
# TestComputeMesonOperatorsMultiscale
# ---------------------------------------------------------------------------


class TestComputeMesonOperatorsMultiscale:
    """Tests for compute_meson_operators with multiscale fields populated."""

    def test_output_shape_multiscale(self):
        n_scales = 3
        T, N = 10, 20
        data = make_prepared_data(
            T=T,
            N=N,
            include_positions=True,
            include_scores=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (n_scales, T)
        assert result["pseudoscalar"].shape == (n_scales, T)

    def test_n_scales_matches(self):
        for n_scales in [2, 4, 5]:
            T, N = 8, 15
            data = make_prepared_data(
                T=T,
                N=N,
                include_positions=True,
                include_multiscale=True,
                n_scales=n_scales,
            )
            config = MesonOperatorConfig()
            result = compute_meson_operators(data, config)
            assert result["scalar"].shape[0] == n_scales
            assert result["pseudoscalar"].shape[0] == n_scales

    def test_multiscale_values_are_real_float32(self):
        data = make_prepared_data(
            T=10,
            N=20,
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert result["scalar"].dtype == torch.float32
        assert result["pseudoscalar"].dtype == torch.float32

    def test_multiscale_values_are_finite(self):
        data = make_prepared_data(
            T=10,
            N=20,
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = MesonOperatorConfig()
        result = compute_meson_operators(data, config)
        assert torch.isfinite(result["scalar"]).all()
        assert torch.isfinite(result["pseudoscalar"]).all()

    def test_multiscale_with_score_directed(self):
        data = make_prepared_data(
            T=10,
            N=20,
            include_positions=True,
            include_scores=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = MesonOperatorConfig(operator_mode="score_directed")
        result = compute_meson_operators(data, config)
        assert result["scalar"].shape == (3, 10)
        assert result["pseudoscalar"].shape == (3, 10)
        assert torch.isfinite(result["scalar"]).all()
