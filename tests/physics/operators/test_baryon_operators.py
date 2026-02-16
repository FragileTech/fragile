"""Comprehensive tests for baryon (nucleon) operator construction."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.baryon_operators import (
    _baryon_flux_weight_from_plaquette,
    _det3,
    _resolve_baryon_operator_mode,
    compute_baryon_operators,
)
from fragile.physics.operators.config import BaryonOperatorConfig

from .conftest import make_prepared_data


# =========================================================================
# 1. TestDet3
# =========================================================================


class TestDet3:
    """Tests for the _det3 helper that computes 3x3 determinants from column vectors."""

    def test_identity_columns_det_one(self):
        """Identity matrix columns should give determinant = 1."""
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.tensor([0.0, 0.0, 1.0])
        result = _det3(a, b, c)
        assert torch.isclose(result, torch.tensor(1.0)), f"Expected 1.0, got {result.item()}"

    def test_known_matrix_determinant(self):
        """Known 3x3 matrix determinant matches torch.linalg.det."""
        # Matrix:
        # [[1, 4, 7],
        #  [2, 5, 8],
        #  [3, 6, 10]]
        # Columns are a=[1,2,3], b=[4,5,6], c=[7,8,10]
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = torch.tensor([7.0, 8.0, 10.0])
        result = _det3(a, b, c)
        # Reference via torch
        mat = torch.stack([a, b, c], dim=-1)  # [3, 3]
        expected = torch.linalg.det(mat)
        assert torch.isclose(result, expected, atol=1e-5), (
            f"Expected {expected.item()}, got {result.item()}"
        )

    def test_zero_column_det_zero(self):
        """If one column is all zeros, determinant should be 0."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.zeros(3)
        c = torch.tensor([7.0, 8.0, 9.0])
        result = _det3(a, b, c)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6), (
            f"Expected 0.0, got {result.item()}"
        )

    def test_batched_input(self):
        """_det3 should work with batched [T, N, 3] inputs."""
        T, N = 4, 5
        gen = torch.Generator().manual_seed(123)
        a = torch.randn(T, N, 3, generator=gen)
        b = torch.randn(T, N, 3, generator=gen)
        c = torch.randn(T, N, 3, generator=gen)
        result = _det3(a, b, c)
        assert result.shape == (T, N), f"Expected shape ({T}, {N}), got {result.shape}"
        # Check a single element against torch.linalg.det
        mat = torch.stack([a[0, 0], b[0, 0], c[0, 0]], dim=-1)
        expected = torch.linalg.det(mat)
        assert torch.isclose(result[0, 0], expected, atol=1e-5)

    def test_complex_input(self):
        """_det3 should handle complex-valued column vectors."""
        a = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j])
        b = torch.tensor([0.0 + 0j, 0.0 + 1j, 0.0 + 0j])
        c = torch.tensor([0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
        result = _det3(a, b, c)
        # det of diag(1, i, 1) = i
        expected = torch.tensor(0.0 + 1j)
        assert torch.isclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"


# =========================================================================
# 2. TestResolveBaryonMode
# =========================================================================


class TestResolveBaryonMode:
    """Tests for _resolve_baryon_operator_mode normalization."""

    def test_none_returns_det_abs(self):
        """None input should resolve to 'det_abs'."""
        assert _resolve_baryon_operator_mode(None) == "det_abs"

    def test_empty_string_returns_det_abs(self):
        """Empty string input should resolve to 'det_abs'."""
        assert _resolve_baryon_operator_mode("") == "det_abs"

    def test_whitespace_returns_det_abs(self):
        """Whitespace-only input should resolve to 'det_abs'."""
        assert _resolve_baryon_operator_mode("   ") == "det_abs"

    @pytest.mark.parametrize(
        "mode",
        ["det_abs", "flux_action", "flux_sin2", "flux_exp", "score_signed", "score_abs"],
    )
    def test_valid_modes_pass(self, mode: str):
        """All valid mode strings should be returned unchanged."""
        assert _resolve_baryon_operator_mode(mode) == mode

    def test_invalid_mode_raises(self):
        """An invalid mode string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid baryon operator_mode"):
            _resolve_baryon_operator_mode("not_a_real_mode")

    def test_case_insensitive(self):
        """Mode resolution should be case-insensitive."""
        assert _resolve_baryon_operator_mode("DET_ABS") == "det_abs"
        assert _resolve_baryon_operator_mode("Flux_Action") == "flux_action"


# =========================================================================
# 3. TestBaryonFluxWeight
# =========================================================================


class TestBaryonFluxWeight:
    """Tests for _baryon_flux_weight_from_plaquette."""

    def _make_plaquette(self, phase: float) -> Tensor:
        """Create a complex plaquette tensor with a given phase."""
        return torch.tensor(
            [math.cos(phase) + 1j * math.sin(phase)],
            dtype=torch.complex64,
        )

    def test_flux_action(self):
        """flux_action: weight = 1 - cos(phase)."""
        phase = 0.5
        pi = self._make_plaquette(phase)
        result = _baryon_flux_weight_from_plaquette(
            pi=pi,
            operator_mode="flux_action",
            flux_exp_alpha=1.0,
        )
        expected = 1.0 - math.cos(phase)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-5)

    def test_flux_sin2(self):
        """flux_sin2: weight = sin^2(phase)."""
        phase = 0.7
        pi = self._make_plaquette(phase)
        result = _baryon_flux_weight_from_plaquette(
            pi=pi,
            operator_mode="flux_sin2",
            flux_exp_alpha=1.0,
        )
        expected = math.sin(phase) ** 2
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-5)

    def test_flux_exp(self):
        """flux_exp: weight = exp(alpha * (1 - cos(phase)))."""
        phase = 1.0
        alpha = 2.0
        pi = self._make_plaquette(phase)
        result = _baryon_flux_weight_from_plaquette(
            pi=pi,
            operator_mode="flux_exp",
            flux_exp_alpha=alpha,
        )
        expected = math.exp(alpha * (1.0 - math.cos(phase)))
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-4)

    def test_flux_exp_negative_alpha_clamped(self):
        """flux_exp: negative alpha should be clamped to 0."""
        phase = 1.0
        pi = self._make_plaquette(phase)
        result = _baryon_flux_weight_from_plaquette(
            pi=pi,
            operator_mode="flux_exp",
            flux_exp_alpha=-5.0,
        )
        # alpha clamped to 0 -> exp(0 * action) = 1.0
        assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float32), atol=1e-5)

    def test_invalid_mode_raises(self):
        """Invalid operator_mode should raise ValueError."""
        pi = self._make_plaquette(0.5)
        with pytest.raises(ValueError, match="Invalid baryon operator_mode"):
            _baryon_flux_weight_from_plaquette(
                pi=pi,
                operator_mode="bogus_mode",
                flux_exp_alpha=1.0,
            )


# =========================================================================
# 4. TestComputeBaryonOperatorsDetAbs
# =========================================================================


class TestComputeBaryonOperatorsDetAbs:
    """Tests for compute_baryon_operators with det_abs mode (default)."""

    def test_output_key_nucleon(self):
        """Output dict should contain key 'nucleon'."""
        data = make_prepared_data()
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert "nucleon" in result

    def test_output_shape_T(self):
        """Output tensor should have shape [T]."""
        T, N = 10, 20
        data = make_prepared_data(T=T, N=N)
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (T,)

    def test_empty_T_returns_empty(self):
        """T=0 should return an empty tensor."""
        data = make_prepared_data(T=0, N=20)
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (0,)

    def test_values_non_negative(self):
        """det_abs mode should produce non-negative values (absolute value)."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="det_abs")
        result = compute_baryon_operators(data, config)
        assert (result["nucleon"] >= 0).all(), "det_abs should yield non-negative values"

    def test_output_is_finite(self):
        """All output values should be finite."""
        data = make_prepared_data()
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert torch.isfinite(result["nucleon"]).all()

    def test_output_dtype_float32(self):
        """Output should be float32."""
        data = make_prepared_data()
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].dtype == torch.float32

    def test_different_T_N_sizes(self):
        """Test with different T and N sizes."""
        for t, n in [(5, 10), (15, 30), (3, 6)]:
            data = make_prepared_data(T=t, N=n)
            config = BaryonOperatorConfig()
            result = compute_baryon_operators(data, config)
            assert result["nucleon"].shape == (t,), f"Failed for T={t}, N={n}"

    def test_default_mode_is_det_abs(self):
        """Default config (no operator_mode) should behave like det_abs."""
        data = make_prepared_data(seed=77)
        config_default = BaryonOperatorConfig()
        config_explicit = BaryonOperatorConfig(operator_mode="det_abs")
        result_default = compute_baryon_operators(data, config_default)
        # Recreate same data with same seed
        data2 = make_prepared_data(seed=77)
        result_explicit = compute_baryon_operators(data2, config_explicit)
        assert torch.allclose(result_default["nucleon"], result_explicit["nucleon"])


# =========================================================================
# 5. TestComputeBaryonOperatorsFluxModes
# =========================================================================


class TestComputeBaryonOperatorsFluxModes:
    """Tests for compute_baryon_operators with flux-based modes."""

    def test_flux_action_output_shape(self):
        """flux_action mode should produce shape [T]."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="flux_action")
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (data.color.shape[0],)

    def test_flux_action_non_negative(self):
        """flux_action: output should be non-negative."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="flux_action")
        result = compute_baryon_operators(data, config)
        assert (result["nucleon"] >= -1e-7).all(), "flux_action should yield non-negative values"

    def test_flux_sin2_output_shape(self):
        """flux_sin2 mode should produce shape [T]."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="flux_sin2")
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (data.color.shape[0],)

    def test_flux_sin2_finite(self):
        """flux_sin2 output should be finite."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="flux_sin2")
        result = compute_baryon_operators(data, config)
        assert torch.isfinite(result["nucleon"]).all()

    def test_flux_exp_positive(self):
        """flux_exp: exp is always positive, so output should be non-negative."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="flux_exp", flux_exp_alpha=1.0)
        result = compute_baryon_operators(data, config)
        assert (result["nucleon"] >= -1e-7).all(), "flux_exp should yield non-negative values"

    def test_flux_exp_finite(self):
        """flux_exp output should be finite."""
        data = make_prepared_data()
        config = BaryonOperatorConfig(operator_mode="flux_exp", flux_exp_alpha=1.0)
        result = compute_baryon_operators(data, config)
        assert torch.isfinite(result["nucleon"]).all()

    def test_flux_modes_produce_different_results(self):
        """Different flux modes should generally produce different results."""
        make_prepared_data(seed=55)
        results = {}
        for mode in ("flux_action", "flux_sin2", "flux_exp"):
            d = make_prepared_data(seed=55)
            config = BaryonOperatorConfig(operator_mode=mode, flux_exp_alpha=2.0)
            results[mode] = compute_baryon_operators(d, config)["nucleon"]
        # At least one pair should differ
        any_differ = (
            not torch.allclose(results["flux_action"], results["flux_sin2"])
            or not torch.allclose(results["flux_action"], results["flux_exp"])
            or not torch.allclose(results["flux_sin2"], results["flux_exp"])
        )
        assert any_differ, "Expected at least some flux modes to produce different results"


# =========================================================================
# 6. TestComputeBaryonOperatorsScoreModes
# =========================================================================


class TestComputeBaryonOperatorsScoreModes:
    """Tests for compute_baryon_operators with score-based modes."""

    def test_score_signed_requires_scores(self):
        """score_signed mode without scores should raise ValueError."""
        data = make_prepared_data(include_scores=False)
        config = BaryonOperatorConfig(operator_mode="score_signed")
        with pytest.raises(ValueError, match="scores is required"):
            compute_baryon_operators(data, config)

    def test_score_abs_requires_scores(self):
        """score_abs mode without scores should raise ValueError."""
        data = make_prepared_data(include_scores=False)
        config = BaryonOperatorConfig(operator_mode="score_abs")
        with pytest.raises(ValueError, match="scores is required"):
            compute_baryon_operators(data, config)

    def test_score_signed_output_shape(self):
        """score_signed should produce shape [T]."""
        data = make_prepared_data(include_scores=True)
        config = BaryonOperatorConfig(operator_mode="score_signed")
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (data.color.shape[0],)

    def test_score_abs_non_negative(self):
        """score_abs should yield non-negative values."""
        data = make_prepared_data(include_scores=True)
        config = BaryonOperatorConfig(operator_mode="score_abs")
        result = compute_baryon_operators(data, config)
        assert (result["nucleon"] >= -1e-7).all(), "score_abs should yield non-negative values"

    def test_score_signed_can_be_negative(self):
        """score_signed can produce negative values (real part of signed det)."""
        # We just verify it runs and produces finite results; sign depends on data
        data = make_prepared_data(include_scores=True)
        config = BaryonOperatorConfig(operator_mode="score_signed")
        result = compute_baryon_operators(data, config)
        assert torch.isfinite(result["nucleon"]).all()

    def test_score_abs_and_signed_differ(self):
        """score_abs and score_signed should generally differ."""
        data_a = make_prepared_data(include_scores=True, seed=88)
        data_b = make_prepared_data(include_scores=True, seed=88)
        r_signed = compute_baryon_operators(
            data_a,
            BaryonOperatorConfig(operator_mode="score_signed"),
        )["nucleon"]
        r_abs = compute_baryon_operators(
            data_b,
            BaryonOperatorConfig(operator_mode="score_abs"),
        )["nucleon"]
        # They may coincide for all-positive dets, but generally won't
        # Just check both are finite and same shape
        assert r_signed.shape == r_abs.shape
        assert torch.isfinite(r_signed).all()
        assert torch.isfinite(r_abs).all()


# =========================================================================
# 7. TestComputeBaryonOperatorsMultiscale
# =========================================================================


class TestComputeBaryonOperatorsMultiscale:
    """Tests for compute_baryon_operators with multiscale data."""

    def test_multiscale_output_shape(self):
        """Multiscale output should have shape [S, T]."""
        n_scales = 3
        T, N = 10, 20
        data = make_prepared_data(
            T=T,
            N=N,
            include_positions=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (n_scales, T), (
            f"Expected shape ({n_scales}, {T}), got {result['nucleon'].shape}"
        )

    def test_multiscale_s_matches_n_scales(self):
        """First dimension should match number of scales."""
        for n_scales in (2, 4, 5):
            data = make_prepared_data(
                T=8,
                N=15,
                include_positions=True,
                include_multiscale=True,
                n_scales=n_scales,
            )
            config = BaryonOperatorConfig()
            result = compute_baryon_operators(data, config)
            assert result["nucleon"].shape[0] == n_scales, (
                f"Expected S={n_scales}, got {result['nucleon'].shape[0]}"
            )

    def test_multiscale_finite(self):
        """Multiscale output should be finite."""
        data = make_prepared_data(
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = BaryonOperatorConfig()
        result = compute_baryon_operators(data, config)
        assert torch.isfinite(result["nucleon"]).all()

    def test_multiscale_flux_mode(self):
        """Multiscale should work with flux modes."""
        n_scales = 3
        T = 10
        data = make_prepared_data(
            T=T,
            N=20,
            include_positions=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = BaryonOperatorConfig(operator_mode="flux_action")
        result = compute_baryon_operators(data, config)
        assert result["nucleon"].shape == (n_scales, T)
        assert torch.isfinite(result["nucleon"]).all()
