"""Comprehensive tests for fragile.physics.operators.glueball_operators."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.config import GlueballOperatorConfig
from fragile.physics.operators.glueball_operators import (
    _glueball_observable_from_plaquette,
    _resolve_glueball_operator_mode,
    _resolve_positive_length,
    compute_glueball_operators,
)

from .conftest import make_prepared_data


# ============================================================================
# TestResolveGlueballMode
# ============================================================================


class TestResolveGlueballMode:
    """Tests for _resolve_glueball_operator_mode."""

    def test_none_with_use_action_false(self):
        """None + use_action_form=False -> 're_plaquette'."""
        result = _resolve_glueball_operator_mode(operator_mode=None, use_action_form=False)
        assert result == "re_plaquette"

    def test_none_with_use_action_true(self):
        """None + use_action_form=True -> 'action_re_plaquette'."""
        result = _resolve_glueball_operator_mode(operator_mode=None, use_action_form=True)
        assert result == "action_re_plaquette"

    def test_empty_string_with_use_action_false(self):
        """Empty string + use_action_form=False -> 're_plaquette'."""
        result = _resolve_glueball_operator_mode(operator_mode="", use_action_form=False)
        assert result == "re_plaquette"

    def test_empty_string_with_use_action_true(self):
        """Empty string + use_action_form=True -> 'action_re_plaquette'."""
        result = _resolve_glueball_operator_mode(operator_mode="  ", use_action_form=True)
        assert result == "action_re_plaquette"

    def test_action_shortcut_normalized(self):
        """'action' -> 'action_re_plaquette'."""
        result = _resolve_glueball_operator_mode(operator_mode="action", use_action_form=False)
        assert result == "action_re_plaquette"

    def test_explicit_re_plaquette_passthrough(self):
        """Explicit 're_plaquette' passes through unchanged."""
        result = _resolve_glueball_operator_mode(
            operator_mode="re_plaquette", use_action_form=True
        )
        assert result == "re_plaquette"

    def test_explicit_phase_action_passthrough(self):
        """Explicit 'phase_action' passes through unchanged."""
        result = _resolve_glueball_operator_mode(
            operator_mode="phase_action", use_action_form=False
        )
        assert result == "phase_action"

    def test_explicit_phase_sin2_passthrough(self):
        """Explicit 'phase_sin2' passes through unchanged."""
        result = _resolve_glueball_operator_mode(operator_mode="phase_sin2", use_action_form=True)
        assert result == "phase_sin2"


# ============================================================================
# TestGlueballObservable
# ============================================================================


class TestGlueballObservable:
    """Tests for _glueball_observable_from_plaquette."""

    @pytest.fixture
    def sample_plaquette(self) -> Tensor:
        """Complex plaquette tensor for testing."""
        # Create plaquettes with known phases and magnitudes
        gen = torch.Generator().manual_seed(77)
        real = torch.randn(5, generator=gen)
        imag = torch.randn(5, generator=gen)
        return torch.complex(real, imag)

    def test_re_plaquette_returns_real_part(self, sample_plaquette: Tensor):
        """re_plaquette mode returns pi.real."""
        result = _glueball_observable_from_plaquette(
            sample_plaquette, operator_mode="re_plaquette"
        )
        expected = sample_plaquette.real.float()
        torch.testing.assert_close(result, expected)

    def test_action_re_plaquette_returns_one_minus_real(self, sample_plaquette: Tensor):
        """action_re_plaquette mode returns 1 - pi.real."""
        result = _glueball_observable_from_plaquette(
            sample_plaquette, operator_mode="action_re_plaquette"
        )
        expected = (1.0 - sample_plaquette.real).float()
        torch.testing.assert_close(result, expected)

    def test_phase_action_returns_one_minus_cos_angle(self, sample_plaquette: Tensor):
        """phase_action mode returns 1 - cos(angle(pi))."""
        result = _glueball_observable_from_plaquette(
            sample_plaquette, operator_mode="phase_action"
        )
        phase = torch.angle(sample_plaquette)
        expected = (1.0 - torch.cos(phase)).float()
        torch.testing.assert_close(result, expected)

    def test_phase_sin2_returns_sin_squared_angle(self, sample_plaquette: Tensor):
        """phase_sin2 mode returns sin^2(angle(pi))."""
        result = _glueball_observable_from_plaquette(sample_plaquette, operator_mode="phase_sin2")
        phase = torch.angle(sample_plaquette)
        expected = torch.sin(phase).square().float()
        torch.testing.assert_close(result, expected)

    def test_invalid_mode_raises_value_error(self, sample_plaquette: Tensor):
        """Invalid operator mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid glueball operator_mode"):
            _glueball_observable_from_plaquette(sample_plaquette, operator_mode="invalid_mode")


# ============================================================================
# TestResolvePositiveLength
# ============================================================================


class TestResolvePositiveLength:
    """Tests for _resolve_positive_length."""

    def test_explicit_positive_box_length_returned(self):
        """Explicit positive box_length is returned directly."""
        positions = torch.randn(10, 20)
        result = _resolve_positive_length(positions_axis=positions, box_length=7.5)
        assert result == 7.5

    def test_none_box_length_computes_from_span(self):
        """None box_length computes length from positions span."""
        positions = torch.tensor([[0.0, 1.0, 2.0, 5.0], [0.0, 1.0, 2.0, 5.0]])
        result = _resolve_positive_length(positions_axis=positions, box_length=None)
        # span = max(5.0) - min(0.0) = 5.0
        assert result == 5.0

    def test_zero_box_length_computes_from_span(self):
        """Zero box_length falls back to positions span."""
        positions = torch.tensor([[1.0, 3.0], [1.0, 3.0]])
        result = _resolve_positive_length(positions_axis=positions, box_length=0.0)
        assert result == 2.0

    def test_minimum_length_is_one(self):
        """When span is less than 1.0, the minimum of 1.0 is returned."""
        positions = torch.tensor([[0.0, 0.1], [0.0, 0.1]])
        result = _resolve_positive_length(positions_axis=positions, box_length=None)
        assert result == 1.0


# ============================================================================
# TestComputeGlueballOperators
# ============================================================================


class TestComputeGlueballOperators:
    """Tests for the main compute_glueball_operators function."""

    def test_output_contains_glueball_key(self):
        """Result dict always contains the 'glueball' key."""
        data = make_prepared_data()
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert "glueball" in result

    def test_output_shape_is_T(self):
        """Output 'glueball' has shape [T]."""
        T = 10
        data = make_prepared_data(T=T)
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert result["glueball"].shape == (T,)

    def test_T_zero_returns_empty(self):
        """T=0 returns an empty tensor for 'glueball'."""
        data = make_prepared_data(T=0, N=5)
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert result["glueball"].shape == (0,)
        assert result["glueball"].dtype == torch.float32

    def test_output_dtype_float32(self):
        """Output tensor has dtype float32."""
        data = make_prepared_data()
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert result["glueball"].dtype == torch.float32

    def test_output_is_finite(self):
        """All output values are finite."""
        data = make_prepared_data()
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert torch.isfinite(result["glueball"]).all()

    def test_output_is_real(self):
        """Output values are real (no imaginary component)."""
        data = make_prepared_data()
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert not result["glueball"].is_complex()

    def test_different_operator_modes_give_different_results(self):
        """Different operator_mode values produce different results."""
        data = make_prepared_data(seed=123)
        results = {}
        for mode in ["re_plaquette", "action_re_plaquette", "phase_action", "phase_sin2"]:
            config = GlueballOperatorConfig(operator_mode=mode)
            results[mode] = compute_glueball_operators(data, config)["glueball"]

        # At least some pairs must differ
        any_different = False
        modes = list(results.keys())
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                if not torch.allclose(results[modes[i]], results[modes[j]], atol=1e-6):
                    any_different = True
                    break
        assert any_different, "Expected at least some operator modes to differ"

    def test_use_action_form_backward_compat(self):
        """use_action_form=True with operator_mode=None gives action_re_plaquette result."""
        data = make_prepared_data(seed=200)
        config_action = GlueballOperatorConfig(operator_mode=None, use_action_form=True)
        config_explicit = GlueballOperatorConfig(
            operator_mode="action_re_plaquette", use_action_form=False
        )
        result_action = compute_glueball_operators(data, config_action)["glueball"]
        result_explicit = compute_glueball_operators(data, config_explicit)["glueball"]
        torch.testing.assert_close(result_action, result_explicit)

    def test_operator_mode_none_defaults_re_plaquette(self):
        """operator_mode=None with use_action_form=False gives re_plaquette result."""
        data = make_prepared_data(seed=300)
        config_default = GlueballOperatorConfig(operator_mode=None, use_action_form=False)
        config_explicit = GlueballOperatorConfig(
            operator_mode="re_plaquette", use_action_form=False
        )
        result_default = compute_glueball_operators(data, config_default)["glueball"]
        result_explicit = compute_glueball_operators(data, config_explicit)["glueball"]
        torch.testing.assert_close(result_default, result_explicit)

    def test_only_glueball_key_without_momentum(self):
        """Without momentum projection, only 'glueball' key is present."""
        data = make_prepared_data()
        config = GlueballOperatorConfig(use_momentum_projection=False)
        result = compute_glueball_operators(data, config)
        assert list(result.keys()) == ["glueball"]


# ============================================================================
# TestComputeGlueballMomentum
# ============================================================================


class TestComputeGlueballMomentum:
    """Tests for momentum projection in compute_glueball_operators."""

    def test_momentum_projection_adds_extra_keys(self):
        """use_momentum_projection=True adds cos and sin mode keys."""
        data = make_prepared_data(include_momentum_axis=True)
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=2)
        result = compute_glueball_operators(data, config)
        assert "glueball" in result
        for n in range(3):  # 0, 1, 2
            assert f"glueball_momentum_cos_{n}" in result
            assert f"glueball_momentum_sin_{n}" in result

    def test_number_of_momentum_modes(self):
        """Number of momentum modes = momentum_mode_max + 1."""
        data = make_prepared_data(include_momentum_axis=True)
        mode_max = 4
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=mode_max)
        result = compute_glueball_operators(data, config)
        n_modes = mode_max + 1
        cos_keys = [k for k in result if k.startswith("glueball_momentum_cos_")]
        sin_keys = [k for k in result if k.startswith("glueball_momentum_sin_")]
        assert len(cos_keys) == n_modes
        assert len(sin_keys) == n_modes

    def test_momentum_mode_shape_T(self):
        """Each momentum mode has shape [T]."""
        T = 10
        data = make_prepared_data(T=T, include_momentum_axis=True)
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=2)
        result = compute_glueball_operators(data, config)
        for n in range(3):
            assert result[f"glueball_momentum_cos_{n}"].shape == (T,)
            assert result[f"glueball_momentum_sin_{n}"].shape == (T,)

    def test_momentum_values_are_finite(self):
        """All momentum mode values are finite."""
        data = make_prepared_data(include_momentum_axis=True)
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=3)
        result = compute_glueball_operators(data, config)
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key}"

    def test_momentum_values_are_float32(self):
        """All momentum mode values have dtype float32."""
        data = make_prepared_data(include_momentum_axis=True)
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=2)
        result = compute_glueball_operators(data, config)
        for key, val in result.items():
            assert val.dtype == torch.float32, f"Wrong dtype in {key}"

    def test_missing_positions_axis_raises(self):
        """use_momentum_projection=True without positions_axis raises ValueError."""
        data = make_prepared_data(include_momentum_axis=False)
        config = GlueballOperatorConfig(use_momentum_projection=True)
        with pytest.raises(ValueError, match="positions_axis is required"):
            compute_glueball_operators(data, config)


# ============================================================================
# TestComputeGlueballMultiscale
# ============================================================================


class TestComputeGlueballMultiscale:
    """Tests for multiscale glueball operators."""

    def test_multiscale_output_shape_S_T(self):
        """Multiscale glueball has shape [S, T]."""
        T, N, S = 10, 20, 3
        data = make_prepared_data(
            T=T,
            N=N,
            include_positions=True,
            include_momentum_axis=True,
            include_multiscale=True,
            n_scales=S,
        )
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert result["glueball"].shape == (S, T)

    def test_multiscale_values_are_finite(self):
        """Multiscale glueball values are all finite."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = GlueballOperatorConfig()
        result = compute_glueball_operators(data, config)
        assert torch.isfinite(result["glueball"]).all()

    def test_multiscale_momentum_modes_shape_S_T(self):
        """Multiscale momentum modes have shape [S, T]."""
        T, N, S = 10, 20, 3
        data = make_prepared_data(
            T=T,
            N=N,
            include_positions=True,
            include_momentum_axis=True,
            include_multiscale=True,
            n_scales=S,
        )
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=2)
        result = compute_glueball_operators(data, config)
        for n in range(3):
            assert result[f"glueball_momentum_cos_{n}"].shape == (S, T)
            assert result[f"glueball_momentum_sin_{n}"].shape == (S, T)

    def test_multiscale_momentum_values_finite(self):
        """Multiscale momentum mode values are all finite."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = GlueballOperatorConfig(use_momentum_projection=True, momentum_mode_max=2)
        result = compute_glueball_operators(data, config)
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key}"
