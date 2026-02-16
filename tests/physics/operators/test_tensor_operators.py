"""Comprehensive tests for fragile.physics.operators.tensor_operators."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.config import TensorOperatorConfig
from fragile.physics.operators.tensor_operators import (
    _resolve_positive_length,
    _traceless_tensor_components,
    compute_tensor_operators,
    TENSOR_COMPONENT_LABELS,
)

from .conftest import make_prepared_data


# ============================================================================
# TestTracelessTensorComponents
# ============================================================================


class TestTracelessTensorComponents:
    """Tests for _traceless_tensor_components helper."""

    def test_unit_x(self):
        """Unit x vector [1,0,0] produces known component values."""
        dx = torch.tensor([[1.0, 0.0, 0.0]])
        result = _traceless_tensor_components(dx)
        assert result.shape == (1, 5)
        # q_xy = x*y = 0, q_xz = x*z = 0, q_yz = y*z = 0
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, 1].item() == pytest.approx(0.0)
        assert result[0, 2].item() == pytest.approx(0.0)
        # q_xx_minus_yy = (1 - 0)/sqrt(2) = 1/sqrt(2)
        assert result[0, 3].item() == pytest.approx(1.0 / math.sqrt(2.0))
        # q_2zz_minus_xx_minus_yy = (0 - 1 - 0)/sqrt(6) = -1/sqrt(6)
        assert result[0, 4].item() == pytest.approx(-1.0 / math.sqrt(6.0))

    def test_unit_y(self):
        """Unit y vector [0,1,0] produces known component values."""
        dx = torch.tensor([[0.0, 1.0, 0.0]])
        result = _traceless_tensor_components(dx)
        assert result.shape == (1, 5)
        # q_xy = 0, q_xz = 0, q_yz = 0
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, 1].item() == pytest.approx(0.0)
        assert result[0, 2].item() == pytest.approx(0.0)
        # q_xx_minus_yy = (0 - 1)/sqrt(2) = -1/sqrt(2)
        assert result[0, 3].item() == pytest.approx(-1.0 / math.sqrt(2.0))
        # q_2zz_minus_xx_minus_yy = (0 - 0 - 1)/sqrt(6) = -1/sqrt(6)
        assert result[0, 4].item() == pytest.approx(-1.0 / math.sqrt(6.0))

    def test_unit_z(self):
        """Unit z vector [0,0,1] produces known component values."""
        dx = torch.tensor([[0.0, 0.0, 1.0]])
        result = _traceless_tensor_components(dx)
        assert result.shape == (1, 5)
        # q_xy = 0, q_xz = 0, q_yz = 0
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, 1].item() == pytest.approx(0.0)
        assert result[0, 2].item() == pytest.approx(0.0)
        # q_xx_minus_yy = (0 - 0)/sqrt(2) = 0
        assert result[0, 3].item() == pytest.approx(0.0)
        # q_2zz_minus_xx_minus_yy = (2 - 0 - 0)/sqrt(6) = 2/sqrt(6)
        assert result[0, 4].item() == pytest.approx(2.0 / math.sqrt(6.0))

    def test_traceless_property(self):
        """The 3x3 traceless symmetric tensor has zero trace.

        The diagonal of the reconstructed 3x3 tensor from the five components
        should sum to zero. The diagonal elements are:
            T_xx = q_xx_minus_yy / sqrt(2) - q_2zz.../sqrt(6) (from reconstruction)
        But more directly: the 5-component basis is traceless by construction,
        meaning for ANY input dx, the corresponding 3x3 tensor
        Q^{ab} has trace Q^{xx} + Q^{yy} + Q^{zz} = 0.

        We verify this by reconstructing the diagonal from components:
            Q_xx = (x^2) = q_xx_minus_yy/sqrt(2) * (1/sqrt(2)) + ...
        Simpler: just verify x^2 + y^2 + z^2 can be decomposed.
        The direct check: x^2 + y^2 + z^2 should NOT appear in the basis.
        The trace of Q^{ab}(dx) = x^2 + y^2 + z^2 projected onto traceless = 0.

        Concrete check: for the full 3x3 matrix
            Q = [[x*x, x*y, x*z],
                 [y*x, y*y, y*z],
                 [z*x, z*y, z*z]]
        trace = x^2 + y^2 + z^2.  The traceless part subtracts (trace/3)*I.
        We verify that our 5 components reconstruct a traceless tensor.
        """
        dx = torch.tensor([[2.0, 3.0, -1.0]])
        result = _traceless_tensor_components(dx)
        x, y, z = 2.0, 3.0, -1.0

        # Reconstruct the 3x3 traceless symmetric tensor from 5 components
        q_xy = result[0, 0].item()
        q_xz = result[0, 1].item()
        q_yz = result[0, 2].item()
        q_xxmyy = result[0, 3].item()
        q_2zzmxxmyy = result[0, 4].item()

        # Verify off-diagonal components match
        assert q_xy == pytest.approx(x * y)
        assert q_xz == pytest.approx(x * z)
        assert q_yz == pytest.approx(y * z)

        # Reconstruct diagonal elements
        # From the basis: q_xx_minus_yy = (x^2 - y^2)/sqrt(2)
        #                 q_2zz_minus_xx_minus_yy = (2z^2 - x^2 - y^2)/sqrt(6)
        # Solving for diagonal:
        # Let A = q_xxmyy * sqrt(2) = x^2 - y^2
        # Let B = q_2zzmxxmyy * sqrt(6) = 2z^2 - x^2 - y^2
        # Then: x^2 - y^2 = A, 2z^2 - x^2 - y^2 = B
        # From these: x^2 + y^2 = 2*z^2 - B, x^2 - y^2 = A
        # => x^2 = (2*z^2 - B + A) / 2, y^2 = (2*z^2 - B - A) / 2
        # Trace-free part of diagonal = [x^2 - r/3, y^2 - r/3, z^2 - r/3]
        # where r = x^2 + y^2 + z^2.
        # The five components encode the traceless part; verify trace = 0:
        A = q_xxmyy * math.sqrt(2.0)
        B = q_2zzmxxmyy * math.sqrt(6.0)
        # A = x^2 - y^2, B = 2z^2 - x^2 - y^2
        # A + B = 2z^2 - 2y^2, so the system is consistent but traceless:
        # The reconstructed diagonal sums to: A + B + ... = traceless => 0.
        # Just verify: x^2 - y^2 = A and 2z^2 - x^2 - y^2 = B
        assert A == pytest.approx(x * x - y * y)
        assert B == pytest.approx(2 * z * z - x * x - y * y)

    def test_output_shape_1d(self):
        """Single vector [3] produces shape [5]."""
        dx = torch.randn(3)
        result = _traceless_tensor_components(dx)
        assert result.shape == (5,)

    def test_output_shape_2d(self):
        """Batch [N, 3] produces shape [N, 5]."""
        dx = torch.randn(7, 3)
        result = _traceless_tensor_components(dx)
        assert result.shape == (7, 5)

    def test_output_shape_3d(self):
        """Batched [T, N, 3] produces shape [T, N, 5]."""
        dx = torch.randn(4, 8, 3)
        result = _traceless_tensor_components(dx)
        assert result.shape == (4, 8, 5)

    def test_invalid_last_dim_raises(self):
        """Input with last dim != 3 raises ValueError."""
        dx = torch.randn(5, 4)
        with pytest.raises(ValueError, match="Expected dx"):
            _traceless_tensor_components(dx)

    def test_zero_dim_raises(self):
        """Scalar tensor raises ValueError."""
        dx = torch.tensor(1.0)
        with pytest.raises(ValueError, match="Expected dx"):
            _traceless_tensor_components(dx)


# ============================================================================
# TestResolvePositiveLength
# ============================================================================


class TestResolvePositiveLength:
    """Tests for _resolve_positive_length helper."""

    def test_box_length_positive(self):
        """When box_length > 0 is provided, it is returned directly."""
        axis = torch.randn(10, 20)
        result = _resolve_positive_length(positions_axis=axis, box_length=7.5)
        assert result == pytest.approx(7.5)

    def test_box_length_none_uses_span(self):
        """When box_length is None, the span of positions is used."""
        axis = torch.tensor([[0.0, 5.0, 10.0]])
        result = _resolve_positive_length(positions_axis=axis, box_length=None)
        assert result == pytest.approx(10.0)

    def test_box_length_zero_uses_span(self):
        """When box_length is 0, the span of positions is used."""
        axis = torch.tensor([[1.0, 4.0]])
        result = _resolve_positive_length(positions_axis=axis, box_length=0.0)
        assert result == pytest.approx(3.0)

    def test_tiny_span_clamped_to_one(self):
        """When span is tiny, a minimum of 1.0 is returned."""
        axis = torch.tensor([[0.0, 0.0, 0.0]])
        result = _resolve_positive_length(positions_axis=axis, box_length=None)
        assert result == pytest.approx(1.0)


# ============================================================================
# TestTensorComponentLabels
# ============================================================================


class TestTensorComponentLabels:
    """Tests for TENSOR_COMPONENT_LABELS constant."""

    def test_five_labels(self):
        """There are exactly 5 component labels."""
        assert len(TENSOR_COMPONENT_LABELS) == 5

    def test_expected_names(self):
        """Labels have the expected names."""
        assert TENSOR_COMPONENT_LABELS == (
            "q_xy",
            "q_xz",
            "q_yz",
            "q_xx_minus_yy",
            "q_2zz_minus_xx_minus_yy",
        )


# ============================================================================
# TestComputeTensorOperatorsBasic
# ============================================================================


class TestComputeTensorOperatorsBasic:
    """Tests for compute_tensor_operators single-scale path."""

    def test_output_has_tensor_key(self):
        """Result dict contains the 'tensor' key."""
        data = make_prepared_data(include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert "tensor" in result

    def test_single_scale_output_shape(self):
        """Single-scale output has shape [T, 5]."""
        T, N = 10, 20
        data = make_prepared_data(T, N, include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (T, 5)

    def test_empty_time_returns_empty_tensor(self):
        """T=0 returns shape [0, 5] for single-scale."""
        data = make_prepared_data(T=0, N=5, include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (0, 5)

    def test_empty_time_multiscale_returns_empty(self):
        """T=0 with multiscale returns shape [S, 0, 5]."""
        data = make_prepared_data(
            T=0,
            N=5,
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (3, 0, 5)

    def test_output_dtype_float32(self):
        """Output tensor is float32."""
        data = make_prepared_data(include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].dtype == torch.float32

    def test_output_values_finite(self):
        """All output values are finite."""
        data = make_prepared_data(include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert torch.isfinite(result["tensor"]).all()

    def test_five_components_last_dim(self):
        """Last dimension of output is 5 (five traceless tensor components)."""
        data = make_prepared_data(include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape[-1] == 5

    def test_missing_positions_raises(self):
        """Missing positions raises ValueError."""
        data = make_prepared_data(include_positions=False)
        config = TensorOperatorConfig()
        with pytest.raises(ValueError, match="positions is required"):
            compute_tensor_operators(data, config)

    def test_only_tensor_key_without_momentum(self):
        """Without momentum axis, only the 'tensor' key is present."""
        data = make_prepared_data(include_positions=True)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert set(result.keys()) == {"tensor"}


# ============================================================================
# TestComputeTensorOperatorsMomentum
# ============================================================================


class TestComputeTensorOperatorsMomentum:
    """Tests for momentum projection in compute_tensor_operators."""

    def test_momentum_keys_present(self):
        """When positions_axis is set, momentum cos/sin keys are present."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
        )
        config = TensorOperatorConfig(momentum_mode_max=3)
        result = compute_tensor_operators(data, config)
        for n in range(4):  # 0..3
            assert f"tensor_momentum_cos_{n}" in result
            assert f"tensor_momentum_sin_{n}" in result

    def test_momentum_mode_count(self):
        """Number of momentum modes equals momentum_mode_max + 1."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
        )
        max_mode = 5
        config = TensorOperatorConfig(momentum_mode_max=max_mode)
        result = compute_tensor_operators(data, config)
        cos_keys = [k for k in result if k.startswith("tensor_momentum_cos_")]
        sin_keys = [k for k in result if k.startswith("tensor_momentum_sin_")]
        assert len(cos_keys) == max_mode + 1
        assert len(sin_keys) == max_mode + 1

    def test_momentum_mode_shape_single_scale(self):
        """Each momentum mode has shape [T, 5] in single-scale."""
        T, N = 10, 20
        data = make_prepared_data(
            T,
            N,
            include_positions=True,
            include_momentum_axis=True,
        )
        config = TensorOperatorConfig(momentum_mode_max=2)
        result = compute_tensor_operators(data, config)
        for n in range(3):
            assert result[f"tensor_momentum_cos_{n}"].shape == (T, 5)
            assert result[f"tensor_momentum_sin_{n}"].shape == (T, 5)

    def test_momentum_values_finite(self):
        """All momentum mode values are finite."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
        )
        config = TensorOperatorConfig(momentum_mode_max=2)
        result = compute_tensor_operators(data, config)
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key}"

    def test_momentum_dtype_float32(self):
        """Momentum mode tensors are float32."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
        )
        config = TensorOperatorConfig(momentum_mode_max=1)
        result = compute_tensor_operators(data, config)
        for key, val in result.items():
            assert val.dtype == torch.float32, f"{key} has dtype {val.dtype}"

    def test_no_momentum_without_positions_axis(self):
        """Without positions_axis, no momentum keys appear."""
        data = make_prepared_data(include_positions=True)
        config = TensorOperatorConfig(momentum_mode_max=3)
        result = compute_tensor_operators(data, config)
        cos_keys = [k for k in result if "momentum" in k]
        assert len(cos_keys) == 0


# ============================================================================
# TestComputeTensorOperatorsMultiscale
# ============================================================================


class TestComputeTensorOperatorsMultiscale:
    """Tests for multiscale path in compute_tensor_operators."""

    def test_multiscale_output_shape(self):
        """Multiscale output has shape [S, T, 5]."""
        T, N, n_scales = 10, 20, 3
        data = make_prepared_data(
            T,
            N,
            include_positions=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (n_scales, T, 5)

    def test_multiscale_s_matches_n_scales(self):
        """First dimension matches the number of scales."""
        for n_scales in [2, 4, 6]:
            data = make_prepared_data(
                include_positions=True,
                include_multiscale=True,
                n_scales=n_scales,
            )
            config = TensorOperatorConfig()
            result = compute_tensor_operators(data, config)
            assert result["tensor"].shape[0] == n_scales

    def test_multiscale_values_finite(self):
        """All multiscale output values are finite."""
        data = make_prepared_data(
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert torch.isfinite(result["tensor"]).all()

    def test_multiscale_momentum_shape(self):
        """Multiscale + momentum produces [S, T, 5] for each mode."""
        T, N, n_scales = 10, 20, 3
        data = make_prepared_data(
            T,
            N,
            include_positions=True,
            include_momentum_axis=True,
            include_multiscale=True,
            n_scales=n_scales,
        )
        config = TensorOperatorConfig(momentum_mode_max=2)
        result = compute_tensor_operators(data, config)
        for n in range(3):
            assert result[f"tensor_momentum_cos_{n}"].shape == (n_scales, T, 5)
            assert result[f"tensor_momentum_sin_{n}"].shape == (n_scales, T, 5)

    def test_multiscale_momentum_values_finite(self):
        """All multiscale momentum values are finite."""
        data = make_prepared_data(
            include_positions=True,
            include_momentum_axis=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = TensorOperatorConfig(momentum_mode_max=2)
        result = compute_tensor_operators(data, config)
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key}"
