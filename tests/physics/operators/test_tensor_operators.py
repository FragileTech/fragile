"""Tests for fragile.physics.operators.tensor_operators (bilinear sigma projection)."""

from __future__ import annotations

import pytest
import torch

from fragile.physics.operators.config import TensorOperatorConfig
from fragile.physics.operators.tensor_operators import (
    _build_sigma_matrices,
    compute_tensor_operators,
)

from .conftest import make_prepared_data


# ============================================================================
# TestBuildSigmaMatrices
# ============================================================================


class TestBuildSigmaMatrices:
    """Tests for _build_sigma_matrices helper."""

    def test_d3_produces_three_matrices(self):
        """d=3 produces 3 antisymmetric sigma matrices (xy, xz, yz)."""
        sigma = _build_sigma_matrices(3, torch.device("cpu"))
        assert sigma.shape == (3, 3, 3)

    def test_d2_produces_one_matrix(self):
        """d=2 produces 1 antisymmetric sigma matrix (xy)."""
        sigma = _build_sigma_matrices(2, torch.device("cpu"))
        assert sigma.shape == (1, 2, 2)

    def test_d1_produces_empty(self):
        """d=1 produces empty tensor."""
        sigma = _build_sigma_matrices(1, torch.device("cpu"))
        assert sigma.shape[0] == 0

    def test_antisymmetric(self):
        """Each sigma matrix is antisymmetric: sigma^T = -sigma."""
        sigma = _build_sigma_matrices(3, torch.device("cpu"))
        for p in range(sigma.shape[0]):
            assert torch.allclose(sigma[p], -sigma[p].T)

    def test_purely_imaginary(self):
        """Each sigma matrix is purely imaginary."""
        sigma = _build_sigma_matrices(3, torch.device("cpu"))
        assert torch.allclose(sigma.real, torch.zeros_like(sigma.real))


# ============================================================================
# TestComputeTensorOperatorsBasic
# ============================================================================


class TestComputeTensorOperatorsBasic:
    """Tests for compute_tensor_operators single-scale path."""

    def test_output_has_tensor_key(self):
        """Result dict contains the 'tensor' key."""
        data = make_prepared_data()
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert "tensor" in result

    def test_single_scale_output_shape(self):
        """Single-scale output has shape [T]."""
        T, N = 10, 20
        data = make_prepared_data(T, N)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (T,)

    def test_empty_time_returns_empty_tensor(self):
        """T=0 returns shape [0] for single-scale."""
        data = make_prepared_data(T=0, N=5)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (0,)

    def test_empty_time_multiscale_returns_empty(self):
        """T=0 with multiscale returns shape [S, 0]."""
        data = make_prepared_data(
            T=0,
            N=5,
            include_multiscale=True,
            n_scales=3,
        )
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].shape == (3, 0)

    def test_output_dtype_float32(self):
        """Output tensor is float32."""
        data = make_prepared_data()
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert result["tensor"].dtype == torch.float32

    def test_output_values_finite(self):
        """All output values are finite."""
        data = make_prepared_data()
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert torch.isfinite(result["tensor"]).all()

    def test_no_positions_required(self):
        """Tensor channel works without positions (bilinear, not position-dependent)."""
        data = make_prepared_data(include_positions=False)
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert "tensor" in result
        assert result["tensor"].shape == (10,)

    def test_only_tensor_key(self):
        """Only the 'tensor' key is present (no momentum keys)."""
        data = make_prepared_data()
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert set(result.keys()) == {"tensor"}


# ============================================================================
# TestComputeTensorOperatorsMultiscale
# ============================================================================


class TestComputeTensorOperatorsMultiscale:
    """Tests for multiscale path in compute_tensor_operators."""

    def test_multiscale_output_shape(self):
        """Multiscale output has shape [S, T]."""
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
        assert result["tensor"].shape == (n_scales, T)

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

    def test_multiscale_only_tensor_key(self):
        """Multiscale produces only the 'tensor' key (no momentum)."""
        data = make_prepared_data(
            include_positions=True,
            include_multiscale=True,
            n_scales=3,
        )
        config = TensorOperatorConfig()
        result = compute_tensor_operators(data, config)
        assert set(result.keys()) == {"tensor"}
