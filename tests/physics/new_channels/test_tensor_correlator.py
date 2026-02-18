"""Regression tests for tensor momentum channel."""

from __future__ import annotations

import math

import torch
import pytest

from fragile.fractalai.qft.tensor_momentum_channels import (
    TensorMomentumCorrelatorConfig,
    TensorMomentumCorrelatorOutput,
    compute_tensor_momentum_correlator_from_color_positions,
    compute_companion_tensor_momentum_correlator,
    TENSOR_COMPONENT_LABELS,
    _traceless_tensor_components,
)

# New-location aliases for parity tests
from fragile.physics.new_channels.tensor_momentum_channels import (
    TensorMomentumCorrelatorConfig as NewTensorConfig,
    _traceless_tensor_components as new_traceless,
    compute_tensor_momentum_correlator_from_color_positions as new_from_color,
    compute_companion_tensor_momentum_correlator as new_companion,
)

from .conftest import MockRunHistory, assert_outputs_equal


# =============================================================================
# Layer A: Analytical / from_color tests
# =============================================================================


class TestTracelessTensorComponents:
    def test_output_shape(self):
        gen = torch.Generator().manual_seed(60)
        dx = torch.randn(4, 10, 3, generator=gen)
        result = _traceless_tensor_components(dx)
        assert result.shape == (4, 10, 5)

    def test_known_values_x_axis(self):
        # dx = [1, 0, 0]
        dx = torch.tensor([[[1.0, 0.0, 0.0]]])
        q = _traceless_tensor_components(dx)
        # q_xy = x*y = 0
        assert abs(q[0, 0, 0].item()) < 1e-7
        # q_xz = x*z = 0
        assert abs(q[0, 0, 1].item()) < 1e-7
        # q_yz = y*z = 0
        assert abs(q[0, 0, 2].item()) < 1e-7
        # q_xx_minus_yy = (x^2 - y^2) / sqrt(2) = 1/sqrt(2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        assert abs(q[0, 0, 3].item() - inv_sqrt2) < 1e-6
        # q_2zz = (2z^2 - x^2 - y^2) / sqrt(6) = (-1) / sqrt(6)
        inv_sqrt6 = 1.0 / math.sqrt(6.0)
        assert abs(q[0, 0, 4].item() - (-inv_sqrt6)) < 1e-6


class TestFromColorPositionsOutput:
    @pytest.fixture
    def tensor_output(self, tiny_color_states, tiny_color_valid, tiny_alive,
                      tiny_companions_distance, tiny_companions_clone, tiny_positions):
        T, N = tiny_color_states.shape[:2]
        gen = torch.Generator().manual_seed(70)
        positions_axis = torch.randn(T, N, generator=gen)
        return compute_tensor_momentum_correlator_from_color_positions(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            positions=tiny_positions,
            positions_axis=positions_axis,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            pair_selection="both",
            eps=1e-12,
            momentum_mode_max=2,
            projection_length=10.0,
            bounds=None,
            pbc=False,
            compute_bootstrap_errors=False,
            n_bootstrap=0,
        )

    def test_output_type(self, tensor_output):
        assert isinstance(tensor_output, TensorMomentumCorrelatorOutput)

    def test_momentum_correlator_shape(self, tensor_output):
        # [n_modes, 5, max_lag+1] = [3, 5, 4]
        assert tensor_output.momentum_correlator.shape == (3, 5, 4)

    def test_momentum_contracted_shape(self, tensor_output):
        # [n_modes, max_lag+1] = [3, 4]
        assert tensor_output.momentum_contracted_correlator.shape == (3, 4)

    def test_component_series_shape(self, tensor_output):
        # [T, 5]
        assert tensor_output.component_series.shape == (5, 5)

    def test_component_labels(self, tensor_output):
        assert tensor_output.component_labels == TENSOR_COMPONENT_LABELS

    def test_momentum_modes_shape(self, tensor_output):
        # momentum_mode_max=2 -> 3 modes
        assert tensor_output.momentum_modes.shape == (3,)

    def test_regression_finite(self, tensor_output):
        assert torch.isfinite(tensor_output.momentum_correlator).all()
        assert torch.isfinite(tensor_output.momentum_contracted_correlator).all()
        assert torch.isfinite(tensor_output.component_series).all()


class TestFromColorEmpty:
    def test_empty_input(self):
        color = torch.zeros(0, 3, 3, dtype=torch.complex64)
        valid = torch.zeros(0, 3, dtype=torch.bool)
        alive = torch.zeros(0, 3, dtype=torch.bool)
        pos = torch.zeros(0, 3, 3)
        pos_axis = torch.zeros(0, 3)
        comp_d = torch.zeros(0, 3, dtype=torch.long)
        comp_c = torch.zeros(0, 3, dtype=torch.long)
        out = compute_tensor_momentum_correlator_from_color_positions(
            color=color, color_valid=valid, positions=pos,
            positions_axis=pos_axis, alive=alive,
            companions_distance=comp_d, companions_clone=comp_c,
            max_lag=5, use_connected=True, pair_selection="both",
            eps=1e-12, momentum_mode_max=2, projection_length=10.0,
            bounds=None, pbc=False,
            compute_bootstrap_errors=False, n_bootstrap=0,
        )
        assert out.momentum_correlator.shape[2] == 6  # max_lag+1
        assert out.n_valid_source_pairs == 0


# =============================================================================
# Layer B: Integration with MockRunHistory
# =============================================================================


class TestCompanionTensor:
    @pytest.fixture
    def config(self):
        return TensorMomentumCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
            momentum_mode_max=2,
            projection_length=10.0,
        )

    @pytest.fixture
    def output(self, mock_history, config):
        return compute_companion_tensor_momentum_correlator(mock_history, config)

    def test_runs_without_error(self, output):
        assert isinstance(output, TensorMomentumCorrelatorOutput)

    def test_output_shapes(self, output):
        # [n_modes, 5, max_lag+1] = [3, 5, 11]
        assert output.momentum_correlator.shape == (3, 5, 11)
        assert output.momentum_contracted_correlator.shape == (3, 11)

    def test_regression_finite(self, output):
        assert torch.isfinite(output.momentum_correlator.sum())
        assert torch.isfinite(output.momentum_contracted_correlator.sum())

    def test_bootstrap_errors(self, mock_history):
        cfg = TensorMomentumCorrelatorConfig(
            max_lag=5, ell0=1.0, momentum_mode_max=1,
            projection_length=10.0,
            compute_bootstrap_errors=True, n_bootstrap=10,
        )
        out = compute_companion_tensor_momentum_correlator(mock_history, cfg)
        assert out.momentum_correlator_err is not None
        assert out.momentum_contracted_correlator_err is not None
        assert out.momentum_correlator_err.shape == out.momentum_correlator.shape
        assert out.momentum_contracted_correlator_err.shape == out.momentum_contracted_correlator.shape


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityTensor:
    """Verify new-location tensor functions produce identical results to originals."""

    def test_traceless_tensor_parity(self):
        gen = torch.Generator().manual_seed(300)
        dx = torch.randn(4, 10, 3, generator=gen)
        old = _traceless_tensor_components(dx)
        new = new_traceless(dx)
        assert torch.equal(old, new)

    def test_from_color_parity(self, tiny_color_states, tiny_color_valid, tiny_alive,
                               tiny_companions_distance, tiny_companions_clone, tiny_positions):
        T, N = tiny_color_states.shape[:2]
        gen = torch.Generator().manual_seed(70)
        positions_axis = torch.randn(T, N, generator=gen)
        kwargs = dict(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            positions=tiny_positions,
            positions_axis=positions_axis,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            pair_selection="both",
            eps=1e-12,
            momentum_mode_max=2,
            projection_length=10.0,
            bounds=None,
            pbc=False,
            compute_bootstrap_errors=False,
            n_bootstrap=0,
        )
        old_out = compute_tensor_momentum_correlator_from_color_positions(**kwargs)
        new_out = new_from_color(**kwargs)
        assert_outputs_equal(old_out, new_out)

    def test_companion_parity(self, mock_history):
        cfg_old = TensorMomentumCorrelatorConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
            momentum_mode_max=2, projection_length=10.0,
        )
        cfg_new = NewTensorConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
            momentum_mode_max=2, projection_length=10.0,
        )
        old_out = compute_companion_tensor_momentum_correlator(mock_history, cfg_old)
        new_out = new_companion(mock_history, cfg_new)
        assert_outputs_equal(old_out, new_out)
