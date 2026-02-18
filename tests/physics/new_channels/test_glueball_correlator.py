"""Regression tests for glueball color channel."""

from __future__ import annotations

import torch
import pytest

from fragile.fractalai.qft.glueball_color_channels import (
    GlueballColorCorrelatorConfig,
    GlueballColorCorrelatorOutput,
    compute_glueball_color_correlator_from_color,
    compute_companion_glueball_color_correlator,
    _glueball_observable_from_plaquette,
    _compute_color_plaquette_for_triplets,
)

# New-location aliases for parity tests
from fragile.physics.new_channels.glueball_color_channels import (
    GlueballColorCorrelatorConfig as NewGlueballConfig,
    _glueball_observable_from_plaquette as new_glueball_obs,
    compute_glueball_color_correlator_from_color as new_from_color,
    compute_companion_glueball_color_correlator as new_companion,
)

from .conftest import MockRunHistory, assert_outputs_equal


# =============================================================================
# Layer A: Analytical / from_color tests
# =============================================================================


class TestGlueballObservable:
    def test_re_plaquette(self):
        pi = torch.tensor([1.0 + 0.5j, 0.0 + 1.0j], dtype=torch.complex64)
        obs = _glueball_observable_from_plaquette(pi, operator_mode="re_plaquette")
        expected = pi.real.float()
        assert torch.allclose(obs, expected)

    def test_action_re_plaquette(self):
        pi = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
        obs = _glueball_observable_from_plaquette(pi, operator_mode="action_re_plaquette")
        expected = (1.0 - pi.real).float()
        assert torch.allclose(obs, expected)


class TestFromColorOutput:
    @pytest.fixture
    def glueball_output(self, tiny_color_states, tiny_color_valid, tiny_alive,
                        tiny_companions_distance, tiny_companions_clone):
        return compute_glueball_color_correlator_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            eps=1e-12,
        )

    def test_output_type(self, glueball_output):
        assert isinstance(glueball_output, GlueballColorCorrelatorOutput)

    def test_correlator_shape(self, glueball_output):
        assert glueball_output.correlator.shape == (4,)

    def test_counts_positive(self, glueball_output):
        assert glueball_output.counts[0].item() > 0

    def test_no_momentum_fields(self, glueball_output):
        assert glueball_output.momentum_modes is None
        assert glueball_output.momentum_correlator is None

    def test_regression_finite(self, glueball_output):
        assert torch.isfinite(glueball_output.correlator).all()
        assert torch.isfinite(glueball_output.operator_glueball_series).all()


class TestFromColorMomentum:
    def test_momentum_fields_present(self, tiny_color_states, tiny_color_valid, tiny_alive,
                                      tiny_companions_distance, tiny_companions_clone):
        gen = torch.Generator().manual_seed(77)
        T, N = tiny_color_states.shape[:2]
        positions_axis = torch.randn(T, N, generator=gen)
        out = compute_glueball_color_correlator_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_momentum_projection=True,
            positions_axis=positions_axis,
            momentum_mode_max=2,
            projection_length=10.0,
        )
        assert out.momentum_modes is not None
        assert out.momentum_correlator is not None
        # Shape: [n_modes, max_lag+1]
        assert out.momentum_correlator.shape == (3, 4)  # n_modes=3, max_lag+1=4


class TestFromColorEmpty:
    def test_empty_input(self):
        color = torch.zeros(0, 3, 3, dtype=torch.complex64)
        valid = torch.zeros(0, 3, dtype=torch.bool)
        alive = torch.zeros(0, 3, dtype=torch.bool)
        comp_d = torch.zeros(0, 3, dtype=torch.long)
        comp_c = torch.zeros(0, 3, dtype=torch.long)
        out = compute_glueball_color_correlator_from_color(
            color=color, color_valid=valid, alive=alive,
            companions_distance=comp_d, companions_clone=comp_c,
            max_lag=5,
        )
        assert out.correlator.shape == (6,)
        assert out.n_valid_source_triplets == 0


# =============================================================================
# Layer B: Integration with MockRunHistory
# =============================================================================


class TestCompanionGlueball:
    @pytest.fixture
    def config(self):
        return GlueballColorCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
        )

    @pytest.fixture
    def output(self, mock_history, config):
        return compute_companion_glueball_color_correlator(mock_history, config)

    def test_runs_without_error(self, output):
        assert isinstance(output, GlueballColorCorrelatorOutput)

    def test_output_shapes(self, output):
        assert output.correlator.shape == (11,)
        assert output.counts.shape == (11,)

    def test_regression_finite(self, output):
        assert torch.isfinite(output.correlator.sum())

    def test_action_form_differs(self, mock_history):
        cfg_re = GlueballColorCorrelatorConfig(
            max_lag=5, ell0=1.0, use_action_form=False,
        )
        cfg_act = GlueballColorCorrelatorConfig(
            max_lag=5, ell0=1.0, use_action_form=True,
        )
        out_re = compute_companion_glueball_color_correlator(mock_history, cfg_re)
        out_act = compute_companion_glueball_color_correlator(mock_history, cfg_act)
        # Connected correlators are identical for Re(pi) vs 1-Re(pi) (same variance),
        # so compare raw correlators instead.
        assert not torch.allclose(out_re.correlator_raw, out_act.correlator_raw, atol=1e-10)


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityGlueball:
    """Verify new-location glueball functions produce identical results to originals."""

    def test_glueball_observable_parity(self):
        pi = torch.tensor([1.0 + 0.5j, 0.0 + 1.0j, 0.5 + 0.3j], dtype=torch.complex64)
        for mode in ("re_plaquette", "action_re_plaquette"):
            old = _glueball_observable_from_plaquette(pi, operator_mode=mode)
            new = new_glueball_obs(pi, operator_mode=mode)
            assert torch.equal(old, new), f"glueball observable differs for mode={mode}"

    def test_from_color_parity(self, tiny_color_states, tiny_color_valid, tiny_alive,
                               tiny_companions_distance, tiny_companions_clone):
        kwargs = dict(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            eps=1e-12,
        )
        old_out = compute_glueball_color_correlator_from_color(**kwargs)
        new_out = new_from_color(**kwargs)
        assert_outputs_equal(old_out, new_out)

    def test_companion_parity(self, mock_history):
        cfg_old = GlueballColorCorrelatorConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
        )
        cfg_new = NewGlueballConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
        )
        old_out = compute_companion_glueball_color_correlator(mock_history, cfg_old)
        new_out = new_companion(mock_history, cfg_new)
        assert_outputs_equal(old_out, new_out)
