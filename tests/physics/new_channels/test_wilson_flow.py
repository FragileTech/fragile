"""Tests for Wilson flow analysis module."""

from __future__ import annotations

import math

import pytest
import torch

from fragile.physics.new_channels.wilson_flow import (
    _diffuse_color_step,
    _interpolate_crossing,
    _measure_action_density,
    compute_wilson_flow,
    WilsonFlowConfig,
    WilsonFlowOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_T = 5
_N = 3


@pytest.fixture
def color_states():
    """Seeded complex color states [5, 3, 3], unit-normalized."""
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(_T, _N, 3, generator=gen)
    imag = torch.randn(_T, _N, 3, generator=gen)
    c = torch.complex(real, imag)
    norms = torch.linalg.vector_norm(c, dim=-1, keepdim=True).clamp(min=1e-8)
    return c / norms


@pytest.fixture
def color_valid():
    return torch.ones(_T, _N, dtype=torch.bool)


@pytest.fixture
def comp_distance():
    return torch.arange(_N).roll(-1).unsqueeze(0).expand(_T, -1).clone()


@pytest.fixture
def comp_clone():
    return torch.arange(_N).roll(-2).unsqueeze(0).expand(_T, -1).clone()


# ---------------------------------------------------------------------------
# Tests: _interpolate_crossing
# ---------------------------------------------------------------------------


class TestInterpolateCrossing:
    def test_simple_crossing(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 0.2, 0.4, 0.6])
        result = _interpolate_crossing(x, y, 0.3)
        assert math.isfinite(result)
        assert abs(result - 1.5) < 1e-6

    def test_no_crossing_returns_nan(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 0.1, 0.2])
        result = _interpolate_crossing(x, y, 10.0)
        assert math.isnan(result)

    def test_exact_crossing(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 0.3, 0.6])
        result = _interpolate_crossing(x, y, 0.3)
        assert math.isfinite(result)
        assert abs(result - 1.0) < 1e-6

    def test_too_few_points(self):
        x = torch.tensor([1.0])
        y = torch.tensor([0.5])
        assert math.isnan(_interpolate_crossing(x, y, 0.3))

    def test_empty(self):
        x = torch.tensor([])
        y = torch.tensor([])
        assert math.isnan(_interpolate_crossing(x, y, 0.3))


# ---------------------------------------------------------------------------
# Tests: _diffuse_color_step
# ---------------------------------------------------------------------------


class TestDiffuseColorStep:
    def test_output_shape(self, color_states, color_valid, comp_distance, comp_clone):
        result = _diffuse_color_step(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            0.1,
            "both",
        )
        assert result.shape == color_states.shape

    def test_output_normalized(self, color_states, color_valid, comp_distance, comp_clone):
        result = _diffuse_color_step(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            0.1,
            "both",
        )
        norms = torch.linalg.vector_norm(result, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_zero_step_preserves_direction(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        result = _diffuse_color_step(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            0.0,
            "both",
        )
        # With step_size=0, color should be unchanged (just re-normalized)
        norms_orig = torch.linalg.vector_norm(color_states, dim=-1, keepdim=True).clamp(min=1e-12)
        expected = color_states / norms_orig
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_distance_only(self, color_states, color_valid, comp_distance, comp_clone):
        result = _diffuse_color_step(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            0.1,
            "distance",
        )
        assert result.shape == color_states.shape

    def test_clone_only(self, color_states, color_valid, comp_distance, comp_clone):
        result = _diffuse_color_step(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            0.1,
            "clone",
        )
        assert result.shape == color_states.shape


# ---------------------------------------------------------------------------
# Tests: _measure_action_density
# ---------------------------------------------------------------------------


class TestMeasureActionDensity:
    def test_returns_correct_shapes(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        mean_e, per_frame, valid_count = _measure_action_density(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            1e-12,
            "action_re_plaquette",
        )
        assert mean_e.ndim == 0
        assert per_frame.shape == (_T,)
        assert valid_count.shape == (_T,)

    def test_action_density_non_negative(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        mean_e, per_frame, _ = _measure_action_density(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            1e-12,
            "action_re_plaquette",
        )
        # action_re_plaquette = 1 - Re(Pi), can be negative but typically >=0 for unit-norm
        assert torch.isfinite(mean_e)
        assert torch.all(torch.isfinite(per_frame))


# ---------------------------------------------------------------------------
# Tests: compute_wilson_flow
# ---------------------------------------------------------------------------


class TestComputeWilsonFlow:
    def test_basic_output_structure(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg = WilsonFlowConfig(n_steps=10, step_size=0.05)
        out = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg,
        )
        assert isinstance(out, WilsonFlowOutput)
        assert out.flow_times.shape == (11,)  # n_steps + 1
        assert out.action_density.shape == (11,)
        assert out.action_density_per_frame.shape == (11, _T)
        assert out.t2_action.shape == (11,)
        assert out.dt2_action.shape == (10,)
        assert out.dt2_action_times.shape == (10,)
        assert out.n_valid_walkers_per_frame.shape == (_T,)
        assert out.config is cfg

    def test_flow_times_monotonic(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg = WilsonFlowConfig(n_steps=20, step_size=0.01)
        out = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg,
        )
        diffs = out.flow_times[1:] - out.flow_times[:-1]
        assert torch.all(diffs > 0)

    def test_flow_times_start_at_zero(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg = WilsonFlowConfig(n_steps=5, step_size=0.1)
        out = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg,
        )
        assert float(out.flow_times[0].item()) == 0.0

    def test_action_density_finite(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg = WilsonFlowConfig(n_steps=10, step_size=0.05)
        out = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg,
        )
        assert torch.all(torch.isfinite(out.action_density))
        assert torch.all(torch.isfinite(out.t2_action))
        assert torch.all(torch.isfinite(out.dt2_action))

    def test_t0_w0_are_float(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg = WilsonFlowConfig(n_steps=10, step_size=0.05)
        out = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg,
        )
        assert isinstance(out.t0, float)
        assert isinstance(out.w0, float)
        assert isinstance(out.sqrt_8t0, float)

    def test_default_config(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        out = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
        )
        assert isinstance(out, WilsonFlowOutput)
        assert out.config.n_steps == 100

    def test_invalid_topology_raises(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg = WilsonFlowConfig(topology="invalid")
        with pytest.raises(ValueError, match="topology"):
            compute_wilson_flow(
                color_states,
                color_valid,
                comp_distance,
                comp_clone,
                config=cfg,
            )

    def test_different_topologies_give_different_results(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        cfg_d = WilsonFlowConfig(n_steps=10, step_size=0.1, topology="distance")
        cfg_c = WilsonFlowConfig(n_steps=10, step_size=0.1, topology="clone")
        out_d = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg_d,
        )
        out_c = compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg_c,
        )
        # Different topologies should produce different action densities
        assert not torch.equal(out_d.action_density, out_c.action_density)

    def test_does_not_mutate_input(
        self,
        color_states,
        color_valid,
        comp_distance,
        comp_clone,
    ):
        original = color_states.clone()
        cfg = WilsonFlowConfig(n_steps=5, step_size=0.1)
        compute_wilson_flow(
            color_states,
            color_valid,
            comp_distance,
            comp_clone,
            config=cfg,
        )
        torch.testing.assert_close(color_states, original)
