"""Regression tests for twistor-inspired companion channels."""

from __future__ import annotations

import pytest
import torch

from fragile.physics.new_channels.twistor_companion_channels import (
    TwistorCompanionCorrelatorConfig,
    TwistorCompanionCorrelatorOutput,
    compute_companion_twistor_correlator,
    compute_twistor_companion_correlator_from_geometry,
)


class TestFromGeometryOutput:
    @pytest.fixture
    def output(self, tiny_positions, tiny_companions_distance, tiny_companions_clone, tiny_alive):
        velocities = 0.5 * tiny_positions + 0.25
        return compute_twistor_companion_correlator_from_geometry(
            positions=tiny_positions,
            velocities=velocities,
            alive_mask=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            delta_t=1.0,
            max_lag=3,
            use_connected=True,
        )

    def test_output_type(self, output):
        assert isinstance(output, TwistorCompanionCorrelatorOutput)

    def test_shapes(self, output):
        assert output.scalar.shape == (4,)
        assert output.pseudoscalar.shape == (4,)
        assert output.glueball.shape == (4,)
        assert output.vector.shape == (4,)
        assert output.axial_vector.shape == (4,)
        assert output.tensor.shape == (4,)
        assert output.counts.shape == (4,)

    def test_operator_series_shape(self, output):
        assert output.operator_scalar_series.shape == (5,)
        assert output.operator_pseudoscalar_series.shape == (5,)
        assert output.operator_glueball_series.shape == (5,)
        assert output.operator_vector_series.shape == (5, 3)
        assert output.operator_axial_vector_series.shape == (5, 3)
        assert output.operator_tensor_series.shape == (5,)

    def test_outputs_are_finite(self, output):
        assert bool(torch.isfinite(output.scalar).all())
        assert bool(torch.isfinite(output.pseudoscalar).all())
        assert bool(torch.isfinite(output.glueball).all())
        assert bool(torch.isfinite(output.vector).all())
        assert bool(torch.isfinite(output.axial_vector).all())
        assert bool(torch.isfinite(output.tensor).all())
        assert bool(torch.isfinite(output.operator_glueball_series).all())
        assert bool(torch.isfinite(output.operator_vector_series).all())
        assert bool(torch.isfinite(output.operator_axial_vector_series).all())
        assert bool(torch.isfinite(output.operator_tensor_series).all())
        assert bool((output.operator_glueball_series >= 0).all())

    def test_counts_positive_at_zero_lag(self, output):
        assert output.counts[0].item() > 0


class TestFromGeometryEmpty:
    def test_empty_input(self):
        positions = torch.zeros(0, 3, 3, dtype=torch.float32)
        velocities = torch.zeros(0, 3, 3, dtype=torch.float32)
        alive = torch.zeros(0, 3, dtype=torch.bool)
        comp_d = torch.zeros(0, 3, dtype=torch.long)
        comp_c = torch.zeros(0, 3, dtype=torch.long)
        out = compute_twistor_companion_correlator_from_geometry(
            positions=positions,
            velocities=velocities,
            alive_mask=alive,
            companions_distance=comp_d,
            companions_clone=comp_c,
            delta_t=1.0,
            max_lag=5,
        )
        assert out.scalar.shape == (6,)
        assert out.pseudoscalar.shape == (6,)
        assert out.glueball.shape == (6,)
        assert out.vector.shape == (6,)
        assert out.axial_vector.shape == (6,)
        assert out.tensor.shape == (6,)
        assert out.n_valid_source_triplets == 0


class TestCompanionTwistor:
    @pytest.fixture
    def config(self):
        return TwistorCompanionCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            velocity_scale=0.5,
        )

    @pytest.fixture
    def output(self, mock_history, config):
        return compute_companion_twistor_correlator(mock_history, config)

    def test_runs_without_error(self, output):
        assert isinstance(output, TwistorCompanionCorrelatorOutput)

    def test_output_shapes(self, output):
        assert output.scalar.shape == (11,)
        assert output.pseudoscalar.shape == (11,)
        assert output.glueball.shape == (11,)
        assert output.vector.shape == (11,)
        assert output.axial_vector.shape == (11,)
        assert output.tensor.shape == (11,)
        assert output.counts.shape == (11,)

    def test_finite_outputs(self, output):
        assert torch.isfinite(output.scalar.sum())
        assert torch.isfinite(output.pseudoscalar.sum())
        assert torch.isfinite(output.glueball.sum())
        assert torch.isfinite(output.vector.sum())
        assert torch.isfinite(output.axial_vector.sum())
        assert torch.isfinite(output.tensor.sum())

    def test_velocity_scale_changes_operator(self, mock_history):
        cfg_lo = TwistorCompanionCorrelatorConfig(max_lag=5, velocity_scale=0.25)
        cfg_hi = TwistorCompanionCorrelatorConfig(max_lag=5, velocity_scale=1.25)
        out_lo = compute_companion_twistor_correlator(mock_history, cfg_lo)
        out_hi = compute_companion_twistor_correlator(mock_history, cfg_hi)
        assert not torch.allclose(out_lo.glueball, out_hi.glueball, atol=1e-10)
