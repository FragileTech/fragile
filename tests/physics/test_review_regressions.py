"""Numerical regression tests for the make-physics correctness review."""

from types import SimpleNamespace
from unittest.mock import patch

import gvar
import numpy as np
import pytest
import torch

from fragile.physics.fractal_gas.cloning import CloneOperator
from fragile.physics.fractal_gas.euclidean_gas import EuclideanGas, SwarmState
from fragile.physics.fractal_gas.fitness import FitnessOperator
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator
from fragile.physics.geometry.delaunai import compute_delaunay_data
from fragile.physics.mass_extraction.config import CovarianceConfig, MassExtractionConfig
from fragile.physics.mass_extraction.data_preparation import correlators_to_gvar
from fragile.physics.mass_extraction.pipeline import extract_masses
from fragile.physics.new_channels.correlator_channels import CorrelatorConfig, extract_mass_aic
from fragile.physics.new_channels.dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
    compute_dirac_operators_from_spinors,
)
from fragile.physics.new_channels.meson_phase_channels import (
    compute_meson_phase_correlator_from_color,
)
from fragile.physics.operators.config import TensorOperatorConfig
from fragile.physics.operators.pipeline import PipelineResult
from fragile.physics.operators.tensor_operators import compute_tensor_correlator
from fragile.physics.qft_utils.color_states import compute_color_states_batch
from fragile.physics.qft_utils.helpers import recorded_time_step, resolve_frame_indices
from fragile.physics.qft_utils.statistics import (
    attach_statistics,
    sample_covariance,
    series_statistics,
    stack_correlators,
)


def kinetic(**params):
    return KineticOperator(gamma=1, beta=1, delta_t=0.01, **params)


def gas(**params):
    op = kinetic(beta_curl=0)
    op.n_kinetic_steps = 1
    return EuclideanGas(
        N=20, d=3, kinetic_op=op, cloning=CloneOperator(), fitness_op=FitnessOperator(), **params
    )


def test_b_step_duration():
    op = kinetic(beta_curl=0)
    state = SwarmState(torch.zeros(2, 3), torch.zeros(2, 3))
    with (
        patch.object(op, "_compute_viscous_force", return_value=torch.ones(2, 3)),
        patch("torch.randn", side_effect=lambda *a, **kw: torch.zeros(*a, **kw)),
    ):
        result = op.apply(state)
    expected = 0.5 * op.dt * (1 + op.c1)
    torch.testing.assert_close(result.v, torch.ones_like(result.v) * expected)


def test_skipped_clones_and_chunk_labels():
    simulator = gas(clone_every=20)
    history = simulator.run(12, seed=42, chunk_size=10)
    assert history.n_recorded == 13
    assert history.recorded_steps == list(range(13))
    assert not history.will_clone.any()
    assert not history.clone_delta_x.any()
    assert not history.clone_delta_v.any()
    torch.testing.assert_close(history.x_before_clone[1:], history.x_after_clone)
    fitness, _ = simulator.fitness_op(
        history.x_before_clone[1], history.rewards[0], history.companions_distance[0]
    )
    torch.testing.assert_close(history.fitness[0], fitness)


def test_viscosity_controls_and_default_composition():
    simulator = gas()
    history = simulator.run(2, seed=2)
    assert not history.terminated_early
    op = simulator.kinetic_op
    op.use_viscous_coupling = False
    force = op._compute_viscous_force(
        torch.randn(2, 3), torch.randn(2, 3), torch.tensor([[0, 1], [1, 0]]), None
    )
    assert not force.any()
    simulator.kinetic_op.use_viscous_coupling = True
    simulator.kinetic_op.viscous_neighbor_weighting = "kernel"
    x = torch.randn(20, 3)
    simulator.kinetic_op.viscous_length_scale = 0.3
    a = simulator._compute_tessellation(x)["edge_weights"]
    simulator.kinetic_op.viscous_length_scale = 3
    b = simulator._compute_tessellation(x)["edge_weights"]
    assert not torch.allclose(a, b)


def test_plain_baoab_has_no_rotation():
    op = kinetic(
        nu=0,
        beta_curl=1,
        curl_field=lambda x: torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]).expand(len(x), 3, 3),
    )
    state = SwarmState(torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    op.integrator = "baoab"
    torch.manual_seed(1)
    a = op.apply(state)
    op.integrator = "boris-baoab"
    torch.manual_seed(1)
    b = op.apply(state)
    assert not torch.allclose(a.v, b.v)


def test_duplicate_particles_are_equivalent():
    torch.manual_seed(17)
    x = torch.randn(20, 3)
    x[1] = x[0]
    geometry = compute_delaunay_data(x, torch.zeros(20), spatial_dims=2)
    degree = torch.bincount(geometry.edge_index[0], minlength=20)
    assert (degree > 0).all()
    torch.testing.assert_close(geometry.metric_tensors[0], geometry.metric_tensors[1])
    torch.testing.assert_close(geometry.ricci_proxy[0], geometry.ricci_proxy[1])


def test_duplicate_sites_in_rank_deficient_swarm():
    from fragile.physics.geometry.delaunai import build_delaunay_edges

    edges = build_delaunay_edges(np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    assert set(map(tuple, edges)) == {
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    }


def test_metadata_time_and_color_stage():
    simulator = gas()
    simulator.kinetic_op.n_kinetic_steps = 3
    simulator.kinetic_op.auto_thermostat = True
    simulator.kinetic_op.temperature = 0.33
    h = simulator.run(12, record_every=5, seed=3)
    assert h.params["kinetic"]["temperature"] == 0.33
    assert h.params["kinetic"]["auto_thermostat"]
    assert recorded_time_step(h) == pytest.approx(0.15)
    assert resolve_frame_indices(h, 0, 1) == [1, 2]
    color, _ = compute_color_states_batch(h, 1, 1, 1, 1)
    v = h.v_after_clone
    force = h.force_viscous
    expected = force.to(torch.complex64) * torch.polar(torch.ones_like(v), v)
    expected /= expected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    torch.testing.assert_close(color, expected)


@pytest.fixture
def paired_colors():
    torch.manual_seed(24)
    color = torch.randn(100, 4, 3, dtype=torch.complex128)
    color /= color.norm(dim=-1, keepdim=True)
    pairs = torch.tensor([1, 0, 3, 2]).expand(100, 4)
    return color, torch.ones(100, 4, dtype=torch.bool), pairs


@pytest.mark.parametrize("method", ["uncorrelated", "block_jackknife", "bootstrap"])
def test_pair_correlator_survives_covariance(paired_colors, method):
    c, valid, pairs = paired_colors
    out = compute_meson_phase_correlator_from_color(c, valid, pairs, pairs, max_lag=5)
    assert out.pseudoscalar[0] > 0.01
    assert out.operator_pseudoscalar_series.abs().max() < 1e-7
    cfg = MassExtractionConfig(covariance=CovarianceConfig(method=method, n_bootstrap=40))
    converted = correlators_to_gvar({"pseudoscalar": out.pseudoscalar}, config=cfg)
    np.testing.assert_allclose(gvar.mean(converted["pseudoscalar"]), out.pseudoscalar.numpy())
    assert gvar.sdev(converted["pseudoscalar"][0]) > 0


def test_replica_covariance_normalization():
    samples = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 9.0]])
    np.testing.assert_allclose(sample_covariance(samples, "bootstrap"), np.cov(samples.T))
    centered = samples - samples.mean(0)
    np.testing.assert_allclose(
        sample_covariance(samples, "block_jackknife"), 2 / 3 * centered.T @ centered
    )


def test_joint_covariance_and_component_contraction():
    torch.manual_seed(4)
    a = torch.randn(100, dtype=torch.float64)
    vector = torch.stack([a, -a], -1)
    stats = series_statistics(vector, 5)
    corr = stats.mean()
    cfg = MassExtractionConfig(covariance=CovarianceConfig(method="block_jackknife"))
    out = correlators_to_gvar({"a": corr, "b": corr}, {"a": vector, "b": vector}, cfg)
    assert gvar.mean(out["a"][0]) > 0
    assert gvar.evalcorr([out["a"][0], out["b"][0]])[0, 1] == pytest.approx(1)


def test_missing_statistics_are_not_silently_replaced():
    with pytest.raises(ValueError, match="required"):
        correlators_to_gvar({"scalar": torch.arange(5.0)}, {"scalar": torch.randn(100)})


def test_multiscale_fit_keys():
    # Explicit assumed errors are appropriate for this exact synthetic curve:
    # this test checks model construction, not covariance estimation.
    corr = torch.exp(-0.3 * torch.arange(20.0))
    result = extract_masses(
        PipelineResult(correlators={"scalar": stack_correlators([corr, corr])}),
        MassExtractionConfig(covariance=CovarianceConfig(method="assumed_relative")),
    )
    assert set(result.channels["scalar"].variant_keys) == {"scalar_scale_0", "scalar_scale_1"}


def test_tensor_pair_products_do_not_cancel(paired_colors):
    color, valid, pairs = paired_colors
    data = SimpleNamespace(
        color=color, color_valid=valid, companions_distance=pairs, companions_clone=pairs
    )
    corr = compute_tensor_correlator(data, TensorOperatorConfig(), 5)
    assert corr[0] > 0.01
    assert hasattr(corr, "correlator_statistics")
    data.scales = torch.tensor([0.5, 2.0])
    data.pairwise_distances = torch.ones(len(color), 4, 4)
    multiscale = compute_tensor_correlator(data, TensorOperatorConfig(), 5)
    assert multiscale.shape == (2, 6)
    assert not multiscale[0].any()
    torch.testing.assert_close(multiscale[1], corr)
    assert len(multiscale.correlator_statistics) == 2


def test_dirac_end_to_end_parity(paired_colors):
    c, valid, pairs = paired_colors
    psi, ok = color_to_dirac_spinor(c)
    inverted, _ = color_to_dirac_spinor(-c.conj())
    gamma = build_dirac_gamma_matrices()
    expected = torch.einsum("ab,...b->...a", gamma["gamma0"], psi)
    torch.testing.assert_close(inverted, expected)
    sample = torch.arange(4).expand(len(c), 4)
    original = compute_dirac_operators_from_spinors(psi, ok, sample, pairs, valid, gamma)
    parity = compute_dirac_operators_from_spinors(inverted, ok, sample, pairs, valid, gamma)
    assert original.pseudoscalar.abs().max() > 0.01
    torch.testing.assert_close(parity.pseudoscalar, -original.pseudoscalar)


@pytest.mark.parametrize("dt", [0.1, 0.4, 1.0])
def test_aic_units_and_explicit_uncertainty(dt):
    corr = torch.exp(-0.3 * dt * torch.arange(30.0))
    result = extract_mass_aic(corr, dt, CorrelatorConfig(window_widths=[5]))
    assert result["mass"] == pytest.approx(0.3, abs=1e-5)
    assert np.isnan(result["mass_error"])
    assert result["uncertainty_method"] == "unavailable"
    measured = extract_mass_aic(corr, dt, CorrelatorConfig(window_widths=[5]), corr * 0.02)
    assert measured["mass_error"] > 0
