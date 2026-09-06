"""Regression tests for the second make-physics review round.

Each test pins one defect found while exercising the dashboard end to end
(simulation core, correlator statistics, fitting, and tab wiring).
"""

from __future__ import annotations

import math

import gvar
import numpy as np
import pytest
import torch

from fragile.physics.fractal_gas.cloning import CloneOperator
from fragile.physics.fractal_gas.euclidean_gas import EuclideanGas
from fragile.physics.fractal_gas.fitness import FitnessOperator, patched_standardization
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator
from fragile.physics.geometry.neighbors import compute_companion_batch
from fragile.physics.geometry.ricci import compute_ricci_proxy_full_metric
from fragile.physics.mass_extraction import extract_masses, MassExtractionConfig
from fragile.physics.mass_extraction.config import ChannelFitConfig, ChannelGroupConfig
from fragile.physics.mass_extraction.priors import _estimate_ground_energy
from fragile.physics.new_channels.correlator_channels import (
    ConvolutionalAICExtractor,
    CorrelatorConfig,
    extract_mass_aic,
)
from fragile.physics.operators.pipeline import PipelineResult
from fragile.physics.qft_utils import _fft_correlator_batched
from fragile.physics.qft_utils.statistics import (
    correlators_with_statistics,
    series_statistics,
)


def _gas(**params) -> EuclideanGas:
    op = KineticOperator(gamma=1, beta=1, delta_t=0.002, nu=3.0, beta_curl=1.0, temperature=0.33)
    op.auto_thermostat = True
    op.n_kinetic_steps = 1
    op.viscous_neighbor_weighting = "riemannian_kernel_volume"
    return EuclideanGas(
        N=24,
        d=3,
        kinetic_op=op,
        cloning=CloneOperator(sigma_x=0.0, alpha_restitution=1.0, epsilon_clone=0.0),
        fitness_op=FitnessOperator(sigma_min=0.0),
        clone_every=20,
        neighbor_weight_modes=[
            "inverse_riemannian_distance",
            "kernel",
            "riemannian_kernel_volume",
        ],
        **params,
    )


def _dashboard_run(steps: int = 25, seed: int = 5, **params):
    gas = _gas(**params)
    zeros = torch.zeros(24, 3)
    return gas.run(steps, x_init=zeros, v_init=zeros, seed=seed)


# --------------------------------------------------------------------------
# Simulation core
# --------------------------------------------------------------------------


def test_seeded_runs_with_curl_are_reproducible():
    a = _dashboard_run()
    b = _dashboard_run()
    assert torch.equal(a.x_final, b.x_final)
    assert torch.equal(a.v_final, b.v_final)


def test_constant_channel_gives_zero_not_nan_z_scores():
    z = patched_standardization(torch.zeros(8), sigma_min=0.0)
    assert torch.equal(z, torch.zeros(8))
    history = _dashboard_run(steps=6)
    assert torch.isfinite(history.z_rewards).all()
    assert torch.isfinite(history.z_distances).all()


def test_cached_graph_is_built_on_the_first_step():
    history = _dashboard_run(steps=4, neighbor_graph_update_every=3)
    # Frame 0 is the initial-state placeholder; every simulated step has a graph.
    assert all(edges.shape[0] > 0 for edges in history.neighbor_edges[1:])


def test_history_reports_requested_steps():
    history = _dashboard_run(steps=7)
    assert history.n_steps == 7
    assert history.final_step == 7
    assert "0.0" in history.summary() or "s/step" in history.summary()


def test_kinetic_operator_accepts_all_of_its_parameters():
    op = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        auto_thermostat=True,
        n_kinetic_steps=2,
        integrator="baoab",
    )
    assert op.auto_thermostat is True
    assert op.n_kinetic_steps == 2
    assert op.integrator == "baoab"


def test_companion_batch_does_not_fill_with_walker_zero():
    history = _dashboard_run(steps=3)
    _, neighbors, _ = compute_companion_batch(history, start_idx=1, neighbor_k=4)
    # Columns beyond the two recorded companions fall back to the sample itself.
    samples = torch.arange(history.N).expand(neighbors.shape[0], -1)
    assert torch.equal(neighbors[..., 2], samples)
    assert torch.equal(neighbors[..., 3], samples)


def test_ricci_scalar_matches_conformally_flat_formula():
    torch.manual_seed(0)
    n = 400
    positions = torch.rand(n, 3, dtype=torch.float64) * 2 - 1
    k = 0.7
    u = k * positions[:, 0]
    metric = torch.exp(2 * u)[:, None, None] * torch.eye(3, dtype=torch.float64)
    knn = torch.cdist(positions, positions).topk(13, largest=False).indices[:, 1:]
    src = torch.arange(n).repeat_interleave(12)
    edge_index = torch.stack([src, knn.reshape(-1)])
    scalar = compute_ricci_proxy_full_metric(positions, metric, edge_index)
    # For g = e^{2u} delta with u linear: R = -2(d-1) e^{-2u} (d-2)/2 |grad u|^2.
    expected = -2.0 * 2.0 * torch.exp(-2 * u) * 0.5 * 1.0 * k**2
    interior = positions.abs().max(dim=1).values < 0.6
    ratio = (scalar[interior] / expected[interior]).median()
    assert 0.8 < float(ratio) < 1.25


def test_boris_rotation_angle_is_recorded():
    op = KineticOperator(gamma=1, beta=1, delta_t=0.01, nu=0.0, beta_curl=1.0)
    op.curl_field = lambda x: torch.tensor([[0.0, 0.0, 3.0]]).expand(len(x), 3)
    from fragile.physics.fractal_gas.euclidean_gas import SwarmState

    state = SwarmState(torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    _, info = op.apply(state, return_info=True)
    angle = info["boris_rotation_angle"]
    expected = 2 * math.atan(0.5 * 1.0 * (0.01 / 2) * 3.0)
    torch.testing.assert_close(angle, torch.full((2,), expected), rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# Correlator statistics
# --------------------------------------------------------------------------


def test_correlators_with_statistics_match_fft_estimator():
    torch.manual_seed(3)
    series = torch.randn(3, 200, dtype=torch.float64)
    fft = _fft_correlator_batched(series, max_lag=20, use_connected=True)
    stats = correlators_with_statistics(series, 20, True, torch.float64)
    for b in range(3):
        torch.testing.assert_close(stats[b], fft[b], rtol=1e-8, atol=1e-10)
        assert stats[b].correlator_statistics.sums.shape == (200, 21)
    exact = series_statistics(series[1], 20, True)
    torch.testing.assert_close(stats[1].correlator_statistics.sums, exact.sums)


# --------------------------------------------------------------------------
# AIC window fits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", [1.0, 1e4])
def test_exact_exponential_has_zero_chi2_in_every_window(amplitude):
    t = torch.arange(81, dtype=torch.float64)
    corr = amplitude * torch.exp(-0.3 * t)
    result = extract_mass_aic(corr, 1.0, CorrelatorConfig(window_widths=[5, 10, 20]), corr * 0.02)
    assert result["mass"] == pytest.approx(0.3, rel=1e-9)
    aic = result["window_aic"]
    finite = aic[torch.isfinite(aic)]
    # chi2 = 0 everywhere, so AIC = 4 + 2 * (points excluded from the window).
    assert finite.min().item() == pytest.approx(4 + 2 * (81 - 20), abs=1e-6)
    assert result["best_window"]["width"] == 20
    assert math.isfinite(result["best_window"]["mass_error"])
    assert result["window_spread"] < 1e-9


def test_windows_without_signal_are_excluded():
    extractor = ConvolutionalAICExtractor(window_widths=[5])
    log_corr = torch.zeros(20, dtype=torch.float64)
    log_err = torch.full((20,), 0.1, dtype=torch.float64)
    log_err[10:] = 2.0  # relative error 200%: not resolved from zero
    out = extractor._fit_single_width_full(log_corr.view(1, 1, -1), log_err.view(1, 1, -1), 5)
    valid = out["valid"].flatten()
    assert valid[:6].all()
    assert not valid[6:].any()


def test_ar1_mass_and_error_are_calibrated():
    torch.manual_seed(0)
    m, T = 0.3, 4000
    rho = math.exp(-m)
    pulls, masses = [], []
    for _ in range(12):
        noise = torch.randn(T, dtype=torch.float64)
        x = torch.zeros(T, dtype=torch.float64)
        for i in range(1, T):
            x[i] = rho * x[i - 1] + noise[i]
        corr = correlators_with_statistics(x.unsqueeze(0), 30, True, torch.float64)[0]
        fit = extract_mass_aic(corr, 1.0, CorrelatorConfig())
        masses.append(fit["mass"])
        pulls.append((fit["mass"] - m) / fit["mass_error"])
    assert abs(float(np.mean(masses)) - m) < 0.05
    assert float(np.std(pulls)) < 2.5


def test_sparse_statistics_degrade_instead_of_raising():
    from fragile.physics.qft_utils.statistics import CorrelatorStatistics

    t = torch.arange(6, dtype=torch.float64)
    corr = torch.exp(-0.3 * t)
    # Eight origins, but only origin 0 carries data beyond lag 0: every jackknife
    # replica that drops that origin has no support at lags >= 1.
    counts = torch.zeros(8, 6, dtype=torch.float64)
    counts[:, 0] = 1
    counts[0, 1:] = 1
    corr.correlator_statistics = CorrelatorStatistics(corr.expand(8, 6) * counts, counts)
    result = extract_mass_aic(corr, 1.0, CorrelatorConfig(window_widths=[5]))
    assert result["uncertainty_method"] == "unavailable"
    assert "uncertainty_note" in result
    assert result["mass"] == pytest.approx(0.3, abs=1e-6)


# --------------------------------------------------------------------------
# Bayesian fits
# --------------------------------------------------------------------------


def _exact_gvar(mass: float, amplitude: float = 1.0, n: int = 41):
    t = np.arange(n)
    means = amplitude * np.exp(-mass * t)
    return {"scalar": gvar.gvar(means, 0.02 * means)}


def test_fastfit_seeding_returns_an_estimate():
    energy = _estimate_ground_energy(_exact_gvar(1.0), "scalar", 2, None)
    assert energy is not None
    assert gvar.mean(energy) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("amplitude", [1.0, 1e4])
def test_bayesian_fit_recovers_mass_for_any_amplitude(amplitude):
    torch.manual_seed(1)
    m, T = 0.3, 3000
    rho = math.exp(-m)
    noise = torch.randn(T, dtype=torch.float64)
    x = torch.zeros(T, dtype=torch.float64)
    for i in range(1, T):
        x[i] = rho * x[i - 1] + noise[i]
    x = x * math.sqrt(amplitude)
    corr = correlators_with_statistics(x.unsqueeze(0), 25, True, torch.float64)[0]
    result = extract_masses(
        PipelineResult(correlators={"scalar": corr}, operators={"scalar": x.float()}, scales=None),
        MassExtractionConfig(
            channel_groups=[
                ChannelGroupConfig(
                    name="scalar", correlator_keys=["scalar"], fit=ChannelFitConfig(tmin=1)
                )
            ]
        ),
    )
    ground = result.channels["scalar"].ground_state_mass
    assert abs(gvar.mean(ground) - m) < 4 * gvar.sdev(ground) + 0.03
    assert isinstance(result.fit, dict)


def test_zero_amplitude_level_is_not_reported_as_ground_state():
    torch.manual_seed(2)
    m, T = 0.3, 3000
    rho = math.exp(-m)
    noise = torch.randn(T, dtype=torch.float64)
    x = torch.zeros(T, dtype=torch.float64)
    for i in range(1, T):
        x[i] = rho * x[i - 1] + noise[i]
    corr = correlators_with_statistics(x.unsqueeze(0), 25, True, torch.float64)[0]
    result = extract_masses(
        PipelineResult(correlators={"scalar": corr}, operators={"scalar": x.float()}, scales=None),
        MassExtractionConfig(
            channel_groups=[
                ChannelGroupConfig(
                    name="scalar", correlator_keys=["scalar"], fit=ChannelFitConfig(tmin=1, nexp=2)
                )
            ]
        ),
    )
    channel = result.channels["scalar"]
    ground = channel.energy_levels[channel.ground_state_index]
    assert abs(gvar.mean(ground) - m) < 4 * gvar.sdev(ground) + 0.05


def test_too_few_lags_raise_a_named_error():
    t = np.arange(4)
    corr = torch.tensor(np.exp(-0.3 * t))
    series = torch.randn(5, dtype=torch.float64)
    corr = correlators_with_statistics(series.unsqueeze(0), 3, True, torch.float64)[0]
    with pytest.raises(ValueError, match="scalar"):
        extract_masses(
            PipelineResult(
                correlators={"scalar": corr}, operators={"scalar": series}, scales=None
            ),
            MassExtractionConfig(
                channel_groups=[
                    ChannelGroupConfig(
                        name="scalar", correlator_keys=["scalar"], fit=ChannelFitConfig(tmin=2)
                    )
                ]
            ),
        )


# --------------------------------------------------------------------------
# Electroweak time axis and graph distances
# --------------------------------------------------------------------------


def test_electroweak_lag_duration_uses_frame_spacing():
    from fragile.physics.electroweak.electroweak_channels import electroweak_lag_duration
    from fragile.physics.qft_utils.helpers import recorded_time_step

    history = _dashboard_run(steps=3)
    base = recorded_time_step(history)
    assert electroweak_lag_duration(history, [20, 40, 60]) == pytest.approx(20 * base)
    assert electroweak_lag_duration(history, [5, 6, 7]) == pytest.approx(base)
    assert electroweak_lag_duration(history, [5]) == pytest.approx(base)


def test_graph_distances_are_lengths_not_weights():
    from fragile.physics.app.smeared_operators import (
        compute_pairwise_distance_matrices_from_history,
    )

    history = _dashboard_run(steps=45, seed=11)
    frame = history.n_recorded - 1
    _, distances = compute_pairwise_distance_matrices_from_history(
        history, frame_indices=[frame], edge_weight_mode="riemannian_kernel_volume"
    )
    positions = history.x_after_clone[frame - 1]
    euclid = torch.cdist(positions, positions)
    finite = torch.isfinite(distances[0]) & (euclid > 0)
    # A shortest path is never shorter than the straight line.
    assert bool((distances[0][finite] >= euclid[finite] * 0.999).all())
    off_diagonal = distances[0][~torch.eye(history.N, dtype=torch.bool)]
    assert float(off_diagonal[torch.isfinite(off_diagonal)].median()) > 1e-3
