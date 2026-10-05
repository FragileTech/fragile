"""Verify phase observations against direct Gaussian integration and source laws."""

from itertools import product
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import torch

from fragile.fractalai.theory.keystone_uniform import (
    clone_token_law,
    complete_fitness,
    FitnessParameters,
    rastrigin_uniform_kinetic_budget,
)
from fragile.fractalai.theory.landscape_phase import (
    alive_phase_variance,
    baoab_landing_phase_moments,
    cloning_phase_flux,
    conditional_phase_count_covariance,
    count_viscous_ou_cross,
    gaussian_phase_moments,
    radial_cap_second_moment_bound,
    rastrigin_gaussian_energy,
    rastrigin_haar_potential_mean,
    rastrigin_jitter_profile,
    rastrigin_reference_central_velocity_bound,
    rastrigin_regional_bound,
    rastrigin_regional_profile,
    surviving_phase_count_moments,
)


def test_original_radial_cap_gaussian_moment_below_jensen_bound():
    nodes, weights = np.polynomial.hermite.hermgauss(70)
    nodes, weights = math.sqrt(2) * nodes, weights / math.sqrt(math.pi)
    x, y = np.meshgrid(0.6 + 0.8 * nodes, -0.3 + 0.2 * nodes, indexing="ij")
    radius = np.sqrt(x**2 + y**2)
    capped_square = (2 * radius / (2 + radius)) ** 2
    actual = np.einsum("i,j,ij->", weights, weights, capped_square)
    raw_mean_square = 0.6**2 + 0.3**2 + 0.8**2 + 0.2**2
    bound = radial_cap_second_moment_bound(
        torch.tensor(raw_mean_square, dtype=torch.float64), velocity_cap=2
    )
    assert actual < float(bound) < min(4, raw_mean_square)


def test_component_haar_native_potential_against_sphere_marginal_quadrature():
    means = torch.tensor([[0.04, 0.2, -0.41], [1.0, -0.97, 0.52]], dtype=torch.float64)
    radii = torch.tensor([[1.8, 0.3], [0.9, 2.0]], dtype=means.dtype)
    b, variance = 0.4, 0.003
    result = rastrigin_haar_potential_mean(
        means, radii, orbital_scale=b, gaussian_variance=variance
    )
    # In dimension three a Haar-rotated fixed vector has a uniform
    # projection on [-radius,radius]. Components are independent,
    # while the rotation within each component is shared by all rows.
    nodes, weights = np.polynomial.legendre.leggauss(30)
    noise, normal_weights = np.polynomial.hermite.hermgauss(30)
    noise *= math.sqrt(2 * variance)
    normal_weights /= math.sqrt(math.pi)
    expected = []
    for mean, radius in zip(means.numpy(), radii.numpy(), strict=True):
        shift = (
            b * (radius[0] * nodes[:, None, None] + radius[1] * nodes[None, :, None])
            + noise[None, None, :]
        )
        energy = sum(
            (center + shift) ** 2 + 10 * (1 - np.cos(2 * math.pi * (center + shift)))
            for center in mean
        )
        expected.append(np.einsum("i,j,k,ijk->", weights / 2, weights / 2, normal_weights, energy))
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-14, atol=2e-14)


def test_native_gaussian_energy_against_joint_ou_quadrature():
    x = torch.tensor([[0.03, 0.24], [0.87, -1.31]], dtype=torch.float64)
    u = torch.tensor([[1.8, -0.7], [-1.2, 0.4]], dtype=torch.float64)
    t, c, q, s = 0.04, 0.91, 0.45, 0.08
    result = rastrigin_gaussian_energy(
        x,
        u,
        half_step=t,
        ou_retention=c,
        ou_amplitude=q,
        position_amplitude=s,
    )
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    nodes, weights = math.sqrt(2) * nodes, weights / math.sqrt(math.pi)

    def force(y):
        return -2 * y - 20 * math.pi * np.sin(2 * math.pi * y)

    def potential(y):
        return y**2 + 10 - 10 * np.cos(2 * math.pi * y)

    fx = force(x.numpy())
    means = x.numpy() + t * (1 + c) * (u.numpy() + t * fx)
    vmeans = c * (u.numpy() + t * fx)
    expected_u, variance_u, expected_k = [], [], []
    for mean_row, vmean_row in zip(means, vmeans, strict=True):
        row_u, row_var, row_k = 0.0, 0.0, 0.0
        for mean, vmean in zip(mean_row, vmean_row, strict=True):
            final_positions = mean + math.sqrt(t**2 * q**2 + s**2) * nodes
            potentials = potential(final_positions)
            eu = np.dot(weights, potentials)
            row_u += eu
            row_var += np.dot(weights, (potentials - eu) ** 2)
            # The same OU draw moves position and velocity. Replacing
            # these by independent normals would miss the force work.
            y2 = mean + t * q * nodes
            raw_velocity = vmean + q * nodes + t * force(y2)
            row_k += np.dot(weights, raw_velocity**2) / 2
        expected_u.append(row_u)
        variance_u.append(row_var)
        expected_k.append(row_k)
    np.testing.assert_allclose(result.potential_mean.numpy(), expected_u, rtol=1e-12)
    np.testing.assert_allclose(result.potential_variance.numpy(), variance_u, rtol=1e-12)
    np.testing.assert_allclose(result.pre_graph_kinetic_mean.numpy(), expected_k, rtol=1e-12)


def test_central_native_velocity_secant_against_unbounded_gaussian_integral():
    result = rastrigin_reference_central_velocity_bound()
    t, c = 0.02, math.exp(-0.04)
    b = t * (1 + c)
    eta = t * b
    q2 = (1 - c**2) / 2
    linear, periodic = c - 2 * eta, 40 * math.pi**2 * eta
    shift = 4 * b
    sinc_min = math.sin(2 * math.pi * shift) / (2 * math.pi * shift)

    def envelope(x):
        r = abs(x)
        g = r + eta * (-2 * r - 20 * math.pi * math.sin(2 * math.pi * r))
        phase = min(g + shift, 0.5)
        # Independently integrate the actual OU sine secant rather than
        # assuming that its square equals the square of its mean.
        return max(
            quad(
                lambda z: (
                    (
                        linear
                        - periodic * sinc * math.cos(2 * math.pi * (phase + t * math.sqrt(q2) * z))
                    )
                    ** 2
                    * norm.pdf(z)
                ),
                -math.inf,
                math.inf,
                epsabs=1e-10,
            )[0]
            for sinc in (sinc_min, 1)
        )

    # Gauss-Hermite integrates the unbounded own-jitter distribution.
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    integral = sum(
        weight * envelope(0.01 + 0.1 * math.sqrt(2) * node)
        for node, weight in zip(nodes, weights, strict=True)
    ) / math.sqrt(math.pi)
    assert integral < result.jitter_secant_second_moment
    assert result.persisting_secant_second_moment == pytest.approx(envelope(0.01))
    assert result.velocity_coefficient == pytest.approx(0.9320202359871004)
    assert result.velocity_coefficient < 0.934


def test_surviving_phase_counts_against_complete_categorical_enumeration():
    p = torch.tensor([[0.2, 0.6, 0.2], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]], dtype=torch.float64)
    mean, second, survival = torch.zeros(3), torch.zeros(3, 3), 0.0
    mean, second = mean.double(), second.double()
    for labels in product(range(3), repeat=3):
        weight = math.prod(float(p[i, label]) for i, label in enumerate(labels))
        if labels == (2, 2, 2):
            continue
        count = torch.bincount(torch.tensor(labels), minlength=3).double() / 3
        mean += weight * count
        second += weight * torch.outer(count, count)
        survival += weight
    result = surviving_phase_count_moments(p)
    expected_mean = mean / survival
    torch.testing.assert_close(result.mean, expected_mean)
    torch.testing.assert_close(
        result.covariance, second / survival - torch.outer(expected_mean, expected_mean)
    )
    assert float(result.survival_probability) == pytest.approx(survival)


def test_evaluated_original_regional_bound_has_no_population_parameter():
    result = rastrigin_regional_bound(
        dimension=3,
        timestep=0.04,
        friction=1,
        clone_jitter=0.1,
        ou_amplitude=math.sqrt((1 - math.exp(-0.08)) / 2),
        position_amplitude=0.02,
        velocity_cap=2,
        restitution=0.5,
        viscosity=0.3,
    )
    assert result.mean_map_squared == pytest.approx(0.5978840815787646)
    assert result.accepted_coordinate_variance == pytest.approx(0.006371638355086486)
    assert result.positional_coefficient == pytest.approx(0.7989420407893824)
    assert result.conservative_floor == pytest.approx(0.20613357968212598)


@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.7])
def test_unrestricted_periodic_jitter_moments_against_gaussian_quadrature(sigma):
    nodes, weights = np.polynomial.hermite.hermgauss(100)
    weights = weights / math.sqrt(math.pi)
    eta = 0.02**2 * (1 + math.exp(-0.04))
    sources = torch.tensor([[-1.03, 0.04, 0.99]], dtype=torch.float64)
    result = rastrigin_jitter_profile(sources, drift_coefficient=eta, clone_jitter=sigma)
    samples = sources.numpy()[:, :, None] + sigma * math.sqrt(2) * nodes
    values = samples - eta * (2 * samples + 20 * math.pi * np.sin(2 * math.pi * samples))
    mean = np.sum(values * weights, axis=2)
    variance = np.sum((values - mean[:, :, None]) ** 2 * weights, axis=2)
    np.testing.assert_allclose(result.mean.numpy(), mean, atol=2e-14)
    np.testing.assert_allclose(result.variance.numpy(), variance, atol=2e-14)


def test_second_count_kick_retains_exact_ou_force_correlation():
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    # Only the Gaussian difference xi_1-xi_2 enters the symmetrized pair.
    difference = 2 * nodes
    weights = weights / math.sqrt(math.pi)
    x = torch.tensor([[-0.6], [0.8]], dtype=torch.float64)
    u = torch.tensor([[1.2], [-0.4]], dtype=torch.float64)
    t, c, q, nu, rho = 0.02, math.exp(-0.04), 0.25, 0.3, 1.0
    b = t * (1 + c)
    result = count_viscous_ou_cross(
        x,
        u,
        half_step=t,
        drift_duration=b,
        ou_coefficient=c,
        ou_amplitude=q,
        viscosity=nu,
        bandwidth=rho,
    )
    delta_m = float((x + b * u)[0, 0] - (x + b * u)[1, 0])
    delta_u = float(u[0, 0] - u[1, 0])
    kernel = np.exp(-((delta_m + t * q * difference) ** 2) / (2 * rho**2))
    expected = -nu / 4 * np.sum(weights * difference * kernel * (c * delta_u + q * difference))
    assert float(result) == pytest.approx(expected, abs=2e-14)


def test_native_rastrigin_wells_and_barriers_have_opposite_curvature():
    profile = rastrigin_regional_profile(-5, 5)
    assert profile.core_curvature_lower == pytest.approx(281.1545679855552)
    assert profile.global_curvature_upper == pytest.approx(396.78417604357435)
    for k, root in zip(profile.integer_centers, profile.stable_roots, strict=True):
        assert k - 0.125 < root < k + 0.125
        assert 2 * root + 20 * math.pi * math.sin(2 * math.pi * root) == pytest.approx(0, abs=1e-9)
        assert 2 + 40 * math.pi**2 * math.cos(2 * math.pi * root) > 0
    for barrier in profile.barriers:
        assert 2 + 40 * math.pi**2 * math.cos(2 * math.pi * barrier) < 0


def test_gaussian_phase_moments_against_direct_density_integrals():
    means = torch.tensor([[-0.7], [1.2]], dtype=torch.float64)
    lower = torch.tensor([[-2.0], [0.0]], dtype=torch.float64)
    upper = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    centers = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    result = gaussian_phase_moments(means, lower, upper, centers, standard_deviation=0.6)
    for i, mean in enumerate(means[:, 0].tolist()):
        for j in range(2):
            lo, hi, center = float(lower[j, 0]), float(upper[j, 0]), float(centers[j, 0])

            def density(x):
                return norm.pdf(x, loc=mean, scale=0.6)

            assert float(result.probability[i, j]) == pytest.approx(quad(density, lo, hi)[0])
            assert float(result.first_moment[i, j, 0]) == pytest.approx(
                quad(lambda x: x * density(x), lo, hi)[0]
            )
            assert float(result.centered_second_moment[i, j]) == pytest.approx(
                quad(lambda x: (x - center) ** 2 * density(x), lo, hi)[0]
            )
    torch.testing.assert_close(result.probability.sum(1), torch.ones(2, dtype=means.dtype))
    torch.testing.assert_close(result.first_moment.sum(1), means)
    raw_second = (
        result.centered_second_moment[:, :2]
        + 2 * (centers * result.first_moment[:, :2]).sum(2)
        - centers.square().sum(1) * result.probability[:, :2]
    )
    torch.testing.assert_close(
        raw_second.sum(1) + result.centered_second_moment[:, -1], means.square().sum(1) + 0.6**2
    )


def test_phase_counts_retain_exterior_and_own_partition_endpoints():
    x = torch.tensor([[-2.0], [0.0], [2.0], [4.0]], dtype=torch.float64)
    lower = torch.tensor([[-2.0], [0.0]], dtype=x.dtype)
    upper = torch.tensor([[0.0], [2.0]], dtype=x.dtype)
    result = gaussian_phase_moments(x, lower, upper, (lower + upper) / 2, standard_deviation=0)
    torch.testing.assert_close(
        result.probability,
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=x.dtype),
    )


def test_cloning_flux_uses_actual_fitness_and_weighted_revival():
    x = torch.tensor([[-0.8], [0.7], [4.0]], dtype=torch.float64)
    alive = torch.tensor([True, True, False])
    parameters = FitnessParameters(0.1, 0.1, 2, 2, 0.1, 0.1, 1, 1, 1, 1e-6)
    fitness = complete_fitness(
        -x[:, 0].square() / 2,
        torch.tensor([1.5, 1.5, 0.0], dtype=x.dtype),
        alive,
        parameters=parameters,
    )
    companions = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0.25, 0.75, 0]], dtype=x.dtype)
    tokens = clone_token_law(fitness, companions, alive, parameters=parameters)
    lower, upper = (
        torch.tensor([[-2.0], [0.0]], dtype=x.dtype),
        torch.tensor([[0.0], [2.0]], dtype=x.dtype),
    )
    centers = (lower + upper) / 2
    flux = cloning_phase_flux(x, tokens, lower, upper, centers, clone_jitter=0.3)
    expected = np.zeros(3)
    for i in range(3):
        for flag in range(2):
            for donor in range(3):
                weight = float(tokens[i, flag, donor]) / 3
                if flag:
                    probs = np.array([
                        norm.cdf(0, loc=float(x[donor, 0]), scale=0.3)
                        - norm.cdf(-2, loc=float(x[donor, 0]), scale=0.3),
                        norm.cdf(2, loc=float(x[donor, 0]), scale=0.3)
                        - norm.cdf(0, loc=float(x[donor, 0]), scale=0.3),
                        0.0,
                    ])
                    probs[2] = 1 - probs[:2].sum()
                else:
                    probs = np.array([float(x[donor, 0] < 0), float(x[donor, 0] >= 0), 0.0])
                expected += weight * probs
    np.testing.assert_allclose(flux.output_fraction.numpy(), expected, atol=1e-14)
    assert float(flux.recipient_to_source[-1].sum()) == pytest.approx(1 / 3)
    assert float(flux.source_to_landing[-1].sum()) == pytest.approx(0)
    assert float(flux.output_fraction[-1]) > 0
    assert float(flux.source_to_landing.sum()) == pytest.approx(1)
    torch.testing.assert_close(flux.row_probability.mean(0), flux.output_fraction)
    covariance = conditional_phase_count_covariance(flux.row_probability)
    assert float(covariance.diagonal().max()) <= 1 / 12
    torch.testing.assert_close(
        covariance.sum(1), torch.zeros(3, dtype=x.dtype), atol=1e-15, rtol=0
    )


def test_different_wells_retain_between_phase_variance_and_alive_centering():
    profile = rastrigin_regional_profile(-1, 1)
    root = profile.stable_roots[-1]
    positions = torch.tensor([[-root], [-root], [root], [root], [1e6]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 2])
    alive = torch.tensor([True, True, True, True, False])
    result = alive_phase_variance(positions, labels, alive)
    assert float(result.within) == pytest.approx(0)
    assert float(result.between) == pytest.approx(root**2)
    torch.testing.assert_close(result.total, result.within + result.between)
    translated = positions + 0.17
    shifted = alive_phase_variance(translated, labels, alive)
    torch.testing.assert_close(result.total, shifted.total)
    torch.testing.assert_close(result.weights, shifted.weights)


@pytest.mark.parametrize("row_normalized", [False, True])
def test_complete_rastrigin_landing_uses_actual_first_viscosity(row_normalized):
    x = torch.tensor([[-0.95], [0.0], [1.02]], dtype=torch.float64)
    v = torch.tensor([[0.3], [-0.2], [0.1]], dtype=x.dtype)
    force = -2 * x - 20 * math.pi * torch.sin(2 * math.pi * x)
    lower, upper = (
        torch.tensor([[-2.0], [0.0]], dtype=x.dtype),
        torch.tensor([[0.0], [2.0]], dtype=x.dtype),
    )
    result = baoab_landing_phase_moments(
        x,
        v,
        force,
        lower,
        upper,
        (lower + upper) / 2,
        timestep=0.04,
        viscosity=0.3,
        bandwidth=1,
        row_normalized=row_normalized,
        ou_coefficient=math.exp(-0.04),
        ou_amplitude=0.2,
        position_amplitude=0.02,
    )
    weights = torch.exp(-torch.cdist(x, x).square() / 2)
    weights.fill_diagonal_(0)
    denom = weights.sum(1, keepdim=True) if row_normalized else 3
    viscosity = 0.3 * (weights @ v - weights.sum(1, keepdim=True) * v) / denom
    expected_mean = x + 0.02 * (1 + math.exp(-0.04)) * (v + 0.02 * (force + viscosity))
    torch.testing.assert_close(result.first_moment.sum(1), expected_mean)


@pytest.mark.parametrize("row_normalized", [False, True])
def test_native_rastrigin_complete_stage_budgets_keep_periodic_force(row_normalized):
    budget = rastrigin_uniform_kinetic_budget(
        dimension=3,
        moment_order=2,
        timestep=0.04,
        viscosity=0.3,
        terminal_half_width=2,
        clone_jitter=0.1,
        restitution=0.5,
        velocity_cap=2,
        ou_coefficient=math.exp(-0.04),
        ou_amplitude=math.sqrt((1 - math.exp(-0.08)) / 2),
        position_amplitude=0.02,
        row_normalized=row_normalized,
    )
    assert budget.prepared_position > 2 * math.sqrt(3)
    assert budget.first_kick_velocity > 4 + 0.02 * 20 * math.pi * math.sqrt(3)
    assert budget.second_kick_velocity > budget.ou_velocity
    assert budget.output_velocity == 2
