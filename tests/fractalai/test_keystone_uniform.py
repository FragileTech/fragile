"""Check signed identities by independent enumeration and actual stage maps."""

from itertools import product
import math
import operator

import numpy as np
import pytest
from scipy.stats import binom
import torch

from fragile.fractalai.theory.keystone_uniform import (
    clone_token_law,
    cloning_variance_balance,
    complete_fitness,
    coupled_baoab_balance,
    FitnessParameters,
    gaussian_row_column_bound,
    global_standardize,
    maximal_token_coupling,
    native_source_pressure_lower_bound,
    quadratic_energy_alive_envelope,
    quadratic_uniform_envelope,
    quadratic_uniform_kinetic_budget,
    rastrigin_box_survival_budget,
    styblinski_tang_uniform_kinetic_budget,
    UniformQuadraticEnvelope,
)


@pytest.mark.parametrize("row_normalized", [False, True])
def test_rastrigin_survival_integral_is_bounded_without_clipping(row_normalized):
    from scipy.integrate import quad
    from scipy.special import ndtr

    result = rastrigin_box_survival_budget(
        dimension=3,
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
    t, c = 0.02, math.exp(-0.04)
    b = t * (1 + c)
    tau = math.hypot(t * math.sqrt((1 - c**2) / 2), 0.02)

    def integrand(z):
        x = 2 + 0.1 * z
        mean = x + b * t * (-2 * x - 20 * math.pi * math.sin(2 * math.pi * x))
        inner = 2 - b * 4
        probability = ndtr((inner - mean) / tau) - ndtr((-inner - mean) / tau)
        return probability * math.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)

    integral = quad(integrand, -math.inf, math.inf, epsabs=1e-12)[0] ** 3
    assert math.exp(result.log_accepted_floor) <= integral
    assert integral == pytest.approx(1.61911926736e-5)
    assert result.landing_slope_lower > 0.68
    assert math.exp(result.log_alive_floor) > (7e-11 if row_normalized else 2e-6)
    assert result.row_column_bound == (7188 if row_normalized else 1)
    assert result.log_count_transform_bound(31, 0) == 0
    # Compare the optimized envelope to every gate fraction, using an
    # independent normal-tail evaluation and the original energy budget.
    p1 = math.exp(result.log_accepted_floor)
    for u in np.linspace(0.001, 1, 1000):
        profile = ndtr(-b * result.velocity_energy_bound / (tau * math.sqrt(3 * u))) ** 3
        assert math.exp(result.log_alive_floor) <= u * profile + (1 - u) * p1 + 1e-18


def test_column_shell_tail_sharpens_the_existing_proof():
    bounds = [gaussian_row_column_bound(3, series_terms=m) for m in (0, 5, 10, 40)]
    assert bounds == sorted(bounds, reverse=True)
    assert 7187 < bounds[-1] < 7188


@pytest.fixture
def parameters():
    return FitnessParameters(0.1, 0.1, 2, 2, 0.1, 0.1, 1, 1, 1, 1e-6)


@pytest.mark.parametrize("values", [[2.0], [2.0, 2.0], [-3.0, 0.0, 2.0, 10.0]])
def test_global_standardizer_operator_norm(values):
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    alive = torch.ones_like(x, dtype=torch.bool)
    jacobian = torch.autograd.functional.jacobian(
        lambda z: global_standardize(z, alive, floor=0.1), x
    )
    assert torch.linalg.matrix_norm(jacobian, ord=2) <= 10 + 1e-10


def test_complete_fitness_retains_measured_normalization_and_ties(parameters):
    alive = torch.tensor([True, True, False])
    reward = torch.tensor([1.0, 1.0, 1e10], dtype=torch.float64)
    diversity = torch.tensor([3.0, 3.0, -1e10], dtype=torch.float64)
    fitness = complete_fitness(reward, diversity, alive, parameters=parameters)
    torch.testing.assert_close(fitness, torch.tensor([1.21, 1.21, 0.0], dtype=torch.float64))
    companion = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0.3, 0.7, 0]], dtype=torch.float64)
    tokens = clone_token_law(fitness, companion, alive, parameters=parameters)
    assert tokens[0, 0, 0] == tokens[1, 0, 1] == 1
    torch.testing.assert_close(tokens[2, 1], companion[2])
    assert parameters.fitness_slopes == pytest.approx((10.5, 10.5))
    assert parameters.gate_slopes == pytest.approx((199.9800019998, 99.9900009999))


def test_singleton_and_revival_use_different_tokens(parameters):
    alive = torch.tensor([False, True, False])
    fitness = torch.tensor([0.0, 1.21, 0.0], dtype=torch.float64)
    companion = torch.tensor([[0.0, 1, 0]] * 3, dtype=torch.float64)
    tokens = clone_token_law(fitness, companion, alive, parameters=parameters)
    assert tokens[1, 0, 1] == 1
    assert tokens[0, 1, 1] == tokens[2, 1, 1] == 1


@pytest.mark.parametrize("n", [1, 2, 7, 128])
def test_mean_fitness_bound_retains_sampled_normalizers_and_dead_extremes(parameters, n):
    generator = torch.Generator().manual_seed(781 + n)
    alive = torch.arange(n + 2) < n
    for skew in (0, 1e-14, 2, 1e8):
        reward = torch.randn(n + 2, dtype=torch.float64, generator=generator) ** 3 * skew
        diversity = torch.randn(n + 2, dtype=torch.float64, generator=generator).exp() * skew
        reward[~alive], diversity[~alive] = 1e100, -1e100
        fitness = complete_fitness(reward, diversity, alive, parameters=parameters)
        assert fitness[alive].mean() <= parameters.unit_power_mean_bound
    assert parameters.unit_power_mean_bound == pytest.approx(1.614)


def test_signed_source_pressure_is_below_exact_complete_token_expectation(parameters):
    generator = torch.Generator().manual_seed(1731)
    points = torch.randn(19, 4, dtype=torch.float64, generator=generator)
    raw = torch.exp(-torch.cdist(points, points).square() / 8)
    alive = torch.arange(19) % 4 != 0
    reward = torch.randn(19, dtype=torch.float64, generator=generator)
    diversity = torch.randn(19, dtype=torch.float64, generator=generator).exp()
    fitness = complete_fitness(reward, diversity, alive, parameters=parameters)
    weights = raw * alive[None, :]
    weights.fill_diagonal_(0)
    companion = weights / weights.sum(dim=1, keepdim=True)
    tokens = clone_token_law(fitness, companion, alive, parameters=parameters)
    exact_excess = tokens.sum(dim=(0, 1)) - alive.to(torch.float64)
    lower = native_source_pressure_lower_bound(
        fitness, raw, alive, clone_regularizer=parameters.clone_regularizer
    )
    assert (lower[alive] <= exact_excess[alive] + 1e-14).all()
    # Revived labels contribute positive source mass even with equal fitness.
    tied = torch.where(alive, torch.full_like(fitness, 1.21), 0)
    tied_tokens = clone_token_law(tied, companion, alive, parameters=parameters)
    assert tied_tokens.sum(dim=(0, 1))[alive].sum() == pytest.approx(19)


def test_source_pressure_singleton_preserves_original_dead_queries():
    alive = torch.tensor([False, True, False])
    fitness = torch.tensor([0.0, 1.21, 0.0], dtype=torch.float64)
    raw = torch.tensor([[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]], dtype=torch.float64)
    bound = native_source_pressure_lower_bound(fitness, raw, alive, clone_regularizer=1e-6)
    torch.testing.assert_close(bound, torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64))


def test_revived_variance_survives_unbounded_dead_coordinate_cancellation(parameters):
    alive = torch.tensor([True, False])
    fitness = torch.tensor([1.21, 0.0], dtype=torch.float64)
    companion = torch.tensor([[1.0, 0], [1, 0]], dtype=torch.float64)
    tokens = clone_token_law(fitness, companion, alive, parameters=parameters)
    positions = torch.tensor([[0.0], [1e100]], dtype=torch.float64)
    balance = cloning_variance_balance(positions, tokens, clone_jitter=0.1)
    assert float(balance.output_variance) == pytest.approx(0.0025)
    assert float(balance.barycenter_variance_loss) == pytest.approx(0.0025)


def test_signed_clone_variance_matches_full_source_enumeration(parameters):
    positions = torch.tensor([[-1.0, 2.0], [3.0, 0.0], [20.0, -40.0]], dtype=torch.float64)
    alive = torch.tensor([True, True, False])
    fitness = torch.tensor([0.2, 2.0, 0.0], dtype=torch.float64)
    companion = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0.3, 0.7, 0]], dtype=torch.float64)
    tokens = clone_token_law(fitness, companion, alive, parameters=parameters)
    ledger = cloning_variance_balance(positions, tokens, clone_jitter=0.4)
    expected = 0.0
    for choices in product(range(6), repeat=3):
        probability = math.prod(
            float(tokens[i].reshape(-1)[choice]) for i, choice in enumerate(choices)
        )
        if probability == 0:
            continue
        donors = [choice % 3 for choice in choices]
        output = positions[donors]
        variance = (output - output.mean(0)).square().sum(1).mean()
        jitter_variance = sum(choice // 3 for choice in choices) * 2 * 0.4**2 * (1 / 3 - 1 / 9)
        expected += probability * (float(variance) + jitter_variance)
    assert float(ledger.output_variance) == pytest.approx(expected)
    assert float(ledger.barycenter_variance_loss) > 0
    assert float(ledger.output_variance) < float(ledger.input_variance)


def test_maximal_tokens_preserve_acceptance_and_both_marginals(parameters):
    companion = torch.tensor([[0.0, 1], [1, 0]], dtype=torch.float64)
    alive = torch.tensor([True, True])
    first = clone_token_law(
        torch.tensor([0.4, 0.7], dtype=torch.float64), companion, alive, parameters=parameters
    )
    second = clone_token_law(
        torch.tensor([0.7, 0.4], dtype=torch.float64), companion, alive, parameters=parameters
    )
    coupling = maximal_token_coupling(first, second)
    torch.testing.assert_close(coupling.sum((3, 4)), first)
    torch.testing.assert_close(coupling.sum((1, 2)), second)
    common = coupling.reshape(2, 4, 4).diagonal(dim1=1, dim2=2).sum(1)
    torch.testing.assert_close(common, torch.minimum(first, second).sum((1, 2)))
    # Different accepted/persisting sources retain their actual jitter flags.
    assert coupling[0, 1, 1, 0, 0] > 0


def test_tiny_residual_coupling_does_not_square_before_dividing():
    first = torch.zeros(2, 2, 2, dtype=torch.float64)
    second = torch.zeros_like(first)
    first[:, 0, 0] = second[:, 0, 0] = 1
    first[:, 1, 0] = 1e-200
    second[:, 1, 1] = 1e-200
    coupling = maximal_token_coupling(first, second)
    assert float(coupling[0, 1, 0, 1, 1]) == pytest.approx(1e-200, abs=0)


@pytest.mark.parametrize("row_normalized", [False, True])
def test_complete_signed_baoab_includes_both_viscous_kicks_and_cap(row_normalized):
    generator = torch.Generator().manual_seed(47)
    x, v, y, w = (torch.randn(7, 3, dtype=torch.float64, generator=generator) for _ in range(4))
    ou = 50 * torch.randn(7, 3, dtype=torch.float64, generator=generator)
    position = 30 * torch.randn(7, 3, dtype=torch.float64, generator=generator)
    balance = coupled_baoab_balance(
        x,
        v,
        y,
        w,
        ou,
        position,
        force=lambda z: -z + 0.2 * torch.sin(z),
        timestep=0.04,
        viscosity=0.3,
        bandwidth=1,
        row_normalized=row_normalized,
        ou_coefficient=math.exp(-0.04),
        ou_amplitude=0.2,
        position_amplitude=0.02,
        velocity_cap=2,
        position_weight=2,
        cross_weight=0.3,
        velocity_weight=1,
    )
    torch.testing.assert_close(
        balance.output_form - balance.input_form,
        balance.signed_kinetic_increment + balance.signed_cap_increment,
    )
    torch.testing.assert_close(
        balance.first_output_positions - balance.second_output_positions,
        balance.raw_position_difference,
    )
    assert torch.linalg.vector_norm(balance.first_output_velocities, dim=1).max() < 2
    assert balance.signed_cap_increment != 0
    assert not torch.allclose(balance.first_force_difference, balance.second_force_difference)


@pytest.mark.parametrize("n_walkers", [2, 7, 31])
def test_count_background_with_two_noncommuting_graphs_and_actual_cap(n_walkers):
    generator = torch.Generator().manual_seed(97)
    t, c, kick = 0.02, math.exp(-0.04), 0.006
    b, a = t * (1 + c), 1 - t**2 * (1 + c)
    beta = b / a

    def matrix(x):
        weights = torch.exp(-torch.cdist(x, x).square() / 2) / n_walkers
        return torch.eye(n_walkers, dtype=x.dtype) - kick * (torch.diag(weights.sum(1)) - weights)

    x, y = (torch.randn(n_walkers, 3, dtype=torch.float64, generator=generator) for _ in range(2))
    first, second = matrix(x), matrix(y)
    r, z = (torch.randn(n_walkers, 3, dtype=x.dtype, generator=generator) for _ in range(2))
    position = a * r + b * (first @ z)
    velocity = -t * (c * (second @ r) + a * r) + (c * second - t * b * torch.eye(n_walkers)) @ (
        first @ z
    )

    def cap(v):
        return 2 * v / (2 + torch.linalg.vector_norm(v, dim=1, keepdim=True))

    def form(r, z):
        return (r.square() + 2 * beta * r * z + z.square()).sum(1).mean()

    for amplitude in (0, 0.1, 10, 1e3):
        noise_center = amplitude * torch.randn(n_walkers, 3, dtype=x.dtype, generator=generator)
        capped = cap(noise_center + velocity) - cap(noise_center)
        assert float(form(position, capped)) <= 0.998492 * float(form(r, z))


def test_reference_envelope_is_population_independent():
    envelope = quadratic_uniform_envelope(
        dimension=3,
        timestep=0.04,
        viscosity=0.3,
        quadratic_coefficient=1,
        terminal_half_width=2,
        clone_jitter=0.1,
        restitution=0.5,
        velocity_cap=2,
        ou_coefficient=math.exp(-0.04),
        ou_amplitude=math.sqrt((1 - math.exp(-0.08)) / 2),
        position_amplitude=0.02,
    )
    assert envelope.log_survival_parameter == pytest.approx(-41.2528745903)
    assert math.exp(envelope.log_clone_survival) == pytest.approx(2.6095491936e-4)
    assert math.exp(envelope.log_inverse_alive_bound()) == pytest.approx(3.623490693e18)
    assert envelope.log_qsd_eigenvalue_lower_bound(200) == pytest.approx(
        math.log(200) - 41.2528745903
    )
    assert math.isfinite(
        envelope.log_exponential_moment(1 / (8 * envelope.maximum_gaussian_variance))
    )
    with pytest.raises(ValueError):
        envelope.log_exponential_moment(1 / (4 * envelope.maximum_gaussian_variance))


@pytest.mark.parametrize("row_normalized", [False, True])
def test_averaged_energy_optimizes_every_actual_clone_fraction(row_normalized):
    from scipy.special import ndtr, ndtri

    coefficients = {
        "dimension": 3,
        "timestep": 0.04,
        "viscosity": 0.3,
        "quadratic_coefficient": 1,
        "terminal_half_width": 2,
        "clone_jitter": 0.1,
        "restitution": 0.5,
        "velocity_cap": 2,
        "ou_coefficient": math.exp(-0.04),
        "ou_amplitude": math.sqrt((1 - math.exp(-0.08)) / 2),
        "position_amplitude": 0.02,
    }
    result = quadratic_energy_alive_envelope(**coefficients, row_normalized=row_normalized)
    row_bound = quadratic_uniform_envelope(**coefficients)
    assert result.log_fraction_test < result.log_accepted_floor
    expected = 1.2613357527748248e-15 if row_normalized else 5.116105717745763e-7
    assert math.exp(result.log_alive_exponent) == pytest.approx(expected, rel=1e-10, abs=0)
    assert result.velocity_energy_bound < row_bound.collision_velocity_bound
    t, c = 0.02, coefficients["ou_coefficient"]
    noise = math.sqrt(row_bound.kinetic_position_variance)
    mean = abs(row_bound.position_coefficient) * 2
    base = (ndtr((2 - mean) / noise) - ndtr((-2 - mean) / noise)) ** 3
    z, k = ndtri(base), t * (1 + c) / noise
    p1 = math.exp(result.log_accepted_floor)
    for theta in (0.2, math.log(2), 4, math.inf):
        transform = -math.expm1(-theta)
        target = transform * expected
        for fraction in np.linspace(0, 1, 101):
            no_copy = (
                fraction * ndtr(z - k * result.velocity_energy_bound / math.sqrt(fraction))
                if fraction
                else 0
            )
            rate = transform * no_copy - (1 - fraction) * math.log1p(-transform * p1)
            assert rate >= target * (1 - 1e-10)
    assert result.log_count_transform_bound(200, math.inf) == pytest.approx(-200 * expected)
    assert result.log_inverse_alive_bound() < row_bound.log_inverse_alive_bound()


@pytest.mark.parametrize("n_walkers", [3, 8, 200])
@pytest.mark.parametrize("row_normalized", [False, True])
def test_strict_gap_revival_broadcasts_actual_donor_coordinate(
    parameters, n_walkers, row_normalized
):
    generator = torch.Generator().manual_seed(13)
    positions = torch.zeros(n_walkers, 3, dtype=torch.float64)
    positions[0, 0], positions[1, 0], positions[2:, 2] = -1, 0.9, 3
    shifted = positions.clone()
    shifted[:2, 1] = 0.1
    alive = torch.arange(n_walkers) < 2

    def token_law(x):
        features = 2 * x / (2 + torch.linalg.vector_norm(x, dim=1, keepdim=True))
        diversity = torch.full((n_walkers,), 0.001, dtype=x.dtype)
        diversity[:2] = torch.sqrt((features[0] - features[1]).square().sum() + 1e-6)
        fitness = complete_fitness(-x.square().sum(1) / 2, diversity, alive, parameters=parameters)
        weights = torch.exp(-torch.cdist(features, features).square() / 8)
        weights[:, ~alive] = 0
        weights[torch.arange(2), torch.arange(2)] = 0
        companion = weights / weights.sum(1, keepdim=True)
        return fitness, clone_token_law(fitness, companion, alive, parameters=parameters)

    first_fitness, first_tokens = token_law(positions)
    second_fitness, second_tokens = token_law(shifted)
    torch.testing.assert_close(first_fitness, second_fitness)
    assert float(first_fitness[1] - first_fitness[0]) > 0.4
    assert float(first_tokens[0, 1].sum()) == pytest.approx(0.4755167715766428)
    # Sum the complete actual token law, including weighted mandatory revival.
    source_mean = first_tokens.sum(1) @ positions
    shifted_mean = second_tokens.sum(1) @ shifted
    torch.testing.assert_close(
        shifted_mean[:, 1] - source_mean[:, 1],
        torch.full((n_walkers,), 0.1, dtype=positions.dtype),
    )

    donors = torch.arange(n_walkers) % 2
    other_donors = 1 - donors
    donors[:2] = other_donors[:2] = 1  # Positive-probability optional accepted edge.
    accepted = torch.ones(n_walkers, dtype=torch.bool)
    accepted[1] = False
    jitter = 0.1 * torch.randn(n_walkers, 3, dtype=positions.dtype, generator=generator)
    prepared = positions[donors] + jitter * accepted[:, None]
    other_prepared = shifted[other_donors] + jitter * accepted[:, None]
    zeros = torch.zeros_like(positions)
    balance = coupled_baoab_balance(
        prepared,
        zeros,
        other_prepared,
        zeros,
        40 * torch.randn(n_walkers, 3, dtype=positions.dtype, generator=generator),
        20 * torch.randn(n_walkers, 3, dtype=positions.dtype, generator=generator),
        force=operator.neg,
        timestep=0.04,
        viscosity=0.3,
        bandwidth=1,
        row_normalized=row_normalized,
        ou_coefficient=math.exp(-0.04),
        ou_amplitude=math.sqrt((1 - math.exp(-0.08)) / 2),
        position_amplitude=0.02,
        velocity_cap=2,
        position_weight=1,
        cross_weight=0,
        velocity_weight=1,
    )
    a_h = 1 - 0.02**2 * (1 + math.exp(-0.04))
    torch.testing.assert_close(
        balance.second_output_positions[:, 1] - balance.first_output_positions[:, 1],
        torch.full((n_walkers,), a_h * 0.1, dtype=positions.dtype),
    )
    assert (a_h * 0.1) ** 2 / (2 * 0.1**2 / n_walkers) == pytest.approx(n_walkers * a_h**2 / 2)


@pytest.mark.parametrize("probability", [0.01, 0.2, 0.8])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_conditional_inverse_count_bound_against_exact_binomial(probability, order):
    envelope = UniformQuadraticEnvelope(
        1, 1, 1, 1, 1, 1, math.log(probability), math.log(probability), math.log(probability)
    )
    for n in (1, 2, 10, 100):
        counts = np.arange(1, n + 1)
        expected = np.dot((n / counts) ** order, binom.pmf(counts, n, probability)) / -math.expm1(
            n * math.log1p(-probability)
        )
        assert expected <= math.exp(envelope.log_inverse_alive_bound(order=order))


def test_subnormal_survival_lower_bound_remains_a_lower_bound():
    envelope = UniformQuadraticEnvelope(1, 1, 1, 1, 1, 1, -1000, -1000, -1000)
    assert envelope.log_qsd_eigenvalue_lower_bound(200) == pytest.approx(math.log(200) - 1000)
    with pytest.raises(ValueError):
        envelope.log_qsd_eigenvalue_lower_bound(0)


def test_gaussian_landing_keeps_log_probability_below_exponent_range():
    envelope = quadratic_uniform_envelope(
        dimension=2000,
        timestep=0.04,
        viscosity=0.3,
        quadratic_coefficient=1,
        terminal_half_width=2,
        clone_jitter=0.1,
        restitution=0.5,
        velocity_cap=2,
        ou_coefficient=0.9,
        ou_amplitude=0.2,
        position_amplitude=0.02,
    )
    assert math.isfinite(envelope.log_survival_parameter)
    assert envelope.log_survival_parameter < -700
    assert math.isfinite(envelope.log_qsd_eigenvalue_lower_bound(10))


def test_stage_moments_do_not_inherit_survival_noise_or_erosion_requirements():
    budget = quadratic_uniform_kinetic_budget(
        dimension=3,
        moment_order=2,
        timestep=1,
        viscosity=0.3,
        quadratic_coefficient=1,
        terminal_half_width=0.1,
        clone_jitter=0,
        restitution=0.5,
        velocity_cap=2,
        ou_coefficient=1,
        ou_amplitude=0,
        position_amplitude=0,
        row_normalized=True,
    )
    assert math.isfinite(budget.second_kick_velocity)
    assert budget.output_velocity == 2


@pytest.mark.parametrize("row_normalized", [False, True])
def test_native_cubic_moment_budget_against_exact_gaussian_polynomial_quadrature(row_normalized):
    budget = styblinski_tang_uniform_kinetic_budget(
        dimension=1,
        moment_order=2,
        timestep=0.04,
        viscosity=0.3,
        terminal_half_width=2,
        clone_jitter=0.1,
        restitution=0.5,
        velocity_cap=2,
        ou_coefficient=0.9,
        ou_amplitude=0,
        position_amplitude=0,
        row_normalized=row_normalized,
    )
    # With N=1 viscosity is exactly zero. The uncapped B2 velocity is a
    # degree-nine polynomial of clone jitter. Ten Hermite nodes integrate its
    # degree-eighteen square exactly, without truncating the Gaussian law.
    nodes, weights = np.polynomial.hermite.hermgauss(10)
    x = 2 + 0.1 * math.sqrt(2) * nodes

    def force(z):
        return -2 * z**3 + 16 * z - 2.5

    u = 0.02 * force(x)
    w = 0.9 * u
    y = x + 0.02 * (u + w)
    z = w + 0.02 * force(y)
    exact_rms = math.sqrt(np.dot(weights, z**2) / math.sqrt(math.pi))
    assert exact_rms <= budget.second_kick_velocity
    assert math.sqrt(np.dot(weights, y**2) / math.sqrt(math.pi)) <= budget.output_position
