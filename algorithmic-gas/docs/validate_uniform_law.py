#!/usr/bin/env python3
"""Check exact arithmetic in the alive-law parameter and normalization proofs.

The Fraction checks reproduce the rational inequalities in the proof; the
elementary inequalities e < 3, pi < 4, and exp(-1/4) > 3/4 are analytic inputs.
Finite algebra examples are regressions; the universal proofs remain in the chapters.
Floating evaluations are labelled diagnostics and are not interval certificates.
"""

from fractions import Fraction as Q
import math


def check_exact_bounds() -> None:
    """Validate each displayed rational margin, independently of floating point."""
    c = a = q_squared = s_squared = Q(1)
    step = Q(2)
    velocity_cap = Q(1, 1000)
    collision = Q(1, 2)
    force_lipschitz = Q(1, 2)
    drift = c * (1 + a)
    eta = c * drift
    collision_cap = (1 + 2 * collision) * velocity_cap
    tau_squared = c * c * q_squared + s_squared
    assert eta * force_lipschitz == 1
    assert drift == 2 and collision_cap == Q(1, 500) and tau_squared == 2
    assert Q(1, 2) * step == q_squared == s_squared

    base_moment = 1 + (drift * velocity_cap) ** 2 + tau_squared
    radius_squared = 4 * base_moment - 1
    assert radius_squared == 11 + Q(16, 10**6)
    assert radius_squared < Q(10, 3) ** 2
    first_radius = Q(1668, 1000)
    ou_center = first_radius
    inverse_radius = Q(1768, 1000)
    exponent = (inverse_radius + ou_center) ** 2 / 2
    exponent += (Q(1, 10) + first_radius + inverse_radius) ** 2 / 2
    assert exponent < 13
    epsilon = Q(5, 10**10)
    assert Q(1, 800 * 3**13) > epsilon

    fourth_moment = Q(16)
    assert Q(199, 100) ** 4 > 12
    assert Q(4, 1000) + Q(199, 100) < 2
    inverse_beta = 2 * fourth_moment / epsilon
    assert inverse_beta == 64 * 10**9
    assert inverse_beta < (3 * 10**5) ** 2
    reward_growth = Q(1, 4)
    reward_envelope = Q(40000)
    reward_square_envelope = Q(8 * 10**9)
    reward_mean = Q(2)
    variance_envelope = Q(2 * 10**10)
    assert reward_growth * (1 + Q(3 * 10**5, 2)) < reward_envelope
    assert 2 * reward_growth**2 * inverse_beta == reward_square_envelope
    assert reward_growth * (1 + 4) < reward_mean
    assert 2 * reward_square_envelope + 4 * reward_mean * reward_envelope < variance_envelope

    inverse_kappa = Q(4, 3)
    measurement_factor = Q(4)
    diversity_sensitivity = Q(66)
    assert 1 + 2 * inverse_kappa < measurement_factor
    assert measurement_factor * (3 + Q(27, 2)) == diversity_sensitivity
    gate_range_coefficient = Q(64, 10)
    gate_derivative = Q(6)
    assert Q(8) / Q(5, 4) == gate_range_coefficient
    assert 1 / Q(5, 4) + 4 * 2 / Q(5, 4) ** 2 < gate_derivative
    fitness_sensitivity = Q(67)
    reward_regularizer = Q(10**6)
    assert (
        2 * reward_envelope / reward_regularizer
        + reward_mean * variance_envelope / reward_regularizer**3
        + diversity_sensitivity
        < fitness_sensitivity
    )
    edge_sensitivity = Q(1200)
    assert (
        2 * gate_range_coefficient * measurement_factor * inverse_kappa
        + gate_range_coefficient * inverse_kappa**2
        + 2 * gate_derivative * fitness_sensitivity * inverse_kappa
        < edge_sensitivity
    )
    full_sensitivity = Q(5000)
    assert (
        2 * gate_range_coefficient * (1 + inverse_kappa) * measurement_factor
        + 4 * edge_sensitivity
        < full_sensitivity
    )

    theta = Q(1, 10**15)
    weighted_mass = Q(3, 2)
    assert theta < 1
    assert theta < Q(3, 4) / (4 * gate_range_coefficient)
    assert theta < epsilon / (8 * weighted_mass**2 * full_sensitivity)
    residual_lipschitz = weighted_mass * theta * full_sensitivity
    assert residual_lipschitz == Q(75, 10**13)
    assert epsilon / 2 - 2 * weighted_mass * residual_lipschitz > Q(2, 10**10)
    assert epsilon / 2 - 2 * weighted_mass * residual_lipschitz - residual_lipschitz**2 > Q(
        2, 10**10
    )
    assert 1 + 4 * (4 + velocity_cap**2) < 18

    prepared_base_moment = 1 + (drift * collision_cap) ** 2 + tau_squared
    prepared_radius_squared = 4 * prepared_base_moment - 1
    assert prepared_radius_squared == 11 + Q(64, 10**6)
    assert prepared_radius_squared < Q(10, 3) ** 2
    finite_measurement_factor = Q(7)
    finite_diversity_sensitivity = Q(18)
    finite_reward_sensitivity = Q(1, 10**5)
    assert 1 + 4 * inverse_kappa < finite_measurement_factor
    assert 3 + Q(27, 2) < finite_diversity_sensitivity
    assert Q(1, 10**6) + Q(1, 2 * 10**18) < finite_reward_sensitivity
    finite_edge_coefficient = Q(2000)
    assert (
        4 * gate_range_coefficient * inverse_kappa
        + (4 * gate_range_coefficient * inverse_kappa + 2 * gate_range_coefficient)
        * finite_measurement_factor
        + 2
        * gate_derivative
        * (finite_reward_sensitivity + finite_measurement_factor * finite_diversity_sensitivity)
        < finite_edge_coefficient
    )
    finite_error_coefficient = Q(20000)
    assert (
        30 * gate_range_coefficient * inverse_kappa + 9 * finite_edge_coefficient
        < finite_error_coefficient
    )
    assert theta < Q(3, 4) / (8 * gate_range_coefficient)
    assert theta < epsilon / (8 * finite_error_coefficient)
    assert epsilon / Q(2) == Q(25, 10**11)
    assert 4 * (4 + velocity_cap**2) < 17


def check_survival_normalization_algebra() -> None:
    """Check sharp positive tilts and the finite-future oscillation recurrence."""
    for lower_root, upper_root in ((Q(1), Q(2)), (Q(1, 2), Q(3, 2)), (Q(2), Q(3))):
        lower = lower_root**2
        upper = upper_root**2
        sharp = (upper_root - lower_root) / (upper_root + lower_root)
        weights = (lower, (lower + upper) / 2, upper)
        for first_count in range(13):
            for second_count in range(13 - first_count):
                probabilities = (
                    Q(first_count, 12),
                    Q(second_count, 12),
                    Q(12 - first_count - second_count, 12),
                )
                mean = sum(p * w for p, w in zip(probabilities, weights, strict=True))
                distance = (
                    sum(p * abs(w / mean - 1) for p, w in zip(probabilities, weights, strict=True))
                    / 2
                )
                chord = (upper - mean) * (mean - lower) / (mean * (upper - lower))
                assert distance <= chord <= sharp
                square_certificate = (mean - lower_root * upper_root) ** 2 / (
                    mean * (upper - lower)
                )
                assert sharp - chord == square_certificate

        # The endpoint law with this mean attains the universal analytic bound.
        probabilities = (
            upper_root / (lower_root + upper_root),
            lower_root / (lower_root + upper_root),
        )
        mean = sum(p * w for p, w in zip(probabilities, (lower, upper), strict=True))
        assert mean == lower_root * upper_root
        distance = (
            sum(p * abs(w / mean - 1) for p, w in zip(probabilities, (lower, upper), strict=True))
            / 2
        )
        assert distance == sharp

    for beta in (Q(0), Q(1, 4), Q(1, 2)):
        for killed_mass in (Q(0), Q(1, 20), Q(1, 10)):
            assert beta + 2 * killed_mass < 1
            coefficient = (beta + killed_mass) / (1 - killed_mass)
            additive = killed_mass / (1 - killed_mass)
            fixed_point = killed_mass / (1 - beta - 2 * killed_mass)
            ratio_bound = (1 - beta - killed_mass) / (1 - beta - 2 * killed_mass)
            assert coefficient * fixed_point + additive == fixed_point
            assert 1 + fixed_point == ratio_bound
            normalized_rows = (
                ((Q(1), Q(0)), (1 - beta, beta)),
                (((1 + beta) / 2, (1 - beta) / 2), ((1 - beta) / 2, (1 + beta) / 2)),
            )
            for first_row, second_row in normalized_rows:
                row_tv = sum(abs(a - b) for a, b in zip(first_row, second_row, strict=True)) / Q(2)
                assert row_tv == beta
                for values in ((Q(1), Q(1)), (1 - killed_mass, Q(1))):
                    for _ in range(41):
                        oscillation = max(values) - min(values)
                        relative = oscillation / min(values)
                        assert max(values) / min(values) <= ratio_bound
                        updated = (
                            (1 - killed_mass)
                            * sum(a * f for a, f in zip(first_row, values, strict=True)),
                            sum(a * f for a, f in zip(second_row, values, strict=True)),
                        )
                        assert max(updated) - min(
                            updated
                        ) <= beta * oscillation + killed_mass * max(values)
                        assert min(updated) >= (1 - killed_mass) * min(values)
                        assert (max(updated) - min(updated)) / min(updated) <= (
                            coefficient * relative + additive
                        )
                        values = updated


def floating_diagnostics() -> dict[str, float]:
    """Evaluate formulas for orientation; these values do not establish the proof."""
    theta = 1e-15
    cap = 1e-3
    collision_cap = 2 * cap
    b = eta = 2.0
    tau_squared = 2.0
    base_moment = 1 + (b * cap) ** 2 + tau_squared
    radius = math.sqrt(4 * base_moment - 1)
    first_radius = ou_center = radius / 2 + cap
    inverse_radius = 0.1 + first_radius
    exponent = (inverse_radius + ou_center) ** 2 / 2
    exponent += (0.1 + first_radius + inverse_radius) ** 2 / 2
    epsilon = math.exp(-exponent) / (200 * math.pi)
    moment = (b * collision_cap + math.sqrt(tau_squared) * 3**0.25) ** 4
    beta = epsilon / (2 * moment)
    weighted_mass = 1 + epsilon / 2
    reward_envelope = 0.25 * (1 + 1 / (2 * math.sqrt(beta)))
    reward_square_envelope = 0.125 * max(1, 1 / beta)
    reward_mean = 0.25 * (1 + math.sqrt(moment))
    variance_envelope = 2 * reward_square_envelope + 4 * reward_mean * reward_envelope
    kappa = math.exp(-0.25)
    measurement_factor = 1 + 2 / kappa
    diversity_range = math.sqrt(8 + 1e-6) - 1e-3
    diversity_sensitivity = measurement_factor * (diversity_range + diversity_range**3 / 2)
    fitness_sensitivity = (
        2 * reward_envelope / 1e6 + reward_mean * variance_envelope / 1e18 + diversity_sensitivity
    )
    delta = math.log(4)
    denominator = 1.25
    gate_range = 4 * delta / denominator
    gate_derivative = 1 / denominator + 4 * delta / denominator**2
    edge_sensitivity = (
        2 * gate_range * measurement_factor / kappa
        + gate_range / kappa**2
        + 2 * gate_derivative * fitness_sensitivity / kappa
    )
    full_sensitivity = 2 * gate_range * (1 + 1 / kappa) * measurement_factor + 4 * edge_sensitivity
    theta_max = min(
        1,
        kappa / (4 * gate_range),
        epsilon / (8 * weighted_mass**2 * full_sensitivity),
    )

    # expm1 retains the small configured fitness range without cancellation.
    fitness_range = math.expm1(2 * theta * math.log(2))
    acceptance = fitness_range / 2
    incoming = acceptance / kappa
    derivative = 0.5 + fitness_range / 4
    powered_derivative = theta * math.exp(theta * math.log(2)) / 4
    actual_fitness_sensitivity = powered_derivative * fitness_sensitivity
    edge_lipschitz = (
        2 * incoming * measurement_factor
        + acceptance / kappa**2
        + 2 * derivative * actual_fitness_sensitivity / kappa
    )
    residual_lipschitz = weighted_mass * (
        2 * (acceptance + incoming) * measurement_factor + 2 * edge_lipschitz / (1 - 2 * incoming)
    )
    contraction_gap = epsilon / 2 - 2 * weighted_mass * residual_lipschitz - residual_lipschitz**2
    prepared_base_moment = 1 + (b * collision_cap) ** 2 + tau_squared
    prepared_radius = math.sqrt(4 * prepared_base_moment - 1)
    prepared_first_radius = prepared_ou_center = prepared_radius / 2 + cap
    prepared_inverse_radius = 0.1 + prepared_first_radius
    prepared_exponent = (prepared_inverse_radius + prepared_ou_center) ** 2 / 2
    prepared_exponent += (0.1 + prepared_first_radius + prepared_inverse_radius) ** 2 / 2
    finite_epsilon = math.exp(-prepared_exponent) / (200 * math.pi)
    finite_measurement_factor = 1 + 4 / kappa
    finite_diversity_sensitivity = diversity_range + diversity_range**3 / 2
    finite_reward_sensitivity = 1e-6 + 0.5e-18
    finite_edge_coefficient = (
        4 * gate_range / kappa
        + (4 * gate_range / kappa + 2 * gate_range) * finite_measurement_factor
        + 2
        * gate_derivative
        * (finite_reward_sensitivity + finite_measurement_factor * finite_diversity_sensitivity)
    )
    finite_error_coefficient = 30 * gate_range / kappa + 9 * finite_edge_coefficient
    finite_theta_max = min(
        1, kappa / (8 * gate_range), finite_epsilon / (8 * finite_error_coefficient)
    )
    finite_edge_lipschitz = (
        4 * incoming
        + (4 * incoming + 2 * acceptance) * finite_measurement_factor
        + 2
        * derivative
        * powered_derivative
        * (finite_reward_sensitivity + finite_measurement_factor * finite_diversity_sensitivity)
    )
    component_extra = math.expm1(8 * incoming)
    finite_e1 = component_extra + 3 * (1 + component_extra) * finite_edge_lipschitz
    finite_e2 = finite_e1 + 6 * incoming
    finite_gap = (
        finite_epsilon - finite_e1 - finite_e2 + finite_epsilon * finite_e1 - finite_e1 * finite_e2
    )
    return {
        "eta": eta,
        "epsilon_2": epsilon,
        "M_4": moment,
        "beta_w": beta,
        "H_0": full_sensitivity,
        "theta_max_w": theta_max,
        "configured_theta": theta,
        "L_rem": residual_lipschitz,
        "1_minus_q_w": contraction_gap,
        "epsilon_f": finite_epsilon,
        "E_2": finite_error_coefficient,
        "theta_max_f": finite_theta_max,
        "1_minus_q_f": finite_gap,
    }


def check_reference_cap_certificates() -> None:
    """Check the RCAP endpoint matrices and noisy-cap margins using fractions."""
    step_half = Q(1, 50)
    beta = Q(1, 25)
    c_lo, c_hi = Q("0.96"), Q("0.9608")
    drift_lo, drift_hi = step_half * (1 + c_lo), step_half * (1 + c_hi)
    position_lo = 1 - step_half * drift_hi
    position_hi = 1 - step_half * drift_lo
    restoring_lo = step_half * (c_lo + position_lo)
    restoring_hi = step_half * (c_hi + position_hi)
    velocity_lo = c_lo - step_half * drift_hi
    velocity_hi = c_hi - step_half * drift_lo
    margin = Q(1, 1000)

    diagonal_0x = 1 - position_hi**2 - margin
    diagonal_0v = 1 - drift_hi**2 - margin
    cross_0 = max(
        abs(beta - position_lo * drift_lo), abs(beta - position_hi * drift_hi)
    )
    assert diagonal_0x > Q("0.00056") and diagonal_0v > Q("0.997")
    assert cross_0 < Q("0.00084")
    assert Q("0.00056") * Q("0.997") > Q("0.00084") ** 2

    diagonal_1x = (
        1 - position_hi**2 + 2 * beta * position_lo * restoring_lo - restoring_hi**2 - margin
    )
    diagonal_1v = (
        1 - drift_hi**2 - 2 * beta * drift_hi * velocity_hi - velocity_hi**2 - margin
    )
    cross_1lo = (
        beta
        - position_hi * drift_hi
        - beta * position_hi * velocity_hi
        + beta * drift_lo * restoring_lo
        + restoring_lo * velocity_lo
    )
    cross_1hi = (
        beta
        - position_lo * drift_lo
        - beta * position_lo * velocity_lo
        + beta * drift_hi * restoring_hi
        + restoring_hi * velocity_hi
    )
    assert diagonal_1x > Q("0.0021") and diagonal_1v > Q("0.0727")
    assert max(abs(cross_1lo), abs(cross_1hi)) < Q("0.00025")
    assert Q("0.0021") * Q("0.0727") > Q("0.00025") ** 2
    assert margin / (1 + beta) == Q(1, 1040)

    q_lo, q_hi = Q("0.196"), Q("0.198")
    assert q_lo**2 < (1 - c_hi**2) / 2
    assert (1 - c_lo**2) / 2 < q_hi**2
    sqrt_dimension_hi = Q("1.733")
    gaussian_width = Q(1, 10)
    viscosity = Q(3, 10)
    cap = Q(2)
    prepared_cap = Q(4)
    kernel_gradient_hi = Q("0.607")
    modified_mass = 1 - step_half**2
    inverse_margin = modified_mass - step_half * viscosity
    provider_moment_hi = (
        c_hi * prepared_cap
        + c_hi * step_half * (2 + gaussian_width) * sqrt_dimension_hi
        + q_hi * sqrt_dimension_hi
    )
    first_position_hi = modified_mass * (2 * sqrt_dimension_hi + 1)
    first_position_hi += step_half * prepared_cap
    assert provider_moment_hi < Q("4.26") and first_position_hi < Q("4.545")
    center_hi = step_half * Q("4.545") + step_half * viscosity * Q("4.26")
    assert center_hi < Q("0.117")
    assert Q("0.117") / inverse_margin < Q("0.118")
    graph_derivative_hi = step_half**2 * viscosity * kernel_gradient_hi
    assert modified_mass + graph_derivative_hi * Q("4.378") < Q("0.99992")
    assert step_half + step_half * viscosity * kernel_gradient_hi * Q("4.378") < Q(
        "0.036"
    )
    assert cap / inverse_margin < Q("2.014")
    assert (cap / 4 + Q("0.117")) / inverse_margin < Q("0.622")
    tail_derivative_hi = cap * (
        modified_mass + graph_derivative_hi * (Q("4.26") + q_hi)
    ) / (cap + inverse_margin * q_lo - Q("0.117"))
    assert tail_derivative_hi < Q("0.963")
    assert (15 * Q("0.99992") ** 2 + Q("0.963") ** 2) / 16 < Q("0.998") ** 2


def check_reference_velocity_burn() -> None:
    """Check RVB burn and full-Gaussian feedback certificates exactly."""
    def check_transition(start: Q, finish: Q, survival_factor: Q = Q(1)) -> None:
        energy = (Q("0.961") * start + Q("0.137")) ** 2 + Q("0.118")
        raw_finish = finish / survival_factor
        assert energy < (2 * raw_finish / (2 - raw_finish)) ** 2

    population_bounds = tuple(
        map(Q, ("2", "1.022", ".739", ".628", ".580", ".559", ".55"))
    )
    for start, finish in zip(population_bounds, population_bounds[1:]):
        check_transition(start, finish)
    check_transition(Q(".55"), Q(".545"))
    check_transition(Q(".545"), Q(".545"))
    survivor_bounds = tuple(
        map(Q, ("2", "1.032", ".750", ".639", ".591", ".570", ".560"))
    )
    for start, finish in zip(survivor_bounds, survivor_bounds[1:]):
        check_transition(start, finish, Q("1.01"))
    check_transition(Q(".56"), Q(".56"), Q("1.01"))
    assert Q(".99") * Q("1.01") ** 2 > 1

    step_half, viscosity = Q(".02"), Q(".3")
    mass = 1 - step_half**2
    source_second = Q("3.47")
    first_position = mass * source_second + step_half * 4
    assert source_second**2 > Q("12.03") and first_position < Q("3.55")
    ou_variance = Q(".0392")
    for speed, provider_limit in ((Q(".55"), Q(".69")), (Q(".56"), Q(".70"))):
        energy = Q(".9608") ** 2 * (speed + step_half * source_second) ** 2
        assert energy + 3 * ou_variance < provider_limit**2

    inverse_margin = mass - step_half * viscosity
    center = step_half * Q("3.55") + step_half * viscosity * Q(".70")
    assert center < Q(".076")
    radial_velocity = (Q(".5") + center) / inverse_margin
    assert radial_velocity < Q(".58")
    kernel_gradient = Q(".607")
    jacobian_velocity = mass + step_half**2 * viscosity * kernel_gradient * Q("1.28")
    jacobian_position = step_half + step_half * viscosity * kernel_gradient * Q("1.28")
    assert jacobian_velocity < Q(".9997") and jacobian_position < Q(".0247")
    drift_upper = step_half * (1 + Q(".9608"))
    feedback_squared = (drift_upper * step_half * viscosity) ** 2
    feedback_squared += (
        Q(".9608") * step_half * viscosity * Q(".9997")
        + step_half**2 * viscosity * Q(".0247")
    ) ** 2
    assert feedback_squared < Q(".00578") ** 2


def check_reference_two_count_balance() -> None:
    """Check RFK principal losses and correlated radial forcing bounds."""
    beta, step_half, viscosity = Q(".04"), Q(".02"), Q(".3")
    alignment = step_half * viscosity
    mass = 1 - step_half**2
    drift_lo, drift_hi = Q(".0392"), Q(".039216")
    position_lo, position_hi = Q(".99921568"), Q(".999216")
    force_lo, force_hi = Q(".0391843136"), Q(".03920032")
    velocity_lo, velocity_hi = Q(".95921568"), Q(".960016")
    f_lo, f_hi = beta * position_lo - force_hi, beta * position_hi - force_lo
    g_lo, g_hi = velocity_lo + beta * drift_lo, velocity_hi + beta * drift_hi
    position_loss, velocity_loss = Q(".0015"), Q(".073")
    assert 1 - position_hi**2 - f_hi**2 - position_loss > Q(".00006")
    assert 1 - drift_hi**2 - g_hi**2 - velocity_loss > Q(".0008")
    cross_lo = beta - position_hi * drift_hi - f_hi * g_hi
    cross_hi = beta - position_lo * drift_lo - f_lo * g_lo
    assert max(abs(cross_lo), abs(cross_hi)) < Q(".0001")
    assert Q(".00006") * Q(".0008") > Q(".0001") ** 2
    assert position_hi**2 + drift_hi**2 < 1

    defect_factor = alignment / (2 - alignment)
    assert position_loss - defect_factor * (beta**2 + (beta - step_half) ** 2) > Q(
        ".00149"
    )
    assert (velocity_loss - defect_factor * (beta - step_half) ** 2) * (
        1 - alignment
    ) ** 2 > Q(".0721")
    assert Q(".00149") / (1 + beta) == Q(149, 104000)

    source_second, source_product = Q("3.47"), Q("4.478")
    assert source_product**2 > Q("20.05")
    kernel_gradient, prepared_cap, prepared_second = Q(".607"), Q(4), Q(".55")
    provider_second, radial_second = Q(".70"), Q(".58")
    inverse_margin = mass - alignment
    spatial_force = drift_hi * alignment * kernel_gradient * (prepared_cap + prepared_second)
    landing_position = position_hi + spatial_force
    landing_x, landing_p = position_hi + 2 * spatial_force, drift_hi
    product_x = landing_position * source_product + spatial_force * source_second
    product_p = drift_hi * (1 + alignment) * source_second
    radial_base = (Q(".5") + alignment * provider_second + step_half**2 * prepared_cap)
    radial_base /= inverse_margin
    radial_position = step_half * mass / inverse_margin
    radial_factor = 2 * provider_second + radial_base + radial_second
    assert alignment * kernel_gradient * (
        radial_factor * landing_x + radial_position * product_x
    ) < Q(".0094")
    assert alignment * kernel_gradient * (
        radial_factor * landing_p + radial_position * product_p
    ) < Q(".000366")


def check_reference_empirical_provider() -> None:
    """Check DEV full-Gaussian empirical provider thresholds exactly."""
    assert Q(".56") ** 2 - Q(".55") ** 2 == Q(".0111")
    assert Q(".4806") / Q(".22") ** 2 < 10
    assert Q(".9608") ** 2 * Q(".63") ** 2 < Q(".367")
    assert Q(".367") + 3 * Q(".0392") < Q(".485")
    variance = 4 * Q(".0392") * Q(".367") + 6 * Q(".0392") ** 2
    assert variance < Q(".067")
    assert Q(".067") / Q(".005") ** 2 == 2680
    assert 10 + 2680 == 2690


def check_reference_restricted_transport() -> None:
    """Check RSA centered-force absorption and TAT alive-law certificates."""
    beta = Q(".04")
    position_loss, velocity_loss = Q(".00149"), Q(".0721")
    delta = position_loss / (1 + beta)
    radius = Q(1, 128)
    force_ratio_squared = Q(".0414") ** 2 / position_loss
    force_ratio_squared += Q(".000366") ** 2 / velocity_loss
    assert force_ratio_squared < Q("1.074") ** 2
    assert delta > Q(".03785") ** 2
    assert 2 * radius * Q("1.074") / Q(".03785") + radius**2 * Q(
        "1.074"
    ) ** 2 < Q(1, 2)
    centered_ratio = (1 + beta) / (1 - beta) * radius**2 / (1 - radius**2)
    assert (1 - delta / 2) * (1 + centered_ratio) < 1 - delta / 3
    assert Q(".999216") ** 2 + Q(".03920032") ** 2 < 1 - Q(1, 40000)
    assert 40 * 640000 < 2**50
    tail_radius = Q(".5") - Q("3.97") * Q(".039216")
    copied_variance = Q(".00041568") + Q(".999216") ** 2 / 100
    assert tail_radius**2 > Q(100, 9) * copied_variance
    assert sum(Q(50, 9) ** k / math.factorial(k) for k in range(11)) > 250
    assert Q(18, 6250) < Q(".003")
    alignment = Q(".006")
    defect_factor = alignment / (2 - alignment)
    noise = 3 * Q(".0392") * (
        Q(".0004") + Q("1.0004") ** 2 + Q(".0004") * defect_factor * Q(".02") ** 2
    ) + Q(".0012")
    assert noise < Q(".119")


def check_reference_position_shapes() -> None:
    """Check DSA53 whole-cap shape margins and TQP own-survival weights."""
    step_half, alignment, beta = Q(".02"), Q(".006"), Q(".04")
    ou_variance = Q(".0392")
    assert sum(Q(1, math.factorial(k)) for k in range(7)) > 1 / Q(".368")
    assert sum(Q(1, math.factorial(k)) for k in range(7)) ** 2 > 1 / Q(".136")
    pair = 16 * step_half**2 * Q(".136") / Q(".998")
    pair += (
        4 * step_half**4 * ou_variance / Q(".998") + 2 * ou_variance
    ) * 6 * Q(".368")
    assert pair < Q(".175") and 2 * pair < Q(".592") ** 2
    force = alignment * Q(".592")
    assert force == Q(".003552")
    f_upper = (beta - step_half) * Q(".999216") - Q(".96") * step_half
    assert f_upper + Q(".9608") * step_half * alignment < Q(".0009")
    assert Q(".9608") + (beta - step_half) * Q(".039216") < Q(".962")
    position_margin = Q(".00149") - 2 * Q(".0009") * force - force**2 - Q(".0004")
    velocity_margin = Q(".0721") - (Q(".962") * force) ** 2 / Q(".0004")
    assert position_margin == Q(".001070989696") > Q(".001")
    assert velocity_margin == Q(".04290986745856") > Q(".04")
    assert 1 - Q(".001") / (1 + beta) == Q(1039, 1040)
    assert Q(".999216") ** 2 + (Q(".0009") + force) ** 2 < Q(1997, 2000)
    for alive_probability in (Q(1, 100), Q(1, 4), Q(1, 2), Q(3, 4), Q(99, 100)):
        survival = alive_probability * (2 - alive_probability)
        singleton = 2 * alive_probability * (1 - alive_probability) / survival
        double = alive_probability**2 / survival
        assert singleton == 2 * (1 - alive_probability) / (2 - alive_probability)
        assert double == alive_probability / (2 - alive_probability)
        assert singleton + double == 1
    assert Q(5, 3) < 2 and Q(5, 6) - 1 == Q(-1, 6)


def check_reference_boundary_response() -> None:
    """Check DSTI Gaussian event and finite survival-response arithmetic."""
    exponent = Q(128, 25)
    assert sum(exponent**k / math.factorial(k) for k in range(10)) > Q(625, 4)
    assert Q(1, 8) / Q(625, 4) == Q(".0008")
    exponent = Q(729, 200)
    upper = sum(exponent**k / math.factorial(k) for k in range(6))
    upper += exponent**6 / (math.factorial(6) * (1 - exponent / 7))
    assert upper < 39 and 8 / Q(2**12) < Q(1, 500)
    event = Q(7, 8) ** 2 * Q(5, 16) * Q(3, 624) * Q(499, 500)
    assert event > Q(".001")
    center = Q("1.436") + Q(".0392") * (Q(".55") - Q(".02") * Q("1.436"))
    assert center == Q("1.456434176")
    assert center + Q(".02") * Q("2.2") > Q("1.5")
    assert Q("1.456") + Q(".04") * Q(".55") + Q(".023") * Q("2.7") < 2
    assert Q("1.5") / Q(".4996") < Q("3.01") and 2 * Q("3.01") == Q("6.02")
    for count in range(1, 30):
        assert Q(1, count**2) <= Q(6, (count + 1) * (count + 2))
    # Finite algebra regressions; the universal identity is proved in DSTI.17.
    step_half, mass = Q(".02"), 1 - Q(".02") ** 2
    for damping in (Q(".96"), Q(".9608"), Q(1)):
        drift = step_half * (1 + damping)
        position, velocity = 1 - step_half * drift, damping - step_half * drift
        force = mass * drift
        cross = (1 - damping) / (2 * drift)
        assert position - velocity == 1 - damping
        assert position * velocity + drift * force == damping
        assert mass * position**2 - 2 * cross * position * force + force**2 == damping * mass
        assert mass * drift**2 + 2 * cross * drift * velocity + velocity**2 == damping
        assert (
            mass * position * drift + cross * (position * velocity - drift * force)
            - force * velocity
        ) == damping * cross


def check_reference_signed_shape_classes() -> None:
    """Check CSB/DSG full-force and signed varying-velocity certificates."""
    alignment = Q(".006")
    noise = (8 * Q(".0004") * Q(".136") + 12 * Q(".0392") * Q(".368")) / Q(".998")
    pair = Q(51, 50) * noise + 51 * Q(".0001") * Q(".368") / Q(".998")
    assert pair < Q(".18") and 2 * pair < Q(".6") ** 2
    assert 2 * Q("1.415") * Q(".005") * Q(".607") < Q(".009")
    alpha, force_x, force_p = Q(".00000212"), Q(".00365"), Q(".000142")
    assert Q(".039216") * alignment * Q(".009") < alpha
    assert alignment * Q(".962") * Q(".009") + Q(".0036") * (
        Q(".999216") + alpha
    ) < force_x
    assert Q(".0036") * Q(".039216") < force_p
    cost_x = 2 * Q(".999216") * alpha + alpha**2 + 2 * Q(".0009") * force_x + force_x**2
    cost_p = 2 * Q(".962") * force_p + force_p**2
    cost_cross = Q(".039216") * alpha + Q(".0009") * force_p
    cost_cross += Q(".962") * force_x + force_x * force_p
    assert Q(".00149") - cost_x - Q(".0004") == Q(".0010658708196656") > Q(".001")
    assert Q(".0721") - cost_p - cost_cross**2 / Q(".0004") > Q(".0409") > Q(".04")
    assert Q(".999216") - Q(".0392") * Q(".994") / 2 == Q(".9797336")
    assert Q(".49972") - Q(".02") * Q(".9796") == Q(".480128")
    baseline = Q(".9797336") ** 2 + Q(".480128") ** 2
    assert baseline == Q("1.19040082335296") < Q("1.1905")
    assert -Q(".18") - Q(".0216") * Q(".36") > -Q(".188")
    assert Q(".49632") > Q(".188") and Q(".49632") > Q(".02")
    gaussian_quadratic = Q(".135") + Q(".0012") + Q(".36") * Q("1.001") ** 2
    assert gaussian_quadratic == Q(".49692036") < Q(".5")
    assert Q(".006") ** 2 * (Q(".98") + Q(".5")) ** 2 < Q(".000081")
    assert Q("1.190581") < Q(".99") * Q("1.21")
    assert Q(".995") ** 2 > Q(".99") and 1 + Q(".04") < Q(67, 64)
    assert 25 * 67 < 42**2 and 42 * 100000 < 2**50
    assert Q(".995") + Q(".001") == Q(249, 250)


def check_reference_bilinear_coefficients() -> None:
    """Check DBL signed coefficients and completion by exact rational algebra."""
    lower_r = Q(".0392") * Q(".99921568") + Q(".96078368") * Q(".0007683072")
    upper_r = Q(".039216") * Q(".999216") + Q(".96158464") * Q(".0007843264")
    assert lower_r == Q(2435756327819, 61035156250000) > Q(".0399074")
    assert upper_r == Q(1218855312347, 30517578125000) < Q(".0399395")
    lower_p = Q(".0392") ** 2 + Q(".96078368") ** 2
    upper_p = Q(".039216") ** 2 + Q(".96158464") ** 2
    assert lower_p == Q(".9246419197543424") > Q(".92464")
    assert upper_p == Q(".9261829145399296") < Q(".926183")
    assert Q(".006") * upper_r < Q(".000240")
    assert Q(".006") * Q(".0007843264") < Q(".000004706")
    # Finite exact regressions; DBL.7 gives the universal square expansion.
    for displacement in (Q(-2), Q(1, 3), Q(3)):
        for velocity in (Q(-3), Q(1, 5), Q(2)):
            for product in (Q(-1), Q(0), Q(5, 7)):
                raw = (lower_r * displacement + upper_p * velocity) * (product - velocity)
                complete = -upper_p * (
                    velocity - product / 2 + lower_r * displacement / (2 * upper_p)
                ) ** 2
                complete += upper_p * product**2 / 4 + lower_r * displacement * product / 2
                complete += lower_r**2 * displacement**2 / (4 * upper_p)
                assert raw == complete


def check_reference_conditional_cap() -> None:
    """Check ICB primitive and CCL local/environment bounds exactly."""
    pair = 3 * 64 * Q(".368") + 12 * Q(".0004") * Q(".136")
    pair += 18 * Q(".0392") * Q("1.001") ** 2 * Q(".368")
    assert pair == Q("70.9168331812608") < Q("8.5") ** 2
    assert Q("3.469") ** 2 > Q("12.03")
    ou_moment = Q(".9608") ** 2 * (Q(".56") + Q(".02") * Q("3.469")) ** 2
    ou_moment += 3 * Q(".0392")
    assert ou_moment == Q(1887781769244361, 3906250000000000) < Q(".49")
    remainder = Q(".006") * Q(".0004") * Q(".368") / 2
    remainder += Q(".006") ** 3 * Q(".49")
    remainder += Q(".006") * Q(".02") * Q(".0392") * Q("8.5")
    assert remainder == Q(".00004053144") < Q(".000041")
    assert Q(".607") * (4 + 3 * Q(".55")) == Q("3.42955") < Q("3.43")
    assert Q(".607") * (4 + 3 * Q(".56")) == Q("3.44776") < Q("3.448")
    assert Q(".607") ** 2 * Q(163, 60) > 1
    population_force = Q(".607") * (4 + Q(".55"))
    finite_force = Q(".607") * (4 + Q(".56"))
    assert population_force == Q("2.76185") < Q(10, 3)
    assert finite_force == Q("2.76792") < Q(10, 3)
    population_gap = Q(".02") * Q(".0392") * (1 - Q(".3") * population_force)
    finite_gap = Q(".02") * Q(".0392") * (1 - Q(".3") * finite_force)
    assert population_gap == Q(".00013441288")
    assert finite_gap == Q(".000132985216")
    assert 1 - population_gap < Q(".999866")
    assert 1 - finite_gap < Q(".999868")
    # Finite exact regressions; CCL.3 proves the identity for every cap eigenvalue.
    beta = Q(".04")
    for cap_eigenvalue in (Q(0), Q(1, 10), Q(2, 3), Q(1)):
        for displacement, velocity in ((Q(1), Q(2)), (Q(-3), Q(5, 7))):
            raw = ((1 - cap_eigenvalue) * velocity + beta * displacement) ** 2
            raw += 2 * cap_eigenvalue * (1 - cap_eigenvalue) * velocity**2
            complete = (1 - cap_eigenvalue**2) * (
                velocity + beta * displacement / (1 + cap_eigenvalue)
            ) ** 2
            complete += 2 * beta**2 * cap_eigenvalue * displacement**2 / (
                1 + cap_eigenvalue
            )
            assert raw == complete


def check_reference_full_gaussian_cap() -> None:
    """Check GCA74's twelve full-tail thresholds and conditional deficit."""
    half_step, alignment, mass = Q(".02"), Q(".006"), Q(".9996")
    assert mass**2 * (1 - Q(".9608") ** 2) / 2 > Q(".195") ** 2
    assert 1 - alignment / mass > Q(".993")
    assert alignment * half_step / (mass * Q(".0392")) < Q(".003063")
    assert Q(".003063") / Q(".993") < Q(".0031")
    assert Q(".003063") * Q(".9608") * 4 < Q(".012")
    assert Q(".021") + Q(".0031") * Q(".993") < Q(".025")
    assert Q(".9608") * (Q(".56") + Q(".02") * Q("3.5")) < Q(".606")
    assert Q(149, 33) ** 2 > 20 and Q(13, 2) * Q(".7") < 5
    ceilings = [
        Q(".0469"), Q(".0644"), Q(".1560"), Q(".2845"),
        Q(".4333"), Q(".5819"), Q(".7132"), Q(".8175"),
        Q(".8926"), Q(".9420"), Q(".9718"), Q(".9883"),
    ]
    previous, deficit = Q(0), Q(0)
    for index, ceiling in enumerate(ceilings, start=1):
        radius = Q(index, 20)
        normalized = (radius + Q(".025")) / (Q(".993") * Q(".195"))
        radial_upper = Q(4, 5) * sum(
            Q((-1) ** order) * normalized ** (2 * order + 3)
            / (2**order * math.factorial(order) * (2 * order + 3))
            for order in range(31)
        )
        tail_gap = 1 - (radius + Q(".025")) / Q(".993")
        assert tail_gap > 0
        tail_upper = Q(".0392") / tail_gap**2
        assert max(radial_upper, tail_upper) + Q(1, 1024) <= ceiling
        value = 1 - (2 / (2 + radius)) ** 2
        deficit += (value - previous) * (1 - ceiling)
        previous = value
    assert Q(".2057") < deficit < Q(".2058")
    assert deficit > Q(41, 200) and 1 - Q(41, 200) == Q(159, 200)


def check_reference_cap_force_consumers() -> None:
    """Check NCA76's complete population source-plan cap-force constants."""
    alignment, half_step = Q(".006"), Q(".02")
    damping, drift = Q(".9608"), Q(".039216")
    gradient, cap_rms = Q(".607"), Q(".892")
    velocity_rms, ou_rms = Q(".55"), Q(".70")
    source_product, source_rms = Q("4.478"), Q("3.47")
    assert cap_rms**2 > Q(159, 200)
    assert source_product**2 > Q("20.05") and source_rms**2 > Q("12.03")
    assert Q("1.733") ** 2 > 3 and Q(".198") ** 2 > Q(".0392")
    denominator = (Q(".96") - half_step * drift) * (1 - alignment)
    stage_floor = Q(".9936")
    local_base = (Q(1, 2) + alignment * ou_rms + half_step**2 * 4) / stage_floor
    local_slope = half_step / stage_floor
    capped_velocity_product = (
        Q(1, 2) + drift * cap_rms * source_product + Q(".198") * Q("1.733")
        + alignment * cap_rms * (damping * velocity_rms + ou_rms)
        + alignment * (local_base + local_slope * source_product)
    ) / denominator
    first_cap = gradient * (capped_velocity_product + 3 * cap_rms * velocity_rms)
    first_force = Q("2.76185")
    complete_first = alignment * (
        (damping + alignment * damping) * first_cap
        + alignment * damping * cap_rms * first_force
    )
    assert capped_velocity_product < Q("1.06") and first_cap < Q("1.54")
    assert complete_first < Q(".00902")
    assert drift * alignment * first_force < Q(".000650")
    position_force = drift * alignment * gradient * (4 + velocity_rms)
    position_base = Q(".999216") + position_force
    derivative_x, derivative_p = Q(".999216") + 2 * position_force, drift
    mixed_x = position_base * source_product + position_force * source_rms
    mixed_p = drift * (1 + alignment) * source_rms
    consumer = 2 * cap_rms * ou_rms + local_base + Q(".58")
    second_x = alignment * gradient * (consumer * derivative_x + local_slope * mixed_x)
    second_p = alignment * gradient * (consumer * derivative_p + local_slope * mixed_p)
    assert second_x < Q(".00885") and second_p < Q(".000344")
    assert complete_first + second_x < Q(".01787")


def check_reference_mean_cap_and_survival() -> None:
    """Check MCM79's sectors and WSD81's mixed loss absorption."""
    step, alignment = Q(".02"), Q(".006")
    c_lower = sum((-Q(".04"))**order / math.factorial(order) for order in range(6))
    c_upper = sum((-Q(".04"))**order / math.factorial(order) for order in range(7))
    ax_lower = 1 - step**2 * (1 + c_upper)
    eta_upper = c_upper / (c_upper + ax_lower)
    q_squared = (1 - c_lower**2) / 2
    assert eta_upper < Q(".491") and Q(".196066")**2 > q_squared
    atan_lower = sum(
        Q((-1)**order, 2 * order + 1) * Q(1, 5)**(2 * order + 1)
        for order in range(4)
    )
    pi_lower = 16 * atan_lower - Q(4, 239)
    assert 8 / pi_lower < Q("1.596")**2
    additive = (
        alignment * Q(".491") * 4
        + Q("1.0056") * Q(".196066") * Q("1.596") + alignment * Q(".70")
    )
    assert additive < Q(".331")
    assert c_upper**2 * Q(".63")**2 + 3 * q_squared < Q(".70")**2
    for source, floor in ((Q("4.478"), Q(".0989")), (Q("3.47"), Q(".1001"))):
        upper_mean = Q("1.003") * (4 * Q(".960016") + Q(".039216") * source)
        upper_mean += Q(".331")
        assert (2 / (2 + upper_mean))**2 > floor

    force, drift, damping = Q("4.856"), Q(".039216"), Q(".9608")
    position = Q(".999216") + drift * alignment * force
    velocity = drift + damping * alignment * force
    upper = Q(109, 100)
    aa = upper - position**2 - velocity**2
    dd = upper - drift**2 - damping**2
    off = position * drift + velocity * damping
    assert aa > 0 and dd > 0 and aa * dd > off**2
    first_floor, force_step = Q(".994"), alignment * force
    aa = Q(11, 10) - 1 - force_step**2 / first_floor**2
    dd = Q(11, 10) - 1 / first_floor**2
    off = force_step / first_floor**2
    assert aa > 0 and dd > 0 and aa * dd > off**2
    assert (damping**2 + 2 * drift**2 + 1) / Q(".96")**2 < Q(21, 10)
    assert Q(11, 10) * Q(21, 10) < Q(5, 2)
    tilt_denominator = Q(499, 500)
    assert Q(1203, 998) < Q("1.206")
    assert (
        2 * Q(".002")**2 * 12 / tilt_denominator**2 + Q(".03") / tilt_denominator
        <= Q(".06")
    )
    assert Q("1.225") - Q("1.206") == Q(19, 1000)
    lost = Q(109, 100) * (Q(159, 20000) + Q(41, 10000))
    assert lost < Q(41, 400) * Q(2, 5) * Q(99, 100)


def check_reference_signed_first_reserves() -> None:
    """Check SFC80's multiplier positivity, first-count cost and residual limitation."""
    multiplier = [
        [Q(".000274"), Q("-.001361"), Q("-.0001797")],
        [Q("-.001361"), Q(".922237"), Q(".0012396")],
        [Q("-.0001797"), Q(".0012396"), Q(".0001181")],
    ]
    assert multiplier[0][0] > 0
    assert multiplier[0][0] * multiplier[1][1] - multiplier[0][1]**2 > 0
    determinant = sum(
        multiplier[0][column] * (
            multiplier[1][(column + 1) % 3] * multiplier[2][(column + 2) % 3]
            - multiplier[1][(column + 2) % 3] * multiplier[2][(column + 1) % 3]
        ) for column in range(3)
    )
    assert determinant == Q(".00000000002862818517") and determinant > 0
    beta, alignment = Q(".04"), Q(".006")
    first_cost = beta**2 * alignment / ((2 - alignment) * (1 - beta))
    for spectral, gap in (
        (Q(".00002"), Q(".0000147")), (Q(".00001"), Q(".0000048"))
    ):
        assert spectral * Q(".9876") - first_cost > gap
    assert alignment**2 / (1 - beta) < Q(".006124")**2
    assert (1 - Q(".006124"))**2 > Q(".9876")
    assert Q(".000135") - Q(".0392") * Q(".01394") < 0


def main() -> None:
    check_exact_bounds()
    print("PASS: exact rational kinetic, moment, feedback, contraction and W2 bounds")
    check_survival_normalization_algebra()
    print("PASS: finite exact sharp-tilt and whole-block oscillation regressions")
    check_reference_cap_certificates()
    print("PASS: exact whole-update cap matrices and reference noisy-cap bounds")
    check_reference_velocity_burn()
    print("PASS: exact six-update survivor burn and full-Gaussian provider bounds")
    check_reference_two_count_balance()
    print("PASS: exact noncommuting count/cap loss and source-weighted B2 bounds")
    check_reference_empirical_provider()
    print("PASS: actual empirical jitter/OU provider concentration certificates")
    check_reference_restricted_transport()
    print("PASS: centered-force absorption and exact finite tied alive-law bounds")
    check_reference_position_shapes()
    print("PASS: arbitrary position-shape kinetic margins and exact alive-readout weights")
    check_reference_boundary_response()
    print("PASS: preserved source-boundary and exact delayed survival-moment bounds")
    check_reference_signed_shape_classes()
    print("PASS: both full spatial forces, signed Gaussian family and alive-tail margins")
    check_reference_bilinear_coefficients()
    print("PASS: exact general bilinear provider coefficients and square completion")
    check_reference_conditional_cap()
    print("PASS: actual OU graph primitive, conditional cap and local first-force bounds")
    check_reference_full_gaussian_cap()
    print("PASS: twelve full-Gaussian cap thresholds and exact conditional deficit")
    check_reference_cap_force_consumers()
    print("PASS: complete full-tail population first/second capped spatial consumers")
    check_reference_mean_cap_and_survival()
    print("PASS: actual mean-cap sectors and source-weighted own-survival deficits")
    check_reference_signed_first_reserves()
    print("PASS: signed first-provider reserves and the absolute-interface limitation")
    print("Floating diagnostics only (not interval certificates):")
    for name, value in floating_diagnostics().items():
        print(f"  {name} = {value:.12g}")


if __name__ == "__main__":
    main()
