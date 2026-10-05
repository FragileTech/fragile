#!/usr/bin/env python3
"""Check repaired cloning-signal formulas using exact rational arithmetic.

Run with ``uv run python docs/validate_cloning_signal.py``. Finite regression
examples detect algebra, orientation, and probability-ordering mistakes; they
do not prove universal theorems or verify a landscape's analytic hypotheses.
The complete analytic proofs remain in the Volume II convergence chapters.
"""

from fractions import Fraction
from itertools import product
from math import comb


Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def mean(values: tuple[Fraction, ...]) -> Fraction:
    """Return the equally weighted empirical mean."""
    return sum(values, Fraction(0)) / len(values)


def variance(values: tuple[Fraction, ...]) -> Fraction:
    """Use population normalization, as in the fitness standardizer."""
    center = mean(values)
    return mean(tuple((value - center) ** 2 for value in values))


def positive_part(value: Fraction) -> Fraction:
    """Return the positive part without introducing floating point."""
    return max(Fraction(0), value)


def vector_mean(points: tuple[Vector, ...]) -> Vector:
    """Return the empirical center in the physical comparison coordinates."""
    return tuple(mean(tuple(point[axis] for point in points)) for axis in range(len(points[0])))


def squared_distance(left: Vector, right: Vector) -> Fraction:
    """Compare squared lengths so no square-root approximation is needed."""
    return sum(((x - y) ** 2 for x, y in zip(left, right, strict=True)), Fraction(0))


def vector_variance(points: tuple[Vector, ...]) -> Fraction:
    """Return the trace of the empirical covariance."""
    center = vector_mean(points)
    return mean(tuple(squared_distance(point, center) for point in points))


def matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply exact matrices without relying on numerical linear algebra."""
    return tuple(
        sum((left * right for left, right in zip(row, vector, strict=True)), Fraction(0))
        for row in matrix
    )


def transpose(matrix: Matrix) -> Matrix:
    """Transpose a nonempty exact matrix."""
    return tuple(tuple(row) for row in zip(*matrix, strict=True))


def matrix_product(left: Matrix, right: Matrix) -> Matrix:
    """Return a matrix product using exact row-column dot products."""
    columns = transpose(right)
    return tuple(matrix_vector(columns, row) for row in left)


def quadratic(matrix: Matrix, vector: Vector) -> Fraction:
    """Evaluate the exact quadratic form of a comparison matrix."""
    image = matrix_vector(matrix, vector)
    return sum((x * y for x, y in zip(vector, image, strict=True)), Fraction(0))


def check_within_group_separation() -> None:
    """Exercise every nontrivial labelled partition of small rational vectors."""
    grid = tuple(Fraction(value, 2) for value in range(4))
    for count in range(2, 5):
        for values in product(grid, repeat=count):
            for mask in range(1, (1 << count) - 1):
                high = tuple(value for index, value in enumerate(values) if mask & (1 << index))
                low = tuple(value for index, value in enumerate(values) if not mask & (1 << index))
                f_high = Fraction(len(high), count)
                f_low = 1 - f_high
                within = f_high * variance(high) + f_low * variance(low)
                between = f_high * f_low * (mean(low) - mean(high)) ** 2
                assert variance(values) == within + between

                width_bound = (
                    f_high * (max(high) - min(high)) ** 2 + f_low * (max(low) - min(low)) ** 2
                ) / 4
                assert within <= width_bound
                assert between >= variance(values) - width_bound
                assert variance(values) <= (max(values) - min(values)) ** 2 / 4

    high = (Fraction(2, 5), Fraction(3, 5))
    low = (Fraction(7, 5), Fraction(8, 5))
    total = variance(high + low)
    within = (variance(high) + variance(low)) / 2
    assert total == Fraction(13, 50)
    assert within == Fraction(1, 100)
    assert total > within
    assert (total - within) / Fraction(1, 4) == (mean(low) - mean(high)) ** 2 == 1

    symmetric = (Fraction(1, 2), Fraction(3, 2))
    assert variance(symmetric + symmetric) == variance(symmetric) == Fraction(1, 4)
    assert mean(symmetric) - mean(symmetric) == 0


def check_averaged_gap_events() -> None:
    """Keep zero-gap outcomes and distinguish signed gaps from magnitudes."""
    # Each outcome has constant values inside two fixed groups of equal size.
    laws = (
        (
            (Fraction(1, 2), Fraction(0)),
            (Fraction(1, 4), Fraction(1)),
            (Fraction(1, 4), Fraction(-1)),
        ),
        ((Fraction(1, 2), Fraction(0)), (Fraction(1, 2), Fraction(1))),
    )
    for law in laws:
        assert sum((probability for probability, _ in law), Fraction(0)) == 1
        expected_total = sum(
            (
                probability * variance((positive_part(-gap),) * 2 + (positive_part(gap),) * 2)
                for probability, gap in law
            ),
            Fraction(0),
        )
        expected_squared_gap = sum((probability * gap**2 for probability, gap in law), Fraction(0))
        Q = expected_total / Fraction(1, 4)
        assert Q == expected_squared_gap == Fraction(1, 2)
        for threshold in (Fraction(1, 4), Fraction(1, 2), Fraction(2, 3)):
            assert threshold**2 < Q <= 1
            probability = sum((weight for weight, gap in law if abs(gap) > threshold), Fraction(0))
            lower = (Q - threshold**2) / (1 - threshold**2)
            assert probability >= lower > 0

    symmetric_law, oriented_law = laws
    assert sum((weight * gap for weight, gap in symmetric_law), Fraction(0)) == 0
    assert all(gap >= 0 for _, gap in oriented_law)
    threshold = Fraction(1, 2)
    assert sum((weight for weight, gap in symmetric_law if gap > threshold), Fraction(0)) < sum(
        (weight for weight, gap in symmetric_law if abs(gap) > threshold), Fraction(0)
    )

    # Equality at the threshold belongs to the complement of the strict event.
    event_probability = Fraction(2, 5)
    Q = (1 - event_probability) * threshold**2 + event_probability
    assert Q - threshold**2 == event_probability * (1 - threshold**2)


def check_logistic_saturation() -> None:
    """Evaluate the logistic map exactly at +/- gain * log(3)."""
    previous_variance = Fraction(0)
    previous_derivative = Fraction(1, 2)
    for gain in range(1, 21):
        power = 3**gain
        low = Fraction(2, 1 + power)
        high = Fraction(2 * power, 1 + power)
        slope_at_endpoint = Fraction(2 * power, (1 + power) ** 2)
        realized_variance = variance((low, high))
        assert mean((low, high)) == 1
        assert realized_variance == Fraction((power - 1) ** 2, (power + 1) ** 2)
        assert previous_variance < realized_variance < 1
        assert 0 < slope_at_endpoint < previous_derivative
        previous_variance = realized_variance
        previous_derivative = slope_at_endpoint

    # Neither a fixed local amplification nor a fixed derivative extends to all gains.
    assert 3**2 * variance((Fraction(1, 2), Fraction(3, 2))) > 1
    assert 20 * previous_derivative < Fraction(3, 8)


def check_vector_cluster_bounds() -> None:
    """Use an equilateral simplex with rational coordinates as a counterexample."""
    simplex = tuple(tuple(Fraction(int(axis == index)) for axis in range(3)) for index in range(3))
    center = vector_mean(simplex)
    diameter_squared = max(squared_distance(left, right) for left in simplex for right in simplex)
    radius_squared = max(squared_distance(point, center) for point in simplex)
    assert diameter_squared == 2
    assert radius_squared == vector_variance(simplex) == Fraction(2, 3)
    assert radius_squared > diameter_squared / 4
    assert vector_variance(simplex) > diameter_squared / 4
    assert radius_squared <= diameter_squared
    assert vector_variance(simplex) <= diameter_squared / 2

    grid = tuple(
        tuple(Fraction(coordinate) for coordinate in point)
        for point in product(range(2), repeat=2)
    )
    for points in product(grid, repeat=3):
        center = vector_mean(points)
        diameter_squared = max(
            squared_distance(left, right) for left in points for right in points
        )
        assert max(squared_distance(point, center) for point in points) <= diameter_squared
        pairwise_variance = sum(
            (squared_distance(left, right) for left in points for right in points), Fraction(0)
        ) / (2 * len(points) ** 2)
        assert vector_variance(points) == pairwise_variance <= diameter_squared / 2


def check_invalid_cluster_accounting() -> None:
    """Apply the actual valid/invalid count rule before selecting contributions."""
    zero = (Fraction(0), Fraction(0))
    far = (Fraction(100), Fraction(100))
    examples = (
        ((zero,) * 8, ((Fraction(1), Fraction(2)),) * 6, (far,) * 2),
        ((zero,) * 6, (far,) * 6),
        ((zero,), ((Fraction(1), Fraction(0)),), (far,)),
    )
    for clusters in examples:
        points = tuple(point for cluster in clusters for point in cluster)
        count = len(points)
        center = vector_mean(points)
        minimum_count = max(5, (count + 19) // 20)
        contributions = tuple(
            Fraction(len(cluster), count) * squared_distance(vector_mean(cluster), center)
            for cluster in clusters
        )
        within = sum(
            (Fraction(len(cluster), count) * vector_variance(cluster) for cluster in clusters),
            Fraction(0),
        )
        between = sum(contributions, Fraction(0))
        assert vector_variance(points) == within + between
        invalid = tuple(
            index for index, cluster in enumerate(clusters) if len(cluster) < minimum_count
        )
        valid = tuple(index for index in range(len(clusters)) if index not in invalid)
        invalid_energy = sum((contributions[index] for index in invalid), Fraction(0))
        valid_energy = sum((contributions[index] for index in valid), Fraction(0))
        assert between == invalid_energy + valid_energy

        for omitted_fraction in (Fraction(1, 10), Fraction(1, 3), Fraction(3, 4)):
            selected: list[int] = []
            selected_energy = Fraction(0)
            for index in sorted(valid, key=contributions.__getitem__, reverse=True):
                if selected_energy >= (1 - omitted_fraction) * valid_energy:
                    break
                selected.append(index)
                selected_energy += contributions[index]
            high_indices = invalid + tuple(selected)
            high_energy = invalid_energy + selected_energy
            assert high_energy >= invalid_energy + (1 - omitted_fraction) * valid_energy
            assert high_energy >= (1 - omitted_fraction) * between
            high_count = sum(len(clusters[index]) for index in high_indices)
            diameter_squared = max(
                squared_distance(left, right) for left in points for right in points
            )
            assert high_energy <= diameter_squared * Fraction(high_count, count)

    # Dropping the two far invalid rows loses most of the between-cluster energy.
    points = tuple(point for cluster in examples[0] for point in cluster)
    center = vector_mean(points)
    energies = tuple(
        Fraction(len(cluster), len(points)) * squared_distance(vector_mean(cluster), center)
        for cluster in examples[0]
    )
    assert sum(energies[:2], Fraction(0)) < Fraction(9, 10) * sum(energies, Fraction(0))


def acceptance(recipient: Fraction, donor: Fraction, denominator: Fraction) -> Fraction:
    """Evaluate acceptance on realized fitness before taking any expectation."""
    return min(Fraction(1), positive_part(donor - recipient) / denominator)


def check_realized_selection() -> None:
    """Check clipping, positive-part comparison, and uniform donor summation."""
    recipient = Fraction(1)
    denominator = Fraction(2)
    actual = (
        acceptance(recipient, Fraction(0), denominator)
        + acceptance(recipient, Fraction(2), denominator)
    ) / 2
    assert actual == Fraction(1, 4)
    assert acceptance(recipient, mean((Fraction(0), Fraction(2))), denominator) == 0

    actual = (
        acceptance(recipient, Fraction(1), denominator)
        + acceptance(recipient, Fraction(5), denominator)
    ) / 2
    naive = acceptance(recipient, mean((Fraction(1), Fraction(5))), denominator)
    assert actual == Fraction(1, 2) < naive == 1

    grid = (Fraction(1, 4), Fraction(1), Fraction(2), Fraction(5))
    epsilon = Fraction(1, 3)
    for values in product(grid, repeat=3):
        count = len(values)
        spread = max(values) - min(values)
        A = max(spread, max(values) + epsilon)
        for index, value in enumerate(values):
            donor_indices = tuple(other for other in range(count) if other != index)
            weights = tuple(Fraction(1 + (index + 2 * other) % 5) for other in donor_indices)
            probabilities = tuple(weight / sum(weights) for weight in weights)
            a = (count - 1) * min(probabilities)
            signal = sum(
                (
                    probability * positive_part(values[other] - value)
                    for probability, other in zip(probabilities, donor_indices, strict=True)
                ),
                Fraction(0),
            )
            actual = sum(
                (
                    probability * acceptance(value, values[other], value + epsilon)
                    for probability, other in zip(probabilities, donor_indices, strict=True)
                ),
                Fraction(0),
            )
            assert actual >= signal / A
            if spread > 0 and value <= mean(values):
                assert signal >= a * variance(values) / (2 * spread)
                assert actual >= a * variance(values) / (2 * spread * A)
            if spread == 0:
                assert signal == actual == 0


def check_finite_population_coverage() -> None:
    """Check the positive-part square and exact cell self-exclusion correction."""
    grid = tuple(Fraction(value, 4) for value in range(9))
    for a, b in product(grid, repeat=2):
        assert positive_part(a - b) ** 2 >= a**2 / 2 - b**2

    C = Fraction(3, 7)
    E = Fraction(1)
    cell_count = 3
    threshold = Fraction(1, 4)
    chi = C * threshold**2 / (2 * E**2 * cell_count**2)
    for count in range(2, 11):
        for first in range(count + 1):
            for second in range(count - first + 1):
                counts = (first, second, count - first - second)
                for fractions in product((Fraction(0), Fraction(1, 2), Fraction(1)), repeat=3):
                    errors = tuple(
                        E * n / count * fraction
                        for n, fraction in zip(counts, fractions, strict=True)
                    )
                    W = sum(errors, Fraction(0))
                    actual = (
                        C
                        * sum(
                            (
                                error * (n - 1) ** 2
                                for error, n in zip(errors, counts, strict=True)
                            ),
                            Fraction(0),
                        )
                        / (count - 1) ** 2
                    )
                    excluded_mass = positive_part(count * W / (E * cell_count) - 1)
                    coverage = C * W * (excluded_mass / (count - 1)) ** 2
                    assert actual >= coverage
                    approximate = positive_part(W / (E * cell_count) - Fraction(1, count))
                    assert excluded_mass / (count - 1) >= approximate
                    assert actual >= chi * (W - threshold) - C * E / count**2
                    if W >= threshold:
                        assert actual >= chi * W - C * E / count**2

    # A singleton has zero centered positional error and enters the low-error branch.
    assert Fraction(0) >= chi * (0 - threshold) - C * E


def check_coarse_cap_kinetic_obstruction() -> None:
    """Exercise the cap-charge witness and one feasible kinetic matrix."""
    alpha = gamma = Fraction(1, 4)
    for h, k, a, beta in product(
        (Fraction(1, 25), Fraction(1, 2), Fraction(1)),
        (Fraction(0), Fraction(1, 4), Fraction(1), Fraction(2)),
        (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)),
        (Fraction(-1, 16), Fraction(0), Fraction(1, 16)),
    ):
        c = h / 2
        B = c * (1 + a)
        eta = c**2 * (1 + a)
        T = (
            (1 - eta * k, B),
            (-c * k * (1 + a - eta * k), a - c * k * B),
        )
        G = ((alpha, beta), (beta, gamma))
        G_hat = ((alpha + abs(beta), beta), (beta, gamma + abs(beta)))
        charged = matrix_product(transpose(T), matrix_product(G_hat, T))
        D = tuple(
            tuple(G[row][column] - charged[row][column] for column in range(2)) for row in range(2)
        )
        witness = (Fraction(1), c * k)
        assert matrix_vector(T, witness) == (Fraction(1), -c * k)
        assert quadratic(D, witness) == 4 * beta * c * k - abs(beta) * (1 + c**2 * k**2)
        if beta <= 0:
            assert quadratic(D, witness) <= 0
        if h == Fraction(1, 25) and k == 1 and beta > 0:
            assert quadratic(D, witness) == -Fraction(2301, 40000) < 0

    # This rational example is a feasible coarse matrix, not a global swarm theorem.
    G = ((Fraction(1, 4), Fraction(1, 16)), (Fraction(1, 16), Fraction(1, 4)))
    T = ((Fraction(5, 8), Fraction(3, 4)), (Fraction(-9, 16), Fraction(1, 8)))
    G_hat = ((Fraction(5, 16), Fraction(1, 16)), (Fraction(1, 16), Fraction(5, 16)))
    charged = matrix_product(transpose(T), matrix_product(G_hat, T))
    D = tuple(
        tuple(G[row][column] - charged[row][column] for column in range(2)) for row in range(2)
    )
    assert D == (
        (Fraction(299, 4096), Fraction(-83, 2048)),
        (Fraction(-83, 2048), Fraction(59, 1024)),
    )
    determinant = D[0][0] * D[1][1] - D[0][1] ** 2
    trace = D[0][0] + D[1][1]
    assert D[0][0] > 0 and determinant == Fraction(21, 8192) > 0
    assert trace == Fraction(535, 4096)
    assert determinant == Fraction(21, 1070) * trace


def total_variation(left: Vector, right: Vector) -> Fraction:
    """Use the probability convention TV = one half of the l1 distance."""
    return sum((abs(x - y) for x, y in zip(left, right, strict=True)), Fraction(0)) / 2


def reweight(probabilities: Vector, weights: Vector) -> Vector:
    """Apply a positive density tilt with its realized normalizer."""
    weighted = tuple(x * y for x, y in zip(probabilities, weights, strict=True))
    total = sum(weighted, Fraction(0))
    return tuple(value / total for value in weighted)


def check_tv_reweighting() -> None:
    """Check positive reweighting and the two-tilt killed-chain identity."""
    probabilities = tuple(
        (Fraction(first, 4), Fraction(second, 4), Fraction(4 - first - second, 4))
        for first in range(5)
        for second in range(5 - first)
    )
    for weights in product((Fraction(1, 2), Fraction(1), Fraction(2)), repeat=3):
        ratio = max(weights) / min(weights)
        for left, right in product(probabilities, repeat=2):
            initial_tv = total_variation(left, right)
            tilted_tv = total_variation(reweight(left, weights), reweight(right, weights))
            assert tilted_tv <= ratio * initial_tv
            if ratio == 1:
                assert tilted_tv == initial_tv

    # P h = r h; Q is the exact Doob transition associated with P and h.
    h = (Fraction(1), Fraction(2))
    inverse_h = tuple(1 / value for value in h)
    r = Fraction(1, 2)
    Q = ((Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 4), Fraction(3, 4)))
    P = tuple(tuple(r * h[i] * Q[i][j] / h[j] for j in range(2)) for i in range(2))
    assert all(0 < sum(row, Fraction(0)) <= 1 for row in P)
    assert matrix_vector(P, h) == tuple(r * value for value in h)
    pi = (Fraction(1, 2), Fraction(1, 2))
    qsd = reweight(pi, inverse_h)
    assert qsd == (Fraction(2, 3), Fraction(1, 3))
    assert matrix_vector(transpose(P), qsd) == tuple(r * value for value in qsd)

    two_state = tuple((Fraction(first, 4), Fraction(4 - first, 4)) for first in range(5))
    P_power = Q_power = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    ratio = max(h) / min(h)
    for steps in range(7):
        conditioned: dict[Vector, Vector] = {}
        for initial in two_state:
            killed = matrix_vector(transpose(P_power), initial)
            survival = sum(killed, Fraction(0))
            conditioned[initial] = tuple(value / survival for value in killed)
            doob_law = matrix_vector(transpose(Q_power), reweight(initial, h))
            assert conditioned[initial] == reweight(doob_law, inverse_h)
            assert total_variation(conditioned[initial], qsd) <= (
                ratio**2 * Fraction(1, 2) ** steps * total_variation(initial, qsd)
            )
        for left, right in product(two_state, repeat=2):
            assert total_variation(conditioned[left], conditioned[right]) <= (
                ratio**2 * Fraction(1, 2) ** steps * total_variation(left, right)
            )
        P_power = matrix_product(P_power, P)
        Q_power = matrix_product(Q_power, Q)


def check_survivor_mark_obstruction() -> None:
    """Check the binomial conditioning algebra used by the marked obstruction."""
    probabilities = (Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    for count, p in product(range(1, 25), probabilities):
        q = 1 - p
        survival = 1 - q**count
        conditional_row_mean = p / survival
        explicit_mean = (
            sum(
                (
                    Fraction(alive, count) * comb(count, alive) * p**alive * q ** (count - alive)
                    for alive in range(1, count + 1)
                ),
                Fraction(0),
            )
            / survival
        )
        assert conditional_row_mean == explicit_mean
        assert conditional_row_mean - p == p * q**count / survival
        numerator = survival - p * count * q ** (count - 1)
        expanded = p**2 * sum(((index + 1) * q**index for index in range(count - 1)), Fraction(0))
        assert numerator == expanded
        derivative = numerator / survival**2
        if count == 1:
            assert conditional_row_mean == 1 and derivative == 0
        else:
            assert 0 < derivative < 1
            next_mean = p / (1 - q ** (count + 1))
            assert p < next_mean < conditional_row_mean < 1

    # At fixed probabilities the conditioned mark gap approaches the raw gap.
    left, right = Fraction(1, 4), Fraction(1, 2)
    raw_gap = right - left
    for count in (2, 4, 8, 16, 32):
        left_tilt = left / (1 - (1 - left) ** count)
        right_tilt = right / (1 - (1 - right) ** count)
        conditioned_gap = right_tilt - left_tilt
        assert conditioned_gap > 0
        error = left * (1 - left) ** count / (1 - (1 - left) ** count) + right * (
            1 - right
        ) ** count / (1 - (1 - right) ** count)
        assert abs(conditioned_gap - raw_gap) <= error
    assert conditioned_gap > Fraction(99, 100) * raw_gap

    # This rational local profile tests order epsilon versus epsilon squared.
    # The Gaussian landing profile and its nonzero derivative are proved in the text.
    count = 4
    base = Fraction(1, 2)
    previous_ratio = Fraction(0)
    for power in range(3, 12):
        epsilon = Fraction(1, 2**power)
        shifted = base - epsilon
        mark_gap = base / (1 - (1 - base) ** count) - shifted / (1 - (1 - shifted) ** count)
        ratio = mark_gap / epsilon**2
        assert ratio > previous_ratio
        previous_ratio = ratio


def check_rastrigin_truncation_bounds() -> None:
    """Check rational constants in the separate analytic force and tail proof."""
    pi_lower = Fraction(3)
    pi_upper = Fraction(22, 7)
    half_separation = Fraction(1, 500)
    force_derivative_floor = -2 + 40 * pi_lower**2 * (1 - 2 * pi_upper**2 * half_separation**2)
    assert force_derivative_floor > 350
    force_abs_bound = 2 * (Fraction(1, 2) + half_separation) + (40 * pi_upper**2 * half_separation)
    eta_upper = Fraction(1, 1250)
    assert Fraction(1, 2) - half_separation - eta_upper * force_abs_bound > Fraction(49, 100)
    assert Fraction(1, 2) + half_separation + eta_upper * force_abs_bound < Fraction(51, 100)
    assert -1 + 40 * pi_upper**2 * half_separation < 0

    tau_squared_upper = Fraction(52, 125000)
    tau_upper = Fraction(51, 2500)
    r = Fraction(68)
    assert tau_squared_upper < tau_upper**2
    assert (2 - Fraction(51, 100)) / tau_upper > r

    # These tail bounds follow analytically from Gaussian integration by parts.
    # This regression checks their rational arithmetic and resulting variance floor.
    exponent = r**2 / 2
    inverse_exponential_upper = 1 / (1 + exponent + exponent**2 / 2)
    mass_upper = 2 * inverse_exponential_upper / r
    first_moment_upper = 2 * inverse_exponential_upper
    second_moment_upper = 2 * (r + 1 / r) * inverse_exponential_upper
    variance_lower = 1 - second_moment_upper - first_moment_upper**2 / (1 - mass_upper) ** 2
    assert 0 < mass_upper < 1
    assert second_moment_upper < Fraction(1, 1000)
    assert variance_lower > Fraction(99, 100)

    raw_gain_lower = 1 + Fraction(350, 2500)
    assert raw_gain_lower == Fraction(57, 50)
    assert Fraction(99, 100) * raw_gain_lower > Fraction(28, 25)


def check_finite_forest_reset_accounting() -> None:
    """Check finite exception exposure and the positive two-step margin algebra."""
    count = 4
    choices = tuple((-1, *range(index + 1, count)) for index in range(count))
    forests = tuple(product(*choices))

    def closure(graphs: tuple[tuple[int, ...], ...], seeds: set[int]) -> set[int]:
        reached = set(seeds)
        while True:
            previous = set(reached)
            for graph in graphs:
                for source, target in enumerate(graph):
                    if target >= 0 and (source in reached or target in reached):
                        reached.update((source, target))
            if reached == previous:
                return reached

    for left, right in product(forests, repeat=2):
        exceptional = {index for index in range(count) if left[index] != right[index]}
        endpoints = set(exceptional)
        for index in exceptional:
            endpoints.update(target for target in (left[index], right[index]) if target >= 0)
        assert len(endpoints) <= 3 * len(exceptional)
        common = tuple(-1 if index in exceptional else left[index] for index in range(count))
        assert common == tuple(
            -1 if index in exceptional else right[index] for index in range(count)
        )
        incident = {
            vertex
            for graph in (left, right)
            for source, target in enumerate(graph)
            if target >= 0
            for vertex in (source, target)
        }
        for mask in range(1 << count):
            first_preparation_bad = {index for index in range(count) if mask & (1 << index)}
            seeds = first_preparation_bad | endpoints
            exposed = closure((common,), seeds)
            # Restoring exceptions only connects components already touching endpoint seeds.
            assert closure((left, right), seeds) <= exposed
            charged = (exposed - first_preparation_bad) | (first_preparation_bad & incident)
            assert charged & first_preparation_bad == first_preparation_bad & incident
            assert first_preparation_bad - incident <= first_preparation_bad - charged

    # An explicit rowwise coupling has common edge mass p/2 and each one-sided
    # edge mass p/2. Conditional on matching, remaining row draws still factorize.
    c = Fraction(1, 8)
    edge_probability = c / (count - 1)
    path_bound = sum((Fraction(1, 1) / factorial for factorial in (1, 1, 2, 6)), Fraction(0))
    for exception_mask in range(1 << (count - 1)):
        row_laws = []
        for index in range(count):
            if exception_mask & (1 << index):
                row_laws.append(((-1, Fraction(1)),))
                continue
            accepted = (count - index - 1) * edge_probability
            matching = 1 - accepted
            row_laws.append((
                (-1, (1 - 3 * accepted / 2) / matching),
                *(
                    (target, edge_probability / (2 * matching))
                    for target in range(index + 1, count)
                ),
            ))
            assert sum((probability for _, probability in row_laws[-1]), Fraction(0)) == 1
            assert all(
                probability <= 2 * c / (count - 1)
                for target, probability in row_laws[-1]
                if target >= 0
            )
        component_means = [Fraction(0)] * count
        total = Fraction(0)
        for realization in product(*row_laws):
            graph = tuple(target for target, _ in realization)
            probability = Fraction(1)
            for _, row_probability in realization:
                probability *= row_probability
            total += probability
            for root in range(count):
                component_means[root] += probability * len(closure((graph,), {root}))
        assert total == 1
        assert all(value <= path_bound for value in component_means)

    # The analytic proof supplies e1 <= theta E0 and e2 <= theta E2; only
    # their exact final margin arithmetic is tested here, not the theorem itself.
    for epsilon, c0, edge_lipschitz in product(
        (Fraction(1, 8), Fraction(1, 2), Fraction(1)),
        (Fraction(1, 4), Fraction(1), Fraction(4)),
        (Fraction(1, 3), Fraction(2), Fraction(10)),
    ):
        E0 = 24 * c0 + 9 * edge_lipschitz
        E2 = E0 + 6 * c0
        theta = min(Fraction(1), 1 / (8 * c0), epsilon / (8 * E2))
        e1, e2 = theta * E0, theta * E2
        assert 8 * theta * c0 <= 1
        assert e1 <= epsilon / 8 and e2 <= epsilon / 8
        contraction_upper = (1 - epsilon + e2) * (1 + e1)
        assert contraction_upper <= 1 - 3 * epsilon / 4 - 7 * epsilon**2 / 64
        assert contraction_upper <= 1 - epsilon / 2 < 1


def main() -> None:
    """Run the exact regression checks without claiming theorem certification."""
    checks = (
        check_within_group_separation,
        check_averaged_gap_events,
        check_logistic_saturation,
        check_vector_cluster_bounds,
        check_invalid_cluster_accounting,
        check_realized_selection,
        check_finite_population_coverage,
        check_coarse_cap_kinetic_obstruction,
        check_tv_reweighting,
        check_survivor_mark_obstruction,
        check_rastrigin_truncation_bounds,
        check_finite_forest_reset_accounting,
    )
    for check in checks:
        check()
    print(f"Cloning-signal exact regression checks passed ({len(checks)} groups).")


if __name__ == "__main__":
    main()
