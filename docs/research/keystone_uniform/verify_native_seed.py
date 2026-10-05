"""Outward interval certificate for the native central-source estimates.

Run from the repository root with::

    UV_CACHE_DIR=/tmp/fragile-uv-cache uv run python \
        docs/research/keystone_uniform/verify_native_seed.py

The program evaluates the primitive formulas in
``20_native_seed_selection.md``. It integrates the unchanged Gaussian laws
by finite lower sums; it does not simulate or truncate those laws. Importing
this module performs no calculation.
"""

from __future__ import annotations

import math

import mpmath as mp


def certificate() -> None:
    """Check all strict margins used in the accompanying proof."""
    mp.mp.dps = 110
    mp.iv.dps = 85
    iv = mp.iv

    def interval(value):
        return iv.mpf(value)

    def lo(value):
        return mp.mpf(value._mpi_[0])

    def hi(value):
        return mp.mpf(value._mpi_[1])

    def show(name, value):
        magnitude = max(abs(lo(value)), abs(hi(value)))
        if magnitude == 0:
            print(f"{name}: [0, 0]")
            return
        scale = mp.mpf(10) ** (25 - int(mp.floor(mp.log10(magnitude))))
        displayed_lower = mp.floor(lo(value) * scale) / scale
        displayed_upper = mp.ceil(hi(value) * scale) / scale
        print(
            f"{name}: "
            f"[{mp.nstr(displayed_lower, 26)}, {mp.nstr(displayed_upper, 26)}]"
        )

    def assert_lower(name, value, threshold):
        assert lo(value) > hi(interval(threshold)), name
        show(name, value)

    def assert_upper(name, value, threshold):
        assert hi(value) < lo(interval(threshold)), name
        show(name, value)

    def G(value):
        return 2 / (1 + iv.exp(-value)) + interval("0.1")

    # There are 401 terms, with highest degree 801. The first omitted
    # integrated exponential term has degree 803. For |x| <= 8, the
    # alternating terms decrease after index 32 and the remainder is
    # bounded by that first omitted term. Its worst-case value is < 1e-270.
    first_omitted = (
        interval(8) ** 803
        / (interval(2) ** 401 * interval(math.factorial(401)) * 803)
        / iv.sqrt(2 * iv.pi)
    )
    assert_upper("cdf_first_omitted_term_at_8", first_omitted, "1e-270")

    # Mills' bound at eight validates both tail endpoint intervals.
    mills_eight = iv.exp(-interval(32)) / (8 * iv.sqrt(2 * iv.pi))
    assert_upper("gaussian_tail_mills_bound_at_8", mills_eight, "6.5e-16")

    def cdf_endpoint(value):
        x = interval(value)
        if value > 8:
            lower = lo(interval(1) - interval("6.5e-16"))
            return interval([lower, 1])
        if value < -8:
            return interval([0, hi(interval("6.5e-16"))])
        term = x
        total = term
        for n in range(400):
            term *= -x * x * (2 * n + 1) / (2 * (n + 1) * (2 * n + 3))
            total += term
        remainder = interval(["-1e-200", "1e-200"])
        return interval("0.5") + total / iv.sqrt(2 * iv.pi) + remainder

    def Phi(value):
        # CDF monotonicity reduces an interval input to its exact endpoints.
        left = cdf_endpoint(lo(value))
        right = cdf_endpoint(hi(value))
        return interval([lo(left), hi(right)])

    eps = interval("0.001")
    u = interval(1) / 8
    reward_B_width = (1 + 20 * iv.pi**2) * eps**2
    reward_A_min = (1 - eps) ** 2
    reward_A_max = 1 + 2 * eps + reward_B_width
    reward_A_width = reward_A_max - reward_A_min
    reward_sd_upper = iv.sqrt(
        interval("0.01")
        + u * (1 - u) * reward_A_max**2
        + (reward_A_width**2 + reward_B_width**2) / 4
    )
    reward_B_lower = G(((1 - u) * reward_A_min - reward_B_width) / reward_sd_upper)
    reward_A_upper = G(reward_A_width / interval("0.1"))
    near_distance_upper = 2 * iv.sqrt(2) * eps
    far_distance_lower = interval(2) / 3 - near_distance_upper
    far_distance_upper = interval(2) / 3 + near_distance_upper
    far_mark_lower = iv.sqrt(far_distance_lower**2 + interval("0.000001"))
    far_mark_upper = iv.sqrt(far_distance_upper**2 + interval("0.000001"))
    far_factor_span = 5 * (far_mark_upper - far_mark_lower)
    far_factor_lower = G(-(far_mark_upper - far_mark_lower) / interval("0.1"))
    cross_weight_lower = iv.exp(-(far_distance_upper**2) / 8)
    cross_weight_upper = iv.exp(-(far_distance_lower**2) / 8)
    near_weight_lower = iv.exp(-(near_distance_upper**2) / 8)
    central_near_upper = u / (u + (1 - u) * cross_weight_lower)
    resident_far_upper = (
        u * cross_weight_upper / (interval("0.75") * near_weight_lower + u * cross_weight_upper)
    )
    gate_lower = (reward_B_lower - reward_A_upper) / (reward_A_upper + interval("0.00001"))
    incoming = (
        (1 - u)
        * cross_weight_lower
        * interval("0.75")
        * near_weight_lower
        * (1 - central_near_upper)
        * gate_lower
    )
    reverse = central_near_upper * resident_far_upper
    central_reward_span = 5 * reward_B_width
    far_far_gate_upper = (
        interval("2.1")
        * (central_reward_span + far_factor_span)
        / (reward_B_lower * far_factor_lower + interval("0.000001"))
    )
    within_central = central_near_upper * (
        central_near_upper + (1 - central_near_upper) * far_far_gate_upper
    )
    diversity_sd_upper = iv.sqrt((far_mark_upper - interval("0.001")) ** 2 / 4 + interval("0.01"))
    standardized_range = (far_mark_upper - interval("0.001")) / interval("0.1")
    derivative_lower = 2 * iv.exp(-standardized_range) / (1 + iv.exp(-standardized_range)) ** 2
    central_far_near_gap = (
        reward_B_lower
        * derivative_lower
        * (far_mark_lower - interval("0.003"))
        / diversity_sd_upper
        - interval("2.1") * central_reward_span
    )
    cross_far_far_gap = (
        reward_B_lower - reward_A_upper
    ) * far_factor_lower - reward_A_upper * far_factor_span
    cross_near_near_gap = (reward_B_lower - reward_A_upper) * interval(
        "0.1"
    ) - reward_A_upper * interval("0.01")
    complete_growth = (
        interval("0.993") * (1 - reverse)
        - interval("0.793") * within_central
        + interval("0.2") * incoming
    )
    for name, value in [
        ("H_B_lower", reward_B_lower),
        ("H_A_upper", reward_A_upper),
        ("favorable_gate_lower", gate_lower),
        ("beta_incoming", incoming),
        ("beta_reverse", reverse),
        ("beta_within_central", within_central),
    ]:
        show(name, value)
    assert_upper("favorable_gate_is_below_one", gate_lower, 1)
    assert_upper("far_far_gate_is_below_one", far_far_gate_upper, 1)
    assert_lower("central_far_near_fitness_gap", central_far_near_gap, "0.0071")
    assert_lower("cross_far_far_fitness_gap", cross_far_far_gap, "0.8581")
    assert_lower("cross_near_near_fitness_gap", cross_near_near_gap, "0.0718")
    assert_lower("band_copy_gain", incoming - reverse, "0.3813")
    assert_lower("band_complete_growth", complete_growth, "1.0387")

    t = interval("0.02")
    c = iv.exp(interval("-0.04"))
    b = t * (1 + c)
    eta = b * t
    ou_variance = (1 - iv.exp(interval("-0.08"))) / 2
    tau = iv.sqrt(t**2 * ou_variance + interval("0.0004"))
    target_radius = interval(1) / 16
    velocity_shift = b * interval("0.002")
    eroded_radius = target_radius - velocity_shift

    def g(x):
        return (1 - 2 * eta) * x - 20 * iv.pi * eta * iv.sin(2 * iv.pi * x)

    assert_lower("native_g_derivative_lower", 1 - eta * (2 + 40 * iv.pi**2), 0)
    assert_lower("eroded_target_radius", eroded_radius, 0)

    def landing(radius, mean):
        return Phi((radius - mean) / tau) - Phi((-radius - mean) / tau)

    persistent_band = landing(target_radius, g(eps) + velocity_shift) ** 3
    persistent_seed = landing(target_radius, interval(0)) ** 3
    assert_lower("persistent_band_landing", persistent_band, "0.99347")
    assert_lower("persistent_seed_landing", persistent_seed, "0.993")

    def jitter_lower(radius, source_offset):
        coordinate_lower = interval(0)
        for j in range(320):
            left = interval(-8) + interval(j) / 20
            right = interval(-8) + interval(j + 1) / 20
            # The source interval endpoint maximum encloses every own jitter
            # in this bin. Odd monotonic g and the even decreasing landing
            # probability give the binwise infimum bound.
            left_source = source_offset + interval("0.1") * left
            right_source = source_offset + interval("0.1") * right
            abs_source_upper = max(
                abs(lo(left_source)),
                abs(hi(left_source)),
                abs(lo(right_source)),
                abs(hi(right_source)),
            )
            mass = Phi(right) - Phi(left)
            probability = landing(radius, g(interval(abs_source_upper)))
            mass_lower = max(mp.mpf(0), lo(mass))
            probability_lower = max(mp.mpf(0), lo(probability))
            coordinate_lower += interval(mass_lower) * interval(probability_lower)
        # Discarding the complementary tails is a lower bound on the
        # unrestricted Gaussian integral, not a change to its law.
        return coordinate_lower**3

    accepted_band = jitter_lower(eroded_radius, eps)
    accepted_seed = jitter_lower(target_radius, interval(0))
    assert_lower("accepted_band_320_bin_lower", accepted_band, "0.20599")
    assert_lower("accepted_seed_320_bin_lower", accepted_seed, "0.2")

    w = iv.exp(-interval(1) / 18)
    delta = iv.sqrt(interval(4) / 9 + interval("0.000001")) - interval("0.001")
    diversity_sd_max = iv.sqrt(delta**2 / 4 + interval("0.01"))
    diversity_ratio_lower = G(delta / (2 * diversity_sd_max)) / interval("1.1")
    reward_B_large_N = G(interval("0.875") / iv.sqrt(interval("0.109375") + interval("0.01")))
    large_N_ratio = reward_B_large_N / interval("1.1") * diversity_ratio_lower
    assert_lower("N_at_least_8_near_gate_ratio", large_N_ratio, "2.4939")
    # The actual clipping threshold is <= 2 + 1e-6/.01 = 2.0001.
    assert_lower("N_at_least_8_clipping_margin", large_N_ratio - interval("2.0001"), 0)

    near_margins = []
    N_two_far_margin = None
    for N in range(2, 8):
        reward_sd = iv.sqrt(interval(N - 1) / N**2 + interval("0.01"))
        resident_reward = G(-1 / (N * reward_sd))
        central_reward = G(interval(N - 1) / (N * reward_sd))
        for k in range(N):
            fraction = interval(k + 1) / N
            diversity_sd = iv.sqrt(fraction * (1 - fraction) * delta**2 + interval("0.01"))
            near_factor = G(-fraction * delta / diversity_sd)
            far_factor = G((1 - fraction) * delta / diversity_sd)
            margin = (
                central_reward * far_factor
                - 2 * resident_reward * near_factor
                - interval("0.000001")
            )
            assert lo(margin) > hi(interval("1.5568")), (N, k)
            near_margins.append(margin)
        if N == 2:
            N_two_far_margin = (central_reward - 2 * resident_reward) * interval("1.1") - interval(
                "0.000001"
            )
    near_margin_enclosure = interval([
        min(lo(value) for value in near_margins),
        min(hi(value) for value in near_margins),
    ])
    assert_lower("N_2_through_7_near_clipping_min", near_margin_enclosure, "1.5568")
    assert N_two_far_margin is not None
    assert_lower("N_2_actual_far_clipping_margin", N_two_far_margin, "0.2896")
    seed_copy_lower = 2 * w / (1 + w) ** 2
    assert_lower("all_N_single_seed_copy_gain", seed_copy_lower, "0.4996")
    seed_spatial_lower = interval("0.993") + interval("0.2") * seed_copy_lower
    assert_lower("all_N_single_seed_complete_spatial_count", seed_spatial_lower, "1.0929")

    # The published finite binomial sum, evaluated with its actual common
    # normalizer, supplies the native N = 128 numerical receipt as well.
    N = 128
    probability = w / (N - 2 + w)
    reward_sd = iv.sqrt(interval(N - 1) / N**2 + interval("0.01"))
    resident_reward = G(-1 / (N * reward_sd))
    central_reward = G(interval(N - 1) / (N * reward_sd))

    def gate(resident_fitness, central_fitness):
        difference = central_fitness - resident_fitness
        numerator = interval([
            max(mp.mpf(0), lo(difference)),
            max(mp.mpf(0), hi(difference)),
        ])
        ratio = numerator / (resident_fitness + interval("0.000001"))
        return interval([min(mp.mpf(1), lo(ratio)), min(mp.mpf(1), hi(ratio))])

    seed_N_128_gain = interval(0)
    for k in range(N):
        fraction = interval(k + 1) / N
        diversity_sd = iv.sqrt(fraction * (1 - fraction) * delta**2 + interval("0.01"))
        near_factor = G(-fraction * delta / diversity_sd)
        far_factor = G((1 - fraction) * delta / diversity_sd)
        binomial_mass = (
            interval(math.comb(N - 1, k)) * probability**k * (1 - probability) ** (N - 1 - k)
        )
        gain_given_k = (N - 1 - k) * gate(
            resident_reward * near_factor, central_reward * far_factor
        ) + k * gate(resident_reward * far_factor, central_reward * far_factor)
        seed_N_128_gain += probability * binomial_mass * gain_given_k
    assert_lower("N_128_exact_copy_gain", seed_N_128_gain, "0.9460819373977")
    assert_upper("N_128_exact_copy_gain_upper", seed_N_128_gain, "0.9460819373978")
    print("All native source certificates passed.")


if __name__ == "__main__":
    certificate()
