"""Outward interval receipt for native dispersed-source and basin producers.

Run from the repository root with::

    UV_CACHE_DIR=/tmp/fragile-uv-cache uv run python \
        docs/research/keystone_uniform/verify_native_source_pressure.py

The fitness-average import is proved by ``verify_logistic_fitness.py``.
This program evaluates original Gaussian formulas and finite lower sums;
it does not simulate or clip their tails. Importing performs no calculation.
"""

import math

import mpmath as mp


def certificate():
    mp.mp.dps = 110
    mp.iv.dps = 85
    iv = mp.iv
    I = iv.mpf

    def lo(x):
        return mp.mpf(x._mpi_[0])

    def hi(x):
        return mp.mpf(x._mpi_[1])

    def out(name, x):
        mag = max(abs(lo(x)), abs(hi(x)))
        scale = mp.mpf(10) ** (25 - int(mp.floor(mp.log10(mag))))
        print(
            name,
            "["
            + mp.nstr(mp.floor(lo(x) * scale) / scale, 26)
            + ", "
            + mp.nstr(mp.ceil(hi(x) * scale) / scale, 26)
            + "]",
            flush=True,
        )

    def upper(name, x, y):
        assert hi(x) < lo(I(y)), name
        out(name, x)

    def lower(name, x, y):
        assert lo(x) > hi(I(y)), name
        out(name, x)

    first = I(8) ** 803 / (I(2) ** 401 * I(math.factorial(401)) * 803 * iv.sqrt(2 * iv.pi))
    upper("cdf_remainder_803_at_8", first, "1e-270")

    def endpoint(x):
        x = I(x)
        term = x
        total = x
        for n in range(400):
            term *= -x * x * (2 * n + 1) / (2 * (n + 1) * (2 * n + 3))
            total += term
        return I(".5") + total / iv.sqrt(2 * iv.pi) + I(["-1e-200", "1e-200"])

    def Phi(x):
        def ep(z):
            if z > 8:
                return I([lo(1 - I("6.5e-16")), 1])
            if z < -8:
                return I([0, hi(I("6.5e-16"))])
            return endpoint(z)

        return I([lo(ep(lo(x))), hi(ep(hi(x)))])

    def chi_tail(x):
        return 2 * (1 - Phi(x)) + iv.sqrt(2 / iv.pi) * x * iv.exp(-x * x / 2)

    def G(x):
        return 2 / (1 + iv.exp(-x)) + I(".1")

    t = I(".02")
    c = iv.exp(I("-.04"))
    b = t * (1 + c)
    eta = t * b
    q2 = (1 - iv.exp(I("-.08"))) / 2
    q = iv.sqrt(q2)
    tau2 = t * t * q2 + I(".0004")
    tau = iv.sqrt(tau2)
    r2 = t * t * q2
    alpha = r2 / (1 + r2)
    m = 1 - 2 * eta
    wa = -2 * c * t
    wave = 2 * iv.pi
    damp = iv.exp(-(wave**2) * tau2 / 2)
    cos_sum = iv.cos(wave * m) + 2
    cos_square_sum = iv.cos(wave * m) ** 2 + 2
    cos_double_sum = iv.cos(2 * wave * m) + 2
    reward_mean = m**2 + 3 * tau2 + 30 - 10 * damp * cos_sum
    cosine_variance = (3 + iv.exp(-2 * wave**2 * tau2) * cos_double_sum) / 2 - (
        damp**2 * cos_square_sum
    )
    square_cosine_covariance = -damp * (
        wave**2 * tau2**2 * cos_sum + 2 * wave * tau2 * m * iv.sin(wave * m)
    )
    reward_variance = (
        6 * tau2**2 + 4 * tau2 * m**2 + 100 * cosine_variance - 20 * square_cosine_covariance
    )
    reward_sd = iv.sqrt(I(".01") + reward_variance)
    lower("actual_reward_mean_lower", reward_mean, "1.24356364743449")
    upper("actual_reward_mean_upper", reward_mean, "1.24356364743450")
    lower("actual_reward_variance_lower", reward_variance, ".04088023948485")
    upper("actual_reward_variance_upper", reward_variance, ".04088023948487")
    lower("actual_reward_sd_lower", reward_sd, ".22556648573060")
    upper("actual_reward_sd_upper", reward_sd, ".22556648573063")
    lam = t * I(".3") * (1 - alpha) * (1 + r2) ** (-I(3) / 2)
    B = 1 - 2 * t * t - lam
    H = 40 * iv.pi**2 * t * t
    ang = 2 * iv.pi * t * q
    theta = 2 * iv.pi * m
    jacbase = q2 * (
        3 * B * B
        - 2 * B * H * iv.exp(-ang * ang / 2) * (iv.cos(theta) + 2)
        + H * H * (3 + iv.exp(-2 * ang * ang) * (iv.cos(2 * theta) + 2)) / 2
    )
    corr = q * lam * alpha * (1 + iv.sqrt(3) / 2) * iv.sqrt(15)
    velocityvar = (iv.sqrt(jacbase) + corr) ** 2
    upper("resident_velocity_feature_variance_trace", velocityvar, ".0806")
    beta = I("1e-520")
    avarlive = (velocityvar + 3 * tau2) / (1 - beta)
    divmean = iv.sqrt(2 * avarlive + I(".000001"))
    divsd = iv.sqrt(I(".01") + 2 * avarlive + I(".000001"))
    upper("resident_diversity_mean_upper", divmean, ".405")
    upper("resident_diversity_sd_upper", divsd, ".417")
    delta = 2 * iv.pi * (1 - m)
    velocitymean = -wa + 2 * t * m + 40 * iv.pi * t * iv.sin(delta / 2) + 2 * beta / (1 - beta)
    upper("resident_velocity_feature_mean_norm_upper", velocitymean, ".091")
    positionmean = 2 * m / (2 + m) + I(9) / 4 * tau2 + 4 * beta / (1 - beta)
    upper("resident_position_feature_mean_norm_upper", positionmean, ".667")
    sourceR = iv.sqrt(I(".87") / (1 + 20 * iv.pi**2))
    sourcepos = 2 * sourceR / (2 + sourceR)
    upper("persistent_source_feature_position_bound", sourcepos, ".0642")
    ED2 = (
        (positionmean + sourcepos) ** 2
        + 3 * tau2 / (1 - beta)
        + (velocitymean + I(".45")) ** 2
        + velocityvar / (1 - beta)
    )
    degree = iv.exp(-ED2 / 8)
    lower("persistent_source_degree_lower", degree, ".89")
    own_measure_distance = 2 * I(".85") / (2 + I(".85")) - sourcepos
    lower("persistent_source_measurement_diversity_lower", own_measure_distance, ".53")
    fmin = G((I("1.24356364743449") - I(".87")) / I(".22556648573063")) * G(
        (own_measure_distance - divmean) / divsd
    )
    lower("persistent_source_fitness_lower", fmin, "2.2")
    xiR = I("4.9")
    kappa = 2 + 40 * iv.pi**2
    sicoef = 1 - t * t * kappa + H * (ang * xiR) ** 2 / 6
    lower("central_prevelocity_coefficient_lower", 1 - t * t * kappa - t * I(".3"), 0)
    h0 = -wa + alpha * m / t
    vmax = q * xiR * sicoef + t * I(".3") * h0
    upper("persistent_prevelocity_bound_on_chi_event", vmax, "0.81818181818181818")
    upper("persistent_velocity_feature_bound", vmax / (1 + vmax), ".45")
    xtail = chi_tail(sourceR / tau)
    vtail = chi_tail(xiR)
    sourceprob = 1 - xtail - vtail
    lower("persistent_joint_source_good_probability", sourceprob, ".985")
    zmeasure = (m - I(".85")) / tau
    lower("resident_low_position_gaussian_tail_z", zmeasure, "7.28")
    mesurebad = iv.exp(-(I("7.28") ** 2) / 2) / (I("7.28") * iv.sqrt(2 * iv.pi))
    ka = iv.exp(-(13 - 6 * iv.sqrt(3)) / 2)
    markbad = mesurebad / (ka * (1 - beta))
    upper("persistent_own_measurement_bad_upper", markbad, "7e-13")
    # This line imports the root's independently proved realized-array
    # bound avgFitness<=1.614; it does not prove that scalar inequality.
    excess = (fmin * degree - I("1.614") / degree) / (fmin + I(".000001"))
    lower("limiting_good_source_signed_excess", excess, ".06")
    reproduction = sourceprob * (1 - markbad) * (1 + excess)
    lower("persistent_second_preparation_mean_lower", reproduction, "1.04")
    upper("gaussian_tail_mills_bound_at_8", iv.exp(-I(32)) / (8 * iv.sqrt(2 * iv.pi)), "6.5e-16")
    eroded = I(".5") - 4 * b

    def g(x):
        return (1 - 2 * eta) * x - 20 * iv.pi * eta * iv.sin(2 * iv.pi * x)

    def landing(mean):
        return Phi((eroded - mean) / tau) - Phi((-eroded - mean) / tau)

    p0 = landing(g(sourceR)) ** 3
    lower("good_source_persistent_second_basin_landing", p0, ".999999999999")
    pJcoord = I(0)
    for j in range(320):
        left = -8 + I(j) / 20
        right = -8 + I(j + 1) / 20
        sxleft = sourceR + I(".1") * left
        sxright = sourceR + I(".1") * right
        xmax = max(abs(lo(sxleft)), abs(hi(sxleft)), abs(lo(sxright)), abs(hi(sxright)))
        mass = Phi(right) - Phi(left)
        lp = landing(g(I(xmax)))
        pJcoord += I(max(mp.mpf(0), lo(mass))) * I(max(mp.mpf(0), lo(lp)))
    pJ = pJcoord**3
    lower("good_source_accepted_second_basin_landing_320_lower", pJ, ".995")
    lower(
        "persistent_atom_second_completed_basin_mean_lower",
        sourceprob * (1 - markbad) * pJ * (1 + excess),
        "1.06",
    )
    lower(
        "rectangle_basin_landing_margin",
        I(".5") - g(sourceR + I(".25")) - 4 * b - I("3.4") * tau,
        0,
    )
    p0rect = (2 * Phi(I("3.4")) - 1) ** 3
    pJrect = (2 * Phi(I("2.5")) - 1) ** 3 * p0rect
    lower("good_source_persistent_rectangle_basin_probability", p0rect, ".9979")
    lower("good_source_accepted_rectangle_basin_probability", pJrect, ".961")
    finite_excess = (I("2.2") * I(".89") - I("1.614") * 1000 / (999 * I(".89"))) / (
        I("2.2") + I(".000001")
    )
    lower("finite_measured_source_signed_excess", finite_excess, ".0648")
    lower("finite_measured_source_complete_basin_mean_lower", pJ * (1 + finite_excess), "1.061")
    incoming_lambda = pJ * finite_excess
    trial_max = 1 / (ka * 999)
    upper("finite_incoming_basin_lower_trial_probability_max", trial_max, ".0037")
    two_births = 1 - iv.exp(-incoming_lambda) * (1 + incoming_lambda / (1 - trial_max))
    lower("finite_measured_source_two_incoming_basin_births_lower", two_births, ".0017")
    lower(
        "finite_measured_source_rectangle_complete_basin_mean",
        pJrect * (1 + finite_excess),
        "1.023",
    )
    rectlambda = pJrect * finite_excess
    recttwo = 1 - iv.exp(-rectlambda) * (1 + rectlambda / (1 - trial_max))
    lower("finite_measured_source_rectangle_two_incoming_births", recttwo, ".0016")
    print("All native reference/source primitive receipts passed.", flush=True)


if __name__ == "__main__":
    certificate()
