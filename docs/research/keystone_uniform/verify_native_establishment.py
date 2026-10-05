"""Standalone outward interval receipts for native inner/basin producers."""

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

    def receipt(name, value):
        magnitude = max(abs(lo(value)), abs(hi(value)))
        scale = mp.mpf(10) ** (25 - int(mp.floor(mp.log10(magnitude))))
        left = mp.floor(lo(value) * scale) / scale
        right = mp.ceil(hi(value) * scale) / scale
        print(name, "[" + mp.nstr(left, 26) + ", " + mp.nstr(right, 26) + "]", flush=True)

    first_omitted = I(8) ** 803 / (I(2) ** 401 * I(math.factorial(401)) * 803) / iv.sqrt(2 * iv.pi)
    assert hi(first_omitted) < lo(I("1e-270"))
    assert hi(iv.exp(-I(32)) / (8 * iv.sqrt(2 * iv.pi))) < lo(I("6.5e-16"))

    def cdf_endpoint(value):
        if value > 8:
            return I([lo(I(1) - I("6.5e-16")), 1])
        if value < -8:
            return I([0, hi(I("6.5e-16"))])
        x = I(value)
        term = x
        total = x
        for n in range(400):
            term *= -x * x * (2 * n + 1) / (2 * (n + 1) * (2 * n + 3))
            total += term
        return I(".5") + total / iv.sqrt(2 * iv.pi) + I(["-1e-200", "1e-200"])

    def Phi(value):
        return I([lo(cdf_endpoint(lo(value))), hi(cdf_endpoint(hi(value)))])

    t = I(".02")
    c = iv.exp(I("-.04"))
    b = t * (1 + c)
    eta = t * b
    q2 = (1 - iv.exp(I("-.08"))) / 2
    tau2 = t * t * q2 + I(".0004")
    tau = iv.sqrt(tau2)
    basin_eroded_radius = I(".5") - 4 * b

    def g(x):
        return (1 - 2 * eta) * x - 20 * iv.pi * eta * iv.sin(2 * iv.pi * x)

    def landing(radius, mean):
        return Phi((radius - mean) / tau) - Phi((-radius - mean) / tau)

    def finite_jitter_lower(radius, source_offset):
        coordinate_lower = I(0)
        for j in range(320):
            left = -8 + I(j) / 20
            right = -8 + I(j + 1) / 20
            source_left = source_offset + I(".1") * left
            source_right = source_offset + I(".1") * right
            abs_bound = max(
                abs(lo(source_left)),
                abs(hi(source_left)),
                abs(lo(source_right)),
                abs(hi(source_right)),
            )
            mass = Phi(right) - Phi(left)
            probability = landing(radius, g(I(abs_bound)))
            coordinate_lower += I(max(mp.mpf(0), lo(mass))) * I(max(mp.mpf(0), lo(probability)))
        return coordinate_lower**3

    persistent_basin = landing(basin_eroded_radius, g(I(".25"))) ** 3
    assert lo(persistent_basin) > hi(1 - I("4e-12"))
    receipt("persistent_inner_to_basin", persistent_basin)
    accepted_basin = finite_jitter_lower(basin_eroded_radius, I(".25"))
    assert lo(accepted_basin) > hi(I(".70894479"))
    receipt("accepted_inner_to_basin_320_bin_lower", accepted_basin)

    persistent_inner = landing(I(".25"), I(0)) ** 3
    assert lo(persistent_inner) > hi(I(".999"))
    receipt("persistent_seed_inner", persistent_inner)
    accepted_inner = finite_jitter_lower(I(".25"), I(0))
    assert lo(accepted_inner) > hi(I(".98870749"))
    receipt("accepted_seed_inner_320_bin_lower", accepted_inner)
    amplification = I(".999") * (1 - iv.exp(-I(".98") * I(".4996")))
    assert lo(amplification) > hi(I(".3867"))
    receipt("uniform_inner_seed_amplification", amplification)

    kappa = 2 + 40 * iv.pi**2
    K = kappa * (1 - t * t * kappa)
    A0 = 1 - 2 * eta
    A1 = 20 * iv.pi * eta
    angle = 2 * iv.pi
    jitter_variance = I(".01")
    mean_g_square = (
        A0**2 * jitter_variance
        - 2 * A0 * A1 * angle * jitter_variance * iv.exp(-(angle**2) * jitter_variance / 2)
        + A1**2 * (1 - iv.exp(-2 * angle**2 * jitter_variance)) / 2
    )
    persistent_price = I(3) / 2 * K * tau2 + 2
    birth_price = I(3) / 2 * K * (mean_g_square + tau2) + 2
    assert hi(persistent_price) < lo(I("2.208"))
    assert hi(birth_price) < lo(I("4.991"))
    receipt("persistent_actual_cap_energy_upper", persistent_price)
    receipt("birth_actual_cap_energy_upper", birth_price)

    death_z = (2 - A1) / iv.sqrt(A0**2 * jitter_variance + tau2)
    assert lo(death_z) > 19
    death_upper = 6 * iv.exp(-(I(19) ** 2) / 2) / (19 * iv.sqrt(2 * iv.pi))
    assert hi(death_upper) < lo(I("1e-78"))
    receipt("birth_source_death_mills_upper", death_upper)
    heat = I(3) / 2 * (1 - t * t * kappa) * (q2 + kappa * I(".0004"))
    receipt("local_modified_energy_original_noise_heat", heat)
    print("All native establishment primitive certificates passed.", flush=True)


if __name__ == "__main__":
    certificate()
