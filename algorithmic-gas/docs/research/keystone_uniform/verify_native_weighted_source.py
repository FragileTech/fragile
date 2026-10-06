"""Outward receipts for original-noise native weighted sources.

Run from the repository root with::

    UV_CACHE_DIR=/tmp/fragile-uv-cache uv run python \\
        docs/research/keystone_uniform/verify_native_weighted_source.py

The program evaluates exact Gaussian moments and analytic tail bounds.
It does not simulate or clip the original laws. Importing it is safe.
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
        scale = mp.mpf(10) ** (25 - int(mp.floor(mp.log10(max(abs(lo(x)), abs(hi(x)))))))
        print(
            name,
            "["
            + mp.nstr(mp.floor(lo(x) * scale) / scale, 26)
            + ", "
            + mp.nstr(mp.ceil(hi(x) * scale) / scale, 26)
            + "]",
            flush=True,
        )

    t = I(".02")
    cO = iv.exp(I("-.04"))
    b = t * (1 + cO)
    eta = t * b
    q2 = (1 - iv.exp(I("-.08"))) / 2
    q = iv.sqrt(q2)
    tau2 = t * t * q2 + I(".0004")
    A0 = 1 - 2 * eta
    A1 = 20 * iv.pi * eta
    angle = 2 * iv.pi
    varJ = I(".01")
    g2 = (
        A0 * A0 * varJ
        - 2 * A0 * A1 * angle * varJ * iv.exp(-angle * angle * varJ / 2)
        + A1 * A1 * (1 - iv.exp(-2 * angle * angle * varJ)) / 2
    )
    EU0 = 3 * (tau2 + 10 * (1 - iv.exp(-2 * iv.pi * iv.pi * tau2)))
    EUJ = 3 * (1 + 20 * iv.pi * iv.pi) * (g2 + tau2)
    beta = I("1e-78")
    cw = I(".25")
    mass0 = iv.exp(-cw * EU0) - beta
    massJ = iv.exp(-cw * EUJ) - beta
    uniform = mass0 + I(".4996") * massJ
    assert lo(uniform) > hi(I("1.14"))
    out("persistent_actual_potential_mean", EU0)
    out("jittered_actual_potential_mean_upper", EUJ)
    out("all_N_first_potential_weight_source_mass_lower", uniform)
    joint_small = iv.exp(-2 * cw * I(".01")) * uniform
    assert lo(joint_small) > hi(I("1.140"))
    out("all_N_first_small_velocity_joint_source_mass_lower", joint_small)
    r2 = t * t * q2
    alpha = r2 / (1 + r2)
    mA = 1 - 2 * eta
    wa = -2 * cO * t
    P = q * (1 - 2 * t * t)
    Q = 20 * iv.pi * t
    av = angle * t * q
    Ebase = 3 * (
        P * P - 2 * P * Q * av * iv.exp(-av * av / 2) + Q * Q * (1 - iv.exp(-2 * av * av)) / 2
    )
    h0 = -wa + alpha * mA / t
    EKE0 = (iv.sqrt(Ebase) + t * I(".3") * h0) ** 2 / 2
    assert lo(1 - t * t * (2 + 40 * iv.pi * iv.pi) - t * I(".3")) > 0
    joint0 = iv.exp(-cw * (EU0 + EKE0)) - beta
    jointJ = iv.exp(-cw * (EUJ + 2)) - beta
    reference = joint0 + iv.exp(-I(1) / 18) * jointJ
    assert lo(reference) > hi(I("1.16"))
    out("persistent_reference_cap_kinetic_energy_mean_upper", EKE0)
    out("first_reference_joint_energy_weight_source_mass_lower", reference)
    z = 10 * cw
    K = 40
    series_tail = (
        2
        * iv.exp(z * z / 4)
        * (z / 2) ** (K + 1)
        / (I(math.factorial(K + 1)) * (1 - z / (2 * (K + 2))))
    )
    assert hi(series_tail) < lo(I("4e-44"))
    out("native_potential_laplace_fourier_tail_bound_K40", series_tail)
    full_series_tail = iv.exp(-z) * series_tail
    assert hi(full_series_tail) < lo(I("2.765e-45"))
    out("native_potential_laplace_full_uniform_error_K40", full_series_tail)
    Lpot = iv.sqrt(2 * cw / iv.e) + 20 * iv.pi * iv.sqrt(3) * cw
    out("potential_price_global_Lipschitz_bound", Lpot)
    haar_argument = 4 * iv.pi * b
    haar_characteristic_floor = 1 - 4 * (2 * iv.pi * b) ** 2 / 6
    haar_potential_charge = 4 * b**2 * (1 + 20 * iv.pi**2 * iv.exp(-2 * iv.pi**2 * tau2))
    assert hi(haar_argument) < lo(I(".492801"))
    assert lo(haar_characteristic_floor) > hi(I(".95952467"))
    assert hi(haar_potential_charge) < lo(I("1.210497"))
    out("maximum_component_Haar_argument", haar_argument)
    out("component_Haar_characteristic_floor", haar_characteristic_floor)
    out("absolute_component_Haar_potential_charge", haar_potential_charge)
    print("All weighted-source primitive receipts passed.", flush=True)


if __name__ == "__main__":
    certificate()
