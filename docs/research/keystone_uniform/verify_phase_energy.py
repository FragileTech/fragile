"""Outward receipt for native KUPC.27–48 and full Gaussian tails."""

import mpmath as mp


def main():
    mp.mp.dps = 100
    mp.iv.dps = 80
    iv = mp.iv
    interval = iv.mpf

    def lower(value):
        return mp.mpf(value._mpi_[0])

    def upper(value):
        return mp.mpf(value._mpi_[1])

    def show(name, value):
        print(name, mp.nstr(lower(value), 28), mp.nstr(upper(value), 28))

    c = iv.exp(-interval(".04"))
    b = interval(".02") * (1 + c)
    eta = interval(".02") * b
    ell = 1 - 2 * eta
    amplitude = 20 * iv.pi * eta
    sigma, delta, bins = interval(".1"), interval(10), 8000
    width = interval(8) / bins
    tail_a = interval(".5") - delta * ell**2 * sigma**2
    assert lower(tail_a) > 0

    def log_mgf_upper(source_radius):
        source_radius = interval(source_radius)
        total = interval(0)
        for j in range(1, bins + 1):
            left, right = width * (j - 1), width * j
            x_right = source_radius + sigma * right
            gr = ell * x_right - amplitude * iv.sin(2 * iv.pi * x_right)
            total += iv.exp(delta * gr**2 - left**2 / 2)
        tail_constant = ell * source_radius + amplitude
        tail_b = 2 * delta * ell * sigma * tail_constant
        assert lower(16 * tail_a - tail_b) > 0
        tail = (
            2
            / iv.sqrt(2 * iv.pi)
            * iv.exp(delta * tail_constant**2 - 64 * tail_a + 8 * tail_b)
            / (16 * tail_a - tail_b)
        )
        return iv.ln(2 * width / iv.sqrt(2 * iv.pi) * total + tail)

    mgf_zero, mgf_band = log_mgf_upper("0"), log_mgf_upper(".005")
    show("zero_source_log_mgf_upper", mgf_zero)
    show("positive_volume_source_log_mgf_upper", mgf_band)
    assert upper(mgf_zero) < mp.mpf(".061")
    assert upper(mgf_band) < mp.mpf(".066")
    q2 = (1 - iv.exp(-interval(".08"))) / 2
    tau2 = interval(".0004") * (1 + q2)
    energy_count = interval(2)
    energy_row = 2 * (interval(".994") + interval(".006") * iv.sqrt(interval(7188)))
    for mode, rg, rn, energy, upper_h, lower_in, zero_rate, band_rate in (
        ("count", ".15", ".05", energy_count, "17.380177", ".4649", ".042", ".027"),
        ("row", ".145", ".037", energy_row, "19.838454", ".0068", ".02725", ".01225"),
    ):
        radius = interval(rg) + b * energy + interval(rn)
        joint_energy = (1 + 20 * iv.pi**2) * radius**2 + 2
        jitter_rate = 10 * interval(rg) ** 2 - 3 * mgf_zero
        band_jitter_rate = 10 * interval(rg) ** 2 - 3 * mgf_band
        z = interval(rn) ** 2 / (3 * tau2)
        noise_rate = interval("1.5") * (z - 1 - iv.ln(z))
        show(mode + " joint_energy", joint_energy)
        show(mode + " jitter_rate", jitter_rate)
        show(mode + " positive_volume_jitter_rate", band_jitter_rate)
        show(mode + " noise_rate", noise_rate)
        assert upper(joint_energy) < mp.mpf(upper_h)
        assert lower(jitter_rate) > mp.mpf(zero_rate)
        assert lower(band_jitter_rate) > mp.mpf(band_rate)
        assert lower(noise_rate) > mp.mpf(lower_in)
    kg = (
        ell**2 * sigma**2
        - 2 * ell * amplitude * (2 * iv.pi) * sigma**2 * iv.exp(-2 * iv.pi**2 * sigma**2)
        + amplitude**2 / 2 * (1 - iv.exp(-8 * iv.pi**2 * sigma**2))
    )
    show("gaussian_native_g_square", kg)
    assert lower(kg) > mp.mpf(".00555616116647902")
    assert upper(kg) < mp.mpf(".00555616116647903")
    for mode, energy, bound in (
        ("count", energy_count, "10.792376"),
        ("row", energy_row, "14.347498"),
    ):
        position_square = (iv.sqrt(3 * kg) + b * energy) ** 2 + 3 * tau2
        joint_energy = (1 + 20 * iv.pi**2) * position_square + 2
        show(mode + " expected_joint_energy", joint_energy)
        assert upper(joint_energy) < mp.mpf(bound)
    print("All native phase-energy margins certified.")


if __name__ == "__main__":
    main()
