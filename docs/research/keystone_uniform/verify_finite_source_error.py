"""Outward primitive receipt for the finite native second-source producer."""

import mpmath as mp


def certificate():
    mp.mp.dps = 110
    mp.iv.dps = 85
    iv = mp.iv
    I = iv.mpf

    def lo(value):
        return mp.mpf(value._mpi_[0])

    def hi(value):
        return mp.mpf(value._mpi_[1])

    def receipt(name, value):
        print(name, "[" + mp.nstr(lo(value), 27) + ", " + mp.nstr(hi(value), 27) + "]")

    def phi(value):
        return iv.exp(-value * value / 2) / iv.sqrt(2 * iv.pi)

    def G(value):
        exp_value = iv.exp(value)
        return I("1.1") + (exp_value - 1) / (exp_value + 1)

    t = I(".02")
    c = iv.exp(I("-.04"))
    eta = t * t * (1 + c)
    q2 = (1 - iv.exp(I("-.08"))) / 2
    q = iv.sqrt(q2)
    tau = iv.sqrt(t * t * q2 + I(".0004"))
    m = 1 - 2 * eta
    beta_bad = phi((m - I(".85")) / tau) / ((m - I(".85")) / tau)
    assert hi(beta_bad) < lo(I("2e-13"))
    receipt("bad_resident_position_Mills_upper", beta_bad)

    six_tail = 12 * phi(I(20)) / 20
    weighted_tail = 6 * phi(I(20)) + 30 * iv.sqrt(2 / iv.pi) * phi(I(20)) / 20
    assert hi(six_tail) < lo(I("1e-86"))
    assert hi(weighted_tail) < lo(I("1e-85"))
    receipt("six_original_Gaussian_tail_upper", six_tail)
    receipt("original_OU_tail_first_moment_upper", weighted_tail)

    rho = I("1e-25") + I("1e-80")
    assert hi(rho) < lo(I("1.01e-25"))
    receipt("rho_at_explicit_population_threshold", rho)
    assert hi(I("1e12") * rho / I("1e-5")) < lo(I("1.01e-8"))

    root_denominator = (1 + t * t * q2) ** I("-1.5") * iv.exp(-I(9) / (2 * (1 + t * t * q2)))
    assert lo(root_denominator) > hi(I(".01"))
    receipt("row_B2_reference_denominator_lower", root_denominator)

    force_lipschitz = 2 + 40 * iv.pi**2
    force_derivative = t * force_lipschitz * t * q
    assert hi(force_derivative) < lo(I(".032"))
    receipt("reference_force_derivative_upper", force_derivative)
    exceptional_velocity = (
        t * (2 * (1 + I(".1") * iv.sqrt(3)) + 20 * iv.pi * iv.sqrt(3))
        + q * iv.sqrt(3)
        + I(".04")
        + q * iv.sqrt(3)
    )
    assert hi(exceptional_velocity) < 4
    receipt("full_exceptional_velocity_first_moment_upper", exceptional_velocity)

    reward_z = (I("1.24356364743449") - I(".87") - I("1e-5")) / (I(".22556648573063") + I("1e-5"))
    diversity_z = (I(".5323923214354588021") - I(".404275741942611773") - I("1e-5")) / (
        I(".416459932674500186") + I("1e-5")
    )
    buffered_fitness = G(reward_z) * G(diversity_z)
    assert lo(buffered_fitness) > hi(I("2.2"))
    receipt("finite_register_buffered_source_fitness_lower", buffered_fitness)
    signed_excess = (I("2.2") * I(".89") - I("1.614") * I(1000) / 999 / I(".89")) / (
        I("2.2") + I("1e-6")
    )
    assert lo(signed_excess) > hi(I(".0648"))
    receipt("actual_finite_signed_source_excess_lower", signed_excess)
    completed_growth = I(".9855") * I(".9965") * I("1.0648")
    assert lo(completed_growth) > hi(I("1.0456"))
    receipt("two_complete_updates_basin_lineage_mean_lower", completed_growth)
    print("All finite native second-source primitive receipts passed.")


if __name__ == "__main__":
    certificate()
