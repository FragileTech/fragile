"""Interval receipt for the finite-population part of the native discovery clock."""

import mpmath as mp


def main():
    mp.mp.dps = 100
    mp.iv.dps = 85
    iv, interval = mp.iv, mp.iv.mpf

    def lower(value):
        return mp.mpf(value._mpi_[0])

    def upper(value):
        return mp.mpf(value._mpi_[1])

    def show(name, value):
        print(name, mp.nstr(lower(value), 28), mp.nstr(upper(value), 28))

    t, nu, vc = interval(".02"), interval(".3"), interval(4)
    c = iv.exp(-interval(".04"))
    q2, s = (1 - iv.exp(-interval(".08"))) / 2, interval(".02")
    q = iv.sqrt(q2)
    lf = 2 + 40 * iv.pi**2
    ell = iv.exp(-interval(".5"))
    rd = 2 * iv.sqrt(interval(3))
    j0, jitter_radius, noise_radius = interval(".1"), interval(29), interval(290)

    def stages(radius):
        b0 = rd + radius
        b1 = (1 + 2 * t * nu) * vc + t * lf * b0
        return b0, b1, b0 + t * b1

    _, b10, bx0 = stages(j0)
    mean = 2 + j0 + t * (1 + c) * b10
    tau = iv.sqrt(t**2 * q2 + s**2)
    z, far_z = (mean - 2) / tau, (mean + 2) / tau

    def phi(value):
        return iv.exp(-(value**2) / 2) / iv.sqrt(2 * iv.pi)

    # Minimum Gaussian density integrated over the full unit ball.
    ball_lower = (4 * iv.pi / 3) * (2 * iv.pi) ** (-interval("1.5")) * iv.exp(-interval(".5"))
    # Both Mills inequalities are analytic and valid for positive arguments.
    interval_lower = phi(z) * z / (1 + z**2) - phi(far_z) / far_z
    assert lower(interval_lower) > 0
    log_survival_lower = iv.ln(ball_lower) + 3 * iv.ln(interval_lower)
    show("log_survival_lower", log_survival_lower)
    assert lower(log_survival_lower) > -6800
    b0, b1, bx = stages(jitter_radius)
    z0 = c * b1 + q * noise_radius
    r2 = bx + t * z0
    target_h = (1 + 2 * t * nu) * z0 + t * lf * r2
    target_r = r2 + s * noise_radius
    for mode, n_max, chi, log_budget in (
        ("count", 9000500000, 1, "7.542e26"),
        ("row", 257157142857143, 2, "6.194e21"),
    ):
        n = interval(n_max)
        kappa = 1 - t**2 * lf - chi * t * nu
        derivative_x = 4 * nu * vc * ell if chi == 1 else 16 * nu * vc * b0
        beta = t**2 * (lf + derivative_x)
        kb = 1 - chi * t * nu
        assert lower(kappa) > mp.mpf(".82")
        assert upper(beta) < mp.mpf(".160" if chi == 1 else ".409")
        show(mode + " beta", beta)
        log_tail = iv.ln(18 * n) - noise_radius**2 / 6
        assert upper(log_tail) < -13600 - mp.log(2)
        raw_radius = (target_h + t * lf * bx0) / kappa
        if chi == 1:
            raw_radius *= iv.sqrt(n)
        raw_position = bx0 + t * raw_radius
        base = 1 + t**2 * lf + 2 * t * nu
        derivative = base + (4 if chi == 1 else 8) * t**2 * nu * ell * raw_radius * (
            1 if chi == 1 else iv.exp(2 * raw_position**2)
        )
        gaussian_cost = (raw_radius + c * b10) ** 2 / (2 * q2) + (
            iv.sqrt(interval(3)) * target_r + bx0 + t * raw_radius
        ) ** 2 / (2 * s**2)
        # q cancels between the lower density and two-update upper density;
        # tau=min(s,sigma_J)=s, so every remaining displayed term is positive.
        log_ratio_upper = (
            -n * iv.ln(ball_lower)
            + n * gaussian_cost
            + 3 * n * (iv.ln(n) / 2 + iv.ln(derivative) - iv.ln((1 - beta) * kb))
            + n * iv.ln(4 * n**2)
            + 13600
            + iv.ln(interval(2))
        )
        show(mode + " log_ratio_upper", log_ratio_upper)
        assert upper(log_ratio_upper) < mp.mpf(log_budget)
    print("All finite-population discovery-clock margins certified.")


if __name__ == "__main__":
    main()
