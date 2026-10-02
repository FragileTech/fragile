#!/usr/bin/env python3
"""Exact arithmetic sanity checks for the stopped transition estimates.

These finite checks supplement, and do not replace, the symbolic proofs.
Run with Python 3; no third-party dependencies are required.
"""
from fractions import Fraction as F
from math import factorial


def geometric_sum(n, probability):
    return sum(((1 - probability) ** j for j in range(n)), F(0))


def validate_rastrigin_constants():
    assert F(1) + F(1, 16) + F(7, 400) == F(27, 25)
    assert F(27, 25) ** 2 / (2 * F(1, 10**6)) == 583200
    assert 583200 * 128 == 74649600
    assert F(99, 100) - F(1, 8) == F(173, 200)
    # Establishing e^122 above this exact Taylor polynomial proves
    # 4*10^6*128*e^-122 < 6*10^-45 without floating-point underflow.
    taylor_lower = sum((F(122) ** k / factorial(k) for k in range(201)), F(0))
    assert taylor_lower > F(4 * 10**6 * 128) * 10**45 / 6
    assert taylor_lower > 4  # 1 - 2e^-122 > 1/2
    tau2_upper = F(401, 400 * 10**6)
    pi_upper = F(22, 7)
    assert 2 * pi_upper * tau2_upper < F(1, 300) ** 2
    # Consequently Gaussian-volume prefactor exceeds (1/2)*(1/8)*300 > 1.
    assert F(1, 2) * F(1, 8) * 300 > 1


def validate_competing_hazards():
    cases = 0
    for pi in range(1, 11):
        for fi in range(11 - pi):
            p = F(pi, 10)
            f = F(fi, 10)
            total_exit = p + f
            for n in range(1, 13):
                exact_target = p * geometric_sum(n, total_exit)
                exact_failure = f * geometric_sum(n, total_exit)
                exact_survival = (1 - total_exit) ** n
                assert exact_target + exact_failure + exact_survival == 1
                assert exact_survival <= (1 - p) ** n
                assert exact_failure <= f * geometric_sum(n, p)
                alternative = 1 - (1 - p) ** n - f * geometric_sum(n, p)
                assert exact_target >= alternative
                assert geometric_sum(n, total_exit) >= n * (1 - (n - 1) * total_exit)
                for ell in (0, 1, 7):
                    assert (1 - total_exit) ** ell >= 1 - ell * total_exit
                cases += 1
    return cases


def validate_coarse_intervals():
    # A nontrivial row interval with a feasible deterministic center. The
    # maxima over interval endpoints need not define a feasible row; their
    # sum is nevertheless a valid (possibly conservative) upper bound.
    lower = (F(1, 10), F(1, 5), F(1, 10))
    upper = (F(3, 5), F(3, 5), F(1, 2))
    center = (F(2, 5), F(2, 5), F(1, 5))
    bound = min(F(1), sum((max(t-l, u-t) for l, u, t in zip(lower, upper, center)), F(0))/2)
    cases = 0
    for i in range(11):
        for j in range(11-i):
            row = (F(i, 10), F(j, 10), F(10-i-j, 10))
            if all(l <= x <= u for l, x, u in zip(lower, row, upper)):
                actual_tv = sum((abs(x-t) for x, t in zip(row, center)), F(0))/2
                assert actual_tv <= bound
                cases += 1
    return cases


def main():
    validate_rastrigin_constants()
    hazard_cases = validate_competing_hazards()
    interval_cases = validate_coarse_intervals()
    print("Passed exact Rastrigin constants and Taylor residence bound;")
    print(f"{hazard_cases} rational competing-hazard cases;")
    print(f"{interval_cases} coarse-transition interval rows.")


if __name__ == "__main__":
    main()
