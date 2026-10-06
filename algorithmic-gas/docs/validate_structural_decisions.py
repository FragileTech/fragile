#!/usr/bin/env python3
"""Check structural confinement and residual formulas with exact arithmetic.

These checks supplement the analytic proofs; they do not certify a landscape's
population-class hypotheses or a numerical quadrature on an infinite domain.
"""
from fractions import Fraction as Q

import sympy as sp


def check_trap_algebra():
    beta, eta, u, g, threshold = sp.symbols('beta eta u g threshold', real=True)
    radial = beta**2 + 2 * eta * beta * u + eta**2 * g**2
    assert sp.expand(radial - threshold - ((beta + eta*u)**2
                     - (threshold + eta**2 * (u**2-g**2)))) == 0
    for chi in (Q(-1), Q(0), Q(1, 4), Q(3, 4)):
        for eta0 in (Q(1, 8), Q(1), Q(2)):
            for growth in (Q(0), Q(1, 3), Q(2)):
                for lam in (Q(0), Q(1, 2), Q(1), Q(4)):
                    a = abs(1-eta0*lam) + eta0*growth
                    # Compare without floating point or square-root rounding.
                    left = (1-chi)*a*a < 1
                    right = a*a < 1/(1-chi)
                    assert left == right
                    r0 = (1-chi)*a*a
                    if 0 < r0 < 1:
                        t = (1-r0)/(2*r0)
                        assert (1+t)*r0 == (1+r0)/2 < 1


def check_residual_convolution():
    for q in (Q(1, 4), Q(1, 2), Q(9, 10)):
        for horizon in range(1, 15):
            r = Q(2, 7)
            defects = [Q(j+1, 101) for j in range(horizon)]
            residuals = [r]
            for e in defects:
                residuals.append(q*residuals[-1]+e)
            closed = (1-q**horizon)/(1-q)*r + sum(
                ((1-q**(horizon-1-j))/(1-q)*defects[j]
                 for j in range(horizon-1)), Q(0))
            assert sum(residuals[:horizon]) == closed
            weights = sum(((1-q**(horizon-1-j))/(1-q)
                           for j in range(horizon-1)), Q(0))
            assert weights == (horizon-(1-q**horizon)/(1-q))/(1-q)


def check_variance_and_flux():
    # Finite populations include diagonal exclusion and arbitrary gate patterns.
    for positions in ((0, 1, 3), (-1, 0, 2, 4), (0, 0, 5)):
        n = len(positions)
        radii = [Q(x*x) for x in positions]
        mean = sum(Q(x) for x in positions)/n
        second = sum(radii)/n
        lo, hi = mean-Q(1, 3), mean+Q(2, 5)
        vlo, vhi = second-Q(1, 7), second+Q(1, 4)
        distance = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
        variance = second-mean*mean
        assert max(0, vlo-max(lo*lo, hi*hi)) <= variance
        assert variance <= max(0, vhi-distance*distance)
        core = [i for i,x in enumerate(positions) if abs(x) <= 1]
        exterior = [i for i in range(n) if i not in core]
        for seed in range(7):
            weight = [[Q(1+(i+j+seed)%4, 4) for j in range(n)] for i in range(n)]
            gate = [[Q((2*i+j+seed)%5, 4) for j in range(n)] for i in range(n)]
            den = [sum(weight[i][j] for j in range(n) if i != j) for i in range(n)]
            p = [[weight[i][j]*gate[i][j]/den[i] if i != j else Q(0)
                  for j in range(n)] for i in range(n)]
            chi = min((sum(p[i][j] for j in core) for i in exterior), default=Q(0))
            donor = [sum(weight[i][j]*gate[i][j]/den[i] for i in range(n))
                     for j in range(n)]  # diagonal overcount is intentional
            delta = max((donor[j] for j in exterior), default=Q(0))
            bmix = sum((donor[j]/n for j in core), Q(0))
            flux = sum((p[i][j]*(radii[j]-radii[i])/n
                        for i in range(n) for j in range(n)), Q(0))
            assert flux <= -(chi-delta)*sum(radii)/n+chi+bmix


def check_chernoff():
    lam, t, n, variance = sp.symbols('lam t n variance', positive=True)
    objective = -lam*t+n*lam**2*variance/2
    assert sp.simplify(objective.subs(lam,t/(n*variance))+t**2/(2*n*variance)) == 0


if __name__ == '__main__':
    check_trap_algebra()
    check_residual_convolution()
    check_variance_and_flux()
    check_chernoff()
    print('Structural decision checks passed: trap algebra, residual convolutions, '
          'variance bands, finite-population flux and escape exponent.')
