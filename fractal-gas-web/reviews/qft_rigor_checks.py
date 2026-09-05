"""Reproduce algebraic checks in qft_rigor_review.md using only the standard library.

Run: uv run --no-project python reviews/qft_rigor_checks.py
These check displayed formulas, not execution of the Torch analysis pipeline.
"""

from cmath import exp as cexp
from fractions import Fraction as Q
import json
from math import cos, exp, log, pi, sqrt


def mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))


def adj(a):
    return tuple(tuple(a[j][i].conjugate() for j in range(2)) for i in range(2))


def trace(a):
    return a[0][0] + a[1][1]


identity = ((1, 0), (0, 1))
a = b = ((0, 1j), (1j, 0))  # i sigma_x
c = ((1j, 0), (0, -1j))  # i sigma_z
for u in (a, b, c):
    assert mul(u, adj(u)) == identity
    assert u[0][0] * u[1][1] - u[0][1] * u[1][0] == 1

# Triangles (a,b,c,a), (c,d,a,c), with U_ab=A, U_bc=I,
# U_ca=C, U_cd=B, U_da=I. Here letters naming matrices and vertices differ.
t1, t2 = mul(a, c), mul(b, adj(c))
naive = mul(t1, t2)
outer = mul(a, b)
rebased = mul(mul(mul(t1, adj(c)), t2), c)
assert rebased == outer and naive != outer

# A positive covariance from independent ordinary and rotating OU processes.
# It obeys C(t) <= 1.5 exp(-t), but its finite-difference rate can be below 1.
def covariance(t):
    return exp(-t) * (1 + 0.5 * cos(t))


t, h = 3 * pi / 2, 0.1
effective_rate = -log(covariance(t + h) / covariance(t)) / h
assert 0 < effective_rate < 1

# Re-recording a two-mode correlator changes finite-time effective-rate ratios.
def mixture(t):
    return exp(-t) + exp(-3 * t)


ratios = [-log(mixture(h) / mixture(0)) / h / 2 for h in (1, 2)]
assert abs(ratios[0] - ratios[1]) > 0.1

# Scalar phase formula in electroweak_spinors.py: h_eff=1, |delta F|=epsilon=1.
u = cexp(1j * pi / 4)
assert abs(u - u.conjugate()) > 1
assert abs(u * u - 1) > 1  # det(u I_2) != 1

# One chosen generation, in left-handed notation: Q, u^c, d^c, L, e^c, nu^c.
charges = [(6, Q(1, 6)), (3, Q(-2, 3)), (3, Q(1, 3)),
           (2, Q(-1, 2)), (1, Q(1)), (1, Q(0))]
anomalies = {
    "U1_cubed": sum(n * y**3 for n, y in charges),
    "gravity_squared_U1": sum(n * y for n, y in charges),
    "SU3_squared_U1": Q(1, 2) * (2 * Q(1, 6) - Q(2, 3) + Q(1, 3)),
    "SU2_squared_U1": Q(1, 2) * (3 * Q(1, 6) - Q(1, 2)),
    "SU3_cubed": 2 - 1 - 1,
    "SU2_doublet_parity": (3 + 1) % 2,
}
assert all(v == 0 for v in anomalies.values())

# Score identities at positive denominators, and pure-gauge telescoping.
vi, vj, eps = Q(1), Q(2), Q(1)
sij, sji = (vj - vi) / (vi + eps), (vi - vj) / (vj + eps)
assert (vi + eps) * sij == -(vj + eps) * sji
assert sij + sji == (vj - vi)**2 / ((vi + eps) * (vj + eps))
theta = [0.2, -0.7, 1.1, 0.2]
holonomy = 1
for left, right in zip(theta, theta[1:]):
    holonomy *= cexp(1j * (left - right))
assert abs(holonomy - 1) < 1e-14

print(json.dumps({
    "nonabelian_naive_normalized_trace": (trace(naive) / 2).real,
    "nonabelian_correct_normalized_trace": (trace(outer) / 2).real,
    "envelope_rate": 1,
    "finite_difference_rate": effective_rate,
    "two_mode_to_rate2_ratios_at_record_intervals_1_and_2": ratios,
    "scalar_proxy_reversal_error": abs(u - u.conjugate()),
    "scalar_proxy_determinant_error": abs(u * u - 1),
    "epsilon_d_using_gY_over_epsilon_d_using_g1_fixed_N1": sqrt(5 / 3),
    "anomaly_coefficients": {k: str(v) for k, v in anomalies.items()},
    "score_and_pure_gauge_checks": "passed",
}, indent=2))
